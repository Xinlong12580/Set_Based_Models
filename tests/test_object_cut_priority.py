"""Ordered object cuts, using only small synthetic arrays."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import EventCut, ExistentialRuleList, ObservableSpec, SetInput
from set_based_models.rule_list.generation import BeamRuleGenerator, _Candidate


SPECS = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("eta", "continuous"),
    ObservableSpec("phi", "continuous"),
    ObservableSpec("type_id", "categorical", {0: "jet", 1: "photon"}),
)
SIGNAL = np.array([
    [[50, .2, .1, 1], [10, 1.2, -.3, 0]],
    [[40, .4, .5, 1], [20, 1.4, -.5, 0]],
    [[30, .6, .9, 1], [30, 1.6, -.7, 0]],
    [[20, .8, 1.3, 1], [40, 1.8, -.9, 0]],
], float)
BACKGROUND = SIGNAL.copy()
BACKGROUND[:, 0, 0] = [5, 10, 15, 20]
BACKGROUND[:, 0, 1] += 1


def score(stats):
    return stats.true_positive_rate - stats.false_positive_rate


def make_model(specs=SPECS, **kwargs):
    options = dict(max_rule_depth=3, beam_width=500, max_candidate_rules=None,
                   max_thresholds=None, min_signal_efficiency=.01,
                   beam_similarity_filter=False, scoring_function=score)
    options.update(kwargs)
    return ExistentialRuleList(specs, **options)


def make_generator(**kwargs):
    options = dict(max_rule_depth=3, beam_width=500, max_candidate_rules=None,
                   max_thresholds=None, minimum_count=1, min_signal_efficiency=.01,
                   scoring_function=score, beam_similarity_filter=False)
    options.update(kwargs)
    return BeamRuleGenerator(**options)


def object_names(rule):
    return [rule.observables[cut.observable_index].name
            for cut in rule.cuts if not isinstance(cut, EventCut)]


class ObjectCutPriorityTests(unittest.TestCase):
    def test_directed_mass_priority_forms_windows_and_preserves_prefixes(self):
        specs = (ObservableSpec("pair_mass", "continuous"),)
        signal = np.array([80, 120, 125, 130, 170], float).reshape(-1, 1, 1)
        background = np.array([10, 90, 110, 150, 200], float).reshape(-1, 1, 1)
        model = make_model(specs, max_rule_depth=3,
                           object_cut_priority=[("pair_mass", ">"), ("pair_mass", "<=")]).fit(
                               signal, background)
        self.assertEqual(model.object_cut_priority, ("pair_mass:>", "pair_mass:<="))
        self.assertEqual({r.depth for r in model.candidate_rules_}, {1, 2, 3})
        for rule in model.candidate_rules_:
            self.assertEqual([c.comparison.value for c in rule.cuts[:2]], [">", "<="][:rule.depth])
            np.testing.assert_array_equal(rule.predict(signal), rule.signal_selection)
            np.testing.assert_array_equal(rule.predict(background), rule.background_selection)
        self.assertTrue(any(r.depth == 2 and r.statistics.signal_accepted == 3
                            and r.statistics.background_accepted == 0 for r in model.candidate_rules_))
        self.assertEqual({r.cuts[2].comparison.value for r in model.candidate_rules_ if r.depth == 3},
                         {">", "<="})

    def test_all_four_numeric_directions_are_supported(self):
        for direction in (">", ">=", "<", "<="):
            with self.subTest(direction=direction):
                model = make_model(max_rule_depth=1, object_cut_priority=[f"pt:{direction}"]).fit(
                    SIGNAL, BACKGROUND)
                self.assertTrue(model.candidate_rules_)
                for rule in model.candidate_rules_:
                    self.assertEqual(object_names(rule), ["pt"])
                    self.assertEqual(rule.cuts[0].comparison.value, direction)
                    np.testing.assert_array_equal(rule.predict(SIGNAL), rule.signal_selection)

    def test_event_cuts_do_not_consume_directed_object_entries(self):
        events = (ObservableSpec("pt", "continuous"),)
        event_signal, event_background = np.arange(4).reshape(-1, 1), np.zeros((4, 1))
        model = make_model(event_observables=events,
                           object_cut_priority=["type_id", "pt:>", "pt:<="]).fit(
                               SIGNAL, BACKGROUND, signal_event_features=event_signal,
                               background_event_features=event_background)
        self.assertTrue(any(not object_names(r) for r in model.candidate_rules_))
        for rule in model.candidate_rules_:
            cuts = [c for c in rule.cuts if not isinstance(c, EventCut)]
            self.assertEqual(object_names(rule), ["type_id", "pt", "pt"][:len(cuts)])
            self.assertEqual([c.comparison.value for c in cuts[1:]], [">", "<="][:max(0, len(cuts) - 1)])

    def test_guarded_directed_cuts_keep_validity_guards(self):
        specs = (ObservableSpec("x", "continuous", validity_column="x_valid"),
                 ObservableSpec("x_valid", "boolean"))
        signal = np.array([[[np.nan, 0]], [[10, 1]], [[20, 1]]])
        background = np.array([[[np.nan, 0]], [[1, 1]], [[2, 1]]])
        model = make_model(specs, object_cut_priority=["x:>", "x:<="]).fit(signal, background)
        self.assertTrue(any(r.depth == 2 for r in model.candidate_rules_))
        for rule in model.candidate_rules_:
            self.assertFalse(rule.signal_selection[0])
            self.assertEqual([c.comparison.value for c in rule.cuts[:2]], [">", "<="][:rule.depth])
            self.assertIn("x_valid == True", rule.to_text())

    def test_priority_direction_normalization_and_unusable_directions(self):
        self.assertEqual(make_model(object_cut_priority=[" pt : >= ", ("eta", " < ")]).object_cut_priority,
                         ("pt:>=", "eta:<"))
        self.assertEqual(make_model(object_cut_priority=[("pt", None)]).object_cut_priority, ("pt",))
        for entry in ("pt:sideways", "pt:", ("pt", "==")):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                make_model(object_cut_priority=[entry])
        with self.assertRaisesRegex(ValueError, "numerical"):
            make_model(object_cut_priority=["type_id:>"])

    def test_repeated_features_follow_every_position_then_release_search(self):
        for priority in (("type_id", "pt", "pt"), ("pt", "eta", "pt")):
            with self.subTest(priority=priority):
                model = make_model(object_cut_priority=priority, max_rule_depth=4).fit(
                    SIGNAL, BACKGROUND)
                self.assertEqual(model.object_cut_priority, priority)
                self.assertEqual({r.depth for r in model.candidate_rules_}, {1, 2, 3, 4})
                for rule in model.candidate_rules_:
                    self.assertEqual(object_names(rule)[:3], list(priority[:rule.depth]))
                    np.testing.assert_array_equal(rule.predict(SIGNAL), rule.signal_selection)
                    np.testing.assert_array_equal(rule.predict(BACKGROUND), rule.background_selection)
                fourth_features = {object_names(r)[3] for r in model.candidate_rules_ if r.depth == 4}
                self.assertTrue({"eta", "phi", "pt"}.issubset(fourth_features))

    def test_repeated_mass_feature_can_form_a_window(self):
        specs = (ObservableSpec("pair_mass", "continuous"),)
        signal = np.array([80, 120, 125, 130, 170], float).reshape(-1, 1, 1)
        background = np.array([10, 90, 110, 150, 200], float).reshape(-1, 1, 1)
        model = make_model(specs, object_cut_priority=["pair_mass", "pair_mass"],
                           max_rule_depth=2).fit(signal, background)
        self.assertTrue(any(r.depth == 1 for r in model.candidate_rules_))
        windows = [r for r in model.candidate_rules_ if r.depth == 2
                   and {c.comparison.value for c in r.cuts} == {">", "<="}]
        self.assertTrue(windows)
        self.assertTrue(any(r.statistics.signal_accepted == 3
                            and r.statistics.background_accepted == 0 for r in windows))

    def test_repeated_category_advances_even_when_masks_are_unchanged(self):
        events = (ObservableSpec("always", "boolean"),)
        values = np.ones((4, 1))
        for similarity in (False, True):
            with self.subTest(similarity=similarity):
                model = make_model(event_observables=events,
                    object_cut_priority=["type_id", "type_id", "pt"],
                    beam_similarity_filter=similarity).fit(
                        SIGNAL[:, :1], BACKGROUND[:, :1],
                        signal_event_features=values, background_event_features=values)
                for rule in model.candidate_rules_:
                    names = object_names(rule)
                    self.assertEqual(names, ["type_id", "type_id", "pt"][:len(names)])
                self.assertTrue(any(object_names(r) == ["type_id", "type_id", "pt"]
                                    for r in model.candidate_rules_))
                second_layer = [r for r in model.candidate_rules_ if r.depth == 2]
                self.assertEqual({tuple(object_names(r)) for r in second_layer},
                                 {(), ("type_id",), ("type_id", "type_id")})

    def test_compulsory_order_and_unrestricted_refinement_afterwards(self):
        model = make_model(object_cut_priority=["type_id", "pt"]).fit(SIGNAL, BACKGROUND)
        self.assertEqual(model.object_cut_priority, ("type_id", "pt"))
        self.assertEqual({r.depth for r in model.candidate_rules_}, {1, 2, 3})
        for rule in model.candidate_rules_:
            self.assertEqual(object_names(rule)[:2], ["type_id", "pt"][:rule.depth])
            np.testing.assert_array_equal(rule.predict(SIGNAL), rule.signal_selection)
            np.testing.assert_array_equal(rule.predict(BACKGROUND), rule.background_selection)
        third_features = {object_names(r)[2] for r in model.candidate_rules_ if r.depth == 3}
        self.assertTrue({"eta", "phi", "pt"}.issubset(third_features))

    def test_candidate_generator_releases_previously_used_features(self):
        signal = SetInput.from_arrays(SIGNAL, SPECS)
        background = SetInput.from_arrays(BACKGROUND, SPECS)
        generator = make_generator(object_cut_priority=["type_id", "pt"])
        root = _Candidate((), signal.slot_mask, background.slot_mask,
                          np.ones(4, bool), np.ones(4, bool))
        for progress, expected in ((0, {"type_id"}), (1, {"pt"}),
                                   (2, {spec.name for spec in SPECS})):
            root.object_priority_progress = progress
            cuts = generator._candidate_cuts(root, signal, background)
            self.assertEqual({SPECS[c.observable_index].name for c in cuts}, expected)

    def test_candidate_generator_consumes_repeated_positions_individually(self):
        signal = SetInput.from_arrays(SIGNAL, SPECS)
        background = SetInput.from_arrays(BACKGROUND, SPECS)
        generator = make_generator(object_cut_priority=["pt", "eta", "pt"])
        parent = _Candidate((), signal.slot_mask, background.slot_mask,
                            np.ones(4, bool), np.ones(4, bool))
        for progress, expected in ((0, {"pt"}), (1, {"eta"}), (2, {"pt"}),
                                   (3, {spec.name for spec in SPECS})):
            parent.object_priority_progress = progress
            cuts = generator._candidate_cuts(parent, signal, background)
            self.assertEqual({SPECS[c.observable_index].name for c in cuts}, expected)

    def test_incomplete_rules_remain_candidates_when_depth_is_small(self):
        for depth in (1, 2):
            with self.subTest(depth=depth):
                model = make_model(max_rule_depth=depth,
                    object_cut_priority=["type_id", "pt", "eta"]).fit(SIGNAL, BACKGROUND)
                self.assertTrue(model.candidate_rules_)
                self.assertEqual(max(r.depth for r in model.candidate_rules_), depth)
                self.assertTrue(all(len(object_names(r)) < 3 for r in model.candidate_rules_))
                selected = model.prune_candidates(similarity_filter=False)
                self.assertEqual(len(selected), len(model.candidate_rules_))

    def test_incomplete_rule_can_rank_first_and_survive_candidate_cap(self):
        model = make_model(object_cut_priority=["type_id", "pt"], max_candidate_rules=1,
                           scoring_function=lambda stats: -stats.rule_depth).fit(SIGNAL, BACKGROUND)
        self.assertEqual(len(model.candidate_rules_), 1)
        self.assertEqual(object_names(model.get_rule(0)), ["type_id"])
        self.assertEqual(model.get_rule(0).generation_score, -1)

    def test_count_and_weighted_statistics_are_unchanged_by_priority(self):
        signal, background = SIGNAL.copy(), BACKGROUND.copy()
        signal[..., 3] = background[..., 3] = 1
        weights = np.array([1, 2, 3, 4])
        model = make_model(object_cut_priority=["type_id", "pt"], minimum_count=2).fit(
            signal, background, signal_weight=weights, background_weight=weights[::-1])
        self.assertTrue(model.candidate_rules_)
        for rule in model.candidate_rules_:
            expected = np.count_nonzero(rule.slot_mask(signal), axis=1) >= 2
            np.testing.assert_array_equal(rule.signal_selection, expected)
            self.assertEqual(rule.statistics.signal_weight_accepted, weights[expected].sum())
            self.assertEqual(rule.statistics.minimum_count, 2)

    def test_constant_required_feature_stops_object_extension_not_prefix_return(self):
        signal, background = SIGNAL.copy(), BACKGROUND.copy()
        signal[..., 0] = background[..., 0] = 10
        model = make_model(object_cut_priority=["type_id", "pt"]).fit(signal, background)
        self.assertTrue(model.candidate_rules_)
        self.assertEqual({tuple(object_names(r)) for r in model.candidate_rules_}, {("type_id",)})

    def test_only_the_branch_with_available_required_thresholds_continues(self):
        signal, background = SIGNAL.copy(), BACKGROUND.copy()
        signal[:, 0, 0] = background[:, 0, 0] = 10  # Photons have constant PT.
        model = make_model(object_cut_priority=["type_id", "pt"]).fit(signal, background)
        for rule in model.candidate_rules_:
            if rule.depth > 1:
                self.assertEqual(rule.cuts[0].accepted_codes, frozenset({0}))
        self.assertTrue(any(r.depth == 1 and r.cuts[0].accepted_codes == {1}
                            for r in model.candidate_rules_))

    def test_disabled_priority_is_exactly_backward_compatible(self):
        original = make_model().fit(SIGNAL, BACKGROUND).candidate_rules_
        for disabled in ([], (), None):
            changed = make_model(object_cut_priority=disabled).fit(SIGNAL, BACKGROUND).candidate_rules_
            with self.subTest(disabled=disabled):
                self.assertEqual([(r.cuts, r.generation_score) for r in changed],
                                 [(r.cuts, r.generation_score) for r in original])
                for left, right in zip(changed, original):
                    np.testing.assert_array_equal(left.signal_selection, right.signal_selection)
                    np.testing.assert_array_equal(left.background_selection, right.background_selection)

    def test_progress_preserves_identical_mask_prefixes_at_both_beam_filters(self):
        # Both type_id=photon and event:type_id=True leave every mask unchanged.
        # The former requires PT next; the latter still requires object type_id.
        events = (ObservableSpec("type_id", "boolean"),)
        signal, background = SIGNAL[:, :1].copy(), BACKGROUND[:, :1].copy()
        values = np.ones((4, 1))
        for similarity in (False, True):
            with self.subTest(similarity=similarity):
                model = make_model(event_observables=events, max_rule_depth=2,
                    object_cut_priority=["type_id", "pt"],
                    beam_similarity_filter=similarity).fit(
                        signal, background, signal_event_features=values,
                        background_event_features=values)
                first = [r for r in model.candidate_rules_ if r.depth == 1]
                self.assertEqual(len(first), 2)
                self.assertEqual({isinstance(r.cuts[0], EventCut) for r in first}, {True, False})
                np.testing.assert_array_equal(first[0].signal_selection, first[1].signal_selection)
                self.assertTrue(any(r.depth == 2 and object_names(r) == ["type_id", "pt"]
                                    for r in model.candidate_rules_))
                self.assertTrue(any(r.depth == 2 and not object_names(r)
                                    for r in model.candidate_rules_))

    def test_event_cuts_do_not_advance_object_priority_or_get_restricted(self):
        events = (ObservableSpec("pt", "continuous"),)
        signal_events, background_events = np.array([[0], [1], [2], [3]]), np.zeros((4, 1))
        model = make_model(event_observables=events, object_cut_priority=["type_id", "pt"]).fit(
            SIGNAL, BACKGROUND, signal_event_features=signal_events,
            background_event_features=background_events)
        self.assertTrue(any(not object_names(r) for r in model.candidate_rules_))
        self.assertTrue(any(any(isinstance(c, EventCut) for c in r.cuts) and object_names(r)
                            for r in model.candidate_rules_))
        for rule in model.candidate_rules_:
            names = object_names(rule)
            self.assertEqual(names[:2], ["type_id", "pt"][:len(names)])
            np.testing.assert_array_equal(rule.predict(SIGNAL, event_features=signal_events),
                                          rule.signal_selection)

    def test_guarded_priority_feature_keeps_automatic_guarding(self):
        specs = (ObservableSpec("x", "continuous", validity_column="x_valid"),
                 ObservableSpec("x_valid", "boolean"), ObservableSpec("other", "continuous"))
        signal = np.array([[[np.nan, 0, 2]], [[10, 1, 3]], [[20, 1, 4]]])
        background = np.array([[[np.nan, 0, 1]], [[1, 1, 2]], [[2, 1, 3]]])
        model = make_model(specs, object_cut_priority=["x", "x"]).fit(signal, background)
        self.assertTrue(model.candidate_rules_)
        for rule in model.candidate_rules_:
            self.assertEqual(object_names(rule)[:2], ["x", "x"][:rule.depth])
            self.assertFalse(rule.signal_selection[0])
            self.assertNotIn("x_valid", object_names(rule))
            self.assertIn("x_valid == True", rule.to_text())

    def test_guard_only_flag_cannot_be_prioritized_without_explicit_opt_in(self):
        specs = (ObservableSpec("x", "continuous", validity_column="x_valid"),
                 ObservableSpec("x_valid", "boolean"))
        with self.assertRaisesRegex(ValueError, "guard-only"):
            make_model(specs, object_cut_priority=["x_valid", "x"])
        signal = np.array([[[0, 0]], [[10, 1]], [[20, 1]]])
        background = np.array([[[0, 0]], [[1, 1]], [[2, 1]]])
        model = make_model(specs, object_cut_priority=["x_valid", "x"],
                           allow_validity_cuts=True).fit(signal, background)
        self.assertTrue(any(object_names(r) == ["x_valid", "x"] for r in model.candidate_rules_))

    def test_order_validated_against_object_schema_only(self):
        with self.assertRaisesRegex(ValueError, "Unknown object_cut_priority"):
            make_model(event_observables=(ObservableSpec("ht", "continuous"),),
                       object_cut_priority=["ht"])
        with self.assertRaises(ValueError):
            make_model((), event_observables=(ObservableSpec("pt", "continuous"),),
                       object_cut_priority=["pt"])
        model = make_model(event_observables=(ObservableSpec("ht", "continuous"),),
                           object_cut_priority=["pt"])
        with self.assertRaisesRegex(ValueError, "requires object tensor"):
            model.fit(signal_event_features=np.ones((4, 1)),
                      background_event_features=np.ones((4, 1)))

    def test_invalid_priority_configuration_is_rejected(self):
        for value in ("type_id", b"pt", {"pt", "type_id"}, {"pt": 1}, 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                make_model(object_cut_priority=value)
        for value in ([""], [" "], [None], [1], ["missing"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_model(object_cut_priority=value)
        self.assertEqual(make_model(object_cut_priority=[" type_id ", "pt"]).object_cut_priority,
                         ("type_id", "pt"))
        self.assertEqual(make_model(object_cut_priority=["pt", " pt "]).object_cut_priority,
                         ("pt", "pt"))

    def test_direct_generator_validates_names_against_schema(self):
        signal, background = (SetInput.from_arrays(x, SPECS) for x in (SIGNAL, BACKGROUND))
        with self.assertRaises(ValueError):
            make_generator(object_cut_priority=["missing"]).generate(signal, background)
        with self.assertRaises(TypeError):
            make_generator(object_cut_priority="pt")


if __name__ == "__main__":
    unittest.main()
