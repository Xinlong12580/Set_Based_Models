"""Native ERL input-mode regressions using synthetic arrays only."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from set_based_models import (
    EventCut, ExistentialRule, ExistentialRuleList, ObservableSpec,
    RuleStatistics, RuleStep, RuleUnion, SetInput, ThresholdCut,
)

OBJECTS = (ObservableSpec("pt", "continuous"),
           ObservableSpec("type", "categorical", {0: "jet", 1: "muon"}))
EVENTS = (ObservableSpec("ht", "continuous"), ObservableSpec("photons", "ordinal"),
          ObservableSpec("flag", "boolean"),
          ObservableSpec("region", "categorical", {0: "a", 1: "b"}))
SIGNAL_EVENTS = np.array([[4, 2, 1, 1], [5, 2, 1, 1], [6, 3, 0, 0], [0, 0, 0, 0]], float)
BACKGROUND_EVENTS = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]], float)
SIGNAL_OBJECTS = np.array([[[40, 1], [10, 0]], [[50, 1], [20, 0]],
                           [[60, 1], [15, 0]], [[0, 0], [0, 0]]], float)
BACKGROUND_OBJECTS = np.array([[[5, 0], [0, 0]], [[10, 0], [0, 0]],
                               [[20, 0], [0, 0]], [[0, 0], [0, 0]]], float)


def score(stats):
    return stats.true_positive_rate - stats.false_positive_rate


def model(objects=(), events=EVENTS, **kwargs):
    return ExistentialRuleList(objects, event_observables=events,
                               max_rule_depth=2, beam_width=40,
                               beam_similarity_filter=False, min_signal_efficiency=.01,
                               scoring_function=score, **kwargs)


class ERLInputModeTests(unittest.TestCase):
    def native(self, **kwargs):
        return model(**kwargs).fit(signal_event_features=SIGNAL_EVENTS,
                                  background_event_features=BACKGROUND_EVENTS)

    def assert_selections_match_predictions(self, fitted, signal=None, background=None,
                                           signal_events=None, background_events=None):
        self.assertTrue(fitted.candidate_rules_)
        for rule in fitted.candidate_rules_:
            np.testing.assert_array_equal(rule.predict(signal, event_features=signal_events,
                slot_mask=None if signal is None else signal[..., 0] > 0), rule.signal_selection)
            np.testing.assert_array_equal(rule.predict(background, event_features=background_events,
                slot_mask=None if background is None else background[..., 0] > 0), rule.background_selection)

    def test_event_only_fit_prediction_and_evaluation_without_objects(self):
        fitted = self.native()
        self.assertEqual(fitted.input_mode_, "event_only")
        self.assertEqual(fitted.observables, ())
        self.assertTrue(all(isinstance(cut, EventCut) for rule in fitted.candidate_rules_ for cut in rule.cuts))
        self.assert_selections_match_predictions(fitted, signal_events=SIGNAL_EVENTS,
                                                background_events=BACKGROUND_EVENTS)
        rule = fitted.candidate_rules_[0]
        performance = fitted.evaluate_rule(rule.rule_id, signal_event_features=SIGNAL_EVENTS,
                                           background_event_features=BACKGROUND_EVENTS)
        self.assertEqual(performance.signal_accepted, rule.statistics.signal_accepted)
        np.testing.assert_array_equal(rule(event_features=SIGNAL_EVENTS), rule.signal_selection)
        self.assertEqual(rule.slot_mask(event_features=SIGNAL_EVENTS).shape, (4, 0))

    def test_object_only_existing_calls_remain_valid(self):
        fitted = model(OBJECTS, ()).fit(SIGNAL_OBJECTS, BACKGROUND_OBJECTS,
                                       signal_slot_mask=SIGNAL_OBJECTS[..., 0] > 0,
                                       background_slot_mask=BACKGROUND_OBJECTS[..., 0] > 0)
        self.assertEqual(fitted.input_mode_, "object_only")
        self.assertTrue(all(not isinstance(cut, EventCut) for rule in fitted.candidate_rules_ for cut in rule.cuts))
        self.assert_selections_match_predictions(fitted, SIGNAL_OBJECTS, BACKGROUND_OBJECTS)

    def test_combined_mode_uses_both_candidate_types(self):
        fitted = model(OBJECTS).fit(SIGNAL_OBJECTS, BACKGROUND_OBJECTS,
                                    signal_slot_mask=SIGNAL_OBJECTS[..., 0] > 0,
                                    background_slot_mask=BACKGROUND_OBJECTS[..., 0] > 0,
                                    signal_event_features=SIGNAL_EVENTS,
                                    background_event_features=BACKGROUND_EVENTS)
        self.assertEqual(fitted.input_mode_, "combined")
        # Identical event selections can be deduplicated; use a second independent
        # event variable so both scopes can represent distinct selections.
        scopes = {isinstance(cut, EventCut) for rule in fitted.candidate_rules_ for cut in rule.cuts}
        self.assertEqual(scopes, {False, True})
        self.assert_selections_match_predictions(fitted, SIGNAL_OBJECTS, BACKGROUND_OBJECTS,
                                                SIGNAL_EVENTS, BACKGROUND_EVENTS)

    def test_event_only_skips_categorical_object_cuts_with_unused_schema(self):
        fitted = self.native(objects=OBJECTS)
        self.assertEqual(fitted.input_mode_, "event_only")
        self.assertTrue(all(isinstance(cut, EventCut) for rule in fitted.candidate_rules_ for cut in rule.cuts))
        self.assert_selections_match_predictions(fitted, signal_events=SIGNAL_EVENTS,
                                                background_events=BACKGROUND_EVENTS)

    def test_combined_rules_preserve_same_slot_and_independent_union_counts(self):
        objects = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        events = (ObservableSpec("ht", "continuous"),)
        stats = RuleStatistics(0, 0, 0, 0, 0., 0., 0., 0., 1, 2)

        def rule(rule_id, indices):
            cuts = (EventCut(ThresholdCut(0, ">", 0)),
                    *(ThresholdCut(index, ">", 0) for index in indices))
            steps = tuple(RuleStep(depth, cut, stats, 0.)
                          for depth, cut in enumerate(cuts, 1))
            return ExistentialRule(rule_id, objects, steps, minimum_count=2,
                                   event_observables=events)

        values = np.array([[[1, -1], [-1, 1]], [[1, 1], [1, 1]],
                           [[1, 1], [1, 1]]], float)
        event_values = np.array([[1], [1], [0]], float)
        conjunction = rule(0, (0, 1))
        np.testing.assert_array_equal(conjunction.predict(values, event_features=event_values),
                                      [False, True, False])
        union = RuleUnion(objects, (rule(1, (0,)), rule(2, (1,))), event_observables=events)
        np.testing.assert_array_equal(union.slot_mask(values, event_features=event_values),
                                      [[True, True], [True, True], [False, False]])
        np.testing.assert_array_equal(union.predict(values, event_features=event_values),
                                      [False, True, False])

    def test_native_event_only_matches_old_masked_dummy_search(self):
        dummy = (ObservableSpec("unused", "continuous"),)
        common = dict(max_rule_depth=3, beam_width=50, scoring_function=score,
                      beam_signal_similarity_threshold=.95, beam_background_similarity_threshold=.95)
        for similarity in (True, False):
            native = ExistentialRuleList(event_observables=EVENTS,
                beam_similarity_filter=similarity, **common).fit(
                signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
            old = ExistentialRuleList(dummy, event_observables=EVENTS,
                beam_similarity_filter=similarity, **common).fit(
                np.zeros((4, 1, 1)), np.zeros((4, 1, 1)),
                signal_slot_mask=np.zeros((4, 1), bool), background_slot_mask=np.zeros((4, 1), bool),
                signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
            with self.subTest(similarity=similarity):
                self.assertEqual([(r.cuts, r.generation_score, r.statistics) for r in native.candidate_rules_],
                                 [(r.cuts, r.generation_score, r.statistics) for r in old.candidate_rules_])
                for new_rule, old_rule in zip(native.candidate_rules_, old.candidate_rules_):
                    np.testing.assert_array_equal(new_rule.signal_selection, old_rule.signal_selection)
                    np.testing.assert_array_equal(new_rule.background_selection, old_rule.background_selection)

    def test_event_only_weighted_statistics_are_preserved(self):
        fitted = model().fit(signal_event_features=SIGNAL_EVENTS,
                             background_event_features=BACKGROUND_EVENTS,
                             signal_weight=[1, 2, 3, 4], background_weight=[4, 3, 2, 1])
        for rule in fitted.candidate_rules_:
            self.assertEqual(rule.statistics.signal_weight_accepted,
                             np.array([1, 2, 3, 4])[rule.signal_selection].sum())
            self.assertEqual(rule.statistics.background_weight_total, 10)

    def test_minimum_count_does_not_constrain_event_only_rules(self):
        fitted = self.native(minimum_count=100)
        self.assert_selections_match_predictions(fitted, signal_events=SIGNAL_EVENTS,
                                                background_events=BACKGROUND_EVENTS)
        self.assertTrue(fitted.candidate_rules_[0].signal_selection.any())

    def test_event_only_unions_and_empty_unions(self):
        fitted = self.native()
        rules = fitted.candidate_rules_[:3]
        union = fitted.build_union([rule.rule_id for rule in rules])
        expected = np.logical_or.reduce([rule.signal_selection for rule in rules])
        np.testing.assert_array_equal(union.predict(event_features=SIGNAL_EVENTS), expected)
        np.testing.assert_array_equal(union(event_features=SIGNAL_EVENTS), expected)
        self.assertEqual(union.slot_mask(event_features=SIGNAL_EVENTS).shape, (4, 0))
        empty = fitted.build_union([])
        np.testing.assert_array_equal(empty.predict(event_features=SIGNAL_EVENTS), [False] * 4)
        self.assertEqual(empty.to_text(), "FALSE")

    def test_union_with_unused_object_schema_can_predict_event_only(self):
        fitted = self.native(objects=OBJECTS)
        first = fitted.candidate_rules_[0]
        union = fitted.build_union([first.rule_id])
        np.testing.assert_array_equal(union.predict(event_features=SIGNAL_EVENTS), first.signal_selection)

    def test_prefix_zero_is_true_for_native_event_only(self):
        rule = self.native().candidate_rules_[0]
        np.testing.assert_array_equal(rule.predict(event_features=SIGNAL_EVENTS, depth=0), [True] * 4)
        self.assertEqual(rule.to_text(depth=0), "TRUE")

    def test_logical_pruning_and_refit_work_natively(self):
        fitted = self.native()
        original = fitted.candidate_rules_
        selected = fitted.prune_candidates(stricter_score_improvement=.01, similarity_filter=False)
        self.assertIs(fitted.candidate_rules_, original)
        self.assertEqual(selected, fitted.selected_rules_)
        self.assertEqual(len(fitted.pruning_result_.decisions), len(original))
        fitted.fit(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
        self.assertEqual(fitted.selected_rules_, [])
        self.assertIsNone(fitted.pruning_result_)

    def test_object_rules_raise_when_objects_are_missing(self):
        fitted = model(OBJECTS, ()).fit(SIGNAL_OBJECTS, BACKGROUND_OBJECTS,
                                      signal_slot_mask=SIGNAL_OBJECTS[..., 0] > 0,
                                      background_slot_mask=BACKGROUND_OBJECTS[..., 0] > 0)
        with self.assertRaisesRegex(ValueError, "object-level cuts"):
            fitted.candidate_rules_[0].predict()
        with self.assertRaises(ValueError):
            fitted.build_union([0]).predict()

    def test_native_slot_queries_raise_clear_errors(self):
        fitted = self.native()
        for callable_ in (fitted.candidate_rules_[0].matches_slots,
                          fitted.build_union([0]).matches_slots):
            with self.assertRaisesRegex(ValueError, "no object slots"):
                callable_(np.empty((4, 0)), event_features=SIGNAL_EVENTS)

    def test_missing_modalities_and_asymmetric_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            ExistentialRuleList()
        with self.assertRaises(ValueError):
            model().fit()
        with self.assertRaises(ValueError):
            model().fit(signal_event_features=SIGNAL_EVENTS)
        with self.assertRaises(ValueError):
            model(OBJECTS).fit(signal=SIGNAL_OBJECTS, signal_event_features=SIGNAL_EVENTS,
                               background_event_features=BACKGROUND_EVENTS)
        with self.assertRaises(ValueError):
            model().fit(signal_event_features=SIGNAL_EVENTS,
                        background_event_features=BACKGROUND_EVENTS, signal_slot_mask=np.zeros((4, 0), bool))

    def test_configured_event_schema_requires_features_in_combined_calls(self):
        with self.assertRaisesRegex(ValueError, "event_features is required"):
            model(OBJECTS).fit(SIGNAL_OBJECTS, BACKGROUND_OBJECTS)

    def test_invalid_event_values_and_weights_are_rejected(self):
        for index, value in ((0, np.nan), (1, .5), (2, 2), (3, 3)):
            bad = SIGNAL_EVENTS.copy()
            bad[0, index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                model().fit(signal_event_features=bad, background_event_features=BACKGROUND_EVENTS)
        for events in (SIGNAL_EVENTS[:, :3], SIGNAL_EVENTS[0], "not numeric"):
            with self.subTest(events=str(events)[:30]), self.assertRaises((ValueError, TypeError)):
                model().fit(signal_event_features=events, background_event_features=BACKGROUND_EVENTS)
        with self.assertRaises(ValueError):
            model().fit(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS,
                        signal_weight=[1, -1, 1, 1])

    def test_zero_sample_prediction_is_supported(self):
        fitted = self.native()
        self.assertEqual(fitted.candidate_rules_[0].predict(event_features=np.empty((0, 4))).shape, (0,))
        self.assertEqual(fitted.build_union([0]).predict(event_features=np.empty((0, 4))).shape, (0,))

    def test_native_set_input_has_no_dummy_slots_or_columns(self):
        data = SetInput.from_arrays(event_features=SIGNAL_EVENTS, event_observables=EVENTS,
                                    y=[1, 1, 1, 0], sample_weight=[1] * 4)
        self.assertEqual(data.X.shape, (4, 0, 0))
        self.assertEqual(data.slot_mask.shape, (4, 0))
        self.assertEqual(data.value_mask.shape, (4, 0, 0))
        self.assertEqual((data.n_samples, data.n_slots, data.n_observables), (4, 0, 0))
        for kwargs in (dict(value_mask=np.empty((4, 0, 0), bool)),
                       dict(event_features=None), dict(event_observables=())):
            args = dict(event_features=SIGNAL_EVENTS, event_observables=EVENTS)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                SetInput.from_arrays(**args)


if __name__ == "__main__":
    unittest.main()
