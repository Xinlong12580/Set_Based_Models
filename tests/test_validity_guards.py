"""Automatic validity guards: synthetic arrays only, no production training."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import (
    CategoryCut, EventCut, ExistentialDecisionTree, ExistentialRule, ExistentialRuleList, ObservableSpec,
    RuleStatistics, RuleStep, RuleUnion, SetInput, ThresholdCut,
)
from set_based_models.core.candidates import event_candidate_cuts
from set_based_models.decision_tree.objectives import gini_gain
from set_based_models.rule_list.pruning import prune_rule_candidates
from casual_tests import _quantitative_reporting as reporting

SPECS = (ObservableSpec("x", "continuous", validity_column="x_valid"),
         ObservableSpec("x_valid", "boolean"))
DISCRETE = (ObservableSpec("x", "categorical", {0: "a", 1: "b"}, validity_column="x_valid"),
            ObservableSpec("x_valid", "boolean"))


def score(stats):
    return stats.true_positive_rate - stats.false_positive_rate


def make_rule(rule_id, *cuts, events=False, count=1, value=.1):
    stats = RuleStatistics(1, 4, 1, 4, 1., 4., 1., 4., len(cuts), count)
    return ExistentialRule(rule_id, () if events else SPECS,
        tuple(RuleStep(i, cut, stats, value) for i, cut in enumerate(cuts, 1)),
        minimum_count=count, event_observables=SPECS if events else ())


class ValidityGuardTests(unittest.TestCase):
    def test_object_mask_excludes_invalid_value_but_keeps_measured_zero(self):
        values = np.array([[[0, 0]], [[0, 1]], [[2, 1]]], float)
        data = SetInput.from_arrays(values, SPECS)
        np.testing.assert_array_equal(data.value_mask[..., 0], [[False], [True], [True]])
        np.testing.assert_array_equal(ThresholdCut(0, "<=", 1).evaluate(data), [[False], [True], [False]])
        np.testing.assert_array_equal(ThresholdCut(0, ">", -1).evaluate(data), [[False], [True], [True]])

    def test_event_mask_excludes_invalid_value_but_keeps_measured_zero(self):
        events = np.array([[0, 0], [0, 1], [2, 1]], float)
        data = SetInput.from_arrays(event_features=events, event_observables=SPECS)
        np.testing.assert_array_equal(data.event_value_mask[:, 0], [False, True, True])
        for comparison, expected in (("<=", [False, True, False]), (">", [False, False, True])):
            np.testing.assert_array_equal(EventCut(ThresholdCut(0, comparison, 1)).evaluate(data), expected)

    def test_invalid_placeholders_may_be_nonfinite_or_invalid_discrete_codes(self):
        for specs in (SPECS, DISCRETE):
            for invalid in (np.nan, np.inf, -999, .5, 999):
                values = np.array([[invalid, 0], [0, 1]], float)
                with self.subTest(kind=specs[0].kind, invalid=invalid):
                    obj = SetInput.from_arrays(values[:, None], specs)
                    evt = SetInput.from_arrays(event_features=values, event_observables=specs)
                    cut = CategoryCut(0, {0}) if specs is DISCRETE else ThresholdCut(0, "<=", 0)
                    np.testing.assert_array_equal(cut.evaluate(obj)[:, 0], [False, True])
                    np.testing.assert_array_equal(EventCut(cut).evaluate(evt), [False, True])

    def test_missing_guards_wrong_kind_self_links_and_chains_are_rejected(self):
        invalid_schemas = (
            (SPECS[0],),
            (SPECS[0], ObservableSpec("x_valid", "continuous")),
            (ObservableSpec("x", "boolean", validity_column="x"),),
            (SPECS[0], ObservableSpec("x_valid", "boolean", validity_column="other"),
             ObservableSpec("other", "boolean")),
        )
        for schema in invalid_schemas:
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError):
                    ExistentialRuleList(schema, scoring_function=score)
                with self.assertRaises(ValueError):
                    ExistentialRuleList(event_observables=schema, scoring_function=score)
        for flag in ("", " ", 1, False):
            with self.assertRaises(ValueError):
                ObservableSpec("x", "continuous", validity_column=flag)

    def test_invalid_flag_values_and_invalid_present_measurements_are_rejected(self):
        for values in ([[np.nan, 1]], [[0, .5]], [[0, np.nan]], [[0, 2]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SetInput.from_arrays(np.array(values)[:, None], SPECS)
                with self.assertRaises(ValueError):
                    SetInput.from_arrays(event_features=values, event_observables=SPECS)

    def test_explicit_value_mask_is_combined_without_mutating_caller(self):
        values = np.array([[[10, 1], [0, 0]]], float)
        mask = np.array([[[False, True], [True, True]]])
        original = mask.copy()
        data = SetInput.from_arrays(values, SPECS, value_mask=mask)
        np.testing.assert_array_equal(data.value_mask[..., 0], [[False, False]])
        np.testing.assert_array_equal(mask, original)
        # A flag which is itself missing cannot validate the associated value.
        mask = np.array([[[True, False], [True, True]]])
        data = SetInput.from_arrays(values, SPECS, value_mask=mask)
        np.testing.assert_array_equal(data.value_mask[..., 0], [[False, False]])

    def test_padding_is_excluded_even_if_its_guard_is_true(self):
        objects = np.array([[[0, 1], [0, 1]]], float)
        data = SetInput.from_arrays(objects, SPECS, slot_mask=np.array([[True, False]]))
        np.testing.assert_array_equal(ThresholdCut(0, "<=", 1).evaluate(data), [[True, False]])

    def test_rule_prediction_slot_helpers_and_union_enforce_same_object_guards(self):
        objects = np.array([[[0, 0], [2, 1]], [[0, 1], [2, 1]], [[0, 1], [0, 1]]], float)
        low = make_rule(0, ThresholdCut(0, "<=", 1))
        high = make_rule(1, ThresholdCut(0, ">", 1))
        np.testing.assert_array_equal(low(objects), [False, True, True])
        np.testing.assert_array_equal(low.matches_slots(objects), [[False, False], [True, False], [True, True]])
        self.assertFalse(low.matches_slot([0, 0]))
        np.testing.assert_array_equal(replace(low, minimum_count=2).predict(objects), [False, False, True])
        union = RuleUnion(SPECS, (low, high))
        np.testing.assert_array_equal(union.predict(objects), [True, True, True])
        np.testing.assert_array_equal(union.slot_mask(objects), [[False, True], [True, True], [True, True]])
        self.assertIn("x_valid == True AND x <= 1", low.to_text())

    def test_native_event_rule_and_union_require_validity_at_prediction(self):
        events = np.array([[0, 0], [0, 1], [2, 1]], float)
        low = make_rule(0, EventCut(ThresholdCut(0, "<=", 1)), events=True)
        high = make_rule(1, EventCut(ThresholdCut(0, ">", 1)), events=True)
        np.testing.assert_array_equal(low(event_features=events), [False, True, False])
        np.testing.assert_array_equal(low.predict(event_features=events, depth=0), [True, True, True])
        union = RuleUnion((), (low, high), event_observables=SPECS)
        np.testing.assert_array_equal(union(event_features=events), [False, True, True])

    def test_validity_guard_is_not_an_extra_search_step(self):
        signal = np.array([[0, 1], [1, 1], [0, 0], [4, 1]], float)
        background = np.array([[3, 1], [4, 1], [0, 0], [5, 1]], float)
        for event_only in (True, False):
            model = ExistentialRuleList(() if event_only else SPECS,
                event_observables=SPECS if event_only else (), max_rule_depth=1,
                beam_similarity_filter=False, min_signal_efficiency=.01, scoring_function=score)
            if event_only:
                model.fit(signal_event_features=signal, background_event_features=background)
            else:
                model.fit(signal[:, None], background[:, None])
            measurements = [r for r in model.candidate_rules_ if r.cuts[0].observable_index == 0]
            self.assertTrue(measurements)
            for rule in measurements:
                self.assertEqual(rule.depth, 1)
                self.assertIn("x_valid == True", rule.to_text())
                kwargs = {"event_features": signal} if event_only else {}
                np.testing.assert_array_equal(rule.predict(None if event_only else signal[:, None], **kwargs),
                                              rule.signal_selection)
                performance = model.evaluate_rule(rule.rule_id,
                    None if event_only else signal[:, None], None if event_only else background[:, None],
                    signal_event_features=signal if event_only else None,
                    background_event_features=background if event_only else None)
                self.assertEqual(performance.signal_accepted, rule.statistics.signal_accepted)

    def test_candidate_thresholds_ignore_event_placeholders(self):
        s = SetInput.from_arrays(event_features=[[0, 1], [2, 1], [999, 0]], event_observables=SPECS)
        b = SetInput.from_arrays(event_features=[[4, 1], [6, 1], [-999, 0]], event_observables=SPECS)
        cuts = event_candidate_cuts(s, b, np.ones(3, bool), np.ones(3, bool), None)
        self.assertEqual({cut.cut.threshold for cut in cuts if isinstance(cut.cut, ThresholdCut)}, {1, 3, 5})
        cuts = event_candidate_cuts(s, b, np.array([2, 0]), np.array([2, 1]), None)
        self.assertEqual({cut.cut.threshold for cut in cuts if isinstance(cut.cut, ThresholdCut)}, {3})

    def test_search_scores_rules_and_selections_are_invariant_to_invalid_fills(self):
        for event_only in (True, False):
            for similarity in (True, False):
                baseline = None
                for fill in (0, -999, 999, np.nan):
                    s = np.array([[0, 1], [2, 1], [fill, 0], [4, 1]], float)
                    b = np.array([[3, 1], [5, 1], [fill, 0], [6, 1]], float)
                    model = ExistentialRuleList(() if event_only else SPECS,
                        event_observables=SPECS if event_only else (), max_rule_depth=2,
                        beam_similarity_filter=similarity, min_signal_efficiency=.01, scoring_function=score)
                    if event_only:
                        model.fit(signal_event_features=s, background_event_features=b)
                    else:
                        model.fit(s[:, None], b[:, None])
                    actual = [(r.cuts, r.generation_score, r.statistics,
                               r.signal_selection.tolist(), r.background_selection.tolist())
                              for r in model.candidate_rules_]
                    if baseline is None:
                        baseline = actual
                    self.assertEqual(actual, baseline)

    def test_logical_pruning_understands_implicit_guards(self):
        for events in (True, False):
            wrap = EventCut if events else lambda cut: cut
            x = wrap(ThresholdCut(0, "<=", 1))
            flag_true = wrap(CategoryCut(1, {1}))
            flag_false = wrap(CategoryCut(1, {0}))
            guarded = make_rule(0, x, events=events)
            explicit = make_rule(1, flag_true, x, events=events)
            merged = prune_rule_candidates([explicit, guarded], stricter_score_improvement=0, similarity_filter=False)
            self.assertEqual([r.rule_id for r in merged.retained], [0])
            self.assertEqual(merged.decisions[1]["status"], "equivalent_rule")
            true = make_rule(2, flag_true, events=events)
            false = make_rule(3, flag_false, events=events)
            result = prune_rule_candidates([guarded, true, false], stricter_score_improvement=0, similarity_filter=False)
            self.assertEqual({r.rule_id for r in result.retained}, {2, 3})
            self.assertEqual(result.decisions[0]["pruned_by_rule_id"], 2)

    def test_report_metadata_and_display_include_guard(self):
        for cut in (ThresholdCut(0, "<=", 1), EventCut(ThresholdCut(0, "<=", 1)), CategoryCut(0, {0})):
            schema = DISCRETE if isinstance(cut, CategoryCut) else SPECS
            record = reporting._cut_record(cut, schema)
            self.assertEqual(record["validity_guard"]["observable"], "x_valid")
            self.assertIn("x_valid == True", record["display_text"])
        ordinal = (ObservableSpec("x", "ordinal", validity_column="x_valid"), SPECS[1])
        self.assertEqual(reporting._predicate_text(ThresholdCut(0, "<", 1.5), ordinal),
                         "x_valid == True AND x <= 1")

    def test_raw_converter_schema_maps_object_and_event_guards(self):
        for item in ({"kind": "continuous", "validity_column": "x_valid"},
                     {"kind": "continuous", "definition": {"validity_feature": "x_valid"}}):
            self.assertEqual(reporting._observable_from_schema("x", item), SPECS[0])
        self.assertIsNone(reporting._observable_from_schema("x", {"kind": "continuous"}).validity_column)
        with self.assertRaises(ValueError):
            reporting._observable_from_schema("x", {"kind": "continuous", "validity_column": "x_valid",
                                                      "definition": {"validity_feature": "other"}})

    def test_mixed_scope_predictions_and_slot_helpers_are_consistent(self):
        stats = RuleStatistics(1, 3, 0, 3, 1., 3., 0., 3., 2, 1)
        rule = ExistentialRule(0, SPECS,
            (RuleStep(1, ThresholdCut(0, "<=", 1), stats, .1),
             RuleStep(2, EventCut(ThresholdCut(0, "<=", 1)), stats, .1)), event_observables=SPECS)
        objects = np.array([[[0, 0], [2, 1]], [[0, 1], [2, 1]], [[0, 1], [2, 1]]], float)
        events = np.array([[0, 1], [0, 0], [0, 1]], float)
        np.testing.assert_array_equal(rule.predict(objects, event_features=events), [False, False, True])
        np.testing.assert_array_equal(rule.matches_slots(objects, event_features=events),
                                      [[False, False], [False, False], [True, False]])
        self.assertFalse(rule.matches_slot([0, 1], event_features=[0, 0]))

    def test_missing_measurements_cannot_satisfy_a_guarded_event_complement_display(self):
        text = reporting._negated_event_text(EventCut(ThresholdCut(0, ">", 1)), SPECS)
        self.assertEqual(text, "NOT (event: x_valid == True AND x > 1)")

    def test_logical_normalization_removes_explicit_repeated_guard(self):
        rule = make_rule(0, CategoryCut(1, {1}), ThresholdCut(0, "<=", 1))
        result = prune_rule_candidates([rule], stricter_score_improvement=0, similarity_filter=False)
        self.assertEqual(result.normalized_cuts[0], (ThresholdCut(0, "<=", 1),))

    def test_shared_tree_candidate_generation_and_slicing_remain_consistent(self):
        dummy = (ObservableSpec("unused", "continuous"),)
        objects = np.zeros((3, 1, 1))
        mask = np.zeros((3, 1), bool)
        baseline = None
        for fill in (0, 999, np.nan):
            s = np.array([[0, 1], [1, 1], [fill, 0]], float)
            b = np.array([[3, 1], [4, 1], [fill, 0]], float)
            tree = ExistentialDecisionTree(dummy, event_observables=SPECS, max_depth=2,
                                          gain_function=gini_gain).fit(
                objects, objects, signal_slot_mask=mask, background_slot_mask=mask,
                signal_event_features=s, background_event_features=b)
            actual = (tree.predict(objects, slot_mask=mask, event_features=s).tolist(),
                      tree.predict(objects, slot_mask=mask, event_features=b).tolist(),
                      [node.cut.to_text(SPECS) if node.cut is not None else None for node in tree.nodes_])
            if baseline is None:
                baseline = actual
            self.assertEqual(actual, baseline)

    def test_guard_only_default_preserves_other_booleans_and_absence_cuts(self):
        specs = (*SPECS, ObservableSpec("physical_valid", "boolean"),
                 ObservableSpec("multiplicity", "ordinal"))
        s = np.array([[0, 1, 1, 0], [2, 1, 0, 1], [99, 0, 1, 0], [4, 1, 0, 2]], float)
        b = np.array([[3, 1, 0, 0], [5, 1, 0, 0], [99, 0, 0, 1], [6, 1, 0, 0]], float)
        for event_only in (True, False):
            fitted = ExistentialRuleList(() if event_only else specs,
                event_observables=specs if event_only else (), max_rule_depth=2,
                beam_width=100, max_candidate_rules=None, beam_similarity_filter=False,
                min_signal_efficiency=.01, scoring_function=score)
            if event_only:
                fitted.fit(signal_event_features=s, background_event_features=b)
            else:
                fitted.fit(s[:, None], b[:, None])
            cuts = [cut for rule in fitted.candidate_rules_ for cut in rule.cuts]
            self.assertFalse(fitted.allow_validity_cuts)
            self.assertTrue(cuts)
            self.assertNotIn(1, {cut.observable_index for cut in cuts})
            # The '_valid' suffix alone does not make a Boolean a guard.
            self.assertIn(2, {cut.observable_index for cut in cuts})
            self.assertTrue(any(cut.observable_index == 3 and
                (cut.cut if isinstance(cut, EventCut) else cut).comparison.value == "<=" for cut in cuts))
            measurements = [r for r in fitted.candidate_rules_ if any(c.observable_index == 0 for c in r.cuts)]
            self.assertTrue(measurements)
            self.assertTrue(all("x_valid == True" in r.to_text() for r in measurements))
            for rule in measurements:
                selection = rule.predict(event_features=s) if event_only else rule.predict(s[:, None])
                self.assertFalse(selection[2])  # false validity still rejects the measurement
                np.testing.assert_array_equal(selection, rule.signal_selection)

    def test_explicit_opt_in_restores_true_and_false_flag_candidates(self):
        s = np.array([[0, 1], [2, 1], [99, 0], [4, 1]], float)
        b = np.array([[3, 1], [5, 1], [99, 0], [6, 1]], float)
        for event_only in (True, False):
            fitted = ExistentialRuleList(() if event_only else SPECS,
                event_observables=SPECS if event_only else (), max_rule_depth=1,
                beam_width=100, beam_similarity_filter=False, min_signal_efficiency=.01,
                allow_validity_cuts=True, scoring_function=score)
            if event_only:
                fitted.fit(signal_event_features=s, background_event_features=b)
            else:
                fitted.fit(s[:, None], b[:, None])
            flags = [r.cuts[0] for r in fitted.candidate_rules_ if r.cuts[0].observable_index == 1]
            self.assertEqual({(c.cut if isinstance(c, EventCut) else c).accepted_codes for c in flags},
                             {frozenset({0}), frozenset({1})})
            for rule in fitted.candidate_rules_:
                if rule.cuts[0].observable_index != 0:
                    continue
                selection = rule.predict(event_features=s) if event_only else rule.predict(s[:, None])
                self.assertFalse(selection[2])

    def test_object_and_event_guard_roles_are_independent(self):
        events = (ObservableSpec("x_valid", "boolean"),)
        fitted = ExistentialRuleList(SPECS, event_observables=events,
            max_rule_depth=1, beam_width=100, beam_similarity_filter=False,
            min_signal_efficiency=.01, scoring_function=score).fit(
                np.array([[[0, 1]], [[2, 0]], [[3, 1]]], float),
                np.array([[[3, 1]], [[4, 0]], [[5, 1]]], float),
                signal_event_features=[[1], [1], [0]], background_event_features=[[0], [0], [0]])
        self.assertTrue(any(isinstance(c, EventCut) for r in fitted.candidate_rules_ for c in r.cuts))
        self.assertFalse(any(not isinstance(c, EventCut) and c.observable_index == 1
                             for r in fitted.candidate_rules_ for c in r.cuts))

    def test_invalid_allow_validity_cuts_values_are_rejected(self):
        for value in (None, 1, 0, "false"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ExistentialRuleList(SPECS, allow_validity_cuts=value)


if __name__ == "__main__":
    unittest.main()
