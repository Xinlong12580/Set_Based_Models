"""Built-in final pruning tests using synthetic rules only."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from set_based_models import (
    CategoryCut, EventCut, ExistentialRule, ExistentialRuleList, ObservableSpec,
    RulePruningResult, RuleStatistics, RuleStep, ThresholdCut,
)
from set_based_models.rule_list.pruning import normalize_rule, rule_implies

OBJECTS = (ObservableSpec("pt", "continuous"),)
EVENTS = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))


def event(index=0, threshold=0):
    return EventCut(ThresholdCut(index, ">", threshold))


def rule(rule_id, score, *cuts, reference_accepted=1):
    steps = tuple(
        RuleStep(depth, cut, RuleStatistics(2, 10, reference_accepted, 10,
                 2., 10., float(reference_accepted), 10., depth, 1), score)
        for depth, cut in enumerate(cuts, 1)
    )
    return ExistentialRule(rule_id, OBJECTS, steps, event_observables=EVENTS,
                           signal_selection=np.array([True, False]),
                           background_selection=np.array([False, True]))


def model_with(*rules):
    model = ExistentialRuleList(OBJECTS, event_observables=EVENTS)
    model.candidate_rules_ = list(rules)
    return model


class BuiltInPruningTests(unittest.TestCase):
    def select(self, model, improvement=0, **kwargs):
        return model.prune_candidates(stricter_score_improvement=improvement,
                                      similarity_filter=False, **kwargs)

    def test_default_preserves_jaccard_policy_and_candidate_identity(self):
        a, b = rule(0, .2, event()), rule(1, .1, event(1))
        model = model_with(a, b)
        candidates = model.candidate_rules_
        self.assertEqual(model.prune_candidates(), [a])
        self.assertIs(model.candidate_rules_, candidates)
        self.assertEqual(model.candidate_rules_, [a, b])
        self.assertIs(model.get_rule(1), b)
        self.assertEqual(model.selected_rules_, [a])
        self.assertIsInstance(model.pruning_result_, RulePruningResult)
        self.assertEqual(model.pruning_result_.decisions[1]["status"], "similarity")

    def test_disabled_filters_keep_every_candidate(self):
        a, b = rule(0, .2, event()), rule(1, .1, event(), event(1))
        model = model_with(a, b)
        self.assertEqual(model.prune_candidates(similarity_filter=False), [a, b])

    def test_tighter_threshold_and_extra_and_share_one_control(self):
        a = rule(0, .1, event())
        b = rule(1, .109, event(threshold=10))
        c = rule(2, .109, event(), event(1))
        model = model_with(b, c, a)
        self.assertEqual(self.select(model, .01), [a])
        for rule_id in (1, 2):
            decision = model.pruning_result_.decisions[rule_id]
            self.assertEqual(decision["status"], "stricter_score_improvement")
            self.assertEqual(decision["retained_witness_rule_id"], 0)
        self.assertEqual(len(model.candidate_rules_), 3)

    def test_equality_at_improvement_boundary_is_pruned(self):
        a, b = rule(0, .1, event()), rule(1, .11, event(threshold=10))
        self.assertEqual(self.select(model_with(b, a), .01), [a])

    def test_sufficient_improvement_keeps_stricter_and_looser(self):
        a, b = rule(0, .1, event()), rule(1, .12, event(threshold=10))
        self.assertEqual(self.select(model_with(b, a), .01), [b, a])

    def test_zero_prunes_tied_or_worse_stricter_rules(self):
        a, b = rule(0, .1, event()), rule(1, .1, event(), event(1))
        self.assertEqual(self.select(model_with(b, a)), [a])

    def test_equivalence_prefers_shortest_and_normalizes_self_nesting(self):
        a = rule(0, .1, event(threshold=50), event(threshold=40))
        b = rule(1, .1, event(threshold=50))
        model = model_with(a, b)
        self.assertEqual(self.select(model), [b])
        self.assertEqual(model.pruning_result_.normalized_cuts[0], (event(threshold=50),))
        self.assertEqual(model.pruning_result_.decisions[0]["status"], "equivalent_rule")

    def test_only_retained_looser_rules_act_as_witnesses(self):
        a, b = rule(0, 0, event()), rule(1, .009, event(threshold=1))
        c = rule(2, .018, event(threshold=1), event(1))
        self.assertEqual(self.select(model_with(c, b, a), .01), [c, a])

    def test_unrelated_columns_are_not_pruned_using_equal_masks(self):
        a, b = rule(0, .2, event()), rule(1, .1, event(1))
        self.assertEqual(self.select(model_with(a, b), 100), [a, b])

    def test_object_and_mixed_rules_receive_logical_pruning(self):
        a = rule(0, .2, event())
        b = rule(1, .1, ThresholdCut(0, ">", 10))
        c = rule(2, .1, event(), ThresholdCut(0, ">", 10))
        d = replace(b, rule_id=3, minimum_count=2)
        model = model_with(a, b, c, d)
        self.assertEqual(self.select(model, 100), [a, b])
        self.assertEqual(model.pruning_result_.normalized_cuts[2], c.cuts)
        for rule_id in (2, 3):
            self.assertEqual(model.pruning_result_.decisions[rule_id]["status"],
                             "stricter_score_improvement")

    def test_rate_filter_prevents_ineligible_witnesses(self):
        a = rule(0, .2, event(), reference_accepted=2)
        b = rule(1, .1, event(), event(1), reference_accepted=1)
        model = model_with(a, b)
        self.assertEqual(self.select(model, 100, max_reference_acceptance=.1), [b])
        self.assertEqual(model.pruning_result_.decisions[0]["status"], "reference_rate_limit")

    def test_rate_filter_also_applies_to_object_rules(self):
        a = rule(0, .2, ThresholdCut(0, ">", 10), reference_accepted=2)
        self.assertEqual(self.select(model_with(a), max_reference_acceptance=.1), [])

    def test_rate_filter_uses_weighted_acceptance(self):
        a = rule(0, .2, event(), reference_accepted=2)
        a = replace(a, steps=(replace(a.steps[0], statistics=replace(
            a.statistics, background_weight_accepted=.5)),))
        self.assertEqual(self.select(model_with(a), max_reference_acceptance=.1), [a])

    def test_similarity_can_be_enabled_alongside_logical_pruning(self):
        a, b = rule(0, .2, event()), rule(1, .1, event(1))
        model = model_with(a, b)
        self.assertEqual(model.prune_candidates(stricter_score_improvement=0), [a])
        self.assertEqual(model.pruning_result_.decisions[1]["status"], "similarity")

    def test_similarity_audit_resolves_logical_witness_chain(self):
        a = rule(0, .1, event())
        b = rule(1, .11, event(), event(1))
        c = rule(2, .2, event(threshold=10))
        model = model_with(c, b, a)
        self.assertEqual(model.prune_candidates(stricter_score_improvement=.02), [c])
        self.assertEqual(model.pruning_result_.decisions[1]["pruned_by_rule_id"], 0)
        self.assertEqual(model.pruning_result_.decisions[1]["retained_witness_rule_id"], 2)

    def test_logical_pruning_does_not_require_training_masks(self):
        a = replace(rule(0, .1, event()), signal_selection=None, background_selection=None)
        model = model_with(a)
        self.assertEqual(self.select(model), [a])
        with self.assertRaises(RuntimeError):
            model.prune_candidates()

    def test_invalid_options_do_not_replace_previous_result(self):
        model = model_with(rule(0, .1, event()))
        self.select(model)
        original = model.pruning_result_
        for name, values in (
            ("stricter_score_improvement", (-1, float("nan"), float("inf"))),
            ("max_reference_acceptance", (-1, 1.1, float("nan"), float("inf"))),
            ("signal_similarity_threshold", (-1, 1.1, float("nan"))),
            ("background_similarity_threshold", (-1, 1.1, float("nan"))),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    model.prune_candidates(**{name: value})
                self.assertIs(model.pruning_result_, original)
        with self.assertRaises(TypeError):
            model.prune_candidates(similarity_filter=1)

    def test_duplicate_ids_and_nonfinite_scores_rejected(self):
        a = rule(0, .1, event())
        with self.assertRaises(ValueError):
            self.select(model_with(a, a))
        with self.assertRaises(ValueError):
            self.select(model_with(rule(0, float("nan"), event())))

    def test_empty_pool_has_empty_audit(self):
        model = model_with()
        self.assertEqual(self.select(model), [])
        self.assertEqual(model.pruning_result_.decisions, {})

    def test_refit_clears_selection_and_audit(self):
        model = model_with(rule(0, .1, event()))
        self.select(model)
        with patch("set_based_models.rule_list.model.BeamRuleGenerator.generate", return_value=[]):
            model.fit(np.zeros((2, 1, 1)), np.zeros((2, 1, 1)),
                      signal_event_features=np.zeros((2, 2)),
                      background_event_features=np.zeros((2, 2)))
        self.assertEqual(model.candidate_rules_, [])
        self.assertEqual(model.selected_rules_, [])
        self.assertIsNone(model.pruning_result_)

    def test_returned_list_does_not_alias_model_selection(self):
        model = model_with(rule(0, .1, event()))
        selected = self.select(model)
        selected.clear()
        self.assertEqual(len(model.selected_rules_), 1)


SLOTS = (
    ObservableSpec("pt", "continuous"), ObservableSpec("eta", "continuous"),
    ObservableSpec("type", "categorical", categories={0: "jet", 1: "photon", 2: "muon"}),
    ObservableSpec("n_charged", "ordinal"), ObservableSpec("valid", "boolean"),
)


class ObjectLogicalPruningTests(unittest.TestCase):
    def make_rule(self, rule_id, score, *cuts, count=1):
        base = rule(rule_id, score, ThresholdCut(0, ">", 0))
        return replace(base, observables=SLOTS, minimum_count=count, steps=tuple(
            replace(base.steps[0], depth=depth, cut=cut)
            for depth, cut in enumerate(cuts, 1)))

    def select(self, *rules, improvement=0, **kwargs):
        model = ExistentialRuleList(SLOTS, event_observables=EVENTS)
        model.candidate_rules_ = list(rules)
        model.prune_candidates(stricter_score_improvement=improvement,
                               similarity_filter=False, **kwargs)
        self.result = model.pruning_result_
        return [rule.rule_id for rule in model.selected_rules_]

    def test_tighter_pt_and_extra_same_object_and(self):
        a = self.make_rule(0, .1, ThresholdCut(0, ">", 40))
        b = self.make_rule(1, .105, ThresholdCut(0, ">", 50))
        c = self.make_rule(2, .105, ThresholdCut(0, ">", 40), ThresholdCut(1, "<", 2))
        self.assertEqual(self.select(c, b, a, improvement=.01), [0])
        for rule_id in (1, 2):
            self.assertEqual(self.result.decisions[rule_id]["retained_witness_rule_id"], 0)
            self.assertAlmostEqual(self.result.decisions[rule_id]["score_improvement"], .005)

    def test_higher_score_keeps_stricter_without_removing_simpler(self):
        a = self.make_rule(0, .1, ThresholdCut(0, ">", 40))
        b = self.make_rule(1, .12, ThresholdCut(0, ">", 50))
        self.assertEqual(self.select(b, a, improvement=.01), [1, 0])
        b = replace(b, steps=(replace(b.steps[0], generation_score=.11),))
        self.assertEqual(self.select(b, a, improvement=.01), [0])

    def test_self_nested_and_reordered_object_cuts_merge(self):
        pt, eta = ThresholdCut(0, ">", 50), ThresholdCut(1, "<", 2)
        a = self.make_rule(0, .1, pt, ThresholdCut(0, ">", 40), eta)
        b = self.make_rule(1, .1, eta, pt)
        self.assertEqual(self.select(a, b), [1])
        self.assertEqual(self.result.normalized_cuts[0], (pt, eta))
        self.assertEqual(self.result.decisions[0]["status"], "equivalent_rule")

    def test_open_and_closed_bounds_are_distinct(self):
        for comparison in (">", "<"):
            with self.subTest(comparison=comparison):
                a = self.make_rule(0, .1, ThresholdCut(0, comparison + "=", 50))
                b = self.make_rule(1, .1, ThresholdCut(0, comparison, 50))
                self.assertEqual(self.select(b, a), [0])
                self.assertEqual(self.result.decisions[1]["status"], "stricter_score_improvement")

    def test_integer_threshold_equivalence(self):
        a = self.make_rule(0, .1, ThresholdCut(3, ">", 2.5))
        b = self.make_rule(1, .1, ThresholdCut(3, ">=", 3))
        self.assertEqual(self.select(a, b), [0])
        self.assertEqual(self.result.normalized_cuts[0], (ThresholdCut(3, ">=", 3),))
        self.assertEqual(self.result.decisions[1]["status"], "equivalent_rule")

    def test_category_subsets_and_intersections(self):
        a = self.make_rule(0, .1, CategoryCut(2, {0, 1}))
        b = self.make_rule(1, .1, CategoryCut(2, {1}))
        self.assertEqual(self.select(b, a), [0])
        c = self.make_rule(2, .1, CategoryCut(2, {0, 1}), CategoryCut(2, {1, 2}))
        self.assertEqual(self.select(c, b), [1])
        self.assertEqual(self.result.normalized_cuts[2], (CategoryCut(2, {1}),))

    def test_distinct_object_types_are_not_related(self):
        a = self.make_rule(0, .2, CategoryCut(2, {0}), ThresholdCut(0, ">", 40))
        b = self.make_rule(1, .1, CategoryCut(2, {1}), ThresholdCut(0, ">", 50))
        self.assertEqual(self.select(a, b, improvement=100), [0, 1])

    def test_event_and_object_column_zero_are_not_interchangeable(self):
        a = self.make_rule(0, .1, event(threshold=40))
        b = self.make_rule(1, .1, ThresholdCut(0, ">", 50))
        self.assertEqual(self.select(a, b, improvement=100), [0, 1])
        self.assertFalse(rule_implies(b.cuts, a.cuts))
        self.assertFalse(rule_implies(a.cuts, b.cuts))

    def test_mixed_thresholds_and_and_conditions(self):
        a = self.make_rule(0, .1, event(threshold=40), ThresholdCut(0, ">", 40))
        b = self.make_rule(1, .105, ThresholdCut(0, ">", 50), event(threshold=50))
        c = self.make_rule(2, .105, *a.cuts, CategoryCut(2, {1}))
        self.assertEqual(self.select(c, b, a, improvement=.01), [0])
        self.assertEqual(self.result.normalized_cuts[1], (event(threshold=50), ThresholdCut(0, ">", 50)))

    def test_mixed_rules_can_be_subsets_of_pure_event_or_object_rules(self):
        for cut in (event(), ThresholdCut(0, ">", 40)):
            with self.subTest(cut=cut):
                a = self.make_rule(0, .1, cut)
                b = self.make_rule(1, .1, event(), ThresholdCut(0, ">", 40))
                self.assertEqual(self.select(b, a), [0])

    def test_minimum_count_is_part_of_equivalence_and_containment(self):
        pt = ThresholdCut(0, ">", 40)
        a = self.make_rule(0, .1, pt, count=1)
        b = self.make_rule(1, .105, pt, count=2)
        self.assertEqual(self.select(b, a, improvement=.01), [0])
        self.assertEqual(self.result.decisions[1]["status"], "stricter_score_improvement")
        b = self.make_rule(1, .12, pt, count=2)
        self.assertEqual(self.select(b, a, improvement=.01), [1, 0])

    def test_tighter_pt_with_lower_count_does_not_imply_higher_count(self):
        a = self.make_rule(0, .1, ThresholdCut(0, ">", 40), count=2)
        b = self.make_rule(1, .1, ThresholdCut(0, ">", 50), count=1)
        self.assertEqual(self.select(b, a, improvement=100), [0, 1])

    def test_event_only_count_is_ignored_but_object_count_is_not(self):
        a = self.make_rule(0, .1, event(), count=100)
        b = self.make_rule(1, .1, event(), count=1)
        c = self.make_rule(2, .1, event(), ThresholdCut(0, ">", 40), count=1)
        self.assertEqual(self.select(c, b, a), [0])
        self.assertEqual(self.result.decisions[1]["status"], "equivalent_rule")
        self.assertEqual(self.result.decisions[2]["status"], "stricter_score_improvement")

    def test_same_object_conjunction_is_not_independent_existence(self):
        a = self.make_rule(0, .1, ThresholdCut(0, ">", 40))
        b = self.make_rule(1, .1, ThresholdCut(1, "<", 1))
        c = self.make_rule(2, .1, *a.cuts, *b.cuts)
        objects = np.array([[[60, 2, 1, 0, 1], [20, 0, 1, 0, 1]]], dtype=float)
        events = np.zeros((1, 2))
        self.assertTrue(a.predict(objects, event_features=events)[0])
        self.assertTrue(b.predict(objects, event_features=events)[0])
        self.assertFalse(c.predict(objects, event_features=events)[0])
        self.assertEqual(self.select(c, b, a), [0, 1])

    def test_normalization_preserves_predictions_with_padding_and_counts(self):
        rng = np.random.default_rng(42)
        objects = rng.normal(size=(100, 5, len(SLOTS)))
        objects[..., 0] *= 30
        objects[..., 2] = rng.integers(0, 3, (100, 5))
        objects[..., 3] = rng.integers(0, 10, (100, 5))
        objects[..., 4] = rng.integers(0, 2, (100, 5))
        mask = rng.random((100, 5)) > .3
        objects[~mask] = np.nan
        events = rng.normal(size=(100, 2))
        cuts = (ThresholdCut(0, ">", 10), ThresholdCut(0, ">", 0),
                CategoryCut(2, {0, 1}), CategoryCut(2, {1, 2}),
                ThresholdCut(3, ">", 2.5), ThresholdCut(3, ">=", 3),
                event(), event(threshold=-1))
        for count in (1, 2, 3):
            original = self.make_rule(0, .1, *cuts, count=count)
            normalized = self.make_rule(1, .1, *normalize_rule(original), count=count)
            np.testing.assert_array_equal(
                original.predict(objects, slot_mask=mask, event_features=events),
                normalized.predict(objects, slot_mask=mask, event_features=events))

    def test_rate_filter_removes_object_witness_before_logical_stage(self):
        a = self.make_rule(0, .1, ThresholdCut(0, ">", 40))
        b = self.make_rule(1, .1, ThresholdCut(0, ">", 50))
        a = replace(a, steps=(replace(a.steps[0], statistics=replace(
            a.statistics, background_weight_accepted=2.)),))
        self.assertEqual(self.select(b, a, max_reference_acceptance=.1), [1])
        self.assertEqual(self.result.decisions[0]["status"], "reference_rate_limit")

    def test_every_proven_relation_agrees_with_synthetic_predictions(self):
        rng = np.random.default_rng(7)
        objects = np.zeros((300, 4, len(SLOTS)))
        objects[..., 0] = rng.integers(0, 4, (300, 4))
        objects[..., 1] = rng.integers(0, 3, (300, 4))
        objects[..., 2] = rng.integers(0, 3, (300, 4))
        objects[..., 3] = rng.integers(0, 5, (300, 4))
        mask = rng.random((300, 4)) > .2
        events = rng.integers(0, 3, (300, 2))
        groups = (
            (ThresholdCut(0, ">", 1),), (ThresholdCut(0, ">", 2),),
            (ThresholdCut(0, ">=", 2),), (ThresholdCut(1, "<", 1),),
            (CategoryCut(2, {0, 1}),), (CategoryCut(2, {1}),),
            (ThresholdCut(0, ">", 1), CategoryCut(2, {1})),
            (event(threshold=1),),
            (event(threshold=1), ThresholdCut(0, ">", 1)),
        )
        rules = [self.make_rule(i, .1, *cuts, count=count)
                 for i, (cuts, count) in enumerate((cuts, count)
                 for cuts in groups for count in (1, 2, 3))]
        selections = {r.rule_id: r.predict(objects, slot_mask=mask, event_features=events)
                      for r in rules}
        # Compare pairs independently so each provable relation is exercised,
        # rather than hidden by a different retained witness in a large pool.
        compared = 0
        for i, left in enumerate(rules):
            for right in rules[i + 1:]:
                self.select(right, left)
                for rule_id, decision in self.result.decisions.items():
                    witness = decision["pruned_by_rule_id"]
                    if witness is None:
                        continue
                    compared += 1
                    self.assertFalse(np.any(selections[rule_id] & ~selections[witness]))
                    if decision["status"] == "equivalent_rule":
                        np.testing.assert_array_equal(selections[rule_id], selections[witness])
        self.assertGreater(compared, 50)

    def test_different_schemas_cannot_be_compared(self):
        a = self.make_rule(0, .1, ThresholdCut(0, ">", 40))
        b = replace(a, rule_id=1, observables=(ObservableSpec("energy", "continuous"),))
        with self.assertRaises(ValueError):
            self.select(a, b)


if __name__ == "__main__":
    unittest.main()
