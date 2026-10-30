"""Guard-aware rule shortening and independent validation candidate pools."""

import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from set_based_models import (
    CategoryCut, EventCut, ExistentialRule, ExistentialRuleList, ObservableSpec,
    RuleStatistics, RuleStep, ThresholdCut,
)


def score(stats):
    return stats.true_positive_rate - stats.false_positive_rate


def event(index=0, comparison=">", threshold=0):
    return EventCut(ThresholdCut(index, comparison, threshold))


def model_with(cut_lists, *, objects=(), events=(), minimum_count=1, priority=()):
    model = ExistentialRuleList(objects, event_observables=events, scoring_function=score,
                               minimum_count=minimum_count, object_cut_priority=priority)
    for rule_id, cuts in enumerate(cut_lists):
        steps = tuple(RuleStep(depth, cut, RuleStatistics(
            0, 1, 0, 1, 0., 1., 0., 1., depth, minimum_count), -999.)
            for depth, cut in enumerate(cuts, 1))
        model.candidate_rules_.append(ExistentialRule(
            rule_id, objects, steps, minimum_count, event_observables=events))
    model.input_mode_ = "combined" if objects and events else "object_only" if objects else "event_only"
    model.statistics_source_ = "training"
    return model


class ERLSelfPruningTests(unittest.TestCase):
    def test_exhaustive_shortening_removes_multiple_conditions_and_nonprefixes(self):
        specs = tuple(ObservableSpec(name, "ordinal") for name in ["a", "b", "c"])
        model = model_with([(event(0), event(1), event(2))], events=specs)
        values = np.array(list(itertools.product([0, 1], repeat=3)))
        def nonmonotone(stats):
            return {1: .4, 2: .1, 4: .8}[stats.signal_accepted]
        result = model.self_prune_candidates(signal_event_features=values, background_event_features=values,
                                             scoring_function=nonmonotone)
        self.assertEqual(result.self_pruning_provenance_["evaluated_variant_count"], 7)
        self.assertEqual(len(result.candidate_rules_), 3)
        self.assertTrue(all(rule.depth == 1 and rule.generation_score == .8 for rule in result.candidate_rules_))
        self.assertEqual({rule.cuts[0].observable_index for rule in result.candidate_rules_}, {0, 1, 2})
        self.assertIsNone(result.pruning_result_)
        self.assertEqual(result.selected_rules_, [])
        self.assertEqual(result.statistics_source_, "recalculated")

    def test_guard_only_rule_survives_without_false_guard_cuts_or_source_mutation(self):
        specs = (ObservableSpec("eta2", "continuous", validity_column="two_photons"),
                 ObservableSpec("two_photons", "boolean"))
        model = model_with([(event(0, "<=", .5),)], events=specs)
        model.prune_candidates(similarity_filter=False)
        before = model.to_dict()
        inputs = dict(signal_event_features=[[.1, 1], [.2, 1], [.9, 1], [np.nan, 0]],
                      background_event_features=[[.1, 1], [np.nan, 0], [np.nan, 0], [np.nan, 0]])
        result = model.self_prune_candidates(**inputs)
        self.assertEqual(len(result.candidate_rules_), 1)
        guard = result.get_rule(0)
        self.assertEqual(guard.cuts, (EventCut(CategoryCut(1, {1})),))
        np.testing.assert_array_equal(guard.predict(event_features=inputs["signal_event_features"]),
                                      [True, True, True, False])
        self.assertEqual(guard.statistics.signal_accepted, 3)
        self.assertEqual(guard.statistics.background_accepted, 1)
        self.assertEqual(guard.generation_score, .5)
        self.assertFalse(result.allow_validity_cuts)
        self.assertEqual(model.to_dict(), before)
        result.prune_candidates(similarity_filter=True, stricter_score_improvement=0)
        self.assertEqual(result.selected_rules_, [guard])

    def test_guarded_measurements_remain_guarded_after_subsetting(self):
        specs = (ObservableSpec("x", "continuous", validity_column="valid"),
                 ObservableSpec("y", "continuous", validity_column="valid"),
                 ObservableSpec("valid", "boolean"))
        model = model_with([(event(0), event(1))], events=specs)
        inputs = dict(signal_event_features=[[2, 3, 1], [-2, 3, 1], [np.nan, np.nan, 0]],
                      background_event_features=[[2, -3, 1], [np.nan, np.nan, 0]])
        result = model.self_prune_candidates(**inputs, stricter_score_improvement=-100)
        for rule in result.candidate_rules_:
            self.assertFalse(rule.predict(event_features=inputs["signal_event_features"])[-1])
            for cut in rule.cuts:
                if isinstance(cut.cut, CategoryCut):
                    self.assertEqual(cut.cut.accepted_codes, {1})
        self.assertTrue(any(rule.cuts == (event(0),) for rule in result.candidate_rules_))
        self.assertTrue(any(rule.cuts == (EventCut(CategoryCut(2, {1})),) for rule in result.candidate_rules_))

    def test_rate_limit_is_applied_before_simpler_variants_can_prune(self):
        specs = tuple(ObservableSpec(name, "ordinal") for name in ["x", "y"])
        model = model_with([(event(0), event(1))], events=specs)
        inputs = dict(signal_event_features=[[1, 1], [1, 0], [1, 0], [1, 0]],
                      background_event_features=[[1, 0], [0, 1], [1, 0], [0, 1]])
        unrestricted = model.self_prune_candidates(**inputs)
        self.assertFalse(any(rule.depth == 2 for rule in unrestricted.candidate_rules_))
        limited = model.self_prune_candidates(**inputs, max_reference_acceptance=0)
        self.assertEqual(len(limited.candidate_rules_), 1)
        self.assertEqual(limited.get_rule(0).cuts, (event(0), event(1)))
        self.assertEqual(limited.get_rule(0).statistics.background_accepted, 0)

    def test_signed_improvement_and_multiple_alternatives(self):
        specs = (ObservableSpec("x", "continuous", validity_column="valid"),
                 ObservableSpec("valid", "boolean"))
        model = model_with([(event(0, "<=", .5),)], events=specs)
        inputs = dict(signal_event_features=[[.1, 1], [.2, 1], [.9, 1], [0, 0]],
                      background_event_features=[[.1, 1], [0, 0], [0, 0], [0, 0]])
        result = model.self_prune_candidates(**inputs, stricter_score_improvement=-.3)
        self.assertEqual(len(result.candidate_rules_), 2)
        self.assertEqual({r.generation_score for r in result.candidate_rules_}, {.25, .5})

    def test_positive_tolerance_removes_small_gain(self):
        specs = (ObservableSpec("x", "continuous", validity_column="valid"),
                 ObservableSpec("valid", "boolean"))
        model = model_with([(event(0),)], events=specs)
        inputs = dict(signal_event_features=[[1, 1], [1, 1]], background_event_features=[[-1, 1], [1, 1]])
        result = model.self_prune_candidates(**inputs, stricter_score_improvement=.5)
        self.assertEqual([r.cuts for r in result.candidate_rules_], [(EventCut(CategoryCut(1, {1})),)])

    def test_pool_deduplicates_across_parents_and_preserves_parent_ids(self):
        specs = (ObservableSpec("x", "continuous"),)
        model = model_with([(event(), event()), (event(),)], events=specs)
        result = model.self_prune_candidates(signal_event_features=[[1], [2]], background_event_features=[[-1]])
        self.assertEqual(len(result.candidate_rules_), 1)
        self.assertEqual(result.get_rule(0).depth, 1)
        self.assertEqual(result.self_pruning_provenance_["rules"], [{"parent_rule_ids": [0, 1]}])
        self.assertIsNone(result.max_candidate_rules)

    def test_all_candidates_are_used_by_default_and_rule_ids_can_restrict(self):
        specs = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        model = model_with([(event(0),), (event(1),)], events=specs)
        inputs = dict(signal_event_features=[[1, 1]], background_event_features=[[-1, -1]])
        model.recalculate_candidates(**inputs)
        model.prune_candidates(similarity_filter=True)
        self.assertEqual(len(model.selected_rules_), 1)
        all_rules = model.self_prune_candidates(**inputs)
        self.assertEqual(len(all_rules.candidate_rules_), 2)
        subset = model.self_prune_candidates(**inputs, rule_ids=[1])
        self.assertEqual(subset.get_rule(0).cuts, (event(1),))
        self.assertEqual(subset.self_pruning_provenance_["source_rule_ids"], [1])

    def test_object_shortening_preserves_same_slot_minimum_count_and_weights(self):
        specs = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        model = model_with([(ThresholdCut(0, ">", 0), ThresholdCut(1, ">", 0))],
                           objects=specs, minimum_count=2)
        signal = np.array([[[1, -1], [-1, 1], [0, 0]], [[1, 1], [1, 1], [0, 0]]])
        background = np.array([[[1, -1], [1, -1], [0, 0]], [[1, 1], [-1, 1], [0, 0]]])
        result = model.self_prune_candidates(
            signal, background, signal_slot_mask=np.array([[1, 1, 0], [1, 1, 0]], bool),
            background_slot_mask=np.array([[1, 1, 0], [1, 1, 0]], bool),
            signal_weight=[3, 7], background_weight=[2, 5], stricter_score_improvement=-100)
        both = next(r for r in result.candidate_rules_ if r.depth == 2)
        np.testing.assert_array_equal(both.signal_selection, [False, True])
        self.assertEqual(both.minimum_count, 2)
        self.assertEqual(both.statistics.signal_weight_accepted, 7)
        self.assertEqual(both.statistics.signal_weight_total, 10)
        self.assertEqual(both.statistics.background_weight_total, 7)
        for rule in result.candidate_rules_:
            for step in rule.steps:
                mask = rule.predict(signal, depth=step.depth)
                self.assertEqual(step.statistics.signal_weight_accepted, np.array([3, 7])[mask].sum())

    def test_mixed_and_guarded_object_variants_round_trip_and_predict(self):
        objects = (ObservableSpec("pt", "continuous", validity_column="valid"),
                   ObservableSpec("valid", "boolean"))
        events = (ObservableSpec("ht", "continuous"),)
        model = model_with([(ThresholdCut(0, ">", 10), event(0, ">", 20))], objects=objects, events=events)
        inputs = dict(signal=np.array([[[30, 1]], [[np.nan, 0]], [[5, 1]]]),
                      background=np.array([[[5, 1]], [[np.nan, 0]]]),
                      signal_event_features=[[40], [40], [40]], background_event_features=[[10], [40]])
        result = model.self_prune_candidates(**inputs, stricter_score_improvement=-100)
        result.prune_candidates(similarity_filter=False, stricter_score_improvement=-100)
        with tempfile.TemporaryDirectory() as directory:
            path = result.save(Path(directory) / "refined.json")
            loaded = ExistentialRuleList.load(path)
        self.assertEqual(loaded.to_dict(), result.to_dict())
        for rule in loaded.candidate_rules_:
            original = result.get_rule(rule.rule_id)
            np.testing.assert_array_equal(rule.predict(inputs["signal"], event_features=inputs["signal_event_features"]),
                                          original.signal_selection)
        np.testing.assert_array_equal(loaded.predict(inputs["signal"], event_features=inputs["signal_event_features"]),
                                      result.predict(inputs["signal"], event_features=inputs["signal_event_features"]))
        self.assertEqual(loaded.self_pruning_provenance_, result.to_dict()["self_pruning_provenance"])

    def test_merged_source_priorities_and_repeated_cuts_remain_enforceable(self):
        specs = (ObservableSpec("pt", "continuous"),)
        cuts = (ThresholdCut(0, ">", 0), ThresholdCut(0, ">", 1))
        left = model_with([cuts], objects=specs, priority=["pt"])
        right = model_with([cuts], objects=specs, priority=["pt", "pt"])
        merged = ExistentialRuleList.merge([left, right], scoring_function=score)
        result = merged.self_prune_candidates([[[2]], [[.5]]], [[[-1]]], require_complete_priority=True,
                                              stricter_score_improvement=-100)
        self.assertEqual(result.statistics_source_, "recalculated")
        self.assertTrue(any(r.depth == 2 for r in result.candidate_rules_))
        result.prune_candidates(similarity_filter=False, require_complete_priority=True,
                                stricter_score_improvement=-100)
        self.assertFalse(any(d["status"] == "incomplete_object_priority"
                             for d in result.pruning_result_.decisions.values()))
        self.assertEqual(merged.statistics_source_, "merged")

    def test_optional_cache_and_empty_pools(self):
        specs = (ObservableSpec("x", "continuous"),)
        model = model_with([(event(),)], events=specs)
        inputs = dict(signal_event_features=[[1]], background_event_features=[[1]])
        with patch.object(ExistentialRuleList, "compute_jaccard_matrices", side_effect=AssertionError):
            result = model.self_prune_candidates(**inputs, compute_jaccard=False)
        self.assertIsNone(result.signal_jaccard_)
        self.assertEqual(len(result.prune_candidates()), 1)
        empty = model.self_prune_candidates(**inputs, max_reference_acceptance=0)
        self.assertEqual(empty.candidate_rules_, [])
        self.assertEqual(empty.signal_jaccard_.shape, (0, 0))
        empty.prune_candidates()
        self.assertEqual(ExistentialRuleList.from_dict(empty.to_dict()).candidate_rules_, [])
        empty_subset = model.self_prune_candidates(**inputs, rule_ids=[])
        self.assertEqual(empty_subset.candidate_rules_, [])

    def test_no_beam_search_and_fresh_cache_matches_new_masks(self):
        specs = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        model = model_with([(event(0), event(1))], events=specs)
        with patch("set_based_models.rule_list.model.BeamRuleGenerator.generate", side_effect=AssertionError):
            result = model.self_prune_candidates(signal_event_features=[[1, 1], [1, -1]],
                                                 background_event_features=[[-1, 1]],
                                                 stricter_score_improvement=-100)
        size = len(result.candidate_rules_)
        self.assertEqual(result.signal_jaccard_.shape, (size, size))
        for a in result.candidate_rules_:
            for b in result.candidate_rules_:
                union = np.count_nonzero(a.signal_selection | b.signal_selection)
                intersection = np.count_nonzero(a.signal_selection & b.signal_selection)
                self.assertEqual(result.signal_jaccard_[a.rule_id, b.rule_id], intersection / union if union else 1)

    def test_loaded_model_requires_scorer_and_failure_leaves_source_unchanged(self):
        specs = (ObservableSpec("x", "continuous"),)
        model = model_with([(event(),)], events=specs)
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        before = loaded.to_dict()
        inputs = dict(signal_event_features=[[1]], background_event_features=[[-1]])
        with self.assertRaises(NotImplementedError):
            loaded.self_prune_candidates(**inputs)
        with self.assertRaisesRegex(ValueError, "finite scores"):
            loaded.self_prune_candidates(**inputs, scoring_function=lambda stats: np.nan)
        self.assertEqual(loaded.to_dict(), before)
        result = loaded.self_prune_candidates(**inputs, scoring_function=score)
        self.assertEqual(result.get_rule(0).generation_score, 1)
        result.fit(signal_event_features=[[1]], background_event_features=[[-1]])
        self.assertIsNone(result.self_pruning_provenance_)


if __name__ == "__main__":
    unittest.main()
