"""Logical-pruning and synthetic CLI tests; no production data is read."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casual_tests.stale._nested_rule_pruning import (
    merge_equivalent_rules, normalize_event_rule, prune_nested_rules, prune_stricter_rules, rule_implies,
)
from casual_tests.stale.run_ERL_Gini_nested_event_only import _prune_with_rate_limit, main
from set_based_models import (
    CategoryCut, EventCut, ExistentialRule, ObservableSpec, RuleStatistics, RuleStep, SetInput, ThresholdCut,
)

EVENTS = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"),
          ObservableSpec("z", "continuous"), ObservableSpec("n", "ordinal"),
          ObservableSpec("category", "categorical", {0: "a", 1: "b", 2: "c"}))
DUMMY = (ObservableSpec("unused", "continuous"),)


def threshold(index=0, comparison=">", value=0):
    return EventCut(ThresholdCut(index, comparison, value))


def make_rule(rule_id, score, *cuts):
    steps = tuple(RuleStep(depth, cut, RuleStatistics(2, 10, 1, 10, 2., 10., 1., 10., depth, 1), score)
                  for depth, cut in enumerate(cuts, start=1))
    # All rules deliberately have the same training selections. These must
    # never be used to infer logical equivalence or containment.
    return ExistentialRule(rule_id, DUMMY, steps, signal_selection=np.array([True, False]),
                           background_selection=np.array([False, True]), event_observables=EVENTS)


class NestedPruningTests(unittest.TestCase):
    def test_complex_lower_score_is_removed_with_retained_witness(self):
        simple = make_rule(0, .1, threshold())
        complex_rule = make_rule(1, .08, threshold(), threshold(1))
        result = prune_nested_rules([complex_rule, simple], score_gap=.01)
        self.assertEqual([rule.rule_id for rule in result.retained], [0])
        self.assertEqual(result.decisions[1]["status"], "nested_score_gap")
        self.assertEqual(result.decisions[1]["pruned_by_rule_id"], 0)
        self.assertEqual(result.decisions[1]["retained_witness_rule_id"], 0)
        self.assertAlmostEqual(result.decisions[1]["score_gap"], .02)

    def test_higher_scoring_complex_rule_does_not_prune_simple(self):
        simple = make_rule(0, .08, threshold())
        complex_rule = make_rule(1, .1, threshold(), threshold(1))
        result = prune_nested_rules([complex_rule, simple], score_gap=0)
        self.assertEqual([rule.rule_id for rule in result.retained], [1, 0])
        self.assertTrue(all(x["status"] == "retained" for x in result.decisions.values()))

    def test_gap_threshold_is_strict_and_larger_values_prune_less(self):
        rules = [make_rule(0, .1, threshold()), make_rule(1, .09, threshold(), threshold(1))]
        for gap, expected in ((0, 1), (.005, 1), (.01, 2), (.02, 2)):
            with self.subTest(gap=gap):
                self.assertEqual(len(prune_nested_rules(rules, score_gap=gap).retained), expected)

    def test_equal_scores_keep_non_equivalent_nested_rules(self):
        rules = [make_rule(0, .1, threshold()), make_rule(1, .1, threshold(), threshold(1))]
        self.assertEqual(len(prune_nested_rules(rules, score_gap=0).retained), 2)

    def test_negative_gap_prunes_equal_scoring_nested_rules(self):
        rules = [make_rule(0, .1, threshold()), make_rule(1, .1, threshold(), threshold(1))]
        result = prune_nested_rules(rules, score_gap=-.001)
        self.assertEqual([r.rule_id for r in result.retained], [0])
        self.assertEqual(result.decisions[1]["score_gap"], 0)

    def test_negative_gap_requires_minimum_improvement_for_complex_rules(self):
        for improvement, retained in ((.0005, [0]), (.001, [1, 0]), (.002, [1, 0])):
            with self.subTest(improvement=improvement):
                rules = [make_rule(0, .1, threshold()),
                         make_rule(1, .1 + improvement, threshold(), threshold(1))]
                result = prune_nested_rules(rules, score_gap=-.001)
                self.assertEqual([r.rule_id for r in result.retained], retained)
                if retained == [0]:
                    self.assertLess(result.decisions[1]["score_gap"], 0)

    def test_negative_scores_use_absolute_gaps(self):
        rules = [make_rule(0, -.1, threshold()), make_rule(1, -.2, threshold(), threshold(1))]
        self.assertEqual(len(prune_nested_rules(rules, score_gap=.05).retained), 1)

    def test_equivalent_rules_merge_to_shortest_even_with_large_gap(self):
        long = make_rule(0, .1, threshold(value=2), threshold(value=1))
        short = make_rule(1, .1, threshold(value=2))
        result = prune_nested_rules([long, short], score_gap=100)
        self.assertEqual([rule.rule_id for rule in result.retained], [1])
        self.assertEqual(result.decisions[0]["status"], "equivalent_rule")

    def test_equivalence_witness_can_be_gap_pruned_without_audit_cycle(self):
        rules = [make_rule(0, .2, threshold()),
                 make_rule(1, .1, threshold(), threshold(1)),
                 make_rule(2, .1, threshold(), threshold(1), threshold(1))]
        result = prune_nested_rules(rules)
        self.assertEqual([r.rule_id for r in result.retained], [0])
        self.assertEqual(result.decisions[2]["status"], "equivalent_rule")
        self.assertEqual(result.decisions[2]["pruned_by_rule_id"], 1)
        self.assertEqual(result.decisions[2]["retained_witness_rule_id"], 0)

    def test_same_training_masks_are_not_logical_containment(self):
        rules = [make_rule(0, .2, threshold()),
                 make_rule(1, .1, threshold(1), threshold(2))]
        self.assertEqual(len(prune_nested_rules(rules).retained), 2)

    def test_same_depth_tighter_threshold_is_not_more_complicated(self):
        rules = [make_rule(0, .2, threshold()), make_rule(1, .1, threshold(value=2))]
        self.assertEqual(len(prune_nested_rules(rules).retained), 2)

    def test_containment_recognizes_tighter_bounds_and_extra_conditions(self):
        rules = [make_rule(0, .2, threshold(value=0)),
                 make_rule(1, .1, threshold(value=2), threshold(1))]
        self.assertEqual(len(prune_nested_rules(rules).retained), 1)

    def test_continuous_strict_and_inclusive_endpoints_are_not_equivalent(self):
        inclusive = normalize_event_rule(make_rule(0, .1, threshold(comparison=">=")))
        strict = normalize_event_rule(make_rule(1, .1, threshold(comparison=">")))
        self.assertTrue(rule_implies(strict, inclusive))
        self.assertFalse(rule_implies(inclusive, strict))
        upper_open = normalize_event_rule(make_rule(2, .1, threshold(comparison="<")))
        upper_closed = normalize_event_rule(make_rule(3, .1, threshold(comparison="<=")))
        self.assertTrue(rule_implies(upper_open, upper_closed))
        self.assertFalse(rule_implies(upper_closed, upper_open))

    def test_weaker_bounds_are_removed_independently_of_cut_order(self):
        a = make_rule(0, .1, threshold(comparison=">="), threshold(comparison=">"),
                      threshold(comparison="<=", value=2), threshold(comparison="<", value=2))
        b = make_rule(1, .1, *reversed(a.cuts))
        self.assertEqual(normalize_event_rule(a), normalize_event_rule(b))
        normalized = normalize_event_rule(a)
        self.assertEqual(len(normalized), 2)
        self.assertEqual({x.cut.comparison.value for x in normalized}, {">", "<"})

    def test_ordinal_bounds_merge_semantically_equivalent_thresholds(self):
        rules = [make_rule(0, .1, threshold(3, ">", .5)),
                 make_rule(1, .1, threshold(3, ">=", 1))]
        self.assertEqual(len(prune_nested_rules(rules).retained), 1)
        for comparison, value, expected in ((">", -1.5, -1), (">=", -1.5, -1),
                                           ("<", -1.5, -2), ("<=", -1.5, -2)):
            rule = make_rule(0, .1, threshold(3, comparison, value))
            self.assertEqual(normalize_event_rule(rule)[0].cut.threshold, expected)

    def test_categories_use_set_containment_and_intersection(self):
        broad = EventCut(CategoryCut(4, {0, 1}))
        narrow = EventCut(CategoryCut(4, {1}))
        simple = make_rule(0, .2, broad)
        complex_rule = make_rule(1, .1, narrow, threshold())
        self.assertEqual(len(prune_nested_rules([simple, complex_rule]).retained), 1)
        combined = make_rule(2, .1, broad, EventCut(CategoryCut(4, {1, 2})))
        self.assertEqual(normalize_event_rule(combined), (narrow,))

    def test_large_ordinal_endpoints_are_not_rounded_into_false_equivalence(self):
        for value in (float(2**53), 1e100):
            strict = make_rule(0, .1, threshold(3, ">", value))
            inclusive = make_rule(1, .1, threshold(3, ">=", value))
            self.assertEqual(normalize_event_rule(strict), strict.cuts)
            self.assertEqual(len(prune_nested_rules([strict, inclusive]).retained), 2)

    def test_normalization_preserves_evaluation_on_unseen_values(self):
        features = np.array([[x, 0., 0., n, category]
                             for x in (-1., 0., .5, 1., 2.)
                             for n in (-2, -1, 0, 1, 2) for category in (0, 1, 2)])
        data = SetInput.from_arrays(np.zeros((len(features), 1, 1)), DUMMY,
                                    event_features=features, event_observables=EVENTS)
        rules = [make_rule(0, .1, threshold(comparison=">="), threshold(comparison=">")),
                 make_rule(1, .1, threshold(3, ">", -.5), threshold(3, "<=", 1.5)),
                 make_rule(2, .1, EventCut(CategoryCut(4, {0, 1})), EventCut(CategoryCut(4, {1, 2})))]
        for rule in rules:
            original = np.logical_and.reduce([cut.evaluate(data) for cut in rule.cuts])
            normalized = np.logical_and.reduce([cut.evaluate(data) for cut in normalize_event_rule(rule)])
            np.testing.assert_array_equal(original, normalized)

    def test_invalid_parameters_and_slot_rules_are_rejected(self):
        for gap in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                prune_nested_rules([], score_gap=gap)
        with self.assertRaises(ValueError):
            prune_nested_rules([make_rule(0, .1, threshold()), make_rule(0, .2, threshold(1))])
        with self.assertRaises(ValueError):
            prune_nested_rules([make_rule(0, math.nan, threshold())])
        with self.assertRaises(ValueError):
            prune_nested_rules([make_rule(0, .1, ThresholdCut(0, ">", 0))])
        self.assertEqual(prune_nested_rules([]).retained, ())


class EquivalenceMergingTests(unittest.TestCase):
    def test_merging_does_not_discard_non_equivalent_subsets(self):
        rules = [make_rule(0, .2, threshold()),
                 make_rule(1, .1, threshold(value=2)),
                 make_rule(2, -.1, threshold(), threshold(1))]
        result = merge_equivalent_rules(rules)
        self.assertEqual([r.rule_id for r in result.retained], [0, 1, 2])
        self.assertTrue(all(d['status'] == 'retained' for d in result.decisions.values()))

    def test_equivalents_prefer_shortest_original_and_record_witness(self):
        rules = [make_rule(0, .1, threshold(value=2), threshold(value=1)),
                 make_rule(1, .1, threshold(value=2))]
        result = merge_equivalent_rules(rules)
        self.assertEqual([r.rule_id for r in result.retained], [1])
        self.assertEqual(result.decisions[0]['status'], 'equivalent_rule')
        self.assertEqual(result.decisions[0]['retained_witness_rule_id'], 1)
        self.assertEqual(set(result.normalized_cuts), {0, 1})

    def test_equivalence_witness_resolves_after_unified_pruning(self):
        rules = [make_rule(0, .1, threshold()),
                 make_rule(1, .1, threshold(), threshold(1)),
                 make_rule(2, .1, threshold(), threshold(1), threshold(1))]
        result = prune_stricter_rules(merge_equivalent_rules(rules), min_score_improvement=0)
        self.assertEqual([r.rule_id for r in result.retained], [0])
        self.assertEqual(result.decisions[2]['status'], 'equivalent_rule')
        self.assertEqual(result.decisions[2]['pruned_by_rule_id'], 1)
        self.assertEqual(result.decisions[2]['retained_witness_rule_id'], 0)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(merge_equivalent_rules([]).retained, ())
        for rules in ([make_rule(0, .1, threshold()), make_rule(0, .1, threshold(1))],
                      [make_rule(0, math.nan, threshold())]):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                merge_equivalent_rules(rules)
        with self.assertRaises(ValueError):
            normalize_event_rule(make_rule(0, .1, ThresholdCut(0, '>', 0)))


class StricterPruningTests(unittest.TestCase):
    def prune(self, rules, improvement=0):
        return prune_stricter_rules(merge_equivalent_rules(rules),
                                   min_score_improvement=improvement)

    def test_additional_AND_conditions_require_strict_improvement(self):
        for gain, expected in ((-.001, [0]), (0, [0]), (.0005, [0]), (.001, [0]), (.002, [1, 0])):
            with self.subTest(gain=gain):
                rules = [make_rule(0, .1, threshold()),
                         make_rule(1, .1 + gain, threshold(), threshold(1))]
                self.assertEqual([r.rule_id for r in self.prune(rules, improvement=.001).retained], expected)

    def test_removed_intermediate_rule_cannot_suppress_a_compound_rule(self):
        rules = [make_rule(0, 0, threshold()),
                 make_rule(1, .009, threshold(value=1)),
                 make_rule(2, .018, threshold(value=1), threshold(1))]
        result = self.prune(rules, improvement=.01)
        self.assertEqual([r.rule_id for r in result.retained], [2, 0])
        self.assertEqual(result.decisions[1]['pruned_by_rule_id'], 0)

    def test_equal_depth_tighter_lower_score_and_equal_score_are_pruned(self):
        for score in (.09, .1):
            with self.subTest(score=score):
                result = self.prune([make_rule(0, .1, threshold()),
                                     make_rule(1, score, threshold(value=2))])
                self.assertEqual([r.rule_id for r in result.retained], [0])
                self.assertEqual(result.decisions[1]["status"], "stricter_score_improvement")

    def test_looser_rule_precedes_higher_scoring_stricter_rule(self):
        for gain, expected in ((.0005, [0]), (.001, [0]), (.002, [1, 0])):
            with self.subTest(gain=gain):
                rules = [make_rule(1, .1 + gain, threshold(value=2)),
                         make_rule(0, .1, threshold())]
                result = self.prune(rules, improvement=.001)
                self.assertEqual([r.rule_id for r in result.retained], expected)

    def test_strict_open_closed_bounds_and_upper_thresholds(self):
        for loose, tight in ((threshold(comparison=">="), threshold(comparison=">")),
                             (threshold(comparison="<", value=50), threshold(comparison="<", value=40))):
            with self.subTest(loose=loose):
                result = self.prune([make_rule(0, .1, loose), make_rule(1, .09, tight)])
                self.assertEqual([r.rule_id for r in result.retained], [0])

    def test_categories_and_extra_conditions_are_supported(self):
        rules = [make_rule(0, .1, EventCut(CategoryCut(4, {0, 1}))),
                 make_rule(1, .1005, EventCut(CategoryCut(4, {1})), threshold())]
        self.assertEqual([r.rule_id for r in self.prune(rules, improvement=.001).retained], [0])

    def test_same_masks_on_unrelated_columns_do_not_prove_containment(self):
        rules = [make_rule(0, .1, threshold()), make_rule(1, .09, threshold(1))]
        self.assertEqual(len(self.prune(rules).retained), 2)

    def test_comparison_uses_retained_witness_not_removed_predecessor(self):
        rules = [make_rule(0, .1, threshold()), make_rule(1, .1006, threshold(value=1)),
                 make_rule(2, .1012, threshold(value=2))]
        result = self.prune(rules, improvement=.001)
        self.assertEqual([r.rule_id for r in result.retained], [2, 0])
        self.assertEqual(result.decisions[1]["pruned_by_rule_id"], 0)

    def test_prior_nested_audit_resolves_to_final_retained_witness(self):
        rules = [make_rule(0, .1, threshold()), make_rule(1, .1005, threshold(value=1)),
                 make_rule(2, .09, threshold(value=1), threshold(1))]
        baseline = prune_nested_rules(rules, score_gap=0)
        self.assertEqual(baseline.decisions[2]["pruned_by_rule_id"], 1)
        result = prune_stricter_rules(baseline, min_score_improvement=.001)
        self.assertEqual(result.decisions[2]["retained_witness_rule_id"], 0)
        self.assertEqual(baseline.decisions[1]["status"], "retained")

    def test_invalid_thresholds_and_empty_pool(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                self.prune([], improvement=value)
        self.assertEqual(self.prune([]).retained, ())
        self.assertEqual(self.prune([], improvement=-.001).retained, ())


class NestedRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nested_erl_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal = self.root / "anomalous.pt"
        self.background = self.root / "reference.pt"
        for path, signal in ((self.signal, True), (self.background, False)):
            features = torch.zeros((200, 3), dtype=torch.int64)
            features[:160 if signal else 80, 0] = 1
            features[:10, 2] = 1
            labels = torch.ones(200, dtype=torch.int64)
            if signal:
                features[:20, 1] = 1
                labels[:20] = 5
            torch.save({"event_features": features,
                        "event_feature_columns": ["broad", "clean", "refinement"],
                        "event_feature_schema": [{"name": name, "kind": "boolean"}
                                                 for name in ("broad", "clean", "refinement")],
                        "label": labels, "obj": "ignored"}, path)

    def run_report(self, name, extra=()):
        target = self.root / name
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(["--signal", str(self.signal), "--background", str(self.background),
                  "--report", str(target), "--print-top", "1", *extra])
        return json.loads(target.read_text()), stdout.getvalue(), target

    def test_cli_prunes_completed_nested_rules_and_saves_audit(self):
        report, output, _ = self.run_report("nested.txt")
        config = report["configuration"]
        self.assertEqual(config["reference_penalty"], 1.)
        self.assertNotIn("nested_score_gap", config)
        self.assertEqual(config["stricter_score_improvement"], 0.)
        self.assertFalse(config["beam_similarity_filter"])
        self.assertFalse(config["final_similarity_pruning"])
        self.assertEqual(config["pruning_stage"], "after_candidate_generation")
        self.assertEqual(config["pruning_implementation"], "ExistentialRuleList.prune_candidates")
        self.assertEqual(report["feature_mode"], "event_only")
        self.assertFalse(report["native_event_info_used"])
        self.assertNotIn("nested_rules_removed", report["summary"])
        self.assertGreater(report["summary"]["stricter_rules_removed"], 0)
        self.assertEqual(config["beam_width"], 100)
        self.assertEqual(config["max_candidate_rules"], 1000)
        records = {record["rule_id"]: record for record in report["results"]["rules"]}
        self.assertEqual(len(records), report["summary"]["candidate_rule_count"])
        for record in records.values():
            self.assertEqual(record["retained_after_nested_pruning"], record["retained_after_similarity_pruning"])
            self.assertAlmostEqual(record["score"], record["gini_gain"] - record["reference_penalty_term"])
            if record["pruning"]["status"] == "stricter_score_improvement":
                witness = records[record["pruning"]["pruned_by_rule_id"]]
                self.assertTrue(witness["retained_after_nested_pruning"])
                self.assertLessEqual(record["score"] - witness["score"], 1e-12)
        self.assertIn("PRUNING AUDIT", output)
        self.assertIn("reason=stricter_score_improvement", output)
        self.assertNotIn("NESTED PRUNING", output)

    def test_runner_calls_built_in_pruning_directly(self):
        from set_based_models import ExistentialRuleList
        calls = []
        original = ExistentialRuleList.prune_candidates

        def tracked(model, **kwargs):
            calls.append((model, kwargs))
            return original(model, **kwargs)

        with patch.object(ExistentialRuleList, "prune_candidates", tracked), patch(
                "casual_tests.stale.run_ERL_Gini_nested_event_only._prune_with_rate_limit",
                side_effect=AssertionError("Runner must call the model, not the compatibility wrapper")):
            report, _, _ = self.run_report("built_in.txt", ["--stricter-score-improvement", "0.01",
                                                           "--max-trigger-rate-khz", "40"])
        self.assertEqual(len(calls), 1)
        model, kwargs = calls[0]
        self.assertEqual(model.input_mode_, "event_only")
        self.assertEqual(model.observables, ())
        self.assertTrue(report["native_event_only"])
        self.assertIsNone(report["dummy_object_input"])
        self.assertEqual(kwargs, {"stricter_score_improvement": .01,
                                  "max_reference_acceptance": .001, "similarity_filter": False})
        self.assertEqual(report["summary"]["pruned_rule_ids"],
                         [rule.rule_id for rule in model.selected_rules_])
        self.assertEqual(len(model.pruning_result_.decisions), len(model.candidate_rules_))

    def test_improvement_changes_only_final_pruning_not_search_or_scores(self):
        small, _, _ = self.run_report("small.txt", ["--stricter-score-improvement", "0"])
        large, _, _ = self.run_report("large.txt", ["--stricter-score-improvement", "10"])
        def candidates(report):
            return [(r["rule_id"], r["raw_rule"], r["score"], r["gini_gain"])
                    for r in report["results"]["rules"]]
        self.assertEqual(candidates(small), candidates(large))
        self.assertEqual(large["configuration"]["stricter_score_improvement"], 10.)
        self.assertGreater(small["summary"]["stricter_rules_removed"], 0)

    def test_rate_limit_changes_only_final_retention_and_preserves_plot_alias(self):
        baseline, _, _ = self.run_report("no_rate.txt", ["--stricter-score-improvement", "0.01"])
        limited, output, target = self.run_report("rate.txt", ["--stricter-score-improvement", "0.01",
                                                               "--max-trigger-rate-khz", "40"])
        config = limited["configuration"]
        self.assertEqual(config["max_reference_acceptance"], .001)
        self.assertEqual(config["input_rate_mhz"], 40.)
        self.assertEqual(config["rate_filter_stage"], "before_equivalence_merging_and_stricter_pruning")
        self.assertIsNone(baseline["configuration"]["max_trigger_rate_khz"])
        self.assertEqual([(r["raw_rule"], r["score"]) for r in baseline["results"]["rules"]],
                         [(r["raw_rule"], r["score"]) for r in limited["results"]["rules"]])
        self.assertGreater(limited["summary"]["rate_rules_removed"], 0)
        self.assertIn("RATE FILTERING AND UNIFIED STRICTER-RULE PRUNING", output)
        self.assertIn("reason=reference_rate_limit", output)
        for record in limited["results"]["rules"]:
            self.assertAlmostEqual(record["estimated_reference_rate_khz"],
                                   40000 * record["metrics"]["background_acceptance"])
            self.assertEqual(record["retained_after_nested_pruning"], record["retained_after_similarity_pruning"])
            if not record["passes_reference_rate_limit"]:
                self.assertEqual(record["pruning"]["status"], "reference_rate_limit")
                self.assertIsNone(record["pruned_rank"])
        from scripts.plot_acceptance_report import load_report, select_paths
        _, selected = select_paths(load_report(target), rules="pruned", direction="mixture", top=10)
        self.assertTrue(selected)
        self.assertTrue(all(r["estimated_reference_rate_khz"] <= 40 for r in selected))

    def test_custom_input_rate_and_zero_budget(self):
        custom, _, _ = self.run_report("custom_rate.txt", ["--max-trigger-rate-khz", "40",
                                                          "--input-rate-mhz", "20"])
        self.assertEqual(custom["configuration"]["max_reference_acceptance"], .002)
        zero, _, _ = self.run_report("zero_rate.txt", ["--max-trigger-rate-khz", "0"])
        retained = [r for r in zero["results"]["rules"] if r["retained_after_nested_pruning"]]
        self.assertTrue(retained)
        self.assertTrue(all(r["estimated_reference_rate_khz"] == 0 for r in retained))

    def test_stricter_cli_filter_keeps_search_scores_and_plot_compatibility(self):
        # Continuous values allow multiple one-cut thresholds on the same column.
        for path, count in ((self.signal, 200), (self.background, 100)):
            features = torch.zeros((200, 1), dtype=torch.float64)
            features[:count, 0] = torch.arange(1, count + 1, dtype=torch.float64)
            torch.save({"event_features": features, "event_feature_columns": ["pt_sum"],
                        "event_feature_schema": [{"name": "pt_sum", "kind": "continuous"}],
                        "label": torch.ones(200, dtype=torch.int64)}, path)
        baseline, _, _ = self.run_report("stricter_default.txt")
        filtered, output, target = self.run_report("stricter_explicit.txt", ["--stricter-score-improvement", "0.001"])
        self.assertEqual(baseline["configuration"]["stricter_score_improvement"], 0.)
        self.assertEqual(filtered["configuration"]["stricter_score_improvement"], .001)
        self.assertEqual([(r["raw_rule"], r["score"]) for r in baseline["results"]["rules"]],
                         [(r["raw_rule"], r["score"]) for r in filtered["results"]["rules"]])
        self.assertGreater(filtered["summary"]["stricter_rules_removed"], 0)
        self.assertIn("STRICTER FILTER", output)
        records = {r["rule_id"]: r for r in filtered["results"]["rules"]}
        for r in records.values():
            self.assertEqual(r["retained_after_final_pruning"], r["retained_after_similarity_pruning"])
            if r["pruning"]["status"] == "stricter_score_improvement":
                witness = records[r["pruning"]["pruned_by_rule_id"]]
                self.assertTrue(witness["retained_after_final_pruning"])
                self.assertLessEqual(r["pruning"]["score_improvement"], .001 + 1e-12)
        from scripts.plot_acceptance_report import load_report, select_paths
        _, selected = select_paths(load_report(target), rules="pruned", direction="mixture", top=10)
        self.assertTrue(all(r["retained_after_final_pruning"] for r in selected))

    def test_invalid_stricter_cli_thresholds_do_not_create_reports(self):
        for value in ("nan", "inf", "oops"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_report("invalid_stricter.txt", ["--stricter-score-improvement", value])
            self.assertFalse((self.root / "invalid_stricter.txt").exists())

    def test_negative_stricter_cli_threshold_is_preserved(self):
        report, _, _ = self.run_report("negative_improvement.txt", ["--stricter-score-improvement", "-0.008"])
        self.assertEqual(report["configuration"]["stricter_score_improvement"], -.008)

    def test_invalid_rate_parameters_do_not_create_reports(self):
        for flag, values in (("--max-trigger-rate-khz", ("-1", "nan", "inf", "oops")),
                             ("--input-rate-mhz", ("0", "-1", "nan", "inf", "oops", "1e308"))):
            for value in values:
                with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.run_report("invalid_rate.txt", [flag, value])
                self.assertFalse((self.root / "invalid_rate.txt").exists())

    def test_pure_gini_and_beam_opt_in_are_recorded(self):
        report, _, _ = self.run_report("pure.txt", ["--reference-penalty", "0", "--beam-similarity-filter",
                                                   "--signal-similarity-threshold", "0.95"])
        self.assertTrue(report["configuration"]["beam_similarity_filter"])
        self.assertEqual(report["configuration"]["beam_signal_similarity_threshold"], .95)
        self.assertTrue(all(r["score"] == r["gini_gain"] for r in report["results"]["rules"]))

    def test_removed_gap_option_is_rejected_and_absent_from_help(self):
        for option in (["--nested-score-gap", "0"], ["--nested-score-gap=-1e-4"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.run_report("removed.txt", option)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((self.root / "removed.txt").exists())
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertNotIn("--nested-score-gap", output.getvalue())
        self.assertIn("--stricter-score-improvement", output.getvalue())

    def test_scientific_improvement_threshold_is_saved(self):
        report, _, _ = self.run_report("scientific.txt", ["--stricter-score-improvement=1e-4"])
        self.assertEqual(report["configuration"]["stricter_score_improvement"], .0001)

    def test_reports_and_input_aliases_are_protected(self):
        self.run_report("existing.txt")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_report("existing.txt")
        self.run_report("existing.txt", ["--overwrite-report"])
        alias = self.root / "alias.pt"
        os.link(self.signal, alias)
        before = self.signal.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_report(str(alias), ["--overwrite-report"])
        self.assertEqual(self.signal.read_bytes(), before)

    def test_existing_plotter_uses_nested_retention_and_score_order(self):
        from scripts.plot_acceptance_report import load_report, make_plot, select_paths
        import matplotlib.pyplot as plt
        _, _, target = self.run_report("plot.txt")
        report = load_report(target)
        eligible, selected = select_paths(report, rules="pruned", direction="mixture", top=10)
        self.assertTrue(selected)
        self.assertTrue(all(r["retained_after_nested_pruning"] for r in eligible))
        self.assertEqual([r["score"] for r in eligible], sorted((r["score"] for r in eligible), reverse=True))
        figure = make_plot(report, top=1)
        try:
            figure.savefig(self.root / "plot.png")
        finally:
            plt.close(figure)
        self.assertTrue((self.root / "plot.png").read_bytes().startswith(b"\x89PNG"))

    def test_constant_ordinal_inputs_and_external_script_invocation(self):
        for path in (self.signal, self.background):
            raw = torch.load(path, weights_only=True)
            raw["event_features"][:] = 0
            for spec in raw["event_feature_schema"]:
                spec["kind"] = "ordinal"
            torch.save(raw, path)
        target = self.root / "external.txt"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "casual_tests/stale/run_ERL_Gini_nested_event_only.py"),
                                 "--signal", str(self.signal), "--background", str(self.background),
                                 "--report", str(target)], cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(target.read_text())["results"]["rules"], [])


class RatePruningTests(unittest.TestCase):
    @staticmethod
    def rule(rule_id, score, accepted, *cuts):
        rule = make_rule(rule_id, score, *cuts)
        last = rule.steps[-1]
        statistics = replace(last.statistics, background_accepted=accepted,
                             background_weight_accepted=float(accepted))
        return replace(rule, steps=(*rule.steps[:-1], replace(last, statistics=statistics)))

    def test_over_budget_simple_rule_cannot_prune_eligible_complex_rule(self):
        simple = self.rule(0, .2, 2, threshold())
        complex_rule = self.rule(1, .1, 1, threshold(), threshold(1))
        result = _prune_with_rate_limit([simple, complex_rule], min_score_improvement=.01,
                                       max_reference_acceptance=.1)
        self.assertEqual([r.rule_id for r in result.retained], [1])
        self.assertEqual(result.decisions[0]["status"], "reference_rate_limit")
        self.assertEqual(result.decisions[1]["status"], "retained")
        self.assertEqual(set(result.normalized_cuts), {0, 1})

    def test_rate_rejected_looser_rule_cannot_prune_stricter_rule(self):
        rules = [self.rule(0, .2, 2, threshold()), self.rule(1, .1, 1, threshold(value=1))]
        result = _prune_with_rate_limit(rules, min_score_improvement=.001, max_reference_acceptance=.1)
        self.assertEqual([r.rule_id for r in result.retained], [1])
        self.assertEqual(result.decisions[0]["status"], "reference_rate_limit")

    def test_eligible_simple_rule_still_prunes_eligible_complex_rule(self):
        rules = [self.rule(0, .2, 1, threshold()),
                 self.rule(1, .1, 0, threshold(), threshold(1))]
        result = _prune_with_rate_limit(rules, min_score_improvement=.01, max_reference_acceptance=.1)
        self.assertEqual([r.rule_id for r in result.retained], [0])
        self.assertEqual(result.decisions[1]["status"], "stricter_score_improvement")

    def test_acceptance_boundary_is_inclusive(self):
        rules = [self.rule(0, .1, 1, threshold())]
        for limit, expected in ((.099, 0), (.1, 1), (.101, 1)):
            with self.subTest(limit=limit):
                result = _prune_with_rate_limit(rules, max_reference_acceptance=limit)
                self.assertEqual(len(result.retained), expected)

    def test_all_rejected_and_empty_candidate_pools_are_supported(self):
        rules = [self.rule(0, .1, 1, threshold())]
        result = _prune_with_rate_limit(rules, max_reference_acceptance=0)
        self.assertEqual(result.retained, ())
        self.assertEqual(result.decisions[0]["status"], "reference_rate_limit")
        self.assertEqual(_prune_with_rate_limit([], max_reference_acceptance=0).retained, ())

    def test_equivalence_representative_is_chosen_only_from_rate_eligible_rules(self):
        short = self.rule(0, .2, 2, threshold(value=1))
        long = self.rule(1, .1, 1, threshold(), threshold(value=1))
        result = _prune_with_rate_limit([short, long], max_reference_acceptance=.1)
        self.assertEqual([r.rule_id for r in result.retained], [1])
        self.assertEqual(result.decisions[0]['status'], 'reference_rate_limit')

    def test_unified_pipeline_has_no_hidden_complexity_gap(self):
        rules = [self.rule(0, 0, 0, threshold()),
                 self.rule(1, .009, 0, threshold(value=1)),
                 self.rule(2, .018, 0, threshold(value=1), threshold(1))]
        result = _prune_with_rate_limit(rules, min_score_improvement=.01)
        self.assertEqual([r.rule_id for r in result.retained], [2, 0])


if __name__ == "__main__":
    unittest.main()
