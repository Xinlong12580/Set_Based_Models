"""Synthetic checks for the new runner; no production tensors or EOS access."""

from contextlib import redirect_stderr, redirect_stdout
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

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casual_tests import _quantitative_reporting as reporting
from casual_tests.stale.run_ERL_Gini_penalized_event_only import ReferencePenalizedGini, main
from set_based_models import RuleStatistics


def statistics(signal=160, background=80, *, depth=1, signal_total=200, background_total=200):
    return RuleStatistics(signal, signal_total, background, background_total,
                          float(signal), float(signal_total), float(background), float(background_total),
                          depth, 1)


class PenalizedGiniScoreTests(unittest.TestCase):
    def test_exact_penalty_uses_fraction_not_count_or_percent(self):
        stats = statistics()
        gain = reporting.gini_rule_score(stats)
        self.assertAlmostEqual(ReferencePenalizedGini()(stats), gain - 0.4)
        self.assertAlmostEqual(ReferencePenalizedGini(2.5)(stats), gain - 1.0)

    def test_zero_penalty_recovers_raw_gini(self):
        for stats in (statistics(), statistics(20, 0), statistics(0, 20)):
            self.assertEqual(ReferencePenalizedGini(0)(stats), reporting.gini_rule_score(stats))

    def test_larger_penalty_lowers_score_only_if_reference_passes(self):
        self.assertLess(ReferencePenalizedGini(2)(statistics()), ReferencePenalizedGini(1)(statistics()))
        self.assertEqual(ReferencePenalizedGini(2)(statistics(20, 0)),
                         ReferencePenalizedGini(1)(statistics(20, 0)))

    def test_full_reference_denominator_and_sample_weights_are_used(self):
        stats = RuleStatistics(2, 10, 3, 20, 5., 30., 2., 50., 3, 1)
        self.assertAlmostEqual(ReferencePenalizedGini()(stats), reporting.gini_rule_score(stats) - 2 / 50)
        self.assertEqual(ReferencePenalizedGini()(statistics(depth=3)),
                         ReferencePenalizedGini()(statistics(depth=1)))

    def test_negative_scores_are_not_clipped_or_rejected(self):
        self.assertLess(ReferencePenalizedGini()(statistics()), 0)
        self.assertTrue(math.isfinite(ReferencePenalizedGini()(statistics())))

    def test_invalid_penalties_are_rejected(self):
        for value in (-1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReferencePenalizedGini(value)


class PenalizedERLRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="penalized_erl_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal = self.root / "anomalous.pt"
        self.background = self.root / "reference.pt"
        for path, signal in ((self.signal, True), (self.background, False)):
            features = torch.zeros((200, 2), dtype=torch.int64)
            features[:160 if signal else 80, 0] = 1
            labels = torch.ones(200, dtype=torch.int64)
            if signal:
                features[:20, 1] = 1
                labels[:20] = 5
            torch.save({
                "event_features": features,
                "event_feature_columns": ["broad_signature", "clean_signature"],
                "event_feature_schema": [{"name": name, "kind": "boolean"}
                                         for name in ("broad_signature", "clean_signature")],
                "label": labels,
                "obj": "ignored", "pf": "ignored",
                "native_event_info": torch.full((200, 10), float("nan")),
            }, path)

    def run_report(self, name, extra=(), runner=main):
        target = self.root / name
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            runner(["--signal", str(self.signal), "--background", str(self.background),
                    "--report", str(target), "--print-top", "1", *extra])
        return json.loads(target.read_text()), stdout.getvalue(), target

    def test_default_penalty_changes_discovery_to_clean_signature(self):
        pure, _, _ = self.run_report("pure.txt", ["--reference-penalty", "0"])
        penalized, stdout, _ = self.run_report("penalized.txt")
        first_pure = next(x for x in pure["results"]["rules"] if x["direction"] == "mixture-enriched")
        first_penalized = next(x for x in penalized["results"]["rules"] if x["direction"] == "mixture-enriched")
        self.assertEqual((first_pure["metrics"]["signal_accepted"],
                          first_pure["metrics"]["background_accepted"]), (160, 80))
        self.assertEqual((first_penalized["metrics"]["signal_accepted"],
                          first_penalized["metrics"]["background_accepted"]), (20, 0))
        self.assertEqual(penalized["configuration"]["reference_penalty"], 1.)
        self.assertIn("score=", stdout)
        self.assertIn("reference_penalty_term=", stdout)
        self.assertTrue(penalized["configuration"]["penalty_applied_during_search"])
        self.assertEqual(penalized["feature_mode"], "event_only")
        self.assertFalse(penalized["native_event_info_used"])
        self.assertEqual(penalized["object_observables"], [])

    def test_report_scores_prefixes_and_ranks_are_truthful(self):
        report, _, _ = self.run_report("scores.txt", ["--reference-penalty", "2.5"])
        rules = report["results"]["rules"]
        self.assertTrue(rules)
        self.assertEqual([r["score"] for r in rules], sorted((r["score"] for r in rules), reverse=True))
        self.assertEqual([r["rank"] for r in rules], list(range(1, len(rules) + 1)))
        for rule in rules:
            self.assertAlmostEqual(rule["reference_penalty_term"],
                                   2.5 * rule["metrics"]["background_acceptance"])
            self.assertAlmostEqual(rule["score"], rule["gini_gain"] - rule["reference_penalty_term"])
            for step in rule["steps"]:
                self.assertAlmostEqual(step["prefix_score"],
                                       step["prefix_gini_gain"] - step["prefix_reference_penalty_term"])
                self.assertAlmostEqual(step["prefix_reference_penalty_term"],
                                       2.5 * step["prefix_reference_acceptance"])
                self.assertEqual(step["scope"], "event")
                self.assertEqual(step["cut"]["scope"], "event")

    def test_zero_penalty_matches_original_runner_candidates_and_pruning(self):
        from casual_tests.stale.run_ERL_Gini_event_only import main as original
        before = (reporting.SIGNAL_SIMILARITY_THRESHOLD, reporting.BACKGROUND_SIMILARITY_THRESHOLD,
                  reporting.gini_rule_score)
        old, _, _ = self.run_report("old.txt", runner=original)
        new, _, _ = self.run_report("new.txt", ["--reference-penalty", "0"])
        fields = ("rank", "rule_id", "raw_rule", "simplified_rule", "gini_gain", "metrics", "pruned_rank")
        self.assertEqual([{k: r[k] for k in fields} for r in old["results"]["rules"]],
                         [{k: r[k] for k in fields} for r in new["results"]["rules"]])
        self.assertEqual(new["summary"], old["summary"])
        self.assertEqual(before, (reporting.SIGNAL_SIMILARITY_THRESHOLD,
                                  reporting.BACKGROUND_SIMILARITY_THRESHOLD, reporting.gini_rule_score))

    def test_similarity_options_reach_the_model_and_report(self):
        report, _, _ = self.run_report("similarity.txt", [
            "--signal-similarity-threshold", "0.95", "--background-similarity-threshold", "0.9",
            "--no-beam-similarity-filter"])
        config = report["configuration"]
        self.assertFalse(config["beam_similarity_filter"])
        for stage in ("beam", "prune"):
            self.assertEqual(config[f"{stage}_signal_similarity_threshold"], .95)
            self.assertEqual(config[f"{stage}_background_similarity_threshold"], .9)

    def test_process_labels_do_not_change_training(self):
        original, _, _ = self.run_report("labels.txt")
        raw = torch.load(self.signal, weights_only=True)
        raw["label"][:] = 99
        torch.save(raw, self.signal)
        relabeled, _, _ = self.run_report("relabeled.txt")
        self.assertEqual([(r["raw_rule"], r["score"]) for r in original["results"]["rules"]],
                         [(r["raw_rule"], r["score"]) for r in relabeled["results"]["rules"]])

    def test_invalid_cli_values_are_rejected_without_writing(self):
        for option, value in (("--reference-penalty", "-1"), ("--reference-penalty", "nan"),
                              ("--reference-penalty", "inf"), ("--reference-penalty", "bad"),
                              ("--signal-similarity-threshold", "1.01"),
                              ("--background-similarity-threshold", "nan")):
            with self.subTest(option=option, value=value):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    self.run_report("invalid.txt", [option, value])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse((self.root / "invalid.txt").exists())

    def test_report_overwrite_is_explicit_and_input_aliases_are_protected(self):
        _, _, target = self.run_report("protected.txt")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_report("protected.txt")
        self.run_report("protected.txt", ["--overwrite-report"])
        alias = self.root / "alias.pt"
        os.link(self.signal, alias)
        before = self.signal.read_bytes()
        for output in (self.signal, alias):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_report(str(output), ["--overwrite-report"])
        self.assertEqual(self.signal.read_bytes(), before)

    def test_constant_inputs_produce_empty_report_not_dummy_cuts(self):
        for path in (self.signal, self.background):
            raw = torch.load(path, weights_only=True)
            raw["event_features"][:] = 0
            for feature in raw["event_feature_schema"]:
                feature["kind"] = "ordinal"
            torch.save(raw, path)
        report, stdout, _ = self.run_report("constant.txt")
        self.assertEqual(report["results"]["rules"], [])
        self.assertEqual(report["summary"]["candidate_rule_count"], 0)
        self.assertIn("(none)", stdout)

    def test_default_report_name_is_distinct(self):
        target = self.root / "reports" / f"ERL_Gini_penalized_{self.root.name}_anomalous_event_only.txt"
        with patch("casual_tests._event_only_reporting.Path.cwd", return_value=self.root), \
                redirect_stdout(io.StringIO()):
            main(["--signal", str(self.signal), "--background", str(self.background)])
        self.assertTrue(target.is_file())

    def test_existing_plotter_loads_and_plots_new_report(self):
        from scripts.plot_acceptance_report import load_report, make_plot, select_paths
        import matplotlib.pyplot as plt
        _, _, target = self.run_report("plot.txt")
        report = load_report(target)
        eligible, selected = select_paths(report, rules="pruned", direction="mixture", top=10)
        self.assertTrue(selected)
        self.assertEqual(eligible, sorted(eligible, key=lambda r: r["score"], reverse=True))
        figure = make_plot(report, top=1)
        try:
            figure.savefig(self.root / "plot.png")
        finally:
            plt.close(figure)
        self.assertTrue((self.root / "plot.png").read_bytes().startswith(b"\x89PNG"))

    def test_script_runs_from_outside_repository(self):
        target = self.root / "subprocess.txt"
        result = subprocess.run([
            sys.executable, "-B", str(ROOT / "casual_tests" / "stale" / "run_ERL_Gini_penalized_event_only.py"),
            "--signal", str(self.signal), "--background", str(self.background),
            "--report", str(target), "--reference-penalty", "2", "--print-top", "1",
        ], cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(target.read_text())["configuration"]["reference_penalty"], 2.)


if __name__ == "__main__":
    unittest.main()
