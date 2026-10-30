"""External acceptance scores and their training/validation/report integration."""

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from casual_tests import scan_ERL, train_ERL, train_ERL_validation, validate_ERL
from casual_tests._erl_common import ReferencePenalizedGini
from casual_tests.erl_scores import AcceptanceRatioRuleScore, ExcessRuleScore, make_score
from scripts.plot_acceptance_report import load_report, make_plot
from set_based_models import ExistentialRuleList, RuleStatistics
from tests.test_erl_trainer import sample


def statistics(a=.4, b=.1):
    return RuleStatistics(1, 4, 1, 10, 10 * a, 10., 50 * b, 50., 1, 1)


class AcceptanceScoreTests(unittest.TestCase):
    def test_weighted_acceptance_and_exact_formulas(self):
        stats = statistics()
        self.assertAlmostEqual(ExcessRuleScore(.02)(stats), .3 / math.sqrt(.12))
        self.assertAlmostEqual(AcceptanceRatioRuleScore(.02)(stats), .4 / math.sqrt(.12))
        for scorer in (ExcessRuleScore(.02), AcceptanceRatioRuleScore(.02)):
            self.assertEqual(scorer(stats), scorer(replace(stats, rule_depth=9)))

    def test_excess_clips_background_enrichment_and_both_handle_zero_passes(self):
        for a, b in ((.1, .4), (.4, .4), (0., 0.), (.2, 0.)):
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(ExcessRuleScore(.01)(statistics(a, b)), max(0., a-b) / math.sqrt(b+.01))
                self.assertAlmostEqual(AcceptanceRatioRuleScore(.01)(statistics(a, b)), a / math.sqrt(b+.01))
        self.assertEqual(ExcessRuleScore()(statistics(.1, .4)), 0.)
        self.assertGreater(AcceptanceRatioRuleScore()(statistics(.1, .4)), 0.)

    def test_larger_delta_reduces_sensitivity_to_one_reference_event(self):
        zero, one = statistics(.1, 0.), statistics(.1, 1/32000)
        for kind in (ExcessRuleScore, AcceptanceRatioRuleScore):
            small, large = kind(1e-6), kind(.001)
            self.assertLess(1-large(one)/large(zero), 1-small(one)/small(zero))
            for delta in (0., -1., math.nan, math.inf):
                with self.subTest(kind=kind, delta=delta), self.assertRaises(ValueError):
                    kind(delta)

    def test_factory_preserves_gini_and_does_not_apply_its_penalty_to_alternatives(self):
        stats = statistics()
        for penalty in (0., 1., 10.):
            self.assertEqual(make_score("gini", reference_penalty=penalty)(stats), ReferencePenalizedGini(penalty)(stats))
            self.assertEqual(make_score("excess", reference_penalty=penalty, delta=.01)(stats), ExcessRuleScore(.01)(stats))
            self.assertEqual(make_score("ratio", reference_penalty=penalty, delta=.01)(stats), AcceptanceRatioRuleScore(.01)(stats))


class ExternalScoreRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="erl_scores_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, signal in (("anomalous", True), ("reference", False)):
            torch.save(sample(signal), self.root / f"{name}_train.pt")
            data = sample(signal)
            data["event_features"] = torch.tensor([[2], [2], [2], [0]] if signal else [[2], [0], [0], [0]])
            torch.save(data, self.root / f"{name}_validation.pt")
        self.prefix = self.root / "outputs" / "example"

    def arguments(self, *extra):
        return ["--signal", str(self.root / "anomalous_train.pt"), "--background", str(self.root / "reference_train.pt"),
                "--output-prefix", str(self.prefix), "--max-rule-depth", "2", "--beam-width", "4",
                "--max-candidate-rules", "8", "--max-thresholds", "3", "--print-top", "1",
                "--print-reference-top", "0", "--print-pruned-top", "1", *extra]

    def assert_scores(self, path, name, delta):
        model = ExistentialRuleList.load(path)
        self.assertTrue(model.candidate_rules_)
        for rule in model.candidate_rules_:
            for step in rule.steps:
                a, b = step.statistics.true_positive_rate, step.statistics.false_positive_rate
                expected = (max(0., a-b) if name == "excess" else a) / math.sqrt(b+delta)
                self.assertAlmostEqual(step.generation_score, expected)
        return model

    def test_alternatives_train_in_every_input_mode_and_report_real_objective(self):
        for name in ("excess", "ratio"):
            for mode, flags in (("event", ["--event-only"]), ("object", ["--object-only"]),
                                ("combined", []), ("pair", ["--pair-tensor", "--object-only"])):
                with self.subTest(score=name, mode=mode), redirect_stdout(io.StringIO()) as output:
                    self.prefix = self.root / name / mode
                    model_path, report_path = train_ERL.main(self.arguments(*flags, "--score", name, "--delta", ".01",
                                                                           "--reference-penalty", "99"))
                model = self.assert_scores(model_path, name, .01)
                self.assertIsNone(model.pruning_result_)
                report = load_report(report_path)
                self.assertEqual(report["configuration"]["score_name"], name)
                self.assertEqual(report["configuration"]["score_delta"], .01)
                self.assertEqual(report["configuration"]["score_parameters"], {"delta": .01})
                self.assertIsNone(report["configuration"]["reference_penalty"])
                self.assertEqual(report["configuration"]["gini_gain_role"], "diagnostic_only")
                self.assertIn("gini_gain_diagnostic=", output.getvalue())
                self.assertNotIn("BEAM SCORE: Gini", output.getvalue())
                for record in report["results"]["rules"]:
                    self.assertIsNone(record["reference_penalty_term"])
                    self.assertEqual(record["score"], model.get_rule(record["rule_id"]).generation_score)
                    for step in record["steps"]:
                        self.assertIsNone(step["prefix_reference_penalty_term"])

    def test_scalar_wrapper_inherits_score_and_allows_independent_validation_choice(self):
        for override in (False, True):
            self.prefix = self.root / str(override) / "single"
            extra = ["--pruning-score", "ratio", "--pruning-score-delta", ".04"] if override else []
            with redirect_stdout(io.StringIO()):
                paths = train_ERL_validation.main(self.arguments("--event-only", "--beam-score", "excess",
                                                               "--beam-score-delta", ".02", *extra))
            self.assert_scores(f"{self.prefix}.training.model.json", "excess", .02)
            self.assert_scores(paths[0], "ratio" if override else "excess", .04 if override else .02)
            config = load_report(paths[1])["configuration"]
            self.assertEqual(config["beam_score_name"], "excess")
            self.assertEqual(config["beam_score_delta"], .02)

    def test_scan_sweeps_scores_and_deltas_then_uses_one_common_validation_score(self):
        with redirect_stdout(io.StringIO()):
            paths = scan_ERL.main(self.arguments("--event-only", "--score", "excess", "ratio",
                "--score-delta", ".01", ".02", "--validation-score", "excess", "--validation-score-delta", ".04"))
        scan = json.loads(Path(f"{self.prefix}.scan.json").read_text())
        self.assertEqual(len(scan["runs"]), 4)
        self.assertEqual({(r["search_parameters"]["score"], r["search_parameters"]["score_delta"]) for r in scan["runs"]},
                         {("excess", .01), ("excess", .02), ("ratio", .01), ("ratio", .02)})
        for run in scan["runs"]:
            p = run["search_parameters"]
            self.assert_scores(run["model"], p["score"], p["score_delta"])
        self.assert_scores(paths[0], "excess", .04)
        report = load_report(paths[1])
        self.assertEqual(report["source_scan_runs"], scan["runs"])
        self.assertEqual(report["configuration"]["score_delta"], .04)
        import matplotlib.pyplot as plt
        figure = make_plot(report, top=1)
        plt.close(figure)
        if importlib.util.find_spec("jsonschema"):
            import jsonschema
            schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/quantitative_report.schema.json").read_text())
            jsonschema.validate(report, schema)

    def test_standalone_and_validate_only_use_new_scores_for_self_pruning_without_training(self):
        with redirect_stdout(io.StringIO()):
            manifest = scan_ERL.main(self.arguments("--event-only", "--train-only"))
        runs = json.loads(manifest.read_text())["runs"]
        originals = {run["model"]: Path(run["model"]).read_bytes() for run in runs}
        with redirect_stdout(io.StringIO()), patch.object(ExistentialRuleList, "fit", side_effect=AssertionError("No training")):
            paths = validate_ERL.main(["--manifest", str(manifest), "--signal", str(self.root / "anomalous_validation.pt"),
                "--background", str(self.root / "reference_validation.pt"), "--event-only", "--score", "ratio", "--delta", ".03",
                "--self-prune", "--output-prefix", str(self.root / "standalone"), "--print-top", "1", "--print-pruned-top", "1"])
            self.assert_scores(paths[0], "ratio", .03)
            paths = scan_ERL.main(self.arguments("--event-only", "--validate-only", "--score", "excess", "--delta", ".05",
                                                 "--self-prune"))
        self.assert_scores(paths[0], "excess", .05)
        for path, before in originals.items():
            self.assertEqual(Path(path).read_bytes(), before)

    def test_uniform_scan_inheritance_and_mixed_grid_defaults(self):
        for score, deltas, expected, delta in ((["excess"], [".02"], "excess", .02),
                                              (["excess", "ratio"], [".01", ".02"], "gini", .001)):
            args = scan_ERL.parser().parse_args(self.arguments("--event-only", "--score", *score, "--delta", *deltas))
            from casual_tests.erl_scores import validation_score_settings
            self.assertEqual(validation_score_settings(args, "validation"), (expected, delta))


if __name__ == "__main__":
    unittest.main()
