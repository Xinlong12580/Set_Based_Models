"""Small end-to-end parameter scans and independent validation selection."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from casual_tests import scan_ERL, train_ERL as training, train_ERL_validation as single_wrapper, validate_ERL
from casual_tests._erl_common import ReferencePenalizedGini
from scripts.plot_acceptance_report import load_report
from set_based_models import CategoryCut, EventCut, ExistentialRuleList, ObservableSpec
from tests.test_erl_self_pruning import event, model_with
from tests.test_erl_trainer import sample


class ERLScanRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="erl_scan_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal = self.root / "anomalous_train.pt"
        self.background = self.root / "reference_train.pt"
        self.validation_signal = self.root / "anomalous_validation.pt"
        self.validation_background = self.root / "reference_validation.pt"
        for signal, training_path, validation_path in ((True, self.signal, self.validation_signal),
                                                       (False, self.background, self.validation_background)):
            data = sample(signal)
            torch.save(data, training_path)
            for name, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[name] = value[:3].clone()
            data["event_features"] = torch.tensor([[2], [2], [2]] if signal else [[2], [0], [0]])
            torch.save(data, validation_path)
        self.prefix = self.root / "outputs with spaces" / "scan"

    def training_args(self, extra=()):
        return ["--signal", str(self.signal), "--background", str(self.background),
                "--output-prefix", str(self.prefix), "--max-rule-depth", "2", "--beam-width", "4",
                "--max-candidate-rules", "8", "--max-thresholds", "3", "--print-top", "1",
                "--print-reference-top", "0", "--print-pruned-top", "1", *extra]

    def train(self, extra=()):
        with redirect_stdout(io.StringIO()):
            manifest = scan_ERL.main(self.training_args([*extra, "--train-only"]))
        return manifest, json.loads(manifest.read_text())

    def test_standalone_trainer_accepts_scalars_and_saves_exactly_one_model(self):
        args = self.training_args(["--event-only", "--novelty-beta", "-.02",
                                   "--beam-signal-similarity-threshold", ".99"])
        parsed = training.parser().parse_args(args)
        self.assertTrue(all(not isinstance(getattr(parsed, key), list) for key in training.SEARCH_PARAMETERS))
        with redirect_stdout(io.StringIO()), patch.object(ExistentialRuleList, "prune_candidates",
                side_effect=AssertionError("Training must not prune")):
            model_path, report_path = training.main(args)
        self.assertEqual(model_path, Path(f"{self.prefix}.training.model.json"))
        self.assertEqual(report_path, Path(f"{self.prefix}.training.report.json"))
        self.assertFalse(Path(f"{self.prefix}.scan.json").exists())
        model = ExistentialRuleList.load(model_path)
        self.assertEqual(model.beam_novelty_weight, -.02)
        self.assertEqual(model.beam_signal_similarity_threshold, .99)
        self.assertIsNone(model.pruning_result_)

    def test_grid_saves_every_unpruned_model_and_scalar_search_metadata(self):
        with patch.object(ExistentialRuleList, "prune_candidates", side_effect=AssertionError("Training must not prune")), \
                patch.object(training, "train", wraps=training.train) as train, \
                patch.object(training, "load_inputs", wraps=training.load_inputs) as load:
            manifest, scan = self.train(["--event-only", "--beam-novelty-weight", "0", ".02",
                                         "--beam-signal-similarity-threshold", ".8", ".99",
                                         "--beam-background-similarity-threshold", ".97", "--beam-similarity-filter"])
        self.assertEqual(train.call_count, 4)
        load.assert_called_once()
        for call in train.call_args_list:
            self.assertTrue(all(not isinstance(getattr(call.args[0], key), list) for key in training.SEARCH_PARAMETERS))
            self.assertIs(call.kwargs["data"], train.call_args_list[0].kwargs["data"])
        self.assertEqual(manifest, Path(f"{self.prefix}.scan.json"))
        self.assertEqual(len(scan["runs"]), 4)
        pairs = set()
        for run in scan["runs"]:
            model = ExistentialRuleList.load(run["model"])
            report = load_report(run["report"])
            parameters = run["search_parameters"]
            pairs.add((parameters["beam_novelty_weight"], parameters["beam_signal_similarity_threshold"]))
            self.assertIsNone(model.pruning_result_)
            self.assertEqual(model.statistics_source_, "training")
            self.assertEqual(model.selected_rules_, [])
            self.assertEqual(model.beam_background_similarity_threshold, .97)
            self.assertEqual(model.beam_novelty_weight, parameters["beam_novelty_weight"])
            self.assertFalse(report["summary"]["final_pruning_applied"])
            self.assertEqual(report["configuration"]["beam_signal_similarity_threshold"], parameters["beam_signal_similarity_threshold"])
            self.assertIsNone(report["configuration"]["final_signal_similarity_threshold"])
            self.assertTrue(all(r["pruning"]["status"] == "not_pruned" for r in report["results"]["rules"]))
        self.assertEqual(pairs, {(0., .8), (0., .99), (.02, .8), (.02, .99)})

    def test_standalone_validation_merges_every_candidate_and_uses_its_own_settings(self):
        manifest, scan = self.train(["--event-only", "--reference-penalty", "0", "2",
                                     "--beam-signal-similarity-threshold", ".99",
                                     "--beam-background-similarity-threshold", ".98"])
        before = {run["model"]: Path(run["model"]).read_bytes() for run in scan["runs"]}
        with redirect_stdout(io.StringIO()), patch.object(ExistentialRuleList, "fit", side_effect=AssertionError("Validation must not search")):
            paths = validate_ERL.main([
                "--manifest", str(manifest), "--signal", str(self.validation_signal),
                "--background", str(self.validation_background), "--event-only", "--reference-penalty", ".25",
                "--final-similarity-filter", "--final-signal-similarity-threshold", ".7",
                "--final-background-similarity-threshold", ".6", "--max-trigger-rate-khz", "20000",
                "--input-rate-mhz", "40", "--stricter-score-improvement", "-.01",
                "--print-top", "1", "--print-reference-top", "0", "--print-pruned-top", "1",
            ])
        model = ExistentialRuleList.load(paths[0])
        report = load_report(paths[1])
        self.assertEqual(len(model.candidate_rules_), sum(len(ExistentialRuleList.load(p).candidate_rules_) for p in before))
        self.assertEqual(model.statistics_source_, "recalculated")
        self.assertEqual(model.pruning_configuration_["signal_similarity_threshold"], .7)
        self.assertEqual(model.pruning_configuration_["background_similarity_threshold"], .6)
        self.assertEqual(model.pruning_configuration_["max_reference_acceptance"], .5)
        self.assertTrue(model.selected_rules_)
        self.assertEqual(report["stage"], "validation")
        self.assertEqual(report["configuration"]["reference_penalty"], .25)
        self.assertEqual(report["configuration"]["final_signal_similarity_threshold"], .7)
        self.assertNotIn("beam_signal_similarity_threshold", report["configuration"])
        self.assertNotIn("beam_width", report["configuration"])
        self.assertEqual(report["source_scan_runs"], scan["runs"])
        self.assertTrue(all(r["source"]["model"] in before for r in report["results"]["rules"]))
        for rule in model.candidate_rules_:
            self.assertEqual(rule.statistics.signal_total, 3)
            self.assertEqual(rule.statistics.background_total, 3)
            self.assertEqual(rule.generation_score, ReferencePenalizedGini(.25)(rule.statistics))
        np.testing.assert_array_equal(model.predict(event_features=[[2], [0]]), [True, False])
        for path, original in before.items():
            self.assertEqual(Path(path).read_bytes(), original)

    def test_wrapper_uses_sibling_validation_files_and_separate_scoring_and_thresholds(self):
        output = io.StringIO()
        with redirect_stdout(output):
            paths = scan_ERL.main(self.training_args([
                "--event-only", "--reference-penalty", "0", "3", "--validation-reference-penalty", ".2",
                "--novelty-beta", "-.01", ".01", "--beam-similarity-filter",
                "--beam-signal-similarity-threshold", ".99", "--beam-background-similarity-threshold", ".98",
                "--final-similarity-filter", "--signal-similarity-threshold", ".7",
                "--background-similarity-threshold", ".6", "--stricter-score-improvement", "-.01",
            ]))
        scan = json.loads(Path(f"{self.prefix}.scan.json").read_text())
        self.assertEqual(len(scan["runs"]), 4)
        report = load_report(paths[1])
        self.assertEqual(Path(report["inputs"]["signal"]["path"]), self.validation_signal)
        self.assertEqual(report["configuration"]["reference_penalty"], .2)
        self.assertEqual(report["configuration"]["final_signal_similarity_threshold"], .7)
        self.assertIn("SEARCH 4/4", output.getvalue())
        self.assertEqual({r["search_parameters"]["reference_penalty"] for r in scan["runs"]}, {0., 3.})

    def test_validate_only_prefers_manifest_and_does_not_load_training_inputs(self):
        manifest, scan = self.train(["--event-only", "--reference-penalty", "0",
                                     "--beam-novelty-weight", "0", ".02"])
        before = {run["model"]: Path(run["model"]).read_bytes() for run in scan["runs"]}
        manifest_before = manifest.read_bytes()
        Path(f"{self.prefix}.run999.training.model.json").write_text("unlisted stale file")
        self.signal.unlink()
        self.background.unlink()
        with redirect_stdout(io.StringIO()), patch.object(scan_ERL, "train_scan", side_effect=AssertionError), \
                patch.object(ExistentialRuleList, "fit", side_effect=AssertionError), \
                patch.object(training, "load_inputs", wraps=training.load_inputs) as load:
            paths = scan_ERL.main(self.training_args([
                "--event-only", "--validate-only", "--validation-reference-penalty", "0", "--self-prune",
            ]))
        load.assert_called_once()
        self.assertEqual(load.call_args.args[:2], (self.validation_signal, self.validation_background))
        report = load_report(paths[1])
        self.assertEqual(report["source_scan_runs"], scan["runs"])
        self.assertTrue(report["configuration"]["self_pruning_applied"])
        self.assertEqual(report["inputs"]["signal"]["event_count"], 3)
        self.assertEqual(manifest.read_bytes(), manifest_before)
        for path, contents in before.items():
            self.assertEqual(Path(path).read_bytes(), contents)

    def test_validate_only_without_manifest_discovers_models_and_accepts_validation_paths(self):
        manifest, scan = self.train(["--event-only", "--beam-novelty-weight", "0", ".02"])
        manifest.unlink()
        Path(f"{self.prefix}.run005.validation.model.json").write_text("not a training model")
        Path(f"{self.prefix}_other.run000.training.model.json").write_text("different prefix")
        with redirect_stdout(io.StringIO()), patch.object(scan_ERL, "train_scan", side_effect=AssertionError), \
                patch.object(ExistentialRuleList, "fit", side_effect=AssertionError):
            paths = scan_ERL.main(self.training_args([
                "--event-only", "--validate-only", "--validation-reference-penalty", "0",
                "--signal", str(self.validation_signal), "--background", str(self.validation_background),
            ]))
        model = ExistentialRuleList.load(paths[0])
        report = load_report(paths[1])
        self.assertEqual(len(model.merge_provenance_["sources"]), 2)
        self.assertEqual({s["name"] for s in model.merge_provenance_["sources"]}, {run["model"] for run in scan["runs"]})
        self.assertIsNone(report["source_scan_runs"])
        self.assertEqual(Path(report["inputs"]["signal"]["path"]), self.validation_signal)
        self.assertEqual(Path(report["inputs"]["background"]["path"]), self.validation_background)
        self.assertEqual(paths[0], Path(f"{self.prefix}.validation.model.json"))

    def test_validate_only_respects_explicit_validation_overrides(self):
        self.train(["--event-only"])
        with redirect_stdout(io.StringIO()), patch.object(scan_ERL, "train_scan", side_effect=AssertionError):
            paths = scan_ERL.main(self.training_args([
                "--event-only", "--validate-only", "--validation-reference-penalty", "0",
                "--signal", str(self.root / "missing_signal_train.pt"),
                "--background", str(self.root / "missing_reference_train.pt"),
                "--validation-signal", str(self.validation_signal),
                "--validation-background", str(self.validation_background),
            ]))
        report = load_report(paths[1])
        self.assertEqual(Path(report["inputs"]["signal"]["path"]), self.validation_signal)
        self.assertEqual(Path(report["inputs"]["background"]["path"]), self.validation_background)

    def test_validate_only_infers_the_same_default_prefix(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        args = self.training_args(["--event-only"])
        index = args.index("--output-prefix")
        del args[index:index + 2]
        with redirect_stdout(io.StringIO()):
            manifest = scan_ERL.main([*args, "--train-only"])
        expected_prefix = self.root / "training_outputs" / f"ERL_{self.root.name}_event_only"
        self.assertEqual(manifest, Path(f"{expected_prefix}.scan.json"))
        self.signal.unlink()
        self.background.unlink()
        with redirect_stdout(io.StringIO()), patch.object(scan_ERL, "train_scan", side_effect=AssertionError):
            paths = scan_ERL.main([*args, "--validate-only", "--validation-reference-penalty", "0"])
        self.assertEqual(paths[0], Path(f"{expected_prefix}.validation.model.json"))

    def test_validate_only_missing_saved_models_does_not_train(self):
        with patch.object(scan_ERL, "train_scan", side_effect=AssertionError), \
                patch.object(training, "load_inputs", side_effect=AssertionError):
            with self.assertRaisesRegex(FileNotFoundError, "No saved scan"):
                scan_ERL.main(self.training_args(["--event-only", "--validate-only"]))

    def test_quick_runner_forwards_validate_only_without_launching_training(self):
        stub = self.root / "python"
        stub.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        stub.chmod(0o755)
        script = Path(__file__).resolve().parents[1] / "quick_runners" / "05_train_SIG_ERL.sh"
        run = subprocess.run(["bash", str(script), "--validate-only"], cwd=self.root,
                             env={**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"]},
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        calls = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(call[0], "casual_tests/scan_ERL.py")
            self.assertTrue(scan_ERL.parser().parse_args(call[1:]).validate_only)

    def test_single_wrapper_delegates_once_with_independent_scores_and_thresholds(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(training, "train", wraps=training.train) as train, \
                patch.object(validate_ERL, "validate", wraps=validate_ERL.validate) as validate:
            model_path, report_path = single_wrapper.main(self.training_args([
                "--event-only", "--beam-reference-penalty", "0", "--pruning-reference-penalty", ".5",
                "--beam-novelty-weight", ".02", "--beam-similarity-filter",
                "--beam-signal-similarity-threshold", ".98", "--beam-background-similarity-threshold", ".97",
                "--final-similarity-filter", "--final-signal-similarity-threshold", ".8",
                "--final-background-similarity-threshold", ".7", "--stricter-score-improvement", "-.01",
            ]))
        train.assert_called_once()
        validate.assert_called_once()
        self.assertEqual(train.call_args.args[0].reference_penalty, 0.)
        self.assertEqual(validate.call_args.args[0].reference_penalty, .5)
        self.assertFalse(Path(f"{self.prefix}.scan.json").exists())
        trained = ExistentialRuleList.load(f"{self.prefix}.training.model.json")
        selected = ExistentialRuleList.load(model_path)
        self.assertIsNone(trained.pruning_result_)
        self.assertEqual(trained.beam_signal_similarity_threshold, .98)
        self.assertEqual(trained.beam_background_similarity_threshold, .97)
        self.assertEqual(selected.pruning_configuration_["signal_similarity_threshold"], .8)
        self.assertEqual(selected.pruning_configuration_["background_similarity_threshold"], .7)
        self.assertEqual([r.cuts for r in trained.candidate_rules_], [r.cuts for r in selected.candidate_rules_])
        for model, penalty, total in ((trained, 0., 4), (selected, .5, 3)):
            for rule in model.candidate_rules_:
                for step in rule.steps:
                    self.assertEqual(step.statistics.signal_total, total)
                    self.assertEqual(step.generation_score, ReferencePenalizedGini(penalty)(step.statistics))
        report = load_report(report_path)
        self.assertEqual(report["configuration"]["beam_reference_penalty"], 0.)
        self.assertEqual(report["configuration"]["pruning_reference_penalty"], .5)
        self.assertEqual(report["configuration"]["scoring_stage"], "validation_pruning")
        self.assertIn("BEAM SCORE: Gini gain - 0", output.getvalue())
        self.assertIn("PRUNING SCORE: Gini gain - 0.5", output.getvalue())

    def test_single_wrapper_shared_similarity_flags_remain_a_beam_fallback(self):
        with redirect_stdout(io.StringIO()):
            single_wrapper.main(self.training_args(["--event-only", "--signal-similarity-threshold", ".9",
                                                    "--background-similarity-threshold", ".85"]))
        trained = ExistentialRuleList.load(f"{self.prefix}.training.model.json")
        self.assertEqual(trained.beam_signal_similarity_threshold, .9)
        self.assertEqual(trained.beam_background_similarity_threshold, .85)

    def test_object_combined_pair_modes_and_per_source_priority_reports(self):
        for name, flags in (("objects", ["--object-only", "--basic-object-features", "--object-cut-priority", "type_id"]),
                            ("combined", []),
                            ("pairs", ["--pair-tensor", "--object-only", "--object-cut-priority", "pt_1:>"])):
            self.prefix = self.root / name
            with self.subTest(mode=name), redirect_stdout(io.StringIO()):
                paths = scan_ERL.main(self.training_args([*flags, "--reference-penalty", "0",
                                       "--validation-reference-penalty", "0", "--require-complete-priority"]))
                report = load_report(paths[1])
                model = ExistentialRuleList.load(paths[0])
                records = {r["rule_id"]: r for r in report["results"]["rules"]}
                self.assertEqual(model.pruning_configuration_["require_complete_priority"], True)
                self.assertEqual(model.object_cut_priority, ())
                for rule in model.selected_rules_:
                    self.assertTrue(records[rule.rule_id]["object_cut_priority_complete"])
                expected = [] if name == "combined" else ["type_id"] if name == "objects" else ["pt_1:>"]
                self.assertTrue(all(r["object_cut_priority"] == expected for r in records.values()))

    def test_validation_accepts_quoted_wildcards_and_rejects_schema_changes(self):
        _, scan = self.train(["--event-only"])
        with redirect_stdout(io.StringIO()):
            paths = validate_ERL.main([
                "--models", str(self.prefix) + ".run*.training.model.json",
                "--signal", str(self.validation_signal), "--background", str(self.validation_background),
                "--event-only", "--output-prefix", str(self.root / "wildcard"),
                "--print-top", "1", "--print-reference-top", "0",
            ])
        self.assertEqual(len(ExistentialRuleList.load(paths[0]).merge_provenance_["sources"]), 1)
        data = torch.load(self.validation_signal, weights_only=True)
        data["event_feature_columns"] = ["different_meaning"]
        torch.save(data, self.validation_signal)
        with self.assertRaisesRegex(ValueError, "schemas"):
            validate_ERL.main(["--models", scan["runs"][0]["model"], "--signal", str(self.validation_signal),
                               "--background", str(self.validation_background), "--event-only"])

    def test_self_pruning_scan_uses_separate_local_gap_and_reports_parentage(self):
        with redirect_stdout(io.StringIO()):
            paths = scan_ERL.main(self.training_args([
                "--event-only", "--reference-penalty", "0", "--validation-reference-penalty", ".2",
                "--beam-novelty-weight", "0", ".02", "--self-prune",
                "--self-prune-score-improvement", "-.02", "--stricter-score-improvement", "-.01",
                "--final-similarity-filter",
            ]))
        model = ExistentialRuleList.load(paths[0])
        report = load_report(paths[1])
        scan = json.loads(Path(f"{self.prefix}.scan.json").read_text())
        source_count = sum(len(ExistentialRuleList.load(run["model"]).candidate_rules_) for run in scan["runs"])
        self.assertEqual(model.self_pruning_provenance_["configuration"]["stricter_score_improvement"], -.02)
        self.assertEqual(model.pruning_configuration_["stricter_score_improvement"], -.01)
        self.assertEqual(report["configuration"]["pruning_stage"], "after_validation_self_pruning")
        self.assertTrue(report["configuration"]["self_pruning_applied"])
        self.assertEqual(report["summary"]["self_pruning_source_rule_count"], source_count)
        self.assertEqual(report["self_pruning_provenance"], model.self_pruning_provenance_)
        self.assertTrue(model.selected_rules_)
        for rule in model.candidate_rules_:
            self.assertEqual(rule.statistics.signal_total, 3)
            self.assertEqual(rule.statistics.background_total, 3)
            self.assertEqual(rule.generation_score, ReferencePenalizedGini(.2)(rule.statistics))
        for record in report["results"]["rules"]:
            self.assertEqual(record["self_pruning_parent_rule_ids"],
                             model.self_pruning_provenance_["rules"][record["rule_id"]]["parent_rule_ids"])
        for run in scan["runs"]:
            trained = ExistentialRuleList.load(run["model"])
            self.assertIsNone(trained.pruning_result_)
            self.assertIsNone(trained.self_pruning_provenance_)
            self.assertFalse(load_report(run["report"])["configuration"]["self_pruning_applied"])

    def test_standalone_self_pruning_keeps_guard_only_and_skips_recalculation(self):
        specs = (ObservableSpec("eta2", "continuous", validity_column="two_photons"),
                 ObservableSpec("two_photons", "boolean"))
        source = model_with([(event(0, "<=", .5),)], events=specs)
        model_path = source.save(self.root / "guard.training.model.json")
        original = model_path.read_bytes()
        for signal, path in ((True, self.validation_signal), (False, self.validation_background)):
            data = sample(signal)
            data["event_features"] = torch.tensor([[.1, 1], [.2, 1], [.9, 1], [0, 0]] if signal else
                                                   [[.1, 1], [0, 0], [0, 0], [0, 0]])
            data["event_feature_columns"] = ["eta2", "two_photons"]
            data["event_feature_schema"] = [
                {"name": "eta2", "kind": "continuous", "validity_column": "two_photons"},
                {"name": "two_photons", "kind": "boolean"}]
            torch.save(data, path)
        with redirect_stdout(io.StringIO()), patch.object(ExistentialRuleList, "fit", side_effect=AssertionError), \
                patch.object(ExistentialRuleList, "recalculate_candidates", side_effect=AssertionError):
            paths = validate_ERL.main([
                "--models", str(model_path), "--signal", str(self.validation_signal),
                "--background", str(self.validation_background), "--event-only", "--reference-penalty", "0",
                "--self-prune", "--stricter-score-improvement", "-.01", "--output-prefix", str(self.prefix),
            ])
        model = ExistentialRuleList.load(paths[0])
        report = load_report(paths[1])
        self.assertEqual(model.get_rule(0).cuts, (EventCut(CategoryCut(1, {1})),))
        self.assertEqual(model.get_rule(0).statistics.signal_accepted, 3)
        self.assertEqual(model.get_rule(0).statistics.background_accepted, 1)
        self.assertEqual(model.self_pruning_provenance_["configuration"]["stricter_score_improvement"], -.01)
        self.assertEqual(report["results"]["rules"][0]["self_pruning_parent_rule_ids"], [0])
        self.assertEqual(model_path.read_bytes(), original)
        np.testing.assert_array_equal(model.predict(event_features=[[.9, 1], [0, 0]]), [True, False])

    def test_self_pruning_supports_objects_pairs_and_combined_inputs(self):
        for name, flags in (("objects", ["--object-only", "--object-cut-priority", "type_id"]),
                            ("pairs", ["--pair-tensor", "--object-only", "--object-cut-priority", "pt_1:>"]),
                            ("combined", [])):
            self.prefix = self.root / name
            if name == "pairs":
                for path in (self.signal, self.background, self.validation_signal, self.validation_background):
                    data = torch.load(path, weights_only=True)
                    data["pair_obj"][..., 0] = data["obj"][..., 0]
                    torch.save(data, path)
            with self.subTest(mode=name), redirect_stdout(io.StringIO()):
                paths = scan_ERL.main(self.training_args([
                    *flags, "--reference-penalty", "0", "--validation-reference-penalty", "0",
                    "--self-prune", "--require-complete-priority", "--max-trigger-rate-khz", "20000",
                ]))
            model = ExistentialRuleList.load(paths[0])
            report = load_report(paths[1])
            self.assertTrue(model.selected_rules_)
            self.assertTrue(report["configuration"]["self_pruning_applied"])
            self.assertTrue(model.self_pruning_provenance_["configuration"]["require_complete_priority"])
            self.assertEqual(model.self_pruning_provenance_["configuration"]["max_reference_acceptance"], .5)
            self.assertTrue(all(r["object_cut_priority_complete"] for r in report["results"]["rules"]))

    def test_self_pruning_is_opt_in_and_train_only_does_not_shorten(self):
        options = scan_ERL.parser().parse_args(self.training_args(["--event-only"]))
        self.assertFalse(options.self_prune)
        with patch.object(ExistentialRuleList, "self_prune_candidates", side_effect=AssertionError):
            manifest, scan = self.train(["--event-only", "--self-prune"])
        self.assertEqual(len(scan["runs"]), 1)
        self.assertFalse(Path(f"{self.prefix}.validation.model.json").exists())

    def test_cli_help_from_outside_repo_and_training_options_are_separate(self):
        scripts = Path(__file__).resolve().parents[1] / "casual_tests"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)
        for name in ("train_ERL.py", "train_ERL_validation.py", "validate_ERL.py", "scan_ERL.py"):
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, "-B", str(scripts / name), "--help"],
                                         cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
        train_options = training.parser()._option_string_actions
        wrapper_options = scan_ERL.parser()._option_string_actions
        validation_options = validate_ERL.parser()._option_string_actions
        for name in training.SEARCH_PARAMETERS:
            option = "--" + name.replace("_", "-")
            self.assertIsNone(train_options[option].nargs)
            self.assertEqual(wrapper_options[option].nargs, "+")
        self.assertNotIn("--final-similarity-filter", train_options)
        self.assertNotIn("--beam-novelty-weight", validation_options)
        self.assertNotIn("--beam-width", validation_options)
        self.assertNotIn("--model", wrapper_options)
        self.assertNotIn("--report", wrapper_options)
        for option in ("--self-prune", "--self-prune-score-improvement"):
            self.assertNotIn(option, train_options)
            self.assertIn(option, validation_options)
            self.assertIn(option, wrapper_options)
            self.assertEqual(wrapper_options[option].nargs, 0 if option == "--self-prune" else None)
        self.assertFalse((scripts / "train_ERL_scan.py").exists())


if __name__ == "__main__":
    unittest.main()
