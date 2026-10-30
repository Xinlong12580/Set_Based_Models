"""Training outputs round-trip through the public core loader; no physics data."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from casual_tests.stale import train_ERL_pruned as trainer
from casual_tests import train_ERL_validation as validation_trainer
from scripts.plot_acceptance_report import load_report
from set_based_models import ExistentialRuleList


def sample(signal):
    pts = [50, 60, 15, 20] if signal else [5, 10, 15, 20]
    objects = torch.tensor([[[pt, .2, 1., 1], [0, 0, 0, 0]] for pt in pts])
    pairs = torch.tensor([[[1, 1, 2 * pt, 1], [0, 0, 0, 0]] for pt in pts])
    categories = {0: "jet", 1: "photon"}
    object_schema = [{"name": name, "kind": "categorical" if name == "type_id" else "continuous",
                      **({"categories": categories} if name == "type_id" else {})}
                     for name in ("pt", "eta", "phi", "type_id")]
    pair_schema = [{"name": "pt_1", "kind": "continuous"}, {"name": "pt_2", "kind": "continuous"},
                   {"name": "pair_mass", "kind": "continuous", "validity_column": "pair_mass_valid"},
                   {"name": "pair_mass_valid", "kind": "boolean"}]
    return {
        "obj": objects, "obj_columns": [item["name"] for item in object_schema], "obj_feature_schema": object_schema,
        "pair_obj": pairs, "pair_slot_mask": pairs[..., 0] > 0,
        "pair_obj_columns": [item["name"] for item in pair_schema], "pair_obj_feature_schema": pair_schema,
        "event_features": torch.tensor([[2 if pt > 30 else 0] for pt in pts]),
        "event_feature_columns": ["n_photons"], "label": torch.full((4,), 2 if signal else 1),
    }


class ERLTrainerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="erl_training_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal, self.background = self.root / "anomalous.pt", self.root / "reference.pt"
        torch.save(sample(True), self.signal)
        torch.save(sample(False), self.background)
        self.prefix = self.root / "training outputs" / "example"

    def args(self, extra=()):
        return ["--signal", str(self.signal), "--background", str(self.background),
                "--output-prefix", str(self.prefix), "--print-top", "1", "--print-reference-top", "0", *extra]

    def run_training(self, extra=()):
        output = io.StringIO()
        with redirect_stdout(output):
            trainer.main(self.args(extra))
        return output.getvalue()

    def test_saved_models_match_fitted_rules_and_selected_or_in_every_input_mode(self):
        for mode, options in (("events", ["--event-only"]), ("objects", ["--object-only"]),
                              ("combined", []), ("pairs", ["--pair-tensor", "--object-only"])):
            with self.subTest(mode=mode):
                self.prefix = self.root / mode / "training"
                captured = []

                def build(*args, **kwargs):
                    model = ExistentialRuleList(*args, **kwargs)
                    captured.append(model)
                    return model

                with patch.object(trainer, "ExistentialRuleList", side_effect=build):
                    log = self.run_training(options)
                report_path, model_path = Path(f"{self.prefix}.report.json"), Path(f"{self.prefix}.model.json")
                report = load_report(report_path)
                loaded = ExistentialRuleList.load(model_path)
                self.assertEqual(loaded.to_dict(), captured[0].to_dict())
                self.assertEqual(report["stage"], "training")
                self.assertEqual(report["trainer"], "casual_tests/train_ERL.py")
                self.assertEqual(report["outputs"], {"model": str(model_path), "training_report": str(report_path)})
                self.assertEqual(report["summary"]["pruned_rule_ids"], [r.rule_id for r in loaded.selected_rules_])
                for path in (self.signal, self.background):
                    data = trainer.load_dataset(path, use_objects=mode != "events", use_events=mode in ("events", "combined"), use_pairs=mode == "pairs")
                    objects = None if mode == "events" else data.objects
                    kwargs = {"slot_mask": None if objects is None else data.slot_mask, "event_features": data.event_features}
                    np.testing.assert_array_equal(loaded.predict(objects, **kwargs), captured[0].predict(objects, **kwargs))
                    for rule, record in zip(loaded.candidate_rules_, report["results"]["rules"]):
                        self.assertEqual(rule.rule_id, record["rule_id"])
                        self.assertEqual(rule.generation_score, record["score"])
                        self.assertEqual(rule.statistics.signal_accepted, record["metrics"]["signal_accepted"])
                self.assertIn("Core model saved to:", log)
                self.assertIn("Training report saved to:", log)

    def test_legacy_report_option_has_a_companion_model_and_new_model_override(self):
        report = self.root / "legacy.txt"
        args = self.args(["--event-only", "--report", str(report)])
        # Without a prefix, a legacy report name also determines its companion.
        del args[args.index("--output-prefix"):args.index("--output-prefix") + 2]
        with redirect_stdout(io.StringIO()):
            trainer.main(args)
        self.assertEqual(json.loads(report.read_text())["outputs"]["model"], str(self.root / "legacy.model.json"))
        self.assertTrue((self.root / "legacy.model.json").is_file())
        custom = self.root / "custom.model.json"
        self.run_training(["--event-only", "--model", str(custom)])
        self.assertTrue(custom.is_file())
        self.assertTrue(Path(f"{self.prefix}.report.json").is_file())

    def test_training_report_suffix_is_not_duplicated_for_companion_model(self):
        report = self.root / "custom.report.json"
        args = ["--signal", str(self.signal), "--background", str(self.background), "--event-only", "--training-report", str(report)]
        with redirect_stdout(io.StringIO()):
            trainer.main(args)
        self.assertTrue((self.root / "custom.model.json").is_file())

    def test_existing_output_is_protected_before_fitting_and_overwrite_is_explicit(self):
        model_path = Path(f"{self.prefix}.model.json")
        model_path.parent.mkdir(parents=True)
        model_path.write_text("keep model")
        with patch.object(trainer.ExistentialRuleList, "fit") as fit, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.main(self.args(["--event-only"]))
        fit.assert_not_called()
        self.assertEqual(model_path.read_text(), "keep model")
        self.assertFalse(Path(f"{self.prefix}.report.json").exists())
        with self.assertWarnsRegex(UserWarning, "Overwriting"):
            self.run_training(["--event-only", "--overwrite-output"])
        self.assertTrue(ExistentialRuleList.load(model_path).selected_rules_)
        with self.assertWarnsRegex(UserWarning, "Overwriting"):
            self.run_training(["--event-only", "--overwrite-report"])

    def test_model_targets_cannot_overwrite_inputs_or_aliases_or_report(self):
        original = self.signal.read_bytes()
        hardlink, symlink = self.root / "linked.pt", self.root / "symlink.pt"
        hardlink.hardlink_to(self.signal)
        symlink.symlink_to(self.signal)
        same_report = self.root / "same.json"
        for target in (self.signal, hardlink, symlink, self.root):
            with self.subTest(target=target), patch.object(trainer.ExistentialRuleList, "fit") as fit:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    trainer.main(self.args(["--event-only", "--model", str(target), "--overwrite-output"]))
                fit.assert_not_called()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.main(self.args(["--model", str(same_report), "--report", str(same_report)]))
        self.assertEqual(self.signal.read_bytes(), original)
        self.assertFalse(same_report.exists())

    def test_empty_final_selection_is_saved_without_falling_back_to_all_candidates(self):
        self.run_training(["--object-only", "--object-cut-priority", *(["pt"] * 5), "--require-complete-priority"])
        loaded = ExistentialRuleList.load(f"{self.prefix}.model.json")
        self.assertTrue(loaded.candidate_rules_)
        self.assertEqual(loaded.selected_rules_, [])
        data = trainer.load_dataset(self.signal, use_events=False)
        self.assertFalse(loaded.predict(data.objects, slot_mask=data.slot_mask).any())

    def test_novelty_cli_alias_is_saved_and_reported(self):
        log = self.run_training(["--event-only", "--novelty-beta", ".02"])
        model = ExistentialRuleList.load(f"{self.prefix}.model.json")
        report = load_report(Path(f"{self.prefix}.report.json"))
        self.assertEqual(model.beam_novelty_weight, .02)
        self.assertEqual(report["configuration"]["beam_novelty_weight"], .02)
        self.assertIn("BEAM NOVELTY: beta=0.02", log)


class ERLValidationTrainerTests(unittest.TestCase):
    def test_every_existing_cli_option_and_default_is_preserved(self):
        old, _, _, _ = trainer._parse_args(["--signal", "anomalous_train.pt", "--background", "reference_train.pt"])
        new = validation_trainer.parser()
        actions = {option: action for action in new._actions for option in action.option_strings}
        for action in old._actions:
            for option in action.option_strings:
                with self.subTest(option=option):
                    self.assertIn(option, actions)
                    self.assertEqual(actions[option].dest, action.dest)
                    self.assertEqual(actions[option].default, action.default)
                    self.assertEqual(actions[option].nargs, action.nargs)

    def test_training_is_unpruned_and_validation_refreshes_every_candidate_then_prunes(self):
        for mode, options in (("events", ["--event-only"]), ("objects", ["--object-only", "--basic-object-features"]),
                              ("combined", []), ("pairs", ["--pair-tensor", "--object-only"])):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="erl_validation_") as temporary:
                root = Path(temporary)
                for name, signal in (("anomalous_train", True), ("reference_train", False),
                                     ("anomalous_validation", False), ("reference_validation", True)):
                    torch.save(sample(signal), root / f"{name}.pt")
                prefix = root / "outputs" / mode
                train_path, validation_path = Path(f"{prefix}.training.model.json"), Path(f"{prefix}.validation.model.json")
                snapshots, pruning_calls = [], []
                original_prune = ExistentialRuleList.prune_candidates

                def prune(model, **kwargs):
                    snapshots.append(train_path.read_bytes())
                    self.assertIsNone(ExistentialRuleList.load(train_path).pruning_result_)
                    pruning_calls.append((model.statistics_source_, kwargs))
                    return original_prune(model, **kwargs)

                arguments = ["--signal", str(root / "anomalous_train.pt"), "--background", str(root / "reference_train.pt"),
                             "--output-prefix", str(prefix), "--reference-penalty", ".2", "--stricter-score-improvement", "-.01",
                             "--beam-similarity-filter", "--final-similarity-filter", "--signal-similarity-threshold", ".9",
                             "--background-similarity-threshold", ".9", "--max-trigger-rate-khz", "40", "--input-rate-mhz", "40",
                             "--beam-novelty-weight", ".01", "--print-top", "1", "--print-reference-top", "1", "--print-pruned-top", "1", *options]
                with patch.object(ExistentialRuleList, "prune_candidates", prune), redirect_stdout(io.StringIO()):
                    validation_trainer.main(arguments)
                self.assertEqual(len(pruning_calls), 1)
                self.assertEqual(pruning_calls[0][0], "recalculated")
                self.assertEqual(pruning_calls[0][1], dict(require_complete_priority=False, stricter_score_improvement=-.01,
                                 max_reference_acceptance=.001, similarity_filter=True, signal_similarity_threshold=.9,
                                 background_similarity_threshold=.9))
                self.assertEqual(train_path.read_bytes(), snapshots[0])
                train = ExistentialRuleList.load(train_path)
                validation = ExistentialRuleList.load(validation_path)
                self.assertIsNone(train.pruning_result_)
                self.assertEqual(train.selected_rules_, [])
                self.assertEqual(train.statistics_source_, "training")
                self.assertEqual(validation.statistics_source_, "recalculated")
                self.assertEqual(train.beam_novelty_weight, .01)
                self.assertEqual(validation.merge_provenance_["sources"][0]["configuration"]["beam_novelty_weight"], .01)
                self.assertIsNotNone(validation.pruning_result_)
                self.assertEqual([(r.rule_id, r.cuts) for r in train.candidate_rules_],
                                 [(r.rule_id, r.cuts) for r in validation.candidate_rules_])
                self.assertTrue(any(a.generation_score != b.generation_score
                                    for a, b in zip(train.candidate_rules_, validation.candidate_rules_)))
                training_report = load_report(Path(f"{prefix}.training.report.json"))
                report = load_report(Path(f"{prefix}.validation.report.json"))
                self.assertFalse(training_report["summary"]["final_pruning_applied"])
                self.assertTrue(all(r["pruning"]["status"] == "not_pruned" for r in training_report["results"]["rules"]))
                self.assertTrue(report["summary"]["final_pruning_applied"])
                self.assertEqual(report["stage"], "validation")
                self.assertEqual(report["inputs"]["signal"]["path"], str(root / "anomalous_validation.pt"))
                self.assertIn("validation_reference", report["configuration"]["rate_estimation"])
                self.assertEqual(report["summary"]["pruned_rule_ids"], [r.rule_id for r in validation.selected_rules_])
                records = report["results"]["rules"]
                self.assertEqual([r["score"] for r in records], sorted((r["score"] for r in records), reverse=True))
                signal = trainer.load_dataset(root / "anomalous_validation.pt", use_objects=mode != "events",
                                              use_events=mode in ("events", "combined"), use_pairs=mode == "pairs",
                                              basic_object_features=mode == "objects")
                for record in records:
                    rule = validation.get_rule(record["rule_id"])
                    selected = rule.predict(None if mode == "events" else signal.objects,
                                            slot_mask=None if mode == "events" else signal.slot_mask,
                                            event_features=signal.event_features)
                    self.assertEqual(int(selected.sum()), record["metrics"]["signal_accepted"])
                    self.assertEqual(rule.statistics.signal_accepted, int(selected.sum()))

    def test_explicit_validation_paths_output_overrides_and_priority_completion(self):
        with tempfile.TemporaryDirectory(prefix="erl_validation_explicit_") as temporary:
            root = Path(temporary)
            for name, signal in (("train_mix", True), ("train_ref", False), ("val_mix", True), ("val_ref", False)):
                torch.save(sample(signal), root / f"{name}.pt")
            model_path, report_path = root / "custom.json", root / "custom.txt"
            with redirect_stdout(io.StringIO()):
                validation_trainer.main(["--signal", str(root / "train_mix.pt"), "--background", str(root / "train_ref.pt"),
                                         "--validation-signal", str(root / "val_mix.pt"), "--validation-background", str(root / "val_ref.pt"),
                                         "--output-prefix", str(root / "study"), "--model", str(model_path), "--report", str(report_path),
                                         "--object-only", "--object-cut-priority", "type_id", "pt:>", "--require-complete-priority",
                                         "--reference-penalty", "0", "--stricter-score-improvement", "-.01", "--print-top", "1"])
            self.assertTrue(report_path.is_file())
            training = ExistentialRuleList.load(model_path)
            self.assertIsNone(training.pruning_result_)
            self.assertTrue(any(r.depth == 1 for r in training.candidate_rules_))
            validation = ExistentialRuleList.load(root / "study.validation.model.json")
            self.assertTrue(validation.selected_rules_)
            self.assertTrue(all(r.depth >= 2 for r in validation.selected_rules_))
            self.assertTrue(validation.pruning_configuration_["require_complete_priority"])


if __name__ == "__main__":
    unittest.main()
