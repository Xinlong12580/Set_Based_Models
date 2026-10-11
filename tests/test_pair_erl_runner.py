"""Pair-input runner checks using small synthetic converter outputs only."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from casual_tests import run_ERL_Gini_nested as runner
from raw_event_conversion import build_object_tensor, build_pair_tensor, object_feature_schema, pair_feature_schema
from raw_event_conversion.objects import OBJECT_COLUMNS
from raw_event_conversion.pairs import PAIR_COLUMNS
from raw_event_conversion.tests.test_pairs import records
from scripts.plot_acceptance_report import load_report, select_paths
from set_based_models import ExistentialRuleList


def fixture(signal):
    events = [{"jets": [{"pt": 15, "mass": 2}],
               "photons": ([{"pt": 60}, {"pt": 40, "phi": 3.}]
                           if signal and row < 6 else [{"pt": 10}, {"pt": 5}])}
              for row in range(12)]
    raw = records(events)
    pairs, mask = build_pair_tensor(raw)
    return {
        "obj": build_object_tensor(raw), "obj_columns": list(OBJECT_COLUMNS),
        "obj_feature_schema": object_feature_schema(),
        "pair_obj": pairs, "pair_slot_mask": mask,
        "pair_obj_columns": list(PAIR_COLUMNS), "pair_obj_feature_schema": pair_feature_schema(),
        "event_features": torch.tensor([[2 if signal and row < 6 else 0] for row in range(12)], dtype=torch.float64),
        "event_feature_columns": ["n_photons_pt20"],
        "event_feature_schema": [{"name": "n_photons_pt20", "kind": "ordinal", "units": "count"}],
        "label": torch.full((12,), 2 if signal else 1, dtype=torch.int64),
    }


class PairERLRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pair_erl_runner_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal = self.root / "anomalous.pt"
        self.background = self.root / "reference.pt"
        self.report = self.root / "report.txt"
        self.samples = {self.signal: fixture(True), self.background: fixture(False)}
        for path, sample in self.samples.items():
            torch.save(sample, path)

    def args(self, extra=(), *, explicit_report=True):
        return ["--signal", str(self.signal), "--background", str(self.background),
                *(["--report", str(self.report)] if explicit_report else []),
                "--print-top", "1", "--print-reference-top", "0", "--print-pruned-top", "1", *extra]

    def assert_cli_error(self, extra=("--pair-tensor", "--object-only")):
        with patch.object(runner.ExistentialRuleList, "fit") as fit:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main(self.args(extra))
            self.assertEqual(error.exception.code, 2)
            fit.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_pair_loader_uses_saved_mask_columns_and_guards(self):
        loaded = runner.load_dataset(self.signal, use_pairs=True)
        self.assertEqual(loaded.object_tensor_key, "pair_obj")
        self.assertEqual(loaded.objects.shape, (12, 231, 68))
        np.testing.assert_array_equal(loaded.objects, self.samples[self.signal]["pair_obj"].numpy())
        np.testing.assert_array_equal(loaded.slot_mask, self.samples[self.signal]["pair_slot_mask"].numpy())
        self.assertEqual([spec.name for spec in loaded.object_observables], list(PAIR_COLUMNS))
        self.assertEqual(loaded.object_observables[PAIR_COLUMNS.index("eta_1")].validity_column, "eta_valid_1")
        self.assertEqual(loaded.object_observables[PAIR_COLUMNS.index("pair_mass")].validity_column, "pair_mass_valid")

    def test_single_particles_remain_default_and_ignore_malformed_pairs(self):
        for key in ("pair_obj", "pair_slot_mask", "pair_obj_columns", "pair_obj_feature_schema"):
            self.samples[self.signal][key] = "ignored malformed pair data"
        torch.save(self.samples[self.signal], self.signal)
        loaded = runner.load_dataset(self.signal)
        self.assertEqual(loaded.object_tensor_key, "obj")
        np.testing.assert_array_equal(loaded.objects, self.samples[self.signal]["obj"].numpy())
        loaded = runner.load_dataset(self.signal, use_objects=False)
        self.assertIsNone(loaded.object_tensor_key)

    def test_pair_only_ignores_missing_single_objects_and_events(self):
        sample = self.samples[self.signal]
        for key in ("obj", "obj_columns", "obj_feature_schema", "event_features", "event_feature_columns", "event_feature_schema"):
            del sample[key]
        torch.save(sample, self.signal)
        loaded = runner.load_dataset(self.signal, use_pairs=True, use_events=False)
        self.assertEqual(loaded.objects.shape[-1], 68)
        self.assertIsNone(loaded.event_features)

    def test_reordering_pair_columns_preserves_named_validation(self):
        sample = self.samples[self.signal]
        order = list(reversed(range(len(PAIR_COLUMNS))))
        sample["pair_obj"] = sample["pair_obj"][..., order]
        for key in ("pair_obj_columns", "pair_obj_feature_schema"):
            sample[key] = [sample[key][index] for index in order]
        torch.save(sample, self.signal)
        loaded = runner.load_dataset(self.signal, use_pairs=True)
        self.assertEqual(loaded.slot_mask.sum(), 36)
        self.assertEqual([spec.name for spec in loaded.object_observables], list(reversed(PAIR_COLUMNS)))

    def test_basic_pair_projection_keeps_required_coordinate_guards(self):
        full = runner.load_dataset(self.signal, use_pairs=True)
        basic = runner.load_dataset(self.signal, use_pairs=True, basic_object_features=True)
        expected = (*runner.BASIC_PAIR_COLUMNS, "eta_valid_1", "phi_valid_1", "eta_valid_2", "phi_valid_2")
        self.assertEqual([spec.name for spec in basic.object_observables], list(expected))
        np.testing.assert_array_equal(basic.objects, full.objects[..., [PAIR_COLUMNS.index(name) for name in expected]])
        np.testing.assert_array_equal(basic.slot_mask, full.slot_mask)
        self.assertEqual(basic.objects.shape, (12, 231, 12))

    def test_pair_mask_is_required_boolean_and_row_slot_aligned(self):
        original = self.samples[self.signal]
        for mask in (None, original["pair_slot_mask"].float(), torch.ones(12, 230, dtype=torch.bool),
                     torch.ones(11, 231, dtype=torch.bool), torch.ones(12, 231, 1, dtype=torch.bool)):
            sample = {**original, "pair_slot_mask": mask}
            torch.save(sample, self.signal)
            with self.subTest(mask_shape=getattr(mask, "shape", None)):
                self.assert_cli_error()

    def test_true_pair_mask_requires_two_positive_finite_member_pts(self):
        original = self.samples[self.signal]
        for name, value in (("pt_1", 0), ("pt_2", -1), ("pt_2", float("nan"))):
            sample = {**original, "pair_obj": original["pair_obj"].clone()}
            slot = torch.nonzero(sample["pair_slot_mask"][0]).flatten()[0]
            sample["pair_obj"][0, slot, PAIR_COLUMNS.index(name)] = value
            torch.save(sample, self.signal)
            with self.subTest(name=name, value=value):
                self.assert_cli_error()

    def test_padded_pairs_and_guarded_measurements_do_not_enter_validation(self):
        sample = self.samples[self.signal]
        sample["pair_obj"][~sample["pair_slot_mask"]] = float("nan")
        sample["pair_obj"][..., PAIR_COLUMNS.index("abs_d0_1")] = float("nan")
        torch.save(sample, self.signal)
        loaded = runner.load_dataset(self.signal, use_pairs=True)
        self.assertEqual(loaded.slot_mask.sum(), 36)

    def test_required_pair_metadata_and_matching_schemas_are_enforced(self):
        original = self.samples[self.signal]
        for key in ("pair_obj", "pair_obj_columns", "pair_obj_feature_schema"):
            sample = {name: value for name, value in original.items() if name != key}
            torch.save(sample, self.signal)
            with self.subTest(key=key):
                self.assert_cli_error()
        sample = {**original, "pair_obj_feature_schema": [dict(item) for item in original["pair_obj_feature_schema"]]}
        sample["pair_obj_feature_schema"][0]["units"] = "MeV"
        torch.save(sample, self.signal)
        self.assert_cli_error()

    def test_pair_switch_conflicts_with_event_only_and_single_member_priority(self):
        self.assert_cli_error(["--pair-tensor", "--event-only"])
        self.assert_cli_error(["--pair-tensor", "--object-cut-priority", "type_id"])
        with self.assertRaisesRegex(ValueError, "Particle pairs"):
            runner.load_dataset("unused.pt", use_objects=False, use_pairs=True)

    def test_pair_cli_fits_reports_and_prunes_with_both_type_priorities(self):
        captured = []

        def build(*args, **kwargs):
            model = ExistentialRuleList(*args, **kwargs)
            captured.append(model)
            return model

        output = io.StringIO()
        with patch.object(runner, "ExistentialRuleList", side_effect=build), redirect_stdout(output):
            runner.main(self.args(["--pair-tensor", "--object-only", "--reference-penalty", "0",
                                   "--object-cut-priority", "type_id_1", "type_id_2",
                                   "--stricter-score-improvement", ".001",
                                   "--beam-similarity-filter", "--final-similarity-filter",
                                   "--signal-similarity-threshold", ".9", "--background-similarity-threshold", ".9",
                                   "--max-trigger-rate-khz", "40"]))
        report = load_report(self.report)
        self.assertEqual(report["feature_mode"], "object_only")
        self.assertEqual(report["object_tensor_key"], "pair_obj")
        self.assertEqual(report["configuration"]["object_tensor_key"], "pair_obj")
        self.assertTrue(report["configuration"]["pair_tensor"])
        self.assertIsNone(report["obj_feature_schema"])
        self.assertEqual(report["pair_obj_feature_schema"], json.loads(json.dumps(pair_feature_schema())))
        self.assertEqual(report["inputs"]["signal"]["object_shape"], [12, 231, 68])
        self.assertIn("EXISTS same pair", output.getvalue())
        self.assertIn("pair_slot_mask", output.getvalue())
        self.assertEqual(report["configuration"]["object_cut_priority"], ["type_id_1", "type_id_2"])
        loaded = runner.load_dataset(self.signal, use_pairs=True, use_events=False)
        for rule, record in zip(captured[0].candidate_rules_, report["results"]["rules"]):
            names = [step["cut"]["observable"] for step in record["steps"]]
            self.assertEqual(names[:2], ["type_id_1", "type_id_2"][:len(names)])
            selected = rule.predict(loaded.objects, slot_mask=loaded.slot_mask)
            self.assertEqual(record["metrics"]["signal_accepted"], int(selected.sum()))
        self.assertTrue(report["results"]["rules"])
        eligible, shown = select_paths(report, rules="all", direction="all", top=5)
        self.assertTrue(eligible)
        self.assertTrue(shown)

    def test_pair_cli_accepts_repeated_mass_priority_and_reports_each_position(self):
        priority = ["type_id_1", "type_id_2", "pair_mass", "pair_mass"]
        with redirect_stdout(io.StringIO()):
            runner.main(self.args(["--pair-tensor", "--object-only", "--reference-penalty", "0",
                                   "--stricter-score-improvement", "-0.008",
                                   "--object-cut-priority", *priority]))
        report = load_report(self.report)
        self.assertEqual(report["configuration"]["object_cut_priority"], priority)
        self.assertEqual(report["configuration"]["stricter_score_improvement"], -.008)
        rules = report["results"]["rules"]
        self.assertTrue(any(r["object_cut_priority_remaining"] == ["pair_mass"] for r in rules))
        for record in rules:
            names = [step["cut"]["observable"] for step in record["steps"]]
            self.assertEqual(names[:4], priority[:len(names)])
            self.assertEqual(record["object_cut_priority_remaining"], priority[len(names):])

    def test_pair_cli_directed_mass_window_priority(self):
        priority = ["type_id_1", "type_id_2", "pair_mass:>", "pair_mass:<="]
        with redirect_stdout(io.StringIO()):
            runner.main(self.args(["--pair-tensor", "--object-only", "--reference-penalty", "0",
                                   "--stricter-score-improvement", "-0.01",
                                   "--object-cut-priority", *priority]))
        report = load_report(self.report)
        self.assertEqual(report["configuration"]["object_cut_priority"], priority)
        self.assertTrue(report["results"]["rules"])
        for record in report["results"]["rules"]:
            steps = record["steps"]
            names = [s["cut"]["observable"] for s in steps]
            self.assertEqual(names[:4], ["type_id_1", "type_id_2", "pair_mass", "pair_mass"][:len(names)])
            self.assertEqual([s["cut"]["comparison"] for s in steps[2:4]], [">", "<="][:max(0, len(steps) - 2)])
            self.assertEqual(record["object_cut_priority_remaining"], priority[len(steps):])

    def test_pair_cli_can_require_completed_directed_windows(self):
        priority = ["type_id_1", "type_id_2", "pair_mass:>", "pair_mass:<="]
        with redirect_stdout(io.StringIO()):
            runner.main(self.args(["--pair-tensor", "--object-only", "--reference-penalty", "0",
                                   "--stricter-score-improvement", "-0.01",
                                   "--object-cut-priority", *priority, "--require-complete-priority"]))
        report = load_report(self.report)
        self.assertTrue(report["configuration"]["require_complete_priority"])
        self.assertGreater(report["summary"]["incomplete_priority_rules_removed"], 0)
        retained = [r for r in report["results"]["rules"] if r["retained_after_final_pruning"]]
        self.assertTrue(retained)
        for record in retained:
            self.assertTrue(record["object_cut_priority_complete"])
            self.assertEqual(record["object_cut_priority_remaining"], [])
            self.assertEqual([s["cut"]["comparison"] for s in record["steps"][2:4]], [">", "<="])

    def test_pair_alias_combined_events_and_basic_features(self):
        with redirect_stdout(io.StringIO()):
            runner.main(self.args(["--use-pair-tensor", "--basic-object-features",
                                   "--object-cut-priority", "type_id_1", "type_id_2"]))
        report = json.loads(self.report.read_text())
        self.assertEqual(report["feature_mode"], "combined")
        self.assertEqual(report["inputs"]["signal"]["object_shape"], [12, 231, 12])
        self.assertEqual(report["inputs"]["signal"]["event_feature_shape"], [12, 1])
        self.assertEqual(report["configuration"]["object_validity_guards"]["eta_1"], "eta_valid_1")
        self.assertTrue(any(step["scope"] == "event" for rule in report["results"]["rules"] for step in rule["steps"]))

    def test_pair_default_report_name_is_distinct(self):
        with patch.object(runner.Path, "cwd", return_value=self.root), redirect_stdout(io.StringIO()):
            runner.main(self.args(["--pair-tensor", "--object-only", "--basic-object-features"], explicit_report=False))
        reports = list((self.root / "reports").glob("*.txt"))
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].name.endswith("_pairs_basic_objects_object_only.txt"))


if __name__ == "__main__":
    unittest.main()
