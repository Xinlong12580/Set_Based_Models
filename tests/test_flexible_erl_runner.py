"""Flexible ERL runner checks using temporary synthetic PT files only."""

from contextlib import redirect_stderr, redirect_stdout
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casual_tests import run_ERL_Gini_nested as runner
from casual_tests.run_ERL_Gini_nested_event_only import main as event_main
from raw_event_conversion.objects import OBJECT_COLUMNS, object_feature_schema
from set_based_models import EventCut, ExistentialRuleList
from scripts.plot_acceptance_report import load_report, select_paths


def fixture(signal):
    objects = torch.zeros((40, 22, 20), dtype=torch.float32)
    objects[:, 0, 0] = 5
    if signal:
        objects[:15, 0, 0] = 50
        objects[:15, 0, 3] = 1
        objects[:15, 0, OBJECT_COLUMNS.index("charge")] = -1
        objects[:15, 0, OBJECT_COLUMNS.index("charge_valid")] = 1
    else:
        objects[:3, 0, 0] = 50
    events = torch.zeros((40, 3), dtype=torch.float64)
    events[:, 1] = 10
    events[:, 2] = 1
    events[10:25 if signal else 12, 0] = 2
    if signal:
        events[20:35, 1] = 100
    return {
        "obj": objects, "obj_columns": list(OBJECT_COLUMNS),
        "obj_feature_schema": object_feature_schema(),
        "event_features": events,
        "event_feature_columns": ["n_photons_pt20", "met", "met_valid"],
        "event_feature_schema": [
            {"name": "n_photons_pt20", "kind": "ordinal", "units": "count"},
            {"name": "met", "kind": "continuous", "units": "GeV"},
            {"name": "met_valid", "kind": "boolean"},
        ],
        "label": torch.full((40,), 7 if signal else 1, dtype=torch.int64),
    }


class FlexibleERLRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flexible_erl_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.signal = self.directory / "anomalous.pt"
        self.background = self.directory / "reference.pt"
        self.report = self.directory / "report.txt"
        torch.save(fixture(True), self.signal)
        torch.save(fixture(False), self.background)

    def args(self, extra=(), report=None):
        return ["--signal", str(self.signal), "--background", str(self.background),
                "--report", str(report or self.report), "--print-top", "1",
                "--print-reference-top", "0", "--print-pruned-top", "1", *extra]

    def run_cli(self, extra=(), report=None, main=None):
        output = io.StringIO()
        with redirect_stdout(output):
            (main or runner.main)(self.args(extra, report))
        return json.loads((report or self.report).read_text()), output.getvalue()

    def assert_cli_error(self, extra=()):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            runner.main(self.args(extra))
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.report.exists())

    def test_combined_default_loads_all_object_columns_and_both_cut_scopes(self):
        captured = []

        def build(*args, **kwargs):
            model = ExistentialRuleList(*args, **kwargs)
            captured.append(model)
            return model

        with patch.object(runner, "ExistentialRuleList", side_effect=build):
            report, log = self.run_cli(["--print-top", "0"])
        model = captured[0]
        self.assertEqual(model.input_mode_, "combined")
        self.assertEqual(len(model.observables), 20)
        self.assertEqual(len(model.event_observables), 3)
        records = report["results"]["rules"]
        self.assertTrue(records)
        self.assertEqual({s["scope"] for r in records for s in r["steps"]}, {"slot", "event"})
        self.assertTrue(any({s["scope"] for s in r["steps"]} == {"slot", "event"} for r in records))
        data = runner.load_dataset(self.signal)
        for rule, record in zip(model.candidate_rules_, records):
            selection = rule.predict(data.objects, slot_mask=data.slot_mask, event_features=data.event_features)
            np.testing.assert_array_equal(selection, rule.signal_selection)
            self.assertEqual(record["metrics"]["signal_accepted"], int(selection.sum()))
            self.assertTrue(record["logical_pruning_applicable"])
            if any(not isinstance(c, EventCut) for c in rule.cuts):
                self.assertIn("same slot", record["simplified_rule"])
        self.assertEqual(report["inputs"]["signal"]["object_shape"], [40, 22, 20])
        self.assertIn("[slot]", log)
        self.assertIn("[event]", log)
        self.assertIn("same-slot predicates and minimum counts are respected", log)

    def test_event_only_is_native_and_identical_to_original_runner(self):
        new, _ = self.run_cli(["--event-only"])
        old, _ = self.run_cli(report=self.directory / "old.txt", main=event_main)
        self.assertEqual(new["feature_mode"], "event_only")
        self.assertTrue(new["native_event_only"])
        self.assertIsNone(new["dummy_object_input"])
        self.assertEqual(new["object_observables"], [])
        self.assertEqual(new["inputs"]["signal"]["object_shape"], [40, 0, 0])
        keys = ("raw_rule", "score", "gini_gain", "pruned_rank", "pruning")
        self.assertEqual([[r[k] for k in keys] for r in new["results"]["rules"]],
                         [[r[k] for k in keys] for r in old["results"]["rules"]])

    def test_object_only_has_no_event_cuts(self):
        report, _ = self.run_cli(["--no-event-features"])
        self.assertEqual(report["feature_mode"], "object_only")
        self.assertEqual(report["event_observables"], [])
        self.assertIsNone(report["inputs"]["signal"]["event_feature_shape"])
        self.assertTrue(report["results"]["rules"])
        for r in report["results"]["rules"]:
            self.assertTrue(r["logical_pruning_applicable"])
            self.assertTrue(all(step["scope"] == "slot" for step in r["steps"]))
        self.assertGreater(report["summary"]["equivalent_rules_removed"]
                           + report["summary"]["stricter_rules_removed"], 0)

    def test_object_cut_priority_orders_cuts_and_keeps_incomplete_reports(self):
        report, log = self.run_cli(["--object-only", "--basic-object-features",
                                   "--object-cut-priority", "type_id", "pt"])
        self.assertEqual(report["configuration"]["object_cut_priority"], ["type_id", "pt"])
        self.assertIn("OBJECT CUT PRIORITY: type_id -> pt -> unrestricted", log)
        self.assertIn("incomplete rules eligible", log)
        records = report["results"]["rules"]
        self.assertTrue(records)
        self.assertTrue(any(r["object_cut_priority_remaining"] == ["pt"] for r in records))
        for record in records:
            names = [s["cut"]["observable"] for s in record["steps"]]
            self.assertEqual(names[:2], ["type_id", "pt"][:len(names)])
        self.assertTrue(any(not r["object_cut_priority_remaining"] for r in records))

    def test_object_cut_priority_in_combined_mode_leaves_events_unrestricted(self):
        report, _ = self.run_cli(["--object-cut-priority", "type_id", "pt"])
        records = report["results"]["rules"]
        self.assertTrue(any(all(s["scope"] == "event" for s in r["steps"]) for r in records))
        for record in records:
            names = [s["cut"]["observable"] for s in record["steps"] if s["scope"] == "slot"]
            self.assertEqual(names[:2], ["type_id", "pt"][:len(names)])
        self.assertEqual(load_report(self.report)["configuration"]["object_cut_priority"],
                         ["type_id", "pt"])

    def test_repeated_priority_entries_are_preserved_in_incomplete_reports(self):
        priority = ["type_id", "type_id", "pt"]
        report, _ = self.run_cli(["--object-only", "--basic-object-features",
                                 "--object-cut-priority", *priority])
        self.assertEqual(report["configuration"]["object_cut_priority"], priority)
        records = report["results"]["rules"]
        self.assertTrue(any(r["object_cut_priority_remaining"] == ["type_id", "pt"]
                            for r in records))
        self.assertTrue(any(not r["object_cut_priority_remaining"] for r in records))
        for record in records:
            names = [s["cut"]["observable"] for s in record["steps"]]
            self.assertEqual(names[:3], priority[:len(names)])
            self.assertEqual(record["object_cut_priority_remaining"], priority[len(names):])

    def test_directed_priority_cli_tracks_uncompleted_upper_bound(self):
        priority = ["pt:>", "pt:<="]
        report, log = self.run_cli(["--object-only", "--basic-object-features",
                                   "--object-cut-priority", *priority])
        self.assertEqual(report["configuration"]["object_cut_priority"], priority)
        self.assertIn("pt:> -> pt:<=", log)
        self.assertTrue(report["results"]["rules"])
        for record in report["results"]["rules"]:
            # Fixture has only PT=5 and PT=50, so each lower bound leaves
            # constant PT and cannot generate an upper-bound refinement.
            self.assertEqual(record["depth"], 1)
            self.assertEqual(record["steps"][0]["cut"]["comparison"], ">")
            self.assertEqual(record["object_cut_priority_remaining"], ["pt:<="])

    def test_completion_requirement_is_final_only_and_audited(self):
        priority = ["type_id", "type_id", "pt"]
        baseline, _ = self.run_cli(["--object-only", "--basic-object-features",
                                    "--object-cut-priority", *priority])
        required, log = self.run_cli(["--object-only", "--basic-object-features",
                                      "--object-cut-priority", *priority, "--require-complete-priority"],
                                     report=self.directory / "complete.txt")
        before, after = baseline["results"]["rules"], required["results"]["rules"]
        self.assertEqual([(r["raw_rule"], r["score"]) for r in before],
                         [(r["raw_rule"], r["score"]) for r in after])
        self.assertTrue(required["configuration"]["require_complete_priority"])
        self.assertIn("complete list required for final retention", log)
        self.assertGreater(required["summary"]["incomplete_priority_rules_removed"], 0)
        self.assertTrue(any(r["retained_after_final_pruning"] for r in after))
        for record in after:
            self.assertEqual(record["object_cut_priority_complete"], not record["object_cut_priority_remaining"])
            if not record["object_cut_priority_complete"]:
                self.assertEqual(record["pruning"]["status"], "incomplete_object_priority")
            if record["retained_after_final_pruning"]:
                self.assertTrue(record["object_cut_priority_complete"])

    def test_completion_can_return_empty_report_when_directed_prefix_cannot_extend(self):
        report, _ = self.run_cli(["--object-only", "--basic-object-features",
                                  "--object-cut-priority", "pt:>", "pt:<=", "--require-complete-priority"])
        self.assertTrue(report["results"]["rules"])
        self.assertEqual(report["summary"]["pruned_rule_count"], 0)
        self.assertTrue(all(r["pruning"]["status"] == "incomplete_object_priority"
                            for r in report["results"]["rules"]))

    def test_completion_with_empty_priority_is_a_noop(self):
        report, _ = self.run_cli(["--event-only", "--require-complete-priority"])
        self.assertEqual(report["configuration"]["object_cut_priority"], [])
        self.assertEqual(report["summary"]["incomplete_priority_rules_removed"], 0)
        self.assertTrue(all(r["object_cut_priority_complete"] for r in report["results"]["rules"]))

    def test_repeated_priority_reporting_does_not_count_event_cuts(self):
        priority = ["pt", "pt"]
        report, _ = self.run_cli(["--basic-object-features", "--object-cut-priority", *priority])
        for record in report["results"]["rules"]:
            names = [s["cut"]["observable"] for s in record["steps"] if s["scope"] == "slot"]
            self.assertEqual(names[:2], priority[:len(names)])
            self.assertEqual(record["object_cut_priority_remaining"], priority[len(names):])
        self.assertTrue(any(all(s["scope"] == "event" for s in r["steps"])
                            and r["object_cut_priority_remaining"] == priority
                            for r in report["results"]["rules"]))

    def test_object_cut_priority_errors_before_training_and_writing_reports(self):
        for extra in (("--event-only", "--object-cut-priority", "pt"),
                      ("--object-cut-priority", "missing"),
                      ("--object-cut-priority", "charge_valid"),
                      ("--basic-object-features", "--object-cut-priority", "charge"),
                      ("--object-cut-priority",)):
            with self.subTest(extra=extra), patch.object(runner.ExistentialRuleList, "fit") as fit:
                self.assert_cli_error(extra)
                fit.assert_not_called()

    def test_default_has_no_object_cut_priority(self):
        report, log = self.run_cli(["--object-only"])
        self.assertEqual(report["configuration"]["object_cut_priority"], [])
        self.assertNotIn("OBJECT CUT PRIORITY:", log)
        self.assertTrue(all(not r["object_cut_priority_remaining"]
                            for r in report["results"]["rules"]))

    def test_basic_loader_projects_values_schema_and_preserves_slot_mask(self):
        full = runner.load_dataset(self.signal)
        basic = runner.load_dataset(self.signal, basic_object_features=True)
        self.assertEqual([spec.name for spec in basic.object_observables],
                         list(runner.BASIC_OBJECT_COLUMNS))
        self.assertEqual(basic.objects.shape, (40, 22, 4))
        np.testing.assert_array_equal(basic.objects, full.objects[..., :4])
        np.testing.assert_array_equal(basic.slot_mask, full.slot_mask)
        np.testing.assert_array_equal(basic.event_features, full.event_features)
        self.assertEqual(basic.object_schema, full.object_schema[:4])
        self.assertEqual(basic.object_observables[3].categories, full.object_observables[3].categories)

    def test_basic_object_flag_keeps_event_features_in_combined_mode(self):
        report, log = self.run_cli(["--basic-object-features"])
        self.assertEqual(report["feature_mode"], "combined")
        self.assertEqual([spec["name"] for spec in report["object_observables"]],
                         list(runner.BASIC_OBJECT_COLUMNS))
        self.assertEqual(len(report["event_observables"]), 3)
        self.assertEqual(report["inputs"]["signal"]["object_shape"], [40, 22, 4])
        self.assertEqual(report["inputs"]["signal"]["event_feature_shape"], [40, 3])
        self.assertTrue(report["configuration"]["basic_object_features"])
        self.assertEqual(report["configuration"]["selected_object_columns"],
                         list(runner.BASIC_OBJECT_COLUMNS))
        self.assertIn("OBJECT FEATURES: basic only (pt, eta, phi, type_id)", log)
        for record in report["results"]["rules"]:
            for step in record["steps"]:
                if step["scope"] == "slot":
                    self.assertIn(step["cut"]["observable"], runner.BASIC_OBJECT_COLUMNS)

    def test_basic_object_only_matches_manually_projected_inputs(self):
        basic, _ = self.run_cli(["--object-only", "--basic-object-features"])
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["obj"] = raw["obj"][..., :4].clone()
            raw["obj_columns"] = raw["obj_columns"][:4]
            raw["obj_feature_schema"] = raw["obj_feature_schema"][:4]
            torch.save(raw, path)
        manual, _ = self.run_cli(["--object-only"], self.directory / "manual.txt")
        self.assertEqual(basic["results"], manual["results"])
        self.assertEqual(basic["summary"], manual["summary"])
        self.assertEqual(basic["obj_feature_schema"], manual["obj_feature_schema"])
        self.assertFalse(manual["configuration"]["basic_object_features"])

    def test_basic_object_features_are_selected_by_name_not_source_position(self):
        order = list(reversed(range(20)))
        raw = fixture(True)
        raw["obj"] = raw["obj"][..., order]
        raw["obj_columns"] = [raw["obj_columns"][i] for i in order]
        raw["obj_feature_schema"] = [raw["obj_feature_schema"][i] for i in order]
        torch.save(raw, self.signal)
        basic = runner.load_dataset(self.signal, basic_object_features=True)
        np.testing.assert_array_equal(basic.objects, fixture(True)["obj"][..., :4].numpy())
        report, _ = self.run_cli(["--object-only", "--basic-object-features"])
        self.assertTrue(report["results"]["rules"])

    def test_basic_mode_ignores_excluded_values_and_schema_differences(self):
        raw = fixture(True)
        raw["obj"][..., 4:] = float("nan")
        raw["obj_feature_schema"][4]["units"] = "different excluded definition"
        torch.save(raw, self.signal)
        report, _ = self.run_cli(["--object-only", "--basic-object-features"])
        self.assertTrue(report["results"]["rules"])
        self.assertEqual(len(report["obj_feature_schema"]), 4)

    def test_basic_mode_still_rejects_invalid_selected_values(self):
        for column, value in ((1, float("nan")), (3, 9)):
            raw = fixture(True)
            raw["obj"][0, 0, column] = value
            torch.save(raw, self.signal)
            with self.subTest(column=column):
                self.assert_cli_error(["--object-only", "--basic-object-features"])

    def test_basic_mode_requires_all_four_named_columns(self):
        for index, name in enumerate(runner.BASIC_OBJECT_COLUMNS):
            raw = fixture(True)
            raw["obj_columns"][index] = "renamed_" + name
            raw["obj_feature_schema"][index]["name"] = "renamed_" + name
            torch.save(raw, self.signal)
            with self.subTest(name=name):
                self.assert_cli_error(["--object-only", "--basic-object-features"])

    def test_basic_mode_is_supported_for_legacy_four_column_inputs(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["obj"] = raw["obj"][..., :4].clone()
            raw["obj_columns"] = raw["obj_columns"][:4]
            del raw["obj_feature_schema"]
            torch.save(raw, path)
        report, _ = self.run_cli(["--object-only", "--basic-object-features"])
        self.assertEqual(len(report["object_observables"]), 4)

    def test_basic_flag_conflicts_with_disabled_objects(self):
        self.assert_cli_error(["--basic-object-features", "--event-only"])
        with self.assertRaises(ValueError):
            runner.load_dataset("missing.pt", use_objects=False, basic_object_features=True)

    def test_basic_default_report_name_is_distinct(self):
        arguments = ["--signal", str(self.signal), "--background", str(self.background),
                     "--object-only", "--basic-object-features", "--print-top", "1",
                     "--print-reference-top", "0", "--print-pruned-top", "1"]
        with patch.object(runner.Path, "cwd", return_value=self.directory), redirect_stdout(io.StringIO()):
            runner.main(arguments)
        defaults = list((self.directory / "reports").glob("*.txt"))
        self.assertEqual(len(defaults), 1)
        self.assertTrue(defaults[0].name.endswith("_basic_objects_object_only.txt"))

    def test_full_runner_guards_object_measurements_and_reports_policy(self):
        # All mass flags are false in this fixture; changing placeholder mass
        # must not add a mass cut or affect scores and selected event counts.
        first, _ = self.run_cli(["--object-only"])
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["obj"][..., OBJECT_COLUMNS.index("mass")] = float("nan")
            torch.save(raw, path)
        changed, log = self.run_cli(["--object-only"], self.directory / "changed_fill.txt")
        self.assertEqual(first["results"], changed["results"])
        self.assertTrue(changed["validity_flags_are_automatic_guards"])
        self.assertEqual(changed["configuration"]["object_validity_guards"]["mass"], "mass_valid")
        self.assertIn("automatically guard cuts", log)
        for record in changed["results"]["rules"]:
            for step in record["steps"]:
                if step["cut"]["observable"] == "charge":
                    self.assertEqual(step["cut"]["validity_guard"]["observable"], "charge_valid")
                    self.assertIn("charge_valid == True", step["display"])

    def test_event_validity_metadata_is_enforced_by_both_erl_runners(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["event_feature_schema"][1]["definition"] = {"validity_feature": "met_valid"}
            raw["event_features"][:10, 1] = float("nan")
            raw["event_features"][:10, 2] = 0
            torch.save(raw, path)
        data = runner.load_dataset(self.signal, use_objects=False)
        self.assertEqual(data.event_observables[1].validity_column, "met_valid")
        new, _ = self.run_cli(["--event-only"])
        old, _ = self.run_cli(report=self.directory / "old_guarded.txt", main=event_main)
        keys = ("raw_rule", "score", "pruned_rank", "pruning")
        self.assertEqual([[r[k] for k in keys] for r in new["results"]["rules"]],
                         [[r[k] for k in keys] for r in old["results"]["rules"]])
        self.assertEqual(new["configuration"]["event_validity_guards"], {"met": "met_valid"})
        self.assertTrue(old["validity_flags_are_automatic_guards"])

    def test_valid_measurements_and_flag_values_still_require_correct_values(self):
        for key, index, value in (("obj", (0, 0, OBJECT_COLUMNS.index("charge")), float("nan")),
                                  ("obj", (0, 0, OBJECT_COLUMNS.index("charge_valid")), .5)):
            raw = fixture(True)  # charge_valid is True for this signal row.
            raw[key][index] = value
            torch.save(raw, self.signal)
            with self.subTest(index=index):
                self.assert_cli_error()

    def test_malformed_and_incompatible_validity_metadata_is_rejected(self):
        for guard in ("not_a_column", "pt", "mass", "", False):
            raw = fixture(True)
            raw["obj_feature_schema"][OBJECT_COLUMNS.index("mass")]["validity_column"] = guard
            torch.save(raw, self.signal)
            with self.subTest(guard=guard):
                self.assert_cli_error()

    def test_guard_only_policy_and_cli_opt_in_apply_to_both_scopes(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["event_feature_schema"][1]["definition"] = {"validity_feature": "met_valid"}
            raw["event_features"][:5, 2] = 0
            torch.save(raw, path)
        default, log = self.run_cli()
        self.assertTrue(default["validity_flags_are_guard_only"])
        self.assertFalse(default["configuration"]["allow_validity_cuts"])
        self.assertIn("guard-only; standalone True/False cuts disabled", log)
        obj_flags = set(default["configuration"]["object_validity_guards"].values())
        evt_flags = set(default["configuration"]["event_validity_guards"].values())
        for record in default["results"]["rules"]:
            for step in record["steps"]:
                excluded = obj_flags if step["scope"] == "slot" else evt_flags
                self.assertNotIn(step["cut"]["observable"], excluded)
        opted, log = self.run_cli(["--allow-validity-cuts"], self.directory / "opt_in.txt")
        self.assertFalse(opted["validity_flags_are_guard_only"])
        self.assertTrue(opted["configuration"]["allow_validity_cuts"])
        self.assertIn("standalone cuts allowed (explicit opt-in)", log)
        self.assertTrue(any(step["cut"]["observable"] in
            (obj_flags if step["scope"] == "slot" else evt_flags)
            for record in opted["results"]["rules"] for step in record["steps"]))

    def test_native_event_runner_exposes_the_same_validity_policy(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["event_feature_schema"][1]["definition"] = {"validity_feature": "met_valid"}
            raw["event_features"][:5, 2] = 0
            torch.save(raw, path)
        for options in ([], ["--allow-validity-cuts"]):
            new, _ = self.run_cli(["--event-only", *options], self.directory / f"new{len(options)}.txt")
            old, _ = self.run_cli(options, self.directory / f"old{len(options)}.txt", main=event_main)
            self.assertEqual(new["validity_flags_are_guard_only"], not bool(options))
            self.assertEqual(old["validity_flags_are_guard_only"], not bool(options))
            keys = ("raw_rule", "score", "pruned_rank", "pruning")
            self.assertEqual([[r[k] for k in keys] for r in new["results"]["rules"]],
                             [[r[k] for k in keys] for r in old["results"]["rules"]])

    def test_disabling_both_is_rejected_before_opening_inputs(self):
        self.assert_cli_error(["--event-only", "--object-only"])
        with self.assertRaises(ValueError):
            runner.load_dataset("missing.pt", use_objects=False, use_events=False)

    def test_disabled_fields_can_be_absent_or_malformed(self):
        for event_only in (True, False):
            for malformed in (False, True):
                for path, signal in ((self.signal, True), (self.background, False)):
                    raw = fixture(signal)
                    keys = ("obj", "obj_columns", "obj_feature_schema") if event_only else (
                        "event_features", "event_feature_columns", "event_feature_schema")
                    for key in keys:
                        if malformed:
                            raw[key] = "ignored malformed field"
                        else:
                            del raw[key]
                    torch.save(raw, path)
                report, _ = self.run_cli(["--event-only" if event_only else "--object-only"],
                                         self.directory / f"disabled_{event_only}_{malformed}.txt")
                self.assertTrue(report["results"]["rules"])

    def test_missing_enabled_fields_and_invalid_labels_are_rejected(self):
        for key in ("obj", "obj_columns", "obj_feature_schema", "event_features", "event_feature_columns"):
            raw = fixture(True)
            del raw[key]
            torch.save(raw, self.signal)
            with self.subTest(key=key):
                self.assert_cli_error()
        for labels in (torch.zeros(40), torch.zeros((40, 1), dtype=torch.int64), torch.zeros(0, dtype=torch.int64)):
            raw = fixture(True)
            raw["label"] = labels
            torch.save(raw, self.signal)
            self.assert_cli_error()

    def test_invalid_active_features_are_rejected_not_zero_filled(self):
        for key, index, value in (("obj", (0, 0, 1), float("nan")),
                                   ("obj", (0, 0, 3), 9),
                                   ("obj", (0, 0, 12), .5),
                                   ("event_features", (0, 1), float("inf")),
                                   ("event_features", (0, 0), .5),
                                   ("event_features", (0, 0), -1)):
            raw = fixture(True)
            raw[key][index] = value
            torch.save(raw, self.signal)
            with self.subTest(key=key, index=index, value=value):
                self.assert_cli_error()

    def test_padded_slots_do_not_enter_search_or_validation(self):
        raw = fixture(True)
        raw["obj"][:, 1:, 1] = float("nan")
        torch.save(raw, self.signal)
        data = runner.load_dataset(self.signal)
        self.assertEqual(int(data.slot_mask.sum()), 40)
        report, _ = self.run_cli(["--object-only"])
        self.assertTrue(report["results"]["rules"])

    def test_rows_widths_order_and_definitions_must_match(self):
        variants = []
        raw = fixture(True)
        raw["obj"] = raw["obj"][:-1]
        variants.append(raw)
        raw = fixture(True)
        raw["obj"] = raw["obj"][:, :21]
        variants.append(raw)
        raw = fixture(True)
        raw["obj_columns"][1] = "phi"
        variants.append(raw)
        raw = fixture(True)
        raw["obj_feature_schema"][0]["units"] = "MeV"
        variants.append(raw)
        raw = fixture(True)
        raw["event_feature_schema"][1]["units"] = "MeV"
        variants.append(raw)
        raw = fixture(True)
        raw["event_features"] = raw["event_features"][:, :2]
        variants.append(raw)
        for i, raw in enumerate(variants):
            torch.save(raw, self.signal)
            with self.subTest(variant=i):
                self.assert_cli_error()

    def test_legacy_four_object_columns_are_supported_without_schema(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["obj"] = raw["obj"][:, :, :4].clone()
            raw["obj_columns"] = list(OBJECT_COLUMNS[:4])
            del raw["obj_feature_schema"]
            torch.save(raw, path)
        report, _ = self.run_cli(["--object-only"])
        self.assertEqual(len(report["object_observables"]), 4)

    def test_slot_mask_uses_named_pt_column_not_fixed_index(self):
        raw = fixture(True)
        order = [1, 0, *range(2, 20)]
        raw["obj"] = raw["obj"][:, :, order]
        raw["obj_columns"] = [raw["obj_columns"][i] for i in order]
        raw["obj_feature_schema"] = [raw["obj_feature_schema"][i] for i in order]
        torch.save(raw, self.signal)
        self.assertEqual(int(runner.load_dataset(self.signal).slot_mask.sum()), 40)

    def test_missing_named_pt_and_bad_object_shapes_are_rejected(self):
        raw = fixture(True)
        raw["obj_columns"][0] = "momentum"
        raw["obj_feature_schema"][0]["name"] = "momentum"
        torch.save(raw, self.signal)
        self.assert_cli_error()
        for tensor in (torch.zeros(40, 20), torch.zeros(40, 0, 20),
                       torch.zeros(40, 22, 20, dtype=torch.complex64)):
            raw = fixture(True)
            raw["obj"] = tensor
            torch.save(raw, self.signal)
            self.assert_cli_error()

    def test_schema_less_event_multiplicities_are_supported(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["event_features"] = raw["event_features"][:, :1].clone()
            raw["event_feature_columns"] = raw["event_feature_columns"][:1]
            del raw["event_feature_schema"]
            torch.save(raw, path)
        report, _ = self.run_cli(["--event-only"])
        self.assertEqual(report["event_observables"][0]["kind"], "ordinal")
        self.assertTrue(report["results"]["rules"])

    def test_penalty_and_current_search_defaults(self):
        report, _ = self.run_cli(["--reference-penalty", "10", "--stricter-score-improvement", ".01"])
        config = report["configuration"]
        for key, value in (("max_rule_depth", 3), ("beam_width", 50), ("max_candidate_rules", 500),
                           ("max_thresholds", 32), ("min_signal_efficiency", .01),
                           ("reference_penalty", 10), ("stricter_score_improvement", .01)):
            self.assertEqual(config[key], value)
        self.assertFalse(config["beam_similarity_filter"])
        self.assertFalse(config["final_similarity_pruning"])
        for r in report["results"]["rules"]:
            self.assertAlmostEqual(r["score"], r["gini_gain"] - 10 * r["metrics"]["background_acceptance"])

    def test_rate_filter_applies_to_object_and_event_rules(self):
        original, _ = self.run_cli()
        limited, _ = self.run_cli(["--max-trigger-rate-khz", "0"], self.directory / "rate.txt")
        self.assertEqual([r["raw_rule"] for r in original["results"]["rules"]],
                         [r["raw_rule"] for r in limited["results"]["rules"]])
        rejected = []
        for r in limited["results"]["rules"]:
            if r["retained_after_final_pruning"]:
                self.assertEqual(r["metrics"]["background_accepted"], 0)
            if r["pruning"]["status"] == "reference_rate_limit":
                rejected.append(any(step["scope"] == "slot" for step in r["steps"]))
        self.assertEqual(set(rejected), {False, True})

    def test_final_similarity_opt_in_does_not_change_search(self):
        original, _ = self.run_cli()
        filtered, _ = self.run_cli(["--final-similarity-filter"], self.directory / "similarity.txt")
        self.assertEqual([r["raw_rule"] for r in original["results"]["rules"]],
                         [r["raw_rule"] for r in filtered["results"]["rules"]])
        self.assertGreater(filtered["summary"]["similarity_rules_removed"], 0)
        self.assertLess(filtered["summary"]["pruned_rule_count"], original["summary"]["pruned_rule_count"])

    def test_beam_similarity_opt_in_and_thresholds_are_recorded(self):
        report, _ = self.run_cli(["--beam-similarity-filter", "--signal-similarity-threshold", ".95",
                                 "--background-similarity-threshold", ".9"])
        config = report["configuration"]
        self.assertTrue(config["beam_similarity_filter"])
        self.assertEqual(config["beam_signal_similarity_threshold"], .95)
        self.assertEqual(config["beam_background_similarity_threshold"], .9)

    def test_stored_labels_do_not_change_search(self):
        original, _ = self.run_cli()
        raw = fixture(True)
        raw["label"].fill_(123)
        torch.save(raw, self.signal)
        changed, _ = self.run_cli(report=self.directory / "labels.txt")
        self.assertEqual([(r["raw_rule"], r["score"]) for r in original["results"]["rules"]],
                         [(r["raw_rule"], r["score"]) for r in changed["results"]["rules"]])

    def test_reports_are_plot_compatible(self):
        for flag in ((), ("--event-only",), ("--object-only",)):
            target = self.directory / f"plot_{len(flag)}_{flag}.txt"
            report, _ = self.run_cli(flag, target)
            loaded = load_report(target)
            eligible, selected = select_paths(loaded, rules="pruned", direction="mixture", top=3)
            self.assertTrue(selected)
            self.assertTrue(all(r["retained_after_final_pruning"] for r in eligible))
            self.assertEqual(len(loaded["results"]["rules"]), len(report["results"]["rules"]))

    def test_report_overwrite_and_input_aliases_are_protected(self):
        self.report.write_text("keep existing report")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(self.args())
        self.assertEqual(self.report.read_text(), "keep existing report")
        self.run_cli(["--overwrite-report"])
        aliases = [self.signal, self.directory / "hardlink.txt", self.directory / "symlink.txt"]
        os.link(self.signal, aliases[1])
        aliases[2].symlink_to(self.signal)
        original = self.signal.read_bytes()
        for target in aliases:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(self.args(["--overwrite-report"], target))
        self.assertEqual(self.signal.read_bytes(), original)

    def test_same_input_file_and_alias_are_rejected(self):
        alias = self.directory / "alias.pt"
        os.link(self.signal, alias)
        for background in (self.signal, alias):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(["--signal", str(self.signal), "--background", str(background),
                             "--report", str(self.report)])
        self.assertFalse(self.report.exists())

    def test_invalid_configuration_is_rejected_without_reports(self):
        for extra in (("--reference-penalty", "-1"), ("--stricter-score-improvement", "nan"),
                      ("--input-rate-mhz", "0"),
                      ("--max-trigger-rate-khz", "inf"), ("--signal-similarity-threshold", "2"),
                      ("--nested-score-gap", "0")):
            with self.subTest(extra=extra):
                self.assert_cli_error(extra)

    def test_negative_stricter_threshold_is_accepted_saved_and_used(self):
        captured = []
        original = runner.ExistentialRuleList.prune_candidates

        def prune(model, **kwargs):
            captured.append(kwargs["stricter_score_improvement"])
            return original(model, **kwargs)

        with patch.object(runner.ExistentialRuleList, "prune_candidates", prune):
            report, _ = self.run_cli(["--object-only", "--stricter-score-improvement", "-0.008"])
        self.assertEqual(captured, [-.008])
        self.assertEqual(report["configuration"]["stricter_score_improvement"], -.008)

    def test_constant_features_can_produce_empty_candidate_report(self):
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = fixture(signal)
            raw["event_features"][:] = torch.tensor([0, 0, 0])
            raw["event_feature_schema"] = [
                {"name": name, "kind": "continuous"} for name in raw["event_feature_columns"]]
            torch.save(raw, path)
        report, _ = self.run_cli(["--event-only"])
        self.assertEqual(report["results"]["rules"], [])
        self.assertEqual(report["summary"]["pruned_rule_count"], 0)

    def test_script_runs_from_outside_repository_and_default_name_contains_mode(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "casual_tests/run_ERL_Gini_nested.py"),
             *self.args(["--object-only"])], cwd=self.directory,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = ["--signal", str(self.signal), "--background", str(self.background),
                     "--event-only", "--print-top", "1", "--print-reference-top", "0",
                     "--print-pruned-top", "1"]
        with patch.object(runner.Path, "cwd", return_value=self.directory), redirect_stdout(io.StringIO()):
            runner.main(arguments)
        defaults = list((self.directory / "reports").glob("*.txt"))
        self.assertEqual(len(defaults), 1)
        self.assertTrue(defaults[0].name.endswith("_event_only.txt"))


if __name__ == "__main__":
    unittest.main()
