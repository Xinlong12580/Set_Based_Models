"""Synthetic checks for event-only runners; no production data is read."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class EventOnlyRunnerTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        temporary = tempfile.TemporaryDirectory(prefix="event_only_runner_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.signal = self.root / "anomalous.pt"
        self.background = self.root / "reference.pt"
        self.columns = ["n_photons", "n_photons_pt30", "n_jets"]
        self.schema = [{"name": name, "kind": "ordinal", "units": "count",
                        "definition": {"comparison": ">", "pt_threshold_GeV": threshold}}
                       for name, threshold in zip(self.columns, (0, 30, 0))]
        self.save(self.signal, True)
        self.save(self.background, False)

    def dataset(self, signal):
        # Mimic 10% injected signal within an otherwise reference-like mixture.
        count = 200
        events = self.torch.zeros((count, 3), dtype=self.torch.int64)
        events[:, 2] = 1
        labels = self.torch.ones(count, dtype=self.torch.int64)
        if signal:
            events[:20, :2] = 2
            labels[:20] = 5
        return {
            "event_features": events, "event_feature_columns": list(self.columns),
            "event_feature_schema": self.schema, "label": labels,
            # Native scalar info is deliberately not included in the search.
            "native_event_info": self.torch.full((count, 10), float("nan")),
        }

    def save(self, path, signal, update=None):
        dataset = self.dataset(signal)
        if update is not None:
            update(dataset)
        self.torch.save(dataset, path)

    def run_report(self, main, name, extra=()):
        target = self.root / name
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--signal", str(self.signal), "--background", str(self.background),
                  "--report", str(target), "--print-top", "1", *extra])
        return json.loads(target.read_text()), output.getvalue(), target

    def assert_event_only_report(self, report):
        self.assertEqual(report["format"], "set_based_models_quantitative_report")
        self.assertEqual(report["feature_mode"], "event_only")
        self.assertEqual(report["object_observables"], [])
        self.assertFalse(report["configuration"]["object_features_enabled"])
        self.assertFalse(report["native_event_info_used"])
        native = report.get("native_event_only", False)
        self.assertEqual(report["inputs"]["signal"]["object_shape"],
                         [200, 0, 0] if native else [200, 1, 1])
        if native:
            self.assertIsNone(report["dummy_object_input"])
        self.assertEqual(report["inputs"]["signal"]["stored_label_counts"], {"1": 180, "5": 20})

    def test_loader_requires_no_objects_and_builds_fully_masked_dummy(self):
        from casual_tests._event_only_reporting import load_event_dataset
        data = load_event_dataset(self.signal)
        self.assertEqual(data.objects.shape, (200, 1, 1))
        self.assertTrue(bool((data.objects == 0).all()))
        self.assertEqual(data.slot_mask.shape, (200, 1))
        self.assertFalse(bool(data.slot_mask.any()))

    def test_native_erl_loader_creates_no_dummy_slots(self):
        from casual_tests._event_only_reporting import load_event_dataset
        data = load_event_dataset(self.signal, native_event_only=True)
        self.assertEqual(data.objects.shape, (200, 0, 0))
        self.assertEqual(data.slot_mask.shape, (200, 0))
        self.assertEqual(data.event_columns, tuple(self.columns))
        self.assertEqual(data.event_features.shape, (200, 3))

    def test_real_objects_are_ignored_even_if_malformed(self):
        from casual_tests._event_only_reporting import load_event_dataset
        self.save(self.signal, True, lambda raw: raw.update(obj="not a tensor", obj_columns=None))
        self.assertEqual(load_event_dataset(self.signal).objects.shape, (200, 1, 1))

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_edt_finds_rare_event_signature_and_reports_only_event_paths(self):
        from casual_tests.run_EDT_event_only import main
        report, output, target = self.run_report(main, "edt.txt")
        self.assert_event_only_report(report)
        self.assertEqual(report["configuration"]["max_depth"], 6)
        self.assertEqual(report["configuration"]["min_samples_leaf"], 20)
        leaves = report["results"]["leaves"]
        self.assertTrue(any(leaf["metrics"]["signal_accepted"] == 20 and
                            leaf["metrics"]["background_accepted"] == 0 for leaf in leaves))
        self.assertTrue(all(step["scope"] == "event" and step["cut"]["scope"] == "event"
                            for leaf in leaves for step in leaf["path"]))
        self.assertIn("dummy slots all masked out", output)
        self.assertIn("180", output)
        from scripts.plot_acceptance_report import load_report, select_paths
        plotted = load_report(target)
        self.assertTrue(select_paths(plotted, rules="all", direction="mixture", top=10)[1])

    def test_erl_finds_rare_event_signature_and_preserves_search_settings(self):
        from casual_tests.stale.run_ERL_Gini_event_only import main
        report, output, target = self.run_report(
            main, "erl.txt", ("--print-reference-top", "0", "--print-pruned-top", "1"))
        self.assert_event_only_report(report)
        config = report["configuration"]
        self.assertEqual((config["max_rule_depth"], config["beam_width"], config["max_thresholds"]),
                         (3, 40, 32))
        self.assertEqual(config["beam_signal_similarity_threshold"], 0.2)
        rules = report["results"]["rules"]
        self.assertTrue(any(rule["metrics"]["signal_accepted"] == 20 and
                            rule["metrics"]["background_accepted"] == 0 for rule in rules))
        self.assertTrue(all(step["scope"] == "event" and step["cut"]["scope"] == "event"
                            for rule in rules for step in rule["steps"]))
        self.assertNotIn("EXISTS", output)
        self.assertNotIn("unused_dummy", output)
        from scripts.plot_acceptance_report import load_report, select_paths
        self.assertTrue(select_paths(load_report(target), rules="pruned", direction="mixture", top=10)[1])

    def test_current_twenty_feature_converter_output_loads(self):
        from casual_tests._event_only_reporting import load_event_dataset
        from data_conversion.stage1_to_pt import build_stage1_dataset
        from data_conversion.tests.test_event_conversion import stage1_fixture
        self.torch.save(build_stage1_dataset(stage1_fixture()), self.signal)
        loaded = load_event_dataset(self.signal)
        self.assertEqual(loaded.event_features.shape, (2, 20))
        self.assertFalse(bool(loaded.slot_mask.any()))

    def test_legacy_counts_without_schema_are_supported(self):
        from casual_tests._event_only_reporting import load_event_dataset
        self.save(self.signal, True, lambda raw: raw.pop("event_feature_schema"))
        self.assertEqual(len(load_event_dataset(self.signal).event_observables), 3)

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_uninformative_events_generate_no_object_cuts_despite_object_contrast(self):
        from casual_tests.run_EDT_event_only import main as edt_main
        from casual_tests.stale.run_ERL_Gini_event_only import main as erl_main
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = self.dataset(signal)
            raw["event_features"][:] = 0
            raw["obj"] = self.torch.full((200, 23, 4), 1000. if signal else 1.)
            raw["obj_columns"] = ["pt", "eta", "phi", "type_id"]
            self.torch.save(raw, path)
        edt, _, _ = self.run_report(edt_main, "constant_edt.txt")
        erl, _, _ = self.run_report(erl_main, "constant_erl.txt")
        self.assertEqual(edt["summary"]["node_count"], 1)
        self.assertEqual(edt["results"]["leaves"][0]["path"], [])
        self.assertEqual(erl["summary"]["candidate_rule_count"], 0)
        self.assertEqual(erl["results"]["rules"], [])

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_schema_continuous_extensions_are_not_rounded_to_counts(self):
        from casual_tests.run_EDT_event_only import main
        for path, signal in ((self.signal, True), (self.background, False)):
            raw = self.dataset(signal)
            values = self.torch.full((200, 1), 0.25, dtype=self.torch.float64)
            if signal:
                values[:20] = 2.75
            raw.update(event_features=values, event_feature_columns=["activity"],
                       event_feature_schema=[{"name": "activity", "kind": "continuous", "units": "GeV"}])
            self.torch.save(raw, path)
        report, _, _ = self.run_report(main, "continuous.txt")
        self.assertEqual(report["event_observables"][0]["kind"], "continuous")
        self.assertTrue(any(step["cut"]["threshold"] == 1.5
                            for leaf in report["results"]["leaves"] for step in leaf["path"]))

    def test_invalid_event_inputs_are_rejected(self):
        from casual_tests._event_only_reporting import load_event_dataset
        cases = [
            lambda raw: raw.pop("event_features"),
            lambda raw: raw.update(event_feature_columns=["x", "x", "y"]),
            lambda raw: raw.update(event_feature_columns=["n_photons"]),
            lambda raw: raw.update(label=self.torch.ones(3)),
            lambda raw: raw.update(event_features=self.torch.zeros((0, 3))),
            lambda raw: raw.update(event_features=self.torch.zeros((200, 0))),
            lambda raw: raw.update(event_features=self.torch.full((200, 3), float("nan"))),
            lambda raw: raw.update(event_features=self.torch.full((200, 3), 0.5)),
            lambda raw: raw.update(event_features=self.torch.full((200, 3), -1)),
            lambda raw: raw.update(event_features=self.torch.zeros((200, 3), dtype=self.torch.complex64)),
            lambda raw: raw.update(event_feature_schema=[]),
            lambda raw: raw.update(event_feature_schema=[{"name": "wrong", "kind": "ordinal"}] * 3),
        ]
        for index, update in enumerate(cases):
            with self.subTest(index=index):
                self.save(self.signal, True, update)
                with self.assertRaises((ValueError, TypeError)):
                    load_event_dataset(self.signal)

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_schema_mismatch_is_rejected_before_training(self):
        from casual_tests.run_EDT_event_only import main
        raw = self.dataset(False)
        raw["event_feature_schema"][1]["definition"]["pt_threshold_GeV"] = 50
        self.torch.save(raw, self.background)
        target = self.root / "wrong_schema.txt"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--signal", str(self.signal), "--background", str(self.background),
                  "--report", str(target)])
        self.assertFalse(target.exists())

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_existing_report_and_inputs_are_protected(self):
        from casual_tests.run_EDT_event_only import main
        target = self.root / "existing.txt"
        target.write_text("keep this")
        for output, extra in ((target, []), (self.signal, ["--overwrite-report"])):
            before = output.read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--signal", str(self.signal), "--background", str(self.background),
                      "--report", str(output), *extra])
            self.assertEqual(output.read_bytes(), before)
        alias = self.root / "alias.txt"
        alias.symlink_to(target)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--signal", str(self.signal), "--background", str(self.background),
                  "--report", str(alias), "--overwrite-report"])
        self.assertEqual(target.read_text(), "keep this")

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_identical_inputs_are_rejected(self):
        from casual_tests.run_EDT_event_only import main
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--signal", str(self.signal), "--background", str(self.signal)])

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_hardlink_aliases_cannot_be_used_as_background_or_report(self):
        from casual_tests.run_EDT_event_only import main
        alias = self.root / "alias.pt"
        os.link(self.signal, alias)
        before = self.signal.read_bytes()
        for background, target in ((alias, self.root / "output.txt"), (self.background, alias)):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--signal", str(self.signal), "--background", str(background),
                      "--report", str(target), "--overwrite-report"])
        self.assertEqual(self.signal.read_bytes(), before)

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_default_report_name_marks_event_only_mode(self):
        from casual_tests.run_EDT_event_only import main
        target = self.root / "reports" / f"EDT_{self.root.name}_anomalous_event_only.txt"
        with patch("casual_tests._event_only_reporting.Path.cwd", return_value=self.root), \
                redirect_stdout(io.StringIO()):
            main(["--signal", str(self.signal), "--background", str(self.background), "--print-top", "1"])
        self.assertTrue(target.is_file())

    def test_old_reporting_globals_remain_unchanged(self):
        from casual_tests import _quantitative_reporting as old
        from casual_tests.stale.run_ERL_Gini_event_only import main
        before = old.OBSERVABLES
        self.run_report(main, "no_mutation.txt")
        self.assertEqual(old.OBSERVABLES, before)
        self.assertEqual([spec.name for spec in old.OBSERVABLES], ["pt", "eta", "phi", "type_id"])

    def test_public_scripts_run_from_outside_repository(self):
        scripts = ["run_ERL_Gini_event_only.py"]
        extension = importlib.util.find_spec("set_based_edt")
        env = os.environ.copy()
        if extension is not None:
            scripts.insert(0, "run_EDT_event_only.py")
            # A relative source PYTHONPATH would resolve against the child's cwd.
            extension_root = str(Path(extension.origin).resolve().parent.parent)
            env["PYTHONPATH"] = extension_root + os.pathsep + env.get("PYTHONPATH", "")
        for script in scripts:
            with self.subTest(script=script):
                target = self.root / f"{script}.txt"
                script_root = ROOT / "casual_tests"
                if script == "run_ERL_Gini_event_only.py":
                    script_root /= "stale"
                result = subprocess.run([
                    sys.executable, "-B", str(script_root / script),
                    "--signal", str(self.signal), "--background", str(self.background),
                    "--report", str(target), "--print-top", "1",
                ], cwd=self.root, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(target.read_text())["feature_mode"], "event_only")


if __name__ == "__main__":
    unittest.main()
