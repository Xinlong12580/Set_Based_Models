"""Smoke tests for the four structured-report runners."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class QuantitativeRunnerTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.object_paths = self._write_pair("object", with_events=False)
        self.event_paths = self._write_pair("event", with_events=True)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_pair(self, name, *, with_events):
        paths = []
        for role, count, label in (("signal", 40, 5), ("background", 60, 1)):
            obj = self.torch.zeros((count, 23, 4), dtype=self.torch.float32)
            obj[:, 0, 0] = 10
            if not with_events and role == "signal":
                obj[:, 1] = self.torch.tensor([30.0, 0.0, 0.0, 2.0])
            dataset = {
                "obj": obj,
                "label": self.torch.full((count,), label, dtype=self.torch.int64),
                "obj_columns": ["pt", "eta", "phi", "type_id"],
            }
            if with_events:
                events = self.torch.zeros((count, 4), dtype=self.torch.int64)
                events[:, 2] = 2 if role == "signal" else 0
                dataset["event_features"] = events
                dataset["event_feature_columns"] = [
                    "n_electrons",
                    "n_muons",
                    "n_photons",
                    "n_jets",
                ]
            path = self.root / f"{name}_{role}.pt"
            self.torch.save(dataset, path)
            paths.append(path)
        return paths

    def _run_and_load(self, main, paths, report_name, extra=()):
        report = self.root / report_name
        args = [
            "--signal",
            str(paths[0]),
            "--background",
            str(paths[1]),
            "--report",
            str(report),
            "--print-top",
            "1",
            *extra,
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            main(args)
        with report.open(encoding="utf-8") as handle:
            parsed = json.load(handle)
        self.assertEqual(parsed["format"], "set_based_models_quantitative_report")
        self.assertEqual(parsed["schema_version"], 1)
        self.assertIn("Structured report saved to", output.getvalue())
        return parsed, output.getvalue()

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_object_edt_report(self):
        from casual_tests.stale.run_EDT_quantitative import main

        report, output = self._run_and_load(main, self.object_paths, "object_edt.txt")
        self.assertEqual(report["model"], "EDT")
        self.assertFalse(report["event_observables"])
        self.assertTrue(report["results"]["leaves"])
        self.assertIn("selected_signal_stored_label_counts", report["results"]["leaves"][0]["metrics"])
        self.assertIn("LEAVES RANKED BY MIXTURE FRACTION", output)

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_event_edt_report(self):
        from casual_tests.stale.run_EDT_with_event_features_quantitative import main

        report, output = self._run_and_load(main, self.event_paths, "event_edt.txt")
        self.assertEqual([item["name"] for item in report["event_observables"]], [
            "n_electrons", "n_muons", "n_photons", "n_jets",
        ])
        paths = [step["selected_condition"] for leaf in report["results"]["leaves"] for step in leaf["path"]]
        self.assertTrue(any("n_photons" in condition for condition in paths))
        self.assertIn("event: n_photons", output)

    def test_object_erl_report(self):
        from casual_tests.stale.run_ERL_Gini_quantitative import main

        report, output = self._run_and_load(
            main,
            self.object_paths,
            "object_erl.txt",
            ("--print-reference-top", "0", "--print-pruned-top", "1"),
        )
        self.assertEqual(report["model"], "ERL_Gini")
        rules = report["results"]["rules"]
        self.assertTrue(rules)
        self.assertIn("effective_depth_after_simplification", rules[0])
        self.assertIn("pruned_rank", rules[0])
        self.assertIn("TOP MIXTURE-ENRICHED RULES", output)

    def test_event_erl_report(self):
        from casual_tests.stale.run_ERL_Gini_with_event_features_quantitative import main

        report, output = self._run_and_load(
            main,
            self.event_paths,
            "event_erl.txt",
            ("--print-reference-top", "1", "--print-pruned-top", "1"),
        )
        rules = report["results"]["rules"]
        self.assertTrue(any("n_photons" in rule["simplified_rule"] for rule in rules))
        self.assertTrue(any(step["scope"] == "event" for rule in rules for step in rule["steps"]))
        self.assertIn("n_photons >= 2", output)


if __name__ == "__main__":
    unittest.main()
