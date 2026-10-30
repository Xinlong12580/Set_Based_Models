from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import torch

from collide2v_trigger import Trigger, TriggerMenu, calibrate_menu, load_scores, parse_erl_model
from test_erl_model_parser import model
from test_erl_parser import data_with_schema, threshold


ROOT = Path(__file__).resolve().parents[2]


class CalibrationRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="trigger runners ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "events"
        self.work.mkdir()
        self.data = data_with_schema([{"met": value} for value in range(10, 110, 10)])
        self.train = self.write_events("train_minbias", self.data)
        self.test = self.write_events("test_minbias", data_with_schema([{"met": value} for value in range(15, 115, 10)]))
        self.write_events("test_SignalA", data_with_schema([{"met": value} for value in (15, 45, 65, 95)]))
        self.write_events("test_SignalB", data_with_schema([{"met": value} for value in (5, 55, 75, 85)]))
        phase2 = TriggerMenu(triggers=[self.reference(name, rate) for name, rate in
                                      (("Reference", 40), ("Discrete", 35), ("TooHigh", 200), ("Zero", 0))])
        self.database = self.root / "phase2.json"
        phase2.save(self.database)
        self.model = model(self.data, [[threshold("met", 10)], [threshold("met", 20)]], [7, 8])
        self.model_path = self.root / "model.json"
        self.menu_path = self.root / "erl.json"
        self.save_erl()
        self.phase2_dir = self.root / "phase2 calibration"
        self.match_dir = self.root / "matched"
        self.comparison_dir = self.root / "comparison"
        self.env = {**os.environ, "PYTHON_BIN": sys.executable, "PYTHONDONTWRITEBYTECODE": "1",
                    "SIG_WORK_DIR": str(self.work), "MINBIAS_TRAIN": str(self.train),
                    "MINBIAS_TEST": str(self.test), "PHASE2_DATABASE": str(self.database),
                    "PHASE2_CALIBRATION": str(self.phase2_dir), "NO_ISOLATION": "1",
                    "COLLISION_RATE_KHZ": "100", "BATCH_SIZE": "4", "MAX_EVENTS": "",
                    "ERL_MODEL": str(self.model_path), "ERL_MENU": str(self.menu_path),
                    "ERL_MATCH_DIR": str(self.match_dir), "COMPARISON_DIR": str(self.comparison_dir),
                    "SIGNALS": "SignalA SignalB", "LOWER": "0", "UPPER": "1000",
                    "SCORE_TOLERANCE": "0.01", "TOLERANCE": "0.01"}
        self.env.pop("PHASE2_MENU", None)

    def write_events(self, name, data):
        path = self.work / f"{name}.pt"
        torch.save(data, path)
        manifest = path.with_suffix(".chunks.json")
        manifest.write_text(json.dumps({"files": [{"path": path.name}]}))
        return manifest

    def reference(self, name, rate):
        return Trigger({"format": "collide2v_trigger", "schema_version": 1, "type": "selection",
                        "name": name, "default_thresholds": {"met": 50}, "target_rate_khz": rate,
                        "selection": {"op": "event", "field": "met", "comparison": ">=", "threshold": "$met"}})

    def save_erl(self):
        for index, record in enumerate(self.model["rules"]):
            record["steps"][-1]["generation_score"] = index + 1
        self.model_path.write_text(json.dumps(self.model))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            menu = TriggerMenu(triggers=parse_erl_model(self.model, self.data, allow_fixed=True))
        menu.save(self.menu_path)

    def make_phase2_report(self):
        with redirect_stdout(io.StringIO()):
            calibrate_menu(self.train, TriggerMenu.load(self.database, isolation=False),
                           collision_rate_khz=100, output_dir=self.phase2_dir)

    def run_script(self, name, **overrides):
        return subprocess.run([str(ROOT / "quick_runners" / name)], cwd=self.root,
                              env={**self.env, **overrides}, text=True, capture_output=True)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_three_runners_match_achieved_rates_and_compare_frozen_menus(self):
        self.assert_success(self.run_script("07_calibrate_phase2.sh"))
        reference_path = self.phase2_dir / "calibrated_thresholds.json"
        self.assertFalse(TriggerMenu.load(reference_path).isolation)
        before = self.menu_path.read_bytes(), reference_path.read_bytes()
        self.assert_success(self.run_script("08_calibrate_erl_to_phase2.sh"))
        self.assertEqual(before, (self.menu_path.read_bytes(), reference_path.read_bytes()))
        matching = json.loads((self.match_dir / "matching.json").read_text())
        self.assertEqual(matching["source_rule_id"], 8)
        matches = {entry["phase2_trigger"]: entry for entry in matching["matches"]}
        self.assertEqual(matches["Discrete"]["phase2_nominal_target_khz"], 35)
        self.assertEqual(matches["Discrete"]["target_rate_khz"], 30)
        self.assertEqual(matches["TooHigh"]["status"], "skipped_phase2_unreachable")
        self.assertEqual(matches["Zero"]["status"], "skipped_zero_rate")
        fitted = TriggerMenu.load(self.match_dir / "matched_menu.json")
        self.assertEqual(len(fitted.triggers), 2)
        self.assertTrue(all(entry["rate_matched"] for entry in matches.values() if entry["erl_trigger"]))
        scores, metadata = load_scores(self.match_dir / "critical_thresholds.npz")
        self.assertEqual(list(scores), ["ERL_rule_8"])
        self.assertEqual(metadata["event_count"], 10)
        snapshots = reference_path.read_bytes(), (self.match_dir / "matched_menu.json").read_bytes()
        self.assert_success(self.run_script("09_compare_erl_phase2.sh"))
        self.assertEqual(snapshots, (reference_path.read_bytes(), (self.match_dir / "matched_menu.json").read_bytes()))
        comparison = json.loads((self.comparison_dir / "comparison.json").read_text())
        self.assertEqual(len(comparison["comparisons"]), 4)
        self.assertEqual(len(comparison["excluded"]), 2)
        self.assertEqual(comparison["samples"]["minbias"]["event_count"], 10)
        self.assertEqual(comparison["samples"]["SignalA"]["event_count"], 4)
        for row in comparison["comparisons"]:
            self.assertEqual(row["signal_efficiency_ratio"], 1)
        with (self.comparison_dir / "comparison.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        leak = self.run_script("09_compare_erl_phase2.sh", MINBIAS_TEST=str(self.train))
        self.assertNotEqual(leak.returncode, 0)
        self.assertIn("independent test minbias", leak.stderr)
        changed = TriggerMenu.load(reference_path)
        changed.thresholds["Reference"]["met"] += 1
        changed.save(reference_path)
        stale = self.run_script("09_compare_erl_phase2.sh")
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("does not match the rate-matching report", stale.stderr)

    def test_unreachable_erl_working_points_are_flagged_and_not_exported_as_matches(self):
        self.make_phase2_report()
        self.model = model(self.data, [[threshold("met", 10)],
                                       [threshold("met", 20), threshold("met", 25, "<=")]], [7, 8])
        self.save_erl()
        self.assert_success(self.run_script("08_calibrate_erl_to_phase2.sh"))
        matching = json.loads((self.match_dir / "matching.json").read_text())
        self.assertEqual(len(TriggerMenu.load(self.match_dir / "matched_menu.json").triggers), 0)
        failures = [entry for entry in matching["matches"] if entry["status"] == "erl_unreachable"]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(entry["erl_trigger"] is None for entry in failures))
        result = self.run_script("09_compare_erl_phase2.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No fitted ERL working points", result.stderr)

    def test_fixed_top_rule_is_not_replaced_by_a_lower_ranked_tunable_rule(self):
        self.make_phase2_report()
        self.model = model(self.data, [[threshold("met", 10)], [threshold("met", 25, "<=")]], [7, 8])
        self.save_erl()
        result = self.run_script("08_calibrate_erl_to_phase2.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("top ERL rule 8 is fixed", result.stderr)

    def test_phase2_runner_can_explicitly_keep_database_isolation(self):
        self.assert_success(self.run_script("07_calibrate_phase2.sh", NO_ISOLATION="0"))
        self.assertTrue(TriggerMenu.load(self.phase2_dir / "calibrated_thresholds.json").isolation)


if __name__ == "__main__":
    unittest.main()
