import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from collide2v_trigger import (PHASE2_MENU, TriggerMenu, calibrate_menu, critical_thresholds, evaluate_rates,
                               fit_rate, input_files, iter_batches, load_scores, scan_plan)
from helpers import dataset


class CalibrationTests(unittest.TestCase):
    def test_critical_thresholds_and_censoring(self):
        data = dataset([{"jets": [{"pt": 20}]}, {"jets": [{"pt": 40}]}, {}, {"jets": [{"pt": 200}]}])
        menu = TriggerMenu(PHASE2_MENU)
        plan = scan_plan(menu, "SinglePuppiJet_230")
        scores = critical_thresholds(data, menu, plan, upper=100, tolerance=0.001)
        np.testing.assert_allclose(scores[[0, 1, 3]], [20, 40, 100], atol=0.001)
        self.assertTrue(np.isnan(scores[2]))

    def test_near_endpoint_is_not_reported_as_upper_censored(self):
        data = dataset([{"jets": [{"pt": 99.999}]}, {"jets": [{"pt": 100}]}])
        menu = TriggerMenu(PHASE2_MENU)
        scores = critical_thresholds(data, menu, scan_plan(menu, "SinglePuppiJet_230"), upper=100, tolerance=0.01)
        self.assertLess(scores[0], 100)
        self.assertEqual(scores[1], 100)

    def test_proportional_multileg_scan_and_fixed_ht_floor(self):
        data = dataset([{"photons": [{"pt": 40}, {"pt": 10}]}])
        menu = TriggerMenu(PHASE2_MENU, isolation=False)
        plan = scan_plan(menu, "DoubleTkIsoPhoton_22_12")
        score = critical_thresholds(data, menu, plan, tolerance=0.001)[0]
        self.assertAlmostEqual(score, 10 * 22 / 12, delta=0.001)
        self.assertEqual(float(plan.thresholds_at(44)["subleading_photon_pt"]), 24)
        ht_plan = scan_plan(menu, "PuppiHT_450")
        self.assertNotIn("ht_jet_pt_min", ht_plan.scale_fields)
        fixed = scan_plan(menu, "DoubleTkIsoPhoton_22_12", scale_fields=["leading_photon_pt"])
        too_soft = dataset([{"photons": [{"pt": 10}, {"pt": 10}]}])
        self.assertTrue(np.isnan(critical_thresholds(too_soft, menu, fixed)[0]))

    def test_fit_methods_and_discrete_rates(self):
        scores = [10, 20, 30, 40, np.nan]
        for method in ("bisection", "monotonic"):
            result = fit_rate(scores, 40, collision_rate_khz=100, upper=100, method=method)
            self.assertEqual(result["passed"], 2)
            self.assertEqual(result["status"], "calibrated")
        tied = fit_rate([10] * 5, 40, collision_rate_khz=100, upper=100)
        self.assertEqual(tied["status"], "discrete")
        self.assertEqual(tied["passed"], 0)
        sparse = fit_rate(scores, 37, collision_rate_khz=100, upper=100)
        self.assertEqual(sparse["status"], "discrete")
        self.assertEqual(sparse["rate_khz"], 20)

    def test_unreachable_and_zero_target(self):
        self.assertEqual(fit_rate([np.nan] * 5, 10)["status"], "unreachable")
        self.assertEqual(fit_rate([100] * 5, 10, upper=100)["status"], "unreachable")
        self.assertEqual(fit_rate([np.nan] * 5, 0)["status"], "unidentified")
        zero = fit_rate([10, 20, 30], 0, upper=100)
        self.assertEqual(zero["passed"], 0)
        self.assertGreater(zero["threshold_gev"], 30)

    def test_chunk_manifest_batching_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dataset([{}, {}, {}]), root / "a.pt")
            torch.save(dataset([{}, {}]), root / "b.pt")
            manifest = root / "events.chunks.json"
            manifest.write_text(json.dumps({"files": [{"path": "a.pt"}, {"path": "b.pt"}]}))
            paths = input_files([manifest, root / "a.pt"])
            self.assertEqual(len(paths), 2)
            self.assertEqual([batch.size for batch in iter_batches(paths, batch_size=2, max_events=4)], [2, 1, 1])

    def test_evaluation_rates_or_bits_and_event_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.pt"
            bits = Path(directory) / "bits.npz"
            torch.save(dataset([{"jets": [{"pt": 250}]}, {"met": 250}, {}, {"met": 300}]), source)
            names = ["SinglePuppiJet_230", "PuppiETmiss_200"]
            report = evaluate_rates(source, TriggerMenu(PHASE2_MENU), triggers=names, collision_rate_khz=120, batch_size=2,
                                    max_events=3, bits_output=bits)
            self.assertEqual(report["event_count"], 3)
            self.assertEqual(report["triggers"]["passes_any"]["rate_khz"], 80)
            with np.load(bits) as saved:
                np.testing.assert_array_equal(saved["passes_any"], [True, True, False])

    def test_complete_calibration_export_and_saved_score_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.pt"
            torch.save(dataset([{"jets": [{"pt": value}]} for value in range(10, 110, 10)]), source)
            name = "SinglePuppiJet_230"
            menu = TriggerMenu(PHASE2_MENU, isolation=False)
            report = calibrate_menu(source, menu, [name], target_rate_khz=20, collision_rate_khz=100,
                                     upper=200, output_dir=root / "calibration", batch_size=4)
            reloaded = TriggerMenu(root / "calibration/calibrated_thresholds.json", isolation=False)
            measured = evaluate_rates(source, reloaded, [name], collision_rate_khz=100)
            self.assertEqual(measured["triggers"][name], report["evaluation"]["triggers"][name])
            self.assertEqual(measured["triggers"][name]["passed"], 2)
            self.assertEqual(menu.thresholds_for(name)["jet_pt"], 230)
            scores_path = root / "calibration/critical_thresholds.npz"
            scores, metadata = load_scores(scores_path)
            self.assertEqual(len(scores[name]), 10)
            self.assertEqual(metadata["event_count"], 10)
            reused = calibrate_menu(scores_file=scores_path, target_rate_khz={name: 40}, collision_rate_khz=100)
            self.assertEqual(reused["evaluation"]["triggers"][name]["passed"], 4)

    def test_unreachable_keeps_defaults_unless_boundary_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.pt"
            torch.save(dataset([{}, {}]), source)
            name = "SinglePuppiJet_230"
            for export in (False, True):
                report = calibrate_menu(source, TriggerMenu(PHASE2_MENU), triggers=[name], target_rate_khz=20, export_unreachable_boundary=export)
                self.assertEqual(report["calibration"][name]["status"], "unreachable")
                self.assertEqual(TriggerMenu.from_dict(report["profile"]).thresholds_for(name)["jet_pt"], 0 if export else 230)
                self.assertEqual(report["evaluation"]["triggers"][name]["passed"], 0)


if __name__ == "__main__":
    unittest.main()
