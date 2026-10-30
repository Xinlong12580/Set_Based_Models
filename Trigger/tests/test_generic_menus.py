from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch

from collide2v_trigger import (PHASE2_MENU, ERLTrigger, Trigger, TriggerMenu,
                               calibrate_menu, evaluate_rates, load_scores, parse_erl_rule)
from collide2v_trigger.cli import main
from helpers import dataset
from test_erl_parser import data_with_schema, rule, threshold


def jet_trigger(name="CustomJet"):
    return Trigger({"format": "collide2v_trigger", "schema_version": 1, "type": "selection",
                    "name": name, "default_thresholds": {"pt_min": 30}, "target_rate_khz": 20,
                    "selection": {"op": "group", "kind": "jets", "thresholds": ["$pt_min"], "eta_max": 2.4}})


class GenericMenuTests(unittest.TestCase):
    def test_empty_menu_and_explicit_preset(self):
        empty = TriggerMenu()
        self.assertEqual(empty.all_triggers, [])
        np.testing.assert_array_equal(empty.trigger_bits(dataset([{}, {}]))["passes_any"], [False, False])
        self.assertEqual(len(TriggerMenu.load(PHASE2_MENU).implementable_triggers), 33)
        with self.assertRaisesRegex(ValueError, "Supply a trigger menu"):
            evaluate_rates("unused.pt")

    def test_custom_definition_does_not_depend_on_trigger_name(self):
        definition = TriggerMenu.load(PHASE2_MENU).triggers["DoubleTkMuon_15_7"].to_dict()
        definition.update(name="NearbyMuons", default_thresholds={"leading_muon_pt": 20, "subleading_muon_pt": 10})
        definition["selection"]["dz_max"] = 2
        menu = TriggerMenu(triggers=[definition])
        data = dataset([{"muons": [{"pt": 20}, {"pt": 10, "dz": 1.9}]},
                        {"muons": [{"pt": 20}, {"pt": 10, "dz": 2}]},
                        {"muons": [{"pt": 19}, {"pt": 10}]}])
        np.testing.assert_array_equal(menu.evaluate("NearbyMuons", data), [True, False, False])
        self.assertEqual(menu.all_triggers, ["NearbyMuons"])

    def test_generic_trigger_save_and_parameter_vector(self):
        trigger = jet_trigger()
        data = dataset([{"jets": [{"pt": 20}]}, {"jets": [{"pt": 40}]}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trigger.json"
            trigger.save(path)
            restored = Trigger.load(path)
            np.testing.assert_array_equal(restored(data, {"pt_min": np.array([10, 50])}), [True, False])

    def test_group_combinations_use_distinct_objects(self):
        definition = jet_trigger().to_dict()
        definition["selection"] = {"op": "multi_group", "groups": [
            {"kind": "jets", "thresholds": ["$pt_min"]},
            {"kind": "jets", "thresholds": ["$pt_min"]}]}
        data = dataset([{"jets": [{"pt": 40}]}, {"jets": [{"pt": 40}, {"pt": 35}]}])
        np.testing.assert_array_equal(Trigger(definition)(data), [False, True])

    def test_mixed_calibration_exports_the_same_menu_format_and_preserves_other_triggers(self):
        data = data_with_schema([{"jets": [{"pt": pt}], "met": met}
                                 for pt, met in zip((20, 40, 60, 80), (100, 200, 300, 400))])
        adjustable = parse_erl_rule(rule(data, [threshold("met", 150)]), data, target_rate_khz=25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = parse_erl_rule(rule(data, [threshold("met", 250, "<=")]), data, name="FixedMET", allow_fixed=True)
        menu = TriggerMenu(triggers=[jet_trigger(), adjustable, fixed, jet_trigger("Untouched")], isolation=False)
        names = ["CustomJet", adjustable.name, fixed.name]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"events.pt"
            torch.save(data, source)
            menu.save(root/"original.json")
            with redirect_stdout(io.StringIO()):
                report = calibrate_menu(source, menu, names, collision_rate_khz=100,
                                         target_rate_khz={"CustomJet": 50}, output_dir=root/"calibrated")
            profile_path = root/"calibrated/calibrated_thresholds.json"
            restored = TriggerMenu.load(profile_path)
            self.assertEqual(restored.all_triggers, menu.all_triggers)
            self.assertFalse(restored.isolation)
            self.assertEqual(restored.thresholds_for("Untouched"), menu.thresholds_for("Untouched"))
            self.assertEqual(restored.target_rate_khz("CustomJet"), 50)
            self.assertEqual(restored.target_rate_khz(adjustable.name), 25)
            self.assertEqual(restored.thresholds_for(fixed.name), {})
            self.assertIsInstance(restored.triggers[adjustable.name], ERLTrigger)
            for path in (root/"original.json", profile_path, PHASE2_MENU):
                payload = json.loads(path.read_text())
                self.assertEqual((payload["format"], payload["schema_version"]), ("collide2v_trigger_menu", 2))
                self.assertTrue(all(entry["format"] == "collide2v_trigger" for entry in payload["triggers"]))
            measured = evaluate_rates(source, restored, names, collision_rate_khz=100)
            self.assertEqual(measured["triggers"], report["evaluation"]["triggers"])
            _, metadata = load_scores(root/"calibrated/critical_thresholds.npz")
            self.assertEqual(TriggerMenu.from_dict(metadata["menu"]).to_dict(), menu.to_dict())
            with redirect_stdout(io.StringIO()):
                replayed = calibrate_menu(scores_file=root/"calibrated/critical_thresholds.npz", collision_rate_khz=100,
                                           target_rate_khz={"CustomJet": 50})
            self.assertEqual(replayed["evaluation"]["triggers"], measured["triggers"])

    def test_version_one_menus_load_without_changing_their_contents(self):
        data = data_with_schema([{"jets": [{"pt": 50}], "met": 250}])
        learned = parse_erl_rule(rule(data, [threshold("met", 200)]), data).to_dict()
        learned.update(format="collide2v_erl_trigger")
        learned.pop("type")
        old = {"format": "collide2v_trigger_menu", "schema_version": 1,
               "object_selection": {"isolation_enabled": False}, "erl_triggers": [learned],
               "thresholds": {"SinglePuppiJet_230": {"jet_pt": 40}, "ERL_rule_7": {"cut_0_met": 300}},
               "target_rates_khz": {"ERL_rule_7": 12}}
        before = deepcopy(old)
        menu = TriggerMenu.from_dict(old)
        self.assertEqual(old, before)
        self.assertEqual(len(menu.all_triggers), 43)
        self.assertTrue(menu.evaluate("SinglePuppiJet_230", data)[0])
        self.assertFalse(menu.evaluate("ERL_rule_7", data)[0])
        self.assertEqual(menu.target_rate_khz("ERL_rule_7"), 12)
        self.assertEqual(menu.to_dict()["schema_version"], 2)

    def test_cli_loads_custom_menu_and_requires_explicit_menu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, menu_path, output = root/"events.pt", root/"menu.json", root/"rates.json"
            torch.save(dataset([{"jets": [{"pt": 40}]}, {}]), source)
            TriggerMenu(triggers=[jet_trigger()]).save(menu_path)
            with redirect_stdout(io.StringIO()):
                main(["evaluate", str(source), "--menu", str(menu_path), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["triggers"]["CustomJet"]["efficiency"], 0.5)
            self.assertEqual(set(report["triggers"]), {"CustomJet", "passes_any"})
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["evaluate", str(source)])


if __name__ == "__main__":
    unittest.main()
