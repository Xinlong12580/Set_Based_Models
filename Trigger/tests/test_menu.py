import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from collide2v_trigger import PHASE2_MENU, TensorBatch, TriggerMenu
from helpers import COLUMNS, dataset


class MenuTests(unittest.TestCase):
    def test_menu_and_unavailable_objects(self):
        menu = TriggerMenu(PHASE2_MENU)
        self.assertEqual(len(menu.all_triggers), 42)
        self.assertEqual(len(menu.implementable_triggers), 33)
        bits = menu.trigger_bits(dataset([{}]))
        self.assertFalse(any(flags[0] for flags in bits.values()))
        with self.assertRaises(NotImplementedError):
            menu.evaluate("SingleStaEG_51", dataset([{}]))

    def test_pt_eta_and_same_object(self):
        data = dataset([{"muons": [{"pt": 22}]}, {"muons": [{"pt": 22, "eta": 2.4}]},
                        {"muons": [{"pt": 30, "eta": 3}, {"pt": 10}]}, {}])
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("SingleTkMuon_22", data), [True, False, False, False])

    def test_relative_dz_units_and_valid_zero(self):
        data = dataset([{"muons": [{"pt": 15, "dz": 0}, {"pt": 7, "dz": 9.99}]},
                        {"muons": [{"pt": 15, "dz": 0}, {"pt": 7, "dz": 10}]},
                        {"muons": [{"pt": 15}, {"pt": 7, "dz_valid": 0}]}])
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("DoubleTkMuon_15_7", data), [True, False, False])

    def test_dz_is_required_only_for_relative_dz_seeds(self):
        data = dataset([{"muons": [{"pt": 30}, {"pt": 10}]}])
        keep = [i for i, name in enumerate(data["obj_columns"]) if not name.startswith("dz")]
        data["obj"] = data["obj"][..., keep]
        data["obj_columns"] = [COLUMNS[i] for i in keep]
        data["obj_feature_schema"] = [spec for spec in data["obj_feature_schema"] if not spec["name"].startswith("dz")]
        self.assertTrue(TriggerMenu(PHASE2_MENU).evaluate("SingleTkMuon_22", data)[0])
        with self.assertRaisesRegex(KeyError, "Reconvert"):
            TriggerMenu(PHASE2_MENU).evaluate("DoubleTkMuon_15_7", data)

    def test_mixed_lepton_dz_and_multiplicity(self):
        events = [{"muons": [{"pt": 20, "dz": 4}], "electrons": [{"pt": 18, "dz": -4}, {"pt": 7}]},
                  {"muons": [{"pt": 20, "dz": 6}], "electrons": [{"pt": 18, "dz": -4}, {"pt": 7}]},
                  {"muons": [{"pt": 20}], "electrons": [{"pt": 18}]}]
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("TkMuon_DoubleTkElectron_17_17_6", dataset(events)),
                                      [True, False, False])

    def test_isolation_and_validity(self):
        data = dataset([{"photons": [{"pt": 30}, {"pt": 20}]},
                        {"photons": [{"pt": 30}, {"pt": 20, "isolation_rho_corr": 0.2}]},
                        {"photons": [{"pt": 30}, {"pt": 20, "isolation_rho_corr_valid": 0}]}])
        name = "DoubleTkIsoPhoton_22_12"
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate(name, data), [True, False, False])
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU, isolation=False).evaluate(name, data), [True, True, True])

    def test_ht_strict_floor_and_eta(self):
        data = dataset([{"jets": [{"pt": 30}, {"pt": 31}, {"pt": 100, "eta": 2.5}]}])
        menu = TriggerMenu(PHASE2_MENU, threshold_overrides={"PuppiHT_135": {"ht": 32}})
        self.assertFalse(menu.evaluate("PuppiHT_135", data)[0])
        self.assertTrue(menu.evaluate("PuppiHT_135", data, {"ht": 31})[0])

    def test_quad_jet_constituent_floor(self):
        data = dataset([{"jets": [{"pt": 80}, {"pt": 60}, {"pt": 40}, {"pt": 25}]}])
        name = "QuadPuppiJets_PuppiHT_400_70_55_40_40"
        cut = {"ht": 0, "fourth_jet_pt": 0}
        self.assertFalse(TriggerMenu(PHASE2_MENU).evaluate(name, data, cut)[0])

    def test_bph_opposite_sign_and_mass(self):
        pair = [{"pt": 5, "phi": 0, "charge": 1}, {"pt": 5, "phi": np.pi, "charge": -1}]
        same = copy.deepcopy(pair)
        same[1]["charge"] = 1
        invalid = copy.deepcopy(pair)
        invalid[1]["charge_valid"] = 0
        data = dataset([{"muons": pair}, {"muons": same}, {"muons": invalid}])
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("BPH_DoubleTkMuon_4_4_m7_18", data), [True, False, False])
        triple = dataset([{"muons": [*pair, {"pt": 3}]}])
        self.assertTrue(TriggerMenu(PHASE2_MENU).evaluate("BPH_TripleTkMuon_5_3_2_m5_17", triple)[0])

    def test_vbf_mass_and_validity(self):
        jets = [{"pt": 200, "eta": 3, "mass": 10}, {"pt": 50, "eta": -3, "mass": 10}]
        invalid = copy.deepcopy(jets)
        invalid[0]["mass_valid"] = 0
        result = TriggerMenu(PHASE2_MENU).evaluate("VBF_DoublePuppiJets_160_35_mjj620", dataset([{"jets": jets}, {"jets": invalid}]))
        np.testing.assert_array_equal(result, [True, False])

    def test_electron_jet_separation(self):
        events = [{"electrons": [{"pt": 30}], "jets": [{"pt": 50, "phi": 1}]},
                  {"electrons": [{"pt": 30}], "jets": [{"pt": 50}]},
                  {"electrons": [{"pt": 30, "eta": 2.2}], "jets": [{"pt": 50}]}]
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("TkElectron_PuppiJet_28_40", dataset(events)), [True, False, False])

    def test_muon_triple_jet_geometry(self):
        jets = [{"pt": 60, "phi": 0.1}, {"pt": 50, "eta": 0.3}, {"pt": 40, "eta": -0.3}]
        far = copy.deepcopy(jets)
        far[0]["phi"] = 1
        data = dataset([{"muons": [{"pt": 15}], "jets": jets}, {"muons": [{"pt": 15}], "jets": far}])
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("TkMuon_TriplePuppiJet_12_40_40_40", data), [True, False])

    def test_reordered_columns_categories_and_slot_mask(self):
        data = dataset([{"muons": [{"pt": 30}]}])
        data["obj"][..., COLUMNS.index("type_id")] += 10
        spec = next(spec for spec in data["obj_feature_schema"] if spec["name"] == "type_id")
        spec["categories"] = {code + 10: label for code, label in spec["categories"].items()}
        order = list(reversed(range(len(COLUMNS))))
        data["obj"] = data["obj"][..., order]
        data["obj_columns"] = [COLUMNS[i] for i in order]
        self.assertTrue(TriggerMenu(PHASE2_MENU).evaluate("SingleTkMuon_22", data)[0])
        data["obj_slot_mask"] = torch.zeros(data["obj"].shape[:2], dtype=torch.bool)
        self.assertFalse(TriggerMenu(PHASE2_MENU).evaluate("SingleTkMuon_22", data)[0])

    def test_met_validity_and_native_fallback(self):
        data = dataset([{"met": 200}, {"met": 300, "met_valid": 0}])
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("PuppiETmiss_200", data), [True, False])
        data.update(native_event_info=data.pop("event_features"), native_event_info_columns=["puppi_met", "unused"],
                    native_event_info_validity=torch.tensor([[True, True], [False, True]]))
        data["event_feature_columns"] = []
        np.testing.assert_array_equal(TriggerMenu(PHASE2_MENU).evaluate("PuppiETmiss_200", data), [True, False])

    def test_file_input_and_threshold_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.pt"
            profile = Path(directory) / "profile.json"
            torch.save(dataset([{"jets": [{"pt": 50}]}]), source)
            profile.write_text(json.dumps({"thresholds": {"SinglePuppiJet_230": {"jet_pt": 40}},
                                           "object_selection": {"isolation_enabled": False}}))
            self.assertTrue(TriggerMenu(profile, isolation=False).evaluate("SinglePuppiJet_230", source)[0])
            self.assertFalse(TriggerMenu(profile).isolation)

    def test_all_seeds_are_monotonic_under_proportional_scaling(self):
        event = {"muons": [{"pt": 20, "charge": 1}, {"pt": 8, "charge": -1, "phi": 1}, {"pt": 3, "phi": 3}],
                 "electrons": [{"pt": 30}, {"pt": 10}], "photons": [{"pt": 40}, {"pt": 15}],
                 "jets": [{"pt": 500, "phi": 0.1}, {"pt": 200, "eta": 0.4}, {"pt": 130, "eta": -0.4},
                          {"pt": 80}, {"pt": 180, "eta": 3}, {"pt": 100, "eta": -3}], "met": 250}
        batch = TensorBatch(dataset([event]))
        for isolation in (False, True):
            menu = TriggerMenu(PHASE2_MENU, isolation=isolation)
            for name in menu.implementable_triggers:
                previous = True
                for scale in (0, 0.1, 0.5, 1, 2, 10):
                    cuts = {field: value if field == "ht_jet_pt_min" else value * scale
                            for field, value in menu.thresholds_for(name).items()}
                    passed = bool(menu.evaluate(name, batch, cuts)[0])
                    self.assertFalse(passed and not previous, name)
                    previous = passed


if __name__ == "__main__":
    unittest.main()
