from pathlib import Path
import tempfile
import unittest

import numpy as np

from collide2v_trigger import PHASE2_MENU, TriggerMenu, parse_erl_model
from test_erl_model_parser import model
from test_erl_parser import data_with_schema, threshold


class MenuSerializationTests(unittest.TestCase):
    def setUp(self):
        self.data = data_with_schema([{"jets": [{"pt": 50}], "met": 150},
                                      {"muons": [{"pt": 30}, {"pt": 10}], "met": 250}, {"met": 350}])

    def assert_same_decisions(self, first, second):
        before, after = first.trigger_bits(self.data), second.trigger_bits(self.data)
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)

    def test_native_menu_settings_and_decisions_round_trip(self):
        menu = TriggerMenu(PHASE2_MENU, isolation=False, electron_iso_max=0.2, photon_iso_max=0.3,
                           threshold_overrides={"SinglePuppiJet_230": {"jet_pt": 40}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/menu.json"
            menu.save(path)
            restored = TriggerMenu.load(path)
            self.assertEqual(menu.to_dict(), restored.to_dict())
            self.assert_same_decisions(menu, restored)
        invalid = menu.to_dict()
        invalid["schema_version"] = 99
        with self.assertRaisesRegex(ValueError, "format/version"):
            TriggerMenu.from_dict(invalid)

    def test_imported_rules_effective_thresholds_and_rates_in_one_file(self):
        source = model(self.data, [[threshold("met", 100)], [threshold("met", 300)]], [8, 7])
        triggers = parse_erl_model(source, self.data, target_rate_khz=20)
        menu = TriggerMenu(triggers=triggers, isolation=False)
        menu.thresholds["ERL_rule_7"]["cut_0_met"] = 200
        menu.target_rates["ERL_rule_7"] = 12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "menu.json"
            menu.save(path)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            restored = TriggerMenu.load(path)
            self.assertEqual(menu.menu_summary(), restored.menu_summary())
            self.assertEqual(restored.target_rate_khz("ERL_rule_7"), 12)
            self.assertEqual(restored.thresholds_for("ERL_rule_7"), {"cut_0_met": 200})
            self.assert_same_decisions(menu, restored)
            profile_reader = TriggerMenu(path, isolation=False)
            self.assertEqual(restored.to_dict(), profile_reader.to_dict())


if __name__ == "__main__":
    unittest.main()
