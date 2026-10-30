from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch

from collide2v_trigger import (ERLTrigger, RuleParseError, TriggerMenu, calibrate_menu,
                               load_scores, parse_erl_model, parse_erl_rule, scan_plan)
from test_erl_model_parser import model
from test_erl_parser import add_photon_count, data_with_schema, rule, threshold


class FixedTriggerTests(unittest.TestCase):
    def setUp(self):
        self.data = data_with_schema([{"met": 150}, {"met": 250}, {"met": 350}])

    def parse_fixed(self):
        source = rule(self.data, [threshold("met", 200, "<=")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trigger = parse_erl_rule(source, self.data, allow_fixed=True)
        self.assertEqual(len(caught), 1)
        self.assertIn("Keeping rule 7 fixed", str(caught[0].message))
        return trigger

    def test_fixed_rule_evaluation_serialization_and_no_parameters(self):
        trigger = self.parse_fixed()
        self.assertFalse(trigger.tunable)
        self.assertEqual(trigger.default_thresholds, {})
        self.assertTrue(all(cut["action"] == "fixed" for cut in trigger.cut_actions))
        np.testing.assert_array_equal(trigger(self.data), [True, False, False])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trigger.json"
            trigger.save(path)
            restored = ERLTrigger.load(path)
            self.assertFalse(restored.tunable)
            self.assertIn("fixed_reason", restored.to_dict())
            np.testing.assert_array_equal(restored(self.data), trigger(self.data))
        with self.assertRaisesRegex(RuleParseError, "Unknown tunable parameters"):
            trigger(self.data, {"invented_threshold": 10})

    def test_unavailable_count_translation_keeps_literal_count(self):
        feature = add_photon_count(self.data, [3, 2, 4], [35, 25, 40])
        source = rule(self.data, [threshold(feature, 3, ">=")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trigger = parse_erl_rule(source, self.data, allow_fixed=True)
        self.assertFalse(trigger.tunable)
        self.assertIn("rank 3", str(caught[0].message))
        self.assertEqual(trigger.cut_actions[0]["observable"], feature)
        self.assertEqual(trigger.cut_actions[0]["threshold"], 3)
        np.testing.assert_array_equal(trigger(self.data), [True, False, True])

    def test_fixed_fallback_does_not_hide_missing_features_or_bad_semantics(self):
        source = rule(self.data, [threshold("met", 200)])
        with self.assertRaisesRegex(RuleParseError, "cut indices"):
            parse_erl_rule(source, self.data, tune=[99], allow_fixed=True)
        source["rule"]["steps"][0]["cut"]["type"] = "unknown"
        with self.assertRaisesRegex(RuleParseError, "Unsupported threshold"):
            parse_erl_rule(source, self.data, allow_fixed=True)
        source = rule(self.data, [threshold("met", 200, "<=")])
        self.data["event_feature_schema"].pop(0)
        with self.assertRaisesRegex(RuleParseError, "Missing event feature"):
            parse_erl_rule(source, self.data, allow_fixed=True)

    def test_model_keeps_fixed_rules_before_optionally_skipping_unparseable_rules(self):
        source = model(self.data, [[threshold("met", 200, "<=")], [threshold("met", 100)],
                                   [threshold("met", 300)]], [8, 7, 9])
        source["rules"][2]["steps"][0]["cut"]["comparison"] = "=="
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            triggers = parse_erl_model(source, self.data, allow_fixed=True, skip_unsupported=True)
        self.assertEqual([trigger.definition["source_rule_id"] for trigger in triggers], [8, 7])
        self.assertEqual([trigger.tunable for trigger in triggers], [True, False])
        self.assertIn("Keeping rule 7 fixed", str(caught[0].message))
        self.assertIn("Skipping rule 9", str(caught[1].message))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(RuleParseError, "Rule 9"):
                parse_erl_model(source, self.data, allow_fixed=True)

    def test_calibration_adjusts_only_tunable_rules_and_restores_fixed_rules(self):
        source = model(self.data, [[threshold("met", 200, "<=")], [threshold("met", 100)]], [7, 8])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            triggers = parse_erl_model(source, self.data, allow_fixed=True)
        menu = TriggerMenu(triggers=triggers)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "events.pt"
            torch.save(self.data, data_path)
            output = root / "calibration"
            report = calibrate_menu(data_path, menu, [trigger.name for trigger in triggers],
                                     target_rate_khz={"ERL_rule_8": 30}, collision_rate_khz=90, output_dir=output)
            self.assertEqual(report["calibration"]["ERL_rule_7"]["status"], "fixed")
            self.assertEqual(report["calibration"]["ERL_rule_7"]["applied_measurement"]["passed"], 1)
            self.assertNotIn("threshold_gev", report["calibration"]["ERL_rule_7"])
            self.assertEqual(report["evaluation"]["triggers"]["ERL_rule_8"]["passed"], 1)
            scores, metadata = load_scores(output / "critical_thresholds.npz")
            self.assertEqual(list(scores), ["ERL_rule_8"])
            self.assertEqual(metadata["fixed_triggers"], ["ERL_rule_7"])
            restored = TriggerMenu(output / "calibrated_thresholds.json")
            self.assertFalse(restored.is_tunable("ERL_rule_7"))
            np.testing.assert_array_equal(restored.evaluate("ERL_rule_7", self.data), [True, False, False])
            reused = calibrate_menu(scores_file=output / "critical_thresholds.npz",
                                     target_rate_khz={"ERL_rule_8": 60}, collision_rate_khz=90)
            self.assertEqual(reused["calibration"]["ERL_rule_7"]["status"], "fixed")
            self.assertEqual(reused["evaluation"]["triggers"]["ERL_rule_8"]["passed"], 2)

    def test_fixed_only_calibration_needs_no_target_or_score_arrays(self):
        trigger = self.parse_fixed()
        menu = TriggerMenu(triggers=[trigger])
        with self.assertRaisesRegex(ValueError, "fixed and has no tunable parameters"):
            scan_plan(menu, trigger.name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.pt"
            torch.save(self.data, source)
            report = calibrate_menu(source, menu, [trigger.name], output_dir=root / "output")
            self.assertEqual(report["calibration"][trigger.name]["status"], "fixed")
            self.assertEqual(report["evaluation"]["triggers"][trigger.name]["passed"], 1)
            scores, metadata = load_scores(root / "output/critical_thresholds.npz")
            self.assertEqual(scores, {})
            self.assertEqual(metadata["plans"], {})
            reused = calibrate_menu(scores_file=root / "output/critical_thresholds.npz")
            self.assertEqual(reused["evaluation"]["triggers"][trigger.name]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
