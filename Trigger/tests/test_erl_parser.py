from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from collide2v_trigger import (ERLTrigger, RuleParseError, TensorBatch, TriggerMenu, calibrate_menu,
                               critical_thresholds, evaluate_rates, parse_erl_rule, scan_plan)
from helpers import dataset


def data_with_schema(events):
    data = dataset(events)
    for spec in data["obj_feature_schema"]:
        name = spec["name"]
        spec["units"] = "GeV" if name in ("pt", "mass") else "1"
        if name.endswith("_valid"):
            spec["kind"] = "boolean"
        if name in ("charge", "dz"):
            spec["units"] = "e" if name == "charge" else "mm"
    data["event_feature_schema"] = [{"name": "met", "kind": "continuous", "units": "GeV", "validity_column": "met_valid"},
                                     {"name": "met_valid", "kind": "boolean", "units": "1"}]
    return data


def threshold(name, value, comparison=">", scope="event"):
    return {"scope": scope, "observable": name, "type": "threshold", "threshold": value, "comparison": comparison}


def category(name, codes, scope="slot"):
    return {"scope": scope, "observable": name, "type": "category", "accepted_codes": codes}


def rule(data, cuts, count=1, slot_source="obj"):
    def schema(entries):
        return [{"name": entry["name"], "kind": entry["kind"], "categories": entry.get("categories"),
                 "validity_column": entry.get("validity_column")} for entry in entries]
    return {"rule": {"rule_id": 7, "minimum_count": count, "steps": [{"cut": cut} for cut in cuts]},
            "observables": schema(data.get(slot_source + "_feature_schema", [])),
            "event_observables": schema(data["event_feature_schema"])}


def add_feature(data, name, values, kind="continuous", units="GeV", definition=None):
    data["event_feature_columns"].append(name)
    data["event_feature_schema"].append({"name": name, "kind": kind, "units": units, "definition": definition or {}})
    data["event_features"] = torch.cat([data["event_features"], torch.tensor(values, dtype=torch.float64)[:, None]], dim=1)


def add_photon_count(data, counts, ranked, floor=20):
    source = "L1T_PhotonTight_PT"
    name = "n_photons" + (f"_pt{floor}" if floor else "")
    add_feature(data, name, counts, "ordinal", "count",
                {"raw_column": source, "pt_threshold_GeV": floor, "comparison": ">", "finite_pt_required": True,
                 "object_truncation": False, "additional_cuts": None})
    add_feature(data, "subleading_pt_photons", ranked, definition={
        "raw_column": source, "rank": 2, "order": "descending", "pt_threshold_GeV": 0,
        "comparison": ">", "finite_pt_required": True, "object_truncation": False,
        "missing_fill": 0, "additional_cuts": None})
    return name


class ParserTests(unittest.TestCase):
    def test_event_threshold_vector_override_and_guards(self):
        data = data_with_schema([{"met": 200}, {"met": 300}, {"met": 400, "met_valid": 0}])
        parsed = parse_erl_rule(rule(data, [threshold("met", 200)]), data)
        parameter = next(iter(parsed.default_thresholds))
        np.testing.assert_array_equal(parsed(data), [False, True, False])
        np.testing.assert_array_equal(parsed(data, {parameter: np.array([199, 301, 0])}), [True, False, False])
        self.assertEqual(parsed.cut_actions[0]["action"], "threshold")

    def test_same_object_conjunction_count_and_padding(self):
        data = data_with_schema([{"jets": [{"pt": 50}], "muons": [{"pt": 5}]},
                                 {"muons": [{"pt": 50}, {"pt": 30}]}, {}])
        cuts = [threshold("pt", 20, scope="slot"), category("type_id", [1])]
        parsed = parse_erl_rule(rule(data, cuts, count=2), data)
        np.testing.assert_array_equal(parsed(data), [False, True, False])
        self.assertEqual(parsed.slot_source, "obj")
        data["obj_slot_mask"] = torch.zeros(data["obj"].shape[:2], dtype=torch.bool)
        np.testing.assert_array_equal(parsed(data), [False, False, False])

    def test_mixed_scope_and_fixed_isolation_guard(self):
        data = data_with_schema([{"met": 300, "muons": [{"pt": 40}]},
                                 {"met": 100, "muons": [{"pt": 40}]},
                                 {"met": 300, "muons": [{"pt": 40, "isolation_rho_corr_valid": 0}]}])
        cuts = [threshold("pt", 20, scope="slot"), threshold("isolation_rho_corr", 0.15, "<=", "slot"),
                threshold("met", 200)]
        parsed = parse_erl_rule(rule(data, cuts), data)
        np.testing.assert_array_equal(parsed(data), [True, False, False])
        self.assertEqual(parsed.cut_actions[1]["action"], "fixed")
        self.assertEqual(len(parsed.default_thresholds), 2)

    def test_saved_model_guard_policy_is_preserved(self):
        data = data_with_schema([{"muons": [{"pt": 40, "isolation_rho_corr_valid": 0}]}])
        source = rule(data, [threshold("pt", 20, scope="slot"), threshold("isolation_rho_corr", 0.15, "<=", "slot")])
        next(spec for spec in source["observables"] if spec["name"] == "isolation_rho_corr")["validity_column"] = None
        self.assertTrue(parse_erl_rule(source, data)(data)[0])

    def test_column_and_category_reordering(self):
        data = data_with_schema([{"muons": [{"pt": 50}]}])
        parsed = parse_erl_rule(rule(data, [category("type_id", [1]), threshold("pt", 20, scope="slot")]), data)
        reordered = deepcopy(data)
        type_index = reordered["obj_columns"].index("type_id")
        reordered["obj"][..., type_index] += 10
        spec = next(spec for spec in reordered["obj_feature_schema"] if spec["name"] == "type_id")
        spec["categories"] = {int(code) + 10: label for code, label in spec["categories"].items()}
        indices = list(reversed(range(len(reordered["obj_columns"]))))
        reordered["obj"] = reordered["obj"][..., indices]
        reordered["obj_columns"] = [reordered["obj_columns"][i] for i in indices]
        self.assertTrue(parsed(reordered)[0])

    def test_count_translation_strict_floor_and_vector_parameters(self):
        data = data_with_schema([{}, {}, {}])
        feature = add_photon_count(data, [2, 1, 0], [35, 20, 0])
        parsed = parse_erl_rule(rule(data, [threshold(feature, 2, ">=")]), data)
        parameter = next(iter(parsed.default_thresholds))
        np.testing.assert_array_equal(parsed(data), [True, False, False])
        np.testing.assert_array_equal(parsed(data, {parameter: np.array([36, 19, 0])}), [False, True, False])
        self.assertEqual(parsed.default_thresholds[parameter], 20)

    def test_noninteger_count_comparator(self):
        data = data_with_schema([{}, {}])
        feature = add_photon_count(data, [2, 1], [35, 20])
        parsed = parse_erl_rule(rule(data, [threshold(feature, 1.5)]), data)
        np.testing.assert_array_equal(parsed(data), [True, False])

    def test_leading_pt_count_translation(self):
        data = data_with_schema([{}, {}])
        feature = add_photon_count(data, [1, 0], [35, 20])
        data["event_feature_columns"][-1] = "leading_pt_photons"
        data["event_feature_schema"][-1]["name"] = "leading_pt_photons"
        data["event_feature_schema"][-1]["definition"]["rank"] = 1
        parsed = parse_erl_rule(rule(data, [threshold(feature, 1, ">=")]), data)
        np.testing.assert_array_equal(parsed(data), [True, False])

    def test_unsupported_saved_actions_and_changed_defaults_fail(self):
        data = data_with_schema([{"met": 300}])
        parsed = parse_erl_rule(rule(data, [threshold("met", 200)]), data)
        invalid = parsed.to_dict()
        invalid["cuts"][0]["action"] = "unknown"
        with self.assertRaisesRegex(RuleParseError, "Unsupported cut action"):
            ERLTrigger(invalid)
        invalid = parsed.to_dict()
        invalid["default_thresholds"][next(iter(parsed.default_thresholds))] = 0
        with self.assertRaisesRegex(RuleParseError, "learned default"):
            ERLTrigger(invalid)

    def test_count_mismatch_and_definition_change_are_errors(self):
        data = data_with_schema([{}])
        feature = add_photon_count(data, [2], [35])
        parsed = parse_erl_rule(rule(data, [threshold(feature, 2, ">=")]), data)
        bad = deepcopy(data)
        bad["event_features"][0, bad["event_feature_columns"].index(feature)] = 0
        with self.assertRaisesRegex(RuleParseError, "disagree"):
            parsed(bad)
        changed = deepcopy(data)
        changed["event_feature_schema"][-1]["definition"]["object_truncation"] = True
        with self.assertRaisesRegex(RuleParseError, "definition changed"):
            parsed(changed)

    def test_zero_floor_offset_scan_and_complete_calibration(self):
        data = data_with_schema([{}, {}, {}, {}])
        feature = add_photon_count(data, [2] * 4, [10, 20, 30, 40], floor=0)
        parsed = parse_erl_rule(rule(data, [threshold(feature, 2, ">=")]), data,
                                name="LearnedDiphoton", target_rate_khz=25)
        menu = TriggerMenu(triggers=[parsed])
        plan = scan_plan(menu, parsed.name)
        self.assertEqual(plan.mode, "offset")
        scores = critical_thresholds(data, menu, plan, upper=100, tolerance=0.001)
        np.testing.assert_allclose(scores, [10, 20, 30, 40], atol=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.pt"
            torch.save(data, source)
            parsed.save(root / "trigger.json")
            restored = ERLTrigger.load(root / "trigger.json")
            np.testing.assert_array_equal(parsed(data), restored(data))
            result = calibrate_menu(source, menu, [parsed.name], collision_rate_khz=100,
                                     upper=100, output_dir=root / "calibration")
            self.assertEqual(result["evaluation"]["triggers"][parsed.name]["passed"], 1)
            profile = TriggerMenu(root / "calibration/calibrated_thresholds.json")
            self.assertEqual(evaluate_rates(source, profile, [parsed.name])["triggers"][parsed.name]["passed"], 1)
            reused = calibrate_menu(scores_file=root / "calibration/critical_thresholds.npz",
                                     target_rate_khz=50, collision_rate_khz=100)
            self.assertEqual(reused["evaluation"]["triggers"][parsed.name]["passed"], 2)

    def test_higher_rank_missing_rank_and_float32_fail(self):
        data = data_with_schema([{}])
        feature = add_photon_count(data, [2], [35])
        with self.assertRaisesRegex(RuleParseError, "rank 3"):
            parse_erl_rule(rule(data, [threshold(feature, 3, ">=")]), data)
        bad = deepcopy(data)
        bad["event_feature_schema"].pop()
        with self.assertRaisesRegex(RuleParseError, "matching full-collection"):
            parse_erl_rule(rule(bad, [threshold(feature, 2, ">=")]), bad)
        bad = deepcopy(data)
        bad["event_features"] = bad["event_features"].float()
        with self.assertRaisesRegex(RuleParseError, "float64"):
            parse_erl_rule(rule(bad, [threshold(feature, 2, ">=")]), bad)

    def test_explicit_selection_can_keep_a_count_fixed(self):
        data = data_with_schema([{"met": 300}])
        feature = add_photon_count(data, [3], [35])
        parsed = parse_erl_rule(rule(data, [threshold(feature, 3, ">="), threshold("met", 200)]), data, tune=[1])
        self.assertTrue(parsed(data)[0])
        self.assertEqual(parsed.cut_actions[0]["action"], "fixed")

    def test_pf_count_error_and_outer_pf_sum_tuning(self):
        data = data_with_schema([{}])
        add_feature(data, "n_charged_pf_pt_gt2_lt5", [3], "ordinal", "count", {"pt_min_GeV": 2, "pt_max_GeV": 5})
        add_feature(data, "sum_pt_charged_pf_pt_gt2_lt5", [11], definition={"pt_min_GeV": 2, "pt_max_GeV": 5})
        with self.assertRaisesRegex(RuleParseError, "PF bins need raw"):
            parse_erl_rule(rule(data, [threshold("n_charged_pf_pt_gt2_lt5", 2, ">=")]), data, tune=[0])
        parsed = parse_erl_rule(rule(data, [threshold("sum_pt_charged_pf_pt_gt2_lt5", 10)]), data)
        self.assertTrue(parsed(data)[0])
        self.assertFalse(parsed(data, {next(iter(parsed.default_thresholds)): 12})[0])

    def test_unsupported_or_untunable_rules_fail(self):
        data = data_with_schema([{}])
        for cuts in ([category("type_id", [1])], [threshold("met", 200, "<=")]):
            with self.assertRaisesRegex(RuleParseError, "no supported tunable"):
                parse_erl_rule(rule(data, cuts), data)
        malformed = threshold("met", 200)
        malformed["type"] = "python_expression"
        with self.assertRaisesRegex(RuleParseError, "Unsupported threshold"):
            parse_erl_rule(rule(data, [malformed]), data)
        with self.assertRaisesRegex(RuleParseError, "printed rule text"):
            parse_erl_rule("EXISTS photon: pt > 20", data)

    def test_missing_changed_and_nonfinite_features_fail(self):
        data = data_with_schema([{"met": 300}])
        parsed = parse_erl_rule(rule(data, [threshold("met", 200)]), data)
        changed = deepcopy(data)
        changed["event_feature_schema"][0]["units"] = "MeV"
        with self.assertRaisesRegex(RuleParseError, "definition changed"):
            parsed(changed)
        missing = deepcopy(data)
        missing["event_feature_schema"] = missing["event_feature_schema"][1:]
        with self.assertRaisesRegex(RuleParseError, "Missing event feature"):
            parsed(missing)
        nonfinite = deepcopy(data)
        nonfinite["event_features"][0, 0] = np.nan
        with self.assertRaisesRegex(RuleParseError, "Nonfinite valid"):
            parsed(nonfinite)

    def test_model_json_selects_exactly_one_rule(self):
        data = data_with_schema([{"met": 300}])
        source = rule(data, [threshold("met", 200)])
        model = {"format": "set_based_models_erl", "schema_version": 1,
                 "observables": source["observables"], "event_observables": source["event_observables"],
                 "rules": [source["rule"], {**source["rule"], "rule_id": 8}]}
        with self.assertRaisesRegex(RuleParseError, "exactly one rule"):
            parse_erl_rule(model, data)
        self.assertTrue(parse_erl_rule(model, data, rule_id=8)(data)[0])

    def test_pair_conjunction_fixed_mass_and_pair_count(self):
        data = data_with_schema([{}, {}])
        columns = ["pt_1", "pt_2", "pair_mass", "pair_mass_valid"]
        data.update(pair_obj=torch.tensor([[[40, 5, 125, 1], [5, 40, 125, 1]],
                                           [[40, 30, 125, 1], [50, 30, 140, 1]]], dtype=torch.float32),
                    pair_slot_mask=torch.ones((2, 2), dtype=torch.bool), pair_obj_columns=columns,
                    pair_obj_feature_schema=[{"name": name, "kind": "boolean" if name.endswith("_valid") else "continuous",
                                              "units": "1" if name.endswith("_valid") else "GeV",
                                              **({"validity_column": "pair_mass_valid"} if name == "pair_mass" else {})}
                                             for name in columns])
        cuts = [threshold("pt_1", 20, scope="slot"), threshold("pt_2", 10, scope="slot"),
                threshold("pair_mass", 120, scope="slot"), threshold("pair_mass", 130, "<=", "slot")]
        parsed = parse_erl_rule(rule(data, cuts, slot_source="pair_obj"), data)
        self.assertEqual(parsed.slot_source, "pair_obj")
        np.testing.assert_array_equal(parsed(data), [False, True])
        self.assertEqual(parsed.cut_actions[2]["action"], "fixed")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pairs.pt"
            torch.save(data, source)
            report = evaluate_rates(source, TriggerMenu(triggers=[parsed]), [parsed.name])
            self.assertEqual(report["triggers"][parsed.name]["passed"], 1)
        missing = deepcopy(data)
        missing.pop("pair_obj")
        with self.assertRaisesRegex(RuleParseError, "requires pair_obj"):
            parsed(missing)


if __name__ == "__main__":
    unittest.main()
