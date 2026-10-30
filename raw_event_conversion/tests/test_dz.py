"""Signed lepton DZ, guarded pair differences and saved-schema compatibility."""

from pathlib import Path
import tempfile
import unittest

import awkward as ak
import numpy as np
import torch

from raw_event_conversion import (
    build_object_tensor, build_pair_tensor, convert_raw_parquet_to_pt, object_feature_schema, pair_feature_schema,
)
from raw_event_conversion.objects import OBJECT_COLUMNS, RAW_OBJECT_COLUMNS
from raw_event_conversion.pairs import PAIR_COLUMNS
from raw_event_conversion.tests.test_pairs import field, pair_slot, records
from set_based_models import ObservableSpec, SetInput, ThresholdCut


def specs(schema):
    return tuple(ObservableSpec(entry["name"], entry["kind"], categories=entry.get("categories"),
                                validity_column=entry.get("validity_column")) for entry in schema)


class DZTensorTests(unittest.TestCase):
    def test_signed_values_follow_pt_ranks_and_zero_is_valid(self):
        leptons = [{"pt": 10, "dz": 0}, {"pt": 40, "dz": -20},
                   {"pt": 30, "dz": None}, {"pt": 20, "dz": np.inf}]
        objects = build_object_tensor(records([{"muons": leptons, "electrons": leptons,
                                                "jets": [{"pt": 50}], "photons": [{"pt": 60}]}, {}]))
        self.assertEqual(objects.shape, (2, 22, 36))
        self.assertEqual(OBJECT_COLUMNS[34:], ("dz", "dz_valid"))
        for start in (10, 14):
            self.assertEqual(objects[0, start:start + 4, 34].tolist(), [-20, 0, 0, 0])
            self.assertEqual(objects[0, start:start + 4, 35].tolist(), [1, 0, 0, 1])
            self.assertEqual(objects[0, start:start + 4, 0].tolist(), [40, 30, 20, 10])
        self.assertFalse(objects[0, :10, 34:].any())
        self.assertFalse(objects[0, 18:, 34:].any())
        self.assertFalse(objects[1].any())

    def test_pair_difference_preserves_sign_and_accepts_equal_dz(self):
        source = records([{"muons": [{"pt": 40, "dz": 20, "phi": np.nan}, {"pt": 30, "dz": 20}],
                           "electrons": [{"pt": 25, "dz": -20}],
                           "jets": [{"pt": 80}], "photons": [{"pt": 60}]}])
        pairs, mask = build_pair_tensor(source)
        self.assertEqual(pairs.shape, (1, 231, 102))
        self.assertEqual(PAIR_COLUMNS[96:], ("dz_1", "dz_valid_1", "dz_2", "dz_valid_2",
                                            "abs_delta_dz", "abs_delta_dz_valid"))
        self.assertEqual(field(pairs, "abs_delta_dz", slot=pair_slot(10, 14)), 40)
        self.assertEqual(field(pairs, "abs_delta_dz_valid", slot=pair_slot(10, 14)), 1)
        self.assertEqual(field(pairs, "dz_1", slot=pair_slot(10, 14)), 20)
        self.assertEqual(field(pairs, "dz_2", slot=pair_slot(10, 14)), -20)
        self.assertEqual(field(pairs, "abs_delta_dz", slot=pair_slot(10, 11)), 0)
        self.assertEqual(field(pairs, "abs_delta_dz_valid", slot=pair_slot(10, 11)), 1)
        for slot in (pair_slot(0, 10), pair_slot(10, 18), pair_slot(0, 18)):
            self.assertEqual(field(pairs, "abs_delta_dz", slot=slot), 0)
            self.assertEqual(field(pairs, "abs_delta_dz_valid", slot=slot), 0)
        self.assertFalse(pairs[~mask].any())

    def test_missing_nonfinite_and_absent_members_are_invalid(self):
        events = [{"muons": [{"pt": 30, "dz": dz}], "electrons": [{"pt": 20, "dz": 0}]}
                  for dz in (None, np.nan, np.inf, -np.inf, 0)]
        events.append({"electrons": [{"pt": 20, "dz": 0}]})
        pairs, mask = build_pair_tensor(records(events))
        slot = pair_slot(10, 14)
        self.assertEqual(pairs[:, slot, PAIR_COLUMNS.index("abs_delta_dz")].tolist(), [0] * 6)
        self.assertEqual(pairs[:, slot, PAIR_COLUMNS.index("abs_delta_dz_valid")].tolist(), [0, 0, 0, 0, 1, 0])
        self.assertEqual(mask[:, slot].tolist(), [True] * 5 + [False])
        self.assertTrue(torch.isfinite(pairs).all())

    def test_difference_is_calculated_before_float32_rounding(self):
        pairs, _ = build_pair_tensor(records([{"muons": [{"pt": 30, "dz": 10000},
                                                         {"pt": 20, "dz": 10000 + 1e-5}]}]))
        slot = pair_slot(10, 11)
        self.assertEqual(field(pairs, "dz_1", slot=slot), field(pairs, "dz_2", slot=slot))
        self.assertAlmostEqual(field(pairs, "abs_delta_dz", slot=slot), 1e-5, places=10)
        self.assertEqual(field(pairs, "abs_delta_dz_valid", slot=slot), 1)

    def test_schema_guards_exclude_padding_and_inapplicable_types_from_cuts(self):
        source = records([{"muons": [{"pt": 30, "dz": 0}], "electrons": [{"pt": 20, "dz": 0}],
                           "jets": [{"pt": 50}], "photons": [{"pt": 40}]},
                          {"muons": [{"pt": 30, "dz": None}], "electrons": [{"pt": 20, "dz": 0}]}])
        objects = build_object_tensor(source)
        pairs, mask = build_pair_tensor(source)
        for tensor, schema, columns, presence, name, counts in (
            (objects, object_feature_schema(), OBJECT_COLUMNS, objects[..., 0] > 0, "dz", [2, 1]),
            (pairs, pair_feature_schema(), PAIR_COLUMNS, mask, "abs_delta_dz", [1, 0]),
        ):
            data = SetInput.from_arrays(tensor.numpy(), specs(schema), slot_mask=presence.numpy())
            passed = ThresholdCut(columns.index(name), "<=", 0).evaluate(data)
            self.assertEqual(passed.sum(axis=1).tolist(), counts)
            feature = next(entry for entry in schema if entry["name"] == name)
            self.assertEqual(feature["units"], "mm")
            self.assertEqual(feature["validity_column"], name + "_valid")
        schema = {entry["name"]: entry for entry in pair_feature_schema()}
        for member in (1, 2):
            self.assertEqual(schema[f"dz_{member}"]["validity_column"], f"dz_valid_{member}")
        self.assertIn("L1T_Electron_DZ", RAW_OBJECT_COLUMNS)
        self.assertIn("L1T_MuonTight_DZ", RAW_OBJECT_COLUMNS)
        self.assertFalse(any(name.endswith("_ErrorDZ") for name in RAW_OBJECT_COLUMNS))

    def test_conversion_saves_signed_dz_and_pair_schema_without_changing_events(self):
        source = records([{"muons": [{"pt": 30, "dz": 20}], "electrons": [{"pt": 20, "dz": -20}]}])
        with tempfile.TemporaryDirectory(prefix="raw_dz_") as temporary:
            raw, ordinary, paired = [Path(temporary) / name for name in ("raw.parquet", "ordinary.pt", "paired.pt")]
            ak.to_parquet(source, raw)
            convert_raw_parquet_to_pt(raw, ordinary, label=1)
            convert_raw_parquet_to_pt(raw, paired, label=1, include_pair_tensor=True)
            obj_data, pair_data = [torch.load(path, weights_only=True) for path in (ordinary, paired)]
        self.assertEqual(obj_data["conversion_metadata"]["schema_version"], 11)
        self.assertEqual(pair_data["conversion_metadata"]["schema_version"], 12)
        self.assertEqual(pair_data["obj_feature_schema"], object_feature_schema())
        self.assertEqual(pair_data["pair_obj_feature_schema"], pair_feature_schema())
        torch.testing.assert_close(obj_data["event_features"], pair_data["event_features"])
        torch.testing.assert_close(obj_data["obj"], pair_data["obj"])
        self.assertEqual(pair_data["obj"][0, 14, OBJECT_COLUMNS.index("dz")].item(), -20)
        self.assertEqual(field(pair_data["pair_obj"], "abs_delta_dz", slot=pair_slot(10, 14)), 40)


if __name__ == "__main__":
    unittest.main()
