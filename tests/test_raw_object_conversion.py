"""Raw object tensor tests; synthetic input only, no production conversion."""

from pathlib import Path
import sys
import tempfile
import unittest

import awkward as ak
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raw_event_conversion import build_object_tensor, convert_raw_parquet_to_pt, object_feature_schema
from raw_event_conversion.objects import OBJECT_COLUMNS, OBJECT_FEATURES, OBJECT_SLOTS, RAW_OBJECT_COLUMNS
from tests.test_raw_event_conversion import raw_fixture, with_pt


class RawObjectTensorTests(unittest.TestCase):
    def feature_records(self):
        records = raw_fixture()[:1]
        for kind, prefix in (("jets", "L1T_JetPuppiAK4"), ("muons", "L1T_MuonTight"),
                             ("electrons", "L1T_Electron"), ("photons", "L1T_PhotonTight")):
            records = with_pt(records, ak.Array([[10, 50, 30]]), prefix + "_PT")
            values = {"D0": [0, -2, np.nan], "IsolationVarRhoCorr": [0, .25, None],
                      "Mass": [0, 12, -1], "NCharged": [0, 3, 1], "NNeutrals": [0, 1, 3],
                      "EhadOverEem": [0, .15, None], "Charge": [1, -1, 0]}
            for _, (suffix, types, _, _) in OBJECT_FEATURES.items():
                if suffix is not None and kind in types:
                    records = ak.with_field(records, [values[suffix]], prefix + "_" + suffix)
        return records

    def feature(self, objects, name):
        return objects[0, :, OBJECT_COLUMNS.index(name)]

    def records(self, pt, eta=None, phi=None):
        if eta is None:
            eta = ak.zeros_like(ak.Array(pt))
        if phi is None:
            phi = ak.zeros_like(ak.Array(pt))
        records = {column: [[] for _ in pt] for column in RAW_OBJECT_COLUMNS}
        records.update({"L1T_JetPuppiAK4_PT": pt, "L1T_JetPuppiAK4_Eta": eta,
                        "L1T_JetPuppiAK4_Phi": phi})
        for column in RAW_OBJECT_COLUMNS:
            if column.startswith("L1T_JetPuppiAK4_") and column not in (
                    "L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"):
                records[column] = ak.zeros_like(ak.Array(pt))
        return ak.Array(records)

    def test_layout_caps_type_ids_and_padding(self):
        objects = build_object_tensor(raw_fixture())
        self.assertEqual(tuple(objects.shape), (3, 22, 20))
        self.assertEqual(objects.dtype, torch.float32)
        self.assertFalse(bool(objects[0].any()))
        self.assertEqual(OBJECT_COLUMNS[:4], ("pt", "eta", "phi", "type_id"))
        start = 0
        for type_id, (kind, width) in enumerate(OBJECT_SLOTS.items()):
            group = objects[2, start:start + width]
            self.assertEqual(group[:, 0].tolist(), [120 if kind == "jets" else 60] * width)
            self.assertEqual(group[:, 3].tolist(), [type_id] * width)
            start += width

    def test_descending_pt_preserves_coordinates_and_stable_ties(self):
        records = self.records([[10, 50, 50, 20]], [[.1, .2, .3, .4]], [[1, 2, 3, -1]])
        objects = build_object_tensor(records)
        torch.testing.assert_close(objects[0, :4, :3], torch.tensor(
            [[50, .2, 2], [50, .3, 3], [20, .4, -1], [10, .1, 1]], dtype=torch.float32))
        self.assertFalse(bool(objects[0, 4:].any()))

    def test_cap_keeps_highest_pt_not_first_producer_objects(self):
        objects = build_object_tensor(self.records([list(range(1, 16))]))
        self.assertEqual(objects[0, :10, 0].tolist(), list(range(15, 5, -1)))

    def test_nonpositive_missing_and_nonfinite_pt_do_not_fill_slots(self):
        pt = [[0, -1, None, np.nan, np.inf, 2, 1]]
        objects = build_object_tensor(self.records(pt))
        self.assertEqual(objects[0, :2, 0].tolist(), [2, 1])
        self.assertFalse(bool(objects[0, 2:].any()))

    def test_invalid_coordinates_are_zero_filled_without_reranking(self):
        records = self.records([[50, 20, 10]], [[None, np.nan, .1]], [[np.inf, -1, None]])
        objects = build_object_tensor(records)
        self.assertTrue(bool(torch.isfinite(objects).all()))
        self.assertEqual(objects[0, :3, 0].tolist(), [50, 20, 10])
        self.assertEqual(objects[0, :3, 1].tolist()[:2], [0, 0])
        self.assertEqual(objects[0, :3, 2].tolist(), [0, -1, 0])

    def test_missing_and_entirely_empty_collections_are_zero_padded(self):
        for entries in ([[], []], [None, None], [[None], [None]]):
            records = ak.Array({column: entries for column in RAW_OBJECT_COLUMNS})
            with self.subTest(entries=entries):
                objects = build_object_tensor(records)
                self.assertEqual(tuple(objects.shape), (2, 22, 20))
                self.assertFalse(bool(objects.any()))

    def test_no_met_or_unweighted_jet_fields_are_required(self):
        objects = build_object_tensor(self.records([[100]]))
        self.assertEqual(objects[0, 0, 0].item(), 100)
        self.assertEqual(objects[..., 3].max().item(), 0)
        self.assertFalse(any("MET" in column or "JetAK4" in column for column in RAW_OBJECT_COLUMNS))

    def test_missing_columns_and_misaligned_angles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing raw object columns"):
            build_object_tensor(ak.without_field(self.records([[10]]), "L1T_Electron_Phi"))
        for column in ("L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "matching object counts"):
                build_object_tensor(ak.with_field(self.records([[10, 20]]), [[0]], column))

    def test_float32_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "float32"):
            build_object_tensor(self.records([[1e40]]))

    def test_pipeline_keeps_events_full_counts_and_source_alignment(self):
        with tempfile.TemporaryDirectory(prefix="raw_objects_") as directory:
            directory = Path(directory)
            source = directory / "raw.parquet"
            output = directory / "events.pt"
            ak.to_parquet(raw_fixture(), source)
            convert_raw_parquet_to_pt(source, output, label=5)
            data = torch.load(output, weights_only=True)
            self.assertEqual(data["obj_columns"], list(OBJECT_COLUMNS))
            self.assertEqual(data["obj_feature_schema"], object_feature_schema())
            self.assertEqual(data["label"].tolist(), [5] * 3)
            self.assertEqual(data["raw_source_row"].tolist(), [0, 1, 2])
            self.assertFalse(bool(data["obj"][0].any()))
            self.assertEqual(data["event_features"][2, :4].tolist(), [7, 8, 9, 15])
            meta = data["conversion_metadata"]
            self.assertEqual(meta["schema_version"], 7)
            self.assertTrue(meta["dummy_pf"])
            self.assertFalse(meta["dummy_pf_and_obj"])
            self.assertTrue(meta["object_truncation"])
            self.assertFalse(meta["object_tensor"]["met_included"])
            self.assertEqual(meta["object_tensor"]["collections"]["L1T_JetPuppiAK4"], 10)
            # No MET slot, but the event-level MET and native info are unchanged.
            met = data["event_feature_columns"].index("met")
            self.assertEqual(data["event_features"][:, met].tolist(), [0, 50, 100])
            self.assertEqual(tuple(data["native_event_info"].shape), (3, 10))
            from casual_tests._event_only_reporting import load_event_dataset
            loaded = load_event_dataset(output)
            self.assertEqual(loaded.objects.shape, (3, 1, 1))
            self.assertFalse(loaded.slot_mask.any())
            np.testing.assert_array_equal(loaded.event_features, data["event_features"].numpy())

    def test_multi_file_row_cap_preserves_object_alignment(self):
        with tempfile.TemporaryDirectory(prefix="raw_objects_") as directory:
            directory = Path(directory)
            files = [directory / "first.parquet", directory / "second.parquet"]
            for path in files:
                ak.to_parquet(raw_fixture(), path)
            output = directory / "events.pt"
            convert_raw_parquet_to_pt(files, output, label=1, max_events=4)
            data = torch.load(output, weights_only=True)
            self.assertEqual(tuple(data["obj"].shape), (4, 22, 20))
            self.assertEqual(data["raw_source_file"].tolist(), [0, 0, 0, 1])
            self.assertFalse(bool(data["obj"][3].any()))
            torch.testing.assert_close(data["obj"][:3], build_object_tensor(raw_fixture()))

    def test_extra_values_follow_selected_pt_ranks(self):
        objects = build_object_tensor(self.feature_records())
        for slot in (10, 14):
            self.assertEqual(self.feature(objects, "abs_d0")[slot].item(), 2)
            self.assertEqual(self.feature(objects, "abs_d0_valid")[slot].item(), 1)
            self.assertEqual(self.feature(objects, "abs_d0_valid")[slot + 1].item(), 0)
            self.assertEqual(self.feature(objects, "abs_d0")[slot + 1].item(), 0)
            self.assertEqual(self.feature(objects, "abs_d0")[slot + 2].item(), 0)
            self.assertEqual(self.feature(objects, "abs_d0_valid")[slot + 2].item(), 1)
        self.assertEqual(self.feature(objects, "mass")[:3].tolist(), [12, 0, 0])
        self.assertEqual(self.feature(objects, "mass_valid")[:3].tolist(), [1, 0, 1])
        self.assertEqual(self.feature(objects, "n_charged")[:3].tolist(), [3, 1, 0])
        self.assertEqual(self.feature(objects, "n_neutrals")[:3].tolist(), [1, 3, 0])
        self.assertEqual(self.feature(objects, "charged_fraction")[:3].tolist(), [.75, .25, 0])
        self.assertEqual(self.feature(objects, "charged_fraction_valid")[:3].tolist(), [1, 1, 0])
        for slot in (10, 14, 18):
            self.assertEqual(self.feature(objects, "isolation_rho_corr")[slot].item(), .25)
            self.assertEqual(self.feature(objects, "isolation_rho_corr_valid")[slot:slot + 3].tolist(), [1, 0, 1])
        for slot in (14, 18):
            self.assertAlmostEqual(self.feature(objects, "ehad_over_eem")[slot].item(), .15, places=6)
            self.assertEqual(self.feature(objects, "ehad_over_eem_valid")[slot:slot + 3].tolist(), [1, 0, 1])
        for slot in (0, 10, 14):
            self.assertEqual(self.feature(objects, "charge")[slot:slot + 3].tolist(), [-1, 0, 1])
            self.assertEqual(self.feature(objects, "charge_valid")[slot:slot + 3].tolist(), [1, 1, 1])

    def test_inapplicable_fields_and_padded_slots_are_zero_false(self):
        objects = build_object_tensor(self.feature_records())
        for name, (_, applicable, _, _) in OBJECT_FEATURES.items():
            start = 0
            for kind, width in OBJECT_SLOTS.items():
                if kind not in applicable:
                    self.assertFalse(bool(self.feature(objects, name)[start:start + width].any()))
                    self.assertFalse(bool(self.feature(objects, name + "_valid")[start:start + width].any()))
                self.assertFalse(bool(objects[0, start + 3:start + width].any()))
                start += width

    def test_invalid_counts_and_undefined_fraction_are_never_physical_zero(self):
        charged, neutral = [0, 1, -1, 1.5, np.nan, 2], [0, 0, 2, 2, 2, None]
        records = self.records([[10] * len(charged)])
        records = ak.with_field(records, [charged], "L1T_JetPuppiAK4_NCharged")
        records = ak.with_field(records, [neutral], "L1T_JetPuppiAK4_NNeutrals")
        objects = build_object_tensor(records)
        self.assertEqual(self.feature(objects, "n_charged_valid")[:6].tolist(), [1, 1, 0, 0, 0, 1])
        self.assertEqual(self.feature(objects, "n_neutrals_valid")[:6].tolist(), [1, 1, 1, 1, 1, 0])
        self.assertEqual(self.feature(objects, "charged_fraction")[:6].tolist(), [0, 1, 0, 0, 0, 0])
        self.assertEqual(self.feature(objects, "charged_fraction_valid")[:6].tolist(), [0, 1, 0, 0, 0, 0])
        self.assertTrue(bool(torch.isfinite(objects).all()))

    def test_zero_charged_fraction_with_positive_neutral_count_is_valid(self):
        records = ak.with_field(self.records([[10]]), [[3]], "L1T_JetPuppiAK4_NNeutrals")
        objects = build_object_tensor(records)
        self.assertEqual(self.feature(objects, "charged_fraction")[0].item(), 0)
        self.assertEqual(self.feature(objects, "charged_fraction_valid")[0].item(), 1)

    def test_nonfinite_extras_are_zero_with_false_flags(self):
        records = self.feature_records()
        for name, (suffix, types, _, _) in OBJECT_FEATURES.items():
            if suffix is None:
                continue
            kind = types[0]
            prefix = {"jets": "L1T_JetPuppiAK4", "muons": "L1T_MuonTight",
                      "electrons": "L1T_Electron", "photons": "L1T_PhotonTight"}[kind]
            changed = ak.with_field(records, [[np.nan, np.inf, None]], prefix + "_" + suffix)
            objects = build_object_tensor(changed)
            start = {"jets": 0, "muons": 10, "electrons": 14, "photons": 18}[kind]
            with self.subTest(name=name):
                self.assertFalse(bool(self.feature(objects, name)[start:start + 3].any()))
                self.assertFalse(bool(self.feature(objects, name + "_valid")[start:start + 3].any()))
                self.assertTrue(bool(torch.isfinite(objects).all()))

    def test_extra_columns_are_required_and_must_align_even_without_valid_pt(self):
        for column in ("L1T_Electron_IsolationVarRhoCorr", "L1T_PhotonTight_EhadOverEem",
                       "L1T_JetPuppiAK4_NCharged", "L1T_MuonTight_Charge"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Missing raw object"):
                build_object_tensor(ak.without_field(raw_fixture(), column))
        records = self.records([[0, -1]])
        with self.assertRaisesRegex(ValueError, "matching object counts"):
            build_object_tensor(ak.with_field(records, [[0]], "L1T_JetPuppiAK4_Mass"))

    def test_schema_covers_all_twenty_columns_and_correct_units(self):
        import json
        schema = object_feature_schema()
        self.assertEqual([field["name"] for field in schema], list(OBJECT_COLUMNS))
        self.assertEqual(len(schema), 20)
        fields = {field["name"]: field for field in schema}
        self.assertEqual(fields["abs_d0"]["units"], "mm")
        self.assertEqual(fields["abs_d0"]["definition"], "abs(D0)")
        self.assertEqual(fields["mass"]["units"], "GeV")
        self.assertEqual(fields["charge"]["kind"], "ordinal")
        self.assertEqual(fields["type_id"]["categories"], {0: "jets", 1: "muons", 2: "electrons", 3: "photons"})
        self.assertEqual(fields["charged_fraction"]["raw_columns"]["jets"],
                         ["L1T_JetPuppiAK4_NCharged", "L1T_JetPuppiAK4_NNeutrals"])
        for name in OBJECT_FEATURES:
            self.assertEqual(fields[name]["validity_column"], name + "_valid")
            self.assertEqual(fields[name + "_valid"]["kind"], "boolean")
        json.dumps(schema, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
