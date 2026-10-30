"""Decoded b-tag bits, shared guards and pair/ERL compatibility; no EOS."""

import json
from pathlib import Path
import tempfile
import unittest

import awkward as ak
import numpy as np
import torch

from raw_event_conversion import (
    build_object_tensor, build_pair_tensor, convert_raw_parquet_to_pt, object_feature_schema, pair_feature_schema,
)
from raw_event_conversion.objects import (
    BTAG_COLUMNS, BTAG_SOURCES, BTAG_WORKING_POINTS, OBJECT_COLUMNS, OBJECT_FEATURES,
)
from raw_event_conversion.pairs import PAIR_COLUMNS, PAIR_FEATURES, PAIR_OBJECT_INDICES
from raw_event_conversion.tests.test_pairs import records
from set_based_models import CategoryCut, ExistentialRuleList, ObservableSpec, SetInput, TPRFPRRuleScore


def specs(schema):
    return tuple(ObservableSpec(entry["name"], entry["kind"], categories=entry.get("categories"),
                                validity_column=entry.get("validity_column")) for entry in schema)


class BTagTensorTests(unittest.TestCase):
    def test_all_six_bits_of_both_masks_are_decoded_independently(self):
        source = records([{"jets": [{"pt": 100 - i, "btag_mask": value,
                                      "btag_phys_mask": value ^ 63}
                                     for i, value in enumerate((0, 1, 2, 4, 8, 16, 32, 63))]}])
        objects = build_object_tensor(source)
        self.assertEqual(objects.shape, (1, 22, 36))
        for family in BTAG_SOURCES:
            self.assertEqual(objects[0, :8, OBJECT_COLUMNS.index(family + "_valid")].tolist(), [1] * 8)
            for wp, bit in BTAG_WORKING_POINTS.items():
                masks = [value if family == "btag" else value ^ 63 for value in (0, 1, 2, 4, 8, 16, 32, 63)]
                expected = [(value >> bit) & 1 for value in masks]
                self.assertEqual(objects[0, :8, OBJECT_COLUMNS.index(f"{family}_{wp}")].tolist(), expected)
        self.assertTrue(all("mask" not in name and "flavor" not in name for name in OBJECT_COLUMNS))

    def test_rank_filter_cap_and_stable_ties_keep_tag_bits_on_the_same_jet(self):
        jets = [{"pt": pt, "btag_mask": index, "btag_phys_mask": 63 - index}
                for index, pt in enumerate((10, 100, 100, 80, 70, 60, 50, 40, 30, 20, 15, 5, 0, -1))]
        objects = build_object_tensor(records([{"jets": jets}]))
        selected = sorted((index for index, jet in enumerate(jets) if jet["pt"] > 0),
                          key=lambda index: -jets[index]["pt"])[:10]
        for slot, original in enumerate(selected):
            for family, raw in (("btag", original), ("btag_phys", 63 - original)):
                for wp, bit in BTAG_WORKING_POINTS.items():
                    self.assertEqual(objects[0, slot, OBJECT_COLUMNS.index(f"{family}_{wp}")].item(),
                                     (raw >> bit) & 1)

    def test_bad_masks_are_guarded_and_do_not_invalidate_the_other_family(self):
        masks = [None, np.nan, np.inf, -1, 1.5, 2**32, 0, 2**31]
        source = records([{"jets": [{"pt": 100 - i, "btag_mask": mask, "btag_phys_mask": 2}
                                     for i, mask in enumerate(masks)]}])
        objects = build_object_tensor(source)
        self.assertTrue(bool(torch.isfinite(objects).all()))
        self.assertEqual(objects[0, :8, OBJECT_COLUMNS.index("btag_valid")].tolist(), [0] * 6 + [1, 1])
        self.assertEqual(objects[0, :8, OBJECT_COLUMNS.index("btag_phys_valid")].tolist(), [1] * 8)
        self.assertEqual(objects[0, :8, OBJECT_COLUMNS.index("btag_phys_medium")].tolist(), [1] * 8)
        for wp in BTAG_WORKING_POINTS:
            self.assertFalse(bool(objects[0, :8, OBJECT_COLUMNS.index("btag_" + wp)].any()))

    def test_nonjets_padding_and_absent_pairs_have_zero_values_and_false_guards(self):
        source = records([{"jets": [{"pt": 20, "btag_mask": 0}],
                           "muons": [{"pt": 15}], "electrons": [{"pt": 12}], "photons": [{"pt": 10}]}])
        objects = build_object_tensor(source)
        self.assertTrue(bool((objects[0, 1:, [OBJECT_COLUMNS.index(name) for name in BTAG_COLUMNS]] == 0).all()))
        self.assertEqual(objects[0, 0, OBJECT_COLUMNS.index("btag_valid")].item(), 1)
        pairs, mask = build_pair_tensor(source)
        self.assertTrue(bool((pairs[~mask] == 0).all()))
        slot = PAIR_OBJECT_INDICES.index((0, 18))
        self.assertEqual(pairs[0, slot, PAIR_COLUMNS.index("btag_valid_1")].item(), 1)
        self.assertEqual(pairs[0, slot, PAIR_COLUMNS.index("btag_valid_2")].item(), 0)

    def test_pair_members_copy_both_masks_and_preserve_original_column_positions(self):
        source = records([{"jets": [{"pt": 50, "btag_mask": 2, "btag_phys_mask": 32},
                                     {"pt": 40, "btag_mask": 4, "btag_phys_mask": 8}]}])
        objects = build_object_tensor(source)
        pairs, _ = build_pair_tensor(source)
        self.assertEqual(pairs.shape, (1, 231, 102))
        original_features = tuple(name for name in OBJECT_FEATURES if name != "dz")
        original_objects = ("pt", "eta", "phi", "type_id", *original_features,
                            *(name + "_valid" for name in original_features))
        self.assertEqual(OBJECT_COLUMNS[:20], original_objects)
        original_pairs = (*(f"{name}_{member}" for member in (1, 2)
                            for name in (*original_objects, "eta_valid", "phi_valid")),
                          *(column for name in PAIR_FEATURES if name != "abs_delta_dz"
                            for column in (name, name + "_valid")))
        self.assertEqual(PAIR_COLUMNS[:68], original_pairs)
        self.assertEqual(OBJECT_COLUMNS[:34], (*original_objects, *BTAG_COLUMNS))
        self.assertEqual(PAIR_COLUMNS[:96], (*original_pairs,
                         *(f"{name}_{member}" for member in (1, 2) for name in BTAG_COLUMNS)))
        for member, slot in ((1, 0), (2, 1)):
            for name in BTAG_COLUMNS:
                self.assertEqual(pairs[0, 0, PAIR_COLUMNS.index(f"{name}_{member}")].item(),
                                 objects[0, slot, OBJECT_COLUMNS.index(name)].item())

    def test_schemas_have_boolean_shared_guards_and_explicit_bit_definitions(self):
        for schema, columns, members in ((object_feature_schema(), OBJECT_COLUMNS, ("",)),
                                         (pair_feature_schema(), PAIR_COLUMNS, ("_1", "_2"))):
            self.assertEqual([entry["name"] for entry in schema], list(columns))
            json.dumps(schema, allow_nan=False)
            fields = {entry["name"]: entry for entry in schema}
            for member in members:
                for family, raw_suffix in BTAG_SOURCES.items():
                    guard = family + "_valid" + member
                    for wp, bit in BTAG_WORKING_POINTS.items():
                        name = f"{family}_{wp}{member}"
                        self.assertEqual(fields[name]["kind"], "boolean")
                        self.assertEqual(fields[name]["validity_column"], guard)
                        self.assertEqual(fields[name]["bit_number"], bit)
                        self.assertEqual(fields[name]["raw_columns"]["jets"], ["L1T_JetPuppiAK4_" + raw_suffix])
                        self.assertIn(name, fields[guard]["value_columns"])
                        self.assertNotIn("validity_column", fields[guard])

    def test_missing_and_misaligned_masks_are_not_silently_untagged(self):
        source = records([{"jets": [{"pt": 50}, {"pt": 20}]}])
        for suffix in BTAG_SOURCES.values():
            column = "L1T_JetPuppiAK4_" + suffix
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "Missing raw object"):
                    build_object_tensor(ak.without_field(source, column))
                with self.assertRaisesRegex(ValueError, "matching object counts"):
                    build_pair_tensor(ak.with_field(source, [[0]], column))

    def test_false_tag_cuts_cannot_select_nonjets_or_invalid_masks(self):
        source = records([{"jets": [{"pt": 30, "btag_mask": 0}, {"pt": 20, "btag_mask": None}],
                           "photons": [{"pt": 40}]}])
        objects = build_object_tensor(source).numpy()
        data = SetInput.from_arrays(objects, specs(object_feature_schema()), slot_mask=objects[..., 0] > 0)
        cut = CategoryCut(OBJECT_COLUMNS.index("btag_medium"), {0})
        selected = np.flatnonzero(cut.evaluate(data)[0]).tolist()
        self.assertEqual(selected, [0])
        pairs, mask = build_pair_tensor(source)
        pair_data = SetInput.from_arrays(pairs.numpy(), specs(pair_feature_schema()), slot_mask=mask.numpy())
        for member in (1, 2):
            passed = CategoryCut(PAIR_COLUMNS.index(f"btag_medium_{member}"), {0}).evaluate(pair_data)[0]
            for slot in np.flatnonzero(passed):
                original = PAIR_OBJECT_INDICES[slot][member - 1]
                self.assertEqual(original, 0)

    def test_erl_can_find_double_tagged_jet_pairs(self):
        signal_raw = records([{"jets": [{"pt": 50, "btag_mask": 2}, {"pt": 40, "btag_mask": 2}]}] * 2)
        background_raw = records([{"jets": [{"pt": 50, "btag_mask": first},
                                             {"pt": 40, "btag_mask": second}]}
                                  for first, second in ((2, 0), (0, 2))])
        signal, sm = build_pair_tensor(signal_raw)
        background, bm = build_pair_tensor(background_raw)
        names = {"btag_medium_1", "btag_valid_1", "btag_medium_2", "btag_valid_2"}
        schema = [entry for entry in pair_feature_schema() if entry["name"] in names]
        indices = [PAIR_COLUMNS.index(entry["name"]) for entry in schema]
        observables = specs(schema)
        model = ExistentialRuleList(observables, scoring_function=TPRFPRRuleScore(),
                                    max_rule_depth=2, beam_width=8, min_signal_efficiency=.01)
        model.fit(signal.numpy()[..., indices], background.numpy()[..., indices],
                  signal_slot_mask=sm.numpy(), background_slot_mask=bm.numpy())
        self.assertTrue(any(rule.signal_selection.all() and not rule.background_selection.any()
                            and rule.depth == 2 for rule in model.candidate_rules_))
        self.assertTrue(all(not observables[cut.observable_index].name.startswith("btag_valid")
                            for rule in model.candidate_rules_ for cut in rule.cuts))

    def test_conversion_and_mixture_preserve_nonzero_decisions_and_schemas(self):
        from scripts.make_reference_anomalous import make_datasets

        raw = records([{"jets": [{"pt": 50, "btag_mask": 2, "btag_phys_mask": 32},
                                  {"pt": 40, "btag_mask": 8, "btag_phys_mask": 1}]}] * 3)
        with tempfile.TemporaryDirectory(prefix="raw_btag_") as temporary:
            source, output = Path(temporary) / "input.parquet", Path(temporary) / "output.pt"
            ak.to_parquet(raw, source)
            convert_raw_parquet_to_pt(source, output, label=1, include_pair_tensor=True)
            saved = torch.load(output, weights_only=True)
        self.assertEqual(saved["conversion_metadata"]["schema_version"], 12)
        self.assertTrue(saved["conversion_metadata"]["object_tensor"]["btag_decisions_decoded"])
        self.assertEqual(saved["event_features"].shape[-1], 193)
        self.assertEqual(saved["obj_feature_schema"], object_feature_schema())
        self.assertEqual(saved["pair_obj_feature_schema"], pair_feature_schema())
        torch.testing.assert_close(saved["obj"], build_object_tensor(raw))
        torch.testing.assert_close(saved["pair_obj"], build_pair_tensor(raw)[0])
        signal = {**saved, "label": torch.full_like(saved["label"], 2)}
        for sample in make_datasets(saved, signal, reference_count=1, background_count=1, signal_count=1):
            for row, source_row in enumerate(sample["source_index"]):
                torch.testing.assert_close(sample["obj"][row], saved["obj"][source_row])
                torch.testing.assert_close(sample["pair_obj"][row], saved["pair_obj"][source_row])


if __name__ == "__main__":
    unittest.main()
