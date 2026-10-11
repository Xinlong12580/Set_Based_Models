"""Pair construction and conversion checks; no production files or EOS access."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import awkward as ak
import numpy as np
import torch

from raw_event_conversion import build_object_tensor, build_pair_tensor, convert_raw_parquet_to_pt, pair_feature_schema
from raw_event_conversion.__main__ import main
from raw_event_conversion.objects import OBJECT_COLUMNS, OBJECT_SLOTS, _source_columns
from raw_event_conversion.pairs import LEPTON_MASSES, PAIR_COLUMNS, PAIR_OBJECT_INDICES


def records(events):
    """Build complete raw columns from small per-event, per-type dictionaries."""
    columns = {}
    for kind in OBJECT_SLOTS:
        for name, column in _source_columns(kind).items():
            defaults = {"n_neutrals": 1, "charge": 1 if kind in LEPTON_MASSES else 0}
            columns[column] = [[obj.get(name, defaults.get(name, 0))
                                for obj in event.get(kind, [])] for event in events]
    for column in ("L1T_PUPPIMET_MET", "L1T_PUPPIMET_Phi", "L1T_MET_MET", "L1T_MET_Phi", "L1T_ScalarHT_HT"):
        columns[column] = [[0.] for _ in events]
    columns["L1T_Rho_Rho"] = [[0.] * 5 for _ in events]
    for suffix in ("PT", "Charge", "D0", "ErrorD0"):
        columns["L1T_PFPart_" + suffix] = [[] for _ in events]
    return ak.Array(columns)


def pair_slot(first, second):
    return PAIR_OBJECT_INDICES.index((first, second))


def field(tensor, name, event=0, slot=None):
    values = tensor[event, :, PAIR_COLUMNS.index(name)]
    return values if slot is None else values[slot].item()


class PairTensorTests(unittest.TestCase):
    def test_counts_caps_empty_and_single_object_events(self):
        source = records([{}, {"photons": [{"pt": 10}]},
                          {"jets": [{"pt": 20}, {"pt": 10}], "photons": [{"pt": 15}]},
                          {kind: [{"pt": 50}] * (cap + 2) for kind, cap in OBJECT_SLOTS.items()}])
        pairs, mask = build_pair_tensor(source)
        self.assertEqual(pairs.shape, (4, 231, 68))
        self.assertEqual(pairs.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.sum(dim=1).tolist(), [0, 0, 3, 231])
        self.assertTrue(torch.all(pairs[~mask] == 0))
        self.assertEqual(len(set(PAIR_OBJECT_INDICES)), 231)
        self.assertTrue(all(a < b for a, b in PAIR_OBJECT_INDICES))

    def test_members_match_original_objects_and_canonical_order(self):
        source = records([{"jets": [{"pt": 10, "mass": 2}, {"pt": 30, "mass": 4}],
                           "photons": [{"pt": 50, "phi": 1}, {"pt": 50, "phi": 2}]}])
        objects = build_object_tensor(source)
        pairs, mask = build_pair_tensor(source)
        for slot in torch.nonzero(mask[0]).flatten().tolist():
            a, b = PAIR_OBJECT_INDICES[slot]
            for member, original in ((1, a), (2, b)):
                actual = [field(pairs, f"{name}_{member}", slot=slot) for name in OBJECT_COLUMNS]
                np.testing.assert_array_equal(actual, objects[0, original].numpy())
        slot = pair_slot(0, 18)
        self.assertEqual(field(pairs, "type_id_1", slot=slot), 0)
        self.assertEqual(field(pairs, "type_id_2", slot=slot), 3)
        slot = pair_slot(18, 19)
        self.assertEqual(field(pairs, "phi_1", slot=slot), 1)
        self.assertEqual(field(pairs, "phi_2", slot=slot), 2)

    def test_massless_back_to_back_and_vector_scalar_sums(self):
        pairs, _ = build_pair_tensor(records([{"photons": [{"pt": 50}, {"pt": 50, "phi": np.pi}]}]))
        slot = pair_slot(18, 19)
        self.assertAlmostEqual(field(pairs, "pair_mass", slot=slot), 100, places=5)
        self.assertAlmostEqual(field(pairs, "pair_pt", slot=slot), 0, places=5)
        self.assertEqual(field(pairs, "sum_pt", slot=slot), 100)
        self.assertEqual(field(pairs, "min_pt", slot=slot), 50)
        self.assertEqual(field(pairs, "pt_balance", slot=slot), 1)
        self.assertEqual(field(pairs, "pair_mass_valid", slot=slot), 1)

    def test_angular_wrapping_and_raw_precision(self):
        pairs, _ = build_pair_tensor(records([{"photons": [
            {"pt": 50, "phi": np.pi - .01}, {"pt": 40, "phi": -np.pi + .01}]}]))
        self.assertAlmostEqual(field(pairs, "abs_delta_phi", slot=pair_slot(18, 19)), .02, places=6)
        # Different raw angles collapse to the same float32 angle, but the pair
        # computation must preserve their original separation before downcast.
        source = records([{"photons": [{"pt": 50, "phi": 1}, {"pt": 40, "phi": 1 + 1e-8}]}])
        pairs, _ = build_pair_tensor(source)
        self.assertGreater(field(pairs, "abs_delta_phi", slot=pair_slot(18, 19)), 0)

    def test_ratios_angles_and_hemispheres(self):
        pairs, _ = build_pair_tensor(records([{"photons": [
            {"pt": 40, "eta": -.3}, {"pt": 20, "eta": .4, "phi": np.pi / 2}]}]))
        slot = pair_slot(18, 19)
        expected = {"min_pt": 20, "sum_pt": 60, "pt_balance": .5, "pair_pt": np.hypot(40, 20),
                    "abs_delta_eta": .7, "abs_delta_phi": np.pi / 2,
                    "delta_r": np.hypot(.7, np.pi / 2), "opposite_eta_hemispheres": 1}
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(field(pairs, name, slot=slot), value, places=5)
                self.assertEqual(field(pairs, name + "_valid", slot=slot), 1)

    def test_lepton_masses_and_collinear_numerical_stability(self):
        for kind, mass in LEPTON_MASSES.items():
            pairs, mask = build_pair_tensor(records([{kind: [{"pt": 1000, "eta": 12}] * 2}]))
            slot = torch.nonzero(mask[0]).item()
            self.assertAlmostEqual(field(pairs, "pair_mass", slot=slot), 2 * mass, places=7)
        pairs, _ = build_pair_tensor(records([{"photons": [{"pt": 1000, "eta": 12}] * 2}]))
        self.assertEqual(field(pairs, "pair_mass", slot=pair_slot(18, 19)), 0)
        self.assertEqual(field(pairs, "pair_mass_valid", slot=pair_slot(18, 19)), 1)

    def test_massive_jets_match_direct_four_vectors(self):
        jets = [{"pt": 70, "eta": .8, "phi": .1, "mass": 15},
                {"pt": 45, "eta": -1.2, "phi": 2.8, "mass": 25}]
        vectors = []
        for jet in jets:
            momentum = np.array([jet["pt"] * np.cos(jet["phi"]), jet["pt"] * np.sin(jet["phi"]),
                                 jet["pt"] * np.sinh(jet["eta"])])
            vectors.append(np.r_[np.sqrt(momentum @ momentum + jet["mass"]**2), momentum])
        total = sum(vectors)
        expected = np.sqrt(total[0]**2 - total[1:] @ total[1:])
        pairs, _ = build_pair_tensor(records([{"jets": jets}]))
        self.assertAlmostEqual(field(pairs, "pair_mass", slot=pair_slot(0, 1)), expected, places=4)

    def test_missing_jet_mass_never_defaults_to_massless(self):
        pairs, _ = build_pair_tensor(records([{"jets": [{"pt": 50, "mass": None}],
                                             "photons": [{"pt": 40, "phi": np.pi}]}]))
        slot = pair_slot(0, 18)
        self.assertEqual(field(pairs, "pair_mass", slot=slot), 0)
        self.assertEqual(field(pairs, "pair_mass_valid", slot=slot), 0)
        self.assertEqual(field(pairs, "pair_pt_valid", slot=slot), 1)

    def test_coordinate_validity_survives_zero_fill_without_reranking(self):
        pairs, _ = build_pair_tensor(records([{"photons": [
            {"pt": 50, "eta": None}, {"pt": 40, "phi": np.nan}, {"pt": 30, "eta": 0, "phi": 1}]}]))
        slot = pair_slot(18, 19)
        self.assertEqual(field(pairs, "pt_1", slot=slot), 50)
        self.assertEqual(field(pairs, "eta_1", slot=slot), 0)
        self.assertEqual(field(pairs, "eta_valid_1", slot=slot), 0)
        self.assertEqual(field(pairs, "phi_valid_2", slot=slot), 0)
        for name in ("pair_mass", "pair_pt", "delta_r", "abs_delta_eta", "abs_delta_phi"):
            self.assertEqual(field(pairs, name + "_valid", slot=slot), 0)
        slot = pair_slot(18, 20)
        self.assertEqual(field(pairs, "abs_delta_phi_valid", slot=slot), 1)
        self.assertEqual(field(pairs, "eta_valid_2", slot=slot), 1)
        self.assertEqual(field(pairs, "min_pt_valid", slot=slot), 1)

    def test_conditional_isolation_d0_and_charge_features(self):
        pairs, _ = build_pair_tensor(records([{"muons": [
            {"pt": 20, "charge": 1, "abs_d0": -.3, "isolation_rho_corr": -.02},
            {"pt": 10, "charge": -1, "abs_d0": .8, "isolation_rho_corr": .15}]}]))
        slot = pair_slot(10, 11)
        self.assertEqual(field(pairs, "opposite_sign", slot=slot), 1)
        self.assertEqual(field(pairs, "opposite_sign_valid", slot=slot), 1)
        self.assertAlmostEqual(field(pairs, "min_abs_d0", slot=slot), .3, places=6)
        self.assertAlmostEqual(field(pairs, "max_isolation_rho_corr", slot=slot), .15, places=6)
        pairs, _ = build_pair_tensor(records([{"jets": [{"pt": 50}], "photons": [{"pt": 40}]}]))
        slot = pair_slot(0, 18)
        for name in ("opposite_sign", "min_abs_d0", "max_isolation_rho_corr"):
            self.assertEqual(field(pairs, name, slot=slot), 0)
            self.assertEqual(field(pairs, name + "_valid", slot=slot), 0)

    def test_unknown_lepton_charge_is_invalid_not_same_sign(self):
        pairs, _ = build_pair_tensor(records([{"muons": [{"pt": 20, "charge": 0}, {"pt": 10}]}]))
        self.assertEqual(field(pairs, "opposite_sign_valid", slot=pair_slot(10, 11)), 0)

    def test_nonfinite_computations_are_guarded_not_saved_as_infinity(self):
        pairs, _ = build_pair_tensor(records([{"photons": [{"pt": 50, "eta": 1000}, {"pt": 40}]}]))
        self.assertTrue(torch.all(torch.isfinite(pairs)))
        self.assertEqual(field(pairs, "pair_mass_valid", slot=pair_slot(18, 19)), 0)

    def test_schema_and_mask_fit_existing_erl_api_with_type_priority(self):
        from set_based_models import ExistentialRuleList, ObservableSpec, SetInput, ThresholdCut, TPRFPRRuleScore
        schema = pair_feature_schema()
        self.assertEqual([spec["name"] for spec in schema], list(PAIR_COLUMNS))
        self.assertEqual(len(set(PAIR_COLUMNS)), 68)
        json.dumps(schema, allow_nan=False)
        specs = tuple(ObservableSpec(spec["name"], spec["kind"], categories=spec.get("categories"),
                                     validity_column=spec.get("validity_column")) for spec in schema)
        signal, sm = build_pair_tensor(records([{"photons": [{"pt": 50}, {"pt": 40}]}]))
        background, bm = build_pair_tensor(records([{"jets": [{"pt": 30}, {"pt": 20}]}]))
        model = ExistentialRuleList(specs, scoring_function=TPRFPRRuleScore(), max_rule_depth=2,
                                    beam_width=8, min_signal_efficiency=.01,
                                    object_cut_priority=["type_id_1", "type_id_2"])
        model.fit(signal.numpy(), background.numpy(), signal_slot_mask=sm.numpy(), background_slot_mask=bm.numpy())
        self.assertTrue(any(rule.depth == 2 for rule in model.candidate_rules_))
        for rule in model.candidate_rules_:
            names = [specs[cut.observable_index].name for cut in rule.cuts]
            self.assertEqual(names, ["type_id_1", "type_id_2"][:rule.depth])
            np.testing.assert_array_equal(rule.predict(signal.numpy(), slot_mask=sm.numpy()), rule.signal_selection)
        invalid, mask = build_pair_tensor(records([{"photons": [{"pt": 50, "eta": None}, {"pt": 40}]}]))
        data = SetInput.from_arrays(invalid.numpy(), specs, slot_mask=mask.numpy())
        self.assertFalse(ThresholdCut(PAIR_COLUMNS.index("eta_1"), "<=", 1).evaluate(data).any())

    def test_erl_can_expand_to_pair_features_after_member_types(self):
        from set_based_models import ExistentialRuleList, ObservableSpec, TPRFPRRuleScore
        # Isolate this derived feature: with every column enabled, pt_2 gives
        # the same selections and ERL legitimately deduplicates min_pt away.
        names = ("type_id_1", "type_id_2", "min_pt", "min_pt_valid")
        specs = tuple(ObservableSpec(spec["name"], spec["kind"], categories=spec.get("categories"),
                                     validity_column=spec.get("validity_column"))
                      for spec in pair_feature_schema() if spec["name"] in names)
        signal, sm = build_pair_tensor(records([{"photons": [{"pt": 50}, {"pt": 40}]}]))
        background, bm = build_pair_tensor(records([{"photons": [{"pt": 50}, {"pt": 10}]}]))
        indices = [PAIR_COLUMNS.index(spec.name) for spec in specs]
        signal, background = signal[..., indices], background[..., indices]
        model = ExistentialRuleList(specs, scoring_function=TPRFPRRuleScore(), max_rule_depth=3,
                                    beam_width=8, min_signal_efficiency=.01,
                                    object_cut_priority=["type_id_1", "type_id_2"])
        model.fit(signal.numpy(), background.numpy(), signal_slot_mask=sm.numpy(), background_slot_mask=bm.numpy())
        rules = [rule for rule in model.candidate_rules_
                 if any(specs[cut.observable_index].name == "min_pt" for cut in rule.cuts)]
        self.assertTrue(rules)
        self.assertTrue(any(rule.predict(signal.numpy(), slot_mask=sm.numpy()).all()
                            and not rule.predict(background.numpy(), slot_mask=bm.numpy()).any()
                            for rule in rules))


class PairConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pair_conversion_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "raw.parquet"
        self.raw = records([{}, {"photons": [{"pt": 50}, {"pt": 40}]}, {"jets": [{"pt": 30}]}])
        ak.to_parquet(self.raw, self.source)

    def convert(self, name, **kwargs):
        output = self.root / name
        convert_raw_parquet_to_pt(self.source, output, label=1, **kwargs)
        return torch.load(output, weights_only=True)

    def test_pair_generation_is_opt_in_and_preserves_original_tensors(self):
        original = self.convert("default.pt")
        paired = self.convert("paired.pt", include_pair_tensor=True)
        self.assertNotIn("pair_obj", original)
        self.assertEqual(original["conversion_metadata"]["schema_version"], 7)
        self.assertEqual(paired["conversion_metadata"]["schema_version"], 8)
        for name, tensor in original.items():
            if isinstance(tensor, torch.Tensor):
                torch.testing.assert_close(paired[name], tensor)
        expected, mask = build_pair_tensor(self.raw)
        torch.testing.assert_close(paired["pair_obj"], expected)
        torch.testing.assert_close(paired["pair_slot_mask"], mask)
        self.assertEqual(paired["pair_object_indices"], [list(pair) for pair in PAIR_OBJECT_INDICES])
        self.assertEqual(paired["pair_obj_columns"], list(PAIR_COLUMNS))
        self.assertEqual(paired["pair_obj_feature_schema"], pair_feature_schema())

    def test_multi_file_concatenation_and_row_cap(self):
        second = self.root / "second.parquet"
        ak.to_parquet(self.raw[1:], second)
        output = self.root / "combined.pt"
        convert_raw_parquet_to_pt([self.source, second], output, label=1, max_events=4, include_pair_tensor=True)
        saved = torch.load(output, weights_only=True)
        self.assertEqual(saved["pair_slot_mask"].sum(dim=1).tolist(), [0, 1, 0, 1])
        self.assertEqual(saved["raw_source_file"].tolist(), [0, 0, 0, 1])
        self.assertEqual(saved["raw_source_row"].tolist(), [0, 1, 2, 0])

    def test_cli_option_and_mixture_preserve_pairs(self):
        from scripts.make_reference_anomalous import make_datasets
        output = self.root / "cli.pt"
        with redirect_stdout(io.StringIO()):
            main(["--input", str(self.source), "--output", str(output), "--label", "1", "--with-pair-tensor"])
        saved = torch.load(output, weights_only=True)
        signal = {**saved, "label": torch.full_like(saved["label"], 2)}
        reference, anomaly = make_datasets(saved, signal, reference_count=2, background_count=1, signal_count=1, seed=7)
        for dataset in (reference, anomaly):
            self.assertEqual(dataset["pair_obj_feature_schema"], saved["pair_obj_feature_schema"])
            for row, (source_id, source_row) in enumerate(zip(dataset["source_id"], dataset["source_index"])):
                source = saved if source_id == 0 else signal
                for key in ("obj", "pair_obj", "pair_slot_mask", "event_features"):
                    torch.testing.assert_close(dataset[key][row], source[key][source_row])

    def test_invalid_pair_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Boolean"):
            self.convert("bad.pt", include_pair_tensor="yes")


if __name__ == "__main__":
    unittest.main()
