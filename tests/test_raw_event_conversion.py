"""Synthetic raw-data tests; no EOS access or production data conversion."""

from contextlib import redirect_stdout
import io
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import awkward as ak
import numpy as np
import torch

from raw_event_conversion import (
    EventFeature, charged_pf_bin_features, charged_pf_impact_features, convert_raw_parquet_to_pt,
    cross_type_angular_features, default_features, leading_pair_angular_features,
    met_angular_features, pt_ratio_features, pt_sum_feature,
    ranked_abs_dxy_features, ranked_abs_eta_features, ranked_pt_feature,
)
from raw_event_conversion.__main__ import main
from raw_event_conversion.features import build_event_tensor
from raw_event_conversion.objects import OBJECT_COLUMNS, RAW_OBJECT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def raw_fixture():
    """Empty, exactly-on-threshold, and >slot-cap objects with empty/padded PF."""
    records = ak.Array({
        "L1T_Electron_PT": [[], [0, 10, 11, 20, 21, 30, 31, 50, 51], [60] * 7],
        "L1T_MuonTight_PT": [[], [0, 5, 6, 10, 11, 20, 21, 40, 41], [60] * 8],
        "L1T_PhotonTight_PT": [[], [0, 10, 11, 20, 21, 30, 31, 50, 51], [60] * 9],
        "L1T_JetPuppiAK4_PT": [[], [0, 20, 21, 30, 31, 50, 51, 100, 101], [120] * 15],
        "L1T_PUPPIMET_MET": [[0.], [50.], [100.]],
        "L1T_PUPPIMET_Phi": [[0.], [1.], [-1.]],
        "L1T_MET_MET": [[0.], [50.], [100.]],
        "L1T_MET_Phi": [[0.], [1.], [-1.]],
        "L1T_ScalarHT_HT": [[0.], [150.], [200.]],
        "L1T_Rho_Rho": [[0.] * 5, [1.] * 5, [2.] * 5],
        "L1T_PFPart_PT": [[], [0.] * 1000, [0.] * 1000],
        "L1T_PFPart_Charge": [[], [0] * 1000, [0] * 1000],
        "L1T_PFPart_D0": [[], [0.] * 1000, [0.] * 1000],
        "L1T_PFPart_ErrorD0": [[], [0.] * 1000, [0.] * 1000],
    })
    for column in list(records.fields):
        if column.endswith("_PT") and not column.startswith("L1T_PFPart_"):
            records = ak.with_field(records, ak.zeros_like(records[column]), column[:-3] + "_Eta")
            records = ak.with_field(records, ak.zeros_like(records[column]), column[:-3] + "_Phi")
            if column in ("L1T_Electron_PT", "L1T_MuonTight_PT"):
                records = ak.with_field(records, ak.zeros_like(records[column]), column[:-3] + "_D0")
    for column in RAW_OBJECT_COLUMNS:
        if column not in records.fields:
            pt = records[column.rsplit("_", 1)[0] + "_PT"]
            records = ak.with_field(records, ak.zeros_like(pt), column)
    return records


def with_pt(records, entries, column):
    """Replace test objects and keep their angular collections aligned."""
    records = ak.with_field(records, entries, column)
    if column in ("L1T_Electron_PT", "L1T_MuonTight_PT"):
        records = ak.with_field(records, ak.zeros_like(entries), column[:-3] + "_D0")
    records = ak.with_field(records, ak.zeros_like(entries), column[:-3] + "_Eta")
    records = ak.with_field(records, ak.zeros_like(entries), column[:-3] + "_Phi")
    for field in RAW_OBJECT_COLUMNS:
        if field.startswith(column.removesuffix("_PT") + "_") and field != column:
            records = ak.with_field(records, ak.zeros_like(entries), field)
    return records


class RawEventConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="raw_event_conversion_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "raw.parquet"
        self.output = self.root / "events.pt"
        ak.to_parquet(raw_fixture(), self.source)

    def load(self):
        return torch.load(self.output, map_location="cpu", weights_only=True)

    def test_lepton_dxy_appends_four_values_and_validity_flags_with_mm_schema(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        names = [f"{rank}_abs_dxy_mm_{kind}" for kind in ("electrons", "muons")
                 for rank in ("leading", "subleading")]
        self.assertEqual(data["event_feature_columns"][185:],
                         [entry for name in names for entry in (name, name + "_valid")])
        np.testing.assert_array_equal(data["event_features"][:, 185:].numpy(),
                                      [[0] * 8, [0, 1] * 4, [0, 1] * 4])
        for spec in data["event_feature_schema"][185::2]:
            self.assertEqual(spec["units"], "mm")
            self.assertEqual(spec["kind"], "continuous")
            self.assertEqual(spec["definition"]["raw_d0_units"], "mm")
            self.assertIn("not reconstructed PV", spec["definition"]["dxy_reference"])
        self.assertEqual(data["conversion_metadata"]["lepton_d0_columns"],
                         {"electrons": "L1T_Electron_D0", "muons": "L1T_MuonTight_D0"})
        self.assertEqual(data["conversion_metadata"]["lepton_d0_units"], "mm")
        torch.testing.assert_close(data["event_features"][:, :185],
                                   build_event_tensor(raw_fixture(), default_features()[:185]))

    def test_lepton_dxy_follows_pt_order_stable_ties_and_keeps_zero_valid(self):
        pt = [[10, 50, 50], [20], [30, 10], [0, -1, np.nan, None, np.inf, 5], [1, 2]]
        d0 = [[-10, -.25, 1.5], [0], [None, .5], [0, 0, 0, 0, 0, -4], [np.inf, np.nan]]
        for kind in ("electrons", "muons"):
            features = ranked_abs_dxy_features(kind)
            records = ak.Array({features[0].columns[0]: pt, features[0].columns[1]: d0})
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(build_event_tensor(records, features).numpy(),
                                              [[.25, 1, 1.5, 1], [0, 1, 0, 0], [0, 0, .5, 1],
                                               [4, 1, 0, 0], [0, 0, 0, 0]])

    def test_lepton_dxy_does_not_promote_lower_pt_lepton_when_top_two_are_invalid(self):
        for kind in ("electrons", "muons"):
            features = ranked_abs_dxy_features(kind)
            records = ak.Array({features[0].columns[0]: [[100, 50, 20]],
                                features[0].columns[1]: [[np.nan, np.inf, -3]]})
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(build_event_tensor(records, features).numpy(), np.zeros((1, 4)))

    def test_lepton_dxy_handles_empty_missing_and_invalid_pt_collections(self):
        for kind in ("electrons", "muons"):
            features = ranked_abs_dxy_features(kind)
            for entries in ([[], []], [[None], [None]], [None, None], [[0, -1, np.nan, np.inf]] * 2):
                records = ak.Array({column: entries for column in features[0].columns})
                with self.subTest(kind=kind, entries=str(entries)[:30]):
                    np.testing.assert_array_equal(build_event_tensor(records, features).numpy(), np.zeros((2, 4)))

    def test_lepton_dxy_preserves_float32_source_values_without_rescaling_or_sigma(self):
        for kind in ("electrons", "muons"):
            features = ranked_abs_dxy_features(kind)
            d0 = ak.values_astype(ak.Array([[-.000123, .234567]]), np.float32)
            records = ak.Array({features[0].columns[0]: [[50, 10]], features[0].columns[1]: d0})
            result = build_event_tensor(records, features)
            self.assertEqual(result.dtype, torch.float64)
            np.testing.assert_array_equal(result[0, ::2].numpy(), np.abs(ak.to_numpy(d0)[0]).astype(np.float64))

    def test_lepton_dxy_rejects_unsupported_types_misalignment_and_missing_columns(self):
        for kind in ("photons", "jets", "taus"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ranked_abs_dxy_features(kind)
        for kind in ("electrons", "muons"):
            features = ranked_abs_dxy_features(kind)
            records = ak.Array({features[0].columns[0]: [[50, 10]], features[0].columns[1]: [[1]]})
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "matching object counts"):
                build_event_tensor(records, features)
            ak.to_parquet(ak.without_field(raw_fixture(), features[0].columns[1]), self.source)
            with self.subTest(missing=kind), self.assertRaises((ValueError, KeyError)):
                convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                          include_native_event_info=False, include_met_features=False)
        self.assertFalse(self.output.exists())

    def test_lepton_dxy_uses_positive_pt_and_is_independent_of_threshold_overrides_and_met(self):
        records = raw_fixture()
        for base in ("L1T_Electron", "L1T_MuonTight"):
            records = with_pt(records, ak.Array([[.5, 1], [], []]), base + "_PT")
            records = ak.with_field(records, ak.Array([[.2, -.1], [], []]), base + "_D0")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                  thresholds={"electrons": [10000], "muons": [10000]},
                                  include_met_features=False, include_native_event_info=False)
        np.testing.assert_array_equal(self.load()["event_features"][:, -8:].numpy(),
                                      [[.1, 1, .2, 1] * 2, [0] * 8, [0] * 8])
        self.assertEqual([feature.schema() for feature in default_features()[-8:]],
                         [feature.schema() for feature in default_features(include_met=False)[-8:]])

    def impact_records(self, d0, error, pt=None, charge=None):
        count = len(d0)
        return ak.Array({"L1T_PFPart_PT": [pt if pt is not None else [3.] * count],
                         "L1T_PFPart_Charge": [charge if charge is not None else [1] * count],
                         "L1T_PFPart_D0": [d0], "L1T_PFPart_ErrorD0": [error]})

    def test_impact_bins_use_mm_absolute_values_and_half_open_boundaries(self):
        d0 = [0, -.05, .1, -.499, .5, -.999, 1, -4.9, 5, -9.9, 10, -20]
        records = self.impact_records(d0, [1] * 12, pt=list(range(3, 15)))
        result = build_event_tensor(records, charged_pf_impact_features()).numpy()[0]
        np.testing.assert_allclose(result[:12], [2, 7, 2, 11, 2, 15, 2, 19, 2, 23, 2, 27])
        np.testing.assert_allclose(result[24:30], [7, 77, 5, 60, 3, 39])
        self.assertEqual(result[-2], 0)

    def test_impact_significance_bins_and_strict_cumulative_thresholds(self):
        values = [0, -1.9, 2, -2.9, 3, -4.9, 5, -9.9, 10, -19.9, 20, -30]
        records = self.impact_records(values, [1] * 12, pt=list(range(3, 15)))
        result = build_event_tensor(records, charged_pf_impact_features()).numpy()[0]
        np.testing.assert_allclose(result[12:24], [2, 7, 2, 11, 2, 15, 2, 19, 2, 23, 2, 27])
        np.testing.assert_allclose(result[30:36], [7, 77, 5, 60, 3, 39])
        self.assertEqual(result[-1], 0)

    def test_impact_significance_is_dimensionless_and_preserves_small_positive_errors(self):
        records = self.impact_records([-.375, .004, .000000000005], [.125, .001, .000000000001])
        result = build_event_tensor(records, charged_pf_impact_features()).numpy()[0]
        self.assertEqual(result[12:24:2].sum(), 3)
        # The tiny but positive uncertainty is not replaced by an epsilon.
        self.assertEqual(result[32], 0)  # Strict significance > 5: exactly 5 does not pass.
        self.assertEqual(result[30], 2)  # 4 and 5 pass > 3; exactly 3 does not.
        self.assertEqual(result[-1], 0)

    def test_impact_invalid_values_are_excluded_not_zero_filled_into_prompt_bins(self):
        records = self.impact_records([0, .1, np.nan, None, np.inf, .5, 1, 2],
                                      [1, 0, 1, 1, 1, -1, np.nan, .2])
        result = build_event_tensor(records, charged_pf_impact_features()).numpy()[0]
        self.assertEqual(result[:12:2].sum(), 5)
        self.assertEqual(result[1:12:2].sum(), 15)
        self.assertEqual(result[12:24:2].sum(), 2)
        self.assertEqual(result[12], 1)  # Genuine D0=0 with positive uncertainty is valid.
        np.testing.assert_array_equal(result[-2:], [3, 6])
        self.assertTrue(np.isfinite(result).all())

    def test_impact_zero_uncertainties_and_overflow_never_manufacture_significance(self):
        for d0, error in (([0, 1, 10], [0, 0, 0]), ([1, 1, 1], [np.inf, None, 1e-320])):
            records = self.impact_records(d0, error)
            with self.subTest(error=error):
                result = build_event_tensor(records, charged_pf_impact_features()).numpy()[0]
                np.testing.assert_array_equal(result[12:24], np.zeros(12))
                np.testing.assert_array_equal(result[30:36], np.zeros(6))
                self.assertEqual(result[-1], 3)
                self.assertTrue(np.isfinite(result).all())

    def test_impact_candidates_require_finite_pt_above_two_and_finite_nonzero_charge(self):
        records = self.impact_records([1] * 9, [1] * 9,
                                      pt=[0, -1, 2, None, np.nan, np.inf, 3, 3, 4],
                                      charge=[1] * 6 + [0, np.nan, -1])
        result = build_event_tensor(records, charged_pf_impact_features()).numpy()[0]
        self.assertEqual(result[:12:2].sum(), 1)
        self.assertEqual(result[1:12:2].sum(), 4)
        self.assertEqual(result[12:24:2].sum(), 1)
        self.assertEqual(result[-2:].sum(), 0)

    def test_impact_empty_missing_and_zero_padded_collections_are_zero(self):
        for entries in ([[], []], [[None], [None]], [None, None], [[0] * 1000, [0] * 1000]):
            records = ak.Array({column: entries for column in
                                ("L1T_PFPart_PT", "L1T_PFPart_Charge", "L1T_PFPart_D0", "L1T_PFPart_ErrorD0")})
            with self.subTest(entries_type=str(entries)[:30]):
                np.testing.assert_array_equal(build_event_tensor(records, charged_pf_impact_features()).numpy(),
                                              np.zeros((2, 38)))

    def test_impact_schema_names_units_and_original_column_order_are_preserved(self):
        features = charged_pf_impact_features()
        self.assertEqual(len(features), 38)
        self.assertEqual(features[0].name, "n_charged_pf_pt_gt2_abs_dxy_mm_ge0_lt0p1")
        self.assertEqual(features[10].name, "n_charged_pf_pt_gt2_abs_dxy_mm_ge10")
        self.assertEqual(features[12].name, "n_charged_pf_pt_gt2_abs_dxysig_ge0_lt2")
        self.assertEqual(features[24].name, "n_charged_pf_pt_gt2_abs_dxy_mm_gt0p5")
        self.assertEqual(features[30].name, "n_charged_pf_pt_gt2_abs_dxysig_gt3")
        for feature in features[:36]:
            definition = feature.schema()["definition"]
            self.assertEqual(definition["raw_d0_units"], "mm")
            self.assertEqual(definition["pt_min_GeV"], 2)
            self.assertEqual(definition["pt_weighting"], "none")
            self.assertIsNone(definition["vertex_selection"])
        self.assertEqual(features[0].schema()["definition"]["range_units"], "mm")
        self.assertEqual(features[12].schema()["definition"]["range_units"], "1")
        self.assertIsNone(features[12].schema()["definition"]["denominator_regularization"])
        json.dumps([feature.schema() for feature in features], allow_nan=False)
        defaults = default_features()
        torch.testing.assert_close(build_event_tensor(raw_fixture(), defaults)[:, :147],
                                   build_event_tensor(raw_fixture(), defaults[:147]))
        self.assertEqual([feature.schema() for feature in defaults[-46:-8]],
                         [feature.schema() for feature in default_features(include_met=False)[-46:-8]])

    def test_impact_required_fields_must_exist_and_align_even_without_met(self):
        records = self.impact_records([1, 2], [1, 1])
        for column in ("L1T_PFPart_D0", "L1T_PFPart_ErrorD0"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "matching object counts"):
                build_event_tensor(ak.with_field(records, ak.Array([[1]]), column), charged_pf_impact_features())
            ak.to_parquet(ak.without_field(raw_fixture(), column), self.source)
            with self.subTest(missing=column), self.assertRaises((ValueError, KeyError)):
                convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                          include_met_features=False, include_native_event_info=False)
        self.assertFalse(self.output.exists())

    def test_impact_conversion_warns_and_saves_invalid_measurement_counts(self):
        records = raw_fixture()
        for column, entries in {"L1T_PFPart_PT": [[3, 4], [5], []],
                                "L1T_PFPart_Charge": [[1, -1], [1], []],
                                "L1T_PFPart_D0": [[np.nan, 1], [0], []],
                                "L1T_PFPart_ErrorD0": [[1, 0], [0], []]}.items():
            records = ak.with_field(records, ak.Array(entries), column)
        ak.to_parquet(records, self.source)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        messages = [str(warning.message) for warning in caught]
        self.assertTrue(any("1 of 3" in message and "abs_dxy_mm" in message for message in messages))
        self.assertTrue(any("3 of 3" in message and "abs_dxysig" in message for message in messages))
        data = self.load()
        np.testing.assert_array_equal(data["event_features"][:, 183:185].numpy(), [[1, 2], [0, 1], [0, 0]])
        np.testing.assert_array_equal(data["event_features"][:, 159:171].numpy(), np.zeros((3, 12)))
        self.assertEqual(data["conversion_metadata"]["pf_impact_parameter_units"], "mm")

    def test_charged_pf_features_append_names_schema_and_strict_window_values(self):
        records = ak.with_field(raw_fixture(), ak.Array([[], [0, 1, 2, 2.5, 3, 5, 7.5, 10, 11, 20], [0.]]),
                                "L1T_PFPart_PT")
        records = ak.with_field(records, ak.Array([[], [1, 1, 1, 1, 0, -1, 1, -1, 1, 0], [1]]),
                                "L1T_PFPart_Charge")
        records = ak.with_field(records, ak.zeros_like(records["L1T_PFPart_PT"]), "L1T_PFPart_D0")
        records = ak.with_field(records, ak.ones_like(records["L1T_PFPart_PT"]), "L1T_PFPart_ErrorD0")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        suffixes = ("pt_gt2", "pt_gt2_lt5", "pt_gt2_lt10", "pt_gt10")
        self.assertEqual(data["event_feature_columns"][139:147],
                         [f"{prefix}_charged_pf_{suffix}" for suffix in suffixes for prefix in ("n", "sum_pt")])
        np.testing.assert_allclose(data["event_features"][:, 139:147].numpy(),
                                   [[0] * 8, [5, 36, 1, 2.5, 3, 15, 1, 11], [0] * 8])
        for spec in data["event_feature_schema"][139:147]:
            self.assertEqual(spec["required_columns"], ["L1T_PFPart_PT", "L1T_PFPart_Charge"])
            self.assertEqual(spec["definition"]["pt_weighting"], "none")
            self.assertIsNone(spec["definition"]["vertex_selection"])
        self.assertEqual([spec["kind"] for spec in data["event_feature_schema"][139:147]],
                         ["ordinal", "continuous"] * 4)
        self.assertEqual([spec["units"] for spec in data["event_feature_schema"][139:147]],
                         ["count", "GeV"] * 4)

    def test_charged_pf_ignores_missing_nonfinite_neutral_and_nonpositive_candidates(self):
        records = ak.Array({
            "L1T_PFPart_PT": [[None, np.nan, np.inf, -np.inf, -1, 0, 2, 3, 4, 5, 6, 7, 8, 9]],
            "L1T_PFPart_Charge": [[1] * 7 + [None, np.nan, np.inf, -np.inf, 0, 1, -1]],
        })
        features = tuple(feature for bounds in ((2, None), (2, 5), (2, 10), (10, None))
                         for feature in charged_pf_bin_features(*bounds))
        np.testing.assert_allclose(build_event_tensor(records, features).numpy(), [[2, 17, 0, 0, 2, 17, 0, 0]])

    def test_charged_pf_empty_missing_and_zero_padded_collections_give_zero(self):
        for entries in ([[], []], [[None], [None]], [None, None], [[0] * 1000, [0] * 1000]):
            records = ak.Array({"L1T_PFPart_PT": entries, "L1T_PFPart_Charge": entries})
            with self.subTest(entries_type=str(entries)[:30]):
                np.testing.assert_array_equal(build_event_tensor(records, charged_pf_bin_features(2, 5)).numpy(),
                                              np.zeros((2, 2)))

    def test_charged_pf_sums_promote_float16_and_do_not_apply_puppi_or_truth_fields(self):
        records = ak.Array({
            "L1T_PFPart_PT": ak.values_astype(ak.Array([[3.125, 4.5], [3.] * 1200]), np.float16),
            "L1T_PFPart_Charge": [[1, -1], [1] * 1200],
            "L1T_PFPart_PuppiW": [[0, 0], [0] * 1200],
            "L1T_PFPart_IsPU": [[1, 1], [1] * 1200],
            "L1T_PUPPIPart_PT": [[9999], [9999]],
        })
        result = build_event_tensor(records, charged_pf_bin_features(2, 5))
        self.assertEqual(result.dtype, torch.float64)
        np.testing.assert_array_equal(result.numpy(), [[2, 7.625], [1200, 3600]])

    def test_charged_pf_bounds_are_validated_and_factory_supports_new_windows(self):
        for bounds in ((True, None), ("2", None), (-1, None), (np.nan, None), (np.inf, None),
                       (2, True), (2, "5"), (2, np.nan), (2, np.inf), (2, 2), (2, 1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                charged_pf_bin_features(*bounds)
        records = ak.Array({"L1T_PFPart_PT": [[2.25, 2.5, 2.75, 3]], "L1T_PFPart_Charge": [[1] * 4]})
        features = charged_pf_bin_features(2.25, 3)
        self.assertEqual(features[0].name, "n_charged_pf_pt_gt2p25_lt3")
        np.testing.assert_allclose(build_event_tensor(records, features).numpy(), [[2, 5.25]])

    def test_charged_pf_rejects_misaligned_or_missing_required_columns_without_saving(self):
        records = ak.Array({"L1T_PFPart_PT": [[3, 4]], "L1T_PFPart_Charge": [[1]]})
        with self.assertRaisesRegex(ValueError, "matching object counts"):
            build_event_tensor(records, charged_pf_bin_features(2))
        for column in ("L1T_PFPart_PT", "L1T_PFPart_Charge"):
            ak.to_parquet(ak.without_field(raw_fixture(), column), self.source)
            with self.subTest(column=column), self.assertRaises((ValueError, KeyError)):
                convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertFalse(self.output.exists())

    def test_charged_pf_features_remain_when_met_disabled_and_counts_are_overridden(self):
        columns = ("L1T_PFPart_PT", "L1T_PFPart_Charge")
        normal = [feature.schema() for feature in default_features() if feature.columns == columns]
        disabled = [feature.schema() for feature in default_features(
            {kind: [] for kind in ("electrons", "muons", "photons", "jets")}, include_met=False)
                    if feature.columns == columns]
        self.assertEqual(normal, disabled)

    def test_all_default_jet_and_met_dependencies_are_puppi(self):
        features = default_features()
        columns = {column for feature in features for column in feature.columns}
        self.assertFalse(any(column.startswith("L1T_JetAK4_") or column.startswith("L1T_MET_")
                             for column in columns))
        self.assertTrue({"L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi",
                         "L1T_PUPPIMET_MET", "L1T_PUPPIMET_Phi"} <= columns)
        self.assertEqual(len(features), 193)
        self.assertEqual(features[25].schema()["definition"]["raw_column"], "L1T_PUPPIMET_MET")

    def test_puppi_sources_drive_jet_met_and_combined_features_not_original_collections(self):
        records = with_pt(raw_fixture()[1:2], ak.Array([[50., 100., 30.]]), "L1T_JetPuppiAK4_PT")
        records = ak.with_field(records, ak.Array([[-1., 2., 4.]]), "L1T_JetPuppiAK4_Eta")
        records = ak.with_field(records, ak.Array([[np.pi - .1, -np.pi + .1, 2.]]), "L1T_JetPuppiAK4_Phi")
        records = ak.with_field(records, ak.Array([[10.]]), "L1T_PUPPIMET_MET")
        records = ak.with_field(records, ak.Array([[np.pi - .2]]), "L1T_PUPPIMET_Phi")
        records = with_pt(records, ak.Array([[9999.]]), "L1T_JetAK4_PT")
        records = ak.with_field(records, ak.Array([[999.]]), "L1T_MET_MET")
        records = ak.with_field(records, ak.Array([[-1.]]), "L1T_MET_Phi")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        values = dict(zip(data["event_feature_columns"], data["event_features"][0].tolist()))
        expected = {"n_jets": 3, "n_jets_pt20": 3, "n_jets_pt30": 2, "n_jets_pt50": 1,
                    "n_jets_pt100": 0, "sum_pt_jets": 180, "sum_pt_all_objects": 782,
                    "met": 10, "met_valid": 1, "sum_pt_over_met": 78.2, "sum_pt_over_met_valid": 1,
                    "leading_pt_jets": 100, "subleading_pt_jets": 50,
                    "leading_pt_fraction_jets": 100 / 180, "pt_balance_jets": .5,
                    "leading_abs_eta_jets": 2, "subleading_abs_eta_jets": 1,
                    "abs_delta_eta_jets": 3, "abs_delta_phi_jets": .2,
                    "delta_r_jets": np.hypot(3, .2), "delta_phi_leading_jet_met": .3,
                    "min_delta_phi_jets_met": .1}
        for kind, singular in (("electrons", "electron"), ("muons", "muon"), ("photons", "photon")):
            expected.update({f"abs_delta_eta_leading_{kind}_jets": 2,
                             f"abs_delta_phi_leading_{kind}_jets": np.pi - .1,
                             f"delta_r_leading_{kind}_jets": np.hypot(2, np.pi - .1),
                             f"delta_phi_leading_{singular}_met": np.pi - .2})
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(values[name], value, places=12)
        self.assertEqual(data["native_event_info"][0, 0].item(), 999)
        self.assertEqual(data["native_event_info"][0, 2].item(), 10)
        # Alter only the original collections: every model feature must be identical.
        changed = with_pt(records, ak.Array([[1., 2., 3., 4.]]), "L1T_JetAK4_PT")
        changed = ak.with_field(changed, ak.Array([[1.]]), "L1T_MET_MET")
        changed = ak.with_field(changed, ak.Array([[2.]]), "L1T_MET_Phi")
        torch.testing.assert_close(build_event_tensor(changed, default_features()), data["event_features"])

    def test_puppi_model_inputs_work_without_original_jets_or_met(self):
        records = raw_fixture()
        for column in ("L1T_MET_MET", "L1T_MET_Phi"):
            records = ak.without_field(records, column)
        ak.to_parquet(records, self.source)
        original_reader = ak.from_parquet
        with patch("raw_event_conversion.convert.ak.from_parquet", wraps=original_reader) as reader:
            convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False)
        columns = reader.call_args.kwargs["columns"]
        self.assertFalse(any(column.startswith("L1T_JetAK4_") or column.startswith("L1T_MET_")
                             for column in columns))
        data = self.load()
        self.assertEqual(tuple(data["event_features"].shape), (3, 193))
        metadata = data["conversion_metadata"]
        self.assertEqual(metadata["schema_version"], 7)
        self.assertEqual(metadata["pf_impact_parameter_units"], "mm")
        self.assertEqual(metadata["object_pt_columns"]["jets"], "L1T_JetPuppiAK4_PT")
        self.assertEqual(metadata["met_columns"], ["L1T_PUPPIMET_MET", "L1T_PUPPIMET_Phi"])

    def test_missing_puppi_sources_never_fall_back_to_original_collections(self):
        records = with_pt(raw_fixture(), ak.Array([[100.]] * 3), "L1T_JetAK4_PT")
        for column in ("L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi",
                       "L1T_PUPPIMET_MET", "L1T_PUPPIMET_Phi"):
            ak.to_parquet(ak.without_field(records, column), self.source)
            with self.subTest(column=column), self.assertRaises((ValueError, KeyError)):
                convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False)
        self.assertFalse(self.output.exists())

    def test_mixer_and_event_runners_reject_old_source_definitions(self):
        from casual_tests._event_only_reporting import _load_pair, reporting
        from scripts.make_reference_anomalous import make_datasets
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        puppi = self.load()
        original = self.load()
        original["event_feature_schema"][3]["required_columns"] = ["L1T_JetAK4_PT"]
        original["event_feature_schema"][3]["definition"]["raw_column"] = "L1T_JetAK4_PT"
        with self.assertRaisesRegex(ValueError, "event_feature_schema"):
            make_datasets(puppi, original, reference_count=1, background_count=1, signal_count=1)
        # Source definitions differ even though names, kinds, shapes and values match.
        original_path = self.root / "original.pt"
        torch.save(original, original_path)
        parser = reporting._parser("test", model="EDT", require_event_features=True)
        args = parser.parse_args(["--signal", str(self.output), "--background", str(original_path)])
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            _load_pair(parser, args, "EDT")

    def test_zero_object_events_and_full_counts_are_preserved_with_empty_pf(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        self.assertEqual(tuple(data["event_features"].shape), (3, 193))
        self.assertEqual(data["event_features"].dtype, torch.float64)
        self.assertEqual(data["event_features"][0].tolist(), [0] * 26 + [1, 0, 0] + [0] * 164)
        self.assertEqual(data["event_features"][1, :20].tolist(), [8] * 4 + [7, 5, 3, 1] * 4)
        self.assertEqual(data["event_features"][2, :4].tolist(), [7, 8, 9, 15])
        self.assertEqual(data["label"].tolist(), [1, 1, 1])
        self.assertEqual(tuple(data["obj"].shape), (3, 22, 20))
        self.assertEqual(tuple(data["pf"].shape), (3, 1, 1))
        self.assertFalse(bool(data["obj"][0].any()))
        self.assertEqual(data["obj_columns"], list(OBJECT_COLUMNS))
        self.assertEqual((data["obj"][2, :, 0] > 0).sum().item(), 22)
        self.assertFalse(bool(data["pf"].any()))

    def test_pt_sums_use_every_object_not_native_ht(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        self.assertEqual(data["event_feature_columns"][20:24],
                         ["sum_pt_electrons", "sum_pt_muons", "sum_pt_photons", "sum_pt_jets"])
        self.assertEqual(data["event_features"][:, 20:24].tolist(),
                         [[0, 0, 0, 0], [224, 154, 224, 404], [420, 480, 540, 1800]])
        self.assertEqual(data["native_event_info"][:, 4].tolist(), [0, 150, 200])
        for spec in data["event_feature_schema"][20:24]:
            self.assertEqual(spec["kind"], "continuous")
            self.assertEqual(spec["units"], "GeV")
            self.assertEqual(spec["definition"]["aggregation"], "scalar_sum")
            self.assertEqual(len(spec["required_columns"]), 1)
            self.assertNotIn("L1T_ScalarHT_HT", spec["required_columns"])
        self.assertTrue(all(spec["kind"] == "ordinal" for spec in data["event_feature_schema"][:20]))

    def test_total_activity_met_and_ratio_are_model_features(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        self.assertEqual(data["event_feature_columns"][24:29],
                         ["sum_pt_all_objects", "met", "met_valid", "sum_pt_over_met", "sum_pt_over_met_valid"])
        np.testing.assert_allclose(data["event_features"][:, 24:29].numpy(),
                                   [[0, 0, 1, 0, 0], [1006, 50, 1, 20.12, 1], [3240, 100, 1, 32.4, 1]])
        torch.testing.assert_close(data["event_features"][:, 24], data["event_features"][:, 20:24].sum(dim=1))
        self.assertTrue(all("L1T_ScalarHT_HT" not in spec["required_columns"]
                            for spec in data["event_feature_schema"]))
        self.assertEqual(data["event_feature_schema"][26]["kind"], "boolean")
        self.assertEqual(data["event_feature_schema"][28]["kind"], "boolean")

    def test_leading_and_subleading_pt_are_appended_for_all_object_types(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        expected_names = [f"{prefix}_pt_{kind}" for kind in ("electrons", "muons", "photons", "jets")
                          for prefix in ("leading", "subleading")]
        self.assertEqual(data["event_feature_columns"][29:37], expected_names)
        self.assertEqual(data["event_features"][:, 29:37].tolist(),
                         [[0] * 8, [51, 50, 41, 40, 51, 50, 101, 100],
                          [60, 60, 60, 60, 60, 60, 120, 120]])
        for spec in data["event_feature_schema"][29:37]:
            self.assertEqual(spec["kind"], "continuous")
            self.assertEqual(spec["units"], "GeV")
            self.assertEqual(spec["definition"]["order"], "descending")
            self.assertFalse(spec["definition"]["object_truncation"])

    def test_ranked_pt_handles_unsorted_ties_invalid_entries_singletons_and_full_collections(self):
        records = ak.concatenate([raw_fixture()] * 2)
        entries = ak.Array([[], [7.125], [30., 60.00000001, 20., 60.],
                            [None, np.nan, np.inf, -np.inf, -1., 0., 8., 3.],
                            [None, np.nan, -1., 0.], list(range(1, 101))])
        for kind in ("electrons", "muons", "photons", "jets"):
            lead, sublead = ranked_pt_feature(kind, 1), ranked_pt_feature(kind, 2)
            records = with_pt(records, entries, lead.columns[0])
            np.testing.assert_array_equal(lead.compute(records), [0., 7.125, 60.00000001, 8., 0., 100.])
            np.testing.assert_array_equal(sublead.compute(records), [0., 0., 60., 3., 0., 99.])
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                  thresholds={kind: [1000] for kind in ("electrons", "muons", "photons", "jets")})
        data = self.load()
        index = data["event_feature_columns"].index("subleading_pt_photons")
        self.assertEqual(data["event_features"][:, index].tolist(), [0., 0., 60., 3., 0., 99.])
        self.assertTrue(bool(torch.isfinite(data["event_features"]).all()))

    def test_ranked_pt_rejects_invalid_type_or_rank(self):
        with self.assertRaises(ValueError):
            ranked_pt_feature("taus")
        for rank in (0, -1, 3, True, 1.0, "1", None):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                ranked_pt_feature("jets", rank)

    def test_ranked_abs_eta_is_appended_for_all_types_with_validity_and_metadata(self):
        records = raw_fixture()
        for column in ("L1T_Electron_Eta", "L1T_MuonTight_Eta", "L1T_PhotonTight_Eta", "L1T_JetPuppiAK4_Eta"):
            records = ak.with_field(records, -ak.local_index(records[column], axis=1) / 10., column)
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        expected_names = [name for kind in ("electrons", "muons", "photons", "jets")
                          for prefix in ("leading", "subleading")
                          for name in (f"{prefix}_abs_eta_{kind}", f"{prefix}_abs_eta_{kind}_valid")]
        self.assertEqual(data["event_feature_columns"][53:69], expected_names)
        np.testing.assert_allclose(data["event_features"][:, 53:69].numpy(),
                                   [[0] * 16, [.8, 1, .7, 1] * 4, [0, 1, .1, 1] * 4])
        self.assertEqual([spec["kind"] for spec in data["event_feature_schema"][53:69]],
                         ["continuous", "boolean"] * 8)
        for spec in data["event_feature_schema"][53:69:2]:
            self.assertEqual(spec["units"], "1")
            self.assertEqual(spec["definition"]["tie_break"], "producer order")
            self.assertEqual(len(spec["required_columns"]), 2)

    def test_ranked_abs_eta_follows_pt_order_and_preserves_ties_and_invalid_eta(self):
        pt = ak.Array([[], [7.125], [30., 60., 20., 60.],
                       [None, np.nan, np.inf, -np.inf, -1., 0., 8., 3.],
                       [90., 80., 70.], list(range(1, 101))])
        eta = ak.Array([[], [-1.125], [.5, -2.5, 4., 1.25], [100.] * 6 + [0., -.75],
                        [None, np.inf, -1.], [-i / 100. for i in range(1, 101)]])
        for kind in ("electrons", "muons", "photons", "jets"):
            features = ranked_abs_eta_features(kind)
            records = ak.Array({features[0].columns[0]: pt, features[0].columns[1]: eta})
            result = build_event_tensor(records, features)
            np.testing.assert_allclose(result.numpy(),
                                       [[0, 0, 0, 0], [1.125, 1, 0, 0], [2.5, 1, 1.25, 1],
                                        [0, 1, .75, 1], [0, 0, 0, 0], [1, 1, .99, 1]])

    def test_invalid_leading_eta_does_not_promote_subleading_object(self):
        features = ranked_abs_eta_features("photons")
        records = ak.Array({"L1T_PhotonTight_PT": [[90., 80., 70.]] * 3,
                            "L1T_PhotonTight_Eta": [[np.nan, -2., 1.], [-np.inf, -2., 1.], [None, -2., 1.]]})
        np.testing.assert_array_equal(build_event_tensor(records, features).numpy(), [[0, 0, 2, 1]] * 3)

    def test_ranked_abs_eta_handles_entirely_empty_and_missing_collections(self):
        features = ranked_abs_eta_features("jets")
        for entries in ([[], [], []], [[None], [None], [None]], [None, None, None]):
            with self.subTest(entries=entries):
                records = ak.Array({"L1T_JetPuppiAK4_PT": entries, "L1T_JetPuppiAK4_Eta": entries})
                np.testing.assert_array_equal(build_event_tensor(records, features).numpy(), np.zeros((3, 4)))

    def test_ranked_abs_eta_rejects_misaligned_collections_and_unknown_type(self):
        features = ranked_abs_eta_features("jets")
        records = ak.Array({"L1T_JetPuppiAK4_PT": [[20., 10.], [30.]], "L1T_JetPuppiAK4_Eta": [[1.], [2., 3.]]})
        with self.assertRaisesRegex(ValueError, "matching object counts"):
            build_event_tensor(records, features)
        with self.assertRaises(ValueError):
            ranked_abs_eta_features("taus")

    def test_missing_eta_column_is_not_silently_filled(self):
        ak.to_parquet(ak.without_field(raw_fixture(), "L1T_PhotonTight_Eta"), self.source)
        with self.assertRaises((ValueError, KeyError)):
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertFalse(self.output.exists())

    def test_ranked_abs_eta_is_independent_of_multiplicity_thresholds(self):
        records = with_pt(raw_fixture(), ak.Array([[10., 5.], [], [20.]]), "L1T_PhotonTight_PT")
        records = ak.with_field(records, ak.Array([[-1., 2.], [], [-3.]]), "L1T_PhotonTight_Eta")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=5, thresholds={"photons": [1000]})
        data = self.load()
        index = data["event_feature_columns"].index("leading_abs_eta_photons")
        np.testing.assert_array_equal(data["event_features"][:, index:index + 4].numpy(),
                                      [[1, 1, 2, 1], [0, 0, 0, 0], [3, 1, 0, 0]])

    def test_pair_angular_features_are_appended_for_all_types_with_schema(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        expected_names = [name for kind in ("electrons", "muons", "photons", "jets")
                          for prefix in ("abs_delta_eta", "abs_delta_phi", "delta_r")
                          for name in (f"{prefix}_{kind}", f"{prefix}_{kind}_valid")]
        self.assertEqual(data["event_feature_columns"][69:93], expected_names)
        np.testing.assert_array_equal(data["event_features"][:, 69:93].numpy(),
                                      [[0] * 24, [0, 1] * 12, [0, 1] * 12])
        specs = data["event_feature_schema"][69:93]
        self.assertEqual([spec["kind"] for spec in specs], ["continuous", "boolean"] * 12)
        self.assertEqual([spec["units"] for spec in specs[::2]], ["1", "rad", "1"] * 4)
        self.assertEqual([len(spec["required_columns"]) for spec in specs[::2]], [2, 2, 3] * 4)
        for spec in specs[::2]:
            self.assertEqual(spec["definition"]["pair"], "PT-leading and PT-subleading objects")
            self.assertEqual(spec["definition"]["tie_break"], "producer order")

    def test_pair_angles_use_pt_ranks_signed_eta_and_stable_ties_for_all_types(self):
        for kind in ("electrons", "muons", "photons", "jets"):
            features = leading_pair_angular_features(kind)
            pt_column, eta_column, phi_column = features[4].columns
            records = ak.Array({
                pt_column: [[20., 60., 30., 50.], [30., 60., 60., 50.],
                            [None, np.nan, np.inf, -1., 0., 20., 10.]],
                eta_column: [[40., -2., 10., 1.], [.3, -2., 1., 50.], [100.] * 5 + [-1., 2.]],
                phi_column: [[1., np.pi - .1, 2., -np.pi + .1],
                             [.4, np.pi - .1, -np.pi + .1, 1.], [0.] * 6 + [np.pi]],
            })
            result = build_event_tensor(records, features)
            np.testing.assert_allclose(result.numpy(),
                                       [[3, 1, .2, 1, np.hypot(3, .2), 1]] * 2
                                       + [[3, 1, np.pi, 1, np.hypot(3, np.pi), 1]], atol=1e-14)

    def test_pair_phi_wraps_boundary_and_is_rotation_and_period_invariant(self):
        features = leading_pair_angular_features("photons")
        records = ak.Array({"L1T_PhotonTight_PT": [[20., 10.]] * 5,
                            "L1T_PhotonTight_Eta": [[0., 0.]] * 5,
                            "L1T_PhotonTight_Phi": [[np.pi - .1, -np.pi + .1], [.1, .3],
                                                   [1.1, 1.3], [.1 + 4*np.pi, .3 - 6*np.pi],
                                                   [np.pi, -np.pi]]})
        result = build_event_tensor(records, features).numpy()
        np.testing.assert_allclose(result[:, 2], [.2, .2, .2, .2, 0], atol=1e-14)
        np.testing.assert_allclose(result[:, 4], result[:, 2], atol=1e-14)
        self.assertTrue(np.all((result[:, 2] >= 0) & (result[:, 2] <= np.pi)))
        np.testing.assert_array_equal(result[:, 1::2], np.ones((5, 3)))

    def test_pair_validity_is_independent_for_eta_phi_and_delta_r_without_promotion(self):
        features = leading_pair_angular_features("muons")
        records = ak.Array({"L1T_MuonTight_PT": [[90., 80., 70.]] * 5,
                            "L1T_MuonTight_Eta": [[None, 1., 2.], [-2., 1., 2.],
                                                 [np.nan, 1., 2.], [-2., 1., 2.], [0., 0., 2.]],
                            "L1T_MuonTight_Phi": [[.2, .5, 1.], [np.inf, .5, 1.],
                                                 [None, .5, 1.], [np.nan, .5, 1.], [0., 0., 1.]]})
        np.testing.assert_allclose(build_event_tensor(records, features).numpy(),
                                   [[0, 0, .3, 1, 0, 0], [3, 1, 0, 0, 0, 0], [0] * 6,
                                    [3, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1]])

    def test_pair_angles_handle_missing_single_and_entirely_empty_collections(self):
        features = leading_pair_angular_features("jets")
        pt_column, eta_column, phi_column = features[4].columns
        for entries in ([[], [10.], [0., np.nan, 5.]], [[], [], []],
                        [[None], [None], [None]], [None, None, None]):
            with self.subTest(entries=entries):
                records = ak.Array({pt_column: entries, eta_column: entries, phi_column: entries})
                np.testing.assert_array_equal(build_event_tensor(records, features).numpy(), np.zeros((3, 6)))

    def test_pair_angles_reject_misaligned_columns_and_unknown_types(self):
        features = leading_pair_angular_features("jets")
        for column in ("L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"):
            records = ak.Array({"L1T_JetPuppiAK4_PT": [[20., 10.], [30.]],
                                "L1T_JetPuppiAK4_Eta": [[1., 2.], [3.]],
                                "L1T_JetPuppiAK4_Phi": [[.1, .2], [.3]]})
            records = ak.with_field(records, ak.Array([[1.], [2., 3.]]), column)
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "matching object counts"):
                build_event_tensor(records, features)
        with self.assertRaises(ValueError):
            leading_pair_angular_features("taus")

    def test_pair_angles_require_raw_phi_column(self):
        ak.to_parquet(ak.without_field(raw_fixture(), "L1T_PhotonTight_Phi"), self.source)
        with self.assertRaises((ValueError, KeyError)):
            convert_raw_parquet_to_pt(self.source, self.output, label=5)
        self.assertFalse(self.output.exists())

    def test_pair_angles_do_not_follow_multiplicity_thresholds(self):
        records = with_pt(raw_fixture(), ak.Array([[10., 5.], [], [20.]]), "L1T_PhotonTight_PT")
        records = ak.with_field(records, ak.Array([[-1., 1.], [], [0.]]), "L1T_PhotonTight_Eta")
        records = ak.with_field(records, ak.Array([[0., np.pi], [], [0.]]), "L1T_PhotonTight_Phi")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=5, thresholds={"photons": [1000]})
        data = self.load()
        index = data["event_feature_columns"].index("abs_delta_eta_photons")
        np.testing.assert_allclose(data["event_features"][:, index:index + 6].numpy(),
                                   [[2, 1, np.pi, 1, np.hypot(2, np.pi), 1], [0] * 6, [0] * 6])

    def test_pair_angles_keep_finite_outputs_when_eta_subtraction_overflows(self):
        features = leading_pair_angular_features("jets")
        records = ak.Array({"L1T_JetPuppiAK4_PT": [[20., 10.]],
                            "L1T_JetPuppiAK4_Eta": [[1e308, -1e308]], "L1T_JetPuppiAK4_Phi": [[0., 0.]]})
        np.testing.assert_array_equal(build_event_tensor(records, features).numpy(), [[0, 0, 0, 1, 0, 0]])

    def test_cross_type_angles_append_exactly_six_pairs_with_schema(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        pairs = list(combinations(("electrons", "muons", "photons", "jets"), 2))
        expected_names = [name for first, second in pairs
                          for prefix in ("abs_delta_eta", "abs_delta_phi", "delta_r")
                          for name in (f"{prefix}_leading_{first}_{second}",
                                       f"{prefix}_leading_{first}_{second}_valid")]
        self.assertEqual(len(pairs), 6)
        self.assertEqual(data["event_feature_columns"][93:129], expected_names)
        np.testing.assert_array_equal(data["event_features"][:, 93:129].numpy(),
                                      [[0] * 36, [0, 1] * 18, [0, 1] * 18])
        specs = data["event_feature_schema"][93:129]
        self.assertEqual([spec["kind"] for spec in specs], ["continuous", "boolean"] * 18)
        self.assertEqual([spec["units"] for spec in specs[::2]], ["1", "rad", "1"] * 6)
        self.assertEqual([len(spec["required_columns"]) for spec in specs[::2]], [4, 4, 6] * 6)
        for index, (first, second) in enumerate(pairs):
            for spec in specs[index * 6:index * 6 + 6:2]:
                self.assertEqual(spec["definition"]["object_types"], [first, second])
                self.assertEqual(spec["definition"]["ranks"], [1, 1])
                self.assertEqual(spec["definition"]["pair"], "PT-leading object of each type")

    def test_cross_type_angles_select_leading_objects_not_nearest_or_subleading(self):
        for first, second in combinations(("electrons", "muons", "photons", "jets"), 2):
            features = cross_type_angular_features(first, second)
            pt_first, pt_second, eta_first, eta_second, phi_first, phi_second = features[4].columns
            records = ak.Array({pt_first: [[20., 100., 100., 80.]], pt_second: [[30., 60., 50.]],
                                eta_first: [[100., -2., 1., 70.]], eta_second: [[50., 1., -2.]],
                                phi_first: [[1., np.pi - .1, -np.pi + .1, 1.]],
                                phi_second: [[2., -np.pi + .1, np.pi - .1]]})
            np.testing.assert_allclose(build_event_tensor(records, features).numpy(),
                                       [[3, 1, .2, 1, np.hypot(3, .2), 1]], atol=1e-14)

    def test_cross_type_singletons_are_valid_and_missing_either_type_is_invalid(self):
        features = cross_type_angular_features("electrons", "photons")
        pt_first, pt_second, eta_first, eta_second, phi_first, phi_second = features[4].columns
        records = ak.Array({pt_first: [[10.], [], [10.], [0., np.inf, None]],
                            pt_second: [[20.], [20.], [], [20.]],
                            eta_first: [[-1.], [], [-1.], [0., 0., 0.]],
                            eta_second: [[1.], [1.], [], [1.]],
                            phi_first: [[0.], [], [0.], [0., 0., 0.]],
                            phi_second: [[np.pi], [np.pi], [], [np.pi]]})
        np.testing.assert_allclose(build_event_tensor(records, features).numpy(),
                                   [[2, 1, np.pi, 1, np.hypot(2, np.pi), 1]] + [[0] * 6] * 3)

    def test_cross_type_invalid_coordinates_have_independent_validity_without_promotion(self):
        features = cross_type_angular_features("muons", "jets")
        pt_first, pt_second, eta_first, eta_second, phi_first, phi_second = features[4].columns
        records = ak.Array({pt_first: [[90., 80.]] * 4, pt_second: [[100., 50.]] * 4,
                            eta_first: [[None, 2.], [-2., 2.], [-2., 2.], [0., 2.]],
                            eta_second: [[1., 2.], [1., 2.], [np.inf, 2.], [0., 2.]],
                            phi_first: [[.2, 1.], [np.nan, 1.], [.2, 1.], [0., 1.]],
                            phi_second: [[.5, 1.], [.5, 1.], [None, 1.], [0., 1.]]})
        np.testing.assert_allclose(build_event_tensor(records, features).numpy(),
                                   [[0, 0, .3, 1, 0, 0], [3, 1, 0, 0, 0, 0], [0] * 6, [0, 1] * 3])

    def test_cross_type_reverse_arguments_have_identical_names_schema_and_values(self):
        for first, second in combinations(("electrons", "muons", "photons", "jets"), 2):
            forward = cross_type_angular_features(first, second)
            reverse = cross_type_angular_features(second, first)
            self.assertEqual([feature.schema() for feature in forward], [feature.schema() for feature in reverse])
            torch.testing.assert_close(build_event_tensor(raw_fixture(), forward),
                                       build_event_tensor(raw_fixture(), reverse))

    def test_cross_type_rejects_same_unknown_or_misaligned_types(self):
        for pair in (("jets", "jets"), ("taus", "jets"), ("muons", "taus")):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                cross_type_angular_features(*pair)
        features = cross_type_angular_features("electrons", "muons")
        for column in ("L1T_Electron_Eta", "L1T_Electron_Phi", "L1T_MuonTight_Eta", "L1T_MuonTight_Phi"):
            records = ak.with_field(raw_fixture(), ak.Array([[], [], []]), column)
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "matching object counts"):
                build_event_tensor(records, features)

    def test_cross_type_angles_are_independent_of_multiplicity_thresholds(self):
        records = with_pt(raw_fixture(), ak.Array([[10.], [], [20.]]), "L1T_Electron_PT")
        records = with_pt(records, ak.Array([[5.], [10.], []]), "L1T_MuonTight_PT")
        records = ak.with_field(records, ak.Array([[-1.], [], [0.]]), "L1T_Electron_Eta")
        records = ak.with_field(records, ak.Array([[1.], [0.], []]), "L1T_MuonTight_Eta")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                  thresholds={"electrons": [1000], "muons": [1000]})
        data = self.load()
        index = data["event_feature_columns"].index("abs_delta_eta_leading_electrons_muons")
        np.testing.assert_array_equal(data["event_features"][:, index:index + 6].numpy(),
                                      [[2, 1, 0, 1, 2, 1], [0] * 6, [0] * 6])

    def test_met_angles_append_requested_names_values_and_validity_schema(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        names = ["delta_phi_leading_jet_met", "min_delta_phi_jets_met", "delta_phi_leading_photon_met",
                 "delta_phi_leading_electron_met", "delta_phi_leading_muon_met"]
        self.assertEqual(data["event_feature_columns"][129:139],
                         [entry for name in names for entry in (name, name + "_valid")])
        np.testing.assert_allclose(data["event_features"][:, 129:139].numpy(),
                                   [[0] * 10, [1, 1] * 5, [1, 1] * 5])
        for spec in data["event_feature_schema"][129:139:2]:
            self.assertEqual(spec["kind"], "continuous")
            self.assertEqual(spec["units"], "rad")
            self.assertEqual(len(spec["required_columns"]), 4)
            self.assertIn("L1T_PUPPIMET_MET", spec["required_columns"])
            self.assertIn("L1T_PUPPIMET_Phi", spec["required_columns"])
        self.assertTrue(all(spec["kind"] == "boolean" for spec in data["event_feature_schema"][130:139:2]))

    def test_leading_met_angles_follow_pt_order_and_stable_ties(self):
        records = raw_fixture()
        for kind, base in (("electrons", "L1T_Electron"), ("muons", "L1T_MuonTight"),
                           ("photons", "L1T_PhotonTight"), ("jets", "L1T_JetPuppiAK4")):
            records = with_pt(records, ak.Array([[20., 60., 60.], [20., 50.], [None, np.inf, 10.]]), base + "_PT")
            records = ak.with_field(records, ak.Array([[.1, 2., .2], [1., 2.5], [0., 0., -2.]]), base + "_Phi")
        records = ak.with_field(records, ak.Array([[10.]] * 3), "L1T_PUPPIMET_MET")
        records = ak.with_field(records, ak.Array([[0.]] * 3), "L1T_PUPPIMET_Phi")
        result = build_event_tensor(records, met_angular_features()).numpy()
        for index in (0, 4, 6, 8):
            np.testing.assert_allclose(result[:, index], [2., 2.5, 2.])
            np.testing.assert_array_equal(result[:, index + 1], [1, 1, 1])

    def test_min_jet_met_angle_uses_all_positive_pt_jets_not_only_leading_two(self):
        records = ak.concatenate([raw_fixture()] * 2)
        records = with_pt(records, ak.Array([[100., 80., .5], [100., 50.], [100., 50.],
                                           [100., 0., -1., np.nan, np.inf], [], [5.]]), "L1T_JetPuppiAK4_PT")
        records = ak.with_field(records, ak.Array([[2., 1., .05], [np.nan, .3], [None, .4],
                                                  [2., 0., 0., 0., 0.], [], [0.]]), "L1T_JetPuppiAK4_Phi")
        records = ak.with_field(records, ak.Array([[10.]] * 6), "L1T_PUPPIMET_MET")
        records = ak.with_field(records, ak.Array([[0.]] * 6), "L1T_PUPPIMET_Phi")
        result = build_event_tensor(records, met_angular_features()).numpy()
        np.testing.assert_allclose(result[:, :4],
                                   [[2, 1, .05, 1], [0, 0, .3, 1], [0, 0, .4, 1],
                                    [2, 1, 2, 1], [0, 0, 0, 0], [0, 1, 0, 1]])

    def test_met_angles_wrap_phi_boundary_and_periodic_shifts(self):
        records = raw_fixture()
        for base in ("L1T_Electron", "L1T_MuonTight", "L1T_PhotonTight", "L1T_JetPuppiAK4"):
            records = with_pt(records, ak.Array([[10.]] * 3), base + "_PT")
            records = ak.with_field(records, ak.Array([[np.pi - .1], [.1 + 4 * np.pi], [np.pi]]), base + "_Phi")
        records = ak.with_field(records, ak.Array([[10.]] * 3), "L1T_PUPPIMET_MET")
        records = ak.with_field(records, ak.Array([[-np.pi + .1], [.3 - 6 * np.pi], [-np.pi]]), "L1T_PUPPIMET_Phi")
        result = build_event_tensor(records, met_angular_features()).numpy()
        np.testing.assert_allclose(result[:, ::2], [[.2] * 5, [.2] * 5, [0] * 5], atol=1e-14)
        np.testing.assert_array_equal(result[:, 1::2], np.ones((3, 5)))
        self.assertTrue(np.all((result[:, ::2] >= 0) & (result[:, ::2] <= np.pi)))

    def test_met_angles_invalidate_zero_missing_negative_or_nonfinite_met_and_phi(self):
        records = ak.concatenate([raw_fixture()[1:2]] * 10)
        records = ak.with_field(records, ak.Array([[0.], [], [None], [np.nan], [np.inf],
                                                  [-1.], [10.], [10.], [10.], [10.]]), "L1T_PUPPIMET_MET")
        records = ak.with_field(records, ak.Array([[0.]] * 6 + [[], [None], [np.nan], [np.inf]]), "L1T_PUPPIMET_Phi")
        np.testing.assert_array_equal(build_event_tensor(records, met_angular_features()).numpy(), np.zeros((10, 10)))

    def test_met_angles_keep_invalid_leading_phi_without_promoting_another_object(self):
        records = raw_fixture()
        for base in ("L1T_Electron", "L1T_MuonTight", "L1T_PhotonTight", "L1T_JetPuppiAK4"):
            records = with_pt(records, ak.Array([[100., 50.]] * 3), base + "_PT")
            records = ak.with_field(records, ak.Array([[None, .4], [np.nan, .4], [np.inf, .4]]), base + "_Phi")
        records = ak.with_field(records, ak.Array([[10.]] * 3), "L1T_PUPPIMET_MET")
        records = ak.with_field(records, ak.Array([[0.]] * 3), "L1T_PUPPIMET_Phi")
        result = build_event_tensor(records, met_angular_features()).numpy()
        np.testing.assert_allclose(result, [[0, 0, .4, 1, 0, 0, 0, 0, 0, 0]] * 3)

    def test_met_angles_handle_entirely_empty_or_missing_object_collections(self):
        for entries in ([[], [], []], [[None], [None], [None]], [None, None, None]):
            records = raw_fixture()
            for base in ("L1T_Electron", "L1T_MuonTight", "L1T_PhotonTight", "L1T_JetPuppiAK4"):
                records = with_pt(records, ak.Array(entries), base + "_PT")
            with self.subTest(entries=entries):
                np.testing.assert_array_equal(build_event_tensor(records, met_angular_features()).numpy(), np.zeros((3, 10)))

    def test_met_angle_native_phi_width_alignment_and_missing_column_are_errors(self):
        for column, entries in (("L1T_PUPPIMET_Phi", [[0., 1.], [0.], [0.]]),
                                ("L1T_JetPuppiAK4_Phi", [[], [], []])):
            records = ak.with_field(raw_fixture(), ak.Array(entries), column)
            with self.subTest(column=column), self.assertRaises(ValueError):
                build_event_tensor(records, met_angular_features())
        ak.to_parquet(ak.without_field(raw_fixture(), "L1T_PUPPIMET_Phi"), self.source)
        with self.assertRaises((ValueError, KeyError)):
            convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False)
        self.assertFalse(self.output.exists())

    def test_met_angles_can_be_disabled_and_do_not_depend_on_count_thresholds(self):
        names = {feature.name for feature in met_angular_features()}
        disabled = default_features(include_met=False)
        self.assertTrue(names.isdisjoint(feature.name for feature in disabled))
        self.assertFalse(any(column in {"L1T_PUPPIMET_MET", "L1T_PUPPIMET_Phi"}
                             for feature in disabled for column in feature.columns))
        convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False,
                                  thresholds={kind: [10000] for kind in ("electrons", "muons", "photons", "jets")})
        data = self.load()
        self.assertNotIn("native_event_info", data)
        index = data["event_feature_columns"].index("delta_phi_leading_jet_met")
        np.testing.assert_allclose(data["event_features"][:, index:index + 10].numpy(), [[0] * 10, [1, 1] * 5, [1, 1] * 5])

    def test_pt_ratio_features_are_appended_and_use_the_same_object_type(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        names = data["event_feature_columns"]
        expected_names = [name for kind in ("electrons", "muons", "photons", "jets")
                          for name in (f"leading_pt_fraction_{kind}", f"leading_pt_fraction_{kind}_valid",
                                       f"pt_balance_{kind}", f"pt_balance_{kind}_valid")]
        self.assertEqual(names[37:53], expected_names)
        for kind in ("electrons", "muons", "photons", "jets"):
            leading = data["event_features"][:, names.index(f"leading_pt_{kind}")]
            subleading = data["event_features"][:, names.index(f"subleading_pt_{kind}")]
            total = data["event_features"][:, names.index(f"sum_pt_{kind}")]
            fraction = data["event_features"][:, names.index(f"leading_pt_fraction_{kind}")]
            balance = data["event_features"][:, names.index(f"pt_balance_{kind}")]
            torch.testing.assert_close(fraction[1:], leading[1:] / total[1:])
            torch.testing.assert_close(balance[1:], subleading[1:] / leading[1:])
            self.assertEqual(fraction[0].item(), 0)
            self.assertEqual(balance[0].item(), 0)
        self.assertEqual([spec["kind"] for spec in data["event_feature_schema"][37:53]],
                         ["continuous", "boolean"] * 8)
        self.assertTrue(all(spec["units"] == "1" for spec in data["event_feature_schema"][37:53]))

    def test_pt_ratios_handle_empty_single_invalid_and_tied_objects(self):
        for kind in ("electrons", "muons", "photons", "jets"):
            features = pt_ratio_features(kind)
            entries = ak.Array([[None, np.nan, np.inf, -np.inf, -1., 0.], [7.125], [60., 30., 60.]])
            records = ak.with_field(raw_fixture(), entries, features[0].columns[0])
            result = build_event_tensor(records, features)
            np.testing.assert_allclose(result.numpy(), [[0, 0, 0, 0], [1, 1, 0, 0], [.4, 1, 1, 1]])
            self.assertTrue(bool(torch.isfinite(result).all()))
        with self.assertRaises(ValueError):
            pt_ratio_features("taus")

    def test_photon_ratio_denominator_is_not_all_object_activity(self):
        records = ak.with_field(raw_fixture(), ak.Array([[], [100., 80.], [100.]]), "L1T_PhotonTight_PT")
        records = ak.with_field(records, ak.Array([[], [1000.], []]), "L1T_JetPuppiAK4_PT")
        result = build_event_tensor(records, pt_ratio_features("photons"))
        np.testing.assert_allclose(result.numpy(), [[0, 0, 0, 0], [100 / 180, 1, .8, 1], [1, 1, 0, 0]])

    def test_ratio_features_support_entirely_empty_or_missing_collections(self):
        for entries in ([[], [], []], [[None], [None], [None]]):
            with self.subTest(entries=entries):
                records = ak.with_field(raw_fixture(), ak.Array(entries), "L1T_PhotonTight_PT")
                result = build_event_tensor(records, pt_ratio_features("photons"))
                np.testing.assert_array_equal(result.numpy(), np.zeros((3, 4)))

    def test_pt_ratios_are_scale_invariant_and_independent_of_count_thresholds(self):
        records = with_pt(raw_fixture(), ak.Array([[100., 80.], [1000., 800.], []]), "L1T_PhotonTight_PT")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1, thresholds={"photons": [2000]})
        data = self.load()
        index = data["event_feature_columns"].index("leading_pt_fraction_photons")
        np.testing.assert_allclose(data["event_features"][:, index:index + 4].numpy(),
                                   [[100 / 180, 1, .8, 1], [100 / 180, 1, .8, 1], [0, 0, 0, 0]])

    def test_met_and_ratio_validity_handle_zero_missing_nonfinite_and_overflow(self):
        records = ak.concatenate([raw_fixture()] * 3)
        records = ak.with_field(records, ak.Array([[0.], [10.], [], [None], [np.nan],
                                                   [np.inf], [-1.], [1e-320], [20.]]), "L1T_PUPPIMET_MET")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()["event_features"]
        self.assertEqual(data[:, 26].tolist(), [1, 1, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(data[:, 28].tolist(), [0, 1, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(data[:, 27].numpy(), [0, 100.6, 0, 0, 0, 0, 0, 0, 162])
        self.assertTrue(bool(torch.isfinite(data).all()))

    def test_zero_activity_positive_met_has_valid_zero_ratio(self):
        records = ak.with_field(raw_fixture()[:1], ak.Array([[10.]]), "L1T_PUPPIMET_MET")
        data = build_event_tensor(records, default_features())
        self.assertEqual(data[0, 24:29].tolist(), [0, 10, 1, 0, 1])

    def test_event_met_does_not_require_other_native_columns(self):
        records = raw_fixture()
        required = {column for feature in default_features() for column in feature.columns}
        required.update(RAW_OBJECT_COLUMNS)
        for column in list(records.fields):
            if column not in required:
                records = ak.without_field(records, column)
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False)
        data = self.load()
        self.assertNotIn("native_event_info", data)
        self.assertEqual(data["event_features"][:, 25].tolist(), [0, 50, 100])

    def test_event_met_wrong_width_raises_even_without_native_info(self):
        records = ak.with_field(raw_fixture(), ak.Array([[1., 2.], [1.], [1.]]), "L1T_PUPPIMET_MET")
        ak.to_parquet(records, self.source)
        with self.assertRaisesRegex(ValueError, "one value"):
            convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False)
        self.assertFalse(self.output.exists())

    def test_pt_sums_ignore_missing_nonfinite_and_negative_entries(self):
        records = ak.with_field(raw_fixture(),
                                ak.Array([[None, np.nan, np.inf, -np.inf, -1., 0., 1.25, 2.5], [], [0.]]),
                                "L1T_PhotonTight_PT")
        values = pt_sum_feature("photons").compute(records)
        np.testing.assert_array_equal(values, [3.75, 0., 0.])
        with self.assertRaises(ValueError):
            pt_sum_feature("taus")

    def test_pt_sums_preserve_fractional_values_and_ignore_count_threshold_overrides(self):
        records = with_pt(raw_fixture(), ak.Array([[1.00000001, 2.00000002], [], []]),
                                "L1T_Electron_PT")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1, thresholds={"electrons": [1000]})
        data = self.load()
        index = data["event_feature_columns"].index("sum_pt_electrons")
        self.assertAlmostEqual(data["event_features"][0, index].item(), 3.00000003, places=14)
        count_index = data["event_feature_columns"].index("n_electrons_pt1000")
        self.assertEqual(data["event_features"][0, count_index].item(), 0)

    def test_rows_files_and_sidecar_remain_aligned_across_total_cap(self):
        second = self.root / "second.parquet"
        ak.to_parquet(raw_fixture(), second)
        convert_raw_parquet_to_pt([self.source, second], self.output, label=5, max_events=5)
        data = self.load()
        self.assertEqual(data["raw_source_file"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(data["raw_source_row"].tolist(), [0, 1, 2, 0, 1])
        self.assertEqual(data["native_event_info"][:, 0].tolist(), [0., 50., 100., 0., 50.])
        self.assertEqual(json.loads(self.output.with_suffix(".sources.json").read_text()),
                         [str(self.source), str(second)])

    def test_later_files_are_not_read_after_cap_is_reached(self):
        convert_raw_parquet_to_pt([self.source, self.root / "not_opened.parquet"],
                                  self.output, label=1, max_events=2)
        self.assertEqual(len(self.load()["label"]), 2)

    def test_only_required_columns_are_read(self):
        original = ak.from_parquet
        with patch("raw_event_conversion.convert.ak.from_parquet", wraps=original) as reader:
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        columns = reader.call_args.kwargs["columns"]
        self.assertEqual(len(columns), 35)
        self.assertEqual({column for column in columns if column.endswith("_Eta")},
                         {"L1T_Electron_Eta", "L1T_MuonTight_Eta", "L1T_PhotonTight_Eta", "L1T_JetPuppiAK4_Eta"})
        self.assertTrue({"L1T_Electron_Phi", "L1T_MuonTight_Phi", "L1T_PhotonTight_Phi", "L1T_JetPuppiAK4_Phi"}
                        <= set(columns))
        self.assertFalse(any("PUPPIPart" in column for column in columns))
        self.assertTrue({"L1T_PFPart_PT", "L1T_PFPart_Charge"} <= set(columns))
        self.assertTrue({"L1T_PFPart_D0", "L1T_PFPart_ErrorD0"} <= set(columns))
        self.assertTrue({"L1T_Electron_D0", "L1T_MuonTight_D0"} <= set(columns))
        self.assertFalse(any(column.endswith("_ErrorD0") and not column.startswith("L1T_PFPart_")
                             for column in columns))
        self.assertFalse(any(column.endswith(("_PID", "_IsPU", "_IsRecoPU", "_PuppiW")) for column in columns))

    def test_missing_and_nonfinite_pt_entries_do_not_count(self):
        records = with_pt(raw_fixture(), ak.Array([[None, np.nan, np.inf, -1., 0., 6.], [], []]),
                                "L1T_MuonTight_PT")
        data = build_event_tensor(records, default_features())
        self.assertEqual(data[0, 1].item(), 1)
        self.assertEqual(data[0, 8].item(), 1)

    def test_no_float16_downcast_changes_threshold_membership(self):
        records = with_pt(raw_fixture(), ak.Array([[30.001], [], []]), "L1T_PhotonTight_PT")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=5)
        data = self.load()
        index = data["event_feature_columns"].index("n_photons_pt30")
        self.assertEqual(data["event_features"][0, index].item(), 1)

    def test_native_values_and_validity_masks(self):
        records = ak.with_field(raw_fixture(), ak.Array([[np.nan], [], [200.]]), "L1T_ScalarHT_HT")
        records = ak.with_field(records, ak.Array([[], [1.] * 5, [2.] * 5]), "L1T_Rho_Rho")
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        data = self.load()
        valid = data["native_event_info_validity"]
        self.assertEqual(data["native_event_info"].dtype, torch.float64)
        self.assertTrue(bool(valid[0, 0]))
        self.assertFalse(bool(valid[0, 1]))
        self.assertFalse(bool(valid[0, 4]))
        self.assertFalse(bool(valid[1, 4]))
        self.assertFalse(bool(valid[0, 5:].any()))
        self.assertTrue(bool(torch.isfinite(data["native_event_info"]).all()))

    def test_native_wrong_width_raises_without_saving(self):
        records = ak.with_field(raw_fixture(), ak.Array([[1., 2.], [1.], [1.]]), "L1T_MET_MET")
        ak.to_parquet(records, self.source)
        with self.assertRaises(ValueError):
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertFalse(self.output.exists())

    def test_native_info_can_be_explicitly_omitted(self):
        records = raw_fixture()
        for name in list(records.fields):
            if (name.endswith(("_MET", "_HT", "_Rho"))
                    or (name.endswith("_Phi") and "MET" in name)):
                records = ak.without_field(records, name)
        ak.to_parquet(records, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1, include_native_event_info=False,
                                  include_met_features=False)
        data = self.load()
        self.assertNotIn("native_event_info", data)
        self.assertEqual(data["event_features"][:, 23].tolist(), [0, 404, 1800])
        self.assertEqual(data["event_features"][:, 24].tolist(), [0, 1006, 3240])
        self.assertNotIn("met", data["event_feature_columns"])
        self.assertFalse(data["conversion_metadata"]["met_features_included"])

    def test_cli_can_disable_met_and_native_info_for_files_without_met(self):
        records = raw_fixture()
        required = {column for feature in default_features(include_met=False) for column in feature.columns}
        required.update(RAW_OBJECT_COLUMNS)
        for column in list(records.fields):
            if column not in required:
                records = ak.without_field(records, column)
        ak.to_parquet(records, self.source)
        with redirect_stdout(io.StringIO()):
            main(["--input", str(self.source), "--output", str(self.output), "--label", "1",
                  "--without-native-event-info", "--without-met-features"])
        self.assertEqual(tuple(self.load()["event_features"].shape), (3, 179))
        self.assertIn("subleading_pt_jets", self.load()["event_feature_columns"])

    def test_threshold_overrides_keep_zero_and_other_defaults(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                  thresholds={"muons": [7.5, 5, 5, 0], "jets": []})
        columns = self.load()["event_feature_columns"]
        self.assertIn("n_muons_pt7p5", columns)
        self.assertEqual(columns.count("n_muons_pt5"), 1)
        self.assertNotIn("n_muons_pt10", columns)
        self.assertIn("n_jets", columns)
        self.assertNotIn("n_jets_pt20", columns)
        self.assertIn("n_photons_pt50", columns)

    def test_extension_columns_and_float64_values_are_preserved(self):
        scales = [1.00000001, 2.00000002, 3.00000003]
        records = ak.with_field(raw_fixture(), ak.Array(scales), "custom_scale")
        ak.to_parquet(records, self.source)
        feature = EventFeature("scale", ("custom_scale",), lambda records: records.custom_scale)
        convert_raw_parquet_to_pt(self.source, self.output, label=1, extra_features=[feature])
        data = self.load()
        self.assertEqual(data["event_features"].dtype, torch.float64)
        self.assertEqual(data["event_features"][:, -1].tolist(), scales)
        self.assertEqual(data["event_feature_schema"][-1]["kind"], "continuous")

    def test_duplicate_or_invalid_extension_outputs_fail(self):
        invalid = [
            EventFeature("n_muons", (), lambda _: [0, 0, 0]),
            EventFeature("bad_shape", (), lambda _: np.zeros((3, 1))),
            EventFeature("bad_value", (), lambda _: [1., np.nan, 2.]),
            EventFeature("bad_integer", (), lambda _: [1.5, 2, 3], kind="ordinal"),
            EventFeature("bad_bool", (), lambda _: [0, 1, 2], kind="boolean"),
            EventFeature("bad_columns", "L1T_JetPuppiAK4_PT", lambda _: [0, 0, 0]),
        ]
        for feature in invalid:
            with self.subTest(feature=feature.name), self.assertRaises((ValueError, TypeError)):
                convert_raw_parquet_to_pt(self.source, self.output, label=1, extra_features=[feature])
        self.assertFalse(self.output.exists())

    def test_missing_required_columns_and_empty_inputs_are_errors(self):
        ak.to_parquet(ak.without_field(raw_fixture(), "L1T_MuonTight_PT"), self.source)
        with self.assertRaises((ValueError, KeyError)):
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        ak.to_parquet(raw_fixture()[:0], self.source)
        with self.assertRaises(ValueError):
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertFalse(self.output.exists())

    def test_invalid_inputs_and_options_fail(self):
        for options in ({"max_events": 0}, {"max_events": True}, {"label": -1},
                        {"label": True}, {"thresholds": {"muons": [-5]}},
                        {"thresholds": {"taus": [10]}}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                convert_raw_parquet_to_pt(self.source, self.output, **({"label": 1} | options))
        for sources in ([], [self.source, self.source]):
            with self.assertRaises(ValueError):
                convert_raw_parquet_to_pt(sources, self.output, label=1)
        for source in (str(self.root), str(self.root / "*.parquet")):
            with self.assertRaises(ValueError):
                convert_raw_parquet_to_pt(source, self.output, label=1)

    def test_existing_output_and_sidecars_are_protected(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            convert_raw_parquet_to_pt(self.source, self.output, label=1, overwrite=False)
        self.assertEqual(self.output.read_bytes(), before)
        other = self.root / "other.pt"
        other.with_suffix(".sources.json").write_text("keep this")
        with self.assertRaises(FileExistsError):
            convert_raw_parquet_to_pt(self.source, other, label=1, overwrite=False)
        self.assertFalse(other.exists())

    def test_default_overwrite_warns_and_updates_both_files(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        second = self.root / "second.parquet"
        ak.to_parquet(raw_fixture(), second)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert_raw_parquet_to_pt(second, self.output, label=5, max_events=1)
        self.assertEqual(len(caught), 2)
        self.assertTrue(all("Overwriting existing file:" in str(item.message) for item in caught))
        self.assertEqual(self.load()["label"].tolist(), [5])
        self.assertEqual(json.loads(self.output.with_suffix(".sources.json").read_text()), [str(second)])

    def test_default_overwrite_handles_existing_sidecar_only(self):
        sidecar = self.output.with_suffix(".sources.json")
        sidecar.write_text("old metadata")
        with self.assertWarnsRegex(UserWarning, "Overwriting existing file:"):
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertEqual(json.loads(sidecar.read_text()), [str(self.source)])
        self.assertEqual(len(self.load()["label"]), 3)

    def test_overwrite_cannot_replace_inputs_or_output_symlinks(self):
        before = self.source.read_bytes()
        for source in (self.source, self.source.as_uri()):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "input file"):
                convert_raw_parquet_to_pt(source, self.source, label=1)
        linked_output = self.root / "linked.pt"
        os.link(self.source, linked_output)
        with self.assertRaisesRegex(ValueError, "input file"):
            convert_raw_parquet_to_pt(self.source, linked_output, label=1)
        sidecar = self.output.with_suffix(".sources.json")
        os.link(self.source, sidecar)
        with self.assertRaisesRegex(ValueError, "input file"):
            convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.output.exists())
        linked_output.unlink()
        linked_output.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            convert_raw_parquet_to_pt(self.source, linked_output, label=1)

    def test_failed_serialization_preserves_existing_outputs(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        sidecar = self.output.with_suffix(".sources.json")
        before = (self.output.read_bytes(), sidecar.read_bytes())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with patch("raw_event_conversion.convert.torch.save", side_effect=RuntimeError("save failed")):
                with self.assertRaisesRegex(RuntimeError, "save failed"):
                    convert_raw_parquet_to_pt(self.source, self.output, label=5)
        self.assertEqual((self.output.read_bytes(), sidecar.read_bytes()), before)
        self.assertEqual(list(self.root.glob(".raw_event_conversion_*")), [])

    def test_cli_overwrites_by_default_and_supports_opt_out(self):
        arguments = ["--input", str(self.source), "--output", str(self.output), "--label", "1"]
        with redirect_stdout(io.StringIO()):
            main(arguments)
            with self.assertWarns(UserWarning):
                main(arguments[:-1] + ["5"])
        self.assertEqual(self.load()["label"].tolist(), [5] * 3)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
            main(arguments + ["--no-overwrite"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.load()["label"].tolist(), [5] * 3)

    def test_mixture_script_preserves_all_new_tensors_and_empty_events(self):
        from scripts.make_reference_anomalous import make_datasets
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        background = self.load()
        signal_path = self.root / "signal.pt"
        convert_raw_parquet_to_pt(self.source, signal_path, label=5)
        signal = torch.load(signal_path, weights_only=True)
        reference, anomaly = make_datasets(background, signal, reference_count=2,
                                           background_count=1, signal_count=1, seed=7)
        for dataset in (reference, anomaly):
            for row, (source_id, source_row) in enumerate(zip(dataset["source_id"], dataset["source_index"])):
                original = background if source_id == 0 else signal
                for key, value in original.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(dataset[key][row], value[source_row])
        self.assertEqual(len(reference["label"]), 2)
        self.assertEqual(len(anomaly["label"]), 2)

    def test_event_only_runners_load_and_run_direct_output(self):
        from casual_tests.run_EDT_event_only import main as edt_main
        from casual_tests.run_ERL_Gini_event_only import main as erl_main
        many = ak.concatenate([raw_fixture()] * 20)
        ak.to_parquet(many, self.source)
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        signal_path = self.root / "signal.pt"
        convert_raw_parquet_to_pt(self.source, signal_path, label=5)
        for runner, name in ((edt_main, "edt"), (erl_main, "erl")):
            report = self.root / f"{name}.txt"
            with redirect_stdout(io.StringIO()):
                runner(["--signal", str(signal_path), "--background", str(self.output),
                        "--report", str(report), "--print-top", "1"])
            self.assertEqual(json.loads(report.read_text())["feature_mode"], "event_only")
            specs = json.loads(report.read_text())["event_feature_schema"]
            self.assertTrue(all(spec["kind"] == "continuous" for spec in specs[20:26]))
            self.assertEqual(specs[26]["kind"], "boolean")
            self.assertEqual(specs[28]["kind"], "boolean")
            self.assertTrue(all(spec["kind"] == "continuous" for spec in specs[29:37]))
            self.assertEqual([spec["kind"] for spec in specs[37:139]], ["continuous", "boolean"] * 51)
            self.assertEqual([spec["kind"] for spec in specs[139:183]], ["ordinal", "continuous"] * 22)
            self.assertEqual([spec["kind"] for spec in specs[183:185]], ["ordinal"] * 2)
            self.assertEqual([spec["kind"] for spec in specs[185:]], ["continuous", "boolean"] * 4)

    def test_module_cli_and_standalone_copy(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--input", str(self.source), "--output", str(self.output), "--label", "1",
                  "--max-events", "1", "--muon-thresholds", "3", "5"])
        self.assertEqual(len(self.load()["label"]), 1)
        self.assertIn("n_muons_pt3", self.load()["event_feature_columns"])
        isolated = self.root / "isolated"
        shutil.copytree(ROOT / "raw_event_conversion", isolated / "raw_event_conversion")
        target = isolated / "copied.pt"
        result = subprocess.run([
            sys.executable, "-B", "-m", "raw_event_conversion", "--input", str(self.source),
            "--output", str(target), "--label", "5",
        ], cwd=isolated, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(torch.load(target, weights_only=True)["label"].tolist(), [5] * 3)


if __name__ == "__main__":
    unittest.main()
