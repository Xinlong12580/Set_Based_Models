"""Event observables computed directly from flat raw Parquet columns."""

from dataclasses import dataclass, field
from itertools import combinations
from numbers import Real
from typing import Callable

import awkward as ak
import numpy as np
import torch


PT_COLUMNS = {
    "electrons": "L1T_Electron_PT",
    "muons": "L1T_MuonTight_PT",
    "photons": "L1T_PhotonTight_PT",
    "jets": "L1T_JetPuppiAK4_PT",
}
ETA_COLUMNS = {kind: column.removesuffix("_PT") + "_Eta" for kind, column in PT_COLUMNS.items()}
PHI_COLUMNS = {kind: column.removesuffix("_PT") + "_Phi" for kind, column in PT_COLUMNS.items()}
LEPTON_D0_COLUMNS = {kind: PT_COLUMNS[kind].removesuffix("_PT") + "_D0"
                     for kind in ("electrons", "muons")}
MET_COLUMNS = ("L1T_PUPPIMET_MET", "L1T_PUPPIMET_Phi")
CHARGED_PF_COLUMNS = ("L1T_PFPart_PT", "L1T_PFPart_Charge")
CHARGED_PF_PT_BINS = ((2, None), (2, 5), (2, 10), (10, None))
PF_IMPACT_COLUMNS = ("L1T_PFPart_D0", "L1T_PFPart_ErrorD0")
# Delphes D0 and ErrorD0 are stored in mm, unchanged by the production writer.
# None denotes an open upper end and remains valid JSON in saved schemas.
PF_IMPACT_BINS = {"abs_dxy_mm": (0, .1, .5, 1, 5, 10, None),
                  "abs_dxysig": (0, 2, 3, 5, 10, 20, None)}
PF_IMPACT_THRESHOLDS = {"abs_dxy_mm": (.5, 1, 5), "abs_dxysig": (3, 5, 10)}
DEFAULT_THRESHOLDS = {
    "electrons": (10, 20, 30, 50),
    "muons": (5, 10, 20, 40),
    "photons": (10, 20, 30, 50),
    "jets": (20, 30, 50, 100),
}


@dataclass(frozen=True)
class EventFeature:
    """A named observable: input columns, calculation, and plain metadata.

    ``compute`` receives raw Awkward records and returns one finite value per
    event. Callables are used during conversion only, never saved in the .pt.
    """

    name: str
    columns: tuple[str, ...]
    compute: Callable
    kind: str = "continuous"
    units: str = "1"
    definition: dict = field(default_factory=dict)

    def schema(self):
        return {"name": self.name, "kind": self.kind, "units": self.units,
                "required_columns": list(self.columns), "definition": self.definition}


def multiplicity_feature(object_type, threshold):
    """Count finite PT values strictly above threshold, without slot caps."""
    if object_type not in PT_COLUMNS:
        raise ValueError(f"Unknown object type: {object_type}")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError("PT thresholds must be real numbers.")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("PT thresholds must be finite and nonnegative.")
    column = PT_COLUMNS[object_type]
    suffix = str(int(threshold)) if threshold.is_integer() else repr(threshold)
    suffix = suffix.replace(".", "p").replace("-", "m").replace("+", "")
    name = f"n_{object_type}" + (f"_pt{suffix}" if threshold else "")

    def compute(records):
        pt = ak.values_astype(records[column], np.float64)
        passed = ak.fill_none(np.isfinite(pt) & (pt > threshold), False)
        return ak.to_numpy(ak.sum(passed, axis=1))

    return EventFeature(name, (column,), compute, "ordinal", "count",
                        {"raw_column": column, "pt_threshold_GeV": threshold,
                         "comparison": ">", "finite_pt_required": True,
                         "object_truncation": False, "additional_cuts": None})


def pt_sum_feature(object_type):
    """Scalar sum of every finite, positive object PT, without slot caps."""
    if object_type not in PT_COLUMNS:
        raise ValueError(f"Unknown object type: {object_type}")
    column = PT_COLUMNS[object_type]

    def compute(records):
        return _sum_pt(records, column)

    return EventFeature(f"sum_pt_{object_type}", (column,), compute, "continuous", "GeV",
                        {"raw_column": column, "aggregation": "scalar_sum",
                         "pt_threshold_GeV": 0, "comparison": ">",
                         "finite_pt_required": True, "object_truncation": False,
                         "additional_cuts": None})


def _sum_pt(records, column):
    pt = ak.values_astype(records[column], np.float64)
    valid = ak.fill_none(np.isfinite(pt) & (pt > 0), False)
    return ak.to_numpy(ak.sum(ak.where(valid, pt, 0.0), axis=1))


def ranked_pt_feature(object_type, rank=1):
    """Leading/subleading PT from the full collection, zero if absent."""
    if object_type not in PT_COLUMNS:
        raise ValueError(f"Unknown object type: {object_type}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (1, 2):
        raise ValueError("rank must be 1 (leading) or 2 (subleading).")
    column = PT_COLUMNS[object_type]
    prefix = "leading" if rank == 1 else "subleading"

    def compute(records):
        return _ranked_pt(records, column)[rank - 1]

    return EventFeature(f"{prefix}_pt_{object_type}", (column,), compute, units="GeV",
                        definition={"raw_column": column, "rank": rank, "order": "descending",
                                    "pt_threshold_GeV": 0, "comparison": ">",
                                    "finite_pt_required": True, "object_truncation": False,
                                    "missing_fill": 0, "presence_condition": f"n_{object_type} >= {rank}",
                                    "additional_cuts": None})


def _ranked_pt(records, column):
    pt = ak.values_astype(records[column], np.float64)
    valid = ak.fill_none(np.isfinite(pt) & (pt > 0), False)
    ordered = ak.pad_none(ak.sort(pt[valid], axis=1, ascending=False), 2, axis=1, clip=True)
    values = np.asarray(ak.to_numpy(ak.fill_none(ordered, 0.0)), dtype=np.float64)
    return values[:, 0], values[:, 1]


def _object_values(records, pt_column, column):
    """Read aligned per-object PT and another coordinate without filtering."""
    pt = ak.fill_none(ak.values_astype(records[pt_column], np.float64), [], axis=0)
    values = ak.fill_none(ak.values_astype(records[column], np.float64), [], axis=0)
    if not np.array_equal(ak.to_numpy(ak.num(pt, axis=1)), ak.to_numpy(ak.num(values, axis=1))):
        raise ValueError(f"{pt_column} and {column} must have matching object counts per event.")
    return pt, values


def charged_pf_bin_features(pt_min, pt_max=None):
    """Count and scalar PT sum of charged raw PF candidates in a strict PT window.

    Use unweighted producer PT and finite nonzero charge. No vertex, eta,
    quality or truth selection is applied; empty collections produce zero.
    """
    for bound in (pt_min,) if pt_max is None else (pt_min, pt_max):
        if (isinstance(bound, (bool, np.bool_)) or not isinstance(bound, Real)
                or not np.isfinite(bound) or bound < 0):
            raise ValueError("PF PT bounds must be finite nonnegative real numbers.")
    pt_min = float(pt_min)
    pt_max = None if pt_max is None else float(pt_max)
    if pt_max is not None and pt_max <= pt_min:
        raise ValueError("PF PT upper bound must be greater than the lower bound.")

    def token(value):
        return (str(int(value)) if value.is_integer() else repr(value)).replace(".", "p").replace("+", "")

    suffix = f"pt_gt{token(pt_min)}"
    if pt_max is not None:
        suffix += f"_lt{token(pt_max)}"

    def selected_pt(records):
        pt, charge = _object_values(records, *CHARGED_PF_COLUMNS)
        selected = np.isfinite(pt) & (pt > pt_min) & np.isfinite(charge) & (charge != 0)
        if pt_max is not None:
            selected = selected & (pt < pt_max)
        return pt, ak.fill_none(selected, False)

    def count(records):
        _, selected = selected_pt(records)
        return ak.to_numpy(ak.sum(selected, axis=1))

    def sum_pt(records):
        pt, selected = selected_pt(records)
        return ak.to_numpy(ak.sum(ak.where(selected, pt, 0.0), axis=1))

    definition = {"raw_pt_column": CHARGED_PF_COLUMNS[0],
                  "raw_charge_column": CHARGED_PF_COLUMNS[1],
                  "pt_min_GeV": pt_min, "pt_max_GeV": pt_max,
                  "lower_comparison": ">", "upper_comparison": "<" if pt_max is not None else None,
                  "charge_selection": "finite and nonzero", "finite_pt_required": True,
                  "pt_weighting": "none", "vertex_selection": None,
                  "additional_cuts": None, "candidate_truncation": False}
    return (
        EventFeature(f"n_charged_pf_{suffix}", CHARGED_PF_COLUMNS, count, "ordinal", "count",
                     {**definition, "aggregation": "count"}),
        EventFeature(f"sum_pt_charged_pf_{suffix}", CHARGED_PF_COLUMNS, sum_pt, units="GeV",
                     definition={**definition, "aggregation": "scalar_sum"}),
    )


def _charged_pf_impact_values(records, observable):
    """Aligned raw PT, eligible charged candidates, magnitude and validity."""
    pt, charge = _object_values(records, *CHARGED_PF_COLUMNS)
    _, d0 = _object_values(records, CHARGED_PF_COLUMNS[0], PF_IMPACT_COLUMNS[0])
    eligible = ak.fill_none(np.isfinite(pt) & (pt > 2) & np.isfinite(charge) & (charge != 0), False)
    value = np.abs(d0)
    valid = ak.fill_none(np.isfinite(d0), False)
    if observable == "abs_dxysig":
        _, error = _object_values(records, CHARGED_PF_COLUMNS[0], PF_IMPACT_COLUMNS[1])
        valid = valid & ak.fill_none(np.isfinite(error) & (error > 0), False)
        # Invalid denominators never become artificial zero/large significance.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = value / ak.where(valid, error, 1.0)
        valid = valid & ak.fill_none(np.isfinite(value), False)
    return pt, eligible, value, valid


def charged_pf_impact_features():
    """Counts/PT sums in |D0| (mm) and |D0|/ErrorD0 bins, plus invalid counts.

    All selections use finite nonzero charge and unweighted PT > 2 GeV.
    Bins are [lower, upper); cumulative cuts are strictly > threshold.
    D0 is the producer's beam-spot/origin impact parameter, not PV-recomputed
    dxy. Significance excludes nonpositive/nonfinite uncertainties and ratios.
    """
    features = []

    def token(value):
        value = float(value)
        return (str(int(value)) if value.is_integer() else repr(value)).replace(".", "p")

    def range_features(observable, lower, upper, cumulative=False):
        columns = (*CHARGED_PF_COLUMNS, PF_IMPACT_COLUMNS[0])
        if observable == "abs_dxysig":
            columns += (PF_IMPACT_COLUMNS[1],)
        suffix = f"pt_gt2_{observable}_{'gt' if cumulative else 'ge'}{token(lower)}"
        if upper is not None:
            suffix += f"_lt{token(upper)}"

        def aggregate(records, scalar_sum):
            pt, eligible, value, valid = _charged_pf_impact_values(records, observable)
            within = value > lower if cumulative else value >= lower
            if upper is not None:
                within = within & (value < upper)
            selected = eligible & valid & ak.fill_none(within, False)
            return ak.to_numpy(ak.sum(ak.where(selected, pt, 0.0) if scalar_sum else selected, axis=1))

        definition = {"raw_pt_column": CHARGED_PF_COLUMNS[0], "raw_charge_column": CHARGED_PF_COLUMNS[1],
                      "raw_d0_column": PF_IMPACT_COLUMNS[0], "raw_d0_units": "mm",
                      "dxy_reference": "producer beam spot (origin if unavailable), not reconstructed PV",
                      "formula": "abs(D0) / ErrorD0" if observable == "abs_dxysig" else "abs(D0)",
                      "range_units": "1" if observable == "abs_dxysig" else "mm",
                      "lower_bound": lower, "upper_bound": upper,
                      "lower_comparison": ">" if cumulative else ">=",
                      "upper_comparison": "<" if upper is not None else None,
                      "pt_min_GeV": 2, "pt_comparison": ">", "pt_weighting": "none",
                      "charge_selection": "finite and nonzero", "finite_pt_required": True,
                      "invalid_measurements": "excluded", "vertex_selection": None,
                      "additional_cuts": None, "candidate_truncation": False}
        if observable == "abs_dxysig":
            definition.update(raw_error_d0_column=PF_IMPACT_COLUMNS[1], raw_error_d0_units="mm",
                              finite_positive_error_required=True, finite_ratio_required=True,
                              denominator_regularization=None)
        return (
            EventFeature(f"n_charged_pf_{suffix}", columns,
                         lambda records: aggregate(records, False), "ordinal", "count",
                         {**definition, "aggregation": "count"}),
            EventFeature(f"sum_pt_charged_pf_{suffix}", columns,
                         lambda records: aggregate(records, True), units="GeV",
                         definition={**definition, "aggregation": "scalar_sum"}),
        )

    for observable, edges in PF_IMPACT_BINS.items():
        for lower, upper in zip(edges[:-1], edges[1:]):
            features.extend(range_features(observable, lower, upper))
    for observable, thresholds in PF_IMPACT_THRESHOLDS.items():
        for threshold in thresholds:
            features.extend(range_features(observable, threshold, None, cumulative=True))
    for observable in PF_IMPACT_BINS:
        columns = (*CHARGED_PF_COLUMNS, PF_IMPACT_COLUMNS[0])
        if observable == "abs_dxysig":
            columns += (PF_IMPACT_COLUMNS[1],)

        def invalid_count(records, observable=observable):
            _, eligible, _, valid = _charged_pf_impact_values(records, observable)
            return ak.to_numpy(ak.sum(eligible & ~valid, axis=1))

        features.append(EventFeature(
            f"n_charged_pf_pt_gt2_{observable}_invalid", columns, invalid_count, "ordinal", "count",
            {"observable": observable, "pt_min_GeV": 2, "pt_comparison": ">",
             "charge_selection": "finite and nonzero", "aggregation": "invalid_measurement_count",
             "invalid_condition": "nonfinite D0 or nonfinite/nonpositive ErrorD0 or nonfinite ratio"
                                  if observable == "abs_dxysig" else "nonfinite D0",
             "interpretation": "measurement availability diagnostic, not displaced particle count"},
        ))
    return tuple(features)


def _ranked_object_values(records, pt_column, column):
    """Values tied to the top two PT objects; missing ranks stay NaN."""
    pt, values = _object_values(records, pt_column, column)
    eligible = ak.fill_none(np.isfinite(pt) & (pt > 0), False)
    if not ak.any(eligible):
        return np.full((len(records), 2), np.nan)
    # Rank by PT only: an invalid associated value must not promote another object.
    order = ak.argsort(pt[eligible], axis=1, ascending=False, stable=True)
    ordered = ak.pad_none(values[eligible][order], 2, axis=1, clip=True)
    return np.asarray(ak.to_numpy(ak.fill_none(ordered, np.nan)), dtype=np.float64)


def _abs_delta_phi(first, second):
    """Shortest absolute azimuthal separation, in radians in [0, pi]."""
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.remainder(first, 2 * np.pi) - np.remainder(second, 2 * np.pi)
        return np.abs(np.arctan2(np.sin(difference), np.cos(difference)))


def ranked_abs_eta_features(object_type):
    """Absolute eta of the PT-leading/subleading objects, with validity flags."""
    if object_type not in PT_COLUMNS:
        raise ValueError(f"Unknown object type: {object_type}")
    pt_column, eta_column = PT_COLUMNS[object_type], ETA_COLUMNS[object_type]

    def ranked_eta(records):
        values = _ranked_object_values(records, pt_column, eta_column)
        valid = np.isfinite(values)
        return np.where(valid, np.abs(values), 0.0), valid

    features = []
    for index, prefix in enumerate(("leading", "subleading")):
        name = f"{prefix}_abs_eta_{object_type}"
        definition = {"raw_pt_column": pt_column, "raw_eta_column": eta_column,
                      "rank": index + 1, "ranking": "descending finite positive PT",
                      "tie_break": "producer order", "finite_eta_required": True,
                      "invalid_fill": 0, "validity_feature": name + "_valid",
                      "object_truncation": False, "additional_cuts": None}
        features.append(EventFeature(
            name, (pt_column, eta_column),
            lambda records, index=index: ranked_eta(records)[0][:, index], definition=definition,
        ))
        features.append(EventFeature(
            name + "_valid", (pt_column, eta_column),
            lambda records, index=index: ranked_eta(records)[1][:, index], kind="boolean",
            definition={"formula": "PT-ranked object exists and its eta is finite",
                        "rank": index + 1, "tie_break": "producer order"},
        ))
    return tuple(features)


def ranked_abs_dxy_features(object_type):
    """Absolute producer D0 (mm) of PT-leading/subleading leptons, with flags."""
    if object_type not in LEPTON_D0_COLUMNS:
        raise ValueError(f"Lepton D0 features support electrons and muons, not {object_type}.")
    pt_column, d0_column = PT_COLUMNS[object_type], LEPTON_D0_COLUMNS[object_type]

    def ranked_dxy(records):
        values = _ranked_object_values(records, pt_column, d0_column)
        valid = np.isfinite(values)
        return np.where(valid, np.abs(values), 0.0), valid

    features = []
    for index, prefix in enumerate(("leading", "subleading")):
        name = f"{prefix}_abs_dxy_mm_{object_type}"
        definition = {"raw_pt_column": pt_column, "raw_d0_column": d0_column,
                      "raw_d0_units": "mm", "formula": "abs(D0)",
                      "dxy_reference": "producer beam spot (origin if unavailable), not reconstructed PV",
                      "rank": index + 1, "ranking": "descending finite positive PT",
                      "tie_break": "producer order", "finite_d0_required": True,
                      "invalid_fill": 0, "validity_feature": name + "_valid",
                      "object_truncation": False, "additional_cuts": None}
        features.append(EventFeature(
            name, (pt_column, d0_column),
            lambda records, index=index: ranked_dxy(records)[0][:, index], units="mm", definition=definition,
        ))
        features.append(EventFeature(
            name + "_valid", (pt_column, d0_column),
            lambda records, index=index: ranked_dxy(records)[1][:, index], kind="boolean",
            definition={"formula": "PT-ranked lepton exists and its D0 is finite",
                        "rank": index + 1, "tie_break": "producer order",
                        "raw_pt_column": pt_column, "raw_d0_column": d0_column},
        ))
    return tuple(features)


def leading_pair_angular_features(object_type):
    """|Delta eta|, wrapped |Delta phi| and Delta R of the top two PT objects."""
    return _pair_angular_features(object_type, object_type)


def cross_type_angular_features(first_type, second_type):
    """Angular separations between the PT-leading objects of two different types."""
    for kind in (first_type, second_type):
        if kind not in PT_COLUMNS:
            raise ValueError(f"Unknown object type: {kind}")
    if first_type == second_type:
        raise ValueError("Cross-type angular features require two different object types.")
    # These separations are symmetric; use one name/order for either argument order.
    first_type, second_type = sorted((first_type, second_type), key=list(PT_COLUMNS).index)
    return _pair_angular_features(first_type, second_type)


def _pair_angular_features(first_type, second_type):
    for kind in (first_type, second_type):
        if kind not in PT_COLUMNS:
            raise ValueError(f"Unknown object type: {kind}")
    same_type = first_type == second_type
    suffix = first_type if same_type else f"leading_{first_type}_{second_type}"
    pair = "PT-leading and PT-subleading objects" if same_type else "PT-leading object of each type"
    pt_columns = (PT_COLUMNS[first_type], PT_COLUMNS[second_type])
    eta_columns = (ETA_COLUMNS[first_type], ETA_COLUMNS[second_type])
    phi_columns = (PHI_COLUMNS[first_type], PHI_COLUMNS[second_type])

    def separation(records, coordinate):
        columns = eta_columns if coordinate == "eta" else phi_columns
        first = _ranked_object_values(records, pt_columns[0], columns[0])
        second = first if same_type else _ranked_object_values(records, pt_columns[1], columns[1])
        values = np.column_stack((first[:, 0], second[:, 1 if same_type else 0]))
        with np.errstate(over="ignore", invalid="ignore"):
            if coordinate == "phi":
                result = _abs_delta_phi(values[:, 0], values[:, 1])
            else:
                result = np.abs(values[:, 0] - values[:, 1])
        valid = np.isfinite(values).all(axis=1) & np.isfinite(result)
        return np.where(valid, result, 0.0), valid

    def delta_r(records):
        deta, eta_valid = separation(records, "eta")
        dphi, phi_valid = separation(records, "phi")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.hypot(deta, dphi)
        valid = eta_valid & phi_valid & np.isfinite(result)
        return np.where(valid, result, 0.0), valid

    definitions = (
        ("abs_delta_eta", (*pt_columns, *eta_columns), "1", "abs(eta_first - eta_second)",
         lambda records: separation(records, "eta")),
        ("abs_delta_phi", (*pt_columns, *phi_columns), "rad",
         "abs(atan2(sin(phi_first - phi_second), cos(phi_first - phi_second)))",
         lambda records: separation(records, "phi")),
        ("delta_r", (*pt_columns, *eta_columns, *phi_columns), "1",
         "hypot(abs_delta_eta, abs_delta_phi)", delta_r),
    )
    features = []
    for prefix, columns, units, formula, compute in definitions:
        name = f"{prefix}_{suffix}"
        columns = tuple(dict.fromkeys(columns))
        definition = {"formula": formula, "pair": pair, "object_types": [first_type, second_type],
                      "ranks": [1, 2 if same_type else 1],
                      "ranking": "descending finite positive PT", "tie_break": "producer order",
                      "invalid_fill": 0, "validity_feature": name + "_valid",
                      "object_truncation": False, "additional_cuts": None}
        features.append(EventFeature(
            name, columns, lambda records, compute=compute: compute(records)[0],
            units=units, definition=definition,
        ))
        features.append(EventFeature(
            name + "_valid", columns, lambda records, compute=compute: compute(records)[1],
            kind="boolean", definition={"requires": "two finite positive PT objects, finite required "
                                        "angular coordinates and a finite separation"},
        ))
    return tuple(features)


def pt_ratio_features(object_type):
    """Within-type leading fraction and subleading/leading balance, with flags."""
    if object_type not in PT_COLUMNS:
        raise ValueError(f"Unknown object type: {object_type}")
    column = PT_COLUMNS[object_type]
    fraction_name = f"leading_pt_fraction_{object_type}"
    balance_name = f"pt_balance_{object_type}"

    def ratios(records):
        leading, subleading = _ranked_pt(records, column)
        total = _sum_pt(records, column)
        fraction_valid = np.isfinite(total) & (total > 0)
        balance_valid = (leading > 0) & (subleading > 0)
        fraction = np.zeros(len(records), dtype=np.float64)
        balance = np.zeros(len(records), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.divide(leading, total, out=fraction, where=fraction_valid)
            np.divide(subleading, leading, out=balance, where=balance_valid)
        fraction_valid &= np.isfinite(fraction)
        balance_valid &= np.isfinite(balance)
        return (np.where(fraction_valid, fraction, 0.0), fraction_valid,
                np.where(balance_valid, balance, 0.0), balance_valid)

    definitions = (
        {"formula": f"leading_pt_{object_type} / sum_pt_{object_type}",
         "validity_feature": fraction_name + "_valid", "presence_condition": f"n_{object_type} >= 1"},
        {"formula": f"subleading_pt_{object_type} / leading_pt_{object_type}",
         "validity_feature": balance_name + "_valid", "presence_condition": f"n_{object_type} >= 2"},
    )
    features = []
    for index, (name, definition) in enumerate(zip((fraction_name, balance_name), definitions)):
        features.append(EventFeature(
            name, (column,), lambda records, index=index: ratios(records)[2 * index],
            definition={**definition, "raw_column": column, "invalid_fill": 0,
                        "finite_positive_pt_only": True, "object_truncation": False},
        ))
        features.append(EventFeature(
            name + "_valid", (column,), lambda records, index=index: ratios(records)[2 * index + 1],
            kind="boolean", definition={"requires": definition["presence_condition"],
                                        "finite_ratio_required": True},
        ))
    return tuple(features)


def sum_pt_all_objects(records):
    """Collection-level scalar sum; overlaps across object types are not removed."""
    return np.sum([_sum_pt(records, column) for column in PT_COLUMNS.values()], axis=0)


def _met_values(records):
    column = MET_COLUMNS[0]
    lengths = ak.to_numpy(ak.fill_none(ak.num(records[column], axis=1), 0))
    if np.any((lengths != 0) & (lengths != 1)):
        raise ValueError(f"{column} must contain one value or be empty per event.")
    values = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(records[column]), np.nan)), dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0)
    return np.where(valid, values, 0.0), valid


def _sum_pt_over_met(records):
    met, valid = _met_values(records)
    valid &= met > 0
    ratio = np.zeros(len(records), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(sum_pt_all_objects(records), met, out=ratio, where=valid)
    valid &= np.isfinite(ratio)
    return np.where(valid, ratio, 0.0), valid


def _met_direction(records):
    """Native MET phi is meaningful only for a finite positive MET magnitude."""
    met, met_valid = _met_values(records)
    column = MET_COLUMNS[1]
    entries = ak.fill_none(records[column], [], axis=0)
    lengths = ak.to_numpy(ak.num(entries, axis=1))
    if np.any((lengths != 0) & (lengths != 1)):
        raise ValueError(f"{column} must contain one value or be empty per event.")
    phi = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(entries), np.nan)), dtype=np.float64)
    return phi, met_valid & (met > 0) & np.isfinite(phi)


def met_angular_features():
    """Leading-object/MET angles and minimum over all finite positive-PT jets."""
    met_columns = MET_COLUMNS

    def leading_angle(records, object_type):
        met_phi, met_valid = _met_direction(records)
        phi = _ranked_object_values(records, PT_COLUMNS[object_type], PHI_COLUMNS[object_type])[:, 0]
        angle = _abs_delta_phi(phi, met_phi)
        valid = met_valid & np.isfinite(phi) & np.isfinite(angle)
        return np.where(valid, angle, 0.0), valid

    def minimum_jet_angle(records):
        met_phi, met_valid = _met_direction(records)
        pt, phi = _object_values(records, PT_COLUMNS["jets"], PHI_COLUMNS["jets"])
        eligible = ak.fill_none(np.isfinite(pt) & (pt > 0) & np.isfinite(phi), False)
        if not ak.any(eligible):
            return np.zeros(len(records)), np.zeros(len(records), dtype=bool)
        angles = _abs_delta_phi(phi, met_phi[:, None])
        minimum = ak.min(ak.where(eligible, angles, np.inf), axis=1)
        result = np.asarray(ak.to_numpy(ak.fill_none(minimum, np.nan)), dtype=np.float64)
        valid = met_valid & np.isfinite(result)
        return np.where(valid, result, 0.0), valid

    definitions = (
        ("delta_phi_leading_jet_met", "jets", "PT-leading object", lambda records: leading_angle(records, "jets")),
        ("min_delta_phi_jets_met", "jets", "minimum over all finite positive-PT jets with finite phi", minimum_jet_angle),
        ("delta_phi_leading_photon_met", "photons", "PT-leading object", lambda records: leading_angle(records, "photons")),
        ("delta_phi_leading_electron_met", "electrons", "PT-leading object", lambda records: leading_angle(records, "electrons")),
        ("delta_phi_leading_muon_met", "muons", "PT-leading object", lambda records: leading_angle(records, "muons")),
    )
    features = []
    for name, kind, selection, compute in definitions:
        columns = (PT_COLUMNS[kind], PHI_COLUMNS[kind], *met_columns)
        definition = {"formula": "abs(atan2(sin(phi_object - phi_met), cos(phi_object - phi_met)))",
                      "object_type": kind, "selection": selection, "pt_threshold_GeV": 0,
                      "comparison": ">", "tie_break": "producer order",
                      "requires": "finite positive MET, finite MET phi and selected object phi",
                      "invalid_fill": 0, "validity_feature": name + "_valid",
                      "object_truncation": False, "additional_cuts": None}
        features.append(EventFeature(
            name, columns, lambda records, compute=compute: compute(records)[0],
            units="rad", definition=definition,
        ))
        features.append(EventFeature(
            name + "_valid", columns, lambda records, compute=compute: compute(records)[1],
            kind="boolean", definition={"requires": definition["requires"], "selection": selection},
        ))
    return tuple(features)


def activity_features(*, include_met=True):
    """Total object PT, native MET and PT/MET with explicit Boolean validity."""
    columns = tuple(PT_COLUMNS.values())
    features = [EventFeature("sum_pt_all_objects", columns, sum_pt_all_objects, units="GeV",
                             definition={"aggregation": "scalar_sum", "pt_threshold_GeV": 0,
                                         "finite_pt_required": True, "comparison": ">",
                                         "object_truncation": False, "overlap_removal": False})]
    if include_met:
        met_column = MET_COLUMNS[0]
        met_definition = {"raw_column": met_column, "invalid_fill": 0,
                          "validity_feature": "met_valid", "zero_is_valid": True}
        ratio_definition = {"formula": "sum_pt_all_objects / met", "invalid_fill": 0,
                            "validity_feature": "sum_pt_over_met_valid",
                            "requires": "finite nonnegative MET strictly above zero and finite ratio"}
        features.extend([
            EventFeature("met", (met_column,), lambda records: _met_values(records)[0],
                         units="GeV", definition=met_definition),
            EventFeature("met_valid", (met_column,), lambda records: _met_values(records)[1],
                         kind="boolean", definition={"formula": "MET exists, is finite and >= 0"}),
            EventFeature("sum_pt_over_met", (*columns, met_column),
                         lambda records: _sum_pt_over_met(records)[0], definition=ratio_definition),
            EventFeature("sum_pt_over_met_valid", (*columns, met_column),
                         lambda records: _sum_pt_over_met(records)[1], kind="boolean",
                         definition={"formula": ratio_definition["requires"]}),
        ])
    return tuple(features)


def default_features(thresholds=None, *, include_met=True):
    """Object observables, optional MET, and charged raw-PF counts/PT sums."""
    grids = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        unknown = set(thresholds) - set(grids)
        if unknown:
            raise ValueError(f"Unknown object types: {sorted(unknown)}")
        grids.update(thresholds)
    features = [multiplicity_feature(kind, 0) for kind in PT_COLUMNS]
    for kind, values in grids.items():
        checked = [multiplicity_feature(kind, value) for value in values]
        unique = {feature.definition["pt_threshold_GeV"]: feature for feature in checked}
        features.extend(unique[value] for value in sorted(unique) if value > 0)
    features.extend(pt_sum_feature(kind) for kind in PT_COLUMNS)
    features.extend(activity_features(include_met=include_met))
    features.extend(ranked_pt_feature(kind, rank) for kind in PT_COLUMNS for rank in (1, 2))
    features.extend(feature for kind in PT_COLUMNS for feature in pt_ratio_features(kind))
    features.extend(feature for kind in PT_COLUMNS for feature in ranked_abs_eta_features(kind))
    features.extend(feature for kind in PT_COLUMNS for feature in leading_pair_angular_features(kind))
    features.extend(feature for first, second in combinations(PT_COLUMNS, 2)
                    for feature in cross_type_angular_features(first, second))
    if include_met:
        features.extend(met_angular_features())
    features.extend(feature for bounds in CHARGED_PF_PT_BINS
                    for feature in charged_pf_bin_features(*bounds))
    features.extend(charged_pf_impact_features())
    features.extend(feature for kind in LEPTON_D0_COLUMNS for feature in ranked_abs_dxy_features(kind))
    return tuple(features)


def build_event_tensor(records, features):
    """Evaluate features, retaining integer counts or float64 continuous values."""
    names = [feature.name for feature in features]
    if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name.strip()
                                                       or name != name.strip() for name in names):
        raise ValueError("Feature names must be nonempty, unique, and free of surrounding whitespace.")
    values = []
    for feature in features:
        if feature.kind not in {"ordinal", "boolean", "continuous"}:
            raise ValueError(f"Unsupported feature kind: {feature.kind}")
        result = feature.compute(records)
        result = ak.to_numpy(result) if isinstance(result, ak.Array) else np.asanyarray(result)
        if np.ma.isMaskedArray(result) and np.any(np.ma.getmaskarray(result)):
            raise ValueError(f"{feature.name} contains missing values.")
        result = np.asarray(result)
        if (result.shape != (len(records),) or result.dtype.kind not in "biuf"
                or not np.all(np.isfinite(result))):
            raise ValueError(f"{feature.name} must return one finite numeric value per event.")
        if feature.kind == "ordinal":
            if (not np.all(result == np.floor(result)) or np.any(result < -(2**63))
                    or np.any(result >= 2**63)):
                raise ValueError(f"{feature.name} must contain int64-compatible integers.")
        if feature.kind == "boolean" and not np.all((result == 0) | (result == 1)):
            raise ValueError(f"{feature.name} must contain only 0 and 1.")
        values.append(result)
    dtype = np.float64 if any(feature.kind == "continuous" for feature in features) else np.int64
    return torch.from_numpy(np.column_stack(values).astype(dtype))
