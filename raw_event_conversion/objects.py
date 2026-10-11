"""Uniform PT-ranked object tensors with explicit validity for extra fields."""

import awkward as ak
import numpy as np
import torch

from .features import ETA_COLUMNS, PHI_COLUMNS, PT_COLUMNS, _object_values


# Keep the old ordering, caps and type IDs, but use PUPPI jets and omit MET.
OBJECT_SLOTS = {"jets": 10, "muons": 4, "electrons": 4, "photons": 4}
# name: (raw suffix, applicable types, schema kind, units)
OBJECT_FEATURES = {
    "abs_d0": ("D0", ("muons", "electrons"), "continuous", "mm"),
    "isolation_rho_corr": ("IsolationVarRhoCorr", ("muons", "electrons", "photons"), "continuous", "1"),
    "mass": ("Mass", ("jets",), "continuous", "GeV"),
    "n_charged": ("NCharged", ("jets",), "ordinal", "count"),
    "n_neutrals": ("NNeutrals", ("jets",), "ordinal", "count"),
    "charged_fraction": (None, ("jets",), "continuous", "1"),
    "ehad_over_eem": ("EhadOverEem", ("electrons", "photons"), "continuous", "1"),
    "charge": ("Charge", ("jets", "muons", "electrons"), "ordinal", "e"),
}
BASE_COLUMNS = ("pt", "eta", "phi", "type_id")
OBJECT_COLUMNS = (*BASE_COLUMNS, *OBJECT_FEATURES,
                  *(name + "_valid" for name in OBJECT_FEATURES))


def _source_columns(kind):
    prefix = PT_COLUMNS[kind].removesuffix("_PT")
    return {
        **dict(zip(BASE_COLUMNS[:3], (PT_COLUMNS[kind], ETA_COLUMNS[kind], PHI_COLUMNS[kind]))),
        **{name: prefix + "_" + suffix for name, (suffix, types, _, _) in OBJECT_FEATURES.items()
           if suffix is not None and kind in types},
    }


RAW_OBJECT_COLUMNS = tuple(column for kind in OBJECT_SLOTS for column in _source_columns(kind).values())


def object_feature_schema():
    """Plain metadata describing types, applicability, sources and validity."""
    schema = [
        {"name": "pt", "kind": "continuous", "units": "GeV"},
        {"name": "eta", "kind": "continuous", "units": "1"},
        {"name": "phi", "kind": "continuous", "units": "rad"},
        {"name": "type_id", "kind": "categorical",
         "categories": {index: kind for index, kind in enumerate(OBJECT_SLOTS)}},
    ]
    for name, (suffix, types, kind, units) in OBJECT_FEATURES.items():
        sources = {object_type: (
            [_source_columns(object_type)["n_charged"], _source_columns(object_type)["n_neutrals"]]
            if suffix is None else [_source_columns(object_type)[name]])
            for object_type in types}
        definition = ("abs(D0)" if name == "abs_d0" else
                      "NCharged / (NCharged + NNeutrals)" if name == "charged_fraction" else suffix)
        schema.append({"name": name, "kind": kind, "units": units,
                       "applicable_types": list(types), "raw_columns": sources,
                       "definition": definition, "invalid_fill": 0, "validity_column": name + "_valid"})
    schema.extend({"name": name + "_valid", "kind": "boolean", "units": "1",
                   "definition": "applicable, present and valid for this selected object",
                   "value_column": name} for name in OBJECT_FEATURES)
    return schema


def _rank_object_arrays(records):
    """Shared float64 ranking, retaining coordinate validity before zero fill."""
    missing = set(RAW_OBJECT_COLUMNS) - set(records.fields)
    if missing:
        raise ValueError(f"Missing raw object columns: {sorted(missing)}")
    shape = (len(records), sum(OBJECT_SLOTS.values()))
    objects = np.zeros((*shape, len(OBJECT_COLUMNS)), dtype=np.float64)
    coordinate_validity = {name: np.zeros(shape, dtype=bool) for name in ("eta", "phi")}
    indices = {name: index for index, name in enumerate(OBJECT_COLUMNS)}
    start = 0
    for type_id, (kind, width) in enumerate(OBJECT_SLOTS.items()):
        columns = _source_columns(kind)
        raw = {name: _object_values(records, PT_COLUMNS[kind], column)[1]
               for name, column in columns.items()}
        pt = raw["pt"]
        present = ak.fill_none(np.isfinite(pt) & (pt > 0), False)
        if not ak.any(present):
            start += width
            continue
        order = ak.argsort(pt[present], axis=1, ascending=False, stable=True)
        ranked = {}
        for name, values in raw.items():
            padded = ak.pad_none(values[present][order], width, axis=1, clip=True)
            ranked[name] = np.asarray(ak.to_numpy(ak.fill_none(padded, np.nan)), dtype=np.float64)
        if "abs_d0" in ranked:
            ranked["abs_d0"] = np.abs(ranked["abs_d0"])
        if kind == "jets":
            charged, neutral = ranked["n_charged"], ranked["n_neutrals"]
            counts_valid = (np.isfinite(charged) & np.isfinite(neutral)
                            & (charged >= 0) & (neutral >= 0)
                            & (charged == np.floor(charged)) & (neutral == np.floor(neutral)))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                total = charged + neutral
                ranked["charged_fraction"] = np.where(
                    counts_valid & np.isfinite(total) & (total > 0), charged / total, np.nan)
        for name, values in ranked.items():
            valid = np.isfinite(values)
            if name in ("mass", "n_charged", "n_neutrals", "ehad_over_eem"):
                valid &= values >= 0
            if name in ("n_charged", "n_neutrals", "charge"):
                valid &= values == np.floor(values)
            values = np.where(valid, values, 0.0)
            objects[:, start:start + width, indices[name]] = values
            if name in coordinate_validity:
                coordinate_validity[name][:, start:start + width] = valid
            if name in OBJECT_FEATURES:
                objects[:, start:start + width, indices[name + "_valid"]] = valid
        objects[:, start:start + width, indices["type_id"]] = np.where(
            np.isfinite(ranked["pt"]), type_id, 0)
        start += width
    return objects, coordinate_validity


def build_object_tensor(records):
    """Return float32 [events, 22, 20], preserving the original object format.

    Finite positive PT defines presence and stable within-type ranking. Extra
    measurements have schema-linked validity flags; eta/phi remain zero-filled
    without flags in this legacy object tensor. No additional physics cuts are
    applied. Event summaries are computed separately before slot caps.
    """
    objects, _ = _rank_object_arrays(records)
    if np.any(np.abs(objects) > np.finfo(np.float32).max):
        raise ValueError("Object values exceed the float32 obj format.")
    return torch.from_numpy(objects.astype(np.float32))
