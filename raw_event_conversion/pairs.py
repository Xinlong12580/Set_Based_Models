"""Distinct particle pairs with copied member properties and guarded kinematics."""

from itertools import combinations

import numpy as np
import torch

from .features import _abs_delta_phi
from .objects import OBJECT_COLUMNS, OBJECT_SLOTS, _rank_object_arrays, object_feature_schema


# Original slots are ordered by type, then decreasing PT within each type.
# Keeping i < j gives canonical members and avoids self/reversed duplicates.
PAIR_OBJECT_INDICES = tuple(combinations(range(sum(OBJECT_SLOTS.values())), 2))
MEMBER_COLUMNS = (*OBJECT_COLUMNS, "eta_valid", "phi_valid")
# name: (schema kind, units, definition)
PAIR_FEATURES = {
    "min_pt": ("continuous", "GeV", "min(pt_1, pt_2)"),
    "sum_pt": ("continuous", "GeV", "pt_1 + pt_2"),
    "pair_pt": ("continuous", "GeV", "magnitude of the vector transverse-momentum sum"),
    "pt_balance": ("continuous", "1", "min(pt_1, pt_2) / max(pt_1, pt_2)"),
    "pair_mass": ("continuous", "GeV", "invariant mass of the two member four-vectors"),
    "abs_delta_eta": ("continuous", "1", "abs(eta_1 - eta_2)"),
    "abs_delta_phi": ("continuous", "rad", "absolute wrapped phi difference in [0, pi]"),
    "delta_r": ("continuous", "1", "hypot(abs_delta_eta, abs_delta_phi)"),
    "max_isolation_rho_corr": ("continuous", "1", "maximum of two valid corrected isolations"),
    "min_abs_d0": ("continuous", "mm", "minimum of two valid absolute stored D0 values"),
    "opposite_sign": ("boolean", "1", "opposite charges for two leptons with charge +/-1"),
    "opposite_eta_hemispheres": ("boolean", "1", "eta_1 and eta_2 have strictly opposite signs"),
}
PAIR_COLUMNS = (
    *(f"{name}_{member}" for member in (1, 2) for name in MEMBER_COLUMNS),
    *(column for name in PAIR_FEATURES for column in (name, name + "_valid")),
)
# GeV; PDG 2025 listings. Jets use their measured, valid stored mass instead.
LEPTON_MASSES = {"muons": 0.1056583755, "electrons": 0.00051099895}


def pair_feature_schema():
    """Plain ObservableSpec-compatible metadata, with all guard links explicit."""
    schema = []
    for member in (1, 2):
        for original in object_feature_schema():
            entry = {**original, "name": f"{original['name']}_{member}", "pair_member": member}
            if "validity_column" in entry:
                entry["validity_column"] += f"_{member}"
            if "value_column" in entry:
                entry["value_column"] += f"_{member}"
            if original["name"] in ("eta", "phi"):
                entry["validity_column"] = f"{original['name']}_valid_{member}"
            schema.append(entry)
        schema.extend({"name": f"{name}_valid_{member}", "kind": "boolean", "units": "1",
                       "value_column": f"{name}_{member}",
                       "definition": "finite raw coordinate on the selected member"}
                      for name in ("eta", "phi"))
    for name, (kind, units, definition) in PAIR_FEATURES.items():
        schema.append({"name": name, "kind": kind, "units": units,
                       "definition": definition, "invalid_fill": 0,
                       "validity_column": name + "_valid"})
        schema.append({"name": name + "_valid", "kind": "boolean", "units": "1",
                       "value_column": name,
                       "definition": "both members exist and required measurements/result are valid"})
    return schema


def _invariant_mass(pt1, pt2, eta1, eta2, delta_phi, mass1, mass2):
    """Stable massive invariant mass; avoids subtracting nearly equal E^2/p^2."""
    mt1, mt2 = np.hypot(pt1, mass1), np.hypot(pt2, mass2)
    y1 = np.arcsinh(pt1 * np.sinh(eta1) / mt1)
    y2 = np.arcsinh(pt2 * np.sinh(eta2) / mt2)
    # Rationalize mt - pt to retain small lepton masses in collinear pairs.
    offset = mass1**2 / (mt1 + pt1) * mt2 + pt1 * mass2**2 / (mt2 + pt2)
    squared = (mass1**2 + mass2**2 + 2 * offset
               + 4 * mt1 * mt2 * np.sinh((y1 - y2) / 2)**2
               + 4 * pt1 * pt2 * np.sin(delta_phi / 2)**2)
    return np.sqrt(squared)


def build_pair_tensor(records):
    """Return (float32 [N,231,68] pairs, bool [N,231] presence mask).

    Pairs use the same capped, ranked objects as obj, but are computed from
    raw float64 values before coordinate zero filling/float32 downcasting.
    Every valid pair contains two distinct objects. Undefined properties are
    zero with a false schema-linked guard. Missing jet mass never becomes a
    massless-jet assumption; electrons/muons use fixed masses and photons zero.
    """
    objects, coordinates = _rank_object_arrays(records)
    source = {name: i for i, name in enumerate(OBJECT_COLUMNS)}
    target = {name: i for i, name in enumerate(PAIR_COLUMNS)}
    indices = np.asarray(PAIR_OBJECT_INDICES)
    first, second = indices[:, 0], indices[:, 1]
    present = (objects[:, first, source["pt"]] > 0) & (objects[:, second, source["pt"]] > 0)
    if np.any(np.abs(objects) > np.finfo(np.float32).max):
        raise ValueError("Member values exceed the float32 pair format.")
    pairs = np.zeros((*present.shape, len(PAIR_COLUMNS)), dtype=np.float32)

    # Gather one column at a time rather than duplicating full float64 members.
    for member, slots in ((1, first), (2, second)):
        for name, column in source.items():
            pairs[..., target[f"{name}_{member}"]] = np.where(present, objects[:, slots, column], 0)
        for name, valid in coordinates.items():
            pairs[..., target[f"{name}_valid_{member}"]] = present & valid[:, slots]

    def value(name, slots):
        return objects[:, slots, source[name]]

    def store(name, values, valid=present):
        valid = present & valid & np.isfinite(values) & (np.abs(values) <= np.finfo(np.float32).max)
        pairs[..., target[name]] = np.where(valid, values, 0)
        pairs[..., target[name + "_valid"]] = valid

    def member_mass(slots):
        types = value("type_id", slots)
        masses = value("mass", slots).copy()
        valid = value("mass_valid", slots).astype(bool)
        for type_id, kind in enumerate(OBJECT_SLOTS):
            if kind != "jets":
                selected = types == type_id
                masses[selected] = LEPTON_MASSES.get(kind, 0.0)
                valid[selected] = True
        return masses, valid

    pt1, pt2 = value("pt", first), value("pt", second)
    eta1, eta2 = value("eta", first), value("eta", second)
    phi1, phi2 = value("phi", first), value("phi", second)
    eta_valid = coordinates["eta"][:, first] & coordinates["eta"][:, second]
    phi_valid = coordinates["phi"][:, first] & coordinates["phi"][:, second]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        store("min_pt", np.minimum(pt1, pt2))
        store("sum_pt", pt1 + pt2)
        store("pt_balance", np.minimum(pt1, pt2) / np.maximum(pt1, pt2))
        store("pair_pt", np.hypot(pt1 * np.cos(phi1) + pt2 * np.cos(phi2),
                                  pt1 * np.sin(phi1) + pt2 * np.sin(phi2)), phi_valid)
        delta_eta, delta_phi = np.abs(eta1 - eta2), _abs_delta_phi(phi1, phi2)
        store("abs_delta_eta", delta_eta, eta_valid)
        store("abs_delta_phi", delta_phi, phi_valid)
        store("delta_r", np.hypot(delta_eta, delta_phi), eta_valid & phi_valid)
        mass1, valid1 = member_mass(first)
        mass2, valid2 = member_mass(second)
        store("pair_mass", _invariant_mass(pt1, pt2, eta1, eta2, delta_phi, mass1, mass2),
              eta_valid & phi_valid & valid1 & valid2)
        for name, original, combine in (
            ("max_isolation_rho_corr", "isolation_rho_corr", np.maximum),
            ("min_abs_d0", "abs_d0", np.minimum),
        ):
            valid = value(original + "_valid", first).astype(bool) & value(original + "_valid", second).astype(bool)
            store(name, combine(value(original, first), value(original, second)), valid)
        charge1, charge2 = value("charge", first), value("charge", second)
        types1, types2 = value("type_id", first), value("type_id", second)
        leptons = [i for i, kind in enumerate(OBJECT_SLOTS) if kind in LEPTON_MASSES]
        charge_valid = (np.isin(types1, leptons) & np.isin(types2, leptons)
                        & value("charge_valid", first).astype(bool) & value("charge_valid", second).astype(bool)
                        & (np.abs(charge1) == 1) & (np.abs(charge2) == 1))
        store("opposite_sign", charge1 * charge2 < 0, charge_valid)
        store("opposite_eta_hemispheres", ((eta1 < 0) & (eta2 > 0)) | ((eta1 > 0) & (eta2 < 0)), eta_valid)
    return torch.from_numpy(pairs), torch.from_numpy(present)
