"""Object combinations and kinematic selections used by declarative triggers."""

from itertools import combinations, product

import numpy as np


def delta_r(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def pair_mass(objects, i, j, mass=None):
    pt, eta, phi = (objects.field(name) for name in ("pt", "eta", "phi"))
    m1, m2 = (objects.field("mass")[:, i], objects.field("mass")[:, j]) if mass is None else (mass, mass)
    px = pt[:, i] * np.cos(phi[:, i]) + pt[:, j] * np.cos(phi[:, j])
    py = pt[:, i] * np.sin(phi[:, i]) + pt[:, j] * np.sin(phi[:, j])
    pz1, pz2 = pt[:, i] * np.sinh(eta[:, i]), pt[:, j] * np.sinh(eta[:, j])
    energy = np.sqrt(pt[:, i] ** 2 + pz1 ** 2 + m1 ** 2) + np.sqrt(pt[:, j] ** 2 + pz2 ** 2 + m2 ** 2)
    return np.sqrt(np.maximum(energy ** 2 - px ** 2 - py ** 2 - (pz1 + pz2) ** 2, 0))


def compatible_dz(members, maximum=10.0):
    values = np.column_stack([objects.field("dz")[:, i] for objects, i in members])
    valid = np.column_stack([objects.valid("dz")[:, i] for objects, i in members]).all(axis=1)
    return valid & (np.max(values, axis=1) - np.min(values, axis=1) < maximum)


def object_groups(objects, thresholds, *, eta_max=2.4, isolation=None, pt_floor=None):
    selected = objects.eligible(eta_max, isolation)
    if pt_floor is not None:
        selected &= objects.field("pt") > np.asarray(pt_floor)[..., None]
    size = len(objects.values)
    cuts = np.sort(np.column_stack([np.broadcast_to(t, (size,)) for t in thresholds]), axis=1)[:, ::-1]
    for indices in combinations(range(len(objects)), len(thresholds)):
        pts = np.sort(objects.field("pt")[:, indices], axis=1)[:, ::-1]
        yield indices, selected[:, indices].all(axis=1) & (pts >= cuts).all(axis=1)


def group(batch, kind, thresholds, *, eta_max=2.4, isolation=None, dz=False,
          dz_max=10.0, delta_eta_max=None, dr_max=None, mass_min=None, mass_window=None,
          opposite_charge=False, pair_mass_hypothesis=None, pt_floor=None):
    objects = batch.objects(kind)
    if dz and "dz" not in objects.columns:
        raise KeyError("Relative-DZ seeds need dz/dz_valid. Reconvert the input with the updated converter.")
    result = np.zeros(batch.size, dtype=bool)
    for indices, passed in object_groups(objects, thresholds, eta_max=eta_max, isolation=isolation, pt_floor=pt_floor):
        if dz:
            passed &= compatible_dz([(objects, i) for i in indices], dz_max)
        if delta_eta_max is not None:
            i, j = indices
            passed &= np.abs(objects.field("eta")[:, i] - objects.field("eta")[:, j]) < delta_eta_max
        if dr_max is not None:
            i, j = indices
            passed &= objects.valid("phi")[:, i] & objects.valid("phi")[:, j]
            passed &= delta_r(objects.field("eta")[:, i], objects.field("phi")[:, i],
                              objects.field("eta")[:, j], objects.field("phi")[:, j]) < dr_max
        if mass_min is not None:
            i, j = indices
            passed &= (objects.valid("mass") & objects.valid("phi"))[:, indices].all(axis=1)
            passed &= pair_mass(objects, i, j) > mass_min
        if opposite_charge or mass_window is not None:
            pair_pass = np.zeros(batch.size, dtype=bool)
            for i, j in combinations(indices, 2):
                opposite = np.ones(batch.size, dtype=bool)
                if opposite_charge:
                    opposite &= objects.valid("charge")[:, i] & objects.valid("charge")[:, j]
                    opposite &= objects.field("charge")[:, i] * objects.field("charge")[:, j] < 0
                if mass_window is not None:
                    mass = pair_mass(objects, i, j, pair_mass_hypothesis)
                    opposite &= objects.valid("phi")[:, i] & objects.valid("phi")[:, j]
                    if pair_mass_hypothesis is None:
                        opposite &= objects.valid("mass")[:, i] & objects.valid("mass")[:, j]
                    opposite &= (mass > mass_window[0]) & (mass < mass_window[1])
                pair_pass |= opposite
            passed &= pair_pass
        result |= passed
    return result


def multi_group(batch, groups, dz=False, dz_max=10.0):
    objects = [batch.objects(settings["kind"]) for settings in groups]
    if dz and any("dz" not in members.columns for members in objects):
        raise KeyError("Relative-DZ seeds need dz/dz_valid. Reconvert the input with the updated converter.")
    result = np.zeros(batch.size, dtype=bool)
    choices = [list(object_groups(members, **{key: value for key, value in settings.items() if key != "kind"}))
               for members, settings in zip(objects, groups)]
    for combination in product(*choices):
        if any(objects[i] is objects[j] and set(combination[i][0]) & set(combination[j][0])
               for i, j in combinations(range(len(objects)), 2)):
            continue
        passed = np.logical_and.reduce([flags for _, flags in combination])
        if dz:
            members = [(collection, index) for collection, (indices, _) in zip(objects, combination) for index in indices]
            passed &= compatible_dz(members, dz_max)
        result |= passed
    return result


def ht(batch, jet_pt_min=30.0, eta_max=2.4):
    jets = batch.objects("jets")
    selected = jets.eligible(eta_max) & (jets.field("pt") > np.asarray(jet_pt_min)[..., None])
    return np.where(selected, jets.field("pt"), 0).sum(axis=1)


def matched_groups(batch, first, second, *, dr_min=None, dr_max=None,
                   second_delta_eta_pair=None, second_delta_eta_max=None):
    left, right = batch.objects(first["kind"]), batch.objects(second["kind"])
    result = np.zeros(batch.size, dtype=bool)
    left_groups = list(object_groups(left, **{key: value for key, value in first.items() if key != "kind"}))
    right_groups = list(object_groups(right, **{key: value for key, value in second.items() if key != "kind"}))

    def coordinates(objects, indices):
        order = np.argsort(-objects.field("pt")[:, indices], axis=1, kind="stable")
        return [np.take_along_axis(values[:, indices], order, axis=1)
                for values in (objects.field("eta"), objects.field("phi"), objects.valid("phi"))]

    for (li, lpass), (ri, rpass) in product(left_groups, right_groups):
        if left is right and set(li) & set(ri):
            continue
        leta, lphi, lvalid = coordinates(left, li)
        reta, rphi, rvalid = coordinates(right, ri)
        passed = lpass & rpass & lvalid[:, 0] & rvalid[:, 0]
        dr = delta_r(leta[:, 0], lphi[:, 0], reta[:, 0], rphi[:, 0])
        if dr_min is not None:
            passed &= dr > dr_min
        if dr_max is not None:
            passed &= dr < dr_max
        if second_delta_eta_pair is not None:
            i, j = second_delta_eta_pair
            passed &= np.abs(reta[:, i] - reta[:, j]) < second_delta_eta_max
        result |= passed
    return result
