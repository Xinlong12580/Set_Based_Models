"""Critical-threshold extraction followed by empirical rate calibration."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .data import as_batch, input_files, iter_batches
from .evaluation import evaluate_rates, write_json
from .menu import TriggerMenu


@dataclass
class ScanPlan:
    trigger: str
    base_thresholds: dict
    scale_fields: tuple
    reference_field: str
    mode: str = "scale"

    def thresholds_at(self, coordinate):
        if self.mode == "offset":
            return {field: self.base_thresholds[field] + np.asarray(coordinate) for field in self.scale_fields}
        scale = np.asarray(coordinate) / self.base_thresholds[self.reference_field]
        return {field: self.base_thresholds[field] * scale for field in self.scale_fields}


def scan_plan(menu, trigger, reference_field=None, scale_fields=None):
    if not menu.is_tunable(trigger):
        raise ValueError(f"Trigger {trigger} is fixed and has no tunable parameters.")
    base = menu.thresholds_for(trigger)
    definition = menu.triggers[trigger].definition
    fields = tuple(definition.get("scale_fields", base)) if scale_fields is None else tuple(scale_fields)
    reference = reference_field or max(fields, key=lambda field: base[field])
    mode = definition.get("scan_mode", "offset" if any(base[field] == 0 for field in fields) else "scale")
    if reference not in fields or (mode == "scale" and base[reference] <= 0):
        raise ValueError("The reference must be a scaled field with a positive base threshold.")
    return ScanPlan(trigger, base, fields, reference, mode)


def critical_thresholds(data, menu, plan, *, lower=0.0, upper=1000.0, tolerance=0.01):
    """Upper estimates; NaN fails at lower, exact upper marks right censoring."""
    if tolerance <= 0 or upper <= lower:
        raise ValueError("Need upper > lower and a positive threshold tolerance.")
    batch = as_batch(data)
    low, high = np.full(batch.size, lower), np.full(batch.size, upper)
    lower_pass = menu.evaluate(plan.trigger, batch, plan.thresholds_at(low))
    upper_pass = menu.evaluate(plan.trigger, batch, plan.thresholds_at(high))
    active = lower_pass & ~upper_pass
    while np.any(active & (high - low > tolerance)):
        middle = (low + high) / 2
        passed = menu.evaluate(plan.trigger, batch, plan.thresholds_at(middle))
        low = np.where(active & passed, middle, low)
        high = np.where(active & ~passed, middle, high)
    scores = np.where(upper_pass, upper, np.minimum(high, np.nextafter(float(upper), -np.inf)))
    return np.where(lower_pass, scores, np.nan)


def scan_thresholds(inputs, menu=None, triggers=None, *, lower=0.0, upper=1000.0, tolerance=0.01,
                    reference_field=None, scale_fields=None, batch_size=10000, max_events=None):
    if menu is None:
        raise ValueError("Supply a trigger menu.")
    names = menu.implementable_triggers if triggers is None else list(triggers)
    fixed = [name for name in names if not menu.is_tunable(name)]
    plans = {name: scan_plan(menu, name, reference_field, scale_fields) for name in names if name not in fixed}
    parts = {name: [] for name in plans}
    total = 0
    for batch in iter_batches(inputs, batch_size, max_events, with_pairs=menu.requires_pairs(names)):
        total += batch.size
        print(f"Scanning events {total - batch.size + 1}-{total}", flush=True)
        for name, plan in plans.items():
            parts[name].append(critical_thresholds(batch, menu, plan, lower=lower, upper=upper, tolerance=tolerance))
    if total == 0:
        raise ValueError("The inputs contain no events.")
    scores = {name: np.concatenate(values) for name, values in parts.items()}
    metadata = {"format": "collide2v_pt_thresholds", "inputs": [str(path) for path in input_files(inputs)],
                "triggers": names, "fixed_triggers": fixed,
                "event_count": total, "max_events": max_events, "lower": lower, "upper": upper,
                "score_tolerance_gev": tolerance, "menu": menu.to_dict(),
                "plans": {name: asdict(plan) for name, plan in plans.items()}}
    return scores, metadata


def save_scores(path, scores, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=json.dumps(metadata), **scores)


def load_scores(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        scores = {name: saved[name].copy() for name in metadata["plans"]}
    return scores, metadata


def fit_rate(scores, target_rate_khz, *, collision_rate_khz=40000.0, lower=0.0, upper=1000.0,
             method="bisection", tolerance=0.01, spacing=0.25):
    if tolerance <= 0 or upper <= lower or spacing <= 0:
        raise ValueError("Need upper > lower and positive tolerance and spacing.")
    scores = np.asarray(scores)
    ordered = np.sort(scores[np.isfinite(scores)])
    total = len(scores)
    points = []

    def measure(threshold):
        passed = int(len(ordered) - np.searchsorted(ordered, threshold, side="left"))
        point = {"threshold_gev": float(threshold), "passed": passed, "efficiency": passed / total,
                 "rate_khz": collision_rate_khz * passed / total}
        points.append(point)
        return point

    low, high = measure(lower), measure(upper)
    if target_rate_khz > low["rate_khz"]:
        chosen, status = low, "unreachable"
    elif target_rate_khz < high["rate_khz"]:
        chosen, status = high, "unreachable"
    elif low["passed"] == 0:
        chosen, status = low, "unidentified"
    elif low["rate_khz"] == target_rate_khz:
        chosen, status = low, "calibrated"
    else:
        if method == "monotonic":
            for threshold in np.arange(lower + spacing, upper, spacing):
                point = measure(threshold)
                if point["rate_khz"] <= target_rate_khz:
                    high = point
                    break
        elif method == "bisection":
            while high["threshold_gev"] - low["threshold_gev"] > tolerance:
                point = measure((low["threshold_gev"] + high["threshold_gev"]) / 2)
                if point["rate_khz"] > target_rate_khz:
                    low = point
                else:
                    high = point
        else:
            raise ValueError(f"Unknown scan method: {method}")
        chosen = high
        status = "calibrated" if np.isclose(chosen["rate_khz"], target_rate_khz, rtol=1e-9, atol=1e-12) else "discrete"
    return {**chosen, "target_rate_khz": target_rate_khz, "event_count": total, "status": status,
            "rate_difference_khz": chosen["rate_khz"] - target_rate_khz,
            "fails_at_lower_bound": int(np.count_nonzero(~np.isfinite(scores))),
            "passes_at_upper_bound": int(np.count_nonzero(scores >= upper)),
            "scan_points": sorted(points, key=lambda point: point["threshold_gev"])}


def calibrate_menu(inputs=None, menu=None, triggers=None, *, output_dir=None, scores_file=None,
                   target_rate_khz=None, collision_rate_khz=40000.0, lower=0.0, upper=1000.0,
                   score_tolerance=0.01, tolerance=0.01, method="bisection", spacing=0.25,
                   reference_field=None, scale_fields=None, batch_size=10000, max_events=None,
                   export_unreachable_boundary=False):
    if scores_file is None:
        scores, metadata = scan_thresholds(inputs, menu, triggers, lower=lower, upper=upper,
                                           tolerance=score_tolerance, reference_field=reference_field,
                                           scale_fields=scale_fields, batch_size=batch_size, max_events=max_events)
    else:
        scores, metadata = load_scores(scores_file)
        cached_definition = metadata.get("menu")
        if cached_definition is None:
            cached_definition = {"format": "collide2v_pt_threshold_profile",
                                 "thresholds": {name: plan["base_thresholds"] for name, plan in metadata["plans"].items()},
                                 "object_selection": metadata["object_selection"],
                                 "erl_triggers": metadata.get("erl_triggers", [])}
        cached_menu = TriggerMenu.from_dict(cached_definition)
        if menu is not None and (menu.object_selection() != cached_menu.object_selection() or any(
                menu.thresholds_for(name) != cached_menu.thresholds_for(name) for name in metadata["plans"])
                or menu.trigger_definitions() != cached_menu.trigger_definitions()):
            raise ValueError("Saved scores use different menu settings.")
        menu = cached_menu
        lower, upper = metadata["lower"], metadata["upper"]
        inputs, max_events = metadata["inputs"], metadata["max_events"]
    names = metadata.get("triggers", list(scores)) if triggers is None else list(triggers)
    results, thresholds = {}, {}
    for name in names:
        target = menu.target_rate_khz(name) if target_rate_khz is None else target_rate_khz
        if isinstance(target_rate_khz, dict):
            target = target_rate_khz.get(name, menu.target_rate_khz(name))
        if not menu.is_tunable(name):
            thresholds[name] = menu.thresholds_for(name)
            results[name] = {"status": "fixed", "thresholds": thresholds[name], "target_rate_khz": target}
            print(f"{name}: fixed; evaluating original cuts without calibration.", flush=True)
            continue
        if target is None:
            raise ValueError(f"Specify target_rate_khz for imported trigger {name}.")
        result = fit_rate(scores[name], target, collision_rate_khz=collision_rate_khz, lower=lower, upper=upper,
                          method=method, tolerance=tolerance, spacing=spacing)
        plan = ScanPlan(**metadata["plans"][name])
        applied = menu.thresholds_for(name)
        if result["status"] not in ("unreachable", "unidentified") or export_unreachable_boundary:
            applied.update({field: float(value) for field, value in plan.thresholds_at(result["threshold_gev"]).items()})
        thresholds[name] = applied
        result["thresholds"] = applied
        results[name] = result
        print(f"{name}: {result['status']}, {result['threshold_gev']:.6g} GeV, "
              f"score rate {result['rate_khz']:.6g} / target {target:g} kHz", flush=True)
    calibrated = TriggerMenu.from_dict(menu.to_dict(), threshold_overrides=thresholds)
    calibrated.target_rates.update({name: result["target_rate_khz"] for name, result in results.items()})
    measured = evaluate_rates(inputs, calibrated, names, collision_rate_khz=collision_rate_khz,
                              batch_size=batch_size, max_events=max_events)
    profile = calibrated.to_dict()
    profile.update(fixed_triggers=[name for name in names if not menu.is_tunable(name)],
                   collision_rate_khz=collision_rate_khz)
    for name, result in results.items():
        result["applied_measurement"] = measured["triggers"][name]
        if result["status"] == "calibrated" and not np.isclose(
                measured["triggers"][name]["rate_khz"], result["target_rate_khz"], rtol=1e-9, atol=1e-12):
            result["status"] = "discrete"
    profile["calibration_complete"] = all(result["status"] in ("calibrated", "fixed") for result in results.values())
    report = {"profile": profile, "scan": metadata, "calibration": results, "evaluation": measured}
    if output_dir is not None:
        directory = Path(output_dir)
        save_scores(directory / "critical_thresholds.npz", scores, metadata)
        write_json(directory / "calibrated_thresholds.json", profile)
        write_json(directory / "calibration.json", report)
    return report
