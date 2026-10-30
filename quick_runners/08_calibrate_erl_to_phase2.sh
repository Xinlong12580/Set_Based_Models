#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIG="${SIG:-ggHgammagamma}"
ERL_MODEL="${ERL_MODEL:-${REPO_ROOT}/training_outputs/${SIG}_scan.validation.model.json}"
ERL_MENU="${ERL_MENU:-${REPO_ROOT}/Trigger/results/${SIG}_menu.json}"
PHASE2_CALIBRATION="${PHASE2_CALIBRATION:-${REPO_ROOT}/Trigger/results/phase2_calibration}"
ERL_MATCH_DIR="${ERL_MATCH_DIR:-${REPO_ROOT}/Trigger/results/${SIG}_phase2_matched}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
LOWER="${LOWER:-0}"
UPPER="${UPPER:-1000}"
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.01}"
TOLERANCE="${TOLERANCE:-0.01}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/Trigger/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "${ERL_MODEL}" "${ERL_MENU}" "${PHASE2_CALIBRATION}" "${ERL_MATCH_DIR}" \
  "${BATCH_SIZE}" "${LOWER}" "${UPPER}" "${SCORE_TOLERANCE}" "${TOLERANCE}" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from collide2v_trigger import (Trigger, TriggerMenu, evaluate_rates, fit_rate,
                               save_scores, scan_plan, scan_thresholds)
from collide2v_trigger.evaluation import write_json

model_path, menu_path, phase2_dir, output_dir = map(Path, sys.argv[1:5])
batch_size = int(sys.argv[5])
lower, upper, score_tolerance, tolerance = map(float, sys.argv[6:])
model = json.loads(model_path.read_text())
rules = {rule["rule_id"]: rule for rule in model["rules"]}
retained = model["pruning"]["selected_rule_ids"]
if not retained:
    raise ValueError("The ERL model has no retained rules.")
rule_id = max(retained, key=lambda index: (rules[index]["steps"][-1]["generation_score"], -index))
source_menu = TriggerMenu.load(menu_path)
source = next(trigger for trigger in source_menu.triggers.values()
              if trigger.definition.get("type") == "erl" and trigger.definition["source_rule_id"] == rule_id)
if not source.tunable:
    raise ValueError(f"The top ERL rule {rule_id} is fixed and cannot be rate-matched.")
fields = ("scope", "observable", "type", "comparison", "threshold", "accepted_codes")
original = [{key: step["cut"][key] for key in fields if key in step["cut"]} for step in rules[rule_id]["steps"]]
converted = [{key: cut[key] for key in fields if key in cut} for cut in source.cut_actions]
if original != converted or rules[rule_id]["minimum_count"] != source.definition["minimum_count"]:
    raise ValueError("The converted top rule differs from the model; rerun step 06.")

reference = json.loads((phase2_dir / "calibration.json").read_text())
phase2 = TriggerMenu.from_dict(reference["profile"])
inputs = reference["scan"]["inputs"]
event_count = reference["scan"]["event_count"]
collision_rate = reference["evaluation"]["collision_rate_khz"]
base = TriggerMenu(triggers=[source], isolation=source_menu.isolation,
                   electron_iso_max=source_menu.electron_iso_max, photon_iso_max=source_menu.photon_iso_max)
base.thresholds[source.name] = source_menu.thresholds_for(source.name)
print(f"Top retained rule: {source.name}, ID {rule_id}, score {rules[rule_id]['steps'][-1]['generation_score']:.6g}", flush=True)
scores, metadata = scan_thresholds(inputs, base, [source.name], lower=lower, upper=upper,
                                  tolerance=score_tolerance, batch_size=batch_size, max_events=event_count)
save_scores(output_dir / "critical_thresholds.npz", scores, metadata)
plan = scan_plan(base, source.name)
matched = TriggerMenu(isolation=source_menu.isolation, electron_iso_max=source_menu.electron_iso_max,
                      photon_iso_max=source_menu.photon_iso_max)
matches = []
for name, calibration in reference["calibration"].items():
    measurement = calibration["applied_measurement"]
    target = measurement["rate_khz"]
    entry = {"phase2_trigger": name, "phase2_status": calibration["status"],
             "phase2_nominal_target_khz": calibration["target_rate_khz"],
             "phase2_measurement": measurement, "target_rate_khz": target,
             "erl_trigger": None, "rate_matched": False}
    matches.append(entry)
    if calibration["status"] not in ("calibrated", "discrete"):
        entry["status"] = "skipped_phase2_" + calibration["status"]
        continue
    if target <= 0:
        entry["status"] = "skipped_zero_rate"
        continue
    fit = fit_rate(scores[source.name], target, collision_rate_khz=collision_rate,
                   lower=lower, upper=upper, tolerance=tolerance)
    entry["fit"] = fit
    if fit["status"] not in ("calibrated", "discrete"):
        entry["status"] = "erl_" + fit["status"]
        continue
    definition = source.to_dict()
    definition.update(name=f"{source.name}__{name}", target_rate_khz=target,
                      matched_phase2_trigger=name, source_trigger=source.name)
    trigger = Trigger.from_dict(definition)
    matched.add_trigger(trigger)
    thresholds = {**base.thresholds_for(source.name),
                  **{key: float(value) for key, value in plan.thresholds_at(fit["threshold_gev"]).items()}}
    matched.thresholds[trigger.name] = thresholds
    entry.update(erl_trigger=trigger.name, status=fit["status"], thresholds=thresholds)

measured = evaluate_rates(inputs, matched, collision_rate_khz=collision_rate,
                         batch_size=batch_size, max_events=event_count)
for entry in matches:
    if entry["erl_trigger"] is None:
        print(f"{entry['phase2_trigger']}: {entry['status']}", flush=True)
        continue
    applied = measured["triggers"][entry["erl_trigger"]]
    entry["erl_measurement"] = applied
    entry["rate_difference_khz"] = applied["rate_khz"] - entry["target_rate_khz"]
    entry["rate_matched"] = bool(np.isclose(applied["rate_khz"], entry["target_rate_khz"], rtol=1e-9, atol=1e-12))
    entry["status"] = "calibrated" if entry["rate_matched"] else "discrete"
    print(f"{entry['phase2_trigger']}: ERL {applied['rate_khz']:g} / reference {entry['target_rate_khz']:g} kHz ({entry['status']})", flush=True)

matched.save(output_dir / "matched_menu.json")
write_json(output_dir / "matching.json", {
    "format": "collide2v_rate_matching", "schema_version": 1,
    "erl_model": str(model_path.resolve()), "erl_menu": str(menu_path.resolve()),
    "source_rule_id": rule_id, "source_trigger": source.name,
    "source_score": rules[rule_id]["steps"][-1]["generation_score"],
    "phase2_calibration": str((phase2_dir / "calibration.json").resolve()),
    "phase2_menu": phase2.to_dict(), "matched_menu": matched.to_dict(),
    "calibration_inputs": metadata["inputs"], "event_count": metadata["event_count"],
    "collision_rate_khz": collision_rate, "matches": matches})
print("Matching results:", dict(Counter(entry["status"] for entry in matches)))
print(f"Saved {len(matched.triggers)} ERL working points to {output_dir / 'matched_menu.json'}")
PY
