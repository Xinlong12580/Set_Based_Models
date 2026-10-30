#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIG="${SIG:-ggHgammagamma}"
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/SIGNAL}"
PHASE2_CALIBRATION="${PHASE2_CALIBRATION:-${REPO_ROOT}/Trigger/results/phase2_calibration}"
PHASE2_MENU="${PHASE2_MENU:-${PHASE2_CALIBRATION}/calibrated_thresholds.json}"
ERL_MATCH_DIR="${ERL_MATCH_DIR:-${REPO_ROOT}/Trigger/results/${SIG}_phase2_matched}"
COMPARISON_DIR="${COMPARISON_DIR:-${REPO_ROOT}/Trigger/results/${SIG}_phase2_comparison}"
MINBIAS_TEST="${MINBIAS_TEST:-${SIG_WORK_DIR}/test_minbias.chunks.json}"
SIGNALS="${SIGNALS:-WJetsToLNu_13TeV-madgraphMLM-pythia8 ggHgammagamma ZZ_leptonic ggHbb}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_EVENTS="${MAX_EVENTS:-}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/Trigger/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "${PHASE2_MENU}" "${ERL_MATCH_DIR}" "${COMPARISON_DIR}" "${MINBIAS_TEST}" \
  "${SIG_WORK_DIR}" "${SIGNALS}" "${BATCH_SIZE}" "${MAX_EVENTS}" <<'PY'
import csv
import json
from pathlib import Path
import sys

from collide2v_trigger import TriggerMenu, evaluate_rates, input_files
from collide2v_trigger.evaluation import write_json

phase2_path, match_dir, output_dir, minbias, work_dir = map(Path, sys.argv[1:6])
signals = sys.argv[6].split()
batch_size = int(sys.argv[7])
max_events = int(sys.argv[8]) if sys.argv[8] else None
matching = json.loads((match_dir / "matching.json").read_text())
phase2 = TriggerMenu.load(phase2_path)
erl = TriggerMenu.load(match_dir / "matched_menu.json")
if phase2.to_dict() != matching["phase2_menu"] or erl.to_dict() != matching["matched_menu"]:
    raise ValueError("A supplied menu does not match the rate-matching report; rerun step 08.")
if set(map(str, input_files(minbias))) & set(matching["calibration_inputs"]):
    raise ValueError("Use independent test minbias files, separate from the calibration inputs.")
matches = [entry for entry in matching["matches"] if entry["erl_trigger"] is not None]
if not matches:
    raise ValueError("No fitted ERL working points to compare; check matching.json.")

menu = TriggerMenu.from_dict(phase2.to_dict())
for entry in matches:
    name = entry["erl_trigger"]
    menu.add_trigger(erl.triggers[name])
    menu.thresholds[name] = erl.thresholds_for(name)
    menu.target_rates[name] = erl.target_rate_khz(name)
names = [entry["phase2_trigger"] for entry in matches] + [entry["erl_trigger"] for entry in matches]
datasets = {"minbias": minbias, **{name: work_dir / f"test_{name}.chunks.json" for name in signals}}
samples = {}
for label, inputs in datasets.items():
    result = evaluate_rates(inputs, menu, names, collision_rate_khz=matching["collision_rate_khz"],
                            batch_size=batch_size, max_events=max_events)
    samples[label] = {key: result[key] for key in ("inputs", "event_count", "triggers")}
    print(f"Evaluated {label}: {result['event_count']} events", flush=True)

rows = []
background = samples["minbias"]
for signal in signals:
    sample = samples[signal]
    print(f"\n{signal}: Phase-2 efficiency / ERL efficiency", flush=True)
    for entry in matches:
        phase2_name, erl_name = entry["phase2_trigger"], entry["erl_trigger"]
        left, right = sample["triggers"][phase2_name], sample["triggers"][erl_name]
        left_bg, right_bg = background["triggers"][phase2_name], background["triggers"][erl_name]
        rows.append({
            "signal": signal, "phase2_trigger": phase2_name, "erl_trigger": erl_name,
            "phase2_fit_status": entry["phase2_status"], "erl_fit_status": entry["status"],
            "calibration_rate_matched": entry["rate_matched"],
            "phase2_nominal_target_khz": entry["phase2_nominal_target_khz"],
            "matched_calibration_rate_khz": entry["target_rate_khz"],
            "phase2_test_rate_khz": left_bg["rate_khz"], "erl_test_rate_khz": right_bg["rate_khz"],
            "background_events": background["event_count"],
            "phase2_background_passed": left_bg["passed"], "erl_background_passed": right_bg["passed"],
            "signal_events": sample["event_count"], "phase2_signal_passed": left["passed"],
            "erl_signal_passed": right["passed"],
            "phase2_signal_efficiency": left["efficiency"], "erl_signal_efficiency": right["efficiency"],
            "signal_efficiency_ratio": right["efficiency"] / left["efficiency"] if left["efficiency"] else None})
        print(f"{phase2_name}: {left['efficiency']:.6g} / {right['efficiency']:.6g}", flush=True)

write_json(output_dir / "comparison.json", {
    "format": "collide2v_trigger_comparison", "schema_version": 1,
    "matching_report": str((match_dir / "matching.json").resolve()),
    "source_rule_id": matching["source_rule_id"], "collision_rate_khz": matching["collision_rate_khz"],
    "phase2_menu": phase2.to_dict(), "matched_menu": erl.to_dict(),
    "max_events": max_events, "samples": samples, "comparisons": rows,
    "excluded": [entry for entry in matching["matches"] if entry["erl_trigger"] is None]})
with (output_dir / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {len(rows)} signal comparisons to {output_dir}")
PY
