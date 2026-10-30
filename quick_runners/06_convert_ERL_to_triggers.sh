#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIG="${SIG:-ggHgammagamma}"
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data}"
ERL_MODEL="${ERL_MODEL:-${REPO_ROOT}/training_outputs/${SIG}_scan.validation.model.json}"
REFERENCE_PT="${REFERENCE_PT:-${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_validation.pt}"
MENU_OUTPUT="${MENU_OUTPUT:-${REPO_ROOT}/Trigger/results/${SIG}_menu.json}"
RULE_IDS="${RULE_IDS:-}"
TARGET_RATE_KHZ="${TARGET_RATE_KHZ:-}"
SKIP_UNSUPPORTED="${SKIP_UNSUPPORTED:-0}"
ALLOW_FIXED="${ALLOW_FIXED:-1}"
BASE_MENU="${BASE_MENU:-}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/Trigger/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "${ERL_MODEL}" "${REFERENCE_PT}" "${MENU_OUTPUT}" "${RULE_IDS}" "${TARGET_RATE_KHZ}" "${SKIP_UNSUPPORTED}" "${ALLOW_FIXED}" "${BASE_MENU}" <<'PY'
import sys

from collide2v_trigger import TriggerMenu, iter_batches, parse_erl_model

model_path, reference_path, output_path, rule_ids, target_rate, skip_unsupported, allow_fixed, base_menu = sys.argv[1:]
ids = [int(value) for value in rule_ids.split()] if rule_ids else None
rate = float(target_rate) if target_rate else None
reference = next(iter_batches(reference_path, with_pairs=True))
print(f"Converting retained rules from {model_path}" if ids is None else f"Converting rules {ids} from {model_path}", flush=True)
triggers = parse_erl_model(model_path, reference, rule_ids=ids, target_rate_khz=rate,
                           skip_unsupported=skip_unsupported == "1", allow_fixed=allow_fixed == "1")
menu = TriggerMenu.load(base_menu) if base_menu else TriggerMenu()
for trigger in triggers:
    menu.add_trigger(trigger)
menu.save(output_path)
tunable = sum(trigger.tunable for trigger in triggers)
print(f"Converted {len(triggers)} ERL rules: {tunable} tunable, {len(triggers) - tunable} fixed. Saved menu: {output_path}")
PY
