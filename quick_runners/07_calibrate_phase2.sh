#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/SIGNAL}"
MINBIAS_TRAIN="${MINBIAS_TRAIN:-${SIG_WORK_DIR}/train_minbias.chunks.json}"
PHASE2_DATABASE="${PHASE2_DATABASE:-${REPO_ROOT}/Trigger/src/collide2v_trigger/menus/phase2_proxy.json}"
PHASE2_CALIBRATION="${PHASE2_CALIBRATION:-${REPO_ROOT}/Trigger/results/phase2_calibration}"
NO_ISOLATION="${NO_ISOLATION:-1}"
COLLISION_RATE_KHZ="${COLLISION_RATE_KHZ:-40000}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_EVENTS="${MAX_EVENTS:-}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/Trigger/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

options=(--menu "${PHASE2_DATABASE}" --output-dir "${PHASE2_CALIBRATION}"
         --collision-rate-khz "${COLLISION_RATE_KHZ}" --batch-size "${BATCH_SIZE}")
if [[ "${NO_ISOLATION}" == 1 ]]; then options+=(--no-isolation); fi
if [[ -n "${MAX_EVENTS}" ]]; then options+=(--max-events "${MAX_EVENTS}"); fi

"${PYTHON_BIN}" -m collide2v_trigger calibrate "${MINBIAS_TRAIN}" "${options[@]}" "$@"
