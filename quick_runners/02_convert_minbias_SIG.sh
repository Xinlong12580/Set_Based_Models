#!/usr/bin/env bash
# Make a separate minbias tensor for this workflow, with event/object/pair inputs.
set -euo pipefail

# Configuration (keep paths/feature settings consistent with the other steps).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/SIGNAL}"
SIG_SOURCE_DIR="${SIG_WORK_DIR}/sources"
MINBIAS_PT="${SIG_WORK_DIR}/minbias.pt"
MAX_EVENTS="${MAX_EVENTS:-100000}" # Total events, not events per file.

cd "${REPO_ROOT}"

mapfile -t sources < "${SIG_SOURCE_DIR}/train_minbias.txt"

"${PYTHON_BIN}" -m raw_event_conversion \
  --input "${sources[@]}" \
  --output "${SIG_WORK_DIR}/train_minbias.pt" \
  --label 1 --max-events "${MAX_EVENTS}" --with-pair-tensor --max-events-per-file 1000 "$@"


mapfile -t sources < "${SIG_SOURCE_DIR}/test_minbias.txt"

"${PYTHON_BIN}" -m raw_event_conversion \
  --input "${sources[@]}" \
  --output "${SIG_WORK_DIR}/test_minbias.pt"  \
  --label 1 --max-events "${MAX_EVENTS}" --with-pair-tensor --max-events-per-file 1000 "$@"
