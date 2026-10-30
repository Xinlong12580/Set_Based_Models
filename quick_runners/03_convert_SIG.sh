#!/usr/bin/env bash
# Convert the signal with the same feature settings as the minbias conversion.
set -euo pipefail

# Configuration (keep paths/feature settings consistent with the other steps).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/SIGNAL}"
SIG_SOURCE_DIR="${SIG_WORK_DIR}/sources"
MAX_EVENTS="${MAX_EVENTS:-4000}" # Total events, not events per file.

cd "${REPO_ROOT}"
source_files="WJetsToLNu_13TeV-madgraphMLM-pythia8.txt ggHgammagamma.txt ZZ_leptonic.txt ggHbb.txt"
for source_file in $source_files; do

    mapfile -t sources < "${SIG_SOURCE_DIR}/train_$source_file"
    "${PYTHON_BIN}" -m raw_event_conversion \
      --input "${sources[@]}" \
      --output $SIG_WORK_DIR/train_"${source_file%.txt}.pt" \
      --label 2 --max-events "${MAX_EVENTS}" --with-pair-tensor --max-events-per-file 400 "$@"

    mapfile -t sources < "${SIG_SOURCE_DIR}/test_$source_file"
    "${PYTHON_BIN}" -m raw_event_conversion \
      --input "${sources[@]}" \
      --output $SIG_WORK_DIR/test_"${source_file%.txt}.pt" \
      --label 2 --max-events "${MAX_EVENTS}" --with-pair-tensor --max-events-per-file 400 "$@"
done
