#!/usr/bin/env bash
# Reference and mixture use disjoint rows of the converted minbias sample.
set -euo pipefail

# Configuration.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/SIGNAL}"
MINBIAS_PT="${SIG_WORK_DIR}/minbias.pt"
SIG_PT="${SIG_WORK_DIR}/SIG.pt"
REFERENCE_COUNT="${REFERENCE_COUNT:-40000}"
BACKGROUND_COUNT="${BACKGROUND_COUNT:-8000}"
SIGNAL_COUNT="${SIGNAL_COUNT:-2000}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"

cd "${REPO_ROOT}"
MINBIAS_PT='minbias'
SIG_PTs="WJetsToLNu_13TeV-madgraphMLM-pythia8 ggHgammagamma ZZ_leptonic ggHbb"
for SIG_PT in $SIG_PTs; do
    "${PYTHON_BIN}" scripts/make_reference_anomalous.py \
      --background "${SIG_WORK_DIR}/train_${MINBIAS_PT}"'*.pt' --signal "${SIG_WORK_DIR}/train_${SIG_PT}"'*.pt' \
      --output-dir "${SIG_WORK_DIR}_train_${SIG_PT}_mixture" \
      --reference-count "${REFERENCE_COUNT}" \
      --background-count "${BACKGROUND_COUNT}" \
      --signal-count "${SIGNAL_COUNT}" --seed "${SAMPLING_SEED}" "$@"
    python scripts/split_train_validation.py \
        --input-dir ${SIG_WORK_DIR}_train_${SIG_PT}_mixture \
        --anomalous-train-fraction 0.5 \
        --reference-train-fraction 0.2
done
REFERENCE_COUNT="${REFERENCE_COUNT:-10000}"
BACKGROUND_COUNT="${BACKGROUND_COUNT:-4000}"
SIGNAL_COUNT="${SIGNAL_COUNT:-1000}"
for SIG_PT in $SIG_PTs; do
    "${PYTHON_BIN}" scripts/make_reference_anomalous.py \
      --background "${SIG_WORK_DIR}/test_${MINBIAS_PT}"'*.pt' --signal "${SIG_WORK_DIR}/test_${SIG_PT}"'*.pt' \
      --output-dir "${SIG_WORK_DIR}_test_${SIG_PT}_mixture" \
      --reference-count "${REFERENCE_COUNT}" \
      --background-count "${BACKGROUND_COUNT}" \
      --signal-count "${SIGNAL_COUNT}" --seed "${SAMPLING_SEED}" "$@"
done
