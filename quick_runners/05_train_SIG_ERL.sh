#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/}"
cd "${REPO_ROOT}"

SIGs="WJetsToLNu_13TeV-madgraphMLM-pythia8 ggHgammagamma ZZ_leptonic ggHbb"
SIGs="ggHgammagamma"


for SIG in $SIGs; do
    python casual_tests/scan_ERL.py \
      --signal "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/anomalous_train.pt" \
      --background "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_train.pt" \
      --event-only \
      --reference-penalty 0 --validation-reference-penalty 0 \
      --beam-novelty-weight 0 0.01 0.03 0.05 0.07 0.1 \
      --beam-similarity-filter \
      --beam-signal-similarity-threshold 0.1 0.2 0.5 0.9 0.95 \
      --beam-background-similarity-threshold 0 \
      --final-similarity-filter \
      --final-signal-similarity-threshold 0.9 \
      --final-background-similarity-threshold 0.9 \
      --stricter-score-improvement 0 \
      --max-trigger-rate-khz 400 --input-rate-mhz 40 \
      --output-prefix training_outputs/ggHgammagamma_scan \
      --self-prune \
      --self-prune-score-improvement 0 \
      --overwrite-output --progress \
      --score excess --score-delta 0.001
      #--overwrite-output --progress --validate-only \
done
exit
for SIG in $SIGs; do
    python casual_tests/train_ERL_validation.py \
      --signal "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/anomalous_train.pt" \
      --background "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_train.pt" \
      --event-only \
      --reference-penalty 0 \
      --stricter-score-improvement 0 \
      --beam-similarity-filter --final-similarity-filter --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
      --max-trigger-rate-khz 40 --input-rate-mhz 40 \
      --output-prefix training_outputs/${SIG}_event --overwrite-output --progress \
      --beam-novelty-weight 0.03
      2>&1 | tee "${SIG}_event.log"

done
exit

for SIG in $SIGs; do
    python casual_tests/train_ERL_validation.py \
      --signal "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/anomalous_train.pt" \
      --background "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_train.pt" \
      --pair-tensor --object-only \
      --object-cut-priority 'type_id_1' 'type_id_2' 'pair_mass:>' 'pair_mass:<=' --require-complete-priority \
      --reference-penalty 0 \
      --stricter-score-improvement 0 \
      --beam-similarity-filter --final-similarity-filter --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
      --max-trigger-rate-khz 40 --input-rate-mhz 40 \
      --output-prefix training_outputs/${SIG}_resonance --overwrite-output --progress \
      2>&1 | tee "${SIG}_resonance.log"
    
    python casual_tests/train_ERL_validation.py \
      --signal "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/anomalous_train.pt" \
      --background "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_train.pt" \
      --pair-tensor --object-only \
      --object-cut-priority 'type_id_1' 'type_id_2' --require-complete-priority \
      --reference-penalty 0 \
      --stricter-score-improvement 0 \
      --beam-similarity-filter --final-similarity-filter --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
      --max-trigger-rate-khz 40 --input-rate-mhz 40 \
      --output-prefix training_outputs/${SIG}_pair --overwrite-output --progress \
      2>&1 | tee "${SIG}_pair.log"

    python casual_tests/train_ERL_validation.py \
      --signal "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/anomalous_train.pt" \
      --background "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_train.pt" \
      --object-only \
      --object-cut-priority 'type_id' --require-complete-priority \
      --reference-penalty 0 \
      --stricter-score-improvement 0 \
      --beam-similarity-filter --final-similarity-filter --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
      --max-trigger-rate-khz 40 --input-rate-mhz 40 \
      --output-prefix training_outputs/${SIG}_object --overwrite-output --progress \
      2>&1 | tee "${SIG}_object.log"
    
    python casual_tests/train_ERL_validation.py \
      --signal "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/anomalous_train.pt" \
      --background "${SIG_WORK_DIR}/SIGNAL_train_${SIG}_mixture/reference_train.pt" \
      --event-only \
      --reference-penalty 0 \
      --stricter-score-improvement 0 \
      --beam-similarity-filter --final-similarity-filter --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
      --max-trigger-rate-khz 40 --input-rate-mhz 40 \
      --output-prefix training_outputs/${SIG}_event --overwrite-output --progress \
      2>&1 | tee "${SIG}_event.log"
done
