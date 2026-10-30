#!/usr/bin/env bash
# Run all five steps. A failed step stops the workflow.
set -euo pipefail

# Configuration: steps have their own settings at the top of each executable.
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STEPS=(01_select_SIG_sources.sh 02_convert_minbias_SIG.sh 03_convert_SIG.sh
       04_make_SIG_mixture.sh 05_train_SIG_ERL.sh)

for step in "${STEPS[@]}"; do
  echo "Running ${step}"
  "${RUNNER_DIR}/${step}"
done
