#!/usr/bin/env bash
# List EOS, filter the production version, and pin the ordered source URLs.
set -euo pipefail

# Configuration (environment variables can override these defaults).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1
EOS_REDIRECTOR="${EOS_REDIRECTOR:-root://eosproject.cern.ch}"
EOS_SAMPLE_DIR="${EOS_SAMPLE_DIR:-/eos/project/f/foundational-model-dataset/samples/production_final}"
SIG_PROCESSES="${SIG_PROCESSES:-WJetsToLNu_13TeV-madgraphMLM-pythia8;ggHgammagamma;ZZ_leptonic;ggHbb}"
SIG_DATASET_VERSION="${SIG_DATASET_VERSION:-collide2v_v1.0}"
MAX_SOURCE_FILES="${MAX_SOURCE_FILES:-400}" # First N matching files; 0 means all.
SIG_WORK_DIR="${SIG_WORK_DIR:-${REPO_ROOT}/data/SIGNAL/}"
SIG_SOURCE_DIR="${SIG_WORK_DIR}/sources"

mkdir -p "${SIG_SOURCE_DIR}"

"${PYTHON_BIN}" - "${EOS_REDIRECTOR}" "${EOS_SAMPLE_DIR}" "${SIG_PROCESSES}" \
  "${SIG_DATASET_VERSION}" "${MAX_SOURCE_FILES}" "${SIG_SOURCE_DIR}" <<'PY'
import sys
from pathlib import Path, PurePosixPath

import fsspec
import pyarrow.parquet as pq

redirector, sample_dir, sig_processes, version, limit, output_dir = sys.argv[1:]
limit = int(limit)
_processes = sig_processes.split(";")
processes = ()
for process in _processes:
    processes = ( (process, f"{process}.txt"), *processes)
for sample, manifest in (("minbias", "minbias.txt"), *processes ):
    directory_url = f"{redirector.rstrip('/')}//{sample_dir.strip('/')}/{sample}"
    fs, directory = fsspec.core.url_to_fs(directory_url)
    # Listings may contain bare names, full paths, or hidden EOS pseudo-files.
    names = sorted({PurePosixPath(name).name for name in fs.ls(directory, detail=False)
                    if name.endswith(".parquet") and not PurePosixPath(name).name.startswith(".")})
    selected = []
    for name in names:
        url = f"{directory_url}/{name}"
        source_fs, source_path = fsspec.core.url_to_fs(url)
        with source_fs.open(source_path, "rb") as handle:
            metadata = pq.read_metadata(handle).metadata or {}
        if metadata.get(b"dataset_version") != version.encode():
            print(f"Skipping {url}: dataset_version does not match {version}", file=sys.stderr)
            continue
        print(f"Added {url}")
        selected.append(url)
        if limit > 0 and len(selected) >= limit:
            break
    if not selected:
        raise SystemExit(f"No {version} Parquet files found under {directory_url}")
    target_train = Path(output_dir) / ( "train_" + manifest)
    target_test = Path(output_dir) / ("test_" + manifest)
    if target_train.exists():
        print(f"Warning: overwriting source manifest {target_train}", file=sys.stderr)
    if target_test.exists():
        print(f"Warning: overwriting source manifest {target_test}", file=sys.stderr)
    selected_train = [f for f in selected if int(f[f.find("RS")+2: f.find(".parquet")]) % 2 == 0] 
    selected_test = [f for f in selected if int(f[f.find("RS")+2: f.find(".parquet")]) % 2 == 1] 
    target_train.write_text("\n".join(selected_train) + "\n", encoding="utf-8")
    target_test.write_text("\n".join(selected_test) + "\n", encoding="utf-8")
    print(f"Saved {len(selected_train)} ordered {sample} source URLs to {target_train}")
    print(f"Saved {len(selected_test)} ordered {sample} source URLs to {target_test}")
    for url in selected:
        print(f"  {url}")
    
PY
