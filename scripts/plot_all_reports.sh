#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_dir="${1:-${repo_root}/reports}"
python_bin="${PYTHON_BIN:-python}"
plot_cache_dir="${TMPDIR:-/tmp}/set_based_models_matplotlib"

if [[ ! -d "${report_dir}" ]]; then
    echo "Report directory does not exist: ${report_dir}" >&2
    exit 1
fi

mkdir -p "${plot_cache_dir}"
shopt -s nullglob
report_files=("${report_dir}"/*.txt)

if (( ${#report_files[@]} == 0 )); then
    echo "No .txt reports found under ${report_dir}" >&2
    exit 1
fi

echo "Found ${#report_files[@]} reports under ${report_dir}"
for report_file in "${report_files[@]}"; do
    output_file="${report_file%.txt}_acceptance.png"
    echo "Plotting $(basename -- "${report_file}") -> $(basename -- "${output_file}")"
    MPLCONFIGDIR="${plot_cache_dir}" "${python_bin}" \
        "${script_dir}/plot_acceptance_report.py" \
        "${report_file}" \
        --output "${output_file}" \
        --overwrite
done

echo "Created ${#report_files[@]} acceptance plots."
