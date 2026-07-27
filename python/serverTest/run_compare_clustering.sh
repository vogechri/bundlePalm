#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"
export PYTHONPATH="${script_dir}/build/generated/proto${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -x "${script_dir}/.venv/bin/python" ]]; then
    python_bin="${script_dir}/.venv/bin/python"
else
    python_bin=${BUNDLE_PALM_PYTHON:-python3}
fi

base_url=${1:-http://grail.cs.washington.edu/projects/bal/data/ladybug/}
problem=${2:-problem-49-7776-pre.txt.bz2}
iterations=${3:-30}
clusters=${4:-10}

for mode in landmark landmark_clean; do
    echo "Running ${mode} clustering"
    BUNDLE_PALM_CLUSTERING=${mode} "${python_bin}" -u client_acc.py \
        "${base_url}" "${problem}" "${iterations}" "${clusters}" \
        > "compare_${mode}.log" 2>&1
done

grep -E "clustering took|Residual balance|Weak camera-cluster|Additional camera copies|DRE BFGS|f\(v\)=" \
    compare_landmark.log compare_landmark_clean.log