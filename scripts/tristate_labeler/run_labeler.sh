#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Accept either an explicit ``--dataset PATH`` option or the dataset path as
# the first positional argument for convenience.
if [[ $# -gt 0 && "${1}" != -* ]]; then
    DATASET_PATH="${1}"
    shift
    set -- --dataset "${DATASET_PATH}" "$@"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_labeler.py" serve "$@"
