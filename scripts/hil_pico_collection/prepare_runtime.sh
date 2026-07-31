#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${HIL_PICO_RUNTIME:-/mnt/workspace/ys/futuring/openpi_runtime/hil_pico_collection}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${RUNTIME_ROOT}/datasets" "${RUNTIME_ROOT}/logs"
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}/packages/openpi-client"
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}/src/hil_pico_collection"

echo "Prepared HIL PICO runtime at ${RUNTIME_ROOT}"
