#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EXP_NAME="${EXP_NAME:-arm_value_hil_pico_v21}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TORCHRUN="${TORCHRUN:-${REPO_ROOT}/../modeltraining/models/acot/env/.venv/bin/torchrun}"

if [[ ! -x "${TORCHRUN}" ]]; then
  echo "torchrun is not executable: ${TORCHRUN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE}" \
  scripts/train/train_arm_value.py \
  arm_value_hil_pico_v21 \
  --exp-name="${EXP_NAME}" \
  "$@"
