#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export HIL_PICO_RUNTIME="${HIL_PICO_RUNTIME:-/mnt/workspace/ys/futuring/openpi_runtime/hil_pico_collection}"
export PYTHONPATH="${REPO_ROOT}/src/hil_pico_collection:${REPO_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN:-python3}" -m hil_pico_collection.ros.recorder_node "$@"
