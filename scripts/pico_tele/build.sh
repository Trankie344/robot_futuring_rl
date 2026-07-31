#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${PICO_TELE_RUNTIME:-/mnt/workspace/ys/futuring/openpi_runtime/pico_tele}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
BUILD_SDK_BRIDGE="${PICO_TELE_BUILD_SDK_BRIDGE:-ON}"

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 ${ROS_DISTRO} is not installed at ${ROS_SETUP}" >&2
  exit 1
fi
if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is not installed" >&2
  exit 1
fi

if [[ "${BUILD_SDK_BRIDGE}" == "ON" ]]; then
  export PICO_ROBOT_SDK_ROOT="${PICO_ROBOT_SDK_ROOT:-${RUNTIME_ROOT}/sdk}"
  if [[ ! -f "${PICO_ROBOT_SDK_ROOT}/include/PXREARobotSDK.h" ]]; then
    echo "PXREA SDK is not prepared at ${PICO_ROBOT_SDK_ROOT}; run prepare_runtime.sh first" >&2
    exit 1
  fi
fi

set +u
source "${ROS_SETUP}"
set -u

mkdir -p "${RUNTIME_ROOT}/build" "${RUNTIME_ROOT}/install" "${RUNTIME_ROOT}/log"
exec colcon --log-base "${RUNTIME_ROOT}/log" build \
  --base-paths "${REPOSITORY_ROOT}/src/pico_tele" \
  --build-base "${RUNTIME_ROOT}/build" \
  --install-base "${RUNTIME_ROOT}/install" \
  --merge-install \
  "$@" \
  --cmake-args "-DPICO_TELE_BUILD_SDK_BRIDGE=${BUILD_SDK_BRIDGE}"
