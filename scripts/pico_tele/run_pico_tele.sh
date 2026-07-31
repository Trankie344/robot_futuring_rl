#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${PICO_TELE_RUNTIME:-/mnt/workspace/ys/futuring/openpi_runtime/pico_tele}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WORKSPACE_SETUP="${RUNTIME_ROOT}/install/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 ${ROS_DISTRO} is not installed at ${ROS_SETUP}" >&2
  exit 1
fi
if [[ ! -f "${WORKSPACE_SETUP}" ]]; then
  echo "PICO ROS 2 workspace is not built at ${WORKSPACE_SETUP}; run build.sh first" >&2
  exit 1
fi

export PICO_ROBOT_SDK_ROOT="${PICO_ROBOT_SDK_ROOT:-${RUNTIME_ROOT}/sdk}"
case "$(uname -m)" in
  x86_64|amd64)
    SDK_LIBRARY_DIR="${PICO_ROBOT_SDK_ROOT}/linux/64"
    ;;
  aarch64|arm64)
    SDK_LIBRARY_DIR="${PICO_ROBOT_SDK_ROOT}/linux_aarch64/64"
    ;;
  *)
    echo "Unsupported host architecture: $(uname -m)" >&2
    exit 1
    ;;
esac
if [[ ! -f "${SDK_LIBRARY_DIR}/libPXREARobotSDK.so" ]]; then
  echo "PXREA SDK library is missing from ${SDK_LIBRARY_DIR}" >&2
  exit 1
fi

set +u
source "${ROS_SETUP}"
source "${WORKSPACE_SETUP}"
set -u

export LD_LIBRARY_PATH="${SDK_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

LAUNCH_ARGS=()
if [[ -n "${PICO_DEVICE_ID:-}" ]]; then
  LAUNCH_ARGS+=("device_id:=${PICO_DEVICE_ID}")
fi
if [[ -n "${PICO_TELE_PARAMS_FILE:-}" ]]; then
  LAUNCH_ARGS+=("params_file:=${PICO_TELE_PARAMS_FILE}")
fi

exec ros2 launch pico_tele_bridge pico_tele.launch.py "${LAUNCH_ARGS[@]}" "$@"
