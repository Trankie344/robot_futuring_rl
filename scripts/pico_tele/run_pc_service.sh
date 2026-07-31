#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${PICO_TELE_RUNTIME:-/mnt/workspace/ys/futuring/openpi_runtime/pico_tele}"
if [[ -x "/opt/apps/roboticsservice/RoboticsServiceProcess" ]]; then
  SERVICE_ROOT="/opt/apps/roboticsservice"
elif [[ -x "${RUNTIME_ROOT}/pc_service_root/opt/apps/roboticsservice/RoboticsServiceProcess" ]]; then
  SERVICE_ROOT="${RUNTIME_ROOT}/pc_service_root/opt/apps/roboticsservice"
else
  echo "XRoboToolkit PC Service is not installed or prepared for this host" >&2
  exit 1
fi

if pgrep -x RoboticsServiceProcess >/dev/null 2>&1; then
  echo "XRoboToolkit PC Service is already running"
  exit 0
fi

SERVICE_SDK_LIBRARY_DIR="$(find "${SERVICE_ROOT}/SDK" -type f -name 'libPXREARobotSDK.so' -printf '%h\n' -quit 2>/dev/null || true)"
export LD_LIBRARY_PATH="${SERVICE_ROOT}:${SERVICE_ROOT}/lib${SERVICE_SDK_LIBRARY_DIR:+:${SERVICE_SDK_LIBRARY_DIR}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QT_PLUGIN_PATH="${SERVICE_ROOT}/plugins${QT_PLUGIN_PATH:+:${QT_PLUGIN_PATH}}"
export QT_QML_PATH="${SERVICE_ROOT}/qml${QT_QML_PATH:+:${QT_QML_PATH}}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
cd "${SERVICE_ROOT}"
exec "${SERVICE_ROOT}/RoboticsServiceProcess" "$@"
