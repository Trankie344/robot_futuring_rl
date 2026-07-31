#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT=""
PROMPT=""
POLICY_HOST="127.0.0.1"
POLICY_PORT="8011"
ROBOT_ADAPTER=""
ROBOT_CONFIG="${SCRIPT_DIR}/../../src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --policy-host)
      POLICY_HOST="$2"
      shift 2
      ;;
    --policy-port)
      POLICY_PORT="$2"
      shift 2
      ;;
    --robot-adapter)
      ROBOT_ADAPTER="$2"
      shift 2
      ;;
    --robot-config)
      ROBOT_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PROMPT}" ]]; then
  echo "--prompt is required" >&2
  exit 2
fi

recorder_args=(--robot-config "${ROBOT_CONFIG}")
adapter_args=()
if [[ -n "${ROBOT_ADAPTER}" ]]; then
  adapter_args=(--robot-adapter "${ROBOT_ADAPTER}")
  recorder_args+=("${adapter_args[@]}")
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  recorder_args+=(--dataset-root "${DATASET_ROOT}")
fi

"${SCRIPT_DIR}/run_recorder.sh" "${recorder_args[@]}" &
recorder_pid=$!
"${SCRIPT_DIR}/run_mode_switcher.sh" --robot-config "${ROBOT_CONFIG}" "${adapter_args[@]}" &
switcher_pid=$!
"${SCRIPT_DIR}/run_rl_token_bridge.sh" --host "${POLICY_HOST}" --port "${POLICY_PORT}" --prompt "${PROMPT}" --robot-config "${ROBOT_CONFIG}" "${adapter_args[@]}" &
bridge_pid=$!

cleanup() {
  kill "${recorder_pid}" "${switcher_pid}" "${bridge_pid}" 2>/dev/null || true
  wait "${recorder_pid}" "${switcher_pid}" "${bridge_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait -n "${recorder_pid}" "${switcher_pid}" "${bridge_pid}"
