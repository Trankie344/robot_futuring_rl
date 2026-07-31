#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/workspace/ys/futuring/openpi}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/xdm/openpi/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_lite28_first_40to90s_delta16_30k_from_lite0030_20000}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/mnt/workspace/ys/futuring/modeltraining/models/acot/env/wandb_api.key}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
WANDB_MODE="${WANDB_MODE:-online}"

cd "$REPO_DIR"
mkdir -p logs

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-lite28_first_30to90s_static10s_delta16_valid_pi05_from_lite0030_20000_bs256_fsdp${FSDP_DEVICES}_30k_${TS}}"
LOG_FILE="${LOG_FILE:-logs/${EXP_NAME}.log}"
SCREEN_NAME="${SCREEN_NAME:-openpi_lite28_delta16_fsdp${FSDP_DEVICES}_${TS}}"
RUN_SCRIPT="${RUN_SCRIPT:-logs/${EXP_NAME}.run.sh}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python env not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
  echo "wandb api key file not found: $WANDB_API_KEY_FILE" >&2
  exit 1
fi

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export WANDB_API_KEY=\$(tr -d '\r\n' < "$WANDB_API_KEY_FILE")
export WANDB_MODE="$WANDB_MODE"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION"
export NCCL_DEBUG=WARN
{
  echo "=== train_vla_lite28_delta16_30k ==="
  echo "date=\$(date -Is)"
  echo "config=$CONFIG_NAME"
  echo "exp_name=$EXP_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "fsdp_devices=$FSDP_DEVICES"
  echo "num_workers=${NUM_WORKERS:-config_default}"
  echo "xla_preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE"
  echo "xla_mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION"
  echo "python=$PYTHON_BIN"
  echo
} >> "$LOG_FILE"
train_args=(
  scripts/train/train.py
  "$CONFIG_NAME"
  --exp-name="$EXP_NAME"
  --fsdp-devices="$FSDP_DEVICES"
)
if [[ -n "$NUM_WORKERS" ]]; then
  train_args+=(--num-workers="$NUM_WORKERS")
fi
exec "$PYTHON_BIN" "\${train_args[@]}" >> "$LOG_FILE" 2>&1
EOF
chmod +x "$RUN_SCRIPT"

screen -dmS "$SCREEN_NAME" bash "$RUN_SCRIPT"

echo "SESSION=$SCREEN_NAME"
echo "LOG=$REPO_DIR/$LOG_FILE"
echo "RUN_SCRIPT=$REPO_DIR/$RUN_SCRIPT"
echo "EXP=$EXP_NAME"
echo "CONFIG=$CONFIG_NAME"
