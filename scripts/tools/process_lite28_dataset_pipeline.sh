#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  process_lite28_dataset_pipeline.sh --source PATH [--source PATH ...] --output PATH [options]

Build the Lite28 LeRobot v2.1 training dataset with the fixed processing flow:
  1. Auto-convert LeRobot3 sources to native LeRobot v2.1 when needed.
  2. Filter by duration and static joints.
  3. Merge sources, set the task prompt, and swap only action gripper columns.
  4. Validate all videos with ffprobe.

Required:
  --source PATH                  Input dataset. May be LeRobot3 or native LeRobot v2.1. Can repeat.
  --output PATH                  Final output LeRobot v2.1 dataset.

Common options:
  --python-bin PATH              Python interpreter. Default: /mnt/workspace/xdm/openpi/.venv/bin/python
  --repo-dir PATH                OpenPI repo. Default: /mnt/workspace/ys/futuring/openpi
  --work-dir PATH                Temporary converted v2.1 sources.
                                 Default: <output parent>/_tmp_<output name>_converted
  --overwrite                    Replace output and converted temp dirs when they already exist.
  --keep-work-dir                Keep temporary converted v2.1 sources after success.

Filtering defaults:
  --min-seconds FLOAT            Default: 30
  --max-seconds FLOAT            Default: 90
  --static-seconds FLOAT         Default: 10
  --static-threshold FLOAT       Default: 0.001
  --metadata-fps FLOAT           Default: 29
  --task TEXT                    Default: fold clothes
  --static-joint-indices LIST    Default: 0,1,2,3,4,5,6,8,9,10,11,12,13,14
  --action-gripper-indices A,B   Default: 7,15
  --no-swap-action-grippers      Keep action unchanged.
  --exclude-episode INDEX        Exclude post-filter episode index. Can repeat.
  --exclude-json PATH            JSON list or bad_episodes/exclude_episodes dict.
  --link-mode MODE               Conversion link mode: auto, hardlink, symlink, copy. Default: auto

Validation defaults:
  --video-workers INT            Default: 16
  --video-timeout-s FLOAT        Default: 20
  --gripper-correlation-threshold FLOAT
                                 Minimum mean abs direct state/action gripper correlation. Default: 0.7
  --gripper-correlation-margin FLOAT
                                 Direct correlation must exceed crossed correlation by this margin. Default: 0.05
  --gripper-correlation-max-lag-frames INT
                                 Search +/- this many frames when computing correlation. Default: 5
  --gripper-correlation-min-std FLOAT
                                 Minimum gripper std required for a meaningful correlation. Default: 0.0001
  --skip-gripper-correlation-validation
                                 Skip state/action gripper correlation validation.

Debug:
  --limit-episodes INT           Build only first N final episodes.
  --skip-video-validation        Skip ffprobe validation.
  -h, --help

Example:
  process_lite28_dataset_pipeline.sh \
    --source /mnt/workspace/ys/futuring/modeltraining/datasets/lite28_day1_lerobot3 \
    --source /mnt/workspace/ys/futuring/modeltraining/datasets/lite28_day2_lerobot3 \
    --output /mnt/workspace/ys/futuring/modeltraining/datasets/lite_28_first_30to90s_static10s_delta16_valid \
    --exclude-episode 631 --exclude-episode 633 --exclude-episode 635 \
    --overwrite
EOF
}

REPO_DIR="${REPO_DIR:-/mnt/workspace/ys/futuring/openpi}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/xdm/openpi/.venv/bin/python}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-/mnt/workspace/ys/futuring/modeltraining/models/acot/scripts/convert_lerobot3_to_native_lerobot.py}"

OUTPUT=""
WORK_DIR=""
OVERWRITE=0
MIN_SECONDS="30"
MAX_SECONDS="90"
STATIC_SECONDS="10"
STATIC_THRESHOLD="0.001"
METADATA_FPS="29"
TASK="fold clothes"
ACTION_GRIPPER_INDICES="7,15"
STATIC_JOINT_INDICES="0,1,2,3,4,5,6,8,9,10,11,12,13,14"
SWAP_ACTION_GRIPPERS=1
LINK_MODE="auto"
VIDEO_WORKERS="16"
VIDEO_TIMEOUT_S="20"
GRIPPER_CORRELATION_THRESHOLD="0.7"
GRIPPER_CORRELATION_MARGIN="0.05"
GRIPPER_CORRELATION_MAX_LAG_FRAMES="5"
GRIPPER_CORRELATION_MIN_STD="0.0001"
LIMIT_EPISODES=""
SKIP_VIDEO_VALIDATION=0
SKIP_GRIPPER_CORRELATION_VALIDATION=0
KEEP_WORK_DIR=0

SOURCES=()
EXCLUDE_EPISODES=()
EXCLUDE_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCES+=("${2:?missing value for --source}")
      shift 2
      ;;
    --output)
      OUTPUT="${2:?missing value for --output}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:?missing value for --python-bin}"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="${2:?missing value for --repo-dir}"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="${2:?missing value for --work-dir}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --min-seconds)
      MIN_SECONDS="${2:?missing value for --min-seconds}"
      shift 2
      ;;
    --max-seconds)
      MAX_SECONDS="${2:?missing value for --max-seconds}"
      shift 2
      ;;
    --static-seconds)
      STATIC_SECONDS="${2:?missing value for --static-seconds}"
      shift 2
      ;;
    --static-threshold)
      STATIC_THRESHOLD="${2:?missing value for --static-threshold}"
      shift 2
      ;;
    --metadata-fps)
      METADATA_FPS="${2:?missing value for --metadata-fps}"
      shift 2
      ;;
    --task)
      TASK="${2:?missing value for --task}"
      shift 2
      ;;
    --action-gripper-indices)
      ACTION_GRIPPER_INDICES="${2:?missing value for --action-gripper-indices}"
      shift 2
      ;;
    --static-joint-indices)
      STATIC_JOINT_INDICES="${2:?missing value for --static-joint-indices}"
      shift 2
      ;;
    --no-swap-action-grippers)
      SWAP_ACTION_GRIPPERS=0
      shift
      ;;
    --exclude-episode)
      EXCLUDE_EPISODES+=("${2:?missing value for --exclude-episode}")
      shift 2
      ;;
    --exclude-json)
      EXCLUDE_JSON="${2:?missing value for --exclude-json}"
      shift 2
      ;;
    --link-mode)
      LINK_MODE="${2:?missing value for --link-mode}"
      shift 2
      ;;
    --video-workers)
      VIDEO_WORKERS="${2:?missing value for --video-workers}"
      shift 2
      ;;
    --video-timeout-s)
      VIDEO_TIMEOUT_S="${2:?missing value for --video-timeout-s}"
      shift 2
      ;;
    --gripper-correlation-threshold)
      GRIPPER_CORRELATION_THRESHOLD="${2:?missing value for --gripper-correlation-threshold}"
      shift 2
      ;;
    --gripper-correlation-margin)
      GRIPPER_CORRELATION_MARGIN="${2:?missing value for --gripper-correlation-margin}"
      shift 2
      ;;
    --gripper-correlation-max-lag-frames)
      GRIPPER_CORRELATION_MAX_LAG_FRAMES="${2:?missing value for --gripper-correlation-max-lag-frames}"
      shift 2
      ;;
    --gripper-correlation-min-std)
      GRIPPER_CORRELATION_MIN_STD="${2:?missing value for --gripper-correlation-min-std}"
      shift 2
      ;;
    --limit-episodes)
      LIMIT_EPISODES="${2:?missing value for --limit-episodes}"
      shift 2
      ;;
    --skip-video-validation)
      SKIP_VIDEO_VALIDATION=1
      shift
      ;;
    --skip-gripper-correlation-validation)
      SKIP_GRIPPER_CORRELATION_VALIDATION=1
      shift
      ;;
    --keep-work-dir)
      KEEP_WORK_DIR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "At least one --source is required." >&2
  exit 2
fi
if [[ -z "$OUTPUT" ]]; then
  echo "--output is required." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python env not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
  echo "Convert script not found: $CONVERT_SCRIPT" >&2
  exit 1
fi

OUTPUT="$(realpath -m "$OUTPUT")"
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(dirname "$OUTPUT")/_tmp_$(basename "$OUTPUT")_converted"
else
  WORK_DIR="$(realpath -m "$WORK_DIR")"
fi

BUILD_SCRIPT="$REPO_DIR/scripts/tools/build_lite28_lerobot_dataset.py"
VIDEO_VALIDATE_SCRIPT="$REPO_DIR/scripts/tools/validate_lerobot_videos.py"
GRIPPER_CORRELATION_VALIDATE_SCRIPT="$REPO_DIR/scripts/tools/validate_state_action_gripper_correlation.py"
VALIDATION_DIR="$OUTPUT/validation"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$WORK_DIR" "$LOG_DIR"

if [[ ! -f "$BUILD_SCRIPT" ]]; then
  echo "Build script not found: $BUILD_SCRIPT" >&2
  exit 1
fi
if [[ $SKIP_VIDEO_VALIDATION -eq 0 && ! -f "$VIDEO_VALIDATE_SCRIPT" ]]; then
  echo "Video validation script not found: $VIDEO_VALIDATE_SCRIPT" >&2
  exit 1
fi
if [[ $SKIP_GRIPPER_CORRELATION_VALIDATION -eq 0 && ! -f "$GRIPPER_CORRELATION_VALIDATE_SCRIPT" ]]; then
  echo "Gripper correlation validation script not found: $GRIPPER_CORRELATION_VALIDATE_SCRIPT" >&2
  exit 1
fi
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CONVERTED_SOURCES=()
CONVERT_LOG="$LOG_DIR/process_lite28_convert_$(date +%Y%m%d_%H%M%S).log"

echo "=== process_lite28_dataset_pipeline ==="
echo "output=$OUTPUT"
echo "work_dir=$WORK_DIR"
echo "python=$PYTHON_BIN"
echo "convert_log=$CONVERT_LOG"

for source in "${SOURCES[@]}"; do
  source="$(realpath -m "$source")"
  if [[ ! -d "$source" ]]; then
    echo "Source directory not found: $source" >&2
    exit 1
  fi

  if [[ -f "$source/meta/episodes.jsonl" && -f "$source/meta/info.json" ]]; then
    echo "Using existing LeRobot v2.1 source: $source"
    CONVERTED_SOURCES+=("$source")
    continue
  fi

  if [[ -f "$source/meta/tasks.parquet" && -d "$source/meta/episodes" && -f "$source/meta/info.json" ]]; then
    converted="$WORK_DIR/$(basename "$source")_lerobot_v21"
    echo "Converting LeRobot3 source: $source -> $converted"
    convert_args=(
      "$CONVERT_SCRIPT"
      --source "$source"
      --output "$converted"
      --link-mode "$LINK_MODE"
      --codebase-version v2.1
    )
    if [[ $OVERWRITE -eq 1 ]]; then
      convert_args+=(--overwrite)
    fi
    "$PYTHON_BIN" "${convert_args[@]}" 2>&1 | tee -a "$CONVERT_LOG"
    CONVERTED_SOURCES+=("$converted")
    continue
  fi

  echo "Unsupported source format: $source" >&2
  echo "Expected either meta/episodes.jsonl for v2.1 or meta/tasks.parquet plus meta/episodes/ for LeRobot3." >&2
  exit 1
done

build_args=(
  "$BUILD_SCRIPT"
  --output "$OUTPUT"
  --min-seconds "$MIN_SECONDS"
  --max-seconds "$MAX_SECONDS"
  --static-seconds "$STATIC_SECONDS"
  --static-threshold "$STATIC_THRESHOLD"
  --static-joint-indices "$STATIC_JOINT_INDICES"
  --metadata-fps "$METADATA_FPS"
  --task "$TASK"
  --action-gripper-indices "$ACTION_GRIPPER_INDICES"
)
for source in "${CONVERTED_SOURCES[@]}"; do
  build_args+=(--source "$source")
done
for episode in "${EXCLUDE_EPISODES[@]}"; do
  build_args+=(--exclude-episode "$episode")
done
if [[ -n "$EXCLUDE_JSON" ]]; then
  build_args+=(--exclude-json "$EXCLUDE_JSON")
fi
if [[ $OVERWRITE -eq 1 ]]; then
  build_args+=(--overwrite)
fi
if [[ $SWAP_ACTION_GRIPPERS -eq 0 ]]; then
  build_args+=(--no-swap-action-grippers)
fi
if [[ -n "$LIMIT_EPISODES" ]]; then
  build_args+=(--limit-episodes "$LIMIT_EPISODES")
fi

BUILD_LOG="$LOG_DIR/process_lite28_build_$(date +%Y%m%d_%H%M%S).log"
echo "Building filtered dataset. build_log=$BUILD_LOG"
"$PYTHON_BIN" "${build_args[@]}" 2>&1 | tee "$BUILD_LOG"

if [[ $SKIP_GRIPPER_CORRELATION_VALIDATION -eq 0 ]]; then
  mkdir -p "$VALIDATION_DIR"
  GRIPPER_CORRELATION_REPORT="$VALIDATION_DIR/state_action_gripper_correlation.json"
  GRIPPER_CORRELATION_LOG="$LOG_DIR/process_lite28_gripper_correlation_$(date +%Y%m%d_%H%M%S).log"
  echo "Validating state/action gripper correlation. gripper_correlation_log=$GRIPPER_CORRELATION_LOG"
  "$PYTHON_BIN" "$GRIPPER_CORRELATION_VALIDATE_SCRIPT" \
    --dataset "$OUTPUT" \
    --output-json "$GRIPPER_CORRELATION_REPORT" \
    --state-gripper-indices "$ACTION_GRIPPER_INDICES" \
    --action-gripper-indices "$ACTION_GRIPPER_INDICES" \
    --threshold "$GRIPPER_CORRELATION_THRESHOLD" \
    --margin "$GRIPPER_CORRELATION_MARGIN" \
    --max-lag-frames "$GRIPPER_CORRELATION_MAX_LAG_FRAMES" \
    --min-std "$GRIPPER_CORRELATION_MIN_STD" 2>&1 | tee "$GRIPPER_CORRELATION_LOG"
fi

if [[ $SKIP_VIDEO_VALIDATION -eq 0 ]]; then
  VIDEO_VALIDATION_DIR="$VALIDATION_DIR/videos"
  mkdir -p "$VIDEO_VALIDATION_DIR"
  VIDEO_LOG="$LOG_DIR/process_lite28_video_validation_$(date +%Y%m%d_%H%M%S).log"
  echo "Validating videos with ffprobe. video_log=$VIDEO_LOG"
  "$PYTHON_BIN" "$VIDEO_VALIDATE_SCRIPT" \
    --dataset "$OUTPUT" \
    --output-dir "$VIDEO_VALIDATION_DIR" \
    --workers "$VIDEO_WORKERS" \
    --timeout-s "$VIDEO_TIMEOUT_S" 2>&1 | tee "$VIDEO_LOG"
fi

if [[ $KEEP_WORK_DIR -eq 0 ]]; then
  case "$WORK_DIR" in
    ""|"/"|"/mnt"|"/mnt/workspace"|"/mnt/workspace/ys"|"/mnt/workspace/ys/futuring"|"/mnt/workspace/ys/futuring/modeltraining"|"/mnt/workspace/ys/futuring/modeltraining/datasets")
      echo "Refusing to clean unsafe work_dir: $WORK_DIR" >&2
      exit 1
      ;;
    *)
      if [[ -d "$WORK_DIR" ]]; then
        echo "Cleaning temporary work_dir: $WORK_DIR"
        rm -rf "$WORK_DIR"
      fi
      ;;
  esac
fi

echo "Dataset processing completed successfully."
echo "output=$OUTPUT"
echo "validation_dir=$VALIDATION_DIR"
