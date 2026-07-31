#!/usr/bin/env bash
set -euo pipefail

python scripts/train/rl_token/train_stage1.py rl_token_stage1 "$@"
