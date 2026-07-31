# RL Token Stage 2 manual workflow

Stage 2 uses 30 Hz episodes with labels `-1`, `0`, `1`, and `2`. Label `2` is optional, may occur at most once, and only on the final frame. A final frame without `2` is still terminal but unsuccessful.

Transitions use horizon 20, stride 2, and forced tail alignment. Reward is the undiscounted sum of the 20 frame labels; gamma applies only between action chunks.

Build one immutable cache shard:

```bash
python scripts/tools/rl_token/build_stage2_cache.py \
  rl_token_stage2 \
  --checkpoint checkpoints/rl_token/pi05_lite0030_rltoken_only/54999 \
  --batch <ready-batch> \
  --training-root /mnt/workspace/ys/futuring/openpi_runtime/rl_token_stage2 \
  --round-id round_000001
```

Append it to replay:

```bash
python scripts/tools/rl_token/publish_replay.py \
  --cache-shard <cache-shard> \
  --admission <admission.json> \
  --output <replay.json> \
  [--previous <previous-replay.json>]
```

Train the round:

```bash
python scripts/train/rl_token/train_stage2.py \
  rl_token_stage2 \
  --checkpoint-dir <round-checkpoints> \
  --round-id round_000001 \
  --admission <admission.json> \
  --replay-snapshot <replay.json>
```

Cache and replay metadata bind the Stage 1/2 config names, reward schema, per-batch tristate label hash, actual Stage 1 checkpoint step, parameter hash, and norm-stat hash. Any mismatch rejects reuse.
