# RL Token Stage 1

Stage 1 trains only the prefix RL Token autoencoder on top of the frozen PI0.5 VLA. The independent registry name is `rl_token_stage1`; the original OpenPI training entrypoints and central config registry are not used.

Prepare assets:

```bash
python scripts/tools/rl_token/prepare_assets.py
python scripts/tools/rl_token/prepare_assets.py --verify-only
```

Train:

```bash
python scripts/train/rl_token/train_stage1.py \
  rl_token_stage1 \
  --exp-name <experiment>
```

The production config uses batch 256, 16 workers, 30,000 steps, 10,000 warmup steps, LR `5e-5`, EMA `0.999`, and checkpoints every 5,000 steps. Frozen VLA parameters compute in BF16; RL Token parameters, gradients, Adam state, and partial EMA remain FP32. Exported deployment params combine the full FP32 VLA base with the RL Token EMA.

Default external paths:

- Base: `checkpoints/rl_token/pi05_lite0030_base/29999`
- Dataset: `data/rl_token/lite0030_stage1`
- Copied Stage 1 checkpoint: `checkpoints/rl_token/pi05_lite0030_rltoken_only/54999`
