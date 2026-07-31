# RL Token actor inference

The actor consumes the frozen `z_rl`, normalized state, and VLA reference chunk, then emits normalized-relative actions with shape `[20,16]`. OpenPI output transforms convert these actions back to the robot action space.

- `mean`: deterministic actor output.
- `collection`: actor output plus AR(1) exploration (`sigma=0.1`, `rho=0.5`).

Serve a completed checkpoint:

```bash
python scripts/tools/rl_token/serve_actor.py \
  --base-checkpoint checkpoints/rl_token/pi05_lite0030_rltoken_only/54999 \
  --actor-checkpoint <stage2-step> \
  --mode mean \
  --port 8011
```

The service verifies the Stage 1 config, checkpoint step, parameter tree hash, norm-stat hash, network interface, and completed-round metadata before loading actor parameters.
