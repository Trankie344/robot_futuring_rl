"""Train one manually selected RL-token Stage 2 data round."""

from __future__ import annotations

import dataclasses
import sys

import tyro

from openpi.training.rl_token import config as rl_token_config
from openpi.training.rl_token.stage2 import native_training


def parse_config(argv: list[str] | None = None) -> native_training.NativeRoundConfig:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("A config name is required: rl_token_stage2 or rl_token_stage2_debug")
    selected = rl_token_config.get_stage2_config(args.pop(0))
    config = tyro.cli(native_training.NativeRoundConfig, args=args)
    if not any(argument.startswith("--runtime.") for argument in args):
        config = dataclasses.replace(config, runtime=selected.runtime)
    return config


def main(argv: list[str] | None = None) -> None:
    config = parse_config(argv)
    final_step = native_training.run_native_round(config)
    print(f"RLT Stage 2 round checkpoint: {final_step}", flush=True)


if __name__ == "__main__":
    main()
