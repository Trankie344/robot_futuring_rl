"""Validate RL Token Stage 1 or Stage 2 checkpoint structure and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.training.rl_token.stage2 import checkpoints
from openpi.training.rl_token.stage2 import feature_identity


def validate_stage1(path: Path, *, asset_id: str) -> dict[str, object]:
    if not path.name.isdigit():
        raise ValueError("Stage 1 checkpoint must be a numeric step directory")
    params = path / "params"
    norm_stats = path / "assets" / asset_id
    return {
        "kind": "stage1",
        "step": int(path.name),
        "params_sha256": feature_identity.checkpoint_tree_sha256(params),
        "norm_stats_sha256": feature_identity.checkpoint_tree_sha256(norm_stats),
    }


def validate_stage2(path: Path) -> dict[str, object]:
    metadata = checkpoints.load_rlt_metadata(path)
    return {
        "kind": "stage2",
        "step": metadata.critic_step,
        "round_id": metadata.round_id,
        "round_complete": metadata.round_complete,
        "stage1_config": metadata.stage1_config,
        "stage2_config": metadata.stage2_config,
        "base_checkpoint_step": metadata.base_checkpoint_step,
        "frozen_params_sha256": metadata.frozen_params_sha256,
        "norm_stats_sha256": metadata.norm_stats_sha256,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("stage1", "stage2"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--asset-id",
        default="lite0030_joints_fps20_openpi_drop_last4s_min20s",
    )
    args = parser.parse_args(argv)
    result = (
        validate_stage1(args.checkpoint, asset_id=args.asset_id)
        if args.kind == "stage1"
        else validate_stage2(args.checkpoint)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
