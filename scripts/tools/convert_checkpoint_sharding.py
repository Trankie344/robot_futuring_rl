#!/usr/bin/env python3
"""Convert an OpenPI JAX checkpoint to a different FSDP sharding layout."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import jax
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding

from scripts.train import train as train_script


def _checkpoint_manager(path: pathlib.Path, *, overwrite: bool) -> ocp.CheckpointManager:
    path = path.resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Target checkpoint directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        path,
        item_handlers={
            "assets": _checkpoints.CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=None,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )


def _copy_assets(source_step_dir: pathlib.Path):
    source_assets = source_step_dir / "assets"

    def callback(target_dir):
        if source_assets.exists():
            shutil.copytree(source_assets, pathlib.Path(target_dir), dirs_exist_ok=True)

    return callback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-exp", required=True)
    parser.add_argument("--target-exp", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--target-fsdp-devices", type=int, default=1)
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = _config.get_config(args.config)
    source_dir = (pathlib.Path(args.checkpoint_base_dir) / config.name / args.source_exp).resolve()
    source_step_dir = source_dir / str(args.step)
    target_dir = (pathlib.Path(args.checkpoint_base_dir) / config.name / args.target_exp).resolve()
    if not source_step_dir.exists():
        raise FileNotFoundError(f"Source checkpoint step does not exist: {source_step_dir}")

    target_config = dataclasses.replace(
        config,
        exp_name=args.target_exp,
        checkpoint_base_dir=args.checkpoint_base_dir,
        fsdp_devices=args.target_fsdp_devices,
        resume=True,
    )

    mesh = sharding.make_mesh(args.target_fsdp_devices)
    rng = jax.random.key(target_config.seed)
    _, init_rng = jax.random.split(rng)

    state_shape, target_state_sharding = train_script.init_train_state(target_config, init_rng, mesh, resume=True)

    source_manager, _ = _checkpoints.initialize_checkpoint_dir(
        source_dir,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    print(f"Restoring {source_step_dir}")
    restored_state = _checkpoints.restore_state(source_manager, state_shape, None, step=args.step)

    print(f"Moving restored state to fsdp_devices={args.target_fsdp_devices} sharding")
    restored_state = jax.device_put(restored_state, target_state_sharding)
    jax.block_until_ready(restored_state)

    target_manager = _checkpoint_manager(target_dir, overwrite=args.overwrite)
    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(restored_state)  # pylint: disable=protected-access
    items = {
        "assets": _copy_assets(source_step_dir),
        "train_state": train_state,
        "params": {"params": params},
    }

    print(f"Saving converted checkpoint to {target_dir / str(args.step)}")
    target_manager.save(args.step, items)
    target_manager.wait_until_finished()

    metadata_path = target_dir / "conversion_metadata.txt"
    metadata_path.write_text(
        "\n".join(
            [
                f"converted_at_unix={time.time()}",
                f"source_dir={source_dir.resolve()}",
                f"source_step={args.step}",
                f"target_fsdp_devices={args.target_fsdp_devices}",
            ]
        )
        + "\n"
    )
    print(f"Done. Converted checkpoint: {target_dir / str(args.step)}")


if __name__ == "__main__":
    main()
