#!/usr/bin/env python3
"""Smoke test for the JAX/Orbax checkpoint path used by scripts/train/train.py."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import types

import jax
import jax.numpy as jnp
from flax import nnx
import optax

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openpi.shared import array_typing as at  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import utils as training_utils  # noqa: E402


class DummyDataLoader:
    def __init__(self, asset_id: str):
        self._data_config = types.SimpleNamespace(
            asset_id=asset_id,
            norm_stats={
                "state": _normalize.NormStats(
                    mean=jnp.array([0.0, 1.0]),
                    std=jnp.array([1.0, 2.0]),
                    q01=jnp.array([-1.0, 0.0]),
                    q99=jnp.array([1.0, 2.0]),
                ),
                "actions": _normalize.NormStats(
                    mean=jnp.array([0.1, 0.2]),
                    std=jnp.array([0.3, 0.4]),
                    q01=jnp.array([-0.5, -0.6]),
                    q99=jnp.array([0.5, 0.6]),
                ),
            },
        )

    def data_config(self):
        return self._data_config


def _make_state() -> training_utils.TrainState:
    tx = optax.adamw(learning_rate=1e-3)
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    ema_params = jax.tree.map(lambda x: x + 1, params)
    with at.disable_typechecking():
        return training_utils.TrainState(
            step=jnp.array(7, dtype=jnp.int32),
            params=params,
            model_def=graphdef,
            tx=tx,
            opt_state=tx.init(params.filter(nnx.Param)),
            ema_decay=0.999,
            ema_params=ema_params,
        )


def main() -> None:
    test_root = pathlib.Path(tempfile.mkdtemp(prefix="openpi_jax_checkpoint_save_test_"))
    try:
        print(f"test_root={test_root}", flush=True)
        checkpoint_dir = test_root / "checkpoints"
        asset_id = "dummy_asset"
        print("creating dummy TrainState", flush=True)
        state = _make_state()
        data_loader = DummyDataLoader(asset_id)

        print("initializing Orbax CheckpointManager", flush=True)
        manager, resuming = _checkpoints.initialize_checkpoint_dir(
            checkpoint_dir,
            keep_period=2,
            overwrite=False,
            resume=False,
        )
        if resuming:
            raise AssertionError("fresh checkpoint test unexpectedly entered resume mode")

        step = 2
        print(f"saving checkpoint step={step}", flush=True)
        _checkpoints.save_state(manager, state, data_loader, step)
        print("waiting for async checkpoint save", flush=True)
        manager.wait_until_finished()
        print("checkpoint save finished", flush=True)

        step_dir = checkpoint_dir / str(step)
        expected_paths = [
            step_dir / "params",
            step_dir / "train_state",
            step_dir / "assets" / asset_id / "norm_stats.json",
        ]
        missing = [str(path) for path in expected_paths if not path.exists()]
        if missing:
            raise AssertionError(f"missing JAX checkpoint outputs: {missing}")

        print("restoring checkpoint", flush=True)
        restored = _checkpoints.restore_state(manager, state, data_loader, step=step)
        if int(restored.step) != int(state.step):
            raise AssertionError(f"unexpected restored step: {restored.step}")

        restored_kernel = restored.ema_params.to_pure_dict()["kernel"]
        expected_kernel = state.ema_params.to_pure_dict()["kernel"]
        if not bool(jnp.allclose(restored_kernel, expected_kernel)):
            raise AssertionError("restored EMA params do not match saved params")

        loaded_stats = _checkpoints.load_norm_stats(step_dir / "assets", asset_id)
        if sorted(loaded_stats.keys()) != ["actions", "state"]:
            raise AssertionError(f"unexpected norm stats keys: {sorted(loaded_stats.keys())}")

        print(f"jax checkpoint save test passed: {step_dir}")
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


if __name__ == "__main__":
    main()
