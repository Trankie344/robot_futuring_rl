from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import operator
from typing import Protocol

from etils import epath
from flax import nnx
from flax import traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import download
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    max_to_keep: int = 1,
) -> tuple[ocp.CheckpointManager, bool]:
    if isinstance(max_to_keep, bool | np.bool_):
        raise ValueError("max_to_keep must be a positive integer")
    try:
        max_to_keep = operator.index(max_to_keep)
    except TypeError as exc:
        raise ValueError("max_to_keep must be a positive integer") from exc
    if max_to_keep <= 0:
        raise ValueError("max_to_keep must be a positive integer")

    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    deployment_base_params_path: epath.Path | str | None = None,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(
            state,
            deployment_base_params_path=deployment_base_params_path,
        )
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        if _is_partial_ema(state):
            restored = checkpoint_manager.restore(
                step,
                items={"train_state": state},
            )
            return restored["train_state"]

        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _parameter_state(state: training_utils.TrainState) -> nnx.State:
    return state.params.filter(nnx.Param)


def _is_partial_ema(state: training_utils.TrainState) -> bool:
    if state.ema_params is None:
        return False
    full_param_paths = set(_parameter_state(state).flat_state())
    ema_param_paths = set(state.ema_params.filter(nnx.Param).flat_state())
    return ema_param_paths != full_param_paths


def _format_path(path: tuple) -> str:
    return "/".join(map(str, path))


def _validate_fp32_source(
    *,
    source_name: str,
    path: tuple,
    value,
    expected_variable: nnx.VariableState,
) -> None:
    expected_shape = tuple(expected_variable.value.shape)
    actual_shape = tuple(value.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{source_name} shape mismatch at {_format_path(path)}: expected {expected_shape}, got {actual_shape}."
        )
    if np.dtype(value.dtype) != np.dtype(np.float32):
        raise ValueError(f"{source_name} dtype mismatch at {_format_path(path)}: expected float32, got {value.dtype}.")


def build_deployment_params(
    state: training_utils.TrainState,
    base_params_path: epath.Path | str | None,
) -> nnx.State:
    """Build a complete FP32 deployment state, overlaying EMA params over an FP32 base."""
    expected_flat = _parameter_state(state).flat_state()
    ema_flat = {} if state.ema_params is None else state.ema_params.filter(nnx.Param).flat_state()

    unexpected_ema_paths = set(ema_flat) - set(expected_flat)
    if unexpected_ema_paths:
        paths = ", ".join(_format_path(path) for path in sorted(unexpected_ema_paths, key=str))
        raise ValueError(f"EMA contains unexpected parameter path(s): {paths}.")

    for path, ema_variable in ema_flat.items():
        _validate_fp32_source(
            source_name="EMA",
            path=path,
            value=ema_variable.value,
            expected_variable=expected_flat[path],
        )

    if base_params_path is None and set(ema_flat) != set(expected_flat):
        raise ValueError(
            "deployment_base_params_path is required to export a complete checkpoint when using partial EMA."
        )

    base_flat: dict[tuple, object] = {}
    if base_params_path is not None:
        local_base_path = download.maybe_download(str(base_params_path))
        base_params = _model.restore_params(local_base_path, restore_type=np.ndarray)
        base_flat = traverse_util.flatten_dict(base_params)

    deployment_flat = {}
    for path, expected_variable in expected_flat.items():
        if path in ema_flat:
            value = ema_flat[path].value
            source_name = "EMA"
        elif base_params_path is not None:
            if path not in base_flat:
                raise ValueError(f"Missing deployment base parameter at {_format_path(path)}.")
            value = base_flat[path]
            source_name = "Deployment base"
        else:
            value = expected_variable.value
            source_name = "Raw parameter"

        _validate_fp32_source(
            source_name=source_name,
            path=path,
            value=value,
            expected_variable=expected_variable,
        )
        deployment_flat[path] = expected_variable.replace(value)

    return nnx.State.from_flat_path(deployment_flat)


def _split_params(
    state: training_utils.TrainState,
    *,
    deployment_base_params_path: epath.Path | str | None = None,
) -> tuple[training_utils.TrainState, at.Params]:
    if _is_partial_ema(state):
        return state, build_deployment_params(state, deployment_base_params_path)
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params=nnx.State({}))
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
