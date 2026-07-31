import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
from openpi.training.rl_token import config as _config
from openpi.training.rl_token.stage1 import checkpoints as _checkpoints
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Load a complete VLA base while leaving the new RL-token root initialized.

    The standalone Stage 1 model adds ``rl_token/**`` after the base checkpoint
    was produced.  Those are the only parameters that may be absent; every
    pre-existing VLA parameter must still be present with the exact expected
    shape and dtype.
    """
    loaded_params = loader.load(params_shape)
    flat_expected = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    missing_paths = set(flat_expected) - set(flat_loaded)
    invalid_missing_paths = sorted(
        (path for path in missing_paths if not path or path[0] != "rl_token"),
        key=str,
    )
    if invalid_missing_paths:
        formatted = ", ".join("/".join(map(str, path)) for path in invalid_missing_paths)
        raise ValueError(f"Base checkpoint is missing non-RL-token parameter(s): {formatted}.")

    # Fill the deliberately absent RL-token leaves with their abstract shape
    # placeholders for strict whole-tree validation. They are removed again
    # below so the freshly initialized module remains untouched.
    for path in missing_paths:
        flat_loaded[path] = flat_expected[path]
    validated_params = traverse_util.unflatten_dict(flat_loaded)
    at.check_pytree_equality(expected=params_shape, got=validated_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in flat_loaded.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _cast_frozen_params(params: nnx.State, freeze_filter: nnx.filterlib.Filter) -> nnx.State:
    """Cast frozen parameters to BF16 without modifying runtime state."""
    return nnx_utils.state_map(
        params,
        nnx.All(nnx.Param, freeze_filter),
        lambda param: param.replace(param.value.astype(jnp.bfloat16)),
    )


def _select_ema_params(params: nnx.State, config: _config.TrainConfig) -> nnx.State:
    """Select only parameters configured for EMA tracking."""
    return params.filter(config.ema_params_filter)


def _update_ema_params(old_ema: nnx.State, new_params: nnx.State, decay: float) -> nnx.State:
    """Update an EMA state while preserving its variable metadata and dtypes."""
    old_flat = old_ema.flat_state()
    new_flat = new_params.flat_state()
    if old_flat.keys() != new_flat.keys():
        raise ValueError("EMA parameters and source parameters must have identical topology.")

    def update(path: tuple, old: nnx.VariableState) -> nnx.VariableState:
        new = new_flat[path]
        if old.type is not new.type:
            raise TypeError(f"EMA variable type mismatch at {path}: {old.type} != {new.type}")
        if old.value.shape != new.value.shape:
            raise ValueError(f"EMA variable shape mismatch at {path}: old {old.value.shape} != new {new.value.shape}")
        decay_value = jnp.asarray(decay, dtype=old.value.dtype)
        updated_value = decay_value * old.value + (1 - decay_value) * new.value.astype(old.value.dtype)
        return old.replace(updated_value.astype(old.value.dtype))

    return old_ema.map(update)


def _select_kernel_params_for_norm(params: nnx.State, config: _config.TrainConfig) -> nnx.State:
    """Select matrix-like parameters used for the parameter norm metric."""
    if getattr(config.model, "rl_token_only", False):
        return params.filter(
            nnx.All(
                config.trainable_filter,
                lambda path, x: path[-1] == "kernel" and x.value.ndim > 1,
            )
        )

    return params.filter(
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        )
    )


def _is_rlt_only(config: _config.TrainConfig) -> bool:
    return bool(getattr(config.model, "rl_token_only", False))


def _require_readable_directory(label: str, path: str) -> None:
    local_path = epath.Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not local_path.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {path}")
    if not os.access(os.fspath(local_path), os.R_OK | os.X_OK):
        raise PermissionError(f"{label} directory is not readable: {path}")


def _require_readable_file(label: str, path: epath.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} must be a file: {path}")
    if not os.access(os.fspath(path), os.R_OK):
        raise PermissionError(f"{label} file is not readable: {path}")


def _validate_rlt_only_paths(config: _config.TrainConfig) -> None:
    """Fail before side effects when an RLT-only run has incomplete local inputs."""
    if not _is_rlt_only(config):
        return
    if not isinstance(config.weight_loader, _weight_loaders.CheckpointWeightLoader):
        raise TypeError("RLT-only config requires CheckpointWeightLoader.")
    if config.deployment_base_params_path is None:
        raise ValueError("RLT-only config requires deployment_base_params_path.")
    if config.weight_loader.params_path != config.deployment_base_params_path:
        raise ValueError("Weight-loader params path and deployment base params path must match.")

    if not config.data.repo_id:
        raise ValueError("RLT-only config requires a dataset directory.")
    _require_readable_directory("dataset", config.data.repo_id)
    _require_readable_directory("step 29999 params", config.weight_loader.params_path)

    assets = config.data.assets
    if not assets.assets_dir:
        raise ValueError("RLT-only config requires an explicit assets_dir.")
    if not assets.asset_id:
        raise ValueError("RLT-only config requires an explicit asset_id.")
    norm_stats = epath.Path(assets.assets_dir) / assets.asset_id / "norm_stats.json"
    _require_readable_file("norm_stats", norm_stats)


def _validate_rlt_only_state(config: _config.TrainConfig, state: training_utils.TrainState) -> None:
    """Validate the exact trainable, precision, EMA, and optimizer contracts."""
    if not _is_rlt_only(config):
        return

    all_params = state.params.filter(nnx.Param).flat_state()
    rlt_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(r"rl_token(?:/.*)?"))
    rlt_params = state.params.filter(rlt_filter).flat_state()
    trainable = state.params.filter(nnx.All(nnx.Param, config.trainable_filter)).flat_state()
    if not rlt_params:
        raise ValueError("No rl_token parameters were found at the exact rl_token root.")
    if set(trainable) != set(rlt_params):
        trainable_names = sorted("/".join(map(str, path)) for path in trainable)
        rlt_names = sorted("/".join(map(str, path)) for path in rlt_params)
        raise ValueError(
            "RLT-only trainable parameter paths do not exactly match rl_token/**: "
            f"trainable={trainable_names}, rlt={rlt_names}"
        )

    for path, variable in rlt_params.items():
        if variable.value.dtype != jnp.float32:
            raise ValueError(f"RLT param {'/'.join(map(str, path))} is not float32.")
    for path, variable in all_params.items():
        if path not in rlt_params and variable.value.dtype != jnp.bfloat16:
            raise ValueError(f"Frozen VLA param {'/'.join(map(str, path))} is not bfloat16.")

    if state.ema_params is None:
        raise ValueError("RLT-only training requires partial EMA.")
    ema_flat = state.ema_params.flat_state()
    if any(variable.type is not nnx.Param for variable in ema_flat.values()):
        raise ValueError("RLT EMA must contain only Param variables.")
    if set(ema_flat) != set(rlt_params):
        raise ValueError("RLT EMA paths do not exactly match rl_token/**.")
    if any(variable.value.dtype != jnp.float32 for variable in ema_flat.values()):
        raise ValueError("All RLT EMA leaves must be float32.")

    expected_opt_state = jax.eval_shape(
        state.tx.init,
        state.params.filter(nnx.All(nnx.Param, config.trainable_filter)),
    )
    try:
        at.check_pytree_equality(
            expected=expected_opt_state,
            got=state.opt_state,
            check_shapes=True,
            check_dtypes=True,
        )
    except ValueError as error:
        raise ValueError("Optimizer state does not exactly match RLT-only parameters.") from error


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16 while keeping runtime state (for example RNG counters) unchanged.
        params = _cast_frozen_params(params, config.freeze_filter)

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else _select_ema_params(params, config),
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _compute_loss_and_metrics(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
) -> tuple[at.Array, dict[str, at.Array]]:
    chunked_loss, metrics = model.compute_loss_with_metrics(rng, observation, actions, train=True)
    return jnp.mean(chunked_loss), metrics


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        return _compute_loss_and_metrics(model, rng, observation, actions)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        if state.ema_params is None:
            raise ValueError("EMA parameters must be initialized when ema_decay is set.")
        new_state = dataclasses.replace(
            new_state,
            ema_params=_update_ema_params(
                state.ema_params,
                _select_ema_params(new_params, config),
                state.ema_decay,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = _select_kernel_params_for_norm(new_params, config)
    info = {
        "loss": loss,
        **loss_metrics,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    _validate_rlt_only_paths(config)

    jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(jax_cache_dir).expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    _validate_rlt_only_state(config, train_state)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                deployment_base_params_path=config.deployment_base_params_path,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.stage1_cli())
