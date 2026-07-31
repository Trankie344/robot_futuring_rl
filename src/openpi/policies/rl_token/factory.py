import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.rl_token import actor_critic as rlt_actor_critic
import openpi.policies.policy as _policy
from openpi.policies.rl_token import actor_policy as _rlt_actor_policy
from openpi.shared import array_typing as at
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training.rl_token import config as _config
from openpi.training.rl_token.stage2 import checkpoints as rlt_stage2_checkpoints
from openpi.training.rl_token.stage2 import feature_identity as _feature_identity
import openpi.transforms as transforms

_RLT_MODEL_ACTION_HORIZON = 50
_RLT_MODEL_ACTION_DIM = 32
_RLT_DEFAULT_PROMPT = "fold clothes"
_RLT_ACTOR_INTERFACE = {
    "z_dim": 2048,
    "state_dim": 16,
    "action_horizon": 20,
    "action_dim": 16,
}


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch  # noqa: PLC0415

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def _validate_rlt_model_and_identity(
    train_config: _config.TrainConfig,
    base_checkpoint_dir: pathlib.Path | str,
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
) -> None:
    if train_config.name != metadata.stage1_config:
        raise ValueError(
            f"stage1_config mismatch: actor requires {metadata.stage1_config!r}, got {train_config.name!r}."
        )
    if not isinstance(train_config.data, _config.SimpleDataConfig):
        raise ValueError("RLT actor deployment requires the locked SimpleDataConfig.")
    if train_config.data.assets.asset_id != metadata.asset_id:
        raise ValueError(
            f"asset_id mismatch: actor requires {metadata.asset_id!r}, got {train_config.data.assets.asset_id!r}."
        )

    base_basename = pathlib.PurePosixPath(str(base_checkpoint_dir).rstrip("/")).name
    if base_basename != str(metadata.base_checkpoint_step):
        raise ValueError(
            "base_checkpoint_step mismatch: actor requires base checkpoint directory basename "
            f"{metadata.base_checkpoint_step}, got {base_basename!r}."
        )

    model_config = train_config.model
    actor_interface = {field: getattr(metadata.network_config, field) for field in _RLT_ACTOR_INTERFACE}
    valid_model = (
        model_config.model_type is _model.ModelType.PI05
        and model_config.action_horizon == _RLT_MODEL_ACTION_HORIZON
        and model_config.action_dim == _RLT_MODEL_ACTION_DIM
        and getattr(model_config, "rl_token_enabled", False) is True
        and getattr(model_config, "rl_token_only", False) is True
        and getattr(model_config, "rl_token_width", None) == metadata.network_config.z_dim
        and actor_interface == _RLT_ACTOR_INTERFACE
    )
    if not valid_model:
        raise ValueError(
            f"RLT actor deployment requires a JAX PI0.5 RLT 50x32 model and actor interface {_RLT_ACTOR_INTERFACE}."
        )
    registered_train_config = _config.get_stage1_config(metadata.stage1_config)
    if train_config is not registered_train_config:
        raise ValueError("RLT actor deployment requires the exact registered TrainConfig object.")


def _expected_actor_params(
    actor: rlt_actor_critic.RLTActor,
):
    network = actor.config
    compute_dtype = network.jnp_compute_dtype
    z_rl = jnp.zeros((1, network.z_dim), dtype=compute_dtype)
    state = jnp.zeros((1, network.state_dim), dtype=compute_dtype)
    reference = jnp.zeros(
        (1, network.action_horizon, network.action_dim),
        dtype=compute_dtype,
    )
    return jax.eval_shape(
        lambda key: actor.init(key, z_rl, state, reference)["params"],
        jax.random.key(0),
    )


def _validate_actor_params(
    actor: rlt_actor_critic.RLTActor,
    actor_params: object,
) -> None:
    expected = _expected_actor_params(actor)
    try:
        at.check_pytree_equality(
            expected=expected,
            got=actor_params,
            check_shapes=True,
            check_dtypes=True,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("actor params do not match the metadata network config exactly") from exc

    leaves = jax.tree_util.tree_leaves(actor_params)
    if not leaves:
        raise ValueError("actor params must contain at least one parameter")
    for leaf in leaves:
        try:
            host = np.asarray(jax.device_get(leaf))
        except (TypeError, ValueError) as exc:
            raise ValueError("actor params must contain only FP32 arrays") from exc
        if host.dtype != np.dtype(np.float32):
            raise ValueError(f"actor params must be FP32, got {host.dtype}.")
        if not np.all(np.isfinite(host)):
            raise ValueError("actor params must contain only finite values")


def _validate_rlt_base_identity(
    base_checkpoint_dir: pathlib.Path,
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
) -> None:
    try:
        params_sha256 = _feature_identity.checkpoint_tree_sha256(base_checkpoint_dir / "params")
        norm_stats_sha256 = _feature_identity.checkpoint_tree_sha256(base_checkpoint_dir / "assets" / metadata.asset_id)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("failed to verify physical RLT base identity") from exc
    if params_sha256 != metadata.frozen_params_sha256:
        raise ValueError("frozen params physical identity mismatch")
    if norm_stats_sha256 != metadata.norm_stats_sha256:
        raise ValueError("norm stats physical identity mismatch")


def _deployment_metadata(
    train_config: _config.TrainConfig,
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
    *,
    mode: _rlt_actor_policy.RLTActorMode,
) -> dict[str, object]:
    network_config = dataclasses.asdict(metadata.network_config)
    for field in ("actor_hidden_dims", "critic_hidden_dims"):
        network_config[field] = list(network_config[field])
    rlt_metadata = {
        "schema_version": metadata.schema_version,
        "stage1_config": metadata.stage1_config,
        "stage2_config": metadata.stage2_config,
        "reward_source": metadata.reward_source,
        "reward_label_values": list(metadata.reward_label_values),
        "completion_label": metadata.completion_label,
        "reward_aggregation": metadata.reward_aggregation,
        "reward_schema_version": metadata.reward_schema_version,
        "asset_id": metadata.asset_id,
        "base_checkpoint_step": metadata.base_checkpoint_step,
        "feature_identity": metadata.feature_identity,
        "frozen_params_sha256": metadata.frozen_params_sha256,
        "norm_stats_sha256": metadata.norm_stats_sha256,
        "round_id": metadata.round_id,
        "admission_sha256": metadata.admission_sha256,
        "replay_snapshot_sha256": metadata.replay_snapshot_sha256,
        "network_config": network_config,
        "algorithm_config": dataclasses.asdict(metadata.algorithm_config),
        "batch_size": metadata.batch_size,
        "round_start_step": metadata.round_start_step,
        "round_critic_updates": metadata.round_critic_updates,
        "critic_step": metadata.critic_step,
        "round_critic_step": metadata.round_critic_step,
        "jax_rng_impl": metadata.jax_rng_impl,
        "round_complete": metadata.round_complete,
        "mode": mode.value,
        "sampler_num_steps": metadata.sampler_num_steps,
    }
    return {
        **(train_config.policy_metadata or {}),
        "rlt_stage2": rlt_metadata,
    }


def create_rlt_actor_policy(
    train_config: _config.TrainConfig,
    base_checkpoint_dir: pathlib.Path | str,
    actor_checkpoint_dir: pathlib.Path | str,
    *,
    mode: _rlt_actor_policy.RLTActorMode,
    sampler_num_steps: int = 10,
    default_prompt: str | None = "fold clothes",
    seed: int = 0,
) -> _rlt_actor_policy.RLTActorPolicy:
    """Create a fixed-base PI0.5 policy controlled by a complete Stage 2 actor."""
    if not isinstance(mode, _rlt_actor_policy.RLTActorMode):
        raise ValueError(f"mode must be an RLTActorMode, got {mode!r}.")
    if type(sampler_num_steps) is not int or sampler_num_steps <= 0:
        raise ValueError("sampler_num_steps must be a positive exact integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an exact uint32 integer")

    actor_checkpoint_dir = download.maybe_download(str(actor_checkpoint_dir))
    metadata = rlt_stage2_checkpoints.load_rlt_metadata(actor_checkpoint_dir)
    if not metadata.round_complete:
        raise ValueError("RLT actor checkpoint step must be round_complete=true.")
    if sampler_num_steps != metadata.sampler_num_steps:
        raise ValueError(
            "sampler_num_steps must match the frozen feature identity "
            f"({sampler_num_steps} != {metadata.sampler_num_steps})."
        )
    if type(default_prompt) is not str or default_prompt != _RLT_DEFAULT_PROMPT:
        raise ValueError(f"default_prompt must remain {_RLT_DEFAULT_PROMPT!r} for the frozen feature identity.")
    _validate_rlt_model_and_identity(train_config, base_checkpoint_dir, metadata)

    actor = rlt_actor_critic.RLTActor(metadata.network_config)
    actor_params = _model.restore_params(actor_checkpoint_dir / "params", dtype=jnp.float32)
    _validate_actor_params(actor, actor_params)

    base_checkpoint_dir = pathlib.Path(download.maybe_download(str(base_checkpoint_dir)))
    if (base_checkpoint_dir / "model.safetensors").exists():
        raise ValueError("RLT actor deployment supports only a JAX base checkpoint.")
    _validate_rlt_base_identity(base_checkpoint_dir, metadata)

    base_assets_dir = base_checkpoint_dir / "assets"
    data_factory = dataclasses.replace(
        train_config.data,
        assets=dataclasses.replace(
            train_config.data.assets,
            assets_dir=str(base_assets_dir),
        ),
    )
    data_config = data_factory.create(base_assets_dir, train_config.model)
    if data_config.asset_id != metadata.asset_id:
        raise ValueError(
            f"asset_id mismatch after data config creation: expected {metadata.asset_id!r}, got {data_config.asset_id!r}."
        )
    norm_stats = _checkpoints.load_norm_stats(base_assets_dir, metadata.asset_id)

    logging.info("Loading fixed RLT base model...")
    base_params = _model.restore_params(base_checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(base_params)

    return _rlt_actor_policy.RLTActorPolicy(
        model,
        actor=actor,
        actor_params=actor_params,
        mode=mode,
        rng=jax.random.key(seed),
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": metadata.sampler_num_steps},
        metadata=_deployment_metadata(
            train_config,
            metadata,
            mode=mode,
        ),
    )
