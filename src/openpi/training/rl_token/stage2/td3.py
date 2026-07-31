"""Pure transition, exploration, and TD3 math primitives for RL-token training."""

import dataclasses
import math
import numbers
import operator
from typing import Literal

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.rl_token import actor_critic as rlt_actor_critic


def _is_positive_integer(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return False
    try:
        return operator.index(value) > 0
    except TypeError:
        return False


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    return result


def _validate_closed_unit_interval(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return result


def _validate_open_unit_interval(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}.")
    return result


@dataclasses.dataclass(frozen=True)
class TD3Config:
    """Hyperparameters shared by the pure TD3 update helpers."""

    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    utd_ratio: int = 5
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    beta: float = 1.0
    noise_sigma: float = 0.1
    noise_rho: float = 0.5
    reference_dropout_rate: float = 0.5
    actor_grad_clip: float = 10.0
    critic_grad_clip: float = 10.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8

    def __post_init__(self) -> None:
        self.validate()
        for name in ("policy_delay", "utd_ratio"):
            object.__setattr__(self, name, int(operator.index(getattr(self, name))))

    def validate(self) -> None:
        """Validate update, exploration, and optimizer hyperparameters."""
        _validate_closed_unit_interval(self.gamma, "gamma")
        _validate_open_unit_interval(self.tau, "tau")

        for name in ("policy_delay", "utd_ratio"):
            value = getattr(self, name)
            if not _is_positive_integer(value):
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        for name in ("actor_lr", "critic_lr", "actor_grad_clip", "critic_grad_clip", "adam_eps"):
            value = _finite_real(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}.")

        for name in ("beta", "noise_sigma"):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative, got {getattr(self, name)!r}.")

        noise_rho = _finite_real(self.noise_rho, "noise_rho")
        if not -1.0 < noise_rho < 1.0:
            raise ValueError(f"noise_rho must be in (-1, 1), got {self.noise_rho!r}.")

        _validate_closed_unit_interval(self.reference_dropout_rate, "reference_dropout_rate")

        for name in ("adam_b1", "adam_b2"):
            value = _finite_real(getattr(self, name), name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)!r}.")


_TD3_CONFIG_FIELDS = frozenset(field.name for field in dataclasses.fields(TD3Config))
_TD3_INTEGER_FIELDS = frozenset({"policy_delay", "utd_ratio"})
_TD3_FLOAT_FIELDS = _TD3_CONFIG_FIELDS - _TD3_INTEGER_FIELDS


class TD3ConfigDecodeError(ValueError):
    """A canonical TD3-config key-set mismatch with a machine-readable reason."""

    def __init__(self, message: str, *, reason: Literal["missing_only", "key_set"]):
        super().__init__(message)
        self.reason = reason


def _normalize_config_json(value: object) -> object:
    """Convert dataclass containers to their exact JSON container representation."""
    if type(value) is dict:
        return {key: _normalize_config_json(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_normalize_config_json(item) for item in value]
    return value


def _exact_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _exact_json_equal(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _exact_json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _invalid_td3_config(label: str) -> ValueError:
    return ValueError(f"{label} is invalid")


def decode_td3_config(
    payload: object,
    *,
    label: str = "algorithm_config",
) -> TD3Config:
    """Decode the exact canonical JSON representation of a TD3 config."""
    if type(label) is not str or not label:
        raise ValueError("label must be a nonempty exact string")
    if type(payload) is not dict:
        raise _invalid_td3_config(label)
    if any(type(field) is not str for field in payload):
        raise _invalid_td3_config(label)

    actual_fields = frozenset(payload)
    missing = _TD3_CONFIG_FIELDS - actual_fields
    extra = actual_fields - _TD3_CONFIG_FIELDS
    if extra:
        raise TD3ConfigDecodeError(
            f"{label} is invalid",
            reason="key_set",
        )

    for field in _TD3_INTEGER_FIELDS.intersection(actual_fields):
        if type(payload[field]) is not int:
            raise _invalid_td3_config(label)
    for field in _TD3_FLOAT_FIELDS.intersection(actual_fields):
        value = payload[field]
        if type(value) is not float or not math.isfinite(value):
            raise _invalid_td3_config(label)

    try:
        config = TD3Config(**dict(payload))
        config.validate()
    except (TypeError, ValueError) as exc:
        raise _invalid_td3_config(label) from exc

    canonical_payload = _normalize_config_json(dataclasses.asdict(config))
    canonical_provided = {field: canonical_payload[field] for field in payload}
    if not _exact_json_equal(canonical_provided, payload):
        raise _invalid_td3_config(label)
    if missing:
        raise TD3ConfigDecodeError(
            f"{label} is invalid",
            reason="missing_only",
        )
    return config


@struct.dataclass
class RLTTransitionBatch:
    """One batch of chunk-level TD3 transitions."""

    z_rl: jax.Array
    next_z_rl: jax.Array
    state_norm: jax.Array
    next_state_norm: jax.Array
    vla_reference: jax.Array
    next_vla_reference: jax.Array
    executed_action: jax.Array
    bc_anchor: jax.Array
    reward: jax.Array
    terminal: jax.Array


def validate_transition_batch(
    batch: RLTTransitionBatch,
    network_config: rlt_actor_critic.RLTActorCriticConfig,
) -> None:
    """Validate the exact tensor-shape contract for one transition batch."""
    z_shape = getattr(batch.z_rl, "shape", ())
    if getattr(batch.z_rl, "ndim", None) != 2:
        raise ValueError(f"z_rl must have rank 2 with shape (B, {network_config.z_dim}), got shape {z_shape}.")

    batch_size = z_shape[0]
    expected_shapes = {
        "z_rl": (batch_size, network_config.z_dim),
        "next_z_rl": (batch_size, network_config.z_dim),
        "state_norm": (batch_size, network_config.state_dim),
        "next_state_norm": (batch_size, network_config.state_dim),
        "vla_reference": (batch_size, network_config.action_horizon, network_config.action_dim),
        "next_vla_reference": (batch_size, network_config.action_horizon, network_config.action_dim),
        "executed_action": (batch_size, network_config.action_horizon, network_config.action_dim),
        "bc_anchor": (batch_size, network_config.action_horizon, network_config.action_dim),
        "reward": (batch_size, 1),
        "terminal": (batch_size, 1),
    }
    for field_name, expected_shape in expected_shapes.items():
        actual_shape = getattr(getattr(batch, field_name), "shape", ())
        if actual_shape != expected_shape:
            raise ValueError(f"{field_name} shape mismatch: expected {expected_shape}, got {actual_shape}.")


def whole_reference_dropout(
    key: jax.Array,
    reference: jax.Array,
    *,
    rate: float,
) -> tuple[jax.Array, jax.Array]:
    """Drop each sample's complete reference action chunk with probability ``rate``."""
    rate = _validate_closed_unit_interval(rate, "rate")
    if getattr(reference, "ndim", None) != 3:
        raise ValueError(
            f"reference must have rank 3 with shape (B, H, A), got shape {getattr(reference, 'shape', ())}."
        )

    reference_fp32 = jnp.asarray(reference, dtype=jnp.float32)
    batch_size = reference_fp32.shape[0]
    if rate == 0.0:
        dropped_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
    elif rate == 1.0:
        dropped_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
    else:
        dropped_mask = jax.random.bernoulli(key, p=rate, shape=(batch_size,))
    dropped = jnp.where(dropped_mask[:, None, None], jnp.zeros((), dtype=jnp.float32), reference_fp32)
    return dropped.astype(jnp.float32), dropped_mask


def sample_ar1_noise(
    key: jax.Array,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
    sigma: float,
    rho: float,
) -> jax.Array:
    """Sample independent stationary AR(1) trajectories for each action chunk."""
    dimensions = {
        "batch_size": batch_size,
        "horizon": horizon,
        "action_dim": action_dim,
    }
    for name, value in dimensions.items():
        if not _is_positive_integer(value):
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    sigma = _finite_real(sigma, "sigma")
    if sigma < 0.0:
        raise ValueError(f"sigma must be nonnegative, got {sigma!r}.")
    rho = _finite_real(rho, "rho")
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho!r}.")

    batch_size = int(operator.index(batch_size))
    horizon = int(operator.index(horizon))
    action_dim = int(operator.index(action_dim))
    sigma_fp32 = jnp.asarray(sigma, dtype=jnp.float32)
    rho_fp32 = jnp.asarray(rho, dtype=jnp.float32)
    innovations = (
        jax.random.normal(
            key,
            (horizon, batch_size, action_dim),
            dtype=jnp.float32,
        )
        * sigma_fp32
    )

    def ar1_step(previous: jax.Array, innovation: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = rho_fp32 * previous + jnp.sqrt(jnp.asarray(1.0, jnp.float32) - rho_fp32**2) * innovation
        return current, current

    _, remaining = jax.lax.scan(ar1_step, innovations[0], innovations[1:])
    time_major = jnp.concatenate((innovations[:1], remaining), axis=0)
    return jnp.swapaxes(time_major, 0, 1).astype(jnp.float32)


def sample_gaussian_action(
    key: jax.Array,
    mean: jax.Array,
    *,
    sigma: float,
    rho: float,
) -> jax.Array:
    """Add per-chunk AR(1) exploration noise and clip to normalized action bounds."""
    if getattr(mean, "ndim", None) != 3:
        raise ValueError(f"mean must have rank 3 with shape (B, H, A), got shape {getattr(mean, 'shape', ())}.")
    batch_size, horizon, action_dim = mean.shape
    noise = sample_ar1_noise(
        key,
        batch_size=batch_size,
        horizon=horizon,
        action_dim=action_dim,
        sigma=sigma,
        rho=rho,
    )
    return jnp.clip(jnp.asarray(mean, dtype=jnp.float32) + noise, -1.0, 1.0).astype(jnp.float32)


def compute_td_target(
    reward: jax.Array,
    terminal: jax.Array,
    next_q1: jax.Array,
    next_q2: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    """Compute a stopped FP32 TD target using the smaller target-critic estimate."""
    gamma = _validate_closed_unit_interval(gamma, "gamma")
    reward_shape = getattr(reward, "shape", ())
    if getattr(reward, "ndim", None) != 2 or reward_shape[-1:] != (1,):
        raise ValueError(f"reward must have shape (B, 1), got {reward_shape}.")
    for field_name, value in (
        ("terminal", terminal),
        ("next_q1", next_q1),
        ("next_q2", next_q2),
    ):
        actual_shape = getattr(value, "shape", ())
        if actual_shape != reward_shape:
            raise ValueError(f"{field_name} shape mismatch: expected {reward_shape}, got {actual_shape}.")

    reward_fp32 = jnp.asarray(reward, dtype=jnp.float32)
    terminal_fp32 = jnp.asarray(terminal, dtype=jnp.float32)
    next_q1_fp32 = jnp.asarray(next_q1, dtype=jnp.float32)
    next_q2_fp32 = jnp.asarray(next_q2, dtype=jnp.float32)
    target = reward_fp32 + jnp.asarray(gamma, dtype=jnp.float32) * (
        jnp.asarray(1.0, dtype=jnp.float32) - terminal_fp32
    ) * jnp.minimum(next_q1_fp32, next_q2_fp32)
    return jax.lax.stop_gradient(target.astype(jnp.float32))


def behavior_cloning_loss(mean: jax.Array, anchor: jax.Array) -> jax.Array:
    """Return batch-mean squared error after summing each action chunk."""
    mean_shape = getattr(mean, "shape", ())
    if getattr(mean, "ndim", None) != 3:
        raise ValueError(f"mean must have rank 3 with shape (B, H, A), got shape {mean_shape}.")
    anchor_shape = getattr(anchor, "shape", ())
    if anchor_shape != mean_shape:
        raise ValueError(f"anchor shape mismatch: expected {mean_shape}, got {anchor_shape}.")

    mean_fp32 = jnp.asarray(mean, dtype=jnp.float32)
    anchor_fp32 = jnp.clip(jnp.asarray(anchor, dtype=jnp.float32), -1.0, 1.0)
    per_sample_loss = jnp.sum(jnp.square(mean_fp32 - anchor_fp32), axis=(1, 2))
    return jnp.mean(per_sample_loss, dtype=jnp.float32).astype(jnp.float32)


def polyak_update(target, online, *, tau: float):
    """Move every target parameter toward its online counterpart in FP32."""
    tau = _validate_closed_unit_interval(tau, "tau")
    tau_fp32 = jnp.asarray(tau, dtype=jnp.float32)
    one_minus_tau = jnp.asarray(1.0, dtype=jnp.float32) - tau_fp32

    def update_leaf(target_leaf, online_leaf):
        target_fp32 = jnp.asarray(target_leaf, dtype=jnp.float32)
        online_fp32 = jnp.asarray(online_leaf, dtype=jnp.float32)
        if target_fp32.shape != online_fp32.shape:
            raise ValueError(f"Polyak leaf shape mismatch: expected {target_fp32.shape}, got {online_fp32.shape}.")
        return (one_minus_tau * target_fp32 + tau_fp32 * online_fp32).astype(jnp.float32)

    return jax.tree_util.tree_map(update_leaf, target, online)


def critic_loss(
    q_params,
    *,
    target_q_params,
    target_actor_params,
    batch: RLTTransitionBatch,
    rng: jax.Array,
    actor: rlt_actor_critic.RLTActor,
    critic: rlt_actor_critic.RLTCritic,
    config: TD3Config,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the summed twin-critic loss against a stopped TD3 target."""
    stopped_target_actor_params = jax.tree_util.tree_map(jax.lax.stop_gradient, target_actor_params)
    stopped_target_q_params = jax.tree_util.tree_map(jax.lax.stop_gradient, target_q_params)

    next_mean = actor.apply(
        {"params": stopped_target_actor_params},
        batch.next_z_rl,
        batch.next_state_norm,
        batch.next_vla_reference,
    )
    next_action = sample_gaussian_action(
        rng,
        next_mean,
        sigma=config.noise_sigma,
        rho=config.noise_rho,
    )
    next_q1 = critic.apply(
        {"params": stopped_target_q_params["q1"]},
        batch.next_z_rl,
        batch.next_state_norm,
        next_action,
    )
    next_q2 = critic.apply(
        {"params": stopped_target_q_params["q2"]},
        batch.next_z_rl,
        batch.next_state_norm,
        next_action,
    )
    target = compute_td_target(
        batch.reward,
        batch.terminal,
        next_q1,
        next_q2,
        gamma=config.gamma,
    )

    q1 = critic.apply(
        {"params": q_params["q1"]},
        batch.z_rl,
        batch.state_norm,
        batch.executed_action,
    ).astype(jnp.float32)
    q2 = critic.apply(
        {"params": q_params["q2"]},
        batch.z_rl,
        batch.state_norm,
        batch.executed_action,
    ).astype(jnp.float32)
    error1 = (q1 - target).astype(jnp.float32)
    error2 = (q2 - target).astype(jnp.float32)
    q1_loss = jnp.mean(jnp.square(error1), dtype=jnp.float32).astype(jnp.float32)
    q2_loss = jnp.mean(jnp.square(error2), dtype=jnp.float32).astype(jnp.float32)
    loss = (q1_loss + q2_loss).astype(jnp.float32)
    absolute_td = jnp.maximum(jnp.abs(error1), jnp.abs(error2)).astype(jnp.float32)
    metrics = {
        "critic/loss": loss,
        "critic/q1_loss": q1_loss,
        "critic/q2_loss": q2_loss,
        "critic/q1_mean": jnp.mean(q1, dtype=jnp.float32).astype(jnp.float32),
        "critic/q1_std": jnp.std(q1, dtype=jnp.float32).astype(jnp.float32),
        "critic/q1_min": jnp.min(q1).astype(jnp.float32),
        "critic/q1_max": jnp.max(q1).astype(jnp.float32),
        "critic/q2_mean": jnp.mean(q2, dtype=jnp.float32).astype(jnp.float32),
        "critic/q2_std": jnp.std(q2, dtype=jnp.float32).astype(jnp.float32),
        "critic/q2_min": jnp.min(q2).astype(jnp.float32),
        "critic/q2_max": jnp.max(q2).astype(jnp.float32),
        "critic/target_mean": jnp.mean(target, dtype=jnp.float32).astype(jnp.float32),
        "critic/target_std": jnp.std(target, dtype=jnp.float32).astype(jnp.float32),
        "critic/target_min": jnp.min(target).astype(jnp.float32),
        "critic/target_max": jnp.max(target).astype(jnp.float32),
        "critic/td_mean": jnp.mean(absolute_td, dtype=jnp.float32).astype(jnp.float32),
        "critic/td_rms": jnp.sqrt(jnp.mean(jnp.square(absolute_td), dtype=jnp.float32)).astype(jnp.float32),
        "critic/td_max": jnp.max(absolute_td).astype(jnp.float32),
    }
    return loss, metrics


def actor_loss(
    actor_params,
    *,
    q1_params,
    batch: RLTTransitionBatch,
    rng: jax.Array,
    actor: rlt_actor_critic.RLTActor,
    critic: rlt_actor_critic.RLTCritic,
    config: TD3Config,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the Q1 policy objective plus behavior-cloning regularization."""
    dropout_key, action_key = jax.random.split(rng)
    actor_reference, dropped_mask = whole_reference_dropout(
        dropout_key,
        batch.vla_reference,
        rate=config.reference_dropout_rate,
    )
    mean = actor.apply(
        {"params": actor_params},
        batch.z_rl,
        batch.state_norm,
        actor_reference,
    ).astype(jnp.float32)
    sampled_action = sample_gaussian_action(
        action_key,
        mean,
        sigma=config.noise_sigma,
        rho=config.noise_rho,
    )

    stopped_q1_params = jax.tree_util.tree_map(jax.lax.stop_gradient, q1_params)
    q1 = critic.apply(
        {"params": stopped_q1_params},
        batch.z_rl,
        batch.state_norm,
        sampled_action,
    ).astype(jnp.float32)
    q_term = (-jnp.mean(q1, dtype=jnp.float32)).astype(jnp.float32)
    bc = behavior_cloning_loss(mean, batch.bc_anchor)
    beta = jnp.asarray(config.beta, dtype=jnp.float32)
    loss = (q_term + beta * bc).astype(jnp.float32)

    clipped_anchor = jnp.clip(jnp.asarray(batch.bc_anchor, dtype=jnp.float32), -1.0, 1.0)
    clipped_vla_reference = jnp.clip(jnp.asarray(batch.vla_reference, dtype=jnp.float32), -1.0, 1.0)
    metrics = {
        "actor/loss": loss,
        "actor/q_term": q_term,
        "actor/bc_loss": bc,
        "actor/reference_drop_fraction": jnp.mean(
            dropped_mask.astype(jnp.float32),
            dtype=jnp.float32,
        ).astype(jnp.float32),
        "actor/mean_rms": jnp.sqrt(jnp.mean(jnp.square(mean), dtype=jnp.float32)).astype(jnp.float32),
        "actor/sample_rms": jnp.sqrt(jnp.mean(jnp.square(sampled_action), dtype=jnp.float32)).astype(jnp.float32),
        "actor/anchor_l1": jnp.mean(jnp.abs(mean - clipped_anchor), dtype=jnp.float32).astype(jnp.float32),
        "actor/vla_l1": jnp.mean(jnp.abs(mean - clipped_vla_reference), dtype=jnp.float32).astype(jnp.float32),
        "actor/saturation_fraction": jnp.mean(
            (jnp.abs(mean) >= jnp.asarray(0.99, dtype=jnp.float32)).astype(jnp.float32),
            dtype=jnp.float32,
        ).astype(jnp.float32),
    }
    return loss, metrics
