"""Standalone Linen actor and critic networks for RL-token training."""

from collections.abc import Mapping
import dataclasses
import operator
from typing import Any, Literal

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

_KAIMING_UNIFORM = nn.initializers.variance_scaling(2.0, "fan_in", "uniform")


def _symmetric_output_uniform(key, shape, dtype=jnp.float32):
    return jax.random.uniform(
        key,
        shape,
        dtype=dtype,
        minval=-3e-3,
        maxval=3e-3,
    )


_OUTPUT_UNIFORM = _symmetric_output_uniform

_SCALAR_DIMENSION_NAMES = (
    "z_dim",
    "state_dim",
    "action_horizon",
    "action_dim",
    "actor_state_proj_dim",
    "actor_reference_proj_dim",
    "critic_state_proj_dim",
    "critic_action_proj_dim",
)
_HIDDEN_DIMENSION_NAMES = ("actor_hidden_dims", "critic_hidden_dims")


def _is_positive_integer(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return False
    try:
        return operator.index(value) > 0
    except TypeError:
        return False


@dataclasses.dataclass(frozen=True)
class RLTActorCriticConfig:
    """Shared dimensions and numerical settings for the RLT actor and critic."""

    z_dim: int = 2048
    state_dim: int = 16
    action_horizon: int = 20
    action_dim: int = 16
    actor_state_proj_dim: int = 2048
    actor_reference_proj_dim: int = 2048
    critic_state_proj_dim: int = 2048
    critic_action_proj_dim: int = 2048
    actor_hidden_dims: tuple[int, int, int] = (1024, 1024, 1024)
    critic_hidden_dims: tuple[int, int, int] = (1024, 1024, 1024)
    compute_dtype: Literal["float32", "bfloat16"] = "bfloat16"

    def __post_init__(self) -> None:
        self.validate()
        for name in _SCALAR_DIMENSION_NAMES:
            object.__setattr__(self, name, int(operator.index(getattr(self, name))))
        for name in _HIDDEN_DIMENSION_NAMES:
            dimensions = getattr(self, name)
            object.__setattr__(self, name, tuple(int(operator.index(dimension)) for dimension in dimensions))

    @property
    def flat_action_dim(self) -> int:
        """Number of scalar values in one action chunk."""
        return self.action_horizon * self.action_dim

    @property
    def jnp_compute_dtype(self) -> jnp.dtype:
        """JAX dtype used for projection and hidden-layer computation."""
        return jnp.dtype(self.compute_dtype)

    def validate(self) -> None:
        """Validate all dimensions and supported compute dtypes."""
        for name in _SCALAR_DIMENSION_NAMES:
            dimension = getattr(self, name)
            if not _is_positive_integer(dimension):
                raise ValueError(f"{name} must be a positive integer, got {dimension!r}.")

        for name in _HIDDEN_DIMENSION_NAMES:
            dimensions_tuple = getattr(self, name)
            if (
                not isinstance(dimensions_tuple, tuple)
                or len(dimensions_tuple) != 3
                or any(not _is_positive_integer(dimension) for dimension in dimensions_tuple)
            ):
                raise ValueError(
                    f"{name} must be a tuple of exactly three positive integers, got {dimensions_tuple!r}."
                )

        if self.compute_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"compute_dtype must be 'float32' or 'bfloat16', got {self.compute_dtype!r}.")


_NETWORK_CONFIG_FIELDS = frozenset(field.name for field in dataclasses.fields(RLTActorCriticConfig))


class NetworkConfigDecodeError(ValueError):
    """A canonical network-config key-set mismatch with a machine-readable reason."""

    def __init__(self, message: str, *, reason: Literal["missing_only", "key_set"]):
        super().__init__(message)
        self.reason = reason


def _decode_network_config_values(
    json_payload: dict[str, Any],
    *,
    label: str,
) -> RLTActorCriticConfig:
    arguments = dict(json_payload)
    for field in _SCALAR_DIMENSION_NAMES:
        if field in arguments and type(arguments[field]) is not int:
            raise ValueError(f"{label}.{field} must be an exact JSON integer")

    for field in _HIDDEN_DIMENSION_NAMES:
        if field not in arguments:
            continue
        dimensions = arguments[field]
        if type(dimensions) is not list:
            raise ValueError(f"{label}.{field} must be an exact JSON array")
        if len(dimensions) != 3 or any(type(dimension) is not int for dimension in dimensions):
            raise ValueError(f"{label}.{field} must contain exactly three JSON integers")
        arguments[field] = tuple(dimensions)

    if "compute_dtype" in arguments and type(arguments["compute_dtype"]) is not str:
        raise ValueError(f"{label}.compute_dtype must be an exact JSON string")

    try:
        config = RLTActorCriticConfig(**arguments)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc

    canonical_payload = dataclasses.asdict(config)
    for field in _HIDDEN_DIMENSION_NAMES:
        canonical_payload[field] = list(canonical_payload[field])
    canonical_provided = {field: canonical_payload[field] for field in json_payload}
    if canonical_provided != json_payload:
        raise ValueError(f"{label} is not the exact canonical config schema")
    return config


def decode_network_config(
    payload: Mapping[str, Any],
    *,
    label: str = "network_config",
) -> RLTActorCriticConfig:
    """Decode the exact canonical JSON representation of an actor-critic config."""
    if type(label) is not str or not label:
        raise ValueError("label must be a nonempty exact string")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")

    try:
        json_payload = dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a JSON object") from exc

    if any(type(field) is not str for field in json_payload):
        raise ValueError(f"{label} keys must be exact JSON strings")
    actual_fields = frozenset(json_payload)
    missing = sorted(_NETWORK_CONFIG_FIELDS - actual_fields)
    extra = sorted(actual_fields - _NETWORK_CONFIG_FIELDS)
    if extra:
        raise NetworkConfigDecodeError(
            f"{label} must have the exact config key set; missing={missing!r}, extra={extra!r}",
            reason="key_set",
        )
    config = _decode_network_config_values(json_payload, label=label)
    if missing:
        raise NetworkConfigDecodeError(
            f"{label} must have the exact config key set; missing={missing!r}, extra=[]",
            reason="missing_only",
        )
    return config


def _validate_common_shapes(
    config: RLTActorCriticConfig,
    z_rl: jax.Array,
    state: jax.Array,
) -> int:
    if z_rl.ndim != 2 or z_rl.shape[1] != config.z_dim:
        raise ValueError(f"z_rl shape mismatch: expected (B, {config.z_dim}), got {z_rl.shape}.")
    batch_size = z_rl.shape[0]
    expected_state_shape = (batch_size, config.state_dim)
    if state.shape != expected_state_shape:
        raise ValueError(f"state shape mismatch: expected {expected_state_shape}, got {state.shape}.")
    return batch_size


class RLTActor(nn.Module):
    """Map an RL token, state, and reference action chunk to bounded actions."""

    config: RLTActorCriticConfig

    @nn.compact
    def __call__(
        self,
        z_rl: jax.Array,
        state: jax.Array,
        reference: jax.Array,
    ) -> jax.Array:
        batch_size = _validate_common_shapes(self.config, z_rl, state)
        expected_reference_shape = (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if reference.shape != expected_reference_shape:
            raise ValueError(f"reference shape mismatch: expected {expected_reference_shape}, got {reference.shape}.")

        compute_dtype = self.config.jnp_compute_dtype
        state_features = nn.Dense(
            self.config.actor_state_proj_dim,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_KAIMING_UNIFORM,
            name="state_projection",
        )(state.astype(compute_dtype))
        state_features = nn.relu(state_features)

        reference_features = nn.Dense(
            self.config.actor_reference_proj_dim,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_KAIMING_UNIFORM,
            name="reference_projection",
        )(reference.reshape(batch_size, self.config.flat_action_dim).astype(compute_dtype))
        reference_features = nn.relu(reference_features)

        hidden = jnp.concatenate(
            [z_rl.astype(compute_dtype), state_features, reference_features],
            axis=-1,
        )
        for index, hidden_dim in enumerate(self.config.actor_hidden_dims):
            hidden = nn.Dense(
                hidden_dim,
                dtype=compute_dtype,
                param_dtype=jnp.float32,
                kernel_init=_KAIMING_UNIFORM,
                name=f"hidden_{index}",
            )(hidden)
            hidden = nn.relu(hidden)

        flat_action = nn.Dense(
            self.config.flat_action_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=_OUTPUT_UNIFORM,
            bias_init=_OUTPUT_UNIFORM,
            name="output",
        )(hidden.astype(jnp.float32))
        flat_action = jnp.tanh(flat_action)
        return flat_action.reshape(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        ).astype(jnp.float32)


class RLTCritic(nn.Module):
    """Score an executed action chunk conditioned on an RL token and state."""

    config: RLTActorCriticConfig

    @nn.compact
    def __call__(
        self,
        z_rl: jax.Array,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        batch_size = _validate_common_shapes(self.config, z_rl, state)
        expected_action_shape = (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if action.shape != expected_action_shape:
            raise ValueError(f"action shape mismatch: expected {expected_action_shape}, got {action.shape}.")

        compute_dtype = self.config.jnp_compute_dtype
        state_features = nn.Dense(
            self.config.critic_state_proj_dim,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_KAIMING_UNIFORM,
            name="state_projection",
        )(state.astype(compute_dtype))
        state_features = nn.relu(state_features)

        action_features = nn.Dense(
            self.config.critic_action_proj_dim,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            kernel_init=_KAIMING_UNIFORM,
            name="action_projection",
        )(action.reshape(batch_size, self.config.flat_action_dim).astype(compute_dtype))
        action_features = nn.relu(action_features)

        hidden = jnp.concatenate(
            [z_rl.astype(compute_dtype), state_features, action_features],
            axis=-1,
        )
        for index, hidden_dim in enumerate(self.config.critic_hidden_dims):
            hidden = nn.Dense(
                hidden_dim,
                dtype=compute_dtype,
                param_dtype=jnp.float32,
                kernel_init=_KAIMING_UNIFORM,
                name=f"hidden_{index}",
            )(hidden)
            hidden = nn.relu(hidden)

        return nn.Dense(
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=_OUTPUT_UNIFORM,
            bias_init=_OUTPUT_UNIFORM,
            name="output",
        )(hidden.astype(jnp.float32)).astype(jnp.float32)
