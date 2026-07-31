"""Online OpenPI policy that replaces the VLA action with an RLT actor action."""

from collections.abc import Mapping, Sequence
import enum
import math
import numbers
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training.rl_token.stage2 import td3 as rlt_td3

_FIXED_INTERFACE_DIMENSIONS = {
    "z_dim": 2048,
    "state_dim": 16,
    "action_horizon": 20,
    "action_dim": 16,
}


class RLTActorMode(enum.Enum):
    """Process-wide choice of deterministic or exploratory actor inference."""

    MEAN = "mean"
    COLLECTION = "collection"


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    return result


def _checked_array(name: str, value: object, expected_shape: tuple[int, ...]) -> jax.Array:
    array = jnp.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise ValueError(f"{name} must have a real floating dtype, got {array.dtype}.")
    if not bool(np.all(np.isfinite(np.asarray(array)))):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class RLTActorPolicy(_base_policy.BasePolicy):
    """Run the frozen VLA/RLT encoder once, then select actions with the online actor."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        actor: rlt_actor_critic.RLTActor,
        actor_params: object,
        mode: RLTActorMode,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        noise_sigma: float = 0.1,
        noise_rho: float = 0.5,
    ):
        if not isinstance(mode, RLTActorMode):
            raise ValueError(f"mode must be an RLTActorMode, got {mode!r}.")
        noise_sigma = _finite_real("noise_sigma", noise_sigma)
        noise_rho = _finite_real("noise_rho", noise_rho)
        if noise_sigma < 0.0:
            raise ValueError(f"noise_sigma must be finite and nonnegative, got {noise_sigma!r}.")
        if not -1.0 < noise_rho < 1.0:
            raise ValueError(f"noise_rho must be finite and in (-1, 1), got {noise_rho!r}.")

        network_config = actor.config
        actual_interface = {field: getattr(network_config, field) for field in _FIXED_INTERFACE_DIMENSIONS}
        if actual_interface != _FIXED_INTERFACE_DIMENSIONS:
            raise ValueError(
                "actor.config must use the fixed RLT actor interface dimensions "
                f"{_FIXED_INTERFACE_DIMENSIONS}, got {actual_interface}."
            )
        if model.action_dim < network_config.action_dim:
            raise ValueError(
                "model.action_dim must be at least the actor action dimension: "
                f"{model.action_dim} < {network_config.action_dim}."
            )
        if model.action_horizon < network_config.action_horizon:
            raise ValueError(
                "model.action_horizon must be at least the actor action horizon: "
                f"{model.action_horizon} < {network_config.action_horizon}."
            )

        self._model = model
        self._network_config = network_config
        self._mode = mode
        self._rng = jax.random.key(0) if rng is None else rng
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = dict(sample_kwargs or {})
        self._metadata = dict(metadata or {})
        self._noise_sigma = float(noise_sigma)
        self._noise_rho = float(noise_rho)

        self._sample_features = nnx_utils.module_jit(model.sample_actions_and_rl_token)
        self._actor_apply = jax.jit(
            lambda z_rl, state, reference: actor.apply(
                {"params": actor_params},
                z_rl,
                state,
                reference,
            )
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda value: value, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)

        full_state = _checked_array(
            "state",
            inputs["state"],
            (1, self._model.action_dim),
        )
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            vla_noise = jnp.asarray(noise)
            if vla_noise.ndim == 2:
                vla_noise = vla_noise[np.newaxis, ...]
            vla_noise = _checked_array(
                "VLA noise",
                vla_noise,
                (1, self._model.action_horizon, self._model.action_dim),
            )
            sample_kwargs["noise"] = vla_noise

        self._rng, vla_rng, actor_noise_rng = jax.random.split(self._rng, 3)
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        vla_actions, z_rl = self._sample_features(vla_rng, observation, **sample_kwargs)

        vla_actions = _checked_array(
            "vla_actions",
            vla_actions,
            (1, self._model.action_horizon, self._model.action_dim),
        )
        z_rl = _checked_array(
            "z_rl",
            z_rl,
            (1, self._network_config.z_dim),
        )

        actor_state = full_state[:, : self._network_config.state_dim]
        reference = vla_actions[
            :,
            : self._network_config.action_horizon,
            : self._network_config.action_dim,
        ]
        mean = self._actor_apply(z_rl, actor_state, reference)
        mean = _checked_array(
            "actor action",
            mean,
            (
                1,
                self._network_config.action_horizon,
                self._network_config.action_dim,
            ),
        )

        if self._mode is RLTActorMode.COLLECTION:
            actor_action = rlt_td3.sample_gaussian_action(
                actor_noise_rng,
                mean,
                sigma=self._noise_sigma,
                rho=self._noise_rho,
            )
            actor_action = _checked_array(
                "actor action",
                actor_action,
                (
                    1,
                    self._network_config.action_horizon,
                    self._network_config.action_dim,
                ),
            )
        else:
            actor_action = mean
        infer_time = time.monotonic() - start_time

        transformed = self._output_transform(
            {
                "state": np.asarray(full_state[0]).copy(),
                "actions": np.asarray(actor_action[0]).copy(),
            }
        )
        if not isinstance(transformed, Mapping) or "actions" not in transformed:
            raise ValueError("output transforms must return a mapping containing actions.")

        transformed_actions = np.asarray(transformed["actions"])
        if not jnp.issubdtype(jnp.dtype(transformed_actions.dtype), jnp.floating):
            raise ValueError(
                f"output transform actions must have a real floating dtype, got {transformed_actions.dtype}."
            )
        expected_shape = (
            self._network_config.action_horizon,
            self._network_config.action_dim,
        )
        if transformed_actions.shape != expected_shape:
            raise ValueError(
                f"output transform actions shape mismatch: expected {expected_shape}, got {transformed_actions.shape}."
            )
        if not bool(np.all(np.isfinite(transformed_actions))):
            raise ValueError("output transform actions must contain only finite values.")

        outputs: dict[str, Any] = dict(transformed)
        outputs["actions"] = transformed_actions
        outputs = jax.tree.map(np.asarray, outputs)
        outputs["policy_timing"] = {"infer_ms": float(infer_time * 1000.0)}
        return outputs

    @property
    def metadata(self) -> dict[str, object]:
        return self._metadata
