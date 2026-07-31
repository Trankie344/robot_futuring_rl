import dataclasses
import math
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.rl_token.pi0 import RLTokenPi0


@dataclasses.dataclass(frozen=True)
class RLTokenPi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional RL-token readout trained from the final-layer prefix representation.
    rl_token_enabled: bool = False
    rl_token_reconstruction_weight: float = 0.0
    rl_token_encoder_depth: int = 2
    rl_token_decoder_depth: int = 2
    rl_token_width: int | None = None
    rl_token_num_heads: int = 16
    rl_token_mlp_dim: int | None = None
    rl_token_max_prefix_len: int | None = None
    rl_token_only: bool = False
    rl_token_dropout: float = 0.0
    rl_token_compute_dtype: Literal["float32", "bfloat16"] = "float32"

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.rl_token_width is None:
            object.__setattr__(self, "rl_token_width", _gemma.get_config(self.paligemma_variant).width)
        if self.rl_token_max_prefix_len is None:
            # 224x224 SigLIP produces 256 tokens per image, and PI0/PI0.5 use three camera views.
            object.__setattr__(self, "rl_token_max_prefix_len", 3 * 256 + self.max_token_len)
        if self.rl_token_reconstruction_weight < 0:
            raise ValueError("rl_token_reconstruction_weight must be non-negative.")
        if not 0.0 <= self.rl_token_dropout < 1.0:
            raise ValueError("rl_token_dropout must satisfy 0.0 <= rl_token_dropout < 1.0.")
        if self.rl_token_compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("rl_token_compute_dtype must be 'float32' or 'bfloat16'.")
        if self.rl_token_only and not self.rl_token_enabled:
            raise ValueError("rl_token_only requires rl_token_enabled=True.")
        if self.rl_token_only and not (
            math.isfinite(self.rl_token_reconstruction_weight) and self.rl_token_reconstruction_weight > 0
        ):
            raise ValueError("rl_token_only requires a positive reconstruction weight.")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "RLTokenPi0":
        from openpi.models.rl_token.pi0 import RLTokenPi0  # noqa: PLC0415

        return RLTokenPi0(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "RLTokenPi0":
        """Load checkpoint parameters while keeping fresh dropout RNG runtime state.

        RL Token checkpoints intentionally export parameters only.  The dropout
        RNG counter is process-local runtime state and must never be restored
        from a training or deployment checkpoint.
        """
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        expected_state = state.filter(nnx.Not(nnx.RngState))
        expected = expected_state.to_pure_dict()
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(expected, params)
        at.check_pytree_equality(expected=expected, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)

        # ``eval_shape`` leaves process-local RNG streams abstract. Deployment
        # checkpoints contain parameters only, so materialize those streams
        # after merging without allocating a second copy of the model weights.
        rng_tags = sorted({stream.key.tag for _, stream in nnx.iter_graph(model) if isinstance(stream, nnx.RngStream)})
        if rng_tags:
            nnx.reseed(model, **{tag: seed for seed, tag in enumerate(rng_tags)})
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.rl_token_only:
            return nnx.Not(nnx_utils.PathRegex(r"rl_token(?:/.*)?"))

        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


# Source-compatible name used by the original two-stage pipeline. New code
# should prefer the explicit standalone name above.
Pi0Config = RLTokenPi0Config
