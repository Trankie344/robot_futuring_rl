import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_api
from openpi.models.rl_token import pi0 as pi0_module
from openpi.models.rl_token import config as pi0_config
from openpi.shared import nnx_utils


def _config(*, rl_token_enabled: bool = True) -> pi0_config.RLTokenPi0Config:
    return pi0_config.RLTokenPi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        action_dim=4,
        max_token_len=8,
        rl_token_enabled=rl_token_enabled,
        rl_token_reconstruction_weight=1.0 if rl_token_enabled else 0.0,
        rl_token_encoder_depth=1,
        rl_token_decoder_depth=1,
        rl_token_width=64,
        rl_token_num_heads=2,
        rl_token_mlp_dim=128,
        rl_token_max_prefix_len=968,
        rl_token_dropout=0.0,
        rl_token_compute_dtype="bfloat16",
    )


def _model(*, rl_token_enabled: bool = True):
    config = _config(rl_token_enabled=rl_token_enabled)
    return config, config.create(jax.random.key(0))


def test_stage2_sampling_matches_existing_actions_and_returns_z():
    config, model = _model()
    observation = config.fake_obs(batch_size=2)
    noise = jax.random.normal(
        jax.random.key(1),
        (2, config.action_horizon, config.action_dim),
    )

    old_actions = nnx_utils.module_jit(model.sample_actions)(
        jax.random.key(2),
        observation,
        num_steps=2,
        noise=noise,
    )
    actions, z_rl = nnx_utils.module_jit(model.sample_actions_and_rl_token)(
        jax.random.key(2),
        observation,
        num_steps=2,
        noise=noise,
    )

    np.testing.assert_array_equal(np.asarray(actions), np.asarray(old_actions))
    assert z_rl.shape == (2, 64)
    assert z_rl.dtype == jnp.bfloat16


def test_stage2_z_matches_direct_prefix_encoder():
    config, model = _model()
    observation = config.fake_obs(batch_size=1)
    noise = jnp.zeros((1, 2, 4), dtype=jnp.float32)

    _, z_from_api = model.sample_actions_and_rl_token(
        jax.random.key(3),
        observation,
        num_steps=1,
        noise=noise,
    )

    processed = model_api.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    mask = pi0_module.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=mask,
        positions=positions,
    )
    assert prefix_out is not None
    assert suffix_out is None
    z_direct = model.rl_token(
        jax.lax.stop_gradient(prefix_out),
        prefix_mask,
        False,  # noqa: FBT003 -- inference uses established positional train API.
        method="encode",
    )

    np.testing.assert_array_equal(np.asarray(z_from_api), np.asarray(z_direct))


def test_stage2_sampling_requires_initialized_rl_token():
    config, model = _model(rl_token_enabled=False)

    with pytest.raises(
        ValueError,
        match="Stage 2 feature extraction requires an initialized RL-token module",
    ):
        model.sample_actions_and_rl_token(
            jax.random.key(3),
            config.fake_obs(batch_size=1),
            num_steps=1,
            noise=jnp.zeros((1, 2, 4), dtype=jnp.float32),
        )


def test_stage2_sampling_runs_one_prefix_forward_and_only_encodes_rl_token(monkeypatch):
    config, model = _model()
    observation = config.fake_obs(batch_size=1)
    noise = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    target_llm = model.PaliGemma.llm
    target_rl_token = model.rl_token
    assert target_rl_token is not None
    bridge_type = type(target_llm)
    original_call = bridge_type.__call__
    prefix_forward_count = 0
    rl_token_methods = []

    def spy_call(module, *args, **kwargs):
        nonlocal prefix_forward_count
        if module is target_llm:
            inputs = args[0]
            if isinstance(inputs, list | tuple) and len(inputs) == 2 and inputs[0] is not None and inputs[1] is None:
                prefix_forward_count += 1
        elif module is target_rl_token:
            rl_token_methods.append(kwargs.get("method"))
        return original_call(module, *args, **kwargs)

    monkeypatch.setattr(bridge_type, "__call__", spy_call)

    model.sample_actions_and_rl_token(
        jax.random.key(4),
        observation,
        num_steps=2,
        noise=noise,
    )

    assert prefix_forward_count == 1
    assert rl_token_methods == ["encode"]
