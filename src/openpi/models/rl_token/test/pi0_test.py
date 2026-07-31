import types

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.rl_token import pi0 as _pi0


class _FakeLLM:
    def __call__(self, inputs, **kwargs):
        prefix_tokens, suffix_tokens = inputs
        assert suffix_tokens is None
        assert kwargs["deterministic"] is True
        assert "adarms_cond" not in kwargs
        return (prefix_tokens * jnp.asarray(2.0, prefix_tokens.dtype), None), None


class _FakeRLT:
    def __call__(self, prefix_out, prefix_mask, *, train):
        del prefix_mask, train
        recon_loss = jnp.mean(prefix_out.astype(jnp.float32))
        metrics = {
            "rl_token/recon_loss": recon_loss,
            "rl_token/valid_tokens": jnp.asarray(4.0, dtype=jnp.float32),
            "rl_token/z_rms": jnp.asarray(3.0, dtype=jnp.float32),
            "rl_token/pred_rms": jnp.asarray(2.0, dtype=jnp.float32),
            "rl_token/target_rms": jnp.asarray(1.0, dtype=jnp.float32),
        }
        return recon_loss, metrics


class _PrefixOnlyFakePi0(_pi0.RLTokenPi0):
    def __init__(self):
        _model.BaseModel.__init__(self, action_dim=2, action_horizon=3, max_token_len=4)
        self.pi05 = True
        self.rl_token_only = True
        self.rl_token_reconstruction_weight = 1.0
        self.rl_token = _FakeRLT()
        self.PaliGemma = types.SimpleNamespace(llm=_FakeLLM())

    def embed_prefix(self, obs):
        batch_size = obs.state.shape[0]
        prefix_tokens = jnp.tile(obs.state[:, None, :], (1, 4, 4)).astype(jnp.bfloat16)
        prefix_mask = jnp.asarray([[True, True, True, False]] * batch_size)
        prefix_ar_mask = jnp.zeros((4,), dtype=jnp.bool_)
        return prefix_tokens, prefix_mask, prefix_ar_mask

    def embed_suffix(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("prefix-only RLT training must not build an action suffix")


def _observation(batch_size: int) -> _model.Observation:
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.ones((batch_size, 2), dtype=jnp.float32),
    )


def _unexpected_random_sampling(*args, **kwargs):
    del args, kwargs
    raise AssertionError("prefix-only RLT training must not sample action noise or time")


def test_rltoken_only_loss_is_prefix_only_and_action_value_independent(monkeypatch):
    model = _PrefixOnlyFakePi0()
    observation = _observation(batch_size=2)
    zero_actions = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    huge_actions = jnp.full((2, 3, 2), 1.0e20, dtype=jnp.float32)
    monkeypatch.setattr(
        _model,
        "preprocess_observation",
        lambda rng, obs, *, train=False: obs,
    )
    monkeypatch.setattr(jax.random, "normal", _unexpected_random_sampling)
    monkeypatch.setattr(jax.random, "beta", _unexpected_random_sampling)

    compute_rng = jax.random.key(0)
    zero_loss, zero_metrics = model.compute_loss_with_metrics(compute_rng, observation, zero_actions, train=True)
    huge_loss, huge_metrics = model.compute_loss_with_metrics(compute_rng, observation, huge_actions, train=True)

    assert zero_loss.shape == zero_actions.shape[:-1]
    np.testing.assert_array_equal(zero_loss, huge_loss)

    def loss_from_state(state):
        state_observation = observation.replace(state=state)
        loss, _ = model.compute_loss_with_metrics(compute_rng, state_observation, zero_actions, train=True)
        return jnp.sum(loss)

    np.testing.assert_array_equal(jax.grad(loss_from_state)(observation.state), jnp.zeros_like(observation.state))
    assert zero_loss.dtype == jnp.float32
    assert zero_metrics.keys() == huge_metrics.keys()
    for name in zero_metrics:
        np.testing.assert_array_equal(zero_metrics[name], huge_metrics[name])
    assert "action_loss" not in zero_metrics
    np.testing.assert_array_equal(
        zero_metrics["rl_token/recon_weighted_loss"],
        zero_metrics["rl_token/recon_loss"],
    )
    np.testing.assert_array_equal(
        zero_loss,
        jnp.broadcast_to(zero_metrics["rl_token/recon_weighted_loss"], zero_actions.shape[:-1]),
    )
