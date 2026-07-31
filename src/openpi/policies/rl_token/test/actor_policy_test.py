from collections.abc import Mapping
import copy
import dataclasses

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.policies.rl_token import actor_policy as rlt_actor_policy
from openpi.shared import nnx_utils
from openpi.training.rl_token.stage2 import td3 as rlt_td3

_NETWORK_CONFIG = rlt_actor_critic.RLTActorCriticConfig(compute_dtype="float32")


class _ProbeActor(nn.Module):
    """Parameter-free Linen actor whose output exposes all three policy inputs."""

    config: rlt_actor_critic.RLTActorCriticConfig

    @nn.compact
    def __call__(
        self,
        z_rl: jax.Array,
        state: jax.Array,
        reference: jax.Array,
    ) -> jax.Array:
        signal = reference + state[:, None, :] / 10.0 + z_rl[:, None, : self.config.action_dim] / 100.0
        return jnp.tanh(signal).astype(jnp.float32)


class _ActorStub:
    def __init__(self, output: np.ndarray):
        self.config = _NETWORK_CONFIG
        self._output = output

    def apply(self, variables, z_rl, state, reference):
        del variables, z_rl, state, reference
        return jnp.asarray(self._output)


class _FakeModel:
    action_dim = 32
    action_horizon = 50

    def __init__(
        self,
        *,
        actions: np.ndarray | None = None,
        z_rl: np.ndarray | None = None,
    ):
        self.actions = (
            np.arange(self.action_horizon * self.action_dim, dtype=np.float32).reshape(
                1, self.action_horizon, self.action_dim
            )
            / 1000.0
            if actions is None
            else actions
        )
        self.z_rl = np.ones((1, _NETWORK_CONFIG.z_dim), dtype=np.float32) if z_rl is None else z_rl
        self.calls: list[tuple[object, object, dict[str, object]]] = []

    def sample_actions_and_rl_token(self, rng, observation, **kwargs):
        self.calls.append((rng, observation, kwargs))
        return jnp.asarray(self.actions), jnp.asarray(self.z_rl)


class _CaptureOutputTransform:
    def __init__(self, *, output_dim: int = 16):
        self.output_dim = output_dim
        self.calls: list[dict[str, np.ndarray]] = []

    def __call__(self, data):
        captured = {key: np.asarray(value).copy() for key, value in data.items()}
        self.calls.append(captured)
        return {**data, "actions": np.asarray(data["actions"])[..., : self.output_dim]}


def _observation(*, state: np.ndarray | None = None) -> dict:
    return {
        "image": {"camera": np.zeros((2, 2, 3), dtype=np.uint8)},
        "image_mask": {"camera": np.ones((), dtype=np.bool_)},
        "state": (
            np.linspace(-0.2, 0.2, 32, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32)
        ),
    }


def _disable_jits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nnx_utils, "module_jit", lambda method: method)
    monkeypatch.setattr(jax, "jit", lambda function: function)


def _make_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: _FakeModel | None = None,
    actor: object | None = None,
    actor_params: object | None = None,
    mode: rlt_actor_policy.RLTActorMode = rlt_actor_policy.RLTActorMode.MEAN,
    seed: int = 7,
    output_transforms=(),
    sample_kwargs: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    noise_sigma: object = 0.1,
    noise_rho: object = 0.5,
) -> tuple[rlt_actor_policy.RLTActorPolicy, _FakeModel, object]:
    _disable_jits(monkeypatch)
    model = _FakeModel() if model is None else model
    actor = _ProbeActor(_NETWORK_CONFIG) if actor is None else actor
    actor_params = {} if actor_params is None else actor_params
    policy = rlt_actor_policy.RLTActorPolicy(
        model,
        actor=actor,
        actor_params=actor_params,
        mode=mode,
        rng=jax.random.key(seed),
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        metadata=metadata,
        noise_sigma=noise_sigma,
        noise_rho=noise_rho,
    )
    return policy, model, actor


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("z_dim", 1024),
        ("state_dim", 8),
        ("action_horizon", 10),
        ("action_dim", 8),
    ],
)
def test_constructor_requires_exact_locked_actor_interface_dimensions(monkeypatch, field, value):
    actor = _ProbeActor(dataclasses.replace(_NETWORK_CONFIG, **{field: value}))

    with pytest.raises(ValueError, match="fixed RLT actor interface"):
        _make_policy(monkeypatch, actor=actor)


def test_constructor_allows_adjustable_projection_and_hidden_dimensions(monkeypatch):
    actor = _ProbeActor(
        dataclasses.replace(
            _NETWORK_CONFIG,
            actor_state_proj_dim=17,
            actor_reference_proj_dim=19,
            actor_hidden_dims=(23, 29, 31),
        )
    )

    policy, _, _ = _make_policy(monkeypatch, actor=actor)

    assert isinstance(policy, rlt_actor_policy.RLTActorPolicy)


def test_mean_uses_actor_slices_and_unbatched_openpi_output_contract(monkeypatch):
    capture = _CaptureOutputTransform()
    policy, model, actor = _make_policy(monkeypatch, output_transforms=(capture,))
    observation = _observation()
    original = copy.deepcopy(observation)

    result = policy.infer(observation)

    assert len(model.calls) == 1
    expected_state = original["state"][None, :16]
    expected_reference = model.actions[:, :20, :16]
    expected_mean = np.asarray(
        actor.apply(
            {"params": {}},
            model.z_rl,
            expected_state,
            expected_reference,
        )
    )
    np.testing.assert_allclose(result["actions"], expected_mean[0], atol=1e-7)
    assert result["actions"].shape == (20, 16)
    assert len(capture.calls) == 1
    assert capture.calls[0]["state"].shape == (32,)
    assert capture.calls[0]["actions"].shape == (20, 16)
    np.testing.assert_allclose(capture.calls[0]["state"], original["state"])
    np.testing.assert_allclose(capture.calls[0]["actions"], expected_mean[0], atol=1e-7)
    np.testing.assert_array_equal(observation["state"], original["state"])


def test_collection_calls_exploration_once_with_locked_parameters(monkeypatch):
    calls = []

    def sample_once(key, mean, *, sigma, rho):
        calls.append((key, np.asarray(mean), sigma, rho))
        return jnp.clip(mean + 0.25, -1.0, 1.0)

    monkeypatch.setattr(rlt_td3, "sample_gaussian_action", sample_once)
    policy, _, _ = _make_policy(
        monkeypatch,
        mode=rlt_actor_policy.RLTActorMode.COLLECTION,
    )

    result = policy.infer(_observation())

    assert len(calls) == 1
    assert calls[0][2:] == (0.1, 0.5)
    assert result["actions"].shape == (20, 16)
    assert np.all(result["actions"] >= -1.0)
    assert np.all(result["actions"] <= 1.0)


def test_mean_mode_never_calls_actor_exploration(monkeypatch):
    def unexpected_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mean mode must not sample actor exploration noise")

    monkeypatch.setattr(rlt_td3, "sample_gaussian_action", unexpected_call)
    policy, _, _ = _make_policy(monkeypatch, mode=rlt_actor_policy.RLTActorMode.MEAN)

    policy.infer(_observation())


def test_collection_seed_is_reproducible_and_advances_for_each_infer(monkeypatch):
    first, _, _ = _make_policy(
        monkeypatch,
        mode=rlt_actor_policy.RLTActorMode.COLLECTION,
        seed=41,
    )
    second, _, _ = _make_policy(
        monkeypatch,
        mode=rlt_actor_policy.RLTActorMode.COLLECTION,
        seed=41,
    )

    first_output = first.infer(_observation())["actions"]
    matching_output = second.infer(_observation())["actions"]
    next_output = first.infer(_observation())["actions"]

    np.testing.assert_array_equal(first_output, matching_output)
    assert not np.array_equal(first_output, next_output)


def test_explicit_vla_noise_is_batched_and_forwarded_with_sample_kwargs(monkeypatch):
    sample_kwargs = {"num_steps": 7}
    policy, model, _ = _make_policy(monkeypatch, sample_kwargs=sample_kwargs)
    noise = np.linspace(-1.0, 1.0, 50 * 32, dtype=np.float32).reshape(50, 32)

    policy.infer(_observation(), noise=noise)

    assert len(model.calls) == 1
    forwarded = model.calls[0][2]
    assert forwarded["num_steps"] == 7
    assert np.asarray(forwarded["noise"]).shape == (1, 50, 32)
    np.testing.assert_array_equal(np.asarray(forwarded["noise"])[0], noise)
    assert sample_kwargs == {"num_steps": 7}


@pytest.mark.parametrize(
    "noise",
    [
        np.zeros((49, 32), dtype=np.float32),
        np.zeros((1, 50, 31), dtype=np.float32),
        np.full((50, 32), np.nan, dtype=np.float32),
    ],
)
def test_rejects_invalid_explicit_vla_noise_before_model_call(monkeypatch, noise):
    policy, model, _ = _make_policy(monkeypatch)

    with pytest.raises(ValueError, match="VLA noise"):
        policy.infer(_observation(), noise=noise)

    assert model.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("noise_sigma", True),
        ("noise_sigma", "0.1"),
        ("noise_rho", False),
        ("noise_rho", "0.5"),
    ],
)
def test_rejects_non_real_or_boolean_actor_noise_settings(monkeypatch, field, value):
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        _make_policy(monkeypatch, **kwargs)


def test_metadata_and_policy_timing_match_base_policy_contract(monkeypatch):
    metadata = {"policy": "rlt", "round": 3}
    policy, _, _ = _make_policy(monkeypatch, metadata=metadata)

    result = policy.infer(_observation())

    assert policy.metadata == metadata
    assert set(result["policy_timing"]) == {"infer_ms"}
    assert isinstance(result["policy_timing"]["infer_ms"], float)
    assert result["policy_timing"]["infer_ms"] >= 0.0


@pytest.mark.parametrize(
    ("model", "actor", "observation", "message"),
    [
        (
            _FakeModel(z_rl=np.ones((1, 2047), dtype=np.float32)),
            None,
            _observation(),
            "z_rl",
        ),
        (
            _FakeModel(actions=np.zeros((1, 49, 32), dtype=np.float32)),
            None,
            _observation(),
            "vla_actions",
        ),
        (
            _FakeModel(),
            None,
            _observation(state=np.zeros((31,), dtype=np.float32)),
            "state",
        ),
        (
            _FakeModel(),
            _ActorStub(np.zeros((1, 19, 16), dtype=np.float32)),
            _observation(),
            "actor action",
        ),
    ],
)
def test_rejects_wrong_feature_and_action_shapes(monkeypatch, model, actor, observation, message):
    policy, _, _ = _make_policy(monkeypatch, model=model, actor=actor)

    with pytest.raises(ValueError, match=message):
        policy.infer(observation)
    if message == "state":
        assert model.calls == []


@pytest.mark.parametrize("source", ["state", "z_rl", "vla_actions", "actor_action"])
def test_rejects_nonfinite_policy_inputs_and_actor_outputs(monkeypatch, source):
    model = _FakeModel()
    actor = None
    observation = _observation()
    if source == "state":
        observation["state"][0] = np.nan
    elif source == "z_rl":
        model.z_rl[0, 0] = np.inf
    elif source == "vla_actions":
        model.actions[0, 0, 0] = np.nan
    else:
        output = np.zeros((1, 20, 16), dtype=np.float32)
        output[0, 0, 0] = np.inf
        actor = _ActorStub(output)
    policy, _, _ = _make_policy(monkeypatch, model=model, actor=actor)

    with pytest.raises(ValueError, match="finite"):
        policy.infer(observation)


@pytest.mark.parametrize(
    "source",
    ["state", "z_rl", "vla_actions", "actor_action", "vla_noise"],
)
def test_rejects_non_real_floating_policy_tensors(monkeypatch, source):
    model = _FakeModel()
    actor = None
    observation = _observation()
    noise = None
    if source == "state":
        observation["state"] = np.zeros((32,), dtype=np.int32)
    elif source == "z_rl":
        model.z_rl = np.ones((1, 2048), dtype=np.int32)
    elif source == "vla_actions":
        model.actions = np.zeros((1, 50, 32), dtype=np.bool_)
    elif source == "actor_action":
        actor = _ActorStub(np.zeros((1, 20, 16), dtype=np.complex64))
    else:
        noise = np.zeros((50, 32), dtype=np.int16)
    policy, _, _ = _make_policy(monkeypatch, model=model, actor=actor)

    with pytest.raises(ValueError, match="real floating"):
        policy.infer(observation, noise=noise)


@pytest.mark.parametrize(
    "bad_actions",
    [
        np.zeros((19, 16), dtype=np.float32),
        np.full((20, 16), np.nan, dtype=np.float32),
    ],
)
def test_rejects_invalid_output_transform_actions(monkeypatch, bad_actions):
    def bad_transform(data):
        return {**data, "actions": bad_actions}

    policy, _, _ = _make_policy(monkeypatch, output_transforms=(bad_transform,))

    with pytest.raises(ValueError, match="output.*actions"):
        policy.infer(_observation())


@pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.complex64])
def test_rejects_non_real_floating_output_transform_actions(monkeypatch, dtype):
    def bad_transform(data):
        return {**data, "actions": np.ones((20, 16), dtype=dtype)}

    policy, _, _ = _make_policy(monkeypatch, output_transforms=(bad_transform,))

    with pytest.raises(ValueError, match="output.*actions.*real floating"):
        policy.infer(_observation())
