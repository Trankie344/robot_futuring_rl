import dataclasses
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import td3 as rlt_td3

# Flax 0.10.2 probes ShapeDtypeStruct values with jnp.shape during parameter application.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:shape requires ndarray or scalar arguments.*:DeprecationWarning:flax\.core\.scope"
)


def _network_config() -> rlt_actor_critic.RLTActorCriticConfig:
    return rlt_actor_critic.RLTActorCriticConfig(
        z_dim=5,
        state_dim=3,
        action_horizon=4,
        action_dim=2,
        actor_state_proj_dim=4,
        actor_reference_proj_dim=4,
        critic_state_proj_dim=4,
        critic_action_proj_dim=4,
        actor_hidden_dims=(4, 4, 4),
        critic_hidden_dims=(4, 4, 4),
        compute_dtype="float32",
    )


def _td3_config_json_payload() -> dict[str, object]:
    return {
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "utd_ratio": 5,
        "actor_lr": 1e-4,
        "critic_lr": 3e-4,
        "beta": 1.0,
        "noise_sigma": 0.1,
        "noise_rho": 0.5,
        "reference_dropout_rate": 0.5,
        "actor_grad_clip": 10.0,
        "critic_grad_clip": 10.0,
        "adam_b1": 0.9,
        "adam_b2": 0.999,
        "adam_eps": 1e-8,
    }


def _transition_batch(
    config: rlt_actor_critic.RLTActorCriticConfig,
    *,
    batch_size: int = 2,
) -> rlt_td3.RLTTransitionBatch:
    z_shape = (batch_size, config.z_dim)
    state_shape = (batch_size, config.state_dim)
    action_shape = (batch_size, config.action_horizon, config.action_dim)
    scalar_shape = (batch_size, 1)
    return rlt_td3.RLTTransitionBatch(
        z_rl=jnp.zeros(z_shape),
        next_z_rl=jnp.ones(z_shape),
        state_norm=jnp.zeros(state_shape),
        next_state_norm=jnp.ones(state_shape),
        vla_reference=jnp.zeros(action_shape),
        next_vla_reference=jnp.ones(action_shape),
        executed_action=jnp.zeros(action_shape),
        bc_anchor=jnp.ones(action_shape),
        reward=jnp.zeros(scalar_shape),
        terminal=jnp.zeros(scalar_shape),
    )


def _loss_batch(
    config: rlt_actor_critic.RLTActorCriticConfig,
    *,
    batch_size: int = 3,
) -> rlt_td3.RLTTransitionBatch:
    keys = jax.random.split(jax.random.key(30), 8)
    z_shape = (batch_size, config.z_dim)
    state_shape = (batch_size, config.state_dim)
    action_shape = (batch_size, config.action_horizon, config.action_dim)
    reward = jnp.asarray([[1.0], [-0.25], [0.5]], dtype=jnp.float32)[:batch_size]
    terminal = jnp.asarray([[0.0], [1.0], [0.0]], dtype=jnp.float32)[:batch_size]
    return rlt_td3.RLTTransitionBatch(
        z_rl=jax.random.normal(keys[0], z_shape, dtype=jnp.float32),
        next_z_rl=jax.random.normal(keys[1], z_shape, dtype=jnp.float32),
        state_norm=jax.random.normal(keys[2], state_shape, dtype=jnp.float32),
        next_state_norm=jax.random.normal(keys[3], state_shape, dtype=jnp.float32),
        vla_reference=jnp.clip(jax.random.normal(keys[4], action_shape, dtype=jnp.float32) * 0.3, -1.0, 1.0),
        next_vla_reference=jnp.clip(
            jax.random.normal(keys[5], action_shape, dtype=jnp.float32) * 0.3,
            -1.0,
            1.0,
        ),
        executed_action=jnp.clip(
            jax.random.normal(keys[6], action_shape, dtype=jnp.float32) * 0.3,
            -1.0,
            1.0,
        ),
        bc_anchor=jnp.clip(jax.random.normal(keys[7], action_shape, dtype=jnp.float32) * 0.5, -1.0, 1.0),
        reward=reward,
        terminal=terminal,
    )


def _loss_modules_and_params(config: rlt_actor_critic.RLTActorCriticConfig, batch: rlt_td3.RLTTransitionBatch):
    actor = rlt_actor_critic.RLTActor(config)
    critic = rlt_actor_critic.RLTCritic(config)
    keys = jax.random.split(jax.random.key(31), 5)
    actor_params = actor.init(
        keys[0],
        batch.z_rl,
        batch.state_norm,
        batch.bc_anchor,
    )["params"]
    target_actor_params = actor.init(
        keys[1],
        batch.next_z_rl,
        batch.next_state_norm,
        batch.next_vla_reference,
    )["params"]
    q_params = {
        "q1": critic.init(keys[2], batch.z_rl, batch.state_norm, batch.executed_action)["params"],
        "q2": critic.init(keys[3], batch.z_rl, batch.state_norm, batch.executed_action)["params"],
    }
    target_q_params = {
        "q1": critic.init(keys[3], batch.next_z_rl, batch.next_state_norm, batch.executed_action)["params"],
        "q2": critic.init(keys[4], batch.next_z_rl, batch.next_state_norm, batch.executed_action)["params"],
    }
    return actor, critic, actor_params, target_actor_params, q_params, target_q_params


def _assert_scalar_fp32_metrics(metrics, expected_keys):
    assert set(metrics) == expected_keys
    for value in metrics.values():
        assert value.shape == ()
        assert value.dtype == jnp.float32
        assert jnp.isfinite(value)


def _assert_metric_values(metrics, expected):
    assert set(metrics) == set(expected)
    for key, expected_value in expected.items():
        np.testing.assert_allclose(metrics[key], expected_value, rtol=1e-6, atol=1e-6)


def _assert_exact_zero_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves
    for leaf in leaves:
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_td3_config_defaults_are_the_public_contract():
    config = rlt_td3.TD3Config()

    assert dataclasses.is_dataclass(config)
    assert config.gamma == 0.99
    assert config.tau == 0.005
    assert config.policy_delay == 2
    assert config.utd_ratio == 5
    assert config.actor_lr == 1e-4
    assert config.critic_lr == 3e-4
    assert config.beta == 1.0
    assert config.noise_sigma == 0.1
    assert config.noise_rho == 0.5
    assert config.reference_dropout_rate == 0.5
    assert config.actor_grad_clip == 10.0
    assert config.critic_grad_clip == 10.0
    assert config.adam_b1 == 0.9
    assert config.adam_b2 == 0.999
    assert config.adam_eps == 1e-8
    assert config.validate() is None

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gamma = 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gamma": -0.01}, "gamma"),
        ({"gamma": 1.01}, "gamma"),
        ({"gamma": float("nan")}, "gamma"),
        ({"tau": 0.0}, "tau"),
        ({"tau": 1.01}, "tau"),
        ({"tau": float("inf")}, "tau"),
        ({"policy_delay": 0}, "policy_delay"),
        ({"utd_ratio": -1}, "utd_ratio"),
        ({"actor_lr": 0.0}, "actor_lr"),
        ({"actor_lr": float("nan")}, "actor_lr"),
        ({"critic_lr": -1.0}, "critic_lr"),
        ({"critic_lr": float("inf")}, "critic_lr"),
        ({"beta": -0.1}, "beta"),
        ({"beta": float("nan")}, "beta"),
        ({"noise_sigma": -0.1}, "noise_sigma"),
        ({"noise_sigma": float("inf")}, "noise_sigma"),
        ({"noise_rho": -1.0}, "noise_rho"),
        ({"noise_rho": 1.0}, "noise_rho"),
        ({"noise_rho": float("nan")}, "noise_rho"),
        ({"reference_dropout_rate": -0.1}, "reference_dropout_rate"),
        ({"reference_dropout_rate": 1.1}, "reference_dropout_rate"),
        ({"reference_dropout_rate": float("inf")}, "reference_dropout_rate"),
        ({"actor_grad_clip": 0.0}, "actor_grad_clip"),
        ({"actor_grad_clip": float("inf")}, "actor_grad_clip"),
        ({"critic_grad_clip": -1.0}, "critic_grad_clip"),
        ({"critic_grad_clip": float("nan")}, "critic_grad_clip"),
        ({"adam_b1": -0.1}, "adam_b1"),
        ({"adam_b1": 1.0}, "adam_b1"),
        ({"adam_b1": float("nan")}, "adam_b1"),
        ({"adam_b2": -0.1}, "adam_b2"),
        ({"adam_b2": 1.0}, "adam_b2"),
        ({"adam_b2": float("inf")}, "adam_b2"),
        ({"adam_eps": 0.0}, "adam_eps"),
        ({"adam_eps": float("nan")}, "adam_eps"),
    ],
)
def test_td3_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rlt_td3.TD3Config(**kwargs)


@pytest.mark.parametrize("field", ["policy_delay", "utd_ratio"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_td3_config_rejects_non_integer_update_counts(field, value):
    with pytest.raises(ValueError, match=field):
        rlt_td3.TD3Config(**{field: value})


def test_td3_config_normalizes_integer_like_update_counts_and_is_hashable():
    config = rlt_td3.TD3Config(policy_delay=np.int64(3), utd_ratio=jnp.int32(7))

    assert type(config.policy_delay) is int
    assert type(config.utd_ratio) is int
    assert isinstance(hash(config), int)


def test_decode_td3_config_accepts_only_the_canonical_json_shape():
    payload = _td3_config_json_payload()

    config = rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")

    assert config == rlt_td3.TD3Config()
    assert dataclasses.asdict(config) == payload


class _DictSubclass(dict):
    pass


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        tuple(_td3_config_json_payload().items()),
        MappingProxyType(_td3_config_json_payload()),
        _DictSubclass(_td3_config_json_payload()),
    ],
)
def test_decode_td3_config_requires_an_exact_builtin_dict(payload):
    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")


class _HostileDict(dict):
    def __iter__(self):
        raise AssertionError("untrusted dict subclass was iterated")

    def keys(self):
        raise AssertionError("untrusted dict subclass keys were read")

    def items(self):
        raise AssertionError("untrusted dict subclass items were read")


def test_decode_td3_config_rejects_a_hostile_dict_subclass_before_inspection():
    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(_HostileDict(), label="policy.algorithm_config")


class _FormatBomb:
    def __init__(self):
        self.format_calls = 0
        self.repr_calls = 0

    def __format__(self, _format_spec: str) -> str:
        self.format_calls += 1
        raise AssertionError("untrusted value was formatted")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise AssertionError("untrusted value was represented")


class _FormatBombStr(str):
    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.format_calls = 0
        instance.repr_calls = 0
        return instance

    def __format__(self, _format_spec: str) -> str:
        self.format_calls += 1
        raise AssertionError("untrusted string subclass was formatted")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise AssertionError("untrusted string subclass was represented")


@pytest.mark.parametrize("label", [_FormatBomb(), _FormatBombStr("policy.algorithm_config")])
def test_decode_td3_config_rejects_nonexact_labels_without_formatting_them(label):
    with pytest.raises(ValueError, match="label must be a nonempty exact string"):
        rlt_td3.decode_td3_config(_td3_config_json_payload(), label=label)

    assert label.format_calls == 0
    assert label.repr_calls == 0


def test_decode_td3_config_rejects_an_empty_label():
    with pytest.raises(ValueError, match="label must be a nonempty exact string"):
        rlt_td3.decode_td3_config(_td3_config_json_payload(), label="")


@pytest.mark.parametrize("invalid_key", [1, _FormatBombStr("gamma")])
def test_decode_td3_config_rejects_nonexact_json_string_keys_without_formatting_them(invalid_key):
    payload = _td3_config_json_payload()
    payload[invalid_key] = payload.pop("gamma")

    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")

    if isinstance(invalid_key, _FormatBombStr):
        assert invalid_key.format_calls == 0
        assert invalid_key.repr_calls == 0


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "missing_only"),
        ("missing_and_extra", "key_set"),
        ("extra", "key_set"),
    ],
)
def test_decode_td3_config_reports_a_typed_key_set_reason(mutation, reason):
    payload = _td3_config_json_payload()
    if mutation != "extra":
        del payload["gamma"]
    if mutation != "missing":
        payload["unexpected"] = 1

    with pytest.raises(rlt_td3.TD3ConfigDecodeError) as exc_info:
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")

    assert str(exc_info.value) == "policy.algorithm_config is invalid"
    assert exc_info.value.reason == reason


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("field", ["policy_delay", "utd_ratio"])
@pytest.mark.parametrize("invalid_value", [True, 2.0, np.int64(2), _IntSubclass(2), "2"])
def test_decode_td3_config_requires_exact_json_integers(field, invalid_value):
    payload = _td3_config_json_payload()
    payload[field] = invalid_value

    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")


class _FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    "field",
    [
        "gamma",
        "tau",
        "actor_lr",
        "critic_lr",
        "beta",
        "noise_sigma",
        "noise_rho",
        "reference_dropout_rate",
        "actor_grad_clip",
        "critic_grad_clip",
        "adam_b1",
        "adam_b2",
        "adam_eps",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [True, 1, np.float64(0.5), _FloatSubclass(0.5), "0.5", [], (), {}],
)
def test_decode_td3_config_requires_exact_json_floats(field, invalid_value):
    payload = _td3_config_json_payload()
    payload[field] = invalid_value

    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_decode_td3_config_rejects_nonfinite_json_numbers(invalid_value):
    payload = _td3_config_json_payload()
    payload["gamma"] = invalid_value

    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")


def test_decode_td3_config_never_formats_an_invalid_scalar():
    payload = _td3_config_json_payload()
    invalid_value = _FormatBomb()
    payload["gamma"] = invalid_value

    with pytest.raises(ValueError, match="policy\\.algorithm_config is invalid"):
        rlt_td3.decode_td3_config(payload, label="policy.algorithm_config")

    assert invalid_value.format_calls == 0
    assert invalid_value.repr_calls == 0


def test_decode_td3_config_does_not_mutate_or_alias_the_input_mapping():
    payload = _td3_config_json_payload()
    before = payload.copy()

    config = rlt_td3.decode_td3_config(payload)
    payload["gamma"] = 0.5

    assert config == rlt_td3.TD3Config()
    assert before == _td3_config_json_payload()


def test_transition_batch_accepts_the_exact_contract():
    config = _network_config()
    batch = _transition_batch(config)

    assert rlt_td3.validate_transition_batch(batch, config) is None
    leaves = jax.tree_util.tree_leaves(batch)
    assert len(leaves) == 10


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("z_rl", jnp.zeros((2, 4)), r"z_rl.*\(2, 5\).*\(2, 4\)"),
        ("next_z_rl", jnp.zeros((3, 5)), r"next_z_rl.*\(2, 5\).*\(3, 5\)"),
        ("state_norm", jnp.zeros((2, 4)), r"state_norm.*\(2, 3\).*\(2, 4\)"),
        ("next_state_norm", jnp.zeros((1, 3)), r"next_state_norm.*\(2, 3\).*\(1, 3\)"),
        ("vla_reference", jnp.zeros((2, 4, 3)), r"vla_reference.*\(2, 4, 2\).*\(2, 4, 3\)"),
        ("next_vla_reference", jnp.zeros((2, 8)), r"next_vla_reference.*\(2, 4, 2\).*\(2, 8\)"),
        ("executed_action", jnp.zeros((3, 4, 2)), r"executed_action.*\(2, 4, 2\).*\(3, 4, 2\)"),
        ("bc_anchor", jnp.zeros((2, 2, 4)), r"bc_anchor.*\(2, 4, 2\).*\(2, 2, 4\)"),
        ("reward", jnp.zeros((2,)), r"reward.*\(2, 1\).*\(2,\)"),
        ("terminal", jnp.zeros((2, 2)), r"terminal.*\(2, 1\).*\(2, 2\)"),
    ],
)
def test_transition_batch_rejects_malformed_field(field, replacement, expected):
    config = _network_config()
    batch = _transition_batch(config).replace(**{field: replacement})

    with pytest.raises(ValueError, match=expected):
        rlt_td3.validate_transition_batch(batch, config)


@pytest.mark.parametrize("bad_z", [jnp.asarray(0.0), jnp.zeros((2, 3, 5))])
def test_transition_batch_rejects_z_rank_with_clear_error(bad_z):
    config = _network_config()
    batch = _transition_batch(config).replace(z_rl=bad_z)

    with pytest.raises(ValueError, match="z_rl.*rank 2"):
        rlt_td3.validate_transition_batch(batch, config)


def test_whole_reference_dropout_drops_complete_samples_and_reports_mask():
    reference = jnp.arange(64 * 4 * 2, dtype=jnp.float32).reshape(64, 4, 2) + 1

    dropped, dropped_mask = rlt_td3.whole_reference_dropout(jax.random.key(1), reference, rate=0.5)

    assert dropped.shape == reference.shape
    assert dropped.dtype == jnp.float32
    assert dropped_mask.shape == (64,)
    assert dropped_mask.dtype == jnp.bool_
    assert jnp.any(dropped_mask)
    assert jnp.any(~dropped_mask)
    np.testing.assert_array_equal(dropped[dropped_mask], 0.0)
    np.testing.assert_array_equal(dropped[~dropped_mask], reference[~dropped_mask])


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_whole_reference_dropout_has_exact_endpoints(rate):
    reference = jnp.ones((3, 4, 2), dtype=jnp.bfloat16)

    dropped, dropped_mask = rlt_td3.whole_reference_dropout(jax.random.key(2), reference, rate=rate)

    assert dropped.dtype == jnp.float32
    if rate == 0.0:
        np.testing.assert_array_equal(dropped, jnp.ones_like(reference, dtype=jnp.float32))
        assert not jnp.any(dropped_mask)
    else:
        np.testing.assert_array_equal(dropped, jnp.zeros_like(reference, dtype=jnp.float32))
        assert jnp.all(dropped_mask)


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf")])
def test_whole_reference_dropout_rejects_invalid_rate(rate):
    with pytest.raises(ValueError, match="rate"):
        rlt_td3.whole_reference_dropout(jax.random.key(3), jnp.ones((2, 4, 2)), rate=rate)


@pytest.mark.parametrize("shape", [(), (4,), (2, 4), (2, 4, 2, 1)])
def test_whole_reference_dropout_rejects_non_chunk_batch_shape(shape):
    with pytest.raises(ValueError, match="reference.*rank 3"):
        rlt_td3.whole_reference_dropout(jax.random.key(4), jnp.ones(shape), rate=0.5)


def test_ar1_noise_is_reproducible_fp32_and_has_expected_statistics():
    key = jax.random.key(5)

    noise = rlt_td3.sample_ar1_noise(
        key,
        batch_size=8192,
        horizon=20,
        action_dim=2,
        sigma=0.1,
        rho=0.5,
    )
    repeated = rlt_td3.sample_ar1_noise(
        key,
        batch_size=8192,
        horizon=20,
        action_dim=2,
        sigma=0.1,
        rho=0.5,
    )

    assert noise.shape == (8192, 20, 2)
    assert noise.dtype == jnp.float32
    np.testing.assert_array_equal(noise, repeated)
    assert abs(float(jnp.std(noise)) - 0.1) < 0.004
    adjacent_correlation = jnp.corrcoef(noise[:, :-1, :].reshape(-1), noise[:, 1:, :].reshape(-1))[0, 1]
    assert abs(float(adjacent_correlation) - 0.5) < 0.03


def test_ar1_noise_resets_each_chunk_with_stationary_first_sample():
    noise = rlt_td3.sample_ar1_noise(
        jax.random.key(6),
        batch_size=8192,
        horizon=20,
        action_dim=1,
        sigma=0.1,
        rho=0.9,
    )

    assert abs(float(jnp.std(noise[:, 0, :])) - 0.1) < 0.004
    adjacent_chunks = jnp.corrcoef(noise[:-1, 0, 0], noise[1:, 0, 0])[0, 1]
    assert abs(float(adjacent_chunks)) < 0.05


def test_ar1_noise_supports_horizon_one_and_zero_sigma():
    horizon_one = rlt_td3.sample_ar1_noise(
        jax.random.key(7),
        batch_size=3,
        horizon=1,
        action_dim=2,
        sigma=0.2,
        rho=-0.2,
    )
    zero = rlt_td3.sample_ar1_noise(
        jax.random.key(8),
        batch_size=3,
        horizon=4,
        action_dim=2,
        sigma=0.0,
        rho=0.5,
    )

    assert horizon_one.shape == (3, 1, 2)
    assert horizon_one.dtype == jnp.float32
    np.testing.assert_array_equal(zero, jnp.zeros((3, 4, 2), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"horizon": 0}, "horizon"),
        ({"action_dim": -1}, "action_dim"),
        ({"sigma": -0.1}, "sigma"),
        ({"sigma": float("nan")}, "sigma"),
        ({"sigma": float("inf")}, "sigma"),
        ({"rho": -1.0}, "rho"),
        ({"rho": 1.0}, "rho"),
        ({"rho": float("nan")}, "rho"),
        ({"rho": float("inf")}, "rho"),
    ],
)
def test_ar1_noise_rejects_invalid_arguments(kwargs, message):
    arguments = {
        "batch_size": 2,
        "horizon": 4,
        "action_dim": 3,
        "sigma": 0.1,
        "rho": 0.5,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        rlt_td3.sample_ar1_noise(jax.random.key(9), **arguments)


@pytest.mark.parametrize("field", ["batch_size", "horizon", "action_dim"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_ar1_noise_rejects_non_integer_dimensions(field, value):
    arguments = {
        "batch_size": 2,
        "horizon": 4,
        "action_dim": 3,
        "sigma": 0.1,
        "rho": 0.5,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        rlt_td3.sample_ar1_noise(jax.random.key(10), **arguments)


def test_sample_gaussian_action_adds_ar1_noise_and_clips():
    mean = jnp.asarray(
        [
            [[2.0, -2.0], [0.2, -0.2], [2.0, -2.0]],
            [[2.0, -2.0], [0.2, -0.2], [2.0, -2.0]],
        ],
        dtype=jnp.bfloat16,
    )

    action = rlt_td3.sample_gaussian_action(jax.random.key(11), mean, sigma=0.1, rho=0.5)

    assert action.shape == mean.shape
    assert action.dtype == jnp.float32
    assert jnp.all(action >= -1.0)
    assert jnp.all(action <= 1.0)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 3, 4, 1)])
def test_sample_gaussian_action_rejects_non_chunk_batch_shape(shape):
    with pytest.raises(ValueError, match="mean.*rank 3"):
        rlt_td3.sample_gaussian_action(jax.random.key(12), jnp.ones(shape), sigma=0.1, rho=0.5)


def test_td_target_uses_twin_minimum_in_both_orderings_and_masks_terminal_transitions():
    reward = jnp.asarray([[1.0], [4.0], [7.0]], dtype=jnp.bfloat16)
    terminal = jnp.asarray([[0.0], [0.0], [1.0]], dtype=jnp.bfloat16)
    next_q1 = jnp.asarray([[2.0], [10.0], [100.0]], dtype=jnp.bfloat16)
    next_q2 = jnp.asarray([[3.0], [8.0], [-100.0]], dtype=jnp.bfloat16)

    target = rlt_td3.compute_td_target(reward, terminal, next_q1, next_q2, gamma=0.99)

    assert target.dtype == jnp.float32
    np.testing.assert_array_equal(
        target,
        jnp.asarray([[2.98], [11.92], [7.0]], dtype=jnp.float32),
    )


def test_td_target_is_stop_gradient_with_respect_to_both_next_critics():
    reward = jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)
    terminal = jnp.zeros((2, 1), dtype=jnp.float32)
    next_q1 = jnp.asarray([[3.0], [5.0]], dtype=jnp.float32)
    next_q2 = jnp.asarray([[4.0], [4.0]], dtype=jnp.float32)

    grad_q1, grad_q2 = jax.grad(
        lambda q1, q2: jnp.sum(rlt_td3.compute_td_target(reward, terminal, q1, q2, gamma=0.99)),
        argnums=(0, 1),
    )(next_q1, next_q2)

    np.testing.assert_array_equal(grad_q1, jnp.zeros_like(next_q1))
    np.testing.assert_array_equal(grad_q2, jnp.zeros_like(next_q2))


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), float("inf")])
def test_td_target_rejects_invalid_gamma(gamma):
    values = jnp.zeros((2, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="gamma"):
        rlt_td3.compute_td_target(values, values, values, values, gamma=gamma)


@pytest.mark.parametrize("field", ["terminal", "next_q1", "next_q2"])
def test_td_target_rejects_mismatched_shapes(field):
    arguments = {
        "reward": jnp.zeros((2, 1), dtype=jnp.float32),
        "terminal": jnp.zeros((2, 1), dtype=jnp.float32),
        "next_q1": jnp.zeros((2, 1), dtype=jnp.float32),
        "next_q2": jnp.zeros((2, 1), dtype=jnp.float32),
        "gamma": 0.99,
    }
    arguments[field] = jnp.zeros((2,), dtype=jnp.float32)

    with pytest.raises(ValueError, match=field):
        rlt_td3.compute_td_target(**arguments)


def test_behavior_cloning_loss_sums_chunk_dimensions_then_means_batch():
    mean = jnp.zeros((2, 4, 2), dtype=jnp.bfloat16)
    anchor = jnp.stack(
        (
            jnp.ones((4, 2), dtype=jnp.float32),
            jnp.full((4, 2), 0.5, dtype=jnp.float32),
        )
    )

    loss = rlt_td3.behavior_cloning_loss(mean, anchor)

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    np.testing.assert_allclose(loss, 5.0, rtol=0.0, atol=0.0)


def test_behavior_cloning_loss_clips_anchor_before_squaring():
    mean = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    anchor = jnp.asarray([[[2.0], [-3.0]]], dtype=jnp.float32)

    loss = rlt_td3.behavior_cloning_loss(mean, anchor)

    np.testing.assert_allclose(loss, 2.0, rtol=0.0, atol=0.0)


def test_behavior_cloning_loss_rejects_shape_mismatch_and_wrong_rank():
    with pytest.raises(ValueError, match="anchor"):
        rlt_td3.behavior_cloning_loss(jnp.zeros((2, 4, 2)), jnp.zeros((2, 8)))
    with pytest.raises(ValueError, match="mean.*rank 3"):
        rlt_td3.behavior_cloning_loss(jnp.zeros((2, 8)), jnp.zeros((2, 8)))


def test_polyak_update_updates_every_leaf_in_fp32():
    target = {
        "a": jnp.asarray([0.0, 2.0], dtype=jnp.bfloat16),
        "b": (jnp.asarray([4.0], dtype=jnp.float16),),
    }
    online = {
        "a": jnp.asarray([10.0, 6.0], dtype=jnp.float16),
        "b": (jnp.asarray([8.0], dtype=jnp.bfloat16),),
    }

    updated = rlt_td3.polyak_update(target, online, tau=0.25)

    assert jax.tree_util.tree_structure(updated) == jax.tree_util.tree_structure(target)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree_util.tree_leaves(updated))
    np.testing.assert_allclose(updated["a"], jnp.asarray([2.5, 3.0], dtype=jnp.float32), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(updated["b"][0], jnp.asarray([5.0], dtype=jnp.float32), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("tau", [-0.1, 1.1, float("nan"), float("inf")])
def test_polyak_update_rejects_invalid_tau(tau):
    tree = {"a": jnp.zeros((1,), dtype=jnp.float32)}
    with pytest.raises(ValueError, match="tau"):
        rlt_td3.polyak_update(tree, tree, tau=tau)


def test_polyak_update_rejects_mismatched_tree_structures():
    with pytest.raises(ValueError, match="Dict key mismatch"):
        rlt_td3.polyak_update(
            {"a": jnp.zeros((1,), dtype=jnp.float32)},
            {"b": jnp.zeros((1,), dtype=jnp.float32)},
            tau=0.25,
        )


def test_polyak_update_rejects_mismatched_leaf_shapes_before_broadcasting():
    with pytest.raises(
        ValueError,
        match=r"Polyak leaf shape mismatch: expected \(2, 1\), got \(2,\)",
    ):
        rlt_td3.polyak_update(
            {"a": jnp.zeros((2, 1), dtype=jnp.float32)},
            {"a": jnp.zeros((2,), dtype=jnp.float32)},
            tau=0.25,
        )


def test_polyak_update_accepts_matching_scalar_leaves():
    updated = rlt_td3.polyak_update(
        {"a": jnp.asarray(2.0, dtype=jnp.bfloat16)},
        {"a": jnp.asarray(6.0, dtype=jnp.bfloat16)},
        tau=0.25,
    )

    assert updated["a"].shape == ()
    assert updated["a"].dtype == jnp.float32
    np.testing.assert_array_equal(updated["a"], jnp.asarray(3.0, dtype=jnp.float32))


@pytest.mark.parametrize(
    ("tau", "expected"),
    [
        (0.0, [2.0]),
        (1.0, [6.0]),
    ],
)
def test_polyak_update_supports_closed_interval_endpoints(tau, expected):
    updated = rlt_td3.polyak_update(
        {"a": jnp.asarray([2.0], dtype=jnp.bfloat16)},
        {"a": jnp.asarray([6.0], dtype=jnp.bfloat16)},
        tau=tau,
    )

    assert updated["a"].dtype == jnp.float32
    np.testing.assert_array_equal(updated["a"], jnp.asarray(expected, dtype=jnp.float32))


def test_array_primitives_are_jittable_with_static_hyperparameters():
    reference = jnp.ones((2, 4, 2), dtype=jnp.float32)
    reward = terminal = q_values = jnp.zeros((2, 1), dtype=jnp.float32)
    params = {"a": jnp.ones((2,), dtype=jnp.float32)}

    dropped, mask = jax.jit(lambda key, value: rlt_td3.whole_reference_dropout(key, value, rate=0.5))(
        jax.random.key(20), reference
    )
    noise = jax.jit(
        lambda key: rlt_td3.sample_ar1_noise(
            key,
            batch_size=2,
            horizon=4,
            action_dim=2,
            sigma=0.1,
            rho=0.5,
        )
    )(jax.random.key(21))
    action = jax.jit(lambda key, value: rlt_td3.sample_gaussian_action(key, value, sigma=0.1, rho=0.5))(
        jax.random.key(22), reference
    )
    target = jax.jit(lambda r, t, q1, q2: rlt_td3.compute_td_target(r, t, q1, q2, gamma=0.99))(
        reward, terminal, q_values, q_values
    )
    bc_loss = jax.jit(rlt_td3.behavior_cloning_loss)(reference, reference)
    updated = jax.jit(lambda target_tree, online_tree: rlt_td3.polyak_update(target_tree, online_tree, tau=0.5))(
        params, params
    )

    assert dropped.shape == reference.shape
    assert mask.shape == (2,)
    assert noise.shape == reference.shape
    assert action.shape == reference.shape
    assert target.shape == reward.shape
    assert bc_loss.shape == ()
    assert updated["a"].dtype == jnp.float32


def test_hyperparameters_are_keyword_only_in_public_helpers():
    key = jax.random.key(23)
    action = jnp.zeros((2, 4, 2), dtype=jnp.float32)
    scalar = jnp.zeros((2, 1), dtype=jnp.float32)
    tree = {"a": jnp.zeros((1,), dtype=jnp.float32)}

    with pytest.raises(TypeError, match="positional"):
        rlt_td3.whole_reference_dropout(key, action, 0.5)
    with pytest.raises(TypeError, match="positional"):
        rlt_td3.sample_ar1_noise(key, 2, 4, 2, 0.1, 0.5)
    with pytest.raises(TypeError, match="positional"):
        rlt_td3.sample_gaussian_action(key, action, 0.1, 0.5)
    with pytest.raises(TypeError, match="positional"):
        rlt_td3.compute_td_target(scalar, scalar, scalar, scalar, 0.99)
    with pytest.raises(TypeError, match="positional"):
        rlt_td3.polyak_update(tree, tree, 0.5)


def test_critic_loss_is_finite_fp32_and_reports_the_exact_metrics():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, _, target_actor_params, q_params, target_q_params = _loss_modules_and_params(
        network_config,
        batch,
    )

    loss, metrics = rlt_td3.critic_loss(
        q_params,
        target_q_params=target_q_params,
        target_actor_params=target_actor_params,
        batch=batch,
        rng=jax.random.key(32),
        actor=actor,
        critic=critic,
        config=rlt_td3.TD3Config(),
    )
    config = rlt_td3.TD3Config()
    next_mean = actor.apply(
        {"params": target_actor_params},
        batch.next_z_rl,
        batch.next_state_norm,
        batch.next_vla_reference,
    )
    next_action = rlt_td3.sample_gaussian_action(
        jax.random.key(32),
        next_mean,
        sigma=config.noise_sigma,
        rho=config.noise_rho,
    )
    next_q1 = critic.apply(
        {"params": target_q_params["q1"]},
        batch.next_z_rl,
        batch.next_state_norm,
        next_action,
    )
    next_q2 = critic.apply(
        {"params": target_q_params["q2"]},
        batch.next_z_rl,
        batch.next_state_norm,
        next_action,
    )
    target = rlt_td3.compute_td_target(
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
    error1 = q1 - target
    error2 = q2 - target
    absolute_td = jnp.maximum(jnp.abs(error1), jnp.abs(error2))
    q1_loss = jnp.mean(jnp.square(error1), dtype=jnp.float32)
    q2_loss = jnp.mean(jnp.square(error2), dtype=jnp.float32)
    expected_metrics = {
        "critic/loss": q1_loss + q2_loss,
        "critic/q1_loss": q1_loss,
        "critic/q2_loss": q2_loss,
        "critic/q1_mean": jnp.mean(q1, dtype=jnp.float32),
        "critic/q1_std": jnp.std(q1, dtype=jnp.float32),
        "critic/q1_min": jnp.min(q1),
        "critic/q1_max": jnp.max(q1),
        "critic/q2_mean": jnp.mean(q2, dtype=jnp.float32),
        "critic/q2_std": jnp.std(q2, dtype=jnp.float32),
        "critic/q2_min": jnp.min(q2),
        "critic/q2_max": jnp.max(q2),
        "critic/target_mean": jnp.mean(target, dtype=jnp.float32),
        "critic/target_std": jnp.std(target, dtype=jnp.float32),
        "critic/target_min": jnp.min(target),
        "critic/target_max": jnp.max(target),
        "critic/td_mean": jnp.mean(absolute_td, dtype=jnp.float32),
        "critic/td_rms": jnp.sqrt(jnp.mean(jnp.square(absolute_td), dtype=jnp.float32)),
        "critic/td_max": jnp.max(absolute_td),
    }

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert jnp.isfinite(loss)
    _assert_scalar_fp32_metrics(
        metrics,
        {
            "critic/loss",
            "critic/q1_loss",
            "critic/q2_loss",
            "critic/q1_mean",
            "critic/q1_std",
            "critic/q1_min",
            "critic/q1_max",
            "critic/q2_mean",
            "critic/q2_std",
            "critic/q2_min",
            "critic/q2_max",
            "critic/target_mean",
            "critic/target_std",
            "critic/target_min",
            "critic/target_max",
            "critic/td_mean",
            "critic/td_rms",
            "critic/td_max",
        },
    )
    np.testing.assert_allclose(
        loss,
        metrics["critic/q1_loss"] + metrics["critic/q2_loss"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(loss, metrics["critic/loss"])
    _assert_metric_values(metrics, expected_metrics)


def test_terminal_critic_loss_is_independent_of_target_actor_and_critics():
    network_config = _network_config()
    batch = _loss_batch(network_config).replace(terminal=jnp.ones((3, 1), dtype=jnp.float32))
    actor, critic, _, target_actor_params, q_params, target_q_params = _loss_modules_and_params(
        network_config,
        batch,
    )
    changed_target_actor_params = jax.tree_util.tree_map(lambda value: value + 3.0, target_actor_params)
    changed_target_q_params = jax.tree_util.tree_map(lambda value: value - 2.0, target_q_params)

    first = rlt_td3.critic_loss(
        q_params,
        target_q_params=target_q_params,
        target_actor_params=target_actor_params,
        batch=batch,
        rng=jax.random.key(33),
        actor=actor,
        critic=critic,
        config=rlt_td3.TD3Config(),
    )
    changed = rlt_td3.critic_loss(
        q_params,
        target_q_params=changed_target_q_params,
        target_actor_params=changed_target_actor_params,
        batch=batch,
        rng=jax.random.key(33),
        actor=actor,
        critic=critic,
        config=rlt_td3.TD3Config(),
    )

    np.testing.assert_array_equal(first[0], changed[0])
    for key in first[1]:
        np.testing.assert_array_equal(first[1][key], changed[1][key])


def test_actor_loss_conditions_on_vla_reference_not_bc_anchor():
    network_config = _network_config()
    action_shape = (3, network_config.action_horizon, network_config.action_dim)
    batch = _loss_batch(network_config).replace(
        vla_reference=jnp.full(action_shape, 0.75, dtype=jnp.float32),
        bc_anchor=jnp.full(action_shape, -0.25, dtype=jnp.float32),
    )
    actor, critic, actor_params, _, q_params, _ = _loss_modules_and_params(network_config, batch)
    config = rlt_td3.TD3Config(reference_dropout_rate=0.0, noise_sigma=0.0, beta=0.0)

    loss, metrics = rlt_td3.actor_loss(
        actor_params,
        q1_params=q_params["q1"],
        batch=batch,
        rng=jax.random.key(34),
        actor=actor,
        critic=critic,
        config=config,
    )
    expected_mean = actor.apply(
        {"params": actor_params},
        batch.z_rl,
        batch.state_norm,
        batch.vla_reference,
    )
    expected_q1 = critic.apply(
        {"params": q_params["q1"]},
        batch.z_rl,
        batch.state_norm,
        expected_mean,
    )
    expected_q_term = -jnp.mean(expected_q1, dtype=jnp.float32)
    expected_mean_rms = jnp.sqrt(jnp.mean(jnp.square(expected_mean), dtype=jnp.float32))

    np.testing.assert_allclose(loss, expected_q_term, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metrics["actor/q_term"], expected_q_term, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metrics["actor/mean_rms"], expected_mean_rms, rtol=1e-6, atol=1e-6)

    changed_batch = batch.replace(vla_reference=jnp.full(action_shape, -0.5, dtype=jnp.float32))
    changed_loss, changed_metrics = rlt_td3.actor_loss(
        actor_params,
        q1_params=q_params["q1"],
        batch=changed_batch,
        rng=jax.random.key(34),
        actor=actor,
        critic=critic,
        config=config,
    )
    changed_expected_mean = actor.apply(
        {"params": actor_params},
        changed_batch.z_rl,
        changed_batch.state_norm,
        changed_batch.vla_reference,
    )

    assert not jnp.array_equal(expected_mean, changed_expected_mean)
    assert not jnp.array_equal(loss, changed_loss)
    assert not jnp.array_equal(metrics["actor/mean_rms"], changed_metrics["actor/mean_rms"])


def test_actor_loss_drops_whole_vla_reference_without_erasing_bc_supervision():
    network_config = _network_config()
    batch = _loss_batch(network_config).replace(bc_anchor=jnp.ones((3, 4, 2), dtype=jnp.float32))
    actor, critic, actor_params, _, q_params, _ = _loss_modules_and_params(network_config, batch)
    config = rlt_td3.TD3Config(reference_dropout_rate=1.0, beta=0.7)
    rng = jax.random.key(35)

    loss, metrics = rlt_td3.actor_loss(
        actor_params,
        q1_params=q_params["q1"],
        batch=batch,
        rng=rng,
        actor=actor,
        critic=critic,
        config=config,
    )
    _, action_key = jax.random.split(rng)
    expected_mean = actor.apply(
        {"params": actor_params},
        batch.z_rl,
        batch.state_norm,
        jnp.zeros_like(batch.vla_reference),
    )
    expected_sample = rlt_td3.sample_gaussian_action(
        action_key,
        expected_mean,
        sigma=config.noise_sigma,
        rho=config.noise_rho,
    )
    expected_q1 = critic.apply(
        {"params": q_params["q1"]},
        batch.z_rl,
        batch.state_norm,
        expected_sample,
    )
    expected_q_term = -jnp.mean(expected_q1, dtype=jnp.float32)
    expected_bc = rlt_td3.behavior_cloning_loss(expected_mean, batch.bc_anchor)

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert jnp.isfinite(loss)
    _assert_scalar_fp32_metrics(
        metrics,
        {
            "actor/loss",
            "actor/q_term",
            "actor/bc_loss",
            "actor/reference_drop_fraction",
            "actor/mean_rms",
            "actor/sample_rms",
            "actor/anchor_l1",
            "actor/vla_l1",
            "actor/saturation_fraction",
        },
    )
    np.testing.assert_array_equal(metrics["actor/reference_drop_fraction"], jnp.asarray(1.0, jnp.float32))
    np.testing.assert_allclose(metrics["actor/bc_loss"], expected_bc, rtol=1e-6, atol=1e-6)
    assert expected_bc > 0.0
    np.testing.assert_allclose(metrics["actor/q_term"], expected_q_term, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        loss,
        expected_q_term + jnp.asarray(config.beta, jnp.float32) * expected_bc,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(loss, metrics["actor/loss"])


def test_actor_loss_metrics_use_distinct_split_keys_and_exact_formulas():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, actor_params, _, q_params, _ = _loss_modules_and_params(network_config, batch)
    config = rlt_td3.TD3Config(
        reference_dropout_rate=0.5,
        noise_sigma=0.2,
        noise_rho=-0.3,
        beta=0.4,
    )
    rng = jax.random.key(41)
    dropout_key, action_key = jax.random.split(rng)

    loss, metrics = rlt_td3.actor_loss(
        actor_params,
        q1_params=q_params["q1"],
        batch=batch,
        rng=rng,
        actor=actor,
        critic=critic,
        config=config,
    )
    actor_reference, dropped_mask = rlt_td3.whole_reference_dropout(
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
    sampled_action = rlt_td3.sample_gaussian_action(
        action_key,
        mean,
        sigma=config.noise_sigma,
        rho=config.noise_rho,
    )
    q1 = critic.apply(
        {"params": q_params["q1"]},
        batch.z_rl,
        batch.state_norm,
        sampled_action,
    ).astype(jnp.float32)
    q_term = -jnp.mean(q1, dtype=jnp.float32)
    bc = rlt_td3.behavior_cloning_loss(mean, batch.bc_anchor)
    expected_metrics = {
        "actor/loss": q_term + jnp.asarray(config.beta, jnp.float32) * bc,
        "actor/q_term": q_term,
        "actor/bc_loss": bc,
        "actor/reference_drop_fraction": jnp.mean(dropped_mask.astype(jnp.float32), dtype=jnp.float32),
        "actor/mean_rms": jnp.sqrt(jnp.mean(jnp.square(mean), dtype=jnp.float32)),
        "actor/sample_rms": jnp.sqrt(jnp.mean(jnp.square(sampled_action), dtype=jnp.float32)),
        "actor/anchor_l1": jnp.mean(
            jnp.abs(mean - jnp.clip(batch.bc_anchor, -1.0, 1.0)),
            dtype=jnp.float32,
        ),
        "actor/vla_l1": jnp.mean(
            jnp.abs(mean - jnp.clip(batch.vla_reference, -1.0, 1.0)),
            dtype=jnp.float32,
        ),
        "actor/saturation_fraction": jnp.mean((jnp.abs(mean) >= 0.99).astype(jnp.float32), dtype=jnp.float32),
    }

    _assert_metric_values(metrics, expected_metrics)
    np.testing.assert_allclose(loss, expected_metrics["actor/loss"], rtol=1e-6, atol=1e-6)

    wrong_reference, wrong_mask = rlt_td3.whole_reference_dropout(
        action_key,
        batch.vla_reference,
        rate=config.reference_dropout_rate,
    )
    wrong_mean = actor.apply(
        {"params": actor_params},
        batch.z_rl,
        batch.state_norm,
        wrong_reference,
    )
    wrong_sample = rlt_td3.sample_gaussian_action(
        dropout_key,
        mean,
        sigma=config.noise_sigma,
        rho=config.noise_rho,
    )
    np.testing.assert_array_equal(dropped_mask, jnp.ones_like(dropped_mask))
    np.testing.assert_array_equal(wrong_mask, jnp.zeros_like(wrong_mask))
    assert not jnp.array_equal(metrics["actor/mean_rms"], jnp.sqrt(jnp.mean(jnp.square(wrong_mean))))
    assert not jnp.array_equal(metrics["actor/sample_rms"], jnp.sqrt(jnp.mean(jnp.square(wrong_sample))))


def test_actor_loss_has_finite_nonzero_gradients_through_sampled_action():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, actor_params, _, q_params, _ = _loss_modules_and_params(network_config, batch)

    gradients = jax.grad(
        lambda params: rlt_td3.actor_loss(
            params,
            q1_params=q_params["q1"],
            batch=batch,
            rng=jax.random.key(35),
            actor=actor,
            critic=critic,
            config=rlt_td3.TD3Config(reference_dropout_rate=0.0, beta=0.0),
        )[0]
    )(actor_params)
    leaves = jax.tree_util.tree_leaves(gradients)

    assert leaves
    assert all(leaf.dtype == jnp.float32 for leaf in leaves)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0.0) for leaf in leaves)


def test_actor_loss_explicitly_stops_every_q1_gradient_leaf():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, actor_params, _, q_params, _ = _loss_modules_and_params(network_config, batch)

    q1_gradients = jax.grad(
        lambda q1: rlt_td3.actor_loss(
            actor_params,
            q1_params=q1,
            batch=batch,
            rng=jax.random.key(36),
            actor=actor,
            critic=critic,
            config=rlt_td3.TD3Config(),
        )[0]
    )(q_params["q1"])

    _assert_exact_zero_tree(q1_gradients)


def test_critic_loss_stops_target_branches_but_not_online_critics():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, _, target_actor_params, q_params, target_q_params = _loss_modules_and_params(
        network_config,
        batch,
    )
    config = rlt_td3.TD3Config()

    target_q_gradients, target_actor_gradients = jax.grad(
        lambda target_q, target_actor: rlt_td3.critic_loss(
            q_params,
            target_q_params=target_q,
            target_actor_params=target_actor,
            batch=batch,
            rng=jax.random.key(37),
            actor=actor,
            critic=critic,
            config=config,
        )[0],
        argnums=(0, 1),
    )(target_q_params, target_actor_params)
    online_gradients = jax.grad(
        lambda online_q: rlt_td3.critic_loss(
            online_q,
            target_q_params=target_q_params,
            target_actor_params=target_actor_params,
            batch=batch,
            rng=jax.random.key(37),
            actor=actor,
            critic=critic,
            config=config,
        )[0]
    )(q_params)

    _assert_exact_zero_tree(target_q_gradients)
    _assert_exact_zero_tree(target_actor_gradients)
    online_leaves = jax.tree_util.tree_leaves(online_gradients)
    assert online_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in online_leaves)
    assert any(jnp.any(leaf != 0.0) for leaf in online_leaves)


def test_actor_and_critic_losses_are_deterministic_and_jittable():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, actor_params, target_actor_params, q_params, target_q_params = _loss_modules_and_params(
        network_config,
        batch,
    )
    config = rlt_td3.TD3Config()

    compiled_critic = jax.jit(
        lambda online_q, target_q, target_actor, transition, key: rlt_td3.critic_loss(
            online_q,
            target_q_params=target_q,
            target_actor_params=target_actor,
            batch=transition,
            rng=key,
            actor=actor,
            critic=critic,
            config=config,
        )
    )
    compiled_actor = jax.jit(
        lambda online_actor, q1, transition, key: rlt_td3.actor_loss(
            online_actor,
            q1_params=q1,
            batch=transition,
            rng=key,
            actor=actor,
            critic=critic,
            config=config,
        )
    )

    critic_first = compiled_critic(
        q_params,
        target_q_params,
        target_actor_params,
        batch,
        jax.random.key(38),
    )
    critic_repeated = compiled_critic(
        q_params,
        target_q_params,
        target_actor_params,
        batch,
        jax.random.key(38),
    )
    actor_first = compiled_actor(actor_params, q_params["q1"], batch, jax.random.key(39))
    actor_repeated = compiled_actor(actor_params, q_params["q1"], batch, jax.random.key(39))

    np.testing.assert_array_equal(critic_first[0], critic_repeated[0])
    np.testing.assert_array_equal(actor_first[0], actor_repeated[0])
    for first_metrics, repeated_metrics in ((critic_first[1], critic_repeated[1]), (actor_first[1], actor_repeated[1])):
        for key in first_metrics:
            np.testing.assert_array_equal(first_metrics[key], repeated_metrics[key])


def test_actor_loss_uses_the_supplied_q1_path():
    network_config = _network_config()
    batch = _loss_batch(network_config)
    actor, critic, actor_params, _, q_params, _ = _loss_modules_and_params(network_config, batch)
    shifted_q1 = jax.tree_util.tree_map(
        lambda value: value + 0.5,
        q_params["q1"],
    )

    baseline = rlt_td3.actor_loss(
        actor_params,
        q1_params=q_params["q1"],
        batch=batch,
        rng=jax.random.key(40),
        actor=actor,
        critic=critic,
        config=rlt_td3.TD3Config(),
    )
    shifted = rlt_td3.actor_loss(
        actor_params,
        q1_params=shifted_q1,
        batch=batch,
        rng=jax.random.key(40),
        actor=actor,
        critic=critic,
        config=rlt_td3.TD3Config(),
    )

    assert not jnp.array_equal(baseline[1]["actor/q_term"], shifted[1]["actor/q_term"])
