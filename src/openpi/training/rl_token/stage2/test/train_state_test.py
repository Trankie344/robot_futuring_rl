import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import td3 as rlt_td3
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state

# Flax 0.10.2 probes ShapeDtypeStruct values with jnp.shape during parameter application.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:shape requires ndarray or scalar arguments.*:DeprecationWarning:flax\.core\.scope"
)


def _network_config(**overrides) -> rlt_actor_critic.RLTActorCriticConfig:
    values = {
        "z_dim": 5,
        "state_dim": 3,
        "action_horizon": 4,
        "action_dim": 2,
        "actor_state_proj_dim": 4,
        "actor_reference_proj_dim": 4,
        "critic_state_proj_dim": 4,
        "critic_action_proj_dim": 4,
        "actor_hidden_dims": (4, 4, 4),
        "critic_hidden_dims": (4, 4, 4),
        "compute_dtype": "float32",
    }
    values.update(overrides)
    return rlt_actor_critic.RLTActorCriticConfig(**values)


def _modules(config=None):
    config = config or _network_config()
    return rlt_actor_critic.RLTActor(config), rlt_actor_critic.RLTCritic(config)


def _initialized(config=None, *, key_seed: int = 0, td3_config=None):
    actor, critic = _modules(config)
    return (
        actor,
        critic,
        *rlt_td3_state.initialize_train_state(
            actor,
            critic,
            td3_config or rlt_td3.TD3Config(),
            jax.random.key(key_seed),
        ),
    )


def _assert_tree_equal(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if hasattr(actual_leaf, "dtype") and jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(actual_leaf),
                jax.random.key_data(expected_leaf),
            )
        else:
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _assert_all_floating_leaves_fp32(tree):
    floating_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    assert floating_leaves
    assert all(leaf.dtype == jnp.float32 for leaf in floating_leaves)


def _assert_tree_allclose(actual, expected, *, rtol=1e-6, atol=1e-7):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if hasattr(actual_leaf, "dtype") and jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(actual_leaf),
                jax.random.key_data(expected_leaf),
            )
        elif hasattr(actual_leaf, "dtype") and jnp.issubdtype(actual_leaf.dtype, jnp.inexact):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _trees_differ(actual, expected, *, rtol=1e-6, atol=1e-8):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    return any(
        not np.allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol)
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        )
    )


def test_initialize_train_state_has_exact_frozen_pytree_contract():
    rng = jax.random.key(4)
    actor, critic = _modules()

    state, actor_tx, critic_tx = rlt_td3_state.initialize_train_state(
        actor,
        critic,
        rlt_td3.TD3Config(),
        rng,
    )

    assert dataclasses.is_dataclass(state)
    assert [field.name for field in dataclasses.fields(state)] == [
        "critic_step",
        "round_critic_step",
        "rng",
        "actor_params",
        "q_params",
        "target_actor_params",
        "target_q_params",
        "actor_opt_state",
        "critic_opt_state",
    ]
    assert isinstance(actor_tx, optax.GradientTransformation)
    assert isinstance(critic_tx, optax.GradientTransformation)
    assert state.critic_step.shape == ()
    assert state.critic_step.dtype == jnp.int32
    assert state.round_critic_step.shape == ()
    assert state.round_critic_step.dtype == jnp.int32
    np.testing.assert_array_equal(state.critic_step, jnp.asarray(0, dtype=jnp.int32))
    np.testing.assert_array_equal(state.round_critic_step, jnp.asarray(0, dtype=jnp.int32))
    assert state.rng.shape == rng.shape
    assert state.rng.dtype == rng.dtype
    np.testing.assert_array_equal(
        jax.random.key_data(state.rng),
        jax.random.key_data(jax.random.split(rng, 4)[3]),
    )
    assert set(state.q_params) == {"q1", "q2"}
    assert set(state.target_q_params) == {"q1", "q2"}
    assert jax.tree_util.tree_structure(state.q_params["q1"]) == jax.tree_util.tree_structure(state.q_params["q2"])
    assert any(
        not np.array_equal(q1_leaf, q2_leaf)
        for q1_leaf, q2_leaf in zip(
            jax.tree_util.tree_leaves(state.q_params["q1"]),
            jax.tree_util.tree_leaves(state.q_params["q2"]),
            strict=True,
        )
    )
    _assert_tree_equal(state.target_actor_params, state.actor_params)
    _assert_tree_equal(state.target_q_params, state.q_params)
    assert all(
        target_leaf is not online_leaf
        for target_leaf, online_leaf in zip(
            jax.tree_util.tree_leaves(state.target_actor_params),
            jax.tree_util.tree_leaves(state.actor_params),
            strict=True,
        )
    )
    assert all(
        target_leaf is not online_leaf
        for target_leaf, online_leaf in zip(
            jax.tree_util.tree_leaves(state.target_q_params),
            jax.tree_util.tree_leaves(state.q_params),
            strict=True,
        )
    )
    _assert_all_floating_leaves_fp32(
        (
            state.actor_params,
            state.q_params,
            state.target_actor_params,
            state.target_q_params,
            state.actor_opt_state,
            state.critic_opt_state,
        )
    )
    assert jax.tree_util.tree_leaves(state)
    assert state.replace(critic_step=jnp.asarray(9, jnp.int32)).critic_step == 9
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.critic_step = jnp.asarray(1, dtype=jnp.int32)


def test_initialize_train_state_is_exactly_reproducible_from_rng():
    first = _initialized(key_seed=5)[2]
    repeated = _initialized(key_seed=5)[2]
    changed = _initialized(key_seed=6)[2]

    _assert_tree_equal(first, repeated)
    assert any(
        not np.array_equal(first_leaf, changed_leaf)
        for first_leaf, changed_leaf in zip(
            jax.tree_util.tree_leaves(first.actor_params),
            jax.tree_util.tree_leaves(changed.actor_params),
            strict=True,
        )
    )


def test_initialize_train_state_rejects_actor_critic_config_mismatch():
    actor = rlt_actor_critic.RLTActor(_network_config())
    critic = rlt_actor_critic.RLTCritic(_network_config(critic_hidden_dims=(5, 4, 4)))

    with pytest.raises(ValueError, match="Actor and critic configs must match exactly"):
        rlt_td3_state.initialize_train_state(
            actor,
            critic,
            rlt_td3.TD3Config(),
            jax.random.key(7),
        )


def test_initialize_train_state_revalidates_network_and_td3_configs():
    bad_network_config = _network_config()
    object.__setattr__(bad_network_config, "z_dim", 0)
    actor = rlt_actor_critic.RLTActor(bad_network_config)
    critic = rlt_actor_critic.RLTCritic(bad_network_config)

    with pytest.raises(ValueError, match="z_dim must be a positive integer"):
        rlt_td3_state.initialize_train_state(
            actor,
            critic,
            rlt_td3.TD3Config(),
            jax.random.key(8),
        )

    good_actor, good_critic = _modules()
    bad_td3_config = rlt_td3.TD3Config()
    object.__setattr__(bad_td3_config, "actor_lr", 0.0)
    with pytest.raises(ValueError, match="actor_lr must be positive"):
        rlt_td3_state.initialize_train_state(
            good_actor,
            good_critic,
            bad_td3_config,
            jax.random.key(8),
        )


def test_make_optimizers_revalidates_config_and_returns_independent_chains():
    config = rlt_td3.TD3Config(actor_lr=2e-4, critic_lr=7e-4)
    actor_tx, critic_tx = rlt_td3_state.make_optimizers(config)
    params = {"value": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}

    actor_state = actor_tx.init(params)
    critic_state = critic_tx.init(params)

    assert isinstance(actor_tx, optax.GradientTransformation)
    assert isinstance(critic_tx, optax.GradientTransformation)
    _assert_all_floating_leaves_fp32((actor_state, critic_state))
    assert actor_state is not critic_state

    bad_config = rlt_td3.TD3Config()
    object.__setattr__(bad_config, "critic_grad_clip", 0.0)
    with pytest.raises(ValueError, match="critic_grad_clip must be positive"):
        rlt_td3_state.make_optimizers(bad_config)


def test_make_optimizers_matches_independent_two_step_clip_then_adam_contract():
    config = rlt_td3.TD3Config(
        actor_lr=0.25,
        critic_lr=0.125,
        actor_grad_clip=0.05,
        critic_grad_clip=0.08,
        adam_b1=0.37,
        adam_b2=0.83,
        adam_eps=2e-4,
    )
    actor_tx, critic_tx = rlt_td3_state.make_optimizers(config)

    def required_chain(*, learning_rate, grad_clip):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(
                learning_rate,
                b1=config.adam_b1,
                b2=config.adam_b2,
                eps=config.adam_eps,
                mu_dtype=jnp.float32,
            ),
        )

    def run_two_steps(tx, initial_params, gradient_sequence):
        params = initial_params
        opt_state = tx.init(params)
        update_history = []
        for gradients in gradient_sequence:
            updates, opt_state = tx.update(gradients, opt_state, params)
            params = optax.apply_updates(params, updates)
            update_history.append(updates)
        return params, opt_state, update_history

    actor_params = {"w": jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float32)}
    actor_gradients = [
        {"w": jnp.asarray([12.0, -0.25, 4.0], dtype=jnp.float32)},
        {"w": jnp.asarray([-1.0, 20.0, -3.0], dtype=jnp.float32)},
    ]
    critic_params = {
        "q1": {"w": jnp.asarray([0.5, -1.5], dtype=jnp.float32)},
        "q2": {"w": jnp.asarray([-0.75, 2.0], dtype=jnp.float32)},
    }
    critic_gradients = [
        {
            "q1": {"w": jnp.asarray([30.0, -2.0], dtype=jnp.float32)},
            "q2": {"w": jnp.asarray([0.01, -100.0], dtype=jnp.float32)},
        },
        {
            "q1": {"w": jnp.asarray([-0.5, 25.0], dtype=jnp.float32)},
            "q2": {"w": jnp.asarray([80.0, 0.2], dtype=jnp.float32)},
        },
    ]

    actual_actor = run_two_steps(actor_tx, actor_params, actor_gradients)
    expected_actor = run_two_steps(
        required_chain(
            learning_rate=config.actor_lr,
            grad_clip=config.actor_grad_clip,
        ),
        actor_params,
        actor_gradients,
    )
    actual_critic = run_two_steps(critic_tx, critic_params, critic_gradients)
    expected_critic = run_two_steps(
        required_chain(
            learning_rate=config.critic_lr,
            grad_clip=config.critic_grad_clip,
        ),
        critic_params,
        critic_gradients,
    )

    _assert_tree_equal(actual_actor, expected_actor)
    _assert_tree_equal(actual_critic, expected_critic)
    _assert_all_floating_leaves_fp32((actual_actor, actual_critic))

    adam_then_clip = optax.chain(
        optax.adam(
            config.actor_lr,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
            mu_dtype=jnp.float32,
        ),
        optax.clip_by_global_norm(config.actor_grad_clip),
    )
    wrong_order_actor = run_two_steps(adam_then_clip, actor_params, actor_gradients)
    assert _trees_differ(actual_actor[0], wrong_order_actor[0])

    separate_q1_tx = required_chain(
        learning_rate=config.critic_lr,
        grad_clip=config.critic_grad_clip,
    )
    separate_q2_tx = required_chain(
        learning_rate=config.critic_lr,
        grad_clip=config.critic_grad_clip,
    )
    separate_q1_params = critic_params["q1"]
    separate_q2_params = critic_params["q2"]
    separate_q1_state = separate_q1_tx.init(separate_q1_params)
    separate_q2_state = separate_q2_tx.init(separate_q2_params)
    for gradients in critic_gradients:
        q1_updates, separate_q1_state = separate_q1_tx.update(
            gradients["q1"],
            separate_q1_state,
            separate_q1_params,
        )
        q2_updates, separate_q2_state = separate_q2_tx.update(
            gradients["q2"],
            separate_q2_state,
            separate_q2_params,
        )
        separate_q1_params = optax.apply_updates(separate_q1_params, q1_updates)
        separate_q2_params = optax.apply_updates(separate_q2_params, q2_updates)
    separate_critic_params = {
        "q1": separate_q1_params,
        "q2": separate_q2_params,
    }
    assert _trees_differ(actual_critic[0], separate_critic_params)


@pytest.mark.parametrize(
    ("chunk_equivalents", "expected_critic", "expected_actor"),
    [
        (0, 0, 0),
        (1, 5, 2),
        (20, 100, 50),
        (137, 685, 342),
    ],
)
def test_round_update_budget_exact_contract(chunk_equivalents, expected_critic, expected_actor):
    critic_updates, actor_updates = rlt_td3_state.round_update_budget(
        chunk_equivalents,
        utd_ratio=5,
        policy_delay=2,
    )

    assert type(critic_updates) is int
    assert type(actor_updates) is int
    assert (critic_updates, actor_updates) == (expected_critic, expected_actor)


def test_round_update_budget_accepts_integer_like_values_and_returns_builtin_ints():
    result = rlt_td3_state.round_update_budget(
        np.int64(3),
        utd_ratio=np.int32(4),
        policy_delay=np.int64(3),
    )

    assert result == (12, 4)
    assert all(type(value) is int for value in result)


@pytest.mark.parametrize("chunk_equivalents", [-1, -3, 1.0, True, False, np.bool_(1), "1", None])
def test_round_update_budget_rejects_invalid_chunk_equivalents(chunk_equivalents):
    with pytest.raises(ValueError, match="chunk_equivalents must be a nonnegative integer"):
        rlt_td3_state.round_update_budget(chunk_equivalents, utd_ratio=5, policy_delay=2)


@pytest.mark.parametrize("name", ["utd_ratio", "policy_delay"])
@pytest.mark.parametrize("value", [0, -1, 1.0, True, False, np.bool_(0), "1", None])
def test_round_update_budget_rejects_invalid_update_counts(name, value):
    kwargs = {"utd_ratio": 5, "policy_delay": 2}
    kwargs[name] = value

    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        rlt_td3_state.round_update_budget(1, **kwargs)


def test_start_new_round_resets_only_round_counter():
    state = _initialized(key_seed=9)[2].replace(
        critic_step=jnp.asarray(17, dtype=jnp.int32),
        round_critic_step=jnp.asarray(6, dtype=jnp.int32),
    )

    reset = rlt_td3_state.start_new_round(state)

    assert reset.round_critic_step.shape == ()
    assert reset.round_critic_step.dtype == jnp.int32
    np.testing.assert_array_equal(reset.round_critic_step, jnp.asarray(0, dtype=jnp.int32))
    np.testing.assert_array_equal(reset.critic_step, state.critic_step)
    for field in dataclasses.fields(state):
        if field.name != "round_critic_step":
            _assert_tree_equal(getattr(reset, field.name), getattr(state, field.name))


def test_saved_round_counter_is_preserved_without_explicit_reset():
    state = _initialized(key_seed=10)[2].replace(round_critic_step=jnp.asarray(4, dtype=jnp.int32))
    copied = jax.tree_util.tree_map(lambda value: value, state)

    np.testing.assert_array_equal(copied.round_critic_step, jnp.asarray(4, dtype=jnp.int32))


_ZERO_ACTOR_METRIC_KEYS = {
    "actor/loss",
    "actor/q_term",
    "actor/bc_loss",
    "actor/reference_drop_fraction",
    "actor/mean_rms",
    "actor/sample_rms",
    "actor/anchor_l1",
    "actor/vla_l1",
    "actor/saturation_fraction",
    "actor/grad_norm",
    "actor/grad_clipped",
}
_ALL_METRIC_KEYS = _ZERO_ACTOR_METRIC_KEYS | {
    "actor/updated",
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
    "critic/grad_norm",
    "critic/grad_clipped",
}


def _transition_batch(
    config: rlt_actor_critic.RLTActorCriticConfig,
    *,
    batch_size: int = 3,
) -> rlt_td3.RLTTransitionBatch:
    keys = jax.random.split(jax.random.key(20), 8)
    z_shape = (batch_size, config.z_dim)
    state_shape = (batch_size, config.state_dim)
    action_shape = (batch_size, config.action_horizon, config.action_dim)
    return rlt_td3.RLTTransitionBatch(
        z_rl=jax.random.normal(keys[0], z_shape, dtype=jnp.float32),
        next_z_rl=jax.random.normal(keys[1], z_shape, dtype=jnp.float32),
        state_norm=jax.random.normal(keys[2], state_shape, dtype=jnp.float32),
        next_state_norm=jax.random.normal(keys[3], state_shape, dtype=jnp.float32),
        vla_reference=jnp.clip(
            0.25 * jax.random.normal(keys[4], action_shape, dtype=jnp.float32),
            -1.0,
            1.0,
        ),
        next_vla_reference=jnp.clip(
            0.25 * jax.random.normal(keys[5], action_shape, dtype=jnp.float32),
            -1.0,
            1.0,
        ),
        executed_action=jnp.clip(
            0.35 * jax.random.normal(keys[6], action_shape, dtype=jnp.float32),
            -1.0,
            1.0,
        ),
        bc_anchor=jnp.clip(
            0.5 + 0.2 * jax.random.normal(keys[7], action_shape, dtype=jnp.float32),
            -1.0,
            1.0,
        ),
        reward=jnp.asarray([[1.0], [-0.5], [0.25]], dtype=jnp.float32)[:batch_size],
        terminal=jnp.asarray([[0.0], [1.0], [0.0]], dtype=jnp.float32)[:batch_size],
    )


def _training_setup(*, config=None, key_seed: int = 40):
    network_config = _network_config()
    actor = rlt_actor_critic.RLTActor(network_config)
    critic = rlt_actor_critic.RLTCritic(network_config)
    config = config or rlt_td3.TD3Config(
        reference_dropout_rate=0.0,
        noise_sigma=0.0,
    )
    state, actor_tx, critic_tx = rlt_td3_state.initialize_train_state(
        actor,
        critic,
        config,
        jax.random.key(key_seed),
    )
    return (
        network_config,
        actor,
        critic,
        config,
        state,
        actor_tx,
        critic_tx,
        _transition_batch(network_config),
    )


def _assert_tree_changed(actual, previous):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(previous)
    assert any(
        not np.array_equal(actual_leaf, previous_leaf)
        for actual_leaf, previous_leaf in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(previous),
            strict=True,
        )
        if not (hasattr(actual_leaf, "dtype") and jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key))
    )


def _assert_scalar_fp32_metrics(metrics):
    assert set(metrics) == _ALL_METRIC_KEYS
    for value in metrics.values():
        assert value.shape == ()
        assert value.dtype == jnp.float32
        assert jnp.isfinite(value)


def _adam_count(opt_state):
    return opt_state[1][0].count


def _run_step(state, batch, actor, critic, config, actor_tx, critic_tx):
    return rlt_td3_state.train_step(
        state,
        batch,
        actor,
        critic,
        config,
        actor_tx,
        critic_tx,
    )


def test_zero_actor_metrics_has_exact_scalar_fp32_zeros():
    metrics = rlt_td3_state._zero_actor_metrics()  # noqa: SLF001

    assert set(metrics) == _ZERO_ACTOR_METRIC_KEYS
    for value in metrics.values():
        assert value.shape == ()
        assert value.dtype == jnp.float32
        np.testing.assert_array_equal(value, jnp.asarray(0.0, dtype=jnp.float32))


def test_first_critic_step_skips_actor_and_changes_only_online_critics():
    _, actor, critic, config, state, actor_tx, critic_tx, batch = _training_setup()

    updated, metrics = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)

    np.testing.assert_array_equal(updated.critic_step, jnp.asarray(1, dtype=jnp.int32))
    np.testing.assert_array_equal(updated.round_critic_step, jnp.asarray(1, dtype=jnp.int32))
    np.testing.assert_array_equal(metrics["actor/updated"], jnp.asarray(0.0, dtype=jnp.float32))
    _assert_tree_equal(updated.actor_params, state.actor_params)
    _assert_tree_equal(updated.actor_opt_state, state.actor_opt_state)
    _assert_tree_equal(updated.target_actor_params, state.target_actor_params)
    _assert_tree_equal(updated.target_q_params, state.target_q_params)
    _assert_tree_changed(updated.q_params, state.q_params)
    _assert_tree_changed(updated.critic_opt_state, state.critic_opt_state)
    np.testing.assert_array_equal(_adam_count(updated.actor_opt_state), jnp.asarray(0, jnp.int32))
    np.testing.assert_array_equal(_adam_count(updated.critic_opt_state), jnp.asarray(1, jnp.int32))
    expected_rng = jax.random.split(state.rng, 3)[0]
    np.testing.assert_array_equal(
        jax.random.key_data(updated.rng),
        jax.random.key_data(expected_rng),
    )
    assert not jnp.array_equal(jax.random.key_data(updated.rng), jax.random.key_data(state.rng))
    for key in _ZERO_ACTOR_METRIC_KEYS:
        np.testing.assert_array_equal(metrics[key], jnp.asarray(0.0, dtype=jnp.float32))
    _assert_scalar_fp32_metrics(metrics)


def test_second_critic_step_updates_actor_and_both_targets_from_updated_online_params():
    _, actor, critic, config, state, actor_tx, critic_tx, batch = _training_setup()
    first, _ = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)

    second, metrics = _run_step(first, batch, actor, critic, config, actor_tx, critic_tx)

    np.testing.assert_array_equal(second.critic_step, jnp.asarray(2, dtype=jnp.int32))
    np.testing.assert_array_equal(second.round_critic_step, jnp.asarray(2, dtype=jnp.int32))
    np.testing.assert_array_equal(metrics["actor/updated"], jnp.asarray(1.0, dtype=jnp.float32))
    _assert_tree_changed(second.actor_params, first.actor_params)
    _assert_tree_changed(second.actor_opt_state, first.actor_opt_state)
    _assert_tree_changed(second.q_params, first.q_params)
    _assert_tree_changed(second.critic_opt_state, first.critic_opt_state)
    _assert_tree_changed(second.target_actor_params, first.target_actor_params)
    _assert_tree_changed(second.target_q_params, first.target_q_params)
    _assert_tree_allclose(
        second.target_actor_params,
        rlt_td3.polyak_update(
            first.target_actor_params,
            second.actor_params,
            tau=config.tau,
        ),
    )
    _assert_tree_allclose(
        second.target_q_params,
        rlt_td3.polyak_update(
            first.target_q_params,
            second.q_params,
            tau=config.tau,
        ),
    )
    np.testing.assert_array_equal(_adam_count(second.actor_opt_state), jnp.asarray(1, jnp.int32))
    np.testing.assert_array_equal(_adam_count(second.critic_opt_state), jnp.asarray(2, jnp.int32))
    assert metrics["actor/grad_norm"] > 0.0
    _assert_scalar_fp32_metrics(metrics)


def test_actor_delay_uses_new_round_counter_not_lifetime_counter():
    _, actor, critic, config, state, actor_tx, critic_tx, batch = _training_setup()
    resumed = state.replace(
        critic_step=jnp.asarray(101, dtype=jnp.int32),
        round_critic_step=jnp.asarray(7, dtype=jnp.int32),
    )
    new_round = rlt_td3_state.start_new_round(resumed)

    first, first_metrics = _run_step(new_round, batch, actor, critic, config, actor_tx, critic_tx)
    second, second_metrics = _run_step(first, batch, actor, critic, config, actor_tx, critic_tx)

    np.testing.assert_array_equal(first.critic_step, jnp.asarray(102, jnp.int32))
    np.testing.assert_array_equal(first.round_critic_step, jnp.asarray(1, jnp.int32))
    np.testing.assert_array_equal(first_metrics["actor/updated"], jnp.asarray(0.0, jnp.float32))
    np.testing.assert_array_equal(second.critic_step, jnp.asarray(103, jnp.int32))
    np.testing.assert_array_equal(second.round_critic_step, jnp.asarray(2, jnp.int32))
    np.testing.assert_array_equal(second_metrics["actor/updated"], jnp.asarray(1.0, jnp.float32))
    np.testing.assert_array_equal(_adam_count(first.actor_opt_state), jnp.asarray(0, jnp.int32))
    np.testing.assert_array_equal(_adam_count(second.actor_opt_state), jnp.asarray(1, jnp.int32))


def test_actor_adam_count_tracks_delayed_updates_across_four_critic_steps():
    _, actor, critic, config, state, actor_tx, critic_tx, batch = _training_setup()
    actor_counts = []
    update_flags = []

    for _ in range(4):
        state, metrics = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)
        actor_counts.append(int(_adam_count(state.actor_opt_state)))
        update_flags.append(float(metrics["actor/updated"]))

    assert actor_counts == [0, 1, 1, 2]
    assert update_flags == [0.0, 1.0, 0.0, 1.0]


def test_gradient_metrics_are_preclip_norms_and_actor_reads_updated_q1():
    config = rlt_td3.TD3Config(
        actor_grad_clip=1e-8,
        critic_grad_clip=1e-8,
        reference_dropout_rate=0.0,
        noise_sigma=0.0,
    )
    _, actor, critic, _, state, actor_tx, critic_tx, batch = _training_setup(
        config=config,
        key_seed=41,
    )
    first, first_metrics = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)
    next_rng, critic_rng, actor_rng = jax.random.split(first.rng, 3)

    (_, expected_critic_metrics), critic_grads = jax.value_and_grad(
        lambda q_params: rlt_td3.critic_loss(
            q_params,
            target_q_params=first.target_q_params,
            target_actor_params=first.target_actor_params,
            batch=batch,
            rng=critic_rng,
            actor=actor,
            critic=critic,
            config=config,
        ),
        has_aux=True,
    )(first.q_params)
    expected_critic_norm = optax.global_norm(critic_grads).astype(jnp.float32)
    critic_updates, expected_critic_opt_state = critic_tx.update(
        critic_grads,
        first.critic_opt_state,
        first.q_params,
    )
    expected_q_params = optax.apply_updates(first.q_params, critic_updates)
    (_, expected_actor_metrics), actor_grads = jax.value_and_grad(
        lambda actor_params: rlt_td3.actor_loss(
            actor_params,
            q1_params=expected_q_params["q1"],
            batch=batch,
            rng=actor_rng,
            actor=actor,
            critic=critic,
            config=config,
        ),
        has_aux=True,
    )(first.actor_params)
    expected_actor_norm = optax.global_norm(actor_grads).astype(jnp.float32)
    actor_updates, expected_actor_opt_state = actor_tx.update(
        actor_grads,
        first.actor_opt_state,
        first.actor_params,
    )
    expected_actor_params = optax.apply_updates(first.actor_params, actor_updates)

    second, metrics = _run_step(first, batch, actor, critic, config, actor_tx, critic_tx)

    assert expected_critic_norm > config.critic_grad_clip
    assert expected_actor_norm > config.actor_grad_clip
    np.testing.assert_allclose(metrics["critic/grad_norm"], expected_critic_norm, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(metrics["actor/grad_norm"], expected_actor_norm, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(metrics["critic/grad_clipped"], jnp.asarray(1.0, jnp.float32))
    np.testing.assert_array_equal(metrics["actor/grad_clipped"], jnp.asarray(1.0, jnp.float32))
    np.testing.assert_array_equal(first_metrics["critic/grad_clipped"], jnp.asarray(1.0, jnp.float32))
    _assert_tree_equal(second.q_params, expected_q_params)
    _assert_tree_equal(second.critic_opt_state, expected_critic_opt_state)
    _assert_tree_allclose(second.actor_params, expected_actor_params)
    _assert_tree_allclose(second.actor_opt_state, expected_actor_opt_state)
    np.testing.assert_array_equal(
        jax.random.key_data(second.rng),
        jax.random.key_data(next_rng),
    )
    for key, value in expected_critic_metrics.items():
        np.testing.assert_allclose(metrics[key], value, rtol=0.0, atol=0.0)
    for key, value in expected_actor_metrics.items():
        np.testing.assert_allclose(metrics[key], value, rtol=1e-6, atol=1e-7)


def test_stochastic_actor_update_uses_independent_actor_rng_and_persists_next_rng():
    config = rlt_td3.TD3Config(
        beta=0.4,
        noise_sigma=0.2,
        noise_rho=-0.35,
        reference_dropout_rate=0.5,
    )
    _, actor, critic, _, state, actor_tx, critic_tx, batch = _training_setup(
        config=config,
        key_seed=57,
    )
    first, _ = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)
    next_rng, critic_rng, actor_rng = jax.random.split(first.rng, 3)

    (_, expected_critic_metrics), critic_grads = jax.value_and_grad(
        lambda q_params: rlt_td3.critic_loss(
            q_params,
            target_q_params=first.target_q_params,
            target_actor_params=first.target_actor_params,
            batch=batch,
            rng=critic_rng,
            actor=actor,
            critic=critic,
            config=config,
        ),
        has_aux=True,
    )(first.q_params)
    critic_grad_norm = optax.global_norm(critic_grads).astype(jnp.float32)
    critic_updates, expected_critic_opt_state = critic_tx.update(
        critic_grads,
        first.critic_opt_state,
        first.q_params,
    )
    expected_q_params = optax.apply_updates(first.q_params, critic_updates)

    def actor_objective(actor_params, rng):
        return rlt_td3.actor_loss(
            actor_params,
            q1_params=expected_q_params["q1"],
            batch=batch,
            rng=rng,
            actor=actor,
            critic=critic,
            config=config,
        )

    (_, expected_actor_metrics), actor_grads = jax.value_and_grad(
        lambda actor_params: actor_objective(actor_params, actor_rng),
        has_aux=True,
    )(first.actor_params)
    actor_grad_norm = optax.global_norm(actor_grads).astype(jnp.float32)
    actor_updates, expected_actor_opt_state = actor_tx.update(
        actor_grads,
        first.actor_opt_state,
        first.actor_params,
    )
    expected_actor_params = optax.apply_updates(first.actor_params, actor_updates)
    expected_target_actor_params = rlt_td3.polyak_update(
        first.target_actor_params,
        expected_actor_params,
        tau=config.tau,
    )
    expected_target_q_params = rlt_td3.polyak_update(
        first.target_q_params,
        expected_q_params,
        tau=config.tau,
    )
    expected_state = first.replace(
        critic_step=jnp.asarray(first.critic_step + 1, dtype=jnp.int32),
        round_critic_step=jnp.asarray(first.round_critic_step + 1, dtype=jnp.int32),
        rng=next_rng,
        actor_params=expected_actor_params,
        q_params=expected_q_params,
        target_actor_params=expected_target_actor_params,
        target_q_params=expected_target_q_params,
        actor_opt_state=expected_actor_opt_state,
        critic_opt_state=expected_critic_opt_state,
    )
    expected_metrics = {
        **expected_critic_metrics,
        "critic/grad_norm": critic_grad_norm,
        "critic/grad_clipped": (critic_grad_norm > jnp.asarray(config.critic_grad_clip, dtype=jnp.float32)).astype(
            jnp.float32
        ),
        **expected_actor_metrics,
        "actor/grad_norm": actor_grad_norm,
        "actor/grad_clipped": (actor_grad_norm > jnp.asarray(config.actor_grad_clip, dtype=jnp.float32)).astype(
            jnp.float32
        ),
        "actor/updated": jnp.asarray(1.0, dtype=jnp.float32),
    }

    actual_state, actual_metrics = _run_step(first, batch, actor, critic, config, actor_tx, critic_tx)

    _assert_tree_allclose(actual_state, expected_state)
    _assert_tree_allclose(actual_metrics, expected_metrics)
    np.testing.assert_array_equal(
        jax.random.key_data(actual_state.rng),
        jax.random.key_data(next_rng),
    )

    (_, wrong_key_actor_metrics), wrong_key_actor_grads = jax.value_and_grad(
        lambda actor_params: actor_objective(actor_params, critic_rng),
        has_aux=True,
    )(first.actor_params)
    wrong_key_actor_updates, _ = actor_tx.update(
        wrong_key_actor_grads,
        first.actor_opt_state,
        first.actor_params,
    )
    wrong_key_actor_params = optax.apply_updates(first.actor_params, wrong_key_actor_updates)
    assert _trees_differ(expected_actor_params, wrong_key_actor_params)
    assert not np.allclose(
        expected_actor_metrics["actor/sample_rms"],
        wrong_key_actor_metrics["actor/sample_rms"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_jitted_train_step_is_finite_deterministic_and_preserves_fp32_state():
    _, actor, critic, config, state, actor_tx, critic_tx, batch = _training_setup(key_seed=42)
    compiled_step = jax.jit(
        lambda train_state, transition: rlt_td3_state.train_step(
            train_state,
            transition,
            actor,
            critic,
            config,
            actor_tx,
            critic_tx,
        )
    )

    first_state, first_metrics = compiled_step(state, batch)
    repeated_state, repeated_metrics = compiled_step(state, batch)

    _assert_tree_equal(first_state, repeated_state)
    _assert_tree_equal(first_metrics, repeated_metrics)
    _assert_scalar_fp32_metrics(first_metrics)
    _assert_all_floating_leaves_fp32(
        (
            first_state.actor_params,
            first_state.q_params,
            first_state.target_actor_params,
            first_state.target_q_params,
            first_state.actor_opt_state,
            first_state.critic_opt_state,
        )
    )
    for counter in (first_state.critic_step, first_state.round_critic_step):
        assert counter.shape == ()
        assert counter.dtype == jnp.int32


def test_train_step_is_exactly_reproducible_from_identical_state_batch_and_rng():
    _, actor, critic, config, state, actor_tx, critic_tx, batch = _training_setup(key_seed=43)

    first = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)
    repeated = _run_step(state, batch, actor, critic, config, actor_tx, critic_tx)

    _assert_tree_equal(first, repeated)
