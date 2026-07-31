"""Optimizer-backed train state and update scheduling for RL-token TD3."""

import operator
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import td3 as rlt_td3


@struct.dataclass
class RLTTrainState:
    """All mutable state needed by one RL-token TD3 update."""

    critic_step: jax.Array
    round_critic_step: jax.Array
    rng: jax.Array
    actor_params: Any
    q_params: Any
    target_actor_params: Any
    target_q_params: Any
    actor_opt_state: Any
    critic_opt_state: Any


def _is_positive_integer(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return False
    try:
        return operator.index(value) > 0
    except TypeError:
        return False


def _is_nonnegative_integer(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return False
    try:
        return operator.index(value) >= 0
    except TypeError:
        return False


def _floating_leaves_to_fp32(tree):
    def convert(leaf):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            return jnp.asarray(leaf, dtype=jnp.float32)
        return leaf

    return jax.tree_util.tree_map(convert, tree)


def _copy_floating_leaves_to_fp32(tree):
    def copy(leaf):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            return jnp.array(leaf, dtype=jnp.float32, copy=True)
        return leaf

    return jax.tree_util.tree_map(copy, tree)


def make_optimizers(
    config: rlt_td3.TD3Config,
) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Build independent clipped Adam optimizers for actor and twin critics."""
    config.validate()
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.actor_grad_clip),
        optax.adam(
            config.actor_lr,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
            mu_dtype=jnp.float32,
        ),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.critic_grad_clip),
        optax.adam(
            config.critic_lr,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
            mu_dtype=jnp.float32,
        ),
    )
    return actor_tx, critic_tx


def initialize_train_state(
    actor: rlt_actor_critic.RLTActor,
    critic: rlt_actor_critic.RLTCritic,
    config: rlt_td3.TD3Config,
    rng: jax.Array,
) -> tuple[RLTTrainState, optax.GradientTransformation, optax.GradientTransformation]:
    """Initialize independent online/target networks and their optimizers."""
    config.validate()
    actor.config.validate()
    critic.config.validate()
    if actor.config != critic.config:
        raise ValueError(
            f"Actor and critic configs must match exactly: actor={actor.config!r}, critic={critic.config!r}."
        )

    network_config = actor.config
    compute_dtype = network_config.jnp_compute_dtype
    z_rl = jnp.zeros((1, network_config.z_dim), dtype=compute_dtype)
    state_norm = jnp.zeros((1, network_config.state_dim), dtype=compute_dtype)
    action = jnp.zeros(
        (1, network_config.action_horizon, network_config.action_dim),
        dtype=compute_dtype,
    )
    actor_key, q1_key, q2_key, next_rng = jax.random.split(rng, 4)
    actor_params = _floating_leaves_to_fp32(actor.init(actor_key, z_rl, state_norm, action)["params"])
    q_params = {
        "q1": _floating_leaves_to_fp32(critic.init(q1_key, z_rl, state_norm, action)["params"]),
        "q2": _floating_leaves_to_fp32(critic.init(q2_key, z_rl, state_norm, action)["params"]),
    }
    target_actor_params = _copy_floating_leaves_to_fp32(actor_params)
    target_q_params = _copy_floating_leaves_to_fp32(q_params)

    actor_tx, critic_tx = make_optimizers(config)
    actor_opt_state = _floating_leaves_to_fp32(actor_tx.init(actor_params))
    critic_opt_state = _floating_leaves_to_fp32(critic_tx.init(q_params))
    state = RLTTrainState(
        critic_step=jnp.asarray(0, dtype=jnp.int32),
        round_critic_step=jnp.asarray(0, dtype=jnp.int32),
        rng=next_rng,
        actor_params=actor_params,
        q_params=q_params,
        target_actor_params=target_actor_params,
        target_q_params=target_q_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
    )
    return state, actor_tx, critic_tx


def round_update_budget(
    chunk_equivalents: int,
    *,
    utd_ratio: int,
    policy_delay: int,
) -> tuple[int, int]:
    """Return critic and delayed-actor update counts for one data round."""
    if not _is_nonnegative_integer(chunk_equivalents):
        raise ValueError(f"chunk_equivalents must be a nonnegative integer, got {chunk_equivalents!r}.")
    for name, value in (("utd_ratio", utd_ratio), ("policy_delay", policy_delay)):
        if not _is_positive_integer(value):
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    critic_updates = int(operator.index(chunk_equivalents)) * int(operator.index(utd_ratio))
    actor_updates = critic_updates // int(operator.index(policy_delay))
    return critic_updates, actor_updates


def start_new_round(state: RLTTrainState) -> RLTTrainState:
    """Explicitly reset only the per-round delayed-update counter."""
    return state.replace(round_critic_step=jnp.asarray(0, dtype=jnp.int32))


def _zero_actor_metrics() -> dict[str, jax.Array]:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {
        "actor/loss": zero,
        "actor/q_term": zero,
        "actor/bc_loss": zero,
        "actor/reference_drop_fraction": zero,
        "actor/mean_rms": zero,
        "actor/sample_rms": zero,
        "actor/anchor_l1": zero,
        "actor/vla_l1": zero,
        "actor/saturation_fraction": zero,
        "actor/grad_norm": zero,
        "actor/grad_clipped": zero,
    }


def train_step(
    state: RLTTrainState,
    batch: rlt_td3.RLTTransitionBatch,
    actor: rlt_actor_critic.RLTActor,
    critic: rlt_actor_critic.RLTCritic,
    config: rlt_td3.TD3Config,
    actor_tx: optax.GradientTransformation,
    critic_tx: optax.GradientTransformation,
) -> tuple[RLTTrainState, dict[str, jax.Array]]:
    """Run one critic update and, on schedule, one actor/target update."""
    next_rng, critic_rng, actor_rng = jax.random.split(state.rng, 3)

    def critic_objective(q_params):
        return rlt_td3.critic_loss(
            q_params,
            target_q_params=state.target_q_params,
            target_actor_params=state.target_actor_params,
            batch=batch,
            rng=critic_rng,
            actor=actor,
            critic=critic,
            config=config,
        )

    (_, critic_metrics), critic_grads = jax.value_and_grad(
        critic_objective,
        has_aux=True,
    )(state.q_params)
    critic_grad_norm = jnp.asarray(
        optax.global_norm(critic_grads),
        dtype=jnp.float32,
    )
    critic_updates, critic_opt_state = critic_tx.update(
        critic_grads,
        state.critic_opt_state,
        state.q_params,
    )
    q_params = optax.apply_updates(state.q_params, critic_updates)
    critic_step = (jnp.asarray(state.critic_step, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)).astype(jnp.int32)
    round_critic_step = (
        jnp.asarray(state.round_critic_step, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    ).astype(jnp.int32)
    critic_metrics = {
        **critic_metrics,
        "critic/grad_norm": critic_grad_norm,
        "critic/grad_clipped": (critic_grad_norm > jnp.asarray(config.critic_grad_clip, dtype=jnp.float32)).astype(
            jnp.float32
        ),
    }

    def update_actor(_):
        def actor_objective(actor_params):
            return rlt_td3.actor_loss(
                actor_params,
                q1_params=q_params["q1"],
                batch=batch,
                rng=actor_rng,
                actor=actor,
                critic=critic,
                config=config,
            )

        (_, actor_metrics), actor_grads = jax.value_and_grad(
            actor_objective,
            has_aux=True,
        )(state.actor_params)
        actor_grad_norm = jnp.asarray(
            optax.global_norm(actor_grads),
            dtype=jnp.float32,
        )
        actor_updates, actor_opt_state = actor_tx.update(
            actor_grads,
            state.actor_opt_state,
            state.actor_params,
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)
        target_actor_params = rlt_td3.polyak_update(
            state.target_actor_params,
            actor_params,
            tau=config.tau,
        )
        target_q_params = rlt_td3.polyak_update(
            state.target_q_params,
            q_params,
            tau=config.tau,
        )
        actor_metrics = {
            **actor_metrics,
            "actor/grad_norm": actor_grad_norm,
            "actor/grad_clipped": (actor_grad_norm > jnp.asarray(config.actor_grad_clip, dtype=jnp.float32)).astype(
                jnp.float32
            ),
        }
        return (
            actor_params,
            actor_opt_state,
            target_actor_params,
            target_q_params,
            actor_metrics,
        )

    def skip_actor(_):
        return (
            state.actor_params,
            state.actor_opt_state,
            state.target_actor_params,
            state.target_q_params,
            _zero_actor_metrics(),
        )

    should_update_actor = jnp.equal(
        jnp.mod(
            round_critic_step,
            jnp.asarray(config.policy_delay, dtype=jnp.int32),
        ),
        jnp.asarray(0, dtype=jnp.int32),
    )
    (
        actor_params,
        actor_opt_state,
        target_actor_params,
        target_q_params,
        actor_metrics,
    ) = jax.lax.cond(
        should_update_actor,
        update_actor,
        skip_actor,
        operand=None,
    )
    metrics = {
        **critic_metrics,
        **actor_metrics,
        "actor/updated": should_update_actor.astype(jnp.float32),
    }
    updated_state = state.replace(
        critic_step=critic_step,
        round_critic_step=round_critic_step,
        rng=next_rng,
        actor_params=actor_params,
        q_params=q_params,
        target_actor_params=target_actor_params,
        target_q_params=target_q_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
    )
    return updated_state, metrics
