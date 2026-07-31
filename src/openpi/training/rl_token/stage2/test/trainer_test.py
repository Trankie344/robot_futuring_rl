from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest

from openpi.models.rl_token import actor_critic
from openpi.training.rl_token.stage2 import replay
from openpi.training.rl_token.stage2 import td3
from openpi.training.rl_token.stage2 import trainer
from openpi.training.rl_token.stage2 import train_state


def _network() -> actor_critic.RLTActorCriticConfig:
    return actor_critic.RLTActorCriticConfig(
        z_dim=8,
        state_dim=3,
        action_horizon=4,
        action_dim=2,
        actor_state_proj_dim=5,
        actor_reference_proj_dim=5,
        critic_state_proj_dim=5,
        critic_action_proj_dim=5,
        actor_hidden_dims=(7, 7, 7),
        critic_hidden_dims=(7, 7, 7),
        compute_dtype="float32",
    )


def test_stage2_production_defaults_match_locked_plan():
    config = trainer.Stage2TrainerConfig()

    assert config.network.actor_hidden_dims == (1024, 1024, 1024)
    assert config.network.critic_hidden_dims == (1024, 1024, 1024)
    assert config.batch_size == 256
    assert config.algorithm.gamma == 0.99
    assert config.algorithm.tau == 0.005
    assert config.algorithm.policy_delay == 2
    assert config.algorithm.utd_ratio == 5
    assert config.algorithm.actor_lr == 1e-4
    assert config.algorithm.critic_lr == 3e-4
    assert config.algorithm.reference_dropout_rate == 0.5
    assert config.algorithm.noise_sigma == 0.1
    assert config.algorithm.noise_rho == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("log_interval", True),
        ("temp_checkpoint_interval", -1),
        ("temp_max_to_keep", 0),
        ("replay_max_open_shards", 1.5),
    ],
)
def test_stage2_runtime_counts_are_exact_positive_integers(field: str, value: object):
    with pytest.raises(ValueError, match=field):
        trainer.Stage2TrainerConfig(**{field: value})


def test_jsonl_metric_sink_writes_canonical_records(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    with trainer.JsonlMetricSink(path) as sink:
        sink(3, {"loss": np.float32(1.25), "actor/updated": 1})

    assert path.read_text(encoding="utf-8") == (
        json.dumps(
            {"actor/updated": 1.0, "critic_step": 3, "loss": 1.25},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


@dataclasses.dataclass
class _Replay:
    network: actor_critic.RLTActorCriticConfig
    total_transitions: int = 8

    def sample_indices(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        return rng.integers(0, self.total_transitions, size=batch_size, dtype=np.int64)

    def gather(self, indices: np.ndarray) -> replay.ReplayBatch:
        batch = len(indices)
        n = self.network
        zeros_z = np.zeros((batch, n.z_dim), dtype=ml_dtypes.bfloat16)
        zeros_state = np.zeros((batch, n.state_dim), dtype=np.float32)
        zeros_action = np.zeros((batch, n.action_horizon, n.action_dim), dtype=np.float32)
        return replay.ReplayBatch(
            z_rl=zeros_z,
            next_z_rl=zeros_z.copy(),
            state_norm=zeros_state,
            next_state_norm=zeros_state.copy(),
            vla_reference=zeros_action,
            next_vla_reference=zeros_action.copy(),
            executed_action=zeros_action.copy(),
            bc_anchor=zeros_action.copy(),
            reward=np.ones((batch, 1), dtype=np.float32),
            terminal=np.zeros((batch, 1), dtype=np.bool_),
            source_global_index=indices.copy(),
        )


def test_run_updates_completes_budget_and_emits_metrics_and_checkpoint():
    network = _network()
    algorithm = td3.TD3Config(
        policy_delay=2,
        utd_ratio=1,
        reference_dropout_rate=0.0,
        noise_sigma=0.0,
    )
    actor = actor_critic.RLTActor(network)
    critic = actor_critic.RLTCritic(network)
    state, actor_tx, critic_tx = train_state.initialize_train_state(
        actor,
        critic,
        algorithm,
        jax.random.key(0),
    )
    metrics: list[tuple[int, dict[str, float]]] = []
    checkpoints: list[int] = []

    result = trainer.run_updates(
        state=state,
        actor=actor,
        critic=critic,
        actor_tx=actor_tx,
        critic_tx=critic_tx,
        algorithm=algorithm,
        replay_buffer=_Replay(network),
        replay_rng=np.random.Generator(np.random.PCG64(7)),
        round_start_step=0,
        round_critic_updates=2,
        batch_size=2,
        log_interval=1,
        temp_checkpoint_interval=1,
        metric_sink=lambda step, values: metrics.append((step, dict(values))),
        checkpoint_sink=lambda saved_state, _rng: checkpoints.append(int(saved_state.critic_step)),
    )

    assert int(result.state.critic_step) == 2
    assert int(result.state.round_critic_step) == 2
    assert result.critic_updates_completed == 2
    assert result.actor_updates_completed == 1
    assert [step for step, _ in metrics] == [1, 2]
    assert checkpoints == [1]
