from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import native_training
from openpi.training.rl_token.stage2 import trainer
from openpi.training.rl_token.stage2 import checkpoints
from openpi.training.rl_token.stage2 import checkpoints as rlt_stage2_checkpoints
from openpi.training.rl_token.stage2 import td3 as rlt_td3
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state

pytestmark = pytest.mark.filterwarnings(
    r"ignore:shape requires ndarray or scalar arguments.*:DeprecationWarning:flax\.core\.scope"
)

_FEATURE_IDENTITY = "f" * 64
_FROZEN_PARAMS_SHA256 = "1" * 64
_NORM_STATS_SHA256 = "2" * 64
_LOADED_PARAMETER_SHA256 = "3" * 64
_LOADED_NORM_STATS_SHA256 = "4" * 64
_FEATURE_CODE_COMMIT = "5" * 40
_SAMPLER_NUM_STEPS = 10


def _feature_manifest(
    *,
    batch_id: str,
    **overrides,
) -> dict[str, object]:
    values = {
        "batch_id": batch_id,
        "feature_identity": _FEATURE_IDENTITY,
        "checkpoint_sha256": _FROZEN_PARAMS_SHA256,
        "norm_stats_sha256": _NORM_STATS_SHA256,
        "loaded_parameter_sha256": _LOADED_PARAMETER_SHA256,
        "loaded_norm_stats_sha256": _LOADED_NORM_STATS_SHA256,
        "stage1_config": rlt_stage2_checkpoints.RLT_STAGE1_CONFIG_NAME,
        "stage2_config": rlt_stage2_checkpoints.RLT_STAGE2_CONFIG_NAME,
        "stage1_checkpoint_step": 54999,
        "reward_source": "tristate",
        "reward_label_values": [-1, 0, 1, 2],
        "completion_label": 2,
        "reward_aggregation": "sum_20_frames",
        "reward_schema_version": 1,
        "asset_id": rlt_stage2_checkpoints.RLT_ASSET_ID,
        "sampler_num_steps": _SAMPLER_NUM_STEPS,
        "default_prompt": "fold clothes",
        "code_commit": _FEATURE_CODE_COMMIT,
    }
    values.update(overrides)
    return values


def _replay_record(
    *,
    batch_id: str,
    admission_sha: str,
    start: int,
    rows: int = 16,
    cache_manifest_sha: str = "c" * 64,
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "root": f"/cache/{batch_id}",
        "admission_sha256": admission_sha,
        "cache_manifest_sha256": cache_manifest_sha,
        "transition_rows": rows,
        "start": start,
        "end": start + rows,
    }


def _replay_snapshot_sha(shards: tuple[dict[str, object], ...]) -> str:
    return identity.sha256_json(
        {
            "schema_version": 1,
            "feature_identity": _FEATURE_IDENTITY,
            "total_transitions": shards[-1]["end"],
            "shards": list(shards),
        }
    )


class _FakeReplay:
    def __init__(
        self,
        *,
        snapshot_sha: str,
        shards: tuple[dict[str, object], ...],
        manifests: tuple[dict[str, object], ...],
    ):
        self.snapshot = SimpleNamespace(
            sha256=snapshot_sha,
            schema_version=1,
            feature_identity=_FEATURE_IDENTITY,
            total_transitions=shards[-1]["end"],
            shards=shards,
        )
        self.shard_verifications = tuple(SimpleNamespace(manifest=manifest) for manifest in manifests)
        self.total_transitions = shards[-1]["end"]
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _network() -> rlt_actor_critic.RLTActorCriticConfig:
    return rlt_actor_critic.RLTActorCriticConfig(
        z_dim=8,
        state_dim=3,
        action_horizon=4,
        action_dim=2,
        actor_state_proj_dim=5,
        actor_reference_proj_dim=6,
        critic_state_proj_dim=5,
        critic_action_proj_dim=6,
        actor_hidden_dims=(7, 7, 7),
        critic_hidden_dims=(7, 7, 7),
        compute_dtype="float32",
    )


def _runtime(
    *,
    network: rlt_actor_critic.RLTActorCriticConfig | None = None,
    algorithm: rlt_td3.TD3Config | None = None,
    batch_size: int = 2,
) -> trainer.Stage2TrainerConfig:
    return trainer.Stage2TrainerConfig(
        network=_network() if network is None else network,
        algorithm=rlt_td3.TD3Config(utd_ratio=2) if algorithm is None else algorithm,
        batch_size=batch_size,
        log_interval=1,
        temp_checkpoint_interval=1,
        temp_max_to_keep=2,
        replay_max_open_shards=2,
    )


def _config(
    tmp_path: Path,
    *,
    round_id: str = "round_000001",
    checkpoint_dir: Path | None = None,
    parent_checkpoint: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
    runtime: trainer.Stage2TrainerConfig | None = None,
) -> native_training.NativeRoundConfig:
    return native_training.NativeRoundConfig(
        checkpoint_dir=tmp_path / round_id if checkpoint_dir is None else checkpoint_dir,
        round_id=round_id,
        admission=tmp_path / f"{round_id}-admission.json",
        replay_snapshot=tmp_path / f"{round_id}-replay.json",
        parent_checkpoint=parent_checkpoint,
        seed=17,
        resume=resume,
        overwrite=overwrite,
        runtime=_runtime() if runtime is None else runtime,
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    round_id: str,
    admission_sha: str = "a" * 64,
    replay_sha: str = "b" * 64,
    batch_id: str = "batch_000001",
    replay_batch_id: str | None = None,
    replay_admission_sha: str | None = None,
    replay_shards: tuple[dict[str, object], ...] | None = None,
    replay_manifests: tuple[dict[str, object], ...] | None = None,
    chunk_equivalents: int = 2,
) -> list[_FakeReplay]:
    opened: list[_FakeReplay] = []
    admission_value = SimpleNamespace(
        path=Path("/admission"),
        round_id=round_id,
        batch_id=batch_id,
        sha256=admission_sha,
        chunk_equivalents=chunk_equivalents,
    )

    monkeypatch.setattr(native_training.admission, "open_admission", lambda _path: admission_value)

    def open_replay(_path, *, max_open_shards):
        assert max_open_shards > 0
        current_batch_id = batch_id if replay_batch_id is None else replay_batch_id
        current_admission_sha = admission_sha if replay_admission_sha is None else replay_admission_sha
        shards = (
            (
                _replay_record(
                    batch_id=current_batch_id,
                    admission_sha=current_admission_sha,
                    start=0,
                ),
            )
            if replay_shards is None
            else replay_shards
        )
        value = _FakeReplay(
            snapshot_sha=replay_sha,
            shards=shards,
            manifests=(
                tuple(_feature_manifest(batch_id=str(record["batch_id"])) for record in shards)
                if replay_manifests is None
                else replay_manifests
            ),
        )
        opened.append(value)
        return value

    monkeypatch.setattr(native_training.replay.ReplayBuffer, "open", open_replay)
    return opened


def _initialized_state(
    runtime: trainer.Stage2TrainerConfig,
    *,
    seed: int = 7,
) -> rlt_td3_state.RLTTrainState:
    state, _, _ = rlt_td3_state.initialize_train_state(
        rlt_actor_critic.RLTActor(runtime.network),
        rlt_actor_critic.RLTCritic(runtime.network),
        runtime.algorithm,
        jax.random.key(seed),
    )
    return state


def _state_with_counters(
    state: rlt_td3_state.RLTTrainState,
    *,
    critic_step: int,
    round_critic_step: int,
    policy_delay: int = 2,
) -> rlt_td3_state.RLTTrainState:
    actor_scale = state.actor_opt_state[1][0]._replace(count=jnp.asarray(critic_step // policy_delay, dtype=jnp.int32))
    critic_scale = state.critic_opt_state[1][0]._replace(count=jnp.asarray(critic_step, dtype=jnp.int32))
    return state.replace(
        critic_step=jnp.asarray(critic_step, dtype=jnp.int32),
        round_critic_step=jnp.asarray(round_critic_step, dtype=jnp.int32),
        actor_opt_state=(
            state.actor_opt_state[0],
            (actor_scale, state.actor_opt_state[1][1]),
        ),
        critic_opt_state=(
            state.critic_opt_state[0],
            (critic_scale, state.critic_opt_state[1][1]),
        ),
    )


def _metadata(
    *,
    state: rlt_td3_state.RLTTrainState,
    runtime: trainer.Stage2TrainerConfig,
    round_id: str,
    admission_sha: str = "a" * 64,
    replay_sha: str = "b" * 64,
    round_start_step: int,
    round_critic_updates: int,
    round_complete: bool,
    replay_rng_state: dict[str, object] | None = None,
    **overrides,
) -> rlt_stage2_checkpoints.RLTCheckpointMetadata:
    if replay_rng_state is None:
        replay_rng_state = np.random.Generator(np.random.PCG64(123)).bit_generator.state
    values = {
        "schema_version": 3,
        "stage1_config": rlt_stage2_checkpoints.RLT_STAGE1_CONFIG_NAME,
        "stage2_config": rlt_stage2_checkpoints.RLT_STAGE2_CONFIG_NAME,
        "asset_id": rlt_stage2_checkpoints.RLT_ASSET_ID,
        "base_checkpoint_step": rlt_stage2_checkpoints.RLT_BASE_CHECKPOINT_STEP,
        "reward_source": "tristate",
        "reward_label_values": (-1, 0, 1, 2),
        "completion_label": 2,
        "reward_aggregation": "sum_20_frames",
        "reward_schema_version": 1,
        "feature_identity": _FEATURE_IDENTITY,
        "frozen_params_sha256": _FROZEN_PARAMS_SHA256,
        "norm_stats_sha256": _NORM_STATS_SHA256,
        "sampler_num_steps": _SAMPLER_NUM_STEPS,
        "round_id": round_id,
        "admission_sha256": admission_sha,
        "replay_snapshot_sha256": replay_sha,
        "network_config": runtime.network,
        "algorithm_config": runtime.algorithm,
        "batch_size": runtime.batch_size,
        "round_start_step": round_start_step,
        "round_critic_updates": round_critic_updates,
        "critic_step": int(state.critic_step),
        "round_critic_step": int(state.round_critic_step),
        "replay_rng_state": copy.deepcopy(replay_rng_state),
        "jax_rng_impl": jax.random.key_impl(state.rng),
        "round_complete": round_complete,
    }
    values.update(overrides)
    return rlt_stage2_checkpoints.RLTCheckpointMetadata(**values)


def _save(
    root: Path,
    *,
    state: rlt_td3_state.RLTTrainState,
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
) -> Path:
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        root,
        keep_period=None,
        overwrite=False,
        resume=False,
        max_to_keep=3,
    )
    assert not resuming
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(manager, state=state, metadata=metadata)
        manager.wait_until_finished()
    finally:
        manager.close()
    return root / str(metadata.critic_step)


def _close_runtime(runtime: native_training.NativeRoundRuntime) -> None:
    try:
        runtime.manager.close()
    finally:
        runtime.replay_buffer.close()


def _assert_tree_equal(expected, actual) -> None:
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    assert actual_tree == expected_tree
    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
        if jax.dtypes.issubdtype(getattr(expected_leaf, "dtype", None), jax.dtypes.prng_key):
            expected_value = jax.random.key_data(expected_leaf)
            actual_value = jax.random.key_data(actual_leaf)
        else:
            expected_value = expected_leaf
            actual_value = actual_leaf
        np.testing.assert_array_equal(np.asarray(expected_value), np.asarray(actual_value))


def test_round_one_initializes_without_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    opened = _install_sources(monkeypatch, round_id="round_000001")
    config = _config(tmp_path)

    runtime = native_training.prepare_native_round(config)
    try:
        assert int(runtime.state.critic_step) == 0
        assert int(runtime.state.round_critic_step) == 0
        assert runtime.metadata.round_start_step == 0
        assert runtime.metadata.round_critic_updates == 4
        assert runtime.metadata.feature_identity == _FEATURE_IDENTITY
        assert runtime.metadata.frozen_params_sha256 == _FROZEN_PARAMS_SHA256
        assert runtime.metadata.norm_stats_sha256 == _NORM_STATS_SHA256
        assert runtime.metadata.sampler_num_steps == _SAMPLER_NUM_STEPS
        assert not runtime.complete
        expected = np.random.Generator(np.random.PCG64(config.seed)).integers(0, 1000, size=8)
        actual = runtime.replay_rng.integers(0, 1000, size=8)
        np.testing.assert_array_equal(actual, expected)
    finally:
        _close_runtime(runtime)

    assert opened[0].closed


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("feature_identity", "a" * 64),
        ("checkpoint_sha256", "6" * 64),
        ("norm_stats_sha256", "7" * 64),
        ("sampler_num_steps", 5),
        ("loaded_parameter_sha256", "8" * 64),
        ("loaded_norm_stats_sha256", "9" * 64),
        ("code_commit", "a" * 40),
    ],
)
def test_feature_contract_rejects_cross_shard_identity_drift(
    field: str,
    replacement: object,
):
    records = (
        _replay_record(batch_id="batch_000001", admission_sha="a" * 64, start=0),
        _replay_record(batch_id="batch_000002", admission_sha="b" * 64, start=16),
    )
    replay_buffer = _FakeReplay(
        snapshot_sha=_replay_snapshot_sha(records),
        shards=records,
        manifests=(
            _feature_manifest(batch_id="batch_000001"),
            _feature_manifest(batch_id="batch_000002", **{field: replacement}),
        ),
    )

    with pytest.raises(ValueError, match=field):
        native_training._feature_contract_from_replay(replay_buffer)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("stage1_config", "wrong"),
        ("stage2_config", "wrong"),
        ("asset_id", "wrong"),
        ("default_prompt", "other task"),
    ],
)
def test_feature_contract_requires_locked_online_identity(
    field: str,
    replacement: str,
):
    records = (_replay_record(batch_id="batch_000001", admission_sha="a" * 64, start=0),)
    replay_buffer = _FakeReplay(
        snapshot_sha=_replay_snapshot_sha(records),
        shards=records,
        manifests=(_feature_manifest(batch_id="batch_000001", **{field: replacement}),),
    )

    with pytest.raises(ValueError, match=field):
        native_training._feature_contract_from_replay(replay_buffer)  # noqa: SLF001


def test_round_one_replay_contains_exactly_its_current_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_history = _replay_record(
        batch_id="hidden_history",
        admission_sha="1" * 64,
        start=0,
    )
    current = _replay_record(
        batch_id="batch_000001",
        admission_sha="a" * 64,
        start=16,
    )
    opened = _install_sources(
        monkeypatch,
        round_id="round_000001",
        replay_shards=(hidden_history, current),
    )
    checkpoint_root = tmp_path / "round_000001"

    with pytest.raises(ValueError, match="round_000001.*exactly one replay shard"):
        native_training.prepare_native_round(
            _config(
                tmp_path,
                checkpoint_dir=checkpoint_root,
            )
        )

    assert not checkpoint_root.exists()
    assert opened[0].closed


def test_round_one_rejects_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_sources(monkeypatch, round_id="round_000001")
    config = _config(tmp_path, parent_checkpoint=tmp_path / "parent" / "1")

    with pytest.raises(ValueError, match="round_000001.*parent"):
        native_training.prepare_native_round(config)


def test_later_round_requires_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_sources(monkeypatch, round_id="round_000002")

    with pytest.raises(ValueError, match="requires.*parent"):
        native_training.prepare_native_round(_config(tmp_path, round_id="round_000002"))


@pytest.mark.parametrize("failure", ["wrong_round", "incomplete"])
def test_parent_must_be_exact_previous_complete_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    parent_round = "round_000003" if failure == "wrong_round" else "round_000001"
    complete = failure == "wrong_round"
    runtime_config = _runtime()
    parent_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=4 if complete else 2,
        round_critic_step=4 if complete else 2,
    )
    parent_root = tmp_path / "parent"
    parent_step = _save(
        parent_root,
        state=parent_state,
        metadata=_metadata(
            state=parent_state,
            runtime=runtime_config,
            round_id=parent_round,
            round_start_step=0,
            round_critic_updates=4,
            round_complete=complete,
        ),
    )
    _install_sources(monkeypatch, round_id="round_000002")

    with pytest.raises(ValueError, match="previous|complete"):
        native_training.prepare_native_round(
            _config(
                tmp_path,
                round_id="round_000002",
                parent_checkpoint=parent_step,
                runtime=runtime_config,
            )
        )
    assert not (tmp_path / "round_000002").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("feature_identity", "6" * 64),
        ("frozen_params_sha256", "7" * 64),
        ("norm_stats_sha256", "8" * 64),
        ("sampler_num_steps", 5),
    ],
)
def test_parent_checkpoint_must_match_current_feature_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
):
    runtime_config = _runtime()
    parent_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=4,
        round_critic_step=4,
    )
    previous_record = _replay_record(
        batch_id="batch_000001",
        admission_sha="1" * 64,
        start=0,
    )
    parent_metadata = _metadata(
        state=parent_state,
        runtime=runtime_config,
        round_id="round_000001",
        replay_sha=_replay_snapshot_sha((previous_record,)),
        round_start_step=0,
        round_critic_updates=4,
        round_complete=True,
        **{field: replacement},
    )
    parent_step = _save(
        tmp_path / f"parent-{field}",
        state=parent_state,
        metadata=parent_metadata,
    )
    current_record = _replay_record(
        batch_id="batch_000002",
        admission_sha="a" * 64,
        start=16,
    )
    _install_sources(
        monkeypatch,
        round_id="round_000002",
        batch_id="batch_000002",
        replay_shards=(previous_record, current_record),
    )

    with pytest.raises(ValueError, match=field):
        native_training.prepare_native_round(
            _config(
                tmp_path,
                round_id="round_000002",
                checkpoint_dir=tmp_path / f"round-2-{field}",
                parent_checkpoint=parent_step,
                runtime=runtime_config,
            )
        )


@pytest.mark.parametrize(
    "relationship",
    ["root", "step", "ancestor", "root_descendant", "step_descendant"],
)
def test_current_checkpoint_root_cannot_overlap_parent_checkpoint_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
):
    runtime_config = _runtime()
    parent_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=4,
        round_critic_step=4,
    )
    parent_root = tmp_path / "parent"
    parent_step = _save(
        parent_root,
        state=parent_state,
        metadata=_metadata(
            state=parent_state,
            runtime=runtime_config,
            round_id="round_000001",
            round_start_step=0,
            round_critic_updates=4,
            round_complete=True,
        ),
    )
    _install_sources(monkeypatch, round_id="round_000002", batch_id="batch_000002")
    checkpoint_dir = {
        "root": parent_root,
        "step": parent_step,
        "ancestor": tmp_path,
        "root_descendant": parent_root / "nested",
        "step_descendant": parent_step / "nested",
    }[relationship]

    prepared = None
    try:
        with pytest.raises(ValueError, match="checkpoint_dir.*parent"):
            prepared = native_training.prepare_native_round(
                _config(
                    tmp_path,
                    round_id="round_000002",
                    checkpoint_dir=checkpoint_dir,
                    parent_checkpoint=parent_step,
                    overwrite=True,
                    runtime=runtime_config,
                )
            )
    finally:
        if prepared is not None:
            _close_runtime(prepared)

    assert parent_step.is_dir()


@pytest.mark.parametrize("history", ["missing", "replaced"])
def test_later_round_replay_must_exactly_extend_parent_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    history: str,
):
    runtime_config = _runtime()
    parent_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=4,
        round_critic_step=4,
    )
    previous_record = _replay_record(
        batch_id="batch_000001",
        admission_sha="1" * 64,
        start=0,
    )
    parent_step = _save(
        tmp_path / "parent",
        state=parent_state,
        metadata=_metadata(
            state=parent_state,
            runtime=runtime_config,
            round_id="round_000001",
            replay_sha=_replay_snapshot_sha((previous_record,)),
            round_start_step=0,
            round_critic_updates=4,
            round_complete=True,
        ),
    )
    current_record = _replay_record(
        batch_id="batch_000002",
        admission_sha="a" * 64,
        start=0 if history == "missing" else 16,
    )
    if history == "missing":
        current_shards = (current_record,)
    else:
        replacement = _replay_record(
            batch_id="replacement",
            admission_sha="2" * 64,
            start=0,
            cache_manifest_sha="d" * 64,
        )
        current_shards = (replacement, current_record)
    _install_sources(
        monkeypatch,
        round_id="round_000002",
        batch_id="batch_000002",
        replay_shards=current_shards,
    )
    current_root = tmp_path / "round_000002"

    prepared = None
    try:
        with pytest.raises(ValueError, match="replay history.*parent"):
            prepared = native_training.prepare_native_round(
                _config(
                    tmp_path,
                    round_id="round_000002",
                    checkpoint_dir=current_root,
                    parent_checkpoint=parent_step,
                    runtime=runtime_config,
                )
            )
    finally:
        if prepared is not None:
            _close_runtime(prepared)

    assert parent_step.is_dir()
    assert not current_root.exists()


def test_new_round_restores_every_leaf_and_resets_only_local_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = _runtime()
    parent_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=4,
        round_critic_step=4,
    )
    parent_rng = np.random.Generator(np.random.PCG64(91))
    parent_rng.integers(0, 10_000, size=19)
    previous_record = _replay_record(
        batch_id="batch_000001",
        admission_sha="1" * 64,
        start=0,
    )
    parent_step = _save(
        tmp_path / "parent",
        state=parent_state,
        metadata=_metadata(
            state=parent_state,
            runtime=runtime_config,
            round_id="round_000001",
            round_start_step=0,
            round_critic_updates=4,
            round_complete=True,
            replay_rng_state=parent_rng.bit_generator.state,
            replay_sha=_replay_snapshot_sha((previous_record,)),
        ),
    )
    current_record = _replay_record(
        batch_id="batch_000002",
        admission_sha="a" * 64,
        start=16,
    )
    _install_sources(
        monkeypatch,
        round_id="round_000002",
        batch_id="batch_000002",
        replay_shards=(previous_record, current_record),
    )

    runtime = native_training.prepare_native_round(
        _config(
            tmp_path,
            round_id="round_000002",
            parent_checkpoint=parent_step,
            runtime=runtime_config,
        )
    )
    try:
        _assert_tree_equal(
            parent_state.replace(round_critic_step=jnp.asarray(0, dtype=jnp.int32)),
            runtime.state,
        )
        expected = parent_rng.integers(0, 10_000, size=12)
        actual = runtime.replay_rng.integers(0, 10_000, size=12)
        np.testing.assert_array_equal(actual, expected)
        assert runtime.metadata.round_start_step == 4
        assert runtime.metadata.critic_step == 4
    finally:
        _close_runtime(runtime)


def test_resume_restores_latest_without_reset_and_replay_rng_bit_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = _runtime()
    saved_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=2,
        round_critic_step=2,
    )
    replay_rng = np.random.Generator(np.random.PCG64(81))
    replay_rng.integers(0, 100_000, size=31)
    checkpoint_root = tmp_path / "current"
    _save(
        checkpoint_root,
        state=saved_state,
        metadata=_metadata(
            state=saved_state,
            runtime=runtime_config,
            round_id="round_000001",
            round_start_step=0,
            round_critic_updates=4,
            round_complete=False,
            replay_rng_state=replay_rng.bit_generator.state,
        ),
    )
    _install_sources(monkeypatch, round_id="round_000001")

    runtime = native_training.prepare_native_round(
        _config(
            tmp_path,
            checkpoint_dir=checkpoint_root,
            resume=True,
            runtime=runtime_config,
        )
    )
    try:
        _assert_tree_equal(saved_state, runtime.state)
        assert int(runtime.state.round_critic_step) == 2
        expected = replay_rng.integers(0, 100_000, size=16)
        actual = runtime.replay_rng.integers(0, 100_000, size=16)
        np.testing.assert_array_equal(actual, expected)
    finally:
        _close_runtime(runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        "round",
        "admission",
        "replay",
        "network",
        "algorithm",
        "batch_size",
        "budget",
        "feature_identity",
        "frozen_params_sha256",
        "norm_stats_sha256",
        "sampler_num_steps",
    ],
)
def test_resume_rejects_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    runtime_config = _runtime()
    saved_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=2,
        round_critic_step=2,
    )
    metadata = _metadata(
        state=saved_state,
        runtime=runtime_config,
        round_id="round_000001",
        round_start_step=0,
        round_critic_updates=4,
        round_complete=False,
    )
    if mutation == "round":
        metadata = dataclasses.replace(metadata, round_id="round_000002")
    elif mutation == "admission":
        metadata = dataclasses.replace(metadata, admission_sha256="c" * 64)
    elif mutation == "replay":
        metadata = dataclasses.replace(metadata, replay_snapshot_sha256="d" * 64)
    elif mutation == "network":
        metadata = dataclasses.replace(
            metadata,
            network_config=dataclasses.replace(runtime_config.network, compute_dtype="bfloat16"),
        )
    elif mutation == "algorithm":
        metadata = dataclasses.replace(
            metadata,
            algorithm_config=dataclasses.replace(runtime_config.algorithm, gamma=0.9),
        )
    elif mutation == "batch_size":
        metadata = dataclasses.replace(metadata, batch_size=3)
    elif mutation == "budget":
        metadata = dataclasses.replace(metadata, round_critic_updates=5)
    elif mutation == "feature_identity":
        metadata = dataclasses.replace(metadata, feature_identity="6" * 64)
    elif mutation == "frozen_params_sha256":
        metadata = dataclasses.replace(metadata, frozen_params_sha256="7" * 64)
    elif mutation == "norm_stats_sha256":
        metadata = dataclasses.replace(metadata, norm_stats_sha256="8" * 64)
    elif mutation == "sampler_num_steps":
        metadata = dataclasses.replace(metadata, sampler_num_steps=5)
    checkpoint_root = tmp_path / f"mismatch-{mutation}"
    _save(checkpoint_root, state=saved_state, metadata=metadata)
    _install_sources(monkeypatch, round_id="round_000001")

    with pytest.raises(ValueError, match="mismatch"):
        native_training.prepare_native_round(
            _config(
                tmp_path,
                checkpoint_dir=checkpoint_root,
                resume=True,
                runtime=runtime_config,
            )
        )


def test_completed_resume_is_read_only_before_metric_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = _runtime()
    final_state = _state_with_counters(
        _initialized_state(runtime_config),
        critic_step=4,
        round_critic_step=4,
    )
    checkpoint_root = tmp_path / "complete"
    final_step = _save(
        checkpoint_root,
        state=final_state,
        metadata=_metadata(
            state=final_state,
            runtime=runtime_config,
            round_id="round_000001",
            round_start_step=0,
            round_critic_updates=4,
            round_complete=True,
        ),
    )
    opened = _install_sources(monkeypatch, round_id="round_000001")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("completed resume performed a mutating training action")

    monkeypatch.setattr(native_training.trainer, "JsonlMetricSink", unexpected)
    monkeypatch.setattr(native_training.trainer, "run_updates", unexpected)

    result = native_training.run_native_round(
        _config(
            tmp_path,
            checkpoint_dir=checkpoint_root,
            resume=True,
            runtime=runtime_config,
        )
    )

    assert result == final_step
    assert not (checkpoint_root / "metrics.jsonl").exists()
    assert opened[0].closed


def test_run_native_round_saves_incomplete_then_complete_native_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = _runtime()
    opened = _install_sources(monkeypatch, round_id="round_000001")
    config = _config(tmp_path, runtime=runtime_config)

    def fake_run_updates(**kwargs):
        assert kwargs["round_start_step"] == 0
        assert kwargs["round_critic_updates"] == 4
        temporary = kwargs["state"].replace(
            critic_step=jnp.asarray(1, dtype=jnp.int32),
            round_critic_step=jnp.asarray(1, dtype=jnp.int32),
        )
        kwargs["checkpoint_sink"](temporary, copy.deepcopy(kwargs["replay_rng"].bit_generator.state))
        final_state = kwargs["state"].replace(
            critic_step=jnp.asarray(4, dtype=jnp.int32),
            round_critic_step=jnp.asarray(4, dtype=jnp.int32),
        )
        kwargs["metric_sink"](4, {"loss": 1.0})
        return trainer.TrainingResult(
            state=final_state,
            replay_rng_state=copy.deepcopy(kwargs["replay_rng"].bit_generator.state),
            critic_updates_completed=4,
            actor_updates_completed=2,
            round_critic_updates_completed=4,
            round_actor_updates_completed=2,
            final_metrics={"loss": 1.0},
        )

    monkeypatch.setattr(native_training.trainer, "run_updates", fake_run_updates)

    final_step = native_training.run_native_round(config)

    assert final_step == config.checkpoint_dir / "4"
    assert rlt_stage2_checkpoints.load_rlt_metadata(config.checkpoint_dir / "1").round_complete is False
    final_metadata = rlt_stage2_checkpoints.load_rlt_metadata(final_step)
    assert final_metadata.round_complete is True
    assert final_metadata.round_critic_step == 4
    assert (config.checkpoint_dir / "metrics.jsonl").is_file()
    assert opened[0].closed


def test_prepare_validates_resume_root_and_replay_tail_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_sources(
        monkeypatch,
        round_id="round_000001",
        replay_batch_id="wrong",
    )
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="resume.*exist"):
        native_training.prepare_native_round(_config(tmp_path, checkpoint_dir=missing, resume=True))
    assert not missing.exists()

    with pytest.raises(ValueError, match="replay.*admission"):
        native_training.prepare_native_round(_config(tmp_path))
