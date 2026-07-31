from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import checkpoints
from openpi.training.rl_token.stage2 import checkpoints as rlt_stage2_checkpoints
from openpi.training.rl_token.stage2 import td3 as rlt_td3
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state

pytestmark = pytest.mark.filterwarnings(
    r"ignore:shape requires ndarray or scalar arguments.*:DeprecationWarning:flax\.core\.scope"
)


def _network_config() -> rlt_actor_critic.RLTActorCriticConfig:
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


def _initialized_state() -> tuple[
    rlt_td3_state.RLTTrainState,
    rlt_actor_critic.RLTActorCriticConfig,
    rlt_td3.TD3Config,
]:
    network = _network_config()
    algorithm = rlt_td3.TD3Config()
    state, _, _ = rlt_td3_state.initialize_train_state(
        rlt_actor_critic.RLTActor(network),
        rlt_actor_critic.RLTCritic(network),
        algorithm,
        jax.random.key(7),
    )
    return (
        state.replace(
            critic_step=jnp.asarray(3, dtype=jnp.int32),
            round_critic_step=jnp.asarray(2, dtype=jnp.int32),
        ),
        network,
        algorithm,
    )


def _metadata(
    state: rlt_td3_state.RLTTrainState,
    network: rlt_actor_critic.RLTActorCriticConfig,
    algorithm: rlt_td3.TD3Config,
    **overrides,
) -> rlt_stage2_checkpoints.RLTCheckpointMetadata:
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
        "feature_identity": "c" * 64,
        "frozen_params_sha256": "d" * 64,
        "norm_stats_sha256": "e" * 64,
        "sampler_num_steps": 10,
        "round_id": "round_000001",
        "admission_sha256": "a" * 64,
        "replay_snapshot_sha256": "b" * 64,
        "network_config": network,
        "algorithm_config": algorithm,
        "batch_size": 256,
        "round_start_step": 1,
        "round_critic_updates": 2,
        "critic_step": int(state.critic_step),
        "round_critic_step": int(state.round_critic_step),
        "replay_rng_state": np.random.Generator(np.random.PCG64(123)).bit_generator.state,
        "jax_rng_impl": jax.random.key_impl(state.rng),
        "round_complete": True,
    }
    values.update(overrides)
    return rlt_stage2_checkpoints.RLTCheckpointMetadata(**values)


def _assert_trees_equal(expected, actual) -> None:
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    assert actual_structure == expected_structure
    assert len(actual_leaves) == len(expected_leaves)
    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
        if jax.dtypes.issubdtype(getattr(expected_leaf, "dtype", None), jax.dtypes.prng_key):
            expected_value = jax.random.key_data(expected_leaf)
            actual_value = jax.random.key_data(actual_leaf)
        else:
            expected_value = expected_leaf
            actual_value = actual_leaf
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(expected_value)),
            np.asarray(jax.device_get(actual_value)),
        )


def _manager(root: Path, *, max_to_keep: int = 3):
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        root,
        keep_period=None,
        overwrite=False,
        resume=False,
        max_to_keep=max_to_keep,
    )
    assert not resuming
    return manager


def test_native_checkpoint_round_trip_and_openpi_actor_params(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    metadata = _metadata(state, network, algorithm)
    root = tmp_path / "native"
    manager = _manager(root)
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=metadata,
        )
        manager.wait_until_finished()

        step_dir = root / "3"
        assert (step_dir / "assets" / rlt_stage2_checkpoints.RLT_METADATA_FILENAME).is_file()
        assert (step_dir / "params").is_dir()
        assert (step_dir / "train_state").is_dir()

        loaded_metadata = rlt_stage2_checkpoints.load_rlt_metadata(step_dir)
        assert loaded_metadata == metadata

        target_state, _, _ = rlt_td3_state.initialize_train_state(
            rlt_actor_critic.RLTActor(network),
            rlt_actor_critic.RLTCritic(network),
            algorithm,
            jax.random.key(999),
        )
        restored, restored_metadata = rlt_stage2_checkpoints.restore_rlt_checkpoint(
            manager,
            target_state=target_state,
            step=3,
        )
        _assert_trees_equal(state, restored)
        assert jax.random.key_impl(restored.rng) == jax.random.key_impl(state.rng)
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(restored.rng)),
            np.asarray(jax.random.key_data(state.rng)),
        )
        assert restored_metadata == metadata

        deployment_actor = _model.restore_params(step_dir / "params", dtype=jnp.float32)
        _assert_trees_equal(state.actor_params, deployment_actor)
    finally:
        manager.close()


def test_native_checkpoint_round_trips_non_threefry_typed_key(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    state = state.replace(rng=jax.random.key(17, impl="rbg"))
    root = tmp_path / "rbg"
    manager = _manager(root)
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=_metadata(state, network, algorithm),
        )
        manager.wait_until_finished()
        target = state.replace(rng=jax.random.key(99, impl="rbg"))
        restored, _ = rlt_stage2_checkpoints.restore_rlt_checkpoint(
            manager,
            target_state=target,
            step=3,
        )
        assert jax.random.key_impl(restored.rng) == jax.random.key_impl(state.rng)
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(restored.rng)),
            np.asarray(jax.random.key_data(state.rng)),
        )
    finally:
        manager.close()


def test_restore_rejects_target_with_different_typed_key_implementation(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    state = state.replace(rng=jax.random.key(17, impl="rbg"))
    root = tmp_path / "rng-implementation"
    manager = _manager(root)
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=_metadata(state, network, algorithm),
        )
        manager.wait_until_finished()
        target = state.replace(rng=jax.random.key(99, impl="unsafe_rbg"))

        with pytest.raises(ValueError, match="implementation"):
            rlt_stage2_checkpoints.restore_rlt_checkpoint(
                manager,
                target_state=target,
                step=3,
            )
    finally:
        manager.close()


def test_manager_close_finishes_complete_native_checkpoint(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    root = tmp_path / "close-waits"
    manager = _manager(root)
    rlt_stage2_checkpoints.save_rlt_checkpoint(
        manager,
        state=state,
        metadata=_metadata(state, network, algorithm),
    )
    manager.close()

    step_dir = root / "3"
    assert (step_dir / "assets" / rlt_stage2_checkpoints.RLT_METADATA_FILENAME).is_file()
    assert (step_dir / "params").is_dir()
    assert (step_dir / "train_state").is_dir()


def test_native_checkpoint_retention_keeps_latest_three_steps(tmp_path: Path):
    initial_state, network, algorithm = _initialized_state()
    root = tmp_path / "retention"
    manager = _manager(root)
    try:
        for step in range(1, 5):
            state = initial_state.replace(
                critic_step=jnp.asarray(step, dtype=jnp.int32),
                round_critic_step=jnp.asarray(step, dtype=jnp.int32),
            )
            metadata = _metadata(
                state,
                network,
                algorithm,
                round_start_step=0,
                round_critic_updates=4,
                round_complete=step == 4,
            )
            rlt_stage2_checkpoints.save_rlt_checkpoint(
                manager,
                state=state,
                metadata=metadata,
            )
            manager.wait_until_finished()

        assert tuple(manager.all_steps()) == (2, 3, 4)
        assert not (root / "1").exists()
        assert all((root / str(step) / "params").is_dir() for step in (2, 3, 4))
    finally:
        manager.close()


class _RejectSaveManager:
    def __init__(self):
        self.called = False

    def save(self, *_args, **_kwargs):
        self.called = True
        raise AssertionError("invalid checkpoint reached Orbax")


class _SkippedSaveManager:
    def save(self, *_args, **_kwargs):
        return False


def test_native_checkpoint_rejects_skipped_orbax_save():
    state, network, algorithm = _initialized_state()

    with pytest.raises(RuntimeError, match="did not accept"):
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            _SkippedSaveManager(),
            state=state,
            metadata=_metadata(state, network, algorithm),
        )


@pytest.mark.parametrize(
    "replace_kwargs",
    [
        {"schema_version": 1},
        {"stage1_config": "wrong"},
        {"stage2_config": "wrong"},
        {"asset_id": "wrong"},
        {"base_checkpoint_step": -1},
        {"reward_source": "progress"},
        {"reward_label_values": (0, 1, 2)},
        {"completion_label": 1},
        {"reward_aggregation": "mean"},
        {"reward_schema_version": 2},
        {"feature_identity": "short"},
        {"frozen_params_sha256": "D" * 64},
        {"norm_stats_sha256": ""},
        {"sampler_num_steps": 0},
        {"sampler_num_steps": True},
        {"round_id": "round_000000"},
        {"admission_sha256": "A" * 64},
        {"replay_snapshot_sha256": "short"},
        {"batch_size": True},
        {"round_critic_updates": 0},
        {"critic_step": 4},
        {"round_critic_step": 3},
        {"round_complete": False},
        {"replay_rng_state": {"bit_generator": "PCG64"}},
        {"jax_rng_impl": "unsafe_rbg"},
    ],
)
def test_invalid_metadata_is_rejected_before_orbax_save(replace_kwargs):
    state, network, algorithm = _initialized_state()
    metadata = dataclasses.replace(
        _metadata(state, network, algorithm),
        **replace_kwargs,
    )
    manager = _RejectSaveManager()

    with pytest.raises(ValueError, match=r".+"):
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=metadata,
        )

    assert not manager.called


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("state", "state"), True),
        (("state", "inc"), True),
        (("state", "inc"), 2),
        (("has_uint32",), -1),
        (("has_uint32",), 2),
        (("uinteger",), True),
        (("uinteger",), -1),
        (("uinteger",), 2**32),
    ],
)
def test_replay_rng_state_requires_exact_pcg64_integers(path: tuple[str, ...], value: object):
    state, network, algorithm = _initialized_state()
    replay_rng_state = np.random.Generator(np.random.PCG64(123)).bit_generator.state
    target = replay_rng_state
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    metadata = dataclasses.replace(
        _metadata(state, network, algorithm),
        replay_rng_state=replay_rng_state,
    )
    manager = _RejectSaveManager()

    with pytest.raises(ValueError, match="replay_rng_state"):
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=metadata,
        )

    assert not manager.called


@pytest.mark.parametrize(
    "field",
    ["network_config", "algorithm_config"],
)
def test_metadata_requires_exact_shared_config_dataclass(field: str):
    state, network, algorithm = _initialized_state()
    metadata = dataclasses.replace(
        _metadata(state, network, algorithm),
        **{field: dataclasses.asdict(getattr(_metadata(state, network, algorithm), field))},
    )
    manager = _RejectSaveManager()

    with pytest.raises(ValueError, match=field):
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=metadata,
        )

    assert not manager.called


def test_load_rejects_noncanonical_and_exact_schema_mutations(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    root = tmp_path / "mutations"
    manager = _manager(root)
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=_metadata(state, network, algorithm),
        )
        manager.wait_until_finished()
    finally:
        manager.close()

    metadata_path = root / "3" / "assets" / rlt_stage2_checkpoints.RLT_METADATA_FILENAME
    original = metadata_path.read_bytes()

    metadata_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="canonical"):
        rlt_stage2_checkpoints.load_rlt_metadata(root / "3")

    value = json.loads(original)
    value["unexpected"] = 1
    metadata_path.write_bytes(identity.canonical_json_bytes(value))
    with pytest.raises(ValueError, match="exact schema"):
        rlt_stage2_checkpoints.load_rlt_metadata(root / "3")

    del value["unexpected"]
    del value["network_config"]["state_dim"]
    metadata_path.write_bytes(identity.canonical_json_bytes(value))
    with pytest.raises(ValueError, match="fields are invalid"):
        rlt_stage2_checkpoints.load_rlt_metadata(root / "3")


def test_load_rejects_schema_v1_metadata(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    root = tmp_path / "schema-v1"
    manager = _manager(root)
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=_metadata(state, network, algorithm),
        )
        manager.wait_until_finished()
    finally:
        manager.close()

    metadata_path = root / "3" / "assets" / rlt_stage2_checkpoints.RLT_METADATA_FILENAME
    payload = json.loads(metadata_path.read_bytes())
    payload["schema_version"] = 1
    metadata_path.write_bytes(identity.canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="schema_version"):
        rlt_stage2_checkpoints.load_rlt_metadata(root / "3")


def test_load_reads_at_most_metadata_limit_plus_one_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    step_dir = tmp_path / "3"
    assets = step_dir / "assets"
    assets.mkdir(parents=True)
    metadata_path = assets / rlt_stage2_checkpoints.RLT_METADATA_FILENAME
    metadata_path.write_bytes(b"x" * (rlt_stage2_checkpoints._MAX_METADATA_BYTES + 1))  # noqa: SLF001

    def forbidden_unbounded_read(_self):
        raise AssertionError("load_rlt_metadata used unbounded read_bytes")

    monkeypatch.setattr(epath.Path, "read_bytes", forbidden_unbounded_read)

    with pytest.raises(ValueError, match="bounded"):
        rlt_stage2_checkpoints.load_rlt_metadata(step_dir)


def test_load_rejects_step_directory_counter_mismatch(tmp_path: Path):
    state, network, algorithm = _initialized_state()
    source_root = tmp_path / "source"
    manager = _manager(source_root)
    try:
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            manager,
            state=state,
            metadata=_metadata(state, network, algorithm),
        )
        manager.wait_until_finished()
    finally:
        manager.close()

    wrong_step = tmp_path / "4"
    assets = wrong_step / "assets"
    assets.mkdir(parents=True)
    source = source_root / "3" / "assets" / rlt_stage2_checkpoints.RLT_METADATA_FILENAME
    (assets / rlt_stage2_checkpoints.RLT_METADATA_FILENAME).write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="directory step"):
        rlt_stage2_checkpoints.load_rlt_metadata(epath.Path(wrong_step))


def test_load_rejects_noncanonical_step_directory_name(tmp_path: Path):
    step_dir = tmp_path / "03"
    assets = step_dir / "assets"
    assets.mkdir(parents=True)
    (assets / rlt_stage2_checkpoints.RLT_METADATA_FILENAME).write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="canonical nonnegative integer"):
        rlt_stage2_checkpoints.load_rlt_metadata(step_dir)


def test_restore_latest_uses_latest_native_step(tmp_path: Path):
    initial_state, network, algorithm = _initialized_state()
    root = tmp_path / "latest"
    manager = _manager(root)
    try:
        for step in (2, 3):
            state = initial_state.replace(
                critic_step=jnp.asarray(step, dtype=jnp.int32),
                round_critic_step=jnp.asarray(step - 1, dtype=jnp.int32),
            )
            rlt_stage2_checkpoints.save_rlt_checkpoint(
                manager,
                state=state,
                metadata=_metadata(
                    state,
                    network,
                    algorithm,
                    round_start_step=1,
                    round_critic_updates=2,
                    round_complete=step == 3,
                ),
            )
            manager.wait_until_finished()

        target_state, _, _ = rlt_td3_state.initialize_train_state(
            rlt_actor_critic.RLTActor(network),
            rlt_actor_critic.RLTCritic(network),
            algorithm,
            jax.random.key(4),
        )
        restored, metadata = rlt_stage2_checkpoints.restore_rlt_checkpoint(
            manager,
            target_state=target_state,
        )
        assert int(restored.critic_step) == 3
        assert metadata.critic_step == 3
    finally:
        manager.close()
