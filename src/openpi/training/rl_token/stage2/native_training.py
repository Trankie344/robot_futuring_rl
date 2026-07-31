"""Manual-round TD3 training backed by native OpenPI checkpoints."""

from __future__ import annotations

import copy
import dataclasses
import operator
from pathlib import Path
import re
import sys
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import checkpoints
from openpi.training.rl_token.stage2 import checkpoints as rlt_stage2_checkpoints
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import replay
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state
from openpi.training.rl_token.stage2 import trainer

_ROUND_ID = re.compile(r"round_([0-9]{6})")
_STEP_NAME = re.compile(r"(?:0|[1-9][0-9]*)")
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SEED = 2**32 - 1
_DEFAULT_PROMPT = "fold clothes"
_CACHE_FEATURE_FIELDS = (
    "feature_identity",
    "checkpoint_sha256",
    "norm_stats_sha256",
    "loaded_parameter_sha256",
    "loaded_norm_stats_sha256",
    "stage1_config",
    "stage2_config",
    "stage1_checkpoint_step",
    "reward_source",
    "reward_label_values",
    "completion_label",
    "reward_aggregation",
    "reward_schema_version",
    "asset_id",
    "sampler_num_steps",
    "default_prompt",
    "code_commit",
)


def _round_index(value: object) -> int:
    match = _ROUND_ID.fullmatch(value) if type(value) is str else None
    if match is None or int(match.group(1)) <= 0:
        raise ValueError("round_id must be a positive exact round_NNNNNN string")
    return int(match.group(1))


def _exact_seed(value: object) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"seed must be an exact integer in [0, {_MAX_SEED}]")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"seed must be an exact integer in [0, {_MAX_SEED}]") from exc
    if not 0 <= normalized <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")
    return int(normalized)


@dataclasses.dataclass(frozen=True)
class NativeRoundConfig:
    """All explicit inputs for one manually started Stage 2 training round."""

    checkpoint_dir: Path
    round_id: str
    admission: Path
    replay_snapshot: Path
    parent_checkpoint: Path | None = None
    seed: int = 0
    resume: bool = False
    overwrite: bool = False
    runtime: trainer.Stage2TrainerConfig = dataclasses.field(default_factory=trainer.Stage2TrainerConfig)

    def __post_init__(self) -> None:
        _round_index(self.round_id)
        if type(self.resume) is not bool or type(self.overwrite) is not bool:
            raise ValueError("resume and overwrite must be exact booleans")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        if self.resume and self.parent_checkpoint is not None:
            raise ValueError("resume cannot be combined with parent_checkpoint")
        if not isinstance(self.runtime, trainer.Stage2TrainerConfig):
            raise ValueError("runtime must be a Stage2TrainerConfig")
        self.runtime.validate()
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "admission", Path(self.admission))
        object.__setattr__(self, "replay_snapshot", Path(self.replay_snapshot))
        if self.parent_checkpoint is not None:
            object.__setattr__(self, "parent_checkpoint", Path(self.parent_checkpoint))
        object.__setattr__(self, "seed", _exact_seed(self.seed))


@dataclasses.dataclass
class NativeRoundRuntime:
    """Open resources and mutable training state for one manual round."""

    manager: ocp.CheckpointManager
    state: rlt_td3_state.RLTTrainState
    actor: rlt_actor_critic.RLTActor
    critic: rlt_actor_critic.RLTCritic
    actor_tx: object
    critic_tx: object
    replay_buffer: replay.ReplayBuffer
    replay_rng: np.random.Generator
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata
    complete: bool


@dataclasses.dataclass(frozen=True)
class _FrozenFeatureContract:
    feature_identity: str
    frozen_params_sha256: str
    norm_stats_sha256: str
    sampler_num_steps: int
    stage1_config: str
    stage2_config: str
    base_checkpoint_step: int


def _state_counter(state: rlt_td3_state.RLTTrainState, field: str) -> int:
    value = np.asarray(jax.device_get(getattr(state, field)))
    if value.shape != () or value.dtype != np.dtype(np.int32):
        raise ValueError(f"state.{field} must be a scalar int32 array")
    return int(value)


def _new_network_state(config: NativeRoundConfig):
    actor = rlt_actor_critic.RLTActor(config.runtime.network)
    critic = rlt_actor_critic.RLTCritic(config.runtime.network)
    state, actor_tx, critic_tx = rlt_td3_state.initialize_train_state(
        actor,
        critic,
        config.runtime.algorithm,
        jax.random.key(config.seed),
    )
    return state, actor, critic, actor_tx, critic_tx


def _generator_from_state(state: dict[str, object]) -> np.random.Generator:
    value = np.random.Generator(np.random.PCG64())
    try:
        value.bit_generator.state = copy.deepcopy(state)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint replay RNG state is invalid") from exc
    return value


def _new_generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _manifest_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(f"replay cache {field} must be a lowercase SHA-256")
    return value


def _feature_contract_from_replay(replay_buffer: object) -> _FrozenFeatureContract:
    try:
        snapshot = replay_buffer.snapshot
        verifications = tuple(replay_buffer.shard_verifications)
        records = tuple(snapshot.shards)
    except (AttributeError, TypeError) as exc:
        raise ValueError("replay buffer does not expose an authenticated feature catalog") from exc
    if not records or len(verifications) != len(records):
        raise ValueError("replay authenticated shard catalog length mismatch")

    feature_identity = _manifest_sha256(snapshot.feature_identity, field="feature_identity")
    baseline: dict[str, object] | None = None
    for index, verification in enumerate(verifications):
        try:
            manifest = verification.manifest
        except AttributeError as exc:
            raise ValueError(f"replay shard {index} has no authenticated manifest") from exc
        if type(manifest) is not dict:
            raise ValueError(f"replay shard {index} authenticated manifest must be a JSON object")
        missing = [field for field in _CACHE_FEATURE_FIELDS if field not in manifest]
        if missing:
            raise ValueError(f"replay shard {index} cache identity is missing {missing[0]}")

        current = {field: manifest[field] for field in _CACHE_FEATURE_FIELDS}
        for field in (
            "feature_identity",
            "checkpoint_sha256",
            "norm_stats_sha256",
            "loaded_parameter_sha256",
            "loaded_norm_stats_sha256",
        ):
            _manifest_sha256(current[field], field=field)
        if current["feature_identity"] != feature_identity:
            raise ValueError(f"replay shard {index} feature_identity differs from ReplaySnapshot")
        if current["stage1_config"] != rlt_stage2_checkpoints.RLT_STAGE1_CONFIG_NAME:
            raise ValueError(f"replay shard {index} stage1_config differs from the locked Stage 1 config")
        if current["stage2_config"] != rlt_stage2_checkpoints.RLT_STAGE2_CONFIG_NAME:
            raise ValueError(f"replay shard {index} stage2_config differs from the locked Stage 2 config")
        checkpoint_step = current["stage1_checkpoint_step"]
        if type(checkpoint_step) is not int or checkpoint_step < 0:
            raise ValueError(f"replay shard {index} stage1_checkpoint_step must be an exact nonnegative integer")
        if current["reward_source"] != "tristate":
            raise ValueError(f"replay shard {index} reward_source must be 'tristate'")
        if current["reward_label_values"] != [-1, 0, 1, 2]:
            raise ValueError(f"replay shard {index} reward_label_values must be [-1, 0, 1, 2]")
        if current["completion_label"] != 2:
            raise ValueError(f"replay shard {index} completion_label must be 2")
        if current["reward_aggregation"] != "sum_20_frames":
            raise ValueError(f"replay shard {index} reward_aggregation must be sum_20_frames")
        if current["reward_schema_version"] != 1:
            raise ValueError(f"replay shard {index} reward_schema_version must be 1")
        if current["asset_id"] != rlt_stage2_checkpoints.RLT_ASSET_ID:
            raise ValueError(f"replay shard {index} asset_id differs from the locked Stage 2 asset")
        if current["default_prompt"] != _DEFAULT_PROMPT:
            raise ValueError(f"replay shard {index} default_prompt must be {_DEFAULT_PROMPT!r}")
        sampler_num_steps = current["sampler_num_steps"]
        if type(sampler_num_steps) is not int or sampler_num_steps <= 0:
            raise ValueError(f"replay shard {index} sampler_num_steps must be an exact positive integer")
        code_commit = current["code_commit"]
        if type(code_commit) is not str or not code_commit:
            raise ValueError(f"replay shard {index} code_commit must be a nonempty string")

        if baseline is None:
            baseline = current
        else:
            for field in _CACHE_FEATURE_FIELDS:
                if current[field] != baseline[field]:
                    raise ValueError(f"replay cache {field} differs across authenticated shards")

    assert baseline is not None
    return _FrozenFeatureContract(
        feature_identity=feature_identity,
        frozen_params_sha256=str(baseline["checkpoint_sha256"]),
        norm_stats_sha256=str(baseline["norm_stats_sha256"]),
        sampler_num_steps=int(baseline["sampler_num_steps"]),
        stage1_config=str(baseline["stage1_config"]),
        stage2_config=str(baseline["stage2_config"]),
        base_checkpoint_step=int(baseline["stage1_checkpoint_step"]),
    )


def _build_metadata(
    *,
    config: NativeRoundConfig,
    state: rlt_td3_state.RLTTrainState,
    admission_sha256: str,
    replay_snapshot_sha256: str,
    round_start_step: int,
    round_critic_updates: int,
    replay_rng_state: dict[str, object],
    round_complete: bool,
    feature_contract: _FrozenFeatureContract,
) -> rlt_stage2_checkpoints.RLTCheckpointMetadata:
    return rlt_stage2_checkpoints.RLTCheckpointMetadata(
        schema_version=3,
        stage1_config=feature_contract.stage1_config,
        stage2_config=feature_contract.stage2_config,
        asset_id=rlt_stage2_checkpoints.RLT_ASSET_ID,
        base_checkpoint_step=feature_contract.base_checkpoint_step,
        reward_source="tristate",
        reward_label_values=(-1, 0, 1, 2),
        completion_label=2,
        reward_aggregation="sum_20_frames",
        reward_schema_version=1,
        feature_identity=feature_contract.feature_identity,
        frozen_params_sha256=feature_contract.frozen_params_sha256,
        norm_stats_sha256=feature_contract.norm_stats_sha256,
        sampler_num_steps=feature_contract.sampler_num_steps,
        round_id=config.round_id,
        admission_sha256=admission_sha256,
        replay_snapshot_sha256=replay_snapshot_sha256,
        network_config=config.runtime.network,
        algorithm_config=config.runtime.algorithm,
        batch_size=config.runtime.batch_size,
        round_start_step=round_start_step,
        round_critic_updates=round_critic_updates,
        critic_step=_state_counter(state, "critic_step"),
        round_critic_step=_state_counter(state, "round_critic_step"),
        replay_rng_state=copy.deepcopy(replay_rng_state),
        jax_rng_impl=jax.random.key_impl(state.rng),
        round_complete=round_complete,
    )


def _metadata_for_state(
    template: rlt_stage2_checkpoints.RLTCheckpointMetadata,
    *,
    state: rlt_td3_state.RLTTrainState,
    replay_rng_state: dict[str, object],
    round_complete: bool,
) -> rlt_stage2_checkpoints.RLTCheckpointMetadata:
    return dataclasses.replace(
        template,
        critic_step=_state_counter(state, "critic_step"),
        round_critic_step=_state_counter(state, "round_critic_step"),
        replay_rng_state=copy.deepcopy(replay_rng_state),
        jax_rng_impl=jax.random.key_impl(state.rng),
        round_complete=round_complete,
    )


def _require_runtime_metadata(
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
    *,
    config: NativeRoundConfig,
) -> None:
    expected = {
        "network_config": config.runtime.network,
        "algorithm_config": config.runtime.algorithm,
        "batch_size": config.runtime.batch_size,
    }
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise ValueError(f"checkpoint {field} mismatch")


def _require_feature_metadata(
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
    *,
    feature_contract: _FrozenFeatureContract,
) -> None:
    expected = dataclasses.asdict(feature_contract)
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise ValueError(f"checkpoint {field} mismatch")


def _require_resume_metadata(
    metadata: rlt_stage2_checkpoints.RLTCheckpointMetadata,
    *,
    config: NativeRoundConfig,
    admission_sha256: str,
    replay_snapshot_sha256: str,
    round_critic_updates: int,
    feature_contract: _FrozenFeatureContract,
) -> None:
    _require_runtime_metadata(metadata, config=config)
    _require_feature_metadata(metadata, feature_contract=feature_contract)
    expected = {
        "round_id": config.round_id,
        "admission_sha256": admission_sha256,
        "replay_snapshot_sha256": replay_snapshot_sha256,
        "round_critic_updates": round_critic_updates,
    }
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise ValueError(f"checkpoint {field} mismatch")


def _close_resources(
    *,
    manager: ocp.CheckpointManager | None,
    replay_buffer: object | None,
    wait: bool,
    primary: BaseException | None,
) -> None:
    errors: list[tuple[str, BaseException]] = []
    if manager is not None:
        if wait:
            try:
                manager.wait_until_finished()
            except BaseException as exc:
                errors.append(("waiting for checkpoint manager", exc))
        try:
            manager.close()
        except BaseException as exc:
            errors.append(("closing checkpoint manager", exc))
    if replay_buffer is not None:
        try:
            replay_buffer.close()
        except BaseException as exc:
            errors.append(("closing replay buffer", exc))
    if primary is not None:
        for label, error in errors:
            primary.add_note(f"{label}: {error}")
        return
    if errors:
        error = RuntimeError("native round resource cleanup failed")
        for label, cause in errors:
            error.add_note(f"{label}: {cause}")
        raise error from errors[0][1]


def _open_parent(
    path: Path,
    *,
    target_state: rlt_td3_state.RLTTrainState,
) -> tuple[rlt_td3_state.RLTTrainState, rlt_stage2_checkpoints.RLTCheckpointMetadata]:
    step_path = Path(path)
    if _STEP_NAME.fullmatch(step_path.name) is None or not step_path.is_dir():
        raise ValueError("parent_checkpoint must be an existing canonical numeric OpenPI step directory")
    step = int(step_path.name)
    manager: ocp.CheckpointManager | None = None
    try:
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            step_path.parent,
            keep_period=None,
            overwrite=False,
            resume=True,
            max_to_keep=3,
        )
        if not resuming or step not in manager.all_steps():
            raise ValueError("parent_checkpoint is not a managed OpenPI checkpoint step")
        return rlt_stage2_checkpoints.restore_rlt_checkpoint(
            manager,
            target_state=target_state,
            step=step,
        )
    finally:
        _close_resources(
            manager=manager,
            replay_buffer=None,
            wait=False,
            primary=sys.exception(),
        )


def _validate_replay_tail(replay_buffer: object, admitted: object) -> None:
    try:
        shards = replay_buffer.snapshot.shards
        final = shards[-1]
        batch_id = final["batch_id"]
        admission_sha256 = final["admission_sha256"]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("replay snapshot does not expose a valid final admission record") from exc
    if batch_id != admitted.batch_id or admission_sha256 != admitted.sha256:
        raise ValueError("replay final shard does not match the current admission")


def _require_parent_replay_lineage(
    snapshot: replay.ReplaySnapshot,
    *,
    parent_replay_snapshot_sha256: str,
) -> None:
    try:
        history = tuple(dict(record) for record in snapshot.shards[:-1])
        if not history:
            raise ValueError
        payload = {
            "schema_version": snapshot.schema_version,
            "feature_identity": snapshot.feature_identity,
            "total_transitions": history[-1]["end"],
            "shards": list(history),
        }
        history_sha256 = identity.sha256_json(payload)
    except (AttributeError, IndexError, KeyError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("replay history does not match the parent checkpoint replay snapshot") from exc
    if history_sha256 != parent_replay_snapshot_sha256:
        raise ValueError("replay history does not match the parent checkpoint replay snapshot")


def _require_checkpoint_outside_parent(
    checkpoint_dir: Path,
    *,
    parent_checkpoint: Path,
) -> None:
    current = checkpoint_dir.resolve()
    parent_root = parent_checkpoint.parent.resolve()
    if current == parent_root or current in parent_root.parents or parent_root in current.parents:
        raise ValueError("checkpoint_dir must not overlap the parent checkpoint root")


def prepare_native_round(config: NativeRoundConfig) -> NativeRoundRuntime:
    """Open and validate one new or resumed manual training round."""
    if not isinstance(config, NativeRoundConfig):
        raise ValueError("config must be a NativeRoundConfig")
    index = _round_index(config.round_id)
    if index == 1 and config.parent_checkpoint is not None:
        raise ValueError("round_000001 does not accept a parent checkpoint")
    if index > 1 and not config.resume and config.parent_checkpoint is None:
        raise ValueError(f"{config.round_id} requires an explicit parent checkpoint")
    if config.resume and not config.checkpoint_dir.is_dir():
        raise ValueError("resume checkpoint_dir must already exist")
    if not config.resume and index > 1:
        assert config.parent_checkpoint is not None
        _require_checkpoint_outside_parent(
            config.checkpoint_dir,
            parent_checkpoint=config.parent_checkpoint,
        )

    current_manager: ocp.CheckpointManager | None = None
    replay_buffer: replay.ReplayBuffer | None = None
    try:
        admitted = admission.open_admission(config.admission)
        if admitted.round_id != config.round_id:
            raise ValueError(f"admission round_id mismatch: {admitted.round_id!r} != {config.round_id!r}")
        replay_buffer = replay.ReplayBuffer.open(
            config.replay_snapshot,
            max_open_shards=config.runtime.replay_max_open_shards,
        )
        _validate_replay_tail(replay_buffer, admitted)
        feature_contract = _feature_contract_from_replay(replay_buffer)
        if index == 1 and len(replay_buffer.snapshot.shards) != 1:
            raise ValueError("round_000001 must contain exactly one replay shard")
        replay_snapshot_sha256 = replay_buffer.snapshot.sha256
        round_critic_updates = int(config.runtime.algorithm.utd_ratio) * int(admitted.chunk_equivalents)
        if round_critic_updates <= 0:
            raise ValueError("round_critic_updates must be positive")

        initial_state, actor, critic, actor_tx, critic_tx = _new_network_state(config)
        prepared_parent: (
            tuple[
                rlt_td3_state.RLTTrainState,
                np.random.Generator,
                rlt_stage2_checkpoints.RLTCheckpointMetadata,
            ]
            | None
        ) = None
        if not config.resume and index > 1:
            assert config.parent_checkpoint is not None
            parent_state, parent_metadata = _open_parent(
                config.parent_checkpoint,
                target_state=initial_state,
            )
            expected_parent_round = f"round_{index - 1:06d}"
            if parent_metadata.round_id != expected_parent_round:
                raise ValueError(
                    "parent checkpoint is not from the exact previous round: "
                    f"{parent_metadata.round_id!r} != {expected_parent_round!r}"
                )
            if not parent_metadata.round_complete:
                raise ValueError("parent checkpoint must be complete")
            _require_runtime_metadata(parent_metadata, config=config)
            _require_feature_metadata(parent_metadata, feature_contract=feature_contract)
            _require_parent_replay_lineage(
                replay_buffer.snapshot,
                parent_replay_snapshot_sha256=parent_metadata.replay_snapshot_sha256,
            )
            parent_round_state = rlt_td3_state.start_new_round(parent_state)
            parent_replay_rng = _generator_from_state(parent_metadata.replay_rng_state)
            parent_round_metadata = _build_metadata(
                config=config,
                state=parent_round_state,
                admission_sha256=admitted.sha256,
                replay_snapshot_sha256=replay_snapshot_sha256,
                round_start_step=parent_metadata.critic_step,
                round_critic_updates=round_critic_updates,
                replay_rng_state=parent_replay_rng.bit_generator.state,
                round_complete=False,
                feature_contract=feature_contract,
            )
            prepared_parent = (parent_round_state, parent_replay_rng, parent_round_metadata)

        current_manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=None,
            overwrite=config.overwrite,
            resume=config.resume,
            max_to_keep=3,
        )
        if config.resume and not resuming:
            raise ValueError("resume checkpoint_dir contains no managed checkpoint step")

        if config.resume:
            state, metadata = rlt_stage2_checkpoints.restore_rlt_checkpoint(
                current_manager,
                target_state=initial_state,
            )
            _require_resume_metadata(
                metadata,
                config=config,
                admission_sha256=admitted.sha256,
                replay_snapshot_sha256=replay_snapshot_sha256,
                round_critic_updates=round_critic_updates,
                feature_contract=feature_contract,
            )
            replay_rng = _generator_from_state(metadata.replay_rng_state)
            complete = metadata.round_complete
        elif index == 1:
            state = initial_state
            replay_rng = _new_generator(config.seed)
            metadata = _build_metadata(
                config=config,
                state=state,
                admission_sha256=admitted.sha256,
                replay_snapshot_sha256=replay_snapshot_sha256,
                round_start_step=0,
                round_critic_updates=round_critic_updates,
                replay_rng_state=replay_rng.bit_generator.state,
                round_complete=False,
                feature_contract=feature_contract,
            )
            complete = False
        else:
            assert prepared_parent is not None
            state, replay_rng, metadata = prepared_parent
            complete = False

        return NativeRoundRuntime(
            manager=current_manager,
            state=state,
            actor=actor,
            critic=critic,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            replay_buffer=replay_buffer,
            replay_rng=replay_rng,
            metadata=metadata,
            complete=complete,
        )
    except BaseException:
        _close_resources(
            manager=current_manager,
            replay_buffer=replay_buffer,
            wait=True,
            primary=sys.exception(),
        )
        raise


def run_native_round(config: NativeRoundConfig) -> Path:
    """Run all remaining updates for a manual round and return its final step."""
    runtime = prepare_native_round(config)
    try:
        current_step = Path(runtime.manager.directory) / str(runtime.metadata.critic_step)
        if runtime.complete:
            return current_step

        def save_temporary(
            state: rlt_td3_state.RLTTrainState,
            replay_rng_state: dict[str, Any],
        ) -> None:
            metadata = _metadata_for_state(
                runtime.metadata,
                state=state,
                replay_rng_state=replay_rng_state,
                round_complete=False,
            )
            rlt_stage2_checkpoints.save_rlt_checkpoint(
                runtime.manager,
                state=state,
                metadata=metadata,
            )

        with trainer.JsonlMetricSink(config.checkpoint_dir / "metrics.jsonl") as metric_sink:
            result = trainer.run_updates(
                state=runtime.state,
                actor=runtime.actor,
                critic=runtime.critic,
                actor_tx=runtime.actor_tx,
                critic_tx=runtime.critic_tx,
                algorithm=config.runtime.algorithm,
                replay_buffer=runtime.replay_buffer,
                replay_rng=runtime.replay_rng,
                round_start_step=runtime.metadata.round_start_step,
                round_critic_updates=runtime.metadata.round_critic_updates,
                batch_size=config.runtime.batch_size,
                log_interval=config.runtime.log_interval,
                temp_checkpoint_interval=config.runtime.temp_checkpoint_interval,
                metric_sink=metric_sink,
                checkpoint_sink=save_temporary,
            )

        final_metadata = _metadata_for_state(
            runtime.metadata,
            state=result.state,
            replay_rng_state=result.replay_rng_state,
            round_complete=True,
        )
        rlt_stage2_checkpoints.save_rlt_checkpoint(
            runtime.manager,
            state=result.state,
            metadata=final_metadata,
        )
        runtime.state = result.state
        runtime.metadata = final_metadata
        runtime.complete = True
        return Path(runtime.manager.directory) / str(final_metadata.critic_step)
    finally:
        _close_resources(
            manager=runtime.manager,
            replay_buffer=runtime.replay_buffer,
            wait=True,
            primary=sys.exception(),
        )
