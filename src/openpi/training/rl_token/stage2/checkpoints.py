"""Native OpenPI/Orbax checkpoints for RL-token Stage 2 training."""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import copy
import dataclasses
import json
import math
import operator
import re
from typing import Any, Protocol

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import td3 as rlt_td3
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state

RLT_METADATA_FILENAME = "rlt_stage2.json"
RLT_STAGE1_CONFIG_NAME = "rl_token_stage1"
RLT_STAGE2_CONFIG_NAME = "rl_token_stage2"
RLT_CONFIG_NAME = RLT_STAGE1_CONFIG_NAME
RLT_ASSET_ID = "lite0030_joints_fps20_openpi_drop_last4s_min20s"
RLT_DEFAULT_BASE_CHECKPOINT_STEP = 54999
RLT_BASE_CHECKPOINT_STEP = RLT_DEFAULT_BASE_CHECKPOINT_STEP

_SCHEMA_VERSION = 3
_MAX_METADATA_BYTES = 1024 * 1024
_ROUND_ID = re.compile(r"round_([0-9]{6})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_STEP = re.compile(r"(?:0|[1-9][0-9]*)")
_PCG64_FIELDS = frozenset({"bit_generator", "state", "has_uint32", "uinteger"})
_PCG64_STATE_FIELDS = frozenset({"state", "inc"})


@dataclasses.dataclass(frozen=True)
class RLTCheckpointMetadata:
    """Small resume/deployment contract stored in the native assets item."""

    schema_version: int
    stage1_config: str
    stage2_config: str
    asset_id: str
    base_checkpoint_step: int
    reward_source: str
    reward_label_values: tuple[int, int, int, int]
    completion_label: int
    reward_aggregation: str
    reward_schema_version: int
    feature_identity: str
    frozen_params_sha256: str
    norm_stats_sha256: str
    sampler_num_steps: int
    round_id: str
    admission_sha256: str
    replay_snapshot_sha256: str
    network_config: rlt_actor_critic.RLTActorCriticConfig
    algorithm_config: rlt_td3.TD3Config
    batch_size: int
    round_start_step: int
    round_critic_updates: int
    critic_step: int
    round_critic_step: int
    replay_rng_state: dict[str, object]
    jax_rng_impl: str
    round_complete: bool


_METADATA_FIELDS = frozenset(field.name for field in dataclasses.fields(RLTCheckpointMetadata))


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """Orbax handler for the small canonical metadata asset."""

    def save(self, directory: epath.Path, args: CallbackSave) -> None:
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs):
    pass


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    max_to_keep: int = 3,
) -> tuple[ocp.CheckpointManager, bool]:
    """Create the Stage 2 Orbax manager without using OpenPI's original entrypoint."""
    if isinstance(max_to_keep, bool | np.bool_):
        raise ValueError("max_to_keep must be a positive integer")
    try:
        max_to_keep = operator.index(max_to_keep)
    except TypeError as exc:
        raise ValueError("max_to_keep must be a positive integer") from exc
    if max_to_keep <= 0:
        raise ValueError("max_to_keep must be a positive integer")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    root = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if root.exists():
        if overwrite:
            root.rmtree()
        elif resume:
            resuming = True
        else:
            raise FileExistsError(f"Checkpoint directory already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manager = ocp.CheckpointManager(
        root,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    if resuming and not tuple(manager.all_steps()):
        resuming = False
    return manager, resuming


def _exact_int(value: object, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        relation = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{field} must be an exact {relation} integer")
    return value


def _state_counter(value: object, *, field: str) -> int:
    try:
        host = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"state.{field} must be a scalar int32 array") from exc
    if host.shape != () or host.dtype != np.dtype(np.int32):
        raise ValueError(f"state.{field} must be a scalar int32 array")
    return int(host)


def _typed_scalar_key_impl(value: object, *, field: str):
    rng_dtype = getattr(value, "dtype", None)
    if (
        rng_dtype is None
        or not jax.dtypes.issubdtype(rng_dtype, jax.dtypes.prng_key)
        or getattr(value, "shape", None) != ()
    ):
        raise ValueError(f"{field} must be one typed JAX PRNG key")
    try:
        jax.random.key_data(value)
        return jax.random.key_impl(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be one typed JAX PRNG key") from exc


def _validate_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 string")
    return value


def _validate_round_id(value: object) -> str:
    match = _ROUND_ID.fullmatch(value) if type(value) is str else None
    if match is None or int(match.group(1)) <= 0:
        raise ValueError("round_id must be a positive exact round_NNNNNN string")
    return value


def _validate_json_tree(value: object, *, field: str, seen: set[int] | None = None) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite JSON values")
        return
    if type(value) not in {dict, list}:
        raise ValueError(f"{field} must contain exact JSON values")
    if seen is None:
        seen = set()
    identity_value = id(value)
    if identity_value in seen:
        raise ValueError(f"{field} must not contain shared or cyclic containers")
    seen.add(identity_value)
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError(f"{field} keys must be exact strings")
            for child in value.values():
                _validate_json_tree(child, field=field, seen=seen)
        else:
            for child in value:
                _validate_json_tree(child, field=field, seen=seen)
    finally:
        seen.remove(identity_value)


def _validate_replay_rng_state(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("replay_rng_state must be an exact JSON object")
    candidate = copy.deepcopy(value)
    _validate_json_tree(candidate, field="replay_rng_state")
    if frozenset(candidate) != _PCG64_FIELDS or candidate["bit_generator"] != "PCG64":
        raise ValueError("replay_rng_state must have the exact PCG64 schema")
    core_state = candidate["state"]
    if type(core_state) is not dict or frozenset(core_state) != _PCG64_STATE_FIELDS:
        raise ValueError("replay_rng_state must have the exact PCG64 schema")
    for field in _PCG64_STATE_FIELDS:
        field_value = core_state[field]
        if type(field_value) is not int or not 0 <= field_value < 2**128:
            raise ValueError(f"replay_rng_state.state.{field} must be an exact uint128 integer")
    if core_state["inc"] % 2 != 1:
        raise ValueError("replay_rng_state.state.inc must be odd")
    has_uint32 = candidate["has_uint32"]
    if type(has_uint32) is not int or has_uint32 not in (0, 1):
        raise ValueError("replay_rng_state.has_uint32 must be exact integer 0 or 1")
    uinteger = candidate["uinteger"]
    if type(uinteger) is not int or not 0 <= uinteger < 2**32:
        raise ValueError("replay_rng_state.uinteger must be an exact uint32 integer")
    bit_generator = np.random.PCG64()
    try:
        bit_generator.state = candidate
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError("replay_rng_state is not a valid PCG64 state") from exc
    normalized = copy.deepcopy(bit_generator.state)
    if normalized != candidate:
        raise ValueError("replay_rng_state is not the canonical PCG64 state")
    return normalized


def _network_payload(config: rlt_actor_critic.RLTActorCriticConfig) -> dict[str, object]:
    if not isinstance(config, rlt_actor_critic.RLTActorCriticConfig):
        raise ValueError("network_config must be an RLTActorCriticConfig")
    payload = dataclasses.asdict(config)
    for field in ("actor_hidden_dims", "critic_hidden_dims"):
        payload[field] = list(payload[field])
    return payload


def _algorithm_payload(config: rlt_td3.TD3Config) -> dict[str, object]:
    if not isinstance(config, rlt_td3.TD3Config):
        raise ValueError("algorithm_config must be a TD3Config")
    return dataclasses.asdict(config)


def _metadata_payload(metadata: RLTCheckpointMetadata) -> dict[str, object]:
    return {
        "schema_version": metadata.schema_version,
        "stage1_config": metadata.stage1_config,
        "stage2_config": metadata.stage2_config,
        "asset_id": metadata.asset_id,
        "base_checkpoint_step": metadata.base_checkpoint_step,
        "reward_source": metadata.reward_source,
        "reward_label_values": list(metadata.reward_label_values),
        "completion_label": metadata.completion_label,
        "reward_aggregation": metadata.reward_aggregation,
        "reward_schema_version": metadata.reward_schema_version,
        "feature_identity": metadata.feature_identity,
        "frozen_params_sha256": metadata.frozen_params_sha256,
        "norm_stats_sha256": metadata.norm_stats_sha256,
        "sampler_num_steps": metadata.sampler_num_steps,
        "round_id": metadata.round_id,
        "admission_sha256": metadata.admission_sha256,
        "replay_snapshot_sha256": metadata.replay_snapshot_sha256,
        "network_config": _network_payload(metadata.network_config),
        "algorithm_config": _algorithm_payload(metadata.algorithm_config),
        "batch_size": metadata.batch_size,
        "round_start_step": metadata.round_start_step,
        "round_critic_updates": metadata.round_critic_updates,
        "critic_step": metadata.critic_step,
        "round_critic_step": metadata.round_critic_step,
        "replay_rng_state": copy.deepcopy(metadata.replay_rng_state),
        "jax_rng_impl": metadata.jax_rng_impl,
        "round_complete": metadata.round_complete,
    }


def _validate_metadata(
    metadata: object,
    *,
    state: rlt_td3_state.RLTTrainState | None = None,
    expected_step: int | None = None,
) -> RLTCheckpointMetadata:
    if type(metadata) is not RLTCheckpointMetadata:
        raise ValueError("metadata must be an exact RLTCheckpointMetadata")
    if type(metadata.schema_version) is not int or metadata.schema_version != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be exact integer {_SCHEMA_VERSION}")
    if type(metadata.stage1_config) is not str or metadata.stage1_config != RLT_STAGE1_CONFIG_NAME:
        raise ValueError(f"stage1_config must be {RLT_STAGE1_CONFIG_NAME!r}")
    if type(metadata.stage2_config) is not str or metadata.stage2_config != RLT_STAGE2_CONFIG_NAME:
        raise ValueError(f"stage2_config must be {RLT_STAGE2_CONFIG_NAME!r}")
    if type(metadata.asset_id) is not str or metadata.asset_id != RLT_ASSET_ID:
        raise ValueError(f"asset_id must be {RLT_ASSET_ID!r}")
    _exact_int(metadata.base_checkpoint_step, field="base_checkpoint_step", minimum=0)
    if metadata.reward_source != "tristate":
        raise ValueError("reward_source must be 'tristate'")
    if tuple(metadata.reward_label_values) != (-1, 0, 1, 2):
        raise ValueError("reward_label_values must be (-1, 0, 1, 2)")
    if type(metadata.completion_label) is not int or metadata.completion_label != 2:
        raise ValueError("completion_label must be exact integer 2")
    if metadata.reward_aggregation != "sum_20_frames":
        raise ValueError("reward_aggregation must be 'sum_20_frames'")
    if type(metadata.reward_schema_version) is not int or metadata.reward_schema_version != 1:
        raise ValueError("reward_schema_version must be exact integer 1")
    _validate_sha256(metadata.feature_identity, field="feature_identity")
    _validate_sha256(metadata.frozen_params_sha256, field="frozen_params_sha256")
    _validate_sha256(metadata.norm_stats_sha256, field="norm_stats_sha256")
    _exact_int(metadata.sampler_num_steps, field="sampler_num_steps", minimum=1)
    _validate_round_id(metadata.round_id)
    _validate_sha256(metadata.admission_sha256, field="admission_sha256")
    _validate_sha256(metadata.replay_snapshot_sha256, field="replay_snapshot_sha256")
    network_payload = _network_payload(metadata.network_config)
    rlt_actor_critic.decode_network_config(network_payload)
    algorithm_payload = _algorithm_payload(metadata.algorithm_config)
    rlt_td3.decode_td3_config(algorithm_payload)
    _exact_int(metadata.batch_size, field="batch_size", minimum=1)
    _exact_int(metadata.round_start_step, field="round_start_step", minimum=0)
    _exact_int(metadata.round_critic_updates, field="round_critic_updates", minimum=1)
    _exact_int(metadata.critic_step, field="critic_step", minimum=0)
    _exact_int(metadata.round_critic_step, field="round_critic_step", minimum=0)
    replay_rng_state = _validate_replay_rng_state(metadata.replay_rng_state)
    if type(metadata.jax_rng_impl) is not str:
        raise ValueError("jax_rng_impl must be an exact JAX PRNG implementation string")
    try:
        canonical_rng_impl = jax.random.key_impl(jax.random.key(0, impl=metadata.jax_rng_impl))
    except (TypeError, ValueError) as exc:
        raise ValueError("jax_rng_impl must name a supported JAX PRNG implementation") from exc
    if canonical_rng_impl != metadata.jax_rng_impl:
        raise ValueError("jax_rng_impl must use its canonical JAX PRNG implementation name")
    if type(metadata.round_complete) is not bool:
        raise ValueError("round_complete must be an exact boolean")
    if metadata.critic_step != metadata.round_start_step + metadata.round_critic_step:
        raise ValueError("critic_step must equal round_start_step plus round_critic_step")
    if metadata.round_critic_step > metadata.round_critic_updates:
        raise ValueError("round_critic_step must not exceed round_critic_updates")
    if metadata.round_complete != (metadata.round_critic_step == metadata.round_critic_updates):
        raise ValueError("round_complete must exactly describe completion of the round budget")
    if expected_step is not None and metadata.critic_step != expected_step:
        raise ValueError("checkpoint directory step does not match metadata critic_step")
    if state is not None:
        if not isinstance(state, rlt_td3_state.RLTTrainState):
            raise ValueError("state must be an RLTTrainState")
        critic_step = _state_counter(state.critic_step, field="critic_step")
        round_critic_step = _state_counter(state.round_critic_step, field="round_critic_step")
        if critic_step != metadata.critic_step or round_critic_step != metadata.round_critic_step:
            raise ValueError("metadata counters must equal RLTTrainState counters")
        state_rng_impl = _typed_scalar_key_impl(state.rng, field="state.rng")
        if state_rng_impl != metadata.jax_rng_impl:
            raise ValueError("metadata jax_rng_impl must equal the state RNG implementation")
    return dataclasses.replace(metadata, replay_rng_state=replay_rng_state)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite metadata JSON constant {value!r}")


def _decode_metadata(payload: bytes) -> RLTCheckpointMetadata:
    if type(payload) is not bytes or len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("RLT metadata must be bounded exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("RLT metadata is not valid canonical JSON") from exc
    _validate_json_tree(value, field="metadata")
    if type(value) is not dict or frozenset(value) != _METADATA_FIELDS:
        raise ValueError("RLT metadata must have the exact schema")
    try:
        canonical = identity.canonical_json_bytes(value)
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("RLT metadata cannot be encoded canonically") from exc
    if payload != canonical:
        raise ValueError("RLT metadata JSON is not canonical")
    try:
        network_config = rlt_actor_critic.decode_network_config(value["network_config"])
        algorithm_config = rlt_td3.decode_td3_config(value["algorithm_config"])
        metadata = RLTCheckpointMetadata(
            schema_version=value["schema_version"],
            stage1_config=value["stage1_config"],
            stage2_config=value["stage2_config"],
            asset_id=value["asset_id"],
            base_checkpoint_step=value["base_checkpoint_step"],
            reward_source=value["reward_source"],
            reward_label_values=tuple(value["reward_label_values"]),
            completion_label=value["completion_label"],
            reward_aggregation=value["reward_aggregation"],
            reward_schema_version=value["reward_schema_version"],
            feature_identity=value["feature_identity"],
            frozen_params_sha256=value["frozen_params_sha256"],
            norm_stats_sha256=value["norm_stats_sha256"],
            sampler_num_steps=value["sampler_num_steps"],
            round_id=value["round_id"],
            admission_sha256=value["admission_sha256"],
            replay_snapshot_sha256=value["replay_snapshot_sha256"],
            network_config=network_config,
            algorithm_config=algorithm_config,
            batch_size=value["batch_size"],
            round_start_step=value["round_start_step"],
            round_critic_updates=value["round_critic_updates"],
            critic_step=value["critic_step"],
            round_critic_step=value["round_critic_step"],
            replay_rng_state=value["replay_rng_state"],
            jax_rng_impl=value["jax_rng_impl"],
            round_complete=value["round_complete"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RLT metadata fields are invalid") from exc
    return _validate_metadata(metadata)


def save_rlt_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    state: rlt_td3_state.RLTTrainState,
    metadata: RLTCheckpointMetadata,
) -> None:
    """Asynchronously save a complete Stage 2 state and its online actor."""
    metadata = _validate_metadata(metadata, state=state)
    payload = identity.canonical_json_bytes(_metadata_payload(metadata))
    serializable_state = state.replace(rng=jax.random.key_data(state.rng))

    def save_assets(directory: epath.Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RLT_METADATA_FILENAME).write_bytes(payload)

    accepted = manager.save(
        metadata.critic_step,
        {
            "assets": save_assets,
            "train_state": serializable_state,
            "params": {"params": state.actor_params},
        },
    )
    if accepted is not True:
        raise RuntimeError(f"Orbax did not accept RLT checkpoint step {metadata.critic_step}")


def load_rlt_metadata(step_dir: epath.Path | str) -> RLTCheckpointMetadata:
    """Load and validate the canonical Stage 2 metadata asset."""
    step_dir = epath.Path(step_dir)
    if _CHECKPOINT_STEP.fullmatch(step_dir.name) is None:
        raise ValueError("checkpoint step directory must have a canonical nonnegative integer name")
    step = int(step_dir.name)
    path = step_dir / "assets" / RLT_METADATA_FILENAME
    try:
        with path.open("rb") as stream:
            payload = stream.read(_MAX_METADATA_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"RLT metadata is missing or unreadable: {path}") from exc
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("RLT metadata must be bounded exact bytes")
    metadata = _decode_metadata(payload)
    return _validate_metadata(metadata, expected_step=step)


def restore_rlt_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    target_state: rlt_td3_state.RLTTrainState,
    step: int | None = None,
) -> tuple[rlt_td3_state.RLTTrainState, RLTCheckpointMetadata]:
    """Restore a complete Stage 2 state against an explicitly initialized target."""
    if not isinstance(target_state, rlt_td3_state.RLTTrainState):
        raise ValueError("target_state must be an RLTTrainState")
    key_impl = _typed_scalar_key_impl(target_state.rng, field="target_state.rng")
    serializable_target = target_state.replace(rng=jax.random.key_data(target_state.rng))

    restore_step = manager.latest_step() if step is None else step
    if isinstance(restore_step, bool | np.bool_):
        raise ValueError("checkpoint step must be a nonnegative integer")
    try:
        restore_step = operator.index(restore_step)
    except TypeError as exc:
        raise ValueError("checkpoint step must be a nonnegative integer") from exc
    if restore_step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")

    metadata = load_rlt_metadata(epath.Path(manager.directory) / str(restore_step))
    if key_impl != metadata.jax_rng_impl:
        raise ValueError(
            f"target_state RNG implementation does not match checkpoint ({key_impl!r} != {metadata.jax_rng_impl!r})"
        )
    restored = manager.restore(
        restore_step,
        items={"train_state": serializable_target},
    )["train_state"]
    try:
        restored = restored.replace(rng=jax.random.wrap_key_data(restored.rng, impl=metadata.jax_rng_impl))
    except (TypeError, ValueError) as exc:
        raise ValueError("restored state contains invalid JAX PRNG key data") from exc
    metadata = _validate_metadata(metadata, state=restored, expected_step=restore_step)
    return restored, metadata
