"""Deterministic Stage 2 TD3 training runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import contextlib
import copy
import dataclasses
import errno
import fcntl
import json
import math
import numbers
import operator
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.training.rl_token.stage2 import td3 as rlt_td3
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state

_RUNTIME_COUNT_FIELDS = (
    "batch_size",
    "log_interval",
    "temp_checkpoint_interval",
    "temp_max_to_keep",
    "replay_max_open_shards",
)


def _positive_count(value: object, *, field: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{field} must be an exact positive integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be an exact positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return int(result)


@dataclasses.dataclass(frozen=True)
class Stage2TrainerConfig:
    """Network, algorithm, and operational defaults for Stage 2 training."""

    network: rlt_actor_critic.RLTActorCriticConfig = dataclasses.field(
        default_factory=rlt_actor_critic.RLTActorCriticConfig
    )
    algorithm: rlt_td3.TD3Config = dataclasses.field(default_factory=rlt_td3.TD3Config)
    batch_size: int = 256
    log_interval: int = 100
    temp_checkpoint_interval: int = 1000
    temp_max_to_keep: int = 2
    replay_max_open_shards: int = 32

    def __post_init__(self) -> None:
        self.validate()
        for field in _RUNTIME_COUNT_FIELDS:
            object.__setattr__(
                self,
                field,
                _positive_count(getattr(self, field), field=field),
            )

    def validate(self) -> None:
        self.network.validate()
        self.algorithm.validate()
        for field in _RUNTIME_COUNT_FIELDS:
            _positive_count(getattr(self, field), field=field)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return only settings that change training semantics."""
        return {
            "network": dataclasses.asdict(self.network),
            "algorithm": dataclasses.asdict(self.algorithm),
            "batch_size": self.batch_size,
        }


_ROUND_ID_PATTERN = re.compile(r"round_([0-9]{6})")
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOWERCASE_SHA1 = re.compile(r"[0-9a-f]{40}")
_MAX_SEED = 2**64 - 1
_RENAME_NOREPLACE = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_ROUND_CONTEXT_FIELDS = frozenset(
    {
        "round_id",
        "new_batch_id",
        "admission_path",
        "admission_sha256",
        "new_shard_root",
        "new_shard_manifest_sha256",
        "frozen_checkpoint_root",
        "replay_snapshot_path",
        "replay_snapshot_sha256",
        "feature_identity",
        "frozen_params_sha256",
        "norm_stats_sha256",
        "config_name",
        "asset_id",
        "sampler_num_steps",
        "network_config",
        "algorithm_config",
        "batch_size",
        "seed",
        "code_commit",
        "trainer_identity",
        "parent_round_final",
        "parent_checkpoint_identity",
        "round_start_step",
        "round_critic_updates",
        "target_critic_step",
    }
)


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    """Final state plus invocation-local and round-cumulative update counts."""

    state: rlt_td3_state.RLTTrainState
    replay_rng_state: dict[str, Any]
    critic_updates_completed: int
    actor_updates_completed: int
    round_critic_updates_completed: int
    round_actor_updates_completed: int
    final_metrics: dict[str, float]


class MetricLogError(RuntimeError):
    """Raised when an append-only metric log violates its durable contract."""


_RESERVED_METRIC_FIELDS = frozenset({"critic_step"})


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _nonnegative_count(value: object, *, field: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{field} must be an exact nonnegative integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be an exact nonnegative integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{field} must be nonnegative, got {value!r}")
    return int(result)


def _normalize_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise MetricLogError("metrics must be a mapping")
    overlap = _RESERVED_METRIC_FIELDS.intersection(metrics)
    if overlap:
        raise MetricLogError(f"metrics must not override reserved fields {sorted(overlap)}")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if type(key) is not str or not key:
            raise MetricLogError("metric names must be nonempty strings")
        if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
            raise MetricLogError(f"metric {key!r} must be a finite real number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise MetricLogError(f"metric {key!r} must be finite")
        result[key] = normalized
    return result


def _canonical_metric_record(step: int, metrics: Mapping[str, object]) -> bytes:
    normalized_metrics = _normalize_metrics(metrics)
    try:
        encoded = json.dumps(
            {"critic_step": step, **normalized_metrics},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise MetricLogError("metric record is not valid finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _metric_directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise MetricLogError("metric log durability requires Linux dirfd, O_DIRECTORY, and O_NOFOLLOW")
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _finish_metric_cleanup(
    primary_error: BaseException | None,
    errors: list[tuple[str, BaseException]],
    *,
    context: str,
) -> None:
    if not errors:
        return
    if primary_error is not None:
        for label, error in errors:
            primary_error.add_note(f"{context} cleanup failed during {label}: {error}")
        return
    aggregate = MetricLogError(f"{context} cleanup failed")
    for label, error in errors:
        aggregate.add_note(f"{label}: {error}")
    raise aggregate from errors[0][1]


def _open_durable_metric_parent(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    descriptors: list[int] = []
    try:
        descriptors.append(
            os.open(
                absolute.anchor,
                _metric_directory_flags(),
            )
        )
        for component in absolute.parts[1:]:
            parent_descriptor = descriptors[-1]
            try:
                child_descriptor = os.open(
                    component,
                    _metric_directory_flags(),
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(
                        component,
                        mode=0o755,
                        dir_fd=parent_descriptor,
                    )
                try:
                    child_descriptor = os.open(
                        component,
                        _metric_directory_flags(),
                        dir_fd=parent_descriptor,
                    )
                except OSError as exc:
                    raise MetricLogError(f"created metric directory cannot be opened safely: {absolute}") from exc
            except OSError as exc:
                raise MetricLogError(
                    f"metric directory components must be real nonsymlink directories: {absolute}"
                ) from exc
            descriptors.append(child_descriptor)
            os.fsync(parent_descriptor)
    except BaseException:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as exc:
                errors.append(("closing metric directory descriptor", exc))
        _finish_metric_cleanup(
            primary_error,
            errors,
            context=f"metric directory {absolute}",
        )
        raise

    final_descriptor = descriptors.pop()
    errors = []
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append(("closing metric ancestor descriptor", exc))
    if errors:
        try:
            os.close(final_descriptor)
        except BaseException as exc:
            errors.append(("closing metric parent descriptor", exc))
        _finish_metric_cleanup(
            None,
            errors,
            context=f"metric directory {absolute}",
        )
    return final_descriptor


class JsonlMetricSink:
    """Crash-recoverable, exclusively locked, append-only canonical JSONL."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._descriptor: int | None = None
        self._locked = False
        self._records: dict[int, bytes] = {}
        self._last_step: int | None = None
        parent_descriptor = _open_durable_metric_parent(self.path.parent)
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            try:
                try:
                    descriptor = os.open(
                        self.path.name,
                        flags | os.O_CREAT | os.O_EXCL,
                        0o644,
                        dir_fd=parent_descriptor,
                    )
                except FileExistsError:
                    descriptor = os.open(
                        self.path.name,
                        flags,
                        dir_fd=parent_descriptor,
                    )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise MetricLogError(f"metric log symlink is forbidden: {self.path}") from exc
                raise MetricLogError(f"metric log must be a regular file: {self.path}") from exc
            self._descriptor = descriptor
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise MetricLogError(f"metric log must be a regular file: {self.path}")
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise MetricLogError(f"metric log exclusive lock is unavailable: {self.path}") from exc
            self._locked = True
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise MetricLogError(f"metric log must be a regular file: {self.path}")
            self._recover_and_validate_history(metadata.st_size)
        except BaseException:
            primary_error = sys.exception()
            self._release_descriptor(primary_error)
            errors: list[tuple[str, BaseException]] = []
            try:
                os.close(parent_descriptor)
            except BaseException as exc:
                errors.append(("closing pinned metric parent", exc))
            _finish_metric_cleanup(
                primary_error,
                errors,
                context=f"metric log constructor {self.path}",
            )
            raise
        errors = []
        try:
            os.close(parent_descriptor)
        except BaseException as exc:
            errors.append(("closing pinned metric parent", exc))
        if errors:
            parent_close_error = MetricLogError(f"metric log constructor cleanup failed: {self.path}")
            for label, error in errors:
                parent_close_error.add_note(f"{label}: {error}")
            self._release_descriptor(parent_close_error)
            raise parent_close_error from errors[0][1]

    def _recover_and_validate_history(self, size: int) -> None:
        descriptor = self._require_open()
        try:
            payload = os.pread(descriptor, size, 0)
        except OSError as exc:
            raise MetricLogError(f"failed to read complete metric history: {self.path}") from exc
        if len(payload) != size:
            raise MetricLogError(f"failed to read complete metric history: {self.path}")
        complete_size = len(payload)
        if payload and not payload.endswith(b"\n"):
            final_newline = payload.rfind(b"\n")
            complete_size = 0 if final_newline < 0 else final_newline + 1
        complete = payload[:complete_size]
        previous_step: int | None = None
        for line_number, line in enumerate(complete.splitlines(keepends=True), start=1):
            step, canonical = self._validate_complete_line(line, line_number=line_number)
            if previous_step is not None and step <= previous_step:
                raise MetricLogError(f"complete metric history must be strictly monotonic at line {line_number}")
            previous_step = step
            self._records[step] = canonical
        self._last_step = previous_step
        if complete_size != len(payload):
            try:
                os.ftruncate(descriptor, complete_size)
                os.fsync(descriptor)
            except OSError as exc:
                raise MetricLogError(f"failed to truncate incomplete metric crash tail: {self.path}") from exc

    def _validate_complete_line(self, line: bytes, *, line_number: int) -> tuple[int, bytes]:
        if not line.endswith(b"\n") or line == b"\n":
            raise MetricLogError(f"invalid complete metric line {line_number}")
        try:
            record = json.loads(
                line[:-1].decode("utf-8"),
                object_pairs_hook=_duplicate_rejecting_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise MetricLogError(f"invalid complete metric line {line_number}") from exc
        if type(record) is not dict or "critic_step" not in record:
            raise MetricLogError(f"invalid complete metric line {line_number}")
        try:
            step = _nonnegative_count(record["critic_step"], field="critic_step")
        except ValueError as exc:
            raise MetricLogError(f"invalid complete metric line {line_number}") from exc
        canonical = _canonical_metric_record(
            step,
            {key: value for key, value in record.items() if key != "critic_step"},
        )
        if canonical != line:
            raise MetricLogError(f"invalid noncanonical complete metric line {line_number}")
        return step, canonical

    def _require_open(self) -> int:
        if self._descriptor is None:
            raise MetricLogError("metric sink is closed")
        return self._descriptor

    def _release_descriptor(
        self,
        primary_error: BaseException | None,
    ) -> None:
        descriptor = self._descriptor
        locked = self._locked
        self._descriptor = None
        self._locked = False
        if descriptor is None:
            return
        errors: list[tuple[str, BaseException]] = []
        if locked:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException as exc:
                errors.append(("unlocking metric log", exc))
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append(("closing metric log", exc))
        _finish_metric_cleanup(
            primary_error,
            errors,
            context=f"metric log {self.path}",
        )

    def _poison(self, error: BaseException) -> None:
        self._release_descriptor(error)

    def __call__(self, step: int, metrics: dict[str, float]) -> None:
        try:
            normalized_step = _nonnegative_count(step, field="critic_step")
        except ValueError as exc:
            raise MetricLogError(str(exc)) from exc
        payload = _canonical_metric_record(normalized_step, metrics)
        previous = self._records.get(normalized_step)
        if previous is not None:
            if previous == payload:
                return
            raise MetricLogError(f"same critic_step {normalized_step} has different metric content")
        if self._last_step is not None and normalized_step < self._last_step:
            raise MetricLogError(f"metric critic_step must be monotonic: {normalized_step} < {self._last_step}")
        descriptor = self._require_open()
        try:
            written = os.write(descriptor, payload)
        except OSError as exc:
            error = MetricLogError(f"failed to append metric critic_step {normalized_step}")
            self._poison(error)
            raise error from exc
        if written != len(payload):
            error = MetricLogError(f"short append write for metric critic_step {normalized_step}")
            try:
                os.fsync(descriptor)
            except OSError as exc:
                error.add_note(f"failed to fsync short metric crash tail: {exc}")
            self._poison(error)
            raise error
        try:
            os.fsync(descriptor)
        except OSError as exc:
            error = MetricLogError(f"failed to append metric critic_step {normalized_step}")
            self._poison(error)
            raise error from exc
        self._records[normalized_step] = payload
        self._last_step = normalized_step

    def close(self) -> None:
        self._release_descriptor(sys.exception())

    def __enter__(self) -> JsonlMetricSink:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        self._release_descriptor(exc_value)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._release_descriptor(None)


MetricSink = Callable[[int, dict[str, float]], None]
CheckpointSink = Callable[
    [rlt_td3_state.RLTTrainState, dict[str, Any]],
    None,
]

_INT32_MAX = int(np.iinfo(np.int32).max)
_REPLAY_BATCH_FIELDS = (
    "z_rl",
    "next_z_rl",
    "state_norm",
    "next_state_norm",
    "vla_reference",
    "next_vla_reference",
    "executed_action",
    "bc_anchor",
    "reward",
    "terminal",
    "source_global_index",
)


def _state_counter(
    state: rlt_td3_state.RLTTrainState,
    field: str,
    *,
    error_type: type[Exception] = ValueError,
) -> int:
    value = getattr(state, field, None)
    if not isinstance(value, jax.Array):
        raise error_type(f"{field} must be a device scalar int32 JAX array, got {type(value).__name__}")
    try:
        host = np.asarray(jax.device_get(value))
    except Exception as exc:
        raise error_type(f"{field} must be a device scalar int32") from exc
    if host.shape != ():
        raise error_type(f"{field} must be a device scalar, got shape {host.shape}")
    if host.dtype != np.dtype(np.int32):
        raise error_type(f"{field} must have dtype int32, got {host.dtype}")
    result = int(host)
    if result < 0:
        raise error_type(f"{field} must be nonnegative, got {result}")
    return result


def _validate_jax_rng(value: object, *, context: str) -> None:
    try:
        data = np.asarray(jax.device_get(jax.random.key_data(value)))
        jax.random.key_impl(value)
    except Exception as exc:
        raise ValueError(f"{context} must be a valid JAX random key") from exc
    if data.shape != (2,) or data.dtype != np.dtype(np.uint32):
        raise ValueError(f"{context} must have two uint32 words, got shape {data.shape} and dtype {data.dtype}")


def _validate_fp32_finite_tree(
    tree: object,
    *,
    context: str,
    require_all_float: bool,
) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{context} must contain array leaves")
    floating_leaves = 0
    for index, leaf in enumerate(leaves):
        if not hasattr(leaf, "dtype"):
            if isinstance(leaf, numbers.Real) and not isinstance(leaf, bool | np.bool_):
                raise ValueError(f"{context} floating leaf {index} must be a float32 array, got {type(leaf).__name__}")
            if require_all_float:
                raise ValueError(f"{context} leaf {index} must be a floating array")
            continue
        dtype = np.dtype(leaf.dtype)
        if jax.dtypes.issubdtype(leaf.dtype, jnp.inexact):
            floating_leaves += 1
            if dtype != np.dtype(np.float32):
                raise ValueError(f"{context} floating leaf {index} must have dtype float32, got {dtype}")
            try:
                finite = bool(np.all(np.isfinite(np.asarray(jax.device_get(leaf)))))
            except Exception as exc:
                raise ValueError(f"{context} floating leaf {index} could not be validated") from exc
            if not finite:
                raise FloatingPointError(f"{context} floating leaf {index} is non-finite")
        elif require_all_float:
            raise ValueError(f"{context} leaf {index} must have dtype float32, got {dtype}")
    if floating_leaves == 0:
        raise ValueError(f"{context} must contain floating leaves")


def _validate_q_tree(tree: object, *, context: str) -> None:
    if not isinstance(tree, Mapping) or set(tree) != {"q1", "q2"}:
        raise ValueError(f"{context} must contain independent q1 and q2 parameter trees")
    _validate_fp32_finite_tree(
        tree["q1"],
        context=f"{context}.q1",
        require_all_float=True,
    )
    _validate_fp32_finite_tree(
        tree["q2"],
        context=f"{context}.q2",
        require_all_float=True,
    )


def _optimizer_count(
    opt_state: object,
    *,
    field: str,
    error_type: type[Exception] = ValueError,
) -> int:
    if type(opt_state) is not tuple or len(opt_state) != 2 or type(opt_state[1]) is not tuple or len(opt_state[1]) != 2:
        raise error_type(f"{field} must use the expected clipped-Adam optimizer state structure")
    adam_state = opt_state[1][0]
    if getattr(adam_state, "_fields", None) != ("count", "mu", "nu"):
        raise error_type(f"{field} must contain exactly one clipped-Adam count state")
    if not isinstance(adam_state.count, jax.Array):
        raise error_type(
            f"{field} count must be a device scalar int32 JAX array, got {type(adam_state.count).__name__}"
        )
    try:
        count = np.asarray(jax.device_get(adam_state.count))
    except Exception as exc:
        raise error_type(f"{field} count must be a device scalar int32") from exc
    if count.shape != ():
        raise error_type(f"{field} count must be a device scalar, got shape {count.shape}")
    if count.dtype != np.dtype(np.int32):
        raise error_type(f"{field} count must have dtype int32, got {count.dtype}")
    result = int(count)
    if result < 0:
        raise error_type(f"{field} count must be nonnegative, got {result}")
    return result


def _validate_optimizer_state(
    opt_state: object,
    *,
    parameter_tree: object,
    field: str,
) -> int:
    count = _optimizer_count(opt_state, field=field)
    if jax.tree_util.tree_leaves(opt_state[0]) or jax.tree_util.tree_leaves(opt_state[1][1]):
        raise ValueError(f"{field} clipped-Adam wrapper states must be stateless")
    adam_state = opt_state[1][0]
    parameter_structure = jax.tree_util.tree_structure(parameter_tree)
    parameter_leaves = jax.tree_util.tree_leaves(parameter_tree)
    for moment_name in ("mu", "nu"):
        moment = getattr(adam_state, moment_name)
        _validate_fp32_finite_tree(
            moment,
            context=f"{field}.{moment_name}",
            require_all_float=True,
        )
        if jax.tree_util.tree_structure(moment) != parameter_structure:
            raise ValueError(f"{field}.{moment_name} structure must match its parameter tree")
        for index, (moment_leaf, parameter_leaf) in enumerate(
            zip(
                jax.tree_util.tree_leaves(moment),
                parameter_leaves,
                strict=True,
            )
        ):
            if tuple(np.shape(moment_leaf)) != tuple(np.shape(parameter_leaf)):
                raise ValueError(f"{field}.{moment_name} leaf {index} shape must match its parameter leaf")
    return count


def validate_training_state(state: rlt_td3_state.RLTTrainState) -> None:
    """Validate the complete persisted Stage 2 state before use or publication."""
    if not isinstance(state, rlt_td3_state.RLTTrainState):
        raise ValueError("state must be an RLTTrainState")
    _state_counter(state, "critic_step")
    _state_counter(state, "round_critic_step")
    _validate_jax_rng(state.rng, context="state.rng")
    _validate_fp32_finite_tree(
        state.actor_params,
        context="online actor parameters",
        require_all_float=True,
    )
    _validate_q_tree(state.q_params, context="online critic parameters")
    _validate_fp32_finite_tree(
        state.target_actor_params,
        context="target actor parameters",
        require_all_float=True,
    )
    _validate_q_tree(state.target_q_params, context="target critic parameters")
    actor_optimizer_count = _validate_optimizer_state(
        state.actor_opt_state,
        parameter_tree=state.actor_params,
        field="actor_opt_state",
    )
    critic_optimizer_count = _validate_optimizer_state(
        state.critic_opt_state,
        parameter_tree=state.q_params,
        field="critic_opt_state",
    )
    critic_step = _state_counter(state, "critic_step")
    if critic_optimizer_count != critic_step:
        raise ValueError(f"critic optimizer count must equal critic_step: {critic_optimizer_count} != {critic_step}")
    if actor_optimizer_count > critic_optimizer_count:
        raise ValueError(
            "actor optimizer count must not exceed critic optimizer count: "
            f"{actor_optimizer_count} > {critic_optimizer_count}"
        )


def _replay_root_context(replay_buffer: object) -> str:
    snapshot = getattr(replay_buffer, "snapshot", None)
    shards = getattr(snapshot, "shards", ())
    roots: list[str] = []
    try:
        iterator = iter(shards)
    except TypeError:
        iterator = iter(())
    for shard in iterator:
        if isinstance(shard, Mapping):
            batch_id = shard.get("batch_id", "<unknown>")
            root = shard.get("root", "<unknown>")
        else:
            batch_id = getattr(shard, "batch_id", "<unknown>")
            root = getattr(shard, "root", "<unknown>")
        roots.append(f"{batch_id}:{root}")
    return "shard/root " + (", ".join(roots) if roots else "<unknown>")


def _replay_field_error(
    replay_buffer: object,
    field: str,
    detail: str,
    *,
    cause: BaseException | None = None,
) -> ValueError:
    error = ValueError(f"replay {_replay_root_context(replay_buffer)} field {field}: {detail}")
    if cause is not None:
        error.__cause__ = cause
    return error


def _validate_sampled_indices(
    indices: object,
    *,
    replay_buffer: object,
    batch_size: int,
    total_transitions: int,
) -> np.ndarray:
    if not isinstance(indices, np.ndarray):
        raise _replay_field_error(
            replay_buffer,
            "sample_indices",
            f"must return a numpy array, got {type(indices).__name__}",
        )
    if indices.shape != (batch_size,):
        raise _replay_field_error(
            replay_buffer,
            "sample_indices",
            f"expected shape {(batch_size,)}, got {indices.shape}",
        )
    if indices.dtype != np.dtype(np.int64):
        raise _replay_field_error(
            replay_buffer,
            "sample_indices",
            f"expected dtype int64, got {indices.dtype}",
        )
    if np.any(indices < 0) or np.any(indices >= total_transitions):
        raise _replay_field_error(
            replay_buffer,
            "sample_indices",
            f"indices must be in [0, {total_transitions})",
        )
    return indices


def _validate_host_replay_batch(
    batch: object,
    *,
    replay_buffer: object,
    sampled_indices: np.ndarray,
    batch_size: int,
    network: rlt_actor_critic.RLTActorCriticConfig,
) -> object:
    try:
        field_names = tuple(field.name for field in dataclasses.fields(batch))
    except (TypeError, AttributeError) as exc:
        raise _replay_field_error(
            replay_buffer,
            "batch",
            f"gather must return a ReplayBatch dataclass, got {type(batch).__name__}",
            cause=exc,
        ) from exc
    if field_names != _REPLAY_BATCH_FIELDS:
        raise _replay_field_error(
            replay_buffer,
            "batch",
            f"ReplayBatch fields must be exactly {_REPLAY_BATCH_FIELDS}, got {field_names}",
        )
    expected_shapes = {
        "z_rl": (batch_size, network.z_dim),
        "next_z_rl": (batch_size, network.z_dim),
        "state_norm": (batch_size, network.state_dim),
        "next_state_norm": (batch_size, network.state_dim),
        "vla_reference": (
            batch_size,
            network.action_horizon,
            network.action_dim,
        ),
        "next_vla_reference": (
            batch_size,
            network.action_horizon,
            network.action_dim,
        ),
        "executed_action": (
            batch_size,
            network.action_horizon,
            network.action_dim,
        ),
        "bc_anchor": (
            batch_size,
            network.action_horizon,
            network.action_dim,
        ),
        "reward": (batch_size, 1),
        "terminal": (batch_size, 1),
        "source_global_index": (batch_size,),
    }
    bf16_fields = {"z_rl", "next_z_rl"}
    fp32_fields = {
        "state_norm",
        "next_state_norm",
        "vla_reference",
        "next_vla_reference",
        "executed_action",
        "bc_anchor",
        "reward",
    }
    for field, expected_shape in expected_shapes.items():
        value = getattr(batch, field)
        if not isinstance(value, np.ndarray):
            raise _replay_field_error(
                replay_buffer,
                field,
                f"must be a numpy array, got {type(value).__name__}",
            )
        if value.shape != expected_shape:
            raise _replay_field_error(
                replay_buffer,
                field,
                f"expected shape {expected_shape}, got {value.shape}",
            )
        if field in bf16_fields and value.dtype != np.dtype(jnp.bfloat16):
            raise _replay_field_error(
                replay_buffer,
                field,
                f"expected dtype bfloat16, got {value.dtype}",
            )
        if field in fp32_fields and value.dtype != np.dtype(np.float32):
            raise _replay_field_error(
                replay_buffer,
                field,
                f"expected dtype float32, got {value.dtype}",
            )
        if field == "terminal" and value.dtype != np.dtype(np.bool_):
            raise _replay_field_error(
                replay_buffer,
                field,
                f"expected host dtype bool, got {value.dtype}",
            )
        if field == "source_global_index" and value.dtype != np.dtype(np.int64):
            raise _replay_field_error(
                replay_buffer,
                field,
                f"expected dtype int64, got {value.dtype}",
            )
        if field in bf16_fields | fp32_fields:
            try:
                finite = bool(np.all(np.isfinite(value.astype(np.float32, copy=False))))
            except Exception as exc:
                raise _replay_field_error(
                    replay_buffer,
                    field,
                    "could not validate finiteness",
                    cause=exc,
                ) from exc
            if not finite:
                raise _replay_field_error(
                    replay_buffer,
                    field,
                    "contains NaN or Inf",
                )
    if not np.array_equal(batch.source_global_index, sampled_indices):
        raise _replay_field_error(
            replay_buffer,
            "source_global_index",
            "does not exactly match sampled global indices",
        )
    return batch


def _as_jax_transition_batch(batch: object) -> rlt_td3.RLTTransitionBatch:
    return rlt_td3.RLTTransitionBatch(
        z_rl=jnp.asarray(batch.z_rl),
        next_z_rl=jnp.asarray(batch.next_z_rl),
        state_norm=jnp.asarray(batch.state_norm),
        next_state_norm=jnp.asarray(batch.next_state_norm),
        vla_reference=jnp.asarray(batch.vla_reference),
        next_vla_reference=jnp.asarray(batch.next_vla_reference),
        executed_action=jnp.asarray(batch.executed_action),
        bc_anchor=jnp.asarray(batch.bc_anchor),
        reward=jnp.asarray(batch.reward),
        terminal=jnp.asarray(batch.terminal, dtype=jnp.float32),
    )


def _host_metrics(metrics: object) -> dict[str, float]:
    try:
        host = jax.device_get(metrics)
    except Exception as exc:
        raise RuntimeError("training metrics could not be transferred to the host") from exc
    if not isinstance(host, Mapping):
        raise RuntimeError("training metrics must be a mapping")
    required = {"critic/loss", "actor/updated"}
    missing = required.difference(host)
    if missing:
        raise RuntimeError(f"training metrics are missing required fields {sorted(missing)}")
    result: dict[str, float] = {}
    for name, value in host.items():
        if type(name) is not str or not name:
            raise RuntimeError("training metric names must be nonempty strings")
        array = np.asarray(value)
        if array.shape != ():
            raise RuntimeError(f"training metric {name!r} must be scalar")
        if array.dtype != np.dtype(np.float32):
            raise RuntimeError(f"training metric {name!r} must have dtype float32, got {array.dtype}")
        normalized = float(array)
        if not math.isfinite(normalized):
            raise FloatingPointError(f"training metric {name!r} is non-finite")
        result[name] = normalized
    if result["actor/updated"] not in {0.0, 1.0}:
        raise RuntimeError("training metric 'actor/updated' must be exactly 0.0 or 1.0")
    return result


def _validate_runtime_inputs(
    *,
    state: rlt_td3_state.RLTTrainState,
    actor: rlt_actor_critic.RLTActor,
    critic: rlt_actor_critic.RLTCritic,
    algorithm: rlt_td3.TD3Config,
    replay_buffer: object,
    replay_rng: np.random.Generator,
    round_start_step: object,
    round_critic_updates: object,
    batch_size: object,
    log_interval: object,
    temp_checkpoint_interval: object,
) -> tuple[int, int, int, int, int, int, int]:
    start = _nonnegative_count(round_start_step, field="round_start_step")
    budget = _positive_count(round_critic_updates, field="round_critic_updates")
    normalized_batch_size = _positive_count(batch_size, field="batch_size")
    normalized_log_interval = _positive_count(log_interval, field="log_interval")
    normalized_checkpoint_interval = _positive_count(
        temp_checkpoint_interval,
        field="temp_checkpoint_interval",
    )
    if type(replay_rng) is not np.random.Generator or type(replay_rng.bit_generator) is not np.random.PCG64:
        raise ValueError("replay_rng must be exactly numpy Generator(PCG64)")
    if not isinstance(actor, rlt_actor_critic.RLTActor) or not isinstance(
        critic,
        rlt_actor_critic.RLTCritic,
    ):
        raise ValueError("actor and critic must be RLT actor/critic modules")
    actor.config.validate()
    critic.config.validate()
    if actor.config != critic.config:
        raise ValueError(
            f"actor and critic network config must match exactly: actor={actor.config!r}, critic={critic.config!r}"
        )
    if not isinstance(algorithm, rlt_td3.TD3Config):
        raise ValueError("algorithm must be a TD3Config")
    algorithm.validate()
    try:
        total_transitions = _positive_count(
            replay_buffer.total_transitions,
            field="replay total_transitions",
        )
    except AttributeError as exc:
        raise ValueError("replay must expose total_transitions") from exc
    validate_training_state(state)
    global_step = _state_counter(state, "critic_step")
    round_step = _state_counter(state, "round_critic_step")
    if global_step != start + round_step:
        raise ValueError(
            f"round_start_step plus round_critic_step must equal critic_step: {start} + {round_step} != {global_step}"
        )
    if round_step > budget:
        raise ValueError(f"round_critic_step {round_step} is beyond round update budget {budget}")
    if start > _INT32_MAX or budget > _INT32_MAX - start:
        raise OverflowError("round_start_step plus round_critic_updates would overflow int32 counters")
    return (
        start,
        budget,
        normalized_batch_size,
        normalized_log_interval,
        normalized_checkpoint_interval,
        total_transitions,
        round_step,
    )


def _clone_pcg64_generator(rng: np.random.Generator) -> np.random.Generator:
    candidate = np.random.Generator(np.random.PCG64())
    candidate.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return candidate


def run_updates(
    *,
    state: rlt_td3_state.RLTTrainState,
    actor: rlt_actor_critic.RLTActor,
    critic: rlt_actor_critic.RLTCritic,
    actor_tx: object,
    critic_tx: object,
    algorithm: rlt_td3.TD3Config,
    replay_buffer: object,
    replay_rng: np.random.Generator,
    round_start_step: int,
    round_critic_updates: int,
    batch_size: int,
    log_interval: int,
    temp_checkpoint_interval: int,
    metric_sink: MetricSink,
    checkpoint_sink: CheckpointSink,
) -> TrainingResult:
    """Run the remaining one-update-per-loop portion of a prepared data round."""
    if not callable(metric_sink):
        raise ValueError("metric_sink must be callable")
    if not callable(checkpoint_sink):
        raise ValueError("checkpoint_sink must be callable")
    (
        normalized_round_start,
        normalized_budget,
        normalized_batch_size,
        normalized_log_interval,
        normalized_checkpoint_interval,
        total_transitions,
        round_step,
    ) = _validate_runtime_inputs(
        state=state,
        actor=actor,
        critic=critic,
        algorithm=algorithm,
        replay_buffer=replay_buffer,
        replay_rng=replay_rng,
        round_start_step=round_start_step,
        round_critic_updates=round_critic_updates,
        batch_size=batch_size,
        log_interval=log_interval,
        temp_checkpoint_interval=temp_checkpoint_interval,
    )
    initial_round_step = round_step
    global_step = _state_counter(state, "critic_step")
    final_metrics: dict[str, float] = {}
    invocation_actor_updates = 0
    step_fn: Callable[..., Any] | None = None

    while round_step < normalized_budget:
        candidate_rng = _clone_pcg64_generator(replay_rng)
        try:
            indices = replay_buffer.sample_indices(
                candidate_rng,
                normalized_batch_size,
            )
        except Exception as exc:
            raise _replay_field_error(
                replay_buffer,
                "sample_indices",
                "sampling failed",
                cause=exc,
            ) from exc
        sampled_indices = _validate_sampled_indices(
            indices,
            replay_buffer=replay_buffer,
            batch_size=normalized_batch_size,
            total_transitions=total_transitions,
        )
        sampled_index_witness = sampled_indices.copy()
        sampled_index_witness.flags.writeable = False
        try:
            host_batch = replay_buffer.gather(sampled_indices.copy())
        except Exception as exc:
            raise _replay_field_error(
                replay_buffer,
                "gather",
                "gather failed",
                cause=exc,
            ) from exc
        host_batch = _validate_host_replay_batch(
            host_batch,
            replay_buffer=replay_buffer,
            sampled_indices=sampled_index_witness,
            batch_size=normalized_batch_size,
            network=actor.config,
        )
        try:
            device_batch = _as_jax_transition_batch(host_batch)
            rlt_td3.validate_transition_batch(device_batch, actor.config)
        except Exception as exc:
            raise _replay_field_error(
                replay_buffer,
                "device_batch",
                "conversion or device validation failed",
                cause=exc,
            ) from exc
        if step_fn is None:
            step_fn = jax.jit(
                lambda value, transition_batch: rlt_td3_state.train_step(
                    value,
                    transition_batch,
                    actor,
                    critic,
                    algorithm,
                    actor_tx,
                    critic_tx,
                )
            )

        previous_global = global_step
        previous_round = round_step
        updated_state, metrics = step_fn(state, device_batch)
        if not isinstance(updated_state, rlt_td3_state.RLTTrainState):
            raise RuntimeError("actual state counter validation requires an RLTTrainState")
        try:
            actual_global = _state_counter(
                updated_state,
                "critic_step",
                error_type=RuntimeError,
            )
            actual_round = _state_counter(
                updated_state,
                "round_critic_step",
                error_type=RuntimeError,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"actual state counter is invalid: {exc}") from exc
        if (
            actual_global != previous_global + 1
            or actual_round != previous_round + 1
            or actual_global != normalized_round_start + actual_round
        ):
            raise RuntimeError(
                "actual state counter increment is inconsistent: "
                f"global {previous_global}->{actual_global}, "
                f"round {previous_round}->{actual_round}"
            )
        try:
            _validate_jax_rng(updated_state.rng, context="post-update state.rng")
        except ValueError as exc:
            raise RuntimeError(f"invalid post-update RNG at global={actual_global}, round={actual_round}") from exc
        try:
            host_metrics = _host_metrics(metrics)
        except (FloatingPointError, RuntimeError) as exc:
            error_type = FloatingPointError if isinstance(exc, FloatingPointError) else RuntimeError
            raise error_type(f"{exc} at global={actual_global}, round={actual_round}") from exc
        expected_actor_updated = float(actual_round % int(algorithm.policy_delay) == 0)
        if host_metrics["actor/updated"] != expected_actor_updated:
            raise RuntimeError(
                "training metric 'actor/updated' violated the post-increment "
                f"round schedule at global={actual_global}, round={actual_round}"
            )
        previous_actor_optimizer_count = _optimizer_count(
            state.actor_opt_state,
            field="actor_opt_state",
            error_type=RuntimeError,
        )
        previous_critic_optimizer_count = _optimizer_count(
            state.critic_opt_state,
            field="critic_opt_state",
            error_type=RuntimeError,
        )
        actual_actor_optimizer_count = _optimizer_count(
            updated_state.actor_opt_state,
            field="actor_opt_state",
            error_type=RuntimeError,
        )
        actual_critic_optimizer_count = _optimizer_count(
            updated_state.critic_opt_state,
            field="critic_opt_state",
            error_type=RuntimeError,
        )
        if actual_critic_optimizer_count != previous_critic_optimizer_count + 1:
            raise RuntimeError(
                "actual critic optimizer count did not increment exactly once: "
                f"{previous_critic_optimizer_count}->{actual_critic_optimizer_count}"
            )
        if actual_critic_optimizer_count != actual_global:
            raise RuntimeError(
                "actual critic optimizer count does not equal actual state counter: "
                f"{actual_critic_optimizer_count} != {actual_global}"
            )
        expected_actor_optimizer_count = previous_actor_optimizer_count + int(expected_actor_updated)
        if actual_actor_optimizer_count != expected_actor_optimizer_count:
            raise RuntimeError(
                "actual actor optimizer count violated the delayed update schedule: "
                f"expected {expected_actor_optimizer_count}, "
                f"got {actual_actor_optimizer_count}"
            )

        is_final = actual_round == normalized_budget
        is_temporary_checkpoint = actual_global % normalized_checkpoint_interval == 0 and not is_final
        if is_final or is_temporary_checkpoint:
            try:
                validate_training_state(updated_state)
            except (ValueError, FloatingPointError) as exc:
                error_type = FloatingPointError if isinstance(exc, FloatingPointError) else ValueError
                raise error_type(
                    f"invalid training state at global={actual_global}, round={actual_round}: {exc}"
                ) from exc

        replay_rng.bit_generator.state = copy.deepcopy(candidate_rng.bit_generator.state)
        state = updated_state
        global_step = actual_global
        round_step = actual_round
        final_metrics = host_metrics
        invocation_actor_updates += int(host_metrics["actor/updated"])

        if global_step % normalized_log_interval == 0 or is_final:
            metric_sink(
                global_step,
                {
                    "round_critic_step": float(round_step),
                    "replay/total_transitions": float(total_transitions),
                    **host_metrics,
                },
            )
        if is_temporary_checkpoint:
            checkpoint_sink(
                copy.deepcopy(state),
                copy.deepcopy(replay_rng.bit_generator.state),
            )

    try:
        validate_training_state(state)
    except (ValueError, FloatingPointError) as exc:
        error_type = FloatingPointError if isinstance(exc, FloatingPointError) else ValueError
        raise error_type(f"invalid final training state at global={global_step}, round={round_step}: {exc}") from exc
    expected_actor_updates = round_step // int(algorithm.policy_delay) - initial_round_step // int(
        algorithm.policy_delay
    )
    if invocation_actor_updates != expected_actor_updates:
        raise RuntimeError("actor update count violated the post-increment policy delay schedule")
    return TrainingResult(
        state=state,
        replay_rng_state=copy.deepcopy(replay_rng.bit_generator.state),
        critic_updates_completed=round_step - initial_round_step,
        actor_updates_completed=invocation_actor_updates,
        round_critic_updates_completed=round_step,
        round_actor_updates_completed=round_step // int(algorithm.policy_delay),
        final_metrics=dict(final_metrics),
    )
