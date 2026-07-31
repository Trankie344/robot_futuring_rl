# ruff: noqa: SLF001

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import errno
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any

import ml_dtypes
import numpy as np

from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import transitions

_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESERVED_MANIFEST_FIELDS = {
    "schema_version",
    "feature_rows",
    "transition_rows",
    "files",
}
_FILE_RECORD_FIELDS = {"path", "size", "sha256", "shape", "dtype"}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class CacheError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class FeatureTable:
    episode_index: np.ndarray
    frame_index: np.ndarray
    z_rl: np.ndarray
    state_norm: np.ndarray
    vla_reference: np.ndarray


@dataclasses.dataclass(frozen=True)
class TransitionTable:
    episode_index: np.ndarray
    start_frame_index: np.ndarray
    current_feature_row: np.ndarray
    next_feature_row: np.ndarray
    executed_action: np.ndarray
    bc_anchor: np.ndarray
    reward: np.ndarray
    terminal: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class ShardFileVerification:
    path: str
    shape: tuple[int, ...]
    dtype: str
    size: int
    sha256: str
    _metadata: _StatWitness


@dataclasses.dataclass(frozen=True, slots=True)
class ShardArrayContract:
    path: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def trailing_shape(self) -> tuple[int, ...]:
        return self.shape[1:]


@dataclasses.dataclass(frozen=True, slots=True)
class _StatWitness:
    device: int
    inode: int
    mode: int
    link_count: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int


_VERIFICATION_SEAL = object()


class ShardVerification:
    """An immutable, descriptor-free proof produced by :func:`authenticate_shard`.

    This is a capability rather than a user-authored manifest wrapper.  The
    private seal prevents callers from manufacturing a verification through
    the supported constructor surface, while ``__slots__`` closes its schema.
    """

    __slots__ = (
        "_array_contract",
        "_feature_rows",
        "_files",
        "_group_metadata",
        "_manifest_bytes",
        "_manifest_metadata",
        "_manifest_sha256",
        "_root",
        "_root_metadata",
        "_seal",
        "_transition_rows",
    )

    def __init__(
        self,
        *,
        _seal: object,
        root: Path,
        manifest_bytes: bytes,
        manifest_sha256: str,
        root_metadata: _StatWitness,
        group_metadata: tuple[tuple[str, _StatWitness], ...],
        manifest_metadata: _StatWitness,
        files: tuple[ShardFileVerification, ...],
        array_contract: tuple[ShardArrayContract, ...],
        feature_rows: int,
        transition_rows: int,
    ) -> None:
        if _seal is not _VERIFICATION_SEAL:
            raise TypeError("ShardVerification values can only be created by authenticate_shard")
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_manifest_bytes", manifest_bytes)
        object.__setattr__(self, "_manifest_sha256", manifest_sha256)
        object.__setattr__(self, "_root_metadata", root_metadata)
        object.__setattr__(self, "_group_metadata", group_metadata)
        object.__setattr__(self, "_manifest_metadata", manifest_metadata)
        object.__setattr__(self, "_files", files)
        object.__setattr__(self, "_array_contract", array_contract)
        object.__setattr__(self, "_feature_rows", feature_rows)
        object.__setattr__(self, "_transition_rows", transition_rows)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ShardVerification is immutable")

    def __reduce__(self) -> object:
        raise TypeError("ShardVerification cannot be serialized or reconstructed")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> dict[str, Any]:
        # A fresh value prevents mutation from changing the authenticated
        # capability while retaining the historical dict-like API.
        return json.loads(self._manifest_bytes.decode("utf-8"))

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def feature_rows(self) -> int:
        return self._feature_rows

    @property
    def transition_rows(self) -> int:
        return self._transition_rows

    @property
    def files(self) -> tuple[ShardFileVerification, ...]:
        return self._files

    @property
    def array_contract(self) -> tuple[ShardArrayContract, ...]:
        return self._array_contract

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardVerification):
            return NotImplemented
        _require_verification(self)
        _require_verification(other)
        return _verification_identity(self) == _verification_identity(other)

    def __hash__(self) -> int:
        _require_verification(self)
        return hash(_verification_identity(self))


@dataclasses.dataclass(eq=False)
class OpenShard:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    features: FeatureTable
    transitions: TransitionTable
    verification: ShardVerification
    _closed: bool = dataclasses.field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._close(sys.exception())

    def _close(self, primary: BaseException | None) -> None:
        if self._closed:
            return
        errors: list[tuple[str, BaseException]] = []
        seen: set[int] = set()
        memory_maps: list[Any] = []
        for table_name, table in (("features", self.features), ("transitions", self.transitions)):
            for field in dataclasses.fields(table):
                value = getattr(table, field.name)
                memory_map = getattr(value, "_mmap", None)
                if memory_map is None or id(memory_map) in seen:
                    continue
                seen.add(id(memory_map))
                memory_maps.append(memory_map)
                if memory_map.closed:
                    continue
                try:
                    _close_memory_map(memory_map)
                except BaseException as exc:
                    errors.append((f"{table_name}.{field.name}", exc))
        self._closed = all(memory_map.closed for memory_map in memory_maps)
        _finish_cleanup(primary, errors, action="close cache shard")

    def __enter__(self) -> OpenShard:
        if self._closed:
            raise CacheError("cache shard is closed")
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> bool:
        self._close(_exception)
        return False


def _close_memory_map(memory_map: Any) -> None:
    """Close hook kept separate so cleanup failures are deterministically testable."""

    memory_map.close()


def _finish_cleanup(
    primary: BaseException | None,
    errors: list[tuple[str, BaseException]],
    *,
    action: str,
) -> None:
    if not errors:
        return
    notes = [f"{action} cleanup failed for {label}: {type(error).__name__}: {error}" for label, error in errors]
    if primary is not None:
        for note in notes:
            primary.add_note(note)
        return
    result = CacheError(notes[0])
    for note in notes[1:]:
        result.add_note(note)
    raise result from errors[0][1]


def _metadata_witness(metadata: os.stat_result) -> _StatWitness:
    return _StatWitness(
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        mode=int(metadata.st_mode),
        link_count=int(metadata.st_nlink),
        uid=int(metadata.st_uid),
        gid=int(metadata.st_gid),
        size=int(metadata.st_size),
        mtime_ns=int(metadata.st_mtime_ns),
        ctime_ns=int(metadata.st_ctime_ns),
    )


def _same_metadata(actual: os.stat_result, expected: _StatWitness) -> bool:
    return _metadata_witness(actual) == expected


def _require_verification(value: object) -> ShardVerification:
    if (
        not isinstance(value, ShardVerification)
        or getattr(value, "_seal", None) is not _VERIFICATION_SEAL
        or type(value) is not ShardVerification
    ):
        raise TypeError("verification must be an authenticated ShardVerification")
    return value


def _verification_identity(verification: ShardVerification) -> tuple[object, ...]:
    return (
        verification._root,
        verification._manifest_bytes,
        verification._manifest_sha256,
        verification._root_metadata,
        verification._group_metadata,
        verification._manifest_metadata,
        verification._files,
        verification._array_contract,
        verification._feature_rows,
        verification._transition_rows,
    )


def _close_descriptors(
    descriptors: list[tuple[str, int]],
    *,
    primary: BaseException | None,
    action: str,
) -> None:
    errors: list[tuple[str, BaseException]] = []
    for label, descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append((label, exc))
    descriptors.clear()
    _finish_cleanup(primary, errors, action=action)


def _close_arrays(
    arrays: list[tuple[str, np.ndarray]],
    *,
    primary: BaseException | None,
    action: str,
) -> None:
    errors: list[tuple[str, BaseException]] = []
    seen: set[int] = set()
    for label, value in reversed(arrays):
        memory_map = getattr(value, "_mmap", None)
        if memory_map is None or id(memory_map) in seen:
            continue
        seen.add(id(memory_map))
        try:
            _close_memory_map(memory_map)
        except BaseException as exc:
            errors.append((label, exc))
    arrays.clear()
    _finish_cleanup(primary, errors, action=action)


def _cleanup_arrays_and_descriptors(
    arrays: list[tuple[str, np.ndarray]],
    descriptors: list[tuple[str, int]],
    *,
    primary: BaseException | None,
    action: str,
) -> None:
    errors: list[tuple[str, BaseException]] = []
    seen: set[int] = set()
    for label, value in reversed(arrays):
        memory_map = getattr(value, "_mmap", None)
        if memory_map is None or id(memory_map) in seen:
            continue
        seen.add(id(memory_map))
        try:
            _close_memory_map(memory_map)
        except BaseException as exc:
            errors.append((f"{label} mmap", exc))
    arrays.clear()
    for label, descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append((f"{label} descriptor", exc))
    descriptors.clear()
    _finish_cleanup(primary, errors, action=action)


def _array(value: Any, *, table: str, field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CacheError(f"{table}.{field} must be a numpy array")
    if value.ndim == 0:
        raise CacheError(f"{table}.{field} must have a leading row dimension")
    return value


def _row_count(table_value: Any, *, table: str) -> int:
    count: int | None = None
    for field in dataclasses.fields(table_value):
        value = _array(getattr(table_value, field.name), table=table, field=field.name)
        if count is None:
            count = int(value.shape[0])
        elif value.shape[0] != count:
            raise CacheError(f"{table}.{field.name} leading row count {value.shape[0]} does not match {count}")
    if count is None or count == 0:
        raise CacheError(f"{table} table must contain at least one row")
    return count


def _signed_vector(value: np.ndarray, *, table: str, field: str) -> None:
    if value.ndim != 1:
        raise CacheError(f"{table}.{field} must be rank 1")
    if value.dtype.kind != "i":
        raise CacheError(f"{table}.{field} must have a signed integer dtype")


def _nonnegative_identity_vector(
    value: np.ndarray,
    *,
    table: str,
    field: str,
) -> None:
    _signed_vector(value, table=table, field=field)
    if np.any(value < 0):
        raise CacheError(f"{table}.{field} must contain only nonnegative values")


def _exact_float(
    value: np.ndarray,
    *,
    table: str,
    field: str,
    rank: int,
    dtype: np.dtype,
) -> None:
    if value.ndim != rank:
        raise CacheError(f"{table}.{field} must be rank {rank}")
    if value.dtype != dtype:
        raise CacheError(f"{table}.{field} must have {dtype} dtype")
    if any(dimension == 0 for dimension in value.shape[1:]):
        raise CacheError(f"{table}.{field} trailing dimensions must be nonempty")
    try:
        finite = bool(np.isfinite(value).all())
    except TypeError as exc:
        raise CacheError(f"{table}.{field} must contain finite values") from exc
    if not finite:
        raise CacheError(f"{table}.{field} must contain only finite values")


def _sorted_unique_keys(
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    *,
    table: str,
) -> None:
    keys = [(int(episode), int(frame)) for episode, frame in zip(episode_index, frame_index, strict=True)]
    if len(keys) != len(set(keys)):
        raise CacheError(f"{table} (episode, frame/start) keys must be unique; duplicate found")
    if keys != sorted(keys):
        raise CacheError(f"{table} (episode, frame/start) keys must be lexicographically sorted")


def _validate_features(features: FeatureTable) -> int:
    if not isinstance(features, FeatureTable):
        raise CacheError("features must be a FeatureTable")
    count = _row_count(features, table="features")
    for field in ("episode_index", "frame_index"):
        _nonnegative_identity_vector(
            _array(getattr(features, field), table="features", field=field),
            table="features",
            field=field,
        )
    _exact_float(
        _array(features.z_rl, table="features", field="z_rl"),
        table="features",
        field="z_rl",
        rank=2,
        dtype=np.dtype(ml_dtypes.bfloat16),
    )
    _exact_float(
        _array(features.state_norm, table="features", field="state_norm"),
        table="features",
        field="state_norm",
        rank=2,
        dtype=np.dtype(np.float32),
    )
    _exact_float(
        _array(features.vla_reference, table="features", field="vla_reference"),
        table="features",
        field="vla_reference",
        rank=3,
        dtype=np.dtype(np.float32),
    )
    _sorted_unique_keys(
        features.episode_index,
        features.frame_index,
        table="features",
    )
    return count


def _validate_transitions(
    transition_table: TransitionTable,
    *,
    features: FeatureTable,
) -> int:
    if not isinstance(transition_table, TransitionTable):
        raise CacheError("transitions must be a TransitionTable")
    count = _row_count(transition_table, table="transitions")
    for field in (
        "episode_index",
        "start_frame_index",
    ):
        _nonnegative_identity_vector(
            _array(getattr(transition_table, field), table="transitions", field=field),
            table="transitions",
            field=field,
        )
    for field in ("current_feature_row", "next_feature_row"):
        _signed_vector(
            _array(getattr(transition_table, field), table="transitions", field=field),
            table="transitions",
            field=field,
        )
    for field in ("executed_action", "bc_anchor"):
        _exact_float(
            _array(getattr(transition_table, field), table="transitions", field=field),
            table="transitions",
            field=field,
            rank=3,
            dtype=np.dtype(np.float32),
        )
    if transition_table.executed_action.shape[1:] != transition_table.bc_anchor.shape[1:]:
        raise CacheError("transitions.executed_action and transitions.bc_anchor trailing shape must match")
    if transition_table.executed_action.shape[1:] != features.vla_reference.shape[1:]:
        raise CacheError("features.vla_reference and transitions action/anchor trailing shape must match")
    reward = _array(transition_table.reward, table="transitions", field="reward")
    if reward.ndim != 2 or reward.shape[1:] != (1,):
        raise CacheError("transitions.reward must have shape [N,1]")
    if reward.dtype != np.dtype(np.float32):
        raise CacheError("transitions.reward must have float32 dtype")
    if not np.isfinite(reward).all():
        raise CacheError("transitions.reward must contain only finite values")
    terminal = _array(transition_table.terminal, table="transitions", field="terminal")
    if terminal.ndim != 2 or terminal.shape[1:] != (1,):
        raise CacheError("transitions.terminal must have shape [N,1]")
    if terminal.dtype != np.dtype(np.bool_):
        raise CacheError("transitions.terminal must have bool dtype")
    _sorted_unique_keys(
        transition_table.episode_index,
        transition_table.start_frame_index,
        table="transitions",
    )
    feature_count = int(features.z_rl.shape[0])
    current = transition_table.current_feature_row
    if np.any(current < 0) or np.any(current >= feature_count):
        raise CacheError(f"transitions.current_feature_row must be in [0,{feature_count})")
    next_rows = transition_table.next_feature_row
    if np.any(next_rows < -1) or np.any(next_rows >= feature_count):
        raise CacheError(f"transitions.next_feature_row must be -1 or in [0,{feature_count})")
    if not np.array_equal(next_rows == -1, terminal[:, 0]):
        raise CacheError("transitions.next_feature_row is -1 if and only if transitions.terminal is true")
    for row in range(count):
        episode_index = int(transition_table.episode_index[row])
        start_frame_index = int(transition_table.start_frame_index[row])
        current_row = int(current[row])
        current_identity = (
            int(features.episode_index[current_row]),
            int(features.frame_index[current_row]),
        )
        expected_current = (episode_index, start_frame_index)
        if current_identity != expected_current:
            raise CacheError(
                "transitions.current_feature_row identity mismatch at row "
                f"{row}: expected {expected_current}, got {current_identity}"
            )
        next_row = int(next_rows[row])
        if next_row >= 0:
            next_identity = (
                int(features.episode_index[next_row]),
                int(features.frame_index[next_row]),
            )
            expected_next = (episode_index, start_frame_index + transitions.ACTION_HORIZON)
            if next_identity != expected_next:
                raise CacheError(
                    "transitions.next_feature_row identity mismatch at row "
                    f"{row}: expected {expected_next}, got {next_identity}"
                )
    return count


def _validate_raw(raw: transitions.RawTransitionTable) -> int:
    if not isinstance(raw, transitions.RawTransitionTable):
        raise CacheError("raw transitions must be a RawTransitionTable")
    count = _row_count(raw, table="raw")
    for field in ("episode_index", "start_frame_index"):
        _nonnegative_identity_vector(
            _array(getattr(raw, field), table="raw", field=field),
            table="raw",
            field=field,
        )
    _exact_float(
        _array(raw.executed_action, table="raw", field="executed_action"),
        table="raw",
        field="executed_action",
        rank=3,
        dtype=np.dtype(np.float32),
    )
    intervention = _array(raw.intervention, table="raw", field="intervention")
    if intervention.ndim != 2:
        raise CacheError("raw.intervention must be rank 2")
    if intervention.dtype != np.dtype(np.bool_):
        raise CacheError("raw.intervention must have bool dtype")
    if intervention.shape[1] != raw.executed_action.shape[1]:
        raise CacheError("raw.intervention horizon must match raw.executed_action horizon")
    reward = _array(raw.reward, table="raw", field="reward")
    if reward.ndim != 2 or reward.shape[1:] != (1,):
        raise CacheError("raw.reward must have shape [N,1]")
    if reward.dtype != np.dtype(np.float32):
        raise CacheError("raw.reward must have float32 dtype")
    if not np.isfinite(reward).all():
        raise CacheError("raw.reward must contain only finite values")
    terminal = _array(raw.terminal, table="raw", field="terminal")
    if terminal.ndim != 2 or terminal.shape[1:] != (1,):
        raise CacheError("raw.terminal must have shape [N,1]")
    if terminal.dtype != np.dtype(np.bool_):
        raise CacheError("raw.terminal must have bool dtype")
    return count


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CacheError(f"{path} must be a finite JSON number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CacheError(f"{path} must contain only JSON string keys")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise CacheError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _validate_identity_fields(identity_fields: Any) -> dict[str, Any]:
    if type(identity_fields) is not dict:
        raise CacheError("identity_fields must be a JSON object")
    _validate_json_value(identity_fields, path="identity_fields")
    overlap = _RESERVED_MANIFEST_FIELDS.intersection(identity_fields)
    if overlap:
        raise CacheError(f"identity_fields contains reserved manifest fields {sorted(overlap)}")
    for field in ("feature_identity", "batch_id"):
        value = identity_fields.get(field)
        if type(value) is not str or not value:
            raise CacheError(f"identity_fields.{field} must be a nonempty string")
    for field, value in identity_fields.items():
        if field.endswith("_sha256") and (type(value) is not str or _LOWERCASE_SHA256.fullmatch(value) is None):
            raise CacheError(f"identity_fields.{field} must be a lowercase 64-hex SHA-256")
    return dict(identity_fields)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_bytes(identity.canonical_json_bytes(manifest))
    _fsync_file(path)


def _atomic_publish_noreplace(staging: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CacheError("atomic no-replace directory publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(staging),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def publish_shard(
    destination: Path,
    *,
    features: FeatureTable,
    transitions: TransitionTable,
    identity_fields: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    feature_rows = _validate_features(features)
    transition_rows = _validate_transitions(
        transitions,
        features=features,
    )
    validated_identity = _validate_identity_fields(identity_fields)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        files: list[dict[str, Any]] = []
        for group_name, table in (
            ("features", features),
            ("transitions", transitions),
        ):
            group = staging / group_name
            group.mkdir()
            for field in dataclasses.fields(table):
                path = group / f"{field.name}.npy"
                array = np.ascontiguousarray(getattr(table, field.name))
                np.save(path, array, allow_pickle=False)
                _fsync_file(path)
                files.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": identity.sha256_file(path),
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                    }
                )
            _fsync_directory(group)
        manifest = {
            "schema_version": 1,
            **validated_identity,
            "feature_rows": feature_rows,
            "transition_rows": transition_rows,
            "files": sorted(files, key=lambda item: item["path"]),
        }
        _write_manifest(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        authenticate_shard(staging)
        _atomic_publish_noreplace(staging, destination)
        staging = None
        _fsync_directory(destination.parent)
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _reject_symlink_chain(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise CacheError(f"cache root is missing: {current}") from exc
        if stat.S_ISLNK(mode):
            location = "root" if current == absolute else "ancestor"
            raise CacheError(f"cache {location} symlink is forbidden: {current}")
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise CacheError(f"cache root is not a real directory: {absolute}")
    return absolute.resolve(strict=True)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheError(f"duplicate JSON key {key!r} in cache manifest")
        result[key] = value
    return result


def _open_pinned_regular(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CacheError(f"{label} cannot be opened safely without O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise CacheError(f"{label} symlink is forbidden") from exc
        if exc.errno == errno.ENOENT:
            raise CacheError(f"{label} is missing") from exc
        raise CacheError(f"{label} failed to open safely: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise CacheError(f"{label} fstat failed: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise CacheError(f"{label} is not a regular file")
    return descriptor, metadata


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise CacheError("safe cache traversal requires O_NOFOLLOW and O_DIRECTORY")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _open_directory_chain(path: Path) -> tuple[Path, int, os.stat_result]:
    """Open every absolute path component with no-follow directory semantics."""

    requested = Path(path)
    if ".." in requested.parts:
        raise CacheError(f"cache root contains a forbidden parent component '..': {requested}")
    absolute = Path(os.path.abspath(requested))
    if absolute != Path(os.path.normpath(absolute)):
        raise CacheError(f"cache root must be normalized: {absolute}")
    flags = _directory_flags()
    descriptors: list[tuple[str, int]] = []
    try:
        descriptor = os.open(absolute.anchor, flags)
        descriptors.append((absolute.anchor, descriptor))
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            current /= component
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    with contextlib.suppress(OSError):
                        component_metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                        if stat.S_ISLNK(component_metadata.st_mode):
                            location = "root" if current == absolute else "ancestor"
                            raise CacheError(f"cache {location} symlink is forbidden: {current}") from exc
                    raise CacheError(f"cache path component is not a real directory: {current}") from exc
                if exc.errno == errno.ENOENT:
                    raise CacheError(f"cache root is missing: {current}") from exc
                raise CacheError(f"cache path component failed to open safely: {current}: {exc}") from exc
            descriptors.append((str(current), child))
            try:
                os.close(descriptor)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    os.close(child)
                raise CacheError(f"cache ancestor descriptor failed to close: {current.parent}: {exc}") from exc
            descriptors.pop(-2)
            descriptor = child
        metadata = os.fstat(descriptor)
        descriptors.clear()
        return absolute, descriptor, metadata
    except BaseException:
        _close_descriptors(
            descriptors,
            primary=sys.exception(),
            action="traverse cache root",
        )
        raise


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> tuple[int, os.stat_result]:
    if "/" in name or name in {"", ".", ".."}:
        raise CacheError(f"unsafe directory component for {label}: {name!r}")
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise CacheError(f"{label} is not a real directory") from exc
        if exc.errno == errno.ENOENT:
            raise CacheError(f"{label} is missing") from exc
        raise CacheError(f"{label} failed to open safely: {exc}") from exc
    try:
        return descriptor, os.fstat(descriptor)
    except BaseException:
        primary = sys.exception()
        try:
            os.close(descriptor)
        except BaseException as cleanup:
            if primary is not None:
                primary.add_note(f"{label} descriptor cleanup failed: {cleanup}")
        raise


def _open_pinned_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    if "/" in name or name in {"", ".", ".."}:
        raise CacheError(f"unsafe file component for {label}: {name!r}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CacheError(f"{label} cannot be opened safely without O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise CacheError(f"{label} symlink is forbidden") from exc
        if exc.errno == errno.ENOENT:
            raise CacheError(f"{label} is missing") from exc
        raise CacheError(f"{label} failed to open safely: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CacheError(f"{label} is not a regular file")
        return descriptor, metadata
    except BaseException:
        primary = sys.exception()
        try:
            os.close(descriptor)
        except BaseException as cleanup:
            if primary is not None:
                primary.add_note(f"{label} descriptor cleanup failed: {cleanup}")
        raise


def _read_fd_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _proc_fd_path(descriptor: int, *, label: str) -> Path:
    proc_root = Path("/proc/self/fd")
    try:
        mode = os.stat(proc_root).st_mode
    except OSError as exc:
        raise CacheError(f"{label} cannot be memory-mapped: /proc/self/fd is unavailable") from exc
    if not stat.S_ISDIR(mode):
        raise CacheError(f"{label} cannot be memory-mapped: /proc/self/fd is not a directory")
    proc_path = proc_root / str(descriptor)
    try:
        os.stat(proc_path)
    except OSError as exc:
        raise CacheError(f"{label} pinned file descriptor is unavailable in /proc/self/fd") from exc
    return proc_path


def _before_mmap_load(_path: Path, _descriptor: int) -> None:
    """Test hook for exercising the verified-inode to mmap race boundary."""


def _semantic_dtype(relative: str, dtype: np.dtype, expected: str) -> np.dtype:
    if relative == "features/z_rl.npy" and expected == "bfloat16" and dtype == np.dtype("V2"):
        return np.dtype(ml_dtypes.bfloat16)
    return dtype


def _read_npy_header(
    descriptor: int,
    *,
    relative: str,
    expected_dtype: str,
) -> tuple[tuple[int, ...], np.dtype]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except Exception as exc:
        raise CacheError(f"{relative} failed to load/parse as a safe numpy array header") from exc
    if fortran_order:
        raise CacheError(f"{relative} must be stored in C order")
    semantic = _semantic_dtype(relative, np.dtype(dtype), expected_dtype)
    return tuple(int(dimension) for dimension in shape), semantic


def _mmap_verified_array(
    descriptor: int,
    *,
    path: Path,
    relative: str,
    shape: tuple[int, ...],
    dtype: str,
) -> np.memmap:
    proc_path = _proc_fd_path(descriptor, label=relative)
    _before_mmap_load(path, descriptor)
    try:
        loaded = np.load(proc_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise CacheError(f"{relative} failed to load as a safe numpy array") from exc
    if not isinstance(loaded, np.memmap):
        raise CacheError(f"{relative} did not open as a memory map")
    value: np.memmap = loaded
    if relative == "features/z_rl.npy" and dtype == "bfloat16" and value.dtype == np.dtype("V2"):
        value = value.view(np.dtype(ml_dtypes.bfloat16))
    if tuple(value.shape) != shape:
        _close_arrays([(relative, value)], primary=sys.exception(), action="reject cache mmap")
        raise CacheError(f"{relative} shape mismatch: expected {list(shape)}, got {list(value.shape)}")
    if str(value.dtype) != dtype:
        _close_arrays([(relative, value)], primary=sys.exception(), action="reject cache mmap")
        raise CacheError(f"{relative} dtype mismatch: expected {dtype}, got {value.dtype}")
    if value.flags.writeable:
        _close_arrays([(relative, value)], primary=sys.exception(), action="reject cache mmap")
        raise CacheError(f"{relative} memory map must be read-only")
    return value


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    descriptor, _ = _open_pinned_regular(
        manifest_path,
        label=f"cache manifest {manifest_path}",
    )
    try:
        payload_bytes = _read_fd_bytes(descriptor)
        manifest_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except CacheError:
        raise
    except Exception as exc:
        raise CacheError(f"failed to parse cache manifest {manifest_path}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)
    if type(payload) is not dict:
        raise CacheError(f"cache manifest must be a JSON object: {manifest_path}")
    return payload, manifest_sha256


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise CacheError(f"cache manifest {field} must be a nonnegative integer")
    return value


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    identity_fields = {key: value for key, value in manifest.items() if key not in _RESERVED_MANIFEST_FIELDS}
    return _validate_identity_fields(identity_fields)


def _expected_paths() -> set[str]:
    return {
        *(f"features/{field.name}.npy" for field in dataclasses.fields(FeatureTable)),
        *(f"transitions/{field.name}.npy" for field in dataclasses.fields(TransitionTable)),
    }


def _validate_file_records(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    records = manifest.get("files")
    if type(records) is not list:
        raise CacheError(f"cache manifest files must be a list: {manifest_path}")
    by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != _FILE_RECORD_FIELDS:
            raise CacheError(f"cache file record {index} must contain exactly {sorted(_FILE_RECORD_FIELDS)}")
        relative = record["path"]
        pure = PurePosixPath(relative) if type(relative) is str else None
        if pure is None or not relative or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise CacheError(f"unsafe cache path in {manifest_path}: {relative!r}")
        if relative in by_path:
            raise CacheError(f"duplicate cache path in {manifest_path}: {relative}")
        if type(record["size"]) is not int or record["size"] < 0:
            raise CacheError(f"{relative} size must be a nonnegative integer")
        if type(record["sha256"]) is not str or _LOWERCASE_SHA256.fullmatch(record["sha256"]) is None:
            raise CacheError(f"{relative} sha256 must be lowercase 64-hex")
        shape = record["shape"]
        if (
            type(shape) is not list
            or not shape
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise CacheError(f"{relative} shape must be a nonempty integer list")
        if type(record["dtype"]) is not str or not record["dtype"]:
            raise CacheError(f"{relative} dtype must be a nonempty string")
        by_path[relative] = record
    expected = _expected_paths()
    if set(by_path) != expected:
        missing = sorted(expected - set(by_path))
        extra = sorted(set(by_path) - expected)
        raise CacheError(f"cache manifest file set mismatch; missing={missing}, extra={extra}")
    return by_path


def _validate_physical_layout(root: Path, expected_paths: set[str]) -> None:
    expected_root = {"manifest.json", "features", "transitions"}
    actual_root = {entry.name for entry in root.iterdir()}
    if actual_root != expected_root:
        raise CacheError(
            f"cache payload root set mismatch; "
            f"missing={sorted(expected_root - actual_root)}, "
            f"extra={sorted(actual_root - expected_root)}"
        )
    for group in ("features", "transitions"):
        group_path = root / group
        mode = os.lstat(group_path).st_mode
        if stat.S_ISLNK(mode):
            raise CacheError(f"cache payload symlink is forbidden: {group_path}")
        if not stat.S_ISDIR(mode):
            raise CacheError(f"cache payload group is not a directory: {group_path}")
        expected_names = {
            PurePosixPath(relative).name for relative in expected_paths if relative.startswith(f"{group}/")
        }
        actual_names = {entry.name for entry in group_path.iterdir()}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise CacheError(f"cache payload {group} set mismatch; missing={missing}, extra={extra}")


def _load_array(
    root: Path,
    relative: str,
    record: dict[str, Any],
) -> np.memmap:
    path = root.joinpath(*PurePosixPath(relative).parts)
    descriptor, metadata = _open_pinned_regular(path, label=relative)
    value: np.memmap | None = None
    try:
        if metadata.st_size != record["size"]:
            raise CacheError(f"{relative} size mismatch: expected {record['size']}, got {metadata.st_size}")
        try:
            actual_sha256 = _sha256_fd(descriptor)
        except Exception as exc:
            raise CacheError(f"{relative} sha256 read failed: {exc}") from exc
        if actual_sha256 != record["sha256"]:
            raise CacheError(f"{relative} sha256 mismatch: expected {record['sha256']}, got {actual_sha256}")
        proc_path = _proc_fd_path(descriptor, label=relative)
        _before_mmap_load(path, descriptor)
        try:
            loaded = np.load(proc_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise CacheError(f"{relative} failed to load as a safe numpy array") from exc
        if not isinstance(loaded, np.memmap):
            raise CacheError(f"{relative} did not open as a memory map")
        value = loaded
        # NPY's dtype descriptor has no spelling for ml_dtypes.bfloat16.
        # Only the fixed z_rl field may restore raw two-byte elements as the
        # authenticated semantic BF16 dtype, without copying the mmap.
        if relative == "features/z_rl.npy" and record["dtype"] == "bfloat16" and value.dtype == np.dtype("V2"):
            value = value.view(np.dtype(ml_dtypes.bfloat16))
        if list(value.shape) != record["shape"]:
            raise CacheError(f"{relative} shape mismatch: manifest {record['shape']}, array {list(value.shape)}")
        if str(value.dtype) != record["dtype"]:
            raise CacheError(f"{relative} dtype mismatch: manifest {record['dtype']}, array {value.dtype}")
        if value.flags.writeable:
            raise CacheError(f"{relative} memory map must be read-only")
        return value
    except BaseException:
        if value is not None:
            memory_map = getattr(value, "_mmap", None)
            if memory_map is not None:
                with contextlib.suppress(OSError, BufferError):
                    memory_map.close()
        raise
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _manifest_from_bytes(payload_bytes: bytes, *, manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except CacheError:
        raise
    except Exception as exc:
        raise CacheError(f"failed to parse cache manifest {manifest_path}") from exc
    if type(payload) is not dict:
        raise CacheError(f"cache manifest must be a JSON object: {manifest_path}")
    return payload


def _read_manifest_at(
    root_descriptor: int,
    root: Path,
) -> tuple[dict[str, Any], bytes, str, _StatWitness]:
    path = root / "manifest.json"
    descriptor, metadata = _open_pinned_regular_at(
        root_descriptor,
        "manifest.json",
        label=f"cache manifest {path}",
    )
    try:
        payload_bytes = _read_fd_bytes(descriptor)
        payload = _manifest_from_bytes(payload_bytes, manifest_path=path)
        return payload, payload_bytes, hashlib.sha256(payload_bytes).hexdigest(), _metadata_witness(metadata)
    finally:
        primary = sys.exception()
        try:
            os.close(descriptor)
        except BaseException as exc:
            _finish_cleanup(primary, [(str(path), exc)], action="read cache manifest")


def _validate_descriptor_layout(
    root_descriptor: int,
    group_descriptors: dict[str, int],
    expected_paths: set[str],
) -> None:
    expected_root = {"manifest.json", "features", "transitions"}
    actual_root = set(os.listdir(root_descriptor))
    if actual_root != expected_root:
        raise CacheError(
            "cache payload root set mismatch; "
            f"missing={sorted(expected_root - actual_root)}, "
            f"extra={sorted(actual_root - expected_root)}"
        )
    for group, descriptor in group_descriptors.items():
        expected_names = {
            PurePosixPath(relative).name for relative in expected_paths if relative.startswith(f"{group}/")
        }
        actual_names = set(os.listdir(descriptor))
        if actual_names != expected_names:
            raise CacheError(
                f"cache payload {group} set mismatch; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )


def _field_relative_paths() -> tuple[str, ...]:
    return (
        *(f"features/{field.name}.npy" for field in dataclasses.fields(FeatureTable)),
        *(f"transitions/{field.name}.npy" for field in dataclasses.fields(TransitionTable)),
    )


def _table_from_values(
    values: dict[str, np.ndarray],
    table_type: type[FeatureTable] | type[TransitionTable],
    group: str,
) -> FeatureTable | TransitionTable:
    return table_type(**{field.name: values[f"{group}/{field.name}.npy"] for field in dataclasses.fields(table_type)})


def _new_verification(
    *,
    root: Path,
    manifest_bytes: bytes,
    manifest_sha256: str,
    root_metadata: _StatWitness,
    group_metadata: tuple[tuple[str, _StatWitness], ...],
    manifest_metadata: _StatWitness,
    files: tuple[ShardFileVerification, ...],
    feature_rows: int,
    transition_rows: int,
) -> ShardVerification:
    array_contract = tuple(ShardArrayContract(path=item.path, shape=item.shape, dtype=item.dtype) for item in files)
    return ShardVerification(
        _seal=_VERIFICATION_SEAL,
        root=root,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
        root_metadata=root_metadata,
        group_metadata=group_metadata,
        manifest_metadata=manifest_metadata,
        files=files,
        array_contract=array_contract,
        feature_rows=feature_rows,
        transition_rows=transition_rows,
    )


def authenticate_shard(root: Path) -> ShardVerification:
    """Fully authenticate a shard once and return an FD-free sealed capability."""

    requested_root = Path(root)
    arrays: list[tuple[str, np.ndarray]] = []
    descriptors: list[tuple[str, int]] = []
    try:
        real_root, root_descriptor, root_stat = _open_directory_chain(requested_root)
        descriptors.append((str(real_root), root_descriptor))
        manifest, manifest_bytes, manifest_sha256, manifest_metadata = _read_manifest_at(
            root_descriptor,
            real_root,
        )
        manifest_path = real_root / "manifest.json"
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise CacheError(f"cache manifest schema_version must be integer 1: {manifest_path}")
        feature_rows = _strict_nonnegative_int(manifest.get("feature_rows"), field="feature_rows")
        transition_rows = _strict_nonnegative_int(manifest.get("transition_rows"), field="transition_rows")
        _manifest_identity(manifest)
        records = _validate_file_records(manifest, manifest_path=manifest_path)

        group_descriptors: dict[str, int] = {}
        group_witnesses: list[tuple[str, _StatWitness]] = []
        for group in ("features", "transitions"):
            descriptor, metadata = _open_directory_at(
                root_descriptor,
                group,
                label=f"cache payload group {real_root / group}",
            )
            descriptors.append((str(real_root / group), descriptor))
            group_descriptors[group] = descriptor
            group_witnesses.append((group, _metadata_witness(metadata)))

        _validate_descriptor_layout(root_descriptor, group_descriptors, set(records))

        values: dict[str, np.ndarray] = {}
        file_verifications: list[ShardFileVerification] = []
        for relative in _field_relative_paths():
            record = records[relative]
            pure = PurePosixPath(relative)
            group, name = pure.parts
            path = real_root / group / name
            descriptor, metadata = _open_pinned_regular_at(
                group_descriptors[group],
                name,
                label=relative,
            )
            descriptors.append((relative, descriptor))
            if int(metadata.st_size) != record["size"]:
                raise CacheError(f"{relative} size mismatch: expected {record['size']}, got {metadata.st_size}")
            actual_sha256 = _sha256_fd(descriptor)
            if actual_sha256 != record["sha256"]:
                raise CacheError(f"{relative} sha256 mismatch: expected {record['sha256']}, got {actual_sha256}")
            header_shape, header_dtype = _read_npy_header(
                descriptor,
                relative=relative,
                expected_dtype=record["dtype"],
            )
            if header_shape != tuple(record["shape"]):
                raise CacheError(f"{relative} shape mismatch: manifest {record['shape']}, header {list(header_shape)}")
            if str(header_dtype) != record["dtype"]:
                raise CacheError(f"{relative} dtype mismatch: manifest {record['dtype']}, header {header_dtype}")
            value = _mmap_verified_array(
                descriptor,
                path=path,
                relative=relative,
                shape=header_shape,
                dtype=record["dtype"],
            )
            arrays.append((relative, value))
            values[relative] = value
            file_verifications.append(
                ShardFileVerification(
                    path=relative,
                    shape=header_shape,
                    dtype=record["dtype"],
                    size=record["size"],
                    sha256=record["sha256"],
                    _metadata=_metadata_witness(metadata),
                )
            )
            os.close(descriptor)
            descriptors.pop()

        features = _table_from_values(values, FeatureTable, "features")
        transition_table = _table_from_values(values, TransitionTable, "transitions")
        assert isinstance(features, FeatureTable)
        assert isinstance(transition_table, TransitionTable)
        actual_feature_rows = _validate_features(features)
        actual_transition_rows = _validate_transitions(transition_table, features=features)
        if feature_rows != actual_feature_rows:
            raise CacheError(
                f"cache manifest feature_rows {feature_rows} does not match features table {actual_feature_rows}"
            )
        if transition_rows != actual_transition_rows:
            raise CacheError(
                f"cache manifest transition_rows {transition_rows} does not match "
                f"transitions table {actual_transition_rows}"
            )
        verification = _new_verification(
            root=real_root,
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            root_metadata=_metadata_witness(root_stat),
            group_metadata=tuple(group_witnesses),
            manifest_metadata=manifest_metadata,
            files=tuple(file_verifications),
            feature_rows=feature_rows,
            transition_rows=transition_rows,
        )
        _quick_verify_verification(verification)
        return verification
    except CacheError:
        raise
    except BaseException as exc:
        raise CacheError(f"failed to authenticate cache shard {requested_root}: {exc}") from exc
    finally:
        _cleanup_arrays_and_descriptors(
            arrays,
            descriptors,
            primary=sys.exception(),
            action="authenticate cache shard",
        )


def _quick_verify_verification(verification: ShardVerification) -> None:
    verification = _require_verification(verification)
    descriptors: list[tuple[str, int]] = []
    try:
        root, root_descriptor, root_metadata = _open_directory_chain(verification.root)
        descriptors.append((str(root), root_descriptor))
        if root != verification.root or not _same_metadata(root_metadata, verification._root_metadata):
            raise CacheError(f"cache root binding changed: {verification.root}")
        group_descriptors: dict[str, int] = {}
        expected_groups = dict(verification._group_metadata)
        for group in ("features", "transitions"):
            descriptor, metadata = _open_directory_at(
                root_descriptor,
                group,
                label=f"cache payload group {root / group}",
            )
            descriptors.append((str(root / group), descriptor))
            group_descriptors[group] = descriptor
            if not _same_metadata(metadata, expected_groups[group]):
                raise CacheError(f"cache payload group binding changed: {root / group}")
        _validate_descriptor_layout(
            root_descriptor,
            group_descriptors,
            {item.path for item in verification.files},
        )
        manifest, manifest_bytes, manifest_sha256, manifest_metadata = _read_manifest_at(root_descriptor, root)
        if (
            manifest_bytes != verification._manifest_bytes
            or manifest_sha256 != verification.manifest_sha256
            or not _same_metadata(
                os.stat("manifest.json", dir_fd=root_descriptor, follow_symlinks=False),
                verification._manifest_metadata,
            )
            or manifest_metadata != verification._manifest_metadata
            or manifest != verification.manifest
        ):
            raise CacheError(f"cache manifest binding changed: {root / 'manifest.json'}")
        for item in verification.files:
            pure = PurePosixPath(item.path)
            group, name = pure.parts
            descriptor, metadata = _open_pinned_regular_at(
                group_descriptors[group],
                name,
                label=item.path,
            )
            descriptors.append((item.path, descriptor))
            if not _same_metadata(metadata, item._metadata) or int(metadata.st_size) != item.size:
                raise CacheError(f"cache payload binding changed: {item.path}")
            shape, dtype = _read_npy_header(
                descriptor,
                relative=item.path,
                expected_dtype=item.dtype,
            )
            if shape != item.shape or str(dtype) != item.dtype:
                raise CacheError(f"cache payload header changed: {item.path}")
            current = os.stat(name, dir_fd=group_descriptors[group], follow_symlinks=False)
            if not _same_metadata(current, item._metadata):
                raise CacheError(f"cache payload path changed while verifying: {item.path}")
            os.close(descriptor)
            descriptors.pop()
        if not _same_metadata(os.fstat(root_descriptor), verification._root_metadata):
            raise CacheError(f"cache root metadata changed while verifying: {root}")
        for group, descriptor in group_descriptors.items():
            expected = expected_groups[group]
            if not _same_metadata(os.fstat(descriptor), expected) or not _same_metadata(
                os.stat(group, dir_fd=root_descriptor, follow_symlinks=False),
                expected,
            ):
                raise CacheError(f"cache payload group path changed while verifying: {root / group}")
        reopened_root, reopened_descriptor, reopened_metadata = _open_directory_chain(root)
        descriptors.append((f"{root} rebound", reopened_descriptor))
        if reopened_root != root or not _same_metadata(reopened_metadata, verification._root_metadata):
            raise CacheError(f"cache root path changed while verifying: {root}")
        os.close(reopened_descriptor)
        descriptors.pop()
    finally:
        _close_descriptors(
            descriptors,
            primary=sys.exception(),
            action="quick verify cache shard",
        )


def verify_verification(
    verification: ShardVerification,
    *,
    full: bool = False,
) -> None:
    verification = _require_verification(verification)
    if type(full) is not bool:
        raise TypeError("full must be an exact bool")
    if not full:
        _quick_verify_verification(verification)
        return
    observed = authenticate_shard(verification.root)
    if observed != verification:
        raise CacheError(f"cache shard full verification changed: {verification.root}")


def open_from_verification(verification: ShardVerification) -> OpenShard:
    """Open read-only mmaps from an authenticated capability without rehashing."""

    verification = _require_verification(verification)
    arrays: list[tuple[str, np.ndarray]] = []
    descriptors: list[tuple[str, int]] = []
    try:
        _quick_verify_verification(verification)
        root, root_descriptor, root_metadata = _open_directory_chain(verification.root)
        descriptors.append((str(root), root_descriptor))
        if not _same_metadata(root_metadata, verification._root_metadata):
            raise CacheError(f"cache root binding changed: {root}")
        group_descriptors: dict[str, int] = {}
        expected_groups = dict(verification._group_metadata)
        for group in ("features", "transitions"):
            descriptor, metadata = _open_directory_at(
                root_descriptor,
                group,
                label=f"cache payload group {root / group}",
            )
            descriptors.append((str(root / group), descriptor))
            group_descriptors[group] = descriptor
            if not _same_metadata(metadata, expected_groups[group]):
                raise CacheError(f"cache payload group binding changed: {root / group}")

        values: dict[str, np.ndarray] = {}
        for item in verification.files:
            pure = PurePosixPath(item.path)
            group, name = pure.parts
            path = root / group / name
            descriptor, metadata = _open_pinned_regular_at(
                group_descriptors[group],
                name,
                label=item.path,
            )
            descriptors.append((item.path, descriptor))
            if not _same_metadata(metadata, item._metadata):
                raise CacheError(f"cache payload binding changed: {item.path}")
            shape, dtype = _read_npy_header(
                descriptor,
                relative=item.path,
                expected_dtype=item.dtype,
            )
            if shape != item.shape or str(dtype) != item.dtype:
                raise CacheError(f"cache payload header changed: {item.path}")
            value = _mmap_verified_array(
                descriptor,
                path=path,
                relative=item.path,
                shape=item.shape,
                dtype=item.dtype,
            )
            arrays.append((item.path, value))
            values[item.path] = value
            current = os.stat(name, dir_fd=group_descriptors[group], follow_symlinks=False)
            if not _same_metadata(current, item._metadata):
                raise CacheError(f"cache payload path changed while opening: {item.path}")
            os.close(descriptor)
            descriptors.pop()

        _quick_verify_verification(verification)
        features = _table_from_values(values, FeatureTable, "features")
        transition_table = _table_from_values(values, TransitionTable, "transitions")
        assert isinstance(features, FeatureTable)
        assert isinstance(transition_table, TransitionTable)
        opened = OpenShard(
            root=verification.root,
            manifest=verification.manifest,
            manifest_sha256=verification.manifest_sha256,
            features=features,
            transitions=transition_table,
            verification=verification,
        )
        arrays.clear()
        return opened
    except CacheError:
        raise
    except BaseException as exc:
        raise CacheError(f"failed to open authenticated cache shard {verification.root}: {exc}") from exc
    finally:
        _cleanup_arrays_and_descriptors(
            arrays,
            descriptors,
            primary=sys.exception(),
            action="open authenticated cache shard",
        )


def open_shard(root: Path) -> OpenShard:
    return open_from_verification(authenticate_shard(root))


def feature_row_lookup(features: FeatureTable) -> dict[tuple[int, int], int]:
    _validate_features(features)
    result: dict[tuple[int, int], int] = {}
    for row, (episode, frame) in enumerate(zip(features.episode_index, features.frame_index, strict=True)):
        key = (int(episode), int(frame))
        if key in result:
            raise CacheError(f"features contain duplicate feature key {key}")
        result[key] = row
    return result


def finalize_transition_table(
    batch: admission.ValidatedBatch,
    plan: transitions.TransitionPlan,
    raw: transitions.RawTransitionTable,
    features: FeatureTable,
) -> TransitionTable:
    if not isinstance(batch, admission.ValidatedBatch):
        raise CacheError("batch must be a ValidatedBatch")
    if not isinstance(plan, transitions.TransitionPlan):
        raise CacheError("plan must be a TransitionPlan")
    try:
        expected_plan = transitions.build_transition_plan(batch)
    except Exception as exc:
        raise CacheError(f"batch {batch.batch_id} canonical transition plan failed: {exc}") from exc
    if plan != expected_plan:
        raise CacheError(
            f"batch {batch.batch_id} transition plan does not exactly match canonical batch identity/order"
        )

    raw_rows = _validate_raw(raw)
    _validate_features(features)
    if len(plan.rows) != raw_rows:
        raise CacheError(
            f"batch {batch.batch_id} raw transition row count {raw_rows} does not match plan {len(plan.rows)}"
        )
    expected_episode = np.asarray(
        [row.episode_index for row in plan.rows],
        dtype=raw.episode_index.dtype,
    )
    expected_start = np.asarray(
        [row.start_frame_index for row in plan.rows],
        dtype=raw.start_frame_index.dtype,
    )
    expected_reward = np.asarray(
        [[row.reward] for row in plan.rows],
        dtype=np.float32,
    )
    expected_terminal = np.asarray(
        [[row.terminal] for row in plan.rows],
        dtype=np.bool_,
    )
    for field, actual, expected in (
        ("episode_index", raw.episode_index, expected_episode),
        ("start_frame_index", raw.start_frame_index, expected_start),
        ("reward", raw.reward, expected_reward),
        ("terminal", raw.terminal, expected_terminal),
    ):
        if not np.array_equal(actual, expected):
            raise CacheError(f"batch {batch.batch_id} raw.{field} does not match transition plan order")
    if raw.executed_action.shape[1:] != features.vla_reference.shape[1:]:
        raise CacheError(
            f"batch {batch.batch_id} raw.executed_action trailing shape does not match features.vla_reference"
        )

    lookup = feature_row_lookup(features)
    expected_keys = {(key.episode_index, key.frame_index) for key in plan.feature_keys}
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys - set(lookup))
        extra = sorted(set(lookup) - expected_keys)
        raise CacheError(
            f"batch {batch.batch_id} feature rows do not exactly match "
            f"transition plan keys; missing={missing}, extra={extra}"
        )

    current_rows: list[int] = []
    next_rows: list[int] = []
    anchors: list[np.ndarray] = []
    for index, row in enumerate(plan.rows):
        context = f"batch {batch.batch_id} episode {row.episode_index} window {row.start_frame_index}"
        current_key = (
            row.current_key.episode_index,
            row.current_key.frame_index,
        )
        try:
            current = lookup[current_key]
            next_row = (
                -1
                if row.next_key is None
                else lookup[
                    (
                        row.next_key.episode_index,
                        row.next_key.frame_index,
                    )
                ]
            )
            anchor = transitions.bc_anchor(
                features.vla_reference[current],
                raw.executed_action[index],
                raw.intervention[index],
            )
        except (KeyError, ValueError) as exc:
            raise CacheError(f"{context}: failed to construct feature rows/BC anchor") from exc
        current_rows.append(current)
        next_rows.append(next_row)
        anchors.append(anchor)

    result = TransitionTable(
        episode_index=raw.episode_index.astype(np.int32, copy=True),
        start_frame_index=raw.start_frame_index.astype(np.int32, copy=True),
        current_feature_row=np.asarray(current_rows, dtype=np.int64),
        next_feature_row=np.asarray(next_rows, dtype=np.int64),
        executed_action=raw.executed_action.astype(np.float32, copy=True),
        bc_anchor=np.stack(anchors, axis=0).astype(np.float32, copy=False),
        reward=raw.reward.astype(np.float32, copy=True),
        terminal=raw.terminal.astype(np.bool_, copy=True),
    )
    _validate_transitions(
        result,
        features=features,
    )
    return result
