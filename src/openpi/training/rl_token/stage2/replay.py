from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import ctypes
import dataclasses
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import threading
from types import MappingProxyType
from typing import Any

import ml_dtypes
import numpy as np

from openpi.training.rl_token.stage2 import cache
from openpi.training.rl_token.stage2 import identity

_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_FIELDS = {
    "schema_version",
    "feature_identity",
    "stage1_config",
    "stage2_config",
    "reward_source",
    "reward_label_values",
    "completion_label",
    "reward_aggregation",
    "reward_schema_version",
    "total_transitions",
    "shards",
}
_SHARD_RECORD_FIELDS = {
    "batch_id",
    "root",
    "admission_sha256",
    "cache_manifest_sha256",
    "tristate_labels_sha256",
    "transition_rows",
    "start",
    "end",
}
_READ_CHUNK_BYTES = 8 * 1024 * 1024
_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_INOTIFY_CONTENT_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
)
_INOTIFY_EVENT = struct.Struct("iIII")


class ReplayError(RuntimeError):
    pass


def _before_integrity_guard_close(_descriptor: int) -> None:
    """Test hook for failures known to happen before the close syscall."""


@dataclasses.dataclass(frozen=True, slots=True)
class _FileWitness:
    device: int
    inode: int
    mode: int
    link_count: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _file_witness(metadata: os.stat_result) -> _FileWitness:
    return _FileWitness(
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


def _finish_cleanup(
    primary: BaseException | None,
    errors: list[tuple[str, BaseException]],
    *,
    action: str,
) -> None:
    if not errors:
        return
    notes = [f"{action} failed for {label}: {type(error).__name__}: {error}" for label, error in errors]
    if primary is not None:
        for note in notes:
            primary.add_note(note)
        return
    result = ReplayError(notes[0])
    for note in notes[1:]:
        result.add_note(note)
    raise result from errors[0][1]


@dataclasses.dataclass(slots=True)
class _WatchScope:
    path: Path
    names: set[str]


class _IntegrityGuard:
    """One inotify descriptor covering the snapshot and every shard tree.

    inotify reports mutations made through this kernel.  NFS changes made by a
    different client are not guaranteed to emit events here; persistent remote
    changes are still caught by the metadata/header checks, and ``full=True``
    rehashes every payload.  We deliberately do not claim protection against a
    remote client changing and perfectly restoring bytes and metadata between
    checks.
    """

    def __init__(self, descriptor: int):
        self._descriptor = descriptor
        self._scopes: dict[int, _WatchScope] = {}

    @classmethod
    def open(cls) -> _IntegrityGuard:
        libc = ctypes.CDLL(None, use_errno=True)
        init = getattr(libc, "inotify_init1", None)
        if init is None:
            raise ReplayError("replay integrity lease requires Linux inotify_init1")
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        descriptor = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if descriptor < 0:
            number = ctypes.get_errno()
            raise ReplayError(f"failed to initialize replay integrity guard: {os.strerror(number)}")
        return cls(descriptor)

    @property
    def closed(self) -> bool:
        return self._descriptor < 0

    def _watch(self, path: Path, names: set[str]) -> int:
        if self.closed:
            raise ReplayError("replay integrity guard is closed")
        libc = ctypes.CDLL(None, use_errno=True)
        add_watch = getattr(libc, "inotify_add_watch", None)
        if add_watch is None:
            raise ReplayError("replay integrity lease requires Linux inotify_add_watch")
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        watch = add_watch(
            self._descriptor,
            os.fsencode(path),
            _INOTIFY_CONTENT_MASK,
        )
        if watch < 0:
            number = ctypes.get_errno()
            raise ReplayError(f"failed to watch replay lineage path {path}: {os.strerror(number)}")
        scope = self._scopes.get(watch)
        if scope is None:
            self._scopes[watch] = _WatchScope(path=path, names=set(names))
        else:
            scope.names.update(names)
        return watch

    def _watch_ancestor_chain(self, path: Path) -> None:
        absolute = Path(os.path.abspath(path))
        if ".." in Path(path).parts or Path(os.path.normpath(absolute)) != absolute:
            raise ReplayError(f"replay integrity path must be normalized: {path}")
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            self._watch(current, {component})
            current /= component

    def arm_snapshot(self, path: Path) -> None:
        self._watch_ancestor_chain(path)

    def arm_shard(self, root: Path) -> None:
        root = Path(root)
        self._watch_ancestor_chain(root)
        self._watch(root, {"manifest.json", "features", "transitions"})
        feature_names = {f"{field.name}.npy" for field in dataclasses.fields(cache.FeatureTable)}
        transition_names = {f"{field.name}.npy" for field in dataclasses.fields(cache.TransitionTable)}
        self._watch(root / "features", feature_names)
        self._watch(root / "transitions", transition_names)

    def _drain_events(
        self,
        *,
        expected_create: tuple[int, str, Path] | None,
    ) -> None:
        if self.closed:
            raise ReplayError("replay integrity guard is closed")
        issues: list[str] = []
        expected_create_count = 0
        while True:
            try:
                payload = os.read(self._descriptor, 64 * 1024)
            except BlockingIOError:
                break
            except OSError as exc:
                raise ReplayError(f"replay integrity event read failed: {exc}") from exc
            if not payload:
                raise ReplayError("replay integrity event stream closed unexpectedly")
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < _INOTIFY_EVENT.size:
                    issues.append("truncated inotify event")
                    break
                watch, mask, _cookie, name_length = _INOTIFY_EVENT.unpack_from(payload, offset)
                offset += _INOTIFY_EVENT.size
                if len(payload) - offset < name_length:
                    issues.append("truncated inotify event name")
                    break
                name_bytes = payload[offset : offset + name_length]
                offset += name_length
                name = name_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="surrogateescape")
                if mask & _IN_Q_OVERFLOW:
                    issues.append("inotify queue overflow")
                    continue
                scope = self._scopes.get(watch)
                if scope is None:
                    issues.append(f"unknown inotify watch descriptor {watch}")
                    continue
                if mask & _IN_IGNORED:
                    issues.append(f"integrity watch was removed for {scope.path}")
                    continue
                self_event = not name and mask & (_IN_ATTRIB | _IN_DELETE_SELF | _IN_MOVE_SELF | _IN_UNMOUNT)
                named_event = bool(name) and name in scope.names and bool(mask & _INOTIFY_CONTENT_MASK)
                if self_event or named_event:
                    location = scope.path if not name else scope.path / name
                    expected_watch = None if expected_create is None else expected_create[0]
                    expected_name = None if expected_create is None else expected_create[1]
                    if watch == expected_watch and name == expected_name and mask == _IN_CREATE:
                        expected_create_count += 1
                    else:
                        issues.append(f"lineage mutation event at {location} (mask=0x{mask:08x})")
        if expected_create is not None and expected_create_count != 1:
            expected_path = expected_create[2]
            issues.append(
                "replay snapshot publication expected exactly one hard-link CREATE "
                f"at {expected_path}, observed {expected_create_count}"
            )
        if issues:
            error = ReplayError(issues[0])
            for issue in issues[1:]:
                error.add_note(issue)
            raise error

    def check(self) -> None:
        self._drain_events(expected_create=None)

    def check_expected_create(self, path: Path) -> None:
        expected = Path(os.path.abspath(path))
        if ".." in Path(path).parts or Path(os.path.normpath(expected)) != expected:
            raise ReplayError(f"expected replay snapshot create path must be normalized: {path}")
        expected_watch = self._watch(expected.parent, {expected.name})
        self._drain_events(
            expected_create=(expected_watch, expected.name, expected),
        )

    def close(self, primary: BaseException | None = None) -> None:
        if self.closed:
            return
        descriptor = self._descriptor
        try:
            _before_integrity_guard_close(descriptor)
        except BaseException as exc:
            _finish_cleanup(
                primary,
                [("shared inotify descriptor pre-close hook", exc)],
                action="close replay integrity guard",
            )
            return
        try:
            os.close(descriptor)
        except BaseException as exc:
            # This guard is Linux-only.  Once close(2) has been called, the
            # numeric descriptor may already have been released and reused
            # even when an error is reported.  Relinquish it unconditionally:
            # probing or retrying the number could inspect or close an
            # unrelated resource allocated by another thread.
            self._descriptor = -1
            self._scopes.clear()
            _finish_cleanup(primary, [("shared inotify descriptor", exc)], action="close replay integrity guard")
            return
        else:
            self._descriptor = -1
            self._scopes.clear()


class _NullIntegrityGuard:
    closed = False

    def check(self) -> None:
        return

    def close(self, _primary: BaseException | None = None) -> None:
        self.closed = True


def _freeze_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(record))


@dataclasses.dataclass(frozen=True)
class ReplaySnapshot:
    path: Path
    schema_version: int
    feature_identity: str
    stage1_config: str
    stage2_config: str
    reward_source: str
    reward_label_values: tuple[int, int, int, int]
    completion_label: int
    reward_aggregation: str
    reward_schema_version: int
    total_transitions: int
    shards: tuple[Mapping[str, Any], ...]
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shards",
            tuple(_freeze_record(record) for record in self.shards),
        )
        object.__setattr__(self, "reward_label_values", tuple(self.reward_label_values))


@dataclasses.dataclass(frozen=True)
class ReplayBatch:
    z_rl: np.ndarray
    next_z_rl: np.ndarray
    state_norm: np.ndarray
    next_state_norm: np.ndarray
    vla_reference: np.ndarray
    next_vla_reference: np.ndarray
    executed_action: np.ndarray
    bc_anchor: np.ndarray
    reward: np.ndarray
    terminal: np.ndarray
    source_global_index: np.ndarray


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"duplicate JSON key {key!r} in replay snapshot")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ReplayError(f"non-finite JSON constant {value!r} in replay snapshot")


def _before_snapshot_read(_path: Path, _descriptor: int) -> None:
    """Test hook for replacing the pathname after its inode is pinned."""


def _read_snapshot_bytes(path: Path) -> tuple[bytes, str, _FileWitness]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ReplayError(f"replay snapshot {path} cannot be opened safely without O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ReplayError(f"replay snapshot symlink is forbidden: {path}") from exc
        if exc.errno == errno.ENOENT:
            raise ReplayError(f"replay snapshot is missing: {path}") from exc
        raise ReplayError(f"replay snapshot failed to open safely: {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayError(f"replay snapshot is not a regular file: {path}")
        _before_snapshot_read(path, descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            chunks.append(chunk)
            digest.update(chunk)
        return b"".join(chunks), digest.hexdigest(), _file_witness(metadata)
    except ReplayError:
        raise
    except OSError as exc:
        raise ReplayError(f"replay snapshot read failed: {path}: {exc}") from exc
    finally:
        errors: list[tuple[str, BaseException]] = []
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append((str(path), exc))
        _finish_cleanup(
            sys.exception(),
            errors,
            action="close replay snapshot descriptor",
        )


def _parse_snapshot(payload_bytes: bytes, *, path: Path) -> dict[str, Any]:
    if b"\x00" in payload_bytes:
        raise ReplayError(f"replay snapshot contains a NUL byte: {path}")
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError(f"failed to parse replay snapshot {path}") from exc
    if type(payload) is not dict:
        raise ReplayError(f"replay snapshot must be a JSON object: {path}")
    if set(payload) != _SNAPSHOT_FIELDS:
        raise ReplayError(f"replay snapshot must contain exactly {sorted(_SNAPSHOT_FIELDS)}")
    return payload


def _lowercase_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ReplayError(f"{field} must be a lowercase 64-hex SHA-256")
    return value


def _reward_contract(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Validate the fixed Stage 2 reward interpretation in a manifest or snapshot."""
    for name in ("stage1_config", "stage2_config"):
        config_name = value.get(name)
        if type(config_name) is not str or not config_name or config_name.strip() != config_name:
            raise ReplayError(f"{field}.{name} must be a nonempty exact string")
    if value.get("reward_source") != "tristate":
        raise ReplayError(f"{field}.reward_source must be 'tristate'")
    if value.get("reward_label_values") != [-1, 0, 1, 2]:
        raise ReplayError(f"{field}.reward_label_values must be [-1, 0, 1, 2]")
    if type(value.get("completion_label")) is not int or value["completion_label"] != 2:
        raise ReplayError(f"{field}.completion_label must be exact integer 2")
    if value.get("reward_aggregation") != "sum_20_frames":
        raise ReplayError(f"{field}.reward_aggregation must be 'sum_20_frames'")
    if type(value.get("reward_schema_version")) is not int or value["reward_schema_version"] != 1:
        raise ReplayError(f"{field}.reward_schema_version must be exact integer 1")
    return {
        "stage1_config": value["stage1_config"],
        "stage2_config": value["stage2_config"],
        "reward_source": value["reward_source"],
        "reward_label_values": list(value["reward_label_values"]),
        "completion_label": value["completion_label"],
        "reward_aggregation": value["reward_aggregation"],
        "reward_schema_version": value["reward_schema_version"],
    }


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise ReplayError(f"{field} must be an exact positive integer")
    if value <= 0:
        raise ReplayError(f"{field} must be positive")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ReplayError(f"{field} must be an exact nonnegative integer")
    return value


def _validate_verified_shard(
    verification: cache.ShardVerification,
    record: Mapping[str, Any],
    *,
    feature_identity: str,
    reward_contract: Mapping[str, Any],
) -> None:
    batch_id = record["batch_id"]
    manifest = verification.manifest
    if verification.root != Path(record["root"]):
        raise ReplayError(f"cache shard real root differs for batch {batch_id}")
    if manifest.get("batch_id") != batch_id:
        raise ReplayError(f"cache batch identity mismatch for batch {batch_id}")
    if manifest.get("feature_identity") != feature_identity:
        raise ReplayError(f"feature identity mismatch for batch {batch_id}")
    if manifest.get("transition_rows") != record["transition_rows"]:
        raise ReplayError(f"transition row count mismatch for batch {batch_id}")
    if verification.manifest_sha256 != record["cache_manifest_sha256"]:
        raise ReplayError(f"cache manifest changed for batch {batch_id}")
    manifest_reward = _reward_contract(manifest, field=f"cache {batch_id}")
    if manifest_reward != dict(reward_contract):
        raise ReplayError(f"reward metadata mismatch for batch {batch_id}")
    labels_sha256 = _lowercase_sha256(
        manifest.get("tristate_labels_sha256"),
        field=f"cache {batch_id}.tristate_labels_sha256",
    )
    if labels_sha256 != record["tristate_labels_sha256"]:
        raise ReplayError(f"tristate label hash mismatch for batch {batch_id}")


def _authenticate_record_shard(
    record: Mapping[str, Any],
    *,
    feature_identity: str,
    reward_contract: Mapping[str, Any],
) -> cache.ShardVerification:
    try:
        verification = cache.authenticate_shard(Path(record["root"]))
    except cache.CacheError as exc:
        raise ReplayError(f"cache shard verification failed for batch {record['batch_id']}: {exc}") from exc
    _validate_verified_shard(
        verification,
        record,
        feature_identity=feature_identity,
        reward_contract=reward_contract,
    )
    return verification


def _normalized_snapshot_path(path: Path) -> Path:
    requested = Path(path)
    if ".." in requested.parts:
        raise ReplayError(f"replay snapshot path contains a forbidden parent component '..': {requested}")
    absolute = Path(os.path.abspath(requested))
    if absolute != Path(os.path.normpath(absolute)):
        raise ReplayError(f"replay snapshot path must be normalized: {requested}")
    return absolute


def _read_snapshot_contract(
    path: Path,
) -> tuple[ReplaySnapshot, _FileWitness, bytes]:
    snapshot_path = _normalized_snapshot_path(path)
    payload_bytes, snapshot_sha256, snapshot_witness = _read_snapshot_bytes(snapshot_path)
    payload = _parse_snapshot(payload_bytes, path=snapshot_path)

    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ReplayError(f"unsupported replay schema in {snapshot_path}")
    feature_identity = payload["feature_identity"]
    if type(feature_identity) is not str or not feature_identity:
        raise ReplayError("replay snapshot feature_identity must be a nonempty string")
    reward_contract = _reward_contract(payload, field="replay snapshot")
    if type(payload["total_transitions"]) is not int or payload["total_transitions"] < 0:
        raise ReplayError("replay snapshot total_transitions must be an exact nonnegative integer")
    records = payload["shards"]
    if type(records) is not list or not records:
        raise ReplayError(f"replay snapshot has no shards: {snapshot_path}")

    expected_start = 0
    seen_batches: set[str] = set()
    verified: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != _SHARD_RECORD_FIELDS:
            raise ReplayError(f"replay shard record {index} must contain exactly {sorted(_SHARD_RECORD_FIELDS)}")
        batch_id = record["batch_id"]
        if type(batch_id) is not str or not batch_id:
            raise ReplayError(f"replay shard record {index} batch_id must be a nonempty string")
        if batch_id in seen_batches:
            raise ReplayError(f"duplicate batch in replay snapshot: {batch_id}")
        seen_batches.add(batch_id)

        root = record["root"]
        if type(root) is not str or not root or not os.path.isabs(root) or os.path.normpath(root) != root:
            raise ReplayError(f"replay shard root must be an absolute normalized path for batch {batch_id}")
        _lowercase_sha256(record["admission_sha256"], field=f"{batch_id}.admission_sha256")
        _lowercase_sha256(
            record["cache_manifest_sha256"],
            field=f"{batch_id}.cache_manifest_sha256",
        )
        _lowercase_sha256(
            record["tristate_labels_sha256"],
            field=f"{batch_id}.tristate_labels_sha256",
        )
        rows = _positive_int(record["transition_rows"], field=f"{batch_id}.transition_rows")
        start = _nonnegative_int(record["start"], field=f"{batch_id}.start")
        end = _nonnegative_int(record["end"], field=f"{batch_id}.end")
        if start != expected_start or end != expected_start + rows:
            raise ReplayError(f"noncontiguous replay offsets for batch {batch_id}")
        expected_start = end
        verified.append(_freeze_record(record))

    if payload["total_transitions"] != expected_start:
        raise ReplayError(f"total transition count mismatch in {snapshot_path}")
    return (
        ReplaySnapshot(
            path=snapshot_path,
            schema_version=1,
            feature_identity=feature_identity,
            stage1_config=reward_contract["stage1_config"],
            stage2_config=reward_contract["stage2_config"],
            reward_source=reward_contract["reward_source"],
            reward_label_values=tuple(reward_contract["reward_label_values"]),
            completion_label=reward_contract["completion_label"],
            reward_aggregation=reward_contract["reward_aggregation"],
            reward_schema_version=reward_contract["reward_schema_version"],
            total_transitions=expected_start,
            shards=tuple(verified),
            sha256=snapshot_sha256,
        ),
        snapshot_witness,
        payload_bytes,
    )


def _parse_snapshot_contract(
    path: Path,
) -> tuple[ReplaySnapshot, _FileWitness]:
    snapshot, witness, _payload = _read_snapshot_contract(path)
    return snapshot, witness


def _snapshot_reward_contract(snapshot: ReplaySnapshot) -> dict[str, Any]:
    return {
        "stage1_config": snapshot.stage1_config,
        "stage2_config": snapshot.stage2_config,
        "reward_source": snapshot.reward_source,
        "reward_label_values": list(snapshot.reward_label_values),
        "completion_label": snapshot.completion_label,
        "reward_aggregation": snapshot.reward_aggregation,
        "reward_schema_version": snapshot.reward_schema_version,
    }


def _verification_contract(
    verification: cache.ShardVerification,
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple((item.path, item.trailing_shape, item.dtype) for item in verification.array_contract)


def _require_verification_capability(
    value: object,
    *,
    field: str,
) -> cache.ShardVerification:
    try:
        return cache._require_verification(value)  # noqa: SLF001 - consume the package-private sealed capability.
    except TypeError as exc:
        raise ReplayError(f"{field} must be an authenticated ShardVerification") from exc


def _require_verification_tuple(
    value: object,
    *,
    field: str,
) -> tuple[cache.ShardVerification, ...]:
    if type(value) is not tuple:
        raise ReplayError(f"{field} must be an exact tuple of authenticated ShardVerification values")
    return tuple(_require_verification_capability(item, field=f"{field}[{index}]") for index, item in enumerate(value))


def _relinquish_integrity_guard(
    guard: _IntegrityGuard,
    primary: BaseException | None,
) -> None:
    """Close an untransferred guard without ever replacing an active primary."""

    errors: list[tuple[str, BaseException]] = []
    try:
        guard.close(primary)
    except BaseException as exc:
        errors.append(("guard close", exc))
    if not guard.closed:
        # A pre-close failure is retryable while a ReplayBuffer owns the
        # guard.  Acquisition failure has no future owner, so invoke close(2)
        # directly and relinquish the descriptor number unconditionally.
        descriptor = guard._descriptor  # noqa: SLF001 - ownership recovery inside this module.
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append(("guard emergency close", exc))
        finally:
            guard._descriptor = -1  # noqa: SLF001 - never retry a possibly reused descriptor number.
            guard._scopes.clear()  # noqa: SLF001 - the descriptor no longer owns these watches.
    _finish_cleanup(
        primary,
        errors,
        action="close untransferred replay integrity guard",
    )


def _arm_snapshot_catalog(
    guard: _IntegrityGuard,
    snapshot: ReplaySnapshot,
    verifications: tuple[cache.ShardVerification, ...],
) -> None:
    armed: set[Path] = set()
    for record in snapshot.shards:
        root = Path(record["root"])
        guard.arm_shard(root)
        armed.add(root)
    for verification in verifications:
        if verification.root not in armed:
            guard.arm_shard(verification.root)
            armed.add(verification.root)


def _quick_validate_verification_catalog(
    snapshot: ReplaySnapshot,
    verifications: tuple[cache.ShardVerification, ...],
    *,
    guard: _IntegrityGuard,
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    if len(verifications) != len(snapshot.shards):
        raise ReplayError(
            "replay shard verification count mismatch: "
            f"snapshot has {len(snapshot.shards)}, supplied {len(verifications)}"
        )
    expected_contract: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
    for index, (record, verification) in enumerate(zip(snapshot.shards, verifications, strict=True)):
        guard.check()
        try:
            cache.verify_verification(verification, full=False)
        except (cache.CacheError, OSError) as exc:
            raise ReplayError(f"cache shard quick verification failed for catalog index {index}: {exc}") from exc
        _validate_verified_shard(
            verification,
            record,
            feature_identity=snapshot.feature_identity,
            reward_contract=_snapshot_reward_contract(snapshot),
        )
        contract = _verification_contract(verification)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ReplayError(f"cache shard trailing shape or dtype mismatch for batch {record['batch_id']}")
        guard.check()
    assert expected_contract is not None
    return expected_contract


def _assert_snapshot_binding_unchanged(
    expected_snapshot: ReplaySnapshot,
    expected_witness: _FileWitness,
    expected_payload: bytes,
    *,
    context: str,
) -> None:
    observed, observed_witness, observed_payload = _read_snapshot_contract(expected_snapshot.path)
    if (
        observed_payload != expected_payload
        or observed.sha256 != expected_snapshot.sha256
        or observed_witness != expected_witness
        or observed != expected_snapshot
    ):
        raise ReplayError(f"replay snapshot binding changed {context}: {expected_snapshot.path}")


def _authenticate_snapshot_shards(
    snapshot: ReplaySnapshot,
    *,
    guard: _IntegrityGuard | None,
    require_common_contract: bool,
) -> tuple[cache.ShardVerification, ...]:
    verifications: list[cache.ShardVerification] = []
    expected_contract: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
    for record in snapshot.shards:
        if guard is not None:
            guard.check()
        verification = _authenticate_record_shard(
            record,
            feature_identity=snapshot.feature_identity,
            reward_contract=_snapshot_reward_contract(snapshot),
        )
        contract = _verification_contract(verification)
        if expected_contract is None:
            expected_contract = contract
        elif require_common_contract and contract != expected_contract:
            raise ReplayError(f"cache shard trailing shape or dtype mismatch for batch {record['batch_id']}")
        verifications.append(verification)
        if guard is not None:
            guard.check()
    return tuple(verifications)


def _open_snapshot_catalog(
    path: Path,
    *,
    guard: _IntegrityGuard | None = None,
    require_common_contract: bool = False,
) -> tuple[ReplaySnapshot, tuple[cache.ShardVerification, ...], _FileWitness]:
    snapshot_path = _normalized_snapshot_path(path)
    if guard is not None:
        guard.arm_snapshot(snapshot_path)
        guard.check()
    snapshot, snapshot_witness = _parse_snapshot_contract(snapshot_path)
    if guard is not None:
        current = os.stat(snapshot.path, follow_symlinks=False)
        if _file_witness(current) != snapshot_witness:
            raise ReplayError(f"replay snapshot path changed while opening: {snapshot.path}")
        for record in snapshot.shards:
            guard.arm_shard(Path(record["root"]))
        guard.check()
    verifications = _authenticate_snapshot_shards(
        snapshot,
        guard=guard,
        require_common_contract=require_common_contract,
    )
    if guard is not None:
        current = os.stat(snapshot.path, follow_symlinks=False)
        if _file_witness(current) != snapshot_witness:
            raise ReplayError(f"replay snapshot path changed while authenticating: {snapshot.path}")
        guard.check()
    return snapshot, verifications, snapshot_witness


def open_snapshot(path: Path) -> ReplaySnapshot:
    snapshot, _, _ = _open_snapshot_catalog(path)
    return snapshot


def create_snapshot(
    destination: Path,
    *,
    previous: ReplaySnapshot | None,
    new_shard: cache.OpenShard,
    admission_sha256: str,
) -> ReplaySnapshot:
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    _lowercase_sha256(admission_sha256, field="admission_sha256")
    if previous is not None and not isinstance(previous, ReplaySnapshot):
        raise ReplayError("previous must be a ReplaySnapshot or None")
    if not isinstance(new_shard, cache.OpenShard):
        raise ReplayError("new_shard must be an OpenShard")

    previous_verified: ReplaySnapshot | None = None
    if previous is not None:
        try:
            previous_verified = open_snapshot(previous.path)
        except ReplayError as exc:
            raise ReplayError(f"previous replay snapshot failed verification: {exc}") from exc
        if previous_verified != previous:
            raise ReplayError("previous replay snapshot differs from the supplied verified object")

    try:
        reopened = cache.authenticate_shard(new_shard.root)
    except cache.CacheError as exc:
        raise ReplayError(f"new cache shard failed verification: {exc}") from exc
    if (
        reopened.root != new_shard.root
        or reopened.manifest != new_shard.manifest
        or reopened.manifest_sha256 != new_shard.manifest_sha256
    ):
        raise ReplayError("supplied shard differs from the reopened verified shard")

    manifest = reopened.manifest
    feature_identity = manifest.get("feature_identity")
    batch_id = manifest.get("batch_id")
    rows = manifest.get("transition_rows")
    reward_contract = _reward_contract(manifest, field="new cache shard")
    labels_sha256 = _lowercase_sha256(
        manifest.get("tristate_labels_sha256"),
        field="new cache shard.tristate_labels_sha256",
    )
    if type(feature_identity) is not str or not feature_identity:
        raise ReplayError("new shard feature identity must be a nonempty string")
    if type(batch_id) is not str or not batch_id:
        raise ReplayError("new shard batch id must be a nonempty string")
    rows = _positive_int(rows, field=f"{batch_id}.transition_rows")

    previous_shards = [] if previous_verified is None else [dict(record) for record in previous_verified.shards]
    if previous_verified is not None and previous_verified.feature_identity != feature_identity:
        raise ReplayError("feature identity differs from previous replay snapshot")
    if previous_verified is not None and _snapshot_reward_contract(previous_verified) != reward_contract:
        raise ReplayError("reward metadata differs from previous replay snapshot")
    if any(record["batch_id"] == batch_id for record in previous_shards):
        raise ReplayError(f"duplicate batch in replay snapshot: {batch_id}")

    start = 0 if previous_verified is None else previous_verified.total_transitions
    new_record = {
        "batch_id": batch_id,
        "root": str(reopened.root),
        "admission_sha256": admission_sha256,
        "cache_manifest_sha256": reopened.manifest_sha256,
        "tristate_labels_sha256": labels_sha256,
        "transition_rows": rows,
        "start": start,
        "end": start + rows,
    }
    payload_shards = [dict(record) for record in previous_shards]
    payload_shards.append(dict(new_record))
    payload = {
        "schema_version": 1,
        "feature_identity": feature_identity,
        **reward_contract,
        "total_transitions": start + rows,
        "shards": payload_shards,
    }
    try:
        identity.atomic_write_json(destination, payload)
    except FileExistsError:
        raise
    except Exception as exc:
        raise ReplayError(f"failed to publish replay snapshot {destination}: {exc}") from exc
    return open_snapshot(destination)


def _note_unverified_destination_state(
    primary: BaseException,
    destination: Path,
    *,
    publisher_returned: bool,
) -> None:
    if not publisher_returned and not os.path.lexists(destination):
        return
    note = (
        f"replay snapshot destination may already be published or occupied at {destination}; "
        "its contents are not verified and must be inspected explicitly"
    )
    if note not in getattr(primary, "__notes__", ()):
        primary.add_note(note)


def _existing_destination_error(destination: Path) -> FileExistsError:
    error = FileExistsError(destination)
    _note_unverified_destination_state(
        error,
        destination,
        publisher_returned=False,
    )
    return error


def create_snapshot_from_verifications(
    destination: Path,
    *,
    previous: ReplaySnapshot | None,
    previous_verifications: tuple[cache.ShardVerification, ...],
    new_verification: cache.ShardVerification,
    admission_sha256: str,
) -> ReplaySnapshot:
    """Append an authenticated shard without repeating payload hashes."""

    destination = _normalized_snapshot_path(destination)
    if os.path.lexists(destination):
        raise _existing_destination_error(destination)
    _lowercase_sha256(admission_sha256, field="admission_sha256")
    if previous is not None and not isinstance(previous, ReplaySnapshot):
        raise ReplayError("previous must be a ReplaySnapshot or None")
    previous_capabilities = _require_verification_tuple(
        previous_verifications,
        field="previous_verifications",
    )
    new_capability = _require_verification_capability(
        new_verification,
        field="new_verification",
    )
    if previous is None and previous_capabilities:
        raise ReplayError("replay shard verification count mismatch: no previous snapshot requires an empty tuple")

    publication_completed = False
    guard = _IntegrityGuard.open()
    try:
        previous_verified: ReplaySnapshot | None = None
        previous_contract: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
        previous_witness: _FileWitness | None = None
        previous_payload: bytes | None = None
        if previous is not None:
            previous_path = _normalized_snapshot_path(previous.path)
            guard.arm_snapshot(previous_path)
            guard.check()
            previous_verified, previous_witness, previous_payload = _read_snapshot_contract(previous_path)
            if previous_verified != previous:
                raise ReplayError("previous replay snapshot differs from the supplied verified object")
            if len(previous_capabilities) != len(previous_verified.shards):
                raise ReplayError(
                    "replay shard verification count mismatch: "
                    f"snapshot has {len(previous_verified.shards)}, "
                    f"supplied {len(previous_capabilities)}"
                )
            _arm_snapshot_catalog(
                guard,
                previous_verified,
                previous_capabilities,
            )
        guard.arm_shard(new_capability.root)
        guard.check()

        if previous_verified is not None:
            previous_contract = _quick_validate_verification_catalog(
                previous_verified,
                previous_capabilities,
                guard=guard,
            )

        guard.check()
        try:
            cache.verify_verification(new_capability, full=False)
        except (cache.CacheError, OSError) as exc:
            raise ReplayError(f"new cache shard quick verification failed: {exc}") from exc
        manifest = new_capability.manifest
        feature_identity = manifest.get("feature_identity")
        batch_id = manifest.get("batch_id")
        rows = manifest.get("transition_rows")
        reward_contract = _reward_contract(manifest, field="new cache shard")
        labels_sha256 = _lowercase_sha256(
            manifest.get("tristate_labels_sha256"),
            field="new cache shard.tristate_labels_sha256",
        )
        if type(feature_identity) is not str or not feature_identity:
            raise ReplayError("new shard feature identity must be a nonempty string")
        if type(batch_id) is not str or not batch_id:
            raise ReplayError("new shard batch id must be a nonempty string")
        rows = _positive_int(rows, field=f"{batch_id}.transition_rows")
        if rows != new_capability.transition_rows:
            raise ReplayError(f"transition row count mismatch for batch {batch_id}")
        _lowercase_sha256(
            new_capability.manifest_sha256,
            field=f"{batch_id}.cache_manifest_sha256",
        )
        new_contract = _verification_contract(new_capability)
        if previous_contract is not None and new_contract != previous_contract:
            raise ReplayError(f"cache shard trailing shape or dtype mismatch for batch {batch_id}")
        guard.check()

        previous_shards = [] if previous_verified is None else [dict(record) for record in previous_verified.shards]
        if previous_verified is not None and previous_verified.feature_identity != feature_identity:
            raise ReplayError("feature identity differs from previous replay snapshot")
        if previous_verified is not None and _snapshot_reward_contract(previous_verified) != reward_contract:
            raise ReplayError("reward metadata differs from previous replay snapshot")
        if any(record["batch_id"] == batch_id for record in previous_shards):
            raise ReplayError(f"duplicate batch in replay snapshot: {batch_id}")
        if previous_verified is not None:
            assert previous_witness is not None
            assert previous_payload is not None
            _assert_snapshot_binding_unchanged(
                previous_verified,
                previous_witness,
                previous_payload,
                context="while appending an authenticated shard",
            )
        guard.check()

        start = 0 if previous_verified is None else previous_verified.total_transitions
        payload_shards = [dict(record) for record in previous_shards]
        payload_shards.append(
            {
                "batch_id": batch_id,
                "root": str(new_capability.root),
                "admission_sha256": admission_sha256,
                "cache_manifest_sha256": new_capability.manifest_sha256,
                "tristate_labels_sha256": labels_sha256,
                "transition_rows": rows,
                "start": start,
                "end": start + rows,
            }
        )
        payload = {
            "schema_version": 1,
            "feature_identity": feature_identity,
            **reward_contract,
            "total_transitions": start + rows,
            "shards": payload_shards,
        }

        destination.parent.mkdir(parents=True, exist_ok=True)
        # The shard guards can already cover an ancestor of the destination.
        # Drain our own parent creation before adding the target filename to
        # that watch; old events must never inherit a newly-added name.
        guard.check()
        if os.path.lexists(destination):
            raise _existing_destination_error(destination)
        guard.arm_snapshot(destination)
        guard.check()
        try:
            identity.atomic_write_json(destination, payload)
        except FileExistsError:
            raise
        except Exception as exc:
            raise ReplayError(f"failed to publish replay snapshot {destination}: {exc}") from exc
        publication_completed = True
        guard.check_expected_create(destination)
        created, created_witness, created_payload = _read_snapshot_contract(destination)
        expected_payload = identity.canonical_json_bytes(payload)
        if created_payload != expected_payload:
            raise ReplayError(f"published replay snapshot differs from the requested payload: {destination}")
        if tuple(created.shards) != tuple(_freeze_record(record) for record in payload_shards):
            raise ReplayError(f"published replay snapshot catalog differs from the requested payload: {destination}")
        guard.check()
        _assert_snapshot_binding_unchanged(
            created,
            created_witness,
            created_payload,
            context="after authenticated publication",
        )
        guard.check()
        all_capabilities = (*previous_capabilities, new_capability)
        _quick_validate_verification_catalog(
            created,
            all_capabilities,
            guard=guard,
        )
        guard.check()
        _assert_snapshot_binding_unchanged(
            created,
            created_witness,
            created_payload,
            context="after post-publication catalog verification",
        )
        guard.check()
        return created
    finally:
        primary = sys.exception()
        if primary is not None:
            _note_unverified_destination_state(
                primary,
                destination,
                publisher_returned=publication_completed,
            )
        _relinquish_integrity_guard(guard, primary)


def _array_contract(shard: cache.OpenShard) -> tuple[tuple[str, tuple[int, ...], np.dtype], ...]:
    values: list[tuple[str, tuple[int, ...], np.dtype]] = []
    for table_name, table in (("features", shard.features), ("transitions", shard.transitions)):
        for field in dataclasses.fields(table):
            value = getattr(table, field.name)
            values.append((f"{table_name}.{field.name}", value.shape[1:], value.dtype))
    return tuple(values)


def _snapshot_path_witness(path: Path) -> _FileWitness:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ReplayError(f"replay snapshot path check failed: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ReplayError(f"replay snapshot is not a regular file: {path}")
    return _file_witness(metadata)


class ReplayBuffer:
    def __init__(self, snapshot: ReplaySnapshot, shards: tuple[cache.OpenShard, ...]):
        if not hasattr(snapshot, "total_transitions") or not hasattr(snapshot, "shards"):
            raise ReplayError("snapshot must provide total_transitions and shards")
        if type(shards) is not tuple or not shards or not all(isinstance(shard, cache.OpenShard) for shard in shards):
            raise ReplayError("shards must be a nonempty tuple of OpenShard values")
        if len(shards) != len(snapshot.shards):
            raise ReplayError("legacy ReplayBuffer shard count must match the snapshot")
        verifications = tuple(shard.verification for shard in shards)
        snapshot_path = getattr(snapshot, "path", None)
        snapshot_witness = _snapshot_path_witness(Path(snapshot_path)) if snapshot_path is not None else None
        self._initialize(
            snapshot=snapshot,
            verifications=verifications,
            snapshot_witness=snapshot_witness,
            max_open_shards=max(1, len(shards)),
            guard=_NullIntegrityGuard(),
            initial_shards=OrderedDict(enumerate(shards)),
        )

    def _initialize(
        self,
        *,
        snapshot: ReplaySnapshot,
        verifications: tuple[cache.ShardVerification, ...],
        snapshot_witness: _FileWitness | None,
        max_open_shards: int,
        guard: _IntegrityGuard | _NullIntegrityGuard,
        initial_shards: OrderedDict[int, cache.OpenShard] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.shard_verifications = verifications
        self._starts = tuple(int(record["start"]) for record in snapshot.shards)
        self._ends = np.asarray(tuple(int(record["end"]) for record in snapshot.shards), dtype=np.int64)
        self._ends.setflags(write=False)
        self._total_transitions = int(snapshot.total_transitions)
        self._snapshot_witness = snapshot_witness
        self._max_open_shards = max_open_shards
        self._guard = guard
        self._lru = OrderedDict() if initial_shards is None else initial_shards
        self._lock = threading.RLock()
        self._closed = False
        self._poison: ReplayError | None = None
        self._lru_hits = 0
        self._lru_misses = 0
        self._lru_evictions = 0
        self._contracts = {item.path: item for item in verifications[0].array_contract}

    @classmethod
    def open(
        cls,
        snapshot_path: Path,
        *,
        max_open_shards: int = 32,
    ) -> ReplayBuffer:
        if (
            isinstance(max_open_shards, bool | np.bool_)
            or not isinstance(max_open_shards, int | np.integer)
            or int(max_open_shards) <= 0
        ):
            raise ValueError("max_open_shards must be an exact positive integer")
        guard = _IntegrityGuard.open()
        try:
            snapshot, verifications, snapshot_witness = _open_snapshot_catalog(
                snapshot_path,
                guard=guard,
                require_common_contract=True,
            )
            value = cls.__new__(cls)
            value._initialize(  # noqa: SLF001 - alternate constructor initializes a fresh instance.
                snapshot=snapshot,
                verifications=verifications,
                snapshot_witness=snapshot_witness,
                max_open_shards=int(max_open_shards),
                guard=guard,
            )
            return value
        except BaseException:
            guard.close(sys.exception())
            raise

    @classmethod
    def open_from_verifications(
        cls,
        snapshot_path: Path,
        shard_verifications: tuple[cache.ShardVerification, ...],
        *,
        max_open_shards: int = 32,
    ) -> ReplayBuffer:
        """Acquire a runtime lease from sealed shard capabilities without rehashing."""

        if (
            isinstance(max_open_shards, bool | np.bool_)
            or not isinstance(max_open_shards, int | np.integer)
            or int(max_open_shards) <= 0
        ):
            raise ValueError("max_open_shards must be an exact positive integer")
        verifications = _require_verification_tuple(
            shard_verifications,
            field="shard_verifications",
        )
        snapshot_path = _normalized_snapshot_path(snapshot_path)
        guard = _IntegrityGuard.open()
        try:
            guard.arm_snapshot(snapshot_path)
            guard.check()
            snapshot, snapshot_witness, snapshot_payload = _read_snapshot_contract(snapshot_path)
            if len(verifications) != len(snapshot.shards):
                raise ReplayError(
                    "replay shard verification count mismatch: "
                    f"snapshot has {len(snapshot.shards)}, supplied {len(verifications)}"
                )
            _arm_snapshot_catalog(guard, snapshot, verifications)
            guard.check()
            _quick_validate_verification_catalog(
                snapshot,
                verifications,
                guard=guard,
            )
            _assert_snapshot_binding_unchanged(
                snapshot,
                snapshot_witness,
                snapshot_payload,
                context="while acquiring authenticated runtime capabilities",
            )
            guard.check()
            value = cls.__new__(cls)
            value._initialize(  # noqa: SLF001 - alternate constructor initializes a fresh instance.
                snapshot=snapshot,
                verifications=verifications,
                snapshot_witness=snapshot_witness,
                max_open_shards=int(max_open_shards),
                guard=guard,
            )
            return value
        except BaseException:
            primary = sys.exception()
            assert primary is not None
            _relinquish_integrity_guard(guard, primary)
            raise

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def max_open_shards(self) -> int:
        return self._max_open_shards

    @property
    def open_shard_count(self) -> int:
        with self._lock:
            return len(self._lru)

    @property
    def lru_hits(self) -> int:
        return self._lru_hits

    @property
    def lru_misses(self) -> int:
        return self._lru_misses

    @property
    def lru_evictions(self) -> int:
        return self._lru_evictions

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_usable(self) -> None:
        if self._closed:
            raise ReplayError("replay buffer is closed")
        if self._poison is not None:
            raise ReplayError(f"replay buffer is poisoned: {self._poison}") from self._poison

    def _poison_with(self, exc: BaseException, *, context: str) -> ReplayError:
        if self._poison is None:
            self._poison = ReplayError(f"{context}: {exc}")
            self._poison.__cause__ = exc
        return self._poison

    def _check_snapshot_path(self) -> None:
        if self._snapshot_witness is None:
            return
        try:
            current = os.stat(self.snapshot.path, follow_symlinks=False)
        except OSError as exc:
            raise ReplayError(f"replay snapshot path check failed: {self.snapshot.path}: {exc}") from exc
        if not stat.S_ISREG(current.st_mode) or _file_witness(current) != self._snapshot_witness:
            raise ReplayError(f"replay snapshot binding changed: {self.snapshot.path}")

    def _check_runtime_guard(self) -> None:
        self._guard.check()
        self._check_snapshot_path()

    def _raise_integrity(self, exc: BaseException, *, context: str) -> None:
        poison = self._poison_with(exc, context=context)
        raise poison from exc

    def _verify_shard_quick(self, shard_id: int) -> None:
        try:
            cache.verify_verification(self.shard_verifications[shard_id], full=False)
        except (cache.CacheError, OSError) as exc:
            self._raise_integrity(exc, context=f"replay shard {shard_id} quick verification failed")

    def _evict_one(self) -> None:
        shard_id, shard = next(iter(self._lru.items()))
        try:
            shard.close()
        except BaseException as exc:
            if shard.closed:
                del self._lru[shard_id]
                self._lru_evictions += 1
            self._raise_integrity(exc, context=f"replay LRU eviction failed for shard {shard_id}")
        if not shard.closed:
            self._raise_integrity(
                ReplayError("OpenShard remained open without reporting a close error"),
                context=f"replay LRU eviction failed for shard {shard_id}",
            )
        del self._lru[shard_id]
        self._lru_evictions += 1

    def _acquire_shard(self, shard_id: int) -> cache.OpenShard:
        shard = self._lru.pop(shard_id, None)
        if shard is not None:
            self._lru_hits += 1
            self._lru[shard_id] = shard
            self._verify_shard_quick(shard_id)
            return shard
        self._lru_misses += 1
        while len(self._lru) >= self._max_open_shards:
            self._evict_one()
        try:
            shard = cache.open_from_verification(self.shard_verifications[shard_id])
            self._guard.check()
        except BaseException as exc:
            if "shard" in locals():
                try:
                    shard.close()
                except BaseException as cleanup:
                    exc.add_note(f"new replay shard cleanup failed: {cleanup}")
            self._raise_integrity(exc, context=f"replay LRU open failed for shard {shard_id}")
        self._lru[shard_id] = shard
        return shard

    def _contract(self, path: str) -> cache.ShardArrayContract:
        try:
            return self._contracts[path]
        except KeyError as exc:
            raise ReplayError(f"replay array contract is missing {path}") from exc

    @staticmethod
    def _contract_dtype(dtype: str) -> np.dtype:
        if dtype == "bfloat16":
            return np.dtype(ml_dtypes.bfloat16)
        return np.dtype(dtype)

    def _allocate(self, path: str, batch_size: int, *, zero: bool = False) -> np.ndarray:
        contract = self._contract(path)
        shape = (batch_size, *contract.trailing_shape)
        dtype = self._contract_dtype(contract.dtype)
        return np.zeros(shape, dtype=dtype) if zero else np.empty(shape, dtype=dtype)

    def sample_indices(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        if (
            isinstance(batch_size, bool | np.bool_)
            or not isinstance(batch_size, int | np.integer)
            or int(batch_size) <= 0
        ):
            raise ValueError("batch_size must be an exact positive integer")
        with self._lock:
            self._ensure_usable()
            try:
                self._check_runtime_guard()
                result = rng.integers(
                    0,
                    self.total_transitions,
                    size=int(batch_size),
                    dtype=np.int64,
                )
                self._check_runtime_guard()
                return result
            except ReplayError as exc:
                self._raise_integrity(exc, context="replay sampling integrity check failed")

    def gather(self, global_indices: np.ndarray) -> ReplayBatch:
        try:
            requested = np.asarray(global_indices)
        except (TypeError, ValueError) as exc:
            raise ValueError("global replay indices must be a rank 1 signed integer array") from exc
        if requested.ndim != 1 or requested.dtype.kind != "i":
            raise ValueError("global replay indices must be a rank 1 signed integer array")
        indices = requested.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.total_transitions):
            raise IndexError("global replay index is out of range")

        with self._lock:
            self._ensure_usable()
            try:
                self._check_runtime_guard()
                shard_ids = np.searchsorted(self._ends, indices, side="right")
                batch_size = indices.size
                z_rl = self._allocate("features/z_rl.npy", batch_size)
                next_z_rl = self._allocate("features/z_rl.npy", batch_size, zero=True)
                state_norm = self._allocate("features/state_norm.npy", batch_size)
                next_state_norm = self._allocate("features/state_norm.npy", batch_size, zero=True)
                vla_reference = self._allocate("features/vla_reference.npy", batch_size)
                next_vla_reference = self._allocate(
                    "features/vla_reference.npy",
                    batch_size,
                    zero=True,
                )
                executed_action = self._allocate("transitions/executed_action.npy", batch_size)
                bc_anchor = self._allocate("transitions/bc_anchor.npy", batch_size)
                reward = self._allocate("transitions/reward.npy", batch_size)
                terminal = self._allocate("transitions/terminal.npy", batch_size)

                requested_shards = list(dict.fromkeys(int(value) for value in shard_ids))
                hit_shards = [shard_id for shard_id in requested_shards if shard_id in self._lru]
                miss_shards = [shard_id for shard_id in requested_shards if shard_id not in self._lru]
                used_shards: list[int] = []
                for shard_id in (*hit_shards, *miss_shards):
                    shard = self._acquire_shard(shard_id)
                    used_shards.append(shard_id)
                    output_rows = np.flatnonzero(shard_ids == shard_id)
                    start = self._starts[shard_id]
                    local_rows = indices[output_rows] - start
                    transition_table = shard.transitions
                    current_rows = transition_table.current_feature_row[local_rows]
                    next_rows = transition_table.next_feature_row[local_rows]

                    z_rl[output_rows] = shard.features.z_rl[current_rows]
                    state_norm[output_rows] = shard.features.state_norm[current_rows]
                    vla_reference[output_rows] = shard.features.vla_reference[current_rows]
                    executed_action[output_rows] = transition_table.executed_action[local_rows]
                    bc_anchor[output_rows] = transition_table.bc_anchor[local_rows]
                    reward[output_rows] = transition_table.reward[local_rows]
                    terminal[output_rows] = transition_table.terminal[local_rows]

                    nonterminal = next_rows >= 0
                    if np.any(nonterminal):
                        output_nonterminal = output_rows[nonterminal]
                        feature_next = next_rows[nonterminal]
                        next_z_rl[output_nonterminal] = shard.features.z_rl[feature_next]
                        next_state_norm[output_nonterminal] = shard.features.state_norm[feature_next]
                        next_vla_reference[output_nonterminal] = shard.features.vla_reference[feature_next]
                    self._guard.check()

                for shard_id in used_shards:
                    self._verify_shard_quick(shard_id)
                self._check_runtime_guard()
                return ReplayBatch(
                    z_rl=np.ascontiguousarray(z_rl),
                    next_z_rl=np.ascontiguousarray(next_z_rl),
                    state_norm=np.ascontiguousarray(state_norm),
                    next_state_norm=np.ascontiguousarray(next_state_norm),
                    vla_reference=np.ascontiguousarray(vla_reference),
                    next_vla_reference=np.ascontiguousarray(next_vla_reference),
                    executed_action=np.ascontiguousarray(executed_action),
                    bc_anchor=np.ascontiguousarray(bc_anchor),
                    reward=np.ascontiguousarray(reward),
                    terminal=np.ascontiguousarray(terminal),
                    source_global_index=np.ascontiguousarray(indices.copy()),
                )
            except ReplayError as exc:
                if exc is self._poison:
                    raise
                self._raise_integrity(exc, context="replay gather integrity check failed")
            except BaseException as exc:
                self._raise_integrity(exc, context="replay gather failed")

    def _quiesce_locked(self, primary: BaseException | None) -> None:
        errors: list[tuple[str, BaseException]] = []
        for shard_id, shard in tuple(self._lru.items()):
            try:
                shard.close()
            except BaseException as exc:
                errors.append((f"shard {shard_id}", exc))
            if shard.closed:
                self._lru.pop(shard_id, None)
            else:
                errors.append((f"shard {shard_id}", ReplayError("shard remained open after close")))
        if errors:
            self._poison_with(errors[0][1], context="replay quiesce failed")
        _finish_cleanup(primary, errors, action="quiesce replay buffer")

    def quiesce(self) -> None:
        with self._lock:
            if self._closed:
                raise ReplayError("replay buffer is closed")
            existing_poison = self._poison
            self._quiesce_locked(sys.exception())
            if existing_poison is not None:
                raise ReplayError(f"replay buffer is poisoned: {existing_poison}") from existing_poison

    def _verify_snapshot_full(self) -> None:
        payload, sha256, witness = _read_snapshot_bytes(self.snapshot.path)
        if sha256 != self.snapshot.sha256 or witness != self._snapshot_witness:
            raise ReplayError(f"replay snapshot changed: {self.snapshot.path}")
        parsed = _parse_snapshot(payload, path=self.snapshot.path)
        if parsed["schema_version"] != self.snapshot.schema_version:
            raise ReplayError(f"replay snapshot schema changed: {self.snapshot.path}")

    def verify_integrity(self, *, full: bool = False) -> None:
        if type(full) is not bool:
            raise TypeError("full must be an exact bool")
        with self._lock:
            self._ensure_usable()
            try:
                if full:
                    self._quiesce_locked(None)
                self._check_runtime_guard()
                if full:
                    self._verify_snapshot_full()
                for verification in self.shard_verifications:
                    cache.verify_verification(verification, full=full)
                self._check_runtime_guard()
            except BaseException as exc:
                if self._poison is exc:
                    raise
                self._raise_integrity(exc, context="replay integrity verification failed")

    def _close(self, primary: BaseException | None) -> None:
        with self._lock:
            if self._closed:
                return
            errors: list[tuple[str, BaseException]] = []
            if not self._guard.closed:
                try:
                    self._guard.check()
                except BaseException as exc:
                    errors.append(("final integrity event drain", exc))
                    self._poison_with(exc, context="replay close integrity check failed")
            for shard_id, shard in tuple(self._lru.items()):
                try:
                    shard.close()
                except BaseException as exc:
                    errors.append((f"shard {shard_id}", exc))
                if shard.closed:
                    self._lru.pop(shard_id, None)
                else:
                    errors.append((f"shard {shard_id}", ReplayError("shard remained open after close")))
            if not self._lru and not self._guard.closed:
                try:
                    self._guard.check()
                except BaseException as exc:
                    errors.append(("post-shard final integrity event drain", exc))
                    self._poison_with(exc, context="replay close integrity check failed")
                try:
                    self._guard.close()
                except BaseException as exc:
                    errors.append(("shared integrity guard", exc))
            self._closed = not self._lru and self._guard.closed
            if errors and self._poison is None:
                self._poison_with(errors[0][1], context="replay close failed")
            _finish_cleanup(primary, errors, action="close replay buffer")

    def close(self) -> None:
        self._close(sys.exception())

    def __enter__(self) -> ReplayBuffer:
        with self._lock:
            self._ensure_usable()
            return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        exception: BaseException | None,
        _traceback: object,
    ) -> bool:
        self._close(exception)
        return False


def as_jax_transition_batch(batch: ReplayBatch):
    import jax.numpy as jnp  # noqa: PLC0415

    from openpi.training.rl_token.stage2 import td3 as rlt_td3  # noqa: PLC0415

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
