from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
from typing import Any
import weakref

from flax import nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import ml_dtypes
import numpy as np
import torch

from openpi.models import model as model_api
from openpi.shared import nnx_utils
from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import cache
from openpi.training.rl_token.stage2 import feature_identity
from openpi.training.rl_token.stage2 import transitions
import openpi.transforms as transforms

DEFAULT_PROMPT = "fold clothes"
SNAPSHOT_READ_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
_MIN_PINNED_DESCRIPTOR = 3
_FEATURE_ACTION_HORIZON = 50
_FEATURE_ACTION_DIM = 32
_FEATURE_REFERENCE_HORIZON = 20
_FEATURE_REFERENCE_DIM = 16
_FEATURE_STATE_DIM = 16
_FEATURE_Z_DIM = 2048
_UNMANIFESTED_SNAPSHOT_FILES = frozenset(
    {
        "READY",
        "migration_manifest.json",
        "meta/tristate_labels.json",
    }
)
_REQUIRED_LEROBOT_MANIFEST_FILES = frozenset(
    {
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/info.json",
        "meta/tasks.jsonl",
    }
)


class BatchSnapshotError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True, order=True)
class _StatWitness:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclasses.dataclass(frozen=True, order=True)
class _FileWitness:
    relative_path: str
    stat: _StatWitness
    sha256: str


@dataclasses.dataclass(frozen=True)
class ValidatedBatchSnapshotWitness:
    canonical_root: str
    directories: tuple[tuple[str, _StatWitness], ...]
    files: tuple[_FileWitness, ...]


@dataclasses.dataclass(frozen=True, order=True)
class _ManifestFile:
    relative_path: str
    size: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class _ParquetIdentity:
    size: int
    sha256: str
    device: int
    inode: int


def _after_snapshot_file_open(_path: Path, _descriptor: int) -> None:
    """Test hook for exercising replacement after a file descriptor is pinned."""


def _stat_witness(metadata: os.stat_result) -> _StatWitness:
    return _StatWitness(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _root_stat_witness(metadata: os.stat_result) -> _StatWitness:
    """Ignore only rename-driven root ctime; retain content-sensitive size/mtime."""
    witness = _stat_witness(metadata)
    return dataclasses.replace(witness, ctime_ns=0)


def _canonical_local_root(batch_root: Path) -> Path:
    root = Path(batch_root)
    if not root.is_absolute() or ".." in root.parts:
        raise BatchSnapshotError(f"local batch root must be an absolute canonical path: {root}")
    normalized = Path(os.path.normpath(os.fspath(root)))
    if normalized != root:
        raise BatchSnapshotError(f"local batch root must be canonical: {root}")
    if getattr(os, "O_DIRECTORY", None) is None or getattr(os, "O_NOFOLLOW", None) is None:
        raise BatchSnapshotError("local batch root cannot be opened safely on this platform")
    return root


def _open_canonical_root(batch_root: Path) -> tuple[Path, int, os.stat_result]:
    root = _canonical_local_root(batch_root)
    descriptor: int | None = None
    current = Path(root.anchor)
    try:
        descriptor = os.open(root.anchor, _DIRECTORY_FLAGS)
        for component in root.parts[1:]:
            current /= component
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise BatchSnapshotError(f"local batch root is not a directory: {root}")
        return root, descriptor, metadata
    except BatchSnapshotError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise BatchSnapshotError(f"local batch root is missing, symlinked, or not a directory: {current}") from exc


def _close_descriptor_safely(descriptor: int) -> None:
    with contextlib.suppress(OSError):
        os.close(descriptor)


def _duplicate_pinned_cloexec(descriptor: int) -> int:
    """Atomically take the lowest free fd; closed-owner guards make later reuse safe."""
    duplicate_command = getattr(fcntl, "F_DUPFD_CLOEXEC", None)
    if duplicate_command is None:
        raise BatchSnapshotError("this platform cannot create a close-on-exec pinned batch descriptor")
    try:
        return int(fcntl.fcntl(descriptor, duplicate_command, _MIN_PINNED_DESCRIPTOR))
    except OSError as exc:
        raise BatchSnapshotError("could not allocate a close-on-exec pinned batch descriptor") from exc


class _PinnedBatchRoot:
    def __init__(
        self,
        canonical_root: Path,
        descriptor: int,
        root_metadata: os.stat_result,
    ):
        self.canonical_root = canonical_root
        self._descriptor = descriptor
        self._root_device = root_metadata.st_dev
        self._root_inode = root_metadata.st_ino
        self._finalizer = weakref.finalize(self, _close_descriptor_safely, descriptor)

    @classmethod
    def open(cls, batch_root: Path) -> _PinnedBatchRoot:
        root, descriptor, root_metadata = _open_canonical_root(batch_root)
        pinned_descriptor: int | None = None
        try:
            pinned_descriptor = _duplicate_pinned_cloexec(descriptor)
            pinned_metadata = os.fstat(pinned_descriptor)
            if (
                pinned_metadata.st_dev != root_metadata.st_dev
                or pinned_metadata.st_ino != root_metadata.st_ino
                or not stat.S_ISDIR(pinned_metadata.st_mode)
            ):
                raise BatchSnapshotError(f"pinned batch root changed while opening: {root}")
            return cls(root, pinned_descriptor, pinned_metadata)
        except Exception:
            if pinned_descriptor is not None:
                _close_descriptor_safely(pinned_descriptor)
            raise
        finally:
            os.close(descriptor)

    @property
    def descriptor(self) -> int:
        if not self._finalizer.alive:
            raise BatchSnapshotError(f"pinned batch root is closed: {self.canonical_root}")
        return self._descriptor

    @property
    def proc_root(self) -> Path:
        return Path("/proc/self/fd") / str(self.descriptor)

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def verify_canonical_binding(self) -> None:
        if self.closed:
            raise BatchSnapshotError(f"pinned batch root is closed: {self.canonical_root}")
        _, descriptor, metadata = _open_canonical_root(self.canonical_root)
        try:
            if metadata.st_dev != self._root_device or metadata.st_ino != self._root_inode:
                raise BatchSnapshotError(f"snapshot root pathname changed during verification: {self.canonical_root}")
        finally:
            os.close(descriptor)

    def close(self) -> None:
        self._finalizer()


def _validate_relative_path(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise BatchSnapshotError(f"{name} must be a nonempty relative POSIX path")
    if "\x00" in value or "\\" in value:
        raise BatchSnapshotError(f"{name} must be a canonical relative POSIX path: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise BatchSnapshotError(f"{name} must be a canonical relative POSIX path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value or not relative.parts:
        raise BatchSnapshotError(f"{name} must be a canonical relative POSIX path: {value!r}")
    return value


def _record_directory(
    directories: dict[str, _StatWitness],
    relative_path: str,
    metadata: os.stat_result,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise BatchSnapshotError(f"snapshot directory is not a directory: {relative_path}")
    witness = _stat_witness(metadata)
    previous = directories.setdefault(relative_path, witness)
    if previous != witness:
        raise BatchSnapshotError(f"snapshot directory changed during verification: {relative_path}")


def _open_relative_directory(
    root_descriptor: int,
    relative_path: str,
    directories: dict[str, _StatWitness] | None = None,
) -> tuple[int, os.stat_result]:
    descriptor = os.dup(root_descriptor)
    current_parts: list[str] = []
    try:
        if relative_path not in {"", "."}:
            canonical = _validate_relative_path(relative_path, "snapshot directory path")
            for component in PurePosixPath(canonical).parts:
                current_parts.append(component)
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
                metadata = os.fstat(descriptor)
                if directories is not None:
                    _record_directory(directories, "/".join(current_parts), metadata)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise BatchSnapshotError(f"snapshot path is not a directory: {relative_path}")
        return descriptor, metadata
    except BatchSnapshotError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        current = "/".join(current_parts) or "."
        raise BatchSnapshotError(f"snapshot directory is missing, symlinked, or invalid: {current}") from exc


def _open_relative_regular(
    root_descriptor: int,
    relative_path: str,
    directories: dict[str, _StatWitness],
) -> tuple[int, os.stat_result]:
    canonical = _validate_relative_path(relative_path, "snapshot file path")
    parts = PurePosixPath(canonical).parts
    parent = "/".join(parts[:-1])
    parent_descriptor: int | None = None
    try:
        parent_descriptor, _ = _open_relative_directory(root_descriptor, parent, directories)
        descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent_descriptor)
    except BatchSnapshotError:
        raise
    except OSError as exc:
        raise BatchSnapshotError(
            f"snapshot file is missing, symlinked, or could not be opened safely: {canonical}"
        ) from exc
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise BatchSnapshotError(f"snapshot file fstat failed: {canonical}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise BatchSnapshotError(f"snapshot file must be a regular file: {canonical}")
    return descriptor, metadata


def _stream_snapshot_file(
    root_descriptor: int,
    root: Path,
    relative_path: str,
    directories: dict[str, _StatWitness],
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    capture: bool = False,
) -> tuple[_FileWitness, bytes | None]:
    descriptor, before = _open_relative_regular(root_descriptor, relative_path, directories)
    absolute_path = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        before_witness = _stat_witness(before)
        if expected_size is not None and before.st_size != expected_size:
            raise BatchSnapshotError(
                f"snapshot file size changed for {relative_path}: expected {expected_size}, got {before.st_size}"
            )
        if expected_device is not None and before.st_dev != expected_device:
            raise BatchSnapshotError(
                f"snapshot file device changed for {relative_path}: expected {expected_device}, got {before.st_dev}"
            )
        if expected_inode is not None and before.st_ino != expected_inode:
            raise BatchSnapshotError(
                f"snapshot file inode changed for {relative_path}: expected {expected_inode}, got {before.st_ino}"
            )
        if capture and before.st_size > _MAX_MANIFEST_BYTES:
            raise BatchSnapshotError(f"snapshot manifest is too large: {before.st_size} bytes")

        _after_snapshot_file_open(absolute_path, descriptor)
        digest = hashlib.sha256()
        payload = bytearray() if capture else None
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, SNAPSHOT_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
            if payload is not None:
                if bytes_read > _MAX_MANIFEST_BYTES:
                    raise BatchSnapshotError(
                        f"snapshot manifest exceeded capture limit while reading: {bytes_read} bytes"
                    )
                payload.extend(chunk)

        after = os.fstat(descriptor)
        if _stat_witness(after) != before_witness or bytes_read != before.st_size or not stat.S_ISREG(after.st_mode):
            raise BatchSnapshotError(f"snapshot file changed while hashing: {relative_path}")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise BatchSnapshotError(
                f"snapshot file sha256 changed for {relative_path}: expected {expected_sha256}, got {actual_sha256}"
            )

        binding_descriptor, binding = _open_relative_regular(root_descriptor, relative_path, directories)
        try:
            if _stat_witness(binding) != before_witness:
                raise BatchSnapshotError(f"snapshot file pathname binding changed while hashing: {relative_path}")
        finally:
            os.close(binding_descriptor)
        return (
            _FileWitness(relative_path, before_witness, actual_sha256),
            None if payload is None else bytes(payload),
        )
    except BatchSnapshotError:
        raise
    except OSError as exc:
        raise BatchSnapshotError(f"snapshot file read failed: {relative_path}") from exc
    finally:
        os.close(descriptor)


def _parse_manifest(payload: bytes, *, expected_batch_id: str | None) -> tuple[_ManifestFile, ...]:
    try:
        value = json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise BatchSnapshotError("snapshot migration manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise BatchSnapshotError("snapshot migration manifest must be an object")
    if expected_batch_id is not None and value.get("batch_id") != expected_batch_id:
        raise BatchSnapshotError(
            f"snapshot migration manifest batch_id {value.get('batch_id')!r} does not match {expected_batch_id!r}"
        )
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BatchSnapshotError("snapshot migration manifest files must be a nonempty list")
    records: list[_ManifestFile] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(raw_files):
        if not isinstance(raw_record, dict):
            raise BatchSnapshotError(f"snapshot manifest file {index} must be an object")
        relative_path = _validate_relative_path(
            raw_record.get("target_path"),
            f"snapshot manifest file {index} target_path",
        )
        if relative_path in seen:
            raise BatchSnapshotError(f"snapshot manifest has duplicate file: {relative_path}")
        seen.add(relative_path)
        size = raw_record.get("size")
        if type(size) is not int or size < 0:
            raise BatchSnapshotError(f"snapshot manifest file {relative_path} size must be nonnegative")
        sha256 = raw_record.get("sha256")
        if type(sha256) is not str or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise BatchSnapshotError(f"snapshot manifest file {relative_path} sha256 is invalid")
        records.append(_ManifestFile(relative_path, size, sha256))
    return tuple(records)


def _parquet_identities(
    batch: admission.ValidatedBatch,
    root: Path,
) -> dict[str, _ParquetIdentity]:
    result: dict[str, _ParquetIdentity] = {}
    for episode in batch.episodes:
        parquet_path = Path(episode.parquet_path)
        try:
            relative_path = parquet_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise BatchSnapshotError(f"snapshot parquet path escapes batch root: {parquet_path}") from exc
        relative_path = _validate_relative_path(relative_path, "snapshot parquet path")
        if relative_path in result:
            raise BatchSnapshotError(f"snapshot has duplicate parquet path: {relative_path}")
        result[relative_path] = _ParquetIdentity(
            size=episode.parquet_size,
            sha256=episode.parquet_sha256,
            device=episode.parquet_device,
            inode=episode.parquet_inode,
        )
    return result


def _verify_directory_witnesses(
    root_descriptor: int,
    directories: dict[str, _StatWitness],
) -> None:
    for relative_path, expected in sorted(directories.items()):
        descriptor, metadata = _open_relative_directory(root_descriptor, relative_path)
        try:
            actual = _root_stat_witness(metadata) if relative_path == "." else _stat_witness(metadata)
            if actual != expected:
                raise BatchSnapshotError(f"snapshot directory changed during verification: {relative_path}")
        finally:
            os.close(descriptor)


def _enumerate_snapshot_files(
    root_descriptor: int,
    directories: dict[str, _StatWitness],
) -> set[str]:
    files: set[str] = set()

    def visit(directory_descriptor: int, prefix: str) -> None:
        before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise BatchSnapshotError(f"snapshot path is not a directory: {prefix or '.'}")
        if prefix:
            _record_directory(directories, prefix, before)
        elif directories.get(".") != _root_stat_witness(before):
            raise BatchSnapshotError("snapshot root changed while listing")
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise BatchSnapshotError(f"snapshot directory could not be listed: {prefix or '.'}") from exc
        for name in names:
            relative_path = f"{prefix}/{name}" if prefix else name
            _validate_relative_path(relative_path, "snapshot directory entry")
            try:
                metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError as exc:
                raise BatchSnapshotError(f"snapshot entry changed while listing: {relative_path}") from exc
            if stat.S_ISREG(metadata.st_mode):
                files.add(relative_path)
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                raise BatchSnapshotError(f"snapshot entries must be regular files or directories: {relative_path}")
            try:
                child_descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
            except OSError as exc:
                raise BatchSnapshotError(
                    f"snapshot directory is missing, symlinked, or invalid: {relative_path}"
                ) from exc
            try:
                opened = os.fstat(child_descriptor)
                if _stat_witness(opened) != _stat_witness(metadata):
                    raise BatchSnapshotError(
                        f"snapshot directory pathname binding changed while listing: {relative_path}"
                    )
                _record_directory(directories, relative_path, opened)
                visit(child_descriptor, relative_path)
            finally:
                os.close(child_descriptor)
        current = os.fstat(directory_descriptor)
        before_witness = _root_stat_witness(before) if not prefix else _stat_witness(before)
        current_witness = _root_stat_witness(current) if not prefix else _stat_witness(current)
        if current_witness != before_witness:
            raise BatchSnapshotError(f"snapshot directory changed while listing: {prefix or '.'}")

    visit(root_descriptor, "")
    return files


def _verify_pinned_batch_snapshot(
    batch: admission.ValidatedBatch,
    pinned: _PinnedBatchRoot,
) -> ValidatedBatchSnapshotWitness:
    root = pinned.canonical_root
    root_descriptor = pinned.descriptor
    root_metadata = os.fstat(root_descriptor)
    directories = {".": _root_stat_witness(root_metadata)}
    files: dict[str, _FileWitness] = {}

    manifest_witness, manifest_payload = _stream_snapshot_file(
        root_descriptor,
        root,
        "migration_manifest.json",
        directories,
        expected_size=None,
        expected_sha256=batch.manifest_sha256,
        capture=True,
    )
    assert manifest_payload is not None
    records = _parse_manifest(manifest_payload, expected_batch_id=batch.batch_id)
    files[manifest_witness.relative_path] = manifest_witness
    records_by_path = {record.relative_path: record for record in records}

    missing_required = sorted(_REQUIRED_LEROBOT_MANIFEST_FILES - set(records_by_path))
    if missing_required:
        raise BatchSnapshotError(f"snapshot manifest is missing required local LeRobot files: {missing_required}")

    actual_files = _enumerate_snapshot_files(root_descriptor, directories)
    expected_files = set(records_by_path) | set(_UNMANIFESTED_SNAPSHOT_FILES)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unmanifested = sorted(actual_files - expected_files)
        raise BatchSnapshotError(
            f"snapshot manifest is not a complete local trust root: missing={missing}, unmanifested={unmanifested}"
        )

    labels_witness, _ = _stream_snapshot_file(
        root_descriptor,
        root,
        "meta/tristate_labels.json",
        directories,
        expected_size=None,
        expected_sha256=batch.labels_sha256,
    )
    files[labels_witness.relative_path] = labels_witness
    ready_witness, _ = _stream_snapshot_file(
        root_descriptor,
        root,
        "READY",
        directories,
        expected_size=0,
        expected_sha256=None,
    )
    files[ready_witness.relative_path] = ready_witness

    parquet_identities = _parquet_identities(batch, root)
    missing_parquets = sorted(set(parquet_identities) - set(records_by_path))
    if missing_parquets:
        raise BatchSnapshotError(f"snapshot manifest is missing admitted parquet files: {missing_parquets}")
    for relative_path, parquet in parquet_identities.items():
        record = records_by_path[relative_path]
        if record.size != parquet.size or record.sha256 != parquet.sha256:
            raise BatchSnapshotError(f"snapshot manifest identity disagrees with admitted parquet: {relative_path}")

    for record in sorted(records):
        parquet = parquet_identities.get(record.relative_path)
        file_witness, _ = _stream_snapshot_file(
            root_descriptor,
            root,
            record.relative_path,
            directories,
            expected_size=record.size,
            expected_sha256=record.sha256,
            expected_device=None if parquet is None else parquet.device,
            expected_inode=None if parquet is None else parquet.inode,
        )
        files[file_witness.relative_path] = file_witness

    _verify_directory_witnesses(root_descriptor, directories)
    pinned.verify_canonical_binding()
    return ValidatedBatchSnapshotWitness(
        canonical_root=str(root),
        directories=tuple(sorted(directories.items())),
        files=tuple(sorted(files.values())),
    )


def verify_validated_batch_snapshot(
    batch: admission.ValidatedBatch,
) -> ValidatedBatchSnapshotWitness:
    if not isinstance(batch, admission.ValidatedBatch):
        raise TypeError("batch must be a ValidatedBatch")
    pinned = _PinnedBatchRoot.open(batch.root)
    try:
        return _verify_pinned_batch_snapshot(batch, pinned)
    finally:
        pinned.close()


def _construct_local_only_lerobot_metadata(
    repo_id: str,
    root: Path,
):
    metadata_class = lerobot_dataset.LeRobotDatasetMetadata
    metadata = metadata_class.__new__(metadata_class)
    metadata.repo_id = repo_id
    metadata.revision = lerobot_dataset.CODEBASE_VERSION
    metadata.root = Path(root)
    try:
        metadata.load_metadata()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        raise BatchSnapshotError(f"local LeRobot metadata is missing or invalid under {root}") from exc
    return metadata


def _construct_local_only_lerobot_dataset(
    repo_id: str,
    root: Path,
    *,
    tolerance_s: float,
):
    dataset_class = lerobot_dataset.LeRobotDataset
    dataset = dataset_class.__new__(dataset_class)
    torch.utils.data.Dataset.__init__(dataset)
    dataset.repo_id = repo_id
    dataset.root = Path(root)
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.episodes = None
    dataset.tolerance_s = tolerance_s
    dataset.revision = lerobot_dataset.CODEBASE_VERSION
    dataset.video_backend = lerobot_dataset.get_safe_default_codec()
    dataset.delta_indices = None
    dataset.image_writer = None
    dataset.episode_buffer = None
    dataset.meta = _construct_local_only_lerobot_metadata(repo_id, root=dataset.root)

    episode_paths = tuple(dataset.get_episodes_file_paths())
    missing_paths = tuple(path for path in episode_paths if not (dataset.root / path).is_file())
    if missing_paths:
        raise BatchSnapshotError(f"local LeRobot episode files are missing: {missing_paths}")
    try:
        dataset.hf_dataset = dataset.load_hf_dataset()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        raise BatchSnapshotError(f"local LeRobot episode data is missing or invalid under {root}") from exc

    dataset.episode_data_index = lerobot_dataset.get_episode_data_index(dataset.meta.episodes, dataset.episodes)
    timestamps = torch.stack(dataset.hf_dataset["timestamp"]).numpy()
    episode_indices = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
    episode_data_index = {key: value.numpy() for key, value in dataset.episode_data_index.items()}
    lerobot_dataset.check_timestamps_sync(
        timestamps,
        episode_indices,
        episode_data_index,
        dataset.fps,
        dataset.tolerance_s,
    )
    return dataset


class _PinnedLeRobotFrameDataset:
    def __init__(
        self,
        batch: admission.ValidatedBatch,
        pinned: _PinnedBatchRoot,
        dataset: Any,
        initial_snapshot: ValidatedBatchSnapshotWitness,
    ):
        self._batch = batch
        self._pinned = pinned
        self._dataset = dataset
        self._initial_snapshot = initial_snapshot

    @property
    def pinned_root(self) -> Path:
        self._ensure_open()
        return self._pinned.proc_root

    def _ensure_open(self) -> None:
        if self._pinned.closed:
            raise BatchSnapshotError(f"batch {self._batch.batch_id} local dataset is closed")

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._dataset)

    def __getitem__(self, index: int):
        self._ensure_open()
        return self._dataset[index]

    def verify_unchanged(self) -> None:
        self._ensure_open()
        current = _verify_pinned_batch_snapshot(self._batch, self._pinned)
        if current != self._initial_snapshot:
            raise BatchSnapshotError(f"batch {self._batch.batch_id} snapshot changed after local dataset construction")

    def close(self) -> None:
        self._pinned.close()

    def __enter__(self) -> _PinnedLeRobotFrameDataset:
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError("pinned local LeRobot datasets cannot be pickled")


def create_lerobot_frame_dataset(batch: admission.ValidatedBatch) -> _PinnedLeRobotFrameDataset:
    if not isinstance(batch, admission.ValidatedBatch):
        raise TypeError("batch must be a ValidatedBatch")
    pinned = _PinnedBatchRoot.open(batch.root)
    try:
        before_snapshot = _verify_pinned_batch_snapshot(batch, pinned)
        dataset = _construct_local_only_lerobot_dataset(
            batch.root.name,
            root=pinned.proc_root,
            tolerance_s=0.05,
        )
        after_snapshot = _verify_pinned_batch_snapshot(batch, pinned)
        if before_snapshot != after_snapshot:
            raise BatchSnapshotError(f"batch {batch.batch_id} snapshot changed while constructing the LeRobot dataset")
        return _PinnedLeRobotFrameDataset(batch, pinned, dataset, before_snapshot)
    except Exception:
        pinned.close()
        raise


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _raw_integer(value: object, name: str) -> int:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"raw {name} must be an integer scalar") from exc
    if array.shape != () or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"raw {name} must be an integer scalar")
    return int(array)


class Stage2ObservationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        batch: admission.ValidatedBatch,
        feature_keys: Sequence[transitions.FeatureKey],
        input_transform: Callable[[dict[str, Any]], dict[str, Any]],
    ):
        if not isinstance(batch, admission.ValidatedBatch):
            raise TypeError("batch must be a ValidatedBatch")
        if not callable(input_transform):
            raise TypeError("input_transform must be callable")
        total_frames = _strict_integer(batch.total_frames, "batch total_frames")
        if total_frames < 0:
            raise ValueError("batch total_frames must be nonnegative")

        episode_by_index: dict[int, admission.ValidatedEpisode] = {}
        episode_offsets: dict[int, int] = {}
        for episode in batch.episodes:
            episode_index = _strict_integer(episode.episode_index, "episode index")
            if episode_index in episode_by_index:
                raise ValueError(f"batch {batch.batch_id} has duplicate episode {episode_index}")
            length = _strict_integer(episode.length, f"episode {episode_index} length")
            dataset_from_index = _strict_integer(
                episode.dataset_from_index,
                f"episode {episode_index} dataset_from_index",
            )
            dataset_to_index = _strict_integer(
                episode.dataset_to_index,
                f"episode {episode_index} dataset_to_index",
            )
            if (
                length < 0
                or dataset_from_index < 0
                or dataset_to_index != dataset_from_index + length
                or dataset_to_index > total_frames
            ):
                raise ValueError(f"batch {batch.batch_id} episode {episode_index} has invalid dataset offsets")
            episode_by_index[episode_index] = episode
            episode_offsets[episode_index] = dataset_from_index

        keys = tuple(feature_keys)
        for key in keys:
            if not isinstance(key, transitions.FeatureKey):
                raise TypeError("feature_keys must contain FeatureKey values")
            if key.batch_id != batch.batch_id:
                raise ValueError(
                    f"feature key batch {key.batch_id!r} does not match validated batch {batch.batch_id!r}"
                )
            episode_index = _strict_integer(key.episode_index, "feature key episode index")
            frame_index = _strict_integer(key.frame_index, "feature key frame index")
            episode = episode_by_index.get(episode_index)
            if episode is None:
                raise ValueError(f"feature key episode {episode_index} is not present in batch {batch.batch_id}")
            if frame_index < 0 or frame_index >= episode.length:
                raise ValueError(
                    f"feature key frame {frame_index} is outside episode {episode_index} length {episode.length}"
                )

        try:
            dataset = create_lerobot_frame_dataset(batch)
        except BatchSnapshotError:
            raise
        except Exception as exc:
            raise RuntimeError(f"batch {batch.batch_id}: failed to construct local LeRobot dataset") from exc
        try:
            dataset_length = len(dataset)
        except Exception:
            dataset.close()
            raise
        if dataset_length != total_frames:
            dataset.close()
            raise ValueError(
                f"LeRobot frame dataset length {dataset_length} does not match validated batch total {total_frames}"
            )

        self._batch = batch
        self._keys = keys
        self._input_transform = input_transform
        self._dataset = dataset
        self._episode_by_index = episode_by_index
        self._episode_offsets = episode_offsets
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise BatchSnapshotError(f"batch {self._batch.batch_id} observation dataset is closed")

    @property
    def pinned_root(self) -> Path:
        self._ensure_open()
        return self._dataset.pinned_root

    @property
    def batch_id(self) -> str:
        self._ensure_open()
        return self._batch.batch_id

    @property
    def feature_keys(self) -> tuple[transitions.FeatureKey, ...]:
        self._ensure_open()
        return self._keys

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._keys)

    def __getitem__(self, index: int):
        self._ensure_open()
        key = self._keys[index]
        episode = self._episode_by_index[key.episode_index]
        global_index = self._episode_offsets[key.episode_index] + key.frame_index
        context = (
            f"batch {self._batch.batch_id} episode {key.episode_index} "
            f"frame {key.frame_index} global index {global_index}"
        )
        try:
            raw_value = self._dataset[global_index]
        except Exception as exc:
            raise RuntimeError(f"{context}: failed to read local observation") from exc
        if not isinstance(raw_value, Mapping):
            raise TypeError(f"{context}: LeRobot observation must be a mapping")
        value = dict(raw_value)
        expected_identity = {
            "index": global_index,
            "episode_index": episode.episode_index,
            "frame_index": key.frame_index,
        }
        for field, expected in expected_identity.items():
            if field not in value:
                raise ValueError(f"{context}: raw observation is missing required {field}")
            try:
                actual = _raw_integer(value[field], field)
            except ValueError as exc:
                raise ValueError(f"{context}: {exc}") from exc
            if actual != expected:
                raise ValueError(f"{context}: raw {field} {actual} does not match expected {expected}")
        try:
            observation = self._input_transform(value)
        except Exception as exc:
            raise RuntimeError(f"{context}: input transform failed") from exc
        return key, observation

    def verify_unchanged(self) -> None:
        self._ensure_open()
        self._dataset.verify_unchanged()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._dataset.close()

    def __enter__(self) -> Stage2ObservationDataset:
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError("Stage2ObservationDataset cannot be pickled while it owns a pinned local batch")


def observation_only_repack(group: transforms.Group) -> transforms.Group:
    result: list[transforms.DataTransformFn] = []
    for transform in group.inputs:
        if not isinstance(transform, transforms.RepackTransform):
            result.append(transform)
            continue
        structure = {key: value for key, value in transform.structure.items() if key != "actions"}
        result.append(dataclasses.replace(transform, structure=structure))
    return transforms.Group(inputs=tuple(result), outputs=group.outputs)


def stage2_prompt_prefix(
    tasks: dict[int, str],
    *,
    prompt_from_task: bool,
) -> tuple[transforms.DataTransformFn, ...]:
    prefix: list[transforms.DataTransformFn] = []
    if prompt_from_task:
        prefix.append(transforms.PromptFromLeRobotTask(tasks))
    prefix.append(transforms.InjectDefaultPrompt(DEFAULT_PROMPT))
    return tuple(prefix)


def build_stage2_input_transform(
    train_config,
    batch: admission.ValidatedBatch,
    norm_stats: dict,
):
    if not isinstance(batch, admission.ValidatedBatch):
        raise TypeError("batch must be a ValidatedBatch")
    pinned = _PinnedBatchRoot.open(batch.root)
    try:
        before_snapshot = _verify_pinned_batch_snapshot(batch, pinned)
        source = train_config.data.create(
            train_config.assets_dirs,
            train_config.model,
        )
        dataset_meta = _construct_local_only_lerobot_metadata(
            batch.root.name,
            root=pinned.proc_root,
        )
        tasks = dict(dataset_meta.tasks)
        after_snapshot = _verify_pinned_batch_snapshot(batch, pinned)
        if before_snapshot != after_snapshot:
            raise BatchSnapshotError(f"batch {batch.batch_id} snapshot changed while constructing LeRobot metadata")
    finally:
        pinned.close()
    data_config = dataclasses.replace(
        source,
        repo_id=str(batch.root),
        norm_stats=norm_stats,
    )
    prefix = stage2_prompt_prefix(
        tasks,
        prompt_from_task=data_config.prompt_from_task,
    )
    transform = transforms.compose(
        [
            *prefix,
            *observation_only_repack(data_config.repack_transforms).inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ]
    )
    return data_config, transform


def _stack_observation_values(*values: object) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    for array in arrays:
        if array.dtype.kind in {"O", "S", "U"}:
            raise TypeError(f"observation leaves must have numeric or boolean dtype, got {array.dtype}")
    return np.ascontiguousarray(np.stack(arrays, axis=0))


def collate_observations(rows):
    if not rows:
        raise ValueError("observation batch must contain at least one row")
    keys, observations = zip(*rows, strict=True)
    stacked = jax.tree.map(
        _stack_observation_values,
        *observations,
    )
    return tuple(keys), stacked


def _noise_for_keys(
    keys: Sequence[transitions.FeatureKey],
    *,
    feature_id: str,
    action_horizon: int,
    action_dim: int,
) -> jax.Array:
    keys = tuple(keys)
    if not keys:
        raise ValueError("noise feature keys must be nonempty")
    values = []
    for key in keys:
        rng = feature_identity.frame_key(
            feature_id,
            key.batch_id,
            key.episode_index,
            key.frame_index,
        )
        values.append(
            jax.random.normal(
                rng,
                (action_horizon, action_dim),
                dtype=jnp.float32,
            )
        )
    return jnp.stack(values, axis=0)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an exact integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _model_dimension(value: object, *, name: str, expected: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"model {name} must be an exact integer equal to {expected}")
    result = int(value)
    if result != expected:
        raise ValueError(f"model {name} must be {expected}, got {result}")
    return result


def _feature_index(value: object, *, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    if result > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds the int32 cache range")
    return result


def _validate_feature_keys(
    feature_keys: Sequence[transitions.FeatureKey],
    *,
    dataset: Stage2ObservationDataset,
) -> tuple[transitions.FeatureKey, ...]:
    try:
        keys = tuple(feature_keys)
    except TypeError as exc:
        raise TypeError("feature_keys must be a finite sequence of FeatureKey values") from exc
    if not keys:
        raise ValueError("feature_keys must be nonempty")
    for row, key in enumerate(keys):
        if not isinstance(key, transitions.FeatureKey):
            raise TypeError(f"feature_keys row {row} must be a FeatureKey")
        if type(key.batch_id) is not str or not key.batch_id:
            raise ValueError(f"feature_keys row {row} batch_id must be a nonempty string")
        _feature_index(
            key.episode_index,
            name=f"feature_keys row {row} episode_index",
        )
        _feature_index(
            key.frame_index,
            name=f"feature_keys row {row} frame_index",
        )
    if keys != tuple(sorted(set(keys))):
        raise ValueError("feature keys must be unique sorted values")
    dataset_batch_id = dataset.batch_id
    for row, key in enumerate(keys):
        if key.batch_id != dataset_batch_id:
            raise ValueError(
                f"feature key row {row} batch {key.batch_id!r} does not match dataset batch {dataset_batch_id!r}"
            )
    if keys != dataset.feature_keys:
        raise ValueError("feature_keys must exactly match the Stage2ObservationDataset frozen feature keys")
    return keys


def _is_real_floating_dtype(dtype: np.dtype) -> bool:
    return dtype == np.dtype(ml_dtypes.bfloat16) or np.issubdtype(dtype, np.floating)


def _host_array(value: object, *, name: str) -> np.ndarray:
    try:
        return np.asarray(jax.device_get(value))
    except Exception as exc:
        raise ValueError(f"{name} could not be materialized on the host") from exc


def _finite(value: np.ndarray, *, name: str) -> None:
    try:
        finite = bool(np.isfinite(value).all())
    except TypeError as exc:
        raise ValueError(f"{name} must contain real finite values") from exc
    if not finite:
        raise ValueError(f"{name} must contain only finite values")


def _observation_from_host(
    raw_observation: object,
    *,
    batch_size: int,
    batch_index: int,
) -> model_api.Observation:
    if not isinstance(raw_observation, Mapping):
        raise TypeError(f"feature loader batch {batch_index} observation must be a mapping")
    if "state" not in raw_observation:
        raise ValueError(f"feature loader batch {batch_index} observation is missing normalized state")
    try:
        host_state = np.asarray(raw_observation["state"])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"feature loader batch {batch_index} normalized state is not array-like") from exc
    if host_state.ndim != 2 or host_state.shape[0] != batch_size or host_state.shape[1] < _FEATURE_STATE_DIM:
        raise ValueError(
            f"feature loader batch {batch_index} normalized state must have shape [B,>=16], got {host_state.shape}"
        )
    if not _is_real_floating_dtype(host_state.dtype):
        raise ValueError(f"feature loader batch {batch_index} normalized state must have a real floating dtype")
    _finite(
        host_state,
        name=f"feature loader batch {batch_index} normalized state",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        host_state_float32 = np.asarray(host_state, dtype=np.float32)
    _finite(
        host_state_float32,
        name=f"feature loader batch {batch_index} normalized state after float32 conversion",
    )

    def to_device(value: object) -> jax.Array:
        try:
            host = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"feature loader batch {batch_index} observation leaf is not array-like") from exc
        if host.dtype.kind in {"O", "S", "U"}:
            raise TypeError(f"feature loader batch {batch_index} observation leaf must have numeric or boolean dtype")
        return jnp.asarray(host)

    try:
        return model_api.Observation.from_dict(jax.tree.map(to_device, raw_observation))
    except Exception as exc:
        raise ValueError(f"feature loader batch {batch_index} could not construct a model Observation") from exc


def _normalized_state_rows(
    observation: model_api.Observation,
    *,
    batch_size: int,
    batch_index: int,
) -> np.ndarray:
    state = _host_array(
        observation.state,
        name=f"feature loader batch {batch_index} normalized state",
    )
    if state.ndim != 2 or state.shape[0] != batch_size or state.shape[1] < _FEATURE_STATE_DIM:
        raise ValueError(
            f"feature loader batch {batch_index} normalized state must have shape [B,>=16], got {state.shape}"
        )
    if not _is_real_floating_dtype(state.dtype):
        raise ValueError(f"feature loader batch {batch_index} normalized state must have a real floating dtype")
    _finite(
        state,
        name=f"feature loader batch {batch_index} normalized state",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        cached = np.asarray(
            state[:, :_FEATURE_STATE_DIM],
            dtype=np.float32,
        )
    _finite(
        cached,
        name=f"feature loader batch {batch_index} normalized state after float32 conversion",
    )
    return np.ascontiguousarray(cached)


def _sampler_rows(
    actions: object,
    z_rl: object,
    *,
    batch_size: int,
    batch_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_rows = _host_array(
        actions,
        name=f"feature loader batch {batch_index} actions",
    )
    expected_action_shape = (
        batch_size,
        _FEATURE_ACTION_HORIZON,
        _FEATURE_ACTION_DIM,
    )
    if action_rows.shape != expected_action_shape:
        raise ValueError(
            f"feature loader batch {batch_index} actions must have shape [B,50,32], "
            f"got {action_rows.shape} for B={batch_size}"
        )
    if action_rows.dtype != np.dtype(np.float32):
        raise ValueError(f"feature loader batch {batch_index} actions must have float32 dtype, got {action_rows.dtype}")
    _finite(
        action_rows,
        name=f"feature loader batch {batch_index} actions",
    )

    z_rows = _host_array(
        z_rl,
        name=f"feature loader batch {batch_index} z_rl",
    )
    expected_z_shape = (batch_size, _FEATURE_Z_DIM)
    if z_rows.shape != expected_z_shape:
        raise ValueError(
            f"feature loader batch {batch_index} z_rl must have shape [B,2048], got {z_rows.shape} for B={batch_size}"
        )
    if not _is_real_floating_dtype(z_rows.dtype):
        raise ValueError(f"feature loader batch {batch_index} z_rl must have a real floating dtype, got {z_rows.dtype}")
    _finite(
        z_rows,
        name=f"feature loader batch {batch_index} z_rl",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        cached_z = np.asarray(z_rows, dtype=ml_dtypes.bfloat16)
    try:
        cached_z_finite = bool(np.isfinite(cached_z).all())
    except TypeError as exc:
        raise ValueError(
            f"feature loader batch {batch_index} z_rl must remain finite after bfloat16 conversion"
        ) from exc
    if not cached_z_finite:
        raise ValueError(f"feature loader batch {batch_index} z_rl must remain finite after bfloat16 conversion")

    reference = np.asarray(
        action_rows[
            :,
            :_FEATURE_REFERENCE_HORIZON,
            :_FEATURE_REFERENCE_DIM,
        ],
        dtype=np.float32,
    )
    _finite(
        reference,
        name=f"feature loader batch {batch_index} VLA reference",
    )
    return (
        np.ascontiguousarray(cached_z),
        np.ascontiguousarray(reference),
    )


def _extract_feature_table(
    *,
    model: nnx.Module,
    dataset: Stage2ObservationDataset,
    feature_keys: Sequence[transitions.FeatureKey],
    feature_id: str,
    micro_batch_size: int,
    num_workers: int,
    sampler_num_steps: int,
) -> cache.FeatureTable:
    keys = _validate_feature_keys(feature_keys, dataset=dataset)
    if type(feature_id) is not str or not feature_id:
        raise ValueError("feature_id must be a nonempty string")
    micro_batch_size = _positive_integer(
        micro_batch_size,
        name="micro_batch_size",
    )
    sampler_num_steps = _positive_integer(
        sampler_num_steps,
        name="sampler_num_steps",
    )
    if type(num_workers) is not int or num_workers != 0:
        raise ValueError("num_workers must be the exact integer zero for pinned Stage 2 datasets")
    action_horizon = _model_dimension(
        getattr(model, "action_horizon", None),
        name="action_horizon",
        expected=_FEATURE_ACTION_HORIZON,
    )
    action_dim = _model_dimension(
        getattr(model, "action_dim", None),
        name="action_dim",
        expected=_FEATURE_ACTION_DIM,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_observations,
    )
    sample = nnx_utils.module_jit(model.sample_actions_and_rl_token)

    row_count = len(keys)
    episode_index = np.empty(row_count, dtype=np.int32)
    frame_index = np.empty(row_count, dtype=np.int32)
    z_cache = np.empty(
        (row_count, _FEATURE_Z_DIM),
        dtype=ml_dtypes.bfloat16,
    )
    state_cache = np.empty(
        (row_count, _FEATURE_STATE_DIM),
        dtype=np.float32,
    )
    reference_cache = np.empty(
        (
            row_count,
            _FEATURE_REFERENCE_HORIZON,
            _FEATURE_REFERENCE_DIM,
        ),
        dtype=np.float32,
    )

    cursor = 0
    for batch_index, loaded in enumerate(loader):
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise TypeError(f"feature loader batch {batch_index} must return exactly (keys, observation)")
        batch_keys, raw_observation = loaded
        if not isinstance(batch_keys, tuple) or not batch_keys:
            raise ValueError(f"feature loader batch {batch_index} keys must be a nonempty tuple")
        batch_size = len(batch_keys)
        expected_keys = keys[cursor : cursor + batch_size]
        if batch_keys != expected_keys:
            raise ValueError(
                f"feature loader keys must exactly preserve the requested order without reorder, duplicate, "
                f"drop, or foreign values at batch {batch_index}: expected {expected_keys}, got {batch_keys}"
            )

        observation = _observation_from_host(
            raw_observation,
            batch_size=batch_size,
            batch_index=batch_index,
        )
        state_rows = _normalized_state_rows(
            observation,
            batch_size=batch_size,
            batch_index=batch_index,
        )
        noise = _noise_for_keys(
            batch_keys,
            feature_id=feature_id,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        try:
            actions, z_rl = sample(
                jax.random.key(0),
                observation,
                num_steps=sampler_num_steps,
                noise=noise,
            )
        except Exception as exc:
            first = batch_keys[0]
            last = batch_keys[-1]
            exc.add_note(
                f"shared-prefix feature sampling failed for loader batch {batch_index}, keys {first} through {last}"
            )
            raise
        z_rows, reference_rows = _sampler_rows(
            actions,
            z_rl,
            batch_size=batch_size,
            batch_index=batch_index,
        )
        end = cursor + batch_size
        episode_index[cursor:end] = [key.episode_index for key in batch_keys]
        frame_index[cursor:end] = [key.frame_index for key in batch_keys]
        z_cache[cursor:end] = z_rows
        state_cache[cursor:end] = state_rows
        reference_cache[cursor:end] = reference_rows
        cursor = end

    if cursor != row_count:
        raise ValueError(
            f"feature loader keys must exactly preserve all requested rows; loaded {cursor} of {row_count}"
        )

    result = cache.FeatureTable(
        episode_index=episode_index,
        frame_index=frame_index,
        z_rl=z_cache,
        state_norm=state_cache,
        vla_reference=reference_cache,
    )
    validated_rows = cache._validate_features(result)  # noqa: SLF001
    if validated_rows != row_count:
        raise RuntimeError(f"cache feature validation returned {validated_rows} rows for {row_count} extracted keys")
    return result


def _guard_message(issues: Sequence[tuple[str, BaseException]]) -> str:
    details = "; ".join(f"{label}: {error}" for label, error in issues)
    return f"frozen feature extraction guard failed: {details}"


def extract_features(
    *,
    model: nnx.Module,
    dataset: Stage2ObservationDataset,
    feature_keys: Sequence[transitions.FeatureKey],
    feature_id: str,
    expected_parameter_sha256: str,
    micro_batch_size: int = 4,
    num_workers: int = 0,
    sampler_num_steps: int = 10,
) -> cache.FeatureTable:
    if not isinstance(dataset, Stage2ObservationDataset):
        raise TypeError("dataset must be a Stage2ObservationDataset whose lifecycle can be owned and closed")

    expected: str | None = None
    before: str | None = None
    result: cache.FeatureTable | None = None
    extraction_error: BaseException | None = None
    guard_issues: list[tuple[str, BaseException]] = []
    try:
        try:
            if (
                type(expected_parameter_sha256) is not str
                or _SHA256_PATTERN.fullmatch(expected_parameter_sha256) is None
            ):
                raise ValueError("expected_parameter_sha256 must be exactly 64 lowercase hexadecimal characters")
            expected = expected_parameter_sha256
        except BaseException as exc:
            extraction_error = exc

        if extraction_error is None:
            try:
                before = feature_identity.parameter_tree_sha256(nnx.state(model, nnx.Param))
                if before != expected:
                    raise RuntimeError(f"parameter hash before extraction {before} does not match expected {expected}")
            except BaseException as exc:
                guard_issues.append(("parameter pre-check", exc))

        if extraction_error is None and not guard_issues:
            try:
                dataset.verify_unchanged()
            except BaseException as exc:
                guard_issues.append(("snapshot pre-check", exc))

        if extraction_error is None and not guard_issues:
            try:
                result = _extract_feature_table(
                    model=model,
                    dataset=dataset,
                    feature_keys=feature_keys,
                    feature_id=feature_id,
                    micro_batch_size=micro_batch_size,
                    num_workers=num_workers,
                    sampler_num_steps=sampler_num_steps,
                )
            except BaseException as exc:
                extraction_error = exc
    finally:
        try:
            dataset.verify_unchanged()
        except BaseException as exc:
            guard_issues.append(("snapshot post-check", exc))
        try:
            after = feature_identity.parameter_tree_sha256(nnx.state(model, nnx.Param))
            if before is not None and after != before:
                raise RuntimeError(f"parameter hash changed during extraction: before {before}, after {after}")
            if expected is not None and after != expected:
                raise RuntimeError(f"parameter hash after extraction {after} does not match expected {expected}")
        except BaseException as exc:
            guard_issues.append(("parameter post-check", exc))
        try:
            dataset.close()
        except BaseException as exc:
            guard_issues.append(("dataset close", exc))

    if guard_issues:
        guard_error = RuntimeError(_guard_message(guard_issues))
        if extraction_error is not None:
            raise guard_error from extraction_error
        raise guard_error
    if extraction_error is not None:
        raise extraction_error
    if result is None:
        raise RuntimeError("feature extraction completed without a result")
    return result


def extract_features_with_frozen_guard(
    *,
    model: nnx.Module,
    dataset: Stage2ObservationDataset,
    feature_keys: Sequence[transitions.FeatureKey],
    feature_id: str,
    expected_parameter_sha256: str,
    micro_batch_size: int = 4,
    num_workers: int = 0,
    sampler_num_steps: int = 10,
) -> cache.FeatureTable:
    return extract_features(
        model=model,
        dataset=dataset,
        feature_keys=feature_keys,
        feature_id=feature_id,
        expected_parameter_sha256=expected_parameter_sha256,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
        sampler_num_steps=sampler_num_steps,
    )
