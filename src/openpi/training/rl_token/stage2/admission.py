from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.rl_token.stage2 import identity

VIDEO_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
REQUIRED_COLUMNS = (
    "observation.state",
    "action",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "intervention",
    "control_mode",
)

_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOWERCASE_SHA1 = re.compile(r"[0-9a-f]{40}")
_ROUND_ID = re.compile(r"round_[0-9]{6}")
_DATA_PATH = "data/chunk-000/episode_{episode_index:06d}.parquet"
_VIDEO_PATH = "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
_READ_CHUNK_BYTES = 8 * 1024 * 1024
_ADMISSION_FIELDS = {
    "schema_version",
    "round_id",
    "admitted_at",
    "code_commit",
    "batch_id",
    "batch_root",
    "manifest_sha256",
    "labels_sha256",
    "episode_fingerprints",
    "episode_lengths",
    "chunk_equivalents",
    "validation_report",
}
_VALIDATION_REPORT_FIELDS = {
    "episode_count",
    "total_frames",
    "video_count",
    "fps",
}


class AdmissionError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ValidatedEpisode:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    task: str
    parquet_path: Path
    parquet_size: int
    parquet_sha256: str
    parquet_device: int
    parquet_inode: int
    labels: np.ndarray
    intervention: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _immutable_array(self.labels))
        object.__setattr__(self, "intervention", _immutable_array(self.intervention))


@dataclasses.dataclass(frozen=True)
class ValidatedBatch:
    batch_id: str
    root: Path
    fps: int
    total_frames: int
    manifest_sha256: str
    labels_sha256: str
    episode_fingerprints: tuple[str, ...]
    episodes: tuple[ValidatedEpisode, ...]

    @property
    def chunk_equivalents(self) -> int:
        return sum(math.ceil(episode.length / 20) for episode in self.episodes)


@dataclasses.dataclass(frozen=True)
class PublishedAdmission:
    path: Path
    round_id: str
    admitted_at: str
    code_commit: str
    batch_id: str
    manifest_sha256: str
    labels_sha256: str
    episode_fingerprints: tuple[str, ...]
    episode_lengths: tuple[int, ...]
    chunk_equivalents: int
    sha256: str


VideoValidator = Callable[[Path, int, float], None]


@dataclasses.dataclass(frozen=True)
class _ManifestFile:
    target_path: str
    size: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class _ValidatedManifest:
    frame_count: int
    episode_fingerprints: tuple[str, ...]
    files: tuple[_ManifestFile, ...]


@dataclasses.dataclass(frozen=True)
class _EpisodeRecord:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    task: str
    source_fingerprint: str


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _after_verified_file_open(_path: Path, _descriptor: int) -> None:
    """Test hook for exercising pathname replacement after fd pinning."""


def _after_admission_file_open(_path: Path, _descriptor: int) -> None:
    """Test hook for exercising admission mutation after fd pinning."""


def _open_nofollow_regular(path: Path) -> tuple[int, os.stat_result]:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise AdmissionError(f"verified file path must be absolute and normalized: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None or os.open not in os.supports_dir_fd:
        raise AdmissionError(f"verified file {path} cannot be opened safely with component-wise O_NOFOLLOW")
    if len(path.parts) < 2:
        raise AdmissionError(f"verified file {path} must name a regular file")
    nonblock = getattr(os, "O_NONBLOCK", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | directory | nofollow | nonblock | cloexec
    file_flags = os.O_RDONLY | nofollow | nonblock | cloexec
    parent_descriptor: int | None = None
    current_path = Path(path.anchor)
    try:
        try:
            parent_descriptor = os.open(path.anchor, directory_flags)
        except OSError as exc:
            raise AdmissionError(f"verified file {path} root failed to open safely: {exc}") from exc
        for component in path.parts[1:-1]:
            current_path /= component
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise AdmissionError(
                        f"verified file {path} ancestor is a symlink or not a directory: {current_path}"
                    ) from exc
                if exc.errno == errno.ENOENT:
                    raise AdmissionError(f"verified file {path} ancestor is missing: {current_path}") from exc
                raise AdmissionError(
                    f"verified file {path} ancestor failed to open safely: {current_path}: {exc}"
                ) from exc
            with contextlib.suppress(OSError):
                os.close(parent_descriptor)
            parent_descriptor = child_descriptor
        try:
            descriptor = os.open(
                path.parts[-1],
                file_flags,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise AdmissionError(f"verified file {path} symlink is forbidden") from exc
            if exc.errno == errno.ENOENT:
                raise AdmissionError(f"verified file {path} is missing") from exc
            raise AdmissionError(f"verified file {path} failed to open safely: {exc}") from exc
    finally:
        if parent_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(parent_descriptor)
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise AdmissionError(f"verified file {path} fstat failed: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise AdmissionError(f"verified file {path} must be a regular file")
    return descriptor, metadata


def _read_verified_regular_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    path = Path(path)
    descriptor, metadata = _open_nofollow_regular(path)
    try:
        if metadata.st_size != expected_size:
            raise AdmissionError(
                f"verified file {path} size mismatch: expected {expected_size}, got {metadata.st_size}"
            )
        if expected_device is not None and metadata.st_dev != expected_device:
            raise AdmissionError(
                f"verified file {path} device mismatch: expected {expected_device}, got {metadata.st_dev}"
            )
        if expected_inode is not None and metadata.st_ino != expected_inode:
            raise AdmissionError(
                f"verified file {path} inode mismatch: expected {expected_inode}, got {metadata.st_ino}"
            )
        _after_verified_file_open(path, descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            chunks.append(chunk)
            digest.update(chunk)
        payload = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev != metadata.st_dev
            or final_metadata.st_ino != metadata.st_ino
            or final_metadata.st_size != metadata.st_size
        ):
            raise AdmissionError(f"verified file {path} changed while being read")
        if len(payload) != expected_size:
            raise AdmissionError(
                f"verified file {path} size mismatch while reading: expected {expected_size}, got {len(payload)}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise AdmissionError(
                f"verified file {path} sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        path_descriptor: int | None = None
        try:
            path_descriptor, path_metadata = _open_nofollow_regular(path)
        except AdmissionError as exc:
            raise AdmissionError(f"verified file {path} pathname changed while being read: {exc}") from exc
        finally:
            if path_descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(path_descriptor)
        if path_metadata.st_dev != metadata.st_dev or path_metadata.st_ino != metadata.st_ino:
            raise AdmissionError(f"verified file {path} pathname changed while being read")
        return payload, metadata
    except AdmissionError:
        raise
    except OSError as exc:
        raise AdmissionError(f"verified file {path} read failed: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def validate_ready_batch(
    batch_root: Path,
    *,
    expected_episodes: int = 20,
    expected_fps: int = 30,
    video_tolerance_s: float = 0.05,
    video_validator: VideoValidator,
) -> ValidatedBatch:
    expected_episodes = _strict_positive_int(expected_episodes, "expected_episodes")
    expected_fps = _strict_positive_int(expected_fps, "expected_fps")
    if (
        isinstance(video_tolerance_s, bool)
        or not isinstance(video_tolerance_s, int | float)
        or not math.isfinite(video_tolerance_s)
        or video_tolerance_s < 0
    ):
        raise AdmissionError("video_tolerance_s must be a finite nonnegative number")
    if not callable(video_validator):
        raise AdmissionError("video_validator must be callable")

    root = _require_real_directory(Path(batch_root))
    _require_safe_file(root, PurePosixPath("READY"))
    manifest_path = _require_safe_file(root, PurePosixPath("migration_manifest.json"))
    labels_path = _require_safe_file(root, PurePosixPath("meta/tristate_labels.json"))
    expert_path = _require_safe_file(root, PurePosixPath("meta/expert_frame_index.json"))

    manifest_bytes = _read_bytes(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _validate_manifest(
        root,
        _loads_json_bytes(manifest_bytes, manifest_path),
        expected_episodes=expected_episodes,
    )
    for record in manifest.files:
        _verify_manifest_file(root, record)
    manifest_by_path = {record.target_path: record for record in manifest.files}

    info_path = root / "meta/info.json"
    info = _validate_info(
        _read_json(info_path),
        expected_episodes=expected_episodes,
        expected_fps=expected_fps,
    )
    metadata_episodes = _validate_episode_jsonl(
        root / "meta/episodes.jsonl",
        expected_episodes=expected_episodes,
        expected_fingerprints=manifest.episode_fingerprints,
    )
    declared_task = _validate_tasks_jsonl(root / "meta/tasks.jsonl")
    if declared_task != metadata_episodes[0].task:
        raise AdmissionError("tasks.jsonl task does not match the unique common task in episodes.jsonl")
    metadata_total_frames = sum(record.length for record in metadata_episodes)
    info_total_frames = _strict_positive_int(info.get("total_frames"), "info total_frames")
    if manifest.frame_count != info_total_frames or info_total_frames != metadata_total_frames:
        raise AdmissionError(
            "frame totals disagree between migration manifest, info.json, and episodes.jsonl: "
            f"{manifest.frame_count}, {info_total_frames}, {metadata_total_frames}"
        )

    episode_lengths = tuple(record.length for record in metadata_episodes)
    labels_bytes = _read_bytes(labels_path)
    labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()
    labels = _load_frame_labels(
        labels_path,
        dataset_name=root.name,
        episode_lengths=episode_lengths,
        source_bytes=labels_bytes,
    )
    expert_masks = _expert_masks(expert_path, episode_lengths)
    episodes: list[ValidatedEpisode] = []
    for record, frame_labels, expert_mask in zip(
        metadata_episodes,
        labels,
        expert_masks,
        strict=True,
    ):
        parquet_relative = _DATA_PATH.format(episode_index=record.episode_index)
        parquet_path = root / parquet_relative
        parquet_manifest = manifest_by_path[parquet_relative]
        try:
            parquet_bytes, parquet_metadata = _read_verified_regular_file(
                parquet_path,
                expected_size=parquet_manifest.size,
                expected_sha256=parquet_manifest.sha256,
            )
            table = pq.read_table(
                pa.BufferReader(parquet_bytes),
                columns=list(REQUIRED_COLUMNS),
            )
        except Exception as exc:
            raise AdmissionError(
                f"batch {root.name} episode {record.episode_index} parquet read failed: {exc}"
            ) from exc
        state, action, intervention = _validate_parquet_episode(table, record)
        del state, action
        if not np.array_equal(expert_mask, intervention):
            frame_index = int(np.flatnonzero(expert_mask != intervention)[0])
            raise AdmissionError(
                f"batch {root.name} episode {record.episode_index} frame {frame_index} "
                f"expert={bool(expert_mask[frame_index])} does not match "
                f"intervention={bool(intervention[frame_index])}"
            )
        for video_key in VIDEO_KEYS:
            video_validator(
                root / "videos/chunk-000" / video_key / f"episode_{record.episode_index:06d}.mp4",
                record.length,
                float(video_tolerance_s),
            )
        episodes.append(
            ValidatedEpisode(
                episode_index=record.episode_index,
                length=record.length,
                dataset_from_index=record.dataset_from_index,
                dataset_to_index=record.dataset_to_index,
                task=record.task,
                parquet_path=parquet_path,
                parquet_size=parquet_manifest.size,
                parquet_sha256=parquet_manifest.sha256,
                parquet_device=parquet_metadata.st_dev,
                parquet_inode=parquet_metadata.st_ino,
                labels=frame_labels,
                intervention=intervention,
            )
        )

    _verify_validation_snapshot(
        root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        labels_path=labels_path,
        labels_sha256=labels_sha256,
        manifest=manifest,
    )
    return ValidatedBatch(
        batch_id=root.name,
        root=root,
        fps=_strict_positive_int(info.get("fps"), "info fps"),
        total_frames=metadata_total_frames,
        manifest_sha256=manifest_sha256,
        labels_sha256=labels_sha256,
        episode_fingerprints=manifest.episode_fingerprints,
        episodes=tuple(episodes),
    )


def admission_payload(
    batch: ValidatedBatch,
    *,
    round_id: str,
    admitted_at: str,
    code_commit: str,
) -> dict[str, object]:
    round_id = _validate_round_id(round_id, "round_id")
    admitted_at = _validate_admitted_at(admitted_at, "admitted_at")
    code_commit = _validate_code_commit(code_commit, "code_commit")
    return {
        "schema_version": 1,
        "round_id": round_id,
        "admitted_at": admitted_at,
        "code_commit": code_commit,
        "batch_id": batch.batch_id,
        "batch_root": str(batch.root),
        "manifest_sha256": batch.manifest_sha256,
        "labels_sha256": batch.labels_sha256,
        "episode_fingerprints": list(batch.episode_fingerprints),
        "episode_lengths": [episode.length for episode in batch.episodes],
        "chunk_equivalents": batch.chunk_equivalents,
        "validation_report": {
            "episode_count": len(batch.episodes),
            "total_frames": batch.total_frames,
            "video_count": len(batch.episodes) * len(VIDEO_KEYS),
            "fps": batch.fps,
        },
    }


def publish_admission(
    batch: ValidatedBatch,
    training_root: Path,
    *,
    round_id: str,
    admitted_at: str,
    code_commit: str,
) -> Path:
    payload = admission_payload(
        batch,
        round_id=round_id,
        admitted_at=admitted_at,
        code_commit=code_commit,
    )
    destination = Path(training_root) / "admissions" / f"{round_id}.json"
    identity.atomic_write_json(destination, payload)
    return destination


def open_admission(path: Path) -> PublishedAdmission:
    path = _absolute_unresolved_path(path)
    source_bytes, sha256 = _read_pinned_admission(path)
    payload = _require_mapping(
        _loads_json_bytes(source_bytes, path),
        "admission",
    )
    try:
        canonical_bytes = identity.canonical_json_bytes(payload)
    except (TypeError, UnicodeError, ValueError) as exc:
        raise AdmissionError(f"admission JSON cannot be encoded canonically: {path}") from exc
    if source_bytes != canonical_bytes:
        raise AdmissionError(f"admission JSON is not canonical: {path}")
    if set(payload) != _ADMISSION_FIELDS:
        raise AdmissionError(f"unexpected admission fields in {path}")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise AdmissionError(f"unsupported admission schema in {path}")

    round_id = _validate_round_id(payload["round_id"], f"round_id in {path}")
    admitted_at = _validate_admitted_at(payload["admitted_at"], f"admitted_at in {path}")
    code_commit = _validate_code_commit(payload["code_commit"], f"code_commit in {path}")
    batch_id = _validate_batch_id(payload["batch_id"], path)
    _validate_batch_root(payload["batch_root"], batch_id, path)
    manifest_sha256 = _validate_sha256(payload["manifest_sha256"], f"manifest_sha256 in {path}")
    labels_sha256 = _validate_sha256(payload["labels_sha256"], f"labels_sha256 in {path}")

    fingerprints_value = payload["episode_fingerprints"]
    if type(fingerprints_value) is not list or len(fingerprints_value) != 20:
        raise AdmissionError(f"invalid episode fingerprints in {path}")
    episode_fingerprints = tuple(
        _validate_sha256(value, f"episode fingerprint {index} in {path}")
        for index, value in enumerate(fingerprints_value)
    )
    if len(set(episode_fingerprints)) != 20:
        raise AdmissionError(f"invalid duplicate episode fingerprint in {path}")

    lengths_value = payload["episode_lengths"]
    if (
        type(lengths_value) is not list
        or len(lengths_value) != 20
        or any(type(length) is not int or length <= 0 for length in lengths_value)
    ):
        raise AdmissionError(f"invalid episode lengths in {path}")
    episode_lengths = tuple(lengths_value)
    expected_chunks = sum((length + 19) // 20 for length in episode_lengths)
    chunk_equivalents = payload["chunk_equivalents"]
    if type(chunk_equivalents) is not int or chunk_equivalents != expected_chunks:
        raise AdmissionError(
            f"chunk_equivalents mismatch in {path}: expected {expected_chunks}, got {chunk_equivalents!r}"
        )

    report = payload["validation_report"]
    if type(report) is not dict or set(report) != _VALIDATION_REPORT_FIELDS:
        raise AdmissionError(f"invalid validation_report in {path}")
    expected_report = {
        "episode_count": 20,
        "total_frames": sum(episode_lengths),
        "video_count": 60,
        "fps": 30,
    }
    if any(type(report[name]) is not int for name in _VALIDATION_REPORT_FIELDS) or report != expected_report:
        raise AdmissionError(f"invalid validation_report in {path}")

    return PublishedAdmission(
        path=path,
        round_id=round_id,
        admitted_at=admitted_at,
        code_commit=code_commit,
        batch_id=batch_id,
        manifest_sha256=manifest_sha256,
        labels_sha256=labels_sha256,
        episode_fingerprints=episode_fingerprints,
        episode_lengths=episode_lengths,
        chunk_equivalents=chunk_equivalents,
        sha256=sha256,
    )


def verify_admission(path: Path, batch: ValidatedBatch) -> None:
    path = Path(path)
    payload = _require_mapping(
        _loads_json_bytes(_read_admission_bytes(path), path),
        "admission",
    )
    expected = admission_payload(
        batch,
        round_id=payload.get("round_id"),
        admitted_at=payload.get("admitted_at"),
        code_commit=payload.get("code_commit"),
    )
    if payload != expected:
        raise AdmissionError(f"admission does not match immutable batch {batch.batch_id}")


def _read_admission_bytes(path: Path) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)

    def nofollow_opener(name: str, flags: int) -> int:
        return os.open(name, flags | nofollow | nonblock | cloexec)

    try:
        with open(path, "rb", opener=nofollow_opener) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise AdmissionError(f"admission path must be a regular file without a final symlink: {path}")
            return stream.read()
    except AdmissionError:
        raise
    except (OSError, ValueError) as exc:
        raise AdmissionError(
            f"admission path must be an existing regular file without a final symlink: {path}"
        ) from exc


def _absolute_unresolved_path(path: Path) -> Path:
    try:
        path = Path(path)
    except (OSError, TypeError, ValueError) as exc:
        raise AdmissionError(f"invalid admission path: {path!r}") from exc
    if "\x00" in str(path):
        raise AdmissionError(f"invalid admission path: {path!r}")
    if not path.is_absolute():
        try:
            path = Path.cwd() / path
        except (OSError, ValueError) as exc:
            raise AdmissionError(f"invalid admission path: {path!r}") from exc
    if ".." in path.parts:
        raise AdmissionError(f"admission path must be normalized without '..': {path}")
    return path


def _stat_read_witness(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_pinned_admission(path: Path) -> tuple[bytes, str]:
    descriptor, initial_metadata = _open_nofollow_regular(path)
    try:
        _after_admission_file_open(path, descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            chunks.append(chunk)
            digest.update(chunk)
        source_bytes = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        if (
            _stat_read_witness(final_metadata) != _stat_read_witness(initial_metadata)
            or len(source_bytes) != initial_metadata.st_size
        ):
            raise AdmissionError(f"admission {path} changed while being read")

        # This second descriptor is only a namespace/inode witness.  It is
        # never read or hashed; both parsing and identity use source_bytes
        # from the first pinned descriptor above.
        path_descriptor: int | None = None
        try:
            path_descriptor, path_metadata = _open_nofollow_regular(path)
        except AdmissionError as exc:
            raise AdmissionError(f"admission {path} pathname changed while being read: {exc}") from exc
        finally:
            if path_descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(path_descriptor)
        if _stat_read_witness(path_metadata) != _stat_read_witness(initial_metadata):
            raise AdmissionError(f"admission {path} pathname changed while being read")
        return source_bytes, digest.hexdigest()
    except AdmissionError:
        raise
    except OSError as exc:
        raise AdmissionError(f"admission {path} read failed: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def validate_video_with_ffprobe(path: Path, expected_frames: int, tolerance_s: float) -> None:
    del tolerance_s
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise AdmissionError(f"ffprobe failed for {path}") from exc
    if completed.returncode != 0:
        raise AdmissionError(f"ffprobe failed for {path}")
    try:
        actual_frames = int(completed.stdout.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise AdmissionError(f"ffprobe returned no frame count for {path}") from exc
    if actual_frames != expected_frames:
        raise AdmissionError(f"video frame count mismatch for {path}: expected {expected_frames}, got {actual_frames}")


def _require_real_directory(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"batch root does not exist: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise AdmissionError(f"batch root must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise AdmissionError(f"batch root must be a real directory: {path}")
    try:
        return path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"cannot resolve batch root: {path}") from exc


def _require_safe_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        try:
            metadata = candidate.lstat()
        except (OSError, ValueError) as exc:
            raise AdmissionError(
                f"batch {root.name} required file is missing or invalid: {relative.as_posix()}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise AdmissionError(f"batch {root.name} path {candidate.relative_to(root)} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise AdmissionError(f"batch {root.name} required path is not a regular file: {relative.as_posix()}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"batch {root.name} file escapes batch root: {relative.as_posix()}") from exc
    return candidate


def _read_json(path: Path) -> Any:
    return _loads_json_bytes(_read_bytes(path), path)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"cannot read file {path}: {exc}") from exc


def _loads_json_bytes(value: bytes, path: Path) -> Any:
    try:
        return json.loads(value)
    except (UnicodeError, ValueError) as exc:
        raise AdmissionError(f"invalid JSON file {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise AdmissionError(f"blank JSONL record in {path} at line {line_number}")
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AdmissionError(f"invalid JSONL record in {path} at line {line_number}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise AdmissionError(f"cannot read JSONL file {path}: {exc}") from exc
    return values


def _validate_manifest(
    root: Path,
    value: Any,
    *,
    expected_episodes: int,
) -> _ValidatedManifest:
    manifest = _require_mapping(value, "migration manifest")
    if _strict_nonnegative_int(manifest.get("schema_version"), "manifest schema_version") != 1:
        raise AdmissionError("migration manifest schema_version must be 1")
    if manifest.get("batch_id") != root.name:
        raise AdmissionError(f"migration manifest batch_id must equal batch root name {root.name!r}")
    if _strict_nonnegative_int(manifest.get("episode_count"), "manifest episode_count") != expected_episodes:
        raise AdmissionError(f"migration manifest must contain {expected_episodes} episodes")
    frame_count = _strict_positive_int(manifest.get("frame_count"), "manifest frame_count")
    _validate_manifest_timestamp(manifest.get("created_at"))

    fingerprints_value = manifest.get("episode_fingerprints")
    if not isinstance(fingerprints_value, list) or len(fingerprints_value) != expected_episodes:
        raise AdmissionError(f"migration manifest episode_fingerprints must contain {expected_episodes} values")
    episode_fingerprints = tuple(
        _validate_sha256(value, f"manifest episode_fingerprints[{index}]")
        for index, value in enumerate(fingerprints_value)
    )
    if len(set(episode_fingerprints)) != len(episode_fingerprints):
        raise AdmissionError("migration manifest contains duplicate episode fingerprints")

    episodes_value = manifest.get("episodes")
    if not isinstance(episodes_value, list) or len(episodes_value) != expected_episodes:
        raise AdmissionError(f"migration manifest episodes must contain {expected_episodes} records")
    fingerprints_by_target: dict[int, str] = {}
    source_identities: set[tuple[str, str, int]] = set()
    for position, raw_record in enumerate(episodes_value):
        record = _require_mapping(raw_record, f"manifest episode record {position}")
        target_index = _strict_nonnegative_int(
            record.get("target_index"),
            f"manifest episode {position} target_index",
        )
        if target_index in fingerprints_by_target:
            raise AdmissionError(f"duplicate manifest target index {target_index}")
        fingerprint = _validate_sha256(
            record.get("fingerprint"),
            f"manifest episode {target_index} fingerprint",
        )
        fingerprints_by_target[target_index] = fingerprint
        source_host = _strict_nonempty_string(
            record.get("source_host"),
            f"manifest episode {target_index} source_host",
        )
        source_root = _validate_source_root(
            record.get("source_dataset_root"),
            f"manifest episode {target_index} source_dataset_root",
        )
        source_index = _strict_nonnegative_int(
            record.get("source_index"),
            f"manifest episode {target_index} source_index",
        )
        source_identity = (source_host, source_root, source_index)
        if source_identity in source_identities:
            raise AdmissionError(f"duplicate manifest source identity in target episode {target_index}")
        source_identities.add(source_identity)

    expected_indices = set(range(expected_episodes))
    if set(fingerprints_by_target) != expected_indices:
        raise AdmissionError(f"manifest target indices must be contiguous 0..{expected_episodes - 1}")
    ordered_episode_fingerprints = tuple(fingerprints_by_target[index] for index in range(expected_episodes))
    if ordered_episode_fingerprints != episode_fingerprints:
        raise AdmissionError("manifest episode_fingerprints do not match episode records by target index")

    files_value = manifest.get("files")
    if not isinstance(files_value, list):
        raise AdmissionError("migration manifest files must be a list")
    files: list[_ManifestFile] = []
    seen_paths: set[str] = set()
    for position, raw_record in enumerate(files_value):
        record = _require_mapping(raw_record, f"manifest file record {position}")
        target_path = _validate_manifest_target_path(record.get("target_path"))
        if target_path in seen_paths:
            raise AdmissionError(f"duplicate manifest file target_path: {target_path}")
        seen_paths.add(target_path)
        files.append(
            _ManifestFile(
                target_path=target_path,
                size=_strict_nonnegative_int(
                    record.get("size"),
                    f"manifest file {target_path} size",
                ),
                sha256=_validate_sha256(
                    record.get("sha256"),
                    f"manifest file {target_path} sha256",
                ),
            )
        )

    required_paths = {
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/expert_frame_index.json",
        *(_DATA_PATH.format(episode_index=episode_index) for episode_index in range(expected_episodes)),
        *(
            (f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4")
            for episode_index in range(expected_episodes)
            for video_key in VIDEO_KEYS
        ),
    }
    missing_paths = sorted(required_paths - seen_paths)
    if missing_paths:
        raise AdmissionError(f"migration manifest is missing consumed core file {missing_paths[0]}")
    return _ValidatedManifest(
        frame_count=frame_count,
        episode_fingerprints=episode_fingerprints,
        files=tuple(files),
    )


def _validate_manifest_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise AdmissionError("manifest created_at must be an ISO-8601 string")
    try:
        timestamp = datetime.datetime.fromisoformat(value)
    except ValueError as exc:
        raise AdmissionError("manifest created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise AdmissionError("manifest created_at must include a timezone offset")


def _validate_source_root(value: Any, name: str) -> str:
    source_root = _strict_nonempty_string(value, name)
    if "\\" in source_root:
        raise AdmissionError(f"{name} must be a normalized absolute POSIX path")
    parsed = PurePosixPath(source_root)
    if (
        not parsed.is_absolute()
        or parsed.as_posix() != source_root
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise AdmissionError(f"{name} must be a normalized absolute POSIX path")
    return source_root


def _validate_round_id(value: Any, name: str) -> str:
    if type(value) is not str or _ROUND_ID.fullmatch(value) is None or int(value.removeprefix("round_")) <= 0:
        raise AdmissionError(f"{name} must match positive round_NNNNNN")
    return value


def _validate_admitted_at(value: Any, name: str) -> str:
    if type(value) is not str:
        raise AdmissionError(f"{name} must be ISO-8601")
    try:
        admitted_time = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise AdmissionError(f"{name} must be ISO-8601") from error
    if admitted_time.tzinfo is None or admitted_time.utcoffset() is None:
        raise AdmissionError(f"{name} must include a timezone")
    return value


def _validate_code_commit(value: Any, name: str) -> str:
    if type(value) is not str or _LOWERCASE_SHA1.fullmatch(value) is None:
        raise AdmissionError(f"{name} must be a full lowercase Git SHA-1")
    return value


def _validate_batch_id(value: Any, path: Path) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise AdmissionError(f"invalid batch_id in {path}")
    return value


def _validate_batch_root(value: Any, batch_id: str, path: Path) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value or "\\" in value:
        raise AdmissionError(f"invalid batch_root in {path}")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or parsed.as_posix() != value or any(part in {".", ".."} for part in parsed.parts):
        raise AdmissionError(f"invalid batch_root in {path}")
    if parsed.name != batch_id:
        raise AdmissionError(f"batch_root does not match batch_id in {path}")
    return value


def _validate_manifest_target_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AdmissionError("manifest target_path must be a nonempty string")
    if "\x00" in value:
        raise AdmissionError("manifest target_path must not contain NUL")
    if "\\" in value:
        raise AdmissionError(f"manifest target_path must be normalized POSIX: {value!r}")
    parsed = PurePosixPath(value)
    if (
        value == "."
        or parsed.is_absolute()
        or parsed.as_posix() != value
        or value.endswith("/")
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        detail = " containing .." if ".." in parsed.parts else ""
        raise AdmissionError(f"unsafe manifest target_path{detail}: {value!r}")
    return value


def _verify_manifest_file(root: Path, record: _ManifestFile) -> None:
    relative = PurePosixPath(record.target_path)
    path = _require_safe_file(root, relative)
    try:
        size = path.stat(follow_symlinks=False).st_size
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"cannot stat manifest file {record.target_path}") from exc
    if size != record.size:
        raise AdmissionError(f"manifest file {record.target_path} size mismatch: expected {record.size}, got {size}")
    sha256 = _sha256_file(path)
    if sha256 != record.sha256:
        raise AdmissionError(f"manifest file {record.target_path} sha256 mismatch")


def _verify_validation_snapshot(
    root: Path,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    labels_path: Path,
    labels_sha256: str,
    manifest: _ValidatedManifest,
) -> None:
    current_manifest = _require_safe_file(root, PurePosixPath("migration_manifest.json"))
    if current_manifest != manifest_path or _sha256_file(current_manifest) != manifest_sha256:
        raise AdmissionError(f"batch {root.name} migration manifest changed during validation")
    current_labels = _require_safe_file(root, PurePosixPath("meta/tristate_labels.json"))
    if current_labels != labels_path or _sha256_file(current_labels) != labels_sha256:
        raise AdmissionError(f"batch {root.name} labels changed during validation")
    for record in manifest.files:
        _verify_manifest_file(root, record)


def _validate_info(
    value: Any,
    *,
    expected_episodes: int,
    expected_fps: int,
) -> dict[str, Any]:
    info = _require_mapping(value, "info.json")
    if info.get("codebase_version") != "v2.1":
        raise AdmissionError("info codebase_version must be v2.1")
    _strict_nonempty_string(info.get("robot_type"), "info robot_type")
    if _strict_nonnegative_int(info.get("fps"), "info fps") != expected_fps:
        raise AdmissionError(f"info fps must be {expected_fps}")
    exact_counts = {
        "total_episodes": expected_episodes,
        "total_tasks": 1,
        "total_videos": expected_episodes * len(VIDEO_KEYS),
        "total_chunks": 1,
        "chunks_size": expected_episodes,
    }
    for name, expected in exact_counts.items():
        if _strict_nonnegative_int(info.get(name), f"info {name}") != expected:
            raise AdmissionError(f"info {name} must be {expected}")
    _strict_positive_int(info.get("total_frames"), "info total_frames")
    if info.get("splits") != {"train": f"0:{expected_episodes}"}:
        raise AdmissionError(f"info splits must equal {{'train': '0:{expected_episodes}'}}")
    if info.get("data_path") != _DATA_PATH:
        raise AdmissionError(f"info data_path must equal {_DATA_PATH!r}")
    if info.get("video_path") != _VIDEO_PATH:
        raise AdmissionError(f"info video_path must equal {_VIDEO_PATH!r}")

    features = _require_mapping(info.get("features"), "info features")
    required_features: dict[str, tuple[str, tuple[int, ...]]] = {
        "observation.state": ("float32", (16,)),
        "action": ("float32", (16,)),
        "intervention": ("bool", (1,)),
        "control_mode": ("int64", (1,)),
        **dict.fromkeys(VIDEO_KEYS, ("video", (480, 640, 3))),
    }
    for feature_name, (dtype, shape) in required_features.items():
        declaration = _require_mapping(
            features.get(feature_name),
            f"info feature {feature_name}",
        )
        if declaration.get("dtype") != dtype:
            raise AdmissionError(f"info feature {feature_name} dtype must be {dtype}")
        if not _is_exact_shape(declaration.get("shape"), shape):
            raise AdmissionError(f"info feature {feature_name} shape must be {list(shape)}")
    return info


def _validate_tasks_jsonl(path: Path) -> str:
    records = _read_jsonl(path)
    if len(records) != 1:
        raise AdmissionError("tasks.jsonl must contain exactly one task record")
    record = _require_mapping(records[0], "tasks.jsonl record")
    task_index = _strict_nonnegative_int(
        record.get("task_index"),
        "tasks.jsonl task_index",
    )
    if task_index != 0:
        raise AdmissionError("tasks.jsonl task_index must be 0")
    return _strict_nonempty_string(record.get("task"), "tasks.jsonl task")


def _validate_episode_jsonl(
    path: Path,
    *,
    expected_episodes: int,
    expected_fingerprints: tuple[str, ...],
) -> tuple[_EpisodeRecord, ...]:
    values = _read_jsonl(path)
    if len(values) != expected_episodes:
        raise AdmissionError(f"episodes.jsonl must contain exactly {expected_episodes} episodes")
    by_index: dict[int, _EpisodeRecord] = {}
    for position, raw_record in enumerate(values):
        record = _require_mapping(raw_record, f"episodes.jsonl record {position}")
        episode_index = _strict_nonnegative_int(
            record.get("episode_index"),
            f"episodes.jsonl record {position} episode_index",
        )
        if episode_index in by_index:
            raise AdmissionError(f"duplicate metadata episode {episode_index}")
        tasks = record.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise AdmissionError(f"episode {episode_index} must contain exactly one task")
        task = _strict_nonempty_string(tasks[0], f"episode {episode_index} task")
        length = _strict_positive_int(record.get("length"), f"episode {episode_index} length")
        dataset_from_index = _strict_nonnegative_int(
            record.get("dataset_from_index"),
            f"episode {episode_index} dataset_from_index",
        )
        dataset_to_index = _strict_nonnegative_int(
            record.get("dataset_to_index"),
            f"episode {episode_index} dataset_to_index",
        )
        source_fingerprint = _validate_sha256(
            record.get("source_fingerprint"),
            f"episode {episode_index} source_fingerprint",
        )
        by_index[episode_index] = _EpisodeRecord(
            episode_index=episode_index,
            length=length,
            dataset_from_index=dataset_from_index,
            dataset_to_index=dataset_to_index,
            task=task,
            source_fingerprint=source_fingerprint,
        )
    if set(by_index) != set(range(expected_episodes)):
        raise AdmissionError(f"episode indices must be exactly 0..{expected_episodes - 1}")

    ordered = tuple(by_index[index] for index in range(expected_episodes))
    expected_start = 0
    tasks: set[str] = set()
    fingerprints: set[str] = set()
    for record in ordered:
        if (
            record.dataset_from_index != expected_start
            or record.dataset_to_index != record.dataset_from_index + record.length
        ):
            raise AdmissionError(f"episode {record.episode_index} has a noncontiguous dataset range")
        if record.source_fingerprint != expected_fingerprints[record.episode_index]:
            raise AdmissionError(f"episode {record.episode_index} source fingerprint does not match manifest")
        if record.source_fingerprint in fingerprints:
            raise AdmissionError(f"episode {record.episode_index} has a duplicate source fingerprint")
        expected_start = record.dataset_to_index
        tasks.add(record.task)
        fingerprints.add(record.source_fingerprint)
    if len(tasks) != 1:
        raise AdmissionError("episodes.jsonl must declare one common nonempty task")
    return ordered


def _load_frame_labels(
    path: Path,
    *,
    dataset_name: str,
    episode_lengths: tuple[int, ...],
    source_bytes: bytes,
) -> tuple[np.ndarray, ...]:
    payload = _require_mapping(_loads_json_bytes(source_bytes, path), "tristate_labels.json")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise AdmissionError("tristate labels datasets must contain exactly one entry")
    dataset = _require_mapping(datasets[0], "tristate labels dataset")
    if dataset.get("dataset_name") != dataset_name:
        raise AdmissionError(f"tristate labels dataset_name must equal batch id {dataset_name!r}")
    episode_values = dataset.get("episodes")
    if not isinstance(episode_values, list) or len(episode_values) != len(episode_lengths):
        raise AdmissionError(f"tristate labels must contain exactly {len(episode_lengths)} episodes")
    by_index: dict[int, dict[str, Any]] = {}
    for position, raw_episode in enumerate(episode_values):
        episode = _require_mapping(raw_episode, f"label episode record {position}")
        episode_index = _strict_nonnegative_int(
            episode.get("episode_index"),
            f"label episode record {position} episode_index",
        )
        if episode_index in by_index:
            raise AdmissionError(f"duplicate label episode {episode_index}")
        by_index[episode_index] = episode
    if set(by_index) != set(range(len(episode_lengths))):
        raise AdmissionError(f"label episode indices must be exactly 0..{len(episode_lengths) - 1}")

    labels: list[np.ndarray] = []
    for episode_index, expected_length in enumerate(episode_lengths):
        frame_states = by_index[episode_index].get("frame_states")
        if not isinstance(frame_states, list):
            raise AdmissionError(f"batch {dataset_name} episode {episode_index} labels must be a list")
        if len(frame_states) != expected_length:
            raise AdmissionError(
                f"batch {dataset_name} episode {episode_index} label length mismatch: "
                f"expected {expected_length}, got {len(frame_states)}"
            )
        terminal_frames: list[int] = []
        for frame_index, frame_state in enumerate(frame_states):
            if frame_state is None:
                raise AdmissionError(f"batch {dataset_name} episode {episode_index} frame {frame_index} label is null")
            if type(frame_state) is not int or frame_state not in {-1, 0, 1, 2}:
                raise AdmissionError(
                    f"batch {dataset_name} episode {episode_index} "
                    f"frame {frame_index} has invalid label {frame_state!r}"
                )
            if frame_state == 2:
                terminal_frames.append(frame_index)
        if len(terminal_frames) > 1:
            raise AdmissionError(f"batch {dataset_name} episode {episode_index} has more than one label 2")
        if terminal_frames and terminal_frames[0] != expected_length - 1:
            raise AdmissionError(
                f"batch {dataset_name} episode {episode_index} frame "
                f"{terminal_frames[0]} label 2 is not on the final frame"
            )
        labels.append(np.asarray(frame_states, dtype=np.int8))
    return tuple(labels)


def _expert_masks(
    path: Path,
    episode_lengths: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    payload = _require_mapping(_read_json(path), "expert_frame_index")
    records = payload.get("episodes")
    if not isinstance(records, list):
        raise AdmissionError("expert_frame_index episodes must be a list")
    masks = [np.zeros(length, dtype=np.bool_) for length in episode_lengths]
    seen: set[int] = set()
    for position, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"expert episode record {position}")
        episode_index = _strict_nonnegative_int(
            record.get("episode_index"),
            f"expert episode record {position} episode_index",
        )
        if episode_index >= len(masks):
            raise AdmissionError(f"expert episode {episode_index} is out of range")
        if episode_index in seen:
            raise AdmissionError(f"duplicate expert episode {episode_index}")
        seen.add(episode_index)
        segments = record.get("segments")
        if not isinstance(segments, list):
            raise AdmissionError(f"expert episode {episode_index} segments must be a list")
        for segment_index, raw_segment in enumerate(segments):
            segment = _require_mapping(
                raw_segment,
                f"expert episode {episode_index} segment {segment_index}",
            )
            start = _strict_nonnegative_int(
                segment.get("start_frame_index"),
                f"expert episode {episode_index} segment {segment_index} start_frame_index",
            )
            end = _strict_nonnegative_int(
                segment.get("end_frame_index"),
                f"expert episode {episode_index} segment {segment_index} end_frame_index",
            )
            if start > end or end >= masks[episode_index].size:
                raise AdmissionError(f"invalid expert segment in episode {episode_index}: [{start},{end}]")
            masks[episode_index][start : end + 1] = True
    return tuple(masks)


def _validate_parquet_episode(
    table: pa.Table,
    record: _EpisodeRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"episode {record.episode_index}"
    if table.num_rows != record.length:
        raise AdmissionError(f"{prefix} parquet row count mismatch: expected {record.length}, got {table.num_rows}")
    if tuple(table.column_names) != REQUIRED_COLUMNS:
        raise AdmissionError(f"{prefix} parquet required columns are missing or reordered")
    state = _fixed_float32_matrix(table["observation.state"], "observation.state", prefix)
    action = _fixed_float32_matrix(table["action"], "action", prefix)
    _validate_index_column(
        table["frame_index"],
        "frame_index",
        np.arange(record.length, dtype=np.int64),
        prefix,
    )
    _validate_index_column(
        table["episode_index"],
        "episode_index",
        np.full(record.length, record.episode_index, dtype=np.int64),
        prefix,
    )
    _validate_index_column(
        table["index"],
        "index",
        np.arange(
            record.dataset_from_index,
            record.dataset_to_index,
            dtype=np.int64,
        ),
        prefix,
    )
    _validate_index_column(
        table["task_index"],
        "task_index",
        np.zeros(record.length, dtype=np.int64),
        prefix,
    )

    intervention_column = table["intervention"]
    if intervention_column.type != pa.bool_():
        raise AdmissionError(f"{prefix} intervention column must have bool type")
    if intervention_column.null_count:
        raise AdmissionError(f"{prefix} intervention column contains null")
    intervention = np.asarray(
        intervention_column.combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    )

    control_mode = table["control_mode"]
    if control_mode.type != pa.int64():
        raise AdmissionError(f"{prefix} control_mode column must have int64 type")
    if control_mode.null_count:
        raise AdmissionError(f"{prefix} control_mode column contains null")
    control_mode.combine_chunks().to_numpy(zero_copy_only=False)
    return state, action, intervention


def _fixed_float32_matrix(
    column: pa.ChunkedArray,
    name: str,
    prefix: str,
) -> np.ndarray:
    column_type = column.type
    if (
        not pa.types.is_fixed_size_list(column_type)
        or column_type.list_size != 16
        or column_type.value_type != pa.float32()
    ):
        raise AdmissionError(f"{prefix} {name} must be fixed_size_list float32 width 16")
    if column.null_count:
        raise AdmissionError(f"{prefix} {name} contains null rows")
    combined = column.combine_chunks()
    if combined.values.null_count:
        raise AdmissionError(f"{prefix} {name} contains null values")
    values = np.asarray(
        combined.values.to_numpy(zero_copy_only=False),
        dtype=np.float32,
    ).reshape(len(combined), 16)
    if not np.isfinite(values).all():
        raise AdmissionError(f"{prefix} {name} values must be finite")
    return values


def _validate_index_column(
    column: pa.ChunkedArray,
    name: str,
    expected: np.ndarray,
    prefix: str,
) -> None:
    if column.type != pa.int64():
        raise AdmissionError(f"{prefix} {name} column must have int64 type")
    if column.null_count:
        raise AdmissionError(f"{prefix} {name} column contains null")
    actual = np.asarray(
        column.combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if not np.array_equal(actual, expected):
        raise AdmissionError(f"{prefix} {name} values do not match metadata")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdmissionError(f"{name} must be an object")
    return value


def _strict_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or "\x00" in value:
        raise AdmissionError(f"{name} must be a nonempty trimmed string")
    return value


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AdmissionError(f"{name} must be a nonnegative integer")
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    value = _strict_nonnegative_int(value, name)
    if value == 0:
        raise AdmissionError(f"{name} must be a positive integer")
    return value


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise AdmissionError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _is_exact_shape(value: Any, expected: tuple[int, ...]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(type(item) is int for item in value)
        and tuple(value) == expected
    )


def _sha256_file(path: Path) -> str:
    try:
        return identity.sha256_file(path)
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"cannot hash file {path}") from exc
