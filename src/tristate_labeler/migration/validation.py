"""Read-only validation for a built HIL migration batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import threading
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from tristate_labeler.dataset import load_dataset

from .ledger import ManifestError, _validate_manifest


class BatchValidationError(RuntimeError):
    """A built migration batch does not satisfy the publication contract."""


VIDEO_ROLES = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
METADATA_FILES = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/expert_frame_index.json",
)
OPTIONAL_FILES = frozenset(
    {"migration_manifest.json", "READY", "meta/tristate_labels.json"}
)
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_FFPROBE_BYTES = 1024 * 1024
FFPROBE_TIMEOUT_SECONDS = 30.0
_PIPE_READ_BYTES = 64 * 1024
_HASH_READ_BYTES = 8 * 1024 * 1024
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_REQUIRED_INDEX_COLUMNS = ("episode_index", "frame_index", "index", "task_index")


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]: ...


class FfprobeOutputLimitError(RuntimeError):
    """The ffprobe process produced more stdout than validation permits."""


class SubprocessRunner:
    def __init__(
        self,
        *,
        timeout_seconds: float = FFPROBE_TIMEOUT_SECONDS,
        max_stdout_bytes: int = MAX_FFPROBE_BYTES,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if (
            isinstance(max_stdout_bytes, bool)
            or not isinstance(max_stdout_bytes, int)
            or max_stdout_bytes <= 0
        ):
            raise ValueError("max_stdout_bytes must be a positive integer")
        self._timeout_seconds = float(timeout_seconds)
        self._max_stdout_bytes = max_stdout_bytes

    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = list(argv)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
            process.kill()
            process.wait()
            raise RuntimeError("ffprobe stdout pipe was not created")

        stdout_pipe = process.stdout
        output = bytearray()
        limit_exceeded = threading.Event()
        reader_errors: list[BaseException] = []

        def read_bounded_stdout() -> None:
            try:
                while len(output) <= self._max_stdout_bytes:
                    remaining = self._max_stdout_bytes + 1 - len(output)
                    chunk = stdout_pipe.read(min(_PIPE_READ_BYTES, remaining))
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > self._max_stdout_bytes:
                        limit_exceeded.set()
                        try:
                            process.kill()
                        except OSError:
                            pass
                        break
            except (OSError, ValueError) as exc:
                reader_errors.append(exc)
            finally:
                try:
                    stdout_pipe.close()
                except OSError:
                    pass

        reader = threading.Thread(
            target=read_bounded_stdout,
            name=f"ffprobe-stdout-{process.pid}",
            daemon=True,
        )
        reader.start()
        timeout_error: subprocess.TimeoutExpired | None = None
        try:
            try:
                returncode = process.wait(timeout=self._timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                timeout_error = exc
                try:
                    process.kill()
                except OSError:
                    pass
                returncode = process.wait()
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()
            reader.join(timeout=5)
            if reader.is_alive():
                try:
                    stdout_pipe.close()
                except OSError:
                    pass
                reader.join(timeout=5)

        if reader.is_alive():
            raise RuntimeError("ffprobe stdout reader did not terminate")
        if timeout_error is not None:
            raise subprocess.TimeoutExpired(command, self._timeout_seconds)
        if limit_exceeded.is_set():
            raise FfprobeOutputLimitError(
                f"ffprobe stdout exceeded {self._max_stdout_bytes} bytes"
            )
        if reader_errors:
            raise RuntimeError(
                f"could not read ffprobe stdout: {reader_errors[0].__class__.__name__}"
            ) from reader_errors[0]
        stdout = bytes(output).decode("utf-8", errors="strict")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr="",
        )


_DEFAULT_RUNNER = SubprocessRunner()


@dataclass(frozen=True)
class ValidationReport:
    episode_count: int
    frame_count: int
    task_count: int
    video_count: int
    parquet_count: int
    ffprobe_count: int
    ffprobe_frame_count_unavailable: int
    labeler_compatible: bool
    full_batch_validation: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible validation summary."""

        return {
            "status": "full_batch_validation_passed",
            "scope": "structure, metadata, parquet indices, ffprobe, and labeler loader",
            "full_batch_validation": self.full_batch_validation,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "task_count": self.task_count,
            "video_count": self.video_count,
            "parquet_count": self.parquet_count,
            "ffprobe_count": self.ffprobe_count,
            "ffprobe_frame_count_unavailable": self.ffprobe_frame_count_unavailable,
            "ffprobe_validated": self.ffprobe_count == self.video_count,
            "labeler_compatible": self.labeler_compatible,
            "checked_counts": {
                "episodes": self.episode_count,
                "frames": self.frame_count,
                "tasks": self.task_count,
                "videos": self.video_count,
                "parquets": self.parquet_count,
            },
        }

    @property
    def summary(self) -> dict[str, object]:
        return self.as_dict()


@dataclass(frozen=True)
class _FileStamp:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _Metadata:
    info: dict[str, object]
    tasks: tuple[dict[str, object], ...]
    episodes: tuple[dict[str, object], ...]
    stats: tuple[dict[str, object], ...]
    expert: dict[str, object]
    task_by_name: dict[str, int]
    episode_lengths: tuple[int, ...]
    episode_task_indices: tuple[int, ...]
    total_frames: int
    video_shapes: dict[str, tuple[int, int, int]]
    feature_specs: dict[str, "_FeatureSpec"]


@dataclass(frozen=True)
class _FeatureSpec:
    dtype: str
    shape: tuple[int, ...]


def _stamp(path: Path, *, expected: str | None = None) -> _FileStamp:
    try:
        details = path.lstat()
    except (OSError, ValueError) as exc:
        raise BatchValidationError(
            f"could not inspect batch path: {exc.__class__.__name__}"
        ) from exc
    if stat.S_ISLNK(details.st_mode):
        raise BatchValidationError("batch paths must not be symbolic links")
    if expected == "directory" and not stat.S_ISDIR(details.st_mode):
        raise BatchValidationError("required batch directory is not a real directory")
    if expected == "file" and not stat.S_ISREG(details.st_mode):
        raise BatchValidationError("batch core files must be real regular files")
    return _FileStamp(
        device=int(details.st_dev),
        inode=int(details.st_ino),
        mode=int(details.st_mode),
        size=int(details.st_size),
        mtime_ns=int(details.st_mtime_ns),
        ctime_ns=int(details.st_ctime_ns),
    )


def _capture_tree(root: Path) -> dict[str, _FileStamp]:
    snapshot = {".": _stamp(root, expected="directory")}
    try:
        walker = os.walk(root, topdown=True, followlinks=False)
        for current, directories, files in walker:
            current_path = Path(current)
            for name in sorted((*directories, *files)):
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                details = _stamp(path)
                if name in directories and not stat.S_ISDIR(details.mode):
                    raise BatchValidationError("batch directory tree contains a non-directory")
                if name in files and not stat.S_ISREG(details.mode):
                    raise BatchValidationError("batch directory tree contains a non-regular file")
                snapshot[relative] = details
    except BatchValidationError:
        raise
    except OSError as exc:
        raise BatchValidationError(
            f"could not enumerate batch structure: {exc.__class__.__name__}"
        ) from exc
    return snapshot


def _expected_layout(expected_episodes: int) -> tuple[set[str], set[str]]:
    directories = {
        "data",
        "data/chunk-000",
        "videos",
        "videos/chunk-000",
        "meta",
        *(f"videos/chunk-000/{role}" for role in VIDEO_ROLES),
    }
    files = {
        *METADATA_FILES,
        *(
            f"data/chunk-000/episode_{index:06d}.parquet"
            for index in range(expected_episodes)
        ),
        *(
            f"videos/chunk-000/{role}/episode_{index:06d}.mp4"
            for role in VIDEO_ROLES
            for index in range(expected_episodes)
        ),
    }
    return directories, files


def _validate_layout(
    root: Path, snapshot: Mapping[str, _FileStamp], expected_episodes: int
) -> None:
    expected_directories, expected_files = _expected_layout(expected_episodes)
    actual_directories = {
        name
        for name, details in snapshot.items()
        if name != "." and stat.S_ISDIR(details.mode)
    }
    actual_files = {
        name
        for name, details in snapshot.items()
        if name != "." and stat.S_ISREG(details.mode)
    }
    if actual_directories != expected_directories:
        raise BatchValidationError("batch directory layout is not the fixed chunk-000 layout")
    missing = expected_files - actual_files
    foreign = actual_files - expected_files - OPTIONAL_FILES
    if missing:
        if any(path.startswith("videos/") for path in missing):
            raise BatchValidationError(
                f"expected {expected_episodes * len(VIDEO_ROLES)} videos"
            )
        raise BatchValidationError("batch core file set is incomplete")
    if foreign:
        raise BatchValidationError("batch contains temporary, misnumbered, or foreign files")
    if not actual_files <= expected_files | OPTIONAL_FILES:
        raise BatchValidationError("batch contains unsupported files")
    # Bind each required path lexically beneath the validated root.
    root_text = os.path.abspath(os.fspath(root))
    for relative in expected_directories | expected_files:
        candidate = os.path.abspath(os.fspath(root / Path(relative)))
        if os.path.commonpath((root_text, candidate)) != root_text:
            raise BatchValidationError("batch core path escapes the dataset root")


def _same_file(before: _FileStamp, after: _FileStamp) -> bool:
    # Windows can expose a directory-entry change time through lstat() while
    # fstat() on the same handle reports the file creation time. Bind handle
    # reads using the portable identity/size/mtime contract instead.
    return (
        before.device,
        before.inode,
        before.mode,
        before.size,
        before.mtime_ns,
    ) == (
        after.device,
        after.inode,
        after.mode,
        after.size,
        after.mtime_ns,
    )


def _same_path_stamp(before: _FileStamp, after: _FileStamp) -> bool:
    return before == after


def _read_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    before = _stamp(path, expected="file")
    if before.size > maximum:
        raise BatchValidationError(f"{label} exceeds the validation size limit")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            opened_stamp = _FileStamp(
                int(opened.st_dev),
                int(opened.st_ino),
                int(opened.st_mode),
                int(opened.st_size),
                int(opened.st_mtime_ns),
                int(opened.st_ctime_ns),
            )
            if not _same_file(before, opened_stamp):
                raise BatchValidationError(f"{label} changed before it was read")
            payload = stream.read(maximum + 1)
            final_open = os.fstat(stream.fileno())
        if len(payload) > maximum:
            raise BatchValidationError(f"{label} exceeds the validation size limit")
        final_stamp = _FileStamp(
            int(final_open.st_dev),
            int(final_open.st_ino),
            int(final_open.st_mode),
            int(final_open.st_size),
            int(final_open.st_mtime_ns),
            int(final_open.st_ctime_ns),
        )
        if not _same_file(before, final_stamp) or not _same_path_stamp(
            before, _stamp(path, expected="file")
        ):
            raise BatchValidationError(f"{label} changed while it was read")
        return payload
    except BatchValidationError:
        raise
    except OSError as exc:
        raise BatchValidationError(
            f"could not read {label}: {exc.__class__.__name__}"
        ) from exc


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value}")
    return parsed


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _decode_json(text: str, *, label: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise BatchValidationError(f"{label} is not strict JSON") from exc


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    payload = _read_bytes(path, maximum=MAX_METADATA_BYTES, label=label)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BatchValidationError(f"{label} is not valid UTF-8") from exc
    value = _decode_json(text, label=label)
    if not isinstance(value, dict):
        raise BatchValidationError(f"{label} must contain a JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    payload = _read_bytes(path, maximum=MAX_METADATA_BYTES, label=label)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BatchValidationError(f"{label} is not valid UTF-8") from exc
    if not text:
        raise BatchValidationError(f"{label} must not be empty")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise BatchValidationError(f"{label} contains an empty JSONL record")
        value = _decode_json(line, label=f"{label} line {line_number}")
        if not isinstance(value, dict):
            raise BatchValidationError(f"{label} records must be JSON objects")
        records.append(value)
    if not records:
        raise BatchValidationError(f"{label} must not be empty")
    return tuple(records)


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BatchValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchValidationError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BatchValidationError(f"{label} must be a positive finite number")
    return result


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BatchValidationError(f"{label} must be a nonempty trimmed string")
    return value


def _exact_index_records(
    records: Sequence[dict[str, object]], *, expected: int, label: str
) -> tuple[dict[str, object], ...]:
    if len(records) != expected:
        raise BatchValidationError(f"{label} count mismatch")
    by_index: dict[int, dict[str, object]] = {}
    for record in records:
        index = _integer(record.get("episode_index"), label=f"{label} episode_index")
        if index in by_index:
            raise BatchValidationError(f"{label} contains duplicate episode_index")
        by_index[index] = record
    if set(by_index) != set(range(expected)):
        raise BatchValidationError(f"{label} episode indices must be contiguous")
    return tuple(by_index[index] for index in range(expected))


def _arrow_type_for_dtype(dtype: str) -> pa.DataType | None:
    factories = {
        "bool": pa.bool_,
        "int8": pa.int8,
        "int16": pa.int16,
        "int32": pa.int32,
        "int64": pa.int64,
        "uint8": pa.uint8,
        "uint16": pa.uint16,
        "uint32": pa.uint32,
        "uint64": pa.uint64,
        "float16": pa.float16,
        "float32": pa.float32,
        "float64": pa.float64,
        "string": pa.string,
    }
    factory = factories.get(dtype)
    return None if factory is None else factory()


def _validate_info(
    info: dict[str, object], *, expected_episodes: int
) -> tuple[
    int,
    int,
    dict[str, tuple[int, int, int]],
    dict[str, _FeatureSpec],
]:
    if info.get("codebase_version") != "v2.1":
        raise BatchValidationError("info codebase_version must be v2.1")
    if info.get("data_path") != "data/chunk-000/episode_{episode_index:06d}.parquet":
        raise BatchValidationError("info data_path template is not the fixed chunk-000 template")
    if info.get("video_path") != "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4":
        raise BatchValidationError("info video_path template is not the fixed chunk-000 template")
    if _integer(info.get("total_episodes"), label="info total_episodes", minimum=1) != expected_episodes:
        raise BatchValidationError("info total_episodes mismatch")
    total_frames = _integer(info.get("total_frames"), label="info total_frames", minimum=1)
    total_tasks = _integer(info.get("total_tasks"), label="info total_tasks", minimum=1)
    if _integer(info.get("total_videos"), label="info total_videos", minimum=1) != expected_episodes * len(VIDEO_ROLES):
        raise BatchValidationError("info total_videos mismatch")
    if _integer(info.get("total_chunks"), label="info total_chunks", minimum=1) != 1:
        raise BatchValidationError("info total_chunks must be one")
    if _integer(info.get("chunks_size"), label="info chunks_size", minimum=1) != expected_episodes:
        raise BatchValidationError("info chunks_size mismatch")
    if info.get("splits") != {"train": f"0:{expected_episodes}"}:
        raise BatchValidationError("info splits must cover the exact batch")
    _positive_number(info.get("fps"), label="info fps")
    _nonempty_string(info.get("robot_type"), label="info robot_type")
    features = info.get("features")
    if not isinstance(features, dict):
        raise BatchValidationError("info features must be an object")
    feature_specs: dict[str, _FeatureSpec] = {}
    for name, raw_feature in features.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise BatchValidationError("info feature names must be nonempty trimmed strings")
        if not isinstance(raw_feature, dict):
            raise BatchValidationError(f"info feature {name} must be an object")
        dtype = raw_feature.get("dtype")
        if not isinstance(dtype, str) or not dtype or dtype != dtype.strip():
            raise BatchValidationError(f"info feature {name} dtype is invalid")
        shape = raw_feature.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
        ):
            raise BatchValidationError(
                f"info feature {name} must have a nonempty positive shape"
            )
        feature_specs[name] = _FeatureSpec(dtype=dtype, shape=tuple(shape))
    actual_video_roles = {
        key
        for key, feature in feature_specs.items()
        if feature.dtype == "video"
    }
    if actual_video_roles != set(VIDEO_ROLES):
        raise BatchValidationError("info must declare exactly the three required video keys")
    for name, feature in feature_specs.items():
        if feature.dtype != "video" and _arrow_type_for_dtype(feature.dtype) is None:
            raise BatchValidationError(f"info feature {name} dtype is unsupported")
    shapes: dict[str, tuple[int, int, int]] = {}
    for role in VIDEO_ROLES:
        shape = feature_specs[role].shape
        if len(shape) != 3:
            raise BatchValidationError(f"video feature {role} must have positive [H, W, C] shape")
        shapes[role] = (shape[0], shape[1], shape[2])
    for name in _REQUIRED_INDEX_COLUMNS:
        feature = feature_specs.get(name)
        arrow_type = None if feature is None else _arrow_type_for_dtype(feature.dtype)
        if (
            feature is None
            or feature.shape != (1,)
            or arrow_type is None
            or not pa.types.is_integer(arrow_type)
        ):
            raise BatchValidationError(
                f"info index feature {name} must declare integer dtype and shape [1]"
            )
    return total_frames, total_tasks, shapes, feature_specs


def _validate_tasks(
    records: Sequence[dict[str, object]], *, expected_tasks: int
) -> dict[str, int]:
    if len(records) != expected_tasks:
        raise BatchValidationError("tasks count does not match info total_tasks")
    by_index: dict[int, str] = {}
    for record in records:
        index = _integer(record.get("task_index"), label="task_index")
        task = _nonempty_string(record.get("task"), label="task")
        if index in by_index or task in by_index.values():
            raise BatchValidationError("tasks must have unique indices and strings")
        by_index[index] = task
    if set(by_index) != set(range(expected_tasks)):
        raise BatchValidationError("task indices must be contiguous")
    return {task: index for index, task in by_index.items()}


def _validate_episodes(
    records: Sequence[dict[str, object]],
    *,
    expected_episodes: int,
    task_by_name: Mapping[str, int],
) -> tuple[tuple[dict[str, object], ...], tuple[int, ...], tuple[int, ...], int]:
    ordered = _exact_index_records(records, expected=expected_episodes, label="episodes metadata")
    lengths: list[int] = []
    task_indices: list[int] = []
    fingerprints: set[str] = set()
    running = 0
    for target_index, record in enumerate(ordered):
        length = _integer(record.get("length"), label="episode length", minimum=1)
        start = _integer(record.get("dataset_from_index"), label="dataset_from_index")
        end = _integer(record.get("dataset_to_index"), label="dataset_to_index", minimum=1)
        if start != running or end != running + length:
            raise BatchValidationError("episode dataset frame ranges are not contiguous")
        tasks = record.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise BatchValidationError("episode tasks must contain exactly one task string")
        task = _nonempty_string(tasks[0], label="episode task")
        if task not in task_by_name:
            raise BatchValidationError("episode task is absent from tasks metadata")
        for field in ("source_host", "source_dataset", "source_dataset_root"):
            _nonempty_string(record.get(field), label=f"episode {field}")
        source_index = _integer(record.get("source_episode_index"), label="source_episode_index")
        _integer(record.get("source_completed_ns"), label="source_completed_ns")
        fingerprint = record.get("source_fingerprint")
        if not isinstance(fingerprint, str) or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise BatchValidationError("episode source_fingerprint must be a SHA-256 digest")
        if fingerprint in fingerprints:
            raise BatchValidationError("episode source_fingerprint values must be unique")
        fingerprints.add(fingerprint)
        dataset_name = str(record["source_dataset"])
        dataset_root = str(record["source_dataset_root"])
        if "/" in dataset_name or "\\" in dataset_name or not dataset_root.startswith("/"):
            raise BatchValidationError("episode source dataset provenance is invalid")
        if dataset_root.rstrip("/").rsplit("/", 1)[-1] != dataset_name or source_index < 0:
            raise BatchValidationError("episode source dataset provenance is inconsistent")
        lengths.append(length)
        task_indices.append(task_by_name[task])
        running = end
    return ordered, tuple(lengths), tuple(task_indices), running


def _validate_expert_records(
    expert: dict[str, object], *, lengths: Sequence[int], expected_episodes: int
) -> None:
    if set(expert) != {"episodes"}:
        raise BatchValidationError("expert metadata must contain only episodes")
    records = expert.get("episodes")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise BatchValidationError("expert episodes must be a list of objects")
    ordered = _exact_index_records(records, expected=expected_episodes, label="expert metadata")
    for index, record in enumerate(ordered):
        segments = record.get("segments", [])
        if not isinstance(segments, list):
            raise BatchValidationError("expert segments must be a list")
        for segment in segments:
            if not isinstance(segment, dict):
                raise BatchValidationError("expert segments must be objects")
            start = _integer(segment.get("start_frame_index"), label="expert segment start")
            end = _integer(segment.get("end_frame_index"), label="expert segment end")
            if start > end or end >= lengths[index]:
                raise BatchValidationError("expert segment is outside its episode")


def _load_and_validate_metadata(root: Path, *, expected_episodes: int) -> _Metadata:
    info = _load_json(root / "meta/info.json", label="info.json")
    tasks = _load_jsonl(root / "meta/tasks.jsonl", label="tasks.jsonl")
    episodes = _load_jsonl(root / "meta/episodes.jsonl", label="episodes.jsonl")
    stats = _load_jsonl(root / "meta/episodes_stats.jsonl", label="episodes_stats.jsonl")
    expert = _load_json(root / "meta/expert_frame_index.json", label="expert_frame_index.json")
    if (root / "meta/tristate_labels.json").exists():
        _load_json(root / "meta/tristate_labels.json", label="tristate_labels.json")
    total_frames, total_tasks, video_shapes, feature_specs = _validate_info(
        info, expected_episodes=expected_episodes
    )
    task_by_name = _validate_tasks(tasks, expected_tasks=total_tasks)
    ordered_episodes, lengths, task_indices, calculated_frames = _validate_episodes(
        episodes,
        expected_episodes=expected_episodes,
        task_by_name=task_by_name,
    )
    if calculated_frames != total_frames:
        raise BatchValidationError("info total_frames does not match episode metadata")
    ordered_stats = _exact_index_records(
        stats, expected=expected_episodes, label="episode statistics"
    )
    _validate_expert_records(
        expert, lengths=lengths, expected_episodes=expected_episodes
    )
    return _Metadata(
        info=info,
        tasks=tasks,
        episodes=ordered_episodes,
        stats=ordered_stats,
        expert=expert,
        task_by_name=task_by_name,
        episode_lengths=lengths,
        episode_task_indices=task_indices,
        total_frames=total_frames,
        video_shapes=video_shapes,
        feature_specs=feature_specs,
    )


def _read_parquet(path: Path) -> pa.Table:
    before = _stamp(path, expected="file")
    try:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        if len(names) != len(set(names)):
            raise BatchValidationError("parquet contains duplicate column names")
        table = parquet.read()
    except BatchValidationError:
        raise
    except Exception as exc:
        raise BatchValidationError(
            f"could not read episode parquet: {exc.__class__.__name__}"
        ) from exc
    if not _same_path_stamp(before, _stamp(path, expected="file")):
        raise BatchValidationError("episode parquet changed while it was read")
    return table


def _array_contains_null(array: pa.Array) -> bool:
    if array.null_count:
        return True
    if pa.types.is_fixed_size_list(array.type) or pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return _array_contains_null(array.values)
    if pa.types.is_struct(array.type):
        return any(_array_contains_null(array.field(index)) for index in range(array.type.num_fields))
    return False


def _validate_feature_schema(table: pa.Table, metadata: _Metadata) -> None:
    expected = {
        name: spec
        for name, spec in metadata.feature_specs.items()
        if spec.dtype != "video"
    }
    if set(table.column_names) != set(expected):
        raise BatchValidationError(
            "parquet schema columns do not match info feature declarations"
        )
    for name, feature in expected.items():
        column = table[name]
        try:
            array = column.combine_chunks()
        except Exception as exc:
            raise BatchValidationError(
                f"parquet feature {name} could not be combined: {exc.__class__.__name__}"
            ) from exc
        if _array_contains_null(array):
            raise BatchValidationError(f"parquet feature {name} contains null values")
        actual_type = array.type
        if pa.types.is_fixed_size_list(actual_type):
            element_type = actual_type.value_type
            element_count = actual_type.list_size
        else:
            element_type = actual_type
            element_count = 1
        expected_type = _arrow_type_for_dtype(feature.dtype)
        expected_count = math.prod(feature.shape)
        if element_type != expected_type or element_count != expected_count:
            raise BatchValidationError(
                f"parquet feature {name} schema does not match info metadata"
            )


def _validate_parquets(root: Path, metadata: _Metadata) -> int:
    global_start = 0
    for episode_index, (length, task_index) in enumerate(
        zip(metadata.episode_lengths, metadata.episode_task_indices, strict=True)
    ):
        path = root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        table = _read_parquet(path)
        if table.num_rows != length:
            raise BatchValidationError("parquet row count does not match episode length")
        if len(table.column_names) != len(set(table.column_names)):
            raise BatchValidationError("parquet contains duplicate column names")
        for name in _REQUIRED_INDEX_COLUMNS:
            position = table.schema.get_field_index(name)
            if position < 0:
                raise BatchValidationError(f"parquet is missing required index column {name}")
            column = table.column(position)
            if not pa.types.is_integer(column.type):
                raise BatchValidationError(f"parquet index column {name} must be integer")
            if column.null_count:
                raise BatchValidationError(f"parquet index column {name} contains nulls")
        _validate_feature_schema(table, metadata)
        expected_values = {
            "episode_index": [episode_index] * length,
            "frame_index": list(range(length)),
            "index": list(range(global_start, global_start + length)),
            "task_index": [task_index] * length,
        }
        for name, expected in expected_values.items():
            if table[name].to_pylist() != expected:
                label = "global index" if name == "index" else name
                raise BatchValidationError(f"parquet {label} is not contiguous or aligned")
        global_start += length
        del table
    if global_start != metadata.total_frames:
        raise BatchValidationError("parquet rows do not match info total_frames")
    return global_start


def _validate_with_labeler(root: Path, metadata: _Metadata, expected_episodes: int) -> None:
    try:
        dataset = load_dataset(root)
    except Exception as exc:
        raise BatchValidationError(
            f"labeler dataset loader rejected the batch: {exc.__class__.__name__}"
        ) from exc
    try:
        if dataset.root != root.resolve():
            raise BatchValidationError("labeler loader root mismatch")
        if float(dataset.fps) != float(metadata.info["fps"]):
            raise BatchValidationError("labeler loader fps mismatch")
        if dataset.total_episodes != expected_episodes or len(dataset.episodes) != expected_episodes:
            raise BatchValidationError("labeler loader episode count mismatch")
        if dataset.total_frames != metadata.total_frames:
            raise BatchValidationError("labeler loader frame count mismatch")
        if set(dataset.video_keys) != set(VIDEO_ROLES) or len(dataset.video_keys) != len(VIDEO_ROLES):
            raise BatchValidationError("labeler loader video key mismatch")
        running = 0
        for expected_index, (loaded, length) in enumerate(
            zip(dataset.episodes, metadata.episode_lengths, strict=True)
        ):
            if (
                loaded.episode_index != expected_index
                or loaded.length != length
                or loaded.dataset_from_index != running
                or loaded.dataset_to_index != running + length
            ):
                raise BatchValidationError("labeler loader episode metadata mismatch")
            if set(loaded.videos) != set(VIDEO_ROLES):
                raise BatchValidationError("labeler loader episode video references mismatch")
            for role in VIDEO_ROLES:
                expected_path = Path(
                    f"videos/chunk-000/{role}/episode_{expected_index:06d}.mp4"
                )
                reference = loaded.videos[role]
                if reference.relative_path != expected_path:
                    raise BatchValidationError("labeler loader video path mismatch")
                if (
                    reference.key != role
                    or reference.chunk_index != 0
                    or reference.file_index != expected_index
                ):
                    raise BatchValidationError("labeler loader video reference mismatch")
            running += length
    except BatchValidationError:
        raise
    except Exception as exc:
        raise BatchValidationError(
            f"labeler dataset loader returned invalid data: {exc.__class__.__name__}"
        ) from exc


def _ffprobe_executable(ffprobe: Path | str) -> str:
    try:
        value = os.fspath(ffprobe)
    except TypeError as exc:
        raise BatchValidationError("ffprobe must be a nonempty path or string") from exc
    if isinstance(value, bytes) or not isinstance(value, str) or not value.strip() or "\0" in value:
        raise BatchValidationError("ffprobe must be a nonempty path or string")
    return value


def _probe_video(
    path: Path,
    *,
    executable: str,
    shape: tuple[int, int, int],
    expected_frames: int,
    runner: CommandRunner,
) -> bool:
    before = _stamp(path, expected="file")
    argv = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=codec_type,width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = runner.run(argv)
    except Exception as exc:
        raise BatchValidationError(
            f"could not run ffprobe: {exc.__class__.__name__}"
        ) from exc
    if not _same_path_stamp(before, _stamp(path, expected="file")):
        raise BatchValidationError("video changed while ffprobe was running")
    try:
        returncode = completed.returncode
        stdout = completed.stdout
    except Exception as exc:
        raise BatchValidationError("ffprobe runner returned a malformed result") from exc
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
        status = returncode if isinstance(returncode, int) and not isinstance(returncode, bool) else "invalid"
        raise BatchValidationError(f"ffprobe failed with exit status {status}")
    if not isinstance(stdout, str):
        raise BatchValidationError("ffprobe output must be text")
    try:
        encoded = stdout.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise BatchValidationError("ffprobe output is not valid UTF-8 text") from exc
    if len(encoded) > MAX_FFPROBE_BYTES:
        raise BatchValidationError("ffprobe output exceeds the validation size limit")
    value = _decode_json(stdout, label="ffprobe output")
    if not isinstance(value, dict):
        raise BatchValidationError("ffprobe output must be a JSON object")
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise BatchValidationError("ffprobe must report exactly one video stream")
    stream = streams[0]
    if stream.get("codec_type") != "video":
        raise BatchValidationError("ffprobe stream codec_type must be video")
    width = _integer(stream.get("width"), label="ffprobe width", minimum=1)
    height = _integer(stream.get("height"), label="ffprobe height", minimum=1)
    if (height, width) != shape[:2]:
        raise BatchValidationError("ffprobe video dimensions do not match info feature shape")
    raw_frames = stream.get("nb_frames")
    if raw_frames in (None, "N/A"):
        return False
    if isinstance(raw_frames, bool):
        raise BatchValidationError("ffprobe nb_frames is invalid")
    try:
        frames = int(raw_frames)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchValidationError("ffprobe nb_frames is invalid") from exc
    if isinstance(raw_frames, float) and not raw_frames.is_integer():
        raise BatchValidationError("ffprobe nb_frames is invalid")
    if frames <= 0 or str(raw_frames).strip() not in {str(frames), f"{frames}.0"}:
        raise BatchValidationError("ffprobe nb_frames is invalid")
    if frames != expected_frames:
        raise BatchValidationError("ffprobe frame count does not match episode length")
    return True


def _validate_videos(
    root: Path,
    metadata: _Metadata,
    *,
    executable: str,
    runner: CommandRunner,
) -> tuple[int, int]:
    count = 0
    unavailable = 0
    for role in VIDEO_ROLES:
        for episode_index, length in enumerate(metadata.episode_lengths):
            path = root / "videos/chunk-000" / role / f"episode_{episode_index:06d}.mp4"
            if not _probe_video(
                path,
                executable=executable,
                shape=metadata.video_shapes[role],
                expected_frames=length,
                runner=runner,
            ):
                unavailable += 1
            count += 1
    return count, unavailable


def _assert_tree_unchanged(root: Path, before: Mapping[str, _FileStamp]) -> None:
    after = _capture_tree(root)
    if dict(before) != after:
        raise BatchValidationError("batch files changed during validation")


def _manifest_integer(
    manifest: Mapping[str, object], name: str, *, minimum: int = 0
) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BatchValidationError(
            f"migration manifest {name} must be an integer >= {minimum}"
        )
    return value


def _validate_manifest_header(manifest: Mapping[str, object]) -> None:
    tool_version = manifest.get("tool_version")
    if (
        not isinstance(tool_version, str)
        or not tool_version
        or tool_version.strip() != tool_version
    ):
        raise BatchValidationError(
            "migration manifest tool_version must be a nonempty trimmed string"
        )

    created_at = manifest.get("created_at")
    if (
        not isinstance(created_at, str)
        or created_at.strip() != created_at
        or "T" not in created_at
    ):
        raise BatchValidationError(
            "migration manifest created_at must be an ISO timestamp with timezone"
        )
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BatchValidationError(
            "migration manifest created_at must be an ISO timestamp with timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BatchValidationError(
            "migration manifest created_at must be an ISO timestamp with timezone"
        )


def _manifest_posix_path(
    value: object,
    *,
    label: str,
    absolute: bool,
    allow_root: bool = False,
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise BatchValidationError(
            f"migration manifest {label} must be a normalized POSIX path"
        )
    parsed = PurePosixPath(value)
    if absolute:
        prefix_is_valid = value.startswith("/") and not value.startswith("//")
        components = value[1:].split("/")
        root_is_valid = allow_root and value == "/"
    else:
        prefix_is_valid = not value.startswith("/")
        components = value.split("/")
        root_is_valid = False
    if (
        not prefix_is_valid
        or parsed.is_absolute() != absolute
        or parsed.as_posix() != value
        or (
            not root_is_valid
            and any(component in {"", ".", ".."} for component in components)
        )
    ):
        raise BatchValidationError(
            f"migration manifest {label} must be a normalized POSIX path"
        )
    return value


def _expected_source_relative_path(role: str, source_index: int) -> str:
    chunk = source_index // 1000
    if role == "parquet":
        return f"data/chunk-{chunk:03d}/episode_{source_index:06d}.parquet"
    return f"videos/chunk-{chunk:03d}/{role}/episode_{source_index:06d}.mp4"


def _sha256_manifest_file(path: Path) -> tuple[int, str]:
    before = _stamp(path, expected="file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            opened_stamp = _FileStamp(
                int(opened.st_dev),
                int(opened.st_ino),
                int(opened.st_mode),
                int(opened.st_size),
                int(opened.st_mtime_ns),
                int(opened.st_ctime_ns),
            )
            if not _same_file(before, opened_stamp):
                raise BatchValidationError(
                    "migration manifest core file changed before hashing"
                )
            for chunk in iter(lambda: stream.read(_HASH_READ_BYTES), b""):
                digest.update(chunk)
            final_open = os.fstat(stream.fileno())
        final_stamp = _FileStamp(
            int(final_open.st_dev),
            int(final_open.st_ino),
            int(final_open.st_mode),
            int(final_open.st_size),
            int(final_open.st_mtime_ns),
            int(final_open.st_ctime_ns),
        )
        if not _same_file(before, final_stamp) or not _same_path_stamp(
            before, _stamp(path, expected="file")
        ):
            raise BatchValidationError(
                "migration manifest core file changed while hashing"
            )
    except BatchValidationError:
        raise
    except OSError as exc:
        raise BatchValidationError(
            f"could not hash migration manifest core file: {exc.__class__.__name__}"
        ) from exc
    return before.size, digest.hexdigest()


def _validate_source_file_records(
    raw_episodes: object,
    metadata: _Metadata,
) -> tuple[int, int]:
    if not isinstance(raw_episodes, list) or len(raw_episodes) != len(metadata.episodes):
        raise BatchValidationError("migration manifest episodes do not match metadata")
    source_file_count = 0
    source_bytes = 0
    for target_index, (raw, episode) in enumerate(
        zip(raw_episodes, metadata.episodes, strict=True)
    ):
        if not isinstance(raw, dict):
            raise BatchValidationError("migration manifest episode must be an object")
        task = episode["tasks"][0]
        expected_mapping = {
            "target_index": target_index,
            "fingerprint": episode["source_fingerprint"],
            "source_host": episode["source_host"],
            "source_dataset": episode["source_dataset"],
            "source_dataset_root": episode["source_dataset_root"],
            "source_index": episode["source_episode_index"],
            "source_task": task,
            "source_completed_ns": episode["source_completed_ns"],
            "target_task_index": metadata.episode_task_indices[target_index],
            "frame_count": metadata.episode_lengths[target_index],
        }
        if any(raw.get(key) != value for key, value in expected_mapping.items()):
            raise BatchValidationError(
                "migration manifest source mapping does not match episode metadata"
            )
        source_files = raw.get("source_files")
        if not isinstance(source_files, list) or len(source_files) != 4:
            raise BatchValidationError(
                "migration manifest source file mapping must contain four files"
            )
        by_role: dict[str, dict[str, object]] = {}
        for source_file in source_files:
            if not isinstance(source_file, dict):
                raise BatchValidationError(
                    "migration manifest source file records must be objects"
                )
            role = source_file.get("role")
            if not isinstance(role, str) or role in by_role:
                raise BatchValidationError(
                    "migration manifest source file roles must be unique"
                )
            by_role[role] = source_file
        if set(by_role) != {"parquet", *VIDEO_ROLES}:
            raise BatchValidationError(
                "migration manifest source file roles are incomplete"
            )
        for role, source_file in by_role.items():
            if set(source_file) != {
                "role",
                "absolute_path",
                "relative_path",
                "target_path",
                "size",
                "mtime_ns",
                "sha256",
            }:
                raise BatchValidationError(
                    "migration manifest source file record has invalid fields"
                )
            expected_target = (
                f"data/chunk-000/episode_{target_index:06d}.parquet"
                if role == "parquet"
                else f"videos/chunk-000/{role}/episode_{target_index:06d}.mp4"
            )
            if source_file.get("target_path") != expected_target:
                raise BatchValidationError(
                    "migration manifest source file target mapping is invalid"
                )
            source_root = _manifest_posix_path(
                episode["source_dataset_root"],
                label="source dataset root",
                absolute=True,
                allow_root=True,
            )
            absolute_path = _manifest_posix_path(
                source_file.get("absolute_path"),
                label="source file absolute_path",
                absolute=True,
            )
            relative_path = _manifest_posix_path(
                source_file.get("relative_path"),
                label="source file relative_path",
                absolute=False,
            )
            expected_relative = _expected_source_relative_path(
                role, int(episode["source_episode_index"])
            )
            expected_absolute = (
                PurePosixPath(source_root) / PurePosixPath(relative_path)
            ).as_posix()
            if (
                relative_path != expected_relative
                or absolute_path != expected_absolute
            ):
                raise BatchValidationError(
                    "migration manifest source file path does not match its source episode"
                )
            size = source_file.get("size")
            digest = source_file.get("sha256")
            mtime_ns = source_file.get("mtime_ns")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or _FINGERPRINT_RE.fullmatch(digest) is None
                or isinstance(mtime_ns, bool)
                or not isinstance(mtime_ns, int)
                or mtime_ns < 0
            ):
                raise BatchValidationError(
                    "migration manifest source file stat or hash is invalid"
                )
            source_file_count += 1
            source_bytes += size
    return source_file_count, source_bytes


def _validate_publication_manifest(
    root: Path,
    *,
    metadata: _Metadata,
    report: ValidationReport,
    expected_episodes: int,
) -> None:
    manifest = _load_json(
        root / "migration_manifest.json", label="migration_manifest.json"
    )
    try:
        normalized = _validate_manifest(manifest)
    except ManifestError as exc:
        raise BatchValidationError(f"migration manifest is invalid: {exc}") from exc
    _validate_manifest_header(manifest)

    if normalized["batch_id"] != root.name:
        raise BatchValidationError("migration manifest batch_id does not match directory")
    if (
        normalized["episode_count"] != expected_episodes
        or normalized["frame_count"] != metadata.total_frames
    ):
        raise BatchValidationError(
            "migration manifest episode or frame count does not match metadata"
        )
    expected_fingerprints = [
        str(episode["source_fingerprint"]) for episode in metadata.episodes
    ]
    if normalized["episode_fingerprints"] != expected_fingerprints:
        raise BatchValidationError(
            "migration manifest fingerprints do not match episode metadata"
        )
    for target_index, (record, episode) in enumerate(
        zip(normalized["episodes"], metadata.episodes, strict=True)
    ):
        expected_mapping = {
            "target_index": target_index,
            "fingerprint": episode["source_fingerprint"],
            "source_host": episode["source_host"],
            "source_dataset_root": episode["source_dataset_root"],
            "source_index": episode["source_episode_index"],
        }
        if record != expected_mapping:
            raise BatchValidationError(
                "migration manifest source mapping does not match metadata"
            )

    _, expected_core_files = _expected_layout(expected_episodes)
    manifest_files = normalized["files"]
    manifest_paths = {str(record["target_path"]) for record in manifest_files}
    if manifest_paths != expected_core_files or len(manifest_files) != len(
        expected_core_files
    ):
        raise BatchValidationError(
            "migration manifest files must exactly cover the 85 core files"
        )
    dataset_bytes = 0
    for record in manifest_files:
        relative = str(record["target_path"])
        actual_size, actual_hash = _sha256_manifest_file(root / Path(relative))
        if actual_size != record["size"] or actual_hash != record["sha256"]:
            raise BatchValidationError(
                "migration manifest core file size or SHA-256 mismatch"
            )
        dataset_bytes += actual_size

    source_file_count, source_bytes = _validate_source_file_records(
        manifest.get("episodes"), metadata
    )
    expected_totals = {
        "dataset_bytes": dataset_bytes,
        "source_bytes": source_bytes,
        "source_file_count": source_file_count,
        "task_count": report.task_count,
        "video_count": report.video_count,
    }
    if any(
        _manifest_integer(
            manifest,
            name,
            minimum=1,
        )
        != expected
        for name, expected in expected_totals.items()
    ):
        raise BatchValidationError(
            "migration manifest dataset or source totals do not match the batch"
        )
    if source_file_count != expected_episodes * 4:
        raise BatchValidationError(
            "migration manifest source file count does not match the batch"
        )
    expected_hosts = sorted(
        {str(episode["source_host"]) for episode in metadata.episodes}
    )
    expected_roots = sorted(
        {str(episode["source_dataset_root"]) for episode in metadata.episodes}
    )
    if (
        manifest.get("source_hosts") != expected_hosts
        or manifest.get("source_dataset_roots") != expected_roots
    ):
        raise BatchValidationError(
            "migration manifest source host or root totals do not match metadata"
        )
    if manifest.get("validation") != report.as_dict():
        raise BatchValidationError(
            "migration manifest validation summary is not the full batch result"
        )


def validate_batch(
    root: Path,
    *,
    expected_episodes: int,
    ffprobe: Path | str,
    runner: CommandRunner = _DEFAULT_RUNNER,
) -> ValidationReport:
    """Validate a complete batch without modifying any batch path."""

    if (
        isinstance(expected_episodes, bool)
        or not isinstance(expected_episodes, int)
        or expected_episodes <= 0
    ):
        raise BatchValidationError("expected_episodes must be a positive integer")
    executable = _ffprobe_executable(ffprobe)
    try:
        root_path = Path(os.path.abspath(os.fspath(root)))
    except (TypeError, ValueError, OSError) as exc:
        raise BatchValidationError("root must be a valid filesystem path") from exc
    before = _capture_tree(root_path)
    _validate_layout(root_path, before, expected_episodes)
    manifest_present = "migration_manifest.json" in before
    ready_present = "READY" in before
    if manifest_present != ready_present:
        raise BatchValidationError(
            "migration manifest and READY must be present together"
        )
    if ready_present and before["READY"].size != 0:
        raise BatchValidationError("READY marker must be an empty regular file")
    metadata = _load_and_validate_metadata(
        root_path, expected_episodes=expected_episodes
    )
    frame_count = _validate_parquets(root_path, metadata)
    _validate_with_labeler(root_path, metadata, expected_episodes)
    ffprobe_count, unavailable = _validate_videos(
        root_path,
        metadata,
        executable=executable,
        runner=runner,
    )
    report = ValidationReport(
        episode_count=expected_episodes,
        frame_count=frame_count,
        task_count=len(metadata.tasks),
        video_count=expected_episodes * len(VIDEO_ROLES),
        parquet_count=expected_episodes,
        ffprobe_count=ffprobe_count,
        ffprobe_frame_count_unavailable=unavailable,
        labeler_compatible=True,
    )
    if manifest_present:
        _validate_publication_manifest(
            root_path,
            metadata=metadata,
            report=report,
            expected_episodes=expected_episodes,
        )
    _assert_tree_unchanged(root_path, before)
    return report
