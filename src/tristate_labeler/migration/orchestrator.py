"""Single-run orchestration for robot-to-inference HIL batch migration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
import copy
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import ContextManager, Protocol, cast
import uuid

from .batch import BuiltBatch, DownloadedEpisode, build_batch
from .ledger import MAX_MANIFEST_BYTES, MigrationLedger, load_valid_ready_manifests
from .models import MigrationConfig, RunStatus, ScanResult, SourceEpisode, SourceFile
from .selection import select_next_batch
from .source import SshSourceScanner
from .transfer import RsyncTransfer, ensure_space
from .validation import ValidationReport, validate_batch


_BATCH_ID_RE = re.compile(r"batch_(\d{6})(?:_[A-Za-z0-9][A-Za-z0-9_-]*)?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SELECTION_MAX_BYTES = 4 * 1024 * 1024
_PENDING_MAX_BYTES = 1024 * 1024
_REUSE_HASH_CHUNK_BYTES = 1024 * 1024
_ROLE_FILENAMES = {
    "parquet": "parquet.parquet",
    "observation.images.top": "top.mp4",
    "observation.images.left_wrist": "left_wrist.mp4",
    "observation.images.right_wrist": "right_wrist.mp4",
}


class OrchestrationError(RuntimeError):
    """A migration run could not preserve the publication contract."""


class MigrationLockedError(OrchestrationError):
    """Another formal migration process already owns the output lock."""


class Scanner(Protocol):
    def scan(self) -> ScanResult: ...

    def revalidate(self, selected: Sequence[SourceEpisode]) -> None: ...


class Transfer(Protocol):
    def fetch(self, source: SourceFile, target: Path) -> None: ...


class Ledger(Protocol):
    def __enter__(self) -> Ledger: ...

    def __exit__(self, *exc_info: object) -> None: ...

    def reconcile_ready(self, ready_root: Path) -> None: ...

    def migrated_fingerprints(self) -> set[str]: ...

    def next_batch_sequence(self) -> int: ...

    def record_published_batch(
        self, manifest: Mapping[str, object], *, manifest_path: Path
    ) -> None: ...


Builder = Callable[..., BuiltBatch]
Validator = Callable[..., ValidationReport]
LedgerFactory = Callable[[Path], ContextManager[Ledger]]
SpaceChecker = Callable[..., None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    available: int
    required: int
    selected: tuple[SourceEpisode, ...] = ()
    batch_id: str | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "available": self.available,
            "required": self.required,
            "batch_id": self.batch_id,
            "dry_run": self.dry_run,
            "selected_fingerprints": [item.fingerprint for item in self.selected],
        }


def make_batch_id(sequence: int, selected: Sequence[SourceEpisode]) -> str:
    """Build the deterministic batch id from its sequence and newest source."""

    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 1 <= sequence <= 999_999
    ):
        raise ValueError("sequence must be an integer from 1 through 999999")
    if not selected:
        raise ValueError("selected must not be empty")
    newest_ns = max(item.completed_ns for item in selected)
    try:
        newest = datetime.fromtimestamp(newest_ns / 1e9, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("selected completion timestamp is outside the supported range") from exc
    return f"batch_{sequence:06d}_{newest:%Y%m%d_%H%M%S}"


def _absolute_lexical(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise OrchestrationError("output path is invalid") from exc


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise OrchestrationError(
            f"could not inspect {label}: {exc.__class__.__name__}"
        ) from exc


def _assert_directory(path: Path, label: str) -> None:
    details = _lstat(path, label)
    if stat.S_ISLNK(details.st_mode):
        raise OrchestrationError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(details.st_mode):
        raise OrchestrationError(f"{label} must be a directory")


def _ensure_safe_directory(path: Path, *, mode: int = 0o700) -> Path:
    """Create a directory chain without following a symlink in that chain."""

    absolute = _absolute_lexical(path)
    anchor = Path(absolute.anchor)
    if not anchor.exists():
        raise OrchestrationError("output path has no existing filesystem anchor")
    _assert_directory(anchor, "filesystem anchor")
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        try:
            details = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=mode)
            except FileExistsError:
                pass
            except OSError as exc:
                raise OrchestrationError(
                    f"could not create output directory: {exc.__class__.__name__}"
                ) from exc
            details = _lstat(current, "output directory")
        except OSError as exc:
            raise OrchestrationError(
                f"could not inspect output directory: {exc.__class__.__name__}"
            ) from exc
        if stat.S_ISLNK(details.st_mode):
            raise OrchestrationError("output directory chain must not contain symlinks")
        if not stat.S_ISDIR(details.st_mode):
            raise OrchestrationError("output directory chain must contain only directories")
    return absolute


def _open_lock_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise OrchestrationError(
            f"could not open migration lock: {exc.__class__.__name__}"
        ) from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise OrchestrationError("migration lock must be a regular file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def exclusive_lock(path: Path):
    """Acquire a cross-platform, nonblocking, process-scoped file lock."""

    lock_path = _absolute_lexical(path)
    _ensure_safe_directory(lock_path.parent)
    try:
        existing = lock_path.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise OrchestrationError(
            f"could not inspect migration lock: {exc.__class__.__name__}"
        ) from exc
    if existing is not None and (
        stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)
    ):
        raise OrchestrationError("migration lock must be a regular non-symlink file")

    descriptor = _open_lock_file(lock_path)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise MigrationLockedError("migration is already running") from exc
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise MigrationLockedError("migration is already running") from exc
                raise OrchestrationError(
                    f"could not lock migration state: {exc.__class__.__name__}"
                ) from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_ready_state(output_root: Path) -> tuple[set[str], int]:
    root = _absolute_lexical(output_root)
    if root.exists() or root.is_symlink():
        _assert_directory(root, "output root")
    manifests = load_valid_ready_manifests(root / "ready")
    fingerprints: set[str] = set()
    maximum = 0
    for _, manifest in manifests:
        fingerprints.update(str(value) for value in manifest["episode_fingerprints"])
        match = _BATCH_ID_RE.fullmatch(str(manifest["batch_id"]))
        if match is not None:
            maximum = max(maximum, int(match.group(1)))
    return fingerprints, maximum + 1


def _remaining_episodes(
    episodes: Sequence[SourceEpisode], migrated: set[str]
) -> tuple[SourceEpisode, ...]:
    remaining: list[SourceEpisode] = []
    seen: set[str] = set()
    for episode in sorted(episodes, key=lambda item: item.sort_key()):
        if episode.fingerprint in migrated or episode.fingerprint in seen:
            continue
        seen.add(episode.fingerprint)
        remaining.append(episode)
    return tuple(remaining)


def _status_result(
    scan: ScanResult,
    remaining: Sequence[SourceEpisode],
    required: int,
    *,
    dry_run: bool,
) -> RunResult:
    return RunResult(
        status=RunStatus.BUSY if scan.busy_roots else RunStatus.WAITING,
        available=len(remaining),
        required=required,
        dry_run=dry_run,
    )


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise OrchestrationError("clock must return a timezone-aware datetime")
    utc = value.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _selection_episode(source: SourceEpisode) -> dict[str, object]:
    return {
        "host": source.host,
        "dataset_root": source.dataset_root,
        "dataset_name": source.dataset_name,
        "source_index": source.source_index,
        "fingerprint": source.fingerprint,
        "completed_ns": source.completed_ns,
        "length": source.length,
        "files": [
            {
                "role": file.role,
                "absolute_path": file.absolute_path,
                "relative_path": file.relative_path,
                "size": file.size,
                "mtime_ns": file.mtime_ns,
                "sha256": file.sha256,
            }
            for file in source.files
        ],
    }


def _selection_document(
    batch_id: str,
    selected: Sequence[SourceEpisode],
    created_at: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": created_at,
        "episode_fingerprints": [item.fingerprint for item in selected],
        "episodes": [_selection_episode(item) for item in selected],
    }


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise OrchestrationError("migration state is not strict JSON") from exc
    return (text + "\n").encode("utf-8")


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonstandard JSON constant {value}")


def _strict_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _read_selection(path: Path) -> dict[str, object]:
    return _read_strict_json_object(
        path,
        label="selection.json",
        maximum_bytes=_SELECTION_MAX_BYTES,
    )


def _read_strict_json_object(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> dict[str, object]:
    try:
        document = _read_regular_bytes(
            path,
            label=label,
            maximum_bytes=maximum_bytes,
        ).decode("utf-8")
        decoded = json.loads(
            document,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except OrchestrationError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise OrchestrationError(f"{label} is invalid") from exc
    if not isinstance(decoded, dict):
        raise OrchestrationError(f"{label} must contain an object")
    return decoded


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OrchestrationError(
                f"{label} must be a regular non-symlink file"
            )
        if before.st_size > maximum_bytes:
            raise OrchestrationError(f"{label} exceeds the size limit")
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise OrchestrationError(f"{label} changed identity while opening")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes:
            raise OrchestrationError(f"{label} exceeds the size limit")
        after = path.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
        ):
            raise OrchestrationError(f"{label} changed while reading")
        return payload
    except OrchestrationError:
        raise
    except OSError as exc:
        raise OrchestrationError(
            f"could not read {label}: {exc.__class__.__name__}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_bound_regular_file(path: Path, *, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OrchestrationError(
                f"{label} must be a regular non-symlink file"
            )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
        ):
            raise OrchestrationError(f"{label} changed identity before removal")
        if os.name == "nt":
            os.close(descriptor)
            descriptor = None
            current = path.lstat()
            if current.st_dev != opened.st_dev or current.st_ino != opened.st_ino:
                raise OrchestrationError(f"{label} changed identity before removal")
        path.unlink()
        _fsync_directory(path.parent)
    except OrchestrationError:
        raise
    except OSError as exc:
        raise OrchestrationError(
            f"could not remove {label}: {exc.__class__.__name__}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_pending_document(
    batch_id: str,
    manifest_payload: bytes,
    fingerprints: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "episode_fingerprints": list(fingerprints),
    }


def _read_publish_pending(path: Path, expected_batch_id: str) -> dict[str, object]:
    document = _read_strict_json_object(
        path,
        label="PUBLISH_PENDING",
        maximum_bytes=_PENDING_MAX_BYTES,
    )
    fingerprints = document.get("episode_fingerprints")
    if (
        document.get("schema_version") != 1
        or document.get("batch_id") != expected_batch_id
        or _SHA256_RE.fullmatch(str(document.get("manifest_sha256"))) is None
        or not isinstance(fingerprints, list)
        or len(fingerprints) != 20
        or any(_SHA256_RE.fullmatch(str(value)) is None for value in fingerprints)
        or len(set(fingerprints)) != len(fingerprints)
    ):
        raise OrchestrationError("PUBLISH_PENDING is invalid")
    return document


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise OrchestrationError(
            f"could not sync migration directory: {exc.__class__.__name__}"
        ) from exc


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename without ever replacing an existing target."""

    if os.name == "nt":
        try:
            os.rename(source, target)
        except FileExistsError as exc:
            raise OrchestrationError(f"publication target already exists: {target.name}") from exc
        except OSError as exc:
            if target.exists():
                raise OrchestrationError(
                    f"publication target already exists: {target.name}"
                ) from exc
            raise OrchestrationError(
                f"could not atomically rename migration path: {exc.__class__.__name__}"
            ) from exc
        return

    _rename_noreplace_at(
        -100,
        os.fspath(source),
        -100,
        os.fspath(target),
        target_label=target.name,
    )


def _rename_noreplace_at(
    source_dir_fd: int,
    source_name: str,
    target_dir_fd: int,
    target_name: str,
    *,
    target_label: str | None = None,
) -> None:
    if os.name != "posix":
        raise OrchestrationError(
            "no-clobber atomic rename is unsupported on this platform"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OrchestrationError("no-clobber atomic rename requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_dir_fd,
        os.fsencode(source_name),
        target_dir_fd,
        os.fsencode(target_name),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        label = target_name if target_label is None else target_label
        raise OrchestrationError(f"publication target already exists: {label}")
    raise OrchestrationError(
        f"could not atomically rename migration path: {os.strerror(error)}"
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(errno.EIO, "short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise OrchestrationError(f"migration state already exists: {path.name}") from exc
    except OrchestrationError:
        raise
    except OSError as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OrchestrationError(
            f"could not write migration state: {exc.__class__.__name__}"
        ) from exc


def _write_atomic_noreplace(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    _write_exclusive(temporary, payload)
    try:
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _quarantine_entry(staging_root: Path, entry: Path, now: datetime) -> None:
    if entry.parent != staging_root or not entry.name.startswith("batch_"):
        raise OrchestrationError(
            "only direct batch staging entries may be quarantined"
        )
    failed = _ensure_safe_directory(staging_root / "failed")
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = failed / f"{entry.name}.{timestamp}.{uuid.uuid4().hex}"
    _rename_noreplace(entry, destination)
    _fsync_directory(staging_root)
    _fsync_directory(failed)


def _quarantine_dataset(
    staging_root: Path,
    dataset_root: Path,
    batch_id: str,
    now: datetime,
) -> None:
    details = _lstat(dataset_root, "staging dataset")
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise OrchestrationError("staging dataset must be a non-symlink directory")
    _quarantine_derived_artifact(
        staging_root,
        dataset_root,
        batch_id,
        "dataset",
        now,
    )


def _quarantine_derived_artifact(
    staging_root: Path,
    artifact: Path,
    batch_id: str,
    kind: str,
    now: datetime,
) -> None:
    failed = _ensure_safe_directory(staging_root / "failed")
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = failed / (
        f"{batch_id}.{kind}.{timestamp}.{uuid.uuid4().hex}"
    )
    _rename_noreplace(artifact, destination)
    _fsync_directory(artifact.parent)
    _fsync_directory(failed)


def _selection_matches(
    document: Mapping[str, object],
    batch_id: str,
    selected: Sequence[SourceEpisode],
) -> bool:
    created_at = document.get("created_at")
    if not _valid_created_at(created_at):
        return False
    return document == _selection_document(
        batch_id, selected, cast(str, created_at)
    )


def _valid_created_at(value: object) -> bool:
    if (
        not isinstance(value, str)
        or value.strip() != value
        or "T" not in value
    ):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _download_path(downloads: Path, target_index: int, role: str) -> Path:
    try:
        filename = _ROLE_FILENAMES[role]
    except KeyError as exc:
        raise OrchestrationError(f"unsupported source file role: {role}") from exc
    return downloads / f"episode_{target_index:06d}" / filename


class _RetainDownloads(RuntimeError):
    """The cleanup tree could not be proven safe, so it must be retained."""


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _is_reparse_point(details: os.stat_result) -> bool:
    attributes = int(getattr(details, "st_file_attributes", 0))
    return stat.S_ISLNK(details.st_mode) or bool(attributes & 0x400)


def _open_bound_directory(path: Path, label: str) -> tuple[int, os.stat_result]:
    before = _lstat(path, label)
    if _is_reparse_point(before) or not stat.S_ISDIR(before.st_mode):
        raise _RetainDownloads(f"{label} is not a bound non-link directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _is_reparse_point(current)
            or not _same_identity(before, opened)
            or not _same_identity(opened, current)
        ):
            raise _RetainDownloads(f"{label} changed identity while opening")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(
    parent_fd: int,
    name: str,
    label: str,
) -> tuple[int, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _is_reparse_point(before) or not stat.S_ISDIR(before.st_mode):
        raise _RetainDownloads(f"{label} is not a non-link directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _is_reparse_point(current)
            or not _same_identity(before, opened)
            or not _same_identity(opened, current)
        ):
            raise _RetainDownloads(f"{label} changed identity while opening")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_at(
    parent_fd: int,
    name: str,
    label: str,
) -> tuple[int, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _is_reparse_point(before) or not stat.S_ISREG(before.st_mode):
        raise _RetainDownloads(f"{label} is not a non-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse_point(current)
            or not _same_identity(before, opened)
            or not _same_identity(opened, current)
        ):
            raise _RetainDownloads(f"{label} changed identity while opening")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _anchored_cleanup_supported() -> bool:
    required_dir_fd = {os.open, os.stat, os.unlink, os.rmdir}
    return (
        os.name == "posix"
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.listdir in os.supports_fd
        and os.stat in os.supports_follow_symlinks
    )


def _anchored_reuse_scan_supported() -> bool:
    required_dir_fd = {os.open, os.stat}
    return (
        os.name == "posix"
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
    )


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, _REUSE_HASH_CHUNK_BYTES)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _allocated_file_bytes(details: os.stat_result) -> int:
    size = max(0, int(details.st_size))
    blocks = getattr(details, "st_blocks", None)
    if isinstance(blocks, int) and not isinstance(blocks, bool) and blocks >= 0:
        return min(size, blocks * 512)
    return size


def _episode_reuse_sources(
    episode: SourceEpisode,
) -> tuple[tuple[str, SourceFile], ...]:
    result: list[tuple[str, SourceFile]] = []
    seen: set[str] = set()
    for source in episode.files:
        filename = _ROLE_FILENAMES.get(source.role)
        if filename is None or filename in seen:
            continue
        seen.add(filename)
        result.append((filename, source))
    return tuple(result)


def _anchored_directory_still_bound(
    parent_fd: int,
    name: str,
    opened: os.stat_result,
    device: int,
) -> bool:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(current.st_mode)
        and not _is_reparse_point(current)
        and current.st_dev == device
        and _same_identity(current, opened)
    )


def _verified_complete_bytes_at(
    parent_fd: int,
    filename: str,
    source: SourceFile,
    device: int,
) -> int:
    descriptor: int | None = None
    try:
        descriptor, opened = _open_regular_at(
            parent_fd,
            filename,
            f"reusable complete {filename}",
        )
        if opened.st_dev != device or opened.st_size != source.size:
            return 0
        digest = _descriptor_sha256(descriptor)
        after = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _is_reparse_point(current)
            or after.st_dev != device
            or current.st_dev != device
            or not _same_identity(opened, after)
            or not _same_identity(after, current)
            or after.st_size != opened.st_size
            or current.st_size != opened.st_size
            or digest != source.sha256
        ):
            return 0
        return source.size
    except (_RetainDownloads, OSError):
        return 0
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verified_partial_bytes_at(
    parent_fd: int,
    filename: str,
    source_size: int,
    device: int,
) -> int:
    descriptor: int | None = None
    try:
        descriptor, opened = _open_regular_at(
            parent_fd,
            filename,
            f"reusable partial {filename}",
        )
        if opened.st_dev != device:
            return 0
        allocated = _allocated_file_bytes(opened)
        after = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _is_reparse_point(current)
            or after.st_dev != device
            or current.st_dev != device
            or not _same_identity(opened, after)
            or not _same_identity(after, current)
            or after.st_size != opened.st_size
            or current.st_size != opened.st_size
            or _allocated_file_bytes(after) != allocated
            or _allocated_file_bytes(current) != allocated
        ):
            return 0
        return min(source_size, allocated)
    except (_RetainDownloads, OSError):
        return 0
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reusable_download_bytes_anchored(
    selected: Sequence[SourceEpisode],
    downloads: Path,
    batch_bytes: int,
) -> int:
    try:
        with ExitStack() as opened:
            downloads_fd, download_details = _open_bound_directory(
                downloads,
                "reusable downloads",
            )
            opened.callback(os.close, downloads_fd)
            total = 0
            for target_index, episode in enumerate(selected):
                episode_name = f"episode_{target_index:06d}"
                try:
                    episode_fd, episode_details = _open_directory_at(
                        downloads_fd,
                        episode_name,
                        f"reusable downloads/{episode_name}",
                    )
                except (_RetainDownloads, OSError):
                    continue
                opened.callback(os.close, episode_fd)
                if episode_details.st_dev != download_details.st_dev:
                    continue

                partial_fd: int | None = None
                partial_details: os.stat_result | None = None
                try:
                    partial_fd, partial_details = _open_directory_at(
                        episode_fd,
                        ".rsync-partial",
                        f"reusable downloads/{episode_name}/.rsync-partial",
                    )
                except (_RetainDownloads, OSError):
                    partial_fd = None
                    partial_details = None
                else:
                    opened.callback(os.close, partial_fd)
                    if partial_details.st_dev != download_details.st_dev:
                        partial_fd = None
                        partial_details = None

                complete_total = 0
                partial_total = 0
                for filename, source in _episode_reuse_sources(episode):
                    complete = _verified_complete_bytes_at(
                        episode_fd,
                        filename,
                        source,
                        download_details.st_dev,
                    )
                    partial = 0
                    if partial_fd is not None:
                        partial = _verified_partial_bytes_at(
                            partial_fd,
                            filename,
                            source.size,
                            download_details.st_dev,
                        )
                    complete_total += complete
                    partial_total += min(source.size - complete, partial)

                if partial_fd is not None and partial_details is not None:
                    if not _anchored_directory_still_bound(
                        episode_fd,
                        ".rsync-partial",
                        partial_details,
                        download_details.st_dev,
                    ):
                        partial_total = 0
                if not _anchored_directory_still_bound(
                    downloads_fd,
                    episode_name,
                    episode_details,
                    download_details.st_dev,
                ):
                    continue
                total += complete_total + partial_total

            try:
                current_downloads = downloads.lstat()
            except OSError:
                return 0
            if (
                _is_reparse_point(current_downloads)
                or not stat.S_ISDIR(current_downloads.st_mode)
                or not _same_identity(download_details, current_downloads)
            ):
                return 0
            return min(batch_bytes, total)
    except (OrchestrationError, _RetainDownloads, OSError):
        return 0


def _stable_path_directory(
    path: Path,
    *,
    device: int | None = None,
) -> os.stat_result | None:
    try:
        before = path.lstat()
        current = path.lstat()
    except OSError:
        return None
    if (
        _is_reparse_point(before)
        or _is_reparse_point(current)
        or not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or not _same_identity(before, current)
        or (device is not None and before.st_dev != device)
    ):
        return None
    return before


def _path_directory_still_bound(
    path: Path,
    opened: os.stat_result,
    device: int,
) -> bool:
    current = _stable_path_directory(path, device=device)
    return current is not None and _same_identity(opened, current)


def _open_stable_regular_path(
    path: Path,
    device: int,
) -> tuple[int, os.stat_result] | None:
    descriptor: int | None = None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if (
            _is_reparse_point(before)
            or not stat.S_ISREG(before.st_mode)
            or before.st_dev != device
        ):
            return None
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse_point(current)
            or opened.st_dev != device
            or current.st_dev != device
            or not _same_identity(before, opened)
            or not _same_identity(opened, current)
        ):
            return None
        result = (descriptor, opened)
        descriptor = None
        return result
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verified_complete_bytes_path(
    path: Path,
    source: SourceFile,
    device: int,
) -> int:
    opened = _open_stable_regular_path(path, device)
    if opened is None:
        return 0
    descriptor, details = opened
    try:
        if details.st_size != source.size:
            return 0
        digest = _descriptor_sha256(descriptor)
        after = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _is_reparse_point(current)
            or after.st_dev != device
            or current.st_dev != device
            or not _same_identity(details, after)
            or not _same_identity(after, current)
            or after.st_size != details.st_size
            or current.st_size != details.st_size
            or digest != source.sha256
        ):
            return 0
        return source.size
    except OSError:
        return 0
    finally:
        os.close(descriptor)


def _verified_partial_bytes_path(
    path: Path,
    source_size: int,
    device: int,
) -> int:
    opened = _open_stable_regular_path(path, device)
    if opened is None:
        return 0
    descriptor, details = opened
    try:
        allocated = _allocated_file_bytes(details)
        after = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _is_reparse_point(current)
            or after.st_dev != device
            or current.st_dev != device
            or not _same_identity(details, after)
            or not _same_identity(after, current)
            or after.st_size != details.st_size
            or current.st_size != details.st_size
            or _allocated_file_bytes(after) != allocated
            or _allocated_file_bytes(current) != allocated
        ):
            return 0
        return min(source_size, allocated)
    except OSError:
        return 0
    finally:
        os.close(descriptor)


def _reusable_download_bytes_path(
    selected: Sequence[SourceEpisode],
    downloads: Path,
    batch_bytes: int,
) -> int:
    download_details = _stable_path_directory(downloads)
    if download_details is None:
        return 0
    total = 0
    for target_index, episode in enumerate(selected):
        episode_path = downloads / f"episode_{target_index:06d}"
        episode_details = _stable_path_directory(
            episode_path,
            device=download_details.st_dev,
        )
        if episode_details is None:
            continue
        partial_path = episode_path / ".rsync-partial"
        partial_details = _stable_path_directory(
            partial_path,
            device=download_details.st_dev,
        )
        complete_total = 0
        partial_total = 0
        for filename, source in _episode_reuse_sources(episode):
            complete = _verified_complete_bytes_path(
                episode_path / filename,
                source,
                download_details.st_dev,
            )
            partial = 0
            if partial_details is not None:
                partial = _verified_partial_bytes_path(
                    partial_path / filename,
                    source.size,
                    download_details.st_dev,
                )
            complete_total += complete
            partial_total += min(source.size - complete, partial)
        if partial_details is not None and not _path_directory_still_bound(
            partial_path,
            partial_details,
            download_details.st_dev,
        ):
            partial_total = 0
        if not _path_directory_still_bound(
            episode_path,
            episode_details,
            download_details.st_dev,
        ):
            continue
        total += complete_total + partial_total
    if not _path_directory_still_bound(
        downloads,
        download_details,
        download_details.st_dev,
    ):
        return 0
    return min(batch_bytes, total)


def _reusable_download_bytes(
    selected: Sequence[SourceEpisode],
    downloads: Path,
) -> int:
    batch_bytes = sum(
        source.size for episode in selected for source in episode.files
    )
    try:
        if _anchored_reuse_scan_supported():
            return _reusable_download_bytes_anchored(
                selected,
                downloads,
                batch_bytes,
            )
        return _reusable_download_bytes_path(selected, downloads, batch_bytes)
    except (OrchestrationError, _RetainDownloads, OSError):
        return 0


class MigrationService:
    """Coordinate one migration attempt without owning a long-running daemon."""

    def __init__(
        self,
        config: MigrationConfig,
        *,
        scanner: Scanner | None = None,
        transfer: Transfer | None = None,
        builder: Builder = build_batch,
        validator: Validator = validate_batch,
        ledger_factory: LedgerFactory = MigrationLedger,
        space_checker: SpaceChecker = ensure_space,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.scanner = SshSourceScanner(config) if scanner is None else scanner
        self.transfer = RsyncTransfer(config) if transfer is None else transfer
        self.builder = builder
        self.validator = validator
        self.ledger_factory = ledger_factory
        self.space_checker = space_checker
        self.clock = (lambda: datetime.now(timezone.utc)) if clock is None else clock

    def run(
        self,
        *,
        scanner: Scanner | None = None,
        dry_run: bool = False,
    ) -> RunResult:
        if self.config.batch_size != 20:
            raise OrchestrationError("migration batch size must be exactly 20")
        source_scanner = self.scanner if scanner is None else scanner
        if not dry_run:
            self._recover_pending_publications_before_scan()
        scan = source_scanner.scan()
        migrated, next_sequence = _read_ready_state(self.config.output_root)
        remaining = _remaining_episodes(scan.episodes, migrated)
        selected = select_next_batch(
            remaining,
            migrated=set(),
            batch_size=self.config.batch_size,
        )
        if not selected:
            return _status_result(
                scan, remaining, self.config.batch_size, dry_run=dry_run
            )
        batch_id = make_batch_id(next_sequence, selected)
        if dry_run:
            return RunResult(
                status=RunStatus.CREATED,
                available=len(remaining),
                required=self.config.batch_size,
                selected=selected,
                batch_id=batch_id,
                dry_run=True,
            )

        output_root = _absolute_lexical(self.config.output_root)
        with exclusive_lock(output_root / "state" / "migrate.lock"):
            return self._locked_run(source_scanner)

    def _locked_run(self, scanner: Scanner) -> RunResult:
        self._recover_pending_publications_locked()
        scan = scanner.scan()
        ready_root = _absolute_lexical(self.config.output_root) / "ready"
        ready_fingerprints, _ = _read_ready_state(self.config.output_root)
        remaining = _remaining_episodes(scan.episodes, ready_fingerprints)
        selected = select_next_batch(
            remaining,
            migrated=set(),
            batch_size=self.config.batch_size,
        )
        if not selected:
            return _status_result(scan, remaining, self.config.batch_size, dry_run=False)

        output_root = _ensure_safe_directory(self.config.output_root)
        state_root = _ensure_safe_directory(output_root / "state")
        ledger_path = state_root / "migration.sqlite3"
        with self.ledger_factory(ledger_path) as ledger:
            ledger.reconcile_ready(ready_root)
            remaining = _remaining_episodes(scan.episodes, ledger.migrated_fingerprints())
            selected = select_next_batch(
                remaining,
                migrated=set(),
                batch_size=self.config.batch_size,
            )
            if not selected:
                return _status_result(
                    scan, remaining, self.config.batch_size, dry_run=False
                )
            batch_id = make_batch_id(ledger.next_batch_sequence(), selected)
            created_at = _utc_text(self.clock())
            batch_root, created_at = self._prepare_staging(
                batch_id, selected, created_at
            )
            downloads = batch_root / "downloads"
            dataset_root = batch_root / "dataset"
            self._recover_staging_dataset(
                batch_root,
                dataset_root,
                batch_id,
            )
            batch_bytes = sum(file.size for item in selected for file in item.files)
            reusable_bytes = _reusable_download_bytes(selected, downloads)
            self.space_checker(
                downloads,
                batch_bytes=batch_bytes,
                reusable_bytes=reusable_bytes,
            )
            downloaded = self._transfer_selected(selected, downloads)
            scanner.revalidate(selected)
            built = self.builder(
                downloaded,
                dataset_root,
                batch_id=batch_id,
                created_at=created_at,
            )
            report = self.validator(
                built.root,
                expected_episodes=self.config.batch_size,
                ffprobe=self.config.ffprobe,
            )
            manifest = copy.deepcopy(built.manifest)
            manifest["validation"] = report.as_dict()
            manifest_payload = _canonical_json(manifest)
            publication_root = self._stage_publication_root(
                batch_root,
                dataset_root,
                batch_id,
            )
            _write_atomic_noreplace(
                publication_root / "migration_manifest.json",
                manifest_payload,
            )
            _write_exclusive(publication_root / "READY", b"")
            self.validator(
                publication_root,
                expected_episodes=self.config.batch_size,
                ffprobe=self.config.ffprobe,
            )
            pending_path = batch_root / "PUBLISH_PENDING"
            _write_atomic_noreplace(
                pending_path,
                _canonical_json(
                    _publish_pending_document(
                        batch_id,
                        manifest_payload,
                        [item.fingerprint for item in selected],
                    )
                ),
            )
            return self._finish_publication(
                ledger=ledger,
                manifest=manifest,
                batch_root=batch_root,
                publication_root=publication_root,
                downloads=downloads,
                batch_id=batch_id,
                selected=selected,
                available=len(remaining),
                pending_path=pending_path,
            )

    def _recover_pending_publications_before_scan(self) -> None:
        if not self._discover_pending_publications():
            return
        output_root = _absolute_lexical(self.config.output_root)
        with exclusive_lock(output_root / "state" / "migrate.lock"):
            self._recover_pending_publications_locked()

    def _discover_pending_publications(self) -> tuple[tuple[Path, str], ...]:
        output_root = _absolute_lexical(self.config.output_root)
        if not (output_root.exists() or output_root.is_symlink()):
            return ()
        _assert_directory(output_root, "output root")
        staging_root = output_root / ".staging"
        if not (staging_root.exists() or staging_root.is_symlink()):
            return ()
        _assert_directory(staging_root, "staging root")

        pending: list[tuple[Path, str]] = []
        for entry in sorted(staging_root.iterdir(), key=lambda path: path.name):
            if not entry.name.startswith("batch_"):
                continue
            details = _lstat(entry, "staging batch entry")
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                continue
            marker = entry / "PUBLISH_PENDING"
            if not (marker.exists() or marker.is_symlink()):
                continue
            if _BATCH_ID_RE.fullmatch(entry.name) is None:
                raise OrchestrationError("PUBLISH_PENDING has an invalid batch id")
            _read_publish_pending(marker, entry.name)
            pending.append((entry, entry.name))
        return tuple(pending)

    def _recover_pending_publications_locked(self) -> None:
        pending = self._discover_pending_publications()
        if not pending:
            return

        output_root = _ensure_safe_directory(self.config.output_root)
        ready_root = _ensure_safe_directory(output_root / "ready")
        state_root = _ensure_safe_directory(output_root / "state")
        with self.ledger_factory(state_root / "migration.sqlite3") as ledger:
            for batch_root, batch_id in pending:
                marker_path = batch_root / "PUBLISH_PENDING"
                marker = _read_publish_pending(marker_path, batch_id)
                staged = batch_root / "publish" / batch_id
                published = ready_root / batch_id
                staged_exists = staged.exists() or staged.is_symlink()
                published_exists = published.exists() or published.is_symlink()
                if staged_exists == published_exists:
                    raise OrchestrationError(
                        "PUBLISH_PENDING must identify exactly one publication location"
                    )
                if staged_exists:
                    self._verify_pending_publication(staged, marker, batch_id)
                    self._publish_dataset(staged, published)
                self._verify_pending_publication(published, marker, batch_id)
                ledger.reconcile_ready(ready_root)
                downloads = batch_root / "downloads"
                self._cleanup_downloads(
                    batch_root,
                    downloads,
                    allow_partial=True,
                )
                _remove_bound_regular_file(marker_path, label="PUBLISH_PENDING")

    def _verify_pending_publication(
        self,
        publication_root: Path,
        marker: Mapping[str, object],
        batch_id: str,
    ) -> dict[str, object]:
        _assert_directory(publication_root, "pending publication")
        if publication_root.name != batch_id:
            raise OrchestrationError("pending publication basename does not match batch id")
        manifest_path = publication_root / "migration_manifest.json"
        payload = _read_regular_bytes(
            manifest_path,
            label="migration_manifest.json",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != marker["manifest_sha256"]:
            raise OrchestrationError("PUBLISH_PENDING manifest hash does not match")
        manifest = _read_strict_json_object(
            manifest_path,
            label="migration_manifest.json",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
        if (
            manifest.get("batch_id") != batch_id
            or manifest.get("episode_fingerprints")
            != marker["episode_fingerprints"]
        ):
            raise OrchestrationError("PUBLISH_PENDING does not match publication manifest")
        ready_payload = _read_regular_bytes(
            publication_root / "READY",
            label="READY",
            maximum_bytes=0,
        )
        if ready_payload != b"":
            raise OrchestrationError("READY must be empty")
        self.validator(
            publication_root,
            expected_episodes=self.config.batch_size,
            ffprobe=self.config.ffprobe,
        )
        return manifest

    def _recover_staging_dataset(
        self,
        batch_root: Path,
        dataset_root: Path,
        batch_id: str,
    ) -> None:
        pending_path = batch_root / "PUBLISH_PENDING"
        if pending_path.exists() or pending_path.is_symlink():
            raise OrchestrationError(
                "pending publication was not recovered before staging reuse"
            )
        publication_root = batch_root / "publish" / batch_id
        if publication_root.exists() or publication_root.is_symlink():
            details = _lstat(publication_root, "staging publication")
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise OrchestrationError(
                    "staging publication must be a non-symlink directory"
                )
            _quarantine_derived_artifact(
                batch_root.parent,
                publication_root,
                batch_id,
                "publication",
                self.clock(),
            )
        if not (dataset_root.exists() or dataset_root.is_symlink()):
            return
        _assert_directory(dataset_root, "staging dataset")
        # A crash can leave a fully marked dataset, but no immutable publication
        # boundary exists until the directory rename.  Never trust those bytes on
        # restart: isolate only the derived dataset and rebuild it from the
        # still-bound, checksum-verified downloads retained by selection.json.
        _quarantine_dataset(
            batch_root.parent,
            dataset_root,
            batch_id,
            self.clock(),
        )

    def _finish_publication(
        self,
        *,
        ledger: Ledger,
        manifest: dict[str, object],
        batch_root: Path,
        publication_root: Path,
        downloads: Path,
        batch_id: str,
        selected: Sequence[SourceEpisode],
        available: int,
        pending_path: Path,
    ) -> RunResult:
        ready_root = _ensure_safe_directory(
            _absolute_lexical(self.config.output_root) / "ready"
        )
        published = ready_root / batch_id
        self._publish_dataset(publication_root, published)
        self.validator(
            published,
            expected_episodes=self.config.batch_size,
            ffprobe=self.config.ffprobe,
        )
        manifest_path = published / "migration_manifest.json"
        ledger.record_published_batch(manifest, manifest_path=manifest_path)
        self._cleanup_downloads(batch_root, downloads)
        _remove_bound_regular_file(pending_path, label="PUBLISH_PENDING")
        return RunResult(
            status=RunStatus.CREATED,
            available=available,
            required=self.config.batch_size,
            selected=tuple(selected),
            batch_id=batch_id,
        )

    def _stage_publication_root(
        self,
        batch_root: Path,
        dataset_root: Path,
        batch_id: str,
    ) -> Path:
        publish_root = _ensure_safe_directory(batch_root / "publish")
        publication_root = publish_root / batch_id
        if publication_root.exists() or publication_root.is_symlink():
            raise OrchestrationError(
                f"staging publication target already exists: {batch_id}"
            )
        _rename_noreplace(dataset_root, publication_root)
        _fsync_directory(batch_root)
        _fsync_directory(publish_root)
        return publication_root

    def _prepare_staging(
        self,
        batch_id: str,
        selected: Sequence[SourceEpisode],
        created_at: str,
    ) -> tuple[Path, str]:
        output_root = _ensure_safe_directory(self.config.output_root)
        staging_root = _ensure_safe_directory(output_root / ".staging")
        now = self.clock()
        for candidate in tuple(staging_root.iterdir()):
            if candidate.name in {"failed", ".cleanup"}:
                continue
            if candidate.name.startswith("batch_") and candidate.name != batch_id:
                _quarantine_entry(staging_root, candidate, now)

        batch_root = staging_root / batch_id
        if batch_root.exists() or batch_root.is_symlink():
            details = _lstat(batch_root, "staging batch entry")
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                _quarantine_entry(staging_root, batch_root, now)
            else:
                selection_path = batch_root / "selection.json"
                try:
                    document = _read_selection(selection_path)
                except OrchestrationError:
                    _quarantine_entry(staging_root, batch_root, now)
                else:
                    if _selection_matches(document, batch_id, selected):
                        value = cast(str, document["created_at"])
                        _ensure_safe_directory(batch_root / "downloads")
                        return batch_root, value
                    _quarantine_entry(staging_root, batch_root, now)

        try:
            batch_root.mkdir(mode=0o700)
        except OSError as exc:
            raise OrchestrationError(
                f"could not create batch staging directory: {exc.__class__.__name__}"
            ) from exc
        _fsync_directory(staging_root)
        selection_path = batch_root / "selection.json"
        _write_atomic_noreplace(
            selection_path,
            _canonical_json(_selection_document(batch_id, selected, created_at)),
        )
        _ensure_safe_directory(batch_root / "downloads")
        return batch_root, created_at

    def _transfer_selected(
        self,
        selected: Sequence[SourceEpisode],
        downloads: Path,
    ) -> tuple[DownloadedEpisode, ...]:
        downloaded: list[DownloadedEpisode] = []
        for target_index, source in enumerate(selected):
            roles = [file.role for file in source.files]
            if len(roles) != len(_ROLE_FILENAMES) or set(roles) != set(_ROLE_FILENAMES):
                raise OrchestrationError("selected episode source file roles are incomplete")
            paths: dict[str, Path] = {}
            for source_file in source.files:
                target = _download_path(downloads, target_index, source_file.role)
                self.transfer.fetch(source_file, target)
                paths[source_file.role] = target
            downloaded.append(DownloadedEpisode(source=source, files=paths))
        return tuple(downloaded)

    def _publish_dataset(self, dataset_root: Path, published: Path) -> None:
        if published.exists() or published.is_symlink():
            raise OrchestrationError(f"publication target already exists: {published.name}")
        _rename_noreplace(dataset_root, published)
        _fsync_directory(dataset_root.parent)
        _fsync_directory(published.parent)

    def _cleanup_downloads(
        self,
        batch_root: Path,
        downloads: Path,
        *,
        allow_partial: bool = False,
    ) -> bool:
        expected = batch_root / "downloads"
        if downloads != expected or downloads.parent != batch_root:
            raise OrchestrationError("download cleanup escaped its batch staging root")

        try:
            batch_details = batch_root.lstat()
        except FileNotFoundError as exc:
            raise OrchestrationError("staging batch disappeared before cleanup") from exc
        try:
            download_details = downloads.lstat()
        except FileNotFoundError:
            if allow_partial:
                return True
            raise OrchestrationError(
                "downloads disappeared before initial cleanup"
            ) from None
        if _is_reparse_point(batch_details) or not stat.S_ISDIR(batch_details.st_mode):
            raise OrchestrationError("staging batch must be a non-link directory")
        if _is_reparse_point(download_details) or not stat.S_ISDIR(
            download_details.st_mode
        ):
            raise OrchestrationError("downloads must be a non-link directory")
        if not _anchored_cleanup_supported():
            return False

        expected_episodes = {f"episode_{index:06d}" for index in range(20)}
        expected_files = set(_ROLE_FILENAMES.values())
        deletion_started = False
        partial_observed = False
        tree_bound = False
        try:
            with ExitStack() as opened:
                staging_fd, _ = _open_bound_directory(
                    batch_root.parent, "staging root"
                )
                opened.callback(os.close, staging_fd)
                batch_fd, opened_batch = _open_directory_at(
                    staging_fd, batch_root.name, "staging batch"
                )
                opened.callback(os.close, batch_fd)
                if not _same_identity(batch_details, opened_batch):
                    raise _RetainDownloads("staging batch changed identity")
                downloads_fd, opened_downloads = _open_directory_at(
                    batch_fd, "downloads", "downloads"
                )
                opened.callback(os.close, downloads_fd)
                if (
                    not _same_identity(download_details, opened_downloads)
                    or opened_downloads.st_dev != opened_batch.st_dev
                ):
                    raise _RetainDownloads("downloads changed identity or filesystem")

                episode_handles: dict[str, tuple[int, os.stat_result]] = {}
                episode_files: dict[str, set[str]] = {}
                file_handles: dict[
                    tuple[str, str], tuple[int, os.stat_result]
                ] = {}
                actual_episodes = set(os.listdir(downloads_fd))
                extra_episodes = actual_episodes - expected_episodes
                missing_episodes = expected_episodes - actual_episodes
                present_episodes = actual_episodes & expected_episodes
                partial_observed = bool(missing_episodes)

                abnormal_episode = False
                missing_files: set[tuple[str, str]] = set()
                extra_files: set[tuple[str, str]] = set()
                for episode_name in sorted(present_episodes):
                    try:
                        episode_entry = os.stat(
                            episode_name,
                            dir_fd=downloads_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        missing_episodes.add(episode_name)
                        partial_observed = True
                        continue
                    except OSError:
                        abnormal_episode = True
                        continue
                    if _is_reparse_point(episode_entry) or not stat.S_ISDIR(
                        episode_entry.st_mode
                    ):
                        abnormal_episode = True
                        continue
                    try:
                        episode_fd, episode_details = _open_directory_at(
                            downloads_fd,
                            episode_name,
                            f"downloads/{episode_name}",
                        )
                    except FileNotFoundError:
                        missing_episodes.add(episode_name)
                        partial_observed = True
                        continue
                    except (_RetainDownloads, OSError):
                        abnormal_episode = True
                        continue
                    opened.callback(os.close, episode_fd)
                    if episode_details.st_dev != opened_downloads.st_dev:
                        abnormal_episode = True
                        continue
                    episode_handles[episode_name] = (episode_fd, episode_details)
                    actual_files = set(os.listdir(episode_fd))
                    episode_files[episode_name] = actual_files
                    missing_files.update(
                        (episode_name, filename)
                        for filename in expected_files - actual_files
                    )
                    extra_files.update(
                        (episode_name, filename)
                        for filename in actual_files - expected_files
                    )

                partial_observed = bool(missing_episodes or missing_files)
                if extra_episodes:
                    if partial_observed or deletion_started:
                        raise OrchestrationError(
                            "partial downloads also contains an unexpected entry"
                        )
                    return False
                if extra_files:
                    if partial_observed or deletion_started:
                        raise OrchestrationError(
                            "partial downloads also contains an unexpected file"
                        )
                    return False
                if abnormal_episode:
                    if partial_observed or deletion_started:
                        raise OrchestrationError(
                            "partial downloads contains an abnormal episode entry"
                        )
                    return False
                if partial_observed and not allow_partial:
                    return False

                abnormal_file = False
                for episode_name, (episode_fd, _) in episode_handles.items():
                    for filename in sorted(episode_files[episode_name]):
                        try:
                            file_fd, file_details = _open_regular_at(
                                episode_fd,
                                filename,
                                f"downloads/{episode_name}/{filename}",
                            )
                        except FileNotFoundError:
                            episode_files[episode_name].discard(filename)
                            missing_files.add((episode_name, filename))
                            partial_observed = True
                            continue
                        except (_RetainDownloads, OSError):
                            abnormal_file = True
                            continue
                        opened.callback(os.close, file_fd)
                        if file_details.st_dev != opened_downloads.st_dev:
                            abnormal_file = True
                            continue
                        file_handles[(episode_name, filename)] = (
                            file_fd,
                            file_details,
                        )
                if abnormal_file:
                    if partial_observed or deletion_started:
                        raise OrchestrationError(
                            "partial downloads contains an abnormal file entry"
                        )
                    return False

                present_episodes = set(episode_handles)
                tree_bound = True

                def verify_held_tree() -> None:
                    if set(os.listdir(downloads_fd)) != present_episodes:
                        raise _RetainDownloads("downloads changed before cleanup")
                    for episode_name, (episode_fd, episode_details) in (
                        episode_handles.items()
                    ):
                        current_episode = os.stat(
                            episode_name,
                            dir_fd=downloads_fd,
                            follow_symlinks=False,
                        )
                        if (
                            _is_reparse_point(current_episode)
                            or not _same_identity(current_episode, episode_details)
                            or set(os.listdir(episode_fd))
                            != episode_files[episode_name]
                        ):
                            raise _RetainDownloads(
                                f"downloads/{episode_name} changed before cleanup"
                            )
                        for filename in episode_files[episode_name]:
                            _, file_details = file_handles[(episode_name, filename)]
                            current_file = os.stat(
                                filename,
                                dir_fd=episode_fd,
                                follow_symlinks=False,
                            )
                            if (
                                _is_reparse_point(current_file)
                                or not stat.S_ISREG(current_file.st_mode)
                                or not _same_identity(current_file, file_details)
                            ):
                                raise _RetainDownloads(
                                    f"downloads/{episode_name}/{filename} changed"
                                )

                verify_held_tree()
                current_downloads = os.stat(
                    "downloads", dir_fd=batch_fd, follow_symlinks=False
                )
                if not _same_identity(current_downloads, opened_downloads):
                    raise _RetainDownloads("downloads changed before cleanup")

                for episode_name in sorted(present_episodes):
                    episode_fd, _ = episode_handles[episode_name]
                    for filename in sorted(episode_files[episode_name]):
                        _, file_details = file_handles[(episode_name, filename)]
                        current_file = os.stat(
                            filename,
                            dir_fd=episode_fd,
                            follow_symlinks=False,
                        )
                        if not _same_identity(current_file, file_details):
                            raise _RetainDownloads("download file changed before unlink")
                        deletion_started = True
                        os.unlink(filename, dir_fd=episode_fd)
                    os.fsync(episode_fd)
                    if os.listdir(episode_fd):
                        raise _RetainDownloads("episode was mutated during cleanup")
                    current_episode = os.stat(
                        episode_name,
                        dir_fd=downloads_fd,
                        follow_symlinks=False,
                    )
                    if not _same_identity(
                        current_episode, episode_handles[episode_name][1]
                    ):
                        raise _RetainDownloads("episode changed before removal")
                    deletion_started = True
                    os.rmdir(episode_name, dir_fd=downloads_fd)
                os.fsync(downloads_fd)
                if os.listdir(downloads_fd):
                    raise _RetainDownloads("downloads was mutated during cleanup")
                current_downloads = os.stat(
                    "downloads",
                    dir_fd=batch_fd,
                    follow_symlinks=False,
                )
                if not _same_identity(current_downloads, opened_downloads):
                    raise _RetainDownloads("downloads changed before removal")
                deletion_started = True
                os.rmdir("downloads", dir_fd=batch_fd)
                os.fsync(batch_fd)
                return True
        except (_RetainDownloads, OrchestrationError, OSError) as exc:
            if not partial_observed and not deletion_started and not tree_bound:
                return False
            raise OrchestrationError(
                f"could not finish anchored staging cleanup: {exc.__class__.__name__}"
            ) from exc
