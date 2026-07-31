from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import contextlib
import ctypes
import dataclasses
import datetime
import errno
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

_ROUND_ID = re.compile(r"round_[0-9]{6}")
_LOWERCASE_SHA1 = re.compile(r"[0-9a-f]{40}")
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_ADMISSION_MAX_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_FEATURE_SEED_VERSION = 1
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_PARAMETER_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


@dataclasses.dataclass
class _DirectoryWitness:
    path: Path
    descriptor: int
    device: int
    inode: int
    sealed_metadata: tuple[int, int, int, int] | None = None
    _closed: bool = False

    @classmethod
    def from_descriptor(cls, path: Path, descriptor: int) -> _DirectoryWitness:
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISDIR(value.st_mode):
                raise RuntimeError(f"not a directory: {path}")
        except BaseException:
            primary_error = sys.exception()
            errors: list[tuple[str, BaseException]] = []
            try:
                os.close(descriptor)
            except BaseException as exc:
                errors.append(("unadopted directory descriptor", exc))
            _finish_cleanup(
                primary_error,
                errors,
                context=f"directory witness construction for {path}",
            )
            raise
        return cls(
            path=path,
            descriptor=descriptor,
            device=value.st_dev,
            inode=value.st_ino,
        )

    def verify(self) -> None:
        if self._closed:
            raise RuntimeError(f"directory witness is already closed: {self.path}")
        held = os.fstat(self.descriptor)
        if not stat.S_ISDIR(held.st_mode) or (held.st_dev, held.st_ino) != (
            self.device,
            self.inode,
        ):
            raise RuntimeError(f"held directory witness changed: {self.path}")
        if self.sealed_metadata is not None and _directory_metadata(held) != self.sealed_metadata:
            raise RuntimeError(f"held sealed directory metadata changed: {self.path}")
        rebound = _open_real_directory_chain(self.path, create=False)
        try:
            if (rebound.device, rebound.inode) != (self.device, self.inode):
                raise RuntimeError(f"directory pathname was renamed, deleted, or replaced: {self.path}")
            rebound_stat = os.fstat(rebound.descriptor)
            if self.sealed_metadata is not None and _directory_metadata(rebound_stat) != self.sealed_metadata:
                raise RuntimeError(f"sealed directory pathname metadata changed: {self.path}")
        finally:
            rebound.close()

    def seal(self) -> None:
        self.verify()
        self.sealed_metadata = _directory_metadata(os.fstat(self.descriptor))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.descriptor)

    def __enter__(self) -> _DirectoryWitness:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _finish_cleanup(
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
    _, first_error = errors[0]
    for label, error in errors[1:]:
        first_error.add_note(f"{context} cleanup also failed during {label}: {error}")
    raise first_error


@contextlib.contextmanager
def _recording_exit_stack(
    cleanup_errors: list[tuple[str, BaseException]],
    *,
    label: str,
    context: str,
) -> Iterator[contextlib.ExitStack]:
    stack = contextlib.ExitStack()
    try:
        yield stack
    except BaseException:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        try:
            stack.close()
        except BaseException as exc:
            errors.append((label, exc))
        _finish_cleanup(
            primary_error,
            errors,
            context=context,
        )
        raise
    else:
        try:
            stack.close()
        except BaseException as exc:
            cleanup_errors.append((label, exc))


@dataclasses.dataclass
class _CacheRootWitness:
    parent: _DirectoryWitness
    root: _DirectoryWitness
    event_descriptor: int
    payload: _PinnedSnapshotTree | None = None
    _closed: bool = False

    def verify(self) -> None:
        if self.payload is not None:
            self.payload.verify()
        try:
            events = os.read(self.event_descriptor, 64 * 1024)
        except BlockingIOError:
            events = b""
        if events:
            raise RuntimeError(f"cache parent changed while final root was pinned: {self.root.path}")
        self.parent.verify()
        self.root.verify()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        close_operations = []
        if self.payload is not None:
            close_operations.append(("cache payload guard", self.payload.close))
        close_operations.extend(
            [
                ("cache root descriptor", self.root.close),
                ("cache parent descriptor", self.parent.close),
                ("cache parent event descriptor", lambda: os.close(self.event_descriptor)),
            ]
        )
        for label, close in close_operations:
            try:
                close()
            except BaseException as exc:
                errors.append((label, exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="final cache witness",
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _directory_metadata(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _watch_directory_changes(
    path: Path,
    *,
    include_child_modifications: bool = False,
) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    init = getattr(libc, "inotify_init1", None)
    add_watch = getattr(libc, "inotify_add_watch", None)
    if init is None or add_watch is None:
        raise RuntimeError("Linux inotify is required for final cache root binding")
    init.argtypes = [ctypes.c_int]
    init.restype = ctypes.c_int
    descriptor = init(getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    try:
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        mask = (
            0x00000004  # IN_ATTRIB
            | 0x00000040  # IN_MOVED_FROM
            | 0x00000080  # IN_MOVED_TO
            | 0x00000100  # IN_CREATE
            | 0x00000200  # IN_DELETE
            | 0x00000400  # IN_DELETE_SELF
            | 0x00000800  # IN_MOVE_SELF
        )
        if include_child_modifications:
            mask |= (
                0x00000002  # IN_MODIFY
                | 0x00000008  # IN_CLOSE_WRITE
            )
        watch = add_watch(descriptor, os.fsencode(path), mask)
        if watch < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return descriptor
    except BaseException:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append(("inotify descriptor", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context=f"directory watch construction for {path}",
        )
        raise


def _pinned_metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, _COPY_CHUNK_BYTES, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


@dataclasses.dataclass
class _PinnedSnapshotFile:
    relative: str
    descriptor: int
    metadata: tuple[int, int, int, int, int, int, int]
    sha256: str | None


@dataclasses.dataclass
class _PinnedSnapshotDirectory:
    relative: str
    descriptor: int
    metadata: tuple[int, int, int, int, int, int, int]
    event_descriptor: int


@dataclasses.dataclass
class _PinnedSnapshotTree:
    directories: list[_PinnedSnapshotDirectory]
    files: list[_PinnedSnapshotFile]
    label: str = "private snapshot"
    _closed: bool = False

    @classmethod
    def open(
        cls,
        root_descriptor: int,
        *,
        label: str = "private snapshot",
        hash_files: bool = True,
    ) -> _PinnedSnapshotTree:
        tree = cls(directories=[], files=[], label=label)

        def walk(directory: int, relative: Path) -> None:
            event_descriptor: int | None = None
            adopted = False
            try:
                metadata = os.fstat(directory)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeError(f"{label} entry is not a directory: {relative}")
                event_descriptor = _watch_directory_changes(
                    Path("/proc/self/fd") / str(directory),
                    include_child_modifications=True,
                )
                tree.directories.append(
                    _PinnedSnapshotDirectory(
                        relative=relative.as_posix(),
                        descriptor=directory,
                        metadata=_pinned_metadata(metadata),
                        event_descriptor=event_descriptor,
                    )
                )
                adopted = True
            except BaseException:
                primary_error = sys.exception()
                errors: list[tuple[str, BaseException]] = []
                if event_descriptor is not None:
                    try:
                        os.close(event_descriptor)
                    except BaseException as exc:
                        errors.append((f"directory watch {relative}", exc))
                if not adopted:
                    try:
                        os.close(directory)
                    except BaseException as exc:
                        errors.append((f"directory descriptor {relative}", exc))
                _finish_cleanup(
                    primary_error,
                    errors,
                    context=f"{label} pinning",
                )
                raise
            for name in sorted(os.listdir(directory)):
                child_relative = relative / name
                child_metadata = os.stat(
                    name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(child_metadata.st_mode):
                    child = os.open(name, _directory_flags(), dir_fd=directory)
                    walk(child, child_relative)
                    continue
                if not stat.S_ISREG(child_metadata.st_mode):
                    raise RuntimeError(f"{label} must contain only regular files and directories: {child_relative}")
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=directory,
                )
                file_adopted = False
                try:
                    opened_metadata = os.fstat(child)
                    if _pinned_metadata(opened_metadata) != _pinned_metadata(child_metadata):
                        raise RuntimeError(f"{label} file changed while pinning: {child_relative}")
                    child_sha256 = _sha256_descriptor(child) if hash_files else None
                    tree.files.append(
                        _PinnedSnapshotFile(
                            relative=child_relative.as_posix(),
                            descriptor=child,
                            metadata=_pinned_metadata(opened_metadata),
                            sha256=child_sha256,
                        )
                    )
                    file_adopted = True
                except BaseException:
                    primary_error = sys.exception()
                    errors: list[tuple[str, BaseException]] = []
                    if not file_adopted:
                        try:
                            os.close(child)
                        except BaseException as exc:
                            errors.append((f"file descriptor {child_relative}", exc))
                    _finish_cleanup(
                        primary_error,
                        errors,
                        context=f"{label} pinning",
                    )
                    raise

        root = os.dup(root_descriptor)
        try:
            walk(root, Path("."))
            tree.verify()
            return tree
        except BaseException:
            primary_error = sys.exception()
            errors: list[tuple[str, BaseException]] = []
            try:
                tree.close()
            except BaseException as exc:
                errors.append(("adopted snapshot tree", exc))
            _finish_cleanup(
                primary_error,
                errors,
                context=f"{label} construction",
            )
            raise

    def verify(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.label} mutation guard is already closed")
        for directory in self.directories:
            try:
                events = os.read(directory.event_descriptor, 64 * 1024)
            except BlockingIOError:
                events = b""
            if events:
                raise RuntimeError(f"{self.label} changed while pinned: directory event at {directory.relative}")
            if _pinned_metadata(os.fstat(directory.descriptor)) != directory.metadata:
                raise RuntimeError(f"{self.label} directory metadata changed while pinned: {directory.relative}")
        for file in self.files:
            if _pinned_metadata(os.fstat(file.descriptor)) != file.metadata:
                raise RuntimeError(f"{self.label} file metadata changed while pinned: {file.relative}")
            if file.sha256 is not None and _sha256_descriptor(file.descriptor) != file.sha256:
                raise RuntimeError(f"{self.label} file content changed while pinned: {file.relative}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        for file in self.files:
            try:
                os.close(file.descriptor)
            except BaseException as exc:
                errors.append((f"file descriptor {file.relative}", exc))
        for directory in reversed(self.directories):
            for close_label, descriptor in (
                (f"directory watch {directory.relative}", directory.event_descriptor),
                (f"directory descriptor {directory.relative}", directory.descriptor),
            ):
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    errors.append((close_label, exc))
        _finish_cleanup(
            primary_error,
            errors,
            context=f"{self.label} guard",
        )


def _open_real_directory_chain(
    path: Path,
    *,
    create: bool,
    mode: int = 0o700,
) -> _DirectoryWitness:
    path = Path(path)
    absolute = Path(os.path.abspath(path))
    if absolute != path or Path(os.path.normpath(os.fspath(path))) != path:
        raise RuntimeError(f"directory path must be absolute and canonical: {path}")
    if not absolute.anchor:
        raise RuntimeError(f"directory path must have an absolute anchor: {path}")

    descriptor = os.open(absolute.anchor, _directory_flags())
    current = Path(absolute.anchor)
    try:
        for component in absolute.parts[1:]:
            current /= component
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise RuntimeError(f"directory component is missing: {current}") from None
                with contextlib.suppress(FileExistsError):
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                try:
                    child = os.open(component, _directory_flags(), dir_fd=descriptor)
                except OSError as exc:
                    raise RuntimeError(f"created directory component could not be opened safely: {current}") from exc
            except OSError as exc:
                raise RuntimeError(f"directory component must be a real directory without symlinks: {current}") from exc
            os.close(descriptor)
            descriptor = child
        return _DirectoryWitness.from_descriptor(absolute, descriptor)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _open_child_directory(
    parent: _DirectoryWitness,
    name: str,
    *,
    create: bool,
    mode: int = 0o700,
) -> _DirectoryWitness:
    name = _safe_component(name, name="directory component")
    parent.verify()
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent.descriptor)
    except FileNotFoundError:
        if not create:
            raise RuntimeError(f"directory component is missing: {parent.path / name}") from None
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, mode=mode, dir_fd=parent.descriptor)
        try:
            descriptor = os.open(name, _directory_flags(), dir_fd=parent.descriptor)
        except OSError as exc:
            raise RuntimeError(f"created directory component could not be opened safely: {parent.path / name}") from exc
    except OSError as exc:
        raise RuntimeError(
            f"directory component must be a real directory without symlinks: {parent.path / name}"
        ) from exc
    return _DirectoryWitness.from_descriptor(parent.path / name, descriptor)


def _stat_snapshot(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _copy_regular_file(
    source_directory: int,
    destination_directory: int,
    name: str,
    expected: os.stat_result,
) -> None:
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    source = os.open(name, source_flags, dir_fd=source_directory)
    destination: int | None = None
    try:
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode) or _stat_snapshot(before) != _stat_snapshot(expected):
            raise RuntimeError(f"source file changed before private snapshot copy: {name}")
        destination = os.open(
            name,
            destination_flags,
            0o600,
            dir_fd=destination_directory,
        )
        copied = 0
        while True:
            chunk = os.read(source, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                view = view[written:]
            copied += len(chunk)
        os.fsync(destination)
        after = os.fstat(source)
        rebound = os.stat(name, dir_fd=source_directory, follow_symlinks=False)
        if (
            copied != before.st_size
            or _stat_snapshot(after) != _stat_snapshot(before)
            or _stat_snapshot(rebound) != _stat_snapshot(before)
        ):
            raise RuntimeError(f"source file changed during private snapshot copy: {name}")
    finally:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        if destination is not None:
            try:
                os.close(destination)
            except BaseException as exc:
                errors.append(("snapshot destination file", exc))
        try:
            os.close(source)
        except BaseException as exc:
            errors.append(("snapshot source file", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context=f"private snapshot file {name}",
        )


def _copy_tree(
    source_directory: int,
    destination_directory: int,
    *,
    relative: Path = Path(),
) -> int:
    directory_before = os.fstat(source_directory)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise RuntimeError(f"snapshot source is not a directory: {relative}")
    names = sorted(os.listdir(source_directory))
    regular_files = 0
    for name in names:
        entry_relative = relative / name
        before = os.stat(name, dir_fd=source_directory, follow_symlinks=False)
        if stat.S_ISREG(before.st_mode):
            _copy_regular_file(
                source_directory,
                destination_directory,
                name,
                before,
            )
            regular_files += 1
            continue
        if stat.S_ISDIR(before.st_mode):
            os.mkdir(name, mode=0o700, dir_fd=destination_directory)
            source_child = os.open(name, _directory_flags(), dir_fd=source_directory)
            destination_child: int | None = None
            try:
                destination_child = os.open(
                    name,
                    _directory_flags(),
                    dir_fd=destination_directory,
                )
                child_before = os.fstat(source_child)
                if _stat_snapshot(child_before) != _stat_snapshot(before):
                    raise RuntimeError(f"source directory changed before snapshot copy: {entry_relative}")
                regular_files += _copy_tree(
                    source_child,
                    destination_child,
                    relative=entry_relative,
                )
                child_after = os.fstat(source_child)
                rebound = os.stat(
                    name,
                    dir_fd=source_directory,
                    follow_symlinks=False,
                )
                if _stat_snapshot(child_after) != _stat_snapshot(child_before) or _stat_snapshot(
                    rebound
                ) != _stat_snapshot(child_before):
                    raise RuntimeError(f"source directory changed during snapshot copy: {entry_relative}")
                os.fsync(destination_child)
            finally:
                primary_error = sys.exception()
                errors: list[tuple[str, BaseException]] = []
                if destination_child is not None:
                    try:
                        os.close(destination_child)
                    except BaseException as exc:
                        errors.append(("destination child descriptor", exc))
                try:
                    os.close(source_child)
                except BaseException as exc:
                    errors.append(("source child descriptor", exc))
                _finish_cleanup(
                    primary_error,
                    errors,
                    context=f"private snapshot directory {entry_relative}",
                )
            continue
        raise RuntimeError(f"snapshot source must contain only regular files and directories: {entry_relative}")

    directory_after = os.fstat(source_directory)
    if sorted(os.listdir(source_directory)) != names or _stat_snapshot(directory_after) != _stat_snapshot(
        directory_before
    ):
        raise RuntimeError(f"snapshot source directory changed during copy: {relative}")
    os.fsync(destination_directory)
    return regular_files


def _remove_tree_contents(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(value.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                _remove_tree_contents(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


@dataclasses.dataclass
class _PrivateSnapshot:
    training_root: _DirectoryWitness
    root: _DirectoryWitness
    name: str
    params: Path
    assets: Path
    norm_root: Path
    _closed: bool = False

    @property
    def proc_root(self) -> Path:
        return Path("/proc/self/fd") / str(self.root.descriptor)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        try:
            self.root.verify()
            _remove_tree_contents(self.root.descriptor)
            self.root.verify()
        except BaseException as exc:
            errors.append(("removing snapshot contents", exc))
        try:
            self.root.close()
        except BaseException as exc:
            errors.append(("closing snapshot root descriptor", exc))
        try:
            rebound = os.stat(
                self.name,
                dir_fd=self.training_root.descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(rebound.st_mode) or (rebound.st_dev, rebound.st_ino) != (
                self.root.device,
                self.root.inode,
            ):
                raise RuntimeError(f"private snapshot pathname changed before cleanup: {self.root.path}")
            os.rmdir(self.name, dir_fd=self.training_root.descriptor)
            os.fsync(self.training_root.descriptor)
        except BaseException as exc:
            errors.append(("removing snapshot root", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="private snapshot",
        )


def _create_private_snapshot(
    training_root: _DirectoryWitness,
    *,
    source_params: _DirectoryWitness,
    source_norm: _DirectoryWitness,
    asset_id: str,
) -> _PrivateSnapshot:
    training_root.verify()
    for _ in range(128):
        name = f".rlt-stage2-snapshot-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=training_root.descriptor)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("could not allocate a unique private Stage 2 snapshot")
    root = _open_child_directory(training_root, name, create=False)
    snapshot = _PrivateSnapshot(
        training_root=training_root,
        root=root,
        name=name,
        params=root.path / "params",
        assets=root.path / "assets",
        norm_root=root.path / "assets" / asset_id,
    )
    try:
        os.fchmod(root.descriptor, 0o700)
        with _open_child_directory(root, "params", create=True) as params:
            if _copy_tree(source_params.descriptor, params.descriptor) <= 0:
                raise RuntimeError("checkpoint params snapshot must contain regular files")
        with (
            _open_child_directory(root, "assets", create=True) as assets,
            _open_child_directory(assets, asset_id, create=True) as norm,
        ):
            if _copy_tree(source_norm.descriptor, norm.descriptor) <= 0:
                raise RuntimeError("checkpoint norm snapshot must contain regular files")
        os.fsync(root.descriptor)
        root.verify()
        return snapshot
    except BaseException as primary_error:
        try:
            snapshot.close()
        except BaseException as cleanup_error:
            primary_error.add_note(f"private snapshot cleanup failed: {cleanup_error}")
        raise


def _is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_canonical_directory(
    value: object,
    *,
    name: str,
    required: bool,
) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute canonical path")
    normalized = Path(os.path.normpath(os.fspath(value)))
    if normalized != value:
        raise ValueError(f"{name} must be an absolute canonical path")
    if os.path.lexists(value):
        if value.is_symlink():
            raise ValueError(f"{name} must not be a symlink")
        if not value.is_dir():
            raise NotADirectoryError(f"{name} is not a directory: {value}")
        try:
            resolved = value.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{name} could not be resolved safely: {value}") from exc
        if resolved != value:
            raise ValueError(f"{name} must not contain symlinked path components: {value}")
    elif required:
        raise FileNotFoundError(f"{name} directory does not exist: {value}")
    else:
        try:
            resolved = value.resolve(strict=False)
        except OSError as exc:
            raise ValueError(f"{name} could not be resolved safely: {value}") from exc
        if resolved != value:
            raise ValueError(f"{name} must not contain symlinked path components: {value}")
    return value


def _validate_positive_exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclasses.dataclass(frozen=True)
class BuildCacheConfig:
    checkpoint: Path
    batch: Path
    training_root: Path
    round_id: str
    stage2_config: str = "rl_token_stage2"
    stage1_config: str = "rl_token_stage1"
    micro_batch_size: int = 4
    num_workers: int = 0
    sampler_num_steps: int = 10

    def validate(self) -> None:
        checkpoint = _validate_canonical_directory(
            self.checkpoint,
            name="checkpoint",
            required=True,
        )
        if not checkpoint.name.isdigit() or (checkpoint.name != "0" and checkpoint.name.startswith("0")):
            raise ValueError("checkpoint must be a canonical numeric Stage 1 step directory")
        batch = _validate_canonical_directory(
            self.batch,
            name="batch",
            required=True,
        )
        training_root = _validate_canonical_directory(
            self.training_root,
            name="training_root",
            required=False,
        )
        if type(self.round_id) is not str:
            raise TypeError("round_id must match round_NNNNNN")
        if _ROUND_ID.fullmatch(self.round_id) is None:
            raise ValueError("round_id must match round_NNNNNN")
        if int(self.round_id.removeprefix("round_")) <= 0:
            raise ValueError("round_id must be a positive round_NNNNNN")
        for field in ("stage1_config", "stage2_config"):
            value = getattr(self, field)
            if type(value) is not str:
                raise TypeError(f"{field} must be an exact nonempty string")
            if not value or value.strip() != value:
                raise ValueError(f"{field} must be an exact nonempty string")
        _validate_positive_exact_int(
            self.micro_batch_size,
            name="micro_batch_size",
        )
        if type(self.num_workers) is not int:
            raise TypeError("num_workers must be the exact integer zero")
        if self.num_workers != 0:
            raise ValueError("num_workers must be the exact integer zero")
        _validate_positive_exact_int(
            self.sampler_num_steps,
            name="sampler_num_steps",
        )

        repo_root = Path(__file__).resolve().parents[3]
        protected = {
            "checkpoint": checkpoint,
            "batch": batch,
            "repo": repo_root,
        }
        for label, path in protected.items():
            if _is_path_within(training_root, path) or _is_path_within(path, training_root):
                raise ValueError(
                    f"training_root must not be nested in or overlap {label}: {training_root} versus {path}"
                )
        if _is_path_within(checkpoint, batch) or _is_path_within(batch, checkpoint):
            raise ValueError("checkpoint and batch directories must not be nested or overlap")


@dataclasses.dataclass(frozen=True)
class LoadedFrozenModel:
    model: object
    train_config: object
    data_config: object
    norm_stats: Mapping[str, object]
    input_transform: object
    feature_id: str
    checkpoint_sha256: str
    norm_stats_sha256: str
    loaded_parameter_sha256: str
    loaded_norm_stats_sha256: str


@dataclasses.dataclass(frozen=True)
class BuildCacheResult:
    destination: Path
    manifest_sha256: str
    feature_rows: int
    transition_rows: int
    feature_identity: str
    batch_id: str


@dataclasses.dataclass(frozen=True)
class _PipelineOutcome:
    result: BuildCacheResult
    final_witness: _CacheRootWitness


def _runtime_imports() -> SimpleNamespace:
    from flax import nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    from openpi.models import model as model_api  # noqa: PLC0415
    from openpi.training import checkpoints  # noqa: PLC0415
    from openpi.training.rl_token import config as training_config  # noqa: PLC0415
    from openpi.training.rl_token.stage2 import admission  # noqa: PLC0415
    from openpi.training.rl_token.stage2 import cache  # noqa: PLC0415
    from openpi.training.rl_token.stage2 import feature_extractor  # noqa: PLC0415
    from openpi.training.rl_token.stage2 import feature_identity  # noqa: PLC0415
    from openpi.training.rl_token.stage2 import transitions  # noqa: PLC0415

    _verify_runtime_module_provenance(
        {
            "openpi.models.model": model_api,
            "openpi.training.rl_token.stage2.admission": admission,
            "openpi.training.rl_token.stage2.cache": cache,
            "openpi.training.rl_token.stage2.feature_extractor": feature_extractor,
            "openpi.training.rl_token.stage2.feature_identity": feature_identity,
            "openpi.training.rl_token.stage2.transitions": transitions,
            "openpi.training.checkpoints": checkpoints,
            "openpi.training.rl_token.config": training_config,
        }
    )

    return SimpleNamespace(
        admission=admission,
        cache=cache,
        checkpoints=checkpoints,
        feature_extractor=feature_extractor,
        feature_identity=feature_identity,
        jax=jax,
        model_api=model_api,
        nnx=nnx,
        training_config=training_config,
        transitions=transitions,
    )


def _verify_runtime_module_provenance(modules: Mapping[str, object]) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if type(raw_path) is not str:
            raise RuntimeError(f"runtime module {name} has no concrete __file__ provenance")
        try:
            module_path = Path(raw_path).resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"runtime module {name} path is unavailable: {raw_path}") from exc
        if not _is_path_within(module_path, repo_root):
            raise RuntimeError(f"runtime module {name} was imported outside the current worktree: {module_path}")


def current_git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if _LOWERCASE_SHA1.fullmatch(commit) is None:
        raise RuntimeError(
            f"unexpected git commit identity; expected 40 lowercase hexadecimal characters, got {commit!r}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("Stage 2 requires a clean Git worktree; tracked, index, and untracked changes are forbidden")
    return commit


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a lowercase SHA-256 hex string")
    if _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _asset_id(train_config: object) -> str:
    data_factory = getattr(train_config, "data", None)
    assets = getattr(data_factory, "assets", None)
    value = getattr(assets, "asset_id", None) or getattr(data_factory, "repo_id", None)
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("Stage 2 requires a nonempty checkpoint asset_id")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("Stage 2 asset_id must be a single safe path component")
    return value


def _train_config_for_checkpoint_assets(
    train_config: object,
    checkpoint_assets: Path,
) -> object:
    data_factory = getattr(train_config, "data", None)
    assets = getattr(data_factory, "assets", None)
    if assets is None:
        raise TypeError("Stage 2 train config data factory must expose assets")
    try:
        checkpoint_asset_config = dataclasses.replace(
            assets,
            assets_dir=str(checkpoint_assets),
        )
        checkpoint_data_factory = dataclasses.replace(
            data_factory,
            assets=checkpoint_asset_config,
        )
        return dataclasses.replace(
            train_config,
            data=checkpoint_data_factory,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Stage 2 train config and data assets must be replaceable dataclasses") from exc


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _input_transform_types(input_transform: object) -> list[str]:
    values = getattr(input_transform, "transforms", None)
    if values is None:
        return [_qualified_type(input_transform)]
    try:
        return [_qualified_type(value) for value in values]
    except TypeError as exc:
        raise TypeError("Stage 2 input transform must expose a finite transform sequence") from exc


def _stable_load_error(
    issues: list[tuple[str, BaseException]],
    load_error: BaseException | None,
) -> None:
    if not issues:
        if load_error is not None:
            raise load_error
        return
    details = "; ".join(f"{label}: {error}" for label, error in issues)
    error = RuntimeError(f"frozen checkpoint identity guard failed: {details}")
    if load_error is not None:
        raise error from load_error
    raise error


def _hash_verified_tree(
    runtime: SimpleNamespace,
    path: Path,
    witness: _DirectoryWitness,
    *,
    name: str,
) -> str:
    witness.verify()
    value = runtime.feature_identity.checkpoint_tree_sha256(path)
    value = _require_sha256(value, name=name)
    witness.verify()
    return value


def _norm_stats_semantic_sha256(norm_stats: Mapping[str, object]) -> str:
    if not norm_stats:
        raise ValueError("checkpoint norm stats mapping must not be empty")
    digest = hashlib.sha256()
    keys = list(norm_stats)
    if any(type(key) is not str or not key for key in keys):
        raise TypeError("checkpoint norm stats keys must be exact nonempty strings")
    for key in sorted(keys):
        stats_value = norm_stats[key]
        if not dataclasses.is_dataclass(stats_value) or isinstance(stats_value, type):
            raise TypeError(f"checkpoint norm stats {key!r} must be a concrete dataclass instance")
        fields = sorted(dataclasses.fields(stats_value), key=lambda field: field.name)
        if not fields:
            raise ValueError(f"checkpoint norm stats {key!r} has no fields")
        digest.update(
            _canonical_json_bytes(
                {
                    "key": key,
                    "type": _qualified_type(stats_value),
                    "fields": [field.name for field in fields],
                }
            )
        )
        for field in fields:
            raw_value = getattr(stats_value, field.name)
            if raw_value is None:
                digest.update(
                    _canonical_json_bytes(
                        {
                            "key": key,
                            "field": field.name,
                            "value": None,
                        }
                    )
                )
                continue
            array = np.asarray(raw_value)
            if array.dtype.kind != "f":
                raise TypeError(f"checkpoint norm stats {key}.{field.name} must have a floating dtype")
            if not np.isfinite(array).all():
                raise ValueError(f"checkpoint norm stats {key}.{field.name} must contain only finite values")
            contiguous = np.ascontiguousarray(array)
            digest.update(
                _canonical_json_bytes(
                    {
                        "key": key,
                        "field": field.name,
                        "dtype": contiguous.dtype.str,
                        "shape": list(contiguous.shape),
                    }
                )
            )
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validated_loaded_parameter_sha256(
    runtime: SimpleNamespace,
    model: object,
) -> str:
    state = runtime.nnx.state(model, runtime.nnx.Param)
    parameter_state = state.filter(runtime.nnx.Param)
    flat_state = parameter_state.flat_state()
    leaves: list[tuple[bytes, list[dict[str, Any]], object]] = []
    for path, variable in flat_state.items():
        canonical_path = [
            {
                "type": _qualified_type(component),
                "value": runtime.feature_identity.canonical_config_value(component),
            }
            for component in path
        ]
        path_bytes = _canonical_json_bytes({"path": canonical_path})
        leaves.append((path_bytes, canonical_path, variable))
    if not leaves:
        raise ValueError("loaded frozen model must contain at least one nnx.Param")
    digest = hashlib.sha256()
    for _, canonical_path, variable in sorted(leaves, key=lambda item: item[0]):
        value = np.asarray(runtime.jax.device_get(variable.value))
        if value.dtype != np.dtype(np.float32):
            raise TypeError(
                f"loaded frozen model parameter {canonical_path!r} must be exact float32 master params; "
                f"got {value.dtype}"
            )
        digest.update(
            _canonical_json_bytes(
                {
                    "path": canonical_path,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            )
        )
        iterator = np.nditer(
            value,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="C",
            buffersize=max(1, _PARAMETER_HASH_CHUNK_BYTES // value.dtype.itemsize),
        )
        for chunk in iterator:
            if not np.isfinite(chunk).all():
                raise ValueError(f"loaded frozen model parameter {canonical_path!r} must contain only finite values")
            digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return digest.hexdigest()


def load_model_and_transforms(
    config: BuildCacheConfig,
    batch: object,
    *,
    runtime: SimpleNamespace | None = None,
    code_commit: str | None = None,
    training_root_witness: _DirectoryWitness | None = None,
) -> LoadedFrozenModel:
    runtime = _runtime_imports() if runtime is None else runtime
    commit = current_git_commit() if code_commit is None else code_commit
    if type(commit) is not str or _LOWERCASE_SHA1.fullmatch(commit) is None:
        raise ValueError("code_commit must be a full lowercase 40-hex Git SHA-1")

    train_config = runtime.training_config.get_stage1_config(config.stage1_config)
    runtime.training_config.get_stage2_config(config.stage2_config)
    if getattr(getattr(train_config, "model", None), "rl_token_enabled", None) is not True:
        raise ValueError("Stage 2 requires train_config.model.rl_token_enabled=True")
    asset_id = _asset_id(train_config)
    source_params_path = config.checkpoint / "params"
    source_assets_path = config.checkpoint / "assets"
    source_norm_path = source_assets_path / asset_id
    owned_training_root: _DirectoryWitness | None = None
    if training_root_witness is None:
        owned_training_root = _open_real_directory_chain(
            config.training_root,
            create=True,
        )
        training_root_witness = owned_training_root

    model: object | None = None
    norm_stats: Mapping[str, object] | None = None
    data_config: object | None = None
    input_transform: object | None = None
    loaded_parameter_sha256: str | None = None
    loaded_norm_stats_sha256: str | None = None
    load_error: BaseException | None = None
    guard_issues: list[tuple[str, BaseException]] = []
    cleanup_issues: list[tuple[str, BaseException]] = []
    checkpoint_before: str | None = None
    norm_before: str | None = None
    patched_train_config: object | None = None
    snapshot: _PrivateSnapshot | None = None
    snapshot_guard: _PinnedSnapshotTree | None = None
    try:
        with _recording_exit_stack(
            cleanup_issues,
            label="source directory witnesses",
            context="frozen checkpoint source acquisition",
        ) as source_stack:
            source_params = source_stack.enter_context(_open_real_directory_chain(source_params_path, create=False))
            source_norm = source_stack.enter_context(_open_real_directory_chain(source_norm_path, create=False))
            checkpoint_before = _hash_verified_tree(
                runtime,
                source_params_path,
                source_params,
                name="checkpoint_sha256",
            )
            norm_before = _hash_verified_tree(
                runtime,
                source_norm_path,
                source_norm,
                name="norm_stats_sha256",
            )
            snapshot = _create_private_snapshot(
                training_root_witness,
                source_params=source_params,
                source_norm=source_norm,
                asset_id=asset_id,
            )
            snapshot.root.verify()
            with _open_real_directory_chain(snapshot.params, create=False) as snapshot_params:
                snapshot_checkpoint_before = _hash_verified_tree(
                    runtime,
                    snapshot.params,
                    snapshot_params,
                    name="snapshot checkpoint_sha256",
                )
            with _open_real_directory_chain(snapshot.norm_root, create=False) as snapshot_norm:
                snapshot_norm_before = _hash_verified_tree(
                    runtime,
                    snapshot.norm_root,
                    snapshot_norm,
                    name="snapshot norm_stats_sha256",
                )
            if snapshot_checkpoint_before != checkpoint_before:
                raise RuntimeError(
                    "private checkpoint snapshot does not match verified source params: "
                    f"source {checkpoint_before}, snapshot {snapshot_checkpoint_before}"
                )
            if snapshot_norm_before != norm_before:
                raise RuntimeError(
                    "private norm snapshot does not match verified source assets: "
                    f"source {norm_before}, snapshot {snapshot_norm_before}"
                )
            checkpoint_post_copy = _hash_verified_tree(
                runtime,
                source_params_path,
                source_params,
                name="checkpoint_sha256 after snapshot copy",
            )
            norm_post_copy = _hash_verified_tree(
                runtime,
                source_norm_path,
                source_norm,
                name="norm_stats_sha256 after snapshot copy",
            )
            if checkpoint_post_copy != checkpoint_before:
                raise RuntimeError(
                    "checkpoint params tree changed during private snapshot copy: "
                    f"before {checkpoint_before}, after {checkpoint_post_copy}"
                )
            if norm_post_copy != norm_before:
                raise RuntimeError(
                    "norm asset tree changed during private snapshot copy: "
                    f"before {norm_before}, after {norm_post_copy}"
                )

            snapshot_guard = _PinnedSnapshotTree.open(snapshot.root.descriptor)
            snapshot_guard.verify()
            with _open_real_directory_chain(snapshot.params, create=False) as snapshot_params:
                checkpoint_under_guard = _hash_verified_tree(
                    runtime,
                    snapshot.params,
                    snapshot_params,
                    name="snapshot checkpoint_sha256 under mutation guard",
                )
            with _open_real_directory_chain(snapshot.norm_root, create=False) as snapshot_norm:
                norm_under_guard = _hash_verified_tree(
                    runtime,
                    snapshot.norm_root,
                    snapshot_norm,
                    name="snapshot norm_stats_sha256 under mutation guard",
                )
            snapshot_guard.verify()
            if checkpoint_under_guard != snapshot_checkpoint_before:
                raise RuntimeError(
                    "private checkpoint snapshot changed while installing mutation guard: "
                    f"before {snapshot_checkpoint_before}, guarded {checkpoint_under_guard}"
                )
            if norm_under_guard != snapshot_norm_before:
                raise RuntimeError(
                    "private norm snapshot changed while installing mutation guard: "
                    f"before {snapshot_norm_before}, guarded {norm_under_guard}"
                )

            load_witnesses = contextlib.ExitStack()
            try:
                snapshot_params_load = load_witnesses.enter_context(
                    _open_real_directory_chain(snapshot.params, create=False)
                )
                snapshot_assets_load = load_witnesses.enter_context(
                    _open_real_directory_chain(snapshot.assets, create=False)
                )
                load_witnesses.enter_context(_open_real_directory_chain(snapshot.norm_root, create=False))
            except BaseException:
                primary_error = sys.exception()
                errors: list[tuple[str, BaseException]] = []
                try:
                    load_witnesses.close()
                except BaseException as exc:
                    errors.append(("private snapshot load witnesses", exc))
                _finish_cleanup(
                    primary_error,
                    errors,
                    context="frozen checkpoint load acquisition",
                )
                raise
            snapshot_load_assets = Path("/proc/self/fd") / str(snapshot_assets_load.descriptor)
            snapshot_load_params = Path("/proc/self/fd") / str(snapshot_params_load.descriptor)
            try:
                patched_train_config = _train_config_for_checkpoint_assets(
                    train_config,
                    snapshot_load_assets,
                )
            except BaseException:
                primary_error = sys.exception()
                errors: list[tuple[str, BaseException]] = []
                try:
                    load_witnesses.close()
                except BaseException as exc:
                    errors.append(("private snapshot load witnesses", exc))
                _finish_cleanup(
                    primary_error,
                    errors,
                    context="frozen checkpoint load configuration",
                )
                raise
            try:
                snapshot.root.verify()
                params = runtime.model_api.restore_params(snapshot_load_params)
                model = patched_train_config.model.load(params)
                norm_stats = runtime.checkpoints.load_norm_stats(
                    snapshot_load_assets,
                    asset_id,
                )
                if norm_stats is None:
                    raise FileNotFoundError(
                        f"checkpoint norm stats are missing from private snapshot: {snapshot.norm_root}"
                    )
                if not isinstance(norm_stats, Mapping):
                    raise TypeError("checkpoint norm stats must be a mapping")
                loaded_norm_stats_sha256 = _norm_stats_semantic_sha256(norm_stats)
                data_config, input_transform = runtime.feature_extractor.build_stage2_input_transform(
                    patched_train_config,
                    batch,
                    norm_stats,
                )
                if getattr(data_config, "asset_id", None) != asset_id:
                    raise ValueError(
                        f"Stage 2 data_config.asset_id {getattr(data_config, 'asset_id', None)!r} "
                        f"does not match checkpoint asset_id {asset_id!r}"
                    )
                loaded_parameter_sha256 = _validated_loaded_parameter_sha256(
                    runtime,
                    model,
                )
            except BaseException as exc:
                load_error = exc
            finally:
                try:
                    snapshot.root.verify()
                    with _open_real_directory_chain(snapshot.params, create=False) as snapshot_params:
                        checkpoint_after = _hash_verified_tree(
                            runtime,
                            snapshot.params,
                            snapshot_params,
                            name="snapshot checkpoint_sha256 after load",
                        )
                    if checkpoint_after != snapshot_checkpoint_before:
                        raise RuntimeError(
                            "private checkpoint snapshot changed during restore/load: "
                            f"before {snapshot_checkpoint_before}, after {checkpoint_after}"
                        )
                except BaseException as exc:
                    guard_issues.append(("private checkpoint snapshot post-check", exc))
                try:
                    snapshot.root.verify()
                    with _open_real_directory_chain(snapshot.norm_root, create=False) as snapshot_norm:
                        norm_after = _hash_verified_tree(
                            runtime,
                            snapshot.norm_root,
                            snapshot_norm,
                            name="snapshot norm_stats_sha256 after load",
                        )
                    if norm_after != snapshot_norm_before:
                        raise RuntimeError(
                            "private norm snapshot changed during load: "
                            f"before {snapshot_norm_before}, after {norm_after}"
                        )
                except BaseException as exc:
                    guard_issues.append(("private norm snapshot post-check", exc))
                try:
                    checkpoint_final = _hash_verified_tree(
                        runtime,
                        source_params_path,
                        source_params,
                        name="checkpoint_sha256 final",
                    )
                    if checkpoint_final != checkpoint_before:
                        raise RuntimeError(
                            "checkpoint params source changed during restore/load: "
                            f"before {checkpoint_before}, final {checkpoint_final}"
                        )
                except BaseException as exc:
                    guard_issues.append(("checkpoint params final check", exc))
                try:
                    norm_final = _hash_verified_tree(
                        runtime,
                        source_norm_path,
                        source_norm,
                        name="norm_stats_sha256 final",
                    )
                    if norm_final != norm_before:
                        raise RuntimeError(
                            f"norm assets source changed during load: before {norm_before}, final {norm_final}"
                        )
                except BaseException as exc:
                    guard_issues.append(("norm assets final check", exc))
                try:
                    snapshot_guard.verify()
                except BaseException as exc:
                    guard_issues.append(("private snapshot recursive mutation guard", exc))
                try:
                    snapshot_guard.close()
                except BaseException as exc:
                    cleanup_issues.append(("private snapshot mutation guard", exc))
                finally:
                    snapshot_guard = None
                try:
                    load_witnesses.close()
                except BaseException as exc:
                    cleanup_issues.append(("private snapshot load witnesses", exc))
    finally:
        active_error = sys.exception()
        if snapshot_guard is not None:
            try:
                snapshot_guard.verify()
            except BaseException as exc:
                guard_issues.append(("private snapshot recursive mutation guard", exc))
            try:
                snapshot_guard.close()
            except BaseException as exc:
                cleanup_issues.append(("private snapshot mutation guard", exc))
        if snapshot is not None:
            try:
                snapshot.close()
            except BaseException as exc:
                cleanup_issues.append(("private snapshot", exc))
        if owned_training_root is not None:
            try:
                owned_training_root.close()
            except BaseException as exc:
                cleanup_issues.append(("owned training root", exc))
        if active_error is not None:
            for label, error in guard_issues:
                active_error.add_note(f"frozen checkpoint load guard failed during {label}: {error}")
            _finish_cleanup(
                active_error,
                cleanup_issues,
                context="frozen checkpoint load",
            )
        elif guard_issues:
            guard_issues.extend((f"cleanup {label}", error) for label, error in cleanup_issues)
        elif load_error is not None:
            _finish_cleanup(
                load_error,
                cleanup_issues,
                context="frozen checkpoint load",
            )
        else:
            _finish_cleanup(
                None,
                cleanup_issues,
                context="frozen checkpoint load",
            )

    _stable_load_error(guard_issues, load_error)
    if checkpoint_before is None or norm_before is None or patched_train_config is None:
        raise RuntimeError("frozen checkpoint snapshot loading completed without verified identities")
    if (
        model is None
        or norm_stats is None
        or data_config is None
        or input_transform is None
        or loaded_parameter_sha256 is None
        or loaded_norm_stats_sha256 is None
    ):
        raise RuntimeError("frozen model loading completed without all required artifacts")

    base_model_config_signature = runtime.feature_identity.transform_signature(patched_train_config.model)
    model_config_signature = {
        "model": base_model_config_signature,
        "loaded_parameter_sha256": loaded_parameter_sha256,
    }
    base_transform_config_signature = runtime.feature_identity.transform_signature(
        {
            "stage1_config": config.stage1_config,
            "stage2_config": config.stage2_config,
            "asset_id": asset_id,
            "repack": data_config.repack_transforms,
            "data": data_config.data_transforms,
            "model": data_config.model_transforms,
            "stage2_input_transform_types": _input_transform_types(input_transform),
            "default_prompt": runtime.feature_extractor.DEFAULT_PROMPT,
            "use_quantile_norm": data_config.use_quantile_norm,
            "video_tolerance_s": data_config.video_tolerance_s,
        }
    )
    transform_config_signature = {
        "transform": base_transform_config_signature,
        "loaded_norm_stats_sha256": loaded_norm_stats_sha256,
    }
    feature_id = runtime.feature_identity.build_feature_identity(
        runtime.feature_identity.FeatureIdentityInput(
            checkpoint_sha256=checkpoint_before,
            norm_stats_sha256=norm_before,
            model_config=model_config_signature,
            transform_config=transform_config_signature,
            sampler_num_steps=config.sampler_num_steps,
            seed_version=_FEATURE_SEED_VERSION,
            code_commit=commit,
        )
    )
    feature_id = _require_sha256(feature_id, name="feature_identity")
    return LoadedFrozenModel(
        model=model,
        train_config=patched_train_config,
        data_config=data_config,
        norm_stats=norm_stats,
        input_transform=input_transform,
        feature_id=feature_id,
        checkpoint_sha256=checkpoint_before,
        norm_stats_sha256=norm_before,
        loaded_parameter_sha256=loaded_parameter_sha256,
        loaded_norm_stats_sha256=loaded_norm_stats_sha256,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_strict_json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    absolute = Path(os.path.abspath(path))
    if absolute != path:
        raise RuntimeError(f"admission path must be absolute and canonical: {path}")
    parent = path.parent
    try:
        if parent.resolve(strict=True) != parent:
            raise RuntimeError(f"admission parent must not contain symlinks: {parent}")
    except OSError as exc:
        raise RuntimeError(f"admission parent is missing or invalid: {parent}") from exc

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"admission must be a regular file without symlinks: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"admission must be a regular file: {path}")
        if before.st_size > _ADMISSION_MAX_BYTES:
            raise RuntimeError(f"admission exceeds {_ADMISSION_MAX_BYTES} bytes: {path}")
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > _ADMISSION_MAX_BYTES:
                raise RuntimeError(f"admission exceeded read limit while reading: {path}")
        after = os.fstat(descriptor)
        if (
            _stat_identity(after) != _stat_identity(before)
            or bytes_read != before.st_size
            or not stat.S_ISREG(after.st_mode)
        ):
            raise RuntimeError(f"admission changed while being read: {path}")
        try:
            bound = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"admission pathname changed while being read: {path}") from exc
        if _stat_identity(bound) != _stat_identity(before) or not stat.S_ISREG(bound.st_mode):
            raise RuntimeError(f"admission pathname changed while being read: {path}")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError(f"admission is invalid strict JSON: {path}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"admission must contain one JSON object: {path}")
    return value


def _read_strict_json_object_at(
    parent: _DirectoryWitness,
    name: str,
) -> dict[str, Any]:
    name = _safe_component(name, name="admission filename")
    path = parent.path / name
    parent.verify()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    except OSError as exc:
        raise RuntimeError(f"admission must be a regular file without symlinks: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"admission must be a regular file: {path}")
        if before.st_size > _ADMISSION_MAX_BYTES:
            raise RuntimeError(f"admission exceeds {_ADMISSION_MAX_BYTES} bytes: {path}")
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > _ADMISSION_MAX_BYTES:
                raise RuntimeError(f"admission exceeded read limit while reading: {path}")
        after = os.fstat(descriptor)
        if (
            _stat_identity(after) != _stat_identity(before)
            or bytes_read != before.st_size
            or not stat.S_ISREG(after.st_mode)
        ):
            raise RuntimeError(f"admission changed while being read: {path}")
        try:
            rebound = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(f"admission pathname changed while being read: {path}") from exc
        if _stat_identity(rebound) != _stat_identity(before) or not stat.S_ISREG(rebound.st_mode):
            raise RuntimeError(f"admission pathname changed while being read: {path}")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    parent.verify()
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError(f"admission is invalid strict JSON: {path}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"admission must contain one JSON object: {path}")
    return value


def _before_bound_admission_publish_hook() -> None:
    """Test hook immediately before the held-dirfd admission link."""


def _after_bound_admission_publish_hook() -> None:
    """Test hook immediately after the held-dirfd admission link."""


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write made no forward progress")
        remaining = remaining[written:]


def _atomic_json_noreplace_at(
    parent: _DirectoryWitness,
    name: str,
    payload: dict[str, Any],
) -> None:
    name = _safe_component(name, name="admission filename")
    encoded = _canonical_json_bytes(payload)
    temporary_name: str | None = None
    descriptor: int | None = None
    for _ in range(128):
        candidate = f".{name}.tmp-{secrets.token_hex(16)}"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent.descriptor,
            )
            temporary_name = candidate
            break
        except FileExistsError:
            continue
    if descriptor is None or temporary_name is None:
        raise RuntimeError("could not allocate a unique admission staging file")

    publish_error: BaseException | None = None
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    except BaseException as exc:
        publish_error = exc
    descriptor_to_close = descriptor
    descriptor = None
    try:
        os.close(descriptor_to_close)
    except BaseException as exc:
        if publish_error is None:
            publish_error = exc
        else:
            publish_error.add_note(f"admission staging close also failed: {exc}")

    if publish_error is None:
        entered_hook = False
        try:
            _before_bound_admission_publish_hook()
            entered_hook = True
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent.descriptor,
                dst_dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except BaseException as exc:
            publish_error = exc
        finally:
            if entered_hook:
                try:
                    _after_bound_admission_publish_hook()
                except BaseException as exc:
                    if publish_error is None:
                        publish_error = exc
                    else:
                        publish_error.add_note(f"admission publish hook cleanup also failed: {exc}")

    try:
        os.unlink(temporary_name, dir_fd=parent.descriptor)
    except FileNotFoundError:
        pass
    except BaseException as exc:
        if publish_error is None:
            publish_error = exc
        else:
            publish_error.add_note(f"admission staging cleanup also failed: {exc}")
    try:
        os.fsync(parent.descriptor)
    except BaseException as exc:
        if publish_error is None:
            publish_error = exc
        else:
            publish_error.add_note(f"admission parent fsync also failed: {exc}")
    if publish_error is not None:
        raise publish_error


def _publish_or_verify_round(
    config: BuildCacheConfig,
    batch: object,
    *,
    code_commit: str,
    runtime: SimpleNamespace,
    admissions_witness: _DirectoryWitness | None = None,
) -> Path:
    expected_parent = config.training_root / "admissions"
    owned_witness: _DirectoryWitness | None = None
    if admissions_witness is None:
        owned_witness = _open_real_directory_chain(expected_parent, create=True)
        admissions_witness = owned_witness
    elif admissions_witness.path != expected_parent:
        raise RuntimeError(f"admissions witness {admissions_witness.path} does not match expected {expected_parent}")
    filename = f"{config.round_id}.json"
    admission_path = expected_parent / filename
    try:
        admissions_witness.verify()
        try:
            existing = os.stat(
                filename,
                dir_fd=admissions_witness.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is None:
            admitted_at = datetime.datetime.now(datetime.UTC).isoformat()
            new_payload = runtime.admission.admission_payload(
                batch,
                round_id=config.round_id,
                admitted_at=admitted_at,
                code_commit=code_commit,
            )
            if type(new_payload) is not dict:
                raise RuntimeError("admission_payload must return one JSON object")
            with contextlib.suppress(FileExistsError):
                _atomic_json_noreplace_at(
                    admissions_witness,
                    filename,
                    new_payload,
                )

        payload = _read_strict_json_object_at(admissions_witness, filename)
        payload_round = payload.get("round_id")
        if type(payload_round) is not str or payload_round != config.round_id:
            raise RuntimeError(
                f"existing admission round_id {payload_round!r} does not match requested {config.round_id!r}"
            )
        admitted_at = payload.get("admitted_at")
        admitted_commit = payload.get("code_commit")
        expected = runtime.admission.admission_payload(
            batch,
            round_id=config.round_id,
            admitted_at=admitted_at,
            code_commit=admitted_commit,
        )
        if payload != expected:
            raise RuntimeError(
                f"admission payload does not exactly bind round {config.round_id} "
                f"to immutable batch {getattr(batch, 'batch_id', '<unknown>')}"
            )
        admissions_witness.verify()
        runtime.admission.verify_admission(admission_path, batch)
        admissions_witness.verify()
        if _read_strict_json_object_at(admissions_witness, filename) != payload:
            raise RuntimeError(f"admission changed while verifying immutable round {config.round_id}")
        return admission_path
    finally:
        if owned_witness is not None:
            owned_witness.close()


def _safe_component(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty safe path component")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty safe path component")
    return value


def _cache_identity_fields(
    config: BuildCacheConfig,
    batch: object,
    loaded: LoadedFrozenModel,
    *,
    code_commit: str,
    default_prompt: str,
) -> dict[str, Any]:
    fingerprints = getattr(batch, "episode_fingerprints", None)
    if not isinstance(fingerprints, tuple) or not fingerprints:
        raise ValueError("batch episode_fingerprints must be a nonempty tuple")
    normalized_fingerprints = [
        _require_sha256(value, name=f"episode_fingerprints[{index}]") for index, value in enumerate(fingerprints)
    ]
    if type(default_prompt) is not str or not default_prompt:
        raise ValueError("default_prompt must be a nonempty string")
    asset_id = _safe_component(
        getattr(loaded.data_config, "asset_id", None),
        name="asset_id",
    )
    return {
        "feature_identity": _require_sha256(
            loaded.feature_id,
            name="feature_identity",
        ),
        "checkpoint_sha256": _require_sha256(
            loaded.checkpoint_sha256,
            name="checkpoint_sha256",
        ),
        "norm_stats_sha256": _require_sha256(
            loaded.norm_stats_sha256,
            name="norm_stats_sha256",
        ),
        "loaded_parameter_sha256": _require_sha256(
            loaded.loaded_parameter_sha256,
            name="loaded_parameter_sha256",
        ),
        "loaded_norm_stats_sha256": _require_sha256(
            loaded.loaded_norm_stats_sha256,
            name="loaded_norm_stats_sha256",
        ),
        "batch_id": _safe_component(
            getattr(batch, "batch_id", None),
            name="batch_id",
        ),
        "migration_manifest_sha256": _require_sha256(
            getattr(batch, "manifest_sha256", None),
            name="migration_manifest_sha256",
        ),
        "labels_sha256": _require_sha256(
            getattr(batch, "labels_sha256", None),
            name="labels_sha256",
        ),
        "episode_fingerprints": normalized_fingerprints,
        "round_id": config.round_id,
        "config_name": config.stage1_config,
        "stage1_checkpoint_step": int(config.checkpoint.name),
        **runtime_reward_metadata(config, batch),
        "asset_id": asset_id,
        "sampler_num_steps": config.sampler_num_steps,
        "default_prompt": default_prompt,
        "code_commit": code_commit,
    }


def runtime_reward_metadata(config: BuildCacheConfig, batch: object) -> dict[str, object]:
    """Bind every cache shard to the exact Stage 2 reward interpretation."""
    labels_sha256 = _require_sha256(
        getattr(batch, "labels_sha256", None),
        name="tristate_labels_sha256",
    )
    return {
        "stage1_config": config.stage1_config,
        "stage2_config": config.stage2_config,
        "reward_source": "tristate",
        "reward_label_values": [-1, 0, 1, 2],
        "completion_label": 2,
        "reward_aggregation": "sum_20_frames",
        "reward_schema_version": 1,
        "tristate_labels_sha256": labels_sha256,
    }


@dataclasses.dataclass
class _OutputRoots:
    training_root: _DirectoryWitness
    admissions: _DirectoryWitness
    feature_cache: _DirectoryWitness

    def verify(self) -> None:
        self.training_root.verify()
        self.admissions.verify()
        self.feature_cache.verify()

    def close(self) -> None:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        for label, close in (
            ("feature cache root", self.feature_cache.close),
            ("admissions root", self.admissions.close),
            ("training root", self.training_root.close),
        ):
            try:
                close()
            except BaseException as exc:
                errors.append((label, exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="Stage 2 output roots",
        )

    def __enter__(self) -> _OutputRoots:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _prepare_output_roots(config: BuildCacheConfig) -> _OutputRoots:
    training_root = _open_real_directory_chain(config.training_root, create=True)
    admissions: _DirectoryWitness | None = None
    feature_cache: _DirectoryWitness | None = None
    try:
        admissions = _open_child_directory(
            training_root,
            "admissions",
            create=True,
        )
        feature_cache = _open_child_directory(
            training_root,
            "feature_cache",
            create=True,
        )
        result = _OutputRoots(
            training_root=training_root,
            admissions=admissions,
            feature_cache=feature_cache,
        )
        result.verify()
        return result
    except BaseException:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        for label, close in (
            ("feature cache root", None if feature_cache is None else feature_cache.close),
            ("admissions root", None if admissions is None else admissions.close),
            ("training root", training_root.close),
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append((label, exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="Stage 2 output root preparation",
        )
        raise


def _table_row_count(table: object, *, name: str) -> int:
    if not dataclasses.is_dataclass(table) or isinstance(table, type):
        raise TypeError(f"{name} must be a concrete dataclass table")
    fields = dataclasses.fields(table)
    if not fields:
        raise ValueError(f"{name} must contain at least one field")
    counts: set[int] = set()
    for field in fields:
        value = getattr(table, field.name)
        shape = getattr(value, "shape", None)
        if not isinstance(shape, tuple) or not shape:
            raise ValueError(f"{name}.{field.name} must expose a nonempty tuple shape")
        count = shape[0]
        if type(count) is not int:
            count = int(count)
        counts.add(count)
    if len(counts) != 1:
        raise ValueError(f"{name} fields do not have one exact row count: {sorted(counts)}")
    count = counts.pop()
    if count <= 0:
        raise ValueError(f"{name} must have at least one row")
    return count


def _after_cache_open_hook(destination: Path) -> None:
    del destination


class _HashingWriter:
    def __init__(self, descriptor: int):
        self.descriptor = descriptor
        self.digest = hashlib.sha256()
        self.position = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        view = memoryview(value).cast("B")
        total = len(view)
        while view:
            written = os.write(self.descriptor, view)
            if written <= 0:
                raise OSError("write made no forward progress")
            self.digest.update(view[:written])
            self.position += written
            view = view[written:]
        return total

    def flush(self) -> None:
        return None

    def tell(self) -> int:
        return self.position


@dataclasses.dataclass
class _StagedCache:
    parent: _DirectoryWitness
    root: _DirectoryWitness
    name: str
    manifest: dict[str, Any]
    manifest_sha256: str
    published: bool = False
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        if not self.published:
            try:
                held = os.fstat(self.root.descriptor)
                rebound = os.stat(
                    self.name,
                    dir_fd=self.parent.descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(held.st_mode)
                    or not stat.S_ISDIR(rebound.st_mode)
                    or (held.st_dev, held.st_ino) != (rebound.st_dev, rebound.st_ino)
                ):
                    raise RuntimeError(f"cache staging pathname changed before cleanup: {self.parent.path / self.name}")
                _remove_tree_contents(self.root.descriptor)
                os.fsync(self.root.descriptor)
                os.rmdir(self.name, dir_fd=self.parent.descriptor)
                os.fsync(self.parent.descriptor)
            except BaseException as exc:
                errors.append(("removing cache staging tree", exc))
        try:
            self.root.close()
        except BaseException as exc:
            errors.append(("closing cache staging descriptor", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="cache staging",
        )


def _write_cache_array_at(
    group: _DirectoryWitness,
    *,
    group_name: str,
    field_name: str,
    value: object,
) -> dict[str, Any]:
    field_name = _safe_component(field_name, name="cache field")
    filename = f"{field_name}.npy"
    array = np.ascontiguousarray(value)
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=group.descriptor,
    )
    writer = _HashingWriter(descriptor)
    try:
        np.save(writer, array, allow_pickle=False)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != writer.position:
            raise RuntimeError(f"cache array size changed while writing: {group_name}/{filename}")
    except BaseException as primary_error:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            primary_error.add_note(f"cache array close also failed: {cleanup_error}")
        raise
    else:
        os.close(descriptor)
    return {
        "path": f"{group_name}/{filename}",
        "size": writer.position,
        "sha256": writer.digest.hexdigest(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _write_cache_manifest_at(
    root: _DirectoryWitness,
    manifest: dict[str, Any],
) -> str:
    payload = _canonical_json_bytes(manifest)
    descriptor = os.open(
        "manifest.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=root.descriptor,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except BaseException as primary_error:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            primary_error.add_note(f"cache manifest close also failed: {cleanup_error}")
        raise
    else:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _cache_validator(runtime: SimpleNamespace, name: str) -> object:
    validator = getattr(runtime.cache, name, None)
    if not callable(validator):
        raise RuntimeError(f"Stage 2 cache runtime is missing required validator cache.{name}")
    return validator


def _stage_cache(
    runtime: SimpleNamespace,
    parent: _DirectoryWitness,
    *,
    destination_name: str,
    features: object,
    transitions: object,
    identity_fields: dict[str, Any],
) -> _StagedCache:
    validate_features = _cache_validator(runtime, "_validate_features")
    validate_transitions = _cache_validator(runtime, "_validate_transitions")
    validate_identity = _cache_validator(runtime, "_validate_identity_fields")
    feature_rows = validate_features(features)
    transition_rows = validate_transitions(transitions, features=features)
    validated_identity = validate_identity(identity_fields)
    if type(feature_rows) is not int or feature_rows <= 0:
        raise RuntimeError("cache feature validator must return a positive exact row count")
    if type(transition_rows) is not int or transition_rows <= 0:
        raise RuntimeError("cache transition validator must return a positive exact row count")
    if type(validated_identity) is not dict:
        raise RuntimeError("cache identity validator must return a JSON object")

    destination_name = _safe_component(destination_name, name="cache destination")
    root: _DirectoryWitness | None = None
    staging_name: str | None = None
    for _ in range(128):
        candidate = f".{destination_name}.stage-{secrets.token_hex(16)}"
        try:
            os.mkdir(candidate, mode=0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(
                candidate,
                _directory_flags(),
                dir_fd=parent.descriptor,
            )
            root = _DirectoryWitness.from_descriptor(
                parent.path / candidate,
                descriptor,
            )
        except BaseException:
            with contextlib.suppress(OSError):
                os.rmdir(candidate, dir_fd=parent.descriptor)
            raise
        staging_name = candidate
        break
    if root is None or staging_name is None:
        raise RuntimeError("could not allocate a unique cache staging directory")

    staged = _StagedCache(
        parent=parent,
        root=root,
        name=staging_name,
        manifest={},
        manifest_sha256="",
    )
    try:
        os.fchmod(root.descriptor, 0o700)
        files: list[dict[str, Any]] = []
        for group_name, table in (
            ("features", features),
            ("transitions", transitions),
        ):
            with _open_child_directory(root, group_name, create=True) as group:
                os.fchmod(group.descriptor, 0o700)
                if not dataclasses.is_dataclass(table) or isinstance(table, type):
                    raise TypeError(f"{group_name} must be a concrete dataclass table")
                files.extend(
                    [
                        _write_cache_array_at(
                            group,
                            group_name=group_name,
                            field_name=field.name,
                            value=getattr(table, field.name),
                        )
                        for field in dataclasses.fields(table)
                    ]
                )
                os.fsync(group.descriptor)
        staged.manifest = {
            "schema_version": 1,
            **validated_identity,
            "feature_rows": feature_rows,
            "transition_rows": transition_rows,
            "files": sorted(files, key=lambda item: item["path"]),
        }
        staged.manifest_sha256 = _write_cache_manifest_at(
            root,
            staged.manifest,
        )
        os.fsync(root.descriptor)
        return staged
    except BaseException as primary_error:
        try:
            staged.close()
        except BaseException as cleanup_error:
            primary_error.add_note(f"cache staging cleanup also failed: {cleanup_error}")
        raise


def _before_bound_cache_publish_hook(destination: Path) -> None:
    del destination


def _after_bound_cache_publish_hook(destination: Path) -> None:
    del destination


def _renameat2_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace cache publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _publish_staged_cache(
    staged: _StagedCache,
    *,
    destination: Path,
    destination_name: str,
    staging_witness: _CacheRootWitness,
) -> None:
    publish_error: BaseException | None = None
    entered_hook = False
    try:
        _before_bound_cache_publish_hook(destination)
        entered_hook = True
        staging_witness.verify()
        _renameat2_noreplace_at(
            staged.parent.descriptor,
            staged.name,
            destination_name,
        )
        staged.published = True
    except BaseException as exc:
        publish_error = exc
    finally:
        if entered_hook:
            try:
                _after_bound_cache_publish_hook(destination)
            except BaseException as exc:
                if publish_error is None:
                    publish_error = exc
                else:
                    publish_error.add_note(f"cache publish hook cleanup also failed: {exc}")
    if publish_error is not None:
        raise publish_error
    os.fsync(staged.parent.descriptor)


def _entry_exists_at(parent: _DirectoryWitness, name: str) -> bool:
    try:
        os.stat(
            name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return True


def _verify_opened_shard(
    opened: object,
    *,
    destination: Path,
    expected_manifest: dict[str, Any],
    expected_manifest_sha256: str,
) -> None:
    opened_root = Path(getattr(opened, "root", ""))
    if opened_root != destination:
        raise RuntimeError(f"reread cache root {opened_root} does not match destination {destination}")
    opened_manifest = getattr(opened, "manifest", None)
    if type(opened_manifest) is not dict:
        raise RuntimeError("reread cache manifest must be a JSON object")
    if opened_manifest != expected_manifest:
        raise RuntimeError("reread cache manifest content does not exactly match the staged expected manifest")
    opened_manifest_sha256 = _require_sha256(
        getattr(opened, "manifest_sha256", None),
        name="reread cache manifest_sha256",
    )
    if opened_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("reread cache manifest SHA-256 does not exactly match the staged expected manifest")


def _open_cache_payload_witness(
    parent: _DirectoryWitness,
    name: str,
    *,
    label: str,
) -> _CacheRootWitness:
    name = _safe_component(name, name="cache destination")
    parent_witness: _DirectoryWitness | None = None
    root_witness: _DirectoryWitness | None = None
    event_descriptor: int | None = None
    payload: _PinnedSnapshotTree | None = None
    try:
        parent.verify()
        parent_witness = _DirectoryWitness.from_descriptor(
            parent.path,
            os.dup(parent.descriptor),
        )
        parent_witness.seal()
        event_descriptor = _watch_directory_changes(Path("/proc/self/fd") / str(parent_witness.descriptor))
        root_witness = _open_child_directory(
            parent_witness,
            name,
            create=False,
        )
        root_witness.seal()
        payload = _PinnedSnapshotTree.open(
            root_witness.descriptor,
            label=label,
            hash_files=False,
        )
        witness = _CacheRootWitness(
            parent=parent_witness,
            root=root_witness,
            event_descriptor=event_descriptor,
            payload=payload,
        )
        witness.verify()
        return witness
    except BaseException as primary_error:
        errors: list[tuple[str, BaseException]] = []
        for cleanup_label, close in (
            ("payload guard", None if payload is None else payload.close),
            ("cache root", None if root_witness is None else root_witness.close),
            (
                "cache parent event descriptor",
                None if event_descriptor is None else lambda: os.close(event_descriptor),
            ),
            ("cache parent", None if parent_witness is None else parent_witness.close),
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append((cleanup_label, exc))
        _finish_cleanup(
            primary_error,
            errors,
            context=f"{label} witness construction",
        )
        raise


def _close_opened_cache_shard(opened: object) -> None:
    primary_error = sys.exception()
    errors: list[tuple[str, BaseException]] = []
    seen: set[int] = set()
    for table_name in ("features", "transitions"):
        table = getattr(opened, table_name, None)
        if not dataclasses.is_dataclass(table) or isinstance(table, type):
            continue
        for field in dataclasses.fields(table):
            value = getattr(table, field.name)
            memory_map = getattr(value, "_mmap", None)
            if memory_map is None or id(memory_map) in seen:
                continue
            seen.add(id(memory_map))
            try:
                memory_map.close()
            except BaseException as exc:
                errors.append((f"{table_name}.{field.name} memory map", exc))
    _finish_cleanup(
        primary_error,
        errors,
        context="opened cache shard",
    )


def _close_cache_validation_resources(
    opened: object | None,
    witness: _CacheRootWitness,
) -> None:
    primary_error = sys.exception()
    errors: list[tuple[str, BaseException]] = []
    if opened is not None:
        try:
            _close_opened_cache_shard(opened)
        except BaseException as exc:
            errors.append(("opened shard", exc))
    try:
        witness.close()
    except BaseException as exc:
        errors.append(("payload witness", exc))
    _finish_cleanup(
        primary_error,
        errors,
        context="cache validation",
    )


def _before_staged_cache_validation_hook(staging: Path) -> None:
    del staging


def _validate_staged_cache(
    runtime: SimpleNamespace,
    staged: _StagedCache,
) -> _CacheRootWitness:
    witness = _open_cache_payload_witness(
        staged.parent,
        staged.name,
        label="cache staging payload",
    )
    opened: object | None = None
    try:
        witness.verify()
        _before_staged_cache_validation_hook(staged.root.path)
        opened = runtime.cache.open_shard(staged.root.path)
        witness.verify()
        _verify_opened_shard(
            opened,
            destination=staged.root.path,
            expected_manifest=staged.manifest,
            expected_manifest_sha256=staged.manifest_sha256,
        )
        witness.verify()
    except BaseException:
        _close_cache_validation_resources(opened, witness)
        raise
    try:
        _close_opened_cache_shard(opened)
        witness.verify()
    except BaseException:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        try:
            witness.close()
        except BaseException as exc:
            errors.append(("payload witness", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="staged cache validation",
        )
        raise
    return witness


def _open_verified_cache(
    runtime: SimpleNamespace,
    destination: Path,
    *,
    expected_manifest: dict[str, Any],
    expected_manifest_sha256: str,
    feature_parent: _DirectoryWitness | None = None,
) -> tuple[object, _CacheRootWitness]:
    owned_parent: _DirectoryWitness | None = None
    if feature_parent is None:
        owned_parent = _open_real_directory_chain(destination.parent, create=False)
        feature_parent = owned_parent
    elif feature_parent.path != destination.parent:
        raise RuntimeError(
            f"cache parent witness {feature_parent.path} does not match destination {destination.parent}"
        )
    witness: _CacheRootWitness | None = None
    try:
        witness = _open_cache_payload_witness(
            feature_parent,
            destination.name,
            label="final cache payload",
        )
        if owned_parent is not None:
            owned_parent.close()
            owned_parent = None
    except BaseException:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        if witness is not None:
            try:
                witness.close()
            except BaseException as exc:
                errors.append(("constructed cache payload witness", exc))
        if owned_parent is not None:
            try:
                owned_parent.close()
            except BaseException as exc:
                errors.append(("owned cache parent", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="verified cache open setup",
        )
        raise
    opened: object | None = None
    try:
        witness.verify()
        opened = runtime.cache.open_shard(destination)
        _after_cache_open_hook(destination)
        witness.verify()
        _verify_opened_shard(
            opened,
            destination=destination,
            expected_manifest=expected_manifest,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        witness.verify()
        return opened, witness
    except BaseException:
        _close_cache_validation_resources(opened, witness)
        raise


def _publish_or_reuse_cache(
    runtime: SimpleNamespace,
    destination: Path,
    *,
    features: object,
    transitions: object,
    identity_fields: dict[str, Any],
    feature_parent: _DirectoryWitness | None = None,
) -> tuple[object, _CacheRootWitness]:
    destination_name = _safe_component(destination.name, name="cache destination")
    owned_parent: _DirectoryWitness | None = None
    if feature_parent is None:
        owned_parent = _open_real_directory_chain(destination.parent, create=True)
        feature_parent = owned_parent
    elif feature_parent.path != destination.parent:
        raise RuntimeError(
            f"cache parent witness {feature_parent.path} does not match destination {destination.parent}"
        )
    staged: _StagedCache | None = None
    staging_witness: _CacheRootWitness | None = None
    final_resources: tuple[object, _CacheRootWitness] | None = None
    try:
        feature_parent.verify()
        staged = _stage_cache(
            runtime,
            feature_parent,
            destination_name=destination_name,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )
        staging_witness = _validate_staged_cache(runtime, staged)
        publish_error: BaseException | None = None
        parent_sync_succeeded = False
        try:
            _publish_staged_cache(
                staged,
                destination=destination,
                destination_name=destination_name,
                staging_witness=staging_witness,
            )
            parent_sync_succeeded = True
        except BaseException as exc:
            publish_error = exc
        staging_witness_cleanup_error: BaseException | None = None
        try:
            staging_witness.close()
        except BaseException as exc:
            staging_witness_cleanup_error = exc
        finally:
            staging_witness = None

        expected_manifest = staged.manifest
        expected_manifest_sha256 = staged.manifest_sha256
        renamed_by_us = staged.published
        staged_cleanup_error: BaseException | None = None
        try:
            staged.close()
        except BaseException as exc:
            staged_cleanup_error = exc
        finally:
            staged = None

        cleanup_errors = []
        if staging_witness_cleanup_error is not None:
            cleanup_errors.append(("staging payload witness", staging_witness_cleanup_error))
        if staged_cleanup_error is not None:
            cleanup_errors.append(("staged cache", staged_cleanup_error))

        if publish_error is not None:
            if not _entry_exists_at(feature_parent, destination_name):
                _finish_cleanup(
                    publish_error,
                    cleanup_errors,
                    context="cache publication",
                )
                raise publish_error
            if renamed_by_us and not parent_sync_succeeded:
                try:
                    os.fsync(feature_parent.descriptor)
                except BaseException as retry_error:
                    publish_error.add_note(f"cache parent fsync retry also failed: {retry_error}")
                    _finish_cleanup(
                        publish_error,
                        cleanup_errors,
                        context="cache publication",
                    )
                    raise publish_error from publish_error.__cause__
        if cleanup_errors:
            if publish_error is not None:
                _finish_cleanup(
                    publish_error,
                    cleanup_errors,
                    context="cache publication",
                )
                raise publish_error
            _finish_cleanup(
                None,
                cleanup_errors,
                context="cache publication",
            )
        try:
            final_resources = _open_verified_cache(
                runtime,
                destination,
                expected_manifest=expected_manifest,
                expected_manifest_sha256=expected_manifest_sha256,
                feature_parent=feature_parent,
            )
        except BaseException as reuse_error:
            if publish_error is None:
                raise
            if isinstance(publish_error, FileExistsError):
                reuse_error.add_note(
                    "atomic cache publication observed an existing destination; "
                    f"the existing shard was left untouched: {publish_error}"
                )
                raise
            publish_error.add_note(
                "cache destination appeared after publication failed, but exact "
                f"content verify-and-reuse also failed: {reuse_error}"
            )
            raise publish_error from publish_error.__cause__
        return final_resources
    finally:
        primary_error = sys.exception()
        errors: list[tuple[str, BaseException]] = []
        for label, close in (
            (
                "staging payload witness",
                None if staging_witness is None else staging_witness.close,
            ),
            ("staged cache", None if staged is None else staged.close),
            ("owned cache parent", None if owned_parent is None else owned_parent.close),
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append((label, exc))
        if final_resources is not None and (primary_error is not None or errors):
            opened, final_witness = final_resources
            try:
                _close_opened_cache_shard(opened)
            except BaseException as exc:
                errors.append(("final opened cache shard", exc))
            try:
                final_witness.close()
            except BaseException as exc:
                errors.append(("final cache witness", exc))
        _finish_cleanup(
            primary_error,
            errors,
            context="cache publish/reuse",
        )


def _cleanup_note(label: str, error: BaseException) -> str:
    return f"Stage 2 cache builder cleanup failed during {label}: {error}"


def _clear_exception_frames(error: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if current.__traceback__ is not None:
            with contextlib.suppress(RuntimeError):
                traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(child for child in nested if isinstance(child, BaseException))


def _run_pipeline(
    config: BuildCacheConfig,
    *,
    runtime: SimpleNamespace,
    code_commit: str,
) -> _PipelineOutcome:
    batch = runtime.admission.validate_ready_batch(
        config.batch,
        video_validator=runtime.admission.validate_video_with_ffprobe,
    )
    with _prepare_output_roots(config) as roots, contextlib.ExitStack() as resources:
        _publish_or_verify_round(
            config,
            batch,
            code_commit=code_commit,
            runtime=runtime,
            admissions_witness=roots.admissions,
        )
        plan = runtime.transitions.build_transition_plan(batch)
        loaded = load_model_and_transforms(
            config,
            batch,
            runtime=runtime,
            code_commit=code_commit,
            training_root_witness=roots.training_root,
        )
        feature_parent = resources.enter_context(
            _open_child_directory(
                roots.feature_cache,
                loaded.feature_id,
                create=True,
            )
        )
        normalizer = runtime.transitions.Stage2Normalizer.from_norm_stats(loaded.norm_stats)
        raw = runtime.transitions.build_raw_transition_table(
            batch,
            plan,
            normalizer,
        )

        dataset = runtime.feature_extractor.Stage2ObservationDataset(
            batch,
            plan.feature_keys,
            loaded.input_transform,
        )
        features = runtime.feature_extractor.extract_features_with_frozen_guard(
            model=loaded.model,
            dataset=dataset,
            feature_keys=plan.feature_keys,
            feature_id=loaded.feature_id,
            expected_parameter_sha256=loaded.loaded_parameter_sha256,
            micro_batch_size=config.micro_batch_size,
            num_workers=config.num_workers,
            sampler_num_steps=config.sampler_num_steps,
        )

        transition_table = runtime.cache.finalize_transition_table(
            batch,
            plan,
            raw,
            features,
        )
        identity_fields = _cache_identity_fields(
            config,
            batch,
            loaded,
            code_commit=code_commit,
            default_prompt=runtime.feature_extractor.DEFAULT_PROMPT,
        )
        destination = feature_parent.path / _safe_component(batch.batch_id, name="batch_id")
        publish_commit = current_git_commit()
        if publish_commit != code_commit:
            raise RuntimeError(
                f"Git commit changed while building Stage 2 cache: start {code_commit}, before publish {publish_commit}"
            )
        roots.verify()
        feature_parent.verify()
        opened, final_witness = _publish_or_reuse_cache(
            runtime,
            destination,
            features=features,
            transitions=transition_table,
            identity_fields=identity_fields,
            feature_parent=feature_parent,
        )
        try:
            manifest = getattr(opened, "manifest", None)
            if type(manifest) is not dict:
                raise RuntimeError("reread cache manifest must be a JSON object")
            manifest_sha256 = _require_sha256(
                getattr(opened, "manifest_sha256", None),
                name="cache manifest_sha256",
            )
            feature_rows = manifest.get("feature_rows")
            transition_rows = manifest.get("transition_rows")
            if type(feature_rows) is not int or feature_rows <= 0:
                raise RuntimeError("cache manifest feature_rows must be a positive integer")
            if type(transition_rows) is not int or transition_rows <= 0:
                raise RuntimeError("cache manifest transition_rows must be a positive integer")
            result = BuildCacheResult(
                destination=destination,
                manifest_sha256=manifest_sha256,
                feature_rows=feature_rows,
                transition_rows=transition_rows,
                feature_identity=loaded.feature_id,
                batch_id=batch.batch_id,
            )
            roots.verify()
            feature_parent.verify()
            final_witness.verify()
            return _PipelineOutcome(
                result=result,
                final_witness=final_witness,
            )
        except BaseException:
            final_witness.close()
            raise


def run(
    config: BuildCacheConfig,
    *,
    runtime: SimpleNamespace | None = None,
) -> BuildCacheResult:
    config.validate()
    active_runtime: SimpleNamespace | None = runtime
    result: BuildCacheResult | None = None
    final_witness: _CacheRootWitness | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[tuple[str, BaseException]] = []
    try:
        if active_runtime is None:
            active_runtime = _runtime_imports()
        commit = current_git_commit()
        outcome = _run_pipeline(
            config,
            runtime=active_runtime,
            code_commit=commit,
        )
        result = outcome.result
        final_witness = outcome.final_witness
    except BaseException as exc:
        primary_error = exc
        _clear_exception_frames(primary_error)
    finally:
        try:
            gc.collect()
        except BaseException as exc:
            _clear_exception_frames(exc)
            cleanup_errors.append(("gc.collect", exc))
        if active_runtime is not None:
            try:
                active_runtime.jax.clear_caches()
            except BaseException as exc:
                _clear_exception_frames(exc)
                cleanup_errors.append(("jax.clear_caches", exc))
        try:
            gc.collect()
        except BaseException as exc:
            _clear_exception_frames(exc)
            cleanup_errors.append(("post-jax gc.collect", exc))

    if primary_error is not None:
        if final_witness is not None:
            try:
                final_witness.close()
            except BaseException as exc:
                primary_error.add_note(_cleanup_note("final cache witness", exc))
        for label, error in cleanup_errors:
            primary_error.add_note(_cleanup_note(label, error))
        raise primary_error
    if cleanup_errors:
        if final_witness is not None:
            try:
                final_witness.verify()
            finally:
                final_witness.close()
        details = "; ".join(_cleanup_note(label, error) for label, error in cleanup_errors)
        raise RuntimeError(f"Stage 2 cache builder cleanup failed: {details}") from cleanup_errors[0][1]
    if result is None:
        if final_witness is not None:
            final_witness.close()
        raise RuntimeError("Stage 2 cache builder completed without a result")
    if final_witness is None:
        raise RuntimeError("Stage 2 cache builder completed without a final cache witness")
    try:
        final_witness.verify()
    finally:
        final_witness.close()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one immutable RLT Stage 2 frozen-feature cache shard.",
    )
    parser.add_argument("stage2_config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument(
        "--stage1-config",
        default="rl_token_stage1",
    )
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler-num-steps", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    result = run(
        BuildCacheConfig(
            checkpoint=arguments.checkpoint,
            batch=arguments.batch,
            training_root=arguments.training_root,
            round_id=arguments.round_id,
            stage2_config=arguments.stage2_config,
            stage1_config=arguments.stage1_config,
            micro_batch_size=arguments.micro_batch_size,
            num_workers=arguments.num_workers,
            sampler_num_steps=arguments.sampler_num_steps,
        )
    )
    print(
        json.dumps(
            dataclasses.asdict(result),
            default=os.fspath,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
