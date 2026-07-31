"""Build a portable, reindexed LeRobot v2.1 annotation batch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import tempfile
from types import MappingProxyType

import pyarrow as pa
import pyarrow.parquet as pq

from .ledger import ManifestError, _validate_manifest
from .models import SourceEpisode, SourceFile


class BatchBuildError(RuntimeError):
    """A downloaded selection could not be rewritten into a safe batch."""


VIDEO_ROLES = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
REQUIRED_ROLES = ("parquet", *VIDEO_ROLES)
EXPECTED_EPISODES = 20
DEFAULT_TOOL_VERSION = "tristate-labeler-hil-migration/1"

_BATCH_ID_RE = re.compile(r"batch_\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_-]*)?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_HARDLINK_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EACCES,
    errno.EPERM,
    getattr(errno, "ENOTSUP", -1),
    getattr(errno, "EOPNOTSUPP", -1),
    getattr(errno, "ENOSYS", -1),
}


@dataclass(frozen=True)
class DownloadedEpisode:
    """A scanned source episode and its verified local role-to-file mapping."""

    source: SourceEpisode
    files: Mapping[str, Path]

    def __post_init__(self) -> None:
        detached = {key: Path(value) for key, value in self.files.items()}
        object.__setattr__(self, "files", MappingProxyType(detached))


@dataclass(frozen=True)
class BuiltBatch:
    root: Path
    batch_id: str
    episode_count: int
    frame_count: int
    file_count: int
    manifest: dict[str, object]

    @property
    def video_count(self) -> int:
        return int(self.manifest["video_count"])


@dataclass(frozen=True)
class _PathIdentity:
    path: Path
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class _Layout:
    root: Path
    downloads: Path
    base_identities: tuple[_PathIdentity, ...]


@dataclass(frozen=True)
class _VerifiedDownload:
    role: str
    path: Path
    source_file: SourceFile
    identity: _PathIdentity


class _BuildGuard:
    """Track only paths created by this build and bind them to directory identities."""

    def __init__(
        self,
        *,
        layout: _Layout,
        root_identity: _PathIdentity,
        remove_root: bool,
        directory_fds: Mapping[Path, int] | None = None,
    ) -> None:
        self.root = layout.root
        self.downloads = layout.downloads
        self._base_identities = layout.base_identities
        self._root_identity = root_identity
        self._remove_root = remove_root
        self._created_dirs: dict[Path, _PathIdentity] = {}
        self._created_files: dict[Path, _PathIdentity] = {}
        self._directory_fds = dict(directory_fds or {})
        self._anchored = bool(self._directory_fds)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in reversed(tuple(dict.fromkeys(self._directory_fds.values()))):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._directory_fds.clear()

    def __enter__(self) -> _BuildGuard:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _directory_fd(self, path: Path) -> int:
        try:
            return self._directory_fds[path]
        except KeyError as exc:
            raise BatchBuildError("anchored directory handle is unavailable") from exc

    def _capture_at(self, path: Path, *, kind: str, context: str) -> _PathIdentity:
        path = _absolute_lexical(path)
        if not self._anchored:
            return _capture_identity(path, kind=kind, context=context)
        parent_fd = self._directory_fd(path.parent)
        try:
            details = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise BatchBuildError(
                f"could not inspect {context}: {exc.__class__.__name__}"
            ) from exc
        return _identity_from_stat(path, details, kind=kind, context=context)

    def _entry_matches(
        self, path: Path, identity: _PathIdentity, *, kind: str
    ) -> bool:
        try:
            current = self._capture_at(path, kind=kind, context="tracked entry")
        except BatchBuildError:
            return False
        return _same_identity(identity, current)

    def io_path(self, path: Path) -> Path:
        path = _absolute_lexical(path)
        if not self._anchored:
            return path
        return Path(f"/proc/self/fd/{self._directory_fd(path.parent)}/{path.name}")

    def open_for_write(self, path: Path, identity: _PathIdentity) -> int:
        self.assert_file(path, identity)
        flags = os.O_WRONLY | os.O_TRUNC
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            if self._anchored:
                descriptor = os.open(
                    path.name,
                    flags,
                    dir_fd=self._directory_fd(path.parent),
                )
            else:
                descriptor = os.open(path, flags)
        except OSError as exc:
            raise BatchBuildError(
                f"could not open owned file: {exc.__class__.__name__}"
            ) from exc
        current = _identity_from_stat(
            path,
            os.fstat(descriptor),
            kind="file",
            context="opened owned file",
        )
        if not _same_identity(identity, current):
            os.close(descriptor)
            raise BatchBuildError("opened file identity changed")
        try:
            self.assert_current()
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _directory_identities(self) -> tuple[_PathIdentity, ...]:
        return (
            *self._base_identities,
            self._root_identity,
            *self._created_dirs.values(),
        )

    def assert_current(self) -> None:
        if self._closed:
            raise BatchBuildError("build guard is closed")
        _assert_identities(self._directory_identities())
        if self._anchored:
            for path, descriptor in self._directory_fds.items():
                expected = self._known_directory(path)
                if expected is None:
                    raise BatchBuildError("anchored directory is not tracked")
                current = _identity_from_stat(
                    path,
                    os.fstat(descriptor),
                    kind="directory",
                    context="anchored directory",
                )
                if not _same_identity(expected, current):
                    raise BatchBuildError("anchored directory identity changed")

    def _known_directory(self, path: Path) -> _PathIdentity | None:
        for identity in self._directory_identities():
            if identity.path == path:
                return identity
        return None

    def ensure_directory(self, path: Path) -> _PathIdentity:
        path = _absolute_lexical(path)
        if path == self.root:
            self.assert_current()
            return self._root_identity
        if not path.is_relative_to(self.root):
            raise BatchBuildError("build directory escaped the dataset root")
        known = self._known_directory(path)
        if known is not None:
            self.assert_current()
            return known

        parent_identity = self.ensure_directory(path.parent)
        self.assert_current()
        if self._anchored:
            parent_fd = self._directory_fd(path.parent)
            try:
                os.mkdir(path.name, mode=0o755, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise BatchBuildError(
                    "unexpected file or directory appeared in the clean dataset root"
                ) from exc
            except OSError as exc:
                raise BatchBuildError(
                    f"could not create anchored dataset directory: {exc.__class__.__name__}"
                ) from exc

            created_identity: _PathIdentity | None = None
            try:
                created_identity = _directory_identity_at(
                    path,
                    parent_fd,
                    context="created anchored directory",
                )
                descriptor, identity = _open_directory_at(
                    path,
                    parent_fd,
                    expected=created_identity,
                    context="created anchored directory",
                )
            except Exception:
                if created_identity is not None:
                    _remove_directory_at_if_identity(
                        path,
                        parent_fd,
                        created_identity,
                    )
                raise
            self._created_dirs[path] = identity
            self._directory_fds[path] = descriptor
            self.assert_current()
            return identity
        try:
            path.mkdir()
        except FileExistsError as exc:
            raise BatchBuildError(
                "unexpected file or directory appeared in the clean dataset root"
            ) from exc
        except OSError as exc:
            raise BatchBuildError(
                f"could not create dataset directory: {exc.__class__.__name__}"
            ) from exc
        identity = _capture_identity(path, kind="directory", context="created directory")
        _assert_identity(parent_identity)
        self.assert_current()
        self._created_dirs[path] = identity
        return identity

    def prepare_file(self, path: Path) -> None:
        path = _absolute_lexical(path)
        if path == self.root or not path.is_relative_to(self.root):
            raise BatchBuildError("build file escaped the dataset root")
        self.ensure_directory(path.parent)
        self.assert_current()
        try:
            if self._anchored:
                os.stat(
                    path.name,
                    dir_fd=self._directory_fd(path.parent),
                    follow_symlinks=False,
                )
            else:
                path.lstat()
        except FileNotFoundError:
            self.assert_current()
            return
        except OSError as exc:
            raise BatchBuildError(
                f"could not inspect build target: {exc.__class__.__name__}"
            ) from exc
        raise BatchBuildError("build target already exists in the clean dataset root")

    def record_file(self, path: Path) -> _PathIdentity:
        path = _absolute_lexical(path)
        identity = self._capture_at(path, kind="file", context="created file")
        self._created_files[path] = identity
        self.assert_current()
        return identity

    def assert_file(self, path: Path, identity: _PathIdentity | None = None) -> _PathIdentity:
        path = _absolute_lexical(path)
        expected = self._created_files.get(path) if identity is None else identity
        if expected is None:
            raise BatchBuildError("build file was not created by this run")
        current = self._capture_at(path, kind="file", context="owned file")
        if not _same_identity(expected, current):
            raise BatchBuildError("owned file identity changed")
        self.assert_current()
        return expected

    def make_temp(self, target: Path) -> tuple[Path, _PathIdentity]:
        target = _absolute_lexical(target)
        self.ensure_directory(target.parent)
        self.assert_current()
        if self._anchored:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            parent_fd = self._directory_fd(target.parent)
            descriptor: int | None = None
            temporary: Path | None = None
            for _ in range(100):
                candidate = f".{target.name}.{secrets.token_hex(8)}.tmp"
                try:
                    descriptor = os.open(
                        candidate,
                        flags,
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError:
                    continue
                except OSError as exc:
                    raise BatchBuildError(
                        f"could not create anchored temporary file: {exc.__class__.__name__}"
                    ) from exc
                temporary = target.parent / candidate
                break
            if descriptor is None or temporary is None:
                raise BatchBuildError("could not allocate a unique temporary file")
            try:
                identity = _identity_from_stat(
                    temporary,
                    os.fstat(descriptor),
                    kind="file",
                    context="anchored temporary file",
                )
            finally:
                os.close(descriptor)
            self._created_files[temporary] = identity
            self.assert_current()
            return temporary, identity
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
        except OSError as exc:
            raise BatchBuildError(
                f"could not create unique temporary file: {exc.__class__.__name__}"
            ) from exc
        os.close(descriptor)
        temporary = Path(raw_path)
        identity = _capture_identity(
            temporary, kind="file", context="unique temporary file"
        )
        self._created_files[temporary] = identity
        self.assert_current()
        return temporary, identity

    def install_temp(
        self,
        temporary: Path,
        target: Path,
        identity: _PathIdentity,
    ) -> _PathIdentity:
        self.assert_file(temporary, identity)
        self.prepare_file(target)
        try:
            if self._anchored:
                os.link(
                    temporary.name,
                    target.name,
                    src_dir_fd=self._directory_fd(temporary.parent),
                    dst_dir_fd=self._directory_fd(target.parent),
                    follow_symlinks=False,
                )
            else:
                os.link(temporary, target)
        except OSError as exc:
            raise BatchBuildError(
                f"could not install temporary file without clobbering: {exc.__class__.__name__}"
            ) from exc
        installed = self._capture_at(target, kind="file", context="installed file")
        if not _same_identity(identity, installed):
            raise BatchBuildError("installed file identity changed")
        self._created_files[target] = installed
        try:
            if self._anchored:
                os.unlink(
                    temporary.name,
                    dir_fd=self._directory_fd(temporary.parent),
                )
            else:
                temporary.unlink()
        except OSError as exc:
            raise BatchBuildError(
                f"could not remove installed temporary link: {exc.__class__.__name__}"
            ) from exc
        else:
            self._created_files.pop(temporary, None)
        self.assert_current()
        return installed

    def create_exclusive_file(self, target: Path) -> tuple[int, _PathIdentity]:
        self.prepare_file(target)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            if self._anchored:
                descriptor = os.open(
                    target.name,
                    flags,
                    0o600,
                    dir_fd=self._directory_fd(target.parent),
                )
            else:
                descriptor = os.open(target, flags, 0o600)
        except OSError as exc:
            raise BatchBuildError(
                f"could not create exclusive output file: {exc.__class__.__name__}"
            ) from exc
        try:
            identity = _identity_from_stat(
                target,
                os.fstat(descriptor),
                kind="file",
                context="exclusive output file",
            )
            self._created_files[target] = identity
            self.assert_current()
            return descriptor, identity
        except Exception:
            os.close(descriptor)
            raise

    def link_external_file(self, source: Path, target: Path) -> None:
        if self._anchored:
            os.link(
                source,
                target.name,
                dst_dir_fd=self._directory_fd(target.parent),
                follow_symlinks=False,
            )
        else:
            os.link(source, target)

    def discard_file(self, path: Path, identity: _PathIdentity) -> None:
        if self._anchored:
            if not self._entry_matches(path, identity, kind="file"):
                return
            try:
                os.unlink(path.name, dir_fd=self._directory_fd(path.parent))
            except OSError:
                return
            self._created_files.pop(path, None)
            return
        if not _identities_current(self._directory_identities()):
            return
        parent_identity = self._known_directory(path.parent)
        if parent_identity is None or not _identities_current((parent_identity,)):
            return
        if not _identity_at_path(path, identity, kind="file"):
            return
        try:
            path.unlink()
        except OSError:
            return
        if _identities_current((parent_identity,)):
            self._created_files.pop(path, None)

    def assert_owned_tree(self, expected_files: Sequence[Path]) -> None:
        self.assert_current()
        expected = {_absolute_lexical(path) for path in expected_files}
        if set(self._created_files) != expected:
            raise BatchBuildError("built dataset owned file set is incomplete")

        seen_files: set[Path] = set()
        seen_dirs: set[Path] = set()
        pending = [self.root]
        while pending:
            directory = pending.pop()
            self.assert_current()
            try:
                entries = tuple(os.scandir(directory))
            except OSError as exc:
                raise BatchBuildError(
                    f"could not inspect built dataset tree: {exc.__class__.__name__}"
                ) from exc
            for entry in entries:
                path = _absolute_lexical(Path(entry.path))
                try:
                    if entry.is_symlink():
                        raise BatchBuildError("built dataset contains a foreign symlink")
                    if entry.is_dir(follow_symlinks=False):
                        identity = self._created_dirs.get(path)
                        if identity is None or not _identity_at_path(
                            path, identity, kind="directory"
                        ):
                            raise BatchBuildError(
                                "built dataset contains a foreign directory"
                            )
                        seen_dirs.add(path)
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        identity = self._created_files.get(path)
                        if identity is None or not _identity_at_path(
                            path, identity, kind="file"
                        ):
                            raise BatchBuildError("built dataset contains a foreign file")
                        seen_files.add(path)
                    else:
                        raise BatchBuildError("built dataset contains a non-regular entry")
                except OSError as exc:
                    raise BatchBuildError(
                        f"could not inspect built dataset entry: {exc.__class__.__name__}"
                    ) from exc
        self.assert_current()
        if seen_files != expected or seen_dirs != set(self._created_dirs):
            raise BatchBuildError("built dataset tree changed during inspection")

    def cleanup(self) -> None:
        if self._anchored:
            self._cleanup_anchored()
            return
        # Identity mismatch means ownership is no longer provable. Stop without
        # touching the replacement path or any of its possible referents.
        if not _identities_current(self._directory_identities()):
            return

        for path, identity in sorted(
            tuple(self._created_files.items()),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if not _identities_current(self._directory_identities()):
                return
            parent_identity = self._known_directory(path.parent)
            if parent_identity is None or not _identities_current((parent_identity,)):
                return
            if _identity_at_path(path, identity, kind="file"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
                if not _identities_current((parent_identity,)):
                    return
            self._created_files.pop(path, None)

        for path, identity in sorted(
            tuple(self._created_dirs.items()),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if not _identities_current(self._directory_identities()):
                return
            parent_identity = self._known_directory(path.parent)
            if parent_identity is None or not _identities_current((parent_identity,)):
                return
            if not _identity_at_path(path, identity, kind="directory"):
                return
            try:
                path.rmdir()
            except FileNotFoundError:
                self._created_dirs.pop(path, None)
            except OSError as exc:
                if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST, errno.EACCES}:
                    return
            else:
                self._created_dirs.pop(path, None)

        if self._remove_root:
            remaining = (*self._base_identities, self._root_identity)
            if not _identities_current(remaining):
                return
            try:
                self.root.rmdir()
            except OSError:
                pass

    def _cleanup_anchored(self) -> None:
        for path, identity in sorted(
            tuple(self._created_files.items()),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if not self._entry_matches(path, identity, kind="file"):
                continue
            try:
                os.unlink(path.name, dir_fd=self._directory_fd(path.parent))
            except OSError:
                continue
            self._created_files.pop(path, None)

        for path, identity in sorted(
            tuple(self._created_dirs.items()),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if not self._entry_matches(path, identity, kind="directory"):
                continue
            try:
                os.rmdir(path.name, dir_fd=self._directory_fd(path.parent))
            except OSError:
                continue
            descriptor = self._directory_fds.pop(path, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            self._created_dirs.pop(path, None)

        if self._remove_root and self._entry_matches(
            self.root, self._root_identity, kind="directory"
        ):
            try:
                os.rmdir(
                    self.root.name,
                    dir_fd=self._directory_fd(self.root.parent),
                )
            except OSError:
                pass


_DIRECTORY_FD_SUPPORTED = (
    os.name == "posix"
    and Path("/proc/self/fd").is_dir()
    and all(
        function in os.supports_dir_fd
        for function in (os.open, os.mkdir, os.stat, os.unlink, os.rmdir, os.link)
    )
)


def _directory_fd_supported() -> bool:
    # Capability is a property of the native runtime, not of wrappers installed
    # later for tracing, fault injection, or tests.
    return _DIRECTORY_FD_SUPPORTED


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _directory_identity_at(
    path: Path,
    parent_fd: int,
    *,
    context: str,
) -> _PathIdentity:
    try:
        details = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise BatchBuildError(
            f"could not inspect {context}: {exc.__class__.__name__}"
        ) from exc
    return _identity_from_stat(
        path,
        details,
        kind="directory",
        context=context,
    )


def _open_directory_at(
    path: Path,
    parent_fd: int,
    *,
    expected: _PathIdentity | None = None,
    context: str,
) -> tuple[int, _PathIdentity]:
    observed = _directory_identity_at(path, parent_fd, context=context)
    if expected is not None and not _same_identity(expected, observed):
        raise BatchBuildError(f"{context} identity changed before opening")

    descriptor: int | None = None
    keep_open = False
    try:
        descriptor = os.open(
            path.name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        opened = _identity_from_stat(
            path,
            os.fstat(descriptor),
            kind="directory",
            context=context,
        )
        if not _same_identity(observed, opened):
            raise BatchBuildError(f"{context} identity changed while opening")
        keep_open = True
        return descriptor, opened
    except OSError as exc:
        raise BatchBuildError(
            f"could not open {context}: {exc.__class__.__name__}"
        ) from exc
    except Exception:
        raise
    finally:
        if descriptor is not None and not keep_open:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_directory_chain(path: Path) -> int:
    path = _absolute_lexical(path)
    anchor = Path(path.anchor)
    try:
        descriptor = os.open(anchor, _directory_open_flags())
    except OSError as exc:
        raise BatchBuildError(
            f"could not open directory-chain anchor: {exc.__class__.__name__}"
        ) from exc

    current = anchor
    try:
        _identity_from_stat(
            anchor,
            os.fstat(descriptor),
            kind="directory",
            context="directory-chain anchor",
        )
        for component in path.parts[1:]:
            child = current / component
            child_fd, _ = _open_directory_at(
                child,
                descriptor,
                context="anchored directory-chain component",
            )
            previous_fd = descriptor
            descriptor = child_fd
            try:
                os.close(previous_fd)
            except OSError:
                pass
            current = child
        return descriptor
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _remove_directory_at_if_identity(
    path: Path,
    parent_fd: int,
    identity: _PathIdentity,
) -> None:
    try:
        current = _directory_identity_at(
            path,
            parent_fd,
            context="failed created directory",
        )
        if _same_identity(identity, current):
            os.rmdir(path.name, dir_fd=parent_fd)
    except (BatchBuildError, OSError):
        pass


def _absolute_lexical(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise BatchBuildError(
            f"could not normalize local path: {exc.__class__.__name__}"
        ) from exc


def _lstat(path: Path, context: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise BatchBuildError(
            f"could not inspect {context}: {exc.__class__.__name__}"
        ) from exc


def _capture_identity(path: Path, *, kind: str, context: str) -> _PathIdentity:
    path = _absolute_lexical(path)
    details = _lstat(path, context)
    return _identity_from_stat(path, details, kind=kind, context=context)


def _identity_from_stat(
    path: Path,
    details: os.stat_result,
    *,
    kind: str,
    context: str,
) -> _PathIdentity:
    if stat.S_ISLNK(details.st_mode):
        raise BatchBuildError(f"{context} must not be a symlink")
    if kind == "directory" and not stat.S_ISDIR(details.st_mode):
        raise BatchBuildError(f"{context} must be a directory")
    if kind == "file" and not stat.S_ISREG(details.st_mode):
        raise BatchBuildError(f"{context} must be a regular file")
    return _PathIdentity(
        path=_absolute_lexical(path),
        device=int(details.st_dev),
        inode=int(details.st_ino),
        mode=stat.S_IFMT(details.st_mode),
    )


def _same_identity(left: _PathIdentity, right: _PathIdentity) -> bool:
    return (
        left.device == right.device
        and left.inode == right.inode
        and left.mode == right.mode
    )


def _identity_at_path(path: Path, identity: _PathIdentity, *, kind: str) -> bool:
    try:
        current = _capture_identity(path, kind=kind, context="tracked path")
    except BatchBuildError:
        return False
    return _same_identity(identity, current)


def _assert_identity(identity: _PathIdentity) -> None:
    kind = "directory" if identity.mode == stat.S_IFDIR else "file"
    try:
        current = _capture_identity(
            identity.path, kind=kind, context="tracked path identity"
        )
    except BatchBuildError as exc:
        raise BatchBuildError("tracked path identity changed") from exc
    if not _same_identity(identity, current):
        raise BatchBuildError("tracked path identity changed")


def _assert_identities(identities: Sequence[_PathIdentity]) -> None:
    for identity in identities:
        _assert_identity(identity)


def _identities_current(identities: Sequence[_PathIdentity]) -> bool:
    try:
        _assert_identities(identities)
    except BatchBuildError:
        return False
    return True


def _validate_layout(dataset_root: Path, batch_id: str) -> _Layout:
    if _BATCH_ID_RE.fullmatch(batch_id) is None:
        raise BatchBuildError(
            "batch_id must start with batch_ followed by a six-digit sequence"
        )
    root = _absolute_lexical(dataset_root)
    if (
        root.name != "dataset"
        or root.parent.name != batch_id
        or root.parent.parent.name != ".staging"
    ):
        raise BatchBuildError(
            "dataset must use .staging/<batch_id>/dataset layout"
        )

    staging = root.parent.parent
    batch_root = root.parent
    downloads = batch_root / "downloads"
    identities = (
        _capture_identity(
            staging, kind="directory", context="staging directory"
        ),
        _capture_identity(
            batch_root, kind="directory", context="batch staging directory"
        ),
        _capture_identity(
            downloads, kind="directory", context="downloads directory"
        ),
    )
    _assert_identities(identities)
    return _Layout(root=root, downloads=downloads, base_identities=identities)


def _validate_created_at(created_at: str) -> None:
    if not isinstance(created_at, str) or created_at.strip() != created_at or "T" not in created_at:
        raise BatchBuildError("created_at must be an ISO timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BatchBuildError("created_at must be an ISO timestamp with timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BatchBuildError("created_at must be an ISO timestamp with timezone")


def _plain_json(value: object, path: str = "metadata") -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BatchBuildError(f"{path} JSON object keys must be strings")
            result[key] = _plain_json(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BatchBuildError(f"{path} JSON numbers must be finite")
        return value
    raise BatchBuildError(f"{path} contains a value that is not JSON-compatible")


def _rewrite_episode_index(value: object, target_index: int) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                target_index
                if key == "episode_index"
                else _rewrite_episode_index(item, target_index)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_rewrite_episode_index(item, target_index) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BatchBuildError(
            f"could not hash downloaded file: {exc.__class__.__name__}"
        ) from exc
    return digest.hexdigest()


def _validate_source_file(source_file: SourceFile, role: str) -> None:
    if source_file.role != role:
        raise BatchBuildError("source file role does not match downloaded role")
    if (
        isinstance(source_file.size, bool)
        or not isinstance(source_file.size, int)
        or source_file.size < 0
    ):
        raise BatchBuildError("source file size must be a nonnegative integer")
    if (
        isinstance(source_file.mtime_ns, bool)
        or not isinstance(source_file.mtime_ns, int)
        or source_file.mtime_ns < 0
    ):
        raise BatchBuildError("source file mtime_ns must be a nonnegative integer")
    if _SHA256_RE.fullmatch(source_file.sha256) is None:
        raise BatchBuildError("source file SHA-256 must be a lowercase 64-hex digest")
    absolute = PurePosixPath(source_file.absolute_path)
    relative = PurePosixPath(source_file.relative_path)
    if (
        "\x00" in source_file.absolute_path
        or "\\" in source_file.absolute_path
        or not absolute.is_absolute()
        or ".." in absolute.parts
        or absolute.as_posix() != source_file.absolute_path
    ):
        raise BatchBuildError("source file absolute path must be normalized POSIX")
    if (
        "\x00" in source_file.relative_path
        or "\\" in source_file.relative_path
        or relative.is_absolute()
        or relative.as_posix() in {"", "."}
        or ".." in relative.parts
        or relative.as_posix() != source_file.relative_path
    ):
        raise BatchBuildError("source file relative path must be normalized POSIX")


def _validate_path_chain(downloads: Path, path: Path) -> _PathIdentity:
    if path == downloads or not path.is_relative_to(downloads):
        raise BatchBuildError("downloaded file must be inside sibling downloads directory")
    current = downloads
    for part in path.relative_to(downloads).parts:
        current = current / part
        details = _lstat(current, "downloaded file")
        if stat.S_ISLNK(details.st_mode):
            raise BatchBuildError("downloaded file path must not contain a symlink")
    if not stat.S_ISREG(details.st_mode):
        raise BatchBuildError("downloaded input must be a regular file")
    return _capture_identity(path, kind="file", context="downloaded input")


def _verify_download_snapshot(
    snapshot: _VerifiedDownload,
    *,
    check_directories: Callable[[], None],
    downloads: Path,
) -> None:
    check_directories()
    before = _validate_path_chain(downloads, snapshot.path)
    if not _same_identity(snapshot.identity, before):
        raise BatchBuildError("downloaded file identity changed from verified snapshot")
    details = _lstat(snapshot.path, "downloaded snapshot")
    if details.st_size != snapshot.source_file.size:
        raise BatchBuildError("downloaded file size changed from verified snapshot")
    digest = _sha256_file(snapshot.path)
    after = _validate_path_chain(downloads, snapshot.path)
    check_directories()
    if not _same_identity(snapshot.identity, after):
        raise BatchBuildError("downloaded file identity changed from verified snapshot")
    if digest != snapshot.source_file.sha256:
        raise BatchBuildError("downloaded file SHA-256 changed from verified snapshot")


def _validate_inputs(
    downloaded_episodes: Sequence[DownloadedEpisode], layout: _Layout
) -> tuple[dict[str, _VerifiedDownload], ...]:
    if len(downloaded_episodes) != EXPECTED_EPISODES:
        raise BatchBuildError("batch must contain exactly 20 episodes")

    fingerprints: set[str] = set()
    identities: set[tuple[str, str, int]] = set()
    local_paths: set[Path] = set()
    local_file_ids: set[tuple[int, int]] = set()
    snapshot_maps: list[dict[str, _VerifiedDownload]] = []

    def check_directories() -> None:
        _assert_identities(layout.base_identities)

    for position, downloaded in enumerate(downloaded_episodes):
        if not isinstance(downloaded, DownloadedEpisode):
            raise BatchBuildError(f"episode {position} is not a DownloadedEpisode")
        source = downloaded.source
        if _SHA256_RE.fullmatch(source.fingerprint) is None:
            raise BatchBuildError("episode fingerprint must be a lowercase 64-hex digest")
        if source.fingerprint in fingerprints:
            raise BatchBuildError("batch contains a duplicate episode fingerprint")
        fingerprints.add(source.fingerprint)

        identity = (source.host, source.dataset_root, source.source_index)
        if identity in identities:
            raise BatchBuildError("batch contains a duplicate source identity")
        identities.add(identity)
        if not source.host or not source.dataset_name or not source.task:
            raise BatchBuildError("source host, dataset name, and task must be nonempty")
        dataset_root = PurePosixPath(source.dataset_root)
        if (
            not dataset_root.is_absolute()
            or ".." in dataset_root.parts
            or dataset_root.as_posix() != source.dataset_root
        ):
            raise BatchBuildError("source dataset root must be normalized absolute POSIX")
        if (
            isinstance(source.source_index, bool)
            or not isinstance(source.source_index, int)
            or source.source_index < 0
        ):
            raise BatchBuildError("source episode index must be a nonnegative integer")
        if (
            isinstance(source.length, bool)
            or not isinstance(source.length, int)
            or source.length <= 0
        ):
            raise BatchBuildError("source episode length must be positive")
        if (
            isinstance(source.completed_ns, bool)
            or not isinstance(source.completed_ns, int)
            or source.completed_ns < 0
        ):
            raise BatchBuildError("source completion time must be a nonnegative integer")

        source_map: dict[str, SourceFile] = {}
        for source_file in source.files:
            if source_file.role in source_map:
                raise BatchBuildError("source files contain a duplicate role")
            source_map[source_file.role] = source_file
        if set(source_map) != set(REQUIRED_ROLES):
            raise BatchBuildError("source files must contain exactly the required roles")
        if set(downloaded.files) != set(REQUIRED_ROLES):
            raise BatchBuildError("download mapping must contain exactly the required roles")

        snapshot_map: dict[str, _VerifiedDownload] = {}
        for role in REQUIRED_ROLES:
            source_file = source_map[role]
            _validate_source_file(source_file, role)
            local_path = _absolute_lexical(Path(downloaded.files[role]))
            check_directories()
            identity = _validate_path_chain(layout.downloads, local_path)
            if local_path in local_paths:
                raise BatchBuildError("download role paths must not alias one another")
            local_paths.add(local_path)
            file_id = (identity.device, identity.inode)
            if identity.inode and file_id in local_file_ids:
                raise BatchBuildError("download role files must not alias one another")
            if identity.inode:
                local_file_ids.add(file_id)
            snapshot = _VerifiedDownload(
                role=role,
                path=local_path,
                source_file=source_file,
                identity=identity,
            )
            _verify_download_snapshot(
                snapshot,
                check_directories=check_directories,
                downloads=layout.downloads,
            )
            snapshot_map[role] = snapshot

        for field_name in (
            "info",
            "task_record",
            "episode_record",
            "stats_record",
            "expert_record",
        ):
            _plain_json(getattr(source, field_name), f"episode[{position}].{field_name}")
        snapshot_maps.append(snapshot_map)

    return tuple(snapshot_maps)


def _prepare_target(layout: _Layout) -> _BuildGuard:
    if _directory_fd_supported():
        return _prepare_target_anchored(layout)
    _assert_identities(layout.base_identities)
    root = layout.root
    created = False
    try:
        details = root.lstat()
    except FileNotFoundError:
        try:
            root.mkdir()
        except OSError as exc:
            raise BatchBuildError(
                f"could not create dataset directory: {exc.__class__.__name__}"
            ) from exc
        created = True
    except OSError as exc:
        raise BatchBuildError(
            f"could not inspect dataset directory: {exc.__class__.__name__}"
        ) from exc
    else:
        if stat.S_ISLNK(details.st_mode):
            raise BatchBuildError("dataset target must not be a symlink")
        if not stat.S_ISDIR(details.st_mode):
            raise BatchBuildError("dataset target must be a directory")
    root_identity = _capture_identity(
        root, kind="directory", context="dataset target"
    )
    try:
        _assert_identities(layout.base_identities)
        if any(root.iterdir()):
            raise BatchBuildError("dataset target must be empty")
        _assert_identity(root_identity)
    except Exception as exc:
        if (
            created
            and _identities_current(layout.base_identities)
            and _identity_at_path(root, root_identity, kind="directory")
        ):
            try:
                root.rmdir()
            except OSError:
                pass
        if isinstance(exc, BatchBuildError):
            raise
        raise BatchBuildError(
            f"could not inspect dataset directory: {exc.__class__.__name__}"
        ) from exc
    return _BuildGuard(
        layout=layout,
        root_identity=root_identity,
        remove_root=created,
    )


def _prepare_target_anchored(layout: _Layout) -> _BuildGuard:
    _assert_identities(layout.base_identities)
    directory_fds: dict[Path, int] = {}
    root_identity: _PathIdentity | None = None
    created = False
    try:
        staging_identity, batch_identity, downloads_identity = layout.base_identities
        staging_parent_fd = _open_directory_chain(staging_identity.path.parent)
        try:
            staging_fd, _ = _open_directory_at(
                staging_identity.path,
                staging_parent_fd,
                expected=staging_identity,
                context="staging directory",
            )
        finally:
            try:
                os.close(staging_parent_fd)
            except OSError:
                pass
        directory_fds[staging_identity.path] = staging_fd

        batch_fd, _ = _open_directory_at(
            batch_identity.path,
            staging_fd,
            expected=batch_identity,
            context="batch staging directory",
        )
        directory_fds[batch_identity.path] = batch_fd

        downloads_fd, _ = _open_directory_at(
            downloads_identity.path,
            batch_fd,
            expected=downloads_identity,
            context="downloads directory",
        )
        directory_fds[downloads_identity.path] = downloads_fd

        batch_fd = directory_fds[layout.root.parent]
        try:
            root_identity = _directory_identity_at(
                layout.root,
                batch_fd,
                context="dataset target",
            )
        except FileNotFoundError:
            os.mkdir(layout.root.name, mode=0o755, dir_fd=batch_fd)
            created = True
            root_identity = _directory_identity_at(
                layout.root,
                batch_fd,
                context="dataset target",
            )

        root_fd, opened_root_identity = _open_directory_at(
            layout.root,
            batch_fd,
            expected=root_identity,
            context="dataset target",
        )
        directory_fds[layout.root] = root_fd
        root_identity = opened_root_identity
        if os.listdir(root_fd):
            raise BatchBuildError("dataset target must be empty")
        guard = _BuildGuard(
            layout=layout,
            root_identity=root_identity,
            remove_root=created,
            directory_fds=directory_fds,
        )
        guard.assert_current()
        return guard
    except Exception as exc:
        if created and root_identity is not None:
            batch_fd = directory_fds.get(layout.root.parent)
            if batch_fd is not None:
                _remove_directory_at_if_identity(
                    layout.root,
                    batch_fd,
                    root_identity,
                )
        for descriptor in reversed(tuple(dict.fromkeys(directory_fds.values()))):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(exc, BatchBuildError):
            raise
        raise BatchBuildError(
            f"could not prepare anchored dataset target: {exc.__class__.__name__}"
        ) from exc


def _integer_bounds(data_type: pa.DataType) -> tuple[int, int]:
    bits = data_type.bit_width
    if pa.types.is_signed_integer(data_type):
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _replace_int_column(
    table: pa.Table, name: str, values: Sequence[int]
) -> pa.Table:
    position = table.schema.get_field_index(name)
    if position < 0:
        raise BatchBuildError(f"parquet is missing required column {name}")
    field = table.schema.field(position)
    if not pa.types.is_integer(field.type):
        raise BatchBuildError(f"parquet column {name} must have an integer type")
    minimum, maximum = _integer_bounds(field.type)
    if any(value < minimum or value > maximum for value in values):
        raise BatchBuildError(
            f"parquet column {name} cannot represent rewritten index values"
        )
    try:
        array = pa.array(values, type=field.type)
    except (OverflowError, TypeError, ValueError) as exc:
        raise BatchBuildError(
            f"parquet column {name} cannot represent rewritten index values"
        ) from exc
    return table.set_column(position, field, array)


def rewrite_parquet(
    source: Path,
    target: Path,
    *,
    episode_index: int,
    global_start: int,
    task_index: int,
    expected_rows: int | None = None,
    _guard: _BuildGuard | None = None,
    _verify_source: Callable[[], None] | None = None,
) -> int:
    if _verify_source is not None:
        _verify_source()
    try:
        table = pq.read_table(source)
    except Exception as exc:
        if _verify_source is not None:
            _verify_source()
        raise BatchBuildError(
            f"could not read source parquet: {exc.__class__.__name__}"
        ) from exc
    if _verify_source is not None:
        _verify_source()
    rows = table.num_rows
    if expected_rows is not None and rows != expected_rows:
        raise BatchBuildError(
            f"source parquet row count {rows} does not match metadata length {expected_rows}"
        )
    table = _replace_int_column(table, "episode_index", [episode_index] * rows)
    table = _replace_int_column(table, "frame_index", list(range(rows)))
    table = _replace_int_column(
        table, "index", list(range(global_start, global_start + rows))
    )
    table = _replace_int_column(table, "task_index", [task_index] * rows)

    target = _absolute_lexical(target)
    temporary: Path | None = None
    temporary_identity: _PathIdentity | None = None
    try:
        if _guard is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.lstat()
            except FileNotFoundError:
                pass
            else:
                raise BatchBuildError("rewritten parquet target already exists")
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(raw_path)
            temporary_identity = _capture_identity(
                temporary, kind="file", context="parquet temporary file"
            )
        else:
            _guard.prepare_file(target)
            temporary, temporary_identity = _guard.make_temp(target)

        parquet_destination = (
            temporary if _guard is None else _guard.io_path(temporary)
        )
        pq.write_table(table, parquet_destination)
        if _guard is None:
            _assert_identity(temporary_identity)
            try:
                target.lstat()
            except FileNotFoundError:
                pass
            else:
                raise BatchBuildError("rewritten parquet target already exists")
            try:
                os.link(temporary, target)
            except OSError as exc:
                raise BatchBuildError(
                    f"could not install rewritten parquet without clobbering: {exc.__class__.__name__}"
                ) from exc
            installed_identity = _capture_identity(
                target,
                kind="file",
                context="installed rewritten parquet",
            )
            if not _same_identity(temporary_identity, installed_identity):
                raise BatchBuildError("installed rewritten parquet identity changed")
            try:
                temporary.unlink()
            except OSError as exc:
                if not _identity_at_path(target, installed_identity, kind="file"):
                    raise BatchBuildError(
                        "rewritten parquet target identity changed during finalization"
                    ) from exc
                try:
                    target.unlink()
                except OSError:
                    # The complete no-clobber target is already published. If it
                    # cannot be rolled back and still has the installed identity,
                    # report success instead of leaving a target after an error.
                    if not _identity_at_path(
                        target, installed_identity, kind="file"
                    ):
                        raise BatchBuildError(
                            "rewritten parquet target identity changed during finalization"
                        ) from exc
                    if _identity_at_path(
                        temporary, temporary_identity, kind="file"
                    ):
                        try:
                            temporary.unlink()
                        except OSError:
                            pass
                    temporary = None
                else:
                    raise BatchBuildError(
                        "could not finalize rewritten parquet; publication rolled back"
                    ) from exc
            else:
                temporary = None
        else:
            _guard.assert_file(temporary, temporary_identity)
            _guard.install_temp(temporary, target, temporary_identity)
            temporary = None
    except BatchBuildError:
        if temporary is not None and temporary_identity is not None:
            if _guard is None:
                if _identity_at_path(temporary, temporary_identity, kind="file"):
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
            else:
                _guard.discard_file(temporary, temporary_identity)
        raise
    except Exception as exc:
        if temporary is not None and temporary_identity is not None:
            if _guard is None:
                if _identity_at_path(temporary, temporary_identity, kind="file"):
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
            else:
                _guard.discard_file(temporary, temporary_identity)
        raise BatchBuildError(
            f"could not write rewritten parquet: {exc.__class__.__name__}"
        ) from exc
    return rows


def _verify_built_video(
    target: Path,
    identity: _PathIdentity,
    snapshot: _VerifiedDownload,
    guard: _BuildGuard,
) -> None:
    guard.assert_file(target, identity)
    io_path = guard.io_path(target)
    try:
        details = io_path.stat()
    except OSError as exc:
        raise BatchBuildError(
            f"could not inspect built video: {exc.__class__.__name__}"
        ) from exc
    if details.st_size != snapshot.source_file.size:
        raise BatchBuildError("built video size differs from verified source snapshot")
    if _sha256_file(io_path) != snapshot.source_file.sha256:
        raise BatchBuildError("built video SHA-256 differs from verified source snapshot")
    guard.assert_file(target, identity)


def _copy_file_to_fd(source: Path, descriptor: int) -> None:
    try:
        source_details = source.stat(follow_symlinks=False)
        with source.open("rb") as stream:
            while True:
                chunk = stream.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError(errno.EIO, "short write to owned video file")
                    view = view[written:]
        os.fsync(descriptor)
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            try:
                fchmod(descriptor, stat.S_IMODE(source_details.st_mode))
            except (NotImplementedError, OSError):
                # Windows and some network filesystems cannot apply mode by fd.
                pass
        if os.utime in os.supports_fd:
            try:
                os.utime(
                    descriptor,
                    ns=(source_details.st_atime_ns, source_details.st_mtime_ns),
                )
            except (NotImplementedError, OSError):
                # Never fall back to the logical target path: retaining the held
                # descriptor is more important than timestamp preservation.
                pass
    except OSError as exc:
        raise BatchBuildError(
            f"could not copy downloaded video to owned file: {exc.__class__.__name__}"
        ) from exc


def _install_video(
    snapshot: _VerifiedDownload, target: Path, guard: _BuildGuard
) -> _PathIdentity:
    _verify_download_snapshot(
        snapshot,
        check_directories=guard.assert_current,
        downloads=guard.downloads,
    )
    guard.prepare_file(target)
    try:
        guard.link_external_file(snapshot.path, target)
    except OSError as exc:
        if exc.errno not in _HARDLINK_FALLBACK_ERRNOS:
            raise BatchBuildError(
                f"could not hardlink downloaded video: {exc.__class__.__name__}"
            ) from exc
        try:
            descriptor, identity = guard.create_exclusive_file(target)
            try:
                _copy_file_to_fd(snapshot.path, descriptor)
            finally:
                os.close(descriptor)
        except OSError as copy_exc:
            raise BatchBuildError(
                f"could not copy downloaded video: {copy_exc.__class__.__name__}"
            ) from copy_exc
    else:
        identity = guard.record_file(target)

    _verify_download_snapshot(
        snapshot,
        check_directories=guard.assert_current,
        downloads=guard.downloads,
    )
    _verify_built_video(target, identity, snapshot, guard)
    return identity


def _json_text(value: object) -> str:
    plain = _plain_json(value)
    try:
        return json.dumps(
            plain,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise BatchBuildError("metadata could not be encoded as strict JSON") from exc


def _write_atomic(path: Path, content: str, guard: _BuildGuard) -> None:
    guard.prepare_file(path)
    temporary, identity = guard.make_temp(path)
    try:
        guard.assert_file(temporary, identity)
        descriptor = guard.open_for_write(temporary, identity)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        guard.assert_file(temporary, identity)
        guard.install_temp(temporary, path, identity)
    except Exception as exc:
        guard.discard_file(temporary, identity)
        if isinstance(exc, BatchBuildError):
            raise
        raise BatchBuildError(
            f"could not write metadata file: {exc.__class__.__name__}"
        ) from exc


def _write_json(path: Path, value: object, guard: _BuildGuard) -> None:
    _write_atomic(path, _json_text(value), guard)


def _write_jsonl(
    path: Path, records: Sequence[object], guard: _BuildGuard
) -> None:
    _write_atomic(path, "".join(_json_text(record) for record in records), guard)


def _target_path_for_role(target_index: int, role: str) -> str:
    if role == "parquet":
        return f"data/chunk-000/episode_{target_index:06d}.parquet"
    return f"videos/chunk-000/{role}/episode_{target_index:06d}.mp4"


def _build_metadata(
    downloaded_episodes: Sequence[DownloadedEpisode],
    task_indices: Mapping[str, int],
    *,
    total_frames: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    info = dict(_plain_json(downloaded_episodes[0].source.info, "info"))
    info.update(
        {
            "data_path": "data/chunk-000/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4",
            "total_episodes": EXPECTED_EPISODES,
            "total_frames": total_frames,
            "total_tasks": len(task_indices),
            "total_videos": EXPECTED_EPISODES * len(VIDEO_ROLES),
            "total_chunks": 1,
            "chunks_size": EXPECTED_EPISODES,
            "splits": {"train": f"0:{EXPECTED_EPISODES}"},
        }
    )

    tasks_by_index: list[dict[str, object] | None] = [None] * len(task_indices)
    for downloaded in downloaded_episodes:
        source = downloaded.source
        task_index = task_indices[source.task]
        if tasks_by_index[task_index] is None:
            task_record = dict(_plain_json(source.task_record, "task_record"))
            task_record.update({"task_index": task_index, "task": source.task})
            tasks_by_index[task_index] = task_record

    episodes: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    expert_records: list[dict[str, object]] = []
    global_start = 0
    for target_index, downloaded in enumerate(downloaded_episodes):
        source = downloaded.source
        episode = dict(
            _rewrite_episode_index(
                _plain_json(source.episode_record, "episode_record"), target_index
            )
        )
        episode.update(
            {
                "episode_index": target_index,
                "tasks": [source.task],
                "length": source.length,
                "dataset_from_index": global_start,
                "dataset_to_index": global_start + source.length,
                "source_host": source.host,
                "source_dataset": source.dataset_name,
                "source_dataset_root": source.dataset_root,
                "source_episode_index": source.source_index,
                "source_fingerprint": source.fingerprint,
                "source_completed_ns": source.completed_ns,
            }
        )
        episodes.append(episode)

        statistic = dict(
            _rewrite_episode_index(
                _plain_json(source.stats_record, "stats_record"), target_index
            )
        )
        statistic["episode_index"] = target_index
        stats.append(statistic)

        expert = dict(
            _rewrite_episode_index(
                _plain_json(source.expert_record, "expert_record"), target_index
            )
        )
        expert["episode_index"] = target_index
        expert_records.append(expert)
        global_start += source.length

    return (
        info,
        [record for record in tasks_by_index if record is not None],
        episodes,
        stats,
        {"episodes": expert_records},
    )


def _file_manifest(
    root: Path, paths: Sequence[str], guard: _BuildGuard
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in sorted(paths):
        path = root / Path(relative)
        identity = guard.assert_file(path)
        io_path = guard.io_path(path)
        try:
            details = io_path.stat()
        except OSError as exc:
            raise BatchBuildError(
                f"could not inspect built dataset file: {exc.__class__.__name__}"
            ) from exc
        if not stat.S_ISREG(details.st_mode):
            raise BatchBuildError("built dataset core files must be regular files")
        digest = _sha256_file(io_path)
        guard.assert_file(path, identity)
        records.append(
            {
                "target_path": relative,
                "size": details.st_size,
                "sha256": digest,
            }
        )
    return records


def _source_file_manifest(source_file: SourceFile, target_path: str) -> dict[str, object]:
    return {
        "role": source_file.role,
        "absolute_path": source_file.absolute_path,
        "relative_path": source_file.relative_path,
        "target_path": target_path,
        "size": source_file.size,
        "mtime_ns": source_file.mtime_ns,
        "sha256": source_file.sha256,
    }


def build_batch(
    downloaded_episodes: Sequence[DownloadedEpisode],
    dataset_root: Path,
    *,
    batch_id: str,
    created_at: str,
    tool_version: str = DEFAULT_TOOL_VERSION,
) -> BuiltBatch:
    """Rewrite exactly twenty verified downloads into a clean staging dataset."""

    _validate_created_at(created_at)
    if not isinstance(tool_version, str) or not tool_version:
        raise BatchBuildError("tool_version must be a nonempty string")
    layout = _validate_layout(Path(dataset_root), batch_id)
    root = layout.root
    selected = tuple(downloaded_episodes)
    snapshot_maps = _validate_inputs(selected, layout)

    task_indices: dict[str, int] = {}
    for downloaded in selected:
        task_indices.setdefault(downloaded.source.task, len(task_indices))

    guard = _prepare_target(layout)
    try:
        global_start = 0
        core_paths: list[str] = []
        built_videos: list[tuple[Path, _PathIdentity, _VerifiedDownload]] = []
        for target_index, downloaded in enumerate(selected):
            task_index = task_indices[downloaded.source.task]
            parquet_relative = _target_path_for_role(target_index, "parquet")
            parquet_snapshot = snapshot_maps[target_index]["parquet"]

            def verify_parquet(
                snapshot: _VerifiedDownload = parquet_snapshot,
            ) -> None:
                _verify_download_snapshot(
                    snapshot,
                    check_directories=guard.assert_current,
                    downloads=guard.downloads,
                )

            rows = rewrite_parquet(
                parquet_snapshot.path,
                root / Path(parquet_relative),
                episode_index=target_index,
                global_start=global_start,
                task_index=task_index,
                expected_rows=downloaded.source.length,
                _guard=guard,
                _verify_source=verify_parquet,
            )
            global_start += rows
            core_paths.append(parquet_relative)
            for role in VIDEO_ROLES:
                video_relative = _target_path_for_role(target_index, role)
                video_target = root / Path(video_relative)
                video_snapshot = snapshot_maps[target_index][role]
                video_identity = _install_video(
                    video_snapshot,
                    video_target,
                    guard,
                )
                built_videos.append(
                    (video_target, video_identity, video_snapshot)
                )
                core_paths.append(video_relative)

        info, tasks, episodes, stats, expert = _build_metadata(
            selected, task_indices, total_frames=global_start
        )
        metadata = {
            "meta/info.json": info,
            "meta/tasks.jsonl": tasks,
            "meta/episodes.jsonl": episodes,
            "meta/episodes_stats.jsonl": stats,
            "meta/expert_frame_index.json": expert,
        }
        _write_json(root / "meta/info.json", info, guard)
        _write_jsonl(root / "meta/tasks.jsonl", tasks, guard)
        _write_jsonl(root / "meta/episodes.jsonl", episodes, guard)
        _write_jsonl(root / "meta/episodes_stats.jsonl", stats, guard)
        _write_json(root / "meta/expert_frame_index.json", expert, guard)
        core_paths.extend(metadata)
        expected_core_files = EXPECTED_EPISODES * (1 + len(VIDEO_ROLES)) + len(metadata)
        if len(core_paths) != expected_core_files or len(set(core_paths)) != expected_core_files:
            raise BatchBuildError("built dataset core file set is incomplete or duplicated")

        core_file_paths = [root / Path(relative) for relative in core_paths]
        guard.assert_owned_tree(core_file_paths)
        files = _file_manifest(root, core_paths, guard)

        # This final pass happens after every source consumer and after target
        # hashing. Only after it succeeds may the returned build-level summary
        # claim that all source snapshots were verified.
        for snapshot_map in snapshot_maps:
            for role in REQUIRED_ROLES:
                _verify_download_snapshot(
                    snapshot_map[role],
                    check_directories=guard.assert_current,
                    downloads=guard.downloads,
                )
        for target, identity, snapshot in built_videos:
            _verify_built_video(target, identity, snapshot, guard)
        guard.assert_owned_tree(core_file_paths)
        all_source_files_verified = True

        manifest_episodes: list[dict[str, object]] = []
        for target_index, (downloaded, snapshot_map) in enumerate(
            zip(selected, snapshot_maps, strict=True)
        ):
            source = downloaded.source
            manifest_episodes.append(
                {
                    "target_index": target_index,
                    "fingerprint": source.fingerprint,
                    "source_host": source.host,
                    "source_dataset": source.dataset_name,
                    "source_dataset_root": source.dataset_root,
                    "source_index": source.source_index,
                    "source_task": source.task,
                    "source_completed_ns": source.completed_ns,
                    "target_task_index": task_indices[source.task],
                    "frame_count": source.length,
                    "source_files": [
                        _source_file_manifest(
                            snapshot_map[role].source_file,
                            _target_path_for_role(target_index, role),
                        )
                        for role in REQUIRED_ROLES
                    ],
                }
            )

        source_bytes = sum(
            snapshot.source_file.size
            for snapshot_map in snapshot_maps
            for snapshot in snapshot_map.values()
        )
        dataset_bytes = sum(int(record["size"]) for record in files)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "batch_id": batch_id,
            "created_at": created_at,
            "tool_version": tool_version,
            "episode_count": EXPECTED_EPISODES,
            "frame_count": global_start,
            "task_count": len(task_indices),
            "video_count": EXPECTED_EPISODES * len(VIDEO_ROLES),
            "source_file_count": EXPECTED_EPISODES * len(REQUIRED_ROLES),
            "source_bytes": source_bytes,
            "dataset_bytes": dataset_bytes,
            "source_hosts": sorted({item.source.host for item in selected}),
            "source_dataset_roots": sorted(
                {item.source.dataset_root for item in selected}
            ),
            "episode_fingerprints": [item.source.fingerprint for item in selected],
            "episodes": manifest_episodes,
            "files": files,
            "validation": {
                "status": "build_artifacts_verified",
                "scope": "source snapshots and built core files; ffprobe and labeler validation pending",
                "core_file_count": len(files),
                "all_source_files_verified": all_source_files_verified,
                "full_batch_validation": False,
            },
        }
        try:
            _validate_manifest(manifest)
        except ManifestError as exc:
            raise BatchBuildError(f"built manifest is invalid: {exc}") from exc

        return BuiltBatch(
            root=root,
            batch_id=batch_id,
            episode_count=EXPECTED_EPISODES,
            frame_count=global_start,
            file_count=len(files),
            manifest=manifest,
        )
    except Exception as exc:
        guard.cleanup()
        if isinstance(exc, BatchBuildError):
            raise
        raise BatchBuildError(
            f"could not build dataset batch: {exc.__class__.__name__}"
        ) from exc
    finally:
        guard.close()
