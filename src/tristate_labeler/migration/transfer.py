"""Resumable, locally verified transfer of scanned robot source files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import stat

from .models import MigrationConfig, SourceFile
from .source import CommandRunner, SubprocessRunner


class TransferError(RuntimeError):
    """A source file could not be transferred safely."""


class InsufficientSpaceError(TransferError):
    """The destination filesystem lacks the required free space."""


class ChecksumMismatchError(TransferError):
    """Downloaded bytes do not match the scanned source snapshot."""


GIB = 1024**3
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_RUNNER = SubprocessRunner()


def required_free_bytes(batch_bytes: int) -> int:
    """Return batch storage plus the required safety reserve."""
    if isinstance(batch_bytes, bool) or not isinstance(batch_bytes, int):
        raise TypeError("batch_bytes must be a non-negative integer")
    if batch_bytes < 0:
        raise ValueError("batch_bytes must be a non-negative integer")
    return batch_bytes + max(10 * GIB, (batch_bytes + 4) // 5)


def ensure_space(
    path: Path,
    *,
    batch_bytes: int,
    reusable_bytes: int = 0,
) -> None:
    """Require enough space on the nearest existing destination ancestor."""
    required = required_free_bytes(batch_bytes)
    if isinstance(reusable_bytes, bool) or not isinstance(reusable_bytes, int):
        raise TypeError("reusable_bytes must be a non-negative integer")
    if reusable_bytes < 0 or reusable_bytes > batch_bytes:
        raise ValueError(
            "reusable_bytes must be between zero and batch_bytes"
        )
    required = max(0, required - reusable_bytes)
    candidate = Path(path)
    while True:
        try:
            candidate.stat()
            break
        except FileNotFoundError:
            parent = candidate.parent
            if parent == candidate:
                raise TransferError("could not inspect destination disk space") from None
            candidate = parent
        except OSError as exc:
            raise TransferError(
                f"could not inspect destination disk space: {exc.__class__.__name__}"
            ) from exc

    try:
        available = shutil.disk_usage(candidate).free
    except OSError as exc:
        raise TransferError(
            f"could not inspect destination disk space: {exc.__class__.__name__}"
        ) from exc
    if available < required:
        raise InsufficientSpaceError(
            f"insufficient disk space: required {required} bytes, "
            f"available {available} bytes"
        )


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RsyncTransfer:
    def __init__(
        self,
        config: MigrationConfig,
        runner: CommandRunner = _DEFAULT_RUNNER,
    ) -> None:
        self._config = config
        self._runner = runner

    def _normalize_target(self, target: Path) -> Path:
        try:
            staging_root = Path(
                os.path.abspath(os.fspath(self._config.output_root / ".staging"))
            )
            normalized = Path(os.path.abspath(os.fspath(target)))
        except (OSError, TypeError, ValueError) as exc:
            raise TransferError(
                f"could not validate local staging target: {exc.__class__.__name__}"
            ) from exc
        if normalized == staging_root or not normalized.is_relative_to(staging_root):
            raise TransferError("local target must be strictly inside the staging directory")
        return normalized

    @staticmethod
    def _validate_parent_chain(parent: Path) -> None:
        for component in reversed((parent, *parent.parents)):
            try:
                component_stat = component.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise TransferError(
                    f"could not inspect local target parent: {exc.__class__.__name__}"
                ) from exc
            if stat.S_ISLNK(component_stat.st_mode):
                raise TransferError("local target parent must not be a symlink")
            if not stat.S_ISDIR(component_stat.st_mode):
                raise TransferError("local target parent must be a directory")

    def _argv(self, source: SourceFile, target: Path) -> list[str]:
        ssh_command = shlex.join(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={self._config.known_hosts_file}",
                "-i",
                str(self._config.identity_file),
            ]
        )
        return [
            "rsync",
            "-a",
            "--protect-args",
            "--partial",
            "--partial-dir=.rsync-partial",
            "-e",
            ssh_command,
            "--",
            f"{self._config.robot}:{source.absolute_path}",
            str(target),
        ]

    @staticmethod
    def _target_stat(target: Path):
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TransferError(
                f"could not inspect local target: {exc.__class__.__name__}"
            ) from exc
        if stat.S_ISLNK(target_stat.st_mode):
            raise TransferError("local target must not be a symlink")
        if not stat.S_ISREG(target_stat.st_mode):
            raise TransferError("local target must be a regular file")
        return target_stat

    @staticmethod
    def _hash_target(target: Path) -> str:
        try:
            return sha256_file(target)
        except OSError as exc:
            raise TransferError(
                f"could not read local target: {exc.__class__.__name__}"
            ) from exc

    @staticmethod
    def _remove_mismatched_target(target: Path) -> None:
        try:
            target.unlink()
        except OSError as exc:
            raise TransferError(
                f"could not remove mismatched local target: {exc.__class__.__name__}"
            ) from exc

    def fetch(self, source: SourceFile, target: Path) -> None:
        target = self._normalize_target(Path(target))
        self._validate_parent_chain(target.parent)
        existing = self._target_stat(target)
        if (
            existing is not None
            and existing.st_size == source.size
            and self._hash_target(target) == source.sha256
        ):
            return
        if existing is not None:
            self._remove_mismatched_target(target)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TransferError(
                f"could not create local target directory: {exc.__class__.__name__}"
            ) from exc
        self._validate_parent_chain(target.parent)

        try:
            completed = self._runner.run(self._argv(source, target))
        except OSError as exc:
            raise TransferError(f"could not start rsync: {exc.__class__.__name__}") from exc
        if completed.returncode != 0:
            raise TransferError(
                f"rsync failed with exit status {completed.returncode}"
            )

        downloaded = self._target_stat(target)
        if downloaded is None:
            raise ChecksumMismatchError("rsync did not produce the local target file")
        if downloaded.st_size != source.size:
            self._remove_mismatched_target(target)
            raise ChecksumMismatchError(
                f"downloaded size mismatch: expected {source.size} bytes, "
                f"received {downloaded.st_size} bytes"
            )
        if self._hash_target(target) != source.sha256:
            self._remove_mismatched_target(target)
            raise ChecksumMismatchError("downloaded SHA-256 mismatch")
