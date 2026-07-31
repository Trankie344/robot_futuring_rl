"""Local SSH transport and conversion for the read-only robot source probe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
from typing import Protocol

from . import remote_probe
from .models import MigrationConfig, ScanResult, SourceEpisode, SourceFile


class SourceScanError(RuntimeError):
    """The remote source inventory could not be obtained safely."""


class SourceChangedError(RuntimeError):
    """A selected source episode no longer matches its scanned snapshot."""


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(argv),
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
        )


_DEFAULT_RUNNER = SubprocessRunner()
_SOURCE_ROLES = frozenset(
    {
        "parquet",
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError
    return value


def _require_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError
    return value


def _require_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError
    return value


def _require_int(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError
    return value


def _normalized_posix_absolute(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or not path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError
    return value


def _normalized_posix_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError
    return value


def _require_sha256(record: Mapping[str, object], field: str) -> str:
    value = _require_string(record, field)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError
    return value


def _source_file(value: object, dataset_root: str) -> SourceFile:
    record = _require_mapping(value)
    absolute_path = _normalized_posix_absolute(_require_string(record, "absolute_path"))
    relative_path = _normalized_posix_relative(_require_string(record, "relative_path"))
    expected_path = (PurePosixPath(dataset_root) / PurePosixPath(relative_path)).as_posix()
    if absolute_path != expected_path:
        raise ValueError
    return SourceFile(
        role=_require_string(record, "role"),
        absolute_path=absolute_path,
        relative_path=relative_path,
        size=_require_int(record, "size"),
        mtime_ns=_require_int(record, "mtime_ns"),
        sha256=_require_sha256(record, "sha256"),
    )


def _metadata(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError
    return value


def _source_episode(value: object, host: str) -> SourceEpisode:
    record = _require_mapping(value)
    dataset_root = _normalized_posix_absolute(_require_string(record, "dataset_root"))
    files = tuple(
        _source_file(item, dataset_root)
        for item in _require_list(record.get("files"))
    )
    if {file.role for file in files} != _SOURCE_ROLES or len(files) != len(_SOURCE_ROLES):
        raise ValueError
    return SourceEpisode(
        host=host,
        dataset_root=dataset_root,
        dataset_name=_require_string(record, "dataset_name"),
        source_index=_require_int(record, "source_index"),
        task=_require_string(record, "task"),
        length=_require_int(record, "length"),
        completed_ns=_require_int(record, "completed_ns"),
        fingerprint=_require_sha256(record, "fingerprint"),
        files=files,
        info=_metadata(record, "info"),
        task_record=_metadata(record, "task_record"),
        episode_record=_metadata(record, "episode_record"),
        stats_record=_metadata(record, "stats_record"),
        expert_record=_metadata(record, "expert_record"),
    )


def _decode_scan_result(stdout: str, host: str) -> ScanResult:
    try:
        payload = json.loads(stdout)
        root = _require_mapping(payload)
        episodes = tuple(_source_episode(item, host) for item in _require_list(root.get("episodes")))
        busy_items = tuple(
            _normalized_posix_absolute(item)
            for item in _require_list(root.get("busy_roots"))
            if isinstance(item, str)
        )
        if len(busy_items) != len(_require_list(root.get("busy_roots"))):
            raise ValueError
        rejected: list[dict[str, str]] = []
        for item in _require_list(root.get("rejected_roots")):
            record = _require_mapping(item)
            rejected.append(
                {
                    "dataset_root": _normalized_posix_absolute(
                        _require_string(record, "dataset_root")
                    ),
                    "reason": _require_string(record, "reason"),
                }
            )
        keys = [(episode.dataset_root, episode.source_index) for episode in episodes]
        if len(keys) != len(set(keys)):
            raise ValueError
        return ScanResult(
            episodes=episodes,
            busy_roots=busy_items,
            rejected_roots=tuple(rejected),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SourceScanError("invalid remote probe response") from exc


class SshSourceScanner:
    def __init__(
        self,
        config: MigrationConfig,
        runner: CommandRunner = _DEFAULT_RUNNER,
    ) -> None:
        self._config = config
        self._runner = runner
        self._probe_source = Path(remote_probe.__file__).read_text(encoding="utf-8")

    def _argv(self, stable_seconds: float) -> list[str]:
        return [
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
            "--",
            self._config.robot,
            "python3",
            "-",
            "--source-root",
            shlex.quote(self._config.source_root),
            "--stable-seconds",
            shlex.quote(str(stable_seconds)),
        ]

    def _scan(self, stable_seconds: float) -> ScanResult:
        try:
            completed = self._runner.run(
                self._argv(stable_seconds),
                input_text=self._probe_source,
            )
        except OSError as exc:
            raise SourceScanError(f"could not start ssh: {exc.__class__.__name__}") from exc
        if completed.returncode != 0:
            raise SourceScanError(
                f"remote source probe failed with exit status {completed.returncode}"
            )
        return _decode_scan_result(completed.stdout, self._config.robot)

    def scan(self) -> ScanResult:
        return self._scan(self._config.stable_seconds)

    def revalidate(self, selected: Sequence[SourceEpisode]) -> None:
        if not selected:
            return
        refreshed = self._scan(0.0)
        current = {
            (episode.dataset_root, episode.source_index): episode
            for episode in refreshed.episodes
        }
        for original in selected:
            key = (original.dataset_root, original.source_index)
            replacement = current.get(key)
            if replacement is None or not _same_source(original, replacement):
                raise SourceChangedError(
                    f"selected source changed: {original.dataset_root} episode {original.source_index}"
                )


def _same_source(original: SourceEpisode, replacement: SourceEpisode) -> bool:
    return original == replacement
