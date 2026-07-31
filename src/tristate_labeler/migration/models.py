from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path


class RunStatus(str, Enum):
    CREATED = "CREATED"
    WAITING = "WAITING"
    BUSY = "BUSY"
    ERROR = "ERROR"


class _FrozenDict(dict):
    """A JSON-compatible dictionary that cannot be changed after construction."""

    def __init__(
        self,
        values: Mapping[str, object] | Iterable[tuple[str, object]] = (),
        /,
        **kwargs: object,
    ) -> None:
        items = values.items() if isinstance(values, Mapping) else values
        dict.__init__(self, ((key, _freeze_json(value)) for key, value in items))
        dict.update(self, ((key, _freeze_json(value)) for key, value in kwargs.items()))

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDict:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[object, tuple[dict[str, object]]]:
        return (type(self), (dict(self),))

    def __setitem__(self, key: object, value: object) -> None:
        raise TypeError("metadata is immutable")

    def __delitem__(self, key: object) -> None:
        raise TypeError("metadata is immutable")

    def clear(self) -> None:
        raise TypeError("metadata is immutable")

    def pop(self, key: object, default: object = None) -> object:
        raise TypeError("metadata is immutable")

    def popitem(self) -> tuple[object, object]:
        raise TypeError("metadata is immutable")

    def setdefault(self, key: object, default: object = None) -> object:
        raise TypeError("metadata is immutable")

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError("metadata is immutable")

    def __ior__(self, other: object) -> _FrozenDict:
        raise TypeError("metadata is immutable")


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return _FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class SourceFile:
    role: str
    absolute_path: str
    relative_path: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class SourceEpisode:
    host: str
    dataset_root: str
    dataset_name: str
    source_index: int
    task: str
    length: int
    completed_ns: int
    fingerprint: str
    files: tuple[SourceFile, ...]
    info: dict[str, object]
    task_record: dict[str, object]
    episode_record: dict[str, object]
    stats_record: dict[str, object]
    expert_record: dict[str, object]

    def __post_init__(self) -> None:
        for field_name in (
            "info",
            "task_record",
            "episode_record",
            "stats_record",
            "expert_record",
        ):
            object.__setattr__(self, field_name, _FrozenDict(getattr(self, field_name)))

    def sort_key(self) -> tuple[int, str, int, str]:
        return (self.completed_ns, self.dataset_name, self.source_index, self.fingerprint)


@dataclass(frozen=True)
class ScanResult:
    episodes: tuple[SourceEpisode, ...]
    busy_roots: tuple[str, ...] = ()
    rejected_roots: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected_roots", tuple(_FrozenDict(root) for root in self.rejected_roots))


@dataclass(frozen=True)
class MigrationConfig:
    output_root: Path
    robot: str = "zme@lite-0030.taild22f37.ts.net"
    source_root: str = "/home/zme/datasets"
    batch_size: int = 20
    stable_seconds: float = 3.0
    identity_file: Path = Path.home() / ".ssh/id_ed25519_hil_transfer"
    known_hosts_file: Path = Path.home() / ".ssh/known_hosts_hil_transfer"
    ffprobe: Path = Path("ffprobe")

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.stable_seconds):
            raise ValueError("stable_seconds must be finite")
        if self.stable_seconds < 0:
            raise ValueError("stable_seconds must be non-negative")
