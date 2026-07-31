from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REQUIRED_VIDEO_KEYS: tuple[str, ...] = (
    "observation.images.ground",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)

LABEL_PROGRESSIVE = 1
LABEL_REGRESSIVE = -1
LABEL_STAGNANT = 0
LABEL_DONE = 2
VALID_LABELS: frozenset[int] = frozenset({LABEL_PROGRESSIVE, LABEL_REGRESSIVE, LABEL_STAGNANT})
VALID_FRAME_STATES: frozenset[int] = frozenset({
    LABEL_PROGRESSIVE,
    LABEL_REGRESSIVE,
    LABEL_STAGNANT,
    LABEL_DONE,
})
INTERVAL_FRAMES = 30
INTERVAL_COUNT = 4
WINDOW_FRAMES = INTERVAL_FRAMES * INTERVAL_COUNT

TASK_PENDING = "pending"
TASK_LOCKED = "locked"
TASK_COMPLETED = "completed"
TASK_REVIEW = "review"


@dataclass(frozen=True)
class VideoRef:
    key: str
    chunk_index: int
    file_index: int
    relative_path: Path


@dataclass(frozen=True)
class ExpertSegment:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    videos: Mapping[str, VideoRef]
    expert_segments: tuple[ExpertSegment, ...] = ()


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    root: Path
    fps: int
    total_episodes: int
    total_frames: int
    video_keys: tuple[str, ...]
    episodes: tuple[EpisodeInfo, ...]
