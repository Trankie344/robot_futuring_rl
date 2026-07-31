"""Read-only LeRobot v2.1 dataset and video browsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from hil_pico_collection.recording.v21_writer import CHUNK_SIZE, IMAGE_FEATURES, episode_chunk_name

VIDEO_KEYS = ("top", "left_wrist", "right_wrist")
VIDEO_FEATURE_BY_KEY = {image_key: feature_key for feature_key, image_key in IMAGE_FEATURES.items()}


class ReplayDataset:
    """Browse saved metadata and videos without issuing robot commands."""

    def __init__(self, root: Any, fps: int | None = None) -> None:
        self.root = Path(root)
        info = self._read_info()
        self.fps = int(fps or info.get("fps", 30) or 30)
        self.chunks_size = int(info.get("chunks_size", CHUNK_SIZE) or CHUNK_SIZE)
        features = info.get("features", {})
        if not isinstance(features, Mapping):
            features = {}
        configured_videos = {
            key.removeprefix("observation.images."): key
            for key, value in features.items()
            if key.startswith("observation.images.") and isinstance(value, Mapping) and value.get("dtype") == "video"
        }
        self.video_feature_by_key = configured_videos or dict(VIDEO_FEATURE_BY_KEY)
        self.video_keys = tuple(self.video_feature_by_key)

    def list_episodes(self) -> list[dict[str, Any]]:
        tasks = self._read_tasks()
        metadata = self._read_episode_metadata()
        result = []
        for path in sorted((self.root / "data").glob("chunk-*/episode_*.parquet")):
            episode_index = _episode_index_from_path(path)
            if episode_index is not None:
                result.append(self._summary(episode_index, path, tasks, metadata))
        return result

    def episode_summary(self, episode_index: int) -> dict[str, Any]:
        path = self.parquet_path(episode_index)
        if not path.exists():
            raise FileNotFoundError(f"episode {int(episode_index):06d} does not exist")
        return self._summary(
            int(episode_index),
            path,
            self._read_tasks(),
            self._read_episode_metadata(),
        )

    def video_path(self, episode_index: int, video_key: str) -> Path:
        if video_key not in self.video_feature_by_key:
            raise ValueError(f"video_key must be one of: {', '.join(self.video_keys)}")
        feature_key = self.video_feature_by_key[video_key]
        chunk = episode_chunk_name(int(episode_index), self.chunks_size)
        path = self.root / "videos" / chunk / feature_key / f"episode_{int(episode_index):06d}.mp4"
        self._assert_under_root(path)
        return path

    def parquet_path(self, episode_index: int) -> Path:
        chunk = episode_chunk_name(int(episode_index), self.chunks_size)
        path = self.root / "data" / chunk / f"episode_{int(episode_index):06d}.parquet"
        self._assert_under_root(path)
        return path

    def _summary(
        self,
        episode_index: int,
        parquet_path: Path,
        tasks: Mapping[int, str],
        metadata: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        frame_data = pd.read_parquet(parquet_path, columns=["task_index", "frame_index"])
        frame_count = len(frame_data)
        task_index = int(frame_data["task_index"].iloc[0]) if frame_count else 0
        videos = {}
        for key in self.video_keys:
            path = self.video_path(episode_index, key)
            videos[key] = {
                "exists": path.exists(),
                "path": path.relative_to(self.root).as_posix(),
            }
        return {
            "episode_index": int(episode_index),
            "task": str(tasks.get(task_index, "")),
            "task_index": task_index,
            "frame_count": int(frame_count),
            "fps": self.fps,
            "status": "saved",
            "metadata": dict(metadata.get(int(episode_index), {})),
            "videos": videos,
        }

    def _read_info(self) -> dict[str, Any]:
        path = self.root / "meta" / "info.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _read_tasks(self) -> dict[int, str]:
        return {
            int(item["task_index"]): str(item["task"])
            for item in _read_jsonl(self.root / "meta" / "tasks.jsonl")
            if "task_index" in item and "task" in item
        }

    def _read_episode_metadata(self) -> dict[int, Mapping[str, Any]]:
        return {
            int(item["episode_index"]): item.get("metadata", {})
            for item in _read_jsonl(self.root / "meta" / "episodes.jsonl")
            if "episode_index" in item
        }

    def _assert_under_root(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"refusing to read outside dataset root: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_index_from_path(path: Path) -> int | None:
    prefix = "episode_"
    if not path.stem.startswith(prefix):
        return None
    try:
        return int(path.stem[len(prefix) :])
    except ValueError:
        return None
