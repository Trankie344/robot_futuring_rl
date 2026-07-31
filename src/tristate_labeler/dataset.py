from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from .models import DatasetInfo, EpisodeInfo, ExpertSegment, REQUIRED_VIDEO_KEYS, VideoRef

INFO_REQUIRED_SCALARS = ("fps", "total_episodes", "total_frames")
EPISODE_REQUIRED_COLUMNS = (
    "episode_index",
    "length",
    "dataset_from_index",
    "dataset_to_index",
    *(f"videos/{key}/{field}" for key in REQUIRED_VIDEO_KEYS for field in ("chunk_index", "file_index")),
)


def load_dataset(root: Path) -> DatasetInfo:
    root = root.resolve()
    info_path = root / "meta" / "info.json"
    info = _read_info(info_path)
    expert_segments = _read_expert_segments(root)
    if (root / "meta" / "episodes.jsonl").exists():
        video_keys = _video_keys_from_info(info)
        episodes = _read_v21_episodes(root, info, video_keys, expert_segments)
    else:
        _validate_legacy_required_video_keys(info, info_path)
        video_keys = REQUIRED_VIDEO_KEYS
        episodes = _read_episodes(root, expert_segments)
    return DatasetInfo(
        name=root.name,
        root=root,
        fps=int(info["fps"]),
        total_episodes=int(info["total_episodes"]),
        total_frames=int(info["total_frames"]),
        video_keys=video_keys,
        episodes=episodes,
    )


def load_dataset_collection(root: Path) -> tuple[DatasetInfo, ...]:
    root = root.resolve()
    if (root / "meta" / "info.json").exists():
        return (load_dataset(root),)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {root}")
    children = sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "meta" / "info.json").exists()
    )
    if not children:
        raise FileNotFoundError(f"No LeRobot datasets found directly under {root}")
    datasets = []
    for child in children:
        try:
            datasets.append(load_dataset(child))
        except Exception as exc:
            raise RuntimeError(f"Failed to load LeRobot dataset at {child}: {exc}") from exc
    return tuple(datasets)


def _read_info(info_path: Path) -> dict[str, object]:
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata file: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    _validate_info(info, info_path)
    return info


def _validate_info(info: dict[str, object], info_path: Path) -> None:
    missing_scalars = [field for field in INFO_REQUIRED_SCALARS if field not in info]
    if missing_scalars:
        raise ValueError(f"Missing required field(s) {', '.join(missing_scalars)} in {info_path}")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Missing required field features in {info_path}")


def _validate_legacy_required_video_keys(info: dict[str, object], info_path: Path) -> None:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Missing required field features in {info_path}")
    for key in REQUIRED_VIDEO_KEYS:
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(f"Missing required video feature {key} in {info_path}")
        if feature.get("dtype") != "video":
            raise ValueError(f"Required feature {key} must have dtype video in {info_path}")


def _video_keys_from_info(info: dict[str, object]) -> tuple[str, ...]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Missing required field features in dataset metadata")
    video_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if not video_keys:
        raise ValueError("Dataset metadata does not define any video features")

    ordered = []
    for preferred_key in (
        "observation.images.top",
        "observation.images.ground",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ):
        if preferred_key in video_keys:
            ordered.append(preferred_key)
    ordered.extend(key for key in video_keys if key not in ordered)
    return tuple(ordered)


def _read_expert_segments(root: Path) -> dict[int, tuple[ExpertSegment, ...]]:
    path = root / "meta" / "expert_frame_index.json"
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError(f"Invalid expert segment metadata in {path}")

    by_episode: dict[int, tuple[ExpertSegment, ...]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        episode_index = int(episode["episode_index"])
        segments = []
        for segment in episode.get("segments", []):
            segments.append(
                ExpertSegment(
                    start_frame=int(segment["start_frame_index"]),
                    end_frame=int(segment["end_frame_index"]) + 1,
                )
            )
        by_episode[episode_index] = tuple(segments)
    return by_episode


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset episodes file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_v21_episodes(
    root: Path,
    info: dict[str, object],
    video_keys: tuple[str, ...],
    expert_segments: dict[int, tuple[ExpertSegment, ...]],
) -> tuple[EpisodeInfo, ...]:
    data_path_template = str(info.get("data_path", ""))
    video_path_template = str(info.get("video_path", ""))
    if not data_path_template:
        raise ValueError(f"Missing required field data_path in {root / 'meta' / 'info.json'}")
    if not video_path_template:
        raise ValueError(f"Missing required field video_path in {root / 'meta' / 'info.json'}")

    rows = sorted(_read_jsonl(root / "meta" / "episodes.jsonl"), key=lambda row: int(row["episode_index"]))
    episodes = []
    running_from = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        data_path = root / Path(data_path_template.format(episode_index=episode_index))
        if not data_path.exists():
            raise FileNotFoundError(f"Missing episode parquet file: {data_path}")
        videos = MappingProxyType(
            {
                key: _v21_video_ref(root, video_path_template, key, episode_index)
                for key in video_keys
            }
        )
        episodes.append(
            EpisodeInfo(
                episode_index=episode_index,
                length=length,
                dataset_from_index=running_from,
                dataset_to_index=running_from + length,
                videos=videos,
                expert_segments=expert_segments.get(episode_index, ()),
            )
        )
        running_from += length
    return tuple(episodes)


def _v21_video_ref(root: Path, template: str, key: str, episode_index: int) -> VideoRef:
    relative_path = Path(template.format(video_key=key, episode_index=episode_index))
    if not (root / relative_path).exists():
        raise FileNotFoundError(f"Missing video file for {key}: {root / relative_path}")
    return VideoRef(key=key, chunk_index=0, file_index=episode_index, relative_path=relative_path)


def _read_episodes(root: Path, expert_segments: dict[int, tuple[ExpertSegment, ...]]) -> tuple[EpisodeInfo, ...]:
    episodes_root = root / "meta" / "episodes"
    files = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {episodes_root}")

    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    _validate_episode_columns(df, episodes_root)
    df = df.sort_values("episode_index")
    return tuple(_episode_from_row(root, row, expert_segments) for row in df.to_dict(orient="records"))


def _validate_episode_columns(df: pd.DataFrame, episodes_root: Path) -> None:
    missing_columns = [column for column in EPISODE_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) {', '.join(missing_columns)} in episode parquet files under {episodes_root}"
        )


def _episode_from_row(
    root: Path,
    row: dict[str, object],
    expert_segments: dict[int, tuple[ExpertSegment, ...]],
) -> EpisodeInfo:
    episode_index = int(row["episode_index"])
    videos = MappingProxyType({key: _video_ref_for_row(root, row, key) for key in REQUIRED_VIDEO_KEYS})
    return EpisodeInfo(
        episode_index=episode_index,
        length=int(row["length"]),
        dataset_from_index=int(row["dataset_from_index"]),
        dataset_to_index=int(row["dataset_to_index"]),
        videos=videos,
        expert_segments=expert_segments.get(episode_index, ()),
    )


def _video_ref_for_row(root: Path, row: dict[str, object], key: str) -> VideoRef:
    chunk_index = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    relative_path = Path(f"videos/{key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    if not (root / relative_path).exists():
        raise FileNotFoundError(f"Missing video file for {key}: {root / relative_path}")
    return VideoRef(key=key, chunk_index=chunk_index, file_index=file_index, relative_path=relative_path)
