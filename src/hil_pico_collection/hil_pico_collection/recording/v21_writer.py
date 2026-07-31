"""LeRobot v2.1 filesystem writer for sealed HIL Pico episodes."""

import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .episode_buffer import SealedEpisode

CHUNK_SIZE = 1000
CHUNK_NAME = "chunk-000"
DEFAULT_VIDEO_SHAPE = [480, 640, 3]
VIDEO_SAMPLE_FRAME_COUNT = 100


def _video_stream_info(height: int, width: int, fps: int) -> Dict[str, Any]:
    """Return the descriptor produced by pinned LeRobot's get_video_info()."""
    return {
        "video.height": int(height),
        "video.width": int(width),
        "video.codec": "mpeg4",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": int(fps),
        "video.channels": 3,
        "has_audio": False,
    }


EXPERT_FRAME_INDEX_FILE_NAME = "expert_frame_index.json"
IMAGE_FEATURES = {
    "observation.images.top": "top",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}
METADATA_FILE_NAMES = (
    "info.json",
    "episodes.jsonl",
    "tasks.jsonl",
    "episodes_stats.jsonl",
    EXPERT_FRAME_INDEX_FILE_NAME,
)
DATASET_VECTOR_NAMES = [
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "left_gripper",
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "right_gripper",
]

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": [16],
        "names": list(DATASET_VECTOR_NAMES),
    },
    "action": {
        "dtype": "float32",
        "shape": [16],
        "names": list(DATASET_VECTOR_NAMES),
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": list(DEFAULT_VIDEO_SHAPE),
        "names": ["height", "width", "channels"],
        "info": _video_stream_info(DEFAULT_VIDEO_SHAPE[0], DEFAULT_VIDEO_SHAPE[1], 30),
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": list(DEFAULT_VIDEO_SHAPE),
        "names": ["height", "width", "channels"],
        "info": _video_stream_info(DEFAULT_VIDEO_SHAPE[0], DEFAULT_VIDEO_SHAPE[1], 30),
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": list(DEFAULT_VIDEO_SHAPE),
        "names": ["height", "width", "channels"],
        "info": _video_stream_info(DEFAULT_VIDEO_SHAPE[0], DEFAULT_VIDEO_SHAPE[1], 30),
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
    "capture_timestamp": {"dtype": "float64", "shape": [1], "names": None},
    "intervention": {"dtype": "bool", "shape": [1], "names": None},
    "control_mode": {"dtype": "int64", "shape": [1], "names": None},
}

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("observation.state", pa.list_(pa.float32(), 16)),
        pa.field("action", pa.list_(pa.float32(), 16)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("capture_timestamp", pa.float64()),
        pa.field("intervention", pa.bool_()),
        pa.field("control_mode", pa.int64()),
    ]
)


def _feature_template(
    state_names: Sequence[str],
    action_names: Sequence[str],
    image_shapes: Mapping[str, Sequence[int]],
    fps: int,
) -> Dict[str, Any]:
    features = deepcopy(FEATURES)
    features["observation.state"]["shape"] = [len(state_names)]
    features["observation.state"]["names"] = list(state_names)
    features["action"]["shape"] = [len(action_names)]
    features["action"]["names"] = list(action_names)
    for feature_key in IMAGE_FEATURES:
        features.pop(feature_key, None)
    for image_name, configured_shape in image_shapes.items():
        shape = [int(value) for value in configured_shape]
        features[f"observation.images.{image_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
            "info": _video_stream_info(shape[0], shape[1], fps),
        }
    return features


def _parquet_schema(state_dimension: int, action_dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), int(state_dimension))),
            pa.field("action", pa.list_(pa.float32(), int(action_dimension))),
            *list(PARQUET_SCHEMA)[2:],
        ]
    )


class LeRobotV21Writer:
    """Write sealed episodes using the LeRobot v2.1 chunked layout."""

    def __init__(
        self,
        root: Any,
        fps: int = 30,
        robot_type: str = "pico_hil",
        *,
        state_names: Optional[Sequence[str]] = None,
        action_names: Optional[Sequence[str]] = None,
        image_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> None:
        self.root = Path(root)
        self.fps = int(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.robot_type = str(robot_type)
        self.state_names = list(state_names or DATASET_VECTOR_NAMES)
        self.action_names = list(action_names or DATASET_VECTOR_NAMES)
        if not self.state_names or len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must be non-empty and unique")
        if not self.action_names or len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action_names must be non-empty and unique")
        self.enforce_image_shapes = image_shapes is not None
        configured_shapes = image_shapes or {image_key: DEFAULT_VIDEO_SHAPE for image_key in IMAGE_FEATURES.values()}
        self.image_features = {
            f"observation.images.{str(image_key)}": str(image_key) for image_key in configured_shapes
        }
        self.configured_image_shapes = {
            str(image_key): [int(value) for value in shape] for image_key, shape in configured_shapes.items()
        }
        if not self.image_features:
            raise ValueError("image_shapes must contain at least one image")
        for image_key, shape in self.configured_image_shapes.items():
            if len(shape) != 3 or min(shape) <= 0:
                raise ValueError(f"image shape for {image_key} must be [height,width,channels]")
        self.features = _feature_template(
            self.state_names,
            self.action_names,
            self.configured_image_shapes,
            self.fps,
        )
        self.parquet_schema = _parquet_schema(len(self.state_names), len(self.action_names))
        self._force_recompute_stats = False

    def write_episode(self, sealed: SealedEpisode) -> int:
        """Write one sealed episode and return its active dataset episode index."""
        try:
            frames = sealed.frames
            self._validate_frames(frames)
            self._validate_existing_dataset_contract(frames)

            old_tasks_by_index = self._read_tasks()
            old_episode_metadata = self._read_episode_metadata()
            tasks_by_index = dict(old_tasks_by_index)
            task = str(sealed.task)
            task_index = self._resolve_task_index(tasks_by_index, task, sealed.task_index)
            episode_index = self._allocate_episode_index(int(sealed.episode_index))
            global_index_start = self._total_active_frames()
            staging_dir = self._staging_episode_dir(episode_index)
            metadata_backup_dir = staging_dir / "metadata"
            cleanup_staging = True
            rollback_failed = False
            final_paths_reserved = False

            self._remove_path(staging_dir)
            try:
                self._ensure_episode_paths_absent(episode_index)
                final_paths_reserved = True

                self._write_parquet(
                    self._staged_parquet_path(staging_dir, episode_index),
                    frames,
                    episode_index,
                    task_index,
                    global_index_start,
                )
                for feature_key, image_key in self.image_features.items():
                    self._write_video(
                        self._staged_video_path(staging_dir, feature_key, episode_index),
                        frames,
                        image_key,
                    )

                self._move_staged_episode_files(staging_dir, episode_index)

                episode_metadata = dict(old_episode_metadata)
                episode_metadata[episode_index] = _json_ready(dict(sealed.metadata))
                original_metadata_files = self._backup_metadata_files(metadata_backup_dir)
                try:
                    self._write_metadata(tasks_by_index, episode_metadata)
                except Exception as exc:
                    rollback_errors = []
                    try:
                        self._cleanup_episode_files(episode_index)
                    except Exception as rollback_exc:
                        rollback_errors.append(rollback_exc)
                    try:
                        self._restore_metadata_files(metadata_backup_dir, original_metadata_files)
                    except Exception as rollback_exc:
                        rollback_errors.append(rollback_exc)

                    if rollback_errors:
                        cleanup_staging = False
                        rollback_failed = True
                        details = "; ".join(str(error) for error in rollback_errors)
                        raise RuntimeError("write_episode failed and rollback failed: {0}".format(details)) from exc
                    raise
            except Exception:
                if cleanup_staging:
                    self._remove_path(staging_dir)
                if final_paths_reserved and not rollback_failed:
                    self._cleanup_episode_files(episode_index)
                raise
            finally:
                if cleanup_staging:
                    self._remove_path(staging_dir)

            return episode_index
        finally:
            cleanup = getattr(sealed, "cleanup", None)
            if callable(cleanup):
                cleanup()

    def delete_episode(self, index: int) -> None:
        """Move one active episode aside, renumber later episodes, and rebuild metadata."""
        episode_index = int(index)
        active_indices = self._active_episode_indices()
        if episode_index not in active_indices:
            raise FileNotFoundError("episode {0:06d} does not exist".format(episode_index))

        tasks_by_index = self._read_tasks()
        old_episode_metadata = self._read_episode_metadata()
        new_episode_metadata = {}
        for old_index in active_indices:
            if old_index == episode_index:
                continue
            new_index = old_index - 1 if old_index > episode_index else old_index
            if old_index in old_episode_metadata:
                new_episode_metadata[new_index] = old_episode_metadata[old_index]

        transaction_dir = self._delete_transaction_dir(episode_index)
        metadata_backup_dir = transaction_dir / "metadata"
        cleanup_transaction = True
        journal = []
        self._remove_path(transaction_dir)
        original_metadata_files = self._backup_metadata_files(metadata_backup_dir)

        try:
            self._move_episode_to_deleted(episode_index, journal)

            for old_index in active_indices:
                if old_index <= episode_index:
                    continue
                new_index = old_index - 1
                self._move_episode_files(old_index, new_index, journal)

            self._normalize_active_parquet_indices()
            self._force_recompute_stats = True
            self._write_metadata(tasks_by_index, new_episode_metadata)
        except Exception as exc:
            rollback_errors = []
            try:
                self._rollback_delete_journal(journal)
                self._normalize_active_parquet_indices()
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
            try:
                self._restore_metadata_files(metadata_backup_dir, original_metadata_files)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)

            if rollback_errors:
                cleanup_transaction = False
                details = "; ".join(str(error) for error in rollback_errors)
                raise RuntimeError("delete_episode failed and rollback failed: {0}".format(details)) from exc
            raise
        finally:
            self._force_recompute_stats = False
            if cleanup_transaction:
                self._remove_path(transaction_dir)

    def _validate_frames(self, frames: Sequence[Mapping[str, Any]]) -> None:
        if not frames:
            raise ValueError("cannot write an episode with no frames")

        image_shapes: Dict[str, Any] = {}
        for frame_number, frame in enumerate(frames):
            _vector(frame, "observation.state", len(self.state_names))
            _vector(frame, "action", len(self.action_names))
            for image_key in self.image_features.values():
                image = _rgb_image(frame, image_key, frame_number)
                configured_shape = self.configured_image_shapes[image_key]
                if self.enforce_image_shapes and list(image.shape) != configured_shape:
                    raise ValueError(
                        f"image {image_key} frame {frame_number} has shape {image.shape}, "
                        f"expected configured shape {tuple(configured_shape)}"
                    )
                shape = image.shape[:2]
                expected_shape = image_shapes.setdefault(image_key, shape)
                if shape != expected_shape:
                    raise ValueError(
                        "image {0} frame {1} has shape {2}, expected height {3} width {4}".format(
                            image_key,
                            frame_number,
                            image.shape,
                            expected_shape[0],
                            expected_shape[1],
                        )
                    )

    def _validate_existing_dataset_contract(self, frames: Sequence[Mapping[str, Any]]) -> None:
        info_path = self._meta_dir / "info.json"
        if not info_path.exists():
            return

        info = self._read_json(info_path)
        required_info_keys = {
            "codebase_version",
            "robot_type",
            "fps",
            "total_episodes",
            "total_frames",
            "total_tasks",
            "total_videos",
            "total_chunks",
            "chunks_size",
            "splits",
            "data_path",
            "video_path",
            "features",
        }
        missing = sorted(required_info_keys - set(info))
        if missing:
            raise ValueError(
                "existing dataset is incompatible with LeRobot v2.1; missing info fields: {0}".format(
                    ", ".join(missing)
                )
            )
        if info.get("codebase_version") != "v2.1":
            raise ValueError("existing dataset is incompatible: codebase_version must be v2.1")
        if int(info.get("fps")) != self.fps:
            raise ValueError("existing dataset is incompatible: fps differs from writer fps")
        if str(info.get("robot_type")) != self.robot_type:
            raise ValueError("existing dataset is incompatible: robot_type differs from writer robot_type")
        if int(info.get("chunks_size")) != CHUNK_SIZE:
            raise ValueError("existing dataset is incompatible: chunks_size must be {0}".format(CHUNK_SIZE))
        if info.get("data_path") != "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet":
            raise ValueError("existing dataset is incompatible: non-canonical data_path")
        if info.get("video_path") != "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4":
            raise ValueError("existing dataset is incompatible: non-canonical video_path")

        existing_features = info.get("features")
        if not isinstance(existing_features, Mapping):
            raise ValueError("existing dataset is incompatible: features must be an object")
        for vector_key in ("observation.state", "action"):
            feature = existing_features.get(vector_key, {})
            expected_names = self.state_names if vector_key == "observation.state" else self.action_names
            if feature.get("names") != expected_names:
                raise ValueError(
                    "existing dataset has incompatible state/action ordering; use a new root or migrate it first"
                )

        first_frame = next(iter(frames))
        for feature_key, image_key in self.image_features.items():
            feature = existing_features.get(feature_key, {})
            expected_shape = feature.get("shape")
            actual_shape = list(_rgb_image(first_frame, image_key, 0).shape)
            if expected_shape != actual_shape:
                raise ValueError(
                    "{0} camera dimensions {1} differ from existing dataset dimensions {2}".format(
                        image_key,
                        actual_shape,
                        expected_shape,
                    )
                )
            expected_video_info = _video_stream_info(actual_shape[0], actual_shape[1], self.fps)
            if (
                feature.get("dtype") != "video"
                or feature.get("names") != ["height", "width", "channels"]
                or feature.get("info") != expected_video_info
                or "video_info" in feature
            ):
                raise ValueError(
                    "existing dataset is incompatible: feature {0} has invalid video metadata".format(feature_key)
                )

        for feature_key, expected in self.features.items():
            if feature_key in self.image_features:
                continue
            actual = existing_features.get(feature_key, {})
            for property_name in ("dtype", "shape", "names"):
                if actual.get(property_name) != expected.get(property_name):
                    raise ValueError(
                        "existing dataset is incompatible: feature {0} has invalid {1}".format(
                            feature_key,
                            property_name,
                        )
                    )

        for episode_index in self._active_episode_indices():
            schema = pq.read_schema(self._parquet_path(episode_index))
            for expected_field in self.parquet_schema:
                field_index = schema.get_field_index(expected_field.name)
                if field_index < 0 or schema.field(field_index).type != expected_field.type:
                    raise ValueError(
                        "existing dataset is incompatible: Parquet feature {0} has a non-canonical type".format(
                            expected_field.name
                        )
                    )

    def _write_parquet(
        self,
        path: Path,
        frames: Iterable[Mapping[str, Any]],
        episode_index: int,
        task_index: int,
        global_index_start: int,
    ) -> None:
        columns = {
            "observation.state": [],
            "action": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "index": [],
            "task_index": [],
            "capture_timestamp": [],
            "intervention": [],
            "control_mode": [],
        }
        for row_index, frame in enumerate(frames):
            columns["observation.state"].append(_vector(frame, "observation.state", len(self.state_names)).tolist())
            columns["action"].append(_vector(frame, "action", len(self.action_names)).tolist())
            canonical_timestamp = row_index / float(self.fps)
            columns["timestamp"].append(canonical_timestamp)
            columns["frame_index"].append(int(row_index))
            columns["episode_index"].append(int(episode_index))
            columns["index"].append(int(global_index_start + row_index))
            columns["task_index"].append(int(task_index))
            columns["capture_timestamp"].append(float(frame.get("timestamp", canonical_timestamp)))
            columns["intervention"].append(bool(frame.get("intervention", False)))
            columns["control_mode"].append(int(frame.get("control_mode", 0)))

        self._write_parquet_columns(path, columns)

    def _write_parquet_columns(self, path: Path, columns: Mapping[str, List[Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {field.name: columns[field.name] for field in self.parquet_schema},
            schema=self.parquet_schema,
        )
        pq.write_table(table, path)

    def _write_video(self, path: Path, frames: Iterable[Mapping[str, Any]], image_key: str) -> None:
        iterator = iter(frames)
        try:
            first_frame = next(iterator)
        except StopIteration as exc:
            raise ValueError("cannot write video with no frames") from exc

        first = _rgb_image(first_frame, image_key, 0)
        height, width = first.shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("failed to open cv2 VideoWriter for {0}".format(path))

        try:
            writer.write(cv2.cvtColor(np.ascontiguousarray(first), cv2.COLOR_RGB2BGR))
            for frame_number, frame in enumerate(iterator, start=1):
                rgb = _rgb_image(frame, image_key, frame_number)
                if rgb.shape[:2] != (height, width):
                    raise ValueError(
                        "image {0} frame {1} has shape {2}, expected height {3} width {4}".format(
                            image_key,
                            frame_number,
                            rgb.shape,
                            height,
                            width,
                        )
                    )
                bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

    def _allocate_episode_index(self, requested_index: int) -> int:
        active_indices = self._active_episode_indices()
        if requested_index >= 0 and requested_index not in active_indices and requested_index == len(active_indices):
            return requested_index

        candidate = 0
        used = set(active_indices)
        while candidate in used:
            candidate += 1
        return candidate

    def _resolve_task_index(
        self,
        tasks_by_index: Dict[int, str],
        task: str,
        sealed_task_index: Optional[int],
    ) -> int:
        if sealed_task_index is not None:
            task_index = int(sealed_task_index)
            if task_index < 0:
                raise ValueError("task_index must be non-negative")
            existing = tasks_by_index.get(task_index)
            if existing is not None and existing != task:
                raise ValueError(
                    "task_index {0} already maps to {1!r}, not {2!r}".format(
                        task_index,
                        existing,
                        task,
                    )
                )
            tasks_by_index[task_index] = task
            return task_index

        for task_index, existing_task in sorted(tasks_by_index.items()):
            if existing_task == task:
                return task_index

        task_index = max(tasks_by_index.keys(), default=-1) + 1
        tasks_by_index[task_index] = task
        return task_index

    def _write_metadata(
        self,
        tasks_by_index: Dict[int, str],
        episode_metadata: Dict[int, Mapping[str, Any]],
    ) -> None:
        episodes = []
        stats = []
        expert_frame_indexes = []
        total_frames = 0
        features = self._dataset_features()
        existing_stats = {} if self._force_recompute_stats else self._read_episode_stats()

        for episode_index in self._active_episode_indices():
            parquet_path = self._parquet_path(episode_index)
            df = pd.read_parquet(parquet_path)
            length = int(len(df))
            total_frames += length
            task_index = int(df["task_index"].iloc[0]) if length else 0
            tasks_by_index.setdefault(task_index, "")

            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [tasks_by_index[task_index]],
                    "length": length,
                    "metadata": _json_ready(dict(episode_metadata.get(episode_index, {}))),
                }
            )
            episode_stats = existing_stats.get(episode_index)
            if not _stats_match_features(episode_stats, features):
                episode_stats = self._episode_stats(episode_index, df, features)
            stats.append(
                {
                    "episode_index": episode_index,
                    "stats": episode_stats,
                }
            )
            expert_frame_indexes.append(
                {
                    "episode_index": episode_index,
                    "segments": _expert_frame_segments(df),
                }
            )

        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self._meta_dir / "info.json",
            self._info(total_frames, len(episodes), tasks_by_index, features),
        )
        self._write_jsonl(
            self._meta_dir / "tasks.jsonl",
            [{"task_index": task_index, "task": task} for task_index, task in sorted(tasks_by_index.items())],
        )
        self._write_jsonl(self._meta_dir / "episodes.jsonl", episodes)
        self._write_jsonl(self._meta_dir / "episodes_stats.jsonl", stats)
        self._write_json(
            self._meta_dir / EXPERT_FRAME_INDEX_FILE_NAME,
            {"episodes": expert_frame_indexes},
        )

    def _dataset_features(self) -> Dict[str, Any]:
        features = deepcopy(self.features)
        existing_info = self._read_json(self._meta_dir / "info.json")
        existing_features = existing_info.get("features", {}) if isinstance(existing_info, Mapping) else {}
        active_indices = self._active_episode_indices()

        for feature_key in self.image_features:
            shape = None
            if active_indices:
                video_path = self._video_path(feature_key, active_indices[0])
                shape = self._video_shape(video_path)
            if shape is None:
                existing_feature = existing_features.get(feature_key, {})
                existing_shape = existing_feature.get("shape") if isinstance(existing_feature, Mapping) else None
                if isinstance(existing_shape, list) and len(existing_shape) == 3:
                    shape = [int(value) for value in existing_shape]
            if shape is None:
                image_key = self.image_features[feature_key]
                shape = list(self.configured_image_shapes[image_key])

            features[feature_key]["shape"] = shape
            features[feature_key]["info"] = _video_stream_info(shape[0], shape[1], self.fps)
        return features

    def _episode_stats(
        self,
        episode_index: int,
        df: pd.DataFrame,
        features: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        result = {}
        for feature_key, feature in features.items():
            if feature.get("dtype") == "video":
                result[feature_key] = self._video_stats(
                    self._video_path(feature_key, episode_index),
                    expected_frame_count=len(df),
                )
                continue
            if feature_key not in df.columns:
                raise ValueError("Parquet is missing declared feature {0}".format(feature_key))
            values = df[feature_key].tolist()
            result[feature_key] = _numeric_stats(values)
        return result

    def _video_shape(self, path: Path) -> Optional[List[int]]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("failed to open video metadata for {0}".format(path))
        try:
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        finally:
            capture.release()
        if height <= 0 or width <= 0:
            raise RuntimeError("video has invalid dimensions: {0}".format(path))
        return [height, width, 3]

    def _video_stats(self, path: Path, expected_frame_count: int) -> Dict[str, Any]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("failed to open video for statistics: {0}".format(path))

        try:
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            if frame_count <= 0:
                raise RuntimeError("video has no frames: {0}".format(path))
            if frame_count != int(expected_frame_count):
                raise RuntimeError(
                    "video frame count {0} does not match Parquet row count {1}: {2}".format(
                        frame_count,
                        expected_frame_count,
                        path,
                    )
                )
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(video_fps) or abs(video_fps - self.fps) > 1e-3:
                raise RuntimeError(
                    "video fps {0} does not match dataset fps {1}: {2}".format(
                        video_fps,
                        self.fps,
                        path,
                    )
                )
            sample_count = min(frame_count, VIDEO_SAMPLE_FRAME_COUNT)
            sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=np.int64)
            channel_min = np.full(3, np.inf, dtype=np.float64)
            channel_max = np.full(3, -np.inf, dtype=np.float64)
            channel_sum = np.zeros(3, dtype=np.float64)
            channel_sum_squares = np.zeros(3, dtype=np.float64)
            pixel_count = 0

            for frame_index in sample_indices.tolist():
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError("failed to decode video frame {0} from {1}".format(frame_index, path))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                pixels = rgb.reshape(-1, 3)
                channel_min = np.minimum(channel_min, pixels.min(axis=0))
                channel_max = np.maximum(channel_max, pixels.max(axis=0))
                channel_sum += pixels.sum(axis=0)
                channel_sum_squares += np.square(pixels).sum(axis=0)
                pixel_count += int(pixels.shape[0])
        finally:
            capture.release()

        mean = channel_sum / float(pixel_count)
        variance = np.maximum(channel_sum_squares / float(pixel_count) - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            "min": channel_min.reshape(3, 1, 1).tolist(),
            "max": channel_max.reshape(3, 1, 1).tolist(),
            "mean": mean.reshape(3, 1, 1).tolist(),
            "std": std.reshape(3, 1, 1).tolist(),
            "count": [int(sample_count)],
        }

    def _backup_metadata_files(self, backup_dir: Path) -> set:
        self._remove_path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        existing_files = set()
        for path in self._metadata_paths():
            self._assert_under_root(path)
            if not path.exists():
                continue
            backup_path = backup_dir / path.name
            self._assert_under_root(backup_path)
            shutil.copy2(path, backup_path)
            existing_files.add(path.name)
        return existing_files

    def _restore_metadata_files(self, backup_dir: Path, existing_files: set) -> None:
        for path in self._metadata_paths():
            self._assert_under_root(path)
            backup_path = backup_dir / path.name
            if path.name in existing_files:
                self._assert_under_root(backup_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, path)
            else:
                self._remove_path(path)

    def _metadata_paths(self) -> List[Path]:
        return [self._meta_dir / name for name in METADATA_FILE_NAMES]

    def _rollback_delete_journal(self, journal: List[Any]) -> None:
        errors = []
        for entry in reversed(journal):
            try:
                action = entry[0]
                if action == "move":
                    _action, moved_to, original_path = entry
                    if moved_to.exists():
                        self._safe_move(moved_to, original_path)
                        self._prune_deleted_empty_parents(moved_to.parent)
                else:
                    raise RuntimeError("unknown delete rollback action: {0}".format(action))
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _prune_deleted_empty_parents(self, path: Path) -> None:
        deleted_root = self._meta_dir / "deleted"
        try:
            path.resolve().relative_to(deleted_root.resolve())
        except ValueError:
            return

        current = path
        while current != deleted_root:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _info(
        self,
        total_frames: int,
        total_episodes: int,
        tasks_by_index: Mapping[int, str],
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": deepcopy(features),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "total_episodes": int(total_episodes),
            "total_frames": int(total_frames),
            "total_tasks": int(len(tasks_by_index)),
            "total_videos": int(total_episodes * len(self.image_features)),
            "total_chunks": int(math.ceil(total_episodes / float(CHUNK_SIZE))) if total_episodes else 0,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": "0:{0}".format(int(total_episodes))},
        }

    def _read_tasks(self) -> Dict[int, str]:
        tasks = {}
        for item in self._read_jsonl(self._meta_dir / "tasks.jsonl"):
            if "task_index" not in item or "task" not in item:
                continue
            tasks[int(item["task_index"])] = str(item["task"])
        return tasks

    def _read_episode_metadata(self) -> Dict[int, Mapping[str, Any]]:
        metadata = {}
        for item in self._read_jsonl(self._meta_dir / "episodes.jsonl"):
            if "episode_index" not in item:
                continue
            metadata[int(item["episode_index"])] = item.get("metadata", {})
        return metadata

    def _read_episode_stats(self) -> Dict[int, Mapping[str, Any]]:
        stats = {}
        for item in self._read_jsonl(self._meta_dir / "episodes_stats.jsonl"):
            if "episode_index" not in item or not isinstance(item.get("stats"), Mapping):
                continue
            stats[int(item["episode_index"])] = item["stats"]
        return stats

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        text = json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n"
        self._write_text_atomic(path, text)

    def _write_jsonl(self, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        text = "".join(json.dumps(_json_ready(row), sort_keys=True) + "\n" for row in rows)
        self._write_text_atomic(path, text)

    def _write_text_atomic(self, path: Path, text: str) -> None:
        self._assert_under_root(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(".{0}.tmp".format(path.name))
        self._assert_under_root(temp_path)
        self._remove_path(temp_path)
        try:
            temp_path.write_text(text, encoding="utf-8")
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                self._remove_path(temp_path)

    def _active_episode_indices(self) -> List[int]:
        data_root = self.root / "data"
        if not data_root.exists():
            return []
        indices = []
        for path in data_root.glob("chunk-*/episode_*.parquet"):
            index = _episode_index_from_path(path)
            if index is not None:
                indices.append(index)
        return sorted(indices)

    def _total_active_frames(self) -> int:
        total = 0
        for episode_index in self._active_episode_indices():
            total += int(pq.ParquetFile(self._parquet_path(episode_index)).metadata.num_rows)
        return total

    def _ensure_episode_paths_absent(self, episode_index: int) -> None:
        for relative_path in self._episode_relative_paths(episode_index):
            path = self.root / relative_path
            self._assert_under_root(path)
            if path.exists():
                raise FileExistsError(str(path))

    def _move_staged_episode_files(self, staging_dir: Path, episode_index: int) -> None:
        for relative_path in self._episode_relative_paths(episode_index):
            self._safe_move(staging_dir / relative_path, self.root / relative_path)

    def _cleanup_episode_files(self, episode_index: int) -> None:
        for relative_path in self._episode_relative_paths(episode_index):
            self._remove_path(self.root / relative_path)

    def _move_episode_to_deleted(self, episode_index: int, journal: Optional[List[Any]] = None) -> None:
        deleted_dir = self._deleted_episode_dir(episode_index)
        for relative_path in self._episode_relative_paths(episode_index):
            source = self.root / relative_path
            if not source.exists():
                continue
            destination = deleted_dir / relative_path
            self._safe_move(source, destination)
            if journal is not None:
                journal.append(("move", destination, source))

    def _move_episode_files(
        self,
        old_index: int,
        new_index: int,
        journal: Optional[List[Any]] = None,
    ) -> None:
        old_paths = self._episode_relative_paths(old_index)
        new_paths = self._episode_relative_paths(new_index)
        for old_relative_path, new_relative_path in zip(old_paths, new_paths):
            source = self.root / old_relative_path
            if not source.exists():
                continue
            destination = self.root / new_relative_path
            self._safe_move(source, destination)
            if journal is not None:
                journal.append(("move", destination, source))

    def _normalize_active_parquet_indices(self) -> None:
        global_index_start = 0
        for episode_index in self._active_episode_indices():
            path = self._parquet_path(episode_index)
            global_index_start += self._rewrite_parquet_contract_at_path(
                path,
                episode_index,
                global_index_start,
            )

    def _rewrite_parquet_contract_at_path(
        self,
        path: Path,
        episode_index: int,
        global_index_start: int,
    ) -> int:
        columns = pq.read_table(path).to_pydict()
        row_count = len(columns["episode_index"])
        incoming_timestamps = columns.get("capture_timestamp", columns.get("timestamp", []))
        columns["frame_index"] = list(range(row_count))
        columns["episode_index"] = [int(episode_index)] * row_count
        columns["index"] = [int(global_index_start + offset) for offset in range(row_count)]
        columns["timestamp"] = [offset / float(self.fps) for offset in range(row_count)]
        columns["capture_timestamp"] = [float(value) for value in incoming_timestamps]
        temp_path = path.with_name(path.name + ".tmp")
        self._remove_path(temp_path)
        try:
            self._write_parquet_columns(temp_path, columns)
            self._assert_under_root(path)
            self._assert_under_root(temp_path)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                self._remove_path(temp_path)
        return row_count

    def _safe_move(self, source: Path, destination: Path) -> None:
        self._assert_under_root(source)
        self._assert_under_root(destination)
        if destination.exists():
            raise FileExistsError(str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    def _remove_path(self, path: Path) -> None:
        self._assert_under_root(path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _assert_under_root(self, path: Path) -> None:
        root = self.root.resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("refusing to operate outside dataset root: {0}".format(path)) from exc

    def _episode_relative_paths(self, episode_index: int) -> List[Path]:
        name = _episode_file_name(episode_index)
        chunk_name = episode_chunk_name(episode_index)
        paths = [Path("data") / chunk_name / name.replace(".mp4", ".parquet")]
        for feature_key in self.image_features:
            paths.append(Path("videos") / chunk_name / feature_key / name)
        return paths

    def _deleted_episode_dir(self, episode_index: int) -> Path:
        base = self._meta_dir / "deleted" / _episode_stem(episode_index)
        if not base.exists():
            return base
        counter = 1
        while True:
            candidate = (
                self._meta_dir
                / "deleted"
                / "{0}_{1:03d}".format(
                    _episode_stem(episode_index),
                    counter,
                )
            )
            if not candidate.exists():
                return candidate
            counter += 1

    def _staging_episode_dir(self, episode_index: int) -> Path:
        return self._meta_dir / "tmp" / ("write_" + _episode_stem(episode_index))

    def _delete_transaction_dir(self, episode_index: int) -> Path:
        return self._meta_dir / "tmp" / ("delete_" + _episode_stem(episode_index))

    def _staged_parquet_path(self, staging_dir: Path, episode_index: int) -> Path:
        return (
            staging_dir / Path("data") / episode_chunk_name(episode_index) / (_episode_stem(episode_index) + ".parquet")
        )

    def _staged_video_path(self, staging_dir: Path, feature_key: str, episode_index: int) -> Path:
        return (
            staging_dir
            / Path("videos")
            / episode_chunk_name(episode_index)
            / feature_key
            / (_episode_stem(episode_index) + ".mp4")
        )

    def _parquet_path(self, episode_index: int) -> Path:
        return self._data_dir / episode_chunk_name(episode_index) / (_episode_stem(episode_index) + ".parquet")

    def _video_path(self, feature_key: str, episode_index: int) -> Path:
        return (
            self._video_dir / episode_chunk_name(episode_index) / feature_key / (_episode_stem(episode_index) + ".mp4")
        )

    @property
    def _data_dir(self) -> Path:
        return self.root / "data"

    @property
    def _video_dir(self) -> Path:
        return self.root / "videos"

    @property
    def _meta_dir(self) -> Path:
        return self.root / "meta"


def episode_chunk_name(episode_index: int, chunks_size: int = CHUNK_SIZE) -> str:
    episode_index = int(episode_index)
    chunks_size = int(chunks_size)
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if chunks_size <= 0:
        raise ValueError("chunks_size must be positive")
    return "chunk-{0:03d}".format(episode_index // chunks_size)


def _episode_stem(episode_index: int) -> str:
    return "episode_{0:06d}".format(int(episode_index))


def _episode_file_name(episode_index: int) -> str:
    return _episode_stem(episode_index) + ".mp4"


def _episode_index_from_path(path: Path) -> Optional[int]:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem[len(prefix) :])
    except ValueError:
        return None


def _expert_frame_segments(df: pd.DataFrame) -> List[Dict[str, int]]:
    segments: List[Dict[str, int]] = []
    start_frame_index: Optional[int] = None
    end_frame_index: Optional[int] = None

    for row_number, row in df.iterrows():
        frame_index = int(row["frame_index"]) if "frame_index" in df else int(row_number)
        intervention = bool(row["intervention"]) if "intervention" in df else False
        is_expert_frame = intervention

        if is_expert_frame:
            if start_frame_index is None:
                start_frame_index = frame_index
            end_frame_index = frame_index
            continue

        if start_frame_index is not None and end_frame_index is not None:
            segments.append(
                {
                    "start_frame_index": int(start_frame_index),
                    "end_frame_index": int(end_frame_index),
                }
            )
            start_frame_index = None
            end_frame_index = None

    if start_frame_index is not None and end_frame_index is not None:
        segments.append(
            {
                "start_frame_index": int(start_frame_index),
                "end_frame_index": int(end_frame_index),
            }
        )

    return segments


def _stats_match_features(
    stats: Optional[Mapping[str, Any]],
    features: Mapping[str, Any],
) -> bool:
    if not isinstance(stats, Mapping) or set(stats) != set(features):
        return False
    required = {"min", "max", "mean", "std", "count"}
    return all(isinstance(stats.get(key), Mapping) and set(stats[key]) == required for key in features)


def _numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("cannot compute numeric statistics for shape {0}".format(array.shape))

    numeric = array.astype(np.float64)
    return {
        "min": _json_ready(array.min(axis=0)),
        "max": _json_ready(array.max(axis=0)),
        "mean": numeric.mean(axis=0).tolist(),
        "std": numeric.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def _vector(frame: Mapping[str, Any], key: str, dimension: int) -> np.ndarray:
    if key not in frame:
        raise ValueError("frame is missing {0}".format(key))
    value = np.asarray(frame[key], dtype=np.float32)
    if value.shape != (dimension,):
        raise ValueError(f"{key} must have shape ({dimension},), got {value.shape}")
    return value


def _rgb_image(frame: Mapping[str, Any], image_key: str, frame_number: int) -> np.ndarray:
    images = frame.get("images")
    if not isinstance(images, Mapping):
        raise ValueError("frame {0} is missing images".format(frame_number))
    if image_key not in images:
        raise ValueError("frame {0} is missing image {1}".format(frame_number, image_key))

    image = np.asarray(images[image_key])
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image {0} frame {1} must have RGB shape (height, width, 3), got {2}".format(
                image_key,
                frame_number,
                image.shape,
            )
        )
    if image.dtype != np.uint8:
        raise ValueError(
            "image {0} frame {1} must have dtype uint8, got {2}".format(
                image_key,
                frame_number,
                image.dtype,
            )
        )
    return image


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(child) for child in value]
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    return value
