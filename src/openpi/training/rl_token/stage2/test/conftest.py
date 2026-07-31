from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

VIDEO_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ready_batch(
    root: Path,
    *,
    lengths: tuple[int, ...] = (22,) * 20,
    completed_labels: bool = True,
) -> Path:
    assert len(lengths) == 20
    (root / "data/chunk-000").mkdir(parents=True)
    for key in VIDEO_KEYS:
        (root / "videos/chunk-000" / key).mkdir(parents=True)

    episodes = []
    stats = []
    expert = []
    label_episodes = []
    global_index = 0
    for episode_index, length in enumerate(lengths):
        frames = np.arange(length, dtype=np.int64)
        intervention = np.zeros(length, dtype=np.bool_)
        expert_start = 3
        if episode_index == 0 and length > expert_start:
            expert_end = min(4, length - 1)
            intervention[expert_start : expert_end + 1] = True
            expert.append(
                {
                    "episode_index": episode_index,
                    "segments": [{"start_frame_index": expert_start, "end_frame_index": expert_end}],
                }
            )
        table = pa.table(
            {
                "observation.state": pa.array(
                    [[float(frame)] * 16 for frame in frames],
                    type=pa.list_(pa.float32(), 16),
                ),
                "action": pa.array(
                    [[float(frame + 1)] * 16 for frame in frames],
                    type=pa.list_(pa.float32(), 16),
                ),
                "timestamp": pa.array(frames / 30.0, type=pa.float32()),
                "frame_index": pa.array(frames, type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "index": pa.array(
                    np.arange(global_index, global_index + length),
                    type=pa.int64(),
                ),
                "task_index": pa.array([0] * length, type=pa.int64()),
                "intervention": pa.array(intervention, type=pa.bool_()),
                "control_mode": pa.array([1] * length, type=pa.int64()),
            }
        )
        parquet_path = root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, parquet_path)
        for key in VIDEO_KEYS:
            (root / "videos/chunk-000" / key / f"episode_{episode_index:06d}.mp4").write_text(
                json.dumps({"frame_count": length}),
                encoding="utf-8",
            )
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": ["pick and place"],
                "length": length,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                "source_fingerprint": f"{episode_index + 1:064x}",
            }
        )
        stats.append({"episode_index": episode_index, "stats": {}})
        labels = [0] * length if completed_labels else [None] * length
        if completed_labels and episode_index == 0:
            labels[-1] = 2
        label_episodes.append(
            {
                "episode_index": episode_index,
                "frame_states": labels,
                "segments": [],
                "done_frames": [],
            }
        )
        global_index += length

    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "pico_hil",
            "fps": 30,
            "total_episodes": 20,
            "total_frames": global_index,
            "total_tasks": 1,
            "total_videos": 60,
            "total_chunks": 1,
            "chunks_size": 20,
            "splits": {"train": "0:20"},
            "data_path": "data/chunk-000/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [16]},
                "action": {"dtype": "float32", "shape": [16]},
                "intervention": {"dtype": "bool", "shape": [1]},
                "control_mode": {"dtype": "int64", "shape": [1]},
                **{key: {"dtype": "video", "shape": [480, 640, 3]} for key in VIDEO_KEYS},
            },
        },
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick and place"}])
    _write_jsonl(root / "meta/episodes.jsonl", episodes)
    _write_jsonl(root / "meta/episodes_stats.jsonl", stats)
    _write_json(root / "meta/expert_frame_index.json", {"episodes": expert})
    _write_json(
        root / "meta/tristate_labels.json",
        {
            "version": 2,
            "label_semantics": "per_frame_state_for_robot_progress_and_done",
            "states": [-1, 0, 1, 2, "done"],
            "datasets": [
                {
                    "dataset_id": 1,
                    "dataset_name": root.name,
                    "root_path": str(root),
                    "episodes": label_episodes,
                }
            ],
        },
    )

    core_paths = [
        *(root / "data/chunk-000").glob("*.parquet"),
        *(root / "videos/chunk-000").glob("*/*.mp4"),
        root / "meta/info.json",
        root / "meta/tasks.jsonl",
        root / "meta/episodes.jsonl",
        root / "meta/episodes_stats.jsonl",
        root / "meta/expert_frame_index.json",
    ]
    files = [
        {
            "target_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(core_paths)
    ]
    fingerprints = [record["source_fingerprint"] for record in episodes]
    manifest = {
        "schema_version": 1,
        "batch_id": root.name,
        "created_at": "2026-07-23T12:00:00+08:00",
        "episode_count": 20,
        "frame_count": global_index,
        "episode_fingerprints": fingerprints,
        "episodes": [
            {
                "target_index": index,
                "fingerprint": fingerprint,
                "source_host": "zme@robot",
                "source_dataset_root": f"/home/zme/datasets/source_{index:02d}",
                "source_index": index,
            }
            for index, fingerprint in enumerate(fingerprints)
        ],
        "files": files,
    }
    _write_json(root / "migration_manifest.json", manifest)
    (root / "READY").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def ready_batch(tmp_path: Path) -> Path:
    return build_ready_batch(tmp_path / "batch_000001_test")
