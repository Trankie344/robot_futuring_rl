import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hil_pico_collection.recording.episode_buffer import EpisodeBuffer, SealedEpisode, SpilledFrameSequence
from hil_pico_collection.recording.v21_writer import FEATURES, LeRobotV21Writer


def _rgb(value):
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    image[:, :, 1] = (value + 1) % 255
    image[:, :, 2] = (value + 2) % 255
    return image


def _sealed_episode(index=0, task="pick cup", task_index=None, frame_count=3):
    frames = []
    for frame_index in range(frame_count):
        frames.append(
            {
                "observation.state": np.arange(16, dtype=np.float32) + frame_index,
                "action": np.arange(100, 116, dtype=np.float32) + frame_index,
                "images": {
                    "top": _rgb(10 + frame_index),
                    "left_wrist": _rgb(20 + frame_index),
                    "right_wrist": _rgb(30 + frame_index),
                },
                "timestamp": 1.0 + frame_index / 30.0,
                "frame_index": frame_index,
                "episode_index": index,
                "task": task,
                "intervention": frame_index == 1,
                "control_mode": 5 if frame_index == 1 else 1,
            }
        )
    return SealedEpisode(
        episode_index=index,
        task=task,
        task_index=task_index,
        frames=frames,
        metadata={"source": "test"},
    )


def _jsonl(path):
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def _json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _active_episode_files(root):
    paths = []
    paths.extend((root / "data").glob("chunk-*/episode_*.parquet"))
    paths.extend((root / "videos").glob("chunk-*/*/episode_*.mp4"))
    return sorted(path.relative_to(root).as_posix() for path in paths)


def _metadata_file_bytes(root):
    meta_dir = root / "meta"
    names = ["info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]
    return {name: (meta_dir / name).read_bytes() for name in names if (meta_dir / name).exists()}


def test_metadata_writes_replace_temp_files_atomically(tmp_path, monkeypatch):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    replaced = []
    original_replace = Path.replace

    def spy_replace(self, target):
        replaced.append((Path(self), Path(target)))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy_replace)

    writer._write_json(tmp_path / "meta/info.json", {"ok": True})
    writer._write_jsonl(tmp_path / "meta/episodes.jsonl", [{"episode_index": 0}])

    assert json.loads((tmp_path / "meta/info.json").read_text()) == {"ok": True}
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == [{"episode_index": 0}]
    assert [target.name for _source, target in replaced] == ["info.json", "episodes.jsonl"]
    assert all(source.parent == target.parent for source, target in replaced)
    assert all(source.name.startswith(".") and source.name.endswith(".tmp") for source, _target in replaced)


def _assert_v21_parquet_schema(path):
    schema = pq.read_schema(path)
    state_type = schema.field("observation.state").type
    action_type = schema.field("action").type

    assert pa.types.is_fixed_size_list(state_type)
    assert state_type.list_size == 16
    assert pa.types.is_float32(state_type.value_type)
    assert pa.types.is_fixed_size_list(action_type)
    assert action_type.list_size == 16
    assert pa.types.is_float32(action_type.value_type)
    assert pa.types.is_float32(schema.field("timestamp").type)
    assert pa.types.is_int64(schema.field("frame_index").type)
    assert pa.types.is_int64(schema.field("episode_index").type)
    assert pa.types.is_int64(schema.field("index").type)
    assert pa.types.is_int64(schema.field("task_index").type)
    assert pa.types.is_float64(schema.field("capture_timestamp").type)
    assert schema.field("intervention").type == pa.bool_()
    assert pa.types.is_int64(schema.field("control_mode").type)


def test_features_describe_required_v21_fields():
    assert FEATURES["observation.state"]["shape"] == [16]
    assert FEATURES["action"]["shape"] == [16]
    for key in [
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]:
        assert FEATURES[key]["dtype"] == "video"
        assert FEATURES[key]["names"] == ["height", "width", "channels"]
        assert "info" in FEATURES[key]
        assert "video_info" not in FEATURES[key]
    for key in [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "capture_timestamp",
        "intervention",
        "control_mode",
    ]:
        assert FEATURES[key]["shape"] == [1]
        assert FEATURES[key]["names"] is None


def test_write_episode_creates_v21_layout(tmp_path):
    LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil").write_episode(_sealed_episode())

    assert (tmp_path / "data/chunk-000/episode_000000.parquet").exists()
    assert (tmp_path / "videos/chunk-000/observation.images.top/episode_000000.mp4").exists()
    assert (tmp_path / "videos/chunk-000/observation.images.left_wrist/episode_000000.mp4").exists()
    assert (tmp_path / "videos/chunk-000/observation.images.right_wrist/episode_000000.mp4").exists()
    assert json.loads((tmp_path / "meta/info.json").read_text())["codebase_version"] == "v2.1"
    assert len(pd.read_parquet(tmp_path / "data/chunk-000/episode_000000.parquet")) == 3


def test_write_episode_streams_spilled_frames_and_cleans_tmp(tmp_path):
    source = _sealed_episode(index=0, task="pick cup", frame_count=5)
    buffer = EpisodeBuffer(
        episode_index=0,
        task="pick cup",
        spill_chunk_size=2,
        spill_root=tmp_path / "spill",
    )
    for frame in source.frames:
        buffer.append(frame)
    sealed = buffer.seal()
    assert isinstance(sealed.frames, SpilledFrameSequence)
    chunk_paths = list(sealed.frames.chunk_paths)
    assert all(path.exists() for path in chunk_paths)

    LeRobotV21Writer(root=tmp_path / "dataset", fps=30, robot_type="pico_hil").write_episode(sealed)

    parquet_path = tmp_path / "dataset/data/chunk-000/episode_000000.parquet"
    assert len(pd.read_parquet(parquet_path)) == 5
    assert not any(path.exists() for path in chunk_paths)


def test_parquet_omits_images_and_metadata_is_consistent(tmp_path):
    LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil").write_episode(
        _sealed_episode(index=7, task="pick cup")
    )

    parquet_path = tmp_path / "data/chunk-000/episode_000000.parquet"
    _assert_v21_parquet_schema(parquet_path)
    df = pd.read_parquet(parquet_path)

    assert set(df.columns) == {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "capture_timestamp",
        "intervention",
        "control_mode",
    }
    assert "images" not in df.columns
    assert not any(column.startswith("observation.images") for column in df.columns)
    assert list(df.loc[0, "observation.state"]) == list(np.arange(16, dtype=np.float32))
    assert list(df.loc[0, "action"]) == list(np.arange(100, 116, dtype=np.float32))
    assert df["episode_index"].tolist() == [0, 0, 0]
    assert df["frame_index"].tolist() == [0, 1, 2]
    assert df["index"].tolist() == [0, 1, 2]
    assert df["timestamp"].tolist() == pytest.approx([0.0, 1.0 / 30.0, 2.0 / 30.0])
    assert df["capture_timestamp"].tolist() == pytest.approx([1.0, 1.0 + 1.0 / 30.0, 1.0 + 2.0 / 30.0])
    assert df["task_index"].tolist() == [0, 0, 0]
    assert df["intervention"].tolist() == [False, True, False]
    assert df["control_mode"].tolist() == [1, 5, 1]

    assert _jsonl(tmp_path / "meta/tasks.jsonl") == [{"task_index": 0, "task": "pick cup"}]
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == [
        {
            "episode_index": 0,
            "tasks": ["pick cup"],
            "length": 3,
            "metadata": {"source": "test"},
        }
    ]
    episode_stats = _jsonl(tmp_path / "meta/episodes_stats.jsonl")
    assert episode_stats[0]["episode_index"] == 0
    stats = episode_stats[0]["stats"]
    assert set(stats) == set(_json(tmp_path / "meta/info.json")["features"])
    for value in stats.values():
        assert set(value) == {"min", "max", "mean", "std", "count"}
    assert stats["observation.state"]["count"] == [3]
    assert stats["observation.state"]["min"] == pytest.approx(np.arange(16, dtype=np.float32).tolist())
    assert stats["observation.state"]["max"] == pytest.approx((np.arange(16) + 2).tolist())


def test_info_contains_complete_v21_dataset_and_video_metadata(tmp_path):
    LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil").write_episode(_sealed_episode())

    info = _json(tmp_path / "meta/info.json")
    assert info["codebase_version"] == "v2.1"
    assert info["robot_type"] == "pico_hil"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3
    assert info["total_tasks"] == 1
    assert info["total_videos"] == 3
    assert info["total_chunks"] == 1
    assert info["chunks_size"] == 1000
    assert info["splits"] == {"train": "0:1"}
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["video_path"] == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    for key in ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"]:
        feature = info["features"][key]
        assert feature["shape"] == [4, 6, 3]
        assert feature["names"] == ["height", "width", "channels"]
        assert feature["info"] == {
            "video.height": 4,
            "video.width": 6,
            "video.fps": 30,
            "video.codec": "mpeg4",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.channels": 3,
            "has_audio": False,
        }


def test_global_index_continues_across_episodes(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first", frame_count=2))
    writer.write_episode(_sealed_episode(index=1, task="second", frame_count=3))

    first = pd.read_parquet(tmp_path / "data/chunk-000/episode_000000.parquet")
    second = pd.read_parquet(tmp_path / "data/chunk-000/episode_000001.parquet")
    assert first["index"].tolist() == [0, 1]
    assert second["index"].tolist() == [2, 3, 4]


def test_episode_paths_roll_over_at_v21_chunk_boundary(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")

    assert writer._parquet_path(999) == tmp_path / "data/chunk-000/episode_000999.parquet"
    assert writer._parquet_path(1000) == tmp_path / "data/chunk-001/episode_001000.parquet"


def test_append_rejects_incompatible_historical_vector_order(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first"))
    info_path = tmp_path / "meta/info.json"
    info = _json(info_path)
    old_left_first_names = [
        "left_joint_0",
        "left_joint_1",
        "left_joint_2",
        "left_joint_3",
        "left_joint_4",
        "left_joint_5",
        "left_joint_6",
        "left_gripper",
        "right_joint_0",
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_joint_6",
        "right_gripper",
    ]
    info["features"]["observation.state"]["names"] = old_left_first_names
    info["features"]["action"]["names"] = old_left_first_names
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible.*ordering"):
        writer.write_episode(_sealed_episode(index=1, task="second"))

    assert not (tmp_path / "data/chunk-000/episode_000001.parquet").exists()


def test_append_rejects_camera_dimensions_that_differ_from_dataset_contract(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first"))
    sealed = _sealed_episode(index=1, task="second")
    for frame in sealed.frames:
        frame["images"]["top"] = np.zeros((8, 6, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="top.*dimensions"):
        writer.write_episode(sealed)

    assert not (tmp_path / "data/chunk-000/episode_000001.parquet").exists()


def test_append_rejects_legacy_video_feature_metadata(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first"))
    info_path = tmp_path / "meta/info.json"
    info = _json(info_path)
    feature = info["features"]["observation.images.top"]
    feature["names"] = ["height", "width", "channel"]
    feature["video_info"] = feature.pop("info")
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="observation.images.top.*metadata"):
        writer.write_episode(_sealed_episode(index=1, task="second"))

    assert not (tmp_path / "data/chunk-000/episode_000001.parquet").exists()


def test_write_episode_records_teleoperation_frame_ranges(tmp_path):
    sealed = _sealed_episode(frame_count=6)
    for frame, mode in zip(sealed.frames, [1, 5, 5, 1, 5, 1]):
        frame["control_mode"] = mode
        frame["intervention"] = mode == 5

    LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil").write_episode(sealed)

    assert _json(tmp_path / "meta/expert_frame_index.json") == {
        "episodes": [
            {
                "episode_index": 0,
                "segments": [
                    {"start_frame_index": 1, "end_frame_index": 2},
                    {"start_frame_index": 4, "end_frame_index": 4},
                ],
            }
        ]
    }


def test_write_episode_restores_metadata_file_bytes_when_metadata_write_fails(tmp_path, monkeypatch):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first task"))
    original_files = _active_episode_files(tmp_path)
    original_metadata = _metadata_file_bytes(tmp_path)

    def corrupt_then_fail(tasks_by_index, episode_metadata):
        (tmp_path / "meta/episodes.jsonl").write_text("CORRUPTED BEFORE EXCEPTION\n")
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(writer, "_write_metadata", corrupt_then_fail)

    with pytest.raises(RuntimeError, match="metadata write failed"):
        writer.write_episode(_sealed_episode(index=0, task="second task"))

    assert _active_episode_files(tmp_path) == original_files
    assert _metadata_file_bytes(tmp_path) == original_metadata
    assert not (tmp_path / "data/chunk-000/episode_000001.parquet").exists()
    assert not (tmp_path / "videos/chunk-000/observation.images.top/episode_000001.mp4").exists()


def test_rejects_non_uint8_images_without_active_outputs(tmp_path):
    sealed = _sealed_episode()
    sealed.frames[0]["images"]["top"] = sealed.frames[0]["images"]["top"].astype(np.float32) / 255.0
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")

    with pytest.raises(ValueError, match="uint8"):
        writer.write_episode(sealed)

    assert _active_episode_files(tmp_path) == []
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == []


def test_failed_video_write_leaves_no_active_episode_outputs_or_metadata(tmp_path, monkeypatch):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    calls = []

    def fail_after_first_video(path, frames, image_key):
        calls.append(image_key)
        if len(calls) == 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial video")
            return
        raise RuntimeError("video write failed")

    monkeypatch.setattr(writer, "_write_video", fail_after_first_video)

    with pytest.raises(RuntimeError, match="video write failed"):
        writer.write_episode(_sealed_episode())

    assert calls == ["top", "left_wrist"]
    assert _active_episode_files(tmp_path) == []
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == []


def test_rejects_negative_task_index_without_active_outputs(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")

    with pytest.raises(ValueError, match="task_index"):
        writer.write_episode(_sealed_episode(task_index=-1))

    assert _active_episode_files(tmp_path) == []
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == []


def test_delete_missing_episode_raises_file_not_found(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")

    with pytest.raises(FileNotFoundError, match="episode 000000"):
        writer.delete_episode(0)


def test_delete_episode_moves_removed_files_and_renumbers_active_dataset(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first task"))
    writer.write_episode(_sealed_episode(index=0, task="second task"))

    writer.delete_episode(0)

    assert (tmp_path / "data/chunk-000/episode_000000.parquet").exists()
    assert not (tmp_path / "data/chunk-000/episode_000001.parquet").exists()
    assert (tmp_path / "videos/chunk-000/observation.images.top/episode_000000.mp4").exists()
    assert not (tmp_path / "videos/chunk-000/observation.images.top/episode_000001.mp4").exists()
    assert (tmp_path / "meta/deleted/episode_000000/data/chunk-000/episode_000000.parquet").exists()
    assert (
        tmp_path / "meta/deleted/episode_000000/videos/chunk-000/observation.images.top/episode_000000.mp4"
    ).exists()

    episodes = _jsonl(tmp_path / "meta/episodes.jsonl")
    assert len(episodes) == 1
    assert episodes[0]["episode_index"] == 0
    assert episodes[0]["tasks"] == ["second task"]
    assert episodes[0]["length"] == 3

    parquet_path = tmp_path / "data/chunk-000/episode_000000.parquet"
    _assert_v21_parquet_schema(parquet_path)
    df = pd.read_parquet(parquet_path)
    assert df["episode_index"].tolist() == [0, 0, 0]
    assert df["index"].tolist() == [0, 1, 2]
    assert _json(tmp_path / "meta/expert_frame_index.json") == {
        "episodes": [
            {
                "episode_index": 0,
                "segments": [{"start_frame_index": 1, "end_frame_index": 1}],
            }
        ]
    }


def test_delete_episode_rolls_back_when_metadata_write_fails(tmp_path, monkeypatch):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first task"))
    writer.write_episode(_sealed_episode(index=0, task="second task"))
    original_files = _active_episode_files(tmp_path)
    original_episodes = _jsonl(tmp_path / "meta/episodes.jsonl")

    def fail_metadata_write(tasks_by_index, episode_metadata):
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(writer, "_write_metadata", fail_metadata_write)

    with pytest.raises(RuntimeError, match="metadata write failed"):
        writer.delete_episode(0)

    assert _active_episode_files(tmp_path) == original_files
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == original_episodes
    assert not (tmp_path / "meta/deleted/episode_000000/data/chunk-000/episode_000000.parquet").exists()

    first_df = pd.read_parquet(tmp_path / "data/chunk-000/episode_000000.parquet")
    second_df = pd.read_parquet(tmp_path / "data/chunk-000/episode_000001.parquet")
    assert first_df["episode_index"].tolist() == [0, 0, 0]
    assert second_df["episode_index"].tolist() == [1, 1, 1]


def test_repeated_delete_uses_new_deleted_directory_on_collision(tmp_path):
    writer = LeRobotV21Writer(root=tmp_path, fps=30, robot_type="pico_hil")
    writer.write_episode(_sealed_episode(index=0, task="first task"))
    writer.delete_episode(0)
    writer.write_episode(_sealed_episode(index=0, task="second task"))

    writer.delete_episode(0)

    assert (tmp_path / "meta/deleted/episode_000000/data/chunk-000/episode_000000.parquet").exists()
    assert (tmp_path / "meta/deleted/episode_000000_001/data/chunk-000/episode_000000.parquet").exists()
    assert _active_episode_files(tmp_path) == []
    assert _jsonl(tmp_path / "meta/episodes.jsonl") == []
