import numpy as np
import pytest

from hil_pico_collection.recording.episode_buffer import SealedEpisode
from hil_pico_collection.recording.v21_writer import LeRobotV21Writer
from hil_pico_collection.replay.dataset import ReplayDataset


def rgb(value):
    return np.full((4, 6, 3), value, np.uint8)


def episode(frame_count=2):
    frames = []
    for index in range(frame_count):
        frames.append(
            {
                "observation.state": np.arange(16, dtype=np.float32) + index,
                "action": np.arange(16, dtype=np.float32) + index,
                "images": {key: rgb(index) for key in ("top", "left_wrist", "right_wrist")},
                "timestamp": index / 30,
                "frame_index": index,
                "episode_index": 0,
                "task": "fold",
                "intervention": index == 1,
                "control_mode": 5 if index == 1 else 1,
            }
        )
    return SealedEpisode(episode_index=0, task="fold", frames=frames)


def test_dataset_lists_saved_episode_and_videos(tmp_path):
    writer = LeRobotV21Writer(tmp_path)
    writer.write_episode(episode())
    dataset = ReplayDataset(tmp_path)
    summary = dataset.list_episodes()[0]
    assert summary["episode_index"] == 0
    assert summary["task"] == "fold"
    assert summary["frame_count"] == 2
    assert all(item["exists"] for item in summary["videos"].values())


def test_episode_summary_rejects_missing_episode(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayDataset(tmp_path).episode_summary(9)


@pytest.mark.parametrize("key", ["bad", "", "../top"])
def test_video_path_rejects_unknown_key(tmp_path, key):
    with pytest.raises(ValueError):
        ReplayDataset(tmp_path).video_path(0, key)


def test_replay_module_is_read_only_dataset_browser():
    import hil_pico_collection.replay.dataset as module

    assert not hasattr(module, "command_fields_from_vector")
    assert not hasattr(module, "normalize_replay_source")
