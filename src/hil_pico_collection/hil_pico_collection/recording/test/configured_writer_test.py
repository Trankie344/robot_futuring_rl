import json

import numpy as np
import pandas as pd

from hil_pico_collection.recording.episode_buffer import SealedEpisode
from hil_pico_collection.recording.v21_writer import LeRobotV21Writer
from hil_pico_collection.replay.dataset import ReplayDataset


def test_writer_and_replay_support_configured_vectors_and_images(tmp_path):
    writer = LeRobotV21Writer(
        tmp_path,
        state_names=["joint_b", "joint_a", "gripper"],
        action_names=["target", "gripper_target"],
        image_shapes={"front_rgb": (4, 6, 3)},
    )
    frame = {
        "observation.state": np.asarray([2, 1, 3], np.float32),
        "action": np.asarray([4, 5], np.float32),
        "images": {"front_rgb": np.zeros((4, 6, 3), np.uint8)},
        "timestamp": 0.0,
        "intervention": False,
        "control_mode": 1,
    }
    writer.write_episode(SealedEpisode(0, "task", [frame]))

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["names"] == ["joint_b", "joint_a", "gripper"]
    assert info["features"]["action"]["names"] == ["target", "gripper_target"]
    assert info["features"]["observation.images.front_rgb"]["shape"] == [4, 6, 3]
    table = pd.read_parquet(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    assert len(table["observation.state"].iloc[0]) == 3
    assert len(table["action"].iloc[0]) == 2
    replay = ReplayDataset(tmp_path)
    assert replay.video_keys == ("front_rgb",)
    assert replay.video_path(0, "front_rgb").exists()


def test_writer_rejects_image_shape_that_differs_from_config(tmp_path):
    writer = LeRobotV21Writer(
        tmp_path,
        state_names=["state"],
        action_names=["action"],
        image_shapes={"front": (4, 6, 3)},
    )
    frame = {
        "observation.state": np.zeros(1, np.float32),
        "action": np.zeros(1, np.float32),
        "images": {"front": np.zeros((8, 8, 3), np.uint8)},
    }
    try:
        writer.write_episode(SealedEpisode(0, "task", [frame]))
    except ValueError as exc:
        assert "configured shape" in str(exc)
    else:
        raise AssertionError("writer accepted a mismatched configured image shape")
