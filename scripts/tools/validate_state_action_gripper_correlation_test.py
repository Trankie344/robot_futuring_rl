import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.tools.validate_state_action_gripper_correlation import validate_dataset


def _write_dataset(root: Path, *, swapped: bool) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 200}) + "\n")

    rng = np.random.default_rng(7)
    left = rng.normal(size=200)
    right = rng.normal(size=200)
    state = np.zeros((200, 16), dtype=np.float32)
    action = np.zeros((200, 16), dtype=np.float32)
    state[:, 7] = left
    state[:, 15] = right
    if swapped:
        action[:, 7] = right
        action[:, 15] = left
    else:
        action[:, 7] = left
        action[:, 15] = right

    df = pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "episode_index": np.zeros(200, dtype=np.int64),
            "frame_index": np.arange(200, dtype=np.int64),
            "index": np.arange(200, dtype=np.int64),
        }
    )
    df.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet", index=False)


def test_validate_dataset_passes_when_direct_mapping_is_correlated(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, swapped=False)

    report = validate_dataset(dataset, state_indices=(7, 15), action_indices=(7, 15), threshold=0.9, margin=0.2)

    assert report["passed"] is True
    assert report["direct_mean_abs_corr"] > 0.99
    assert report["crossed_mean_abs_corr"] < 0.2


def test_validate_dataset_fails_when_action_grippers_are_crossed(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, swapped=True)

    with pytest.raises(SystemExit):
        validate_dataset(dataset, state_indices=(7, 15), action_indices=(7, 15), threshold=0.9, margin=0.2)
