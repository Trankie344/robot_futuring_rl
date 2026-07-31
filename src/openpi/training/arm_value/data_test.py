import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.models.arm_value.config import ArmValueModelConfig
from openpi.training.arm_value.config import ArmValueDataConfig
from openpi.training.arm_value.data import ArmProgressTable
from openpi.training.arm_value.data import ArmValueDataset
from openpi.training.arm_value.data import ProgressRecord
from openpi.training.arm_value.data import load_state_norm_stats


def test_progress_windows_clamp_to_episode_and_create_tristate_targets():
    table = ArmProgressTable(
        [
            ProgressRecord(0, 0, 0, 0.0),
            ProgressRecord(1, 0, 1, 0.0),
            ProgressRecord(2, 0, 2, 0.5),
            ProgressRecord(3, 0, 3, 0.25),
            ProgressRecord(4, 0, 4, 0.25),
        ],
        interval_eps=1e-3,
    )
    indices = table.window_indices(3, n_history_steps=2, frame_gap=2)
    assert indices == [0, 1, 3]
    progress = table.progress_sequence(indices, episode_index=0)
    np.testing.assert_array_equal(table.interval_targets(progress), np.asarray([0, 1]))

    consecutive = table.progress_sequence([1, 2, 3, 4], episode_index=0)
    np.testing.assert_array_equal(table.interval_targets(consecutive), np.asarray([1, -1, 0]))


def test_progress_parquet_filters_dataset_and_invalid_rows(tmp_path):
    path = tmp_path / "progress.parquet"
    pq.write_table(
        pa.table(
            {
                "index": [0, 1, 2, 0],
                "dataset_index": [0, 0, 0, 1],
                "episode_index": [0, 0, 0, 0],
                "frame_index": [0, 1, 2, 0],
                "progress": [0.0, None, 1.0, 0.5],
                "valid_label": [True, False, True, True],
            }
        ),
        path,
    )
    table = ArmProgressTable.from_parquet(path, dataset_index=0, interval_eps=1e-3)
    assert table.valid_indices(n_history_steps=1, frame_gap=1) == [0]
    summary = table.summary(n_history_steps=1, frame_gap=1)
    assert summary["filtered_rows"] == 2


def test_progress_parquet_requires_frame_index_and_bounded_progress(tmp_path):
    missing_frame_path = tmp_path / "missing_frame.parquet"
    pq.write_table(
        pa.table({"index": [0], "episode_index": [0], "progress": [0.0]}),
        missing_frame_path,
    )
    with pytest.raises(ValueError, match="frame_index"):
        ArmProgressTable.from_parquet(missing_frame_path, dataset_index=0, interval_eps=1e-3)

    invalid_progress_path = tmp_path / "invalid_progress.parquet"
    pq.write_table(
        pa.table(
            {
                "index": [0],
                "episode_index": [0],
                "frame_index": [0],
                "progress": [1.5],
            }
        ),
        invalid_progress_path,
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ArmProgressTable.from_parquet(invalid_progress_path, dataset_index=0, interval_eps=1e-3)


def test_state_norm_stats_support_openpi_format_and_validate_shape(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {"state": {"mean": [1.0, 2.0], "std": [0.5, 0.0]}}}))
    mean, std = load_state_norm_stats(path, state_dim=2)
    np.testing.assert_array_equal(mean, np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(std, np.asarray([0.5, 1e-6], dtype=np.float32))
    with pytest.raises(ValueError, match="must have shape"):
        load_state_norm_stats(path, state_dim=3)


@pytest.mark.parametrize(
    ("sample_episode", "sample_frame", "error"),
    [(1, 0, "Episode mismatch"), (0, 1, "Frame mismatch")],
)
def test_dataset_rejects_sidecar_alignment_mismatches(sample_episode, sample_frame, error):
    dataset = ArmValueDataset.__new__(ArmValueDataset)
    dataset.data_config = ArmValueDataConfig(repo_id="fake", progress_path="", norm_stats_path="")
    dataset.model_config = ArmValueModelConfig(
        clip_pretrained_path="__debug__",
        n_history_steps=1,
        frame_gap=1,
        max_state_dim=2,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
    )
    dataset.indices = [0]
    dataset.dataset = [{"episode_index": sample_episode, "frame_index": sample_frame}]
    dataset.progress_table = ArmProgressTable([ProgressRecord(0, 0, 0, 0.0)], interval_eps=1e-3)
    with pytest.raises(ValueError, match=error):
        dataset[0]
