import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.arm_awbc.weighting import ArmRABCWeighter


def test_arm_rabc_weighter_computes_weights_from_progress(tmp_path):
    progress_path = tmp_path / "progress.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "index": [0, 1, 2, 3],
                "dataset_index": [0, 0, 0, 0],
                "episode_index": [7, 7, 7, 7],
                "frame_index": [0, 1, 2, 3],
                "episode_length": [4, 4, 4, 4],
                "progress": [0.0, 0.0, 0.5, 1.0],
            }
        ),
        progress_path,
    )

    weighter = ArmRABCWeighter(progress_path, chunk_size=1, kappa=0.01, fallback_weight=0.0)

    assert 0.0 < weighter.compute_weight(0) < 1.0
    assert weighter.compute_weight(1) == 1.0
    assert weighter.compute_weight(2) == 1.0
    assert weighter.compute_weight(99) == 0.0
