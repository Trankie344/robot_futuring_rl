import json

import pyarrow.parquet as pq

from scripts.tools.compute_arm_progress_from_tristate import build_progress_rows
from scripts.tools.compute_arm_progress_from_tristate import load_episode_lengths
from scripts.tools.compute_arm_progress_from_tristate import write_progress_parquet


def test_compute_arm_progress_from_tristate_writes_progress_and_report(tmp_path):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "length": 4}),
                json.dumps({"episode_index": 1, "length": 3}),
            ]
        )
        + "\n"
    )
    labels = {
        "version": 2,
        "label_semantics": "per_frame_state_for_robot_progress_and_done",
        "states": [-1, 0, 1, "done"],
        "datasets": [
            {
                "dataset_id": "demo",
                "dataset_name": "demo",
                "root_path": "/unused",
                "episodes": [
                    {"episode_index": 0, "frame_states": [0, 1, 1, "done"], "segments": [], "done_frames": [3]},
                    {"episode_index": 1, "frame_states": [None, -1, 1], "segments": [], "done_frames": []},
                ],
            }
        ],
    }
    labels_path = tmp_path / "tristate_labels.json"
    labels_path.write_text(json.dumps(labels))

    episode_lengths = load_episode_lengths(dataset)
    rows, report = build_progress_rows(episode_lengths, labels_path, dataset_index=0)
    output_path = tmp_path / "progress.parquet"
    write_progress_parquet(output_path, rows, report)

    table = pq.read_table(output_path).to_pydict()

    assert table["index"] == [0, 1, 2, 3, 4, 5, 6]
    assert table["episode_index"] == [0, 0, 0, 0, 1, 1, 1]
    assert table["valid_label"] == [True, True, True, True, False, True, True]
    assert table["progress"][0] == 0.0
    assert table["progress"][2] == 1.0
    assert table["progress"][3] == 1.0
    assert table["progress"][4] is None
    assert report["matched_episodes"] == 2
    assert report["invalid_frames"] == 1
