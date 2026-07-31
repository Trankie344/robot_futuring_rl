from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .models import DatasetInfo, LABEL_DONE, TASK_COMPLETED
from .tasks import normalize_frame_labels


def write_tristate_metadata(conn: sqlite3.Connection, datasets_by_id: dict[int, DatasetInfo]) -> list[Path]:
    rows = conn.execute(
        """
        SELECT
            t.id AS task_id,
            t.dataset_id,
            t.episode_index,
            t.start_frame,
            t.end_frame,
            a.labels_json,
            a.created_at
        FROM annotations AS a
        JOIN tasks AS t ON t.id = a.task_id
        WHERE t.status = ?
        ORDER BY t.dataset_id, t.episode_index, t.start_frame, t.id
        """,
        (TASK_COMPLETED,),
    ).fetchall()

    written: list[Path] = []
    for dataset_id, dataset in datasets_by_id.items():
        dataset_rows = [row for row in rows if int(row["dataset_id"]) == dataset_id]
        payload = _metadata_payload(dataset_id, dataset, dataset_rows)
        ready_path = dataset.root / "meta" / "tristate_labels.json"
        partial_path = dataset.root / "meta" / "tristate_labels.partial.json"
        if payload["ready"]:
            _atomic_write_json(ready_path, payload)
            partial_path.unlink(missing_ok=True)
            written.append(ready_path)
        else:
            ready_path.unlink(missing_ok=True)
            _atomic_write_json(partial_path, payload)
            written.append(partial_path)
    return written


def _metadata_payload(dataset_id: int, dataset: DatasetInfo, rows: list[sqlite3.Row]) -> dict[str, object]:
    rows_by_episode: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)

    episodes = []
    for episode in dataset.episodes:
        frame_states = [None] * episode.length
        for row in rows_by_episode.get(episode.episode_index, []):
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            expected = end_frame - start_frame
            labels = normalize_frame_labels(json.loads(str(row["labels_json"])), expected_count=expected)
            for offset, label in enumerate(labels):
                frame_index = start_frame + offset
                if 0 <= frame_index < episode.length:
                    frame_states[frame_index] = label

        completion_frames = [index for index, label in enumerate(frame_states) if label == LABEL_DONE]
        if len(completion_frames) > 1:
            raise ValueError(f"Episode {episode.episode_index} has more than one completion label 2")
        if completion_frames and completion_frames[0] != episode.length - 1:
            raise ValueError(
                f"Episode {episode.episode_index} completion label 2 must be at final frame "
                f"{episode.length - 1}, got {completion_frames[0]}"
            )

        episodes.append(
            {
                "episode_index": episode.episode_index,
                "frame_states": frame_states,
                "segments": _segments_from_frame_states(frame_states),
                "done_frames": completion_frames,
            }
        )

    ready = all(all(label is not None for label in episode["frame_states"]) for episode in episodes)
    return {
        "version": 3,
        "ready": ready,
        "label_semantics": "per_frame_state_with_terminal_completion",
        "states": [-1, 0, 1, LABEL_DONE],
        "datasets": [
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "root_path": str(dataset.root),
                "episodes": episodes,
            }
        ],
    }


def _segments_from_frame_states(frame_states: list[object]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    start = None
    current = None
    for index, label in enumerate(frame_states):
        if label is None:
            if current is not None and start is not None:
                segments.append({"start_frame": start, "end_frame": index, "state": current})
            start = None
            current = None
            continue
        if current is None:
            start = index
            current = label
            continue
        if label != current:
            segments.append({"start_frame": start, "end_frame": index, "state": current})
            start = index
            current = label
    if current is not None and start is not None:
        segments.append({"start_frame": start, "end_frame": len(frame_states), "state": current})
    return segments


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
