from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .models import LABEL_DONE, TASK_COMPLETED
from .tasks import normalize_frame_labels

EXPORT_FIELDS = (
    "dataset_id",
    "episode_index",
    "start_frame",
    "end_frame",
    "frame_labels",
    "done",
    "done_frame_count",
    "created_at",
)


@dataclass(frozen=True)
class ExportResult:
    jsonl_path: Path
    csv_path: Path
    count: int


def _annotation_rows(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
            t.dataset_id,
            t.episode_index,
            t.start_frame,
            t.end_frame,
            a.labels_json,
            a.done,
            a.created_at
        FROM annotations AS a
        JOIN tasks AS t ON t.id = a.task_id
        WHERE t.status = ?
        ORDER BY t.id
        """,
        (TASK_COMPLETED,),
    ).fetchall()
    annotation_rows: list[dict[str, object]] = []
    for row in rows:
        frame_labels = normalize_frame_labels(
            json.loads(str(row["labels_json"])),
            expected_count=int(row["end_frame"]) - int(row["start_frame"]),
        )
        annotation_rows.append(
            {
                "dataset_id": int(row["dataset_id"]),
                "episode_index": int(row["episode_index"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "frame_labels": frame_labels,
                "done": bool(row["done"]),
                "done_frame_count": sum(1 for label in frame_labels if label == LABEL_DONE),
                "created_at": str(row["created_at"]),
            }
        )
    return annotation_rows


def export_annotations(conn: sqlite3.Connection, out_dir: Path) -> ExportResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "annotations.jsonl"
    csv_path = out_dir / "annotations.csv"
    rows = _annotation_rows(conn)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["frame_labels"] = json.dumps(row["frame_labels"], separators=(",", ":"))
            csv_row["done"] = "true" if row["done"] else "false"
            writer.writerow(csv_row)

    return ExportResult(jsonl_path=jsonl_path, csv_path=csv_path, count=len(rows))
