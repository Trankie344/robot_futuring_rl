from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


ANNOTATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL UNIQUE REFERENCES tasks(id),
    labels_json TEXT NOT NULL,
    done INTEGER NOT NULL CHECK(done IN (0, 1)),
    device_id TEXT,
    nickname TEXT,
    created_at TEXT NOT NULL
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            root_path TEXT NOT NULL UNIQUE,
            fps INTEGER NOT NULL,
            total_episodes INTEGER NOT NULL,
            total_frames INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id),
            episode_index INTEGER NOT NULL,
            start_frame INTEGER NOT NULL,
            end_frame INTEGER NOT NULL,
            stride_source INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'locked', 'completed', 'review')),
            locked_by TEXT,
            locked_until TEXT,
            skip_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(dataset_id, episode_index, start_frame, end_frame)
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(dataset_id, status, locked_until, id);
        {ANNOTATIONS_TABLE_SQL}
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            task_id INTEGER REFERENCES tasks(id),
            event_type TEXT NOT NULL,
            device_id TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    _migrate_annotations_table(conn)


def _migrate_annotations_table(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(annotations)").fetchall()}
    if "label" not in columns or "labels_json" in columns:
        return

    rows = conn.execute(
        """
        SELECT id, task_id, label, done, device_id, nickname, created_at
        FROM annotations
        ORDER BY id
        """
    ).fetchall()
    conn.execute("ALTER TABLE annotations RENAME TO annotations_legacy")
    conn.executescript(ANNOTATIONS_TABLE_SQL)
    for row in rows:
        labels_json = json.dumps([int(row["label"])] * 4, separators=(",", ":"))
        conn.execute(
            """
            INSERT INTO annotations(id, task_id, labels_json, done, device_id, nickname, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["id"]),
                int(row["task_id"]),
                labels_json,
                int(row["done"]),
                row["device_id"],
                row["nickname"],
                str(row["created_at"]),
            ),
        )
    conn.execute("DROP TABLE annotations_legacy")
