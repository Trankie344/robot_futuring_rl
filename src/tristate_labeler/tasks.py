from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone

from .database import utc_now
from .models import (
    DatasetInfo,
    EpisodeInfo,
    INTERVAL_COUNT,
    LABEL_DONE,
    TASK_COMPLETED,
    TASK_LOCKED,
    TASK_PENDING,
    TASK_REVIEW,
    VALID_FRAME_STATES,
    VALID_LABELS,
    WINDOW_FRAMES,
)

LOCK_SECONDS = 300
SKIP_REVIEW_THRESHOLD = 2


def _lock_until() -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=LOCK_SECONDS))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _task_row(conn: sqlite3.Connection, task_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown task: {task_id}")
    return row


def _normalize_dataset_ids(dataset_ids: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(dataset_ids, int):
        return (dataset_ids,)
    normalized = tuple(int(dataset_id) for dataset_id in dataset_ids)
    if not normalized:
        raise ValueError("At least one dataset_id is required")
    return normalized


def _dataset_id_clause(dataset_ids: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    placeholders = ", ".join("?" for _ in dataset_ids)
    return f"dataset_id IN ({placeholders})", dataset_ids


def _validate_locked_owner(row: sqlite3.Row, device_id: str) -> None:
    if row["locked_by"] != device_id:
        raise PermissionError("Task is locked by another device")
    if row["status"] != TASK_LOCKED:
        raise ValueError(f"Task is not locked: {row['status']}")


def _log_event(
    conn: sqlite3.Connection,
    *,
    task_id: int,
    event_type: str,
    device_id: str,
    payload: dict[str, object],
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO events(task_id, event_type, device_id, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (task_id, event_type, device_id, json.dumps(payload, sort_keys=True), created_at),
    )


def ensure_dataset_row(conn: sqlite3.Connection, dataset: DatasetInfo) -> int:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO datasets(name, root_path, fps, total_episodes, total_frames, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(root_path) DO UPDATE SET
            name=excluded.name,
            fps=excluded.fps,
            total_episodes=excluded.total_episodes,
            total_frames=excluded.total_frames
        """,
        (dataset.name, str(dataset.root), dataset.fps, dataset.total_episodes, dataset.total_frames, now),
    )
    row = conn.execute("SELECT id FROM datasets WHERE root_path = ?", (str(dataset.root),)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to create dataset row for {dataset.root}")
    return int(row["id"])


def generate_tasks(
    conn: sqlite3.Connection,
    dataset_id: int,
    episodes: Sequence[EpisodeInfo],
    stride: int,
    window_frames: int = WINDOW_FRAMES,
) -> int:
    if stride not in {30, 15, 10, 5}:
        raise ValueError(f"Unsupported stride: {stride}")

    inserted = 0
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for episode in episodes:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO tasks(
                    dataset_id, episode_index, start_frame, end_frame, stride_source,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    episode.episode_index,
                    0,
                    episode.length,
                    stride,
                    TASK_PENDING,
                    now,
                    now,
                ),
            )
            inserted += cursor.rowcount
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return inserted


def progress_counts(conn: sqlite3.Connection, dataset_id: int | Iterable[int]) -> dict[str, int]:
    counts = {
        TASK_PENDING: 0,
        TASK_LOCKED: 0,
        TASK_COMPLETED: 0,
        TASK_REVIEW: 0,
    }
    dataset_ids = _normalize_dataset_ids(dataset_id)
    where_clause, params = _dataset_id_clause(dataset_ids)
    rows = conn.execute(
        f"""
        SELECT status, COUNT(*) AS count
        FROM tasks
        WHERE {where_clause}
        GROUP BY status
        """,
        params,
    ).fetchall()
    for row in rows:
        counts[str(row["status"])] = int(row["count"])
    return counts


def claim_next_task(
    conn: sqlite3.Connection,
    dataset_id: int | Iterable[int],
    device_id: str,
    nickname: str | None = None,
    exclude_task_id: int | None = None,
) -> sqlite3.Row | None:
    dataset_ids = _normalize_dataset_ids(dataset_id)
    where_clause, dataset_params = _dataset_id_clause(dataset_ids)
    now = utc_now()
    lock_until = _lock_until()

    def select_task(excluded_task_id: int | None) -> sqlite3.Row | None:
        exclude_clause = ""
        params: list[object] = list(dataset_params)
        if excluded_task_id is not None:
            exclude_clause = "AND id != ?"
            params.append(excluded_task_id)
        params.extend((TASK_PENDING, TASK_LOCKED, now))
        return conn.execute(
            f"""
            SELECT *
            FROM tasks
            WHERE {where_clause}
              {exclude_clause}
              AND (
                  status = ?
                  OR (status = ? AND (locked_until IS NULL OR locked_until < ?))
              )
            ORDER BY id
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()

    conn.execute("BEGIN IMMEDIATE")
    try:
        task = select_task(exclude_task_id)
        if task is None and exclude_task_id is not None:
            task = select_task(None)
        if task is None:
            conn.execute("COMMIT")
            return None

        conn.execute(
            """
            UPDATE tasks
            SET status = ?, locked_by = ?, locked_until = ?, updated_at = ?
            WHERE id = ?
            """,
            (TASK_LOCKED, device_id, lock_until, now, task["id"]),
        )
        _log_event(
            conn,
            task_id=task["id"],
            event_type="claim",
            device_id=device_id,
            payload={"nickname": nickname, "locked_until": lock_until},
            created_at=now,
        )
        claimed = _task_row(conn, task["id"])
        conn.execute("COMMIT")
        return claimed
    except Exception:
        conn.execute("ROLLBACK")
        raise


def heartbeat_task(conn: sqlite3.Connection, task_id: int, device_id: str) -> sqlite3.Row:
    now = utc_now()
    lock_until = _lock_until()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _task_row(conn, task_id)
        _validate_locked_owner(row, device_id)
        conn.execute(
            "UPDATE tasks SET locked_until = ?, updated_at = ? WHERE id = ?",
            (lock_until, now, task_id),
        )
        _log_event(
            conn,
            task_id=task_id,
            event_type="heartbeat",
            device_id=device_id,
            payload={"locked_until": lock_until},
            created_at=now,
        )
        updated = _task_row(conn, task_id)
        conn.execute("COMMIT")
        return updated
    except Exception:
        conn.execute("ROLLBACK")
        raise


def submit_task(
    conn: sqlite3.Connection,
    task_id: int,
    device_id: str,
    frame_labels: Sequence[int],
    nickname: str | None = None,
) -> sqlite3.Row:
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _task_row(conn, task_id)
        _validate_locked_owner(row, device_id)
        normalized_labels = normalize_frame_labels(
            frame_labels,
            expected_count=int(row["end_frame"]) - int(row["start_frame"]),
        )
        done = LABEL_DONE in normalized_labels
        labels_json = json.dumps(normalized_labels, separators=(",", ":"))
        conn.execute(
            """
            INSERT INTO annotations(task_id, labels_json, done, device_id, nickname, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_id, labels_json, int(done), device_id, nickname, now),
        )
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, locked_by = NULL, locked_until = NULL, updated_at = ?
            WHERE id = ?
            """,
            (TASK_COMPLETED, now, task_id),
        )
        _log_event(
            conn,
            task_id=task_id,
            event_type="submit",
            device_id=device_id,
            payload={
                "done": bool(done),
                "frame_count": len(normalized_labels),
                "done_frame_count": sum(1 for label in normalized_labels if label == LABEL_DONE),
                "nickname": nickname,
            },
            created_at=now,
        )
        updated = _task_row(conn, task_id)
        conn.execute("COMMIT")
        return updated
    except Exception:
        conn.execute("ROLLBACK")
        raise


def normalize_frame_labels(labels: Sequence[int], expected_count: int) -> list[int]:
    normalized: list[int] = []
    if len(labels) != expected_count:
        raise ValueError(f"Expected {expected_count} frame labels, got {len(labels)}")

    for frame_index, label in enumerate(labels):
        if type(label) is not int:
            raise ValueError(f"Invalid label at frame {frame_index}: {label!r}")
        if label not in VALID_FRAME_STATES:
            raise ValueError(f"Invalid label at frame {frame_index}: {label!r}")
        normalized.append(label)

    completion_frames = [index for index, label in enumerate(normalized) if label == LABEL_DONE]
    if len(completion_frames) > 1:
        raise ValueError("Completion label 2 may appear at most once per episode")
    if completion_frames and completion_frames[0] != expected_count - 1:
        raise ValueError(
            f"Completion label 2 may only appear at the final frame {expected_count - 1}, "
            f"got frame {completion_frames[0]}"
        )
    return normalized


def _normalize_frame_labels(labels: Sequence[int], expected_count: int) -> list[int]:
    """Backward-compatible internal name for the strict integer label validator."""
    return normalize_frame_labels(labels, expected_count)


def skip_task(
    conn: sqlite3.Connection,
    task_id: int,
    device_id: str,
    nickname: str | None = None,
) -> sqlite3.Row:
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _task_row(conn, task_id)
        _validate_locked_owner(row, device_id)
        skip_count = int(row["skip_count"]) + 1
        next_status = TASK_REVIEW if skip_count >= SKIP_REVIEW_THRESHOLD else TASK_PENDING
        conn.execute(
            """
            UPDATE tasks
            SET status = ?,
                locked_by = NULL,
                locked_until = NULL,
                skip_count = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (next_status, skip_count, now, task_id),
        )
        _log_event(
            conn,
            task_id=task_id,
            event_type="skip",
            device_id=device_id,
            payload={"nickname": nickname, "skip_count": skip_count, "status": next_status},
            created_at=now,
        )
        updated = _task_row(conn, task_id)
        conn.execute("COMMIT")
        return updated
    except Exception:
        conn.execute("ROLLBACK")
        raise
