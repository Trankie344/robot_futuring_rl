from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictInt

from .database import connect, init_db
from .dataset import load_dataset_collection
from .exporter import export_annotations
from .models import DatasetInfo, EpisodeInfo, INTERVAL_COUNT, INTERVAL_FRAMES
from .tasks import (
    claim_next_task,
    ensure_dataset_row,
    generate_tasks,
    heartbeat_task,
    progress_counts,
    skip_task,
    submit_task,
)
from .tristate_metadata import write_tristate_metadata

CONTEXT_FRAMES = 0
CONTEXT_SECONDS = 0
PLAYBACK_RATES = [0.25, 0.5, 0.75, 1.0]
DEFAULT_PLAYBACK_RATE = 0.5
STRIDE_OPTIONS = [30, 15, 10, 5]


class GenerateRequest(BaseModel):
    stride: int = 10


class DevicePayload(BaseModel):
    device_id: str
    nickname: str | None = None


class ClaimPayload(DevicePayload):
    exclude_task_id: int | None = None


class SubmitPayload(DevicePayload):
    frame_labels: list[StrictInt]


def create_app(
    dataset_root: Path,
    db_path: Path,
    initial_stride: int = 10,
    export_dir: Path | None = None,
) -> FastAPI:
    datasets = load_dataset_collection(dataset_root)
    export_root = export_dir if export_dir is not None else Path(db_path).parent / "exports"

    dataset_ids: list[int] = []
    with connect(db_path) as conn:
        init_db(conn)
        for dataset in datasets:
            dataset_id = ensure_dataset_row(conn, dataset)
            generate_tasks(conn, dataset_id, dataset.episodes, stride=initial_stride)
            dataset_ids.append(dataset_id)

    dataset_ids_tuple = tuple(dataset_ids)
    datasets_by_id = dict(zip(dataset_ids_tuple, datasets))
    first_dataset = datasets[0]
    first_dataset_id = dataset_ids_tuple[0]
    allowed_video_paths = _video_allowlist(datasets_by_id)

    app = FastAPI(title="Tri-State + Completion Labeler")
    app.state.dataset = first_dataset
    app.state.datasets = datasets
    app.state.dataset_id = first_dataset_id
    app.state.dataset_ids = dataset_ids_tuple
    app.state.datasets_by_id = datasets_by_id
    app.state.allowed_video_paths = allowed_video_paths
    app.state.db_path = Path(db_path)
    app.state.export_dir = Path(export_root)

    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "dataset_ids": list(dataset_ids_tuple),
            "dataset_id": first_dataset_id,
            "dataset_name": first_dataset.name,
            "dataset_count": len(datasets),
            "datasets": [
                {
                    "id": dataset_id,
                    "name": dataset.name,
                    "fps": dataset.fps,
                }
                for dataset_id, dataset in zip(dataset_ids_tuple, datasets)
            ],
            "fps": first_dataset.fps,
            "window_frames": None,
            "window_mode": "full_episode",
            "interval_frames": INTERVAL_FRAMES,
            "interval_count": INTERVAL_COUNT,
            "context_frames": CONTEXT_FRAMES,
            "context_seconds": CONTEXT_SECONDS,
            "video_keys": list(first_dataset.video_keys),
            "stride_options": STRIDE_OPTIONS,
            "playback_rates": PLAYBACK_RATES,
            "default_playback_rate": DEFAULT_PLAYBACK_RATE,
        }

    @app.post("/api/tasks/generate")
    def generate(body: GenerateRequest) -> dict[str, int]:
        try:
            with connect(db_path) as conn:
                inserted = 0
                for current_dataset_id, current_dataset in datasets_by_id.items():
                    inserted += generate_tasks(
                        conn,
                        current_dataset_id,
                        current_dataset.episodes,
                        stride=body.stride,
                    )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"inserted": inserted}

    @app.post("/api/tasks/claim")
    def claim(body: ClaimPayload) -> dict[str, object | None]:
        with connect(db_path) as conn:
            row = claim_next_task(
                conn,
                dataset_ids_tuple,
                device_id=body.device_id,
                nickname=body.nickname,
                exclude_task_id=body.exclude_task_id,
            )
        return {"task": _task_payload(row, datasets_by_id) if row is not None else None}

    @app.post("/api/tasks/{task_id}/heartbeat")
    def heartbeat(task_id: int, body: DevicePayload) -> dict[str, bool]:
        try:
            with connect(db_path) as conn:
                heartbeat_task(conn, task_id=task_id, device_id=body.device_id)
        except (KeyError, PermissionError, ValueError) as exc:
            _raise_task_error(exc)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/submit")
    def submit(task_id: int, body: SubmitPayload) -> dict[str, bool]:
        try:
            with connect(db_path) as conn:
                submit_task(
                    conn,
                    task_id=task_id,
                    device_id=body.device_id,
                    frame_labels=body.frame_labels,
                    nickname=body.nickname,
                )
                write_tristate_metadata(conn, datasets_by_id)
        except (KeyError, PermissionError, ValueError, sqlite3.IntegrityError) as exc:
            _raise_task_error(exc)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/skip")
    def skip(task_id: int, body: DevicePayload) -> dict[str, bool]:
        try:
            with connect(db_path) as conn:
                skip_task(conn, task_id=task_id, device_id=body.device_id, nickname=body.nickname)
        except (KeyError, PermissionError, ValueError) as exc:
            _raise_task_error(exc)
        return {"ok": True}

    @app.get("/api/progress")
    def progress() -> dict[str, int]:
        with connect(db_path) as conn:
            return progress_counts(conn, dataset_ids_tuple)

    @app.post("/api/export")
    def export() -> dict[str, object]:
        with connect(db_path) as conn:
            result = export_annotations(conn, export_root)
        return {
            "jsonl_path": str(result.jsonl_path),
            "csv_path": str(result.csv_path),
            "count": result.count,
        }

    @app.get("/videos/{dataset_id}/{video_path:path}")
    def videos(dataset_id: int, video_path: str) -> FileResponse:
        dataset = datasets_by_id.get(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        requested = (dataset.root / video_path).resolve()
        try:
            relative_path = requested.relative_to(dataset.root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Video path escapes dataset root") from exc
        if (dataset_id, relative_path.as_posix()) not in allowed_video_paths:
            raise HTTPException(status_code=404, detail="Video not found")
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(requested)

    return app


def _video_allowlist(datasets_by_id: dict[int, DatasetInfo]) -> frozenset[tuple[int, str]]:
    return frozenset(
        (dataset_id, video.relative_path.as_posix())
        for dataset_id, dataset in datasets_by_id.items()
        for episode in dataset.episodes
        for video in episode.videos.values()
    )


def _task_payload(row: sqlite3.Row, datasets_by_id: dict[int, DatasetInfo]) -> dict[str, object]:
    dataset_id = int(row["dataset_id"])
    dataset = datasets_by_id[dataset_id]
    episode = _episode_for_task(dataset, int(row["episode_index"]))
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    context_start_frame = max(0, start_frame - CONTEXT_FRAMES)
    label_start_time = start_frame / dataset.fps
    label_end_time = end_frame / dataset.fps
    label_intervals = []
    for index in range(INTERVAL_COUNT):
        interval_start_frame = start_frame + index * INTERVAL_FRAMES
        interval_end_frame = interval_start_frame + INTERVAL_FRAMES
        label_intervals.append(
            {
                "index": index,
                "start_frame": interval_start_frame,
                "end_frame": interval_end_frame,
                "start_time": interval_start_frame / dataset.fps,
                "end_time": interval_end_frame / dataset.fps,
            }
        )
    return {
        "id": int(row["id"]),
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "fps": dataset.fps,
        "episode_index": int(row["episode_index"]),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "context_start_frame": context_start_frame,
        "context_start_time": context_start_frame / dataset.fps,
        "label_start_time": label_start_time,
        "label_end_time": label_end_time,
        "start_time": label_start_time,
        "end_time": label_end_time,
        "frame_count": end_frame - start_frame,
        "label_intervals": label_intervals,
        "expert_segments": _expert_segments_for_task(episode, start_frame, end_frame, dataset.fps),
        "videos": {
            key: f"/videos/{dataset_id}/{quote(video.relative_path.as_posix(), safe='/')}"
            for key, video in episode.videos.items()
        },
    }


def _episode_for_task(dataset: DatasetInfo, episode_index: int) -> EpisodeInfo:
    for episode in dataset.episodes:
        if episode.episode_index == episode_index:
            return episode
    raise KeyError(f"Unknown episode: {episode_index}")


def _expert_segments_for_task(
    episode: EpisodeInfo,
    start_frame: int,
    end_frame: int,
    fps: int,
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for segment in episode.expert_segments:
        clipped_start = max(start_frame, segment.start_frame)
        clipped_end = min(end_frame, segment.end_frame)
        if clipped_start >= clipped_end:
            continue
        segments.append(
            {
                "start_frame": clipped_start,
                "end_frame": clipped_end,
                "start_time": clipped_start / fps,
                "end_time": clipped_end / fps,
            }
        )
    return segments


def _raise_task_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError) and (
        str(exc).startswith("Invalid label") or "frame labels" in str(exc)
    ):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, sqlite3.IntegrityError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise HTTPException(status_code=409, detail=str(exc)) from exc


def run_server(dataset_root: Path, db_path: Path, host: str, port: int, stride: int) -> None:
    import uvicorn

    uvicorn.run(create_app(dataset_root=dataset_root, db_path=db_path, initial_stride=stride), host=host, port=port)
