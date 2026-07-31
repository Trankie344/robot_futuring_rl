"""FastAPI recording controls and read-only dataset/video browser."""

from __future__ import annotations

import queue
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from hil_pico_collection.recording.episode_buffer import EpisodeStatus, SaveJobQueue
from hil_pico_collection.replay.dataset import ReplayDataset

INDEX_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HIL PICO Recorder</title>
<style>
body{font-family:sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f4f6f8}
section{background:white;border:1px solid #d8dde3;padding:1rem;margin:1rem 0;border-radius:.5rem}
button,input{font:inherit;padding:.55rem;margin:.2rem}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:.5rem;border-bottom:1px solid #ddd}.error{color:#a33}
</style></head>
<body><h1>HIL PICO Recorder</h1>
<section><input id="task" value="pick object"><button onclick="startEpisode()">Start</button>
<button onclick="endEpisode()">End</button><button onclick="resetRobot()">Reset request</button>
<span id="message" class="error"></span></section>
<section><pre id="status">loading</pre></section>
<section><h2>Dataset episodes</h2><table><thead><tr><th>Episode</th><th>Task</th><th>Frames</th><th>Video</th></tr></thead>
<tbody id="episodes"></tbody></table></section>
<script>
async function json(url,options){const r=await fetch(url,options||{});const d=await r.json();if(!r.ok)throw Error(d.detail||r.statusText);return d}
async function startEpisode(){try{await json('/api/episodes/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:document.getElementById('task').value})});refresh()}catch(e){message.textContent=e.message}}
async function endEpisode(){try{await json('/api/episodes/end',{method:'POST'});refresh()}catch(e){message.textContent=e.message}}
async function resetRobot(){try{const d=await json('/api/robot/reset',{method:'POST'});message.textContent=d.accepted?'reset accepted':'reset rejected'}catch(e){message.textContent=e.message}}
async function refresh(){try{document.getElementById('status').textContent=JSON.stringify(await json('/api/status'),null,2);const d=await json('/api/replay/episodes');const b=document.getElementById('episodes');b.innerHTML='';for(const e of d.episodes){const r=document.createElement('tr');r.innerHTML='<td>'+e.episode_index+'</td><td>'+e.task+'</td><td>'+e.frame_count+'</td><td><a href="/replay?episode_index='+e.episode_index+'">open</a></td>';b.appendChild(r)}}catch(e){message.textContent=e.message}}
refresh();setInterval(refresh,1000)
</script></body></html>"""


REPLAY_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HIL PICO Video Replay</title><style>body{font-family:sans-serif;margin:1rem}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}video{width:100%;background:#111}</style></head>
<body><h1>Dataset video replay</h1><div id="videos" class="videos"></div>
<script>
const id=new URLSearchParams(location.search).get('episode_index');
fetch('/api/replay/episodes/'+encodeURIComponent(id)).then(r=>r.json()).then(e=>{const root=document.getElementById('videos');for(const [name,item] of Object.entries(e.videos)){const panel=document.createElement('div');const title=document.createElement('h2');title.textContent=name;const video=document.createElement('video');video.controls=true;if(item.exists)video.src=item.url;panel.append(title,video);root.appendChild(panel)}})
</script></body></html>"""


class StartEpisodeRequest(BaseModel):
    task: str


class SaveCoordinator:
    """Single-writer asynchronous save queue."""

    def __init__(self, writer: Any) -> None:
        self.writer = writer
        self.jobs = SaveJobQueue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def enqueue(self, sealed: Any) -> dict[str, Any]:
        job = self.jobs.enqueue(sealed)
        self._ensure_worker()
        return self.jobs.status_for_job(int(job.metadata["job_id"]), include_job_id=True) or {}

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._drain, name="hil-pico-save", daemon=True)
            self._thread.start()

    def _drain(self) -> None:
        while True:
            try:
                sealed = self.jobs.get_nowait()
            except queue.Empty:
                with self._lock:
                    try:
                        sealed = self.jobs.get_nowait()
                    except queue.Empty:
                        self._thread = None
                        return
            try:
                written_index = self.writer.write_episode(sealed)
            except Exception as exc:
                self.jobs.set_status(sealed, EpisodeStatus.failed, error=str(exc))
            else:
                self.jobs.set_status(sealed, EpisodeStatus.saved, episode_index=written_index)
            finally:
                sealed.cleanup()


def create_app(
    core: Any,
    writer: Any,
    *,
    reset_request: Any = None,
) -> FastAPI:
    """Create the WebUI; robot reset is a generic fire-and-forget request."""

    app = FastAPI(title="HIL PICO collection")
    coordinator = SaveCoordinator(writer)
    app.state.core = core
    app.state.writer = writer
    app.state.save_coordinator = coordinator
    app.state.reset_request = reset_request

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_PAGE

    @app.get("/replay", response_class=HTMLResponse)
    def replay_page() -> str:
        return REPLAY_PAGE

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "recording": _is_recording(core),
            "current_frame_count": int(getattr(core, "current_frame_count", 0) or 0),
            "drop_count": int(getattr(core, "drop_count", 0) or 0),
            "last_drop_reason": getattr(core, "last_drop_reason", None),
            "last_error": getattr(core, "last_error", None),
            "current_mode": _current_mode(core),
            "camera_ages": _camera_ages(core),
            "save_queue": coordinator.jobs.list_status(include_job_id=True),
        }

    @app.get("/api/episodes")
    def episodes() -> dict[str, list[dict[str, Any]]]:
        saved = _reader(writer).list_episodes()
        saved_indices = {item["episode_index"] for item in saved}
        jobs = [
            item
            for item in coordinator.jobs.list_status(include_job_id=True)
            if item.get("status") != EpisodeStatus.saved.value or item.get("episode_index") not in saved_indices
        ]
        return {"episodes": sorted(saved + jobs, key=_episode_sort_key)}

    @app.post("/api/episodes/start")
    def start_episode(request: StartEpisodeRequest) -> dict[str, Any]:
        task = request.task.strip()
        if not task:
            raise HTTPException(status_code=422, detail="task must not be empty")
        try:
            buffer = core.start_episode(task)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"episode_index": int(buffer.episode_index), "recording": True}

    @app.post("/api/episodes/end")
    def end_episode() -> dict[str, Any]:
        try:
            sealed = core.end_episode()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return coordinator.enqueue(sealed)

    @app.delete("/api/episodes/{episode_index}")
    def delete_episode(episode_index: int) -> dict[str, Any]:
        if _is_recording(core):
            raise HTTPException(status_code=409, detail="cannot delete while recording")
        try:
            writer.delete_episode(int(episode_index))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        coordinator.jobs.reconcile_after_delete(int(episode_index))
        return {"deleted": True, "episode_index": int(episode_index)}

    @app.delete("/api/save-jobs/{job_id}")
    def clear_save_job(job_id: int) -> dict[str, Any]:
        try:
            removed = coordinator.jobs.remove_job(int(job_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if removed is None:
            raise HTTPException(status_code=404, detail="save job does not exist")
        removed.cleanup()
        return {"deleted": True, "job_id": int(job_id)}

    @app.get("/api/replay/episodes")
    def replay_episodes() -> dict[str, list[dict[str, Any]]]:
        return {"episodes": _reader(writer).list_episodes()}

    @app.get("/api/replay/episodes/{episode_index}")
    def replay_episode(episode_index: int) -> dict[str, Any]:
        try:
            summary = _reader(writer).episode_summary(int(episode_index))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        for key, item in summary["videos"].items():
            item["url"] = f"/api/replay/episodes/{int(episode_index)}/video/{key}"
        return summary

    @app.get("/api/replay/episodes/{episode_index}/video/{video_key}")
    def replay_video(episode_index: int, video_key: str) -> FileResponse:
        try:
            path = _reader(writer).video_path(int(episode_index), video_key)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail="video does not exist")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/robot/reset")
    @app.post("/api/robot/reset-arm", include_in_schema=False)
    def reset() -> dict[str, bool]:
        handler = app.state.reset_request
        if handler is None:
            raise HTTPException(status_code=503, detail="reset request publisher is not configured")
        handler()
        return {"accepted": True}

    return app


def _reader(writer: Any) -> ReplayDataset:
    return ReplayDataset(writer.root, fps=getattr(writer, "fps", None))


def _is_recording(core: Any) -> bool:
    if hasattr(core, "recording"):
        return bool(core.recording)
    return getattr(core, "_active_buffer", None) is not None


def _current_mode(core: Any) -> int | None:
    cache = getattr(core, "status_cache", None)
    adapter = getattr(getattr(core, "raw_core", core), "adapter", None)
    if cache is None or adapter is None:
        return None
    try:
        message, _ = cache.snapshot(0.5)
        return int(adapter.parse_status(message).control_mode)
    except Exception:
        return None


def _camera_ages(core: Any) -> dict[str, float]:
    cache = getattr(core, "image_cache", None)
    if cache is None:
        return {}
    try:
        _, ages = cache.snapshot(60.0)
        return {str(key): float(value) for key, value in ages.items()}
    except Exception:
        return {}


def _episode_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    episode_index = item.get("episode_index")
    job_id = item.get("job_id")
    return (
        int(episode_index) if episode_index is not None else 10**12,
        int(job_id) if job_id is not None else -1,
    )
