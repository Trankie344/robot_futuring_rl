"""In-memory episode buffering and save-job status helpers."""

import gc
import os
import pickle
import queue
import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

DEFAULT_SPILL_CHUNK_SIZE = 500


class EpisodeStatus(str, Enum):
    recording = "recording"
    queued = "queued"
    saving = "saving"
    saved = "saved"
    failed = "failed"


def copy_frame(frame: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _copy_frame_value(value) for key, value in frame.items()}


def _copy_frame_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_frame_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_copy_frame_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy_frame_value(child) for child in value)
    return value


@dataclass
class SealedEpisode:
    episode_index: int
    task: Any
    frames: Sequence[Dict[str, Any]]
    status: EpisodeStatus = EpisodeStatus.queued
    task_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> None:
        cleanup = getattr(self.frames, "cleanup", None)
        if callable(cleanup):
            cleanup()


class SpilledFrameSequence(Sequence[Dict[str, Any]]):
    """Re-iterable frame sequence backed by temporary chunk files."""

    def __init__(
        self,
        chunk_paths: Sequence[Any],
        chunk_lengths: Sequence[int],
        *,
        cleanup_dir: Optional[Any] = None,
    ) -> None:
        if len(chunk_paths) != len(chunk_lengths):
            raise ValueError("chunk_paths and chunk_lengths must have the same length")
        self._chunk_paths = tuple(Path(path) for path in chunk_paths)
        self._chunk_lengths = tuple(int(length) for length in chunk_lengths)
        self._cleanup_dir = Path(cleanup_dir) if cleanup_dir is not None else None
        self._length = sum(self._chunk_lengths)

    @property
    def chunk_paths(self) -> Sequence[Path]:
        return self._chunk_paths

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for path in self._chunk_paths:
            for frame in self._load_chunk(path):
                yield frame

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError("frame index must be an integer")
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)

        offset = index
        for path, chunk_length in zip(self._chunk_paths, self._chunk_lengths):
            if offset < chunk_length:
                return copy_frame(self._load_chunk(path)[offset])
            offset -= chunk_length
        raise IndexError(index)

    def cleanup(self) -> None:
        if self._cleanup_dir is not None:
            shutil.rmtree(self._cleanup_dir, ignore_errors=True)
            return
        for path in self._chunk_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _load_chunk(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("rb") as handle:
            frames = pickle.load(handle)
        if not isinstance(frames, list):
            raise ValueError("spilled frame chunk is not a list: {0}".format(path))
        return frames


class EpisodeBuffer:
    def __init__(
        self,
        episode_index: int,
        task: Any,
        task_index: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        spill_chunk_size: int = DEFAULT_SPILL_CHUNK_SIZE,
        spill_root: Optional[Any] = None,
    ) -> None:
        self.episode_index = int(episode_index)
        self.task = task
        self.task_index = task_index
        self.metadata = dict(metadata or {})
        self.spill_chunk_size = int(spill_chunk_size)
        self.spill_root = Path(spill_root) if spill_root is not None else None
        self.status = EpisodeStatus.recording
        self._frames: List[Dict[str, Any]] = []
        self._spilled_chunk_paths: List[Path] = []
        self._spilled_chunk_lengths: List[int] = []
        self._spill_futures: List[Future] = []
        self._spill_executor: Optional[ThreadPoolExecutor] = None
        self._spill_dir: Optional[Path] = None
        self._spilled_frame_count = 0
        self._sealed: Optional[SealedEpisode] = None

    @property
    def current_frame_count(self) -> int:
        return self._spilled_frame_count + len(self._frames)

    def append(self, frame: Dict[str, Any]) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot append to a sealed episode")
        self._frames.append(copy_frame(frame))
        if self.spill_chunk_size > 0 and len(self._frames) >= self.spill_chunk_size:
            self._queue_current_frames_for_spill()

    def seal(self) -> SealedEpisode:
        if self._sealed is None:
            if self._spilled_chunk_paths or self._spill_futures:
                self._queue_current_frames_for_spill()
                self._wait_for_spills()
                frames: Sequence[Dict[str, Any]] = SpilledFrameSequence(
                    self._spilled_chunk_paths,
                    self._spilled_chunk_lengths,
                    cleanup_dir=self._spill_dir,
                )
            else:
                frames = [copy_frame(frame) for frame in self._frames]
            self.status = EpisodeStatus.queued
            self._sealed = SealedEpisode(
                episode_index=self.episode_index,
                task=self.task,
                task_index=self.task_index,
                metadata=dict(self.metadata),
                frames=frames,
                status=EpisodeStatus.queued,
            )
        return self._sealed

    def cleanup(self) -> None:
        if self._sealed is not None:
            self._sealed.cleanup()
        elif self._spill_dir is not None:
            self._wait_for_spills()
            shutil.rmtree(self._spill_dir, ignore_errors=True)

    def _queue_current_frames_for_spill(self) -> None:
        if not self._frames:
            return
        spill_dir = self._ensure_spill_dir()
        chunk_index = len(self._spilled_chunk_paths)
        final_path = spill_dir / "chunk_{0:06d}.pkl".format(chunk_index)
        frames = self._frames
        frame_count = len(frames)

        self._frames = []
        self._spilled_chunk_paths.append(final_path)
        self._spilled_chunk_lengths.append(frame_count)
        self._spilled_frame_count += frame_count
        self._spill_futures.append(self._spill_executor_instance().submit(_write_spill_chunk, final_path, frames))

    def _ensure_spill_dir(self) -> Path:
        if self._spill_dir is None:
            if self.spill_root is not None:
                self.spill_root.mkdir(parents=True, exist_ok=True)
            self._spill_dir = Path(
                tempfile.mkdtemp(
                    prefix="hil_pico_ep_{0:06d}_".format(self.episode_index),
                    dir=str(self.spill_root) if self.spill_root is not None else None,
                )
            )
        return self._spill_dir

    def _spill_executor_instance(self) -> ThreadPoolExecutor:
        if self._spill_executor is None:
            self._spill_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="hil-pico-spill",
            )
        return self._spill_executor

    def _wait_for_spills(self) -> None:
        errors = []
        for future in self._spill_futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        self._spill_futures = []
        if self._spill_executor is not None:
            self._spill_executor.shutdown(wait=True)
            self._spill_executor = None
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))


def spill_sealed_episode(
    sealed: SealedEpisode,
    *,
    spill_chunk_size: int = DEFAULT_SPILL_CHUNK_SIZE,
    spill_root: Optional[Any] = None,
) -> SealedEpisode:
    if isinstance(sealed.frames, SpilledFrameSequence):
        return sealed

    spill_chunk_size = max(1, int(spill_chunk_size))
    frames = list(sealed.frames)
    if not frames:
        return sealed

    spill_parent = Path(spill_root) if spill_root is not None else None
    if spill_parent is not None:
        spill_parent.mkdir(parents=True, exist_ok=True)
    spill_dir = Path(
        tempfile.mkdtemp(
            prefix="hil_pico_save_{0:06d}_".format(int(sealed.episode_index)),
            dir=str(spill_parent) if spill_parent is not None else None,
        )
    )
    chunk_paths: List[Path] = []
    chunk_lengths: List[int] = []

    try:
        for chunk_index, start_index in enumerate(range(0, len(frames), spill_chunk_size)):
            chunk = [copy_frame(frame) for frame in frames[start_index : start_index + spill_chunk_size]]
            path = spill_dir / "chunk_{0:06d}.pkl".format(chunk_index)
            _write_spill_chunk(path, chunk)
            chunk_paths.append(path)
            chunk_lengths.append(len(chunk))
    except Exception:
        shutil.rmtree(spill_dir, ignore_errors=True)
        raise

    return SealedEpisode(
        episode_index=sealed.episode_index,
        task=sealed.task,
        frames=SpilledFrameSequence(chunk_paths, chunk_lengths, cleanup_dir=spill_dir),
        status=sealed.status,
        task_index=sealed.task_index,
        metadata=dict(sealed.metadata),
    )


def _write_spill_chunk(final_path: Path, frames: List[Dict[str, Any]]) -> None:
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
        del frames
        gc.collect()


class SaveJobQueue:
    def __init__(self) -> None:
        self._queue: "queue.Queue[SealedEpisode]" = queue.Queue()
        self._lock = threading.RLock()
        self._episodes: List[SealedEpisode] = []
        self._next_job_id = 0

    def enqueue(self, sealed: SealedEpisode) -> SealedEpisode:
        with self._lock:
            sealed.status = EpisodeStatus.queued
            sealed.metadata.setdefault("original_episode_index", int(sealed.episode_index))
            if "job_id" not in sealed.metadata:
                sealed.metadata["job_id"] = self._next_job_id
                self._next_job_id += 1
            self._episodes.append(sealed)
            self._queue.put(sealed)
        return sealed

    def get_nowait(self) -> SealedEpisode:
        with self._lock:
            sealed = self._queue.get_nowait()
            sealed.status = EpisodeStatus.saving
            return sealed

    def list_status(self, *, include_job_id: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._status_dict(sealed, include_job_id=include_job_id) for sealed in self._episodes]

    def set_status(
        self,
        sealed: SealedEpisode,
        status: EpisodeStatus,
        *,
        episode_index: Optional[int] = None,
        error: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._lock:
            if episode_index is not None:
                sealed.episode_index = int(episode_index)
                sealed.metadata["written_episode_index"] = int(episode_index)
            sealed.status = EpisodeStatus(status)
            if error is None:
                sealed.metadata.pop("save_error", None)
            else:
                sealed.metadata["save_error"] = str(error)
            if extra_metadata:
                sealed.metadata.update(dict(extra_metadata))

    def status_for_episode(
        self,
        episode_index: int,
        *,
        include_job_id: bool = False,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            for sealed in self._episodes:
                if self._matches_episode(sealed, int(episode_index)):
                    return self._status_dict(sealed, include_job_id=include_job_id)
        return None

    def status_for_job(
        self,
        job_id: int,
        *,
        include_job_id: bool = False,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            for sealed in self._episodes:
                if sealed.metadata.get("job_id") == int(job_id):
                    return self._status_dict(sealed, include_job_id=include_job_id)
        return None

    def remove_finished(self, episode_index: int) -> Optional[SealedEpisode]:
        with self._lock:
            for index, sealed in enumerate(self._episodes):
                if not self._matches_episode(sealed, int(episode_index)):
                    continue
                if sealed.status in (
                    EpisodeStatus.recording,
                    EpisodeStatus.queued,
                    EpisodeStatus.saving,
                ):
                    raise RuntimeError("episode is not finished")
                return self._episodes.pop(index)
        return None

    def remove_job(self, job_id: int) -> Optional[SealedEpisode]:
        with self._lock:
            for index, sealed in enumerate(self._episodes):
                if sealed.metadata.get("job_id") != int(job_id):
                    continue
                if sealed.status in (
                    EpisodeStatus.recording,
                    EpisodeStatus.queued,
                    EpisodeStatus.saving,
                ):
                    raise RuntimeError("job is not finished")
                return self._episodes.pop(index)
        return None

    def reconcile_after_delete(self, deleted_episode_index: int) -> None:
        deleted_episode_index = int(deleted_episode_index)
        with self._lock:
            kept = []
            for sealed in self._episodes:
                dataset_index = self._dataset_episode_index(sealed)
                if dataset_index is None:
                    kept.append(sealed)
                    continue
                if dataset_index == deleted_episode_index:
                    continue
                if dataset_index > deleted_episode_index:
                    dataset_index -= 1
                    sealed.episode_index = dataset_index
                    sealed.metadata["written_episode_index"] = dataset_index
                kept.append(sealed)
            self._episodes = kept

    def _matches_episode(self, sealed: SealedEpisode, episode_index: int) -> bool:
        if sealed.status in (EpisodeStatus.queued, EpisodeStatus.saving, EpisodeStatus.recording):
            return int(sealed.metadata.get("original_episode_index", sealed.episode_index)) == int(episode_index)
        dataset_index = self._dataset_episode_index(sealed)
        return dataset_index is not None and dataset_index == int(episode_index)

    def _status_dict(self, sealed: SealedEpisode, *, include_job_id: bool = False) -> Dict[str, Any]:
        status_value = EpisodeStatus(sealed.status)
        original_index = int(sealed.metadata.get("original_episode_index", sealed.episode_index))
        dataset_index = self._dataset_episode_index(sealed)
        episode_index = dataset_index
        if episode_index is None and status_value in (
            EpisodeStatus.recording,
            EpisodeStatus.queued,
            EpisodeStatus.saving,
        ):
            episode_index = original_index

        status = {
            "episode_index": episode_index,
            "task": sealed.task,
            "frame_count": len(sealed.frames),
            "status": status_value.value,
        }
        if include_job_id:
            status["job_id"] = sealed.metadata.get("job_id")
            status["original_episode_index"] = original_index
            if dataset_index is not None:
                status["dataset_episode_index"] = dataset_index
            if status_value == EpisodeStatus.failed and dataset_index is None:
                status["delete_url"] = "/api/save-jobs/{0}".format(sealed.metadata.get("job_id"))
        if "save_error" in sealed.metadata:
            status["error"] = sealed.metadata["save_error"]
        if dataset_index is not None:
            status["written_episode_index"] = dataset_index
        return status

    def _dataset_episode_index(self, sealed: SealedEpisode) -> Optional[int]:
        if "written_episode_index" not in sealed.metadata:
            return None
        return int(sealed.metadata["written_episode_index"])
