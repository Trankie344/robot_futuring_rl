"""Pure-Python recorder core for synchronized HIL PICO observations."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from hil_pico_collection.adapters.base import RobotAdapter

from .cache import StaleDataError
from .episode_buffer import EpisodeBuffer, EpisodeStatus, SealedEpisode, copy_frame


class RecorderCore:
    """Record synchronized state, executed action and three camera frames at 30 Hz."""

    def __init__(
        self,
        status_cache: Any,
        action_cache: Any,
        image_cache: Any,
        adapter: RobotAdapter | None = None,
        *,
        status_max_age_s: float = 0.10,
        action_max_age_s: float = 0.50,
        image_max_age_s: float = 0.100,
        status_stale_abort_count: int = 20,
        action_stale_abort_count: int = 20,
        image_age_average_window: int = 30,
        image_age_average_max_s: float = 1.0 / 28.0,
        image_stale_abort_count: int = 10,
    ) -> None:
        self.status_cache = status_cache
        self.action_cache = action_cache
        self.image_cache = image_cache
        if adapter is None:
            from hil_pico_collection.adapters.arm_interfaces import ArmInterfacesAdapter

            adapter = ArmInterfacesAdapter(
                status_message_type=object,
                command_message_type=object,
                mode_service_type=object,
                secondary_service_type=object,
            )
        self.adapter = adapter
        self.status_max_age_s = float(status_max_age_s)
        self.action_max_age_s = float(action_max_age_s)
        self.image_max_age_s = float(image_max_age_s)
        self.status_stale_abort_count = int(status_stale_abort_count)
        self.action_stale_abort_count = int(action_stale_abort_count)
        self.image_age_average_window = int(image_age_average_window)
        self.image_age_average_max_s = float(image_age_average_max_s)
        self.image_stale_abort_count = int(image_stale_abort_count)
        self.drop_count = 0
        self.last_drop_reason: str | None = None
        self.last_error: str | None = None
        self.failed_episodes: list[SealedEpisode] = []
        self._image_age_histories: dict[str, deque[float]] = {}
        self._status_stale_drop_count = 0
        self._action_stale_drop_count = 0
        self._image_stale_drop_count = 0
        self._active_buffer: EpisodeBuffer | None = None
        self._next_episode_index = 0

    @property
    def current_frame_count(self) -> int:
        return 0 if self._active_buffer is None else self._active_buffer.current_frame_count

    @property
    def recording(self) -> bool:
        return self._active_buffer is not None

    def start_episode(self, task: Any) -> EpisodeBuffer:
        if self._active_buffer is not None:
            raise RuntimeError("an episode is already recording")
        if not str(task).strip():
            raise ValueError("task must not be empty")
        self.drop_count = 0
        self.last_drop_reason = None
        self.last_error = None
        self._clear_stale_state()
        self._active_buffer = EpisodeBuffer(episode_index=self._next_episode_index, task=task)
        return self._active_buffer

    def record_tick(self, now_s: float | None = None) -> dict[str, Any] | None:
        if self._active_buffer is None:
            self.last_drop_reason = "not recording"
            return None
        now_s = time.monotonic() if now_s is None else float(now_s)

        try:
            status_message, _ = self.status_cache.snapshot(self.status_max_age_s, now_s=now_s)
        except StaleDataError as exc:
            return self._drop_or_abort_stale("status", exc, "_status_stale_drop_count", self.status_stale_abort_count)
        try:
            action_message, _ = self.action_cache.snapshot(self.action_max_age_s, now_s=now_s)
        except StaleDataError as exc:
            return self._drop_or_abort_stale("action", exc, "_action_stale_drop_count", self.action_stale_abort_count)
        try:
            images, image_ages = self.image_cache.snapshot(self.image_max_age_s, now_s=now_s)
        except StaleDataError as exc:
            self._image_stale_drop_count += 1
            reason = f"image: {exc}"
            if self._image_stale_drop_count < self.image_stale_abort_count:
                return self._drop(reason)
            return self._abort_episode(
                f"{reason}; stale image frame threshold exceeded {self.image_stale_abort_count} time(s)"
            )

        for key, age_s in image_ages.items():
            history = self._image_age_history_for(str(key))
            history.append(float(age_s))
            if self.image_age_average_window <= 0 or len(history) < self.image_age_average_window:
                continue
            average_age_s = sum(history) / len(history)
            if average_age_s > self.image_age_average_max_s:
                return self._abort_episode(
                    f"image: {key} average image age {average_age_s:.4f}s exceeds "
                    f"1/28s ({self.image_age_average_max_s:.4f}s)"
                )

        try:
            sample = self.adapter.parse_status(status_message)
            action = self.adapter.parse_executed_action(action_message)
        except (AttributeError, TypeError, ValueError) as exc:
            return self._drop(f"mapping: {exc}")

        frame = {
            "timestamp": now_s,
            "frame_index": self._active_buffer.current_frame_count,
            "episode_index": self._active_buffer.episode_index,
            "task": self._active_buffer.task,
            "observation.state": sample.state,
            "action": action,
            "images": images,
            "intervention": sample.intervention,
            "control_mode": sample.control_mode,
        }
        self._active_buffer.append(frame)
        return copy_frame(frame)

    def end_episode(self) -> SealedEpisode:
        if self._active_buffer is None:
            raise RuntimeError("no episode is recording")
        sealed = self._active_buffer.seal()
        self._active_buffer = None
        self._next_episode_index = max(self._next_episode_index, sealed.episode_index + 1)
        self._clear_stale_state()
        return sealed

    def _drop(self, reason: str) -> None:
        self.drop_count += 1
        self.last_drop_reason = reason
        return None

    def _drop_or_abort_stale(
        self,
        label: str,
        exc: StaleDataError,
        counter_name: str,
        abort_count: int,
    ) -> None:
        count = int(getattr(self, counter_name)) + 1
        setattr(self, counter_name, count)
        reason = f"{label}: {exc}"
        if count < abort_count:
            return self._drop(reason)
        return self._abort_episode(f"{reason}; stale {label} threshold exceeded {abort_count} time(s)")

    def _abort_episode(self, reason: str) -> None:
        self.drop_count += 1
        self.last_drop_reason = reason
        self.last_error = reason
        if self._active_buffer is None:
            return None
        sealed = self._active_buffer.seal()
        sealed.status = EpisodeStatus.failed
        sealed.metadata["save_error"] = reason
        self.failed_episodes.append(sealed)
        sealed.cleanup()
        self._active_buffer = None
        self._next_episode_index = max(self._next_episode_index, sealed.episode_index + 1)
        self._clear_stale_state()
        return None

    def _image_age_history_for(self, key: str) -> deque[float]:
        maxlen = max(1, self.image_age_average_window)
        history = self._image_age_histories.get(key)
        if history is None or history.maxlen != maxlen:
            history = deque(maxlen=maxlen)
            self._image_age_histories[key] = history
        return history

    def _clear_stale_state(self) -> None:
        self._status_stale_drop_count = 0
        self._action_stale_drop_count = 0
        self._image_stale_drop_count = 0
        for history in self._image_age_histories.values():
            history.clear()
