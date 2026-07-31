"""Freshness caches for latest scalar values and camera images."""

import threading
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


class StaleDataError(RuntimeError):
    """Raised when required cached data is missing or too old."""


_STALE_TOLERANCE_S = 1e-12


def _is_stale(age_s: float, max_age_s: float) -> bool:
    return age_s - max_age_s > _STALE_TOLERANCE_S


class LatestValueCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._value: Any = None
        self._received_s: Optional[float] = None

    def update(self, value: Any, received_s: Optional[float] = None) -> None:
        with self._lock:
            if received_s is None:
                received_s = time.monotonic()
            self._value = value
            self._received_s = float(received_s)

    def age_s(self, now_s: Optional[float] = None) -> Optional[float]:
        with self._lock:
            if self._received_s is None:
                return None
            if now_s is None:
                now_s = time.monotonic()
            return float(now_s) - self._received_s

    def snapshot(self, max_age_s: float, now_s: Optional[float] = None) -> Tuple[Any, float]:
        with self._lock:
            if self._received_s is None:
                raise StaleDataError("no value has been received")
            if now_s is None:
                now_s = time.monotonic()
            age = float(now_s) - self._received_s
            if _is_stale(age, max_age_s):
                raise StaleDataError(f"cached value is stale: age {age:.3f}s exceeds {max_age_s:.3f}s")
            return self._value, age


class LatestImageCache:
    def __init__(self, required_keys: Iterable[str]) -> None:
        self._lock = threading.RLock()
        self._required_keys = tuple(required_keys)
        self._images: Dict[str, np.ndarray] = {}
        self._received_s: Dict[str, float] = {}

    def update(self, key: str, image: np.ndarray, received_s: Optional[float] = None) -> None:
        with self._lock:
            if key not in self._required_keys:
                raise KeyError(key)
            if received_s is None:
                received_s = time.monotonic()
            self._images[key] = np.array(image, copy=True)
            self._received_s[key] = float(received_s)

    def snapshot(
        self, max_age_s: float, now_s: Optional[float] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        with self._lock:
            if now_s is None:
                now_s = time.monotonic()

            missing = [key for key in self._required_keys if key not in self._images]
            if missing:
                raise StaleDataError(f"missing required image(s): {', '.join(missing)}")

            ages = {key: float(now_s) - self._received_s[key] for key in self._required_keys}
            stale = {key: age for key, age in ages.items() if _is_stale(age, max_age_s)}
            if stale:
                details = ", ".join(
                    f"{key} stale: age {age:.3f}s exceeds {max_age_s:.3f}s" for key, age in stale.items()
                )
                raise StaleDataError(details)

            images = {key: self._images[key].copy() for key in self._required_keys}
            return images, ages
