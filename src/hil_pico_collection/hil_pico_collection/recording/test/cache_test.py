import threading

import numpy as np
import pytest

from hil_pico_collection.recording.cache import LatestImageCache, LatestValueCache, StaleDataError


def test_latest_value_cache_accepts_fresh_value_and_reports_age():
    cache = LatestValueCache()
    cache.update("value", received_s=10.0)

    assert cache.snapshot(max_age_s=0.5, now_s=10.25) == ("value", 0.25)
    assert cache.age_s(now_s=10.25) == 0.25


def test_latest_value_cache_raises_for_missing_or_stale_value():
    cache = LatestValueCache()

    with pytest.raises(StaleDataError, match="no value"):
        cache.snapshot(max_age_s=1.0, now_s=10.0)
    assert cache.age_s(now_s=10.0) is None

    cache.update("old", received_s=5.0)
    with pytest.raises(StaleDataError, match="stale"):
        cache.snapshot(max_age_s=1.0, now_s=7.0)


def test_latest_value_cache_allows_boundary_age_with_float_roundoff():
    cache = LatestValueCache()
    cache.update("boundary", received_s=1.0)

    value, age = cache.snapshot(max_age_s=0.036, now_s=1.036)

    assert value == "boundary"
    assert age == pytest.approx(0.036)


def test_latest_image_cache_allows_boundary_age_with_float_roundoff():
    cache = LatestImageCache(required_keys=["front"])
    image = np.array([[1]], dtype=np.uint8)
    cache.update("front", image, received_s=1.0)

    images, ages = cache.snapshot(max_age_s=0.036, now_s=1.036)

    np.testing.assert_array_equal(images["front"], image)
    assert ages == pytest.approx({"front": 0.036})


def test_latest_image_cache_requires_all_required_cameras_and_returns_images_and_ages():
    cache = LatestImageCache(required_keys=["left", "right"])
    left = np.array([[1, 2]], dtype=np.uint8)
    right = np.array([[3, 4]], dtype=np.uint8)

    cache.update("left", left, received_s=10.0)
    with pytest.raises(StaleDataError, match="missing.*right"):
        cache.snapshot(max_age_s=1.0, now_s=10.1)

    cache.update("right", right, received_s=10.2)
    images, ages = cache.snapshot(max_age_s=1.0, now_s=10.5)

    np.testing.assert_array_equal(images["left"], left)
    np.testing.assert_array_equal(images["right"], right)
    assert ages == pytest.approx({"left": 0.5, "right": 0.3})


def test_latest_image_cache_copies_images_on_update_and_rejects_unknown_keys():
    cache = LatestImageCache(required_keys=["front"])
    image = np.array([[1, 2]], dtype=np.uint8)
    cache.update("front", image, received_s=3.0)
    image[0, 0] = 99

    images, ages = cache.snapshot(max_age_s=1.0, now_s=3.1)

    assert images["front"][0, 0] == 1
    assert ages == pytest.approx({"front": 0.1})
    with pytest.raises(KeyError):
        cache.update("side", np.array([[5]], dtype=np.uint8), received_s=3.2)


def test_latest_image_cache_raises_for_stale_camera():
    cache = LatestImageCache(required_keys=["front"])
    cache.update("front", np.array([[1]], dtype=np.uint8), received_s=1.0)

    with pytest.raises(StaleDataError, match="front.*stale"):
        cache.snapshot(max_age_s=0.5, now_s=2.0)


def _assert_blocks_behind_lock(lock, target):
    started = threading.Event()
    finished = threading.Event()
    errors = []

    lock.acquire()

    def run_target():
        started.set()
        try:
            target()
        except Exception as exc:  # pragma: no cover - reported by assertion below
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=run_target)
    thread.start()
    assert started.wait(timeout=1.0)
    assert not finished.wait(timeout=0.05)

    lock.release()
    assert finished.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert errors == []


def test_latest_caches_serialize_operations_with_their_lock():
    value_cache = LatestValueCache()
    image_cache = LatestImageCache(required_keys=["front"])
    image_cache.update("front", np.array([[1]], dtype=np.uint8), received_s=10.0)

    _assert_blocks_behind_lock(
        value_cache._lock,
        lambda: value_cache.update("value", received_s=1.0),
    )
    _assert_blocks_behind_lock(
        value_cache._lock,
        lambda: value_cache.age_s(now_s=1.1),
    )
    _assert_blocks_behind_lock(
        image_cache._lock,
        lambda: image_cache.update("front", np.array([[2]], dtype=np.uint8), received_s=10.1),
    )
    _assert_blocks_behind_lock(
        image_cache._lock,
        lambda: image_cache.snapshot(max_age_s=1.0, now_s=10.2),
    )
