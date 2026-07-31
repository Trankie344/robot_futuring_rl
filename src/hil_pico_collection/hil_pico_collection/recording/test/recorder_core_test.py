import queue as queue_module
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hil_pico_collection.adapters.arm_interfaces import AUTONOMY_JOINT
from hil_pico_collection.recording.cache import LatestImageCache, LatestValueCache
from hil_pico_collection.recording.episode_buffer import (
    EpisodeBuffer,
    EpisodeStatus,
    SaveJobQueue,
    SealedEpisode,
    SpilledFrameSequence,
)
from hil_pico_collection.recording.recorder_core import RecorderCore


def joint(value):
    return SimpleNamespace(joint_status=[value])


def fake_arm_status():
    return SimpleNamespace(
        left_arm=[joint(10), joint(11), joint(12), joint(13), joint(14), joint(15), joint(16)],
        right_arm=[joint(20), joint(21), joint(22), joint(23), joint(24), joint(25), joint(26)],
        gripper=[joint(30), joint(31)],
        other_status=[AUTONOMY_JOINT, 0],
    )


def fake_auto_arm_cmd():
    return SimpleNamespace(
        left_joint_command=[100, 101, 102, 103, 104, 105, 106],
        right_joint_command=[200, 201, 202, 203, 204, 205, 206],
        gripper_command=[300, 301],
    )


def rgb(value):
    return np.full((2, 3, 3), value, dtype=np.uint8)


def fresh_caches(received_s=1.0):
    status_cache = LatestValueCache()
    action_cache = LatestValueCache()
    image_cache = LatestImageCache(required_keys=["top", "left_wrist", "right_wrist"])

    status_cache.update(fake_arm_status(), received_s=received_s)
    action_cache.update(fake_auto_arm_cmd(), received_s=received_s)
    image_cache.update("top", rgb(10), received_s=received_s)
    image_cache.update("left_wrist", rgb(20), received_s=received_s)
    image_cache.update("right_wrist", rgb(30), received_s=received_s)

    return status_cache, action_cache, image_cache


def update_caches(status_cache, action_cache, image_cache, received_s):
    status_cache.update(fake_arm_status(), received_s=received_s)
    action_cache.update(fake_auto_arm_cmd(), received_s=received_s)
    image_cache.update("top", rgb(10), received_s=received_s)
    image_cache.update("left_wrist", rgb(20), received_s=received_s)
    image_cache.update("right_wrist", rgb(30), received_s=received_s)


def update_caches_with_image_ages(status_cache, action_cache, image_cache, now_s, image_ages):
    status_cache.update(fake_arm_status(), received_s=now_s)
    action_cache.update(fake_auto_arm_cmd(), received_s=now_s)
    image_cache.update("top", rgb(10), received_s=now_s - image_ages["top"])
    image_cache.update("left_wrist", rgb(20), received_s=now_s - image_ages["left_wrist"])
    image_cache.update("right_wrist", rgb(30), received_s=now_s - image_ages["right_wrist"])


def test_record_tick_aborts_episode_after_ten_stale_image_frames():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache, status_max_age_s=1.0, action_max_age_s=1.0)
    core.start_episode(task="pick cup")

    accepted = core.record_tick(now_s=1.099)
    for index in range(9):
        rejected = core.record_tick(now_s=1.101 + index * 0.001)
        assert not rejected
        assert core.current_frame_count == 1
        assert len(core.failed_episodes) == 0
        assert core.last_error is None

    rejected = core.record_tick(now_s=1.110)

    assert accepted
    assert not rejected
    assert core.current_frame_count == 0
    assert core.drop_count == 10
    assert "image" in core.last_drop_reason
    assert "0.100s" in core.last_drop_reason
    assert core.last_error == core.last_drop_reason
    assert len(core.failed_episodes) == 1
    failed = core.failed_episodes[0]
    assert failed.status is EpisodeStatus.failed
    assert failed.metadata["save_error"] == core.last_drop_reason
    assert len(failed.frames) == 1
    assert failed.frames[0]["observation.state"].shape == (16,)
    assert failed.frames[0]["action"].shape == (16,)
    assert failed.frames[0]["action"][7] == 300
    assert failed.frames[0]["action"][15] == 301
    assert failed.frames[0]["intervention"] is False
    assert failed.frames[0]["control_mode"] == AUTONOMY_JOINT
    assert failed.frames[0]["timestamp"] == pytest.approx(1.099)
    assert failed.frames[0]["frame_index"] == 0
    assert sorted(failed.frames[0]["images"]) == ["left_wrist", "right_wrist", "top"]
    with pytest.raises(RuntimeError, match="no episode"):
        core.end_episode()

    next_buffer = core.start_episode(task="place cup")
    assert next_buffer.episode_index == 1


def test_record_tick_accepts_camera_boundary_with_float_roundoff():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache)
    core.start_episode(task="pick cup")

    accepted = core.record_tick(now_s=1.100)
    sealed = core.end_episode()

    assert accepted
    assert core.drop_count == 0
    assert core.last_drop_reason is None
    assert len(sealed.frames) == 1
    assert sealed.frames[0]["timestamp"] == pytest.approx(1.100)


def test_record_tick_allows_action_older_than_200ms_before_500ms_threshold():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache, status_max_age_s=1.0, image_max_age_s=1.0)
    core.start_episode(task="pick cup")

    status_cache.update(fake_arm_status(), received_s=1.201)
    image_cache.update("top", rgb(10), received_s=1.201)
    image_cache.update("left_wrist", rgb(20), received_s=1.201)
    image_cache.update("right_wrist", rgb(30), received_s=1.201)
    accepted = core.record_tick(now_s=1.201)

    assert accepted
    assert core.current_frame_count == 1
    assert core.drop_count == 0
    assert core.last_drop_reason is None


def test_record_tick_aborts_episode_after_twenty_stale_status_frames():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(action_cache=action_cache, image_cache=image_cache, status_cache=status_cache)
    core.start_episode(task="pick cup")

    for index in range(19):
        now_s = 1.101 + index * 0.001
        action_cache.update(fake_auto_arm_cmd(), received_s=now_s)
        image_cache.update("top", rgb(10), received_s=now_s)
        image_cache.update("left_wrist", rgb(20), received_s=now_s)
        image_cache.update("right_wrist", rgb(30), received_s=now_s)
        rejected = core.record_tick(now_s=now_s)
        assert not rejected
        assert core.current_frame_count == 0
        assert len(core.failed_episodes) == 0
        assert core.last_error is None

    now_s = 1.120
    action_cache.update(fake_auto_arm_cmd(), received_s=now_s)
    image_cache.update("top", rgb(10), received_s=now_s)
    image_cache.update("left_wrist", rgb(20), received_s=now_s)
    image_cache.update("right_wrist", rgb(30), received_s=now_s)
    rejected = core.record_tick(now_s=now_s)

    assert not rejected
    assert core.current_frame_count == 0
    assert core.drop_count == 20
    assert "status" in core.last_drop_reason
    assert "0.100s" in core.last_drop_reason
    assert core.last_error == core.last_drop_reason
    assert len(core.failed_episodes) == 1
    assert core.failed_episodes[0].status is EpisodeStatus.failed
    assert core.failed_episodes[0].metadata["save_error"] == core.last_drop_reason
    with pytest.raises(RuntimeError, match="no episode"):
        core.end_episode()


def test_record_tick_aborts_episode_after_twenty_stale_action_frames():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache, status_max_age_s=1.0)
    core.start_episode(task="pick cup")

    for index in range(19):
        now_s = 1.501 + index * 0.001
        status_cache.update(fake_arm_status(), received_s=now_s)
        image_cache.update("top", rgb(10), received_s=now_s)
        image_cache.update("left_wrist", rgb(20), received_s=now_s)
        image_cache.update("right_wrist", rgb(30), received_s=now_s)
        rejected = core.record_tick(now_s=now_s)
        assert not rejected
        assert core.current_frame_count == 0
        assert len(core.failed_episodes) == 0
        assert core.last_error is None

    now_s = 1.520
    status_cache.update(fake_arm_status(), received_s=now_s)
    image_cache.update("top", rgb(10), received_s=now_s)
    image_cache.update("left_wrist", rgb(20), received_s=now_s)
    image_cache.update("right_wrist", rgb(30), received_s=now_s)
    rejected = core.record_tick(now_s=now_s)

    assert not rejected
    assert core.current_frame_count == 0
    assert core.drop_count == 20
    assert "action" in core.last_drop_reason
    assert "0.500s" in core.last_drop_reason
    assert core.last_error == core.last_drop_reason
    assert len(core.failed_episodes) == 1
    assert core.failed_episodes[0].status is EpisodeStatus.failed
    assert core.failed_episodes[0].metadata["save_error"] == core.last_drop_reason
    with pytest.raises(RuntimeError, match="no episode"):
        core.end_episode()


def test_record_tick_aborts_when_recent_camera_age_average_exceeds_28hz_budget():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache)
    core.start_episode(task="pick cup")

    for index in range(29):
        now_s = 10.0 + index / 30.0
        update_caches(status_cache, action_cache, image_cache, received_s=now_s - 0.036)
        assert core.record_tick(now_s=now_s)

    now_s = 10.0 + 29 / 30.0
    update_caches(status_cache, action_cache, image_cache, received_s=now_s - 0.036)
    rejected = core.record_tick(now_s=now_s)

    assert not rejected
    assert core.current_frame_count == 0
    assert len(core.failed_episodes) == 1
    assert len(core.failed_episodes[0].frames) == 29
    assert "top" in core.last_error
    assert "average image age" in core.last_error
    assert "1/28s" in core.last_error


def test_record_tick_aborts_when_one_camera_age_average_exceeds_28hz_budget():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache)
    core.start_episode(task="pick cup")

    for index in range(29):
        now_s = 30.0 + index / 30.0
        update_caches_with_image_ages(
            status_cache,
            action_cache,
            image_cache,
            now_s,
            {"top": 0.020, "left_wrist": 0.036, "right_wrist": 0.020},
        )
        assert core.record_tick(now_s=now_s)

    now_s = 30.0 + 29 / 30.0
    update_caches_with_image_ages(
        status_cache,
        action_cache,
        image_cache,
        now_s,
        {"top": 0.020, "left_wrist": 0.036, "right_wrist": 0.020},
    )
    rejected = core.record_tick(now_s=now_s)

    assert not rejected
    assert len(core.failed_episodes) == 1
    assert len(core.failed_episodes[0].frames) == 29
    assert "left_wrist" in core.last_error
    assert "top" not in core.last_error
    assert "right_wrist" not in core.last_error


def test_record_tick_tracks_recent_camera_age_average_per_camera():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache)
    core.start_episode(task="pick cup")
    keys = ["top", "left_wrist", "right_wrist"]

    for index in range(30):
        now_s = 40.0 + index / 30.0
        delayed_key = keys[index % len(keys)]
        ages = {"top": 0.005, "left_wrist": 0.005, "right_wrist": 0.005}
        ages[delayed_key] = 0.090
        update_caches_with_image_ages(status_cache, action_cache, image_cache, now_s, ages)
        assert core.record_tick(now_s=now_s)

    sealed = core.end_episode()

    assert len(sealed.frames) == 30
    assert core.last_error is None


def test_record_tick_keeps_recording_when_recent_camera_age_average_is_within_budget():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache)
    core.start_episode(task="pick cup")

    for index in range(30):
        now_s = 20.0 + index / 30.0
        update_caches(status_cache, action_cache, image_cache, received_s=now_s - 0.030)
        assert core.record_tick(now_s=now_s)

    sealed = core.end_episode()

    assert len(sealed.frames) == 30
    assert core.last_error is None


def test_episode_buffer_seals_frames_and_rejects_late_appends():
    buffer = EpisodeBuffer(episode_index=7, task="place cup")
    frame = {"frame_index": 0}

    buffer.append(frame)
    sealed = buffer.seal()

    assert sealed.episode_index == 7
    assert sealed.task == "place cup"
    assert sealed.frames == [frame]
    assert sealed.status is EpisodeStatus.queued
    with pytest.raises(RuntimeError, match="sealed"):
        buffer.append({"frame_index": 1})


def test_episode_buffer_spills_completed_chunks_to_tmp(tmp_path):
    buffer = EpisodeBuffer(
        episode_index=7,
        task="place cup",
        spill_chunk_size=2,
        spill_root=tmp_path,
    )

    for frame_index in range(5):
        buffer.append({"frame_index": frame_index, "images": {"top": rgb(frame_index)}})

    assert buffer.current_frame_count == 5
    sealed = buffer.seal()

    assert isinstance(sealed.frames, SpilledFrameSequence)
    assert len(sealed.frames) == 5
    assert len(sealed.frames.chunk_paths) == 3
    assert all(path.exists() for path in sealed.frames.chunk_paths)
    assert [frame["frame_index"] for frame in sealed.frames] == [0, 1, 2, 3, 4]

    sealed.cleanup()

    assert not any(path.exists() for path in sealed.frames.chunk_paths)


def test_save_job_queue_tracks_queued_and_claimed_status():
    queue = SaveJobQueue()
    sealed = SealedEpisode(episode_index=2, task="lift cube", frames=[{"frame_index": 0}])

    queued = queue.enqueue(sealed)

    assert queued is sealed
    assert queue.list_status() == [
        {
            "episode_index": 2,
            "task": "lift cube",
            "frame_count": 1,
            "status": "queued",
        }
    ]
    assert queue.get_nowait() is sealed
    assert sealed.status is EpisodeStatus.saving
    assert queue.list_status()[0]["status"] == "saving"
    with pytest.raises(queue_module.Empty):
        queue.get_nowait()


def test_save_job_queue_get_nowait_claims_under_status_lock():
    class NotifyingQueue(queue_module.Queue):
        def __init__(self):
            super().__init__()
            self.got = threading.Event()

        def get_nowait(self):
            item = super().get_nowait()
            self.got.set()
            return item

    save_queue = SaveJobQueue()
    save_queue._queue = NotifyingQueue()
    sealed = SealedEpisode(episode_index=4, task="stack cube", frames=[])
    save_queue.enqueue(sealed)

    claimed = []
    errors = []
    started = threading.Event()
    finished = threading.Event()
    save_queue._lock.acquire()

    def claim():
        started.set()
        try:
            claimed.append(save_queue.get_nowait())
        except Exception as exc:  # pragma: no cover - reported by assertion below
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=claim)
    thread.start()
    try:
        assert started.wait(timeout=1.0)
        got_while_locked = save_queue._queue.got.wait(timeout=0.05)
        assert not finished.is_set()
    finally:
        save_queue._lock.release()
        assert finished.wait(timeout=1.0)
        thread.join(timeout=1.0)

    assert got_while_locked is False
    assert errors == []
    assert claimed == [sealed]
    assert sealed.status is EpisodeStatus.saving


def test_episode_buffer_snapshots_mutable_frame_data():
    buffer = EpisodeBuffer(episode_index=3, task="sort blocks")
    original = {
        "timestamp": 1.0,
        "frame_index": 0,
        "observation.state": np.arange(16, dtype=np.float32),
        "action": np.arange(100, 116, dtype=np.float32),
        "images": {
            "top": rgb(10),
            "left_wrist": rgb(20),
            "right_wrist": rgb(30),
        },
        "labels": ["keep", {"scores": [1, 2]}],
    }

    buffer.append(original)
    original["observation.state"][0] = 999
    original["action"][0] = 888
    original["images"]["top"][0, 0, 0] = 77
    original["labels"][1]["scores"].append(3)
    sealed = buffer.seal()
    original["images"]["left_wrist"][0, 0, 0] = 66
    original["labels"][0] = "changed"

    sealed_frame = sealed.frames[0]
    assert sealed_frame is not original
    assert sealed_frame["observation.state"][0] == 0
    assert sealed_frame["action"][0] == 100
    assert sealed_frame["images"]["top"][0, 0, 0] == 10
    assert sealed_frame["images"]["left_wrist"][0, 0, 0] == 20
    assert sealed_frame["labels"] == ["keep", {"scores": [1, 2]}]


def test_record_tick_returns_copy_that_cannot_corrupt_sealed_episode():
    status_cache, action_cache, image_cache = fresh_caches(received_s=1.0)
    core = RecorderCore(status_cache, action_cache, image_cache)
    core.start_episode(task="pick cup")

    returned_frame = core.record_tick(now_s=1.02)
    returned_frame["observation.state"][0] = 999
    returned_frame["action"][0] = 888
    returned_frame["images"]["top"][0, 0, 0] = 77
    sealed = core.end_episode()

    sealed_frame = sealed.frames[0]
    assert returned_frame is not sealed_frame
    assert returned_frame["images"] is not sealed_frame["images"]
    assert sealed_frame["observation.state"][0] == 20
    assert sealed_frame["action"][0] == 200
    assert sealed_frame["images"]["top"][0, 0, 0] == 10
