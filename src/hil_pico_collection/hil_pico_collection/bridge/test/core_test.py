from types import SimpleNamespace

import numpy as np

from hil_pico_collection.adapters.base import RobotStateSample
from hil_pico_collection.bridge.core import RLTokenBridgeCore
from hil_pico_collection.recording.cache import LatestImageCache, LatestValueCache


class Adapter:
    def parse_status(self, message):
        return RobotStateSample(
            np.asarray(message.state, np.float32),
            message.mode,
            message.intervention,
            message.enabled,
        )

    def build_command(self, action, stamp, frame_id):
        return (np.asarray(action), stamp, frame_id)


class Policy:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def infer(self, observation):
        self.requests.append(observation)
        if self.fail:
            raise ConnectionError("offline")
        return {"actions": np.zeros((20, 16), np.float32)}


def caches(enabled=True, intervention=False):
    status = LatestValueCache()
    images = LatestImageCache(("top", "left_wrist", "right_wrist"))
    status.update(SimpleNamespace(state=np.arange(16), mode=1, enabled=enabled, intervention=intervention))
    for key in ("top", "left_wrist", "right_wrist"):
        images.update(key, np.zeros((2, 3, 3), np.uint8))
    return status, images


def core(status, images, policy, published):
    now = [0.0]

    def monotonic():
        return now[0]

    def sleep(seconds):
        now[0] += seconds

    return RLTokenBridgeCore(
        status_cache=status,
        image_cache=images,
        adapter=Adapter(),
        policy_client=policy,
        publish_command=published.append,
        stamp=lambda: "stamp",
        prompt="fold clothes",
        monotonic=monotonic,
        sleep=sleep,
    )


def test_infer_and_execute_sends_exact_observation_and_full_chunk():
    status, images = caches()
    policy = Policy()
    published = []
    result = core(status, images, policy, published).infer_and_execute()
    assert result.completed is True
    assert len(published) == 30
    assert set(policy.requests[0]) == {"images", "state", "prompt"}
    assert policy.requests[0]["state"].tolist() == list(range(16))


def test_no_inference_without_model_control():
    status, images = caches(enabled=False)
    policy = Policy()
    result = core(status, images, policy, []).infer_and_execute()
    assert result.sent_count == 0
    assert policy.requests == []


def test_no_inference_during_intervention():
    status, images = caches(intervention=True)
    policy = Policy()
    result = core(status, images, policy, []).infer_and_execute()
    assert result.sent_count == 0
    assert policy.requests == []


def test_inference_disconnect_sends_no_commands():
    status, images = caches()
    result = core(status, images, Policy(fail=True), []).infer_and_execute()
    assert result.sent_count == 0
    assert "offline" in result.reason


def test_intervention_mid_chunk_discards_remaining_commands():
    status, images = caches()
    published = []
    bridge = core(status, images, Policy(), published)

    def publish(message):
        published.append(message)
        if len(published) == 5:
            status.update(SimpleNamespace(state=np.arange(16), mode=5, enabled=False, intervention=True))

    bridge.publish_command = publish
    result = bridge.infer_and_execute()
    assert result.completed is False
    assert result.sent_count == 5


def test_fresh_inference_after_control_resumes():
    status, images = caches(enabled=False)
    policy = Policy()
    bridge = core(status, images, policy, [])
    assert bridge.infer_and_execute().sent_count == 0
    status.update(SimpleNamespace(state=np.arange(16) + 1, mode=1, enabled=True, intervention=False))
    assert bridge.infer_and_execute().completed is True
    assert policy.requests[-1]["state"][0] == 1
