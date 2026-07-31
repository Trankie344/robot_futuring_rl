from types import SimpleNamespace

import numpy as np
import pytest

from hil_pico_collection.bridge.protocol import (
    RLTokenPolicyClient,
    build_observation,
    validate_inference_response,
    validate_server_metadata,
)


def metadata(**network_overrides):
    network = {"state_dim": 16, "action_dim": 16, "action_horizon": 20}
    network.update(network_overrides)
    return {"rlt_stage2": {"round_complete": True, "network_config": network}}


def images():
    return {
        "top": np.zeros((4, 5, 3), np.uint8),
        "left_wrist": np.ones((4, 5, 3), np.uint8),
        "right_wrist": np.full((4, 5, 3), 2, np.uint8),
    }


def test_validate_server_metadata_accepts_completed_stage2():
    result = validate_server_metadata(metadata())
    assert result["rlt_stage2"]["round_complete"] is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"rlt_stage2": None},
        {"rlt_stage2": {"round_complete": False, "network_config": {}}},
    ],
)
def test_validate_server_metadata_rejects_missing_or_incomplete(value):
    with pytest.raises(ValueError):
        validate_server_metadata(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("state_dim", 15), ("action_dim", 32), ("action_horizon", 50)],
)
def test_validate_server_metadata_rejects_wrong_interface(field, value):
    with pytest.raises(ValueError, match=field):
        validate_server_metadata(metadata(**{field: value}))


def test_build_observation_returns_exact_wire_keys_and_copies_state():
    state = np.arange(16, dtype=np.float32)
    result = build_observation(images(), state, "fold clothes")
    assert set(result) == {"images", "state", "prompt"}
    assert set(result["images"]) == {"top", "left_wrist", "right_wrist"}
    state[0] = 100
    assert result["state"][0] == 0


@pytest.mark.parametrize(
    "bad_images",
    [
        {},
        {"top": np.zeros((2, 2, 3), np.uint8)},
        {**images(), "extra": np.zeros((2, 2, 3), np.uint8)},
        {**images(), "top": np.zeros((2, 2), np.uint8)},
        {**images(), "top": np.zeros((2, 2, 3), np.float32)},
    ],
)
def test_build_observation_rejects_invalid_images(bad_images):
    with pytest.raises(ValueError):
        build_observation(bad_images, np.zeros(16), "task")


@pytest.mark.parametrize("state", [np.zeros(15), np.zeros((1, 16)), np.full(16, np.nan)])
def test_build_observation_rejects_invalid_state(state):
    with pytest.raises(ValueError):
        build_observation(images(), state, "task")


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_build_observation_rejects_empty_prompt(prompt):
    with pytest.raises(ValueError):
        build_observation(images(), np.zeros(16), prompt)


def test_validate_response_accepts_timing_metadata():
    response = validate_inference_response(
        {
            "actions": np.zeros((20, 16)),
            "policy_timing": {"total_ms": 1},
            "server_timing": {"infer_ms": 2},
        }
    )
    assert response["actions"].dtype == np.float32
    assert response["actions"].shape == (20, 16)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"actions": np.zeros((50, 16))},
        {"actions": np.zeros((20, 32))},
        {"actions": np.full((20, 16), np.inf)},
    ],
)
def test_validate_response_rejects_malformed_actions(response):
    with pytest.raises(ValueError):
        validate_inference_response(response)


class FakeWireClient:
    def __init__(self, host, port, api_key):
        self.arguments = (host, port, api_key)
        self._ws = SimpleNamespace(close=lambda: setattr(self, "closed", True))
        self.closed = False

    def get_server_metadata(self):
        return metadata()

    def infer(self, observation):
        return {"actions": np.ones((20, 16))}


def test_policy_client_validates_metadata_response_and_close():
    wire = []

    def factory(**kwargs):
        client = FakeWireClient(**kwargs)
        wire.append(client)
        return client

    client = RLTokenPolicyClient("host", 8011, api_key="key", client_factory=factory)
    assert client.infer({"test": True})["actions"].shape == (20, 16)
    client.close()
    assert wire[0].closed is True
