from copy import deepcopy

import numpy as np
import pytest
import yaml

from hil_pico_collection.bridge.execution import resample_action_chunk
from hil_pico_collection.bridge.protocol import build_observation, validate_inference_response, validate_server_metadata
from hil_pico_collection.protocol_config import default_robot_config_path, parse_robot_protocol_config


def protocol():
    raw = yaml.safe_load(default_robot_config_path().read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["state"] = {"dimension": 3, "order": ["s2", "s0", "s1"], "sources": {"s2": "s[2]", "s0": "s[0]", "s1": "s[1]"}}
    raw["action"] = {
        "dimension": 2,
        "order": ["a1", "a0"],
        "observed_sources": {"a1": "a[1]", "a0": "a[0]"},
        "command_targets": {"a1": "out[1]", "a0": "out[0]"},
        "limits": {},
    }
    raw["images"] = {
        "front": {
            "topic": "/front",
            "message_type": "fake:Image",
            "width": 5,
            "height": 4,
            "channels": 3,
        }
    }
    raw["inference"] = {"action_horizon": 4, "model_hz": 4, "command_hz": 6}
    return parse_robot_protocol_config(raw)


def test_dynamic_protocol_validates_metadata_dimensions():
    metadata = {
        "rlt_stage2": {
            "round_complete": True,
            "network_config": {"state_dim": 3, "action_dim": 2, "action_horizon": 4},
        }
    }
    assert validate_server_metadata(metadata, protocol()) == metadata
    metadata["rlt_stage2"]["network_config"]["action_dim"] = 16
    with pytest.raises(ValueError, match="action_dim"):
        validate_server_metadata(metadata, protocol())


def test_dynamic_protocol_builds_named_sized_observation():
    observation = build_observation(
        {"front": np.zeros((4, 5, 3), np.uint8)},
        np.zeros(3),
        "task",
        protocol(),
    )
    assert set(observation["images"]) == {"front"}
    with pytest.raises(ValueError, match="configured shape"):
        build_observation({"front": np.zeros((8, 8, 3), np.uint8)}, np.zeros(3), "task", protocol())


def test_dynamic_protocol_validates_action_horizon_and_dimension():
    result = validate_inference_response({"actions": np.zeros((4, 2))}, protocol())
    assert result["actions"].shape == (4, 2)
    with pytest.raises(ValueError, match="shape"):
        validate_inference_response({"actions": np.zeros((20, 16))}, protocol())


def test_dynamic_resampling_uses_configured_dimensions():
    actions = np.arange(8, dtype=np.float32).reshape(4, 2)
    result = resample_action_chunk(actions, 6, expected_horizon=4, expected_action_dim=2)
    assert result.shape == (6, 2)
    np.testing.assert_array_equal(result[0], actions[0])
    np.testing.assert_array_equal(result[-1], actions[-1])
