from types import SimpleNamespace

import numpy as np
import pytest

from hil_pico_collection.adapters.configured import ConfiguredRobotAdapter, read_path, write_path
from hil_pico_collection.protocol_config import parse_robot_protocol_config


class Status:
    pass


class Command:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.joint_targets = []
        self.gripper = [0.0]


def raw_config():
    return {
        "schema_version": 1,
        "name": "test_robot",
        "robot": {
            "robot_type": "test",
            "status_topic": "/status",
            "command_topic": "/command",
            "reset_request_topic": "/reset",
            "change_control_mode_topic": "/mode",
            "status_message_type": "fake:Status",
            "command_message_type": "fake:Command",
            "header_stamp_path": "header.stamp",
            "header_frame_id_path": "header.frame_id",
        },
        "state": {
            "dimension": 3,
            "order": ["joint_b", "gripper", "joint_a"],
            "sources": {
                "joint_b": ["missing", "joints[1]"],
                "gripper": "gripper",
                "joint_a": "joints[0]",
            },
        },
        "action": {
            "dimension": 2,
            "order": ["joint_target", "gripper_target"],
            "observed_sources": {
                "joint_target": "executed[0]",
                "gripper_target": "executed_gripper",
            },
            "command_targets": {
                "joint_target": "joint_targets[0]",
                "gripper_target": "gripper[0]",
            },
            "limits": {
                "joint_target": {"min": -2, "max": 2},
                "gripper_target": {"min": 0, "max": 1},
            },
        },
        "status": {
            "control_mode": "mode.primary",
            "intervention": {"path": "mode.primary", "equals": 5},
            "model_control_enabled": [
                {"path": "mode.primary", "equals": 1},
                {"path": "mode.secondary", "equals": 1},
            ],
        },
        "images": {
            "front": {
                "topic": "/front",
                "message_type": "fake:Image",
                "width": 4,
                "height": 3,
                "channels": 3,
            }
        },
        "inference": {"action_horizon": 4, "model_hz": 4, "command_hz": 6},
        "mode_controller": {"factory": "fake:mode_factory", "options": {"answer": 42}},
    }


def resolver(spec):
    symbols = {
        "fake:Status": Status,
        "fake:Command": Command,
        "fake:Image": object,
        "fake:mode_factory": lambda **kwargs: kwargs,
    }
    return symbols[spec]


def adapter():
    return ConfiguredRobotAdapter(parse_robot_protocol_config(raw_config()), symbol_resolver=resolver)


def status(primary=1, secondary=1):
    return SimpleNamespace(
        joints=[10.0, 20.0],
        gripper=30.0,
        mode=SimpleNamespace(primary=primary, secondary=secondary),
    )


def test_configured_adapter_maps_state_in_declared_model_order():
    sample = adapter().parse_status(status())
    assert sample.state.tolist() == [20.0, 30.0, 10.0]
    assert sample.model_control_enabled is True


@pytest.mark.parametrize(
    ("primary", "secondary", "intervention", "enabled"),
    [(5, 0, True, False), (1, 0, False, False), (1, 1, False, True)],
)
def test_configured_adapter_uses_declared_mode_conditions(primary, secondary, intervention, enabled):
    sample = adapter().parse_status(status(primary, secondary))
    assert sample.intervention is intervention
    assert sample.model_control_enabled is enabled


def test_configured_adapter_maps_executed_action_and_builds_command():
    configured = adapter()
    executed = SimpleNamespace(executed=[0.5], executed_gripper=0.75)
    np.testing.assert_array_equal(configured.parse_executed_action(executed), [0.5, 0.75])
    command = configured.build_command([1.5, 0.25], "stamp", "frame")
    assert command.joint_targets == [1.5]
    assert command.gripper == [0.25]
    assert command.header.stamp == "stamp"
    assert command.header.frame_id == "frame"


@pytest.mark.parametrize("action", [[3.0, 0.5], [0.0, -0.1], [0.0, 1.1]])
def test_configured_adapter_enforces_per_field_limits(action):
    with pytest.raises(ValueError):
        adapter().build_command(action, None, "")


def test_configured_adapter_invokes_configured_mode_factory():
    controller = adapter().create_mode_controller("node")
    assert controller["node"] == "node"
    assert controller["options"] == {"answer": 42}


def test_field_path_helpers_support_attributes_dicts_and_list_growth():
    root = SimpleNamespace(data={"values": [1]})
    assert read_path(root, "data.values[0]") == 1
    write_path(root, "data.values[2]", 3)
    assert root.data["values"] == [1, 0.0, 3]


def test_field_path_helpers_reject_invalid_syntax():
    with pytest.raises(ValueError, match="invalid field path"):
        read_path(SimpleNamespace(), "field[-1]")
