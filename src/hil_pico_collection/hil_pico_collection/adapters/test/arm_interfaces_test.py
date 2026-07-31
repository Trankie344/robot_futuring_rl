from types import SimpleNamespace

import numpy as np
import pytest

from hil_pico_collection.adapters.arm_interfaces import (
    AUTONOMY_JOINT,
    ArmInterfacesAdapter,
    ArmLimits,
)
from hil_pico_collection.adapters.base import CANONICAL_NAMES, RobotStateSample


class Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class JointCommand:
    def __init__(self):
        self.header = Header()
        self.left_joint_command = []
        self.right_joint_command = []
        self.gripper_command = [0.0] * 7


class ShortCommand:
    def __init__(self):
        self.header = Header()
        self.left_command = []
        self.right_command = []
        self.gripper_command = []


class Service:
    class Request:
        pass


def adapter(command_type=JointCommand, limits=None):
    return ArmInterfacesAdapter(
        status_message_type=object,
        command_message_type=command_type,
        mode_service_type=Service,
        secondary_service_type=Service,
        limits=limits,
    )


def joint(value):
    return SimpleNamespace(joint_status=[value])


def status(primary=AUTONOMY_JOINT, secondary=AUTONOMY_JOINT):
    return SimpleNamespace(
        left_arm=[joint(10 + i) for i in range(7)],
        right_arm=[joint(20 + i) for i in range(7)],
        gripper=[joint(30), joint(31)],
        other_status=[primary, secondary],
    )


def test_canonical_names_lock_right_left_gripper_order():
    assert len(CANONICAL_NAMES) == 16
    assert CANONICAL_NAMES[:7] == tuple(f"right_joint_{i}" for i in range(7))
    assert CANONICAL_NAMES[7] == "left_gripper"
    assert CANONICAL_NAMES[8:15] == tuple(f"left_joint_{i}" for i in range(7))
    assert CANONICAL_NAMES[15] == "right_gripper"


def test_parse_status_uses_training_order_and_physical_grippers():
    sample = adapter().parse_status(status())
    np.testing.assert_array_equal(
        sample.state,
        np.asarray([20, 21, 22, 23, 24, 25, 26, 30, 10, 11, 12, 13, 14, 15, 16, 31], np.float32),
    )


@pytest.mark.parametrize(
    ("primary", "secondary", "intervention", "enabled"),
    [
        (1, 1, False, True),
        (1, 0, False, False),
        (5, 0, True, False),
        (0, 0, False, False),
    ],
)
def test_parse_status_mode_semantics(primary, secondary, intervention, enabled):
    sample = adapter().parse_status(status(primary, secondary))
    assert sample.control_mode == primary
    assert sample.intervention is intervention
    assert sample.model_control_enabled is enabled


def test_parse_status_accepts_scalar_joint_standins():
    message = SimpleNamespace(
        left_arm=list(range(7)),
        right_arm=list(range(10, 17)),
        gripper=[20, 21],
        other_status=[1, 1],
    )
    assert adapter().parse_status(message).state.tolist() == [10, 11, 12, 13, 14, 15, 16, 20, 0, 1, 2, 3, 4, 5, 6, 21]


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(left_arm=[1], right_arm=range(7), gripper=[1, 2], other_status=[1, 1]),
        SimpleNamespace(left_arm=range(7), right_arm=range(7), gripper=[1], other_status=[1, 1]),
        SimpleNamespace(left_arm=range(7), right_arm=range(7), gripper=[1, 2], other_status=[1]),
    ],
)
def test_parse_status_rejects_short_fields(message):
    with pytest.raises(ValueError):
        adapter().parse_status(message)


def test_parse_status_rejects_nonfinite_joint():
    message = status()
    message.right_arm[3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        adapter().parse_status(message)


@pytest.mark.parametrize(
    "left_field,right_field", [("left_joint_command", "right_joint_command"), ("left_command", "right_command")]
)
def test_parse_executed_action_accepts_both_command_field_variants(left_field, right_field):
    message = SimpleNamespace(gripper_command=[300, 301])
    setattr(message, left_field, list(range(100, 107)))
    setattr(message, right_field, list(range(200, 207)))
    action = adapter().parse_executed_action(message)
    assert action.tolist() == [200, 201, 202, 203, 204, 205, 206, 300, 100, 101, 102, 103, 104, 105, 106, 301]


@pytest.mark.parametrize(
    "command_type,left_field,right_field",
    [(JointCommand, "left_joint_command", "right_joint_command"), (ShortCommand, "left_command", "right_command")],
)
def test_build_command_inverts_canonical_mapping(command_type, left_field, right_field):
    message = adapter(command_type).build_command(np.arange(16), "stamp", "frame")
    assert message.header.stamp == "stamp"
    assert message.header.frame_id == "frame"
    assert getattr(message, right_field) == pytest.approx(range(7))
    assert getattr(message, left_field) == pytest.approx(range(8, 15))
    assert message.gripper_command[:2] == pytest.approx([7, 15])


@pytest.mark.parametrize("bad", [np.zeros(15), np.zeros((1, 16)), np.full(16, np.inf)])
def test_build_command_rejects_invalid_actions(bad):
    with pytest.raises(ValueError):
        adapter().build_command(bad, None, "")


def test_soft_limits_reject_out_of_range_action():
    limits = ArmLimits(lower=np.full(16, -1), upper=np.full(16, 1))
    with pytest.raises(ValueError, match="upper"):
        adapter(limits=limits).build_command(np.full(16, 2), None, "")


def test_robot_state_sample_copies_input_and_rejects_invalid_shape():
    state = np.zeros(16, dtype=np.float32)
    sample = RobotStateSample(state, 1, False, True)
    state[0] = 9
    assert sample.state[0] == 0
    with pytest.raises(ValueError, match="vector"):
        RobotStateSample(np.zeros((1, 15)), 1, False, True)
