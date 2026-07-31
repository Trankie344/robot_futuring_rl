from pathlib import Path

INTERFACE_ROOT = Path(__file__).resolve().parents[1] / "msg"


def _message(name: str) -> str:
    return (INTERFACE_ROOT / name).read_text(encoding="utf-8")


def test_pico_state_is_one_synchronized_aggregate() -> None:
    source = _message("PicoState.msg")
    assert "std_msgs/Header header" in source
    assert "builtin_interfaces/Time receipt_stamp" in source
    assert "string device_id" in source
    assert "pico_tele_interfaces/TrackedPose head" in source
    assert "pico_tele_interfaces/ControllerState left_controller" in source
    assert "pico_tele_interfaces/ControllerState right_controller" in source


def test_operator_commands_remain_robot_agnostic() -> None:
    source = _message("OperatorCommand.msg")
    assert "COMMAND_RESET=1" in source
    assert "COMMAND_TOGGLE_CONTROL_MODE=2" in source
    assert "arm_interfaces" not in source
