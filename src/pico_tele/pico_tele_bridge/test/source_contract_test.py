from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_core_has_no_robot_specific_arm_dependencies() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PACKAGE_ROOT / "src").glob("*.cpp"))
    )
    assert "arm_interfaces" not in sources
    assert "/tele_vr_cmd" not in sources
    assert "action_manager" not in sources


def test_command_router_uses_generic_topics() -> None:
    source = (PACKAGE_ROOT / "src" / "command_router.cpp").read_text(encoding="utf-8")
    assert '"/pico_tele/reset_request"' in source
    assert '"/change_ctrl_mode"' in source
    assert "std_msgs::msg::Empty" in source
    assert "std_msgs::msg::Bool" in source


def test_sdk_bridge_uses_pxrea_callback_instead_of_reimplementing_tcp() -> None:
    source = (PACKAGE_ROOT / "src" / "pico_sdk_bridge.cpp").read_text(encoding="utf-8")
    assert "PXREAInit" in source
    assert "PXREADeviceStateJson" in source
    assert "63901" not in source


def test_default_config_uses_safe_gestures_and_three_nodes() -> None:
    config = (PACKAGE_ROOT / "config" / "pico_tele.yaml").read_text(encoding="utf-8")
    launch = (PACKAGE_ROOT / "launch" / "pico_tele.launch.py").read_text(encoding="utf-8")
    assert "hold_duration_s: 1.0" in config
    assert "input_timeout_s: 0.25" in config
    assert launch.count("Node(") == 3
