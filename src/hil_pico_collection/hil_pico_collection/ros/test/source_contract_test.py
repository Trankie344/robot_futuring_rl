from pathlib import Path

from hil_pico_collection.ros import recorder_node

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_source_uses_only_canonical_topics():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.rglob("*.py") if "test" not in path.parts
    )
    assert "/auto_arm_cmd" in source
    assert "/arm_status" in source
    assert "/pico_tele/reset_request" in source
    assert "/change_ctrl_mode" in source
    assert "/execute_arm_cmd" not in source


def test_forbidden_legacy_features_are_not_migrated():
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.rglob("*.py") if "test" not in path.parts
    )
    for forbidden in ("send_pyobj", "recv_pyobj", "VLA_GRANT", "input_events", "pedal_mode_switcher"):
        assert forbidden not in source


def test_dataset_root_defaults_under_external_runtime(monkeypatch):
    monkeypatch.setenv("HIL_PICO_RUNTIME", "/tmp/hil-runtime")
    assert recorder_node.default_dataset_root() == "/tmp/hil-runtime/datasets/hil_pico_v21"
    assert recorder_node.resolve_dataset_root(None, today="20260731").endswith("hil_pico_v21_20260731")


def test_fixed_recording_rate_is_30hz():
    assert recorder_node.DEFAULT_FPS == 30


def test_ros_nodes_expose_shared_robot_config_option():
    recorder_args = recorder_node.build_arg_parser().parse_args([])
    assert recorder_args.robot_config.endswith("config/zme_dual_arm.yaml")
    source = "\n".join(path.read_text(encoding="utf-8") for path in (PACKAGE_ROOT / "ros").glob("*.py"))
    assert "--robot-config" in source
    assert "protocol.status_topic" in source
    assert "protocol.command_topic" in source
