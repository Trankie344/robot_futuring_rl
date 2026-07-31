from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]


def test_scripts_do_not_contain_legacy_control_paths():
    source = "\n".join(path.read_text(encoding="utf-8") for path in SCRIPTS.glob("*.sh"))
    assert "/execute_arm_cmd" not in source
    assert "pedal" not in source.lower()
    assert "grant" not in source.lower()
    assert "zmq" not in source.lower()


def test_stack_requires_prompt_and_supports_dataset_override():
    source = (SCRIPTS / "run_hil_stack.sh").read_text(encoding="utf-8")
    assert "--prompt is required" in source
    assert "--dataset-root" in source
    assert "--robot-config" in source
    assert "--robot-adapter" in source


def test_bridge_and_recorder_use_standalone_package():
    assert "hil_pico_collection.ros.recorder_node" in (SCRIPTS / "run_recorder.sh").read_text()
    assert "hil_pico_collection.bridge.ros_node" in (SCRIPTS / "run_rl_token_bridge.sh").read_text()
