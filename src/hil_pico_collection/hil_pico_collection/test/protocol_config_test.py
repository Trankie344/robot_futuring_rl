from copy import deepcopy

import pytest
import yaml

from hil_pico_collection.protocol_config import (
    default_robot_config_path,
    load_robot_protocol_config,
    parse_robot_protocol_config,
)


def raw_zme():
    return yaml.safe_load(default_robot_config_path().read_text(encoding="utf-8"))


def test_bundled_zme_config_locks_current_training_contract():
    config = load_robot_protocol_config()
    assert config.name == "zme_dual_arm_rl_token"
    assert config.state.dimension == 16
    assert config.action.dimension == 16
    assert config.state.order[7] == "left_gripper"
    assert config.state.order[15] == "right_gripper"
    assert config.action_horizon == 20
    assert config.command_count == 30
    assert config.image_names == ("top", "left_wrist", "right_wrist")
    assert all(image.shape == (480, 640, 3) for image in config.images)


def test_default_config_path_is_packaged_with_module():
    path = default_robot_config_path()
    assert path.name == "zme_dual_arm.yaml"
    assert path.is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update(schema_version=99),
        lambda raw: raw["state"].update(dimension=15),
        lambda raw: raw["action"].update(dimension=15),
        lambda raw: raw["images"]["top"].update(channels=1),
        lambda raw: raw["status"].update(model_control_enabled=[]),
    ],
)
def test_config_rejects_incompatible_contracts(mutation):
    raw = raw_zme()
    mutation(raw)
    with pytest.raises(ValueError):
        parse_robot_protocol_config(raw)


def test_config_requires_bindings_for_every_ordered_state():
    raw = raw_zme()
    del raw["state"]["sources"]["right_joint_0"]
    with pytest.raises(ValueError, match="keys must match"):
        parse_robot_protocol_config(raw)


def test_config_rejects_unknown_action_limit_name():
    raw = raw_zme()
    raw["action"]["limits"]["unknown"] = {"min": 0, "max": 1}
    with pytest.raises(ValueError, match="unknown"):
        parse_robot_protocol_config(raw)


def test_loading_missing_config_reports_path(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match=str(path)):
        load_robot_protocol_config(path)


def test_config_copy_can_change_image_names_sizes_and_model_dimensions():
    raw = deepcopy(raw_zme())
    raw["state"] = {"dimension": 2, "order": ["joint_b", "joint_a"], "sources": {"joint_b": "b", "joint_a": "a"}}
    raw["action"] = {
        "dimension": 1,
        "order": ["target"],
        "observed_sources": {"target": "executed[0]"},
        "command_targets": {"target": "command[0]"},
        "limits": {"target": {"min": -1, "max": 1}},
    }
    raw["images"] = {
        "front_rgb": {
            "topic": "/front",
            "message_type": "sensor_msgs.msg:Image",
            "width": 320,
            "height": 240,
            "channels": 3,
            "resize": True,
        }
    }
    raw["inference"]["action_horizon"] = 8
    config = parse_robot_protocol_config(raw)
    assert config.state.order == ("joint_b", "joint_a")
    assert config.action.order == ("target",)
    assert config.image_names == ("front_rgb",)
    assert config.images[0].shape == (240, 320, 3)
    assert config.action_horizon == 8
