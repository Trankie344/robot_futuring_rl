import dataclasses

import pytest

from openpi.training.arm_value.config import get_config


def test_hil_config_exposes_all_requested_model_parameters():
    config = get_config("arm_value_hil_pico_v21")
    assert config.model.clip_pretrained_path == "./checkpoints/clip-vit-base-patch32"
    assert config.model.clip_local_files_only
    assert config.model.n_history_steps == 4
    assert config.model.frame_gap == 30
    assert config.model.max_state_dim == 32
    assert config.model.hidden_dim == 768
    assert config.model.num_heads == 12
    assert config.model.num_layers == 8
    assert config.model.lambda_interval == 1.0
    assert config.model.lambda_cls == 1.0
    assert config.device == "cuda"

    overridden = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            n_history_steps=6,
            frame_gap=15,
            max_state_dim=48,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            lambda_interval=1.5,
            lambda_cls=0.5,
        ),
    )
    assert overridden.model.sequence_length == 7
    assert overridden.model.frame_gap == 15
    assert overridden.model.max_state_dim == 48


def test_unknown_arm_value_config_has_suggestion():
    with pytest.raises(ValueError, match="Did you mean"):
        get_config("arm_value_hil_pico")
