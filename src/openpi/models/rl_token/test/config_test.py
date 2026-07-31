import pytest

from openpi.models.rl_token.config import RLTokenPi0Config


def test_stage1_model_defaults_and_fixed_action_horizon():
    config = RLTokenPi0Config(
        pi05=True,
        rl_token_enabled=True,
        rl_token_only=True,
        rl_token_reconstruction_weight=1.0,
    )

    assert config.action_horizon == 50
    assert config.rl_token_encoder_depth == 2
    assert config.rl_token_decoder_depth == 2
    assert config.rl_token_num_heads == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rl_token_only": True},
        {
            "rl_token_enabled": True,
            "rl_token_only": True,
            "rl_token_reconstruction_weight": 0.0,
        },
        {"rl_token_dropout": 1.0},
        {"rl_token_compute_dtype": "float16"},
    ],
)
def test_invalid_rl_token_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RLTokenPi0Config(**kwargs)
