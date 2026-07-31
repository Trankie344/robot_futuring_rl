import pytest

from openpi.training.rl_token import config


def test_config_registry_has_only_requested_names():
    assert set(config._CONFIGS_DICT) == {
        "rl_token_stage1",
        "rl_token_stage1_debug",
        "rl_token_stage2",
        "rl_token_stage2_debug",
    }


def test_stage1_production_hyperparameters():
    value = config.get_stage1_config("rl_token_stage1")

    assert value.batch_size == 256
    assert value.num_workers == 16
    assert value.num_train_steps == 30_000
    assert value.save_interval == 5_000
    assert value.keep_period == 5_000
    assert value.lr_schedule.warmup_steps == 10_000
    assert value.lr_schedule.peak_lr == 5e-5
    assert value.lr_schedule.decay_lr == 5e-5


def test_cross_stage_lookup_is_rejected():
    with pytest.raises(ValueError, match="not a Stage 1"):
        config.get_stage1_config("rl_token_stage2")
    with pytest.raises(ValueError, match="not a Stage 2"):
        config.get_stage2_config("rl_token_stage1")
