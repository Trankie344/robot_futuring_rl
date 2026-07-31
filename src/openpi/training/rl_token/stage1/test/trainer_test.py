from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from openpi.shared import nnx_utils
from openpi.training import weight_loaders
from openpi.training.rl_token import config as rl_token_config
from openpi.training.rl_token.stage1 import trainer


def _state() -> nnx.State:
    return nnx.State(
        {
            "PaliGemma": {
                "kernel": nnx.VariableState(nnx.Param, jnp.ones((2, 2), dtype=jnp.float32))
            },
            "rl_token": {
                "encoder": {
                    "kernel": nnx.VariableState(nnx.Param, jnp.full((2, 2), 2.0, dtype=jnp.float32))
                },
                "rl_token": nnx.VariableState(nnx.Param, jnp.full((1, 4), 3.0, dtype=jnp.float32)),
            },
        }
    )


def test_stage1_config_trains_only_exact_rl_token_root():
    config = rl_token_config.get_stage1_config("rl_token_stage1")
    selected = _state().filter(config.trainable_filter).flat_state()

    assert {path[0] for path in selected} == {"rl_token"}
    assert config.model.rl_token_only is True
    assert config.ema_decay == 0.999


def test_frozen_vla_casts_to_bfloat16_while_rl_token_stays_fp32():
    config = rl_token_config.get_stage1_config("rl_token_stage1")
    converted = trainer._cast_frozen_params(_state(), config.freeze_filter)  # noqa: SLF001

    assert converted["PaliGemma"]["kernel"].value.dtype == jnp.bfloat16
    assert converted["rl_token"]["encoder"]["kernel"].value.dtype == jnp.float32


def test_partial_ema_selects_and_updates_only_rl_token_parameters():
    config = rl_token_config.get_stage1_config("rl_token_stage1")
    old = trainer._select_ema_params(_state(), config)  # noqa: SLF001
    new_state = nnx_utils.state_map(
        _state(),
        nnx_utils.PathRegex(r"rl_token(?:/.*)?"),
        lambda variable: variable.replace(variable.value + 2.0),
    )
    new = trainer._select_ema_params(new_state, config)  # noqa: SLF001
    updated = trainer._update_ema_params(old, new, decay=0.5)  # noqa: SLF001

    assert {path[0] for path in updated.flat_state()} == {"rl_token"}
    assert jnp.all(updated["rl_token"]["encoder"]["kernel"].value == 3.0)


def _local_config(tmp_path: Path) -> rl_token_config.RLTokenTrainConfig:
    config = rl_token_config.get_stage1_config("rl_token_stage1")
    dataset = tmp_path / "dataset"
    params = tmp_path / "29999" / "params"
    assets = tmp_path / "29999" / "assets"
    norm = assets / config.data.assets.asset_id / "norm_stats.json"
    dataset.mkdir()
    params.mkdir(parents=True)
    norm.parent.mkdir(parents=True)
    norm.write_text("{}", encoding="utf-8")
    return dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            repo_id=str(dataset),
            assets=dataclasses.replace(config.data.assets, assets_dir=str(assets)),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params)),
        deployment_base_params_path=str(params),
    )


def test_stage1_path_preflight_accepts_complete_local_inputs(tmp_path: Path):
    trainer._validate_rlt_only_paths(_local_config(tmp_path))  # noqa: SLF001


def test_stage1_path_preflight_rejects_checkpoint_identity_mismatch(tmp_path: Path):
    config = dataclasses.replace(
        _local_config(tmp_path),
        deployment_base_params_path=str(tmp_path / "other"),
    )

    with pytest.raises(ValueError, match="must match"):
        trainer._validate_rlt_only_paths(config)  # noqa: SLF001


def test_stage1_config_uses_fp32_optimizer_and_bf16_compute_contract():
    config = rl_token_config.get_stage1_config("rl_token_stage1")

    assert config.model.dtype == "bfloat16"
    assert config.model.rl_token_compute_dtype == "bfloat16"
    assert config.model.rl_token_dropout == 0.1
    assert config.model.rl_token_width == 2048
    assert config.model.rl_token_num_heads == 16
    assert config.model.rl_token_mlp_dim == 8192
