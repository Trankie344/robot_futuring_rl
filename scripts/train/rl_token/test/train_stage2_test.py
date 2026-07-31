from pathlib import Path

import pytest

from openpi.training.rl_token.stage2 import native_training
from scripts.train.rl_token import train_stage2


def _args(tmp_path: Path, name: str = "rl_token_stage2") -> list[str]:
    return [
        name,
        "--checkpoint-dir",
        str(tmp_path / "round_000001"),
        "--round-id",
        "round_000001",
        "--admission",
        str(tmp_path / "admission.json"),
        "--replay-snapshot",
        str(tmp_path / "replay.json"),
        "--seed",
        "19",
    ]


def test_parse_config_accepts_documented_stage2_command(tmp_path: Path):
    config = train_stage2.parse_config(_args(tmp_path))

    assert config.checkpoint_dir == tmp_path / "round_000001"
    assert config.round_id == "round_000001"
    assert config.seed == 19
    assert config.runtime.batch_size == 256


def test_debug_name_selects_small_runtime(tmp_path: Path):
    config = train_stage2.parse_config(_args(tmp_path, "rl_token_stage2_debug"))

    assert config.runtime.batch_size == 2
    assert config.runtime.network.actor_hidden_dims == (64, 64, 64)


def test_main_runs_once_and_prints_final_step(tmp_path: Path, monkeypatch, capsys):
    expected = tmp_path / "round_000001" / "123"
    calls: list[native_training.NativeRoundConfig] = []
    monkeypatch.setattr(native_training, "run_native_round", lambda config: calls.append(config) or expected)

    train_stage2.main(_args(tmp_path))

    assert len(calls) == 1
    assert capsys.readouterr().out == f"RLT Stage 2 round checkpoint: {expected}\n"


def test_unknown_config_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="not found"):
        train_stage2.parse_config(_args(tmp_path, "old_stage2_name"))
