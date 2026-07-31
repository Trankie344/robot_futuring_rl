from __future__ import annotations

import dataclasses
import difflib
from pathlib import Path
from typing import Literal

from openpi.models.arm_value.config import ArmValueModelConfig


@dataclasses.dataclass(frozen=True)
class ArmValueDataConfig:
    repo_id: str
    progress_path: str
    norm_stats_path: str
    camera_key: str = "observation.images.top"
    state_key: str = "observation.state"
    dataset_index: int = 0
    video_tolerance_s: float = 0.3
    interval_eps: float = 1e-3

    def __post_init__(self) -> None:
        if not self.camera_key:
            raise ValueError("camera_key cannot be empty")
        if not self.state_key:
            raise ValueError("state_key cannot be empty")
        if self.video_tolerance_s <= 0.0:
            raise ValueError("video_tolerance_s must be positive")
        if self.interval_eps < 0.0:
            raise ValueError("interval_eps must be non-negative")


@dataclasses.dataclass(frozen=True)
class ArmValueTrainConfig:
    name: str
    model: ArmValueModelConfig
    data: ArmValueDataConfig
    exp_name: str = ""
    output_base_dir: str = "./checkpoints/arm_value"
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 4
    num_train_steps: int = 5_000
    learning_rate: float = 5e-5
    end_learning_rate: float = 0.0
    warmup_steps: int = 150
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    log_interval: int = 20
    save_interval: int = 500
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    wandb_enabled: bool = True
    wandb_project: str = "openpi-arm-value"
    overwrite: bool = False
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.num_train_steps < 1:
            raise ValueError("num_train_steps must be positive")
        if not 0 <= self.warmup_steps < self.num_train_steps:
            raise ValueError("warmup_steps must be in [0, num_train_steps)")
        if self.learning_rate <= 0.0 or self.end_learning_rate < 0.0:
            raise ValueError("learning rates must be non-negative and peak learning rate positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.log_interval < 1 or self.save_interval < 1:
            raise ValueError("log_interval and save_interval must be positive")

    @property
    def output_dir(self) -> Path:
        if not self.exp_name:
            raise ValueError("--exp-name must be set")
        return (Path(self.output_base_dir) / self.name / self.exp_name).resolve()


_HIL_DATASET = "/mnt/workspace/ys/futuring/modeltraining/datasets/hil_pico_v21/hil_pico_v21_20260704"
_HIL_PROGRESS = "/mnt/workspace/robot_task_raw/lite-0028/arm_progress_hil_pico_v21_20260704_from_tristate.parquet"
_HIL_NORM_STATS = (
    "/mnt/workspace/ys/futuring/modeltraining/models/acot/config/assets/"
    "pi05_hil_pico_v21_20260704_delta16/hil_pico_v21_20260704/norm_stats.json"
)

_CONFIGS = [
    ArmValueTrainConfig(
        name="arm_value_hil_pico_v21",
        model=ArmValueModelConfig(),
        data=ArmValueDataConfig(
            repo_id=_HIL_DATASET,
            progress_path=_HIL_PROGRESS,
            norm_stats_path=_HIL_NORM_STATS,
        ),
        device="cuda",
    ),
    ArmValueTrainConfig(
        name="arm_value_debug",
        model=ArmValueModelConfig(
            clip_pretrained_path="__debug__",
            n_history_steps=2,
            frame_gap=2,
            max_state_dim=8,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ),
        data=ArmValueDataConfig(repo_id="fake", progress_path="", norm_stats_path=""),
        exp_name="debug",
        batch_size=2,
        num_workers=0,
        num_train_steps=1,
        warmup_steps=0,
        log_interval=1,
        save_interval=1,
        precision="float32",
        device="cpu",
        wandb_enabled=False,
        overwrite=True,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("ARM value config names must be unique")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> ArmValueTrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT, n=1, cutoff=0.0)
        closest_text = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_text}")
    return _CONFIGS_DICT[config_name]


def cli() -> ArmValueTrainConfig:
    import tyro  # noqa: PLC0415

    return tyro.extras.overridable_config_cli({name: (name, config) for name, config in _CONFIGS_DICT.items()})
