from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch

from openpi.training.arm_value.config import ArmValueTrainConfig

_RESUME_TRAIN_FIELDS = (
    "seed",
    "batch_size",
    "num_train_steps",
    "learning_rate",
    "end_learning_rate",
    "warmup_steps",
    "weight_decay",
    "max_grad_norm",
    "precision",
)


def assert_resume_compatible(saved_config: dict[str, Any], config: ArmValueTrainConfig) -> None:
    current_config = dataclasses.asdict(config)
    mismatches: list[str] = []
    for section in ("model", "data"):
        saved_section = saved_config.get(section)
        current_section = current_config[section]
        if saved_section != current_section:
            mismatches.append(f"{section}: saved={saved_section!r}, current={current_section!r}")
    mismatches.extend(
        (f"{field}: saved={saved_config.get(field)!r}, current={current_config[field]!r}")
        for field in _RESUME_TRAIN_FIELDS
        if saved_config.get(field) != current_config[field]
    )
    if mismatches:
        raise ValueError("Incompatible ARM value checkpoint configuration: " + "; ".join(mismatches))


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: ArmValueTrainConfig,
    progress_summary: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "config": dataclasses.asdict(config),
            "progress_summary": progress_summary,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: ArmValueTrainConfig | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if config is not None:
        assert_resume_compatible(checkpoint["config"], config)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
