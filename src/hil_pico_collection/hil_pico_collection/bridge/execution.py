"""Action-chunk resampling and interruptible command execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import ACTION_DIM, ACTION_HORIZON

MODEL_HZ = 20
COMMAND_HZ = 30
COMMAND_COUNT = 30


@dataclass(frozen=True)
class ExecutionResult:
    sent_count: int
    completed: bool
    reason: str | None = None


def resample_action_chunk(
    actions: Any,
    output_count: int = COMMAND_COUNT,
    *,
    expected_horizon: int = ACTION_HORIZON,
    expected_action_dim: int = ACTION_DIM,
) -> np.ndarray:
    """Linearly resample 20 absolute actions to the 30 Hz robot command grid."""

    values = np.asarray(actions, dtype=np.float32)
    expected_shape = (int(expected_horizon), int(expected_action_dim))
    if values.shape != expected_shape:
        raise ValueError(f"actions must have shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("actions must contain only finite values")
    output_count = int(output_count)
    if output_count < 2:
        raise ValueError("output_count must be at least 2")
    source = np.linspace(0.0, expected_horizon - 1, expected_horizon)
    target = np.linspace(0.0, expected_horizon - 1, output_count)
    result = np.empty((output_count, expected_action_dim), dtype=np.float32)
    for dimension in range(expected_action_dim):
        result[:, dimension] = np.interp(target, source, values[:, dimension])
    return result


def execute_action_chunk(
    actions: Any,
    *,
    guard: Any,
    build_command: Any,
    publish_command: Any,
    stamp: Any,
    frame_id: str = "rl_token_stage2",
    command_hz: float = COMMAND_HZ,
    output_count: int = COMMAND_COUNT,
    expected_horizon: int = ACTION_HORIZON,
    expected_action_dim: int = ACTION_DIM,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> ExecutionResult:
    """Publish one full chunk unless arbitration, freshness or adapter checks fail."""

    commands = resample_action_chunk(
        actions,
        output_count,
        expected_horizon=expected_horizon,
        expected_action_dim=expected_action_dim,
    )
    period = 1.0 / float(command_hz)
    start = monotonic()
    sent = 0
    for index, action in enumerate(commands):
        try:
            allowed = bool(guard())
        except Exception as exc:
            return ExecutionResult(sent, False, f"guard failed: {exc}")
        if not allowed:
            return ExecutionResult(sent, False, "model control lost or state became stale")
        try:
            message = build_command(action, stamp(), frame_id)
            publish_command(message)
        except Exception as exc:
            return ExecutionResult(sent, False, f"command rejected: {exc}")
        sent += 1
        deadline = start + (index + 1) * period
        remaining = deadline - monotonic()
        if remaining > 0:
            sleep(remaining)
    return ExecutionResult(sent, True)
