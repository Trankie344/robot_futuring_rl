"""Robot-independent adapter contracts used by recording and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

CANONICAL_DIM = 16
CANONICAL_NAMES = (
    *(f"right_joint_{index}" for index in range(7)),
    "left_gripper",
    *(f"left_joint_{index}" for index in range(7)),
    "right_gripper",
)


@dataclass(frozen=True)
class RobotStateSample:
    """Canonical robot state and arbitration information for one status frame."""

    state: np.ndarray
    control_mode: int
    intervention: bool
    model_control_enabled: bool

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.ndim != 1 or state.size == 0:
            raise ValueError(f"robot state must be a non-empty vector, got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("robot state must contain only finite values")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "control_mode", int(self.control_mode))
        object.__setattr__(self, "intervention", bool(self.intervention))
        object.__setattr__(self, "model_control_enabled", bool(self.model_control_enabled))


@runtime_checkable
class ModeController(Protocol):
    """Robot-side mode switch operation initiated by /change_ctrl_mode."""

    def toggle(self, sample: RobotStateSample) -> bool:
        """Request the opposite autonomous/PICO mode."""


@runtime_checkable
class RobotAdapter(Protocol):
    """Maps robot-specific ROS messages to the canonical HIL contract."""

    status_message_type: Any
    command_message_type: Any

    def parse_status(self, message: Any) -> RobotStateSample:
        """Parse a robot status message."""

    def parse_executed_action(self, message: Any) -> np.ndarray:
        """Parse the action actually sent/executed by the robot."""

    def build_command(self, action: Any, stamp: Any, frame_id: str) -> Any:
        """Build a robot-specific ROS command from a canonical absolute action."""

    def create_mode_controller(self, node: Any) -> ModeController:
        """Create the robot-specific mode controller bound to a ROS node."""
