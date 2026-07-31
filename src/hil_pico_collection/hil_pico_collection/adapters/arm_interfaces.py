"""Default adapter for the ZME arm_interfaces ROS message contract."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .base import CANONICAL_DIM, RobotStateSample

AUTONOMY_JOINT = 1
TELEOPERATION_VR = 5
EXECUTER_MODE_SERVICE = "/switch_arm_executer_mode"
SECONDARY_MODE_SERVICE = "/switch_arm_secondary_mode"


def _sequence(message: Any, field_names: Iterable[str], minimum: int) -> tuple[Sequence[Any], str]:
    for field_name in field_names:
        if not hasattr(message, field_name):
            continue
        value = getattr(message, field_name)
        try:
            length = len(value)
        except TypeError as exc:
            raise ValueError(f"{field_name} must be a sequence") from exc
        if length < minimum:
            raise ValueError(f"{field_name} must contain at least {minimum} values, got {length}")
        return value, field_name
    names = "/".join(field_names)
    raise ValueError(f"one of {names} is required")


def _joint_value(value: Any, label: str) -> float:
    if hasattr(value, "joint_status"):
        values = getattr(value, "joint_status")
        if len(values) < 1:
            raise ValueError(f"{label}.joint_status must contain at least 1 value")
        value = values[0]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _joint_block(values: Sequence[Any], field_name: str) -> list[float]:
    return [_joint_value(values[index], f"{field_name}[{index}]") for index in range(7)]


def _canonical_vector(
    right: Sequence[Any],
    right_name: str,
    left_gripper: Any,
    left_gripper_name: str,
    left: Sequence[Any],
    left_name: str,
    right_gripper: Any,
    right_gripper_name: str,
) -> np.ndarray:
    result = np.asarray(
        _joint_block(right, right_name)
        + [_joint_value(left_gripper, left_gripper_name)]
        + _joint_block(left, left_name)
        + [_joint_value(right_gripper, right_gripper_name)],
        dtype=np.float32,
    )
    if result.shape != (CANONICAL_DIM,) or not np.isfinite(result).all():
        raise ValueError("mapped arm vector must be finite and 16-dimensional")
    return result


@dataclass(frozen=True)
class ArmLimits:
    """Optional adapter-level soft limits; robot firmware remains authoritative."""

    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    def validate(self, action: np.ndarray) -> None:
        if self.lower is not None:
            lower = np.asarray(self.lower, dtype=np.float32)
            if lower.shape != (CANONICAL_DIM,):
                raise ValueError("lower soft limits must have shape (16,)")
            if np.any(action < lower):
                raise ValueError("action violates adapter lower soft limits")
        if self.upper is not None:
            upper = np.asarray(self.upper, dtype=np.float32)
            if upper.shape != (CANONICAL_DIM,):
                raise ValueError("upper soft limits must have shape (16,)")
            if np.any(action > upper):
                raise ValueError("action violates adapter upper soft limits")


class ArmInterfacesAdapter:
    """Map arm_interfaces messages using the RL Token training order."""

    def __init__(
        self,
        *,
        status_message_type: Any = None,
        command_message_type: Any = None,
        mode_service_type: Any = None,
        secondary_service_type: Any = None,
        limits: ArmLimits | None = None,
        service_timeout_s: float = 0.25,
    ) -> None:
        if status_message_type is None or command_message_type is None:
            from arm_interfaces.msg import ArmStatus, AutonomyArmCommand

            status_message_type = status_message_type or ArmStatus
            command_message_type = command_message_type or AutonomyArmCommand
        if mode_service_type is None or secondary_service_type is None:
            from arm_interfaces.srv import TrajFollowMode, TrajFollowSecondaryMode

            mode_service_type = mode_service_type or TrajFollowMode
            secondary_service_type = secondary_service_type or TrajFollowSecondaryMode
        self.status_message_type = status_message_type
        self.command_message_type = command_message_type
        self.mode_service_type = mode_service_type
        self.secondary_service_type = secondary_service_type
        self.limits = limits or ArmLimits()
        self.service_timeout_s = float(service_timeout_s)

    def parse_status(self, message: Any) -> RobotStateSample:
        left, left_name = _sequence(message, ("left_arm",), 7)
        right, right_name = _sequence(message, ("right_arm",), 7)
        grippers, gripper_name = _sequence(message, ("gripper",), 2)
        other_status, _ = _sequence(message, ("other_status",), 2)
        primary = int(other_status[0])
        secondary = int(other_status[1])
        state = _canonical_vector(
            right,
            right_name,
            grippers[0],
            f"{gripper_name}[0]",
            left,
            left_name,
            grippers[1],
            f"{gripper_name}[1]",
        )
        return RobotStateSample(
            state=state,
            control_mode=primary,
            intervention=primary == TELEOPERATION_VR,
            model_control_enabled=primary == AUTONOMY_JOINT and secondary == AUTONOMY_JOINT,
        )

    def parse_executed_action(self, message: Any) -> np.ndarray:
        left, left_name = _sequence(message, ("left_command", "left_joint_command"), 7)
        right, right_name = _sequence(message, ("right_command", "right_joint_command"), 7)
        grippers, gripper_name = _sequence(message, ("gripper_command",), 2)
        return _canonical_vector(
            right,
            right_name,
            grippers[0],
            f"{gripper_name}[0]",
            left,
            left_name,
            grippers[1],
            f"{gripper_name}[1]",
        )

    def build_command(self, action: Any, stamp: Any, frame_id: str) -> Any:
        vector = np.asarray(action, dtype=np.float32)
        if vector.shape != (CANONICAL_DIM,):
            raise ValueError(f"action must have shape (16,), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("action must contain only finite values")
        self.limits.validate(vector)

        message = self.command_message_type()
        if hasattr(message, "header"):
            message.header.stamp = stamp
            message.header.frame_id = str(frame_id)
        _assign_first(message, ("right_command", "right_joint_command"), vector[:7].tolist())
        _assign_first(message, ("left_command", "left_joint_command"), vector[8:15].tolist())
        grippers = [float(vector[7]), float(vector[15])]
        current = getattr(message, "gripper_command", None)
        if current is not None and len(current) > 2:
            grippers.extend([0.0] * (len(current) - 2))
        message.gripper_command = grippers
        return message

    def create_mode_controller(self, node: Any) -> "ArmInterfacesModeController":
        return ArmInterfacesModeController(self, node)


def _assign_first(message: Any, field_names: Iterable[str], values: list[float]) -> None:
    for field_name in field_names:
        if hasattr(message, field_name):
            setattr(message, field_name, values)
            return
    raise ValueError(f"command message is missing {'/'.join(field_names)}")


class ArmInterfacesModeController:
    """Translate a generic toggle into the two robot-specific mode services."""

    def __init__(
        self,
        adapter: Any,
        node: Any,
        *,
        executer_service: str = EXECUTER_MODE_SERVICE,
        secondary_service: str = SECONDARY_MODE_SERVICE,
        autonomy_primary: int = AUTONOMY_JOINT,
        autonomy_secondary: int = AUTONOMY_JOINT,
        teleoperation_primary: int = TELEOPERATION_VR,
    ) -> None:
        self._adapter = adapter
        self._node = node
        self._executer_service = str(executer_service)
        self._secondary_service = str(secondary_service)
        self._autonomy_primary = int(autonomy_primary)
        self._autonomy_secondary = int(autonomy_secondary)
        self._teleoperation_primary = int(teleoperation_primary)
        self._lock = threading.Lock()
        self._busy = False
        self._executer_client = node.create_client(adapter.mode_service_type, self._executer_service)
        self._secondary_client = node.create_client(adapter.secondary_service_type, self._secondary_service)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def toggle(self, sample: RobotStateSample) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        if sample.intervention:
            request = self._adapter.secondary_service_type.Request()
            request.trajectory_following_mode = self._autonomy_primary
            request.trajectory_following_secondary_mode = self._autonomy_secondary
            client = self._secondary_client
            service_name = self._secondary_service
        elif sample.control_mode in (0, self._autonomy_primary):
            request = self._adapter.mode_service_type.Request()
            request.trajectory_following_mode = self._teleoperation_primary
            client = self._executer_client
            service_name = self._executer_service
        else:
            self._finish(False, f"unsupported control mode {sample.control_mode}")
            return False

        if not client.wait_for_service(timeout_sec=self._adapter.service_timeout_s):
            self._finish(False, f"service unavailable: {service_name}")
            return False
        try:
            future = client.call_async(request)
            future.add_done_callback(lambda done: self._on_done(done, service_name))
        except Exception as exc:
            self._finish(False, f"service call failed for {service_name}: {exc}")
            return False
        return True

    def _on_done(self, future: Any, service_name: str) -> None:
        try:
            response = future.result()
            if hasattr(response, "result"):
                ok = bool(response.result)
            elif hasattr(response, "success"):
                ok = bool(response.success)
            else:
                ok = False
            detail = service_name if ok else f"{service_name} reported failure"
        except Exception as exc:
            ok = False
            detail = f"{service_name}: {exc}"
        self._finish(ok, detail)

    def _finish(self, ok: bool, detail: str) -> None:
        with self._lock:
            self._busy = False
        logger = self._node.get_logger()
        if ok:
            logger.info(f"mode switch succeeded: {detail}")
        else:
            logger.warn(f"mode switch rejected: {detail}")


def create_adapter(**kwargs: Any) -> ArmInterfacesAdapter:
    return ArmInterfacesAdapter(**kwargs)


def create_configured_mode_controller(
    *,
    node: Any,
    adapter: Any,
    options: dict[str, Any],
    symbol_resolver: Any,
) -> ArmInterfacesModeController:
    required = ("mode_service_type", "secondary_service_type")
    missing = [name for name in required if not options.get(name)]
    if missing:
        raise ValueError(f"ZME mode controller options are missing: {missing}")
    adapter.mode_service_type = symbol_resolver(str(options["mode_service_type"]))
    adapter.secondary_service_type = symbol_resolver(str(options["secondary_service_type"]))
    adapter.service_timeout_s = float(options.get("service_timeout_s", 0.25))
    return ArmInterfacesModeController(
        adapter,
        node,
        executer_service=str(options.get("executer_service", EXECUTER_MODE_SERVICE)),
        secondary_service=str(options.get("secondary_service", SECONDARY_MODE_SERVICE)),
        autonomy_primary=int(options.get("autonomy_primary", AUTONOMY_JOINT)),
        autonomy_secondary=int(options.get("autonomy_secondary", AUTONOMY_JOINT)),
        teleoperation_primary=int(options.get("teleoperation_primary", TELEOPERATION_VR)),
    )
