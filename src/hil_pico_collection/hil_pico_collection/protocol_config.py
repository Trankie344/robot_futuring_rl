"""Declarative robot, observation and inference protocol configuration."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
DEFAULT_CONFIG_NAME = "zme_dual_arm.yaml"


@dataclass(frozen=True)
class ImageSpec:
    name: str
    topic: str
    message_type: str
    width: int
    height: int
    channels: int = 3
    resize: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)


@dataclass(frozen=True)
class VectorSpec:
    dimension: int
    order: tuple[str, ...]
    sources: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ActionSpec:
    dimension: int
    order: tuple[str, ...]
    observed_sources: Mapping[str, tuple[str, ...]]
    command_targets: Mapping[str, tuple[str, ...]]
    limits: Mapping[str, tuple[float | None, float | None]]


@dataclass(frozen=True)
class ConditionSpec:
    paths: tuple[str, ...]
    equals: Any


@dataclass(frozen=True)
class ModeControllerSpec:
    factory: str
    options: Mapping[str, Any]


@dataclass(frozen=True)
class RobotProtocolConfig:
    schema_version: int
    name: str
    robot_type: str
    status_topic: str
    command_topic: str
    reset_request_topic: str
    change_control_mode_topic: str
    status_message_type: str
    command_message_type: str
    header_stamp_path: str | None
    header_frame_id_path: str | None
    state: VectorSpec
    action: ActionSpec
    control_mode_paths: tuple[str, ...]
    intervention_condition: ConditionSpec
    model_control_conditions: tuple[ConditionSpec, ...]
    images: tuple[ImageSpec, ...]
    action_horizon: int
    model_hz: float
    command_hz: float
    mode_controller: ModeControllerSpec

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(image.name for image in self.images)

    @property
    def image_by_name(self) -> dict[str, ImageSpec]:
        return {image.name: image for image in self.images}

    @property
    def command_count(self) -> int:
        return max(2, int(round(self.action_horizon * self.command_hz / self.model_hz)))


def default_robot_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / DEFAULT_CONFIG_NAME


def load_robot_protocol_config(path: str | Path | None = None) -> RobotProtocolConfig:
    config_path = default_robot_config_path() if path is None else Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"robot protocol config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid robot protocol YAML {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("robot protocol config root must be a mapping")
    return parse_robot_protocol_config(raw)


def parse_robot_protocol_config(raw: Mapping[str, Any]) -> RobotProtocolConfig:
    schema_version = _positive_int(raw.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported robot protocol schema_version: {schema_version}")

    robot = _mapping(raw.get("robot"), "robot")
    state_raw = _mapping(raw.get("state"), "state")
    action_raw = _mapping(raw.get("action"), "action")
    status_raw = _mapping(raw.get("status"), "status")
    inference = _mapping(raw.get("inference"), "inference")

    state_order = _unique_names(state_raw.get("order"), "state.order")
    state_dimension = _positive_int(state_raw.get("dimension"), "state.dimension")
    if len(state_order) != state_dimension:
        raise ValueError("state.dimension must equal len(state.order)")
    state_sources = _binding_map(state_raw.get("sources"), state_order, "state.sources")

    action_order = _unique_names(action_raw.get("order"), "action.order")
    action_dimension = _positive_int(action_raw.get("dimension"), "action.dimension")
    if len(action_order) != action_dimension:
        raise ValueError("action.dimension must equal len(action.order)")
    observed = _binding_map(action_raw.get("observed_sources"), action_order, "action.observed_sources")
    targets = _binding_map(action_raw.get("command_targets"), action_order, "action.command_targets")
    limits = _limits(action_raw.get("limits", {}), action_order)

    images_raw = _mapping(raw.get("images"), "images")
    images = []
    for name, value in images_raw.items():
        image = _mapping(value, f"images.{name}")
        channels = _positive_int(image.get("channels", 3), f"images.{name}.channels")
        if channels != 3:
            raise ValueError(f"images.{name}.channels must be 3 for RGB policy observations")
        images.append(
            ImageSpec(
                name=_name(name, "image name"),
                topic=_nonempty(image.get("topic"), f"images.{name}.topic"),
                message_type=_nonempty(image.get("message_type"), f"images.{name}.message_type"),
                width=_positive_int(image.get("width"), f"images.{name}.width"),
                height=_positive_int(image.get("height"), f"images.{name}.height"),
                channels=channels,
                resize=bool(image.get("resize", False)),
            )
        )
    if not images:
        raise ValueError("images must contain at least one observation")
    if len({image.name for image in images}) != len(images):
        raise ValueError("image names must be unique")

    mode_raw = _mapping(raw.get("mode_controller"), "mode_controller")
    mode_options = mode_raw.get("options", {})
    if not isinstance(mode_options, Mapping):
        raise ValueError("mode_controller.options must be a mapping")
    model_control_raw = _sequence(
        status_raw.get("model_control_enabled"),
        "status.model_control_enabled",
    )
    if not model_control_raw:
        raise ValueError("status.model_control_enabled must not be empty")

    return RobotProtocolConfig(
        schema_version=schema_version,
        name=_nonempty(raw.get("name"), "name"),
        robot_type=_nonempty(robot.get("robot_type"), "robot.robot_type"),
        status_topic=_nonempty(robot.get("status_topic"), "robot.status_topic"),
        command_topic=_nonempty(robot.get("command_topic"), "robot.command_topic"),
        reset_request_topic=_nonempty(robot.get("reset_request_topic"), "robot.reset_request_topic"),
        change_control_mode_topic=_nonempty(
            robot.get("change_control_mode_topic"),
            "robot.change_control_mode_topic",
        ),
        status_message_type=_nonempty(robot.get("status_message_type"), "robot.status_message_type"),
        command_message_type=_nonempty(robot.get("command_message_type"), "robot.command_message_type"),
        header_stamp_path=_optional_path(robot.get("header_stamp_path")),
        header_frame_id_path=_optional_path(robot.get("header_frame_id_path")),
        state=VectorSpec(state_dimension, state_order, state_sources),
        action=ActionSpec(action_dimension, action_order, observed, targets, limits),
        control_mode_paths=_paths(status_raw.get("control_mode"), "status.control_mode"),
        intervention_condition=_condition(status_raw.get("intervention"), "status.intervention"),
        model_control_conditions=tuple(
            _condition(value, f"status.model_control_enabled[{index}]") for index, value in enumerate(model_control_raw)
        ),
        images=tuple(images),
        action_horizon=_positive_int(inference.get("action_horizon"), "inference.action_horizon"),
        model_hz=_positive_float(inference.get("model_hz"), "inference.model_hz"),
        command_hz=_positive_float(inference.get("command_hz"), "inference.command_hz"),
        mode_controller=ModeControllerSpec(
            factory=_nonempty(mode_raw.get("factory"), "mode_controller.factory"),
            options=dict(mode_options),
        ),
    )


def import_symbol(spec: str) -> Any:
    module_name, separator, symbol_name = str(spec).partition(":")
    if not separator:
        module_name, separator, symbol_name = str(spec).rpartition(".")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"symbol must use module:attribute syntax: {spec!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"symbol does not exist: {spec}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


def _nonempty(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _name(value: Any, label: str) -> str:
    result = _nonempty(value, label)
    if any(character.isspace() for character in result):
        raise ValueError(f"{label} must not contain whitespace")
    return result


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _unique_names(value: Any, label: str) -> tuple[str, ...]:
    result = tuple(_name(item, label) for item in _sequence(value, label))
    if not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique names")
    return result


def _paths(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = (value.strip(),)
    else:
        result = tuple(str(item).strip() for item in _sequence(value, label))
    if not result or any(not path for path in result):
        raise ValueError(f"{label} must contain at least one non-empty field path")
    return result


def _binding_map(value: Any, order: tuple[str, ...], label: str) -> dict[str, tuple[str, ...]]:
    mapping = _mapping(value, label)
    if set(mapping) != set(order):
        missing = sorted(set(order) - set(mapping))
        extra = sorted(set(mapping) - set(order))
        raise ValueError(f"{label} keys must match order; missing={missing}, extra={extra}")
    return {name: _paths(mapping[name], f"{label}.{name}") for name in order}


def _condition(value: Any, label: str) -> ConditionSpec:
    condition = _mapping(value, label)
    if "equals" not in condition:
        raise ValueError(f"{label}.equals is required")
    return ConditionSpec(
        paths=_paths(condition.get("path"), f"{label}.path"),
        equals=condition["equals"],
    )


def _limits(value: Any, order: tuple[str, ...]) -> dict[str, tuple[float | None, float | None]]:
    mapping = _mapping(value, "action.limits")
    extra = sorted(set(mapping) - set(order))
    if extra:
        raise ValueError(f"action.limits contains unknown names: {extra}")
    result = {}
    for name in order:
        raw = mapping.get(name, {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"action.limits.{name} must be a mapping")
        lower = None if raw.get("min") is None else float(raw["min"])
        upper = None if raw.get("max") is None else float(raw["max"])
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"action.limits.{name} min exceeds max")
        result[name] = (lower, upper)
    return result


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    return _nonempty(value, "header path")
