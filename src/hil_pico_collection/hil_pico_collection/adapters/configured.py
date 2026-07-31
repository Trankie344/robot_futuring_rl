"""Generic ROS robot adapter driven entirely by RobotProtocolConfig."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from hil_pico_collection.protocol_config import ConditionSpec, RobotProtocolConfig, import_symbol

from .base import RobotStateSample

_PATH_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")


class ConfiguredRobotAdapter:
    """Map arbitrary ROS messages using ordered field bindings from YAML."""

    def __init__(
        self,
        config: RobotProtocolConfig,
        *,
        symbol_resolver: Callable[[str], Any] = import_symbol,
    ) -> None:
        self.config = config
        self._symbol_resolver = symbol_resolver
        self.status_message_type = symbol_resolver(config.status_message_type)
        self.command_message_type = symbol_resolver(config.command_message_type)

    def parse_status(self, message: Any) -> RobotStateSample:
        state = _read_vector(message, self.config.state.order, self.config.state.sources, "state")
        control_mode = int(read_first_path(message, self.config.control_mode_paths))
        return RobotStateSample(
            state=state,
            control_mode=control_mode,
            intervention=_matches(message, self.config.intervention_condition),
            model_control_enabled=all(
                _matches(message, condition) for condition in self.config.model_control_conditions
            ),
        )

    def parse_executed_action(self, message: Any) -> np.ndarray:
        return _read_vector(
            message,
            self.config.action.order,
            self.config.action.observed_sources,
            "action",
        )

    def build_command(self, action: Any, stamp: Any, frame_id: str) -> Any:
        vector = np.asarray(action, dtype=np.float32)
        expected = (self.config.action.dimension,)
        if vector.shape != expected:
            raise ValueError(f"action must have shape {expected}, got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("action must contain only finite values")
        self._validate_limits(vector)

        message = self.command_message_type()
        if self.config.header_stamp_path is not None:
            write_path(message, self.config.header_stamp_path, stamp)
        if self.config.header_frame_id_path is not None:
            write_path(message, self.config.header_frame_id_path, str(frame_id))
        for index, name in enumerate(self.config.action.order):
            write_first_path(message, self.config.action.command_targets[name], float(vector[index]))
        return message

    def create_mode_controller(self, node: Any) -> Any:
        factory = self._symbol_resolver(self.config.mode_controller.factory)
        return factory(
            node=node,
            adapter=self,
            options=dict(self.config.mode_controller.options),
            symbol_resolver=self._symbol_resolver,
        )

    def _validate_limits(self, vector: np.ndarray) -> None:
        for index, name in enumerate(self.config.action.order):
            lower, upper = self.config.action.limits[name]
            value = float(vector[index])
            if lower is not None and value < lower:
                raise ValueError(f"action {name}={value} is below configured minimum {lower}")
            if upper is not None and value > upper:
                raise ValueError(f"action {name}={value} exceeds configured maximum {upper}")


def read_first_path(root: Any, paths: Sequence[str]) -> Any:
    errors = []
    for path in paths:
        try:
            return read_path(root, path)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    raise ValueError(f"none of the configured field paths could be read: {'; '.join(errors)}")


def read_path(root: Any, path: str) -> Any:
    value = root
    for token in _parse_path(path):
        if isinstance(token, int):
            value = value[token]
        elif isinstance(value, dict):
            value = value[token]
        else:
            value = getattr(value, token)
    return value


def write_first_path(root: Any, paths: Sequence[str], value: Any) -> None:
    errors = []
    for path in paths:
        try:
            write_path(root, path, value)
            return
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    raise ValueError(f"none of the configured command paths could be written: {'; '.join(errors)}")


def write_path(root: Any, path: str, value: Any) -> None:
    tokens = _parse_path(path)
    if not tokens:
        raise ValueError("field path must not be empty")
    parent = root
    for token in tokens[:-1]:
        if isinstance(token, int):
            parent = _sequence_item(parent, token)
        elif isinstance(parent, dict):
            parent = parent[token]
        else:
            parent = getattr(parent, token)

    final = tokens[-1]
    if isinstance(final, int):
        _set_sequence_item(parent, final, value)
    elif isinstance(parent, dict):
        parent[final] = value
    else:
        setattr(parent, final, value)


def _parse_path(path: str) -> tuple[str | int, ...]:
    source = str(path).strip()
    if not source:
        raise ValueError("field path must not be empty")
    tokens: list[str | int] = []
    position = 0
    while position < len(source):
        if source[position] == ".":
            position += 1
            continue
        match = _PATH_TOKEN.match(source, position)
        if match is None:
            raise ValueError(f"invalid field path near {source[position:]!r}")
        attribute, index = match.groups()
        tokens.append(attribute if attribute is not None else int(index))
        position = match.end()
    return tuple(tokens)


def _sequence_item(sequence: Any, index: int) -> Any:
    try:
        return sequence[index]
    except IndexError:
        if isinstance(sequence, list):
            sequence.extend([None] * (index + 1 - len(sequence)))
            if sequence[index] is None:
                raise IndexError(f"intermediate list item {index} does not exist")
        raise


def _set_sequence_item(sequence: Any, index: int, value: Any) -> None:
    if index >= len(sequence):
        if isinstance(sequence, list):
            sequence.extend([0.0] * (index + 1 - len(sequence)))
        elif hasattr(sequence, "extend"):
            sequence.extend([0.0] * (index + 1 - len(sequence)))
        else:
            raise IndexError(index)
    sequence[index] = value


def _read_vector(
    message: Any,
    order: tuple[str, ...],
    sources: Any,
    label: str,
) -> np.ndarray:
    values = []
    for name in order:
        try:
            value = float(read_first_path(message, sources[name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"failed to read {label} field {name}: {exc}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{label} field {name} must be finite")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _matches(message: Any, condition: ConditionSpec) -> bool:
    return read_first_path(message, condition.paths) == condition.equals
