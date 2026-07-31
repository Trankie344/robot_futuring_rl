"""Dynamic robot-adapter loader."""

from __future__ import annotations

import importlib
from typing import Any

from hil_pico_collection.protocol_config import RobotProtocolConfig, import_symbol, load_robot_protocol_config

from .base import RobotAdapter
from .configured import ConfiguredRobotAdapter

DEFAULT_ROBOT_ADAPTER = "hil_pico_collection.adapters.arm_interfaces:create_adapter"


def load_robot_adapter(spec: str = DEFAULT_ROBOT_ADAPTER, **factory_kwargs: Any) -> RobotAdapter:
    """Load a module:factory robot adapter without importing ROS globally."""

    module_name, separator, factory_name = str(spec).partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("robot adapter must use module:factory syntax")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, factory_name)
    except AttributeError as exc:
        raise ValueError(f"robot adapter factory does not exist: {spec}") from exc
    adapter = factory(**factory_kwargs)
    for name in (
        "parse_status",
        "parse_executed_action",
        "build_command",
        "create_mode_controller",
        "status_message_type",
        "command_message_type",
    ):
        if not hasattr(adapter, name):
            raise TypeError(f"robot adapter {spec!r} is missing {name}")
    return adapter


def load_configured_robot_adapter(
    config_path: str | None = None,
    *,
    symbol_resolver: Any = import_symbol,
) -> tuple[RobotProtocolConfig, ConfiguredRobotAdapter]:
    config = load_robot_protocol_config(config_path)
    return config, ConfiguredRobotAdapter(config, symbol_resolver=symbol_resolver)
