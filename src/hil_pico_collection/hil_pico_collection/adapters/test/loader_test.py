import sys
from types import ModuleType

import pytest

from hil_pico_collection.adapters.loader import load_configured_robot_adapter, load_robot_adapter
from hil_pico_collection.protocol_config import default_robot_config_path


class Adapter:
    status_message_type = object
    command_message_type = object

    parse_status = object()
    parse_executed_action = object()
    build_command = object()
    create_mode_controller = object()


def test_load_robot_adapter_calls_factory(monkeypatch):
    module = ModuleType("fake_hil_adapter")
    module.make = lambda answer=None: (Adapter(), answer)[0]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert isinstance(load_robot_adapter("fake_hil_adapter:make", answer=42), Adapter)


@pytest.mark.parametrize("spec", ["missing_separator", ":factory", "module:"])
def test_load_robot_adapter_rejects_invalid_spec(spec):
    with pytest.raises(ValueError, match="module:factory"):
        load_robot_adapter(spec)


def test_load_robot_adapter_rejects_missing_factory(monkeypatch):
    module = ModuleType("empty_hil_adapter")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="does not exist"):
        load_robot_adapter("empty_hil_adapter:make")


def test_load_configured_robot_adapter_returns_shared_protocol_and_adapter():
    protocol, adapter = load_configured_robot_adapter(
        str(default_robot_config_path()),
        symbol_resolver=lambda spec: type(spec.rsplit(":", 1)[-1], (), {}),
    )
    assert adapter.config is protocol
    assert protocol.state.dimension == 16
    assert protocol.command_topic == "/auto_arm_cmd"
