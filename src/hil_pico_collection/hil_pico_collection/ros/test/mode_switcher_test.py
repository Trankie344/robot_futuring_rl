from types import SimpleNamespace

import numpy as np

from hil_pico_collection.adapters.base import RobotStateSample
from hil_pico_collection.ros.mode_switcher import _make_mode_switcher_node_class


class Node:
    def __init__(self, name):
        self.name = name
        self.subscriptions = {}
        self.logs = []

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions[topic] = callback
        return object()

    def get_logger(self):
        return SimpleNamespace(
            warn=lambda message: self.logs.append(("warn", message)),
            info=lambda message: self.logs.append(("info", message)),
        )


class Controller:
    def __init__(self):
        self.samples = []
        self.accept = True

    def toggle(self, sample):
        self.samples.append(sample)
        return self.accept


class Adapter:
    status_message_type = object

    def __init__(self):
        self.controller = Controller()

    def parse_status(self, message):
        if message == "bad":
            raise ValueError("bad status")
        return RobotStateSample(np.zeros(16), message, message == 5, message == 1)

    def create_mode_controller(self, node):
        return self.controller


def make_node():
    ros = SimpleNamespace(Node=Node, Bool=object)
    adapter = Adapter()
    node = _make_mode_switcher_node_class(ros)(adapter, 0.5, "/change_ctrl_mode")
    return node, adapter


def test_true_topic_toggles_from_latest_status():
    node, adapter = make_node()
    node._on_status(1)
    node._on_toggle(SimpleNamespace(data=True))
    assert len(adapter.controller.samples) == 1
    assert adapter.controller.samples[0].control_mode == 1


def test_false_topic_is_ignored():
    node, adapter = make_node()
    node._on_status(1)
    node._on_toggle(SimpleNamespace(data=False))
    assert adapter.controller.samples == []


def test_toggle_without_status_is_ignored():
    node, adapter = make_node()
    node._on_toggle(SimpleNamespace(data=True))
    assert adapter.controller.samples == []
    assert node.logs


def test_invalid_status_does_not_replace_cache():
    node, adapter = make_node()
    node._on_status("bad")
    node._on_toggle(SimpleNamespace(data=True))
    assert adapter.controller.samples == []


def test_duplicate_busy_request_is_reported():
    node, adapter = make_node()
    adapter.controller.accept = False
    node._on_status(1)
    node._on_toggle(SimpleNamespace(data=True))
    assert any("ignored" in message for _, message in node.logs)
