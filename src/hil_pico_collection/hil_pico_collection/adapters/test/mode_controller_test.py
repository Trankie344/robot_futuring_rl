from types import SimpleNamespace

import numpy as np

from hil_pico_collection.adapters.arm_interfaces import (
    EXECUTER_MODE_SERVICE,
    SECONDARY_MODE_SERVICE,
    ArmInterfacesAdapter,
)
from hil_pico_collection.adapters.base import RobotStateSample


class Service:
    class Request:
        pass


class Future:
    def __init__(self, response):
        self.response = response
        self.callbacks = []

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        callback(self)

    def result(self):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class Client:
    def __init__(self, *, available=True, response=None):
        self.available = available
        self.response = response or SimpleNamespace(result=True)
        self.requests = []

    def wait_for_service(self, timeout_sec):
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        return Future(self.response)


class Node:
    def __init__(self):
        self.clients = {
            EXECUTER_MODE_SERVICE: Client(),
            SECONDARY_MODE_SERVICE: Client(),
        }
        self.logs = []

    def create_client(self, service_type, name):
        return self.clients[name]

    def get_logger(self):
        return SimpleNamespace(
            info=lambda message: self.logs.append(("info", message)),
            warn=lambda message: self.logs.append(("warn", message)),
        )


def adapter():
    return ArmInterfacesAdapter(
        status_message_type=object,
        command_message_type=object,
        mode_service_type=Service,
        secondary_service_type=Service,
    )


def sample(mode, intervention, enabled):
    return RobotStateSample(
        state=np.zeros(16),
        control_mode=mode,
        intervention=intervention,
        model_control_enabled=enabled,
    )


def test_teleoperation_toggle_requests_autonomous_primary_and_secondary_modes():
    node = Node()
    controller = adapter().create_mode_controller(node)
    assert controller.toggle(sample(5, True, False)) is True
    request = node.clients[SECONDARY_MODE_SERVICE].requests[0]
    assert request.trajectory_following_mode == 1
    assert request.trajectory_following_secondary_mode == 1
    assert controller.busy is False


def test_autonomous_toggle_requests_pico_mode():
    node = Node()
    controller = adapter().create_mode_controller(node)
    assert controller.toggle(sample(1, False, True)) is True
    request = node.clients[EXECUTER_MODE_SERVICE].requests[0]
    assert request.trajectory_following_mode == 5


def test_unavailable_service_rejects_request_and_clears_busy():
    node = Node()
    node.clients[EXECUTER_MODE_SERVICE].available = False
    controller = adapter().create_mode_controller(node)
    assert controller.toggle(sample(1, False, True)) is False
    assert controller.busy is False
    assert any("unavailable" in message for _, message in node.logs)


def test_service_failure_is_logged_and_allows_retry():
    node = Node()
    node.clients[EXECUTER_MODE_SERVICE].response = SimpleNamespace(result=False)
    controller = adapter().create_mode_controller(node)
    assert controller.toggle(sample(1, False, True)) is True
    assert controller.busy is False
    assert controller.toggle(sample(1, False, True)) is True
    assert len(node.clients[EXECUTER_MODE_SERVICE].requests) == 2


def test_unknown_mode_is_rejected_without_service_call():
    node = Node()
    controller = adapter().create_mode_controller(node)
    assert controller.toggle(sample(9, False, False)) is False
    assert node.clients[EXECUTER_MODE_SERVICE].requests == []
    assert node.clients[SECONDARY_MODE_SERVICE].requests == []
