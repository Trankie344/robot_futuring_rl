import multiprocessing
import socket
import time

import numpy as np

from hil_pico_collection.bridge.protocol import RLTokenPolicyClient, build_observation


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(port):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class Policy:
        def infer(self, observation):
            assert set(observation) == {"images", "state", "prompt"}
            assert set(observation["images"]) == {"top", "left_wrist", "right_wrist"}
            actions = np.tile(np.asarray(observation["state"], np.float32), (20, 1))
            return {"actions": actions, "policy_timing": {"fake_ms": 0.0}}

    metadata = {
        "rlt_stage2": {
            "round_complete": True,
            "network_config": {"state_dim": 16, "action_dim": 16, "action_horizon": 20},
        }
    }
    WebsocketPolicyServer(Policy(), "127.0.0.1", port, metadata).serve_forever()


def test_official_openpi_server_and_client_wire_protocol():
    port = _free_port()
    process = multiprocessing.get_context("fork").Process(target=_serve, args=(port,), daemon=True)
    process.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.02)
        client = RLTokenPolicyClient("127.0.0.1", port)
        images = {key: np.zeros((2, 3, 3), np.uint8) for key in ("top", "left_wrist", "right_wrist")}
        result = client.infer(build_observation(images, np.arange(16), "task"))
        np.testing.assert_array_equal(result["actions"][0], np.arange(16))
        assert "server_timing" in result
        client.close()
    finally:
        process.terminate()
        process.join(timeout=2)
