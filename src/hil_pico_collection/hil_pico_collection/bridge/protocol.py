"""Strict RL Token WebSocket request, response and metadata contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from hil_pico_collection.protocol_config import RobotProtocolConfig

CAMERA_KEYS = ("top", "left_wrist", "right_wrist")
STATE_DIM = 16
ACTION_DIM = 16
ACTION_HORIZON = 20


def validate_server_metadata(
    metadata: Any,
    protocol: RobotProtocolConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("RL Token server metadata must be a mapping")
    stage2 = metadata.get("rlt_stage2")
    if not isinstance(stage2, Mapping):
        raise ValueError("RL Token server metadata must contain rlt_stage2")
    if stage2.get("round_complete") is not True:
        raise ValueError("RL Token actor checkpoint must have round_complete=true")
    interface = stage2.get("network_config", stage2)
    if not isinstance(interface, Mapping):
        raise ValueError("rlt_stage2 network_config must be a mapping")
    expected = {
        "state_dim": STATE_DIM if protocol is None else protocol.state.dimension,
        "action_dim": ACTION_DIM if protocol is None else protocol.action.dimension,
        "action_horizon": ACTION_HORIZON if protocol is None else protocol.action_horizon,
    }
    for key, value in expected.items():
        if int(interface.get(key, -1)) != value:
            raise ValueError(f"RL Token metadata {key} must be {value}, got {interface.get(key)!r}")
    return dict(metadata)


def build_observation(
    images: Any,
    state: Any,
    prompt: str,
    protocol: RobotProtocolConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(images, Mapping):
        raise ValueError("images must be a mapping")
    image_names = CAMERA_KEYS if protocol is None else protocol.image_names
    if set(images) != set(image_names):
        raise ValueError(f"images must contain exactly {image_names}")
    normalized_images: dict[str, np.ndarray] = {}
    image_specs = {} if protocol is None else protocol.image_by_name
    for key in image_names:
        image = np.asarray(images[key])
        if image.dtype != np.uint8:
            raise ValueError(f"{key} image must use uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{key} image must have shape [H,W,3], got {image.shape}")
        if protocol is not None and image.shape != image_specs[key].shape:
            raise ValueError(f"{key} image must have configured shape {image_specs[key].shape}, got {image.shape}")
        normalized_images[key] = np.ascontiguousarray(image)
    state_array = np.asarray(state, dtype=np.float32)
    state_dimension = STATE_DIM if protocol is None else protocol.state.dimension
    if state_array.shape != (state_dimension,):
        raise ValueError(f"state must have shape ({state_dimension},), got {state_array.shape}")
    if not np.isfinite(state_array).all():
        raise ValueError("state must contain only finite values")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return {
        "images": normalized_images,
        "state": state_array.copy(),
        "prompt": prompt,
    }


def validate_inference_response(
    response: Any,
    protocol: RobotProtocolConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ValueError("RL Token response must be a mapping")
    if "actions" not in response:
        raise ValueError("RL Token response is missing actions")
    actions = np.asarray(response["actions"], dtype=np.float32)
    expected_shape = (
        ACTION_HORIZON if protocol is None else protocol.action_horizon,
        ACTION_DIM if protocol is None else protocol.action.dimension,
    )
    if actions.shape != expected_shape:
        raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions must contain only finite values")
    result = dict(response)
    result["actions"] = actions.copy()
    return result


class RLTokenPolicyClient:
    """Thin validated wrapper around OpenPI's official WebSocket client."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        api_key: str | None = None,
        client_factory: Any = None,
        protocol: RobotProtocolConfig | None = None,
    ) -> None:
        if client_factory is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            client_factory = WebsocketClientPolicy
        self._client = client_factory(host=host, port=int(port), api_key=api_key)
        self._protocol = protocol
        self.metadata = validate_server_metadata(self._client.get_server_metadata(), protocol)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return validate_inference_response(self._client.infer(dict(observation)), self._protocol)

    def close(self) -> None:
        websocket = getattr(self._client, "_ws", None)
        close = getattr(websocket, "close", None)
        if callable(close):
            close()
