"""ROS image decoding with no import-time ROS dependency."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from hil_pico_collection.protocol_config import ImageSpec


class RosImageDecoder:
    def __init__(self, bridge_class: Any = None) -> None:
        self._bridge = bridge_class() if bridge_class is not None else None

    def decode(self, message: Any) -> np.ndarray:
        if _looks_compressed(message):
            return _decode_compressed(message)
        if self._bridge is not None:
            try:
                return _as_rgb_uint8(self._bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"))
            except Exception:
                pass
        return _decode_raw(message)


def prepare_configured_image(image: Any, spec: ImageSpec) -> np.ndarray:
    result = _as_rgb_uint8(image)
    if result.shape == spec.shape:
        return result
    if not spec.resize:
        raise ValueError(f"{spec.name} image must have configured shape {spec.shape}, got {result.shape}")
    resized = cv2.resize(result, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    if resized.shape != spec.shape:
        raise ValueError(f"resized {spec.name} image has shape {resized.shape}, expected {spec.shape}")
    return np.ascontiguousarray(resized)


def _looks_compressed(message: Any) -> bool:
    return hasattr(message, "format") and not hasattr(message, "height")


def _decode_compressed(message: Any) -> np.ndarray:
    encoded = np.frombuffer(bytes(getattr(message, "data")), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("compressed image could not be decoded")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _as_rgb_uint8(image: Any) -> np.ndarray:
    result = np.asarray(image)
    if result.dtype != np.uint8:
        result = result.astype(np.uint8, copy=False)
    if result.ndim == 2:
        result = np.repeat(result[:, :, None], 3, axis=2)
    if result.ndim != 3 or result.shape[2] < 3:
        raise ValueError("decoded image must have at least three channels")
    return np.ascontiguousarray(result[:, :, :3])


def _decode_raw(message: Any) -> np.ndarray:
    height = int(getattr(message, "height"))
    width = int(getattr(message, "width"))
    encoding = _encoding(message)
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "yuyv": 2,
        "yuv422": 2,
        "yuv422_yuy2": 2,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding without cv_bridge: {encoding}")
    channels = channels_by_encoding[encoding]
    row_width = width * channels
    step = int(getattr(message, "step", row_width) or row_width)
    if step < row_width:
        raise ValueError("image step is smaller than width * channels")
    data = np.frombuffer(bytes(getattr(message, "data")), dtype=np.uint8)
    if data.size < height * step:
        raise ValueError("image data is shorter than height * step")
    pixels = data[: height * step].reshape(height, step)[:, :row_width].reshape(height, width, channels)
    if encoding == "rgb8":
        rgb = pixels
    elif encoding == "bgr8":
        rgb = pixels[:, :, ::-1]
    elif encoding == "rgba8":
        rgb = pixels[:, :, :3]
    elif encoding == "bgra8":
        rgb = pixels[:, :, [2, 1, 0]]
    elif channels == 2:
        rgb = cv2.cvtColor(pixels, cv2.COLOR_YUV2RGB_YUY2)
    else:
        rgb = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(rgb)


def _encoding(message: Any) -> str:
    value = getattr(message, "encoding", "rgb8")
    if hasattr(value, "data"):
        raw = bytes(int(item) for item in value.data)
        value = raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    return str(value or "rgb8").lower().replace("-", "_")
