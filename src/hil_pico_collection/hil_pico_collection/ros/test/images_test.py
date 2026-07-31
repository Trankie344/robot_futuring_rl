from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hil_pico_collection.protocol_config import ImageSpec
from hil_pico_collection.ros.images import RosImageDecoder, prepare_configured_image


def raw(encoding, pixels):
    height, width = pixels.shape[:2]
    return SimpleNamespace(
        height=height,
        width=width,
        encoding=encoding,
        step=pixels.strides[0],
        data=pixels.tobytes(),
    )


def test_decode_rgb_image():
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    np.testing.assert_array_equal(RosImageDecoder().decode(raw("rgb8", pixels)), pixels)


def test_decode_bgr_image_to_rgb():
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0] = 10
    result = RosImageDecoder().decode(raw("bgr8", bgr))
    assert np.all(result[..., 2] == 10)


def test_decode_mono_image_to_three_channels():
    mono = np.arange(6, dtype=np.uint8).reshape(2, 3)
    result = RosImageDecoder().decode(raw("mono8", mono))
    assert result.shape == (2, 3, 3)
    np.testing.assert_array_equal(result[..., 0], mono)


def test_decode_compressed_image():
    bgr = np.full((4, 5, 3), 25, np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    message = SimpleNamespace(format="jpeg", data=encoded.tobytes())
    assert RosImageDecoder().decode(message).shape == (4, 5, 3)


@pytest.mark.parametrize("encoding", ["bad", "32FC1"])
def test_decode_rejects_unknown_raw_encoding(encoding):
    with pytest.raises(ValueError):
        RosImageDecoder().decode(raw(encoding, np.zeros((2, 3, 3), np.uint8)))


def test_configured_image_can_resize_to_policy_shape():
    spec = ImageSpec("front", "/front", "fake:Image", width=6, height=4, resize=True)
    result = prepare_configured_image(np.zeros((8, 12, 3), np.uint8), spec)
    assert result.shape == (4, 6, 3)


def test_configured_image_rejects_mismatch_without_resize():
    spec = ImageSpec("front", "/front", "fake:Image", width=6, height=4, resize=False)
    with pytest.raises(ValueError, match="configured shape"):
        prepare_configured_image(np.zeros((8, 12, 3), np.uint8), spec)
