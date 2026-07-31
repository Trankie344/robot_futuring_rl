import numpy as np
import pytest

from hil_pico_collection.bridge.execution import execute_action_chunk, resample_action_chunk


def chunk():
    return np.arange(20 * 16, dtype=np.float32).reshape(20, 16)


def test_resample_20_to_30_preserves_endpoints():
    result = resample_action_chunk(chunk())
    assert result.shape == (30, 16)
    np.testing.assert_array_equal(result[0], chunk()[0])
    np.testing.assert_array_equal(result[-1], chunk()[-1])


def test_resample_is_linear_per_dimension():
    values = np.zeros((20, 16), np.float32)
    values[:, 3] = np.linspace(-1, 1, 20)
    result = resample_action_chunk(values)
    np.testing.assert_allclose(result[:, 3], np.linspace(-1, 1, 30), atol=1e-6)


@pytest.mark.parametrize("values", [np.zeros((19, 16)), np.zeros((20, 15)), np.full((20, 16), np.nan)])
def test_resample_rejects_invalid_chunk(values):
    with pytest.raises(ValueError):
        resample_action_chunk(values)


def test_execute_publishes_all_30_commands_at_30hz():
    now = [10.0]
    published = []

    def monotonic():
        return now[0]

    def sleep(seconds):
        now[0] += seconds

    result = execute_action_chunk(
        chunk(),
        guard=lambda: True,
        build_command=lambda action, stamp, frame_id: (action.copy(), stamp, frame_id),
        publish_command=published.append,
        stamp=lambda: "stamp",
        monotonic=monotonic,
        sleep=sleep,
    )
    assert result.completed is True
    assert result.sent_count == 30
    assert len(published) == 30
    assert now[0] == pytest.approx(11.0)
    assert published[0][1:] == ("stamp", "rl_token_stage2")


def test_execute_stops_immediately_when_control_is_lost():
    calls = [0]

    def guard():
        calls[0] += 1
        return calls[0] <= 4

    published = []
    result = execute_action_chunk(
        chunk(),
        guard=guard,
        build_command=lambda action, stamp, frame_id: action,
        publish_command=published.append,
        stamp=lambda: None,
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )
    assert result.completed is False
    assert result.sent_count == 4
    assert len(published) == 4


def test_execute_stops_when_adapter_rejects_command():
    result = execute_action_chunk(
        chunk(),
        guard=lambda: True,
        build_command=lambda *_: (_ for _ in ()).throw(ValueError("limit")),
        publish_command=lambda _: None,
        stamp=lambda: None,
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )
    assert result.sent_count == 0
    assert "limit" in result.reason


def test_execute_stops_when_publish_fails():
    result = execute_action_chunk(
        chunk(),
        guard=lambda: True,
        build_command=lambda action, *_: action,
        publish_command=lambda _: (_ for _ in ()).throw(RuntimeError("publisher")),
        stamp=lambda: None,
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )
    assert result.sent_count == 0
    assert "publisher" in result.reason
