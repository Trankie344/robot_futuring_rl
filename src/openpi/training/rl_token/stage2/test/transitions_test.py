from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import transitions
import openpi.transforms as openpi_transforms


def _episode(
    episode_index: int,
    length: int,
    *,
    labels: np.ndarray | None = None,
) -> admission.ValidatedEpisode:
    if labels is None:
        labels = np.zeros(length, dtype=np.int8)
    return admission.ValidatedEpisode(
        episode_index=episode_index,
        length=length,
        dataset_from_index=0,
        dataset_to_index=length,
        task="fold clothes",
        parquet_path=Path(f"/not-read/episode_{episode_index:06d}.parquet"),
        parquet_size=0,
        parquet_sha256="0" * 64,
        parquet_device=0,
        parquet_inode=0,
        labels=labels,
        intervention=np.zeros(length, dtype=np.bool_),
    )


def _batch(
    lengths: tuple[int, ...],
    *,
    episode_indices: tuple[int, ...] | None = None,
) -> admission.ValidatedBatch:
    if episode_indices is None:
        episode_indices = tuple(range(len(lengths)))
    episodes = tuple(_episode(index, length) for index, length in zip(episode_indices, lengths, strict=True))
    return admission.ValidatedBatch(
        batch_id="batch_000001_test",
        root=Path("/not-read"),
        fps=30,
        total_frames=sum(lengths),
        manifest_sha256="a" * 64,
        labels_sha256="b" * 64,
        episode_fingerprints=tuple(f"{index + 1:064x}" for index in range(len(lengths))),
        episodes=episodes,
    )


def _stats() -> dict[str, openpi_transforms.NormStats]:
    return {
        "state": openpi_transforms.NormStats(
            mean=np.zeros(16, dtype=np.float32),
            std=np.ones(16, dtype=np.float32),
            q01=np.linspace(-2.0, -0.5, 16, dtype=np.float32),
            q99=np.linspace(0.75, 3.0, 16, dtype=np.float32),
        ),
        "actions": openpi_transforms.NormStats(
            mean=np.zeros(16, dtype=np.float32),
            std=np.ones(16, dtype=np.float32),
            q01=np.linspace(-4.0, -1.0, 16, dtype=np.float32),
            q99=np.linspace(1.5, 5.0, 16, dtype=np.float32),
        ),
    }


def _write_action_state_parquet(
    path: Path,
    states: np.ndarray,
    actions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = states.shape[0]
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(),
                    type=pa.list_(pa.float32(), 16),
                ),
                "action": pa.array(
                    actions.tolist(),
                    type=pa.list_(pa.float32(), 16),
                ),
                "timestamp": pa.array(
                    np.arange(length, dtype=np.float32) / np.float32(30.0),
                    type=pa.float32(),
                ),
                "fps": pa.array([30] * length, type=pa.int32()),
            }
        ),
        path,
    )


def _disk_batch(
    tmp_path: Path,
    *,
    lengths: tuple[int, ...] = (22, 23),
    episode_indices: tuple[int, ...] = (7, 3),
) -> tuple[admission.ValidatedBatch, dict[int, tuple[np.ndarray, np.ndarray]]]:
    episodes: list[admission.ValidatedEpisode] = []
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode_index, length in zip(episode_indices, lengths, strict=True):
        state_base = np.float32(episode_index * 0.25)
        states = np.arange(length * 16, dtype=np.float32).reshape(length, 16) / np.float32(64.0) + state_base
        actions = (
            np.arange(length * 16, dtype=np.float32).reshape(length, 16) / np.float32(32.0)
            + state_base
            + np.float32(1.0)
        )
        path = tmp_path / f"episode_{episode_index:06d}.parquet"
        _write_action_state_parquet(path, states, actions)
        metadata = path.stat()
        intervention = np.zeros(length, dtype=np.bool_)
        intervention[np.arange(length) % 4 == 1] = True
        labels = (np.arange(length, dtype=np.int8) % 3).astype(np.int8)
        episodes.append(
            dataclasses.replace(
                _episode(episode_index, length, labels=labels),
                parquet_path=path,
                parquet_size=metadata.st_size,
                parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parquet_device=metadata.st_dev,
                parquet_inode=metadata.st_ino,
                intervention=intervention,
            )
        )
        arrays[episode_index] = (states, actions)
    batch = dataclasses.replace(
        _batch(lengths, episode_indices=episode_indices),
        root=tmp_path,
        episodes=tuple(episodes),
    )
    return batch, arrays


def _refresh_episode_parquet_identity(
    batch: admission.ValidatedBatch,
    episode_index: int,
) -> admission.ValidatedBatch:
    episodes = list(batch.episodes)
    position = next(index for index, episode in enumerate(episodes) if episode.episode_index == episode_index)
    episode = episodes[position]
    metadata = episode.parquet_path.stat()
    episodes[position] = dataclasses.replace(
        episode,
        parquet_size=metadata.st_size,
        parquet_sha256=hashlib.sha256(episode.parquet_path.read_bytes()).hexdigest(),
        parquet_device=metadata.st_dev,
        parquet_inode=metadata.st_ino,
    )
    return dataclasses.replace(batch, episodes=tuple(episodes))


def test_transition_dataclasses_have_exact_frozen_schema_and_feature_order():
    assert tuple(field.name for field in dataclasses.fields(transitions.FeatureKey)) == (
        "batch_id",
        "episode_index",
        "frame_index",
    )
    assert tuple(field.name for field in dataclasses.fields(transitions.TransitionRow)) == (
        "batch_id",
        "episode_index",
        "start_frame_index",
        "current_key",
        "next_key",
        "reward",
        "terminal",
    )
    assert tuple(field.name for field in dataclasses.fields(transitions.TransitionPlan)) == (
        "rows",
        "feature_keys",
        "chunk_equivalents",
    )

    late = transitions.FeatureKey("batch", 1, 0)
    early = transitions.FeatureKey("batch", 0, 20)
    assert sorted((late, early)) == [early, late]
    with pytest.raises(dataclasses.FrozenInstanceError):
        early.frame_index = 21

    row = transitions.TransitionRow(
        batch_id="batch",
        episode_index=0,
        start_frame_index=0,
        current_key=early,
        next_key=late,
        reward=0.0,
        terminal=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.reward = 1.0
    plan = transitions.TransitionPlan((row,), (early, late), 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.chunk_equivalents = 2


@pytest.mark.parametrize("length", range(20, 46))
def test_transition_starts_cover_stride_two_and_force_exact_tail(length: int):
    assert transitions.ACTION_HORIZON == 20
    assert transitions.TRANSITION_STRIDE == 2
    starts = transitions.transition_starts(length)
    tail = length - transitions.ACTION_HORIZON
    expected = tuple(sorted({*range(0, tail + 1, 2), tail}))

    assert starts == expected
    assert starts == tuple(sorted(set(starts)))
    assert starts[0] == 0
    assert starts[-1] == tail
    assert all(
        tuple(range(start, start + transitions.ACTION_HORIZON)) == tuple(range(start, start + 20)) for start in starts
    )
    assert all(start >= 0 and start + transitions.ACTION_HORIZON <= length for start in starts)
    for start in starts:
        labels = np.zeros(length, dtype=np.int8)
        labels[start] = 1
        labels[start + transitions.ACTION_HORIZON - 1] = 2
        if start > 0:
            labels[start - 1] = 2
        if start + transitions.ACTION_HORIZON < length:
            labels[start + transitions.ACTION_HORIZON] = 2
        assert transitions.chunk_reward(labels, start) == 3.0


def test_transition_starts_accept_numpy_integer():
    assert transitions.transition_starts(np.int64(23)) == (0, 2, 3)


@pytest.mark.parametrize(
    ("length", "error"),
    [
        (True, "must be an integer"),
        (False, "must be an integer"),
        (20.0, "must be an integer"),
        (np.float32(20), "must be an integer"),
        ("20", "must be an integer"),
        (None, "must be an integer"),
        (-1, "at least 20"),
        (19, "at least 20"),
    ],
)
def test_transition_starts_reject_invalid_or_short_lengths(length: object, error: str):
    with pytest.raises(ValueError, match=error):
        transitions.transition_starts(length)  # type: ignore[arg-type]


def test_reward_sums_exactly_twenty_collection_frames_without_discount():
    labels = np.zeros(24, dtype=np.int8)
    labels[[3, 11, 19]] = [1, 1, 2]

    reward_at_zero = transitions.chunk_reward(labels, 0)
    reward_at_four = transitions.chunk_reward(labels, 4)

    assert reward_at_zero == 4.0
    assert reward_at_four == 3.0
    assert isinstance(reward_at_zero, float)


def test_reward_preserves_negative_progress_and_completion_values():
    labels = np.asarray([-1, 0, 1, 2] * 5, dtype=np.int8)

    assert transitions.chunk_reward(labels, 0) == 10.0
    assert transitions.reward_metadata("a" * 64) == {
        "stage1_config": "rl_token_stage1",
        "stage2_config": "rl_token_stage2",
        "reward_source": "tristate",
        "reward_label_values": [-1, 0, 1, 2],
        "completion_label": 2,
        "reward_aggregation": "sum_20_frames",
        "reward_schema_version": 1,
        "tristate_labels_sha256": "a" * 64,
    }


def test_overlapping_reward_windows_intentionally_repeat_sparse_events():
    labels = np.zeros(24, dtype=np.int8)
    labels[[3, 11, 19]] = [1, 1, 2]

    assert transitions.chunk_reward(labels, 0) == 4.0
    assert transitions.chunk_reward(labels, 2) == 4.0


@pytest.mark.parametrize(
    ("start", "error"),
    [
        (True, "must be an integer"),
        (False, "must be an integer"),
        (0.0, "must be an integer"),
        (np.float32(0), "must be an integer"),
        ("0", "must be an integer"),
        (None, "must be an integer"),
        (-1, "must be nonnegative"),
        (5, "incomplete reward window"),
    ],
)
def test_chunk_reward_rejects_invalid_start_or_incomplete_window(start: object, error: str):
    labels = np.zeros(24, dtype=np.int8)
    with pytest.raises(ValueError, match=error):
        transitions.chunk_reward(labels, start)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("labels", "error"),
    [
        (np.zeros((20, 1), dtype=np.int8), "one-dimensional integer array"),
        (np.zeros((1, 20), dtype=np.int8), "one-dimensional integer array"),
        (np.zeros(20, dtype=np.bool_), "one-dimensional integer array"),
        (np.zeros(20, dtype=np.float32), "one-dimensional integer array"),
        (np.zeros(19, dtype=np.int8), "incomplete reward window"),
    ],
)
def test_chunk_reward_rejects_non_vector_noninteger_or_short_labels(labels: np.ndarray, error: str):
    with pytest.raises(ValueError, match=error):
        transitions.chunk_reward(labels, 0)


def test_chunk_reward_accepts_numpy_integer_start_and_does_not_mutate_labels():
    labels = np.arange(24, dtype=np.int8) % 3
    before = labels.copy()
    writeable = labels.flags.writeable

    assert transitions.chunk_reward(labels, np.int64(2)) == float(np.sum(labels[2:22], dtype=np.int64))
    np.testing.assert_array_equal(labels, before)
    assert labels.flags.writeable is writeable


def test_build_plan_preserves_episode_then_start_order_and_uses_t_plus_twenty():
    first_labels = np.zeros(22, dtype=np.int8)
    first_labels[-1] = 2
    second_labels = np.zeros(23, dtype=np.int8)
    second_labels[[2, 22]] = [1, 2]
    batch = dataclasses.replace(
        _batch((22, 23), episode_indices=(7, 3)),
        episodes=(
            _episode(7, 22, labels=first_labels),
            _episode(3, 23, labels=second_labels),
        ),
    )

    plan = transitions.build_transition_plan(batch)

    assert [(row.episode_index, row.start_frame_index) for row in plan.rows] == [
        (7, 0),
        (7, 2),
        (3, 0),
        (3, 2),
        (3, 3),
    ]
    assert [(row.next_frame_index, row.terminal) for row in plan.rows] == [
        (20, False),
        (None, True),
        (20, False),
        (22, False),
        (None, True),
    ]
    assert [row.reward for row in plan.rows] == [0.0, 2.0, 1.0, 1.0, 2.0]
    for row in plan.rows:
        assert row.batch_id == batch.batch_id
        assert row.current_key == transitions.FeatureKey(batch.batch_id, row.episode_index, row.start_frame_index)
        if row.terminal:
            assert row.next_key is None
        else:
            assert row.next_key == transitions.FeatureKey(
                batch.batch_id,
                row.episode_index,
                row.start_frame_index + transitions.ACTION_HORIZON,
            )


def test_build_plan_integrates_with_admission_and_has_exact_episode_feature_keys(ready_batch: Path):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    plan = transitions.build_transition_plan(batch)
    episode_zero = [row for row in plan.rows if row.episode_index == 0]
    episode_zero_keys = tuple(key for key in plan.feature_keys if key.episode_index == 0)

    assert [(row.start_frame_index, row.next_frame_index, row.terminal) for row in episode_zero] == [
        (0, 20, False),
        (2, None, True),
    ]
    assert episode_zero_keys == (
        transitions.FeatureKey(batch.batch_id, 0, 0),
        transitions.FeatureKey(batch.batch_id, 0, 2),
        transitions.FeatureKey(batch.batch_id, 0, 20),
    )
    assert plan.feature_keys == tuple(sorted(set(plan.feature_keys)))


def test_build_plan_feature_keys_are_exact_sorted_unique_set_across_episodes():
    batch = _batch((42, 23), episode_indices=(1, 0))

    plan = transitions.build_transition_plan(batch)
    expected = {row.current_key for row in plan.rows} | {row.next_key for row in plan.rows if row.next_key is not None}

    assert plan.feature_keys == tuple(sorted(expected))
    assert plan.feature_keys == tuple(sorted(set(plan.feature_keys)))
    assert plan.feature_keys.count(transitions.FeatureKey(batch.batch_id, 1, 20)) == 1
    assert {key.episode_index for key in plan.feature_keys} == {0, 1}


@pytest.mark.parametrize("length", range(20, 46))
def test_build_plan_has_one_tail_terminal_and_exact_nonterminal_next(length: int):
    batch = _batch((length,))

    plan = transitions.build_transition_plan(batch)
    starts = transitions.transition_starts(length)

    assert len(plan.rows) == len(starts)
    assert [row.start_frame_index for row in plan.rows] == list(starts)
    assert sum(row.terminal for row in plan.rows) == 1
    assert plan.rows[-1].start_frame_index == length - transitions.ACTION_HORIZON
    assert plan.rows[-1].terminal
    assert plan.rows[-1].next_key is None
    for row in plan.rows[:-1]:
        assert not row.terminal
        assert row.next_frame_index == row.start_frame_index + transitions.ACTION_HORIZON


@pytest.mark.parametrize("length", range(1, 20))
def test_build_plan_short_episode_error_identifies_batch_and_episode(length: int):
    batch = _batch((length,), episode_indices=(7,))

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id} episode 7",
    ) as error:
        transitions.build_transition_plan(batch)

    assert isinstance(error.value.__cause__, ValueError)
    assert "at least 20" in str(error.value.__cause__)


@pytest.mark.parametrize(
    ("length", "labels", "window", "cause"),
    [
        (20, np.zeros((20, 1), dtype=np.int8), 0, "one-dimensional integer array"),
        (22, np.zeros(21, dtype=np.int8), 2, "incomplete reward window"),
    ],
)
def test_build_plan_bad_reward_window_identifies_batch_episode_and_window(
    length: int,
    labels: np.ndarray,
    window: int,
    cause: str,
):
    batch = dataclasses.replace(
        _batch((length,), episode_indices=(7,)),
        episodes=(_episode(7, length, labels=labels),),
    )

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id} episode 7 window {window}",
    ) as error:
        transitions.build_transition_plan(batch)

    assert isinstance(error.value.__cause__, ValueError)
    assert cause in str(error.value.__cause__)


def test_build_plan_rejects_empty_batch_with_batch_identity():
    batch = dataclasses.replace(
        _batch((20,)),
        total_frames=0,
        episode_fingerprints=(),
        episodes=(),
    )

    with pytest.raises(ValueError, match=rf"batch {batch.batch_id}.*no episodes"):
        transitions.build_transition_plan(batch)


def test_chunk_equivalents_are_independent_of_overlapping_transition_count():
    lengths = tuple(range(20, 46))
    batch = _batch(lengths)

    plan = transitions.build_transition_plan(batch)

    assert plan.chunk_equivalents == sum(math.ceil(length / 20) for length in lengths)
    assert len(plan.rows) == sum(len(transitions.transition_starts(length)) for length in lengths)
    assert len(plan.rows) > plan.chunk_equivalents


def test_build_plan_does_not_consult_timing_or_parquet_and_does_not_mutate_input():
    labels = np.zeros(22, dtype=np.int8)
    labels[-1] = 2
    labels_before = labels.copy()
    frame_labels = labels

    class NoTimingEpisode:
        episode_index = 0
        length = 22
        labels = frame_labels

        @property
        def fps(self):
            raise AssertionError("transition planning must not read fps")

        @property
        def timestamps(self):
            raise AssertionError("transition planning must not read timestamps")

        @property
        def parquet_path(self):
            raise AssertionError("transition planning must not read parquet")

    class NoTimingBatch:
        batch_id = "batch_000001_test"
        episodes = (NoTimingEpisode(),)
        chunk_equivalents = 2

        @property
        def fps(self):
            raise AssertionError("transition planning must not read fps")

        @property
        def timestamps(self):
            raise AssertionError("transition planning must not read timestamps")

    batch = NoTimingBatch()
    episodes_before = batch.episodes

    plan = transitions.build_transition_plan(batch)  # type: ignore[arg-type]

    assert len(plan.rows) == 2
    assert batch.episodes is episodes_before
    np.testing.assert_array_equal(labels, labels_before)


def test_stage2_normalizer_matches_exact_openpi_delta_then_quantile_normalize():
    stats = _stats()
    state = np.arange(16, dtype=np.float32) / np.float32(10.0)
    actions = np.stack([state + np.float32(step) / np.float32(20.0) for step in range(20)])
    state_before = state.copy()
    actions_before = actions.copy()
    stats_before = {
        key: (value.mean.copy(), value.std.copy(), value.q01.copy(), value.q99.copy()) for key, value in stats.items()
    }
    normalizer = transitions.Stage2Normalizer.from_norm_stats(stats)

    actual = normalizer.executed_action(state, actions)

    expected_data = {"state": state.copy(), "actions": actions.copy()}
    expected_data = openpi_transforms.DeltaActions(openpi_transforms.make_bool_mask(16))(expected_data)
    expected_data = openpi_transforms.Normalize(
        stats,
        use_quantiles=True,
        strict=True,
    )(expected_data)
    np.testing.assert_allclose(actual, expected_data["actions"], rtol=0, atol=0)
    assert actual.shape == (20, 16)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(actions, actions_before)
    for key, value in stats.items():
        for actual_stat, expected_stat in zip(
            (value.mean, value.std, value.q01, value.q99),
            stats_before[key],
            strict=True,
        ):
            np.testing.assert_array_equal(actual_stat, expected_stat)


def test_stage2_normalizer_state_matches_exact_openpi_quantile_normalize():
    stats = _stats()
    state = np.linspace(-1.75, 1.75, 16, dtype=np.float32)
    before = state.copy()
    normalizer = transitions.Stage2Normalizer.from_norm_stats(stats)

    actual = normalizer.state(state)
    expected = openpi_transforms.Normalize(
        {"state": stats["state"]},
        use_quantiles=True,
        strict=True,
    )({"state": state.copy()})["state"]

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert actual.shape == (16,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(state, before)


def test_stage2_normalizer_is_frozen_and_deep_copies_stats():
    stats = _stats()
    original_q01 = stats["state"].q01.copy()
    normalizer = transitions.Stage2Normalizer.from_norm_stats(stats)
    baseline = normalizer.state(np.zeros(16, dtype=np.float32))

    stats["state"].q01[:] = np.float32(-100.0)
    stats["state"] = _stats()["state"]

    np.testing.assert_array_equal(normalizer.norm_stats["state"].q01, original_q01)
    np.testing.assert_array_equal(
        normalizer.state(np.zeros(16, dtype=np.float32)),
        baseline,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        normalizer.norm_stats = stats
    with pytest.raises(TypeError):
        normalizer.norm_stats["state"] = _stats()["state"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        normalizer.norm_stats["state"].q01 = np.zeros(16, dtype=np.float32)
    for field_name in ("mean", "std", "q01", "q99"):
        value = getattr(normalizer.norm_stats["state"], field_name)
        assert not value.flags.owndata
        with pytest.raises(ValueError, match="WRITEABLE"):
            value.setflags(write=True)
        with pytest.raises(ValueError, match="read-only"):
            value[0] = np.float32(123.0)
    np.testing.assert_array_equal(
        normalizer.state(np.zeros(16, dtype=np.float32)),
        baseline,
    )


@pytest.mark.parametrize(
    "stats",
    [
        {"state": _stats()["state"]},
        {"actions": _stats()["actions"]},
        (),
    ],
)
def test_stage2_normalizer_requires_state_and_action_stats(stats: object):
    with pytest.raises(ValueError, match=r"norm stats.*actions.*state|norm stats.*state.*actions"):
        transitions.Stage2Normalizer.from_norm_stats(stats)


@pytest.mark.parametrize(
    ("key", "q01", "q99", "error"),
    [
        ("state", None, np.ones(16, dtype=np.float32), "quantile"),
        ("actions", np.zeros(16, dtype=np.float32), None, "quantile"),
        (
            "state",
            np.zeros(15, dtype=np.float32),
            np.ones(15, dtype=np.float32),
            "16",
        ),
        (
            "actions",
            np.full(16, np.nan, dtype=np.float32),
            np.ones(16, dtype=np.float32),
            "finite",
        ),
        (
            "actions",
            np.ones(16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
            "q99",
        ),
    ],
)
def test_stage2_normalizer_rejects_bad_quantile_stats(
    key: str,
    q01: np.ndarray | None,
    q99: np.ndarray | None,
    error: str,
):
    stats = _stats()
    current = stats[key]
    stats[key] = openpi_transforms.NormStats(
        mean=current.mean,
        std=current.std,
        q01=q01,
        q99=q99,
    )

    with pytest.raises(ValueError, match=error):
        transitions.Stage2Normalizer.from_norm_stats(stats)


def test_stage2_normalizer_keeps_equal_quantiles_legal():
    stats = _stats()
    for key in ("state", "actions"):
        current = stats[key]
        equal_quantile = np.zeros(16, dtype=np.float32)
        stats[key] = openpi_transforms.NormStats(
            mean=current.mean,
            std=current.std,
            q01=equal_quantile.copy(),
            q99=equal_quantile.copy(),
        )

    normalizer = transitions.Stage2Normalizer.from_norm_stats(stats)

    state = normalizer.state(np.zeros(16, dtype=np.float32))
    action = normalizer.executed_action(
        np.zeros(16, dtype=np.float32),
        np.zeros((20, 16), dtype=np.float32),
    )
    assert np.isfinite(state).all()
    assert np.isfinite(action).all()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_stage2_normalizer_rejects_complex_state_without_mutating_input(dtype):
    state = np.full(16, 1.0 + 2.0j, dtype=dtype)
    before = state.copy()
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())

    with pytest.raises(ValueError, match="state"):
        normalizer.state(state)

    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_stage2_normalizer_rejects_complex_start_state_without_mutating_input(dtype):
    state = np.full(16, 1.0 + 2.0j, dtype=dtype)
    actions = np.zeros((20, 16), dtype=np.float32)
    state_before = state.copy()
    actions_before = actions.copy()
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())

    with pytest.raises(ValueError, match="start state"):
        normalizer.executed_action(state, actions)

    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(actions, actions_before)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_stage2_normalizer_rejects_complex_actions_without_mutating_input(dtype):
    state = np.zeros(16, dtype=np.float32)
    actions = np.full((20, 16), 1.0 + 2.0j, dtype=dtype)
    state_before = state.copy()
    actions_before = actions.copy()
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())

    with pytest.raises(ValueError, match="actions"):
        normalizer.executed_action(state, actions)

    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(actions, actions_before)


@pytest.mark.parametrize("field_name", ["mean", "std", "q01", "q99"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_stage2_normalizer_rejects_complex_norm_stats_without_mutating_input(
    field_name: str,
    dtype,
):
    stats = _stats()
    current = stats["state"]
    values = {name: np.asarray(getattr(current, name)).copy() for name in ("mean", "std", "q01", "q99")}
    values[field_name] = values[field_name].astype(dtype) + dtype(1.0j)
    before = values[field_name].copy()
    stats["state"] = openpi_transforms.NormStats(**values)

    with pytest.raises(ValueError, match=field_name):
        transitions.Stage2Normalizer.from_norm_stats(stats)

    np.testing.assert_array_equal(values[field_name], before)


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(15, dtype=np.float32),
        np.zeros(17, dtype=np.float32),
        np.zeros((1, 16), dtype=np.float32),
        np.full(16, np.nan, dtype=np.float32),
        np.full(16, np.inf, dtype=np.float32),
    ],
)
def test_stage2_normalizer_rejects_bad_state_input(state: np.ndarray):
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())

    with pytest.raises(ValueError, match=r"state.*finite.*shape|state.*shape.*finite"):
        normalizer.state(state)


@pytest.mark.parametrize(
    ("state", "actions"),
    [
        (np.zeros(15, dtype=np.float32), np.zeros((20, 16), dtype=np.float32)),
        (np.zeros(16, dtype=np.float32), np.zeros((19, 16), dtype=np.float32)),
        (np.zeros(16, dtype=np.float32), np.zeros((20, 15), dtype=np.float32)),
        (np.full(16, np.nan, dtype=np.float32), np.zeros((20, 16), dtype=np.float32)),
        (np.zeros(16, dtype=np.float32), np.full((20, 16), np.inf, dtype=np.float32)),
    ],
)
def test_stage2_normalizer_rejects_bad_executed_action_input(
    state: np.ndarray,
    actions: np.ndarray,
):
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())

    with pytest.raises(ValueError, match=r"finite.*shape|shape.*finite"):
        normalizer.executed_action(state, actions)


def test_bc_anchor_replaces_only_intervention_frames_without_mutating_inputs():
    reference = np.full((20, 16), -0.25, dtype=np.float32)
    executed = np.full((20, 16), 0.75, dtype=np.float32)
    intervention = np.zeros(20, dtype=np.bool_)
    intervention[[3, 4, 17]] = True
    reference_before = reference.copy()
    executed_before = executed.copy()
    intervention_before = intervention.copy()

    anchor = transitions.bc_anchor(reference, executed, intervention)

    np.testing.assert_array_equal(anchor[intervention], executed[intervention])
    np.testing.assert_array_equal(anchor[~intervention], reference[~intervention])
    assert anchor.shape == (20, 16)
    assert anchor.dtype == np.float32
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(executed, executed_before)
    np.testing.assert_array_equal(intervention, intervention_before)


@pytest.mark.parametrize("value", [False, True])
def test_bc_anchor_all_false_or_all_true(value):
    reference = np.arange(320, dtype=np.float32).reshape(20, 16)
    executed = reference + np.float32(1000.0)
    intervention = np.full(20, value, dtype=np.bool_)

    actual = transitions.bc_anchor(reference, executed, intervention)

    np.testing.assert_array_equal(actual, executed if value else reference)


@pytest.mark.parametrize("target", ["reference", "executed"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_bc_anchor_rejects_complex_actions_without_mutating_inputs(
    target: str,
    dtype,
):
    reference = np.zeros((20, 16), dtype=np.float32)
    executed = np.ones((20, 16), dtype=np.float32)
    if target == "reference":
        reference = reference.astype(dtype) + dtype(2.0j)
    else:
        executed = executed.astype(dtype) + dtype(2.0j)
    intervention = np.zeros(20, dtype=np.bool_)
    reference_before = reference.copy()
    executed_before = executed.copy()
    intervention_before = intervention.copy()

    with pytest.raises(ValueError, match=target):
        transitions.bc_anchor(reference, executed, intervention)

    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(executed, executed_before)
    np.testing.assert_array_equal(intervention, intervention_before)


@pytest.mark.parametrize(
    ("reference", "executed", "intervention", "error"),
    [
        (
            np.zeros((19, 16), dtype=np.float32),
            np.zeros((20, 16), dtype=np.float32),
            np.zeros(20, dtype=np.bool_),
            "20, ?16",
        ),
        (
            np.zeros((20, 16), dtype=np.float32),
            np.zeros((20, 15), dtype=np.float32),
            np.zeros(20, dtype=np.bool_),
            "20, ?16",
        ),
        (
            np.zeros((20, 16), dtype=np.float32),
            np.zeros((20, 16), dtype=np.float32),
            np.zeros((20, 1), dtype=np.bool_),
            "20",
        ),
        (
            np.zeros((20, 16), dtype=np.float32),
            np.zeros((20, 16), dtype=np.float32),
            np.zeros(20, dtype=np.int8),
            "bool",
        ),
        (
            np.zeros((20, 16), dtype=np.float32),
            np.zeros((20, 16), dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            "bool",
        ),
        (
            np.full((20, 16), np.nan, dtype=np.float32),
            np.zeros((20, 16), dtype=np.float32),
            np.zeros(20, dtype=np.bool_),
            "finite",
        ),
        (
            np.zeros((20, 16), dtype=np.float32),
            np.full((20, 16), np.inf, dtype=np.float32),
            np.zeros(20, dtype=np.bool_),
            "finite",
        ),
    ],
)
def test_bc_anchor_rejects_invalid_shape_mask_dtype_or_nonfinite(
    reference: np.ndarray,
    executed: np.ndarray,
    intervention: np.ndarray,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        transitions.bc_anchor(reference, executed, intervention)


def test_raw_transition_table_has_exact_frozen_schema():
    assert tuple(field.name for field in dataclasses.fields(transitions.RawTransitionTable)) == (
        "episode_index",
        "start_frame_index",
        "executed_action",
        "intervention",
        "reward",
        "terminal",
    )
    table = transitions.RawTransitionTable(
        episode_index=np.zeros(1, dtype=np.int32),
        start_frame_index=np.zeros(1, dtype=np.int32),
        executed_action=np.zeros((1, 20, 16), dtype=np.float32),
        intervention=np.zeros((1, 20), dtype=np.bool_),
        reward=np.zeros((1, 1), dtype=np.float32),
        terminal=np.zeros((1, 1), dtype=np.bool_),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.reward = np.ones((1, 1), dtype=np.float32)


def test_build_raw_transition_table_preserves_plan_order_and_exact_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch, raw_arrays = _disk_batch(tmp_path)
    canonical = transitions.build_transition_plan(batch)
    order = (canonical.rows[-1], canonical.rows[0], *canonical.rows[1:-1])
    plan = dataclasses.replace(canonical, rows=order)
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())
    original_read_table = pq.read_table
    reads: list[tuple[object, tuple[str, ...]]] = []

    def recording_read_table(source: object, *, columns: list[str]):
        assert isinstance(source, pa.BufferReader)
        reads.append((source, tuple(columns)))
        return original_read_table(source, columns=columns)

    monkeypatch.setattr(transitions.pq, "read_table", recording_read_table)

    actual = transitions.build_raw_transition_table(batch, plan, normalizer)

    assert actual.episode_index.shape == (len(order),)
    assert actual.start_frame_index.shape == (len(order),)
    assert actual.executed_action.shape == (len(order), 20, 16)
    assert actual.intervention.shape == (len(order), 20)
    assert actual.reward.shape == (len(order), 1)
    assert actual.terminal.shape == (len(order), 1)
    assert actual.episode_index.dtype == np.int32
    assert actual.start_frame_index.dtype == np.int32
    assert actual.executed_action.dtype == np.float32
    assert actual.intervention.dtype == np.bool_
    assert actual.reward.dtype == np.float32
    assert actual.terminal.dtype == np.bool_
    np.testing.assert_array_equal(
        actual.episode_index,
        np.asarray([row.episode_index for row in order], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        actual.start_frame_index,
        np.asarray([row.start_frame_index for row in order], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        actual.reward,
        np.asarray([[row.reward] for row in order], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        actual.terminal,
        np.asarray([[row.terminal] for row in order], dtype=np.bool_),
    )
    episodes_by_id = {episode.episode_index: episode for episode in batch.episodes}
    for output_index, row in enumerate(order):
        states, actions = raw_arrays[row.episode_index]
        start = row.start_frame_index
        np.testing.assert_array_equal(
            actual.executed_action[output_index],
            normalizer.executed_action(states[start], actions[start : start + 20]),
        )
        np.testing.assert_array_equal(
            actual.intervention[output_index],
            episodes_by_id[row.episode_index].intervention[start : start + 20],
        )
    assert len(reads) == len(batch.episodes)
    assert all(columns == ("observation.state", "action") for _, columns in reads)


def test_build_raw_rejects_replaced_parquet_after_admission(tmp_path: Path):
    batch, raw_arrays = _disk_batch(
        tmp_path,
        lengths=(20,),
        episode_indices=(5,),
    )
    episode = batch.episodes[0]
    states, actions = raw_arrays[5]
    replacement = tmp_path / "replacement.parquet"
    _write_action_state_parquet(
        replacement,
        states,
        actions + np.float32(100.0),
    )
    os.replace(replacement, episode.parquet_path)

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 5.*parquet.*(size|sha256|inode|device)",
    ):
        transitions.build_raw_transition_table(
            batch,
            transitions.build_transition_plan(batch),
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )


def test_build_raw_rejects_symlinked_parquet_after_admission(tmp_path: Path):
    batch, _ = _disk_batch(
        tmp_path,
        lengths=(20,),
        episode_indices=(5,),
    )
    episode = batch.episodes[0]
    target = tmp_path / "original.parquet"
    episode.parquet_path.rename(target)
    episode.parquet_path.symlink_to(target)

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 5.*parquet",
    ):
        transitions.build_raw_transition_table(
            batch,
            transitions.build_transition_plan(batch),
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )


def test_build_raw_rejects_symlinked_parquet_ancestor_after_admission(tmp_path: Path):
    parquet_parent = tmp_path / "batch/data"
    batch, _ = _disk_batch(
        parquet_parent,
        lengths=(20,),
        episode_indices=(5,),
    )
    moved_parent = tmp_path / "moved-data"
    parquet_parent.rename(moved_parent)
    parquet_parent.symlink_to(moved_parent, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 5.*parquet.*(ancestor|symlink|directory)",
    ):
        transitions.build_raw_transition_table(
            batch,
            transitions.build_transition_plan(batch),
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )


def test_verified_parquet_ancestor_chain_failures_do_not_leak_fds(tmp_path: Path):
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("requires Linux /proc fd accounting")
    parquet_parent = tmp_path / "batch/data"
    batch, _ = _disk_batch(
        parquet_parent,
        lengths=(20,),
        episode_indices=(5,),
    )
    episode = batch.episodes[0]
    moved_parent = tmp_path / "moved-data"
    parquet_parent.rename(moved_parent)
    parquet_parent.symlink_to(moved_parent, target_is_directory=True)
    before = len(os.listdir("/proc/self/fd"))

    for _ in range(20):
        with pytest.raises(admission.AdmissionError, match="ancestor|symlink|directory"):
            admission._read_verified_regular_file(  # noqa: SLF001
                episode.parquet_path,
                expected_size=episode.parquet_size,
                expected_sha256=episode.parquet_sha256,
                expected_device=episode.parquet_device,
                expected_inode=episode.parquet_inode,
            )

    assert len(os.listdir("/proc/self/fd")) == before


def test_verified_parquet_rechecks_ancestor_chain_after_pinned_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parquet_parent = tmp_path / "batch/data"
    batch, _ = _disk_batch(
        parquet_parent,
        lengths=(20,),
        episode_indices=(5,),
    )
    episode = batch.episodes[0]
    moved_parent = tmp_path / "moved-data"

    def replace_ancestor(_path: Path, _descriptor: int) -> None:
        parquet_parent.rename(moved_parent)
        parquet_parent.symlink_to(moved_parent, target_is_directory=True)

    monkeypatch.setattr(admission, "_after_verified_file_open", replace_ancestor)

    with pytest.raises(admission.AdmissionError, match=r"pathname changed.*(ancestor|symlink|directory)"):
        admission._read_verified_regular_file(  # noqa: SLF001
            episode.parquet_path,
            expected_size=episode.parquet_size,
            expected_sha256=episode.parquet_sha256,
            expected_device=episode.parquet_device,
            expected_inode=episode.parquet_inode,
        )


def test_build_raw_rejects_same_inode_content_change_after_admission(tmp_path: Path):
    batch, raw_arrays = _disk_batch(
        tmp_path,
        lengths=(20,),
        episode_indices=(5,),
    )
    episode = batch.episodes[0]
    states, actions = raw_arrays[5]
    replacement = tmp_path / "replacement.parquet"
    _write_action_state_parquet(
        replacement,
        states,
        actions + np.float32(50.0),
    )
    inode_before = episode.parquet_path.stat().st_ino
    episode.parquet_path.write_bytes(replacement.read_bytes())
    assert episode.parquet_path.stat().st_ino == inode_before

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 5.*parquet.*(size|sha256)",
    ):
        transitions.build_raw_transition_table(
            batch,
            transitions.build_transition_plan(batch),
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )


def test_build_raw_parquet_verification_failures_do_not_leak_fds(tmp_path: Path):
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("requires Linux /proc fd accounting")
    batch, raw_arrays = _disk_batch(
        tmp_path,
        lengths=(20,),
        episode_indices=(5,),
    )
    episode = batch.episodes[0]
    states, actions = raw_arrays[5]
    replacement = tmp_path / "replacement.parquet"
    _write_action_state_parquet(
        replacement,
        states,
        actions + np.float32(25.0),
    )
    episode.parquet_path.write_bytes(replacement.read_bytes())
    plan = transitions.build_transition_plan(batch)
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())
    before = len(os.listdir("/proc/self/fd"))

    for _ in range(20):
        with pytest.raises(ValueError, match="parquet"):
            transitions.build_raw_transition_table(batch, plan, normalizer)

    assert len(os.listdir("/proc/self/fd")) == before


def test_build_raw_transition_table_uses_only_start_state_for_all_twenty_actions(
    tmp_path: Path,
):
    batch, raw_arrays = _disk_batch(
        tmp_path,
        lengths=(20,),
        episode_indices=(5,),
    )
    plan = transitions.build_transition_plan(batch)
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())
    before = transitions.build_raw_transition_table(batch, plan, normalizer)
    states, actions = raw_arrays[5]
    changed_states = states.copy()
    changed_states[1] += np.float32(1234.0)
    _write_action_state_parquet(batch.episodes[0].parquet_path, changed_states, actions)
    batch = _refresh_episode_parquet_identity(batch, 5)

    after = transitions.build_raw_transition_table(batch, plan, normalizer)

    np.testing.assert_array_equal(after.executed_action, before.executed_action)


def test_build_raw_transition_table_nonzero_and_tail_windows_use_exact_raw_slices(
    tmp_path: Path,
):
    batch, raw_arrays = _disk_batch(
        tmp_path,
        lengths=(23,),
        episode_indices=(9,),
    )
    plan = transitions.build_transition_plan(batch)
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())

    actual = transitions.build_raw_transition_table(batch, plan, normalizer)

    assert actual.start_frame_index.tolist() == [0, 2, 3]
    assert actual.terminal[:, 0].tolist() == [False, False, True]
    states, actions = raw_arrays[9]
    for output_index, start in enumerate((0, 2, 3)):
        expected = normalizer.executed_action(states[start], actions[start : start + 20])
        np.testing.assert_array_equal(actual.executed_action[output_index], expected)


def test_build_raw_transition_table_does_not_read_fps_timestamp_or_resample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch, raw_arrays = _disk_batch(
        tmp_path,
        lengths=(21,),
        episode_indices=(4,),
    )
    plan = transitions.build_transition_plan(batch)
    normalizer = transitions.Stage2Normalizer.from_norm_stats(_stats())
    original_read_table = pq.read_table

    def guarded_read_table(source: object, *, columns: list[str]):
        assert columns == ["observation.state", "action"]
        assert isinstance(source, pa.BufferReader)
        return original_read_table(source, columns=columns)

    monkeypatch.setattr(transitions.pq, "read_table", guarded_read_table)

    actual = transitions.build_raw_transition_table(batch, plan, normalizer)

    states, actions = raw_arrays[4]
    assert actual.start_frame_index.tolist() == [0, 1]
    for index, start in enumerate((0, 1)):
        expected = normalizer.executed_action(states[start], actions[start : start + 20])
        np.testing.assert_array_equal(actual.executed_action[index], expected)


def test_build_raw_transition_table_rejects_plan_batch_mismatch_with_context(
    tmp_path: Path,
):
    batch, _ = _disk_batch(tmp_path, lengths=(20,), episode_indices=(2,))
    plan = transitions.build_transition_plan(batch)
    bad_row = dataclasses.replace(
        plan.rows[0],
        batch_id="batch_wrong",
        current_key=dataclasses.replace(plan.rows[0].current_key, batch_id="batch_wrong"),
    )
    bad_plan = dataclasses.replace(plan, rows=(bad_row,))

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 2.*window 0.*batch_wrong",
    ):
        transitions.build_raw_transition_table(
            batch,
            bad_plan,
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )


def test_build_raw_transition_table_rejects_invalid_window_with_context(
    tmp_path: Path,
):
    batch, _ = _disk_batch(tmp_path, lengths=(20,), episode_indices=(2,))
    plan = transitions.build_transition_plan(batch)
    bad_row = dataclasses.replace(
        plan.rows[0],
        start_frame_index=1,
        current_key=dataclasses.replace(plan.rows[0].current_key, frame_index=1),
    )
    bad_plan = dataclasses.replace(plan, rows=(bad_row,))

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 2.*window 1",
    ):
        transitions.build_raw_transition_table(
            batch,
            bad_plan,
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )


def test_build_raw_transition_table_chains_parquet_error_with_batch_episode_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch, _ = _disk_batch(tmp_path, lengths=(20,), episode_indices=(2,))
    plan = transitions.build_transition_plan(batch)

    def broken_read_table(*_args, **_kwargs):
        raise OSError("parquet exploded")

    monkeypatch.setattr(transitions.pq, "read_table", broken_read_table)

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 2.*parquet",
    ) as error:
        transitions.build_raw_transition_table(
            batch,
            plan,
            transitions.Stage2Normalizer.from_norm_stats(_stats()),
        )

    assert isinstance(error.value.__cause__, OSError)
    assert "parquet exploded" in str(error.value.__cause__)


def test_build_raw_transition_table_chains_transform_error_with_window_context(
    tmp_path: Path,
):
    batch, _ = _disk_batch(tmp_path, lengths=(20,), episode_indices=(2,))
    plan = transitions.build_transition_plan(batch)

    class BrokenNormalizer:
        def executed_action(self, *_args):
            raise RuntimeError("transform exploded")

    with pytest.raises(
        ValueError,
        match=rf"batch {batch.batch_id}.*episode 2.*window 0.*transform",
    ) as error:
        transitions.build_raw_transition_table(
            batch,
            plan,
            BrokenNormalizer(),  # type: ignore[arg-type]
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "transform exploded" in str(error.value.__cause__)
