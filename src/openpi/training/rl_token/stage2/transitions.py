from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.rl_token.stage2 import admission
import openpi.transforms as openpi_transforms

ACTION_HORIZON = 20
TRANSITION_STRIDE = 2
STAGE1_CONFIG_NAME = "rl_token_stage1"
STAGE2_CONFIG_NAME = "rl_token_stage2"
REWARD_SOURCE = "tristate"
REWARD_LABEL_VALUES = (-1, 0, 1, 2)
COMPLETION_LABEL = 2
REWARD_AGGREGATION = "sum_20_frames"
REWARD_SCHEMA_VERSION = 1


def reward_metadata(tristate_labels_sha256: str) -> dict[str, object]:
    """Return the exact cache/replay reward provenance contract."""
    if (
        type(tristate_labels_sha256) is not str
        or len(tristate_labels_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tristate_labels_sha256)
    ):
        raise ValueError("tristate_labels_sha256 must be a lowercase SHA-256 string")
    return {
        "stage1_config": STAGE1_CONFIG_NAME,
        "stage2_config": STAGE2_CONFIG_NAME,
        "reward_source": REWARD_SOURCE,
        "reward_label_values": list(REWARD_LABEL_VALUES),
        "completion_label": COMPLETION_LABEL,
        "reward_aggregation": REWARD_AGGREGATION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "tristate_labels_sha256": tristate_labels_sha256,
    }


@dataclasses.dataclass(frozen=True, order=True)
class FeatureKey:
    batch_id: str
    episode_index: int
    frame_index: int


@dataclasses.dataclass(frozen=True)
class TransitionRow:
    batch_id: str
    episode_index: int
    start_frame_index: int
    current_key: FeatureKey
    next_key: FeatureKey | None
    reward: float
    terminal: bool

    @property
    def next_frame_index(self) -> int | None:
        return None if self.next_key is None else self.next_key.frame_index


@dataclasses.dataclass(frozen=True)
class TransitionPlan:
    rows: tuple[TransitionRow, ...]
    feature_keys: tuple[FeatureKey, ...]
    chunk_equivalents: int


@dataclasses.dataclass(frozen=True, eq=False)
class _FrozenNormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def _immutable_stat_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


@dataclasses.dataclass(frozen=True, eq=False)
class Stage2Normalizer:
    norm_stats: Mapping[str, _FrozenNormStats]

    def __post_init__(self) -> None:
        required = {"state", "actions"}
        if not isinstance(self.norm_stats, Mapping) or not required.issubset(self.norm_stats):
            raise ValueError(f"norm stats must contain {sorted(required)}")
        frozen: dict[str, _FrozenNormStats] = {}
        for key in sorted(required):
            stats = self.norm_stats[key]
            _validate_quantile_stats(stats, key)
            frozen[key] = _FrozenNormStats(
                mean=_immutable_stat_array(stats.mean),
                std=_immutable_stat_array(stats.std),
                q01=_immutable_stat_array(stats.q01),
                q99=_immutable_stat_array(stats.q99),
            )
        object.__setattr__(self, "norm_stats", MappingProxyType(frozen))

    @classmethod
    def from_norm_stats(
        cls,
        norm_stats: Mapping[str, openpi_transforms.NormStats],
    ) -> Stage2Normalizer:
        return cls(norm_stats=norm_stats)

    def state(self, state: np.ndarray) -> np.ndarray:
        state_array = _finite_float32_array(
            state,
            shape=(16,),
            name="state",
        )
        try:
            data = openpi_transforms.Normalize(
                {"state": self.norm_stats["state"]},
                use_quantiles=True,
                strict=True,
            )({"state": state_array})
        except Exception as exc:
            raise ValueError("state quantile normalization failed") from exc
        result = np.asarray(data["state"], dtype=np.float32)[..., :16]
        if result.shape != (16,) or not np.isfinite(result).all():
            raise ValueError("normalized state must be finite with shape (16,)")
        return result.copy()

    def executed_action(
        self,
        start_state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        state_array = _finite_float32_array(
            start_state,
            shape=(16,),
            name="start state",
        )
        actions_array = _finite_float32_array(
            actions,
            shape=(ACTION_HORIZON, 16),
            name="actions",
        )
        data = {
            "state": state_array,
            "actions": actions_array,
        }
        try:
            data = openpi_transforms.DeltaActions(openpi_transforms.make_bool_mask(16))(data)
            data = openpi_transforms.Normalize(
                {"actions": self.norm_stats["actions"]},
                use_quantiles=True,
                strict=True,
            )(data)
        except Exception as exc:
            raise ValueError("executed action delta and quantile transform failed") from exc
        result = np.asarray(data["actions"], dtype=np.float32)[..., :16]
        if result.shape != (ACTION_HORIZON, 16) or not np.isfinite(result).all():
            raise ValueError("normalized executed action must be finite with shape (20,16)")
        return result.copy()


@dataclasses.dataclass(frozen=True)
class RawTransitionTable:
    episode_index: np.ndarray
    start_frame_index: np.ndarray
    executed_action: np.ndarray
    intervention: np.ndarray
    reward: np.ndarray
    terminal: np.ndarray


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def _validate_quantile_stats(
    stats: openpi_transforms.NormStats,
    key: str,
) -> None:
    values: dict[str, np.ndarray] = {}
    for field_name in ("mean", "std", "q01", "q99"):
        value = getattr(stats, field_name, None)
        if value is None:
            if field_name in {"q01", "q99"}:
                raise ValueError(f"norm stats {key} must contain quantile q01 and q99")
            raise ValueError(f"norm stats {key} must contain {field_name}")
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"norm stats {key} {field_name} must be a numeric vector of length 16") from exc
        if array.shape != (16,) or not _is_real_numeric_dtype(array.dtype):
            raise ValueError(f"norm stats {key} {field_name} must be a numeric vector of length 16")
        if not np.isfinite(array).all():
            raise ValueError(f"norm stats {key} {field_name} must be finite")
        values[field_name] = array
    if np.any(values["std"] < 0):
        raise ValueError(f"norm stats {key} std must be nonnegative")
    if np.any(values["q99"] < values["q01"]):
        raise ValueError(f"norm stats {key} q99 must be greater than or equal to q01")


def _finite_float32_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    message = f"{name} must be finite with shape {shape}"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != shape or not _is_real_numeric_dtype(array.dtype):
        raise ValueError(message)
    try:
        finite = np.isfinite(array).all()
    except TypeError as exc:
        raise ValueError(message) from exc
    if not finite:
        raise ValueError(message)
    result = np.asarray(array, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError(message)
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def transition_starts(length: int) -> tuple[int, ...]:
    length = _integer(length, "episode length")
    if length < ACTION_HORIZON:
        raise ValueError(f"episode must have at least {ACTION_HORIZON} frames, got {length}")
    tail = length - ACTION_HORIZON
    return tuple(sorted({*range(0, tail + 1, TRANSITION_STRIDE), tail}))


def chunk_reward(labels: np.ndarray, start: int) -> float:
    start = _integer(start, "reward window start")
    if start < 0:
        raise ValueError(f"reward window start must be nonnegative, got {start}")
    try:
        labels_array = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be a one-dimensional integer array") from exc
    if labels_array.ndim != 1 or not np.issubdtype(labels_array.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    window = labels_array[start : start + ACTION_HORIZON]
    if window.shape != (ACTION_HORIZON,):
        raise ValueError(f"incomplete reward window at frame {start}")
    return float(np.sum(window, dtype=np.int64))


def bc_anchor(
    reference: np.ndarray,
    executed_action: np.ndarray,
    intervention: np.ndarray,
) -> np.ndarray:
    try:
        reference_array = np.asarray(reference)
        executed_array = np.asarray(executed_action)
        intervention_array = np.asarray(intervention)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference and executed action must have shape (20,16), and intervention must have shape (20,)"
        ) from exc
    if reference_array.shape != (ACTION_HORIZON, 16) or executed_array.shape != (ACTION_HORIZON, 16):
        raise ValueError("reference and executed action must have shape (20,16)")
    if intervention_array.shape != (ACTION_HORIZON,):
        raise ValueError("intervention must have shape (20,)")
    if intervention_array.dtype != np.dtype(np.bool_):
        raise ValueError("intervention must have bool dtype")
    for value, name in (
        (reference_array, "reference"),
        (executed_array, "executed action"),
    ):
        if not _is_real_numeric_dtype(value.dtype) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    reference_float32 = reference_array.astype(np.float32, copy=False)
    executed_float32 = executed_array.astype(np.float32, copy=False)
    if not np.isfinite(reference_float32).all() or not np.isfinite(executed_float32).all():
        raise ValueError("reference and executed action must remain finite in float32")
    return np.where(
        intervention_array[:, None],
        executed_float32,
        reference_float32,
    ).astype(np.float32, copy=False)


def build_transition_plan(batch: admission.ValidatedBatch) -> TransitionPlan:
    if not batch.episodes:
        raise ValueError(f"batch {batch.batch_id} has no episodes")
    rows: list[TransitionRow] = []
    feature_keys: set[FeatureKey] = set()
    for episode in batch.episodes:
        try:
            starts = transition_starts(episode.length)
        except ValueError as exc:
            raise ValueError(f"batch {batch.batch_id} episode {episode.episode_index}: {exc}") from exc
        for start in starts:
            try:
                terminal = start == episode.length - ACTION_HORIZON
                current = FeatureKey(batch.batch_id, episode.episode_index, start)
                next_key = (
                    None
                    if terminal
                    else FeatureKey(
                        batch.batch_id,
                        episode.episode_index,
                        start + ACTION_HORIZON,
                    )
                )
                row = TransitionRow(
                    batch_id=batch.batch_id,
                    episode_index=episode.episode_index,
                    start_frame_index=start,
                    current_key=current,
                    next_key=next_key,
                    reward=chunk_reward(episode.labels, start),
                    terminal=terminal,
                )
            except ValueError as exc:
                raise ValueError(
                    f"batch {batch.batch_id} episode {episode.episode_index} window {start}: {exc}"
                ) from exc
            feature_keys.add(current)
            if next_key is not None:
                feature_keys.add(next_key)
            rows.append(row)
    return TransitionPlan(
        rows=tuple(rows),
        feature_keys=tuple(sorted(feature_keys)),
        chunk_equivalents=batch.chunk_equivalents,
    )


def _validate_plan_for_batch(
    batch: admission.ValidatedBatch,
    plan: TransitionPlan,
) -> dict[int, admission.ValidatedEpisode]:
    episode_by_id: dict[int, admission.ValidatedEpisode] = {}
    expected_windows: set[tuple[int, int]] = set()
    for episode in batch.episodes:
        episode_index = _integer(episode.episode_index, "episode index")
        if episode_index in episode_by_id:
            raise ValueError(f"batch {batch.batch_id} has duplicate episode {episode_index}")
        episode_by_id[episode_index] = episode
        try:
            starts = transition_starts(episode.length)
        except ValueError as exc:
            raise ValueError(f"batch {batch.batch_id} episode {episode_index}: {exc}") from exc
        expected_windows.update((episode_index, start) for start in starts)

    if not episode_by_id:
        raise ValueError(f"batch {batch.batch_id} has no episodes")
    if plan.chunk_equivalents != batch.chunk_equivalents:
        raise ValueError(
            f"batch {batch.batch_id} plan chunk_equivalents {plan.chunk_equivalents} "
            f"does not match batch {batch.chunk_equivalents}"
        )
    if len(plan.rows) != len(expected_windows):
        raise ValueError(
            f"batch {batch.batch_id} plan row count {len(plan.rows)} does not match expected {len(expected_windows)}"
        )

    seen_windows: set[tuple[int, int]] = set()
    expected_feature_keys: set[FeatureKey] = set()
    max_float32 = float(np.finfo(np.float32).max)
    for row in plan.rows:
        context = f"batch {batch.batch_id} episode {row.episode_index} window {row.start_frame_index}"
        if row.batch_id != batch.batch_id:
            raise ValueError(f"{context}: plan row batch id {row.batch_id!r} does not match")
        try:
            episode_index = _integer(row.episode_index, "episode index")
            start = _integer(row.start_frame_index, "window start")
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
        episode = episode_by_id.get(episode_index)
        if episode is None:
            raise ValueError(f"{context}: episode is not present in batch")
        identity = (episode_index, start)
        if identity not in expected_windows:
            raise ValueError(f"{context}: invalid transition window")
        if identity in seen_windows:
            raise ValueError(f"{context}: duplicate transition window")
        seen_windows.add(identity)

        expected_current = FeatureKey(batch.batch_id, episode_index, start)
        expected_terminal = start == episode.length - ACTION_HORIZON
        expected_next = None if expected_terminal else FeatureKey(batch.batch_id, episode_index, start + ACTION_HORIZON)
        if row.current_key != expected_current:
            raise ValueError(f"{context}: current feature key does not match window identity")
        if not isinstance(row.terminal, bool | np.bool_) or bool(row.terminal) != expected_terminal:
            raise ValueError(f"{context}: terminal flag does not match tail window")
        if row.next_key != expected_next:
            raise ValueError(f"{context}: next feature key does not match transition topology")
        if (
            isinstance(row.reward, bool | np.bool_)
            or not isinstance(row.reward, int | float | np.integer | np.floating)
            or not np.isfinite(row.reward)
            or abs(float(row.reward)) > max_float32
        ):
            raise ValueError(f"{context}: reward must be finite and representable in float32")
        expected_feature_keys.add(expected_current)
        if expected_next is not None:
            expected_feature_keys.add(expected_next)

    if seen_windows != expected_windows:
        missing = sorted(expected_windows - seen_windows)
        raise ValueError(f"batch {batch.batch_id} plan is missing transition windows {missing}")
    if plan.feature_keys != tuple(sorted(expected_feature_keys)):
        raise ValueError(f"batch {batch.batch_id} plan feature keys do not match its rows")
    return episode_by_id


def _read_action_state_episode(
    table: pa.Table,
    *,
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if table.num_rows != expected_length:
        raise ValueError(f"parquet row count {table.num_rows} does not match episode length {expected_length}")
    arrays: list[np.ndarray] = []
    expected_type = pa.list_(pa.float32(), 16)
    for column_name in ("observation.state", "action"):
        try:
            column = table[column_name]
        except KeyError as exc:
            raise ValueError(f"parquet is missing {column_name}") from exc
        if column.type != expected_type:
            raise ValueError(f"parquet {column_name} must be fixed_size_list float32 width 16")
        combined = column.combine_chunks()
        if combined.null_count or combined.values.null_count:
            raise ValueError(f"parquet {column_name} must not contain nulls")
        array = np.asarray(combined.to_pylist(), dtype=np.float32)
        if array.shape != (expected_length, 16):
            raise ValueError(f"parquet {column_name} must have shape ({expected_length},16)")
        if not np.isfinite(array).all():
            raise ValueError(f"parquet {column_name} must be finite")
        arrays.append(array)
    return arrays[0], arrays[1]


def build_raw_transition_table(
    batch: admission.ValidatedBatch,
    plan: TransitionPlan,
    normalizer: Stage2Normalizer,
) -> RawTransitionTable:
    episode_by_id = _validate_plan_for_batch(batch, plan)
    needed_episodes = {row.episode_index for row in plan.rows}
    episode_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    intervention_by_episode: dict[int, np.ndarray] = {}
    for episode_index, episode in episode_by_id.items():
        if episode_index not in needed_episodes:
            continue
        try:
            parquet_bytes, _ = admission._read_verified_regular_file(  # noqa: SLF001
                episode.parquet_path,
                expected_size=episode.parquet_size,
                expected_sha256=episode.parquet_sha256,
                expected_device=episode.parquet_device,
                expected_inode=episode.parquet_inode,
            )
            table = pq.read_table(
                pa.BufferReader(parquet_bytes),
                columns=["observation.state", "action"],
            )
            episode_arrays[episode_index] = _read_action_state_episode(
                table,
                expected_length=episode.length,
            )
        except Exception as exc:
            raise ValueError(
                f"batch {batch.batch_id} episode {episode_index}: failed to read action/state parquet: {exc}"
            ) from exc
        intervention = np.asarray(episode.intervention)
        if intervention.shape != (episode.length,) or intervention.dtype != np.dtype(np.bool_):
            error = ValueError(f"intervention must have bool dtype and shape ({episode.length},)")
            raise ValueError(f"batch {batch.batch_id} episode {episode_index}: invalid intervention") from error
        intervention_by_episode[episode_index] = intervention.copy()

    episode_ids: list[int] = []
    starts: list[int] = []
    actions_out: list[np.ndarray] = []
    intervention_out: list[np.ndarray] = []
    rewards: list[list[float]] = []
    terminals: list[list[bool]] = []
    for row in plan.rows:
        episode_index = int(row.episode_index)
        start = int(row.start_frame_index)
        context = f"batch {batch.batch_id} episode {episode_index} window {start}"
        states, actions = episode_arrays[episode_index]
        try:
            transformed = normalizer.executed_action(
                states[start],
                actions[start : start + ACTION_HORIZON],
            )
        except Exception as exc:
            raise ValueError(f"{context}: executed action transform failed") from exc
        intervention = intervention_by_episode[episode_index][start : start + ACTION_HORIZON].copy()
        if intervention.shape != (ACTION_HORIZON,):
            error = ValueError("incomplete intervention window")
            raise ValueError(f"{context}: invalid intervention window") from error
        episode_ids.append(episode_index)
        starts.append(start)
        actions_out.append(transformed)
        intervention_out.append(intervention)
        rewards.append([float(row.reward)])
        terminals.append([bool(row.terminal)])

    result = RawTransitionTable(
        episode_index=np.asarray(episode_ids, dtype=np.int32),
        start_frame_index=np.asarray(starts, dtype=np.int32),
        executed_action=np.stack(actions_out, axis=0).astype(np.float32, copy=False),
        intervention=np.stack(intervention_out, axis=0).astype(np.bool_, copy=False),
        reward=np.asarray(rewards, dtype=np.float32),
        terminal=np.asarray(terminals, dtype=np.bool_),
    )
    if not np.isfinite(result.executed_action).all() or not np.isfinite(result.reward).all():
        raise ValueError(f"batch {batch.batch_id}: raw transition table must be finite")
    return result
