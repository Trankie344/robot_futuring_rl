from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import errno
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from datasets import config as datasets_config
from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openpi.models import model as model_api
from openpi.models.rl_token import config as pi0_config
from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import cache
from openpi.training.rl_token.stage2 import feature_extractor
from openpi.training.rl_token.stage2 import feature_identity
from openpi.training.rl_token.stage2 import transitions
from openpi.training.rl_token.stage2.test.conftest import build_ready_batch
from openpi.shared import nnx_utils
from openpi.training.rl_token import config as training_config
import openpi.transforms as transforms


class _FakeFrameDataset:
    def __init__(self, rows: list[Mapping[str, Any]], *, reported_length: int | None = None):
        self.rows = rows
        self.reported_length = len(rows) if reported_length is None else reported_length
        self.requested_indices: list[int] = []

    def __len__(self) -> int:
        return self.reported_length

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        self.requested_indices.append(index)
        return self.rows[index]


class _ExplodingFrameDataset:
    def __init__(self, length: int, error: Exception):
        self.length = length
        self.error = error

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> Mapping[str, Any]:
        raise self.error


@dataclasses.dataclass(frozen=True)
class _TraceTransform:
    name: str

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "trace": (*data.get("trace", ()), self.name)}


@dataclasses.dataclass(frozen=True)
class _StaticDataFactory:
    source: training_config.DataConfig
    calls: list[tuple[Path, object]]

    def create(self, assets_dirs: Path, model: object) -> training_config.DataConfig:
        self.calls.append((assets_dirs, model))
        return self.source


class _FakeTokenizer:
    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        assert state is not None
        checksum = int(np.rint(np.asarray(state, dtype=np.float64).sum() * 100))
        tokens = np.asarray([len(prompt), checksum, 17, 0], dtype=np.int32)
        mask = np.asarray([True, True, True, False], dtype=np.bool_)
        return tokens, mask


class _OwnedExtractionDataset(feature_extractor.Stage2ObservationDataset):
    """Small synchronous dataset that exercises extraction ownership without disk IO."""

    def __init__(
        self,
        keys: tuple[transitions.FeatureKey, ...],
        observations: tuple[dict[str, Any], ...],
        *,
        returned_keys: tuple[transitions.FeatureKey, ...] | None = None,
        reported_length: int | None = None,
        mutate_snapshot_on_get: bool = False,
    ):
        assert len(keys) == len(observations)
        self._frozen_keys = keys
        self._observations = observations
        self._returned_keys = keys if returned_keys is None else returned_keys
        self._reported_length = len(keys) if reported_length is None else reported_length
        self._mutate_snapshot_on_get = mutate_snapshot_on_get
        self._snapshot_changed = False
        self.verify_calls = 0
        self.close_calls = 0
        self._test_closed = False

    @property
    def feature_keys(self) -> tuple[transitions.FeatureKey, ...]:
        return self._frozen_keys

    @property
    def batch_id(self) -> str:
        return self._frozen_keys[0].batch_id if self._frozen_keys else "batch_000001"

    @property
    def closed(self) -> bool:
        return self._test_closed

    def __len__(self) -> int:
        if self._test_closed:
            raise feature_extractor.BatchSnapshotError("test extraction dataset is closed")
        return self._reported_length

    def __getitem__(self, index: int):
        if self._test_closed:
            raise feature_extractor.BatchSnapshotError("test extraction dataset is closed")
        if self._mutate_snapshot_on_get:
            self._snapshot_changed = True
        return self._returned_keys[index], self._observations[index]

    def verify_unchanged(self) -> None:
        if self._test_closed:
            raise feature_extractor.BatchSnapshotError("test extraction dataset is closed")
        self.verify_calls += 1
        if self._snapshot_changed:
            raise feature_extractor.BatchSnapshotError("test batch snapshot changed during lazy iteration")

    def close(self) -> None:
        self.close_calls += 1
        self._test_closed = True


class _FakeExtractionModel(nnx.Module):
    def __init__(
        self,
        *,
        action_horizon: int = 50,
        action_dim: int = 32,
        fault: str | None = None,
        mutate_parameter: bool = False,
        sample_error: bool = False,
        mutate_non_parameter: bool = False,
    ):
        self.guard_parameter = nnx.Param(jnp.asarray([1.0], dtype=jnp.float32))
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.fault = fault
        self.mutate_parameter = mutate_parameter
        self.sample_error = sample_error
        self.mutate_non_parameter = mutate_non_parameter
        self.prefix_calls = 0
        self.non_parameter_counter = 0

    def sample_actions_and_rl_token(
        self,
        _rng: jax.Array,
        observation: model_api.Observation,
        *,
        num_steps: int,
        noise: jax.Array,
    ):
        assert num_steps > 0
        self.prefix_calls += 1
        if self.mutate_parameter:
            self.guard_parameter.value = self.guard_parameter.value + 1
        if self.mutate_non_parameter:
            self.non_parameter_counter += 1
        if self.sample_error:
            raise RuntimeError("fake shared-prefix sample failed")

        actions: Any = noise
        flattened = jnp.reshape(noise, (noise.shape[0], -1))
        repeats = math.ceil(2048 / flattened.shape[1])
        z_rl: Any = jnp.tile(flattened, (1, repeats))[:, :2048]
        if self.fault == "action_batch":
            actions = actions[:-1]
        elif self.fault == "action_horizon":
            actions = actions[:, :-1]
        elif self.fault == "action_dim":
            actions = actions[:, :, :-1]
        elif self.fault == "action_dtype":
            actions = actions.astype(jnp.bfloat16)
        elif self.fault == "action_nan":
            actions = actions.at[0, 0, 0].set(jnp.nan)
        elif self.fault == "action_inf":
            actions = actions.at[0, 0, 0].set(jnp.inf)
        elif self.fault == "z_batch":
            z_rl = z_rl[:-1]
        elif self.fault == "z_width":
            z_rl = z_rl[:, :-1]
        elif self.fault == "z_dtype":
            z_rl = z_rl.astype(jnp.int32)
        elif self.fault == "z_bfloat16":
            z_rl = z_rl.astype(jnp.bfloat16)
        elif self.fault == "z_nan":
            z_rl = z_rl.at[0, 0].set(jnp.nan)
        elif self.fault == "z_inf":
            z_rl = z_rl.at[0, 0].set(jnp.inf)
        elif self.fault == "z_overflow":
            z_rl = np.full((noise.shape[0], 2048), 1e300, dtype=np.float64)
        return actions, z_rl


@pytest.fixture
def direct_module_jit(monkeypatch: pytest.MonkeyPatch) -> list[Callable[..., Any]]:
    constructions: list[Callable[..., Any]] = []

    def identity_module_jit(method: Callable[..., Any], *_args: Any, **_kwargs: Any):
        constructions.append(method)
        return method

    monkeypatch.setattr(feature_extractor.nnx_utils, "module_jit", identity_module_jit)
    return constructions


def _extraction_keys(count: int = 5, *, batch_id: str = "batch_000001") -> tuple[transitions.FeatureKey, ...]:
    return tuple(transitions.FeatureKey(batch_id, index // 3, index % 3) for index in range(count))


def _extraction_observation(
    row: int,
    *,
    state_width: int = 20,
    state_dtype: Any = np.float32,
    state_fault: str | None = None,
) -> dict[str, Any]:
    state = np.arange(state_width, dtype=state_dtype) + row * 100
    if state_fault == "nan":
        state[0] = np.nan
    elif state_fault == "inf":
        state[0] = np.inf
    elif state_fault == "overflow":
        state = np.full(state_width, 1e300, dtype=np.float64)
    return {
        "image": {"dummy": np.full((2, 2, 3), row, dtype=np.float32)},
        "image_mask": {"dummy": np.ones((), dtype=np.bool_)},
        "state": state,
    }


def _owned_extraction_dataset(
    keys: tuple[transitions.FeatureKey, ...],
    *,
    returned_keys: tuple[transitions.FeatureKey, ...] | None = None,
    reported_length: int | None = None,
    state_width: int = 20,
    state_fault: str | None = None,
    mutate_snapshot_on_get: bool = False,
) -> _OwnedExtractionDataset:
    observations = tuple(
        _extraction_observation(
            row,
            state_width=state_width,
            state_fault=state_fault,
        )
        for row in range(len(keys))
    )
    return _OwnedExtractionDataset(
        keys,
        observations,
        returned_keys=returned_keys,
        reported_length=reported_length,
        mutate_snapshot_on_get=mutate_snapshot_on_get,
    )


def _parameter_sha256(model: nnx.Module) -> str:
    return feature_identity.parameter_tree_sha256(nnx.state(model, nnx.Param))


def _extract(
    dataset: feature_extractor.Stage2ObservationDataset,
    model: nnx.Module,
    keys: tuple[transitions.FeatureKey, ...],
    **overrides: Any,
) -> cache.FeatureTable:
    arguments = {
        "model": model,
        "dataset": dataset,
        "feature_keys": keys,
        "feature_id": "frozen-feature-v1",
        "expected_parameter_sha256": _parameter_sha256(model),
        "micro_batch_size": 4,
        "num_workers": 0,
        "sampler_num_steps": 10,
    }
    arguments.update(overrides)
    return feature_extractor.extract_features(**arguments)


def _validated_batch(ready_batch: Path) -> admission.ValidatedBatch:
    return admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def _rows_for_batch(batch: admission.ValidatedBatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any] | None] = [None] * batch.total_frames
    for episode in batch.episodes:
        for frame_index in range(episode.length):
            global_index = episode.dataset_from_index + frame_index
            rows[global_index] = {
                "observation.state": torch.full((16,), float(global_index), dtype=torch.float32),
                "index": torch.tensor(global_index, dtype=torch.int64),
                "episode_index": torch.tensor(episode.episode_index, dtype=torch.int64),
                "frame_index": torch.tensor(frame_index, dtype=torch.int64),
            }
    assert all(row is not None for row in rows)
    return [row for row in rows if row is not None]


def _install_fake_dataset(
    monkeypatch: pytest.MonkeyPatch,
    batch: admission.ValidatedBatch,
    *,
    rows: list[Mapping[str, Any]] | None = None,
    reported_length: int | None = None,
) -> _FakeFrameDataset:
    fake = _FakeFrameDataset(
        _rows_for_batch(batch) if rows is None else rows,
        reported_length=reported_length,
    )
    monkeypatch.setattr(
        feature_extractor,
        "_construct_local_only_lerobot_dataset",
        lambda *_args, **_kwargs: fake,
    )
    return fake


def _capture_lerobot_constructor(
    monkeypatch: pytest.MonkeyPatch,
    batch: admission.ValidatedBatch,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fake = _FakeFrameDataset(_rows_for_batch(batch))

    def constructor(*args: Any, **kwargs: Any) -> _FakeFrameDataset:
        calls.append((args, kwargs))
        return fake

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", constructor)
    return calls


def _replace_batch_root_with_copy(
    batch: admission.ValidatedBatch,
    *,
    symlink: bool,
) -> None:
    root = batch.root
    accepted = root.with_name(f"{root.name}-accepted")
    foreign = root.with_name(f"{root.name}-foreign")
    shutil.copytree(root, foreign)
    root.rename(accepted)
    if symlink:
        root.symlink_to(foreign, target_is_directory=True)
    else:
        foreign.rename(root)


def _norm_stats(*, q01: float = -2.0, q99: float = 6.0) -> dict[str, transforms.NormStats]:
    def stats() -> transforms.NormStats:
        return transforms.NormStats(
            mean=np.zeros(16, dtype=np.float32),
            std=np.ones(16, dtype=np.float32),
            q01=np.full(16, q01, dtype=np.float32),
            q99=np.full(16, q99, dtype=np.float32),
        )

    return {"state": stats(), "actions": stats()}


def _make_train_config(
    source: training_config.DataConfig,
    tmp_path: Path,
) -> tuple[SimpleNamespace, _StaticDataFactory]:
    factory = _StaticDataFactory(source, [])
    model = object()
    return SimpleNamespace(data=factory, assets_dirs=tmp_path / "assets", model=model), factory


def _patch_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tasks: dict[int, str],
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _FakeMetadata:
        def __init__(self, *args: Any, **kwargs: Any):
            calls.append((args, kwargs))
            self.tasks = tasks

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_metadata", _FakeMetadata)
    return calls


def _key_with_boolean_frame(batch: admission.ValidatedBatch) -> transitions.FeatureKey:
    boolean_frame = True
    return transitions.FeatureKey(batch.batch_id, 0, boolean_frame)


def _assert_descriptor_closed(descriptor: int) -> None:
    with pytest.raises(OSError, match="Bad file descriptor") as exc_info:
        os.fstat(descriptor)
    assert exc_info.value.errno == errno.EBADF


def _descriptor_from_proc_root(root: Path) -> int:
    root = Path(root)
    assert root.parent == Path("/proc/self/fd")
    return int(root.name)


def test_create_lerobot_frame_dataset_uses_canonical_local_root_without_action_window(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    expected = object()

    def fake_dataset(*args: Any, **kwargs: Any) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", fake_dataset)

    batch = _validated_batch(ready_batch)
    result = feature_extractor.create_lerobot_frame_dataset(batch)
    proc_root = Path(captured["kwargs"]["root"])
    descriptor = _descriptor_from_proc_root(proc_root)
    try:
        assert result is not expected
        assert captured["args"][0] == ready_batch.name
        assert proc_root.parent == Path("/proc/self/fd")
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        assert "delta_timestamps" not in captured["kwargs"]
        assert captured["kwargs"]["tolerance_s"] == 0.05
    finally:
        result.close()
    _assert_descriptor_closed(descriptor)


def test_create_lerobot_frame_dataset_rejects_missing_root_without_ctor_or_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    missing = tmp_path / "missing"
    calls: list[object] = []
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset,
        "LeRobotDataset",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(TypeError, match="ValidatedBatch"):
        feature_extractor.create_lerobot_frame_dataset(missing)

    assert calls == []
    assert not missing.exists()


def test_create_lerobot_frame_dataset_rejects_noncanonical_root_before_ctor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset,
        "LeRobotDataset",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    noncanonical = ready_batch / ".." / ready_batch.name

    with pytest.raises(TypeError, match="ValidatedBatch"):
        feature_extractor.create_lerobot_frame_dataset(noncanonical)

    assert calls == []


def test_create_lerobot_frame_dataset_rejects_final_symlink_before_ctor(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    link = tmp_path / "batch-link"
    link.symlink_to(ready_batch, target_is_directory=True)
    calls: list[object] = []
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset,
        "LeRobotDataset",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(TypeError, match="ValidatedBatch"):
        feature_extractor.create_lerobot_frame_dataset(link)

    assert calls == []


def test_create_lerobot_frame_dataset_rejects_ancestor_symlink_before_ctor(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(ready_batch.parent, target_is_directory=True)
    through_link = linked_parent / ready_batch.name
    calls: list[object] = []
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset,
        "LeRobotDataset",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(TypeError, match="ValidatedBatch"):
        feature_extractor.create_lerobot_frame_dataset(through_link)

    assert calls == []


@pytest.mark.parametrize("symlink", [True, False])
def test_observation_dataset_rejects_replaced_batch_root_before_lerobot_ctor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    symlink: bool,
):
    batch = _validated_batch(ready_batch)
    calls = _capture_lerobot_constructor(monkeypatch, batch)
    _replace_batch_root_with_copy(batch, symlink=symlink)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="snapshot|symlink|inode"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert calls == []


def test_observation_dataset_rejects_replaced_ancestor_before_lerobot_ctor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = build_ready_batch(tmp_path / "ancestor" / "batch_000001_ancestor")
    batch = _validated_batch(root)
    calls = _capture_lerobot_constructor(monkeypatch, batch)
    ancestor = root.parent
    accepted_ancestor = ancestor.with_name("accepted-ancestor")
    ancestor.rename(accepted_ancestor)
    ancestor.symlink_to(accepted_ancestor, target_is_directory=True)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="symlink|directory"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert calls == []


@pytest.mark.parametrize(
    "relative_path",
    [
        "migration_manifest.json",
        "meta/tristate_labels.json",
        "meta/info.json",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.top/episode_000000.mp4",
    ],
)
def test_observation_dataset_rejects_snapshot_content_changes_before_lerobot_ctor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
):
    batch = _validated_batch(ready_batch)
    calls = _capture_lerobot_constructor(monkeypatch, batch)
    target = batch.root / relative_path
    payload = bytearray(target.read_bytes())
    assert payload
    payload[0] ^= 0x01
    target.write_bytes(payload)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="snapshot|sha256|changed"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert calls == []


def test_observation_dataset_rejects_same_content_new_parquet_inode_before_lerobot_ctor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    calls = _capture_lerobot_constructor(monkeypatch, batch)
    target = batch.episodes[0].parquet_path
    replacement = target.with_name("replacement.parquet")
    shutil.copy2(target, replacement)
    os.replace(replacement, target)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="inode|snapshot|changed"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert calls == []


def test_verify_validated_batch_snapshot_detects_root_replacement_during_scan(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    root = batch.root
    accepted = root.with_name(f"{root.name}-accepted")
    foreign = root.with_name(f"{root.name}-foreign")
    shutil.copytree(root, foreign)
    replaced = False

    def replace_root(_path: Path, _descriptor: int) -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        root.rename(accepted)
        foreign.rename(root)

    monkeypatch.setattr(feature_extractor, "_after_snapshot_file_open", replace_root)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="root.*changed|snapshot"):
        feature_extractor.verify_validated_batch_snapshot(batch)

    assert replaced is True


def test_verify_validated_batch_snapshot_detects_file_replacement_during_hash(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    target = batch.root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    replacement = batch.root.parent / "replacement.mp4"
    shutil.copy2(target, replacement)
    replaced = False

    def replace_video(path: Path, _descriptor: int) -> None:
        nonlocal replaced
        if path == target and not replaced:
            replaced = True
            os.replace(replacement, target)

    monkeypatch.setattr(feature_extractor, "_after_snapshot_file_open", replace_video)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="changed|binding|snapshot"):
        feature_extractor.verify_validated_batch_snapshot(batch)

    assert replaced is True


def test_verify_validated_batch_snapshot_detects_same_size_write_during_hash(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    target = batch.root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    modified = False

    def modify_video(path: Path, _descriptor: int) -> None:
        nonlocal modified
        if path != target or modified:
            return
        modified = True
        with path.open("r+b") as stream:
            first = stream.read(1)
            assert first
            stream.seek(0)
            stream.write(bytes([first[0] ^ 0x01]))
            stream.flush()
            os.fsync(stream.fileno())

    monkeypatch.setattr(feature_extractor, "_after_snapshot_file_open", modify_video)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="changed|sha256|snapshot"):
        feature_extractor.verify_validated_batch_snapshot(batch)

    assert modified is True


def test_observation_dataset_detects_constructor_aba_file_swap(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    target = batch.root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    original_payload = target.read_bytes()
    bad_payload = bytearray(original_payload)
    bad_payload[0] ^= 0x01
    bad = batch.root.parent / "bad.mp4"
    bad.write_bytes(bad_payload)
    held = batch.root.parent / "held.mp4"
    observed: list[bytes] = []
    fake = _FakeFrameDataset(_rows_for_batch(batch))

    def constructor(*_args: Any, **_kwargs: Any) -> _FakeFrameDataset:
        os.replace(target, held)
        os.replace(bad, target)
        observed.append(target.read_bytes())
        os.replace(target, bad)
        os.replace(held, target)
        return fake

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", constructor)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="changed.*construct|witness|snapshot"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert observed == [bytes(bad_payload)]
    assert target.read_bytes() == original_payload


def test_observation_dataset_reads_pinned_original_during_temporary_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = build_ready_batch(tmp_path / "ancestor" / "batch_000001_test")
    batch = _validated_batch(root)
    ancestor = root.parent
    held_ancestor = ancestor.with_name("ancestor-held")
    foreign_ancestor = ancestor.with_name("ancestor-foreign")
    shutil.copytree(ancestor, foreign_ancestor)
    foreign_root = foreign_ancestor / root.name
    relative_video = Path("videos/chunk-000/observation.images.top/episode_000000.mp4")
    original_payload = (root / relative_video).read_bytes()
    bad_payload = bytearray((foreign_root / relative_video).read_bytes())
    bad_payload[0] ^= 0x01
    (foreign_root / relative_video).write_bytes(bad_payload)
    observed: list[bytes] = []
    fake = _FakeFrameDataset(_rows_for_batch(batch))

    def constructor(*_args: Any, **kwargs: Any) -> _FakeFrameDataset:
        ancestor.rename(held_ancestor)
        foreign_ancestor.rename(ancestor)
        observed.append((Path(kwargs["root"]) / relative_video).read_bytes())
        ancestor.rename(foreign_ancestor)
        held_ancestor.rename(ancestor)
        return fake

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", constructor)

    dataset = feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)
    try:
        assert observed == [original_payload]
    finally:
        dataset.close()


def test_observation_dataset_reads_pinned_original_during_temporary_root_swap(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    root = batch.root
    held_root = root.with_name(f"{root.name}-held")
    foreign = root.with_name(f"{root.name}-foreign")
    shutil.copytree(root, foreign)
    relative_video = Path("videos/chunk-000/observation.images.top/episode_000000.mp4")
    original_payload = (root / relative_video).read_bytes()
    bad_payload = bytearray((foreign / relative_video).read_bytes())
    bad_payload[0] ^= 0x01
    (foreign / relative_video).write_bytes(bad_payload)
    observed: list[bytes] = []
    fake = _FakeFrameDataset(_rows_for_batch(batch))

    def constructor(*_args: Any, **kwargs: Any) -> _FakeFrameDataset:
        root.rename(held_root)
        foreign.rename(root)
        observed.append((Path(kwargs["root"]) / relative_video).read_bytes())
        root.rename(foreign)
        held_root.rename(root)
        return fake

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", constructor)

    dataset = feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)
    try:
        assert observed == [original_payload]
    finally:
        dataset.close()


def test_observation_dataset_ignores_unrelated_parent_sibling_change(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    sibling = batch.root.parent / "unrelated-sibling"
    fake = _FakeFrameDataset(_rows_for_batch(batch))
    proc_roots: list[Path] = []

    def constructor(*_args: Any, **kwargs: Any) -> _FakeFrameDataset:
        proc_roots.append(Path(kwargs["root"]))
        sibling.write_text("unrelated\n", encoding="utf-8")
        return fake

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", constructor)

    dataset = feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)
    try:
        assert len(dataset) == 0
        assert proc_roots[0].parent == Path("/proc/self/fd")
    finally:
        dataset.close()


def test_observation_dataset_close_context_pickle_and_verify_contract(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    _capture_lerobot_constructor(monkeypatch, batch)
    key = transitions.FeatureKey(batch.batch_id, 0, 0)

    with feature_extractor.Stage2ObservationDataset(batch, [key], lambda value: value) as dataset:
        dataset.verify_unchanged()
        with pytest.raises(TypeError, match="pickle"):
            pickle.dumps(dataset)
        proc_root = dataset.pinned_root
        descriptor = _descriptor_from_proc_root(proc_root)
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC

    dataset.close()
    _assert_descriptor_closed(descriptor)
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        len(dataset)
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        dataset[0]
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        dataset.verify_unchanged()


def test_observation_dataset_finalizer_closes_descriptor_as_fallback(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    _capture_lerobot_constructor(monkeypatch, batch)
    dataset = feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)
    descriptor = _descriptor_from_proc_root(dataset.pinned_root)

    del dataset
    gc.collect()

    _assert_descriptor_closed(descriptor)


def test_observation_dataset_verify_unchanged_detects_lazy_iteration_mutation(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    _capture_lerobot_constructor(monkeypatch, batch)
    dataset = feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)
    target = batch.root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 0x01
    target.write_bytes(payload)

    try:
        with pytest.raises(feature_extractor.BatchSnapshotError, match="changed|sha256"):
            dataset.verify_unchanged()
    finally:
        dataset.close()


def test_observation_dataset_constructor_failure_closes_pinned_descriptor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    descriptors: list[int] = []

    def fail_constructor(*_args: Any, **kwargs: Any) -> object:
        descriptors.append(_descriptor_from_proc_root(Path(kwargs["root"])))
        raise ValueError("injected LeRobot construction failure")

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", fail_constructor)

    with pytest.raises(RuntimeError, match="construct") as exc_info:
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert descriptors
    _assert_descriptor_closed(descriptors[0])


def test_observation_dataset_binding_failure_closes_pinned_descriptor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    root = batch.root
    accepted = root.with_name(f"{root.name}-accepted")
    foreign = root.with_name(f"{root.name}-foreign")
    shutil.copytree(root, foreign)
    descriptors: list[int] = []
    fake = _FakeFrameDataset(_rows_for_batch(batch))

    def replace_binding(*_args: Any, **kwargs: Any) -> _FakeFrameDataset:
        descriptors.append(_descriptor_from_proc_root(Path(kwargs["root"])))
        root.rename(accepted)
        foreign.rename(root)
        return fake

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", replace_binding)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="root pathname changed"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)

    assert descriptors
    _assert_descriptor_closed(descriptors[0])


def test_pinned_descriptor_allocator_reuses_lowest_available_cloexec_fd(tmp_path: Path):
    source = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    first: int | None = None
    second: int | None = None
    try:
        first = feature_extractor._duplicate_pinned_cloexec(source)  # noqa: SLF001
        assert fcntl.fcntl(first, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        os.close(first)
        first_closed = first
        first = None
        second = feature_extractor._duplicate_pinned_cloexec(source)  # noqa: SLF001
        assert second == first_closed
        assert fcntl.fcntl(second, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    finally:
        if first is not None:
            os.close(first)
        if second is not None:
            os.close(second)
        os.close(source)


def test_pinned_descriptor_allocator_works_with_low_nofile_limit(tmp_path: Path):
    script = """
import fcntl
import os
import resource
import sys

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
limit = 64 if hard == resource.RLIM_INFINITY else min(64, hard)
assert limit > 3
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
from openpi.training.rl_token.stage2 import feature_extractor

source = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
try:
    duplicate = feature_extractor._duplicate_pinned_cloexec(source)
    try:
        assert 3 <= duplicate < limit
        assert fcntl.fcntl(duplicate, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    finally:
        os.close(duplicate)
finally:
    os.close(source)
"""
    subprocess.run(
        [sys.executable, "-c", script, os.fspath(tmp_path)],
        check=True,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def test_observation_dataset_reuses_bounded_cloexec_proc_fd_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first_root = build_ready_batch(tmp_path / "batch_000001_first")
    second_root = build_ready_batch(tmp_path / "batch_000002_second")
    first_batch = _validated_batch(first_root)
    second_batch = _validated_batch(second_root)
    proc_descriptors: list[int] = []

    def constructor(*_args: Any, **kwargs: Any) -> _FakeFrameDataset:
        descriptor = _descriptor_from_proc_root(Path(kwargs["root"]))
        proc_descriptors.append(descriptor)
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        batch = first_batch if len(proc_descriptors) == 1 else second_batch
        return _FakeFrameDataset(_rows_for_batch(batch))

    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_dataset", constructor)
    first = feature_extractor.Stage2ObservationDataset(first_batch, [], lambda value: value)
    first.close()
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        len(first)
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        first.verify_unchanged()
    second = feature_extractor.Stage2ObservationDataset(second_batch, [], lambda value: value)
    try:
        assert proc_descriptors[1] == proc_descriptors[0]
    finally:
        second.close()


def test_create_lerobot_frame_dataset_rejects_incomplete_forged_manifest_before_ctor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    manifest_path = batch.root / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [record for record in manifest["files"] if record["target_path"] != "meta/tasks.jsonl"]
    payload = (json.dumps(manifest) + "\n").encode()
    manifest_path.write_bytes(payload)
    forged = dataclasses.replace(batch, manifest_sha256=hashlib.sha256(payload).hexdigest())
    calls: list[object] = []
    monkeypatch.setattr(
        feature_extractor,
        "_construct_local_only_lerobot_dataset",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(feature_extractor.BatchSnapshotError, match="required|complete.*trust root"):
        feature_extractor.create_lerobot_frame_dataset(forged)

    assert calls == []


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "/absolute",
        "a/",
        "a//b",
        "a/./b",
        "a/../b",
        "a\\b",
        "a\x00b",
    ],
)
def test_validate_relative_path_rejects_noncanonical_or_non_posix_spelling(value: str):
    with pytest.raises(feature_extractor.BatchSnapshotError, match="relative POSIX|canonical"):
        feature_extractor._validate_relative_path(value, "test path")  # noqa: SLF001


def test_stream_snapshot_manifest_enforces_capture_limit_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "snapshot-root"
    root.mkdir()
    target = root / "manifest.json"
    target.write_bytes(b"12345678")
    _, root_descriptor, root_metadata = feature_extractor._open_canonical_root(root)  # noqa: SLF001
    directories = {".": feature_extractor._root_stat_witness(root_metadata)}  # noqa: SLF001
    opened_descriptors: list[int] = []

    def grow_after_open(path: Path, descriptor: int) -> None:
        assert path == target
        opened_descriptors.append(descriptor)
        with path.open("ab") as stream:
            stream.write(b"abcdefgh")

    monkeypatch.setattr(feature_extractor, "_MAX_MANIFEST_BYTES", 8)
    monkeypatch.setattr(feature_extractor, "_after_snapshot_file_open", grow_after_open)
    try:
        with pytest.raises(feature_extractor.BatchSnapshotError, match="exceeded capture limit"):
            feature_extractor._stream_snapshot_file(  # noqa: SLF001
                root_descriptor,
                root,
                "manifest.json",
                directories,
                expected_size=None,
                expected_sha256=None,
                capture=True,
            )
    finally:
        os.close(root_descriptor)

    assert opened_descriptors
    _assert_descriptor_closed(opened_descriptors[0])


def test_build_stage2_input_transform_detects_tasks_aba_and_closes_descriptor(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    target = batch.root / "meta/tasks.jsonl"
    held = tmp_path / "held-tasks.jsonl"
    bad = tmp_path / "bad-tasks.jsonl"
    bad.write_text('{"task_index": 0, "task": "foreign"}\n', encoding="utf-8")
    descriptors: list[int] = []

    class _AbaMetadata:
        def __init__(self, *_args: Any, **kwargs: Any):
            root = Path(kwargs["root"])
            descriptors.append(_descriptor_from_proc_root(root))
            os.replace(target, held)
            os.replace(bad, target)
            self.tasks = {0: json.loads((root / "meta/tasks.jsonl").read_text())["task"]}
            os.replace(target, bad)
            os.replace(held, target)

    source = training_config.DataConfig(repo_id="remote/source", prompt_from_task=True)
    train_config, _ = _make_train_config(source, tmp_path)
    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_metadata", _AbaMetadata)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="changed|snapshot"):
        feature_extractor.build_stage2_input_transform(train_config, batch, {})

    assert descriptors
    _assert_descriptor_closed(descriptors[0])


def test_build_stage2_input_transform_never_calls_hf_fallback_when_tasks_disappear(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    target = batch.root / "meta/tasks.jsonl"
    held = tmp_path / "held-tasks.jsonl"
    removed = False
    fallback_calls: list[object] = []
    original_load_metadata = feature_extractor.lerobot_dataset.LeRobotDatasetMetadata.load_metadata

    def remove_tasks_then_load(metadata: Any) -> None:
        nonlocal removed
        if not removed:
            removed = True
            os.replace(target, held)
        original_load_metadata(metadata)

    def forbidden_pull(metadata: Any, *args: Any, **kwargs: Any) -> None:
        fallback_calls.append((metadata, args, kwargs))
        raise AssertionError("Hugging Face metadata fallback must be unreachable")

    def forbidden_safe_version(*args: Any, **kwargs: Any) -> None:
        fallback_calls.append(("get_safe_version", args, kwargs))
        raise AssertionError("Hugging Face version fallback must be unreachable")

    source = training_config.DataConfig(repo_id="remote/source", prompt_from_task=True)
    train_config, _ = _make_train_config(source, tmp_path)
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset.LeRobotDatasetMetadata,
        "load_metadata",
        remove_tasks_then_load,
    )
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset.LeRobotDatasetMetadata,
        "pull_from_repo",
        forbidden_pull,
    )
    monkeypatch.setattr(feature_extractor.lerobot_dataset, "get_safe_version", forbidden_safe_version)

    try:
        with pytest.raises(feature_extractor.BatchSnapshotError, match="local.*metadata|missing"):
            feature_extractor.build_stage2_input_transform(train_config, batch, {})
    finally:
        if held.exists():
            os.replace(held, target)

    assert removed is True
    assert fallback_calls == []


def test_create_lerobot_dataset_never_calls_hf_fallback_when_parquet_disappears(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    target = batch.episodes[0].parquet_path
    held = tmp_path / "held.parquet"
    removed = False
    fallback_calls: list[object] = []
    original_get_paths = feature_extractor.lerobot_dataset.LeRobotDataset.get_episodes_file_paths

    def remove_parquet_then_get_paths(dataset: Any) -> list[str]:
        nonlocal removed
        paths = original_get_paths(dataset)
        if not removed:
            removed = True
            os.replace(target, held)
        return paths

    def forbidden_download(dataset: Any, *args: Any, **kwargs: Any) -> None:
        fallback_calls.append((dataset, args, kwargs))
        raise AssertionError("Hugging Face episode fallback must be unreachable")

    def forbidden_safe_version(*args: Any, **kwargs: Any) -> None:
        fallback_calls.append(("get_safe_version", args, kwargs))
        raise AssertionError("Hugging Face version fallback must be unreachable")

    monkeypatch.setattr(
        feature_extractor.lerobot_dataset.LeRobotDataset,
        "get_episodes_file_paths",
        remove_parquet_then_get_paths,
    )
    monkeypatch.setattr(
        feature_extractor.lerobot_dataset.LeRobotDataset,
        "download_episodes",
        forbidden_download,
    )
    monkeypatch.setattr(feature_extractor.lerobot_dataset, "get_safe_version", forbidden_safe_version)

    try:
        with pytest.raises(feature_extractor.BatchSnapshotError, match="local.*episode|missing"):
            feature_extractor.create_lerobot_frame_dataset(batch)
    finally:
        if held.exists():
            os.replace(held, target)

    assert removed is True
    assert fallback_calls == []


def test_build_stage2_input_transform_copies_tasks_and_closes_pinned_root(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    tasks = {0: "original task"}
    descriptors: list[int] = []

    class _Metadata:
        def __init__(self, *_args: Any, **kwargs: Any):
            descriptors.append(_descriptor_from_proc_root(Path(kwargs["root"])))
            self.tasks = tasks

    source = training_config.DataConfig(repo_id="remote/source", prompt_from_task=True)
    train_config, _ = _make_train_config(source, tmp_path)
    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_metadata", _Metadata)

    data_config, transform = feature_extractor.build_stage2_input_transform(train_config, batch, {})
    tasks[0] = "mutated after build"

    assert data_config.repo_id == str(batch.root)
    assert transform({"task_index": np.asarray(0)})["prompt"] == "original task"
    assert descriptors
    _assert_descriptor_closed(descriptors[0])


def test_build_stage2_input_transform_reads_pinned_tasks_during_root_swap(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    root = batch.root
    held = root.with_name(f"{root.name}-held")
    foreign = root.with_name(f"{root.name}-foreign")
    shutil.copytree(root, foreign)
    (foreign / "meta/tasks.jsonl").write_text(
        '{"task_index": 0, "task": "foreign task"}\n',
        encoding="utf-8",
    )
    descriptors: list[int] = []

    class _Metadata:
        def __init__(self, *_args: Any, **kwargs: Any):
            proc_root = Path(kwargs["root"])
            descriptors.append(_descriptor_from_proc_root(proc_root))
            root.rename(held)
            foreign.rename(root)
            record = json.loads((proc_root / "meta/tasks.jsonl").read_text(encoding="utf-8"))
            self.tasks = {int(record["task_index"]): record["task"]}
            root.rename(foreign)
            held.rename(root)

    source = training_config.DataConfig(repo_id="remote/source", prompt_from_task=True)
    train_config, _ = _make_train_config(source, ready_batch.parent)
    monkeypatch.setattr(feature_extractor, "_construct_local_only_lerobot_metadata", _Metadata)

    _, transform = feature_extractor.build_stage2_input_transform(train_config, batch, {})

    assert transform({"task_index": np.asarray(0)})["prompt"] == "pick and place"
    _assert_descriptor_closed(descriptors[0])


def test_verify_validated_batch_snapshot_streams_and_closes_every_descriptor(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    descriptors: list[int] = []
    requested_sizes: list[int] = []
    real_os_read = os.read

    def record_descriptor(_path: Path, descriptor: int) -> None:
        descriptors.append(descriptor)

    def tracked_read(descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        return real_os_read(descriptor, size)

    def forbid_read_bytes(_path: Path) -> bytes:
        raise AssertionError("snapshot verification must stream instead of using Path.read_bytes")

    monkeypatch.setattr(feature_extractor, "_after_snapshot_file_open", record_descriptor)
    monkeypatch.setattr(feature_extractor.os, "read", tracked_read)
    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)

    feature_extractor.verify_validated_batch_snapshot(batch)

    assert descriptors
    assert requested_sizes
    assert max(requested_sizes) <= feature_extractor.SNAPSHOT_READ_CHUNK_BYTES
    for descriptor in set(descriptors):
        _assert_descriptor_closed(descriptor)


def test_verify_validated_batch_snapshot_closes_descriptor_after_read_error(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    descriptors: list[int] = []

    def fail_after_open(_path: Path, descriptor: int) -> None:
        descriptors.append(descriptor)
        raise OSError("injected snapshot read failure")

    monkeypatch.setattr(feature_extractor, "_after_snapshot_file_open", fail_after_open)

    with pytest.raises(feature_extractor.BatchSnapshotError, match="read failed"):
        feature_extractor.verify_validated_batch_snapshot(batch)

    assert descriptors
    for descriptor in descriptors:
        _assert_descriptor_closed(descriptor)


def test_observation_dataset_maps_nonzero_episode_and_frame_to_exact_global_offset(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    episode = batch.episodes[7]
    key = transitions.FeatureKey(batch.batch_id, episode.episode_index, 5)
    fake = _install_fake_dataset(monkeypatch, batch)
    dataset = feature_extractor.Stage2ObservationDataset(
        batch,
        [key],
        input_transform=lambda value: {"state": value["observation.state"].numpy()},
    )

    returned_key, observation = dataset[0]

    assert returned_key is key
    assert fake.requested_indices == [episode.dataset_from_index + key.frame_index]
    np.testing.assert_array_equal(
        observation["state"],
        np.full(16, episode.dataset_from_index + key.frame_index, dtype=np.float32),
    )


def test_observation_dataset_freezes_keys_and_tail_frame_needs_no_action(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    episode = batch.episodes[-1]
    key = transitions.FeatureKey(batch.batch_id, episode.episode_index, episode.length - 1)
    keys = [key]
    fake = _install_fake_dataset(monkeypatch, batch)

    def transform_without_action(value: dict[str, Any]) -> dict[str, Any]:
        assert "action" not in value
        value["seen"] = True
        return {"state": value["observation.state"], "seen": value["seen"]}

    dataset = feature_extractor.Stage2ObservationDataset(batch, keys, transform_without_action)
    keys.clear()
    returned_key, observation = dataset[0]

    assert len(dataset) == 1
    assert returned_key is key
    assert observation["seen"] is True
    assert fake.rows[-1].get("seen") is None


@pytest.mark.parametrize(
    ("make_key", "match"),
    [
        (lambda batch: transitions.FeatureKey("wrong-batch", 0, 0), "batch"),
        (lambda batch: transitions.FeatureKey(batch.batch_id, 999, 0), "episode"),
        (lambda batch: transitions.FeatureKey(batch.batch_id, 0, -1), "frame"),
        (lambda batch: transitions.FeatureKey(batch.batch_id, 0, batch.episodes[0].length), "frame"),
        (_key_with_boolean_frame, "frame"),
        (lambda batch: transitions.FeatureKey(batch.batch_id, 0, 1.5), "frame"),
    ],
)
def test_observation_dataset_rejects_invalid_feature_keys_at_construction(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_key: Callable[[admission.ValidatedBatch], transitions.FeatureKey],
    match: str,
):
    batch = _validated_batch(ready_batch)
    _install_fake_dataset(monkeypatch, batch)

    with pytest.raises(ValueError, match=match):
        feature_extractor.Stage2ObservationDataset(batch, [make_key(batch)], lambda value: value)


def test_observation_dataset_rejects_non_feature_key(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    _install_fake_dataset(monkeypatch, batch)

    with pytest.raises(TypeError, match="FeatureKey"):
        feature_extractor.Stage2ObservationDataset(batch, [("not", "a", "key")], lambda value: value)


def test_observation_dataset_rejects_dataset_total_length_mismatch(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    _install_fake_dataset(monkeypatch, batch, reported_length=batch.total_frames - 1)

    with pytest.raises(ValueError, match="length"):
        feature_extractor.Stage2ObservationDataset(batch, [], lambda value: value)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("index", 1),
        ("episode_index", 1),
        ("frame_index", 1),
    ],
)
def test_observation_dataset_rejects_raw_frame_identity_mismatch(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong_value: int,
):
    batch = _validated_batch(ready_batch)
    rows = _rows_for_batch(batch)
    rows[0] = {**rows[0], field: torch.tensor(wrong_value)}
    _install_fake_dataset(monkeypatch, batch, rows=rows)
    key = transitions.FeatureKey(batch.batch_id, 0, 0)
    dataset = feature_extractor.Stage2ObservationDataset(batch, [key], lambda value: value)

    with pytest.raises(ValueError, match=field):
        dataset[0]


@pytest.mark.parametrize("missing_field", ["index", "episode_index", "frame_index"])
def test_observation_dataset_requires_every_raw_identity_field(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
):
    batch = _validated_batch(ready_batch)
    rows = _rows_for_batch(batch)
    del rows[0][missing_field]
    _install_fake_dataset(monkeypatch, batch, rows=rows)
    key = transitions.FeatureKey(batch.batch_id, 0, 0)
    dataset = feature_extractor.Stage2ObservationDataset(batch, [key], lambda value: value)

    with pytest.raises(ValueError, match=rf"batch {batch.batch_id}.*episode 0.*frame 0.*{missing_field}"):
        dataset[0]


def test_observation_dataset_read_error_has_frame_context_and_preserves_cause(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    error = ValueError("video decode failed")
    monkeypatch.setattr(
        feature_extractor,
        "_construct_local_only_lerobot_dataset",
        lambda *_args, **_kwargs: _ExplodingFrameDataset(batch.total_frames, error),
    )
    key = transitions.FeatureKey(batch.batch_id, 0, 0)
    dataset = feature_extractor.Stage2ObservationDataset(batch, [key], lambda value: value)

    try:
        with pytest.raises(
            RuntimeError,
            match=rf"batch {batch.batch_id}.*episode 0.*frame 0.*global index 0.*read",
        ) as exc_info:
            dataset[0]
        assert exc_info.value.__cause__ is error
        dataset.verify_unchanged()
    finally:
        dataset.close()


def test_observation_dataset_transform_error_has_frame_context_and_preserves_cause(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch = _validated_batch(ready_batch)
    _install_fake_dataset(monkeypatch, batch)
    error = KeyError("state transform failed")

    def fail_transform(_value: dict[str, Any]) -> dict[str, Any]:
        raise error

    key = transitions.FeatureKey(batch.batch_id, 0, 0)
    dataset = feature_extractor.Stage2ObservationDataset(batch, [key], fail_transform)

    try:
        with pytest.raises(
            RuntimeError,
            match=rf"batch {batch.batch_id}.*episode 0.*frame 0.*global index 0.*transform",
        ) as exc_info:
            dataset[0]
        assert exc_info.value.__cause__ is error
        dataset.verify_unchanged()
    finally:
        dataset.close()


def test_observation_only_repack_removes_only_top_level_actions_without_mutating_group():
    passthrough = _TraceTransform("before")
    output = _TraceTransform("output")
    first = transforms.RepackTransform(
        {
            "images": {"top": "observation.images.top"},
            "state": "observation.state",
            "actions": "action",
            "nested": {"actions": "nested.action"},
        }
    )
    second = transforms.RepackTransform({"state": "observation.state"})
    group = transforms.Group(inputs=[passthrough, first, second], outputs=[output])

    result = feature_extractor.observation_only_repack(group)

    assert isinstance(result.inputs, tuple)
    assert result.inputs[0] is passthrough
    assert result.outputs is group.outputs
    assert first.structure["actions"] == "action"
    assert result.inputs[1].structure == {
        "images": {"top": "observation.images.top"},
        "state": "observation.state",
        "nested": {"actions": "nested.action"},
    }
    assert result.inputs[2].structure == {"state": "observation.state"}


def test_stage2_prompt_prefix_defaults_only_when_missing_and_preserves_explicit_prompt():
    transform = transforms.compose(feature_extractor.stage2_prompt_prefix({}, prompt_from_task=False))

    assert np.asarray(transform({})["prompt"]).item() == "fold clothes"
    assert transform({"prompt": "place the cup"})["prompt"] == "place the cup"


def test_stage2_prompt_prefix_task_is_authoritative_over_missing_or_explicit_prompt():
    transform = transforms.compose(
        feature_extractor.stage2_prompt_prefix({3: "sort the blocks"}, prompt_from_task=True)
    )

    assert transform({"task_index": np.asarray(3)})["prompt"] == "sort the blocks"
    assert transform({"task_index": np.asarray(3), "prompt": "stale prompt"})["prompt"] == "sort the blocks"


def test_stage2_prompt_prefix_rejects_unknown_task_when_task_prompt_is_enabled():
    transform = transforms.compose(feature_extractor.stage2_prompt_prefix({}, prompt_from_task=True))

    with pytest.raises(ValueError, match="task_index"):
        transform({"task_index": np.asarray(0)})


def test_build_stage2_input_transform_rejects_missing_local_root_before_factory_or_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = training_config.DataConfig(repo_id="remote/source")
    train_config, factory = _make_train_config(source, tmp_path)
    metadata_calls = _patch_metadata(monkeypatch, {})
    missing = tmp_path / "missing"

    with pytest.raises(TypeError, match="ValidatedBatch"):
        feature_extractor.build_stage2_input_transform(train_config, missing, {})

    assert factory.calls == []
    assert metadata_calls == []
    assert not missing.exists()


def test_build_stage2_input_transform_preserves_order_and_uses_caller_quantile_stats(
    tmp_path: Path,
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repack = transforms.RepackTransform(
        {
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "trace": "trace",
        }
    )
    data_transform = _TraceTransform("data")
    model_transform = _TraceTransform("model")
    source_stats = _norm_stats(q01=-100.0, q99=100.0)
    caller_stats = _norm_stats(q01=0.0, q99=10.0)
    source = training_config.DataConfig(
        repo_id="remote/source",
        norm_stats=source_stats,
        repack_transforms=transforms.Group(inputs=[repack]),
        data_transforms=transforms.Group(inputs=[data_transform]),
        model_transforms=transforms.Group(inputs=[model_transform]),
        use_quantile_norm=True,
        prompt_from_task=True,
    )
    train_config, factory = _make_train_config(source, tmp_path)
    metadata_calls = _patch_metadata(monkeypatch, {0: "task prompt"})

    data_config, transform = feature_extractor.build_stage2_input_transform(
        train_config,
        _validated_batch(ready_batch),
        caller_stats,
    )

    assert factory.calls == [(train_config.assets_dirs, train_config.model)]
    assert source.repo_id == "remote/source"
    assert source.norm_stats is source_stats
    assert data_config.repo_id == str(ready_batch)
    assert data_config.norm_stats is caller_stats
    assert metadata_calls[0][0][0] == ready_batch.name
    metadata_root = Path(metadata_calls[0][1]["root"])
    assert metadata_root.parent == Path("/proc/self/fd")
    _assert_descriptor_closed(_descriptor_from_proc_root(metadata_root))
    assert isinstance(transform, transforms.CompositeTransform)
    assert [type(item) for item in transform.transforms] == [
        transforms.PromptFromLeRobotTask,
        transforms.InjectDefaultPrompt,
        transforms.RepackTransform,
        _TraceTransform,
        transforms.Normalize,
        _TraceTransform,
    ]
    normalize = transform.transforms[4]
    assert isinstance(normalize, transforms.Normalize)
    assert normalize.norm_stats is caller_stats
    assert normalize.use_quantiles is True
    result = transform(
        {
            "observation.state": np.full(16, 5.0, dtype=np.float32),
            "action": np.ones((50, 16), dtype=np.float32),
            "task_index": np.asarray(0),
            "trace": (),
        }
    )
    assert result["trace"] == ("data", "model")
    np.testing.assert_allclose(result["state"], np.zeros(16, dtype=np.float32), atol=1e-6)
    assert "actions" not in result


def test_build_stage2_input_transform_matches_training_observation_fields_exactly(
    tmp_path: Path,
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repack = transforms.RepackTransform(
        {
            "images": {
                "top": "observation.images.top",
                "left_wrist": "observation.images.left_wrist",
                "right_wrist": "observation.images.right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
    )
    data_group = transforms.Group(
        inputs=[
            training_config.Lite0030Inputs(),
            transforms.DeltaActions(transforms.make_bool_mask(16)),
        ]
    )
    model_group = transforms.Group(
        inputs=[
            transforms.ResizeImages(8, 8),
            transforms.TokenizePrompt(_FakeTokenizer(), discrete_state_input=True),
            transforms.PadStatesAndActions(32),
        ]
    )
    norm_stats = _norm_stats()
    source = training_config.DataConfig(
        repo_id="remote/source",
        repack_transforms=transforms.Group(inputs=[repack]),
        data_transforms=data_group,
        model_transforms=model_group,
        use_quantile_norm=True,
        prompt_from_task=True,
    )
    train_config, _ = _make_train_config(source, tmp_path)
    tasks = {0: "pick and place"}
    _patch_metadata(monkeypatch, tasks)
    _, stage2_transform = feature_extractor.build_stage2_input_transform(
        train_config,
        _validated_batch(ready_batch),
        norm_stats,
    )
    image = np.linspace(0.0, 1.0, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    raw_observation = {
        "observation.images.top": image,
        "observation.images.left_wrist": image[:, :, ::-1].copy(),
        "observation.images.right_wrist": image[:, ::-1, :].copy(),
        "observation.state": np.linspace(-1.0, 1.0, 16, dtype=np.float32),
        "task_index": np.asarray(0),
    }
    training_row = {
        **raw_observation,
        "action": np.linspace(-1.0, 1.0, 50 * 16, dtype=np.float32).reshape(50, 16),
    }
    training_transform = transforms.compose(
        [
            transforms.PromptFromLeRobotTask(tasks),
            *source.repack_transforms.inputs,
            *source.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=True),
            *source.model_transforms.inputs,
        ]
    )

    stage2_result = stage2_transform(dict(raw_observation))
    training_result = training_transform(dict(training_row))

    assert "actions" not in stage2_result
    for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
        np.testing.assert_array_equal(stage2_result[key], training_result[key])
    assert stage2_result["image_mask"] == training_result["image_mask"]
    assert tuple(stage2_result["image"]) == tuple(training_result["image"])
    for key in stage2_result["image"]:
        np.testing.assert_array_equal(stage2_result["image"][key], training_result["image"][key])


@pytest.mark.parametrize(
    ("prompt_from_task", "row", "expected"),
    [
        (False, {}, "fold clothes"),
        (False, {"prompt": "explicit"}, "explicit"),
        (True, {"task_index": np.asarray(0)}, "dataset task"),
        (True, {"task_index": np.asarray(0), "prompt": "explicit"}, "dataset task"),
    ],
)
def test_built_transform_prompt_precedence(
    tmp_path: Path,
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompt_from_task: bool,
    row: dict[str, Any],
    expected: str,
):
    source = training_config.DataConfig(
        repo_id="remote/source",
        norm_stats={},
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({"prompt": "prompt"})]),
        use_quantile_norm=True,
        prompt_from_task=prompt_from_task,
    )
    train_config, _ = _make_train_config(source, tmp_path)
    _patch_metadata(monkeypatch, {0: "dataset task"})
    _, transform = feature_extractor.build_stage2_input_transform(
        train_config,
        _validated_batch(ready_batch),
        {},
    )

    assert transform(dict(row))["prompt"] == expected


def test_real_lerobot_torchcodec_video_reads_through_consecutive_proc_fd_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the real proc-fd video compatibility smoke")
    soft_nofile_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    pinned_descriptor = 4096 if soft_nofile_limit == resource.RLIM_INFINITY else min(4096, soft_nofile_limit - 1)
    if pinned_descriptor < 256:
        pytest.skip("RLIMIT_NOFILE is too low to isolate pinned descriptors from runtime allocations")
    monkeypatch.setattr(feature_extractor, "_MIN_PINNED_DESCRIPTOR", pinned_descriptor)

    first_root = build_ready_batch(
        tmp_path / "batch_000001_real_video",
        lengths=(2,) * 20,
    )
    second_root = build_ready_batch(
        tmp_path / "batch_000002_real_video",
        lengths=(2,) * 20,
    )
    second_parquet = second_root / "data/chunk-000/episode_000000.parquet"
    second_table = pq.read_table(second_parquet)
    control_mode_index = second_table.schema.get_field_index("control_mode")
    second_table = second_table.set_column(
        control_mode_index,
        "control_mode",
        pa.array([7] * len(second_table), type=pa.int64()),
    )
    pq.write_table(second_table, second_parquet)

    def install_valid_videos(root: Path, *, source: str, name: str) -> admission.ValidatedBatch:
        sample_video = tmp_path / name
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
                "-frames:v",
                "2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                os.fspath(sample_video),
            ],
            check=True,
        )
        for video_path in root.glob("videos/chunk-000/*/*.mp4"):
            shutil.copyfile(sample_video, video_path)
        manifest_path = root / "migration_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            path = root / record["target_path"]
            record["size"] = path.stat().st_size
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return _validated_batch(root)

    first_batch = install_valid_videos(
        first_root,
        source="testsrc2=size=640x480:rate=30",
        name="first.mp4",
    )
    second_batch = install_valid_videos(
        second_root,
        source="color=c=red:size=640x480:rate=30",
        name="second.mp4",
    )

    cache_root = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", cache_root)
    monkeypatch.setattr(datasets_config, "DOWNLOADED_DATASETS_PATH", cache_root / "downloads")

    first = feature_extractor.create_lerobot_frame_dataset(first_batch)
    first_descriptor = _descriptor_from_proc_root(first.pinned_root)
    try:
        assert first_descriptor == pinned_descriptor
        first_row = first[0]
        assert first._dataset.video_backend == "torchcodec"  # noqa: SLF001
        for key in (
            "observation.images.top",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ):
            assert first_row[key].shape == (3, 480, 640)
            assert first_row[key].dtype == torch.float32
    finally:
        first.close()
    _assert_descriptor_closed(first_descriptor)
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        first[0]
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        len(first)

    second = feature_extractor.create_lerobot_frame_dataset(second_batch)
    second_descriptor = _descriptor_from_proc_root(second.pinned_root)
    try:
        second_row = second[0]
        assert second_descriptor == pinned_descriptor == first_descriptor
        assert int(first_row["control_mode"]) == 1
        assert int(second_row["control_mode"]) == 7
        assert second_row["observation.images.top"].shape == (3, 480, 640)
        assert (
            abs(first_row["observation.images.top"].mean().item() - second_row["observation.images.top"].mean().item())
            > 0.05
        )
    finally:
        second.close()
    _assert_descriptor_closed(second_descriptor)


def test_collate_observations_keeps_keys_order_and_stacks_numpy_torch_and_jax_leaves():
    keys = (
        transitions.FeatureKey("batch", 1, 2),
        transitions.FeatureKey("batch", 3, 4),
    )
    rows = [
        (
            keys[0],
            {
                "numpy": np.asarray([1, 2], dtype=np.float32),
                "torch": torch.asarray([3, 4], dtype=torch.int64),
                "jax": jnp.asarray([True, False]),
                "nested": {"scalar": np.asarray(5, dtype=np.int16)},
                "python_scalar": 11.5,
            },
        ),
        (
            keys[1],
            {
                "numpy": np.asarray([6, 7], dtype=np.float32),
                "torch": torch.asarray([8, 9], dtype=torch.int64),
                "jax": jnp.asarray([False, True]),
                "nested": {"scalar": np.asarray(10, dtype=np.int16)},
                "python_scalar": 12.5,
            },
        ),
    ]

    collated_keys, observations = feature_extractor.collate_observations(rows)

    assert collated_keys == keys
    leaves = jax.tree.leaves(observations)
    assert all(isinstance(value, np.ndarray) for value in leaves)
    assert all(value.flags.c_contiguous for value in leaves)
    assert observations["numpy"].shape == (2, 2)
    assert observations["numpy"].dtype == np.float32
    assert observations["torch"].shape == (2, 2)
    assert observations["torch"].dtype == np.int64
    assert observations["jax"].shape == (2, 2)
    assert observations["jax"].dtype == np.bool_
    assert observations["nested"]["scalar"].shape == (2,)
    assert observations["nested"]["scalar"].dtype == np.int16
    assert observations["python_scalar"].shape == (2,)
    assert observations["python_scalar"].dtype == np.float64


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray(["text"], dtype=np.str_),
        np.asarray([b"bytes"], dtype=np.bytes_),
        np.asarray([object()], dtype=np.object_),
    ],
)
def test_collate_observations_rejects_string_bytes_and_object_leaves(invalid: np.ndarray):
    key = transitions.FeatureKey("batch", 0, 0)
    rows = [(key, {"invalid": invalid}), (key, {"invalid": invalid.copy()})]

    with pytest.raises(TypeError, match="numeric"):
        feature_extractor.collate_observations(rows)


def test_collate_observations_rejects_empty_rows():
    with pytest.raises(ValueError, match="at least one"):
        feature_extractor.collate_observations([])


def test_noise_for_keys_matches_independent_frame_key_oracle_and_fixed_sha256():
    key = transitions.FeatureKey("batch_000001", 7, 19)
    expected = np.asarray(
        jax.random.normal(
            feature_identity.frame_key("frozen-feature-v1", "batch_000001", 7, 19),
            (50, 32),
            dtype=jnp.float32,
        )
    )

    actual = np.asarray(
        feature_extractor._noise_for_keys(  # noqa: SLF001
            (key,),
            feature_id="frozen-feature-v1",
            action_horizon=50,
            action_dim=32,
        )
    )

    assert actual.shape == (1, 50, 32)
    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual[0], expected)
    assert hashlib.sha256(actual[0].tobytes(order="C")).hexdigest() == (
        "5fdeba96fbba5b48d19bf8f043f8e5164a4fa01033446ed263a06f8f0ed2e89a"
    )


def test_extract_features_is_microbatch_invariant_and_uses_one_shared_prefix_per_batch(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(5)
    model = _FakeExtractionModel()
    first_dataset = _owned_extraction_dataset(keys)
    first = _extract(first_dataset, model, keys, micro_batch_size=1)
    first_call_delta = model.prefix_calls
    second_dataset = _owned_extraction_dataset(keys)
    second = _extract(second_dataset, model, keys, micro_batch_size=4)
    second_call_delta = model.prefix_calls - first_call_delta

    np.testing.assert_array_equal(first.episode_index, second.episode_index)
    np.testing.assert_array_equal(first.frame_index, second.frame_index)
    np.testing.assert_array_equal(first.z_rl, second.z_rl)
    np.testing.assert_array_equal(first.state_norm, second.state_norm)
    np.testing.assert_array_equal(first.vla_reference, second.vla_reference)
    assert first_call_delta == math.ceil(len(keys) / 1)
    assert second_call_delta == math.ceil(len(keys) / 4)
    assert len(direct_module_jit) == 2
    assert first_dataset.close_calls == 1
    assert second_dataset.close_calls == 1


def test_extract_features_pre_parameter_mismatch_never_constructs_sampler_or_samples(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    model = _FakeExtractionModel()
    dataset = _owned_extraction_dataset(keys)

    with pytest.raises(RuntimeError, match="parameter.*expected"):
        _extract(
            dataset,
            model,
            keys,
            expected_parameter_sha256="0" * 64,
        )

    assert model.prefix_calls == 0
    assert direct_module_jit == []
    assert dataset.close_calls == 1


def test_noise_changes_when_any_identity_coordinate_changes():
    base = transitions.FeatureKey("batch_000001", 7, 19)
    variants = (
        ("frozen-feature-v1", base),
        ("frozen-feature-v2", base),
        ("frozen-feature-v1", dataclasses.replace(base, batch_id="batch_000002")),
        ("frozen-feature-v1", dataclasses.replace(base, episode_index=8)),
        ("frozen-feature-v1", dataclasses.replace(base, frame_index=20)),
    )

    digests = {
        hashlib.sha256(
            np.asarray(
                feature_extractor._noise_for_keys(  # noqa: SLF001
                    (key,),
                    feature_id=feature_id,
                    action_horizon=50,
                    action_dim=32,
                )
            ).tobytes(order="C")
        ).hexdigest()
        for feature_id, key in variants
    }

    assert len(digests) == len(variants)


def test_extract_features_has_exact_cache_shapes_slices_dtypes_and_contiguity(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(5)
    model = _FakeExtractionModel()
    dataset = _owned_extraction_dataset(keys)

    result = _extract(dataset, model, keys)

    assert result.episode_index.shape == (5,)
    assert result.frame_index.shape == (5,)
    assert result.z_rl.shape == (5, 2048)
    assert result.state_norm.shape == (5, 16)
    assert result.vla_reference.shape == (5, 20, 16)
    assert result.episode_index.dtype == np.dtype(np.int32)
    assert result.frame_index.dtype == np.dtype(np.int32)
    assert result.z_rl.dtype == np.dtype(ml_dtypes.bfloat16)
    assert result.state_norm.dtype == np.dtype(np.float32)
    assert result.vla_reference.dtype == np.dtype(np.float32)
    assert all(
        value.flags.c_contiguous
        for value in (
            result.episode_index,
            result.frame_index,
            result.z_rl,
            result.state_norm,
            result.vla_reference,
        )
    )
    np.testing.assert_array_equal(
        result.episode_index,
        np.asarray([key.episode_index for key in keys], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.frame_index,
        np.asarray([key.frame_index for key in keys], dtype=np.int32),
    )
    expected_state = np.stack(
        [_extraction_observation(row)["state"][:16] for row in range(len(keys))],
        axis=0,
    ).astype(np.float32)
    np.testing.assert_array_equal(result.state_norm, expected_state)

    expected_noise = np.stack(
        [
            np.asarray(
                jax.random.normal(
                    feature_identity.frame_key(
                        "frozen-feature-v1",
                        key.batch_id,
                        key.episode_index,
                        key.frame_index,
                    ),
                    (50, 32),
                    dtype=jnp.float32,
                )
            )
            for key in keys
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result.vla_reference, expected_noise[:, :20, :16])
    flattened = expected_noise.reshape(len(keys), -1)
    expected_z = np.tile(flattened, (1, math.ceil(2048 / flattened.shape[1])))[:, :2048]
    np.testing.assert_array_equal(
        result.z_rl,
        expected_z.astype(ml_dtypes.bfloat16),
    )
    assert dataset.verify_calls == 2
    assert dataset.close_calls == 1
    assert len(direct_module_jit) == 1


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("empty", "nonempty"),
        ("duplicate", "unique sorted"),
        ("unsorted", "unique sorted"),
        ("foreign", "batch"),
        ("mismatch", "exactly match"),
    ],
)
def test_extract_features_rejects_invalid_or_dataset_mismatched_keys_and_closes(
    kind: str,
    match: str,
    direct_module_jit: list[Callable[..., Any]],
):
    valid = _extraction_keys(3)
    dataset = _owned_extraction_dataset(valid)
    passed = valid
    if kind == "empty":
        passed = ()
    elif kind == "duplicate":
        passed = (valid[0], valid[0], valid[1])
    elif kind == "unsorted":
        passed = (valid[1], valid[0], valid[2])
    elif kind == "foreign":
        passed = tuple(dataclasses.replace(key, batch_id="foreign") for key in valid)
    elif kind == "mismatch":
        passed = valid[:-1]

    with pytest.raises((TypeError, ValueError), match=match):
        _extract(dataset, _FakeExtractionModel(), passed)

    assert dataset.close_calls == 1


def test_extract_features_rejects_feature_index_that_overflows_int32_cache_identity(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = (transitions.FeatureKey("batch_000001", 2**31, 0),)
    dataset = _owned_extraction_dataset(keys)

    with pytest.raises(ValueError, match="int32 cache range"):
        _extract(dataset, _FakeExtractionModel(), keys)

    assert dataset.close_calls == 1


@pytest.mark.parametrize("kind", ["reorder", "duplicate", "drop", "foreign"])
def test_extract_features_rejects_loader_reorder_duplicate_drop_or_foreign_key(
    kind: str,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(5)
    returned = keys
    reported_length = None
    if kind == "reorder":
        returned = (keys[1], keys[0], *keys[2:])
    elif kind == "duplicate":
        returned = (keys[0], keys[0], *keys[2:])
    elif kind == "drop":
        reported_length = len(keys) - 1
    elif kind == "foreign":
        returned = (dataclasses.replace(keys[0], batch_id="foreign"), *keys[1:])
    dataset = _owned_extraction_dataset(
        keys,
        returned_keys=returned,
        reported_length=reported_length,
    )

    with pytest.raises(ValueError, match="loader.*exact"):
        _extract(dataset, _FakeExtractionModel(), keys)

    assert dataset.close_calls == 1


def test_extract_features_builds_strict_synchronous_dataloader(
    monkeypatch: pytest.MonkeyPatch,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    original = torch.utils.data.DataLoader
    captured: dict[str, Any] = {}

    def data_loader(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_extractor.torch.utils.data, "DataLoader", data_loader)

    _extract(dataset, _FakeExtractionModel(), keys)

    assert captured["shuffle"] is False
    assert captured["drop_last"] is False
    assert captured["num_workers"] == 0
    assert captured["collate_fn"] is feature_extractor.collate_observations
    assert "persistent_workers" not in captured or captured["persistent_workers"] is False


@pytest.mark.parametrize("num_workers", [-1, 1, True])
def test_extract_features_rejects_every_nonzero_or_nonexact_worker_count(
    num_workers: object,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)

    with pytest.raises(ValueError, match="num_workers.*zero"):
        _extract(dataset, _FakeExtractionModel(), keys, num_workers=num_workers)

    assert dataset.close_calls == 1
    assert direct_module_jit == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("micro_batch_size", 0, "micro_batch_size.*positive"),
        ("micro_batch_size", True, "micro_batch_size.*integer"),
        ("sampler_num_steps", 0, "sampler_num_steps.*positive"),
        ("sampler_num_steps", True, "sampler_num_steps.*integer"),
        ("feature_id", "", "feature_id.*nonempty"),
        ("expected_parameter_sha256", "A" * 64, "expected_parameter_sha256"),
        ("expected_parameter_sha256", "not-a-hash", "expected_parameter_sha256"),
    ],
)
def test_extract_features_rejects_invalid_arguments_and_still_closes_owned_dataset(
    field: str,
    value: object,
    match: str,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)

    with pytest.raises((TypeError, ValueError), match=match):
        _extract(dataset, _FakeExtractionModel(), keys, **{field: value})

    assert dataset.close_calls == 1


@pytest.mark.parametrize(
    ("action_horizon", "action_dim", "match"),
    [
        (49, 32, "action_horizon.*50"),
        (50, 31, "action_dim.*32"),
        (True, 32, "action_horizon.*integer"),
    ],
)
def test_extract_features_locks_production_model_action_shape_before_sampling(
    action_horizon: int,
    action_dim: int,
    match: str,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(
        action_horizon=action_horizon,
        action_dim=action_dim,
    )

    with pytest.raises(ValueError, match=match):
        _extract(dataset, model, keys)

    assert model.prefix_calls == 0
    assert direct_module_jit == []
    assert dataset.close_calls == 1


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("action_batch", r"actions.*shape.*\[B,50,32\]"),
        ("action_horizon", r"actions.*shape.*\[B,50,32\]"),
        ("action_dim", r"actions.*shape.*\[B,50,32\]"),
        ("action_dtype", "actions.*float32"),
        ("action_nan", "actions.*finite"),
        ("action_inf", "actions.*finite"),
        ("z_batch", r"z_rl.*shape.*\[B,2048\]"),
        ("z_width", r"z_rl.*shape.*\[B,2048\]"),
        ("z_dtype", "z_rl.*floating"),
        ("z_nan", "z_rl.*finite"),
        ("z_inf", "z_rl.*finite"),
        ("z_overflow", "z_rl.*finite.*conversion"),
    ],
)
def test_extract_features_rejects_bad_sampler_shape_dtype_finite_and_overflow(
    fault: str,
    match: str,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)

    with pytest.raises(ValueError, match=match):
        _extract(dataset, _FakeExtractionModel(fault=fault), keys)

    assert dataset.close_calls == 1


@pytest.mark.parametrize(
    ("state_width", "state_fault", "match"),
    [
        (15, None, r"state.*\[B,>=16\]"),
        (20, "nan", "state.*finite"),
        (20, "inf", "state.*finite"),
        (20, "overflow", "state.*finite"),
    ],
)
def test_extract_features_rejects_short_nonfinite_or_overflowed_normalized_state(
    state_width: int,
    state_fault: str | None,
    match: str,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(
        keys,
        state_width=state_width,
        state_fault=state_fault,
    )
    model = _FakeExtractionModel()

    with pytest.raises(ValueError, match=match):
        _extract(dataset, model, keys)

    assert model.prefix_calls == 0
    assert dataset.close_calls == 1


def test_extract_features_rejects_wrong_rank_normalized_state_before_sampling(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    observations = tuple(
        {
            **_extraction_observation(row),
            "state": np.zeros((1, 20), dtype=np.float32),
        }
        for row in range(2)
    )
    dataset = _OwnedExtractionDataset(keys, observations)
    model = _FakeExtractionModel()

    with pytest.raises(ValueError, match=r"state.*\[B,>=16\]"):
        _extract(dataset, model, keys)

    assert model.prefix_calls == 0
    assert dataset.close_calls == 1


def test_extract_features_accepts_bfloat16_z_and_materializes_bfloat16_cache(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(fault="z_bfloat16")

    result = _extract(dataset, model, keys)

    assert result.z_rl.dtype == np.dtype(ml_dtypes.bfloat16)
    assert np.isfinite(result.z_rl).all()


def test_extract_features_with_frozen_guard_exposes_the_strict_owned_api(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel()

    result = feature_extractor.extract_features_with_frozen_guard(
        model=model,
        dataset=dataset,
        feature_keys=keys,
        feature_id="frozen-feature-v1",
        expected_parameter_sha256=_parameter_sha256(model),
        micro_batch_size=1,
        num_workers=0,
        sampler_num_steps=10,
    )

    assert result.z_rl.shape == (2, 2048)
    assert model.prefix_calls == 2
    assert dataset.close_calls == 1


def test_extract_features_detects_parameter_mutation_after_sampling_and_closes(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(mutate_parameter=True)

    with pytest.raises(RuntimeError, match="parameter.*changed"):
        _extract(dataset, model, keys)

    assert model.prefix_calls == 1
    assert dataset.verify_calls == 2
    assert dataset.close_calls == 1


def test_extract_features_parameter_guard_wins_over_sample_error_and_chains_it(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(
        mutate_parameter=True,
        sample_error=True,
    )

    with pytest.raises(RuntimeError, match="parameter.*changed") as exc_info:
        _extract(dataset, model, keys)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "fake shared-prefix sample failed" in str(exc_info.value.__cause__)
    assert dataset.close_calls == 1


def test_extract_features_ignores_nonparameter_state_mutation_for_frozen_guard(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(mutate_non_parameter=True)

    result = _extract(dataset, model, keys)

    assert result.z_rl.shape == (2, 2048)
    assert model.non_parameter_counter == 1
    assert dataset.close_calls == 1


def test_extract_features_lazy_snapshot_mutation_is_a_guard_failure_and_closes(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(
        keys,
        mutate_snapshot_on_get=True,
    )

    with pytest.raises(RuntimeError, match="snapshot.*changed"):
        _extract(dataset, _FakeExtractionModel(), keys)

    assert dataset.verify_calls == 2
    assert dataset.close_calls == 1


def test_extract_features_combines_snapshot_and_parameter_guard_failures_over_sample_error(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(
        keys,
        mutate_snapshot_on_get=True,
    )
    model = _FakeExtractionModel(
        mutate_parameter=True,
        sample_error=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        _extract(dataset, model, keys)

    message = str(exc_info.value)
    assert "snapshot" in message
    assert "parameter" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "fake shared-prefix sample failed" in str(exc_info.value.__cause__)
    assert dataset.close_calls == 1


def test_extract_features_preserves_sample_error_when_post_guards_pass(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(sample_error=True)

    with pytest.raises(RuntimeError, match="fake shared-prefix sample failed") as exc_info:
        _extract(dataset, model, keys)

    assert exc_info.value.__cause__ is None
    assert dataset.verify_calls == 2
    assert dataset.close_calls == 1


def test_extract_features_runs_parameter_post_hash_even_after_sampling_error(
    monkeypatch: pytest.MonkeyPatch,
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    model = _FakeExtractionModel(sample_error=True)
    original = feature_identity.parameter_tree_sha256
    expected = original(nnx.state(model, nnx.Param))
    calls = 0

    def spy(state: nnx.State) -> str:
        nonlocal calls
        calls += 1
        return original(state)

    monkeypatch.setattr(feature_extractor.feature_identity, "parameter_tree_sha256", spy)

    with pytest.raises(RuntimeError, match="fake shared-prefix sample failed"):
        feature_extractor.extract_features(
            model=model,
            dataset=dataset,
            feature_keys=keys,
            feature_id="frozen-feature-v1",
            expected_parameter_sha256=expected,
            micro_batch_size=4,
            num_workers=0,
            sampler_num_steps=10,
        )

    assert calls == 2
    assert dataset.close_calls == 1


def test_extract_features_pre_snapshot_failure_never_constructs_sampler_and_still_post_checks(
    direct_module_jit: list[Callable[..., Any]],
):
    keys = _extraction_keys(2)
    dataset = _owned_extraction_dataset(keys)
    dataset._snapshot_changed = True  # noqa: SLF001
    model = _FakeExtractionModel()

    with pytest.raises(RuntimeError, match="snapshot.*changed"):
        _extract(dataset, model, keys)

    assert direct_module_jit == []
    assert model.prefix_calls == 0
    assert dataset.verify_calls == 2
    assert dataset.close_calls == 1


def test_extract_features_plan_keys_drive_exact_current_next_topology_without_fake_terminal_next(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_module_jit: list[Callable[..., Any]],
):
    batch = _validated_batch(ready_batch)
    plan = transitions.build_transition_plan(batch)
    _install_fake_dataset(monkeypatch, batch)

    def transform(row: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(row["observation.state"], dtype=np.float32)
        return {
            "image": {"dummy": np.zeros((2, 2, 3), dtype=np.float32)},
            "image_mask": {"dummy": np.ones((), dtype=np.bool_)},
            "state": state,
        }

    dataset = feature_extractor.Stage2ObservationDataset(
        batch,
        plan.feature_keys,
        transform,
    )
    features = _extract(
        dataset,
        _FakeExtractionModel(),
        plan.feature_keys,
        micro_batch_size=4,
    )
    raw = transitions.RawTransitionTable(
        episode_index=np.asarray([row.episode_index for row in plan.rows], dtype=np.int32),
        start_frame_index=np.asarray([row.start_frame_index for row in plan.rows], dtype=np.int32),
        executed_action=np.zeros((len(plan.rows), 20, 16), dtype=np.float32),
        intervention=np.zeros((len(plan.rows), 20), dtype=np.bool_),
        reward=np.asarray([[row.reward] for row in plan.rows], dtype=np.float32),
        terminal=np.asarray([[row.terminal] for row in plan.rows], dtype=np.bool_),
    )

    finalized = cache.finalize_transition_table(batch, plan, raw, features)

    expected_keys = [(key.episode_index, key.frame_index) for key in plan.feature_keys]
    assert list(zip(features.episode_index.tolist(), features.frame_index.tolist(), strict=True)) == expected_keys
    np.testing.assert_array_equal(
        finalized.next_feature_row == -1,
        finalized.terminal[:, 0],
    )
    for row, planned in enumerate(plan.rows):
        if planned.next_key is None:
            assert finalized.next_feature_row[row] == -1
        else:
            next_row = int(finalized.next_feature_row[row])
            assert (
                int(features.episode_index[next_row]),
                int(features.frame_index[next_row]),
            ) == (
                planned.next_key.episode_index,
                planned.next_key.frame_index,
            )
    with pytest.raises(feature_extractor.BatchSnapshotError, match="closed"):
        len(dataset)


def test_small_real_pi0_module_jit_accepts_numpy_collate_and_returns_actions_and_z():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        action_dim=4,
        max_token_len=8,
        rl_token_enabled=True,
        rl_token_reconstruction_weight=1.0,
        rl_token_encoder_depth=1,
        rl_token_decoder_depth=1,
        rl_token_width=64,
        rl_token_num_heads=2,
        rl_token_mlp_dim=128,
        rl_token_max_prefix_len=968,
        rl_token_dropout=0.0,
        rl_token_compute_dtype="bfloat16",
    )
    model = config.create(jax.random.key(0))
    batched = {key: value for key, value in config.fake_obs(batch_size=2).to_dict().items() if value is not None}
    keys = _extraction_keys(2)
    rows = [
        (
            keys[index],
            jax.tree.map(lambda value, row=index: np.asarray(value[row]), batched),
        )
        for index in range(2)
    ]

    collated_keys, raw_observation = feature_extractor.collate_observations(rows)
    observation = model_api.Observation.from_dict(
        jax.tree.map(lambda value: jnp.asarray(np.asarray(value)), raw_observation)
    )
    noise = jax.random.normal(jax.random.key(1), (2, 2, 4), dtype=jnp.float32)
    actions, z_rl = nnx_utils.module_jit(model.sample_actions_and_rl_token)(
        jax.random.key(0),
        observation,
        num_steps=1,
        noise=noise,
    )

    assert collated_keys == keys
    assert actions.shape == (2, 2, 4)
    assert actions.dtype == jnp.float32
    assert z_rl.shape == (2, 64)
    assert z_rl.dtype == jnp.bfloat16
