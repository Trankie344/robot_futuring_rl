from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import weakref

import numpy as np
import pytest

from scripts.tools.rl_token import build_stage2_cache as rlt_stage2_build_cache

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_N = "8" * 64
_COMMIT = "1" * 40


@dataclasses.dataclass(frozen=True)
class _FakeAssets:
    assets_dir: str | None
    asset_id: str | None


@dataclasses.dataclass(frozen=True)
class _FakeDataFactory:
    repo_id: str
    assets: _FakeAssets


@dataclasses.dataclass(frozen=True)
class _FakeModelConfig:
    calls: list[tuple[str, Any]]
    rl_token_enabled: bool = True
    action_horizon: int = 50
    action_dim: int = 32

    def load(self, params: Any):
        self.calls.append(("model_load", params))
        return _FakeModel()


@dataclasses.dataclass(frozen=True)
class _FakeTrainConfig:
    model: _FakeModelConfig
    data: _FakeDataFactory
    assets_dirs: Path


@dataclasses.dataclass(frozen=True)
class _FakeDataConfig:
    asset_id: str
    repack_transforms: object
    data_transforms: object
    model_transforms: object
    use_quantile_norm: bool = True
    video_tolerance_s: float = 0.05


@dataclasses.dataclass(frozen=True)
class _FakeFeatureIdentityInput:
    checkpoint_sha256: str
    norm_stats_sha256: str
    model_config: dict[str, Any]
    transform_config: dict[str, Any]
    sampler_num_steps: int
    seed_version: int
    code_commit: str


class _FakeModel:
    pass


@dataclasses.dataclass(frozen=True)
class _FakeNormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class _FakeState:
    def __init__(self, dtype: np.dtype[Any] | None = None):
        if dtype is None:
            dtype = np.dtype(np.float32)
        self.variable = SimpleNamespace(value=np.ones((2, 2), dtype=dtype))

    def filter(self, selector: object) -> _FakeState:
        assert selector is _FakeNNX.Param
        return self

    def flat_state(self) -> dict[tuple[str], object]:
        return {("weights",): self.variable}


class _FakeNNX:
    Param = object()

    @staticmethod
    def state(model: object, selector: object):
        assert isinstance(model, _FakeModel)
        assert selector is _FakeNNX.Param
        return _FakeState()


def _config(
    tmp_path: Path,
    **overrides: Any,
) -> rlt_stage2_build_cache.BuildCacheConfig:
    checkpoint = tmp_path / "checkpoint" / "54999"
    batch = tmp_path / "batch_000001"
    training_root = tmp_path / "training"
    checkpoint.mkdir(parents=True)
    batch.mkdir()
    training_root.mkdir()
    values: dict[str, Any] = {
        "checkpoint": checkpoint,
        "batch": batch,
        "training_root": training_root,
        "round_id": "round_000001",
    }
    values.update(overrides)
    return rlt_stage2_build_cache.BuildCacheConfig(**values)


def _batch(config: rlt_stage2_build_cache.BuildCacheConfig) -> SimpleNamespace:
    return SimpleNamespace(
        batch_id=config.batch.name,
        root=config.batch,
        manifest_sha256=_SHA_C,
        labels_sha256=_SHA_D,
        episode_fingerprints=("e" * 64, "f" * 64),
    )


def _load_runtime(
    config: rlt_stage2_build_cache.BuildCacheConfig,
    *,
    checkpoint_hashes: tuple[str, ...] = (_SHA_A, _SHA_A),
    norm_hashes: tuple[str, ...] = (_SHA_B, _SHA_B),
    restore_error: BaseException | None = None,
    rl_token_enabled: bool = True,
) -> tuple[SimpleNamespace, list[Any], _FakeDataConfig, object]:
    calls: list[Any] = []
    params_path = config.checkpoint / "params"
    norm_root = config.checkpoint / "assets" / "asset"
    params_path.mkdir(parents=True)
    norm_root.mkdir(parents=True)
    (params_path / "checkpoint.bin").write_bytes(b"frozen-params")
    (norm_root / "norm.json").write_bytes(b"frozen-norm")
    model_config = _FakeModelConfig(calls, rl_token_enabled=rl_token_enabled)
    train_config = _FakeTrainConfig(
        model=model_config,
        data=_FakeDataFactory(
            repo_id="remote/source",
            assets=_FakeAssets("/wrong/original/assets", "asset"),
        ),
        assets_dirs=Path("/wrong/config/assets"),
    )
    actual_data_config = _FakeDataConfig(
        asset_id="asset",
        repack_transforms=object(),
        data_transforms=object(),
        model_transforms=object(),
    )
    actual_transform = object()
    hash_values = {
        params_path: list(checkpoint_hashes),
        norm_root: list(norm_hashes),
    }

    def checkpoint_tree_sha256(path: Path) -> str:
        path = Path(path)
        calls.append(("disk_hash", path))
        if path in hash_values:
            values = hash_values[path]
            if len(values) > 1:
                return values.pop(0)
            return values[0]
        if path.name == "params" and ".rlt-stage2-snapshot-" in path.parent.name:
            return checkpoint_hashes[0]
        if path.name == "asset" and ".rlt-stage2-snapshot-" in path.parents[1].name:
            return norm_hashes[0]
        raise AssertionError(f"unexpected tree hash path: {path}")

    def restore_params(path: Path, *args: Any, **kwargs: Any):
        calls.append(("restore_params", Path(path), args, kwargs))
        if restore_error is not None:
            raise restore_error
        return {"weights": "fp32"}

    def load_norm_stats(assets_dir: Path, asset_id: str):
        calls.append(("load_norm_stats", Path(assets_dir), asset_id))
        stats = _FakeNormStats(
            mean=np.asarray([0.0, 1.0], dtype=np.float32),
            std=np.asarray([1.0, 2.0], dtype=np.float32),
        )
        return {"state": stats, "actions": stats}

    def build_stage2_input_transform(
        passed_train_config: _FakeTrainConfig,
        passed_batch: object,
        norm_stats: object,
    ):
        calls.append(
            (
                "build_input_transform",
                passed_train_config,
                passed_batch,
                norm_stats,
            )
        )
        assets_dir = Path(passed_train_config.data.assets.assets_dir)
        assert assets_dir.parent == Path("/proc/self/fd")
        assert assets_dir.is_dir()
        return actual_data_config, actual_transform

    def parameter_tree_sha256(state: object) -> str:
        calls.append(("loaded_parameter_hash", state))
        return _SHA_C

    def transform_signature(value: object) -> dict[str, Any]:
        calls.append(("transform_signature", value))
        return {"signature_index": len([call for call in calls if call[0] == "transform_signature"])}

    def build_feature_identity(value: _FakeFeatureIdentityInput) -> str:
        calls.append(("build_feature_identity", value))
        return _SHA_D

    runtime = SimpleNamespace(
        training_config=SimpleNamespace(
            get_stage1_config=lambda name: calls.append(("get_stage1_config", name)) or train_config,
            get_stage2_config=lambda name: calls.append(("get_stage2_config", name)) or object(),
        ),
        model_api=SimpleNamespace(restore_params=restore_params),
        checkpoints=SimpleNamespace(load_norm_stats=load_norm_stats),
        feature_extractor=SimpleNamespace(
            DEFAULT_PROMPT="fold clothes",
            build_stage2_input_transform=build_stage2_input_transform,
        ),
        feature_identity=SimpleNamespace(
            checkpoint_tree_sha256=checkpoint_tree_sha256,
            parameter_tree_sha256=parameter_tree_sha256,
            canonical_config_value=lambda value: value,
            transform_signature=transform_signature,
            build_feature_identity=build_feature_identity,
            FeatureIdentityInput=_FakeFeatureIdentityInput,
        ),
        nnx=_FakeNNX,
        jax=SimpleNamespace(device_get=lambda value: value),
    )
    return runtime, calls, actual_data_config, actual_transform


def _train_config_only_runtime(calls: list[Any]) -> SimpleNamespace:
    train_config = _FakeTrainConfig(
        model=_FakeModelConfig(calls),
        data=_FakeDataFactory(
            repo_id="remote/source",
            assets=_FakeAssets("/wrong/original/assets", "asset"),
        ),
        assets_dirs=Path("/wrong/config/assets"),
    )
    return SimpleNamespace(
        training_config=SimpleNamespace(
            get_stage1_config=lambda name: train_config,
            get_stage2_config=lambda name: object(),
        ),
    )


@dataclasses.dataclass(frozen=True)
class _Features:
    value: np.ndarray


@dataclasses.dataclass(frozen=True)
class _Transitions:
    value: np.ndarray


def _validate_fake_features(features: _Features) -> int:
    assert isinstance(features, _Features)
    return int(features.value.shape[0])


def _validate_fake_transitions(
    transitions: _Transitions,
    *,
    features: _Features,
) -> int:
    assert isinstance(transitions, _Transitions)
    assert isinstance(features, _Features)
    return int(transitions.value.shape[0])


def _validate_fake_identity(identity_fields: dict[str, Any]) -> dict[str, Any]:
    assert type(identity_fields) is dict
    return dict(identity_fields)


def _open_fake_staged_shard(
    path: Path,
    *,
    manifest_update: dict[str, Any] | None = None,
) -> SimpleNamespace:
    payload = (path / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    if manifest_update:
        manifest.update(manifest_update)
    return SimpleNamespace(
        root=path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _real_cache_inputs() -> tuple[SimpleNamespace, object, object, dict[str, str]]:
    import ml_dtypes

    from openpi.training.rl_token.stage2 import cache as cache_api

    features = cache_api.FeatureTable(
        episode_index=np.asarray([0, 0], dtype=np.int32),
        frame_index=np.asarray([0, 20], dtype=np.int32),
        z_rl=np.zeros((2, 4), dtype=ml_dtypes.bfloat16),
        state_norm=np.zeros((2, 16), dtype=np.float32),
        vla_reference=np.zeros((2, 20, 16), dtype=np.float32),
    )
    transitions = cache_api.TransitionTable(
        episode_index=np.asarray([0], dtype=np.int32),
        start_frame_index=np.asarray([0], dtype=np.int32),
        current_feature_row=np.asarray([0], dtype=np.int32),
        next_feature_row=np.asarray([1], dtype=np.int32),
        executed_action=np.zeros((1, 20, 16), dtype=np.float32),
        bc_anchor=np.zeros((1, 20, 16), dtype=np.float32),
        reward=np.zeros((1, 1), dtype=np.float32),
        terminal=np.zeros((1, 1), dtype=np.bool_),
    )
    return (
        SimpleNamespace(cache=cache_api),
        features,
        transitions,
        {"feature_identity": _SHA_D, "batch_id": "batch"},
    )


class _OwnedDataset:
    def __init__(self, calls: list[Any]):
        self.calls = calls
        self.closed = False

    def close(self) -> None:
        self.calls.append("dataset_close")
        self.closed = True


def _run_runtime(
    config: rlt_stage2_build_cache.BuildCacheConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extraction_error: BaseException | None = None,
    reread_manifest_update: dict[str, Any] | None = None,
) -> tuple[SimpleNamespace, list[Any], weakref.ReferenceType[object]]:
    calls: list[Any] = []
    batch = _batch(config)
    plan = SimpleNamespace(feature_keys=("key",), rows=("row",))
    raw = object()
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transition_table = _Transitions(np.asarray([3, 4], dtype=np.int32))
    model_ref: weakref.ReferenceType[object] | None = None

    class Model:
        pass

    def fake_load(
        passed_config: rlt_stage2_build_cache.BuildCacheConfig,
        passed_batch: object,
        *,
        runtime: object,
        code_commit: str,
        training_root_witness: object,
    ):
        nonlocal model_ref
        calls.append("load")
        assert passed_config is config
        assert passed_batch is batch
        assert code_commit == _COMMIT
        assert training_root_witness is not None
        model = Model()
        model_ref = weakref.ref(model)
        return rlt_stage2_build_cache.LoadedFrozenModel(
            model=model,
            train_config=object(),
            data_config=SimpleNamespace(asset_id="asset"),
            norm_stats={"state": object(), "actions": object()},
            input_transform="transform",
            feature_id=_SHA_D,
            checkpoint_sha256=_SHA_A,
            norm_stats_sha256=_SHA_B,
            loaded_parameter_sha256=_SHA_C,
            loaded_norm_stats_sha256=_SHA_N,
        )

    monkeypatch.setattr(rlt_stage2_build_cache, "current_git_commit", lambda: _COMMIT)
    monkeypatch.setattr(rlt_stage2_build_cache, "load_model_and_transforms", fake_load)
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_publish_or_verify_round",
        lambda *args, **kwargs: calls.append("round") or config.training_root / "admissions/round_000001.json",
    )

    class FakeAdmission:
        validate_video_with_ffprobe = object()

        @staticmethod
        def validate_ready_batch(path: Path, *, video_validator: object):
            calls.append("admit")
            assert path == config.batch
            assert video_validator is FakeAdmission.validate_video_with_ffprobe
            return batch

    class FakeNormalizer:
        @classmethod
        def from_norm_stats(cls, norm_stats: object):
            calls.append("normalizer")
            return cls()

    class FakeTransitions:
        Stage2Normalizer = FakeNormalizer

        @staticmethod
        def build_transition_plan(passed_batch: object):
            calls.append("plan")
            assert passed_batch is batch
            return plan

        @staticmethod
        def build_raw_transition_table(passed_batch: object, passed_plan: object, normalizer: object):
            calls.append("raw")
            assert (passed_batch, passed_plan) == (batch, plan)
            assert isinstance(normalizer, FakeNormalizer)
            return raw

    def make_observation_dataset(
        passed_batch: object,
        keys: object,
        transform: object,
    ):
        calls.append("dataset")
        assert (passed_batch, keys, transform) == (
            batch,
            plan.feature_keys,
            "transform",
        )
        return _OwnedDataset(calls)

    def extract_features_with_frozen_guard(**kwargs: Any):
        calls.append(("extract", kwargs["expected_parameter_sha256"]))
        assert kwargs["expected_parameter_sha256"] == _SHA_C
        assert kwargs["num_workers"] == 0
        dataset = kwargs["dataset"]
        dataset.close()
        if extraction_error is not None:
            raise extraction_error
        return features

    fake_feature_extractor = SimpleNamespace(
        DEFAULT_PROMPT="fold clothes",
        Stage2ObservationDataset=make_observation_dataset,
        extract_features_with_frozen_guard=extract_features_with_frozen_guard,
    )

    expected_identity = {
        "feature_identity": _SHA_D,
        "checkpoint_sha256": _SHA_A,
        "norm_stats_sha256": _SHA_B,
        "loaded_parameter_sha256": _SHA_C,
        "loaded_norm_stats_sha256": _SHA_N,
        "batch_id": config.batch.name,
        "migration_manifest_sha256": _SHA_C,
        "labels_sha256": _SHA_D,
        "episode_fingerprints": ["e" * 64, "f" * 64],
        "round_id": config.round_id,
        "config_name": config.stage1_config,
        "stage1_checkpoint_step": 54999,
        "stage1_config": config.stage1_config,
        "stage2_config": config.stage2_config,
        "reward_source": "tristate",
        "reward_label_values": [-1, 0, 1, 2],
        "completion_label": 2,
        "reward_aggregation": "sum_20_frames",
        "reward_schema_version": 1,
        "tristate_labels_sha256": _SHA_D,
        "asset_id": "asset",
        "sampler_num_steps": config.sampler_num_steps,
        "default_prompt": "fold clothes",
        "code_commit": _COMMIT,
    }

    class FakeCache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)

        @staticmethod
        def finalize_transition_table(
            passed_batch: object,
            passed_plan: object,
            passed_raw: object,
            passed_features: object,
        ):
            calls.append("finalize")
            assert (passed_batch, passed_plan, passed_raw, passed_features) == (
                batch,
                plan,
                raw,
                features,
            )
            return transition_table

        @staticmethod
        def _validate_identity_fields(identity_fields: dict[str, Any]):
            assert identity_fields == expected_identity
            calls.append(("stage", identity_fields))
            return dict(identity_fields)

        @staticmethod
        def open_shard(destination: Path):
            calls.append(("open", destination))
            return _open_fake_staged_shard(
                destination,
                manifest_update=reread_manifest_update,
            )

    class FakeJax:
        @staticmethod
        def clear_caches() -> None:
            calls.append("clear_caches")

    runtime = SimpleNamespace(
        admission=FakeAdmission,
        transitions=FakeTransitions,
        feature_extractor=fake_feature_extractor,
        cache=FakeCache,
        jax=FakeJax,
    )
    return runtime, calls, lambda: None if model_ref is None else model_ref()


def test_config_requires_explicit_checkpoint_batch_round_and_training_root(tmp_path: Path):
    config = rlt_stage2_build_cache.BuildCacheConfig(
        checkpoint=tmp_path / "missing",
        batch=tmp_path / "batch",
        training_root=tmp_path / "training",
        round_id="round_000001",
    )

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        config.validate()


@pytest.mark.parametrize("field", ["checkpoint", "batch", "training_root"])
def test_config_requires_absolute_canonical_paths(tmp_path: Path, field: str):
    config = _config(tmp_path)
    values = dataclasses.asdict(config)
    values[field] = Path("relative")

    with pytest.raises(ValueError, match=field):
        rlt_stage2_build_cache.BuildCacheConfig(**values).validate()


def test_config_rejects_symlinked_input_and_training_nested_in_inputs(tmp_path: Path):
    config = _config(tmp_path)
    link = tmp_path / "batch-link"
    link.symlink_to(config.batch, target_is_directory=True)
    linked = dataclasses.replace(config, batch=link)

    with pytest.raises(ValueError, match="batch.*symlink|canonical"):
        linked.validate()

    nested = config.checkpoint / "training"
    nested.mkdir()
    with pytest.raises(ValueError, match="training_root.*nested|overlap"):
        dataclasses.replace(config, training_root=nested).validate()


@pytest.mark.parametrize("linked_component", ["params", "assets"])
def test_load_rejects_checkpoint_tree_ancestor_symlink(
    tmp_path: Path,
    linked_component: str,
):
    config = _config(tmp_path)
    outside = tmp_path / f"outside-{linked_component}"
    outside.mkdir()
    if linked_component == "params":
        (outside / "checkpoint.bin").write_bytes(b"outside")
        (config.checkpoint / "params").symlink_to(outside, target_is_directory=True)
        real_assets = config.checkpoint / "assets" / "asset"
        real_assets.mkdir(parents=True)
        (real_assets / "norm.json").write_bytes(b"norm")
    else:
        norm = outside / "asset"
        norm.mkdir()
        (norm / "norm.json").write_bytes(b"outside")
        (config.checkpoint / "assets").symlink_to(outside, target_is_directory=True)
        params = config.checkpoint / "params"
        params.mkdir()
        (params / "checkpoint.bin").write_bytes(b"params")

    with pytest.raises(RuntimeError, match="symlink|real directory"):
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=_train_config_only_runtime([]),
            code_commit=_COMMIT,
        )

    assert list(config.training_root.glob(".rlt-stage2-snapshot-*")) == []


@pytest.mark.parametrize(
    "round_id",
    ["round_1", "round_000000", "round_0000001", "../round_000001", 1, "ROUND_000001"],
)
def test_config_requires_exact_round_format(tmp_path: Path, round_id: object):
    config = _config(tmp_path, round_id=round_id)

    with pytest.raises((TypeError, ValueError), match="round_id"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("micro_batch_size", True),
        ("micro_batch_size", 0),
        ("micro_batch_size", 1.0),
        ("sampler_num_steps", False),
        ("sampler_num_steps", -1),
        ("sampler_num_steps", "10"),
        ("num_workers", False),
        ("num_workers", -1),
        ("num_workers", 1),
    ],
)
def test_config_rejects_nonexact_loader_and_sampler_integers(
    tmp_path: Path,
    field: str,
    value: object,
):
    config = _config(tmp_path, **{field: value})

    with pytest.raises((TypeError, ValueError), match=field):
        config.validate()


def test_invalid_worker_count_is_rejected_before_runtime_import_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path, num_workers=1)
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_runtime_imports",
        lambda: (_ for _ in ()).throw(AssertionError("runtime imports must be unreachable")),
    )

    with pytest.raises(ValueError, match="num_workers.*zero"):
        rlt_stage2_build_cache.run(config)


def test_invalid_ready_batch_creates_no_training_output_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    config.training_root.rmdir()
    runtime, _, _ = _run_runtime(config, monkeypatch)
    runtime.admission.validate_ready_batch = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("invalid ready batch")
    )

    with pytest.raises(ValueError, match="invalid ready batch"):
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert not config.training_root.exists()


def test_current_git_commit_requires_full_lowercase_sha1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        rlt_stage2_build_cache.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="A" * 40 + "\n"),
    )

    with pytest.raises(RuntimeError, match="40.*lowercase|commit"):
        rlt_stage2_build_cache.current_git_commit()


def test_current_git_commit_fails_closed_for_dirty_worktree(monkeypatch: pytest.MonkeyPatch):
    results = iter(
        [
            SimpleNamespace(stdout=_COMMIT + "\n"),
            SimpleNamespace(stdout="?? untracked.py\n"),
        ]
    )
    monkeypatch.setattr(
        rlt_stage2_build_cache.subprocess,
        "run",
        lambda *args, **kwargs: next(results),
    )

    with pytest.raises(RuntimeError, match="dirty|clean"):
        rlt_stage2_build_cache.current_git_commit()


def test_production_runtime_module_provenance_rejects_import_outside_worktree(
    tmp_path: Path,
):
    outside = tmp_path / "shadow_openpi.py"
    outside.write_text("# shadow module\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside.*worktree|provenance"):
        rlt_stage2_build_cache._verify_runtime_module_provenance(  # noqa: SLF001
            {"openpi.shadow": SimpleNamespace(__file__=str(outside))}
        )


def test_run_rejects_output_subdirectory_symlink_before_external_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (config.training_root / "admissions").symlink_to(outside, target_is_directory=True)
    runtime, _, _ = _run_runtime(config, monkeypatch)
    marker = outside / "escaped.json"

    def escaped_publish(*args: Any, **kwargs: Any) -> Path:
        marker.write_text("escaped", encoding="utf-8")
        return config.training_root / "admissions" / "round_000001.json"

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_publish_or_verify_round",
        escaped_publish,
    )

    with pytest.raises(RuntimeError, match="symlink|directory"):
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert not marker.exists()


def test_run_rejects_feature_cache_symlink_with_zero_outside_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    (config.training_root / "feature_cache").symlink_to(
        outside,
        target_is_directory=True,
    )
    runtime, calls, _ = _run_runtime(config, monkeypatch)

    with pytest.raises(RuntimeError, match="symlink|real directory"):
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert list(outside.iterdir()) == []
    assert not any(isinstance(call, tuple) and call[0] == "stage" for call in calls)


def test_run_rejects_feature_identity_parent_symlink_before_cache_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    outside = tmp_path / "outside-feature"
    outside.mkdir()
    feature_cache = config.training_root / "feature_cache"
    feature_cache.mkdir()
    (feature_cache / _SHA_D).symlink_to(outside, target_is_directory=True)
    runtime, calls, _ = _run_runtime(config, monkeypatch)

    with pytest.raises(RuntimeError, match="symlink|real directory"):
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert list(outside.iterdir()) == []
    assert not any(isinstance(call, tuple) and call[0] == "stage" for call in calls)


def test_cli_help_needs_no_jax_lerobot_or_huggingface_import(tmp_path: Path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    for module in ("jax", "lerobot", "huggingface_hub"):
        (blocked / f"{module}.py").write_text(
            f"raise RuntimeError('{module} import forbidden for --help')\n",
            encoding="utf-8",
        )
    script = Path(rlt_stage2_build_cache.__file__)
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(blocked),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint" in completed.stdout
    assert "--batch" in completed.stdout
    assert "--training-root" in completed.stdout
    assert "--round-id" in completed.stdout


def test_real_train_config_can_be_rebound_to_checkpoint_assets_without_mutation(
    tmp_path: Path,
):
    runtime = rlt_stage2_build_cache._runtime_imports()  # noqa: SLF001
    original = runtime.training_config.get_stage1_config("rl_token_stage1")
    checkpoint_assets = tmp_path / "checkpoint" / "assets"

    patched = rlt_stage2_build_cache._train_config_for_checkpoint_assets(  # noqa: SLF001
        original,
        checkpoint_assets,
    )

    assert patched is not original
    assert patched.data is not original.data
    assert patched.data.assets.assets_dir == str(checkpoint_assets)
    assert original.data.assets.assets_dir != str(checkpoint_assets)
    assert rlt_stage2_build_cache._asset_id(patched) == (  # noqa: SLF001
        "lite0030_joints_fps20_openpi_drop_last4s_min20s"
    )
    assert patched.model.rl_token_enabled is True
    signature = runtime.feature_identity.transform_signature(patched.model)
    assert signature["fields"]["rl_token_width"] == 2048


def test_load_model_keeps_disk_and_loaded_parameter_identities_separate(
    tmp_path: Path,
):
    config = _config(tmp_path)
    runtime, calls, actual_data_config, actual_transform = _load_runtime(config)
    batch = _batch(config)

    loaded = rlt_stage2_build_cache.load_model_and_transforms(
        config,
        batch,
        runtime=runtime,
        code_commit=_COMMIT,
    )

    assert loaded.checkpoint_sha256 == _SHA_A
    assert loaded.norm_stats_sha256 == _SHA_B
    assert len(loaded.loaded_parameter_sha256) == 64
    assert rlt_stage2_build_cache._require_sha256(  # noqa: SLF001
        loaded.loaded_norm_stats_sha256,
        name="loaded norm stats",
    )
    assert loaded.feature_id == _SHA_D
    assert loaded.data_config is actual_data_config
    assert loaded.input_transform is actual_transform
    restore = next(call for call in calls if call[0] == "restore_params")
    assert restore[1] != config.checkpoint / "params"
    assert restore[1].parent == Path("/proc/self/fd")
    assert not restore[1].exists()
    assert restore[2:] == ((), {})
    norm_load = next(call for call in calls if call[0] == "load_norm_stats")
    assert norm_load[1] != config.checkpoint / "assets"
    assert norm_load[1].parent == Path("/proc/self/fd")
    assert norm_load[2] == "asset"
    identity_call = next(call for call in calls if call[0] == "build_feature_identity")
    identity_input = identity_call[1]
    assert identity_input.checkpoint_sha256 == _SHA_A
    assert identity_input.norm_stats_sha256 == _SHA_B
    assert identity_input.model_config["loaded_parameter_sha256"] == loaded.loaded_parameter_sha256
    assert identity_input.transform_config["loaded_norm_stats_sha256"] == loaded.loaded_norm_stats_sha256
    assert identity_input.code_commit == _COMMIT
    assert identity_input.sampler_num_steps == config.sampler_num_steps
    assert calls.index(identity_call) > max(index for index, call in enumerate(calls) if call[0] == "disk_hash")
    build_call = next(call for call in calls if call[0] == "build_input_transform")
    transform_signature_calls = [call for call in calls if call[0] == "transform_signature"]
    assert transform_signature_calls[-1][1]["default_prompt"] == "fold clothes"
    built_assets = Path(build_call[1].data.assets.assets_dir)
    assert built_assets.parent == Path("/proc/self/fd")


def test_source_swap_to_b_and_back_cannot_change_private_snapshot_load(
    tmp_path: Path,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)
    source = config.checkpoint / "params"
    parked = config.checkpoint / "params-a"
    observed: list[bytes] = []

    def restore_during_source_swap(snapshot_params: Path):
        source.rename(parked)
        source.mkdir()
        (source / "checkpoint.bin").write_bytes(b"attacker-b")
        try:
            observed.append((Path(snapshot_params) / "checkpoint.bin").read_bytes())
        finally:
            (source / "checkpoint.bin").unlink()
            source.rmdir()
            parked.rename(source)
        return {"weights": "fp32"}

    runtime.model_api.restore_params = restore_during_source_swap

    loaded = rlt_stage2_build_cache.load_model_and_transforms(
        config,
        _batch(config),
        runtime=runtime,
        code_commit=_COMMIT,
    )

    assert observed == [b"frozen-params"]
    assert (source / "checkpoint.bin").read_bytes() == b"frozen-params"
    assert len(loaded.loaded_parameter_sha256) == 64
    assert list(config.training_root.glob(".rlt-stage2-snapshot-*")) == []


def test_snapshot_file_swap_to_b_and_back_during_restore_is_rejected(
    tmp_path: Path,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)

    def restore_during_snapshot_swap(snapshot_params: Path):
        checkpoint_file = Path(snapshot_params) / "checkpoint.bin"
        original = checkpoint_file.read_bytes()
        checkpoint_file.write_bytes(b"attacker-b")
        try:
            return {"weights": "loaded-from-b"}
        finally:
            checkpoint_file.write_bytes(original)

    runtime.model_api.restore_params = restore_during_snapshot_swap

    with pytest.raises(RuntimeError, match="snapshot.*changed|identity guard"):
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )

    assert list(config.training_root.glob(".rlt-stage2-snapshot-*")) == []


def test_loaded_parameter_validation_and_hash_use_one_tree_traversal(
    tmp_path: Path,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)
    runtime.feature_identity.parameter_tree_sha256 = lambda state: (_ for _ in ()).throw(
        AssertionError("second parameter traversal is forbidden")
    )

    loaded = rlt_stage2_build_cache.load_model_and_transforms(
        config,
        _batch(config),
        runtime=runtime,
        code_commit=_COMMIT,
    )

    assert len(loaded.loaded_parameter_sha256) == 64


def test_source_witness_close_failure_cannot_mask_second_acquisition_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)
    real_open = rlt_stage2_build_cache._open_real_directory_chain  # noqa: SLF001
    primary = RuntimeError("injected source norm acquisition failure")
    cleanup_error = OSError("injected source params close failure")

    def open_with_failures(path: Path, *, create: bool, **kwargs: object):
        if path == config.checkpoint / "assets" / "asset":
            raise primary
        witness = real_open(path, create=create, **kwargs)
        if path == config.checkpoint / "params":
            real_close = witness.close

            def close_then_raise() -> None:
                real_close()
                raise cleanup_error

            witness.close = close_then_raise
        return witness

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_open_real_directory_chain",
        open_with_failures,
    )

    with pytest.raises(RuntimeError, match="source norm acquisition") as exc_info:
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )

    assert exc_info.value is primary
    assert "source params close failure" in "\n".join(getattr(primary, "__notes__", ()))


def test_validated_parameter_hash_streams_chunks_and_matches_public_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    from flax import nnx

    from openpi.training.rl_token.stage2 import feature_identity

    matrix = np.arange(120, dtype=np.float32).reshape(10, 12)[:, ::2]
    scalar = np.asarray(3.0, dtype=np.float32)
    assert matrix.flags.c_contiguous is False
    state = nnx.State(
        {
            "matrix": nnx.Param(matrix),
            "scalar": nnx.Param(scalar),
        }
    )
    expected = feature_identity.parameter_tree_sha256(state)

    device_get_values: list[object] = []
    runtime = SimpleNamespace(
        nnx=SimpleNamespace(
            Param=nnx.Param,
            state=lambda model, selector: state,
        ),
        jax=SimpleNamespace(
            device_get=lambda value: device_get_values.append(value) or value,
        ),
        feature_identity=feature_identity,
    )
    real_asarray = np.asarray
    real_nditer = np.nditer
    iterator_buffer_sizes: list[int] = []

    class _NoWholeLeafTobytes(np.ndarray):
        def tobytes(self, order: str = "C") -> bytes:
            raise AssertionError("whole-leaf ndarray.tobytes() is forbidden")

    def guarded_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        return real_asarray(value, *args, **kwargs).view(_NoWholeLeafTobytes)

    def tracked_nditer(*args: object, **kwargs: object):
        iterator_buffer_sizes.append(kwargs["buffersize"])
        return real_nditer(*args, **kwargs)

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_PARAMETER_HASH_CHUNK_BYTES",
        16,
        raising=False,
    )
    monkeypatch.setattr(rlt_stage2_build_cache.np, "asarray", guarded_asarray)
    monkeypatch.setattr(rlt_stage2_build_cache.np, "nditer", tracked_nditer, raising=False)

    actual = rlt_stage2_build_cache._validated_loaded_parameter_sha256(  # noqa: SLF001
        runtime,
        object(),
    )

    assert actual == expected
    assert len(device_get_values) == 2
    assert {id(value) for value in device_get_values} == {id(matrix), id(scalar)}
    assert iterator_buffer_sizes == [4, 4]


@pytest.mark.parametrize("entry_kind", ["symlink", "fifo"])
def test_private_snapshot_copy_rejects_internal_symlink_and_special_node(
    tmp_path: Path,
    entry_kind: str,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)
    entry = config.checkpoint / "params" / f"bad-{entry_kind}"
    if entry_kind == "symlink":
        entry.symlink_to(config.checkpoint / "params" / "checkpoint.bin")
    else:
        os.mkfifo(entry)

    with pytest.raises((RuntimeError, ValueError), match="regular|directories|symlink"):
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )

    assert list(config.training_root.glob(".rlt-stage2-snapshot-*")) == []


def test_copy_tree_closes_source_child_when_destination_child_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    child = source / "child"
    child.mkdir(parents=True)
    destination.mkdir()
    (child / "value.bin").write_bytes(b"value")
    original_open = os.open
    source_fd = original_open(source, rlt_stage2_build_cache._directory_flags())  # noqa: SLF001
    destination_fd = original_open(
        destination,
        rlt_stage2_build_cache._directory_flags(),  # noqa: SLF001
    )

    def fail_destination_child_open(
        path: str | bytes,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "child" and dir_fd == destination_fd:
            raise OSError("injected destination child open failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        rlt_stage2_build_cache.os,
        "open",
        fail_destination_child_open,
    )
    try:
        with pytest.raises(OSError, match="injected"):
            rlt_stage2_build_cache._copy_tree(  # noqa: SLF001
                source_fd,
                destination_fd,
            )
        leaked: list[int] = []
        for entry in Path("/proc/self/fd").iterdir():
            with contextlib.suppress(OSError, ValueError):
                if Path(os.readlink(entry)) == child:
                    leaked.append(int(entry.name))
        try:
            assert leaked == []
        finally:
            for descriptor in leaked:
                os.close(descriptor)
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def test_pinned_snapshot_tree_owns_file_fd_before_fstat_can_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "value.bin").write_bytes(b"value")
    original_open = os.open
    original_fstat = os.fstat
    original_close = os.close
    root_fd = original_open(root, rlt_stage2_build_cache._directory_flags())  # noqa: SLF001
    opened_file: list[int] = []
    primary = OSError("injected pinned file fstat failure")

    def tracking_open(path: str | bytes | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "value.bin":
            opened_file.append(descriptor)
        return descriptor

    def failing_fstat(descriptor: int):
        if opened_file and descriptor == opened_file[-1]:
            raise primary
        return original_fstat(descriptor)

    monkeypatch.setattr(rlt_stage2_build_cache.os, "open", tracking_open)
    monkeypatch.setattr(rlt_stage2_build_cache.os, "fstat", failing_fstat)
    try:
        with pytest.raises(OSError, match="pinned file fstat") as exc_info:
            rlt_stage2_build_cache._PinnedSnapshotTree.open(root_fd)  # noqa: SLF001
        assert exc_info.value is primary
        assert len(opened_file) == 1
        with pytest.raises(OSError, match="Bad file descriptor"):
            original_fstat(opened_file[0])
    finally:
        for descriptor in opened_file:
            with contextlib.suppress(OSError):
                original_close(descriptor)
        original_close(root_fd)


def test_pinned_snapshot_tree_close_preserves_primary_and_attempts_every_fd(
    monkeypatch: pytest.MonkeyPatch,
):
    pipes = [os.pipe() for _ in range(2)]
    descriptors = [read_descriptor for read_descriptor, _ in pipes]
    for _, write_descriptor in pipes:
        os.close(write_descriptor)
    original_close = os.close
    original_fstat = os.fstat
    cleanup_errors = {
        descriptors[0]: OSError("first injected pinned close failure"),
        descriptors[1]: OSError("second injected pinned close failure"),
    }
    tree = rlt_stage2_build_cache._PinnedSnapshotTree(  # noqa: SLF001
        directories=[],
        files=[
            rlt_stage2_build_cache._PinnedSnapshotFile(  # noqa: SLF001
                relative=f"file-{index}",
                descriptor=descriptor,
                metadata=(0, 0, 0, 0, 0, 0, 0),
                sha256=None,
            )
            for index, descriptor in enumerate(descriptors)
        ],
    )
    primary = RuntimeError("injected pinned verification failure")

    def close_then_raise(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor in cleanup_errors:
            raise cleanup_errors[descriptor]

    monkeypatch.setattr(rlt_stage2_build_cache.os, "close", close_then_raise)

    def fail_then_close() -> None:
        try:
            raise primary
        except RuntimeError:
            tree.close()
            raise

    try:
        with pytest.raises(RuntimeError, match="pinned verification") as exc_info:
            fail_then_close()
        assert exc_info.value is primary
        notes = "\n".join(getattr(primary, "__notes__", ()))
        assert "first injected pinned close failure" in notes
        assert "second injected pinned close failure" in notes
        for descriptor in descriptors:
            with pytest.raises(OSError, match="Bad file descriptor"):
                original_fstat(descriptor)
    finally:
        for descriptor in descriptors:
            with contextlib.suppress(OSError):
                original_close(descriptor)


def test_copy_tree_close_failures_do_not_mask_primary_and_close_both_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "child").mkdir(parents=True)
    destination.mkdir()
    original_open = os.open
    original_close = os.close
    original_fstat = os.fstat
    original_copy_tree = rlt_stage2_build_cache._copy_tree  # noqa: SLF001
    source_fd = original_open(source, rlt_stage2_build_cache._directory_flags())  # noqa: SLF001
    destination_fd = original_open(destination, rlt_stage2_build_cache._directory_flags())  # noqa: SLF001
    child_fds: list[int] = []
    primary = RuntimeError("injected recursive copy failure")

    def tracking_open(path: str | bytes | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "child":
            child_fds.append(descriptor)
        return descriptor

    def recursive_failure(*args: object, relative: Path = Path(), **kwargs: object) -> int:
        if relative == Path("child"):
            raise primary
        return original_copy_tree(*args, relative=relative, **kwargs)

    def close_then_raise(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor in child_fds:
            raise OSError(f"injected child close failure {descriptor}")

    monkeypatch.setattr(rlt_stage2_build_cache.os, "open", tracking_open)
    monkeypatch.setattr(rlt_stage2_build_cache.os, "close", close_then_raise)
    monkeypatch.setattr(rlt_stage2_build_cache, "_copy_tree", recursive_failure)
    try:
        with pytest.raises(RuntimeError, match="recursive copy") as exc_info:
            recursive_failure(source_fd, destination_fd)
        assert exc_info.value is primary
        assert len(child_fds) == 2
        notes = "\n".join(getattr(primary, "__notes__", ()))
        assert notes.count("injected child close failure") == 2
        for descriptor in child_fds:
            with pytest.raises(OSError, match="Bad file descriptor"):
                original_fstat(descriptor)
    finally:
        for descriptor in child_fds:
            with contextlib.suppress(OSError):
                original_close(descriptor)
        original_close(destination_fd)
        original_close(source_fd)


def test_prepare_output_roots_close_failures_preserve_primary_and_close_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    primary = RuntimeError("injected output root verification failure")

    class FailingWitness:
        def __init__(self, path: Path, *, verify_error: BaseException | None = None):
            self.path = path
            self.verify_error = verify_error
            self.closed = False

        def verify(self) -> None:
            if self.verify_error is not None:
                raise self.verify_error

        def close(self) -> None:
            self.closed = True
            raise OSError(f"injected {self.path.name} close failure")

    training = FailingWitness(config.training_root)
    admissions = FailingWitness(config.training_root / "admissions")
    feature_cache = FailingWitness(
        config.training_root / "feature_cache",
        verify_error=primary,
    )
    children = iter((admissions, feature_cache))
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_open_real_directory_chain",
        lambda *args, **kwargs: training,
    )
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_open_child_directory",
        lambda *args, **kwargs: next(children),
    )

    with pytest.raises(RuntimeError, match="output root verification") as exc_info:
        rlt_stage2_build_cache._prepare_output_roots(config)  # noqa: SLF001
    assert exc_info.value is primary
    assert feature_cache.closed
    assert admissions.closed
    assert training.closed
    notes = "\n".join(getattr(primary, "__notes__", ()))
    assert notes.count("close failure") == 3


def test_snapshot_creation_preserves_cleanup_failure_as_primary_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)
    del runtime
    primary = RuntimeError("snapshot copy failed")
    original_close = rlt_stage2_build_cache._PrivateSnapshot.close  # noqa: SLF001

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_copy_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(primary),
    )

    def close_then_raise(snapshot: object) -> None:
        original_close(snapshot)
        raise RuntimeError("snapshot cleanup failed")

    monkeypatch.setattr(
        rlt_stage2_build_cache._PrivateSnapshot,  # noqa: SLF001
        "close",
        close_then_raise,
    )

    with (
        rlt_stage2_build_cache._open_real_directory_chain(  # noqa: SLF001
            config.training_root,
            create=False,
        ) as training,
        rlt_stage2_build_cache._open_real_directory_chain(  # noqa: SLF001
            config.checkpoint / "params",
            create=False,
        ) as params,
        rlt_stage2_build_cache._open_real_directory_chain(  # noqa: SLF001
            config.checkpoint / "assets" / "asset",
            create=False,
        ) as norm,
        pytest.raises(RuntimeError, match="snapshot copy failed") as exc_info,
    ):
        rlt_stage2_build_cache._create_private_snapshot(  # noqa: SLF001
            training,
            source_params=params,
            source_norm=norm,
            asset_id="asset",
        )

    assert exc_info.value is primary
    assert any("snapshot cleanup failed" in note for note in getattr(exc_info.value, "__notes__", ()))
    assert list(config.training_root.glob(".rlt-stage2-snapshot-*")) == []


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.ones((2, 2), dtype=np.float16), "float32"),
        (np.asarray([[np.nan]], dtype=np.float32), "finite"),
        (np.asarray([[np.inf]], dtype=np.float32), "finite"),
    ],
)
def test_load_rejects_non_fp32_or_nonfinite_master_params(
    tmp_path: Path,
    value: np.ndarray,
    match: str,
):
    config = _config(tmp_path)
    runtime, _, _, _ = _load_runtime(config)

    class State(_FakeState):
        def __init__(self):
            self.variable = SimpleNamespace(value=value)

    runtime.nnx = SimpleNamespace(
        Param=_FakeNNX.Param,
        state=lambda model, selector: State(),
    )

    with pytest.raises((TypeError, ValueError), match=match):
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )
    assert list(config.training_root.glob(".rlt-stage2-snapshot-*")) == []


def test_loaded_norm_semantic_hash_binds_fields_dtype_shape_value_and_finite():
    base = {
        "state": _FakeNormStats(
            mean=np.asarray([0.0, 1.0], dtype=np.float32),
            std=np.asarray([1.0, 2.0], dtype=np.float32),
        )
    }
    changed_value = {
        "state": dataclasses.replace(
            base["state"],
            mean=np.asarray([0.0, 2.0], dtype=np.float32),
        )
    }
    changed_dtype = {
        "state": dataclasses.replace(
            base["state"],
            mean=np.asarray([0.0, 1.0], dtype=np.float64),
        )
    }

    base_sha = rlt_stage2_build_cache._norm_stats_semantic_sha256(base)  # noqa: SLF001
    assert base_sha != rlt_stage2_build_cache._norm_stats_semantic_sha256(  # noqa: SLF001
        changed_value
    )
    assert base_sha != rlt_stage2_build_cache._norm_stats_semantic_sha256(  # noqa: SLF001
        changed_dtype
    )
    with pytest.raises(ValueError, match="finite"):
        rlt_stage2_build_cache._norm_stats_semantic_sha256(  # noqa: SLF001
            {
                "state": dataclasses.replace(
                    base["state"],
                    mean=np.asarray([np.nan], dtype=np.float32),
                )
            }
        )


@pytest.mark.parametrize(
    ("checkpoint_hashes", "norm_hashes", "match"),
    [
        ((_SHA_A, _SHA_D), (_SHA_B, _SHA_B), "checkpoint.*changed"),
        ((_SHA_A, _SHA_A), (_SHA_B, _SHA_D), "norm.*changed"),
    ],
)
def test_load_rejects_pre_post_disk_mutation_before_feature_identity(
    tmp_path: Path,
    checkpoint_hashes: tuple[str, str],
    norm_hashes: tuple[str, str],
    match: str,
):
    config = _config(tmp_path)
    runtime, calls, _, _ = _load_runtime(
        config,
        checkpoint_hashes=checkpoint_hashes,
        norm_hashes=norm_hashes,
    )

    with pytest.raises(RuntimeError, match=match):
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )

    assert [call for call in calls if call[0] == "build_feature_identity"] == []


def test_disk_mutation_guard_wins_over_restore_error_and_chains_original(tmp_path: Path):
    config = _config(tmp_path)
    original = ValueError("restore failed")
    runtime, _, _, _ = _load_runtime(
        config,
        checkpoint_hashes=(_SHA_A, _SHA_A, _SHA_D),
        restore_error=original,
    )

    with pytest.raises(RuntimeError, match="checkpoint.*changed") as exc_info:
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )

    assert exc_info.value.__cause__ is original


def test_load_rejects_disabled_rl_token_before_restore(tmp_path: Path):
    config = _config(tmp_path)
    runtime, calls, _, _ = _load_runtime(config, rl_token_enabled=False)

    with pytest.raises(ValueError, match="rl_token_enabled"):
        rlt_stage2_build_cache.load_model_and_transforms(
            config,
            _batch(config),
            runtime=runtime,
            code_commit=_COMMIT,
        )

    assert [call for call in calls if call[0] == "restore_params"] == []


def test_existing_admission_payload_cannot_spoof_another_round(
    tmp_path: Path,
):
    config = _config(tmp_path)
    batch = _batch(config)
    admission_path = config.training_root / "admissions" / f"{config.round_id}.json"
    admission_path.parent.mkdir()
    wrong_payload = {
        "round_id": "round_000002",
        "admitted_at": "2026-07-24T00:00:00+00:00",
        "code_commit": "2" * 40,
        "batch_id": batch.batch_id,
    }
    admission_path.write_text(json.dumps(wrong_payload), encoding="utf-8")

    class FakeAdmission:
        @staticmethod
        def admission_payload(
            passed_batch: object,
            *,
            round_id: str,
            admitted_at: str,
            code_commit: str,
        ):
            assert passed_batch is batch
            return {
                "round_id": round_id,
                "admitted_at": admitted_at,
                "code_commit": code_commit,
                "batch_id": batch.batch_id,
            }

        @staticmethod
        def verify_admission(path: Path, passed_batch: object) -> None:
            assert (path, passed_batch) == (admission_path, batch)

        @staticmethod
        def publish_admission(*args: Any, **kwargs: Any) -> Path:
            raise AssertionError("existing admission must not be overwritten")

    runtime = SimpleNamespace(admission=FakeAdmission)

    with pytest.raises(RuntimeError, match="round_id|round"):
        rlt_stage2_build_cache._publish_or_verify_round(  # noqa: SLF001
            config,
            batch,
            code_commit=_COMMIT,
            runtime=runtime,
        )


def test_new_admission_is_published_then_reread_and_strictly_verified(
    tmp_path: Path,
):
    config = _config(tmp_path)
    batch = _batch(config)
    verified: list[object] = []

    class FakeAdmission:
        @staticmethod
        def admission_payload(
            passed_batch: object,
            *,
            round_id: str,
            admitted_at: str,
            code_commit: str,
        ):
            assert passed_batch is batch
            return {
                "round_id": round_id,
                "admitted_at": admitted_at,
                "code_commit": code_commit,
                "batch_id": batch.batch_id,
            }

        @classmethod
        def publish_admission(
            cls,
            passed_batch: object,
            training_root: Path,
            *,
            round_id: str,
            admitted_at: str,
            code_commit: str,
        ) -> Path:
            path = training_root / "admissions" / f"{round_id}.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    cls.admission_payload(
                        passed_batch,
                        round_id=round_id,
                        admitted_at=admitted_at,
                        code_commit=code_commit,
                    )
                ),
                encoding="utf-8",
            )
            return path

        @staticmethod
        def verify_admission(path: Path, passed_batch: object) -> None:
            verified.append((path, passed_batch))

    result = rlt_stage2_build_cache._publish_or_verify_round(  # noqa: SLF001
        config,
        batch,
        code_commit=_COMMIT,
        runtime=SimpleNamespace(admission=FakeAdmission),
    )

    assert result == config.training_root / "admissions/round_000001.json"
    payload = json.loads(result.read_text())
    assert payload["round_id"] == config.round_id
    assert payload["code_commit"] == _COMMIT
    assert verified == [(result, batch)]


def test_admission_publish_is_bound_to_held_parent_during_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    batch = _batch(config)
    admissions = config.training_root / "admissions"
    admissions.mkdir()
    parked = config.training_root / "admissions-parked"
    outside = tmp_path / "outside-admissions"
    outside.mkdir()
    swapped = False

    def enter_swap(*args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        assert not swapped
        admissions.rename(parked)
        admissions.symlink_to(outside, target_is_directory=True)
        swapped = True

    def leave_swap(*args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        assert swapped
        admissions.unlink()
        parked.rename(admissions)
        swapped = False

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_before_bound_admission_publish_hook",
        enter_swap,
        raising=False,
    )
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_after_bound_admission_publish_hook",
        leave_swap,
        raising=False,
    )

    class FakeAdmission:
        @staticmethod
        def admission_payload(
            passed_batch: object,
            *,
            round_id: str,
            admitted_at: str,
            code_commit: str,
        ) -> dict[str, object]:
            assert passed_batch is batch
            return {
                "round_id": round_id,
                "admitted_at": admitted_at,
                "code_commit": code_commit,
                "batch_id": batch.batch_id,
            }

        @classmethod
        def publish_admission(
            cls,
            passed_batch: object,
            training_root: Path,
            *,
            round_id: str,
            admitted_at: str,
            code_commit: str,
        ) -> Path:
            enter_swap()
            try:
                destination = training_root / "admissions" / f"{round_id}.json"
                destination.write_text(
                    json.dumps(
                        cls.admission_payload(
                            passed_batch,
                            round_id=round_id,
                            admitted_at=admitted_at,
                            code_commit=code_commit,
                        )
                    ),
                    encoding="utf-8",
                )
                return destination
            finally:
                leave_swap()

        @staticmethod
        def verify_admission(path: Path, passed_batch: object) -> None:
            assert passed_batch is batch

    with rlt_stage2_build_cache._open_real_directory_chain(  # noqa: SLF001
        admissions,
        create=False,
    ) as witness:
        try:
            result = rlt_stage2_build_cache._publish_or_verify_round(  # noqa: SLF001
                config,
                batch,
                code_commit=_COMMIT,
                runtime=SimpleNamespace(admission=FakeAdmission),
                admissions_witness=witness,
            )
        except RuntimeError:
            result = None

    assert not swapped
    assert list(outside.iterdir()) == []
    if result is not None:
        assert result == admissions / f"{config.round_id}.json"
        assert result.is_file()


def test_existing_admission_preserves_original_code_commit(
    tmp_path: Path,
):
    config = _config(tmp_path)
    batch = _batch(config)
    admission_path = config.training_root / "admissions" / f"{config.round_id}.json"
    admission_path.parent.mkdir()
    original_commit = "2" * 40
    payload = {
        "round_id": config.round_id,
        "admitted_at": "2026-07-24T00:00:00+00:00",
        "code_commit": original_commit,
        "batch_id": batch.batch_id,
    }
    admission_path.write_text(json.dumps(payload), encoding="utf-8")
    verified: list[object] = []

    class FakeAdmission:
        @staticmethod
        def admission_payload(
            passed_batch: object,
            *,
            round_id: str,
            admitted_at: str,
            code_commit: str,
        ):
            assert passed_batch is batch
            return {
                "round_id": round_id,
                "admitted_at": admitted_at,
                "code_commit": code_commit,
                "batch_id": batch.batch_id,
            }

        @staticmethod
        def verify_admission(path: Path, passed_batch: object) -> None:
            verified.append((path, passed_batch))

        @staticmethod
        def publish_admission(*args: Any, **kwargs: Any) -> Path:
            raise AssertionError("existing admission must not be overwritten")

    result = rlt_stage2_build_cache._publish_or_verify_round(  # noqa: SLF001
        config,
        batch,
        code_commit=_COMMIT,
        runtime=SimpleNamespace(admission=FakeAdmission),
    )

    assert result == admission_path
    assert verified == [(admission_path, batch)]
    assert json.loads(admission_path.read_text())["code_commit"] == original_commit


def test_cache_destination_component_is_validated_before_parent_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    acquisitions: list[Path] = []

    def record_acquisition(path: Path, **unused: object) -> object:
        acquisitions.append(path)
        return object()

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_open_real_directory_chain",
        record_acquisition,
    )

    with pytest.raises(ValueError, match="safe path component"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            SimpleNamespace(),
            tmp_path / "bad\\batch",
            features=object(),
            transitions=object(),
            identity_fields={},
        )

    assert acquisitions == []


def test_post_rename_parent_fsync_failure_requires_successful_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    destination.parent.mkdir(parents=True)
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transitions = _Transitions(np.asarray([3, 4], dtype=np.int32))
    identity_fields = {"feature_identity": _SHA_D, "batch_id": "batch"}
    calls: list[str] = []
    parent_fsync_attempts = 0
    remaining_failures = 1
    original_fsync = os.fsync

    class Cache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)
        _validate_identity_fields = staticmethod(_validate_fake_identity)

        @staticmethod
        def open_shard(path: Path):
            calls.append("open")
            return _open_fake_staged_shard(path)

    def flaky_parent_fsync(descriptor: int) -> None:
        nonlocal parent_fsync_attempts, remaining_failures
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            target = None
        if target == destination.parent and destination.exists():
            parent_fsync_attempts += 1
            if remaining_failures:
                remaining_failures -= 1
                raise OSError("parent fsync failed after rename")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", flaky_parent_fsync)

    runtime = SimpleNamespace(cache=Cache)
    _, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()
    assert parent_fsync_attempts == 2

    parent_fsync_attempts = 0
    _, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()

    assert parent_fsync_attempts == 1
    assert destination.is_dir()


def test_parent_fsync_retry_precedes_staging_cleanup_failure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    destination.parent.mkdir(parents=True)
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transitions = _Transitions(np.asarray([3, 4], dtype=np.int32))
    identity_fields = {"feature_identity": _SHA_D, "batch_id": "batch"}
    original_fsync = os.fsync
    original_close = rlt_stage2_build_cache._StagedCache.close  # noqa: SLF001
    parent_fsync_attempts = 0
    remaining_failures = 1
    cleanup_error = OSError("injected staged cache close failure")

    class Cache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)
        _validate_identity_fields = staticmethod(_validate_fake_identity)
        open_shard = staticmethod(_open_fake_staged_shard)

    def flaky_parent_fsync(descriptor: int) -> None:
        nonlocal parent_fsync_attempts, remaining_failures
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            target = None
        if target == destination.parent and destination.exists():
            parent_fsync_attempts += 1
            if remaining_failures:
                remaining_failures -= 1
                raise OSError("parent fsync failed after rename")
        original_fsync(descriptor)

    def close_then_raise(staged: object) -> None:
        was_published = staged.published
        original_close(staged)
        if was_published:
            raise cleanup_error

    monkeypatch.setattr(os, "fsync", flaky_parent_fsync)
    monkeypatch.setattr(
        rlt_stage2_build_cache._StagedCache,  # noqa: SLF001
        "close",
        close_then_raise,
    )

    with pytest.raises(OSError, match="parent fsync failed") as exc_info:
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            SimpleNamespace(cache=Cache),
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert parent_fsync_attempts == 2
    assert destination.is_dir()
    assert "staged cache close failure" in "\n".join(getattr(exc_info.value, "__notes__", ()))


def test_persistent_parent_fsync_failure_reports_error_but_next_run_can_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    destination.parent.mkdir(parents=True)
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transitions = _Transitions(np.asarray([3, 4], dtype=np.int32))
    identity_fields = {"feature_identity": _SHA_D, "batch_id": "batch"}
    original_fsync = os.fsync
    fail_parent = True

    class Cache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)
        _validate_identity_fields = staticmethod(_validate_fake_identity)
        open_shard = staticmethod(_open_fake_staged_shard)

    def persistent_parent_failure(descriptor: int) -> None:
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            target = None
        if fail_parent and target == destination.parent and destination.exists():
            raise OSError("persistent parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", persistent_parent_failure)
    runtime = SimpleNamespace(cache=Cache)

    with pytest.raises(OSError, match="parent fsync"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            runtime,
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert destination.is_dir()
    fail_parent = False
    _, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()
    assert destination.is_dir()


def test_real_cache_publish_reuses_only_exact_array_content(
    tmp_path: Path,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()

    opened, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()
    manifest_bytes = (destination / "manifest.json").read_bytes()
    destination_identity = (destination.stat().st_dev, destination.stat().st_ino)
    expected_manifest_sha256 = opened.manifest_sha256
    del opened

    reopened, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()
    assert reopened.manifest_sha256 == expected_manifest_sha256
    del reopened

    changed_features = dataclasses.replace(
        features,
        state_norm=np.ones_like(features.state_norm),
    )
    with pytest.raises(RuntimeError, match="content|manifest"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            runtime,
            destination,
            features=changed_features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert (destination.stat().st_dev, destination.stat().st_ino) == destination_identity
    assert (destination / "manifest.json").read_bytes() == manifest_bytes
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_corrupted_staging_fails_real_open_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()

    def corrupt_before_validation(staging: Path) -> None:
        target = staging / "features" / "state_norm.npy"
        descriptor = os.open(target, os.O_RDWR)
        try:
            original = os.pread(descriptor, 1, target.stat().st_size - 1)
            os.pwrite(descriptor, bytes([original[0] ^ 0xFF]), target.stat().st_size - 1)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_before_staged_cache_validation_hook",
        corrupt_before_validation,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="staging|sha256|cache"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            runtime,
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert not destination.exists()
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_staging_mutation_after_real_open_is_rejected_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()

    def corrupt_after_validation(unused_destination: Path) -> None:
        del unused_destination
        (staging,) = destination.parent.glob(".batch.stage-*")
        target = staging / "features" / "state_norm.npy"
        descriptor = os.open(target, os.O_RDWR)
        try:
            offset = target.stat().st_size - 1
            original = os.pread(descriptor, 1, offset)
            os.pwrite(descriptor, bytes([original[0] ^ 0xFF]), offset)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_before_bound_cache_publish_hook",
        corrupt_after_validation,
    )

    with pytest.raises(RuntimeError, match="staging|payload|changed|mutation|sha256"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            runtime,
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert not destination.exists()
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_staging_open_memmaps_are_closed_before_publish_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()
    real_open_shard = runtime.cache.open_shard
    staging_memory_maps: list[object] = []

    def tracking_open_shard(root: Path):
        opened = real_open_shard(root)
        if ".stage-" in Path(root).name:
            for table in (opened.features, opened.transitions):
                for field in dataclasses.fields(table):
                    memory_map = getattr(getattr(table, field.name), "_mmap", None)
                    if memory_map is not None:
                        staging_memory_maps.append(memory_map)
        return opened

    def assert_staging_maps_closed(unused_destination: Path) -> None:
        del unused_destination
        assert staging_memory_maps
        assert all(memory_map.closed for memory_map in staging_memory_maps)

    monkeypatch.setattr(runtime.cache, "open_shard", tracking_open_shard)
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_before_bound_cache_publish_hook",
        assert_staging_maps_closed,
    )

    _, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()


def test_staging_witness_close_failure_cannot_be_swallowed_by_exact_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()
    real_validate = rlt_stage2_build_cache._validate_staged_cache  # noqa: SLF001
    cleanup_error = OSError("injected staging witness close failure")

    class CloseFailureWitness:
        def __init__(self, witness: object):
            self.witness = witness

        def verify(self) -> None:
            self.witness.verify()

        def close(self) -> None:
            self.witness.close()
            raise cleanup_error

    def validate_with_close_failure(runtime: object, staged: object):
        return CloseFailureWitness(real_validate(runtime, staged))

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_validate_staged_cache",
        validate_with_close_failure,
    )

    returned: tuple[object, object] | None = None
    try:
        with pytest.raises(OSError, match="staging witness close") as exc_info:
            returned = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
                runtime,
                destination,
                features=features,
                transitions=transitions,
                identity_fields=identity_fields,
            )
        assert exc_info.value is cleanup_error
    finally:
        if returned is not None:
            rlt_stage2_build_cache._close_opened_cache_shard(returned[0])  # noqa: SLF001
            returned[1].close()

    assert destination.is_dir()
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_owned_parent_close_failure_closes_final_opened_shard_and_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()
    real_open_directory = rlt_stage2_build_cache._open_real_directory_chain  # noqa: SLF001
    real_open_verified = rlt_stage2_build_cache._open_verified_cache  # noqa: SLF001
    cleanup_error = OSError("injected owned parent close failure")
    final_resources: list[tuple[object, object]] = []
    owned_parent_captured = False

    def open_with_failing_owned_parent(path: Path, *, create: bool, **kwargs: object):
        nonlocal owned_parent_captured
        witness = real_open_directory(path, create=create, **kwargs)
        if path == destination.parent and create and not owned_parent_captured:
            owned_parent_captured = True
            real_close = witness.close

            def close_then_raise() -> None:
                real_close()
                raise cleanup_error

            witness.close = close_then_raise
        return witness

    def track_final_resources(*args: object, **kwargs: object):
        result = real_open_verified(*args, **kwargs)
        final_resources.append(result)
        return result

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_open_real_directory_chain",
        open_with_failing_owned_parent,
    )
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_open_verified_cache",
        track_final_resources,
    )

    try:
        with pytest.raises(OSError, match="owned parent close failure") as exc_info:
            rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
                runtime,
                destination,
                features=features,
                transitions=transitions,
                identity_fields=identity_fields,
            )
        assert exc_info.value is cleanup_error
        assert len(final_resources) == 1
        opened, final_witness = final_resources[0]
        assert final_witness._closed is True  # noqa: SLF001
        for table in (opened.features, opened.transitions):
            for field in dataclasses.fields(table):
                memory_map = getattr(getattr(table, field.name), "_mmap", None)
                if memory_map is not None:
                    assert memory_map.closed
    finally:
        for opened, final_witness in final_resources:
            with contextlib.suppress(BaseException):
                rlt_stage2_build_cache._close_opened_cache_shard(opened)  # noqa: SLF001
            with contextlib.suppress(BaseException):
                final_witness.close()


@pytest.mark.parametrize("mutation_mode", ["leave_changed", "restore_original"])
def test_final_payload_mutation_during_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_mode: str,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    runtime, features, transitions, identity_fields = _real_cache_inputs()
    _, witness = rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
        runtime,
        destination,
        features=features,
        transitions=transitions,
        identity_fields=identity_fields,
    )
    witness.close()

    def mutate_payload(path: Path) -> None:
        target = path / "features" / "state_norm.npy"
        descriptor = os.open(target, os.O_RDWR)
        try:
            offset = target.stat().st_size - 1
            original = os.pread(descriptor, 1, offset)
            os.pwrite(descriptor, bytes([original[0] ^ 0xFF]), offset)
            if mutation_mode == "restore_original":
                os.pwrite(descriptor, original, offset)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_after_cache_open_hook",
        mutate_payload,
    )

    with pytest.raises(RuntimeError, match="payload|changed|mutation|sha256"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            runtime,
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert destination.is_dir()
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_existing_cache_identity_mismatch_is_never_published_deleted_or_overwritten(
    tmp_path: Path,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("keep", encoding="utf-8")
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transitions = _Transitions(np.asarray([3, 4], dtype=np.int32))
    identity_fields = {"feature_identity": _SHA_D, "batch_id": "batch"}

    class Cache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)
        _validate_identity_fields = staticmethod(_validate_fake_identity)

        @staticmethod
        def open_shard(path: Path):
            return SimpleNamespace(
                root=path,
                manifest={
                    "schema_version": 1,
                    **identity_fields,
                    "rogue_identity": "not-allowed",
                    "feature_rows": 2,
                    "transition_rows": 2,
                    "files": [],
                },
                manifest_sha256="9" * 64,
            )

    with pytest.raises(RuntimeError, match="identity|manifest"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            SimpleNamespace(cache=Cache),
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_existing_cache_same_identity_and_rows_but_different_content_is_rejected(
    tmp_path: Path,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("keep", encoding="utf-8")
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transitions = _Transitions(np.asarray([3, 4], dtype=np.int32))
    identity_fields = {"feature_identity": _SHA_D, "batch_id": "batch"}
    existing_manifest = {
        "schema_version": 1,
        **identity_fields,
        "feature_rows": 2,
        "transition_rows": 2,
        "files": [
            {
                "path": "features/value.npy",
                "size": 999,
                "sha256": "0" * 64,
                "shape": [2],
                "dtype": "int32",
            }
        ],
    }

    class Cache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)
        _validate_identity_fields = staticmethod(_validate_fake_identity)

        @staticmethod
        def open_shard(path: Path):
            return SimpleNamespace(
                root=path,
                manifest=dict(existing_manifest),
                manifest_sha256="9" * 64,
            )

    with pytest.raises(RuntimeError, match="content|manifest|files"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            SimpleNamespace(cache=Cache),
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(destination.parent.glob(".batch.stage-*")) == []


def test_cache_manifest_close_failure_is_only_a_note_on_primary_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root_path = tmp_path / "staging"
    root_path.mkdir()
    primary = ValueError("manifest write failed")
    original_open = os.open
    original_close = os.close
    manifest_descriptor: int | None = None
    close_failed = False

    with rlt_stage2_build_cache._open_real_directory_chain(  # noqa: SLF001
        root_path,
        create=False,
    ) as root:

        def record_open(
            path: str | bytes,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal manifest_descriptor
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if path == "manifest.json" and dir_fd == root.descriptor:
                manifest_descriptor = descriptor
            return descriptor

        def close_with_failure(descriptor: int) -> None:
            nonlocal close_failed
            original_close(descriptor)
            if descriptor == manifest_descriptor and not close_failed:
                close_failed = True
                raise OSError("manifest close failed")

        monkeypatch.setattr(os, "open", record_open)
        monkeypatch.setattr(os, "close", close_with_failure)
        monkeypatch.setattr(
            rlt_stage2_build_cache,
            "_write_all",
            lambda *args, **kwargs: (_ for _ in ()).throw(primary),
        )

        with pytest.raises(ValueError, match="manifest write failed") as exc_info:
            rlt_stage2_build_cache._write_cache_manifest_at(  # noqa: SLF001
                root,
                {"schema_version": 1},
            )

    assert exc_info.value is primary
    assert any("manifest close failed" in note for note in getattr(exc_info.value, "__notes__", ()))


def test_cache_root_witness_close_attempts_every_descriptor_after_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    event_descriptor, event_writer = os.pipe()
    original_close = os.close

    class ClosingWitness:
        def __init__(self, label: str, *, fail: bool = False):
            self.label = label
            self.fail = fail

        def close(self) -> None:
            calls.append(self.label)
            if self.fail:
                raise OSError(f"{self.label} close failed")

    def record_close(descriptor: int) -> None:
        if descriptor == event_descriptor:
            calls.append("event")
        original_close(descriptor)

    monkeypatch.setattr(os, "close", record_close)
    witness = rlt_stage2_build_cache._CacheRootWitness(  # noqa: SLF001
        parent=ClosingWitness("parent"),
        root=ClosingWitness("root", fail=True),
        event_descriptor=event_descriptor,
    )
    try:
        with pytest.raises(OSError, match="root close failed"):
            witness.close()
    finally:
        original_close(event_writer)

    assert calls == ["root", "parent", "event"]


def test_cache_root_rename_away_and_back_after_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "cache" / "feature" / "batch"
    destination.mkdir(parents=True)
    parked = destination.with_name("batch-parked")
    features = _Features(np.asarray([1, 2], dtype=np.int32))
    transitions = _Transitions(np.asarray([3, 4], dtype=np.int32))
    identity_fields = {"feature_identity": _SHA_D, "batch_id": "batch"}
    manifest = {
        "schema_version": 1,
        **identity_fields,
        "feature_rows": 2,
        "transition_rows": 2,
        "files": [],
    }

    class Cache:
        _validate_features = staticmethod(_validate_fake_features)
        _validate_transitions = staticmethod(_validate_fake_transitions)
        _validate_identity_fields = staticmethod(_validate_fake_identity)

        @staticmethod
        def open_shard(path: Path):
            if ".stage-" in path.name:
                manifest_bytes = (path / "manifest.json").read_bytes()
                return SimpleNamespace(
                    root=path,
                    manifest=json.loads(manifest_bytes),
                    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                )
            return SimpleNamespace(
                root=path,
                manifest=dict(manifest),
                manifest_sha256="9" * 64,
            )

    def rename_away_and_back(path: Path) -> None:
        path.rename(parked)
        path.mkdir()
        path.rmdir()
        parked.rename(path)

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_after_cache_open_hook",
        rename_away_and_back,
    )

    with pytest.raises(RuntimeError, match="changed|replaced|renamed"):
        rlt_stage2_build_cache._publish_or_reuse_cache(  # noqa: SLF001
            SimpleNamespace(cache=Cache),
            destination,
            features=features,
            transitions=transitions,
            identity_fields=identity_fields,
        )


def test_cache_publish_is_bound_to_held_feature_parent_during_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, _, _ = _run_runtime(config, monkeypatch)
    feature_parent = config.training_root / "feature_cache" / _SHA_D
    parked = config.training_root / "feature_cache" / f"{_SHA_D}-parked"
    outside = tmp_path / "outside-cache-publish"
    outside.mkdir()
    swapped = False

    def enter_swap(*args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        assert not swapped
        feature_parent.rename(parked)
        feature_parent.symlink_to(outside, target_is_directory=True)
        swapped = True

    def leave_swap(*args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        assert swapped
        feature_parent.unlink()
        parked.rename(feature_parent)
        swapped = False

    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_before_bound_cache_publish_hook",
        enter_swap,
        raising=False,
    )
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "_after_bound_cache_publish_hook",
        leave_swap,
        raising=False,
    )

    try:
        result = rlt_stage2_build_cache.run(config, runtime=runtime)
    except RuntimeError:
        result = None

    assert not swapped
    assert list(outside.iterdir()) == []
    if result is not None:
        assert result.destination == feature_parent / config.batch.name
        assert result.destination.is_dir()


def test_run_orders_strict_pipeline_and_passes_loaded_parameter_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, calls, model = _run_runtime(config, monkeypatch)
    monkeypatch.setattr(
        rlt_stage2_build_cache.gc,
        "collect",
        lambda: calls.append("gc_collect") or 0,
    )

    result = rlt_stage2_build_cache.run(config, runtime=runtime)

    assert result.destination == config.training_root / "feature_cache" / _SHA_D / config.batch.name
    assert calls[:8] == [
        "admit",
        "round",
        "plan",
        "load",
        "normalizer",
        "raw",
        "dataset",
        ("extract", _SHA_C),
    ]
    assert calls[8] == "dataset_close"
    assert calls[9] == "finalize"
    assert calls[10][0] == "stage"
    assert calls[11][0] == "open"
    assert calls[-3:] == ["gc_collect", "clear_caches", "gc_collect"]
    assert model() is None


def test_cache_identity_contains_batch_fingerprints_and_all_model_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, calls, _ = _run_runtime(config, monkeypatch)

    rlt_stage2_build_cache.run(config, runtime=runtime)

    identity_fields = next(call[1] for call in calls if isinstance(call, tuple) and call[0] == "stage")
    assert identity_fields["episode_fingerprints"] == ["e" * 64, "f" * 64]
    assert identity_fields["loaded_parameter_sha256"] == _SHA_C
    assert identity_fields["loaded_norm_stats_sha256"] == _SHA_N
    assert identity_fields["checkpoint_sha256"] == _SHA_A
    assert identity_fields["norm_stats_sha256"] == _SHA_B


def test_run_rechecks_clean_commit_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, calls, _ = _run_runtime(config, monkeypatch)
    commits = iter([_COMMIT, "2" * 40])
    monkeypatch.setattr(
        rlt_stage2_build_cache,
        "current_git_commit",
        lambda: next(commits),
    )

    with pytest.raises(RuntimeError, match="Git commit changed"):
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert not any(isinstance(call, tuple) and call[0] == "stage" for call in calls)


def test_run_rereads_and_rejects_any_manifest_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, calls, _ = _run_runtime(
        config,
        monkeypatch,
        reread_manifest_update={"loaded_parameter_sha256": "0" * 64},
    )

    with pytest.raises(RuntimeError, match="reread.*manifest|identity"):
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert any(isinstance(call, tuple) and call[0] == "stage" for call in calls)
    assert any(isinstance(call, tuple) and call[0] == "open" for call in calls)


def test_upstream_failure_never_calls_cache_publish_and_preserves_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    primary = ValueError("feature extraction failed")
    runtime, calls, model = _run_runtime(
        config,
        monkeypatch,
        extraction_error=primary,
    )

    with pytest.raises(ValueError, match="feature extraction failed") as exc_info:
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert exc_info.value is primary
    assert not any(isinstance(call, tuple) and call[0] == "stage" for call in calls)
    assert "dataset_close" in calls
    assert "clear_caches" in calls
    assert model() is None
    training_root_text = str(config.training_root)
    open_targets = []
    for entry in Path("/proc/self/fd").iterdir():
        with contextlib.suppress(OSError):
            open_targets.append(os.readlink(entry))
    assert not any(training_root_text in target for target in open_targets)


def test_caught_failure_releases_model_jit_payload_and_file_while_preserving_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, _, model = _run_runtime(config, monkeypatch)
    payload_ref: weakref.ReferenceType[object] | None = None
    file_ref: weakref.ReferenceType[object] | None = None
    payload_path = config.training_root / "jit-payload.bin"
    payload_path.write_bytes(b"payload")

    class JitPayload:
        def __init__(self):
            self.large = np.zeros((1024, 1024), dtype=np.float32)

    def fail_with_heavy_locals(**kwargs: Any):
        nonlocal payload_ref, file_ref
        payload = JitPayload()
        opened_file = payload_path.open("rb")
        payload_ref = weakref.ref(payload)
        file_ref = weakref.ref(opened_file)
        kwargs["dataset"].close()
        try:
            raise KeyError("inner cause")
        except KeyError as cause:
            raise ValueError("feature extraction failed with heavy locals") from cause

    runtime.feature_extractor.extract_features_with_frozen_guard = fail_with_heavy_locals

    with pytest.raises(ValueError, match="heavy locals") as exc_info:
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert isinstance(exc_info.value.__cause__, KeyError)
    assert str(exc_info.value.__cause__) == "'inner cause'"
    assert model() is None
    assert payload_ref is not None
    assert payload_ref() is None
    assert file_ref is not None
    assert file_ref() is None
    fd_targets: list[str] = []
    for entry in Path("/proc/self/fd").iterdir():
        with contextlib.suppress(OSError):
            fd_targets.append(os.readlink(entry))
    assert str(payload_path) not in fd_targets


def test_cleanup_errors_do_not_swallow_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    primary = ValueError("feature extraction failed")
    runtime, _, _ = _run_runtime(
        config,
        monkeypatch,
        extraction_error=primary,
    )
    monkeypatch.setattr(
        rlt_stage2_build_cache.gc,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("gc failed")),
    )
    runtime.jax.clear_caches = lambda: (_ for _ in ()).throw(RuntimeError("jax cleanup failed"))

    with pytest.raises(ValueError, match="feature extraction failed") as exc_info:
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert exc_info.value is primary
    notes = getattr(exc_info.value, "__notes__", ())
    assert any("gc failed" in note for note in notes)
    assert any("jax cleanup failed" in note for note in notes)


def test_cleanup_error_tracebacks_release_heavy_locals_and_open_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, _, _ = _run_runtime(config, monkeypatch)
    payload_path = config.training_root / "cleanup-payload.bin"
    payload_path.write_bytes(b"payload")
    payload_refs: list[weakref.ReferenceType[object]] = []
    file_refs: list[weakref.ReferenceType[object]] = []

    class CleanupPayload:
        def __init__(self):
            self.large = np.zeros((1024, 1024), dtype=np.float32)

    def failing_collect() -> None:
        payload = CleanupPayload()
        opened_file = payload_path.open("rb")
        payload_refs.append(weakref.ref(payload))
        file_refs.append(weakref.ref(opened_file))
        raise RuntimeError("gc cleanup failed with heavy locals")

    monkeypatch.setattr(rlt_stage2_build_cache.gc, "collect", failing_collect)

    with pytest.raises(RuntimeError, match="cleanup.*heavy locals") as exc_info:
        rlt_stage2_build_cache.run(config, runtime=runtime)

    assert exc_info.value.__cause__ is not None
    assert payload_refs
    assert all(reference() is None for reference in payload_refs)
    assert file_refs
    assert all(reference() is None for reference in file_refs)
    fd_targets: list[str] = []
    for entry in Path("/proc/self/fd").iterdir():
        with contextlib.suppress(OSError):
            fd_targets.append(os.readlink(entry))
    assert str(payload_path) not in fd_targets


def test_success_path_reports_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    runtime, _, _ = _run_runtime(config, monkeypatch)
    monkeypatch.setattr(
        rlt_stage2_build_cache.gc,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("gc failed")),
    )

    with pytest.raises(RuntimeError, match="cleanup.*gc failed"):
        rlt_stage2_build_cache.run(config, runtime=runtime)
