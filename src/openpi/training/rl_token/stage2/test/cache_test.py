# ruff: noqa: SLF001

from __future__ import annotations

import concurrent.futures
import dataclasses
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
from typing import Any

import ml_dtypes
import numpy as np
import pytest

from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import cache
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import replay
from openpi.training.rl_token.stage2 import transitions
from openpi.training.rl_token.stage2.test.conftest import build_ready_batch
import openpi.transforms as openpi_transforms


def _feature_table() -> cache.FeatureTable:
    return cache.FeatureTable(
        episode_index=np.array([0, 0, 0], dtype=np.int32),
        frame_index=np.array([0, 2, 20], dtype=np.int32),
        z_rl=np.ones((3, 8), dtype=ml_dtypes.bfloat16),
        state_norm=np.ones((3, 3), dtype=np.float32),
        vla_reference=np.ones((3, 4, 2), dtype=np.float32),
    )


def _transition_table() -> cache.TransitionTable:
    return cache.TransitionTable(
        episode_index=np.array([0, 0], dtype=np.int32),
        start_frame_index=np.array([0, 2], dtype=np.int32),
        current_feature_row=np.array([0, 1], dtype=np.int64),
        next_feature_row=np.array([2, -1], dtype=np.int64),
        executed_action=np.ones((2, 4, 2), dtype=np.float32),
        bc_anchor=np.ones((2, 4, 2), dtype=np.float32),
        reward=np.array([[0.0], [2.0]], dtype=np.float32),
        terminal=np.array([[False], [True]], dtype=np.bool_),
    )


def _identity_fields() -> dict[str, Any]:
    return {
        "feature_identity": "model_id",
        "batch_id": "batch_1",
        "manifest_sha256": "1" * 64,
        "labels_sha256": "2" * 64,
        "sampler": {"steps": 10, "enabled": True, "tags": ["fixed"]},
    }


def _publish(tmp_path: Path, name: str = "batch_1") -> Path:
    destination = tmp_path / "feature_cache" / "model_id" / name
    cache.publish_shard(
        destination,
        features=_feature_table(),
        transitions=_transition_table(),
        identity_fields=_identity_fields(),
    )
    return destination


def _read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_bytes(identity.canonical_json_bytes(manifest))


def _record(manifest: dict[str, Any], relative: str) -> dict[str, Any]:
    return next(record for record in manifest["files"] if record["path"] == relative)


def _refresh_record(root: Path, manifest: dict[str, Any], relative: str) -> None:
    path = root / relative
    record = _record(manifest, relative)
    record["size"] = path.stat().st_size
    record["sha256"] = identity.sha256_file(path)


def test_publish_and_open_memory_mapped_shard(tmp_path: Path):
    destination = tmp_path / "feature_cache/model_id/batch_1"
    manifest = cache.publish_shard(
        destination,
        features=_feature_table(),
        transitions=_transition_table(),
        identity_fields=_identity_fields(),
    )

    opened = cache.open_shard(destination)

    assert manifest == opened.manifest
    assert opened.root == destination.resolve()
    assert opened.manifest_sha256 == hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()
    assert isinstance(opened.features.z_rl, np.memmap)
    assert opened.features.z_rl.dtype == ml_dtypes.bfloat16
    assert opened.features.state_norm.dtype == np.float32
    assert opened.transitions.executed_action.dtype == np.float32
    assert opened.transitions.terminal.dtype == np.bool_
    for table in (opened.features, opened.transitions):
        for field in dataclasses.fields(table):
            array = getattr(table, field.name)
            assert isinstance(array, np.memmap)
            assert not array.flags.writeable
    assert (destination / "manifest.json").read_bytes() == identity.canonical_json_bytes(manifest)
    assert [record["path"] for record in manifest["files"]] == sorted(record["path"] for record in manifest["files"])
    assert all(set(record) == {"dtype", "path", "sha256", "shape", "size"} for record in manifest["files"])
    assert manifest["feature_rows"] == 3
    assert manifest["transition_rows"] == 2


def test_bfloat16_bit_pattern_roundtrips_as_read_only_memmap(tmp_path: Path):
    values = np.linspace(-3.5, 4.25, 24, dtype=np.float32).reshape(3, 8)
    z_rl = values.astype(ml_dtypes.bfloat16)
    features = dataclasses.replace(_feature_table(), z_rl=z_rl)
    destination = tmp_path / "batch_1"
    cache.publish_shard(
        destination,
        features=features,
        transitions=_transition_table(),
        identity_fields=_identity_fields(),
    )

    opened = cache.open_shard(destination).features.z_rl

    assert isinstance(opened, np.memmap)
    assert opened.dtype == np.dtype(ml_dtypes.bfloat16)
    assert not opened.flags.writeable
    np.testing.assert_array_equal(opened.view(np.uint16), z_rl.view(np.uint16))
    np.testing.assert_array_equal(opened, z_rl)


def test_open_shard_is_an_idempotent_context_managed_mmap_owner(tmp_path: Path):
    with cache.open_shard(_publish(tmp_path)) as opened:
        memory_maps = [
            getattr(table, field.name)._mmap
            for table in (opened.features, opened.transitions)
            for field in dataclasses.fields(table)
        ]
        assert len(memory_maps) == 13
        assert all(not memory_map.closed for memory_map in memory_maps)

    assert all(memory_map.closed for memory_map in memory_maps)
    opened.close()
    assert all(memory_map.closed for memory_map in memory_maps)


def test_authentication_returns_sealed_fd_free_verification_and_fast_open_does_not_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _publish(tmp_path)
    verification = cache.authenticate_shard(root)

    assert verification.root == root.resolve()
    assert verification.manifest["batch_id"] == "batch_1"
    assert verification.manifest_sha256 == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    assert len(verification.files) == 13
    with pytest.raises(TypeError):
        cache.ShardVerification()

    def forbid_full_hash(_descriptor: int) -> str:
        raise AssertionError("fast shard open must not hash an authenticated payload")

    def forbid_full_semantic_validation(*_args: Any, **_kwargs: Any) -> int:
        raise AssertionError("fast shard open must not scan arrays for finite values")

    monkeypatch.setattr(cache, "_sha256_fd", forbid_full_hash)
    monkeypatch.setattr(cache, "_validate_features", forbid_full_semantic_validation)
    monkeypatch.setattr(cache, "_validate_transitions", forbid_full_semantic_validation)
    with cache.open_from_verification(verification) as opened:
        np.testing.assert_array_equal(opened.transitions.reward, _transition_table().reward)
    cache.verify_verification(verification, full=False)


def test_verify_verification_full_rehashes_payload_but_quick_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    verification = cache.authenticate_shard(_publish(tmp_path))
    original = cache._sha256_fd
    hashed: list[int] = []

    def record_hash(descriptor: int) -> str:
        hashed.append(descriptor)
        return original(descriptor)

    monkeypatch.setattr(cache, "_sha256_fd", record_hash)
    cache.verify_verification(verification, full=False)
    assert hashed == []

    cache.verify_verification(verification, full=True)
    assert len(hashed) == 13


def test_quick_verification_rechecks_payload_path_after_reading_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _publish(tmp_path)
    verification = cache.authenticate_shard(root)
    relative = "transitions/reward.npy"
    path = root / relative
    replacement = tmp_path / "replacement-reward.npy"
    np.save(replacement, np.array([[91.0], [92.0]], dtype=np.float32), allow_pickle=False)
    assert replacement.stat().st_size == path.stat().st_size
    original = cache._read_npy_header
    replaced = False

    def replace_after_header(*args: Any, **kwargs: Any) -> tuple[tuple[int, ...], np.dtype]:
        nonlocal replaced
        result = original(*args, **kwargs)
        if kwargs["relative"] == relative and not replaced:
            os.replace(replacement, path)
            replaced = True
        return result

    monkeypatch.setattr(cache, "_read_npy_header", replace_after_header)
    with pytest.raises(cache.CacheError, match="binding changed|path changed"):
        cache.verify_verification(verification, full=False)
    assert replaced


def test_open_shard_partial_mmap_failure_closes_every_already_opened_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _publish(tmp_path)
    verification = cache.authenticate_shard(root)
    original = cache._mmap_verified_array
    opened: list[np.memmap] = []

    def fail_after_five(*args: Any, **kwargs: Any) -> np.memmap:
        if len(opened) == 5:
            raise RuntimeError("injected mmap failure")
        value = original(*args, **kwargs)
        opened.append(value)
        return value

    monkeypatch.setattr(cache, "_mmap_verified_array", fail_after_five)
    with pytest.raises(cache.CacheError, match="injected mmap failure"):
        cache.open_from_verification(verification)

    assert len(opened) == 5
    assert all(value._mmap.closed for value in opened)


def test_open_shard_close_failure_preserves_primary_and_retries_only_unclosed_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    opened = cache.open_shard(_publish(tmp_path))
    memory_maps = [
        getattr(table, field.name)._mmap
        for table in (opened.features, opened.transitions)
        for field in dataclasses.fields(table)
    ]
    blocked = memory_maps[4]
    original = cache._close_memory_map
    calls: list[Any] = []
    fail_once = True

    def injected_close(memory_map: Any) -> None:
        nonlocal fail_once
        calls.append(memory_map)
        if memory_map is blocked and fail_once:
            fail_once = False
            raise BufferError("injected close before release")
        original(memory_map)

    monkeypatch.setattr(cache, "_close_memory_map", injected_close)
    primary = RuntimeError("body primary")
    with pytest.raises(RuntimeError) as caught, opened:
        raise primary

    assert caught.value is primary
    assert any("injected close before release" in note for note in getattr(primary, "__notes__", ()))
    if opened.closed:
        raise AssertionError("OpenShard became terminal despite one mmap remaining open")
    assert not blocked.closed
    assert all(memory_map.closed for memory_map in memory_maps if memory_map is not blocked)

    calls.clear()
    opened.close()
    if not opened.closed:
        raise AssertionError("OpenShard did not become terminal after the retry succeeded")
    assert len(calls) == 1
    assert calls[0] is blocked
    assert all(memory_map.closed for memory_map in memory_maps)


def test_non_z_void_payload_cannot_use_bfloat16_reinterpretation(tmp_path: Path):
    destination = _publish(tmp_path)
    relative = "features/state_norm.npy"
    path = destination / relative
    original_shape = _feature_table().state_norm.shape
    np.save(path, np.zeros(original_shape, dtype=np.dtype("V4")), allow_pickle=False)
    manifest = _read_manifest(destination)
    _refresh_record(destination, manifest, relative)
    _write_manifest(destination, manifest)

    with pytest.raises(cache.CacheError, match=r"state_norm\.npy dtype mismatch"):
        cache.open_shard(destination)


def test_authentication_rejects_path_replacement_after_hash_before_mmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = _publish(tmp_path)
    relative = "transitions/reward.npy"
    original_path = destination / relative
    replacement = tmp_path / "replacement.npy"
    replacement_values = np.array([[91.0], [92.0]], dtype=np.float32)
    np.save(replacement, replacement_values, allow_pickle=False)
    assert replacement.stat().st_size == original_path.stat().st_size
    manifest = _read_manifest(destination)
    expected_sha256 = _record(manifest, relative)["sha256"]
    replaced = False

    def replace_after_verification(path: Path, _descriptor: int) -> None:
        nonlocal replaced
        if path == original_path and not replaced:
            os.replace(replacement, original_path)
            replaced = True

    monkeypatch.setattr(cache, "_before_mmap_load", replace_after_verification)

    with pytest.raises(cache.CacheError, match="binding changed"):
        cache.open_shard(destination)

    assert replaced
    np.testing.assert_array_equal(
        np.load(original_path, allow_pickle=False),
        replacement_values,
    )
    assert identity.sha256_file(original_path) != expected_sha256


def test_authentication_rejects_path_becoming_symlink_before_mmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = _publish(tmp_path)
    original_path = destination / "transitions/reward.npy"
    target = tmp_path / "symlink-target.npy"
    target_values = np.array([[71.0], [72.0]], dtype=np.float32)
    np.save(target, target_values, allow_pickle=False)
    assert target.stat().st_size == original_path.stat().st_size
    replaced = False

    def symlink_after_verification(path: Path, _descriptor: int) -> None:
        nonlocal replaced
        if path == original_path and not replaced:
            path.unlink()
            path.symlink_to(target)
            replaced = True

    monkeypatch.setattr(cache, "_before_mmap_load", symlink_after_verification)

    with pytest.raises(cache.CacheError, match="binding changed|symlink"):
        cache.open_shard(destination)

    assert replaced
    assert original_path.is_symlink()
    np.testing.assert_array_equal(
        np.load(original_path, allow_pickle=False),
        target_values,
    )


def test_open_rejects_parent_component_before_lexical_normalization_can_hide_symlink(
    tmp_path: Path,
):
    base = tmp_path / "base"
    base.mkdir()
    destination = _publish(base)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "subdir").mkdir(parents=True)
    (base / "link").symlink_to(elsewhere / "subdir", target_is_directory=True)
    requested = base / "link" / ".." / destination.relative_to(base)

    with pytest.raises(cache.CacheError, match=r"parent component|normalized|\.\."):
        cache.open_shard(requested)


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_repeated_open_releases_pinning_fds_and_memmaps_survive_pin_close(
    tmp_path: Path,
):
    destination = _publish(tmp_path)
    gc.collect()
    before = _open_fd_count()

    for _ in range(101):
        opened = cache.open_shard(destination)
        reward = opened.transitions.reward
        np.testing.assert_array_equal(reward, _transition_table().reward)
        assert isinstance(reward, np.memmap)
        assert not reward.flags.writeable
        del opened
        np.testing.assert_array_equal(reward, _transition_table().reward)
        del reward
    gc.collect()

    assert _open_fd_count() <= before


@pytest.mark.parametrize("failure", ["hash", "header"])
def test_repeated_payload_failures_do_not_leak_file_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    destination = _publish(tmp_path)
    error = "injected fd hash failure"
    if failure == "hash":

        def fail_hash(_descriptor: int) -> str:
            raise OSError(error)

        monkeypatch.setattr(cache, "_sha256_fd", fail_hash)
    else:
        relative = "features/episode_index.npy"
        path = destination / relative
        with path.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"BROKEN")
        manifest = _read_manifest(destination)
        _refresh_record(destination, manifest, relative)
        _write_manifest(destination, manifest)
        error = "failed to load"
    gc.collect()
    before = _open_fd_count()

    for _ in range(101):
        with pytest.raises(cache.CacheError, match=error):
            cache.open_shard(destination)
    gc.collect()

    assert _open_fd_count() <= before


def test_read_only_mmaps_reject_writes(tmp_path: Path):
    opened = cache.open_shard(_publish(tmp_path))
    with pytest.raises(ValueError, match="read-only"):
        opened.features.z_rl[0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        opened.transitions.reward[0, 0] = 1


def test_publish_refuses_existing_nonempty_or_empty_destination(tmp_path: Path):
    destination = _publish(tmp_path)
    original = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        cache.publish_shard(
            destination,
            features=_feature_table(),
            transitions=_transition_table(),
            identity_fields=_identity_fields(),
        )
    assert (destination / "manifest.json").read_bytes() == original

    empty = tmp_path / "already_empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        cache.publish_shard(
            empty,
            features=_feature_table(),
            transitions=_transition_table(),
            identity_fields=_identity_fields(),
        )
    assert list(empty.iterdir()) == []


def test_publish_validates_everything_before_filesystem_mutation(tmp_path: Path):
    destination = tmp_path / "not-created" / "batch_1"
    bad = dataclasses.replace(_feature_table(), z_rl=np.ones((3, 8), dtype=np.float32))
    with pytest.raises(cache.CacheError, match=r"features\.z_rl.*bfloat16"):
        cache.publish_shard(
            destination,
            features=bad,
            transitions=_transition_table(),
            identity_fields=_identity_fields(),
        )
    assert not destination.parent.exists()


@pytest.mark.parametrize(
    ("table_name", "field", "value", "error"),
    [
        ("features", "episode_index", np.array([0, 0, 0], dtype=np.uint32), "signed integer"),
        ("features", "frame_index", np.array([False, False, False]), "signed integer"),
        ("features", "frame_index", np.zeros((3, 1), dtype=np.int32), "rank 1"),
        ("features", "z_rl", np.ones((3, 8), dtype=np.float32), "bfloat16"),
        ("features", "z_rl", np.ones((3, 2, 4), dtype=ml_dtypes.bfloat16), "rank 2"),
        ("features", "state_norm", np.ones((3, 3), dtype=np.float64), "float32"),
        ("features", "state_norm", np.ones((3, 3, 1), dtype=np.float32), "rank 2"),
        ("features", "vla_reference", np.ones((3, 4, 2), dtype=np.float64), "float32"),
        ("features", "vla_reference", np.ones((3, 8), dtype=np.float32), "rank 3"),
        ("transitions", "current_feature_row", np.array([0, 1], dtype=np.uint64), "signed integer"),
        ("transitions", "next_feature_row", np.array([2.0, -1.0], dtype=np.float32), "signed integer"),
        ("transitions", "episode_index", np.zeros((2, 1), dtype=np.int32), "rank 1"),
        ("transitions", "executed_action", np.ones((2, 4, 2), dtype=np.float64), "float32"),
        ("transitions", "executed_action", np.ones((2, 8), dtype=np.float32), "rank 3"),
        ("transitions", "bc_anchor", np.ones((2, 4, 2), dtype=np.float64), "float32"),
        ("transitions", "reward", np.ones((2,), dtype=np.float32), r"shape \[N,1\]"),
        ("transitions", "reward", np.ones((2, 1), dtype=np.float64), "float32"),
        ("transitions", "terminal", np.zeros((2,), dtype=np.bool_), r"shape \[N,1\]"),
        ("transitions", "terminal", np.zeros((2, 1), dtype=np.int8), "bool"),
    ],
)
def test_publish_rejects_wrong_field_dtype_or_rank(
    tmp_path: Path,
    table_name: str,
    field: str,
    value: np.ndarray,
    error: str,
):
    features = _feature_table()
    transition_table = _transition_table()
    if table_name == "features":
        features = dataclasses.replace(features, **{field: value})
    else:
        transition_table = dataclasses.replace(transition_table, **{field: value})
    with pytest.raises(cache.CacheError, match=rf"{table_name}\.{field}.*{error}"):
        cache.publish_shard(
            tmp_path / "shard",
            features=features,
            transitions=transition_table,
            identity_fields=_identity_fields(),
        )


@pytest.mark.parametrize(
    ("table_name", "field"),
    [
        ("features", "z_rl"),
        ("features", "state_norm"),
        ("features", "vla_reference"),
        ("transitions", "executed_action"),
        ("transitions", "bc_anchor"),
        ("transitions", "reward"),
    ],
)
def test_publish_rejects_nonfinite_float_arrays(tmp_path: Path, table_name: str, field: str):
    features = _feature_table()
    transition_table = _transition_table()
    table = features if table_name == "features" else transition_table
    value = getattr(table, field).copy()
    value.flat[0] = np.nan
    if table_name == "features":
        features = dataclasses.replace(features, **{field: value})
    else:
        transition_table = dataclasses.replace(transition_table, **{field: value})
    with pytest.raises(cache.CacheError, match=rf"{table_name}\.{field}.*finite"):
        cache.publish_shard(
            tmp_path / "shard",
            features=features,
            transitions=transition_table,
            identity_fields=_identity_fields(),
        )


@pytest.mark.parametrize("which", ["features", "transitions"])
@pytest.mark.parametrize("failure", ["leading", "empty", "unsorted", "duplicate"])
def test_publish_rejects_row_count_and_key_invariants(tmp_path: Path, which: str, failure: str):
    features = _feature_table()
    transition_table = _transition_table()
    if which == "features":
        if failure == "leading":
            features = dataclasses.replace(features, state_norm=features.state_norm[:2])
        elif failure == "empty":
            features = cache.FeatureTable(
                *(getattr(features, field.name)[:0] for field in dataclasses.fields(features))
            )
        elif failure == "unsorted":
            features = dataclasses.replace(features, frame_index=np.array([2, 0, 20], dtype=np.int32))
        else:
            features = dataclasses.replace(features, frame_index=np.array([0, 0, 20], dtype=np.int32))
    elif failure == "leading":
        transition_table = dataclasses.replace(transition_table, reward=transition_table.reward[:1])
    elif failure == "empty":
        transition_table = cache.TransitionTable(
            *(getattr(transition_table, field.name)[:0] for field in dataclasses.fields(transition_table))
        )
    elif failure == "unsorted":
        transition_table = dataclasses.replace(
            transition_table,
            start_frame_index=np.array([2, 0], dtype=np.int32),
        )
    else:
        transition_table = dataclasses.replace(
            transition_table,
            start_frame_index=np.array([0, 0], dtype=np.int32),
        )
    with pytest.raises(cache.CacheError, match=which):
        cache.publish_shard(
            tmp_path / "shard",
            features=features,
            transitions=transition_table,
            identity_fields=_identity_fields(),
        )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"current_feature_row": np.array([-1, 1], dtype=np.int64)}, "current_feature_row"),
        ({"current_feature_row": np.array([3, 1], dtype=np.int64)}, "current_feature_row"),
        ({"current_feature_row": np.array([1, 0], dtype=np.int64)}, "current_feature_row.*identity"),
        ({"next_feature_row": np.array([-2, -1], dtype=np.int64)}, "next_feature_row"),
        ({"next_feature_row": np.array([3, -1], dtype=np.int64)}, "next_feature_row"),
        ({"next_feature_row": np.array([1, -1], dtype=np.int64)}, "next_feature_row.*identity"),
        ({"next_feature_row": np.array([-1, -1], dtype=np.int64)}, "if and only if"),
        ({"next_feature_row": np.array([2, 1], dtype=np.int64)}, "if and only if"),
    ],
)
def test_publish_rejects_feature_row_and_terminal_relation(
    tmp_path: Path,
    changes: dict[str, np.ndarray],
    error: str,
):
    transition_table = dataclasses.replace(_transition_table(), **changes)
    with pytest.raises(cache.CacheError, match=error):
        cache.publish_shard(
            tmp_path / "shard",
            features=_feature_table(),
            transitions=transition_table,
            identity_fields=_identity_fields(),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "current_feature_row",
            np.array([1, 0], dtype=np.int64),
            "current_feature_row.*identity",
        ),
        (
            "next_feature_row",
            np.array([1, -1], dtype=np.int64),
            "next_feature_row.*identity",
        ),
    ],
)
def test_open_rejects_authenticated_but_topologically_wrong_feature_rows(
    tmp_path: Path,
    field: str,
    replacement: np.ndarray,
    error: str,
):
    destination = _publish(tmp_path)
    relative = f"transitions/{field}.npy"
    np.save(destination / relative, replacement, allow_pickle=False)
    manifest = _read_manifest(destination)
    _refresh_record(destination, manifest, relative)
    _write_manifest(destination, manifest)

    with pytest.raises(cache.CacheError, match=error):
        cache.open_shard(destination)


@pytest.mark.parametrize(
    ("which", "field"),
    [
        ("features", "episode_index"),
        ("features", "frame_index"),
        ("transitions", "episode_index"),
        ("transitions", "start_frame_index"),
    ],
)
def test_publish_rejects_negative_episode_or_frame_identity(
    tmp_path: Path,
    which: str,
    field: str,
):
    features = _feature_table()
    transition_table = _transition_table()
    table = features if which == "features" else transition_table
    replacement = getattr(table, field).copy()
    replacement[0] = -1
    if which == "features":
        features = dataclasses.replace(features, **{field: replacement})
    else:
        transition_table = dataclasses.replace(
            transition_table,
            **{field: replacement},
        )
    with pytest.raises(cache.CacheError, match=rf"{which}\.{field}.*nonnegative"):
        cache.publish_shard(
            tmp_path / "shard",
            features=features,
            transitions=transition_table,
            identity_fields=_identity_fields(),
        )


def test_publish_rejects_action_reference_trailing_shape_mismatch(tmp_path: Path):
    transition_table = dataclasses.replace(
        _transition_table(),
        executed_action=np.ones((2, 20, 16), dtype=np.float32),
        bc_anchor=np.ones((2, 20, 16), dtype=np.float32),
    )
    with pytest.raises(cache.CacheError, match="trailing shape"):
        cache.publish_shard(
            tmp_path / "shard",
            features=_feature_table(),
            transitions=transition_table,
            identity_fields=_identity_fields(),
        )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"feature_identity": ""}, "feature_identity"),
        ({"feature_identity": 1}, "feature_identity"),
        ({"batch_id": ""}, "batch_id"),
        ({"batch_id": None}, "batch_id"),
        ({"schema_version": 2}, "reserved"),
        ({"files": []}, "reserved"),
        ({"feature_rows": 1}, "reserved"),
        ({"transition_rows": 1}, "reserved"),
        ({"manifest_sha256": "A" * 64}, "manifest_sha256"),
        ({"labels_sha256": "1" * 63}, "labels_sha256"),
        ({"path_value": Path("not-json")}, "JSON"),
        ({"tuple_value": ("not", "json")}, "JSON"),
        ({"nan_value": float("nan")}, "JSON"),
        ({1: "non-string-key"}, "JSON"),
    ],
)
def test_publish_rejects_invalid_or_reserved_identity(
    tmp_path: Path,
    changes: dict[Any, Any],
    error: str,
):
    fields = _identity_fields()
    fields.update(changes)
    with pytest.raises(cache.CacheError, match=error):
        cache.publish_shard(
            tmp_path / "shard",
            features=_feature_table(),
            transitions=_transition_table(),
            identity_fields=fields,
        )
    assert not (tmp_path / "shard").exists()


@pytest.mark.parametrize("mode", ["size", "sha256", "header"])
def test_open_rejects_corrupt_array(tmp_path: Path, mode: str):
    destination = _publish(tmp_path)
    relative = "transitions/reward.npy"
    path = destination / relative
    manifest = _read_manifest(destination)
    if mode == "size":
        with path.open("ab") as stream:
            stream.write(b"x")
        error = "size mismatch"
    elif mode == "sha256":
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            byte = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([byte[0] ^ 0xFF]))
        error = "sha256 mismatch"
    else:
        with path.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"BROKEN")
        _refresh_record(destination, manifest, relative)
        _write_manifest(destination, manifest)
        error = "failed to load"
    with pytest.raises(cache.CacheError, match=rf"reward\.npy.*{error}"):
        cache.open_shard(destination)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("unsafe", "unsafe cache path"),
        ("duplicate", "duplicate cache path"),
        ("missing_record", "file set"),
        ("extra_record", "file set"),
        ("wrong_size_type", "size"),
        ("wrong_hash_type", "sha256"),
        ("wrong_shape_type", "shape"),
        ("wrong_dtype_type", "dtype"),
    ],
)
def test_open_rejects_malformed_manifest_records(tmp_path: Path, mutation: str, error: str):
    destination = _publish(tmp_path)
    manifest = _read_manifest(destination)
    if mutation == "unsafe":
        manifest["files"][0]["path"] = "../escape.npy"
    elif mutation == "duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
    elif mutation == "missing_record":
        manifest["files"].pop()
    elif mutation == "extra_record":
        manifest["files"].append(
            {
                "path": "features/extra.npy",
                "size": 1,
                "sha256": "0" * 64,
                "shape": [1],
                "dtype": "int8",
            }
        )
    elif mutation == "wrong_size_type":
        manifest["files"][0]["size"] = True
    elif mutation == "wrong_hash_type":
        manifest["files"][0]["sha256"] = "A" * 64
    elif mutation == "wrong_shape_type":
        manifest["files"][0]["shape"] = "3"
    else:
        manifest["files"][0]["dtype"] = 1
    _write_manifest(destination, manifest)
    with pytest.raises(cache.CacheError, match=error):
        cache.open_shard(destination)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("feature_rows", True, "feature_rows"),
        ("feature_rows", -1, "feature_rows"),
        ("transition_rows", "2", "transition_rows"),
        ("files", {}, "files"),
        ("feature_identity", "", "feature_identity"),
        ("batch_id", 1, "batch_id"),
        ("manifest_sha256", "Z" * 64, "manifest_sha256"),
    ],
)
def test_open_rejects_wrong_manifest_schema_types(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
):
    destination = _publish(tmp_path)
    manifest = _read_manifest(destination)
    manifest[field] = value
    _write_manifest(destination, manifest)
    with pytest.raises(cache.CacheError, match=error):
        cache.open_shard(destination)


def test_open_rejects_duplicate_json_keys(tmp_path: Path):
    destination = _publish(tmp_path)
    original = (destination / "manifest.json").read_text(encoding="utf-8").rstrip()
    duplicate = original[:-1] + ',"schema_version":1}\n'
    (destination / "manifest.json").write_text(duplicate, encoding="utf-8")
    with pytest.raises(cache.CacheError, match="duplicate JSON key"):
        cache.open_shard(destination)


@pytest.mark.parametrize("kind", ["extra_file", "extra_directory", "missing_payload"])
def test_open_requires_exact_physical_payload_set(tmp_path: Path, kind: str):
    destination = _publish(tmp_path)
    if kind == "extra_file":
        (destination / "features/unlisted.npy").write_bytes(b"not listed")
    elif kind == "extra_directory":
        (destination / "features/nested").mkdir()
    else:
        (destination / "features/state_norm.npy").unlink()
    with pytest.raises(cache.CacheError, match="payload|missing"):
        cache.open_shard(destination)


def test_open_rejects_root_manifest_and_payload_symlinks(tmp_path: Path):
    destination = _publish(tmp_path)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(cache.CacheError, match="symlink"):
        cache.open_shard(root_link)

    manifest = destination / "manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    manifest.replace(outside_manifest)
    manifest.symlink_to(outside_manifest)
    with pytest.raises(cache.CacheError, match="manifest.*symlink|symlink.*manifest"):
        cache.open_shard(destination)

    manifest.unlink()
    outside_manifest.replace(manifest)
    payload = destination / "features/state_norm.npy"
    outside_payload = tmp_path / "outside.npy"
    payload.replace(outside_payload)
    payload.symlink_to(outside_payload)
    with pytest.raises(cache.CacheError, match="state_norm.npy.*symlink|symlink.*state_norm.npy"):
        cache.open_shard(destination)


def test_open_shard_rejects_fifo_manifest_without_blocking(tmp_path: Path):
    root = tmp_path / "fifo-cache"
    root.mkdir()
    os.mkfifo(root / "manifest.json")
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def probe() -> None:
        try:
            cache.open_shard(root)
        except cache.CacheError as exc:
            send.send(str(exc))
        else:
            send.send("ACCEPTED")
        finally:
            send.close()

    process = context.Process(target=probe)
    process.start()
    send.close()
    try:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            raise AssertionError("open_shard blocked on a FIFO manifest")
        assert process.exitcode == 0
        assert receive.poll(timeout=0.1)
        result = receive.recv()
    finally:
        receive.close()

    assert result != "ACCEPTED"
    assert "regular file" in result


def test_open_rejects_symlinked_root_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    destination = _publish(real_parent)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    through_alias = alias / destination.relative_to(real_parent)
    with pytest.raises(cache.CacheError, match="ancestor.*symlink|symlink.*ancestor"):
        cache.open_shard(through_alias)


@pytest.mark.parametrize("stage", ["save", "manifest", "publish"])
def test_publish_failure_cleans_staging_and_never_leaves_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
):
    destination = tmp_path / "cache" / "batch_1"
    destination.parent.mkdir()
    if stage == "save":
        original = cache.np.save
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected save")
            return original(*args, **kwargs)

        monkeypatch.setattr(cache.np, "save", fail_second)
    elif stage == "manifest":
        monkeypatch.setattr(
            cache,
            "_write_manifest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected manifest")),
        )
    else:
        monkeypatch.setattr(
            cache,
            "_atomic_publish_noreplace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected publish")),
        )
    with pytest.raises(RuntimeError, match="injected"):
        cache.publish_shard(
            destination,
            features=_feature_table(),
            transitions=_transition_table(),
            identity_fields=_identity_fields(),
        )
    assert not os.path.lexists(destination)
    assert list(destination.parent.glob(f".{destination.name}.*")) == []


def test_concurrent_publication_has_exactly_one_winner_and_never_overwrites(tmp_path: Path):
    destination = tmp_path / "cache" / "batch_1"
    barrier = threading.Barrier(2)

    def publish() -> tuple[str, bytes | None]:
        barrier.wait(timeout=5)
        try:
            cache.publish_shard(
                destination,
                features=_feature_table(),
                transitions=_transition_table(),
                identity_fields=_identity_fields(),
            )
            return "success", (destination / "manifest.json").read_bytes()
        except FileExistsError:
            return "exists", None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20) for future in (pool.submit(publish), pool.submit(publish))]
    assert sorted(status for status, _ in results) == ["exists", "success"]
    success_bytes = next(payload for status, payload in results if status == "success")
    assert (destination / "manifest.json").read_bytes() == success_bytes
    cache.open_shard(destination)
    assert list(destination.parent.glob(f".{destination.name}.*")) == []


def test_feature_row_lookup_is_exact_and_rejects_duplicates():
    assert cache.feature_row_lookup(_feature_table()) == {(0, 0): 0, (0, 2): 1, (0, 20): 2}
    duplicate = dataclasses.replace(
        _feature_table(),
        frame_index=np.array([0, 0, 20], dtype=np.int32),
    )
    with pytest.raises(cache.CacheError, match="duplicate"):
        cache.feature_row_lookup(duplicate)


def _batch_plan_raw_features() -> tuple[
    admission.ValidatedBatch,
    transitions.TransitionPlan,
    transitions.RawTransitionTable,
    cache.FeatureTable,
]:
    labels = np.zeros(22, dtype=np.int8)
    labels[1] = 1
    labels[21] = 2
    intervention = np.zeros(22, dtype=np.bool_)
    intervention[[1, 4, 20]] = True
    batch = admission.ValidatedBatch(
        batch_id="batch_1",
        root=Path("/unused/batch_1"),
        fps=30,
        total_frames=22,
        manifest_sha256="1" * 64,
        labels_sha256="2" * 64,
        episode_fingerprints=("3" * 64,),
        episodes=(
            admission.ValidatedEpisode(
                episode_index=0,
                length=22,
                dataset_from_index=0,
                dataset_to_index=22,
                task="fold clothes",
                parquet_path=Path("/unused/episode.parquet"),
                parquet_size=0,
                parquet_sha256="0" * 64,
                parquet_device=0,
                parquet_inode=0,
                labels=labels,
                intervention=intervention,
            ),
        ),
    )
    plan = transitions.build_transition_plan(batch)
    executed = np.stack(
        [
            np.full((20, 16), 0.5, dtype=np.float32),
            np.full((20, 16), 0.75, dtype=np.float32),
        ]
    )
    raw = transitions.RawTransitionTable(
        episode_index=np.array([0, 0], dtype=np.int32),
        start_frame_index=np.array([0, 2], dtype=np.int32),
        executed_action=executed,
        intervention=np.stack([intervention[:20], intervention[2:22]]),
        reward=np.array([[1], [2]], dtype=np.float32),
        terminal=np.array([[False], [True]], dtype=np.bool_),
    )
    references = np.stack(
        [
            np.full((20, 16), -0.1, dtype=np.float32),
            np.full((20, 16), -0.2, dtype=np.float32),
            np.full((20, 16), -0.3, dtype=np.float32),
        ]
    )
    features = cache.FeatureTable(
        episode_index=np.array([0, 0, 0], dtype=np.int32),
        frame_index=np.array([0, 2, 20], dtype=np.int32),
        z_rl=np.ones((3, 2048), dtype=ml_dtypes.bfloat16),
        state_norm=np.ones((3, 16), dtype=np.float32),
        vla_reference=references,
    )
    return batch, plan, raw, features


def test_finalize_maps_current_next_and_builds_framewise_hil_anchor():
    batch, plan, raw, features = _batch_plan_raw_features()
    result = cache.finalize_transition_table(batch, plan, raw, features)

    np.testing.assert_array_equal(result.episode_index, [0, 0])
    np.testing.assert_array_equal(result.start_frame_index, [0, 2])
    np.testing.assert_array_equal(result.current_feature_row, [0, 1])
    np.testing.assert_array_equal(result.next_feature_row, [2, -1])
    first_mask = raw.intervention[0]
    np.testing.assert_array_equal(result.bc_anchor[0, first_mask], raw.executed_action[0, first_mask])
    np.testing.assert_array_equal(result.bc_anchor[0, ~first_mask], features.vla_reference[0, ~first_mask])
    second_mask = raw.intervention[1]
    np.testing.assert_array_equal(result.bc_anchor[1, second_mask], raw.executed_action[1, second_mask])
    np.testing.assert_array_equal(result.bc_anchor[1, ~second_mask], features.vla_reference[1, ~second_mask])
    assert result.executed_action.dtype == np.float32
    assert result.bc_anchor.dtype == np.float32
    assert result.reward.dtype == np.float32
    assert result.terminal.dtype == np.bool_


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_finalize_rejects_missing_or_extra_feature_keys(mode: str):
    batch, plan, raw, features = _batch_plan_raw_features()
    if mode == "missing":
        features = cache.FeatureTable(*(getattr(features, field.name)[:-1] for field in dataclasses.fields(features)))
    else:
        features = cache.FeatureTable(
            episode_index=np.append(features.episode_index, np.int32(0)),
            frame_index=np.append(features.frame_index, np.int32(21)),
            z_rl=np.concatenate([features.z_rl, features.z_rl[:1]]),
            state_norm=np.concatenate([features.state_norm, features.state_norm[:1]]),
            vla_reference=np.concatenate([features.vla_reference, features.vla_reference[:1]]),
        )
    with pytest.raises(cache.CacheError, match="feature.*exactly"):
        cache.finalize_transition_table(batch, plan, raw, features)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("episode_index", np.array([0, 1], dtype=np.int32), "episode_index"),
        ("start_frame_index", np.array([2, 0], dtype=np.int32), "start_frame_index"),
        ("reward", np.array([[1], [3]], dtype=np.float32), "reward"),
        ("terminal", np.array([[True], [False]], dtype=np.bool_), "terminal"),
    ],
)
def test_finalize_rejects_raw_plan_mismatch(
    field: str,
    replacement: np.ndarray,
    error: str,
):
    batch, plan, raw, features = _batch_plan_raw_features()
    raw = dataclasses.replace(raw, **{field: replacement})
    with pytest.raises(cache.CacheError, match=error):
        cache.finalize_transition_table(batch, plan, raw, features)


def test_finalize_rejects_batch_and_plan_identity_mismatch():
    batch, plan, raw, features = _batch_plan_raw_features()
    wrong_batch = dataclasses.replace(batch, batch_id="other")
    with pytest.raises(cache.CacheError, match="batch.*id|batch_id"):
        cache.finalize_transition_table(wrong_batch, plan, raw, features)

    wrong_row = dataclasses.replace(plan.rows[0], batch_id="other")
    wrong_plan = dataclasses.replace(plan, rows=(wrong_row, *plan.rows[1:]))
    with pytest.raises(cache.CacheError, match="batch.*id|batch_id"):
        cache.finalize_transition_table(batch, wrong_plan, raw, features)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("executed_action", np.ones((2, 20, 16), dtype=np.float64), "executed_action"),
        ("intervention", np.ones((2, 20), dtype=np.int8), "intervention"),
        ("reward", np.ones((2, 1), dtype=np.float64), "reward"),
        ("terminal", np.ones((2, 1), dtype=np.int8), "terminal"),
    ],
)
def test_finalize_validates_raw_table(
    field: str,
    replacement: np.ndarray,
    error: str,
):
    batch, plan, raw, features = _batch_plan_raw_features()
    raw = dataclasses.replace(raw, **{field: replacement})
    with pytest.raises(cache.CacheError, match=error):
        cache.finalize_transition_table(batch, plan, raw, features)


def test_finalize_does_not_modify_any_input_array():
    batch, plan, raw, features = _batch_plan_raw_features()
    raw_before = {field.name: getattr(raw, field.name).copy() for field in dataclasses.fields(raw)}
    features_before = {field.name: getattr(features, field.name).copy() for field in dataclasses.fields(features)}

    result = cache.finalize_transition_table(batch, plan, raw, features)
    result.executed_action[0, 0, 0] = 999
    result.bc_anchor[0, 0, 0] = 999

    for field in dataclasses.fields(raw):
        np.testing.assert_array_equal(getattr(raw, field.name), raw_before[field.name])
    for field in dataclasses.fields(features):
        np.testing.assert_array_equal(getattr(features, field.name), features_before[field.name])


def test_ready_batch_to_replay_snapshot_end_to_end(tmp_path: Path):
    batch_root = build_ready_batch(
        tmp_path / "batch_000001_integration",
        lengths=(23, *((22,) * 19)),
    )
    input_hashes = {
        path.relative_to(batch_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(batch_root.rglob("*"))
        if path.is_file()
    }
    expected_batch_id = "batch_000001_integration"
    expected_feature_identity = "synthetic-feature-v1"
    expected_manifest_sha256 = input_hashes["migration_manifest.json"]
    expected_labels_sha256 = input_hashes["meta/tristate_labels.json"]
    expected_transitions = tuple(
        (
            episode_index,
            start,
            None if start == (3 if episode_index == 0 else 2) else start + 20,
            2.0 if episode_index == 0 and start == 3 else 0.0,
            start == (3 if episode_index == 0 else 2),
        )
        for episode_index in range(20)
        for start in ((0, 2, 3) if episode_index == 0 else (0, 2))
    )
    expected_feature_pairs = (
        (0, 0),
        (0, 2),
        (0, 3),
        (0, 20),
        (0, 22),
        *(
            pair
            for episode_index in range(1, 20)
            for pair in (
                (episode_index, 0),
                (episode_index, 2),
                (episode_index, 20),
            )
        ),
    )
    expected_feature_keys = tuple(
        transitions.FeatureKey(expected_batch_id, episode_index, frame_index)
        for episode_index, frame_index in expected_feature_pairs
    )
    row_by_pair = {pair: row for row, pair in enumerate(expected_feature_pairs)}
    expected_episode = np.asarray(
        [episode_index for episode_index, _, _, _, _ in expected_transitions],
        dtype=np.int32,
    )
    expected_start = np.asarray(
        [start for _, start, _, _, _ in expected_transitions],
        dtype=np.int32,
    )
    expected_current_rows = np.asarray(
        [row_by_pair[(episode_index, start)] for episode_index, start, _, _, _ in expected_transitions],
        dtype=np.int64,
    )
    expected_next_rows = np.asarray(
        [
            -1 if next_frame is None else row_by_pair[(episode_index, next_frame)]
            for episode_index, _, next_frame, _, _ in expected_transitions
        ],
        dtype=np.int64,
    )
    expected_reward = np.asarray(
        [[reward] for _, _, _, reward, _ in expected_transitions],
        dtype=np.float32,
    )
    expected_terminal = np.asarray(
        [[terminal] for _, _, _, _, terminal in expected_transitions],
        dtype=np.bool_,
    )
    assert len(expected_transitions) == 41
    assert len(expected_feature_pairs) == 62
    assert expected_feature_pairs == tuple(sorted(expected_feature_pairs))
    video_calls: list[tuple[str, int, float]] = []

    def validate_synthetic_video(path: Path, expected_frames: int, tolerance_s: float) -> None:
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "frame_count": expected_frames,
        }
        assert tolerance_s == 0.05
        video_calls.append((path.relative_to(batch_root).as_posix(), expected_frames, tolerance_s))

    batch = admission.validate_ready_batch(
        batch_root,
        video_validator=validate_synthetic_video,
    )
    assert len(video_calls) == 20 * len(admission.VIDEO_KEYS)
    assert batch.batch_id == expected_batch_id
    assert batch.manifest_sha256 == expected_manifest_sha256
    assert batch.labels_sha256 == expected_labels_sha256
    assert batch.total_frames == 23 + 19 * 22
    assert batch.chunk_equivalents == 40
    assert [episode.length for episode in batch.episodes] == [23, *((22,) * 19)]

    admission_path = admission.publish_admission(
        batch,
        tmp_path / "training",
        round_id="round_000001",
        admitted_at="2026-07-24T09:30:00+08:00",
        code_commit="a" * 40,
    )
    admission.verify_admission(admission_path, batch)
    admission_sha256 = hashlib.sha256(admission_path.read_bytes()).hexdigest()

    labels_before = tuple(episode.labels.copy() for episode in batch.episodes)
    intervention_before = tuple(episode.intervention.copy() for episode in batch.episodes)
    plan = transitions.build_transition_plan(batch)
    assert plan.chunk_equivalents == 40
    assert plan.feature_keys == expected_feature_keys
    assert len(plan.rows) == len(expected_transitions)
    for row, (episode_index, start, next_frame, reward, terminal) in zip(
        plan.rows,
        expected_transitions,
        strict=True,
    ):
        assert row.batch_id == expected_batch_id
        assert row.episode_index == episode_index
        assert row.start_frame_index == start
        assert row.current_key == transitions.FeatureKey(expected_batch_id, episode_index, start)
        assert row.next_key == (
            None if next_frame is None else transitions.FeatureKey(expected_batch_id, episode_index, next_frame)
        )
        assert row.next_frame_index == next_frame
        assert row.reward == reward
        assert row.terminal is terminal

    zeros = np.zeros(16, dtype=np.float32)
    ones = np.ones(16, dtype=np.float32)
    normalizer = transitions.Stage2Normalizer.from_norm_stats(
        {
            "state": openpi_transforms.NormStats(
                mean=zeros,
                std=ones,
                q01=zeros,
                q99=np.full(16, 30.0, dtype=np.float32),
            ),
            "actions": openpi_transforms.NormStats(
                mean=zeros,
                std=ones,
                q01=zeros,
                q99=np.full(16, 20.0, dtype=np.float32),
            ),
        }
    )
    raw = transitions.build_raw_transition_table(batch, plan, normalizer)
    expected_action_window = np.arange(1, 21, dtype=np.float32)[:, None] / np.float32(20.0 + 1e-6) * np.float32(
        2.0
    ) - np.float32(1.0)
    expected_action_window = np.broadcast_to(expected_action_window, (20, 16))
    expected_executed = np.ascontiguousarray(
        np.broadcast_to(
            expected_action_window,
            (len(expected_transitions), 20, 16),
        )
    )
    expected_intervention = np.zeros((len(expected_transitions), 20), dtype=np.bool_)
    for row, (episode_index, start, _, _, _) in enumerate(expected_transitions):
        if episode_index == 0:
            for absolute_frame in (3, 4):
                relative_frame = absolute_frame - start
                if 0 <= relative_frame < 20:
                    expected_intervention[row, relative_frame] = True
    np.testing.assert_array_equal(raw.episode_index, expected_episode)
    np.testing.assert_array_equal(raw.start_frame_index, expected_start)
    np.testing.assert_allclose(
        raw.executed_action,
        expected_executed,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(raw.intervention, expected_intervention)
    np.testing.assert_array_equal(raw.reward, expected_reward)
    np.testing.assert_array_equal(raw.terminal, expected_terminal)
    assert raw.episode_index.dtype == np.int32
    assert raw.start_frame_index.dtype == np.int32
    assert raw.executed_action.dtype == np.float32
    assert raw.intervention.dtype == np.bool_
    assert raw.reward.dtype == np.float32
    assert raw.terminal.dtype == np.bool_
    assert [np.flatnonzero(expected_intervention[index]).tolist() for index in range(3)] == [
        [3, 4],
        [1, 2],
        [0, 1],
    ]

    feature_episode = np.asarray(
        [episode_index for episode_index, _ in expected_feature_pairs],
        dtype=np.int32,
    )
    feature_frame = np.asarray(
        [frame_index for _, frame_index in expected_feature_pairs],
        dtype=np.int32,
    )
    z_rl = np.stack(
        [
            np.full(256, episode_index + frame_index / 32.0, dtype=np.float32)
            for episode_index, frame_index in expected_feature_pairs
        ]
    ).astype(ml_dtypes.bfloat16)
    expected_state_norm = np.stack(
        [
            np.full(
                16,
                frame_index / np.float32(30.0 + 1e-6) * np.float32(2.0) - np.float32(1.0),
                dtype=np.float32,
            )
            for _, frame_index in expected_feature_pairs
        ]
    )
    state_norm = np.stack(
        [normalizer.state(np.full(16, frame_index, dtype=np.float32)) for _, frame_index in expected_feature_pairs]
    ).astype(np.float32)
    np.testing.assert_allclose(
        state_norm,
        expected_state_norm,
        rtol=1e-6,
        atol=1e-6,
    )
    reference_grid = np.arange(20 * 16, dtype=np.float32).reshape(20, 16) / np.float32(1000.0)
    vla_reference = np.stack(
        [
            reference_grid + np.float32(episode_index) + np.float32(frame_index) / np.float32(100.0)
            for episode_index, frame_index in expected_feature_pairs
        ]
    ).astype(np.float32)
    features = cache.FeatureTable(
        episode_index=np.ascontiguousarray(feature_episode),
        frame_index=np.ascontiguousarray(feature_frame),
        z_rl=np.ascontiguousarray(z_rl),
        state_norm=np.ascontiguousarray(state_norm),
        vla_reference=np.ascontiguousarray(vla_reference),
    )
    assert list(zip(features.episode_index.tolist(), features.frame_index.tolist(), strict=True)) == [
        (key.episode_index, key.frame_index) for key in expected_feature_keys
    ]

    raw_before = {field.name: getattr(raw, field.name).copy() for field in dataclasses.fields(raw)}
    features_before = {field.name: getattr(features, field.name).copy() for field in dataclasses.fields(features)}
    expected_reference = vla_reference[expected_current_rows]
    expected_anchor = np.where(
        expected_intervention[:, :, None],
        expected_executed,
        expected_reference,
    ).astype(np.float32)
    transition_table = cache.finalize_transition_table(batch, plan, raw, features)
    np.testing.assert_array_equal(transition_table.episode_index, expected_episode)
    np.testing.assert_array_equal(transition_table.start_frame_index, expected_start)
    np.testing.assert_array_equal(
        transition_table.current_feature_row,
        expected_current_rows,
    )
    np.testing.assert_array_equal(
        transition_table.next_feature_row,
        expected_next_rows,
    )
    np.testing.assert_allclose(
        transition_table.executed_action,
        expected_executed,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transition_table.bc_anchor,
        expected_anchor,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(transition_table.reward, expected_reward)
    np.testing.assert_array_equal(transition_table.terminal, expected_terminal)
    transition_before = {
        field.name: getattr(transition_table, field.name).copy() for field in dataclasses.fields(transition_table)
    }

    shard_root = tmp_path / "feature_cache" / expected_feature_identity / expected_batch_id
    published_manifest = cache.publish_shard(
        shard_root,
        features=features,
        transitions=transition_table,
            identity_fields={
                "feature_identity": expected_feature_identity,
                "batch_id": expected_batch_id,
                "migration_manifest_sha256": expected_manifest_sha256,
                "labels_sha256": expected_labels_sha256,
                "admission_sha256": admission_sha256,
                "config_name": "rl_token_stage1",
                "stage1_config": "rl_token_stage1",
                "stage2_config": "rl_token_stage2",
                "stage1_checkpoint_step": 54999,
                "reward_source": "tristate",
                "reward_label_values": [-1, 0, 1, 2],
                "completion_label": 2,
                "reward_aggregation": "sum_20_frames",
                "reward_schema_version": 1,
                "tristate_labels_sha256": expected_labels_sha256,
            },
    )
    artifact_oracles = (
        ("features/episode_index.npy", feature_episode),
        ("features/frame_index.npy", feature_frame),
        ("features/state_norm.npy", expected_state_norm),
        ("features/vla_reference.npy", vla_reference),
        ("features/z_rl.npy", z_rl),
        ("transitions/bc_anchor.npy", expected_anchor),
        ("transitions/current_feature_row.npy", expected_current_rows),
        ("transitions/episode_index.npy", expected_episode),
        ("transitions/executed_action.npy", expected_executed),
        ("transitions/next_feature_row.npy", expected_next_rows),
        ("transitions/reward.npy", expected_reward),
        ("transitions/start_frame_index.npy", expected_start),
        ("transitions/terminal.npy", expected_terminal),
    )
    assert [relative for relative, _ in artifact_oracles] == sorted(relative for relative, _ in artifact_oracles)
    expected_file_records = []
    approximate_artifacts = {
        "features/state_norm.npy",
        "transitions/bc_anchor.npy",
        "transitions/executed_action.npy",
    }
    for relative, oracle in artifact_oracles:
        artifact_path = shard_root / relative
        artifact_bytes = artifact_path.read_bytes()
        with artifact_path.open("rb") as stream:
            artifact = np.load(stream, allow_pickle=False)
        if relative == "features/z_rl.npy":
            assert artifact.dtype == np.dtype("V2")
            artifact = artifact.view(ml_dtypes.bfloat16)
        assert artifact.shape == oracle.shape
        assert artifact.dtype == oracle.dtype
        if relative in approximate_artifacts:
            np.testing.assert_allclose(
                artifact,
                oracle,
                rtol=1e-6,
                atol=1e-6,
            )
        else:
            np.testing.assert_array_equal(artifact, oracle)
        expected_file_records.append(
            {
                "path": relative,
                "size": artifact_path.stat().st_size,
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "shape": list(oracle.shape),
                "dtype": str(oracle.dtype),
            }
        )
    expected_manifest = {
        "schema_version": 1,
        "feature_identity": expected_feature_identity,
        "batch_id": expected_batch_id,
        "migration_manifest_sha256": expected_manifest_sha256,
            "labels_sha256": expected_labels_sha256,
            "admission_sha256": admission_sha256,
            "config_name": "rl_token_stage1",
            "stage1_config": "rl_token_stage1",
            "stage2_config": "rl_token_stage2",
            "stage1_checkpoint_step": 54999,
            "reward_source": "tristate",
            "reward_label_values": [-1, 0, 1, 2],
            "completion_label": 2,
            "reward_aggregation": "sum_20_frames",
            "reward_schema_version": 1,
            "tristate_labels_sha256": expected_labels_sha256,
        "feature_rows": len(expected_feature_pairs),
        "transition_rows": len(expected_transitions),
        "files": expected_file_records,
    }
    expected_manifest_bytes = (
        json.dumps(
            expected_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path = shard_root / "manifest.json"
    actual_manifest_bytes = manifest_path.read_bytes()
    actual_manifest = json.loads(actual_manifest_bytes.decode("utf-8"))
    assert published_manifest == expected_manifest
    assert actual_manifest == expected_manifest
    assert actual_manifest_bytes == expected_manifest_bytes
    expected_cache_manifest_sha256 = hashlib.sha256(expected_manifest_bytes).hexdigest()
    assert hashlib.sha256(actual_manifest_bytes).hexdigest() == expected_cache_manifest_sha256

    opened = cache.open_shard(shard_root)
    assert opened.manifest == expected_manifest
    assert opened.root == shard_root.resolve()
    assert opened.manifest["feature_identity"] == expected_feature_identity
    assert opened.manifest["batch_id"] == expected_batch_id
    assert opened.manifest["migration_manifest_sha256"] == expected_manifest_sha256
    assert opened.manifest["labels_sha256"] == expected_labels_sha256
    assert opened.manifest["admission_sha256"] == admission_sha256
    assert opened.manifest["feature_rows"] == len(expected_feature_pairs)
    assert opened.manifest["transition_rows"] == len(expected_transitions)
    np.testing.assert_array_equal(opened.features.episode_index, feature_episode)
    np.testing.assert_array_equal(opened.features.frame_index, feature_frame)
    np.testing.assert_array_equal(opened.features.z_rl, z_rl)
    np.testing.assert_allclose(
        opened.features.state_norm,
        expected_state_norm,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(opened.features.vla_reference, vla_reference)
    assert opened.features.episode_index.dtype == np.int32
    assert opened.features.frame_index.dtype == np.int32
    assert opened.features.z_rl.dtype == ml_dtypes.bfloat16
    assert opened.features.state_norm.dtype == np.float32
    assert opened.features.vla_reference.dtype == np.float32
    assert opened.transitions.episode_index.dtype == np.int32
    assert opened.transitions.start_frame_index.dtype == np.int32
    assert opened.transitions.current_feature_row.dtype == np.int64
    assert opened.transitions.next_feature_row.dtype == np.int64
    assert opened.transitions.executed_action.dtype == np.float32
    assert opened.transitions.bc_anchor.dtype == np.float32
    assert opened.transitions.reward.dtype == np.float32
    assert opened.transitions.terminal.dtype == np.bool_
    for table in (opened.features, opened.transitions):
        for field in dataclasses.fields(table):
            value = getattr(table, field.name)
            assert value.flags.c_contiguous
            assert not value.flags.writeable

    np.testing.assert_array_equal(opened.transitions.episode_index, expected_episode)
    np.testing.assert_array_equal(opened.transitions.start_frame_index, expected_start)
    np.testing.assert_array_equal(
        opened.transitions.current_feature_row,
        expected_current_rows,
    )
    np.testing.assert_array_equal(
        opened.transitions.next_feature_row,
        expected_next_rows,
    )
    np.testing.assert_array_equal(
        opened.transitions.reward,
        expected_reward,
    )
    np.testing.assert_array_equal(
        opened.transitions.terminal,
        expected_terminal,
    )
    np.testing.assert_allclose(
        opened.transitions.executed_action,
        expected_executed,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        opened.transitions.bc_anchor,
        expected_anchor,
        rtol=1e-6,
        atol=1e-6,
    )

    snapshot_path = tmp_path / "replay" / "round_000001.json"
    expected_snapshot_record = {
        "batch_id": expected_batch_id,
        "root": str(shard_root.resolve()),
        "admission_sha256": admission_sha256,
            "cache_manifest_sha256": expected_cache_manifest_sha256,
            "tristate_labels_sha256": expected_labels_sha256,
        "transition_rows": len(expected_transitions),
        "start": 0,
        "end": len(expected_transitions),
    }
    expected_snapshot_payload = {
            "schema_version": 1,
            "feature_identity": expected_feature_identity,
            "stage1_config": "rl_token_stage1",
            "stage2_config": "rl_token_stage2",
            "reward_source": "tristate",
            "reward_label_values": [-1, 0, 1, 2],
            "completion_label": 2,
            "reward_aggregation": "sum_20_frames",
            "reward_schema_version": 1,
            "total_transitions": len(expected_transitions),
        "shards": [expected_snapshot_record],
    }
    expected_snapshot_bytes = (
        json.dumps(
            expected_snapshot_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    expected_snapshot_sha256 = hashlib.sha256(expected_snapshot_bytes).hexdigest()
    created_snapshot = replay.create_snapshot(
        snapshot_path,
        previous=None,
        new_shard=opened,
        admission_sha256=admission_sha256,
    )
    actual_snapshot_bytes = snapshot_path.read_bytes()
    actual_snapshot_payload = json.loads(actual_snapshot_bytes.decode("utf-8"))
    assert actual_snapshot_payload == expected_snapshot_payload
    assert actual_snapshot_bytes == expected_snapshot_bytes
    assert hashlib.sha256(actual_snapshot_bytes).hexdigest() == expected_snapshot_sha256
    assert created_snapshot.path == snapshot_path.resolve()
    assert created_snapshot.schema_version == 1
    assert created_snapshot.feature_identity == expected_feature_identity
    assert created_snapshot.total_transitions == len(expected_transitions)
    assert created_snapshot.sha256 == expected_snapshot_sha256
    assert len(created_snapshot.shards) == 1
    assert dict(created_snapshot.shards[0]) == expected_snapshot_record

    reopened_snapshot = replay.open_snapshot(snapshot_path)
    assert reopened_snapshot.path == snapshot_path.resolve()
    assert reopened_snapshot.schema_version == 1
    assert reopened_snapshot.feature_identity == expected_feature_identity
    assert reopened_snapshot.total_transitions == len(expected_transitions)
    assert reopened_snapshot.sha256 == expected_snapshot_sha256
    assert len(reopened_snapshot.shards) == 1
    assert dict(reopened_snapshot.shards[0]) == expected_snapshot_record
    replay_buffer = replay.ReplayBuffer.open(snapshot_path)
    assert replay_buffer.total_transitions == len(expected_transitions)
    global_indices = np.random.default_rng(7).permutation(len(expected_transitions)).astype(np.int64)
    gathered = replay_buffer.gather(global_indices)
    np.testing.assert_array_equal(gathered.source_global_index, global_indices)

    current_rows = expected_current_rows[global_indices]
    next_rows = expected_next_rows[global_indices]
    np.testing.assert_array_equal(gathered.z_rl, z_rl[current_rows])
    np.testing.assert_allclose(
        gathered.state_norm,
        expected_state_norm[current_rows],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(gathered.vla_reference, vla_reference[current_rows])
    np.testing.assert_allclose(
        gathered.executed_action,
        expected_executed[global_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        gathered.bc_anchor,
        expected_anchor[global_indices],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(gathered.reward, expected_reward[global_indices])
    np.testing.assert_array_equal(gathered.terminal, expected_terminal[global_indices])

    nonterminal = next_rows >= 0
    expected_next_z_rl = np.zeros(
        (len(expected_transitions), 256),
        dtype=ml_dtypes.bfloat16,
    )
    expected_next_state_norm = np.zeros(
        (len(expected_transitions), 16),
        dtype=np.float32,
    )
    expected_next_vla_reference = np.zeros(
        (len(expected_transitions), 20, 16),
        dtype=np.float32,
    )
    expected_next_z_rl[nonterminal] = z_rl[next_rows[nonterminal]]
    expected_next_state_norm[nonterminal] = expected_state_norm[next_rows[nonterminal]]
    expected_next_vla_reference[nonterminal] = vla_reference[next_rows[nonterminal]]
    np.testing.assert_array_equal(gathered.next_z_rl, expected_next_z_rl)
    np.testing.assert_allclose(
        gathered.next_state_norm,
        expected_next_state_norm,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        gathered.next_vla_reference,
        expected_next_vla_reference,
    )
    assert np.count_nonzero(gathered.next_z_rl[~nonterminal]) == 0
    assert np.count_nonzero(gathered.next_state_norm[~nonterminal]) == 0
    assert np.count_nonzero(gathered.next_vla_reference[~nonterminal]) == 0
    np.testing.assert_array_equal(~nonterminal, gathered.terminal[:, 0])
    assert gathered.z_rl.dtype == ml_dtypes.bfloat16
    assert gathered.next_z_rl.dtype == ml_dtypes.bfloat16
    assert gathered.state_norm.dtype == np.float32
    assert gathered.next_state_norm.dtype == np.float32
    assert gathered.vla_reference.dtype == np.float32
    assert gathered.next_vla_reference.dtype == np.float32
    assert gathered.executed_action.dtype == np.float32
    assert gathered.bc_anchor.dtype == np.float32
    assert gathered.reward.dtype == np.float32
    assert gathered.terminal.dtype == np.bool_
    assert gathered.source_global_index.dtype == np.int64
    for field in dataclasses.fields(gathered):
        assert getattr(gathered, field.name).flags.c_contiguous

    np.testing.assert_allclose(
        transition_table.bc_anchor[0, [3, 4]],
        expected_executed[0, [3, 4]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transition_table.bc_anchor[1, [1, 2]],
        expected_executed[1, [1, 2]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transition_table.bc_anchor[2, [0, 1]],
        expected_executed[2, [0, 1]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        raw.reward[:3, 0],
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
    )

    for episode, expected_labels, expected_intervention in zip(
        batch.episodes,
        labels_before,
        intervention_before,
        strict=True,
    ):
        np.testing.assert_array_equal(episode.labels, expected_labels)
        np.testing.assert_array_equal(episode.intervention, expected_intervention)
    for field in dataclasses.fields(raw):
        np.testing.assert_array_equal(getattr(raw, field.name), raw_before[field.name])
    for field in dataclasses.fields(features):
        np.testing.assert_array_equal(getattr(features, field.name), features_before[field.name])
    for field in dataclasses.fields(transition_table):
        np.testing.assert_array_equal(
            getattr(transition_table, field.name),
            transition_before[field.name],
        )
    assert {
        path.relative_to(batch_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(batch_root.rglob("*"))
        if path.is_file()
    } == input_hashes
