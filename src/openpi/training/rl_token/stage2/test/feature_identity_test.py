from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import tokenizer as tokenizer_api
from openpi.training.rl_token.stage2 import feature_identity
from openpi.training.rl_token.stage2 import identity


@dataclasses.dataclass(frozen=True)
class _NestedConfig:
    width: int
    enabled: bool


def _identity_input(**updates) -> feature_identity.FeatureIdentityInput:
    values = {
        "checkpoint_sha256": "1" * 64,
        "norm_stats_sha256": "2" * 64,
        "model_config": {
            "action_horizon": 50,
            "nested": _NestedConfig(width=32, enabled=True),
        },
        "transform_config": {
            "use_quantile_norm": True,
            "default_prompt": "fold clothes",
        },
        "sampler_num_steps": 10,
        "seed_version": 1,
        "code_commit": "abc123",
    }
    values.update(updates)
    return feature_identity.FeatureIdentityInput(**values)


def test_feature_identity_input_is_frozen():
    value = _identity_input()

    with pytest.raises(dataclasses.FrozenInstanceError):
        value.seed_version = 2


def test_feature_identity_matches_canonical_schema_v1_payload():
    value = _identity_input(model_config={"action_horizon": 50})
    expected = identity.sha256_json(
        {
            "schema_version": 1,
            "checkpoint_sha256": "1" * 64,
            "norm_stats_sha256": "2" * 64,
            "model_config": {"action_horizon": 50},
            "transform_config": {
                "default_prompt": "fold clothes",
                "use_quantile_norm": True,
            },
            "sampler_num_steps": 10,
            "seed_version": 1,
            "code_commit": "abc123",
        }
    )

    assert feature_identity.build_feature_identity(value) == expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("checkpoint_sha256", "3" * 64),
        ("norm_stats_sha256", "4" * 64),
        ("model_config", {"action_horizon": 20}),
        (
            "transform_config",
            {"use_quantile_norm": True, "default_prompt": "pick up the block"},
        ),
        ("sampler_num_steps", 5),
        ("seed_version", 2),
        ("code_commit", "def456"),
    ],
)
def test_feature_identity_changes_when_any_input_changes(field: str, replacement):
    base = _identity_input()
    changed = dataclasses.replace(base, **{field: replacement})

    assert feature_identity.build_feature_identity(changed) != feature_identity.build_feature_identity(base)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("checkpoint_sha256", "a" * 63),
        ("checkpoint_sha256", "A" * 64),
        ("checkpoint_sha256", "g" * 64),
        ("checkpoint_sha256", 1),
        ("norm_stats_sha256", "0" * 65),
        ("norm_stats_sha256", "F" * 64),
        ("norm_stats_sha256", None),
    ],
)
def test_feature_identity_rejects_invalid_sha256_fields(field: str, bad_value):
    with pytest.raises((TypeError, ValueError), match=field):
        feature_identity.build_feature_identity(_identity_input(**{field: bad_value}))


@pytest.mark.parametrize("field", ["sampler_num_steps", "seed_version"])
@pytest.mark.parametrize("bad_value", [True, False, 0, -1, 1.0, "1", None])
def test_feature_identity_requires_exact_positive_integer_versions(field: str, bad_value):
    with pytest.raises((TypeError, ValueError), match=field):
        feature_identity.build_feature_identity(_identity_input(**{field: bad_value}))


@pytest.mark.parametrize("bad_value", ["", "   ", 123, None])
def test_feature_identity_requires_nonempty_code_commit(bad_value):
    with pytest.raises((TypeError, ValueError), match="code_commit"):
        feature_identity.build_feature_identity(_identity_input(code_commit=bad_value))


def test_canonical_config_value_supports_exact_scalars_sequences_dataclass_and_tokenizer():
    tokenizer = object.__new__(tokenizer_api.PaligemmaTokenizer)
    value = {
        "none": None,
        "bool": True,
        "int": 3,
        "float": 1.25,
        "str": "fold clothes",
        "tuple": (1, "two"),
        "list": [False, 4],
        "nested": _NestedConfig(width=16, enabled=False),
        "tokenizer": tokenizer,
    }

    canonical = feature_identity.canonical_config_value(value)

    assert canonical["none"] is None
    assert canonical["bool"] is True
    assert canonical["int"] == 3
    assert canonical["float"] == 1.25
    assert canonical["str"] == "fold clothes"
    assert canonical["tuple"] == [1, "two"]
    assert canonical["list"] == [False, 4]
    assert canonical["nested"] == {
        "type": f"{_NestedConfig.__module__}.{_NestedConfig.__qualname__}",
        "fields": {"width": 16, "enabled": False},
    }
    assert canonical["tokenizer"] == {
        "type": "openpi.models.tokenizer.PaligemmaTokenizer",
    }
    assert "0x" not in json.dumps(canonical, sort_keys=True)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_config_value_rejects_nonfinite_float(bad_value: float):
    with pytest.raises(ValueError, match="finite"):
        feature_identity.canonical_config_value(bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [
        b"bytes",
        {1, 2},
        np.int64(1),
        np.bool_("true"),
        {"valid": object()},
        {1.5: "bad key"},
    ],
)
def test_canonical_config_value_rejects_unsupported_values_and_keys(bad_value):
    with pytest.raises(TypeError, match="unsupported|keys"):
        feature_identity.canonical_config_value(bad_value)


def test_canonical_config_value_rejects_stringified_mapping_key_collision():
    with pytest.raises(ValueError, match="collision"):
        feature_identity.canonical_config_value({1: "integer", "1": "string"})


def test_canonical_config_value_sorts_string_and_integer_mapping_keys_canonically():
    first = feature_identity.canonical_config_value({"z": 0, 10: "ten", "2": "two"})
    second = feature_identity.canonical_config_value({"2": "two", 10: "ten", "z": 0})

    assert first == second == {"10": "ten", "2": "two", "z": 0}


def test_transform_signature_is_structural_and_address_free():
    first = feature_identity.transform_signature(
        transforms.Group(
            inputs=(
                transforms.ResizeImages(224, 224),
                transforms.PadStatesAndActions(32),
            )
        )
    )
    different_resize = feature_identity.transform_signature(
        transforms.Group(
            inputs=(
                transforms.ResizeImages(224, 256),
                transforms.PadStatesAndActions(32),
            )
        )
    )
    different_padding = feature_identity.transform_signature(
        transforms.Group(
            inputs=(
                transforms.ResizeImages(224, 224),
                transforms.PadStatesAndActions(16),
            )
        )
    )
    prompt_default = feature_identity.transform_signature(
        transforms.Group(inputs=(transforms.InjectDefaultPrompt("fold clothes"),))
    )
    other_prompt_default = feature_identity.transform_signature(
        transforms.Group(inputs=(transforms.InjectDefaultPrompt("pick up the block"),))
    )

    assert first != different_resize
    assert first != different_padding
    assert prompt_default != other_prompt_default
    assert "0x" not in json.dumps(first, sort_keys=True)


@pytest.mark.parametrize("value", [None, [transforms.ResizeImages(224, 224)], "transform"])
def test_transform_signature_requires_structured_root(value):
    with pytest.raises(TypeError, match="root"):
        feature_identity.transform_signature(value)


def test_frame_key_is_stable_and_matches_canonical_first_eight_digest_bytes():
    first = feature_identity.frame_key("feature", "batch", 2, 10)
    repeated = feature_identity.frame_key("feature", "batch", 2, 10)

    np.testing.assert_array_equal(jax.random.key_data(first), np.array([2419697375, 2880008483], dtype=np.uint32))
    np.testing.assert_array_equal(jax.random.key_data(first), jax.random.key_data(repeated))


@pytest.mark.parametrize(
    ("episode_index", "frame_index"),
    [(2, 11), (3, 10)],
)
def test_frame_key_changes_when_either_index_changes(episode_index: int, frame_index: int):
    baseline = feature_identity.frame_key("feature", "batch", 2, 10)
    changed = feature_identity.frame_key("feature", "batch", episode_index, frame_index)

    assert not np.array_equal(jax.random.key_data(baseline), jax.random.key_data(changed))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("feature_identity", ""),
        ("feature_identity", 1),
        ("batch_id", ""),
        ("batch_id", None),
        ("episode_index", True),
        ("episode_index", -1),
        ("episode_index", 1.0),
        ("frame_index", False),
        ("frame_index", -1),
        ("frame_index", "1"),
    ],
)
def test_frame_key_validates_nonempty_strings_and_exact_nonnegative_indices(field: str, bad_value):
    values = {
        "feature_identity": "feature",
        "batch_id": "batch",
        "episode_index": 2,
        "frame_index": 10,
    }
    values[field] = bad_value

    with pytest.raises((TypeError, ValueError), match=field):
        feature_identity.frame_key(**values)


def test_parameter_tree_sha256_is_path_order_stable_for_heterogeneous_components():
    left = nnx.State(
        {
            "1": nnx.Param(jnp.array([3.0], dtype=jnp.float32)),
            1: nnx.Param(jnp.array([2.0], dtype=jnp.float32)),
            "nested": {0: nnx.Param(jnp.array([1.0], dtype=jnp.float32))},
        }
    )
    right = nnx.State(
        {
            "nested": {0: nnx.Param(jnp.array([1.0], dtype=jnp.float32))},
            1: nnx.Param(jnp.array([2.0], dtype=jnp.float32)),
            "1": nnx.Param(jnp.array([3.0], dtype=jnp.float32)),
        }
    )

    first = feature_identity.parameter_tree_sha256(left)

    assert first == feature_identity.parameter_tree_sha256(right)
    assert len(first) == 64
    assert first == first.lower()
    int_path = nnx.State({1: nnx.Param(jnp.array([1.0], dtype=jnp.float32))})
    str_path = nnx.State({"1": nnx.Param(jnp.array([1.0], dtype=jnp.float32))})
    assert feature_identity.parameter_tree_sha256(int_path) != feature_identity.parameter_tree_sha256(str_path)


def test_parameter_tree_sha256_hashes_variable_state_from_real_nnx_module():
    first = nnx.state(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))
    repeated = nnx.state(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))
    changed = nnx.state(nnx.Linear(2, 3, rngs=nnx.Rngs(1)))
    empty_digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    first_digest = feature_identity.parameter_tree_sha256(first)

    assert first_digest != empty_digest
    assert first_digest == feature_identity.parameter_tree_sha256(repeated)
    assert first_digest != feature_identity.parameter_tree_sha256(changed)


@pytest.mark.parametrize(
    "state",
    [
        nnx.State({}),
        nnx.State({"stat": nnx.BatchStat(jnp.array([1.0], dtype=jnp.float32))}),
    ],
)
def test_parameter_tree_sha256_rejects_state_without_parameters(state: nnx.State):
    with pytest.raises(ValueError, match="nnx.Param"):
        feature_identity.parameter_tree_sha256(state)


@pytest.mark.parametrize(
    "changed",
    [
        nnx.State({"other": nnx.Param(jnp.array([1.0, 2.0], dtype=jnp.float32))}),
        nnx.State({"param": nnx.Param(jnp.array([1, 2], dtype=jnp.int32))}),
        nnx.State({"param": nnx.Param(jnp.array([[1.0, 2.0]], dtype=jnp.float32))}),
        nnx.State({"param": nnx.Param(jnp.array([1.0, 3.0], dtype=jnp.float32))}),
    ],
)
def test_parameter_tree_sha256_changes_with_path_dtype_shape_or_value(changed: nnx.State):
    baseline = nnx.State({"param": nnx.Param(jnp.array([1.0, 2.0], dtype=jnp.float32))})

    assert feature_identity.parameter_tree_sha256(changed) != feature_identity.parameter_tree_sha256(baseline)


def test_parameter_tree_sha256_ignores_non_parameter_variables():
    first = nnx.State(
        {
            "param": nnx.Param(jnp.array([1.0], dtype=jnp.float32)),
            "stat": nnx.BatchStat(jnp.array([2.0], dtype=jnp.float32)),
        }
    )
    second = nnx.State(
        {
            "param": nnx.Param(jnp.array([1.0], dtype=jnp.float32)),
            "stat": nnx.BatchStat(jnp.array([999.0], dtype=jnp.float32)),
        }
    )

    assert feature_identity.parameter_tree_sha256(first) == feature_identity.parameter_tree_sha256(second)


def test_parameter_tree_sha256_device_gets_one_parameter_leaf_at_a_time(monkeypatch: pytest.MonkeyPatch):
    state = nnx.State(
        {
            "a": nnx.Param(jnp.array([1.0], dtype=jnp.float32)),
            "b": nnx.Param(jnp.array([2.0], dtype=jnp.float32)),
            "ignored": nnx.BatchStat(jnp.array([3.0], dtype=jnp.float32)),
        }
    )
    real_device_get = feature_identity.jax.device_get
    seen = []

    def track_device_get(value):
        seen.append(value)
        return real_device_get(value)

    monkeypatch.setattr(feature_identity.jax, "device_get", track_device_get)

    feature_identity.parameter_tree_sha256(state)

    assert len(seen) == 2
    assert all(not isinstance(value, dict | nnx.State) for value in seen)


def test_checkpoint_tree_sha256_is_stable_and_changes_with_content_or_relative_path(tmp_path: Path):
    root = tmp_path / "checkpoint"
    nested = root / "params"
    nested.mkdir(parents=True)
    first_file = root / "metadata.json"
    second_file = nested / "weights.bin"
    first_file.write_bytes(b'{"step": 55000}\n')
    second_file.write_bytes(b"\x00\x01\x02")

    baseline = feature_identity.checkpoint_tree_sha256(root)

    assert baseline == feature_identity.checkpoint_tree_sha256(root)
    assert len(baseline) == 64
    second_file.write_bytes(b"\x00\x01\x03")
    assert feature_identity.checkpoint_tree_sha256(root) != baseline
    second_file.write_bytes(b"\x00\x01\x02")
    renamed = nested / "renamed.bin"
    second_file.rename(renamed)
    assert feature_identity.checkpoint_tree_sha256(root) != baseline


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_checkpoint_tree_sha256_rejects_internal_symlink(tmp_path: Path, kind: str):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "real.bin").write_bytes(b"weights")
    if kind == "file":
        (root / "link").symlink_to(root / "real.bin")
    else:
        target = tmp_path / "outside"
        target.mkdir()
        (target / "payload").write_bytes(b"outside")
        (root / "link").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        feature_identity.checkpoint_tree_sha256(root)


def test_checkpoint_tree_sha256_rejects_symlink_root(tmp_path: Path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "weights").write_bytes(b"weights")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="root.*symlink|symlink.*root"):
        feature_identity.checkpoint_tree_sha256(linked_root)


def test_checkpoint_tree_sha256_requires_real_nonempty_directory(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="missing"):
        feature_identity.checkpoint_tree_sha256(missing)

    regular_file = tmp_path / "file"
    regular_file.write_bytes(b"not a directory")
    with pytest.raises(NotADirectoryError, match="file"):
        feature_identity.checkpoint_tree_sha256(regular_file)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        feature_identity.checkpoint_tree_sha256(empty)


def test_checkpoint_tree_sha256_hashes_size_and_content_from_one_streamed_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    payload = root / "weights"
    chunk_size = 8 * 1024 * 1024
    content = b"A" * chunk_size + b"tail"
    payload.write_bytes(content)
    real_open = feature_identity.os.open
    real_read = feature_identity.os.read
    real_fstat = feature_identity.os.fstat
    opened_fds = []
    read_calls = []
    fstat_calls = []

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == payload:
            opened_fds.append(descriptor)
            assert flags & os.O_ACCMODE == os.O_RDONLY
            assert flags & os.O_CLOEXEC
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
        return descriptor

    def track_read(descriptor, size):
        if descriptor in opened_fds:
            read_calls.append((descriptor, size))
        return real_read(descriptor, size)

    def track_fstat(descriptor):
        if descriptor in opened_fds:
            fstat_calls.append(descriptor)
        return real_fstat(descriptor)

    def fail_read_bytes(_path):
        raise AssertionError("checkpoint hashing must stream files")

    def fail_sha256_file(_path):
        raise AssertionError("checkpoint hashing must not reopen files through identity.sha256_file")

    monkeypatch.setattr(feature_identity.os, "open", track_open)
    monkeypatch.setattr(feature_identity.os, "read", track_read)
    monkeypatch.setattr(feature_identity.os, "fstat", track_fstat)
    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    monkeypatch.setattr(feature_identity.identity, "sha256_file", fail_sha256_file)

    result = feature_identity.checkpoint_tree_sha256(root)

    file_digest = hashlib.sha256(content).digest()
    expected = hashlib.sha256(
        identity.canonical_json_bytes({"path": "weights", "size": len(content)}) + file_digest
    ).hexdigest()
    assert result == expected
    assert len(opened_fds) == 1
    assert fstat_calls == [opened_fds[0], opened_fds[0]]
    assert read_calls == [(opened_fds[0], chunk_size)] * 3


@pytest.mark.parametrize("replacement_kind", ["symlink", "regular"])
def test_checkpoint_tree_sha256_rejects_path_replaced_after_walk_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    payload = root / "weights"
    payload.write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.write_bytes(b"different")
    real_open = feature_identity.os.open
    replaced = False

    def replace_then_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if Path(path) == payload and not replaced:
            replaced = True
            payload.unlink()
            if replacement_kind == "symlink":
                payload.symlink_to(outside)
            else:
                outside.replace(payload)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(feature_identity.os, "open", replace_then_open)

    with pytest.raises(ValueError, match="changed|symlink|identity"):
        feature_identity.checkpoint_tree_sha256(root)

    assert replaced


def test_checkpoint_tree_sha256_rejects_path_replaced_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    payload = root / "weights"
    payload.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different")
    real_open = feature_identity.os.open
    real_read = feature_identity.os.read
    target_fds = set()
    replaced = False

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == payload:
            target_fds.add(descriptor)
        return descriptor

    def replace_at_eof(descriptor, size):
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if descriptor in target_fds and not chunk and not replaced:
            replaced = True
            payload.unlink()
            replacement.replace(payload)
        return chunk

    monkeypatch.setattr(feature_identity.os, "open", track_open)
    monkeypatch.setattr(feature_identity.os, "read", replace_at_eof)

    with pytest.raises(ValueError, match="changed|identity"):
        feature_identity.checkpoint_tree_sha256(root)

    assert replaced


def test_checkpoint_tree_sha256_rejects_same_size_in_place_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    payload = root / "weights"
    payload.write_bytes(b"A" * (8 * 1024 * 1024 + 1))
    real_open = feature_identity.os.open
    real_read = feature_identity.os.read
    target_fds = set()
    mutated = False

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == payload:
            target_fds.add(descriptor)
        return descriptor

    def mutate_after_first_chunk(descriptor, size):
        nonlocal mutated
        chunk = real_read(descriptor, size)
        if descriptor in target_fds and chunk and not mutated:
            writer = real_open(payload, os.O_WRONLY | os.O_CLOEXEC)
            try:
                os.pwrite(writer, b"Z", 0)
                os.fsync(writer)
            finally:
                os.close(writer)
            mutated = True
        return chunk

    monkeypatch.setattr(feature_identity.os, "open", track_open)
    monkeypatch.setattr(feature_identity.os, "read", mutate_after_first_chunk)

    with pytest.raises(ValueError, match="changed|stable"):
        feature_identity.checkpoint_tree_sha256(root)

    assert mutated


def test_checkpoint_tree_sha256_propagates_walk_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "weights").write_bytes(b"weights")

    def failing_walk(_root, *, topdown, onerror, followlinks):
        assert topdown is True
        assert followlinks is False
        assert onerror is not None
        onerror(PermissionError("walk denied"))
        if False:
            yield

    monkeypatch.setattr(feature_identity.os, "walk", failing_walk)

    with pytest.raises(PermissionError, match="walk denied"):
        feature_identity.checkpoint_tree_sha256(root)


def test_checkpoint_tree_sha256_closes_descriptor_after_repeated_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():
        pytest.skip("descriptor leak assertion requires Linux /proc")
    root = tmp_path / "checkpoint"
    root.mkdir()
    payload = root / "weights"
    payload.write_bytes(b"weights")
    real_open = feature_identity.os.open
    target_fds = set()

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == payload:
            target_fds.add(descriptor)
        return descriptor

    def fail_read(descriptor, _size):
        if descriptor in target_fds:
            raise OSError("injected read failure")
        raise AssertionError(f"unexpected descriptor read: {descriptor}")

    monkeypatch.setattr(feature_identity.os, "open", track_open)
    monkeypatch.setattr(feature_identity.os, "read", fail_read)
    before = len(os.listdir(proc_fds))

    for _ in range(32):
        with pytest.raises(OSError, match="injected read failure"):
            feature_identity.checkpoint_tree_sha256(root)

    assert len(os.listdir(proc_fds)) == before
