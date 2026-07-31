from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import tokenizer as tokenizer_api
from openpi.training.rl_token.stage2 import identity

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class FeatureIdentityInput:
    checkpoint_sha256: str
    norm_stats_sha256: str
    model_config: dict[str, Any]
    transform_config: dict[str, Any]
    sampler_num_steps: int
    seed_version: int
    code_commit: str


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_sha256(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a lowercase SHA-256 hex string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _validate_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact positive integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def canonical_config_value(value: Any) -> Any:
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("floating-point config values must be finite")
        return value
    if type(value) is str:
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _qualified_type(value),
            "fields": {
                field.name: canonical_config_value(getattr(value, field.name)) for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, tokenizer_api.PaligemmaTokenizer):
        return {"type": _qualified_type(value)}
    if isinstance(value, tuple | list):
        return [canonical_config_value(item) for item in value]
    if isinstance(value, dict):
        canonical_items: list[tuple[str, Any]] = []
        seen_keys: set[str] = set()
        for key, item in value.items():
            if type(key) not in (str, int):
                raise TypeError("config mapping keys must be exact strings or integers")
            canonical_key = str(key)
            if canonical_key in seen_keys:
                raise ValueError(f"config mapping key collision after string conversion: {canonical_key!r}")
            seen_keys.add(canonical_key)
            canonical_items.append((canonical_key, canonical_config_value(item)))
        return dict(sorted(canonical_items))
    raise TypeError(f"unsupported value in feature identity: {_qualified_type(value)}")


def transform_signature(value: Any) -> dict[str, Any]:
    signature = canonical_config_value(value)
    if not isinstance(signature, dict):
        raise TypeError("transform signature root must be a dataclass or mapping")
    return signature


def build_feature_identity(value: FeatureIdentityInput) -> str:
    if not isinstance(value, FeatureIdentityInput):
        raise TypeError("value must be a FeatureIdentityInput")
    checkpoint_sha256 = _validate_sha256(value.checkpoint_sha256, "checkpoint_sha256")
    norm_stats_sha256 = _validate_sha256(value.norm_stats_sha256, "norm_stats_sha256")
    if not isinstance(value.model_config, dict):
        raise TypeError("model_config must be a mapping")
    if not isinstance(value.transform_config, dict):
        raise TypeError("transform_config must be a mapping")
    sampler_num_steps = _validate_positive_int(value.sampler_num_steps, "sampler_num_steps")
    seed_version = _validate_positive_int(value.seed_version, "seed_version")
    if type(value.code_commit) is not str:
        raise TypeError("code_commit must be a nonempty string")
    if not value.code_commit.strip():
        raise ValueError("code_commit must be a nonempty string")
    payload = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "norm_stats_sha256": norm_stats_sha256,
        "model_config": canonical_config_value(value.model_config),
        "transform_config": canonical_config_value(value.transform_config),
        "sampler_num_steps": sampler_num_steps,
        "seed_version": seed_version,
        "code_commit": value.code_commit,
    }
    return identity.sha256_json(payload)


def _validate_nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a nonempty string")
    if not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _validate_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact nonnegative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def frame_key(
    feature_identity: str,
    batch_id: str,
    episode_index: int,
    frame_index: int,
) -> jax.Array:
    feature_identity = _validate_nonempty_string(feature_identity, "feature_identity")
    batch_id = _validate_nonempty_string(batch_id, "batch_id")
    episode_index = _validate_nonnegative_int(episode_index, "episode_index")
    frame_index = _validate_nonnegative_int(frame_index, "frame_index")
    digest = hashlib.sha256(
        identity.canonical_json_bytes(
            {
                "feature_identity": feature_identity,
                "batch_id": batch_id,
                "episode_index": episode_index,
                "frame_index": frame_index,
            }
        )
    ).digest()
    words = np.frombuffer(digest[:8], dtype="<u4").copy()
    return jax.random.wrap_key_data(jnp.asarray(words, dtype=jnp.uint32))


def _canonical_path(path: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": _qualified_type(component),
            "value": canonical_config_value(component),
        }
        for component in path
    ]


def parameter_tree_sha256(state: nnx.State) -> str:
    if not isinstance(state, nnx.State):
        raise TypeError("state must be an nnx.State")
    leaves = []
    for path, variable in state.filter(nnx.Param).flat_state().items():
        canonical_path = _canonical_path(path)
        path_bytes = identity.canonical_json_bytes({"path": canonical_path})
        leaves.append((path_bytes, canonical_path, variable))
    if not leaves:
        raise ValueError("state must contain at least one nnx.Param")
    digest = hashlib.sha256()
    for _path_bytes, canonical_path, variable in sorted(leaves, key=lambda item: item[0]):
        value = np.asarray(jax.device_get(variable.value))
        digest.update(
            identity.canonical_json_bytes(
                {
                    "path": canonical_path,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            )
        )
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _stat_snapshot(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_size,
        value.st_dev,
        value.st_ino,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _raise_walk_error(error: OSError) -> None:
    raise error


def _hash_checkpoint_file(
    path: Path,
    expected_snapshot: tuple[int, int, int, int, int],
) -> tuple[int, bytes]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"checkpoint file changed or could not be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"checkpoint entry is not a regular file: {path}")
        before_snapshot = _stat_snapshot(before)
        if before_snapshot != expected_snapshot:
            raise ValueError(f"checkpoint file changed between traversal and open: {path}")

        file_digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, _CHECKPOINT_HASH_CHUNK_BYTES)
            if not chunk:
                break
            bytes_read += len(chunk)
            file_digest.update(chunk)

        after = os.fstat(descriptor)
        if not stat.S_ISREG(after.st_mode):
            raise ValueError(f"checkpoint entry stopped being a regular file while hashing: {path}")
        if _stat_snapshot(after) != before_snapshot or bytes_read != before.st_size:
            raise ValueError(f"checkpoint file changed while hashing; stable snapshot unavailable: {path}")

        try:
            final_path_stat = path.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"checkpoint file changed before identity verification completed: {path}") from error
        if not stat.S_ISREG(final_path_stat.st_mode) or _stat_snapshot(final_path_stat) != before_snapshot:
            raise ValueError(f"checkpoint file changed before identity verification completed: {path}")
        return before.st_size, file_digest.digest()
    finally:
        os.close(descriptor)


def checkpoint_tree_sha256(root: Path) -> str:
    unresolved = Path(root)
    if unresolved.is_symlink():
        raise ValueError(f"checkpoint root must not be a symlink: {unresolved}")
    if not unresolved.exists():
        raise FileNotFoundError(f"checkpoint root does not exist: {unresolved}")
    if not unresolved.is_dir():
        raise NotADirectoryError(f"checkpoint root is not a directory: {unresolved}")
    resolved = unresolved.resolve(strict=True)

    files: list[tuple[str, Path, tuple[int, int, int, int, int]]] = []
    for directory_name, directory_names, file_names in os.walk(
        resolved,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        directory = Path(directory_name)
        for name in directory_names:
            path = directory / name
            path_stat = path.stat(follow_symlinks=False)
            if stat.S_ISLNK(path_stat.st_mode):
                raise ValueError(f"checkpoint must not contain symlink: {path}")
            if not stat.S_ISDIR(path_stat.st_mode):
                raise ValueError(f"checkpoint traversal entry is not a directory: {path}")
        for name in file_names:
            path = directory / name
            path_stat = path.stat(follow_symlinks=False)
            if stat.S_ISLNK(path_stat.st_mode):
                raise ValueError(f"checkpoint must not contain symlink: {path}")
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValueError(f"checkpoint must contain only regular files and directories: {path}")
            files.append((path.relative_to(resolved).as_posix(), path, _stat_snapshot(path_stat)))

    if not files:
        raise ValueError(f"checkpoint root must contain at least one regular file; directory is empty: {unresolved}")

    digest = hashlib.sha256()
    for relative, path, expected_snapshot in sorted(files):
        size, file_digest = _hash_checkpoint_file(path, expected_snapshot)
        digest.update(identity.canonical_json_bytes({"path": relative, "size": size}))
        digest.update(file_digest)
    return digest.hexdigest()
