"""Deterministic, contract-safe selection of source migration batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math

from .models import SourceEpisode


class IncompatibleSourceError(ValueError):
    """A complete earliest batch spans incompatible source contracts."""


def _normalize_json(value: object) -> object:
    """Convert frozen JSON-compatible metadata to a canonical mutable shape."""
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("source contract metadata keys must be strings")
            normalized[key] = _normalize_json(value[key])
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source contract metadata numbers must be finite")
        return value
    raise ValueError(
        f"source contract metadata is not JSON-compatible: {type(value).__name__}"
    )


def _source_contract(episode: SourceEpisode) -> dict[str, object]:
    info = episode.info
    features = info.get("features")
    normalized_features = _normalize_json(features)
    declared_video_roles: list[str] = []
    if isinstance(features, Mapping):
        declared_video_roles = sorted(
            key
            for key, description in features.items()
            if isinstance(key, str)
            and isinstance(description, Mapping)
            and description.get("dtype") == "video"
        )

    contract: dict[str, object] = {
        "fps": _normalize_json(info.get("fps")),
        "robot_type": _normalize_json(info.get("robot_type")),
        "features": normalized_features,
        "declared_video_roles": declared_video_roles,
        "actual_video_roles": sorted(
            source_file.role
            for source_file in episode.files
            if source_file.role != "parquet"
        ),
    }
    if "codebase_version" in info:
        contract["codebase_version"] = _normalize_json(info["codebase_version"])
    return contract


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def contract_signature(episode: SourceEpisode) -> str:
    """Return a stable digest of the episode's normalized source contract."""
    payload = _canonical_json(_source_contract(episode)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_MISSING = object()


def _display(value: object) -> str:
    if value is _MISSING:
        return "<missing>"
    return _canonical_json(value)


def _contract_differences(
    old: object,
    new: object,
    *,
    path: str = "",
) -> list[str]:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        differences: list[str] = []
        for key in sorted(set(old) | set(new)):
            child_path = f"{path}.{key}" if path else str(key)
            old_value = old.get(key, _MISSING)
            new_value = new.get(key, _MISSING)
            if old_value is _MISSING or new_value is _MISSING:
                differences.append(
                    f"{child_path}: {_display(old_value)} -> {_display(new_value)}"
                )
            else:
                differences.extend(
                    _contract_differences(old_value, new_value, path=child_path)
                )
        return differences

    if type(old) is not type(new) or old != new:
        return [f"{path}: {_display(old)} -> {_display(new)}"]
    return []


def select_next_batch(
    episodes: Iterable[SourceEpisode],
    *,
    migrated: set[str],
    batch_size: int,
) -> tuple[SourceEpisode, ...]:
    """Select the single earliest complete compatible batch, if one exists."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")

    unique: list[SourceEpisode] = []
    seen: set[str] = set()
    for episode in sorted(episodes, key=lambda candidate: candidate.sort_key()):
        if episode.fingerprint in migrated or episode.fingerprint in seen:
            continue
        seen.add(episode.fingerprint)
        unique.append(episode)

    if len(unique) < batch_size:
        return ()

    selected = tuple(unique[:batch_size])
    expected = _source_contract(selected[0])
    for episode in selected[1:]:
        actual = _source_contract(episode)
        differences = _contract_differences(expected, actual)
        if differences:
            detail = "; ".join(differences)
            raise IncompatibleSourceError(
                "incompatible source contract between "
                f"{selected[0].dataset_name} episode {selected[0].source_index} and "
                f"{episode.dataset_name} episode {episode.source_index}: {detail}"
            )
    return selected
