"""Self-contained, read-only probe for complete LeRobot v2.1 HIL datasets.

This file is sent verbatim to a robot and executed with ``python3 -``.  Keep it
independent from the rest of :mod:`tristate_labeler`; only the Python standard
library and PyArrow are available remotely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


_REQUIRED_METADATA = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/expert_frame_index.json",
)
_STABLE_COLUMNS = (
    "capture_timestamp",
    "observation.state",
    "action",
    "intervention",
    "control_mode",
)
_REQUIRED_VIDEO_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
_FileEntry = Tuple[str, int, int, str]


class _InvalidRoot(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _InvalidRoot(f"unreadable metadata {path.name}: {exc.__class__.__name__}") from exc
    if not isinstance(value, dict):
        raise _InvalidRoot(f"metadata {path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise _InvalidRoot(f"metadata {path.name} line {line_number} must be an object")
            records.append(value)
    except _InvalidRoot:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _InvalidRoot(f"unreadable metadata {path.name}: {exc.__class__.__name__}") from exc
    return records


def _require_int(record: Mapping[str, object], field: str, context: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _InvalidRoot(f"{context} has invalid {field}")
    return value


def _resolve_episode_path(template: str, episode_index: int) -> str:
    return template.format(
        episode_index=episode_index,
        episode_chunk=episode_index // 1000,
    )


def _dataset_path(root: Path, relative: str, context: str) -> Path:
    try:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise _InvalidRoot(f"{context} escapes the dataset root")
        resolved_root = root.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise _InvalidRoot(f"{context} escapes the dataset root") from exc
        return resolved
    except _InvalidRoot:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise _InvalidRoot(f"{context} is invalid") from exc


def _semantic_fingerprint(
    parquet_path: Path,
    video_paths: Mapping[str, Path],
    task: str,
) -> str:
    table = pq.read_table(parquet_path)
    video_hashes = {role: _sha256(path) for role, path in sorted(video_paths.items())}
    return _semantic_fingerprint_from_table(table, video_hashes, task)


def _semantic_fingerprint_from_table(
    full_table: pa.Table,
    video_hashes: Mapping[str, str],
    task: str,
) -> str:
    selected = full_table.select(list(_STABLE_COLUMNS)).combine_chunks()
    canonical_schema = pa.schema(
        [
            pa.field(field.name, field.type, nullable=field.nullable)
            for field in selected.schema
        ],
        metadata=None,
    )
    table = pa.Table.from_arrays(selected.columns, schema=canonical_schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=max(1, table.num_rows))
    parquet_semantic = hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()
    payload = {
        "task": task,
        "row_count": table.num_rows,
        "parquet_semantic": parquet_semantic,
        "video_hashes": video_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_file(
    role: str,
    path: Path,
    file_cache: Mapping[Path, _FileEntry],
) -> dict[str, object]:
    resolved = path.resolve()
    relative_path, size, mtime_ns, sha256 = file_cache[resolved]
    return {
        "role": role,
        "absolute_path": str(resolved),
        "relative_path": relative_path,
        "size": size,
        "mtime_ns": mtime_ns,
        "sha256": sha256,
    }


def _metadata_digest(root: Path, file_cache: Mapping[Path, _FileEntry]) -> str:
    digest = hashlib.sha256()
    for relative in _REQUIRED_METADATA:
        entry = file_cache.get((root / relative).resolve())
        if entry is None:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry[3].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _snapshot_files(root: Path) -> dict[Path, _FileEntry]:
    snapshot: dict[Path, _FileEntry] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        resolved = path.resolve()
        if resolved in snapshot:
            raise ValueError("multiple source paths resolve to one file")
        stat = path.stat()
        snapshot[resolved] = (
            path.relative_to(root).as_posix(),
            stat.st_size,
            stat.st_mtime_ns,
            _sha256(path),
        )
    return snapshot


def _file_state(file_cache: Mapping[Path, _FileEntry]) -> list[list[object]]:
    return [list(entry) for entry in sorted(file_cache.values(), key=lambda entry: entry[0])]


def _has_transaction(root: Path) -> bool:
    transaction_root = root / "meta" / "tmp"
    if not transaction_root.is_dir():
        return False
    return any(
        child.is_dir() and (child.name.startswith("write_") or child.name.startswith("delete_"))
        for child in transaction_root.iterdir()
    )


def _indexed_records(
    records: Sequence[dict[str, object]],
    context: str,
) -> dict[int, dict[str, object]]:
    indexed: dict[int, dict[str, object]] = {}
    for record in records:
        index = _require_int(record, "episode_index", context)
        if index in indexed:
            raise _InvalidRoot(f"duplicate {context} episode_index {index}")
        indexed[index] = record
    return indexed


def _validate_root(
    root: Path,
    file_cache: Mapping[Path, _FileEntry],
) -> list[dict[str, object]]:
    for relative in _REQUIRED_METADATA:
        if (root / relative).resolve() not in file_cache:
            raise _InvalidRoot(f"missing required metadata {relative}")

    info = _read_json(root / "meta" / "info.json")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    episode_records = _read_jsonl(root / "meta" / "episodes.jsonl")
    stats_records = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
    expert_document = _read_json(root / "meta" / "expert_frame_index.json")

    if info.get("codebase_version") != "v2.1":
        raise _InvalidRoot("info.json codebase_version must be v2.1")
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    features = info.get("features")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise _InvalidRoot("info.json has invalid data_path or video_path")
    if not isinstance(features, dict):
        raise _InvalidRoot("info.json has invalid features")
    declared_video_keys = {
        key
        for key, description in features.items()
        if isinstance(key, str)
        and isinstance(description, dict)
        and description.get("dtype") == "video"
    }
    if declared_video_keys != set(_REQUIRED_VIDEO_KEYS):
        raise _InvalidRoot("info.json video features do not match the required camera contract")
    video_keys = list(_REQUIRED_VIDEO_KEYS)

    if _require_int(info, "total_episodes", "info.json") != len(episode_records):
        raise _InvalidRoot("info.json total_episodes does not match episode records")
    if _require_int(info, "total_tasks", "info.json") != len(tasks):
        raise _InvalidRoot("info.json total_tasks does not match task records")

    lengths = [_require_int(record, "length", "episode record") for record in episode_records]
    if _require_int(info, "total_frames", "info.json") != sum(lengths):
        raise _InvalidRoot("info.json total_frames does not match episode lengths")
    expected_videos = len(episode_records) * len(video_keys)
    if _require_int(info, "total_videos", "info.json") != expected_videos:
        raise _InvalidRoot("info.json total_videos does not match declared video files")

    task_by_name: dict[str, dict[str, object]] = {}
    for record in tasks:
        task = record.get("task")
        if not isinstance(task, str):
            raise _InvalidRoot("task record has invalid task")
        if task in task_by_name:
            raise _InvalidRoot(f"duplicate task record {task!r}")
        task_by_name[task] = record

    stats_by_index = _indexed_records(stats_records, "statistics")
    expert_episodes = expert_document.get("episodes")
    if not isinstance(expert_episodes, list) or not all(isinstance(item, dict) for item in expert_episodes):
        raise _InvalidRoot("expert_frame_index.json has invalid episodes")
    expert_by_index = _indexed_records(expert_episodes, "expert")
    if len(stats_by_index) != len(episode_records):
        raise _InvalidRoot("statistics record count does not match episode records")
    if len(expert_by_index) != len(episode_records):
        raise _InvalidRoot("expert record count does not match episode records")

    episodes: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    used_parquet_paths: set[Path] = set()
    used_video_paths: set[Path] = set()
    for episode_record in episode_records:
        source_index = _require_int(episode_record, "episode_index", "episode record")
        if source_index in seen_indices:
            raise _InvalidRoot(f"duplicate episode_index {source_index}")
        seen_indices.add(source_index)
        length = _require_int(episode_record, "length", "episode record")
        episode_tasks = episode_record.get("tasks")
        if (
            not isinstance(episode_tasks, list)
            or len(episode_tasks) != 1
            or not isinstance(episode_tasks[0], str)
        ):
            raise _InvalidRoot(f"episode {source_index} must declare exactly one task")
        task = episode_tasks[0]
        if task not in task_by_name:
            raise _InvalidRoot(f"episode {source_index} references unknown task")
        if source_index not in stats_by_index:
            raise _InvalidRoot(f"episode {source_index} is missing statistics metadata")
        if source_index not in expert_by_index:
            raise _InvalidRoot(f"episode {source_index} is missing expert metadata")

        try:
            parquet_relative = _resolve_episode_path(data_template, source_index)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise _InvalidRoot("info.json data_path is not a valid episode template") from exc
        parquet_path = _dataset_path(root, parquet_relative, "data_path")
        if parquet_path in used_parquet_paths:
            raise _InvalidRoot("data_path reuses one parquet for multiple episodes")
        used_parquet_paths.add(parquet_path)
        if parquet_path not in file_cache:
            raise _InvalidRoot(f"episode {source_index} missing parquet {parquet_relative}")

        video_paths: dict[str, Path] = {}
        for key in video_keys:
            try:
                relative = _resolve_episode_path(
                    video_template.replace("{video_key}", key),
                    source_index,
                )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                raise _InvalidRoot("info.json video_path is not a valid episode template") from exc
            path = _dataset_path(root, relative, "video_path")
            if path in used_video_paths:
                raise _InvalidRoot("video_path reuses one video for multiple episodes")
            used_video_paths.add(path)
            if path not in file_cache:
                raise _InvalidRoot(f"episode {source_index} missing video {key}: {relative}")
            video_paths[key] = path

        try:
            full_table = pq.read_table(parquet_path)
            if full_table.num_rows != length:
                raise _InvalidRoot(
                    f"episode {source_index} parquet row count does not match length"
                )
            video_hashes = {
                role: file_cache[path][3]
                for role, path in sorted(video_paths.items())
            }
            fingerprint = _semantic_fingerprint_from_table(full_table, video_hashes, task)
        except _InvalidRoot:
            raise
        except Exception as exc:
            raise _InvalidRoot(
                f"episode {source_index} parquet is unreadable: {exc.__class__.__name__}"
            ) from exc

        files = [_source_file("parquet", parquet_path, file_cache)]
        files.extend(
            _source_file(role, path, file_cache)
            for role, path in video_paths.items()
        )
        episodes.append(
            {
                "dataset_root": str(root.resolve()),
                "dataset_name": root.name,
                "source_index": source_index,
                "task": task,
                "length": length,
                "completed_ns": max(int(file["mtime_ns"]) for file in files),
                "fingerprint": fingerprint,
                "files": files,
                "info": info,
                "task_record": task_by_name[task],
                "episode_record": episode_record,
                "stats_record": stats_by_index[source_index],
                "expert_record": expert_by_index[source_index],
            }
        )
    discovered_parquets = {
        path for path, entry in file_cache.items() if entry[0].endswith(".parquet")
    }
    discovered_videos = {
        path for path, entry in file_cache.items() if entry[0].endswith(".mp4")
    }
    if used_parquet_paths != discovered_parquets:
        raise _InvalidRoot("referenced parquet file set does not match discovered parquet files")
    if used_video_paths != discovered_videos:
        raise _InvalidRoot("referenced video file set does not match discovered video files")

    episodes.sort(key=lambda episode: int(episode["source_index"]))
    return episodes


def _scan_root_once(root: Path) -> dict[str, object]:
    dataset_root = str(root.resolve())
    try:
        if _has_transaction(root):
            return {
                "dataset_root": dataset_root,
                "status": "busy",
            }
        file_cache = _snapshot_files(root)
        snapshot: dict[str, object] = {
            "dataset_root": dataset_root,
            "metadata_digest": _metadata_digest(root, file_cache),
            "file_state": _file_state(file_cache),
        }
        try:
            snapshot["episodes"] = _validate_root(root, file_cache)
            snapshot["status"] = "candidate"
        except _InvalidRoot as exc:
            snapshot["status"] = "rejected"
            snapshot["reason"] = str(exc)
        return snapshot
    except (OSError, ValueError) as exc:
        return {
            "dataset_root": dataset_root,
            "status": "rejected",
            "reason": f"source root is unreadable: {exc.__class__.__name__}",
        }


def _scan_parent_once(source_parent: Path) -> dict[str, object]:
    parent = Path(source_parent).expanduser().resolve()
    roots = []
    if parent.is_dir():
        roots = sorted(
            (
                child
                for child in parent.iterdir()
                if child.is_dir() and child.name.startswith("hil_pico_v21_")
            ),
            key=lambda child: child.name,
        )
    return {
        "source_parent": str(parent),
        "roots": [_scan_root_once(root) for root in roots],
    }


def _compare_and_finalize(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    first_roots = {
        str(root["dataset_root"]): root
        for root in first.get("roots", [])
        if isinstance(root, dict) and isinstance(root.get("dataset_root"), str)
    }
    second_roots = {
        str(root["dataset_root"]): root
        for root in second.get("roots", [])
        if isinstance(root, dict) and isinstance(root.get("dataset_root"), str)
    }
    episodes: list[object] = []
    busy_roots: list[str] = []
    rejected_roots: list[dict[str, str]] = []
    for dataset_root in sorted(first_roots.keys() | second_roots.keys()):
        before = first_roots.get(dataset_root)
        after = second_roots.get(dataset_root)
        if before is None or after is None:
            busy_roots.append(dataset_root)
            continue
        if before.get("status") == "busy" or after.get("status") == "busy":
            busy_roots.append(dataset_root)
            continue
        if (
            before.get("metadata_digest") != after.get("metadata_digest")
            or before.get("file_state") != after.get("file_state")
            or before.get("episodes") != after.get("episodes")
        ):
            busy_roots.append(dataset_root)
            continue
        if before.get("status") != "candidate" or after.get("status") != "candidate":
            reason = after.get("reason", before.get("reason", "invalid source root"))
            rejected_roots.append(
                {
                    "dataset_root": dataset_root,
                    "reason": str(reason),
                }
            )
            continue
        root_episodes = after.get("episodes", [])
        if isinstance(root_episodes, list):
            episodes.extend(root_episodes)
    return {
        "episodes": episodes,
        "busy_roots": busy_roots,
        "rejected_roots": rejected_roots,
    }


def probe_source_parent(source_parent: Path, stable_seconds: float) -> dict[str, object]:
    first = _scan_parent_once(source_parent)
    if stable_seconds:
        time.sleep(stable_seconds)
    second = _scan_parent_once(source_parent)
    return _compare_and_finalize(first, second)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only HIL v2.1 source probe")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--stable-seconds", required=True, type=float)
    args = parser.parse_args(argv)
    if not math.isfinite(args.stable_seconds) or args.stable_seconds < 0:
        parser.error("--stable-seconds must be finite and non-negative")
    result = probe_source_parent(args.source_root, args.stable_seconds)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
