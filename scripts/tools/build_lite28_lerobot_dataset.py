#!/usr/bin/env python3
"""Build the Lite28 training dataset from converted LeRobot v2.1 sources.

This script fixes the processing flow used for Lite28:

1. Read one or more LeRobot v2.1 source datasets.
2. Keep episodes with duration strictly between the configured limits.
3. Drop episodes where robot joints stay static for too long.
4. Merge and renumber episodes into one output dataset.
5. Set every prompt/task to "fold clothes".
6. Swap only action gripper columns 7 and 15 by default.
   The observation.state columns are intentionally left unchanged.
7. Recompute stats and write a build_report.json.

Example:
    python scripts/tools/build_lite28_lerobot_dataset.py \
      --source /path/to/_tmp_lite_28_first_2026-07-04_joints_lerobot_v21 \
      --source /path/to/_tmp_lite_28_first_2026-07-05_joints_lerobot_v21 \
      --source /path/to/_tmp_lite_28_first_2026-07-06_joints_lerobot_v21 \
      --output /path/to/lite_28_first_30to90s_static10s_delta16_valid \
      --exclude-episode 631 --exclude-episode 633 --exclude-episode 635 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    )


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    rel_path = info["data_path"].format(
        episode_chunk=_episode_chunk(episode_index, int(info["chunks_size"])),
        episode_index=episode_index,
    )
    return root / rel_path


def _video_path(root: Path, info: dict[str, Any], video_key: str, episode_index: int) -> Path:
    rel_path = info["video_path"].format(
        episode_chunk=_episode_chunk(episode_index, int(info["chunks_size"])),
        video_key=video_key,
        episode_index=episode_index,
    )
    return root / rel_path


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _stats_array(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack(series.to_numpy()).astype(np.float64)
    return np.asarray(series.to_numpy(), dtype=np.float64).reshape(-1, 1)


def _feature_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _stat_keys(info: dict[str, Any]) -> list[str]:
    video_key_set = set(_video_keys(info))
    return [
        key
        for key, feature in info["features"].items()
        if key not in video_key_set and isinstance(feature, dict) and "dtype" in feature
    ]


def _median_dt(df: pd.DataFrame, fallback_fps: float) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return 1.0 / fallback_fps
    timestamps = np.asarray(df["timestamp"].to_numpy(), dtype=np.float64)
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0 / fallback_fps
    return float(np.median(diffs))


def _episode_duration_s(df: pd.DataFrame, fallback_fps: float) -> float:
    if len(df) == 0:
        return 0.0
    if "timestamp" not in df.columns or len(df) < 2:
        return float(len(df)) / fallback_fps
    timestamps = np.asarray(df["timestamp"].to_numpy(), dtype=np.float64)
    finite = timestamps[np.isfinite(timestamps)]
    if len(finite) < 2:
        return float(len(df)) / fallback_fps
    dt = _median_dt(df, fallback_fps)
    return float(finite[-1] - finite[0] + dt)


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _static_seconds(
    df: pd.DataFrame,
    fallback_fps: float,
    joint_indices: tuple[int, ...],
    threshold: float,
) -> float:
    if "observation.state" not in df.columns or len(df) < 2:
        return 0.0
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    max_index = max(joint_indices)
    if state.shape[1] <= max_index:
        raise ValueError(f"observation.state has dim {state.shape[1]}, cannot use joint index {max_index}")
    diffs = np.abs(np.diff(state[:, joint_indices], axis=0)).max(axis=1)
    longest = _longest_true_run(diffs <= threshold)
    return float(longest) * _median_dt(df, fallback_fps)


def _swap_vector_columns(series: pd.Series, first_index: int, second_index: int) -> pd.Series:
    def swap_one(value: Any) -> np.ndarray:
        array = np.asarray(value).copy()
        array[first_index], array[second_index] = array[second_index], array[first_index]
        return array

    return series.apply(swap_one)


def _load_exclude_json(path: Path) -> set[int]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("bad_episodes") or data.get("exclude_episodes") or data.get("episodes") or []
    if not isinstance(data, list):
        raise ValueError(f"Unsupported exclude json format: {path}")
    return {int(item) for item in data}


def _parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        raise argparse.ArgumentTypeError("index list cannot be empty")
    return indices


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_metadata_skeleton(output: Path) -> None:
    (output / "meta").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a filtered Lite28 LeRobot v2.1 dataset with fixed task and gripper handling."
    )
    parser.add_argument("--source", type=Path, action="append", required=True, help="Input LeRobot v2.1 dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset path.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--static-seconds", type=float, default=10.0)
    parser.add_argument("--static-threshold", type=float, default=0.001)
    parser.add_argument("--static-joint-indices", type=_parse_indices, default=DEFAULT_JOINT_INDICES)
    parser.add_argument("--task", default="fold clothes")
    parser.add_argument("--metadata-fps", type=float, default=29.0)
    parser.add_argument("--swap-action-grippers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-gripper-indices", type=_parse_indices, default=(7, 15))
    parser.add_argument(
        "--exclude-episode",
        type=int,
        action="append",
        default=[],
        help="Episode index to exclude after duration/static filtering, before final renumbering.",
    )
    parser.add_argument(
        "--exclude-json",
        type=Path,
        help="JSON list, or dict with bad_episodes/exclude_episodes, using post-filter episode indices.",
    )
    parser.add_argument("--limit-episodes", type=int, help="Debug only: stop after this many final episodes.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.min_seconds >= args.max_seconds:
        raise ValueError("--min-seconds must be smaller than --max-seconds")
    if len(args.action_gripper_indices) != 2:
        raise ValueError("--action-gripper-indices must contain exactly two indices")

    sources = [source.resolve() for source in args.source]
    output = args.output.resolve()
    _prepare_output(output, args.overwrite)
    _copy_metadata_skeleton(output)

    source_infos = [json.loads((source / "meta" / "info.json").read_text()) for source in sources]
    base_info = dict(source_infos[0])
    base_features = set(base_info["features"])
    for source, info in zip(sources[1:], source_infos[1:], strict=True):
        if set(info["features"]) != base_features:
            raise ValueError(f"Feature keys differ in source: {source}")

    chunks_size = int(base_info["chunks_size"])
    video_keys = _video_keys(base_info)
    stat_keys = _stat_keys(base_info)
    exclude_indices = set(args.exclude_episode)
    if args.exclude_json:
        exclude_indices.update(_load_exclude_json(args.exclude_json.resolve()))

    new_episodes: list[dict[str, Any]] = []
    new_episode_stats: list[dict[str, Any]] = []
    global_stats: dict[str, list[np.ndarray]] = {key: [] for key in stat_keys}
    selected: list[dict[str, Any]] = []
    dropped_duration: list[dict[str, Any]] = []
    dropped_static: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    selected_by_source = [0 for _ in sources]
    input_episodes = 0
    input_frames = 0
    output_frames = 0
    global_index = 0
    link_counts = {"hardlink": 0, "copy": 0}

    for source_id, (source, info) in enumerate(zip(sources, source_infos, strict=True)):
        episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
        source_fps = float(info["fps"])
        input_episodes += len(episodes)

        for episode in episodes:
            old_episode_index = int(episode["episode_index"])
            old_data = _data_path(source, info, old_episode_index)
            df = pd.read_parquet(old_data)
            length = int(len(df))
            input_frames += length

            duration_s = _episode_duration_s(df, source_fps)
            base_record = {
                "source_id": source_id,
                "source": str(source),
                "old_episode_index": old_episode_index,
                "length": length,
                "duration_s": duration_s,
                "source_fps": source_fps,
            }
            if not (args.min_seconds < duration_s < args.max_seconds):
                dropped_duration.append(base_record)
                continue

            static_s = _static_seconds(df, source_fps, args.static_joint_indices, args.static_threshold)
            base_record["max_static_seconds"] = static_s
            if static_s > args.static_seconds:
                dropped_static.append(base_record)
                continue

            post_filter_index = len(selected) + len(excluded)
            if post_filter_index in exclude_indices:
                excluded.append({**base_record, "post_filter_episode_index": post_filter_index})
                continue

            new_episode_index = len(new_episodes)
            if args.swap_action_grippers:
                first_index, second_index = args.action_gripper_indices
                df["action"] = _swap_vector_columns(df["action"], first_index, second_index)

            df["episode_index"] = new_episode_index
            df["frame_index"] = np.arange(length, dtype=np.int64)
            df["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
            if "task_index" in df.columns:
                df["task_index"] = 0

            new_chunk = _episode_chunk(new_episode_index, chunks_size)
            new_data = output / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_episode_index:06d}.parquet"
            new_data.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(new_data, index=False)

            per_episode_stats = {}
            for key in stat_keys:
                if key in df.columns:
                    values = _stats_array(df[key])
                    global_stats[key].append(values)
                    per_episode_stats[key] = _feature_stats(values)
            new_episode_stats.append({"episode_index": new_episode_index, "stats": per_episode_stats})

            for video_key in video_keys:
                old_video = _video_path(source, info, video_key, old_episode_index)
                new_video = (
                    output
                    / "videos"
                    / f"chunk-{new_chunk:03d}"
                    / video_key
                    / f"episode_{new_episode_index:06d}.mp4"
                )
                method = _link_or_copy(old_video, new_video)
                link_counts[method] += 1

            episode_record = {
                "episode_index": new_episode_index,
                "tasks": [args.task],
                "length": length,
            }
            new_episodes.append(episode_record)
            selected.append(
                {
                    **base_record,
                    "post_filter_episode_index": post_filter_index,
                    "new_episode_index": new_episode_index,
                }
            )
            selected_by_source[source_id] += 1
            global_index += length
            output_frames += length

            if args.progress_every > 0 and len(new_episodes) % args.progress_every == 0:
                print(f"wrote {len(new_episodes)} episodes, {output_frames} frames", flush=True)
            if args.limit_episodes is not None and len(new_episodes) >= args.limit_episodes:
                break
        if args.limit_episodes is not None and len(new_episodes) >= args.limit_episodes:
            break

    if not new_episodes:
        raise ValueError("No episodes matched the requested filters.")

    stats = {
        key: _feature_stats(np.concatenate(chunks, axis=0))
        for key, chunks in global_stats.items()
        if chunks
    }

    base_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    base_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    base_info["fps"] = args.metadata_fps
    base_info["total_episodes"] = len(new_episodes)
    base_info["total_frames"] = global_index
    base_info["total_chunks"] = _episode_chunk(len(new_episodes) - 1, chunks_size) + 1
    base_info["total_videos"] = len(new_episodes) * len(video_keys)
    base_info["total_tasks"] = 1
    base_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    task_rows = [{"task_index": 0, "task": args.task}]
    (output / "meta" / "info.json").write_text(json.dumps(base_info, indent=2, ensure_ascii=False) + "\n")
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    _write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    _write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", new_episode_stats)

    report = {
        "sources": [str(source) for source in sources],
        "output": str(output),
        "duration_filter": {
            "min_seconds_exclusive": args.min_seconds,
            "max_seconds_exclusive": args.max_seconds,
            "duration_uses": "timestamp median dt when available, otherwise source metadata fps",
        },
        "static_filter": {
            "joint_indices": list(args.static_joint_indices),
            "max_abs_diff_threshold_per_frame": args.static_threshold,
            "max_static_seconds_exclusive": args.static_seconds,
        },
        "task": args.task,
        "metadata_fps": args.metadata_fps,
        "source_fps": [float(info["fps"]) for info in source_infos],
        "input_episodes": input_episodes,
        "input_frames": input_frames,
        "duration_kept_before_static": input_episodes - len(dropped_duration),
        "output_episodes": len(new_episodes),
        "output_frames": output_frames,
        "dropped_by_duration": len(dropped_duration),
        "dropped_by_static": len(dropped_static),
        "excluded_after_filters": len(excluded),
        "selected_by_source": selected_by_source,
        "video_keys": video_keys,
        "link_counts": link_counts,
        "action_gripper_swap": (
            f"swapped action columns {args.action_gripper_indices[0]} and "
            f"{args.action_gripper_indices[1]}; observation.state unchanged"
            if args.swap_action_grippers
            else "disabled; observation.state unchanged"
        ),
        "exclude_episode_indices_post_filter": sorted(exclude_indices),
        "selected": selected,
        "dropped_duration": dropped_duration,
        "dropped_static": dropped_static,
        "excluded": excluded,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(
        f"Built {len(new_episodes)} episodes, {output_frames} frames. "
        f"Dropped duration={len(dropped_duration)}, static={len(dropped_static)}, "
        f"excluded={len(excluded)}. Wrote: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
