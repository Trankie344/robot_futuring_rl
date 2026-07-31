#!/usr/bin/env python3
"""Merge LeRobot v2.1 datasets and renumber episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _stats_array(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack(series.to_numpy()).astype(np.float64)
    return np.asarray(series.to_numpy(), dtype=np.float64).reshape(-1, 1)


def _feature_stats(values: np.ndarray) -> dict:
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


def _data_path(dataset: Path, info: dict, episode_index: int) -> Path:
    episode_chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    rel_path = info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset / rel_path


def _video_path(dataset: Path, info: dict, video_key: str, episode_index: int) -> Path:
    episode_chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    rel_path = info["video_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset / rel_path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = [source.resolve() for source in args.source]
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    source_infos = [json.loads((source / "meta" / "info.json").read_text()) for source in sources]
    fps_values = {float(info["fps"]) for info in source_infos}
    if len(fps_values) != 1:
        raise ValueError(f"All sources must have the same fps, got: {sorted(fps_values)}")

    feature_keys = [set(info["features"]) for info in source_infos]
    if any(keys != feature_keys[0] for keys in feature_keys[1:]):
        raise ValueError("All sources must expose the same feature keys.")

    base_info = dict(source_infos[0])
    chunks_size = int(base_info["chunks_size"])
    video_keys = [
        key
        for key, feature in base_info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    stat_keys = [
        key
        for key, feature in base_info["features"].items()
        if key not in video_keys and isinstance(feature, dict) and "dtype" in feature
    ]

    task_to_index: dict[str, int] = {}
    task_rows: list[dict] = []
    new_episodes: list[dict] = []
    new_episode_stats: list[dict] = []
    global_stats: dict[str, list[np.ndarray]] = {key: [] for key in stat_keys}
    source_report: list[dict] = []
    link_counts = {"hardlink": 0, "copy": 0}
    global_index = 0
    new_episode_index = 0

    for source, info in zip(sources, source_infos, strict=True):
        episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
        source_frames = 0

        for episode in episodes:
            old_episode_index = int(episode["episode_index"])
            old_data = _data_path(source, info, old_episode_index)
            df = pd.read_parquet(old_data)
            length = int(len(df))

            tasks = episode.get("tasks") or []
            if not tasks:
                tasks = [""]
            for task in tasks:
                if task not in task_to_index:
                    task_to_index[task] = len(task_to_index)
                    task_rows.append({"task_index": task_to_index[task], "task": task})

            df["episode_index"] = new_episode_index
            df["frame_index"] = np.arange(length, dtype=np.int64)
            df["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
            if "task_index" in df.columns:
                df["task_index"] = task_to_index[tasks[0]]

            new_chunk = _episode_chunk(new_episode_index, chunks_size)
            new_data = output / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_episode_index:06d}.parquet"
            new_data.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(new_data, index=False)

            for key in stat_keys:
                if key in df.columns:
                    values = _stats_array(df[key])
                    global_stats[key].append(values)

            per_episode_stats = {
                key: _feature_stats(_stats_array(df[key]))
                for key in stat_keys
                if key in df.columns
            }
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

            new_episodes.append(
                {
                    "episode_index": new_episode_index,
                    "tasks": tasks,
                    "length": length,
                }
            )
            source_frames += length
            global_index += length
            new_episode_index += 1

        source_report.append(
            {
                "source": str(source),
                "episodes": len(episodes),
                "frames": source_frames,
            }
        )

    stats = {
        key: _feature_stats(np.concatenate(chunks, axis=0))
        for key, chunks in global_stats.items()
        if chunks
    }

    base_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    base_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    base_info["total_episodes"] = len(new_episodes)
    base_info["total_frames"] = global_index
    base_info["total_chunks"] = _episode_chunk(len(new_episodes) - 1, chunks_size) + 1
    base_info["total_videos"] = len(new_episodes) * len(video_keys)
    base_info["total_tasks"] = len(task_rows)
    base_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(base_info, indent=2, ensure_ascii=False) + "\n")
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    _write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    _write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", new_episode_stats)

    report = {
        "output": str(output),
        "sources": source_report,
        "output_episodes": len(new_episodes),
        "output_frames": global_index,
        "fps": float(next(iter(fps_values))),
        "link_counts": link_counts,
    }
    (output / "merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Merged {len(sources)} datasets into {output}: {len(new_episodes)} episodes, {global_index} frames")


if __name__ == "__main__":
    main()
