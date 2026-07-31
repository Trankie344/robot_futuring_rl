#!/usr/bin/env python3
"""Copy a LeRobot v2.1 dataset while excluding selected episodes."""

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
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


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


def _data_path(root: Path, info: dict, episode_index: int) -> Path:
    rel_path = info["data_path"].format(
        episode_chunk=_episode_chunk(episode_index, int(info["chunks_size"])),
        episode_index=episode_index,
    )
    return root / rel_path


def _video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
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


def _load_exclude(path: Path) -> set[int]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("bad_episodes") or data.get("exclude_episodes") or []
    return {int(x) for x in data}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    exclude = _load_exclude(args.exclude_json.resolve())
    if not exclude:
        raise ValueError("exclude list is empty")

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    video_keys = [
        key
        for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    stat_keys = [
        key
        for key, feature in info["features"].items()
        if key not in video_keys and isinstance(feature, dict) and "dtype" in feature
    ]

    task_rows = _read_jsonl(source / "meta" / "tasks.jsonl")
    new_episodes: list[dict] = []
    new_episode_stats: list[dict] = []
    global_stats: dict[str, list[np.ndarray]] = {key: [] for key in stat_keys}
    mapping: list[dict] = []
    dropped: list[int] = []
    link_counts = {"hardlink": 0, "copy": 0}
    global_index = 0

    for old_episode in episodes:
        old_index = int(old_episode["episode_index"])
        if old_index in exclude:
            dropped.append(old_index)
            continue

        new_index = len(new_episodes)
        old_data = _data_path(source, info, old_index)
        df = pd.read_parquet(old_data)
        length = int(len(df))
        df["episode_index"] = new_index
        df["frame_index"] = np.arange(length, dtype=np.int64)
        df["index"] = np.arange(global_index, global_index + length, dtype=np.int64)

        new_chunk = _episode_chunk(new_index, chunks_size)
        new_data = output / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_index:06d}.parquet"
        new_data.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(new_data, index=False)

        per_episode_stats = {}
        for key in stat_keys:
            if key in df.columns:
                values = _stats_array(df[key])
                global_stats[key].append(values)
                per_episode_stats[key] = _feature_stats(values)
        new_episode_stats.append({"episode_index": new_index, "stats": per_episode_stats})

        for video_key in video_keys:
            old_video = _video_path(source, info, video_key, old_index)
            new_video = (
                output
                / "videos"
                / f"chunk-{new_chunk:03d}"
                / video_key
                / f"episode_{new_index:06d}.mp4"
            )
            method = _link_or_copy(old_video, new_video)
            link_counts[method] += 1

        new_episodes.append(
            {
                "episode_index": new_index,
                "tasks": old_episode.get("tasks", []),
                "length": length,
            }
        )
        mapping.append(
            {
                "old_episode_index": old_index,
                "new_episode_index": new_index,
                "length": length,
            }
        )
        global_index += length

    if not new_episodes:
        raise ValueError("all episodes were excluded")

    stats = {
        key: _feature_stats(np.concatenate(chunks, axis=0))
        for key, chunks in global_stats.items()
        if chunks
    }

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_index
    info["total_chunks"] = _episode_chunk(len(new_episodes) - 1, chunks_size) + 1
    info["total_videos"] = len(new_episodes) * len(video_keys)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}

    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    _write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    _write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", new_episode_stats)

    report = {
        "source": str(source),
        "output": str(output),
        "excluded_episode_indices": sorted(exclude),
        "dropped_episode_indices": dropped,
        "input_episodes": len(episodes),
        "output_episodes": len(new_episodes),
        "input_frames": sum(int(ep["length"]) for ep in episodes),
        "output_frames": global_index,
        "video_keys": video_keys,
        "link_counts": link_counts,
        "mapping": mapping,
    }
    (output / "exclude_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Filtered {len(episodes)} -> {len(new_episodes)} episodes, "
        f"{global_index} frames. Wrote: {output}"
    )


if __name__ == "__main__":
    main()
