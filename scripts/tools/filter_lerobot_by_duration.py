#!/usr/bin/env python3
"""Filter a LeRobot v2.1 dataset by episode duration and renumber episodes."""

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


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-seconds", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    chunks_size = int(info["chunks_size"])
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    episode_stats = {row["episode_index"]: row for row in _read_jsonl(source / "meta" / "episodes_stats.jsonl")}

    selected = []
    for episode in episodes:
        duration_s = float(episode["length"]) / fps
        if args.min_seconds <= duration_s <= args.max_seconds:
            selected.append((episode, duration_s))

    if not selected:
        raise ValueError("No episodes matched the requested duration range.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta" / "tasks.jsonl", output / "meta" / "tasks.jsonl")

    video_keys = [
        key
        for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]

    new_episodes = []
    new_episode_stats = []
    all_values: dict[str, list[np.ndarray]] = {"observation.state": [], "action": []}
    global_index = 0
    link_counts = {"hardlink": 0, "copy": 0}
    duration_report = []

    for new_index, (old_episode, duration_s) in enumerate(selected):
        old_index = int(old_episode["episode_index"])
        old_chunk = _episode_chunk(old_index, chunks_size)
        new_chunk = _episode_chunk(new_index, chunks_size)

        old_data = source / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_index:06d}.parquet"
        new_data = output / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_index:06d}.parquet"
        new_data.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(old_data)
        length = int(len(df))
        df["episode_index"] = new_index
        df["frame_index"] = np.arange(length, dtype=np.int64)
        df["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        df.to_parquet(new_data, index=False)

        for key in all_values:
            all_values[key].append(np.stack(df[key].to_numpy()))

        for video_key in video_keys:
            old_video = (
                source
                / "videos"
                / f"chunk-{old_chunk:03d}"
                / video_key
                / f"episode_{old_index:06d}.mp4"
            )
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
                "tasks": old_episode["tasks"],
                "length": length,
            }
        )

        old_stats = episode_stats[old_index]
        new_stats = dict(old_stats)
        new_stats["episode_index"] = new_index
        new_episode_stats.append(new_stats)

        duration_report.append(
            {
                "old_episode_index": old_index,
                "new_episode_index": new_index,
                "length": length,
                "duration_s": duration_s,
            }
        )
        global_index += length

    stats = {key: _feature_stats(np.concatenate(chunks, axis=0)) for key, chunks in all_values.items()}

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_index
    info["total_chunks"] = _episode_chunk(len(new_episodes) - 1, chunks_size) + 1
    info["total_videos"] = len(new_episodes) * len(video_keys)
    info["splits"] = {"train": f"0:{len(new_episodes)}"}

    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    _write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", new_episode_stats)

    report = {
        "source": str(source),
        "output": str(output),
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "input_episodes": len(episodes),
        "output_episodes": len(new_episodes),
        "dropped_episodes": len(episodes) - len(new_episodes),
        "output_frames": global_index,
        "fps": fps,
        "link_counts": link_counts,
        "selected": duration_report,
    }
    (output / "filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Filtered {len(episodes)} -> {len(new_episodes)} episodes, "
        f"{global_index} frames. Wrote: {output}"
    )


if __name__ == "__main__":
    main()
