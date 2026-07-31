#!/usr/bin/env python3
"""Validate all video files in a LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from concurrent import futures
import json
from pathlib import Path
import subprocess
import time


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _video_keys(info: dict) -> list[str]:
    return [
        key
        for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
    return root / info["video_path"].format(
        episode_chunk=_episode_chunk(episode_index, int(info["chunks_size"])),
        video_key=video_key,
        episode_index=episode_index,
    )


def _check_video(item: tuple[int, str, Path, float]) -> dict:
    episode_index, video_key, path, timeout_s = item
    record = {
        "episode_index": episode_index,
        "video_key": video_key,
        "path": str(path),
        "ok": False,
        "size": None,
        "duration": None,
        "error": None,
    }
    try:
        if not path.exists():
            record["error"] = "missing"
            return record
        record["size"] = path.stat().st_size
        if record["size"] <= 0:
            record["error"] = "empty"
            return record
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,codec_name,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        if proc.returncode != 0:
            record["error"] = (proc.stderr or proc.stdout or f"ffprobe exit {proc.returncode}").strip()
            return record
        parsed = json.loads(proc.stdout)
        streams = parsed.get("streams") or []
        if not streams:
            record["error"] = "no_video_stream"
            return record
        duration = (parsed.get("format") or {}).get("duration")
        record["duration"] = float(duration) if duration is not None else None
        record["stream"] = streams[0]
        record["ok"] = True
        return record
    except subprocess.TimeoutExpired:
        record["error"] = f"ffprobe_timeout_{timeout_s}s"
        return record
    except Exception as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    video_keys = _video_keys(info)

    jobs = [
        (int(ep["episode_index"]), key, _video_path(root, info, key, int(ep["episode_index"])), args.timeout_s)
        for ep in episodes
        for key in video_keys
    ]

    started = time.time()
    bad: list[dict] = []
    all_records: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(_check_video, jobs), start=1):
            all_records.append(record)
            if not record["ok"]:
                bad.append(record)
                print(
                    f"BAD episode={record['episode_index']} key={record['video_key']} "
                    f"error={record['error']}",
                    flush=True,
                )
            if i % 100 == 0 or i == len(jobs):
                print(f"checked {i}/{len(jobs)} videos, bad={len(bad)}", flush=True)

    bad_episodes = sorted({int(row["episode_index"]) for row in bad})
    summary = {
        "dataset": str(root),
        "videos_checked": len(jobs),
        "episodes_checked": len(episodes),
        "video_keys": video_keys,
        "bad_videos": len(bad),
        "bad_episodes": bad_episodes,
        "bad_episode_count": len(bad_episodes),
        "elapsed_s": time.time() - started,
    }

    (output_dir / "video_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "bad_episodes.json").write_text(json.dumps(bad_episodes, indent=2) + "\n")
    with (output_dir / "bad_videos.jsonl").open("w") as f:
        for row in bad:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output_dir / "video_validation_all.jsonl").open("w") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
