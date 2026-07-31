#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

COMPLETION_LABEL = 2
VALID_FRAME_STATES = frozenset({-1, 0, 1, COMPLETION_LABEL})
NONTERMINAL_PROGRESS_MAX = 0.99


def load_episode_lengths(dataset_path: str | Path) -> list[tuple[int, int]]:
    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"LeRobot episodes metadata not found: {episodes_path}")

    episodes = []
    with episodes_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            episode_index = int(item["episode_index"])
            length = item.get("length", item.get("num_frames", item.get("episode_length")))
            if length is None and "from" in item and "to" in item:
                length = int(item["to"]) - int(item["from"])
            if length is None:
                raise ValueError(f"Cannot determine episode length from metadata row: {item}")
            episodes.append((episode_index, int(length)))
    return sorted(episodes, key=lambda row: row[0])


def _load_label_episodes(labels_path: str | Path, dataset_index: int) -> dict[int, dict[str, Any]]:
    with Path(labels_path).open() as f:
        labels = json.load(f)
    if labels.get("ready") is False:
        raise ValueError(f"Tri-state labels are not ready: {labels_path}")
    datasets = labels.get("datasets", [])
    if dataset_index < 0 or dataset_index >= len(datasets):
        raise IndexError(f"dataset_index={dataset_index} out of range for {len(datasets)} label datasets")
    episodes: dict[int, dict[str, Any]] = {}
    for episode in datasets[dataset_index].get("episodes", []):
        episode_index = int(episode["episode_index"])
        if episode_index in episodes:
            raise ValueError(f"Duplicate tri-state label episode {episode_index}")
        episodes[episode_index] = episode
    return episodes


def _state_to_progress(frame_states: list[Any]) -> tuple[list[float], Counter]:
    if not frame_states:
        raise ValueError("frame_states must not be empty")

    stats: Counter = Counter()
    completion_frames: list[int] = []
    cumulative_scores: list[float] = []
    cumulative = 0.0

    for frame_index, state in enumerate(frame_states):
        if type(state) is not int or state not in VALID_FRAME_STATES:
            raise ValueError(
                f"frame_states[{frame_index}] must be an integer in -1, 0, 1, 2; got {state!r}"
            )
        stats[str(state)] += 1
        if state == COMPLETION_LABEL:
            completion_frames.append(frame_index)
            continue
        cumulative += float(state)
        cumulative_scores.append(cumulative)

    if len(completion_frames) > 1:
        raise ValueError("Completion label 2 may appear at most once per episode")
    if completion_frames and completion_frames[0] != len(frame_states) - 1:
        raise ValueError(
            f"Completion label 2 may only appear at final frame {len(frame_states) - 1}, "
            f"got frame {completion_frames[0]}"
        )

    if not cumulative_scores:
        normalized_scores: list[float] = []
    else:
        finite_scores = np.asarray(cumulative_scores, dtype=float)
        score_min = float(np.min(finite_scores))
        score_max = float(np.max(finite_scores))
        denom = score_max - score_min
        if denom <= 1e-8:
            normalized_scores = [0.0] * len(cumulative_scores)
        else:
            normalized_scores = [
                float((score - score_min) / denom) * NONTERMINAL_PROGRESS_MAX
                for score in cumulative_scores
            ]

    progress = normalized_scores
    if completion_frames:
        progress.append(1.0)
    return progress, stats


def build_progress_rows(
    episode_lengths: list[tuple[int, int]],
    labels_path: str | Path,
    *,
    dataset_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_episodes = _load_label_episodes(labels_path, dataset_index)
    rows: list[dict[str, Any]] = []
    report_counter: Counter = Counter()
    state_counter: Counter = Counter()
    global_index = 0

    for episode_index, episode_length in episode_lengths:
        label_episode = label_episodes.get(episode_index)
        if label_episode is None:
            raise ValueError(f"Missing tri-state labels for episode {episode_index}")
        report_counter["matched_episodes"] += 1
        frame_states = label_episode.get("frame_states")
        if not isinstance(frame_states, list):
            raise ValueError(f"Episode {episode_index} frame_states must be a list")
        if len(frame_states) != episode_length:
            raise ValueError(
                f"Episode {episode_index} frame_states length mismatch: "
                f"expected {episode_length}, got {len(frame_states)}"
            )

        progress, episode_state_counter = _state_to_progress(frame_states)
        state_counter.update(episode_state_counter)
        if frame_states[-1] == COMPLETION_LABEL:
            report_counter["completed_episodes"] += 1
        else:
            report_counter["unfinished_episodes"] += 1

        for frame_index in range(episode_length):
            frame_progress = progress[frame_index]
            raw_state = frame_states[frame_index]
            report_counter["valid_frames"] += 1
            rows.append(
                {
                    "index": global_index + frame_index,
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "episode_length": episode_length,
                    "progress": frame_progress,
                    "valid_label": True,
                    "raw_state": raw_state,
                }
            )
        global_index += episode_length

    extra_label_episodes = sorted(set(label_episodes) - {index for index, _ in episode_lengths})
    if extra_label_episodes:
        raise ValueError(f"Tri-state labels contain unknown episodes: {extra_label_episodes}")

    report = {
        "label_path": str(labels_path),
        "dataset_index": dataset_index,
        "dataset_episodes": len(episode_lengths),
        "label_episodes": len(label_episodes),
        "output_rows": len(rows),
        "nonterminal_progress_max": NONTERMINAL_PROGRESS_MAX,
        **dict(report_counter),
        "state_counts": dict(state_counter),
    }
    return rows, report


def write_progress_parquet(output_path: str | Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not rows:
        raise ValueError("No progress rows were produced")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {key: [row.get(key) for row in rows] for key in rows[0]}
    table = pa.Table.from_pydict(columns)
    metadata = dict(table.schema.metadata or {})
    metadata[b"progress_source"] = b"tristate_labels"
    metadata[b"progress_report"] = json.dumps(report, ensure_ascii=True).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ARM RA-BC progress parquet from tristate labels.")
    parser.add_argument("--dataset", required=True, help="LeRobot v2.1 dataset root.")
    parser.add_argument(
        "--labels",
        default="/mnt/workspace/robot_task_raw/lite-0028/tristate_labels.json",
        help="Path to tristate_labels.json.",
    )
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--dataset-index", type=int, default=0, help="Label dataset index inside tristate JSON.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_lengths = load_episode_lengths(args.dataset)
    rows, report = build_progress_rows(episode_lengths, args.labels, dataset_index=args.dataset_index)
    write_progress_parquet(args.output, rows, report)
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
