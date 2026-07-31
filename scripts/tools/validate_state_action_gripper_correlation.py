#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_indices(value: str) -> tuple[int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated indices")
    return parts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    rel_path = info["data_path"].format(
        episode_chunk=_episode_chunk(episode_index, int(info["chunks_size"])),
        episode_index=episode_index,
    )
    return root / rel_path


def _as_matrix(series: pd.Series, key: str) -> np.ndarray:
    try:
        return np.stack(series.to_numpy()).astype(np.float64)
    except ValueError as exc:
        raise ValueError(f"Column {key!r} must contain fixed-size vector values") from exc


def _pearson_at_lag(x: np.ndarray, y: np.ndarray, lag: int, min_std: float) -> float | None:
    if lag > 0:
        x = x[:-lag]
        y = y[lag:]
    elif lag < 0:
        x = x[-lag:]
        y = y[:lag]
    if len(x) < 3:
        return None
    if float(np.std(x)) < min_std or float(np.std(y)) < min_std:
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _best_abs_correlation(x: np.ndarray, y: np.ndarray, max_lag_frames: int, min_std: float) -> dict[str, Any]:
    best_corr: float | None = None
    best_lag: int | None = None
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        corr = _pearson_at_lag(x, y, lag, min_std)
        if corr is None:
            continue
        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    return {
        "corr": best_corr,
        "abs_corr": None if best_corr is None else abs(best_corr),
        "lag": best_lag,
    }


def _collect_gripper_arrays(
    dataset: Path,
    state_indices: tuple[int, int],
    action_indices: tuple[int, int],
    state_key: str,
    action_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    info = json.loads((dataset / "meta" / "info.json").read_text())
    episodes = _read_jsonl(dataset / "meta" / "episodes.jsonl")
    state_chunks = []
    action_chunks = []
    report = {
        "episodes": len(episodes),
        "frames": 0,
        "state_key": state_key,
        "action_key": action_key,
        "state_indices": list(state_indices),
        "action_indices": list(action_indices),
    }

    max_state_index = max(state_indices)
    max_action_index = max(action_indices)
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        data_path = _data_path(dataset, info, episode_index)
        df = pd.read_parquet(data_path, columns=[state_key, action_key])
        if state_key not in df.columns or action_key not in df.columns:
            raise ValueError(f"{data_path} must contain columns {state_key!r} and {action_key!r}")
        state = _as_matrix(df[state_key], state_key)
        action = _as_matrix(df[action_key], action_key)
        if state.shape[1] <= max_state_index:
            raise ValueError(f"{data_path} {state_key} dim={state.shape[1]} cannot use index {max_state_index}")
        if action.shape[1] <= max_action_index:
            raise ValueError(f"{data_path} {action_key} dim={action.shape[1]} cannot use index {max_action_index}")
        state_chunks.append(state[:, state_indices])
        action_chunks.append(action[:, action_indices])
        report["frames"] += int(len(df))

    return np.concatenate(state_chunks, axis=0), np.concatenate(action_chunks, axis=0), report


def validate_dataset(
    dataset: str | Path,
    *,
    state_indices: tuple[int, int],
    action_indices: tuple[int, int],
    threshold: float,
    margin: float,
    max_lag_frames: int = 5,
    min_std: float = 1e-4,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> dict[str, Any]:
    dataset = Path(dataset)
    state, action, report = _collect_gripper_arrays(dataset, state_indices, action_indices, state_key, action_key)
    direct = [
        _best_abs_correlation(state[:, 0], action[:, 0], max_lag_frames, min_std),
        _best_abs_correlation(state[:, 1], action[:, 1], max_lag_frames, min_std),
    ]
    crossed = [
        _best_abs_correlation(state[:, 0], action[:, 1], max_lag_frames, min_std),
        _best_abs_correlation(state[:, 1], action[:, 0], max_lag_frames, min_std),
    ]

    direct_abs = [item["abs_corr"] for item in direct]
    crossed_abs = [item["abs_corr"] for item in crossed]
    if any(value is None for value in direct_abs) or any(value is None for value in crossed_abs):
        report.update(
            {
                "passed": False,
                "failure_reason": "insufficient gripper variation for correlation",
                "direct": direct,
                "crossed": crossed,
            }
        )
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))

    direct_mean = float(np.mean(direct_abs))
    crossed_mean = float(np.mean(crossed_abs))
    passed = direct_mean >= threshold and direct_mean >= crossed_mean + margin
    report.update(
        {
            "passed": passed,
            "threshold": threshold,
            "margin": margin,
            "max_lag_frames": max_lag_frames,
            "min_std": min_std,
            "direct": direct,
            "crossed": crossed,
            "direct_mean_abs_corr": direct_mean,
            "crossed_mean_abs_corr": crossed_mean,
            "failure_reason": None
            if passed
            else "direct correlation is below threshold or not sufficiently above crossed correlation",
        }
    )
    if not passed:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate state/action gripper dimension correlation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-gripper-indices", type=_parse_indices, default=(7, 15))
    parser.add_argument("--action-gripper-indices", type=_parse_indices, default=(7, 15))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-lag-frames", type=int, default=5)
    parser.add_argument("--min-std", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        report = validate_dataset(
            args.dataset,
            state_indices=args.state_gripper_indices,
            action_indices=args.action_gripper_indices,
            threshold=args.threshold,
            margin=args.margin,
            max_lag_frames=args.max_lag_frames,
            min_std=args.min_std,
        )
    except SystemExit as exc:
        try:
            report = json.loads(str(exc))
        except json.JSONDecodeError:
            raise
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1) from None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

