#!/usr/bin/env python3
"""Compute OpenPI norm stats from LeRobot parquet files without decoding videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from openpi.shared import normalize
from openpi.training import config as _config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _episode_path(root: Path, info: dict, episode_index: int) -> Path:
    data_path = info["data_path"]
    if "{episode_chunk" in data_path:
        episode_chunk = _episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
        return root / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return root / data_path.format(episode_index=episode_index)


def _stack_column(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack(df[key].to_numpy()).astype(np.float32, copy=False)


def _action_windows(actions: np.ndarray, horizon: int, chunk_size: int) -> np.ndarray:
    length = actions.shape[0]
    offsets = np.arange(horizon, dtype=np.int64)
    windows = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        indices = np.minimum(np.arange(start, stop, dtype=np.int64)[:, None] + offsets[None, :], length - 1)
        windows.append(actions[indices])
    return np.concatenate(windows, axis=0)


def _make_delta_mask(name: str, action_dim: int) -> np.ndarray:
    if name == "all":
        return np.ones(action_dim, dtype=bool)
    if name == "joints":
        return np.asarray((True,) * 7 + (False,) + (True,) * 7 + (False,), dtype=bool)
    raise ValueError(f"Unsupported delta mask: {name}")


def _delta_actions(windows: np.ndarray, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    windows = windows.copy()
    windows[..., : mask.shape[0]] -= np.where(mask, state[:, None, : mask.shape[0]], 0.0)
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--delta-mask",
        choices=("joints", "all"),
        default="all",
        help="Action dimensions converted to deltas before computing action norm stats.",
    )
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    root = Path(data_config.repo_id)
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    horizon = int(train_config.model.action_horizon)
    action_dim = int(info["features"]["action"]["shape"][0])
    delta_mask = _make_delta_mask(args.delta_mask, action_dim)
    print(f"Using delta mask {args.delta_mask}: {delta_mask.tolist()}")

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for episode in tqdm.tqdm(episodes, desc="Computing parquet stats"):
        episode_index = int(episode["episode_index"])
        df = pd.read_parquet(_episode_path(root, info, episode_index), columns=["observation.state", "action"])
        state = _stack_column(df, "observation.state")
        actions = _stack_column(df, "action")

        stats["state"].update(state)
        for start in range(0, len(actions), args.chunk_size):
            stop = min(start + args.chunk_size, len(actions))
            offsets = np.arange(horizon, dtype=np.int64)
            indices = np.minimum(np.arange(start, stop, dtype=np.int64)[:, None] + offsets[None, :], len(actions) - 1)
            action_windows = _delta_actions(actions[indices], state[start:stop], delta_mask)
            stats["actions"].update(action_windows)

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}

    assets = getattr(train_config.data, "assets", None)
    assets_dir = Path(getattr(assets, "assets_dir", None) or train_config.assets_dirs)
    output_path = assets_dir / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
