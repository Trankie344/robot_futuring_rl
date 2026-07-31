#!/usr/bin/env python3
"""Validate every sample in an OpenPI/LeRobot dataset with DataLoader workers."""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _touch_and_check(value, name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _touch_and_check(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _touch_and_check(item, f"{name}.{i}")
        return

    shape = value.shape
    arr = np.asarray(value)
    if arr.size and np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values, shape={shape}, dtype={arr.dtype}")


class SafeOpenPIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, episode_starts: list[int], episodes: list[dict]):
        self._dataset = dataset
        self._episode_starts = episode_starts
        self._episodes = episodes

    def __len__(self) -> int:
        return self._episode_starts[-1]

    def _episode_for_index(self, index: int) -> tuple[int, int]:
        episode_pos = bisect.bisect_right(self._episode_starts, index) - 1
        episode = self._episodes[episode_pos]
        return int(episode["episode_index"]), int(index - self._episode_starts[episode_pos])

    def __getitem__(self, index: int) -> dict:
        episode_index, episode_offset = self._episode_for_index(index)
        try:
            sample = self._dataset[index]
            if not isinstance(sample, dict):
                raise TypeError(f"expected dict sample, got {type(sample)!r}")
            for key in ("state", "actions", "image"):
                if key not in sample:
                    raise KeyError(f"missing {key!r}; keys={sorted(sample)}")
                _touch_and_check(sample[key], key)
            if "image_mask" in sample:
                _touch_and_check(sample["image_mask"], "image_mask")
            return {
                "ok": True,
                "global_index": int(index),
                "episode_index": episode_index,
                "episode_offset": episode_offset,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "global_index": int(index),
                "episode_index": episode_index,
                "episode_offset": episode_offset,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }


def _identity_collate(batch: list[dict]) -> list[dict]:
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-errors", type=int, default=0, help="0 means keep scanning all samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means scan to the end.")
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    episode_starts = [0]
    for episode in episodes:
        episode_starts.append(episode_starts[-1] + int(episode["length"]))
    total_samples = episode_starts[-1]

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=str(dataset_root)))
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    transformed = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=True)
    safe_dataset = SafeOpenPIDataset(transformed, episode_starts, episodes)

    end_index = total_samples if args.limit <= 0 else min(total_samples, args.start_index + args.limit)
    indices = list(range(args.start_index, end_index))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(safe_dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    errors: list[dict] = []
    checked = 0
    started = time.time()
    error_path = output_dir / "openpi_dataset_loading_errors.jsonl"
    with error_path.open("w") as error_file:
        for batch in loader:
            for row in batch:
                checked += 1
                if not row["ok"]:
                    errors.append(row)
                    error_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    error_file.flush()
                    print(
                        "BAD "
                        f"episode={row['episode_index']} offset={row['episode_offset']} "
                        f"global={row['global_index']} error={row['error']}",
                        flush=True,
                    )
                    if args.max_errors and len(errors) >= args.max_errors:
                        break
            if checked % 1000 == 0 or errors:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"checked_samples={checked}/{len(indices)} errors={len(errors)} "
                    f"rate={checked / elapsed:.1f}/s",
                    flush=True,
                )
            if args.max_errors and len(errors) >= args.max_errors:
                break

    bad_episodes = sorted({int(row["episode_index"]) for row in errors})
    summary = {
        "config_name": args.config_name,
        "dataset": str(dataset_root),
        "episodes": len(episodes),
        "total_samples": total_samples,
        "scan_start_index": args.start_index,
        "scan_end_index": end_index,
        "samples_checked": checked,
        "errors": len(errors),
        "bad_episodes": bad_episodes,
        "bad_episode_count": len(bad_episodes),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "elapsed_s": time.time() - started,
    }
    (output_dir / "openpi_dataset_loading_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "bad_episodes.json").write_text(json.dumps(bad_episodes, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
