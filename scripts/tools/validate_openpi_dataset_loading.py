#!/usr/bin/env python3
"""Validate OpenPI/LeRobot dataset loading on selected frames from each episode."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time
import traceback

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _touch_shapes(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _touch_shapes(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _touch_shapes(item)
        return
    _ = value.shape


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sample_offsets(length: int, action_horizon: int, mode: str) -> list[int]:
    if mode == "all":
        return list(range(length))
    safe_last = max(0, length - action_horizon)
    offsets = [0, length // 2, safe_last]
    return sorted(set(max(0, min(length - 1, x)) for x in offsets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["episode_samples", "all"], default="episode_samples")
    parser.add_argument("--max-errors", type=int, default=100)
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=str(dataset_root)))
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=True)

    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    errors: list[dict] = []
    checked = 0
    started = time.time()
    global_start = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        offsets = _sample_offsets(length, cfg.model.action_horizon, "all" if args.mode == "all" else "episode_samples")
        for offset in offsets:
            global_index = global_start + offset
            try:
                sample = dataset[global_index]
                # Touch shapes to force transform outputs to materialize.
                if not isinstance(sample, dict):
                    raise TypeError(f"expected dict sample, got {type(sample)!r}")
                if "state" not in sample or "actions" not in sample or "image" not in sample:
                    raise KeyError(f"missing expected keys, got {sorted(sample)}")
                _touch_shapes(sample["state"])
                _touch_shapes(sample["actions"])
                _touch_shapes(sample["image"])
                if "image_mask" in sample:
                    _touch_shapes(sample["image_mask"])
            except Exception as exc:  # noqa: BLE001
                row = {
                    "episode_index": episode_index,
                    "episode_offset": offset,
                    "global_index": global_index,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(row)
                print(
                    f"BAD episode={episode_index} offset={offset} global={global_index} error={exc!r}",
                    flush=True,
                )
                if len(errors) >= args.max_errors:
                    break
            checked += 1
            if checked % 100 == 0:
                print(f"checked_samples={checked} errors={len(errors)}", flush=True)
        if len(errors) >= args.max_errors:
            break
        global_start += length

    summary = {
        "config_name": args.config_name,
        "dataset": str(dataset_root),
        "mode": args.mode,
        "episodes": len(episodes),
        "samples_checked": checked,
        "errors": len(errors),
        "elapsed_s": time.time() - started,
    }
    (output_dir / "openpi_dataset_loading_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "openpi_dataset_loading_errors.jsonl").open("w") as f:
        for row in errors:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
