"""Resume-copy the external RL Token checkpoints and Stage 1 dataset."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
import subprocess

import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    source_base: Path = Path(
        "/mnt/workspace/xdm/openpi/checkpoints/pi05_lite0030_joints_full_finetune/"
        "lite0030_pi05_relact_absstate_drop_last4s_min20s_bs256_30k/29999"
    )
    source_stage1: Path = Path(
        "/mnt/workspace/xdm/openpi-rltoken-rda-lite0028-white-77274f7/checkpoints/"
        "pi05_lite0030_joints_rltoken_only/lite0030_pi05_rltoken_only_bs512_30k_20260715/54999"
    )
    source_dataset: Path = Path(
        "/mnt/workspace/robot_task_raw/lite-0030/merged/"
        "lite-0030_2026-06-17_18_22_joints_filtered_fps20_lerobotv21_openpi_drop_last4s_min20s"
    )
    destination_root: Path = Path(".")
    verify_only: bool = False


def _rsync(source: Path, destination: Path, *, verify_only: bool) -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is required; install it before preparing RL Token assets")
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--partial", "--info=progress2"]
    if verify_only:
        command = ["rsync", "-rcn", "--delete"]
    command.extend([f"{source}/", f"{destination}/"])
    subprocess.run(command, check=True)


def run(args: Args) -> None:
    root = args.destination_root.resolve()
    copies = (
        (args.source_base, root / "checkpoints/rl_token/pi05_lite0030_base/29999"),
        (args.source_stage1, root / "checkpoints/rl_token/pi05_lite0030_rltoken_only/54999"),
        (args.source_dataset, root / "data/rl_token/lite0030_stage1"),
    )
    for source, destination in copies:
        if not source.is_dir():
            raise FileNotFoundError(source)
        _rsync(source, destination, verify_only=args.verify_only)


if __name__ == "__main__":
    run(tyro.cli(Args))
