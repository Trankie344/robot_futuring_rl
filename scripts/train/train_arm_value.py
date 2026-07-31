from __future__ import annotations

import dataclasses
import logging
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from openpi.training.arm_value.checkpoint import load_checkpoint
from openpi.training.arm_value.checkpoint import save_checkpoint
from openpi.training.arm_value.config import ArmValueTrainConfig
from openpi.training.arm_value.config import cli
from openpi.training.arm_value.data import ArmValueCollator
from openpi.training.arm_value.data import ArmValueDataset
from openpi.training.arm_value.data import FakeArmValueDataset


def _setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _validate_cuda_support(device_index: int) -> None:
    capability = torch.cuda.get_device_capability(device_index)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and required_arch not in supported_arches:
        device_name = torch.cuda.get_device_name(device_index)
        supported_text = ", ".join(sorted(supported_arches))
        raise RuntimeError(
            f"CUDA device {device_name!r} requires {required_arch}, but this PyTorch build supports "
            f"{supported_text}. Install a compatible PyTorch/CUDA build or run with --device=cpu."
        )


def _setup_distributed(config: ArmValueTrainConfig) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("ARM value training requested CUDA, but torch.cuda.is_available() is false")
        _validate_cuda_support(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device


def _validate_clip_assets(config: ArmValueTrainConfig) -> None:
    if config.model.clip_pretrained_path == "__debug__":
        return
    path = Path(config.model.clip_pretrained_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"CLIP checkpoint directory does not exist: {path}. Download "
            "openai/clip-vit-base-patch32 into ./checkpoints/clip-vit-base-patch32 "
            "or override --model.clip-pretrained-path."
        )
    required_files = ("config.json", "preprocessor_config.json")
    missing = [name for name in required_files if not (path / name).is_file()]
    has_weights = any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    has_tokenizer = (path / "tokenizer.json").is_file() or (
        (path / "vocab.json").is_file() and (path / "merges.txt").is_file()
    )
    if missing or not has_weights or not has_tokenizer:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if not has_weights:
            details.append("missing model.safetensors or pytorch_model.bin")
        if not has_tokenizer:
            details.append("missing tokenizer.json or vocab.json+merges.txt")
        raise FileNotFoundError(f"Incomplete CLIP checkpoint at {path}: {'; '.join(details)}")


def _seed_everything(seed: int, rank: int, device: torch.device) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _prepare_output_dir(config: ArmValueTrainConfig, rank: int, world_size: int) -> Path:
    output_dir = config.output_dir
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()) and config.resume_from is None:
            if not config.overwrite:
                raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    return output_dir


def _build_data_loader(
    config: ArmValueTrainConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[DataLoader, DistributedSampler | None, dict[str, Any]]:
    if config.data.repo_id == "fake":
        dataset = FakeArmValueDataset(config.model)
        collate_fn = None
    else:
        dataset = ArmValueDataset(config.data, config.model)
        collate_fn = ArmValueCollator(
            config.model.clip_pretrained_path,
            local_files_only=config.model.clip_local_files_only,
        )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    if len(data_loader) == 0:
        raise ValueError("ARM value data loader has no complete batches; reduce batch_size or provide more samples")
    return data_loader, sampler, dict(dataset.progress_summary)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ArmValueTrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    end_ratio = config.end_learning_rate / config.learning_rate

    def scale(step: int) -> float:
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return float(step + 1) / float(config.warmup_steps)
        decay_steps = max(1, config.num_train_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_ratio + (1.0 - end_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        reduced[key] = float(tensor.cpu())
    return reduced


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def main(config: ArmValueTrainConfig) -> None:
    rank, local_rank, world_size, device = _setup_distributed(config)
    _setup_logging(rank)
    wandb_run = None
    try:
        _seed_everything(config.seed, rank, device)
        _validate_clip_assets(config)
        output_dir = _prepare_output_dir(config, rank, world_size)
        logging.info("ARM value config: %s", config)
        logging.info(
            "Training on rank=%d local_rank=%d world_size=%d device=%s",
            rank,
            local_rank,
            world_size,
            device,
        )

        data_loader, sampler, progress_summary = _build_data_loader(
            config,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        model = config.model.create().to(device)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_scheduler(optimizer, config)
        start_step = 0
        if config.resume_from is not None:
            checkpoint = load_checkpoint(
                config.resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
            )
            start_step = int(checkpoint["step"]) + 1
            logging.info("Resumed ARM value training from step %d", start_step)

        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )

        if rank == 0 and config.wandb_enabled:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.exp_name,
                config={
                    "train": dataclasses.asdict(config),
                    "progress_summary": progress_summary,
                },
            )

        autocast_enabled = config.precision == "bfloat16"
        cpu_has_bf16 = bool(getattr(torch.backends.cpu, "has_bf16", False))
        if autocast_enabled and device.type == "cpu" and not cpu_has_bf16:
            raise RuntimeError("bfloat16 training was requested but this CPU does not support bf16")

        iterator = iter(data_loader)
        epoch = 0
        for step in range(start_step, config.num_train_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(data_loader)
                batch = next(iterator)
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                outputs = model(**batch)
                loss = outputs["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            metrics = {
                **outputs,
                "grad_norm": torch.as_tensor(grad_norm, device=device),
                "learning_rate": torch.tensor(optimizer.param_groups[0]["lr"], device=device),
            }
            if step % config.log_interval == 0 or step == config.num_train_steps - 1:
                reduced = _reduce_metrics(metrics, world_size)
                if rank == 0:
                    logging.info(
                        "step=%d %s",
                        step,
                        " ".join(f"{key}={value:.6f}" for key, value in reduced.items()),
                    )
                    if wandb_run is not None:
                        wandb_run.log(reduced, step=step)

            should_save = (step + 1) % config.save_interval == 0 or step == config.num_train_steps - 1
            if rank == 0 and should_save:
                unwrapped = _unwrap_model(model)
                save_checkpoint(
                    output_dir / f"step_{step:08d}.pt",
                    model=unwrapped,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=config,
                    progress_summary=progress_summary,
                )
                save_checkpoint(
                    output_dir / "latest.pt",
                    model=unwrapped,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=config,
                    progress_summary=progress_summary,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(cli())
