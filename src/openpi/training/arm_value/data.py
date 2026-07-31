from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from openpi.models.arm_value.config import ArmValueModelConfig
from openpi.training.arm_value.config import ArmValueDataConfig


@dataclasses.dataclass(frozen=True)
class ProgressRecord:
    index: int
    episode_index: int
    frame_index: int
    progress: float | None


class ArmProgressTable:
    """Validated per-frame progress lookup used to construct ARM targets."""

    def __init__(self, records: Sequence[ProgressRecord], *, interval_eps: float) -> None:
        self.interval_eps = float(interval_eps)
        self.records: dict[int, ProgressRecord] = {}
        self.episode_bounds: dict[int, tuple[int, int]] = {}
        for record in records:
            if record.index in self.records:
                raise ValueError(f"Duplicate progress index {record.index}")
            self.records[record.index] = record
            bounds = self.episode_bounds.get(record.episode_index)
            if bounds is None:
                self.episode_bounds[record.episode_index] = (record.index, record.index + 1)
            else:
                self.episode_bounds[record.episode_index] = (
                    min(bounds[0], record.index),
                    max(bounds[1], record.index + 1),
                )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        dataset_index: int,
        interval_eps: float,
    ) -> ArmProgressTable:
        path = Path(path)
        data = pq.read_table(path).to_pydict()
        required = {"index", "episode_index", "frame_index", "progress"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Progress parquet {path} is missing columns: {sorted(missing)}")

        dataset_indices = data.get("dataset_index", [0] * len(data["index"]))
        valid_labels = data.get("valid_label", [True] * len(data["index"]))
        records = []
        for index, row_dataset_index, episode_index, frame_index, progress, valid_label in zip(
            data["index"],
            dataset_indices,
            data["episode_index"],
            data["frame_index"],
            data["progress"],
            valid_labels,
            strict=True,
        ):
            if int(row_dataset_index) != dataset_index:
                continue
            value = None
            if bool(valid_label) and progress is not None and np.isfinite(progress):
                value = float(progress)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Progress at index {index} must be in [0, 1], got {value}")
            records.append(ProgressRecord(int(index), int(episode_index), int(frame_index), value))
        if not records:
            raise ValueError(f"No progress rows for dataset_index={dataset_index} in {path}")
        return cls(records, interval_eps=interval_eps)

    def window_indices(self, current_index: int, *, n_history_steps: int, frame_gap: int) -> list[int]:
        current = self.records.get(int(current_index))
        if current is None:
            raise KeyError(f"Progress index {current_index} is missing")
        episode_start, _ = self.episode_bounds[current.episode_index]
        return [max(episode_start, current.index - step * frame_gap) for step in range(n_history_steps, -1, -1)]

    def progress_sequence(self, indices: Sequence[int], *, episode_index: int) -> np.ndarray | None:
        values = []
        for index in indices:
            record = self.records.get(int(index))
            if record is None or record.episode_index != episode_index or record.progress is None:
                return None
            values.append(record.progress)
        return np.asarray(values, dtype=np.float32)

    def valid_indices(self, *, n_history_steps: int, frame_gap: int) -> list[int]:
        valid = []
        for current_index, record in sorted(self.records.items()):
            indices = self.window_indices(
                current_index,
                n_history_steps=n_history_steps,
                frame_gap=frame_gap,
            )
            if self.progress_sequence(indices, episode_index=record.episode_index) is not None:
                valid.append(current_index)
        return valid

    def interval_targets(self, progress_sequence: np.ndarray) -> np.ndarray:
        deltas = progress_sequence[1:] - progress_sequence[:-1]
        targets = np.zeros_like(deltas, dtype=np.int64)
        targets[deltas > self.interval_eps] = 1
        targets[deltas < -self.interval_eps] = -1
        return targets

    def summary(self, *, n_history_steps: int, frame_gap: int) -> dict[str, int | float]:
        valid = self.valid_indices(n_history_steps=n_history_steps, frame_gap=frame_gap)
        return {
            "total_rows": len(self.records),
            "valid_rows": len(valid),
            "filtered_rows": len(self.records) - len(valid),
            "episodes": len(self.episode_bounds),
            "interval_eps": self.interval_eps,
        }


def load_state_norm_stats(path: str | Path, *, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    stats: Mapping[str, Any] = payload.get("norm_stats", payload)
    state_stats = stats.get("state", stats.get("observation.state"))
    if state_stats is None:
        raise ValueError(f"No state statistics found in {path}")
    mean = np.asarray(state_stats["mean"], dtype=np.float32)
    std = np.asarray(state_stats["std"], dtype=np.float32)
    if mean.shape != (state_dim,) or std.shape != (state_dim,):
        raise ValueError(f"State norm stats must have shape {(state_dim,)}, got mean={mean.shape}, std={std.shape}")
    return mean, np.maximum(std, 1e-6)


def _task_text(tasks: Any, task_index: int) -> str:
    value = tasks.get(task_index, tasks.get(str(task_index))) if isinstance(tasks, Mapping) else tasks[task_index]
    if isinstance(value, Mapping):
        value = value.get("task", value.get("description", value.get("name")))
    if value is None:
        raise ValueError(f"Cannot resolve task text for task_index={task_index}")
    return str(value)


class ArmValueDataset(Dataset):
    """LeRobot-backed causal observation dataset for ARM value training."""

    def __init__(self, data_config: ArmValueDataConfig, model_config: ArmValueModelConfig) -> None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: PLC0415

        self.data_config = data_config
        self.model_config = model_config
        self.metadata = LeRobotDatasetMetadata(data_config.repo_id)
        offsets = [
            -step * model_config.frame_gap / float(self.metadata.fps)
            for step in range(model_config.n_history_steps, -1, -1)
        ]
        self.dataset = LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={data_config.camera_key: offsets, data_config.state_key: offsets},
            tolerance_s=data_config.video_tolerance_s,
        )
        self.progress_table = ArmProgressTable.from_parquet(
            data_config.progress_path,
            dataset_index=data_config.dataset_index,
            interval_eps=data_config.interval_eps,
        )
        valid_indices = self.progress_table.valid_indices(
            n_history_steps=model_config.n_history_steps,
            frame_gap=model_config.frame_gap,
        )
        self.indices = [index for index in valid_indices if 0 <= index < len(self.dataset)]
        if not self.indices:
            raise ValueError("No valid ARM value samples remain after progress and dataset alignment")

        first_state = np.asarray(self.dataset[self.indices[0]][data_config.state_key])
        state_dim = int(first_state.shape[-1])
        if state_dim > model_config.max_state_dim:
            raise ValueError(f"State dimension {state_dim} exceeds max_state_dim={model_config.max_state_dim}")
        self.state_mean, self.state_std = load_state_norm_stats(data_config.norm_stats_path, state_dim=state_dim)
        self.progress_summary = self.progress_table.summary(
            n_history_steps=model_config.n_history_steps,
            frame_gap=model_config.frame_gap,
        )
        self.progress_summary["aligned_rows"] = len(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        global_index = self.indices[item]
        sample = self.dataset[global_index]
        current_record = self.progress_table.records[global_index]
        sample_episode_index = int(np.asarray(sample["episode_index"]).reshape(-1)[-1])
        sample_frame_index = int(np.asarray(sample["frame_index"]).reshape(-1)[-1])
        if sample_episode_index != current_record.episode_index:
            raise ValueError(
                f"Episode mismatch at index {global_index}: dataset={sample_episode_index} "
                f"progress={current_record.episode_index}"
            )
        if sample_frame_index != current_record.frame_index:
            raise ValueError(
                f"Frame mismatch at index {global_index}: dataset={sample_frame_index} "
                f"progress={current_record.frame_index}"
            )
        window_indices = self.progress_table.window_indices(
            global_index,
            n_history_steps=self.model_config.n_history_steps,
            frame_gap=self.model_config.frame_gap,
        )
        progress_sequence = self.progress_table.progress_sequence(
            window_indices,
            episode_index=current_record.episode_index,
        )
        if progress_sequence is None:
            raise RuntimeError(f"Progress sequence became invalid for prevalidated index {global_index}")

        states = np.asarray(sample[self.data_config.state_key], dtype=np.float32)
        if states.ndim == 1:
            states = states[None]
        states = (states - self.state_mean) / self.state_std
        states = np.pad(states, ((0, 0), (0, self.model_config.max_state_dim - states.shape[-1])))

        images = np.asarray(sample[self.data_config.camera_key])
        if images.ndim == 3:
            images = images[None]
        task_index = int(np.asarray(sample.get("task_index", 0)).reshape(-1)[-1])
        return {
            "images": images,
            "states": states.astype(np.float32),
            "lengths": np.asarray(self.model_config.sequence_length, dtype=np.int64),
            "interval_targets": self.progress_table.interval_targets(progress_sequence),
            "progress": np.asarray(progress_sequence[-1], dtype=np.float32),
            "task_description": _task_text(self.metadata.tasks, task_index),
            "index": np.asarray(global_index, dtype=np.int64),
        }


def _image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a single image with 3 dimensions, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


class ArmValueCollator:
    def __init__(self, clip_pretrained_path: str, *, local_files_only: bool = True) -> None:
        from transformers import CLIPProcessor  # noqa: PLC0415

        self.processor = CLIPProcessor.from_pretrained(
            clip_pretrained_path,
            local_files_only=local_files_only,
        )

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(samples)
        sequence_length = int(np.asarray(samples[0]["images"]).shape[0])
        flat_images = [_image_to_hwc_uint8(frame) for sample in samples for frame in np.asarray(sample["images"])]
        pixel_values = self.processor.image_processor(images=flat_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.reshape(batch_size, sequence_length, *pixel_values.shape[1:])
        text = self.processor.tokenizer(
            [sample["task_description"] for sample in samples],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return {
            "images": pixel_values,
            "states": torch.from_numpy(np.stack([sample["states"] for sample in samples])),
            "lengths": torch.from_numpy(np.stack([sample["lengths"] for sample in samples])),
            "interval_targets": torch.from_numpy(np.stack([sample["interval_targets"] for sample in samples])),
            "progress": torch.from_numpy(np.stack([sample["progress"] for sample in samples])),
            "text_input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
        }


class FakeArmValueDataset(Dataset):
    """Deterministic, already-collated samples for CPU smoke tests."""

    def __init__(self, model_config: ArmValueModelConfig, *, size: int = 8) -> None:
        self.model_config = model_config
        self.size = size
        self.progress_summary = {"total_rows": size, "valid_rows": size, "filtered_rows": 0, "episodes": 1}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        sequence_length = self.model_config.sequence_length
        interval_targets = torch.tensor(
            [(-1, 0, 1)[(index + step) % 3] for step in range(sequence_length - 1)], dtype=torch.long
        )
        return {
            "images": torch.rand(sequence_length, 3, 32, 32, generator=generator),
            "states": torch.randn(sequence_length, self.model_config.max_state_dim, generator=generator),
            "lengths": torch.tensor(sequence_length, dtype=torch.long),
            "interval_targets": interval_targets,
            "progress": torch.tensor(float(index % 2), dtype=torch.float32),
            "text_input_ids": torch.randint(0, 128, (8,), generator=generator),
            "text_attention_mask": torch.ones(8, dtype=torch.long),
        }
