from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


class ArmRABCWeighter:
    """Compute ARM RA-BC per-sample weights from per-frame progress."""

    progress_column = "progress"

    def __init__(
        self,
        progress_path: str | Path,
        *,
        chunk_size: int,
        kappa: float = 0.01,
        epsilon: float = 1e-6,
        fallback_weight: float = 0.0,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        self.progress_path = Path(progress_path)
        self.chunk_size = int(chunk_size)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.fallback_weight = float(fallback_weight)

        table = pq.read_table(self.progress_path)
        data = table.to_pydict()
        if self.progress_column not in data:
            raise ValueError(f"Column {self.progress_column!r} is missing from {self.progress_path}")

        self.progress_lookup: dict[int, float] = {}
        self.episode_lookup: dict[int, tuple[int, int]] = {}
        self.episode_boundaries: dict[tuple[int, int], dict[str, int]] = {}
        self._load_rows(data)
        self._compute_global_delta_stats()

    def _load_rows(self, data: dict[str, list[Any]]) -> None:
        indices = data["index"]
        dataset_indices = data.get("dataset_index", [0] * len(indices))
        episode_indices = data["episode_index"]
        progress_values = data[self.progress_column]

        for idx, dataset_idx, episode_idx, progress in zip(
            indices, dataset_indices, episode_indices, progress_values, strict=True
        ):
            global_idx = int(idx)
            episode_key = (int(dataset_idx), int(episode_idx))
            self.episode_lookup[global_idx] = episode_key
            if progress is not None and np.isfinite(progress):
                self.progress_lookup[global_idx] = float(progress)

            bounds = self.episode_boundaries.setdefault(episode_key, {"start": global_idx, "end": global_idx + 1})
            bounds["start"] = min(bounds["start"], global_idx)
            bounds["end"] = max(bounds["end"], global_idx + 1)

        logging.info("Loaded %d valid ARM progress rows from %s", len(self.progress_lookup), self.progress_path)

    def _compute_global_delta_stats(self) -> None:
        deltas = []
        for global_idx, progress in self.progress_lookup.items():
            future_progress = self._future_progress(global_idx)
            if future_progress is not None:
                deltas.append(future_progress - progress)

        if not deltas:
            self.delta_mean = 0.0
            self.delta_std = self.epsilon
            logging.warning("No valid ARM RA-BC progress deltas found in %s", self.progress_path)
            return

        self.delta_mean = max(float(np.mean(deltas)), 0.0)
        self.delta_std = max(float(np.std(deltas)), self.epsilon)
        logging.info("ARM RA-BC delta stats: mean=%.6f std=%.6f", self.delta_mean, self.delta_std)

    def _future_progress(self, global_idx: int) -> float | None:
        episode_key = self.episode_lookup.get(global_idx)
        if episode_key is None:
            return None
        bounds = self.episode_boundaries.get(episode_key)
        if bounds is None:
            return None
        future_idx = min(global_idx + self.chunk_size, bounds["end"] - 1)
        return self.progress_lookup.get(future_idx)

    def _compute_delta(self, global_idx: int) -> float:
        progress = self.progress_lookup.get(global_idx)
        if progress is None:
            return float("nan")
        future_progress = self._future_progress(global_idx)
        if future_progress is None:
            return float("nan")
        return future_progress - progress

    def _compute_weights(self, deltas: np.ndarray) -> np.ndarray:
        valid_mask = ~np.isnan(deltas)
        lower_bound = self.delta_mean - 2.0 * self.delta_std
        soft_weights = (deltas - lower_bound) / (4.0 * self.delta_std + self.epsilon)
        soft_weights = np.clip(soft_weights, 0.0, 1.0)

        weights = np.zeros_like(deltas, dtype=np.float32)
        weights[deltas > self.kappa] = 1.0
        moderate_mask = (deltas >= 0.0) & (deltas <= self.kappa)
        weights[moderate_mask] = soft_weights[moderate_mask]
        weights[~valid_mask] = self.fallback_weight
        return weights

    def compute_weight(self, index: int) -> float:
        delta = self._compute_delta(int(index))
        return float(self._compute_weights(np.asarray([delta], dtype=np.float32))[0])


class ArmWeightedDataset:
    """Dataset wrapper that attaches an ARM RA-BC sample weight to each sample."""

    def __init__(
        self,
        dataset,
        progress_path: str | Path,
        *,
        action_horizon: int,
        kappa: float = 0.01,
        fallback_weight: float = 0.0,
    ) -> None:
        self._dataset = dataset
        self._weighter = ArmRABCWeighter(
            progress_path,
            chunk_size=action_horizon,
            kappa=kappa,
            fallback_weight=fallback_weight,
        )

    def __getitem__(self, index):
        sample = dict(self._dataset[index])
        sample_index = sample.get("index", index)
        if hasattr(sample_index, "item"):
            sample_index = sample_index.item()
        sample["sample_weight"] = np.asarray(self._weighter.compute_weight(int(sample_index)), dtype=np.float32)
        return sample

    def __len__(self) -> int:
        return len(self._dataset)

