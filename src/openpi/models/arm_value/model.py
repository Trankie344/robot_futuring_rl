# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI-local ARM value model adapted from FluxVLA's ARM implementation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing_extensions import override

from openpi.models.arm_value.config import ArmValueModelConfig


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = self.alpha * (1.0 - p_t) ** self.gamma * ce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class TemporalAdvantageTransformer(nn.Module):
    """Shared temporal encoder with interval and task-success heads."""

    def __init__(
        self,
        *,
        video_dim: int,
        state_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        center_idx: int,
    ) -> None:
        super().__init__()
        self.center_idx = center_idx
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.interval_head = nn.Linear(2 * hidden_dim, 3)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        video_features: torch.Tensor,
        state_features: torch.Tensor,
        text_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = video_features.shape
        if sequence_length < 2:
            raise ValueError(f"ARM expects at least two frames, got {sequence_length}")

        fused_tokens = (
            self.video_proj(video_features)
            + self.state_proj(state_features)
            + self.text_proj(text_features).unsqueeze(1).expand(batch_size, sequence_length, -1)
        )
        step_ids = torch.arange(sequence_length, device=fused_tokens.device).unsqueeze(0)
        padding_mask = step_ids >= lengths.to(fused_tokens.device).unsqueeze(1)
        hidden_states = self.transformer(fused_tokens, src_key_padding_mask=padding_mask)

        pair_features = torch.cat([hidden_states[:, :-1], hidden_states[:, 1:]], dim=-1)
        interval_logits = self.interval_head(pair_features)
        center_idx = min(self.center_idx, sequence_length - 1)
        cls_logits = self.cls_head(hidden_states[:, center_idx]).squeeze(-1)
        return interval_logits, cls_logits


class _DebugClipEncoder(nn.Module):
    """Small CLIP-shaped encoder used only by the registered debug config."""

    def __init__(self, projection_dim: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=projection_dim)
        self.image_projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, projection_dim))
        self.text_embedding = nn.Embedding(128, projection_dim)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_projection(pixel_values.float())

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.text_embedding(input_ids)
        mask = attention_mask.to(embedded.dtype).unsqueeze(-1)
        return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _feature_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError(f"Expected a tensor-like CLIP feature output, got {type(value)!r}")


class ArmValueModel(nn.Module):
    """CLIP-backed ARM value model with interval and success objectives."""

    def __init__(self, config: ArmValueModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.clip_pretrained_path == "__debug__":
            self.clip_model = _DebugClipEncoder()
        else:
            from transformers import CLIPModel  # noqa: PLC0415

            self.clip_model = CLIPModel.from_pretrained(
                config.clip_pretrained_path,
                local_files_only=config.clip_local_files_only,
            )

        projection_dim = int(self.clip_model.config.projection_dim)
        self.temporal_model = TemporalAdvantageTransformer(
            video_dim=projection_dim,
            state_dim=config.max_state_dim,
            text_dim=projection_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            center_idx=config.n_history_steps,
        )
        self.focal_loss = FocalLoss(alpha=2.0, gamma=2.0)
        if config.freeze_clip_backbone:
            self.clip_model.requires_grad_(requires_grad=False)

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_clip_backbone:
            self.clip_model.eval()
        return self

    def _pad_states(self, states: torch.Tensor) -> torch.Tensor:
        state_dim = states.shape[-1]
        if state_dim > self.config.max_state_dim:
            raise ValueError(f"State dimension {state_dim} exceeds max_state_dim={self.config.max_state_dim}")
        if state_dim == self.config.max_state_dim:
            return states
        return functional.pad(states, (0, self.config.max_state_dim - state_dim))

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            images = images.unsqueeze(2)
        if images.ndim != 6:
            raise ValueError(f"Expected images [B,T,N,C,H,W] or [B,T,C,H,W], got {tuple(images.shape)}")
        batch_size, sequence_length, num_cameras, channels, height, width = images.shape
        if num_cameras != 1:
            raise ValueError(f"ARM value model supports exactly one camera, got {num_cameras}")
        flat_images = images.reshape(batch_size * sequence_length, channels, height, width)
        features = _feature_tensor(self.clip_model.get_image_features(pixel_values=flat_images))
        return features.reshape(batch_size, sequence_length, -1)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return _feature_tensor(self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask))

    def _predict_logits(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(dtype=next(self.clip_model.parameters()).dtype)
        states = self._pad_states(states.to(dtype=images.dtype))
        image_features = self._encode_images(images)
        text_features = self._encode_text(text_input_ids, text_attention_mask)
        return self.temporal_model(image_features, states, text_features, lengths)

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        states: torch.Tensor,
        lengths: torch.Tensor,
        interval_targets: torch.Tensor,
        progress: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        interval_logits, cls_logits = self._predict_logits(images, text_input_ids, text_attention_mask, states, lengths)
        expected_shape = interval_logits.shape[:2]
        if tuple(interval_targets.shape) != tuple(expected_shape):
            raise ValueError(
                f"interval_targets shape mismatch: expected {tuple(expected_shape)}, got {tuple(interval_targets.shape)}"
            )
        if torch.any((interval_targets < -1) | (interval_targets > 1)):
            raise ValueError("interval_targets must contain only -1, 0, or 1")
        mapped_targets = interval_targets.long() + 1
        interval_loss = functional.cross_entropy(interval_logits.reshape(-1, 3), mapped_targets.reshape(-1))
        interval_acc = (interval_logits.argmax(dim=-1) == mapped_targets).float().mean()

        progress = progress.float().reshape(-1)
        success_targets = (progress >= 1.0 - self.config.success_eps).float()
        cls_loss = self.focal_loss(cls_logits, success_targets)
        cls_acc = ((torch.sigmoid(cls_logits) >= 0.5).float() == success_targets).float().mean()
        total_loss = self.config.lambda_interval * interval_loss + self.config.lambda_cls * cls_loss
        return {
            "loss": total_loss,
            "arm_interval_loss": interval_loss.detach(),
            "arm_cls_loss": cls_loss.detach(),
            "arm_interval_acc": interval_acc.detach(),
            "arm_cls_acc": cls_acc.detach(),
        }

    @torch.inference_mode()
    def predict_advantage(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        states: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        return_interval_probs: bool = False,
    ):
        if lengths is None:
            lengths = torch.full((images.shape[0],), images.shape[1], device=images.device, dtype=torch.long)
        interval_logits, cls_logits = self._predict_logits(images, text_input_ids, text_attention_mask, states, lengths)
        success_prob = torch.sigmoid(cls_logits)
        interval_probs = interval_logits.softmax(dim=-1)
        interval_pred = interval_probs.argmax(dim=-1) - 1
        if return_interval_probs:
            return success_prob, interval_pred, interval_probs
        return success_prob, interval_pred
