from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ArmValueModelConfig:
    """Configuration for the standalone ARM value model."""

    clip_pretrained_path: str = "./checkpoints/clip-vit-base-patch32"
    clip_local_files_only: bool = True
    n_history_steps: int = 4
    frame_gap: int = 30
    max_state_dim: int = 32
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 8
    dropout: float = 0.1
    lambda_interval: float = 1.0
    lambda_cls: float = 1.0
    success_eps: float = 1e-3
    freeze_clip_backbone: bool = True

    def __post_init__(self) -> None:
        if not self.clip_pretrained_path:
            raise ValueError("clip_pretrained_path cannot be empty")
        if self.n_history_steps < 1:
            raise ValueError("n_history_steps must be at least 1")
        if self.frame_gap < 1:
            raise ValueError("frame_gap must be at least 1")
        if self.max_state_dim < 1:
            raise ValueError("max_state_dim must be at least 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.lambda_interval < 0.0 or self.lambda_cls < 0.0:
            raise ValueError("ARM loss weights must be non-negative")
        if self.lambda_interval == 0.0 and self.lambda_cls == 0.0:
            raise ValueError("lambda_interval and lambda_cls cannot both be zero")
        if not 0.0 <= self.success_eps < 1.0:
            raise ValueError("success_eps must be in [0, 1)")

    @property
    def sequence_length(self) -> int:
        return self.n_history_steps + 1

    def create(self):
        from openpi.models.arm_value.model import ArmValueModel  # noqa: PLC0415

        return ArmValueModel(self)
