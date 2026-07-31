import dataclasses

import pytest
import torch

from openpi.models.arm_value.config import ArmValueModelConfig


def _debug_config(**overrides) -> ArmValueModelConfig:
    config = ArmValueModelConfig(
        clip_pretrained_path="__debug__",
        n_history_steps=2,
        frame_gap=2,
        max_state_dim=8,
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    return dataclasses.replace(config, **overrides)


def _batch(config: ArmValueModelConfig) -> dict[str, torch.Tensor]:
    return {
        "images": torch.randn(2, config.sequence_length, 3, 16, 16),
        "states": torch.randn(2, config.sequence_length, 6),
        "lengths": torch.full((2,), config.sequence_length, dtype=torch.long),
        "interval_targets": torch.tensor([[-1, 0], [0, 1]], dtype=torch.long),
        "progress": torch.tensor([0.25, 1.0]),
        "text_input_ids": torch.randint(0, 128, (2, 8)),
        "text_attention_mask": torch.ones(2, 8, dtype=torch.long),
    }


def test_arm_value_config_validates_public_structure_parameters():
    with pytest.raises(ValueError, match="divisible"):
        _debug_config(hidden_dim=30, num_heads=4)
    with pytest.raises(ValueError, match="cannot both be zero"):
        _debug_config(lambda_interval=0.0, lambda_cls=0.0)


def test_arm_value_model_computes_weighted_losses_and_predictions():
    config = _debug_config(lambda_interval=1.5, lambda_cls=0.25)
    model = config.create()
    outputs = model(**_batch(config))
    expected = 1.5 * outputs["arm_interval_loss"] + 0.25 * outputs["arm_cls_loss"]
    torch.testing.assert_close(outputs["loss"].detach(), expected)

    outputs["loss"].backward()
    assert all(parameter.grad is None for parameter in model.clip_model.parameters())
    assert any(parameter.grad is not None for parameter in model.temporal_model.parameters())

    batch = _batch(config)
    success, interval, probabilities = model.predict_advantage(
        batch["images"],
        batch["text_input_ids"],
        batch["text_attention_mask"],
        batch["states"],
        return_interval_probs=True,
    )
    assert success.shape == (2,)
    assert interval.shape == (2, config.n_history_steps)
    assert probabilities.shape == (2, config.n_history_steps, 3)


def test_arm_value_model_rejects_multiple_cameras():
    config = _debug_config()
    model = config.create()
    batch = _batch(config)
    batch["images"] = torch.randn(2, config.sequence_length, 2, 3, 16, 16)
    with pytest.raises(ValueError, match="exactly one camera"):
        model(**batch)


def test_arm_value_model_rejects_invalid_interval_labels():
    config = _debug_config()
    model = config.create()
    batch = _batch(config)
    batch["interval_targets"][0, 0] = 2
    with pytest.raises(ValueError, match="only -1, 0, or 1"):
        model(**batch)
