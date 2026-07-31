import dataclasses

import pytest
import torch

from openpi.training.arm_value.checkpoint import load_checkpoint
from openpi.training.arm_value.checkpoint import save_checkpoint
from openpi.training.arm_value.config import get_config


def test_arm_value_checkpoint_round_trip_and_compatibility(tmp_path):
    config = dataclasses.replace(
        get_config("arm_value_debug"),
        output_base_dir=str(tmp_path),
        exp_name="checkpoint_test",
    )
    model = config.model.create()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=7,
        config=config,
        progress_summary={"valid_rows": 4},
    )

    with torch.no_grad():
        next(model.temporal_model.parameters()).zero_()
    checkpoint = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, config=config)
    assert checkpoint["step"] == 7
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])

    incompatible = dataclasses.replace(config, model=dataclasses.replace(config.model, hidden_dim=64))
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(path, model=model, config=incompatible)
