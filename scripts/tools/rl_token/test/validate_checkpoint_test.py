from pathlib import Path

import pytest

from scripts.tools.rl_token import validate_checkpoint


def test_stage1_validator_requires_numeric_step(tmp_path: Path):
    with pytest.raises(ValueError, match="numeric"):
        validate_checkpoint.validate_stage1(tmp_path / "latest", asset_id="asset")
