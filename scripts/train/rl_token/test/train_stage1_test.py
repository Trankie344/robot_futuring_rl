from openpi.training.rl_token import config
from scripts.train.rl_token import train_stage1


def test_stage1_entrypoint_uses_standalone_registry(monkeypatch):
    expected = config.get_stage1_config("rl_token_stage1_debug")
    calls = []
    monkeypatch.setattr(train_stage1.rl_token_config, "stage1_cli", lambda: expected)
    monkeypatch.setattr(train_stage1.trainer, "main", lambda value: calls.append(value))

    train_stage1.main()

    assert calls == [expected]
