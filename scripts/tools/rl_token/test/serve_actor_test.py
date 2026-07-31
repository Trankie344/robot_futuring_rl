from pathlib import Path

import tyro

from openpi.policies.rl_token import actor_policy
from scripts.tools.rl_token import serve_actor


def test_cli_parses_documented_actor_service_arguments():
    args = tyro.cli(
        serve_actor.Args,
        args=[
            "--base-checkpoint=/base/54999",
            "--actor-checkpoint=/actor/5000",
            "--mode=collection",
            "--port=8011",
        ],
    )

    assert args.base_checkpoint == Path("/base/54999")
    assert args.actor_checkpoint == Path("/actor/5000")
    assert args.mode is actor_policy.RLTActorMode.COLLECTION
    assert args.stage1_config == "rl_token_stage1"
    assert args.port == 8011


def test_create_policy_uses_standalone_factory(monkeypatch):
    args = serve_actor.Args(Path("/base/54999"), Path("/actor/5"))
    train_config = object()
    expected = object()
    calls = []
    monkeypatch.setattr(serve_actor.rl_token_config, "get_stage1_config", lambda name: train_config)
    monkeypatch.setattr(
        serve_actor.factory,
        "create_rlt_actor_policy",
        lambda *positional, **keyword: calls.append((positional, keyword)) or expected,
    )

    result = serve_actor.create_policy(args)

    assert result is expected
    assert calls[0][0] == (train_config, args.base_checkpoint, args.actor_checkpoint)
    assert calls[0][1]["mode"] is actor_policy.RLTActorMode.MEAN
