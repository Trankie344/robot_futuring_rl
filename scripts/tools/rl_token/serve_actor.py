"""Serve a completed RL Token Stage 2 actor through OpenPI WebSocket inference."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies.rl_token import actor_policy
from openpi.policies.rl_token import factory
from openpi.serving import websocket_policy_server
from openpi.training.rl_token import config as rl_token_config


@dataclasses.dataclass(frozen=True)
class Args:
    base_checkpoint: Path
    actor_checkpoint: Path
    mode: tyro.conf.EnumChoicesFromValues[actor_policy.RLTActorMode] = actor_policy.RLTActorMode.MEAN
    port: int = 8011
    stage1_config: str = "rl_token_stage1"
    sampler_num_steps: int = 10
    seed: int = 0
    default_prompt: str = "fold clothes"
    record: bool = False


def create_policy(args: Args) -> _policy.BasePolicy:
    train_config = rl_token_config.get_stage1_config(args.stage1_config)
    return factory.create_rlt_actor_policy(
        train_config,
        args.base_checkpoint,
        args.actor_checkpoint,
        mode=args.mode,
        sampler_num_steps=args.sampler_num_steps,
        default_prompt=args.default_prompt,
        seed=args.seed,
    )


def main(args: Args | None = None) -> None:
    args = tyro.cli(Args) if args is None else args
    policy = create_policy(args)
    metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records/rl_token")

    hostname = socket.gethostname()
    logging.info("Serving RL Token actor from %s on 0.0.0.0:%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
