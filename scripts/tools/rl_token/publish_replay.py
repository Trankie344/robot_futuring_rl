"""Atomically append one authenticated cache shard to a replay snapshot."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import tyro

from openpi.training.rl_token.stage2 import cache
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import replay


@dataclasses.dataclass(frozen=True)
class Args:
    cache_shard: Path
    admission: Path
    output: Path
    previous: Path | None = None


def run(args: Args) -> replay.ReplaySnapshot:
    previous = None if args.previous is None else replay.open_snapshot(args.previous)
    admission_sha256 = identity.sha256_file(args.admission)
    with cache.open_shard(args.cache_shard) as shard:
        return replay.create_snapshot(
            args.output,
            previous=previous,
            new_shard=shard,
            admission_sha256=admission_sha256,
        )


def main() -> None:
    snapshot = run(tyro.cli(Args))
    print(
        json.dumps(
            {
                "path": str(snapshot.path),
                "sha256": snapshot.sha256,
                "total_transitions": snapshot.total_transitions,
                "shards": len(snapshot.shards),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
