from pathlib import Path

from scripts.tools.rl_token import publish_replay


def test_publish_replay_args_keep_explicit_paths():
    args = publish_replay.Args(
        cache_shard=Path("/cache/batch"),
        admission=Path("/admission.json"),
        output=Path("/replay.json"),
    )

    assert args.previous is None
    assert args.output == Path("/replay.json")
