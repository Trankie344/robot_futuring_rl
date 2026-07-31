from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from openpi.training.rl_token.stage2 import cache
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2 import replay
from openpi.training.rl_token.stage2 import td3 as rlt_td3


def _reward_identity(labels_sha256: str) -> dict[str, object]:
    return {
        "stage1_config": "rl_token_stage1",
        "stage2_config": "rl_token_stage2",
        "reward_source": "tristate",
        "reward_label_values": [-1, 0, 1, 2],
        "completion_label": 2,
        "reward_aggregation": "sum_20_frames",
        "reward_schema_version": 1,
        "tristate_labels_sha256": labels_sha256,
    }


def _tables(
    rows: int,
    *,
    marker: float,
    z_dim: int = 256,
    state_dim: int = 16,
    horizon: int = 20,
    action_dim: int = 16,
) -> tuple[cache.FeatureTable, cache.TransitionTable]:
    feature_rows = rows * 2
    feature_row = np.arange(feature_rows, dtype=np.float32)
    z_rl = np.broadcast_to(marker + feature_row[:, None], (feature_rows, z_dim)).astype(ml_dtypes.bfloat16)
    state_norm = np.broadcast_to(
        marker + 100.0 + feature_row[:, None],
        (feature_rows, state_dim),
    ).astype(np.float32)
    vla_reference = np.broadcast_to(
        marker + 200.0 + feature_row[:, None, None],
        (feature_rows, horizon, action_dim),
    ).astype(np.float32)
    current = np.arange(rows, dtype=np.int64)
    next_row = np.arange(rows, feature_rows, dtype=np.int64)
    next_row[-1] = -1
    transition_row = np.arange(rows, dtype=np.float32)
    executed_action = np.broadcast_to(
        marker + 300.0 + transition_row[:, None, None],
        (rows, horizon, action_dim),
    ).astype(np.float32)
    bc_anchor = np.broadcast_to(
        marker + 400.0 + transition_row[:, None, None],
        (rows, horizon, action_dim),
    ).astype(np.float32)
    terminal = np.zeros((rows, 1), dtype=np.bool_)
    terminal[-1, 0] = True
    feature_frame_index = np.concatenate(
        (
            np.arange(rows, dtype=np.int32),
            np.arange(rows, dtype=np.int32) + 20,
        )
    )
    return (
        cache.FeatureTable(
            episode_index=np.zeros(feature_rows, dtype=np.int32),
            frame_index=feature_frame_index,
            z_rl=np.ascontiguousarray(z_rl),
            state_norm=np.ascontiguousarray(state_norm),
            vla_reference=np.ascontiguousarray(vla_reference),
        ),
        cache.TransitionTable(
            episode_index=np.zeros(rows, dtype=np.int32),
            start_frame_index=np.arange(rows, dtype=np.int32),
            current_feature_row=current,
            next_feature_row=next_row,
            executed_action=np.ascontiguousarray(executed_action),
            bc_anchor=np.ascontiguousarray(bc_anchor),
            reward=(marker + 500.0 + transition_row).reshape(rows, 1).astype(np.float32),
            terminal=terminal,
        ),
    )


def _publish(
    tmp_path: Path,
    batch_id: str,
    rows: int,
    *,
    marker: float,
    feature_identity: str = "feature-v1",
    z_dim: int = 256,
    state_dim: int = 16,
) -> cache.OpenShard:
    features, transitions = _tables(
        rows,
        marker=marker,
        z_dim=z_dim,
        state_dim=state_dim,
    )
    root = tmp_path / "cache" / feature_identity / batch_id
    cache.publish_shard(
        root,
        features=features,
        transitions=transitions,
        identity_fields={
            "feature_identity": feature_identity,
            "batch_id": batch_id,
            "migration_manifest_sha256": f"{int(marker) % 10}" * 64,
            "labels_sha256": f"{(int(marker) + 1) % 10}" * 64,
            **_reward_identity(f"{(int(marker) + 1) % 10}" * 64),
        },
    )
    return cache.open_shard(root)


@pytest.fixture
def two_shards(tmp_path: Path) -> Iterator[tuple[cache.OpenShard, cache.OpenShard]]:
    shards = (
        _publish(tmp_path, "batch-2", 2, marker=10.0),
        _publish(tmp_path, "batch-6", 6, marker=20.0),
    )
    try:
        yield shards
    finally:
        for shard in shards:
            shard.close()


@pytest.fixture
def snapshot(two_shards: tuple[cache.OpenShard, cache.OpenShard], tmp_path: Path) -> replay.ReplaySnapshot:
    first, second = two_shards
    first_snapshot = replay.create_snapshot(
        tmp_path / "replay/round_000001.json",
        previous=None,
        new_shard=first,
        admission_sha256="1" * 64,
    )
    return replay.create_snapshot(
        tmp_path / "replay/round_000002.json",
        previous=first_snapshot,
        new_shard=second,
        admission_sha256="2" * 64,
    )


def _read_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(identity.canonical_json_bytes(payload))


@pytest.fixture
def verification_catalog(
    two_shards: tuple[cache.OpenShard, cache.OpenShard],
) -> tuple[cache.ShardVerification, cache.ShardVerification]:
    return tuple(shard.verification for shard in two_shards)


def _write_snapshot_variant(
    tmp_path: Path,
    source: replay.ReplaySnapshot,
    name: str,
    mutation,
) -> Path:
    payload = _read_payload(source.path)
    mutation(payload)
    destination = tmp_path / "snapshot-variants" / name
    destination.parent.mkdir()
    _write_payload(destination, payload)
    return destination


def _single_snapshot_payload(
    verification: cache.ShardVerification,
    *,
    admission_sha256: str,
) -> dict[str, Any]:
    manifest = verification.manifest
    rows = verification.transition_rows
    return {
        "schema_version": 1,
        "feature_identity": manifest["feature_identity"],
        **{key: manifest[key] for key in _reward_identity("0" * 64) if key != "tristate_labels_sha256"},
        "total_transitions": rows,
        "shards": [
            {
                "batch_id": manifest["batch_id"],
                "root": str(verification.root),
                "admission_sha256": admission_sha256,
                "cache_manifest_sha256": verification.manifest_sha256,
                "tristate_labels_sha256": manifest["tristate_labels_sha256"],
                "transition_rows": rows,
                "start": 0,
                "end": rows,
            }
        ],
    }


def test_public_dataclasses_have_exact_frozen_contracts():
    assert [field.name for field in dataclasses.fields(replay.ReplaySnapshot)] == [
        "path",
        "schema_version",
        "feature_identity",
        "stage1_config",
        "stage2_config",
        "reward_source",
        "reward_label_values",
        "completion_label",
        "reward_aggregation",
        "reward_schema_version",
        "total_transitions",
        "shards",
        "sha256",
    ]
    assert [field.name for field in dataclasses.fields(replay.ReplayBatch)] == [
        "z_rl",
        "next_z_rl",
        "state_norm",
        "next_state_norm",
        "vla_reference",
        "next_vla_reference",
        "executed_action",
        "bc_anchor",
        "reward",
        "terminal",
        "source_global_index",
    ]
    assert replay.ReplaySnapshot.__dataclass_params__.frozen
    assert replay.ReplayBatch.__dataclass_params__.frozen


def test_replay_buffer_lru_opens_on_demand_and_preserves_cross_shard_order(
    snapshot: replay.ReplaySnapshot,
):
    with replay.ReplayBuffer.open(snapshot.path, max_open_shards=1) as buffer:
        assert buffer.open_shard_count == 0
        requested = np.array([7, 0, 7, 2, 1], dtype=np.int64)
        batch = buffer.gather(requested)

        np.testing.assert_array_equal(batch.source_global_index, requested)
        np.testing.assert_array_equal(batch.reward[:, 0], np.array([525.0, 510.0, 525.0, 520.0, 511.0]))
        assert buffer.open_shard_count == 1
        assert buffer.lru_misses == 2
        assert buffer.lru_hits == 0
        assert buffer.lru_evictions == 1

        buffer.gather(np.array([7], dtype=np.int64))
        buffer.gather(np.array([6], dtype=np.int64))
        assert buffer.lru_misses == 3
        assert buffer.lru_hits == 1


def test_replay_buffer_open_authenticates_each_shard_once_without_open_snapshot_double_hash(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    original = cache.authenticate_shard
    roots: list[Path] = []

    def record_authentication(root: Path) -> cache.ShardVerification:
        roots.append(Path(root))
        return original(root)

    monkeypatch.setattr(cache, "authenticate_shard", record_authentication)
    with replay.ReplayBuffer.open(snapshot.path, max_open_shards=1):
        pass

    assert roots == [Path(record["root"]) for record in snapshot.shards]


def test_replay_buffer_open_defaults_to_32_leases(snapshot: replay.ReplaySnapshot):
    with replay.ReplayBuffer.open(snapshot.path) as buffer:
        assert buffer.max_open_shards == 32


def test_capability_fast_create_open_gather_and_quiesce_never_rehash_or_reauthenticate(
    two_shards: tuple[cache.OpenShard, cache.OpenShard],
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first_verification, second_verification = verification_catalog

    def forbid_hash(_descriptor: int) -> str:
        raise AssertionError("capability fast paths must not rehash authenticated payloads")

    def forbid_authentication(_root: Path) -> cache.ShardVerification:
        raise AssertionError("capability fast paths must not authenticate a shard again")

    monkeypatch.setattr(cache, "_sha256_fd", forbid_hash)
    monkeypatch.setattr(cache, "authenticate_shard", forbid_authentication)
    first = replay.create_snapshot_from_verifications(
        tmp_path / "fast/round_000001.json",
        previous=None,
        previous_verifications=(),
        new_verification=first_verification,
        admission_sha256="1" * 64,
    )
    second = replay.create_snapshot_from_verifications(
        tmp_path / "fast/round_000002.json",
        previous=first,
        previous_verifications=(first_verification,),
        new_verification=second_verification,
        admission_sha256="2" * 64,
    )

    with replay.ReplayBuffer.open_from_verifications(
        second.path,
        verification_catalog,
        max_open_shards=1,
    ) as buffer:
        batch = buffer.gather(np.array([7, 0, 2], dtype=np.int64))
        np.testing.assert_array_equal(batch.reward[:, 0], np.array([525.0, 510.0, 520.0]))
        buffer.quiesce()
        assert buffer.open_shard_count == 0
        reopened = buffer.gather(np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(reopened.reward[:, 0], np.array([511.0]))

    assert first.total_transitions == 2
    assert second.total_transitions == 8
    assert [record["root"] for record in second.shards] == [str(shard.root) for shard in two_shards]


@pytest.mark.parametrize(
    ("catalog_factory", "error"),
    [
        (lambda first, second: (first,), "count"),
        (lambda first, second: (first, second, second), "count"),
        (lambda first, second: (second, first), "root|batch"),
    ],
)
def test_fast_open_rejects_missing_extra_or_reordered_capabilities(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    catalog_factory,
    error: str,
):
    with pytest.raises(replay.ReplayError, match=error):
        replay.ReplayBuffer.open_from_verifications(
            snapshot.path,
            catalog_factory(*verification_catalog),
        )


def test_fast_paths_require_exact_tuples_of_sealed_capabilities(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
):
    forged = cache.ShardVerification.__new__(cache.ShardVerification)
    with pytest.raises(replay.ReplayError, match="exact tuple"):
        replay.ReplayBuffer.open_from_verifications(snapshot.path, list(verification_catalog))
    with pytest.raises(replay.ReplayError, match="authenticated ShardVerification"):
        replay.ReplayBuffer.open_from_verifications(
            snapshot.path,
            (verification_catalog[0], forged),
        )
    with pytest.raises(replay.ReplayError, match="exact tuple"):
        replay.create_snapshot_from_verifications(
            tmp_path / "bad-list.json",
            previous=snapshot,
            previous_verifications=list(verification_catalog),
            new_verification=verification_catalog[0],
            admission_sha256="3" * 64,
        )
    with pytest.raises(replay.ReplayError, match="authenticated ShardVerification"):
        replay.create_snapshot_from_verifications(
            tmp_path / "bad-new.json",
            previous=None,
            previous_verifications=(),
            new_verification=forged,
            admission_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    ("name", "mutation", "error"),
    [
        (
            "wrong-root.json",
            lambda payload: payload["shards"][0].__setitem__("root", payload["shards"][1]["root"]),
            "root",
        ),
        (
            "wrong-manifest.json",
            lambda payload: payload["shards"][0].__setitem__(
                "cache_manifest_sha256",
                payload["shards"][1]["cache_manifest_sha256"],
            ),
            "manifest",
        ),
        (
            "wrong-batch.json",
            lambda payload: payload["shards"][0].__setitem__("batch_id", "wrong-batch"),
            "batch",
        ),
        (
            "wrong-feature.json",
            lambda payload: payload.__setitem__("feature_identity", "wrong-feature"),
            "feature identity",
        ),
    ],
)
def test_fast_open_rejects_snapshot_catalog_identity_mismatches(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    name: str,
    mutation,
    error: str,
):
    path = _write_snapshot_variant(tmp_path, snapshot, name, mutation)
    with pytest.raises(replay.ReplayError, match=error):
        replay.ReplayBuffer.open_from_verifications(path, verification_catalog)


def test_fast_open_rejects_snapshot_row_mismatch(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
):
    def mutate(payload: dict[str, Any]) -> None:
        payload["shards"][0]["transition_rows"] = 1
        payload["shards"][0]["end"] = 1
        payload["shards"][1]["start"] = 1
        payload["shards"][1]["end"] = 7
        payload["total_transitions"] = 7

    path = _write_snapshot_variant(tmp_path, snapshot, "wrong-rows.json", mutate)
    with pytest.raises(replay.ReplayError, match="row count"):
        replay.ReplayBuffer.open_from_verifications(path, verification_catalog)


def test_fast_paths_reject_non_common_array_contract(
    two_shards: tuple[cache.OpenShard, cache.OpenShard],
    tmp_path: Path,
):
    first = replay.create_snapshot(
        tmp_path / "legacy/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    incompatible = _publish(tmp_path, "incompatible", 2, marker=30.0, state_dim=15)
    combined = replay.create_snapshot(
        tmp_path / "legacy/combined.json",
        previous=first,
        new_shard=incompatible,
        admission_sha256="2" * 64,
    )

    with pytest.raises(replay.ReplayError, match="trailing shape|array contract"):
        replay.ReplayBuffer.open_from_verifications(
            combined.path,
            (two_shards[0].verification, incompatible.verification),
        )
    with pytest.raises(replay.ReplayError, match="trailing shape|array contract"):
        replay.create_snapshot_from_verifications(
            tmp_path / "fast/incompatible.json",
            previous=first,
            previous_verifications=(two_shards[0].verification,),
            new_verification=incompatible.verification,
            admission_sha256="2" * 64,
        )


def test_fast_open_rejects_payload_mutation_after_authentication(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
):
    path = verification_catalog[0].root / "transitions/reward.npy"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    with pytest.raises(replay.ReplayError, match="verification|binding|integrity"):
        replay.ReplayBuffer.open_from_verifications(snapshot.path, verification_catalog)


def test_fast_open_arms_guard_before_quick_verification_and_catches_mutate_restore(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    monkeypatch: pytest.MonkeyPatch,
):
    path = verification_catalog[0].root / "transitions/reward.npy"
    original = path.read_bytes()
    calls = 0

    def mutate_and_restore(
        _verification: cache.ShardVerification,
        *,
        full: bool = False,
    ) -> None:
        nonlocal calls
        assert not full
        calls += 1
        if calls == 1:
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            path.write_bytes(original)

    monkeypatch.setattr(cache, "verify_verification", mutate_and_restore)
    with pytest.raises(replay.ReplayError, match="mutation|integrity"):
        replay.ReplayBuffer.open_from_verifications(snapshot.path, verification_catalog)
    assert calls == 1


def test_fast_open_compares_snapshot_bytes_and_witness_before_and_after_quick_verification(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _read_payload(snapshot.path)
    rewritten = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert rewritten != snapshot.path.read_bytes()
    calls = 0
    original_verify = cache.verify_verification

    def rewrite_snapshot(
        verification: cache.ShardVerification,
        *,
        full: bool = False,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            snapshot.path.write_bytes(rewritten)
        original_verify(verification, full=full)

    monkeypatch.setattr(
        replay._IntegrityGuard,  # noqa: SLF001 - isolate the independent snapshot witness check.
        "check",
        lambda _self: None,
    )
    monkeypatch.setattr(cache, "verify_verification", rewrite_snapshot)
    with pytest.raises(replay.ReplayError, match="snapshot.*changed|binding changed"):
        replay.ReplayBuffer.open_from_verifications(snapshot.path, verification_catalog)


def test_fast_open_acquisition_failure_closes_guard_and_preserves_primary(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    monkeypatch: pytest.MonkeyPatch,
):
    original_open = replay._IntegrityGuard.open  # noqa: SLF001 - white-box acquisition ownership test.
    original_arm_shard = replay._IntegrityGuard.arm_shard  # noqa: SLF001
    original_close = replay._IntegrityGuard.close  # noqa: SLF001
    captured: list[tuple[replay._IntegrityGuard, int]] = []
    primary = RuntimeError("injected capability acquisition failure")

    def capture_guard() -> replay._IntegrityGuard:
        guard = original_open()
        captured.append((guard, guard._descriptor))  # noqa: SLF001 - assert exact FD ownership.
        return guard

    def fail_after_arm(self: replay._IntegrityGuard, root: Path) -> None:
        original_arm_shard(self, root)
        raise primary

    def close_then_fail(
        self: replay._IntegrityGuard,
        close_primary: BaseException | None = None,
    ) -> None:
        original_close(self, close_primary)
        raise OSError("injected post-close cleanup failure")

    monkeypatch.setattr(
        replay._IntegrityGuard,  # noqa: SLF001 - inject at the acquisition ownership seams.
        "open",
        classmethod(lambda _cls: capture_guard()),
    )
    monkeypatch.setattr(replay._IntegrityGuard, "arm_shard", fail_after_arm)  # noqa: SLF001
    monkeypatch.setattr(replay._IntegrityGuard, "close", close_then_fail)  # noqa: SLF001
    with pytest.raises(RuntimeError) as caught:
        replay.ReplayBuffer.open_from_verifications(snapshot.path, verification_catalog)

    assert caught.value is primary
    assert len(captured) == 1
    guard, descriptor = captured[0]
    assert guard.closed
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptor)
    assert any("post-close cleanup failure" in note for note in getattr(primary, "__notes__", ()))


def test_fast_open_snapshot_read_failure_closes_snapshot_fd_and_guard(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    monkeypatch: pytest.MonkeyPatch,
):
    original_open = replay._IntegrityGuard.open  # noqa: SLF001 - white-box acquisition ownership test.
    captured_guards: list[tuple[replay._IntegrityGuard, int]] = []
    captured_snapshot_fds: list[int] = []
    primary = RuntimeError("injected snapshot read failure")

    def capture_guard() -> replay._IntegrityGuard:
        guard = original_open()
        captured_guards.append((guard, guard._descriptor))  # noqa: SLF001 - assert exact FD ownership.
        return guard

    def fail_snapshot_read(_path: Path, descriptor: int) -> None:
        captured_snapshot_fds.append(descriptor)
        raise primary

    monkeypatch.setattr(
        replay._IntegrityGuard,  # noqa: SLF001 - capture the exact guard acquired by the fast path.
        "open",
        classmethod(lambda _cls: capture_guard()),
    )
    monkeypatch.setattr(replay, "_before_snapshot_read", fail_snapshot_read)
    with pytest.raises(RuntimeError) as caught:
        replay.ReplayBuffer.open_from_verifications(snapshot.path, verification_catalog)

    assert caught.value is primary
    assert len(captured_snapshot_fds) == 1
    assert len(captured_guards) == 1
    for descriptor in (captured_snapshot_fds[0], captured_guards[0][1]):
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)
    assert captured_guards[0][0].closed


@pytest.mark.parametrize(
    ("catalog_factory", "error"),
    [
        (lambda first, second: (first,), "count"),
        (lambda first, second: (first, second, second), "count"),
        (lambda first, second: (second, first), "root|batch"),
    ],
)
def test_fast_create_rejects_missing_extra_or_reordered_previous_capabilities_without_destination(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    catalog_factory,
    error: str,
):
    new_shard = _publish(tmp_path, "new-batch", 2, marker=30.0)
    destination = tmp_path / "fast-create-invalid-catalog/output.json"
    with pytest.raises(replay.ReplayError, match=error):
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=catalog_factory(*verification_catalog),
            new_verification=new_shard.verification,
            admission_sha256="3" * 64,
        )
    assert not os.path.lexists(destination)


@pytest.mark.parametrize("admission", ["A" * 64, "1" * 63, "g" * 64, 1, None])
def test_fast_create_rejects_bad_admission_before_creating_destination(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    admission: object,
):
    destination = tmp_path / "bad-admission/output.json"
    with pytest.raises(replay.ReplayError, match="admission_sha256"):
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256=admission,  # type: ignore[arg-type]
        )
    assert not destination.parent.exists()


def test_fast_create_existing_destination_is_untouched_and_explicitly_unverified(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
):
    destination = tmp_path / "existing/output.json"
    destination.parent.mkdir()
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError) as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256="1" * 64,
        )

    assert destination.read_bytes() == b"sentinel"
    assert any("not verified" in note for note in getattr(caught.value, "__notes__", ()))


def test_fast_create_rejects_tampered_previous_snapshot_without_destination(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
):
    payload = _read_payload(snapshot.path)
    payload["feature_identity"] = "tampered-feature"
    _write_payload(snapshot.path, payload)
    new_shard = _publish(tmp_path, "new-batch", 2, marker=30.0)
    destination = tmp_path / "tampered-previous/output.json"

    with pytest.raises(replay.ReplayError, match="previous replay snapshot"):
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=verification_catalog,
            new_verification=new_shard.verification,
            admission_sha256="3" * 64,
        )
    assert not os.path.lexists(destination)


@pytest.mark.parametrize("mismatch", ["duplicate", "feature"])
def test_fast_create_rejects_duplicate_batch_or_feature_change_without_destination(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    mismatch: str,
):
    if mismatch == "duplicate":
        new_verification = verification_catalog[0]
        error = "duplicate batch"
    else:
        new_verification = _publish(
            tmp_path,
            "new-feature-batch",
            2,
            marker=30.0,
            feature_identity="feature-v2",
        ).verification
        error = "feature identity"
    destination = tmp_path / f"{mismatch}/output.json"

    with pytest.raises(replay.ReplayError, match=error):
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=verification_catalog,
            new_verification=new_verification,
            admission_sha256="3" * 64,
        )
    assert not os.path.lexists(destination)


@pytest.mark.parametrize("target", ["old", "new"])
def test_fast_create_rejects_persistent_old_or_new_payload_change_after_authentication(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    target: str,
):
    new_shard = _publish(tmp_path, "new-batch", 2, marker=30.0)
    target_verification = verification_catalog[0] if target == "old" else new_shard.verification
    path = target_verification.root / "transitions/reward.npy"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    destination = tmp_path / f"persistent-{target}/output.json"

    with pytest.raises(replay.ReplayError, match="verification|binding|integrity"):
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=verification_catalog,
            new_verification=new_shard.verification,
            admission_sha256="3" * 64,
        )
    assert not os.path.lexists(destination)


@pytest.mark.parametrize("target", ["old", "new"])
def test_fast_create_arms_all_shards_before_quick_verify_and_catches_mutate_restore(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
):
    new_shard = _publish(tmp_path, "new-batch", 2, marker=30.0)
    target_verification = verification_catalog[0] if target == "old" else new_shard.verification
    path = target_verification.root / "transitions/reward.npy"
    original = path.read_bytes()
    original_verify = cache.verify_verification
    injected = False

    def mutate_restore_during_target_verify(
        verification: cache.ShardVerification,
        *,
        full: bool = False,
    ) -> None:
        nonlocal injected
        assert not full
        if verification is target_verification and not injected:
            injected = True
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            path.write_bytes(original)
            return
        original_verify(verification, full=full)

    monkeypatch.setattr(cache, "verify_verification", mutate_restore_during_target_verify)
    destination = tmp_path / f"mutate-restore-{target}/output.json"
    with pytest.raises(replay.ReplayError, match="mutation|integrity"):
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=verification_catalog,
            new_verification=new_shard.verification,
            admission_sha256="3" * 64,
        )

    assert injected
    assert not os.path.lexists(destination)


def test_fast_create_arms_destination_before_link_and_rejects_real_mutate_restore(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "destination-mutate-restore/output.json"
    original_link = identity.os.link
    mutated = False

    def link_then_mutate_restore(source: Path, target: Path) -> None:
        nonlocal mutated
        original_link(source, target)
        path = Path(target)
        original = path.read_bytes()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        path.write_bytes(original)
        mutated = True

    monkeypatch.setattr(identity.os, "link", link_then_mutate_restore)
    with pytest.raises(replay.ReplayError, match="publication|mutation|CREATE") as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256="1" * 64,
        )

    assert mutated
    assert destination.read_bytes() == identity.canonical_json_bytes(
        _single_snapshot_payload(verification_catalog[0], admission_sha256="1" * 64)
    )
    assert any("may already be published" in note for note in getattr(caught.value, "__notes__", ()))


def test_fast_create_accepts_expected_create_through_alias_of_already_watched_directory(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
):
    cache_root = verification_catalog[0].root.parents[1]
    alias = tmp_path / "cache-alias"
    alias.symlink_to(cache_root, target_is_directory=True)
    destination = alias / "output.json"

    created = replay.create_snapshot_from_verifications(
        destination,
        previous=None,
        previous_verifications=(),
        new_verification=verification_catalog[0],
        admission_sha256="1" * 64,
    )

    assert created.path == destination
    assert destination.read_bytes() == identity.canonical_json_bytes(
        _single_snapshot_payload(verification_catalog[0], admission_sha256="1" * 64)
    )


def test_fast_create_publisher_failure_before_link_leaves_no_destination(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "publisher-before-link/output.json"
    primary = OSError("injected publisher failure before hard link")

    def fail_before_link(_source: Path, _target: Path) -> None:
        raise primary

    monkeypatch.setattr(identity.os, "link", fail_before_link)
    with pytest.raises(replay.ReplayError) as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256="1" * 64,
        )

    assert caught.value.__cause__ is primary
    assert not os.path.lexists(destination)
    assert not any("may already be published" in note for note in getattr(caught.value, "__notes__", ()))


def test_fast_create_publisher_failure_after_link_keeps_complete_destination_for_audit(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "publisher-after-link/output.json"
    original_link = identity.os.link
    primary = OSError("injected publisher failure after hard link")

    def link_then_fail(source: Path, target: Path) -> None:
        original_link(source, target)
        raise primary

    monkeypatch.setattr(identity.os, "link", link_then_fail)
    with pytest.raises(replay.ReplayError) as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256="1" * 64,
        )

    assert caught.value.__cause__ is primary
    assert destination.read_bytes() == identity.canonical_json_bytes(
        _single_snapshot_payload(verification_catalog[0], admission_sha256="1" * 64)
    )
    notes = getattr(caught.value, "__notes__", ())
    assert any("may already be published" in note for note in notes)
    assert any("not verified" in note for note in notes)


def test_fast_create_racing_destination_is_never_overwritten_or_claimed_trustworthy(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "publisher-race/output.json"

    def occupy_before_link(_source: Path, target: Path) -> None:
        Path(target).write_bytes(b"competitor")
        raise FileExistsError(target)

    monkeypatch.setattr(identity.os, "link", occupy_before_link)
    with pytest.raises(FileExistsError) as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256="1" * 64,
        )

    assert destination.read_bytes() == b"competitor"
    notes = getattr(caught.value, "__notes__", ())
    assert any("may already be published" in note for note in notes)
    assert any("not verified" in note for note in notes)
    assert not any("verified destination" in note for note in notes)


@pytest.mark.parametrize("target", ["old", "new"])
def test_fast_create_postpublish_catalog_quick_verify_catches_persistent_change_without_events(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
):
    new_shard = _publish(tmp_path, "new-batch", 2, marker=30.0)
    target_verification = verification_catalog[0] if target == "old" else new_shard.verification
    target_path = target_verification.root / "transitions/reward.npy"
    original_payload = target_path.read_bytes()
    original_publish = identity.atomic_write_json

    def publish_then_persistently_change(path: Path, payload: dict[str, Any]) -> None:
        original_publish(path, payload)
        target_path.write_bytes(original_payload[:-1] + bytes([original_payload[-1] ^ 1]))

    monkeypatch.setattr(identity, "atomic_write_json", publish_then_persistently_change)
    monkeypatch.setattr(replay._IntegrityGuard, "check", lambda _self: None)  # noqa: SLF001
    monkeypatch.setattr(
        replay._IntegrityGuard,  # noqa: SLF001 - emulate a filesystem without local mutation events.
        "check_expected_create",
        lambda _self, _path: None,
        raising=False,
    )
    destination = tmp_path / f"postpublish-{target}/output.json"
    with pytest.raises(replay.ReplayError, match="quick verification|binding") as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=verification_catalog,
            new_verification=new_shard.verification,
            admission_sha256="3" * 64,
        )

    assert destination.read_bytes() == identity.canonical_json_bytes(
        {
            "schema_version": 1,
            "feature_identity": "feature-v1",
            **replay._snapshot_reward_contract(snapshot),  # noqa: SLF001
            "total_transitions": snapshot.total_transitions + new_shard.verification.transition_rows,
            "shards": [
                *(dict(record) for record in snapshot.shards),
                {
                    "batch_id": new_shard.verification.manifest["batch_id"],
                    "root": str(new_shard.verification.root),
                    "admission_sha256": "3" * 64,
                    "cache_manifest_sha256": new_shard.verification.manifest_sha256,
                    "tristate_labels_sha256": new_shard.verification.manifest["tristate_labels_sha256"],
                    "transition_rows": new_shard.verification.transition_rows,
                    "start": snapshot.total_transitions,
                    "end": snapshot.total_transitions + new_shard.verification.transition_rows,
                },
            ],
        }
    )
    assert any("may already be published" in note for note in getattr(caught.value, "__notes__", ()))


def test_fast_create_failure_closes_guard_preserves_primary_and_cleanup_note(
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original_open = replay._IntegrityGuard.open  # noqa: SLF001 - white-box acquisition ownership test.
    original_close = replay._IntegrityGuard.close  # noqa: SLF001
    captured: list[tuple[replay._IntegrityGuard, int]] = []
    primary = RuntimeError("injected fast-create verification failure")

    def capture_guard() -> replay._IntegrityGuard:
        guard = original_open()
        captured.append((guard, guard._descriptor))  # noqa: SLF001 - assert exact FD ownership.
        return guard

    def fail_verification(
        _verification: cache.ShardVerification,
        *,
        full: bool = False,
    ) -> None:
        assert not full
        raise primary

    def close_then_fail(
        self: replay._IntegrityGuard,
        close_primary: BaseException | None = None,
    ) -> None:
        original_close(self, close_primary)
        raise OSError("injected fast-create post-close cleanup failure")

    monkeypatch.setattr(
        replay._IntegrityGuard,  # noqa: SLF001 - capture the exact guard acquired by create.
        "open",
        classmethod(lambda _cls: capture_guard()),
    )
    monkeypatch.setattr(cache, "verify_verification", fail_verification)
    monkeypatch.setattr(replay._IntegrityGuard, "close", close_then_fail)  # noqa: SLF001
    destination = tmp_path / "create-cleanup/output.json"
    with pytest.raises(RuntimeError) as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=None,
            previous_verifications=(),
            new_verification=verification_catalog[0],
            admission_sha256="1" * 64,
        )

    assert caught.value is primary
    assert len(captured) == 1
    guard, descriptor = captured[0]
    assert guard.closed
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptor)
    assert any("post-close cleanup failure" in note for note in getattr(primary, "__notes__", ()))
    assert not os.path.lexists(destination)


def test_fast_create_previous_snapshot_read_failure_closes_snapshot_fd_and_guard(
    snapshot: replay.ReplaySnapshot,
    verification_catalog: tuple[cache.ShardVerification, cache.ShardVerification],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    new_shard = _publish(tmp_path, "new-batch", 2, marker=30.0)
    original_open = replay._IntegrityGuard.open  # noqa: SLF001 - white-box acquisition ownership test.
    captured_guards: list[tuple[replay._IntegrityGuard, int]] = []
    captured_snapshot_fds: list[int] = []
    primary = RuntimeError("injected fast-create previous snapshot read failure")

    def capture_guard() -> replay._IntegrityGuard:
        guard = original_open()
        captured_guards.append((guard, guard._descriptor))  # noqa: SLF001
        return guard

    def fail_snapshot_read(_path: Path, descriptor: int) -> None:
        captured_snapshot_fds.append(descriptor)
        raise primary

    monkeypatch.setattr(
        replay._IntegrityGuard,  # noqa: SLF001 - capture the exact guard acquired by create.
        "open",
        classmethod(lambda _cls: capture_guard()),
    )
    monkeypatch.setattr(replay, "_before_snapshot_read", fail_snapshot_read)
    destination = tmp_path / "create-snapshot-fd/output.json"
    with pytest.raises(RuntimeError) as caught:
        replay.create_snapshot_from_verifications(
            destination,
            previous=snapshot,
            previous_verifications=verification_catalog,
            new_verification=new_shard.verification,
            admission_sha256="3" * 64,
        )

    assert caught.value is primary
    assert len(captured_snapshot_fds) == 1
    assert len(captured_guards) == 1
    for descriptor in (captured_snapshot_fds[0], captured_guards[0][1]):
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)
    assert captured_guards[0][0].closed
    assert not os.path.lexists(destination)


@pytest.mark.parametrize("invalid_cap", [True, np.bool_(), 0, -1, 1.0, "4"])
def test_replay_buffer_open_rejects_non_positive_or_inexact_lease_caps(
    tmp_path: Path,
    invalid_cap: object,
):
    with pytest.raises(ValueError, match="exact positive integer"):
        replay.ReplayBuffer.open(tmp_path / "unused.json", max_open_shards=invalid_cap)  # type: ignore[arg-type]


def test_replay_buffer_quiesce_reopens_but_close_is_terminal(snapshot: replay.ReplaySnapshot):
    buffer = replay.ReplayBuffer.open(snapshot.path, max_open_shards=1)
    buffer.gather(np.array([0], dtype=np.int64))
    assert buffer.open_shard_count == 1

    buffer.quiesce()
    assert buffer.open_shard_count == 0
    np.testing.assert_array_equal(buffer.gather(np.array([0], dtype=np.int64)).source_global_index, [0])
    buffer.close()
    buffer.close()

    with pytest.raises(replay.ReplayError, match="closed"):
        buffer.gather(np.array([0], dtype=np.int64))
    with pytest.raises(replay.ReplayError, match="closed"):
        buffer.sample_indices(np.random.default_rng(0), 1)
    with pytest.raises(replay.ReplayError, match="closed"):
        buffer.verify_integrity()


def test_replay_buffer_detects_tamper_then_restore_and_remains_poisoned(snapshot: replay.ReplaySnapshot):
    buffer = replay.ReplayBuffer.open(snapshot.path, max_open_shards=1)
    root = Path(snapshot.shards[0]["root"])
    path = root / "transitions/reward.npy"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    path.write_bytes(original)

    with pytest.raises(replay.ReplayError, match="integrity|poison"):
        buffer.gather(np.array([0], dtype=np.int64))
    with pytest.raises(replay.ReplayError, match="poison"):
        buffer.gather(np.array([0], dtype=np.int64))
    buffer.close()


def test_empty_gather_uses_catalog_without_opening_a_shard(snapshot: replay.ReplaySnapshot):
    with replay.ReplayBuffer.open(snapshot.path, max_open_shards=1) as buffer:
        batch = buffer.gather(np.array([], dtype=np.int64))

        assert buffer.open_shard_count == 0
        assert batch.z_rl.shape == (0, 256)
        assert batch.state_norm.shape == (0, 16)
        assert batch.vla_reference.shape == (0, 20, 16)
        assert batch.executed_action.shape == (0, 20, 16)
        assert batch.reward.shape == (0, 1)
        assert batch.terminal.shape == (0, 1)


def test_gather_quiesce_and_reopen_never_rehash_authenticated_payloads(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    with replay.ReplayBuffer.open(snapshot.path, max_open_shards=1) as buffer:

        def forbid_hash(_descriptor: int) -> str:
            raise AssertionError("runtime LRU fast path must not rehash payloads")

        monkeypatch.setattr(cache, "_sha256_fd", forbid_hash)
        buffer.gather(np.array([0, 7, 1], dtype=np.int64))
        buffer.quiesce()
        buffer.gather(np.array([7], dtype=np.int64))
        buffer.verify_integrity(full=False)


def test_full_integrity_quiesces_and_rehashes_every_shard(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    with replay.ReplayBuffer.open(snapshot.path, max_open_shards=1) as buffer:
        buffer.gather(np.array([0], dtype=np.int64))
        original = cache._sha256_fd  # noqa: SLF001 - white-box assertion that quick checks do not hash.
        hashes = 0

        def record_hash(descriptor: int) -> str:
            nonlocal hashes
            hashes += 1
            return original(descriptor)

        monkeypatch.setattr(cache, "_sha256_fd", record_hash)
        buffer.verify_integrity(full=False)
        assert hashes == 0
        buffer.verify_integrity(full=True)
        assert hashes == 26
        assert buffer.open_shard_count == 0


def test_replay_close_failure_preserves_body_primary_and_can_retry(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    buffer = replay.ReplayBuffer.open(snapshot.path, max_open_shards=1)
    buffer.gather(np.array([0], dtype=np.int64))
    shard = next(iter(buffer._lru.values()))  # noqa: SLF001 - inject a close failure into the active lease.
    original = cache.OpenShard.close
    fail_once = True

    def injected_close(self: cache.OpenShard) -> None:
        nonlocal fail_once
        if self is shard and fail_once:
            fail_once = False
            raise BufferError("injected replay shard close failure")
        original(self)

    monkeypatch.setattr(cache.OpenShard, "close", injected_close)
    primary = RuntimeError("training body failed")
    with pytest.raises(RuntimeError) as caught, buffer:
        raise primary

    assert caught.value is primary
    assert any("injected replay shard close failure" in note for note in getattr(primary, "__notes__", ()))
    assert not shard.closed
    assert not buffer._guard.closed  # noqa: SLF001 - the lease must survive a retryable shard close failure.

    path = Path(snapshot.shards[0]["root"]) / "transitions/reward.npy"
    original_bytes = path.read_bytes()
    path.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
    path.write_bytes(original_bytes)

    with pytest.raises(replay.ReplayError, match="integrity|mutation"):
        buffer.close()
    assert shard.closed
    assert buffer.closed


def test_replay_guard_preclose_failure_retains_owned_descriptor_for_retry(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    buffer = replay.ReplayBuffer.open(snapshot.path)
    guard = buffer._guard  # noqa: SLF001 - white-box ownership state is the contract under test.
    descriptor = guard._descriptor  # noqa: SLF001 - exact guard FD must remain retryable, never guessed.
    preclose_calls = 0

    def fail_once(candidate: int) -> None:
        nonlocal preclose_calls
        assert candidate == descriptor
        preclose_calls += 1
        if preclose_calls == 1:
            raise OSError("injected inotify pre-close failure")

    monkeypatch.setattr(replay, "_before_integrity_guard_close", fail_once, raising=False)
    with pytest.raises(replay.ReplayError, match="pre-close failure"):
        buffer.close()

    assert not guard.closed
    assert not buffer.closed
    os.fstat(descriptor)
    buffer.close()
    assert guard.closed
    assert buffer.closed
    assert preclose_calls == 2


def test_replay_guard_never_retries_a_reused_descriptor_after_close_was_called(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    buffer = replay.ReplayBuffer.open(snapshot.path)
    guard = buffer._guard  # noqa: SLF001 - white-box terminal ownership is the contract under test.
    descriptor = guard._descriptor  # noqa: SLF001 - force immediate reuse of this exact descriptor number.
    original_close = os.close
    guard_close_calls = 0
    replacement = -1

    def close_reuse_then_fail(candidate: int) -> None:
        nonlocal guard_close_calls, replacement
        if candidate == descriptor:
            guard_close_calls += 1
            if guard_close_calls == 1:
                original_close(candidate)
                replacement = os.open("/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                assert replacement == candidate
                raise OSError("injected post-close failure after descriptor reuse")
        original_close(candidate)

    monkeypatch.setattr(replay.os, "close", close_reuse_then_fail)
    try:
        with pytest.raises(replay.ReplayError, match="post-close failure"):
            buffer.close()

        assert guard.closed
        assert buffer.closed
        os.fstat(replacement)
        buffer.close()
        assert guard_close_calls == 1
        os.fstat(replacement)
    finally:
        monkeypatch.setattr(replay.os, "close", original_close)
        if not guard.closed:
            guard._descriptor = -1  # noqa: SLF001 - prevent a broken implementation from closing the replacement.
            guard._scopes.clear()  # noqa: SLF001 - test-only cleanup after relinquishing the reused number.
            buffer._closed = True  # noqa: SLF001 - the guard was neutralized solely for safe test cleanup.
        if replacement >= 0:
            original_close(replacement)


def test_replay_close_final_drain_preserves_body_and_reports_mutation(
    snapshot: replay.ReplaySnapshot,
    monkeypatch: pytest.MonkeyPatch,
):
    buffer = replay.ReplayBuffer.open(snapshot.path, max_open_shards=1)
    buffer.gather(np.array([0], dtype=np.int64))
    shard = next(iter(buffer._lru.values()))  # noqa: SLF001 - inject mutation during active lease cleanup.
    path = Path(snapshot.shards[0]["root"]) / "transitions/reward.npy"
    original_bytes = path.read_bytes()
    original_close = cache.OpenShard.close

    def mutate_during_close(self: cache.OpenShard) -> None:
        if self is shard:
            path.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
            path.write_bytes(original_bytes)
        original_close(self)

    monkeypatch.setattr(cache.OpenShard, "close", mutate_during_close)
    primary = RuntimeError("training body failed during replay close")
    with pytest.raises(RuntimeError) as caught, buffer:
        raise primary

    assert caught.value is primary
    assert buffer.closed
    assert shard.closed
    assert any("mutation" in note or "integrity" in note for note in getattr(primary, "__notes__", ()))


def _build_many_shard_snapshot(tmp_path: Path, shard_count: int) -> Path:
    roots: list[tuple[Path, dict[str, Any]]] = []
    for index in range(shard_count):
        batch_id = f"batch-{index:03d}"
        features, transitions = _tables(1, marker=float(index))
        root = tmp_path / "cache" / "feature-v1" / batch_id
        manifest = cache.publish_shard(
            root,
            features=features,
            transitions=transitions,
            identity_fields={
                "feature_identity": "feature-v1",
                "batch_id": batch_id,
                "migration_manifest_sha256": f"{index % 10}" * 64,
                "labels_sha256": f"{(index + 1) % 10}" * 64,
                **_reward_identity(f"{(index + 1) % 10}" * 64),
            },
        )
        roots.append((root, manifest))
    records = []
    for index, (root, manifest) in enumerate(roots):
        records.append(
            {
                "batch_id": manifest["batch_id"],
                "root": str(root),
                "admission_sha256": f"{(index + 2) % 10}" * 64,
                "cache_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                "tristate_labels_sha256": manifest["tristate_labels_sha256"],
                "transition_rows": 1,
                "start": index,
                "end": index + 1,
            }
        )
    path = tmp_path / "replay" / "many.json"
    identity.atomic_write_json(
        path,
        {
            "schema_version": 1,
            "feature_identity": "feature-v1",
            **{key: value for key, value in _reward_identity("0" * 64).items() if key != "tristate_labels_sha256"},
            "total_transitions": shard_count,
            "shards": records,
        },
    )
    return path


@pytest.mark.parametrize("file_limit", [96, 128])
def test_lru_works_with_41_shards_under_low_fd_limit(
    tmp_path: Path,
    file_limit: int,
):
    snapshot_path = _build_many_shard_snapshot(tmp_path, 41)
    # The fork must measure this probe, not unreachable mmap cycles left by
    # earlier tests in the same pytest process.
    gc.collect()
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def probe() -> None:
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (file_limit, hard))
            baseline = len(os.listdir("/proc/self/fd"))
            with replay.ReplayBuffer.open(snapshot_path, max_open_shards=4) as buffer:
                requested = np.array([*range(40, -1, -1), 0, 40, 17], dtype=np.int64)
                batch = buffer.gather(requested)
                steady = len(os.listdir("/proc/self/fd"))
                open_shards = buffer.open_shard_count
                misses = buffer.lru_misses
            after = len(os.listdir("/proc/self/fd"))
            send.send(
                {
                    "ok": True,
                    "indices": batch.source_global_index.tolist(),
                    "baseline": baseline,
                    "steady": steady,
                    "after": after,
                    "open": open_shards,
                    "misses": misses,
                }
            )
        except BaseException as exc:
            send.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            send.close()

    process = context.Process(target=probe)
    process.start()
    send.close()
    try:
        process.join(timeout=45)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise AssertionError("low-FD replay probe did not finish")
        assert process.exitcode == 0
        assert receive.poll(timeout=1)
        result = receive.recv()
    finally:
        receive.close()

    assert result["ok"], result
    assert result["indices"] == [*range(40, -1, -1), 0, 40, 17]
    assert result["open"] == 4
    assert result["misses"] == 41
    assert result["steady"] - result["baseline"] <= 13 * 4 + 4
    assert result["after"] <= result["baseline"]


def test_snapshot_append_is_immutable_and_rejects_duplicate_batch(
    two_shards: tuple[cache.OpenShard, cache.OpenShard],
    tmp_path: Path,
):
    first, second = two_shards
    snapshot1 = replay.create_snapshot(
        tmp_path / "replay/round_000001.json",
        previous=None,
        new_shard=first,
        admission_sha256="1" * 64,
    )
    snapshot1_bytes = snapshot1.path.read_bytes()
    snapshot2 = replay.create_snapshot(
        tmp_path / "replay/round_000002.json",
        previous=snapshot1,
        new_shard=second,
        admission_sha256="2" * 64,
    )
    assert snapshot1.path.read_bytes() == snapshot1_bytes
    assert snapshot1.total_transitions == 2
    assert snapshot2.total_transitions == 8
    assert [record["transition_rows"] for record in snapshot2.shards] == [2, 6]
    assert [(record["start"], record["end"]) for record in snapshot2.shards] == [(0, 2), (2, 8)]
    assert all(Path(record["root"]).is_absolute() for record in snapshot2.shards)
    with pytest.raises(replay.ReplayError, match="duplicate batch"):
        replay.create_snapshot(
            tmp_path / "replay/duplicate.json",
            previous=snapshot2,
            new_shard=first,
            admission_sha256="3" * 64,
        )


def test_create_snapshot_never_overwrites_existing_destination(two_shards, tmp_path: Path):
    destination = tmp_path / "replay/existing.json"
    destination.parent.mkdir()
    destination.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replay.create_snapshot(
            destination,
            previous=None,
            new_shard=two_shards[0],
            admission_sha256="1" * 64,
        )
    assert destination.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize("value", ["A" * 64, "1" * 63, "g" * 64, 1, None])
def test_create_snapshot_requires_exact_admission_hash(two_shards, tmp_path: Path, value: Any):
    with pytest.raises(replay.ReplayError, match="admission_sha256"):
        replay.create_snapshot(
            tmp_path / f"replay/bad-{type(value).__name__}-{str(value)[:3]}.json",
            previous=None,
            new_shard=two_shards[0],
            admission_sha256=value,
        )


def test_create_snapshot_reopens_previous_and_rejects_file_tamper(two_shards, tmp_path: Path):
    previous = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    payload = _read_payload(previous.path)
    payload["feature_identity"] = "tampered"
    _write_payload(previous.path, payload)
    with pytest.raises(replay.ReplayError, match="previous replay snapshot"):
        replay.create_snapshot(
            tmp_path / "replay/second.json",
            previous=previous,
            new_shard=two_shards[1],
            admission_sha256="2" * 64,
        )


def test_snapshot_records_reject_direct_mutation(two_shards, tmp_path: Path):
    previous = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    with pytest.raises(TypeError):
        previous.shards[0]["start"] = 999
    with pytest.raises(TypeError):
        previous.shards[0]["end"] = 999
    with pytest.raises(TypeError):
        previous.shards[0]["transition_rows"] = 999


def test_snapshot_constructor_copies_and_freezes_external_records(two_shards, tmp_path: Path):
    previous = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    external_record = dict(previous.shards[0])
    reconstructed = replay.ReplaySnapshot(
        path=previous.path,
        schema_version=previous.schema_version,
        feature_identity=previous.feature_identity,
        stage1_config=previous.stage1_config,
        stage2_config=previous.stage2_config,
        reward_source=previous.reward_source,
        reward_label_values=previous.reward_label_values,
        completion_label=previous.completion_label,
        reward_aggregation=previous.reward_aggregation,
        reward_schema_version=previous.reward_schema_version,
        total_transitions=previous.total_transitions,
        shards=(external_record,),
        sha256=previous.sha256,
    )
    external_record["start"] = 999
    assert reconstructed.shards[0]["start"] == 0
    with pytest.raises(TypeError):
        reconstructed.shards[0]["start"] = 999

    appended = replay.create_snapshot(
        tmp_path / "replay/second.json",
        previous=reconstructed,
        new_shard=two_shards[1],
        admission_sha256="2" * 64,
    )
    assert appended.total_transitions == 8
    assert [(record["start"], record["end"]) for record in appended.shards] == [(0, 2), (2, 8)]


def test_create_snapshot_reopens_new_shard_and_rejects_mutated_manifest(two_shards, tmp_path: Path):
    opened = two_shards[0]
    supplied = dataclasses.replace(opened, manifest={**opened.manifest, "batch_id": "not-the-real-batch"})
    with pytest.raises(replay.ReplayError, match="supplied shard"):
        replay.create_snapshot(
            tmp_path / "replay/first.json",
            previous=None,
            new_shard=supplied,
            admission_sha256="1" * 64,
        )


def test_create_snapshot_rejects_feature_identity_change(two_shards, tmp_path: Path):
    first = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    other = _publish(tmp_path, "other", 2, marker=30.0, feature_identity="feature-v2")
    with pytest.raises(replay.ReplayError, match="feature identity"):
        replay.create_snapshot(
            tmp_path / "replay/second.json",
            previous=first,
            new_shard=other,
            admission_sha256="2" * 64,
        )


def test_snapshot_manifest_hash_comes_from_pinned_verified_file_bytes(
    monkeypatch: pytest.MonkeyPatch,
    two_shards,
    tmp_path: Path,
):
    def forbidden_path_hash(_path: Path) -> str:
        raise AssertionError("replay must not hash a manifest path after cache.open_shard")

    monkeypatch.setattr(identity, "sha256_file", forbidden_path_hash)
    created = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    expected = hashlib.sha256((two_shards[0].root / "manifest.json").read_bytes()).hexdigest()
    assert two_shards[0].manifest_sha256 == expected
    assert created.shards[0]["cache_manifest_sha256"] == expected


def test_open_snapshot_rejects_symlink_and_nonregular_file(snapshot, tmp_path: Path):
    link = tmp_path / "snapshot-link.json"
    link.symlink_to(snapshot.path)
    with pytest.raises(replay.ReplayError, match="symlink"):
        replay.open_snapshot(link)
    with pytest.raises(replay.ReplayError, match="regular file"):
        replay.open_snapshot(tmp_path)


@pytest.mark.parametrize("entry_point", ["open_snapshot", "replay_buffer"])
def test_snapshot_entry_rejects_parent_components_before_lexical_collapse(
    snapshot: replay.ReplaySnapshot,
    tmp_path: Path,
    entry_point: str,
):
    alias_base = tmp_path / "alias-base"
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "subdir").mkdir(parents=True)
    alias_base.mkdir()
    alias_snapshot = alias_base / "snapshot.json"
    alias_snapshot.write_bytes(snapshot.path.read_bytes())
    (alias_base / "link").symlink_to(elsewhere / "subdir", target_is_directory=True)
    ambiguous = alias_base / "link" / ".." / "snapshot.json"
    assert ".." in ambiguous.parts
    assert not (elsewhere / "snapshot.json").exists()

    def open_then_close() -> object:
        if entry_point == "open_snapshot":
            return replay.open_snapshot(ambiguous)
        buffer = replay.ReplayBuffer.open(ambiguous)
        buffer.close()
        return buffer

    with pytest.raises(replay.ReplayError, match="parent component|normalized"):
        open_then_close()


def test_open_snapshot_rejects_fifo_without_blocking(tmp_path: Path):
    fifo = tmp_path / "snapshot.fifo"
    os.mkfifo(fifo)
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def probe() -> None:
        try:
            replay.open_snapshot(fifo)
        except replay.ReplayError as exc:
            send.send(str(exc))
        else:
            send.send("ACCEPTED")
        finally:
            send.close()

    process = context.Process(target=probe)
    process.start()
    send.close()
    try:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            raise AssertionError("open_snapshot blocked on a FIFO")
        assert process.exitcode == 0
        assert receive.poll(timeout=0.1)
        result = receive.recv()
    finally:
        receive.close()

    assert result != "ACCEPTED"
    assert "regular file" in result


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda payload: payload.update(schema_version=True), "schema"),
        (lambda payload: payload.update(schema_version=2), "schema"),
        (lambda payload: payload.update(feature_identity=""), "feature_identity"),
        (lambda payload: payload.update(total_transitions=True), "total_transitions"),
        (lambda payload: payload.update(total_transitions=999), "total transition"),
        (lambda payload: payload.update(shards=[]), "no shards"),
        (lambda payload: payload.update(extra=True), "exactly"),
        (lambda payload: payload["shards"][0].pop("root"), "exactly"),
        (lambda payload: payload["shards"][0].update(extra=True), "exactly"),
        (lambda payload: payload["shards"][0].update(batch_id=""), "batch_id"),
        (
            lambda payload: payload["shards"][1].update(batch_id=payload["shards"][0]["batch_id"]),
            "duplicate batch",
        ),
        (lambda payload: payload["shards"][0].update(root="relative/cache"), "absolute normalized"),
        (lambda payload: payload["shards"][0].update(root="/tmp/cache/../cache"), "absolute normalized"),
        (lambda payload: payload["shards"][0].update(admission_sha256="A" * 64), "admission_sha256"),
        (lambda payload: payload["shards"][0].update(cache_manifest_sha256="0" * 63), "cache_manifest"),
        (lambda payload: payload["shards"][0].update(transition_rows=True), "transition_rows"),
        (lambda payload: payload["shards"][0].update(transition_rows=0), "positive"),
        (lambda payload: payload["shards"][0].update(start=True), "start"),
        (lambda payload: payload["shards"][0].update(end=True), "end"),
        (lambda payload: payload["shards"][0].update(start=1), "noncontiguous"),
        (lambda payload: payload["shards"][0].update(end=99), "noncontiguous"),
    ],
)
def test_open_snapshot_rejects_malformed_contract(snapshot, mutation, error: str):
    payload = _read_payload(snapshot.path)
    mutation(payload)
    _write_payload(snapshot.path, payload)
    with pytest.raises(replay.ReplayError, match=error):
        replay.open_snapshot(snapshot.path)


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b"[]",
        b"\xff",
    ],
)
def test_open_snapshot_rejects_malformed_json_bytes(tmp_path: Path, payload: bytes):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(replay.ReplayError):
        replay.open_snapshot(path)


def test_open_snapshot_rejects_a_real_nul_byte_via_dedicated_branch(tmp_path: Path):
    path = tmp_path / "nul.json"
    path.write_bytes(b'{"schema_version":1}\x00')
    with pytest.raises(replay.ReplayError, match="NUL byte"):
        replay.open_snapshot(path)


def test_open_snapshot_detects_changed_cache_manifest(snapshot):
    root = Path(snapshot.shards[0]["root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["labels_sha256"] = "9" * 64
    (root / "manifest.json").write_bytes(identity.canonical_json_bytes(manifest))
    with pytest.raises(replay.ReplayError, match="cache manifest changed"):
        replay.open_snapshot(snapshot.path)


def test_open_snapshot_rejects_semantically_equal_manifest_byte_rewrite(snapshot):
    root = Path(snapshot.shards[0]["root"])
    path = root / "manifest.json"
    original_bytes = path.read_bytes()
    payload = json.loads(original_bytes)
    reordered = dict(reversed(tuple(payload.items())))
    rewritten = (json.dumps(reordered, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert json.loads(rewritten) == payload
    assert hashlib.sha256(rewritten).digest() != hashlib.sha256(original_bytes).digest()
    path.write_bytes(rewritten)
    assert cache.open_shard(root).manifest == payload

    with pytest.raises(replay.ReplayError, match="cache manifest changed"):
        replay.open_snapshot(snapshot.path)


def test_snapshot_read_and_sha_are_pinned_to_same_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: replay.ReplaySnapshot,
    tmp_path: Path,
):
    original_bytes = snapshot.path.read_bytes()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(original_bytes.replace(b"feature-v1", b"feature-v9"))

    def replace_after_open(path: Path, _descriptor: int) -> None:
        os.replace(replacement, path)

    monkeypatch.setattr(replay, "_before_snapshot_read", replace_after_open)
    reopened = replay.open_snapshot(snapshot.path)
    assert reopened.feature_identity == "feature-v1"
    assert reopened.sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_replay_open_reports_shard_authentication_failure_without_a_second_open_pass(
    monkeypatch: pytest.MonkeyPatch,
    two_shards,
    tmp_path: Path,
):
    snapshot = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=two_shards[0],
        admission_sha256="1" * 64,
    )
    calls = 0

    def fail_authentication(_root: Path) -> cache.ShardVerification:
        nonlocal calls
        calls += 1
        raise cache.CacheError("injected full authentication failure")

    monkeypatch.setattr(cache, "authenticate_shard", fail_authentication)
    with pytest.raises(replay.ReplayError, match="injected full authentication failure"):
        replay.ReplayBuffer.open(snapshot.path)
    assert calls == 1


def test_replay_open_rejects_trailing_shape_mismatch(tmp_path: Path):
    first_shard = _publish(tmp_path, "first", 2, marker=10.0)
    second_shard = _publish(tmp_path, "second", 2, marker=20.0, state_dim=15)
    first = replay.create_snapshot(
        tmp_path / "replay/first.json",
        previous=None,
        new_shard=first_shard,
        admission_sha256="1" * 64,
    )
    second = replay.create_snapshot(
        tmp_path / "replay/second.json",
        previous=first,
        new_shard=second_shard,
        admission_sha256="2" * 64,
    )
    with pytest.raises(replay.ReplayError, match="trailing shape"):
        replay.ReplayBuffer.open(second.path)


def test_replay_buffer_copies_offsets_and_total_before_public_records_can_change(two_shards):
    mutable_record = {
        "start": 0,
        "end": 2,
    }
    fake_snapshot = SimpleNamespace(
        total_transitions=2,
        shards=(mutable_record,),
    )
    buffer = replay.ReplayBuffer(fake_snapshot, (two_shards[0],))
    assert not vars(buffer)["_ends"].flags.writeable

    mutable_record["start"] = 100
    mutable_record["end"] = 102
    fake_snapshot.total_transitions = 999

    assert buffer.total_transitions == 2
    gathered = buffer.gather(np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(
        np.asarray(gathered.z_rl[:, 0], dtype=np.float32),
        np.array([10, 10], dtype=np.float32),
    )
    np.testing.assert_array_equal(gathered.source_global_index, np.array([0, 0], dtype=np.int64))


def test_gather_preserves_cross_shard_order_duplicates_and_exact_values(snapshot):
    buffer = replay.ReplayBuffer.open(snapshot.path)
    indices = np.array([2, 0, 7, 2, 1, 6], dtype=np.int64)
    original = indices.copy()
    batch = buffer.gather(indices)
    np.testing.assert_array_equal(indices, original)
    np.testing.assert_array_equal(batch.source_global_index, indices)
    np.testing.assert_array_equal(batch.reward[:, 0], np.array([520, 510, 525, 520, 511, 524], np.float32))
    np.testing.assert_array_equal(batch.terminal[:, 0], np.array([False, False, True, False, True, False]))
    np.testing.assert_array_equal(
        np.asarray(batch.z_rl[:, 0], dtype=np.float32),
        np.array([20, 10, 25, 20, 11, 24], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(batch.next_z_rl[:, 0], dtype=np.float32),
        np.array([26, 12, 0, 26, 0, 30], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.next_state_norm[:, 0],
        np.array([126, 112, 0, 126, 0, 130], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.next_vla_reference[:, 0, 0],
        np.array([226, 212, 0, 226, 0, 230], dtype=np.float32),
    )
    assert batch.z_rl.dtype == np.dtype(ml_dtypes.bfloat16)
    assert batch.next_z_rl.dtype == np.dtype(ml_dtypes.bfloat16)
    assert batch.state_norm.dtype == np.float32
    assert batch.vla_reference.dtype == np.float32
    assert batch.executed_action.dtype == np.float32
    assert batch.bc_anchor.dtype == np.float32
    assert batch.reward.dtype == np.float32
    assert batch.terminal.dtype == np.bool_
    for field in dataclasses.fields(batch):
        assert getattr(batch, field.name).flags.c_contiguous


def test_empty_gather_has_exact_zero_leading_shapes_and_dtypes(snapshot):
    buffer = replay.ReplayBuffer.open(snapshot.path)
    batch = buffer.gather(np.array([], dtype=np.int64))
    assert batch.z_rl.shape == (0, 256)
    assert batch.next_z_rl.shape == (0, 256)
    assert batch.state_norm.shape == (0, 16)
    assert batch.vla_reference.shape == (0, 20, 16)
    assert batch.executed_action.shape == (0, 20, 16)
    assert batch.reward.shape == (0, 1)
    assert batch.terminal.shape == (0, 1)
    assert batch.source_global_index.shape == (0,)
    for field in dataclasses.fields(batch):
        assert getattr(batch, field.name).flags.c_contiguous


@pytest.mark.parametrize(
    "indices",
    [
        np.array([[0]], dtype=np.int64),
        np.array([0.0], dtype=np.float32),
        np.array([0], dtype=np.uint64),
        np.array([False], dtype=np.bool_),
    ],
)
def test_gather_rejects_wrong_rank_or_non_signed_integer_dtype(snapshot, indices: np.ndarray):
    with pytest.raises(ValueError, match="rank 1 signed integer"):
        replay.ReplayBuffer.open(snapshot.path).gather(indices)


@pytest.mark.parametrize("indices", [np.array([-1], np.int64), np.array([8], np.int64)])
def test_gather_rejects_out_of_range_indices(snapshot, indices: np.ndarray):
    with pytest.raises(IndexError, match="out of range"):
        replay.ReplayBuffer.open(snapshot.path).gather(indices)


def test_sample_indices_is_deterministic_in_range_and_uniform_by_transition(snapshot):
    buffer = replay.ReplayBuffer.open(snapshot.path)
    first = buffer.sample_indices(np.random.default_rng(7), 100_000)
    second = buffer.sample_indices(np.random.default_rng(7), 100_000)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert int(first.min()) >= 0
    assert int(first.max()) < 8
    counts = np.bincount(first, minlength=8)
    assert counts.max() / counts.min() < 1.1
    assert counts[:2].sum() / counts.sum() == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("batch_size", [0, -1, 1.0, True, np.asarray([1], dtype=np.bool_)[0], "2"])
def test_sample_indices_requires_exact_positive_integer(snapshot, batch_size: Any):
    with pytest.raises(ValueError, match="positive integer"):
        replay.ReplayBuffer.open(snapshot.path).sample_indices(np.random.default_rng(0), batch_size)


def test_sample_indices_accepts_numpy_signed_integer(snapshot):
    result = replay.ReplayBuffer.open(snapshot.path).sample_indices(np.random.default_rng(0), np.int64(3))
    assert result.shape == (3,)


def test_as_jax_transition_batch_maps_exact_core_fields_and_dtypes(snapshot):
    host = replay.ReplayBuffer.open(snapshot.path).gather(np.array([0, 1, 7], dtype=np.int64))
    device = replay.as_jax_transition_batch(host)
    assert isinstance(device, rlt_td3.RLTTransitionBatch)
    assert [field.name for field in dataclasses.fields(device)] == [
        "z_rl",
        "next_z_rl",
        "state_norm",
        "next_state_norm",
        "vla_reference",
        "next_vla_reference",
        "executed_action",
        "bc_anchor",
        "reward",
        "terminal",
    ]
    assert device.z_rl.dtype == jnp.bfloat16
    assert device.next_z_rl.dtype == jnp.bfloat16
    for field in (
        "state_norm",
        "next_state_norm",
        "vla_reference",
        "next_vla_reference",
        "executed_action",
        "bc_anchor",
        "reward",
        "terminal",
    ):
        assert getattr(device, field).dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(device.terminal), host.terminal.astype(np.float32))
