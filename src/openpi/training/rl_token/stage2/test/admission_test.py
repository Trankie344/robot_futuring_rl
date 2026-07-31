from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.rl_token.stage2 import admission
from openpi.training.rl_token.stage2 import identity
from openpi.training.rl_token.stage2.test.conftest import VIDEO_KEYS
from openpi.training.rl_token.stage2.test.conftest import build_ready_batch


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _refresh_manifest_file(root: Path, target_path: str) -> None:
    manifest_path = root / "migration_manifest.json"
    manifest = _read_json(manifest_path)
    assert isinstance(manifest, dict)
    records = [
        record for record in manifest["files"] if isinstance(record, dict) and record.get("target_path") == target_path
    ]
    assert len(records) == 1
    path = root / target_path
    records[0]["size"] = path.stat().st_size
    records[0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)


def _rewrite_parquet(
    root: Path,
    episode_index: int,
    column_name: str,
    column: pa.Array,
) -> None:
    relative = f"data/chunk-000/episode_{episode_index:06d}.parquet"
    path = root / relative
    table = pq.read_table(path)
    index = table.column_names.index(column_name)
    pq.write_table(table.set_column(index, column_name, column), path)
    _refresh_manifest_file(root, relative)


def _rewrite_parquet_table(
    root: Path,
    episode_index: int,
    table: pa.Table,
    *,
    row_group_size: int | None = None,
) -> Path:
    relative = f"data/chunk-000/episode_{episode_index:06d}.parquet"
    path = root / relative
    pq.write_table(table, path, row_group_size=row_group_size)
    _refresh_manifest_file(root, relative)
    return path


def _label_payload(root: Path) -> dict[str, object]:
    payload = _read_json(root / "meta/tristate_labels.json")
    assert isinstance(payload, dict)
    return payload


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _expert_payload(root: Path) -> dict[str, object]:
    payload = _read_json(root / "meta/expert_frame_index.json")
    assert isinstance(payload, dict)
    return payload


def _write_expert_and_refresh(root: Path, payload: object) -> None:
    _write_json(root / "meta/expert_frame_index.json", payload)
    _refresh_manifest_file(root, "meta/expert_frame_index.json")


ROUND_ID = "round_000001"
ADMITTED_AT = "2026-07-24T04:30:00+08:00"
CODE_COMMIT = "a" * 40


class _StringSubclass(str):
    pass


class _EscapingRoundId(str):
    def __format__(self, _format_spec: str) -> str:
        return "../../escaped_admission"


def _publish_test_admission(
    ready_batch: Path,
    training_root: Path,
) -> tuple[admission.ValidatedBatch, Path]:
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    path = admission.publish_admission(
        batch,
        training_root,
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )
    return batch, path


def _replace_admission_payload(path: Path, payload: object) -> None:
    path.unlink()
    identity.atomic_write_json(path, payload)


def test_validate_ready_batch_returns_exact_twenty_complete_episodes(ready_batch: Path):
    result = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert result.batch_id == ready_batch.name
    assert result.root == ready_batch.resolve()
    assert result.fps == 30
    assert len(result.episodes) == 20
    assert result.total_frames == 440
    assert result.chunk_equivalents == 40
    assert len(result.episode_fingerprints) == 20
    first_parquet = ready_batch / "data/chunk-000/episode_000000.parquet"
    assert result.episodes[0].parquet_size == first_parquet.stat().st_size
    assert result.episodes[0].parquet_sha256 == hashlib.sha256(first_parquet.read_bytes()).hexdigest()
    assert len(result.manifest_sha256) == 64
    assert len(result.labels_sha256) == 64
    assert result.episodes[0].labels[-1] == 2
    assert result.episodes[0].task == "pick and place"
    assert result.episodes[0].parquet_path == (ready_batch / "data/chunk-000/episode_000000.parquet")
    assert not result.episodes[0].labels.flags.writeable
    assert not result.episodes[0].intervention.flags.writeable


def test_validated_episode_arrays_are_irreversibly_read_only(ready_batch: Path):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    episode = batch.episodes[0]
    labels_before = episode.labels.copy()
    intervention_before = episode.intervention.copy()

    assert not episode.labels.flags.owndata
    assert not episode.intervention.flags.owndata
    with pytest.raises(ValueError, match="WRITEABLE"):
        episode.labels.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        episode.intervention.setflags(write=True)

    np.testing.assert_array_equal(episode.labels, labels_before)
    np.testing.assert_array_equal(episode.intervention, intervention_before)


def test_hil_default_closed_segments_equal_parquet_intervention(ready_batch: Path):
    result = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    np.testing.assert_array_equal(
        np.flatnonzero(result.episodes[0].intervention),
        np.array([3, 4]),
    )


def test_expert_masks_treat_end_frame_index_as_inclusive(tmp_path: Path):
    path = tmp_path / "expert_frame_index.json"
    _write_json(
        path,
        {
            "episodes": [
                {
                    "episode_index": 0,
                    "segments": [{"start_frame_index": 1, "end_frame_index": 3}],
                }
            ]
        },
    )

    masks = admission._expert_masks(path, (5,) * 20)  # noqa: SLF001

    assert len(masks) == 20
    assert all(mask.dtype == np.bool_ for mask in masks)
    np.testing.assert_array_equal(np.flatnonzero(masks[0]), np.array([1, 2, 3]))
    assert not masks[1].any()


def test_hil_mismatch_rejects_entire_batch_at_first_frame(ready_batch: Path):
    payload = _expert_payload(ready_batch)
    payload["episodes"][0]["segments"][0]["end_frame_index"] = 5
    _write_expert_and_refresh(ready_batch, payload)

    with pytest.raises(
        admission.AdmissionError,
        match=rf"batch {ready_batch.name}.*episode 0.*frame 5.*expert.*intervention",
    ):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_missing_expert_episode_means_no_intervention(ready_batch: Path):
    result = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert not result.episodes[1].intervention.any()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"episodes": {}}, r"episodes.*list"),
        ({"episodes": [7]}, r"record 0.*object"),
        (
            {"episodes": [{"episode_index": True, "segments": []}]},
            r"episode_index.*nonnegative integer",
        ),
        (
            {"episodes": [{"episode_index": 0.0, "segments": []}]},
            r"episode_index.*nonnegative integer",
        ),
        (
            {"episodes": [{"episode_index": float("nan"), "segments": []}]},
            r"episode_index.*nonnegative integer",
        ),
        (
            {"episodes": [{"segments": []}]},
            r"episode_index.*nonnegative integer",
        ),
        (
            {"episodes": [{"episode_index": -1, "segments": []}]},
            r"episode_index.*nonnegative integer",
        ),
        (
            {"episodes": [{"episode_index": 20, "segments": []}]},
            r"episode 20.*range",
        ),
        (
            {
                "episodes": [
                    {"episode_index": 0, "segments": []},
                    {"episode_index": 0, "segments": []},
                ]
            },
            r"duplicate.*episode 0",
        ),
        (
            {"episodes": [{"episode_index": 0, "segments": {}}]},
            r"episode 0.*segments.*list",
        ),
        (
            {"episodes": [{"episode_index": 0, "segments": [7]}]},
            r"episode 0.*segment 0.*object",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": True, "end_frame_index": 1}],
                    }
                ]
            },
            r"start_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 0.0, "end_frame_index": 1}],
                    }
                ]
            },
            r"start_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": float("nan"), "end_frame_index": 1}],
                    }
                ]
            },
            r"start_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"end_frame_index": 1}],
                    }
                ]
            },
            r"start_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 0, "end_frame_index": True}],
                    }
                ]
            },
            r"end_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 0, "end_frame_index": 1.0}],
                    }
                ]
            },
            r"end_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 0, "end_frame_index": float("nan")}],
                    }
                ]
            },
            r"end_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 0}],
                    }
                ]
            },
            r"end_frame_index.*nonnegative integer",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 4, "end_frame_index": 3}],
                    }
                ]
            },
            r"segment.*episode 0.*\[4,3\]",
        ),
        (
            {
                "episodes": [
                    {
                        "episode_index": 0,
                        "segments": [{"start_frame_index": 0, "end_frame_index": 5}],
                    }
                ]
            },
            r"segment.*episode 0.*\[0,5\]",
        ),
    ],
)
def test_expert_masks_reject_malformed_records(
    tmp_path: Path,
    payload: object,
    error: str,
):
    path = tmp_path / "expert_frame_index.json"
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=error):
        admission._expert_masks(path, (5,) * 20)  # noqa: SLF001


def test_expert_masks_union_overlapping_segments(tmp_path: Path):
    path = tmp_path / "expert_frame_index.json"
    _write_json(
        path,
        {
            "episodes": [
                {
                    "episode_index": 0,
                    "segments": [
                        {"start_frame_index": 1, "end_frame_index": 3},
                        {"start_frame_index": 3, "end_frame_index": 4},
                        {"start_frame_index": 2, "end_frame_index": 2},
                    ],
                }
            ]
        },
    )

    masks = admission._expert_masks(path, (6,) * 20)  # noqa: SLF001

    np.testing.assert_array_equal(np.flatnonzero(masks[0]), np.array([1, 2, 3, 4]))


def test_validate_ready_batch_wraps_malformed_expert_json_after_manifest_refresh(
    ready_batch: Path,
):
    path = ready_batch / "meta/expert_frame_index.json"
    path.write_text("{bad", encoding="utf-8")
    _refresh_manifest_file(ready_batch, "meta/expert_frame_index.json")

    with pytest.raises(admission.AdmissionError, match=r"invalid JSON.*expert_frame_index.json"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_does_not_modify_input_tree(ready_batch: Path):
    before = _tree_bytes(ready_batch)

    admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert _tree_bytes(ready_batch) == before


def test_validate_ready_batch_rejects_missing_ready(ready_batch: Path):
    (ready_batch / "READY").unlink()

    with pytest.raises(admission.AdmissionError, match="READY"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_nineteen_metadata_episodes(ready_batch: Path):
    path = ready_batch / "meta/episodes.jsonl"
    _write_jsonl(path, _read_jsonl(path)[:-1])
    _refresh_manifest_file(ready_batch, "meta/episodes.jsonl")

    with pytest.raises(admission.AdmissionError, match=r"episodes.*20|indices"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_duplicate_metadata_episode(ready_batch: Path):
    path = ready_batch / "meta/episodes.jsonl"
    records = _read_jsonl(path)
    records[-1] = records[0]
    _write_jsonl(path, records)
    _refresh_manifest_file(ready_batch, "meta/episodes.jsonl")

    with pytest.raises(admission.AdmissionError, match=r"duplicate.*episode 0"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_duplicate_label_episode(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"][0]["episodes"][-1] = payload["datasets"][0]["episodes"][0]
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"duplicate.*label.*episode 0"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_label_length_mismatch(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"][0]["episodes"][3]["frame_states"].pop()
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"episode 3.*length"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_pending_null_labels(tmp_path: Path):
    root = build_ready_batch(tmp_path / "batch_pending", completed_labels=False)

    with pytest.raises(admission.AdmissionError, match=r"batch_pending.*episode 0.*frame 0.*null"):
        admission.validate_ready_batch(root, video_validator=lambda *_: None)


@pytest.mark.parametrize("bad_label", ["done", 3, True, 0.0])
def test_validate_ready_batch_rejects_legacy_unknown_or_bool_frame_state(
    ready_batch: Path,
    bad_label: object,
):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"][0]["episodes"][0]["frame_states"][5] = bad_label
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"episode 0.*frame 5"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_accepts_negative_progress_label(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"][0]["episodes"][0]["frame_states"][5] = -1
    _write_json(path, payload)

    result = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert result.episodes[0].labels[5] == -1


def test_validate_ready_batch_rejects_two_terminal_labels(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"][0]["episodes"][0]["frame_states"][5] = 2
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"episode 0.*more than one.*2"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_nonfinal_terminal_label(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    states = payload["datasets"][0]["episodes"][1]["frame_states"]
    states[5] = 2
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"episode 1.*frame 5.*final"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_ignores_legacy_label_metadata(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["states"] = [-100, "done", None]
    payload["datasets"][0]["episodes"][0]["segments"] = [{"invalid": True}]
    payload["datasets"][0]["episodes"][0]["done_frames"] = ["anything"]
    _write_json(path, payload)

    result = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert result.episodes[0].labels[-1] == 2


def test_validate_ready_batch_rejects_multiple_label_datasets(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"].append(payload["datasets"][0])
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"datasets.*exactly one"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_wrong_label_dataset_name(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    payload = _label_payload(ready_batch)
    payload["datasets"][0]["dataset_name"] = "another_batch"
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"dataset_name"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_non_30_hz(ready_batch: Path):
    path = ready_batch / "meta/info.json"
    payload = _read_json(path)
    payload["fps"] = 20
    _write_json(path, payload)
    _refresh_manifest_file(ready_batch, "meta/info.json")

    with pytest.raises(admission.AdmissionError, match=r"fps.*30"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_modified_core_file(ready_batch: Path):
    path = ready_batch / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)

    with pytest.raises(admission.AdmissionError, match=r"episode_000000.parquet.*(size|sha256)"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_hashes_manifest_file_not_parsed_by_task3(ready_batch: Path):
    path = ready_batch / "meta/tasks.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(admission.AdmissionError, match=r"tasks.jsonl.*(size|sha256)"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_checks_hash_before_parsing_core_metadata(ready_batch: Path):
    path = ready_batch / "meta/info.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(admission.AdmissionError, match=r"info.json.*(size|sha256)"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_unsafe_manifest_parent_path(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["files"].append({"target_path": "../escape", "size": 0, "sha256": "0" * 64})
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"unsafe.*\.\."):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_nul_manifest_target_path(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["files"].append({"target_path": "meta/\x00evil", "size": 0, "sha256": "0" * 64})
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"(target_path|NUL|null)"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/absolute/file",
        "./meta/info.json",
        "meta//info.json",
        "meta\\info.json",
        "meta/info.json/",
        ".",
        "",
    ],
)
def test_validate_ready_batch_rejects_noncanonical_manifest_paths(
    ready_batch: Path,
    bad_path: str,
):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["files"][0]["target_path"] = bad_path
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"(unsafe|normalized|target_path)"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_duplicate_manifest_file(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["files"].append(dict(payload["files"][0]))
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"duplicate.*manifest.*file"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_missing_consumed_manifest_file(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    missing = "videos/chunk-000/observation.images.top/episode_000019.mp4"
    payload["files"] = [record for record in payload["files"] if record["target_path"] != missing]
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"manifest.*missing.*episode_000019.mp4"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize(
    "missing",
    [
        "meta/tasks.jsonl",
        "meta/episodes_stats.jsonl",
    ],
)
def test_validate_ready_batch_requires_all_consumed_metadata_records(
    ready_batch: Path,
    missing: str,
):
    manifest_path = ready_batch / "migration_manifest.json"
    payload = _read_json(manifest_path)
    payload["files"] = [record for record in payload["files"] if record["target_path"] != missing]
    _write_json(manifest_path, payload)

    with pytest.raises(admission.AdmissionError, match=rf"manifest.*missing.*{Path(missing).name}"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize(
    "records",
    [
        [],
        [
            {"task_index": 0, "task": "pick and place"},
            {"task_index": 1, "task": "pick and place"},
        ],
        [{"task_index": 1, "task": "pick and place"}],
        [{"task_index": 0, "task": ""}],
        [{"task_index": 0, "task": "different task"}],
    ],
)
def test_validate_ready_batch_requires_single_matching_task_record(
    ready_batch: Path,
    records: list[dict[str, object]],
):
    path = ready_batch / "meta/tasks.jsonl"
    _write_jsonl(path, records)
    _refresh_manifest_file(ready_batch, "meta/tasks.jsonl")

    with pytest.raises(admission.AdmissionError, match=r"tasks\.jsonl|task.*match"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_duplicate_manifest_target_index(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["episodes"][1]["target_index"] = 0
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"target.*indices|duplicate.*target"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_duplicate_source_identity(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    for field in ("source_host", "source_dataset_root", "source_index"):
        payload["episodes"][1][field] = payload["episodes"][0][field]
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"duplicate.*source.*identity"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_host", "zme@\x00robot"),
        ("source_dataset_root", "/home/zme/\x00dataset"),
    ],
)
def test_validate_ready_batch_rejects_nul_source_identity(
    ready_batch: Path,
    field: str,
    value: str,
):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["episodes"][0][field] = value
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=field):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_wraps_nul_batch_root_as_admission_error(tmp_path: Path):
    path = Path(str(tmp_path / "batch") + "\x00bad")

    with pytest.raises(admission.AdmissionError, match=r"batch root"):
        admission.validate_ready_batch(path, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_non_lowercase_fingerprint(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["episode_fingerprints"][0] = "A" * 64
    payload["episodes"][0]["fingerprint"] = "A" * 64
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"fingerprint"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_metadata_fingerprint_mismatch(ready_batch: Path):
    path = ready_batch / "meta/episodes.jsonl"
    records = _read_jsonl(path)
    records[0]["source_fingerprint"] = "f" * 64
    _write_jsonl(path, records)
    _refresh_manifest_file(ready_batch, "meta/episodes.jsonl")

    with pytest.raises(admission.AdmissionError, match=r"episode 0.*fingerprint"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_noncontiguous_dataset_range(ready_batch: Path):
    path = ready_batch / "meta/episodes.jsonl"
    records = _read_jsonl(path)
    records[1]["dataset_from_index"] = 23
    _write_jsonl(path, records)
    _refresh_manifest_file(ready_batch, "meta/episodes.jsonl")

    with pytest.raises(admission.AdmissionError, match=r"episode 1.*dataset.*range"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_cross_metadata_frame_total(ready_batch: Path):
    path = ready_batch / "meta/info.json"
    payload = _read_json(path)
    payload["total_frames"] += 1
    _write_json(path, payload)
    _refresh_manifest_file(ready_batch, "meta/info.json")

    with pytest.raises(admission.AdmissionError, match=r"frame totals disagree"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize(
    ("column_name", "values", "error"),
    [
        ("frame_index", [1] * 22, "frame_index"),
        ("episode_index", [4] * 22, "episode_index"),
        ("index", list(range(1, 23)), r"\bindex\b"),
        ("task_index", [1] * 22, "task_index"),
    ],
)
def test_validate_ready_batch_rejects_wrong_parquet_indices(
    ready_batch: Path,
    column_name: str,
    values: list[int],
    error: str,
):
    _rewrite_parquet(ready_batch, 0, column_name, pa.array(values, type=pa.int64()))

    with pytest.raises(admission.AdmissionError, match=error):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_checks_all_parquet_row_groups(ready_batch: Path):
    path = ready_batch / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    values = np.arange(22, dtype=np.int64)
    values[17] = 999
    column_index = table.column_names.index("frame_index")
    table = table.set_column(
        column_index,
        "frame_index",
        pa.array(values, type=pa.int64()),
    )
    path = _rewrite_parquet_table(ready_batch, 0, table, row_group_size=11)
    assert pq.ParquetFile(path).metadata.num_row_groups == 2

    with pytest.raises(admission.AdmissionError, match=r"frame_index"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize("column_name", ["observation.state", "action"])
def test_validate_ready_batch_rejects_state_or_action_shape(
    ready_batch: Path,
    column_name: str,
):
    values = pa.array([[0.0] * 15 for _ in range(22)], type=pa.list_(pa.float32(), 15))
    _rewrite_parquet(ready_batch, 0, column_name, values)

    with pytest.raises(admission.AdmissionError, match=rf"{column_name}.*16"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


@pytest.mark.parametrize("column_name", ["observation.state", "action"])
def test_validate_ready_batch_rejects_state_or_action_dtype(
    ready_batch: Path,
    column_name: str,
):
    values = pa.array([[0.0] * 16 for _ in range(22)], type=pa.list_(pa.float64(), 16))
    _rewrite_parquet(ready_batch, 0, column_name, values)

    with pytest.raises(admission.AdmissionError, match=rf"{column_name}.*float32"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_nonfinite_action(ready_batch: Path):
    values = np.zeros((22, 16), dtype=np.float32)
    values[4, 7] = np.inf
    action = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 16)
    _rewrite_parquet(ready_batch, 0, "action", action)

    with pytest.raises(admission.AdmissionError, match=r"action.*finite"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_nonfinite_state(ready_batch: Path):
    values = np.zeros((22, 16), dtype=np.float32)
    values[15, 3] = np.nan
    state = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 16)
    _rewrite_parquet(ready_batch, 0, "observation.state", state)

    with pytest.raises(admission.AdmissionError, match=r"observation.state.*finite"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_top_level_null_action_from_public_entry(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original = pq.read_table
    call_index = 0

    def injecting_read_table(source: object, *args, **kwargs):
        nonlocal call_index
        assert isinstance(source, pa.BufferReader)
        table = original(source, *args, **kwargs)
        current_index = call_index
        call_index += 1
        if current_index != 0:
            return table
        action = table["action"].combine_chunks()
        mask = pa.array([False] * 4 + [True] + [False] * 17, type=pa.bool_())
        with_null = pa.FixedSizeListArray.from_arrays(
            action.values,
            16,
            mask=mask,
        )
        column_index = table.column_names.index("action")
        return table.set_column(column_index, "action", with_null)

    monkeypatch.setattr(admission.pq, "read_table", injecting_read_table)

    with pytest.raises(admission.AdmissionError, match=r"action.*null"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_null_action(ready_batch: Path):
    values: list[float | None] = [0.0] * (22 * 16)
    values[4 * 16 + 7] = None
    _rewrite_parquet(
        ready_batch,
        0,
        "action",
        pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float32()), 16),
    )

    with pytest.raises(admission.AdmissionError, match=r"action.*null"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_non_boolean_intervention(ready_batch: Path):
    _rewrite_parquet(
        ready_batch,
        0,
        "intervention",
        pa.array([0] * 22, type=pa.int64()),
    )

    with pytest.raises(admission.AdmissionError, match=r"intervention.*bool"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_null_intervention(ready_batch: Path):
    values: list[bool | None] = [False] * 22
    values[12] = None
    _rewrite_parquet(
        ready_batch,
        0,
        "intervention",
        pa.array(values, type=pa.bool_()),
    )

    with pytest.raises(admission.AdmissionError, match=r"intervention.*null"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_null_control_mode(ready_batch: Path):
    values: list[int | None] = [1] * 22
    values[3] = None
    _rewrite_parquet(ready_batch, 0, "control_mode", pa.array(values, type=pa.int64()))

    with pytest.raises(admission.AdmissionError, match=r"control_mode.*null"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_non_int64_control_mode(ready_batch: Path):
    _rewrite_parquet(
        ready_batch,
        0,
        "control_mode",
        pa.array([1] * 22, type=pa.int32()),
    )

    with pytest.raises(admission.AdmissionError, match=r"control_mode.*int64"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_wrong_control_mode_declaration(ready_batch: Path):
    path = ready_batch / "meta/info.json"
    payload = _read_json(path)
    payload["features"]["control_mode"]["dtype"] = "int32"
    _write_json(path, payload)
    _refresh_manifest_file(ready_batch, "meta/info.json")

    with pytest.raises(admission.AdmissionError, match=r"control_mode.*dtype.*int64"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_bool_manifest_target_index(ready_batch: Path):
    path = ready_batch / "migration_manifest.json"
    payload = _read_json(path)
    payload["episodes"][0]["target_index"] = True
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"target_index.*integer"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_reads_only_required_parquet_columns(
    ready_batch: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original = pq.read_table
    calls: list[tuple[str, ...]] = []

    def recording_read_table(*args, **kwargs):
        calls.append(tuple(kwargs["columns"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(admission.pq, "read_table", recording_read_table)

    admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert calls == [admission.REQUIRED_COLUMNS] * 20


def test_validate_ready_batch_rejects_symlink_root(ready_batch: Path, tmp_path: Path):
    link = tmp_path / "linked_batch"
    link.symlink_to(ready_batch, target_is_directory=True)

    with pytest.raises(admission.AdmissionError, match=r"batch root.*symlink"):
        admission.validate_ready_batch(link, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_symlink_ready(ready_batch: Path):
    path = ready_batch / "READY"
    path.unlink()
    target = ready_batch / "REAL_READY"
    target.write_text("", encoding="utf-8")
    path.symlink_to(target)

    with pytest.raises(admission.AdmissionError, match=r"READY.*symlink"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_symlink_core_file(ready_batch: Path):
    path = ready_batch / "data/chunk-000/episode_000000.parquet"
    target = path.with_suffix(".real")
    path.rename(target)
    path.symlink_to(target.name)

    with pytest.raises(admission.AdmissionError, match=r"episode_000000.parquet.*symlink"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_symlink_core_ancestor(ready_batch: Path):
    path = ready_batch / "data/chunk-000"
    target = ready_batch / "data/real-chunk"
    path.rename(target)
    path.symlink_to(target.name, target_is_directory=True)

    with pytest.raises(admission.AdmissionError, match=r"chunk-000.*symlink"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_rejects_symlink_labels(ready_batch: Path):
    path = ready_batch / "meta/tristate_labels.json"
    target = path.with_suffix(".real")
    path.rename(target)
    path.symlink_to(target.name)

    with pytest.raises(admission.AdmissionError, match=r"tristate_labels.json.*symlink"):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_calls_video_validator_sixty_times_in_order(ready_batch: Path):
    calls: list[tuple[Path, int, float]] = []

    admission.validate_ready_batch(
        ready_batch,
        video_validator=lambda *args: calls.append(args),
    )

    expected = [
        (
            ready_batch / "videos/chunk-000" / key / f"episode_{episode_index:06d}.mp4",
            22,
            0.05,
        )
        for episode_index in range(20)
        for key in VIDEO_KEYS
    ]
    assert calls == expected


def test_validate_ready_batch_propagates_video_validator_error(ready_batch: Path):
    def fail(*_args):
        raise ValueError("decoder unavailable")

    with pytest.raises(ValueError, match="decoder unavailable"):
        admission.validate_ready_batch(ready_batch, video_validator=fail)


def test_validate_ready_batch_rejects_labels_changed_during_video_validation(
    ready_batch: Path,
):
    mutated = False

    def mutate_once(*_args):
        nonlocal mutated
        if mutated:
            return
        payload = _label_payload(ready_batch)
        payload["datasets"][0]["episodes"][0]["frame_states"][6] = 1
        _write_json(ready_batch / "meta/tristate_labels.json", payload)
        mutated = True

    with pytest.raises(admission.AdmissionError, match=r"labels.*changed.*validation"):
        admission.validate_ready_batch(ready_batch, video_validator=mutate_once)


def test_validate_ready_batch_rejects_manifest_changed_during_video_validation(
    ready_batch: Path,
):
    mutated = False

    def mutate_once(*_args):
        nonlocal mutated
        if mutated:
            return
        path = ready_batch / "migration_manifest.json"
        payload = _read_json(path)
        assert isinstance(payload, dict)
        payload["review_mutation"] = True
        _write_json(path, payload)
        mutated = True

    with pytest.raises(admission.AdmissionError, match=r"manifest.*changed.*validation"):
        admission.validate_ready_batch(ready_batch, video_validator=mutate_once)


def test_validate_ready_batch_rejects_core_file_changed_during_video_validation(
    ready_batch: Path,
):
    mutated = False
    path = ready_batch / "data/chunk-000/episode_000000.parquet"

    def mutate_once(*_args):
        nonlocal mutated
        if mutated:
            return
        with path.open("ab") as stream:
            stream.write(b"\n")
        mutated = True

    with pytest.raises(admission.AdmissionError, match=r"episode_000000.parquet.*(size|sha256)"):
        admission.validate_ready_batch(ready_batch, video_validator=mutate_once)


def test_validate_ready_batch_wraps_malformed_manifest_json(ready_batch: Path):
    (ready_batch / "migration_manifest.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(
        admission.AdmissionError,
        match=rf"{ready_batch.name}.*migration_manifest.json|migration_manifest.json",
    ):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_wraps_malformed_labels_json(ready_batch: Path):
    (ready_batch / "meta/tristate_labels.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(
        admission.AdmissionError,
        match=rf"{ready_batch.name}.*tristate_labels.json|tristate_labels.json",
    ):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_validate_ready_batch_wraps_malformed_episodes_jsonl(ready_batch: Path):
    path = ready_batch / "meta/episodes.jsonl"
    path.write_text("{bad\n", encoding="utf-8")
    _refresh_manifest_file(ready_batch, "meta/episodes.jsonl")

    with pytest.raises(
        admission.AdmissionError,
        match=rf"{ready_batch.name}.*episodes.jsonl|episodes.jsonl",
    ):
        admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)


def test_admission_payload_is_exact(ready_batch: Path):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    payload = admission.admission_payload(
        batch,
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )

    assert payload == {
        "schema_version": 1,
        "round_id": ROUND_ID,
        "admitted_at": ADMITTED_AT,
        "code_commit": CODE_COMMIT,
        "batch_id": ready_batch.name,
        "batch_root": str(ready_batch.resolve()),
        "manifest_sha256": batch.manifest_sha256,
        "labels_sha256": batch.labels_sha256,
        "episode_fingerprints": list(batch.episode_fingerprints),
        "episode_lengths": [22] * 20,
        "chunk_equivalents": 40,
        "validation_report": {
            "episode_count": 20,
            "total_frames": 440,
            "video_count": 60,
            "fps": 30,
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("round_id", 1, r"round_id.*round_NNNNNN"),
        ("round_id", "round_00001", r"round_id.*round_NNNNNN"),
        ("round_id", "ROUND_000001", r"round_id.*round_NNNNNN"),
        ("round_id", _StringSubclass(ROUND_ID), r"round_id.*round_NNNNNN"),
        ("admitted_at", 1, r"admitted_at.*ISO-8601"),
        ("admitted_at", "not-a-time", r"admitted_at.*ISO-8601"),
        ("admitted_at", "2026-07-24T04:30:00", r"admitted_at.*timezone"),
        ("admitted_at", _StringSubclass(ADMITTED_AT), r"admitted_at.*ISO-8601"),
        ("code_commit", 1, r"code_commit.*lowercase Git SHA-1"),
        ("code_commit", "a" * 39, r"code_commit.*lowercase Git SHA-1"),
        ("code_commit", "A" * 40, r"code_commit.*lowercase Git SHA-1"),
        ("code_commit", _StringSubclass(CODE_COMMIT), r"code_commit.*lowercase Git SHA-1"),
    ],
)
def test_admission_payload_rejects_invalid_publication_identity(
    ready_batch: Path,
    field: str,
    value: object,
    error: str,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    kwargs: dict[str, object] = {
        "round_id": ROUND_ID,
        "admitted_at": ADMITTED_AT,
        "code_commit": CODE_COMMIT,
    }
    kwargs[field] = value

    with pytest.raises(admission.AdmissionError, match=error):
        admission.admission_payload(batch, **kwargs)


def test_publish_admission_rejects_round_id_str_subclass_path_escape(
    ready_batch: Path,
    tmp_path: Path,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    with pytest.raises(admission.AdmissionError, match=r"round_id.*round_NNNNNN"):
        admission.publish_admission(
            batch,
            tmp_path / "training",
            round_id=_EscapingRoundId(ROUND_ID),
            admitted_at=ADMITTED_AT,
            code_commit=CODE_COMMIT,
        )

    assert not (tmp_path / "escaped_admission.json").exists()


def test_publish_admission_is_no_overwrite_and_preserves_first_bytes(
    ready_batch: Path,
    tmp_path: Path,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    training_root = tmp_path / "training"
    path = admission.publish_admission(
        batch,
        training_root,
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )
    first_bytes = path.read_bytes()

    with pytest.raises(FileExistsError):
        admission.publish_admission(
            batch,
            training_root,
            round_id=ROUND_ID,
            admitted_at=ADMITTED_AT,
            code_commit="b" * 40,
        )

    assert path == training_root / "admissions" / f"{ROUND_ID}.json"
    assert path.read_bytes() == first_bytes


def test_publish_admission_does_not_modify_validated_batch_tree(
    ready_batch: Path,
    tmp_path: Path,
):
    before = _tree_bytes(ready_batch)
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    admission.publish_admission(
        batch,
        tmp_path / "training",
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )

    assert _tree_bytes(ready_batch) == before
    assert all(not episode.labels.flags.writeable for episode in batch.episodes)
    assert all(not episode.intervention.flags.writeable for episode in batch.episodes)


def test_open_admission_recomputes_chunk_equivalents_from_one_pinned_read(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    expected_bytes = path.read_bytes()

    def forbidden_path_hash(_path: Path) -> str:
        raise AssertionError("open_admission must hash the bytes read from its pinned descriptor")

    monkeypatch.setattr(admission.identity, "sha256_file", forbidden_path_hash)
    opened = admission.open_admission(path)

    assert opened.path == path
    assert opened.round_id == ROUND_ID
    assert opened.admitted_at == ADMITTED_AT
    assert opened.code_commit == CODE_COMMIT
    assert opened.batch_id == batch.batch_id
    assert opened.manifest_sha256 == batch.manifest_sha256
    assert opened.labels_sha256 == batch.labels_sha256
    assert opened.episode_fingerprints == batch.episode_fingerprints
    assert opened.episode_lengths == tuple(episode.length for episode in batch.episodes)
    assert opened.chunk_equivalents == sum((length + 19) // 20 for length in opened.episode_lengths)
    assert opened.sha256 == hashlib.sha256(expected_bytes).hexdigest()


def test_open_admission_uses_exact_integer_chunk_equivalents(
    ready_batch: Path,
    tmp_path: Path,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    assert isinstance(payload, dict)
    lengths = payload["episode_lengths"]
    assert isinstance(lengths, list)
    lengths[0] = 18_014_398_509_482_001
    exact_chunks = sum((length + 19) // 20 for length in lengths)
    payload["chunk_equivalents"] = exact_chunks
    report = payload["validation_report"]
    assert isinstance(report, dict)
    report["total_frames"] = sum(lengths)
    _replace_admission_payload(path, payload)

    assert admission.open_admission(path).chunk_equivalents == exact_chunks

    payload["chunk_equivalents"] = exact_chunks - 1
    _replace_admission_payload(path, payload)
    with pytest.raises(admission.AdmissionError, match=r"chunk_equivalents"):
        admission.open_admission(path)


def test_open_admission_wraps_extreme_episode_length_as_admission_error(
    ready_batch: Path,
    tmp_path: Path,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    assert isinstance(payload, dict)
    lengths = payload["episode_lengths"]
    assert isinstance(lengths, list)
    lengths[0] = 10**4000
    payload["chunk_equivalents"] = 1
    report = payload["validation_report"]
    assert isinstance(report, dict)
    report["total_frames"] = sum(lengths)
    _replace_admission_payload(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"chunk_equivalents"):
        admission.open_admission(path)


def test_open_admission_does_not_reopen_ready_dataset(
    ready_batch: Path,
    tmp_path: Path,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    moved_batch = ready_batch.with_name(f"{ready_batch.name}-moved")
    ready_batch.rename(moved_batch)

    opened = admission.open_admission(path)

    assert opened.batch_id == ready_batch.name


def test_open_admission_does_not_resolve_path_before_nofollow_open(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")

    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("open_admission must not resolve a path before nofollow validation")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)

    assert admission.open_admission(path).path == path


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", 2, r"schema"),
        ("schema_version", True, r"schema"),
        ("round_id", "round_000000", r"round"),
        ("round_id", "round_00001", r"round"),
        ("round_id", 1, r"round"),
        ("admitted_at", "2026-07-24T04:30:00", r"admitted_at.*timezone"),
        ("admitted_at", 1, r"admitted_at"),
        ("code_commit", "A" * 40, r"code_commit"),
        ("code_commit", 1, r"code_commit"),
        ("manifest_sha256", "f" * 63, r"manifest_sha256"),
        ("manifest_sha256", 7, r"manifest_sha256"),
        ("labels_sha256", "F" * 64, r"labels_sha256"),
        ("labels_sha256", 7, r"labels_sha256"),
        ("episode_fingerprints", "not-a-list", r"fingerprint"),
        ("episode_lengths", "not-a-list", r"episode lengths"),
        ("chunk_equivalents", 40.0, r"chunk_equivalents"),
        ("chunk_equivalents", True, r"chunk_equivalents"),
        ("validation_report", [], r"validation_report"),
    ],
)
def test_open_admission_rejects_wrong_scalar_or_container_types(
    ready_batch: Path,
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    assert isinstance(payload, dict)
    payload[field] = value
    _replace_admission_payload(path, payload)

    with pytest.raises(admission.AdmissionError, match=error):
        admission.open_admission(path)


@pytest.mark.parametrize(
    ("batch_id", "batch_root", "error"),
    [
        ("", None, r"batch_id"),
        (" bad", None, r"batch_id"),
        ("bad/name", None, r"batch_id"),
        (7, None, r"batch_id"),
        (None, "relative/batch", r"batch_root"),
        (None, "/tmp/../tmp/batch", r"batch_root"),
        (None, 7, r"batch_root"),
        ("other-batch", None, r"batch_root.*batch_id|batch_id.*batch_root"),
    ],
)
def test_open_admission_rejects_invalid_batch_identity(
    ready_batch: Path,
    tmp_path: Path,
    batch_id: object | None,
    batch_root: object | None,
    error: str,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    assert isinstance(payload, dict)
    if batch_id is not None:
        payload["batch_id"] = batch_id
    if batch_root is not None:
        payload["batch_root"] = batch_root
    _replace_admission_payload(path, payload)

    with pytest.raises(admission.AdmissionError, match=error):
        admission.open_admission(path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", r"fields"),
        ("extra", r"fields"),
        ("nineteen_lengths", r"episode lengths"),
        ("zero_length", r"episode lengths"),
        ("bool_length", r"episode lengths"),
        ("float_length", r"episode lengths"),
        ("nineteen_fingerprints", r"fingerprint"),
        ("duplicate_fingerprint", r"fingerprint"),
        ("malformed_fingerprint", r"fingerprint"),
        ("integer_fingerprint", r"fingerprint"),
        ("forged_chunks", r"chunk_equivalents"),
    ],
)
def test_open_admission_rejects_forged_shape_or_identity(
    ready_batch: Path,
    tmp_path: Path,
    mutation: str,
    error: str,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    assert isinstance(payload, dict)
    if mutation == "missing":
        del payload["validation_report"]
    elif mutation == "extra":
        payload["unexpected"] = True
    elif mutation == "nineteen_lengths":
        payload["episode_lengths"] = payload["episode_lengths"][:-1]
    elif mutation == "zero_length":
        payload["episode_lengths"][0] = 0
    elif mutation == "bool_length":
        payload["episode_lengths"][0] = True
    elif mutation == "float_length":
        payload["episode_lengths"][0] = 22.0
    elif mutation == "nineteen_fingerprints":
        payload["episode_fingerprints"] = payload["episode_fingerprints"][:-1]
    elif mutation == "duplicate_fingerprint":
        payload["episode_fingerprints"][1] = payload["episode_fingerprints"][0]
    elif mutation == "malformed_fingerprint":
        payload["episode_fingerprints"][0] = "g" * 64
    elif mutation == "integer_fingerprint":
        payload["episode_fingerprints"][0] = 7
    else:
        payload["chunk_equivalents"] += 1
    _replace_admission_payload(path, payload)

    with pytest.raises(admission.AdmissionError, match=error):
        admission.open_admission(path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", r"validation_report"),
        ("extra", r"validation_report"),
        ("episode_count_bool", r"validation_report"),
        ("total_frames_float", r"validation_report"),
        ("video_count_wrong", r"validation_report"),
        ("fps_wrong", r"validation_report"),
    ],
)
def test_open_admission_requires_exact_validation_report(
    ready_batch: Path,
    tmp_path: Path,
    mutation: str,
    error: str,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    assert isinstance(payload, dict)
    report = payload["validation_report"]
    if mutation == "missing":
        del report["fps"]
    elif mutation == "extra":
        report["unexpected"] = 1
    elif mutation == "episode_count_bool":
        report["episode_count"] = True
    elif mutation == "total_frames_float":
        report["total_frames"] = float(report["total_frames"])
    elif mutation == "video_count_wrong":
        report["video_count"] = 59
    else:
        report["fps"] = 20
    _replace_admission_payload(path, payload)

    with pytest.raises(admission.AdmissionError, match=error):
        admission.open_admission(path)


def test_open_admission_rejects_noncanonical_json_bytes(
    ready_batch: Path,
    tmp_path: Path,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    payload = _read_json(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(admission.AdmissionError, match=r"canonical"):
        admission.open_admission(path)


@pytest.mark.parametrize("link_level", ["training", "admissions", "file"])
def test_open_admission_rejects_symlink_in_any_path_component(
    ready_batch: Path,
    tmp_path: Path,
    link_level: str,
):
    training_root = tmp_path / "training"
    _batch, path = _publish_test_admission(ready_batch, training_root)
    if link_level == "file":
        target = path.with_name("real-admission.json")
        path.rename(target)
        path.symlink_to(target.name)
    elif link_level == "admissions":
        linked = training_root / "admissions"
        target = training_root / "real-admissions"
        linked.rename(target)
        linked.symlink_to(target.name, target_is_directory=True)
    else:
        target = tmp_path / "real-training"
        training_root.rename(target)
        training_root.symlink_to(target.name, target_is_directory=True)

    with pytest.raises(admission.AdmissionError, match=r"symlink|directory"):
        admission.open_admission(path)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_open_admission_rejects_missing_or_nonregular_path(
    ready_batch: Path,
    tmp_path: Path,
    kind: str,
):
    _batch, published = _publish_test_admission(ready_batch, tmp_path / "training")
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    assert path != published

    with pytest.raises(admission.AdmissionError, match=r"regular file|missing"):
        admission.open_admission(path)


def test_open_admission_rejects_non_directory_ancestor(tmp_path: Path):
    ancestor = tmp_path / "not-a-directory"
    ancestor.write_text("regular file", encoding="utf-8")

    with pytest.raises(admission.AdmissionError, match=r"ancestor.*not a directory|directory"):
        admission.open_admission(ancestor / "admission.json")


def test_open_admission_rejects_fifo_without_blocking(tmp_path: Path):
    fifo = tmp_path / "admission.fifo"
    os.mkfifo(fifo)
    script = """
import sys
from pathlib import Path
from openpi.training.rl_token.stage2 import admission

try:
    admission.open_admission(Path(sys.argv[1]))
except admission.AdmissionError as exc:
    print(exc)
else:
    raise SystemExit("open_admission accepted FIFO")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(fifo)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "regular file" in completed.stdout


def test_open_admission_rejects_same_size_in_place_rewrite_during_read(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")

    def rewrite_same_inode(opened_path: Path, _descriptor: int) -> None:
        payload = opened_path.read_bytes()
        marker = b'"code_commit":"' + (b"a" * 40) + b'"'
        replacement = b'"code_commit":"' + (b"b" * 40) + b'"'
        assert marker in payload
        opened_path.write_bytes(payload.replace(marker, replacement))

    monkeypatch.setattr(
        admission,
        "_after_admission_file_open",
        rewrite_same_inode,
        raising=False,
    )

    with pytest.raises(admission.AdmissionError, match=r"changed while being read"):
        admission.open_admission(path)


def test_open_admission_rejects_path_replacement_after_descriptor_pin(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")

    def replace_path(opened_path: Path, _descriptor: int) -> None:
        payload = _read_json(opened_path)
        payload["code_commit"] = "b" * 40
        opened_path.rename(opened_path.with_suffix(".original"))
        identity.atomic_write_json(opened_path, payload)

    monkeypatch.setattr(
        admission,
        "_after_admission_file_open",
        replace_path,
        raising=False,
    )

    with pytest.raises(admission.AdmissionError, match=r"pathname changed|changed while being read"):
        admission.open_admission(path)


def test_open_admission_rejects_a_b_a_replacement_during_namespace_witness(
    ready_batch: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _batch, path = _publish_test_admission(ready_batch, tmp_path / "training")
    initial_ctime_ns = path.stat().st_ctime_ns
    real_open = admission._open_nofollow_regular  # noqa: SLF001
    open_count = 0

    def swap_before_namespace_witness(opened_path: Path) -> tuple[int, os.stat_result]:
        nonlocal open_count
        open_count += 1
        if open_count == 2:
            original_path = opened_path.with_suffix(".original")
            transient_path = opened_path.with_suffix(".transient")
            payload = _read_json(opened_path)
            assert isinstance(payload, dict)
            payload["code_commit"] = "b" * 40
            opened_path.rename(original_path)
            identity.atomic_write_json(opened_path, payload)
            opened_path.rename(transient_path)
            original_path.rename(opened_path)
            assert opened_path.stat().st_ctime_ns != initial_ctime_ns
        return real_open(opened_path)

    monkeypatch.setattr(admission, "_open_nofollow_regular", swap_before_namespace_witness)

    with pytest.raises(admission.AdmissionError, match=r"pathname changed"):
        admission.open_admission(path)


def test_admission_payload_rejects_round_zero(
    ready_batch: Path,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    with pytest.raises(admission.AdmissionError, match=r"round_id.*positive|round_NNNNNN"):
        admission.admission_payload(
            batch,
            round_id="round_000000",
            admitted_at=ADMITTED_AT,
            code_commit=CODE_COMMIT,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered"])
def test_verify_admission_rejects_any_payload_difference(
    ready_batch: Path,
    tmp_path: Path,
    mutation: str,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    path = admission.publish_admission(
        batch,
        tmp_path / "training",
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )
    admission.verify_admission(path, batch)
    payload = _read_json(path)
    assert isinstance(payload, dict)
    if mutation == "missing":
        del payload["chunk_equivalents"]
    elif mutation == "extra":
        payload["unexpected"] = True
    else:
        payload["validation_report"]["fps"] = 31
    _write_json(path, payload)

    with pytest.raises(admission.AdmissionError, match=r"admission.*immutable batch"):
        admission.verify_admission(path, batch)


def test_verify_admission_wraps_malformed_json(
    ready_batch: Path,
    tmp_path: Path,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    path = tmp_path / "malformed.json"
    path.write_text("{bad", encoding="utf-8")

    with pytest.raises(admission.AdmissionError, match=r"invalid JSON.*malformed.json"):
        admission.verify_admission(path, batch)


def test_verify_admission_rejects_final_symlink(
    ready_batch: Path,
    tmp_path: Path,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    target = admission.publish_admission(
        batch,
        tmp_path / "training",
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )
    link = tmp_path / "admission-link.json"
    link.symlink_to(target)

    with pytest.raises(admission.AdmissionError, match=r"admission.*regular file|symlink"):
        admission.verify_admission(link, batch)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_verify_admission_rejects_missing_or_nonregular_path(
    ready_batch: Path,
    tmp_path: Path,
    kind: str,
):
    batch = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()

    with pytest.raises(admission.AdmissionError, match=r"admission.*regular file"):
        admission.verify_admission(path, batch)


def test_verified_file_reader_rejects_fifo_without_blocking(tmp_path: Path):
    fifo = tmp_path / "parquet.fifo"
    os.mkfifo(fifo)
    script = """
import hashlib
import sys
from pathlib import Path
from openpi.training.rl_token.stage2 import admission

try:
    admission._read_verified_regular_file(
        Path(sys.argv[1]),
        expected_size=0,
        expected_sha256=hashlib.sha256(b"").hexdigest(),
    )
except admission.AdmissionError as exc:
    print(exc)
else:
    raise SystemExit("verified file reader accepted FIFO")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(fifo)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "regular file" in completed.stdout


def test_verify_admission_rejects_fifo_without_blocking(tmp_path: Path):
    fifo = tmp_path / "admission.fifo"
    os.mkfifo(fifo)
    script = """
import sys
from pathlib import Path
from openpi.training.rl_token.stage2 import admission

try:
    admission.verify_admission(Path(sys.argv[1]), None)
except admission.AdmissionError as exc:
    print(exc)
else:
    raise SystemExit("verify_admission accepted FIFO")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(fifo)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "regular file" in completed.stdout


def test_legal_relabel_revalidates_but_invalidates_old_admission(
    ready_batch: Path,
    tmp_path: Path,
):
    original = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)
    path = admission.publish_admission(
        original,
        tmp_path / "training",
        round_id=ROUND_ID,
        admitted_at=ADMITTED_AT,
        code_commit=CODE_COMMIT,
    )
    labels_path = ready_batch / "meta/tristate_labels.json"
    labels = _label_payload(ready_batch)
    labels["datasets"][0]["episodes"][0]["frame_states"][6] = 1
    _write_json(labels_path, labels)

    relabeled = admission.validate_ready_batch(ready_batch, video_validator=lambda *_: None)

    assert relabeled.labels_sha256 != original.labels_sha256
    with pytest.raises(admission.AdmissionError, match=r"admission.*immutable batch"):
        admission.verify_admission(path, relabeled)


def test_validate_video_with_ffprobe_uses_safe_exact_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "video with spaces.mp4"
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="22\n", stderr="")

    monkeypatch.setattr(admission.subprocess, "run", fake_run)

    admission.validate_video_with_ffprobe(path, 22, 0.05)

    assert captured["argv"] == [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    assert captured["kwargs"] == {
        "capture_output": True,
        "text": True,
        "timeout": 30,
        "check": False,
    }


def test_validate_video_with_ffprobe_rejects_process_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stdout="", stderr="bad"),
    )

    with pytest.raises(admission.AdmissionError, match=r"ffprobe failed"):
        admission.validate_video_with_ffprobe(tmp_path / "bad.mp4", 22, 0.05)


def test_validate_video_with_ffprobe_wraps_missing_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "video.mp4"

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(admission.subprocess, "run", missing)

    with pytest.raises(admission.AdmissionError, match=rf"ffprobe failed.*{path.name}"):
        admission.validate_video_with_ffprobe(path, 22, 0.05)


@pytest.mark.parametrize("stdout", ["", "N/A\n", "1.5\n"])
def test_validate_video_with_ffprobe_rejects_missing_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
):
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=stdout, stderr=""),
    )

    with pytest.raises(admission.AdmissionError, match=r"no frame count"):
        admission.validate_video_with_ffprobe(tmp_path / "bad.mp4", 22, 0.05)


def test_validate_video_with_ffprobe_rejects_frame_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="21\n", stderr=""),
    )

    with pytest.raises(admission.AdmissionError, match=r"expected 22.*got 21"):
        admission.validate_video_with_ffprobe(tmp_path / "bad.mp4", 22, 0.05)


def test_validate_video_with_ffprobe_wraps_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(admission.subprocess, "run", timeout)

    with pytest.raises(admission.AdmissionError, match=r"ffprobe failed"):
        admission.validate_video_with_ffprobe(tmp_path / "slow.mp4", 22, 0.05)
