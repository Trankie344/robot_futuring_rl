"""Portable manifest validation and a transactional SQLite migration index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from types import TracebackType


class ManifestError(ValueError):
    """A published migration manifest is malformed or conflicts with the ledger."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BATCH_ID_RE = re.compile(r"batch_(\d{6})(?:_[A-Za-z0-9][A-Za-z0-9_-]*)?")
# Safely above the expected size of a portable 20-episode migration manifest.
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ManifestError(f"nonstandard JSON numeric constant: {value}")


def _read_manifest_document(manifest_path: Path, batch_id: str) -> object:
    try:
        with manifest_path.open("rb") as stream:
            payload = stream.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ManifestError(
            f"published batch {batch_id} could not read migration_manifest.json"
        ) from exc
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ManifestError(
            f"published batch {batch_id} manifest exceeds the 16 MiB size limit"
        )

    try:
        document = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(
            f"published batch {batch_id} has malformed UTF-8 migration_manifest.json"
        ) from exc
    try:
        return json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except ManifestError as exc:
        raise ManifestError(
            f"published batch {batch_id} has invalid JSON manifest: {exc}"
        ) from exc
    except RecursionError as exc:
        raise ManifestError(
            f"published batch {batch_id} JSON manifest is too deeply nested"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"published batch {batch_id} has malformed migration_manifest.json"
        ) from exc


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ManifestError(f"{path} must be an array")
    return value


def _string(value: object, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "a nonempty string" if nonempty else "a string"
        raise ManifestError(f"{path} must be {qualifier}")
    return value


def _integer(value: object, path: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "a positive integer" if positive else "a nonnegative integer"
        raise ManifestError(f"{path} must be {qualifier}")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ManifestError(f"{path} must be a lowercase 64-hex digest")
    return digest


def _batch_id(value: object) -> str:
    batch_id = _string(value, "batch_id")
    if _BATCH_ID_RE.fullmatch(batch_id) is None:
        raise ManifestError(
            "batch_id must be safe and start with batch_ followed by a six-digit sequence"
        )
    return batch_id


def _created_at(value: object) -> str:
    created_at = _string(value, "created_at")
    if created_at.strip() != created_at or "T" not in created_at:
        raise ManifestError("created_at must be a nonempty ISO-style timestamp")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError("created_at must be a nonempty ISO-style timestamp") from exc
    return created_at


def _absolute_posix(value: object, path: str) -> str:
    raw = _string(value, path)
    parsed = PurePosixPath(raw)
    if (
        "\x00" in raw
        or "\\" in raw
        or not parsed.is_absolute()
        or ".." in parsed.parts
        or parsed.as_posix() != raw
    ):
        raise ManifestError(f"{path} must be a normalized absolute POSIX path")
    return raw


def _relative_posix(value: object, path: str) -> str:
    raw = _string(value, path)
    parsed = PurePosixPath(raw)
    if (
        "\x00" in raw
        or "\\" in raw
        or parsed.is_absolute()
        or ".." in parsed.parts
        or parsed.as_posix() != raw
        or raw == "."
    ):
        raise ManifestError(f"{path} must be a normalized nonempty relative POSIX path")
    return raw


def _validate_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    root = _mapping(manifest, "manifest")
    if type(root.get("schema_version")) is not int or root.get("schema_version") != 1:
        raise ManifestError("schema_version must equal 1")

    batch_id = _batch_id(root.get("batch_id"))
    created_at = _created_at(root.get("created_at"))
    episode_count = _integer(root.get("episode_count"), "episode_count", positive=True)
    frame_count = _integer(root.get("frame_count"), "frame_count")

    fingerprint_values = _sequence(
        root.get("episode_fingerprints"), "episode_fingerprints"
    )
    fingerprints = [
        _sha256(value, f"episode_fingerprints[{index}]")
        for index, value in enumerate(fingerprint_values)
    ]
    if len(fingerprints) != episode_count:
        raise ManifestError("episode_fingerprints length must equal episode_count")
    if len(fingerprints) != len(set(fingerprints)):
        raise ManifestError("episode_fingerprints contains a duplicate fingerprint")

    episode_values = _sequence(root.get("episodes"), "episodes")
    if len(episode_values) != episode_count:
        raise ManifestError("episodes length must equal episode_count")
    episodes: list[dict[str, object]] = []
    for position, value in enumerate(episode_values):
        record = _mapping(value, f"episodes[{position}]")
        episodes.append(
            {
                "target_index": _integer(
                    record.get("target_index"), f"episodes[{position}].target_index"
                ),
                "fingerprint": _sha256(
                    record.get("fingerprint"), f"episodes[{position}].fingerprint"
                ),
                "source_host": _string(
                    record.get("source_host"), f"episodes[{position}].source_host"
                ),
                "source_dataset_root": _absolute_posix(
                    record.get("source_dataset_root"),
                    f"episodes[{position}].source_dataset_root",
                ),
                "source_index": _integer(
                    record.get("source_index"), f"episodes[{position}].source_index"
                ),
            }
        )

    episodes.sort(key=lambda record: int(record["target_index"]))
    target_indices = [int(record["target_index"]) for record in episodes]
    if target_indices != list(range(episode_count)):
        raise ManifestError("episode target_index values must be contiguous from 0")
    ordered_fingerprints = [str(record["fingerprint"]) for record in episodes]
    if ordered_fingerprints != fingerprints:
        raise ManifestError(
            "episode_fingerprints must exactly match episodes ordered by target_index"
        )
    source_identities = [
        (
            record["source_host"],
            record["source_dataset_root"],
            record["source_index"],
        )
        for record in episodes
    ]
    if len(source_identities) != len(set(source_identities)):
        raise ManifestError("episodes contains a duplicate source episode")

    file_values = _sequence(root.get("files"), "files")
    files: list[dict[str, object]] = []
    for position, value in enumerate(file_values):
        record = _mapping(value, f"files[{position}]")
        files.append(
            {
                "target_path": _relative_posix(
                    record.get("target_path"), f"files[{position}].target_path"
                ),
                "sha256": _sha256(
                    record.get("sha256"), f"files[{position}].sha256"
                ),
                "size": _integer(record.get("size"), f"files[{position}].size"),
            }
        )
    files.sort(key=lambda record: str(record["target_path"]))
    target_paths = [str(record["target_path"]) for record in files]
    if len(target_paths) != len(set(target_paths)):
        raise ManifestError("files contains a duplicate target_path")

    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": created_at,
        "episode_count": episode_count,
        "frame_count": frame_count,
        "episode_fingerprints": fingerprints,
        "episodes": episodes,
        "files": files,
    }


class MigrationLedger:
    """SQLite index of batches whose portable manifests have been published."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._connection = sqlite3.connect(self.path)
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                  batch_id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  manifest_path TEXT NOT NULL UNIQUE,
                  episode_count INTEGER NOT NULL,
                  frame_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                  fingerprint TEXT PRIMARY KEY,
                  batch_id TEXT NOT NULL REFERENCES batches(batch_id),
                  target_index INTEGER NOT NULL,
                  source_host TEXT NOT NULL,
                  source_dataset_root TEXT NOT NULL,
                  source_index INTEGER NOT NULL,
                  UNIQUE(batch_id, target_index)
                );
                CREATE TABLE IF NOT EXISTS files (
                  batch_id TEXT NOT NULL REFERENCES batches(batch_id),
                  target_path TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  size INTEGER NOT NULL,
                  PRIMARY KEY(batch_id, target_path)
                );
                """
            )
        except BaseException:
            self._connection.close()
            raise

    def __enter__(self) -> MigrationLedger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._connection.close()

    def migrated_fingerprints(self) -> set[str]:
        return {
            str(row[0])
            for row in self._connection.execute("SELECT fingerprint FROM episodes")
        }

    def next_batch_sequence(self) -> int:
        maximum = 0
        for (batch_id,) in self._connection.execute("SELECT batch_id FROM batches"):
            match = _BATCH_ID_RE.fullmatch(str(batch_id))
            if match is not None:
                maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def _matches_recorded_batch(
        self,
        manifest: Mapping[str, object],
        manifest_path: str,
    ) -> bool:
        batch_id = str(manifest["batch_id"])
        batch = self._connection.execute(
            """
            SELECT created_at, manifest_path, episode_count, frame_count
            FROM batches WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        expected_batch = (
            manifest["created_at"],
            manifest_path,
            manifest["episode_count"],
            manifest["frame_count"],
        )
        if batch != expected_batch:
            return False

        recorded_episodes = self._connection.execute(
            """
            SELECT target_index, fingerprint, source_host, source_dataset_root, source_index
            FROM episodes WHERE batch_id = ? ORDER BY target_index
            """,
            (batch_id,),
        ).fetchall()
        expected_episodes = [
            (
                episode["target_index"],
                episode["fingerprint"],
                episode["source_host"],
                episode["source_dataset_root"],
                episode["source_index"],
            )
            for episode in manifest["episodes"]
        ]
        if recorded_episodes != expected_episodes:
            return False

        recorded_files = self._connection.execute(
            """
            SELECT target_path, sha256, size
            FROM files WHERE batch_id = ? ORDER BY target_path
            """,
            (batch_id,),
        ).fetchall()
        expected_files = [
            (file["target_path"], file["sha256"], file["size"])
            for file in manifest["files"]
        ]
        return recorded_files == expected_files

    def _existing_reuse(self, batch_id: str, manifest_path: str) -> list[tuple[str, str]]:
        return self._connection.execute(
            "SELECT batch_id, manifest_path FROM batches "
            "WHERE batch_id = ? OR manifest_path = ?",
            (batch_id, manifest_path),
        ).fetchall()

    def _insert_validated_batch(
        self,
        manifest: Mapping[str, object],
        manifest_path: str,
    ) -> None:
        batch_id = str(manifest["batch_id"])
        self._connection.execute(
            """
            INSERT INTO batches(
                batch_id, created_at, manifest_path, episode_count, frame_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                manifest["created_at"],
                manifest_path,
                manifest["episode_count"],
                manifest["frame_count"],
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO episodes(
                fingerprint, batch_id, target_index, source_host,
                source_dataset_root, source_index
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    episode["fingerprint"],
                    batch_id,
                    episode["target_index"],
                    episode["source_host"],
                    episode["source_dataset_root"],
                    episode["source_index"],
                )
                for episode in manifest["episodes"]
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO files(batch_id, target_path, sha256, size)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    batch_id,
                    file["target_path"],
                    file["sha256"],
                    file["size"],
                )
                for file in manifest["files"]
            ],
        )

    def record_published_batch(
        self,
        manifest: Mapping[str, object],
        *,
        manifest_path: Path | None = None,
    ) -> None:
        validated = _validate_manifest(manifest)
        if manifest_path is None:
            helper_path = manifest.get("manifest_path")
            if not isinstance(helper_path, str) or not helper_path:
                raise ManifestError(
                    "manifest_path must be explicit or a nonempty manifest_path string field"
                )
            path_text = helper_path
        else:
            path_text = str(manifest_path)

        batch_id = str(validated["batch_id"])
        try:
            with self._connection:
                existing = self._existing_reuse(batch_id, path_text)
                if existing:
                    if (
                        len(existing) == 1
                        and existing[0] == (batch_id, path_text)
                        and self._matches_recorded_batch(validated, path_text)
                    ):
                        return
                    raise ManifestError(
                        "could not record manifest: conflicting batch_id or manifest_path reuse"
                    )
                self._insert_validated_batch(validated, path_text)
        except sqlite3.IntegrityError as exc:
            raise ManifestError(
                "could not record manifest: SQLite constraint conflict"
            ) from exc

    def reconcile_ready(self, ready_root: Path) -> None:
        ready = load_valid_ready_manifests(ready_root)
        prepared = [
            (str(manifest_path), manifest)
            for manifest_path, manifest in ready
        ]
        try:
            with self._connection:
                expected_counts = (
                    len(prepared),
                    sum(int(manifest["episode_count"]) for _, manifest in prepared),
                    sum(len(manifest["files"]) for _, manifest in prepared),
                )
                actual_counts = (
                    self._connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
                    self._connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                    self._connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                )
                exact = actual_counts == expected_counts and all(
                    self._matches_recorded_batch(manifest, path_text)
                    for path_text, manifest in prepared
                )
                if exact:
                    return

                self._connection.execute("DELETE FROM files")
                self._connection.execute("DELETE FROM episodes")
                self._connection.execute("DELETE FROM batches")
                for path_text, manifest in prepared:
                    self._insert_validated_batch(manifest, path_text)
        except sqlite3.IntegrityError as exc:
            raise ManifestError(
                "could not reconcile READY manifests: SQLite constraint conflict"
            ) from exc


def load_valid_ready_manifests(
    ready_root: Path,
) -> tuple[tuple[Path, dict[str, object]], ...]:
    """Read and validate direct published batch manifests without writing state."""
    root = Path(ready_root)
    if root.is_symlink():
        raise ManifestError("ready_root must not be a directory symlink")
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ManifestError("ready_root must be a directory")

    loaded: list[tuple[Path, dict[str, object]]] = []
    for batch_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if batch_dir.is_symlink():
            raise ManifestError(f"batch directory {batch_dir.name} must not be a symlink")
        if not batch_dir.is_dir():
            continue
        ready_path = batch_dir / "READY"
        if ready_path.is_symlink():
            raise ManifestError(f"{batch_dir.name}/READY must be a regular file")
        if not ready_path.exists():
            continue
        if not ready_path.is_file():
            raise ManifestError(f"{batch_dir.name}/READY must be a regular file")

        manifest_path = batch_dir / "migration_manifest.json"
        if manifest_path.is_symlink():
            raise ManifestError(
                f"{batch_dir.name}/migration_manifest.json must be a regular file"
            )
        if not manifest_path.is_file():
            raise ManifestError(
                f"published batch {batch_dir.name} is missing migration_manifest.json"
            )
        decoded = _read_manifest_document(manifest_path, batch_dir.name)
        try:
            validated = _validate_manifest(_mapping(decoded, "manifest"))
        except ManifestError as exc:
            raise ManifestError(
                f"published batch {batch_dir.name} has invalid manifest: {exc}"
            ) from exc
        if validated["batch_id"] != batch_dir.name:
            raise ManifestError(
                f"published directory {batch_dir.name} conflicts with manifest batch_id "
                f"{validated['batch_id']}"
            )
        loaded.append((manifest_path, validated))

    loaded.sort(key=lambda item: (str(item[1]["batch_id"]), item[0].as_posix()))
    owners: dict[str, str] = {}
    for _, manifest in loaded:
        batch_id = str(manifest["batch_id"])
        for fingerprint in manifest["episode_fingerprints"]:
            previous = owners.setdefault(str(fingerprint), batch_id)
            if previous != batch_id:
                raise ManifestError(
                    "duplicate episode fingerprint across ready batches: "
                    f"{fingerprint} appears in {previous} and {batch_id}"
                )
    return tuple(loaded)
