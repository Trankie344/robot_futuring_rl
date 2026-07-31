import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from openpi.training.rl_token.stage2 import identity


def test_canonical_json_hash_ignores_mapping_insertion_order():
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    canonical = b'{"a":1,"b":[2,3]}\n'
    expected_digest = "06d1ac940bec12987f319657ce46130daa57ab2d831421ddb892eba6a4509692"

    assert hashlib.sha256(canonical).hexdigest() == expected_digest
    assert identity.sha256_json(left) == identity.sha256_json(right)
    assert identity.canonical_json_bytes(left) == canonical
    assert identity.sha256_json(left) == expected_digest


def test_canonical_json_preserves_unicode_as_utf8():
    value = {"任务": "叠衣服", "message": "你好"}
    assert identity.canonical_json_bytes(value) == '{"message":"你好","任务":"叠衣服"}\n'.encode()


def test_sha256_file_hashes_bytes(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert identity.sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_atomic_write_json_refuses_existing_destination(tmp_path: Path):
    path = tmp_path / "manifest.json"
    identity.atomic_write_json(path, {"version": 1})
    original = path.read_bytes()
    assert json.loads(original) == {"version": 1}
    with pytest.raises(FileExistsError):
        identity.atomic_write_json(path, {"version": 2})
    assert path.read_bytes() == original
    assert _temporary_paths(path) == []


def test_atomic_write_json_rejects_nan_without_publishing(tmp_path: Path):
    path = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        identity.atomic_write_json(path, {"value": float("nan")})

    assert not path.exists()
    assert _temporary_paths(path) == []


def test_atomic_write_json_cleans_temporary_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"
    real_fdopen = identity.os.fdopen

    class _FailingWriter:
        def __init__(self, stream):
            self._stream = stream

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            self._stream.close()

        def write(self, _payload):
            raise OSError("write failed")

    def fail_fdopen(descriptor, mode):
        return _FailingWriter(real_fdopen(descriptor, mode))

    monkeypatch.setattr(identity.os, "fdopen", fail_fdopen)

    with pytest.raises(OSError, match="write failed"):
        identity.atomic_write_json(path, {"version": 1})

    assert not path.exists()
    assert _temporary_paths(path) == []


def test_atomic_write_json_cleans_temporary_after_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"

    def fail_link(_source, _destination):
        raise OSError("link failed")

    monkeypatch.setattr(identity.os, "link", fail_link)

    with pytest.raises(OSError, match="link failed"):
        identity.atomic_write_json(path, {"version": 1})

    assert not path.exists()
    assert _temporary_paths(path) == []


def test_sha256_file_reads_fixed_size_until_eof(monkeypatch: pytest.MonkeyPatch):
    chunks = iter((b"abc", b"def", b""))
    read_sizes = []

    class _FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return None

        def read(self, size):
            read_sizes.append(size)
            return next(chunks)

    def fake_open(_path, mode):
        assert mode == "rb"
        return _FakeStream()

    monkeypatch.setattr(Path, "open", fake_open)

    assert identity.sha256_file(Path("unused")) == ("bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721")
    assert read_sizes == [8 * 1024 * 1024] * 3


def test_atomic_write_json_closes_descriptor_when_fdopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"
    captured_descriptors = []

    def fail_fdopen(descriptor, _mode):
        captured_descriptors.append(descriptor)
        raise OSError("fdopen failed")

    monkeypatch.setattr(identity.os, "fdopen", fail_fdopen)

    with pytest.raises(OSError, match="fdopen failed"):
        identity.atomic_write_json(path, {"version": 1})

    assert not path.exists()
    assert _temporary_paths(path) == []
    assert len(captured_descriptors) == 1
    descriptor = captured_descriptors[0]
    try:
        with pytest.raises(OSError, match=r".+") as exc_info:
            os.fstat(descriptor)
        assert exc_info.value.errno == errno.EBADF
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def test_atomic_write_json_loses_publish_race_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"
    real_link = identity.os.link

    def race_link(source, destination):
        Path(destination).write_bytes(b"race winner")
        real_link(source, destination)

    monkeypatch.setattr(identity.os, "link", race_link)

    with pytest.raises(FileExistsError):
        identity.atomic_write_json(path, {"version": 1})

    assert path.read_bytes() == b"race winner"
    assert _temporary_paths(path) == []


def test_atomic_write_json_fsyncs_file_then_directory_and_closes_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"
    real_open = identity.os.open
    real_fsync = identity.os.fsync
    real_close = identity.os.close
    directory_descriptors = []
    closed_descriptors = []
    fsync_modes = []

    def track_open(file, flags, *args, **kwargs):
        descriptor = real_open(file, flags, *args, **kwargs)
        if Path(file) == path.parent and flags == os.O_RDONLY:
            directory_descriptors.append(descriptor)
        return descriptor

    def track_fsync(descriptor):
        fsync_modes.append(os.fstat(descriptor).st_mode)
        return real_fsync(descriptor)

    def track_close(descriptor):
        closed_descriptors.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(identity.os, "open", track_open)
    monkeypatch.setattr(identity.os, "fsync", track_fsync)
    monkeypatch.setattr(identity.os, "close", track_close)

    identity.atomic_write_json(path, {"version": 1})

    assert len(fsync_modes) == 2
    assert stat.S_ISREG(fsync_modes[0])
    assert stat.S_ISDIR(fsync_modes[1])
    assert len(directory_descriptors) == 1
    assert directory_descriptors[0] in closed_descriptors
    with pytest.raises(OSError, match=r".+") as exc_info:
        os.fstat(directory_descriptors[0])
    assert exc_info.value.errno == errno.EBADF
    assert _temporary_paths(path) == []


def test_atomic_write_json_file_fsync_failure_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"

    def fail_fsync(_descriptor):
        raise OSError("file fsync failed")

    monkeypatch.setattr(identity.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="file fsync failed"):
        identity.atomic_write_json(path, {"version": 1})

    assert not path.exists()
    assert _temporary_paths(path) == []


def test_atomic_write_json_directory_open_failure_keeps_published_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"
    real_open = identity.os.open

    def fail_directory_open(file, flags, *args, **kwargs):
        if Path(file) == path.parent and flags == os.O_RDONLY:
            raise OSError("directory open failed")
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(identity.os, "open", fail_directory_open)

    with pytest.raises(OSError, match="directory open failed"):
        identity.atomic_write_json(path, {"version": 1})

    assert path.read_bytes() == identity.canonical_json_bytes({"version": 1})
    assert _temporary_paths(path) == []


def test_atomic_write_json_directory_fsync_failure_closes_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "manifest.json"
    real_fsync = identity.os.fsync
    fsync_modes = []
    fsync_descriptors = []

    def fail_directory_fsync(descriptor):
        fsync_descriptors.append(descriptor)
        fsync_modes.append(os.fstat(descriptor).st_mode)
        if len(fsync_descriptors) == 2:
            raise OSError("directory fsync failed")
        return real_fsync(descriptor)

    monkeypatch.setattr(identity.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        identity.atomic_write_json(path, {"version": 1})

    assert len(fsync_modes) == 2
    assert stat.S_ISREG(fsync_modes[0])
    assert stat.S_ISDIR(fsync_modes[1])
    directory_descriptor = fsync_descriptors[1]
    with pytest.raises(OSError, match=r".+") as exc_info:
        os.fstat(directory_descriptor)
    assert exc_info.value.errno == errno.EBADF
    assert path.read_bytes() == identity.canonical_json_bytes({"version": 1})
    assert _temporary_paths(path) == []


def _temporary_paths(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))
