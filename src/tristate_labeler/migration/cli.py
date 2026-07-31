"""Command-line contract for one robot-to-inference migration attempt."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import stat
import time
import uuid

from .models import MigrationConfig, RunStatus
from .orchestrator import (
    MigrationService,
    OrchestrationError,
    RunResult,
    _ensure_safe_directory,
    _fsync_directory,
)


ServiceFactory = Callable[[MigrationConfig], MigrationService]


def _stable_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull one immutable 20-episode HIL batch from the robot."
    )
    parser.add_argument(
        "--robot", default="zme@lite-0030.taild22f37.ts.net"
    )
    parser.add_argument("--source-root", default="/home/zme/datasets")
    parser.add_argument(
        "--output-root", type=Path, default=Path("/home/zme/hil_rl_data")
    )
    parser.add_argument("--batch-size", type=int, choices=(20,), default=20)
    parser.add_argument("--stable-seconds", type=_stable_seconds, default=3.0)
    parser.add_argument(
        "--identity-file",
        type=Path,
        default=Path.home() / ".ssh/id_ed25519_hil_transfer",
    )
    parser.add_argument(
        "--known-hosts-file",
        type=Path,
        default=Path.home() / ".ssh/known_hosts_hil_transfer",
    )
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _event(
    *,
    run_id: str,
    status: str,
    available: int | None,
    required: int,
    batch_id: str | None,
    elapsed_seconds: float,
    error_type: str | None,
    dry_run: bool,
    selected_fingerprints: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "timestamp": _timestamp(),
        "run_id": run_id,
        "status": status,
        "available": available,
        "required": required,
        "batch_id": batch_id,
        "elapsed_seconds": float(elapsed_seconds),
        "error_type": error_type,
        "dry_run": dry_run,
        "selected_count": len(selected_fingerprints),
        "selected_fingerprints": list(selected_fingerprints),
    }


def _json_line(value: object) -> bytes:
    try:
        document = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise OrchestrationError("migration log event is not strict JSON") from exc
    return (document + "\n").encode("utf-8")


def _append_event(output_root: Path, event: dict[str, object]) -> None:
    logs_root = _ensure_safe_directory(Path(output_root) / "logs")
    path = logs_root / "migration.jsonl"
    try:
        before = path.lstat()
    except FileNotFoundError:
        before = None
    except OSError as exc:
        raise OrchestrationError(
            f"could not inspect migration log: {exc.__class__.__name__}"
        ) from exc
    if before is not None and (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise OrchestrationError(
            "migration log must be a regular non-symlink single-link file"
        )

    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_ISLNK(current.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise OrchestrationError(
                "migration log changed identity while it was opened"
            )
        payload = _json_line(event)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short append to migration log")
        os.fsync(descriptor)
    except OrchestrationError:
        raise
    except OSError as exc:
        raise OrchestrationError(
            f"could not append migration log: {exc.__class__.__name__}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_directory(logs_root)


def _result_event(run_id: str, result: RunResult, elapsed: float) -> dict[str, object]:
    return _event(
        run_id=run_id,
        status=result.status.value,
        available=result.available,
        required=result.required,
        batch_id=result.batch_id,
        elapsed_seconds=elapsed,
        error_type=None,
        dry_run=result.dry_run,
        selected_fingerprints=tuple(item.fingerprint for item in result.selected),
    )


def _print_terminal(event: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return
    status = str(event["status"])
    if status == RunStatus.ERROR.value:
        print(f"ERROR {event['error_type']}")
        return
    available = int(event["available"])
    required = int(event["required"])
    if status == RunStatus.CREATED.value:
        print(f"CREATED {event['batch_id']} {available}/{required}")
    else:
        print(f"{status} {available}/{required}")


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: ServiceFactory = MigrationService,
) -> int:
    args = _parser().parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    config = MigrationConfig(
        output_root=args.output_root,
        robot=args.robot,
        source_root=args.source_root,
        batch_size=args.batch_size,
        stable_seconds=args.stable_seconds,
        identity_file=args.identity_file,
        known_hosts_file=args.known_hosts_file,
        ffprobe=args.ffprobe,
    )
    run_id = uuid.uuid4().hex
    started = time.monotonic()
    start_event = _event(
        run_id=run_id,
        status="STARTED",
        available=None,
        required=config.batch_size,
        batch_id=None,
        elapsed_seconds=0.0,
        error_type=None,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        try:
            _append_event(config.output_root, start_event)
        except Exception as exc:
            error = _event(
                run_id=run_id,
                status=RunStatus.ERROR.value,
                available=None,
                required=config.batch_size,
                batch_id=None,
                elapsed_seconds=max(0.0, time.monotonic() - started),
                error_type=type(exc).__name__,
                dry_run=False,
            )
            _print_terminal(error, json_output=args.json_output)
            return 1

    try:
        service = service_factory(config)
        result = service.run(dry_run=args.dry_run)
        if not isinstance(result, RunResult):
            raise OrchestrationError("migration service returned an invalid result")
        terminal = _result_event(
            run_id, result, max(0.0, time.monotonic() - started)
        )
        if not args.dry_run:
            _append_event(config.output_root, terminal)
        _print_terminal(terminal, json_output=args.json_output)
        return 0 if result.status is not RunStatus.ERROR else 1
    except Exception as exc:
        terminal = _event(
            run_id=run_id,
            status=RunStatus.ERROR.value,
            available=None,
            required=config.batch_size,
            batch_id=None,
            elapsed_seconds=max(0.0, time.monotonic() - started),
            error_type=type(exc).__name__,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            try:
                _append_event(config.output_root, terminal)
            except Exception:
                pass
        _print_terminal(terminal, json_output=args.json_output)
        return 1
