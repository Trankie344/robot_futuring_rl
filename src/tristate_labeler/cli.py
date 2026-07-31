from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tristate-labeler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the LAN labeling web server")
    serve.add_argument("--dataset", type=Path, required=True)
    serve.add_argument("--db", type=Path, default=Path("workspace/labeler.db"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--stride", type=int, default=10, choices=[30, 15, 10, 5])

    init_cmd = subparsers.add_parser("init", help="Initialize database and generate tasks")
    init_cmd.add_argument("--dataset", type=Path, required=True)
    init_cmd.add_argument("--db", type=Path, default=Path("workspace/labeler.db"))
    init_cmd.add_argument("--stride", type=int, default=10, choices=[30, 15, 10, 5])

    export_cmd = subparsers.add_parser("export", help="Export annotations")
    export_cmd.add_argument("--db", type=Path, default=Path("workspace/labeler.db"))
    export_cmd.add_argument("--out", type=Path, default=Path("exports"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from .app import run_server

        run_server(dataset_root=args.dataset, db_path=args.db, host=args.host, port=args.port, stride=args.stride)
        return 0

    if args.command == "init":
        from .database import connect, init_db
        from .dataset import load_dataset_collection
        from .tasks import ensure_dataset_row, generate_tasks

        datasets = load_dataset_collection(args.dataset)
        inserted = 0
        with connect(args.db) as conn:
            init_db(conn)
            for dataset in datasets:
                dataset_id = ensure_dataset_row(conn, dataset)
                inserted += generate_tasks(conn, dataset_id, dataset.episodes, stride=args.stride)
        print(f"Generated {inserted} new tasks for {len(datasets)} dataset(s)")
        return 0

    if args.command == "export":
        from .database import connect
        from .exporter import export_annotations

        with connect(args.db) as conn:
            result = export_annotations(conn, args.out)
        print(f"Wrote {result.jsonl_path} and {result.csv_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
