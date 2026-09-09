"""CLI for the isolated schema-v3 Gaze post-hoc outer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.vap.gaze_outer_experiment import (
    aggregate_gaze_outer_matrix,
    audit_gaze_outer_matrix,
    freeze_gaze_outer_protocol,
    run_gaze_outer_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, run, audit, or aggregate the exact-eight C0/CM/CG "
            "post-hoc Gaze nested-LOGO experiment."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="freeze the create-once protocol")
    freeze.add_argument("source_matrix", type=Path)
    freeze.add_argument("destination", type=Path)

    run = commands.add_parser("run", help="run all 12 post-hoc outer units")
    run.add_argument("protocol", type=Path)
    run.add_argument("source_matrix", type=Path)
    run.add_argument("output", type=Path)

    audit = commands.add_parser("audit", help="audit a completed 12-unit matrix")
    audit.add_argument("matrix", type=Path)
    audit.add_argument("source_matrix", type=Path)

    aggregate = commands.add_parser(
        "aggregate",
        help="pool saved out-of-group probabilities without model inference",
    )
    aggregate.add_argument("matrix", type=Path)
    aggregate.add_argument("source_matrix", type=Path)
    aggregate.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        artifact = freeze_gaze_outer_protocol(args.source_matrix, args.destination)
        payload = {
            "protocol_directory": str(artifact.directory),
            "protocol_sha256": artifact.protocol_sha256,
        }
    elif args.command == "run":
        artifact = run_gaze_outer_matrix(
            args.protocol,
            args.source_matrix,
            args.output,
        )
        payload = {
            "matrix_directory": str(artifact.directory),
            "completed_units": artifact.index["completed_units"],
            "production_eligible": False,
            "counted_eligible": False,
        }
    elif args.command == "audit":
        artifact = audit_gaze_outer_matrix(args.matrix, args.source_matrix)
        payload = {
            "matrix_directory": str(artifact.directory),
            "completed_units": artifact.index["completed_units"],
            "audit_ok": True,
        }
    else:
        output = aggregate_gaze_outer_matrix(
            args.matrix,
            args.source_matrix,
            args.output,
        )
        payload = {
            "aggregate_path": str(output),
            "production_eligible": False,
            "counted_eligible": False,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
