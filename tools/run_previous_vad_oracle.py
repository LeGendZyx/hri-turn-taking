"""CLI for the isolated one-tick previous-VAD oracle diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.vap.oracle_prev_vad_experiment import (
    aggregate_oracle_outer_matrix,
    audit_oracle_outer_aggregate,
    audit_oracle_outer_matrix,
    freeze_oracle_outer_protocol,
    run_oracle_outer_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, run, audit, or aggregate the exact-eight O0/O1 one-tick "
            "previous-VAD post-hoc nested-LOGO diagnostic."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="freeze the create-once Oracle protocol")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("canonical_protocol", type=Path)
    freeze.add_argument("destination", type=Path)

    run = commands.add_parser("run", help="run all eight post-hoc Oracle units")
    run.add_argument("protocol", type=Path)
    run.add_argument("source", type=Path)
    run.add_argument("output", type=Path)

    audit = commands.add_parser("audit", help="audit a completed eight-unit matrix")
    audit.add_argument("matrix", type=Path)
    audit.add_argument("source", type=Path)

    audit_aggregate = commands.add_parser(
        "audit-aggregate",
        help="independently replay-audit a published aggregate",
    )
    audit_aggregate.add_argument("aggregate", type=Path)
    audit_aggregate.add_argument("matrix", type=Path)
    audit_aggregate.add_argument("source", type=Path)

    aggregate = commands.add_parser(
        "aggregate",
        help="pool saved out-of-group probabilities without model inference",
    )
    aggregate.add_argument("matrix", type=Path)
    aggregate.add_argument("source", type=Path)
    aggregate.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        artifact = freeze_oracle_outer_protocol(
            args.source,
            args.canonical_protocol,
            args.destination,
        )
        payload = {
            "protocol_directory": str(artifact.directory),
            "protocol_sha256": artifact.protocol_sha256,
            "oracle_gold_input": True,
            "deployable": False,
            "production_eligible": False,
            "counted_eligible": False,
        }
    elif args.command == "run":
        artifact = run_oracle_outer_matrix(args.protocol, args.source, args.output)
        payload = {
            "matrix_directory": str(artifact.directory),
            "completed_units": artifact.index["completed_units"],
            "oracle_gold_input": True,
            "deployable": False,
            "production_eligible": False,
            "counted_eligible": False,
        }
    elif args.command == "audit":
        artifact = audit_oracle_outer_matrix(args.matrix, args.source)
        payload = {
            "matrix_directory": str(artifact.directory),
            "completed_units": artifact.index["completed_units"],
            "audit_ok": True,
            "oracle_gold_input": True,
            "deployable": False,
        }
    elif args.command == "audit-aggregate":
        artifact = audit_oracle_outer_aggregate(
            args.aggregate,
            args.matrix,
            args.source,
        )
        payload = {
            "aggregate_path": str(artifact.path),
            "aggregate_sha256": artifact.sha256,
            "audit_ok": True,
            "oracle_gold_input": True,
            "deployable": False,
        }
    else:
        output = aggregate_oracle_outer_matrix(args.matrix, args.source, args.output)
        payload = {
            "aggregate_path": str(output),
            "oracle_gold_input": True,
            "deployable": False,
            "production_eligible": False,
            "counted_eligible": False,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
