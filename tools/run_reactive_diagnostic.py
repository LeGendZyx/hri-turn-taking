"""Run the exact-eight causal ``voiced``-only reactive diagnostic.

The two explicit phases make the gold-independence boundary auditable:
``trigger`` seals every acoustic-only trigger row, then ``evaluate`` may open
human-reviewed-Transcript-derived VAD for event definitions and labels.  The
``run`` convenience command executes those phases in that order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.vap.reactive_diagnostic import (
    evaluate_trigger_artifact,
    generate_trigger_artifact,
    run_reactive_diagnostic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("trigger", "seal all gold-independent voiced-only triggers"),
        ("evaluate", "evaluate an existing sealed trigger artifact against gold"),
        ("run", "seal triggers, then perform the separately gated evaluation"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--protocol", required=True, type=Path)
        subparser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "trigger":
        path = generate_trigger_artifact(args.protocol, args.output)
        role = "trigger_manifest"
    elif args.command == "evaluate":
        path = evaluate_trigger_artifact(args.protocol, args.output)
        role = "evaluation_manifest"
    else:
        path = run_reactive_diagnostic(args.protocol, args.output)
        role = "evaluation_manifest"
    print(json.dumps({role: str(path)}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
