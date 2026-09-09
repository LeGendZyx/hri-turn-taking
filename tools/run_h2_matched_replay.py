"""Run the isolated, post-hoc, non-counted H2 matched replay.

The commands preserve a process boundary between outer trace generation and
evaluation-only gold access.  Run ``freeze`` first, then
``calibrate-and-seal-traces``, then ``evaluate``.  ``audit`` can validate any
completed prefix without starting training or model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.vap.h2_matched_replay import (
    ACOUSTIC_ROOT,
    PRIMARY_MATRIX,
    SENSITIVITY_MATRIX,
    audit_h2_matched_replay,
    calibrate_and_seal_traces,
    evaluate_h2_matched_replay,
    freeze_h2_matched_replay,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="freeze fixed A+H artifacts and seal gold-free outer views"
    )
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--primary-matrix", type=Path, default=PRIMARY_MATRIX)
    freeze.add_argument("--sensitivity-matrix", type=Path, default=SENSITIVITY_MATRIX)

    traces = subparsers.add_parser(
        "calibrate-and-seal-traces",
        help="fit inner-only calibration and seal gold-free outer policy traces",
    )
    traces.add_argument("--freeze", required=True, type=Path)
    traces.add_argument("--output", required=True, type=Path)
    traces.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate sealed traces against outer gold"
    )
    evaluate.add_argument("--freeze", required=True, type=Path)
    evaluate.add_argument("--traces", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)

    audit = subparsers.add_parser("audit", help="replay-audit a completed phase prefix")
    audit.add_argument("--freeze", required=True, type=Path)
    audit.add_argument("--traces", type=Path)
    audit.add_argument("--evaluation", type=Path)
    audit.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        path = freeze_h2_matched_replay(
            args.output,
            primary_matrix=args.primary_matrix,
            sensitivity_matrix=args.sensitivity_matrix,
        )
        payload = {"freeze_manifest": str(path)}
    elif args.command == "calibrate-and-seal-traces":
        path = calibrate_and_seal_traces(
            args.freeze,
            args.output,
            acoustic_root=args.acoustic_root,
        )
        payload = {"trace_seal_manifest": str(path)}
    elif args.command == "evaluate":
        path = evaluate_h2_matched_replay(
            args.freeze,
            args.traces,
            args.output,
            acoustic_root=args.acoustic_root,
        )
        payload = {"evaluation_manifest": str(path)}
    else:
        payload = audit_h2_matched_replay(
            args.freeze,
            trace_directory=args.traces,
            evaluation_directory=args.evaluation,
            acoustic_root=args.acoustic_root,
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
