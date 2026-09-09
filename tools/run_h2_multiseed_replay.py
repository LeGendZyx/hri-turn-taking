"""Run the isolated seed-42/73 extension of the post-hoc H2 matched replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.vap.h2_multiseed_replay import (
    ACOUSTIC_ROOT,
    SEED13_EVALUATION_ROOT,
    aggregate_h2_multiseed,
    audit_h2_multiseed_aggregate,
    audit_h2_multiseed_seed,
    calibrate_and_seal_h2_multiseed_traces,
    evaluate_h2_multiseed_seed,
    freeze_h2_multiseed_seed,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="freeze one canonical seed-42/73 matrix pair and gold-free projections",
    )
    freeze.add_argument("--seed", required=True, type=int, choices=(42, 73))
    freeze.add_argument("--output", required=True, type=Path)

    traces = subparsers.add_parser(
        "calibrate-and-seal-traces",
        help="fit inner-only calibration and seal gold-free outer traces",
    )
    traces.add_argument("--freeze", required=True, type=Path)
    traces.add_argument("--output", required=True, type=Path)
    traces.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate one seed only after its trace seal exists",
    )
    evaluate.add_argument("--freeze", required=True, type=Path)
    evaluate.add_argument("--traces", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)

    audit_seed = subparsers.add_parser(
        "audit-seed",
        help="cold-audit any completed prefix for one extension seed",
    )
    audit_seed.add_argument("--freeze", required=True, type=Path)
    audit_seed.add_argument("--traces", type=Path)
    audit_seed.add_argument("--evaluation", type=Path)
    audit_seed.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="bind immutable seed13 plus audited seed42/73 evaluations",
    )
    aggregate.add_argument("--seed13-evaluation", type=Path, default=SEED13_EVALUATION_ROOT)
    aggregate.add_argument("--seed42-evaluation", required=True, type=Path)
    aggregate.add_argument("--seed73-evaluation", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)

    audit_aggregate = subparsers.add_parser(
        "audit-aggregate",
        help="cold-rebuild and audit the three-seed aggregate",
    )
    audit_aggregate.add_argument("--aggregate", required=True, type=Path)
    audit_aggregate.add_argument(
        "--seed13-evaluation", type=Path, default=SEED13_EVALUATION_ROOT
    )
    audit_aggregate.add_argument("--seed42-evaluation", required=True, type=Path)
    audit_aggregate.add_argument("--seed73-evaluation", required=True, type=Path)
    audit_aggregate.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        payload = {
            "freeze_manifest": str(
                freeze_h2_multiseed_seed(args.output, model_seed=args.seed)
            )
        }
    elif args.command == "calibrate-and-seal-traces":
        payload = {
            "trace_seal_manifest": str(
                calibrate_and_seal_h2_multiseed_traces(
                    args.freeze,
                    args.output,
                    acoustic_root=args.acoustic_root,
                )
            )
        }
    elif args.command == "evaluate":
        payload = {
            "evaluation_manifest": str(
                evaluate_h2_multiseed_seed(
                    args.freeze,
                    args.traces,
                    args.output,
                    acoustic_root=args.acoustic_root,
                )
            )
        }
    elif args.command == "audit-seed":
        payload = audit_h2_multiseed_seed(
            args.freeze,
            trace_directory=args.traces,
            evaluation_directory=args.evaluation,
            acoustic_root=args.acoustic_root,
        )
    elif args.command == "aggregate":
        payload = {
            "aggregate_manifest": str(
                aggregate_h2_multiseed(
                    args.output,
                    seed13_evaluation=args.seed13_evaluation,
                    seed42_evaluation=args.seed42_evaluation,
                    seed73_evaluation=args.seed73_evaluation,
                    acoustic_root=args.acoustic_root,
                )
            )
        }
    else:
        payload = audit_h2_multiseed_aggregate(
            args.aggregate,
            seed13_evaluation=args.seed13_evaluation,
            seed42_evaluation=args.seed42_evaluation,
            seed73_evaluation=args.seed73_evaluation,
            acoustic_root=args.acoustic_root,
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
