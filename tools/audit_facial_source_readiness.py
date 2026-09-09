"""Print the read-only Facial delivery evidence/readiness report as JSON."""

from __future__ import annotations

import argparse
import json

from hri_turn_taking.data import analysis_base
from hri_turn_taking.data.facial_source_readiness import (
    FacialSourceReadinessError,
    audit_facial_source_matrix,
    audit_facial_source_readiness,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory repository-local Facial CSV evidence against sealed identity "
            "and canonical analysis ticks. This tool has no writer or promotion path."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", choices=tuple(analysis_base.EXPECTED_TICK_RANGES))
    group.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = (
            audit_facial_source_matrix()
            if args.all
            else audit_facial_source_readiness(args.scene)
        )
    except (OSError, ValueError, FacialSourceReadinessError) as exc:
        raise SystemExit(f"Facial source readiness audit failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
