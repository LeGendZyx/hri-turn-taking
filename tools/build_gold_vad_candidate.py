"""Build/audit candidate-only gold-VAD boundary evidence.

This tool intentionally has no flag that can make an artifact assembly-eligible
or counted-gold-ready.  Formal source paths are derived from the scene ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.data.gold_vad_candidate import (
    GoldVadCandidateError,
    create_blind_audit_decisions,
    load_gold_vad_candidate,
    reveal_blind_audit,
    write_gold_vad_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Candidate-only Transcript/lavalier gold-VAD evidence (never formal gold)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="publish one candidate evidence bundle")
    build.add_argument("--scene", required=True)
    build.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="fully replay a candidate bundle")
    audit.add_argument("candidate", type=Path)

    decisions = subparsers.add_parser(
        "init-decisions", help="create a separate suggestion-free blind decision template"
    )
    decisions.add_argument("candidate", type=Path)
    decisions.add_argument("--output", type=Path, required=True)

    reveal = subparsers.add_parser(
        "reveal", help="reveal automatic suggestions after every blind decision is complete"
    )
    reveal.add_argument("candidate", type=Path)
    reveal.add_argument("decisions", type=Path)
    reveal.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "build":
            print(write_gold_vad_candidate(args.scene, args.output))
        elif args.command == "audit":
            bundle = load_gold_vad_candidate(args.candidate)
            print(
                json.dumps(
                    {
                        "scene": bundle["scene"],
                        "candidate_only": bundle["candidate_only"],
                        "assembly_eligible": bundle["assembly_eligible"],
                        "counted_gold_ready": bundle["counted_gold_ready"],
                        "source_selection": bundle["source_selection"],
                        "n_candidates": bundle["diagnostics"]["n_total_candidates"],
                        "n_blind_audit_items": bundle["blind_audit_queue"]["n_selected"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "init-decisions":
            print(create_blind_audit_decisions(args.candidate, args.output))
        else:
            print(reveal_blind_audit(args.candidate, args.decisions, args.output))
    except (GoldVadCandidateError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
