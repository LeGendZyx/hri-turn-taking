"""Author/audit owner-attested Transcript-derived VAD and targeted clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.data.transcript_derived_vad import (
    SEMANTIC_CANDIDATES,
    TranscriptDerivedVadError,
    export_candidate_clips,
    load_transcript_derived_vad,
    write_owner_attestation,
    write_semantic_decisions,
    write_transcript_derived_vad,
)


def _decision(value: str) -> dict[str, object]:
    try:
        candidate_id, status, payload = value.split("=", maxsplit=2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "decision must be ID=accepted|rejected|replaced:JSON_EVENT=NOTE"
        ) from exc
    replacement: dict[str, object] | None = None
    if status.startswith("replaced:"):
        try:
            parsed = json.loads(status.removeprefix("replaced:"))
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                "replaced decision must contain one JSON event object"
            ) from exc
        if type(parsed) is not dict:
            raise argparse.ArgumentTypeError("replaced decision must contain one JSON event object")
        replacement = parsed
        status = "replaced"
    if status not in {"accepted", "rejected", "replaced"}:
        raise argparse.ArgumentTypeError("decision status is unsupported")
    if not candidate_id or not payload:
        raise argparse.ArgumentTypeError("decision ID and note must be non-empty")
    return {
        "candidate_id": candidate_id,
        "status": status,
        "replacement": replacement,
        "note": payload,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Owner-attested human-reviewed Transcript-derived VAD; no blind/full "
            "re-review and no arbitrary audio threshold."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    attest = commands.add_parser("attest")
    attest.add_argument("--scene", required=True)
    attest.add_argument("--owner", required=True)
    attest.add_argument("--attested-at", required=True)

    decide = commands.add_parser("decide")
    decide.add_argument("--scene", required=True)
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--decided-at", required=True)
    decide.add_argument(
        "--decision",
        action="append",
        default=[],
        type=_decision,
        help="ID=accepted=NOTE, ID=rejected=NOTE, or ID=replaced:{JSON_EVENT}=NOTE",
    )

    build = commands.add_parser("build")
    build.add_argument("--scene", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--scene", required=True)

    clips = commands.add_parser("clips")
    clips.add_argument("--candidate-id", required=True)
    clips.add_argument("--mode", choices=("separate", "mix"), default="separate")

    catalog = commands.add_parser("catalog")
    catalog.add_argument("--scene")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "attest":
            print(
                write_owner_attestation(
                    args.scene,
                    owner=args.owner,
                    attested_at=args.attested_at,
                )
            )
        elif args.command == "decide":
            print(
                write_semantic_decisions(
                    args.scene,
                    reviewer=args.reviewer,
                    decided_at=args.decided_at,
                    decisions=args.decision,
                )
            )
        elif args.command == "build":
            result = write_transcript_derived_vad(args.scene)
            print(result.target_csv)
        elif args.command == "audit":
            target = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "processed"
                / "transcript_derived_vad_targets"
                / args.scene
                / "vad_target.v1.csv"
            )
            result = load_transcript_derived_vad(target)
            print(
                json.dumps(
                    {
                        "scene": result.contract["scene"],
                        "research_label": result.contract["research_label"],
                        "counted_gold_ready": result.contract["counted_gold_ready"],
                        "n_ticks": len(result.target_df),
                        "structural_qc": result.contract["structural_qc"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "clips":
            print(export_candidate_clips(args.candidate_id, mode=args.mode))
        else:
            value = (
                {args.scene: SEMANTIC_CANDIDATES[args.scene]} if args.scene else SEMANTIC_CANDIDATES
            )
            print(json.dumps(value, indent=2, sort_keys=True))
    except (KeyError, OSError, TranscriptDerivedVadError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
