"""Read-only audit or explicit publication of gold-channel time-map contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hri_turn_taking.data.audio_contract import ACCEPTED_SCENES
from hri_turn_taking.data.audio_gold_alignment import (
    audit_audio_gold_alignment,
    write_audio_gold_alignment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fully replay published audio and sealed pose artifacts, then derive the "
            "non-destructive gold-only per-channel time maps. The default is read-only."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", choices=ACCEPTED_SCENES)
    selection.add_argument("--all", action="store_true", help="Audit all ten frozen scenes")
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output path; valid only with --scene and never overwrites",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory; valid only with --all and never overwrites",
    )
    return parser


def _summary(report: dict) -> dict:
    channels = report["mapping_contract"]["channels"]
    replay = report["formal_window_replay"]["channel_decisions"]
    return {
        "scene": report["scene"],
        "counted_gold_ready": report["counted_gold_ready"],
        "correction_s": {speaker: channels[speaker]["correction_s"] for speaker in channels},
        "authorized_tick_range": report["authorization"]["authorized_tick_range"],
        "corrected_quality": {
            speaker: {
                "status": decision["status"],
                "coarse_warning": decision["coarse_warning"],
                "median_residual_ms": decision["coarse_32ms"]["median_residual_ms"],
                "p95_abs_residual_ms": decision["coarse_32ms"][
                    "p95_abs_residual_ms"
                ],
                "drift_span_ms": decision["coarse_32ms"]["drift_span_ms"],
                "unique_peak_fraction": decision["coarse_32ms"][
                    "unique_peak_fraction"
                ],
                "median_peak_correlation": decision["coarse_32ms"][
                    "median_peak_correlation"
                ],
            }
            for speaker, decision in replay.items()
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output is not None and args.scene is None:
        raise SystemExit("--output requires --scene")
    if args.output_dir is not None and not args.all:
        raise SystemExit("--output-dir requires --all")
    if args.output is not None and args.output_dir is not None:
        raise SystemExit("choose only one publication option")

    scenes = ACCEPTED_SCENES if args.all else (args.scene,)
    summaries = []
    for scene in scenes:
        if args.output is not None:
            destination = write_audio_gold_alignment(scene, args.output)
            report = audit_audio_gold_alignment(scene)
            summary = _summary(report)
            summary["published"] = str(destination)
        elif args.output_dir is not None:
            destination = args.output_dir / f"{scene}.audio_gold_alignment.json"
            write_audio_gold_alignment(scene, destination)
            report = audit_audio_gold_alignment(scene)
            summary = _summary(report)
            summary["published"] = str(destination.resolve())
        else:
            summary = _summary(audit_audio_gold_alignment(scene))
            summary["published"] = None
        summaries.append(summary)
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
