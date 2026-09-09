"""Read-only real-audio source/synchronization audit.

By default this command writes nothing: it prints a deterministic JSON summary.
An explicit ``--output`` (one scene) or ``--output-dir`` (``--all``) opt-in is
required to publish replayable contracts, and publication is no-clobber.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from hri_turn_taking.data.audio_contract import (
    ACCEPTED_SCENES,
    audit_audio_scene,
    validate_analysis_window,
    write_audio_contract,
)


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": report["scene"],
        "counted_gold_ready": report["counted_gold_ready"],
        "master_channel_index_to_speaker": report["master_sample_replay"][
            "channel_index_to_speaker"
        ],
        "upstream_channel_declaration_status": report["master_sample_replay"]["declaration_status"],
        "hpe_anchor_evidence_strength": report["hpe_mic_a_anchor"]["upstream_evidence_strength"],
        "reference_sync": {
            key: report["reference_to_equal_weight_master"][key]
            for key in (
                "quality_pass",
                "median_residual_ms",
                "p95_abs_residual_ms",
                "drift_span_ms",
                "unique_peak_fraction",
                "failure_reasons",
            )
        },
        "inter_lavalier_sync": {
            speaker: {
                key: result[key]
                for key in (
                    "quality_pass",
                    "median_residual_ms",
                    "p95_abs_residual_ms",
                    "drift_span_ms",
                    "unique_peak_fraction",
                    "proposed_target_correction_ms",
                    "failure_reasons",
                )
            }
            for speaker, result in report["inter_lavalier_to_mic_a"].items()
        },
        "analysis_window_contract": report["analysis_window_contract"],
        "gates": report["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit canonical WAV identity, master permutation, and segmented sync; "
            "never modifies source audio and never declares counted gold ready."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", choices=ACCEPTED_SCENES)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit no-clobber contract path; valid only with --scene.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit no-clobber contract directory; valid only with --all.",
    )
    parser.add_argument("--analysis-start-s", type=Decimal)
    parser.add_argument("--analysis-end-s", type=Decimal)
    args = parser.parse_args()
    if (args.analysis_start_s is None) != (args.analysis_end_s is None):
        parser.error("--analysis-start-s and --analysis-end-s must be supplied together")
    if args.output is not None and args.scene is None:
        parser.error("--output is valid only with --scene")
    if args.output_dir is not None and not args.all:
        parser.error("--output-dir is valid only with --all")
    if args.output is not None and args.output_dir is not None:
        parser.error("--output and --output-dir are mutually exclusive")

    scenes = ACCEPTED_SCENES if args.all else (args.scene,)
    summaries: list[dict[str, Any]] = []
    for scene in scenes:
        report = audit_audio_scene(scene)
        summary = _summary(report)
        if args.analysis_start_s is not None:
            summary["requested_analysis_window"] = validate_analysis_window(
                report, args.analysis_start_s, args.analysis_end_s
            )
        if args.output is not None:
            summary["published_contract"] = str(write_audio_contract(scene, args.output))
        elif args.output_dir is not None:
            destination = args.output_dir / f"{scene}.audio_contract.json"
            summary["published_contract"] = str(write_audio_contract(scene, destination))
        summaries.append(summary)
    print(json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
