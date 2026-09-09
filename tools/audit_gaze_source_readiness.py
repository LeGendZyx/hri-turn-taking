"""Read-only ten-scene nullable 2D Gaze evidence preflight.

The tool prints coverage only.  It has no output path or publish mode.
"""

from __future__ import annotations

import argparse
import json
import sys

from hri_turn_taking.data import analysis_base
from hri_turn_taking.data.gaze_source_readiness import (
    MATRIX_ARTIFACT_KIND,
    GazeSourceReadinessError,
    audit_gaze_source_preflight,
    validate_frozen_preflight_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay repository-local sealed Gaze/identity/analysis evidence and "
            "report nullable 2D coverage in memory. No feature is written, published, "
            "admitted to assembly, or marked counted-ready."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", choices=tuple(analysis_base.EXPECTED_TICK_RANGES))
    group.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    scenes = tuple(analysis_base.EXPECTED_TICK_RANGES) if args.all else (args.scene,)
    results = {}
    try:
        for index, scene in enumerate(scenes, start=1):
            print(
                f"[gaze-source-preflight] repo-local sealed replay {index}/{len(scenes)}: {scene}",
                file=sys.stderr,
                flush=True,
            )
            results[scene] = audit_gaze_source_preflight(scene)
        if args.all:
            validate_frozen_preflight_matrix(results)
    except (OSError, ValueError, GazeSourceReadinessError) as exc:
        raise SystemExit(f"Gaze source preflight failed: {exc}") from exc

    summary = {
        "artifact_kind": MATRIX_ARTIFACT_KIND,
        "output_role": "read_only_nullable_2d_gaze_evidence_coverage",
        "assembly_eligible": False,
        "counted_input_ready": False,
        "writer_available": False,
        "artifact_or_feature_published": False,
        "raw_video_opened_or_hydrated": False,
        "n_scenes": len(results),
        "n_analysis_ticks": sum(
            result.report["analysis_base"]["n_ticks"] for result in results.values()
        ),
        "n_evidence_rows": sum(len(result.evidence_df) for result in results.values()),
        "scenes": {
            scene: {
                "n_ticks": result.report["analysis_base"]["n_ticks"],
                "source_fps": result.report["source"]["source_fps"],
                "accepted_window": result.report["source"]["accepted_window"],
                "slot_presence": result.report["analysis_base"]["slot_presence"],
                "gaze_target_coverage": result.report["evidence"]["coverage"][
                    "gaze_target_coverage"
                ],
                "gaze_robot_coverage": result.report["evidence"]["coverage"]["gaze_robot_coverage"],
                "robot_slot_gaze_observed": False,
            }
            for scene, result in results.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
