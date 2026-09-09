"""Read-only ASD source preflight; never downloads media or emits features."""

from __future__ import annotations

import argparse
import json

from hri_turn_taking.data import analysis_base
from hri_turn_taking.data.asd_source_preflight import (
    ASDSourcePreflightError,
    audit_asd_source_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay sealed ASD camera/identity/time/audio source contracts. "
            "Video files are stat-only: offline placeholders are not opened or hydrated."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", choices=tuple(analysis_base.EXPECTED_TICK_RANGES))
    group.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    scenes = (
        tuple(analysis_base.EXPECTED_TICK_RANGES) if args.all else (args.scene,)
    )
    reports = {}
    try:
        for index, scene in enumerate(scenes, start=1):
            print(
                f"[asd-source-preflight] sealed replay {index}/{len(scenes)}: {scene}",
                flush=True,
            )
            reports[scene] = audit_asd_source_preflight(scene)
    except (OSError, ValueError, ASDSourcePreflightError) as exc:
        raise SystemExit(f"ASD source preflight failed: {exc}") from exc
    summary = {
        "artifact_kind": "vap_real_asd_source_preflight_matrix",
        "assembly_eligible": False,
        "counted_input_ready": False,
        "stock_asd_executed": False,
        "raw_video_opened_or_hydrated": False,
        "n_scenes": len(reports),
        "n_ticks": sum(
            report["source_plan"]["analysis_tick_range"]["n_ticks"]
            for report in reports.values()
        ),
        "scenes": {
            scene: {
                "source_fps": report["source_plan"]["source_fps"],
                "first_segment_only": report["source_plan"][
                    "accepted_source_window"
                ]["first_segment_only"],
                "video_byte_states": sorted(
                    {
                        source["video"]["bytes_status"]
                        for source in report["source_plan"][
                            "participant_sources"
                        ].values()
                    }
                ),
            }
            for scene, report in reports.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
