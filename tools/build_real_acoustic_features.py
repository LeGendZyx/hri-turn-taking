"""Read-only real acoustic preflight; this tool has no publish mode."""

from __future__ import annotations

import argparse
import json

from hri_turn_taking.data import analysis_base
from hri_turn_taking.data.real_acoustic_features import (
    RealAcousticFeatureError,
    derive_real_acoustic_preflight,
    validate_frozen_preflight_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fully replay canonical real audio/analysis sources and derive causal "
            "energy/F0/voiced in memory. No artifact is published: owner approval "
            "of threshold 0.45 does not promote this legacy feasibility scaffold."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", choices=tuple(analysis_base.EXPECTED_TICK_RANGES))
    group.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    scenes = (
        tuple(analysis_base.EXPECTED_TICK_RANGES)
        if args.all
        else (args.scene,)
    )
    results = {}
    try:
        for index, scene in enumerate(scenes, start=1):
            print(
                f"[real-acoustic-preflight] strict replay "
                f"{index}/{len(scenes)}: {scene}",
                flush=True,
            )
            results[scene] = derive_real_acoustic_preflight(scene)
        if args.all:
            validate_frozen_preflight_matrix(results)
    except (OSError, ValueError, RealAcousticFeatureError) as exc:
        raise SystemExit(f"real acoustic preflight failed: {exc}") from exc
    summary = {
        "artifact_kind": "vap_real_causal_acoustic_preflight_matrix",
        "parameter_status": "owner_approval_pending",
        "assembly_eligible": False,
        "counted_input_ready": False,
        "n_scenes": len(results),
        "n_rows": sum(len(result.feature_df) for result in results.values()),
        "scenes": {
            scene: {
                "n_rows": len(result.feature_df),
                "tick_start_inclusive": result.report["analysis_base"][
                    "tick_start_inclusive"
                ],
                "tick_end_exclusive": result.report["analysis_base"][
                    "tick_end_exclusive"
                ],
                "voiced_fraction": float(result.feature_df["voiced"].mean()),
            }
            for scene, result in results.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
