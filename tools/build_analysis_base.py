"""Replay or explicitly publish the ten-scene common-window pose bases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hri_turn_taking.data.analysis_base import (
    EXPECTED_TICK_RANGES,
    EXPECTED_TOTAL_ROWS,
    AnalysisBaseError,
    AnalysisBaseResult,
    default_analysis_base_authority,
    derive_analysis_base,
    reaudit_analysis_base,
    validate_frozen_matrix,
    write_analysis_base,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fully replay the canonical pose intake and gold-audio alignment, then "
            "derive the exact integer-tick common-window crop. The default is read-only."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", choices=tuple(EXPECTED_TICK_RANGES))
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--reaudit",
        type=Path,
        help="Explicit full raw-source re-audit of one published CSV",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Publish only to the code-owned data/processed/analysis_base path; "
            "otherwise the command is read-only"
        ),
    )
    return parser


def _summary(
    scene: str,
    result: AnalysisBaseResult,
    *,
    published: Path | None,
) -> dict[str, object]:
    frame_df = result.frame_df
    provenance = result.provenance
    contract = provenance["analysis_window_contract"]
    return {
        "scene": scene,
        "artifact_kind": provenance["artifact_kind"],
        "authorized_tick_range": contract["authorized_tick_range"],
        "channel_alignment": contract["channel_alignment"],
        "d3_quality": contract["d3_quality"],
        "n_rows": len(frame_df),
        "trainable": provenance["trainable"],
        "published": None if published is None else str(published.resolve()),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.reaudit is not None and args.publish:
        raise SystemExit("--reaudit is read-only and cannot be combined with publication")

    if args.reaudit is not None:
        try:
            print(
                f"[analysis-base] strict raw re-audit: {args.reaudit}",
                file=sys.stderr,
                flush=True,
            )
            result = reaudit_analysis_base(args.reaudit)
        except (AnalysisBaseError, OSError, ValueError) as exc:
            raise SystemExit(f"analysis-base raw re-audit failed: {exc}") from exc
        scene = result.provenance["session_id"]
        payload = {
            "scenes": [_summary(scene, result, published=result.output_csv)],
            "n_scenes": 1,
            "n_rows": len(result.frame_df),
            "explicit_raw_reaudit": True,
            "formal_training_ready": False,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        return 0

    scenes = tuple(EXPECTED_TICK_RANGES) if args.all else (args.scene,)
    authority = default_analysis_base_authority()
    results = {}
    summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            print(
                f"[analysis-base] strict replay {index}/{len(scenes)}: {scene}",
                file=sys.stderr,
                flush=True,
            )
            if args.publish:
                published = authority.analysis_base_root / f"{scene}.csv"
                result = write_analysis_base(
                    scene,
                    published,
                    authority=authority,
                )
            else:
                published = None
                result = derive_analysis_base(scene, authority=authority)
            results[scene] = result
            summaries.append(_summary(scene, result, published=published))
        if args.all:
            validate_frozen_matrix(results)
    except (AnalysisBaseError, OSError, ValueError) as exc:
        raise SystemExit(f"analysis-base gate failed: {exc}") from exc

    payload = {
        "scenes": summaries,
        "n_scenes": len(summaries),
        "n_rows": sum(int(item["n_rows"]) for item in summaries),
        "expected_total_rows": EXPECTED_TOTAL_ROWS if args.all else None,
        "matrix_publication_semantics": (
            "per_scene_no_clobber_not_atomic_across_ten_scenes"
            if args.publish and args.all
            else None
        ),
        "completion_inventory_written": False,
        "formal_training_ready": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
