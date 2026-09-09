"""Build, review-seal, or audit one canonical formal acoustic scene."""

from __future__ import annotations

import argparse
import json

from hri_turn_taking.data import analysis_base
from hri_turn_taking.data.formal_acoustic_features import (
    FormalAcousticFeatureError,
    build_pending_acoustic_scene,
    load_pending_acoustic_scene,
    load_reviewed_acoustic_scene,
    reaudit_formal_acoustic_scene,
    seal_reviewed_acoustic_scene,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Operate on exactly one code-owned formal energy/F0/voiced bundle. "
            "This tool never writes an independent review receipt."
        )
    )
    parser.add_argument("scene", choices=tuple(analysis_base.EXPECTED_TICK_RANGES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-pending", action="store_true")
    mode.add_argument("--seal-reviewed", action="store_true")
    mode.add_argument("--audit-pending", action="store_true")
    mode.add_argument("--audit-reviewed", action="store_true")
    mode.add_argument(
        "--raw-reaudit",
        action="store_true",
        help="Recompute from the complete canonical master; requires a reviewed seal.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.build_pending:
            result = build_pending_acoustic_scene(args.scene)
            state = "review_pending"
            paths = result.paths
        elif args.seal_reviewed:
            result = seal_reviewed_acoustic_scene(args.scene)
            state = "accepted"
            paths = result.pending.paths
        elif args.audit_pending:
            result = load_pending_acoustic_scene(args.scene)
            state = "review_pending"
            paths = result.paths
        elif args.audit_reviewed:
            result = load_reviewed_acoustic_scene(args.scene)
            state = "accepted"
            paths = result.pending.paths
        else:
            result = reaudit_formal_acoustic_scene(args.scene)
            state = "accepted_raw_reaudited"
            paths = result.pending.paths
    except (FormalAcousticFeatureError, OSError, ValueError) as exc:
        raise SystemExit(f"formal acoustic operation failed: {exc}") from exc

    payload = {
        "scene": args.scene,
        "review_status": state,
        "n_rows": len(
            result.pending.feature_df
            if hasattr(result, "pending")
            else result.feature_df
        ),
        "source_csv": str(paths.source_csv),
        "evidence_json": str(paths.evidence_json),
        "feature_csv": str(paths.feature_csv),
        "pending_manifest": str(paths.pending_manifest),
        "review_receipt": str(paths.review_receipt),
        "feature_manifest": str(paths.feature_manifest),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
