"""Create, audit, or finalize a non-destructive Transcript review artifact."""

from __future__ import annotations

import argparse
import json

from hri_turn_taking.data.transcript_bridge import (
    load_transcript_bridge_bundle,
    publish_transcript_bridge_bundle,
)
from hri_turn_taking.data.transcript_review import (
    FORMAL_SCENES,
    canonical_final_review_path,
    canonical_initial_draft_path,
    finalize_review,
    load_final_review,
    load_review_draft,
    review_progress,
    write_review_draft,
)
from hri_turn_taking.data.transcript_review_ui import (
    launch_review_gui,
    preflight_review_audio,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hash-bound manual review of coarse Transcript events. This tool never "
            "edits corpus sources and never produces counted gold VAD."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialise = subparsers.add_parser(
        "init", help="create a gold-window-scoped draft queue plus excluded inventory"
    )
    initialise.add_argument("--scene", required=True, choices=FORMAL_SCENES)
    initialise.add_argument(
        "--output",
        help="optional compatibility argument; must equal the canonical scene path",
    )

    audit = subparsers.add_parser("audit", help="replay a draft and print progress")
    audit.add_argument("draft")

    audit_final = subparsers.add_parser(
        "audit-final", help="fully replay an already finalized coarse review"
    )
    audit_final.add_argument("final")

    gui = subparsers.add_parser(
        "gui",
        help=(
            "review events with correction-mapped A/B/C/R audio; saves only new "
            "draft revisions"
        ),
    )
    gui.add_argument("draft")

    finalise = subparsers.add_parser(
        "finalize", help="publish a final coarse review only after every decision"
    )
    finalise.add_argument("draft")
    finalise.add_argument(
        "--output",
        help="optional compatibility argument; must equal the canonical scene path",
    )
    finalise.add_argument("--reviewer", required=True)
    finalise.add_argument("--reviewed-at", required=True)
    finalise.add_argument(
        "--method",
        default="manual listening to all four correction-mapped lavalier channels",
    )

    bridge = subparsers.add_parser(
        "bridge",
        help="publish a confirmed review-v3 final as acceptance-v3 plus coarse 25-Hz raster",
    )
    bridge.add_argument("final")
    bridge.add_argument(
        "--output-dir",
        help="optional compatibility argument; must equal the canonical scene path",
    )

    audit_bridge = subparsers.add_parser(
        "audit-bridge",
        help="fully replay a final→acceptance→coarse-raster bundle",
    )
    audit_bridge.add_argument("bundle")

    args = parser.parse_args()
    if args.command == "init":
        preflight_review_audio(args.scene)
        path = write_review_draft(
            args.scene, args.output or canonical_initial_draft_path(args.scene)
        )
        print(path)
        return
    if args.command == "audit":
        print(json.dumps(review_progress(load_review_draft(args.draft)), indent=2))
        return
    if args.command == "audit-final":
        final = load_final_review(args.final)
        print(
            json.dumps(
                {
                    "scene": final["scene"],
                    "output_role": final["output_role"],
                    "review_status": final["review_status"],
                    "counted_gold_ready": final["counted_gold_ready"],
                    "reviewer": final["manual_review"]["reviewer"],
                    "reviewed_at": final["manual_review"]["reviewed_at"],
                    "source_event_count": final["scope"]["source_event_count"],
                    "in_scope_event_count": final["manual_review"]["event_summary"][
                        "n_events"
                    ],
                    "excluded_event_count": final["scope"]["excluded_event_count"],
                    "crossing_event_count": final["scope"]["crossing_event_count"],
                },
                indent=2,
            )
        )
        return
    if args.command == "gui":
        launch_review_gui(args.draft)
        return
    if args.command == "bridge":
        print(publish_transcript_bridge_bundle(args.final, args.output_dir))
        return
    if args.command == "audit-bridge":
        bundle = load_transcript_bridge_bundle(args.bundle)
        print(
            json.dumps(
                {
                    "scene": bundle.scene,
                    "output_role": bundle.acceptance.output_role,
                    "review_status": bundle.receipt["review_status"],
                    "counted_gold_ready": bundle.acceptance.counted_gold_ready,
                    "scope": dict(bundle.acceptance.scope_summary),
                    "coarse_activity_rows": bundle.receipt["coarse_activity"][
                        "n_rows"
                    ],
                    "review_audit": bundle.receipt["review_audit"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    draft = load_review_draft(args.draft)
    path = finalize_review(
        args.draft,
        args.output or canonical_final_review_path(draft["scene"]),
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        method=args.method,
    )
    print(path)


if __name__ == "__main__":
    main()
