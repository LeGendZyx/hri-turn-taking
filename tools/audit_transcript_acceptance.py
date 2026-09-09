"""Read-only audit of a versioned Transcript activity-reference manifest."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal

from hri_turn_taking.data.transcript_acceptance import (
    SPEAKERS,
    TIMECODE_FPS_NUMERATOR,
    load_transcript_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a hash-bound Transcript acceptance manifest without modifying "
            "any upstream or derived file."
        )
    )
    parser.add_argument("manifest", help="Path to a transcript activity-reference manifest")
    args = parser.parse_args()

    accepted = load_transcript_acceptance(args.manifest)
    authorized_start = Decimal(accepted.authorized_start_tick) / 25
    authorized_end = Decimal(accepted.authorized_end_tick) / 25
    union_seconds = {
        speaker: str(
            sum(
                (
                    max(
                        Decimal(0),
                        min(
                            Decimal(end) / TIMECODE_FPS_NUMERATOR
                            + accepted.transcript_offset_sec,
                            authorized_end,
                        )
                        - max(
                            Decimal(start) / TIMECODE_FPS_NUMERATOR
                            + accepted.transcript_offset_sec,
                            authorized_start,
                        ),
                    )
                    for start, end in intervals
                ),
                start=Decimal(0),
            )
        )
        for speaker, intervals in accepted.speaker_intervals.items()
    }
    event_counts = {
        speaker: sum(event.speaker == speaker for event in accepted.events)
        for speaker in SPEAKERS
    }
    print(
        json.dumps(
            {
                "scene": accepted.scene,
                "output_role": accepted.output_role,
                "counted_gold_ready": accepted.counted_gold_ready,
                "review_status": "confirmed",
                "n_events": len(accepted.events),
                "scope": dict(accepted.scope_summary),
                "event_counts": event_counts,
                "union_activity_seconds": union_seconds,
                "transcript_offset_sec": str(accepted.transcript_offset_sec),
                "timecode_quantization_error_bound_ms": str(
                    accepted.timecode_quantization_error_bound_ms
                ),
                "offset_lag_quantization_error_bound_ms": str(
                    accepted.offset_lag_quantization_error_bound_ms
                ),
                "transcript_side_error_bound_ms": str(
                    accepted.transcript_side_error_bound_ms
                ),
                "hpe_stream": accepted.hpe_stream,
                "hpe_offset_sec": str(accepted.hpe_offset_sec),
                "hpe_error_bound_ms": str(accepted.hpe_error_bound_ms),
                "combined_error_bound_ms": str(accepted.combined_error_bound_ms),
                "hpe_quality_status": accepted.hpe_quality_status,
                "source_normalizations": list(accepted.source_normalizations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
