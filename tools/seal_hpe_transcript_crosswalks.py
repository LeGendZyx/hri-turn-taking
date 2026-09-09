"""Strictly re-audit and seal finalized HPE/Transcript crosswalk artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hri_turn_taking.data.hpe_transcript_crosswalk import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    DEFAULT_SEALED_LEDGER_PATH,
    HPETranscriptCrosswalkError,
    build_sealed_crosswalk_inventory,
    sha256_file,
    write_sealed_crosswalk_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-replay every accepted finalized crosswalk, then write a "
            "version-controlled exact-hash inventory and immutable ledger copy."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/annotations/hpe_transcript_crosswalk"),
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--ledger-snapshot",
        type=Path,
        default=DEFAULT_SEALED_LEDGER_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/hpe_transcript_crosswalk_inventory.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace both existing seal outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    try:
        inventory = build_sealed_crosswalk_inventory(
            repository_root=root,
            artifact_root=args.artifact_root,
            ledger_path=args.ledger,
            sealed_ledger_path=args.ledger_snapshot,
        )
        inventory_path = (
            args.output.resolve()
            if args.output.is_absolute()
            else (root / args.output).resolve()
        )
        written_inventory, written_ledger = write_sealed_crosswalk_bundle(
            inventory,
            inventory_path,
            repository_root=root,
            overwrite=args.overwrite,
        )
    except HPETranscriptCrosswalkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "inventory": str(written_inventory),
                "inventory_sha256": sha256_file(written_inventory),
                "ledger_snapshot": str(written_ledger),
                "ledger_snapshot_sha256": sha256_file(written_ledger),
                "scenes": sorted(inventory["scenes"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
