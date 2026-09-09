"""Real-data intake: HPE delivery directory -> validated 25 fps wide table.

Consumes one per-scenario delivery directory from Head_pose_extraction
(``outputs/delivery/<group>__<scenario>/`` — COORDINATION.md D1–D3 contract):

- ``world_persons.csv`` — leveled (+Y-up) world positions + facing unit vectors,
  on the local avi frame axis (``timestamp_sec``).
- ``participant_map.csv`` — ``person,participant`` rows (labels A/B/C + R).
- ``mapping.json`` — mapping evidence + ``review_status`` (proposed/confirmed).
- ``offset.json`` — ``{"streams": {<stream>: seconds}}``; shared mic-A time =
  local time + offset (D3).

Gates are fail-closed: unconfirmed mappings, missing files/streams, label-set
violations and unmapped persons all raise ``DeliveryGateError`` instead of
producing a silently misaligned table.

Slot convention (AGENTS.md §7, 2026-07-10): P1 = R (robot-player /
demonstrator), P2 = A, P3 = B, P4 = C; 3-person groups pad slot 4.

The emitted table contains every column ``vap-train`` expects, but only the
head-pose-derived features are real at this stage; all other columns are
zero-filled placeholders and are listed as such in the provenance report —
the table is NOT trainable until gold VAD (and the remaining modalities) are
merged in. ``doa_angle_deg`` is a permanent constant on real data (no mic
array was recorded; user decision 2026-07-10 — the column is kept only so the
synthetic-schema layout is preserved, and a constant is inert after
normalisation).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data.hpe_transcript_crosswalk import (
    DEFAULT_SEALED_INVENTORY_PATH,
    SEALED_DOWNSTREAM_KIND,
    SEALED_DOWNSTREAM_VERSION,
    HPETranscriptCrosswalkError,
    load_sealed_crosswalk_contract,
)
from hri_turn_taking.data.multiparty import (
    GLOBAL_FEATURE_COLUMNS,
    MAX_PARTICIPANTS,
    PARTICIPANT_FEATURE_NAMES,
    participant_column,
    validate_multiparty_dataframe,
)

REAL_PROVENANCE_FORMAT_VERSION = 2
REAL_INTAKE_ARTIFACT_KIND = "vap_real_pose_intake"
HPE_HEAD_POSE_STREAM_KEY = "6cam"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SLOT_BY_PARTICIPANT = {"R": 1, "A": 2, "B": 3, "C": 4}
VALID_PARTICIPANTS = set(SLOT_BY_PARTICIPANT)

WORLD_PERSONS_REQUIRED = [
    "frame_index",
    "timestamp_sec",
    "x_w",
    "y_w",
    "z_w",
    "facing_x",
    "facing_y",
    "facing_z",
    "facing_src_abs_yaw",
    "person",
]

#: features this intake stage can derive from world_persons.csv alone
REAL_PARTICIPANT_FEATURES = ["head_yaw_deg", "pose_valid", "pose_quality"]

#: the only participant-label sets a real delivery may use (D2; Codex R1.1-4)
VALID_LABEL_SETS = ({"R", "A", "B"}, {"R", "A", "B", "C"})

#: mapping.json fields that must bind a confirmed mapping to THIS delivery
REQUIRED_BINDING_FIELDS = ["delivery", "group", "scenario", "participant_map"]
_NORMALISED_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DeliveryGateError(ValueError):
    """A delivery directory failed a fail-closed intake gate."""


@dataclass(frozen=True)
class IntakeConfig:
    stream_key: str = HPE_HEAD_POSE_STREAM_KEY
    frame_rate_hz: int = 25
    max_gap_s: float = 0.2  # grid frames farther than this from a real sample are invalid
    allow_unconfirmed: bool = False
    # production deliveries must ship a complete offset_quality.json (Codex R1 ③);
    # relaxing this is debug-only, like allow_unconfirmed
    allow_missing_offset_quality: bool = False
    # two fragments of the same participant sampled at the same instant must agree
    # on position within this tolerance (HPE identity merge_dist_m), else the
    # mapping is contradictory (Codex R1 ①). Real deliveries DO contain such
    # overlaps (multi-camera ghost tracklets), so the resolution is explicit:
    # "reject" (default, fail-closed) or "dominant" (keep the fragment with the
    # most support rows at each conflicting instant; recorded in provenance)
    fragment_conflict_m: float = 0.35
    # simultaneous fragments that agree on position can still disagree on the
    # facing direction we consume (Codex R1.4-3); component-vector spread of
    # 0.5 on unit vectors ≈ 29° divergence
    fragment_facing_conflict: float = 0.5
    fragment_overlap_policy: str = "reject"  # "reject" | "dominant"


DEFAULT_INTAKE_CONFIG = IntakeConfig()


@dataclass
class Delivery:
    directory: Path
    group: str
    scenario: str
    world_persons: pd.DataFrame
    # This is the HPE-owned participant_map.csv proposal.  It remains bound as
    # provenance, but it is never the semantic identity authority for VAP v2.
    participant_map: dict[str, str]
    review_status: str
    offsets: dict[str, float]
    unmapped_persons: list[str]  # transient tracks the mapping deliberately skips (议题 5②)
    offset_quality: dict | None  # offset_quality.json payload, if delivered
    mapping_meta: dict = field(default_factory=dict)  # full mapping.json payload

    @property
    def session_id(self) -> str:
        return f"{self.group}__{self.scenario}"


@dataclass
class IntakeResult:
    frame_df: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_no_clobber(staged: Path, destination: Path) -> None:
    """Atomically publish one staged file without replacing an existing path."""

    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise DeliveryGateError(
            f"Intake output appeared while materialisation was running: {destination}"
        ) from exc
    except OSError as exc:
        raise DeliveryGateError(
            f"Could not publish intake output without overwriting: {destination}: {exc}"
        ) from exc


def _reserve_staged_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    """Create a writer-private staging inode in the destination directory."""

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=prefix,
        suffix=suffix,
        dir=directory,
        delete=False,
    ) as handle:
        return Path(handle.name)


_DELIVERY_SOURCE_NAMES = {
    "world_persons": "world_persons.csv",
    "participant_map": "participant_map.csv",
    "mapping": "mapping.json",
    "offset": "offset.json",
    "offset_quality": "offset_quality.json",
}


def _delivery_source_paths(directory: Path) -> dict[str, Path]:
    return {
        name: directory / filename for name, filename in _DELIVERY_SOURCE_NAMES.items()
    }


def _snapshot_delivery_sources(directory: Path) -> dict[str, dict[str, object]]:
    """Bind one immutable view of every upstream file consumed by intake."""
    paths = _delivery_source_paths(directory)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DeliveryGateError(
            f"Delivery is missing required production source files: {missing}"
        )
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def _canonical_hpe_person(value: object, *, context: str) -> str:
    """Canonicalise the sealed integer HPE ``person`` identifier."""
    if isinstance(value, bool) or pd.isna(value):
        raise DeliveryGateError(f"{context} contains a non-integer HPE person {value!r}")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        integer = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        integer = int(value)
    else:
        raise DeliveryGateError(f"{context} contains a non-integer HPE person {value!r}")
    if integer < 1:
        raise DeliveryGateError(f"{context} contains invalid HPE person {integer}")
    return str(integer)


def _normalise_crosswalk_contract(
    contract: dict[str, Any], *, delivery: Delivery
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate the downstream contract shape needed by deterministic intake."""
    if not isinstance(contract, dict):
        raise DeliveryGateError("Sealed HPE→Transcript crosswalk contract is not an object")
    if contract.get("artifact_kind") != SEALED_DOWNSTREAM_KIND:
        raise DeliveryGateError("Sealed HPE→Transcript crosswalk has the wrong artifact kind")
    contract_version = contract.get("format_version")
    if type(contract_version) is not int or contract_version != SEALED_DOWNSTREAM_VERSION:
        raise DeliveryGateError("Unsupported sealed HPE→Transcript crosswalk version")
    if contract.get("scene") != delivery.session_id:
        raise DeliveryGateError(
            f"Sealed crosswalk scene {contract.get('scene')!r} does not match delivery "
            f"{delivery.session_id!r}"
        )

    window = contract.get("accepted_window")
    required_window = {
        "scene",
        "camera",
        "local_start_inclusive",
        "local_end_exclusive",
        "logical_start_inclusive",
        "logical_end_exclusive",
        "filter_rule",
    }
    if not isinstance(window, dict) or set(window) != required_window:
        raise DeliveryGateError("Sealed crosswalk accepted_window is malformed")
    start = window.get("logical_start_inclusive")
    end = window.get("logical_end_exclusive")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or window.get("scene") != delivery.session_id
        or window.get("filter_rule") != "start <= frame_index < end"
    ):
        raise DeliveryGateError("Sealed crosswalk logical frame window is invalid")
    source_fps = contract.get("source_fps")
    if (
        isinstance(source_fps, bool)
        or not isinstance(source_fps, (int, float))
        or not math.isfinite(float(source_fps))
        or float(source_fps) <= 0.0
    ):
        raise DeliveryGateError("Sealed crosswalk source_fps must be a finite positive number")
    timing = contract.get("hpe_logical_timing_authority")
    timing_fps = timing.get("source_fps") if isinstance(timing, dict) else None
    if (
        not isinstance(timing, dict)
        or set(timing) != {
            "axis",
            "source_fps",
            "evidence_cameras",
            "validation_complete",
        }
        or timing.get("axis") != "hpe_delivery_logical_frame_index"
        or isinstance(timing_fps, bool)
        or not isinstance(timing_fps, (int, float))
        or not math.isfinite(float(timing_fps))
        or not math.isclose(
            float(timing_fps),
            float(source_fps),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or timing.get("validation_complete") != contract.get("hpe_validation_authority")
    ):
        raise DeliveryGateError("Sealed crosswalk HPE logical timing authority is malformed")
    evidence_cameras = timing.get("evidence_cameras")
    if not isinstance(evidence_cameras, list) or not evidence_cameras:
        raise DeliveryGateError("Sealed crosswalk has no HPE logical timing camera")
    for authority in evidence_cameras:
        authority_fps = (
            authority.get("source_fps") if isinstance(authority, dict) else None
        )
        if (
            not isinstance(authority, dict)
            or set(authority) != {"camera", "source_fps", "run_metadata"}
            or not isinstance(authority.get("camera"), str)
            or not re.fullmatch(r"Cam_[1-9][0-9]*", authority["camera"])
            or isinstance(authority_fps, bool)
            or not isinstance(authority_fps, (int, float))
            or not math.isfinite(float(authority_fps))
            or not math.isclose(
                float(authority_fps),
                float(source_fps),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise DeliveryGateError(
                "Sealed crosswalk HPE logical timing camera binding is malformed"
            )

    expected_raw = contract.get("expected_hpe_persons")
    if not isinstance(expected_raw, list) or not expected_raw:
        raise DeliveryGateError("Sealed crosswalk expected_hpe_persons is malformed")
    expected = [
        _canonical_hpe_person(person, context="expected_hpe_persons")
        for person in expected_raw
    ]
    if expected != sorted(set(expected), key=int):
        raise DeliveryGateError("Sealed crosswalk expected_hpe_persons must be unique and sorted")

    records = contract.get("person_mapping")
    if not isinstance(records, list) or not records:
        raise DeliveryGateError("Sealed crosswalk person_mapping is malformed")
    mapping: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "hpe_person",
            "participant",
            "slot",
        }:
            raise DeliveryGateError("Sealed crosswalk person_mapping record is malformed")
        person = _canonical_hpe_person(
            record["hpe_person"], context="sealed person_mapping"
        )
        participant = record["participant"]
        if participant not in VALID_PARTICIPANTS:
            raise DeliveryGateError(
                f"Sealed crosswalk maps person {person} to invalid participant "
                f"{participant!r}"
            )
        expected_slot = f"P{SLOT_BY_PARTICIPANT[participant]}"
        if record["slot"] != expected_slot:
            raise DeliveryGateError(
                f"Sealed crosswalk slot for person {person} must be {expected_slot}"
            )
        if person in mapping:
            raise DeliveryGateError(f"Sealed crosswalk repeats HPE person {person}")
        mapping[person] = participant
    if sorted(mapping, key=int) != expected:
        raise DeliveryGateError(
            "Sealed crosswalk person_mapping does not exactly cover expected_hpe_persons"
        )
    if set(mapping.values()) not in VALID_LABEL_SETS:
        raise DeliveryGateError(
            f"Sealed crosswalk participant set {sorted(set(mapping.values()))} is not legal"
        )

    proposal = contract.get("upstream_proposal")
    if not isinstance(proposal, dict):
        raise DeliveryGateError("Sealed crosswalk upstream_proposal must be an object")
    normalised_proposal = {
        _canonical_hpe_person(person, context="crosswalk upstream_proposal"): str(label)
        .strip()
        .upper()
        for person, label in proposal.items()
    }
    if normalised_proposal != delivery.participant_map:
        raise DeliveryGateError(
            "Current HPE participant_map.csv differs from the proposal sealed with the "
            "human HPE→Transcript crosswalk"
        )
    return mapping, window


def _validate_contract_source_bindings(
    contract: dict[str, Any],
    *,
    delivery_sources: dict[str, dict[str, object]],
    repository_root: Path,
) -> None:
    """Recheck the sealed authority and its small files after pose materialisation."""
    world_binding = contract.get("delivery_world_persons")
    if (
        not isinstance(world_binding, dict)
        or world_binding.get("path") != delivery_sources["world_persons"]["path"]
        or world_binding.get("sha256")
        != delivery_sources["world_persons"]["sha256"]
    ):
        raise DeliveryGateError(
            "Sealed crosswalk is not bound to this delivery's world_persons.csv"
        )
    inventory = contract.get("inventory")
    if not isinstance(inventory, dict):
        raise DeliveryGateError("Sealed crosswalk inventory binding is malformed")
    inventory_path = Path(str(inventory.get("path", ""))).resolve()
    if not inventory_path.is_file() or _sha256(inventory_path) != inventory.get("sha256"):
        raise DeliveryGateError("Sealed crosswalk inventory changed during intake")

    def verify_binding(binding: object, label: str) -> None:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise DeliveryGateError(f"Sealed crosswalk {label} binding is malformed")
        path = Path(str(binding["path"])).resolve()
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise DeliveryGateError(f"Sealed crosswalk {label} changed during intake")

    verify_binding(contract.get("hpe_validation_authority"), "HPE validation authority")
    timing = contract["hpe_logical_timing_authority"]
    verify_binding(timing["validation_complete"], "HPE timing validation authority")
    for authority in timing["evidence_cameras"]:
        verify_binding(
            authority["run_metadata"],
            f"{authority['camera']} run-metadata timing authority",
        )

    artifact_dir_value = contract.get("artifact_dir")
    files = contract.get("artifact_files")
    if not isinstance(artifact_dir_value, str) or not isinstance(files, dict):
        raise DeliveryGateError("Sealed crosswalk artifact bindings are malformed")
    artifact_dir = (repository_root / artifact_dir_value).resolve()
    try:
        artifact_dir.relative_to(repository_root)
    except ValueError as exc:
        raise DeliveryGateError(
            "Sealed crosswalk artifact directory escapes the repository"
        ) from exc
    for name, binding in files.items():
        path = artifact_dir / name
        if (
            not isinstance(binding, dict)
            or not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            raise DeliveryGateError(f"Sealed crosswalk file changed during intake: {name}")


def _formal_intake_config(config: IntakeConfig) -> dict[str, object]:
    return {
        "stream_key": config.stream_key,
        "frame_rate_hz": config.frame_rate_hz,
        "max_gap_s": float(config.max_gap_s),
        "fragment_conflict_m": float(config.fragment_conflict_m),
        "fragment_facing_conflict": float(config.fragment_facing_conflict),
        "fragment_overlap_policy": config.fragment_overlap_policy,
    }


def _accepted_output_grid(
    window: dict[str, Any], *, source_fps: float, output_fps: int, offset_s: float
) -> tuple[np.ndarray, int, int]:
    """Return ticks on mic-A time whose source time lies in the half-open window."""
    start_shared = float(window["logical_start_inclusive"]) / source_fps + offset_s
    end_shared = float(window["logical_end_exclusive"]) / source_fps + offset_s

    def first_tick_at_or_after(boundary: float) -> int:
        """Smallest integer tick whose represented time is >= ``boundary``.

        Multiplying by the output rate and subtracting an arbitrary epsilon can
        move a genuinely post-boundary value onto the previous tick.  Start from
        ``floor`` only as a candidate, then compare the represented tick time
        directly against the final floating-point boundary contract.
        """

        tick = math.floor(boundary * output_fps)
        while tick / output_fps < boundary:
            tick += 1
        while (tick - 1) / output_fps >= boundary:
            tick -= 1
        return int(tick)

    # A tick is included iff start_shared <= tick/output_fps < end_shared;
    # therefore both limits use the first representable tick at or after their
    # respective boundary, with the latter retained as an exclusive index.
    start_tick = first_tick_at_or_after(start_shared)
    end_tick_exclusive = first_tick_at_or_after(end_shared)
    if end_tick_exclusive <= start_tick:
        raise DeliveryGateError("Accepted crosswalk window spans no 25 Hz output frames")
    ticks = np.arange(start_tick, end_tick_exclusive, dtype=np.int64)
    return ticks.astype(float) / output_fps, start_tick, end_tick_exclusive


def _require_normalised_delivery_id(value: str, field_name: str) -> str:
    if value != value.strip() or not _NORMALISED_ID_RE.fullmatch(value):
        raise DeliveryGateError(
            f"Delivery {field_name} {value!r} must be a non-empty normalised identifier "
            "with no leading/trailing whitespace"
        )
    return value


def load_delivery(delivery_dir: str | Path) -> Delivery:
    directory = Path(delivery_dir)
    if not directory.is_dir():
        raise DeliveryGateError(f"Delivery directory not found: {directory}")
    name = directory.name
    if "__" not in name:
        raise DeliveryGateError(
            f"Delivery directory name {name!r} is not <group>__<scenario> — "
            "group membership must be recoverable for person-disjoint splits (D2)"
        )
    group, scenario = name.split("__", 1)
    _require_normalised_delivery_id(group, "group")
    _require_normalised_delivery_id(scenario, "scenario")

    paths = {
        "world_persons": directory / "world_persons.csv",
        "participant_map": directory / "participant_map.csv",
        "mapping": directory / "mapping.json",
        "offset": directory / "offset.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise DeliveryGateError(f"Delivery is missing required files: {missing}")

    world_persons = pd.read_csv(paths["world_persons"])
    absent = [c for c in WORLD_PERSONS_REQUIRED if c not in world_persons.columns]
    if absent:
        raise DeliveryGateError(
            f"world_persons.csv is missing contract columns {absent} — "
            "column renames must be announced in COORDINATION.md first (D4)"
        )

    map_df = pd.read_csv(paths["participant_map"])
    if not {"person", "participant"}.issubset(map_df.columns):
        raise DeliveryGateError("participant_map.csv must have 'person,participant' columns")
    map_df = map_df.assign(
        person=map_df["person"].astype(str),
        participant=map_df["participant"].astype(str).str.strip().str.upper(),
    )
    conflicts = (
        map_df.drop_duplicates().groupby("person")["participant"].nunique()
    )
    conflicted = sorted(conflicts[conflicts > 1].index)
    if conflicted:
        raise DeliveryGateError(
            f"participant_map.csv maps persons {conflicted} to more than one participant — "
            "the mapping is self-contradictory"
        )
    duplicated = sorted(
        map_df.loc[map_df["person"].duplicated(keep=False), "person"].unique().tolist()
    )
    if duplicated:
        raise DeliveryGateError(
            f"participant_map.csv contains duplicate person rows {duplicated}; each "
            "upstream person must have exactly one row"
        )
    if map_df["person"].map(lambda value: not value or value != value.strip()).any():
        raise DeliveryGateError(
            "participant_map.csv person identifiers must be non-empty and trimmed"
        )
    participant_map = dict(zip(map_df["person"], map_df["participant"], strict=True))

    mapping_meta = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    review_status = str(mapping_meta.get("review_status", "missing"))
    # unmapped_persons must be an ARRAY of scalar person ids (Codex R1.2-3): a
    # bare string "p1" would iterate to ['p', '1'] and silently dodge the
    # mapped/unmapped intersection check
    raw_unmapped = mapping_meta.get("unmapped_persons", [])
    if not isinstance(raw_unmapped, list) or any(
        isinstance(p, (dict, list)) or isinstance(p, bool) for p in raw_unmapped
    ):
        raise DeliveryGateError(
            "mapping.json unmapped_persons must be a JSON array of scalar person ids "
            f"(got {type(raw_unmapped).__name__})"
        )
    unmapped_persons = [str(p) for p in raw_unmapped]

    offset_payload = json.loads(paths["offset"].read_text(encoding="utf-8"))
    streams = offset_payload.get("streams")
    if not isinstance(streams, dict):
        raise DeliveryGateError("offset.json must use the D3 {'streams': {...}} format")
    # offset values must be JSON numbers BEFORE conversion (Codex R1.3-3):
    # float("-1.2") would silently accept a string offset, and bool is not a time
    for key, value in streams.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(
            float(value)
        ):
            raise DeliveryGateError(
                f"offset.json stream {key!r} offset must be a finite JSON number "
                f"(got {type(value).__name__} {value!r})"
            )
    offsets = {str(k): float(v) for k, v in streams.items()}

    quality_path = directory / "offset_quality.json"
    offset_quality = (
        json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    )

    return Delivery(
        directory=directory,
        group=group,
        scenario=scenario,
        world_persons=world_persons,
        participant_map=participant_map,
        review_status=review_status,
        offsets=offsets,
        unmapped_persons=unmapped_persons,
        offset_quality=offset_quality,
        mapping_meta=mapping_meta,
    )


def validate_delivery(delivery: Delivery, config: IntakeConfig) -> None:
    if delivery.review_status != "confirmed":
        raise DeliveryGateError(
            f"mapping.json review_status={delivery.review_status!r}; the sealed crosswalk "
            "supplies semantic identity, but formal intake still requires a confirmed "
            "HPE delivery provenance chain"
        )

    # HPE's map remains only the semantic proposal; nevertheless its source
    # contract must be a structurally legal D2 delivery before the independent
    # sealed crosswalk may supersede individual label assignments.
    labels = set(delivery.participant_map.values())
    if labels not in VALID_LABEL_SETS:
        raise DeliveryGateError(
            f"Upstream HPE proposal label set {sorted(labels)} is not a legal D2 set "
            f"({[sorted(item) for item in VALID_LABEL_SETS]})"
        )

    # Bind the HPE proposal JSON and CSV to the same concrete delivery.
    meta = delivery.mapping_meta
    missing_fields = [k for k in REQUIRED_BINDING_FIELDS if k not in meta]
    if missing_fields:
        raise DeliveryGateError(
            f"mapping.json is missing required binding fields {missing_fields} — "
            "the upstream proposal must be bound to a specific delivery"
        )
    if str(meta["delivery"]) != delivery.directory.name:
        raise DeliveryGateError(
            f"mapping.json declares delivery {meta['delivery']!r} but the directory is "
            f"{delivery.directory.name!r} — proposal provenance does not transfer"
        )
    for key, expected in (("group", delivery.group), ("scenario", delivery.scenario)):
        if str(meta[key]) != expected:
            raise DeliveryGateError(
                f"mapping.json {key}={meta[key]!r} disagrees with directory name "
                f"({expected!r}) — proposal provenance does not transfer"
            )
    if not isinstance(meta["participant_map"], dict):
        raise DeliveryGateError(
            "mapping.json participant_map must be a JSON object of person -> participant "
            f"(got {type(meta['participant_map']).__name__})"
        )
    normalised = {str(k): str(v).strip().upper() for k, v in meta["participant_map"].items()}
    if normalised != delivery.participant_map:
        raise DeliveryGateError(
            "mapping.json participant_map disagrees with participant_map.csv — "
            "the reviewed mapping does not cover this CSV"
        )
    overlap = set(delivery.participant_map) & set(delivery.unmapped_persons)
    if overlap:
        raise DeliveryGateError(
            f"persons {sorted(overlap)} are both mapped and listed in unmapped_persons — "
            "the mapping is self-contradictory"
        )

    csv_persons = {str(person) for person in delivery.world_persons["person"].dropna().unique()}
    proposed = set(delivery.participant_map)
    undeclared = csv_persons - proposed - set(delivery.unmapped_persons)
    if undeclared:
        raise DeliveryGateError(
            f"world_persons.csv contains persons {sorted(undeclared)} absent from both the "
            "HPE proposal and mapping.json.unmapped_persons"
        )
    if proposed - csv_persons:
        raise DeliveryGateError(
            f"participant_map.csv proposes persons {sorted(proposed - csv_persons)} that "
            "never appear in world_persons.csv"
        )

    if config.stream_key not in delivery.offsets:
        raise DeliveryGateError(
            f"offset.json has no stream {config.stream_key!r} (streams: "
            f"{sorted(delivery.offsets)}) — cannot place world_persons on the mic A axis (D3)"
        )

    # 议题 5① / Codex R1 ③: offset_quality.json is required and fully asserted,
    # not best-effort — a missing or incomplete quality report must not pass
    if delivery.offset_quality is None:
        if not config.allow_missing_offset_quality:
            raise DeliveryGateError(
                "Delivery has no offset_quality.json — production intake requires the "
                "D3 quality report; pass allow_missing_offset_quality only for debugging"
            )
    else:
        _validate_offset_quality(delivery, config)



#: exact D3 sign convention: shared time = stream-local time + offset (议题 5①);
#: whitespace/`_s`-suffix tolerant, but the DIRECTION must match — the reverse
#: formula ("stream_local = shared + offset") is a sign error, not a variant
_CONVENTION_RE = re.compile(
    r"^shared_?time=stream_?local_?time\+offset(_s)?$"
)


def _require_finite_number(value: object, name: str) -> float:
    """JSON-type-strict number gate (Codex R1.2-2): booleans and numeric
    strings are type errors, not values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeliveryGateError(
            f"offset_quality.json {name} must be a JSON number (got {type(value).__name__})"
        )
    if not np.isfinite(float(value)):
        raise DeliveryGateError(f"offset_quality.json {name} is not finite: {value!r}")
    return float(value)


def _validate_offset_quality(delivery: Delivery, config: IntakeConfig) -> None:
    """Every field is REQUIRED with its exact JSON type — a partial or
    type-sloppy quality report must not pass (R1.1-2 / R1.2-2)."""
    quality_doc = delivery.offset_quality or {}
    convention = quality_doc.get("convention")
    if convention is None or not _CONVENTION_RE.fullmatch(str(convention).replace(" ", "")):
        raise DeliveryGateError(
            f"offset_quality.json convention {convention!r} missing or does not match "
            "'shared_time = stream_local_time + offset_s' exactly (议题 5①)"
        )
    # timebase_keys is the 议题 5① object mapping physical stream names to
    # timebase descriptions — the stream must be a KEY; a value match would let
    # a document without a real 6cam entry pass (Codex R1.2-2)
    timebase_keys = quality_doc.get("timebase_keys")
    if not isinstance(timebase_keys, dict):
        raise DeliveryGateError(
            "offset_quality.json timebase_keys must be a JSON object keyed by stream "
            f"name (议题 5①; got {type(timebase_keys).__name__})"
        )
    if config.stream_key not in {str(k) for k in timebase_keys}:
        raise DeliveryGateError(
            f"offset_quality.json timebase_keys {sorted(map(str, timebase_keys))} do not "
            f"include {config.stream_key!r} as a key"
        )
    stream_quality = quality_doc.get("quality", {}).get(config.stream_key)
    if not isinstance(stream_quality, dict):
        raise DeliveryGateError(
            f"offset_quality.json has no quality entry for {config.stream_key!r}"
        )
    required = ["offset_s", "self_check_ok", "error_bound_ms", "status"]
    missing = [k for k in required if k not in stream_quality]
    if missing:
        raise DeliveryGateError(
            f"offset_quality.json quality[{config.stream_key!r}] is missing required "
            f"fields {missing}"
        )
    offset_value = _require_finite_number(stream_quality["offset_s"], "offset_s")
    if abs(offset_value - delivery.offsets[config.stream_key]) > 1e-6:
        raise DeliveryGateError(
            f"offset_quality.json offset_s for {config.stream_key!r} disagrees with "
            "offset.json — final offsets must match (议题 5①b)"
        )
    self_check = stream_quality["self_check_ok"]
    if not isinstance(self_check, bool):
        raise DeliveryGateError(
            "offset_quality.json self_check_ok must be a JSON boolean "
            f"(got {type(self_check).__name__} {self_check!r} — the string 'false' is truthy)"
        )
    if not self_check:
        raise DeliveryGateError(
            f"offset_quality.json self_check_ok is false for {config.stream_key!r}"
        )
    bound = _require_finite_number(stream_quality["error_bound_ms"], "error_bound_ms")
    if bound < 0:
        raise DeliveryGateError(f"offset_quality.json error_bound_ms invalid: {bound!r}")
    if not isinstance(stream_quality["status"], str) or stream_quality["status"] not in {
        "ok",
        "degraded",
    }:
        raise DeliveryGateError(
            f"offset_quality.json status {stream_quality['status']!r} is not ok/degraded"
        )


def _validate_world_persons_values(kept: pd.DataFrame) -> None:
    """Reject non-finite samples and degenerate facing vectors on authority rows."""
    numeric = [
        "timestamp_sec",
        "x_w",
        "y_w",
        "z_w",
        "facing_x",
        "facing_y",
        "facing_z",
        "facing_src_abs_yaw",
    ]
    try:
        values = kept[numeric].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DeliveryGateError(
            "world_persons.csv authority pose columns must be numeric"
        ) from exc
    if not np.isfinite(values).all():
        bad = [
            column
            for column in numeric
            if not np.isfinite(kept[column].to_numpy(dtype=float)).all()
        ]
        raise DeliveryGateError(f"world_persons.csv has NaN/Inf in columns {bad}")
    horizontal = np.linalg.norm(kept[["facing_x", "facing_z"]].to_numpy(dtype=float), axis=1)
    if (horizontal < 1e-6).any():
        raise DeliveryGateError(
            "world_persons.csv contains facing vectors with no horizontal component — "
            "head yaw would be undefined"
        )


def _horizontal_bearing_deviation_deg(
    facing_xz: np.ndarray, to_target_xz: np.ndarray
) -> np.ndarray:
    """Angle (deg, [0, 180]) between horizontal facing and direction-to-target."""

    def _norm(v: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(v, axis=1, keepdims=True)
        length[length < 1e-9] = 1.0
        return v / length

    cos = np.clip((_norm(facing_xz) * _norm(to_target_xz)).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _validate_intake_config(config: IntakeConfig) -> None:
    """Fail-closed config gate (Codex R1.4-2): a NaN ``max_gap_s`` makes every
    gap comparison false and would silently re-enable unbounded staleness."""
    if config.stream_key != HPE_HEAD_POSE_STREAM_KEY:
        raise DeliveryGateError(
            "IntakeConfig.stream_key must be '6cam' for world_persons.csv; "
            "participants_rgb and robot_rgb offsets describe different physical "
            "timebases and cannot substitute for the HPE six-camera axis"
        )
    if config.allow_unconfirmed or config.allow_missing_offset_quality:
        raise DeliveryGateError(
            "Real intake provenance v2 has no debug identity/D3 bypass: "
            "allow_unconfirmed and allow_missing_offset_quality must both be false"
        )
    gap = config.max_gap_s
    if (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not np.isfinite(float(gap))
        or not 0.0 < float(gap) <= 1.0
    ):
        raise DeliveryGateError(
            f"IntakeConfig.max_gap_s must be a finite number in (0, 1.0] seconds "
            f"(got {gap!r}) — the approved perception staleness bound is 0.2 s"
        )
    if (
        isinstance(config.frame_rate_hz, bool)
        or not isinstance(config.frame_rate_hz, int)
        or config.frame_rate_hz != 25
    ):
        raise DeliveryGateError(
            f"IntakeConfig.frame_rate_hz must be the canonical integer 25 (got "
            f"{config.frame_rate_hz!r})"
        )
    for name in ("fragment_conflict_m", "fragment_facing_conflict"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise DeliveryGateError(f"IntakeConfig.{name} must be finite and positive")
    if config.fragment_overlap_policy not in {"reject", "dominant"}:
        raise DeliveryGateError(
            f"IntakeConfig.fragment_overlap_policy must be 'reject' or 'dominant' "
            f"(got {config.fragment_overlap_policy!r})"
        )


def run_intake(
    delivery_dir: str | Path,
    config: IntakeConfig = DEFAULT_INTAKE_CONFIG,
    *,
    crosswalk_inventory: str | Path = DEFAULT_SEALED_INVENTORY_PATH,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> IntakeResult:
    _validate_intake_config(config)
    root = Path(repository_root).resolve()
    directory = Path(delivery_dir).resolve()
    delivery_snapshot_before = _snapshot_delivery_sources(directory)
    delivery = load_delivery(delivery_dir)
    validate_delivery(delivery, config)
    try:
        contract = load_sealed_crosswalk_contract(
            delivery.session_id,
            crosswalk_inventory,
            repository_root=root,
        )
    except (HPETranscriptCrosswalkError, OSError, ValueError) as exc:
        raise DeliveryGateError(
            f"Sealed HPE→Transcript identity authority rejected {delivery.session_id}: {exc}"
        ) from exc
    person_to_participant, accepted_window = _normalise_crosswalk_contract(
        contract, delivery=delivery
    )
    _validate_contract_source_bindings(
        contract,
        delivery_sources=delivery_snapshot_before,
        repository_root=root,
    )

    fps = config.frame_rate_hz
    offset_s = delivery.offsets[config.stream_key]
    wp = delivery.world_persons.copy()
    try:
        raw_frame_index = pd.to_numeric(wp["frame_index"], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise DeliveryGateError("world_persons.csv frame_index must be numeric") from exc
    if not np.isfinite(raw_frame_index).all() or not np.equal(
        raw_frame_index, np.floor(raw_frame_index)
    ).all():
        raise DeliveryGateError("world_persons.csv frame_index must contain finite integers")
    wp["frame_index"] = raw_frame_index.astype(np.int64)
    logical_start = int(accepted_window["logical_start_inclusive"])
    logical_end = int(accepted_window["logical_end_exclusive"])
    within_window = (wp["frame_index"] >= logical_start) & (
        wp["frame_index"] < logical_end
    )
    n_outside_window_rows = int((~within_window).sum())
    wp = wp.loc[within_window].copy()
    if wp.empty:
        raise DeliveryGateError("world_persons.csv has no rows in the accepted frame window")
    wp["person"] = [
        _canonical_hpe_person(value, context="world_persons.csv")
        for value in wp["person"].tolist()
    ]
    actual_persons = sorted(wp["person"].unique().tolist(), key=int)
    expected_persons = sorted(person_to_participant, key=int)
    if actual_persons != expected_persons:
        raise DeliveryGateError(
            "world_persons.csv persons inside the accepted window differ from the sealed "
            f"crosswalk: actual={actual_persons}, expected={expected_persons}"
        )
    _validate_world_persons_values(wp)
    source_fps = float(contract["source_fps"])
    expected_source_t = wp["frame_index"].to_numpy(dtype=float) / source_fps
    actual_source_t = wp["timestamp_sec"].to_numpy(dtype=float)
    if not np.allclose(actual_source_t, expected_source_t, rtol=0.0, atol=1e-5):
        raise DeliveryGateError(
            "world_persons.csv timestamp_sec is not frame_index/source_fps inside the "
            "sealed accepted window"
        )
    # Once the delivered timestamp column is shown consistent, derive the
    # consumed time from the sealed logical-frame clock itself.  This avoids
    # six-decimal CSV rounding changing a past-only boundary decision.
    wp["shared_t"] = expected_source_t + offset_s

    # merge sequential person fragments into one series per participant label
    # (Codex R1 ① — real deliveries map several tracklets to the same A/B/C/R);
    # simultaneous samples from different fragments must agree on position AND
    # facing direction (Codex R1.4-3: position-agreeing ghost tracklets can
    # still disagree on facing, which is the feature we actually consume)
    wp["participant"] = wp["person"].map(person_to_participant)
    per_label: dict[str, pd.DataFrame] = {}
    fragments: dict[str, list[str]] = {}
    fragment_overlap: dict[str, dict] = {}
    for label, g in wp.groupby("participant"):
        fragments[label] = sorted(g["person"].unique())
        # at an overlapping instant the winner is chosen by PER-ROW local
        # quality only (lowest facing_src_abs_yaw, ties by person id) — a
        # fragment's future track length must never decide a past frame
        # (Codex R1.4-3; the earlier whole-session row-count support did)
        g = g.assign(_quality=g["facing_src_abs_yaw"].astype(float)).sort_values(
            ["shared_t", "_quality", "person"], ascending=[True, True, True]
        )
        dup = g.duplicated("shared_t", keep=False)
        if dup.any():
            grouped = g[dup].groupby("shared_t")
            pos_spread = grouped[["x_w", "z_w"]].agg(lambda s: s.max() - s.min())
            pos_dist = np.hypot(pos_spread["x_w"], pos_spread["z_w"])
            facing_spread = grouped[["facing_x", "facing_z"]].agg(lambda s: s.max() - s.min())
            facing_dist = np.hypot(facing_spread["facing_x"], facing_spread["facing_z"])
            n_pos = int((pos_dist > config.fragment_conflict_m).sum())
            n_facing = int((facing_dist > config.fragment_facing_conflict).sum())
            fragment_overlap[label] = {
                "overlapping_instants": int(len(pos_spread)),
                "conflicting_instants": n_pos,
                "max_spread_m": float(pos_dist.max()),
                "facing_conflicting_instants": n_facing,
                "max_facing_spread": float(facing_dist.max()),
            }
            if (n_pos or n_facing) and config.fragment_overlap_policy != "dominant":
                raise DeliveryGateError(
                    f"participant {label!r}: fragments {fragments[label]} give conflicting "
                    f"positions (> {config.fragment_conflict_m} m) at {n_pos} instants and "
                    f"conflicting facing directions (component spread > "
                    f"{config.fragment_facing_conflict}) at {n_facing} instants — the "
                    "mapping cannot be merged automatically; set "
                    "fragment_overlap_policy='dominant' only after reviewing the overlap"
                )
            g = g.drop_duplicates("shared_t", keep="first")
        per_label[label] = g.drop(columns="_quality")

    grid, start_tick, end_tick_exclusive = _accepted_output_grid(
        accepted_window,
        source_fps=source_fps,
        output_fps=fps,
        offset_s=offset_s,
    )

    n_grid = grid.shape[0]
    data: dict[str, np.ndarray] = {"timestamp_s": grid}
    missing_fraction: dict[str, float] = {}
    quality: dict[str, object] = {}

    # pass 1: per-label resampled series + own-sampling validity, STRICT
    # past-only (Codex R1.4-2): each grid frame holds the latest source sample
    # at or before it (zero-order hold) and is valid only while that sample is
    # at most max_gap_s old. No interpolation — a sample at t > grid_t can
    # never influence the frame at grid_t, so the stream is exactly what a
    # deployed robot could have computed (zero lookahead; staleness is bounded
    # by max_gap_s, it is not added pipeline latency). Regression-tested with
    # Codex's adjacent-sample and beyond-gap counterexamples.
    interp: dict[str, dict[str, np.ndarray]] = {}
    own_valid: dict[str, np.ndarray] = {}
    for label, samples in per_label.items():
        t_src = samples["shared_t"].to_numpy()
        latest = np.searchsorted(t_src, grid, side="right") - 1
        has_past = latest >= 0
        clipped = np.clip(latest, 0, len(t_src) - 1)
        interp[label] = {
            col: np.where(has_past, samples[col].to_numpy()[clipped], 0.0)
            for col in ("x_w", "z_w", "facing_x", "facing_z")
        }
        age = grid - t_src[clipped]
        # defence in depth: held source vectors are validated upstream, but a
        # degenerate horizontal facing must never become a silent yaw
        facing_norm = np.hypot(interp[label]["facing_x"], interp[label]["facing_z"])
        own_valid[label] = (
            has_past & (age <= config.max_gap_s) & (facing_norm >= 1e-6)
        )

    # centroid of the non-R participants, for R's own deviation target. Per the
    # Codex R1.2-5 ruling the centroid uses only the members validly observed
    # at that frame, and R's yaw needs AT LEAST 2 valid members (2/2 in 3-party
    # groups) — a single-member "centroid" would degenerate into a bearing to
    # one person and smuggle addressee semantics into the yaw feature
    others = [lbl for lbl in per_label if lbl != "R"]
    member_valid = np.stack([own_valid[p] for p in others])  # (M, T)
    n_valid_members = member_valid.sum(axis=0)
    weights = member_valid.astype(float)
    denom = np.maximum(n_valid_members, 1).astype(float)
    centroid_x = (weights * np.stack([interp[p]["x_w"] for p in others])).sum(axis=0) / denom
    centroid_z = (weights * np.stack([interp[p]["z_w"] for p in others])).sum(axis=0) / denom
    members_ok = n_valid_members >= 2

    present_slots = {SLOT_BY_PARTICIPANT[label] for label in per_label}

    # pass 2: head yaw + pose validity/quality. The yaw target depends on OTHER
    # participants' positions, so validity propagates (R1.1-4 / R1.2-5): a
    # non-R yaw is only valid where R is also validly observed; R's yaw is only
    # valid where >= 2 centroid members are. pose_quality is the continuous
    # robot-observable companion (R: valid-member fraction; non-R: 1 iff R
    # valid) and is NOT overloaded into the binary pose_valid flag
    for label in per_label:
        slot = SLOT_BY_PARTICIPANT[label]
        facing = np.stack([interp[label]["facing_x"], interp[label]["facing_z"]], axis=1)
        pos = np.stack([interp[label]["x_w"], interp[label]["z_w"]], axis=1)
        if label == "R":
            target = np.stack([centroid_x, centroid_z], axis=1) - pos
            valid = own_valid[label] & members_ok
            pose_quality = np.where(own_valid[label], n_valid_members / len(others), 0.0)
        else:
            target = np.stack([interp["R"]["x_w"], interp["R"]["z_w"]], axis=1) - pos
            valid = own_valid[label] & own_valid["R"]
            pose_quality = None  # non-R quality tracks final validity, set below
        # a zero-length bearing (target on top of the observer) leaves the yaw
        # undefined — invalid, not a silent 90° (Codex R1.2-5)
        valid = valid & (np.linalg.norm(target, axis=1) >= 1e-9)
        if pose_quality is None:
            pose_quality = valid.astype(float)
        yaw = _horizontal_bearing_deviation_deg(facing, target)
        data[participant_column(slot, "head_yaw_deg")] = np.where(valid, yaw, 0.0)
        data[participant_column(slot, "pose_valid")] = valid.astype(float)
        data[participant_column(slot, "pose_quality")] = np.clip(pose_quality, 0.0, 1.0)
        missing_fraction[f"P{slot}"] = float(1.0 - valid.mean())
        quality[f"P{slot}_mean_facing_src_abs_yaw"] = float(
            per_label[label]["facing_src_abs_yaw"].mean()
        )
        quality[f"P{slot}_pose_quality_mean"] = float(np.clip(pose_quality, 0.0, 1.0).mean())
    quality["r_valid_member_count_histogram"] = {
        str(k): int((n_valid_members == k).sum()) for k in range(len(others) + 1)
    }

    zeros = np.zeros(n_grid)
    placeholder_columns: list[str] = []
    # P{i}_present is STATIC ("this participant exists in the session"); tracker
    # dropouts are carried by the per-frame P{i}_pose_valid feature instead —
    # gold-VAD supervision must not be discarded on vision gaps (plan A, R1.1-1)
    for slot in range(1, MAX_PARTICIPANTS + 1):
        present = slot in present_slots
        data[participant_column(slot, "present")] = np.full(n_grid, int(present))
        for name in [*PARTICIPANT_FEATURE_NAMES, "vad"]:
            column = participant_column(slot, name)
            if column in data:
                continue
            data[column] = zeros
            if present:
                placeholder_columns.append(column)
    for name in GLOBAL_FEATURE_COLUMNS:
        data[name] = zeros
        placeholder_columns.append(name)

    frame_df = pd.DataFrame(data)
    frame_df.insert(0, "session_id", delivery.session_id)
    validate_multiparty_dataframe(frame_df)

    real_columns = sorted(
        participant_column(SLOT_BY_PARTICIPANT[label], name)
        for label in per_label
        for name in REAL_PARTICIPANT_FEATURES
    )
    offset_quality_summary = None
    if delivery.offset_quality is not None:
        stream_quality = delivery.offset_quality["quality"][config.stream_key]
        offset_quality_summary = {
            "stream": config.stream_key,
            "offset_s": float(stream_quality["offset_s"]),
            "self_check_ok": stream_quality["self_check_ok"],
            "convention": delivery.offset_quality["convention"],
            "status": stream_quality["status"],
            "error_bound_ms": float(stream_quality["error_bound_ms"]),
            "timebase_keys": delivery.offset_quality["timebase_keys"],
        }
    delivery_snapshot_after = _snapshot_delivery_sources(directory)
    if delivery_snapshot_after != delivery_snapshot_before:
        raise DeliveryGateError(
            "An HPE delivery source changed while intake was running; refusing a mixed snapshot"
        )
    _validate_contract_source_bindings(
        contract,
        delivery_sources=delivery_snapshot_after,
        repository_root=root,
    )
    world_persons_source = {
        "path": delivery_snapshot_after["world_persons"]["path"],
        "sha256": delivery_snapshot_after["world_persons"]["sha256"],
        "size_bytes": delivery_snapshot_after["world_persons"]["size_bytes"],
    }
    participant_by_slot = {
        f"P{SLOT_BY_PARTICIPANT[label]}": label for label in per_label
    }
    persons_by_participant = {
        label: sorted(
            [
                person
                for person, mapped_label in person_to_participant.items()
                if mapped_label == label
            ],
            key=int,
        )
        for label in per_label
    }
    pose_source_columns = [
        "frame_index",
        "timestamp_sec",
        "person",
        "x_w",
        "z_w",
        "facing_x",
        "facing_z",
        "facing_src_abs_yaw",
    ]
    column_sources = {
        participant_column(SLOT_BY_PARTICIPANT[label], name): {
            "participant": participant_by_slot[f"P{SLOT_BY_PARTICIPANT[label]}"],
            "slot": f"P{SLOT_BY_PARTICIPANT[label]}",
            "purpose": "model_input",
            "source_kind": "hpe_pose_intake_v2",
            "source_columns": pose_source_columns,
            "hpe_persons": persons_by_participant[label],
            "identity_authority": "sealed_hpe_transcript_crosswalk",
            "identity_inventory_sha256": contract["inventory"]["sha256"],
            "transform": "strict_past_only_pose_v2",
            **world_persons_source,
        }
        for label in per_label
        for name in REAL_PARTICIPANT_FEATURES
    }
    provenance = {
        "data_kind": "real",
        "artifact_kind": REAL_INTAKE_ARTIFACT_KIND,
        "format_version": REAL_PROVENANCE_FORMAT_VERSION,
        "session_id": delivery.session_id,
        "group_id": delivery.group,
        "scenario_id": delivery.scenario,
        "delivery_dir": str(delivery.directory.resolve()),
        "review_status": "confirmed",
        "identity_authority": "sealed_hpe_transcript_crosswalk",
        "identity_contract": contract,
        "upstream_hpe_review_status": delivery.review_status,
        "stream_key": config.stream_key,
        "offset_applied_s": offset_s,
        "offset_quality": offset_quality_summary,
        "d3_sources": {
            "offset": dict(delivery_snapshot_after["offset"]),
            "offset_quality": dict(delivery_snapshot_after["offset_quality"]),
        },
        "intake_sources": delivery_snapshot_after,
        "intake_config": _formal_intake_config(config),
        "accepted_window": dict(accepted_window),
        "source_frame_rate_hz": source_fps,
        "source_time_range_local_s": [
            logical_start / source_fps,
            logical_end / source_fps,
        ],
        "output_tick_range": {
            "start_inclusive": start_tick,
            "end_exclusive": end_tick_exclusive,
        },
        "frame_rate_hz": fps,
        "n_frames": int(n_grid),
        "time_range_shared_s": [float(grid[0]), float(grid[-1])],
        "participant_to_slot": {
            label: f"P{SLOT_BY_PARTICIPANT[label]}"
            for label in set(person_to_participant.values())
        },
        "person_to_participant": dict(person_to_participant),
        "upstream_person_to_participant_proposal": dict(delivery.participant_map),
        "fragments_per_participant": fragments,
        "fragment_overlap": fragment_overlap,
        "fragment_overlap_policy": config.fragment_overlap_policy,
        "validity_note": (
            "P{i}_present is static (slot exists); per-frame observation gaps are "
            "carried by P{i}_pose_valid=0 with zeroed features (STRICT past-only "
            "zero-order hold — each frame holds the latest sample at or before it, "
            "zero lookahead, staleness <= max_gap_s [Codex R1.4-2]; frames with no "
            "sample within max_gap_s before them are invalid; degenerate facing "
            "is invalid; "
            f"max_gap_s={config.max_gap_s}; non-R validity requires R validity, "
            "R validity requires >= 2 valid centroid members [Codex R1.2-5]; "
            "P{i}_pose_quality carries the continuous valid-member fraction)"
        ),
        "unmapped_persons_dropped": {
            "persons": list(delivery.unmapped_persons),
            "rows": 0,
        },
        "rows_outside_accepted_window": n_outside_window_rows,
        "real_columns": real_columns,
        "column_sources": column_sources,
        "placeholder_columns": sorted(placeholder_columns),
        "trainable": False,  # gold VAD is still a placeholder
        "missing_sample_fraction": missing_fraction,
        "facing_quality": quality,
        "doa_note": "no mic array recorded; doa_angle_deg is a permanent constant on real data",
    }
    return IntakeResult(frame_df=frame_df, provenance=provenance)


def assert_trainable(dataset_csv: str | Path, allow_untrainable: bool = False) -> None:
    """Hard training gate (Codex R1 ④): refuse placeholder intake tables.

    A real-data table written by ``write_intake`` carries a sibling
    ``.provenance.json`` with ``trainable: false`` until gold VAD (and the other
    required modalities) are merged in. Training on such a table would fit the
    model to zero-filled placeholders, so it is an error, not a warning.
    Bare CSVs are never accepted as implicitly synthetic; the main source
    loader also validates a synthetic sidecar against the CSV content hash.
    """
    provenance_path = Path(dataset_csv).with_suffix(".provenance.json")
    if not provenance_path.exists():
        raise DeliveryGateError(
            f"Dataset {dataset_csv} has no {provenance_path.name}; bare CSV files are "
            "not trusted as synthetic"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("trainable", True) or allow_untrainable:
        return
    raise DeliveryGateError(
        f"{provenance_path.name} marks this table trainable=false (placeholder columns: "
        f"{provenance.get('placeholder_columns', 'unknown')}). Refusing to train until the "
        "gold-VAD merge flips it; pass --allow-untrainable only for pipeline debugging."
    )


def write_intake(
    delivery_dir: str | Path,
    output_csv: str | Path,
    config: IntakeConfig = DEFAULT_INTAKE_CONFIG,
    *,
    crosswalk_inventory: str | Path = DEFAULT_SEALED_INVENTORY_PATH,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> IntakeResult:
    output_path = Path(output_csv)
    provenance_path = output_path.with_suffix(".provenance.json")
    if output_path.exists() or provenance_path.exists():
        raise DeliveryGateError(
            f"Refusing to overwrite existing intake output: {output_path} or "
            f"{provenance_path}"
        )
    result = run_intake(
        delivery_dir,
        config,
        crosswalk_inventory=crosswalk_inventory,
        repository_root=repository_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = _reserve_staged_path(
        output_path.parent,
        prefix=f".{output_path.name}.tmp.",
        suffix=".csv",
    )
    sidecar_tmp = _reserve_staged_path(
        provenance_path.parent,
        prefix=f".{provenance_path.name}.",
        suffix=".json.tmp",
    )
    published: list[Path] = []
    try:
        result.frame_df.to_csv(csv_tmp, index=False)
        # Match the production loader's exact binary64 round-trip semantics.
        # The default Pandas parser may return a neighbouring float for a
        # shortest-round-trip decimal, which would make a just-written pose
        # table fail its own deterministic replay gate.
        persisted = pd.read_csv(csv_tmp, float_precision="round_trip")
        columns = persisted.columns.tolist()
        bound_provenance = {
            **result.provenance,
            "csv_sha256": _sha256(csv_tmp),
            "columns": columns,
            "dtypes": {column: str(persisted[column].dtype) for column in columns},
            "n_rows": len(persisted),
            "n_frames": len(persisted),
        }
        sidecar_tmp.write_text(
            json.dumps(
                bound_provenance,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        staged_csv_sha256 = bound_provenance["csv_sha256"]
        staged_sidecar_sha256 = _sha256(sidecar_tmp)
        final_sources = _snapshot_delivery_sources(Path(delivery_dir).resolve())
        if final_sources != result.provenance["intake_sources"]:
            raise DeliveryGateError(
                "An HPE delivery source changed before intake publication; refusing output"
            )
        _validate_contract_source_bindings(
            result.provenance["identity_contract"],
            delivery_sources=final_sources,
            repository_root=Path(repository_root).resolve(),
        )
        if output_path.exists() or provenance_path.exists():
            raise DeliveryGateError(
                "Intake output appeared while materialisation was running; refusing overwrite"
            )
        if (
            _sha256(csv_tmp) != staged_csv_sha256
            or _sha256(sidecar_tmp) != staged_sidecar_sha256
        ):
            raise DeliveryGateError(
                "Intake staging bytes changed before publication; refusing output"
            )
        _publish_no_clobber(csv_tmp, output_path)
        published.append(output_path)
        _publish_no_clobber(sidecar_tmp, provenance_path)
        published.append(provenance_path)
        if (
            _sha256(output_path) != staged_csv_sha256
            or _sha256(provenance_path) != staged_sidecar_sha256
        ):
            raise DeliveryGateError(
                "Published intake bytes differ from the validated staged pair"
            )
        published_sources = _snapshot_delivery_sources(Path(delivery_dir).resolve())
        if published_sources != result.provenance["intake_sources"]:
            raise DeliveryGateError(
                "An HPE delivery source changed during intake publication; refusing output"
            )
        _validate_contract_source_bindings(
            result.provenance["identity_contract"],
            delivery_sources=published_sources,
            repository_root=Path(repository_root).resolve(),
        )
        result.provenance = bound_provenance
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for published_path in reversed(published):
            try:
                published_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{published_path}: {cleanup_exc}")
        if cleanup_errors:
            raise DeliveryGateError(
                "Intake publication failed and rollback was incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    finally:
        csv_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
    return result
