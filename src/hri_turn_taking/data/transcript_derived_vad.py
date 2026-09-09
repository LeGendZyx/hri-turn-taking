"""Owner-attested, human-reviewed Transcript-derived VAD labels.

This is a new contract, not a migration or impersonation of review-v3.  It
records the owner's existing annotation claim and asks only for explicit
decisions on the four impossible in-window intervals.  Upstream files remain
immutable.  Labels are half-open utterance unions rasterised on the canonical
25 Hz mic-A analysis grid.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import (
    analysis_base,
    assembly,
    audio_contract,
    audio_gold_alignment,
    transcript_acceptance,
    transcript_fs_guard,
    transcript_review,
)
from hri_turn_taking.data.source_paths import corpus_root

ATTESTATION_KIND = "vap_transcript_existing_annotation_owner_attestation"
DECISIONS_KIND = "vap_transcript_targeted_semantic_decisions"
CONTRACT_KIND = "vap_human_reviewed_transcript_derived_vad_contract"
FORMAT_VERSION = 1
DECISIONS_FORMAT_VERSION = 2
OUTPUT_ROLE = "human_reviewed_transcript_derived_vad"
RESEARCH_LABEL = "human-reviewed-transcript-derived VAD"
FRAME_RATE_HZ = 25
SPEAKERS = ("A", "B", "C", "R")
PARTICIPANT_TO_SLOT = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
SLOT_TO_PARTICIPANT = {slot: person for person, slot in PARTICIPANT_TO_SLOT.items()}
TRUST_MODEL = "trusted_single_writer_local_workspace_integrity_and_reproducibility"
_PRODUCTION_TOKEN = object()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_RE = re.compile(r"^[0-9]{2}_[0-9]{2}(?:_[0-9]+)?__[0-9]+$")
_EVENT_FIELDS = ("start_timecode", "end_timecode", "speaker", "text")

# Only these four source-indexed in-window anomalies require a new decision.
# The fifth old review-v3 proposal (05_14__3/e146) is outside the canonical
# analysis window and is intentionally absent.
SEMANTIC_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    scene: () for scene in analysis_base.EXPECTED_TICK_RANGES
}
SEMANTIC_CANDIDATES.update(
    {
        "05_14__2": (
            {
                "candidate_id": "05-14-2-e27-event",
                "source_index": 27,
                "expected_event": {
                    "start_timecode": "00:00:39:09",
                    "end_timecode": "00:00:29:21",
                    "speaker": "B",
                    "text": "Oh, okay.",
                },
                "proposed_event": {
                    "start_timecode": "00:00:39:09",
                    "end_timecode": "00:00:39:21",
                    "speaker": "B",
                    "text": "Oh, okay.",
                },
                "timeline_local_clip_start_s": "38.8",
                "timeline_local_clip_end_s_exclusive": "40.2",
            },
        ),
        "05_15__2": (
            {
                "candidate_id": "05-15-2-e76-event",
                "source_index": 76,
                "expected_event": {
                    "start_timecode": "00:03:54:21",
                    "end_timecode": "00:03:27:02",
                    "speaker": "B",
                    "text": "Are we convinced by this restaurant?",
                },
                "proposed_event": {
                    "start_timecode": "00:03:54:21",
                    "end_timecode": "00:03:57:02",
                    "speaker": "A",
                    "text": "Are we convinced by this restaurant?",
                },
                "timeline_local_clip_start_s": "234.2",
                "timeline_local_clip_end_s_exclusive": "238.3",
            },
        ),
        "05_15__3": (
            {
                "candidate_id": "05-15-3-e17-event",
                "source_index": 17,
                "expected_event": {
                    "start_timecode": "00:00:34:25",
                    "end_timecode": "00:00:34:25",
                    "speaker": "R",
                    "text": "Okay, xxxx.",
                },
                "proposed_event": {
                    "start_timecode": "00:00:33:25",
                    "end_timecode": "00:00:34:25",
                    "speaker": "R",
                    "text": "Okay, let's stop there.",
                },
                "timeline_local_clip_start_s": "33.4",
                "timeline_local_clip_end_s_exclusive": "35.8",
            },
            {
                "candidate_id": "05-15-3-e88-event",
                "source_index": 88,
                "expected_event": {
                    "start_timecode": "00:02:36:19",
                    "end_timecode": "00:02:28:15",
                    "speaker": "A",
                    "text": "Thank you, dear. For letting me ask first.",
                },
                "proposed_event": {
                    "start_timecode": "00:02:36:19",
                    "end_timecode": "00:02:38:15",
                    "speaker": "A",
                    "text": "Thank you, dear. For letting me ask first.",
                },
                "timeline_local_clip_start_s": "156.1",
                "timeline_local_clip_end_s_exclusive": "159",
            },
        ),
    }
)


class TranscriptDerivedVadError(ValueError):
    """A transcript-derived target failed a source, decision, or replay gate."""


@dataclass(frozen=True, kw_only=True)
class TranscriptDerivedVadAuthority:
    """Code-owned source/output roots; manifests never redefine authority."""

    corpus_root: Path
    analysis_base_root: Path
    gold_alignment_root: Path
    audio_contract_root: Path
    annotation_root: Path
    target_root: Path
    clip_root: Path
    analysis_base_authority: object | None = None
    _production_marker: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "corpus_root",
            "analysis_base_root",
            "gold_alignment_root",
            "audio_contract_root",
            "annotation_root",
            "target_root",
            "clip_root",
        ):
            object.__setattr__(
                self,
                name,
                transcript_fs_guard.lexical_absolute(getattr(self, name)),
            )

    @property
    def production(self) -> bool:
        return self._production_marker is _PRODUCTION_TOKEN


@dataclass(frozen=True)
class _SourceSnapshot:
    scene: str
    timeline_path: Path
    timeline_raw: bytes
    offset_path: Path
    offset_raw: bytes
    analysis_csv: Path
    analysis_sidecar: Path
    frame_df: pd.DataFrame
    analysis_provenance: dict[str, Any]
    gold_path: Path
    gold_report: dict[str, Any]
    audio_contract_path: Path
    audio_report: dict[str, Any]
    transcript_offset_sec: Decimal
    reference_duration_sec: Decimal
    start_tick: int
    end_tick: int
    participant_to_slot: Mapping[str, str]
    present_by_slot: Mapping[str, int]
    corrections_s: Mapping[str, Decimal]
    lavalier_paths: Mapping[str, Path]
    bindings: tuple[dict[str, Any], ...]
    authority_scope: str


@dataclass(frozen=True)
class TranscriptDerivedVadResult:
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    contract: dict[str, Any]
    manifest: dict[str, Any]
    source_csv: Path | None = None
    target_csv: Path | None = None


def default_transcript_derived_vad_authority() -> TranscriptDerivedVadAuthority:
    root = Path(__file__).resolve().parents[3]
    return TranscriptDerivedVadAuthority(
        corpus_root=corpus_root(root),
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        gold_alignment_root=root / "data" / "annotations" / "audio_gold_alignment",
        audio_contract_root=root / "data" / "annotations" / "audio_contract",
        annotation_root=root / "data" / "annotations" / "transcript_derived_vad",
        target_root=root / "data" / "processed" / "transcript_derived_vad_targets",
        clip_root=root / "outputs" / "transcript_targeted_reaudit",
        _production_marker=_PRODUCTION_TOKEN,
    )


def _fail(message: str) -> None:
    raise TranscriptDerivedVadError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TranscriptDerivedVadError("artifact is not strict finite JSON") from exc


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranscriptDerivedVadError(f"{label} is not strict UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        _fail(f"{label} contains forbidden non-finite constant {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise TranscriptDerivedVadError(
            f"{label} is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _type_exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _type_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _guarded_binding(path: Path, *, media_type: str) -> tuple[dict[str, Any], bytes]:
    try:
        with transcript_fs_guard.guarded_read(path, allow_windows_read_metadata_churn=True) as held:
            raw = transcript_fs_guard.read_guarded_bytes(held)
            binding = {
                "path": str(held.path),
                "size_bytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "media_type": media_type,
            }
            return binding, raw
    except (OSError, transcript_fs_guard.TranscriptPathGuardError) as exc:
        raise TranscriptDerivedVadError(f"unsafe or unreadable source {path}: {exc}") from exc


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{label} must not be bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TranscriptDerivedVadError(f"{label} is not an exact decimal") from exc
    if not result.is_finite():
        _fail(f"{label} must be finite")
    return result


def _canonical_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _validate_scene(scene: str) -> None:
    if type(scene) is not str or _SCENE_RE.fullmatch(scene) is None:
        _fail("scene must use canonical group__scenario spelling")
    if scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail(f"scene {scene!r} is outside the sealed ten-scene corpus")


def _canonical_source_paths(
    scene: str, authority: TranscriptDerivedVadAuthority
) -> dict[str, Path]:
    _validate_scene(scene)
    group, scenario = scene.rsplit("__", maxsplit=1)
    transcript_dir = authority.corpus_root / group / "transcript" / scenario
    return {
        "timeline": transcript_dir / "timeline_video.json",
        "offset": transcript_dir / "offset.json",
        "analysis_csv": authority.analysis_base_root / f"{scene}.csv",
        "analysis_sidecar": authority.analysis_base_root / f"{scene}.provenance.json",
        "gold_alignment": (authority.gold_alignment_root / f"{scene}.audio_gold_alignment.json"),
        "audio_contract": authority.audio_contract_root / f"{scene}.audio_contract.json",
    }


def _canonical_annotation_dir(scene: str, authority: TranscriptDerivedVadAuthority) -> Path:
    _validate_scene(scene)
    return authority.annotation_root / scene


def canonical_attestation_path(
    scene: str, authority: TranscriptDerivedVadAuthority | None = None
) -> Path:
    selected = authority or default_transcript_derived_vad_authority()
    return _canonical_annotation_dir(scene, selected) / "owner_attestation.v1.json"


def canonical_decisions_path(
    scene: str, authority: TranscriptDerivedVadAuthority | None = None
) -> Path:
    selected = authority or default_transcript_derived_vad_authority()
    return _canonical_annotation_dir(scene, selected) / "semantic_decisions.v2.json"


def _target_paths(scene: str, authority: TranscriptDerivedVadAuthority) -> dict[str, Path]:
    root = authority.target_root / scene
    return {
        "source": root / "vad_source.v1.csv",
        "target": root / "vad_target.v1.csv",
        "manifest": root / "vad_target.v1.manifest.json",
        "contract": root / "vad_contract.v1.json",
    }


def _binding_list_hash(bindings: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(bindings)).rstrip(b"\n"))


def _load_source_snapshot(
    scene: str, *, authority: TranscriptDerivedVadAuthority
) -> _SourceSnapshot:
    """Freshly replay every formal source needed for one label table."""

    paths = _canonical_source_paths(scene, authority)
    timeline_binding, timeline_raw = _guarded_binding(
        paths["timeline"], media_type="application/json"
    )
    offset_binding, offset_raw = _guarded_binding(paths["offset"], media_type="application/json")
    analysis_binding, _ = _guarded_binding(paths["analysis_csv"], media_type="text/csv")
    sidecar_binding, _ = _guarded_binding(paths["analysis_sidecar"], media_type="application/json")
    gold_binding, _ = _guarded_binding(paths["gold_alignment"], media_type="application/json")
    audio_binding, audio_raw = _guarded_binding(
        paths["audio_contract"], media_type="application/json"
    )
    try:
        dataset = assembly._load_dataset_file(
            paths["analysis_csv"],
            data_kind="real",
            analysis_base_authority=authority.analysis_base_authority,
        )
        provenance = dataset.provenance
        if (
            type(provenance) is not dict
            or provenance.get("artifact_kind") != analysis_base.ARTIFACT_KIND
            or provenance.get("session_id") != scene
        ):
            _fail("target source must be the canonical vap_real_analysis_base")
        gold_report = audio_gold_alignment.load_audio_gold_alignment(paths["gold_alignment"])
        # The formal gold-alignment loader above already performs the complete
        # audio-contract/source replay. Parse the exact bytes held before that
        # replay instead of decoding the same multi-GB WAV payloads twice.
        audio_report = audio_contract._strict_json_loads(audio_raw, "bound audio contract")
        audio_contract._validate_contract_shape(audio_report)
    except TranscriptDerivedVadError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise TranscriptDerivedVadError(f"formal source replay failed: {exc}") from exc
    if gold_report.get("scene") != scene or audio_report.get("scene") != scene:
        _fail("gold-alignment/audio-contract scene differs from requested scene")
    if gold_report.get("authorization", {}).get("gold_vad_use_authorized") is not True:
        _fail("gold-alignment does not authorize downstream label use")

    tick_value = provenance.get("output_tick_range")
    if type(tick_value) is not dict:
        tick_value = provenance.get("analysis_window_contract", {}).get("authorized_tick_range")
    if type(tick_value) is not dict:
        _fail("analysis base lacks its authorised tick range")
    start_tick = tick_value.get("start_inclusive")
    end_tick = tick_value.get("end_exclusive")
    if (
        type(start_tick) is not int
        or type(end_tick) is not int
        or start_tick < 0
        or end_tick <= start_tick
    ):
        _fail("analysis tick bounds must be exact positive JSON integers")
    if authority.production and (start_tick, end_tick) != analysis_base.EXPECTED_TICK_RANGES[scene]:
        _fail("analysis ticks differ from the frozen ten-scene inventory")
    timestamps = dataset.frame_df["timestamp_s"].to_numpy(dtype=float)
    ticks = np.rint(timestamps * FRAME_RATE_HZ).astype(np.int64)
    if not np.array_equal(ticks, np.arange(start_tick, end_tick, dtype=np.int64)):
        _fail("analysis timestamps are not exact consecutive 25-Hz ticks")
    if not np.array_equal(timestamps, ticks.astype(float) / FRAME_RATE_HZ):
        _fail("analysis timestamps are not exact tick/25 values")

    mapping = provenance.get("participant_to_slot")
    if type(mapping) is not dict or mapping not in (
        {"R": "P1", "A": "P2", "B": "P3"},
        PARTICIPANT_TO_SLOT,
    ):
        _fail("analysis participant_to_slot must be canonical R/P1,A/P2,B/P3[,C/P4]")
    present: dict[str, int] = {}
    for slot in ("P1", "P2", "P3", "P4"):
        values = dataset.frame_df[f"{slot}_present"].to_numpy(dtype=float)
        if len(set(values.tolist())) != 1 or values[0] not in {0.0, 1.0}:
            _fail(f"{slot}_present must be one static exact binary value")
        present[slot] = int(values[0])
    if any(present[PARTICIPANT_TO_SLOT[p]] != 1 for p in mapping):
        _fail("every mapped participant must have present=1")
    if any(
        present[slot] != int(participant in mapping)
        for participant, slot in PARTICIPANT_TO_SLOT.items()
    ):
        _fail("present mask and participant_to_slot disagree")

    gold_ticks = gold_report.get("authorization", {}).get("authorized_tick_range")
    if type(gold_ticks) is not dict or (
        gold_ticks.get("start_inclusive"),
        gold_ticks.get("end_exclusive"),
    ) != (start_tick, end_tick):
        _fail("gold-alignment and analysis windows differ")
    channels = gold_report.get("mapping_contract", {}).get("channels")
    if type(channels) is not dict or set(channels) != set(SPEAKERS):
        _fail("gold-alignment mapping must cover exactly A/B/C/R")
    corrections = {
        speaker: _decimal(channels[speaker].get("correction_s"), f"{speaker} correction")
        for speaker in SPEAKERS
    }
    sources = audio_report.get("sources")
    if type(sources) is not dict:
        _fail("audio contract lacks source bindings")
    reference = sources.get("reference")
    if type(reference) is not dict or type(reference.get("path")) is not str:
        _fail("audio contract lacks its Transcript reference WAV")
    try:
        raw_offset_value = transcript_acceptance._strict_json_loads(offset_raw, "transcript offset")
        normalized_offset, _ = transcript_acceptance._normalize_transcript_offset_source(
            offset_raw,
            raw_offset_value,
            transcript_review._normalization(),
        )
        offset_sec, _ = transcript_acceptance._validate_transcript_offset(
            normalized_offset,
            scene=scene,
            reference_wav=Path(reference["path"]),
        )
        reference_duration = transcript_acceptance._require_real_number(
            normalized_offset["reference_duration_sec"],
            "reference_duration_sec",
        )
    except (ValueError, KeyError) as exc:
        raise TranscriptDerivedVadError(f"Transcript offset replay failed: {exc}") from exc

    lav_paths: dict[str, Path] = {}
    extra_bindings: list[dict[str, Any]] = []
    for speaker in SPEAKERS:
        item = sources.get(speaker)
        if type(item) is not dict or type(item.get("path")) is not str:
            _fail(f"audio contract lacks {speaker} lavalier source")
        path = transcript_fs_guard.lexical_absolute(item["path"])
        try:
            transcript_fs_guard.guard_regular_file_path(path)
        except transcript_fs_guard.TranscriptPathGuardError as exc:
            raise TranscriptDerivedVadError(f"unsafe {speaker} lavalier path: {exc}") from exc
        binding = {
            "path": str(path),
            "size_bytes": item.get("size_bytes"),
            "sha256": item.get("sha256"),
            "media_type": "audio/wav",
        }
        if (
            type(binding["size_bytes"]) is not int
            or binding["size_bytes"] < 0
            or type(binding["sha256"]) is not str
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            _fail(f"audio contract {speaker} binding shape is invalid")
        lav_paths[speaker] = path
        extra_bindings.append(binding)
    bindings = tuple(
        sorted(
            (
                timeline_binding,
                offset_binding,
                analysis_binding,
                sidecar_binding,
                gold_binding,
                audio_binding,
                *extra_bindings,
            ),
            key=lambda item: str(item["path"]).casefold(),
        )
    )
    return _SourceSnapshot(
        scene=scene,
        timeline_path=paths["timeline"],
        timeline_raw=timeline_raw,
        offset_path=paths["offset"],
        offset_raw=offset_raw,
        analysis_csv=paths["analysis_csv"],
        analysis_sidecar=paths["analysis_sidecar"],
        frame_df=dataset.frame_df.copy(),
        analysis_provenance=json.loads(json.dumps(provenance)),
        gold_path=paths["gold_alignment"],
        gold_report=json.loads(json.dumps(gold_report)),
        audio_contract_path=paths["audio_contract"],
        audio_report=json.loads(json.dumps(audio_report)),
        transcript_offset_sec=offset_sec,
        reference_duration_sec=reference_duration,
        start_tick=start_tick,
        end_tick=end_tick,
        participant_to_slot=dict(mapping),
        present_by_slot=present,
        corrections_s=corrections,
        lavalier_paths=lav_paths,
        bindings=bindings,
        authority_scope="production" if authority.production else "fixture",
    )


def _mechanical_repairs(scene: str) -> list[dict[str, Any]]:
    plan = transcript_review.FORMAL_REVIEW_PLANS[scene]
    operations = plan.get("mechanical_repairs")
    if type(operations) is not list:
        _fail("code-owned mechanical repair plan is malformed")
    result = json.loads(json.dumps(operations))
    if any(item.get("classification") != "mechanical" for item in result):
        _fail("mechanical repair plan contains a semantic operation")
    return result


def _candidate_catalog(scene: str) -> list[dict[str, Any]]:
    return json.loads(json.dumps(list(SEMANTIC_CANDIDATES[scene])))


def _candidate_catalog_sha256(scene: str) -> str:
    return _sha256_bytes(_canonical_json_bytes(_candidate_catalog(scene)).rstrip(b"\n"))


def _validate_identity_and_time(identity: Any, timestamp: Any, *, label: str) -> tuple[str, str]:
    if (
        type(identity) is not str
        or not identity
        or identity != identity.strip()
        or type(timestamp) is not str
        or not timestamp
        or timestamp != timestamp.strip()
    ):
        _fail(f"{label} identity/timestamp must be non-empty trimmed strings")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TranscriptDerivedVadError(f"{label} timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        _fail(f"{label} timestamp must include a UTC offset")
    return identity, timestamp


def _attestation_payload(
    snapshot: _SourceSnapshot,
    *,
    owner: str,
    attested_at: str,
) -> dict[str, Any]:
    owner, attested_at = _validate_identity_and_time(owner, attested_at, label="owner attestation")
    return {
        "artifact_kind": ATTESTATION_KIND,
        "format_version": FORMAT_VERSION,
        "scene": snapshot.scene,
        "authority_scope": snapshot.authority_scope,
        "output_role": OUTPUT_ROLE,
        "research_label": RESEARCH_LABEL,
        "owner": {"identity": owner, "attested_at": attested_at},
        "annotation_claim": {
            "existing_human_annotation": True,
            "targeted_reaudit_only": True,
            "all_events_reaudited": False,
            "speaker_assignments_already_human_reviewed": True,
            "overlap_already_human_reviewed": True,
            "quiet_speech_already_human_reviewed": True,
            "pauses_already_human_reviewed": True,
            "new_full_listening_review_required": False,
            "seven_day_repeat_review_required": False,
            "blind_fifty_items_per_scene_required": False,
            "attestation_basis": (
                "project owner statement on 2026-08-16 that the ten Transcript "
                "annotations are reliable human-reviewed speaker segments"
            ),
        },
        "review_v3_dependency": {
            "required": False,
            "accepted_as_attestation": False,
            "reason": "new owner-attested contract; no all-events re-review is claimed",
        },
        "source_bindings": list(snapshot.bindings),
        "source_set_sha256": _binding_list_hash(snapshot.bindings),
        "trust_boundary": {
            "model": TRUST_MODEL,
            "trusted_single_writer": True,
            "hostile_concurrent_replacement_resistance_claimed": False,
            "no_overwrite_required": True,
        },
    }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _fail(
            f"{label} keys differ from v{FORMAT_VERSION}; "
            f"missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _validate_attestation_value(
    value: Any,
    *,
    snapshot: _SourceSnapshot,
    path: Path,
    authority: TranscriptDerivedVadAuthority,
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("owner attestation must be one JSON object")
    _require_exact_keys(
        value,
        {
            "artifact_kind",
            "format_version",
            "scene",
            "authority_scope",
            "output_role",
            "research_label",
            "owner",
            "annotation_claim",
            "review_v3_dependency",
            "source_bindings",
            "source_set_sha256",
            "trust_boundary",
        },
        "owner attestation",
    )
    if (
        value["artifact_kind"] != ATTESTATION_KIND
        or type(value["format_version"]) is not int
        or value["format_version"] != FORMAT_VERSION
        or value["scene"] != snapshot.scene
        or value["authority_scope"] != snapshot.authority_scope
        or value["output_role"] != OUTPUT_ROLE
        or value["research_label"] != RESEARCH_LABEL
    ):
        _fail("owner attestation role/version/scene/authority is invalid")
    if authority.production and path != canonical_attestation_path(snapshot.scene, authority):
        _fail("production owner attestation was copied or re-anchored")
    owner_value = value["owner"]
    if type(owner_value) is not dict:
        _fail("owner attestation owner must be an object")
    _require_exact_keys(owner_value, {"identity", "attested_at"}, "owner")
    _validate_identity_and_time(
        owner_value["identity"], owner_value["attested_at"], label="owner attestation"
    )
    claim = value["annotation_claim"]
    if type(claim) is not dict:
        _fail("annotation_claim must be an object")
    expected_claim = _attestation_payload(
        snapshot,
        owner=owner_value["identity"],
        attested_at=owner_value["attested_at"],
    )["annotation_claim"]
    if not _type_exact_equal(claim, expected_claim):
        _fail("annotation_claim was weakened or changed")
    dependency = value["review_v3_dependency"]
    if (
        type(dependency) is not dict
        or dependency.get("required") is not False
        or dependency.get("accepted_as_attestation") is not False
    ):
        _fail("old review-v3 cannot impersonate this owner attestation")
    if not _type_exact_equal(value["source_bindings"], list(snapshot.bindings)):
        _fail("owner attestation source bytes/paths differ from fresh authority")
    source_hash = value["source_set_sha256"]
    if (
        type(source_hash) is not str
        or _SHA256_RE.fullmatch(source_hash) is None
        or source_hash != _binding_list_hash(snapshot.bindings)
    ):
        _fail("owner attestation source_set_sha256 is invalid")
    trust = value["trust_boundary"]
    if (
        type(trust) is not dict
        or trust.get("model") != TRUST_MODEL
        or trust.get("trusted_single_writer") is not True
        or trust.get("hostile_concurrent_replacement_resistance_claimed") is not False
        or trust.get("no_overwrite_required") is not True
    ):
        _fail("owner attestation trust boundary is invalid")
    return value


def _binding_for_bytes(path: Path, raw: bytes, *, media_type: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "media_type": media_type,
    }


def _write_no_clobber(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        guarded = transcript_fs_guard.guard_output_path(path)
    except transcript_fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptDerivedVadError(f"unsafe output path: {exc}") from exc
    if guarded.exists():
        _fail(f"refusing to overwrite existing artifact: {guarded}")
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{guarded.name}.tmp.",
            suffix=guarded.suffix,
            dir=guarded.parent,
            delete=False,
        ) as handle:
            staged = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staged, guarded)
        except FileExistsError as exc:
            raise TranscriptDerivedVadError(f"refusing concurrent overwrite: {guarded}") from exc
        if guarded.read_bytes() != raw:
            _fail("published bytes changed")
        return guarded
    except BaseException:
        if staged is not None and staged.exists() and guarded.exists():
            try:
                if os.path.samefile(staged, guarded):
                    guarded.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        if staged is not None and staged.exists():
            staged.unlink(missing_ok=True)


def _write_owner_attestation_with_authority(
    scene: str,
    *,
    owner: str,
    attested_at: str,
    authority: TranscriptDerivedVadAuthority,
) -> Path:
    snapshot = _load_source_snapshot(scene, authority=authority)
    path = canonical_attestation_path(scene, authority)
    raw = _canonical_json_bytes(
        _attestation_payload(snapshot, owner=owner, attested_at=attested_at)
    )
    _write_no_clobber(path, raw)
    _load_attestation_with_snapshot(path, snapshot=snapshot, authority=authority)
    return path


def write_owner_attestation(
    scene: str,
    *,
    owner: str,
    attested_at: str,
) -> Path:
    """Publish the owner's existing-annotation statement at the canonical path."""

    return _write_owner_attestation_with_authority(
        scene,
        owner=owner,
        attested_at=attested_at,
        authority=default_transcript_derived_vad_authority(),
    )


def _load_attestation_with_snapshot(
    path: str | Path,
    *,
    snapshot: _SourceSnapshot,
    authority: TranscriptDerivedVadAuthority,
) -> tuple[dict[str, Any], bytes]:
    guarded = transcript_fs_guard.lexical_absolute(path)
    binding, raw = _guarded_binding(guarded, media_type="application/json")
    del binding
    value = _validate_attestation_value(
        _strict_json_loads(raw, "owner attestation"),
        snapshot=snapshot,
        path=guarded,
        authority=authority,
    )
    return value, raw


def _normalise_event_overlay(value: Any, *, label: str, require_positive: bool) -> dict[str, str]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact event object")
    _require_exact_keys(value, set(_EVENT_FIELDS), label)
    start = value["start_timecode"]
    end = value["end_timecode"]
    if type(start) is not str or type(end) is not str:
        _fail(f"{label} timecodes must be strings")
    start_frame = transcript_acceptance.parse_timecode_frames(start)
    end_frame = transcript_acceptance.parse_timecode_frames(end)
    if require_positive and start_frame >= end_frame:
        _fail(f"{label} must be a positive interval")
    speaker = value["speaker"]
    if type(speaker) is not str or speaker not in SPEAKERS:
        _fail(f"{label} speaker must be A/B/C/R")
    text = value["text"]
    if type(text) is not str or not text.strip() or text != text.strip():
        _fail(f"{label} text must be a non-empty trimmed string")
    return {field: value[field] for field in _EVENT_FIELDS}


def _validate_candidate_preimages(snapshot: _SourceSnapshot) -> None:
    try:
        timeline, _ = transcript_acceptance._apply_repair_operations(
            snapshot.timeline_raw,
            {"operations": _mechanical_repairs(snapshot.scene)},
        )
    except ValueError as exc:
        raise TranscriptDerivedVadError(f"mechanical exact-preimage replay failed: {exc}") from exc
    for candidate in _candidate_catalog(snapshot.scene):
        if type(candidate) is not dict:
            _fail("candidate catalog entries must be exact objects")
        _require_exact_keys(
            candidate,
            {
                "candidate_id",
                "source_index",
                "expected_event",
                "proposed_event",
                "timeline_local_clip_start_s",
                "timeline_local_clip_end_s_exclusive",
            },
            "candidate",
        )
        candidate_id = candidate["candidate_id"]
        if (
            type(candidate_id) is not str
            or not candidate_id
            or candidate_id != candidate_id.strip()
        ):
            _fail("candidate_id must be a non-empty trimmed string")
        index = candidate["source_index"]
        if type(index) is not int or index < 0 or index >= len(timeline):
            _fail(f"candidate {candidate_id} source index does not exist")
        expected = _normalise_event_overlay(
            candidate["expected_event"],
            label=f"candidate {candidate_id} expected_event",
            require_positive=False,
        )
        proposed = _normalise_event_overlay(
            candidate["proposed_event"],
            label=f"candidate {candidate_id} proposed_event",
            require_positive=True,
        )
        event = timeline[index]
        actual = (
            {field: event.get(field) for field in _EVENT_FIELDS} if type(event) is dict else None
        )
        if not _type_exact_equal(actual, expected):
            _fail(f"candidate {candidate_id} exact event preimage changed")
        if _type_exact_equal(expected, proposed):
            _fail(f"candidate {candidate_id} proposal does not change the event")


def _normalise_decisions(
    scene: str, decisions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        _fail("semantic decisions must be a sequence of decision objects")
    catalog = _candidate_catalog(scene)
    if len(decisions) != len(catalog):
        _fail("semantic decisions must cover exactly the scene's targeted candidates")
    result: list[dict[str, Any]] = []
    for index, (raw, candidate) in enumerate(zip(decisions, catalog, strict=True)):
        if type(raw) is not dict:
            _fail(f"decision {index} must be an exact object")
        _require_exact_keys(
            raw,
            {"candidate_id", "status", "replacement", "note"},
            f"decision {index}",
        )
        if raw["candidate_id"] != candidate["candidate_id"]:
            _fail("semantic decisions are missing, duplicated, or reordered")
        status = raw["status"]
        if status not in {"accepted", "rejected", "replaced"}:
            _fail(f"decision {candidate['candidate_id']} status is unsupported")
        note = raw["note"]
        if type(note) is not str or not note.strip() or note != note.strip():
            _fail(f"decision {candidate['candidate_id']} requires a trimmed human note")
        replacement = raw["replacement"]
        if status in {"accepted", "rejected"}:
            if replacement is not None:
                _fail(f"{status} decision replacement must be JSON null")
            normalized_replacement = None
        else:
            normalized_replacement = _normalise_event_overlay(
                replacement,
                label=f"decision {candidate['candidate_id']} replacement",
                require_positive=True,
            )
            if any(
                _type_exact_equal(normalized_replacement, candidate[key])
                for key in ("expected_event", "proposed_event")
            ):
                _fail("replacement event must differ from both source and proposed events")
        result.append(
            {
                "candidate_id": raw["candidate_id"],
                "status": status,
                "replacement": normalized_replacement,
                "note": note,
            }
        )
    return result


def _decisions_payload(
    snapshot: _SourceSnapshot,
    *,
    attestation_path: Path,
    attestation_raw: bytes,
    reviewer: str,
    decided_at: str,
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reviewer, decided_at = _validate_identity_and_time(
        reviewer, decided_at, label="semantic decisions"
    )
    _validate_candidate_preimages(snapshot)
    normalized = _normalise_decisions(snapshot.scene, decisions)
    return {
        "artifact_kind": DECISIONS_KIND,
        "format_version": DECISIONS_FORMAT_VERSION,
        "scene": snapshot.scene,
        "authority_scope": snapshot.authority_scope,
        "targeted_reaudit_only": True,
        "all_events_reaudited": False,
        "reviewer": {"identity": reviewer, "decided_at": decided_at},
        "owner_attestation": _binding_for_bytes(
            attestation_path, attestation_raw, media_type="application/json"
        ),
        "timeline_sha256": _sha256_bytes(snapshot.timeline_raw),
        "candidate_catalog": _candidate_catalog(snapshot.scene),
        "candidate_catalog_sha256": _candidate_catalog_sha256(snapshot.scene),
        "decisions": normalized,
    }


def _write_semantic_decisions_with_authority(
    scene: str,
    *,
    reviewer: str,
    decided_at: str,
    decisions: Sequence[Mapping[str, Any]],
    authority: TranscriptDerivedVadAuthority,
) -> Path:
    snapshot = _load_source_snapshot(scene, authority=authority)
    attestation_path = canonical_attestation_path(scene, authority)
    _attestation, attestation_raw = _load_attestation_with_snapshot(
        attestation_path, snapshot=snapshot, authority=authority
    )
    payload = _decisions_payload(
        snapshot,
        attestation_path=attestation_path,
        attestation_raw=attestation_raw,
        reviewer=reviewer,
        decided_at=decided_at,
        decisions=decisions,
    )
    path = canonical_decisions_path(scene, authority)
    raw = _canonical_json_bytes(payload)
    _write_no_clobber(path, raw)
    _load_decisions_with_snapshot(path, snapshot=snapshot, authority=authority)
    return path


def write_semantic_decisions(
    scene: str,
    *,
    reviewer: str,
    decided_at: str,
    decisions: Sequence[Mapping[str, Any]],
) -> Path:
    """Publish explicit accept/reject/replacement decisions for one scene."""

    return _write_semantic_decisions_with_authority(
        scene,
        reviewer=reviewer,
        decided_at=decided_at,
        decisions=decisions,
        authority=default_transcript_derived_vad_authority(),
    )


def _load_decisions_with_snapshot(
    path: str | Path,
    *,
    snapshot: _SourceSnapshot,
    authority: TranscriptDerivedVadAuthority,
) -> tuple[dict[str, Any], bytes]:
    guarded = transcript_fs_guard.lexical_absolute(path)
    if authority.production and guarded != canonical_decisions_path(snapshot.scene, authority):
        _fail("production semantic decisions were copied or re-anchored")
    _binding, raw = _guarded_binding(guarded, media_type="application/json")
    value = _strict_json_loads(raw, "semantic decisions")
    if type(value) is not dict:
        _fail("semantic decisions must be one JSON object")
    _require_exact_keys(
        value,
        {
            "artifact_kind",
            "format_version",
            "scene",
            "authority_scope",
            "targeted_reaudit_only",
            "all_events_reaudited",
            "reviewer",
            "owner_attestation",
            "timeline_sha256",
            "candidate_catalog",
            "candidate_catalog_sha256",
            "decisions",
        },
        "semantic decisions",
    )
    if (
        value["artifact_kind"] != DECISIONS_KIND
        or type(value["format_version"]) is not int
        or value["format_version"] != DECISIONS_FORMAT_VERSION
        or value["scene"] != snapshot.scene
        or value["authority_scope"] != snapshot.authority_scope
        or value["targeted_reaudit_only"] is not True
        or value["all_events_reaudited"] is not False
    ):
        _fail("semantic decisions role/version/scope is invalid")
    reviewer = value["reviewer"]
    if type(reviewer) is not dict:
        _fail("semantic decisions reviewer must be an object")
    _require_exact_keys(reviewer, {"identity", "decided_at"}, "semantic reviewer")
    _validate_identity_and_time(
        reviewer["identity"], reviewer["decided_at"], label="semantic decisions"
    )
    attestation_path = canonical_attestation_path(snapshot.scene, authority)
    _attestation, attestation_raw = _load_attestation_with_snapshot(
        attestation_path, snapshot=snapshot, authority=authority
    )
    expected = _decisions_payload(
        snapshot,
        attestation_path=attestation_path,
        attestation_raw=attestation_raw,
        reviewer=reviewer["identity"],
        decided_at=reviewer["decided_at"],
        decisions=value["decisions"],
    )
    if not _type_exact_equal(value, expected):
        _fail("semantic decisions do not replay from exact sources/attestation")
    return value, raw


def _semantic_operations(
    scene: str, decisions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    catalog = _candidate_catalog(scene)
    normalized = _normalise_decisions(scene, decisions)
    operation_for_field = {
        "start_timecode": "correct_timecode_boundary",
        "end_timecode": "correct_timecode_boundary",
        "speaker": "correct_speaker_identity",
        "text": "correct_text_content",
    }
    operations: list[dict[str, Any]] = []
    for candidate, decision in zip(catalog, normalized, strict=True):
        if decision["status"] == "rejected":
            continue
        destination = (
            candidate["proposed_event"]
            if decision["status"] == "accepted"
            else decision["replacement"]
        )
        assert type(destination) is dict
        for event_field in _EVENT_FIELDS:
            expected = candidate["expected_event"][event_field]
            replacement = destination[event_field]
            if expected == replacement:
                continue
            operations.append(
                {
                    "repair_id": f"{candidate['candidate_id']}-{event_field}",
                    "operation": operation_for_field[event_field],
                    "source_index": candidate["source_index"],
                    "field": event_field,
                    "expected": expected,
                    "replacement": replacement,
                    "classification": "manual_owner",
                    "evidence": decision["note"],
                    "review_status": "confirmed",
                }
            )
    return operations


def _accepted_material(
    snapshot: _SourceSnapshot,
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[transcript_acceptance.TranscriptEvent, ...],
    dict[str, tuple[tuple[int, int], ...]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    mechanical = _mechanical_repairs(snapshot.scene)
    semantic = _semantic_operations(snapshot.scene, decisions)
    try:
        mechanically_repaired, _ = transcript_acceptance._apply_repair_operations(
            snapshot.timeline_raw, {"operations": mechanical}
        )
        (
            in_scope_indices,
            excluded_inventory,
            scope_summary,
            _prevalence,
        ) = transcript_acceptance._scope_material(
            mechanically_repaired,
            transcript_offset_sec=snapshot.transcript_offset_sec,
            authorized_start_tick=snapshot.start_tick,
            authorized_end_tick=snapshot.end_tick,
        )
        for candidate in _candidate_catalog(snapshot.scene):
            if candidate["source_index"] not in in_scope_indices:
                _fail(f"targeted candidate {candidate['candidate_id']} is no longer in-window")
        all_repairs = [*mechanical, *semantic]
        repaired, repair_ids = transcript_acceptance._apply_repair_operations(
            snapshot.timeline_raw, {"operations": all_repairs}
        )
        events, intervals = transcript_acceptance._canonical_events(
            repaired,
            reference_duration_sec=snapshot.reference_duration_sec,
            source_indices=in_scope_indices,
            transcript_offset_sec=snapshot.transcript_offset_sec,
            authorized_start_s=Decimal(snapshot.start_tick) / FRAME_RATE_HZ,
            authorized_end_s=Decimal(snapshot.end_tick) / FRAME_RATE_HZ,
        )
    except TranscriptDerivedVadError:
        raise
    except ValueError as exc:
        raise TranscriptDerivedVadError(
            "resolved Transcript events are not valid positive in-window intervals; "
            f"a rejected impossible candidate may require an explicit replacement: {exc}"
        ) from exc
    if not events:
        _fail("resolved Transcript contains no event in the analysis window")
    for event in events:
        canonical_slot = PARTICIPANT_TO_SLOT[event.speaker]
        if (
            snapshot.participant_to_slot.get(event.speaker) != canonical_slot
            or snapshot.present_by_slot.get(canonical_slot) != 1
        ):
            _fail(
                "accepted in-window Transcript event references an absent participant: "
                f"source_index={event.source_index}, speaker={event.speaker}, "
                f"canonical_slot={canonical_slot}"
            )
    material = {
        "events": [
            {
                "source_index": event.source_index,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "speaker": event.speaker,
                "text": event.text,
            }
            for event in events
        ],
        "repairs": all_repairs,
        "repair_ids": list(repair_ids),
        "scope": scope_summary,
        "excluded_event_inventory": list(excluded_inventory),
    }
    return events, intervals, material, repaired


def _ceil_shared_tick(frame: int, offset: Decimal) -> int:
    shared = (Decimal(frame) / transcript_acceptance.TIMECODE_FPS_NUMERATOR) + offset
    return int((shared * FRAME_RATE_HZ).to_integral_value(rounding=ROUND_CEILING))


def _rasterize(
    snapshot: _SourceSnapshot,
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, np.ndarray]:
    n_rows = snapshot.end_tick - snapshot.start_tick
    result = {speaker: np.zeros(n_rows, dtype=np.int8) for speaker in SPEAKERS}
    for speaker in SPEAKERS:
        slot = PARTICIPANT_TO_SLOT[speaker]
        if snapshot.present_by_slot[slot] == 0:
            if intervals[speaker]:
                _fail(f"accepted intervals reference absent participant {speaker}/{slot}")
            continue
        for start_frame, end_frame in intervals[speaker]:
            start = max(
                snapshot.start_tick,
                _ceil_shared_tick(start_frame, snapshot.transcript_offset_sec),
            )
            end = min(
                snapshot.end_tick,
                _ceil_shared_tick(end_frame, snapshot.transcript_offset_sec),
            )
            if end > start:
                result[speaker][start - snapshot.start_tick : end - snapshot.start_tick] = 1
    for speaker in SPEAKERS:
        slot = PARTICIPANT_TO_SLOT[speaker]
        if snapshot.present_by_slot[slot] == 0 and np.any(result[speaker]):
            _fail(f"absent participant {speaker}/{slot} received non-zero VAD")
    return result


def _overlap_qc(
    events: Sequence[transcript_acceptance.TranscriptEvent],
    repaired_timeline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_order_regressions = 0
    previous_start: int | None = None
    for event in repaired_timeline:
        start = transcript_acceptance.parse_timecode_frames(event["start_timecode"])
        if previous_start is not None and start < previous_start:
            source_order_regressions += 1
        previous_start = start
    same_speaker = 0
    cross_speaker = 0
    ordered = sorted(events, key=lambda item: (item.start_frame, item.end_frame))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if right.start_frame >= left.end_frame:
                break
            if right.end_frame > left.start_frame:
                if right.speaker == left.speaker:
                    same_speaker += 1
                else:
                    cross_speaker += 1
    return {
        "source_order_regression_count": source_order_regressions,
        "same_speaker_overlap_pair_count_before_union": same_speaker,
        "cross_speaker_overlap_pair_count_preserved": cross_speaker,
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def _binding_for_existing_artifact(
    path: Path,
    raw: bytes,
    *,
    media_type: str,
    artifact_kind: str,
    format_version: int = FORMAT_VERSION,
) -> dict[str, Any]:
    return {
        **_binding_for_bytes(path, raw, media_type=media_type),
        "artifact_kind": artifact_kind,
        "format_version": format_version,
    }


def _build_with_snapshot(
    snapshot: _SourceSnapshot,
    *,
    attestation_path: Path,
    attestation_raw: bytes,
    decisions_path: Path | None,
    decisions_raw: bytes | None,
    decisions_value: Mapping[str, Any] | None,
    authority: TranscriptDerivedVadAuthority,
) -> TranscriptDerivedVadResult:
    decisions = [] if decisions_value is None else decisions_value["decisions"]
    if _candidate_catalog(snapshot.scene) and decisions_value is None:
        _fail("this scene requires explicit targeted semantic decisions")
    events, intervals, material, repaired_timeline = _accepted_material(snapshot, decisions)
    values = _rasterize(snapshot, intervals)
    ticks = np.arange(snapshot.start_tick, snapshot.end_tick, dtype=np.int64)
    source_df = pd.DataFrame(
        {
            "session_id": [snapshot.scene] * len(ticks),
            "timestamp_s": ticks.astype(np.float64) / FRAME_RATE_HZ,
            **{
                f"{slot}_present": np.full(
                    len(ticks), snapshot.present_by_slot[slot], dtype=np.int8
                )
                for slot in ("P1", "P2", "P3", "P4")
            },
            **{
                f"{slot}_vad": values[SLOT_TO_PARTICIPANT[slot]]
                for slot in ("P1", "P2", "P3", "P4")
            },
        }
    )
    target_columns = [
        "session_id",
        "timestamp_s",
        "P1_vad",
        "P2_vad",
        "P3_vad",
        "P4_vad",
    ]
    target_df = source_df[target_columns].copy()
    for slot in ("P1", "P2", "P3", "P4"):
        values_set = set(target_df[f"{slot}_vad"].tolist())
        if not values_set.issubset({0, 1}):
            _fail(f"{slot}_vad is not exact binary")
        if snapshot.present_by_slot[slot] == 0 and values_set != {0}:
            _fail(f"absent {slot} must remain an explicit all-zero target")
    source_raw = _csv_bytes(source_df)
    target_raw = _csv_bytes(target_df)
    paths = _target_paths(snapshot.scene, authority)
    attestation_binding = _binding_for_existing_artifact(
        attestation_path,
        attestation_raw,
        media_type="application/json",
        artifact_kind=ATTESTATION_KIND,
    )
    decisions_binding = (
        None
        if decisions_path is None or decisions_raw is None
        else _binding_for_existing_artifact(
            decisions_path,
            decisions_raw,
            media_type="application/json",
            artifact_kind=DECISIONS_KIND,
            format_version=DECISIONS_FORMAT_VERSION,
        )
    )
    speaker_counts = {
        speaker: sum(event.speaker == speaker for event in events) for speaker in SPEAKERS
    }
    active_ticks = {speaker: int(values[speaker].sum()) for speaker in SPEAKERS}
    degenerate_present = [
        f"{slot}_vad"
        for slot in ("P1", "P2", "P3", "P4")
        if snapshot.present_by_slot[slot] == 1 and set(target_df[f"{slot}_vad"].tolist()) != {0, 1}
    ]
    qc = {
        "automatic_structural_qc_pass": True,
        "source_event_count": material["scope"]["source_event_count"],
        "in_window_event_count": len(events),
        "excluded_event_count": material["scope"]["excluded_event_count"],
        "crossing_event_count": material["scope"]["crossing_event_count"],
        "speaker_event_counts": speaker_counts,
        "speaker_union_interval_counts": {speaker: len(intervals[speaker]) for speaker in SPEAKERS},
        "speaker_active_tick_counts": active_ticks,
        "degenerate_present_vad_columns_diagnostic": degenerate_present,
        **_overlap_qc(events, repaired_timeline),
        "hash_replay_pass": True,
        "window_exact_match": True,
        "binary_values_pass": True,
        "absent_slot_zero_pass": True,
    }
    counted_ready = snapshot.authority_scope == "production"
    contract = {
        "artifact_kind": CONTRACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": snapshot.scene,
        "authority_scope": snapshot.authority_scope,
        "output_role": OUTPUT_ROLE,
        "research_label": RESEARCH_LABEL,
        "assembly_eligible": counted_ready,
        "counted_gold_ready": counted_ready,
        "source_kind": "human_reviewed_transcript_segments",
        "claims": {
            "existing_human_annotation": True,
            "targeted_reaudit_only": True,
            "all_events_reaudited": False,
            "sample_perfect_lavalier_vad_claimed": False,
            "lavalier_boundary_refinement_applied": False,
        },
        "timebase": {
            "axis": "mic_A_shared_seconds",
            "formula": transcript_acceptance.SHARED_TIME_FORMULA,
            "timecode_fps_numerator": 30,
            "timecode_fps_denominator": 1,
            "drop_frame": False,
            "timecode_quantization_error_bound_ms": str(
                transcript_acceptance.TIMECODE_QUANTIZATION_ERROR_BOUND_MS
            ),
            "output_frame_rate_hz": FRAME_RATE_HZ,
            "raster_rule": "active iff start_shared <= tick/25 < end_shared",
            "same_speaker_intervals": "half_open_union_before_rasterization",
            "analysis_window_only": True,
            "outside_window_filled_as_silence": False,
        },
        "known_label_noise": {
            "within_utterance_pauses_may_remain_active": True,
            "thirty_fps_boundary_quantization": True,
            "consequence": (
                "labels are reliable human-reviewed utterance activity, not "
                "sample-accurate physiological voice activity"
            ),
        },
        "participant_to_slot": dict(snapshot.participant_to_slot),
        "present_by_slot": dict(snapshot.present_by_slot),
        "authorized_tick_range": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_inclusive": snapshot.start_tick,
            "end_exclusive": snapshot.end_tick,
            "n_ticks": snapshot.end_tick - snapshot.start_tick,
        },
        "owner_attestation": attestation_binding,
        "targeted_semantic_decisions": decisions_binding,
        "mechanical_repairs": _mechanical_repairs(snapshot.scene),
        "semantic_decisions": list(decisions),
        "accepted_semantics_sha256": _sha256_bytes(_canonical_json_bytes(material).rstrip(b"\n")),
        "source_bindings": list(snapshot.bindings),
        "source_set_sha256": _binding_list_hash(snapshot.bindings),
        "source_csv": _binding_for_bytes(paths["source"], source_raw, media_type="text/csv"),
        "target_csv": _binding_for_bytes(paths["target"], target_raw, media_type="text/csv"),
        "structural_qc": qc,
        "audio_consistency_diagnostics": {
            "formal_channel_time_map_replayed": True,
            "new_signal_threshold_applied": False,
            "hard_scientific_gate": False,
            "reason": (
                "owner accepted the existing human annotation; no arbitrary "
                "last-minute audio threshold is introduced"
            ),
        },
        "trust_boundary": {
            "model": TRUST_MODEL,
            "trusted_single_writer": True,
            "canonical_no_overwrite": True,
            "multi_file_os_atomicity_claimed": False,
            "hostile_concurrent_replacement_resistance_claimed": False,
            "writer_source_replay": "one_fresh_full_replay_before_publication",
            "writer_postpublication_verification": (
                "local_four_file_exact_bytes_sha256_and_schema_only"
            ),
            "writer_postpublication_fresh_source_replay_claimed": False,
            "standalone_loader_fresh_source_replay_required": True,
            "residual_concurrent_source_change_risk_accepted": True,
        },
    }
    contract_raw = _canonical_json_bytes(contract)
    participant_by_slot = {
        slot: participant for participant, slot in snapshot.participant_to_slot.items()
    }
    vad_columns = [f"P{slot}_vad" for slot in range(1, 5)]
    source_name = paths["source"].name
    source_hash = _sha256_bytes(source_raw)
    manifest = {
        "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
        "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
        "session_id": snapshot.scene,
        "participant_to_slot": dict(snapshot.participant_to_slot),
        "supplied_columns": vad_columns,
        "feature_csv_sha256": _sha256_bytes(target_raw),
        "source_files": [{"path": source_name, "sha256": source_hash}],
        "purpose": "gold_target",
        "causality": {
            "causal": False,
            "lookahead_s": None,
            "window_alignment": "offline",
        },
        "column_sources": {
            column: {
                "participant": participant_by_slot.get(column.split("_", 1)[0]),
                "slot": column.split("_", 1)[0],
                "source_path": source_name,
                "source_sha256": source_hash,
                "source_column": column,
                "source_participant": participant_by_slot.get(column.split("_", 1)[0]),
                "source_keys": {
                    "session_id": "session_id",
                    "timestamp_s": "timestamp_s",
                },
                "value_transform": {"kind": "identity"},
            }
            for column in vad_columns
        },
        "target_contract": _binding_for_existing_artifact(
            paths["contract"],
            contract_raw,
            media_type="application/json",
            artifact_kind=CONTRACT_KIND,
        ),
        "authority_scope": snapshot.authority_scope,
        "research_label": RESEARCH_LABEL,
        "assembly_eligible": counted_ready,
        "counted_gold_ready": counted_ready,
    }
    return TranscriptDerivedVadResult(
        source_df=source_df,
        target_df=target_df,
        contract=contract,
        manifest=manifest,
    )


def _build_with_loaded_snapshot(
    snapshot: _SourceSnapshot,
    *,
    authority: TranscriptDerivedVadAuthority,
) -> TranscriptDerivedVadResult:
    scene = snapshot.scene
    attestation_path = canonical_attestation_path(scene, authority)
    _attestation, attestation_raw = _load_attestation_with_snapshot(
        attestation_path, snapshot=snapshot, authority=authority
    )
    if _candidate_catalog(scene):
        decisions_path: Path | None = canonical_decisions_path(scene, authority)
        decisions_value, decisions_raw = _load_decisions_with_snapshot(
            decisions_path, snapshot=snapshot, authority=authority
        )
    else:
        decisions_path = None
        decisions_value = None
        decisions_raw = None
    return _build_with_snapshot(
        snapshot,
        attestation_path=attestation_path,
        attestation_raw=attestation_raw,
        decisions_path=decisions_path,
        decisions_raw=decisions_raw,
        decisions_value=decisions_value,
        authority=authority,
    )


def _build_with_authority(
    scene: str,
    *,
    authority: TranscriptDerivedVadAuthority,
) -> TranscriptDerivedVadResult:
    snapshot = _load_source_snapshot(scene, authority=authority)
    return _build_with_loaded_snapshot(snapshot, authority=authority)


def build_transcript_derived_vad(scene: str) -> TranscriptDerivedVadResult:
    """Build one production target in memory without writing it."""

    return _build_with_authority(scene, authority=default_transcript_derived_vad_authority())


def _remove_if_exact(path: Path, raw: bytes) -> None:
    try:
        if path.is_file() and path.read_bytes() == raw:
            path.unlink()
    except OSError:
        pass


def _verify_published_target_bundle_locally(
    scene: str,
    *,
    authority: TranscriptDerivedVadAuthority,
    expected: TranscriptDerivedVadResult,
    payloads: Sequence[tuple[Path, bytes]],
) -> TranscriptDerivedVadResult:
    """Verify only the four just-published files against their in-memory payloads."""

    paths = _target_paths(scene, authority)
    expected_paths = {
        paths["source"]: ("source CSV", "text/csv"),
        paths["target"]: ("target CSV", "text/csv"),
        paths["contract"]: ("contract", "application/json"),
        paths["manifest"]: ("manifest", "application/json"),
    }
    supplied_payloads = dict(payloads)
    if len(supplied_payloads) != len(payloads) or set(supplied_payloads) != set(expected_paths):
        _fail("local publication verification requires exactly the canonical four-file bundle")

    actual: dict[Path, bytes] = {}
    for path, (label, media_type) in expected_paths.items():
        binding, raw = _guarded_binding(path, media_type=media_type)
        staged = supplied_payloads[path]
        if raw != staged:
            _fail(f"published VAD {label} differs from staged bytes")
        if binding["size_bytes"] != len(staged) or binding["sha256"] != _sha256_bytes(staged):
            _fail(f"published VAD {label} size/hash differs from staged payload")
        actual[path] = raw

    source_raw = actual[paths["source"]]
    target_raw = actual[paths["target"]]
    contract_raw = actual[paths["contract"]]
    manifest_raw = actual[paths["manifest"]]
    parsed_contract = _strict_json_loads(contract_raw, "target contract")
    parsed_manifest = _strict_json_loads(manifest_raw, "target manifest")
    if not _type_exact_equal(parsed_contract, expected.contract):
        _fail("published VAD contract differs from staged schema")
    if not _type_exact_equal(parsed_manifest, expected.manifest):
        _fail("published VAD manifest differs from staged schema")
    if parsed_manifest.get("feature_csv_sha256") != _sha256_bytes(target_raw):
        _fail("published VAD manifest does not hash-bind the target CSV")
    if parsed_manifest.get("source_files") != [
        {"path": paths["source"].name, "sha256": _sha256_bytes(source_raw)}
    ]:
        _fail("published VAD manifest does not hash-bind the source CSV")
    try:
        source_df = pd.read_csv(io.BytesIO(source_raw))
        target_df = pd.read_csv(io.BytesIO(target_raw))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise TranscriptDerivedVadError(f"published VAD CSV is invalid: {exc}") from exc
    if not source_df.equals(expected.source_df.astype(source_df.dtypes.to_dict())):
        _fail("published VAD source CSV dtype/value round-trip changed")
    if not target_df.equals(expected.target_df.astype(target_df.dtypes.to_dict())):
        _fail("published VAD target CSV dtype/value round-trip changed")
    return TranscriptDerivedVadResult(
        source_df=expected.source_df,
        target_df=expected.target_df,
        contract=expected.contract,
        manifest=expected.manifest,
        source_csv=paths["source"],
        target_csv=paths["target"],
    )


def _write_target_bundle_with_authority(
    scene: str,
    *,
    authority: TranscriptDerivedVadAuthority,
) -> TranscriptDerivedVadResult:
    result = _build_with_authority(scene, authority=authority)
    paths = _target_paths(scene, authority)
    source_raw = _csv_bytes(result.source_df)
    target_raw = _csv_bytes(result.target_df)
    contract_raw = _canonical_json_bytes(result.contract)
    manifest_raw = _canonical_json_bytes(result.manifest)
    payloads = (
        (paths["source"], source_raw),
        (paths["target"], target_raw),
        (paths["contract"], contract_raw),
        (paths["manifest"], manifest_raw),
    )
    if any(path.exists() for path, _raw in payloads):
        _fail("refusing to overwrite any member of the canonical target bundle")
    published: list[tuple[Path, bytes]] = []
    try:
        for path, raw in payloads:
            _write_no_clobber(path, raw)
            published.append((path, raw))
        loaded = _verify_published_target_bundle_locally(
            scene,
            authority=authority,
            expected=result,
            payloads=payloads,
        )
        return loaded
    except BaseException:
        # The owner explicitly selected a trusted, single-writer local workspace.
        # Rollback removes only byte-identical files written by this invocation.
        for path, raw in reversed(published):
            _remove_if_exact(path, raw)
        raise


def write_transcript_derived_vad(scene: str) -> TranscriptDerivedVadResult:
    """Publish one canonical source/target/contract/manifest bundle, no-clobber."""

    return _write_target_bundle_with_authority(
        scene, authority=default_transcript_derived_vad_authority()
    )


def _load_target_bundle_with_authority(
    target_csv: str | Path,
    *,
    authority: TranscriptDerivedVadAuthority,
) -> TranscriptDerivedVadResult:
    target_path = transcript_fs_guard.lexical_absolute(target_csv)
    target_binding, target_raw = _guarded_binding(target_path, media_type="text/csv")
    del target_binding
    try:
        scene = target_path.parent.name
        _validate_scene(scene)
    except TranscriptDerivedVadError:
        raise
    paths = _target_paths(scene, authority)
    if target_path != paths["target"]:
        _fail("target CSV was copied, renamed, or re-anchored")
    source_binding, source_raw = _guarded_binding(paths["source"], media_type="text/csv")
    contract_binding, contract_raw = _guarded_binding(
        paths["contract"], media_type="application/json"
    )
    manifest_binding, manifest_raw = _guarded_binding(
        paths["manifest"], media_type="application/json"
    )
    del source_binding, contract_binding, manifest_binding
    parsed_contract = _strict_json_loads(contract_raw, "target contract")
    parsed_manifest = _strict_json_loads(manifest_raw, "target manifest")
    try:
        source_df = pd.read_csv(io.BytesIO(source_raw))
        target_df = pd.read_csv(io.BytesIO(target_raw))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise TranscriptDerivedVadError(f"published VAD CSV is invalid: {exc}") from exc
    # Standalone consumption/audit always performs one fresh full source replay.
    # The deadline-oriented writer deliberately uses a separate local-only verifier
    # after publication under the accepted trusted single-writer boundary.
    snapshot = _load_source_snapshot(scene, authority=authority)
    expected = _build_with_loaded_snapshot(snapshot, authority=authority)
    if source_raw != _csv_bytes(expected.source_df):
        _fail("published VAD source CSV differs from deterministic replay")
    if target_raw != _csv_bytes(expected.target_df):
        _fail("published VAD target CSV differs from deterministic replay")
    if not _type_exact_equal(parsed_contract, expected.contract):
        _fail("published VAD contract differs from deterministic replay")
    if not _type_exact_equal(parsed_manifest, expected.manifest):
        _fail("published VAD manifest differs from deterministic replay")
    if not source_df.equals(expected.source_df.astype(source_df.dtypes.to_dict())):
        _fail("published VAD source CSV dtype/value round-trip changed")
    if not target_df.equals(expected.target_df.astype(target_df.dtypes.to_dict())):
        _fail("published VAD target CSV dtype/value round-trip changed")
    try:
        assembly._validate_feature_manifest(
            paths["target"],
            ["P1_vad", "P2_vad", "P3_vad", "P4_vad"],
            snapshot.frame_df,
            snapshot.analysis_provenance,
            forbidden_generated_paths={
                _canonical_source_paths(scene, authority)["analysis_csv"],
                _canonical_source_paths(scene, authority)["analysis_sidecar"],
            },
        )
    except (OSError, ValueError) as exc:
        raise TranscriptDerivedVadError(f"assembly target-manifest replay failed: {exc}") from exc
    return TranscriptDerivedVadResult(
        source_df=expected.source_df,
        target_df=expected.target_df,
        contract=expected.contract,
        manifest=expected.manifest,
        source_csv=paths["source"],
        target_csv=paths["target"],
    )


def load_transcript_derived_vad(
    target_csv: str | Path,
) -> TranscriptDerivedVadResult:
    """Replay a canonical production target from live sources and decisions."""

    return _load_target_bundle_with_authority(
        target_csv, authority=default_transcript_derived_vad_authority()
    )


def _candidate_by_id(candidate_id: str) -> tuple[str, dict[str, Any]]:
    if type(candidate_id) is not str or not candidate_id:
        _fail("candidate_id must be a non-empty string")
    matches = [
        (scene, candidate)
        for scene, candidates in SEMANTIC_CANDIDATES.items()
        for candidate in candidates
        if candidate["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        _fail("candidate_id is not one of the four targeted semantic candidates")
    scene, candidate = matches[0]
    return scene, dict(candidate)


def _wav_clip_bytes(
    path: Path,
    *,
    local_start_s: Decimal,
    local_end_s: Decimal,
) -> tuple[bytes, np.ndarray, int, dict[str, Any]]:
    binding, raw = _guarded_binding(path, media_type="audio/wav")
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            n_frames = handle.getnframes()
            if channels != 1 or width != 2 or rate <= 0:
                _fail("candidate clips require mono PCM16 lavalier WAVs")
            start = int((local_start_s * rate).to_integral_value(rounding=ROUND_CEILING))
            end = int((local_end_s * rate).to_integral_value(rounding=ROUND_CEILING))
            if start < 0 or end <= start or end > n_frames:
                _fail("candidate clip escapes real correction-mapped audio")
            handle.setpos(start)
            payload = handle.readframes(end - start)
    except (EOFError, wave.Error) as exc:
        raise TranscriptDerivedVadError(f"invalid lavalier WAV {path}: {exc}") from exc
    samples = np.frombuffer(payload, dtype="<i2").copy()
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.astype("<i2", copy=False).tobytes())
    detail = {
        "source": binding,
        "sample_rate_hz": rate,
        "start_sample_inclusive": start,
        "end_sample_exclusive": end,
        "n_samples": end - start,
        "local_start_s": _canonical_decimal(local_start_s),
        "local_end_s_exclusive": _canonical_decimal(local_end_s),
    }
    return output.getvalue(), samples, rate, detail


def _export_candidate_clips_with_authority(
    candidate_id: str,
    *,
    mode: str,
    authority: TranscriptDerivedVadAuthority,
) -> Path:
    if mode not in {"separate", "mix"}:
        _fail("clip mode must be 'separate' or 'mix'")
    scene, candidate = _candidate_by_id(candidate_id)
    snapshot = _load_source_snapshot(scene, authority=authority)
    _validate_candidate_preimages(snapshot)
    local_timeline_start = _decimal(
        candidate["timeline_local_clip_start_s"], "candidate clip start"
    )
    local_timeline_end = _decimal(
        candidate["timeline_local_clip_end_s_exclusive"], "candidate clip end"
    )
    shared_start = local_timeline_start + snapshot.transcript_offset_sec
    shared_end = local_timeline_end + snapshot.transcript_offset_sec
    formal_start = Decimal(snapshot.start_tick) / FRAME_RATE_HZ
    formal_end = Decimal(snapshot.end_tick) / FRAME_RATE_HZ
    shared_start = max(shared_start, formal_start)
    shared_end = min(shared_end, formal_end)
    if shared_end <= shared_start:
        _fail("candidate listening window does not intersect the formal analysis window")
    clips: dict[str, bytes] = {}
    arrays: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    rate: int | None = None
    for speaker in SPEAKERS:
        correction = snapshot.corrections_s[speaker]
        raw, samples, speaker_rate, detail = _wav_clip_bytes(
            snapshot.lavalier_paths[speaker],
            local_start_s=shared_start - correction,
            local_end_s=shared_end - correction,
        )
        if rate is not None and rate != speaker_rate:
            _fail("candidate lavalier sample rates differ")
        rate = speaker_rate
        clips[f"{speaker}.wav"] = raw
        arrays[speaker] = samples
        details[speaker] = {
            **detail,
            "correction_s": _canonical_decimal(correction),
            "inverse_formula": "channel_local_time_s = shared_time_s - correction_s",
        }
    assert rate is not None
    if len({len(values) for values in arrays.values()}) != 1:
        _fail("correction-mapped candidate clips have unequal sample counts")
    if mode == "mix":
        stack = np.stack([arrays[speaker].astype(np.int32) for speaker in SPEAKERS])
        mixed = np.rint(stack.mean(axis=0)).clip(-32768, 32767).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(mixed.tobytes())
        output_payloads = {"mix.wav": buffer.getvalue()}
    else:
        output_payloads = clips
    output_dir = authority.clip_root / candidate_id
    if output_dir.exists():
        _fail(f"refusing to overwrite existing candidate clip directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "artifact_kind": "vap_transcript_targeted_candidate_clip_receipt",
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "candidate": candidate,
        "mode": mode,
        "read_only_source_export": True,
        "source_audio_modified": False,
        "shared_interval": {
            "axis": "mic_A_shared_seconds",
            "start_s": _canonical_decimal(shared_start),
            "end_s_exclusive": _canonical_decimal(shared_end),
        },
        "per_speaker": details,
        "outputs": {
            name: {
                "path": str(output_dir / name),
                "size_bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
            for name, raw in output_payloads.items()
        },
    }
    written: list[tuple[Path, bytes]] = []
    try:
        for name, raw in output_payloads.items():
            path = output_dir / name
            _write_no_clobber(path, raw)
            written.append((path, raw))
        receipt_raw = _canonical_json_bytes(receipt)
        receipt_path = output_dir / "receipt.v1.json"
        _write_no_clobber(receipt_path, receipt_raw)
        written.append((receipt_path, receipt_raw))
        return output_dir
    except BaseException:
        for path, raw in reversed(written):
            _remove_if_exact(path, raw)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise


def export_candidate_clips(candidate_id: str, *, mode: str = "separate") -> Path:
    """Export only one of the four correction-aware short listening windows."""

    return _export_candidate_clips_with_authority(
        candidate_id,
        mode=mode,
        authority=default_transcript_derived_vad_authority(),
    )


__all__ = [
    "ATTESTATION_KIND",
    "CONTRACT_KIND",
    "DECISIONS_KIND",
    "DECISIONS_FORMAT_VERSION",
    "FORMAT_VERSION",
    "OUTPUT_ROLE",
    "RESEARCH_LABEL",
    "SEMANTIC_CANDIDATES",
    "TranscriptDerivedVadAuthority",
    "TranscriptDerivedVadError",
    "TranscriptDerivedVadResult",
    "build_transcript_derived_vad",
    "canonical_attestation_path",
    "canonical_decisions_path",
    "default_transcript_derived_vad_authority",
    "export_candidate_clips",
    "load_transcript_derived_vad",
    "write_owner_attestation",
    "write_semantic_decisions",
    "write_transcript_derived_vad",
]
