"""Fail-closed authoring workflow for human review of Transcript events.

This module is deliberately upstream of :mod:`transcript_acceptance`.  It
creates a hash-bound *draft* queue, but it never edits the corpus timeline or
audio and never silently turns a proposal into an accepted correction.  A
final review can be emitted only after every event and semantic candidate inside
the formal gold-alignment window has an explicit human decision.  Events wholly
outside that window remain immutable inventory.  Even then its role remains a
coarse Transcript reference and ``counted_gold_ready`` is permanently false.

The trust model is reproducibility and integrity under trusted repository
code, not a cryptographic signature and not a substitute for listening.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from hri_turn_taking.data import audio_contract as audio_contract_module
from hri_turn_taking.data import audio_gold_alignment as gold_alignment_module
from hri_turn_taking.data import transcript_acceptance as acceptance_module
from hri_turn_taking.data import transcript_fs_guard as fs_guard
from hri_turn_taking.data.source_paths import corpus_root, hpe_delivery_root

DRAFT_ARTIFACT_KIND = "vap_transcript_manual_review_draft"
FINAL_ARTIFACT_KIND = "vap_transcript_manual_review_final"
FORMAT_VERSION = 3
OUTPUT_ROLE = acceptance_module.OUTPUT_ROLE
TRUST_MODEL = "trusted_code_integrity_and_reproducibility_not_signature"
SPEAKERS = acceptance_module.SPEAKERS
FORMAL_SCENES = audio_contract_module.ACCEPTED_SCENES
DEFAULT_PADDING_S = Decimal("1.5")
PROJECT_ROOT = fs_guard.lexical_absolute(__file__).parents[3]
FORMAL_REVIEW_ROOT = PROJECT_ROOT / "data" / "annotations" / "transcript_review"
INITIAL_DRAFT_FILENAME = "draft.rev0000.json"
FINAL_REVIEW_FILENAME = "final.v3.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_RE = re.compile(r"^[0-9]{2}_[0-9]{2}(?:_[0-9]+)?__[0-9]+$")
_DRAFT_FILENAME_RE = re.compile(r"^draft\.rev[0-9]{4}\.json$")
_DECISIONS = frozenset({"unreviewed", "keep", "change"})
_CANDIDATE_DECISIONS = frozenset({"unresolved", "accepted", "rejected"})


class TranscriptReviewError(ValueError):
    """Raised when a review artifact or transition is not trustworthy."""


@dataclass(frozen=True)
class ReviewSourceAuthority:
    """Code-owned roots; review JSON cannot redefine source authority."""

    corpus_root: Path
    hpe_delivery_root: Path
    audio_contract_root: Path
    gold_alignment_root: Path | None = None
    pose_base_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_root", Path(self.corpus_root).resolve())
        object.__setattr__(
            self, "hpe_delivery_root", Path(self.hpe_delivery_root).resolve()
        )
        object.__setattr__(
            self, "audio_contract_root", Path(self.audio_contract_root).resolve()
        )
        object.__setattr__(
            self,
            "gold_alignment_root",
            Path(
                self.gold_alignment_root
                or Path(self.audio_contract_root).resolve().parent / "audio_gold_alignment"
            ).resolve(),
        )
        object.__setattr__(
            self,
            "pose_base_root",
            Path(
                self.pose_base_root
                or Path(self.audio_contract_root).resolve().parent / "intake_v2"
            ).resolve(),
        )


@dataclass(frozen=True)
class CanonicalDraftSnapshot:
    """One canonical draft held against replacement for a complete operation."""

    path: Path
    scene: str
    raw: bytes
    value: dict[str, Any]


@dataclass(frozen=True)
class CanonicalFinalSnapshot:
    """One canonical final and its exact bound canonical draft, both held."""

    path: Path
    scene: str
    raw: bytes
    value: dict[str, Any]
    draft: CanonicalDraftSnapshot


def default_source_authority() -> ReviewSourceAuthority:
    """Return the fixed production authority for the sealed ten-scene set."""

    project_root = Path(__file__).resolve().parents[3]
    return ReviewSourceAuthority(
        corpus_root=corpus_root(project_root),
        hpe_delivery_root=hpe_delivery_root(project_root),
        audio_contract_root=project_root / "data" / "annotations" / "audio_contract",
        gold_alignment_root=(
            project_root / "data" / "annotations" / "audio_gold_alignment"
        ),
        pose_base_root=project_root / "data" / "processed" / "intake_v2",
    )


def _fail(message: str) -> None:
    raise TranscriptReviewError(message)


def canonical_review_scene_dir(scene: str) -> Path:
    """Return the sole production directory authorised for one formal scene."""

    _scene_parts(scene)
    return FORMAL_REVIEW_ROOT / scene


def canonical_initial_draft_path(scene: str) -> Path:
    """Return the sole initial-draft path for one production scene."""

    return canonical_review_scene_dir(scene) / INITIAL_DRAFT_FILENAME


def canonical_final_review_path(scene: str) -> Path:
    """Return the sole confirmed-final path for one production scene."""

    return canonical_review_scene_dir(scene) / FINAL_REVIEW_FILENAME


def _ensure_formal_review_scene_dir(scene: str) -> Path:
    scene_dir = canonical_review_scene_dir(scene)
    annotations_root = FORMAL_REVIEW_ROOT.parent
    try:
        fs_guard.guard_directory(annotations_root)
        try:
            FORMAL_REVIEW_ROOT.lstat()
        except FileNotFoundError:
            FORMAL_REVIEW_ROOT.mkdir(exist_ok=False)
        fs_guard.guard_directory(FORMAL_REVIEW_ROOT)
        try:
            scene_dir.lstat()
        except FileNotFoundError:
            scene_dir.mkdir(exist_ok=False)
        return fs_guard.guard_directory(scene_dir)[0]
    except (fs_guard.TranscriptPathGuardError, OSError) as exc:
        raise TranscriptReviewError(
            f"cannot establish canonical formal review directory under {annotations_root}"
        ) from exc


def _require_canonical_review_output(
    path: str | Path, *, scene: str, final: bool
) -> Path:
    expected = (
        canonical_final_review_path(scene)
        if final
        else canonical_initial_draft_path(scene)
    )
    absolute = fs_guard.lexical_absolute(path)
    if absolute != expected:
        kind = "final" if final else "initial draft"
        _fail(f"formal {kind} path must be exactly {expected}")
    _ensure_formal_review_scene_dir(scene)
    try:
        return fs_guard.guard_output_path(absolute)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewError(f"formal review output path is unsafe: {exc}") from exc


def require_canonical_draft_path(path: str | Path) -> tuple[Path, str]:
    """Preflight one production draft before any source replay or resolve."""

    with held_canonical_draft(path) as snapshot:
        return snapshot.path, snapshot.scene


def require_canonical_final_path(path: str | Path) -> tuple[Path, str]:
    """Preflight the single production final for its declared scene."""

    with held_canonical_final(path) as snapshot:
        return snapshot.path, snapshot.scene


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranscriptReviewError(f"{label} is not strict UTF-8") from exc

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
        raise TranscriptReviewError(
            f"{label} is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys do not match the v{FORMAT_VERSION} contract "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or value != value.strip() or (not allow_empty and not value):
        _fail(f"{label} must be a string without surrounding whitespace")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact JSON integer (not bool or float)")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _require_decimal_string(value: Any, label: str) -> Decimal:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise TranscriptReviewError(f"{label} is not a decimal string") from exc
    if not result.is_finite():
        _fail(f"{label} must be finite")
    if _decimal_text(result) != value:
        _fail(f"{label} is not canonically formatted")
    return result


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        _fail("cannot encode a non-finite decimal")
    rounded = value.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_EVEN)
    text = format(rounded, "f").rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


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
        raise TranscriptReviewError("artifact cannot be encoded as strict JSON") from exc


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _type_exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _type_exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path, *, media_type: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        _fail(f"required source does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "media_type": media_type,
    }


def _scene_parts(scene: str) -> tuple[str, str]:
    if type(scene) is not str or _SCENE_RE.fullmatch(scene) is None:
        _fail("scene must use canonical group__scenario spelling")
    if scene not in FORMAL_SCENES:
        _fail(f"scene {scene!r} is outside the sealed ten-scene review set")
    return scene.rsplit("__", maxsplit=1)


def _require_current_review_format(value: Any, label: str) -> int:
    version = _require_exact_int(value, f"{label}.format_version")
    if version in {1, 2}:
        _fail(
            f"{label} review-v{version} is stale and cannot be migrated in place; "
            f"recreate the draft with review-v{FORMAT_VERSION} and repeat human review"
        )
    if version != FORMAT_VERSION:
        _fail(f"{label} format_version {version} is unsupported")
    return version


def _draft_header_from_raw(raw: bytes, *, path: Path, label: str) -> CanonicalDraftSnapshot:
    value = _require_object(_strict_json_loads(raw, label), label)
    if value.get("artifact_kind") != DRAFT_ARTIFACT_KIND:
        _fail(f"{label} artifact_kind is unsupported")
    _require_current_review_format(value.get("format_version"), label)
    scene = _require_string(value.get("scene"), f"{label}.scene")
    _scene_parts(scene)
    expected_parent = canonical_review_scene_dir(scene)
    if path.parent != expected_parent or _DRAFT_FILENAME_RE.fullmatch(path.name) is None:
        _fail(
            "formal draft must be a canonical draft.revNNNN.json inside "
            f"{expected_parent}"
        )
    return CanonicalDraftSnapshot(path, scene, raw, value)


@contextmanager
def held_canonical_draft(path: str | Path) -> Iterator[CanonicalDraftSnapshot]:
    """Hold the actual canonical draft through the caller's complete replay."""

    try:
        guarded = fs_guard.guard_regular_file_path(path)
        with fs_guard.guarded_read(guarded) as held:
            snapshot = _draft_header_from_raw(
                fs_guard.read_guarded_bytes(held),
                path=guarded,
                label="formal draft",
            )
            yield snapshot
            fs_guard.verify_guarded_read(held)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewError(f"formal draft path is unsafe: {exc}") from exc


@contextmanager
def held_canonical_final(path: str | Path) -> Iterator[CanonicalFinalSnapshot]:
    """Hold the actual canonical final and its exact canonical draft together."""

    try:
        guarded = fs_guard.guard_regular_file_path(path)
        with fs_guard.guarded_read(guarded) as held_final:
            raw_final = fs_guard.read_guarded_bytes(held_final)
            value = _require_object(
                _strict_json_loads(raw_final, "formal final"), "formal final"
            )
            if value.get("artifact_kind") != FINAL_ARTIFACT_KIND:
                _fail("formal final artifact_kind is unsupported")
            _require_current_review_format(
                value.get("format_version"), "formal final"
            )
            scene = _require_string(value.get("scene"), "formal final.scene")
            _scene_parts(scene)
            expected = canonical_final_review_path(scene)
            if guarded != expected:
                _fail(f"formal final path must be exactly {expected}")
            binding = _require_object(value.get("draft"), "formal final.draft")
            _require_exact_keys(
                binding,
                {"path", "size_bytes", "sha256", "media_type"},
                "formal final.draft",
            )
            if binding["media_type"] != "application/json":
                _fail("formal final.draft.media_type must be application/json")
            draft_path = fs_guard.lexical_absolute(
                _require_string(binding.get("path"), "formal final.draft.path")
            )
            with fs_guard.guarded_read(draft_path) as held_draft:
                raw_draft = fs_guard.read_guarded_bytes(held_draft)
                expected_size = _require_exact_int(
                    binding.get("size_bytes"),
                    "formal final.draft.size_bytes",
                    minimum=0,
                )
                expected_digest = binding.get("sha256")
                if (
                    type(expected_digest) is not str
                    or _SHA256_RE.fullmatch(expected_digest) is None
                ):
                    _fail("formal final.draft.sha256 must be a lowercase SHA-256 digest")
                if len(raw_draft) != expected_size or hashlib.sha256(
                    raw_draft
                ).hexdigest() != expected_digest:
                    _fail("formal final's bound draft bytes changed or disappeared")
                draft = _draft_header_from_raw(
                    raw_draft, path=draft_path, label="formal final bound draft"
                )
                if draft.scene != scene:
                    _fail("formal final scene differs from its canonical bound draft")
                snapshot = CanonicalFinalSnapshot(
                    guarded, scene, raw_final, value, draft
                )
                yield snapshot
                fs_guard.verify_guarded_read(held_draft)
            fs_guard.verify_guarded_read(held_final)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewError(f"formal final path is unsafe: {exc}") from exc


def _canonical_paths(scene: str, authority: ReviewSourceAuthority) -> dict[str, Path]:
    group, scenario = _scene_parts(scene)
    transcript_dir = authority.corpus_root / group / "transcript" / scenario
    return {
        "timeline": (transcript_dir / "timeline_video.json").resolve(),
        "offset": (transcript_dir / "offset.json").resolve(),
        "audio_contract": (
            authority.audio_contract_root / f"{scene}.audio_contract.json"
        ).resolve(),
        "gold_alignment": (
            authority.gold_alignment_root / f"{scene}.audio_gold_alignment.json"
        ).resolve(),
    }


def _normalization() -> dict[str, Any]:
    return {
        "contract": acceptance_module.SOURCE_NORMALIZATION_CONTRACT,
        "version": 1,
        "operations": [
            {
                "normalization_id": acceptance_module.LEGACY_OFFSET_FPS_NORMALIZATION_ID,
                "operation": acceptance_module.LEGACY_OFFSET_FPS_NORMALIZATION,
                "source": "offset",
                "json_path": "$.fps",
                "byte_offset": 148,
                "expected_json_token": "30.0",
                "expected_json_type": "float",
                "canonical_json_type": "integer",
                "canonical_value": 30,
                "evidence": (
                    "Legacy generator serialized the frozen 30/1 fps value as the exact "
                    "JSON float token 30.0; normalization changes type only."
                ),
                "review_status": "confirmed",
            }
        ],
    }


def _field_repair(
    repair_id: str,
    *,
    source_index: int,
    field: str,
    expected: str,
    replacement: str,
    operation: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "repair_id": repair_id,
        "operation": operation,
        "source_index": source_index,
        "field": field,
        "expected": expected,
        "replacement": replacement,
        "classification": "mechanical",
        "evidence": evidence,
        "review_status": "confirmed",
    }


def _candidate(
    candidate_id: str,
    *,
    source_index: int,
    field: str,
    expected: str,
    replacement: str,
    evidence: str,
    listening_window_s: tuple[str, str],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "operation": "correct_timecode_boundary",
        "source_index": source_index,
        "field": field,
        "expected": expected,
        "replacement": replacement,
        "classification": "manual_audio",
        "evidence": evidence,
        "timeline_local_listening_window_s": {
            "start_s": listening_window_s[0],
            "end_s_exclusive": listening_window_s[1],
        },
        "decision": {"status": "unresolved", "note": ""},
    }


_TRIM_EVIDENCE = "Remove one trailing ASCII space; semantic timecode digits are unchanged."
_SPEAKER_TRIM_EVIDENCE = "Remove one trailing ASCII space; canonical speaker identity is unchanged."


# Source indices are zero-based.  Every operation is bound to the immutable
# production preimage; changing an upstream byte makes draft construction fail.
FORMAL_REVIEW_PLANS: dict[str, dict[str, Any]] = {
    scene: {
        "mechanical_repairs": [],
        "semantic_candidates": [],
        "event_flags": {},
        "scene_flags": [],
    }
    for scene in FORMAL_SCENES
}

FORMAL_REVIEW_PLANS["05_14__2"].update(
    {
        "mechanical_repairs": [
            _field_repair(
                f"05-14-2-e{event}-end-trailing-space",
                source_index=event,
                field="end_timecode",
                expected=value + " ",
                replacement=value,
                operation="normalize_timecode",
                evidence=_TRIM_EVIDENCE,
            )
            for event, value in (
                (106, "00:03:25:24"),
                (108, "00:03:32:26"),
                (109, "00:03:37:22"),
                (110, "00:03:41:14"),
            )
        ],
        "semantic_candidates": [
            _candidate(
                "05-14-2-e27-end-39-21",
                source_index=27,
                field="end_timecode",
                expected="00:00:29:21",
                replacement="00:00:39:21",
                evidence=(
                    "Candidate repairs a negative interval for B 'Oh, okay.'; B lavalier "
                    "dominates most of 39:09-39:21, but the overlapping A event requires "
                    "owner listening."
                ),
                listening_window_s=("38.8", "40.2"),
            )
        ],
        "event_flags": {
            27: ["semantic_boundary_candidate", "overlap_context"],
            79: ["long_event_gt_15s"],
            113: ["long_event_gt_15s"],
        },
        "scene_flags": ["semantic_boundary_decision_required"],
    }
)

FORMAL_REVIEW_PLANS["05_14__3"].update(
    {
        "mechanical_repairs": [
            _field_repair(
                "05-14-3-e151-ff30-carry",
                source_index=151,
                field="end_timecode",
                expected="00:09:37:30",
                replacement="00:09:38:00",
                operation="normalize_timecode",
                evidence="Unique 30/1 non-drop-frame carry from FF=30 into the next second.",
            )
        ],
        "semantic_candidates": [
            _candidate(
                "05-14-3-e146-start-09-10-15",
                source_index=146,
                field="start_timecode",
                expected="00:09:40:15",
                replacement="00:09:10:15",
                evidence=(
                    "Original start is after the end. R energy supports the tens-digit "
                    "candidate around 09:10, while neighbouring C/R overlap still requires "
                    "owner listening."
                ),
                listening_window_s=("549.5", "551.8"),
            )
        ],
        "event_flags": {
            3: ["same_speaker_overlap"],
            5: ["same_speaker_overlap"],
            101: ["long_event_gt_15s"],
            123: ["same_speaker_overlap"],
            124: ["same_speaker_overlap"],
            146: ["semantic_boundary_candidate", "overlap_context"],
        },
        "scene_flags": [
            "semantic_boundary_decision_required",
        ],
    }
)

FORMAL_REVIEW_PLANS["05_15__2"].update(
    {
        "mechanical_repairs": [
            _field_repair(
                "05-15-2-e68-end-trailing-space",
                source_index=68,
                field="end_timecode",
                expected="00:03:17:08 ",
                replacement="00:03:17:08",
                operation="normalize_timecode",
                evidence=_TRIM_EVIDENCE,
            )
        ],
        "semantic_candidates": [
            _candidate(
                "05-15-2-e76-end-03-57-02",
                source_index=76,
                field="end_timecode",
                expected="00:03:27:02",
                replacement="00:03:57:02",
                evidence=(
                    "Candidate repairs a negative interval for B; B remains energetic after "
                    "57:02, so this is a listening proposal rather than an accepted end."
                ),
                listening_window_s=("234.2", "238.3"),
            )
        ],
        "event_flags": {
            **{
                index: ["suspected_speaker_identity_mismatch"]
                for index in (55, 70, 72, 74, 75, 77, 78, 80, 81, 84)
            },
            76: ["semantic_boundary_candidate", "suspected_speaker_identity_mismatch"],
        },
        "scene_flags": [
            "semantic_boundary_decision_required",
            "targeted_speaker_assignment_review_required",
        ],
    }
)

FORMAL_REVIEW_PLANS["05_15__3"].update(
    {
        "mechanical_repairs": [],
        "semantic_candidates": [
            _candidate(
                "05-15-3-e17-end-34-28",
                source_index=17,
                field="end_timecode",
                expected="00:00:34:25",
                replacement="00:00:34:28",
                evidence=(
                    "This is the unique minimal positive interval before e18, but continuous "
                    "R energy provides no acoustic boundary; owner must review segmentation."
                ),
                listening_window_s=("34", "35.8"),
            ),
            _candidate(
                "05-15-3-e88-end-02-38-15",
                source_index=88,
                field="end_timecode",
                expected="00:02:28:15",
                replacement="00:02:38:15",
                evidence=(
                    "A lavalier dominates the proposed interval and drops afterwards, but the "
                    "tens-digit correction still requires owner listening."
                ),
                listening_window_s=("156.1", "159"),
            ),
        ],
        "event_flags": {
            15: ["suspected_speaker_identity_mismatch"],
            17: ["semantic_boundary_candidate", "segmentation_ambiguous"],
            53: ["suspected_speaker_identity_mismatch"],
            64: ["same_speaker_overlap"],
            65: ["same_speaker_overlap"],
            66: ["suspected_speaker_identity_mismatch"],
            68: ["suspected_speaker_identity_mismatch"],
            76: ["same_speaker_overlap", "suspected_speaker_identity_mismatch"],
            77: ["same_speaker_overlap"],
            88: ["semantic_boundary_candidate"],
            91: ["same_speaker_overlap"],
            92: ["same_speaker_overlap"],
        },
        "scene_flags": [
            "semantic_boundary_decision_required",
            "targeted_speaker_assignment_review_required",
        ],
    }
)

FORMAL_REVIEW_PLANS["05_15__4"].update(
    {
        "event_flags": {
            **{
                index: ["suspected_speaker_identity_mismatch"]
                for index in (2, 4, 16, 17, 21, 22, 43, 55)
            },
            **{
                index: ["repeated_text_cluster_requires_review"]
                for index in range(7, 21)
            },
        },
        "scene_flags": [
            "source_pre_review_speaker_collapse_a_zero_events",
            "full_speaker_text_reannotation_required",
        ],
    }
)
for _index in (16, 17):
    FORMAL_REVIEW_PLANS["05_15__4"]["event_flags"][_index] = [
        "repeated_text_cluster_requires_review",
        "suspected_speaker_identity_mismatch",
    ]

FORMAL_REVIEW_PLANS["05_15__5"].update(
    {
        "event_flags": {
            index: ["suspected_speaker_identity_mismatch"]
            for index in (
                12,
                16,
                19,
                25,
                27,
                29,
                31,
                33,
                39,
                41,
                44,
                46,
                49,
                52,
                59,
                67,
                82,
                89,
                94,
                97,
            )
        },
        "scene_flags": [
            "source_pre_review_speaker_collapse_b_98_of_119_events",
            "full_speaker_reannotation_required",
        ],
    }
)

FORMAL_REVIEW_PLANS["05_15__6"].update(
    {
        "mechanical_repairs": [
            {
                "repair_id": "05-15-6-e93-missing-text-quote",
                "operation": "insert_missing_text_opening_quote",
                "source_index": 93,
                "field": "text",
                "expected": '"text": after we visited Louvre Museum."',
                "replacement": '"text": "after we visited Louvre Museum."',
                "classification": "mechanical",
                "evidence": "Insert the uniquely missing JSON opening quote without changing text.",
                "review_status": "confirmed",
                "byte_offset": 13863,
            },
            *[
                _field_repair(
                    f"05-15-6-e{event}-speaker-trailing-space",
                    source_index=event,
                    field="speaker",
                    expected="R ",
                    replacement="R",
                    operation="normalize_speaker_whitespace",
                    evidence=_SPEAKER_TRIM_EVIDENCE,
                )
                for event in (137, 139, 160)
            ],
        ],
        "event_flags": {
            90: ["suspected_speaker_identity_mismatch"],
            110: ["same_speaker_overlap"],
            112: ["same_speaker_overlap"],
            156: ["same_speaker_overlap", "identical_start_time"],
            157: ["suspected_speaker_identity_mismatch"],
            158: ["same_speaker_overlap", "identical_start_time", "source_order_regression"],
        },
        "scene_flags": [
            "source_pre_review_speaker_c_low_prevalence",
            "targeted_speaker_overlap_review_required",
        ],
    }
)

FORMAL_REVIEW_PLANS["03_20__4"].update(
    {
        "event_flags": {
            18: ["long_event_gt_15s"],
            36: ["suspected_speaker_identity_mismatch"],
        },
        "scene_flags": [
            "targeted_speaker_assignment_review_required",
        ],
    }
)
FORMAL_REVIEW_PLANS["03_26__3"].update(
    {
        "event_flags": {11: ["same_speaker_overlap"], 12: ["same_speaker_overlap"]},
        "scene_flags": [
            "same_speaker_overlap_review_required",
        ],
    }
)
FORMAL_REVIEW_PLANS["05_14__4"].update(
    {
        "event_flags": {130: ["long_event_gt_15s"]},
        "scene_flags": ["long_event_boundary_review_required"],
    }
)


def _parse_audio_contract(
    path: Path,
    *,
    scene: str,
    authority: ReviewSourceAuthority,
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    audio_authority = audio_contract_module.AudioSourceAuthority(
        corpus_root=authority.corpus_root,
        hpe_delivery_root=authority.hpe_delivery_root,
    )
    try:
        report = audio_contract_module._load_audio_contract_with_authority(
            path, authority=audio_authority
        )
        if report.get("scene") != scene:
            _fail("audio contract scene does not match review scene")
    except (audio_contract_module.AudioContractError, KeyError) as exc:
        raise TranscriptReviewError(f"audio contract replay failed: {exc}") from exc
    if path.read_bytes() != raw:
        _fail("audio contract changed during full replay")
    return report, raw


def _parse_gold_alignment(
    path: Path,
    *,
    scene: str,
    authority: ReviewSourceAuthority,
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    gold_authority = gold_alignment_module.GoldAlignmentAuthority(
        audio_manifest_root=authority.audio_contract_root,
        pose_base_root=authority.pose_base_root,
    )
    try:
        report = gold_alignment_module._load_with_authority(
            path, authority=gold_authority
        )
    except (gold_alignment_module.AudioGoldAlignmentError, OSError, KeyError) as exc:
        raise TranscriptReviewError(f"gold-alignment replay failed: {exc}") from exc
    if report.get("scene") != scene:
        _fail("gold-alignment scene does not match review scene")
    if path.read_bytes() != raw:
        _fail("gold-alignment artifact changed during full replay")
    gates = _require_object(report.get("gates"), "gold-alignment gates")
    authorization = _require_object(
        report.get("authorization"), "gold-alignment authorization"
    )
    if gates.get("alignment_prerequisite_pass") is not True:
        _fail("gold-alignment prerequisite did not pass")
    if authorization.get("gold_vad_use_authorized") is not True:
        _fail("gold-alignment does not authorize downstream use")
    return report, raw


def _source_bundle(
    scene: str, authority: ReviewSourceAuthority
) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any], dict[str, Any]]:
    paths = _canonical_paths(scene, authority)
    for name, path in paths.items():
        if not path.is_file():
            _fail(f"canonical {name} source does not exist: {path}")
    audio_report, _audio_raw = _parse_audio_contract(
        paths["audio_contract"], scene=scene, authority=authority
    )
    gold_report, _gold_raw = _parse_gold_alignment(
        paths["gold_alignment"], scene=scene, authority=authority
    )
    audio_sources = _require_object(audio_report.get("sources"), "audio contract.sources")
    bindings = {
        "timeline": _file_binding(paths["timeline"], media_type="application/json"),
        "offset": _file_binding(paths["offset"], media_type="application/json"),
        "audio_contract": _file_binding(
            paths["audio_contract"], media_type="application/json"
        ),
        "gold_alignment": _file_binding(
            paths["gold_alignment"], media_type="application/json"
        ),
        "reference_wav": {
            key: audio_sources["reference"][key]
            for key in ("path", "size_bytes", "sha256", "media_type")
        },
        "lavalier_wavs": {
            speaker: {
                key: audio_sources[speaker][key]
                for key in ("path", "size_bytes", "sha256", "media_type")
            }
            for speaker in SPEAKERS
        },
    }
    return (
        bindings,
        paths["timeline"].read_bytes(),
        paths["offset"].read_bytes(),
        audio_report,
        gold_report,
    )


def _verify_minimal_binding(
    binding_value: Any,
    *,
    expected_path: Path,
    expected_media_type: str,
    label: str,
) -> None:
    binding = _require_object(binding_value, label)
    _require_exact_keys(binding, {"path", "size_bytes", "sha256", "media_type"}, label)
    path_text = _require_string(binding["path"], f"{label}.path")
    path = Path(path_text).resolve()
    if path != expected_path.resolve():
        _fail(f"{label}.path is not the canonical authority path")
    size = _require_exact_int(binding["size_bytes"], f"{label}.size_bytes", minimum=0)
    digest = binding["sha256"]
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if binding["media_type"] != expected_media_type:
        _fail(f"{label}.media_type is inconsistent")
    if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
        _fail(f"{label} bytes changed or disappeared")


def _verify_source_bindings(
    sources_value: Any,
    *,
    scene: str,
    authority: ReviewSourceAuthority,
) -> None:
    sources = _require_object(sources_value, "sources")
    _require_exact_keys(
        sources,
        {
            "timeline",
            "offset",
            "audio_contract",
            "gold_alignment",
            "reference_wav",
            "lavalier_wavs",
        },
        "sources",
    )
    paths = _canonical_paths(scene, authority)
    _verify_minimal_binding(
        sources["timeline"],
        expected_path=paths["timeline"],
        expected_media_type="application/json",
        label="sources.timeline",
    )
    _verify_minimal_binding(
        sources["offset"],
        expected_path=paths["offset"],
        expected_media_type="application/json",
        label="sources.offset",
    )
    _verify_minimal_binding(
        sources["audio_contract"],
        expected_path=paths["audio_contract"],
        expected_media_type="application/json",
        label="sources.audio_contract",
    )
    _verify_minimal_binding(
        sources["gold_alignment"],
        expected_path=paths["gold_alignment"],
        expected_media_type="application/json",
        label="sources.gold_alignment",
    )
    audio_report, _raw = _parse_audio_contract(
        paths["audio_contract"], scene=scene, authority=authority
    )
    _gold_report, _gold_raw = _parse_gold_alignment(
        paths["gold_alignment"], scene=scene, authority=authority
    )
    audio_sources = _require_object(audio_report["sources"], "audio contract.sources")
    _verify_minimal_binding(
        sources["reference_wav"],
        expected_path=Path(audio_sources["reference"]["path"]),
        expected_media_type="audio/wav",
        label="sources.reference_wav",
    )
    lavs = _require_object(sources["lavalier_wavs"], "sources.lavalier_wavs")
    _require_exact_keys(lavs, set(SPEAKERS), "sources.lavalier_wavs")
    for speaker in SPEAKERS:
        _verify_minimal_binding(
            lavs[speaker],
            expected_path=Path(audio_sources[speaker]["path"]),
            expected_media_type="audio/wav",
            label=f"sources.lavalier_wavs.{speaker}",
        )


def _offset_values(
    raw_offset: bytes,
    source_normalizations: dict[str, Any],
    *,
    scene: str,
    reference_wav: Path,
) -> tuple[Decimal, Decimal]:
    try:
        raw_value = acceptance_module._strict_json_loads(raw_offset, "transcript offset")
        canonical, _operations = acceptance_module._normalize_transcript_offset_source(
            raw_offset, raw_value, source_normalizations
        )
        offset_sec, _processing_sr = acceptance_module._validate_transcript_offset(
            canonical, scene=scene, reference_wav=reference_wav
        )
        duration = acceptance_module._require_real_number(
            canonical["reference_duration_sec"], "reference_duration_sec"
        )
    except (acceptance_module.TranscriptAcceptanceError, KeyError) as exc:
        raise TranscriptReviewError(f"transcript offset replay failed: {exc}") from exc
    return offset_sec, duration


def _event_snapshot(event: dict[str, Any], source_index: int) -> dict[str, Any]:
    _require_exact_keys(
        event,
        {"start_timecode", "end_timecode", "speaker", "text"},
        f"timeline event {source_index}",
    )
    start = _require_string(event["start_timecode"], f"event {source_index}.start")
    end = _require_string(event["end_timecode"], f"event {source_index}.end")
    speaker = _require_string(event["speaker"], f"event {source_index}.speaker")
    text = _require_string(event["text"], f"event {source_index}.text")
    try:
        acceptance_module.parse_timecode_frames(start)
        acceptance_module.parse_timecode_frames(end)
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewError(f"event {source_index} timecode invalid: {exc}") from exc
    if speaker not in SPEAKERS:
        _fail(f"event {source_index} speaker must be exactly A/B/C/R after mechanical repair")
    return {
        "start_timecode": start,
        "end_timecode": end,
        "speaker": speaker,
        "text": text,
    }


def _blank_review() -> dict[str, Any]:
    return {
        "reviewed": False,
        "speaker": {"action": "unreviewed", "replacement": None},
        "start_timecode": {"action": "unreviewed", "replacement": None},
        "end_timecode": {"action": "unreviewed", "replacement": None},
        "note": "",
    }


def _frame_seconds(frame: int) -> Decimal:
    return Decimal(frame) / Decimal(acceptance_module.TIMECODE_FPS_NUMERATOR)


def _clip_item(
    *,
    source_index: int,
    snapshot: dict[str, Any],
    candidates: list[dict[str, Any]],
    offset_sec: Decimal,
    lav_bindings: dict[str, Any],
    audio_report: dict[str, Any],
    gold_report: dict[str, Any],
) -> dict[str, Any]:
    start_value = snapshot["start_timecode"]
    end_value = snapshot["end_timecode"]
    boundary_candidate_ids: list[str] = []
    for candidate in candidates:
        if candidate["field"] == "start_timecode":
            start_value = candidate["replacement"]
            boundary_candidate_ids.append(candidate["candidate_id"])
        elif candidate["field"] == "end_timecode":
            end_value = candidate["replacement"]
            boundary_candidate_ids.append(candidate["candidate_id"])
    try:
        start_frame = acceptance_module.parse_timecode_frames(start_value)
        end_frame = acceptance_module.parse_timecode_frames(end_value)
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewError(
            f"clip interval for event {source_index} is invalid: {exc}"
        ) from exc
    if end_frame <= start_frame:
        _fail(
            f"event {source_index} has no positive correction-aware listening interval"
        )
    shared_start = _frame_seconds(start_frame) + offset_sec
    shared_end = _frame_seconds(end_frame) + offset_sec
    try:
        start_tick, end_tick, authorized_start, authorized_end = (
            acceptance_module._gold_window(gold_report)
        )
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewError(f"gold window is invalid: {exc}") from exc
    intersection_start = max(shared_start, authorized_start)
    intersection_end = min(shared_end, authorized_end)
    if intersection_end <= intersection_start:
        _fail(f"event {source_index} has no listening interval inside the gold window")
    padded_shared_start = max(authorized_start, intersection_start - DEFAULT_PADDING_S)
    padded_shared_end = min(authorized_end, intersection_end + DEFAULT_PADDING_S)
    mapping = _require_object(gold_report.get("mapping_contract"), "mapping_contract")
    channels = _require_object(mapping.get("channels"), "mapping_contract.channels")
    sources = _require_object(audio_report.get("sources"), "audio contract.sources")
    per_speaker: dict[str, Any] = {}
    for speaker in SPEAKERS:
        channel = _require_object(channels.get(speaker), f"gold channel {speaker}")
        try:
            correction = gold_alignment_module._decimal_string(
                channel.get("correction_s"), f"gold channel {speaker}.correction_s"
            )
        except gold_alignment_module.AudioGoldAlignmentError as exc:
            raise TranscriptReviewError(str(exc)) from exc
        local_event_start = intersection_start - correction
        local_event_end = intersection_end - correction
        local_extract_start = padded_shared_start - correction
        local_extract_end = padded_shared_end - correction
        header = _require_object(
            _require_object(sources.get(speaker), f"audio source {speaker}").get(
                "wave_header"
            ),
            f"audio source {speaker}.wave_header",
        )
        n_frames = _require_exact_int(
            header.get("n_frames"), f"audio source {speaker}.n_frames", minimum=1
        )
        rate = _require_exact_int(
            header.get("sample_rate_hz"),
            f"audio source {speaker}.sample_rate_hz",
            minimum=1,
        )
        local_end = Decimal(n_frames) / Decimal(rate)
        if (
            local_extract_start < 0
            or local_extract_end > local_end
            or local_extract_end <= local_extract_start
        ):
            _fail(
                f"event {source_index} clip for {speaker} escapes real corrected audio"
            )
        per_speaker[speaker] = {
            "wav": dict(lav_bindings[speaker]),
            "correction_s": _decimal_text(correction),
            "inverse_formula": "channel_local_time_s = shared_time_s - correction_s",
            "event_local_start_s": _decimal_text(local_event_start),
            "event_local_end_s_exclusive": _decimal_text(local_event_end),
            "extract_local_start_s": _decimal_text(local_extract_start),
            "extract_local_end_s_exclusive": _decimal_text(local_extract_end),
        }
    return {
        "clip_id": f"event-{source_index:06d}",
        "source_index": source_index,
        "reason": "review_every_transcript_event",
        "interval_basis": (
            "semantic_candidate_for_listening_only"
            if boundary_candidate_ids
            else "mechanically_normalized_source"
        ),
        "candidate_ids_for_listening_only": boundary_candidate_ids,
        "timeline_local_start_frame": start_frame,
        "timeline_local_end_frame_exclusive": end_frame,
        "event_shared_start_s": _decimal_text(intersection_start),
        "event_shared_end_s_exclusive": _decimal_text(intersection_end),
        "pre_padding_s": _decimal_text(DEFAULT_PADDING_S),
        "post_padding_s": _decimal_text(DEFAULT_PADDING_S),
        "extract_shared_start_s": _decimal_text(padded_shared_start),
        "extract_shared_end_s_exclusive": _decimal_text(padded_shared_end),
        "authorized_tick_range": {
            "frame_rate_hz": acceptance_module.OUTPUT_RATE_HZ,
            "start_inclusive": start_tick,
            "end_exclusive": end_tick,
        },
        "audio_axis": "mic_A_shared_seconds",
        "shared_time_formula": acceptance_module.SHARED_TIME_FORMULA,
        "per_speaker_corrected_audio_read_only": per_speaker,
    }


def _audio_end_s(audio_report: dict[str, Any]) -> Decimal:
    sources = _require_object(audio_report["sources"], "audio contract.sources")
    ends: list[Decimal] = []
    for speaker in SPEAKERS:
        header = _require_object(
            sources[speaker].get("wave_header"), f"audio source {speaker}.wave_header"
        )
        frames = _require_exact_int(
            header.get("n_frames"), f"audio source {speaker}.n_frames", minimum=1
        )
        rate = _require_exact_int(
            header.get("sample_rate_hz"), f"audio source {speaker}.sample_rate_hz", minimum=1
        )
        ends.append(Decimal(frames) / Decimal(rate))
    return min(ends)


def _immutable_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "decision"}


def _immutable_event(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "review"}


def _prevalence_flags(prevalence: dict[str, Any], stage: str) -> set[str]:
    flags: set[str] = set()
    for speaker in SPEAKERS:
        item = _require_object(prevalence.get(speaker), f"{stage}.{speaker}")
        count = _require_exact_int(
            item.get("event_count"), f"{stage}.{speaker}.event_count", minimum=0
        )
        fraction = _require_decimal_string(
            item.get("event_fraction"), f"{stage}.{speaker}.event_fraction"
        )
        prefix = f"{stage}_speaker_{speaker.lower()}"
        if count == 0:
            flags.add(f"{prefix}_zero_events")
        if fraction >= Decimal("0.8"):
            flags.add(f"{prefix}_dominant_ge_80pct")
        if count > 0 and fraction <= Decimal("0.05"):
            flags.add(f"{prefix}_low_le_5pct")
    return flags


def _build_review_draft_with_authority(
    scene: str,
    *,
    authority: ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic draft in memory without publishing it."""

    _scene_parts(scene)
    selected_plan = FORMAL_REVIEW_PLANS[scene] if plan is None else plan
    bindings, raw_timeline, raw_offset, audio_report, gold_report = _source_bundle(
        scene, authority
    )
    source_normalizations = _normalization()
    offset_sec, _duration = _offset_values(
        raw_offset,
        source_normalizations,
        scene=scene,
        reference_wav=Path(bindings["reference_wav"]["path"]),
    )
    mechanical = selected_plan["mechanical_repairs"]
    try:
        timeline_value, _ids = acceptance_module._apply_repair_operations(
            raw_timeline, {"operations": mechanical}
        )
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewError(f"mechanical repair replay failed: {exc}") from exc
    try:
        authorized_start_tick, authorized_end_tick, _start_s, _end_s = (
            acceptance_module._gold_window(gold_report)
        )
        (
            in_scope_indices,
            excluded_inventory_base,
            scope_summary,
            source_prevalence,
        ) = acceptance_module._scope_material(
            timeline_value,
            transcript_offset_sec=offset_sec,
            authorized_start_tick=authorized_start_tick,
            authorized_end_tick=authorized_end_tick,
        )
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewError(f"gold-window scope replay failed: {exc}") from exc
    candidates = [
        json.loads(json.dumps(item))
        for item in selected_plan["semantic_candidates"]
        if item["source_index"] in in_scope_indices
    ]
    by_event: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_event.setdefault(candidate["source_index"], []).append(candidate)
    repair_targets = {repair["source_index"] for repair in mechanical}
    event_items: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    lav_bindings = _require_object(bindings["lavalier_wavs"], "lavalier bindings")
    for source_index, event_value in enumerate(timeline_value):
        event = _require_object(event_value, f"timeline event {source_index}")
        snapshot = _event_snapshot(event, source_index)
        if source_index not in in_scope_indices:
            continue
        flags = set(selected_plan["event_flags"].get(source_index, []))
        if source_index in repair_targets:
            flags.add("mechanical_repair_applied_in_review_view")
        start_frame = acceptance_module.parse_timecode_frames(snapshot["start_timecode"])
        end_frame = acceptance_module.parse_timecode_frames(snapshot["end_timecode"])
        if end_frame <= start_frame:
            flags.add("source_nonpositive_interval")
        if end_frame - start_frame > 15 * acceptance_module.TIMECODE_FPS_NUMERATOR:
            flags.add("long_event_gt_15s")
        event_items.append(
            {
                "event_id": f"event-{source_index:06d}",
                "source_index": source_index,
                "source": snapshot,
                "source_semantic_sha256": _canonical_json_sha256(snapshot),
                "flags": sorted(flags),
                "review": _blank_review(),
            }
        )
        clips.append(
            _clip_item(
                source_index=source_index,
                snapshot=snapshot,
                candidates=by_event.get(source_index, []),
                offset_sec=offset_sec,
                lav_bindings=lav_bindings,
                audio_report=audio_report,
                gold_report=gold_report,
            )
        )
    # Recompute generic review hot spots instead of trusting only the hand-written
    # catalogue. Non-positive candidate events are left for their explicit semantic
    # decision and are not used to infer overlap.
    valid_by_speaker: dict[str, list[tuple[int, int, int]]] = {
        speaker: [] for speaker in SPEAKERS
    }
    previous_start: int | None = None
    previous_index: int | None = None
    event_by_source_index = {item["source_index"]: item for item in event_items}
    for item in event_items:
        source_index = item["source_index"]
        start = acceptance_module.parse_timecode_frames(item["source"]["start_timecode"])
        end = acceptance_module.parse_timecode_frames(item["source"]["end_timecode"])
        if previous_start is not None and start < previous_start:
            item["flags"] = sorted(set(item["flags"]) | {"source_order_regression"})
            if previous_index is not None:
                previous = event_by_source_index[previous_index]
                previous["flags"] = sorted(
                    set(previous["flags"]) | {"source_order_regression_context"}
                )
        if previous_start is not None and start == previous_start:
            item["flags"] = sorted(set(item["flags"]) | {"identical_start_time"})
        previous_start = start
        previous_index = source_index
        if end > start:
            valid_by_speaker[item["source"]["speaker"]].append((start, end, source_index))
    for speaker_items in valid_by_speaker.values():
        ordered = sorted(speaker_items)
        active_end = -1
        active_index: int | None = None
        for start, end, source_index in ordered:
            if start < active_end and active_index is not None:
                for index in (active_index, source_index):
                    event_by_source_index[index]["flags"] = sorted(
                        set(event_by_source_index[index]["flags"])
                        | {"same_speaker_overlap"}
                    )
            if end > active_end:
                active_end = end
                active_index = source_index
    speaker_prevalence = source_prevalence
    excluded_by_index = {
        item["source_index"]: dict(item) for item in excluded_inventory_base
    }
    for source_index, item in excluded_by_index.items():
        item["flags"] = sorted(selected_plan["event_flags"].get(source_index, []))
        item["semantic_candidates"] = [
            _immutable_candidate(candidate)
            for candidate in selected_plan["semantic_candidates"]
            if candidate["source_index"] == source_index
        ]
    excluded_inventory = [
        excluded_by_index[index] for index in sorted(excluded_by_index)
    ]
    scene_flags = {
        flag
        for flag in selected_plan["scene_flags"]
        if flag != "inter_lavalier_alignment_prerequisite_failed"
    }
    scene_flags = {
        (
            f"source_pre_review_{flag}"
            if flag.startswith("speaker_collapse_") or flag == "speaker_c_low_prevalence"
            else flag
        )
        for flag in scene_flags
    }
    scene_flags.update(
        _prevalence_flags(
            speaker_prevalence["source_pre_review"], "source_pre_review"
        )
    )
    scene_flags.update(
        _prevalence_flags(
            speaker_prevalence["in_scope_pre_review"], "in_scope_pre_review"
        )
    )
    artifact = {
        "artifact_kind": DRAFT_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "counted_gold_ready": False,
        "review_status": "draft",
        "trust_model": TRUST_MODEL,
        "sources": bindings,
        "source_normalizations": source_normalizations,
        "mechanical_repairs": mechanical,
        "semantic_candidates": candidates,
        "scene_flags": sorted(scene_flags),
        "scope": scope_summary,
        "excluded_event_inventory": excluded_inventory,
        "speaker_prevalence": speaker_prevalence,
        "events": event_items,
        "clip_plan": clips,
    }
    # Close the initial read/use window even for callers that only build in memory.
    _verify_source_bindings(bindings, scene=scene, authority=authority)
    return artifact


def build_review_draft(scene: str) -> dict[str, Any]:
    """Build, but do not write, a production draft for one sealed scene."""

    return _build_review_draft_with_authority(scene, authority=default_source_authority())


def _validate_candidate_decision(value: Any, *, label: str) -> None:
    decision = _require_object(value, label)
    _require_exact_keys(decision, {"status", "note"}, label)
    status = decision["status"]
    if status not in _CANDIDATE_DECISIONS:
        _fail(f"{label}.status is unsupported")
    note = _require_string(decision["note"], f"{label}.note", allow_empty=True)
    if status == "unresolved" and note:
        _fail(f"{label}.note must be empty while unresolved")
    if status != "unresolved" and not note:
        _fail(f"{label}.note is required for an explicit candidate decision")


def _validate_field_decision(
    value: Any,
    *,
    label: str,
    original: str,
    speaker: bool,
) -> None:
    decision = _require_object(value, label)
    _require_exact_keys(decision, {"action", "replacement"}, label)
    action = decision["action"]
    if action not in _DECISIONS:
        _fail(f"{label}.action is unsupported")
    replacement = decision["replacement"]
    if action in {"unreviewed", "keep"}:
        if replacement is not None:
            _fail(f"{label}.replacement must be null for {action}")
        return
    replacement = _require_string(replacement, f"{label}.replacement")
    if replacement == original:
        _fail(f"{label}.change must differ from the reviewed source value")
    if speaker:
        if replacement not in SPEAKERS:
            _fail(f"{label}.replacement must be A/B/C/R")
    else:
        try:
            acceptance_module.parse_timecode_frames(replacement)
        except acceptance_module.TranscriptAcceptanceError as exc:
            raise TranscriptReviewError(f"{label}.replacement is invalid: {exc}") from exc


def _validate_event_review(item: dict[str, Any], *, label: str) -> None:
    review = _require_object(item["review"], f"{label}.review")
    _require_exact_keys(
        review,
        {"reviewed", "speaker", "start_timecode", "end_timecode", "note"},
        f"{label}.review",
    )
    if type(review["reviewed"]) is not bool:
        _fail(f"{label}.review.reviewed must be an exact JSON boolean")
    note = _require_string(
        review["note"], f"{label}.review.note", allow_empty=True
    )
    source = _require_object(item["source"], f"{label}.source")
    _validate_field_decision(
        review["speaker"],
        label=f"{label}.review.speaker",
        original=source["speaker"],
        speaker=True,
    )
    for field in ("start_timecode", "end_timecode"):
        _validate_field_decision(
            review[field],
            label=f"{label}.review.{field}",
            original=source[field],
            speaker=False,
        )
    actions = [review[field]["action"] for field in ("speaker", "start_timecode", "end_timecode")]
    if review["reviewed"]:
        if "unreviewed" in actions:
            _fail(f"{label} cannot be reviewed while a field remains unreviewed")
        if "change" in actions and not note:
            _fail(f"{label}.review.note is required for any manual change")
    elif actions != ["unreviewed", "unreviewed", "unreviewed"] or note:
        _fail(f"{label} unreviewed state must remain blank and internally consistent")


def _validate_draft_object(
    artifact: Any,
    *,
    authority: ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    draft = _require_object(artifact, "review draft")
    _require_exact_keys(
        draft,
        {
            "artifact_kind",
            "format_version",
            "scene",
            "output_role",
            "counted_gold_ready",
            "review_status",
            "trust_model",
            "sources",
            "source_normalizations",
            "mechanical_repairs",
            "semantic_candidates",
            "scene_flags",
            "scope",
            "excluded_event_inventory",
            "speaker_prevalence",
            "events",
            "clip_plan",
        },
        "review draft",
    )
    if draft["artifact_kind"] != DRAFT_ARTIFACT_KIND:
        _fail("review draft artifact_kind is unsupported")
    _require_current_review_format(draft["format_version"], "review draft")
    scene = _require_string(draft["scene"], "scene")
    _scene_parts(scene)
    if draft["output_role"] != OUTPUT_ROLE:
        _fail("review draft output_role is unsupported")
    if draft["counted_gold_ready"] is not False:
        _fail("Transcript review can never declare counted gold readiness")
    if draft["review_status"] != "draft":
        _fail("review draft status must remain draft")
    if draft["trust_model"] != TRUST_MODEL:
        _fail("review draft trust model is unsupported")
    _verify_source_bindings(draft["sources"], scene=scene, authority=authority)
    expected = _build_review_draft_with_authority(scene, authority=authority, plan=plan)
    for key in (
        "artifact_kind",
        "format_version",
        "scene",
        "output_role",
        "counted_gold_ready",
        "review_status",
        "trust_model",
        "sources",
        "source_normalizations",
        "mechanical_repairs",
        "scene_flags",
        "scope",
        "excluded_event_inventory",
        "speaker_prevalence",
        "clip_plan",
    ):
        if not _type_exact_json_equal(draft[key], expected[key]):
            _fail(f"review draft {key} does not replay from authoritative sources")
    candidates = draft["semantic_candidates"]
    expected_candidates = expected["semantic_candidates"]
    if type(candidates) is not list or len(candidates) != len(expected_candidates):
        _fail("review draft semantic candidate coverage changed")
    candidate_ids: set[str] = set()
    for index, (candidate, expected_candidate) in enumerate(
        zip(candidates, expected_candidates, strict=True)
    ):
        candidate = _require_object(candidate, f"semantic_candidates[{index}]")
        candidate_id = _require_string(
            candidate.get("candidate_id"), f"semantic_candidates[{index}].candidate_id"
        )
        if candidate_id in candidate_ids:
            _fail(f"duplicate semantic candidate {candidate_id!r}")
        candidate_ids.add(candidate_id)
        if not _type_exact_json_equal(
            _immutable_candidate(candidate), _immutable_candidate(expected_candidate)
        ):
            _fail(f"semantic candidate {candidate_id!r} immutable evidence changed")
        _validate_candidate_decision(
            candidate.get("decision"), label=f"semantic_candidates[{index}].decision"
        )
    events = draft["events"]
    expected_events = expected["events"]
    if type(events) is not list or len(events) != len(expected_events):
        _fail("review draft must cover every source event exactly once")
    event_ids: set[str] = set()
    source_indices: set[int] = set()
    for index, (item, expected_item) in enumerate(zip(events, expected_events, strict=True)):
        item = _require_object(item, f"events[{index}]")
        event_id = _require_string(item.get("event_id"), f"events[{index}].event_id")
        source_index = _require_exact_int(
            item.get("source_index"), f"events[{index}].source_index", minimum=0
        )
        if event_id in event_ids or source_index in source_indices:
            _fail("review draft contains a duplicate event id or source index")
        event_ids.add(event_id)
        source_indices.add(source_index)
        if not _type_exact_json_equal(
            _immutable_event(item), _immutable_event(expected_item)
        ):
            _fail(f"event {source_index} immutable source snapshot changed")
        _validate_event_review(item, label=f"events[{index}]")
    # Close the source-validation/use window after all mutable review fields were parsed.
    _verify_source_bindings(draft["sources"], scene=scene, authority=authority)
    return draft


def _load_review_draft_with_authority(
    path: str | Path,
    *,
    authority: ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    formal_snapshot: CanonicalDraftSnapshot | None = None,
) -> dict[str, Any]:
    lexical_path = fs_guard.lexical_absolute(path)
    if formal_snapshot is not None:
        if lexical_path != formal_snapshot.path:
            _fail("actual replayed draft differs from the held canonical path")
        if formal_scene is None or formal_scene != formal_snapshot.scene:
            _fail("held canonical draft scene was not propagated into replay")
        draft_path = formal_snapshot.path
        raw = formal_snapshot.raw
    else:
        draft_path = Path(path).resolve()
        if not draft_path.is_file():
            _fail(f"review draft does not exist: {draft_path}")
        raw = draft_path.read_bytes()
    draft = _strict_json_loads(raw, "review draft")
    validated = _validate_draft_object(draft, authority=authority, plan=plan)
    if formal_scene is not None:
        if validated["scene"] != formal_scene:
            _fail("actual formal draft scene changed after canonical preflight")
        expected_parent = canonical_review_scene_dir(formal_scene)
        if (
            lexical_path.parent != expected_parent
            or _DRAFT_FILENAME_RE.fullmatch(lexical_path.name) is None
        ):
            _fail(
                "actual replayed draft is outside its canonical formal scene directory"
            )
    if formal_snapshot is None and draft_path.read_bytes() != raw:
        _fail("review draft changed during validation")
    return validated


def load_review_draft(path: str | Path) -> dict[str, Any]:
    """Load and replay a production draft against fixed source authority."""

    with held_canonical_draft(path) as snapshot:
        return _load_review_draft_with_authority(
            snapshot.path,
            authority=default_source_authority(),
            formal_scene=snapshot.scene,
            formal_snapshot=snapshot,
        )


def _repair_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "repair_id": candidate["candidate_id"],
        "operation": candidate["operation"],
        "source_index": candidate["source_index"],
        "field": candidate["field"],
        "expected": candidate["expected"],
        "replacement": candidate["replacement"],
        "classification": candidate["classification"],
        "evidence": candidate["decision"]["note"],
        "review_status": "confirmed",
    }


def _resolved_repairs(draft: dict[str, Any]) -> list[dict[str, Any]]:
    repairs = [json.loads(json.dumps(item)) for item in draft["mechanical_repairs"]]
    mechanical_by_target = {
        (item["source_index"], item["field"]): item
        for item in draft["mechanical_repairs"]
    }
    candidate_by_target: dict[tuple[int, str], dict[str, Any]] = {}
    for candidate in draft["semantic_candidates"]:
        target = (candidate["source_index"], candidate["field"])
        if target in candidate_by_target:
            _fail("multiple semantic candidates target the same event field")
        candidate_by_target[target] = candidate
        status = candidate["decision"]["status"]
        if status == "unresolved":
            _fail(f"semantic candidate {candidate['candidate_id']!r} remains unresolved")
        if status == "accepted":
            repairs.append(_repair_from_candidate(candidate))

    for item in draft["events"]:
        review = item["review"]
        if review["reviewed"] is not True:
            _fail(f"event {item['source_index']} has not been explicitly reviewed")
        for field, operation in (
            ("speaker", "correct_speaker_identity"),
            ("start_timecode", "correct_timecode_boundary"),
            ("end_timecode", "correct_timecode_boundary"),
        ):
            decision = review[field]
            target = (item["source_index"], field)
            candidate = candidate_by_target.get(target)
            if candidate is not None:
                status = candidate["decision"]["status"]
                if status == "accepted":
                    if (
                        decision["action"] != "change"
                        or decision["replacement"] != candidate["replacement"]
                    ):
                        _fail(
                            f"accepted candidate {candidate['candidate_id']!r} is not mirrored "
                            "by the event field decision"
                        )
                    continue
                if status == "rejected" and (
                    decision["action"] == "change"
                    and decision["replacement"] == candidate["replacement"]
                ):
                    _fail(
                        f"rejected candidate {candidate['candidate_id']!r} was still selected"
                    )
            if decision["action"] != "change":
                continue
            expected = item["source"][field]
            if target in mechanical_by_target:
                mechanical = mechanical_by_target[target]
                repairs = [
                    repair
                    for repair in repairs
                    if (repair["source_index"], repair["field"]) != target
                ]
                expected = mechanical["expected"]
            repairs.append(
                {
                    "repair_id": f"owner-event-{item['source_index']:06d}-{field}",
                    "operation": operation,
                    "source_index": item["source_index"],
                    "field": field,
                    "expected": expected,
                    "replacement": decision["replacement"],
                    "classification": "manual_owner",
                    "evidence": review["note"],
                    "review_status": "confirmed",
                }
            )
    return repairs


def _final_scene_flags(draft_flags: list[str]) -> set[str]:
    """Keep review-v3 flag stage semantics explicit.

    ``scene_flags`` is retained as review-trigger provenance. Facts already
    namespaced to their measurement stage remain unchanged; everything that
    merely triggered the now-completed review is preserved as historical
    ``review_trigger_*`` provenance rather than an apparently live blocker.
    """

    factual_prefixes = (
        "source_pre_review_",
        "in_scope_pre_review_",
        "accepted_post_review_",
    )
    return {
        flag
        if flag.startswith(factual_prefixes) or flag.startswith("review_trigger_")
        else f"review_trigger_{flag}"
        for flag in draft_flags
    }


def _validate_reviewer(reviewer: Any, reviewed_at: Any, method: Any) -> tuple[str, str, str]:
    reviewer_text = _require_string(reviewer, "reviewer")
    reviewed_at_text = _require_string(reviewed_at, "reviewed_at")
    method_text = _require_string(method, "method")
    try:
        timestamp = datetime.fromisoformat(reviewed_at_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TranscriptReviewError("reviewed_at must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        _fail("reviewed_at must include a UTC offset")
    return reviewer_text, reviewed_at_text, method_text


def _final_review_from_draft(
    draft: dict[str, Any],
    *,
    draft_binding: dict[str, Any],
    reviewer: str,
    reviewed_at: str,
    method: str,
) -> dict[str, Any]:
    reviewer, reviewed_at, method = _validate_reviewer(reviewer, reviewed_at, method)
    repairs = _resolved_repairs(draft)
    raw_timeline = Path(draft["sources"]["timeline"]["path"]).read_bytes()
    raw_offset = Path(draft["sources"]["offset"]["path"]).read_bytes()
    offset_sec, reference_duration = _offset_values(
        raw_offset,
        draft["source_normalizations"],
        scene=draft["scene"],
        reference_wav=Path(draft["sources"]["reference_wav"]["path"]),
    )
    try:
        bindings = acceptance_module.derive_manual_review_bindings(
            raw_timeline,
            {"operations": repairs},
            reference_duration_sec=reference_duration,
            transcript_offset_sec=offset_sec,
            authorized_start_tick=draft["scope"][
                "authorized_start_tick_inclusive"
            ],
            authorized_end_tick=draft["scope"]["authorized_end_tick_exclusive"],
        )
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewError(f"resolved event boundaries are not acceptable: {exc}") from exc
    scene_flags = _final_scene_flags(draft["scene_flags"])
    scene_flags.update(
        _prevalence_flags(
            bindings["speaker_prevalence"]["accepted_post_review"],
            "accepted_post_review",
        )
    )
    return {
        "artifact_kind": FINAL_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": draft["scene"],
        "output_role": OUTPUT_ROLE,
        "counted_gold_ready": False,
        "review_status": "confirmed",
        "unresolved_issues": [],
        "trust_model": TRUST_MODEL,
        "draft": draft_binding,
        "sources": draft["sources"],
        "source_normalizations": draft["source_normalizations"],
        "scene_flags": sorted(scene_flags),
        "scope": bindings["scope_summary"],
        "excluded_event_inventory": draft["excluded_event_inventory"],
        "speaker_prevalence": bindings["speaker_prevalence"],
        "repairs": {"operations": repairs},
        "semantic_candidate_decisions": [
            {
                "candidate_id": item["candidate_id"],
                "status": item["decision"]["status"],
                "note": item["decision"]["note"],
            }
            for item in draft["semantic_candidates"]
        ],
        "event_reviews": [
            {
                "event_id": item["event_id"],
                "source_index": item["source_index"],
                "review": item["review"],
            }
            for item in draft["events"]
        ],
        "manual_review": {
            "status": "confirmed",
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "timeline_sha256": bindings["timeline_sha256"],
            "canonical_semantic_sha256": bindings["canonical_semantic_sha256"],
            "event_summary": bindings["event_summary"],
            "repair_set_sha256": bindings["repair_set_sha256"],
            "repair_ids": bindings["repair_ids"],
            "all_in_scope_events_reviewed": True,
            "speaker_assignments_reviewed": True,
            "time_boundaries_reviewed": True,
            "method": method,
        },
        "clip_plan_sha256": _canonical_json_sha256(draft["clip_plan"]),
    }


def _atomic_publish_no_clobber(
    destination: Path,
    payload: bytes,
    *,
    pre_publish_check: Any,
    post_publish_check: Any,
) -> Path:
    destination = destination.resolve()
    if destination.exists():
        _fail(f"refusing to overwrite existing artifact: {destination}")
    if not destination.parent.is_dir():
        _fail(f"output parent must already exist: {destination.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.tmp.",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        pre_publish_check()
        os.link(temporary, destination)
        post_publish_check(destination)
        return destination
    except FileExistsError as exc:
        raise TranscriptReviewError(
            f"refusing to overwrite existing artifact: {destination}"
        ) from exc
    except BaseException:
        # Check ownership by the actual hard-link identity.  This covers an
        # interruption raised after os.link created the destination but before
        # it returned, while never deleting a path another process replaced.
        try:
            still_ours = (
                temporary is not None
                and temporary.exists()
                and destination.exists()
                and os.path.samefile(temporary, destination)
            )
        except OSError:
            still_ours = False
        if still_ours:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_review_draft_with_authority(
    scene: str,
    output: str | Path,
    *,
    authority: ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output)
    draft = _build_review_draft_with_authority(scene, authority=authority, plan=plan)
    payload = _canonical_json_bytes(draft)

    def precheck() -> None:
        replay = _build_review_draft_with_authority(scene, authority=authority, plan=plan)
        if not _type_exact_json_equal(replay, draft):
            _fail("authoritative sources changed before draft publication")

    def postcheck(path: Path) -> None:
        _load_review_draft_with_authority(path, authority=authority, plan=plan)

    return _atomic_publish_no_clobber(
        destination, payload, pre_publish_check=precheck, post_publish_check=postcheck
    )


def write_review_draft(
    scene: str, output: str | Path | None = None
) -> Path:
    """Atomically publish a new production draft; never overwrite a path."""

    destination = _require_canonical_review_output(
        output if output is not None else canonical_initial_draft_path(scene),
        scene=scene,
        final=False,
    )
    return _write_review_draft_with_authority(
        scene, destination, authority=default_source_authority()
    )


def _finalize_review_with_authority(
    draft_path: str | Path,
    output: str | Path,
    *,
    reviewer: str,
    reviewed_at: str,
    method: str,
    authority: ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    formal_snapshot: CanonicalDraftSnapshot | None = None,
) -> Path:
    if formal_snapshot is not None:
        source_path = formal_snapshot.path
        raw_draft = formal_snapshot.raw
    else:
        source_path = Path(draft_path).resolve()
        raw_draft = source_path.read_bytes()
    draft = _load_review_draft_with_authority(
        source_path,
        authority=authority,
        plan=plan,
        formal_scene=formal_scene,
        formal_snapshot=formal_snapshot,
    )
    draft_binding = {
        "path": str(source_path),
        "size_bytes": len(raw_draft),
        "sha256": hashlib.sha256(raw_draft).hexdigest(),
        "media_type": "application/json",
    }
    final = _final_review_from_draft(
        draft,
        draft_binding=draft_binding,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        method=method,
    )
    payload = _canonical_json_bytes(final)

    def precheck() -> None:
        if formal_snapshot is None and source_path.read_bytes() != raw_draft:
            _fail("review draft changed before final publication")
        replayed = _load_review_draft_with_authority(
            source_path,
            authority=authority,
            plan=plan,
            formal_scene=formal_scene,
            formal_snapshot=formal_snapshot,
        )
        rebuilt = _final_review_from_draft(
            replayed,
            draft_binding=draft_binding,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            method=method,
        )
        if not _type_exact_json_equal(rebuilt, final):
            _fail("review or authoritative sources changed before final publication")

    def postcheck(path: Path) -> None:
        loaded = _load_final_review_with_authority(
            path,
            authority=authority,
            plan=plan,
            formal_scene=formal_scene,
        )
        if not _type_exact_json_equal(loaded, final) or path.read_bytes() != payload:
            _fail("published final review does not replay exactly")

    return _atomic_publish_no_clobber(
        Path(output), payload, pre_publish_check=precheck, post_publish_check=postcheck
    )


def _verify_bound_artifact(value: Any, *, label: str) -> tuple[Path, bytes]:
    binding = _require_object(value, label)
    _require_exact_keys(binding, {"path", "size_bytes", "sha256", "media_type"}, label)
    path = Path(_require_string(binding["path"], f"{label}.path")).resolve()
    size = _require_exact_int(binding["size_bytes"], f"{label}.size_bytes", minimum=0)
    digest = binding["sha256"]
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if binding["media_type"] != "application/json":
        _fail(f"{label}.media_type must be application/json")
    if not path.is_file():
        _fail(f"{label} does not exist: {path}")
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        _fail(f"{label} bytes changed or disappeared")
    return path, raw


def _load_final_review_with_authority(
    path: str | Path,
    *,
    authority: ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    formal_snapshot: CanonicalFinalSnapshot | None = None,
) -> dict[str, Any]:
    lexical_final_path = fs_guard.lexical_absolute(path)
    if formal_snapshot is not None:
        if lexical_final_path != formal_snapshot.path:
            _fail("actual replayed final differs from the held canonical path")
        if formal_scene is None or formal_scene != formal_snapshot.scene:
            _fail("held canonical final scene was not propagated into replay")
        final_path = formal_snapshot.path
        raw_final = formal_snapshot.raw
    else:
        final_path = Path(path).resolve()
        if not final_path.is_file():
            _fail(f"final review does not exist: {final_path}")
        raw_final = final_path.read_bytes()
    final = _require_object(_strict_json_loads(raw_final, "final review"), "final review")
    _require_exact_keys(
        final,
        {
            "artifact_kind",
            "format_version",
            "scene",
            "output_role",
            "counted_gold_ready",
            "review_status",
            "unresolved_issues",
            "trust_model",
            "draft",
            "sources",
            "source_normalizations",
            "scene_flags",
            "scope",
            "excluded_event_inventory",
            "speaker_prevalence",
            "repairs",
            "semantic_candidate_decisions",
            "event_reviews",
            "manual_review",
            "clip_plan_sha256",
        },
        "final review",
    )
    if final["artifact_kind"] != FINAL_ARTIFACT_KIND:
        _fail("final review artifact_kind is unsupported")
    _require_current_review_format(final["format_version"], "final review")
    if formal_scene is not None:
        if final["scene"] != formal_scene:
            _fail("actual formal final scene changed after canonical preflight")
        if lexical_final_path != canonical_final_review_path(formal_scene):
            _fail("actual replayed final is outside its canonical formal path")
    if final["output_role"] != OUTPUT_ROLE or final["review_status"] != "confirmed":
        _fail("final review role/status is unsupported")
    if final["counted_gold_ready"] is not False:
        _fail("a final Transcript review can never declare counted gold readiness")
    if type(final["unresolved_issues"]) is not list or final["unresolved_issues"]:
        _fail("final review unresolved_issues must be an empty list")
    if final["trust_model"] != TRUST_MODEL:
        _fail("final review trust model is unsupported")
    if formal_snapshot is not None:
        draft_path = formal_snapshot.draft.path
        raw_draft = formal_snapshot.draft.raw
        binding = _require_object(final["draft"], "draft")
        _require_exact_keys(
            binding, {"path", "size_bytes", "sha256", "media_type"}, "draft"
        )
        if (
            fs_guard.lexical_absolute(_require_string(binding["path"], "draft.path"))
            != draft_path
            or binding["media_type"] != "application/json"
            or _require_exact_int(binding["size_bytes"], "draft.size_bytes", minimum=0)
            != len(raw_draft)
            or binding["sha256"] != hashlib.sha256(raw_draft).hexdigest()
        ):
            _fail("final review does not bind the held canonical draft snapshot")
    else:
        draft_path, raw_draft = _verify_bound_artifact(final["draft"], label="draft")
    if formal_scene is not None:
        bound_draft_path = fs_guard.lexical_absolute(final["draft"]["path"])
        if (
            bound_draft_path.parent != canonical_review_scene_dir(formal_scene)
            or _DRAFT_FILENAME_RE.fullmatch(bound_draft_path.name) is None
        ):
            _fail("actual formal final binds a non-canonical/private draft path")
    draft = _load_review_draft_with_authority(
        draft_path,
        authority=authority,
        plan=plan,
        formal_scene=formal_scene,
        formal_snapshot=(formal_snapshot.draft if formal_snapshot is not None else None),
    )
    if final["scene"] != draft["scene"]:
        _fail("final review scene does not match its bound draft")
    manual = _require_object(final["manual_review"], "manual_review")
    _require_exact_keys(
        manual,
        {
            "status",
            "reviewer",
            "reviewed_at",
            "timeline_sha256",
            "canonical_semantic_sha256",
            "event_summary",
            "repair_set_sha256",
            "repair_ids",
            "all_in_scope_events_reviewed",
            "speaker_assignments_reviewed",
            "time_boundaries_reviewed",
            "method",
        },
        "manual_review",
    )
    if manual["status"] != "confirmed":
        _fail("manual_review.status must be confirmed")
    reviewer, reviewed_at, method = _validate_reviewer(
        manual["reviewer"], manual["reviewed_at"], manual["method"]
    )
    for key in (
        "all_in_scope_events_reviewed",
        "speaker_assignments_reviewed",
        "time_boundaries_reviewed",
    ):
        if manual[key] is not True:
            _fail(f"manual_review.{key} must be the exact JSON boolean true")
    rebuilt = _final_review_from_draft(
        draft,
        draft_binding=final["draft"],
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        method=method,
    )
    if not _type_exact_json_equal(final, rebuilt):
        _fail("final review does not exactly replay from its draft and sources")
    if formal_snapshot is None and (
        draft_path.read_bytes() != raw_draft or final_path.read_bytes() != raw_final
    ):
        _fail("review artifact changed during final validation")
    _verify_source_bindings(final["sources"], scene=final["scene"], authority=authority)
    return final


def load_final_review(path: str | Path) -> dict[str, Any]:
    """Load and fully replay a final coarse Transcript review."""

    with held_canonical_final(path) as snapshot:
        return _load_final_review_with_authority(
            snapshot.path,
            authority=default_source_authority(),
            formal_scene=snapshot.scene,
            formal_snapshot=snapshot,
        )


def finalize_review(
    draft_path: str | Path,
    output: str | Path | None = None,
    *,
    reviewer: str,
    reviewed_at: str,
    method: str = "manual listening to all four correction-mapped lavalier channels",
) -> Path:
    """Finalize a complete human review as coarse, never counted-gold evidence."""

    with held_canonical_draft(draft_path) as snapshot:
        destination = _require_canonical_review_output(
            output
            if output is not None
            else canonical_final_review_path(snapshot.scene),
            scene=snapshot.scene,
            final=True,
        )
        return _finalize_review_with_authority(
            snapshot.path,
            destination,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            method=method,
            authority=default_source_authority(),
            formal_scene=snapshot.scene,
            formal_snapshot=snapshot,
        )


def review_progress(draft: dict[str, Any]) -> dict[str, Any]:
    """Return progress without treating field presence as human confirmation."""

    events = draft["events"]
    reviewed = sum(item["review"]["reviewed"] is True for item in events)
    candidates = draft["semantic_candidates"]
    resolved = sum(item["decision"]["status"] != "unresolved" for item in candidates)
    complete = reviewed == len(events) and resolved == len(candidates)
    decision_consistency_ok = False
    if complete:
        try:
            _resolved_repairs(draft)
        except TranscriptReviewError:
            decision_consistency_ok = False
        else:
            decision_consistency_ok = True
    return {
        "scene": draft["scene"],
        "source_event_count": draft["scope"]["source_event_count"],
        "in_scope_event_count": len(events),
        "excluded_event_count": draft["scope"]["excluded_event_count"],
        "crossing_event_count": draft["scope"]["crossing_event_count"],
        "reviewed_in_scope_events": reviewed,
        "n_semantic_candidates": len(candidates),
        "resolved_semantic_candidates": resolved,
        "decision_consistency_ok": decision_consistency_ok,
        "ready_to_finalize": complete and decision_consistency_ok,
        "counted_gold_ready": False,
    }
