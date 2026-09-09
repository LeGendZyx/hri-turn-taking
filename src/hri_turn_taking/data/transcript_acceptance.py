"""Fail-closed acceptance of manually reviewed Transcript timelines.

Upstream ``timeline_video.json`` files are immutable evidence.  This module
never rewrites them: a versioned acceptance manifest binds their exact bytes
and may declare narrowly scoped, exact-preimage repairs that are applied only
in memory.  Accepted events are represented on the transcript's integer
30-fps frame axis and can then be rasterised onto an exact 25-Hz mic-A shared
timeline.  The rasterised result is a coarse activity reference for audit and
boundary refinement only.  It is deliberately *not* the dissertation's counted
gold VAD, which must be derived independently from per-participant lavalier
audio and pass the separate audio/transcript audit.

Version 3 additionally binds and fully replays the formal audio-gold-alignment
artifact.  Review, acceptance, and rasterisation are restricted to its exact
half-open 25-Hz window; wholly outside source events survive only as immutable
excluded inventory, and crossing events are consumed only on the intersection.

One source-metadata exception is explicit and versioned: legacy Transcript
``offset.json`` files wrote root ``fps`` as the JSON float token ``30.0``.
Only a manifest-declared, byte-preimage-bound normalization of that exact token
to canonical integer ``30`` is accepted; no general float-to-int coercion exists.

The trust model is integrity and reproducibility under trusted repository
code.  It is not a cryptographic signature or a substitute for human review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.data import audio_gold_alignment
from hri_turn_taking.data.source_paths import corpus_root, hpe_delivery_root

ARTIFACT_KIND = "vap_transcript_activity_reference_acceptance"
FORMAT_VERSION = 3
OUTPUT_ROLE = "coarse_transcript_activity_reference"
TIMECODE_FPS_NUMERATOR = 30
TIMECODE_FPS_DENOMINATOR = 1
OUTPUT_RATE_HZ = 25
OFFSET_RMS_HOP_SAMPLES = 512
OFFSET_PROCESSING_RATE_HZ = 16000
SPEAKERS = ("A", "B", "C", "R")
SHARED_TIME_FORMULA = "shared_time = timecode_frames / 30 + transcript_offset_sec"
HPE_TIME_FORMULA = "shared_time = stream_local_time + offset_s"
TIMECODE_QUANTIZATION_RULE = "half_frame_ceiling_to_whole_millisecond"
TIMECODE_QUANTIZATION_ERROR_BOUND_MS = (
    Decimal(1000) * TIMECODE_FPS_DENOMINATOR / (Decimal(2) * TIMECODE_FPS_NUMERATOR)
).to_integral_value(rounding=ROUND_CEILING)
OFFSET_LAG_QUANTIZATION_RULE = "discrete_rms_argmax_half_hop"
OFFSET_LAG_QUANTIZATION_ERROR_BOUND_MS = (
    Decimal(OFFSET_RMS_HOP_SAMPLES) * 1000 / (Decimal(2) * OFFSET_PROCESSING_RATE_HZ)
)
TRANSCRIPT_SIDE_ERROR_BOUND_MS = (
    TIMECODE_QUANTIZATION_ERROR_BOUND_MS + OFFSET_LAG_QUANTIZATION_ERROR_BOUND_MS
)
SOURCE_NORMALIZATION_CONTRACT = "transcript_source_normalizations_v1"
LEGACY_OFFSET_FPS_NORMALIZATION = "normalize_legacy_offset_fps_float"
LEGACY_OFFSET_FPS_NORMALIZATION_ID = "legacy-offset-fps-float-30-v1"

_TIMECODE_RE = re.compile(
    r"^(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):"
    r"(?P<second>[0-9]{2}):(?P<frame>[0-9]{2})$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_RE = re.compile(r"^[0-9]{2}_[0-9]{2}(?:_[0-9]+)?__[0-9]+$")
_REPAIR_CLASSIFICATIONS = frozenset({"mechanical", "manual_audio", "manual_owner"})
_REPAIR_OPERATIONS = frozenset(
    {
        "insert_missing_text_opening_quote",
        "normalize_timecode",
        "normalize_speaker_whitespace",
        "correct_timecode_boundary",
        "correct_speaker_identity",
        "correct_text_content",
    }
)
_TIMECODE_FIELDS = frozenset({"start_timecode", "end_timecode"})
_REVIEW_SEMANTIC_SCHEMA = "transcript_activity_reference_events_v2"
_REPAIR_SET_SCHEMA = "transcript_activity_reference_repairs_v2"
_SCOPE_SCHEMA = "transcript_gold_window_scope_v1"


class TranscriptAcceptanceError(ValueError):
    """Raised when Transcript evidence cannot be accepted safely."""


@dataclass(frozen=True)
class TranscriptSourceAuthority:
    """Trusted roots from which one scene's immutable evidence must originate.

    These roots are caller-owned authority, never manifest-controlled values.
    Tests may inject isolated roots; production callers should use
    :func:`default_source_authority`.
    """

    corpus_root: Path
    hpe_delivery_root: Path
    audio_contract_root: Path | None = None
    pose_base_root: Path | None = None
    gold_alignment_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_root", Path(self.corpus_root).resolve())
        object.__setattr__(self, "hpe_delivery_root", Path(self.hpe_delivery_root).resolve())
        injected_base = Path(self.hpe_delivery_root).resolve().parent
        object.__setattr__(
            self,
            "audio_contract_root",
            Path(self.audio_contract_root or injected_base / "audio_contract").resolve(),
        )
        object.__setattr__(
            self,
            "pose_base_root",
            Path(self.pose_base_root or injected_base / "intake_v2").resolve(),
        )
        object.__setattr__(
            self,
            "gold_alignment_root",
            Path(self.gold_alignment_root or injected_base / "audio_gold_alignment").resolve(),
        )


def default_source_authority() -> TranscriptSourceAuthority:
    """Return this workspace's non-manifest-controlled source authority."""

    project_root = Path(__file__).resolve().parents[3]
    return TranscriptSourceAuthority(
        corpus_root=corpus_root(project_root),
        hpe_delivery_root=hpe_delivery_root(project_root),
        audio_contract_root=project_root / "data" / "annotations" / "audio_contract",
        pose_base_root=project_root / "data" / "processed" / "intake_v2",
        gold_alignment_root=(project_root / "data" / "annotations" / "audio_gold_alignment"),
    )


@dataclass(frozen=True)
class TranscriptEvent:
    """One canonical event on the transcript-local integer-frame axis."""

    source_index: int
    start_frame: int
    end_frame: int
    speaker: str
    text: str


@dataclass(frozen=True)
class AcceptedTranscriptReference:
    """Validated transcript evidence and its shared-time contract."""

    scene: str
    manifest_path: Path
    output_role: str
    counted_gold_ready: bool
    transcript_offset_sec: Decimal
    timecode_quantization_error_bound_ms: Decimal
    offset_lag_quantization_error_bound_ms: Decimal
    transcript_side_error_bound_ms: Decimal
    hpe_stream: str
    hpe_offset_sec: Decimal
    hpe_error_bound_ms: Decimal
    combined_error_bound_ms: Decimal
    hpe_quality_status: str
    events: tuple[TranscriptEvent, ...]
    speaker_intervals: Mapping[str, tuple[tuple[int, int], ...]]
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_normalizations: tuple[Mapping[str, Any], ...]
    authorized_start_tick: int
    authorized_end_tick: int
    scope_summary: Mapping[str, Any]
    speaker_prevalence: Mapping[str, Any]
    excluded_event_inventory: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TranscriptActivityReference:
    """Rasterised coarse reference that is explicitly ineligible as counted gold."""

    scene: str
    output_role: str
    counted_gold_ready: bool
    shared_timestamps: tuple[Decimal, ...]
    values: Mapping[str, np.ndarray]


def _fail(message: str) -> None:
    raise TranscriptAcceptanceError(message)


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranscriptAcceptanceError(f"{label} is not strict UTF-8") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def no_nonfinite_constant(value: str) -> None:
        _fail(f"{label} contains forbidden non-finite JSON constant {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=no_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise TranscriptAcceptanceError(
            f"{label} is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys do not match the versioned contract "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{label} must be a non-empty string without surrounding whitespace")
    return value


def _require_exact_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    positive: bool = False,
) -> int:
    if isinstance(value, bool) or type(value) is not int:
        _fail(f"{label} must be an exact JSON integer (not bool or float)")
    if positive and value <= 0:
        _fail(f"{label} must be a positive integer")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _require_real_number(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a JSON number (not bool)")
    if not math.isfinite(float(value)):
        _fail(f"{label} must be finite")
    return Decimal(str(value))


def _require_decimal_string(value: Any, label: str, *, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise TranscriptAcceptanceError(f"{label} is not a decimal") from exc
    if not result.is_finite() or (nonnegative and result < 0):
        _fail(f"{label} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _resolve_bound_path(manifest_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: str | Path) -> dict[str, Any]:
    """Return the exact binding shape required by an acceptance manifest."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        _fail(f"cannot bind missing file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _scene_parts(scene: str) -> tuple[str, str]:
    if _SCENE_RE.fullmatch(scene) is None:
        _fail("acceptance manifest scene is not canonical")
    group, scenario = scene.rsplit("__", maxsplit=1)
    return group, scenario


def _canonical_source_paths(scene: str, authority: TranscriptSourceAuthority) -> dict[str, Path]:
    group, scenario = _scene_parts(scene)
    transcript_dir = authority.corpus_root / group / "transcript" / scenario
    return {
        "timeline": transcript_dir / "timeline_video.json",
        "offset": transcript_dir / "offset.json",
        "reference_wav": (
            authority.corpus_root
            / group
            / "Processed"
            / scenario
            / "video"
            / "participants_rgb_audio_edited.wav"
        ),
        "hpe_offset": authority.hpe_delivery_root / scene / "offset.json",
        "hpe_offset_quality": (authority.hpe_delivery_root / scene / "offset_quality.json"),
        "gold_alignment": (authority.gold_alignment_root / f"{scene}.audio_gold_alignment.json"),
    }


def _verify_binding(
    binding_value: Any,
    *,
    manifest_dir: Path,
    label: str,
    return_bytes: bool,
    expected_path: Path | None = None,
) -> tuple[Path, bytes | None]:
    binding = _require_object(binding_value, label)
    _require_exact_keys(binding, {"path", "size_bytes", "sha256"}, label)
    path_text = _require_nonempty_string(binding["path"], f"{label}.path")
    size = _require_exact_int(binding["size_bytes"], f"{label}.size_bytes", minimum=0)
    expected_hash = binding["sha256"]
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        _fail(f"{label}.sha256 must be a lowercase SHA-256 digest")
    path = _resolve_bound_path(manifest_dir, path_text)
    if expected_path is not None and path != expected_path.resolve():
        _fail(f"{label}.path is not the canonical authority path for the declared scene")
    if not path.is_file():
        _fail(f"{label} does not exist: {path}")
    if path.stat().st_size != size:
        _fail(f"{label} size no longer matches its manifest binding")
    if return_bytes:
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_hash:
            _fail(f"{label} bytes no longer match their manifest binding")
        return path, raw
    if _sha256_file(path) != expected_hash:
        _fail(f"{label} bytes no longer match their manifest binding")
    return path, None


def _load_gold_alignment_with_authority(
    path: Path,
    *,
    scene: str,
    authority: TranscriptSourceAuthority,
) -> dict[str, Any]:
    """Fully replay the formal channel map/window under code-owned roots."""

    gold_authority = audio_gold_alignment.GoldAlignmentAuthority(
        audio_manifest_root=authority.audio_contract_root,
        pose_base_root=authority.pose_base_root,
    )
    before = path.read_bytes()
    try:
        report = audio_gold_alignment._load_with_authority(path, authority=gold_authority)
    except (OSError, ValueError) as exc:
        raise TranscriptAcceptanceError(
            f"formal audio gold-alignment replay failed: {exc}"
        ) from exc
    if report.get("scene") != scene:
        _fail("formal audio gold-alignment scene does not match Transcript scene")
    if path.read_bytes() != before:
        _fail("formal audio gold-alignment artifact changed during replay")
    gates = _require_object(report.get("gates"), "gold alignment gates")
    authorization = _require_object(report.get("authorization"), "gold alignment authorization")
    if gates.get("alignment_prerequisite_pass") is not True:
        _fail("formal audio gold-alignment prerequisite did not pass")
    if authorization.get("gold_vad_use_authorized") is not True:
        _fail("formal audio gold-alignment does not authorize downstream use")
    return report


def _gold_window(report: Mapping[str, Any]) -> tuple[int, int, Decimal, Decimal]:
    authorization = _require_object(report.get("authorization"), "gold authorization")
    ticks = _require_object(
        authorization.get("authorized_tick_range"), "gold authorized_tick_range"
    )
    _require_exact_keys(
        ticks,
        {"frame_rate_hz", "start_inclusive", "end_exclusive", "n_ticks"},
        "gold authorized_tick_range",
    )
    if _require_exact_int(ticks["frame_rate_hz"], "gold frame_rate_hz") != OUTPUT_RATE_HZ:
        _fail("gold authorized window must use the exact 25-Hz shared grid")
    start = _require_exact_int(ticks["start_inclusive"], "gold start tick", minimum=0)
    end = _require_exact_int(ticks["end_exclusive"], "gold end tick", minimum=1)
    n_ticks = _require_exact_int(ticks["n_ticks"], "gold n_ticks", minimum=1)
    if end <= start or end - start != n_ticks:
        _fail("gold authorized tick range is empty or internally inconsistent")
    return (
        start,
        end,
        Decimal(start) / Decimal(OUTPUT_RATE_HZ),
        Decimal(end) / Decimal(OUTPUT_RATE_HZ),
    )


def parse_timecode_frames(value: str) -> int:
    """Parse strict zero-based, non-drop-frame ``HH:MM:SS:FF`` at 30 fps."""

    if not isinstance(value, str):
        _fail("timecode must be a string")
    match = _TIMECODE_RE.fullmatch(value)
    if match is None:
        _fail(f"timecode is not strict HH:MM:SS:FF: {value!r}")
    hour, minute, second, frame = (
        int(match.group(name)) for name in ("hour", "minute", "second", "frame")
    )
    if minute >= 60 or second >= 60:
        _fail(f"timecode minute/second is outside [0, 59]: {value!r}")
    if frame >= TIMECODE_FPS_NUMERATOR:
        _fail(f"timecode frame is outside [0, 29]: {value!r}")
    return (((hour * 60) + minute) * 60 + second) * TIMECODE_FPS_NUMERATOR + frame


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TranscriptAcceptanceError("value cannot be canonically hashed") from exc
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_repair_common(
    repair: Mapping[str, Any],
    label: str,
    *,
    repair_ids_seen: set[str],
    repair_ids_ordered: list[str],
) -> tuple[str, str, int, str, str, str]:
    repair_id = _require_nonempty_string(repair["repair_id"], f"{label}.repair_id")
    if repair_id in repair_ids_seen:
        _fail(f"duplicate repair_id {repair_id!r}")
    repair_ids_seen.add(repair_id)
    repair_ids_ordered.append(repair_id)
    operation = repair["operation"]
    if operation not in _REPAIR_OPERATIONS:
        _fail(f"{label}.operation is not allowed")
    source_index = _require_exact_int(repair["source_index"], f"{label}.source_index", minimum=0)
    field = _require_nonempty_string(repair["field"], f"{label}.field")
    expected = repair["expected"]
    replacement = repair["replacement"]
    if not isinstance(expected, str) or not isinstance(replacement, str):
        _fail(f"{label} expected/replacement must both be strings")
    if not expected or expected == replacement:
        _fail(f"{label}.replacement must differ from a non-empty expected preimage")
    classification = repair["classification"]
    if classification not in _REPAIR_CLASSIFICATIONS:
        _fail(f"{label}.classification is not allowed")
    _require_nonempty_string(repair["evidence"], f"{label}.evidence")
    if repair["review_status"] != "confirmed":
        _fail(f"{label} is not explicitly confirmed")
    return repair_id, operation, source_index, field, expected, replacement


def _mechanically_normalize_timecode(value: str) -> str:
    trimmed = value.strip(" ")
    if value.strip() != trimmed:
        _fail("mechanical timecode normalization only permits ASCII edge spaces")
    match = _TIMECODE_RE.fullmatch(trimmed)
    if match is None:
        _fail("mechanical timecode normalization requires HH:MM:SS:FF digits")
    hour, minute, second, frame = (
        int(match.group(name)) for name in ("hour", "minute", "second", "frame")
    )
    if minute >= 60 or second >= 60 or frame > TIMECODE_FPS_NUMERATOR:
        _fail("mechanical timecode normalization cannot guess this invalid timecode")
    if frame == TIMECODE_FPS_NUMERATOR:
        total_frames = (((hour * 60) + minute) * 60 + second) * TIMECODE_FPS_NUMERATOR + frame
        hour, remainder = divmod(total_frames, 60 * 60 * TIMECODE_FPS_NUMERATOR)
        minute, remainder = divmod(remainder, 60 * TIMECODE_FPS_NUMERATOR)
        second, frame = divmod(remainder, TIMECODE_FPS_NUMERATOR)
        if hour > 99:
            _fail("normalized timecode exceeds the two-digit hour contract")
        trimmed = f"{hour:02d}:{minute:02d}:{second:02d}:{frame:02d}"
    return trimmed


def _apply_repair_operations(raw: bytes, repairs_value: Any) -> tuple[list[Any], tuple[str, ...]]:
    repairs = _require_object(repairs_value, "repairs")
    _require_exact_keys(repairs, {"operations"}, "repairs")
    operations = repairs["operations"]
    if not isinstance(operations, list):
        _fail("repairs.operations must be a list")

    common_keys = {
        "repair_id",
        "operation",
        "source_index",
        "field",
        "expected",
        "replacement",
        "classification",
        "evidence",
        "review_status",
    }
    repair_ids_seen: set[str] = set()
    repair_ids_ordered: list[str] = []
    targets: set[tuple[int, str]] = set()
    raw_repairs: list[tuple[int, bytes, bytes, str, int, str]] = []
    field_repairs: list[tuple[str, int, str, str, str, str]] = []

    for index, value in enumerate(operations):
        label = f"repairs.operations[{index}]"
        repair = _require_object(value, label)
        operation = repair.get("operation")
        expected_keys = (
            common_keys | {"byte_offset"}
            if operation == "insert_missing_text_opening_quote"
            else common_keys
        )
        _require_exact_keys(repair, expected_keys, label)
        (
            repair_id,
            operation,
            source_index,
            field,
            expected,
            replacement,
        ) = _validate_repair_common(
            repair,
            label,
            repair_ids_seen=repair_ids_seen,
            repair_ids_ordered=repair_ids_ordered,
        )
        target = (source_index, field)
        if target in targets:
            _fail(f"multiple repairs target event {source_index} field {field!r}")
        targets.add(target)

        if operation == "insert_missing_text_opening_quote":
            if field != "text" or repair["classification"] != "mechanical":
                _fail(f"{label} must be a mechanical repair of the text field")
            prefix = '"text": '
            if not expected.startswith(prefix):
                _fail(f"{label} expected fragment must start with the exact text key")
            if replacement != prefix + '"' + expected[len(prefix) :]:
                _fail(f"{label} may insert only the missing opening text quote")
            if len(expected.encode("utf-8")) > 2048:
                _fail(f"{label} is too broad for a narrow syntax repair")
            fragment = _require_object(
                _strict_json_loads(
                    ("{" + replacement + "}").encode("utf-8"),
                    f"{label} replacement fragment",
                ),
                f"{label} replacement fragment",
            )
            _require_exact_keys(fragment, {"text"}, f"{label} replacement fragment")
            semantic_text = fragment["text"]
            if not isinstance(semantic_text, str) or not semantic_text.strip():
                _fail(f"{label} replacement must yield non-empty text")
            offset = _require_exact_int(repair["byte_offset"], f"{label}.byte_offset", minimum=0)
            raw_repairs.append(
                (
                    offset,
                    expected.encode("utf-8"),
                    replacement.encode("utf-8"),
                    repair_id,
                    source_index,
                    semantic_text,
                )
            )
            continue

        if operation == "normalize_timecode":
            if field not in _TIMECODE_FIELDS or repair["classification"] != "mechanical":
                _fail(f"{label} must mechanically normalize a timecode field")
            if replacement != _mechanically_normalize_timecode(expected):
                _fail(f"{label} replacement is not the unique mechanical normalization")
            parse_timecode_frames(replacement)
        elif operation == "normalize_speaker_whitespace":
            if field != "speaker" or repair["classification"] != "mechanical":
                _fail(f"{label} must mechanically normalize speaker whitespace")
            if expected.strip(" ") != replacement or expected.strip() != replacement:
                _fail(f"{label} may remove only surrounding ASCII spaces")
            if replacement not in SPEAKERS:
                _fail(f"{label} normalized speaker is not A/B/C/R")
        elif operation == "correct_timecode_boundary":
            if field not in _TIMECODE_FIELDS or repair["classification"] not in {
                "manual_audio",
                "manual_owner",
            }:
                _fail(f"{label} must be a manually evidenced timecode correction")
            parse_timecode_frames(replacement)
        elif operation == "correct_speaker_identity":
            if field != "speaker" or repair["classification"] not in {
                "manual_audio",
                "manual_owner",
            }:
                _fail(f"{label} must be a manually evidenced speaker correction")
            if replacement not in SPEAKERS:
                _fail(f"{label} corrected speaker is not A/B/C/R")
        elif operation == "correct_text_content":
            if field != "text" or repair["classification"] != "manual_owner":
                _fail(f"{label} must be a manual-owner text correction")
            _require_nonempty_string(expected, f"{label}.expected")
            _require_nonempty_string(replacement, f"{label}.replacement")
        else:  # pragma: no cover - guarded by the allowed-operation set
            _fail(f"unsupported repair operation {operation!r}")
        field_repairs.append((repair_id, source_index, field, expected, replacement, operation))

    occupied: list[tuple[int, int, str]] = []
    for offset, expected, _replacement, repair_id, _source_index, _text in raw_repairs:
        end = offset + len(expected)
        if end > len(raw) or raw[offset:end] != expected:
            _fail(f"repair {repair_id!r} exact byte preimage does not match")
        for other_start, other_end, other_id in occupied:
            if offset < other_end and other_start < end:
                _fail(f"repairs {repair_id!r} and {other_id!r} overlap")
        occupied.append((offset, end, repair_id))

    patched = raw
    for offset, expected, replacement, _repair_id, _source_index, _text in sorted(
        raw_repairs, reverse=True
    ):
        patched = patched[:offset] + replacement + patched[offset + len(expected) :]
    timeline = _strict_json_loads(patched, "repaired timeline")
    if not isinstance(timeline, list):
        _fail("repaired timeline must contain a JSON array")

    for _offset, _expected, _replacement, repair_id, source_index, semantic_text in raw_repairs:
        if source_index >= len(timeline):
            _fail(f"repair {repair_id!r} source_index is outside the event list")
        event = _require_object(timeline[source_index], f"timeline event {source_index}")
        if event.get("text") != semantic_text:
            _fail(f"repair {repair_id!r} does not bind the declared event text")

    for repair_id, source_index, field, expected, replacement, _operation in field_repairs:
        if source_index >= len(timeline):
            _fail(f"repair {repair_id!r} source_index is outside the event list")
        event = _require_object(timeline[source_index], f"timeline event {source_index}")
        if event.get(field) != expected:
            _fail(f"repair {repair_id!r} exact field preimage does not match")
        event[field] = replacement
    return timeline, tuple(repair_ids_ordered)


def _validate_timecode_contract(value: Any) -> None:
    contract = _require_object(value, "timecode_contract")
    _require_exact_keys(
        contract,
        {"fps_numerator", "fps_denominator", "drop_frame", "frame_index_origin"},
        "timecode_contract",
    )
    if _require_exact_int(contract["fps_numerator"], "timecode_contract.fps_numerator") != 30:
        _fail("timecode_contract.fps_numerator must be integer 30")
    if _require_exact_int(contract["fps_denominator"], "timecode_contract.fps_denominator") != 1:
        _fail("timecode_contract.fps_denominator must be integer 1")
    if contract["drop_frame"] is not False:
        _fail("timecode_contract.drop_frame must be false")
    if (
        _require_exact_int(contract["frame_index_origin"], "timecode_contract.frame_index_origin")
        != 0
    ):
        _fail("timecode_contract.frame_index_origin must be integer 0")


def _validate_event_summary(value: Any, expected: Mapping[str, Any]) -> None:
    summary = _require_object(value, "manual_review.event_summary")
    _require_exact_keys(
        summary,
        {
            "n_events",
            "speaker_event_counts",
            "minimum_start_frame",
            "maximum_end_frame",
        },
        "manual_review.event_summary",
    )
    _require_exact_int(summary["n_events"], "manual_review.event_summary.n_events", minimum=1)
    _require_exact_int(
        summary["minimum_start_frame"],
        "manual_review.event_summary.minimum_start_frame",
        minimum=0,
    )
    _require_exact_int(
        summary["maximum_end_frame"],
        "manual_review.event_summary.maximum_end_frame",
        minimum=1,
    )
    counts = _require_object(
        summary["speaker_event_counts"],
        "manual_review.event_summary.speaker_event_counts",
    )
    _require_exact_keys(counts, set(SPEAKERS), "manual_review.event_summary.speaker_event_counts")
    for speaker in SPEAKERS:
        _require_exact_int(
            counts[speaker],
            f"manual_review.event_summary.speaker_event_counts.{speaker}",
            minimum=0,
        )
    if summary != expected:
        _fail("manual_review.event_summary does not match repaired canonical events")


def _validate_manual_review(
    value: Any,
    *,
    timeline_sha256: str,
    canonical_semantic_sha256: str,
    event_summary: Mapping[str, Any],
    repair_set_sha256: str,
    repair_ids: Sequence[str],
) -> None:
    review = _require_object(value, "manual_review")
    _require_exact_keys(
        review,
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
    if review["status"] != "confirmed":
        _fail("manual_review.status must be confirmed")
    _require_nonempty_string(review["reviewer"], "manual_review.reviewer")
    reviewed_at = _require_nonempty_string(review["reviewed_at"], "manual_review.reviewed_at")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TranscriptAcceptanceError(
            "manual_review.reviewed_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed_reviewed_at.tzinfo is None:
        _fail("manual_review.reviewed_at must include a UTC offset")
    _require_nonempty_string(review["method"], "manual_review.method")
    if review["timeline_sha256"] != timeline_sha256:
        _fail("manual review is not bound to the current raw timeline SHA-256")
    if (
        not isinstance(review["canonical_semantic_sha256"], str)
        or not _SHA256_RE.fullmatch(review["canonical_semantic_sha256"])
        or review["canonical_semantic_sha256"] != canonical_semantic_sha256
    ):
        _fail("manual review is not bound to repaired canonical event semantics")
    _validate_event_summary(review["event_summary"], event_summary)
    if (
        not isinstance(review["repair_set_sha256"], str)
        or not _SHA256_RE.fullmatch(review["repair_set_sha256"])
        or review["repair_set_sha256"] != repair_set_sha256
    ):
        _fail("manual review is not bound to the declared repair set")
    declared_ids = review["repair_ids"]
    if not isinstance(declared_ids, list) or any(
        not isinstance(item, str) or not item for item in declared_ids
    ):
        _fail("manual_review.repair_ids must be a list of non-empty strings")
    if declared_ids != list(repair_ids):
        _fail("manual_review.repair_ids do not match the declared repair set")
    for key in (
        "all_in_scope_events_reviewed",
        "speaker_assignments_reviewed",
        "time_boundaries_reviewed",
    ):
        if review[key] is not True:
            _fail(f"manual_review.{key} must be the JSON boolean true")


def _normalize_transcript_offset_source(
    raw: bytes,
    value: Any,
    normalizations_value: Any,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    offset = _require_object(value, "transcript offset")
    normalizations = _require_object(normalizations_value, "source_normalizations")
    _require_exact_keys(
        normalizations,
        {"contract", "version", "operations"},
        "source_normalizations",
    )
    if normalizations["contract"] != SOURCE_NORMALIZATION_CONTRACT:
        _fail("source_normalizations.contract is unsupported")
    if _require_exact_int(normalizations["version"], "source_normalizations.version") != 1:
        _fail("source_normalizations.version is unsupported")
    operations = normalizations["operations"]
    if not isinstance(operations, list):
        _fail("source_normalizations.operations must be a list")

    raw_fps = offset.get("fps")
    if not operations:
        if type(raw_fps) is not int:
            _fail(
                "legacy transcript offset fps 30.0 requires an explicitly declared "
                "source normalization"
            )
        return offset, ()
    if len(operations) != 1:
        _fail("only one narrowly scoped transcript offset fps normalization is allowed")

    operation = _require_object(operations[0], "source_normalizations.operations[0]")
    _require_exact_keys(
        operation,
        {
            "normalization_id",
            "operation",
            "source",
            "json_path",
            "byte_offset",
            "expected_json_token",
            "expected_json_type",
            "canonical_json_type",
            "canonical_value",
            "evidence",
            "review_status",
        },
        "source_normalizations.operations[0]",
    )
    if operation["normalization_id"] != LEGACY_OFFSET_FPS_NORMALIZATION_ID:
        _fail("legacy fps normalization_id does not match the versioned contract")
    if operation["operation"] != LEGACY_OFFSET_FPS_NORMALIZATION:
        _fail("source normalization operation is unsupported")
    if operation["source"] != "offset" or operation["json_path"] != "$.fps":
        _fail("source normalization may target only transcript offset $.fps")
    if operation["expected_json_token"] != "30.0":
        _fail("legacy fps normalization exact token must be 30.0")
    if operation["expected_json_type"] != "float":
        _fail("legacy fps normalization expected type must be float")
    if operation["canonical_json_type"] != "integer":
        _fail("legacy fps normalization canonical type must be integer")
    if (
        _require_exact_int(
            operation["canonical_value"],
            "source_normalizations.operations[0].canonical_value",
        )
        != 30
    ):
        _fail("legacy fps normalization canonical value must be exact integer 30")
    _require_nonempty_string(operation["evidence"], "source_normalizations.operations[0].evidence")
    if operation["review_status"] != "confirmed":
        _fail("source normalization must be explicitly confirmed")
    if type(raw_fps) is not float or not math.isfinite(raw_fps) or raw_fps != 30.0:
        _fail("declared legacy fps normalization requires raw JSON float 30.0")

    token_pattern = re.compile(rb'"fps"[ \t\r\n]*:[ \t\r\n]*(?P<token>[^,}\]\s]+)')
    matches = list(token_pattern.finditer(raw))
    if len(matches) != 1:
        _fail("transcript offset must contain one uniquely locatable root fps token")
    token_match = matches[0]
    token_start = token_match.start("token")
    token = token_match.group("token")
    declared_offset = _require_exact_int(
        operation["byte_offset"],
        "source_normalizations.operations[0].byte_offset",
        minimum=0,
    )
    if declared_offset != token_start or token != b"30.0":
        _fail("legacy fps normalization exact byte preimage does not match $.fps")

    canonical = dict(offset)
    canonical["fps"] = 30
    return canonical, (dict(operation),)


def _validate_transcript_offset(
    value: Any, *, scene: str, reference_wav: Path
) -> tuple[Decimal, int]:
    offset = _require_object(value, "transcript offset")
    required = {
        "session",
        "timeline_reference",
        "frame_reference",
        "fps",
        "offset_sec",
        "offset_samples_processing_sr",
        "lag_frames_rms",
        "master_sr",
        "reference_sr",
        "processing_sr",
        "reference_duration_sec",
        "interpretation",
        "speaker_alignment",
    }
    _require_exact_keys(offset, required, "transcript offset")
    group, session = _scene_parts(scene)
    if offset["session"] != session:
        _fail("transcript offset session does not match the manifest scene")
    timeline_reference = _require_nonempty_string(
        offset["timeline_reference"], "transcript offset.timeline_reference"
    )
    if (
        timeline_reference != "participants_rgb_audio_edited.wav"
        or reference_wav.name != timeline_reference
    ):
        _fail("reference WAV binding does not match offset.timeline_reference")
    if offset["frame_reference"] != "participants_rgb_audio_edited.mp4":
        _fail("transcript offset.frame_reference is not the frozen reference video")
    if _require_exact_int(offset["fps"], "transcript offset.fps") != 30:
        _fail("transcript offset fps must equal exact 30/1")
    processing_sr = _require_exact_int(
        offset["processing_sr"], "transcript offset.processing_sr", positive=True
    )
    if processing_sr != OFFSET_PROCESSING_RATE_HZ:
        _fail("transcript offset.processing_sr must match the frozen 16-kHz contract")
    for key in ("master_sr", "reference_sr"):
        _require_exact_int(offset[key], f"transcript offset.{key}", positive=True)
    offset_sec = _require_real_number(offset["offset_sec"], "transcript offset.offset_sec")
    shift = _require_exact_int(
        offset["offset_samples_processing_sr"], "offset_samples_processing_sr"
    )
    lag = _require_exact_int(offset["lag_frames_rms"], "lag_frames_rms")
    expected_shift = int((offset_sec * processing_sr).to_integral_value(rounding=ROUND_HALF_EVEN))
    if shift != expected_shift:
        _fail("transcript offset sample shift is inconsistent with offset_sec")
    if offset_sec != Decimal(lag * OFFSET_RMS_HOP_SAMPLES) / Decimal(processing_sr):
        _fail("transcript offset_sec is inconsistent with the recorded RMS lag")
    duration = _require_real_number(
        offset["reference_duration_sec"], "transcript offset.reference_duration_sec"
    )
    if duration <= 0:
        _fail("transcript reference duration must be positive")
    interpretation = _require_object(offset["interpretation"], "transcript offset.interpretation")
    _require_exact_keys(
        interpretation, {"offset_positive", "offset_negative"}, "transcript offset.interpretation"
    )
    if interpretation != {
        "offset_positive": "crop beginning of A/B/C/R aligned wavs",
        "offset_negative": "pad silence at beginning of A/B/C/R aligned wavs",
    }:
        _fail("transcript offset interpretation does not match the generator contract")
    alignment = _require_object(offset["speaker_alignment"], "transcript offset.speaker_alignment")
    if set(alignment) != set(SPEAKERS):
        _fail("transcript offset must contain exactly A/B/C/R speaker alignments")
    expected_alignment_keys = {
        "src",
        "dst",
        "sr",
        "offset_sec",
        "shift_samples",
        "action",
        "output_len_samples",
        "output_duration_sec",
    }
    for speaker in SPEAKERS:
        item = _require_object(alignment[speaker], f"speaker_alignment.{speaker}")
        _require_exact_keys(item, expected_alignment_keys, f"speaker_alignment.{speaker}")
        expected_src = f"{group}/{session}/audio/aligned_audio/{speaker}_aligned.wav"
        expected_dst = f"{group}/{session}/audio/aligned_audio/{speaker}_edited.wav"
        if item["src"] != expected_src or item["dst"] != expected_dst:
            _fail(f"speaker_alignment.{speaker} paths do not match the manifest scene")
        if _require_exact_int(item["sr"], f"speaker_alignment.{speaker}.sr") != processing_sr:
            _fail(f"speaker_alignment.{speaker}.sr must equal processing_sr")
        alignment_offset = _require_real_number(
            item["offset_sec"], f"speaker_alignment.{speaker}.offset_sec"
        )
        if alignment_offset != offset_sec:
            _fail(f"speaker_alignment.{speaker}.offset_sec is inconsistent")
        if (
            _require_exact_int(item["shift_samples"], f"speaker_alignment.{speaker}.shift_samples")
            != shift
        ):
            _fail(f"speaker_alignment.{speaker}.shift_samples is inconsistent")
        expected_action = (
            f"crop {shift} samples"
            if shift > 0
            else f"pad {-shift} samples"
            if shift < 0
            else "no shift"
        )
        if item["action"] != expected_action:
            _fail(f"speaker_alignment.{speaker}.action is inconsistent")
        output_len = _require_exact_int(
            item["output_len_samples"],
            f"speaker_alignment.{speaker}.output_len_samples",
            positive=True,
        )
        output_duration = _require_real_number(
            item["output_duration_sec"], f"speaker_alignment.{speaker}.output_duration_sec"
        )
        if (
            output_duration != Decimal(output_len) / Decimal(processing_sr)
            or output_duration != duration
        ):
            _fail(f"speaker_alignment.{speaker} duration is inconsistent")
    return offset_sec, processing_sr


def _validate_hpe_timing(
    hpe_offset_value: Any,
    hpe_quality_value: Any,
    timing_value: Any,
    *,
    transcript_offset_sec: Decimal,
    transcript_processing_sr: int,
) -> tuple[
    str,
    Decimal,
    Decimal,
    Decimal,
    Decimal,
    Decimal,
    Decimal,
    str,
]:
    timing = _require_object(timing_value, "timing")
    _require_exact_keys(
        timing,
        {
            "status",
            "method",
            "formula",
            "transcript_offset_sec",
            "timecode_quantization_rule",
            "timecode_quantization_error_bound_ms",
            "offset_lag_quantization_rule",
            "offset_lag_hop_samples",
            "offset_lag_processing_rate_hz",
            "offset_lag_quantization_error_bound_ms",
            "transcript_side_error_bound_ms",
            "hpe_stream",
            "hpe_offset_sec",
            "hpe_error_bound_ms",
            "hpe_quality_status",
            "combined_error_bound_ms",
            "uncertainty_combination",
            "uncertainty_accepted",
        },
        "timing",
    )
    if timing["status"] != "confirmed":
        _fail("timing.status must be confirmed")
    _require_nonempty_string(timing["method"], "timing.method")
    if timing["formula"] != SHARED_TIME_FORMULA:
        _fail("timing.formula does not match the frozen transcript formula")
    declared_transcript_offset = _require_decimal_string(
        timing["transcript_offset_sec"], "timing.transcript_offset_sec"
    )
    if declared_transcript_offset != transcript_offset_sec:
        _fail("timing transcript offset does not match offset.json")
    if timing["timecode_quantization_rule"] != TIMECODE_QUANTIZATION_RULE:
        _fail("timing.timecode_quantization_rule does not match the frozen rule")
    timecode_uncertainty = _require_decimal_string(
        timing["timecode_quantization_error_bound_ms"],
        "timing.timecode_quantization_error_bound_ms",
        nonnegative=True,
    )
    if timecode_uncertainty != TIMECODE_QUANTIZATION_ERROR_BOUND_MS:
        _fail("timecode uncertainty must be recomputed from the frozen 30-fps contract")
    if timing["offset_lag_quantization_rule"] != OFFSET_LAG_QUANTIZATION_RULE:
        _fail("timing.offset_lag_quantization_rule does not match the frozen rule")
    declared_hop = _require_exact_int(
        timing["offset_lag_hop_samples"], "timing.offset_lag_hop_samples"
    )
    if declared_hop != OFFSET_RMS_HOP_SAMPLES:
        _fail("timing offset-lag hop must equal the frozen 512 samples")
    declared_processing_sr = _require_exact_int(
        timing["offset_lag_processing_rate_hz"],
        "timing.offset_lag_processing_rate_hz",
    )
    if (
        declared_processing_sr != OFFSET_PROCESSING_RATE_HZ
        or declared_processing_sr != transcript_processing_sr
    ):
        _fail("timing offset-lag processing rate must match the frozen 16-kHz source")
    offset_lag_uncertainty = _require_decimal_string(
        timing["offset_lag_quantization_error_bound_ms"],
        "timing.offset_lag_quantization_error_bound_ms",
        nonnegative=True,
    )
    recomputed_offset_lag_uncertainty = (
        Decimal(declared_hop) * 1000 / (Decimal(2) * declared_processing_sr)
    )
    if (
        offset_lag_uncertainty != recomputed_offset_lag_uncertainty
        or offset_lag_uncertainty != OFFSET_LAG_QUANTIZATION_ERROR_BOUND_MS
    ):
        _fail("offset-lag uncertainty must equal the frozen RMS half-hop bound")
    transcript_side_bound = _require_decimal_string(
        timing["transcript_side_error_bound_ms"],
        "timing.transcript_side_error_bound_ms",
        nonnegative=True,
    )
    if (
        transcript_side_bound != timecode_uncertainty + offset_lag_uncertainty
        or transcript_side_bound != TRANSCRIPT_SIDE_ERROR_BOUND_MS
    ):
        _fail("transcript-side uncertainty must equal 17 ms + 16 ms = 33 ms")
    hpe_stream = _require_nonempty_string(timing["hpe_stream"], "timing.hpe_stream")
    if hpe_stream != "6cam":
        _fail("timing.hpe_stream must be the accepted 6cam stream")
    hpe_offset = _require_decimal_string(timing["hpe_offset_sec"], "timing.hpe_offset_sec")
    hpe_bound = _require_decimal_string(
        timing["hpe_error_bound_ms"], "timing.hpe_error_bound_ms", nonnegative=True
    )
    combined_bound = _require_decimal_string(
        timing["combined_error_bound_ms"],
        "timing.combined_error_bound_ms",
        nonnegative=True,
    )
    if timing["uncertainty_combination"] != "conservative_sum":
        _fail("timing.uncertainty_combination must be conservative_sum")
    if combined_bound != transcript_side_bound + hpe_bound:
        _fail("combined timing uncertainty must equal transcript-side 33 ms plus HPE")
    if timing["uncertainty_accepted"] is not True:
        _fail("timing.uncertainty_accepted must be the JSON boolean true")

    hpe_offset_obj = _require_object(hpe_offset_value, "HPE offset")
    streams = _require_object(hpe_offset_obj.get("streams"), "HPE offset.streams")
    if "mic_A" not in streams or hpe_stream not in streams:
        _fail("HPE offset lacks mic_A or the declared 6cam stream")
    if _require_real_number(streams["mic_A"], "HPE mic_A offset") != 0:
        _fail("HPE mic_A anchor offset must be exactly zero")
    if _require_real_number(streams[hpe_stream], "HPE 6cam offset") != hpe_offset:
        _fail("manifest HPE offset does not match HPE offset.json")

    quality = _require_object(hpe_quality_value, "HPE offset quality")
    if quality.get("convention") != HPE_TIME_FORMULA:
        _fail("HPE offset quality uses an unexpected timebase convention")
    timebase_keys = _require_object(quality.get("timebase_keys"), "HPE timebase_keys")
    if "mic_A" not in timebase_keys or hpe_stream not in timebase_keys:
        _fail("HPE offset quality lacks mic_A/6cam timebase keys")
    for key, value in timebase_keys.items():
        _require_nonempty_string(key, "HPE timebase key")
        _require_nonempty_string(value, f"HPE timebase_keys.{key}")
    quality_by_stream = _require_object(quality.get("quality"), "HPE quality")
    mic_quality = _require_object(quality_by_stream.get("mic_A"), "HPE quality.mic_A")
    if _require_real_number(mic_quality.get("offset_s"), "HPE quality.mic_A.offset_s") != 0:
        _fail("HPE quality mic_A anchor must be zero")
    stream_quality = _require_object(quality_by_stream.get(hpe_stream), f"HPE quality.{hpe_stream}")
    quality_offset = _require_real_number(
        stream_quality.get("offset_s"), "HPE quality.6cam.offset_s"
    )
    if quality_offset != hpe_offset:
        _fail("HPE quality offset does not match HPE offset.json")
    if stream_quality.get("self_check_ok") is not True:
        _fail("HPE quality.6cam.self_check_ok must be the JSON boolean true")
    actual_bound = _require_real_number(
        stream_quality.get("error_bound_ms"), "HPE quality.6cam.error_bound_ms"
    )
    if actual_bound < 0 or actual_bound != hpe_bound:
        _fail("manifest HPE error bound does not match HPE quality")
    actual_status = stream_quality.get("status")
    if actual_status not in {"ok", "degraded"}:
        _fail("HPE 6cam quality status must be ok or degraded")
    if timing["hpe_quality_status"] != actual_status:
        _fail("manifest HPE quality status does not match HPE quality")
    return (
        hpe_stream,
        hpe_offset,
        timecode_uncertainty,
        offset_lag_uncertainty,
        transcript_side_bound,
        hpe_bound,
        combined_bound,
        actual_status,
    )


def _canonical_events(
    timeline_value: Any,
    *,
    reference_duration_sec: Decimal,
    source_indices: set[int] | None = None,
    transcript_offset_sec: Decimal | None = None,
    authorized_start_s: Decimal | None = None,
    authorized_end_s: Decimal | None = None,
) -> tuple[tuple[TranscriptEvent, ...], dict[str, tuple[tuple[int, int], ...]]]:
    if not isinstance(timeline_value, list):
        _fail("timeline must contain a JSON array")
    events: list[TranscriptEvent] = []
    required = {"start_timecode", "end_timecode", "speaker", "text"}
    reference_end_frames = reference_duration_sec * TIMECODE_FPS_NUMERATOR
    for source_index, value in enumerate(timeline_value):
        if source_indices is not None and source_index not in source_indices:
            continue
        label = f"timeline event {source_index}"
        event = _require_object(value, label)
        _require_exact_keys(event, required, label)
        speaker = event["speaker"]
        if speaker not in SPEAKERS:
            _fail(f"{label} speaker must be exactly one of A/B/C/R")
        text = event["text"]
        if not isinstance(text, str) or not text.strip():
            _fail(f"{label} text must be a non-empty string")
        start = parse_timecode_frames(event["start_timecode"])
        end = parse_timecode_frames(event["end_timecode"])
        if end <= start:
            _fail(f"{label} must have a strictly positive half-open interval")
        if Decimal(end) > reference_end_frames:
            _fail(f"{label} extends beyond the reference duration")
        if (
            transcript_offset_sec is not None
            and authorized_start_s is not None
            and authorized_end_s is not None
        ):
            relation, _crossing, _reason = _scope_relation(
                start,
                end,
                transcript_offset_sec=transcript_offset_sec,
                authorized_start_s=authorized_start_s,
                authorized_end_s=authorized_end_s,
            )
            if relation != "in_scope":
                _fail(f"{label} no longer intersects the authorized gold window")
        events.append(TranscriptEvent(source_index, start, end, speaker, text))

    ordered = tuple(
        sorted(
            events,
            key=lambda item: (item.start_frame, item.end_frame, item.source_index),
        )
    )
    intervals: dict[str, tuple[tuple[int, int], ...]] = {}
    for speaker in SPEAKERS:
        source = sorted(
            ((item.start_frame, item.end_frame) for item in ordered if item.speaker == speaker),
            key=lambda item: (item[0], item[1]),
        )
        merged: list[list[int]] = []
        for start, end in source:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        intervals[speaker] = tuple((start, end) for start, end in merged)
    return ordered, intervals


def _scope_relation(
    start_frame: int,
    end_frame: int,
    *,
    transcript_offset_sec: Decimal,
    authorized_start_s: Decimal,
    authorized_end_s: Decimal,
) -> tuple[str, bool, str]:
    """Classify one event against the formal half-open gold window.

    A non-positive source interval is tolerated only when both of its typed
    boundaries prove that it is wholly before or wholly after the authorised
    window.  Otherwise it remains in scope so that an explicit reviewed repair
    is required before acceptance.
    """

    start_shared = Decimal(start_frame) / TIMECODE_FPS_NUMERATOR + transcript_offset_sec
    end_shared = Decimal(end_frame) / TIMECODE_FPS_NUMERATOR + transcript_offset_sec
    if end_frame <= start_frame:
        if start_shared >= authorized_end_s and end_shared >= authorized_end_s:
            return "excluded", False, "nonpositive_interval_wholly_after_window"
        if start_shared <= authorized_start_s and end_shared <= authorized_start_s:
            return "excluded", False, "nonpositive_interval_wholly_before_window"
        return "in_scope", False, "nonpositive_interval_requires_in_scope_review"
    if end_shared <= authorized_start_s:
        return "excluded", False, "event_wholly_before_window"
    if start_shared >= authorized_end_s:
        return "excluded", False, "event_wholly_after_window"
    crossing = start_shared < authorized_start_s or end_shared > authorized_end_s
    return "in_scope", crossing, "event_intersects_window"


def _validated_scope_snapshot(value: Any, source_index: int) -> dict[str, Any]:
    label = f"timeline event {source_index}"
    event = _require_object(value, label)
    _require_exact_keys(
        event,
        {"start_timecode", "end_timecode", "speaker", "text"},
        label,
    )
    speaker = event["speaker"]
    if speaker not in SPEAKERS:
        _fail(f"{label} speaker must be exactly one of A/B/C/R")
    text = event["text"]
    if not isinstance(text, str) or not text.strip():
        _fail(f"{label} text must be a non-empty string")
    start_value = _require_nonempty_string(event["start_timecode"], f"{label}.start")
    end_value = _require_nonempty_string(event["end_timecode"], f"{label}.end")
    start = parse_timecode_frames(start_value)
    end = parse_timecode_frames(end_value)
    return {
        "start_timecode": start_value,
        "end_timecode": end_value,
        "speaker": speaker,
        "text": text,
        "start_frame": start,
        "end_frame": end,
    }


def _speaker_prevalence_from_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(snapshots)
    return {
        speaker: {
            "event_count": sum(item["speaker"] == speaker for item in snapshots),
            "event_fraction": (
                _canonical_decimal(
                    Decimal(sum(item["speaker"] == speaker for item in snapshots)) / Decimal(total)
                )
                if total
                else "0"
            ),
        }
        for speaker in SPEAKERS
    }


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        _fail("cannot encode a non-finite Decimal")
    if value == 0:
        return "0"
    rendered = format(value.quantize(Decimal("0.000000001")), "f")
    return rendered.rstrip("0").rstrip(".")


def _scope_material(
    timeline_value: Any,
    *,
    transcript_offset_sec: Decimal,
    authorized_start_tick: int,
    authorized_end_tick: int,
) -> tuple[
    set[int],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
]:
    if not isinstance(timeline_value, list):
        _fail("timeline must contain a JSON array")
    authorized_start_s = Decimal(authorized_start_tick) / Decimal(OUTPUT_RATE_HZ)
    authorized_end_s = Decimal(authorized_end_tick) / Decimal(OUTPUT_RATE_HZ)
    snapshots: list[dict[str, Any]] = []
    in_scope: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    indices: set[int] = set()
    crossing = 0
    for source_index, value in enumerate(timeline_value):
        snapshot = _validated_scope_snapshot(value, source_index)
        snapshots.append(snapshot)
        relation, is_crossing, reason = _scope_relation(
            snapshot["start_frame"],
            snapshot["end_frame"],
            transcript_offset_sec=transcript_offset_sec,
            authorized_start_s=authorized_start_s,
            authorized_end_s=authorized_end_s,
        )
        semantic = {
            key: snapshot[key] for key in ("start_timecode", "end_timecode", "speaker", "text")
        }
        if relation == "in_scope":
            indices.add(source_index)
            in_scope.append(snapshot)
            crossing += int(is_crossing)
        else:
            excluded.append(
                {
                    "event_id": f"event-{source_index:06d}",
                    "source_index": source_index,
                    "source": semantic,
                    "source_semantic_sha256": _canonical_json_sha256(semantic),
                    "exclusion_reason": reason,
                }
            )
    summary = {
        "schema": _SCOPE_SCHEMA,
        "frame_rate_hz": OUTPUT_RATE_HZ,
        "authorized_start_tick_inclusive": authorized_start_tick,
        "authorized_end_tick_exclusive": authorized_end_tick,
        "source_event_count": len(snapshots),
        "in_scope_pre_review_event_count": len(in_scope),
        "excluded_event_count": len(excluded),
        "crossing_event_count": crossing,
        "source_count_conserved": len(snapshots) == len(in_scope) + len(excluded),
    }
    prevalence = {
        "source_pre_review": _speaker_prevalence_from_snapshots(snapshots),
        "in_scope_pre_review": _speaker_prevalence_from_snapshots(in_scope),
    }
    return indices, tuple(excluded), summary, prevalence


def _canonical_review_material(
    raw_timeline: bytes,
    repairs: Any,
    *,
    reference_duration_sec: Decimal,
    transcript_offset_sec: Decimal,
    authorized_start_tick: int,
    authorized_end_tick: int,
) -> tuple[
    tuple[TranscriptEvent, ...],
    dict[str, tuple[tuple[int, int], ...]],
    dict[str, Any],
]:
    repairs_object = _require_object(repairs, "repairs")
    operations = repairs_object.get("operations")
    if not isinstance(operations, list):
        _fail("repairs.operations must be a list")
    operation_objects = [
        _require_object(item, f"repairs.operations[{index}]")
        for index, item in enumerate(operations)
    ]
    mechanical = [item for item in operation_objects if item.get("classification") == "mechanical"]
    timeline_pre_review, _mechanical_ids = _apply_repair_operations(
        raw_timeline, {"operations": mechanical}
    )
    in_scope_indices, excluded_inventory, scope_summary, prevalence = _scope_material(
        timeline_pre_review,
        transcript_offset_sec=transcript_offset_sec,
        authorized_start_tick=authorized_start_tick,
        authorized_end_tick=authorized_end_tick,
    )
    for index, repair in enumerate(operation_objects):
        if repair.get("classification") == "mechanical":
            continue
        source_index = repair.get("source_index")
        if type(source_index) is not int or source_index not in in_scope_indices:
            _fail(f"repairs.operations[{index}] targets an event outside the authorized window")
    timeline_value, repair_ids = _apply_repair_operations(raw_timeline, repairs)
    events, intervals = _canonical_events(
        timeline_value,
        reference_duration_sec=reference_duration_sec,
        source_indices=in_scope_indices,
        transcript_offset_sec=transcript_offset_sec,
        authorized_start_s=Decimal(authorized_start_tick) / Decimal(OUTPUT_RATE_HZ),
        authorized_end_s=Decimal(authorized_end_tick) / Decimal(OUTPUT_RATE_HZ),
    )
    if not events:
        _fail("accepted transcript contains no events")
    semantic_value = {
        "schema": _REVIEW_SEMANTIC_SCHEMA,
        "output_role": OUTPUT_ROLE,
        "counted_gold_ready": False,
        "scope": scope_summary,
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
    }
    event_summary = {
        "n_events": len(events),
        "speaker_event_counts": {
            speaker: sum(event.speaker == speaker for event in events) for speaker in SPEAKERS
        },
        "minimum_start_frame": min(event.start_frame for event in events),
        "maximum_end_frame": max(event.end_frame for event in events),
    }
    accepted_snapshots = [{"speaker": event.speaker} for event in events]
    prevalence["accepted_post_review"] = _speaker_prevalence_from_snapshots(accepted_snapshots)
    scope_summary = dict(scope_summary)
    scope_summary["accepted_post_review_event_count"] = len(events)
    if len(events) != scope_summary["in_scope_pre_review_event_count"]:
        _fail("accepted event count must preserve every in-scope reviewed event")
    bindings = {
        "timeline_sha256": hashlib.sha256(raw_timeline).hexdigest(),
        "canonical_semantic_sha256": _canonical_json_sha256(semantic_value),
        "event_summary": event_summary,
        "repair_set_sha256": _canonical_json_sha256(
            {"schema": _REPAIR_SET_SCHEMA, "repairs": repairs}
        ),
        "repair_ids": list(repair_ids),
        "scope_summary": scope_summary,
        "speaker_prevalence": prevalence,
        "excluded_event_inventory": list(excluded_inventory),
    }
    return events, intervals, bindings


def derive_manual_review_bindings(
    raw_timeline: bytes,
    repairs: Any,
    *,
    reference_duration_sec: Decimal,
    transcript_offset_sec: Decimal,
    authorized_start_tick: int,
    authorized_end_tick: int,
) -> dict[str, Any]:
    """Derive values a reviewer must independently confirm in a manifest.

    This helper performs no acceptance and writes no artifact.  It exists so
    authoring tooling cannot invent a different semantic hash or event summary
    than the fail-closed loader will replay.
    """

    if not isinstance(raw_timeline, bytes):
        _fail("raw_timeline must be exact bytes")
    if not isinstance(reference_duration_sec, Decimal):
        _fail("reference_duration_sec must be Decimal")
    if not isinstance(transcript_offset_sec, Decimal):
        _fail("transcript_offset_sec must be Decimal")
    for value, label in (
        (authorized_start_tick, "authorized_start_tick"),
        (authorized_end_tick, "authorized_end_tick"),
    ):
        _require_exact_int(value, label, minimum=0)
    if authorized_end_tick <= authorized_start_tick:
        _fail("authorized tick range must be a positive half-open interval")
    _events, _intervals, bindings = _canonical_review_material(
        raw_timeline,
        repairs,
        reference_duration_sec=reference_duration_sec,
        transcript_offset_sec=transcript_offset_sec,
        authorized_start_tick=authorized_start_tick,
        authorized_end_tick=authorized_end_tick,
    )
    return bindings


def _load_transcript_acceptance_with_authority(
    path: str | Path,
    *,
    authority: TranscriptSourceAuthority,
) -> AcceptedTranscriptReference:
    """Implementation hook for isolated tests with an explicit trusted root."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        _fail(f"acceptance manifest does not exist: {manifest_path}")
    manifest = _require_object(
        _strict_json_loads(manifest_path.read_bytes(), "acceptance manifest"),
        "acceptance manifest",
    )
    _require_exact_keys(
        manifest,
        {
            "artifact_kind",
            "format_version",
            "scene",
            "output_role",
            "counted_gold_ready",
            "review_status",
            "unresolved_issues",
            "sources",
            "source_normalizations",
            "timecode_contract",
            "repairs",
            "manual_review",
            "timing",
            "scope",
            "speaker_prevalence",
            "excluded_event_inventory",
        },
        "acceptance manifest",
    )
    manifest_version = _require_exact_int(
        manifest["format_version"], "acceptance manifest.format_version"
    )
    if manifest["artifact_kind"] != ARTIFACT_KIND or manifest_version != FORMAT_VERSION:
        _fail("acceptance manifest artifact kind/version is unsupported")
    scene = _require_nonempty_string(manifest["scene"], "acceptance manifest.scene")
    _scene_parts(scene)
    if manifest["output_role"] != OUTPUT_ROLE:
        _fail("acceptance manifest output_role must remain a coarse transcript reference")
    if manifest["counted_gold_ready"] is not False:
        _fail("Transcript acceptance can never declare counted gold-VAD readiness")
    if manifest["review_status"] != "confirmed":
        _fail("acceptance manifest review_status must be confirmed")
    if not isinstance(manifest["unresolved_issues"], list) or manifest["unresolved_issues"]:
        _fail("acceptance manifest unresolved_issues must be an empty list")
    _validate_timecode_contract(manifest["timecode_contract"])

    sources = _require_object(manifest["sources"], "sources")
    source_names = (
        "timeline",
        "offset",
        "reference_wav",
        "hpe_offset",
        "hpe_offset_quality",
        "gold_alignment",
    )
    _require_exact_keys(sources, set(source_names), "sources")
    manifest_dir = manifest_path.parent
    if not isinstance(authority, TranscriptSourceAuthority):
        _fail("authority must be a TranscriptSourceAuthority")
    canonical_paths = _canonical_source_paths(scene, authority)
    verified: dict[str, tuple[Path, bytes | None]] = {}
    for name in source_names:
        verified[name] = _verify_binding(
            sources[name],
            manifest_dir=manifest_dir,
            label=f"sources.{name}",
            return_bytes=name not in {"reference_wav", "gold_alignment"},
            expected_path=canonical_paths[name],
        )

    gold_report = _load_gold_alignment_with_authority(
        verified["gold_alignment"][0], scene=scene, authority=authority
    )
    (
        authorized_start_tick,
        authorized_end_tick,
        _authorized_start_s,
        _authorized_end_s,
    ) = _gold_window(gold_report)

    timeline_binding = _require_object(sources["timeline"], "sources.timeline")
    raw_offset = verified["offset"][1] or b""
    raw_offset_value = _strict_json_loads(raw_offset, "transcript offset")
    offset_value, source_normalizations = _normalize_transcript_offset_source(
        raw_offset,
        raw_offset_value,
        manifest["source_normalizations"],
    )
    transcript_offset, transcript_processing_sr = _validate_transcript_offset(
        offset_value,
        scene=scene,
        reference_wav=verified["reference_wav"][0],
    )
    offset_object = _require_object(offset_value, "transcript offset")
    hpe_offset = _strict_json_loads(verified["hpe_offset"][1] or b"", "HPE offset")
    hpe_quality = _strict_json_loads(verified["hpe_offset_quality"][1] or b"", "HPE offset quality")
    (
        hpe_stream,
        hpe_offset_sec,
        timecode_uncertainty,
        offset_lag_uncertainty,
        transcript_side_bound,
        hpe_bound,
        combined_bound,
        hpe_status,
    ) = _validate_hpe_timing(
        hpe_offset,
        hpe_quality,
        manifest["timing"],
        transcript_offset_sec=transcript_offset,
        transcript_processing_sr=transcript_processing_sr,
    )

    events, intervals, review_bindings = _canonical_review_material(
        verified["timeline"][1] or b"",
        manifest["repairs"],
        reference_duration_sec=_require_real_number(
            offset_object["reference_duration_sec"], "transcript offset.reference_duration_sec"
        ),
        transcript_offset_sec=transcript_offset,
        authorized_start_tick=authorized_start_tick,
        authorized_end_tick=authorized_end_tick,
    )
    for key in ("scope_summary", "speaker_prevalence", "excluded_event_inventory"):
        manifest_key = "scope" if key == "scope_summary" else key
        if not _type_exact_json_equal(manifest[manifest_key], review_bindings[key]):
            _fail(f"acceptance manifest {manifest_key} does not replay from the gold window")
    _validate_manual_review(
        manifest["manual_review"],
        timeline_sha256=timeline_binding["sha256"],
        canonical_semantic_sha256=review_bindings["canonical_semantic_sha256"],
        event_summary=review_bindings["event_summary"],
        repair_set_sha256=review_bindings["repair_set_sha256"],
        repair_ids=review_bindings["repair_ids"],
    )

    # Close the validation/use window: every bound source, including the WAV
    # that is not parsed above, must still have the same canonical path, size,
    # and bytes after all semantic checks have completed.
    for name in source_names:
        _verify_binding(
            sources[name],
            manifest_dir=manifest_dir,
            label=f"sources.{name} final re-verification",
            return_bytes=False,
            expected_path=canonical_paths[name],
        )

    return AcceptedTranscriptReference(
        scene=scene,
        manifest_path=manifest_path,
        output_role=OUTPUT_ROLE,
        counted_gold_ready=False,
        transcript_offset_sec=transcript_offset,
        timecode_quantization_error_bound_ms=timecode_uncertainty,
        offset_lag_quantization_error_bound_ms=offset_lag_uncertainty,
        transcript_side_error_bound_ms=transcript_side_bound,
        hpe_stream=hpe_stream,
        hpe_offset_sec=hpe_offset_sec,
        hpe_error_bound_ms=hpe_bound,
        combined_error_bound_ms=combined_bound,
        hpe_quality_status=hpe_status,
        events=events,
        speaker_intervals=intervals,
        source_bindings={
            name: dict(_require_object(sources[name], f"sources.{name}")) for name in source_names
        },
        source_normalizations=source_normalizations,
        authorized_start_tick=authorized_start_tick,
        authorized_end_tick=authorized_end_tick,
        scope_summary=review_bindings["scope_summary"],
        speaker_prevalence=review_bindings["speaker_prevalence"],
        excluded_event_inventory=tuple(review_bindings["excluded_event_inventory"]),
    )


def load_transcript_acceptance(path: str | Path) -> AcceptedTranscriptReference:
    """Load one manifest against the workspace's fixed production authority.

    The public loader deliberately has no source-root override.  A manifest
    cannot make copied bytes authoritative by naming a different directory.
    """

    return _load_transcript_acceptance_with_authority(
        path,
        authority=default_source_authority(),
    )


def _shared_ticks_25hz(values: Sequence[Decimal | str | int | float]) -> tuple[int, ...]:
    if len(values) == 0:
        _fail("shared timestamps must be a non-empty 25-Hz sequence")
    ticks: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            _fail(f"shared timestamp {index} must not be bool")
        try:
            timestamp = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise TranscriptAcceptanceError(
                f"shared timestamp {index} is not an exact decimal"
            ) from exc
        if not timestamp.is_finite():
            _fail(f"shared timestamp {index} must be finite")
        tick_value = timestamp * OUTPUT_RATE_HZ
        tick = tick_value.to_integral_value()
        if tick_value != tick:
            _fail(f"shared timestamp {index} is not on the exact 25-Hz lattice")
        ticks.append(int(tick))
    for index in range(1, len(ticks)):
        if ticks[index] != ticks[index - 1] + 1:
            _fail("shared timestamps must be strictly consecutive 25-Hz ticks")
    return tuple(ticks)


def rasterize_transcript_activity_reference(
    accepted: AcceptedTranscriptReference,
    shared_timestamps: Sequence[Decimal | str | int | float],
) -> TranscriptActivityReference:
    """Rasterise a coarse Transcript reference onto shared 25-Hz ticks.

    A tick is active exactly when ``start_shared <= tick/25 < end_shared``.
    Same-speaker overlaps have already been unioned; overlaps between different
    speakers remain independent ones in their respective arrays.  The returned
    arrays are not counted gold VAD and must not satisfy model target-readiness
    gates without a separate lavalier-derived artifact.
    """

    ticks = _shared_ticks_25hz(shared_timestamps)
    if ticks[0] < accepted.authorized_start_tick or ticks[-1] >= accepted.authorized_end_tick:
        _fail("rasterization timestamps exceed the formal half-open gold-alignment window")
    result = {speaker: np.zeros(len(ticks), dtype=np.int8) for speaker in SPEAKERS}
    for speaker in SPEAKERS:
        for start_frame, end_frame in accepted.speaker_intervals[speaker]:
            start_shared = (
                Decimal(start_frame) / TIMECODE_FPS_NUMERATOR + accepted.transcript_offset_sec
            )
            end_shared = (
                Decimal(end_frame) / TIMECODE_FPS_NUMERATOR + accepted.transcript_offset_sec
            )
            for index, tick in enumerate(ticks):
                timestamp = Decimal(tick) / OUTPUT_RATE_HZ
                if start_shared <= timestamp < end_shared:
                    result[speaker][index] = 1
    return TranscriptActivityReference(
        scene=accepted.scene,
        output_role=OUTPUT_ROLE,
        counted_gold_ready=False,
        shared_timestamps=tuple(Decimal(tick) / OUTPUT_RATE_HZ for tick in ticks),
        values=result,
    )
