"""Candidate-only transcript-anchored gold-VAD evidence and audit planning.

This module deliberately does *not* create training labels.  It replays the
formal analysis base, confirmed Transcript bridge and corrected lavalier time
map, then proposes acoustic boundary evidence for a subsequent blind human
audit.  Every public artifact is permanently ineligible for assembly/counting.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import tempfile
import wave
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.data import (
    analysis_base,
    assembly,
    audio_contract,
    audio_gold_alignment,
    transcript_acceptance,
    transcript_bridge,
    transcript_fs_guard,
)

ARTIFACT_KIND = "vap_gold_vad_candidate_evidence_bundle"
FORMAT_VERSION = 1
AUDIT_QUEUE_KIND = "vap_gold_vad_candidate_blind_audit_queue"
AUDIT_DECISIONS_KIND = "vap_gold_vad_candidate_blind_decisions"
AUDIT_REVEAL_KIND = "vap_gold_vad_candidate_audit_reveal"
AUDIT_FORMAT_VERSION = 1
TRUST_MODEL = "trusted_code_integrity_and_reproducibility_not_signature"

CANDIDATE_ONLY = True
ASSEMBLY_ELIGIBLE = False
COUNTED_GOLD_READY = False
SOURCE_SELECTION = "owner_approval_pending"

SPEAKERS = ("A", "B", "C", "R")
FRAME_RATE_HZ = 25
ANALYSIS_HOP_MS = 10
RMS_WINDOW_MS = 20
DIAGNOSTIC_RMS_WINDOW_MS = 40
MIN_SPEECH_MS = 60
FILL_GAP_MS = 40
SNAP_RADIUS_MS = 400
MISSED_MIN_MS = 80
DB_EPSILON = 1.0 / 32768.0

_PARAMETERS: dict[str, Any] = {
    "contract": "transcript_anchored_channel_relative_boundary_refinement_candidate_v1",
    "approval_status": SOURCE_SELECTION,
    "frame_rate_hz": FRAME_RATE_HZ,
    "rms_window_ms": RMS_WINDOW_MS,
    "diagnostic_rms_window_ms": DIAGNOSTIC_RMS_WINDOW_MS,
    "analysis_hop_ms": ANALYSIS_HOP_MS,
    "calibration_quantile": "0.10",
    "calibration_scope": "per_speaker_per_session_formal_window",
    "db_epsilon": format(DB_EPSILON, ".17g"),
    "relative_formula": "own_calibrated_db - median(other_calibrated_db)",
    "enter": {"relative_db_gte": "3", "own_above_floor_db_gte": "8"},
    "hold": {"relative_db_gte": "0", "own_above_floor_db_gte": "5"},
    "strong_core": {"relative_db_gte": "6", "own_above_floor_db_gte": "10"},
    "minimum_speech_ms": MIN_SPEECH_MS,
    "fill_gap_ms_lte": FILL_GAP_MS,
    "snap_radius_ms": SNAP_RADIUS_MS,
    "missed_token_probe": {
        "relative_db_gte": "6",
        "own_above_floor_db_gte": "10",
        "own_minus_max_other_db_gte": "-3",
        "minimum_duration_ms": MISSED_MIN_MS,
        "action": "flag_only_never_auto_promote",
    },
    "window_edge_policy": (
        "never_read_outside_formal_half_open_window; flag_insufficient_context"
    ),
    "outside_20ms_context_owner_decision": SOURCE_SELECTION,
    "automatic_label_promotion": False,
}

_AUDIT_QUOTAS = {
    "normal": 10,
    "low_margin_or_big_shift": 10,
    "overlap": 10,
    "short": 10,
    "candidate_only": 5,
    "transcript_active_low": 5,
}
_MIN_AUDIT_PER_SCENE = 50
_MIN_AUDIT_PER_PARTICIPANT = 8
_BLIND_MISSED_CLIP_DURATION_S = Decimal("4")
_BLIND_MISSED_CLIP_GRID_STRIDE_S = Decimal("1")
_CLOUD_RECALL_FLAGS = (
    getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_OPEN", 0x40000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS", 0x400000)
)


class GoldVadCandidateError(ValueError):
    """Raised when candidate evidence cannot be reproduced without ambiguity."""


@contextmanager
def _candidate_error_boundary(label: str) -> Iterator[None]:
    """Normalize every ordinary deep validation failure to the public error type."""

    try:
        yield
    except GoldVadCandidateError:
        raise
    except Exception as exc:
        raise GoldVadCandidateError(f"{label} failed: {exc}") from exc


def _normalize_candidate_errors(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with _candidate_error_boundary(label):
                return function(*args, **kwargs)

        return wrapped

    return decorate


@dataclass(frozen=True)
class GoldVadCandidateAuthority:
    """Code-owned roots; only tests may construct an alternate authority."""

    analysis_base_root: Path
    transcript_bridge_root: Path
    gold_alignment_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_base_root", Path(self.analysis_base_root).resolve())
        object.__setattr__(
            self, "transcript_bridge_root", Path(self.transcript_bridge_root).resolve()
        )
        object.__setattr__(
            self, "gold_alignment_root", Path(self.gold_alignment_root).resolve()
        )


@dataclass(frozen=True)
class _InputSnapshot:
    scene: str
    analysis_csv: Path
    analysis_sidecar: Path
    frame_df: Any
    analysis_provenance: dict[str, Any]
    bridge_path: Path
    bridge: transcript_bridge.TranscriptBridgeBundle
    gold_path: Path
    gold_report: dict[str, Any]
    audio_manifest_path: Path
    audio_report: dict[str, Any]
    start_tick: int
    end_tick: int
    corrections_s: Mapping[str, Decimal]
    source_bindings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _DetectorView:
    window_ms: int
    right_edges_s: np.ndarray
    calibrated_db: Mapping[str, np.ndarray]
    relative_db: Mapping[str, np.ndarray]
    own_minus_max_other_db: Mapping[str, np.ndarray]
    speech: Mapping[str, np.ndarray]
    strong: Mapping[str, np.ndarray]
    floors_db: Mapping[str, float]


def default_gold_vad_candidate_authority() -> GoldVadCandidateAuthority:
    root = Path(__file__).resolve().parents[3]
    return GoldVadCandidateAuthority(
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        transcript_bridge_root=root / "data" / "annotations" / "transcript_bridge",
        gold_alignment_root=root / "data" / "annotations" / "audio_gold_alignment",
    )


def _fail(message: str) -> None:
    raise GoldVadCandidateError(message)


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
        raise GoldVadCandidateError("candidate artifact is not strict finite JSON") from exc


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GoldVadCandidateError(f"{label} is not strict UTF-8") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def no_nonfinite(value: str) -> None:
        _fail(f"{label} contains forbidden non-finite constant {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=no_nonfinite)
    except json.JSONDecodeError as exc:
        raise GoldVadCandidateError(
            f"{label} is not valid JSON at line {exc.lineno}, column {exc.colno}"
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


def _finite(value: float, *, digits: int = 9) -> float:
    number = float(value)
    if not math.isfinite(number):
        _fail("candidate computation produced a non-finite number")
    return round(number, digits)


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{label} must not be bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise GoldVadCandidateError(f"{label} must be an exact decimal") from exc
    if not result.is_finite():
        _fail(f"{label} must be finite")
    return result


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        _fail("cannot serialize a non-finite decimal")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _canonical_paths(
    scene: str, authority: GoldVadCandidateAuthority
) -> tuple[Path, Path, Path, Path]:
    if type(scene) is not str or not scene or scene.strip() != scene:
        _fail("scene must be a non-empty canonical string")
    if any(token in scene for token in ("/", "\\", "..")):
        _fail("scene must not contain path syntax")
    csv_path = authority.analysis_base_root / f"{scene}.csv"
    sidecar = csv_path.with_suffix(".provenance.json")
    bridge_path = authority.transcript_bridge_root / scene
    gold_path = authority.gold_alignment_root / f"{scene}.audio_gold_alignment.json"
    return csv_path, sidecar, bridge_path, gold_path


def _reject_cloud_or_link(path: Path) -> None:
    absolute = transcript_fs_guard.guard_regular_file_path(path)
    value = absolute.lstat()
    attributes = getattr(value, "st_file_attributes", 0)
    if attributes & _CLOUD_RECALL_FLAGS:
        _fail(f"source is an Offline/recall-on-access placeholder: {absolute}")


def _binding_from_path(path: Path) -> dict[str, Any]:
    absolute = transcript_fs_guard.lexical_absolute(path)
    _reject_cloud_or_link(absolute)
    with transcript_fs_guard.guarded_read(absolute) as held:
        digest = transcript_fs_guard.hash_guarded_read(held)
        size = held.opening_state.size_bytes
    return {"path": str(absolute), "size_bytes": size, "sha256": digest}


def _iter_declared_bindings(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "size_bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from _iter_declared_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_declared_bindings(child)


def _validated_source_bindings(
    values: Sequence[Any], *, extras: Sequence[Path]
) -> tuple[dict[str, Any], ...]:
    declared: dict[Path, tuple[int, str]] = {}
    for value in values:
        for binding in _iter_declared_bindings(value):
            path_value = binding.get("path")
            size_value = binding.get("size_bytes")
            digest_value = binding.get("sha256")
            if (
                type(path_value) is not str
                or type(size_value) is not int
                or size_value < 0
                or type(digest_value) is not str
                or len(digest_value) != 64
            ):
                _fail("an upstream source binding has an invalid path/size/hash shape")
            path = transcript_fs_guard.lexical_absolute(path_value)
            previous = declared.get(path)
            current = (size_value, digest_value)
            if previous is not None and previous != current:
                _fail(f"upstream manifests disagree about source bytes: {path}")
            declared[path] = current
    for path in extras:
        absolute = transcript_fs_guard.lexical_absolute(path)
        if absolute not in declared:
            binding = _binding_from_path(absolute)
            declared[absolute] = (binding["size_bytes"], binding["sha256"])

    result: list[dict[str, Any]] = []
    for path in sorted(declared, key=lambda item: str(item).casefold()):
        actual = _binding_from_path(path)
        if (actual["size_bytes"], actual["sha256"]) != declared[path]:
            _fail(f"upstream source binding no longer matches exact bytes: {path}")
        result.append(actual)
    return tuple(result)


@_normalize_candidate_errors("formal candidate input replay")
def _load_inputs_with_authority(
    scene: str, *, authority: GoldVadCandidateAuthority
) -> _InputSnapshot:
    analysis_csv, sidecar, bridge_path, gold_path = _canonical_paths(scene, authority)
    for path in (analysis_csv, sidecar, gold_path):
        _reject_cloud_or_link(path)
    guarded_bridge, _identity = transcript_fs_guard.guard_directory(bridge_path)
    if guarded_bridge.resolve() != bridge_path.resolve():
        _fail("Transcript bridge must be at the code-owned canonical scene path")

    dataset = assembly._load_dataset_file(analysis_csv, data_kind="real")
    provenance = dataset.provenance
    if provenance is None or provenance.get("artifact_kind") != analysis_base.ARTIFACT_KIND:
        _fail("gold-VAD candidates require a formal vap_real_analysis_base parent")
    if provenance.get("session_id") != scene:
        _fail("analysis base session differs from requested scene")
    bridge = transcript_bridge.load_transcript_bridge_bundle(bridge_path)
    if bridge.scene != scene:
        _fail("Transcript bridge scene differs from requested scene")
    if (
        bridge.acceptance.output_role != transcript_acceptance.OUTPUT_ROLE
        or bridge.acceptance.counted_gold_ready is not False
        or bridge.receipt.get("review_status") != "confirmed"
    ):
        _fail("Transcript bridge must be confirmed coarse-reference evidence")
    gold_report = audio_gold_alignment.load_audio_gold_alignment(gold_path)
    if gold_report.get("scene") != scene:
        _fail("gold-audio alignment scene differs from requested scene")
    if gold_report.get("counted_gold_ready") is not False:
        _fail("alignment-only evidence must never claim counted gold readiness")

    tick_range = provenance.get("output_tick_range")
    if type(tick_range) is not dict:
        _fail("analysis base lacks output_tick_range")
    start_tick = tick_range.get("start_inclusive")
    end_tick = tick_range.get("end_exclusive")
    if type(start_tick) is not int or type(end_tick) is not int or end_tick <= start_tick:
        _fail("analysis base has an invalid exact half-open tick range")
    authorized = gold_report.get("authorization", {}).get("authorized_tick_range")
    if type(authorized) is not dict or (
        authorized.get("start_inclusive"), authorized.get("end_exclusive")
    ) != (start_tick, end_tick):
        _fail("analysis base and gold-audio alignment windows differ")
    if (
        bridge.acceptance.authorized_start_tick,
        bridge.acceptance.authorized_end_tick,
    ) != (start_tick, end_tick):
        _fail("Transcript bridge and analysis-base windows differ")
    ticks = np.rint(dataset.frame_df["timestamp_s"].to_numpy(dtype=float) * 25).astype(int)
    if not np.array_equal(ticks, np.arange(start_tick, end_tick, dtype=int)):
        _fail("analysis-base timestamps are not the exact authorised 25-Hz tick grid")

    mapping = gold_report.get("mapping_contract", {}).get("channels")
    if type(mapping) is not dict or set(mapping) != set(SPEAKERS):
        _fail("gold-audio alignment lacks the exact A/B/C/R mapping")
    corrections = {
        speaker: _decimal(mapping[speaker].get("correction_s"), f"{speaker} correction")
        for speaker in SPEAKERS
    }
    audio_binding = gold_report.get("sources", {}).get("audio_contract")
    if type(audio_binding) is not dict or type(audio_binding.get("path")) is not str:
        _fail("gold-audio alignment lacks its bound audio contract")
    audio_manifest = transcript_fs_guard.lexical_absolute(audio_binding["path"])
    audio_report = audio_contract.load_audio_contract(audio_manifest)
    if audio_report.get("scene") != scene:
        _fail("audio contract scene differs from requested scene")

    extras = [
        analysis_csv,
        sidecar,
        gold_path,
        audio_manifest,
        bridge_path / transcript_bridge.ACCEPTANCE_FILENAME,
        bridge_path / transcript_bridge.ACTIVITY_FILENAME,
        bridge_path / transcript_bridge.RECEIPT_FILENAME,
    ]
    source_bindings = _validated_source_bindings(
        [provenance, bridge.receipt, bridge.final_review, gold_report, audio_report],
        extras=extras,
    )
    return _InputSnapshot(
        scene=scene,
        analysis_csv=analysis_csv.resolve(),
        analysis_sidecar=sidecar.resolve(),
        frame_df=dataset.frame_df,
        analysis_provenance=provenance,
        bridge_path=bridge_path.resolve(),
        bridge=bridge,
        gold_path=gold_path.resolve(),
        gold_report=gold_report,
        audio_manifest_path=audio_manifest,
        audio_report=audio_report,
        start_tick=start_tick,
        end_tick=end_tick,
        corrections_s=corrections,
        source_bindings=source_bindings,
    )


def _source_fingerprint(snapshot: _InputSnapshot) -> str:
    material = {
        "scene": snapshot.scene,
        "start_tick": snapshot.start_tick,
        "end_tick": snapshot.end_tick,
        "source_bindings": list(snapshot.source_bindings),
        "corrections_s": {
            speaker: _canonical_decimal(snapshot.corrections_s[speaker])
            for speaker in SPEAKERS
        },
    }
    return _sha256_bytes(_canonical_json_bytes(material))


@contextmanager
def _hold_source_files(
    bindings: Sequence[Mapping[str, Any]],
) -> Iterator[dict[Path, transcript_fs_guard.GuardedReadHandle]]:
    with ExitStack() as stack:
        handles: dict[Path, transcript_fs_guard.GuardedReadHandle] = {}
        for binding in bindings:
            path = transcript_fs_guard.lexical_absolute(str(binding["path"]))
            _reject_cloud_or_link(path)
            held = stack.enter_context(transcript_fs_guard.guarded_read(path))
            if (
                held.opening_state.size_bytes != binding["size_bytes"]
                or transcript_fs_guard.hash_guarded_read(held) != binding["sha256"]
            ):
                _fail(f"source changed before guarded candidate computation: {path}")
            handles[path] = held
        yield handles
        for held in handles.values():
            transcript_fs_guard.verify_guarded_read(held)


def _ceil_decimal_samples(value_s: Decimal, sample_rate: int) -> int:
    return int((value_s * sample_rate).to_integral_value(rounding=ROUND_CEILING))


def _channel_sample_mapping(snapshot: _InputSnapshot) -> dict[str, dict[str, Any]]:
    shared_start = Decimal(snapshot.start_tick) / FRAME_RATE_HZ
    shared_end = Decimal(snapshot.end_tick) / FRAME_RATE_HZ
    result: dict[str, dict[str, Any]] = {}
    for speaker in SPEAKERS:
        local_start = shared_start - snapshot.corrections_s[speaker]
        local_end = shared_end - snapshot.corrections_s[speaker]
        exact_start_sample = local_start * 48_000
        exact_end_sample = local_end * 48_000
        start_sample = _ceil_decimal_samples(local_start, 48_000)
        end_sample = _ceil_decimal_samples(local_end, 48_000)
        result[speaker] = {
            "channel_local_start_s": _canonical_decimal(local_start),
            "channel_local_end_s_exclusive": _canonical_decimal(local_end),
            "start_sample_inclusive": start_sample,
            "end_sample_exclusive": end_sample,
            "n_samples": end_sample - start_sample,
            "start_ceil_residual_samples": _canonical_decimal(
                Decimal(start_sample) - exact_start_sample
            ),
            "end_ceil_residual_samples": _canonical_decimal(
                Decimal(end_sample) - exact_end_sample
            ),
            "sample_rate_hz": 48_000,
        }
    return result


@_normalize_candidate_errors("formal PCM decode")
def _decode_formal_channels(
    snapshot: _InputSnapshot,
    handles: Mapping[Path, transcript_fs_guard.GuardedReadHandle],
) -> dict[str, np.ndarray]:
    sources = snapshot.audio_report.get("sources")
    if type(sources) is not dict:
        _fail("audio contract has no sources object")
    sample_mapping = _channel_sample_mapping(snapshot)
    output: dict[str, np.ndarray] = {}
    expected_samples = (snapshot.end_tick - snapshot.start_tick) * 1920
    for speaker in SPEAKERS:
        binding = sources.get(speaker)
        if type(binding) is not dict or type(binding.get("path")) is not str:
            _fail(f"audio contract lacks {speaker} WAV binding")
        path = transcript_fs_guard.lexical_absolute(binding["path"])
        held = handles.get(path)
        if held is None:
            _fail(f"{speaker} WAV was not held for the complete computation")
        raw = transcript_fs_guard.read_guarded_bytes(held)
        if _sha256_bytes(raw) != binding.get("sha256") or len(raw) != binding.get(
            "size_bytes"
        ):
            _fail(f"{speaker} WAV bytes differ from the replayed audio contract")
        try:
            with wave.open(io.BytesIO(raw), "rb") as stream:
                if (
                    stream.getnchannels() != 1
                    or stream.getsampwidth() != 2
                    or stream.getframerate() != 48_000
                    or stream.getcomptype() != "NONE"
                ):
                    _fail(f"{speaker} WAV must be 48-kHz mono PCM16")
                start_sample = sample_mapping[speaker]["start_sample_inclusive"]
                end_sample = sample_mapping[speaker]["end_sample_exclusive"]
                if (
                    start_sample < 0
                    or end_sample > stream.getnframes()
                    or end_sample <= start_sample
                ):
                    _fail(f"{speaker} formal window exceeds real PCM coverage")
                if end_sample - start_sample != expected_samples:
                    _fail(f"{speaker} formal PCM slice is not the exact 25-Hz window")
                stream.setpos(start_sample)
                pcm = stream.readframes(end_sample - start_sample)
        except (EOFError, wave.Error) as exc:
            raise GoldVadCandidateError(f"{speaker} WAV decode failed") from exc
        if len(pcm) != expected_samples * 2:
            _fail(f"{speaker} WAV returned a short formal PCM slice")
        output[speaker] = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    return output


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _fill_short_false_runs(mask: np.ndarray, maximum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _mask_runs(~result):
        if start > 0 and end < len(result) and end - start <= maximum:
            result[start:end] = True
    return result


def _remove_short_true_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _mask_runs(result):
        if end - start < minimum:
            result[start:end] = False
    return result


def _rms_db(samples: np.ndarray, *, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 1 or not np.isfinite(samples).all():
        _fail("formal PCM must be one-dimensional and finite")
    sample_rate = 48_000
    hop = sample_rate * ANALYSIS_HOP_MS // 1000
    window = sample_rate * window_ms // 1000
    if len(samples) < window or len(samples) % hop != 0:
        _fail("formal PCM is too short or not on the exact 10-ms lattice")
    ends = np.arange(window, len(samples) + 1, hop, dtype=np.int64)
    squared = np.square(samples, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    means = (cumulative[ends] - cumulative[ends - window]) / window
    rms = np.sqrt(np.maximum(means, 0.0))
    db = 20.0 * np.log10(np.maximum(rms, DB_EPSILON))
    return ends, db


def _hysteresis(enter: np.ndarray, hold: np.ndarray) -> np.ndarray:
    if enter.shape != hold.shape:
        _fail("enter/hold masks differ in shape")
    output = np.zeros(len(enter), dtype=bool)
    active = False
    for index in range(len(enter)):
        if not active and bool(enter[index]):
            active = True
        elif active and not bool(hold[index]):
            active = False
        output[index] = active
    output = _fill_short_false_runs(output, FILL_GAP_MS // ANALYSIS_HOP_MS)
    return _remove_short_true_runs(output, MIN_SPEECH_MS // ANALYSIS_HOP_MS)


def _detector_view(
    channels: Mapping[str, np.ndarray], *, start_s: float, window_ms: int
) -> _DetectorView:
    db: dict[str, np.ndarray] = {}
    right_edges: np.ndarray | None = None
    for speaker in SPEAKERS:
        ends, values = _rms_db(channels[speaker], window_ms=window_ms)
        current_edges = start_s + ends.astype(np.float64) / 48_000.0
        if right_edges is None:
            right_edges = current_edges
        elif not np.array_equal(current_edges, right_edges):
            _fail("speaker detector grids differ")
        db[speaker] = values
    assert right_edges is not None
    floors = {speaker: float(np.quantile(db[speaker], 0.10)) for speaker in SPEAKERS}
    calibrated = {speaker: db[speaker] - floors[speaker] for speaker in SPEAKERS}
    relative: dict[str, np.ndarray] = {}
    own_minus_max: dict[str, np.ndarray] = {}
    speech: dict[str, np.ndarray] = {}
    strong: dict[str, np.ndarray] = {}
    for speaker in SPEAKERS:
        others = np.vstack([calibrated[item] for item in SPEAKERS if item != speaker])
        relative[speaker] = calibrated[speaker] - np.median(others, axis=0)
        own_minus_max[speaker] = calibrated[speaker] - np.max(others, axis=0)
        enter = (relative[speaker] >= 3.0) & (calibrated[speaker] >= 8.0)
        hold = (relative[speaker] >= 0.0) & (calibrated[speaker] >= 5.0)
        speech[speaker] = _hysteresis(enter, hold)
        strong[speaker] = (relative[speaker] >= 6.0) & (calibrated[speaker] >= 10.0)
    return _DetectorView(
        window_ms=window_ms,
        right_edges_s=right_edges,
        calibrated_db=calibrated,
        relative_db=relative,
        own_minus_max_other_db=own_minus_max,
        speech=speech,
        strong=strong,
        floors_db=floors,
    )


def _run_intervals(view: _DetectorView, speaker: str) -> list[tuple[float, float, int, int]]:
    window_s = view.window_ms / 1000.0
    return [
        (
            float(view.right_edges_s[start] - window_s),
            float(view.right_edges_s[end - 1]),
            start,
            end,
        )
        for start, end in _mask_runs(view.speech[speaker])
    ]


def _event_shared_interval(
    event: transcript_acceptance.TranscriptEvent,
    accepted: transcript_acceptance.AcceptedTranscriptReference,
    *,
    start_s: float,
    end_s: float,
) -> tuple[float, float]:
    raw_start = float(Decimal(event.start_frame) / 30 + accepted.transcript_offset_sec)
    raw_end = float(Decimal(event.end_frame) / 30 + accepted.transcript_offset_sec)
    clipped_start = max(start_s, raw_start)
    clipped_end = min(end_s, raw_end)
    if clipped_end <= clipped_start:
        _fail(f"accepted Transcript event {event.source_index} does not intersect formal window")
    return clipped_start, clipped_end


def _indices_for_interval(edges: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.flatnonzero((edges > start) & (edges <= end + 1e-12))


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _select_run(
    runs: Sequence[tuple[float, float, int, int]], start: float, end: float
) -> tuple[tuple[float, float, int, int] | None, int]:
    candidates = [
        run
        for run in runs
        if (
            _overlap((start, end), (run[0], run[1])) > 0
            or abs(run[0] - start) <= SNAP_RADIUS_MS / 1000
            or abs(run[1] - end) <= SNAP_RADIUS_MS / 1000
        )
    ]
    if not candidates:
        return None, 0
    ranked = sorted(
        candidates,
        key=lambda run: (
            -_overlap((start, end), (run[0], run[1])),
            abs(run[0] - start) + abs(run[1] - end),
            run[0],
            run[1],
        ),
    )
    return ranked[0], len(candidates)


def _transcript_overlap_flags(
    events: Sequence[transcript_acceptance.TranscriptEvent],
    accepted: transcript_acceptance.AcceptedTranscriptReference,
    *,
    start_s: float,
    end_s: float,
) -> dict[int, bool]:
    intervals = {
        event.source_index: _event_shared_interval(
            event, accepted, start_s=start_s, end_s=end_s
        )
        for event in events
    }
    result: dict[int, bool] = {}
    for event in events:
        result[event.source_index] = any(
            other.speaker != event.speaker
            and _overlap(intervals[event.source_index], intervals[other.source_index]) > 0
            for other in events
        )
    return result


def _joint_boundary_decision(
    original_start: float,
    original_end: float,
    selected: tuple[float, float, int, int],
    *,
    formal_start: float,
    formal_end: float,
) -> tuple[float, float, str, bool]:
    radius_s = SNAP_RADIUS_MS / 1000.0
    proposed_start = (
        selected[0] if abs(selected[0] - original_start) <= radius_s else original_start
    )
    proposed_end = selected[1] if abs(selected[1] - original_end) <= radius_s else original_end
    valid = formal_start <= proposed_start < proposed_end <= formal_end
    if not valid:
        return (
            original_start,
            original_end,
            "transcript_boundaries_retained_joint_order_conflict",
            True,
        )
    return proposed_start, proposed_end, "acoustic_snap_joint_endpoint_decision", False


def _candidate_event(
    event: transcript_acceptance.TranscriptEvent,
    *,
    accepted: transcript_acceptance.AcceptedTranscriptReference,
    primary: _DetectorView,
    diagnostic: _DetectorView,
    formal_start: float,
    formal_end: float,
    transcript_overlap: bool,
) -> dict[str, Any]:
    start, end = _event_shared_interval(
        event, accepted, start_s=formal_start, end_s=formal_end
    )
    selected, n_candidates = _select_run(
        _run_intervals(primary, event.speaker), start, end
    )
    diagnostic_selected, _ = _select_run(
        _run_intervals(diagnostic, event.speaker), start, end
    )
    indices = _indices_for_interval(primary.right_edges_s, start, end)
    if len(indices) == 0:
        _fail(f"Transcript event {event.source_index} has no in-window detector support")
    active_fraction = float(np.mean(primary.speech[event.speaker][indices]))
    strong_fraction = float(np.mean(primary.strong[event.speaker][indices]))
    relative_median = float(np.median(primary.relative_db[event.speaker][indices]))
    calibrated_median = float(np.median(primary.calibrated_db[event.speaker][indices]))
    speaker_scores = {
        speaker: float(np.median(primary.relative_db[speaker][indices]))
        for speaker in SPEAKERS
    }
    dominant = max(SPEAKERS, key=lambda speaker: (speaker_scores[speaker], speaker))

    flags: list[str] = []
    method = "acoustic_snap_joint_endpoint_decision"
    if selected is None:
        suggested_start, suggested_end = start, end
        method = "transcript_boundary_retained_no_acoustic_run"
        flags.extend(["conflict", "transcript_active_low"])
        onset_margin = -999.0
        offset_margin = -999.0
    else:
        suggested_start, suggested_end, method, order_conflict = _joint_boundary_decision(
            start,
            end,
            selected,
            formal_start=formal_start,
            formal_end=formal_end,
        )
        if order_conflict:
            flags.extend(["ambiguity", "order_conflict"])
        onset_index = selected[2]
        offset_index = max(selected[2], selected[3] - 1)
        onset_margin = min(
            float(primary.relative_db[event.speaker][onset_index] - 3.0),
            float(primary.calibrated_db[event.speaker][onset_index] - 8.0),
        )
        offset_margin = min(
            float(primary.relative_db[event.speaker][offset_index]),
            float(primary.calibrated_db[event.speaker][offset_index] - 5.0),
        )
        if n_candidates > 1 or min(onset_margin, offset_margin) < 1.0:
            flags.append("ambiguity")
        if min(onset_margin, offset_margin) < 1.0:
            flags.append("low_margin")
        if abs(suggested_start - start) > 0.2 or abs(suggested_end - end) > 0.2:
            flags.append("big_shift")
    if transcript_overlap:
        flags.append("overlap")
    if end - start < 0.5:
        flags.append("short")
    if active_fraction < 0.5 or strong_fraction < 0.2:
        flags.append("transcript_active_low")
    if dominant != event.speaker and speaker_scores[dominant] > relative_median + 3.0:
        flags.append("conflict")
    radius_s = SNAP_RADIUS_MS / 1000.0
    if (
        start - radius_s < formal_start
        or end + radius_s > formal_end
        or start < formal_start + RMS_WINDOW_MS / 1000.0
    ):
        flags.append("edge_context_insufficient")
    flags = sorted(set(flags))

    diagnostic_payload: dict[str, Any]
    if diagnostic_selected is None:
        diagnostic_payload = {"suggested_start_s": None, "suggested_end_s": None}
    else:
        diagnostic_start, diagnostic_end, diagnostic_method, diagnostic_conflict = (
            _joint_boundary_decision(
                start,
                end,
                diagnostic_selected,
                formal_start=formal_start,
                formal_end=formal_end,
            )
        )
        diagnostic_payload = {
            "suggested_start_s": _finite(diagnostic_start),
            "suggested_end_s": _finite(diagnostic_end),
            "method": diagnostic_method,
            "order_conflict": diagnostic_conflict,
        }
    if not formal_start <= suggested_start < suggested_end <= formal_end:
        _fail("joint endpoint decision emitted an invalid formal half-open interval")
    return {
        "candidate_id": f"event_{event.source_index:06d}",
        "candidate_type": "transcript_anchored_boundary",
        "source_index": event.source_index,
        "speaker": event.speaker,
        "text": event.text,
        "transcript_interval": {
            "start_s": _finite(start),
            "end_s_exclusive": _finite(end),
        },
        "suggestion": {
            "method": method,
            "start_s": _finite(suggested_start),
            "end_s_exclusive": _finite(suggested_end),
            "onset_shift_ms": _finite((suggested_start - start) * 1000.0, digits=6),
            "offset_shift_ms": _finite((suggested_end - end) * 1000.0, digits=6),
            "candidate_run_count": n_candidates,
        },
        "diagnostic_40ms": diagnostic_payload,
        "evidence": {
            "transcript_active_fraction": _finite(active_fraction),
            "strong_core_fraction": _finite(strong_fraction),
            "own_relative_median_db": _finite(relative_median),
            "own_above_floor_median_db": _finite(calibrated_median),
            "onset_margin_db": _finite(onset_margin),
            "offset_margin_db": _finite(offset_margin),
            "dominant_relative_speaker": dominant,
            "dominant_relative_margin_db": _finite(
                speaker_scores[dominant] - relative_median
            ),
        },
        "flags": flags,
        "automatic_promotion": False,
    }


def _missed_candidates(
    primary: _DetectorView,
    *,
    accepted: transcript_acceptance.AcceptedTranscriptReference,
    formal_start: float,
    formal_end: float,
) -> list[dict[str, Any]]:
    intervals_by_speaker: dict[str, list[tuple[float, float]]] = {
        speaker: [] for speaker in SPEAKERS
    }
    for event in accepted.events:
        intervals_by_speaker[event.speaker].append(
            _event_shared_interval(
                event, accepted, start_s=formal_start, end_s=formal_end
            )
        )
    output: list[dict[str, Any]] = []
    for speaker in SPEAKERS:
        raw_strict = (
            primary.strong[speaker]
            & (primary.own_minus_max_other_db[speaker] >= -3.0)
        )
        transcript_mask = _transcript_mask(
            accepted,
            speaker,
            primary.right_edges_s,
            formal_start=formal_start,
            formal_end=formal_end,
        )
        strict_after_subtraction = raw_strict & ~transcript_mask
        strict = _remove_short_true_runs(
            strict_after_subtraction, MISSED_MIN_MS // ANALYSIS_HOP_MS
        )
        n_removed_by_transcript = int(np.count_nonzero(raw_strict & transcript_mask))
        serial = 0
        for start_index, end_index in _mask_runs(strict):
            hop_s = ANALYSIS_HOP_MS / 1000.0
            start = max(formal_start, float(primary.right_edges_s[start_index] - hop_s))
            end = min(formal_end, float(primary.right_edges_s[end_index - 1]))
            indices = np.arange(start_index, end_index)
            cut_left = bool(
                start_index > 0
                and raw_strict[start_index - 1]
                and transcript_mask[start_index - 1]
            )
            cut_right = bool(
                end_index < len(raw_strict)
                and raw_strict[end_index]
                and transcript_mask[end_index]
            )
            adjacent_left = bool(start_index > 0 and transcript_mask[start_index - 1])
            adjacent_right = bool(
                end_index < len(transcript_mask) and transcript_mask[end_index]
            )
            overlap_speakers = sorted(
                other
                for other in SPEAKERS
                if other != speaker
                and any(
                    _overlap((start, end), interval) > 0
                    for interval in intervals_by_speaker[other]
                )
            )
            flags = ["candidate_only"]
            if overlap_speakers:
                flags.append("overlap")
            if cut_left or cut_right:
                flags.append("transcript_subtraction_cut")
            elif adjacent_left or adjacent_right:
                flags.append("transcript_boundary_adjacent")
            serial += 1
            output.append(
                {
                    "candidate_id": f"missed_{speaker}_{serial:05d}",
                    "candidate_type": "missed_token_strict_probe",
                    "source_index": None,
                    "speaker": speaker,
                    "text": None,
                    "transcript_interval": None,
                    "suggestion": {
                        "method": "flag_only_never_auto_promote",
                        "start_s": _finite(start),
                        "end_s_exclusive": _finite(end),
                        "duration_ms": _finite((end - start) * 1000.0, digits=6),
                    },
                    "diagnostic_40ms": None,
                    "evidence": {
                        "minimum_relative_db": _finite(
                            np.min(primary.relative_db[speaker][indices])
                        ),
                        "minimum_own_above_floor_db": _finite(
                            np.min(primary.calibrated_db[speaker][indices])
                        ),
                        "minimum_own_minus_max_other_db": _finite(
                            np.min(primary.own_minus_max_other_db[speaker][indices])
                        ),
                        "overlaps_other_transcript_speakers": overlap_speakers,
                        "transcript_subtraction": {
                            "n_strict_frames_removed_for_speaker": n_removed_by_transcript,
                            "cut_left": cut_left,
                            "cut_right": cut_right,
                            "adjacent_left": adjacent_left,
                            "adjacent_right": adjacent_right,
                            "mask_semantics": "same_speaker_half_open_10ms_grid",
                        },
                    },
                    "flags": flags,
                    "automatic_promotion": False,
                }
            )
    return output


def _retain_original_boundaries(event: dict[str, Any], reasons: str | Sequence[str]) -> None:
    original = event["transcript_interval"]
    if original is None:
        _fail("only Transcript-anchored events can retain original boundaries")
    reason_list = [reasons] if isinstance(reasons, str) else sorted(set(reasons))
    previous = dict(event["suggestion"])
    event["evidence"].setdefault("discarded_cross_event_suggestions", []).append(
        {"reasons": reason_list, "suggestion": previous}
    )
    start = float(original["start_s"])
    end = float(original["end_s_exclusive"])
    event["suggestion"].update(
        {
            "method": "transcript_boundaries_retained_cross_event_conflict",
            "start_s": _finite(start),
            "end_s_exclusive": _finite(end),
            "onset_shift_ms": 0.0,
            "offset_shift_ms": 0.0,
        }
    )
    event["flags"] = sorted(
        set([*event["flags"], "ambiguity", "order_conflict", *reason_list])
    )


def _annotate_cross_event_conflicts(
    events: list[dict[str, Any]], *, formal_start: float, formal_end: float
) -> None:
    ordered = sorted(
        events,
        key=lambda item: (
            item["transcript_interval"]["start_s"],
            item["transcript_interval"]["end_s_exclusive"],
            item["source_index"],
        ),
    )
    conflict_reasons: dict[str, set[str]] = {
        event["candidate_id"]: set() for event in events
    }

    def mark(
        previous: dict[str, Any], current: dict[str, Any], reason: str
    ) -> None:
        conflict_reasons[previous["candidate_id"]].add(reason)
        conflict_reasons[current["candidate_id"]].add(reason)

    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_suggestion = previous["suggestion"]
        current_suggestion = current["suggestion"]
        if (
            previous_suggestion["start_s"] == current_suggestion["start_s"]
            and previous_suggestion["end_s_exclusive"]
            == current_suggestion["end_s_exclusive"]
        ):
            mark(previous, current, "reuse_conflict")
        if current_suggestion["start_s"] < previous_suggestion["start_s"]:
            mark(previous, current, "adjacent_refined_order_conflict")

    for speaker in SPEAKERS:
        speaker_events = [item for item in ordered if item["speaker"] == speaker]
        for previous, current in zip(speaker_events, speaker_events[1:], strict=False):
            previous_original = previous["transcript_interval"]
            current_original = current["transcript_interval"]
            previous_suggestion = previous["suggestion"]
            current_suggestion = current["suggestion"]
            if previous_original["end_s_exclusive"] > current_original["start_s"]:
                mark(previous, current, "same_speaker_coarse_overlap")
            elif (
                previous_suggestion["end_s_exclusive"]
                > current_suggestion["start_s"]
            ):
                mark(previous, current, "refined_same_speaker_overlap")
            if (
                current_suggestion["start_s"] < previous_suggestion["start_s"]
                or current_suggestion["end_s_exclusive"]
                < previous_suggestion["end_s_exclusive"]
            ):
                mark(previous, current, "refined_order_conflict")

    for event in events:
        reasons = conflict_reasons[event["candidate_id"]]
        if reasons:
            _retain_original_boundaries(event, sorted(reasons))

    for event in events:
        suggestion = event["suggestion"]
        if not (
            formal_start
            <= suggestion["start_s"]
            < suggestion["end_s_exclusive"]
            <= formal_end
        ):
            _retain_original_boundaries(event, "order_conflict")
        interval = (
            float(event["suggestion"]["start_s"]),
            float(event["suggestion"]["end_s_exclusive"]),
        )
        speakers = sorted(
            {
                other["speaker"]
                for other in events
                if other is not event
                and other["speaker"] != event["speaker"]
                and _overlap(
                    interval,
                    (
                        float(other["suggestion"]["start_s"]),
                        float(other["suggestion"]["end_s_exclusive"]),
                    ),
                )
                > 0
            }
        )
        event["evidence"]["refined_overlap_speakers"] = speakers
        if speakers:
            event["flags"] = sorted(set([*event["flags"], "overlap"]))


def _binary_lag_diagnostic(reference: np.ndarray, evidence: np.ndarray) -> dict[str, Any]:
    if reference.shape != evidence.shape or reference.ndim != 1:
        _fail("lag diagnostic arrays differ")
    if len(reference) < 10 or len(np.unique(reference)) < 2 or len(np.unique(evidence)) < 2:
        return {
            "lag_ms": None,
            "peak_correlation": None,
            "reason": "degenerate_support",
            "convention": "positive_means_acoustic_evidence_later_than_transcript",
        }
    candidates: list[tuple[float, int, int]] = []
    for lag in range(-40, 41):
        if lag < 0:
            left, right = reference[-lag:], evidence[:lag]
        elif lag > 0:
            left, right = reference[:-lag], evidence[lag:]
        else:
            left, right = reference, evidence
        if len(left) < 10 or np.std(left) == 0 or np.std(right) == 0:
            continue
        candidates.append((float(np.corrcoef(left, right)[0, 1]), lag, len(left)))
    if not candidates:
        return {
            "lag_ms": None,
            "peak_correlation": None,
            "reason": "degenerate_support",
            "convention": "positive_means_acoustic_evidence_later_than_transcript",
        }
    best = max(candidates, key=lambda item: (item[0], -abs(item[1]), -item[1]))
    return {
        "lag_ms": best[1] * ANALYSIS_HOP_MS,
        "peak_correlation": _finite(best[0]),
        "n_support_frames": best[2],
        "reason": None,
        "convention": "positive_means_acoustic_evidence_later_than_transcript",
    }


def _transcript_mask(
    accepted: transcript_acceptance.AcceptedTranscriptReference,
    speaker: str,
    edges: np.ndarray,
    *,
    formal_start: float,
    formal_end: float,
) -> np.ndarray:
    result = np.zeros(len(edges), dtype=bool)
    cell_starts = edges - ANALYSIS_HOP_MS / 1000.0
    for event in accepted.events:
        if event.speaker != speaker:
            continue
        start, end = _event_shared_interval(
            event, accepted, start_s=formal_start, end_s=formal_end
        )
        # A right-aligned 10-ms decision cell is [edge-hop, edge).  Subtract
        # every cell with positive overlap with the same-speaker half-open event.
        result |= (cell_starts < end - 1e-12) & (edges > start + 1e-12)
    return result


def _audit_stratum(candidate: Mapping[str, Any]) -> str:
    flags = set(candidate["flags"])
    if candidate["candidate_type"] == "missed_token_strict_probe":
        return "candidate_only"
    if flags & {"low_margin", "big_shift", "ambiguity"}:
        return "low_margin_or_big_shift"
    if "overlap" in flags:
        return "overlap"
    if "short" in flags:
        return "short"
    if "transcript_active_low" in flags:
        return "transcript_active_low"
    return "normal"


def _rank(seed: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}|{candidate_id}".encode()).hexdigest()


def _hidden_suggestion(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "suggestion": candidate["suggestion"],
        "diagnostic_40ms": candidate["diagnostic_40ms"],
        "evidence": candidate["evidence"],
    }


def _neutral_missed_clip(
    candidate: Mapping[str, Any],
    *,
    seed: str,
    formal_start: float,
    formal_end: float,
) -> dict[str, float]:
    """Choose a coarse scene-grid clip without boundary-derived padding."""

    start_bound = _decimal(formal_start, "formal start")
    end_bound = _decimal(formal_end, "formal end")
    suggestion = candidate["suggestion"]
    candidate_start = _decimal(suggestion["start_s"], "missed candidate start")
    candidate_end = _decimal(
        suggestion["end_s_exclusive"], "missed candidate end"
    )
    if not start_bound <= candidate_start < candidate_end <= end_bound:
        _fail("missed candidate is outside the formal half-open window")

    formal_duration = end_bound - start_bound
    if formal_duration <= _BLIND_MISSED_CLIP_DURATION_S:
        selected_start = start_bound
        selected_end = end_bound
    else:
        last_start = end_bound - _BLIND_MISSED_CLIP_DURATION_S
        n_regular = int(
            (last_start - start_bound) // _BLIND_MISSED_CLIP_GRID_STRIDE_S
        )
        starts = [
            start_bound + index * _BLIND_MISSED_CLIP_GRID_STRIDE_S
            for index in range(n_regular + 1)
        ]
        # A final scene-end anchor is part of the predeclared grid.  It avoids
        # candidate-dependent clamping when the formal duration is not an
        # integer multiple of the coarse stride.
        if starts[-1] != last_start:
            starts.append(last_start)

        containing = [
            start
            for start in starts
            if start <= candidate_start
            and candidate_end <= start + _BLIND_MISSED_CLIP_DURATION_S
        ]
        if containing:
            eligible = containing
        else:
            overlaps = {
                start: max(
                    Decimal("0"),
                    min(start + _BLIND_MISSED_CLIP_DURATION_S, candidate_end)
                    - max(start, candidate_start),
                )
                for start in starts
            }
            maximum_overlap = max(overlaps.values())
            if maximum_overlap <= 0:
                _fail("scene-grid clips do not overlap the missed candidate")
            eligible = [start for start in starts if overlaps[start] == maximum_overlap]

        def clip_rank(start: Decimal) -> str:
            material = (
                f"{seed}|{candidate['candidate_id']}|neutral_missed_clip|"
                f"{_canonical_decimal(start)}"
            )
            return hashlib.sha256(material.encode()).hexdigest()

        selected_start = min(
            eligible, key=lambda start: (clip_rank(start), _canonical_decimal(start))
        )
        selected_end = selected_start + _BLIND_MISSED_CLIP_DURATION_S

    if not start_bound <= selected_start < selected_end <= end_bound:
        _fail("neutral missed clip escaped the formal half-open window")
    return {
        "start_s": _finite(float(selected_start)),
        "end_s_exclusive": _finite(float(selected_end)),
    }


def _blind_row(
    candidate: Mapping[str, Any],
    *,
    audit_id: str,
    seed: str,
    formal_start: float,
    formal_end: float,
) -> dict[str, Any]:
    if candidate["transcript_interval"] is None:
        coarse_event = None
        clip_interval = _neutral_missed_clip(
            candidate,
            seed=seed,
            formal_start=formal_start,
            formal_end=formal_end,
        )
    else:
        transcript_interval = candidate["transcript_interval"]
        centre_start = float(transcript_interval["start_s"])
        centre_end = float(transcript_interval["end_s_exclusive"])
        coarse_event = {
            "source_index": candidate["source_index"],
            "text": candidate["text"],
            "interval": transcript_interval,
        }
        clip_interval = {
            "start_s": _finite(max(formal_start, centre_start - 1.0)),
            "end_s_exclusive": _finite(min(formal_end, centre_end + 1.0)),
        }
    row = {
        "audit_id": audit_id,
        "speaker_channel": candidate["speaker"],
        "coarse_event": coarse_event,
        "clip_interval": clip_interval,
        "suggestion_hidden_until_manual_decision": True,
        "hidden_suggestion_sha256": _sha256_bytes(
            _canonical_json_bytes(_hidden_suggestion(candidate))
        ),
    }
    forbidden = {
        "suggestion",
        "suggested_start_s",
        "suggested_end_s",
        "evidence",
        "audit_reasons",
        "stratum",
        "candidate_type",
        "low_margin",
        "big_shift",
        "ambiguity",
    }
    if forbidden & set(row):
        _fail("blind audit row leaked automated diagnostic or selection information")
    return row


def _build_audit_plan(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    formal_start: float,
    formal_end: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(candidates) < _MIN_AUDIT_PER_SCENE:
        _fail("formal scene has fewer than 50 auditable candidates/events")
    ordered = sorted(
        candidates,
        key=lambda item: (_rank(seed, item["candidate_id"]), item["candidate_id"]),
    )
    by_stratum = {
        stratum: [item for item in ordered if _audit_stratum(item) == stratum]
        for stratum in _AUDIT_QUOTAS
    }
    selected: dict[str, Mapping[str, Any]] = {}
    selection_reasons: dict[str, list[str]] = {}

    def select(item: Mapping[str, Any], reason: str) -> None:
        candidate_id = item["candidate_id"]
        selected[candidate_id] = item
        reasons = selection_reasons.setdefault(candidate_id, [])
        if reason not in reasons:
            reasons.append(reason)

    quota_counts: dict[str, int] = {}
    for stratum, quota in _AUDIT_QUOTAS.items():
        chosen = by_stratum[stratum][:quota]
        quota_counts[stratum] = len(chosen)
        for item in chosen:
            select(item, f"stratum_quota:{stratum}")

    # Every quality flag is an additional mandatory review, never merely a sampler hint.
    for item in ordered:
        if item["flags"]:
            select(item, "mandatory_flagged_evidence")

    # Deterministically redistribute every short quota to still-unselected evidence.
    for item in ordered:
        if len(selected) >= _MIN_AUDIT_PER_SCENE:
            break
        select(item, "minimum_scene_shortfall_reallocation")

    participant_additions: dict[str, int] = {speaker: 0 for speaker in SPEAKERS}
    for speaker in SPEAKERS:
        available = [item for item in ordered if item["speaker"] == speaker]
        target = min(_MIN_AUDIT_PER_PARTICIPANT, len(available))
        present = sum(item["speaker"] == speaker for item in selected.values())
        for item in available:
            if present >= target:
                break
            if item["candidate_id"] not in selected:
                select(item, f"participant_minimum:{speaker}")
                present += 1
                participant_additions[speaker] += 1

    selected_ordered = sorted(
        selected.values(),
        key=lambda item: (_rank(seed, item["candidate_id"]), item["candidate_id"]),
    )
    if len(selected_ordered) < _MIN_AUDIT_PER_SCENE:
        _fail("deterministic audit plan failed the minimum 50-per-scene gate")
    flagged_ids = {item["candidate_id"] for item in candidates if item["flags"]}
    selected_ids = {item["candidate_id"] for item in selected_ordered}
    if not flagged_ids.issubset(selected_ids):
        _fail("audit plan omitted flagged evidence")
    inventory_entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(selected_ordered, start=1):
        audit_id = f"audit_{index:06d}"
        inventory_entries.append(
            {
                "audit_id": audit_id,
                "candidate_id": item["candidate_id"],
                "deterministic_rank_sha256": _rank(seed, item["candidate_id"]),
                "stratum": _audit_stratum(item),
                "selection_reasons": selection_reasons[item["candidate_id"]],
                "diagnostic_flags": list(item["flags"]),
            }
        )
        rows.append(
            _blind_row(
                item,
                audit_id=audit_id,
                seed=seed,
                formal_start=formal_start,
                formal_end=formal_end,
            )
        )
    inventory = {
        "artifact_kind": "vap_gold_vad_candidate_internal_audit_selection_inventory",
        "format_version": AUDIT_FORMAT_VERSION,
        "candidate_only": True,
        "assembly_eligible": False,
        "counted_gold_ready": False,
        "visibility": "custodian_internal_until_completed_blind_decisions",
        "selection_contract": "deterministic_stratified_audit_selection_v1",
        "blind_missed_clip_policy": {
            "duration_s": _canonical_decimal(_BLIND_MISSED_CLIP_DURATION_S),
            "grid_stride_s": _canonical_decimal(_BLIND_MISSED_CLIP_GRID_STRIDE_S),
            "grid": "formal_start_plus_stride_with_scene_end_anchor",
            "candidate_boundary_padding": False,
            "tie_break": "source_hash_seeded_sha256",
        },
        "seed_sha256": seed,
        "minimum_per_scene": _MIN_AUDIT_PER_SCENE,
        "minimum_per_participant_when_possible": _MIN_AUDIT_PER_PARTICIPANT,
        "quotas": dict(_AUDIT_QUOTAS),
        "quota_selected": quota_counts,
        "participant_additions": participant_additions,
        "n_flagged": len(flagged_ids),
        "entries": inventory_entries,
    }
    queue = {
        "artifact_kind": AUDIT_QUEUE_KIND,
        "format_version": AUDIT_FORMAT_VERSION,
        "candidate_only": True,
        "assembly_eligible": False,
        "counted_gold_ready": False,
        "blind_protocol": {
            "suggestion_fields_present": False,
            "manual_decision_required_before_reveal": True,
            "decision_artifact_is_separate": True,
            "decision_autofill_supported": False,
        },
        "selection_contract": "deterministic_candidate_audit_selection_v1",
        "selection_inventory_sha256": _sha256_bytes(_canonical_json_bytes(inventory)),
        "n_selected": len(rows),
        "items": rows,
    }
    return queue, inventory


@_normalize_candidate_errors("candidate evidence computation")
def _compute_payload(
    snapshot: _InputSnapshot,
    channels: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    formal_start = snapshot.start_tick / FRAME_RATE_HZ
    formal_end = snapshot.end_tick / FRAME_RATE_HZ
    primary = _detector_view(channels, start_s=formal_start, window_ms=RMS_WINDOW_MS)
    diagnostic = _detector_view(
        channels, start_s=formal_start, window_ms=DIAGNOSTIC_RMS_WINDOW_MS
    )
    overlaps = _transcript_overlap_flags(
        snapshot.bridge.acceptance.events,
        snapshot.bridge.acceptance,
        start_s=formal_start,
        end_s=formal_end,
    )
    events = [
        _candidate_event(
            event,
            accepted=snapshot.bridge.acceptance,
            primary=primary,
            diagnostic=diagnostic,
            formal_start=formal_start,
            formal_end=formal_end,
            transcript_overlap=overlaps[event.source_index],
        )
        for event in snapshot.bridge.acceptance.events
    ]
    _annotate_cross_event_conflicts(
        events, formal_start=formal_start, formal_end=formal_end
    )
    missed = _missed_candidates(
        primary,
        accepted=snapshot.bridge.acceptance,
        formal_start=formal_start,
        formal_end=formal_end,
    )
    candidates = events + missed

    prevalence_review = {
        speaker: not 0.05 <= float(np.mean(primary.speech[speaker])) <= 0.60
        for speaker in SPEAKERS
    }
    for item in candidates:
        if prevalence_review[item["speaker"]]:
            item["flags"] = sorted(
                set([*item["flags"], "speaker_prevalence_outside_diagnostic_band"])
            )

    support: dict[str, Any] = {}
    common_start = max(primary.right_edges_s[0], diagnostic.right_edges_s[0])
    primary_common = primary.right_edges_s >= common_start - 1e-12
    diagnostic_common = diagnostic.right_edges_s >= common_start - 1e-12
    if not np.array_equal(
        primary.right_edges_s[primary_common], diagnostic.right_edges_s[diagnostic_common]
    ):
        _fail("20/40-ms diagnostic grids failed exact common-support alignment")
    for speaker in SPEAKERS:
        transcript = _transcript_mask(
            snapshot.bridge.acceptance,
            speaker,
            primary.right_edges_s,
            formal_start=formal_start,
            formal_end=formal_end,
        )
        support[speaker] = {
            "floor_db": _finite(primary.floors_db[speaker]),
            "n_10ms_frames": len(primary.right_edges_s),
            "transcript_prevalence": _finite(np.mean(transcript)),
            "candidate_20ms_prevalence": _finite(np.mean(primary.speech[speaker])),
            "prevalence_diagnostic_band": [0.05, 0.60],
            "prevalence_review_required": prevalence_review[speaker],
            "strong_core_prevalence": _finite(np.mean(primary.strong[speaker])),
            "diagnostic_40ms_prevalence": _finite(np.mean(diagnostic.speech[speaker])),
            "diagnostic_20_40_agreement": _finite(
                np.mean(
                    primary.speech[speaker][primary_common]
                    == diagnostic.speech[speaker][diagnostic_common]
                )
            ),
            "transcript_to_strong_lag": _binary_lag_diagnostic(
                transcript, primary.strong[speaker]
            ),
            "n_transcript_events": sum(
                event.speaker == speaker for event in snapshot.bridge.acceptance.events
            ),
            "n_missed_probe_flags": sum(item["speaker"] == speaker for item in missed),
        }

    parameter_sha = _sha256_bytes(_canonical_json_bytes(_PARAMETERS))
    source_fingerprint = _source_fingerprint(snapshot)
    seed = hashlib.sha256(
        f"{snapshot.scene}|{source_fingerprint}|{parameter_sha}".encode()
    ).hexdigest()
    audit_queue, audit_inventory = _build_audit_plan(
        candidates,
        seed=seed,
        formal_start=formal_start,
        formal_end=formal_end,
    )
    flag_counts: dict[str, int] = {}
    for item in candidates:
        for flag in item["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": snapshot.scene,
        "candidate_only": True,
        "assembly_eligible": False,
        "counted_gold_ready": False,
        "source_selection": SOURCE_SELECTION,
        "output_role": "candidate_gold_vad_boundary_evidence_and_blind_audit_plan_only",
        "trust_model": TRUST_MODEL,
        "parameters": json.loads(json.dumps(_PARAMETERS)),
        "parameter_sha256": parameter_sha,
        "formal_window": {
            "axis": "mic_A_shared_seconds",
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_tick_inclusive": snapshot.start_tick,
            "end_tick_exclusive": snapshot.end_tick,
            "start_s": _finite(formal_start),
            "end_s_exclusive": _finite(formal_end),
            "half_open": True,
            "no_extrapolation_or_zero_fill": True,
        },
        "channel_time_mapping": {
            "formula": "channel_local_time_s = shared_time_s - correction_s",
            "correction_s": {
                speaker: _canonical_decimal(snapshot.corrections_s[speaker])
                for speaker in SPEAKERS
            },
            "per_channel_sample_mapping": _channel_sample_mapping(snapshot),
        },
        "sources": {
            "analysis_base": str(snapshot.analysis_csv),
            "transcript_bridge": str(snapshot.bridge_path),
            "audio_gold_alignment": str(snapshot.gold_path),
            "source_fingerprint_sha256": source_fingerprint,
            "bindings": list(snapshot.source_bindings),
        },
        "scientific_boundary": {
            "transcript_role": "coarse_anchor_only_not_counted_gold",
            "missed_token_probe_action": "flag_only_never_auto_promote",
            "produces_p1_to_p4_vad": False,
            "writes_assembly_manifest": False,
            "requires_owner_parameter_approval": True,
            "requires_completed_human_blind_audit": True,
        },
        "diagnostics": {
            "per_speaker": support,
            "n_transcript_candidates": len(events),
            "n_missed_token_probe_flags": len(missed),
            "n_total_candidates": len(candidates),
            "flag_counts": dict(sorted(flag_counts.items())),
        },
        "candidates": candidates,
        "audit_selection_inventory": audit_inventory,
        "blind_audit_queue": audit_queue,
    }


@_normalize_candidate_errors("candidate source-guarded replay")
def _compute_with_authority(
    scene: str, *, authority: GoldVadCandidateAuthority
) -> dict[str, Any]:
    discovery = _load_inputs_with_authority(scene, authority=authority)
    discovery_fingerprint = _source_fingerprint(discovery)
    with _hold_source_files(discovery.source_bindings) as handles:
        replay = _load_inputs_with_authority(scene, authority=authority)
        if (
            _source_fingerprint(replay) != discovery_fingerprint
            or {
                transcript_fs_guard.lexical_absolute(binding["path"])
                for binding in replay.source_bindings
            }
            != set(handles)
        ):
            _fail("input source graph changed while guarded handles were acquired")
        channels = _decode_formal_channels(replay, handles)
        payload = _compute_payload(replay, channels)
        final_replay = _load_inputs_with_authority(scene, authority=authority)
        if _source_fingerprint(final_replay) != discovery_fingerprint:
            _fail("input source graph changed during candidate computation")
        return payload


def build_gold_vad_candidate(scene: str) -> dict[str, Any]:
    """Build candidate evidence in memory; no file or training label is produced."""

    return _compute_with_authority(scene, authority=default_gold_vad_candidate_authority())


def _validate_bundle_shape_impl(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("candidate bundle must be a JSON object")
    required = {
        "artifact_kind",
        "format_version",
        "scene",
        "candidate_only",
        "assembly_eligible",
        "counted_gold_ready",
        "source_selection",
        "output_role",
        "trust_model",
        "parameters",
        "parameter_sha256",
        "formal_window",
        "channel_time_mapping",
        "sources",
        "scientific_boundary",
        "diagnostics",
        "candidates",
        "audit_selection_inventory",
        "blind_audit_queue",
    }
    if set(value) != required:
        _fail("candidate bundle has missing or undeclared fields")
    if (
        value["artifact_kind"] != ARTIFACT_KIND
        or type(value["format_version"]) is not int
        or value["format_version"] != FORMAT_VERSION
        or value["candidate_only"] is not True
        or value["assembly_eligible"] is not False
        or value["counted_gold_ready"] is not False
        or value["source_selection"] != SOURCE_SELECTION
        or value["trust_model"] != TRUST_MODEL
    ):
        _fail("candidate-only role/readiness/version contract is invalid")
    if not _type_exact_equal(value["parameters"], _PARAMETERS):
        _fail("candidate parameters differ from the code-owned pending proposal")
    if value["parameter_sha256"] != _sha256_bytes(_canonical_json_bytes(_PARAMETERS)):
        _fail("candidate parameter hash is invalid")
    boundary = value["scientific_boundary"]
    if (
        type(boundary) is not dict
        or boundary.get("produces_p1_to_p4_vad") is not False
        or boundary.get("writes_assembly_manifest") is not False
        or boundary.get("requires_owner_parameter_approval") is not True
        or boundary.get("requires_completed_human_blind_audit") is not True
    ):
        _fail("candidate scientific boundary was weakened")
    candidates = value["candidates"]
    if type(candidates) is not list:
        _fail("candidate evidence must be a JSON array")
    candidate_ids: set[str] = set()
    for item in candidates:
        if type(item) is not dict or type(item.get("candidate_id")) is not str:
            _fail("candidate evidence row has no canonical candidate ID")
        if item["candidate_id"] in candidate_ids:
            _fail("candidate evidence contains a duplicate candidate ID")
        candidate_ids.add(item["candidate_id"])

    inventory = value["audit_selection_inventory"]
    if type(inventory) is not dict:
        _fail("internal audit selection inventory must be a JSON object")
    seed = inventory.get("seed_sha256")
    if type(seed) is not str or len(seed) != 64:
        _fail("internal audit selection inventory has an invalid deterministic seed")
    formal = value["formal_window"]
    if type(formal) is not dict:
        _fail("candidate formal window must be a JSON object")
    formal_start = formal.get("start_s")
    formal_end = formal.get("end_s_exclusive")
    if (
        type(formal_start) not in {int, float}
        or type(formal_end) not in {int, float}
        or isinstance(formal_start, bool)
        or isinstance(formal_end, bool)
        or not math.isfinite(float(formal_start))
        or not math.isfinite(float(formal_end))
        or not float(formal_start) < float(formal_end)
    ):
        _fail("candidate formal window has invalid finite half-open bounds")

    queue = value["blind_audit_queue"]
    if (
        type(queue) is not dict
        or set(queue)
        != {
            "artifact_kind",
            "format_version",
            "candidate_only",
            "assembly_eligible",
            "counted_gold_ready",
            "blind_protocol",
            "selection_contract",
            "selection_inventory_sha256",
            "n_selected",
            "items",
        }
        or queue.get("artifact_kind") != AUDIT_QUEUE_KIND
        or queue.get("format_version") != AUDIT_FORMAT_VERSION
        or queue.get("candidate_only") is not True
        or queue.get("assembly_eligible") is not False
        or queue.get("counted_gold_ready") is not False
    ):
        _fail("blind audit queue readiness contract is invalid")
    protocol = queue.get("blind_protocol")
    if (
        type(protocol) is not dict
        or protocol.get("suggestion_fields_present") is not False
        or protocol.get("manual_decision_required_before_reveal") is not True
        or protocol.get("decision_artifact_is_separate") is not True
        or protocol.get("decision_autofill_supported") is not False
    ):
        _fail("blind audit protocol was weakened")
    items = queue.get("items")
    if type(items) is not list or len(items) < _MIN_AUDIT_PER_SCENE:
        _fail("blind audit queue has fewer than 50 items")
    audit_ids: list[str] = []
    expected_blind_fields = {
        "audit_id",
        "speaker_channel",
        "coarse_event",
        "clip_interval",
        "suggestion_hidden_until_manual_decision",
        "hidden_suggestion_sha256",
    }
    for index, row in enumerate(items, start=1):
        if type(row) is not dict or set(row) != expected_blind_fields:
            _fail("blind audit item has diagnostic, selection, or undeclared fields")
        audit_id = row.get("audit_id")
        if audit_id != f"audit_{index:06d}":
            _fail("blind audit IDs must be neutral ordered identifiers")
        if row.get("suggestion_hidden_until_manual_decision") is not True:
            _fail("blind audit item does not hide suggestions")
        audit_ids.append(audit_id)
    if len(audit_ids) != len(set(audit_ids)):
        _fail("blind audit queue contains duplicate neutral audit IDs")
    expected_queue, expected_inventory = _build_audit_plan(
        candidates,
        seed=seed,
        formal_start=float(formal_start),
        formal_end=float(formal_end),
    )
    if not _type_exact_equal(inventory, expected_inventory):
        _fail("internal audit selection inventory does not deterministically replay")
    if not _type_exact_equal(queue, expected_queue):
        _fail("blind audit queue does not deterministically replay")
    return value


def _validate_bundle_shape(value: Any) -> dict[str, Any]:
    with _candidate_error_boundary("candidate bundle validation"):
        return _validate_bundle_shape_impl(value)


def _verify_bound_sources(bundle: Mapping[str, Any]) -> None:
    sources = bundle.get("sources")
    if type(sources) is not dict or type(sources.get("bindings")) is not list:
        _fail("candidate bundle lacks exact source bindings")
    seen: set[Path] = set()
    for binding in sources["bindings"]:
        if type(binding) is not dict or set(binding) != {"path", "size_bytes", "sha256"}:
            _fail("candidate source binding shape is invalid")
        path = transcript_fs_guard.lexical_absolute(str(binding["path"]))
        if path in seen:
            _fail("candidate source bindings contain a duplicate path")
        seen.add(path)
        actual = _binding_from_path(path)
        if not _type_exact_equal(actual, binding):
            _fail(f"candidate source changed: {path}")


@_normalize_candidate_errors("candidate artifact load")
def _load_with_authority(
    path: str | Path, *, authority: GoldVadCandidateAuthority
) -> dict[str, Any]:
    candidate_path = transcript_fs_guard.guard_regular_file_path(path)
    with transcript_fs_guard.guarded_read(candidate_path) as held:
        raw = transcript_fs_guard.read_guarded_bytes(held)
        parsed = _validate_bundle_shape(_strict_json_loads(raw, "candidate bundle"))
        _verify_bound_sources(parsed)
        scene = parsed["scene"]
        expected = _compute_with_authority(scene, authority=authority)
        if not _type_exact_equal(parsed, expected):
            _fail("candidate bundle does not exactly replay from current formal sources")
        transcript_fs_guard.verify_guarded_read(held)
        return parsed


def load_gold_vad_candidate(path: str | Path) -> dict[str, Any]:
    """Load only after full formal-source/hash/PCM and deterministic replay."""

    return _load_with_authority(path, authority=default_gold_vad_candidate_authority())


def _path_has_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return False
    return (
        not stat.S_ISLNK(value.st_mode)
        and not (getattr(value, "st_file_attributes", 0) & 0x00000400)
        and (value.st_dev, value.st_ino) == identity
    )


def _destination_is_owned_staged_link(
    staged: Path,
    destination: Path,
    *,
    staged_identity: tuple[int, int],
    expected_bytes: bytes,
) -> bool:
    """Prove both names still refer to our unchanged staged inode."""

    if not (
        _path_has_identity(staged, staged_identity)
        and _path_has_identity(destination, staged_identity)
    ):
        return False
    try:
        if not os.path.samefile(staged, destination):
            return False
        return staged.read_bytes() == expected_bytes and destination.read_bytes() == expected_bytes
    except OSError:
        return False


def _write_no_clobber_bytes(
    destination: Path,
    raw: bytes,
    *,
    post_validate: Callable[[Path], None] | None = None,
) -> Path:
    destination = transcript_fs_guard.guard_output_path(destination)
    if destination.exists():
        _fail(f"refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    staged_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.tmp.",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as handle:
            staged = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        staged_stat = staged.lstat()
        staged_identity = (staged_stat.st_dev, staged_stat.st_ino)
        with transcript_fs_guard.guarded_read(staged) as held:
            if transcript_fs_guard.read_guarded_bytes(held) != raw:
                _fail("staged artifact bytes changed")
        try:
            os.link(staged, destination)
        except FileExistsError as exc:
            raise GoldVadCandidateError(f"refusing concurrent overwrite: {destination}") from exc
        if post_validate is not None:
            post_validate(destination)
        return destination
    except BaseException:
        if (
            staged is not None
            and staged_identity is not None
            and _destination_is_owned_staged_link(
                staged,
                destination,
                staged_identity=staged_identity,
                expected_bytes=raw,
            )
        ):
            try:
                if _path_has_identity(destination, staged_identity):
                    destination.unlink()
            except OSError:
                pass
        raise
    finally:
        if (
            staged is not None
            and staged_identity is not None
            and _path_has_identity(staged, staged_identity)
        ):
            try:
                staged.unlink()
            except OSError:
                pass


@_normalize_candidate_errors("candidate artifact publication")
def _write_with_authority(
    scene: str,
    output: str | Path,
    *,
    authority: GoldVadCandidateAuthority,
) -> Path:
    destination = transcript_fs_guard.lexical_absolute(output)
    payload = _compute_with_authority(scene, authority=authority)
    raw = _canonical_json_bytes(payload)
    return _write_no_clobber_bytes(
        destination,
        raw,
        post_validate=lambda path: _load_with_authority(path, authority=authority),
    )


def write_gold_vad_candidate(scene: str, output: str | Path) -> Path:
    """Publish one no-clobber candidate evidence bundle (never formal gold)."""

    return _write_with_authority(
        scene, output, authority=default_gold_vad_candidate_authority()
    )


def _candidate_binding(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": len(raw), "sha256": _sha256_bytes(raw)}


@_normalize_candidate_errors("blind audit decision template publication")
def create_blind_audit_decisions(
    candidate_path: str | Path, output: str | Path
) -> Path:
    """Create a separate, suggestion-free human decision template."""

    bundle_path = transcript_fs_guard.guard_regular_file_path(candidate_path)
    with transcript_fs_guard.guarded_read(bundle_path) as held_bundle:
        raw_bundle = transcript_fs_guard.read_guarded_bytes(held_bundle)
        bundle = load_gold_vad_candidate(bundle_path)
        transcript_fs_guard.verify_guarded_read(held_bundle)
    decisions = {
        "artifact_kind": AUDIT_DECISIONS_KIND,
        "format_version": AUDIT_FORMAT_VERSION,
        "scene": bundle["scene"],
        "candidate_only": True,
        "assembly_eligible": False,
        "counted_gold_ready": False,
        "candidate_bundle": _candidate_binding(bundle_path, raw_bundle),
        "blind": True,
        "suggestions_present": False,
        "autofilled": False,
        "decisions": [
            {
                "audit_id": row["audit_id"],
                "listened": False,
                "speech_present": None,
                "speaker_correct": None,
                "manual_start_s": None,
                "manual_end_s_exclusive": None,
                "note": "",
            }
            for row in bundle["blind_audit_queue"]["items"]
        ],
    }
    with transcript_fs_guard.guarded_read(bundle_path) as held_bundle:
        if transcript_fs_guard.read_guarded_bytes(held_bundle) != raw_bundle:
            _fail("candidate bundle changed before blind decisions publication")
        return _write_no_clobber_bytes(
            transcript_fs_guard.lexical_absolute(output), _canonical_json_bytes(decisions)
        )


def _load_completed_decisions(
    path: str | Path,
    *,
    candidate_path: Path,
    candidate_raw: bytes,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    guarded = transcript_fs_guard.guard_regular_file_path(path)
    with transcript_fs_guard.guarded_read(guarded) as held:
        value = _strict_json_loads(
            transcript_fs_guard.read_guarded_bytes(held), "blind audit decisions"
        )
    if type(value) is not dict or set(value) != {
        "artifact_kind",
        "format_version",
        "scene",
        "candidate_only",
        "assembly_eligible",
        "counted_gold_ready",
        "candidate_bundle",
        "blind",
        "suggestions_present",
        "autofilled",
        "decisions",
    }:
        _fail("blind decisions artifact has missing or undeclared fields")
    if (
        value["artifact_kind"] != AUDIT_DECISIONS_KIND
        or type(value["format_version"]) is not int
        or value["format_version"] != AUDIT_FORMAT_VERSION
        or value["scene"] != candidate["scene"]
        or value["candidate_only"] is not True
        or value["assembly_eligible"] is not False
        or value["counted_gold_ready"] is not False
        or value["blind"] is not True
        or value["suggestions_present"] is not False
        or value["autofilled"] is not False
    ):
        _fail("blind decisions role/readiness contract is invalid")
    if not _type_exact_equal(
        value["candidate_bundle"], _candidate_binding(candidate_path, candidate_raw)
    ):
        _fail("blind decisions do not bind the exact candidate bundle")
    expected_ids = [row["audit_id"] for row in candidate["blind_audit_queue"]["items"]]
    decisions = value["decisions"]
    if type(decisions) is not list or [item.get("audit_id") for item in decisions] != expected_ids:
        _fail("blind decisions do not exactly cover the ordered audit queue")
    start = candidate["formal_window"]["start_s"]
    end = candidate["formal_window"]["end_s_exclusive"]
    for item in decisions:
        if type(item) is not dict or set(item) != {
            "audit_id",
            "listened",
            "speech_present",
            "speaker_correct",
            "manual_start_s",
            "manual_end_s_exclusive",
            "note",
        }:
            _fail("blind decision row has missing or undeclared fields")
        if item["listened"] is not True or type(item["speech_present"]) is not bool:
            _fail("every audit item must be listened to and receive an explicit speech decision")
        if type(item["speaker_correct"]) is not bool:
            _fail("every audit item requires an explicit speaker-channel decision")
        if type(item["note"]) is not str or not item["note"].strip():
            _fail("every audit item requires a non-empty human note")
        if item["speech_present"]:
            if (
                type(item["manual_start_s"]) is not str
                or type(item["manual_end_s_exclusive"]) is not str
            ):
                _fail("manual boundaries must be canonical decimal strings")
            manual_start_decimal = _decimal(item["manual_start_s"], "manual start")
            manual_end_decimal = _decimal(
                item["manual_end_s_exclusive"], "manual end"
            )
            if (
                _canonical_decimal(manual_start_decimal) != item["manual_start_s"]
                or _canonical_decimal(manual_end_decimal)
                != item["manual_end_s_exclusive"]
            ):
                _fail("manual boundaries must use canonical decimal spelling")
            manual_start = float(manual_start_decimal)
            manual_end = float(manual_end_decimal)
            if not (start <= manual_start < manual_end <= end):
                _fail("manual speech interval must be positive and inside the formal window")
        elif item["manual_start_s"] is not None or item["manual_end_s_exclusive"] is not None:
            _fail("speech-absent decision must keep both manual boundaries null")
    return value


@_normalize_candidate_errors("blind audit reveal publication")
def reveal_blind_audit(
    candidate_path: str | Path,
    decisions_path: str | Path,
    output: str | Path,
) -> Path:
    """Reveal suggestions only after every separate blind decision is complete."""

    bundle_path = transcript_fs_guard.guard_regular_file_path(candidate_path)
    decisions_file = transcript_fs_guard.guard_regular_file_path(decisions_path)
    with ExitStack() as stack:
        held_bundle = stack.enter_context(transcript_fs_guard.guarded_read(bundle_path))
        held_decisions = stack.enter_context(
            transcript_fs_guard.guarded_read(decisions_file)
        )
        raw_bundle = transcript_fs_guard.read_guarded_bytes(held_bundle)
        raw_decisions = transcript_fs_guard.read_guarded_bytes(held_decisions)
        bundle = load_gold_vad_candidate(bundle_path)
        decisions = _load_completed_decisions(
            decisions_file,
            candidate_path=bundle_path,
            candidate_raw=raw_bundle,
            candidate=bundle,
        )
        transcript_fs_guard.verify_guarded_read(held_bundle)
        transcript_fs_guard.verify_guarded_read(held_decisions)
    by_id = {item["candidate_id"]: item for item in bundle["candidates"]}
    inventory_by_audit = {
        item["audit_id"]: item
        for item in bundle["audit_selection_inventory"]["entries"]
    }
    decision_by_id = {item["audit_id"]: item for item in decisions["decisions"]}
    items = []
    for blind in bundle["blind_audit_queue"]["items"]:
        inventory = inventory_by_audit[blind["audit_id"]]
        candidate = by_id[inventory["candidate_id"]]
        if (
            _sha256_bytes(_canonical_json_bytes(_hidden_suggestion(candidate)))
            != blind["hidden_suggestion_sha256"]
        ):
            _fail("hidden suggestion commitment does not match candidate evidence")
        items.append(
            {
                "audit_id": blind["audit_id"],
                "manual_decision": decision_by_id[blind["audit_id"]],
                "revealed_selection_inventory": inventory,
                "revealed_automatic_suggestion": candidate["suggestion"],
                "revealed_diagnostic_40ms": candidate["diagnostic_40ms"],
                "revealed_evidence": candidate["evidence"],
                "automatic_acceptance": False,
            }
        )
    reveal = {
        "artifact_kind": AUDIT_REVEAL_KIND,
        "format_version": AUDIT_FORMAT_VERSION,
        "scene": bundle["scene"],
        "candidate_only": True,
        "assembly_eligible": False,
        "counted_gold_ready": False,
        "candidate_bundle": _candidate_binding(bundle_path, raw_bundle),
        "blind_decisions": _candidate_binding(decisions_file, raw_decisions),
        "manual_decisions_preceded_reveal": True,
        "automatic_label_promotion": False,
        "items": items,
    }
    with ExitStack() as stack:
        held_bundle = stack.enter_context(transcript_fs_guard.guarded_read(bundle_path))
        held_decisions = stack.enter_context(
            transcript_fs_guard.guarded_read(decisions_file)
        )
        if transcript_fs_guard.read_guarded_bytes(held_bundle) != raw_bundle:
            _fail("candidate bundle changed before audit reveal publication")
        if transcript_fs_guard.read_guarded_bytes(held_decisions) != raw_decisions:
            _fail("blind decisions changed before audit reveal publication")
        return _write_no_clobber_bytes(
            transcript_fs_guard.lexical_absolute(output), _canonical_json_bytes(reveal)
        )


__all__ = [
    "ARTIFACT_KIND",
    "ASSEMBLY_ELIGIBLE",
    "CANDIDATE_ONLY",
    "COUNTED_GOLD_READY",
    "FORMAT_VERSION",
    "GoldVadCandidateAuthority",
    "GoldVadCandidateError",
    "SOURCE_SELECTION",
    "build_gold_vad_candidate",
    "create_blind_audit_decisions",
    "load_gold_vad_candidate",
    "reveal_blind_audit",
    "write_gold_vad_candidate",
]
