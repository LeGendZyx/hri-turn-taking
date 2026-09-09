"""Versioned, non-destructive time maps for per-person gold-audio channels.

This module deliberately sits *after* :mod:`audio_contract`.  It cannot change
the source WAVs, cannot promote an audio source audit to counted-gold status,
and cannot accept a caller-provided shift. Identity is tested first on the
formal window; only a channel that fails the frozen coarse/fine hierarchy gets
the single deterministic correction ``-full_file_median_residual``.

The resulting map is authorised only on the intersection of the published
audio coverage and the sealed pose-intake tick range.  It is gold-label
infrastructure and must never be used as a robot-observable model input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.data import assembly, audio_contract

ARTIFACT_KIND = "vap_audio_gold_channel_time_map"
FORMAT_VERSION = 1
OUTPUT_ROLE = "gold_vad_channel_time_alignment_only"
TRUST_MODEL = "trusted_code_integrity_and_reproducibility_not_signature"
FRAME_RATE_HZ = 25
FINE_ENVELOPE_HOP_MS = 5
MAXIMUM_RESOLUTION_DRIFT_MS = 72.0
SPEAKERS = audio_contract.SPEAKERS
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AudioGoldAlignmentError(ValueError):
    """Raised when a gold-channel time map cannot be reproduced exactly."""


@dataclass(frozen=True)
class GoldAlignmentAuthority:
    """Code-owned roots for the two already-published upstream artifacts."""

    audio_manifest_root: Path
    pose_base_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_manifest_root", Path(self.audio_manifest_root).resolve())
        object.__setattr__(self, "pose_base_root", Path(self.pose_base_root).resolve())


@dataclass(frozen=True)
class GoldAlignmentWindow:
    """Exact consumer view of one authorised gold-audio interval.

    Tick bounds are half-open on the 25-Hz mic-A shared grid. ``correction_s``
    applies only to named A/B/C/R gold channels, never to model-input audio.
    """

    scene: str
    start_tick_inclusive: int
    end_tick_exclusive: int
    correction_s: dict[str, Decimal]

    @property
    def frame_rate_hz(self) -> int:
        return FRAME_RATE_HZ

    @property
    def start_s(self) -> Decimal:
        return Decimal(self.start_tick_inclusive) / Decimal(FRAME_RATE_HZ)

    @property
    def end_s_exclusive(self) -> Decimal:
        return Decimal(self.end_tick_exclusive) / Decimal(FRAME_RATE_HZ)

    def channel_local_time(self, speaker: str, tick: int) -> Decimal:
        """Map an authorised shared tick back to a channel-local timestamp."""

        if speaker not in SPEAKERS:
            _fail(f"speaker must be one of {SPEAKERS}")
        if type(tick) is not int:
            _fail("shared tick must be an exact integer")
        if tick < self.start_tick_inclusive or tick >= self.end_tick_exclusive:
            _fail("shared tick is outside the authorised half-open window")
        shared_s = Decimal(tick) / Decimal(FRAME_RATE_HZ)
        return shared_s - self.correction_s[speaker]


@dataclass(frozen=True)
class _PoseSnapshot:
    csv_path: Path
    sidecar_path: Path
    csv_sha256: str
    sidecar_sha256: str
    provenance: dict[str, Any]
    timestamp_s: np.ndarray


def default_gold_alignment_authority() -> GoldAlignmentAuthority:
    project_root = Path(__file__).resolve().parents[3]
    return GoldAlignmentAuthority(
        audio_manifest_root=project_root / "data" / "annotations" / "audio_contract",
        pose_base_root=project_root / "data" / "processed" / "intake_v2",
    )


def _fail(message: str) -> None:
    raise AudioGoldAlignmentError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AudioGoldAlignmentError(f"{label} is not strict UTF-8") from exc

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        _fail(f"{label} contains forbidden non-finite JSON constant {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise AudioGoldAlignmentError(
            f"{label} is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


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


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        _fail("a time-map Decimal must be finite")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _decimal_string(value: Any, label: str) -> Decimal:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise AudioGoldAlignmentError(f"{label} is not decimal") from exc
    if not parsed.is_finite() or _canonical_decimal(parsed) != value:
        _fail(f"{label} must be a finite canonical decimal string")
    return parsed


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact JSON integer")
    return value


def _canonical_paths(scene: str, authority: GoldAlignmentAuthority) -> tuple[Path, Path, Path]:
    if scene not in audio_contract.ACCEPTED_SCENES:
        _fail(f"scene is not in the frozen ten-scene authority: {scene!r}")
    audio_manifest = authority.audio_manifest_root / f"{scene}.audio_contract.json"
    pose_csv = authority.pose_base_root / f"{scene}.csv"
    pose_sidecar = pose_csv.with_suffix(".provenance.json")
    return audio_manifest.resolve(), pose_csv.resolve(), pose_sidecar.resolve()


def _file_binding(path: Path, *, artifact_kind: str, format_version: int) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"required upstream artifact does not exist: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "artifact_kind": artifact_kind,
        "format_version": format_version,
    }


def _replay_audio_manifest(path: Path) -> dict[str, Any]:
    try:
        return audio_contract.load_audio_contract(path)
    except (OSError, ValueError) as exc:
        raise AudioGoldAlignmentError(f"published audio contract replay failed: {exc}") from exc


def _replay_pose_base(csv_path: Path) -> _PoseSnapshot:
    try:
        loaded = assembly._load_dataset_file(csv_path, data_kind="real")
    except (OSError, ValueError) as exc:
        raise AudioGoldAlignmentError(f"sealed pose-base replay failed: {exc}") from exc
    provenance = loaded.provenance
    if type(provenance) is not dict:
        _fail("sealed pose base has no real-data provenance")
    if provenance.get("artifact_kind") != assembly.REAL_INTAKE_ARTIFACT_KIND:
        _fail("gold alignment must bind a pose-intake base, not an assembled feature table")
    if provenance.get("format_version") != assembly.REAL_PROVENANCE_FORMAT_VERSION:
        _fail("sealed pose base has an unsupported provenance version")
    timestamps = loaded.frame_df["timestamp_s"].to_numpy(dtype=np.float64, copy=True)
    return _PoseSnapshot(
        csv_path=csv_path,
        sidecar_path=csv_path.with_suffix(".provenance.json"),
        csv_sha256=loaded.data_sha256,
        sidecar_sha256=loaded.provenance_sha256,
        provenance=provenance,
        timestamp_s=timestamps,
    )


def _require_audio_prerequisites(report: dict[str, Any], scene: str) -> None:
    if report.get("artifact_kind") != audio_contract.ARTIFACT_KIND:
        _fail("upstream audio artifact kind is unsupported")
    if report.get("format_version") != audio_contract.FORMAT_VERSION:
        _fail("upstream audio artifact version is unsupported")
    if report.get("scene") != scene:
        _fail("upstream audio contract scene does not match its canonical filename")
    if report.get("counted_gold_ready") is not False:
        _fail("upstream source evidence must not claim counted-gold readiness")
    if not _type_exact_json_equal(
        report.get("audit_parameters"), audio_contract.DEFAULT_PARAMETERS.as_dict()
    ):
        _fail("upstream audio audit parameters are not the frozen v1 parameters")
    gates = report.get("gates")
    if type(gates) is not dict:
        _fail("upstream audio gates are missing")
    for key in (
        "source_identity_pass",
        "master_exact_permutation_pass",
        "reference_segmented_sync_pass",
    ):
        if gates.get(key) is not True:
            _fail(f"upstream audio prerequisite {key} did not pass")


def _derive_channel_mappings(
    report: dict[str, Any], *, correct_speakers: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Derive identity/candidate maps without accepting a numeric override."""

    correct_speakers = set() if correct_speakers is None else set(correct_speakers)
    if not correct_speakers <= set(SPEAKERS[1:]):
        _fail("correction candidates must be a subset of B/C/R")
    mappings: dict[str, dict[str, Any]] = {
        "A": {
            "kind": "identity_anchor",
            "correction_s": "0",
            "source_quality_pass": True,
            "source_median_residual_ms": None,
            "derivation": "mic_A_shared_axis_anchor",
        }
    }
    audits = report.get("inter_lavalier_to_mic_a")
    if type(audits) is not dict or set(audits) != set(SPEAKERS[1:]):
        _fail("upstream inter-lavalier audit must cover exactly B/C/R")
    for speaker in SPEAKERS[1:]:
        audit = audits[speaker]
        if type(audit) is not dict or type(audit.get("quality_pass")) is not bool:
            _fail(f"upstream {speaker} audit is malformed")
        median = audit.get("median_residual_ms")
        if type(median) not in (int, float) or isinstance(median, bool):
            _fail(f"upstream {speaker} median residual must be a finite number")
        if not math.isfinite(float(median)):
            _fail(f"upstream {speaker} median residual must be finite")
        if speaker not in correct_speakers:
            kind = "identity"
            correction = Decimal(0)
            derivation = "minimal_identity_selected_by_formal_window_hierarchy"
        else:
            kind = "constant_shift"
            correction = -(Decimal(str(median)) / Decimal(1000))
            derivation = "negative_full_file_v1_median_residual"
        mappings[speaker] = {
            "kind": kind,
            "correction_s": _canonical_decimal(correction),
            "source_quality_pass": audit["quality_pass"],
            "source_median_residual_ms": median,
            "derivation": derivation,
        }
    return mappings


def _pose_tick_range(pose: _PoseSnapshot, scene: str) -> tuple[int, int]:
    provenance = pose.provenance
    if provenance.get("session_id") != scene:
        _fail("pose-base session does not match the requested scene")
    if provenance.get("review_status") != "confirmed":
        _fail("pose-base review status is not confirmed")
    if provenance.get("identity_authority") != "sealed_hpe_transcript_crosswalk":
        _fail("pose-base identity authority is not the sealed HPE/Transcript crosswalk")
    if provenance.get("frame_rate_hz") != FRAME_RATE_HZ:
        _fail("pose-base frame rate must be the exact integer 25")
    tick_range = provenance.get("output_tick_range")
    if type(tick_range) is not dict or set(tick_range) != {
        "start_inclusive",
        "end_exclusive",
    }:
        _fail("pose-base output_tick_range is malformed")
    start = _exact_int(tick_range["start_inclusive"], "pose tick start")
    end = _exact_int(tick_range["end_exclusive"], "pose tick end")
    if start < 0 or end <= start or end - start != len(pose.timestamp_s):
        _fail("pose-base tick range is empty or disagrees with its row count")
    expected = np.arange(start, end, dtype=np.float64) / FRAME_RATE_HZ
    if not np.array_equal(pose.timestamp_s, expected):
        _fail("pose-base timestamps are not the exact contiguous 25-Hz tick grid")
    return start, end


def _ceil_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _derive_authorized_window(
    audio_report: dict[str, Any],
    mappings: dict[str, dict[str, Any]],
    pose: _PoseSnapshot,
    scene: str,
) -> dict[str, Any]:
    pose_start_tick, pose_end_tick = _pose_tick_range(pose, scene)
    sources = audio_report.get("sources")
    if type(sources) is not dict:
        _fail("upstream audio source bindings are missing")
    coverage_starts: list[Decimal] = []
    coverage_ends: list[Decimal] = []
    for speaker in SPEAKERS:
        binding = sources.get(speaker)
        if type(binding) is not dict or type(binding.get("wave_header")) is not dict:
            _fail(f"upstream {speaker} WAV header is missing")
        header = binding["wave_header"]
        n_frames = _exact_int(header.get("n_frames"), f"{speaker} n_frames")
        rate_hz = _exact_int(header.get("sample_rate_hz"), f"{speaker} sample_rate_hz")
        if n_frames <= 0 or rate_hz <= 0:
            _fail(f"upstream {speaker} WAV duration is invalid")
        correction = _decimal_string(mappings[speaker]["correction_s"], "correction_s")
        coverage_starts.append(correction)
        coverage_ends.append(Decimal(n_frames) / Decimal(rate_hz) + correction)
    required = audio_report.get("analysis_window_contract", {}).get(
        "required_reference_coverage"
    )
    if type(required) is not dict:
        _fail("upstream required reference coverage is missing")
    try:
        reference_start = Decimal(required["start_s"])
        reference_end = Decimal(required["end_s_exclusive"])
    except (InvalidOperation, KeyError, TypeError) as exc:
        raise AudioGoldAlignmentError("upstream reference coverage is malformed") from exc
    if not reference_start.is_finite() or not reference_end.is_finite():
        _fail("upstream reference coverage must be finite")

    lavalier_start = max(coverage_starts)
    lavalier_end = min(coverage_ends)
    pose_start = Decimal(pose_start_tick) / Decimal(FRAME_RATE_HZ)
    pose_end = Decimal(pose_end_tick) / Decimal(FRAME_RATE_HZ)
    start = max(lavalier_start, reference_start, pose_start)
    end = min(lavalier_end, reference_end, pose_end)
    if end <= start:
        _fail("audio/reference/sealed-pose intersection is empty")
    tick_start = max(pose_start_tick, _ceil_decimal(start * FRAME_RATE_HZ))
    tick_end = min(pose_end_tick, _ceil_decimal(end * FRAME_RATE_HZ))
    if tick_end <= tick_start:
        _fail("audio/reference/sealed-pose intersection contains no 25-Hz tick")
    first = Decimal(tick_start) / Decimal(FRAME_RATE_HZ)
    last = Decimal(tick_end - 1) / Decimal(FRAME_RATE_HZ)
    if first < start or last >= end:
        _fail("half-open Decimal tick projection escaped its source coverage")
    return {
        "axis": "mic_A_shared_seconds",
        "source_common_coverage": {
            "start_s": _canonical_decimal(lavalier_start),
            "end_s_exclusive": _canonical_decimal(lavalier_end),
        },
        "required_reference_coverage": {
            "start_s": _canonical_decimal(reference_start),
            "end_s_exclusive": _canonical_decimal(reference_end),
        },
        "sealed_pose_tick_range": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_inclusive": pose_start_tick,
            "end_exclusive": pose_end_tick,
        },
        "authorized_shared_window": {
            "start_s": _canonical_decimal(start),
            "end_s_exclusive": _canonical_decimal(end),
        },
        "authorized_tick_range": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_inclusive": tick_start,
            "end_exclusive": tick_end,
            "n_ticks": tick_end - tick_start,
        },
        "consumer_inverse_formula": "channel_local_time_s = shared_time_s - correction_s",
        "no_extrapolation": True,
        "missing_coverage_is_never_zero_filled": True,
    }


def _corrected_formal_window_audit(
    audio_report: dict[str, Any],
    mappings: dict[str, dict[str, Any]],
    window: dict[str, Any],
    *,
    envelopes: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    sources = audio_report["sources"]
    if envelopes is None:
        envelopes = {
            speaker: audio_contract._hop_rms_envelope(
                Path(sources[speaker]["path"]), mixdown=False
            )
            for speaker in SPEAKERS
        }
    hop_s = Decimal(audio_contract.ENVELOPE_HOP_MS) / Decimal(1000)
    authorized = window["authorized_shared_window"]
    start = _decimal_string(authorized["start_s"], "authorized start")
    end = _decimal_string(authorized["end_s_exclusive"], "authorized end")
    audit_start = _ceil_decimal(start / hop_s)
    audit_end = int((end / hop_s).to_integral_value(rounding=ROUND_FLOOR))
    if audit_end <= audit_start:
        _fail("authorized window is too short for the frozen 32-ms audit")
    reference = envelopes["A"][audit_start:audit_end]
    results: dict[str, Any] = {}
    for speaker in SPEAKERS[1:]:
        correction = _decimal_string(mappings[speaker]["correction_s"], "correction_s")
        shared_indices = np.arange(audit_start, audit_end, dtype=np.float64)
        local_positions = shared_indices - float(correction / hop_s)
        source_indices = np.arange(len(envelopes[speaker]), dtype=np.float64)
        if local_positions[0] < 0 or local_positions[-1] > source_indices[-1]:
            _fail(f"{speaker} corrected audit would extrapolate beyond real audio")
        corrected = np.interp(local_positions, source_indices, envelopes[speaker])
        audit = audio_contract._segmented_alignment_audit(
            reference,
            corrected,
            declared_lag_hops=0,
            parameters=audio_contract.DEFAULT_PARAMETERS,
        )
        origin_s = audit_start * float(hop_s)
        for segment in audit["segments"]:
            segment["center_s"] = round(segment["center_s"] + origin_s, 6)
        results[speaker] = audit
    all_channels_pass = all(audit["quality_pass"] is True for audit in results.values())
    return {
        "envelope": "fractional_linear_interpolation_of_frozen_32ms_log_rms",
        "audit_start_s": _canonical_decimal(Decimal(audit_start) * hop_s),
        "audit_end_s_exclusive": _canonical_decimal(Decimal(audit_end) * hop_s),
        "parameters": audio_contract.DEFAULT_PARAMETERS.as_dict(),
        "channels_to_mic_A": results,
        "all_channels_pass": all_channels_pass,
    }


def _fine_rms_envelope(path: Path) -> np.ndarray:
    """Independent memory-bounded 5-ms log-RMS envelope for ambiguity resolution."""

    blocks: list[np.ndarray] = []
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            _fail(f"{path.name} must remain mono 16-bit PCM for fine replay")
        rate = handle.getframerate()
        numerator = rate * FINE_ENVELOPE_HOP_MS
        if numerator % 1000:
            _fail(f"{path.name} cannot express an exact 5-ms envelope hop")
        hop = numerator // 1000
        declared = handle.getnframes()
        decoded = 0
        carry = np.empty(0, dtype=np.float64)
        while True:
            raw = handle.readframes(65_536)
            if not raw:
                break
            if len(raw) % 2:
                _fail(f"{path.name} yielded a partial PCM sample")
            signal = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            decoded += signal.size
            combined = np.concatenate((carry, signal)) if carry.size else signal
            complete = combined.size // hop
            if complete:
                framed = combined[: complete * hop].reshape(complete, hop)
                blocks.append(np.sqrt(np.mean(framed * framed, axis=1)))
            carry = combined[complete * hop :].copy()
    if decoded != declared or not blocks:
        _fail(f"{path.name} fine-envelope PCM replay is incomplete")
    return np.log(np.maximum(np.concatenate(blocks), 1e-8))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - float(left.mean())
    right = right - float(right.mean())
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    slopes: list[float] = []
    for index in range(len(x)):
        dx = x[index + 1 :] - x[index]
        dy = y[index + 1 :] - y[index]
        slopes.extend((dy / dx)[dx != 0].tolist())
    return float(np.median(slopes)) if slopes else 0.0


def _fine_segmented_audit(reference: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    parameters = audio_contract.DEFAULT_PARAMETERS
    segment_frames = parameters.segment_seconds * 1000 // FINE_ENVELOPE_HOP_MS
    minimum_frames = parameters.minimum_segment_seconds * 1000 // FINE_ENVELOPE_HOP_MS
    search_hops = math.ceil(parameters.search_radius_ms / FINE_ENVELOPE_HOP_MS)
    exclusion_hops = math.ceil(audio_contract.ENVELOPE_HOP_MS / FINE_ENVELOPE_HOP_MS)
    lower = search_hops
    upper = min(len(reference), len(target)) - search_hops
    segments: list[dict[str, Any]] = []
    start = lower
    while start < upper:
        stop = min(start + segment_frames, upper)
        if stop - start < minimum_frames:
            break
        left = reference[start:stop]
        scores = {
            lag: _correlation(left, target[start + lag : stop + lag])
            for lag in range(-search_hops, search_hops + 1)
        }
        finite = {lag: score for lag, score in scores.items() if math.isfinite(score)}
        if not finite:
            _fail("fine-resolution audit encountered a degenerate envelope segment")
        best_lag = max(finite, key=finite.get)
        best_score = finite[best_lag]
        other = [
            score for lag, score in finite.items() if abs(lag - best_lag) > exclusion_hops
        ]
        second = max(other) if other else -1.0
        refined = float(best_lag)
        if best_lag - 1 in finite and best_lag + 1 in finite:
            curvature = finite[best_lag - 1] - (2 * best_score) + finite[best_lag + 1]
            if abs(curvature) > 1e-12:
                delta = 0.5 * (finite[best_lag - 1] - finite[best_lag + 1]) / curvature
                refined += float(np.clip(delta, -0.5, 0.5))
        margin = best_score - second
        segments.append(
            {
                "center_s": round((start + stop) * FINE_ENVELOPE_HOP_MS / 2000, 6),
                "status": "usable",
                "residual_ms": round(refined * FINE_ENVELOPE_HOP_MS, 6),
                "peak_correlation": round(best_score, 9),
                "second_peak_correlation": round(second, 9),
                "peak_margin": round(margin, 9),
                "unique_peak": (
                    best_score >= parameters.minimum_peak_correlation
                    and margin >= parameters.minimum_peak_margin
                ),
            }
        )
        start = stop
    residuals = np.asarray([item["residual_ms"] for item in segments], dtype=float)
    centers = np.asarray([item["center_s"] for item in segments], dtype=float)
    if not len(residuals):
        _fail("fine-resolution audit found no usable segments")
    median = float(np.median(residuals))
    p95 = float(np.quantile(np.abs(residuals), 0.95))
    drift_rate = _theil_sen_slope(centers, residuals)
    drift_span = drift_rate * float(centers.max() - centers.min()) if len(centers) > 1 else 0.0
    unique_fraction = float(np.mean([item["unique_peak"] for item in segments]))
    median_corr = float(np.median([item["peak_correlation"] for item in segments]))
    reasons: list[str] = []
    if len(segments) < parameters.minimum_usable_segments:
        reasons.append("insufficient_usable_segments")
    if abs(median) > parameters.maximum_abs_median_residual_ms:
        reasons.append("median_residual_exceeds_limit")
    if p95 > parameters.maximum_abs_p95_residual_ms:
        reasons.append("p95_residual_exceeds_limit")
    if abs(drift_span) > parameters.maximum_drift_span_ms:
        reasons.append("drift_span_exceeds_limit")
    if unique_fraction < parameters.minimum_unique_fraction:
        reasons.append("peak_uniqueness_below_limit")
    if median_corr < parameters.minimum_peak_correlation:
        reasons.append("correlation_below_limit")
    return {
        "n_segments": len(segments),
        "n_usable_segments": len(segments),
        "median_residual_ms": round(median, 6),
        "p95_abs_residual_ms": round(p95, 6),
        "drift_ms_per_s": round(drift_rate, 9),
        "drift_span_ms": round(drift_span, 6),
        "unique_peak_fraction": round(unique_fraction, 6),
        "median_peak_correlation": round(median_corr, 9),
        "quality_pass": not reasons,
        "failure_reasons": reasons,
        "segments": segments,
    }


def _fine_channel_audit(
    audio_report: dict[str, Any],
    speaker: str,
    correction_s: str,
    window: dict[str, Any],
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    sources = audio_report["sources"]
    for name in ("A", speaker):
        if name not in cache:
            cache[name] = _fine_rms_envelope(Path(sources[name]["path"]))
    hop_s = Decimal(FINE_ENVELOPE_HOP_MS) / Decimal(1000)
    authorized = window["authorized_shared_window"]
    start_s = _decimal_string(authorized["start_s"], "authorized start")
    end_s = _decimal_string(authorized["end_s_exclusive"], "authorized end")
    start = _ceil_decimal(start_s / hop_s)
    end = int((end_s / hop_s).to_integral_value(rounding=ROUND_FLOOR))
    shared = np.arange(start, end, dtype=np.float64)
    correction = _decimal_string(correction_s, "correction_s")
    positions = shared - float(correction / hop_s)
    source_index = np.arange(len(cache[speaker]), dtype=np.float64)
    if positions[0] < 0 or positions[-1] > source_index[-1]:
        _fail(f"{speaker} fine-resolution replay would extrapolate")
    mapped = np.interp(positions, source_index, cache[speaker])
    audit = _fine_segmented_audit(cache["A"][start:end], mapped)
    origin_s = start * float(hop_s)
    for segment in audit["segments"]:
        segment["center_s"] = round(segment["center_s"] + origin_s, 6)
    return audit


def _resolution_status(coarse: dict[str, Any], fine: dict[str, Any] | None) -> str:
    if coarse["quality_pass"] is True:
        return "coarse_confirmed"
    drift = coarse.get("drift_span_ms")
    if (
        coarse.get("failure_reasons") == ["drift_span_exceeds_limit"]
        and type(drift) in (int, float)
        and not isinstance(drift, bool)
        and abs(float(drift)) <= MAXIMUM_RESOLUTION_DRIFT_MS
        and fine is not None
        and fine["quality_pass"] is True
    ):
        return "resolution_confirmed"
    return "failed"


def _assert_stable_audio(before: dict[str, Any], after: dict[str, Any]) -> None:
    if not _type_exact_json_equal(before, after):
        _fail("audio sources or their published contract changed during gold-map replay")


def _assert_stable_pose(before: _PoseSnapshot, after: _PoseSnapshot) -> None:
    if (
        before.csv_sha256 != after.csv_sha256
        or before.sidecar_sha256 != after.sidecar_sha256
        or not _type_exact_json_equal(before.provenance, after.provenance)
        or not np.array_equal(before.timestamp_s, after.timestamp_s)
    ):
        _fail("pose base or its provenance changed during gold-map replay")


def _audit_scene_with_authority(
    scene: str, *, authority: GoldAlignmentAuthority
) -> dict[str, Any]:
    audio_path, pose_csv, pose_sidecar = _canonical_paths(scene, authority)
    audio_before = _replay_audio_manifest(audio_path)
    _require_audio_prerequisites(audio_before, scene)
    pose_before = _replay_pose_base(pose_csv)
    envelopes_32ms = {
        speaker: audio_contract._hop_rms_envelope(
            Path(audio_before["sources"][speaker]["path"]), mixdown=False
        )
        for speaker in SPEAKERS
    }
    fine_cache: dict[str, np.ndarray] = {}
    corrected_speakers: set[str] = set()
    identity_evidence: dict[str, Any] | None = None
    final_decisions: dict[str, Any] = {}
    # Identity can transition to the one deterministic full-file-median map,
    # never back again.  At most three transitions are therefore possible.
    for _ in range(len(SPEAKERS)):
        mappings = _derive_channel_mappings(
            audio_before, correct_speakers=corrected_speakers
        )
        window = _derive_authorized_window(audio_before, mappings, pose_before, scene)
        coarse = _corrected_formal_window_audit(
            audio_before, mappings, window, envelopes=envelopes_32ms
        )
        decisions: dict[str, Any] = {}
        promote: set[str] = set()
        for speaker in SPEAKERS[1:]:
            channel_coarse = coarse["channels_to_mic_A"][speaker]
            fine: dict[str, Any] | None = None
            drift = channel_coarse.get("drift_span_ms")
            fine_eligible = (
                channel_coarse.get("failure_reasons") == ["drift_span_exceeds_limit"]
                and type(drift) in (int, float)
                and not isinstance(drift, bool)
                and abs(float(drift)) <= MAXIMUM_RESOLUTION_DRIFT_MS
            )
            if fine_eligible:
                fine = _fine_channel_audit(
                    audio_before,
                    speaker,
                    mappings[speaker]["correction_s"],
                    window,
                    fine_cache,
                )
            status = _resolution_status(channel_coarse, fine)
            decisions[speaker] = {
                "attempt_count": 2 if speaker in corrected_speakers else 1,
                "selected_mapping_kind": mappings[speaker]["kind"],
                "status": status,
                "coarse_warning": status == "resolution_confirmed",
                "coarse_32ms": channel_coarse,
                "fine_5ms": fine,
            }
            if status == "failed" and speaker not in corrected_speakers:
                promote.add(speaker)
        if identity_evidence is None:
            identity_evidence = {
                speaker: copy for speaker, copy in decisions.items()
            }
        if not promote:
            final_decisions = decisions
            selected_audit = coarse
            break
        corrected_speakers.update(promote)
    else:  # pragma: no cover - monotonic three-channel loop makes this defensive only
        _fail("channel-map hierarchy did not converge")

    formal_pass = all(
        decision["status"] in {"coarse_confirmed", "resolution_confirmed"}
        for decision in final_decisions.values()
    )
    window["gold_vad_use_authorized"] = formal_pass

    audio_after = _replay_audio_manifest(audio_path)
    pose_after = _replay_pose_base(pose_csv)
    _assert_stable_audio(audio_before, audio_after)
    _assert_stable_pose(pose_before, pose_after)

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "counted_gold_ready": False,
        "trust_model": TRUST_MODEL,
        "sources": {
            "audio_contract": _file_binding(
                audio_path,
                artifact_kind=audio_contract.ARTIFACT_KIND,
                format_version=audio_contract.FORMAT_VERSION,
            ),
            "pose_csv": _file_binding(
                pose_csv,
                artifact_kind=assembly.REAL_INTAKE_ARTIFACT_KIND,
                format_version=assembly.REAL_PROVENANCE_FORMAT_VERSION,
            ),
            "pose_provenance": _file_binding(
                pose_sidecar,
                artifact_kind=assembly.REAL_INTAKE_ARTIFACT_KIND,
                format_version=assembly.REAL_PROVENANCE_FORMAT_VERSION,
            ),
        },
        "mapping_contract": {
            "shared_axis": "mic_A_shared_seconds",
            "forward_formula": "shared_time_s = channel_local_time_s + correction_s",
            "allowed_mapping_kinds": ["identity_anchor", "identity", "constant_shift"],
            "linear_or_manual_maps_allowed": False,
            "channels": mappings,
        },
        "authorization": window,
        "formal_window_replay": {
            "selection_rule": (
                "identity_first_then_negative_full_file_v1_median; "
                "no_formal_window_parameter_fitting"
            ),
            "resolution_rule": {
                "coarse_hop_ms": audio_contract.ENVELOPE_HOP_MS,
                "fine_hop_ms": FINE_ENVELOPE_HOP_MS,
                "coarse_drift_limit_ms": (
                    audio_contract.DEFAULT_PARAMETERS.maximum_drift_span_ms
                ),
                "resolution_max_abs_coarse_drift_ms": MAXIMUM_RESOLUTION_DRIFT_MS,
                "fine_uses_all_original_40ms_quality_gates": True,
                "fine_envelope": "nonoverlapping_rms_hop_then_log",
                "fine_segment_seconds": (
                    audio_contract.DEFAULT_PARAMETERS.segment_seconds
                ),
                "fine_minimum_segment_seconds": (
                    audio_contract.DEFAULT_PARAMETERS.minimum_segment_seconds
                ),
                "fine_search_radius_requested_ms": (
                    audio_contract.DEFAULT_PARAMETERS.search_radius_ms
                ),
                "fine_search_radius_realized_ms": (
                    math.ceil(
                        audio_contract.DEFAULT_PARAMETERS.search_radius_ms
                        / FINE_ENVELOPE_HOP_MS
                    )
                    * FINE_ENVELOPE_HOP_MS
                ),
                "fine_peak_exclusion_requested_ms": (
                    audio_contract.ENVELOPE_HOP_MS
                ),
                "fine_peak_exclusion_realized_ms": (
                    math.ceil(
                        audio_contract.ENVELOPE_HOP_MS / FINE_ENVELOPE_HOP_MS
                    )
                    * FINE_ENVELOPE_HOP_MS
                ),
            },
            "identity_attempt": identity_evidence,
            "selected_32ms": selected_audit,
            "channel_decisions": final_decisions,
            "all_channels_authorized": formal_pass,
        },
        "gates": {
            "published_audio_contract_full_replay_pass": True,
            "sealed_pose_base_full_replay_pass": True,
            "deterministic_mapping_rule_pass": True,
            "formal_window_all_coarse_pass": all(
                decision["coarse_32ms"]["quality_pass"] is True
                for decision in final_decisions.values()
            ),
            "formal_window_hierarchical_sync_pass": formal_pass,
            "resolution_override_used": any(
                decision["status"] == "resolution_confirmed"
                for decision in final_decisions.values()
            ),
            "alignment_prerequisite_pass": formal_pass,
            "counted_gold_ready": False,
            "counted_gold_blocker": (
                "time alignment alone is not transcript-conditioned gold-VAD validation"
            ),
        },
    }
    _verify_bound_sources(report, authority=authority)
    return report


def audit_audio_gold_alignment(scene: str) -> dict[str, Any]:
    """Fully replay one production scene without writing an artifact."""

    return _audit_scene_with_authority(scene, authority=default_gold_alignment_authority())


def gold_alignment_window(scene: str) -> GoldAlignmentWindow:
    """Return the fully replayed Decimal/tick consumer contract for one scene."""

    report = audit_audio_gold_alignment(scene)
    if report["gates"]["alignment_prerequisite_pass"] is not True:
        _fail("scene has a deterministic map but failed formal-window sync; use is withheld")
    ticks = report["authorization"]["authorized_tick_range"]
    channels = report["mapping_contract"]["channels"]
    return GoldAlignmentWindow(
        scene=scene,
        start_tick_inclusive=ticks["start_inclusive"],
        end_tick_exclusive=ticks["end_exclusive"],
        correction_s={
            speaker: _decimal_string(channels[speaker]["correction_s"], "correction_s")
            for speaker in SPEAKERS
        },
    )


def _validate_report_shape(report: dict[str, Any]) -> None:
    expected = {
        "artifact_kind",
        "format_version",
        "scene",
        "output_role",
        "counted_gold_ready",
        "trust_model",
        "sources",
        "mapping_contract",
        "authorization",
        "formal_window_replay",
        "gates",
    }
    if type(report) is not dict or set(report) != expected:
        _fail("gold-alignment artifact has unsupported top-level keys")
    if (
        report["artifact_kind"] != ARTIFACT_KIND
        or type(report["format_version"]) is not int
        or report["format_version"] != FORMAT_VERSION
    ):
        _fail("gold-alignment artifact kind or version is unsupported")
    if report["scene"] not in audio_contract.ACCEPTED_SCENES:
        _fail("gold-alignment artifact scene is outside the frozen authority")
    if report["output_role"] != OUTPUT_ROLE or report["trust_model"] != TRUST_MODEL:
        _fail("gold-alignment artifact role or trust model is unsupported")
    if report["counted_gold_ready"] is not False:
        _fail("a channel time map can never claim counted-gold readiness")
    mapping = report.get("mapping_contract")
    if type(mapping) is not dict or mapping.get("linear_or_manual_maps_allowed") is not False:
        _fail("linear or manually supplied channel maps are forbidden")
    gates = report.get("gates")
    if type(gates) is not dict or gates.get("counted_gold_ready") is not False:
        _fail("gold-alignment gates must preserve counted_gold_ready=false")


def _verify_one_binding(
    binding: Any,
    expected: Path,
    label: str,
    *,
    artifact_kind: str,
    format_version: int,
) -> None:
    if type(binding) is not dict or set(binding) != {
        "path",
        "size_bytes",
        "sha256",
        "artifact_kind",
        "format_version",
    }:
        _fail(f"{label} binding is malformed")
    if type(binding["path"]) is not str or Path(binding["path"]).resolve() != expected:
        _fail(f"{label} binding does not name its code-owned canonical path")
    if type(binding["size_bytes"]) is not int or binding["size_bytes"] < 0:
        _fail(f"{label} size must be an exact non-negative integer")
    if type(binding["sha256"]) is not str or not _SHA256_RE.fullmatch(binding["sha256"]):
        _fail(f"{label} SHA-256 is malformed")
    if binding["artifact_kind"] != artifact_kind:
        _fail(f"{label} artifact_kind is unsupported")
    if type(binding["format_version"]) is not int or binding["format_version"] != format_version:
        _fail(f"{label} format_version is unsupported")
    if not expected.is_file():
        _fail(f"{label} canonical source disappeared")
    if expected.stat().st_size != binding["size_bytes"]:
        _fail(f"{label} source size changed")
    if _sha256_file(expected) != binding["sha256"]:
        _fail(f"{label} source SHA-256 changed")


def _verify_bound_sources(report: dict[str, Any], *, authority: GoldAlignmentAuthority) -> None:
    scene = report.get("scene")
    if type(scene) is not str:
        _fail("gold-alignment scene must be a string")
    audio_path, pose_csv, pose_sidecar = _canonical_paths(scene, authority)
    sources = report.get("sources")
    if type(sources) is not dict or set(sources) != {
        "audio_contract",
        "pose_csv",
        "pose_provenance",
    }:
        _fail("gold-alignment sources must bind exactly audio contract and pose base pair")
    _verify_one_binding(
        sources["audio_contract"],
        audio_path,
        "audio_contract",
        artifact_kind=audio_contract.ARTIFACT_KIND,
        format_version=audio_contract.FORMAT_VERSION,
    )
    _verify_one_binding(
        sources["pose_csv"],
        pose_csv,
        "pose_csv",
        artifact_kind=assembly.REAL_INTAKE_ARTIFACT_KIND,
        format_version=assembly.REAL_PROVENANCE_FORMAT_VERSION,
    )
    _verify_one_binding(
        sources["pose_provenance"],
        pose_sidecar,
        "pose_provenance",
        artifact_kind=assembly.REAL_INTAKE_ARTIFACT_KIND,
        format_version=assembly.REAL_PROVENANCE_FORMAT_VERSION,
    )


def _load_with_authority(path: str | Path, *, authority: GoldAlignmentAuthority) -> dict[str, Any]:
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        _fail(f"gold-alignment artifact does not exist: {artifact_path}")
    report = _strict_json_loads(artifact_path.read_bytes(), "gold-alignment artifact")
    _validate_report_shape(report)
    _verify_bound_sources(report, authority=authority)
    replayed = _audit_scene_with_authority(report["scene"], authority=authority)
    if not _type_exact_json_equal(report, replayed):
        _fail("gold-alignment artifact does not exactly match a fresh full replay")
    _verify_bound_sources(report, authority=authority)
    return report


def load_audio_gold_alignment(path: str | Path) -> dict[str, Any]:
    """Load an artifact only after full audio/pose/source/time-map replay."""

    return _load_with_authority(path, authority=default_gold_alignment_authority())


def _publish_no_clobber(staged: Path, destination: Path) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise AudioGoldAlignmentError(
            f"refusing to overwrite existing gold-alignment artifact: {destination}"
        ) from exc


def _write_with_authority(
    scene: str,
    output: str | Path,
    *,
    authority: GoldAlignmentAuthority,
) -> Path:
    destination = Path(output).resolve()
    if destination.exists():
        _fail(f"refusing to overwrite existing gold-alignment artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = _audit_scene_with_authority(scene, authority=authority)
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
            payload = (
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_bound_sources(report, authority=authority)
        _publish_no_clobber(temporary, destination)
        _load_with_authority(destination, authority=authority)
        return destination
    except BaseException:
        # ``os.link`` may have created the destination before an injected
        # KeyboardInterrupt/SystemExit is raised back to us.  Conversely, a
        # concurrent owner may replace the destination while the post-publish
        # replay is running.  The staged hard link remains our ownership token:
        # remove the destination only while it still names that exact inode.
        if temporary is not None and temporary.exists() and destination.exists():
            try:
                if os.path.samefile(temporary, destination):
                    destination.unlink()
            except (FileNotFoundError, OSError):
                # A concurrent path change means ownership cannot be proven;
                # preserving the path is the only safe fail-closed action.
                pass
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_audio_gold_alignment(scene: str, output: str | Path) -> Path:
    """Publish one fully replayed no-clobber artifact at an explicit path."""

    return _write_with_authority(
        scene,
        output,
        authority=default_gold_alignment_authority(),
    )


def validate_authorized_tick_range(
    artifact_path: str | Path, start_inclusive: int, end_exclusive: int
) -> dict[str, int]:
    """Fail closed unless a requested 25-Hz range is inside the sealed window."""

    if type(start_inclusive) is not int or type(end_exclusive) is not int:
        _fail("requested tick bounds must be exact integers")
    if end_exclusive <= start_inclusive:
        _fail("requested tick range must be a positive half-open interval")
    report = load_audio_gold_alignment(artifact_path)
    if report["gates"]["alignment_prerequisite_pass"] is not True:
        _fail("gold-audio use is withheld because formal-window synchronization failed")
    authorized = report["authorization"]["authorized_tick_range"]
    if (
        start_inclusive < authorized["start_inclusive"]
        or end_exclusive > authorized["end_exclusive"]
    ):
        _fail("requested ticks exceed the sealed nonpadding gold-audio window")
    return {
        "frame_rate_hz": FRAME_RATE_HZ,
        "start_inclusive": start_inclusive,
        "end_exclusive": end_exclusive,
    }
