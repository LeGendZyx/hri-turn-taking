"""Fail-closed source and synchronization audit for the real audio corpus.

The contract in this module is intentionally narrower than a gold-VAD contract.
It binds the immutable audio/timing sources, proves that ``master_aligned.wav``
is an exact permutation of the four named mono tracks, and measures segmented
timing residuals.  It never edits audio and can never declare counted gold VAD
ready.

Production entry points derive every source path from a code-owned authority.
Manifest paths therefore cannot make copied or renamed files authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import wave
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.data.source_paths import corpus_root, hpe_delivery_root

ARTIFACT_KIND = "vap_audio_source_sync_contract"
FORMAT_VERSION = 1
OUTPUT_ROLE = "audio_source_and_sync_quality_evidence"
SPEAKERS = ("A", "B", "C", "R")
ACCEPTED_SCENES = (
    "03_20__4",
    "03_26__3",
    "05_14__2",
    "05_14__3",
    "05_14__4",
    "05_15__2",
    "05_15__3",
    "05_15__4",
    "05_15__5",
    "05_15__6",
)

ALIGNED_RATE_HZ = 48_000
REFERENCE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2
ENVELOPE_HOP_SAMPLES_16K = 512
ENVELOPE_HOP_MS = 32
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_RE = re.compile(r"^[0-9]{2}_[0-9]{2}(?:_[0-9]+)?__[0-9]+$")


class AudioContractError(ValueError):
    """Raised when source identity or structural evidence is not trustworthy."""


@dataclass(frozen=True)
class AudioSourceAuthority:
    """Caller-owned roots; manifest content is never allowed to change them."""

    corpus_root: Path
    hpe_delivery_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_root", Path(self.corpus_root).resolve())
        object.__setattr__(self, "hpe_delivery_root", Path(self.hpe_delivery_root).resolve())


@dataclass(frozen=True)
class AlignmentAuditParameters:
    """Frozen v1 quality thresholds and segmentation parameters."""

    segment_seconds: int = 30
    minimum_segment_seconds: int = 12
    search_radius_ms: int = 256
    peak_exclusion_hops: int = 1
    minimum_peak_correlation: float = 0.20
    minimum_peak_margin: float = 0.025
    minimum_unique_fraction: float = 0.75
    minimum_usable_segments: int = 3
    maximum_abs_median_residual_ms: float = 24.0
    maximum_abs_p95_residual_ms: float = 40.0
    maximum_drift_span_ms: float = 40.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "envelope": "nonoverlapping_rms_hop_then_log",
            "envelope_hop_samples_16khz": ENVELOPE_HOP_SAMPLES_16K,
            "envelope_hop_ms": ENVELOPE_HOP_MS,
            "segment_seconds": self.segment_seconds,
            "minimum_segment_seconds": self.minimum_segment_seconds,
            "search_radius_ms": self.search_radius_ms,
            "peak_exclusion_hops": self.peak_exclusion_hops,
            "minimum_peak_correlation": self.minimum_peak_correlation,
            "minimum_peak_margin": self.minimum_peak_margin,
            "minimum_unique_fraction": self.minimum_unique_fraction,
            "minimum_usable_segments": self.minimum_usable_segments,
            "maximum_abs_median_residual_ms": self.maximum_abs_median_residual_ms,
            "maximum_abs_p95_residual_ms": self.maximum_abs_p95_residual_ms,
            "maximum_drift_span_ms": self.maximum_drift_span_ms,
        }


DEFAULT_PARAMETERS = AlignmentAuditParameters()


def default_source_authority() -> AudioSourceAuthority:
    project_root = Path(__file__).resolve().parents[3]
    return AudioSourceAuthority(
        corpus_root=corpus_root(project_root),
        hpe_delivery_root=hpe_delivery_root(project_root),
    )


def _fail(message: str) -> None:
    raise AudioContractError(message)


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AudioContractError(f"{label} is not strict UTF-8") from exc

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
        raise AudioContractError(
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


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be a non-empty string without surrounding whitespace")
    return value


def _require_exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact JSON integer (not bool or float)")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _require_number(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a JSON number (not bool)")
    if not math.isfinite(float(value)):
        _fail(f"{label} must be finite")
    return Decimal(str(value))


def _require_decimal_string(value: Any, label: str) -> Decimal:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise AudioContractError(f"{label} must be a decimal string") from exc
    if not result.is_finite():
        _fail(f"{label} must be finite")
    return result


def _scene_parts(scene: str) -> tuple[str, str]:
    if type(scene) is not str or _SCENE_RE.fullmatch(scene) is None:
        _fail("scene must use the canonical group__scenario spelling")
    if scene not in ACCEPTED_SCENES:
        _fail(f"scene {scene!r} is outside the sealed ten-scene dissertation set")
    return scene.rsplit("__", maxsplit=1)


def canonical_source_paths(scene: str, authority: AudioSourceAuthority) -> dict[str, Path]:
    """Return the only accepted source paths for one scene."""

    group, scenario = _scene_parts(scene)
    aligned = authority.corpus_root / group / "Processed" / scenario / "audio" / "aligned_audio"
    transcript = authority.corpus_root / group / "transcript" / scenario
    delivery = authority.hpe_delivery_root / scene
    paths = {speaker: aligned / f"{speaker}_aligned.wav" for speaker in SPEAKERS}
    paths.update(
        {
            "master": aligned / "master_aligned.wav",
            "reference": (
                authority.corpus_root
                / group
                / "Processed"
                / scenario
                / "video"
                / "participants_rgb_audio_edited.wav"
            ),
            "alignment_report": aligned / "offset_report.json",
            "transcript_offset": transcript / "offset.json",
            "hpe_offset": delivery / "offset.json",
            "hpe_offset_quality": delivery / "offset_quality.json",
        }
    )
    return {name: path.resolve() for name, path in paths.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wave_header(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            return {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate_hz": handle.getframerate(),
                "n_frames": handle.getnframes(),
                "compression_type": handle.getcomptype(),
            }
    except (wave.Error, EOFError) as exc:
        raise AudioContractError(f"invalid WAV container: {path}") from exc


def _verify_complete_pcm_payload(
    path: Path, *, expected_header: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Read every declared PCM frame and reject short or partial payloads."""

    try:
        with wave.open(str(path), "rb") as handle:
            header = {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate_hz": handle.getframerate(),
                "n_frames": handle.getnframes(),
                "compression_type": handle.getcomptype(),
            }
            if expected_header is not None and header != expected_header:
                _fail(f"{path.name} WAV header changed before PCM payload replay")
            frame_size = header["channels"] * header["sample_width_bytes"]
            if frame_size <= 0:
                _fail(f"{path.name} declares an invalid PCM frame size")
            declared_frames = header["n_frames"]
            decoded_frames = 0
            while decoded_frames < declared_frames:
                requested = min(65_536, declared_frames - decoded_frames)
                raw = handle.readframes(requested)
                if len(raw) % frame_size:
                    _fail(f"{path.name} ends with a partial, non-frame-aligned PCM block")
                block_frames = len(raw) // frame_size
                if block_frames == 0:
                    break
                decoded_frames += block_frames
            if decoded_frames != declared_frames:
                _fail(
                    f"{path.name} PCM payload ended before declared n_frames "
                    f"({decoded_frames} != {declared_frames})"
                )
            extra = handle.readframes(1)
            if len(extra) % frame_size:
                _fail(f"{path.name} has a trailing partial PCM frame")
            if extra:
                _fail(f"{path.name} PCM payload exceeds declared n_frames")
    except (wave.Error, EOFError) as exc:
        raise AudioContractError(f"invalid or truncated WAV payload: {path}") from exc
    return {
        "frame_size_bytes": frame_size,
        "decoded_frames": decoded_frames,
        "frame_aligned": True,
        "complete": True,
    }


def _source_binding(path: Path, *, is_wave: bool) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"required canonical source does not exist: {path}")
    stat = path.stat()
    header = _wave_header(path) if is_wave else None
    payload = _verify_complete_pcm_payload(path, expected_header=header) if is_wave else None
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(path),
        "media_type": "audio/wav" if is_wave else "application/json",
        "wave_header": header,
        "pcm_payload": payload,
    }


def _validate_wave_headers(bindings: dict[str, dict[str, Any]]) -> None:
    for speaker in SPEAKERS:
        header = bindings[speaker]["wave_header"]
        expected = {
            "channels": 1,
            "sample_width_bytes": SAMPLE_WIDTH_BYTES,
            "sample_rate_hz": ALIGNED_RATE_HZ,
            "compression_type": "NONE",
        }
        for key, value in expected.items():
            if header[key] != value:
                _fail(f"{speaker}_aligned.wav {key} must equal {value!r}")
    mono_frames = {bindings[speaker]["wave_header"]["n_frames"] for speaker in SPEAKERS}
    if len(mono_frames) != 1:
        _fail("A/B/C/R aligned WAVs must have exactly equal frame counts")
    master = bindings["master"]["wave_header"]
    expected_master = {
        "channels": 4,
        "sample_width_bytes": SAMPLE_WIDTH_BYTES,
        "sample_rate_hz": ALIGNED_RATE_HZ,
        "compression_type": "NONE",
    }
    for key, value in expected_master.items():
        if master[key] != value:
            _fail(f"master_aligned.wav {key} must equal {value!r}")
    if master["n_frames"] not in mono_frames:
        _fail("master_aligned.wav frame count must equal every named mono track")
    reference = bindings["reference"]["wave_header"]
    expected_reference = {
        "channels": 1,
        "sample_width_bytes": SAMPLE_WIDTH_BYTES,
        "sample_rate_hz": REFERENCE_RATE_HZ,
        "compression_type": "NONE",
    }
    for key, value in expected_reference.items():
        if reference[key] != value:
            _fail(f"reference WAV {key} must equal {value!r}")
    if reference["n_frames"] <= 0:
        _fail("reference WAV must contain at least one frame")


def _discover_master_permutation(paths: dict[str, Path]) -> tuple[str, ...]:
    """Compare PCM samples in bounded chunks and discover channel ownership."""

    candidates = [set(SPEAKERS) for _ in SPEAKERS]
    block_frames = 65_536
    with ExitStack() as stack:
        master = stack.enter_context(wave.open(str(paths["master"]), "rb"))
        monos = {
            speaker: stack.enter_context(wave.open(str(paths[speaker]), "rb"))
            for speaker in SPEAKERS
        }
        consumed = 0
        while consumed < master.getnframes():
            count = min(block_frames, master.getnframes() - consumed)
            raw_master = master.readframes(count)
            if len(raw_master) % (4 * SAMPLE_WIDTH_BYTES):
                _fail("master WAV yielded a partial, non-frame-aligned PCM block")
            master_values = np.frombuffer(raw_master, dtype="<i2")
            if master_values.size != count * 4:
                _fail("master WAV ended before its declared frame count")
            master_values = master_values.reshape(count, 4)
            mono_values: dict[str, np.ndarray] = {}
            for speaker, handle in monos.items():
                raw = handle.readframes(count)
                if len(raw) % SAMPLE_WIDTH_BYTES:
                    _fail(f"{speaker} WAV yielded a partial, non-frame-aligned PCM block")
                values = np.frombuffer(raw, dtype="<i2")
                if values.size != count:
                    _fail(f"{speaker} WAV ended before its declared frame count")
                mono_values[speaker] = values
            for channel in range(4):
                candidates[channel] = {
                    speaker
                    for speaker in candidates[channel]
                    if np.array_equal(master_values[:, channel], mono_values[speaker])
                }
            consumed += count
    if any(len(candidate) != 1 for candidate in candidates):
        _fail(
            "master channels are not a unique exact sample-wise permutation of A/B/C/R "
            f"(candidates={[sorted(value) for value in candidates]})"
        )
    order = tuple(next(iter(candidate)) for candidate in candidates)
    if set(order) != set(SPEAKERS):
        _fail("master contains a duplicated or missing named mono track")
    return order


def _hop_rms_envelope(path: Path, *, mixdown: bool) -> np.ndarray:
    """Return a memory-bounded 32-ms RMS envelope at the WAV's native rate."""

    values: list[np.ndarray] = []
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_size = channels * sample_width
        declared_frames = handle.getnframes()
        decoded_frames = 0
        hop_numerator = rate * ENVELOPE_HOP_SAMPLES_16K
        if hop_numerator % REFERENCE_RATE_HZ:
            _fail(f"{path.name} rate cannot express the frozen 32-ms hop exactly")
        hop = hop_numerator // REFERENCE_RATE_HZ
        carry = np.empty(0, dtype=np.float64)
        while True:
            raw = handle.readframes(65_536)
            if not raw:
                break
            if len(raw) % frame_size:
                _fail(f"{path.name} yielded a partial, non-frame-aligned PCM block")
            decoded_frames += len(raw) // frame_size
            block = np.frombuffer(raw, dtype="<i2")
            if block.size % channels:
                _fail(f"{path.name} has a truncated PCM frame")
            block = block.reshape(-1, channels).astype(np.float64)
            if channels == 1:
                signal = block[:, 0]
            elif mixdown:
                signal = block.mean(axis=1)
            else:
                _fail(f"{path.name} requires an explicit mixdown rule")
            signal /= 32768.0
            combined = np.concatenate((carry, signal)) if carry.size else signal
            complete = combined.size // hop
            if complete:
                framed = combined[: complete * hop].reshape(complete, hop)
                values.append(np.sqrt(np.mean(framed * framed, axis=1)))
            carry = combined[complete * hop :].copy()
        if decoded_frames != declared_frames:
            _fail(
                f"{path.name} PCM payload ended before declared n_frames "
                f"({decoded_frames} != {declared_frames})"
            )
    if not values:
        _fail(f"{path.name} is too short for one 32-ms envelope hop")
    rms = np.concatenate(values)
    return np.log(np.maximum(rms, 1e-8))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - float(left.mean())
    right = right - float(right.mean())
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    slopes: list[float] = []
    for index in range(len(x)):
        dx = x[index + 1 :] - x[index]
        dy = y[index + 1 :] - y[index]
        slopes.extend((dy / dx)[dx != 0].tolist())
    return float(np.median(slopes)) if slopes else 0.0


def _segmented_alignment_audit(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    declared_lag_hops: int,
    parameters: AlignmentAuditParameters,
) -> dict[str, Any]:
    """Measure local lag after a declared offset, including ambiguity and drift."""

    segment_frames = max(1, round(parameters.segment_seconds * 1000 / ENVELOPE_HOP_MS))
    minimum_frames = max(1, round(parameters.minimum_segment_seconds * 1000 / ENVELOPE_HOP_MS))
    search_hops = max(1, parameters.search_radius_ms // ENVELOPE_HOP_MS)
    lower = max(0, -declared_lag_hops + search_hops)
    upper = min(len(reference), len(target) - declared_lag_hops - search_hops)
    segments: list[dict[str, Any]] = []
    start = lower
    while start < upper:
        stop = min(start + segment_frames, upper)
        if stop - start < minimum_frames:
            break
        left = reference[start:stop]
        scores: dict[int, float] = {}
        for residual in range(-search_hops, search_hops + 1):
            target_start = start + declared_lag_hops + residual
            target_stop = stop + declared_lag_hops + residual
            scores[residual] = _correlation(left, target[target_start:target_stop])
        finite = {lag: score for lag, score in scores.items() if math.isfinite(score)}
        center_s = ((start + stop) / 2) * ENVELOPE_HOP_MS / 1000
        if not finite:
            segments.append(
                {
                    "center_s": round(center_s, 6),
                    "status": "degenerate_envelope",
                    "residual_ms": None,
                    "peak_correlation": None,
                    "second_peak_correlation": None,
                    "peak_margin": None,
                    "unique_peak": False,
                }
            )
            start = stop
            continue
        best_lag = max(finite, key=finite.get)
        best_score = finite[best_lag]
        other_scores = [
            score
            for lag, score in finite.items()
            if abs(lag - best_lag) > parameters.peak_exclusion_hops
        ]
        second = max(other_scores) if other_scores else -1.0
        refined_lag = float(best_lag)
        if best_lag - 1 in finite and best_lag + 1 in finite:
            before = finite[best_lag - 1]
            after = finite[best_lag + 1]
            curvature = before - (2 * best_score) + after
            if abs(curvature) > 1e-12:
                delta = 0.5 * (before - after) / curvature
                refined_lag += float(np.clip(delta, -0.5, 0.5))
        margin = best_score - second
        unique = (
            best_score >= parameters.minimum_peak_correlation
            and margin >= parameters.minimum_peak_margin
        )
        segments.append(
            {
                "center_s": round(center_s, 6),
                "status": "usable",
                "residual_ms": round(refined_lag * ENVELOPE_HOP_MS, 6),
                "peak_correlation": round(best_score, 9),
                "second_peak_correlation": round(second, 9),
                "peak_margin": round(margin, 9),
                "unique_peak": unique,
            }
        )
        start = stop

    usable = [segment for segment in segments if segment["status"] == "usable"]
    residuals = np.asarray([segment["residual_ms"] for segment in usable], dtype=float)
    centers = np.asarray([segment["center_s"] for segment in usable], dtype=float)
    if residuals.size:
        median = float(np.median(residuals))
        p95 = float(np.quantile(np.abs(residuals), 0.95))
        slope_ms_per_s = _theil_sen_slope(centers, residuals)
        drift_span = (
            slope_ms_per_s * float(centers.max() - centers.min()) if centers.size > 1 else 0.0
        )
        unique_fraction = float(np.mean([bool(segment["unique_peak"]) for segment in usable]))
        median_corr = float(np.median([segment["peak_correlation"] for segment in usable]))
    else:
        median = p95 = slope_ms_per_s = drift_span = unique_fraction = median_corr = float("nan")
    reasons: list[str] = []
    if len(usable) < parameters.minimum_usable_segments:
        reasons.append("insufficient_usable_segments")
    if not math.isfinite(median) or abs(median) > parameters.maximum_abs_median_residual_ms:
        reasons.append("median_residual_exceeds_limit")
    if not math.isfinite(p95) or p95 > parameters.maximum_abs_p95_residual_ms:
        reasons.append("p95_residual_exceeds_limit")
    if not math.isfinite(drift_span) or abs(drift_span) > parameters.maximum_drift_span_ms:
        reasons.append("drift_span_exceeds_limit")
    if not math.isfinite(unique_fraction) or unique_fraction < parameters.minimum_unique_fraction:
        reasons.append("peak_uniqueness_below_limit")
    if not math.isfinite(median_corr) or median_corr < parameters.minimum_peak_correlation:
        reasons.append("correlation_below_limit")
    return {
        "declared_lag_hops": declared_lag_hops,
        "declared_lag_ms": declared_lag_hops * ENVELOPE_HOP_MS,
        "n_segments": len(segments),
        "n_usable_segments": len(usable),
        "median_residual_ms": round(median, 6) if math.isfinite(median) else None,
        "p95_abs_residual_ms": round(p95, 6) if math.isfinite(p95) else None,
        "drift_ms_per_s": (round(slope_ms_per_s, 9) if math.isfinite(slope_ms_per_s) else None),
        "drift_span_ms": round(drift_span, 6) if math.isfinite(drift_span) else None,
        "unique_peak_fraction": (
            round(unique_fraction, 6) if math.isfinite(unique_fraction) else None
        ),
        "median_peak_correlation": (round(median_corr, 9) if math.isfinite(median_corr) else None),
        "quality_pass": not reasons,
        "failure_reasons": reasons,
        "segments": segments,
    }


def _parse_transcript_offset(
    raw: bytes, scene: str, reference_header: dict[str, Any]
) -> dict[str, Any]:
    value = _require_object(_strict_json_loads(raw, "transcript offset"), "transcript offset")
    _require_exact_keys(
        value,
        {
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
        },
        "transcript offset",
    )
    group, scenario = _scene_parts(scene)
    if value.get("session") != scenario:
        _fail("transcript offset session does not match the requested scene")
    if value.get("timeline_reference") != "participants_rgb_audio_edited.wav":
        _fail("transcript offset has an unexpected reference WAV")
    if value.get("frame_reference") != "participants_rgb_audio_edited.mp4":
        _fail("transcript offset has an unexpected reference video")
    fps = value.get("fps")
    if type(fps) is not float or not math.isfinite(fps) or fps != 30.0:
        _fail("legacy transcript offset fps must be the exact JSON float 30.0")
    processing_rate = _require_exact_int(value.get("processing_sr"), "processing_sr")
    if processing_rate != REFERENCE_RATE_HZ:
        _fail("transcript offset processing_sr must be exact integer 16000")
    if _require_exact_int(value.get("master_sr"), "master_sr") != REFERENCE_RATE_HZ:
        _fail("transcript offset master_sr must describe the 16-kHz processing copy")
    if _require_exact_int(value.get("reference_sr"), "reference_sr") != REFERENCE_RATE_HZ:
        _fail("transcript offset reference_sr must be 16000")
    lag = _require_exact_int(value.get("lag_frames_rms"), "lag_frames_rms")
    shift = _require_exact_int(
        value.get("offset_samples_processing_sr"), "offset_samples_processing_sr"
    )
    offset = _require_number(value.get("offset_sec"), "offset_sec")
    expected = Decimal(lag * ENVELOPE_HOP_SAMPLES_16K) / Decimal(REFERENCE_RATE_HZ)
    if offset != expected or shift != lag * ENVELOPE_HOP_SAMPLES_16K:
        _fail("transcript offset seconds/samples/lag are not exactly self-consistent")
    duration = _require_number(value.get("reference_duration_sec"), "reference_duration_sec")
    actual_duration = Decimal(reference_header["n_frames"]) / Decimal(REFERENCE_RATE_HZ)
    if abs(duration - actual_duration) > Decimal(1) / Decimal(REFERENCE_RATE_HZ):
        _fail("transcript offset reference duration disagrees with the bound WAV header")
    alignment = _require_object(value.get("speaker_alignment"), "speaker_alignment")
    if set(alignment) != set(SPEAKERS):
        _fail("transcript offset speaker_alignment must contain exactly A/B/C/R")
    interpretation = _require_object(value.get("interpretation"), "interpretation")
    if interpretation != {
        "offset_positive": "crop beginning of A/B/C/R aligned wavs",
        "offset_negative": "pad silence at beginning of A/B/C/R aligned wavs",
    }:
        _fail("transcript offset interpretation does not match the upstream generator")
    for speaker in SPEAKERS:
        item = _require_object(alignment[speaker], f"speaker_alignment.{speaker}")
        _require_exact_keys(
            item,
            {
                "src",
                "dst",
                "sr",
                "offset_sec",
                "shift_samples",
                "action",
                "output_len_samples",
                "output_duration_sec",
            },
            f"speaker_alignment.{speaker}",
        )
        expected_src = f"{group}/{scenario}/audio/aligned_audio/{speaker}_aligned.wav"
        if item.get("src") != expected_src:
            _fail(f"speaker_alignment.{speaker}.src does not match the scene")
        expected_dst = f"{group}/{scenario}/audio/aligned_audio/{speaker}_edited.wav"
        if item.get("dst") != expected_dst:
            _fail(f"speaker_alignment.{speaker}.dst does not match the scene")
        if _require_number(item.get("offset_sec"), f"{speaker}.offset_sec") != offset:
            _fail(f"speaker_alignment.{speaker}.offset_sec is inconsistent")
        if _require_exact_int(item.get("shift_samples"), f"{speaker}.shift_samples") != shift:
            _fail(f"speaker_alignment.{speaker}.shift_samples is inconsistent")
        if _require_exact_int(item.get("sr"), f"{speaker}.sr") != REFERENCE_RATE_HZ:
            _fail(f"speaker_alignment.{speaker}.sr must be exact integer 16000")
        expected_action = (
            f"crop {shift} samples"
            if shift > 0
            else f"pad {-shift} samples"
            if shift < 0
            else "no shift"
        )
        if item.get("action") != expected_action:
            _fail(f"speaker_alignment.{speaker}.action is inconsistent")
        output_length = _require_exact_int(
            item.get("output_len_samples"), f"{speaker}.output_len_samples"
        )
        if output_length != reference_header["n_frames"]:
            _fail(f"speaker_alignment.{speaker}.output length is inconsistent")
        output_duration = _require_number(
            item.get("output_duration_sec"), f"{speaker}.output_duration_sec"
        )
        if abs(output_duration - actual_duration) > Decimal(1) / Decimal(REFERENCE_RATE_HZ):
            _fail(f"speaker_alignment.{speaker}.output duration is inconsistent")
    return {
        "offset_sec": str(offset),
        "lag_hops": lag,
        "processing_rate_hz": processing_rate,
        "reference_duration_s": str(actual_duration),
        "required_shared_coverage": {
            "start_s": str(offset),
            "end_s_exclusive": str(offset + actual_duration),
        },
    }


def _parse_hpe_anchor(
    offset_raw: bytes,
    quality_raw: bytes,
    *,
    scene: str,
    a_binding: dict[str, Any],
) -> dict[str, Any]:
    offset = _require_object(_strict_json_loads(offset_raw, "HPE offset"), "HPE offset")
    streams = _require_object(offset.get("streams"), "HPE offset.streams")
    if "mic_A" not in streams or _require_number(streams["mic_A"], "mic_A offset") != 0:
        _fail("HPE offset must define mic_A as the exact zero anchor")
    if "6cam" not in streams:
        _fail("HPE offset must contain the accepted 6cam bridge")
    _require_number(streams["6cam"], "HPE 6cam offset")
    quality = _require_object(
        _strict_json_loads(quality_raw, "HPE offset quality"), "HPE offset quality"
    )
    if quality.get("convention") != "shared_time = stream_local_time + offset_s":
        _fail("HPE timing convention is not the shared mic-A formula")
    keys = _require_object(quality.get("timebase_keys"), "HPE timebase_keys")
    mic_key = _require_string(keys.get("mic_A"), "HPE timebase_keys.mic_A")
    if not mic_key.startswith("A_aligned.wav"):
        _fail("HPE mic_A timebase key does not name A_aligned.wav")
    _require_string(keys.get("6cam"), "HPE timebase_keys.6cam")
    quality_map = _require_object(quality.get("quality"), "HPE quality")
    mic = _require_object(quality_map.get("mic_A"), "HPE quality.mic_A")
    if _require_number(mic.get("offset_s"), "HPE mic_A quality offset") != 0:
        _fail("HPE mic_A quality offset must be zero")
    if _require_number(mic.get("error_bound_ms"), "HPE mic_A error bound") != 0:
        _fail("HPE mic_A anchor error bound must be zero by definition")
    if mic.get("status") != "ok":
        _fail("HPE mic_A anchor status must be ok")
    result: dict[str, Any] = {
        "anchor": "mic_A",
        "offset_s": "0",
        "canonical_wav": a_binding,
        "upstream_exact_anchor_binding": False,
        "upstream_evidence_strength": "semantic_anchor_only",
    }
    bundle = quality.get("rgb_d3_source_bundle")
    if bundle is None:
        return result
    bundle = _require_object(bundle, "HPE rgb_d3_source_bundle")
    _require_string(bundle.get("gate_version"), "HPE source bundle gate_version")
    if bundle.get("delivery") != scene:
        _fail("HPE source bundle delivery does not match the scene")
    anchor = _require_object(bundle.get("anchor"), "HPE rgb_d3_source_bundle.anchor")
    _require_exact_keys(
        anchor,
        {"path", "sha256", "sample_rate_hz", "samples", "duration_s"},
        "HPE rgb_d3_source_bundle.anchor",
    )
    if (
        Path(_require_string(anchor.get("path"), "HPE anchor path")).resolve()
        != Path(a_binding["path"]).resolve()
    ):
        _fail("HPE source bundle anchor path is not this scene's canonical A WAV")
    if anchor.get("sha256") != a_binding["sha256"]:
        _fail("HPE source bundle anchor hash disagrees with the current A WAV")
    header = a_binding["wave_header"]
    if (
        _require_exact_int(anchor.get("sample_rate_hz"), "HPE anchor sample rate")
        != header["sample_rate_hz"]
    ):
        _fail("HPE source bundle anchor sample rate mismatch")
    if _require_exact_int(anchor.get("samples"), "HPE anchor samples") != header["n_frames"]:
        _fail("HPE source bundle anchor frame count mismatch")
    anchor_duration = _require_number(anchor.get("duration_s"), "HPE anchor duration")
    actual_duration = Decimal(header["n_frames"]) / Decimal(header["sample_rate_hz"])
    if abs(anchor_duration - actual_duration) > Decimal(1) / Decimal(header["sample_rate_hz"]):
        _fail("HPE source bundle anchor duration mismatch")
    result["upstream_exact_anchor_binding"] = True
    result["upstream_evidence_strength"] = "exact_path_hash_header"
    return result


def _declared_channel_order(raw: bytes) -> tuple[str, ...] | None:
    report = _require_object(
        _strict_json_loads(raw, "upstream alignment report"),
        "upstream alignment report",
    )
    declared_rate = _require_exact_int(report.get("sample_rate"), "alignment report sample_rate")
    if declared_rate != ALIGNED_RATE_HZ:
        _fail("alignment report sample rate must be exact integer 48000")
    channels = report.get("channels")
    if channels is None:
        allowed_legacy = (
            {"sample_rate", "offset_B", "offset_C", "offset_R"},
            {"sample_rate", "offset_A", "offset_B", "offset_C"},
        )
        if set(report) not in allowed_legacy:
            _fail("legacy alignment report does not match a known exact schema")
        for key, value in report.items():
            if key != "sample_rate":
                _require_exact_int(value, f"alignment report {key}")
        return None
    _require_exact_keys(report, {"offsets", "sample_rate", "channels"}, "alignment report")
    offsets = _require_object(report["offsets"], "alignment report.offsets")
    if set(offsets) != {"A", "B", "C"}:
        _fail("alignment report offsets must contain exactly A/B/C")
    for speaker, value in offsets.items():
        _require_exact_int(value, f"alignment report.offsets.{speaker}")
    if type(channels) is not list or len(channels) != 4:
        _fail("alignment report channels must be a four-item JSON array")
    if any(type(value) is not str for value in channels) or set(channels) != set(SPEAKERS):
        _fail("alignment report channels must be a permutation of A/B/C/R")
    return tuple(channels)


def _bind_sources(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    wave_names = set(SPEAKERS) | {"master", "reference"}
    return {name: _source_binding(path, is_wave=name in wave_names) for name, path in paths.items()}


def _verify_sources_from_report(report: dict[str, Any], *, authority: AudioSourceAuthority) -> None:
    scene = _require_string(report.get("scene"), "contract.scene")
    expected_paths = canonical_source_paths(scene, authority)
    sources = _require_object(report.get("sources"), "contract.sources")
    if set(sources) != set(expected_paths):
        _fail("contract sources do not exactly cover the canonical source set")
    wave_names = set(SPEAKERS) | {"master", "reference"}
    for name, expected_path in expected_paths.items():
        binding = _require_object(sources[name], f"contract.sources.{name}")
        _require_exact_keys(
            binding,
            {
                "path",
                "size_bytes",
                "sha256",
                "media_type",
                "wave_header",
                "pcm_payload",
            },
            f"contract.sources.{name}",
        )
        path = Path(_require_string(binding["path"], f"sources.{name}.path")).resolve()
        if path != expected_path:
            _fail(f"sources.{name}.path is not canonical for the declared scene")
        size = _require_exact_int(binding["size_bytes"], f"sources.{name}.size_bytes", minimum=0)
        digest = binding["sha256"]
        if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
            _fail(f"sources.{name}.sha256 must be a lowercase SHA-256 digest")
        expected_type = "audio/wav" if name in wave_names else "application/json"
        if binding["media_type"] != expected_type:
            _fail(f"sources.{name}.media_type is inconsistent")
        if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
            _fail(f"sources.{name} bytes changed or disappeared")
        if name in wave_names:
            header = _require_object(binding["wave_header"], f"sources.{name}.wave_header")
            _require_exact_keys(
                header,
                {
                    "channels",
                    "sample_width_bytes",
                    "sample_rate_hz",
                    "n_frames",
                    "compression_type",
                },
                f"sources.{name}.wave_header",
            )
            for key in ("channels", "sample_width_bytes", "sample_rate_hz", "n_frames"):
                _require_exact_int(header[key], f"sources.{name}.wave_header.{key}", minimum=0)
            _require_string(header["compression_type"], f"sources.{name}.compression_type")
            if _wave_header(path) != header:
                _fail(f"sources.{name} WAV header changed")
            payload = _require_object(binding["pcm_payload"], f"sources.{name}.pcm_payload")
            _require_exact_keys(
                payload,
                {"frame_size_bytes", "decoded_frames", "frame_aligned", "complete"},
                f"sources.{name}.pcm_payload",
            )
            _require_exact_int(
                payload["frame_size_bytes"],
                f"sources.{name}.pcm_payload.frame_size_bytes",
                minimum=1,
            )
            _require_exact_int(
                payload["decoded_frames"],
                f"sources.{name}.pcm_payload.decoded_frames",
                minimum=0,
            )
            if payload["frame_aligned"] is not True or payload["complete"] is not True:
                _fail(f"sources.{name}.pcm_payload must declare exact boolean completion")
            actual_payload = _verify_complete_pcm_payload(path, expected_header=header)
            if not _type_exact_json_equal(payload, actual_payload):
                _fail(f"sources.{name} PCM payload replay changed")
        elif binding["wave_header"] is not None or binding["pcm_payload"] is not None:
            _fail(f"sources.{name} WAV metadata must be null for JSON evidence")


def _audit_scene_with_authority(
    scene: str,
    *,
    authority: AudioSourceAuthority,
    parameters: AlignmentAuditParameters = DEFAULT_PARAMETERS,
) -> dict[str, Any]:
    paths = canonical_source_paths(scene, authority)
    bindings = _bind_sources(paths)
    _validate_wave_headers(bindings)
    discovered_order = _discover_master_permutation(paths)
    declared_order = _declared_channel_order(paths["alignment_report"].read_bytes())
    transcript = _parse_transcript_offset(
        paths["transcript_offset"].read_bytes(),
        scene,
        bindings["reference"]["wave_header"],
    )
    hpe_anchor = _parse_hpe_anchor(
        paths["hpe_offset"].read_bytes(),
        paths["hpe_offset_quality"].read_bytes(),
        scene=scene,
        a_binding=bindings["A"],
    )

    master_envelope = _hop_rms_envelope(paths["master"], mixdown=True)
    reference_envelope = _hop_rms_envelope(paths["reference"], mixdown=False)
    reference_audit = _segmented_alignment_audit(
        reference_envelope,
        master_envelope,
        declared_lag_hops=transcript["lag_hops"],
        parameters=parameters,
    )
    reference_audit["proposed_offset_correction_ms"] = reference_audit["median_residual_ms"]
    mono_envelopes = {
        speaker: _hop_rms_envelope(paths[speaker], mixdown=False) for speaker in SPEAKERS
    }
    inter_lavalier: dict[str, Any] = {}
    for speaker in SPEAKERS[1:]:
        audit = _segmented_alignment_audit(
            mono_envelopes["A"],
            mono_envelopes[speaker],
            declared_lag_hops=0,
            parameters=parameters,
        )
        median = audit["median_residual_ms"]
        audit["proposed_target_correction_ms"] = -median if median is not None else None
        inter_lavalier[speaker] = audit

    mono_duration = Decimal(bindings["A"]["wave_header"]["n_frames"]) / Decimal(ALIGNED_RATE_HZ)
    required = transcript["required_shared_coverage"]
    required_start = Decimal(required["start_s"])
    required_end = Decimal(required["end_s_exclusive"])
    usable_start = max(Decimal(0), required_start)
    usable_end = min(mono_duration, required_end)
    full_required_coverage = required_start >= 0 and required_end <= mono_duration
    channel_declaration_status = (
        "absent"
        if declared_order is None
        else "matches_sample_replay"
        if declared_order == discovered_order
        else "conflicts_with_sample_replay"
    )
    inter_pass = all(value["quality_pass"] for value in inter_lavalier.values())
    quality_blockers: list[str] = []
    if not reference_audit["quality_pass"]:
        quality_blockers.append("reference_to_master_segmented_sync_failed")
    if not inter_pass:
        quality_blockers.append("inter_lavalier_segmented_sync_failed")
    if usable_end <= usable_start:
        quality_blockers.append("no_nonpadding_common_audio_coverage")
    elif not full_required_coverage:
        quality_blockers.append("reference_coverage_requires_explicit_crop")
    report: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "counted_gold_ready": False,
        "trust_model": "trusted_code_integrity_and_reproducibility_not_signature",
        "sources": bindings,
        "master_sample_replay": {
            "exact_permutation": True,
            "channel_index_to_speaker": list(discovered_order),
            "upstream_declared_order": list(declared_order) if declared_order else None,
            "declaration_status": channel_declaration_status,
            "identity_rule": "sample_replay_only_never_channel_index_or_report",
        },
        "transcript_timing": transcript,
        "hpe_mic_a_anchor": hpe_anchor,
        "audit_parameters": parameters.as_dict(),
        "reference_to_equal_weight_master": reference_audit,
        "inter_lavalier_to_mic_a": inter_lavalier,
        "analysis_window_contract": {
            "axis": "mic_A_shared_seconds",
            "required_reference_coverage": required,
            "aligned_lavalier_coverage": {
                "start_s": "0",
                "end_s_exclusive": str(mono_duration),
            },
            "nonpadding_intersection": {
                "start_s": str(usable_start),
                "end_s_exclusive": str(usable_end),
            },
            "full_required_coverage_available": full_required_coverage,
            "missing_coverage_is_never_zero_filled": True,
        },
        "gates": {
            "source_identity_pass": True,
            "master_exact_permutation_pass": True,
            "reference_segmented_sync_pass": reference_audit["quality_pass"],
            "inter_lavalier_segmented_sync_pass": inter_pass,
            "alignment_prerequisite_pass": not quality_blockers,
            "quality_blockers": quality_blockers,
            "counted_gold_ready": False,
            "counted_gold_blocker": (
                "source/sync evidence is not a gold-VAD annotation or manual audit"
            ),
        },
    }
    # Close the read/validation window before returning any usable result.
    _verify_sources_from_report(report, authority=authority)
    return report


def audit_audio_scene(scene: str) -> dict[str, Any]:
    """Audit one production scene without writing any artifact."""

    return _audit_scene_with_authority(scene, authority=default_source_authority())


def _validate_analysis_window_with_authority(
    report: dict[str, Any],
    start_s: Decimal | str | int | float,
    end_s: Decimal | str | int | float,
    *,
    authority: AudioSourceAuthority,
) -> dict[str, str]:
    """Authority-injected implementation used by isolated tests."""

    _validate_contract_shape(report)
    replayed = _audit_scene_with_authority(report["scene"], authority=authority)
    if not _type_exact_json_equal(report, replayed):
        _fail("analysis-window report does not match a fresh full audio replay")

    def decimal(value: Decimal | str | int | float, label: str) -> Decimal:
        if isinstance(value, bool):
            _fail(f"{label} must not be bool")
        try:
            result = value if isinstance(value, Decimal) else Decimal(str(value))
        except InvalidOperation as exc:
            raise AudioContractError(f"{label} is not numeric") from exc
        if not result.is_finite():
            _fail(f"{label} must be finite")
        return result

    start = decimal(start_s, "analysis start")
    end = decimal(end_s, "analysis end")
    if end <= start:
        _fail("analysis window must be a positive half-open interval")
    contract = _require_object(report.get("analysis_window_contract"), "analysis window contract")
    intersection = _require_object(
        contract.get("nonpadding_intersection"), "analysis nonpadding intersection"
    )
    available_start = _require_decimal_string(intersection.get("start_s"), "coverage start")
    available_end = _require_decimal_string(intersection.get("end_s_exclusive"), "coverage end")
    if start < available_start or end > available_end:
        _fail(
            "analysis window lacks real audio coverage; unavailable time must be cropped, "
            "never filled as silence"
        )
    return {"start_s": str(start), "end_s_exclusive": str(end)}


def validate_analysis_window(
    report: dict[str, Any],
    start_s: Decimal | str | int | float,
    end_s: Decimal | str | int | float,
) -> dict[str, str]:
    """Accept a production interval only after live source and payload replay."""

    return _validate_analysis_window_with_authority(
        report,
        start_s,
        end_s,
        authority=default_source_authority(),
    )


def _validate_contract_shape(report: dict[str, Any]) -> None:
    _require_exact_keys(
        report,
        {
            "artifact_kind",
            "format_version",
            "scene",
            "output_role",
            "counted_gold_ready",
            "trust_model",
            "sources",
            "master_sample_replay",
            "transcript_timing",
            "hpe_mic_a_anchor",
            "audit_parameters",
            "reference_to_equal_weight_master",
            "inter_lavalier_to_mic_a",
            "analysis_window_contract",
            "gates",
        },
        "audio contract",
    )
    if report["artifact_kind"] != ARTIFACT_KIND:
        _fail("audio contract artifact_kind is unsupported")
    version = _require_exact_int(report["format_version"], "audio contract.format_version")
    if version != FORMAT_VERSION:
        _fail("audio contract format version is unsupported")
    _scene_parts(_require_string(report["scene"], "audio contract.scene"))
    if report["output_role"] != OUTPUT_ROLE:
        _fail("audio contract output_role is unsupported")
    if report["counted_gold_ready"] is not False:
        _fail("an audio source contract can never claim counted gold readiness")
    if report["trust_model"] != "trusted_code_integrity_and_reproducibility_not_signature":
        _fail("audio contract trust model is unsupported")


def _load_audio_contract_with_authority(
    path: str | Path, *, authority: AudioSourceAuthority
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        _fail(f"audio contract does not exist: {manifest_path}")
    report = _require_object(
        _strict_json_loads(manifest_path.read_bytes(), "audio contract"), "audio contract"
    )
    _validate_contract_shape(report)
    _verify_sources_from_report(report, authority=authority)
    replayed = _audit_scene_with_authority(report["scene"], authority=authority)
    if not _type_exact_json_equal(report, replayed):
        _fail("audio contract does not exactly match a fresh source and quality replay")
    _verify_sources_from_report(report, authority=authority)
    return report


def load_audio_contract(path: str | Path) -> dict[str, Any]:
    """Load and fully replay a production audio contract."""

    return _load_audio_contract_with_authority(path, authority=default_source_authority())


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


def _write_staged_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_audio_contract_with_authority(
    scene: str,
    output: str | Path,
    *,
    authority: AudioSourceAuthority,
) -> Path:
    """Build one deterministic, atomic, no-clobber contract (testable authority hook)."""

    destination = Path(output).resolve()
    if destination.exists():
        _fail(f"refusing to overwrite existing audio contract: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = _audit_scene_with_authority(scene, authority=authority)
    temporary: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.tmp.",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        _write_staged_json(temporary, report)
        _verify_sources_from_report(report, authority=authority)
        os.link(temporary, destination)
        published = True
        _load_audio_contract_with_authority(destination, authority=authority)
        return destination
    except BaseException:
        if published and temporary is not None:
            try:
                still_ours = destination.exists() and os.path.samefile(temporary, destination)
            except OSError:
                still_ours = False
            if still_ours:
                destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_audio_contract(scene: str, output: str | Path) -> Path:
    """Build a production contract without overwriting an existing artifact."""

    return _write_audio_contract_with_authority(scene, output, authority=default_source_authority())
