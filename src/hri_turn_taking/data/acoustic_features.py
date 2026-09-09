"""Causal energy/F0 evidence extraction from an explicitly named mono PCM source.

This module implements the owner-frozen acoustic algorithm, but deliberately
does not decide which real recording is the dissertation's combined-mix source.
Its optional ``.npz`` output is candidate evidence, not a VAP feature manifest:
it cannot be passed directly to :mod:`hri_turn_taking.data.assembly`.

Every output at tick ``t`` depends only on PCM whose sample interval has ended
at or before ``t``.  Invalid start-up windows remain NaN with explicit validity
masks; neither source-boundary padding nor resampler warm-up is called audio.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.signal import firwin, lfilter

ARTIFACT_KIND = "vap_causal_acoustic_candidate_evidence"
FORMAT_VERSION = 1
EXTRACTOR_ID = "causal_log_rms_nacf_v1"
OUTPUT_ROLE = "unapproved_combined_mix_candidate_evidence"

TARGET_SAMPLE_RATE_HZ = 16_000
FRAME_RATE_HZ = 25
SAMPLES_PER_TICK = TARGET_SAMPLE_RATE_HZ // FRAME_RATE_HZ
ENERGY_WINDOW_SAMPLES = 640
PITCH_WINDOW_SAMPLES = 1_280
F0_MIN_HZ = 50
F0_MAX_HZ = 500
VOICING_THRESHOLD = 0.45
ENERGY_FLOOR = 1e-6
MIN_CENTERED_POWER = 1e-12
RESAMPLE_TAPS_PER_FACTOR = 64
RESAMPLE_CUTOFF_NYQUIST_RATIO = 0.90
RESAMPLE_KAISER_BETA = 8.6
RESAMPLE_STOPBAND_START_HZ = TARGET_SAMPLE_RATE_HZ // 2
RESAMPLE_MINIMUM_STOPBAND_ATTENUATION_DB = 80.0
RESAMPLE_GROUP_DELAY_MS = 2.0
MAX_SOURCE_SAMPLE_RATE_HZ = 192_000

_ARRAY_NAMES = (
    "tick_index",
    "source_local_timestamp_s",
    "array_energy",
    "energy_valid",
    "f0_proxy",
    "voiced",
    "pitch_valid",
)
_NPZ_MEMBERS = {*(f"{name}.npy" for name in _ARRAY_NAMES), "metadata_json.npy"}
_SHA256_RE_LENGTH = 64


class AcousticFeatureError(ValueError):
    """Raised when extraction or evidence validation fails closed."""


@dataclass(frozen=True)
class CausalResampledPCM:
    """A 16 kHz mono stream plus per-sample real-history validity."""

    samples: np.ndarray
    valid: np.ndarray
    source_rate_hz: int
    source_frame_count: int
    decimation_factor: int
    filter_taps: int


@dataclass(frozen=True)
class AcousticFeatureEvidence:
    """Nullable feature evidence on exact 25 Hz source-local ticks."""

    tick_index: np.ndarray
    source_local_timestamp_s: np.ndarray
    array_energy: np.ndarray
    energy_valid: np.ndarray
    f0_proxy: np.ndarray
    voiced: np.ndarray
    pitch_valid: np.ndarray

    @property
    def model_input_valid(self) -> np.ndarray:
        """Rows where all three frozen acoustic inputs are real and usable."""

        return self.energy_valid & self.pitch_valid


@dataclass(frozen=True)
class LoadedAcousticEvidence:
    """A fully validated candidate artifact and its exact metadata."""

    features: AcousticFeatureEvidence
    metadata: dict[str, Any]


def extractor_parameters() -> dict[str, Any]:
    """Return the exact version-1 algorithm contract serialized in artifacts."""

    return {
        "extractor_id": EXTRACTOR_ID,
        "runtime_versions": {
            "numpy": np.__version__,
            "scipy": scipy_version,
        },
        "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "frame_rate_hz": FRAME_RATE_HZ,
        "energy": {
            "representation": "natural_log_rms",
            "window_alignment": "trailing_right_edge_inclusive",
            "window_ms": 40,
            "window_samples": ENERGY_WINDOW_SAMPLES,
            "floor": ENERGY_FLOOR,
        },
        "pitch": {
            "representation": "log2_f0_over_100hz",
            "window_alignment": "trailing_right_edge_inclusive",
            "window_ms": 80,
            "window_samples": PITCH_WINDOW_SAMPLES,
            "mean_removal": True,
            "estimator": "zero_lag_energy_normalized_autocorrelation",
            "peak_selection": "highest_local_maximum_shortest_lag_on_exact_tie",
            "lag_interpolation": "none",
            "minimum_hz": F0_MIN_HZ,
            "maximum_hz": F0_MAX_HZ,
            "voicing_threshold_inclusive": VOICING_THRESHOLD,
            "minimum_centered_power": MIN_CENTERED_POWER,
            "unvoiced_f0_proxy": 0.0,
        },
        "resampler": {
            "algorithm": "causal_fir_lowpass_then_integer_decimation",
            "accepted_source_rate": "integer_multiple_of_16000_hz_up_to_192000_hz",
            "taps_per_decimation_factor": RESAMPLE_TAPS_PER_FACTOR,
            "cutoff_fraction_of_target_nyquist": RESAMPLE_CUTOFF_NYQUIST_RATIO,
            "stopband_start_hz": RESAMPLE_STOPBAND_START_HZ,
            "minimum_stopband_attenuation_db": RESAMPLE_MINIMUM_STOPBAND_ATTENUATION_DB,
            "window": "kaiser",
            "kaiser_beta": RESAMPLE_KAISER_BETA,
            "group_delay_ms": RESAMPLE_GROUP_DELAY_MS,
            "group_delay_policy": "not_compensated_causal_runtime_latency",
            "startup_policy": "invalid_until_full_real_fir_history",
            "tail_policy": "discard_incomplete_target_sample_no_padding",
        },
        "invalid_value": "IEEE_NaN_with_explicit_validity_mask",
    }


def _fail(message: str) -> None:
    raise AcousticFeatureError(message)


def _require_exact_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{name} must be an exact integer (not bool or float)")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be at least {minimum}")
    return value


def _type_exact_equal(left: object, right: object) -> bool:
    """Compare JSON-like values without Python's bool/int/float coercions."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_dict = left  # type: ignore[assignment]
        right_dict = right  # type: ignore[assignment]
        return set(left_dict) == set(right_dict) and all(
            _type_exact_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    if type(left) is list:
        left_list = left  # type: ignore[assignment]
        right_list = right  # type: ignore[assignment]
        return len(left_list) == len(right_list) and all(
            _type_exact_equal(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _normalise_mono_pcm(pcm: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(pcm)
    if values.ndim != 1:
        _fail("mono PCM must be a one-dimensional array")
    if values.size == 0:
        _fail("mono PCM must contain at least one sample")
    if (
        values.dtype == np.bool_
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        _fail("mono PCM must use a numeric, non-boolean dtype")
    if np.issubdtype(values.dtype, np.integer):
        if values.dtype != np.dtype(np.int16):
            _fail("integer PCM must use signed 16-bit samples")
        normalised = values.astype(np.float64) / 32768.0
    else:
        normalised = values.astype(np.float64)
    if not np.isfinite(normalised).all():
        _fail("mono PCM contains NaN or infinity")
    if (np.abs(normalised) > 1.0).any():
        _fail("floating-point PCM must be normalized to [-1, 1]")
    return np.ascontiguousarray(normalised)


def causal_resample_to_16khz(
    pcm: np.ndarray | list[float], source_rate_hz: int
) -> CausalResampledPCM:
    """Convert normalized mono PCM to 16 kHz without future-sample access.

    The real corpus is 48 kHz and the runtime target is 16 kHz.  Version 1
    intentionally accepts only integer-rate decimation (16/32/48/... kHz), so
    there is no hidden fractional or bidirectional interpolation policy.
    """

    source_rate_hz = _require_exact_int(source_rate_hz, "source_rate_hz", minimum=1)
    if source_rate_hz % TARGET_SAMPLE_RATE_HZ != 0:
        _fail("source_rate_hz must be a positive integer multiple of 16000 for extractor v1")
    if source_rate_hz > MAX_SOURCE_SAMPLE_RATE_HZ:
        _fail("source_rate_hz exceeds the extractor-v1 maximum of 192000 Hz")
    samples = _normalise_mono_pcm(pcm)
    factor = source_rate_hz // TARGET_SAMPLE_RATE_HZ
    if factor == 1:
        return CausalResampledPCM(
            samples=samples.copy(),
            valid=np.ones(samples.shape, dtype=np.bool_),
            source_rate_hz=source_rate_hz,
            source_frame_count=int(samples.size),
            decimation_factor=1,
            filter_taps=1,
        )

    tap_count = RESAMPLE_TAPS_PER_FACTOR * factor + 1
    taps = firwin(
        tap_count,
        cutoff=(TARGET_SAMPLE_RATE_HZ / 2) * RESAMPLE_CUTOFF_NYQUIST_RATIO,
        window=("kaiser", RESAMPLE_KAISER_BETA),
        fs=source_rate_hz,
        pass_zero="lowpass",
        scale=True,
    )
    filtered = lfilter(taps, np.array([1.0], dtype=np.float64), samples)
    # Output sample j covers the completed source block ending at
    # (j + 1) / 16 kHz.  Selecting factor-1, 2*factor-1, ... never reads a
    # source sample whose interval ends after that output time.
    source_end_indices = np.arange(factor - 1, samples.size, factor, dtype=np.int64)
    output = np.ascontiguousarray(filtered[source_end_indices], dtype=np.float64)
    # lfilter initializes missing negative-time history to zero.  Those samples
    # are retained only as explicitly invalid evidence, never as real audio.
    valid = source_end_indices >= tap_count - 1
    return CausalResampledPCM(
        samples=output,
        valid=np.ascontiguousarray(valid, dtype=np.bool_),
        source_rate_hz=source_rate_hz,
        source_frame_count=int(samples.size),
        decimation_factor=factor,
        filter_taps=tap_count,
    )


def _validate_tick_indices(tick_indices: object, maximum_exclusive: int) -> np.ndarray:
    try:
        raw = list(tick_indices)  # type: ignore[arg-type]
    except TypeError as exc:
        raise AcousticFeatureError("tick_indices must be an iterable of exact integers") from exc
    if not raw:
        _fail("tick_indices must contain at least one requested 25 Hz tick")
    invalid_type = any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    )
    if invalid_type:
        _fail("tick_indices must contain exact integers (not bool or float)")
    integer_values = [int(value) for value in raw]
    if any(value < 0 for value in integer_values):
        _fail("tick_indices must be non-negative")
    if any(value >= maximum_exclusive for value in integer_values):
        _fail("tick_indices contains a timestamp outside the half-open PCM coverage")
    ticks = np.asarray(integer_values, dtype=np.int64)
    if (np.diff(ticks) <= 0).any():
        _fail("tick_indices must be strictly increasing and unique")
    return ticks


def _default_tick_stop(resampled_sample_count: int) -> int:
    # Tick k is at k / 25 seconds.  Source coverage is half-open, so include
    # only k*640 < sample_count.  This is ceil(sample_count / 640).
    return (resampled_sample_count + SAMPLES_PER_TICK - 1) // SAMPLES_PER_TICK


def _pitch_from_window(window: np.ndarray) -> tuple[float, float]:
    centered = window - float(np.mean(window))
    zero_lag_power = float(np.dot(centered, centered))
    if not math.isfinite(zero_lag_power) or zero_lag_power <= MIN_CENTERED_POWER:
        return 0.0, 0.0
    lag_min = math.ceil(TARGET_SAMPLE_RATE_HZ / F0_MAX_HZ)
    lag_max = math.floor(TARGET_SAMPLE_RATE_HZ / F0_MIN_HZ)
    # Compute from lag 1 so the lower 500 Hz boundary is accepted only when it
    # is a genuine local peak.  Looking only inside [32, 320] would mistake the
    # still-descending shoulder of a 50 Hz period for a spurious 500 Hz peak.
    correlations = np.fromiter(
        (
            float(np.dot(centered[:-lag], centered[lag:])) / zero_lag_power
            for lag in range(1, lag_max + 1)
        ),
        dtype=np.float64,
        count=lag_max,
    )
    candidates = [
        lag
        for lag in range(lag_min, lag_max + 1)
        if correlations[lag - 1] >= correlations[lag - 2]
        and (lag == lag_max or correlations[lag - 1] >= correlations[lag])
    ]
    if not candidates:
        return 0.0, 0.0
    # Python's max preserves the first candidate on an exact tie, hence the
    # shortest-lag rule is deterministic.
    lag = max(candidates, key=lambda candidate: correlations[candidate - 1])
    peak = float(correlations[lag - 1])
    if not math.isfinite(peak) or peak < VOICING_THRESHOLD:
        return 0.0, 0.0
    f0_hz = TARGET_SAMPLE_RATE_HZ / lag
    return math.log2(f0_hz / 100.0), 1.0


def extract_causal_acoustic_features(
    pcm: np.ndarray | list[float],
    source_rate_hz: int,
    *,
    tick_indices: object | None = None,
) -> AcousticFeatureEvidence:
    """Extract nullable causal features at exact 25 Hz source-local ticks."""

    resampled = causal_resample_to_16khz(pcm, source_rate_hz)
    maximum_tick_exclusive = _default_tick_stop(resampled.samples.size)
    if maximum_tick_exclusive <= 0:
        _fail("PCM is too short to contain a 25 Hz timestamp inside its coverage")
    if tick_indices is None:
        ticks = np.arange(maximum_tick_exclusive, dtype=np.int64)
    else:
        ticks = _validate_tick_indices(tick_indices, maximum_tick_exclusive)

    count = int(ticks.size)
    energy = np.full(count, np.nan, dtype=np.float64)
    energy_valid = np.zeros(count, dtype=np.bool_)
    pitch = np.full(count, np.nan, dtype=np.float64)
    voiced = np.full(count, np.nan, dtype=np.float64)
    pitch_valid = np.zeros(count, dtype=np.bool_)

    for row, tick in enumerate(ticks):
        end = int(tick) * SAMPLES_PER_TICK
        energy_start = end - ENERGY_WINDOW_SAMPLES
        if (
            energy_start >= 0
            and end <= resampled.samples.size
            and bool(resampled.valid[energy_start:end].all())
        ):
            window = resampled.samples[energy_start:end]
            rms = math.sqrt(float(np.mean(np.square(window, dtype=np.float64))))
            energy[row] = math.log(max(rms, ENERGY_FLOOR))
            energy_valid[row] = True

        pitch_start = end - PITCH_WINDOW_SAMPLES
        if (
            pitch_start >= 0
            and end <= resampled.samples.size
            and bool(resampled.valid[pitch_start:end].all())
        ):
            pitch[row], voiced[row] = _pitch_from_window(
                resampled.samples[pitch_start:end]
            )
            pitch_valid[row] = True

    return AcousticFeatureEvidence(
        tick_index=ticks,
        source_local_timestamp_s=ticks.astype(np.float64) / FRAME_RATE_HZ,
        array_energy=energy,
        energy_valid=energy_valid,
        f0_proxy=pitch,
        voiced=voiced,
        pitch_valid=pitch_valid,
    )


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _read_mono_pcm16_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    before_hash, before_size = _sha256_path(resolved)
    chunks: list[np.ndarray] = []
    try:
        with resolved.open("rb") as raw_handle:
            with wave.open(raw_handle, "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                header_frames = wav.getnframes()
                compression = wav.getcomptype()
                if channels != 1:
                    _fail("acoustic evidence source must be mono")
                if sample_width != 2 or compression != "NONE":
                    _fail("acoustic evidence source must be uncompressed PCM16 WAV")
                _require_exact_int(sample_rate, "WAV sample_rate_hz", minimum=1)
                decoded_frames = 0
                while decoded_frames < header_frames:
                    requested = min(65_536, header_frames - decoded_frames)
                    raw = wav.readframes(requested)
                    if not raw:
                        _fail("WAV payload ended before the declared frame count")
                    if len(raw) % 2:
                        _fail("WAV payload has a partial PCM16 frame")
                    block = np.frombuffer(raw, dtype="<i2").copy()
                    decoded_frames += int(block.size)
                    chunks.append(block)
                if decoded_frames != header_frames:
                    _fail("decoded WAV frame count does not match the header")
                if wav.readframes(1):
                    _fail("WAV contains decodable PCM beyond its declared frame count")
    except (EOFError, OSError, wave.Error) as exc:
        raise AcousticFeatureError(f"could not decode complete mono PCM16 WAV {resolved}") from exc

    after_hash, after_size = _sha256_path(resolved)
    if (after_hash, after_size) != (before_hash, before_size):
        _fail("WAV source changed while it was being decoded")
    pcm = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
    if pcm.size != header_frames:
        _fail("WAV payload is empty or short relative to its header")
    binding = {
        "path": str(resolved),
        "sha256": before_hash,
        "byte_length": before_size,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "header_frame_count": header_frames,
        "decoded_frame_count": int(pcm.size),
        "compression_type": compression,
        "complete_payload": True,
    }
    return pcm, binding


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _metadata_for(
    source_binding: dict[str, Any],
    candidate_label: str,
    features: AcousticFeatureEvidence,
) -> dict[str, Any]:
    arrays = {
        name: {
            "dtype": str(getattr(features, name).dtype),
            "shape": list(getattr(features, name).shape),
            "sha256": _array_sha256(getattr(features, name)),
        }
        for name in _ARRAY_NAMES
    }
    return {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "output_role": OUTPUT_ROLE,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "source_selection": {
            "status": "unapproved_candidate",
            "candidate_label": candidate_label,
        },
        "source": source_binding,
        "extractor": extractor_parameters(),
        "tick_domain": {
            "name": "source_local_time",
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_tick_inclusive": int(features.tick_index[0]),
            "stop_tick_exclusive": int(features.tick_index[-1]) + 1,
            "ticks_are_contiguous": bool(
                np.array_equal(
                    features.tick_index,
                    np.arange(
                        int(features.tick_index[0]),
                        int(features.tick_index[-1]) + 1,
                        dtype=np.int64,
                    ),
                )
            ),
        },
        "arrays": arrays,
    }


def _strict_metadata_load(raw: np.ndarray) -> dict[str, Any]:
    if raw.dtype != np.dtype(np.uint8) or raw.ndim != 1:
        _fail("metadata_json must be a one-dimensional uint8 array")
    try:
        text = raw.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AcousticFeatureError("metadata_json is not strict UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"metadata_json contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        _fail(f"metadata_json contains forbidden constant {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise AcousticFeatureError("metadata_json is not valid JSON") from exc
    if type(value) is not dict:
        _fail("metadata_json must contain a JSON object")
    return value


def _validate_feature_arrays(features: AcousticFeatureEvidence) -> None:
    expected_dtypes = {
        "tick_index": np.dtype(np.int64),
        "source_local_timestamp_s": np.dtype(np.float64),
        "array_energy": np.dtype(np.float64),
        "energy_valid": np.dtype(np.bool_),
        "f0_proxy": np.dtype(np.float64),
        "voiced": np.dtype(np.float64),
        "pitch_valid": np.dtype(np.bool_),
    }
    lengths: set[int] = set()
    for name, dtype in expected_dtypes.items():
        array = getattr(features, name)
        if array.dtype != dtype or array.ndim != 1:
            _fail(f"{name} must be a one-dimensional {dtype} array")
        lengths.add(int(array.size))
    if lengths == {0} or len(lengths) != 1:
        _fail("evidence arrays must have one shared non-zero length")
    ticks = features.tick_index
    if (ticks < 0).any() or (np.diff(ticks) <= 0).any():
        _fail("tick_index must be strictly increasing, unique and non-negative")
    expected_times = ticks.astype(np.float64) / FRAME_RATE_HZ
    if not np.array_equal(features.source_local_timestamp_s, expected_times):
        _fail("source_local_timestamp_s does not exactly match tick_index / 25")
    for values, valid, name in (
        (features.array_energy, features.energy_valid, "array_energy"),
        (features.f0_proxy, features.pitch_valid, "f0_proxy"),
        (features.voiced, features.pitch_valid, "voiced"),
    ):
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            _fail(f"{name} must be finite when valid and NaN when invalid")
    valid_voiced = features.voiced[features.pitch_valid]
    if not np.isin(valid_voiced, (0.0, 1.0)).all():
        _fail("voiced must be binary on pitch-valid rows")
    unvoiced = features.pitch_valid & (features.voiced == 0.0)
    if (features.f0_proxy[unvoiced] != 0.0).any():
        _fail("f0_proxy must be exactly zero on valid unvoiced rows")


def _load_npz_arrays(path: Path) -> tuple[AcousticFeatureEvidence, dict[str, Any]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise AcousticFeatureError(f"could not read acoustic evidence archive {path}") from exc
    if len(names) != len(set(names)) or set(names) != _NPZ_MEMBERS:
        _fail("acoustic evidence archive members do not exactly match format v1")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in _ARRAY_NAMES}
            metadata = _strict_metadata_load(archive["metadata_json"])
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AcousticFeatureError(f"could not decode acoustic evidence archive {path}") from exc
    features = AcousticFeatureEvidence(**arrays)
    _validate_feature_arrays(features)
    return features, metadata


def _validate_metadata(
    metadata: dict[str, Any],
    features: AcousticFeatureEvidence,
) -> None:
    expected_top = {
        "artifact_kind",
        "format_version",
        "output_role",
        "assembly_eligible",
        "counted_input_ready",
        "source_selection",
        "source",
        "extractor",
        "tick_domain",
        "arrays",
    }
    if set(metadata) != expected_top:
        _fail("metadata keys do not exactly match acoustic evidence format v1")
    if metadata["artifact_kind"] != ARTIFACT_KIND:
        _fail("acoustic evidence artifact_kind is unsupported")
    if type(metadata["format_version"]) is not int or metadata["format_version"] != FORMAT_VERSION:
        _fail("acoustic evidence format_version is unsupported")
    if metadata["output_role"] != OUTPUT_ROLE:
        _fail("acoustic evidence output_role is unsupported")
    if metadata["assembly_eligible"] is not False or metadata["counted_input_ready"] is not False:
        _fail("candidate acoustic evidence cannot claim assembly or counted-input readiness")
    selection = metadata["source_selection"]
    if type(selection) is not dict or set(selection) != {"status", "candidate_label"}:
        _fail("source_selection does not match format v1")
    label = selection["candidate_label"]
    if (
        selection["status"] != "unapproved_candidate"
        or type(label) is not str
        or not label
        or label != label.strip()
    ):
        _fail("source selection must remain a named, unapproved candidate")
    if not _type_exact_equal(metadata["extractor"], extractor_parameters()):
        _fail("extractor parameters do not exactly match version 1")

    source = metadata["source"]
    expected_source_keys = {
        "path",
        "sha256",
        "byte_length",
        "channels",
        "sample_width_bytes",
        "sample_rate_hz",
        "header_frame_count",
        "decoded_frame_count",
        "compression_type",
        "complete_payload",
    }
    if type(source) is not dict or set(source) != expected_source_keys:
        _fail("source binding does not exactly match format v1")
    if (
        type(source["path"]) is not str
        or not source["path"]
        or not Path(source["path"]).is_absolute()
    ):
        _fail("source.path must be a non-empty absolute path")
    sha = source["sha256"]
    if (
        type(sha) is not str
        or len(sha) != _SHA256_RE_LENGTH
        or any(char not in "0123456789abcdef" for char in sha)
    ):
        _fail("source.sha256 must be a lowercase SHA-256")
    for key in (
        "byte_length",
        "channels",
        "sample_width_bytes",
        "sample_rate_hz",
        "header_frame_count",
        "decoded_frame_count",
    ):
        _require_exact_int(source[key], f"source.{key}", minimum=1)
    if (
        source["channels"] != 1
        or source["sample_width_bytes"] != 2
        or source["compression_type"] != "NONE"
        or source["complete_payload"] is not True
        or source["header_frame_count"] != source["decoded_frame_count"]
    ):
        _fail("source binding is not a complete mono PCM16 WAV")

    tick_domain = metadata["tick_domain"]
    expected_tick_domain = {
        "name",
        "frame_rate_hz",
        "start_tick_inclusive",
        "stop_tick_exclusive",
        "ticks_are_contiguous",
    }
    if type(tick_domain) is not dict or set(tick_domain) != expected_tick_domain:
        _fail("tick_domain does not exactly match format v1")
    if (
        tick_domain["name"] != "source_local_time"
        or type(tick_domain["frame_rate_hz"]) is not int
        or tick_domain["frame_rate_hz"] != 25
    ):
        _fail("tick_domain must be the 25 Hz source-local axis")
    if (
        type(tick_domain["start_tick_inclusive"]) is not int
        or type(tick_domain["stop_tick_exclusive"]) is not int
        or type(tick_domain["ticks_are_contiguous"]) is not bool
    ):
        _fail("tick_domain numeric/boolean values must use exact JSON types")
    expected_contiguous = np.array_equal(
        features.tick_index,
        np.arange(
            int(features.tick_index[0]),
            int(features.tick_index[-1]) + 1,
            dtype=np.int64,
        ),
    )
    if (
        tick_domain["start_tick_inclusive"] != int(features.tick_index[0])
        or tick_domain["stop_tick_exclusive"] != int(features.tick_index[-1]) + 1
        or tick_domain["ticks_are_contiguous"] is not bool(expected_contiguous)
    ):
        _fail("tick_domain does not bind the evidence tick array")

    arrays = metadata["arrays"]
    if type(arrays) is not dict or set(arrays) != set(_ARRAY_NAMES):
        _fail("arrays metadata must bind exactly all evidence arrays")
    for name in _ARRAY_NAMES:
        record = arrays[name]
        array = getattr(features, name)
        if type(record) is not dict or set(record) != {"dtype", "shape", "sha256"}:
            _fail(f"arrays.{name} does not match format v1")
        expected_record = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
        if not _type_exact_equal(record, expected_record):
            _fail(f"arrays.{name} does not bind the stored array")


def _replay_evidence(
    features: AcousticFeatureEvidence,
    metadata: dict[str, Any],
) -> None:
    source_path = Path(metadata["source"]["path"])
    pcm, current_binding = _read_mono_pcm16_wav(source_path)
    if current_binding != metadata["source"]:
        _fail("bound WAV source changed after acoustic evidence extraction")
    replay = extract_causal_acoustic_features(
        pcm,
        current_binding["sample_rate_hz"],
        tick_indices=features.tick_index.tolist(),
    )
    for name in _ARRAY_NAMES:
        if not np.array_equal(
            getattr(features, name),
            getattr(replay, name),
            equal_nan=True,
        ):
            _fail(f"stored {name} does not match deterministic source replay")
    # The replay above can be comparatively expensive.  Recheck the source at
    # the end so a replacement that happens after decoding but before return
    # cannot leave the caller holding a stale validation result.
    final_hash, final_size = _sha256_path(source_path)
    if (
        final_hash != metadata["source"]["sha256"]
        or final_size != metadata["source"]["byte_length"]
    ):
        _fail("bound WAV source changed during acoustic evidence replay")


def load_acoustic_evidence(path: str | Path) -> LoadedAcousticEvidence:
    """Load, structurally validate, re-hash, and deterministically replay evidence."""

    artifact_path = Path(path).resolve(strict=True)
    features, metadata = _load_npz_arrays(artifact_path)
    _validate_metadata(metadata, features)
    _replay_evidence(features, metadata)
    return LoadedAcousticEvidence(features=features, metadata=metadata)


def _write_stage(
    stage: Path,
    features: AcousticFeatureEvidence,
    metadata: dict[str, Any],
) -> None:
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    with stage.open("wb") as handle:
        np.savez(
            handle,
            **{name: getattr(features, name) for name in _ARRAY_NAMES},
            metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
        )


def build_acoustic_evidence(
    wav_path: str | Path,
    destination: str | Path,
    *,
    candidate_label: str,
    tick_start_inclusive: int = 0,
    tick_stop_exclusive: int | None = None,
) -> LoadedAcousticEvidence:
    """Publish one replayable, atomic no-clobber candidate evidence artifact.

    This function never approves the source or produces a formal feature
    manifest.  A later owner decision and provenance-bound assembly builder are
    required before any values may populate the canonical VAP wide table.
    """

    if (
        type(candidate_label) is not str
        or not candidate_label
        or candidate_label != candidate_label.strip()
    ):
        _fail("candidate_label must be a non-empty string without surrounding whitespace")
    start = _require_exact_int(tick_start_inclusive, "tick_start_inclusive", minimum=0)
    output = Path(destination).resolve()
    if output.suffix.lower() != ".npz":
        _fail("acoustic candidate evidence destination must use the .npz suffix")
    if output.exists():
        _fail(f"destination already exists; refusing to overwrite {output}")
    if not output.parent.is_dir():
        _fail("destination parent directory must already exist")

    pcm, binding = _read_mono_pcm16_wav(Path(wav_path))
    sample_rate = binding["sample_rate_hz"]
    # Compute the exact v1 target length without first extracting every tick.
    factor = sample_rate // TARGET_SAMPLE_RATE_HZ if sample_rate % TARGET_SAMPLE_RATE_HZ == 0 else 0
    if factor <= 0:
        _fail("WAV sample rate is unsupported by extractor v1")
    target_count = binding["decoded_frame_count"] // factor
    maximum_tick_exclusive = _default_tick_stop(target_count)
    stop = (
        maximum_tick_exclusive
        if tick_stop_exclusive is None
        else _require_exact_int(tick_stop_exclusive, "tick_stop_exclusive", minimum=1)
    )
    if start >= stop or stop > maximum_tick_exclusive:
        _fail("requested tick interval is empty or outside half-open PCM coverage")
    ticks = list(range(start, stop))
    features = extract_causal_acoustic_features(pcm, sample_rate, tick_indices=ticks)
    if not bool(features.model_input_valid.any()):
        _fail("requested tick interval has no row with complete real 80 ms history")
    metadata = _metadata_for(binding, candidate_label, features)

    stage: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.tmp.",
            suffix=".npz",
            delete=False,
        ) as handle:
            stage = Path(handle.name)
        _write_stage(stage, features, metadata)
        load_acoustic_evidence(stage)
        os.link(stage, output)
        loaded = load_acoustic_evidence(output)
        return loaded
    except FileExistsError as exc:
        raise AcousticFeatureError(
            f"destination already exists; refusing to overwrite {output}"
        ) from exc
    except BaseException:
        # ``os.link`` is atomic, but a wrapper/interruption can raise after the
        # link was created and before the caller records completion.  Inspect
        # inode identity; never remove a concurrent owner's different target.
        if stage is not None:
            try:
                if output.exists() and os.path.samefile(stage, output):
                    output.unlink()
            except OSError:
                pass
        raise
    finally:
        if stage is not None:
            try:
                stage.unlink(missing_ok=True)
            except OSError:
                pass
