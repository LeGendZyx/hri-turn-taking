"""Read-only real-corpus preflight for the frozen causal acoustic extractor.

This module closes the source/timeline gap between the generic mono extractor
and the real dissertation corpus.  It deliberately does *not* publish a v3
feature manifest.  The owner froze the N-A-C-F voicing threshold at 0.45 on
2026-08-16, but every result from this legacy feasibility module remains
preflight-only and cannot become an assembly input by renaming it.

The approved source *kind* is the sample-replayed four-channel
``master_aligned.wav``.  The mix converts each int16 frame to float64, takes
the four-channel arithmetic mean, and divides by ``32768``; there is no
intermediate PCM16 rounding and no
gold-only per-channel time correction.  Features are extracted from the full
master and selected only at the canonical analysis-base mic-A ticks, preserving
the complete causal FIR and 80-ms history for the first output row.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import (
    acoustic_features,
    analysis_base,
    assembly,
    audio_contract,
)

ARTIFACT_KIND = "vap_real_causal_acoustic_preflight"
FORMAT_VERSION = 1
OUTPUT_ROLE = "preflight_only_owner_parameter_approval_pending"
SCAFFOLD_ARTIFACT_KIND = "vap_real_causal_acoustic_read_only_scaffold"
SCAFFOLD_EVIDENCE_KIND = "vap_real_causal_acoustic_evidence_scaffold"
SCAFFOLD_MANIFEST_KIND = "vap_feature_source_read_only_scaffold"
SCAFFOLD_FORMAT_VERSION = 1
PARAMETER_STATUS = "owner_approval_pending"
APPROVED_SOURCE_STATUS = "owner_selected_source_kind_2026_08_15"
MIX_RULE = "int16_to_float64_frame_mean_then_divide_by_32768"
AXIS = "mic_A_shared_seconds"
FEATURE_COLUMNS = ("array_energy", "f0_proxy", "voiced")
SOURCE_KIND = "sample_replayed_equal_weight_master_aligned"
DEPLOYMENT_DOMAIN_SHIFT = (
    "training source is an equal-weight close-talk lavalier surrogate; "
    "deployment observes a physical robot microphone"
)
PARTICIPANT_TO_SLOT = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
_EVIDENCE_DTYPES = {
    "tick_index": np.dtype(np.int64),
    "source_local_timestamp_s": np.dtype(np.float64),
    "array_energy": np.dtype(np.float64),
    "energy_valid": np.dtype(np.bool_),
    "f0_proxy": np.dtype(np.float64),
    "voiced": np.dtype(np.float64),
    "pitch_valid": np.dtype(np.bool_),
}
FRAME_RATE_HZ = 25
MASTER_CHANNELS = 4
MASTER_SAMPLE_RATE_HZ = 48_000
MASTER_SAMPLE_WIDTH_BYTES = 2


class RealAcousticFeatureError(ValueError):
    """A real acoustic source, timeline, or extraction gate failed closed."""


@dataclass(frozen=True, kw_only=True)
class RealAcousticAuthority:
    """Code-owned real source and canonical analysis-base authorities."""

    analysis: analysis_base.AnalysisBaseAuthority
    audio: audio_contract.AudioSourceAuthority

    def __post_init__(self) -> None:
        if type(self.analysis) is not analysis_base.AnalysisBaseAuthority:
            raise RealAcousticFeatureError("analysis authority has the wrong exact type")
        if type(self.audio) is not audio_contract.AudioSourceAuthority:
            raise RealAcousticFeatureError("audio authority has the wrong exact type")


@dataclass(frozen=True)
class RealAcousticPreflight:
    """In-memory, non-publishable real acoustic feature preflight."""

    scene: str
    feature_df: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class ReadOnlyAcousticArtifactScaffold:
    """Immutable in-memory bytes that exercise, but cannot enter, assembly v3.

    The payload has no destination path and this module exposes no writer.  Its
    manifest deliberately uses ``SCAFFOLD_MANIFEST_KIND`` rather than
    :data:`assembly.FEATURE_MANIFEST_KIND`, so saving or renaming the bytes
    cannot make them acceptable to :func:`assembly.assemble_session`.
    """

    scene: str
    source_csv_payload: bytes
    evidence_json_payload: bytes
    feature_csv_payload: bytes
    feature_manifest_json_payload: bytes


@dataclass(frozen=True)
class LoadedReadOnlyAcousticArtifactScaffold:
    """Strictly replayed view of a read-only in-memory scaffold."""

    scene: str
    source_df: pd.DataFrame
    evidence: dict[str, Any]
    feature_df: pd.DataFrame
    feature_manifest: dict[str, Any]


def default_real_acoustic_authority() -> RealAcousticAuthority:
    """Return the sole production source/path authority."""

    return RealAcousticAuthority(
        analysis=analysis_base.default_analysis_base_authority(),
        audio=audio_contract.default_source_authority(),
    )


def _fail(message: str) -> None:
    raise RealAcousticFeatureError(message)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact integer (not bool or float)")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _sha256_digest(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _source_selection_contract() -> dict[str, Any]:
    return {
        "status": APPROVED_SOURCE_STATUS,
        "source_kind": SOURCE_KIND,
        "mix_rule": MIX_RULE,
        "gold_channel_time_corrections_applied": False,
        "deployment_domain_shift": DEPLOYMENT_DOMAIN_SHIFT,
    }


def _preflight_causality_contract() -> dict[str, Any]:
    return {
        "causal": True,
        "lookahead_s": 0.0,
        "window_alignment": "trailing",
        "full_master_processed_before_tick_selection": True,
        "resampler_group_delay_compensated": False,
    }


def _manifest_causality_contract() -> dict[str, Any]:
    return {
        "causal": True,
        "lookahead_s": 0.0,
        "window_alignment": "trailing",
    }


def _analysis_key_schema(frame: pd.DataFrame) -> dict[str, Any]:
    columns = ["session_id", "timestamp_s"]
    return {
        "columns": columns,
        "dtypes": {column: str(frame[column].dtype) for column in columns},
    }


def _feature_array_contract(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        values = frame[name].to_numpy()
        result[name] = {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "n_rows": len(frame),
            "sha256": _sha256_array(values),
        }
    return result


def _master_source_contract(binding: dict[str, Any]) -> dict[str, Any]:
    expected_binding_keys = {
        "path",
        "size_bytes",
        "sha256",
        "media_type",
        "wave_header",
        "pcm_payload",
    }
    if type(binding) is not dict or set(binding) != expected_binding_keys:
        _fail("audio contract master source binding has the wrong exact schema")
    header = binding["wave_header"]
    expected_header_keys = {
        "channels",
        "sample_width_bytes",
        "sample_rate_hz",
        "n_frames",
        "compression_type",
    }
    if type(header) is not dict or set(header) != expected_header_keys:
        _fail("audio contract master WAV header has the wrong exact schema")
    payload = binding["pcm_payload"]
    expected_payload_keys = {
        "frame_size_bytes",
        "decoded_frames",
        "frame_aligned",
        "complete",
    }
    if type(payload) is not dict or set(payload) != expected_payload_keys:
        _fail("audio contract master PCM payload has the wrong exact schema")
    return {
        "path": binding["path"],
        "size_bytes": binding["size_bytes"],
        "sha256": binding["sha256"],
        "media_type": binding["media_type"],
        "wave_header": {key: header[key] for key in expected_header_keys},
        "pcm_payload": {key: payload[key] for key in expected_payload_keys},
    }


def _load_analysis_dataset(
    scene: str,
    *,
    authority: RealAcousticAuthority,
) -> assembly._DatasetFile:
    if scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail(f"scene is outside the frozen ten-scene analysis inventory: {scene}")
    path = (authority.analysis.analysis_base_root / f"{scene}.csv").resolve()
    try:
        return assembly._load_dataset_file(
            path,
            data_kind="real",
            analysis_base_authority=authority.analysis,
        )
    except (OSError, ValueError) as exc:
        raise RealAcousticFeatureError(
            f"canonical analysis base failed sealed replay for {scene}: {exc}"
        ) from exc


def _load_audio_report(
    scene: str,
    *,
    authority: RealAcousticAuthority,
) -> tuple[dict[str, Any], Path, str, int]:
    manifest = (authority.analysis.audio_manifest_root / f"{scene}.audio_contract.json").resolve()
    if not manifest.is_file():
        _fail(f"canonical audio contract does not exist: {manifest}")
    before_hash, before_size = _sha256_file(manifest)
    try:
        report = audio_contract._load_audio_contract_with_authority(
            manifest,
            authority=authority.audio,
        )
    except (OSError, ValueError) as exc:
        raise RealAcousticFeatureError(
            f"formal audio contract failed full source replay for {scene}: {exc}"
        ) from exc
    after_hash, after_size = _sha256_file(manifest)
    if (after_hash, after_size) != (before_hash, before_size):
        _fail("audio contract changed while the real acoustic source was loaded")
    if report.get("scene") != scene:
        _fail("audio contract scene does not match the requested scene")
    return report, manifest, before_hash, before_size


def _binding_header(binding: dict[str, Any]) -> dict[str, Any]:
    header = binding.get("wave_header")
    if type(header) is not dict:
        _fail("audio contract master binding has no wave_header object")
    return header


def _read_equal_weight_master(
    path: Path,
    *,
    binding: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the complete four-channel PCM16 master to an order-invariant mix."""

    resolved = path.resolve(strict=True)
    if binding.get("path") != str(resolved):
        _fail("audio contract master path is not the canonical master source")
    declared_hash = binding.get("sha256")
    declared_size = binding.get("size_bytes")
    if type(declared_hash) is not str or type(declared_size) is not int:
        _fail("audio contract master byte binding is malformed")
    before_hash, before_size = _sha256_file(resolved)
    if (before_hash, before_size) != (declared_hash, declared_size):
        _fail("master bytes differ from the fully replayed audio contract")

    header = _binding_header(binding)
    expected_header = {
        "channels": MASTER_CHANNELS,
        "sample_width_bytes": MASTER_SAMPLE_WIDTH_BYTES,
        "sample_rate_hz": MASTER_SAMPLE_RATE_HZ,
        "compression_type": "NONE",
    }
    for key, expected in expected_header.items():
        actual = header.get(key)
        if type(actual) is not type(expected) or actual != expected:
            _fail(f"master {key} must equal {expected!r}")
    declared_frames = _exact_int(header.get("n_frames"), "master n_frames", minimum=1)

    pcm: np.ndarray | None = None
    try:
        with resolved.open("rb") as raw_handle:
            with wave.open(raw_handle, "rb") as wav:
                actual_header = {
                    "channels": wav.getnchannels(),
                    "sample_width_bytes": wav.getsampwidth(),
                    "sample_rate_hz": wav.getframerate(),
                    "compression_type": wav.getcomptype(),
                    "n_frames": wav.getnframes(),
                }
                if actual_header != {**expected_header, "n_frames": declared_frames}:
                    _fail("master WAV header changed after the audio-contract replay")
                pcm = np.empty(declared_frames, dtype=np.float64)
                decoded_frames = 0
                frame_bytes = MASTER_CHANNELS * MASTER_SAMPLE_WIDTH_BYTES
                while decoded_frames < declared_frames:
                    requested = min(65_536, declared_frames - decoded_frames)
                    raw = wav.readframes(requested)
                    if not raw:
                        _fail("master WAV payload ended before its declared frame count")
                    if len(raw) % frame_bytes:
                        _fail("master WAV yielded a partial multi-channel PCM frame")
                    values = np.frombuffer(raw, dtype="<i2")
                    if values.size != requested * MASTER_CHANNELS:
                        _fail("master WAV returned an unexpected PCM frame count")
                    channels = values.reshape(requested, MASTER_CHANNELS).astype(np.float64)
                    # Four int16 values and their division by four are exactly
                    # representable in float64.  This is order-invariant and
                    # deliberately avoids an intermediate quantized mono WAV.
                    mix = channels.mean(axis=1) / 32768.0
                    pcm[decoded_frames : decoded_frames + requested] = mix
                    decoded_frames += requested
                if decoded_frames != declared_frames:
                    _fail("decoded master frame count differs from the audio contract")
                if wav.readframes(1):
                    _fail("master WAV has decodable PCM beyond its declared frame count")
    except (EOFError, OSError, wave.Error) as exc:
        raise RealAcousticFeatureError(
            f"could not decode complete four-channel master WAV {resolved}"
        ) from exc

    after_hash, after_size = _sha256_file(resolved)
    if (after_hash, after_size) != (before_hash, before_size):
        _fail("master source changed while the equal-weight mix was decoded")
    if pcm is None or pcm.size != declared_frames or not np.isfinite(pcm).all():
        _fail("equal-weight master mix is empty, non-finite, or incomplete")
    replay = {
        "path": str(resolved),
        "sha256": before_hash,
        "size_bytes": before_size,
        "input_dtype": "little_endian_signed_pcm16",
        "mix_arithmetic_dtype": "float64",
        "output_dtype": "float64",
        "channel_count": MASTER_CHANNELS,
        "sample_rate_hz": MASTER_SAMPLE_RATE_HZ,
        "decoded_frame_count": int(pcm.size),
        "mix_rule": MIX_RULE,
        "intermediate_pcm16_quantization": False,
        "channel_order_invariant": True,
    }
    return np.ascontiguousarray(pcm), replay


def _base_ticks(dataset: assembly._DatasetFile, scene: str) -> np.ndarray:
    provenance = dataset.provenance
    if type(provenance) is not dict:
        _fail("analysis base has no real-data provenance")
    if provenance.get("artifact_kind") != analysis_base.ARTIFACT_KIND:
        _fail("real acoustic preflight requires a canonical analysis-base parent")
    if provenance.get("session_id") != scene:
        _fail("analysis-base session_id does not match the requested scene")
    tick_range = provenance.get("output_tick_range")
    if type(tick_range) is not dict or set(tick_range) != {
        "start_inclusive",
        "end_exclusive",
    }:
        _fail("analysis-base output_tick_range is malformed")
    start = _exact_int(tick_range["start_inclusive"], "analysis tick start", minimum=0)
    end = _exact_int(tick_range["end_exclusive"], "analysis tick end", minimum=1)
    if (start, end) != analysis_base.EXPECTED_TICK_RANGES[scene] or start >= end:
        _fail("analysis-base ticks differ from the frozen ten-scene matrix")
    ticks = np.arange(start, end, dtype=np.int64)
    frame = dataset.frame_df
    if len(frame) != ticks.size:
        _fail("analysis-base row count does not match its exact tick interval")
    expected_session = np.full(ticks.size, scene, dtype=object)
    if not np.array_equal(
        frame["session_id"].to_numpy(dtype=object),
        expected_session,
    ):
        _fail("analysis-base session keys are not constant and canonical")
    expected_time = ticks.astype(np.float64) / FRAME_RATE_HZ
    if not np.array_equal(frame["timestamp_s"].to_numpy(dtype=np.float64), expected_time):
        _fail("analysis-base timestamps do not exactly equal integer tick / 25")
    return ticks


def _feature_frame(
    dataset: assembly._DatasetFile,
    features: acoustic_features.AcousticFeatureEvidence,
) -> pd.DataFrame:
    if not bool(features.model_input_valid.all()):
        _fail("an analysis-base tick lacks complete real causal acoustic history")
    for name in FEATURE_COLUMNS:
        values = getattr(features, name)
        if not np.isfinite(values).all():
            _fail(f"formal preflight feature {name} contains a non-finite value")
    if not np.isin(features.voiced, (0.0, 1.0)).all():
        _fail("formal preflight voiced values are not binary")
    if (features.f0_proxy[features.voiced == 0.0] != 0.0).any():
        _fail("formal preflight f0_proxy is non-zero on an unvoiced row")
    return pd.DataFrame(
        {
            "session_id": pd.Series(
                dataset.frame_df["session_id"].to_numpy(dtype=object),
                dtype=object,
            ),
            "timestamp_s": dataset.frame_df["timestamp_s"].to_numpy(dtype=np.float64),
            "array_energy": features.array_energy,
            "f0_proxy": features.f0_proxy,
            "voiced": features.voiced,
        }
    )


def _validate_evidence_contract(
    features: acoustic_features.AcousticFeatureEvidence,
    ticks: np.ndarray,
) -> None:
    if type(features) is not acoustic_features.AcousticFeatureEvidence:
        _fail("acoustic extractor returned the wrong exact evidence type")
    expected_shape = ticks.shape
    for name, expected_dtype in _EVIDENCE_DTYPES.items():
        value = getattr(features, name)
        if type(value) is not np.ndarray:
            _fail(f"acoustic evidence {name} is not an exact NumPy array")
        if value.shape != expected_shape:
            _fail(f"acoustic evidence {name} shape differs from the analysis ticks")
        if value.dtype != expected_dtype:
            _fail(f"acoustic evidence {name} dtype differs from the frozen contract")
    if not np.array_equal(features.tick_index, ticks):
        _fail("acoustic extractor returned different tick keys")
    expected_time = ticks.astype(np.float64) / FRAME_RATE_HZ
    if not np.array_equal(features.source_local_timestamp_s, expected_time):
        _fail("acoustic evidence timestamps do not exactly equal tick / 25")


def _report(
    *,
    scene: str,
    dataset: assembly._DatasetFile,
    ticks: np.ndarray,
    frame: pd.DataFrame,
    audio_report: dict[str, Any],
    audio_manifest: Path,
    audio_manifest_sha256: str,
    audio_manifest_size: int,
    master_replay: dict[str, Any],
    features: acoustic_features.AcousticFeatureEvidence,
) -> dict[str, Any]:
    provenance = dataset.provenance
    assert provenance is not None
    sources = audio_report.get("sources")
    if type(sources) is not dict:
        _fail("audio contract report has no exact sources object")
    master_source = _master_source_contract(sources.get("master"))
    return {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "parameter_status": _parameter_gate(),
        "source_selection": _source_selection_contract(),
        "axis": AXIS,
        "analysis_base": {
            "path": str(dataset.path.resolve()),
            "sha256": dataset.data_sha256,
            "provenance_path": str(dataset.path.with_suffix(".provenance.json").resolve()),
            "provenance_sha256": dataset.provenance_sha256,
            "artifact_kind": provenance["artifact_kind"],
            "tick_start_inclusive": int(ticks[0]),
            "tick_end_exclusive": int(ticks[-1]) + 1,
            "n_rows": int(ticks.size),
            "key_schema": _analysis_key_schema(frame),
        },
        "participant_to_slot": provenance["participant_to_slot"],
        "audio_contract": {
            "path": str(audio_manifest),
            "sha256": audio_manifest_sha256,
            "size_bytes": audio_manifest_size,
            "artifact_kind": audio_report["artifact_kind"],
            "format_version": audio_report["format_version"],
            "master_exact_permutation": audio_report["master_sample_replay"][
                "exact_permutation"
            ],
            "master_source": master_source,
        },
        "master_mix_replay": master_replay,
        "extractor": acoustic_features.extractor_parameters(),
        "causality": _preflight_causality_contract(),
        "validity": {
            "energy_valid_rows": int(features.energy_valid.sum()),
            "pitch_valid_rows": int(features.pitch_valid.sum()),
            "all_model_input_rows_valid": True,
            "missing_rows_zero_filled": False,
        },
        "feature_arrays": _feature_array_contract(frame),
    }
def derive_real_acoustic_preflight(
    scene: str,
    *,
    authority: RealAcousticAuthority | None = None,
) -> RealAcousticPreflight:
    """Fully replay one scene and derive non-publishable real acoustic inputs."""

    selected = default_real_acoustic_authority() if authority is None else authority
    if type(selected) is not RealAcousticAuthority:
        _fail("real acoustic authority has the wrong exact type")
    dataset = _load_analysis_dataset(scene, authority=selected)
    ticks = _base_ticks(dataset, scene)
    audio_report, manifest, manifest_hash, manifest_size = _load_audio_report(
        scene,
        authority=selected,
    )
    paths = audio_contract.canonical_source_paths(scene, selected.audio)
    master_binding = audio_report.get("sources", {}).get("master")
    if type(master_binding) is not dict:
        _fail("audio contract has no canonical master source binding")
    pcm, master_replay = _read_equal_weight_master(
        paths["master"],
        binding=master_binding,
    )
    features = acoustic_features.extract_causal_acoustic_features(
        pcm,
        MASTER_SAMPLE_RATE_HZ,
        tick_indices=ticks,
    )
    _validate_evidence_contract(features, ticks)
    frame = _feature_frame(dataset, features)
    report = _report(
        scene=scene,
        dataset=dataset,
        ticks=ticks,
        frame=frame,
        audio_report=audio_report,
        audio_manifest=manifest,
        audio_manifest_sha256=manifest_hash,
        audio_manifest_size=manifest_size,
        master_replay=master_replay,
        features=features,
    )
    # The source contract performs its own final checks.  Recheck the two
    # bytes that directly own this preflight after feature computation.
    if _sha256_file(manifest) != (manifest_hash, manifest_size):
        _fail("audio contract changed during real acoustic feature extraction")
    master_hash, master_size = _sha256_file(paths["master"])
    if (master_hash, master_size) != (
        master_replay["sha256"],
        master_replay["size_bytes"],
    ):
        _fail("master source changed during real acoustic feature extraction")
    return RealAcousticPreflight(scene=scene, feature_df=frame, report=report)


def validate_frozen_preflight_matrix(
    results: dict[str, RealAcousticPreflight],
) -> None:
    """Require exact ten-scene coverage and row total without approving parameters."""

    if set(results) != set(analysis_base.EXPECTED_TICK_RANGES):
        _fail("real acoustic preflight matrix does not exactly cover the ten scenes")
    total = 0
    for scene, (start, end) in analysis_base.EXPECTED_TICK_RANGES.items():
        result = results[scene]
        if result.scene != scene or len(result.feature_df) != end - start:
            _fail(f"real acoustic preflight row count changed for {scene}")
        if result.report.get("assembly_eligible") is not False:
            _fail("a preflight result cannot claim assembly eligibility")
        total += len(result.feature_df)
    if total != analysis_base.EXPECTED_TOTAL_ROWS:
        _fail("real acoustic preflight total differs from 78,582 rows")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _type_exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _type_exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _type_exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _parameter_gate() -> dict[str, Any]:
    return {
        "status": PARAMETER_STATUS,
        "parameter": "normalized_autocorrelation_voicing_threshold",
        "candidate_value": acoustic_features.VOICING_THRESHOLD,
        "blocker": "explicit owner freeze before formal feature publication",
    }


def _promotion_gate() -> dict[str, Any]:
    return {
        "owner_freeze_token": None,
        "owner_freeze_supported_in_this_format": False,
        "independent_new_review_required": True,
        "promotion_implemented": False,
    }


def _json_payload(value: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RealAcousticFeatureError("scaffold JSON is not finite and serialisable") from exc
    return text.encode("utf-8")


def _strict_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        _fail(f"{label} payload must be exact immutable bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RealAcousticFeatureError(f"{label} is not strict UTF-8") from exc

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
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise RealAcousticFeatureError(f"{label} is not valid JSON") from exc

    def reject_nested_nonfinite(item: Any, path: str) -> None:
        if type(item) is float and not math.isfinite(item):
            _fail(f"{label} contains forbidden non-finite number at {path}")
        if type(item) is dict:
            for key, nested in item.items():
                reject_nested_nonfinite(nested, f"{path}.{key}")
        elif type(item) is list:
            for index, nested in enumerate(item):
                reject_nested_nonfinite(nested, f"{path}[{index}]")

    reject_nested_nonfinite(value, "$")
    if type(value) is not dict:
        _fail(f"{label} must contain one JSON object")
    return value


def _csv_payload(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _strict_csv_payload(payload: bytes, label: str) -> pd.DataFrame:
    if type(payload) is not bytes:
        _fail(f"{label} payload must be exact immutable bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RealAcousticFeatureError(f"{label} is not strict UTF-8") from exc
    try:
        header = next(csv.reader(io.StringIO(text, newline="")), [])
    except csv.Error as exc:
        raise RealAcousticFeatureError(f"{label} has an invalid CSV header") from exc
    expected = ["session_id", "timestamp_s", *FEATURE_COLUMNS]
    if header != expected or len(header) != len(set(header)):
        _fail(f"{label} columns must exactly equal {expected}")
    expected_dtypes = {
        "session_id": np.dtype(object),
        "timestamp_s": np.dtype(np.float64),
        **{column: np.dtype(np.float64) for column in FEATURE_COLUMNS},
    }
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            dtype=expected_dtypes,
            float_precision="round_trip",
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
        )
    except Exception as exc:
        raise RealAcousticFeatureError(f"{label} could not be parsed as CSV") from exc
    if _csv_payload(frame) != payload:
        _fail(f"{label} is not the complete canonical CSV byte encoding")
    return frame


def _validate_scaffold_frame(frame: pd.DataFrame, scene: str) -> np.ndarray:
    expected_columns = ["session_id", "timestamp_s", *FEATURE_COLUMNS]
    if type(frame) is not pd.DataFrame or list(frame.columns) != expected_columns:
        _fail("scaffold feature columns differ from the exact global-acoustic schema")
    if len(frame) == 0:
        _fail("scaffold feature frame must contain at least one row")
    expected_dtypes = {
        "session_id": np.dtype(object),
        "timestamp_s": np.dtype(np.float64),
        **{column: np.dtype(np.float64) for column in FEATURE_COLUMNS},
    }
    for column, expected_dtype in expected_dtypes.items():
        if frame[column].dtype != expected_dtype:
            _fail(
                f"scaffold column {column} must have exact dtype {expected_dtype}; "
                "bool/int/string coercion is forbidden"
            )
    sessions = frame["session_id"].to_numpy(copy=False)
    if any(type(value) is not str for value in sessions):
        _fail("scaffold session_id object values must all be exact strings")
    if not np.array_equal(sessions, np.full(len(frame), scene, dtype=object)):
        _fail("scaffold session keys are not constant and canonical")
    times = frame["timestamp_s"].to_numpy(copy=False)
    values = {column: frame[column].to_numpy(copy=False) for column in FEATURE_COLUMNS}
    if not np.isfinite(times).all() or not all(
        np.isfinite(value).all() for value in values.values()
    ):
        _fail("scaffold timestamps/features contain NaN or infinity")
    scaled = times * FRAME_RATE_HZ
    rounded = np.rint(scaled)
    if (rounded < 0).any() or (rounded > np.iinfo(np.int64).max).any():
        _fail("scaffold timestamps are outside the supported non-negative tick range")
    ticks = rounded.astype(np.int64)
    expected_times = ticks.astype(np.float64) / FRAME_RATE_HZ
    if _sha256_array(times) != _sha256_array(expected_times):
        _fail("scaffold timestamps do not exactly equal integer tick / 25")
    if len(ticks) > 1 and not np.array_equal(np.diff(ticks), np.ones(len(ticks) - 1)):
        _fail("scaffold ticks must be contiguous, increasing, and unique")
    voiced = values["voiced"]
    if not np.isin(voiced, (0.0, 1.0)).all():
        _fail("scaffold voiced values must be binary")
    if (values["f0_proxy"][voiced == 0.0] != 0.0).any():
        _fail("scaffold f0_proxy must be zero on unvoiced rows")
    return ticks


def _column_array_sha256(value: np.ndarray) -> str:
    if value.dtype != np.dtype(object):
        return _sha256_array(value)
    digest = hashlib.sha256()
    for item in value.tolist():
        if type(item) is not str:
            _fail("object-array hashing accepts only exact strings")
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _frame_column_contract(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        column: {
            "dtype": str(frame[column].dtype),
            "shape": list(frame[column].shape),
            "sha256": _column_array_sha256(frame[column].to_numpy(copy=False)),
        }
        for column in frame.columns
    }


def _frame_contract(frame: pd.DataFrame, ticks: np.ndarray) -> dict[str, Any]:
    return {
        "columns": list(frame.columns),
        "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
        "n_rows": len(frame),
        "tick_start_inclusive": int(ticks[0]),
        "tick_end_exclusive": int(ticks[-1]) + 1,
        "column_arrays": _frame_column_contract(frame),
    }


def _require_report_path(value: Any, label: str, expected_name: str) -> Path:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be a non-empty canonical path string")
    path = Path(value)
    if not path.is_absolute() or path.name != expected_name:
        _fail(f"{label} does not name the expected absolute source")
    return path


def _validate_analysis_report(
    value: Any,
    *,
    scene: str,
    frame: pd.DataFrame,
    ticks: np.ndarray,
) -> None:
    expected_keys = {
        "path",
        "sha256",
        "provenance_path",
        "provenance_sha256",
        "artifact_kind",
        "tick_start_inclusive",
        "tick_end_exclusive",
        "n_rows",
        "key_schema",
    }
    if type(value) is not dict or set(value) != expected_keys:
        _fail("preflight analysis_base has the wrong exact schema")
    data_path = _require_report_path(value["path"], "analysis_base.path", f"{scene}.csv")
    provenance_path = _require_report_path(
        value["provenance_path"],
        "analysis_base.provenance_path",
        f"{scene}.provenance.json",
    )
    if provenance_path != data_path.with_suffix(".provenance.json"):
        _fail("preflight analysis-base data/provenance paths are inconsistent")
    _sha256_digest(value["sha256"], "analysis_base.sha256")
    _sha256_digest(value["provenance_sha256"], "analysis_base.provenance_sha256")
    expected = {
        "artifact_kind": analysis_base.ARTIFACT_KIND,
        "tick_start_inclusive": int(ticks[0]),
        "tick_end_exclusive": int(ticks[-1]) + 1,
        "n_rows": len(ticks),
        "key_schema": _analysis_key_schema(frame),
    }
    actual = {key: value[key] for key in expected}
    if not _type_exact_json_equal(actual, expected):
        _fail("preflight analysis_base does not bind the exact source-frame schema")


def _validate_audio_report(value: Any, scene: str) -> dict[str, Any]:
    expected_keys = {
        "path",
        "sha256",
        "size_bytes",
        "artifact_kind",
        "format_version",
        "master_exact_permutation",
        "master_source",
    }
    if type(value) is not dict or set(value) != expected_keys:
        _fail("preflight audio_contract has the wrong exact schema")
    _require_report_path(
        value["path"],
        "audio_contract.path",
        f"{scene}.audio_contract.json",
    )
    _sha256_digest(value["sha256"], "audio_contract.sha256")
    _exact_int(value["size_bytes"], "audio_contract.size_bytes", minimum=1)
    expected_identity = {
        "artifact_kind": audio_contract.ARTIFACT_KIND,
        "format_version": audio_contract.FORMAT_VERSION,
        "master_exact_permutation": True,
    }
    actual_identity = {key: value[key] for key in expected_identity}
    if not _type_exact_json_equal(actual_identity, expected_identity):
        _fail("preflight audio_contract identity/permutation contract changed")

    master = _master_source_contract(value["master_source"])
    _require_report_path(master["path"], "master_source.path", "master_aligned.wav")
    _sha256_digest(master["sha256"], "master_source.sha256")
    _exact_int(master["size_bytes"], "master_source.size_bytes", minimum=1)
    header = master["wave_header"]
    n_frames = _exact_int(header["n_frames"], "master_source.n_frames", minimum=1)
    expected_header = {
        "channels": MASTER_CHANNELS,
        "sample_width_bytes": MASTER_SAMPLE_WIDTH_BYTES,
        "sample_rate_hz": MASTER_SAMPLE_RATE_HZ,
        "n_frames": n_frames,
        "compression_type": "NONE",
    }
    if not _type_exact_json_equal(header, expected_header):
        _fail("preflight master WAV header changed exact type or value")
    expected_payload = {
        "frame_size_bytes": MASTER_CHANNELS * MASTER_SAMPLE_WIDTH_BYTES,
        "decoded_frames": n_frames,
        "frame_aligned": True,
        "complete": True,
    }
    if master["media_type"] != "audio/wav" or not _type_exact_json_equal(
        master["pcm_payload"], expected_payload
    ):
        _fail("preflight master source media/PCM schema is incomplete")
    return master


def _validate_master_replay(value: Any, master: dict[str, Any]) -> None:
    expected = {
        "path": master["path"],
        "sha256": master["sha256"],
        "size_bytes": master["size_bytes"],
        "input_dtype": "little_endian_signed_pcm16",
        "mix_arithmetic_dtype": "float64",
        "output_dtype": "float64",
        "channel_count": MASTER_CHANNELS,
        "sample_rate_hz": MASTER_SAMPLE_RATE_HZ,
        "decoded_frame_count": master["wave_header"]["n_frames"],
        "mix_rule": MIX_RULE,
        "intermediate_pcm16_quantization": False,
        "channel_order_invariant": True,
    }
    if not _type_exact_json_equal(value, expected):
        _fail("preflight master replay does not exactly bind path/hash/header/schema")


def _validate_preflight_for_scaffold(
    preflight: RealAcousticPreflight,
    ticks: np.ndarray,
) -> dict[str, str]:
    if type(preflight) is not RealAcousticPreflight:
        _fail("scaffold input must be an exact RealAcousticPreflight")
    report = preflight.report
    expected_keys = {
        "artifact_kind",
        "format_version",
        "scene",
        "output_role",
        "assembly_eligible",
        "counted_input_ready",
        "parameter_status",
        "source_selection",
        "axis",
        "analysis_base",
        "participant_to_slot",
        "audio_contract",
        "master_mix_replay",
        "extractor",
        "causality",
        "validity",
        "feature_arrays",
    }
    if type(report) is not dict or set(report) != expected_keys:
        _fail("preflight report has the wrong complete exact schema")
    expected_identity = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": preflight.scene,
        "output_role": OUTPUT_ROLE,
        "assembly_eligible": False,
        "counted_input_ready": False,
    }
    actual_identity = {key: report[key] for key in expected_identity}
    if not _type_exact_json_equal(actual_identity, expected_identity):
        _fail("scaffold input is not the supported real acoustic preflight")
    if not _type_exact_json_equal(report["parameter_status"], _parameter_gate()):
        _fail("preflight does not preserve the owner-pending 0.45 parameter gate")
    if not _type_exact_json_equal(
        report["source_selection"], _source_selection_contract()
    ):
        _fail("preflight does not preserve the approved source/mix boundary")
    if report["axis"] != AXIS:
        _fail("preflight does not use the mic-A shared axis")
    _validate_analysis_report(
        report["analysis_base"],
        scene=preflight.scene,
        frame=preflight.feature_df,
        ticks=ticks,
    )
    if not _type_exact_json_equal(report["participant_to_slot"], PARTICIPANT_TO_SLOT):
        _fail("preflight participant_to_slot differs from the canonical A/B/C/R mapping")
    master = _validate_audio_report(report["audio_contract"], preflight.scene)
    _validate_master_replay(report["master_mix_replay"], master)
    if not _type_exact_json_equal(
        report["extractor"], acoustic_features.extractor_parameters()
    ):
        _fail("preflight extractor contract changed exact type or value")
    if not _type_exact_json_equal(
        report["causality"], _preflight_causality_contract()
    ):
        _fail("preflight does not preserve the frozen causal extraction contract")
    expected_validity = {
        "energy_valid_rows": len(ticks),
        "pitch_valid_rows": len(ticks),
        "all_model_input_rows_valid": True,
        "missing_rows_zero_filled": False,
    }
    if not _type_exact_json_equal(report["validity"], expected_validity):
        _fail("preflight does not prove complete non-padding feature validity")
    if not _type_exact_json_equal(
        report["feature_arrays"], _feature_array_contract(preflight.feature_df)
    ):
        _fail("preflight feature_arrays do not bind dtype/shape/hash/value replay")
    return dict(PARTICIPANT_TO_SLOT)


def _strict_replay_preflight_for_scaffold(
    preflight: RealAcousticPreflight,
) -> tuple[RealAcousticPreflight, np.ndarray, dict[str, str]]:
    if type(preflight) is not RealAcousticPreflight:
        _fail("scaffold input must be an exact RealAcousticPreflight")
    if type(preflight.scene) is not str or not preflight.scene:
        _fail("preflight scene must be a non-empty string")
    if type(preflight.feature_df) is not pd.DataFrame:
        _fail("preflight feature frame must be an exact pandas DataFrame")
    frame = preflight.feature_df.copy(deep=True)
    ticks = _validate_scaffold_frame(frame, preflight.scene)
    report_payload = _json_payload(preflight.report)
    report = _strict_json_payload(report_payload, "preflight report")
    if not _type_exact_json_equal(preflight.report, report):
        _fail("preflight report does not survive strict JSON type/value replay")
    replayed = RealAcousticPreflight(
        scene=preflight.scene,
        feature_df=frame,
        report=report,
    )
    mapping = _validate_preflight_for_scaffold(replayed, ticks)
    return replayed, ticks, mapping


def _preflight_report_binding(report: dict[str, Any]) -> dict[str, Any]:
    payload = _json_payload(report)
    return {
        "payload_id": "embedded_preflight_report",
        "sha256": _sha256_bytes(payload),
        "byte_length": len(payload),
    }


def build_read_only_artifact_scaffold(
    scene: str,
    *,
    authority: RealAcousticAuthority | None = None,
) -> ReadOnlyAcousticArtifactScaffold:
    """Build non-publishable, assembly-rejected payloads entirely in memory.

    This is deliberately a fixture/review scaffold.  The caller supplies only
    a frozen scene and optional code-owned authority; source replay is always
    fresh.  There is no output path, owner token, write branch, or promotion
    branch in format v1.
    """

    if type(scene) is not str or scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail("scaffold scene is outside the frozen analysis inventory")
    preflight = derive_real_acoustic_preflight(scene, authority=authority)
    replayed, ticks, mapping = _strict_replay_preflight_for_scaffold(preflight)
    frame = replayed.feature_df
    source_payload = _csv_payload(frame)
    feature_payload = _csv_payload(frame.copy(deep=True))
    source_binding = {
        "payload_id": "source_csv",
        "sha256": _sha256_bytes(source_payload),
        "byte_length": len(source_payload),
    }
    evidence = {
        "artifact_kind": SCAFFOLD_EVIDENCE_KIND,
        "format_version": SCAFFOLD_FORMAT_VERSION,
        "scene": replayed.scene,
        "output_role": "read_only_in_memory_evidence_scaffold",
        "assembly_eligible": False,
        "counted_input_ready": False,
        "parameter_status": _parameter_gate(),
        "promotion_gate": _promotion_gate(),
        "source_csv": source_binding,
        "frame_contract": _frame_contract(frame, ticks),
        "preflight_report_binding": _preflight_report_binding(replayed.report),
        "preflight_report": replayed.report,
    }
    evidence_payload = _json_payload(evidence)
    source_path = "payload:source_csv"
    source_hash = source_binding["sha256"]
    column_sources = {
        column: {
            "participant": None,
            "slot": None,
            "source_path": source_path,
            "source_sha256": source_hash,
            "source_column": column,
            "source_participant": None,
            "source_keys": {
                "session_id": "session_id",
                "timestamp_s": "timestamp_s",
            },
            "value_transform": {"kind": "identity"},
        }
        for column in FEATURE_COLUMNS
    }
    manifest = {
        "artifact_kind": SCAFFOLD_MANIFEST_KIND,
        "format_version": SCAFFOLD_FORMAT_VERSION,
        "target_contract": {
            "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
            "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
            "status": "preview_only_not_emitted",
        },
        "session_id": replayed.scene,
        "participant_to_slot": mapping,
        "supplied_columns": list(FEATURE_COLUMNS),
        "feature_csv_sha256": _sha256_bytes(feature_payload),
        "source_files": [
            {
                "path": source_path,
                "sha256": source_hash,
                "byte_length": len(source_payload),
            },
            {
                "path": "payload:evidence_json",
                "sha256": _sha256_bytes(evidence_payload),
                "byte_length": len(evidence_payload),
            },
        ],
        "purpose": "model_input",
        "causality": _manifest_causality_contract(),
        "column_sources": column_sources,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "parameter_status": _parameter_gate(),
        "promotion_gate": _promotion_gate(),
    }
    return ReadOnlyAcousticArtifactScaffold(
        scene=replayed.scene,
        source_csv_payload=source_payload,
        evidence_json_payload=evidence_payload,
        feature_csv_payload=feature_payload,
        feature_manifest_json_payload=_json_payload(manifest),
    )


def strict_replay_read_only_artifact_scaffold(
    scaffold: ReadOnlyAcousticArtifactScaffold,
) -> LoadedReadOnlyAcousticArtifactScaffold:
    """Strictly replay every in-memory byte without accessing a source path."""

    if type(scaffold) is not ReadOnlyAcousticArtifactScaffold:
        _fail("strict replay requires an exact read-only scaffold")
    if type(scaffold.scene) is not str or not scaffold.scene:
        _fail("scaffold scene must be a non-empty string")
    source = _strict_csv_payload(scaffold.source_csv_payload, "source CSV scaffold")
    feature = _strict_csv_payload(scaffold.feature_csv_payload, "feature CSV scaffold")
    source_ticks = _validate_scaffold_frame(source, scaffold.scene)
    feature_ticks = _validate_scaffold_frame(feature, scaffold.scene)
    if not np.array_equal(source_ticks, feature_ticks) or not _type_exact_json_equal(
        _frame_column_contract(source), _frame_column_contract(feature)
    ):
        _fail(
            "feature CSV does not bitwise replay every typed independent source column"
        )
    evidence = _strict_json_payload(
        scaffold.evidence_json_payload,
        "evidence JSON scaffold",
    )
    expected_evidence_keys = {
        "artifact_kind",
        "format_version",
        "scene",
        "output_role",
        "assembly_eligible",
        "counted_input_ready",
        "parameter_status",
        "promotion_gate",
        "source_csv",
        "frame_contract",
        "preflight_report_binding",
        "preflight_report",
    }
    if set(evidence) != expected_evidence_keys:
        _fail("evidence scaffold keys differ from format v1")
    if (
        evidence["artifact_kind"] != SCAFFOLD_EVIDENCE_KIND
        or type(evidence["format_version"]) is not int
        or evidence["format_version"] != SCAFFOLD_FORMAT_VERSION
        or evidence["scene"] != scaffold.scene
        or evidence["output_role"] != "read_only_in_memory_evidence_scaffold"
    ):
        _fail("evidence scaffold identity/version is unsupported")
    if evidence["assembly_eligible"] is not False:
        _fail("evidence scaffold cannot claim assembly eligibility")
    if evidence["counted_input_ready"] is not False:
        _fail("evidence scaffold cannot claim counted-input readiness")
    if not _type_exact_json_equal(evidence["parameter_status"], _parameter_gate()):
        _fail("evidence scaffold changed the owner-pending 0.45 parameter gate")
    if not _type_exact_json_equal(evidence["promotion_gate"], _promotion_gate()):
        _fail("evidence scaffold cannot add or activate an owner token")
    expected_source_binding = {
        "payload_id": "source_csv",
        "sha256": _sha256_bytes(scaffold.source_csv_payload),
        "byte_length": len(scaffold.source_csv_payload),
    }
    if not _type_exact_json_equal(evidence["source_csv"], expected_source_binding):
        _fail("evidence scaffold does not hash-bind the source CSV payload")
    expected_frame_contract = _frame_contract(source, source_ticks)
    if not _type_exact_json_equal(evidence["frame_contract"], expected_frame_contract):
        _fail("evidence scaffold frame contract differs from exact CSV replay")
    replay_preflight = RealAcousticPreflight(
        scene=scaffold.scene,
        feature_df=source,
        report=evidence["preflight_report"],
    )
    replayed, replayed_ticks, _ = _strict_replay_preflight_for_scaffold(
        replay_preflight
    )
    if not np.array_equal(replayed_ticks, source_ticks):
        _fail("preflight report replay changed the exact source ticks")
    expected_report_binding = _preflight_report_binding(replayed.report)
    if not _type_exact_json_equal(
        evidence["preflight_report_binding"], expected_report_binding
    ):
        _fail("evidence scaffold does not hash-bind the complete preflight report")

    manifest = _strict_json_payload(
        scaffold.feature_manifest_json_payload,
        "feature manifest scaffold",
    )
    expected_manifest_keys = {
        "artifact_kind",
        "format_version",
        "target_contract",
        "session_id",
        "participant_to_slot",
        "supplied_columns",
        "feature_csv_sha256",
        "source_files",
        "purpose",
        "causality",
        "column_sources",
        "assembly_eligible",
        "counted_input_ready",
        "parameter_status",
        "promotion_gate",
    }
    if set(manifest) != expected_manifest_keys:
        _fail("feature manifest scaffold keys differ from format v1")
    if manifest["artifact_kind"] == assembly.FEATURE_MANIFEST_KIND:
        _fail("read-only scaffold must never use the assembly-accepted manifest kind")
    if (
        manifest["artifact_kind"] != SCAFFOLD_MANIFEST_KIND
        or type(manifest["format_version"]) is not int
        or manifest["format_version"] != SCAFFOLD_FORMAT_VERSION
        or manifest["session_id"] != scaffold.scene
    ):
        _fail("feature manifest scaffold identity/version is unsupported")
    expected_target = {
        "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
        "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
        "status": "preview_only_not_emitted",
    }
    if not _type_exact_json_equal(manifest["target_contract"], expected_target):
        _fail("feature manifest target_contract is not a preview-only v3 declaration")
    if manifest["assembly_eligible"] is not False:
        _fail("feature manifest scaffold cannot claim assembly eligibility")
    if manifest["counted_input_ready"] is not False:
        _fail("feature manifest scaffold cannot claim counted-input readiness")
    if not _type_exact_json_equal(manifest["parameter_status"], _parameter_gate()):
        _fail("feature manifest changed the owner-pending 0.45 parameter gate")
    if not _type_exact_json_equal(manifest["promotion_gate"], _promotion_gate()):
        _fail("feature manifest scaffold cannot add or activate an owner token")
    if not _type_exact_json_equal(manifest["participant_to_slot"], PARTICIPANT_TO_SLOT):
        _fail("feature manifest participant_to_slot is not canonical")
    if not _type_exact_json_equal(manifest["supplied_columns"], list(FEATURE_COLUMNS)):
        _fail("feature manifest supplied_columns differ from the acoustic block")
    if manifest["feature_csv_sha256"] != _sha256_bytes(scaffold.feature_csv_payload):
        _fail("feature manifest does not hash-bind the feature CSV payload")
    expected_source_files = [
        {
            "path": "payload:source_csv",
            "sha256": _sha256_bytes(scaffold.source_csv_payload),
            "byte_length": len(scaffold.source_csv_payload),
        },
        {
            "path": "payload:evidence_json",
            "sha256": _sha256_bytes(scaffold.evidence_json_payload),
            "byte_length": len(scaffold.evidence_json_payload),
        },
    ]
    if not _type_exact_json_equal(manifest["source_files"], expected_source_files):
        _fail("feature manifest source_files do not bind both in-memory source payloads")
    if manifest["purpose"] != "model_input" or not _type_exact_json_equal(
        manifest["causality"], _manifest_causality_contract()
    ):
        _fail("feature manifest does not preserve causal model-input semantics")
    expected_column_sources = {
        column: {
            "participant": None,
            "slot": None,
            "source_path": "payload:source_csv",
            "source_sha256": _sha256_bytes(scaffold.source_csv_payload),
            "source_column": column,
            "source_participant": None,
            "source_keys": {
                "session_id": "session_id",
                "timestamp_s": "timestamp_s",
            },
            "value_transform": {"kind": "identity"},
        }
        for column in FEATURE_COLUMNS
    }
    if not _type_exact_json_equal(manifest["column_sources"], expected_column_sources):
        _fail("feature manifest column_sources fail the global identity preview contract")
    return LoadedReadOnlyAcousticArtifactScaffold(
        scene=scaffold.scene,
        source_df=source,
        evidence=evidence,
        feature_df=feature,
        feature_manifest=manifest,
    )


__all__ = [
    "ARTIFACT_KIND",
    "FORMAT_VERSION",
    "LoadedReadOnlyAcousticArtifactScaffold",
    "ReadOnlyAcousticArtifactScaffold",
    "RealAcousticAuthority",
    "RealAcousticFeatureError",
    "build_read_only_artifact_scaffold",
    "default_real_acoustic_authority",
    "derive_real_acoustic_preflight",
    "strict_replay_read_only_artifact_scaffold",
    "validate_frozen_preflight_matrix",
]
