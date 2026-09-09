"""Review-gated formal causal-acoustic artifacts for the real corpus.

The owner froze the N-A-C-F voicing threshold at 0.45 on 2026-08-16 and
accepted a trusted local single-user/single-writer publication boundary.  This
module turns the already accepted read-only source replay into a separately
versioned artifact contract without silently declaring the implementation
reviewed.

``build_pending_acoustic_scene`` publishes a source CSV, evidence JSON,
feature CSV, and a *pending* manifest.  The pending manifest deliberately does
not use :data:`assembly.FEATURE_MANIFEST_KIND`, and the assembler therefore
rejects it.  An independently authored, canonical review receipt is required
before ``seal_reviewed_acoustic_scene`` can publish the feature CSV's sibling
assembly-v3 manifest.  No API in this module writes or self-approves a receipt.

Raw build/re-audit always replays the complete four-channel master through the
causal extractor.  Ordinary downstream loading is sealed: it re-hashes all
bound artifacts and live source bytes but does not decode the multi-GB WAV or
recompute features.  This is integrity and reproducibility under the recorded
single-writer trust boundary, not protection against a malicious process that
can replace trusted code and every artifact together.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
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
    real_acoustic_features,
)

FORMAT_VERSION = 1
EVIDENCE_ARTIFACT_KIND = "vap_real_causal_acoustic_evidence"
PENDING_MANIFEST_KIND = "vap_feature_source_review_pending"
REVIEW_RECEIPT_KIND = "vap_real_causal_acoustic_review_receipt"
OUTPUT_ROLE = "formal_model_input_pending_independent_review"
REVIEWED_OUTPUT_ROLE = "formal_model_input_independently_reviewed"
REVIEW_STATUS_PENDING = "review_pending"
REVIEW_STATUS_ACCEPTED = "accepted"
CANONICAL_PRODUCTION_SCOPE = "canonical_production_counted"
NON_COUNTED_AUTHORITY_SCOPE = "noncanonical_fixture_non_counted"
VOICING_THRESHOLD = 0.45
FEATURE_COLUMNS = real_acoustic_features.FEATURE_COLUMNS
KEY_COLUMNS = assembly.KEY_COLUMNS
FRAME_COLUMNS = (*KEY_COLUMNS, *FEATURE_COLUMNS)
REVIEW_SCOPE = (
    "canonical_source_timeline_and_mix_binding",
    "causal_extractor_and_threshold_contract",
    "artifact_type_hash_and_replay_gates",
    "assembly_v3_manifest_conformance",
    "trusted_local_single_user_single_writer_boundary",
)
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FormalAcousticFeatureError(ValueError):
    """A formal acoustic source, artifact, or review gate failed closed."""


def _real_authority_snapshot(
    authority: real_acoustic_features.RealAcousticAuthority,
) -> tuple[object, ...]:
    """Freeze every production-relevant nested authority field by value."""

    analysis = authority.analysis
    audio = authority.audio
    return (
        analysis.pose_base_root,
        analysis.gold_alignment_root,
        analysis.audio_manifest_root,
        analysis.analysis_base_root,
        analysis.descendant_output_paths,
        analysis.enforce_frozen_inventory,
        audio.corpus_root,
        audio.hpe_delivery_root,
    )


# Capture the code-owned source binding once, before test/fixture seams can replace
# ``default_real_acoustic_authority``.  The separate value snapshot means actual
# mutation of a captured nested field still fails the production comparison.
_CANONICAL_REAL_ACOUSTIC_AUTHORITY = (
    real_acoustic_features.default_real_acoustic_authority()
)
_CANONICAL_REAL_AUTHORITY_SNAPSHOT = _real_authority_snapshot(
    _CANONICAL_REAL_ACOUSTIC_AUTHORITY
)


@dataclass(frozen=True, kw_only=True)
class FormalAcousticAuthority:
    """Code-owned source, output, and review-receipt authority."""

    real: real_acoustic_features.RealAcousticAuthority
    output_root: Path
    review_root: Path
    enforce_frozen_inventory: bool = True

    def __post_init__(self) -> None:
        if type(self.real) is not real_acoustic_features.RealAcousticAuthority:
            raise FormalAcousticFeatureError("real authority has the wrong exact type")
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())
        object.__setattr__(self, "review_root", Path(self.review_root).resolve())
        if type(self.enforce_frozen_inventory) is not bool:
            raise FormalAcousticFeatureError(
                "enforce_frozen_inventory must be an exact boolean"
            )

    @property
    def production_scope(self) -> str:
        """Return a code-derived scope; callers cannot opt into production."""

        return (
            CANONICAL_PRODUCTION_SCOPE
            if _has_canonical_production_scope(self)
            else NON_COUNTED_AUTHORITY_SCOPE
        )


@dataclass(frozen=True)
class FormalAcousticPaths:
    """Canonical artifact paths for one scene."""

    source_csv: Path
    evidence_json: Path
    feature_csv: Path
    pending_manifest: Path
    feature_manifest: Path
    review_receipt: Path


@dataclass(frozen=True)
class PendingAcousticArtifact:
    """Strictly loaded pending formal feature bundle."""

    scene: str
    paths: FormalAcousticPaths
    source_df: pd.DataFrame
    feature_df: pd.DataFrame
    evidence: dict[str, Any]
    pending_manifest: dict[str, Any]


@dataclass(frozen=True)
class ReviewedAcousticArtifact:
    """Strictly loaded independently reviewed feature bundle."""

    pending: PendingAcousticArtifact
    review_receipt: dict[str, Any]
    feature_manifest: dict[str, Any]

    @property
    def scene(self) -> str:
        return self.pending.scene


def _new_default_formal_acoustic_authority() -> FormalAcousticAuthority:
    root = Path(__file__).resolve().parents[3]
    output_root = root / "data" / "processed" / "acoustic_features"
    return FormalAcousticAuthority(
        real=_CANONICAL_REAL_ACOUSTIC_AUTHORITY,
        output_root=output_root,
        review_root=output_root,
    )


_DEFAULT_FORMAL_ACOUSTIC_AUTHORITY = _new_default_formal_acoustic_authority()


def _canonical_production_fields_match(authority: FormalAcousticAuthority) -> bool:
    """Check every code-owned production root and frozen-inventory setting."""

    root = Path(__file__).resolve().parents[3]
    output_root = (root / "data" / "processed" / "acoustic_features").resolve()
    if (
        type(authority.real) is not real_acoustic_features.RealAcousticAuthority
        or type(authority.real.analysis) is not analysis_base.AnalysisBaseAuthority
        or type(authority.real.audio) is not audio_contract.AudioSourceAuthority
    ):
        return False
    return (
        authority.output_root == output_root
        and authority.review_root == output_root
        and authority.enforce_frozen_inventory is True
        and _real_authority_snapshot(authority.real)
        == _CANONICAL_REAL_AUTHORITY_SNAPSHOT
        and authority.real.analysis.enforce_frozen_inventory is True
    )


def _has_canonical_production_scope(authority: FormalAcousticAuthority) -> bool:
    """Recognise only the private default singleton with canonical live roots."""

    return (
        authority is _DEFAULT_FORMAL_ACOUSTIC_AUTHORITY
        and _canonical_production_fields_match(authority)
    )


def _require_canonical_production_authority(authority: FormalAcousticAuthority) -> None:
    if not _has_canonical_production_scope(authority):
        _fail(
            "reviewed sealing/loading requires the code-owned default canonical "
            "production authority; fixture and caller-constructed authorities are "
            "permanently non-counted"
        )


def default_formal_acoustic_authority() -> FormalAcousticAuthority:
    """Return the sole code-owned canonical production authority singleton."""

    return _DEFAULT_FORMAL_ACOUSTIC_AUTHORITY


def _fail(message: str) -> None:
    raise FormalAcousticFeatureError(message)


def _select_authority(authority: object | None) -> FormalAcousticAuthority:
    selected = default_formal_acoustic_authority() if authority is None else authority
    if type(selected) is not FormalAcousticAuthority:
        _fail("formal acoustic authority has the wrong exact type")
    return selected


def _validate_scene(scene: object, authority: FormalAcousticAuthority) -> str:
    if type(scene) is not str or not scene or scene != scene.strip():
        _fail("scene must be one non-empty canonical identifier")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*", scene):
        _fail("scene must use the canonical group__scenario spelling")
    if authority.enforce_frozen_inventory and scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail(f"scene is outside the frozen ten-scene analysis inventory: {scene!r}")
    return scene


def canonical_acoustic_paths(
    scene: str,
    *,
    authority: FormalAcousticAuthority | None = None,
) -> FormalAcousticPaths:
    """Derive every path from code-owned roots; no caller path is accepted."""

    selected = _select_authority(authority)
    scene = _validate_scene(scene, selected)
    root = selected.output_root
    return FormalAcousticPaths(
        source_csv=(root / f"{scene}.source.csv").resolve(),
        evidence_json=(root / f"{scene}.evidence.json").resolve(),
        feature_csv=(root / f"{scene}.csv").resolve(),
        pending_manifest=(root / f"{scene}.pending.manifest.json").resolve(),
        feature_manifest=(root / f"{scene}.manifest.json").resolve(),
        review_receipt=(selected.review_root / f"{scene}.review.json").resolve(),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _file_binding(path: Path) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": size}


def _payload_binding(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _type_exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _type_exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _type_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalAcousticFeatureError(
            "formal acoustic JSON must be finite and serialisable"
        ) from exc


def _strict_json_bytes(
    payload: bytes,
    label: str,
    *,
    require_canonical: bool,
) -> dict[str, Any]:
    if type(payload) is not bytes:
        _fail(f"{label} must be exact immutable bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormalAcousticFeatureError(f"{label} is not strict UTF-8") from exc

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
        raise FormalAcousticFeatureError(f"{label} is not valid JSON") from exc

    def reject_nested_nonfinite(item: Any, location: str) -> None:
        if type(item) is float and not math.isfinite(item):
            _fail(f"{label} contains non-finite number at {location}")
        if type(item) is dict:
            for key, nested in item.items():
                reject_nested_nonfinite(nested, f"{location}.{key}")
        elif type(item) is list:
            for index, nested in enumerate(item):
                reject_nested_nonfinite(nested, f"{location}[{index}]")

    reject_nested_nonfinite(value, "$")
    if type(value) is not dict:
        _fail(f"{label} must contain one JSON object")
    if require_canonical and _json_bytes(value) != payload:
        _fail(f"{label} is not the complete canonical JSON byte encoding")
    return value


def _read_strict_json(
    path: Path,
    label: str,
    *,
    require_canonical: bool,
) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        _fail(f"canonical {label} does not exist: {path}")
    payload = path.read_bytes()
    value = _strict_json_bytes(payload, label, require_canonical=require_canonical)
    return value, payload


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return real_acoustic_features._csv_payload(frame)


def _strict_csv(path: Path, label: str) -> tuple[pd.DataFrame, bytes]:
    if not path.is_file():
        _fail(f"canonical {label} does not exist: {path}")
    payload = path.read_bytes()
    try:
        frame = real_acoustic_features._strict_csv_payload(payload, label)
    except (OSError, ValueError) as exc:
        raise FormalAcousticFeatureError(str(exc)) from exc
    return frame, payload


def _owner_parameter_decision() -> dict[str, Any]:
    return {
        "status": "owner_frozen",
        "decision_date": "2026-08-16",
        "parameter": "normalized_autocorrelation_voicing_threshold",
        "value": VOICING_THRESHOLD,
        "outcome_inspected_before_decision": False,
    }


def _trust_boundary() -> dict[str, Any]:
    return {
        "status": "owner_accepted",
        "decision_date": "2026-08-16",
        "model": "trusted_local_single_user_single_writer_workspace",
        "no_overwrite_required": True,
        "source_hash_and_replay_gates_retained": True,
        "malicious_concurrent_replacement_resistance_claimed": False,
        "residual_risk": (
            "a malicious process with write access during publication can violate the "
            "single-writer assumption"
        ),
    }


def _runtime_contract() -> dict[str, Any]:
    parameters = acoustic_features.extractor_parameters()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy": parameters["runtime_versions"]["numpy"],
        "scipy": parameters["runtime_versions"]["scipy"],
    }


def _implementation_binding() -> dict[str, Any]:
    paths = {
        "formal_builder": Path(__file__).resolve(),
        "real_source_wrapper": Path(real_acoustic_features.__file__).resolve(),
        "causal_extractor": Path(acoustic_features.__file__).resolve(),
    }
    return {name: _file_binding(path) for name, path in paths.items()}


def _validate_threshold_contract() -> None:
    threshold = acoustic_features.extractor_parameters()["pitch"][
        "voicing_threshold_inclusive"
    ]
    if type(threshold) is not float or threshold != VOICING_THRESHOLD:
        _fail("extractor voicing threshold differs from owner-frozen float 0.45")
    if acoustic_features.VOICING_THRESHOLD != VOICING_THRESHOLD:
        _fail("extractor module constant differs from owner-frozen float 0.45")


def _frame_contract(frame: pd.DataFrame, ticks: np.ndarray) -> dict[str, Any]:
    return real_acoustic_features._frame_contract(frame, ticks)


def _validate_frame(frame: pd.DataFrame, scene: str) -> np.ndarray:
    try:
        return real_acoustic_features._validate_scaffold_frame(frame, scene)
    except (OSError, ValueError) as exc:
        raise FormalAcousticFeatureError(str(exc)) from exc


def _expected_tick_range(scene: str, ticks: np.ndarray) -> tuple[int, int]:
    expected = analysis_base.EXPECTED_TICK_RANGES.get(scene)
    actual = (int(ticks[0]), int(ticks[-1]) + 1)
    if expected is not None and expected != actual:
        _fail(f"formal acoustic tick range {actual} differs from frozen range {expected}")
    return actual


def _validate_preflight(
    preflight: real_acoustic_features.RealAcousticPreflight,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    try:
        replayed, ticks, mapping = (
            real_acoustic_features._strict_replay_preflight_for_scaffold(preflight)
        )
    except (OSError, ValueError) as exc:
        raise FormalAcousticFeatureError(
            f"fresh real acoustic preflight failed strict replay: {exc}"
        ) from exc
    _expected_tick_range(replayed.scene, ticks)
    _validate_threshold_contract()
    return replayed.feature_df.copy(deep=True), ticks, mapping


def _pending_payloads_from_preflight(
    preflight: real_acoustic_features.RealAcousticPreflight,
    *,
    authority: FormalAcousticAuthority,
) -> tuple[FormalAcousticPaths, dict[Path, bytes]]:
    frame, ticks, mapping = _validate_preflight(preflight)
    scene = _validate_scene(preflight.scene, authority)
    paths = canonical_acoustic_paths(scene, authority=authority)
    source_payload = _csv_bytes(frame)
    feature_payload = _csv_bytes(frame.copy(deep=True))
    source_binding = _payload_binding(paths.source_csv, source_payload)
    feature_binding = _payload_binding(paths.feature_csv, feature_payload)
    evidence = {
        "artifact_kind": EVIDENCE_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "review_status": REVIEW_STATUS_PENDING,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "authority_scope": authority.production_scope,
        "owner_parameter_decision": _owner_parameter_decision(),
        "publication_trust_boundary": _trust_boundary(),
        "runtime": _runtime_contract(),
        "implementation": _implementation_binding(),
        "axis": real_acoustic_features.AXIS,
        "source_selection": preflight.report["source_selection"],
        "analysis_base": preflight.report["analysis_base"],
        "participant_to_slot": mapping,
        "audio_contract": preflight.report["audio_contract"],
        "master_mix_replay": preflight.report["master_mix_replay"],
        "extractor": preflight.report["extractor"],
        "causality": preflight.report["causality"],
        "validity": preflight.report["validity"],
        "source_csv": source_binding,
        "feature_csv": feature_binding,
        "frame_contract": _frame_contract(frame, ticks),
        "derivation_preflight_report": preflight.report,
    }
    evidence_payload = _json_bytes(evidence)
    evidence_binding = _payload_binding(paths.evidence_json, evidence_payload)
    column_sources = {
        column: {
            "participant": None,
            "slot": None,
            "source_path": str(paths.source_csv),
            "source_sha256": source_binding["sha256"],
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
    pending = {
        "artifact_kind": PENDING_MANIFEST_KIND,
        "format_version": FORMAT_VERSION,
        "target_contract": {
            "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
            "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
            "status": "blocked_until_independent_review_receipt",
        },
        "session_id": scene,
        "participant_to_slot": mapping,
        "supplied_columns": list(FEATURE_COLUMNS),
        "feature_csv_sha256": feature_binding["sha256"],
        "source_files": [source_binding, evidence_binding],
        "purpose": "model_input",
        "causality": real_acoustic_features._manifest_causality_contract(),
        "column_sources": column_sources,
        "review_status": REVIEW_STATUS_PENDING,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "authority_scope": authority.production_scope,
        "owner_parameter_decision": _owner_parameter_decision(),
        "publication_trust_boundary": _trust_boundary(),
    }
    return paths, {
        paths.source_csv: source_payload,
        paths.evidence_json: evidence_payload,
        paths.feature_csv: feature_payload,
        paths.pending_manifest: _json_bytes(pending),
    }


def _publish_payloads(payloads: dict[Path, bytes], label: str) -> None:
    if not payloads:
        _fail(f"{label} has no payloads to publish")
    destinations = tuple(payloads)
    if len(destinations) != len(set(destinations)):
        _fail(f"{label} contains duplicate destinations")
    parents = {path.parent for path in destinations}
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in destinations if os.path.lexists(path)]
    if existing:
        _fail(f"{label} refuses to overwrite existing paths: {existing}")

    staged: list[tuple[Path, tuple[int, int]]] = []
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for destination, payload in payloads.items():
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.tmp.",
                suffix=destination.suffix,
                dir=destination.parent,
                delete=False,
            ) as handle:
                stage = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            identity = assembly._file_identity(stage)
            staged.append((stage, identity))
            if stage.read_bytes() != payload:
                _fail(f"{label} staging bytes changed before publication")
        for (destination, payload), (stage, identity) in zip(
            payloads.items(), staged, strict=True
        ):
            if _sha256_file(stage) != (_sha256_bytes(payload), len(payload)):
                _fail(f"{label} staging hash changed before publication")
            assembly._publish_no_clobber(stage, destination)
            published.append((destination, identity))
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for destination, identity in reversed(published):
            cleanup = assembly._remove_owned_path(destination, identity)
            if not cleanup.removed:
                cleanup_errors.append(
                    f"{destination}: {cleanup.detail or 'owned path preserved'}"
                )
        if cleanup_errors:
            raise FormalAcousticFeatureError(
                f"{label} failed and rollback was incomplete: " + "; ".join(cleanup_errors)
            ) from exc
        raise
    finally:
        for stage, identity in staged:
            assembly._remove_owned_path(stage, identity)


def _require_binding(
    value: Any,
    expected_path: Path,
    expected_payload: bytes,
    label: str,
) -> None:
    expected = _payload_binding(expected_path, expected_payload)
    if not _type_exact_equal(value, expected):
        _fail(f"{label} does not bind the canonical path, bytes, and length")


def _validate_live_source_bindings(
    evidence: dict[str, Any],
    *,
    authority: FormalAcousticAuthority,
) -> None:
    scene = evidence["scene"]
    analysis_record = evidence["analysis_base"]
    audio_record = evidence["audio_contract"]
    master_record = evidence["master_mix_replay"]
    expected_analysis = (
        authority.real.analysis.analysis_base_root / f"{scene}.csv"
    ).resolve()
    expected_analysis_sidecar = expected_analysis.with_suffix(".provenance.json")
    expected_audio = (
        authority.real.analysis.audio_manifest_root
        / f"{scene}.audio_contract.json"
    ).resolve()
    expected_master = audio_contract.canonical_source_paths(
        scene, authority.real.audio
    )["master"].resolve()
    expected_paths = {
        "analysis_base.path": (analysis_record["path"], expected_analysis),
        "analysis_base.provenance_path": (
            analysis_record["provenance_path"],
            expected_analysis_sidecar,
        ),
        "audio_contract.path": (audio_record["path"], expected_audio),
        "master_mix_replay.path": (master_record["path"], expected_master),
    }
    for label, (declared, expected) in expected_paths.items():
        if type(declared) is not str or Path(declared).resolve() != expected:
            _fail(f"{label} is not the live code-owned canonical source")
        if not expected.is_file():
            _fail(f"{label} canonical source is missing: {expected}")
    for label, path, digest_key, size_key in (
        ("analysis CSV", expected_analysis, "sha256", None),
        ("analysis sidecar", expected_analysis_sidecar, "provenance_sha256", None),
        ("audio contract", expected_audio, "sha256", "size_bytes"),
        ("master WAV", expected_master, "sha256", "size_bytes"),
    ):
        record = (
            analysis_record
            if label.startswith("analysis")
            else audio_record
            if label == "audio contract"
            else master_record
        )
        digest, size = _sha256_file(path)
        if record[digest_key] != digest:
            _fail(f"{label} hash changed after formal acoustic derivation")
        if size_key is not None and record[size_key] != size:
            _fail(f"{label} byte length changed after formal acoustic derivation")

    header = evidence["audio_contract"]["master_source"]["wave_header"]
    try:
        with wave.open(str(expected_master), "rb") as handle:
            actual_header = {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate_hz": handle.getframerate(),
                "n_frames": handle.getnframes(),
                "compression_type": handle.getcomptype(),
            }
    except (EOFError, OSError, wave.Error) as exc:
        raise FormalAcousticFeatureError(
            f"master WAV header cannot be replayed: {exc}"
        ) from exc
    if not _type_exact_equal(actual_header, header):
        _fail("master WAV header changed after formal acoustic derivation")
    if evidence["audio_contract"]["master_exact_permutation"] is not True:
        _fail("formal acoustic evidence does not bind exact four-channel permutation replay")


def _validate_pending_evidence(
    evidence: dict[str, Any],
    *,
    scene: str,
    paths: FormalAcousticPaths,
    source: pd.DataFrame,
    source_payload: bytes,
    feature_payload: bytes,
    authority: FormalAcousticAuthority,
) -> tuple[np.ndarray, dict[str, str]]:
    expected_keys = {
        "artifact_kind",
        "format_version",
        "scene",
        "output_role",
        "review_status",
        "assembly_eligible",
        "counted_input_ready",
        "authority_scope",
        "owner_parameter_decision",
        "publication_trust_boundary",
        "runtime",
        "implementation",
        "axis",
        "source_selection",
        "analysis_base",
        "participant_to_slot",
        "audio_contract",
        "master_mix_replay",
        "extractor",
        "causality",
        "validity",
        "source_csv",
        "feature_csv",
        "frame_contract",
        "derivation_preflight_report",
    }
    if set(evidence) != expected_keys:
        _fail("formal acoustic evidence keys differ from format v1")
    identity = {
        "artifact_kind": EVIDENCE_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "review_status": REVIEW_STATUS_PENDING,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "authority_scope": authority.production_scope,
    }
    if not _type_exact_equal({key: evidence[key] for key in identity}, identity):
        _fail("formal acoustic evidence identity/review state is unsupported")
    if not _type_exact_equal(evidence["owner_parameter_decision"], _owner_parameter_decision()):
        _fail("formal acoustic evidence changed the owner-frozen 0.45 decision")
    if not _type_exact_equal(evidence["publication_trust_boundary"], _trust_boundary()):
        _fail("formal acoustic evidence changed the accepted trust boundary")
    if not _type_exact_equal(evidence["runtime"], _runtime_contract()):
        _fail("formal acoustic evidence runtime differs from the current extractor runtime")
    if not _type_exact_equal(evidence["implementation"], _implementation_binding()):
        _fail("formal acoustic implementation bytes changed; rebuild and re-review")
    if evidence["axis"] != real_acoustic_features.AXIS:
        _fail("formal acoustic evidence is not on the mic-A shared axis")
    _require_binding(evidence["source_csv"], paths.source_csv, source_payload, "source_csv")
    _require_binding(evidence["feature_csv"], paths.feature_csv, feature_payload, "feature_csv")
    ticks = _validate_frame(source, scene)
    _expected_tick_range(scene, ticks)
    if not _type_exact_equal(evidence["frame_contract"], _frame_contract(source, ticks)):
        _fail("formal acoustic frame contract differs from exact source CSV replay")
    preflight = real_acoustic_features.RealAcousticPreflight(
        scene=scene,
        feature_df=source.copy(deep=True),
        report=evidence["derivation_preflight_report"],
    )
    _, replay_ticks, mapping = _validate_preflight(preflight)
    if not np.array_equal(ticks, replay_ticks):
        _fail("embedded preflight changed the formal source tick keys")
    report = preflight.report
    for key in (
        "source_selection",
        "analysis_base",
        "participant_to_slot",
        "audio_contract",
        "master_mix_replay",
        "extractor",
        "causality",
        "validity",
    ):
        if not _type_exact_equal(evidence[key], report[key]):
            _fail(f"formal acoustic evidence field {key!r} differs from preflight replay")
    _validate_live_source_bindings(evidence, authority=authority)
    return ticks, mapping


def _expected_pending_manifest(
    *,
    scene: str,
    paths: FormalAcousticPaths,
    source_payload: bytes,
    evidence_payload: bytes,
    feature_payload: bytes,
    mapping: dict[str, str],
    authority: FormalAcousticAuthority,
) -> dict[str, Any]:
    source_binding = _payload_binding(paths.source_csv, source_payload)
    evidence_binding = _payload_binding(paths.evidence_json, evidence_payload)
    return {
        "artifact_kind": PENDING_MANIFEST_KIND,
        "format_version": FORMAT_VERSION,
        "target_contract": {
            "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
            "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
            "status": "blocked_until_independent_review_receipt",
        },
        "session_id": scene,
        "participant_to_slot": mapping,
        "supplied_columns": list(FEATURE_COLUMNS),
        "feature_csv_sha256": _sha256_bytes(feature_payload),
        "source_files": [source_binding, evidence_binding],
        "purpose": "model_input",
        "causality": real_acoustic_features._manifest_causality_contract(),
        "column_sources": {
            column: {
                "participant": None,
                "slot": None,
                "source_path": str(paths.source_csv),
                "source_sha256": source_binding["sha256"],
                "source_column": column,
                "source_participant": None,
                "source_keys": {
                    "session_id": "session_id",
                    "timestamp_s": "timestamp_s",
                },
                "value_transform": {"kind": "identity"},
            }
            for column in FEATURE_COLUMNS
        },
        "review_status": REVIEW_STATUS_PENDING,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "authority_scope": authority.production_scope,
        "owner_parameter_decision": _owner_parameter_decision(),
        "publication_trust_boundary": _trust_boundary(),
    }


def load_pending_acoustic_scene(
    scene: str,
    *,
    authority: FormalAcousticAuthority | None = None,
) -> PendingAcousticArtifact:
    """Sealed-load one canonical pending bundle without recomputing audio."""

    selected = _select_authority(authority)
    scene = _validate_scene(scene, selected)
    paths = canonical_acoustic_paths(scene, authority=selected)
    source, source_payload = _strict_csv(paths.source_csv, "formal acoustic source CSV")
    feature, feature_payload = _strict_csv(paths.feature_csv, "formal acoustic feature CSV")
    source_ticks = _validate_frame(source, scene)
    feature_ticks = _validate_frame(feature, scene)
    if not np.array_equal(source_ticks, feature_ticks) or not _type_exact_equal(
        real_acoustic_features._frame_column_contract(source),
        real_acoustic_features._frame_column_contract(feature),
    ):
        _fail("feature CSV does not bitwise replay every typed source column")
    evidence, evidence_payload = _read_strict_json(
        paths.evidence_json,
        "formal acoustic evidence",
        require_canonical=True,
    )
    _, mapping = _validate_pending_evidence(
        evidence,
        scene=scene,
        paths=paths,
        source=source,
        source_payload=source_payload,
        feature_payload=feature_payload,
        authority=selected,
    )
    pending, pending_payload = _read_strict_json(
        paths.pending_manifest,
        "pending acoustic manifest",
        require_canonical=True,
    )
    expected_pending = _expected_pending_manifest(
        scene=scene,
        paths=paths,
        source_payload=source_payload,
        evidence_payload=evidence_payload,
        feature_payload=feature_payload,
        mapping=mapping,
        authority=selected,
    )
    if not _type_exact_equal(pending, expected_pending):
        _fail("pending acoustic manifest differs from deterministic format-v1 replay")
    if pending["artifact_kind"] == assembly.FEATURE_MANIFEST_KIND:
        _fail("pending acoustic manifest must never use the assembly-accepted kind")
    # Final checks narrow the normal single-writer read window.
    expected_hashes = {
        paths.source_csv: _sha256_bytes(source_payload),
        paths.evidence_json: _sha256_bytes(evidence_payload),
        paths.feature_csv: _sha256_bytes(feature_payload),
        paths.pending_manifest: _sha256_bytes(pending_payload),
    }
    for path, digest in expected_hashes.items():
        if _sha256_file(path)[0] != digest:
            _fail(f"pending acoustic artifact changed during sealed load: {path}")
    return PendingAcousticArtifact(
        scene=scene,
        paths=paths,
        source_df=source,
        feature_df=feature,
        evidence=evidence,
        pending_manifest=pending,
    )


def build_pending_acoustic_scene(
    scene: str,
    *,
    authority: FormalAcousticAuthority | None = None,
) -> PendingAcousticArtifact:
    """Raw-replay and publish one no-overwrite, review-pending formal bundle."""

    selected = _select_authority(authority)
    scene = _validate_scene(scene, selected)
    paths = canonical_acoustic_paths(scene, authority=selected)
    if paths.feature_manifest.exists() or paths.review_receipt.exists():
        _fail("pending build refuses a pre-existing reviewed manifest or review receipt")
    try:
        preflight = real_acoustic_features.derive_real_acoustic_preflight(
            scene,
            authority=selected.real,
        )
    except (OSError, ValueError) as exc:
        raise FormalAcousticFeatureError(
            f"strict raw acoustic derivation failed for {scene}: {exc}"
        ) from exc
    derived_paths, payloads = _pending_payloads_from_preflight(
        preflight,
        authority=selected,
    )
    if derived_paths != paths:
        _fail("pending payload derivation changed canonical output paths")
    _publish_payloads(payloads, f"pending acoustic scene {scene}")
    try:
        return load_pending_acoustic_scene(scene, authority=selected)
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for path in reversed(tuple(payloads)):
            try:
                identity = assembly._file_identity(path)
                cleanup = assembly._remove_owned_path(path, identity)
                if not cleanup.removed:
                    cleanup_errors.append(str(path))
            except OSError:
                cleanup_errors.append(str(path))
        if cleanup_errors:
            raise FormalAcousticFeatureError(
                "pending acoustic publication failed validation and cleanup was incomplete: "
                + ", ".join(cleanup_errors)
            ) from exc
        raise


def _review_bundle(pending: PendingAcousticArtifact) -> dict[str, Any]:
    paths = pending.paths
    return {
        "source_csv": _file_binding(paths.source_csv),
        "evidence_json": _file_binding(paths.evidence_json),
        "feature_csv": _file_binding(paths.feature_csv),
        "pending_manifest": _file_binding(paths.pending_manifest),
    }


def _load_review_receipt(
    pending: PendingAcousticArtifact,
) -> tuple[dict[str, Any], bytes]:
    receipt, payload = _read_strict_json(
        pending.paths.review_receipt,
        "independent acoustic review receipt",
        require_canonical=False,
    )
    expected_keys = {
        "artifact_kind",
        "format_version",
        "scene",
        "decision",
        "independent_review",
        "reviewer_id",
        "reviewed_at_utc",
        "review_scope",
        "reviewed_bundle",
        "notes",
    }
    if set(receipt) != expected_keys:
        _fail("independent acoustic review receipt keys differ from format v1")
    if (
        receipt["artifact_kind"] != REVIEW_RECEIPT_KIND
        or type(receipt["format_version"]) is not int
        or receipt["format_version"] != FORMAT_VERSION
        or receipt["scene"] != pending.scene
        or receipt["decision"] != REVIEW_STATUS_ACCEPTED
        or receipt["independent_review"] is not True
    ):
        _fail("independent acoustic review receipt identity/decision is not accepted")
    reviewer = receipt["reviewer_id"]
    reviewed_at = receipt["reviewed_at_utc"]
    notes = receipt["notes"]
    if type(reviewer) is not str or not reviewer.strip() or reviewer != reviewer.strip():
        _fail("review receipt reviewer_id must be a non-empty trimmed string")
    if type(reviewed_at) is not str or _UTC_RE.fullmatch(reviewed_at) is None:
        _fail("review receipt reviewed_at_utc must use YYYY-MM-DDTHH:MM:SSZ")
    if type(notes) is not str:
        _fail("review receipt notes must be a JSON string")
    if not _type_exact_equal(receipt["review_scope"], list(REVIEW_SCOPE)):
        _fail("review receipt does not cover the complete frozen review scope")
    if not _type_exact_equal(receipt["reviewed_bundle"], _review_bundle(pending)):
        _fail("review receipt does not bind the current canonical pending bundle")
    return receipt, payload


def _expected_reviewed_manifest(
    pending: PendingAcousticArtifact,
    receipt: dict[str, Any],
    receipt_payload: bytes,
) -> dict[str, Any]:
    paths = pending.paths
    source_binding = _file_binding(paths.source_csv)
    evidence_binding = _file_binding(paths.evidence_json)
    pending_binding = _file_binding(paths.pending_manifest)
    receipt_binding = _payload_binding(paths.review_receipt, receipt_payload)
    pending_manifest = pending.pending_manifest
    return {
        "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
        "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
        "session_id": pending.scene,
        "participant_to_slot": pending_manifest["participant_to_slot"],
        "supplied_columns": list(FEATURE_COLUMNS),
        "feature_csv_sha256": _file_binding(paths.feature_csv)["sha256"],
        "source_files": [
            source_binding,
            evidence_binding,
            pending_binding,
            receipt_binding,
        ],
        "purpose": "model_input",
        "causality": real_acoustic_features._manifest_causality_contract(),
        "column_sources": pending_manifest["column_sources"],
        "review_status": REVIEW_STATUS_ACCEPTED,
        "assembly_eligible": True,
        "counted_input_ready": True,
        "authority_scope": CANONICAL_PRODUCTION_SCOPE,
        "output_role": REVIEWED_OUTPUT_ROLE,
        "owner_parameter_decision": _owner_parameter_decision(),
        "publication_trust_boundary": _trust_boundary(),
        "review_gate": {
            "receipt": receipt_binding,
            "reviewer_id": receipt["reviewer_id"],
            "reviewed_at_utc": receipt["reviewed_at_utc"],
            "independent_review": True,
        },
    }


def _validate_assembly_v3_manifest(
    pending: PendingAcousticArtifact,
) -> dict[str, Any]:
    base_df = pending.source_df.loc[:, list(KEY_COLUMNS)].copy()
    base_provenance = {
        "participant_to_slot": pending.pending_manifest["participant_to_slot"]
    }
    try:
        return assembly._validate_feature_manifest(
            pending.paths.feature_csv,
            list(FEATURE_COLUMNS),
            base_df,
            base_provenance,
            forbidden_generated_paths={
                Path(pending.evidence["analysis_base"]["path"]).resolve(),
                Path(pending.evidence["analysis_base"]["provenance_path"]).resolve(),
            },
        )
    except (OSError, ValueError) as exc:
        raise FormalAcousticFeatureError(
            f"reviewed acoustic manifest fails assembly-v3 replay: {exc}"
        ) from exc


def load_reviewed_acoustic_scene(
    scene: str,
    *,
    authority: FormalAcousticAuthority | None = None,
) -> ReviewedAcousticArtifact:
    """Sealed-load a reviewed scene and replay the actual assembly-v3 gate."""

    selected = _select_authority(authority)
    _require_canonical_production_authority(selected)
    pending = load_pending_acoustic_scene(scene, authority=selected)
    receipt, receipt_payload = _load_review_receipt(pending)
    manifest, manifest_payload = _read_strict_json(
        pending.paths.feature_manifest,
        "reviewed acoustic feature manifest",
        require_canonical=True,
    )
    expected = _expected_reviewed_manifest(pending, receipt, receipt_payload)
    if not _type_exact_equal(manifest, expected):
        _fail("reviewed acoustic manifest differs from deterministic review-gate replay")
    _validate_assembly_v3_manifest(pending)
    if _sha256_file(pending.paths.feature_manifest)[0] != _sha256_bytes(manifest_payload):
        _fail("reviewed acoustic manifest changed during sealed replay")
    return ReviewedAcousticArtifact(
        pending=pending,
        review_receipt=receipt,
        feature_manifest=manifest,
    )


def seal_reviewed_acoustic_scene(
    scene: str,
    *,
    authority: FormalAcousticAuthority | None = None,
) -> ReviewedAcousticArtifact:
    """Publish the assembly-v3 manifest only after an external receipt passes."""

    selected = _select_authority(authority)
    _require_canonical_production_authority(selected)
    pending = load_pending_acoustic_scene(scene, authority=selected)
    if pending.paths.feature_manifest.exists():
        _fail("reviewed acoustic manifest already exists; refusing to overwrite")
    receipt, receipt_payload = _load_review_receipt(pending)
    manifest = _expected_reviewed_manifest(pending, receipt, receipt_payload)
    _publish_payloads(
        {pending.paths.feature_manifest: _json_bytes(manifest)},
        f"reviewed acoustic manifest {scene}",
    )
    try:
        return load_reviewed_acoustic_scene(scene, authority=selected)
    except BaseException as exc:
        try:
            identity = assembly._file_identity(pending.paths.feature_manifest)
            cleanup = assembly._remove_owned_path(pending.paths.feature_manifest, identity)
        except OSError:
            cleanup = assembly._OwnedPathCleanup(removed=False)
        if not cleanup.removed:
            raise FormalAcousticFeatureError(
                "reviewed acoustic manifest failed validation and could not be removed"
            ) from exc
        raise


def reaudit_formal_acoustic_scene(
    scene: str,
    *,
    authority: FormalAcousticAuthority | None = None,
    require_reviewed: bool = True,
) -> PendingAcousticArtifact | ReviewedAcousticArtifact:
    """Full raw-source replay of a published pending or reviewed scene."""

    selected = _select_authority(authority)
    scene = _validate_scene(scene, selected)
    if type(require_reviewed) is not bool:
        _fail("require_reviewed must be an exact boolean")
    if require_reviewed:
        load_reviewed_acoustic_scene(scene, authority=selected)
    else:
        load_pending_acoustic_scene(scene, authority=selected)
    try:
        preflight = real_acoustic_features.derive_real_acoustic_preflight(
            scene,
            authority=selected.real,
        )
    except (OSError, ValueError) as exc:
        raise FormalAcousticFeatureError(
            f"strict raw acoustic re-audit failed for {scene}: {exc}"
        ) from exc
    _, expected_payloads = _pending_payloads_from_preflight(
        preflight,
        authority=selected,
    )
    for path, expected in expected_payloads.items():
        if path.read_bytes() != expected:
            _fail(f"published acoustic artifact differs from fresh raw replay: {path}")
    # Reopen after the expensive raw replay to narrow the accepted single-writer window.
    return (
        load_reviewed_acoustic_scene(scene, authority=selected)
        if require_reviewed
        else load_pending_acoustic_scene(scene, authority=selected)
    )


__all__ = [
    "CANONICAL_PRODUCTION_SCOPE",
    "EVIDENCE_ARTIFACT_KIND",
    "FORMAT_VERSION",
    "FormalAcousticAuthority",
    "FormalAcousticFeatureError",
    "NON_COUNTED_AUTHORITY_SCOPE",
    "PENDING_MANIFEST_KIND",
    "PendingAcousticArtifact",
    "REVIEW_RECEIPT_KIND",
    "ReviewedAcousticArtifact",
    "build_pending_acoustic_scene",
    "canonical_acoustic_paths",
    "default_formal_acoustic_authority",
    "load_pending_acoustic_scene",
    "load_reviewed_acoustic_scene",
    "reaudit_formal_acoustic_scene",
    "seal_reviewed_acoustic_scene",
]
