"""Leakage-safe post-hoc matched replay for the dissertation H2 question.

The module deliberately lives beside, rather than inside, the historical outer
runner.  It has four create-once phases:

``freeze``
    Validate the fixed A+H 5-epoch primary and 15-epoch sensitivity matrices and
    publish outer probability views which contain no labels, VAD, or validity.
``calibrate_and_seal_traces``
    Fit and select every parameter from the three inner group-OOF partitions,
    then generate outer traces from only the gold-free views and reviewed local
    acoustic descendants.
``evaluate``
    Open outer gold only after traces are sealed and score the same frozen 303
    robot-entry opportunities.
``audit``
    Re-hash the phase boundary and reject promotion, mutation, or overwrite.

Every artifact is post-hoc, exploratory, non-production, and non-counted.  The
local acoustic loader intentionally does not call ``load_reviewed_acoustic_scene``:
that loader replays live raw sources.  Here, already sealed E-drive descendants
are hash checked against their review receipt, assembled provenance, and frozen
run readiness while raw WAV replay is explicitly recorded as skipped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

from hri_turn_taking.vap import thesis_backend
from hri_turn_taking.vap.events import extract_shift_hold_events
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, p_now_p_future
from hri_turn_taking.vap.outer_experiment import (
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_EXPECTED_N_ROWS,
    DEADLINE_PROTOCOL_ID,
    DEADLINE_SESSION_TO_GROUP,
    validate_outer_run_manifest,
    validate_outer_split_manifest,
)

FRAME_RATE_HZ = 25
ROBOT_SLOT = 0
EXPECTED_N_EVENTS = 303
EXPECTED_N_POSITIVE = 111
EXPECTED_N_NEGATIVE = 192
OUTER_FOLD_IDS = ("outer_01", "outer_02", "outer_03", "outer_04")
INNER_FOLD_IDS = ("inner_01", "inner_02", "inner_03")
SPEECH_FEATURES = ("array_energy", "voiced", "f0_proxy")
SPEECH_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(5, 100, 5))
REACTIVE_SILENCE_CANDIDATES = (5, 10, 15, 20, 25, 30, 38)
PREDICTIVE_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(5, 101, 5))
MIN_CONSECUTIVE_FRAMES = 3
REFRACTORY_FRAMES = 25
PRIMARY_PRE_GAP_FRAMES = 10

RESULT_ROLE = "posthoc_exploratory_h2_matched_replay"
COUNTED_ELIGIBLE = False
PRODUCTION_ELIGIBLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_MATRIX = PROJECT_ROOT / "outputs" / "thesis_deadline_20260816" / "matrix_seed13_v1"
SENSITIVITY_MATRIX = (
    PROJECT_ROOT / "outputs" / "thesis_posthoc_15epoch_20260817" / "matrix_seed13_v1"
)
ACOUSTIC_ROOT = PROJECT_ROOT / "data" / "processed" / "acoustic_features"
LEGACY_REACTIVE_EVENT = (
    PROJECT_ROOT / "tmp" / "reactive_diag_preflight_r1" / "robot_entry_opportunities.csv"
)
LEGACY_REACTIVE_EVENT_SHA256 = (
    "92de19251a815553b6bac5ebd890b3089a3ab5f8f8882d0e6faa05a7dc717285"
)

FREEZE_FILENAME = "freeze_manifest.json"
PROJECTION_SEAL_FILENAME = "projection_seal.json"
TRACE_SEAL_FILENAME = "trace_seal_manifest.json"
EVALUATION_FILENAME = "evaluation_manifest.json"
EVENT_FILENAME = "matched_robot_entry_events.csv"
METRICS_FILENAME = "matched_metrics.json"

TRACE_COLUMNS = (
    "session_id",
    "group_id",
    "outer_fold_id",
    "tick_25hz",
    "timestamp_s",
    "speech_probability",
    "speech_detected",
    "reactive_silent_run_frames",
    "reactive_ready",
    "reactive_trigger",
    "robot_p_now",
    "robot_p_future",
    "predictive_trigger",
)

EVENT_COLUMNS = (
    "budget_id",
    "max_epochs",
    "policy_id",
    "window_id",
    "outer_fold_id",
    "event_id",
    "session_id",
    "group_id",
    "event_index",
    "gap_start_tick",
    "gap_end_tick_exclusive",
    "previous_slot",
    "next_slot",
    "robot_entry_label",
    "triggered",
    "first_trigger_tick",
    "latency_s",
)


class H2MatchedReplayError(ValueError):
    """A matched-replay integrity, leakage, or protocol gate failed."""


@dataclass(frozen=True)
class GoldFreeSessionView:
    """Only the fields allowed to cross into outer trace generation."""

    session_id: str
    group_id: str
    ticks: np.ndarray
    probabilities: np.ndarray
    present: np.ndarray


@dataclass(frozen=True)
class EOnlyAcousticSession:
    """Reviewed local acoustic descendant; no raw source is opened."""

    session_id: str
    ticks: np.ndarray
    features: np.ndarray
    bindings: dict[str, object]


@dataclass(frozen=True)
class CalibrationSession:
    """One inner-OOF session, including gold permitted only for calibration."""

    session_id: str
    group_id: str
    ticks: np.ndarray
    probabilities: np.ndarray
    vad: np.ndarray
    valid: np.ndarray
    present: np.ndarray
    acoustic_features: np.ndarray


@dataclass(frozen=True)
class MatchedReplayState:
    """Minimal causal state needed to continue trace generation across chunks."""

    reactive_silent_run: int = 0
    predictive_consecutive: int = 0
    predictive_refractory: int = 0


def _fail(message: str) -> None:
    raise H2MatchedReplayError(message)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path_value: str | Path) -> dict[str, object]:
    path = _readable_local_path(path_value, "JSON")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                H2MatchedReplayError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise H2MatchedReplayError(f"cannot parse strict JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_f_path(path_value: str | Path, role: str) -> Path:
    path = Path(path_value)
    if path.drive.lower() == "f:":
        _fail(f"{role} must not dereference F-drive raw/shared data: {path}")
    return path.resolve()


def _readable_local_path(path_value: str | Path, role: str) -> Path:
    path = _reject_f_path(path_value, role)
    if path.is_symlink() or not path.is_file():
        _fail(f"{role} must be one existing non-symlink file: {path}")
    return path


def _file_binding(path_value: str | Path) -> dict[str, object]:
    path = _readable_local_path(path_value, "bound file")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_file_binding(binding: object, *, role: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size_bytes"}:
        _fail(f"{role} binding fields are not exact")
    path = _readable_local_path(binding["path"], role)
    size = binding["size_bytes"]
    if (
        not _is_sha256(binding["sha256"])
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or path.stat().st_size != size
        or _sha256_file(path) != binding["sha256"]
    ):
        _fail(f"{role} bytes differ from the sealed binding")
    return path


def _validate_opaque_binding(binding: object, *, role: str) -> dict[str, object]:
    """Validate provenance syntax without dereferencing the recorded path."""

    if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size_bytes"}:
        _fail(f"{role} opaque binding fields are not exact")
    size = binding["size_bytes"]
    if (
        not isinstance(binding["path"], str)
        or not binding["path"]
        or not Path(binding["path"]).is_absolute()
        or not _is_sha256(binding["sha256"])
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        _fail(f"{role} opaque binding is malformed")
    return dict(binding)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_create_once(
    destination: str | Path,
    files: dict[Path, bytes],
) -> Path:
    output = _reject_f_path(destination, "output directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matched-replay directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for relative, payload in files.items():
            if relative.is_absolute() or ".." in relative.parts:
                _fail("published relative path escapes its output directory")
            _write_exclusive(stage / relative, payload)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite matched-replay directory: {output}")
        stage.replace(output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return output


def _timestamps_to_ticks(values: object, *, role: str) -> np.ndarray:
    try:
        timestamps = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise H2MatchedReplayError(f"{role} timestamps must be numeric") from exc
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.isfinite(timestamps).all():
        _fail(f"{role} timestamps must be one non-empty finite vector")
    raw = timestamps * FRAME_RATE_HZ
    ticks = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, ticks, atol=1e-7, rtol=0.0):
        _fail(f"{role} timestamps are not exact 25 Hz ticks")
    if len(ticks) > 1 and not np.equal(np.diff(ticks), 1).all():
        _fail(f"{role} ticks are not contiguous")
    return ticks


def _validate_probability_session(record: object) -> GoldFreeSessionView:
    allowed_source_fields = {
        "session_id",
        "group_id",
        "n_frames",
        "tick_25hz",
        "probabilities",
        "labels",
        "vad",
        "valid",
        "present",
    }
    if not isinstance(record, dict) or set(record) != allowed_source_fields:
        _fail("saved prediction session fields differ from the version-1 contract")
    session_id = record["session_id"]
    group_id = record["group_id"]
    n_frames = record["n_frames"]
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(group_id, str)
        or not group_id
        or isinstance(n_frames, bool)
        or not isinstance(n_frames, int)
        or n_frames <= 0
    ):
        _fail("saved prediction session identity is invalid")
    ticks = np.asarray(record["tick_25hz"], dtype=np.int64)
    probabilities = np.asarray(record["probabilities"], dtype=np.float64)
    present = np.asarray(record["present"], dtype=np.float64)
    if ticks.shape != (n_frames,) or probabilities.shape != (n_frames, 4, 4):
        _fail("gold-free probability view has an unexpected tick/probability shape")
    if present.shape != (4,) or not np.isin(present, (0.0, 1.0)).all():
        _fail("gold-free probability view present mask must be binary length four")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        _fail("gold-free probability view probabilities must be finite in [0, 1]")
    if len(ticks) > 1 and not np.equal(np.diff(ticks), 1).all():
        _fail("gold-free probability view ticks must be contiguous")
    return GoldFreeSessionView(
        session_id=session_id,
        group_id=group_id,
        ticks=ticks.copy(),
        probabilities=probabilities.copy(),
        present=present.copy(),
    )


def gold_free_probability_view(
    prediction_payload: dict[str, object],
    *,
    budget_id: str,
    max_epochs: int,
) -> dict[str, object]:
    """Project a strict saved-prediction payload onto an explicit no-gold schema."""

    sessions = prediction_payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        _fail("saved prediction has no sessions")
    views = [_validate_probability_session(record) for record in sessions]
    return {
        "artifact_kind": "vap_h2_gold_free_outer_probability_view",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "budget_id": budget_id,
        "max_epochs": max_epochs,
        "condition_id": prediction_payload.get("condition_id"),
        "outer_fold_id": prediction_payload.get("outer_fold_id"),
        "seed": prediction_payload.get("seed"),
        "frame_rate_hz": prediction_payload.get("frame_rate_hz"),
        "checkpoint_sha256": prediction_payload.get("checkpoint_sha256"),
        "run_manifest_sha256": prediction_payload.get("run_manifest_sha256"),
        "allowed_session_fields": [
            "session_id",
            "group_id",
            "n_frames",
            "tick_25hz",
            "probabilities",
            "present",
        ],
        "forbidden_fields": ["labels", "vad", "valid", "features", "floor_holder"],
        "sessions": [
            {
                "session_id": view.session_id,
                "group_id": view.group_id,
                "n_frames": int(view.ticks.size),
                "tick_25hz": view.ticks.tolist(),
                "probabilities": view.probabilities.tolist(),
                "present": view.present.tolist(),
            }
            for view in views
        ],
    }


def load_gold_free_probability_view(path_value: str | Path) -> tuple[GoldFreeSessionView, ...]:
    """Load the sealed projection and reject any accidental gold-bearing field."""

    payload = _read_json(path_value)
    required = {
        "artifact_kind",
        "format_version",
        "result_role",
        "production_eligible",
        "counted_eligible",
        "budget_id",
        "max_epochs",
        "condition_id",
        "outer_fold_id",
        "seed",
        "frame_rate_hz",
        "checkpoint_sha256",
        "run_manifest_sha256",
        "allowed_session_fields",
        "forbidden_fields",
        "sessions",
    }
    if set(payload) != required:
        _fail("gold-free view top-level fields are not exact")
    if (
        payload["artifact_kind"] != "vap_h2_gold_free_outer_probability_view"
        or payload["format_version"] != 1
        or payload["result_role"] != RESULT_ROLE
        or payload["production_eligible"] is not False
        or payload["counted_eligible"] is not False
        or payload["condition_id"] != "A+H"
        or payload["seed"] != 13
        or payload["frame_rate_hz"] != FRAME_RATE_HZ
    ):
        _fail("gold-free view identity is invalid or attempted promotion")
    allowed = {
        "session_id",
        "group_id",
        "n_frames",
        "tick_25hz",
        "probabilities",
        "present",
    }
    records = payload["sessions"]
    if not isinstance(records, list) or not records:
        _fail("gold-free view has no session records")
    result: list[GoldFreeSessionView] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != allowed:
            _fail("gold-free session fields are not exact")
        synthetic_source = {
            **record,
            "labels": np.zeros((record["n_frames"], 4, 4)).tolist(),
            "vad": np.zeros((record["n_frames"], 4)).tolist(),
            "valid": np.ones(record["n_frames"]).tolist(),
        }
        result.append(_validate_probability_session(synthetic_source))
    return tuple(result)


def _strict_csv(path: Path, *, role: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise H2MatchedReplayError(f"cannot parse {role} CSV {path}: {exc}") from exc


def _binding_matches(binding: object, path: Path) -> bool:
    if not isinstance(binding, dict):
        return False
    return (
        binding.get("path") == str(path.resolve())
        and binding.get("sha256") == _sha256_file(path)
        and binding.get("size_bytes") == path.stat().st_size
    )


def load_e_only_acoustic_descendant(
    scene: str,
    *,
    assembled_source_artifact: dict[str, object],
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> EOnlyAcousticSession:
    """Load one reviewed acoustic descendant without opening or hashing raw WAV.

    Paths embedded in evidence as historical provenance remain opaque strings.
    Every path dereferenced by this function is independently rejected if it is
    on ``F:`` and is hash-bound to the frozen assembled/run authority.
    """

    if scene not in DEADLINE_SESSION_TO_GROUP:
        _fail(f"scene is outside the exact-eight corpus: {scene}")
    root = _reject_f_path(acoustic_root, "acoustic descendant root")
    if root.is_symlink() or not root.is_dir():
        _fail("acoustic descendant root must be one existing non-symlink directory")
    paths = {
        "source_csv": root / f"{scene}.source.csv",
        "evidence_json": root / f"{scene}.evidence.json",
        "feature_csv": root / f"{scene}.csv",
        "pending_manifest": root / f"{scene}.pending.manifest.json",
        "feature_manifest": root / f"{scene}.manifest.json",
        "review_receipt": root / f"{scene}.review.json",
    }
    for role, path in paths.items():
        paths[role] = _readable_local_path(path, f"{scene} {role}")

    feature = _strict_csv(paths["feature_csv"], role="reviewed acoustic feature")
    source = _strict_csv(paths["source_csv"], role="reviewed acoustic source")
    expected_columns = {"session_id", "timestamp_s", *SPEECH_FEATURES}
    if set(feature.columns) != expected_columns or tuple(feature.columns) != (
        "session_id",
        "timestamp_s",
        "array_energy",
        "f0_proxy",
        "voiced",
    ):
        _fail(f"reviewed acoustic columns differ for {scene}")
    if not feature.equals(source):
        _fail(f"reviewed acoustic feature/source copies differ for {scene}")
    if feature.empty or feature["session_id"].astype(str).tolist() != [scene] * len(feature):
        _fail(f"reviewed acoustic session identity differs for {scene}")
    ticks = _timestamps_to_ticks(feature["timestamp_s"], role=f"reviewed acoustic {scene}")
    values = feature.loc[:, SPEECH_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values[:, 1], (0.0, 1.0)).all():
        _fail(f"reviewed acoustic values are invalid for {scene}")
    if np.any((values[:, 1] == 0.0) & (values[:, 2] != 0.0)):
        _fail(f"f0_proxy must be zero when voiced is zero for {scene}")

    feature_manifest = _read_json(paths["feature_manifest"])
    review = _read_json(paths["review_receipt"])
    pending = _read_json(paths["pending_manifest"])
    _read_json(paths["evidence_json"])
    if (
        feature_manifest.get("artifact_kind") != "vap_feature_source"
        or feature_manifest.get("format_version") != 3
        or feature_manifest.get("session_id") != scene
        or feature_manifest.get("purpose") != "model_input"
        or feature_manifest.get("assembly_eligible") is not True
        or feature_manifest.get("counted_input_ready") is not True
        or feature_manifest.get("review_status") != "accepted"
        or feature_manifest.get("supplied_columns") != [
            "array_energy",
            "f0_proxy",
            "voiced",
        ]
        or feature_manifest.get("causality")
        != {"causal": True, "lookahead_s": 0.0, "window_alignment": "trailing"}
        or feature_manifest.get("feature_csv_sha256") != _sha256_file(paths["feature_csv"])
    ):
        _fail(f"reviewed acoustic feature manifest is not accepted/causal for {scene}")
    if (
        review.get("artifact_kind") != "vap_real_causal_acoustic_review_receipt"
        or review.get("format_version") != 1
        or review.get("scene") != scene
        or review.get("decision") != "accepted"
        or review.get("independent_review") is not True
    ):
        _fail(f"independent acoustic review receipt is invalid for {scene}")
    reviewed_bundle = review.get("reviewed_bundle")
    expected_review_paths = {
        "source_csv": paths["source_csv"],
        "evidence_json": paths["evidence_json"],
        "feature_csv": paths["feature_csv"],
        "pending_manifest": paths["pending_manifest"],
    }
    if not isinstance(reviewed_bundle, dict) or any(
        not _binding_matches(reviewed_bundle.get(role), path)
        for role, path in expected_review_paths.items()
    ):
        _fail(f"independent acoustic review no longer binds local bytes for {scene}")
    source_files = feature_manifest.get("source_files")
    expected_source_files = (
        paths["source_csv"],
        paths["evidence_json"],
        paths["pending_manifest"],
        paths["review_receipt"],
    )
    if not isinstance(source_files, list) or len(source_files) != 4 or any(
        not _binding_matches(binding, path)
        for binding, path in zip(source_files, expected_source_files, strict=True)
    ):
        _fail(f"reviewed acoustic manifest source bindings differ for {scene}")
    if pending.get("session_id") != scene or pending.get("review_status") != "review_pending":
        _fail(f"pending acoustic manifest identity differs for {scene}")

    expected_source_keys = {
        "artifact_kind",
        "data_path",
        "data_sha256",
        "format_version",
        "identity_authority",
        "identity_inventory_sha256",
        "provenance_path",
        "provenance_sha256",
    }
    if not isinstance(assembled_source_artifact, dict) or set(
        assembled_source_artifact
    ) != expected_source_keys:
        _fail(f"frozen assembled source binding fields differ for {scene}")
    assembled_csv = _readable_local_path(
        assembled_source_artifact["data_path"], f"assembled CSV {scene}"
    )
    assembled_provenance_path = _readable_local_path(
        assembled_source_artifact["provenance_path"], f"assembled provenance {scene}"
    )
    if (
        assembled_csv.stem != scene
        or assembled_provenance_path.name != f"{scene}.provenance.json"
        or assembled_source_artifact["data_sha256"] != _sha256_file(assembled_csv)
        or assembled_source_artifact["provenance_sha256"]
        != _sha256_file(assembled_provenance_path)
    ):
        _fail(f"frozen assembled source bytes differ for {scene}")
    assembled = _read_json(assembled_provenance_path)
    feature_sources = assembled.get("feature_sources")
    matches = [
        item
        for item in feature_sources
        if isinstance(item, dict)
        and item.get("columns") == ["array_energy", "f0_proxy", "voiced"]
    ] if isinstance(feature_sources, list) else []
    if len(matches) != 1:
        _fail(f"assembled provenance lacks one acoustic descendant for {scene}")
    expected_descendant = {
        "path": str(paths["feature_csv"]),
        "sha256": _sha256_file(paths["feature_csv"]),
        "manifest_path": str(paths["feature_manifest"]),
        "manifest_sha256": _sha256_file(paths["feature_manifest"]),
        "causality": {"causal": True, "lookahead_s": 0.0, "window_alignment": "trailing"},
    }
    if {key: matches[0].get(key) for key in expected_descendant} != expected_descendant:
        _fail(f"assembled acoustic descendant differs from reviewed local bytes for {scene}")

    bindings = {
        "feature_csv": _file_binding(paths["feature_csv"]),
        "feature_manifest": _file_binding(paths["feature_manifest"]),
        "review_receipt": _file_binding(paths["review_receipt"]),
        "pending_manifest": _file_binding(paths["pending_manifest"]),
        "evidence_json": _file_binding(paths["evidence_json"]),
        "assembled_csv": _file_binding(assembled_csv),
        "assembled_provenance": _file_binding(assembled_provenance_path),
        "raw_source_replay": "skipped_sealed_descendant_only",
        "raw_source_paths_dereferenced": False,
    }
    return EOnlyAcousticSession(
        session_id=scene,
        ticks=ticks,
        features=values,
        bindings=bindings,
    )


def _speech_target(session: CalibrationSession) -> np.ndarray:
    target = np.any(session.vad > 0.5, axis=1).astype(np.int8)
    if np.unique(target).size != 2:
        _fail(f"speech detector target is single-class for {session.session_id}")
    return target


def _fit_detector_arrays(features: np.ndarray, target: np.ndarray) -> dict[str, object]:
    if features.ndim != 2 or features.shape[1] != len(SPEECH_FEATURES):
        _fail("speech detector features must have shape (frames, 3)")
    if target.shape != (features.shape[0],) or np.unique(target).size != 2:
        _fail("speech detector fit requires a two-class target")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        random_state=13,
    )
    model.fit(scaled, target)
    if model.n_iter_.max() >= model.max_iter:
        _fail("speech detector did not converge within 1000 iterations")
    return {
        "feature_columns": list(SPEECH_FEATURES),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coefficient": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "estimator": {
            "class": "sklearn.linear_model.LogisticRegression",
            "standardizer": "sklearn.preprocessing.StandardScaler",
            "C": 1.0,
            "penalty": "L2_via_l1_ratio_0",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 0.0001,
            "random_state": 13,
        },
    }


def apply_speech_detector(model: dict[str, object], features: object) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model.get("scaler_mean"), dtype=np.float64)
    scale = np.asarray(model.get("scaler_scale"), dtype=np.float64)
    coefficient = np.asarray(model.get("coefficient"), dtype=np.float64)
    intercept = model.get("intercept")
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or mean.shape != (3,)
        or scale.shape != (3,)
        or coefficient.shape != (3,)
        or not isinstance(intercept, (int, float))
        or isinstance(intercept, bool)
        or not all(np.isfinite(array).all() for array in (values, mean, scale, coefficient))
        or not math.isfinite(float(intercept))
        or (scale <= 0).any()
    ):
        _fail("serialized speech detector is malformed")
    logits = ((values - mean) / scale) @ coefficient + float(intercept)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))


def fit_cross_fitted_speech_detector(
    sessions: tuple[CalibrationSession, ...],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Cross-fit detector predictions by inner group and fit one final outer-train model."""

    if not sessions:
        _fail("speech-detector calibration requires inner OOF sessions")
    groups = tuple(sorted({session.group_id for session in sessions}))
    if len(groups) != 3:
        _fail("speech-detector calibration requires exactly three outer-train groups")
    crossfit: dict[str, np.ndarray] = {}
    fold_records = []
    for held_out in groups:
        train = tuple(session for session in sessions if session.group_id != held_out)
        validation = tuple(session for session in sessions if session.group_id == held_out)
        fit_x = np.concatenate([session.acoustic_features for session in train])
        fit_y = np.concatenate([_speech_target(session) for session in train])
        model = _fit_detector_arrays(fit_x, fit_y)
        for session in validation:
            crossfit[session.session_id] = apply_speech_detector(
                model, session.acoustic_features
            )
        fold_records.append(
            {
                "validation_group_id": held_out,
                "fit_group_ids": [group for group in groups if group != held_out],
                "fit_session_ids": [session.session_id for session in train],
                "validation_session_ids": [session.session_id for session in validation],
                "model": model,
            }
        )
    all_x = np.concatenate([session.acoustic_features for session in sessions])
    all_y = np.concatenate([_speech_target(session) for session in sessions])
    final_model = _fit_detector_arrays(all_x, all_y)
    contract = {
        "target": "current_frame_any_participant_human_reviewed_Transcript_derived_VAD",
        "crossfit_scheme": "leave_one_of_three_outer_train_groups_out",
        "outer_test_data_used": False,
        "outer_train_group_ids": list(groups),
        "folds": fold_records,
        "final_fit_group_ids": list(groups),
        "final_fit_session_ids": [session.session_id for session in sessions],
        "final_model": final_model,
    }
    return contract, crossfit


def select_speech_threshold(
    sessions: tuple[CalibrationSession, ...],
    crossfit_probabilities: dict[str, np.ndarray],
) -> dict[str, object]:
    labels = np.concatenate([_speech_target(session) for session in sessions])
    scores = np.concatenate(
        [crossfit_probabilities[session.session_id] for session in sessions]
    )
    records = []
    for threshold in SPEECH_THRESHOLDS:
        predictions = (scores >= threshold).astype(np.int8)
        records.append(
            {
                "threshold": threshold,
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
                "speech_recall": float(recall_score(labels, predictions, zero_division=0)),
            }
        )
    selected = min(
        records,
        key=lambda item: (
            -item["balanced_accuracy"],
            -item["speech_recall"],
            abs(item["threshold"] - 0.5),
            -item["threshold"],
        ),
    )
    return {
        "selection_rule": (
            "max_frame_balanced_accuracy_then_speech_recall_then_closest_to_0.5"
        ),
        "grid": list(SPEECH_THRESHOLDS),
        "candidates": records,
        "selected": selected,
    }


def replay_matched_policy(
    probabilities: object,
    acoustic_features: object,
    present: object,
    *,
    detector_model: dict[str, object],
    speech_threshold: float,
    reactive_silent_frames: int,
    now_threshold: float,
    future_threshold: float,
    state: MatchedReplayState | None = None,
) -> tuple[dict[str, np.ndarray], MatchedReplayState]:
    """Replay both policies causally; no gold or floor-holder argument exists."""

    probs = np.asarray(probabilities, dtype=np.float64)
    acoustic = np.asarray(acoustic_features, dtype=np.float64)
    participant_mask = np.asarray(present, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[1:] != (4, 4):
        _fail("matched replay probabilities must have shape (frames, 4, 4)")
    if acoustic.shape != (probs.shape[0], 3):
        _fail("matched replay acoustics must align one-to-one with probabilities")
    if participant_mask.shape != (4,) or not np.isin(participant_mask, (0.0, 1.0)).all():
        _fail("matched replay present mask must be binary length four")
    if participant_mask[ROBOT_SLOT] <= 0:
        _fail("demonstration policy requires the P1 robot-role slot")
    thresholds = (speech_threshold, now_threshold, future_threshold)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in thresholds
        )
        or isinstance(reactive_silent_frames, bool)
        or not isinstance(reactive_silent_frames, int)
        or reactive_silent_frames <= 0
    ):
        _fail("matched replay thresholds/state parameters are invalid")
    current = MatchedReplayState() if state is None else state
    speech_probability = apply_speech_detector(detector_model, acoustic)
    speech_detected = (speech_probability >= speech_threshold).astype(np.int8)
    p_now, p_future = p_now_p_future(probs, DEFAULT_LABEL_CONFIG)
    robot_now = np.asarray(p_now[:, ROBOT_SLOT], dtype=np.float64)
    robot_future = np.asarray(p_future[:, ROBOT_SLOT], dtype=np.float64)

    n_frames = probs.shape[0]
    silent_run = np.zeros(n_frames, dtype=np.int64)
    reactive_ready = np.zeros(n_frames, dtype=np.int8)
    reactive_trigger = np.zeros(n_frames, dtype=np.int8)
    predictive_trigger = np.zeros(n_frames, dtype=np.int8)
    run = current.reactive_silent_run
    consecutive = current.predictive_consecutive
    refractory = current.predictive_refractory
    for index in range(n_frames):
        run = 0 if speech_detected[index] else run + 1
        silent_run[index] = run
        reactive_ready[index] = int(run >= reactive_silent_frames)
        reactive_trigger[index] = int(run == reactive_silent_frames)

        if refractory > 0:
            refractory -= 1
            continue
        wants_floor = (
            robot_now[index] > now_threshold
            and robot_future[index] > future_threshold
        )
        consecutive = consecutive + 1 if wants_floor else 0
        if consecutive >= MIN_CONSECUTIVE_FRAMES:
            predictive_trigger[index] = 1
            consecutive = 0
            refractory = REFRACTORY_FRAMES
    return (
        {
            "speech_probability": speech_probability,
            "speech_detected": speech_detected,
            "reactive_silent_run_frames": silent_run,
            "reactive_ready": reactive_ready,
            "reactive_trigger": reactive_trigger,
            "robot_p_now": robot_now,
            "robot_p_future": robot_future,
            "predictive_trigger": predictive_trigger,
        },
        MatchedReplayState(run, consecutive, refractory),
    )


def _event_id(
    session_id: str,
    start_tick: int,
    end_tick: int,
    previous: int,
    next_: int,
) -> str:
    return f"{session_id}:{start_tick}:{end_tick}:P{previous + 1}:P{next_ + 1}"


def _opportunities(session: CalibrationSession) -> list[object]:
    events = extract_shift_hold_events(
        session.vad,
        frame_rate_hz=FRAME_RATE_HZ,
        min_gap_s=0.2,
        max_gap_s=2.0,
        context_s=0.4,
    )
    return [
        event
        for event in events
        if event.previous_speaker != ROBOT_SLOT
        and session.present[ROBOT_SLOT] > 0
        and bool(session.valid[event.gap_start_f : event.gap_end_f].all())
    ]


def score_trace_events(
    session: CalibrationSession,
    trigger: object,
    *,
    pre_gap_frames: int,
    policy_id: str,
) -> list[dict[str, object]]:
    """Classify the frozen robot-entry opportunities with one sealed trigger trace."""

    values = np.asarray(trigger, dtype=np.int8)
    if values.shape != (session.ticks.size,) or not np.isin(values, (0, 1)).all():
        _fail("policy trigger must be one aligned binary vector")
    if (
        isinstance(pre_gap_frames, bool)
        or not isinstance(pre_gap_frames, int)
        or pre_gap_frames < 0
    ):
        _fail("pre_gap_frames must be a non-negative integer")
    records = []
    for index, event in enumerate(_opportunities(session)):
        lo = max(0, event.gap_start_f - pre_gap_frames)
        hi = event.gap_end_f
        local = np.flatnonzero(values[lo:hi] == 1)
        first_frame = None if local.size == 0 else lo + int(local[0])
        start_tick = int(session.ticks[event.gap_start_f])
        end_tick = int(session.ticks[event.gap_end_f])
        records.append(
            {
                "event_id": _event_id(
                    session.session_id,
                    start_tick,
                    end_tick,
                    event.previous_speaker,
                    event.next_speaker,
                ),
                "session_id": session.session_id,
                "group_id": session.group_id,
                "event_index": index,
                "gap_start_tick": start_tick,
                "gap_end_tick_exclusive": end_tick,
                "previous_slot": f"P{event.previous_speaker + 1}",
                "next_slot": f"P{event.next_speaker + 1}",
                "robot_entry_label": int(event.next_speaker == ROBOT_SLOT),
                "policy_id": policy_id,
                "triggered": int(first_frame is not None),
                "first_trigger_tick": (
                    None if first_frame is None else int(session.ticks[first_frame])
                ),
                "latency_s": (
                    None
                    if first_frame is None
                    else float((int(session.ticks[first_frame]) - start_tick) / FRAME_RATE_HZ)
                ),
            }
        )
    return records


def _binary_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([record["robot_entry_label"] for record in records], dtype=np.int8)
    decisions = np.asarray([record["triggered"] for record in records], dtype=np.int8)
    tp = int(np.sum((labels == 1) & (decisions == 1)))
    fp = int(np.sum((labels == 0) & (decisions == 1)))
    fn = int(np.sum((labels == 1) & (decisions == 0)))
    tn = int(np.sum((labels == 0) & (decisions == 0)))

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else float(numerator / denominator)

    recall = rate(tp, tp + fn)
    specificity = rate(tn, tn + fp)
    latencies = [
        float(record["latency_s"])
        for record in records
        if record["robot_entry_label"] == 1
        and record["triggered"] == 1
        and record["latency_s"] is not None
    ]
    return {
        "n_opportunities": len(records),
        "n_positive": tp + fn,
        "n_negative": fp + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "balanced_accuracy": (
            None if recall is None or specificity is None else float((recall + specificity) / 2)
        ),
        "positive_recall": recall,
        "negative_false_entry_rate": rate(fp, fp + tn),
        "positive_hit_latency_mean_s": (
            None if not latencies else float(np.mean(latencies))
        ),
    }


def validate_exact_event_support(records: list[dict[str, object]]) -> None:
    """Enforce the frozen exact-eight support and globally unique event IDs."""

    event_ids = [record.get("event_id") for record in records]
    labels = [record.get("robot_entry_label") for record in records]
    if (
        len(records) != EXPECTED_N_EVENTS
        or sum(label == 1 for label in labels) != EXPECTED_N_POSITIVE
        or sum(label == 0 for label in labels) != EXPECTED_N_NEGATIVE
        or any(not isinstance(value, str) or not value for value in event_ids)
        or len(set(event_ids)) != len(event_ids)
    ):
        _fail("robot-entry support differs from the frozen 303/111/192 unique-ID contract")


def _select_reactive(
    sessions: tuple[CalibrationSession, ...],
    speech_scores: dict[str, np.ndarray],
    speech_threshold: float,
) -> dict[str, object]:
    candidates = []
    for silent_frames in REACTIVE_SILENCE_CANDIDATES:
        records: list[dict[str, object]] = []
        for session in sessions:
            # Use the already cross-fitted probabilities directly.
            speech = speech_scores[session.session_id]
            detected = (speech >= speech_threshold).astype(np.int8)
            run = 0
            trigger = np.zeros(detected.size, dtype=np.int8)
            for frame_index, observed in enumerate(detected):
                run = 0 if observed else run + 1
                trigger[frame_index] = int(run == silent_frames)
            records.extend(
                score_trace_events(
                    session,
                    trigger,
                    pre_gap_frames=PRIMARY_PRE_GAP_FRAMES,
                    policy_id="reactive",
                )
            )
        metrics = _binary_metrics(records)
        candidates.append({"silent_frames": silent_frames, "metrics": metrics})

    def key(item: dict[str, object]) -> tuple[float, float, float, float, int]:
        metrics = item["metrics"]
        ba = metrics["balanced_accuracy"]
        fpr = metrics["negative_false_entry_rate"]
        recall = metrics["positive_recall"]
        latency = metrics["positive_hit_latency_mean_s"]
        return (
            -(float(ba) if ba is not None else -math.inf),
            float(fpr) if fpr is not None else math.inf,
            -(float(recall) if recall is not None else -math.inf),
            float(latency) if latency is not None else math.inf,
            int(item["silent_frames"]),
        )

    selected = min(candidates, key=key)
    return {
        "selection_rule": (
            "max_inner_robot_entry_balanced_accuracy_then_low_FPR_then_high_recall_"
            "then_low_positive_hit_latency"
        ),
        "candidate_silent_frames": list(REACTIVE_SILENCE_CANDIDATES),
        "candidates": candidates,
        "selected": selected,
    }


def _predictive_trigger(
    session: CalibrationSession,
    now_threshold: float,
    future_threshold: float,
) -> np.ndarray:
    # Acoustics and reactive settings are deliberately irrelevant to the
    # demonstration policy.  This direct implementation matches
    # ``DemonstrationPolicy`` and has no floor-holder input.
    p_now, p_future = p_now_p_future(session.probabilities, DEFAULT_LABEL_CONFIG)
    trigger = np.zeros(session.ticks.size, dtype=np.int8)
    consecutive = 0
    refractory = 0
    for index in range(session.ticks.size):
        if refractory > 0:
            refractory -= 1
            continue
        wants = (
            p_now[index, ROBOT_SLOT] > now_threshold
            and p_future[index, ROBOT_SLOT] > future_threshold
        )
        consecutive = consecutive + 1 if wants else 0
        if consecutive >= MIN_CONSECUTIVE_FRAMES:
            trigger[index] = 1
            consecutive = 0
            refractory = REFRACTORY_FRAMES
    return trigger


def _select_predictive(
    sessions: tuple[CalibrationSession, ...],
    *,
    reactive_fpr: float,
) -> dict[str, object]:
    candidates = []
    for now_threshold in PREDICTIVE_THRESHOLDS:
        for future_threshold in PREDICTIVE_THRESHOLDS:
            records: list[dict[str, object]] = []
            for session in sessions:
                records.extend(
                    score_trace_events(
                        session,
                        _predictive_trigger(session, now_threshold, future_threshold),
                        pre_gap_frames=PRIMARY_PRE_GAP_FRAMES,
                        policy_id="predictive_demonstration_P1",
                    )
                )
            metrics = _binary_metrics(records)
            candidates.append(
                {
                    "now_threshold": now_threshold,
                    "future_threshold": future_threshold,
                    "metrics": metrics,
                    "passes_reactive_fpr_constraint": (
                        metrics["negative_false_entry_rate"] is not None
                        and metrics["negative_false_entry_rate"] <= reactive_fpr + 1e-12
                    ),
                }
            )
    feasible = [item for item in candidates if item["passes_reactive_fpr_constraint"]]
    if not feasible:
        _fail("no predictive threshold pair satisfies the inner reactive-FPR constraint")

    def key(item: dict[str, object]) -> tuple[float, float, float, float, float]:
        metrics = item["metrics"]
        recall = metrics["positive_recall"]
        latency = metrics["positive_hit_latency_mean_s"]
        fpr = metrics["negative_false_entry_rate"]
        return (
            -(float(recall) if recall is not None else -math.inf),
            float(latency) if latency is not None else math.inf,
            float(fpr) if fpr is not None else math.inf,
            -float(item["now_threshold"]),
            -float(item["future_threshold"]),
        )

    selected = min(feasible, key=key)
    return {
        "selection_rule": (
            "require_inner_FPR_le_reactive_then_max_recall_then_low_positive_hit_latency_"
            "then_low_FPR_then_higher_thresholds"
        ),
        "grid": list(PREDICTIVE_THRESHOLDS),
        "min_consecutive_frames": MIN_CONSECUTIVE_FRAMES,
        "refractory_frames": REFRACTORY_FRAMES,
        "policy": "DemonstrationPolicy_P1_no_floor_holder",
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "selected": selected,
    }


def _bound_result_artifact(
    unit_directory: Path,
    binding: object,
    *,
    role: str,
    inner_fold_id: str | None = None,
) -> Path:
    required = {"relative_path", "sha256", "size_bytes"}
    if inner_fold_id is not None:
        required.add("inner_fold_id")
    if not isinstance(binding, dict) or set(binding) != required:
        _fail(f"{role} result binding fields are not exact")
    if inner_fold_id is not None and binding["inner_fold_id"] != inner_fold_id:
        _fail(f"{role} inner-fold identity differs")
    relative = binding["relative_path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        _fail(f"{role} result relative path is invalid")
    path = _readable_local_path(unit_directory / relative, role)
    if unit_directory.resolve() not in path.parents:
        _fail(f"{role} result path escapes its unit directory")
    size = binding["size_bytes"]
    if (
        not _is_sha256(binding["sha256"])
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or path.stat().st_size != size
        or _sha256_file(path) != binding["sha256"]
    ):
        _fail(f"{role} result bytes differ from their manifest")
    return path


def _validate_matrix_index(
    matrix_root: Path,
    *,
    budget_id: str,
    expected_epochs: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    index_path = _readable_local_path(matrix_root / "matrix_index.json", "matrix index")
    index = _read_json(index_path)
    required = {
        "artifact_kind", "format_version", "result_role", "data_kind",
        "production_eligible", "counted_eligible", "analysis_variant", "seed",
        "condition_ids", "outer_fold_ids", "expected_unit_count",
        "completed_unit_count", "protocol_directory", "run_manifest_sha256",
        "included_session_ids", "group_ids", "corpus_exclusions", "units",
    }
    if set(index) != required:
        _fail(f"{budget_id} matrix index fields are not exact")
    if (
        index["artifact_kind"] != "vap_thesis_primary_matrix_index"
        or index["format_version"] != 1
        or index["result_role"] != thesis_backend.RESULT_ROLE
        or index["data_kind"] != "real"
        or index["production_eligible"] is not False
        or index["counted_eligible"] is not False
        or index["analysis_variant"] != "primary"
        or index["seed"] != 13
        or index["condition_ids"] != ["A", "H", "A+H"]
        or index["outer_fold_ids"] != list(OUTER_FOLD_IDS)
        or index["expected_unit_count"] != 12
        or index["completed_unit_count"] != 12
        or index["included_session_ids"] != list(DEADLINE_SESSION_TO_GROUP)
        or index["group_ids"] != sorted(set(DEADLINE_SESSION_TO_GROUP.values()))
        or index["corpus_exclusions"] != dict(DEADLINE_CORPUS_EXCLUSIONS)
        or not _is_sha256(index["run_manifest_sha256"])
    ):
        _fail(f"{budget_id} matrix identity is invalid or attempted promotion")

    protocol_directory = _reject_f_path(index["protocol_directory"], "outer protocol")
    split_path = _readable_local_path(
        protocol_directory / "split_manifest.json", "outer split manifest"
    )
    run_path = _readable_local_path(
        protocol_directory / "run_manifest.json", "outer run manifest"
    )
    split = _read_json(split_path)
    run = _read_json(run_path)
    try:
        validate_outer_split_manifest(split)
        validate_outer_run_manifest(run, split)
    except ValueError as exc:
        raise H2MatchedReplayError(f"{budget_id} outer protocol is invalid: {exc}") from exc
    run_payload_sha256 = hashlib.sha256(
        json.dumps(run, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if (
        split.get("protocol_id") != DEADLINE_PROTOCOL_ID
        or split.get("source_session_to_group") != dict(DEADLINE_SESSION_TO_GROUP)
        or split.get("corpus_exclusions") != dict(DEADLINE_CORPUS_EXCLUSIONS)
        or run.get("max_epochs") != expected_epochs
        or run.get("seeds") != [13, 42, 73]
        or index["run_manifest_sha256"] != run_payload_sha256
    ):
        _fail(f"{budget_id} protocol does not match its fixed epoch/corpus contract")
    return index, split, run


def _source_artifacts_by_scene(run: dict[str, object]) -> dict[str, dict[str, object]]:
    readiness = run.get("readiness_by_condition")
    condition = readiness.get("A+H") if isinstance(readiness, dict) else None
    artifacts = condition.get("source_artifacts") if isinstance(condition, dict) else None
    if not isinstance(artifacts, list) or len(artifacts) != 8:
        _fail("A+H frozen readiness must bind eight assembled source artifacts")
    result: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("data_path"), str):
            _fail("A+H assembled source artifact is malformed")
        scene = Path(artifact["data_path"]).stem
        if scene in result or scene not in DEADLINE_SESSION_TO_GROUP:
            _fail("A+H assembled source artifact identity is duplicated or unexpected")
        result[scene] = artifact
    if tuple(sorted(result)) != tuple(sorted(DEADLINE_SESSION_TO_GROUP)):
        _fail("A+H assembled source artifacts do not cover exact-eight")
    return result


def _folds_by_id(split: dict[str, object]) -> dict[str, dict[str, object]]:
    variants = split.get("analysis_variants")
    primary = variants.get("primary") if isinstance(variants, dict) else None
    folds = primary.get("outer_folds") if isinstance(primary, dict) else None
    if not isinstance(folds, list) or len(folds) != 4:
        _fail("outer protocol lacks the four primary folds")
    result = {fold.get("outer_fold_id"): fold for fold in folds if isinstance(fold, dict)}
    if tuple(result) != OUTER_FOLD_IDS:
        _fail("outer protocol primary fold order differs")
    return result


def _matrix_unit_records(index: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    units = index.get("units")
    if not isinstance(units, list) or len(units) != 12:
        _fail("matrix index must bind exactly twelve units")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for unit in units:
        if not isinstance(unit, dict):
            _fail("matrix unit record is malformed")
        key = (unit.get("condition_id"), unit.get("outer_fold_id"))
        if key in result:
            _fail("matrix unit record is duplicated")
        result[key] = unit
    expected = {
        (condition, fold)
        for condition in ("A", "H", "A+H")
        for fold in OUTER_FOLD_IDS
    }
    if set(result) != expected:
        _fail("matrix index does not cover the paired A/H/A+H folds exactly")
    return result


def _planned_binding(final_path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(final_path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_inner_checkpoint_scope(
    checkpoint: dict[str, object],
    prediction: dict[str, object],
    *,
    checkpoint_sha256: str,
    fold: dict[str, object],
    inner: dict[str, object],
) -> None:
    """Require one inner OOF prediction to originate from the other two groups."""

    train_ids = inner["train_session_ids"]
    validation_ids = inner["validation_session_ids"]
    outer_ids = fold["outer_test_session_ids"]
    if (
        checkpoint.get("condition_id") != "A+H"
        or checkpoint.get("outer_fold_id") != fold["outer_fold_id"]
        or checkpoint.get("inner_fold_id") != inner["inner_fold_id"]
        or checkpoint.get("fit_role") != "inner_selection"
        or checkpoint.get("seed") != 13
        or checkpoint.get("train_session_ids") != train_ids
        or set(train_ids) & set(validation_ids)
        or set(train_ids) & set(outer_ids)
        or set(validation_ids) & set(outer_ids)
        or prediction.get("checkpoint_sha256") != checkpoint_sha256
    ):
        _fail(
            f"{fold['outer_fold_id']}/{inner['inner_fold_id']} checkpoint/OOF scope differs"
        )
    observed_train_groups = {DEADLINE_SESSION_TO_GROUP[scene] for scene in train_ids}
    if observed_train_groups != set(inner["train_group_ids"]):
        _fail("inner checkpoint does not train on exactly the other two inner groups")


def _validate_final_checkpoint_scope(
    checkpoint: dict[str, object],
    prediction: dict[str, object],
    *,
    checkpoint_sha256: str,
    fold: dict[str, object],
) -> None:
    """Require the final checkpoint to use exactly the outer-training sessions."""

    train_ids = fold["outer_train_session_ids"]
    test_ids = fold["outer_test_session_ids"]
    if (
        checkpoint.get("condition_id") != "A+H"
        or checkpoint.get("outer_fold_id") != fold["outer_fold_id"]
        or checkpoint.get("inner_fold_id") is not None
        or checkpoint.get("fit_role") != "fixed_epoch_outer_train_from_scratch"
        or checkpoint.get("seed") != 13
        or checkpoint.get("train_session_ids") != train_ids
        or set(train_ids) & set(test_ids)
        or prediction.get("checkpoint_sha256") != checkpoint_sha256
    ):
        _fail(f"{fold['outer_fold_id']} final checkpoint/outer scope differs")


def _freeze_budget(
    matrix_root: Path,
    *,
    destination: Path,
    budget_id: str,
    max_epochs: int,
) -> tuple[dict[str, object], dict[Path, bytes]]:
    index, split, run = _validate_matrix_index(
        matrix_root, budget_id=budget_id, expected_epochs=max_epochs
    )
    folds = _folds_by_id(split)
    units = _matrix_unit_records(index)
    source_artifacts = _source_artifacts_by_scene(run)
    files: dict[Path, bytes] = {}
    fold_records: dict[str, object] = {}
    for fold_id in OUTER_FOLD_IDS:
        unit = units[("A+H", fold_id)]
        result_path = _readable_local_path(unit.get("result_manifest"), "A+H result manifest")
        expected_result = (
            matrix_root / "A_plus_H" / fold_id / "result_manifest.json"
        ).resolve()
        if (
            result_path != expected_result
            or unit.get("seed") != 13
            or unit.get("result_manifest_sha256") != _sha256_file(result_path)
        ):
            _fail(f"{budget_id}/{fold_id} A+H result binding differs from its matrix")
        result = _read_json(result_path)
        fold = folds[fold_id]
        if (
            result.get("result_role") != "backend_contract_smoke"
            or result.get("counted_experiment_complete") is not False
            or result.get("analysis_variant") != "primary"
            or result.get("condition_id") != "A+H"
            or result.get("feature_set") != "audio_plus_head_pose"
            or result.get("outer_fold_id") != fold_id
            or result.get("seed") != 13
            or result.get("outer_test_session_ids") != fold["outer_test_session_ids"]
            or result.get("outer_evaluation_count") != 1
            or result.get("run_manifest_sha256") != index["run_manifest_sha256"]
        ):
            _fail(f"{budget_id}/{fold_id} result identity differs from fixed A+H")
        bindings = result.get("artifact_bindings")
        if not isinstance(bindings, dict):
            _fail(f"{budget_id}/{fold_id} result lacks artifact bindings")
        outer_path = _bound_result_artifact(
            result_path.parent, bindings.get("outer_predictions"), role="outer prediction"
        )
        outer_payload = thesis_backend.load_prediction_artifact(outer_path)
        if [item["session_id"] for item in outer_payload["sessions"]] != fold[
            "outer_test_session_ids"
        ]:
            _fail(f"{budget_id}/{fold_id} outer prediction sessions differ from split")
        view = gold_free_probability_view(
            outer_payload, budget_id=budget_id, max_epochs=max_epochs
        )
        view_bytes = _pretty_json_bytes(view)
        view_relative = Path("outer_probability_views") / budget_id / f"{fold_id}.json"
        files[view_relative] = view_bytes

        inner_bindings = bindings.get("inner_oof_predictions")
        checkpoint_bindings = bindings.get("inner_checkpoints")
        if not isinstance(inner_bindings, list) or len(inner_bindings) != 3:
            _fail(f"{budget_id}/{fold_id} must bind three inner OOF artifacts")
        if not isinstance(checkpoint_bindings, list) or len(checkpoint_bindings) != 3:
            _fail(f"{budget_id}/{fold_id} must bind three inner checkpoints")
        frozen_inner = []
        for inner, binding, checkpoint_binding in zip(
            fold["inner_folds"], inner_bindings, checkpoint_bindings, strict=True
        ):
            inner_id = inner["inner_fold_id"]
            path = _bound_result_artifact(
                result_path.parent,
                binding,
                role="inner OOF prediction",
                inner_fold_id=inner_id,
            )
            payload = thesis_backend._load_prediction_artifact(
                path, expected_inner_fold_id=inner_id
            )
            checkpoint_path = _bound_result_artifact(
                result_path.parent,
                checkpoint_binding,
                role="inner checkpoint",
                inner_fold_id=inner_id,
            )
            checkpoint = thesis_backend.load_checkpoint_artifact(checkpoint_path)
            if (
                payload["condition_id"] != "A+H"
                or payload["outer_fold_id"] != fold_id
                or [item["session_id"] for item in payload["sessions"]]
                != inner["validation_session_ids"]
            ):
                _fail(f"{budget_id}/{fold_id}/{inner_id} inner OOF scope differs")
            _validate_inner_checkpoint_scope(
                checkpoint,
                payload,
                checkpoint_sha256=str(checkpoint_binding["sha256"]),
                fold=fold,
                inner=inner,
            )
            frozen_inner.append(
                {
                    "inner_fold_id": inner_id,
                    "train_group_ids": inner["train_group_ids"],
                    "train_session_ids": inner["train_session_ids"],
                    "validation_group_ids": inner["validation_group_ids"],
                    "validation_session_ids": inner["validation_session_ids"],
                    "artifact": _file_binding(path),
                    "checkpoint_provenance": _file_binding(checkpoint_path),
                }
            )
        final_checkpoint_path = _bound_result_artifact(
            result_path.parent,
            bindings.get("final_checkpoint"),
            role="final checkpoint",
        )
        final_checkpoint = thesis_backend.load_checkpoint_artifact(final_checkpoint_path)
        _validate_final_checkpoint_scope(
            final_checkpoint,
            outer_payload,
            checkpoint_sha256=_sha256_file(final_checkpoint_path),
            fold=fold,
        )
        fold_records[fold_id] = {
            "outer_train_group_ids": fold["outer_train_group_ids"],
            "outer_test_group_ids": fold["outer_test_group_ids"],
            "outer_train_session_ids": fold["outer_train_session_ids"],
            "outer_test_session_ids": fold["outer_test_session_ids"],
            "result_manifest": _file_binding(result_path),
            "inner_oof": frozen_inner,
            "final_checkpoint_provenance": _file_binding(final_checkpoint_path),
            "outer_prediction_provenance": _file_binding(outer_path),
            "gold_free_outer_view": _planned_binding(destination / view_relative, view_bytes),
        }
    protocol_directory = Path(index["protocol_directory"])
    return (
        {
            "budget_id": budget_id,
            "max_epochs": max_epochs,
            "matrix_index": _file_binding(matrix_root / "matrix_index.json"),
            "protocol_directory": str(protocol_directory.resolve()),
            "split_manifest": _file_binding(protocol_directory / "split_manifest.json"),
            "run_manifest": _file_binding(protocol_directory / "run_manifest.json"),
            "run_manifest_payload_sha256": index["run_manifest_sha256"],
            "assembled_source_artifacts": source_artifacts,
            "folds": fold_records,
        },
        files,
    )


def freeze_h2_matched_replay(
    destination: str | Path,
    *,
    primary_matrix: str | Path = PRIMARY_MATRIX,
    sensitivity_matrix: str | Path = SENSITIVITY_MATRIX,
) -> Path:
    """Freeze fixed A+H matrices and publish no-gold outer probability views."""

    output = _reject_f_path(destination, "H2 freeze output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 freeze: {output}")
    primary = _reject_f_path(primary_matrix, "5-epoch matrix")
    sensitivity = _reject_f_path(sensitivity_matrix, "15-epoch matrix")
    if primary.is_symlink() or sensitivity.is_symlink():
        _fail("matrix roots must not be symlinks")
    primary_record, primary_files = _freeze_budget(
        primary, destination=output, budget_id="primary_5ep", max_epochs=5
    )
    sensitivity_record, sensitivity_files = _freeze_budget(
        sensitivity, destination=output, budget_id="sensitivity_15ep", max_epochs=15
    )
    if primary_record["assembled_source_artifacts"] != sensitivity_record[
        "assembled_source_artifacts"
    ]:
        _fail("5-epoch and 15-epoch matrices do not bind identical assembled sources")
    files = {**primary_files, **sensitivity_files}
    projection_seal = {
        "artifact_kind": "vap_h2_gold_free_projection_seal",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "self_contained": True,
        "source_predictions_are_opaque_provenance": True,
        "analysis_order": ["primary_5ep", "sensitivity_15ep"],
        "views": {
            budget_id: {
                fold_id: record["gold_free_outer_view"]
                for fold_id, record in budget["folds"].items()
            }
            for budget_id, budget in (
                ("primary_5ep", primary_record),
                ("sensitivity_15ep", sensitivity_record),
            )
        },
    }
    projection_bytes = _pretty_json_bytes(projection_seal)
    files[Path(PROJECTION_SEAL_FILENAME)] = projection_bytes
    manifest = {
        "artifact_kind": "vap_h2_matched_replay_freeze",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "analysis_order": ["primary_5ep", "sensitivity_15ep"],
        "condition_id": "A+H",
        "seed": 13,
        "corpus": {
            "included_session_ids": list(DEADLINE_SESSION_TO_GROUP),
            "session_to_group": dict(DEADLINE_SESSION_TO_GROUP),
            "row_count": DEADLINE_EXPECTED_N_ROWS,
            "corpus_exclusions": dict(DEADLINE_CORPUS_EXCLUSIONS),
        },
        "budgets": {
            "primary_5ep": primary_record,
            "sensitivity_15ep": sensitivity_record,
        },
        "projection_seal": _planned_binding(
            output / PROJECTION_SEAL_FILENAME, projection_bytes
        ),
        "outer_boundary": {
            "strict_saved_artifact_validated_before_projection": True,
            "trace_phase_allowed_fields": [
                "session_id", "group_id", "tick_25hz", "probabilities", "present"
            ],
            "trace_phase_forbidden_fields": ["labels", "vad", "valid", "floor_holder"],
            "outer_gold_opened_during_freeze_projection": True,
            "reason": (
                "legacy saved artifact embeds gold; projection discards it before trace phase"
            ),
        },
        "acoustic_boundary": {
            "loader": "E_only_sealed_acoustic_descendant",
            "raw_source_replay": "skipped",
            "F_drive_paths_dereferenced": False,
        },
        "implementation": _file_binding(Path(__file__)),
    }
    files[Path(FREEZE_FILENAME)] = _pretty_json_bytes(manifest)
    published = _publish_directory_create_once(output, files)
    audit_h2_matched_replay(published)
    return published / FREEZE_FILENAME


def _load_freeze(path_value: str | Path) -> tuple[Path, dict[str, object]]:
    root = _reject_f_path(path_value, "H2 freeze directory")
    if root.is_symlink() or not root.is_dir():
        _fail("H2 freeze path must be one existing non-symlink directory")
    manifest = _read_json(root / FREEZE_FILENAME)
    required = {
        "artifact_kind", "format_version", "result_role", "production_eligible",
        "counted_eligible", "h2_established", "analysis_order", "condition_id",
        "seed", "corpus", "budgets", "projection_seal", "outer_boundary",
        "acoustic_boundary", "implementation",
    }
    if set(manifest) != required:
        _fail("H2 freeze manifest fields are not exact")
    if (
        manifest["artifact_kind"] != "vap_h2_matched_replay_freeze"
        or manifest["format_version"] != 1
        or manifest["result_role"] != RESULT_ROLE
        or manifest["production_eligible"] is not False
        or manifest["counted_eligible"] is not False
        or manifest["h2_established"] is not False
        or manifest["analysis_order"] != ["primary_5ep", "sensitivity_15ep"]
        or manifest["condition_id"] != "A+H"
        or manifest["seed"] != 13
    ):
        _fail("H2 freeze identity is invalid or attempted promotion")
    verify_file_binding(manifest["implementation"], role="H2 implementation")
    budgets = manifest["budgets"]
    if not isinstance(budgets, dict) or tuple(budgets) != (
        "primary_5ep", "sensitivity_15ep"
    ):
        _fail("H2 freeze budgets are not exact")
    for budget_id, expected_epochs in (("primary_5ep", 5), ("sensitivity_15ep", 15)):
        budget = budgets[budget_id]
        if not isinstance(budget, dict) or budget.get("max_epochs") != expected_epochs:
            _fail(f"{budget_id} freeze budget is malformed")
        for role in ("matrix_index", "split_manifest", "run_manifest"):
            _validate_opaque_binding(budget.get(role), role=f"{budget_id} {role}")
        folds = budget.get("folds")
        if not isinstance(folds, dict) or tuple(folds) != OUTER_FOLD_IDS:
            _fail(f"{budget_id} freeze folds differ")
        for fold_id, fold in folds.items():
            if not isinstance(fold, dict):
                _fail(f"{budget_id}/{fold_id} freeze fold is malformed")
            _validate_opaque_binding(fold.get("result_manifest"), role="result manifest")
            _validate_opaque_binding(
                fold.get("final_checkpoint_provenance"), role="final checkpoint provenance"
            )
            _validate_opaque_binding(
                fold.get("outer_prediction_provenance"),
                role="outer prediction provenance",
            )
            view_path = verify_file_binding(
                fold.get("gold_free_outer_view"), role="gold-free outer view"
            )
            if root not in view_path.parents:
                _fail("gold-free view is not owned by its freeze directory")
            load_gold_free_probability_view(view_path)
            inner = fold.get("inner_oof")
            if not isinstance(inner, list) or len(inner) != 3:
                _fail(f"{budget_id}/{fold_id} freeze lacks three inner OOF bindings")
            for record, inner_id in zip(inner, INNER_FOLD_IDS, strict=True):
                if not isinstance(record, dict) or record.get("inner_fold_id") != inner_id:
                    _fail(f"{budget_id}/{fold_id} inner OOF order differs")
                _validate_opaque_binding(record.get("artifact"), role="inner OOF artifact")
                _validate_opaque_binding(
                    record.get("checkpoint_provenance"),
                    role="inner checkpoint provenance",
                )
    projection_path = verify_file_binding(
        manifest["projection_seal"], role="gold-free projection seal"
    )
    if projection_path != (root / PROJECTION_SEAL_FILENAME).resolve():
        _fail("gold-free projection seal is not owned by its freeze directory")
    projection = _read_json(projection_path)
    if (
        set(projection)
        != {
            "artifact_kind", "format_version", "result_role", "production_eligible",
            "counted_eligible", "self_contained",
            "source_predictions_are_opaque_provenance", "analysis_order", "views",
        }
        or projection["artifact_kind"] != "vap_h2_gold_free_projection_seal"
        or projection["format_version"] != 1
        or projection["result_role"] != RESULT_ROLE
        or projection["production_eligible"] is not False
        or projection["counted_eligible"] is not False
        or projection["self_contained"] is not True
        or projection["source_predictions_are_opaque_provenance"] is not True
        or projection["analysis_order"] != manifest["analysis_order"]
    ):
        _fail("gold-free projection seal identity is invalid")
    expected_views = {
        budget_id: {
            fold_id: fold["gold_free_outer_view"]
            for fold_id, fold in budgets[budget_id]["folds"].items()
        }
        for budget_id in manifest["analysis_order"]
    }
    if projection["views"] != expected_views:
        _fail("gold-free projection seal does not bind the exact frozen views")
    return root, manifest


def _calibration_session(
    record: dict[str, object], acoustic: EOnlyAcousticSession
) -> CalibrationSession:
    view = _validate_probability_session(record)
    vad = np.asarray(record["vad"], dtype=np.float64)
    valid = np.asarray(record["valid"], dtype=bool)
    if vad.shape != (view.ticks.size, 4) or valid.shape != (view.ticks.size,):
        _fail("inner OOF gold shape differs from its probabilities")
    if not np.array_equal(view.ticks, acoustic.ticks):
        _fail(f"inner OOF and acoustic keys differ for {view.session_id}")
    return CalibrationSession(
        session_id=view.session_id,
        group_id=view.group_id,
        ticks=view.ticks,
        probabilities=view.probabilities,
        vad=vad,
        valid=valid,
        present=view.present,
        acoustic_features=acoustic.features,
    )


def _load_inner_calibration_sessions(
    budget: dict[str, object],
    fold_id: str,
    *,
    acoustic_root: Path,
) -> tuple[tuple[CalibrationSession, ...], dict[str, EOnlyAcousticSession]]:
    fold = budget["folds"][fold_id]
    source_artifacts = budget["assembled_source_artifacts"]
    sessions: list[CalibrationSession] = []
    acoustics: dict[str, EOnlyAcousticSession] = {}
    for inner_record in fold["inner_oof"]:
        inner_id = inner_record["inner_fold_id"]
        path = verify_file_binding(inner_record["artifact"], role="inner OOF artifact")
        payload = thesis_backend._load_prediction_artifact(
            path, expected_inner_fold_id=inner_id
        )
        if payload["outer_fold_id"] != fold_id or payload["condition_id"] != "A+H":
            _fail("inner OOF identity differs during calibration")
        for record in payload["sessions"]:
            scene = record["session_id"]
            if scene not in fold["outer_train_session_ids"]:
                _fail("outer-test session attempted to enter inner calibration")
            if scene not in acoustics:
                acoustics[scene] = load_e_only_acoustic_descendant(
                    scene,
                    assembled_source_artifact=source_artifacts[scene],
                    acoustic_root=acoustic_root,
                )
            sessions.append(_calibration_session(record, acoustics[scene]))
    if (
        [session.session_id for session in sessions] != fold["outer_train_session_ids"]
        or set(session.group_id for session in sessions) != set(fold["outer_train_group_ids"])
        or set(session.group_id for session in sessions) & set(fold["outer_test_group_ids"])
    ):
        _fail("inner calibration scope is not exactly the three outer-train groups")
    return tuple(sessions), acoustics


def _calibrate_fold(sessions: tuple[CalibrationSession, ...]) -> dict[str, object]:
    detector, crossfit = fit_cross_fitted_speech_detector(sessions)
    speech_threshold = select_speech_threshold(sessions, crossfit)
    selected_speech = speech_threshold["selected"]["threshold"]
    reactive = _select_reactive(sessions, crossfit, selected_speech)
    reactive_fpr = reactive["selected"]["metrics"]["negative_false_entry_rate"]
    if reactive_fpr is None:
        _fail("reactive calibration has no inner negative support")
    predictive = _select_predictive(sessions, reactive_fpr=float(reactive_fpr))
    return {
        "artifact_kind": "vap_h2_inner_only_calibration",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "calibration_scope": "three_inner_group_OOF_only_no_outer_test",
        "speech_detector": detector,
        "speech_threshold": speech_threshold,
        "reactive_policy": reactive,
        "predictive_policy": predictive,
    }


def _outer_trace_frame(
    views: tuple[GoldFreeSessionView, ...],
    acoustics: dict[str, EOnlyAcousticSession],
    calibration: dict[str, object],
    *,
    fold_id: str,
) -> pd.DataFrame:
    detector = calibration["speech_detector"]["final_model"]
    speech_threshold = calibration["speech_threshold"]["selected"]["threshold"]
    silent_frames = calibration["reactive_policy"]["selected"]["silent_frames"]
    predictive = calibration["predictive_policy"]["selected"]
    pieces = []
    for view in views:
        acoustic = acoustics[view.session_id]
        if not np.array_equal(view.ticks, acoustic.ticks):
            _fail(f"outer probability/acoustic keys differ for {view.session_id}")
        trace, _ = replay_matched_policy(
            view.probabilities,
            acoustic.features,
            view.present,
            detector_model=detector,
            speech_threshold=float(speech_threshold),
            reactive_silent_frames=int(silent_frames),
            now_threshold=float(predictive["now_threshold"]),
            future_threshold=float(predictive["future_threshold"]),
        )
        pieces.append(
            pd.DataFrame(
                {
                    "session_id": view.session_id,
                    "group_id": view.group_id,
                    "outer_fold_id": fold_id,
                    "tick_25hz": view.ticks,
                    "timestamp_s": view.ticks.astype(np.float64) / FRAME_RATE_HZ,
                    **trace,
                },
                columns=TRACE_COLUMNS,
            )
        )
    return pd.concat(pieces, ignore_index=True)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def calibrate_and_seal_traces(
    freeze_directory: str | Path,
    destination: str | Path,
    *,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> Path:
    """Calibrate on inner OOF only and seal outer traces without opening outer gold."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    output = _reject_f_path(destination, "H2 trace output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 traces: {output}")
    local_acoustic_root = _reject_f_path(acoustic_root, "acoustic descendant root")
    files: dict[Path, bytes] = {}
    records: dict[str, object] = {}
    for budget_id in freeze["analysis_order"]:
        budget = freeze["budgets"][budget_id]
        budget_records: dict[str, object] = {}
        for fold_id in OUTER_FOLD_IDS:
            fold = budget["folds"][fold_id]
            sessions, inner_acoustics = _load_inner_calibration_sessions(
                budget, fold_id, acoustic_root=local_acoustic_root
            )
            calibration = _calibrate_fold(sessions)
            calibration.update(
                {
                    "budget_id": budget_id,
                    "max_epochs": budget["max_epochs"],
                    "outer_fold_id": fold_id,
                    "outer_train_group_ids": fold["outer_train_group_ids"],
                    "outer_test_group_ids": fold["outer_test_group_ids"],
                    "outer_train_session_ids": fold["outer_train_session_ids"],
                    "outer_test_session_ids": fold["outer_test_session_ids"],
                    "inner_oof_sources": [item["artifact"] for item in fold["inner_oof"]],
                    "inner_acoustic_sources": {
                        scene: artifact.bindings for scene, artifact in inner_acoustics.items()
                    },
                    "outer_group_used_for_fit": False,
                }
            )
            calibration_rel = Path("calibration") / budget_id / f"{fold_id}.json"
            calibration_bytes = _pretty_json_bytes(calibration)
            files[calibration_rel] = calibration_bytes

            view_path = verify_file_binding(
                fold["gold_free_outer_view"], role="gold-free outer view"
            )
            views = load_gold_free_probability_view(view_path)
            if [view.session_id for view in views] != fold["outer_test_session_ids"]:
                _fail("gold-free outer view differs from the held-out split")
            outer_acoustics = {
                view.session_id: load_e_only_acoustic_descendant(
                    view.session_id,
                    assembled_source_artifact=budget["assembled_source_artifacts"][
                        view.session_id
                    ],
                    acoustic_root=local_acoustic_root,
                )
                for view in views
            }
            trace = _outer_trace_frame(views, outer_acoustics, calibration, fold_id=fold_id)
            trace_rel = Path("traces") / budget_id / f"{fold_id}.csv"
            trace_bytes = _csv_bytes(trace)
            files[trace_rel] = trace_bytes
            budget_records[fold_id] = {
                "calibration": _planned_binding(output / calibration_rel, calibration_bytes),
                "trace": {
                    **_planned_binding(output / trace_rel, trace_bytes),
                    "row_count": len(trace),
                    "ordered_columns": list(TRACE_COLUMNS),
                },
                "gold_free_outer_view": fold["gold_free_outer_view"],
                "outer_acoustic_sources": {
                    scene: artifact.bindings for scene, artifact in outer_acoustics.items()
                },
            }
        records[budget_id] = budget_records
    seal = {
        "artifact_kind": "vap_h2_matched_replay_trace_seal",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "freeze_manifest": _file_binding(freeze_root / FREEZE_FILENAME),
        "analysis_order": freeze["analysis_order"],
        "trace_generation_boundary": {
            "outer_gold_opened": False,
            "outer_fields_consumed": [
                "session_id", "group_id", "tick_25hz", "probabilities", "present"
            ],
            "floor_holder_consumed": False,
            "raw_audio_replayed": False,
            "F_drive_paths_dereferenced": False,
        },
        "policies": {
            "reactive": "causal_acoustic_speech_detector_plus_selected_silence_run",
            "predictive": "DemonstrationPolicy_P1_dual_threshold",
            "min_consecutive_frames": MIN_CONSECUTIVE_FRAMES,
            "refractory_frames": REFRACTORY_FRAMES,
        },
        "budgets": records,
        "implementation": _file_binding(Path(__file__)),
    }
    files[Path(TRACE_SEAL_FILENAME)] = _pretty_json_bytes(seal)
    published = _publish_directory_create_once(output, files)
    audit_h2_matched_replay(
        freeze_root,
        trace_directory=published,
        acoustic_root=local_acoustic_root,
    )
    return published / TRACE_SEAL_FILENAME


def _verify_trace_binding(binding: object, *, role: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {
        "path", "sha256", "size_bytes", "row_count", "ordered_columns"
    }:
        _fail(f"{role} trace binding fields are not exact")
    return verify_file_binding(
        {key: binding[key] for key in ("path", "sha256", "size_bytes")}, role=role
    )


def _load_trace_seal(
    path_value: str | Path,
    *,
    freeze_root: Path,
    freeze: dict[str, object],
    acoustic_root: Path,
) -> tuple[Path, dict[str, object]]:
    root = _reject_f_path(path_value, "H2 trace directory")
    if root.is_symlink() or not root.is_dir():
        _fail("H2 trace directory must be an existing non-symlink directory")
    seal = _read_json(root / TRACE_SEAL_FILENAME)
    required = {
        "artifact_kind", "format_version", "result_role", "production_eligible",
        "counted_eligible", "h2_established", "freeze_manifest", "analysis_order",
        "trace_generation_boundary", "policies", "budgets", "implementation",
    }
    if set(seal) != required:
        _fail("H2 trace seal fields are not exact")
    if (
        seal["artifact_kind"] != "vap_h2_matched_replay_trace_seal"
        or seal["format_version"] != 1
        or seal["result_role"] != RESULT_ROLE
        or seal["production_eligible"] is not False
        or seal["counted_eligible"] is not False
        or seal["h2_established"] is not False
        or seal["analysis_order"] != ["primary_5ep", "sensitivity_15ep"]
        or seal["trace_generation_boundary"].get("outer_gold_opened") is not False
        or seal["trace_generation_boundary"].get("floor_holder_consumed") is not False
        or seal["trace_generation_boundary"].get("raw_audio_replayed") is not False
    ):
        _fail("H2 trace seal identity/boundary is invalid or attempted promotion")
    frozen = verify_file_binding(seal["freeze_manifest"], role="trace-bound freeze")
    if frozen != (freeze_root / FREEZE_FILENAME).resolve():
        _fail("trace seal binds a different freeze manifest")
    verify_file_binding(seal["implementation"], role="trace implementation")
    budgets = seal["budgets"]
    if not isinstance(budgets, dict) or tuple(budgets) != (
        "primary_5ep", "sensitivity_15ep"
    ):
        _fail("trace seal budgets are not exact")
    for budget_id, folds in budgets.items():
        if not isinstance(folds, dict) or tuple(folds) != OUTER_FOLD_IDS:
            _fail(f"{budget_id} trace folds are not exact")
        for fold_id, record in folds.items():
            if not isinstance(record, dict) or set(record) != {
                "calibration", "trace", "gold_free_outer_view", "outer_acoustic_sources"
            }:
                _fail(f"{budget_id}/{fold_id} trace record fields are not exact")
            calibration_path = verify_file_binding(
                record["calibration"], role="inner-only calibration"
            )
            trace_path = _verify_trace_binding(record["trace"], role="sealed trace")
            if root not in calibration_path.parents or root not in trace_path.parents:
                _fail("trace/calibration artifact is not owned by its seal directory")
            calibration = _read_json(calibration_path)
            frozen_fold = freeze["budgets"][budget_id]["folds"][fold_id]
            frozen_budget = freeze["budgets"][budget_id]
            if (
                calibration.get("calibration_scope")
                != "three_inner_group_OOF_only_no_outer_test"
                or calibration.get("outer_group_used_for_fit") is not False
                or calibration.get("budget_id") != budget_id
                or calibration.get("max_epochs") != frozen_budget["max_epochs"]
                or calibration.get("outer_fold_id") != fold_id
                or calibration.get("outer_train_group_ids")
                != frozen_fold["outer_train_group_ids"]
                or calibration.get("outer_test_group_ids")
                != frozen_fold["outer_test_group_ids"]
                or calibration.get("outer_train_session_ids")
                != frozen_fold["outer_train_session_ids"]
                or calibration.get("outer_test_session_ids")
                != frozen_fold["outer_test_session_ids"]
                or set(calibration.get("outer_train_group_ids", []))
                & set(calibration.get("outer_test_group_ids", []))
            ):
                _fail("calibration scope includes or ambiguously binds an outer-test group")
            if record["gold_free_outer_view"] != frozen_fold["gold_free_outer_view"]:
                _fail("trace record does not bind its frozen gold-free view")
            view_path = verify_file_binding(
                frozen_fold["gold_free_outer_view"], role="trace replay gold-free view"
            )
            if freeze_root not in view_path.parents:
                _fail("trace replay view is not owned by the freeze directory")
            views = load_gold_free_probability_view(view_path)
            if [view.session_id for view in views] != frozen_fold["outer_test_session_ids"]:
                _fail("trace replay view sessions differ from its frozen outer fold")
            acoustics = {
                view.session_id: load_e_only_acoustic_descendant(
                    view.session_id,
                    assembled_source_artifact=frozen_budget["assembled_source_artifacts"][
                        view.session_id
                    ],
                    acoustic_root=acoustic_root,
                )
                for view in views
            }
            replayed_sources = {
                scene: acoustic.bindings for scene, acoustic in acoustics.items()
            }
            if replayed_sources != record["outer_acoustic_sources"]:
                _fail("trace acoustic descendants differ from their sealed bindings")
            replayed = _outer_trace_frame(views, acoustics, calibration, fold_id=fold_id)
            replayed_bytes = _csv_bytes(replayed)
            try:
                observed_bytes = trace_path.read_bytes()
            except OSError as exc:
                raise H2MatchedReplayError(f"cannot read sealed trace: {exc}") from exc
            if observed_bytes != replayed_bytes:
                _fail("sealed trace bytes differ from calibration/view/acoustic replay")
            try:
                trace = pd.read_csv(trace_path)
            except (OSError, pd.errors.ParserError) as exc:
                raise H2MatchedReplayError(f"cannot parse sealed trace: {exc}") from exc
            if (
                tuple(trace.columns) != TRACE_COLUMNS
                or len(trace) != record["trace"]["row_count"]
                or record["trace"]["ordered_columns"] != list(TRACE_COLUMNS)
                or "floor_holder" in trace.columns
            ):
                _fail("sealed trace schema/row count differs")
    return root, seal


def _gold_session_from_record(record: dict[str, object]) -> CalibrationSession:
    view = _validate_probability_session(record)
    vad = np.asarray(record["vad"], dtype=np.float64)
    valid = np.asarray(record["valid"], dtype=bool)
    if vad.shape != (view.ticks.size, 4) or valid.shape != (view.ticks.size,):
        _fail("outer evaluation gold shape differs from probability rows")
    return CalibrationSession(
        session_id=view.session_id,
        group_id=view.group_id,
        ticks=view.ticks,
        probabilities=view.probabilities,
        vad=vad,
        valid=valid,
        present=view.present,
        acoustic_features=np.zeros((view.ticks.size, 3), dtype=np.float64),
    )


def _trace_for_session(frame: pd.DataFrame, session: CalibrationSession) -> pd.DataFrame:
    rows = frame.loc[frame["session_id"].astype(str) == session.session_id]
    if rows.empty:
        _fail(f"sealed trace lacks outer session {session.session_id}")
    ticks = rows["tick_25hz"].to_numpy(dtype=np.int64)
    if not np.array_equal(ticks, session.ticks):
        _fail(f"sealed trace keys differ from evaluation-only gold for {session.session_id}")
    return rows


def _legacy_crosscheck(event_ids: set[str]) -> dict[str, object]:
    if not LEGACY_REACTIVE_EVENT.is_file():
        return {
            "role": "non_authoritative_crosscheck_only",
            "present": False,
            "expected_sha256": LEGACY_REACTIVE_EVENT_SHA256,
            "observed_sha256": None,
            "sha256_matches": False,
            "event_ids_match": None,
        }
    observed = _sha256_file(LEGACY_REACTIVE_EVENT)
    id_match: bool | None = None
    if observed == LEGACY_REACTIVE_EVENT_SHA256:
        try:
            legacy = pd.read_csv(LEGACY_REACTIVE_EVENT, usecols=["event_id"])
            id_match = set(legacy["event_id"].astype(str)) == event_ids
        except (OSError, ValueError, pd.errors.ParserError):
            id_match = False
    return {
        "role": "non_authoritative_crosscheck_only",
        "present": True,
        "expected_sha256": LEGACY_REACTIVE_EVENT_SHA256,
        "observed_sha256": observed,
        "sha256_matches": observed == LEGACY_REACTIVE_EVENT_SHA256,
        "event_ids_match": id_match,
    }


def _rebuild_evaluation_payloads(
    freeze: dict[str, object],
    seal: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], set[str]]:
    """Rebuild canonical event and metric payloads from sealed traces plus outer gold."""

    all_records: list[dict[str, object]] = []
    pooled: dict[str, object] = {}
    reference_event_ids: set[str] | None = None
    for budget_id in freeze["analysis_order"]:
        budget = freeze["budgets"][budget_id]
        budget_metrics: dict[str, object] = {}
        fold_inputs: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
        for fold_id in OUTER_FOLD_IDS:
            fold = budget["folds"][fold_id]
            prediction_path = verify_file_binding(
                fold["outer_prediction_provenance"],
                role="evaluation-only outer prediction",
            )
            prediction = thesis_backend.load_prediction_artifact(prediction_path)
            if (
                prediction.get("condition_id") != "A+H"
                or prediction.get("outer_fold_id") != fold_id
                or prediction.get("seed") != 13
                or prediction.get("checkpoint_sha256")
                != fold["final_checkpoint_provenance"]["sha256"]
                or [record["session_id"] for record in prediction["sessions"]]
                != fold["outer_test_session_ids"]
            ):
                _fail("evaluation-only outer prediction differs from its freeze")
            trace_binding = seal["budgets"][budget_id][fold_id]["trace"]
            trace_path = _verify_trace_binding(trace_binding, role="sealed trace")
            try:
                trace = pd.read_csv(trace_path)
            except (OSError, pd.errors.ParserError) as exc:
                raise H2MatchedReplayError(f"cannot parse sealed trace: {exc}") from exc
            fold_inputs[fold_id] = (prediction, trace)
        for policy_id, trigger_column in (
            ("reactive", "reactive_trigger"),
            ("predictive_demonstration_P1", "predictive_trigger"),
        ):
            policy_metrics: dict[str, object] = {}
            for window_id, pre_gap_frames in (
                ("primary_pre_gap_0.4s_to_gap_end", PRIMARY_PRE_GAP_FRAMES),
                ("sensitivity_gap_only", 0),
            ):
                records: list[dict[str, object]] = []
                fold_metrics: dict[str, object] = {}
                for fold_id in OUTER_FOLD_IDS:
                    prediction, trace = fold_inputs[fold_id]
                    fold_records: list[dict[str, object]] = []
                    for source_record in prediction["sessions"]:
                        session = _gold_session_from_record(source_record)
                        rows = _trace_for_session(trace, session)
                        event_rows = score_trace_events(
                            session,
                            rows[trigger_column].to_numpy(dtype=np.int8),
                            pre_gap_frames=pre_gap_frames,
                            policy_id=policy_id,
                        )
                        for event in event_rows:
                            event.update(
                                {
                                    "budget_id": budget_id,
                                    "max_epochs": budget["max_epochs"],
                                    "outer_fold_id": fold_id,
                                    "window_id": window_id,
                                }
                            )
                        fold_records.extend(event_rows)
                    fold_metrics[fold_id] = _binary_metrics(fold_records)
                    records.extend(fold_records)
                validate_exact_event_support(records)
                ids = {str(record["event_id"]) for record in records}
                if reference_event_ids is None:
                    reference_event_ids = ids
                elif ids != reference_event_ids:
                    _fail("budgets/policies/windows do not score identical event IDs")
                policy_metrics[window_id] = {
                    "pooled": _binary_metrics(records),
                    "outer_folds": fold_metrics,
                }
                all_records.extend(records)
            budget_metrics[policy_id] = policy_metrics
        pooled[budget_id] = budget_metrics
    if reference_event_ids is None:
        _fail("H2 evaluation produced no robot-entry events")
    event_frame = pd.DataFrame(all_records, columns=EVENT_COLUMNS)
    metrics = {
        "artifact_kind": "vap_h2_matched_replay_metrics",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "primary_budget": "primary_5ep",
        "sensitivity_budget": "sensitivity_15ep",
        "primary_window": "primary_pre_gap_0.4s_to_gap_end",
        "sensitivity_window": "sensitivity_gap_only",
        "event_support_per_budget_policy_window": {
            "total": EXPECTED_N_EVENTS,
            "positive": EXPECTED_N_POSITIVE,
            "negative": EXPECTED_N_NEGATIVE,
        },
        "results": pooled,
        "claim_boundary": (
            "posthoc matched offline replay; no robot action feedback, TTS startup, "
            "far-field robot-ear audio, or live deployment"
        ),
    }
    return event_frame, metrics, reference_event_ids


def evaluate_h2_matched_replay(
    freeze_directory: str | Path,
    trace_directory: str | Path,
    destination: str | Path,
    *,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> Path:
    """Evaluate sealed traces against outer gold in a separately gated phase."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    local_acoustic_root = _reject_f_path(acoustic_root, "acoustic descendant root")
    trace_root, seal = _load_trace_seal(
        trace_directory,
        freeze_root=freeze_root,
        freeze=freeze,
        acoustic_root=local_acoustic_root,
    )
    output = _reject_f_path(destination, "H2 evaluation output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 evaluation: {output}")

    event_frame, metrics, reference_event_ids = _rebuild_evaluation_payloads(freeze, seal)
    event_bytes = _csv_bytes(event_frame)
    metrics_bytes = _pretty_json_bytes(metrics)
    event_path = output / EVENT_FILENAME
    metrics_path = output / METRICS_FILENAME
    evaluation = {
        "artifact_kind": "vap_h2_matched_replay_evaluation",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "freeze_manifest": _file_binding(freeze_root / FREEZE_FILENAME),
        "trace_seal_manifest": _file_binding(trace_root / TRACE_SEAL_FILENAME),
        "outer_gold_opened_only_after_trace_seal": True,
        "event_csv": {
            **_planned_binding(event_path, event_bytes),
            "row_count": len(event_frame),
            "ordered_columns": list(EVENT_COLUMNS),
        },
        "metrics": _planned_binding(metrics_path, metrics_bytes),
        "legacy_voiced_diagnostic": _legacy_crosscheck(reference_event_ids),
        "scientific_boundary": {
            "same_303_event_ids_across_all_comparisons": True,
            "main_window": "pre_gap_0.4s_to_gap_end",
            "gap_only_is_sensitivity": True,
            "outer_threshold_tuning": False,
            "live_robot_or_TTS_latency_simulated": False,
            "production_eligible": False,
            "counted_eligible": False,
        },
        "implementation": _file_binding(Path(__file__)),
    }
    files = {
        Path(EVENT_FILENAME): event_bytes,
        Path(METRICS_FILENAME): metrics_bytes,
        Path(EVALUATION_FILENAME): _pretty_json_bytes(evaluation),
    }
    published = _publish_directory_create_once(output, files)
    audit_h2_matched_replay(
        freeze_root,
        trace_directory=trace_root,
        evaluation_directory=published,
        acoustic_root=local_acoustic_root,
    )
    return published / EVALUATION_FILENAME


def audit_h2_matched_replay(
    freeze_directory: str | Path,
    *,
    trace_directory: str | Path | None = None,
    evaluation_directory: str | Path | None = None,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> dict[str, object]:
    """Replay-audit any completed prefix of the create-once H2 pipeline."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    local_acoustic_root = _reject_f_path(acoustic_root, "acoustic descendant root")
    result: dict[str, object] = {
        "freeze": "passed",
        "trace_seal": "not_requested",
        "evaluation": "not_requested",
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
    }
    trace_root: Path | None = None
    trace_seal: dict[str, object] | None = None
    if trace_directory is not None:
        trace_root, trace_seal = _load_trace_seal(
            trace_directory,
            freeze_root=freeze_root,
            freeze=freeze,
            acoustic_root=local_acoustic_root,
        )
        result["trace_seal"] = "passed"
    if evaluation_directory is not None:
        if trace_root is None:
            _fail("evaluation audit requires the trace directory")
        evaluation_root = _reject_f_path(evaluation_directory, "H2 evaluation directory")
        if evaluation_root.is_symlink() or not evaluation_root.is_dir():
            _fail("H2 evaluation directory must be an existing non-symlink directory")
        manifest = _read_json(evaluation_root / EVALUATION_FILENAME)
        required = {
            "artifact_kind", "format_version", "result_role", "production_eligible",
            "counted_eligible", "h2_established", "freeze_manifest",
            "trace_seal_manifest", "outer_gold_opened_only_after_trace_seal",
            "event_csv", "metrics", "legacy_voiced_diagnostic",
            "scientific_boundary", "implementation",
        }
        if (
            set(manifest) != required
            or manifest.get("artifact_kind") != "vap_h2_matched_replay_evaluation"
            or manifest.get("format_version") != 1
            or manifest.get("result_role") != RESULT_ROLE
            or manifest.get("production_eligible") is not False
            or manifest.get("counted_eligible") is not False
            or manifest.get("h2_established") is not False
            or manifest.get("outer_gold_opened_only_after_trace_seal") is not True
        ):
            _fail("H2 evaluation identity is invalid or attempted promotion")
        frozen = verify_file_binding(manifest.get("freeze_manifest"), role="evaluation freeze")
        sealed = verify_file_binding(
            manifest.get("trace_seal_manifest"), role="evaluation trace seal"
        )
        if (
            frozen != (freeze_root / FREEZE_FILENAME).resolve()
            or sealed != (trace_root / TRACE_SEAL_FILENAME).resolve()
        ):
            _fail("H2 evaluation binds different freeze/trace artifacts")
        verify_file_binding(manifest.get("implementation"), role="evaluation implementation")
        expected_boundary = {
            "same_303_event_ids_across_all_comparisons": True,
            "main_window": "pre_gap_0.4s_to_gap_end",
            "gap_only_is_sensitivity": True,
            "outer_threshold_tuning": False,
            "live_robot_or_TTS_latency_simulated": False,
            "production_eligible": False,
            "counted_eligible": False,
        }
        if manifest.get("scientific_boundary") != expected_boundary:
            _fail("H2 evaluation scientific boundary differs")
        event_binding = manifest.get("event_csv")
        if not isinstance(event_binding, dict) or set(event_binding) != {
            "path", "sha256", "size_bytes", "row_count", "ordered_columns"
        }:
            _fail("H2 evaluation event binding fields are not exact")
        event_path = verify_file_binding(
            {key: event_binding[key] for key in ("path", "sha256", "size_bytes")},
            role="H2 event CSV",
        )
        metrics_path = verify_file_binding(manifest.get("metrics"), role="H2 metrics")
        if evaluation_root not in event_path.parents or evaluation_root not in metrics_path.parents:
            _fail("H2 evaluation outputs are not owned by their directory")
        if trace_seal is None:
            _fail("evaluation audit lacks a validated trace seal")
        rebuilt_events, rebuilt_metrics, event_ids = _rebuild_evaluation_payloads(
            freeze, trace_seal
        )
        rebuilt_event_bytes = _csv_bytes(rebuilt_events)
        rebuilt_metrics_bytes = _pretty_json_bytes(rebuilt_metrics)
        try:
            observed_event_bytes = event_path.read_bytes()
            observed_metrics_bytes = metrics_path.read_bytes()
        except OSError as exc:
            raise H2MatchedReplayError(f"cannot read H2 evaluation outputs: {exc}") from exc
        if observed_event_bytes != rebuilt_event_bytes:
            _fail("H2 event CSV bytes differ from sealed-trace/outer-gold replay")
        if observed_metrics_bytes != rebuilt_metrics_bytes:
            _fail("H2 metric bytes differ from sealed-trace/outer-gold replay")
        if (
            event_binding["row_count"] != len(rebuilt_events)
            or event_binding["ordered_columns"] != list(EVENT_COLUMNS)
            or manifest["legacy_voiced_diagnostic"] != _legacy_crosscheck(event_ids)
        ):
            _fail("H2 evaluation manifest differs from canonical rebuilt outputs")
        result["evaluation"] = "passed"
    result["analysis_order"] = freeze["analysis_order"]
    return result


__all__ = [
    "CalibrationSession",
    "EOnlyAcousticSession",
    "GoldFreeSessionView",
    "H2MatchedReplayError",
    "MatchedReplayState",
    "apply_speech_detector",
    "audit_h2_matched_replay",
    "calibrate_and_seal_traces",
    "evaluate_h2_matched_replay",
    "fit_cross_fitted_speech_detector",
    "freeze_h2_matched_replay",
    "gold_free_probability_view",
    "load_e_only_acoustic_descendant",
    "load_gold_free_probability_view",
    "replay_matched_policy",
    "score_trace_events",
    "select_speech_threshold",
    "validate_exact_event_support",
    "verify_file_binding",
]
