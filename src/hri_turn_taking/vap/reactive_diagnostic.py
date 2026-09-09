"""Post-hoc exact-eight diagnostic for a causal ``voiced`` silence rule.

This module intentionally keeps trigger generation separate from gold-event
evaluation.  The trigger phase reads only the independently reviewed formal
acoustic ``voiced`` flag.  Human-reviewed-Transcript-derived VAD is opened only
after the complete trigger CSV has been sealed.  The result is exploratory,
non-production, non-counted evidence; ``voiced`` is a periodicity flag, not a
complete speech detector, so this is not the formal H2 reactive baseline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data.formal_acoustic_features import (
    ReviewedAcousticArtifact,
    load_reviewed_acoustic_scene,
)
from hri_turn_taking.vap.events import (
    TurnEvent,
    extract_shift_hold_events,
    robot_entry_scores,
)
from hri_turn_taking.vap.labels import VAPLabelConfig
from hri_turn_taking.vap.outer_experiment import (
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_EXPECTED_N_ROWS,
    DEADLINE_PROTOCOL_ID,
    DEADLINE_SESSION_TO_GROUP,
    validate_outer_run_manifest,
    validate_outer_split_manifest,
)

FRAME_RATE_HZ = 25
SILENCE_SECONDS = 1.5
REQUIRED_SILENT_FRAMES = 38
ROBOT_SLOT = 0
EXPECTED_N_OPPORTUNITIES = 303
EXPECTED_N_POSITIVE = 111
EXPECTED_N_NEGATIVE = 192

TRIGGER_FILENAME = "trigger_frames.csv"
TRIGGER_MANIFEST_FILENAME = "trigger_manifest.json"
EVENT_FILENAME = "robot_entry_opportunities.csv"
METRICS_FILENAME = "metrics.json"
EVALUATION_MANIFEST_FILENAME = "evaluation_manifest.json"

TRIGGER_COLUMNS = (
    "session_id",
    "group_id",
    "outer_fold_id",
    "tick_25hz",
    "timestamp_s",
    "observed_any_speech",
    "silent_run_frames",
    "reactive_ready",
    "trigger",
)
EVENT_COLUMNS = (
    "event_id",
    "session_id",
    "group_id",
    "outer_fold_id",
    "event_index",
    "gap_start_tick",
    "gap_end_tick_exclusive",
    "gap_frames",
    "previous_slot",
    "next_slot",
    "robot_entry_label",
    "triggered",
    "first_trigger_tick",
    "latency_s",
    "lead_vs_demonstrator_s",
    "silent_run_frames_at_gap_start",
    "missed_positive",
    "false_entry_negative",
)


class ReactiveDiagnosticError(ValueError):
    """A causal-input, protocol, artifact, or evaluation gate failed."""


@dataclass(frozen=True)
class ReactiveReplayState:
    """State carried between adjacent chunks of one session."""

    silent_run_frames: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.silent_run_frames, bool)
            or not isinstance(self.silent_run_frames, int)
            or self.silent_run_frames < 0
        ):
            raise ReactiveDiagnosticError(
                "silent_run_frames must be a non-negative integer"
            )


@dataclass(frozen=True)
class RobotEntryOpportunity:
    """One gold-defined evaluation opportunity, never a trigger input."""

    gap_start_f: int
    gap_end_f: int
    previous_speaker: int
    next_speaker: int
    label: int


@dataclass(frozen=True)
class FrozenDiagnosticProtocol:
    """Strictly validated exact-eight protocol metadata."""

    directory: Path
    split: dict[str, object]
    run: dict[str, object]
    split_payload_sha256: str
    split_file_sha256: str
    run_payload_sha256: str
    run_file_sha256: str
    session_to_fold: dict[str, str]


def _fail(message: str) -> None:
    raise ReactiveDiagnosticError(message)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReactiveDiagnosticError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReactiveDiagnosticError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")
    return payload


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReactiveDiagnosticError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise ReactiveDiagnosticError(f"cannot stat file {resolved}: {exc}") from exc
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(size_bytes),
    }


def verify_file_binding(binding: object, *, role: str) -> Path:
    """Recheck one exact path/size/SHA-256 binding."""

    if not isinstance(binding, dict) or set(binding) < {"path", "sha256", "size_bytes"}:
        _fail(f"{role} binding is malformed")
    path_value = binding.get("path")
    digest = binding.get("sha256")
    size = binding.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        _fail(f"{role} path must be a non-empty string")
    path = Path(path_value)
    if not path.is_absolute() or path.resolve() != path:
        _fail(f"{role} path must be normalized and absolute")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail(f"{role} sha256 must be lowercase SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        _fail(f"{role} size_bytes must be a non-negative integer")
    try:
        current_size = path.stat().st_size
    except OSError as exc:
        raise ReactiveDiagnosticError(f"cannot stat {role} {path}: {exc}") from exc
    if current_size != size or _sha256_file(path) != digest:
        _fail(f"{role} differs from its frozen byte binding")
    return path


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ReactiveDiagnosticError(f"cannot publish {path}: {exc}") from exc


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame, columns: tuple[str, ...]) -> bytes:
    if tuple(frame.columns) != columns:
        _fail("CSV columns differ from the frozen artifact schema")
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10f",
    ).encode("utf-8")


def _timestamps_to_ticks(values: object, *, role: str) -> np.ndarray:
    try:
        timestamps = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReactiveDiagnosticError(f"{role} timestamps are not numeric") from exc
    if timestamps.ndim != 1 or timestamps.size == 0:
        _fail(f"{role} timestamps must be one non-empty vector")
    if not np.isfinite(timestamps).all():
        _fail(f"{role} timestamps must be finite")
    scaled = timestamps * FRAME_RATE_HZ
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0.0, atol=1e-9):
        _fail(f"{role} timestamps do not map exactly to 25 Hz ticks")
    ticks = rounded.astype(np.int64)
    if ticks.size > 1 and not np.equal(np.diff(ticks), 1).all():
        _fail(f"{role} ticks must be strictly contiguous")
    return ticks


def _session_to_fold(split: dict[str, object]) -> dict[str, str]:
    variants = split.get("analysis_variants")
    if not isinstance(variants, dict) or not isinstance(variants.get("primary"), dict):
        _fail("protocol lacks the primary analysis variant")
    folds = variants["primary"].get("outer_folds")
    if not isinstance(folds, list) or len(folds) != 4:
        _fail("primary protocol must contain exactly four outer folds")
    mapping: dict[str, str] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            _fail("outer fold must be a JSON object")
        fold_id = fold.get("outer_fold_id")
        sessions = fold.get("outer_test_session_ids")
        groups = fold.get("outer_test_group_ids")
        if (
            not isinstance(fold_id, str)
            or not isinstance(sessions, list)
            or not isinstance(groups, list)
            or len(groups) != 1
        ):
            _fail("outer fold identity/test membership is malformed")
        for session in sessions:
            if not isinstance(session, str) or session in mapping:
                _fail("outer-test sessions must be unique strings")
            mapping[session] = fold_id
    if mapping != dict(DEADLINE_SESSION_TO_GROUP) | {}:
        # Values differ (fold ids versus groups); compare key order/set explicitly.
        if tuple(mapping) != tuple(DEADLINE_SESSION_TO_GROUP):
            _fail("outer folds do not cover the exact-eight sessions once")
    return mapping


def load_exact_eight_protocol(path_value: str | Path) -> FrozenDiagnosticProtocol:
    """Strictly load the code-owned exact-eight protocol without opening gold data."""

    directory = Path(path_value).resolve()
    split_path = directory / "split_manifest.json"
    run_path = directory / "run_manifest.json"
    split = _read_json(split_path)
    run = _read_json(run_path)
    try:
        validate_outer_split_manifest(split)
        validate_outer_run_manifest(run, split)
    except ValueError as exc:
        raise ReactiveDiagnosticError(f"outer protocol validation failed: {exc}") from exc
    if split.get("protocol_id") != DEADLINE_PROTOCOL_ID:
        _fail("diagnostic requires the exact-eight deadline protocol")
    if split.get("source_session_to_group") != dict(DEADLINE_SESSION_TO_GROUP):
        _fail("protocol does not bind the exact-eight session/group mapping")
    if split.get("corpus_exclusions") != dict(DEADLINE_CORPUS_EXCLUSIONS):
        _fail("protocol does not bind the two owner-pending exclusions")
    if run.get("condition_ids") != ["A", "H", "A+H"]:
        _fail("protocol does not bind the exact A/H/A+H matrix")
    label_config = run.get("label_config")
    if not isinstance(label_config, dict) or label_config.get("frame_rate_hz") != 25:
        _fail("protocol does not freeze the 25 Hz label grid")
    readiness = run.get("readiness_by_condition")
    if not isinstance(readiness, dict) or not isinstance(readiness.get("A"), dict):
        _fail("protocol lacks audio-condition readiness")
    if readiness["A"].get("n_rows") != DEADLINE_EXPECTED_N_ROWS:
        _fail("protocol does not bind the exact 64,299-row view")
    split_payload_sha256 = _payload_sha256(split)
    if run.get("split_manifest_sha256") != split_payload_sha256:
        _fail("run manifest does not bind the loaded split payload")
    return FrozenDiagnosticProtocol(
        directory=directory,
        split=split,
        run=run,
        split_payload_sha256=split_payload_sha256,
        split_file_sha256=_sha256_file(split_path),
        run_payload_sha256=_payload_sha256(run),
        run_file_sha256=_sha256_file(run_path),
        session_to_fold=_session_to_fold(split),
    )


def replay_voiced(
    voiced: object,
    *,
    state: ReactiveReplayState | None = None,
    required_silent_frames: int = REQUIRED_SILENT_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ReactiveReplayState]:
    """Replay one causal chunk and return run length, readiness, and rising triggers."""

    if (
        isinstance(required_silent_frames, bool)
        or not isinstance(required_silent_frames, int)
        or required_silent_frames <= 0
    ):
        _fail("required_silent_frames must be a positive integer")
    initial = ReactiveReplayState() if state is None else state
    if not isinstance(initial, ReactiveReplayState):
        _fail("state must be ReactiveReplayState or None")
    try:
        values = np.asarray(voiced, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReactiveDiagnosticError("voiced must be numeric") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        _fail("voiced must be one finite vector")
    if not np.isin(values, (0.0, 1.0)).all():
        _fail("voiced must be binary")

    silent_run = np.empty(values.size, dtype=np.int64)
    ready = np.zeros(values.size, dtype=np.int8)
    trigger = np.zeros(values.size, dtype=np.int8)
    run = initial.silent_run_frames
    for index, observed in enumerate(values):
        if observed == 1.0:
            run = 0
        else:
            run += 1
        silent_run[index] = run
        ready[index] = int(run >= required_silent_frames)
        trigger[index] = int(run == required_silent_frames)
    return silent_run, ready, trigger, ReactiveReplayState(run)


def _validate_acoustic_frame(frame: pd.DataFrame, *, scene: str) -> tuple[np.ndarray, np.ndarray]:
    expected = {"session_id", "timestamp_s", "array_energy", "f0_proxy", "voiced"}
    if set(frame.columns) != expected:
        _fail(f"reviewed acoustic frame has unexpected columns for {scene}")
    if frame.empty or frame["session_id"].astype(str).tolist() != [scene] * len(frame):
        _fail(f"reviewed acoustic frame has the wrong session identity for {scene}")
    timestamps = frame["timestamp_s"].to_numpy(dtype=np.float64)
    ticks = _timestamps_to_ticks(timestamps, role=f"reviewed acoustic {scene}")
    voiced = frame["voiced"].to_numpy(dtype=np.float64)
    if not np.isfinite(voiced).all() or not np.isin(voiced, (0.0, 1.0)).all():
        _fail(f"reviewed acoustic voiced values are not finite binary for {scene}")
    return ticks, voiced


def build_trigger_frame(
    acoustic_frames: dict[str, pd.DataFrame],
    *,
    session_to_group: dict[str, str],
    session_to_fold: dict[str, str],
) -> pd.DataFrame:
    """Build all trigger rows from acoustic frames; no gold argument exists."""

    expected_sessions = tuple(DEADLINE_SESSION_TO_GROUP)
    if tuple(acoustic_frames) != expected_sessions:
        _fail("acoustic frames must follow the exact-eight canonical session order")
    if session_to_group != dict(DEADLINE_SESSION_TO_GROUP):
        _fail("trigger build requires the frozen exact-eight group mapping")
    if set(session_to_fold) != set(expected_sessions):
        _fail("trigger build requires one outer fold for every session")

    pieces: list[pd.DataFrame] = []
    for scene in expected_sessions:
        source = acoustic_frames[scene]
        ticks, voiced = _validate_acoustic_frame(source, scene=scene)
        silent_run, ready, trigger, _ = replay_voiced(voiced)
        pieces.append(
            pd.DataFrame(
                {
                    "session_id": scene,
                    "group_id": session_to_group[scene],
                    "outer_fold_id": session_to_fold[scene],
                    "tick_25hz": ticks,
                    "timestamp_s": source["timestamp_s"].to_numpy(dtype=np.float64),
                    "observed_any_speech": voiced.astype(np.int8),
                    "silent_run_frames": silent_run,
                    "reactive_ready": ready,
                    "trigger": trigger,
                },
                columns=TRIGGER_COLUMNS,
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != DEADLINE_EXPECTED_N_ROWS:
        _fail("trigger frame does not contain exactly 64,299 rows")
    return result


def _acoustic_binding(artifact: ReviewedAcousticArtifact) -> dict[str, object]:
    manifest = artifact.feature_manifest
    paths = artifact.pending.paths
    if manifest.get("causality") != {
        "causal": True,
        "lookahead_s": 0.0,
        "window_alignment": "trailing",
    }:
        _fail(f"reviewed acoustic causality differs for {artifact.scene}")
    if manifest.get("supplied_columns") != ["array_energy", "f0_proxy", "voiced"]:
        _fail(f"reviewed acoustic columns differ for {artifact.scene}")
    if manifest.get("review_status") != "accepted":
        _fail(f"reviewed acoustic source is not accepted for {artifact.scene}")
    feature = _file_binding(paths.feature_csv)
    manifest_binding = _file_binding(paths.feature_manifest)
    if manifest.get("feature_csv_sha256") != feature["sha256"]:
        _fail(f"reviewed acoustic manifest does not bind its CSV for {artifact.scene}")
    return {
        "scene": artifact.scene,
        "feature_csv": feature,
        "feature_manifest": manifest_binding,
        "review_receipt": _file_binding(paths.review_receipt),
        "source_files": manifest.get("source_files"),
        "causality": manifest["causality"],
        "input_semantics": (
            "combined_mix_normalized_autocorrelation_voiced_flag_not_speech_vad"
        ),
        "row_count": int(len(artifact.pending.feature_df)),
        "tick_start": int(
            _timestamps_to_ticks(
                artifact.pending.feature_df["timestamp_s"],
                role=f"reviewed acoustic {artifact.scene}",
            )[0]
        ),
        "tick_end_exclusive": int(
            _timestamps_to_ticks(
                artifact.pending.feature_df["timestamp_s"],
                role=f"reviewed acoustic {artifact.scene}",
            )[-1]
            + 1
        ),
    }


def _load_acoustic_inputs(
    protocol: FrozenDiagnosticProtocol,
    *,
    loader: Callable[[str], ReviewedAcousticArtifact] = load_reviewed_acoustic_scene,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: dict[str, pd.DataFrame] = {}
    bindings: list[dict[str, object]] = []
    for scene in DEADLINE_SESSION_TO_GROUP:
        artifact = loader(scene)
        if not isinstance(artifact, ReviewedAcousticArtifact):
            _fail("reviewed acoustic loader returned the wrong artifact type")
        frames[scene] = artifact.pending.feature_df.copy()
        bindings.append(_acoustic_binding(artifact))
    trigger_frame = build_trigger_frame(
        frames,
        session_to_group=dict(DEADLINE_SESSION_TO_GROUP),
        session_to_fold=protocol.session_to_fold,
    )
    return trigger_frame, bindings


def _trigger_manifest(
    protocol: FrozenDiagnosticProtocol,
    trigger_frame: pd.DataFrame,
    acoustic_bindings: list[dict[str, object]],
    *,
    csv_binding: dict[str, object],
) -> dict[str, object]:
    module_path = Path(__file__).resolve()
    sessions = []
    for scene in DEADLINE_SESSION_TO_GROUP:
        rows = trigger_frame.loc[trigger_frame["session_id"] == scene]
        sessions.append(
            {
                "session_id": scene,
                "group_id": DEADLINE_SESSION_TO_GROUP[scene],
                "outer_fold_id": protocol.session_to_fold[scene],
                "row_count": int(len(rows)),
                "tick_start": int(rows["tick_25hz"].iloc[0]),
                "tick_end_exclusive": int(rows["tick_25hz"].iloc[-1] + 1),
                "trigger_count": int(rows["trigger"].sum()),
            }
        )
    return {
        "artifact_kind": "vap_exact_eight_voiced_reactive_triggers",
        "format_version": 1,
        "result_role": "posthoc_exploratory_diagnostic",
        "production_eligible": False,
        "counted_eligible": False,
        "h2_baseline_complete": False,
        "analysis_variant": "posthoc_exact_eight_voiced_reactive",
        "input_semantics": (
            "combined_mix_normalized_autocorrelation_voiced_flag_not_speech_vad"
        ),
        "gold_independence": {
            "trigger_input_columns": ["session_id", "timestamp_s", "voiced"],
            "gold_sources": [],
            "prediction_sources": [],
            "gold_opened_during_trigger_generation": False,
        },
        "protocol": {
            "directory": str(protocol.directory),
            "protocol_id": protocol.split["protocol_id"],
            "split_manifest_payload_sha256": protocol.split_payload_sha256,
            "split_manifest_file_sha256": protocol.split_file_sha256,
            "run_manifest_payload_sha256": protocol.run_payload_sha256,
            "run_manifest_file_sha256": protocol.run_file_sha256,
        },
        "corpus": {
            "included_session_ids": list(DEADLINE_SESSION_TO_GROUP),
            "session_to_group": dict(DEADLINE_SESSION_TO_GROUP),
            "session_to_outer_fold": protocol.session_to_fold,
            "corpus_exclusions": dict(DEADLINE_CORPUS_EXCLUSIONS),
            "row_count": int(len(trigger_frame)),
        },
        "algorithm": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "nominal_silence_seconds": SILENCE_SECONDS,
            "required_silent_frames": REQUIRED_SILENT_FRAMES,
            "state_scope": "whole_session_reset_only_at_session_boundary",
            "ready_semantics": "true_from_38th_consecutive_unvoiced_tick",
            "trigger_semantics": "one_rising_trigger_on_38th_tick_per_unvoiced_run",
            "implementation": _file_binding(module_path),
        },
        "acoustic_sources": acoustic_bindings,
        "trigger_csv": {
            **csv_binding,
            "row_count": int(len(trigger_frame)),
            "ordered_columns": list(TRIGGER_COLUMNS),
            "dtypes": {
                "session_id": "string",
                "group_id": "string",
                "outer_fold_id": "string",
                "tick_25hz": "int64",
                "timestamp_s": "float64",
                "observed_any_speech": "int8_binary",
                "silent_run_frames": "int64",
                "reactive_ready": "int8_binary",
                "trigger": "int8_binary",
            },
        },
        "sessions": sessions,
        "scientific_boundary": {
            "complete_speech_vad": False,
            "formal_h2_baseline": False,
            "robot_ear_audio": False,
            "robot_action_feedback_simulated": False,
        },
    }


def generate_trigger_artifact(
    protocol_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    """Seal all 64,299 gold-independent trigger rows without opening gold data."""

    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory: {output}")
    protocol = load_exact_eight_protocol(protocol_directory)
    trigger_frame, acoustic_bindings = _load_acoustic_inputs(protocol)
    csv_payload = _csv_bytes(trigger_frame, TRIGGER_COLUMNS)

    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / TRIGGER_FILENAME
    _write_bytes_exclusive(csv_path, csv_payload)
    csv_binding = _file_binding(csv_path)
    manifest = _trigger_manifest(
        protocol,
        trigger_frame,
        acoustic_bindings,
        csv_binding=csv_binding,
    )
    manifest_path = output / TRIGGER_MANIFEST_FILENAME
    _write_bytes_exclusive(manifest_path, _json_bytes(manifest))
    return manifest_path


def _validate_trigger_frame(frame: pd.DataFrame, manifest: dict[str, object]) -> None:
    if tuple(frame.columns) != TRIGGER_COLUMNS:
        _fail("trigger CSV columns differ from the sealed schema")
    if len(frame) != DEADLINE_EXPECTED_N_ROWS:
        _fail("trigger CSV does not contain exactly 64,299 rows")
    if frame["session_id"].astype(str).drop_duplicates().tolist() != list(
        DEADLINE_SESSION_TO_GROUP
    ):
        _fail("trigger CSV session order differs from exact-eight")
    for column in ("observed_any_speech", "reactive_ready", "trigger"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            _fail(f"trigger CSV {column} must be finite binary")
    for scene, rows in frame.groupby("session_id", sort=False):
        ticks = rows["tick_25hz"].to_numpy()
        if not np.issubdtype(ticks.dtype, np.integer):
            _fail("trigger tick_25hz must retain an integer dtype")
        if len(ticks) > 1 and not np.equal(np.diff(ticks), 1).all():
            _fail(f"trigger ticks are not contiguous for {scene}")
        timestamp_ticks = _timestamps_to_ticks(
            rows["timestamp_s"].to_numpy(), role=f"sealed trigger {scene}"
        )
        if not np.array_equal(ticks.astype(np.int64), timestamp_ticks):
            _fail(f"trigger timestamp/tick binding differs for {scene}")
        _, expected_ready, expected_trigger, _ = replay_voiced(
            rows["observed_any_speech"].to_numpy()
        )
        if not np.array_equal(rows["reactive_ready"].to_numpy(), expected_ready):
            _fail(f"sealed reactive readiness does not replay for {scene}")
        if not np.array_equal(rows["trigger"].to_numpy(), expected_trigger):
            _fail(f"sealed reactive triggers do not replay for {scene}")
    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict) or corpus.get("row_count") != len(frame):
        _fail("trigger manifest row count differs from CSV")


def _load_sealed_trigger(
    protocol: FrozenDiagnosticProtocol,
    output: Path,
    *,
    replay_acoustic: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = output / TRIGGER_MANIFEST_FILENAME
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_kind") != "vap_exact_eight_voiced_reactive_triggers":
        _fail("trigger manifest has the wrong artifact kind")
    if manifest.get("format_version") != 1:
        _fail("trigger manifest has the wrong format version")
    for key in ("production_eligible", "counted_eligible", "h2_baseline_complete"):
        if manifest.get(key) is not False:
            _fail(f"trigger manifest may not promote {key}")
    protocol_binding = manifest.get("protocol")
    expected_protocol = {
        "directory": str(protocol.directory),
        "protocol_id": protocol.split["protocol_id"],
        "split_manifest_payload_sha256": protocol.split_payload_sha256,
        "split_manifest_file_sha256": protocol.split_file_sha256,
        "run_manifest_payload_sha256": protocol.run_payload_sha256,
        "run_manifest_file_sha256": protocol.run_file_sha256,
    }
    if protocol_binding != expected_protocol:
        _fail("trigger manifest belongs to a different frozen protocol")
    gold = manifest.get("gold_independence")
    if not isinstance(gold, dict) or gold.get("gold_sources") != []:
        _fail("trigger manifest must contain no gold source")
    csv_binding = manifest.get("trigger_csv")
    csv_path = verify_file_binding(csv_binding, role="trigger CSV")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ReactiveDiagnosticError(f"cannot parse trigger CSV: {exc}") from exc
    _validate_trigger_frame(frame, manifest)

    implementation = manifest.get("algorithm")
    if not isinstance(implementation, dict):
        _fail("trigger algorithm binding is missing")
    verify_file_binding(implementation.get("implementation"), role="trigger implementation")
    if replay_acoustic:
        fresh, bindings = _load_acoustic_inputs(protocol)
        if bindings != manifest.get("acoustic_sources"):
            _fail("reviewed acoustic bindings changed after trigger sealing")
        if _csv_bytes(fresh, TRIGGER_COLUMNS) != _csv_bytes(frame, TRIGGER_COLUMNS):
            _fail("trigger CSV differs from current reviewed-acoustic replay")
    return frame, manifest


def extract_robot_entry_opportunities(
    vad: object,
    *,
    present: object,
    valid: object,
    robot_slot: int = ROBOT_SLOT,
    frame_rate_hz: int = FRAME_RATE_HZ,
    min_gap_s: float = 0.2,
    max_gap_s: float = 2.0,
    context_s: float = 0.4,
) -> list[RobotEntryOpportunity]:
    """Reuse the current event extractor with the exact robot-entry filters."""

    activity = np.asarray(vad, dtype=np.float64)
    if activity.ndim != 2 or not np.isfinite(activity).all():
        _fail("vad must be one finite (frames, participants) matrix")
    n_frames, n_participants = activity.shape
    if not 0 <= robot_slot < n_participants:
        _fail("robot_slot is outside the participant axis")
    present_mask = np.asarray(present, dtype=np.float64)
    if present_mask.shape != (n_participants,) or not np.isin(
        present_mask, (0.0, 1.0)
    ).all():
        _fail("present must be one binary participant mask")
    validity = np.asarray(valid)
    if validity.shape != (n_frames,) or not np.isin(validity, (0, 1)).all():
        _fail("valid must be one binary frame mask")
    if present_mask[robot_slot] <= 0:
        return []

    opportunities: list[RobotEntryOpportunity] = []
    events: list[TurnEvent] = extract_shift_hold_events(
        activity,
        frame_rate_hz=frame_rate_hz,
        min_gap_s=min_gap_s,
        max_gap_s=max_gap_s,
        context_s=context_s,
    )
    for event in events:
        if event.previous_speaker == robot_slot:
            continue
        if not bool(validity[event.gap_start_f : event.gap_end_f].all()):
            continue
        opportunities.append(
            RobotEntryOpportunity(
                gap_start_f=event.gap_start_f,
                gap_end_f=event.gap_end_f,
                previous_speaker=event.previous_speaker,
                next_speaker=event.next_speaker,
                label=int(event.next_speaker == robot_slot),
            )
        )
    return opportunities


def _crosscheck_existing_scorer(
    opportunities: list[RobotEntryOpportunity],
    *,
    vad: np.ndarray,
    present: np.ndarray,
    valid: np.ndarray,
    label_config: VAPLabelConfig,
) -> None:
    dummy_probabilities = np.zeros(
        (vad.shape[0], vad.shape[1], len(label_config.bin_edges_s)), dtype=np.float32
    )
    labels, scores = robot_entry_scores(
        dummy_probabilities,
        vad,
        robot_slot=ROBOT_SLOT,
        frame_rate_hz=FRAME_RATE_HZ,
        label_config=label_config,
        present=present,
        max_gap_s=2.0,
        valid=valid,
    )
    expected = [opportunity.label for opportunity in opportunities]
    if labels != expected or len(scores) != len(expected):
        _fail("standalone opportunity contract differs from robot_entry_scores")


def _ticks_from_gold_rows(rows: pd.DataFrame, *, scene: str) -> np.ndarray:
    timestamp_ticks = _timestamps_to_ticks(
        rows["timestamp_s"].to_numpy(dtype=np.float64), role=f"gold evaluation {scene}"
    )
    if "tick_25hz" not in rows.columns:
        return timestamp_ticks
    raw = rows["tick_25hz"].to_numpy()
    if not np.issubdtype(raw.dtype, np.integer):
        _fail(f"gold tick_25hz must retain integer dtype for {scene}")
    ticks = raw.astype(np.int64)
    if not np.array_equal(ticks, timestamp_ticks):
        _fail(f"gold timestamp/tick values differ for {scene}")
    return ticks


def _event_id(scene: str, start_tick: int, end_tick: int, previous: int, next_: int) -> str:
    return f"{scene}:{start_tick}:{end_tick}:P{previous + 1}:P{next_ + 1}"


def _event_rows_for_session(
    scene: str,
    group_id: str,
    fold_id: str,
    ticks: np.ndarray,
    trigger_rows: pd.DataFrame,
    opportunities: list[RobotEntryOpportunity],
) -> list[dict[str, object]]:
    trigger = trigger_rows["trigger"].to_numpy(dtype=np.int8)
    silent_run = trigger_rows["silent_run_frames"].to_numpy(dtype=np.int64)
    records: list[dict[str, object]] = []
    for index, event in enumerate(opportunities):
        start_tick = int(ticks[event.gap_start_f])
        end_tick = int(ticks[event.gap_end_f])
        local = np.flatnonzero(trigger[event.gap_start_f : event.gap_end_f] == 1)
        triggered = local.size > 0
        first_tick: int | None = None
        latency: float | None = None
        lead: float | None = None
        if triggered:
            first_frame = event.gap_start_f + int(local[0])
            first_tick = int(ticks[first_frame])
            latency = float((first_tick - start_tick) / FRAME_RATE_HZ)
            if event.label == 1:
                lead = float((end_tick - first_tick) / FRAME_RATE_HZ)
        records.append(
            {
                "event_id": _event_id(
                    scene,
                    start_tick,
                    end_tick,
                    event.previous_speaker,
                    event.next_speaker,
                ),
                "session_id": scene,
                "group_id": group_id,
                "outer_fold_id": fold_id,
                "event_index": index,
                "gap_start_tick": start_tick,
                "gap_end_tick_exclusive": end_tick,
                "gap_frames": int(event.gap_end_f - event.gap_start_f),
                "previous_slot": f"P{event.previous_speaker + 1}",
                "next_slot": f"P{event.next_speaker + 1}",
                "robot_entry_label": event.label,
                "triggered": int(triggered),
                "first_trigger_tick": first_tick,
                "latency_s": latency,
                "lead_vs_demonstrator_s": lead,
                "silent_run_frames_at_gap_start": int(silent_run[event.gap_start_f]),
                "missed_positive": int(event.label == 1 and not triggered),
                "false_entry_negative": int(event.label == 0 and triggered),
            }
        )
    return records


def _rate(numerator: int, denominator: int, reason: str) -> dict[str, object]:
    if denominator == 0:
        return {"value": None, "na_reason": reason}
    return {"value": float(numerator / denominator), "na_reason": None}


def _distribution(values: list[float], *, empty_reason: str) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "na_reason": empty_reason,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "na_reason": None,
    }


def _event_metrics(frame: pd.DataFrame) -> dict[str, object]:
    labels = frame["robot_entry_label"].to_numpy(dtype=np.int8)
    entered = frame["triggered"].to_numpy(dtype=np.int8)
    tp = int(np.sum((labels == 1) & (entered == 1)))
    fn = int(np.sum((labels == 1) & (entered == 0)))
    fp = int(np.sum((labels == 0) & (entered == 1)))
    tn = int(np.sum((labels == 0) & (entered == 0)))
    recall = _rate(tp, tp + fn, "no positive robot-entry opportunities")
    specificity = _rate(tn, tn + fp, "no negative robot-entry opportunities")
    precision = _rate(tp, tp + fp, "no triggered opportunities")
    f1 = _rate(2 * tp, 2 * tp + fp + fn, "F1 denominator is zero")
    if recall["value"] is None or specificity["value"] is None:
        balanced = {
            "value": None,
            "na_reason": "balanced accuracy requires both classes",
        }
    else:
        balanced = {
            "value": float((recall["value"] + specificity["value"]) / 2.0),
            "na_reason": None,
        }
    positive_hits = frame.loc[
        (frame["robot_entry_label"] == 1) & (frame["triggered"] == 1)
    ]
    all_hits = frame.loc[frame["triggered"] == 1]
    return {
        "n_opportunities": int(len(frame)),
        "n_positive": int(tp + fn),
        "n_negative": int(fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_recall": recall,
        "positive_miss_rate": _rate(fn, tp + fn, "no positive robot-entry opportunities"),
        "negative_false_entry_rate": _rate(
            fp, fp + tn, "no negative robot-entry opportunities"
        ),
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced,
        "positive_hit_latency_s": _distribution(
            positive_hits["latency_s"].dropna().astype(float).tolist(),
            empty_reason="no triggered positive opportunities",
        ),
        "positive_hit_lead_vs_demonstrator_s": _distribution(
            positive_hits["lead_vs_demonstrator_s"].dropna().astype(float).tolist(),
            empty_reason="no triggered positive opportunities",
        ),
        "all_triggered_opportunity_latency_s": _distribution(
            all_hits["latency_s"].dropna().astype(float).tolist(),
            empty_reason="no triggered opportunities",
        ),
    }


def _trigger_audit(
    trigger_frame: pd.DataFrame,
    vad_by_session: dict[str, np.ndarray],
    opportunities_by_session: dict[str, list[RobotEntryOpportunity]],
    *,
    sessions: set[str] | None = None,
) -> dict[str, int]:
    selected = set(DEADLINE_SESSION_TO_GROUP) if sessions is None else sessions
    total = collision = eligible = other_silence = 0
    for scene in DEADLINE_SESSION_TO_GROUP:
        if scene not in selected:
            continue
        rows = trigger_frame.loc[trigger_frame["session_id"] == scene]
        indices = np.flatnonzero(rows["trigger"].to_numpy(dtype=np.int8) == 1)
        vad = vad_by_session[scene]
        events = opportunities_by_session[scene]
        for index in indices:
            total += 1
            if float(vad[index].sum()) > 0:
                collision += 1
            elif any(event.gap_start_f <= index < event.gap_end_f for event in events):
                eligible += 1
            else:
                other_silence += 1
    return {
        "total_rising_triggers": total,
        "triggers_during_gold_speech": collision,
        "triggers_in_eligible_gaps": eligible,
        "triggers_in_other_or_noneligible_gold_silence": other_silence,
        "unmatched_to_eligible_gap": total - eligible,
    }


def _verify_protocol_source_bindings(protocol: FrozenDiagnosticProtocol) -> None:
    bindings = protocol.split.get("source_bindings")
    if not isinstance(bindings, list) or len(bindings) != 16:
        _fail("protocol must bind eight assembled CSV/provenance pairs")
    for index, binding in enumerate(bindings):
        verify_file_binding(binding, role=f"protocol source binding {index}")


def _verify_readiness_binding(
    report_payload: dict[str, object], protocol: FrozenDiagnosticProtocol
) -> None:
    expected = protocol.run["readiness_by_condition"]["A"]
    comparisons = {
        "data_kind": report_payload.get("data_kind"),
        "feature_set": report_payload.get("feature_set"),
        "n_rows": report_payload.get("n_rows"),
        "session_ids": report_payload.get("session_ids"),
        "group_ids": report_payload.get("group_ids"),
        "source_artifacts": report_payload.get("source_artifacts"),
        "target_ready": report_payload.get("target_ready"),
        "selected_feature_ready": report_payload.get("selected_feature_ready"),
    }
    expected_comparisons = {key: expected.get(key) for key in comparisons}
    if comparisons != expected_comparisons:
        _fail("fresh evaluation-only source audit differs from frozen A readiness")


def _verify_acoustic_descendant_bindings(
    protocol: FrozenDiagnosticProtocol,
    trigger_manifest: dict[str, object],
) -> list[dict[str, object]]:
    acoustic = trigger_manifest.get("acoustic_sources")
    if not isinstance(acoustic, list) or len(acoustic) != 8:
        _fail("trigger manifest must bind eight reviewed acoustic sources")
    by_scene = {
        str(item.get("scene")): item for item in acoustic if isinstance(item, dict)
    }
    evaluation_sources: list[dict[str, object]] = []
    source_artifacts = protocol.run["readiness_by_condition"]["A"]["source_artifacts"]
    for artifact in source_artifacts:
        scene = Path(artifact["data_path"]).stem
        provenance_path = verify_file_binding(
            {
                "path": artifact["provenance_path"],
                "sha256": artifact["provenance_sha256"],
                "size_bytes": Path(artifact["provenance_path"]).stat().st_size,
            },
            role=f"assembled provenance {scene}",
        )
        provenance = _read_json(provenance_path)
        feature_sources = provenance.get("feature_sources")
        if not isinstance(feature_sources, list):
            _fail(f"assembled provenance lacks feature sources for {scene}")
        matches = [
            item
            for item in feature_sources
            if isinstance(item, dict)
            and item.get("columns") == ["array_energy", "f0_proxy", "voiced"]
        ]
        if len(matches) != 1 or scene not in by_scene:
            _fail(f"assembled provenance lacks one acoustic descendant for {scene}")
        descendant = matches[0]
        trigger_source = by_scene[scene]
        expected = {
            "path": trigger_source["feature_csv"]["path"],
            "sha256": trigger_source["feature_csv"]["sha256"],
            "manifest_path": trigger_source["feature_manifest"]["path"],
            "manifest_sha256": trigger_source["feature_manifest"]["sha256"],
            "causality": trigger_source["causality"],
        }
        observed = {key: descendant.get(key) for key in expected}
        if observed != expected:
            _fail(f"assembled acoustic descendant differs from trigger source for {scene}")
        evaluation_sources.append(
            {
                "scene": scene,
                "assembled_csv": {
                    "path": artifact["data_path"],
                    "sha256": artifact["data_sha256"],
                },
                "assembled_provenance": {
                    "path": artifact["provenance_path"],
                    "sha256": artifact["provenance_sha256"],
                },
                "target_role": "evaluation_only_human_reviewed_Transcript_derived_VAD",
            }
        )
    return evaluation_sources


def _label_config(run: dict[str, object]) -> VAPLabelConfig:
    payload = run.get("label_config")
    if not isinstance(payload, dict):
        _fail("run manifest label_config is missing")
    try:
        return VAPLabelConfig(**payload)
    except (TypeError, ValueError) as exc:
        raise ReactiveDiagnosticError(f"invalid run label_config: {exc}") from exc


def _build_evaluation(
    protocol: FrozenDiagnosticProtocol,
    trigger_frame: pd.DataFrame,
    trigger_manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    # Gold-capable imports and source reads are deliberately delayed until the
    # trigger artifact has passed sealed acoustic replay above.
    from hri_turn_taking.data.assembly import load_and_audit_vap_source
    from hri_turn_taking.vap.dataset import sessions_from_dataframe

    _verify_protocol_source_bindings(protocol)
    source = Path(str(protocol.run["source"])).resolve()
    frame, report = load_and_audit_vap_source(
        source,
        data_kind="real",
        feature_set="audio_only",
        excluded_session_ids=tuple(DEADLINE_CORPUS_EXCLUSIONS),
    )
    report_payload = report.to_dict()
    _verify_readiness_binding(report_payload, protocol)
    evaluation_sources = _verify_acoustic_descendant_bindings(protocol, trigger_manifest)
    config = _label_config(protocol.run)
    sessions = sessions_from_dataframe(frame, label_config=config)
    by_session = {session.session_id: session for session in sessions}
    if tuple(sorted(by_session)) != tuple(DEADLINE_SESSION_TO_GROUP):
        _fail("evaluation source does not contain the exact-eight sessions")

    event_records: list[dict[str, object]] = []
    vad_by_session: dict[str, np.ndarray] = {}
    opportunities_by_session: dict[str, list[RobotEntryOpportunity]] = {}
    for scene in DEADLINE_SESSION_TO_GROUP:
        session = by_session[scene]
        rows = frame.loc[frame["session_id"].astype(str) == scene].sort_values(
            "timestamp_s"
        )
        ticks = _ticks_from_gold_rows(rows, scene=scene)
        trigger_rows = trigger_frame.loc[trigger_frame["session_id"] == scene]
        trigger_ticks = trigger_rows["tick_25hz"].to_numpy(dtype=np.int64)
        if not np.array_equal(ticks, trigger_ticks):
            _fail(f"acoustic trigger and evaluation-only gold keys differ for {scene}")
        opportunities = extract_robot_entry_opportunities(
            session.vad,
            present=session.present,
            valid=session.valid,
        )
        _crosscheck_existing_scorer(
            opportunities,
            vad=session.vad,
            present=session.present,
            valid=session.valid,
            label_config=config,
        )
        event_records.extend(
            _event_rows_for_session(
                scene,
                DEADLINE_SESSION_TO_GROUP[scene],
                protocol.session_to_fold[scene],
                ticks,
                trigger_rows,
                opportunities,
            )
        )
        vad_by_session[scene] = session.vad
        opportunities_by_session[scene] = opportunities

    events = pd.DataFrame(event_records, columns=EVENT_COLUMNS)
    n_positive = int(events["robot_entry_label"].sum())
    n_negative = int(len(events) - n_positive)
    if (
        len(events) != EXPECTED_N_OPPORTUNITIES
        or n_positive != EXPECTED_N_POSITIVE
        or n_negative != EXPECTED_N_NEGATIVE
    ):
        _fail("robot-entry support differs from the frozen 303/111/192 contract")
    if events["event_id"].duplicated().any():
        _fail("robot-entry event IDs must be globally unique")

    folds: dict[str, object] = {}
    primary_folds = protocol.split["analysis_variants"]["primary"]["outer_folds"]
    for fold in primary_folds:
        fold_id = fold["outer_fold_id"]
        fold_events = events.loc[events["outer_fold_id"] == fold_id]
        fold_sessions = set(fold["outer_test_session_ids"])
        folds[fold_id] = {
            "outer_test_group_ids": fold["outer_test_group_ids"],
            "outer_test_session_ids": fold["outer_test_session_ids"],
            "event_metrics": _event_metrics(fold_events),
            "global_trigger_audit": _trigger_audit(
                trigger_frame,
                vad_by_session,
                opportunities_by_session,
                sessions=fold_sessions,
            ),
        }
    metrics = {
        "artifact_kind": "vap_exact_eight_voiced_reactive_metrics",
        "format_version": 1,
        "result_role": "posthoc_exploratory_diagnostic",
        "production_eligible": False,
        "counted_eligible": False,
        "h2_baseline_complete": False,
        "analysis_variant": "posthoc_exact_eight_voiced_reactive",
        "pooled": {
            "event_metrics": _event_metrics(events),
            "global_trigger_audit": _trigger_audit(
                trigger_frame, vad_by_session, opportunities_by_session
            ),
        },
        "outer_folds": folds,
        "opportunity_contract": {
            "frame_rate_hz": 25,
            "min_gap_s": 0.2,
            "max_gap_s": 2.0,
            "context_s": 0.4,
            "robot_slot": "P1",
            "previous_speaker_must_not_be_robot": True,
            "full_future_horizon_required_over_gap": True,
            "support": {"total": 303, "positive": 111, "negative": 192},
            "existing_robot_entry_scorer_crosscheck": "passed",
        },
        "metric_boundary": {
            "average_precision_reported": False,
            "reason": "fixed binary trigger is not a VAP probability ranking",
            "direct_h2_latency_claim_allowed": False,
        },
    }
    return events, metrics, evaluation_sources


def evaluate_trigger_artifact(
    protocol_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    """Evaluate a previously sealed trigger artifact against evaluation-only gold."""

    output = Path(output_directory).resolve()
    if not output.is_dir():
        _fail("diagnostic directory must already contain a sealed trigger artifact")
    targets = (
        output / EVENT_FILENAME,
        output / METRICS_FILENAME,
        output / EVALUATION_MANIFEST_FILENAME,
    )
    if any(path.exists() for path in targets):
        raise FileExistsError(f"refusing to overwrite evaluation in {output}")
    protocol = load_exact_eight_protocol(protocol_directory)
    trigger_frame, trigger_manifest = _load_sealed_trigger(
        protocol, output, replay_acoustic=True
    )
    events, metrics, evaluation_sources = _build_evaluation(
        protocol, trigger_frame, trigger_manifest
    )

    event_payload = _csv_bytes(events, EVENT_COLUMNS)
    metrics_payload = _json_bytes(metrics)
    event_path, metrics_path, manifest_path = targets
    _write_bytes_exclusive(event_path, event_payload)
    _write_bytes_exclusive(metrics_path, metrics_payload)
    evaluation_manifest = {
        "artifact_kind": "vap_exact_eight_voiced_reactive_evaluation",
        "format_version": 1,
        "result_role": "posthoc_exploratory_diagnostic",
        "production_eligible": False,
        "counted_eligible": False,
        "h2_baseline_complete": False,
        "analysis_variant": "posthoc_exact_eight_voiced_reactive",
        "trigger_manifest": _file_binding(output / TRIGGER_MANIFEST_FILENAME),
        "trigger_csv": _file_binding(output / TRIGGER_FILENAME),
        "evaluation_only_gold_sources": evaluation_sources,
        "protocol_source_bindings": protocol.split["source_bindings"],
        "event_csv": {
            **_file_binding(event_path),
            "row_count": int(len(events)),
            "ordered_columns": list(EVENT_COLUMNS),
        },
        "metrics": _file_binding(metrics_path),
        "support": {
            "n_opportunities": int(len(events)),
            "n_positive": int(events["robot_entry_label"].sum()),
            "n_negative": int(len(events) - events["robot_entry_label"].sum()),
        },
        "causal_and_gold_boundary": {
            "trigger_artifact_sealed_before_gold_load": True,
            "trigger_replayed_from_reviewed_acoustic_before_gold_load": True,
            "gold_used_only_for_opportunity_labels_and_posthoc_trigger_classification": True,
            "voiced_is_complete_speech_vad": False,
            "formal_h2_baseline": False,
            "live_robot_or_tts_latency_simulated": False,
        },
        "comparability": {
            "same_exact_eight_sessions": True,
            "same_four_outer_groups": True,
            "same_303_robot_entry_opportunities": True,
            "comparable_to_vap_ranking_ap": False,
            "paired_policy_latency_comparison_available": False,
        },
    }
    _write_bytes_exclusive(manifest_path, _json_bytes(evaluation_manifest))
    return manifest_path


def run_reactive_diagnostic(
    protocol_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run trigger sealing first, then the separately gated gold evaluation."""

    generate_trigger_artifact(protocol_directory, output_directory)
    return evaluate_trigger_artifact(protocol_directory, output_directory)


__all__ = [
    "EVENT_COLUMNS",
    "EXPECTED_N_NEGATIVE",
    "EXPECTED_N_OPPORTUNITIES",
    "EXPECTED_N_POSITIVE",
    "FRAME_RATE_HZ",
    "REQUIRED_SILENT_FRAMES",
    "ReactiveDiagnosticError",
    "ReactiveReplayState",
    "RobotEntryOpportunity",
    "TRIGGER_COLUMNS",
    "build_trigger_frame",
    "evaluate_trigger_artifact",
    "extract_robot_entry_opportunities",
    "generate_trigger_artifact",
    "load_exact_eight_protocol",
    "replay_voiced",
    "run_reactive_diagnostic",
    "verify_file_binding",
]
