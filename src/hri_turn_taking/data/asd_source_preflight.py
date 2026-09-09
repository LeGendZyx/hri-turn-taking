"""Read-only source and causality preflight for a future real-corpus ASD stage.

This module deliberately does not import, download, or execute an ASD model.  It
only replays the sealed corpus authorities and constructs an exact past-only
camera/audio plan.  Its reports can never be used as assembly feature manifests.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import analysis_base, assembly, audio_contract
from hri_turn_taking.data import hpe_transcript_crosswalk as crosswalk_module
from hri_turn_taking.data.source_paths import corpus_root

ARTIFACT_KIND = "vap_real_asd_source_preflight"
FORMAT_VERSION = 1
FRAME_RATE_HZ = 25
PARTICIPANT_TO_SLOT = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
FIRST_SEGMENT_ONLY = frozenset({"05_14__3", "05_14__4"})
LIGHT_ASD_COMMIT = "ed38c232de5efe0261dbd68627c0ade7cdfe14eb"
LR_ASD_COMMIT = "1b6dcd2d8fc2895683de6508ec6294ec47d388ca"

_SCENE_RE = re.compile(r"^[0-9]{2}_[0-9]{2}__[0-9]+$")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_BINDING_KEYS = {"path", "size_bytes", "sha256"}


class ASDSourcePreflightError(ValueError):
    """A source, identity, timebase, or non-leakage gate failed closed."""


@dataclass(frozen=True, kw_only=True)
class ASDSourceAuthority:
    """Code-owned canonical roots; callers cannot redirect individual files."""

    repository_root: Path
    corpus_root: Path
    analysis_base_root: Path
    crosswalk_inventory: Path
    crosswalk_root: Path
    audio_contract_root: Path
    enforce_frozen_inventory: bool = True

    def __post_init__(self) -> None:
        for field in (
            "repository_root",
            "corpus_root",
            "analysis_base_root",
            "crosswalk_inventory",
            "crosswalk_root",
            "audio_contract_root",
        ):
            value = getattr(self, field)
            if not isinstance(value, Path):
                raise ASDSourcePreflightError(f"{field} must be a pathlib.Path")
            object.__setattr__(self, field, _lexical_absolute(value))
        if type(self.enforce_frozen_inventory) is not bool:
            raise ASDSourcePreflightError("enforce_frozen_inventory must be a boolean")


@dataclass(frozen=True, kw_only=True)
class ASDSourceDocuments:
    """Already loaded, path-bound documents used by the pure contract builder."""

    analysis: assembly._DatasetFile
    crosswalk: dict[str, Any]
    crosswalk_manifest: dict[str, Any]
    validation_manifest: dict[str, Any]
    identity_manifest: dict[str, Any]
    audio_report: dict[str, Any]
    bindings: dict[str, dict[str, Any]]
    cam6_relabeled: pd.DataFrame | None = None
    cam6_run_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.analysis) is not assembly._DatasetFile:
            raise ASDSourcePreflightError("analysis must be an exact gated dataset file")
        for field in (
            "crosswalk",
            "crosswalk_manifest",
            "validation_manifest",
            "identity_manifest",
            "audio_report",
            "bindings",
        ):
            if type(getattr(self, field)) is not dict:
                raise ASDSourcePreflightError(f"{field} must be an exact dict")
        if self.cam6_relabeled is not None and type(self.cam6_relabeled) is not pd.DataFrame:
            raise ASDSourcePreflightError("cam6_relabeled must be a DataFrame or None")
        if self.cam6_run_metadata is not None and type(self.cam6_run_metadata) is not dict:
            raise ASDSourcePreflightError("cam6_run_metadata must be a dict or None")


def default_source_authority() -> ASDSourceAuthority:
    root = Path(__file__).resolve().parents[3]
    return ASDSourceAuthority(
        repository_root=root,
        corpus_root=corpus_root(root),
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        crosswalk_inventory=root / "configs" / "hpe_transcript_crosswalk_inventory.json",
        crosswalk_root=root / "data" / "annotations" / "hpe_transcript_crosswalk",
        audio_contract_root=root / "data" / "annotations" / "audio_contract",
    )


def _fail(message: str) -> None:
    raise ASDSourcePreflightError(message)


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat(path: Path) -> os.stat_result:
    return path.lstat()


def _secure_existing_path(path: Path, *, root: Path, label: str) -> Path:
    """Reject links/reparse points lexically before any resolving operation."""

    lexical_root = _lexical_absolute(root)
    lexical_path = _lexical_absolute(path)
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ASDSourcePreflightError(f"{label} escapes its authority root") from exc
    current = lexical_root
    chain = [current]
    for part in relative.parts:
        current = current / part
        chain.append(current)
    for index, candidate in enumerate(chain):
        try:
            info = _lstat(candidate)
        except OSError as exc:
            raise ASDSourcePreflightError(
                f"{label} path component is unavailable: {candidate}"
            ) from exc
        attributes = int(getattr(info, "st_file_attributes", 0))
        if stat.S_ISLNK(info.st_mode) or attributes & _REPARSE_POINT:
            _fail(f"{label} path contains a symlink/junction/reparse point: {candidate}")
        leaf = index == len(chain) - 1
        if leaf and not stat.S_ISREG(info.st_mode):
            _fail(f"{label} leaf must be a regular file")
        if not leaf and not stat.S_ISDIR(info.st_mode):
            _fail(f"{label} ancestor is not a directory: {candidate}")
    resolved_root = lexical_root.resolve(strict=True)
    resolved = lexical_path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ASDSourcePreflightError(
            f"{label} resolved outside its authority root"
        ) from exc
    return resolved


def _file_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ASDSourcePreflightError(f"{label} cannot be lstat'ed: {exc}") from exc
    attributes = int(getattr(info, "st_file_attributes", 0))
    if stat.S_ISLNK(info.st_mode) or attributes & _REPARSE_POINT or not stat.S_ISREG(
        info.st_mode
    ):
        _fail(f"{label} must be a regular non-symlink non-reparse file")
    return (int(info.st_dev), int(info.st_ino), int(info.st_size),
            int(info.st_mtime_ns), attributes)


def _stable_read(
    path: Path, label: str, *, root: Path | None = None
) -> tuple[bytes, dict[str, Any]]:
    lexical = _lexical_absolute(path)
    secured = _secure_existing_path(
        lexical, root=lexical.parent if root is None else root, label=label
    )
    before = _file_identity(secured, label)
    try:
        raw = secured.read_bytes()
    except OSError as exc:
        raise ASDSourcePreflightError(f"{label} is unreadable: {exc}") from exc
    after = _file_identity(secured, label)
    if before != after or len(raw) != before[2]:
        _fail(f"{label} changed during the stable read")
    return raw, {
        "path": str(secured),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ASDSourcePreflightError(f"{label} is not UTF-8") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(
                f"{label} contains forbidden JSON constant {token}"
            ),
        )
    except json.JSONDecodeError as exc:
        raise ASDSourcePreflightError(f"{label} is invalid JSON") from exc
    if type(value) is not dict:
        _fail(f"{label} must contain one JSON object")
    return value


def _strict_json_bound(
    path: Path, label: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw, binding = _stable_read(path, label)
    return _strict_json_bytes(raw, label), raw, binding


def _assert_binding(binding: Any, *, path: Path, label: str) -> None:
    if type(binding) is not dict or set(binding) != _BINDING_KEYS:
        _fail(f"{label} binding must contain exactly {sorted(_BINDING_KEYS)}")
    _exact_path(binding["path"], path, f"{label} binding path")
    if (
        type(binding["size_bytes"]) is not int
        or binding["size_bytes"] < 0
        or type(binding["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is None
    ):
        _fail(f"{label} binding size/hash is malformed")


def _exact_path(actual: Any, expected: Path, label: str) -> None:
    if type(actual) is not str or Path(actual).resolve() != expected.resolve():
        _fail(f"{label} is not the code-owned canonical path")


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact integer (boolean/float rejected)")
    return value


def _finite_fraction(value: Any, label: str) -> Fraction:
    if type(value) not in {str, int, float} or type(value) is bool:
        _fail(f"{label} must be a finite decimal scalar")
    if type(value) is float and not math.isfinite(value):
        _fail(f"{label} must be finite")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ASDSourcePreflightError(f"{label} must be a finite decimal scalar") from exc
    return result


def _exact_rate(value: Any, label: str) -> int:
    if type(value) not in {int, float} or type(value) is bool:
        _fail(f"{label} must be an integer-valued finite number")
    if not math.isfinite(float(value)) or not float(value).is_integer():
        _fail(f"{label} must be an integer-valued finite number")
    result = int(value)
    if result not in {30, 60}:
        _fail(f"{label} must be 30 or 60")
    return result


def camera_plan(scene: str) -> dict[str, str]:
    """Return the frozen participant-to-camera plan for exactly ten scenes."""

    if type(scene) is not str or _SCENE_RE.fullmatch(scene) is None:
        _fail("scene must use the canonical day__scenario spelling")
    if scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail("scene is outside the frozen ten-scene dissertation inventory")
    participant_camera = "Cam_1" if scene.startswith("05_15__") else "Cam_2"
    return {"R": "Cam_6", "A": participant_camera, "B": participant_camera,
            "C": participant_camera}


def past_only_frame_indices(
    ticks: list[int] | tuple[int, ...] | np.ndarray,
    *,
    source_fps: int,
    d3_offset_s: str | int | float,
    accepted_start: int,
    accepted_end: int,
) -> np.ndarray:
    """Map 25-Hz shared ticks to the latest source frame at or before each tick.

    All arithmetic is rational.  There is no nearest-frame rounding,
    interpolation, or tolerance join.
    """

    if type(source_fps) is not int or source_fps not in {30, 60}:
        _fail("source_fps must be the exact integer 30 or 60")
    start = _exact_int(accepted_start, "accepted_start")
    end = _exact_int(accepted_end, "accepted_end")
    if start < 0 or end <= start:
        _fail("accepted source-frame window is invalid")
    if type(ticks) not in {list, tuple, np.ndarray}:
        _fail("ticks must be a list, tuple, or ndarray")
    values = list(ticks)
    if not values:
        _fail("ticks must be non-empty")
    if any(type(value) not in {int, np.int64} for value in values):
        _fail("every tick must be an exact integer (boolean/float rejected)")
    integer_ticks = [int(value) for value in values]
    if any(value < 0 for value in integer_ticks):
        _fail("ticks must be non-negative")
    if any(
        right <= left
        for left, right in zip(integer_ticks, integer_ticks[1:], strict=False)
    ):
        _fail("ticks must be strictly increasing and unique")
    offset = _finite_fraction(d3_offset_s, "d3_offset_s")
    frames = np.asarray(
        [((Fraction(tick, FRAME_RATE_HZ) - offset) * source_fps).__floor__()
         for tick in integer_ticks],
        dtype=np.int64,
    )
    if np.any(frames < start) or np.any(frames >= end):
        _fail("past-only camera mapping escapes the accepted half-open source window")
    for tick, frame in zip(integer_ticks, frames, strict=True):
        frame_time = Fraction(int(frame), source_fps) + offset
        next_time = Fraction(int(frame) + 1, source_fps) + offset
        target = Fraction(tick, FRAME_RATE_HZ)
        if frame_time > target or next_time <= target:
            _fail("past-only rational source-frame invariant failed")
    return frames


def _require_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or not keys.issubset(value):
        _fail(f"{label} lacks required fields {sorted(keys)}")
    return value


def _media_probe(path: Path) -> dict[str, Any]:
    """Stat a path without opening its data stream (no OneDrive hydration)."""

    try:
        info = path.lstat()
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "path_gate_pass": False,
            "bytes_status": "unavailable",
            "ready_for_extraction": False,
            "file_opened_or_hydrated": False,
            "error": type(exc).__name__,
        }
    attributes = int(getattr(info, "st_file_attributes", 0))
    if stat.S_ISLNK(info.st_mode) or attributes & _REPARSE_POINT or not stat.S_ISREG(
        info.st_mode
    ):
        kind = (
            "symlink"
            if stat.S_ISLNK(info.st_mode)
            else "reparse_point"
            if attributes & _REPARSE_POINT
            else "non_regular"
        )
        return {
            "path": str(path),
            "exists": True,
            "path_gate_pass": False,
            "path_kind": kind,
            "declared_size_bytes": int(info.st_size),
            "bytes_status": "rejected_without_following_or_opening",
            "ready_for_extraction": False,
            "file_opened_or_hydrated": False,
        }
    return {
        "path": str(path),
        "exists": True,
        "path_gate_pass": True,
        "path_kind": "regular_file",
        "declared_size_bytes": int(info.st_size),
        "bytes_status": "bytes_unverified_no_hash_read",
        "ready_for_extraction": False,
        "file_opened_or_hydrated": False,
    }


def _validate_probe(value: Any, *, expected_path: Path, label: str) -> dict[str, Any]:
    common = {
        "path",
        "exists",
        "path_gate_pass",
        "bytes_status",
        "ready_for_extraction",
        "file_opened_or_hydrated",
    }
    if type(value) is not dict or not common.issubset(value):
        _fail(f"{label} media probe is malformed")
    _exact_path(value["path"], expected_path, f"{label} media probe")
    for key in (
        "exists",
        "path_gate_pass",
        "ready_for_extraction",
        "file_opened_or_hydrated",
    ):
        if type(value[key]) is not bool:
            _fail(f"{label} media probe {key} must be a boolean")
    if type(value["bytes_status"]) is not str:
        _fail(f"{label} media probe bytes_status must be a string")
    if value["ready_for_extraction"] or value["file_opened_or_hydrated"]:
        _fail(f"{label} preflight probe cannot claim ready/opened media")
    status = value["bytes_status"]
    if status == "unavailable":
        expected_keys = common | {"error"}
        valid_state = (
            value["exists"] is False
            and value["path_gate_pass"] is False
            and type(value["error"]) is str
        )
    elif status == "rejected_without_following_or_opening":
        expected_keys = common | {"path_kind", "declared_size_bytes"}
        valid_state = (
            value["exists"] is True
            and value["path_gate_pass"] is False
            and value["path_kind"] in {"symlink", "reparse_point", "non_regular"}
            and type(value["declared_size_bytes"]) is int
        )
    elif status == "bytes_unverified_no_hash_read":
        expected_keys = common | {"path_kind", "declared_size_bytes"}
        valid_state = (
            value["exists"] is True
            and value["path_gate_pass"] is True
            and value["path_kind"] == "regular_file"
            and type(value["declared_size_bytes"]) is int
        )
    else:
        _fail(f"{label} media probe reports a forbidden byte state")
    if set(value) != expected_keys or not valid_state:
        _fail(f"{label} media probe state machine is inconsistent")
    return value


def _replace_camera(path: Path, source_camera: str, target_camera: str) -> Path:
    parts = list(path.parts)
    matches = [index for index, part in enumerate(parts) if part == source_camera]
    if len(matches) != 1:
        _fail("declared visual path does not contain exactly one camera directory")
    parts[matches[0]] = target_camera
    name = parts[-1]
    prefix = f"{source_camera}_"
    if not name.startswith(prefix):
        _fail("declared visual basename is not camera-prefixed")
    parts[-1] = f"{target_camera}_{name[len(prefix):]}"
    return Path(*parts)


def _identity_video_path(
    declared: dict[str, Any], *, scene: str, camera: str, authority: ASDSourceAuthority
) -> Path:
    day, scenario = scene.rsplit("__", maxsplit=1)
    value = declared.get("source_path")
    if type(value) is not str:
        _fail("identity source_path must be a string")
    relative = PurePosixPath(value)
    expected_parent = "03_26_6cam" if day == "03_26" else "6cams"
    if (
        len(relative.parts) != 6
        or relative.parts[:5] != ("Data", day, expected_parent, scenario, camera)
        or ".." in relative.parts
        or not relative.name.startswith(f"{camera}_")
        or relative.suffix.lower() != ".avi"
    ):
        _fail("identity source_path differs from the exact scene/camera AVI pattern")
    return _lexical_absolute(authority.corpus_root.joinpath(*relative.parts[1:]))


def _association_coverage(
    selected_frames: np.ndarray,
    *,
    relabeled: pd.DataFrame | None,
    hpe_person: int,
    binding_available: bool,
) -> dict[str, Any]:
    total = int(selected_frames.size)
    base = {
        "denominator_analysis_ticks": total,
        "valid": 0,
        "missing": total,
        "ambiguous": 0,
        "valid_fraction": 0.0,
        "missing_is_zero_filled": False,
        "r_source_valid": False,
    }
    if not binding_available or relabeled is None:
        return {"status": "blocked_no_sealed_cam6_binding", **base}
    required = {
        "frame_index",
        "person",
        "face_matched",
        "face_bbox_x1",
        "face_bbox_y1",
        "face_bbox_x2",
        "face_bbox_y2",
        "face_association_reason",
    }
    if not required.issubset(relabeled.columns):
        return {
            "status": "blocked_missing_face_bbox_or_association_schema",
            "missing_schema_columns": sorted(required - set(relabeled.columns)),
            **base,
        }
    selected_set = set(int(value) for value in selected_frames)
    rows = relabeled.loc[
        relabeled["frame_index"].isin(selected_set) & (relabeled["person"] == hpe_person)
    ]
    grouped = {int(frame): group for frame, group in rows.groupby("frame_index")}
    valid = 0
    missing = 0
    ambiguous = 0
    for frame in selected_frames:
        candidates = grouped.get(int(frame))
        if candidates is None or candidates.empty:
            missing += 1
            continue
        if len(candidates) != 1:
            ambiguous += 1
            continue
        row = candidates.iloc[0]
        reason = row["face_association_reason"]
        values = np.asarray(
            [row["face_bbox_x1"], row["face_bbox_y1"],
             row["face_bbox_x2"], row["face_bbox_y2"]],
            dtype=np.float64,
        )
        reason_text = "" if pd.isna(reason) else str(reason).strip().lower()
        reason_ambiguous = any(
            marker in reason_text for marker in ("ambiguous", "conflict", "multiple")
        )
        if reason_ambiguous:
            ambiguous += 1
        elif (
            row["face_matched"] == 1
            and np.all(np.isfinite(values))
            and values[2] > values[0]
            and values[3] > values[1]
            and reason_text
        ):
            valid += 1
        else:
            missing += 1
    if valid + missing + ambiguous != total:
        _fail("R face/head association coverage does not partition analysis ticks")
    return {
        "status": "ready" if valid == total else "blocked_incomplete_or_ambiguous",
        "denominator_analysis_ticks": total,
        "valid": valid,
        "missing": missing,
        "ambiguous": ambiguous,
        "valid_fraction": valid / total,
        "missing_is_zero_filled": False,
        "r_source_valid": valid == total,
    }


def _validate_audio(
    scene: str,
    report: dict[str, Any],
    *,
    manifest_path: Path,
    authority: ASDSourceAuthority,
) -> dict[str, Any]:
    if report.get("artifact_kind") != audio_contract.ARTIFACT_KIND:
        _fail("ASD audio must come from a canonical audio source contract")
    if report.get("format_version") != audio_contract.FORMAT_VERSION:
        _fail("audio contract format version changed")
    if report.get("scene") != scene:
        _fail("audio contract scene differs from ASD scene")
    if report.get("output_role") != audio_contract.OUTPUT_ROLE:
        _fail("audio contract output role changed")
    if report.get("counted_gold_ready") is not False:
        _fail("audio source contract cannot claim counted gold readiness")
    expected_manifest = authority.audio_contract_root / f"{scene}.audio_contract.json"
    if manifest_path.resolve() != expected_manifest.resolve():
        _fail("audio contract was copied away from its code-owned canonical path")
    sources = _require_keys(report.get("sources"), {"A", "B", "C", "R", "master"},
                            "audio sources")
    expected_paths = audio_contract.canonical_source_paths(
        scene,
        audio_contract.AudioSourceAuthority(
            corpus_root=authority.corpus_root,
            hpe_delivery_root=authority.repository_root.parent / "Head_pose_extraction"
            / "outputs" / "delivery",
        ),
    )
    master = _require_keys(
        sources["master"],
        {"path", "sha256", "size_bytes", "wave_header", "pcm_payload"},
        "audio master",
    )
    _exact_path(master["path"], expected_paths["master"], "audio master")
    if (
        type(master["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", master["sha256"]) is None
        or type(master["size_bytes"]) is not int
        or master["size_bytes"] <= 44
    ):
        _fail("audio master byte binding is malformed")
    header = _require_keys(
        master["wave_header"],
        {
            "channels",
            "sample_width_bytes",
            "sample_rate_hz",
            "compression_type",
            "n_frames",
        },
        "audio master wave header",
    )
    if (
        type(header["channels"]) is not int
        or header["channels"] != 4
        or type(header["sample_width_bytes"]) is not int
        or header["sample_width_bytes"] != 2
        or type(header["sample_rate_hz"]) is not int
        or header["sample_rate_hz"] != 48_000
        or header["compression_type"] != "NONE"
        or type(header["n_frames"]) is not int
        or header["n_frames"] <= 0
    ):
        _fail("audio master must remain four-channel 48-kHz PCM16")
    payload = _require_keys(
        master["pcm_payload"],
        {"complete", "decoded_frames", "frame_aligned", "frame_size_bytes"},
        "audio master PCM payload",
    )
    if (
        payload["complete"] is not True
        or payload["frame_aligned"] is not True
        or type(payload["decoded_frames"]) is not int
        or payload["decoded_frames"] != header["n_frames"]
        or type(payload["frame_size_bytes"]) is not int
        or payload["frame_size_bytes"] != 8
    ):
        _fail("audio master PCM payload is incomplete or inconsistent")
    replay = _require_keys(
        report.get("master_sample_replay"),
        {"channel_index_to_speaker", "exact_permutation", "identity_rule"},
        "audio master sample replay",
    )
    if (
        replay["exact_permutation"] is not True
        or type(replay["channel_index_to_speaker"]) is not list
        or len(replay["channel_index_to_speaker"]) != 4
        or set(replay["channel_index_to_speaker"]) != {"A", "B", "C", "R"}
        or replay["identity_rule"]
        != "sample_replay_only_never_channel_index_or_report"
    ):
        _fail("audio master is not the exact sample-replayed lavalier permutation")
    for speaker in ("A", "B", "C", "R"):
        source = _require_keys(sources[speaker], {"path"}, f"audio {speaker}")
        _exact_path(source["path"], expected_paths[speaker], f"audio {speaker}")
    return {
        "accepted_source": "audio_contract.sources.master",
        "transform": "exact_int64_sum_then_float64_divide_by_4x32768",
        "channel_semantics": "equal_weight_mono_of_four_lavalier_container",
        "single_lavalier_input_forbidden": True,
        "transcript_or_gold_input_forbidden": True,
        "gold_channel_time_corrections_applied": False,
        "master_sha256": master["sha256"],
        "manifest_path": str(manifest_path.resolve()),
    }


def _validate_documents(
    scene: str,
    documents: ASDSourceDocuments,
    authority: ASDSourceAuthority,
    probe: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    expected_analysis = authority.analysis_base_root / f"{scene}.csv"
    if documents.analysis.path.resolve() != expected_analysis.resolve():
        _fail("analysis base was copied away from its code-owned canonical path")
    provenance = documents.analysis.provenance
    if provenance.get("artifact_kind") != analysis_base.ARTIFACT_KIND:
        _fail("ASD preflight requires a canonical analysis-base parent")
    if provenance.get("session_id") != scene or provenance.get("trainable") is not False:
        _fail("analysis-base scene/trainable state is invalid")
    expected_range = analysis_base.EXPECTED_TICK_RANGES[scene]
    tick_range = _require_keys(provenance.get("output_tick_range"),
                               {"start_inclusive", "end_exclusive"},
                               "analysis output tick range")
    start = _exact_int(tick_range["start_inclusive"], "analysis tick start")
    end = _exact_int(tick_range["end_exclusive"], "analysis tick end")
    if (start, end) != expected_range or len(documents.analysis.frame_df) != end - start:
        _fail("analysis base differs from the frozen tick inventory")
    times = documents.analysis.frame_df["timestamp_s"].to_numpy(dtype=np.float64)
    expected_times = np.arange(start, end, dtype=np.float64) / FRAME_RATE_HZ
    if not np.array_equal(times, expected_times):
        _fail("analysis keys are not exact integer 25-Hz ticks")

    crosswalk = documents.crosswalk
    if crosswalk.get("artifact_kind") != crosswalk_module.SEALED_DOWNSTREAM_KIND:
        _fail("identity source is not a sealed downstream crosswalk")
    if crosswalk.get("scene") != scene:
        _fail("crosswalk scene differs from ASD scene")
    if provenance.get("identity_contract") != crosswalk:
        _fail("analysis-base identity contract differs from fresh sealed replay")
    accepted = _require_keys(crosswalk.get("accepted_window"),
                             {"camera", "local_start_inclusive", "local_end_exclusive"},
                             "crosswalk accepted window")
    source_fps = _exact_rate(crosswalk.get("source_fps"), "crosswalk source_fps")
    planned = camera_plan(scene)
    if accepted["camera"] != planned["A"]:
        _fail("sealed accepted camera differs from the code-owned ASD camera plan")
    accepted_start = _exact_int(accepted["local_start_inclusive"], "accepted start")
    accepted_end = _exact_int(accepted["local_end_exclusive"], "accepted end")

    mapping = crosswalk.get("person_mapping")
    if type(mapping) is not list or len(mapping) != 4:
        _fail("sealed crosswalk must contain exactly four person mappings")
    by_participant: dict[str, dict[str, Any]] = {}
    for row in mapping:
        if type(row) is not dict or set(row) != {"hpe_person", "participant", "slot"}:
            _fail("sealed person mapping record is malformed")
        participant = row["participant"]
        if (
            participant not in PARTICIPANT_TO_SLOT
            or row["slot"] != PARTICIPANT_TO_SLOT[participant]
        ):
            _fail("sealed participant-to-slot mapping changed")
        _exact_int(row["hpe_person"], "hpe_person")
        if participant in by_participant:
            _fail("sealed person mapping repeats a participant")
        by_participant[participant] = row
    if set(by_participant) != set(PARTICIPANT_TO_SLOT):
        _fail("sealed person mapping does not cover A/B/C/R")

    identity = documents.identity_manifest
    if identity.get("artifact_kind") != "vap_manual_identity_annotation":
        _fail("identity manifest artifact kind changed")
    if identity.get("format_version") != 2:
        _fail("identity manifest format version changed")
    declared = _require_keys(
        _require_keys(identity.get("source"), {"declared_video"}, "identity source")[
            "declared_video"
        ],
        {"source_path", "camera_dir", "frame_count", "fps", "dataset", "session"},
        "identity declared video",
    )
    day, scenario = scene.rsplit("__", maxsplit=1)
    if declared["dataset"] != day or declared["session"] != scenario:
        _fail("identity source dataset/session differs from the ASD scene")
    if (
        declared["camera_dir"] != planned["A"]
        or _exact_rate(declared["fps"], "identity declared fps") != source_fps
    ):
        _fail("identity source camera/FPS differs from the code-owned ASD plan")
    scope = _require_keys(
        identity.get("annotation_scope"),
        {
            "review_status",
            "source_fps",
            "gaze_local_start_frame_inclusive",
            "gaze_local_end_frame_exclusive",
        },
        "identity annotation scope",
    )
    if (
        scope["review_status"] != "confirmed"
        or _exact_rate(scope["source_fps"], "identity scope source_fps")
        != source_fps
        or _exact_int(
            scope["gaze_local_start_frame_inclusive"], "identity scope start"
        )
        != accepted_start
        or _exact_int(scope["gaze_local_end_frame_exclusive"], "identity scope end")
        != accepted_end
    ):
        _fail("identity annotation scope differs from the sealed accepted window")
    first_segment_frames = _exact_int(declared["frame_count"], "declared frame_count")
    if accepted_end > first_segment_frames:
        _fail("accepted identity window escapes the declared first video segment")
    participant_lexical = _identity_video_path(
        declared, scene=scene, camera=planned["A"], authority=authority
    )
    participant_path = _secure_existing_path(
        participant_lexical,
        root=authority.corpus_root,
        label="participant AVI",
    )
    diagnostic_robot_lexical = _replace_camera(
        participant_lexical, planned["A"], planned["R"]
    )
    diagnostic_robot_path = _secure_existing_path(
        diagnostic_robot_lexical,
        root=authority.corpus_root,
        label="diagnostic robot AVI",
    )

    validation = documents.validation_manifest
    pose_sources = _require_keys(validation.get("pose_sources"), {"cameras"},
                                 "HPE validation pose sources")["cameras"]
    if type(pose_sources) is not dict or planned["R"] not in pose_sources:
        _fail("HPE validation lacks the planned robot camera source manifest")
    robot_source = _require_keys(pose_sources[planned["R"]],
                                 {"bound_camera", "source_fps", "processed_frames"},
                                 "robot-camera HPE source")
    if (
        robot_source["bound_camera"] != planned["R"]
        or _exact_rate(robot_source["source_fps"], "robot source_fps") != source_fps
    ):
        _fail("robot-camera source manifest disagrees with the ASD plan")
    robot_frames = _exact_int(robot_source["processed_frames"], "robot processed_frames")
    if accepted_end > robot_frames:
        _fail("accepted window escapes the robot-camera source manifest")
    cam6_binding_available = (
        documents.cam6_relabeled is not None
        and documents.cam6_run_metadata is not None
    )
    if documents.cam6_run_metadata is not None:
        segments = documents.cam6_run_metadata.get("segments")
        if type(segments) is not list or not segments or any(
            type(value) is not str for value in segments
        ):
            _fail("bound Cam6 run metadata has no exact segment list")
        robot_path = _secure_existing_path(
            _lexical_absolute(segments[0]),
            root=authority.corpus_root,
            label="bound Cam6 AVI",
        )
        if robot_path != diagnostic_robot_path:
            _fail("bound Cam6 run metadata input differs from the code-owned camera path")
        if scene in FIRST_SEGMENT_ONLY and len(segments) < 2:
            _fail("first-segment-only scene lacks its bound multi-segment Cam6 manifest")
    else:
        robot_path = diagnostic_robot_path

    offset = _require_keys(provenance.get("offset_quality"), {"offset_s"},
                           "analysis D3 quality")["offset_s"]
    ticks = np.arange(start, end, dtype=np.int64)
    selected_frames = past_only_frame_indices(
        ticks,
        source_fps=source_fps,
        d3_offset_s=offset,
        accepted_start=accepted_start,
        accepted_end=accepted_end,
    )
    audio_path = authority.audio_contract_root / f"{scene}.audio_contract.json"
    audio_plan = _validate_audio(
        scene,
        documents.audio_report,
        manifest_path=audio_path,
        authority=authority,
    )
    participant_probe = _validate_probe(
        probe(participant_path), expected_path=participant_path, label="participant video"
    )
    robot_probe = _validate_probe(
        probe(robot_path), expected_path=robot_path, label="robot video"
    )
    r_coverage = _association_coverage(
        selected_frames,
        relabeled=documents.cam6_relabeled,
        hpe_person=by_participant["R"]["hpe_person"],
        binding_available=cam6_binding_available,
    )
    return {
        "scene": scene,
        "analysis_tick_range": {
            "start_inclusive": start,
            "end_exclusive": end,
            "n_ticks": end - start,
        },
        "source_fps": source_fps,
        "accepted_source_window": {
            "start_inclusive": accepted_start,
            "end_exclusive": accepted_end,
            "first_segment_only": scene in FIRST_SEGMENT_ONLY,
            "later_segments_excluded_not_zero_filled": scene in FIRST_SEGMENT_ONLY,
        },
        "participant_sources": {
            participant: {
                "slot": PARTICIPANT_TO_SLOT[participant],
                "hpe_person": by_participant[participant]["hpe_person"],
                "camera": planned[participant],
                "video": robot_probe if participant == "R" else participant_probe,
                **({"face_head_association": r_coverage} if participant == "R" else {}),
            }
            for participant in ("R", "A", "B", "C")
        },
        "past_only_resampling": {
            "rule": "floor((tick/25 - d3_offset_s) * source_fps)",
            "arithmetic": "exact_rational",
            "nearest_frame": False,
            "interpolation": False,
            "lookahead": False,
            "first_selected_frame": int(selected_frames[0]),
            "last_selected_frame": int(selected_frames[-1]),
            "duplicate_source_frames_allowed": True,
        },
        "audio_input": audio_plan,
    }


def _build_preflight_report(
    scene: str,
    documents: ASDSourceDocuments,
    *,
    authority: ASDSourceAuthority,
    media_probe: Callable[[Path], dict[str, Any]] = _media_probe,
    fixture_only: bool = True,
) -> dict[str, Any]:
    """Private fixture seam; production callers must use ``audit_*``."""

    if type(authority) is not ASDSourceAuthority:
        _fail("authority has the wrong exact type")
    if type(documents) is not ASDSourceDocuments:
        _fail("documents has the wrong exact type")
    if not callable(media_probe):
        _fail("media_probe must be callable")
    if type(fixture_only) is not bool:
        _fail("fixture_only must be an exact boolean")
    if not fixture_only and media_probe is not _media_probe:
        _fail("production ASD preflight cannot inject a media probe")
    if authority.enforce_frozen_inventory and scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail("scene is outside the frozen inventory")
    if not documents.bindings:
        _fail("source bindings must be non-empty")
    binding_paths: set[Path] = set()
    for name, binding in documents.bindings.items():
        if type(name) is not str or not name:
            _fail("source binding names must be non-empty strings")
        if type(binding) is not dict or set(binding) != _BINDING_KEYS:
            _fail(f"source binding {name!r} has a non-exact schema")
        lexical_binding_path = _lexical_absolute(binding.get("path", ""))
        path = _secure_existing_path(
            lexical_binding_path,
            root=lexical_binding_path.parent,
            label=f"source binding {name}",
        )
        _assert_binding(binding, path=path, label=f"source binding {name}")
        if path in binding_paths:
            _fail("source bindings repeat a path")
        binding_paths.add(path)
        _raw, live = _stable_read(path, f"source binding {name}")
        if live != binding:
            _fail(f"source binding {name!r} is not a live stable file binding")
    plan = _validate_documents(scene, documents, authority, media_probe)
    video_reports = [
        source["video"] for source in plan["participant_sources"].values()
    ]
    opened = any(report["file_opened_or_hydrated"] for report in video_reports)
    if opened:
        _fail("a source probe claims that raw video was opened or hydrated")
    r_association = plan["participant_sources"]["R"]["face_head_association"]
    readiness_blockers = [
        "stock Light-ASD and LR-ASD are noncausal",
        "visual source bytes are deliberately unverified by this stat-only preflight",
        "no causal ASD implementation or output exists",
        "future perturbation/prefix/chunk tests have not been run",
    ]
    if r_association["r_source_valid"] is not True:
        readiness_blockers.append(
            f"R Cam6 face/head association is blocked: {r_association['status']}"
        )
    return {
        "artifact_kind": (
            "vap_real_asd_source_preflight_fixture" if fixture_only else ARTIFACT_KIND
        ),
        "format_version": FORMAT_VERSION,
        "fixture_only": fixture_only,
        "output_role": (
            "test_fixture_no_formal_shape_or_authority"
            if fixture_only
            else "read_only_source_preflight_no_model_execution"
        ),
        "scene": scene,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "formal_feature_published": False,
        "stock_asd_executed": False,
        "raw_video_opened_or_hydrated": opened,
        "source_plan": plan,
        "source_bindings": documents.bindings,
        "upstream_models": {
            "Light-ASD": {
                "repository": "Junhua-Liao/Light-ASD",
                "commit": LIGHT_ASD_COMMIT,
                "stock_noncausal": True,
                "evidence": [
                    "model/Encoder.py:symmetric_temporal_padding",
                    "model/Classifier.py:BGRU_reverse_then_second_GRU",
                    "dataLoader.py:validation_whole_track",
                ],
            },
            "LR-ASD": {
                "repository": "Junhua-Liao/LR-ASD",
                "commit": LR_ASD_COMMIT,
                "stock_noncausal": True,
                "evidence": [
                    "model/Encoder.py:symmetric_temporal_convolution",
                    "model/Classifier.py:forward_backward_GRU_fusion",
                    "dataLoader.py:validation_whole_track",
                ],
            },
        },
        "future_causal_acceptance_tests": {
            "overall_pass": False,
            "tests": {
                name: {
                    "required": True,
                    "status": "not_run_no_model_output",
                    "pass": False,
                }
                for name in (
                    "future_perturbation_invariance",
                    "prefix_equivalence",
                    "chunk_boundary_equivalence",
                )
            },
            "any_failure_blocks_feature_publication": True,
        },
        "readiness_blockers": readiness_blockers,
    }


def _declared_binding_matches(
    declared: Any, actual: dict[str, Any], *, label: str
) -> None:
    if type(declared) is not dict or not {"path", "sha256"}.issubset(declared):
        _fail(f"{label} declared binding is malformed")
    _exact_path(declared["path"], Path(actual["path"]), f"{label} declared path")
    if declared["sha256"] != actual["sha256"]:
        _fail(f"{label} differs from its sealed SHA-256 binding")
    if "size_bytes" in declared and declared["size_bytes"] != actual["size_bytes"]:
        _fail(f"{label} differs from its sealed size binding")


def _load_bound_json(
    declared: Any, *, expected_root: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(declared) is not dict or type(declared.get("path")) is not str:
        _fail(f"{label} declared binding is malformed")
    lexical = _lexical_absolute(declared["path"])
    path = _secure_existing_path(lexical, root=expected_root, label=label)
    value, _raw, actual = _strict_json_bound(path, label)
    _declared_binding_matches(declared, actual, label=label)
    return value, actual


def _load_bound_csv(
    declared: Any, *, expected_root: Path, label: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if type(declared) is not dict or type(declared.get("path")) is not str:
        _fail(f"{label} declared binding is malformed")
    lexical = _lexical_absolute(declared["path"])
    path = _secure_existing_path(lexical, root=expected_root, label=label)
    raw, actual = _stable_read(path, label, root=expected_root)
    _declared_binding_matches(declared, actual, label=label)
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ASDSourcePreflightError(f"{label} CSV is unreadable") from exc
    if (
        declared.get("columns") != list(frame.columns)
        or declared.get("parsed_dtypes") != [str(value) for value in frame.dtypes]
        or declared.get("n_rows") != len(frame)
    ):
        _fail(f"{label} parsed schema/row count differs from its sealed binding")
    return frame, actual


def _assert_gated_snapshot_hashes(
    gated: assembly._DatasetFile,
    analysis_binding: dict[str, Any],
    provenance_binding: dict[str, Any],
) -> None:
    if analysis_binding.get("sha256") != gated.data_sha256:
        _fail("analysis CSV stable-read SHA differs from the gated dataset snapshot")
    if provenance_binding.get("sha256") != gated.provenance_sha256:
        _fail("analysis provenance stable-read SHA differs from the gated snapshot")


def _load_documents(scene: str, authority: ASDSourceAuthority) -> ASDSourceDocuments:
    hpe_root = authority.repository_root.parent / "Head_pose_extraction"
    analysis_path = _secure_existing_path(
        authority.analysis_base_root / f"{scene}.csv",
        root=authority.analysis_base_root,
        label="analysis CSV",
    )
    provenance_path = _secure_existing_path(
        authority.analysis_base_root / f"{scene}.provenance.json",
        root=authority.analysis_base_root,
        label="analysis provenance",
    )
    inventory_path = _secure_existing_path(
        authority.crosswalk_inventory,
        root=authority.repository_root,
        label="sealed crosswalk inventory",
    )
    analysis_authority = analysis_base.default_analysis_base_authority()
    analysis_authority = analysis_base.AnalysisBaseAuthority(
        pose_base_root=analysis_authority.pose_base_root,
        gold_alignment_root=analysis_authority.gold_alignment_root,
        audio_manifest_root=analysis_authority.audio_manifest_root,
        analysis_base_root=authority.analysis_base_root,
        enforce_frozen_inventory=authority.enforce_frozen_inventory,
    )
    gated = assembly._load_dataset_file(
        analysis_path,
        data_kind="real",
        analysis_base_authority=analysis_authority,
    )
    crosswalk = crosswalk_module.load_sealed_crosswalk_contract(
        scene,
        inventory_path,
        repository_root=authority.repository_root,
    )
    manifest_path = _secure_existing_path(
        authority.crosswalk_root / scene / crosswalk_module.MANIFEST_NAME,
        root=authority.crosswalk_root,
        label="crosswalk manifest",
    )
    crosswalk_manifest, _crosswalk_raw, crosswalk_binding = _strict_json_bound(
        manifest_path, "crosswalk manifest"
    )
    artifact_record = _require_keys(
        crosswalk,
        {"artifact_dir", "artifact_files"},
        "sealed crosswalk downstream contract",
    )
    expected_manifest = _require_keys(
        artifact_record["artifact_files"].get(crosswalk_module.MANIFEST_NAME),
        {"sha256", "size_bytes"},
        "sealed crosswalk manifest inventory binding",
    )
    expected_artifact_dir = (
        authority.repository_root / artifact_record["artifact_dir"]
    ).resolve()
    if manifest_path.parent.resolve() != expected_artifact_dir:
        _fail("crosswalk manifest directory differs from the sealed contract")
    if (
        crosswalk_binding["sha256"] != expected_manifest["sha256"]
        or crosswalk_binding["size_bytes"] != expected_manifest["size_bytes"]
    ):
        _fail("crosswalk manifest differs from the sealed inventory binding")
    validation_binding = _require_keys(
        crosswalk_manifest.get("sources", {}).get("hpe", {}).get("validation_complete"),
        {"path", "sha256"},
        "crosswalk HPE validation binding",
    )
    validation, validation_actual = _load_bound_json(
        validation_binding,
        expected_root=hpe_root,
        label="HPE validation manifest",
    )
    identity_binding = _require_keys(
        crosswalk_manifest.get("sources", {}).get("gaze", {}).get("manifest"),
        {"path", "sha256"},
        "crosswalk identity manifest binding",
    )
    identity, identity_actual = _load_bound_json(
        identity_binding,
        expected_root=authority.repository_root,
        label="identity manifest",
    )

    relabeled_bindings = crosswalk_manifest.get("sources", {}).get(
        "relabeled_cameras"
    )
    run_bindings = crosswalk_manifest.get("sources", {}).get("hpe_run_metadata")
    if type(relabeled_bindings) is not dict or type(run_bindings) is not dict:
        _fail("crosswalk camera source bindings are malformed")
    cam6_relabeled: pd.DataFrame | None = None
    cam6_run: dict[str, Any] | None = None
    cam6_relabeled_actual: dict[str, Any] | None = None
    cam6_run_actual: dict[str, Any] | None = None
    has_relabeled = "Cam_6" in relabeled_bindings
    has_run = "Cam_6" in run_bindings
    if has_relabeled != has_run:
        _fail("sealed Cam6 relabeled/run-metadata bindings must appear together")
    if has_relabeled:
        cam6_relabeled, cam6_relabeled_actual = _load_bound_csv(
            relabeled_bindings["Cam_6"],
            expected_root=hpe_root,
            label="Cam6 canonical relabeled CSV",
        )
        cam6_run, cam6_run_actual = _load_bound_json(
            run_bindings["Cam_6"],
            expected_root=hpe_root,
            label="Cam6 run metadata",
        )

    audio_path = _secure_existing_path(
        authority.audio_contract_root / f"{scene}.audio_contract.json",
        root=authority.audio_contract_root,
        label="audio contract",
    )
    audio_manifest, _audio_raw, audio_actual = _strict_json_bound(
        audio_path, "audio contract"
    )
    audio_authority = audio_contract.AudioSourceAuthority(
        corpus_root=authority.corpus_root,
        hpe_delivery_root=authority.repository_root.parent / "Head_pose_extraction"
        / "outputs" / "delivery",
    )
    audio = audio_contract._load_audio_contract_with_authority(
        audio_path, authority=audio_authority
    )
    if not audio_contract._type_exact_json_equal(audio_manifest, audio):
        _fail("audio manifest differs from its complete fresh source replay")

    _analysis_raw, analysis_actual = _stable_read(
        analysis_path, "analysis CSV", root=authority.analysis_base_root
    )
    _provenance_raw, provenance_actual = _stable_read(
        provenance_path, "analysis provenance", root=authority.analysis_base_root
    )
    _inventory_raw, inventory_actual = _stable_read(
        inventory_path, "sealed crosswalk inventory", root=authority.repository_root
    )
    _assert_gated_snapshot_hashes(gated, analysis_actual, provenance_actual)
    bindings = {
        "analysis_csv": analysis_actual,
        "analysis_provenance": provenance_actual,
        "sealed_crosswalk_inventory": inventory_actual,
        "crosswalk_manifest": crosswalk_binding,
        "hpe_validation_manifest": validation_actual,
        "identity_manifest": identity_actual,
        "audio_contract": audio_actual,
    }
    if cam6_relabeled_actual is not None and cam6_run_actual is not None:
        bindings["cam6_relabeled_csv"] = cam6_relabeled_actual
        bindings["cam6_run_metadata"] = cam6_run_actual

    # End-of-snapshot replay: the sealed contract and every directly consumed
    # small source document must still be byte-identical after all parsing.
    crosswalk_after = crosswalk_module.load_sealed_crosswalk_contract(
        scene,
        inventory_path,
        repository_root=authority.repository_root,
    )
    if not audio_contract._type_exact_json_equal(crosswalk, crosswalk_after):
        _fail("sealed crosswalk changed during ASD preflight")
    for name, initial in bindings.items():
        _raw_after, after = _stable_read(Path(initial["path"]), name)
        if after != initial:
            _fail(f"source document {name} changed during ASD preflight")
    return ASDSourceDocuments(
        analysis=gated,
        crosswalk=crosswalk,
        crosswalk_manifest=crosswalk_manifest,
        validation_manifest=validation,
        identity_manifest=identity,
        audio_report=audio,
        bindings=bindings,
        cam6_relabeled=cam6_relabeled,
        cam6_run_metadata=cam6_run,
    )


def audit_asd_source_preflight(
    scene: str,
    *,
    authority: ASDSourceAuthority | None = None,
) -> dict[str, Any]:
    """Strictly replay canonical source documents and return an in-memory report."""

    selected = default_source_authority() if authority is None else authority
    if type(selected) is not ASDSourceAuthority:
        _fail("authority has the wrong exact type")
    documents = _load_documents(scene, selected)
    return _build_preflight_report(
        scene, documents, authority=selected, fixture_only=False
    )


__all__ = [
    "ARTIFACT_KIND",
    "ASDSourceAuthority",
    "ASDSourcePreflightError",
    "FIRST_SEGMENT_ONLY",
    "FORMAT_VERSION",
    "LIGHT_ASD_COMMIT",
    "LR_ASD_COMMIT",
    "audit_asd_source_preflight",
    "camera_plan",
    "default_source_authority",
    "past_only_frame_indices",
]
