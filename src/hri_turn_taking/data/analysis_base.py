"""Provenance-bound common-window crop for real VAP pose bases.

The pose intake table deliberately spans the complete human-reviewed HPE
window.  Gold audio can cover a slightly smaller interval after its per-channel
time maps are replayed.  This module creates the *only* supported bridge between
those two stages: a row-offset slice on the exact integer 25-Hz tick grid.

No interpolation, tolerance join, padding, or caller-supplied time bound is
supported.  A published analysis base remains ``trainable=false``; it merely
establishes the common key set on which independently sourced inputs and gold
targets may later be assembled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import assembly, audio_gold_alignment
from hri_turn_taking.data.multiparty import (
    MAX_PARTICIPANTS,
    participant_column,
)

ARTIFACT_KIND = "vap_real_analysis_base"
CONTRACT_KIND = "vap_real_analysis_window_contract"
CONTRACT_VERSION = 1
FRAME_RATE_HZ = 25
TRUST_MODEL = "trusted_code_integrity_and_reproducibility_not_signature"

# Frozen before any counted real-data result.  A changed upstream window needs
# a new reviewed corpus contract; it must not silently widen this dataset.
EXPECTED_TICK_RANGES: dict[str, tuple[int, int]] = {
    "03_20__4": (376, 5332),
    "03_26__3": (315, 4636),
    "05_14__2": (601, 7351),
    "05_14__3": (624, 12338),
    "05_14__4": (687, 12383),
    "05_15__2": (656, 6973),
    "05_15__3": (586, 6398),
    "05_15__4": (618, 7458),
    "05_15__5": (481, 7924),
    "05_15__6": (629, 13362),
}
EXPECTED_TOTAL_ROWS = 78_582
_BINDING_FIELDS = frozenset({"csv_sha256", "columns", "dtypes", "n_rows"})
_SCENE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*$"
)


class AnalysisBaseError(ValueError):
    """A common-window crop failed a fail-closed source or replay gate."""


@dataclass(frozen=True, kw_only=True)
class AnalysisBaseAuthority:
    """Code-owned formal source, base, and descendant path authority."""

    pose_base_root: Path
    gold_alignment_root: Path
    audio_manifest_root: Path
    analysis_base_root: Path
    descendant_output_paths: tuple[Path, ...] = ()
    enforce_frozen_inventory: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_base_root", Path(self.pose_base_root).resolve())
        object.__setattr__(
            self,
            "gold_alignment_root",
            Path(self.gold_alignment_root).resolve(),
        )
        object.__setattr__(
            self,
            "audio_manifest_root",
            Path(self.audio_manifest_root).resolve(),
        )
        object.__setattr__(
            self,
            "analysis_base_root",
            Path(self.analysis_base_root).resolve(),
        )
        if type(self.descendant_output_paths) is not tuple:
            raise AnalysisBaseError("descendant_output_paths must be an exact tuple")
        descendants = tuple(Path(path).resolve() for path in self.descendant_output_paths)
        if len(descendants) != len(set(descendants)):
            raise AnalysisBaseError("descendant_output_paths contains duplicates")
        if any(path.suffix.lower() != ".csv" for path in descendants):
            raise AnalysisBaseError("every authorised descendant path must end in .csv")
        object.__setattr__(self, "descendant_output_paths", descendants)
        if type(self.enforce_frozen_inventory) is not bool:
            raise AnalysisBaseError("enforce_frozen_inventory must be a boolean")


@dataclass(frozen=True)
class AnalysisBaseResult:
    """In-memory or materialised analysis-base snapshot."""

    frame_df: pd.DataFrame
    provenance: dict[str, Any]
    output_csv: Path | None = None

    @property
    def provenance_path(self) -> Path | None:
        return (
            None
            if self.output_csv is None
            else self.output_csv.with_suffix(".provenance.json")
        )


@dataclass(frozen=True)
class _SourceSnapshot:
    scene: str
    parent: assembly._DatasetFile
    parent_sidecar: Path
    gold_path: Path
    gold_sha256: str
    gold_size_bytes: int
    gold_report: dict[str, Any]
    expected_frame: pd.DataFrame
    contract: dict[str, Any]


def default_analysis_base_authority() -> AnalysisBaseAuthority:
    root = Path(__file__).resolve().parents[3]
    formal_matrix_root = root / "data" / "processed" / "formal_a_h_ah_v1"
    return AnalysisBaseAuthority(
        pose_base_root=root / "data" / "processed" / "intake_v2",
        gold_alignment_root=root / "data" / "annotations" / "audio_gold_alignment",
        audio_manifest_root=root / "data" / "annotations" / "audio_contract",
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        descendant_output_paths=tuple(
            formal_matrix_root / f"{scene}.csv" for scene in EXPECTED_TICK_RANGES
        ),
    )


def _select_authority(authority: object | None) -> AnalysisBaseAuthority:
    if authority is None:
        return default_analysis_base_authority()
    if type(authority) is not AnalysisBaseAuthority:
        raise AnalysisBaseError("analysis_base_authority has the wrong exact type")
    return authority


def _fail(message: str) -> None:
    raise AnalysisBaseError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _type_exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _type_exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_json_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _strict_json_load(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AnalysisBaseError(f"analysis-base sidecar is unreadable UTF-8: {exc}") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"analysis-base sidecar contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        _fail(f"analysis-base sidecar contains forbidden JSON constant {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise AnalysisBaseError(
            f"analysis-base sidecar is invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}"
        ) from exc
    if type(value) is not dict:
        _fail("analysis-base sidecar must contain one JSON object")
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact JSON integer")
    return value


def _validate_scene(scene: str, authority: AnalysisBaseAuthority) -> None:
    if type(scene) is not str or not _SCENE_RE.fullmatch(scene):
        _fail("scene must be one normalised group__scenario identifier")
    if authority.enforce_frozen_inventory and scene not in EXPECTED_TICK_RANGES:
        _fail(f"scene is outside the frozen ten-scene analysis inventory: {scene!r}")


def _canonical_analysis_base_path(
    scene: str,
    authority: AnalysisBaseAuthority,
) -> Path:
    _validate_scene(scene, authority)
    return (authority.analysis_base_root / f"{scene}.csv").resolve()


def _canonical_paths(
    scene: str, authority: AnalysisBaseAuthority
) -> tuple[Path, Path, Path]:
    _validate_scene(scene, authority)
    parent_csv = (authority.pose_base_root / f"{scene}.csv").resolve()
    parent_sidecar = parent_csv.with_suffix(".provenance.json")
    gold_path = (
        authority.gold_alignment_root / f"{scene}.audio_gold_alignment.json"
    ).resolve()
    return parent_csv, parent_sidecar, gold_path


def validate_analysis_descendant_destination(
    scene: str,
    output_csv: str | Path,
    *,
    authority: AnalysisBaseAuthority | None = None,
) -> None:
    """Require one exact code-owned destination for an analysis descendant."""

    selected = _select_authority(authority)
    _validate_scene(scene, selected)
    output = Path(output_csv).resolve()
    if output not in selected.descendant_output_paths:
        _fail(
            "analysis descendant output is not present in the code-owned "
            f"descendant_output_paths authority: {output}"
        )


def _file_binding(
    path: Path,
    *,
    artifact_kind: str,
    format_version: int,
) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"required upstream artifact does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "artifact_kind": artifact_kind,
        "format_version": format_version,
    }


def _assert_same_dataset_snapshot(
    before: assembly._DatasetFile, after: assembly._DatasetFile
) -> None:
    if (
        before.path.resolve() != after.path.resolve()
        or before.data_sha256 != after.data_sha256
        or before.provenance_sha256 != after.provenance_sha256
        or not before.frame_df.equals(after.frame_df)
        or not _type_exact_json_equal(before.provenance, after.provenance)
    ):
        _fail("pose parent or its provenance changed during analysis-window replay")


def _assert_source_bytes_unchanged(
    parent: assembly._DatasetFile,
    parent_sidecar: Path,
    gold_path: Path,
    *,
    gold_sha256: str,
    gold_size_bytes: int,
) -> None:
    if (
        not parent.path.is_file()
        or not parent_sidecar.is_file()
        or _sha256_file(parent.path) != parent.data_sha256
        or _sha256_file(parent_sidecar) != parent.provenance_sha256
        or not gold_path.is_file()
        or gold_path.stat().st_size != gold_size_bytes
        or _sha256_file(gold_path) != gold_sha256
    ):
        _fail("a bound parent/gold source changed during analysis snapshot derivation")


def _validate_parent(parent: assembly._DatasetFile, scene: str) -> dict[str, Any]:
    provenance = parent.provenance
    if type(provenance) is not dict:
        _fail("analysis parent has no real-data provenance")
    if provenance.get("artifact_kind") != assembly.REAL_INTAKE_ARTIFACT_KIND:
        _fail("analysis parent must be a pose-intake artifact, not generated/assembled data")
    if provenance.get("format_version") != assembly.REAL_PROVENANCE_FORMAT_VERSION:
        _fail("analysis parent has an unsupported real provenance version")
    if provenance.get("session_id") != scene:
        _fail("analysis parent session does not match the requested scene")
    if provenance.get("trainable") is not False:
        _fail("pose-intake parent must remain trainable=false")
    if provenance.get("frame_rate_hz") != FRAME_RATE_HZ:
        _fail("analysis parent frame rate must be the exact integer 25")
    return provenance


def _tick_range(value: Any, label: str, *, require_n_ticks: bool) -> tuple[int, int]:
    expected_keys = {
        "frame_rate_hz",
        "start_inclusive",
        "end_exclusive",
        *( ["n_ticks"] if require_n_ticks else [] ),
    }
    if type(value) is not dict or set(value) != expected_keys:
        _fail(f"{label} must contain exactly {sorted(expected_keys)}")
    if _exact_int(value["frame_rate_hz"], f"{label}.frame_rate_hz") != FRAME_RATE_HZ:
        _fail(f"{label}.frame_rate_hz must be 25")
    start = _exact_int(value["start_inclusive"], f"{label}.start_inclusive")
    end = _exact_int(value["end_exclusive"], f"{label}.end_exclusive")
    if start < 0 or end <= start:
        _fail(f"{label} must be a non-empty non-negative half-open interval")
    if require_n_ticks:
        n_ticks = _exact_int(value["n_ticks"], f"{label}.n_ticks")
        if n_ticks != end - start:
            _fail(f"{label}.n_ticks disagrees with its half-open bounds")
    return start, end


def _channel_alignment(gold_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping = gold_report.get("mapping_contract", {}).get("channels")
    decisions = gold_report.get("formal_window_replay", {}).get("channel_decisions")
    if type(mapping) is not dict or set(mapping) != set(audio_gold_alignment.SPEAKERS):
        _fail("gold-alignment mapping must cover exactly A/B/C/R")
    if type(decisions) is not dict or set(decisions) != set(
        audio_gold_alignment.SPEAKERS[1:]
    ):
        _fail("gold-alignment decisions must cover exactly B/C/R")
    result: dict[str, dict[str, Any]] = {}
    for speaker in audio_gold_alignment.SPEAKERS:
        channel = mapping[speaker]
        if type(channel) is not dict:
            _fail(f"gold-alignment mapping for {speaker} is malformed")
        correction = channel.get("correction_s")
        mapping_kind = channel.get("kind")
        if type(correction) is not str or type(mapping_kind) is not str:
            _fail(f"gold-alignment mapping for {speaker} lacks kind/correction")
        try:
            parsed_correction = audio_gold_alignment._decimal_string(
                correction,
                f"mapping_contract.channels.{speaker}.correction_s",
            )
        except ValueError as exc:
            raise AnalysisBaseError(str(exc)) from exc
        if speaker == "A":
            if mapping_kind != "identity_anchor" or parsed_correction != 0:
                _fail("gold-alignment A mapping must remain the zero identity anchor")
            status = "identity_anchor"
            coarse_warning = False
        else:
            decision = decisions[speaker]
            if type(decision) is not dict or type(decision.get("coarse_warning")) is not bool:
                _fail(f"gold-alignment decision for {speaker} is malformed")
            status = decision.get("status")
            coarse_warning = decision["coarse_warning"]
            if status not in {"coarse_confirmed", "resolution_confirmed"}:
                _fail(f"gold-alignment decision for {speaker} is not authorised")
            if mapping_kind not in {"identity", "constant_shift"}:
                _fail(f"gold-alignment mapping kind for {speaker} is unsupported")
            if mapping_kind == "identity" and parsed_correction != 0:
                _fail(f"identity mapping for {speaker} must have zero correction")
            if mapping_kind == "constant_shift" and parsed_correction == 0:
                _fail(f"constant-shift mapping for {speaker} must be non-zero")
            if decision.get("selected_mapping_kind") not in {None, mapping_kind}:
                _fail(f"gold-alignment decision/mapping kind disagree for {speaker}")
            if (status == "coarse_confirmed") is coarse_warning:
                _fail(f"gold-alignment status/coarse_warning disagree for {speaker}")
        result[speaker] = {
            "mapping_kind": mapping_kind,
            "correction_s": correction,
            "status": status,
            "coarse_warning": coarse_warning,
        }
    return result


def _derive_contract(
    *,
    scene: str,
    parent: assembly._DatasetFile,
    parent_sidecar: Path,
    gold_path: Path,
    gold_report: dict[str, Any],
    start_tick: int,
    end_tick: int,
    parent_start: int,
    parent_end: int,
    authority: AnalysisBaseAuthority,
) -> dict[str, Any]:
    provenance = _validate_parent(parent, scene)
    offset_quality = provenance.get("offset_quality")
    if type(offset_quality) is not dict:
        _fail("analysis parent lacks its complete D3 quality summary")
    return {
        "artifact_kind": CONTRACT_KIND,
        "format_version": CONTRACT_VERSION,
        "scene": scene,
        "axis": "mic_A_shared_seconds",
        "frame_rate_hz": FRAME_RATE_HZ,
        "crop_semantics": "integer_row_offset_slice_start_inclusive_end_exclusive",
        "no_interpolation_or_tolerance_join": True,
        "no_padding_or_extrapolation": True,
        "trust_model": TRUST_MODEL,
        "output_authority": {
            "canonical_analysis_base_csv": str(
                _canonical_analysis_base_path(scene, authority)
            ),
            "descendants_require_live_code_owned_path_authority": True,
        },
        "publication_semantics": {
            "method": "two_phase_no_clobber_hardlink_pair",
            "each_destination_replace": False,
            "os_atomic_pair_claimed": False,
            "partial_pair_is_fail_closed": True,
            "baseexception_owned_link_rollback": True,
        },
        "validation_modes": {
            "strict_build_replay": {
                "performed": True,
                "method": "full_load_audio_gold_alignment_raw_source_replay",
                "required_for": ["publication", "explicit_raw_reaudit"],
            },
            "downstream_sealed_replay": {
                "required_on_every_load": True,
                "method": (
                    "strict_gold_manifest_and_bound_source_bytes_plus_parent_slice_replay"
                ),
                "raw_wav_decode_performed": False,
            },
            "explicit_raw_reaudit_api": "reaudit_analysis_base",
        },
        "parent_pose_base": {
            "csv": _file_binding(
                parent.path,
                artifact_kind=assembly.REAL_INTAKE_ARTIFACT_KIND,
                format_version=assembly.REAL_PROVENANCE_FORMAT_VERSION,
            ),
            "provenance": _file_binding(
                parent_sidecar,
                artifact_kind=assembly.REAL_INTAKE_ARTIFACT_KIND,
                format_version=assembly.REAL_PROVENANCE_FORMAT_VERSION,
            ),
        },
        "audio_gold_alignment": _file_binding(
            gold_path,
            artifact_kind=audio_gold_alignment.ARTIFACT_KIND,
            format_version=audio_gold_alignment.FORMAT_VERSION,
        ),
        "parent_tick_range": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_inclusive": parent_start,
            "end_exclusive": parent_end,
        },
        "authorized_tick_range": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "start_inclusive": start_tick,
            "end_exclusive": end_tick,
            "n_ticks": end_tick - start_tick,
        },
        "row_slice": {
            "start_offset_inclusive": start_tick - parent_start,
            "end_offset_exclusive": end_tick - parent_start,
            "n_rows": end_tick - start_tick,
        },
        "channel_alignment": _channel_alignment(gold_report),
        "d3_quality": json.loads(json.dumps(offset_quality)),
        "accepted_pose_window": json.loads(
            json.dumps(provenance.get("accepted_window"))
        ),
        "first_segment_only": scene in {"05_14__3", "05_14__4"},
        "gold_alignment_counted_gold_ready": False,
        "analysis_base_trainable": False,
    }


def _slice_parent(
    parent: assembly._DatasetFile,
    gold_report: dict[str, Any],
    scene: str,
    authority: AnalysisBaseAuthority,
) -> tuple[pd.DataFrame, dict[str, Any], int, int, int, int]:
    provenance = _validate_parent(parent, scene)
    parent_range = provenance.get("output_tick_range")
    if type(parent_range) is not dict or set(parent_range) != {
        "start_inclusive",
        "end_exclusive",
    }:
        _fail("analysis parent output_tick_range is malformed")
    parent_start = _exact_int(parent_range["start_inclusive"], "parent tick start")
    parent_end = _exact_int(parent_range["end_exclusive"], "parent tick end")
    if parent_start < 0 or parent_end <= parent_start:
        _fail("analysis parent has an invalid half-open tick range")
    if parent_end - parent_start != len(parent.frame_df):
        _fail("analysis parent tick range disagrees with its CSV row count")
    expected_parent_time = np.arange(parent_start, parent_end, dtype=np.float64) / FRAME_RATE_HZ
    if not np.array_equal(
        parent.frame_df["timestamp_s"].to_numpy(dtype=np.float64),
        expected_parent_time,
    ):
        _fail("analysis parent is not an exact contiguous integer 25-Hz grid")

    if gold_report.get("scene") != scene:
        _fail("gold-alignment scene does not match the analysis parent")
    gates = gold_report.get("gates")
    authorization = gold_report.get("authorization")
    if (
        type(gates) is not dict
        or gates.get("alignment_prerequisite_pass") is not True
        or gates.get("counted_gold_ready") is not False
        or gold_report.get("counted_gold_ready") is not False
        or type(authorization) is not dict
        or authorization.get("gold_vad_use_authorized") is not True
    ):
        _fail("gold-alignment artifact does not authorise common-window consumption")
    start_tick, end_tick = _tick_range(
        authorization.get("authorized_tick_range"),
        "gold authorized_tick_range",
        require_n_ticks=True,
    )
    if start_tick < parent_start or end_tick > parent_end:
        _fail("gold authorized ticks exceed the bound pose parent")
    if authority.enforce_frozen_inventory:
        expected_range = EXPECTED_TICK_RANGES.get(scene)
        if expected_range != (start_tick, end_tick):
            _fail(
                f"scene {scene} authorised ticks {(start_tick, end_tick)} differ from "
                f"the frozen corpus range {expected_range}"
            )
    start_offset = start_tick - parent_start
    end_offset = end_tick - parent_start
    frame = parent.frame_df.iloc[start_offset:end_offset].copy().reset_index(drop=True)
    expected_time = np.arange(start_tick, end_tick, dtype=np.float64) / FRAME_RATE_HZ
    if len(frame) != end_tick - start_tick or not np.array_equal(
        frame["timestamp_s"].to_numpy(dtype=np.float64), expected_time
    ):
        _fail("integer row-offset crop failed to reproduce the authorised 25-Hz ticks")
    if frame["session_id"].drop_duplicates().tolist() != [scene]:
        _fail("cropped analysis frame contains the wrong session identity")
    return frame, provenance, start_tick, end_tick, parent_start, parent_end


def _gold_pose_binding_matches_parent(
    gold_report: dict[str, Any], parent: assembly._DatasetFile, parent_sidecar: Path
) -> None:
    sources = gold_report.get("sources")
    if type(sources) is not dict:
        _fail("gold-alignment artifact has no bound sources")
    expected = {
        "pose_csv": (parent.path.resolve(), parent.data_sha256),
        "pose_provenance": (parent_sidecar.resolve(), parent.provenance_sha256),
    }
    for name, (expected_path, expected_hash) in expected.items():
        binding = sources.get(name)
        if (
            type(binding) is not dict
            or type(binding.get("path")) is not str
            or Path(binding["path"]).resolve() != expected_path
            or binding.get("sha256") != expected_hash
        ):
            _fail(
                "gold-alignment artifact is not bound to this exact pose parent "
                f"CSV/sidecar ({name})"
            )


def _gold_authority(
    authority: AnalysisBaseAuthority,
) -> audio_gold_alignment.GoldAlignmentAuthority:
    return audio_gold_alignment.GoldAlignmentAuthority(
        audio_manifest_root=authority.audio_manifest_root,
        pose_base_root=authority.pose_base_root,
    )


def _load_gold_alignment_strict(
    path: Path, *, authority: AnalysisBaseAuthority
) -> dict[str, Any]:
    """Replay the formal report all the way back through its raw WAV evidence."""

    gold_authority = _gold_authority(authority)
    default = audio_gold_alignment.default_gold_alignment_authority()
    if gold_authority == default:
        # Keep the public raw-replay API as the production trust boundary.
        return audio_gold_alignment.load_audio_gold_alignment(path)
    # Tests and explicitly isolated audits may use a code-owned temporary
    # authority.  This is the same full loader with only its roots injected.
    return audio_gold_alignment._load_with_authority(path, authority=gold_authority)


def _load_gold_alignment_sealed(
    path: Path, *, authority: AnalysisBaseAuthority
) -> dict[str, Any]:
    """Validate sealed formal bytes without repeatedly decoding multi-GB WAVs.

    This mode is deliberately narrower than the strict build/re-audit path. It
    validates the strict JSON/report contract and re-hashes every source that
    the formal gold report binds (the small audio contract and exact pose
    pair), but trusts the already-reviewed gold report's raw-audio replay.  The
    analysis sidecar separately binds the complete gold-report bytes and the
    warning/correction snapshot used by the crop.
    """

    if not path.is_file():
        _fail(f"gold-alignment artifact does not exist: {path}")
    try:
        report = audio_gold_alignment._strict_json_loads(
            path.read_bytes(), "gold-alignment artifact"
        )
        audio_gold_alignment._validate_report_shape(report)
        audio_gold_alignment._verify_bound_sources(
            report,
            authority=_gold_authority(authority),
        )
    except (OSError, ValueError) as exc:
        raise AnalysisBaseError(
            f"gold-alignment sealed downstream replay failed: {exc}"
        ) from exc
    return report


def _load_source_snapshot(
    scene: str,
    *,
    authority: AnalysisBaseAuthority,
    strict_raw_replay: bool,
) -> _SourceSnapshot:
    if type(strict_raw_replay) is not bool:
        _fail("strict_raw_replay must be a boolean")
    parent_csv, parent_sidecar, gold_path = _canonical_paths(scene, authority)
    try:
        parent_before = assembly._load_dataset_file(parent_csv, data_kind="real")
    except (OSError, ValueError) as exc:
        raise AnalysisBaseError(f"pose parent full replay failed: {exc}") from exc
    _validate_parent(parent_before, scene)
    if not parent_sidecar.is_file() or not gold_path.is_file():
        _fail("canonical pose sidecar or gold-alignment artifact is missing")
    gold_hash_before = _sha256_file(gold_path)
    gold_size_before = gold_path.stat().st_size
    try:
        if strict_raw_replay:
            gold_report = _load_gold_alignment_strict(gold_path, authority=authority)
        else:
            gold_report = _load_gold_alignment_sealed(gold_path, authority=authority)
    except (OSError, ValueError) as exc:
        mode = "full raw" if strict_raw_replay else "sealed downstream"
        raise AnalysisBaseError(f"gold-alignment {mode} replay failed: {exc}") from exc
    if (
        not gold_path.is_file()
        or gold_path.stat().st_size != gold_size_before
        or _sha256_file(gold_path) != gold_hash_before
    ):
        _fail("gold-alignment artifact changed while the analysis source was loaded")
    _gold_pose_binding_matches_parent(gold_report, parent_before, parent_sidecar)
    try:
        parent_after = assembly._load_dataset_file(parent_csv, data_kind="real")
    except (OSError, ValueError) as exc:
        raise AnalysisBaseError(f"pose parent final replay failed: {exc}") from exc
    _assert_same_dataset_snapshot(parent_before, parent_after)
    _gold_pose_binding_matches_parent(gold_report, parent_after, parent_sidecar)
    frame, provenance, start, end, parent_start, parent_end = _slice_parent(
        parent_after, gold_report, scene, authority
    )
    contract = _derive_contract(
        scene=scene,
        parent=parent_after,
        parent_sidecar=parent_sidecar,
        gold_path=gold_path,
        gold_report=gold_report,
        start_tick=start,
        end_tick=end,
        parent_start=parent_start,
        parent_end=parent_end,
        authority=authority,
    )
    _assert_source_bytes_unchanged(
        parent_after,
        parent_sidecar,
        gold_path,
        gold_sha256=gold_hash_before,
        gold_size_bytes=gold_size_before,
    )
    return _SourceSnapshot(
        scene=scene,
        parent=parent_after,
        parent_sidecar=parent_sidecar,
        gold_path=gold_path,
        gold_sha256=gold_hash_before,
        gold_size_bytes=gold_size_before,
        gold_report=gold_report,
        expected_frame=frame,
        contract=contract,
    )


def _expected_analysis_provenance(
    snapshot: _SourceSnapshot,
    *,
    analysis_base_csv: Path | None,
) -> dict[str, Any]:
    parent = snapshot.parent.provenance
    assert parent is not None
    result = json.loads(json.dumps(parent))
    ticks = snapshot.contract["authorized_tick_range"]
    start = ticks["start_inclusive"]
    end = ticks["end_exclusive"]
    result.update(
        {
            "artifact_kind": ARTIFACT_KIND,
            "format_version": assembly.REAL_PROVENANCE_FORMAT_VERSION,
            "output_tick_range": {
                "start_inclusive": start,
                "end_exclusive": end,
            },
            "n_frames": end - start,
            "time_range_shared_s": [
                start / FRAME_RATE_HZ,
                (end - 1) / FRAME_RATE_HZ,
            ],
            "trainable": False,
            "analysis_base_csv": (
                None if analysis_base_csv is None else str(analysis_base_csv.resolve())
            ),
            "analysis_window_contract": snapshot.contract,
        }
    )
    # A pose intake has neither key.  Remove them defensively so a future
    # intake version cannot make this crop look like a feature assembly.
    result.pop("assembly", None)
    result.pop("feature_sources", None)
    result.pop("trainability", None)
    result.pop("inert_columns", None)
    for key in _BINDING_FIELDS:
        result.pop(key, None)
    return result


def derive_analysis_base(
    scene: str,
    *,
    authority: AnalysisBaseAuthority | None = None,
) -> AnalysisBaseResult:
    """Build one fully replayed crop in memory without publishing files."""

    selected = _select_authority(authority)
    snapshot = _load_source_snapshot(
        scene,
        authority=selected,
        strict_raw_replay=True,
    )
    provenance = _expected_analysis_provenance(snapshot, analysis_base_csv=None)
    return AnalysisBaseResult(
        frame_df=snapshot.expected_frame.copy(),
        provenance=provenance,
    )


def _verify_inherited_contract(
    provenance: dict[str, Any],
    snapshot: _SourceSnapshot,
    *,
    artifact_kind: str,
    csv_path: Path,
    authority: AnalysisBaseAuthority,
) -> None:
    anchor_value = provenance.get("analysis_base_csv")
    if type(anchor_value) is not str:
        _fail("published analysis provenance must bind its original analysis_base_csv")
    anchor = Path(anchor_value).resolve()
    canonical_anchor = _canonical_analysis_base_path(snapshot.scene, authority)
    if anchor != canonical_anchor:
        _fail(
            "analysis_base_csv does not match the live code-owned canonical base authority"
        )
    if artifact_kind == ARTIFACT_KIND:
        if csv_path != canonical_anchor:
            _fail("analysis-base CSV is not at its code-owned canonical published path")
    else:
        validate_analysis_descendant_destination(
            snapshot.scene,
            csv_path,
            authority=authority,
        )
        if anchor == csv_path:
            _fail("assembled descendant cannot claim itself as the original analysis base")
        try:
            if anchor.is_file() and csv_path.is_file() and os.path.samefile(anchor, csv_path):
                _fail("assembled descendant cannot hard-link itself to its analysis base")
        except OSError as exc:
            raise AnalysisBaseError(
                f"could not verify analysis-base lineage file ownership: {exc}"
            ) from exc
    expected = _expected_analysis_provenance(
        snapshot,
        analysis_base_csv=anchor,
    )
    if artifact_kind == ARTIFACT_KIND:
        actual_core = {
            key: value for key, value in provenance.items() if key not in _BINDING_FIELDS
        }
        if set(actual_core) != set(expected):
            _fail(
                "analysis-base provenance fields differ from deterministic replay; "
                f"missing={sorted(set(expected) - set(actual_core))}, "
                f"extra={sorted(set(actual_core) - set(expected))}"
            )
        for key, value in expected.items():
            if not _type_exact_json_equal(actual_core.get(key), value):
                _fail(f"analysis-base provenance field {key!r} differs from replay")
        return

    # An assembled descendant may change feature declarations and assembly
    # metadata, but never identity, timing authority, the crop, or pose values.
    mutable = {
        "artifact_kind",
        "format_version",
        "real_columns",
        "placeholder_columns",
        "trainable",
        "trainability",
        "inert_columns",
        "feature_sources",
        "column_sources",
        "assembly",
        *_BINDING_FIELDS,
    }
    for key, value in expected.items():
        if key in mutable:
            continue
        if not _type_exact_json_equal(provenance.get(key), value):
            _fail(f"assembled descendant changed analysis-base field {key!r}")
    parent_sources = snapshot.parent.provenance.get("column_sources", {})  # type: ignore[union-attr]
    actual_sources = provenance.get("column_sources")
    if type(actual_sources) is not dict:
        _fail("assembled descendant has no column_sources object")
    for column, binding in parent_sources.items():
        if not _type_exact_json_equal(actual_sources.get(column), binding):
            _fail(f"assembled descendant changed inherited pose source {column!r}")

    assembly_record = provenance.get("assembly")
    base_path_value = (
        assembly_record.get("base_csv") if type(assembly_record) is dict else None
    )
    if type(base_path_value) is not str:
        _fail("assembled descendant has no immediate base_csv binding")
    base_path = Path(base_path_value).resolve()
    if base_path == csv_path:
        _fail("assembled analysis lineage cannot point to itself")
    try:
        immediate = assembly._load_dataset_file(
            base_path,
            data_kind="real",
            analysis_base_authority=authority,
        )
    except (OSError, ValueError) as exc:
        raise AnalysisBaseError(f"assembled analysis lineage replay failed: {exc}") from exc
    immediate_provenance = immediate.provenance
    if (
        type(immediate_provenance) is not dict
        or immediate_provenance.get("artifact_kind")
        not in {ARTIFACT_KIND, assembly.REAL_ASSEMBLED_ARTIFACT_KIND}
        or not _type_exact_json_equal(
            immediate_provenance.get("analysis_window_contract"), snapshot.contract
        )
        or immediate_provenance.get("analysis_base_csv") != str(anchor)
    ):
        _fail("assembled descendant did not preserve an analysis-base immediate lineage")


def validate_analysis_base_artifact(
    csv_path: str | Path,
    frame_df: pd.DataFrame,
    provenance: dict[str, Any],
    sidecar_path: str | Path,
    *,
    authority: AnalysisBaseAuthority | None = None,
    strict_raw_replay: bool = False,
) -> None:
    """Replay the common-window contract for a base or assembled descendant.

    This is called by :mod:`assembly` after its generic real-data gates. Its
    ordinary downstream mode replays the sealed formal manifest, exact bound
    parent bytes and deterministic slice without re-decoding multi-GB WAVs.
    ``strict_raw_replay=True`` is reserved for publication and explicit source
    re-audit and invokes the complete gold-audio loader.
    """

    path = Path(csv_path).resolve()
    sidecar = Path(sidecar_path).resolve()
    selected = _select_authority(authority)
    parsed = _strict_json_load(sidecar)
    if not _type_exact_json_equal(parsed, provenance):
        _fail("analysis-base sidecar changed or was parsed ambiguously")
    artifact_kind = provenance.get("artifact_kind")
    if artifact_kind not in {ARTIFACT_KIND, assembly.REAL_ASSEMBLED_ARTIFACT_KIND}:
        _fail("analysis-window contract is allowed only on an analysis base/descendant")
    scene = provenance.get("session_id")
    if type(scene) is not str:
        _fail("analysis-base session_id must be a string")
    snapshot = _load_source_snapshot(
        scene,
        authority=selected,
        strict_raw_replay=strict_raw_replay,
    )
    contract = provenance.get("analysis_window_contract")
    if not _type_exact_json_equal(contract, snapshot.contract):
        _fail("analysis_window_contract differs from fresh parent/gold replay")

    expected = snapshot.expected_frame
    if list(frame_df.columns) != list(expected.columns):
        # Assembled descendants retain the same full canonical schema; feature
        # assembly replaces values, not columns or key layout.
        _fail("analysis-base/descendant columns differ from the cropped parent schema")
    if len(frame_df) != len(expected):
        _fail("analysis-base/descendant row count differs from the authorised crop")
    pose_columns = sorted(assembly._expected_intake_pose_columns(expected))
    replay_columns = [
        *assembly.KEY_COLUMNS,
        *(
            participant_column(slot, "present")
            for slot in range(1, MAX_PARTICIPANTS + 1)
        ),
        *pose_columns,
    ]
    if artifact_kind == ARTIFACT_KIND:
        if not frame_df.equals(expected):
            _fail("analysis-base CSV is not the exact deterministic parent row slice")
    else:
        if not np.array_equal(
            frame_df["session_id"].to_numpy(dtype=object),
            expected["session_id"].to_numpy(dtype=object),
        ):
            _fail("assembled descendant changed cropped session keys")
        numeric = [column for column in replay_columns if column != "session_id"]
        if not np.array_equal(
            frame_df[numeric].to_numpy(dtype=float),
            expected[numeric].to_numpy(dtype=float),
        ):
            _fail("assembled descendant changed cropped ticks/presence/HPE pose values")
    _verify_inherited_contract(
        provenance,
        snapshot,
        artifact_kind=artifact_kind,
        csv_path=path,
        authority=selected,
    )


def _publish_no_clobber(staged: Path, destination: Path) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise AnalysisBaseError(
            f"analysis output appeared during publication: {destination}"
        ) from exc
    except OSError as exc:
        raise AnalysisBaseError(
            f"could not publish analysis output without overwriting {destination}: {exc}"
        ) from exc


def _reserve_staged_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=prefix,
        suffix=suffix,
        dir=directory,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _snapshot_equal(left: _SourceSnapshot, right: _SourceSnapshot) -> bool:
    return (
        left.scene == right.scene
        and left.parent.data_sha256 == right.parent.data_sha256
        and left.parent.provenance_sha256 == right.parent.provenance_sha256
        and left.gold_sha256 == right.gold_sha256
        and left.gold_size_bytes == right.gold_size_bytes
        and left.expected_frame.equals(right.expected_frame)
        and _type_exact_json_equal(left.contract, right.contract)
    )


def _load_owned_published_pair(
    output: Path,
    sidecar: Path,
    *,
    authority: AnalysisBaseAuthority,
    csv_identity: tuple[int, int],
    sidecar_identity: tuple[int, int],
    csv_sha256: str,
    sidecar_sha256: str,
) -> assembly._DatasetFile:
    """Replay one pair while held handles bind every hash to the same inodes."""

    try:
        with assembly._held_file_snapshot_guard(
            (
                (output, csv_identity, csv_sha256),
                (sidecar, sidecar_identity, sidecar_sha256),
            )
        ):
            loaded = assembly._load_dataset_file(
                output,
                data_kind="real",
                analysis_base_authority=authority,
            )
            if (
                loaded.data_sha256 != csv_sha256
                or loaded.provenance_sha256 != sidecar_sha256
            ):
                _fail("published analysis replay returned unexpected bytes")
    except (OSError, ValueError) as exc:
        raise AnalysisBaseError(
            f"published analysis pair failed held-handle replay: {exc}"
        ) from exc
    return loaded


def write_analysis_base(
    scene: str,
    output_csv: str | Path,
    *,
    authority: AnalysisBaseAuthority | None = None,
) -> AnalysisBaseResult:
    """Publish one CSV/sidecar pair using an exact, no-clobber two-phase write."""

    selected = _select_authority(authority)
    output = Path(output_csv).resolve()
    sidecar = output.with_suffix(".provenance.json")
    parent_csv, parent_sidecar, gold_path = _canonical_paths(scene, selected)
    canonical_output = _canonical_analysis_base_path(scene, selected)
    forbidden = {parent_csv, parent_sidecar, gold_path}
    if output.suffix.lower() != ".csv":
        _fail("analysis output must use a .csv suffix")
    if output != canonical_output:
        _fail(
            "analysis output must equal the code-owned canonical path "
            f"{canonical_output}"
        )
    if output in forbidden or sidecar in forbidden:
        _fail("analysis output pair must differ from every bound source")
    if output.exists() or sidecar.exists():
        _fail(f"refusing to overwrite analysis output pair: {output}, {sidecar}")

    snapshot = _load_source_snapshot(
        scene,
        authority=selected,
        strict_raw_replay=True,
    )
    provenance = _expected_analysis_provenance(
        snapshot,
        analysis_base_csv=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp: Path | None = None
    sidecar_tmp: Path | None = None
    csv_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        csv_tmp = _reserve_staged_path(
            output.parent,
            prefix=f".{output.name}.tmp.",
            suffix=".csv",
        )
        csv_identity = assembly._file_identity(csv_tmp)
        sidecar_tmp = _reserve_staged_path(
            sidecar.parent,
            prefix=f".{sidecar.name}.tmp.",
            suffix=".json",
        )
        sidecar_identity = assembly._file_identity(sidecar_tmp)
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            snapshot.expected_frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = assembly._read_csv(csv_tmp, error_type=AnalysisBaseError)
        bound = {
            **provenance,
            **assembly._csv_binding(csv_tmp, persisted),
            "n_frames": len(persisted),
        }
        with sidecar_tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(
                    bound,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        staged_csv_hash = _sha256_file(csv_tmp)
        staged_sidecar_hash = _sha256_file(sidecar_tmp)
        # Re-hash all formal bytes immediately before publication.  The first
        # and final snapshots perform the expensive full raw-audio replay; this
        # sealed middle check closes the publication TOCTOU window without a
        # redundant WAV decode.
        latest = _load_source_snapshot(
            scene,
            authority=selected,
            strict_raw_replay=False,
        )
        if not _snapshot_equal(snapshot, latest):
            _fail("analysis sources changed before publication")
        if (
            _sha256_file(csv_tmp) != staged_csv_hash
            or _sha256_file(sidecar_tmp) != staged_sidecar_hash
        ):
            _fail("analysis staging bytes changed before publication")
        published.append((output, csv_identity))
        _publish_no_clobber(csv_tmp, output)
        published.append((sidecar, sidecar_identity))
        _publish_no_clobber(sidecar_tmp, sidecar)
        loaded = _load_owned_published_pair(
            output,
            sidecar,
            authority=selected,
            csv_identity=csv_identity,
            sidecar_identity=sidecar_identity,
            csv_sha256=staged_csv_hash,
            sidecar_sha256=staged_sidecar_hash,
        )
        final_sources = _load_source_snapshot(
            scene,
            authority=selected,
            strict_raw_replay=True,
        )
        if not _snapshot_equal(snapshot, final_sources):
            _fail("analysis sources changed during publication")
        # The strict source replay can take minutes. Re-open and hold both
        # targets while the exact pair is loaded one final time.
        loaded = _load_owned_published_pair(
            output,
            sidecar,
            authority=selected,
            csv_identity=csv_identity,
            sidecar_identity=sidecar_identity,
            csv_sha256=staged_csv_hash,
            sidecar_sha256=staged_sidecar_hash,
        )
        return AnalysisBaseResult(
            frame_df=loaded.frame_df,
            provenance=loaded.provenance or {},
            output_csv=output,
        )
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for destination, identity in reversed(published):
            cleanup = assembly._remove_owned_path(destination, identity)
            if not cleanup.removed:
                recovery = (
                    f"; recovery={cleanup.recovery_path}"
                    if cleanup.recovery_path is not None
                    else ""
                )
                cleanup_errors.append(
                    f"{destination}: {cleanup.detail or 'path preserved'}{recovery}"
                )
        if cleanup_errors:
            raise AnalysisBaseError(
                "analysis publication failed and rollback preserved unowned paths: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    finally:
        if csv_tmp is not None and csv_identity is not None:
            assembly._remove_owned_path(csv_tmp, csv_identity)
        if sidecar_tmp is not None and sidecar_identity is not None:
            assembly._remove_owned_path(sidecar_tmp, sidecar_identity)


def reaudit_analysis_base(
    csv_path: str | Path,
    *,
    authority: AnalysisBaseAuthority | None = None,
) -> AnalysisBaseResult:
    """Explicitly replay a published base/descendant through raw WAV sources."""

    path = Path(csv_path).resolve()
    try:
        # Ordinary loading first exercises the exact sealed downstream path
        # used by every dataset fold and descendant.
        loaded = assembly._load_dataset_file(
            path,
            data_kind="real",
            analysis_base_authority=authority,
        )
    except (OSError, ValueError) as exc:
        raise AnalysisBaseError(
            f"analysis-base sealed pre-audit failed: {exc}"
        ) from exc
    provenance = loaded.provenance
    if type(provenance) is not dict:
        _fail("analysis-base raw re-audit requires real provenance")
    sidecar = path.with_suffix(".provenance.json")
    validate_analysis_base_artifact(
        path,
        loaded.frame_df,
        provenance,
        sidecar,
        authority=authority,
        strict_raw_replay=True,
    )
    return AnalysisBaseResult(
        frame_df=loaded.frame_df.copy(),
        provenance=json.loads(json.dumps(provenance)),
        output_csv=path,
    )


def validate_frozen_matrix(results: dict[str, AnalysisBaseResult]) -> None:
    """Assert the exact ten-scene inventory and aggregate row count."""

    if set(results) != set(EXPECTED_TICK_RANGES):
        _fail("analysis result set does not exactly match the frozen ten-scene inventory")
    for scene, expected_range in EXPECTED_TICK_RANGES.items():
        contract = results[scene].provenance["analysis_window_contract"]
        ticks = contract["authorized_tick_range"]
        if (ticks["start_inclusive"], ticks["end_exclusive"]) != expected_range:
            _fail(f"analysis result for {scene} differs from the frozen tick range")
        if len(results[scene].frame_df) != expected_range[1] - expected_range[0]:
            _fail(f"analysis result for {scene} has the wrong row count")
    total = sum(len(result.frame_df) for result in results.values())
    if total != EXPECTED_TOTAL_ROWS:
        _fail(f"analysis matrix has {total} rows; expected exactly {EXPECTED_TOTAL_ROWS}")


__all__ = [
    "ARTIFACT_KIND",
    "AnalysisBaseAuthority",
    "AnalysisBaseError",
    "AnalysisBaseResult",
    "EXPECTED_TICK_RANGES",
    "EXPECTED_TOTAL_ROWS",
    "default_analysis_base_authority",
    "derive_analysis_base",
    "reaudit_analysis_base",
    "validate_analysis_base_artifact",
    "validate_analysis_descendant_destination",
    "validate_frozen_matrix",
    "write_analysis_base",
]
