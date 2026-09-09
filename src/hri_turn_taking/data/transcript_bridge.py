"""Controlled bridge from a confirmed Transcript review to coarse activity.

The manual-review final is the sole authoring surface.  This module fully
replays that final (and therefore its bound draft, audio contract, gold
alignment and live source bytes), deterministically derives the existing
Transcript-acceptance v3 artifact, and rasterises only the authorised half-open
25-Hz window.  A bundle receipt binds all three layers so the per-event review
audit is not lost when the compact acceptance artifact is consumed later.

Nothing emitted here is counted gold VAD.  Counted labels remain gated on the
separate lavalier-assisted refinement and human audit.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from hri_turn_taking.data import transcript_acceptance as acceptance_module
from hri_turn_taking.data import transcript_fs_guard as fs_guard
from hri_turn_taking.data import transcript_review as review_module

BUNDLE_ARTIFACT_KIND = "vap_transcript_review_bridge_bundle"
BUNDLE_FORMAT_VERSION = 2
TRUST_MODEL = review_module.TRUST_MODEL
ACCEPTANCE_FILENAME = "transcript_acceptance.v3.json"
ACTIVITY_FILENAME = "coarse_activity_25hz.csv"
RECEIPT_FILENAME = "bridge_receipt.json"
ACTIVITY_COLUMNS = ("tick", "timestamp_sec", "A", "B", "C", "R")
_PAYLOAD_FILENAMES = (ACCEPTANCE_FILENAME, ACTIVITY_FILENAME, RECEIPT_FILENAME)
FORMAL_BRIDGE_ROOT = (
    review_module.PROJECT_ROOT / "data" / "annotations" / "transcript_bridge"
)


class TranscriptBridgeError(ValueError):
    """Raised when a final review cannot be bridged without losing evidence."""


@dataclass(frozen=True)
class TranscriptBridgeBundle:
    """One fully replayed coarse-reference bundle."""

    path: Path
    scene: str
    final_review: dict[str, Any]
    acceptance: acceptance_module.AcceptedTranscriptReference
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _CanonicalBundleSnapshot:
    path: Path
    scene: str
    receipt_raw: bytes
    receipt: dict[str, Any]
    final: review_module.CanonicalFinalSnapshot


def _fail(message: str) -> None:
    raise TranscriptBridgeError(message)


def _require_current_bundle_format(value: Any) -> int:
    if type(value) is not int:
        _fail("bridge receipt format_version must be an integer")
    if value == 1:
        _fail(
            "bridge receipt v1 is stale; rebuild it from a canonical review-v3 final"
        )
    if value != BUNDLE_FORMAT_VERSION:
        _fail(f"bridge receipt format_version {value} is unsupported")
    return value


def canonical_bridge_bundle_path(scene: str) -> Path:
    """Return the only production bundle path authorised for one scene."""

    review_module._scene_parts(scene)
    return FORMAL_BRIDGE_ROOT / scene


def _ensure_formal_bridge_root() -> Path:
    annotations_root = FORMAL_BRIDGE_ROOT.parent
    try:
        fs_guard.guard_directory(annotations_root)
        try:
            FORMAL_BRIDGE_ROOT.lstat()
        except FileNotFoundError:
            FORMAL_BRIDGE_ROOT.mkdir(exist_ok=False)
        return fs_guard.guard_directory(FORMAL_BRIDGE_ROOT)[0]
    except (fs_guard.TranscriptPathGuardError, OSError) as exc:
        raise TranscriptBridgeError(
            f"cannot establish canonical formal bridge root under {annotations_root}"
        ) from exc


def _require_canonical_bundle_path(path: str | Path, *, scene: str) -> Path:
    expected = canonical_bridge_bundle_path(scene)
    absolute = fs_guard.lexical_absolute(path)
    if absolute != expected:
        _fail(f"formal bridge bundle path must be exactly {expected}")
    return absolute


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
        raise TranscriptBridgeError("bridge artifact is not strict JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_binding(path: Path, raw: bytes) -> dict[str, Any]:
    absolute = fs_guard.lexical_absolute(path)
    return {
        "path": str(absolute),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "media_type": "application/json",
    }


def _minimal_file_binding(path: Path) -> dict[str, Any]:
    """Bind one source from the exact bytes of a held no-follow descriptor."""

    try:
        with fs_guard.guarded_read(path) as held:
            raw = fs_guard.read_guarded_bytes(held)
            return {
                "path": str(held.path),
                "size_bytes": len(raw),
                "sha256": _sha256(raw),
            }
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptBridgeError(f"source path is unsafe: {exc}") from exc


def _binding_candidates(value: Any) -> Iterator[dict[str, Any]]:
    """Yield nested file bindings without treating arbitrary ``path`` keys as files."""

    if isinstance(value, dict):
        if {"path", "size_bytes", "sha256"} <= value.keys():
            yield value
        for nested in value.values():
            yield from _binding_candidates(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _binding_candidates(nested)


def _binding_path_and_expected(
    binding: dict[str, Any], label: str
) -> tuple[Path, int, str, str | None]:
    path_text = binding.get("path")
    size = binding.get("size_bytes")
    digest = binding.get("sha256")
    media_type = binding.get("media_type")
    if type(path_text) is not str or not path_text or not Path(path_text).is_absolute():
        _fail(f"{label}.path must be a non-empty absolute path")
    if type(size) is not int or size < 0:
        _fail(f"{label}.size_bytes must be a non-negative integer")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if media_type is not None and type(media_type) is not str:
        _fail(f"{label}.media_type must be a string when present")
    return fs_guard.lexical_absolute(path_text), size, digest, media_type


@contextmanager
def _held_binding_graph(value: Any) -> Iterator[dict[Path, bytes]]:
    """Hold and replay every declared source binding reachable from JSON.

    JSON sources (draft, audio contract, gold alignment, etc.) are expanded
    recursively.  All handles remain live until the caller's complete replay
    finishes, denying WRITE/DELETE on Windows and closing pathname TOCTOU on
    every platform through final identity/state checks.
    """

    raw_by_path: dict[Path, bytes] = {}
    expected_by_path: dict[Path, tuple[int, str, str | None]] = {}
    queue: list[Any] = [value]
    with ExitStack() as stack:
        while queue:
            current = queue.pop()
            for index, binding in enumerate(_binding_candidates(current)):
                path, size, digest, media_type = _binding_path_and_expected(
                    binding, f"declared binding {index}"
                )
                expected = (size, digest, media_type)
                previous = expected_by_path.get(path)
                if previous is not None:
                    if previous[:2] != expected[:2] or (
                        previous[2] is not None
                        and media_type is not None
                        and previous[2] != media_type
                    ):
                        _fail(f"conflicting bindings name the same source path: {path}")
                    if previous[2] is None and media_type is not None:
                        expected_by_path[path] = expected
                    continue
                expected_by_path[path] = expected
                held = stack.enter_context(fs_guard.guarded_read(path))
                raw = fs_guard.read_guarded_bytes(held)
                if len(raw) != size or _sha256(raw) != digest:
                    _fail(f"declared source binding no longer matches: {path}")
                raw_by_path[path] = raw
                if media_type == "application/json" or path.suffix.lower() == ".json":
                    queue.append(
                        review_module._strict_json_loads(
                            raw, f"declared JSON source {path}"
                        )
                    )
        yield raw_by_path
        for path in raw_by_path:
            # ExitStack performs the held-descriptor checks as it unwinds;
            # this path guard also makes ancestor replacement explicit here.
            fs_guard.guard_regular_file_path(path)


def _local_binding(
    filename: str,
    raw: bytes,
    *,
    media_type: str,
    n_rows: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": filename,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "media_type": media_type,
    }
    if n_rows is not None:
        result["n_rows"] = n_rows
        result["columns"] = list(ACTIVITY_COLUMNS)
    return result


def _without_media_type(binding_value: Any, label: str) -> dict[str, Any]:
    binding = review_module._require_object(binding_value, label)
    review_module._require_exact_keys(
        binding, {"path", "size_bytes", "sha256", "media_type"}, label
    )
    return {key: binding[key] for key in ("path", "size_bytes", "sha256")}


def _acceptance_authority(
    authority: review_module.ReviewSourceAuthority,
) -> acceptance_module.TranscriptSourceAuthority:
    return acceptance_module.TranscriptSourceAuthority(
        corpus_root=authority.corpus_root,
        hpe_delivery_root=authority.hpe_delivery_root,
        audio_contract_root=authority.audio_contract_root,
        pose_base_root=authority.pose_base_root,
        gold_alignment_root=authority.gold_alignment_root,
    )


def _timing_from_sources(
    final: dict[str, Any],
    *,
    authority: review_module.ReviewSourceAuthority,
) -> dict[str, Any]:
    scene = final["scene"]
    canonical = acceptance_module._canonical_source_paths(
        scene, _acceptance_authority(authority)
    )
    try:
        with ExitStack() as stack:
            held = {
                name: stack.enter_context(fs_guard.guarded_read(canonical[name]))
                for name in ("offset", "hpe_offset", "hpe_offset_quality")
            }
            fs_guard.guard_regular_file_path(canonical["reference_wav"])
            raw_offset = fs_guard.read_guarded_bytes(held["offset"])
            offset_value = acceptance_module._strict_json_loads(
                raw_offset, "transcript offset"
            )
            normalized, _operations = (
                acceptance_module._normalize_transcript_offset_source(
                    raw_offset, offset_value, final["source_normalizations"]
                )
            )
            transcript_offset, processing_rate = (
                acceptance_module._validate_transcript_offset(
                    normalized, scene=scene, reference_wav=canonical["reference_wav"]
                )
            )
            hpe_offset_value = acceptance_module._strict_json_loads(
                fs_guard.read_guarded_bytes(held["hpe_offset"]), "HPE offset"
            )
            hpe_quality_value = acceptance_module._strict_json_loads(
                fs_guard.read_guarded_bytes(held["hpe_offset_quality"]),
                "HPE offset quality",
            )
            streams = acceptance_module._require_object(
                acceptance_module._require_object(
                    hpe_offset_value, "HPE offset"
                ).get("streams"),
                "HPE offset.streams",
            )
            hpe_offset = acceptance_module._require_real_number(
                streams.get("6cam"), "HPE 6cam offset"
            )
            quality = acceptance_module._require_object(
                acceptance_module._require_object(
                    hpe_quality_value, "HPE offset quality"
                ).get("quality"),
                "HPE quality",
            )
            stream_quality = acceptance_module._require_object(
                quality.get("6cam"), "HPE quality.6cam"
            )
            hpe_bound = acceptance_module._require_real_number(
                stream_quality.get("error_bound_ms"),
                "HPE quality.6cam.error_bound_ms",
            )
            status = stream_quality.get("status")
            if status not in {"ok", "degraded"}:
                _fail("HPE 6cam status must be ok or degraded")
            combined = acceptance_module.TRANSCRIPT_SIDE_ERROR_BOUND_MS + hpe_bound
            timing = {
            "status": "confirmed",
            "method": "replayed Transcript offset plus HPE D3 provenance",
            "formula": acceptance_module.SHARED_TIME_FORMULA,
            "transcript_offset_sec": acceptance_module._canonical_decimal(
                transcript_offset
            ),
            "timecode_quantization_rule": (
                acceptance_module.TIMECODE_QUANTIZATION_RULE
            ),
            "timecode_quantization_error_bound_ms": acceptance_module._canonical_decimal(
                acceptance_module.TIMECODE_QUANTIZATION_ERROR_BOUND_MS
            ),
            "offset_lag_quantization_rule": (
                acceptance_module.OFFSET_LAG_QUANTIZATION_RULE
            ),
            "offset_lag_hop_samples": acceptance_module.OFFSET_RMS_HOP_SAMPLES,
            "offset_lag_processing_rate_hz": processing_rate,
            "offset_lag_quantization_error_bound_ms": acceptance_module._canonical_decimal(
                acceptance_module.OFFSET_LAG_QUANTIZATION_ERROR_BOUND_MS
            ),
            "transcript_side_error_bound_ms": acceptance_module._canonical_decimal(
                acceptance_module.TRANSCRIPT_SIDE_ERROR_BOUND_MS
            ),
            "hpe_stream": "6cam",
            "hpe_offset_sec": acceptance_module._canonical_decimal(hpe_offset),
            "hpe_error_bound_ms": acceptance_module._canonical_decimal(hpe_bound),
            "hpe_quality_status": status,
            "combined_error_bound_ms": acceptance_module._canonical_decimal(combined),
            "uncertainty_combination": "conservative_sum",
            "uncertainty_accepted": True,
        }
            # Reuse the consuming validator before any bytes are publishable.
            acceptance_module._validate_hpe_timing(
                hpe_offset_value,
                hpe_quality_value,
                timing,
                transcript_offset_sec=transcript_offset,
                transcript_processing_sr=processing_rate,
            )
            return timing
    except (
        acceptance_module.TranscriptAcceptanceError,
        fs_guard.TranscriptPathGuardError,
        KeyError,
        OSError,
    ) as exc:
        raise TranscriptBridgeError(f"timing replay failed: {exc}") from exc


def _acceptance_manifest_from_final(
    final: dict[str, Any],
    *,
    authority: review_module.ReviewSourceAuthority,
) -> dict[str, Any]:
    """Derive acceptance-v3 from one validated Transcript review-v3 final."""

    scene = final["scene"]
    paths = acceptance_module._canonical_source_paths(
        scene, _acceptance_authority(authority)
    )
    sources = final["sources"]
    manifest = {
        "artifact_kind": acceptance_module.ARTIFACT_KIND,
        "format_version": acceptance_module.FORMAT_VERSION,
        "scene": scene,
        "output_role": acceptance_module.OUTPUT_ROLE,
        "counted_gold_ready": False,
        "review_status": "confirmed",
        "unresolved_issues": [],
        "sources": {
            "timeline": _without_media_type(sources["timeline"], "sources.timeline"),
            "offset": _without_media_type(sources["offset"], "sources.offset"),
            "reference_wav": _without_media_type(
                sources["reference_wav"], "sources.reference_wav"
            ),
            "hpe_offset": _minimal_file_binding(paths["hpe_offset"]),
            "hpe_offset_quality": _minimal_file_binding(paths["hpe_offset_quality"]),
            "gold_alignment": _without_media_type(
                sources["gold_alignment"], "sources.gold_alignment"
            ),
        },
        "source_normalizations": final["source_normalizations"],
        "timecode_contract": {
            "fps_numerator": acceptance_module.TIMECODE_FPS_NUMERATOR,
            "fps_denominator": acceptance_module.TIMECODE_FPS_DENOMINATOR,
            "drop_frame": False,
            "frame_index_origin": 0,
        },
        "repairs": final["repairs"],
        "manual_review": final["manual_review"],
        "timing": _timing_from_sources(final, authority=authority),
        "scope": final["scope"],
        "speaker_prevalence": final["speaker_prevalence"],
        "excluded_event_inventory": final["excluded_event_inventory"],
    }
    return json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))


def _raster_bytes(
    accepted: acceptance_module.AcceptedTranscriptReference,
) -> tuple[bytes, int]:
    ticks = range(accepted.authorized_start_tick, accepted.authorized_end_tick)
    timestamps = [Decimal(tick) / acceptance_module.OUTPUT_RATE_HZ for tick in ticks]
    raster = acceptance_module.rasterize_transcript_activity_reference(
        accepted, timestamps
    )
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(ACTIVITY_COLUMNS)
    for row_index, tick in enumerate(
        range(accepted.authorized_start_tick, accepted.authorized_end_tick)
    ):
        writer.writerow(
            [
                tick,
                acceptance_module._canonical_decimal(
                    Decimal(tick) / acceptance_module.OUTPUT_RATE_HZ
                ),
                *[
                    int(raster.values[speaker][row_index])
                    for speaker in acceptance_module.SPEAKERS
                ],
            ]
        )
    return output.getvalue().encode("utf-8"), len(timestamps)


def _audit_from_final(final: dict[str, Any]) -> dict[str, Any]:
    scope = final["scope"]
    source_count = scope["source_event_count"]
    in_scope_count = scope["in_scope_pre_review_event_count"]
    excluded_count = scope["excluded_event_count"]
    accepted_count = scope["accepted_post_review_event_count"]
    conserved = (
        scope.get("source_count_conserved") is True
        and source_count == in_scope_count + excluded_count
        and accepted_count == in_scope_count
        and final["manual_review"]["event_summary"]["n_events"] == accepted_count
    )
    if not conserved:
        _fail("final review event/source/excluded counts are not conserved")
    return {
        "accepted_events_source": "confirmed_final_review_post_review_repairs_only",
        "accepted_events_derive_only_from_final_post_review": True,
        "source_event_count": source_count,
        "in_scope_pre_review_event_count": in_scope_count,
        "accepted_post_review_event_count": accepted_count,
        "excluded_event_count": excluded_count,
        "crossing_event_count": scope["crossing_event_count"],
        "source_count_conserved": True,
        "authorized_tick_range": {
            "frame_rate_hz": acceptance_module.OUTPUT_RATE_HZ,
            "start_inclusive": scope["authorized_start_tick_inclusive"],
            "end_exclusive": scope["authorized_end_tick_exclusive"],
            "n_ticks": (
                scope["authorized_end_tick_exclusive"]
                - scope["authorized_start_tick_inclusive"]
            ),
        },
        "repair_set_sha256": final["manual_review"]["repair_set_sha256"],
        "semantic_candidate_decisions_sha256": review_module._canonical_json_sha256(
            final["semantic_candidate_decisions"]
        ),
        "event_reviews_sha256": review_module._canonical_json_sha256(
            final["event_reviews"]
        ),
        "final_scene_flags_sha256": review_module._canonical_json_sha256(
            final["scene_flags"]
        ),
    }


def _build_payloads(
    final_path: Path,
    *,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    formal_snapshot: review_module.CanonicalFinalSnapshot | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    try:
        guarded_final = fs_guard.guard_regular_file_path(final_path)
        with fs_guard.guarded_read(guarded_final) as held_final:
            raw_final = fs_guard.read_guarded_bytes(held_final)
            if formal_snapshot is not None and (
                guarded_final != formal_snapshot.path
                or raw_final != formal_snapshot.raw
            ):
                _fail("actual bridge input differs from the held canonical final snapshot")
            untrusted_final = review_module._require_object(
                review_module._strict_json_loads(raw_final, "final review preflight"),
                "final review preflight",
            )
            with _held_binding_graph(untrusted_final) as source_bytes:
                try:
                    final = review_module._load_final_review_with_authority(
                        guarded_final,
                        authority=authority,
                        plan=plan,
                        formal_scene=formal_scene,
                        formal_snapshot=formal_snapshot,
                    )
                except (review_module.TranscriptReviewError, OSError) as exc:
                    raise TranscriptBridgeError(
                        f"final review replay failed: {exc}"
                    ) from exc
                if final["format_version"] != review_module.FORMAT_VERSION:
                    _fail("only the current transcript-review final version is bridgeable")
                if final["review_status"] != "confirmed":
                    _fail("only a confirmed final review is bridgeable")

                acceptance = _acceptance_manifest_from_final(final, authority=authority)
                acceptance_raw = _canonical_json_bytes(acceptance)

                # Keep all acceptance sources held while its real v3 loader
                # replays the exact temporary bytes and builds the raster.
                temporary_path: Path | None = None
                try:
                    with _held_binding_graph(acceptance):
                        with tempfile.NamedTemporaryFile(
                            mode="wb",
                            prefix="vap-transcript-acceptance-",
                            suffix=".json",
                            delete=False,
                        ) as handle:
                            temporary_path = Path(handle.name)
                            handle.write(acceptance_raw)
                            handle.flush()
                            os.fsync(handle.fileno())
                        accepted = (
                            acceptance_module._load_transcript_acceptance_with_authority(
                                temporary_path,
                                authority=_acceptance_authority(authority),
                            )
                        )
                        activity_raw, n_rows = _raster_bytes(accepted)
                except (acceptance_module.TranscriptAcceptanceError, OSError) as exc:
                    raise TranscriptBridgeError(
                        f"derived acceptance replay failed: {exc}"
                    ) from exc
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

                draft_binding = final["draft"]
                draft_path, _size, _digest, _media_type = _binding_path_and_expected(
                    draft_binding, "final.draft"
                )
                raw_draft = source_bytes.get(draft_path)
                if raw_draft is None:
                    _fail("the final's bound draft was not held during replay")
                if formal_snapshot is not None and (
                    draft_path != formal_snapshot.draft.path
                    or raw_draft != formal_snapshot.draft.raw
                ):
                    _fail("bridge replay left the held canonical draft snapshot")
                receipt = {
                    "artifact_kind": BUNDLE_ARTIFACT_KIND,
                    "format_version": BUNDLE_FORMAT_VERSION,
                    "scene": final["scene"],
                    "output_role": acceptance_module.OUTPUT_ROLE,
                    "counted_gold_ready": False,
                    "review_status": "confirmed",
                    "trust_model": TRUST_MODEL,
                    "final_review": _file_binding(guarded_final, raw_final),
                    "draft_review": _file_binding(draft_path, raw_draft),
                    "acceptance": _local_binding(
                        ACCEPTANCE_FILENAME,
                        acceptance_raw,
                        media_type="application/json",
                    ),
                    "coarse_activity": _local_binding(
                        ACTIVITY_FILENAME,
                        activity_raw,
                        media_type="text/csv",
                        n_rows=n_rows,
                    ),
                    "review_audit": _audit_from_final(final),
                }
                if not review_module._type_exact_json_equal(
                    receipt["draft_review"], draft_binding
                ):
                    _fail("final review draft binding changed or is not exact")
                receipt_raw = _canonical_json_bytes(receipt)
                payloads = {
                    ACCEPTANCE_FILENAME: acceptance_raw,
                    ACTIVITY_FILENAME: activity_raw,
                    RECEIPT_FILENAME: receipt_raw,
                }
                replayed = review_module._load_final_review_with_authority(
                    guarded_final,
                    authority=authority,
                    plan=plan,
                    formal_scene=formal_scene,
                    formal_snapshot=formal_snapshot,
                )
                if not review_module._type_exact_json_equal(replayed, final):
                    _fail("final review or live sources changed before bridge publication")
                fs_guard.verify_guarded_read(held_final)
                return payloads, final
    except TranscriptBridgeError:
        raise
    except (
        acceptance_module.TranscriptAcceptanceError,
        fs_guard.TranscriptPathGuardError,
        review_module.TranscriptReviewError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        raise TranscriptBridgeError(f"final review replay failed: {exc}") from exc


def _open_local_binding(
    stack: ExitStack,
    bundle: Path,
    value: Any,
    *,
    expected_name: str,
    media_type: str,
    activity: bool,
) -> tuple[Path, bytes]:
    binding = review_module._require_object(value, expected_name)
    expected_keys = {"path", "size_bytes", "sha256", "media_type"}
    if activity:
        expected_keys |= {"n_rows", "columns"}
    review_module._require_exact_keys(binding, expected_keys, expected_name)
    if binding["path"] != expected_name or binding["media_type"] != media_type:
        _fail(f"{expected_name} binding is not bundle-local and canonical")
    path = bundle / expected_name
    held = stack.enter_context(fs_guard.guarded_read(path))
    raw = fs_guard.read_guarded_bytes(held)
    if (
        type(binding["size_bytes"]) is not int
        or binding["size_bytes"] != len(raw)
        or binding["sha256"] != _sha256(raw)
    ):
        _fail(f"{expected_name} bytes do not match the receipt")
    return held.path, raw


def _load_bundle_with_authority(
    bundle_path: str | Path,
    *,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    require_canonical_bundle_path: bool = True,
    formal_bundle_snapshot: _CanonicalBundleSnapshot | None = None,
    formal_final_snapshot: review_module.CanonicalFinalSnapshot | None = None,
) -> TranscriptBridgeBundle:
    try:
        bundle, opening_identity = fs_guard.guard_directory(bundle_path)
        if (
            formal_scene is not None
            and require_canonical_bundle_path
            and bundle != canonical_bridge_bundle_path(formal_scene)
        ):
            _fail("actual replayed bridge directory is outside its canonical scene path")
        if {item.name for item in bundle.iterdir()} != set(_PAYLOAD_FILENAMES):
            _fail("bridge bundle has missing or undeclared files")
        with ExitStack() as stack:
            receipt_path = bundle / RECEIPT_FILENAME
            held_receipt = stack.enter_context(fs_guard.guarded_read(receipt_path))
            raw_receipt = fs_guard.read_guarded_bytes(held_receipt)
            if formal_bundle_snapshot is not None and (
                bundle != formal_bundle_snapshot.path
                or raw_receipt != formal_bundle_snapshot.receipt_raw
            ):
                _fail("actual bundle replay differs from the held canonical receipt")
            receipt = review_module._require_object(
                review_module._strict_json_loads(raw_receipt, "bridge receipt"),
                "bridge receipt",
            )
            review_module._require_exact_keys(
                receipt,
                {
                    "artifact_kind",
                    "format_version",
                    "scene",
                    "output_role",
                    "counted_gold_ready",
                    "review_status",
                    "trust_model",
                    "final_review",
                    "draft_review",
                    "acceptance",
                    "coarse_activity",
                    "review_audit",
                },
                "bridge receipt",
            )
            if (
                receipt["artifact_kind"] != BUNDLE_ARTIFACT_KIND
            ):
                _fail("bridge receipt artifact_kind is unsupported")
            _require_current_bundle_format(receipt["format_version"])
            if formal_scene is not None and receipt["scene"] != formal_scene:
                _fail("actual bridge receipt scene changed after canonical preflight")
            if (
                receipt["output_role"] != acceptance_module.OUTPUT_ROLE
                or receipt["counted_gold_ready"] is not False
                or receipt["review_status"] != "confirmed"
                or receipt["trust_model"] != TRUST_MODEL
            ):
                _fail("bridge receipt role/readiness/status/trust contract is invalid")

            final_binding = review_module._require_object(
                receipt["final_review"], "final_review"
            )
            review_module._require_exact_keys(
                final_binding,
                {"path", "size_bytes", "sha256", "media_type"},
                "final_review",
            )
            final_path, final_size, final_digest, final_media_type = (
                _binding_path_and_expected(final_binding, "final_review")
            )
            if final_media_type != "application/json":
                _fail("final_review.media_type must be application/json")
            held_final = stack.enter_context(fs_guard.guarded_read(final_path))
            raw_final = fs_guard.read_guarded_bytes(held_final)
            if len(raw_final) != final_size or _sha256(raw_final) != final_digest:
                _fail("final_review bytes changed or disappeared")
            untrusted_final = review_module._require_object(
                review_module._strict_json_loads(raw_final, "bound final review"),
                "bound final review",
            )
            source_bytes = stack.enter_context(_held_binding_graph(untrusted_final))
            final = review_module._load_final_review_with_authority(
                final_path,
                authority=authority,
                plan=plan,
                formal_scene=formal_scene,
                formal_snapshot=(
                    formal_bundle_snapshot.final
                    if formal_bundle_snapshot is not None
                    else formal_final_snapshot
                ),
            )
            if receipt["scene"] != final["scene"]:
                _fail("bridge receipt scene differs from its final review")
            if not review_module._type_exact_json_equal(
                receipt["draft_review"], final["draft"]
            ):
                _fail("bridge receipt does not retain the final review's exact draft binding")
            draft_path, _draft_size, _draft_digest, _draft_media_type = (
                _binding_path_and_expected(final["draft"], "draft_review")
            )
            if draft_path not in source_bytes:
                _fail("the bound draft was not held during bundle replay")

            acceptance_path, raw_acceptance = _open_local_binding(
                stack,
                bundle,
                receipt["acceptance"],
                expected_name=ACCEPTANCE_FILENAME,
                media_type="application/json",
                activity=False,
            )
            _activity_path, raw_activity = _open_local_binding(
                stack,
                bundle,
                receipt["coarse_activity"],
                expected_name=ACTIVITY_FILENAME,
                media_type="text/csv",
                activity=True,
            )
            accepted = acceptance_module._load_transcript_acceptance_with_authority(
                acceptance_path, authority=_acceptance_authority(authority)
            )
            expected_manifest = _acceptance_manifest_from_final(final, authority=authority)
            actual_manifest = review_module._strict_json_loads(
                raw_acceptance, "bridged acceptance"
            )
            if not review_module._type_exact_json_equal(actual_manifest, expected_manifest):
                _fail("acceptance does not deterministically replay from the final review")
            expected_activity, n_rows = _raster_bytes(accepted)
            if raw_activity != expected_activity:
                _fail("coarse activity CSV does not replay from acceptance v3")
            if receipt["coarse_activity"]["n_rows"] != n_rows or receipt[
                "coarse_activity"
            ]["columns"] != list(ACTIVITY_COLUMNS):
                _fail("coarse activity row/column metadata is inconsistent")
            if not review_module._type_exact_json_equal(
                receipt["review_audit"], _audit_from_final(final)
            ):
                _fail("bridge review audit does not replay from the final review")

            fs_guard.verify_guarded_read(held_receipt)
            fs_guard.verify_guarded_read(held_final)
            if fs_guard.directory_identity(bundle) != opening_identity:
                _fail("bridge bundle directory changed during replay")
            return TranscriptBridgeBundle(bundle, final["scene"], final, accepted, receipt)
    except TranscriptBridgeError:
        raise
    except (
        acceptance_module.TranscriptAcceptanceError,
        fs_guard.TranscriptPathGuardError,
        review_module.TranscriptReviewError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        raise TranscriptBridgeError(f"bridge bundle replay failed: {exc}") from exc


@contextmanager
def _held_canonical_bundle(
    bundle_path: str | Path,
) -> Iterator[_CanonicalBundleSnapshot]:
    """Hold the canonical receipt and its actual final/draft through replay."""

    try:
        bundle, directory_identity = fs_guard.guard_directory(bundle_path)
        with fs_guard.guarded_read(bundle / RECEIPT_FILENAME) as held_receipt:
            receipt_raw = fs_guard.read_guarded_bytes(held_receipt)
            receipt = review_module._require_object(
                review_module._strict_json_loads(
                    receipt_raw,
                    "formal bridge receipt preflight",
                ),
                "formal bridge receipt preflight",
            )
            if receipt.get("artifact_kind") != BUNDLE_ARTIFACT_KIND:
                _fail("formal bridge receipt artifact_kind is unsupported")
            _require_current_bundle_format(receipt.get("format_version"))
            scene = review_module._require_string(receipt.get("scene"), "scene")
            review_module._scene_parts(scene)
            _require_canonical_bundle_path(bundle, scene=scene)
            final_binding = review_module._require_object(
                receipt.get("final_review"), "final_review"
            )
            review_module._require_exact_keys(
                final_binding,
                {"path", "size_bytes", "sha256", "media_type"},
                "final_review",
            )
            final_path, final_size, final_digest, final_media = (
                _binding_path_and_expected(final_binding, "final_review")
            )
            if final_media != "application/json":
                _fail("formal bridge final_review.media_type must be application/json")
            with review_module.held_canonical_final(final_path) as held_final:
                if held_final.scene != scene:
                    _fail("formal bridge receipt scene differs from its canonical final")
                if (
                    len(held_final.raw) != final_size
                    or _sha256(held_final.raw) != final_digest
                ):
                    _fail("formal bridge receipt does not bind its held canonical final")
                if not review_module._type_exact_json_equal(
                    receipt.get("draft_review"), held_final.value.get("draft")
                ):
                    _fail("formal bridge receipt does not bind the held canonical draft")
                yield _CanonicalBundleSnapshot(
                    bundle, scene, receipt_raw, receipt, held_final
                )
                fs_guard.verify_guarded_read(held_receipt)
                if fs_guard.directory_identity(bundle) != directory_identity:
                    _fail("formal bridge directory changed during complete replay")
    except TranscriptBridgeError:
        raise
    except (
        fs_guard.TranscriptPathGuardError,
        review_module.TranscriptReviewError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        raise TranscriptBridgeError(f"formal bridge authority failed: {exc}") from exc


def load_transcript_bridge_bundle(
    bundle_path: str | Path,
) -> TranscriptBridgeBundle:
    """Fully replay a production bundle against held canonical authority."""

    try:
        with _held_canonical_bundle(bundle_path) as snapshot:
            return _load_bundle_with_authority(
                snapshot.path,
                authority=review_module.default_source_authority(),
                formal_scene=snapshot.scene,
                formal_bundle_snapshot=snapshot,
            )
    except TranscriptBridgeError:
        raise
    except (
        acceptance_module.TranscriptAcceptanceError,
        fs_guard.TranscriptPathGuardError,
        review_module.TranscriptReviewError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        raise TranscriptBridgeError(f"bridge bundle replay failed: {exc}") from exc


@dataclass(frozen=True)
class _PublicationOwnership:
    token: str
    stage_path: Path
    directory_identity: tuple[int, int]


def _leaf_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An uninspectable leaf is not safely absent.
        return True
    return True


def _owned_directory_matches(
    path: Path,
    ownership: _PublicationOwnership,
    payloads: dict[str, bytes],
    *,
    allow_partial: bool,
) -> bool:
    """Conservatively prove physical identity and every extant payload byte."""

    try:
        guarded, identity = fs_guard.guard_directory(path)
        if identity != ownership.directory_identity:
            return False
        if guarded == ownership.stage_path and not guarded.name.endswith(
            f".{ownership.token}"
        ):
            return False
        names = {item.name for item in guarded.iterdir()}
        expected_names = set(payloads)
        if allow_partial:
            if not names <= expected_names:
                return False
        elif names != expected_names:
            return False
        with ExitStack() as stack:
            for name in names:
                held = stack.enter_context(fs_guard.guarded_read(guarded / name))
                if fs_guard.read_guarded_bytes(held) != payloads[name]:
                    return False
            if fs_guard.directory_identity(guarded) != ownership.directory_identity:
                return False
        return True
    except BaseException:
        return False


def _restore_uncertain_quarantine(quarantine: Path, original: Path) -> None:
    """Best-effort preservation only; never delete an uncertain directory."""

    try:
        fs_guard.guard_directory(quarantine.parent)
        if not _leaf_exists_no_follow(original):
            os.replace(quarantine, original)
    except BaseException:
        # Leaving the unique quarantine in place preserves the foreign/uncertain
        # bytes for inspection and is safer than guessing ownership.
        return


def _remove_owned_directory(
    path: Path,
    ownership: _PublicationOwnership,
    payloads: dict[str, bytes],
    *,
    allow_partial: bool,
) -> bool:
    """Quarantine then delete only the exact physical directory we created."""

    path = fs_guard.lexical_absolute(path)
    if not _owned_directory_matches(
        path, ownership, payloads, allow_partial=allow_partial
    ):
        return False
    quarantine = path.with_name(
        f".{path.name}.rollback.{ownership.token}.{secrets.token_hex(16)}"
    )
    try:
        fs_guard.guard_output_path(quarantine)
        if _leaf_exists_no_follow(quarantine):
            return False
        # The unique rename selects one directory atomically.  We re-check its
        # physical identity and all bytes *after* that selection, so a same-name
        # replacement is preserved rather than recursively deleted.
        os.replace(path, quarantine)
    except BaseException:
        return False
    try:
        if not _owned_directory_matches(
            quarantine, ownership, payloads, allow_partial=allow_partial
        ):
            _restore_uncertain_quarantine(quarantine, path)
            return False
        shutil.rmtree(quarantine)
        return True
    except BaseException:
        # Partial cleanup is still isolated under the invocation-unique
        # quarantine name; never broaden deletion after an error.
        return False


def _publish_bundle_with_authority(
    final_path: str | Path,
    output_dir: str | Path,
    *,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    formal_snapshot: review_module.CanonicalFinalSnapshot | None = None,
) -> Path:
    try:
        destination = fs_guard.guard_output_path(output_dir)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptBridgeError(f"bridge output path is unsafe: {exc}") from exc
    corpus_root = fs_guard.lexical_absolute(authority.corpus_root)
    hpe_root = fs_guard.lexical_absolute(authority.hpe_delivery_root)
    if destination.is_relative_to(corpus_root) or destination.is_relative_to(hpe_root):
        _fail("bridge bundles may not be written inside corpus or HPE source roots")
    if _leaf_exists_no_follow(destination):
        _fail(f"refusing to overwrite existing bundle: {destination}")
    if formal_scene is not None and destination != canonical_bridge_bundle_path(
        formal_scene
    ):
        _fail("formal bridge publication destination is not scene-canonical")
    parent, parent_identity = fs_guard.guard_directory(destination.parent)
    final_source = fs_guard.lexical_absolute(final_path)
    payloads, _final = _build_payloads(
        final_source,
        authority=authority,
        plan=plan,
        formal_scene=formal_scene,
        formal_snapshot=formal_snapshot,
    )
    token = secrets.token_hex(16)
    staging = parent / f".{destination.name}.staging.{token}"
    try:
        staging.mkdir(exist_ok=False)
        ownership = _PublicationOwnership(
            token=token,
            stage_path=staging,
            directory_identity=fs_guard.directory_identity(staging),
        )
    except (FileExistsError, fs_guard.TranscriptPathGuardError, OSError) as exc:
        raise TranscriptBridgeError(
            "could not create an exclusively owned staging directory"
        ) from exc
    try:
        for name in _PAYLOAD_FILENAMES:
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        _load_bundle_with_authority(
            staging,
            authority=authority,
            plan=plan,
            formal_scene=formal_scene,
            require_canonical_bundle_path=False,
            formal_final_snapshot=formal_snapshot,
        )
        replayed_payloads, _replayed_final = _build_payloads(
            final_source,
            authority=authority,
            plan=plan,
            formal_scene=formal_scene,
            formal_snapshot=formal_snapshot,
        )
        if replayed_payloads != payloads:
            _fail("review or live sources changed before bridge publication")
        if fs_guard.directory_identity(parent) != parent_identity:
            _fail("bridge output parent changed during publication")
        if _leaf_exists_no_follow(destination):
            _fail(f"refusing to overwrite existing bundle: {destination}")
        os.rename(staging, destination)
        if fs_guard.directory_identity(destination) != ownership.directory_identity:
            _fail("published bridge path does not name the owned staging directory")
        _load_bundle_with_authority(
            destination,
            authority=authority,
            plan=plan,
            formal_scene=formal_scene,
            formal_final_snapshot=formal_snapshot,
        )
        return destination
    except FileExistsError as exc:
        raise TranscriptBridgeError(
            f"refusing to overwrite existing bundle: {destination}"
        ) from exc
    except BaseException:
        _remove_owned_directory(
            destination, ownership, payloads, allow_partial=False
        )
        raise
    finally:
        # A foreign caller may reuse the old staging name after rename.  The
        # physical-identity check prevents this invocation from removing it.
        _remove_owned_directory(staging, ownership, payloads, allow_partial=True)


def publish_transcript_bridge_bundle(
    final_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Publish one no-clobber final→acceptance→raster production bundle."""

    try:
        with review_module.held_canonical_final(final_path) as snapshot:
            _ensure_formal_bridge_root()
            destination = _require_canonical_bundle_path(
                output_dir
                if output_dir is not None
                else canonical_bridge_bundle_path(snapshot.scene),
                scene=snapshot.scene,
            )
            return _publish_bundle_with_authority(
                snapshot.path,
                destination,
                authority=review_module.default_source_authority(),
                formal_scene=snapshot.scene,
                formal_snapshot=snapshot,
            )
    except TranscriptBridgeError:
        raise
    except (
        acceptance_module.TranscriptAcceptanceError,
        fs_guard.TranscriptPathGuardError,
        review_module.TranscriptReviewError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        raise TranscriptBridgeError(f"bridge publication failed: {exc}") from exc
