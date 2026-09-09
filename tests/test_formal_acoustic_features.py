from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hri_turn_taking.data import assembly
from hri_turn_taking.data import formal_acoustic_features as formal
from hri_turn_taking.data import real_acoustic_features as real_acoustics

SCENE = "03_20__4"


def _fixture_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> formal.FormalAcousticAuthority:
    helper_path = Path(__file__).with_name("test_real_acoustic_features.py")
    helpers = runpy.run_path(str(helper_path), run_name="formal_acoustic_fixture_helpers")
    real_authority, _, _ = helpers["_install_derive_harness"](tmp_path, monkeypatch)
    dataset = helpers["_dataset"](tmp_path)
    sidecar = dataset.path.with_suffix(".provenance.json")
    sidecar.write_text('{"fixture":"sealed-analysis-sidecar"}', encoding="utf-8")
    dataset = replace(
        dataset,
        data_sha256=hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
        provenance_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        real_acoustics,
        "_load_analysis_dataset",
        lambda scene, *, authority: dataset,
    )
    output = tmp_path / "formal"
    return formal.FormalAcousticAuthority(
        real=real_authority,
        output_root=output,
        review_root=output,
        enforce_frozen_inventory=False,
    )


def _receipt(pending: formal.PendingAcousticArtifact) -> dict[str, Any]:
    def binding(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    return {
        "artifact_kind": "vap_real_causal_acoustic_review_receipt",
        "format_version": 1,
        "scene": pending.scene,
        "decision": "accepted",
        "independent_review": True,
        "reviewer_id": "independent-reviewer-fixture",
        "reviewed_at_utc": "2026-08-16T12:00:00Z",
        "review_scope": [
            "canonical_source_timeline_and_mix_binding",
            "causal_extractor_and_threshold_contract",
            "artifact_type_hash_and_replay_gates",
            "assembly_v3_manifest_conformance",
            "trusted_local_single_user_single_writer_boundary",
        ],
        "reviewed_bundle": {
            "source_csv": binding(pending.paths.source_csv),
            "evidence_json": binding(pending.paths.evidence_json),
            "feature_csv": binding(pending.paths.feature_csv),
            "pending_manifest": binding(pending.paths.pending_manifest),
        },
        "notes": "Fixture review of the artifact contract; not a real-data approval.",
    }


def _write_receipt(
    pending: formal.PendingAcousticArtifact,
    *,
    update: Any | None = None,
) -> None:
    receipt = _receipt(pending)
    if update is not None:
        update(receipt)
    pending.paths.review_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def test_pending_builder_is_canonical_strict_and_assembly_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)

    pending = formal.build_pending_acoustic_scene(SCENE, authority=authority)

    assert pending.scene == SCENE
    assert len(pending.feature_df) == 3
    assert pending.paths.source_csv.is_file()
    assert pending.paths.evidence_json.is_file()
    assert pending.paths.feature_csv.is_file()
    assert pending.paths.pending_manifest.is_file()
    assert not pending.paths.feature_manifest.exists()
    assert not pending.paths.review_receipt.exists()
    assert pending.pending_manifest["artifact_kind"] == formal.PENDING_MANIFEST_KIND
    assert pending.pending_manifest["artifact_kind"] != assembly.FEATURE_MANIFEST_KIND
    assert pending.pending_manifest["review_status"] == "review_pending"
    assert pending.pending_manifest["assembly_eligible"] is False
    assert pending.pending_manifest["authority_scope"] == (
        formal.NON_COUNTED_AUTHORITY_SCOPE
    )
    assert pending.evidence["owner_parameter_decision"]["value"] == 0.45
    assert pending.evidence["publication_trust_boundary"]["model"] == (
        "trusted_local_single_user_single_writer_workspace"
    )
    assert pending.evidence["source_selection"][
        "gold_channel_time_corrections_applied"
    ] is False
    with pytest.raises(assembly.AssemblyGateError, match="has no sibling"):
        assembly._validate_feature_manifest(
            pending.paths.feature_csv,
            list(formal.FEATURE_COLUMNS),
            pending.source_df.loc[:, list(assembly.KEY_COLUMNS)],
            {
                "participant_to_slot": pending.pending_manifest[
                    "participant_to_slot"
                ]
            },
        )


def test_literal_receipt_validates_but_fixture_authority_cannot_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)
    pending = formal.build_pending_acoustic_scene(SCENE, authority=authority)
    _write_receipt(pending)

    receipt, _ = formal._load_review_receipt(pending)
    assert receipt["reviewer_id"] == "independent-reviewer-fixture"
    assert all(
        set(binding) == {"path", "sha256", "size_bytes"}
        for binding in receipt["reviewed_bundle"].values()
    )
    with pytest.raises(
        formal.FormalAcousticFeatureError,
        match="default canonical production authority",
    ):
        formal.seal_reviewed_acoustic_scene(SCENE, authority=authority)
    assert not pending.paths.feature_manifest.exists()
    assert pending.pending_manifest["assembly_eligible"] is False
    assert pending.pending_manifest["counted_input_ready"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", True, "identity/decision"),
        ("independent_review", 1, "identity/decision"),
        ("decision", "pending", "identity/decision"),
        ("reviewed_at_utc", "2026-08-16", "reviewed_at_utc"),
    ],
)
def test_review_receipt_rejects_bool_int_or_unaccepted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)
    pending = formal.build_pending_acoustic_scene(SCENE, authority=authority)
    _write_receipt(pending, update=lambda receipt: receipt.__setitem__(field, value))

    with pytest.raises(formal.FormalAcousticFeatureError, match=message):
        formal._load_review_receipt(pending)
    assert not pending.paths.feature_manifest.exists()


def test_pending_loader_rejects_hash_tamper_nan_and_stale_master(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)
    pending = formal.build_pending_acoustic_scene(SCENE, authority=authority)

    evidence = json.loads(pending.paths.evidence_json.read_text(encoding="utf-8"))
    evidence["owner_parameter_decision"]["value"] = 0.46
    pending.paths.evidence_json.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(
        formal.FormalAcousticFeatureError,
        match="owner-frozen 0.45",
    ):
        formal.load_pending_acoustic_scene(SCENE, authority=authority)

    # Rebuild in an isolated root for a malformed numeric CSV.
    second_root = tmp_path / "second"
    second = replace(
        authority,
        output_root=second_root,
        review_root=second_root,
    )
    second_pending = formal.build_pending_acoustic_scene(SCENE, authority=second)
    payload = second_pending.paths.source_csv.read_bytes().replace(
        b",-13.815510557964274,",
        b",NaN,",
        1,
    )
    if payload == second_pending.paths.source_csv.read_bytes():
        payload = payload.replace(b",0.0,0.0,0.0", b",NaN,0.0,0.0", 1)
    second_pending.paths.source_csv.write_bytes(payload)
    with pytest.raises(
        formal.FormalAcousticFeatureError,
        match="could not be parsed|canonical CSV|NaN|non-finite",
    ):
        formal.load_pending_acoustic_scene(SCENE, authority=second)

    third_root = tmp_path / "third"
    third = replace(
        authority,
        output_root=third_root,
        review_root=third_root,
    )
    third_pending = formal.build_pending_acoustic_scene(SCENE, authority=third)
    master = Path(third_pending.evidence["master_mix_replay"]["path"])
    master.write_bytes(master.read_bytes() + b"stale")
    with pytest.raises(
        formal.FormalAcousticFeatureError,
        match="master WAV hash changed",
    ):
        formal.load_pending_acoustic_scene(SCENE, authority=third)


def test_copied_bundle_cannot_become_authoritative_under_a_new_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)
    pending = formal.build_pending_acoustic_scene(SCENE, authority=authority)
    copied_root = tmp_path / "copied"
    copied_root.mkdir()
    for path in (
        pending.paths.source_csv,
        pending.paths.evidence_json,
        pending.paths.feature_csv,
        pending.paths.pending_manifest,
    ):
        shutil.copy2(path, copied_root / path.name)
    copied_authority = replace(
        authority,
        output_root=copied_root,
        review_root=copied_root,
    )

    with pytest.raises(
        formal.FormalAcousticFeatureError,
        match="canonical path",
    ):
        formal.load_pending_acoustic_scene(SCENE, authority=copied_authority)


def test_pending_no_overwrite_and_unforgeable_default_production_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)
    formal.build_pending_acoustic_scene(SCENE, authority=authority)
    with pytest.raises(formal.FormalAcousticFeatureError, match="overwrite"):
        formal.build_pending_acoustic_scene(SCENE, authority=authority)

    production = formal.default_formal_acoustic_authority()
    assert production is formal.default_formal_acoustic_authority()
    assert production.production_scope == formal.CANONICAL_PRODUCTION_SCOPE
    assert production.enforce_frozen_inventory is True
    assert production.real != real_acoustics.default_real_acoustic_authority()
    expected_root = (
        Path(formal.__file__).resolve().parents[3]
        / "data"
        / "processed"
        / "acoustic_features"
    ).resolve()
    assert production.output_root == expected_root
    assert production.review_root == expected_root
    changed_real = replace(
        production.real,
        analysis=replace(
            production.real.analysis,
            analysis_base_root=tmp_path / "changed-analysis-root",
        ),
    )
    changed_fields = replace(production, real=changed_real)
    assert formal._canonical_production_fields_match(production) is True
    assert formal._canonical_production_fields_match(changed_fields) is False
    assert changed_fields.production_scope == formal.NON_COUNTED_AUTHORITY_SCOPE
    assert replace(production).production_scope == formal.NON_COUNTED_AUTHORITY_SCOPE
    assert authority.production_scope == formal.NON_COUNTED_AUTHORITY_SCOPE
    with pytest.raises(TypeError, match="production_scope"):
        formal.FormalAcousticAuthority(
            real=authority.real,
            output_root=tmp_path / "forged",
            review_root=tmp_path / "forged",
            production_scope=formal.CANONICAL_PRODUCTION_SCOPE,
        )


def test_raw_reaudit_rederives_every_pending_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _fixture_authority(tmp_path, monkeypatch)
    formal.build_pending_acoustic_scene(SCENE, authority=authority)

    result = formal.reaudit_formal_acoustic_scene(
        SCENE,
        authority=authority,
        require_reviewed=False,
    )

    assert type(result) is formal.PendingAcousticArtifact
