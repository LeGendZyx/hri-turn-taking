"""Tests for the real-data intake stage (delivery dir -> wide table)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hri_turn_taking.data.intake as intake_module
from hri_turn_taking.data.assembly import DataSourceGateError, load_vap_dataframe
from hri_turn_taking.data.intake import (
    DeliveryGateError,
    IntakeConfig,
    _accepted_output_grid,
    assert_trainable,
    load_delivery,
    run_intake,
    write_intake,
)
from hri_turn_taking.data.multiparty import validate_multiparty_dataframe

FPS_SRC = 25.0
OFFSET_6CAM = -1.2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _sealed_crosswalk_contract_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supply a small, fully hash-bound sealed contract for synthetic deliveries."""
    inventory_path = tmp_path / "sealed_inventory.json"
    inventory_path.write_text('{"test":"sealed"}', encoding="utf-8")
    artifact_names = [
        "test_intake.py",
        "test_real_data_assembly.py",
        "test_cli_gates.py",
        "test_vap.py",
        "test_predictions.py",
    ]

    def fake_contract(
        scene: str,
        inventory_path_arg: str | Path,
        *,
        repository_root: str | Path,
    ) -> dict:
        del inventory_path_arg
        delivery = tmp_path / scene
        control = json.loads(
            (delivery / ".test_crosswalk.json").read_text(encoding="utf-8")
        )
        world_path = delivery / "world_persons.csv"
        hpe_authority = {
            "path": str((delivery / "mapping.json").resolve()),
            "sha256": _sha256(delivery / "mapping.json"),
        }
        artifact_files = {
            name: {
                "size_bytes": (Path(repository_root) / "tests" / name).stat().st_size,
                "sha256": _sha256(Path(repository_root) / "tests" / name),
            }
            for name in artifact_names
        }
        mapping = control["authority_mapping"]
        return {
            "artifact_kind": "vap_hpe_transcript_identity_downstream_contract",
            "format_version": control.get("contract_format_version", 1),
            "inventory": {
                "path": str(inventory_path),
                "repository_path": "tests/fixtures/sealed_inventory.json",
                "sha256": _sha256(inventory_path),
                "artifact_kind": "vap_hpe_transcript_crosswalk_inventory",
                "format_version": 1,
            },
            "scene": scene,
            "accepted_window": control["accepted_window"],
            "source_fps": control["source_fps"],
            "hpe_logical_timing_authority": {
                "axis": "hpe_delivery_logical_frame_index",
                "source_fps": control["source_fps"],
                "evidence_cameras": [
                    {
                        "camera": "Cam_2",
                        "source_fps": control["source_fps"],
                        "run_metadata": hpe_authority,
                    }
                ],
                "validation_complete": hpe_authority,
            },
            "expected_hpe_persons": sorted(map(int, mapping), key=int),
            "person_mapping": [
                {
                    "hpe_person": int(person),
                    "participant": participant,
                    "slot": "P" + str({"R": 1, "A": 2, "B": 3, "C": 4}[participant]),
                }
                for person, participant in sorted(mapping.items(), key=lambda item: int(item[0]))
            ],
            "artifact_dir": "tests",
            "artifact_files": artifact_files,
            "ledger_snapshot_sha256": "0" * 64,
            "upstream_proposal": control["upstream_proposal"],
            "upstream_mismatch": control["upstream_mismatch"],
            "delivery_world_persons": {
                "path": str(world_path.resolve()),
                "sha256": _sha256(world_path),
            },
            "hpe_validation_authority": hpe_authority,
        }

    monkeypatch.setattr(
        "hri_turn_taking.data.intake.load_sealed_crosswalk_contract", fake_contract
    )


def make_delivery(
    tmp_path: Path,
    *,
    name: str = "g01__s2",
    review_status: str = "confirmed",
    participants: dict[str, str] | None = None,
    streams: dict[str, float] | None = None,
    extra_person_rows: pd.DataFrame | None = None,
    unmapped_persons: list[str] | None = None,
    offset_quality: dict | None = None,
    include_offset_quality: bool = True,
    mapping_extra: dict | None = None,
    geometry_overrides: dict | None = None,
    times_by_person: dict[str, np.ndarray] | None = None,
    abs_yaw_by_person: dict[str, float] | None = None,
    crosswalk_participants: dict[str, str] | None = None,
    accepted_window: tuple[int, int] | None = None,
    source_fps: float = FPS_SRC,
) -> Path:
    """Fabricate a minimal HPE delivery directory.

    Geometry (leveled, +Y up; horizontal plane = x/z): R stands at the origin
    facing +z; person p1 stands at (0, 0, 2) facing straight back at R (-z,
    deviation 0 deg); person p2 stands at (2, 0, 0) facing +z (90 deg off the
    bearing to R).
    """
    directory = tmp_path / name
    directory.mkdir(parents=True)
    participants = participants or {"1": "R", "2": "A", "3": "B"}

    default_times = np.arange(0, 100) / source_fps
    rows = []
    geometry = {
        "1": ((0.0, 0.0), (0.0, 1.0)),
        "2": ((0.0, 2.0), (0.0, -1.0)),
        "3": ((2.0, 0.0), (0.0, 1.0)),
        **(geometry_overrides or {}),
    }
    times_by_person = times_by_person or {}
    abs_yaw_by_person = abs_yaw_by_person or {}
    for person in participants:
        (x, z), (fx, fz) = geometry.get(person, ((1.0, 1.0), (1.0, 0.0)))
        for t in times_by_person.get(person, default_times):
            rows.append(
                {
                    "frame_index": round(t * source_fps),
                    "timestamp_sec": t,
                    "world_person": person,
                    "n_cams": 3,
                    "cams": "2|3|4",
                    "x_w": x,
                    "y_w": 1.6,
                    "z_w": z,
                    "facing_x": fx,
                    "facing_y": 0.0,
                    "facing_z": fz,
                    "facing_src_cam": 2,
                    "facing_src_abs_yaw": abs_yaw_by_person.get(person, 10.0),
                    "person": person,
                    "tracklet": 0,
                }
            )
    world_df = pd.DataFrame(rows)
    if extra_person_rows is not None:
        world_df = pd.concat([world_df, extra_person_rows], ignore_index=True)
    world_df.to_csv(directory / "world_persons.csv", index=False)

    pd.DataFrame(
        {"person": list(participants), "participant": list(participants.values())}
    ).to_csv(directory / "participant_map.csv", index=False)
    mapping: dict = {"review_status": review_status, "participant_map": participants}
    if "__" in name:
        group, scenario = name.split("__", 1)
        mapping.update({"group": group, "scenario": scenario, "delivery": name})
    if unmapped_persons is not None:
        mapping["unmapped_persons"] = unmapped_persons
    mapping.update(mapping_extra or {})
    (directory / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    streams = streams if streams is not None else {"6cam": OFFSET_6CAM}
    (directory / "offset.json").write_text(json.dumps({"streams": streams}), encoding="utf-8")
    if offset_quality is None and include_offset_quality:
        offset_quality = {
            "convention": "shared_time = stream_local_time + offset_s",
            "timebase_keys": {key: "avi_frame_axis" for key in streams},
            "quality": {
                key: {"offset_s": value, "self_check_ok": True, "error_bound_ms": 33,
                      "status": "ok"}
                for key, value in streams.items()
            },
        }
    if offset_quality is not None:
        (directory / "offset_quality.json").write_text(
            json.dumps(offset_quality), encoding="utf-8"
        )
    authority = crosswalk_participants or participants
    if accepted_window is None:
        accepted_window = (
            int(world_df["frame_index"].min()),
            int(world_df["frame_index"].max()) + 1,
        )
    differences = [
        {
            "hpe_person": int(person),
            "reviewed_label": participant,
            "upstream_label": participants.get(person),
        }
        for person, participant in sorted(authority.items(), key=lambda item: int(item[0]))
        if participants.get(person) != participant
    ]
    control = {
        "source_fps": source_fps,
        "authority_mapping": authority,
        "upstream_proposal": participants,
        "upstream_mismatch": {
            "differences": differences,
            "acknowledgement": "test fixture independent review" if differences else "",
        },
        "accepted_window": {
            "scene": name,
            "camera": "Cam_2",
            "local_start_inclusive": accepted_window[0],
            "local_end_exclusive": accepted_window[1],
            "logical_start_inclusive": accepted_window[0],
            "logical_end_exclusive": accepted_window[1],
            "filter_rule": "start <= frame_index < end",
        },
    }
    (directory / ".test_crosswalk.json").write_text(
        json.dumps(control), encoding="utf-8"
    )
    return directory


def test_intake_produces_valid_wide_table(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path)
    result = run_intake(directory)
    frame_df = result.frame_df

    validate_multiparty_dataframe(frame_df)
    assert frame_df["session_id"].unique().tolist() == ["g01__s2"]
    # slot convention: R -> P1, A -> P2, B -> P3; slot 4 padded
    assert frame_df["P1_present"].eq(1).all()
    assert frame_df["P4_present"].eq(0).all()
    # 25 fps grid on the shared axis: local times + (-1.2 s) offset
    timestamps = frame_df["timestamp_s"].to_numpy()
    assert timestamps[0] == pytest.approx(OFFSET_6CAM, abs=1 / 25)
    assert np.allclose(np.diff(timestamps), 1 / 25)


def test_intake_head_yaw_geometry(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path)
    frame_df = run_intake(directory).frame_df
    # A (p1) faces R directly -> 0 deg; B (p2) faces 90 deg off the bearing to R
    assert frame_df["P2_head_yaw_deg"].abs().max() < 1e-6
    assert frame_df["P3_head_yaw_deg"].to_numpy() == pytest.approx(90.0, abs=1e-6)
    # R faces +z; the others' centroid is at (1, 1) from origin -> 45 deg
    assert frame_df["P1_head_yaw_deg"].to_numpy() == pytest.approx(45.0, abs=1e-6)


@pytest.mark.parametrize(
    ("scene", "start", "end", "source_fps", "offset_s", "expected_frames"),
    [
        ("03_20__4", 600, 6600, 30.0, -6.7334, 5000),
        ("03_26__3", 390, 5610, 30.0, -1.5665, 4350),
        ("05_14__2", 1800, 18000, 60.0, -5.9873, 6750),
        ("05_14__3", 1800, 29913, 60.0, -5.0551, 11714),
        ("05_14__4", 1800, 30008, 60.0, -4.8509, 11754),
        ("05_15__2", 900, 8700, 30.0, -3.7621, 6500),
        ("05_15__3", 900, 8060, 30.0, -6.5999, 5966),
        ("05_15__4", 900, 9150, 30.0, -6.7, 6875),
        ("05_15__5", 900, 9900, 30.0, -10.7654, 7500),
        ("05_15__6", 900, 16200, 30.0, -4.864, 12750),
    ],
)
def test_accepted_window_to_25hz_grid_counts(
    scene: str,
    start: int,
    end: int,
    source_fps: float,
    offset_s: float,
    expected_frames: int,
) -> None:
    window = {
        "scene": scene,
        "logical_start_inclusive": start,
        "logical_end_exclusive": end,
    }
    grid, start_tick, end_tick = _accepted_output_grid(
        window,
        source_fps=source_fps,
        output_fps=25,
        offset_s=offset_s,
    )

    assert len(grid) == expected_frames
    assert len(grid) == end_tick - start_tick
    assert np.all(grid >= start / source_fps + offset_s - 1e-12)
    assert np.all(grid < end / source_fps + offset_s)


def test_accepted_grid_does_not_snap_a_post_tick_boundary_backwards() -> None:
    window = {
        "scene": "counterexample",
        "logical_start_inclusive": 0,
        "logical_end_exclusive": 1,
    }
    grid, start_tick, end_tick = _accepted_output_grid(
        window,
        source_fps=25.0,
        output_fps=25,
        offset_s=2e-11,
    )

    assert (start_tick, end_tick) == (1, 2)
    assert grid.tolist() == [0.04]
    assert np.all(grid >= 2e-11)
    assert np.all(grid < 1 / 25 + 2e-11)


def test_sealed_crosswalk_overrides_swapped_hpe_proposal(tmp_path: Path) -> None:
    proposal = {"1": "R", "2": "B", "3": "A"}
    authority = {"1": "R", "2": "A", "3": "B"}
    directory = make_delivery(
        tmp_path,
        name="g31__s1",
        participants=proposal,
        crosswalk_participants=authority,
    )

    result = run_intake(directory)

    assert result.provenance["person_to_participant"] == authority
    assert result.provenance["upstream_person_to_participant_proposal"] == proposal
    assert result.frame_df["P2_head_yaw_deg"].abs().max() < 1e-6
    assert result.frame_df["P3_head_yaw_deg"].to_numpy() == pytest.approx(90.0)


@pytest.mark.parametrize("bad_version", [1.0, True])
def test_sealed_crosswalk_version_requires_exact_integer(
    tmp_path: Path, bad_version: object
) -> None:
    directory = make_delivery(tmp_path, name="g35__s1")
    control_path = directory / ".test_crosswalk.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["contract_format_version"] = bad_version
    control_path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DeliveryGateError, match="Unsupported sealed"):
        run_intake(directory)


def test_05_14_window_is_end_exclusive(tmp_path: Path) -> None:
    start, end = 1800, 29913
    times = np.array([start, end - 1, end], dtype=float) / 60.0
    directory = make_delivery(
        tmp_path,
        name="05_14__3",
        source_fps=60.0,
        streams={"6cam": -5.0551},
        accepted_window=(start, end),
        times_by_person={"1": times, "2": times, "3": times},
    )

    result = run_intake(directory)

    assert len(result.frame_df) == 11714
    assert result.provenance["accepted_window"]["logical_end_exclusive"] == end
    assert result.provenance["rows_outside_accepted_window"] == 3


def test_intake_uses_hpe_timing_authority_when_gaze_camera_is_excluded(
    tmp_path: Path,
) -> None:
    directory = make_delivery(tmp_path, name="05_15__3", source_fps=30.0)
    control_path = directory / ".test_crosswalk.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    # 05_15 identity review uses Gaze Cam_1, but production HPE timing evidence
    # comes from Cam_2. Intake must consume contract.source_fps, never look up
    # accepted_window.camera in the HPE pose-camera inventory.
    control["accepted_window"]["camera"] = "Cam_1"
    control_path.write_text(json.dumps(control), encoding="utf-8")

    result = run_intake(directory)

    assert result.provenance["accepted_window"]["camera"] == "Cam_1"
    assert result.provenance["identity_contract"]["hpe_logical_timing_authority"][
        "evidence_cameras"
    ][0]["camera"] == "Cam_2"


def test_intake_rejects_unconfirmed_hpe_provenance(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path, review_status="proposed")
    with pytest.raises(DeliveryGateError, match="confirmed HPE delivery"):
        run_intake(directory)
    # provenance v2 has no debug bypass configuration
    with pytest.raises(DeliveryGateError, match="no debug"):
        run_intake(directory, IntakeConfig(allow_unconfirmed=True))


def test_intake_rejects_contract_violations(tmp_path: Path) -> None:
    no_r = make_delivery(tmp_path, name="g02__s1", participants={"2": "A", "3": "B"})
    with pytest.raises(DeliveryGateError, match="legal D2"):
        run_intake(no_r)

    # the label set is closed: {R,A,B} or {R,A,B,C} only (Codex R1.1-4)
    sparse = make_delivery(
        tmp_path, name="g12__s1", participants={"1": "R", "3": "C"}
    )
    with pytest.raises(DeliveryGateError, match="legal D2"):
        run_intake(sparse)

    bad_stream = make_delivery(tmp_path, name="g03__s1", streams={"robot_rgb": 0.5})
    with pytest.raises(DeliveryGateError, match="6cam"):
        run_intake(bad_stream)

    bad_name = make_delivery(tmp_path, name="scenario2")
    with pytest.raises(DeliveryGateError, match="group"):
        load_delivery(bad_name)


def test_intake_cannot_substitute_rgb_offset_for_six_camera_timebase(
    tmp_path: Path,
) -> None:
    directory = make_delivery(
        tmp_path,
        name="g26__s1",
        streams={"participants_rgb": 0.5},
    )
    output = tmp_path / "must_not_exist.csv"

    with pytest.raises(DeliveryGateError, match="stream_key must be '6cam'"):
        write_intake(
            directory,
            output,
            IntakeConfig(stream_key="participants_rgb"),
        )

    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_intake_rejects_unmapped_person(tmp_path: Path) -> None:
    stray = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "world_person": "ghost",
                "n_cams": 2,
                "cams": "2|3",
                "x_w": 1.0,
                "y_w": 1.6,
                "z_w": 1.0,
                "facing_x": 1.0,
                "facing_y": 0.0,
                "facing_z": 0.0,
                "facing_src_cam": 2,
                "facing_src_abs_yaw": 5.0,
                "person": "ghost",
                "tracklet": 9,
            }
        ]
    )
    directory = make_delivery(tmp_path, extra_person_rows=stray)
    with pytest.raises(DeliveryGateError, match="absent from both"):
        run_intake(directory)

    # Sealed v2 requires exact final-person coverage inside the accepted window;
    # an upstream unmapped declaration cannot bypass the human crosswalk.
    declared = make_delivery(
        tmp_path, name="g05__s1", extra_person_rows=stray, unmapped_persons=["ghost"]
    )
    with pytest.raises(DeliveryGateError, match="non-integer HPE person"):
        run_intake(declared)


def full_offset_quality(**overrides) -> dict:
    entry = {"offset_s": OFFSET_6CAM, "self_check_ok": True, "error_bound_ms": 33,
             "status": "ok"}
    entry.update({k: v for k, v in overrides.items() if k in entry})
    doc = {
        "convention": "shared_time = stream_local_time + offset_s",
        "timebase_keys": {"6cam": "avi_frame_axis"},
        "quality": {"6cam": entry},
    }
    if "convention" in overrides:
        doc["convention"] = overrides["convention"]
    if overrides.get("drop_timebase_keys"):
        doc.pop("timebase_keys")
    if overrides.get("timebase_list"):
        doc["timebase_keys"] = ["6cam"]
    if overrides.get("timebase_value_only"):
        doc["timebase_keys"] = {"head_pose_csv": "6cam"}
    if overrides.get("drop_field"):
        entry.pop(overrides["drop_field"])
    return doc


def test_intake_offset_quality_assertions(tmp_path: Path) -> None:
    run_intake(make_delivery(tmp_path, name="g06__s1", offset_quality=full_offset_quality()))

    cases = [
        ("g07__s1", full_offset_quality(offset_s=OFFSET_6CAM + 0.5), "disagrees"),
        ("g09__s1", full_offset_quality(self_check_ok=False), "self_check_ok"),
        # every field is REQUIRED (Codex R1.1-2), and the reverse sign
        # convention is a sign error, not an accepted variant
        ("g13__s1", full_offset_quality(drop_field="status"), "missing required"),
        ("g14__s1", full_offset_quality(drop_timebase_keys=True), "timebase_keys"),
        (
            "g15__s1",
            full_offset_quality(convention="stream_local_time = shared_time + offset_s"),
            "convention",
        ),
        # JSON-type strictness (Codex R1.2-2): the string "false" is truthy,
        # numeric strings are not numbers, and a list of stream names is not
        # the 议题 5① timebase_keys object
        ("g17__s1", full_offset_quality(self_check_ok="false"), "boolean"),
        ("g18__s1", full_offset_quality(error_bound_ms="33"), "JSON number"),
        ("g19__s1", full_offset_quality(offset_s=str(OFFSET_6CAM)), "JSON number"),
        ("g20__s1", full_offset_quality(timebase_list=True), "JSON object"),
        ("g21__s1", full_offset_quality(timebase_value_only=True), "as a key"),
    ]
    for name, payload, match in cases:
        directory = make_delivery(tmp_path, name=name, offset_quality=payload)
        with pytest.raises(DeliveryGateError, match=match):
            run_intake(directory)

    # production intake refuses deliveries with no quality report at all (R1 ③)
    absent = make_delivery(tmp_path, name="g08__s1", include_offset_quality=False)
    with pytest.raises(DeliveryGateError, match="offset_quality"):
        run_intake(absent)
    with pytest.raises(DeliveryGateError, match="no debug"):
        run_intake(absent, IntakeConfig(allow_missing_offset_quality=True))


def test_intake_merges_person_fragments(tmp_path: Path) -> None:
    # real HPE maps several sequential tracklets to one participant (Codex R1 ①):
    # p1 covers the first half, p1b (same geometry) the second half of A's series
    directory = make_delivery(
        tmp_path,
        participants={"1": "R", "2": "A", "5": "A", "3": "B"},
        geometry_overrides={"5": ((0.0, 2.0), (0.0, -1.0))},
        times_by_person={
            "2": np.arange(0, 50) / FPS_SRC,
            "5": np.arange(50, 100) / FPS_SRC,
        },
    )
    result = run_intake(directory)
    assert result.provenance["fragments_per_participant"]["A"] == ["2", "5"]
    frame_df = result.frame_df
    assert frame_df["P2_present"].eq(1).all()
    assert frame_df["P2_head_yaw_deg"].abs().max() < 1e-6

    # overlapping fragments that disagree on position are a mapping contradiction
    conflicting = make_delivery(
        tmp_path,
        name="g10__s1",
        participants={"1": "R", "2": "A", "5": "A", "3": "B"},
        geometry_overrides={"5": ((3.0, 3.0), (1.0, 0.0))},
    )
    with pytest.raises(DeliveryGateError, match="conflicting"):
        run_intake(conflicting)

    # explicit dominant-fragment resolution keeps the better-supported tracklet
    # and records the overlap honestly in provenance
    resolved = run_intake(conflicting, IntakeConfig(fragment_overlap_policy="dominant"))
    overlap = resolved.provenance["fragment_overlap"]["A"]
    assert overlap["conflicting_instants"] > 0
    assert resolved.provenance["fragment_overlap_policy"] == "dominant"


def test_intake_rejects_mapping_binding_mismatch(tmp_path: Path) -> None:
    # a confirmed mapping.json must describe THIS delivery (Codex R1 ②)
    wrong_delivery = make_delivery(
        tmp_path, mapping_extra={"delivery": "g99__s9", "group": "g99", "scenario": "s9"}
    )
    with pytest.raises(DeliveryGateError, match="does not transfer"):
        run_intake(wrong_delivery)

    wrong_map = make_delivery(
        tmp_path,
        name="g11__s1",
        mapping_extra={"participant_map": {"1": "R", "2": "B", "3": "A"}},
    )
    with pytest.raises(DeliveryGateError, match="participant_map disagrees"):
        run_intake(wrong_map)


def test_intake_masks_frames_outside_sampled_span(tmp_path: Path) -> None:
    # B (p2) only has samples for the first half; those frames must be flagged
    # pose_valid=0, not silently extrapolated (Codex R1 ⑤ / R1.1-1). Slot
    # presence stays static — B does not "leave the conversation"
    directory = make_delivery(
        tmp_path, times_by_person={"3": np.arange(0, 50) / FPS_SRC}
    )
    result = run_intake(directory)
    frame_df = result.frame_df
    for column in ("P1_present", "P2_present", "P3_present"):
        assert frame_df[column].eq(1).all()
    masked = frame_df["P3_pose_valid"].eq(0)
    assert 0.4 < masked.mean() < 0.6
    assert frame_df.loc[masked, "P3_head_yaw_deg"].abs().max() == 0.0
    # validity propagates through the geometry (R1.1-4): R's yaw target is the
    # others' centroid, which includes B — so R is masked on the same frames
    assert frame_df.loc[masked, "P1_pose_valid"].eq(0).all()
    # A's target is R (validly observed throughout), so A stays valid
    assert frame_df["P2_pose_valid"].eq(1).all()
    assert result.provenance["missing_sample_fraction"]["P3"] == pytest.approx(
        masked.mean(), abs=1e-9
    )


def test_intake_requires_binding_fields(tmp_path: Path) -> None:
    # a confirmed mapping.json with no binding fields must not vouch for
    # anything (Codex R1.1-3)
    directory = make_delivery(tmp_path)
    (directory / "mapping.json").write_text(
        json.dumps({"review_status": "confirmed"}), encoding="utf-8"
    )
    with pytest.raises(DeliveryGateError, match="binding fields"):
        run_intake(directory)


def test_intake_rejects_contradictory_mappings(tmp_path: Path) -> None:
    # one person mapped to two different participants in the CSV
    directory = make_delivery(tmp_path)
    (directory / "participant_map.csv").write_text(
        "person,participant\npr,R\np1,A\np1,B\np2,B\n", encoding="utf-8"
    )
    with pytest.raises(DeliveryGateError, match="more than one participant"):
        run_intake(directory)

    duplicate = make_delivery(tmp_path, name="g15__s2")
    (duplicate / "participant_map.csv").write_text(
        "person,participant\npr,R\np1,A\np1,A\np2,B\n", encoding="utf-8"
    )
    with pytest.raises(DeliveryGateError, match="duplicate person rows"):
        run_intake(duplicate)

    # a person both mapped and declared unmapped
    tangled = make_delivery(tmp_path, name="g16__s1", unmapped_persons=["2"])
    with pytest.raises(DeliveryGateError, match="both mapped and listed"):
        run_intake(tangled)


def test_intake_to_dataset_preserves_pose_validity(tmp_path: Path) -> None:
    # full-chain check (Codex R1.1-1): the per-frame validity written by intake
    # must survive into the model's feature tensors, so a masked frame is
    # distinguishable from a genuine 0-degree yaw
    from hri_turn_taking.data.multiparty import PARTICIPANT_FEATURE_NAMES
    from hri_turn_taking.vap.dataset import sessions_from_dataframe

    directory = make_delivery(tmp_path, times_by_person={"3": np.arange(0, 50) / FPS_SRC})
    frame_df = run_intake(directory).frame_df
    sessions = sessions_from_dataframe(frame_df)
    assert len(sessions) == 1
    session = sessions[0]
    # slots 1..3 exist for the whole session; slot 4 is padding
    assert session.present.tolist() == [1.0, 1.0, 1.0, 0.0]
    n_features = len(PARTICIPANT_FEATURE_NAMES)
    pose_valid_p3 = session.features[:, 2 * n_features + PARTICIPANT_FEATURE_NAMES.index(
        "pose_valid")]
    assert pose_valid_p3.min() == 0.0 and pose_valid_p3.max() == 1.0
    assert pose_valid_p3.mean() == pytest.approx(
        frame_df["P3_pose_valid"].mean(), abs=1e-6
    )


def test_split_sessions_group_gate() -> None:
    # Real grouping comes from gated provenance, never punctuation in session_id.
    from types import SimpleNamespace

    from hri_turn_taking.vap.dataset import split_sessions

    real = [
        SimpleNamespace(session_id=f"recording-{i}", group_id=f"g{i // 2}")
        for i in range(4)
    ]
    train, val = split_sessions(real)
    assert train and val
    assert not ({s.group_id for s in train} & {s.group_id for s in val})

    incomplete = [
        SimpleNamespace(session_id="a", group_id="g1"),
        SimpleNamespace(session_id="b", group_id=None),
    ]
    with pytest.raises(ValueError, match="group provenance is incomplete"):
        split_sessions(incomplete)

    synthetic = [
        SimpleNamespace(session_id=session_id, group_id=None)
        for session_id in ("g1-part1", "g1-part2", "g2-part1", "g2-part2")
    ]
    train, val = split_sessions(synthetic, group_delimiter="-")
    assert not (
        {s.session_id.split("-")[0] for s in train}
        & {s.session_id.split("-")[0] for s in val}
    )


def test_assert_trainable_gate(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path)
    output = tmp_path / "out" / "g01__s2.csv"
    write_intake(directory, output)
    with pytest.raises(DeliveryGateError, match="trainable"):
        assert_trainable(output)
    assert_trainable(output, allow_untrainable=True)
    # A caller cannot turn a bare CSV into trusted synthetic data by naming it so.
    plain = tmp_path / "synthetic.csv"
    plain.write_text("session_id\n", encoding="utf-8")
    with pytest.raises(DeliveryGateError, match="not trusted as synthetic"):
        assert_trainable(plain)


def test_intake_rejects_malformed_unmapped_persons(tmp_path: Path) -> None:
    # Codex R1.2-3: a bare string "p1" iterates to ['p', '1'] and would dodge
    # the mapped/unmapped intersection check — it must be rejected as a type
    directory = make_delivery(tmp_path, mapping_extra={"unmapped_persons": "p1"})
    with pytest.raises(DeliveryGateError, match="array of scalar"):
        load_delivery(directory)

    nested = make_delivery(
        tmp_path, name="g22__s1", mapping_extra={"unmapped_persons": [["2"]]}
    )
    with pytest.raises(DeliveryGateError, match="array of scalar"):
        load_delivery(nested)

    not_object = make_delivery(
        tmp_path, name="g23__s1", mapping_extra={"participant_map": [["1", "R"]]}
    )
    with pytest.raises(DeliveryGateError, match="JSON object"):
        run_intake(not_object)


def test_intake_relaxed_r_validity_and_pose_quality(tmp_path: Path) -> None:
    # Codex R1.2-5 ruling: R's yaw is valid with >= 2 of 3 valid members in a
    # 4-party group; the centroid uses only the validly observed members, and
    # pose_quality carries the continuous valid-member fraction
    directory = make_delivery(
        tmp_path,
        participants={"1": "R", "2": "A", "3": "B", "4": "C"},
        geometry_overrides={"4": ((-2.0, 0.0), (0.0, 1.0))},
        times_by_person={"4": np.arange(0, 50) / FPS_SRC},  # C drops out halfway
    )
    frame_df = run_intake(directory).frame_df
    early = frame_df["P4_pose_valid"].eq(1)
    late = ~early
    assert 0.4 < late.mean() < 0.6
    # with all three members valid the centroid is at (0, 2/3) -> yaw 0 for R
    assert frame_df.loc[early, "P1_pose_valid"].eq(1).all()
    assert frame_df.loc[early, "P1_head_yaw_deg"].abs().max() < 1e-6
    assert frame_df.loc[early, "P1_pose_quality"].eq(1.0).all()
    # with C missing, 2/3 members remain: R STAYS valid (relaxed rule), the
    # centroid moves to (1, 1) -> 45 deg, and quality drops to 2/3
    assert frame_df.loc[late, "P1_pose_valid"].eq(1).all()
    assert frame_df.loc[late, "P1_head_yaw_deg"].to_numpy() == pytest.approx(45.0, abs=1e-6)
    assert frame_df.loc[late, "P1_pose_quality"].to_numpy() == pytest.approx(2 / 3, abs=1e-9)
    # non-R quality is binary: 1 iff its target (R) chain is valid
    assert frame_df["P2_pose_quality"].isin((0.0, 1.0)).all()


def test_intake_single_member_centroid_is_invalid(tmp_path: Path) -> None:
    # with only ONE valid member the "centroid" bearing degenerates into a
    # bearing to that person (addressee semantics) — R must be invalid there
    directory = make_delivery(
        tmp_path,
        times_by_person={"3": np.arange(0, 50) / FPS_SRC},
    )
    frame_df = run_intake(directory).frame_df
    late = frame_df["P3_pose_valid"].eq(0)
    assert frame_df.loc[late, "P1_pose_valid"].eq(0).all()
    assert frame_df.loc[late, "P1_pose_quality"].to_numpy() == pytest.approx(0.5, abs=1e-9)


def test_intake_zero_length_bearing_is_invalid(tmp_path: Path) -> None:
    # A standing exactly on top of R has an undefined bearing — the frame must
    # be pose_valid=0, not a silent 90 degrees (Codex R1.2-5)
    directory = make_delivery(tmp_path, geometry_overrides={"2": ((0.0, 0.0), (0.0, -1.0))})
    frame_df = run_intake(directory).frame_df
    assert frame_df["P2_pose_valid"].eq(0).all()
    assert frame_df["P2_head_yaw_deg"].abs().max() == 0.0


def test_intake_rejects_non_number_offsets(tmp_path: Path) -> None:
    # Codex R1.3-3: float("-1.2") must not silently accept a string offset
    directory = make_delivery(tmp_path, streams={"6cam": "-1.2"})
    with pytest.raises(DeliveryGateError, match="JSON number"):
        load_delivery(directory)
    boolean = make_delivery(tmp_path, name="g24__s1", streams={"6cam": True})
    with pytest.raises(DeliveryGateError, match="JSON number"):
        load_delivery(boolean)


def test_intake_interpolation_never_bridges_gaps(tmp_path: Path) -> None:
    # Codex R1.3-4: plain np.interp across a long tracking gap leaks the far
    # (future) endpoint into frames near the gap start. With gap-aware segment
    # interpolation, changing data BEYOND the gap must not change any valid
    # frame before it.
    early = np.arange(0, 50) / FPS_SRC  # 0 .. 1.96 s
    late = np.arange(75, 100) / FPS_SRC  # 3.0 .. 3.96 s (gap 1.04 s >> 0.2 s)

    def build(name: str, late_geometry: tuple) -> pd.DataFrame:
        stray = pd.DataFrame(
            [
                {
                    "frame_index": round(t * FPS_SRC),
                    "timestamp_sec": t,
                    "world_person": "2",
                    "n_cams": 3,
                    "cams": "2|3|4",
                    "x_w": late_geometry[0][0],
                    "y_w": 1.6,
                    "z_w": late_geometry[0][1],
                    "facing_x": late_geometry[1][0],
                    "facing_y": 0.0,
                    "facing_z": late_geometry[1][1],
                    "facing_src_cam": 2,
                    "facing_src_abs_yaw": 10.0,
                    "person": "2",
                    "tracklet": 0,
                }
                for t in late
            ]
        )
        directory = make_delivery(
            tmp_path,
            name=name,
            times_by_person={"2": early},
            extra_person_rows=stray,
        )
        return run_intake(directory).frame_df

    base = build("g25__s1", ((0.0, 2.0), (0.0, -1.0)))
    # move A's post-gap position/facing wildly (10 m away, rotated 90 deg)
    moved = build("g26__s1", ((10.0, 2.0), (1.0, 0.0)))

    # holding tolerates up to max_gap_s (0.2 s) of staleness past the last
    # sample; beyond that the gap frames are invalid until the next sample
    in_gap = (base["timestamp_s"] > 2.17 + OFFSET_6CAM) & (
        base["timestamp_s"] < 3.0 + OFFSET_6CAM
    )
    assert in_gap.any()
    assert base.loc[in_gap, "P2_pose_valid"].eq(0).all()
    before_gap = base["P2_pose_valid"].eq(1) & (base["timestamp_s"] <= 2.0 + OFFSET_6CAM)
    assert before_gap.any()
    assert np.array_equal(
        base.loc[before_gap, "P2_head_yaw_deg"].to_numpy(),
        moved.loc[before_gap, "P2_head_yaw_deg"].to_numpy(),
    )


def test_intake_resampling_is_strict_past_only(tmp_path: Path) -> None:
    # Codex R1.4-2: even between ADJACENT samples (well within max_gap_s) a
    # later sample must not influence earlier frames — frames hold the latest
    # past observation, they are not interpolated toward the next one
    def build(name: str, second_sample: tuple) -> pd.DataFrame:
        extra = pd.DataFrame(
            [
                {
                    "frame_index": 5,
                    "timestamp_sec": 0.2,
                    "world_person": "2",
                    "n_cams": 3,
                    "cams": "2|3|4",
                    "x_w": second_sample[0][0],
                    "y_w": 1.6,
                    "z_w": second_sample[0][1],
                    "facing_x": second_sample[1][0],
                    "facing_y": 0.0,
                    "facing_z": second_sample[1][1],
                    "facing_src_cam": 2,
                    "facing_src_abs_yaw": 10.0,
                    "person": "2",
                    "tracklet": 0,
                }
            ]
        )
        directory = make_delivery(
            tmp_path,
            name=name,
            times_by_person={"2": np.array([0.0])},
            extra_person_rows=extra,
        )
        return run_intake(directory).frame_df

    steady = build("g27__s1", ((0.0, 2.0), (0.0, -1.0)))  # same pose at t=0.2
    turned = build("g28__s1", ((0.0, 2.0), (1.0, 0.0)))  # rotated 90 deg at t=0.2

    between = (steady["timestamp_s"] >= OFFSET_6CAM) & (
        steady["timestamp_s"] < 0.2 + OFFSET_6CAM - 1e-9
    )
    assert between.sum() == 5  # frames at 0.00 .. 0.16
    assert steady.loc[between, "P2_pose_valid"].eq(1).all()
    # every in-between frame holds the t=0 sample (yaw 0), bit-identical in
    # both variants — no blending toward the future sample
    assert steady.loc[between, "P2_head_yaw_deg"].abs().max() == 0.0
    assert np.array_equal(
        steady.loc[between, "P2_head_yaw_deg"].to_numpy(),
        turned.loc[between, "P2_head_yaw_deg"].to_numpy(),
    )
    # the new observation takes effect exactly AT its own timestamp
    at_sample = np.isclose(turned["timestamp_s"], 0.2 + OFFSET_6CAM)
    assert turned.loc[at_sample, "P2_head_yaw_deg"].to_numpy() == pytest.approx(
        90.0, abs=1e-6
    )


def test_intake_rejects_subnanosecond_future_sample_from_past_frame(
    tmp_path: Path,
) -> None:
    future_delta_s = 5e-10
    extra = pd.DataFrame(
        [
            {
                "frame_index": 5,
                "timestamp_sec": 0.2,
                "world_person": "2",
                "n_cams": 3,
                "cams": "2|3|4",
                "x_w": 0.0,
                "y_w": 1.6,
                "z_w": 2.0,
                "facing_x": 1.0,
                "facing_y": 0.0,
                "facing_z": 0.0,
                "facing_src_cam": 2,
                "facing_src_abs_yaw": 10.0,
                "person": "2",
                "tracklet": 0,
            }
        ]
    )
    directory = make_delivery(
        tmp_path,
        name="g39__s1",
        times_by_person={"2": np.array([0.0])},
        extra_person_rows=extra,
        streams={"6cam": OFFSET_6CAM + future_delta_s},
    )

    frame_df = run_intake(directory).frame_df
    at_grid_tick = np.isclose(
        frame_df["timestamp_s"].to_numpy(),
        -1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert at_grid_tick.sum() == 1
    assert frame_df.loc[at_grid_tick, "P2_pose_valid"].eq(1).all()
    assert frame_df.loc[at_grid_tick, "P2_head_yaw_deg"].abs().max() == 0.0


def test_intake_rejects_bad_max_gap(tmp_path: Path) -> None:
    # Codex R1.4-2: NaN makes every gap comparison false — must fail closed
    directory = make_delivery(tmp_path)
    for bad in (float("nan"), 0.0, -0.2, 2.0):
        with pytest.raises(DeliveryGateError, match="max_gap_s"):
            run_intake(directory, IntakeConfig(max_gap_s=bad))


def test_intake_fragment_facing_conflict_and_local_winner(tmp_path: Path) -> None:
    # Codex R1.4-3: fragments agreeing on position but not facing are still a
    # conflict (facing is the consumed feature), and the overlap winner must be
    # chosen by per-instant local quality — never whole-track length, so
    # adding future rows to the losing fragment cannot flip past frames
    participants = {"1": "R", "2": "A", "5": "A", "3": "B"}
    geometry = {"5": ((0.0, 2.0), (0.0, 1.0))}  # same spot as person 2, opposite facing
    conflicted = make_delivery(
        tmp_path,
        name="g29__s1",
        participants=participants,
        geometry_overrides=geometry,
        abs_yaw_by_person={"2": 5.0, "5": 40.0},
    )
    with pytest.raises(DeliveryGateError, match="facing directions"):
        run_intake(conflicted)

    # dominant policy: p1 wins every overlapping instant on local quality
    # (abs_yaw 5 < 40) even though p1b's track is longer (future support)
    longer = make_delivery(
        tmp_path,
        name="g30__s1",
        participants=participants,
        geometry_overrides=geometry,
        abs_yaw_by_person={"2": 5.0, "5": 40.0},
        times_by_person={"5": np.arange(0, 150) / FPS_SRC},
    )
    frame_df = run_intake(longer, IntakeConfig(fragment_overlap_policy="dominant")).frame_df
    overlap_span = frame_df["timestamp_s"] <= 99 / FPS_SRC + OFFSET_6CAM
    assert frame_df.loc[overlap_span, "P2_head_yaw_deg"].abs().max() < 1e-6  # p1: faces R
    tail = (frame_df["timestamp_s"] >= 101 / FPS_SRC + OFFSET_6CAM) & frame_df[
        "P2_pose_valid"
    ].eq(1)
    assert tail.any()
    assert frame_df.loc[tail, "P2_head_yaw_deg"].to_numpy() == pytest.approx(
        180.0, abs=1e-6
    )  # only p1b remains after p1 ends (until R's own validity runs out)


def test_multiparty_validation_rejects_mask_violations(tmp_path: Path) -> None:
    # value-level schema gates (Codex R1.2-1): dynamic present, non-binary
    # pose_valid and validity on absent slots must fail validation instead of
    # being silently crushed downstream
    directory = make_delivery(tmp_path)
    base = run_intake(directory).frame_df

    dynamic = base.copy()
    dynamic.loc[dynamic.index[:10], "P2_present"] = 0
    dynamic.loc[dynamic.index[:10], ["P2_pose_valid", "P2_pose_quality"]] = 0
    with pytest.raises(ValueError, match="STATIC"):
        validate_multiparty_dataframe(dynamic)

    fractional = base.copy()
    fractional["P2_pose_valid"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        validate_multiparty_dataframe(fractional)

    ghost_valid = base.copy()
    ghost_valid["P4_pose_valid"] = 1.0
    with pytest.raises(ValueError, match="absent slots"):
        validate_multiparty_dataframe(ghost_valid)

    # final-gate finiteness and zero-fill semantics (Codex R1.3-4)
    nan_input = base.copy()
    nan_input.loc[nan_input.index[0], "P2_asd_prob"] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_multiparty_dataframe(nan_input)

    inf_global = base.copy()
    inf_global.loc[inf_global.index[0], "array_energy"] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_multiparty_dataframe(inf_global)

    stale_yaw = base.copy()
    invalid_rows = stale_yaw["P3_pose_valid"].eq(0)
    if not invalid_rows.any():  # ensure at least one invalid frame to poison
        stale_yaw.loc[stale_yaw.index[:5], "P3_pose_valid"] = 0.0
        invalid_rows = stale_yaw["P3_pose_valid"].eq(0)
    stale_yaw.loc[invalid_rows, "P3_head_yaw_deg"] = 33.0
    with pytest.raises(ValueError, match="zero-filled"):
        validate_multiparty_dataframe(stale_yaw)


def test_intake_provenance_and_outputs(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path)
    output = tmp_path / "out" / "g01__s2.csv"
    result = write_intake(directory, output)

    assert output.exists()
    provenance_path = output.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["trainable"] is False
    assert provenance["data_kind"] == "real"
    assert provenance["artifact_kind"] == "vap_real_pose_intake"
    assert provenance["format_version"] == 2
    assert provenance["identity_authority"] == "sealed_hpe_transcript_crosswalk"
    assert provenance["identity_contract"]["scene"] == "g01__s2"
    assert provenance["accepted_window"]["filter_rule"] == "start <= frame_index < end"
    persisted = pd.read_csv(output, float_precision="round_trip")
    assert provenance["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance["columns"] == persisted.columns.tolist()
    assert provenance["dtypes"] == {
        column: str(persisted[column].dtype) for column in persisted
    }
    assert provenance["n_rows"] == len(persisted)
    assert provenance["group_id"] == "g01"
    assert provenance["scenario_id"] == "s2"
    assert provenance["offset_applied_s"] == OFFSET_6CAM
    assert provenance["offset_quality"] == {
        "stream": "6cam",
        "offset_s": OFFSET_6CAM,
        "self_check_ok": True,
        "convention": "shared_time = stream_local_time + offset_s",
        "status": "ok",
        "error_bound_ms": 33.0,
        "timebase_keys": {"6cam": "avi_frame_axis"},
    }
    assert set(provenance["d3_sources"]) == {"offset", "offset_quality"}
    assert all(
        len(source["sha256"]) == 64 and Path(source["path"]).is_file()
        for source in provenance["d3_sources"].values()
    )
    assert provenance["participant_to_slot"] == {"R": "P1", "A": "P2", "B": "P3"}
    # only head yaw is real at this stage; gold VAD must be a listed placeholder
    assert set(provenance["real_columns"]) == {
        f"P{i}_{name}"
        for i in (1, 2, 3)
        for name in ("head_yaw_deg", "pose_valid", "pose_quality")
    }
    assert "P1_vad" in provenance["placeholder_columns"]
    assert "doa_angle_deg" in provenance["placeholder_columns"]
    assert set(provenance["column_sources"]) == set(provenance["real_columns"])
    assert provenance["column_sources"]["P1_head_yaw_deg"]["participant"] == "R"
    assert provenance["column_sources"]["P1_head_yaw_deg"]["slot"] == "P1"
    assert result.frame_df.equals(
        pd.read_csv(output, float_precision="round_trip").astype(
            result.frame_df.dtypes.to_dict()
        )
    )


def test_intake_rejects_delivery_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_delivery(tmp_path, name="g32__s1")
    world_path = directory / "world_persons.csv"
    original = intake_module._horizontal_bearing_deviation_deg
    changed = False

    def mutate_source(facing: np.ndarray, target: np.ndarray) -> np.ndarray:
        nonlocal changed
        if not changed:
            world_path.write_text(
                world_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            changed = True
        return original(facing, target)

    monkeypatch.setattr(
        intake_module, "_horizontal_bearing_deviation_deg", mutate_source
    )
    with pytest.raises(DeliveryGateError, match="changed while intake was running"):
        run_intake(directory)


def test_write_intake_never_overwrites_existing_pair(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path, name="g33__s1")
    output = tmp_path / "intake_v2" / "g33__s1.csv"
    write_intake(directory, output)
    original_csv = output.read_bytes()
    original_sidecar = output.with_suffix(".provenance.json").read_bytes()

    with pytest.raises(DeliveryGateError, match="Refusing to overwrite"):
        write_intake(directory, output)

    assert output.read_bytes() == original_csv
    assert output.with_suffix(".provenance.json").read_bytes() == original_sidecar


def test_write_intake_rolls_back_csv_when_sidecar_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_delivery(tmp_path, name="g34__s1")
    output = tmp_path / "intake_v2" / "g34__s1.csv"
    provenance_path = output.with_suffix(".provenance.json")
    original_link = intake_module.os.link

    def fail_sidecar_link(source: Path, target: Path) -> None:
        if Path(target) == provenance_path:
            raise OSError("injected sidecar publication failure")
        original_link(source, target)

    monkeypatch.setattr(intake_module.os, "link", fail_sidecar_link)
    with pytest.raises(DeliveryGateError, match="Could not publish intake output"):
        write_intake(directory, output)

    assert not output.exists()
    assert not provenance_path.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp.*.csv"))
    assert not list(provenance_path.parent.glob(f".{provenance_path.name}.*.json.tmp"))


@pytest.mark.parametrize("concurrent_target", ["csv", "sidecar"])
def test_write_intake_never_clobbers_a_concurrent_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent_target: str
) -> None:
    directory = make_delivery(tmp_path, name="g36__s1")
    output = tmp_path / "intake_v2" / "g36__s1.csv"
    sidecar = output.with_suffix(".provenance.json")
    target = output if concurrent_target == "csv" else sidecar
    original_link = intake_module.os.link
    injected = False

    def create_concurrent_owner(source: Path, destination: Path) -> None:
        nonlocal injected
        if Path(destination) == target and not injected:
            target.write_bytes(b"CONCURRENT_OWNER")
            injected = True
        original_link(source, destination)

    monkeypatch.setattr(intake_module.os, "link", create_concurrent_owner)
    with pytest.raises(DeliveryGateError, match="appeared while materialisation"):
        write_intake(directory, output)

    assert target.read_bytes() == b"CONCURRENT_OWNER"
    other = sidecar if target == output else output
    assert not other.exists()


def test_write_intake_rolls_back_pair_if_delivery_changes_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_delivery(tmp_path, name="g37__s1")
    world_path = directory / "world_persons.csv"
    output = tmp_path / "intake_v2" / "g37__s1.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = intake_module._publish_no_clobber
    changed = False

    def mutate_then_publish(staged: Path, destination: Path) -> None:
        nonlocal changed
        if not changed:
            world_path.write_text(
                world_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            changed = True
        original_publish(staged, destination)

    monkeypatch.setattr(intake_module, "_publish_no_clobber", mutate_then_publish)
    with pytest.raises(DeliveryGateError, match="changed during intake publication"):
        write_intake(directory, output)

    assert not output.exists()
    assert not sidecar.exists()


def test_write_intake_rolls_back_csv_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_delivery(tmp_path, name="g38__s1")
    output = tmp_path / "intake_v2" / "g38__s1.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = intake_module._publish_no_clobber
    calls = 0

    def interrupt_second_publish(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        original_publish(staged, destination)

    monkeypatch.setattr(intake_module, "_publish_no_clobber", interrupt_second_publish)
    with pytest.raises(KeyboardInterrupt):
        write_intake(directory, output)

    assert not output.exists()
    assert not sidecar.exists()


def test_write_intake_rejects_staged_csv_tamper_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_delivery(tmp_path, name="g40__s1")
    output = tmp_path / "intake_v2" / "g40__s1.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = intake_module._publish_no_clobber
    tampered = False

    def tamper_then_publish(staged: Path, destination: Path) -> None:
        nonlocal tampered
        if Path(destination) == output and not tampered:
            staged.write_bytes(staged.read_bytes() + b"\n")
            tampered = True
        original_publish(staged, destination)

    monkeypatch.setattr(intake_module, "_publish_no_clobber", tamper_then_publish)
    with pytest.raises(DeliveryGateError, match="Published intake bytes differ"):
        write_intake(directory, output)

    assert not output.exists()
    assert not sidecar.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp.*.csv"))


def test_intake_staging_paths_are_writer_private(tmp_path: Path) -> None:
    first = intake_module._reserve_staged_path(
        tmp_path, prefix=".intake.tmp.", suffix=".csv"
    )
    second = intake_module._reserve_staged_path(
        tmp_path, prefix=".intake.tmp.", suffix=".csv"
    )
    try:
        assert first != second
        assert first.is_file()
        assert second.is_file()
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_renamed_intake_without_sidecar_cannot_be_declared_synthetic(tmp_path: Path) -> None:
    directory = make_delivery(tmp_path)
    output = tmp_path / "intake" / "g01__s2.csv"
    write_intake(directory, output)
    renamed = tmp_path / "renamed_synthetic.csv"
    shutil.copyfile(output, renamed)

    with pytest.raises(DataSourceGateError, match="bare CSV files are never trusted"):
        load_vap_dataframe(renamed, data_kind="synthetic")
