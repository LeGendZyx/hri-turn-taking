from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hri_turn_taking.cli import app
from hri_turn_taking.data.assembly import load_and_audit_vap_source


@pytest.mark.parametrize(
    "arguments",
    [
        ["vap-audit"],
        ["vap-train"],
        ["vap-predict", "dataset.csv"],
        ["vap-baselines", "dataset.csv"],
        ["vap-demo"],
    ],
)
def test_vap_data_consumers_require_explicit_data_kind(arguments: list[str]) -> None:
    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert "Missing option '--data-kind'" in result.output


def test_vap_generate_writes_a_verifiable_synthetic_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic.csv"

    result = CliRunner().invoke(
        app,
        [
            "vap-generate",
            "--output",
            str(output),
            "--sessions",
            "1",
            "--duration-s",
            "1",
            "--participants",
            "3",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    frame_df, report = load_and_audit_vap_source(output, data_kind="synthetic")
    assert len(frame_df) == 25
    assert report.structural_ready is True
    assert output.with_suffix(".provenance.json").is_file()


def test_vap_intake_uses_v2_output_and_sealed_crosswalk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivery = tmp_path / "g01__s1"
    delivery.mkdir()
    captured: dict[str, object] = {}

    def fake_write_intake(
        delivery_dir: Path,
        output: Path,
        *,
        crosswalk_inventory: Path,
    ) -> SimpleNamespace:
        captured.update(
            delivery=delivery_dir,
            output=output,
            crosswalk_inventory=crosswalk_inventory,
        )
        return SimpleNamespace(
            provenance={
                "session_id": "g01__s1",
                "n_frames": 10,
                "identity_authority": "sealed_hpe_transcript_crosswalk",
                "accepted_window": {
                    "logical_start_inclusive": 0,
                    "logical_end_exclusive": 12,
                },
                "stream_key": "6cam",
                "offset_applied_s": -1.0,
                "real_columns": [],
                "placeholder_columns": [],
            }
        )

    monkeypatch.setattr(
        "hri_turn_taking.data.intake.write_intake", fake_write_intake
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["vap-intake", str(delivery)])

    assert result.exit_code == 0, result.output
    assert captured["output"] == Path("data/processed/intake_v2/g01__s1.csv")
    assert captured["crosswalk_inventory"] == Path(
        "configs/hpe_transcript_crosswalk_inventory.json"
    )


@pytest.mark.parametrize(
    "removed_flag",
    ["--allow-unconfirmed", "--allow-missing-offset-quality", "--fps", "--stream-key"],
)
def test_vap_intake_has_no_formal_debug_overrides(
    tmp_path: Path, removed_flag: str
) -> None:
    delivery = tmp_path / "g01__s1"
    delivery.mkdir()

    result = CliRunner().invoke(app, ["vap-intake", str(delivery), removed_flag, "1"])

    assert result.exit_code == 2
    assert "No such option" in result.output
