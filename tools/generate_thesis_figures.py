"""Generate reproducible, data-bound figures for the dissertation and demo.

The figures are deliberately static and result-free: they explain frozen data
quality, label construction, evaluation splits, and the deadline A/H/A+H model
conditions without reading raw recordings or running a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from hri_turn_taking.data.multiparty import (
    GLOBAL_FEATURE_COLUMNS,
    PARTICIPANT_FEATURE_NAMES,
)
from hri_turn_taking.vap.dataset import TRANSFORMED_GLOBAL_FEATURES
from hri_turn_taking.vap.feature_sets import get_feature_set
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, future_activity_labels
from hri_turn_taking.vap.model import VAPModelConfig
from hri_turn_taking.vap.outer_experiment import (
    D3_40MS_DESCRIPTIVE_SESSION_IDS,
    D3_FROZEN_VALUES,
    FROZEN_SEEDS,
    PAIRED_D3_80MS_SESSION_IDS,
    PRIMARY_D3_SESSION_IDS,
    PRIMARY_SESSION_TO_GROUP,
)

FIGURE_FORMAT_VERSION = 1
FIXED_ARTIFACT_TIME = datetime(2026, 8, 16, tzinfo=UTC)
FIGURE_STEMS = (
    "d3_uncertainty",
    "transcript_vad_vap_timeline",
    "nested_logo_protocol",
    "participant_vap_architecture",
)

# Okabe-Ito-inspired palette: distinct in common colour-vision deficiencies.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
DARK = "#222222"
MID_GREY = "#777777"
LIGHT_GREY = "#E6E6E6"
PALE_BLUE = "#DCEEF8"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class D3Record:
    """One selected scene's locally mirrored D3 synchronization evidence."""

    session_id: str
    group_id: str
    offset_s: float
    quality_status: str
    error_bound_ms: float


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_d3_records(repo_root: Path) -> tuple[D3Record, ...]:
    """Load and cross-check the ten selected D3 records from the formal mirror."""

    inventory_path = repo_root / "configs" / "hpe_delivery_inventory.json"
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    deliveries = payload.get("deliveries")
    if not isinstance(deliveries, list):
        raise ValueError("HPE delivery inventory has no deliveries list")
    by_scene = {
        entry.get("session_id"): entry
        for entry in deliveries
        if isinstance(entry, dict) and isinstance(entry.get("session_id"), str)
    }

    records: list[D3Record] = []
    for session_id in PRIMARY_D3_SESSION_IDS:
        entry = by_scene.get(session_id)
        if entry is None:
            raise ValueError(f"D3 inventory is missing selected scene {session_id}")
        group_id = entry.get("group_id")
        d3 = entry.get("d3_6cam")
        if group_id != PRIMARY_SESSION_TO_GROUP[session_id] or not isinstance(d3, dict):
            raise ValueError(f"D3 inventory group/quality structure differs for {session_id}")
        if (
            d3.get("status") != "published"
            or d3.get("shared_timeline_eligible") is not True
            or d3.get("self_check_ok") is not True
        ):
            raise ValueError(f"selected scene {session_id} does not have accepted D3 evidence")

        offset_s = d3.get("offset_s")
        error_bound_ms = d3.get("error_bound_ms")
        quality_status = d3.get("quality_status")
        if (
            isinstance(offset_s, bool)
            or not isinstance(offset_s, (int, float))
            or not math.isfinite(float(offset_s))
            or isinstance(error_bound_ms, bool)
            or not isinstance(error_bound_ms, (int, float))
            or not math.isfinite(float(error_bound_ms))
            or float(error_bound_ms) < 0
            or quality_status not in {"ok", "degraded"}
        ):
            raise ValueError(f"selected scene {session_id} has malformed D3 values")

        frozen = D3_FROZEN_VALUES[session_id]
        observed = {
            "offset_s": float(offset_s),
            "status": quality_status,
            "self_check_ok": d3.get("self_check_ok"),
            "error_bound_ms": float(error_bound_ms),
        }
        if observed != frozen:
            raise ValueError(
                f"D3 inventory and frozen experiment protocol disagree for {session_id}"
            )
        records.append(
            D3Record(
                session_id=session_id,
                group_id=group_id,
                offset_s=float(offset_s),
                quality_status=quality_status,
                error_bound_ms=float(error_bound_ms),
            )
        )

    observed_80 = tuple(record.session_id for record in records if record.error_bound_ms <= 80)
    observed_40 = tuple(record.session_id for record in records if record.error_bound_ms <= 40)
    if observed_80 != PAIRED_D3_80MS_SESSION_IDS:
        raise ValueError("D3 <=80 ms subset differs from the frozen paired sensitivity set")
    if observed_40 != D3_40MS_DESCRIPTIVE_SESSION_IDS:
        raise ValueError("D3 <=40 ms subset differs from the frozen descriptive set")
    return tuple(records)


def _uncertainty_colour(value: float) -> tuple[str, str]:
    if value <= 40:
        return BLUE, ""
    if value <= 80:
        return ORANGE, "//"
    return VERMILLION, "xx"


def make_d3_uncertainty_figure(records: tuple[D3Record, ...]) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    values = [record.error_bound_ms for record in records]
    x = np.arange(len(records))
    styles = [_uncertainty_colour(value) for value in values]
    bars = ax.bar(
        x,
        values,
        width=0.7,
        color=[style[0] for style in styles],
        edgecolor=DARK,
        linewidth=0.55,
    )
    for bar, style, value in zip(bars, styles, values, strict=True):
        bar.set_hatch(style[1])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(40, color=BLUE, linestyle="--", linewidth=1.15, label="40 ms target")
    ax.axhline(
        80,
        color=VERMILLION,
        linestyle=(0, (5, 2)),
        linewidth=1.15,
        label="80 ms prior diagnostic threshold",
    )
    ax.set_xticks(x, [record.session_id for record in records], rotation=35, ha="right")
    ax.set_ylabel("Published D3 error bound (ms)")
    ax.set_ylim(0, max(values) + 18)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncols=2,
    )
    fig.suptitle("Six-camera to mic-A synchronization uncertainty", y=0.985)
    fig.text(
        0.5,
        0.905,
        (
            f"Source corpus: n={len(records)}   "
            "Deadline evaluation: n=8   central offsets only"
        ),
        ha="center",
        va="center",
        fontsize=8,
        color=DARK,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.24, top=0.78)
    return fig


def label_schematic_data() -> dict[str, Any]:
    """Return deterministic illustrative intervals and labels using production code."""

    config = DEFAULT_LABEL_CONFIG
    intervals = {
        "P1": ((0.00, 0.16), (0.92, 1.52)),
        "P2": ((0.12, 0.44), (1.44, 2.00)),
        "P3": ((0.36, 0.76), (1.80, 1.96)),
    }
    n_frames = config.horizon_frames
    times = np.arange(n_frames, dtype=float) / config.frame_rate_hz
    vad = np.zeros((n_frames, len(intervals)), dtype=float)
    for participant_index, participant_intervals in enumerate(intervals.values()):
        for start_s, end_s in participant_intervals:
            vad[:, participant_index] = np.maximum(
                vad[:, participant_index],
                ((times >= start_s) & (times < end_s)).astype(float),
            )
    labels, valid = future_activity_labels(vad, config)
    if not valid[0]:
        raise AssertionError("the schematic horizon must be complete at its first frame")
    return {
        "intervals": intervals,
        "times": times,
        "vad": vad,
        "labels_at_t0": labels[0],
    }


def make_label_timeline_figure() -> Figure:
    schematic = label_schematic_data()
    config = DEFAULT_LABEL_CONFIG
    participants = tuple(schematic["intervals"])
    colours = (BLUE, ORANGE, GREEN)

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.25, 0.9, 1.15), hspace=0.78)
    axes = [fig.add_subplot(grid[index]) for index in range(3)]
    transcript_ax, raster_ax, target_ax = axes

    for row, (participant, colour) in enumerate(zip(participants, colours, strict=True)):
        y = len(participants) - 1 - row
        for start_s, end_s in schematic["intervals"][participant]:
            transcript_ax.broken_barh(
                [(start_s, end_s - start_s)],
                (y - 0.3, 0.6),
                facecolors=colour,
                edgecolors=DARK,
                linewidth=0.55,
            )
            transcript_ax.plot(start_s, y, marker="|", color=DARK, markersize=9)
            transcript_ax.plot(
                end_s,
                y,
                marker="o",
                markerfacecolor=WHITE,
                markeredgecolor=DARK,
                markersize=4,
            )
    transcript_ax.set_yticks(range(len(participants)), participants[::-1])
    transcript_ax.set_xlim(0, config.bin_edges_s[-1])
    transcript_ax.set_ylim(-0.55, len(participants) - 0.45)
    transcript_ax.set_xticks(config.bin_edges_s)
    transcript_ax.set_xlabel("Time after projection point t (s)")
    transcript_ax.set_title("1. Human-reviewed speaker intervals, represented as [start, end)")
    transcript_ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.6)

    raster_ax.imshow(
        schematic["vad"].T,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, config.bin_edges_s[-1], len(participants) - 0.5, -0.5),
        cmap=matplotlib.colors.ListedColormap((WHITE, SKY)),
        vmin=0,
        vmax=1,
    )
    raster_ax.set_yticks(range(len(participants)), participants)
    raster_ax.set_xticks(config.bin_edges_s)
    raster_ax.set_xlabel("25 Hz frame grid (40 ms per frame)")
    raster_ax.set_title("2. Per-participant VAD raster (union within each speaker)")
    raster_ax.set_xlim(0, config.bin_edges_s[-1])
    for edge in (0.0, *config.bin_edges_s):
        raster_ax.axvline(edge, color=MID_GREY, linewidth=0.55)

    labels = schematic["labels_at_t0"]
    target_ax.imshow(
        labels,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap((LIGHT_GREY, BLUE)),
        vmin=0,
        vmax=1,
    )
    for row in range(labels.shape[0]):
        for column in range(labels.shape[1]):
            target_ax.text(
                column,
                row,
                str(int(labels[row, column])),
                ha="center",
                va="center",
                color=WHITE if labels[row, column] else DARK,
                fontsize=9,
                fontweight="bold",
            )
    edges = (0.0, *config.bin_edges_s)
    target_ax.set_xticks(
        range(config.n_bins),
        [f"[{left:g}, {right:g}) s" for left, right in zip(edges[:-1], edges[1:], strict=True)],
    )
    target_ax.set_yticks(range(len(participants)), participants)
    target_ax.set_title(
        f"3. Four future-bin targets at t (positive when activity >= {config.active_ratio:.1f})"
    )
    target_ax.set_xlabel("Participant-wise VAP target: time x participant x 4 bins")
    for spine in target_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
        spine.set_linewidth(0.5)

    fig.suptitle("Transcript-derived VAD and future-activity label construction", y=0.985)
    fig.text(
        0.99,
        0.012,
        "Illustrative schematic, not a corpus sample",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MID_GREY,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.13, top=0.91)
    return fig


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = DARK,
    fontsize: float = 8,
    linewidth: float = 0.8,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = DARK,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=colour,
            linewidth=0.85,
            connectionstyle=connectionstyle,
        )
    )


def _ordered_groups() -> tuple[str, ...]:
    return tuple(dict.fromkeys(PRIMARY_SESSION_TO_GROUP.values()))


def make_nested_logo_figure() -> Figure:
    groups = _ordered_groups()
    if len(groups) != 4:
        raise ValueError(f"nested LOGO schematic requires four groups, found {len(groups)}")
    group_colours = dict(zip(groups, (BLUE, ORANGE, GREEN, PURPLE), strict=True))

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.35))
    for fold_index, (ax, test_group) in enumerate(zip(axes.flat, groups, strict=True), start=1):
        development = tuple(group for group in groups if group != test_group)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.02, 0.95, f"Outer fold {fold_index}", fontsize=9.5, fontweight="bold", va="top")

        _box(
            ax,
            0.03,
            0.59,
            0.59,
            0.24,
            "",
            facecolor="#F7F7F7",
        )
        ax.text(
            0.325,
            0.765,
            "Development groups",
            ha="center",
            va="center",
            fontsize=8.2,
        )
        chip_width = 0.155
        for chip_index, group in enumerate(development):
            chip_x = 0.065 + chip_index * 0.18
            ax.add_patch(
                Rectangle(
                    (chip_x, 0.625),
                    chip_width,
                    0.065,
                    facecolor=group_colours[group],
                    edgecolor=DARK,
                    linewidth=0.45,
                )
            )
            ax.text(
                chip_x + chip_width / 2,
                0.6575,
                group,
                ha="center",
                va="center",
                fontsize=7.6,
                color=DARK,
            )

        _box(
            ax,
            0.03,
            0.27,
            0.59,
            0.20,
            (
                "Inner LOGO model selection\n"
                + "Validation: "
                + " | ".join(development)
            ),
            facecolor=PALE_BLUE,
            fontsize=7.7,
        )
        _arrow(ax, (0.325, 0.59), (0.325, 0.48))
        _box(
            ax,
            0.73,
            0.47,
            0.24,
            0.27,
            f"Held-out test\n{test_group}\n(evaluate once)",
            facecolor=group_colours[test_group],
            fontsize=8.0,
        )
        ax.plot([0.62, 0.675, 0.675], [0.37, 0.37, 0.605], color=DARK, linewidth=0.85)
        _arrow(ax, (0.675, 0.605), (0.73, 0.605))
        ax.text(
            0.325,
            0.125,
            "Select epoch, then refit on the\nthree development groups",
            ha="center",
            va="center",
            fontsize=7.5,
            color=DARK,
        )

    fig.suptitle("Four-group nested leave-one-group-out evaluation", y=0.985)
    fig.text(
        0.5,
        0.018,
        "No outer-test group enters fitting or model selection. Primary analysis: seed 13.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.09, top=0.91, wspace=0.11, hspace=0.14)
    return fig


def make_architecture_figure() -> Figure:
    participant_dims = len(PARTICIPANT_FEATURE_NAMES)
    global_dims = len(TRANSFORMED_GLOBAL_FEATURES)
    config = VAPModelConfig(
        n_participant_features=participant_dims,
        n_global_features=global_dims,
    )
    if len(GLOBAL_FEATURE_COLUMNS) + 4 * participant_dims != 32:
        raise ValueError("raw canonical input schema no longer has the documented 32 columns")

    fig, ax = plt.subplots(figsize=(7.6, 4.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.012, 0.56, 0.235, 0.31, "", facecolor="#FAFAFA")
    ax.text(0.1295, 0.84, "Primary input conditions", ha="center", va="center", fontsize=9)
    rows = (
        ("A", "global block:\nenergy, F0, voiced\nDoA fixed to zero", BLUE),
        ("H", "participant block:\nyaw, validity, quality", ORANGE),
        ("A+H", "union of A and H", GREEN),
    )
    audio_set = get_feature_set("audio_only")
    head_set = get_feature_set("head_pose_only")
    combined_set = get_feature_set("audio_plus_head_pose")
    if (
        combined_set.participant_features != head_set.participant_features
        or combined_set.global_features != audio_set.global_features
    ):
        raise ValueError("A+H is no longer the exact union of the A and H feature sets")
    for row_index, (label, description, colour) in enumerate(rows):
        y = 0.755 - row_index * 0.09
        ax.text(
            0.028,
            y,
            label,
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=colour,
        )
        ax.text(
            0.077,
            y,
            description,
            ha="left",
            va="center",
            fontsize=8,
        )
    _box(
        ax,
        0.268,
        0.58,
        0.155,
        0.27,
        (
            "Participant tokens\n"
            "P1  P2  P3  P4\n\n"
            "own features +\n"
            "shared global features\n"
            "presence mask"
        ),
        facecolor=PALE_BLUE,
        fontsize=7.7,
    )
    _arrow(ax, (0.247, 0.715), (0.268, 0.715))

    _box(
        ax,
        0.449,
        0.63,
        0.125,
        0.17,
        f"Shared\nembedding\n{participant_dims + global_dims} to {config.d_model}",
        facecolor="#F2E7F3",
        fontsize=7.9,
    )
    _arrow(ax, (0.423, 0.715), (0.449, 0.715))

    lookback_s = config.attention_window / DEFAULT_LABEL_CONFIG.frame_rate_hz
    _box(
        ax,
        0.603,
        0.545,
        0.225,
        0.34,
        (
            f"{config.n_layers} shared blocks\n\n"
            "Causal temporal attention\n"
            f"ALiBi, {config.attention_window} frames ({lookback_s:g} s)\n"
            "Cross-participant attention\n"
            "Feed-forward network"
        ),
        facecolor="#FFF2CC",
        fontsize=7.6,
    )
    _arrow(ax, (0.574, 0.715), (0.603, 0.715))

    _box(
        ax,
        0.857,
        0.595,
        0.13,
        0.24,
        (
            "Shared output\n"
            "head\n"
            "4 future bins\n"
            "per participant\n\n"
            "Shape: T x P x 4"
        ),
        facecolor="#DDEFD9",
        fontsize=7.7,
    )
    _arrow(ax, (0.828, 0.715), (0.857, 0.715))

    ax.text(
        0.5,
        0.49,
        (
            "All conditions share the same canonical layout; inactive inputs are zeroed "
            "before normalization."
        ),
        ha="center",
        va="center",
        fontsize=8,
        color=MID_GREY,
    )
    ax.text(0.5, 0.42, "Participant-wise causal VAP", ha="center", fontsize=9.5, fontweight="bold")
    ax.plot([0.07, 0.93], [0.39, 0.39], color=LIGHT_GREY, linewidth=0.8)
    _box(
        ax,
        0.10,
        0.15,
        0.22,
        0.14,
        "A: global timing evidence\n(slot-symmetric predictions)",
        facecolor="#E2F0F8",
        edgecolor=BLUE,
        fontsize=8.2,
    )
    _box(
        ax,
        0.39,
        0.15,
        0.22,
        0.14,
        "H: participant identity cues\nfrom causal head pose",
        facecolor="#FFF1D6",
        edgecolor=ORANGE,
        fontsize=8.2,
    )
    _box(
        ax,
        0.68,
        0.15,
        0.22,
        0.14,
        "A+H: timing + identity cues\n(same model capacity)",
        facecolor="#DDEFD9",
        edgecolor=GREEN,
        fontsize=8.2,
    )
    ax.text(
        0.5,
        0.045,
        (
            "32 raw columns -> 33 model dimensions after DoA sin/cos; each token sees "
            f"{participant_dims} participant + {global_dims} global dimensions. "
            "Auxiliary current-VAD loss weight = 0.0; only the four-bin tensor is exported."
        ),
        ha="center",
        va="center",
        fontsize=8,
        color=DARK,
        wrap=True,
    )
    fig.suptitle("Participant-structured VAP architecture and primary input conditions", y=0.97)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.93)
    return fig


def _save_figure(fig: Figure, output_dir: Path, stem: str, dpi: int) -> dict[str, Any]:
    if isinstance(dpi, bool) or dpi < 300:
        raise ValueError("PNG output must use at least 300 dpi")
    output_dir.mkdir(parents=True, exist_ok=True)
    created = FIXED_ARTIFACT_TIME
    formats = {
        "pdf": {
            "Title": stem.replace("_", " "),
            "Author": "Yixuan Zhuang",
            "Creator": "tools/generate_thesis_figures.py",
            "CreationDate": created,
            "ModDate": created,
        },
        "png": {
            "Title": stem.replace("_", " "),
            "Author": "Yixuan Zhuang",
            "Software": "tools/generate_thesis_figures.py",
            "Creation Time": created.isoformat(),
        },
    }
    result: dict[str, Any] = {}
    for suffix, metadata in formats.items():
        destination = output_dir / f"{stem}.{suffix}"
        temporary = output_dir / f".{stem}.tmp.{suffix}"
        fig.savefig(
            temporary,
            format=suffix,
            dpi=dpi if suffix == "png" else None,
            facecolor=WHITE,
            metadata=metadata,
        )
        os.replace(temporary, destination)
        result[suffix] = {
            "file": destination.name,
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
    result["size_inches"] = [float(value) for value in fig.get_size_inches()]
    result["png_dpi"] = dpi
    plt.close(fig)
    return result


def _source_bindings(repo_root: Path) -> dict[str, dict[str, Any]]:
    relative_paths = (
        "configs/hpe_delivery_inventory.json",
        "src/hri_turn_taking/data/multiparty.py",
        "src/hri_turn_taking/vap/dataset.py",
        "src/hri_turn_taking/vap/feature_sets.py",
        "src/hri_turn_taking/vap/labels.py",
        "src/hri_turn_taking/vap/model.py",
        "src/hri_turn_taking/vap/outer_experiment.py",
        "tools/generate_thesis_figures.py",
    )
    bindings = {}
    for relative_path in relative_paths:
        path = repo_root / relative_path
        bindings[relative_path] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return bindings


def generate_all_figures(
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    dpi: int = 300,
) -> dict[str, Any]:
    """Generate all four figure pairs and a deterministic provenance manifest."""

    root = (repo_root or _repo_root_from_script()).resolve()
    destination = output_dir.resolve()
    _configure_style()
    records = load_d3_records(root)
    figures = {
        "d3_uncertainty": make_d3_uncertainty_figure(records),
        "transcript_vad_vap_timeline": make_label_timeline_figure(),
        "nested_logo_protocol": make_nested_logo_figure(),
        "participant_vap_architecture": make_architecture_figure(),
    }
    if tuple(figures) != FIGURE_STEMS:
        raise AssertionError("figure implementation differs from the frozen output inventory")

    outputs = {
        stem: _save_figure(figure, destination, stem, dpi)
        for stem, figure in figures.items()
    }
    schematic = label_schematic_data()
    groups = _ordered_groups()
    manifest = {
        "artifact_kind": "vap_dissertation_static_figure_bundle",
        "format_version": FIGURE_FORMAT_VERSION,
        "generator": "tools/generate_thesis_figures.py",
        "fixed_artifact_time_utc": FIXED_ARTIFACT_TIME.isoformat(),
        "render_environment": {"matplotlib": matplotlib.__version__},
        "source_bindings": _source_bindings(root),
        "d3": {
            "records": [
                {
                    "session_id": record.session_id,
                    "group_id": record.group_id,
                    "offset_s": record.offset_s,
                    "quality_status": record.quality_status,
                    "error_bound_ms": record.error_bound_ms,
                }
                for record in records
            ],
            "thresholds_ms": [40, 80],
            "paired_le80_session_ids": list(PAIRED_D3_80MS_SESSION_IDS),
            "descriptive_le40_session_ids": list(D3_40MS_DESCRIPTIVE_SESSION_IDS),
        },
        "label_schematic": {
            "frame_rate_hz": DEFAULT_LABEL_CONFIG.frame_rate_hz,
            "half_open_intervals": True,
            "bin_edges_s": list(DEFAULT_LABEL_CONFIG.bin_edges_s),
            "active_ratio": DEFAULT_LABEL_CONFIG.active_ratio,
            "labels_at_t0": np.asarray(schematic["labels_at_t0"], dtype=int).tolist(),
            "illustrative_not_corpus_sample": True,
        },
        "nested_logo": {
            "groups": list(groups),
            "outer_folds": [
                {
                    "test_group": test_group,
                    "development_groups": [group for group in groups if group != test_group],
                }
                for test_group in groups
            ],
            "seeds": list(FROZEN_SEEDS),
            "seed_role": "protocol_allowed_values",
            "deadline_executed_seeds": [13],
        },
        "architecture": {
            "conditions": {
                "A": "audio_only",
                "H": "head_pose_only",
                "A+H": "audio_plus_head_pose",
            },
            "raw_canonical_columns": 32,
            "transformed_model_dimensions": 33,
            "participant_dimensions_per_token": len(PARTICIPANT_FEATURE_NAMES),
            "global_dimensions_per_token": len(TRANSFORMED_GLOBAL_FEATURES),
            "n_bins": DEFAULT_LABEL_CONFIG.n_bins,
        },
        "outputs": outputs,
    }
    manifest_path = destination / "thesis_figures.manifest.json"
    temporary = destination / ".thesis_figures.manifest.tmp.json"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = _repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Generate four static, provenance-bound thesis figure pairs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "report" / "Figures",
        help="Destination directory (default: report/Figures).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=root,
        help="Repository root containing configs/ and src/.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (minimum 300).")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = generate_all_figures(
        args.output_dir,
        repo_root=args.repo_root,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "artifact_kind": manifest["artifact_kind"],
                "output_dir": str(args.output_dir.resolve()),
                "figures": list(manifest["outputs"]),
                "manifest": str((args.output_dir / "thesis_figures.manifest.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
