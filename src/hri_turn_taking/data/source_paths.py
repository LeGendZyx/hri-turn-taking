"""Operator-controlled roots for private, non-versioned source data."""

from __future__ import annotations

import os
from pathlib import Path


def configured_path(environment_variable: str, fallback: Path) -> Path:
    """Return an absolute operator-configured path without consulting manifests."""

    value = os.environ.get(environment_variable)
    if value is None or not value.strip():
        return fallback.expanduser().resolve()
    return Path(value).expanduser().resolve()


def corpus_root(project_root: Path) -> Path:
    """Return the private corpus root selected by the local operator."""

    return configured_path("HRI_CORPUS_ROOT", project_root / "data" / "raw")


def hpe_delivery_root(project_root: Path) -> Path:
    """Return the local head-pose delivery root."""

    return configured_path(
        "HRI_HPE_DELIVERY_ROOT",
        project_root.parent / "Head_pose_extraction" / "outputs" / "delivery",
    )
