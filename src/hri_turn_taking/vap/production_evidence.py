"""Read-only production-evidence foundations for future counted VAP runs.

Nothing in this module publishes an artifact, registers a backend, opens the
counted-result gate, or trains a model.  It captures and freshly replays source
and environment state, defines a provenance-backed real 25 Hz tick authority,
and implements a pure D3 pose-timing sensitivity transform for fixtures and
already-loaded tables.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from hri_turn_taking.data import analysis_base, assembly
from hri_turn_taking.vap.feature_sets import active_feature_columns

FORMAT_VERSION = 1
COUNTED_ELIGIBLE = False
SOURCE_STATE_KIND = "vap_production_source_state"
ENVIRONMENT_KIND = "vap_production_environment"
REAL_TICK_AUTHORITY_KIND = "vap_real_tick_authority_snapshot"
D3_CONTRACT_KIND = "vap_d3_pose_reassembly_contract"
DATAFRAME_DIGEST_METHOD = "vap_canonical_dataframe_sha256_v1"

EVIDENCE_BLOCKERS = (
    "owner_filesystem_trust_decision_pending",
    "atomic_formal_run_index_not_implemented",
    "formal_data_readiness_not_authorized",
    "publisher_intentionally_absent",
    "counted_backend_not_registered",
)

ROOT_SOURCE_FILES = ("AGENTS.md", "pyproject.toml")
SOURCE_TREE_ROOTS = ("configs", "src")
EXPLICIT_PROTOCOL_DOCS = (
    "docs/PROJECT_SCOPE.md",
    "docs/IDENTITY_DATA_STATUS.md",
    "docs/TRAINING_DATA_STATUS.md",
    "docs/final_model_plan.md",
    "docs/methodology.md",
    "docs/experiment_plan.md",
    "docs/analysis_base_contract.md",
    "docs/acoustic_feature_contract.md",
    "docs/audio_data_contract.md",
    "docs/audio_gold_alignment_contract.md",
    "docs/gold_vad_candidate_contract.md",
    "docs/hpe_transcript_crosswalk.md",
    "docs/transcript_manual_review.md",
    "docs/production_backend_contract.md",
    "docs/production_evidence_contract.md",
)
FORBIDDEN_SOURCE_COMPONENTS = frozenset(
    {"data", "models", "model", "logs", "log", "report", "reports", "outputs", "output"}
)
_SRC_SUFFIXES = frozenset({".py", ".pyi"})
_CONFIG_SUFFIXES = frozenset({".json", ".md", ".toml", ".yaml", ".yml"})
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ProductionEvidenceGateError(ValueError):
    """A source, environment, time-axis, or D3 evidence gate failed."""


def _fail(message: str) -> None:
    raise ProductionEvidenceGateError(message)


def _base_payload(kind: str) -> dict[str, object]:
    return {
        "artifact_kind": kind,
        "format_version": FORMAT_VERSION,
        "evidence_role": "in_memory_foundation_not_a_formal_result",
        "counted_eligible": False,
        "publisher_available": False,
        "hard_blockers": list(EVIDENCE_BLOCKERS),
    }


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionEvidenceGateError(f"value is not strict JSON: {exc}") from exc


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        qualifier = f" >= {minimum}" if minimum is not None else ""
        _fail(f"{name} must be an exact integer{qualifier}")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _normalised_string(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be a non-empty normalised string")
    return value


def _sha256_string(value: object, name: str) -> str:
    value = _normalised_string(value, name)
    if not _SHA_RE.fullmatch(value):
        _fail(f"{name} must be a lowercase SHA-256 string")
    return value


def _sorted_unique_strings(value: object, name: str) -> list[str]:
    if type(value) not in {list, tuple}:
        _fail(f"{name} must be an exact list/tuple of strings")
    result = [_normalised_string(item, name) for item in value]
    if len(result) != len(set(result)):
        _fail(f"{name} cannot contain duplicates")
    return sorted(result)


def _same_json(left: object, right: object) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _validate_base(payload: dict[str, object], kind: str) -> None:
    expected = _base_payload(kind)
    for key, value in expected.items():
        if key not in payload or not _same_json(payload[key], value):
            _fail(f"{kind} base field {key!r} differs from the code-owned gate")


def _safe_relative_path(value: object, name: str) -> str:
    path = _normalised_string(value, name)
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or ".." in pure.parts
        or "\\" in path
        or path.startswith("./")
    ):
        _fail(f"{name} must be one canonical repository-relative POSIX path")
    if pure.parts[0].lower() in FORBIDDEN_SOURCE_COMPONENTS:
        _fail(f"{name} enters a forbidden raw/output tree")
    return path


def _validate_source_scope_path(value: object) -> str:
    path = _safe_relative_path(value, "source relative_path")
    pure = PurePosixPath(path)
    if path in ROOT_SOURCE_FILES or path in EXPLICIT_PROTOCOL_DOCS:
        return path
    if pure.parts[0] == "src" and pure.suffix.lower() in _SRC_SUFFIXES:
        return path
    if pure.parts[0] == "configs" and pure.suffix.lower() in _CONFIG_SUFFIXES:
        return path
    _fail(f"source path is outside the code-owned scope: {path}")


def _source_parent_directories(paths: list[str]) -> list[str]:
    directories: set[str] = set()
    for path in paths:
        parent = PurePosixPath(path).parent
        if parent.as_posix() == ".":
            directories.add(".")
            continue
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return sorted(directories)


def _entry_is_forbidden_reparse(path: Path, info: os.stat_result) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & reparse_flag)


def _safe_entry_stat(path: Path, name: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProductionEvidenceGateError(f"cannot inspect {name}: {path}: {exc}") from exc
    if _entry_is_forbidden_reparse(path, info):
        _fail(f"{name} is a symlink, junction, or Windows reparse point: {path}")
    return info


def _safe_repository_root(repository_root: str | Path) -> Path:
    supplied = Path(repository_root)
    info = _safe_entry_stat(supplied, "repository root")
    if not stat.S_ISDIR(info.st_mode):
        _fail("repository root must be a regular non-reparse directory")
    return supplied.resolve(strict=True)


def _declared_source_entry(root: Path, relative: str) -> tuple[Path, os.stat_result]:
    declared = _validate_source_scope_path(relative)
    current = root
    info: os.stat_result | None = None
    for part in PurePosixPath(declared).parts:
        current = current / part
        info = _safe_entry_stat(current, f"declared source segment for {declared}")
    if info is None:
        _fail("declared source path cannot be empty")
    return current, info


def _stable_file_record(
    root: Path,
    path: Path,
    *,
    declared_relative: str,
) -> dict[str, object]:
    declared_relative = _validate_source_scope_path(declared_relative)
    before = _safe_entry_stat(path, "source file")
    if not stat.S_ISREG(before.st_mode):
        _fail(f"source inventory accepts regular files only: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        _fail(f"source file escapes repository root: {path}")
    if relative != declared_relative:
        _fail(
            "resolved source path differs from its declared canonical path: "
            f"{declared_relative} -> {relative}"
        )
    digest = _sha256_file(path)
    after = _safe_entry_stat(path, "source file")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        _fail(f"source file changed while being hashed: {relative}")
    return {
        "relative_path": declared_relative,
        "size_bytes": before.st_size,
        "sha256": digest,
    }


def _source_scope_declaration() -> dict[str, object]:
    return {
        "root_files": list(ROOT_SOURCE_FILES),
        "tree_roots": list(SOURCE_TREE_ROOTS),
        "explicit_protocol_docs": list(EXPLICIT_PROTOCOL_DOCS),
        "git_command_used": False,
        "dirty_and_untracked_source_bytes_in_scope": True,
        "forbidden_components": sorted(FORBIDDEN_SOURCE_COMPONENTS),
    }


def _source_paths(root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    paths: list[tuple[Path, str]] = []
    directories = set(
        _source_parent_directories([*ROOT_SOURCE_FILES, *EXPLICIT_PROTOCOL_DOCS])
    )
    for relative in (*ROOT_SOURCE_FILES, *EXPLICIT_PROTOCOL_DOCS):
        path, info = _declared_source_entry(root, relative)
        if not stat.S_ISREG(info.st_mode):
            _fail(f"required code/protocol source is missing or unsafe: {relative}")
        paths.append((path, relative))
    for tree_name in SOURCE_TREE_ROOTS:
        tree = root / tree_name
        tree_info = _safe_entry_stat(tree, "source tree root")
        if not stat.S_ISDIR(tree_info.st_mode):
            _fail(f"required source tree is missing or unsafe: {tree_name}")
        directories.add(tree_name)
        allowed = _SRC_SUFFIXES if tree_name == "src" else _CONFIG_SUFFIXES
        pending = [tree]
        while pending:
            directory = pending.pop()
            try:
                entries = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise ProductionEvidenceGateError(
                    f"cannot enumerate source directory {directory}: {exc}"
                ) from exc
            child_directories = []
            for path in entries:
                info = _safe_entry_stat(path, "source tree entry")
                if stat.S_ISDIR(info.st_mode):
                    if path.name == "__pycache__":
                        continue
                    directories.add(path.relative_to(root).as_posix())
                    child_directories.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    _fail(f"source tree contains a non-regular entry: {path}")
                if path.suffix.lower() not in allowed:
                    _fail(f"source tree contains a non-code/config suffix: {path}")
                paths.append((path, path.relative_to(root).as_posix()))
            pending.extend(reversed(child_directories))
    resolved = [path.resolve() for path, _relative in paths]
    if len(resolved) != len(set(resolved)):
        _fail("source scope contains duplicate physical paths")
    ordered = sorted(paths, key=lambda item: item[1])
    directories.update(
        _source_parent_directories([relative for _path, relative in ordered])
    )
    return ordered, sorted(directories)


def build_source_state_snapshot(repository_root: str | Path) -> dict[str, object]:
    """Hash the live source bytes directly, including dirty/untracked code."""
    root = _safe_repository_root(repository_root)
    paths_before, directories_before = _source_paths(root)
    records_before = [
        _stable_file_record(root, path, declared_relative=relative)
        for path, relative in paths_before
    ]
    paths_after, directories_after = _source_paths(root)
    records_after = [
        _stable_file_record(root, path, declared_relative=relative)
        for path, relative in paths_after
    ]
    if directories_before != directories_after or not _same_json(records_before, records_after):
        _fail("source tree changed during the complete inventory capture")
    records = records_after
    directories = directories_after
    inventory = {
        "scope": _source_scope_declaration(),
        "directories": directories,
        "files": records,
    }
    return {
        **_base_payload(SOURCE_STATE_KIND),
        "repository_root": str(root),
        **inventory,
        "inventory_sha256": _json_sha256(inventory),
    }


def validate_source_state_snapshot(
    payload: object,
    *,
    repository_root: str | Path,
    fresh_replay: bool = True,
) -> dict[str, object]:
    if type(fresh_replay) is not bool:
        _fail("fresh_replay must be an exact boolean")
    expected_keys = {
        *set(_base_payload(SOURCE_STATE_KIND)),
        "repository_root",
        "scope",
        "directories",
        "files",
        "inventory_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        _fail("source-state fields are not exact")
    _validate_base(payload, SOURCE_STATE_KIND)
    root = _safe_repository_root(repository_root)
    if payload["repository_root"] != str(root):
        _fail("source-state repository root differs from the live root")
    expected_scope = _source_scope_declaration()
    if not _same_json(payload["scope"], expected_scope):
        _fail("source-state scope differs from the code-owned scope")
    directories = payload["directories"]
    files = payload["files"]
    if (
        type(directories) is not list
        or any(type(directory) is not str for directory in directories)
        or directories != sorted(set(directories))
    ):
        _fail("source-state directory set must be sorted and exact")
    if type(files) is not list or not files:
        _fail("source-state file inventory cannot be empty")
    paths = []
    for record in files:
        if type(record) is not dict or set(record) != {"relative_path", "size_bytes", "sha256"}:
            _fail("source-state file record fields are not exact")
        paths.append(_validate_source_scope_path(record["relative_path"]))
        _exact_int(record["size_bytes"], "source size_bytes", minimum=0)
        _sha256_string(record["sha256"], "source sha256")
    if paths != sorted(set(paths)):
        _fail("source-state file paths must be sorted and unique")
    live_paths, live_directories = _source_paths(root)
    live_relative_paths = [relative for _path, relative in live_paths]
    if paths != live_relative_paths:
        _fail("source-state file set differs from the live code-owned scope")
    if directories != live_directories:
        _fail("source-state directory set differs from the live code-owned scope")
    inventory = {"scope": payload["scope"], "directories": directories, "files": files}
    if payload["inventory_sha256"] != _json_sha256(inventory):
        _fail("source-state inventory hash mismatch")
    if fresh_replay:
        fresh = build_source_state_snapshot(root)
        if not _same_json(payload, fresh):
            _fail("source-state differs from fresh filesystem replay")
    return payload


def _canonical_distribution_inventory() -> list[dict[str, str]]:
    records = []
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            _fail("installed distribution has no canonical Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = distribution.version
        _normalised_string(name, "distribution name")
        _normalised_string(version, f"distribution {name} version")
        records.append({"name": name, "version": version})
    records.sort(key=lambda item: (item["name"], item["version"]))
    if len(records) != len({record["name"] for record in records}):
        _fail("installed distribution inventory contains duplicate canonical names")
    return records


def _physical_cpu_count() -> tuple[int | None, str]:
    if importlib.util.find_spec("psutil") is None:
        return None, "unavailable_psutil_not_installed"
    import psutil

    value = psutil.cpu_count(logical=False)
    if value is None:
        return None, "unavailable_from_psutil"
    return int(value), "psutil"


def build_environment_snapshot() -> dict[str, object]:
    """Capture the live interpreter/runtime; callers cannot supply fields."""
    import numpy
    import pandas

    physical, provider = _physical_cpu_count()
    logical = os.cpu_count()
    if logical is None or logical < 1:
        _fail("logical CPU count is unavailable")
    inventory = _canonical_distribution_inventory()
    runtime = {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "version_info": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
        },
        "packages": {
            "torch": str(torch.__version__),
            "numpy": str(numpy.__version__),
            "pandas": str(pandas.__version__),
            "scikit_learn": str(sklearn.__version__),
        },
        "cpu": {
            "identifier": platform.processor() or platform.machine(),
            "physical_core_count": physical,
            "physical_count_provider": provider,
            "logical_core_count": logical,
        },
        "torch_runtime": {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "cuda_available": torch.cuda.is_available(),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "installed_distributions": inventory,
        "distribution_inventory_sha256": _json_sha256(inventory),
        "inventory_method": "importlib.metadata_no_pip_subprocess",
    }
    return {
        **_base_payload(ENVIRONMENT_KIND),
        **runtime,
        "environment_sha256": _json_sha256(runtime),
    }


def validate_environment_snapshot(
    payload: object, *, fresh_replay: bool = True
) -> dict[str, object]:
    if type(fresh_replay) is not bool:
        _fail("fresh_replay must be an exact boolean")
    expected = {
        *set(_base_payload(ENVIRONMENT_KIND)),
        "python",
        "operating_system",
        "packages",
        "cpu",
        "torch_runtime",
        "installed_distributions",
        "distribution_inventory_sha256",
        "inventory_method",
        "environment_sha256",
    }
    if type(payload) is not dict or set(payload) != expected:
        _fail("environment snapshot fields are not exact")
    _validate_base(payload, ENVIRONMENT_KIND)
    nested_keys = {
        "python": {"executable", "version", "implementation", "version_info"},
        "operating_system": {"system", "release", "version", "machine", "architecture"},
        "packages": {"torch", "numpy", "pandas", "scikit_learn"},
        "cpu": {
            "identifier",
            "physical_core_count",
            "physical_count_provider",
            "logical_core_count",
        },
        "torch_runtime": {
            "num_threads",
            "num_interop_threads",
            "deterministic_algorithms",
            "deterministic_warn_only",
            "mkldnn_enabled",
            "cuda_available",
            "default_dtype",
        },
    }
    for key, keys in nested_keys.items():
        if type(payload[key]) is not dict or set(payload[key]) != keys:
            _fail(f"environment {key} fields are not exact")
    python = payload["python"]
    for key in ("executable", "version", "implementation"):
        _normalised_string(python[key], f"python {key}")
    if (
        type(python["version_info"]) is not list
        or len(python["version_info"]) != 3
        or any(type(value) is not int or value < 0 for value in python["version_info"])
    ):
        _fail("python version_info must contain three exact non-negative integers")
    for key, value in payload["operating_system"].items():
        _normalised_string(value, f"operating_system {key}")
    for key, value in payload["packages"].items():
        _normalised_string(value, f"package {key}")
    cpu = payload["cpu"]
    _normalised_string(cpu["identifier"], "cpu identifier")
    _normalised_string(cpu["physical_count_provider"], "physical_count_provider")
    if cpu["physical_core_count"] is not None:
        _exact_int(cpu["physical_core_count"], "physical_core_count", minimum=1)
    _exact_int(cpu["logical_core_count"], "logical_core_count", minimum=1)
    runtime = payload["torch_runtime"]
    for key in ("num_threads", "num_interop_threads"):
        _exact_int(runtime[key], f"torch_runtime {key}", minimum=1)
    for key in (
        "deterministic_algorithms",
        "deterministic_warn_only",
        "mkldnn_enabled",
        "cuda_available",
    ):
        if type(runtime[key]) is not bool:
            _fail(f"torch_runtime {key} must be a boolean")
    _normalised_string(runtime["default_dtype"], "torch default_dtype")
    distributions = payload["installed_distributions"]
    if type(distributions) is not list:
        _fail("installed_distributions must be a list")
    previous = None
    for record in distributions:
        if type(record) is not dict or set(record) != {"name", "version"}:
            _fail("distribution record fields are not exact")
        current = (
            _normalised_string(record["name"], "distribution name"),
            _normalised_string(record["version"], "distribution version"),
        )
        if current[0] != re.sub(r"[-_.]+", "-", current[0]).lower():
            _fail("distribution name is not canonical")
        if previous is not None and (current <= previous or current[0] == previous[0]):
            _fail("distribution inventory must be canonically sorted and unique")
        previous = current
    if payload["distribution_inventory_sha256"] != _json_sha256(distributions):
        _fail("distribution inventory hash mismatch")
    if payload["inventory_method"] != "importlib.metadata_no_pip_subprocess":
        _fail("environment inventory method differs from the code-owned method")
    runtime_payload = {
        key: payload[key]
        for key in (
            "python",
            "operating_system",
            "packages",
            "cpu",
            "torch_runtime",
            "installed_distributions",
            "distribution_inventory_sha256",
            "inventory_method",
        )
    }
    if payload["environment_sha256"] != _json_sha256(runtime_payload):
        _fail("environment hash mismatch")
    if fresh_replay and not _same_json(payload, build_environment_snapshot()):
        _fail("environment snapshot differs from fresh live replay")
    return payload


def _selected_analysis_authority(
    authority: analysis_base.AnalysisBaseAuthority | None,
) -> analysis_base.AnalysisBaseAuthority:
    if authority is None:
        return analysis_base.default_analysis_base_authority()
    if type(authority) is not analysis_base.AnalysisBaseAuthority:
        _fail("analysis_base_authority has the wrong exact type")
    return authority


def _feature_set_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        _fail("feature_sets must be a non-empty exact tuple")
    result = tuple(_normalised_string(item, "feature_set") for item in value)
    if result != tuple(sorted(set(result))):
        _fail("feature_sets must be sorted and unique")
    for feature_set in result:
        try:
            active_feature_columns(feature_set)
        except ValueError as exc:
            raise ProductionEvidenceGateError(str(exc)) from exc
    return result


def _digest_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _canonical_dataframe_state(frame: pd.DataFrame) -> dict[str, object]:
    if type(frame) is not pd.DataFrame or frame.empty:
        _fail("canonical dataframe must be one non-empty exact pandas DataFrame")
    columns = frame.columns.tolist()
    if any(type(column) is not str for column in columns) or len(columns) != len(
        set(columns)
    ):
        _fail("canonical dataframe columns must be unique exact strings")
    dtypes = [str(frame[column].dtype) for column in columns]
    for dtype in dtypes:
        _normalised_string(dtype, "canonical dataframe dtype")
    metadata = {
        "method": DATAFRAME_DIGEST_METHOD,
        "n_rows": len(frame),
        "columns": columns,
        "dtypes": dtypes,
    }
    digest = hashlib.sha256()
    _digest_length_prefixed(digest, _canonical_json(metadata))
    for column in columns:
        series = frame[column]
        kind = getattr(series.dtype, "kind", None)
        if kind in {"b", "i", "u", "f"}:
            try:
                numpy_dtype = np.dtype(series.dtype).newbyteorder("<")
                values = series.to_numpy(dtype=numpy_dtype, copy=True)
            except (TypeError, ValueError) as exc:
                raise ProductionEvidenceGateError(
                    f"canonical dataframe column {column!r} has an unsupported dtype"
                ) from exc
            if values.ndim != 1 or len(values) != len(frame):
                _fail(f"canonical dataframe column {column!r} is not one-dimensional")
            if kind == "f" and not np.isfinite(values).all():
                _fail(f"canonical dataframe column {column!r} contains NaN/Inf")
            payload = np.ascontiguousarray(values, dtype=numpy_dtype).tobytes(order="C")
            _digest_length_prefixed(digest, payload)
            continue
        if kind == "O":
            values = series.to_numpy(dtype=object, copy=True)
            for value in values:
                if type(value) is not str:
                    _fail(
                        f"canonical dataframe object column {column!r} "
                        "must contain exact strings"
                    )
                _digest_length_prefixed(digest, value.encode("utf-8"))
            continue
        _fail(f"canonical dataframe column {column!r} has unsupported dtype {series.dtype}")
    return {
        "dataframe_digest_method": DATAFRAME_DIGEST_METHOD,
        "dataframe_n_rows": len(frame),
        "dataframe_columns": columns,
        "dataframe_dtypes": dtypes,
        "dataframe_content_sha256": digest.hexdigest(),
    }


def _validate_dataframe_state(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("dataframe_digest_method") != DATAFRAME_DIGEST_METHOD:
        _fail("canonical dataframe digest method mismatch")
    n_rows = _exact_int(payload.get("dataframe_n_rows"), "dataframe_n_rows", minimum=1)
    columns = payload.get("dataframe_columns")
    dtypes = payload.get("dataframe_dtypes")
    if (
        type(columns) is not list
        or not columns
        or any(type(column) is not str for column in columns)
        or len(columns) != len(set(columns))
    ):
        _fail("canonical dataframe columns must be a non-empty unique string list")
    if (
        type(dtypes) is not list
        or len(dtypes) != len(columns)
        or any(type(dtype) is not str or not dtype for dtype in dtypes)
    ):
        _fail("canonical dataframe dtypes must exactly cover the columns")
    digest = _sha256_string(
        payload.get("dataframe_content_sha256"),
        "canonical dataframe content hash",
    )
    return {
        "dataframe_digest_method": DATAFRAME_DIGEST_METHOD,
        "dataframe_n_rows": n_rows,
        "dataframe_columns": columns,
        "dataframe_dtypes": dtypes,
        "dataframe_content_sha256": digest,
    }


def _exact_ticks(frame: pd.DataFrame, scene: str) -> np.ndarray:
    if "session_id" not in frame or "timestamp_s" not in frame:
        _fail("real tick authority requires session_id and timestamp_s")
    session_ids = frame["session_id"].astype(str).unique().tolist()
    if session_ids != [scene]:
        _fail("real tick authority dataframe/session provenance disagree")
    timestamps = frame["timestamp_s"].to_numpy(dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 1 or not np.isfinite(timestamps).all():
        _fail("real tick authority timestamps must be finite and non-empty")
    ticks = np.rint(timestamps * 25.0).astype(np.int64)
    if not np.array_equal(timestamps, ticks.astype(np.float64) / 25.0):
        _fail("real timestamps are not exact 25 Hz tick values")
    if ticks[0] < 0 or not np.array_equal(np.diff(ticks), np.ones(len(ticks) - 1, dtype=int)):
        _fail("real tick authority requires contiguous non-negative 25 Hz ticks")
    if "tick_25hz" in frame:
        declared = frame["tick_25hz"].to_numpy()
        if declared.dtype.kind not in "iu":
            _fail("real tick_25hz must use an exact integer dtype")
        if declared.dtype.kind == "u" and declared.size and declared.max() > np.iinfo(np.int64).max:
            _fail("real tick_25hz exceeds signed 64-bit range")
        if not np.array_equal(declared.astype(np.int64), ticks):
            _fail("real tick_25hz differs from timestamp_s * 25")
    return ticks


def _canonical_real_csv(
    scene: str,
    artifact_kind: str,
    path: Path,
    authority: analysis_base.AnalysisBaseAuthority,
) -> None:
    if artifact_kind == analysis_base.ARTIFACT_KIND:
        expected = (authority.analysis_base_root / f"{scene}.csv").resolve()
        if path != expected:
            _fail("analysis-base path differs from the code-owned canonical path")
    elif artifact_kind == assembly.REAL_ASSEMBLED_ARTIFACT_KIND:
        if path not in authority.descendant_output_paths:
            _fail("assembled path is absent from code-owned descendant authority")
    else:
        _fail("real tick authority accepts analysis-base/assembled artifacts only")


def _offset_quality(provenance: dict[str, Any]) -> dict[str, object]:
    quality = provenance.get("offset_quality")
    if type(quality) is not dict:
        contract = provenance.get("analysis_window_contract")
        quality = contract.get("d3_quality") if type(contract) is dict else None
    expected = {
        "stream",
        "offset_s",
        "self_check_ok",
        "convention",
        "status",
        "error_bound_ms",
        "timebase_keys",
    }
    if type(quality) is not dict or set(quality) != expected:
        _fail("real tick authority requires the exact D3 offset-quality summary")
    if quality["stream"] != "6cam" or quality["self_check_ok"] is not True:
        _fail("D3 quality must bind a self-checked 6cam stream")
    _finite_number(quality["offset_s"], "D3 offset_s")
    bound = _finite_number(quality["error_bound_ms"], "D3 error_bound_ms")
    if bound < 0.0:
        _fail("D3 error_bound_ms cannot be negative")
    if quality["status"] not in {"ok", "degraded"}:
        _fail("D3 status must be ok/degraded")
    _normalised_string(quality["convention"], "D3 convention")
    if type(quality["timebase_keys"]) is not dict or "6cam" not in quality["timebase_keys"]:
        _fail("D3 timebase_keys must contain 6cam")
    for key, value in quality["timebase_keys"].items():
        _normalised_string(key, "D3 timebase key")
        _normalised_string(value, f"D3 timebase value for {key}")
    return json.loads(json.dumps(quality, allow_nan=False))


def _real_tick_record(
    csv_path: Path,
    *,
    feature_sets: tuple[str, ...],
    authority: analysis_base.AnalysisBaseAuthority,
) -> dict[str, object]:
    sidecar = csv_path.with_suffix(".provenance.json")
    if (
        not csv_path.is_file()
        or csv_path.is_symlink()
        or not sidecar.is_file()
        or sidecar.is_symlink()
    ):
        _fail("real tick input must be one regular CSV/sidecar pair")
    data_before = _sha256_file(csv_path)
    provenance_before = _sha256_file(sidecar)
    data_size_before = csv_path.stat().st_size
    provenance_size_before = sidecar.stat().st_size
    try:
        loaded = assembly._load_dataset_file(
            csv_path,
            data_kind="real",
            analysis_base_authority=authority,
        )
    except (OSError, ValueError) as exc:
        raise ProductionEvidenceGateError(f"real dataset replay failed: {exc}") from exc
    provenance = loaded.provenance
    if type(provenance) is not dict:
        _fail("real tick input has no validated provenance")
    scene = _normalised_string(provenance.get("session_id"), "real session_id")
    group = _normalised_string(provenance.get("group_id"), "real group_id")
    artifact_kind = _normalised_string(provenance.get("artifact_kind"), "artifact_kind")
    format_version = _exact_int(
        provenance.get("format_version"),
        "real provenance format_version",
        minimum=1,
    )
    _canonical_real_csv(scene, artifact_kind, csv_path, authority)
    ticks = _exact_ticks(loaded.frame_df, scene)
    dataframe_state = _canonical_dataframe_state(loaded.frame_df)
    if dataframe_state["dataframe_n_rows"] != len(ticks):
        _fail("canonical dataframe row count differs from its tick sequence")
    contract = provenance.get("analysis_window_contract")
    tick_range = contract.get("authorized_tick_range") if type(contract) is dict else None
    if type(tick_range) is not dict:
        _fail("real tick input has no analysis authorized_tick_range")
    frame_rate_hz = _exact_int(
        tick_range.get("frame_rate_hz"),
        "authorized frame_rate_hz",
        minimum=1,
    )
    start_inclusive = _exact_int(
        tick_range.get("start_inclusive"),
        "authorized start_inclusive",
        minimum=0,
    )
    end_exclusive = _exact_int(
        tick_range.get("end_exclusive"),
        "authorized end_exclusive",
        minimum=1,
    )
    n_ticks = _exact_int(
        tick_range.get("n_ticks"),
        "authorized n_ticks",
        minimum=1,
    )
    if (
        frame_rate_hz != 25
        or start_inclusive != int(ticks[0])
        or end_exclusive != int(ticks[-1]) + 1
        or n_ticks != len(ticks)
    ):
        _fail("real dataframe ticks differ from analysis-window provenance")
    readiness = {}
    target_report = assembly._readiness_from_loaded(
        csv_path, [loaded], data_kind="real", feature_set=None
    )
    if type(target_report.target_ready) is not bool:
        _fail("target readiness replay did not return an exact boolean")
    for feature_set in feature_sets:
        report = assembly._readiness_from_loaded(
            csv_path, [loaded], data_kind="real", feature_set=feature_set
        )
        if type(report.selected_feature_ready) is not bool:
            _fail("selected feature readiness replay did not return an exact boolean")
        readiness[feature_set] = {
            "selected_feature_ready": report.selected_feature_ready,
            "selected_feature_blockers": _sorted_unique_strings(
                report.selected_feature_blockers,
                "selected feature blocker",
            ),
        }
    if _sha256_file(csv_path) != data_before or _sha256_file(sidecar) != provenance_before:
        _fail("real CSV/sidecar changed during tick-authority replay")
    if loaded.data_sha256 != data_before or loaded.provenance_sha256 != provenance_before:
        _fail("real loader hashes disagree with live CSV/sidecar bytes")
    return {
        "scene": scene,
        "group_id": group,
        "artifact_kind": artifact_kind,
        "format_version": format_version,
        "csv_path": str(csv_path),
        "csv_size_bytes": data_size_before,
        "csv_sha256": data_before,
        "provenance_path": str(sidecar.resolve()),
        "provenance_size_bytes": provenance_size_before,
        "provenance_sha256": provenance_before,
        "tick_25hz": ticks.tolist(),
        "tick_start": int(ticks[0]),
        "tick_end_exclusive": int(ticks[-1]) + 1,
        "n_frames": len(ticks),
        **dataframe_state,
        "real_columns": _sorted_unique_strings(
            provenance.get("real_columns"),
            "real column",
        ),
        "placeholder_columns": _sorted_unique_strings(
            provenance.get("placeholder_columns"),
            "placeholder column",
        ),
        "target_ready": target_report.target_ready,
        "target_blockers": _sorted_unique_strings(
            target_report.target_blockers,
            "target blocker",
        ),
        "feature_readiness": readiness,
        "d3_quality": _offset_quality(provenance),
    }


def build_real_tick_authority_snapshot(
    csv_paths: tuple[str | Path, ...],
    *,
    feature_sets: tuple[str, ...],
    authority: analysis_base.AnalysisBaseAuthority | None = None,
) -> dict[str, object]:
    """Replay canonical real CSV/sidecar pairs into one in-memory authority."""
    if type(csv_paths) is not tuple or not csv_paths:
        _fail("csv_paths must be a non-empty exact tuple")
    if any(type(path) is not str and not isinstance(path, Path) for path in csv_paths):
        _fail("every real tick CSV path must be an exact string/Path")
    selected = _selected_analysis_authority(authority)
    features = _feature_set_tuple(feature_sets)
    supplied_paths = tuple(Path(path) for path in csv_paths)
    if any(path.is_symlink() for path in supplied_paths):
        _fail("real tick CSV paths cannot be symlinks")
    paths = tuple(path.resolve() for path in supplied_paths)
    if paths != tuple(sorted(set(paths), key=str)):
        _fail("real tick CSV paths must be resolved, sorted, and unique")
    records = tuple(
        _real_tick_record(path, feature_sets=features, authority=selected) for path in paths
    )
    scenes = tuple(record["scene"] for record in records)
    if scenes != tuple(sorted(set(scenes))):
        _fail("real tick scenes must be sorted and unique")
    inventory = {"feature_sets": list(features), "records": list(records)}
    return {
        **_base_payload(REAL_TICK_AUTHORITY_KIND),
        "time_axis": "mic_A_shared_seconds",
        "frame_rate_hz": 25,
        **inventory,
        "inventory_sha256": _json_sha256(inventory),
    }


def _validate_real_tick_structure(payload: object) -> dict[str, object]:
    expected = {
        *set(_base_payload(REAL_TICK_AUTHORITY_KIND)),
        "time_axis",
        "frame_rate_hz",
        "feature_sets",
        "records",
        "inventory_sha256",
    }
    if type(payload) is not dict or set(payload) != expected:
        _fail("real tick authority fields are not exact")
    _validate_base(payload, REAL_TICK_AUTHORITY_KIND)
    if payload["time_axis"] != "mic_A_shared_seconds" or payload["frame_rate_hz"] != 25:
        _fail("real tick authority axis/rate mismatch")
    feature_sets_list = payload["feature_sets"]
    if type(feature_sets_list) is not list:
        _fail("real tick feature_sets must be a JSON list")
    features = _feature_set_tuple(tuple(feature_sets_list))
    records = payload["records"]
    if type(records) is not list or not records:
        _fail("real tick records must be a non-empty list")
    expected_record = {
        "scene",
        "group_id",
        "artifact_kind",
        "format_version",
        "csv_path",
        "csv_size_bytes",
        "csv_sha256",
        "provenance_path",
        "provenance_size_bytes",
        "provenance_sha256",
        "tick_25hz",
        "tick_start",
        "tick_end_exclusive",
        "n_frames",
        "dataframe_digest_method",
        "dataframe_n_rows",
        "dataframe_columns",
        "dataframe_dtypes",
        "dataframe_content_sha256",
        "real_columns",
        "placeholder_columns",
        "target_ready",
        "target_blockers",
        "feature_readiness",
        "d3_quality",
    }
    scenes = []
    for record in records:
        if type(record) is not dict or set(record) != expected_record:
            _fail("real tick record fields are not exact")
        scene = _normalised_string(record["scene"], "real tick scene")
        scenes.append(scene)
        _normalised_string(record["group_id"], "real tick group_id")
        artifact_kind = _normalised_string(record["artifact_kind"], "real tick artifact_kind")
        if artifact_kind not in {
            analysis_base.ARTIFACT_KIND,
            assembly.REAL_ASSEMBLED_ARTIFACT_KIND,
        }:
            _fail("real tick artifact_kind is not analysis-base/assembled")
        _exact_int(record["format_version"], "real tick format_version", minimum=1)
        for key in ("csv_path", "provenance_path"):
            path = Path(_normalised_string(record[key], key))
            if not path.is_absolute() or path != path.resolve():
                _fail(f"{key} must be an absolute canonical path")
        if Path(record["provenance_path"]) != Path(record["csv_path"]).with_suffix(
            ".provenance.json"
        ):
            _fail("real tick CSV/provenance paths are not one canonical pair")
        for key in ("csv_size_bytes", "provenance_size_bytes"):
            _exact_int(record[key], key, minimum=0)
        _sha256_string(record["csv_sha256"], "csv_sha256")
        _sha256_string(record["provenance_sha256"], "provenance_sha256")
        ticks = record["tick_25hz"]
        if type(ticks) is not list or not ticks or any(type(tick) is not int for tick in ticks):
            _fail("real tick_25hz must be a non-empty exact integer list")
        if ticks[0] < 0 or any(b != a + 1 for a, b in zip(ticks, ticks[1:], strict=False)):
            _fail("real tick_25hz must be contiguous and non-negative")
        tick_start = _exact_int(record["tick_start"], "tick_start", minimum=0)
        tick_end = _exact_int(
            record["tick_end_exclusive"],
            "tick_end_exclusive",
            minimum=1,
        )
        n_frames = _exact_int(record["n_frames"], "n_frames", minimum=1)
        if (
            tick_start != ticks[0]
            or tick_end != ticks[-1] + 1
            or n_frames != len(ticks)
        ):
            _fail("real tick bounds/count disagree with tick_25hz")
        dataframe_state = _validate_dataframe_state(record)
        if dataframe_state["dataframe_n_rows"] != n_frames:
            _fail("canonical dataframe row count differs from real tick n_frames")
        if type(record["target_ready"]) is not bool:
            _fail("target_ready must be a boolean")
        for key in ("real_columns", "placeholder_columns", "target_blockers"):
            values = record[key]
            if type(values) is not list or values != sorted(set(values)):
                _fail(f"{key} must be a sorted unique string list")
            for value in values:
                _normalised_string(value, key)
        if set(record["real_columns"]) & set(record["placeholder_columns"]):
            _fail("real and placeholder column declarations overlap")
        if record["target_ready"] == bool(record["target_blockers"]):
            _fail("target readiness boolean/blockers are inconsistent")
        readiness = record["feature_readiness"]
        if type(readiness) is not dict or list(readiness) != list(features):
            _fail("feature readiness must cover the exact ordered feature sets")
        for _feature_set, value in readiness.items():
            if type(value) is not dict or set(value) != {
                "selected_feature_ready",
                "selected_feature_blockers",
            }:
                _fail("selected feature readiness fields are not exact")
            if type(value["selected_feature_ready"]) is not bool:
                _fail("selected_feature_ready must be a boolean")
            blockers = value["selected_feature_blockers"]
            if type(blockers) is not list or blockers != sorted(set(blockers)):
                _fail("selected_feature_blockers must be a sorted unique list")
            for blocker in blockers:
                _normalised_string(blocker, "selected feature blocker")
            if value["selected_feature_ready"] == bool(blockers):
                _fail("selected feature readiness boolean/blockers are inconsistent")
        _offset_quality({"offset_quality": record["d3_quality"]})
    if scenes != sorted(set(scenes)):
        _fail("real tick records must be sorted and unique by scene")
    inventory = {"feature_sets": feature_sets_list, "records": records}
    if payload["inventory_sha256"] != _json_sha256(inventory):
        _fail("real tick inventory hash mismatch")
    return payload


def validate_real_tick_authority_snapshot(
    payload: object,
    *,
    authority: analysis_base.AnalysisBaseAuthority | None = None,
    fresh_replay: bool = True,
) -> dict[str, object]:
    if type(fresh_replay) is not bool:
        _fail("fresh_replay must be an exact boolean")
    checked = _validate_real_tick_structure(payload)
    if fresh_replay:
        paths = tuple(record["csv_path"] for record in checked["records"])
        features = tuple(checked["feature_sets"])
        fresh = build_real_tick_authority_snapshot(
            paths,
            feature_sets=features,
            authority=authority,
        )
        if not _same_json(checked, fresh):
            _fail("real tick authority differs from fresh CSV/sidecar replay")
    return checked


def build_d3_reassembly_contract(
    tick_authority: object,
    *,
    scene: str,
    feature_set: str,
    authority: analysis_base.AnalysisBaseAuthority | None = None,
    fresh_replay: bool = True,
) -> dict[str, object]:
    """Freeze central/minus/plus semantics from one bound scene's D3 evidence."""
    tick_snapshot = validate_real_tick_authority_snapshot(
        tick_authority,
        authority=authority,
        fresh_replay=fresh_replay,
    )
    scene = _normalised_string(scene, "D3 scene")
    feature_set = _normalised_string(feature_set, "D3 feature_set")
    matches = [record for record in tick_snapshot["records"] if record["scene"] == scene]
    if len(matches) != 1:
        _fail("D3 scene must occur exactly once in the real tick authority")
    record = matches[0]
    if feature_set not in tick_snapshot["feature_sets"]:
        _fail("D3 feature_set was not frozen in the real tick authority")
    try:
        active = active_feature_columns(feature_set)
    except ValueError as exc:
        raise ProductionEvidenceGateError(str(exc)) from exc
    active_yaw = sorted(column for column in active if column.endswith("_head_yaw_deg"))
    shifted = []
    for yaw in active_yaw:
        prefix = yaw.removesuffix("head_yaw_deg")
        companions = (yaw, f"{prefix}pose_valid", f"{prefix}pose_quality")
        if any(column not in active for column in companions):
            _fail("an active head-yaw condition must include its validity/quality channels")
        shifted.extend(companions)
    shifted = sorted(set(shifted))
    bound = _finite_number(record["d3_quality"]["error_bound_ms"], "D3 error_bound_ms")
    mode = "strict_past_only_pose_shift" if active_yaw else "identity_no_active_head_yaw"
    deltas = {
        "central": 0.0,
        "minus": -bound if active_yaw else 0.0,
        "plus": bound if active_yaw else 0.0,
    }
    core = {
        "scene": scene,
        "group_id": record["group_id"],
        "feature_set": feature_set,
        "active_feature_columns": active,
        "active_head_yaw_columns": active_yaw,
        "shifted_pose_columns": shifted,
        "mode": mode,
        "recorded_error_bound_ms": bound,
        "variant_aligned_source_delta_ms": deltas,
        "selection_rule": "latest_shifted_pose_sample_at_or_before_target",
        "no_interpolation": True,
        "no_extrapolation": True,
        "common_output_tick_intersection": True,
        "non_pose_columns_identical_across_variants": True,
        "gold_and_acoustics_identical_across_variants": True,
        "tick_authority_inventory_sha256": tick_snapshot["inventory_sha256"],
        "input_tick_25hz": list(record["tick_25hz"]),
        "source_csv_sha256": record["csv_sha256"],
        "source_provenance_sha256": record["provenance_sha256"],
        "dataframe_digest_method": record["dataframe_digest_method"],
        "dataframe_n_rows": record["dataframe_n_rows"],
        "dataframe_columns": record["dataframe_columns"],
        "dataframe_dtypes": record["dataframe_dtypes"],
        "dataframe_content_sha256": record["dataframe_content_sha256"],
        "d3_quality": record["d3_quality"],
    }
    return {
        **_base_payload(D3_CONTRACT_KIND),
        **core,
        "contract_sha256": _json_sha256(core),
    }


def _validate_d3_contract(payload: object) -> dict[str, object]:
    core_keys = {
        "scene",
        "group_id",
        "feature_set",
        "active_feature_columns",
        "active_head_yaw_columns",
        "shifted_pose_columns",
        "mode",
        "recorded_error_bound_ms",
        "variant_aligned_source_delta_ms",
        "selection_rule",
        "no_interpolation",
        "no_extrapolation",
        "common_output_tick_intersection",
        "non_pose_columns_identical_across_variants",
        "gold_and_acoustics_identical_across_variants",
        "tick_authority_inventory_sha256",
        "input_tick_25hz",
        "source_csv_sha256",
        "source_provenance_sha256",
        "dataframe_digest_method",
        "dataframe_n_rows",
        "dataframe_columns",
        "dataframe_dtypes",
        "dataframe_content_sha256",
        "d3_quality",
    }
    expected = {*set(_base_payload(D3_CONTRACT_KIND)), *core_keys, "contract_sha256"}
    if type(payload) is not dict or set(payload) != expected:
        _fail("D3 contract fields are not exact")
    _validate_base(payload, D3_CONTRACT_KIND)
    for key in ("scene", "group_id", "feature_set"):
        _normalised_string(payload[key], f"D3 {key}")
    active_values = payload["active_feature_columns"]
    if (
        type(active_values) is not list
        or any(type(value) is not str for value in active_values)
        or len(active_values) != len(set(active_values))
    ):
        _fail("D3 active_feature_columns must be a unique canonical list")
    for key in ("active_head_yaw_columns", "shifted_pose_columns"):
        values = payload[key]
        if (
            type(values) is not list
            or any(type(value) is not str for value in values)
            or values != sorted(set(values))
        ):
            _fail(f"D3 {key} must be sorted and unique")
    try:
        active = active_feature_columns(payload["feature_set"])
    except ValueError as exc:
        raise ProductionEvidenceGateError(str(exc)) from exc
    if payload["active_feature_columns"] != active:
        _fail("D3 active columns differ from the code-owned feature set")
    expected_yaw = sorted(column for column in active if column.endswith("_head_yaw_deg"))
    if payload["active_head_yaw_columns"] != expected_yaw:
        _fail("D3 active head-yaw columns differ from the feature set")
    expected_shifted = sorted(
        column
        for yaw in expected_yaw
        for column in (
            yaw,
            f"{yaw.removesuffix('head_yaw_deg')}pose_valid",
            f"{yaw.removesuffix('head_yaw_deg')}pose_quality",
        )
    )
    if payload["shifted_pose_columns"] != expected_shifted:
        _fail("D3 shifted pose columns differ from the active yaw contract")
    bound = _finite_number(payload["recorded_error_bound_ms"], "D3 error bound")
    if bound < 0.0:
        _fail("D3 error bound cannot be negative")
    expected_mode = "strict_past_only_pose_shift" if expected_yaw else "identity_no_active_head_yaw"
    if payload["mode"] != expected_mode:
        _fail("D3 mode differs from active head-yaw status")
    expected_deltas = {
        "central": 0.0,
        "minus": -bound if expected_yaw else 0.0,
        "plus": bound if expected_yaw else 0.0,
    }
    if not _same_json(payload["variant_aligned_source_delta_ms"], expected_deltas):
        _fail("D3 deltas are not exactly bounded by recorded error")
    if payload["selection_rule"] != "latest_shifted_pose_sample_at_or_before_target":
        _fail("D3 selection rule mismatch")
    for key in (
        "no_interpolation",
        "no_extrapolation",
        "common_output_tick_intersection",
        "non_pose_columns_identical_across_variants",
        "gold_and_acoustics_identical_across_variants",
    ):
        if payload[key] is not True:
            _fail(f"D3 {key} must be true")
    _sha256_string(payload["tick_authority_inventory_sha256"], "tick authority hash")
    input_ticks = payload["input_tick_25hz"]
    if (
        type(input_ticks) is not list
        or not input_ticks
        or any(type(tick) is not int for tick in input_ticks)
        or input_ticks[0] < 0
        or any(b != a + 1 for a, b in zip(input_ticks, input_ticks[1:], strict=False))
    ):
        _fail("D3 input_tick_25hz must be contiguous exact integers")
    _sha256_string(payload["source_csv_sha256"], "D3 source CSV hash")
    _sha256_string(payload["source_provenance_sha256"], "D3 source provenance hash")
    dataframe_state = _validate_dataframe_state(payload)
    if dataframe_state["dataframe_n_rows"] != len(input_ticks):
        _fail("D3 dataframe row count differs from input ticks")
    quality = _offset_quality({"offset_quality": payload["d3_quality"]})
    if _finite_number(quality["error_bound_ms"], "D3 quality error bound") != bound:
        _fail("D3 contract bound differs from its offset-quality evidence")
    core = {key: payload[key] for key in core_keys}
    if payload["contract_sha256"] != _json_sha256(core):
        _fail("D3 contract hash mismatch")
    return payload


@dataclass(frozen=True)
class D3ReassemblyResult:
    variants: dict[str, pd.DataFrame]
    audit: dict[str, object]


def _frame_tick_values(frame: pd.DataFrame) -> np.ndarray:
    if "tick_25hz" in frame:
        raw = frame["tick_25hz"].to_numpy()
        if raw.dtype.kind not in "iu":
            _fail("tick_25hz must use an integer dtype")
        if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
            _fail("tick_25hz exceeds signed 64-bit range")
        ticks = raw.astype(np.int64)
    elif "timestamp_s" in frame:
        timestamp = frame["timestamp_s"].to_numpy(dtype=np.float64)
        ticks = np.rint(timestamp * 25.0).astype(np.int64)
        if not np.array_equal(timestamp, ticks.astype(np.float64) / 25.0):
            _fail("D3 timestamps are not exact 25 Hz values")
    else:
        _fail("D3 reassembly requires tick_25hz or timestamp_s")
    if ticks.size < 1 or ticks[0] < 0 or not np.array_equal(
        np.diff(ticks), np.ones(ticks.size - 1, dtype=np.int64)
    ):
        _fail("D3 source ticks must be contiguous and non-negative")
    if "timestamp_s" in frame:
        timestamp = frame["timestamp_s"].to_numpy(dtype=np.float64)
        if not np.array_equal(timestamp, ticks.astype(np.float64) / 25.0):
            _fail("D3 timestamp_s differs from tick_25hz")
    return ticks


def _d3_audit_base(contract: dict[str, object]) -> dict[str, object]:
    return {
        "evidence_role": "in_memory_foundation_not_a_formal_result",
        "counted_eligible": False,
        "publisher_available": False,
        "hard_blockers": list(EVIDENCE_BLOCKERS),
        "contract_sha256": contract["contract_sha256"],
        "tick_authority_inventory_sha256": contract[
            "tick_authority_inventory_sha256"
        ],
        "source_csv_sha256": contract["source_csv_sha256"],
        "source_provenance_sha256": contract["source_provenance_sha256"],
        "dataframe_content_sha256": contract["dataframe_content_sha256"],
    }


def reassemble_d3_variants(
    frame_df: pd.DataFrame,
    contract: object,
) -> D3ReassemblyResult:
    """Pure, no-interpolation D3 central/minus/plus pose reassembly."""
    checked = _validate_d3_contract(contract)
    if not isinstance(frame_df, pd.DataFrame) or frame_df.empty:
        _fail("D3 source must be one non-empty dataframe")
    frame = frame_df.copy(deep=True)
    if "session_id" not in frame:
        _fail("D3 dataframe has no session_id")
    session_values = frame["session_id"].tolist()
    if any(type(value) is not str for value in session_values) or list(
        dict.fromkeys(session_values)
    ) != [checked["scene"]]:
        _fail("D3 dataframe session differs from contract")
    ticks = _frame_tick_values(frame)
    if ticks.tolist() != checked["input_tick_25hz"]:
        _fail("D3 dataframe ticks differ from the bound real tick authority")
    actual_dataframe_state = _canonical_dataframe_state(frame)
    expected_dataframe_state = _validate_dataframe_state(checked)
    if not _same_json(actual_dataframe_state, expected_dataframe_state):
        _fail("D3 dataframe differs from the canonical CSV-loaded content/schema")
    shifted_columns = checked["shifted_pose_columns"]
    missing_active = [
        column for column in checked["active_feature_columns"] if column not in frame
    ]
    if missing_active:
        _fail(f"D3 dataframe lacks active feature columns: {missing_active}")
    missing = [column for column in shifted_columns if column not in frame]
    if missing:
        _fail(f"D3 dataframe lacks shifted pose columns: {missing}")
    for column in shifted_columns:
        try:
            values = frame[column].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ProductionEvidenceGateError(
                f"D3 shifted column {column} is not numeric"
            ) from exc
        if not np.isfinite(values).all():
            _fail(f"D3 shifted column {column} contains a non-finite value")
        if column.endswith("_pose_valid") and not np.isin(values, (0.0, 1.0)).all():
            _fail(f"D3 shifted validity column {column} is not binary")
        if column.endswith("_pose_quality") and ((values < 0.0) | (values > 1.0)).any():
            _fail(f"D3 shifted quality column {column} is outside [0, 1]")
    if checked["mode"] == "identity_no_active_head_yaw":
        variants = {name: frame.copy(deep=True) for name in ("central", "minus", "plus")}
        return D3ReassemblyResult(
            variants=variants,
            audit={
                **_d3_audit_base(checked),
                "mode": checked["mode"],
                "scene": checked["scene"],
                "input_n_frames": len(frame),
                "output_n_frames": len(frame),
                "output_tick_start": int(ticks[0]),
                "output_tick_end_exclusive": int(ticks[-1]) + 1,
                "variant_source_tick_25hz": {
                    name: ticks.tolist() for name in ("central", "minus", "plus")
                },
                "protected_columns_identical": True,
                "identity_no_shift": True,
            },
        )

    aligned_by_variant = {}
    selected_by_variant = {}
    valid_by_variant = {}
    for name in ("central", "minus", "plus"):
        delta_ms = checked["variant_aligned_source_delta_ms"][name]
        aligned = ticks.astype(np.float64) + delta_ms / 40.0
        selected = np.searchsorted(aligned, ticks.astype(np.float64), side="right") - 1
        valid = (selected >= 0) & (ticks.astype(np.float64) <= aligned[-1])
        aligned_by_variant[name] = aligned
        selected_by_variant[name] = selected
        valid_by_variant[name] = valid
    common = np.logical_and.reduce(tuple(valid_by_variant.values()))
    target_indices = np.flatnonzero(common)
    if target_indices.size < 1:
        _fail("D3 variants have no common non-extrapolated output ticks")
    variants = {}
    source_ticks = {}
    for name in ("central", "minus", "plus"):
        source_indices = selected_by_variant[name][target_indices]
        aligned = aligned_by_variant[name]
        target_ticks = ticks[target_indices].astype(np.float64)
        if (aligned[source_indices] > target_ticks).any():
            _fail("D3 selection used a future shifted pose sample")
        has_next = source_indices + 1 < len(aligned)
        if (aligned[source_indices[has_next] + 1] <= target_ticks[has_next]).any():
            _fail("D3 selection did not choose the latest past sample")
        output = frame.iloc[target_indices].copy(deep=True).reset_index(drop=True)
        for column in shifted_columns:
            output.loc[:, column] = frame[column].to_numpy()[source_indices]
        variants[name] = output
        source_ticks[name] = ticks[source_indices].tolist()
    protected = [column for column in frame.columns if column not in shifted_columns]
    central_protected = variants["central"][protected]
    if any(
        not variants[name][protected].equals(central_protected) for name in ("minus", "plus")
    ):
        _fail("D3 reassembly changed gold/acoustic/non-pose columns across variants")
    output_ticks = ticks[target_indices]
    return D3ReassemblyResult(
        variants=variants,
        audit={
            **_d3_audit_base(checked),
            "mode": checked["mode"],
            "scene": checked["scene"],
            "input_n_frames": len(frame),
            "output_n_frames": len(target_indices),
            "output_tick_start": int(output_ticks[0]),
            "output_tick_end_exclusive": int(output_ticks[-1]) + 1,
            "variant_source_tick_25hz": source_ticks,
            "protected_columns_identical": True,
            "identity_no_shift": False,
        },
    )


__all__ = [
    "COUNTED_ELIGIBLE",
    "D3ReassemblyResult",
    "EVIDENCE_BLOCKERS",
    "EXPLICIT_PROTOCOL_DOCS",
    "ProductionEvidenceGateError",
    "build_d3_reassembly_contract",
    "build_environment_snapshot",
    "build_real_tick_authority_snapshot",
    "build_source_state_snapshot",
    "reassemble_d3_variants",
    "validate_environment_snapshot",
    "validate_real_tick_authority_snapshot",
    "validate_source_state_snapshot",
]
