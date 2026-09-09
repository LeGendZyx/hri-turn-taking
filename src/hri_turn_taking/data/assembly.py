"""Fail-closed assembly and readiness checks for real multiparty VAP tables.

The intake stage writes one canonical 25 Hz table per real session together
with a ``.provenance.json`` sidecar.  This module replaces intake placeholders
with independently extracted feature files without any temporal join: every
feature file must carry the exact same ``(session_id, timestamp_s)`` keys in
the exact same row order.  A mismatch is an error because sorting, nearest
neighbour joins, or interpolation could silently add lookahead or misalignment.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from hri_turn_taking.data.intake import (
    HPE_HEAD_POSE_STREAM_KEY,
    REAL_INTAKE_ARTIFACT_KIND,
    REAL_PROVENANCE_FORMAT_VERSION,
    REPOSITORY_ROOT,
    DeliveryGateError,
    IntakeConfig,
    run_intake,
)
from hri_turn_taking.data.multiparty import (
    GLOBAL_FEATURE_COLUMNS,
    MAX_PARTICIPANTS,
    PARTICIPANT_FEATURE_NAMES,
    SyntheticConfig,
    generate_multiparty_dataset,
    input_feature_columns,
    participant_column,
    vad_columns,
    validate_multiparty_dataframe,
)

KEY_COLUMNS = ("session_id", "timestamp_s")
INERT_REAL_COLUMNS = frozenset({"doa_angle_deg"})
CANONICAL_SUPPLY_COLUMNS = frozenset([*vad_columns(), *input_feature_columns()])
DATA_KINDS = frozenset({"real", "synthetic"})
FEATURE_MANIFEST_FORMAT_VERSION = 3
FEATURE_MANIFEST_KIND = "vap_feature_source"
SYNTHETIC_PROVENANCE_FORMAT_VERSION = 2
SYNTHETIC_PROVENANCE_KIND = "vap_synthetic_dataset"
SYNTHETIC_GENERATOR_ID = "hri_turn_taking.data.multiparty.generate_multiparty_dataset"
SYNTHETIC_GENERATOR_CONTRACT_VERSION = 2
SYNTHETIC_TRUST_MODEL = (
    "Replaying the repository's fixed generator contract prevents accidental real/synthetic "
    "misclassification, renaming, and ordinary re-signing of an arbitrary CSV. It does not "
    "defend against an attacker who can modify trusted project code or all artifacts together."
)
REAL_ASSEMBLED_ARTIFACT_KIND = "vap_real_assembled_session"
REAL_ANALYSIS_BASE_ARTIFACT_KIND = "vap_real_analysis_base"
SOURCE_VALUE_VERIFICATION_METHOD = "exact_keyed_values"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_D3_CONVENTION_RE = re.compile(r"^shared_?time=stream_?local_?time\+offset(_s)?$")
_NORMALISED_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RealDataGateError(ValueError):
    """A real-data artifact failed a fail-closed structural gate."""


class AssemblyGateError(RealDataGateError):
    """A base table or feature file cannot be assembled safely."""


class DataSourceGateError(RealDataGateError):
    """A file/directory data source is structurally unsafe or not trainable."""


class _HeldFileSnapshotError(OSError):
    """A held validation handle no longer proves the expected live path bytes."""


@dataclass(frozen=True)
class AssemblyResult:
    """Materialised result returned by :func:`assemble_session`."""

    frame_df: pd.DataFrame
    provenance: dict
    output_csv: Path

    @property
    def provenance_path(self) -> Path:
        return self.output_csv.with_suffix(".provenance.json")


@dataclass(frozen=True)
class ReadinessReport:
    """Purpose-specific readiness after the non-bypassable structural gate."""

    source: str
    data_kind: str
    files: tuple[str, ...]
    n_rows: int
    n_sessions: int
    n_groups: int
    session_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_artifacts: tuple[dict[str, object], ...] = field(default_factory=tuple)
    structural_ready: bool = True
    target_ready: bool = False
    target_blockers: tuple[str, ...] = field(default_factory=tuple)
    feature_set: str | None = None
    selected_feature_ready: bool | None = None
    selected_feature_blockers: tuple[str, ...] = field(default_factory=tuple)
    experiment_blockers: tuple[str, ...] = field(default_factory=tuple)
    placeholder_columns: tuple[str, ...] = field(default_factory=tuple)
    scientific_warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def session_count(self) -> int:
        """Human-facing alias for ``n_sessions``."""
        return self.n_sessions

    @property
    def group_count(self) -> int:
        """Human-facing alias for ``n_groups``."""
        return self.n_groups

    @property
    def training_blockers(self) -> tuple[str, ...]:
        """Compatibility alias; selected-feature blockers remain separate."""
        return self.target_blockers

    @property
    def trainable(self) -> bool:
        """Compatibility alias for target readiness, not input readiness."""
        return self.target_ready

    @property
    def blockers(self) -> tuple[str, ...]:
        """Compatibility alias for the old target-tier blocker field."""
        return self.target_blockers

    @property
    def warnings(self) -> tuple[str, ...]:
        """Short alias for scientific limitations that do not block loading."""
        return self.scientific_warnings

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation for CLI/report consumers."""
        return {
            "source": self.source,
            "data_kind": self.data_kind,
            "files": list(self.files),
            "n_rows": self.n_rows,
            "n_sessions": self.n_sessions,
            "n_groups": self.n_groups,
            "session_ids": list(self.session_ids),
            "group_ids": list(self.group_ids),
            "source_artifacts": [dict(artifact) for artifact in self.source_artifacts],
            "structural_ready": self.structural_ready,
            "target_ready": self.target_ready,
            "target_blockers": list(self.target_blockers),
            "feature_set": self.feature_set,
            "selected_feature_ready": self.selected_feature_ready,
            "selected_feature_blockers": list(self.selected_feature_blockers),
            "trainable": self.trainable,
            "blockers": list(self.blockers),
            "experiment_blockers": list(self.experiment_blockers),
            "placeholder_columns": list(self.placeholder_columns),
            "scientific_warnings": list(self.scientific_warnings),
        }


@dataclass(frozen=True)
class _DatasetFile:
    path: Path
    frame_df: pd.DataFrame
    provenance: dict | None
    data_sha256: str
    provenance_sha256: str


def _sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".provenance.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_no_clobber(staged: Path, destination: Path) -> None:
    """Atomically publish one staged assembly file without replacing a peer."""

    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise AssemblyGateError(
            f"Assembly output appeared while materialisation was running: {destination}"
        ) from exc
    except OSError as exc:
        raise AssemblyGateError(
            f"Could not publish assembly output without overwriting: {destination}: {exc}"
        ) from exc


def _reserve_staged_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    """Create a writer-private staging inode in the destination directory."""

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=prefix,
        suffix=suffix,
        dir=directory,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _file_identity(path: Path) -> tuple[int, int]:
    """Capture the filesystem identity of one writer-owned inode."""

    stat = path.stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _windows_open_fd(
    path: Path,
    *,
    access: int,
    share_mode: int,
) -> int:
    """Open a Windows file handle with an explicit sharing contract."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        access,
        share_mode,
        None,
        3,  # OPEN_EXISTING
        0x00000080 | 0x00200000,  # FILE_ATTRIBUTE_NORMAL | OPEN_REPARSE_POINT
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), str(path))
    try:
        return msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        raise


@contextmanager
def _open_held_validation_file(path: Path) -> Iterator[BinaryIO]:
    """Hold one no-follow read handle while a published artifact is replayed.

    Windows permits concurrent readers but denies WRITE and DELETE opens for
    the lifetime of this handle. POSIX uses ``O_NOFOLLOW`` and verifies the
    live directory entry before and after the replay.
    """

    if os.name == "nt":
        fd = _windows_open_fd(
            path,
            access=0x80000000,  # GENERIC_READ
            share_mode=0x00000001,  # FILE_SHARE_READ only
        )
    else:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            yield handle
    except BaseException:
        # ``fdopen`` owns the descriptor once constructed. If construction
        # itself failed, closing an already-closed descriptor is harmlessly
        # ignored here.
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _hash_held_handle(handle: BinaryIO, path: Path) -> str:
    """Hash from the already-held handle; ``path`` exists for audit/test context."""

    del path
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _snapshot_held_path(handle: BinaryIO, path: Path) -> tuple[tuple[int, int], str]:
    """fstat -> hash/read -> fstat using one continuously held handle."""

    before = os.fstat(handle.fileno())
    before_identity = (before.st_dev, before.st_ino)
    if not stat.S_ISREG(before.st_mode):
        raise _HeldFileSnapshotError(f"held validation path is not a regular file: {path}")
    held_hash = _hash_held_handle(handle, path)
    after = os.fstat(handle.fileno())
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", None),
    )
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", None),
    )
    if before_state != after_state:
        raise _HeldFileSnapshotError(
            f"held validation file changed while it was being read: {path}"
        )
    return before_identity, held_hash


@contextmanager
def _held_file_snapshot_guard(
    expectations: tuple[tuple[Path, tuple[int, int], str], ...],
) -> Iterator[None]:
    """Keep all target handles held while a caller replays the exact pair."""

    with ExitStack() as stack:
        held = [
            (path, identity, expected_hash, stack.enter_context(_open_held_validation_file(path)))
            for path, identity, expected_hash in expectations
        ]

        def validate() -> None:
            for path, identity, expected_hash, handle in held:
                held_identity, held_hash = _snapshot_held_path(handle, path)
                if held_identity != identity or held_hash != expected_hash:
                    raise _HeldFileSnapshotError(
                        f"held validation identity/bytes changed: {path}"
                    )
                try:
                    live_identity = _file_identity(path)
                except OSError as exc:
                    raise _HeldFileSnapshotError(
                        f"held validation path disappeared: {path}"
                    ) from exc
                if live_identity != identity:
                    raise _HeldFileSnapshotError(
                        f"live path no longer names the held validation inode: {path}"
                    )

        validate()
        try:
            yield
        finally:
            validate()


@dataclass(frozen=True)
class _OwnedPathCleanup:
    removed: bool
    foreign_preserved: bool = False
    recovery_path: Path | None = None
    detail: str | None = None


def _unique_recovery_path(path: Path) -> Path:
    for _ in range(32):
        candidate = path.with_name(f".{path.name}.recovery.{uuid.uuid4().hex}")
        if not _path_lexists(candidate):
            return candidate
    raise OSError(f"could not allocate a unique recovery path beside {path}")


def _windows_delete_exact_path(path: Path, identity: tuple[int, int]) -> bool:
    """Delete the held Windows directory entry without a check/unlink gap."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    try:
        fd = _windows_open_fd(
            path,
            access=0x80000000 | 0x00010000,  # GENERIC_READ | DELETE
            share_mode=0x00000001,  # FILE_SHARE_READ only
        )
    except OSError:
        return False
    delete_requested = False
    try:
        file_stat = os.fstat(fd)
        if (file_stat.st_dev, file_stat.st_ino) != identity:
            return False
        try:
            if _file_identity(path) != identity:
                return False
        except OSError:
            return False

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

        disposition = FileDispositionInfo(True)
        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(fd)
        if not set_information(
            handle,
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            return False
        delete_requested = True
    finally:
        os.close(fd)
    return delete_requested and not _path_lexists(path)


def _delete_exact_quarantine(path: Path, identity: tuple[int, int]) -> bool:
    if os.name == "nt":
        return _windows_delete_exact_path(path, identity)
    try:
        with _open_held_validation_file(path) as handle:
            held_identity, _ = _snapshot_held_path(handle, path)
            if held_identity != identity or _file_identity(path) != identity:
                return False
            # Under the documented trusted single-writer POSIX boundary, the
            # random quarantine name is not concurrently rewritten.
            path.unlink()
    except OSError:
        return False
    return not _path_lexists(path)


def _restore_foreign_quarantine(recovery: Path, original: Path) -> bool:
    """Restore a foreign inode without replacing any newer path owner."""

    try:
        os.link(recovery, original)
    except (FileExistsError, OSError):
        return False
    try:
        identity = _file_identity(recovery)
        if _file_identity(original) != identity:
            return False
    except OSError:
        return False
    return _delete_exact_quarantine(recovery, identity)


def _remove_owned_path(path: Path, identity: tuple[int, int]) -> _OwnedPathCleanup:
    """Atomically quarantine a path, then delete only a proven writer inode."""

    recovery = _unique_recovery_path(path)
    try:
        os.rename(path, recovery)
    except FileNotFoundError:
        return _OwnedPathCleanup(removed=True)
    except OSError as exc:
        return _OwnedPathCleanup(
            removed=False,
            recovery_path=None,
            detail=f"could not quarantine path safely: {exc}",
        )
    try:
        moved_identity = _file_identity(recovery)
    except OSError as exc:
        return _OwnedPathCleanup(
            removed=False,
            recovery_path=recovery,
            detail=f"could not identify quarantined path: {exc}",
        )
    if moved_identity == identity:
        if _delete_exact_quarantine(recovery, identity):
            return _OwnedPathCleanup(removed=True)
        return _OwnedPathCleanup(
            removed=False,
            recovery_path=recovery if _path_lexists(recovery) else None,
            detail="writer-owned quarantine could not be deleted safely",
        )

    restored = _restore_foreign_quarantine(recovery, path)
    return _OwnedPathCleanup(
        removed=False,
        foreign_preserved=True,
        recovery_path=None if restored else recovery,
        detail=(
            "foreign replacement preserved and restored to its original path"
            if restored
            else "foreign replacement preserved at recovery path"
        ),
    )


def _dataframe_schema(frame_df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    columns = frame_df.columns.tolist()
    return columns, {column: str(frame_df[column].dtype) for column in columns}


def _csv_binding(csv_path: Path, frame_df: pd.DataFrame) -> dict[str, object]:
    columns, dtypes = _dataframe_schema(frame_df)
    return {
        "csv_sha256": _sha256(csv_path),
        "columns": columns,
        "dtypes": dtypes,
        "n_rows": len(frame_df),
    }


def _type_exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int/float coercions."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if set(left) != set(right):  # type: ignore[arg-type]
            return False
        return all(
            _type_exact_json_equal(left[key], right[key])  # type: ignore[index]
            for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _type_exact_json_equal(a, b)
            for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return bool(left == right)


def _validate_csv_binding(
    provenance: dict,
    csv_path: Path,
    frame_df: pd.DataFrame,
    sidecar: Path,
) -> None:
    expected = _csv_binding(csv_path, frame_df)
    digest = provenance.get("csv_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise DataSourceGateError(
            f"{sidecar.name} csv_sha256 must be a lowercase 64-character SHA-256"
        )
    for field_name, actual in expected.items():
        if not _type_exact_json_equal(provenance.get(field_name), actual):
            relation = "does not match" if field_name == "csv_sha256" else "does not bind"
            raise DataSourceGateError(
                f"{sidecar.name} {field_name} {relation} exactly to final CSV "
                f"{csv_path.name}"
            )


def _read_csv(csv_path: Path, *, error_type: type[RealDataGateError]) -> pd.DataFrame:
    if not csv_path.is_file():
        raise error_type(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise error_type(f"Only CSV files are supported by the real-data gate: {csv_path}")
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except OSError as exc:
        raise error_type(f"Could not read CSV header {csv_path}: {exc}") from exc
    if not header:
        raise error_type(f"CSV file has no header: {csv_path}")
    duplicate_headers = sorted({name for name in header if header.count(name) > 1})
    if duplicate_headers:
        raise error_type(f"{csv_path.name} has duplicate header columns: {duplicate_headers}")
    try:
        # Pandas' default high-speed float parser can land one ULP away from
        # the binary64 value represented by a shortest-round-trip decimal.
        # Intake replay is intentionally bit-exact, so always use the parser
        # mode that honours the CSV writer's round-trip representation.
        return pd.read_csv(csv_path, float_precision="round_trip")
    except Exception as exc:
        raise error_type(f"Could not parse CSV {csv_path}: {exc}") from exc


def _require_string_list(provenance: dict, key: str, sidecar: Path) -> list[str]:
    value = provenance.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataSourceGateError(f"{sidecar.name} {key} must be a JSON array of strings")
    if len(value) != len(set(value)):
        raise DataSourceGateError(f"{sidecar.name} {key} contains duplicate columns")
    return value


def _validate_data_kind(data_kind: str) -> None:
    if data_kind not in DATA_KINDS:
        raise DataSourceGateError(
            f"data_kind must be explicitly one of {sorted(DATA_KINDS)}, got {data_kind!r}"
        )


def _require_finite_json_number(value: object, name: str, sidecar: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataSourceGateError(f"{sidecar.name} {name} must be a JSON number")
    number = float(value)
    if not np.isfinite(number):
        raise DataSourceGateError(f"{sidecar.name} {name} must be finite")
    return number


def _require_normalised_id(provenance: dict, key: str, sidecar: Path) -> str:
    value = provenance.get(key)
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _NORMALISED_ID_RE.fullmatch(value)
    ):
        raise DataSourceGateError(
            f"{sidecar.name} {key} must be a non-empty normalised identifier with no "
            "leading/trailing whitespace"
        )
    return value


def _resolve_bound_source(
    source: object,
    *,
    owner_path: Path,
    error_type: type[RealDataGateError],
    field_name: str,
    forbidden_paths: set[Path] | None = None,
) -> tuple[Path, str]:
    if not isinstance(source, dict):
        raise error_type(f"{owner_path.name} {field_name} must be a JSON object")
    raw_path = source.get("path")
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise error_type(f"{owner_path.name} {field_name}.path must be a non-empty string")
    raw_sha256 = source.get("sha256")
    if not isinstance(raw_sha256, str) or not _SHA256_RE.fullmatch(raw_sha256):
        raise error_type(
            f"{owner_path.name} {field_name}.sha256 must be a lowercase "
            "64-character SHA-256"
        )
    resolved = Path(raw_path)
    if not resolved.is_absolute():
        resolved = owner_path.parent / resolved
    resolved = resolved.resolve()
    if forbidden_paths and resolved in forbidden_paths:
        raise error_type(
            f"{owner_path.name} {field_name} must identify a concrete upstream source, "
            "not the generated feature/data artifact itself"
        )
    if not resolved.is_file():
        raise error_type(f"{owner_path.name} {field_name} source file does not exist: {resolved}")
    if _sha256(resolved) != raw_sha256:
        raise error_type(f"{owner_path.name} {field_name} source hash does not match: {resolved}")
    return resolved, raw_sha256


def _present_slot_names(frame_df: pd.DataFrame) -> set[str]:
    first = frame_df.iloc[0]
    return {
        f"P{slot}"
        for slot in range(1, MAX_PARTICIPANTS + 1)
        if float(first[participant_column(slot, "present")]) == 1.0
    }


def _required_real_declarations(frame_df: pd.DataFrame) -> set[str]:
    present_slots = _present_slot_names(frame_df)
    required = {
        participant_column(slot, feature)
        for slot in range(1, MAX_PARTICIPANTS + 1)
        if f"P{slot}" in present_slots
        for feature in [*PARTICIPANT_FEATURE_NAMES, "vad"]
    }
    required.update(GLOBAL_FEATURE_COLUMNS)
    return required


def _expected_intake_pose_columns(frame_df: pd.DataFrame) -> set[str]:
    present_slots = _present_slot_names(frame_df)
    return {
        f"{slot}_{feature}"
        for slot in present_slots
        for feature in ("head_yaw_deg", "pose_valid", "pose_quality")
    }


def _validate_participant_to_slot(
    provenance: dict, frame_df: pd.DataFrame, sidecar: Path
) -> dict[str, str]:
    mapping = provenance.get("participant_to_slot")
    if not isinstance(mapping, dict) or any(
        not isinstance(participant, str)
        or not participant.strip()
        or not isinstance(slot, str)
        for participant, slot in mapping.items()
    ):
        raise DataSourceGateError(
            f"{sidecar.name} participant_to_slot must be a JSON object of strings"
        )
    normalised = {participant.strip().upper(): slot for participant, slot in mapping.items()}
    if len(normalised) != len(mapping):
        raise DataSourceGateError(
            f"{sidecar.name} participant_to_slot contains duplicate participant labels "
            "after case normalisation"
        )
    expected_by_slot = {"P1": "R", "P2": "A", "P3": "B", "P4": "C"}
    present_slots = _present_slot_names(frame_df)
    expected = {
        participant: slot
        for slot, participant in expected_by_slot.items()
        if slot in present_slots
    }
    if normalised != expected:
        raise DataSourceGateError(
            f"{sidecar.name} participant_to_slot={normalised!r} does not match the "
            f"canonical present-slot binding {expected!r}"
        )
    return normalised


def _expected_column_binding(
    column: str, participant_to_slot: dict[str, str]
) -> tuple[str | None, str | None]:
    match = re.fullmatch(r"(P[1-4])_.+", column)
    if match is None:
        return None, None
    slot = match.group(1)
    participant_by_slot = {
        bound_slot: participant for participant, bound_slot in participant_to_slot.items()
    }
    return participant_by_slot.get(slot), slot


def _validate_real_column_sources(
    provenance: dict,
    real_columns: list[str],
    participant_to_slot: dict[str, str],
    csv_path: Path,
    sidecar: Path,
) -> None:
    sources = provenance.get("column_sources")
    if not isinstance(sources, dict):
        raise DataSourceGateError(
            f"{sidecar.name} column_sources must be a JSON object covering every real column"
        )
    missing = sorted(set(real_columns) - set(sources))
    extra = sorted(set(sources) - set(real_columns))
    if missing or extra:
        raise DataSourceGateError(
            f"{sidecar.name} column_sources must cover exactly real_columns; "
            f"missing={missing}, extra={extra}"
        )
    forbidden = {csv_path.resolve(), sidecar.resolve()}
    intake_pose_columns = {
        f"{slot}_{feature}"
        for slot in participant_to_slot.values()
        for feature in ("head_yaw_deg", "pose_valid", "pose_quality")
    }
    authority_map = _normalise_person_map(
        provenance.get("person_to_participant"), sidecar, "person_to_participant"
    )
    contract = provenance.get("identity_contract")
    inventory_sha = (
        contract.get("inventory", {}).get("sha256")
        if isinstance(contract, dict)
        else None
    )
    expected_pose_source_columns = [
        "frame_index",
        "timestamp_sec",
        "person",
        "x_w",
        "z_w",
        "facing_x",
        "facing_z",
        "facing_src_abs_yaw",
    ]
    for column in real_columns:
        record = sources[column]
        if not isinstance(record, dict):
            raise DataSourceGateError(
                f"{sidecar.name} column_sources[{column!r}] must be a JSON object"
            )
        expected_participant, expected_slot = _expected_column_binding(
            column, participant_to_slot
        )
        if record.get("participant") != expected_participant or record.get("slot") != expected_slot:
            raise DataSourceGateError(
                f"{sidecar.name} column_sources[{column!r}] must bind participant="
                f"{expected_participant!r}, slot={expected_slot!r}"
            )
        expected_purpose = "gold_target" if column in set(vad_columns()) else "model_input"
        if record.get("purpose") != expected_purpose:
            raise DataSourceGateError(
                f"{sidecar.name} column_sources[{column!r}].purpose must be "
                f"{expected_purpose!r}"
            )
        source_path, source_hash = _resolve_bound_source(
            record,
            owner_path=sidecar,
            error_type=DataSourceGateError,
            field_name=f"column_sources[{column!r}]",
            forbidden_paths=forbidden,
        )
        if column in intake_pose_columns:
            participant_persons = sorted(
                [
                    person
                    for person, participant in authority_map.items()
                    if participant == expected_participant
                ],
                key=int,
            )
            expected_fields = {
                "participant",
                "slot",
                "purpose",
                "source_kind",
                "source_columns",
                "hpe_persons",
                "identity_authority",
                "identity_inventory_sha256",
                "transform",
                "path",
                "sha256",
                "size_bytes",
            }
            if set(record) != expected_fields:
                raise DataSourceGateError(
                    f"{sidecar.name} column_sources[{column!r}] must use the exact "
                    "HPE pose-intake v2 identity binding fields"
                )
            if (
                record.get("source_kind") != "hpe_pose_intake_v2"
                or record.get("source_columns") != expected_pose_source_columns
                or record.get("hpe_persons") != participant_persons
                or record.get("identity_authority")
                != "sealed_hpe_transcript_crosswalk"
                or record.get("identity_inventory_sha256") != inventory_sha
                or record.get("transform") != "strict_past_only_pose_v2"
            ):
                raise DataSourceGateError(
                    f"{sidecar.name} column_sources[{column!r}] does not bind the "
                    "canonical participant/person identity transform"
                )
            if (
                source_path.stat().st_size != record.get("size_bytes")
                or source_hash != provenance.get("intake_sources", {})
                .get("world_persons", {})
                .get("sha256")
            ):
                raise DataSourceGateError(
                    f"{sidecar.name} column_sources[{column!r}] is not bound to the "
                    "canonical world_persons source snapshot"
                )


def _normalise_person_map(value: object, owner: Path, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(person, str) or not isinstance(participant, str)
        for person, participant in value.items()
    ):
        raise DataSourceGateError(
            f"{owner.name} {field_name} must be a JSON object of person/participant strings"
        )
    normalised = {
        person: participant.strip().upper() for person, participant in value.items()
    }
    if any(not person or person != person.strip() for person in normalised):
        raise DataSourceGateError(
            f"{owner.name} {field_name} person identifiers must be non-empty and trimmed"
        )
    return normalised


def _replay_participant_map_csv(path: Path, owner: Path) -> dict[str, str]:
    try:
        map_df = pd.read_csv(path)
    except Exception as exc:
        raise DataSourceGateError(
            f"{owner.name} bound participant_map.csv is unreadable: {exc}"
        ) from exc
    if not {"person", "participant"}.issubset(map_df.columns):
        raise DataSourceGateError(
            f"{owner.name} bound participant_map.csv lacks person/participant columns"
        )
    if map_df[["person", "participant"]].isna().any().any():
        raise DataSourceGateError(
            f"{owner.name} bound participant_map.csv contains missing person/participant values"
        )
    normalised = map_df.assign(
        person=map_df["person"].astype(str),
        participant=map_df["participant"].astype(str).str.strip().str.upper(),
    )
    if normalised["person"].map(lambda value: not value or value != value.strip()).any():
        raise DataSourceGateError(
            f"{owner.name} bound participant_map.csv person identifiers must be trimmed"
        )
    conflicts = normalised.groupby("person", sort=False)["participant"].nunique()
    conflicted = sorted(conflicts[conflicts > 1].index.tolist())
    if conflicted:
        raise DataSourceGateError(
            f"{owner.name} bound participant_map.csv maps persons {conflicted} to "
            "conflicting participants"
        )
    duplicated = sorted(normalised.loc[normalised["person"].duplicated(False), "person"].unique())
    if duplicated:
        raise DataSourceGateError(
            f"{owner.name} bound participant_map.csv contains duplicate person rows: {duplicated}"
        )
    return dict(
        zip(normalised["person"], normalised["participant"], strict=True)
    )


def _validate_intake_sources(
    provenance: dict,
    participant_to_slot: dict[str, str],
    csv_path: Path,
    sidecar: Path,
) -> None:
    sources = provenance.get("intake_sources")
    required = {
        "world_persons",
        "participant_map",
        "mapping",
        "offset",
        "offset_quality",
    }
    if not isinstance(sources, dict) or set(sources) != required:
        raise DataSourceGateError(
            f"{sidecar.name} intake_sources must bind exactly {sorted(required)}"
        )
    forbidden = {csv_path.resolve(), sidecar.resolve()}
    resolved: dict[str, Path] = {}
    for source_name in sorted(required):
        binding = sources[source_name]
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise DataSourceGateError(
                f"{sidecar.name} intake_sources.{source_name} must bind exactly "
                "path, sha256, and size_bytes"
            )
        source_path, _ = _resolve_bound_source(
            binding,
            owner_path=sidecar,
            error_type=DataSourceGateError,
            field_name=f"intake_sources.{source_name}",
            forbidden_paths=forbidden,
        )
        size = binding.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or source_path.stat().st_size != size
        ):
            raise DataSourceGateError(
                f"{sidecar.name} intake_sources.{source_name}.size_bytes does not "
                "match the source"
            )
        resolved[source_name] = source_path

    delivery_dir_value = provenance.get("delivery_dir")
    if not isinstance(delivery_dir_value, str) or not delivery_dir_value.strip():
        raise DataSourceGateError(f"{sidecar.name} delivery_dir must be a non-empty string")
    delivery_dir = Path(delivery_dir_value).resolve()
    if any(path.parent != delivery_dir for path in resolved.values()):
        raise DataSourceGateError(
            f"{sidecar.name} intake_sources must all come from delivery_dir {delivery_dir}"
        )

    mapping_doc = _read_json_object(
        resolved["mapping"], owner=sidecar, field_name="intake_sources.mapping"
    )
    if mapping_doc.get("review_status") != "confirmed":
        raise DataSourceGateError(
            f"{sidecar.name} bound mapping.json review_status must be 'confirmed'"
        )
    if mapping_doc.get("delivery") != delivery_dir.name:
        raise DataSourceGateError(
            f"{sidecar.name} bound mapping.json delivery does not match delivery_dir"
        )
    for key in ("group", "scenario"):
        provenance_key = f"{key}_id"
        if mapping_doc.get(key) != provenance.get(provenance_key):
            raise DataSourceGateError(
                f"{sidecar.name} bound mapping.json {key} does not match {provenance_key}"
            )
    mapping_person_map = _normalise_person_map(
        mapping_doc.get("participant_map"), sidecar, "bound mapping.json participant_map"
    )
    proposal_person_map = _normalise_person_map(
        provenance.get("upstream_person_to_participant_proposal"),
        sidecar,
        "upstream_person_to_participant_proposal",
    )
    if mapping_person_map != proposal_person_map:
        raise DataSourceGateError(
            f"{sidecar.name} bound mapping.json participant_map does not match the "
            "recorded upstream proposal"
        )

    csv_person_map = _replay_participant_map_csv(resolved["participant_map"], sidecar)
    if csv_person_map != proposal_person_map:
        raise DataSourceGateError(
            f"{sidecar.name} bound participant_map.csv does not match the recorded "
            "upstream proposal"
        )
    provenance_person_map = _normalise_person_map(
        provenance.get("person_to_participant"), sidecar, "person_to_participant"
    )
    expected_labels = set(participant_to_slot)
    if set(provenance_person_map.values()) != expected_labels:
        raise DataSourceGateError(
            f"{sidecar.name} person_to_participant labels do not match participant_to_slot"
        )


_INTAKE_REPLAY_FIELDS = (
    "delivery_dir",
    "review_status",
    "identity_authority",
    "identity_contract",
    "upstream_hpe_review_status",
    "stream_key",
    "offset_applied_s",
    "offset_quality",
    "d3_sources",
    "intake_sources",
    "intake_config",
    "accepted_window",
    "source_frame_rate_hz",
    "source_time_range_local_s",
    "output_tick_range",
    "frame_rate_hz",
    "n_frames",
    "time_range_shared_s",
    "participant_to_slot",
    "person_to_participant",
    "upstream_person_to_participant_proposal",
    "fragments_per_participant",
    "fragment_overlap",
    "fragment_overlap_policy",
    "validity_note",
    "unmapped_persons_dropped",
    "rows_outside_accepted_window",
    "missing_sample_fraction",
    "facing_quality",
    "doa_note",
)


def _validate_identity_contract_and_replay_pose(
    provenance: dict,
    frame_df: pd.DataFrame,
    csv_path: Path,
    sidecar: Path,
) -> None:
    """Replay the sealed identity authority and deterministic pose/grid stage."""
    if provenance.get("identity_authority") != "sealed_hpe_transcript_crosswalk":
        raise DataSourceGateError(
            f"{sidecar.name} identity_authority must be the sealed HPE→Transcript "
            "crosswalk"
        )
    contract = provenance.get("identity_contract")
    if not isinstance(contract, dict):
        raise DataSourceGateError(f"{sidecar.name} identity_contract must be an object")
    inventory = contract.get("inventory")
    if not isinstance(inventory, dict) or not isinstance(inventory.get("path"), str):
        raise DataSourceGateError(
            f"{sidecar.name} identity_contract.inventory binding is malformed"
        )
    config_payload = provenance.get("intake_config")
    expected_config_fields = {
        "stream_key",
        "frame_rate_hz",
        "max_gap_s",
        "fragment_conflict_m",
        "fragment_facing_conflict",
        "fragment_overlap_policy",
    }
    if not isinstance(config_payload, dict) or set(config_payload) != expected_config_fields:
        raise DataSourceGateError(
            f"{sidecar.name} intake_config must bind exactly the production pose settings"
        )
    try:
        config = IntakeConfig(**config_payload)
        replay = run_intake(
            provenance.get("delivery_dir"),
            config,
            crosswalk_inventory=inventory["path"],
            repository_root=REPOSITORY_ROOT,
        )
    except (DeliveryGateError, OSError, TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{sidecar.name} sealed crosswalk/pose replay failed: {exc}"
        ) from exc

    expected = replay.frame_df
    pose_columns = sorted(_expected_intake_pose_columns(expected))
    replay_columns = [
        *KEY_COLUMNS,
        *(participant_column(slot, "present") for slot in range(1, MAX_PARTICIPANTS + 1)),
        *pose_columns,
    ]
    missing = sorted(set(replay_columns) - set(frame_df))
    if missing or len(frame_df) != len(expected):
        raise DataSourceGateError(
            f"{sidecar.name} does not retain the sealed intake grid/pose shape; "
            f"missing={missing}, rows={len(frame_df)}, expected_rows={len(expected)}"
        )
    if not np.array_equal(
        frame_df["session_id"].to_numpy(dtype=object),
        expected["session_id"].to_numpy(dtype=object),
    ):
        raise DataSourceGateError(f"{sidecar.name} session keys differ from pose replay")
    numeric_columns = [column for column in replay_columns if column != "session_id"]
    try:
        actual_values = frame_df[numeric_columns].to_numpy(dtype=float)
        expected_values = expected[numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{sidecar.name} retained intake keys/pose columns must be numeric"
        ) from exc
    if not np.array_equal(actual_values, expected_values):
        raise DataSourceGateError(
            f"{sidecar.name} grid, present masks, or HPE pose values differ from the "
            "sealed deterministic intake replay"
        )
    for field_name in _INTAKE_REPLAY_FIELDS:
        if not _type_exact_json_equal(
            provenance.get(field_name), replay.provenance.get(field_name)
        ):
            raise DataSourceGateError(
                f"{sidecar.name} {field_name} differs from sealed intake replay"
            )
    expected_pose_sources = {
        column: replay.provenance["column_sources"][column] for column in pose_columns
    }
    actual_sources = provenance.get("column_sources")
    if not isinstance(actual_sources, dict) or any(
        not _type_exact_json_equal(actual_sources.get(column), binding)
        for column, binding in expected_pose_sources.items()
    ):
        raise DataSourceGateError(
            f"{sidecar.name} HPE pose column identity bindings differ from replay"
        )


def _read_json_object(path: Path, *, owner: Path, field_name: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceGateError(
            f"{owner.name} {field_name} is not a readable JSON object: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DataSourceGateError(f"{owner.name} {field_name} must contain a JSON object")
    return payload


def _validate_d3_sources(
    provenance: dict,
    summary: dict,
    stream_key: str,
    csv_path: Path,
    sidecar: Path,
) -> None:
    d3_sources = provenance.get("d3_sources")
    if not isinstance(d3_sources, dict) or set(d3_sources) != {"offset", "offset_quality"}:
        raise DataSourceGateError(
            f"{sidecar.name} d3_sources must bind exactly offset and offset_quality"
        )
    forbidden = {csv_path.resolve(), sidecar.resolve()}
    offset_path, _ = _resolve_bound_source(
        d3_sources["offset"],
        owner_path=sidecar,
        error_type=DataSourceGateError,
        field_name="d3_sources.offset",
        forbidden_paths=forbidden,
    )
    quality_path, _ = _resolve_bound_source(
        d3_sources["offset_quality"],
        owner_path=sidecar,
        error_type=DataSourceGateError,
        field_name="d3_sources.offset_quality",
        forbidden_paths=forbidden,
    )
    offset_doc = _read_json_object(
        offset_path, owner=sidecar, field_name="d3_sources.offset"
    )
    streams = offset_doc.get("streams")
    if not isinstance(streams, dict) or stream_key not in streams:
        raise DataSourceGateError(
            f"{sidecar.name} bound offset.json has no streams[{stream_key!r}]"
        )
    source_offset = _require_finite_json_number(
        streams[stream_key], f"bound offset streams[{stream_key!r}]", sidecar
    )
    if abs(source_offset - float(summary["offset_s"])) > 1e-6:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.offset_s does not match bound offset.json"
        )

    quality_doc = _read_json_object(
        quality_path, owner=sidecar, field_name="d3_sources.offset_quality"
    )
    timebase_keys = quality_doc.get("timebase_keys")
    if not isinstance(timebase_keys, dict) or stream_key not in timebase_keys:
        raise DataSourceGateError(
            f"{sidecar.name} bound offset_quality.json timebase_keys must include "
            f"{stream_key!r} as a key"
        )
    if summary.get("timebase_keys") != timebase_keys:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.timebase_keys does not match the bound "
            "offset_quality.json"
        )
    if quality_doc.get("convention") != summary.get("convention"):
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.convention does not match the bound "
            "offset_quality.json"
        )
    source_convention = quality_doc.get("convention")
    if not isinstance(source_convention, str) or not _D3_CONVENTION_RE.fullmatch(
        source_convention.replace(" ", "")
    ):
        raise DataSourceGateError(
            f"{sidecar.name} bound offset_quality.json has an invalid D3 convention"
        )
    quality = quality_doc.get("quality")
    source_summary = quality.get(stream_key) if isinstance(quality, dict) else None
    if not isinstance(source_summary, dict):
        raise DataSourceGateError(
            f"{sidecar.name} bound offset_quality.json has no quality[{stream_key!r}]"
        )
    source_quality_offset = _require_finite_json_number(
        source_summary.get("offset_s"),
        f"bound offset_quality quality[{stream_key!r}].offset_s",
        sidecar,
    )
    source_error_bound = _require_finite_json_number(
        source_summary.get("error_bound_ms"),
        f"bound offset_quality quality[{stream_key!r}].error_bound_ms",
        sidecar,
    )
    if source_summary.get("self_check_ok") is not True:
        raise DataSourceGateError(
            f"{sidecar.name} bound offset_quality.json self_check_ok must be true"
        )
    if source_summary.get("status") not in {"ok", "degraded"}:
        raise DataSourceGateError(
            f"{sidecar.name} bound offset_quality.json status must be ok/degraded"
        )
    if source_error_bound < 0.0 or abs(source_quality_offset - source_offset) > 1e-6:
        raise DataSourceGateError(
            f"{sidecar.name} bound D3 source values are inconsistent"
        )
    for field_name in ("offset_s", "self_check_ok", "status", "error_bound_ms"):
        if source_summary.get(field_name) != summary.get(field_name):
            raise DataSourceGateError(
                f"{sidecar.name} offset_quality.{field_name} does not match the bound "
                "offset_quality.json"
            )


def _validate_offset_summary(provenance: dict, csv_path: Path, sidecar: Path) -> None:
    summary = provenance.get("offset_quality")
    if not isinstance(summary, dict):
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality must be a complete D3 JSON object"
        )
    required = {
        "stream",
        "offset_s",
        "self_check_ok",
        "convention",
        "status",
        "error_bound_ms",
        "timebase_keys",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality is missing required fields: {missing}"
        )
    stream_key = provenance.get("stream_key")
    if not isinstance(stream_key, str) or not stream_key.strip():
        raise DataSourceGateError(f"{sidecar.name} stream_key must be a non-empty string")
    if stream_key != HPE_HEAD_POSE_STREAM_KEY:
        raise DataSourceGateError(
            f"{sidecar.name} stream_key must be {HPE_HEAD_POSE_STREAM_KEY!r} for "
            "world_persons.csv; RGB offsets cannot substitute for the HPE six-camera axis"
        )
    if summary["stream"] != stream_key:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.stream must match stream_key {stream_key!r}"
        )
    timebase_keys = summary["timebase_keys"]
    if not isinstance(timebase_keys, dict) or stream_key not in timebase_keys:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.timebase_keys must be an object containing "
            f"{stream_key!r} as a key"
        )
    convention = summary["convention"]
    if not isinstance(convention, str) or not _D3_CONVENTION_RE.fullmatch(
        convention.replace(" ", "")
    ):
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.convention must state "
            "'shared_time = stream_local_time + offset_s'"
        )
    if summary["self_check_ok"] is not True:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.self_check_ok must be the JSON boolean true"
        )
    if summary["status"] not in {"ok", "degraded"}:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.status must be 'ok' or 'degraded'"
        )
    bound = _require_finite_json_number(
        summary["error_bound_ms"], "offset_quality.error_bound_ms", sidecar
    )
    if bound < 0.0:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.error_bound_ms must be a finite non-negative "
            "JSON number"
        )
    offset_s = _require_finite_json_number(
        summary["offset_s"], "offset_quality.offset_s", sidecar
    )
    offset_applied = _require_finite_json_number(
        provenance.get("offset_applied_s"), "offset_applied_s", sidecar
    )
    if abs(offset_s - offset_applied) > 1e-6:
        raise DataSourceGateError(
            f"{sidecar.name} offset_quality.offset_s does not match offset_applied_s"
        )
    _validate_d3_sources(provenance, summary, stream_key, csv_path, sidecar)


def _validate_synthetic_provenance(
    provenance: dict,
    csv_path: Path,
    frame_df: pd.DataFrame,
    sidecar: Path,
) -> None:
    if provenance.get("artifact_kind") != SYNTHETIC_PROVENANCE_KIND:
        raise DataSourceGateError(
            f"{sidecar.name} artifact_kind must be {SYNTHETIC_PROVENANCE_KIND!r}"
        )
    version = provenance.get("format_version")
    if type(version) is not int or version != SYNTHETIC_PROVENANCE_FORMAT_VERSION:
        raise DataSourceGateError(
            f"{sidecar.name} format_version must be {SYNTHETIC_PROVENANCE_FORMAT_VERSION}"
        )
    _validate_csv_binding(provenance, csv_path, frame_df, sidecar)
    n_frames = provenance.get("n_frames")
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames != len(frame_df):
        raise DataSourceGateError(
            f"{sidecar.name} n_frames={n_frames!r} does not match {len(frame_df)} CSV rows"
        )
    frame_rate = provenance.get("frame_rate_hz")
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, int) or frame_rate != 25:
        raise DataSourceGateError(
            f"{sidecar.name} frame_rate_hz must be the canonical integer 25"
        )
    session_ids = frame_df["session_id"].drop_duplicates().tolist()
    if provenance.get("session_ids") != session_ids:
        raise DataSourceGateError(
            f"{sidecar.name} session_ids do not bind exactly to the CSV session order"
        )
    n_sessions = provenance.get("n_sessions")
    if (
        isinstance(n_sessions, bool)
        or not isinstance(n_sessions, int)
        or n_sessions != len(session_ids)
    ):
        raise DataSourceGateError(
            f"{sidecar.name} n_sessions={n_sessions!r} does not match "
            f"{len(session_ids)} CSV sessions"
        )
    generator = provenance.get("generator")
    if not isinstance(generator, dict) or set(generator) != {
        "id",
        "contract_version",
        "config",
    }:
        raise DataSourceGateError(
            f"{sidecar.name} generator must bind exactly id, contract_version, and config"
        )
    if generator.get("id") != SYNTHETIC_GENERATOR_ID:
        raise DataSourceGateError(
            f"{sidecar.name} generator.id must be the fixed project generator "
            f"{SYNTHETIC_GENERATOR_ID!r}"
        )
    contract_version = generator.get("contract_version")
    if (
        type(contract_version) is not int
        or contract_version != SYNTHETIC_GENERATOR_CONTRACT_VERSION
    ):
        raise DataSourceGateError(
            f"{sidecar.name} synthetic generator contract is stale "
            f"(found {generator.get('contract_version')!r}, required "
            f"{SYNTHETIC_GENERATOR_CONTRACT_VERSION}); re-run vap-generate"
        )
    if provenance.get("trust_model") != SYNTHETIC_TRUST_MODEL:
        raise DataSourceGateError(
            f"{sidecar.name} trust_model must state the fixed synthetic provenance boundary"
        )
    config = _synthetic_config_from_payload(generator.get("config"), sidecar)
    _verify_fixed_synthetic_generation(frame_df, config, sidecar)


def _synthetic_config_from_payload(payload: object, owner: Path) -> SyntheticConfig:
    expected_names = {item.name for item in fields(SyntheticConfig)}
    if not isinstance(payload, dict) or set(payload) != expected_names:
        raise DataSourceGateError(
            f"{owner.name} generator.config must contain exactly {sorted(expected_names)}"
        )
    defaults = SyntheticConfig()
    for item in fields(SyntheticConfig):
        value = payload[item.name]
        default = getattr(defaults, item.name)
        if isinstance(default, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            valid = valid and np.isfinite(float(value))
        if not valid:
            raise DataSourceGateError(
                f"{owner.name} generator.config.{item.name} has the wrong JSON type/value"
            )
    try:
        config = SyntheticConfig(**payload)
        # Exercise generator's semantic config validation before a sidecar can be issued.
        if config.frame_rate_hz != 25 or config.max_participants != MAX_PARTICIPANTS:
            raise ValueError("canonical synthetic data requires 25 Hz and four schema slots")
    except (TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{owner.name} generator.config is not a valid fixed synthetic config: {exc}"
        ) from exc
    return config


def _verify_fixed_synthetic_generation(
    frame_df: pd.DataFrame,
    config: SyntheticConfig,
    owner: Path,
) -> None:
    """Replay the one trusted generator and compare parsed schema, dtypes, and values.

    This is an integrity/provenance guard against accidental relabelling and ordinary
    re-signing. It is not a signature and cannot protect against an adversary able to
    modify this trusted generator implementation and every bound artifact together.
    """
    try:
        regenerated = generate_multiparty_dataset(config)
    except (TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{owner.name} fixed synthetic generator replay failed: {exc}"
        ) from exc
    buffer = io.StringIO()
    regenerated.to_csv(buffer, index=False)
    buffer.seek(0)
    reparsed = pd.read_csv(buffer, float_precision="round_trip")
    if not frame_df.equals(reparsed):
        raise DataSourceGateError(
            f"{owner.name} CSV does not exactly match deterministic replay of the fixed "
            "synthetic generator contract (ordered columns, parsed dtypes, and values)"
        )


def write_synthetic_provenance(
    dataset_csv: str | Path,
    *,
    generator_config: dict[str, object],
    overwrite: bool = False,
) -> Path:
    """Verify and bind output from the repository's one fixed synthetic generator.

    Arbitrary generator names are intentionally unsupported: the supplied full config is
    replayed and must reproduce the CSV exactly before provenance is issued.
    """
    csv_path = Path(dataset_csv)
    frame_df = _read_csv(csv_path, error_type=DataSourceGateError)
    try:
        validate_multiparty_dataframe(frame_df)
    except (TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{csv_path.name} failed multiparty schema validation: {exc}"
        ) from exc
    config = _synthetic_config_from_payload(generator_config, csv_path)
    _verify_fixed_synthetic_generation(frame_df, config, csv_path)
    sidecar = _sidecar_path(csv_path)
    if sidecar.exists() and not overwrite:
        raise DataSourceGateError(f"Refusing to overwrite synthetic provenance: {sidecar}")
    provenance = {
        "data_kind": "synthetic",
        "artifact_kind": SYNTHETIC_PROVENANCE_KIND,
        "format_version": SYNTHETIC_PROVENANCE_FORMAT_VERSION,
        **_csv_binding(csv_path, frame_df),
        "n_frames": len(frame_df),
        "frame_rate_hz": 25,
        "session_ids": frame_df["session_id"].drop_duplicates().tolist(),
        "n_sessions": int(frame_df["session_id"].nunique()),
        "generator": {
            "id": SYNTHETIC_GENERATOR_ID,
            "contract_version": SYNTHETIC_GENERATOR_CONTRACT_VERSION,
            "config": asdict(config),
        },
        "trust_model": SYNTHETIC_TRUST_MODEL,
    }
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def _read_and_validate_provenance(
    csv_path: Path,
    frame_df: pd.DataFrame,
    *,
    data_kind: str,
    analysis_base_authority: object | None = None,
) -> dict | None:
    _validate_data_kind(data_kind)
    sidecar = _sidecar_path(csv_path)
    session_ids = frame_df["session_id"].drop_duplicates().tolist()
    if not sidecar.exists():
        raise DataSourceGateError(
            f"{data_kind.capitalize()} data CSV {csv_path.name} has no {sidecar.name}; "
            "bare CSV files are never trusted from a caller-supplied data_kind"
        )
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceGateError(f"Could not parse provenance sidecar {sidecar}: {exc}") from exc
    if not isinstance(provenance, dict):
        raise DataSourceGateError(f"{sidecar.name} must contain a JSON object")
    declared_kind = provenance.get("data_kind")
    if declared_kind != data_kind:
        raise DataSourceGateError(
            f"{sidecar.name} data_kind={declared_kind!r} does not match requested "
            f"data_kind={data_kind!r}; real provenance cannot be relabelled synthetic"
        )
    if data_kind == "synthetic":
        _validate_synthetic_provenance(provenance, csv_path, frame_df, sidecar)
        return provenance
    artifact_kind = provenance.get("artifact_kind")
    if artifact_kind not in {
        REAL_INTAKE_ARTIFACT_KIND,
        REAL_ANALYSIS_BASE_ARTIFACT_KIND,
        REAL_ASSEMBLED_ARTIFACT_KIND,
    }:
        supported_real_kinds = sorted(
            [
                REAL_INTAKE_ARTIFACT_KIND,
                REAL_ANALYSIS_BASE_ARTIFACT_KIND,
                REAL_ASSEMBLED_ARTIFACT_KIND,
            ]
        )
        raise DataSourceGateError(
            f"{sidecar.name} artifact_kind must be one of "
            f"{supported_real_kinds}"
        )
    format_version = provenance.get("format_version")
    if (
        type(format_version) is not int
        or format_version != REAL_PROVENANCE_FORMAT_VERSION
    ):
        if type(format_version) is int and format_version == 1:
            raise DataSourceGateError(
                f"{sidecar.name} is stale real provenance v1; rebuild its base with "
                "vap-intake v2 from the sealed HPE→Transcript crosswalk and re-run any "
                "assembly (do not relabel or edit the old table)"
            )
        raise DataSourceGateError(
            f"{sidecar.name} format_version must be {REAL_PROVENANCE_FORMAT_VERSION}"
        )
    _validate_csv_binding(provenance, csv_path, frame_df, sidecar)
    if len(session_ids) != 1:
        raise DataSourceGateError(
            f"{sidecar.name} can bind only one session, but {csv_path.name} contains "
            f"{session_ids}"
        )
    declared_session = provenance.get("session_id")
    if not isinstance(declared_session, str) or declared_session != session_ids[0]:
        raise DataSourceGateError(
            f"{sidecar.name} session_id={declared_session!r} does not bind exactly to "
            f"CSV session {session_ids[0]!r}"
        )
    n_frames = provenance.get("n_frames")
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames != len(frame_df):
        raise DataSourceGateError(
            f"{sidecar.name} n_frames={n_frames!r} does not match {len(frame_df)} CSV rows"
        )
    frame_rate = provenance.get("frame_rate_hz")
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, int) or frame_rate != 25:
        raise DataSourceGateError(
            f"{sidecar.name} frame_rate_hz must be the canonical integer 25"
        )
    if not isinstance(provenance.get("trainable"), bool):
        raise DataSourceGateError(f"{sidecar.name} trainable must be a JSON boolean")
    if provenance.get("review_status") != "confirmed":
        raise DataSourceGateError(
            f"{sidecar.name} review_status must be 'confirmed' for formal real data"
        )
    _require_normalised_id(provenance, "group_id", sidecar)
    _require_normalised_id(provenance, "scenario_id", sidecar)
    participant_to_slot = _validate_participant_to_slot(provenance, frame_df, sidecar)
    real_columns = _require_string_list(provenance, "real_columns", sidecar)
    placeholders = _require_string_list(provenance, "placeholder_columns", sidecar)
    overlap = sorted(set(real_columns) & set(placeholders))
    if overlap:
        raise DataSourceGateError(
            f"{sidecar.name} marks columns as both real and placeholder: {overlap}"
        )
    declared = set(real_columns) | set(placeholders)
    required = _required_real_declarations(frame_df)
    undeclared = sorted(required - declared)
    extra = sorted(declared - required)
    if undeclared or extra:
        raise DataSourceGateError(
            f"{sidecar.name} real_columns/placeholder_columns must partition exactly all "
            f"present-slot canonical inputs + gold VAD and global inputs; "
            f"undeclared={undeclared}, extra={extra}"
        )
    missing_from_csv = sorted(
        column for column in set(real_columns) | set(placeholders) if column not in frame_df
    )
    if missing_from_csv:
        raise DataSourceGateError(
            f"{sidecar.name} lists columns missing from its CSV: {missing_from_csv}"
        )
    missing_vad, degenerate_vad = _vad_trainability(frame_df, set(real_columns))
    expected_trainable = not missing_vad and not degenerate_vad
    if provenance["trainable"] is not expected_trainable:
        raise DataSourceGateError(
            f"{sidecar.name} trainable={provenance['trainable']!r} disagrees with "
            f"replayed gold-VAD readiness {expected_trainable!r}"
        )
    expected_trainability = {
        "required_gold_vad_columns": _present_vad_columns(frame_df),
        "missing_gold_vad_columns": missing_vad,
        "degenerate_gold_vad_columns": degenerate_vad,
    }
    if artifact_kind == REAL_ASSEMBLED_ARTIFACT_KIND:
        if provenance.get("trainability") != expected_trainability:
            raise DataSourceGateError(
                f"{sidecar.name} trainability does not match replayed gold-VAD readiness"
            )
    elif "trainability" in provenance:
        raise DataSourceGateError(
            f"{sidecar.name} unassembled real provenance must not declare assembled "
            "trainability details"
        )
    if "doa_angle_deg" not in placeholders or "doa_angle_deg" in real_columns:
        raise DataSourceGateError(
            f"{sidecar.name} must keep doa_angle_deg declared as placeholder/inert on "
            "real recordings"
        )
    try:
        real_doa = frame_df["doa_angle_deg"].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{sidecar.name} real-data doa_angle_deg must be a numeric inert zero constant"
        ) from exc
    if not np.equal(real_doa, 0.0).all():
        raise DataSourceGateError(
            f"{sidecar.name} real-data doa_angle_deg must remain strictly zero; "
            "geometry-simulated or oracle DoA is forbidden"
        )
    declared_inert = provenance.get("inert_columns")
    if artifact_kind == REAL_ASSEMBLED_ARTIFACT_KIND:
        if declared_inert != sorted(INERT_REAL_COLUMNS):
            raise DataSourceGateError(
                f"{sidecar.name} inert_columns must be exactly "
                f"{sorted(INERT_REAL_COLUMNS)} for assembled real data"
            )
    elif declared_inert is not None and declared_inert != sorted(INERT_REAL_COLUMNS):
        raise DataSourceGateError(
            f"{sidecar.name} cannot declare real-data inert_columns other than "
            f"{sorted(INERT_REAL_COLUMNS)}"
        )
    _validate_real_column_sources(
        provenance, real_columns, participant_to_slot, csv_path, sidecar
    )
    _validate_intake_sources(
        provenance, participant_to_slot, csv_path, sidecar
    )
    _validate_offset_summary(provenance, csv_path, sidecar)
    analysis_contract = provenance.get("analysis_window_contract")
    if artifact_kind == REAL_INTAKE_ARTIFACT_KIND:
        if analysis_contract is not None:
            raise DataSourceGateError(
                f"{sidecar.name} pose-intake provenance cannot claim an analysis window"
            )
        _validate_identity_contract_and_replay_pose(
            provenance, frame_df, csv_path, sidecar
        )
    elif artifact_kind == REAL_ANALYSIS_BASE_ARTIFACT_KIND or analysis_contract is not None:
        if analysis_contract is None:
            raise DataSourceGateError(
                f"{sidecar.name} analysis-base artifact has no analysis_window_contract"
            )
        try:
            from hri_turn_taking.data.analysis_base import (
                validate_analysis_base_artifact,
            )

            validate_analysis_base_artifact(
                csv_path,
                frame_df,
                provenance,
                sidecar,
                authority=analysis_base_authority,
            )
        except (OSError, ValueError) as exc:
            raise DataSourceGateError(
                f"{sidecar.name} analysis-window replay failed: {exc}"
            ) from exc
    else:
        _validate_identity_contract_and_replay_pose(
            provenance, frame_df, csv_path, sidecar
        )
    intake_pose_columns = _expected_intake_pose_columns(frame_df)
    if artifact_kind in {REAL_INTAKE_ARTIFACT_KIND, REAL_ANALYSIS_BASE_ARTIFACT_KIND}:
        if set(real_columns) != intake_pose_columns:
            raise DataSourceGateError(
                f"{sidecar.name} unassembled pose-base real_columns must be exactly "
                f"{sorted(intake_pose_columns)}; feature/gold promotion requires assembly"
            )
        if provenance.get("feature_sources", []) != [] or provenance.get("assembly") is not None:
            raise DataSourceGateError(
                f"{sidecar.name} unassembled pose-base provenance cannot contain "
                "assembly/feature_sources"
            )
    else:
        if not intake_pose_columns.issubset(real_columns):
            raise DataSourceGateError(
                f"{sidecar.name} assembled real_columns dropped required intake pose columns"
            )
        assembly_record = provenance.get("assembly")
        required_assembly_fields = {
            "base_csv",
            "base_sha256",
            "output_csv",
            "supplied_columns",
        }
        if (
            not isinstance(assembly_record, dict)
            or set(assembly_record) != required_assembly_fields
        ):
            raise DataSourceGateError(
                f"{sidecar.name} assembled artifact must retain exactly the assembly "
                f"fields {sorted(required_assembly_fields)}"
            )
        declared_output = assembly_record.get("output_csv")
        if (
            not isinstance(declared_output, str)
            or Path(declared_output).resolve() != csv_path.resolve()
        ):
            raise DataSourceGateError(
                f"{sidecar.name} assembly.output_csv must bind exactly to its current CSV path"
            )
        _resolve_bound_source(
            {
                "path": assembly_record["base_csv"],
                "sha256": assembly_record["base_sha256"],
            },
            owner_path=sidecar,
            error_type=DataSourceGateError,
            field_name="assembly.base_csv",
            forbidden_paths={csv_path.resolve(), sidecar.resolve()},
        )
        manifest_columns = _validate_assembled_feature_sources(
            provenance, frame_df, csv_path, sidecar
        )
        supplied_columns = assembly_record["supplied_columns"]
        if (
            not isinstance(supplied_columns, list)
            or any(not isinstance(column, str) for column in supplied_columns)
            or len(supplied_columns) != len(set(supplied_columns))
            or set(supplied_columns) != manifest_columns
        ):
            raise DataSourceGateError(
                f"{sidecar.name} assembly.supplied_columns must match the uniquely "
                "covering feature manifests"
            )
        promoted_columns = set(real_columns) - intake_pose_columns
        manifest_promotions = manifest_columns - INERT_REAL_COLUMNS
        if promoted_columns != manifest_promotions:
            raise DataSourceGateError(
                f"{sidecar.name} promoted real/gold columns must be uniquely covered by "
                "feature manifests; "
                f"promoted={sorted(promoted_columns)}, manifest={sorted(manifest_promotions)}"
            )
    return provenance


def _load_dataset_file(
    csv_path: Path,
    *,
    data_kind: str,
    analysis_base_authority: object | None = None,
) -> _DatasetFile:
    sidecar = _sidecar_path(csv_path)
    data_hash_before = _sha256(csv_path) if csv_path.is_file() else None
    provenance_hash_before = _sha256(sidecar) if sidecar.is_file() else None
    frame_df = _read_csv(csv_path, error_type=DataSourceGateError)
    try:
        validate_multiparty_dataframe(frame_df)
    except (TypeError, ValueError) as exc:
        raise DataSourceGateError(
            f"{csv_path.name} failed multiparty schema validation: {exc}"
        ) from exc
    provenance = _read_and_validate_provenance(
        csv_path,
        frame_df,
        data_kind=data_kind,
        analysis_base_authority=analysis_base_authority,
    )
    data_hash_after = _sha256(csv_path)
    provenance_hash_after = _sha256(sidecar)
    if data_hash_before != data_hash_after or provenance_hash_before != provenance_hash_after:
        raise DataSourceGateError(
            f"Data source {csv_path.name} or its provenance changed while being loaded; "
            "refusing a mixed snapshot"
        )
    return _DatasetFile(
        path=csv_path,
        frame_df=frame_df,
        provenance=provenance,
        data_sha256=data_hash_after,
        provenance_sha256=provenance_hash_after,
    )


def _load_dataset_file_with_analysis_authority(
    csv_path: Path,
    *,
    data_kind: str,
    analysis_base_authority: object | None,
) -> _DatasetFile:
    """Preserve the ordinary loader call shape unless authority is explicit."""

    if analysis_base_authority is None:
        return _load_dataset_file(csv_path, data_kind=data_kind)
    return _load_dataset_file(
        csv_path,
        data_kind=data_kind,
        analysis_base_authority=analysis_base_authority,
    )


def _present_vad_columns(frame_df: pd.DataFrame) -> list[str]:
    first = frame_df.iloc[0]
    return [
        participant_column(slot, "vad")
        for slot in range(1, MAX_PARTICIPANTS + 1)
        if float(first[participant_column(slot, "present")]) == 1.0
    ]


def _vad_trainability(
    frame_df: pd.DataFrame, real_columns: set[str] | None
) -> tuple[list[str], list[str]]:
    required = _present_vad_columns(frame_df)
    missing = (
        []
        if real_columns is None
        else [column for column in required if column not in real_columns]
    )
    degenerate = [
        column
        for column in required
        if set(frame_df[column].to_numpy(dtype=float).tolist()) != {0.0, 1.0}
    ]
    return missing, degenerate


def _target_issues(dataset_file: _DatasetFile) -> list[str]:
    """Gold-target readiness only; structural provenance was already gated."""
    issues: list[str] = []
    provenance = dataset_file.provenance
    for session_id, group in dataset_file.frame_df.groupby("session_id", sort=False):
        if provenance is None or provenance.get("data_kind") == "synthetic":
            real_columns = None
        else:
            real_columns = set(provenance["real_columns"])
        missing, degenerate = _vad_trainability(group, real_columns)
        if missing:
            issues.append(f"{session_id}: missing real gold VAD columns {missing}")
        if degenerate:
            issues.append(
                f"{session_id}: gold VAD columns lack both speech and silence labels "
                f"{degenerate}"
            )
    return issues


def _normalise_feature_paths(feature_csvs: object) -> list[Path]:
    if isinstance(feature_csvs, (str, Path)):
        paths = [Path(feature_csvs)]
    else:
        try:
            paths = [Path(path) for path in feature_csvs]  # type: ignore[union-attr]
        except TypeError as exc:
            raise AssemblyGateError("feature_csvs must be a path or an iterable of paths") from exc
    if not paths:
        raise AssemblyGateError("At least one feature CSV is required for assembly")
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise AssemblyGateError("feature_csvs contains the same file more than once")
    return paths


def _feature_manifest_path(feature_path: Path) -> Path:
    return feature_path.with_suffix(".manifest.json")


def _require_sha256(value: object, name: str, manifest_path: Path) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AssemblyGateError(
            f"{manifest_path.name} {name} must be a lowercase 64-character SHA-256"
        )
    return value


def _validate_feature_manifest(
    feature_path: Path,
    feature_columns: list[str],
    base_df: pd.DataFrame,
    base_provenance: dict,
    *,
    forbidden_generated_paths: set[Path] | None = None,
) -> dict:
    manifest_path = _feature_manifest_path(feature_path)
    if not manifest_path.is_file():
        raise AssemblyGateError(
            f"{feature_path.name} has no sibling {manifest_path.name}; every real feature "
            "or gold-target CSV requires a verifiable source manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyGateError(f"Could not parse feature manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AssemblyGateError(f"{manifest_path.name} must contain a JSON object")
    version = manifest.get("format_version")
    if type(version) is not int or version != FEATURE_MANIFEST_FORMAT_VERSION:
        raise AssemblyGateError(
            f"{manifest_path.name} format_version must be {FEATURE_MANIFEST_FORMAT_VERSION}"
        )
    if manifest.get("artifact_kind") != FEATURE_MANIFEST_KIND:
        raise AssemblyGateError(
            f"{manifest_path.name} artifact_kind must be {FEATURE_MANIFEST_KIND!r}"
        )
    session_ids = base_df["session_id"].drop_duplicates().tolist()
    if manifest.get("session_id") != session_ids[0]:
        raise AssemblyGateError(
            f"{manifest_path.name} session_id does not bind to {session_ids[0]!r}"
        )
    if manifest.get("participant_to_slot") != base_provenance["participant_to_slot"]:
        raise AssemblyGateError(
            f"{manifest_path.name} participant_to_slot does not exactly match base provenance"
        )
    supplied_columns = manifest.get("supplied_columns")
    if (
        not isinstance(supplied_columns, list)
        or any(not isinstance(column, str) for column in supplied_columns)
        or supplied_columns != feature_columns
        or len(supplied_columns) != len(set(supplied_columns))
    ):
        raise AssemblyGateError(
            f"{manifest_path.name} supplied_columns must exactly match the feature CSV "
            f"columns in order: {feature_columns}"
        )
    expected_csv_hash = _require_sha256(
        manifest.get("feature_csv_sha256"), "feature_csv_sha256", manifest_path
    )
    actual_csv_hash = _sha256(feature_path)
    if expected_csv_hash != actual_csv_hash:
        raise AssemblyGateError(
            f"{manifest_path.name} feature_csv_sha256 does not match {feature_path.name}"
        )
    feature_df = _read_csv(feature_path, error_type=AssemblyGateError)

    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise AssemblyGateError(
            f"{manifest_path.name} source_files must be a non-empty JSON array"
        )
    resolved_sources: dict[str, tuple[Path, str]] = {}
    forbidden_sources = {feature_path.resolve(), manifest_path.resolve()}
    forbidden_sources.update(
        path.resolve() for path in (forbidden_generated_paths or set())
    )
    forbidden_generated_hashes = {
        _sha256(path.resolve())
        for path in (forbidden_generated_paths or set())
        if path.resolve().is_file() and path.resolve().suffix.lower() == ".csv"
    }
    for index, source in enumerate(source_files):
        if not isinstance(source, dict):
            raise AssemblyGateError(
                f"{manifest_path.name} source_files[{index}] must be an object"
            )
        source_name = source.get("path")
        if not isinstance(source_name, str) or not source_name.strip():
            raise AssemblyGateError(
                f"{manifest_path.name} source_files[{index}].path must be non-empty"
            )
        if source_name in resolved_sources:
            raise AssemblyGateError(
                f"{manifest_path.name} source_files contains duplicate path {source_name!r}"
            )
        resolved_source = _resolve_bound_source(
            source,
            owner_path=manifest_path,
            error_type=AssemblyGateError,
            field_name=f"source_files[{index}]",
            forbidden_paths=forbidden_sources,
        )
        resolved_path = resolved_source[0]
        source_hash = resolved_source[1]
        if source_hash in forbidden_generated_hashes:
            raise AssemblyGateError(
                f"{manifest_path.name} cannot use a byte-for-byte copy of a generated "
                "VAP intake/assembly artifact as independent feature ownership"
            )
        if resolved_path.suffix.lower() == ".csv":
            try:
                with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    source_header = next(csv.reader(handle), [])
            except OSError as exc:
                raise AssemblyGateError(
                    f"Could not read source CSV header {resolved_path}: {exc}"
                ) from exc
            if source_header == base_df.columns.tolist():
                raise AssemblyGateError(
                    f"{manifest_path.name} cannot use a copied generated VAP wide table "
                    "as independent feature ownership"
                )
        generated_sidecar = _sidecar_path(resolved_path)
        if generated_sidecar.is_file():
            try:
                generated_provenance = json.loads(
                    generated_sidecar.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise AssemblyGateError(
                    f"{manifest_path.name} source {resolved_path} has an unreadable "
                    "generated-artifact sidecar"
                ) from exc
            if isinstance(generated_provenance, dict) and generated_provenance.get(
                "artifact_kind"
            ) in {
                REAL_INTAKE_ARTIFACT_KIND,
                REAL_ANALYSIS_BASE_ARTIFACT_KIND,
                REAL_ASSEMBLED_ARTIFACT_KIND,
            }:
                raise AssemblyGateError(
                    f"{manifest_path.name} cannot use generated VAP intake/analysis/assembly "
                    f"artifact {resolved_path} as independent feature ownership"
                )
        resolved_sources[source_name] = resolved_source

    has_target = any(column in set(vad_columns()) for column in feature_columns)
    has_input = any(column in set(input_feature_columns()) for column in feature_columns)
    if has_target and has_input:
        raise AssemblyGateError(
            f"{feature_path.name} mixes model inputs and gold targets; use separate files/manifests"
        )
    expected_purpose = "gold_target" if has_target else "model_input"
    if manifest.get("purpose") != expected_purpose:
        raise AssemblyGateError(
            f"{manifest_path.name} purpose must be {expected_purpose!r} for its supplied columns"
        )
    causality = manifest.get("causality")
    if not isinstance(causality, dict):
        raise AssemblyGateError(f"{manifest_path.name} causality must be a JSON object")
    causal = causality.get("causal")
    alignment = causality.get("window_alignment")
    lookahead = causality.get("lookahead_s")
    if expected_purpose == "model_input":
        if causal is not True or alignment != "trailing":
            raise AssemblyGateError(
                f"{manifest_path.name} model_input must declare causal=true and "
                "window_alignment='trailing'"
            )
        lookahead_value = _require_finite_json_number(
            lookahead, "causality.lookahead_s", manifest_path
        )
        if lookahead_value != 0.0:
            raise AssemblyGateError(
                f"{manifest_path.name} model_input must declare lookahead_s=0"
            )
    elif causal is True:
        lookahead_value = _require_finite_json_number(
            lookahead, "causality.lookahead_s", manifest_path
        )
        if lookahead_value != 0.0 or alignment != "trailing":
            raise AssemblyGateError(
                f"{manifest_path.name} causal gold_target must be trailing with lookahead_s=0"
            )
    elif causal is False:
        if alignment != "offline" or lookahead is not None:
            raise AssemblyGateError(
                f"{manifest_path.name} offline gold_target must declare "
                "window_alignment='offline' and lookahead_s=null"
            )
    else:
        raise AssemblyGateError(
            f"{manifest_path.name} causality.causal must be a JSON boolean"
        )

    column_sources = manifest.get("column_sources")
    if not isinstance(column_sources, dict) or set(column_sources) != set(feature_columns):
        raise AssemblyGateError(
            f"{manifest_path.name} column_sources must cover exactly {feature_columns}"
        )
    resolved_column_sources: dict[str, dict[str, object]] = {}
    source_frames: dict[Path, pd.DataFrame] = {}
    source_use: dict[tuple[Path, str], str | None] = {}
    for column in feature_columns:
        record = column_sources[column]
        if not isinstance(record, dict):
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] must be an object"
            )
        if "participant" not in record or "slot" not in record:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] must explicitly bind "
                "participant and slot (use JSON null for a global column)"
            )
        expected_participant, expected_slot = _expected_column_binding(
            column, base_provenance["participant_to_slot"]
        )
        if record["participant"] != expected_participant or record["slot"] != expected_slot:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] does not match base "
                f"participant/slot binding participant={expected_participant!r}, "
                f"slot={expected_slot!r}"
            )
        if record.get("source_participant") != expected_participant:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}].source_participant "
                f"must be {expected_participant!r}"
            )
        source_name = record.get("source_path")
        if not isinstance(source_name, str) or source_name not in resolved_sources:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}].source_path must name "
                "one exact source_files path"
            )
        source_hash = _require_sha256(
            record.get("source_sha256"),
            f"column_sources[{column!r}].source_sha256",
            manifest_path,
        )
        resolved_path, declared_hash = resolved_sources[source_name]
        if source_hash != declared_hash:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}].source_sha256 does not "
                "match its source_files entry"
            )
        source_column = record.get("source_column")
        if not isinstance(source_column, str) or not source_column.strip():
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}].source_column must be "
                "a non-empty upstream column name"
            )
        if source_column != column:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] uses the identity "
                "contract, so source_column must exactly equal the canonical output "
                f"column {column!r}; got {source_column!r}"
            )
        source_keys = record.get("source_keys")
        if (
            not isinstance(source_keys, dict)
            or set(source_keys) != set(KEY_COLUMNS)
            or any(not isinstance(source_keys[key], str) or not source_keys[key].strip()
                   for key in KEY_COLUMNS)
        ):
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}].source_keys must map "
                "session_id and timestamp_s to concrete upstream columns"
            )
        value_transform = record.get("value_transform")
        if value_transform != {"kind": "identity"}:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}].value_transform must be "
                "the supported deterministic identity transform"
            )
        if resolved_path.suffix.lower() != ".csv":
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] exact value replay "
                "requires an upstream CSV"
            )
        source_df = source_frames.get(resolved_path)
        if source_df is None:
            source_df = _read_csv(resolved_path, error_type=AssemblyGateError)
            source_frames[resolved_path] = source_df
        required_source_columns = {source_column, *source_keys.values()}
        missing_source_columns = sorted(required_source_columns - set(source_df.columns))
        if missing_source_columns:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] upstream CSV lacks "
                f"columns {missing_source_columns}"
            )
        if len(source_df) != len(feature_df):
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] upstream row count "
                "does not match feature CSV"
            )
        source_sessions = source_df[source_keys["session_id"]].to_numpy(dtype=object)
        source_timestamps = source_df[source_keys["timestamp_s"]].to_numpy(dtype=float)
        if not np.array_equal(
            source_sessions, feature_df["session_id"].to_numpy(dtype=object)
        ) or not np.array_equal(
            source_timestamps, feature_df["timestamp_s"].to_numpy(dtype=float)
        ):
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] upstream keys/order "
                "do not match the feature CSV"
            )
        try:
            source_values = source_df[source_column].to_numpy(dtype=float)
            feature_values = feature_df[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] values must be numeric"
            ) from exc
        if not np.isfinite(source_values).all() or not np.array_equal(
            source_values, feature_values
        ):
            raise AssemblyGateError(
                f"{manifest_path.name} column_sources[{column!r}] feature values do not "
                "match source_column under the declared deterministic transform"
            )
        source_identity = (resolved_path, source_column)
        previous_participant = source_use.get(source_identity)
        if source_identity in source_use and previous_participant != expected_participant:
            raise AssemblyGateError(
                f"{manifest_path.name} reuses upstream {source_column!r} for different "
                "output participants"
            )
        source_use[source_identity] = expected_participant
        resolved_column_sources[column] = {
            "participant": expected_participant,
            "slot": expected_slot,
            "purpose": expected_purpose,
            "path": str(resolved_path),
            "sha256": source_hash,
            "source_column": source_column,
            "source_participant": expected_participant,
            "source_keys": source_keys,
            "value_transform": value_transform,
        }
    validated = dict(manifest)
    validated["_resolved_column_sources"] = resolved_column_sources
    return validated


def _validate_assembled_feature_sources(
    provenance: dict,
    frame_df: pd.DataFrame,
    csv_path: Path,
    sidecar: Path,
) -> set[str]:
    feature_sources = provenance.get("feature_sources", [])
    if not isinstance(feature_sources, list):
        raise DataSourceGateError(f"{sidecar.name} feature_sources must be a JSON array")
    if provenance.get("assembly") is not None and not feature_sources:
        raise DataSourceGateError(
            f"{sidecar.name} assembled provenance must retain feature_sources"
        )
    seen_columns: set[str] = set()
    seen_upstream_columns: dict[tuple[Path, str], str | None] = {}
    forbidden = {csv_path.resolve(), sidecar.resolve()}
    assembly_record = provenance.get("assembly")
    if isinstance(assembly_record, dict) and isinstance(
        assembly_record.get("base_csv"), str
    ):
        bound_base = Path(assembly_record["base_csv"]).resolve()
        forbidden.update({bound_base, _sidecar_path(bound_base).resolve()})
    for index, source_record in enumerate(feature_sources):
        if not isinstance(source_record, dict):
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}] must be an object"
            )
        feature_path, _ = _resolve_bound_source(
            source_record,
            owner_path=sidecar,
            error_type=DataSourceGateError,
            field_name=f"feature_sources[{index}]",
            forbidden_paths=forbidden,
        )
        manifest_path_value = source_record.get("manifest_path")
        manifest_sha256 = source_record.get("manifest_sha256")
        manifest_path, _ = _resolve_bound_source(
            {"path": manifest_path_value, "sha256": manifest_sha256},
            owner_path=sidecar,
            error_type=DataSourceGateError,
            field_name=f"feature_sources[{index}].manifest",
            forbidden_paths=forbidden | {feature_path},
        )
        expected_manifest_path = _feature_manifest_path(feature_path).resolve()
        if manifest_path != expected_manifest_path:
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}] manifest_path is not the "
                "feature CSV's sibling manifest"
            )
        columns = source_record.get("columns")
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) for column in columns)
            or not columns
        ):
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}].columns must be non-empty strings"
            )
        repeated = sorted(set(columns) & seen_columns)
        if repeated:
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources repeat supplied columns {repeated}"
            )
        seen_columns.update(columns)
        try:
            validated_manifest = _validate_feature_manifest(
                feature_path,
                columns,
                frame_df,
                provenance,
                forbidden_generated_paths=forbidden,
            )
        except AssemblyGateError as exc:
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}] manifest failed revalidation: {exc}"
            ) from exc
        for key in ("purpose", "causality", "column_sources", "source_files"):
            if source_record.get(key) != validated_manifest.get(key):
                raise DataSourceGateError(
                    f"{sidecar.name} feature_sources[{index}].{key} does not match its "
                    "hash-bound manifest"
                )
        resolved_sources = validated_manifest["_resolved_column_sources"]
        for column in columns:
            resolved_record = resolved_sources[column]
            upstream_identity = (
                Path(resolved_record["path"]).resolve(),
                str(resolved_record["source_column"]),
            )
            source_participant = resolved_record["source_participant"]
            previous_participant = seen_upstream_columns.get(upstream_identity)
            if (
                upstream_identity in seen_upstream_columns
                and previous_participant != source_participant
            ):
                raise DataSourceGateError(
                    f"{sidecar.name} reuses one upstream source column across different "
                    "participants in separate feature manifests"
                )
            seen_upstream_columns[upstream_identity] = source_participant
            if column in INERT_REAL_COLUMNS:
                continue
            if provenance.get("column_sources", {}).get(column) != resolved_record:
                raise DataSourceGateError(
                    f"{sidecar.name} column_sources[{column!r}] does not exactly match "
                    "the uniquely covering feature manifest"
                )
        feature_df = _read_csv(feature_path, error_type=DataSourceGateError)
        n_frames = source_record.get("n_frames")
        if (
            isinstance(n_frames, bool)
            or not isinstance(n_frames, int)
            or n_frames != len(feature_df)
        ):
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}].n_frames does not match its CSV"
            )
        if len(feature_df) != len(frame_df) or any(key not in feature_df for key in KEY_COLUMNS):
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}] no longer matches dataset rows"
            )
        if not np.array_equal(
            feature_df["session_id"].to_numpy(dtype=object),
            frame_df["session_id"].to_numpy(dtype=object),
        ) or not np.array_equal(
            feature_df["timestamp_s"].to_numpy(dtype=float),
            frame_df["timestamp_s"].to_numpy(dtype=float),
        ):
            raise DataSourceGateError(
                f"{sidecar.name} feature_sources[{index}] keys/order differ from dataset"
            )
        for column in columns:
            if column not in frame_df or not np.array_equal(
                feature_df[column].to_numpy(dtype=float),
                frame_df[column].to_numpy(dtype=float),
            ):
                raise DataSourceGateError(
                    f"{sidecar.name} feature_sources[{index}] column {column!r} no longer "
                    "matches the assembled dataset"
                )
    return seen_columns


def _validate_feature_file(
    feature_path: Path,
    base_df: pd.DataFrame,
    base_provenance: dict,
    placeholder_columns: set[str],
    supplied_columns: set[str],
    base_path: Path,
) -> tuple[pd.DataFrame, list[str], dict]:
    feature_df = _read_csv(feature_path, error_type=AssemblyGateError)
    missing_keys = [key for key in KEY_COLUMNS if key not in feature_df]
    if missing_keys:
        raise AssemblyGateError(
            f"{feature_path.name} is missing exact-key columns: {missing_keys}"
        )
    feature_columns = [column for column in feature_df if column not in KEY_COLUMNS]
    if not feature_columns:
        raise AssemblyGateError(f"{feature_path.name} supplies no feature or gold-VAD columns")
    unknown = sorted(set(feature_columns) - CANONICAL_SUPPLY_COLUMNS)
    if unknown:
        raise AssemblyGateError(
            f"{feature_path.name} supplies non-canonical columns: {unknown}"
        )
    repeated = sorted(set(feature_columns) & supplied_columns)
    if repeated:
        raise AssemblyGateError(
            f"Feature columns are supplied by more than one file: {repeated}"
        )
    not_placeholders = sorted(set(feature_columns) - placeholder_columns)
    if not_placeholders:
        raise AssemblyGateError(
            f"{feature_path.name} would overwrite non-placeholder columns: "
            f"{not_placeholders}"
        )
    if len(feature_df) != len(base_df):
        raise AssemblyGateError(
            f"{feature_path.name} has {len(feature_df)} rows; exact-key base has "
            f"{len(base_df)}"
        )
    feature_sessions = feature_df["session_id"].to_numpy(dtype=object)
    base_sessions = base_df["session_id"].to_numpy(dtype=object)
    try:
        feature_timestamps = feature_df["timestamp_s"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssemblyGateError(f"{feature_path.name} timestamp_s must be numeric") from exc
    base_timestamps = base_df["timestamp_s"].to_numpy(dtype=float)
    if not np.array_equal(feature_sessions, base_sessions) or not np.array_equal(
        feature_timestamps, base_timestamps
    ):
        raise AssemblyGateError(
            f"{feature_path.name} keys/order do not exactly match the intake table; "
            "sorting, interpolation, and tolerance joins are intentionally forbidden"
        )
    for column in feature_columns:
        try:
            values = feature_df[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AssemblyGateError(
                f"{feature_path.name} column {column!r} must be numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise AssemblyGateError(
                f"{feature_path.name} column {column!r} contains NaN/Inf"
            )
    if "doa_angle_deg" in feature_columns and (
        feature_df["doa_angle_deg"].to_numpy(dtype=float) != 0.0
    ).any():
        raise AssemblyGateError(
            "doa_angle_deg must remain an inert zero constant on real recordings; "
            "geometry-simulated or oracle DoA is forbidden"
        )
    manifest = _validate_feature_manifest(
        feature_path,
        feature_columns,
        base_df,
        base_provenance,
        forbidden_generated_paths={
            base_path.resolve(),
            _sidecar_path(base_path).resolve(),
        },
    )
    return feature_df, feature_columns, manifest


def assemble_session(
    base_csv: str | Path,
    feature_csvs: object,
    output_csv: str | Path,
    *,
    analysis_base_authority: object | None = None,
) -> AssemblyResult:
    """Exact-key merge one intake session and materialise CSV + provenance.

    Only columns listed as placeholders by the base sidecar may be supplied.
    ``trainable`` becomes true only when every present participant has a real
    gold-VAD source and each such column contains both speech and silence.
    Remaining input placeholders are reported but do not counterfeit this
    target-readiness gate.  Real-data DoA remains a zero-valued inert column.
    """
    base_path = Path(base_csv).resolve()
    output_path = Path(output_csv).resolve()
    feature_paths = _normalise_feature_paths(feature_csvs)
    if output_path.suffix.lower() != ".csv":
        raise AssemblyGateError("output_csv must have a .csv suffix")
    input_paths = {base_path.resolve(), *(path.resolve() for path in feature_paths)}
    if output_path.resolve() in input_paths:
        raise AssemblyGateError("output_csv must differ from the base and every feature CSV")
    output_sidecar = _sidecar_path(output_path)
    if output_path.exists() or output_sidecar.exists():
        raise AssemblyGateError(
            f"Refusing to overwrite existing assembly output: {output_path} or "
            f"{output_sidecar}"
        )

    try:
        base_file = _load_dataset_file_with_analysis_authority(
            base_path,
            data_kind="real",
            analysis_base_authority=analysis_base_authority,
        )
    except DataSourceGateError as exc:
        raise AssemblyGateError(str(exc)) from exc
    base_df = base_file.frame_df
    provenance = base_file.provenance
    assert provenance is not None
    session_ids = base_df["session_id"].drop_duplicates().tolist()
    if len(session_ids) != 1:
        raise AssemblyGateError("base_csv must contain exactly one real intake session")
    if provenance.get("analysis_window_contract") is not None:
        try:
            from hri_turn_taking.data.analysis_base import (
                validate_analysis_descendant_destination,
            )

            validate_analysis_descendant_destination(
                session_ids[0],
                output_path,
                authority=analysis_base_authority,
            )
        except (OSError, ValueError) as exc:
            raise AssemblyGateError(
                f"Analysis descendant destination is not authorised: {exc}"
            ) from exc

    placeholders = set(provenance["placeholder_columns"])
    supplied: set[str] = set()
    merged = base_df.copy()
    source_records: list[dict] = []
    column_sources = provenance.get("column_sources", {})
    if not isinstance(column_sources, dict):
        raise AssemblyGateError("base provenance column_sources must be a JSON object")
    column_sources = dict(column_sources)

    for feature_path in feature_paths:
        feature_df, feature_columns, manifest = _validate_feature_file(
            feature_path,
            base_df,
            provenance,
            placeholders,
            supplied,
            base_path,
        )
        digest = _sha256(feature_path)
        resolved_source = str(feature_path.resolve())
        manifest_path = _feature_manifest_path(feature_path)
        source_records.append(
            {
                "path": resolved_source,
                "sha256": digest,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "columns": feature_columns,
                "n_frames": len(feature_df),
                "purpose": manifest["purpose"],
                "causality": manifest["causality"],
                "column_sources": manifest["column_sources"],
                "source_files": manifest["source_files"],
            }
        )
        for column in feature_columns:
            merged[column] = feature_df[column].to_numpy()
            if column not in INERT_REAL_COLUMNS:
                column_sources[column] = manifest["_resolved_column_sources"][column]
        supplied.update(feature_columns)

    try:
        validate_multiparty_dataframe(merged)
    except (TypeError, ValueError) as exc:
        raise AssemblyGateError(f"Assembled table failed multiparty validation: {exc}") from exc

    updated = json.loads(json.dumps(provenance))
    previous_sources = updated.get("feature_sources", [])
    if not isinstance(previous_sources, list):
        raise AssemblyGateError("base provenance feature_sources must be a JSON array")
    supplied_real = supplied - INERT_REAL_COLUMNS
    real_columns = set(updated["real_columns"]) | supplied_real
    remaining_placeholders = placeholders - supplied_real
    # DoA is permanently inert on this corpus even if a source repeats its
    # all-zero representation; it must never be promoted to a real feature.
    remaining_placeholders.update(placeholders & INERT_REAL_COLUMNS)
    missing_vad, degenerate_vad = _vad_trainability(merged, real_columns)
    all_feature_sources = [*previous_sources, *source_records]
    all_supplied_columns = sorted(
        {
            column
            for source_record in all_feature_sources
            for column in source_record["columns"]
        }
    )
    updated.update(
        {
            "artifact_kind": REAL_ASSEMBLED_ARTIFACT_KIND,
            "format_version": REAL_PROVENANCE_FORMAT_VERSION,
            "real_columns": sorted(real_columns),
            "placeholder_columns": sorted(remaining_placeholders),
            "trainable": not missing_vad and not degenerate_vad,
            "trainability": {
                "required_gold_vad_columns": _present_vad_columns(merged),
                "missing_gold_vad_columns": missing_vad,
                "degenerate_gold_vad_columns": degenerate_vad,
            },
            "inert_columns": sorted(INERT_REAL_COLUMNS),
            "feature_sources": all_feature_sources,
            "column_sources": column_sources,
            "assembly": {
                "base_csv": str(base_path.resolve()),
                "base_sha256": _sha256(base_path),
                "output_csv": str(output_path.resolve()),
                "supplied_columns": all_supplied_columns,
            },
        }
    )

    # Replay every historical and newly added manifest against the complete
    # candidate artifact before creating either output file.  Individual
    # feature validation is not sufficient for invariants that span chained
    # assemblies, such as duplicate inert columns or reuse of one upstream
    # participant column by another participant in a later manifest.
    try:
        validated_manifest_columns = _validate_assembled_feature_sources(
            updated, merged, output_path, output_sidecar
        )
    except DataSourceGateError as exc:
        raise AssemblyGateError(
            f"Assembled artifact failed pre-write manifest validation: {exc}"
        ) from exc
    if validated_manifest_columns != set(all_supplied_columns):
        raise AssemblyGateError(
            "Assembled artifact feature manifests do not uniquely cover all supplied columns"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp: Path | None = None
    sidecar_tmp: Path | None = None
    csv_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        csv_tmp = _reserve_staged_path(
            output_path.parent,
            prefix=f".{output_path.name}.tmp.",
            suffix=".csv",
        )
        csv_identity = _file_identity(csv_tmp)
        sidecar_tmp = _reserve_staged_path(
            output_sidecar.parent,
            prefix=f".{output_sidecar.name}.",
            suffix=".json.tmp",
        )
        sidecar_identity = _file_identity(sidecar_tmp)
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            merged.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = _read_csv(csv_tmp, error_type=AssemblyGateError)
        updated.update(_csv_binding(csv_tmp, persisted))
        updated["n_frames"] = len(persisted)
        with sidecar_tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False)
            )
            handle.flush()
            os.fsync(handle.fileno())
        staged_csv_sha256 = str(updated["csv_sha256"])
        staged_sidecar_sha256 = _sha256(sidecar_tmp)
        latest_base = _load_dataset_file_with_analysis_authority(
            base_path,
            data_kind="real",
            analysis_base_authority=analysis_base_authority,
        )
        if (
            latest_base.data_sha256 != base_file.data_sha256
            or latest_base.provenance_sha256 != base_file.provenance_sha256
        ):
            raise AssemblyGateError(
                "Base CSV or provenance changed while assembly was running"
            )
        try:
            _validate_assembled_feature_sources(
                updated, persisted, output_path, output_sidecar
            )
        except DataSourceGateError as exc:
            raise AssemblyGateError(
                f"Assembly sources changed before publication: {exc}"
            ) from exc
        if (
            _sha256(csv_tmp) != staged_csv_sha256
            or _sha256(sidecar_tmp) != staged_sidecar_sha256
        ):
            raise AssemblyGateError(
                "Assembly staging bytes changed before publication"
            )
        published.append((output_path, csv_identity))
        _publish_no_clobber(csv_tmp, output_path)
        published.append((output_sidecar, sidecar_identity))
        _publish_no_clobber(sidecar_tmp, output_sidecar)
        try:
            with _held_file_snapshot_guard(
                (
                    (output_path, csv_identity, staged_csv_sha256),
                    (output_sidecar, sidecar_identity, staged_sidecar_sha256),
                )
            ):
                published_file = _load_dataset_file_with_analysis_authority(
                    output_path,
                    data_kind="real",
                    analysis_base_authority=analysis_base_authority,
                )
                if (
                    published_file.data_sha256 != staged_csv_sha256
                    or published_file.provenance_sha256 != staged_sidecar_sha256
                ):
                    raise AssemblyGateError(
                        "Published assembly replay returned unexpected bytes"
                    )
        except (DataSourceGateError, _HeldFileSnapshotError, OSError) as exc:
            raise AssemblyGateError(
                f"Published assembly failed final source replay: {exc}"
            ) from exc
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for published_path, identity in reversed(published):
            try:
                cleanup = _remove_owned_path(published_path, identity)
                if not cleanup.removed:
                    recovery = (
                        f"; recovery={cleanup.recovery_path}"
                        if cleanup.recovery_path is not None
                        else ""
                    )
                    cleanup_errors.append(
                        f"{published_path}: {cleanup.detail or 'path preserved'}{recovery}"
                    )
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{published_path}: {cleanup_exc}")
        if cleanup_errors:
            raise AssemblyGateError(
                "Assembly publication failed and rollback was incomplete: "
                + "; ".join(cleanup_errors)
            ) from exc
        raise
    finally:
        if csv_tmp is not None and csv_identity is not None:
            _remove_owned_path(csv_tmp, csv_identity)
        if sidecar_tmp is not None and sidecar_identity is not None:
            _remove_owned_path(sidecar_tmp, sidecar_identity)
    return AssemblyResult(frame_df=merged, provenance=updated, output_csv=output_path)


def _source_csv_paths(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise DataSourceGateError(f"VAP data source file must be CSV: {path}")
        return [path]
    if not path.is_dir():
        raise DataSourceGateError(f"VAP data source not found: {path}")
    csv_paths = sorted(path.glob("*.csv"), key=lambda item: item.name.casefold())
    if not csv_paths:
        raise DataSourceGateError(f"VAP data source directory contains no *.csv files: {path}")
    return csv_paths


def _load_source_files(
    source: str | Path,
    *,
    data_kind: str,
    analysis_base_authority: object | None = None,
    excluded_session_ids: tuple[str, ...] = (),
) -> list[_DatasetFile]:
    _validate_data_kind(data_kind)
    loaded = [
        _load_dataset_file_with_analysis_authority(
            path,
            data_kind=data_kind,
            analysis_base_authority=analysis_base_authority,
        )
        for path in _source_csv_paths(source)
    ]
    owner_by_session: dict[str, Path] = {}
    for dataset_file in loaded:
        for session_id in dataset_file.frame_df["session_id"].drop_duplicates():
            previous = owner_by_session.get(session_id)
            if previous is not None:
                raise DataSourceGateError(
                    f"Session {session_id!r} appears in both {previous.name} and "
                    f"{dataset_file.path.name}; directory concatenation refuses duplicates"
                )
            owner_by_session[session_id] = dataset_file.path
    if not isinstance(excluded_session_ids, tuple) or any(
        not isinstance(session_id, str)
        or not session_id
        or session_id != session_id.strip()
        for session_id in excluded_session_ids
    ):
        raise DataSourceGateError(
            "excluded_session_ids must be a tuple of non-empty normalized strings"
        )
    if len(excluded_session_ids) != len(set(excluded_session_ids)):
        raise DataSourceGateError("excluded_session_ids cannot contain duplicates")
    excluded = set(excluded_session_ids)
    missing = sorted(excluded - set(owner_by_session))
    if missing:
        raise DataSourceGateError(
            f"Explicitly excluded sessions are absent from the source: {missing}"
        )
    selected: list[_DatasetFile] = []
    for dataset_file in loaded:
        file_sessions = set(
            dataset_file.frame_df["session_id"].drop_duplicates().astype(str)
        )
        excluded_in_file = file_sessions & excluded
        if excluded_in_file and excluded_in_file != file_sessions:
            raise DataSourceGateError(
                "A multi-session source file cannot be partially excluded: "
                f"{dataset_file.path} contains {sorted(file_sessions)}"
            )
        if not excluded_in_file:
            selected.append(dataset_file)
    if not selected:
        raise DataSourceGateError("Explicit exclusions removed every source session")
    return selected


def load_vap_dataframe(
    source: str | Path,
    *,
    data_kind: str,
    analysis_base_authority: object | None = None,
    excluded_session_ids: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load data after structural gates only.

    Target and selected-feature readiness are deliberately reported by
    :func:`audit_vap_source`; they are purpose-specific and must not be bundled
    with (or used to bypass) confirmed mapping and D3 provenance checks.
    """
    frame_df, _ = load_and_audit_vap_source(
        source,
        data_kind=data_kind,
        analysis_base_authority=analysis_base_authority,
        excluded_session_ids=excluded_session_ids,
    )
    return frame_df


def _concatenate_loaded(
    loaded: list[_DatasetFile], *, data_kind: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_file in loaded:
        frame = dataset_file.frame_df.copy()
        if data_kind == "real":
            assert dataset_file.provenance is not None
            frame["group_id"] = dataset_file.provenance["group_id"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _readiness_from_loaded(
    source: str | Path,
    loaded: list[_DatasetFile],
    *,
    data_kind: str,
    feature_set: str | None = None,
    group_delimiter: str | None = None,
) -> ReadinessReport:
    """Audit separate target, selected-input, and experiment readiness layers."""
    _validate_data_kind(data_kind)
    if data_kind == "real" and group_delimiter is not None:
        raise DataSourceGateError(
            "group_delimiter is synthetic-only; real group_id comes from gated provenance"
        )
    if group_delimiter is not None and (
        not isinstance(group_delimiter, str) or not group_delimiter
    ):
        raise DataSourceGateError("group_delimiter must be None or a non-empty string")
    frame_df = pd.concat([item.frame_df for item in loaded], ignore_index=True)
    session_ids = tuple(frame_df["session_id"].drop_duplicates().tolist())
    if data_kind == "real":
        group_ids = tuple(
            dict.fromkeys(
                str(dataset_file.provenance["group_id"])
                for dataset_file in loaded
                if dataset_file.provenance is not None
            )
        )
    else:
        group_ids = tuple(
            dict.fromkeys(
                session_id.split(group_delimiter, 1)[0]
                if group_delimiter is not None and group_delimiter in session_id
                else session_id
                for session_id in session_ids
            )
        )
    target_blockers = sorted(
        {issue for dataset_file in loaded for issue in _target_issues(dataset_file)}
    )
    placeholders = sorted(
        {
            column
            for dataset_file in loaded
            if dataset_file.provenance is not None
            for column in dataset_file.provenance.get("placeholder_columns", [])
        }
    )

    selected_feature_blockers: list[str] = []
    selected_feature_ready: bool | None = None
    if feature_set is not None:
        from hri_turn_taking.vap.feature_sets import active_feature_columns

        try:
            active_columns = active_feature_columns(feature_set)
        except ValueError as exc:
            raise DataSourceGateError(str(exc)) from exc
        selected_feature_ready = True
        if data_kind == "real":
            for dataset_file in loaded:
                assert dataset_file.provenance is not None
                present_slots = _present_slot_names(dataset_file.frame_df)
                selected = {
                    column
                    for column in active_columns
                    if not column.startswith("P")
                    or column.split("_", 1)[0] in present_slots
                }
                placeholder_set = set(dataset_file.provenance["placeholder_columns"])
                missing = sorted((selected & placeholder_set) - INERT_REAL_COLUMNS)
                if missing:
                    selected_feature_blockers.append(
                        f"{dataset_file.provenance['session_id']}: selected feature set "
                        f"{feature_set!r} has placeholder inputs {missing}"
                    )
            selected_feature_ready = not selected_feature_blockers

    experiment_blockers: list[str] = []
    if len(group_ids) < 3:
        experiment_blockers.append(
            "Formal counted train/inner-validation/outer-test evaluation requires at "
            f"least three group-disjoint recording groups; found {len(group_ids)}"
        )
    warnings: list[str] = []
    for dataset_file in loaded:
        provenance = dataset_file.provenance
        if provenance is None or provenance.get("data_kind") != "real":
            continue
        session_id = provenance["session_id"]
        summary = provenance.get("offset_quality")
        if summary is None:
            continue
        bound = float(summary["error_bound_ms"])
        if bound > 40.0:
            warnings.append(
                f"{session_id}: alignment error bound {bound:g} ms exceeds one 25 Hz "
                "frame (40 ms); retain this as a scientific limitation"
            )
        if summary["status"] == "degraded":
            warnings.append(
                f"{session_id}: stream alignment quality status is degraded"
            )

    return ReadinessReport(
        source=str(Path(source).resolve()),
        data_kind=data_kind,
        files=tuple(str(item.path.resolve()) for item in loaded),
        n_rows=len(frame_df),
        n_sessions=len(session_ids),
        n_groups=len(group_ids),
        session_ids=session_ids,
        group_ids=group_ids,
        source_artifacts=tuple(
            {
                "data_path": str(dataset_file.path.resolve()),
                "data_sha256": dataset_file.data_sha256,
                "provenance_path": str(_sidecar_path(dataset_file.path).resolve())
                if dataset_file.provenance is not None
                else None,
                "provenance_sha256": dataset_file.provenance_sha256
                if dataset_file.provenance is not None
                else None,
                "artifact_kind": dataset_file.provenance.get("artifact_kind")
                if dataset_file.provenance is not None
                else None,
                "format_version": dataset_file.provenance.get("format_version")
                if dataset_file.provenance is not None
                else None,
                "identity_authority": dataset_file.provenance.get("identity_authority")
                if dataset_file.provenance is not None
                else None,
                "identity_inventory_sha256": dataset_file.provenance
                .get("identity_contract", {})
                .get("inventory", {})
                .get("sha256")
                if dataset_file.provenance is not None
                else None,
            }
            for dataset_file in loaded
        ),
        structural_ready=True,
        target_ready=not target_blockers,
        target_blockers=tuple(target_blockers),
        feature_set=feature_set,
        selected_feature_ready=selected_feature_ready,
        selected_feature_blockers=tuple(selected_feature_blockers),
        experiment_blockers=tuple(experiment_blockers),
        placeholder_columns=tuple(placeholders),
        scientific_warnings=tuple(warnings),
    )


def load_and_audit_vap_source(
    source: str | Path,
    *,
    data_kind: str,
    feature_set: str | None = None,
    group_delimiter: str | None = None,
    analysis_base_authority: object | None = None,
    excluded_session_ids: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, ReadinessReport]:
    """Load and audit one immutable source snapshot.

    The CSVs and provenance sidecars are parsed once, then the returned frame and
    report (including all source hashes) are derived from that same in-memory set.
    Consumers that must couple readiness, inference, and artifact metadata should
    use this API instead of separate load/audit calls.
    """
    _validate_data_kind(data_kind)
    if data_kind == "real" and group_delimiter is not None:
        raise DataSourceGateError(
            "group_delimiter is synthetic-only; real group_id comes from gated provenance"
        )
    if group_delimiter is not None and (
        not isinstance(group_delimiter, str) or not group_delimiter
    ):
        raise DataSourceGateError("group_delimiter must be None or a non-empty string")
    loaded = _load_source_files(
        source,
        data_kind=data_kind,
        analysis_base_authority=analysis_base_authority,
        excluded_session_ids=excluded_session_ids,
    )
    frame_df = _concatenate_loaded(loaded, data_kind=data_kind)
    report = _readiness_from_loaded(
        source,
        loaded,
        data_kind=data_kind,
        feature_set=feature_set,
        group_delimiter=group_delimiter,
    )
    return frame_df, report


def audit_vap_source(
    source: str | Path,
    *,
    data_kind: str,
    feature_set: str | None = None,
    group_delimiter: str | None = None,
    analysis_base_authority: object | None = None,
    excluded_session_ids: tuple[str, ...] = (),
) -> ReadinessReport:
    """Audit target, selected-input, and experiment readiness from one snapshot."""
    _, report = load_and_audit_vap_source(
        source,
        data_kind=data_kind,
        feature_set=feature_set,
        group_delimiter=group_delimiter,
        analysis_base_authority=analysis_base_authority,
        excluded_session_ids=excluded_session_ids,
    )
    return report


__all__ = [
    "AssemblyGateError",
    "AssemblyResult",
    "DataSourceGateError",
    "ReadinessReport",
    "RealDataGateError",
    "REAL_ANALYSIS_BASE_ARTIFACT_KIND",
    "assemble_session",
    "audit_vap_source",
    "load_and_audit_vap_source",
    "load_vap_dataframe",
    "write_synthetic_provenance",
]
