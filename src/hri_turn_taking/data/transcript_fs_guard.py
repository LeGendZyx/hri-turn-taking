"""No-follow path and immutable-read guards for Transcript artifacts.

Paths are normalised lexically, never with ``Path.resolve()``, before every
component has been checked with ``lstat``.  On Windows, read handles share only
READ access, so WRITE and DELETE remain denied for the complete guarded
operation.  POSIX uses ``O_NOFOLLOW`` plus before/after identity and state
checks; this is an integrity gate on a trusted filesystem, not a cryptographic
signature against a privileged writer.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
CHUNK_BYTES = 1024 * 1024


class TranscriptPathGuardError(ValueError):
    """Raised when a path or held file changes or traverses a link."""


@dataclass(frozen=True)
class FileState:
    mode: int
    device: int
    inode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    change_time_token: int
    attributes: int


@dataclass(frozen=True)
class GuardedReadHandle:
    path: Path
    fd: int
    opening_state: FileState
    ancestors: tuple[tuple[Path, int, int, int, int], ...]
    allow_windows_read_metadata_churn: bool = False


def lexical_absolute(path: str | Path) -> Path:
    """Return an absolute, ``..``-collapsed path without following links."""

    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse(value: os.stat_result) -> bool:
    return bool(
        getattr(value, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT
    )


def _reject_link_or_reparse(path: Path, value: os.stat_result) -> None:
    if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        raise TranscriptPathGuardError(
            f"path component must not be a symlink/reparse point/junction: {path}"
        )


def _ancestor_states(path: Path) -> tuple[tuple[Path, int, int, int, int], ...]:
    states: list[tuple[Path, int, int, int, int]] = []
    for ancestor in reversed(path.parents):
        try:
            value = ancestor.lstat()
        except OSError as exc:
            raise TranscriptPathGuardError(
                f"cannot inspect path ancestor without following links: {ancestor}"
            ) from exc
        _reject_link_or_reparse(ancestor, value)
        if not stat.S_ISDIR(value.st_mode):
            raise TranscriptPathGuardError(f"path ancestor is not a directory: {ancestor}")
        states.append(
            (
                ancestor,
                value.st_dev,
                value.st_ino,
                0 if os.name == "nt" else value.st_mtime_ns,
                0 if os.name == "nt" else value.st_ctime_ns,
            )
        )
    return tuple(states)


def guard_directory(path: str | Path) -> tuple[Path, tuple[int, int]]:
    """Validate one existing directory and all lexical ancestors without follow."""

    absolute = lexical_absolute(path)
    ancestors = _ancestor_states(absolute)
    try:
        value = absolute.lstat()
    except OSError as exc:
        raise TranscriptPathGuardError(f"directory does not exist: {absolute}") from exc
    _reject_link_or_reparse(absolute, value)
    if not stat.S_ISDIR(value.st_mode):
        raise TranscriptPathGuardError(f"path is not a directory: {absolute}")
    if _ancestor_states(absolute) != ancestors:
        raise TranscriptPathGuardError(f"directory ancestor changed: {absolute}")
    return absolute, (value.st_dev, value.st_ino)


def guard_regular_file_path(path: str | Path) -> Path:
    """Validate an existing regular leaf and every ancestor without opening it."""

    absolute = lexical_absolute(path)
    ancestors = _ancestor_states(absolute)
    try:
        value = absolute.lstat()
    except OSError as exc:
        raise TranscriptPathGuardError(f"file does not exist: {absolute}") from exc
    _reject_link_or_reparse(absolute, value)
    if not stat.S_ISREG(value.st_mode):
        raise TranscriptPathGuardError(f"path is not a regular file: {absolute}")
    if _ancestor_states(absolute) != ancestors:
        raise TranscriptPathGuardError(f"file ancestor changed: {absolute}")
    return absolute


def guard_output_path(path: str | Path) -> Path:
    """Validate a not-yet-created leaf under a real, no-follow parent."""

    absolute = lexical_absolute(path)
    guard_directory(absolute.parent)
    try:
        value = absolute.lstat()
    except FileNotFoundError:
        return absolute
    except OSError as exc:
        raise TranscriptPathGuardError(f"cannot inspect output leaf: {absolute}") from exc
    _reject_link_or_reparse(absolute, value)
    return absolute


def directory_identity(path: str | Path) -> tuple[int, int]:
    """Return a guarded directory physical identity."""

    _absolute, identity = guard_directory(path)
    return identity


def _windows_extended_path(path: Path) -> str:
    value = str(path)
    if value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _open_read_fd(path: Path) -> int:
    if os.name != "nt":
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        return os.open(path, flags)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
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
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        _windows_extended_path(path),
        0x80000000,  # GENERIC_READ
        0x00000001,  # FILE_SHARE_READ only: deny WRITE and DELETE sharing.
        None,
        3,  # OPEN_EXISTING
        0x08000000 | 0x00200000,  # SEQUENTIAL_SCAN | OPEN_REPARSE_POINT
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), str(path))
    raw_handle = int(handle)
    try:
        return msvcrt.open_osfhandle(
            raw_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
    except BaseException as exc:
        if not close_handle(wintypes.HANDLE(raw_handle)):
            error = ctypes.get_last_error()
            exc.add_note(
                "CreateFileW cleanup also failed: "
                f"[{error}] {ctypes.FormatError(error)}"
            )
        raise


def _windows_change_time_and_attributes(fd: int) -> tuple[int, int]:
    if os.name != "nt":
        value = os.fstat(fd)
        return value.st_ctime_ns, getattr(value, "st_file_attributes", 0)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_info.restype = wintypes.BOOL
    info = FileBasicInfo()
    if not get_info(
        wintypes.HANDLE(msvcrt.get_osfhandle(fd)),
        0,  # FileBasicInfo
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error))
    return int(info.ChangeTime), int(info.FileAttributes)


def _file_state(fd: int) -> FileState:
    value = os.fstat(fd)
    change_time, attributes = _windows_change_time_and_attributes(fd)
    return FileState(
        mode=value.st_mode,
        device=value.st_dev,
        inode=value.st_ino,
        link_count=value.st_nlink,
        size_bytes=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        change_time_token=change_time,
        attributes=attributes,
    )


def _path_matches_state(path_value: os.stat_result, state: FileState) -> bool:
    return (
        path_value.st_dev == state.device
        and path_value.st_ino == state.inode
        and stat.S_IFMT(path_value.st_mode) == stat.S_IFMT(state.mode)
        and path_value.st_nlink == state.link_count
        and path_value.st_size == state.size_bytes
        and path_value.st_mtime_ns == state.mtime_ns
        and (os.name == "nt" or path_value.st_ctime_ns == state.ctime_ns)
    )


def _held_state_matches(handle: GuardedReadHandle, state: FileState) -> bool:
    if not (os.name == "nt" and handle.allow_windows_read_metadata_churn):
        return state == handle.opening_state
    opening = handle.opening_state
    # OneDrive can touch both CRT st_ctime and native ChangeTime when ordinary
    # reads hydrate/access an otherwise regular local file.  This opt-in is
    # restricted to a trusted single-writer workflow; the Windows handle still
    # denies WRITE and DELETE sharing for the complete read.  Every physical,
    # content-relevant and path-replay field remains pinned.
    return (
        state.mode == opening.mode
        and state.device == opening.device
        and state.inode == opening.inode
        and state.link_count == opening.link_count
        and state.size_bytes == opening.size_bytes
        and state.mtime_ns == opening.mtime_ns
        and state.attributes == opening.attributes
    )


def verify_guarded_read(handle: GuardedReadHandle) -> None:
    """Recheck held physical state, leaf path and every lexical ancestor."""

    state = _file_state(handle.fd)
    if not _held_state_matches(handle, state):
        raise TranscriptPathGuardError(f"held file changed: {handle.path}")
    if not stat.S_ISREG(state.mode) or state.attributes & FILE_ATTRIBUTE_REPARSE_POINT:
        raise TranscriptPathGuardError(f"held file is not regular/no-follow: {handle.path}")
    try:
        path_value = handle.path.lstat()
    except OSError as exc:
        raise TranscriptPathGuardError(f"held file path disappeared: {handle.path}") from exc
    _reject_link_or_reparse(handle.path, path_value)
    if not _path_matches_state(path_value, state):
        raise TranscriptPathGuardError(
            f"path no longer names the held physical file: {handle.path}"
        )
    if _ancestor_states(handle.path) != handle.ancestors:
        raise TranscriptPathGuardError(f"held file ancestor changed: {handle.path}")


@contextmanager
def guarded_read(
    path: str | Path, *, allow_windows_read_metadata_churn: bool = False
) -> Iterator[GuardedReadHandle]:
    """Hold one no-follow file read lock through the caller's complete operation.

    ``allow_windows_read_metadata_churn`` is only for an explicitly trusted
    single-writer workflow.  It tolerates read-induced ctime/ChangeTime churn;
    Windows WRITE/DELETE sharing remains denied and all other state is strict.
    """

    absolute = lexical_absolute(path)
    fd: int | None = None
    failure: BaseException | None = None
    try:
        ancestors = _ancestor_states(absolute)
        before = absolute.lstat()
        _reject_link_or_reparse(absolute, before)
        if not stat.S_ISREG(before.st_mode):
            raise TranscriptPathGuardError(f"path is not a regular file: {absolute}")
        fd = _open_read_fd(absolute)
        opening = _file_state(fd)
        if (
            not stat.S_ISREG(opening.mode)
            or opening.attributes & FILE_ATTRIBUTE_REPARSE_POINT
            or not _path_matches_state(before, opening)
        ):
            raise TranscriptPathGuardError(f"file changed while opening: {absolute}")
        handle = GuardedReadHandle(
            absolute,
            fd,
            opening,
            ancestors,
            allow_windows_read_metadata_churn=allow_windows_read_metadata_churn,
        )
        verify_guarded_read(handle)
        yield handle
        verify_guarded_read(handle)
    except OSError as exc:
        failure = TranscriptPathGuardError(f"cannot open guarded file: {absolute}")
        raise failure from exc
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as close_error:
                if failure is not None:
                    failure.add_note(f"guarded descriptor cleanup also failed: {close_error}")
                else:
                    raise TranscriptPathGuardError(
                        f"cannot close guarded file: {absolute}"
                    ) from close_error


def hash_guarded_read(handle: GuardedReadHandle) -> str:
    """SHA-256 the complete bytes from the same held descriptor."""

    verify_guarded_read(handle)
    os.lseek(handle.fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(handle.fd, CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
    verify_guarded_read(handle)
    return digest.hexdigest()


def read_guarded_bytes(handle: GuardedReadHandle) -> bytes:
    """Read exact bytes from the held descriptor after no-follow validation."""

    verify_guarded_read(handle)
    os.lseek(handle.fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(handle.fd, CHUNK_BYTES)
        if not block:
            break
        chunks.append(block)
    verify_guarded_read(handle)
    return b"".join(chunks)
