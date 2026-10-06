# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Safe on-disk state: atomic writes, private files and directories, locks, frozen settings.

Files we create are ``0600`` and directories ``0700``: datasets and run state hold trace
content that should not be readable by other local users.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Self

from pup_finetune.errors import StateError

LOCK_FILE_NAME = ".pup-finetune.lock"

_PRIVATE_FILE = 0o600
_PRIVATE_DIR = 0o700


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` and any missing parents as ``0700`` directories.

    Directories that already exist keep their permissions; only the ones created here are set
    to ``0700``, independent of the umask.
    """
    path = Path(path)
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise StateError(f"{current} exists and is not a directory")
    for directory in reversed(missing):
        with contextlib.suppress(FileExistsError):
            directory.mkdir(mode=_PRIVATE_DIR)
            directory.chmod(_PRIVATE_DIR)
    if not path.is_dir():
        raise StateError(f"{path} exists and is not a directory")
    return path


def make_private_dir(path: Path) -> Path:
    """Like :func:`ensure_private_dir`, but also set ``path`` itself to ``0700`` if it existed.

    Use it for a directory we take ownership of, such as an empty directory a dataset is
    created in. Existing parents keep their permissions.
    """
    path = ensure_private_dir(path)
    path.chmod(_PRIVATE_DIR)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` atomically; the file ends up ``0600``.

    Writes a temporary file in the same directory, fsyncs it, renames it over ``path`` and
    fsyncs the directory, so readers see either the old or the new content, never a mix.
    The parent directory must already exist.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.chmod(_PRIVATE_FILE)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, data: Any) -> None:
    """Replace ``path`` with ``data`` as indented, key-sorted JSON (see :func:`atomic_write_text`)."""
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    atomic_write_text(path, text + "\n")


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append ``rows`` as JSON lines, flushed and fsynced; returns the number of rows written.

    The file is created ``0600`` when missing. If an earlier append was torn (the file does
    not end with a newline), the partial line is terminated first so the new rows stay intact;
    :func:`read_jsonl` then reports the partial line with its line number.
    """
    path = Path(path)
    lines = [canonical_json(row) + "\n" for row in rows]
    created = not path.exists()
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE)
    with os.fdopen(fd, "r+b") as handle:
        size = os.fstat(fd).st_size
        prefix = b"" if size == 0 or os.pread(fd, 1, size - 1) == b"\n" else b"\n"
        handle.write(prefix + "".join(lines).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _fsync_dir(path.parent)
    return len(lines)


class _NonFiniteNumberError(ValueError):
    """A ``NaN``, ``Infinity`` or out-of-range number, which Python's ``json`` reads.

    JSON has no such values and :func:`canonical_json` cannot write them back, so a state
    file holding one is corrupt.
    """


def _reject_constant(constant: str) -> NoReturn:
    raise _NonFiniteNumberError(f"{constant} is not a JSON value")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _NonFiniteNumberError(f"{text} is out of range for a number")
    return value


def _loads(text: str) -> Any:
    """``json.loads``, but ``NaN``, ``Infinity`` and overflowing numbers raise ``ValueError``."""
    return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)


def read_json(path: Path) -> Any:
    """Parse a JSON file; invalid JSON raises :class:`StateError`.

    ``NaN``, ``Infinity`` and numbers beyond float range are invalid. A missing file raises
    ``FileNotFoundError`` so callers can tell absent from corrupt.
    """
    path = Path(path)
    try:
        return _loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise StateError(f"{path}: not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise StateError(f"{path}:{exc.lineno}: invalid JSON: {exc.msg}") from exc
    except _NonFiniteNumberError as exc:
        raise StateError(f"{path}: invalid JSON: {exc}") from exc


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object in a JSON Lines file, skipping blank lines.

    A line that is not a JSON object raises :class:`StateError` naming its line number.
    """
    for _, row in read_jsonl_numbered(path):
        yield row


def read_jsonl_numbered(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Like :func:`read_jsonl`, but yield ``(line_number, row)`` with 1-based line numbers."""
    path = Path(path)
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row, problem = parse_jsonl_line(line)
            if problem is not None:
                raise StateError(f"{path}:{line_number}: {problem}")
            yield line_number, row


def parse_jsonl_line(line: bytes) -> tuple[dict[str, Any], str | None]:
    """Parse one JSONL line as a JSON object, without raising.

    Returns ``(object, None)``, or ``({}, problem)`` when the line is not valid UTF-8, not
    valid JSON (including numbers too long or nesting too deep to decode), or not an object.
    """
    try:
        data = _loads(line.decode("utf-8"))
    except UnicodeDecodeError:
        return {}, "not valid UTF-8"
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc.msg}"
    except (ValueError, RecursionError) as exc:
        return {}, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return {}, "expected a JSON object"
    return data, None


class DirectoryLock:
    """Exclusive, non-blocking lock on a directory, held through ``<dir>/.pup-finetune.lock``.

    Use as a context manager. The lock is released when the context exits or the process
    dies. It is not reentrant: a second lock on the same directory fails, even in-process.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def lock_path(self) -> Path:
        return self.path / LOCK_FILE_NAME

    def __enter__(self) -> Self:
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, _PRIVATE_FILE)
        except OSError as exc:
            raise StateError(f"cannot lock {self.path}: {exc.strerror or exc}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise StateError(
                f"another pup-finetune process is using {self.path}",
                hint="wait for it to finish, or use a different directory",
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically: sorted keys, no whitespace."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def settings_hash(settings: Mapping[str, Any]) -> str:
    """The sha256 hex digest of ``settings`` as canonical JSON."""
    return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()


class FrozenSettings:
    """Settings recorded the first time a directory is used, and enforced afterwards."""

    @staticmethod
    def check_or_write(directory: Path, name: str, settings: Mapping[str, Any]) -> str:
        """Write ``<directory>/<name>.json`` once; later calls must pass identical settings.

        Returns the settings hash. Raises :class:`StateError` when the stored settings differ
        or are corrupt. Callers should hold a :class:`DirectoryLock` on ``directory``.
        """
        if not name or Path(name).name != name or name.startswith("."):
            raise ValueError(f"invalid settings name: {name!r}")
        path = Path(directory) / f"{name}.json"
        expected = canonical_json(settings)
        if not path.exists():
            atomic_write_json(path, dict(settings))
            return settings_hash(settings)
        stored = read_json(path)
        if not isinstance(stored, dict):
            raise StateError(f"{path}: expected a JSON object")
        if canonical_json(stored) != expected:
            changed = sorted(
                key
                for key in set(stored) | set(settings)
                if canonical_json(stored.get(key)) != canonical_json(settings.get(key))
            )
            raise StateError(
                f"settings changed; use a new directory ({path} records different settings)",
                hint=f"changed: {', '.join(changed)}" if changed else None,
            )
        return settings_hash(settings)
