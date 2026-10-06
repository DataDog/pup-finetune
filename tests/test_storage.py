# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pup_finetune.errors import StateError
from pup_finetune.storage import (
    LOCK_FILE_NAME,
    DirectoryLock,
    FrozenSettings,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    ensure_private_dir,
    read_json,
    read_jsonl,
    read_jsonl_numbered,
    settings_hash,
)


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def loose_umask() -> Iterator[None]:
    # A permissive umask proves the 0600/0700 modes do not depend on the caller's umask.
    previous = os.umask(0o002)
    yield
    os.umask(previous)


def test_atomic_write_text_replaces_content_privately(tmp_path: Path, loose_umask: object) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old")
    target.chmod(0o644)
    atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert mode(target) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_atomic_write_json_round_trips(tmp_path: Path, loose_umask: object) -> None:
    target = tmp_path / "run.json"
    atomic_write_json(target, {"b": [1, 2], "a": "é"})
    assert read_json(target) == {"a": "é", "b": [1, 2]}
    assert target.read_text().endswith("\n")
    assert mode(target) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_invalid_json_data_is_rejected_before_touching_disk(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    atomic_write_json(target, {"keep": True})
    with pytest.raises(ValueError):
        atomic_write_json(target, {"bad": float("nan")})
    with pytest.raises(ValueError):
        append_jsonl(tmp_path / "rows.jsonl", [{"bad": float("nan")}])
    assert read_json(target) == {"keep": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_atomic_write_failure_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    atomic_write_json(target, {"keep": True})
    with pytest.raises(TypeError):
        atomic_write_text(target, b"bytes")  # type: ignore[arg-type]
    assert read_json(target) == {"keep": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_append_jsonl_creates_private_file_and_appends(tmp_path: Path, loose_umask: object) -> None:
    target = tmp_path / "labels.jsonl"
    assert append_jsonl(target, [{"id": "a", "keep": True}]) == 1
    assert mode(target) == 0o600
    assert append_jsonl(target, iter([{"id": "b"}, {"id": "c"}])) == 2
    assert append_jsonl(target, []) == 0
    assert list(read_jsonl(target)) == [{"id": "a", "keep": True}, {"id": "b"}, {"id": "c"}]
    assert len(target.read_text().splitlines()) == 3


def test_append_jsonl_terminates_a_torn_line(tmp_path: Path) -> None:
    target = tmp_path / "labels.jsonl"
    target.write_text('{"id": "a"}\n{"id": "tor')
    append_jsonl(target, [{"id": "b"}])
    assert target.read_text() == '{"id": "a"}\n{"id": "tor\n{"id":"b"}\n'
    with pytest.raises(StateError, match=r"labels\.jsonl:2: invalid JSON"):
        list(read_jsonl(target))


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert list(read_jsonl(target)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_numbered_counts_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert list(read_jsonl_numbered(target)) == [(1, {"a": 1}), (3, {"a": 2})]


def test_read_jsonl_error_names_the_line(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n{"a": 2}\n{not json\n')
    rows = read_jsonl(target)
    assert next(rows) == {"a": 1}
    with pytest.raises(StateError, match=r"rows\.jsonl:3: invalid JSON"):
        list(rows)


def test_read_jsonl_invalid_utf8_names_the_line(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_bytes(b'{"a":1}\n{"b":"\xff"}\n')
    rows = read_jsonl(target)
    assert next(rows) == {"a": 1}
    with pytest.raises(StateError, match=r"rows\.jsonl:2: not valid UTF-8"):
        list(rows)


def test_read_json_invalid_utf8_raises_state_error(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    target.write_bytes(b'{"b": "\xff"}\n')
    with pytest.raises(StateError, match=r"run\.json: not valid UTF-8"):
        read_json(target)


def test_read_jsonl_rejects_non_objects(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text("[1, 2]\n")
    with pytest.raises(StateError, match=r"rows\.jsonl:1: expected a JSON object"):
        list(read_jsonl(target))


def test_read_json_corrupt_raises_state_error(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    target.write_text('{\n  "a": \n')
    with pytest.raises(StateError, match=r"run\.json:\d+: invalid JSON"):
        read_json(target)


NON_FINITE_NUMBERS = ["NaN", "Infinity", "-Infinity", "1e999"]


@pytest.mark.parametrize("number", NON_FINITE_NUMBERS)
def test_read_json_rejects_non_finite_numbers(tmp_path: Path, number: str) -> None:
    target = tmp_path / "run.json"
    target.write_text(f'{{"loss": {number}}}\n')
    with pytest.raises(StateError, match=rf"run\.json: invalid JSON: {re.escape(number)} "):
        read_json(target)


@pytest.mark.parametrize("number", NON_FINITE_NUMBERS)
def test_read_jsonl_rejects_non_finite_numbers_naming_the_line(tmp_path: Path, number: str) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text(f'{{"a": 1}}\n{{"loss": [{number}]}}\n')
    rows = read_jsonl(target)
    assert next(rows) == {"a": 1}
    with pytest.raises(StateError, match=rf"rows\.jsonl:2: invalid JSON: {re.escape(number)} "):
        list(rows)


def test_read_json_missing_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")


def test_ensure_private_dir_creates_parents_0700(tmp_path: Path, loose_umask: object) -> None:
    leaf = tmp_path / "a" / "b" / "c"
    assert ensure_private_dir(leaf) == leaf
    for directory in (tmp_path / "a", tmp_path / "a" / "b", leaf):
        assert directory.is_dir()
        assert mode(directory) == 0o700


def test_ensure_private_dir_leaves_existing_dirs_alone(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    existing.chmod(0o755)
    ensure_private_dir(existing)
    assert mode(existing) == 0o755
    ensure_private_dir(existing / "new")
    assert mode(existing) == 0o755
    assert mode(existing / "new") == 0o700


def test_ensure_private_dir_rejects_a_file(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("")
    with pytest.raises(StateError, match="not a directory"):
        ensure_private_dir(target)
    with pytest.raises(StateError, match="not a directory"):
        ensure_private_dir(target / "child")


def test_directory_lock_is_exclusive(tmp_path: Path) -> None:
    with DirectoryLock(tmp_path):
        assert mode(tmp_path / LOCK_FILE_NAME) == 0o600
        with (
            pytest.raises(StateError, match="another pup-finetune process is using"),
            DirectoryLock(tmp_path),
        ):
            pass
    with DirectoryLock(tmp_path):
        pass


def test_directory_lock_on_missing_dir_raises_state_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(StateError, match=r"cannot lock .*missing"), DirectoryLock(missing):
        pass


def test_directory_lock_releases_on_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), DirectoryLock(tmp_path):
        raise RuntimeError("boom")
    with DirectoryLock(tmp_path):
        pass


_TRY_LOCK = """
import sys
from pathlib import Path
from pup_finetune.errors import StateError
from pup_finetune.storage import DirectoryLock
try:
    with DirectoryLock(Path(sys.argv[1])):
        pass
except StateError:
    sys.exit(3)
"""


def _try_lock_in_subprocess(directory: Path) -> int:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", _TRY_LOCK, str(directory)], check=False, capture_output=True
    )
    return completed.returncode


def test_directory_lock_is_held_across_processes(tmp_path: Path) -> None:
    with DirectoryLock(tmp_path):
        assert _try_lock_in_subprocess(tmp_path) == 3
    assert _try_lock_in_subprocess(tmp_path) == 0


def test_settings_hash_is_canonical() -> None:
    a = settings_hash({"b": 1, "a": [1, {"y": 2, "x": 1}]})
    b = settings_hash({"a": [1, {"x": 1, "y": 2}], "b": 1})
    assert a == b
    assert len(a) == 64
    assert settings_hash({"a": 1}) != settings_hash({"a": 2})


def test_frozen_settings_written_once_then_enforced(tmp_path: Path) -> None:
    settings = {"ml_app": "demo", "seed": 42}
    digest = FrozenSettings.check_or_write(tmp_path, "settings", settings)
    path = tmp_path / "settings.json"
    assert json.loads(path.read_text()) == settings
    assert mode(path) == 0o600
    assert digest == settings_hash(settings)
    assert FrozenSettings.check_or_write(tmp_path, "settings", {"seed": 42, "ml_app": "demo"}) == (
        digest
    )


def test_frozen_settings_mismatch_raises(tmp_path: Path) -> None:
    FrozenSettings.check_or_write(tmp_path, "settings", {"ml_app": "demo", "seed": 42})
    with pytest.raises(StateError, match="settings changed; use a new directory") as info:
        FrozenSettings.check_or_write(tmp_path, "settings", {"ml_app": "demo", "seed": 7})
    assert info.value.hint == "changed: seed"
    assert json.loads((tmp_path / "settings.json").read_text())["seed"] == 42


@pytest.mark.parametrize("stored", ["[1, 2]\n", "{not json\n", '{"seed": NaN}\n'])
def test_frozen_settings_corrupt_file_raises(tmp_path: Path, stored: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(stored)
    with pytest.raises(StateError):
        FrozenSettings.check_or_write(tmp_path, "settings", {"seed": 42})
    assert path.read_text() == stored


@pytest.mark.parametrize("name", ["", "a/b", "..", ".hidden"])
def test_frozen_settings_rejects_bad_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        FrozenSettings.check_or_write(tmp_path, name, {})
