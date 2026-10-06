# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""scripts/check_headers.py flags tracked Python files and shell scripts without the header."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_headers.py"


def _load_script() -> ModuleType:
    # scripts/ is not a package (and is not shipped in the wheel), so load the file by path.
    spec = importlib.util.spec_from_file_location("check_headers", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_headers = _load_script()

# Variables git honors over cwd; if inherited (git hooks, `git rebase --exec`), the
# tmp-repo commands below would act on the developer's real repository instead.
_GIT_ENV_OVERRIDES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")


@pytest.fixture(autouse=True)
def _isolate_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _GIT_ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


_HEADER = "\n".join(check_headers.HEADER) + "\n"


def _write(root: Path, path: str, text: str) -> str:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)
    return path


def _git_repo(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S607
    for path, text in files.items():
        _write(root, path, text)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S607


def test_header_constant_matches_this_file() -> None:
    assert Path(__file__).read_text().startswith(_HEADER)


def test_good_header_passes(tmp_path: Path) -> None:
    path = _write(tmp_path, "src/pkg/mod.py", _HEADER + "x = 1\n")
    assert check_headers.find_violations(tmp_path, [path]) == []


def test_missing_header_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "src/pkg/mod.py", "x = 1\n")
    assert check_headers.find_violations(tmp_path, [path]) == [path]


def test_altered_header_is_reported(tmp_path: Path) -> None:
    altered = _HEADER.replace("2026-present", "2025-present")
    path = _write(tmp_path, "src/pkg/mod.py", altered)
    assert check_headers.find_violations(tmp_path, [path]) == [path]


def test_header_must_be_at_the_top(tmp_path: Path) -> None:
    path = _write(tmp_path, "src/pkg/mod.py", '"""Docstring."""\n' + _HEADER)
    assert check_headers.find_violations(tmp_path, [path]) == [path]


def test_shebang_then_header_passes(tmp_path: Path) -> None:
    script = _write(tmp_path, "scripts/run.sh", "#!/usr/bin/env bash\n" + _HEADER + "echo hi\n")
    launcher = _write(tmp_path, "bin/tool", "#!/bin/sh\n" + _HEADER)
    assert check_headers.find_violations(tmp_path, [script, launcher]) == []


def test_shell_scripts_and_bin_files_without_header_are_reported(tmp_path: Path) -> None:
    script = _write(tmp_path, "scripts/run.sh", "#!/usr/bin/env bash\necho hi\n")
    launcher = _write(tmp_path, "bin/tool", "#!/bin/sh\nexec true\n")
    assert check_headers.find_violations(tmp_path, [script, launcher]) == [launcher, script]


@pytest.mark.parametrize(
    "path",
    [
        "src/pup_finetune/_vendor/fw_cookbook/renderer.py",
        "tests/fixtures/llmobs/sample.py",
        "src/pup_finetune/_version.py",
    ],
)
def test_exempt_paths_are_skipped(tmp_path: Path, path: str) -> None:
    _write(tmp_path, path, "x = 1\n")
    assert check_headers.find_violations(tmp_path, [path]) == []


@pytest.mark.parametrize("path", ["pyproject.toml", ".pre-commit-config.yaml", "README.md"])
def test_non_code_files_are_skipped(tmp_path: Path, path: str) -> None:
    _write(tmp_path, path, "no header\n")
    assert check_headers.find_violations(tmp_path, [path]) == []


def test_tracked_but_deleted_file_is_skipped(tmp_path: Path) -> None:
    assert check_headers.find_violations(tmp_path, ["src/pkg/gone.py"]) == []


def test_tracked_files_lists_git_index(tmp_path: Path) -> None:
    _git_repo(tmp_path, {"a.py": "x = 1\n"})
    _write(tmp_path, "untracked.py", "x = 1\n")
    assert check_headers.tracked_files(tmp_path) == ["a.py"]


def test_main_returns_zero_when_all_headers_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git_repo(tmp_path, {"a.py": _HEADER, "README.md": "# readme\n"})
    assert check_headers.main(tmp_path) == 0
    assert capsys.readouterr().err == ""


def test_main_lists_offenders_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git_repo(tmp_path, {"good.py": _HEADER, "bad.py": "x = 1\n", "bin/tool": "#!/bin/sh\n"})
    assert check_headers.main(tmp_path) == 1
    err = capsys.readouterr().err
    assert "bad.py" in err
    assert "bin/tool" in err
    assert "good.py" not in err
