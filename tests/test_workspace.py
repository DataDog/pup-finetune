# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""How a bare run id names a run under ``./runs``, and when ``./runs`` or ``./datasets`` gets
a ``.gitignore``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pup_finetune import workspace
from pup_finetune.workspace import ParentSetup, default_run_dir, find_run_dir, ignore_new_parent


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _run(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "run.json").write_text("{}")
    return path


def test_the_default_run_directory_is_under_runs() -> None:
    assert default_run_dir("ft-1") == Path("runs/ft-1")


def test_an_existing_path_is_used_as_given() -> None:
    _run(Path("runs/abc"))
    Path("abc").mkdir()
    assert find_run_dir(Path("abc")) == (Path("abc"), False)


def test_a_bare_id_names_a_run_under_runs() -> None:
    _run(Path("runs/abc"))
    assert find_run_dir(Path("abc")) == (Path("runs/abc"), True)


@pytest.mark.parametrize("given", ["sub/abc", "ABC", "-abc", "missing"])
def test_anything_else_is_left_for_the_command_to_report(given: str) -> None:
    _run(Path("runs/abc"))
    _run(Path("runs/sub/abc"))
    assert find_run_dir(Path(given)) == (Path(given), False)


def test_a_directory_without_run_json_under_runs_is_not_a_run() -> None:
    Path("runs/abc").mkdir(parents=True)
    assert find_run_dir(Path("abc")) == (Path("abc"), False)


@pytest.mark.parametrize("parent", ["runs", "datasets"])
def test_creating_runs_or_datasets_adds_a_gitignore(parent: str) -> None:
    setup = ignore_new_parent(Path(parent) / "x")
    assert setup.notes == [f"created {parent}/.gitignore so trace-derived files stay out of git"]
    assert setup.warnings == []
    assert (Path(parent) / ".gitignore").read_text() == "*\n"
    assert Path(parent).stat().st_mode & 0o777 == 0o700
    # Nothing is left of the temporary directory it was made in.
    assert sorted(entry.name for entry in Path().iterdir()) == [parent]
    assert ignore_new_parent(Path(parent) / "y") == ParentSetup([], [])


@pytest.mark.parametrize("path", ["elsewhere/x", "runs/a/b", "runs"])
def test_other_directories_get_nothing(path: str) -> None:
    assert ignore_new_parent(Path(path)) == ParentSetup([], [])
    assert not list(Path().rglob(".gitignore"))


def test_a_runs_directory_made_meanwhile_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    real_rename = Path.rename

    def raced(self: Path, target: Path) -> Path:
        Path("runs").mkdir()  # another process made it first
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", raced)
    setup = ignore_new_parent(Path("runs/x"))
    assert setup.warnings == []
    assert sorted(entry.name for entry in Path().iterdir()) == ["runs"]


def _git_repo() -> None:
    Path(".git").mkdir()


def test_an_existing_unignored_runs_in_a_git_work_tree_is_warned_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git_repo()
    Path("runs").mkdir()
    monkeypatch.setattr(workspace, "_ignored_by_git", lambda directory, top: False)
    setup = ignore_new_parent(Path("runs/x"))
    assert setup.notes == []
    assert len(setup.warnings) == 1
    assert "runs/ is in a git work tree" in setup.warnings[0]
    assert "/runs/" in setup.warnings[0]


def test_no_warning_when_git_ignores_it_or_it_has_a_gitignore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git_repo()
    Path("runs").mkdir()
    monkeypatch.setattr(workspace, "_ignored_by_git", lambda directory, top: True)
    assert ignore_new_parent(Path("runs/x")) == ParentSetup([], [])
    monkeypatch.setattr(workspace, "_ignored_by_git", lambda directory, top: False)
    Path("runs/.gitignore").write_text("*\n")
    assert ignore_new_parent(Path("runs/x")) == ParentSetup([], [])


def test_no_warning_outside_a_git_work_tree() -> None:
    Path("runs").mkdir()
    assert ignore_new_parent(Path("runs/x")) == ParentSetup([], [])
