# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune skill show` and `skill install`, end to end through the Typer app."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import skill
from pup_finetune.cli import app
from pup_finetune.cli import skill as skill_cli

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeNpx:
    """Stands in for running `npx skills add`: records each argv and environment.

    It returns ``exit_code``, or raises ``error`` when one is set.
    """

    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.error: OSError | None = None
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> int:
        self.calls.append(list(argv))
        self.envs.append(dict(env))
        if self.error is not None:
            raise self.error
        return self.exit_code


@pytest.fixture
def no_npx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skill_cli, "find_npx", lambda: None)


@pytest.fixture
def npx(monkeypatch: pytest.MonkeyPatch) -> FakeNpx:
    fake = FakeNpx()
    monkeypatch.setattr(skill_cli, "find_npx", lambda: "/usr/local/bin/npx")
    monkeypatch.setattr(skill_cli, "run_npx", fake)
    return fake


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


def invoke(*args: str) -> tuple[Result, dict[str, Any]]:
    result = runner.invoke(app, ["skill", *args])
    return result, json.loads(result.stdout) if result.stdout.strip() else {}


def test_show_prints_the_path_and_content() -> None:
    result, data = invoke("show")
    assert result.exit_code == 0, result.output
    assert data["name"] == "datadog-finetune"
    assert Path(data["path"]) == skill.skill_path()
    assert data["content"] == skill.skill_text()
    assert data["content"].startswith("---\nname: datadog-finetune\n")


def test_install_without_npx_copies_into_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_npx: None
) -> None:
    monkeypatch.chdir(tmp_path)
    result, data = invoke("install")
    assert result.exit_code == 0, result.output
    assert data["method"] == "copy"
    assert Path(data["base"]) == Path.cwd()
    paths = [Path(entry["path"]) for entry in data["installed"]]
    assert paths == [
        Path.cwd() / ".agents/skills/datadog-finetune/SKILL.md",
        Path.cwd() / ".claude/skills/datadog-finetune/SKILL.md",
    ]
    for path in paths:
        assert path.read_text(encoding="utf-8") == skill.skill_text()
    assert [entry["state"] for entry in data["installed"]] == ["created", "created"]


def test_install_with_dir_copies_there_even_when_npx_exists(tmp_path: Path, npx: FakeNpx) -> None:
    result, data = invoke("install", "--dir", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert npx.calls == []
    assert data["method"] == "copy"
    assert (tmp_path / ".agents/skills/datadog-finetune/SKILL.md").is_file()
    assert (tmp_path / ".claude/skills/datadog-finetune/SKILL.md").is_file()


def test_install_global_without_npx_uses_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path, no_npx: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    result, data = invoke("install", "-g")
    assert result.exit_code == 0, result.output
    assert Path(data["base"]) == home
    assert (home / ".agents/skills/datadog-finetune/SKILL.md").is_file()
    assert (home / ".claude/skills/datadog-finetune/SKILL.md").is_file()
    assert not (project / ".agents").exists()


def test_install_refuses_to_overwrite_a_different_copy_without_yes(
    tmp_path: Path, no_npx: None
) -> None:
    target = tmp_path / ".claude/skills/datadog-finetune/SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("an older skill\n")
    result, _ = invoke("install", "--dir", str(tmp_path))
    assert result.exit_code == 2
    assert "different copy" in result.stderr
    assert target.read_text() == "an older skill\n"
    result, data = invoke("install", "--dir", str(tmp_path), "-y")
    assert result.exit_code == 0, result.output
    assert [entry["state"] for entry in data["installed"]] == ["created", "updated"]
    assert target.read_text(encoding="utf-8") == skill.skill_text()


def test_install_with_npx_runs_skills_add_on_the_bundled_directory(npx: FakeNpx) -> None:
    result, data = invoke("install", "-a", "claude-code", "-a", "codex", "-g", "-y")
    assert result.exit_code == 0, result.output
    source = str(skill.bundled_skills_dir())
    assert npx.calls == [
        [
            "/usr/local/bin/npx",
            "--yes",
            "skills",
            "add",
            source,
            "--skill",
            "datadog-finetune",
            "--global",
            "--yes",
            "--agent",
            "claude-code",
            "--agent",
            "codex",
        ]
    ]
    assert data["method"] == "npx"
    assert data["source"] == source
    assert data["agents"] == ["claude-code", "codex"]
    assert data["global"] is True
    assert data["installed"] == []
    assert "skills CLI" in data["note"]


def test_npx_runs_without_any_secret_in_its_environment(
    monkeypatch: pytest.MonkeyPatch, npx: FakeNpx
) -> None:
    secrets = {
        "DD_ACCESS_TOKEN": "synthetic-access-token",
        "DD_API_KEY": "synthetic-api-key",
        "DD_APP_KEY": "synthetic-app-key",
        "DD_APPLICATION_KEY": "synthetic-application-key",
        "FIREWORKS_API_KEY": "synthetic-fireworks-key",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PUP_FINETUNE_SKILL_TEST", "kept")
    result, _ = invoke("install")
    assert result.exit_code == 0, result.output
    [env] = npx.envs
    for name in secrets:
        assert name not in env
    assert env["PUP_FINETUNE_SKILL_TEST"] == "kept"
    assert env["PATH"] == os.environ["PATH"]


def test_npx_that_cannot_start_exits_1_with_the_dir_hint(npx: FakeNpx) -> None:
    npx.error = FileNotFoundError(2, "No such file or directory", "npx")
    result, data = invoke("install")
    assert result.exit_code == 1
    assert data == {}
    assert "could not run npx" in result.stderr
    assert "--dir ." in result.stderr


def test_run_npx_sends_the_child_stdout_to_stderr(capfd: pytest.CaptureFixture[str]) -> None:
    code = skill_cli.run_npx(
        [sys.executable, "-c", "import os; print('child', os.environ.get('SKILL_ENV'))"],
        {"SKILL_ENV": "given", "PATH": os.environ.get("PATH", "")},
    )
    assert code == 0
    out, err = capfd.readouterr()
    assert out == ""
    assert "child given" in err


def test_agent_with_the_copy_install_warns_that_it_is_unused(tmp_path: Path, no_npx: None) -> None:
    result, data = invoke("install", "--dir", str(tmp_path), "-a", "claude-code")
    assert result.exit_code == 0, result.output
    assert data["method"] == "copy"
    assert "--agent is used only by the skills CLI" in result.stderr
    assert len(data["installed"]) == 2


def test_install_with_npx_passes_only_the_options_given(npx: FakeNpx) -> None:
    result, _ = invoke("install")
    assert result.exit_code == 0, result.output
    assert npx.calls[0][1:] == [
        "--yes",
        "skills",
        "add",
        str(skill.bundled_skills_dir()),
        "--skill",
        "datadog-finetune",
    ]


def test_a_failed_npx_install_exits_1_with_the_fallback_hint(
    monkeypatch: pytest.MonkeyPatch, npx: FakeNpx
) -> None:
    npx.exit_code = 3
    result, data = invoke("install")
    assert result.exit_code == 1
    assert data == {}
    assert "exited 3" in result.stderr
    assert "--dir ." in result.stderr


def test_global_and_dir_together_are_a_usage_error(tmp_path: Path, no_npx: None) -> None:
    result, _ = invoke("install", "-g", "--dir", str(tmp_path))
    assert result.exit_code == 2
    assert not (tmp_path / ".agents").exists()


def test_install_is_allowed_in_read_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_npx: None
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    result, data = invoke("install", "--dir", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert data["method"] == "copy"


@pytest.mark.live
@pytest.mark.skipif(shutil.which("npx") is None, reason="needs npx on PATH")
def test_the_skills_cli_discovers_the_skill() -> None:
    """`npx skills add ./skills --list` finds `datadog-finetune` (downloads the skills CLI)."""
    npx = shutil.which("npx")
    assert npx is not None
    completed = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [npx, "--yes", "skills", "add", "./skills", "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "datadog-finetune" in completed.stdout + completed.stderr
