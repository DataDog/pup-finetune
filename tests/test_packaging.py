# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Packaging: the sdist ships an explicit allowlist, and the wheel ships the agent skill."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
SKILL_SOURCE = "skills/datadog-finetune/SKILL.md"
SKILL_IN_WHEEL = "pup_finetune/_skills/datadog-finetune/SKILL.md"


def _sdist_include() -> list[Any]:
    config = tomllib.loads(_PYPROJECT.read_text())
    sdist = config.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
    include = sdist.get("sdist", {}).get("include")
    assert isinstance(include, list), "the sdist target needs an explicit include allowlist"
    return include


def test_sdist_include_is_an_explicit_allowlist() -> None:
    include = _sdist_include()
    assert {"/src", "/tests", "/pyproject.toml", "/README.md"} <= set(include)
    for entry in include:
        assert isinstance(entry, str)
        # Anchored, literal paths only: a bare or wildcard pattern could match untracked files.
        assert entry.startswith("/"), entry
        assert not any(char in entry for char in "*?["), entry
        assert entry != "/", entry


def _hatch_wheel() -> dict[str, Any]:
    config = tomllib.loads(_PYPROJECT.read_text())
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert isinstance(wheel, dict)
    return wheel


def test_the_wheel_force_includes_the_skill_from_the_repository_root() -> None:
    force = _hatch_wheel()["force-include"]
    assert force == {SKILL_SOURCE: SKILL_IN_WHEEL}
    assert (_PYPROJECT.parent / SKILL_SOURCE).is_file()
    # The wheel is built from the sdist, so the sdist must carry the skill too.
    assert "/skills" in _sdist_include()


def _build(out_dir: Path) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv on PATH")
    # --offline: the build backend comes from uv's cache, filled when the project was synced.
    return subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [uv, "build", "--offline", "--out-dir", str(out_dir)],
        cwd=_PYPROJECT.parent,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_the_built_wheel_ships_the_skill_and_the_rubric_template(tmp_path: Path) -> None:
    completed = _build(tmp_path)
    if completed.returncode != 0 and "offline" in completed.stderr.lower():
        pytest.skip(f"the build backend is not in uv's cache: {completed.stderr.strip()}")
    assert completed.returncode == 0, completed.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        assert SKILL_IN_WHEEL in names
        assert "pup_finetune/curation/rubric_template.md" in names
        shipped = wheel.read(SKILL_IN_WHEEL)
    assert shipped == (_PYPROJECT.parent / SKILL_SOURCE).read_bytes()
