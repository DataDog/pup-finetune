# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""bin/pup-finetune: the POSIX sh dev launcher used for `pup extension install --local`."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "bin" / "pup-finetune"
HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "#\n"
    "# Unless explicitly stated otherwise all files in this repository are licensed under the "
    "Apache License Version 2.0.\n"
    "#\n"
    "# This product includes software developed at Datadog (https://www.datadoghq.com/).\n"
    "# Copyright 2026-present Datadog, Inc.\n"
)


@pytest.fixture
def fake_uv(tmp_path: Path) -> Path:
    """A directory holding a fake `uv` that prints its working directory and arguments."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\necho "cwd=$(pwd -P)"\nfor arg in "$@"; do echo "arg=$arg"; done\n')
    uv.chmod(0o755)
    return bin_dir


def run_with_fake_uv(launcher: Path, fake_uv: Path, cwd: Path, *args: str) -> list[str]:
    env = {"PATH": f"{fake_uv}{os.pathsep}/usr/bin{os.pathsep}/bin", "HOME": str(cwd)}
    completed = subprocess.run(  # noqa: S603 - the repo's launcher, list args
        [str(launcher), *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return completed.stdout.splitlines()


def expected_lines(cwd: Path, *args: str) -> list[str]:
    uv_args = ["run", "--quiet", "--project", str(REPO_ROOT.resolve()), "pup-finetune", *args]
    return [f"cwd={cwd.resolve()}", *(f"arg={arg}" for arg in uv_args)]


def test_launcher_is_an_executable_posix_script_with_the_header() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    text = LAUNCHER.read_text()
    assert text.startswith("#!/bin/sh\n" + HEADER)


def test_launcher_runs_uv_for_this_checkout_and_keeps_the_cwd(
    fake_uv: Path, tmp_path: Path
) -> None:
    work = tmp_path / "work dir"
    work.mkdir()
    lines = run_with_fake_uv(LAUNCHER, fake_uv, work, "doctor", "--arg with space")
    assert lines == expected_lines(work, "doctor", "--arg with space")


def test_launcher_follows_a_chain_of_relative_and_absolute_symlinks(
    fake_uv: Path, tmp_path: Path
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    absolute = tmp_path / "b" / "finetune"
    absolute.symlink_to(LAUNCHER)
    relative = tmp_path / "a" / "pup-finetune"
    relative.symlink_to(Path("..") / "b" / "finetune")
    lines = run_with_fake_uv(relative, fake_uv, tmp_path, "--version")
    assert lines == expected_lines(tmp_path, "--version")


def test_launcher_reports_missing_uv(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"PATH": f"{empty}{os.pathsep}/usr/bin{os.pathsep}/bin"}
    if shutil.which("uv", path=env["PATH"]):
        pytest.skip("uv is installed in a system directory")
    completed = subprocess.run(  # noqa: S603 - the repo's launcher, list args
        [str(LAUNCHER), "--version"], env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 127
    assert "uv not found" in completed.stderr


def test_launcher_stops_on_a_symlink_cycle(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)
    # A cyclic link cannot be executed, so run the script text with $0 set to the link.
    completed = subprocess.run(  # noqa: S603 - the repo's launcher text, list args
        ["/bin/sh", "-c", LAUNCHER.read_text(), str(first)],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 1
    assert "too many levels of symlinks" in completed.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_launcher_runs_version_through_uv(tmp_path: Path) -> None:
    completed = subprocess.run(  # noqa: S603 - the repo's launcher, list args
        [str(LAUNCHER), "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    # uv may rebuild the editable install (and its generated version) first, so compare the
    # shape rather than this process's already-imported version string.
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    assert re.fullmatch(r"\d+\.\d+[\w.+-]*", lines[0]), lines
