# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import re
import subprocess
import sys

from typer.testing import CliRunner

from pup_finetune import __version__
from pup_finetune.cli import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """Strip ANSI styling; Typer forces rich styling on CI (for example under GITHUB_ACTIONS)."""
    return _ANSI.sub("", text)


def test_version_prints_version_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_version_is_non_empty() -> None:
    assert __version__


def test_help_exits_zero_and_mentions_finetune() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "finetune" in plain(result.output)


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    output = plain(result.output)
    assert "Usage" in output
    assert "--version" in output


def test_python_dash_m_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pup_finetune", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == __version__
