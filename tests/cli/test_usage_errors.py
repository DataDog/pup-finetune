# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Click's usage errors print as our own errors do: two plain lines, never Typer's box."""

from __future__ import annotations

import io
import re

import pytest
from typer.testing import CliRunner

from pup_finetune.cli import app
from pup_finetune.cli._common import report_click_error, run_app

runner = CliRunner()


class _Terminal(io.StringIO):
    """A stderr that says it is a terminal."""

    def isatty(self) -> bool:
        return True


def run_on_a_terminal(args: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    terminal = _Terminal()
    monkeypatch.setattr("sys.stderr", terminal)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    with pytest.raises(SystemExit) as info:
        run_app(app, args)
    assert isinstance(info.value.code, int)
    return info.value.code, terminal.getvalue()


@pytest.mark.parametrize(
    ("args", "error", "hint"),
    [
        (
            ["train"],
            "train needs --dataset and --model",
            "pup-finetune help train",
        ),
        (
            ["dataset", "pull"],
            "dataset pull needs DIR",
            "pup-finetune help dataset pull",
        ),
        (["evalute"], "No such command 'evalute'. Did you mean 'evaluate'?", "pup-finetune help"),
        (
            ["train", "--conf"],
            "No such option: --conf (Possible options: --confirm)",
            "pup-finetune help train",
        ),
        (["dataset", "nope"], "No such command 'nope'.", "pup-finetune help dataset"),
        (["status", "a", "b"], "Got unexpected extra argument(s) (b)", "pup-finetune help status"),
    ],
    ids=["missing options", "missing argument", "a typo", "a prefix", "in a group", "extra"],
)
def test_usage_errors_are_two_plain_lines(args: list[str], error: str, hint: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == f"Error [usage]: {error}\nHint: see `{hint}`\n"


def test_on_a_terminal_too_there_is_no_box_and_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    code, err = run_on_a_terminal(["train", "--provider", "fake"], monkeypatch)
    assert code == 2
    assert err == (
        "Error [usage]: train needs --dataset and --model\nHint: see `pup-finetune help train`\n"
    )


def test_under_pup_the_hint_names_pup_finetune_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_OUTPUT", "json")
    code, err = run_on_a_terminal(["train", "--max-epochs", "x"], monkeypatch)
    assert code == 2
    assert err.splitlines() == [
        "Error [usage]: Invalid value for '--max-epochs': 'x' is not a valid int.",
        "Hint: see `pup finetune help train`",
    ]


@pytest.mark.parametrize("args", [[], ["dataset"], ["auth"]])
def test_a_bare_group_prints_its_help_on_stdout_and_exits_2(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "Usage: pup-finetune" in re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert result.stderr == ""


def test_another_click_failure_is_reported_as_unexpected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Failure(Exception):
        exit_code = 1

        def format_message(self) -> str:
            return "could not open the file"

    assert report_click_error(Failure()) == 1
    assert capsys.readouterr().err == "Error [unexpected]: could not open the file\n"


def test_suggestions_never_name_a_hidden_option() -> None:
    # --eval-max-cases-per-trajectory is the hidden old name of --eval-max-cases.
    result = runner.invoke(app, ["train", "--eval-max-cases-per", "3"])
    assert result.exit_code == 2
    first = result.stderr.splitlines()[0]
    assert "--eval-max-cases," in first
    assert "--eval-max-cases-per-trajectory" not in first
    assert "--expect" not in first
