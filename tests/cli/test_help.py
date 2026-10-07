# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`help`, the help path pup does not intercept, and usage errors that point to it."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Annotated, Any

import pytest
import typer
import typer.main
from typer import rich_utils
from typer.testing import CliRunner

from pup_finetune.cli import app
from pup_finetune.cli._common import prog_name, run_app
from pup_finetune.cli.help import HELP_WIDTH
from pup_finetune.runtime import Runtime

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """Strip ANSI styling; Typer forces rich styling on CI (for example under GITHUB_ACTIONS)."""
    return _ANSI.sub("", text)


def command_paths(command: Any, ctx: Any, path: tuple[str, ...] = ()) -> Iterator[list[str]]:
    """The words naming every visible command and group, the program itself first."""
    yield list(path)
    if hasattr(command, "list_commands"):
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None and not sub.hidden:
                yield from command_paths(sub, ctx, (*path, name))


def all_paths() -> list[list[str]]:
    root = typer.main.get_command(app)
    return list(command_paths(root, typer.Context(root, info_name="pup-finetune")))


@pytest.fixture
def width(monkeypatch: pytest.MonkeyPatch) -> int:
    """A fixed rendering width above HELP_WIDTH, so `help` and `--help` render alike."""
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", 150)
    return 150


def test_every_command_is_covered() -> None:
    names = {" ".join(path) for path in all_paths()}
    assert {"", "help", "train", "dataset", "dataset pull", "auth login"} <= names
    assert "_worker" not in names


@pytest.mark.parametrize("path", all_paths(), ids=lambda path: " ".join(path) or "<root>")
def test_help_prints_what_dash_dash_help_prints(path: list[str], width: int) -> None:
    del width
    expected = runner.invoke(app, [*path, "--help"])
    result = runner.invoke(app, ["help", *path])
    assert expected.exit_code == 0
    assert result.exit_code == 0, result.stderr
    assert result.stdout == expected.stdout
    assert result.stderr == ""
    assert f"Usage: pup-finetune {' '.join(path)}".rstrip() in plain(result.stdout)


def test_help_alone_prints_the_top_level_help(width: int) -> None:
    del width
    result = runner.invoke(app, ["help"])
    assert result.exit_code == 0
    assert result.stdout == runner.invoke(app, ["--help"]).stdout
    assert "Usage: pup-finetune [OPTIONS] COMMAND" in plain(result.stdout)


def test_an_unknown_command_is_a_usage_error_naming_the_commands() -> None:
    result = runner.invoke(app, ["help", "dataset", "nope"])
    assert result.exit_code == 2
    assert result.stdout == ""
    error = result.stderr.splitlines()[0]
    assert error.startswith("Error: `pup-finetune dataset` has no command `nope`")
    for name in ("pull", "show", "push", "triage"):
        assert name in error
    assert "Hint: run `pup-finetune help dataset`" in result.stderr


def test_an_unknown_top_level_command_names_the_top_level_commands() -> None:
    result = runner.invoke(app, ["help", "nope"])
    assert result.exit_code == 2
    assert "train" in result.stderr
    assert "dataset" in result.stderr
    assert "_worker" not in result.stderr


@pytest.mark.parametrize("words", [["train", "extra"], ["_worker"]])
def test_words_past_a_command_or_a_hidden_command_are_refused(words: list[str]) -> None:
    result = runner.invoke(app, ["help", *words])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith("Error: ")


def longest_line(text: str) -> int:
    return max(len(line) for line in plain(text).splitlines())


def test_off_a_terminal_help_is_at_least_help_width_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", None)
    result = runner.invoke(app, ["help", "train"], env={"COLUMNS": "80"})
    assert result.exit_code == 0
    assert longest_line(result.stdout) == HELP_WIDTH
    assert "…" not in result.stdout  # nothing cut off
    assert "--eval-max-cases-per-trajectory" in plain(result.stdout)
    assert rich_utils.MAX_WIDTH is None  # restored
    narrow = runner.invoke(app, ["train", "--help"], env={"COLUMNS": "80"})
    assert longest_line(narrow.stdout) == 80  # --help itself is unchanged


@pytest.mark.parametrize("env", [{"COLUMNS": "200"}, {}])
def test_off_a_terminal_a_larger_width_is_kept(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", None if env else 200)
    result = runner.invoke(app, ["help", "train"], env=env)
    assert result.exit_code == 0
    assert longest_line(result.stdout) == 200


def usage_error(capsys: pytest.CaptureFixture[str], args: list[str]) -> str:
    with pytest.raises(SystemExit) as info:
        run_app(app, args)
    assert info.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    return plain(err)


def test_standalone_usage_errors_name_pup_finetune_and_point_to_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    err = usage_error(capsys, ["train", "--bogus"])
    assert "Usage: pup-finetune train [OPTIONS]" in err
    assert "Try 'pup-finetune help train' for help." in err
    assert "--help'" not in err


def test_under_pup_usage_errors_name_pup_finetune_and_point_to_help(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_OUTPUT", "json")
    err = usage_error(capsys, ["dataset", "show"])
    assert "Usage: pup finetune dataset show [OPTIONS]" in err
    assert "Try 'pup finetune help dataset show' for help." in err
    err = usage_error(capsys, ["nope"])
    assert "Try 'pup finetune help' for help." in err


def test_under_pup_help_prints_its_text_on_stdout_naming_pup_finetune(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_OUTPUT", "json")
    # Any pup call would fail here: help text never goes through `pup format`.
    monkeypatch.setenv("PUP_FINETUNE_PUP", "/nonexistent/pup")
    with pytest.raises(SystemExit) as info:
        run_app(app, ["help", "dataset", "pull"])
    assert info.value.code in (0, None)
    out, err = capsys.readouterr()
    assert plain(out).lstrip().startswith("Usage: pup finetune dataset pull [OPTIONS]")
    assert "--ml-app" in plain(out)
    assert err == ""


def test_prog_name_follows_the_runtime() -> None:
    assert prog_name(Runtime()) == "pup-finetune"
    assert prog_name(Runtime(output_format="json")) == "pup finetune"


def test_other_programs_keep_the_dash_dash_help_hint() -> None:
    other = typer.Typer(add_completion=False)

    @other.command()
    def run(count: Annotated[int, typer.Option()] = 0) -> None:
        del count

    @other.command()
    def stop() -> None:
        pass

    result = runner.invoke(other, ["run", "--count", "x"], prog_name="other")
    assert result.exit_code == 2
    assert "Try 'other run --help' for help." in plain(result.stderr)
