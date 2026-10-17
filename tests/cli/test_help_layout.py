# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""How `--help` looks: 80 columns with nothing cut off, short, ordered, and described once.

Help is rendered as a pipe gets it: stdout is not a terminal and no ``COLUMNS`` is set, so
Rich falls back to 80 columns (one test checks that in a real process). Every command is
described by a :class:`~pup_finetune.cli._presentation.CommandDoc`, whose examples must parse
with the app and whose safety footer ends the help.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
import typer
import typer.main
from typer import rich_utils
from typer.testing import CliRunner

from pup_finetune.cli import app
from pup_finetune.cli._presentation import (
    BUILD_A_DATASET,
    COMMAND_ORDER,
    DOCS,
    EVALUATE_AND_SHIP,
    SET_UP,
    TRAIN,
    Safety,
    wrap_example,
)

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

WIDTH = 80
"""What help renders at with stdout piped and no ``COLUMNS``: every agent sees this."""
MAX_LINES = 60
LINE_LIMITS = {"train": 65}
"""Commands allowed more than :data:`MAX_LINES`, and why: ``train`` has 37 options in six
panels, which take 50 lines before its usage, description and examples (60 at the least)."""

SHELL_STOP = {">", ">>", "<", "|", "&&", "||", ";"}


def plain(text: str) -> str:
    return _ANSI.sub("", text)


def paths(command: Any, ctx: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Every visible command and group, the program itself first."""
    yield path
    if hasattr(command, "list_commands"):
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None and not sub.hidden:
                yield from paths(sub, ctx, (*path, name))


def root() -> tuple[Any, Any]:
    command = typer.main.get_command(app)
    return command, typer.Context(command, info_name="pup finetune")


def every_path() -> list[tuple[str, ...]]:
    return list(paths(*root()))


def command_at(path: tuple[str, ...]) -> Any:
    command, ctx = root()
    for name in path:
        command = command.get_command(ctx, name)
        assert command is not None, path
    return command


def leaves() -> list[str]:
    return [" ".join(path) for path in every_path() if path and not is_group(path)]


def is_group(path: tuple[str, ...]) -> bool:
    return hasattr(command_at(path), "list_commands")


@pytest.fixture
def narrow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", WIDTH)


def rendered(path: tuple[str, ...]) -> str:
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    return plain(result.stdout)


@pytest.mark.parametrize("path", every_path(), ids=lambda path: " ".join(path) or "<root>")
def test_help_fits_80_columns_with_nothing_cut_off_and_stays_short(
    path: tuple[str, ...], narrow: None
) -> None:
    del narrow
    text = rendered(path)
    lines = text.splitlines()
    assert max(len(line) for line in lines) <= WIDTH
    assert "…" not in text, [line for line in lines if "…" in line]
    name = " ".join(path)
    limit = LINE_LIMITS.get(name, MAX_LINES)
    assert len(lines) <= limit, f"{name or '<root>'}: {len(lines)} lines, more than {limit}"
    assert "{" not in lines[1], lines[1]  # Usage: ... DIR, never {DIR}


def test_piped_help_without_columns_renders_at_80() -> None:
    env = {key: value for key, value in os.environ.items() if key not in ("COLUMNS", "LINES")}
    env.pop("TERMINAL_WIDTH", None)
    result = subprocess.run(
        [sys.executable, "-m", "pup_finetune", "dataset", "pull", "--help"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert max(len(line) for line in plain(result.stdout).splitlines()) == WIDTH


@pytest.mark.parametrize("path", every_path(), ids=lambda path: " ".join(path) or "<root>")
def test_dash_h_is_help(path: tuple[str, ...]) -> None:
    assert runner.invoke(app, [*path, "-h"]).stdout == runner.invoke(app, [*path, "--help"]).stdout


def test_commands_are_listed_in_workflow_order() -> None:
    command, ctx = root()
    shown = [name for name in command.list_commands(ctx) if name != "_worker"]
    assert shown[:5] == ["doctor", "auth", "models", "skill", "help"]
    assert shown[5:] == [
        "dataset",
        "benchmark",
        "train",
        "status",
        "runs",
        "resume",
        "cancel",
        "evaluate",
        "deploy",
        "undeploy",
    ]
    dataset = command.get_command(ctx, "dataset")
    dataset_ctx = dataset.make_context("dataset", [], parent=ctx, resilient_parsing=True)
    assert dataset.list_commands(dataset_ctx) == [
        "pull",
        "import",
        "show",
        "sample",
        "label",
        "triage",
        "push",
        "fetch",
    ]
    assert set(leaves()) == set(COMMAND_ORDER)
    assert leaves() == list(COMMAND_ORDER)


def test_the_top_level_help_groups_commands_under_the_flow() -> None:
    command, ctx = root()
    panels = {
        name: getattr(command.get_command(ctx, name), "rich_help_panel", None)
        for name in command.list_commands(ctx)
        if name != "_worker"
    }
    assert panels == {
        "doctor": SET_UP,
        "auth": SET_UP,
        "models": SET_UP,
        "skill": SET_UP,
        "help": SET_UP,
        "dataset": BUILD_A_DATASET,
        "benchmark": TRAIN,
        "train": TRAIN,
        "status": TRAIN,
        "runs": TRAIN,
        "resume": TRAIN,
        "cancel": TRAIN,
        "evaluate": EVALUATE_AND_SHIP,
        "deploy": EVALUATE_AND_SHIP,
        "undeploy": EVALUATE_AND_SHIP,
    }
    text = " ".join(rendered(()).split())
    assert "Flow: dataset pull -> review -> dataset push -> train -> evaluate -> deploy" in text


def test_train_groups_its_options_in_panels() -> None:
    panels: dict[str, list[str]] = {}
    for param in command_at(("train",)).params:
        if param.param_type_name == "option" and not param.hidden:
            panels.setdefault(param.rich_help_panel, []).append(param.opts[-1])
    assert list(panels) == [
        "Required",
        "Run",
        "Training",
        "Budget",
        "Evaluate after training (with --evaluate)",
        "Advanced",
    ]
    assert panels["Required"] == ["--dataset", "--model"]
    assert panels["Run"] == [
        "--provider",
        "--run-dir",
        "--run-id",
        "--detach",
        "--remote",
        "--team",
        "--confirm",
    ]
    assert "--max-seq-len" in panels["Training"]
    assert "--provider-option" in panels["Advanced"]
    assert all(flag.startswith("--eval") for flag in panels[list(panels)[4]])


def sentences(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text.replace("e.g.", "eg").replace("i.e.", "ie")))


@pytest.mark.parametrize("name", leaves())
def test_every_command_is_described_once(name: str, narrow: None) -> None:
    del narrow
    doc = DOCS[name]
    command = command_at(tuple(name.split()))
    assert command.short_help == doc.short
    assert len(doc.short.split()) <= 12
    assert sentences(doc.description) <= 3, doc.description
    takes_something = any(not param.hidden for param in command.params)
    assert len(doc.examples) >= (2 if takes_something else 1)
    text = " ".join(rendered(tuple(name.split())).split())
    assert " ".join(doc.description.split()) in text
    if doc.more:  # after \f: only the generated docs show it
        assert " ".join(doc.more.split())[:40] not in text
    assert text.endswith(doc.safety.footer())


@pytest.mark.parametrize("name", leaves())
def test_the_safety_record_matches_the_command(name: str) -> None:
    safety = DOCS[name].safety
    has_confirm = any(param.name == "confirm" for param in command_at(tuple(name.split())).params)
    assert (safety.confirm is not None) == has_confirm


def test_read_only_refuses_confirm_for_paid_runs_and_remote_writes_only() -> None:
    refused = {name for name in leaves() if DOCS[name].safety.read_only == "confirm_refused"}
    assert refused == {
        "dataset push",
        "benchmark",
        "train",
        "resume",
        "evaluate",
        "deploy",
        "undeploy",
    }


def test_the_footer_says_cost_data_and_read_only() -> None:
    free = Safety(data="reads Datadog", recipients="nowhere", without="previews")
    assert free.footer() == "Free · reads Datadog · PUP_READ_ONLY: allowed"
    paid = Safety(
        data="sends rows to the provider",
        recipients="the provider",
        without="previews",
        confirm="trains",
        paid=True,
        read_only="confirm_refused",
    )
    assert paid.footer() == (
        "Paid with --confirm · sends rows to the provider · PUP_READ_ONLY: preview only"
    )
    with pytest.raises(ValueError, match="takes --confirm"):
        Safety(data="x", recipients="x", without="x", paid=True)


def example_problem(example: str) -> str | None:
    """Why ``example`` does not parse with the app as a complete command, or ``None``."""
    words = shlex.split(example)
    while words and re.fullmatch(r"[A-Z_]+=\S*", words[0]):
        words.pop(0)  # an environment assignment
    if words[:2] != ["pup", "finetune"]:
        return "does not start with pup finetune"
    words = words[2:]
    stop = next((index for index, word in enumerate(words) if word in SHELL_STOP), len(words))
    words = words[:stop]
    group, ctx = root()
    while words and hasattr(group, "list_commands"):
        name = words.pop(0)
        sub = group.get_command(ctx, name)
        if sub is None:
            return f"no command {name!r}"
        ctx = sub.make_context(name, [], parent=ctx, resilient_parsing=True)
        group = sub
    try:
        with group.make_context(ctx.info_name, words, parent=ctx.parent):
            return None
    except Exception as exc:
        message = str(getattr(exc, "message", exc))
        if "does not exist" in message:
            return None  # a file the example names, e.g. --file rows.jsonl
        return f"{type(exc).__name__}: {message}"


@pytest.mark.parametrize(
    ("name", "example"),
    [(name, example) for name in leaves() for example in DOCS[name].examples],
    ids=lambda value: value if " " not in value else None,
)
def test_every_example_runs_as_written(name: str, example: str) -> None:
    assert example_problem(example) is None, (name, example, example_problem(example))
    words = shlex.split(example)
    assert " ".join(words).find(f"pup finetune {name}") >= 0


def test_the_example_checker_catches_mistakes() -> None:
    assert example_problem("pup finetune dataset preview") is not None
    assert example_problem("pup finetune train --bogus") is not None
    assert example_problem("pup finetune nope") is not None
    assert example_problem("pup finetune dataset label ./ds --file missing.jsonl") is None
    assert example_problem("PUP_FINETUNE_DEV=1 pup finetune doctor") is None


def test_long_examples_continue_on_the_next_line() -> None:
    lines = wrap_example(
        "pup finetune train --provider fireworks --model qwen3p8-27b --dataset ds "
        "--run-dir run --detach --confirm"
    )
    assert len(lines) == 2
    assert lines[0].endswith(" \\")
    assert lines[1].startswith("    --run-dir run")
    assert all(len(line) <= 74 for line in lines)
    assert wrap_example("pup finetune triage --print-template > rubric.md") == [
        "pup finetune triage --print-template > rubric.md"
    ]
