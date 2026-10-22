# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Every `pup finetune` command and flag the skill, the README and the evaluation guide mention
exists in the CLI.

So does every `--provider` and `--model` they pass, and the skill and the README name each real
provider.

Also: the README names the Pup version `doctor` requires.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import typer
import typer.main

from pup_finetune.cli import app
from pup_finetune.datadog.pup import MIN_PUP_VERSION
from pup_finetune.providers import get_provider, providers_for_model
from pup_finetune.runtime import Runtime

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / "skills" / "datadog-finetune" / "SKILL.md"
README = REPO_ROOT / "README.md"
GUIDE = REPO_ROOT / "docs" / "evaluate.md"
TRAINING_DOCS = [SKILL, README]
"""The docs whose examples train on both providers, so their `--model` checks must name both."""
DOCS = [*TRAINING_DOCS, GUIDE]

PREFIX = ("pup", "finetune")
SHELL_STOP = {">", ">>", "|", "&&", "||", ";", "#", "\\"}

# Flags of other tools the docs use: `pup llm-obs spans search` and `get-content` (as
# `pup --no-agent llm-obs spans search --help` lists them in pup 1.10.7), `pup extension
# install`, pup's own global flags and `uv sync`.
OTHER_TOOL_FLAGS = {
    "--ml-app",
    "--span-kind",
    "--span-name",
    "--from",
    "--to",
    "--limit",
    "--summary",
    "--query",
    "--tags",
    "--trace-id",
    "--span-id",
    "--field",
    "--local",
    "--link",
    "--description",
    "--jq",
    "--output",
    "--read-only",
    "--no-agent",
    "--all-extras",
    "--extra",
}

_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_SPAN = re.compile(r"`([^`\n]+)`")
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def code_snippets(text: str) -> Iterator[str]:
    """Every line of every fenced block, then every inline code span outside them."""
    for block in _FENCE.findall(text):
        yield from block.splitlines()
    yield from _SPAN.findall(_FENCE.sub("", text))


def invocations(snippet: str) -> Iterator[list[str]]:
    """The words after each `pup finetune` in ``snippet``, up to a shell operator."""
    words = snippet.split()
    for index in range(len(words) - 1):
        if tuple(words[index : index + 2]) != PREFIX:
            continue
        rest: list[str] = []
        for word in words[index + 2 :]:
            if word in SHELL_STOP or word == "pup":
                break
            rest.append(word)
        yield rest


def root() -> tuple[Any, Any]:
    command = typer.main.get_command(app)
    return command, typer.Context(command, info_name="pup finetune")


def resolve(words: list[str]) -> tuple[Any, list[str], list[str]]:
    """The command ``words`` name, the names that led to it, and the words after them."""
    command, ctx = root()
    path: list[str] = []
    rest = list(words)
    while rest and hasattr(command, "list_commands"):
        sub = command.get_command(ctx, rest[0])
        if sub is None:
            break
        command = sub
        path.append(rest.pop(0))
    return command, path, rest


def option_names(command: Any) -> set[str]:
    names = {"--help"}
    for param in command.params:
        if param.param_type_name == "option":
            names.update(param.opts)
            names.update(param.secondary_opts)
    return names


def every_option_name() -> set[str]:
    command, ctx = root()
    names: set[str] = set()
    stack = [command]
    while stack:
        current = stack.pop()
        if hasattr(current, "list_commands"):
            stack.extend(current.get_command(ctx, name) for name in current.list_commands(ctx))
        else:
            names |= option_names(current)
    return names | option_names(command)


def help_problems(snippet: str, words: list[str]) -> list[str]:
    """What is wrong with ``pup finetune help <words>``: it must name a command, nothing more."""
    _, path, rest = resolve(words)
    if not path or rest:
        return [f"{snippet!r}: pup finetune help names no command {words}"]
    return []


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_command_and_its_flags_exist(doc: Path) -> None:
    problems: list[str] = []
    seen = 0
    for snippet in code_snippets(doc.read_text(encoding="utf-8")):
        for words in invocations(snippet):
            if not words or words[0].startswith("<"):
                continue
            command, path, rest = resolve(words)
            name = " ".join(["pup finetune", *path])
            if hasattr(command, "list_commands"):
                if rest and rest[0] in option_names(command):
                    continue  # a group's own option, such as `pup finetune --help`
                problems.append(f"{snippet!r}: {name} has no subcommand {rest[:1]}")
                continue
            seen += 1
            if path == ["help"] and rest and not rest[0].startswith("<"):
                problems += help_problems(snippet, rest)
                continue
            known = option_names(command)
            for word in rest:
                flag = word.split("=", 1)[0]
                if flag.startswith("-") and flag not in known:
                    problems.append(f"{snippet!r}: {name} has no option {flag}")
    assert not problems, "\n".join(problems)
    assert seen >= 5, f"found only {seen} `pup finetune` commands in {doc.name}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_flag_mentioned_belongs_to_a_command(doc: Path) -> None:
    known = every_option_name() | OTHER_TOOL_FLAGS
    flags = set(_FLAG.findall(doc.read_text(encoding="utf-8")))
    unknown = sorted(flags - known)
    assert not unknown, f"{doc.name} mentions flags no command has: {unknown}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_provider_and_model_passed_exists(doc: Path) -> None:
    providers: set[str] = set()
    for snippet in code_snippets(doc.read_text(encoding="utf-8")):
        for words in invocations(snippet):
            values = dict(itertools.pairwise(words))
            name, model = values.get("--provider"), values.get("--model")
            if model is not None and not model.startswith("<"):
                offered_by = providers_for_model(model, Runtime())
                assert name is not None or len(offered_by) == 1, (
                    f"{doc.name}: --model {model} needs --provider (offered by {offered_by})"
                )
                name = name or offered_by[0]
                get_provider(name, Runtime()).resolve_model(model)
                providers.add(name)
            elif name is not None and not name.startswith("<"):
                get_provider(name, Runtime())
                providers.add(name)
    if doc in TRAINING_DOCS:
        assert {"fireworks", "baseten"} <= providers, f"{doc.name} names only {providers}"


def test_the_checker_catches_a_wrong_command_and_flag() -> None:
    command, path, rest = resolve(["dataset", "pull", "DIR", "--ml-app", "x"])
    assert rest == ["DIR", "--ml-app", "x"]
    assert path == ["dataset", "pull"]
    assert "--ml-app" in option_names(command)
    assert "--not-a-flag" not in option_names(command)
    command, path, _ = resolve(["dataset", "nope"])
    assert path == ["dataset"]
    assert hasattr(command, "list_commands")
    assert list(invocations("pup finetune dataset triage --print-template > rubric.md")) == [
        ["dataset", "triage", "--print-template"]
    ]
    assert help_problems("s", ["dataset", "pull"]) == []
    assert help_problems("s", ["dataset", "nope"]) != []
    assert help_problems("s", ["nope"]) != []


def test_the_readme_names_the_minimum_pup_version() -> None:
    minimum = ".".join(str(part) for part in MIN_PUP_VERSION)
    assert f"Pup {minimum} or later" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
