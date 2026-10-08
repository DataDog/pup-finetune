# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Every command a hint, error or help text in ``src/`` tells the reader to run must work.

Each backticked ``pup finetune ...`` in a string literal (f-strings included, a field read as
``1``) must parse with the Typer app: the command exists and every option it names exists
and takes the value given. Parsing only: nothing runs. A hint may leave out required options
(``see `pup finetune evaluate```), so a missing parameter is accepted, and so is a path that
does not exist here. Each backticked ``uv ...`` must be on a small allowlist, since ``uv sync``
is exact and a bare ``uv sync --extra X`` uninstalls every other extra; that form may appear
only as the warning ``a bare `uv sync --extra X```.
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import typer
import typer.main

from pup_finetune.cli import app

SRC = Path(__file__).resolve().parent.parent / "src" / "pup_finetune"

COMMAND = re.compile(r"(?<!`)`((?:pup finetune|uv)(?: [^`]*)?)`(?!`)")
"""A command in single backticks (docs in double backticks are names, not commands)."""

FIELD = "1"
"""What an f-string field reads as: a valid path, number and name alike."""

UV_ALLOWED = frozenset({"uv sync --all-extras", "uv sync --all-extras --reinstall"})
"""The ``uv`` commands a hint may offer."""

UV_WARNED = re.compile(r"uv sync --extra \S+")
"""A ``uv`` command a hint may name only as what not to run."""


def _text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_text(part) or FIELD for part in node.values)
    return None


def _docstrings(tree: ast.AST) -> set[int]:
    """The ids of string statements: docstrings and attribute docstrings, which document
    names rather than tell the reader what to run."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and _text(node.value) is not None
    }


def strings(path: Path) -> Iterator[str]:
    """Every string literal in ``path`` that is not a docstring, each f-string whole."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstrings(tree)
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            inside.update(id(child) for child in ast.walk(node) if child is not node)
    for node in ast.walk(tree):
        if id(node) in skip or id(node) in inside:
            continue
        text = _text(node)
        if text is not None:
            yield text


def commands() -> Iterator[tuple[str, str, str]]:
    """``(file, the string, the command)`` for every backticked command in ``src/``."""
    for path in sorted(SRC.rglob("*.py")):
        for text in strings(path):
            for match in COMMAND.finditer(text):
                yield str(path.relative_to(SRC)), text, match[1]


def _parse_problem(command: str) -> str | None:
    """Why ``command`` (``pup finetune ...``) does not parse with the app, or ``None``."""
    words = shlex.split(command)[2:]
    group: Any = typer.main.get_command(app)
    ctx = group.make_context("pup finetune", [], resilient_parsing=True)
    while words and hasattr(group, "list_commands"):
        sub = group.get_command(ctx, words[0])
        if sub is None:
            return f"no command {words[0]!r}"
        name = words.pop(0)
        if hasattr(sub, "list_commands"):
            ctx = sub.make_context(name, [], parent=ctx, resilient_parsing=True)
            group = sub
            continue
        try:
            with sub.make_context(name, words, parent=ctx):
                return None
        except Exception as exc:
            kind = type(exc).__name__
            message = str(getattr(exc, "message", exc))
            # Click's own, or ours naming every missing parameter (raised from Click's).
            missing = "MissingParameter" in (kind, type(exc.__cause__).__name__)
            if missing or "does not exist" in message:
                return None
            return f"{kind}: {message}"
    if words:
        return f"{words} after a command"
    return None  # names the program or a group


def test_the_scan_finds_the_hints() -> None:
    found = [command for _, _, command in commands()]
    assert sum(command.startswith("pup finetune ") for command in found) >= 20
    assert any(command.startswith("uv ") for command in found)


@pytest.mark.parametrize(
    ("where", "command"),
    sorted({(where, command) for where, _, command in commands() if command.startswith("pup")}),
)
def test_every_pup_finetune_command_in_a_hint_parses(where: str, command: str) -> None:
    problem = _parse_problem(command)
    assert problem is None, f"{where}: `{command}`: {problem}"


def test_every_uv_command_in_a_hint_is_safe() -> None:
    problems = []
    for where, text, command in commands():
        if not command.startswith("uv"):
            continue
        if command in UV_ALLOWED:
            continue
        if UV_WARNED.fullmatch(command) and f"a bare `{command}`" in text:
            continue
        problems.append(f"{where}: `{command}`")
    assert problems == []


def test_the_parser_check_catches_a_wrong_flag() -> None:
    assert _parse_problem("pup finetune train --no-publish") is not None
    assert _parse_problem("pup finetune evaluate run --eval-no-publish") is not None
    assert _parse_problem("pup finetune nope") is not None
    assert _parse_problem("pup finetune train --eval-no-publish") is None
    assert _parse_problem("pup finetune evaluate run --no-publish") is None
