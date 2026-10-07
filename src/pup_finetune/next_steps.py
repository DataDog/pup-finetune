# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What a result says to do next: ``next_command`` and ``next_steps``.

The rule every command follows: ``next_command`` is a command that runs exactly as written
(after the user's yes, when it ends in ``--confirm``), or ``None``. It never holds a
placeholder or ``...``. Alternatives, and commands that need a value only the user or the
agent can supply, go in ``next_steps``: a list of ``{"command", "why"}`` whose commands may
hold placeholders in one style, ``<UPPER_CASE>`` (:data:`PLACEHOLDER`). A step is never
copied into ``next_command``.

:func:`command` builds a ``next_command`` (every word shell-quoted), :func:`template` a
step's command (placeholders left as they are) and :func:`step` the step;
:func:`problems` checks a result against the rule.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any, Final

__all__ = ["PLACEHOLDER", "PROGRAM", "command", "problems", "step", "template"]

PROGRAM: Final = ("pup", "finetune")
"""The words every command starts with."""

PLACEHOLDER: Final = re.compile(r"<[A-Z][A-Z0-9_]*>")
"""A value a ``next_steps`` command leaves to the reader, e.g. ``<NAME>``."""


def command(*words: object) -> str:
    """``pup finetune`` and ``words``, each shell-quoted: a command that runs as written."""
    return shlex.join([*PROGRAM, *(str(word) for word in words)])


def template(*words: object) -> str:
    """Like :func:`command`, but a word that is a :data:`PLACEHOLDER` is left unquoted."""
    return " ".join(
        text if PLACEHOLDER.fullmatch(text) else shlex.quote(text)
        for text in (*PROGRAM, *(str(word) for word in words))
    )


def step(command: str, why: str) -> dict[str, str]:
    """One ``next_steps`` entry."""
    return {"command": command, "why": why}


def problems(result: Any, *, where: str = "result") -> list[str]:
    """How ``result`` (and every mapping inside it) breaks the rule; empty when it does not.

    A ``next_command`` must be ``None`` or a non-empty command with no ``...`` and no ``<``;
    ``next_steps`` must be a list of ``{"command", "why"}`` whose commands have no ``...``
    and no ``<`` outside a :data:`PLACEHOLDER`. The old ``next_commands`` list is refused.
    """
    found: list[str] = []
    if isinstance(result, Mapping):
        for key, value in result.items():
            path = f"{where}.{key}"
            if key == "next_command":
                found += _command_problems(value, path, placeholders=False)
            elif key == "next_steps":
                found += _steps_problems(value, path)
            elif key == "next_commands":
                found.append(f"{path}: use next_steps")
            else:
                found += problems(value, where=path)
    elif isinstance(result, Sequence) and not isinstance(result, str | bytes):
        for index, item in enumerate(result):
            found += problems(item, where=f"{where}[{index}]")
    return found


def _command_problems(value: object, path: str, *, placeholders: bool) -> list[str]:
    if value is None and not placeholders:
        return []
    if not isinstance(value, str) or not value.strip():
        return [f"{path}: expected a command, got {value!r}"]
    text = PLACEHOLDER.sub("", value) if placeholders else value
    found = []
    if "..." in text:
        found.append(f"{path}: {value!r} holds '...'")
    if "<" in text or ">" in text:
        found.append(f"{path}: {value!r} holds a placeholder that is not <UPPER_CASE>")
    try:
        words = shlex.split(value)
    except ValueError:
        words = []
    if words[: len(PROGRAM)] != list(PROGRAM):
        found.append(f"{path}: {value!r} is not a {' '.join(PROGRAM)} command")
    return found


def _steps_problems(value: object, path: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{path}: expected a list, got {value!r}"]
    found: list[str] = []
    for index, item in enumerate(value):
        where = f"{path}[{index}]"
        if not isinstance(item, Mapping) or set(item) != {"command", "why"}:
            found.append(f"{where}: expected {{command, why}}, got {item!r}")
            continue
        if not isinstance(item["why"], str) or not item["why"].strip():
            found.append(f"{where}.why: expected a reason")
        found += _command_problems(item["command"], f"{where}.command", placeholders=True)
    return found
