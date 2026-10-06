# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The agent skill ships in the package and stays aligned with the CLI and the pinned SDK."""

from __future__ import annotations

import re
import shlex
import tomllib
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

import typer.main

from pup_finetune.cli import app
from pup_finetune.providers.baseten.registration import TESTED_LOOPS_VERSION

SKILL = resources.files("pup_finetune") / "skills" / "pup-finetune" / "SKILL.md"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

_FENCED = re.compile(r"^```[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_SEPARATORS = frozenset({"|", "||", "&&", ";"})


def _frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    head, closed, _ = text[4:].partition("\n---\n")
    assert closed, "the frontmatter is not closed"
    fields: dict[str, str] = {}
    for line in head.splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            fields[key] = value.strip()
    return fields


def _invocations(text: str) -> Iterator[tuple[str, list[str]]]:
    """Each ``pup finetune ...`` command in a fenced block: its line and its arguments."""
    for block in _FENCED.findall(text):
        for line in block.replace("\\\n", " ").splitlines():
            if "pup finetune" not in line:
                continue
            tokens = shlex.split(line, comments=True)
            for start in range(len(tokens) - 1):
                if tokens[start : start + 2] == ["pup", "finetune"]:
                    args: list[str] = []
                    for token in tokens[start + 2 :]:
                        if token in _SEPARATORS:
                            break
                        args.append(token)
                    yield line, args


def _options(command: Any) -> set[str]:
    names = (name for param in command.params for name in (*param.opts, *param.secondary_opts))
    return {"--help", *(name for name in names if name.startswith("-"))}


def _all_options(command: Any) -> set[str]:
    subcommands = getattr(command, "commands", {}).values()
    return _options(command).union(*(_all_options(sub) for sub in subcommands))


def _problems(text: str) -> list[str]:
    """Every command or flag in ``text``'s code blocks that the Typer app does not have."""
    root = typer.main.get_command(app)
    problems: list[str] = []
    for line, args in _invocations(text):
        command, path = root, ["pup", "finetune"]
        rest = list(args)
        while rest and rest[0] in getattr(command, "commands", {}):
            path.append(rest.pop(0))
            command = command.commands[path[-1]]
        if hasattr(command, "commands") and rest and not rest[0].startswith("-"):
            problems.append(f"{line!r}: `{' '.join(path)}` has no command {rest[0]!r}")
            continue
        options = _options(command)
        for token in rest:
            flag = token.partition("=")[0]
            if flag.startswith("-") and flag not in options:
                problems.append(f"{line!r}: `{' '.join(path)}` has no option {flag}")
    return problems


def test_skill_is_packaged_with_name_and_description() -> None:
    assert SKILL.is_file()
    fields = _frontmatter(SKILL.read_text(encoding="utf-8"))
    assert fields.get("name") == "pup-finetune"
    assert 0 < len(fields.get("description", "")) <= 1024


def test_skill_commands_and_flags_exist_in_the_cli() -> None:
    text = SKILL.read_text(encoding="utf-8")
    commands = {" ".join(args[:1]) for _, args in _invocations(text)}
    assert {"train", "deploy", "undeploy", "doctor", "auth"} <= commands
    assert _problems(text) == []


def test_every_flag_the_skill_mentions_exists_in_the_cli() -> None:
    mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", SKILL.read_text(encoding="utf-8")))
    assert mentioned - _all_options(typer.main.get_command(app)) == set()


def test_skill_check_reports_unknown_commands_and_flags() -> None:
    text = (
        "```sh\n"
        "pup finetune trian --train a.jsonl\n"
        "pup finetune deploy --run-dir d \\\n  --gpu H100:1 --confirm  # comment --ignored\n"
        "pup finetune auth login baseten --stdin | pup finetune undeploy --nope=1\n"
        "```\n"
    )
    assert _problems(text) == [
        "'pup finetune trian --train a.jsonl': `pup finetune` has no command 'trian'",
        "'pup finetune deploy --run-dir d    --gpu H100:1 --confirm  # comment --ignored': "
        "`pup finetune deploy` has no option --gpu",
        "'pup finetune auth login baseten --stdin | pup finetune undeploy --nope=1': "
        "`pup finetune undeploy` has no option --nope",
    ]


def test_baseten_extra_pins_the_tested_loops_version() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    pins = [
        requirement.removeprefix("baseten-loops==")
        for requirement in project["optional-dependencies"]["baseten"]
        if requirement.startswith("baseten-loops==")
    ]
    assert pins == [TESTED_LOOPS_VERSION]
