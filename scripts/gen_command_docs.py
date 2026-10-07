#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Generate the command reference and the exclusion-code table from the code.

Usage: uv run python scripts/gen_command_docs.py [--check]

Writes ``docs/commands.md`` from the Typer app (every visible command, its help, arguments
and options) and the exclusion-code table between the markers in ``docs/data-format.md``
from ``pup_finetune.data.exclusions``. With ``--check``, writes nothing and exits 1 when
either file is out of date.

The output is deterministic: it is built from the commands' parameters, not from rendered
``--help`` (which depends on the terminal width, color settings and ``GITHUB_ACTIONS``).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMANDS_DOC = REPO_ROOT / "docs" / "commands.md"
DATA_FORMAT_DOC = REPO_ROOT / "docs" / "data-format.md"

PROG = "pup finetune"
EXCLUSIONS_BEGIN = "<!-- BEGIN GENERATED: exclusion codes (scripts/gen_command_docs.py) -->"
EXCLUSIONS_END = "<!-- END GENERATED: exclusion codes -->"

# Commands in workflow order; any command not listed follows, sorted by name.
ORDER: tuple[str, ...] = (
    "doctor",
    "auth login",
    "auth status",
    "auth logout",
    "models list",
    "dataset preview",
    "dataset pull",
    "dataset import",
    "dataset show",
    "dataset sample",
    "dataset label",
    "dataset triage",
    "dataset push",
    "train",
    "status",
    "resume",
    "cancel",
    "evaluate",
    "deploy",
    "undeploy",
    "skill show",
    "skill install",
)

_TYPE_NAMES = {
    "STR": "TEXT",
    "STRING": "TEXT",
    "INT": "INTEGER",
    "FLOAT": "FLOAT",
    "BOOL": "BOOLEAN",
}


def iter_commands(command: Any, ctx: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    """Every visible leaf command under ``command``, as (``"group name"``, command)."""
    if hasattr(command, "list_commands"):
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None and not sub.hidden:
                yield from iter_commands(sub, ctx, (*path, name))
    elif path:
        yield " ".join(path), command


def ordered(commands: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    rank = {name: index for index, name in enumerate(ORDER)}
    return sorted(commands, key=lambda item: (rank.get(item[0], len(ORDER)), item[0]))


def md(text: str, *, cell: bool = False) -> str:
    """``text`` safe in Markdown prose (or a table ``cell``); code spans are kept as written."""
    parts = text.split("`")
    for index in range(0, len(parts), 2):  # outside code spans
        parts[index] = (
            parts[index]
            .replace("\\", "\\\\")
            .replace("*", "\\*")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    joined = "`".join(parts)
    return joined.replace("|", "\\|") if cell else joined


def paragraphs(text: str) -> list[str]:
    """``text`` as paragraphs, each on one line."""
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text.strip()) if block]


def value_of(param: Any) -> str:
    """What an option takes, e.g. ``INTEGER (x>=1)`` or ``omit|preserve``; empty for flags."""
    if getattr(param, "is_flag", False):
        return ""
    kind = param.type
    if param.metavar:
        value = str(param.metavar)
    elif hasattr(kind, "choices"):
        value = "|".join(str(choice) for choice in kind.choices)
    elif hasattr(kind, "file_okay") and hasattr(kind, "dir_okay"):
        value = "FILE" if not kind.dir_okay else "DIR" if not kind.file_okay else "PATH"
    else:
        name = str(getattr(kind, "name", "") or "").upper()
        if name.endswith(" RANGE"):
            name = name.removesuffix(" RANGE")
        value = _TYPE_NAMES.get(name, name or "VALUE")
        if hasattr(kind, "min") and hasattr(kind, "max"):
            value += f" ({_range(kind)})"
    if param.multiple:
        value += ", repeatable"
    return value


def _range(kind: Any) -> str:
    low = None if kind.min is None else f"{kind.min}{'<' if kind.min_open else '<='}"
    high = None if kind.max is None else f"{'<' if kind.max_open else '<='}{kind.max}"
    if low is not None and high is not None:
        return f"{low}x{high}"
    if low is not None:
        return f"x{'>' if kind.min_open else '>='}{kind.min}"
    return f"x{high}"


def default_of(param: Any) -> str:
    if param.required:
        return "required"
    if getattr(param, "is_flag", False) or not getattr(param, "show_default", True):
        return ""
    default = param.default
    if default is None or callable(default) or default == () or default == []:
        return ""
    return f"`{default}`"


def usage(name: str, command: Any) -> str:
    words = [PROG, name]
    if any(param.param_type_name == "option" for param in command.params):
        words.append("[OPTIONS]")
    for param in command.params:
        if param.param_type_name != "argument":
            continue
        metavar = str(param.metavar or param.name.upper())
        words.append(metavar if param.required else f"[{metavar}]")
    return " ".join(words)


def render_command(name: str, command: Any) -> list[str]:
    lines = [f"## `{PROG} {name}`", ""]
    for paragraph in paragraphs(command.help or command.short_help or ""):
        lines += [md(paragraph), ""]
    lines += ["```", usage(name, command), "```", ""]
    arguments = [p for p in command.params if p.param_type_name == "argument"]
    options = [p for p in command.params if p.param_type_name == "option" and not p.hidden]
    if arguments:
        lines += ["| Argument | Description |", "| --- | --- |"]
        for param in arguments:
            metavar = str(param.metavar or param.name.upper())
            required = "" if param.required else " Optional."
            lines.append(f"| `{metavar}` | {md(param.help or '', cell=True)}{required} |")
        lines.append("")
    if options:
        lines += ["| Option | Value | Default | Description |", "| --- | --- | --- | --- |"]
        for param in options:
            names = ", ".join(f"`{opt}`" for opt in [*param.opts, *param.secondary_opts])
            value = value_of(param)
            cell = f"`{value}`".replace("|", "\\|") if value else ""
            help_text = md(param.help or "", cell=True)
            lines.append(f"| {names} | {cell} | {default_of(param)} | {help_text} |")
        lines.append("")
    return lines


def summary_of(command: Any) -> str:
    """The command's one-line summary: its short help, else its help's first sentence."""
    if command.short_help:
        return " ".join(command.short_help.split())
    first = paragraphs(command.help or "")
    return first[0] if first else ""


def render_commands() -> str:
    """The content of ``docs/commands.md``."""
    import typer
    import typer.main

    from pup_finetune.cli import app

    root = typer.main.get_command(app)
    ctx = typer.Context(root, info_name=PROG)
    commands = ordered(list(iter_commands(root, ctx)))
    lines = [
        "<!-- Generated by scripts/gen_command_docs.py from the CLI. Do not edit by hand. -->",
        "",
        "# Command reference",
        "",
        f"Every `{PROG}` command, with its arguments and options. `{PROG}` builds a training "
        "dataset from your Datadog Agent Observability traces, fine-tunes a model on it, "
        "evaluates it and deploys it. Each command prints one JSON "
        "result on stdout and its progress on stderr, and exits `0` on success, `1` on an "
        "external or runtime failure and `2` on a usage or validation error or a preview that "
        "needs `--confirm`. Paid and remote-changing commands preview unless you pass "
        "`--confirm`. Run any command with `--help` for the same text in your terminal.",
        "",
        "Outside Pup, run the same commands as `pup-finetune <command>`.",
        "",
        "## Commands",
        "",
    ]
    for name, command in commands:
        anchor = re.sub(r"[^a-z0-9 -]", "", f"{PROG} {name}".lower()).replace(" ", "-")
        summary = md(summary_of(command))
        lines.append(f"- [`{PROG} {name}`](#{anchor}): {summary}")
    lines.append("")
    for name, command in commands:
        lines += render_command(name, command)
    return "\n".join(lines).rstrip("\n") + "\n"


def render_exclusions() -> str:
    """The exclusion-code table that goes between the markers in ``docs/data-format.md``."""
    from pup_finetune.data.exclusions import EXPLANATIONS, ExclusionCode
    from pup_finetune.data.schema import RowErrorCode

    lines = ["| Code | Kind | Why the row was left out |", "| --- | --- | --- |"]
    for code in [*RowErrorCode, *ExclusionCode]:
        kind = "schema" if isinstance(code, RowErrorCode) else "dataset"
        explanation = EXPLANATIONS[code]
        text = md(explanation[0].upper() + explanation[1:], cell=True)
        lines.append(f"| `{code.value}` | {kind} | {text}. |")
    return "\n".join(lines)


def with_exclusions(text: str) -> str:
    """``text`` (``docs/data-format.md``) with its generated table replaced."""
    begin, end = text.find(EXCLUSIONS_BEGIN), text.find(EXCLUSIONS_END)
    if begin < 0 or end < begin:
        raise ValueError(f"{DATA_FORMAT_DOC.name} lacks the exclusion-code markers")
    head = text[: begin + len(EXCLUSIONS_BEGIN)]
    return f"{head}\n{render_exclusions()}\n{text[end:]}"


def expected() -> dict[Path, str]:
    """Each generated file and the content it should have."""
    return {
        COMMANDS_DOC: render_commands(),
        DATA_FORMAT_DOC: with_exclusions(DATA_FORMAT_DOC.read_text(encoding="utf-8")),
    }


def stale() -> list[Path]:
    """The generated files whose content differs from :func:`expected`."""
    out = []
    for path, content in expected().items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != content:
            out.append(path)
    return out


def main(argv: Sequence[str] = ()) -> int:
    if list(argv) == ["--check"]:
        paths = stale()
        for path in paths:
            print(
                f"{path.relative_to(REPO_ROOT)} is out of date; run "
                "`uv run python scripts/gen_command_docs.py`",
                file=sys.stderr,
            )
        return 1 if paths else 0
    if argv:
        print(__doc__, file=sys.stderr)
        return 2
    for path, content in expected().items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
