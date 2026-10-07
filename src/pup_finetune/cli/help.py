# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`help`: the text ``--help`` prints, through a command pup does not intercept.

Under a coding agent, pup answers ``pup finetune <command> --help`` with its own schema and
never runs us. ``pup finetune help <command>`` has no ``--help`` token, so it reaches us, and
prints the help that ``<command> --help`` prints, rendered from the same Click objects. It is
one of the few commands whose stdout is text rather than one JSON document.

Click's usage errors end with a line pointing to this command (``Try 'pup finetune help
train' for help.``) instead of to ``--help`` (:func:`install_help_hint`).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

import typer
from typer import rich_utils

from pup_finetune.cli._common import PROG_NAME, PUP_PROG_NAME, run_command
from pup_finetune.errors import UsageError

HELP_WIDTH = 120
"""The narrowest width help renders at off a terminal, so no flag name or default is cut."""


def register(app: typer.Typer) -> None:
    """Add ``help`` to the root ``app`` and point usage errors to it."""
    app.command(
        "help",
        help=(
            "Show the help for a command: the same text as `<command> --help`. "
            "Under a coding agent pup answers `--help` itself, so use this instead. "
            "With no command, shows the top-level help. Prints text, not JSON."
        ),
        short_help="Show the help for a command.",
    )(show_help)
    install_help_hint()


@run_command
def show_help(
    ctx: typer.Context,
    command: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="COMMAND",
            help="The command, such as `train` or `dataset pull`; none for the top-level help.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Print the help of ``command`` (the program's own with none), as ``--help`` does."""
    target = resolve(ctx.find_root(), command or [])
    with _help_width():
        # As Click's own --help callback prints it.
        typer.echo(target.get_help(), color=target.color)


def resolve(root: Any, words: list[str]) -> Any:
    """The context ``<words> --help`` would print help for, below ``root``'s context.

    Built the way Click builds a subcommand's context, without parsing arguments, so the help
    is the same. Raises :class:`UsageError` naming the known commands for an unknown word.
    """
    ctx = root
    for word in words:
        group = ctx.command
        sub = group.get_command(ctx, word) if hasattr(group, "list_commands") else None
        if sub is None or sub.hidden:
            raise _unknown(ctx, word)
        ctx = sub.context_class(sub, info_name=word, parent=ctx, **sub.context_settings)
    return ctx


def _unknown(ctx: Any, word: str) -> UsageError:
    path = ctx.command_path
    if not hasattr(ctx.command, "list_commands"):
        return UsageError(f"`{path}` has no subcommand `{word}`", hint=f"run `{_help_hint(path)}`")
    known = [
        name
        for name in ctx.command.list_commands(ctx)
        if (sub := ctx.command.get_command(ctx, name)) is not None and not sub.hidden
    ]
    return UsageError(
        f"`{path}` has no command `{word}`; the commands are: {', '.join(known)}",
        hint=f"run `{_help_hint(path)}`",
    )


def _help_hint(command_path: str) -> str:
    return _help_for(command_path) or f"{command_path} --help"


def _stdout_is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


@contextmanager
def _help_width() -> Iterator[None]:
    """Render help at :data:`HELP_WIDTH` columns or more when stdout is not a terminal.

    A larger width (``TERMINAL_WIDTH``, which Typer reads, or ``COLUMNS``) is kept. On a
    terminal the help fits the terminal, as ``--help`` does.
    """
    previous = rich_utils.MAX_WIDTH
    if not _stdout_is_tty():
        columns = os.environ.get("COLUMNS", "")
        current = previous or (int(columns) if columns.isdigit() else 0)
        rich_utils.MAX_WIDTH = max(current, HELP_WIDTH)
    try:
        yield
    finally:
        rich_utils.MAX_WIDTH = previous


def _help_for(command_path: str) -> str | None:
    """The ``help`` command for ``command_path`` (e.g. ``pup finetune help train``).

    ``None`` when ``command_path`` is not under this program.
    """
    for prog in (PUP_PROG_NAME, PROG_NAME):
        if command_path == prog or command_path.startswith(f"{prog} "):
            return f"{prog} help{command_path.removeprefix(prog)}"
    return None


class _HelpHint(str):
    """Typer's "Try ..." line for usage errors, pointing to ``help`` for our commands.

    Typer formats the line with ``command_path`` and ``help_option``; for any other program
    it reads as before.
    """

    __slots__ = ()

    def format(self, *args: object, **kwargs: object) -> str:
        command_path = str(kwargs.get("command_path", ""))
        command = _help_for(command_path)
        if command is None:
            return super().format(*args, **kwargs)
        return f"Try [blue]'{command}'[/] for help."


def install_help_hint() -> None:
    """Make Click's usage errors say ``Try 'pup finetune help <command>'`` (idempotent)."""
    if not isinstance(rich_utils.RICH_HELP, _HelpHint):
        rich_utils.RICH_HELP = _HelpHint(rich_utils.RICH_HELP)
