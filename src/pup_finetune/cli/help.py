# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`help`: the text ``--help`` prints, through a command pup does not intercept.

Under a coding agent, pup answers ``pup finetune <command> --help`` with its own schema and
never runs us. ``pup finetune help <command>`` has no ``--help`` token, so it reaches us, and
prints the help that ``<command> --help`` prints, rendered from the same Click objects at the
same width. It is one of the few commands whose stdout is text rather than one JSON document.

Usage errors end with a hint pointing to this command (``Hint: see `pup finetune help
train```) instead of to ``--help`` (:func:`~pup_finetune.cli._common.report_click_error`).
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from pup_finetune.cli._common import help_command, run_command
from pup_finetune.cli._presentation import SET_UP, CommandDoc, Safety
from pup_finetune.errors import UsageError

DOC = CommandDoc(
    path="help",
    short="Show the help for a command, as --help does.",
    description=(
        "Show the help for a command: the same text as `COMMAND --help`, or the top-level "
        "help with no command. Under a coding agent pup answers --help itself, so use this "
        "instead. Prints text, not JSON."
    ),
    examples=("pup finetune help train", "pup finetune help dataset pull"),
    safety=Safety(data="reads nothing", recipients="nowhere", without="prints the help"),
    panel=SET_UP,
)


def register(app: typer.Typer) -> None:
    """Add ``help`` to the root ``app``."""
    app.command(**DOC.typer())(show_help)


@run_command
def show_help(
    ctx: typer.Context,
    command: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="COMMAND",
            help="A command, such as `train` or `dataset pull`.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Print the help of ``command`` (the program's own with none), as ``--help`` does."""
    target = resolve(ctx.find_root(), command or [])
    # As Click's own --help callback prints it, at the same width.
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
    return help_command(command_path) or f"{command_path} --help"
