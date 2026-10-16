# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""How commands present themselves in ``--help``: order, panels, examples and safety.

Every visible command is described once, by a :class:`CommandDoc`: a short help that names
its gate, a description of at most three sentences that ``--help`` shows, more text after
``\\f`` (Typer cuts there; ``scripts/gen_command_docs.py`` keeps it for
``docs/commands.md``), up to four examples (two at least for a command that takes anything)
and a :class:`Safety` record. The record is the
one source of what the command costs, where its data goes and what ``PUP_READ_ONLY`` does to
it: it renders as the footer of the command's help and as the safety table of
``docs/commands.md``.

:class:`Command` and :class:`Group` are the Click classes every command and group is built
with: commands are listed in workflow order (:data:`COMMAND_ORDER`), required arguments read
``DIR`` in the usage line rather than ``{DIR}``, the option type column has no range (the help
text says what matters), and ``--help`` joins an option panel of a command whose options are
grouped (``--confirm``'s, else the last), so no panel holds it alone. A usage error names
every missing required parameter at once and prints as ``Error [usage]: ...`` with a
``Hint:`` naming the ``help`` command, never in Typer's box.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Literal

import typer
from rich.markup import escape
from typer import rich_utils
from typer.core import TyperCommand, TyperGroup

from pup_finetune.cli._common import click_module, plain_click_errors
from pup_finetune.display import joined

__all__ = [
    "BUILD_A_DATASET",
    "COMMAND_ORDER",
    "DOCS",
    "EVALUATE_AND_SHIP",
    "FLOW",
    "SET_UP",
    "TRAIN",
    "Command",
    "CommandDoc",
    "Group",
    "Safety",
    "default_note",
    "describe",
    "group",
    "local",
    "wrap_example",
]

COMMAND_ORDER: Final[tuple[str, ...]] = (
    "doctor",
    "auth login",
    "auth status",
    "auth logout",
    "models list",
    "skill show",
    "skill install",
    "help",
    "dataset pull",
    "dataset import",
    "dataset show",
    "dataset sample",
    "dataset label",
    "dataset triage",
    "dataset push",
    "dataset fetch",
    "benchmark",
    "train",
    "status",
    "runs",
    "resume",
    "cancel",
    "evaluate",
    "deploy",
    "undeploy",
)
"""Every visible command in workflow order: how ``--help`` lists them (a group where its
first command is) and how ``docs/commands.md`` orders them. A command not listed follows."""

SET_UP: Final = "Set up"
BUILD_A_DATASET: Final = "Build a dataset"
TRAIN: Final = "Train"
EVALUATE_AND_SHIP: Final = "Evaluate and ship"
"""The panels of the top-level help."""

FLOW: Final = "dataset pull -> review -> benchmark -> train -> evaluate -> deploy"
"""The workflow, in one line under the top-level description."""

EXAMPLE_WIDTH: Final = 74
"""How wide an example line may be before it continues on the next one (with `` \\``), so
an example fits the 80 columns help renders at off a terminal."""


def default_note(value: object) -> str:
    """``[default: value]`` styled as Typer styles a default, for an option whose parameter
    default is ``None`` (so the command can tell "not given" apart) or is described in words.

    ``scripts/gen_command_docs.py`` moves it into the Default column of the docs.
    """
    return f"[dim]\\[default: {escape(str(value))}][/dim]"


def describe(shown: str, more: str = "") -> str:
    """A command's help: ``shown`` (at most three sentences, what ``--help`` prints), then
    ``more`` after ``\\f``, which only the generated docs show."""
    return f"{shown}\f\n\n{more}" if more else shown


ReadOnly = Literal["allowed", "confirm_refused"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Safety:
    """What a command costs, where its data goes and how ``PUP_READ_ONLY`` treats it.

    ``data`` is a phrase for the footer (``reads Datadog``); ``recipients`` names where data
    goes for the docs table. ``without`` is what the command does without ``--confirm`` (or at
    all, for a command without one) and ``confirm`` what ``--confirm`` does, ``None`` when the
    command has no ``--confirm``. ``paid`` is whether ``--confirm`` runs something billed.
    ``read_only`` is ``allowed`` (nothing it does is refused with ``PUP_READ_ONLY``) or
    ``confirm_refused`` (its preview runs, its ``--confirm`` is refused); ``offline_allowed``
    says its ``--confirm`` is still allowed when nothing leaves the machine (the offline
    fake provider and judge).
    """

    data: str
    recipients: str
    without: str
    confirm: str | None = None
    paid: bool = False
    read_only: ReadOnly = "allowed"
    offline_allowed: bool = False

    def __post_init__(self) -> None:
        if self.paid and self.confirm is None:
            raise ValueError("a paid command takes --confirm")
        if self.read_only == "confirm_refused" and self.confirm is None:
            raise ValueError("only --confirm can be refused under PUP_READ_ONLY")
        if self.offline_allowed and self.read_only != "confirm_refused":
            raise ValueError("offline_allowed qualifies a refused --confirm")

    def footer(self) -> str:
        """One line, e.g. ``Free · reads Datadog · PUP_READ_ONLY: allowed``."""
        cost = "Paid with --confirm" if self.paid else "Free"
        refused = self.read_only == "confirm_refused"
        read_only = "preview only" if refused else "allowed"
        if self.offline_allowed:
            read_only += " unless offline"
        return " · ".join([cost, self.data, f"PUP_READ_ONLY: {read_only}"])

    def table_row(self) -> tuple[str, str, str, str]:
        """The cells after the command in the docs table: without and with ``--confirm``,
        where data goes, and ``PUP_READ_ONLY``."""
        refused = self.read_only == "confirm_refused"
        read_only = "preview allowed; `--confirm` refused" if refused else "allowed"
        if self.offline_allowed:
            read_only += ", except a run that stays offline (the fake provider and judge)"
        with_confirm = self.confirm if self.confirm is not None else "no `--confirm`"
        return self.without, with_confirm, self.recipients, read_only


def local(without: str) -> Safety:
    """The record of a command that only reads and writes local files and does ``without``."""
    return Safety(data="local files only", recipients="nowhere: local files", without=without)


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandDoc:
    """How one command presents itself (see the module docstring)."""

    path: str
    """The command's words below the program, e.g. ``dataset pull``."""
    short: str
    """A dozen words or fewer that name the gate, for command lists."""
    description: str
    """At most three sentences, what ``--help`` prints (or, written with :func:`describe`,
    those and :attr:`more`)."""
    examples: tuple[str, ...]
    """Commands that run as written: 2 to 4, or 1 for a command that takes nothing."""
    safety: Safety
    more: str = ""
    """The rest of the description, only in the generated docs."""
    panel: str | None = None
    """The top-level help panel, for a top-level command."""

    def __post_init__(self) -> None:
        if "\f" in self.description:  # written with describe(shown, more)
            shown, _, more = self.description.partition("\f")
            object.__setattr__(self, "description", shown.strip())
            object.__setattr__(self, "more", " ".join([more.strip(), self.more]).strip())
        if len(self.short.split()) > 12:
            raise ValueError(f"{self.path}: the short help has more than 12 words")
        if not 1 <= len(self.examples) <= 4:
            # Two or more for a command that takes anything (a test checks it).
            raise ValueError(f"{self.path}: give 1 to 4 examples")
        if self.path in DOCS:
            raise ValueError(f"{self.path} is described twice")
        DOCS[self.path] = self

    @property
    def name(self) -> str:
        return self.path.rsplit(" ", 1)[-1]

    def help(self) -> str:
        return describe(self.description, self.more)

    def epilog(self) -> str:
        """The examples, then the safety footer."""
        lines = ["Examples:"]
        for example in self.examples:
            for line in wrap_example(example):
                # escape() would double a trailing backslash, so the continuation is added after.
                body, continued = (line[:-2], " \\") if line.endswith(" \\") else (line, "")
                lines.append(f"  {escape(body)}{continued}")
        return "\n".join([*lines, "", escape(self.safety.footer())])

    def typer(self) -> dict[str, Any]:
        """The keyword arguments for ``app.command`` (or ``@app.command``)."""
        options: dict[str, Any] = {
            "name": self.name,
            "cls": Command,
            "short_help": self.short,
            "help": self.help(),
            "epilog": self.epilog(),
        }
        if self.panel is not None:
            options["rich_help_panel"] = self.panel
        return options


DOCS: dict[str, CommandDoc] = {}
"""Every :class:`CommandDoc`, by command path."""


def wrap_example(example: str, width: int = EXAMPLE_WIDTH) -> list[str]:
    """``example`` in lines of at most ``width`` characters (an option stays with its value),
    each but the last ending in `` \\`` and the rest indented, as a shell continues them."""
    units: list[str] = []
    for word in shlex.split(example, posix=False):  # words as written, quotes kept
        pairs = units and units[-1].startswith("-") and " " not in units[-1]
        if pairs and not word.startswith("-"):
            units[-1] = f"{units[-1]} {word}"
        else:
            units.append(word)
    lines: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if current and len(candidate) + 2 > width:
            lines.append(current)
            current = f"    {unit}"
        else:
            current = candidate
    lines.append(current)
    return [f"{line} \\" for line in lines[:-1]] + [lines[-1]]


def _path(ctx: Any) -> tuple[str, ...]:
    """The command words of ``ctx`` below the program."""
    words: list[str] = []
    while ctx is not None and ctx.parent is not None:
        words.append(str(ctx.info_name))
        ctx = ctx.parent
    return tuple(reversed(words))


def order_key(path: Sequence[str]) -> int:
    """Where the command (or group) at ``path`` comes in :data:`COMMAND_ORDER`."""
    prefix = " ".join(path)
    for index, name in enumerate(COMMAND_ORDER):
        if name == prefix or name.startswith(f"{prefix} "):
            return index
    return len(COMMAND_ORDER)


_HELP_STYLE: Final[dict[str, Any]] = {
    # No "[x>=1]" after the metavar: the help text says what matters.
    "RANGE_STRING": "",
    # One space after each column instead of one on each side, so the columns Typer leaves
    # empty (short and negative names) cost a space each, not two, and more of the 80
    # columns go to the help text.
    "STYLE_OPTIONS_TABLE_PADDING": (0, 1, 0, 0),
}


@contextmanager
def _help_style() -> Iterator[None]:
    """Render help with :data:`_HELP_STYLE` in place of Typer's own settings."""
    previous = {name: getattr(rich_utils, name) for name in _HELP_STYLE}
    for name, value in _HELP_STYLE.items():
        setattr(rich_utils, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(rich_utils, name, value)


def _no_metavar(*args: Any, **kwargs: Any) -> str:
    del args, kwargs
    return ""


@contextmanager
def _untyped_arguments(params: Sequence[Any]) -> Iterator[None]:
    """Render the Arguments panel without Typer's ``<path>`` and ``<str>`` type column: an
    argument's metavar (``DIR``) and help say what it takes.

    Typer fills that column from the type's ``get_metavar``, so it is shadowed on each
    argument's type while help renders, and restored after (a type may be shared).
    """
    kinds = {id(param.type): param.type for param in params if param.param_type_name == "argument"}
    for kind in kinds.values():
        kind.get_metavar = _no_metavar
    try:
        yield
    finally:
        for kind in kinds.values():
            del kind.get_metavar


def _param_name(param: Any) -> str:
    """How a missing parameter is named: an option by its long name, an argument by its
    metavar (``DIR``)."""
    if param.param_type_name == "option":
        return str(next((name for name in param.opts if name.startswith("--")), param.opts[0]))
    return str(param.human_readable_name)


class Command(TyperCommand):
    """A command as every ``pup finetune`` command is built (see the module docstring)."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        """Parse as Click does, but name every missing required parameter at once
        (``train needs --dataset and --model``) rather than one per run."""
        given = list(args)  # Click's parser consumes the list it is given
        exceptions = click_module("exceptions")
        try:
            return super().parse_args(ctx, args)
        except exceptions.MissingParameter as exc:
            missing = self._missing(ctx, given) or [exc.param]
            names = joined([_param_name(param) for param in missing if param is not None])
            raise exceptions.UsageError(f"{' '.join(_path(ctx))} needs {names}", ctx=ctx) from exc

    def _missing(self, ctx: Any, given: list[str]) -> list[Any]:
        """The required parameters ``given`` leaves without a value, in help order."""
        try:
            opts, _, _ = self.make_parser(ctx).parse_args(args=list(given))
        except click_module("exceptions").ClickException:
            return []
        return [
            param
            for param in self.get_params(ctx)
            if param.required and param.value_is_missing(param.consume_value(ctx, opts)[0])
        ]

    def collect_usage_pieces(self, ctx: Any) -> list[str]:
        pieces: list[str] = super().collect_usage_pieces(ctx)
        return [
            piece[1:-1] if piece.startswith("{") and piece.endswith("}") else piece
            for piece in pieces
        ]

    def get_help_option(self, ctx: Any) -> Any:
        """``--help``, in the option panel of ``--confirm`` (else the last panel) when the
        options are grouped, so no panel holds it alone."""
        option = super().get_help_option(ctx)
        shown = [param for param in self.params if not getattr(param, "hidden", False)]
        panels = [panel for param in shown if (panel := getattr(param, "rich_help_panel", None))]
        if option is None or not panels:
            return option
        confirm = next((param for param in shown if param.name == "confirm"), None)
        option.rich_help_panel = getattr(confirm, "rich_help_panel", None) or panels[-1]
        return option

    def format_help(self, ctx: Any, formatter: Any) -> None:
        with _help_style(), _untyped_arguments(self.params):
            super().format_help(ctx, formatter)


class Group(TyperGroup):
    """A command group whose commands are listed in workflow order, and whose Click errors
    (a usage error in it or in any command below it) print as our own errors do
    (:func:`~pup_finetune.cli._common.plain_click_errors`)."""

    def make_context(
        self, info_name: str | None, args: list[str], parent: Any | None = None, **extra: Any
    ) -> Any:
        with plain_click_errors():
            return super().make_context(info_name, args, parent, **extra)

    def invoke(self, ctx: Any) -> Any:
        with plain_click_errors():
            return super().invoke(ctx)

    def list_commands(self, ctx: Any) -> list[str]:
        names: list[str] = super().list_commands(ctx)
        # Listed from another command's context (as tools walking the tree do), a group
        # knows its own name only.
        path = _path(ctx) if ctx.command is self or not self.name else (self.name,)
        ranked = sorted(enumerate(names), key=lambda item: (order_key((*path, item[1])), item[0]))
        return [name for _, name in ranked]


def group(name: str, *, short: str, description: str, **options: Any) -> typer.Typer:
    """A command group's Typer app, built with :class:`Group`."""
    return typer.Typer(
        name=name,
        cls=Group,
        short_help=short,
        help=description,
        no_args_is_help=True,
        **options,
    )
