# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Options that take effect only with another one, and the check that refuses them alone.

A command keeps a table next to its options, mapping each dependent option's parameter
name to the parameter that enables it (``{"eval_judge_model": "evaluate"}``), and calls
:func:`require_enablers` with it first: an option given on the command line without its
enabler is a :class:`~pup_finetune.errors.UsageError` (exit 2), never silently ignored.
Each dependent option's help names its enabler (``With <enabler>:``, or ``Share of
<enabler> ...``), or the title of its option panel does (``Evaluate after training (with
--evaluate)``); a test holds the help to the table. :func:`require_alone` refuses anything
given with an option that does something else entirely (``dataset triage
--print-template``). :func:`renamed` reads a renamed option, accepting its old name (a hidden
option) for one release.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pup_finetune.cli._common import current_context
from pup_finetune.errors import UsageError
from pup_finetune.output import warn

__all__ = ["flag_name", "given", "renamed", "require_alone", "require_enablers"]

_NOT_GIVEN = frozenset({"DEFAULT", "DEFAULT_MAP"})
"""Click's parameter sources that mean the user did not pass the option."""


def _context() -> Any:
    ctx = current_context()
    if ctx is None:
        raise RuntimeError("flag checks run inside a command")
    return ctx


def _shown(param: Any) -> str:
    opts = [opt for opt in param.opts if opt.startswith("--")]
    if not opts:  # an argument: its metavar, e.g. DIR
        return str(param.human_readable_name)
    return str(max(opts, key=len))


def flag_name(command: Any, name: str) -> str:
    """The long option of ``command``'s parameter ``name``, e.g. ``--eval-judge-model`` (an
    argument's metavar, e.g. ``DIR``)."""
    for param in command.params:
        if param.name == name:
            return _shown(param)
    raise KeyError(name)


def given(ctx: Any, name: str) -> bool:
    """Whether the parameter ``name`` was passed (on the command line or in the
    environment), not left at its default."""
    source = ctx.get_parameter_source(name)
    return source is not None and source.name not in _NOT_GIVEN


def _present(ctx: Any, name: str) -> bool:
    value = ctx.params.get(name)
    if isinstance(value, bool):
        return value
    return value is not None or given(ctx, name)


def require_enablers(needs: Mapping[str, str]) -> None:
    """Raise :class:`UsageError` when an option of ``needs`` is given without its enabler.

    ``needs`` maps a parameter name to the parameter it needs; the enabler counts when it
    is on (a flag) or has a value, even a falsy one such as ``--max-spend-usd 0`` (whose own
    validation then reports it).
    """
    ctx = _context()
    missing: dict[str, list[str]] = {}
    for name, enabler in needs.items():
        if given(ctx, name) and not _present(ctx, enabler):
            missing.setdefault(enabler, []).append(flag_name(ctx.command, name))
    for enabler, flags in missing.items():
        needed = flag_name(ctx.command, enabler)
        names = " and ".join([", ".join(flags[:-1]), flags[-1]] if len(flags) > 1 else flags)
        verb = "need" if len(flags) > 1 else "needs"
        raise UsageError(
            f"{names} {verb} {needed}",
            hint=f"pass {needed} too, or leave {'them' if len(flags) > 1 else 'it'} out",
        )


def require_alone(name: str) -> None:
    """Raise :class:`UsageError` when the parameter ``name`` (e.g. ``--print-template``,
    which does something else entirely) is given together with any other parameter."""
    ctx = _context()
    if not given(ctx, name):
        return
    others = [
        _shown(param)
        for param in ctx.command.params
        if param.name != name and param.name is not None and given(ctx, param.name)
    ]
    if others:
        flag = flag_name(ctx.command, name)
        raise UsageError(
            f"{flag} takes nothing else, but {', '.join(others)} was given too",
            hint=f"run {flag} on its own",
        )


def renamed(name: str, old: str) -> Any:
    """The value of the parameter ``name``, or of ``old`` (a hidden option under the name
    ``name`` had before) when only that was given, with a warning that it was renamed.

    Raises :class:`UsageError` when both are given.
    """
    ctx = _context()
    if not given(ctx, old):
        return ctx.params[name]
    new_flag, old_flag = flag_name(ctx.command, name), flag_name(ctx.command, old)
    if given(ctx, name):
        raise UsageError(
            f"{old_flag} is the old name of {new_flag}, but both were given",
            hint=f"pass {new_flag} only",
        )
    warn(f"{old_flag} is now {new_flag}; the old name will be removed in a later release")
    return ctx.params[old]
