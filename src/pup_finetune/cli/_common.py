# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Shared plumbing for commands: the runtime, result emission and error-to-exit-code mapping."""

from __future__ import annotations

import functools
import importlib
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import typer

from pup_finetune import output
from pup_finetune.cli._summary import summarize
from pup_finetune.errors import FinetuneError
from pup_finetune.output import echo_warnings, emit, emit_error, warn
from pup_finetune.redaction import install_log_redaction, redact
from pup_finetune.runtime import Runtime

PROG_NAME = "pup-finetune"
"""The program name standalone: the console script's name."""
PUP_EXTENSION = "finetune"
"""The name pup runs us under, as ``pup finetune``."""
PUP_PROG_NAME = f"pup {PUP_EXTENSION}"
"""The program name under pup, so usage lines and hints name a command that works there."""

_CLI_FRAMEWORK_MODULES = frozenset({"click", "typer"})


@dataclass(frozen=True, slots=True)
class CommandResult:
    """A result to emit followed by a non-zero exit, e.g. a report whose checks failed.

    Commands that succeed return a plain mapping; return this when the result itself is the
    useful output but the exit code must still signal failure.
    """

    data: Mapping[str, Any]
    exit_code: int = 0


def get_runtime() -> Runtime:
    """The runtime for the current command, read from the environment."""
    return Runtime.from_env()


def prog_name(runtime: Runtime | None = None) -> str:
    """``pup finetune`` under pup (``PUP_OUTPUT`` set), else ``pup-finetune``."""
    return PUP_PROG_NAME if (runtime or get_runtime()).under_pup else PROG_NAME


def _click_context_stack() -> Any:
    """The ``globals`` module of the Click that Typer's own ``Context`` derives from.

    Found from the class rather than imported, since newer Typer releases vendor Click under
    ``typer._click`` and older ones use Click itself.
    """
    base = next(
        cls
        for cls in typer.Context.__mro__
        if cls.__name__ == "Context" and cls.__module__.endswith(".core")
    )
    return importlib.import_module(base.__module__.rpartition(".")[0] + ".globals")


def current_context() -> Any | None:
    """The Click context of the command running now, or ``None`` outside one."""
    return _click_context_stack().get_current_context(silent=True)


def invoked_command_path() -> str | None:
    """The words naming the command running now below the program, e.g. ``dataset show``.

    ``None`` outside a command (or for the program itself).
    """
    ctx = current_context()
    words: list[str] = []
    while ctx is not None and ctx.parent is not None:
        words.append(str(ctx.info_name))
        ctx = ctx.parent
    return " ".join(reversed(words)) or None


def _pup_command(path: str | None) -> str | None:
    """The command at ``path`` as pup's envelope names it, e.g. ``finetune dataset show``."""
    return f"{PUP_EXTENSION} {path}" if path else None


def echo_outcome(result: Mapping[str, Any], runtime: Runtime, path: str | None) -> None:
    """What stderr gets after ``result`` was written to stdout: every warning in it, then
    the closing summary of the command at ``path`` (see :mod:`pup_finetune.cli._summary`),
    unless ``PUP_FINETUNE_QUIET`` is on."""
    echo_warnings(result)
    if runtime.quiet:
        return
    found = summarize(path, result)
    if found is not None:
        output.summary(found.lines, found.line)
    sys.stderr.flush()


def handle_exception(exc: BaseException, runtime: Runtime, *, command: str | None = None) -> int:
    """Report ``exc`` on stderr (and stdout for confirmation previews); return the exit code.

    ``command`` is the command as pup's envelope names it (see :func:`emit`), if known.
    """
    if isinstance(exc, FinetuneError):
        emit_error(exc, runtime=runtime, command=command)
        return exc.exit_code
    if runtime.debug:
        sys.stderr.write(redact("".join(traceback.format_exception(exc))))
    detail = str(exc).strip().splitlines()
    message = f"{type(exc).__name__}: {detail[0]}" if detail else type(exc).__name__
    suffix = "" if runtime.debug else " (set PUP_FINETUNE_DEBUG=1 for a traceback)"
    print(redact(f"Error: unexpected {message}{suffix}"), file=sys.stderr)
    sys.stderr.flush()
    return 1


def _is_cli_framework_exception(exc: BaseException) -> bool:
    """Whether ``exc`` is one of Typer's or Click's own signals or usage errors.

    Checked by defining module rather than by class so it holds across Typer releases, whose
    exception classes differ (some re-export Click's, newer ones vendor Click under ``typer``).
    """
    return any(
        cls.__module__.partition(".")[0] in _CLI_FRAMEWORK_MODULES for cls in type(exc).__mro__
    )


def run_command[**P](
    fn: Callable[P, Mapping[str, Any] | CommandResult | None],
) -> Callable[P, None]:
    """Decorate a command so its returned result is emitted and its errors map to exit codes.

    The command returns the result mapping (or ``None`` when it already emitted output) and
    raises :class:`FinetuneError` for expected failures. A :class:`CommandResult` is emitted
    and then exits with its ``exit_code``. What the environment held that was read in the
    safe direction (:attr:`Runtime.warnings`) is printed first, once. After the result, its
    warnings and the command's closing summary go to stderr (:func:`echo_outcome`). Apply
    below ``@app.command()``.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        runtime = get_runtime()
        path = invoked_command_path()
        command = _pup_command(path)
        for message in runtime.warnings:
            warn(message)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            if _is_cli_framework_exception(exc):
                # Typer's own signals and usage errors (e.g. BadParameter, exit 2) keep their
                # handling.
                raise
            raise typer.Exit(handle_exception(exc, runtime, command=command)) from exc
        if isinstance(result, CommandResult):
            emit(result.data, runtime=runtime, command=command)
            echo_outcome(result.data, runtime, path)
            if result.exit_code:
                raise typer.Exit(result.exit_code)
        elif result is not None:
            emit(result, runtime=runtime, command=command)
            echo_outcome(result, runtime, path)

    return wrapper


def run_app(app: typer.Typer, args: Sequence[str] | None = None) -> NoReturn:
    """Run ``app`` and exit, mapping exceptions that escape commands to exit codes.

    Typer/Click usage errors keep their own handling (a message on stderr, exit 2). The
    program is named :func:`prog_name`, so usage lines say ``pup finetune`` under pup. Every
    log record created from here on is redacted of known secrets, whichever handler emits it.
    """
    install_log_redaction()
    try:
        app(args=list(args) if args is not None else None, prog_name=prog_name())
    except Exception as exc:
        sys.exit(handle_exception(exc, get_runtime()))
    sys.exit(0)
