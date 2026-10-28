# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Shared plumbing for commands: the runtime, result emission and error-to-exit-code mapping."""

from __future__ import annotations

import contextlib
import functools
import importlib
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

import typer

from pup_finetune import output
from pup_finetune.cli._summary import preview_card, summarize
from pup_finetune.cli._views import TABLE_VIEWS
from pup_finetune.errors import ConfirmRequired, FinetuneError, Interrupted, UsageError
from pup_finetune.output import (
    echo_preview,
    echo_warnings,
    emit,
    emit_error,
    emit_preview,
    error_line,
    warn,
)
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


def with_settings(
    preview: Mapping[str, Any], settings: Mapping[str, Any], defaults_used: Sequence[str]
) -> dict[str, Any]:
    """``preview`` with ``settings`` (the values that will be used) and ``defaults_used``
    (the ones no flag set) just before its ``data_destination`` (else at the end)."""
    fields = {"settings": dict(settings), "defaults_used": list(defaults_used)}
    shown: dict[str, Any] = {}
    for key, value in preview.items():
        if key == "data_destination":
            shown.update(fields)
        shown[key] = value
    return {**shown, **fields} if "settings" not in shown else shown


def get_runtime() -> Runtime:
    """The runtime for the current command, read from the environment."""
    return Runtime.from_env()


def prog_name(runtime: Runtime | None = None) -> str:
    """``pup finetune`` under pup (``PUP_OUTPUT`` set), else ``pup-finetune``."""
    return PUP_PROG_NAME if (runtime or get_runtime()).under_pup else PROG_NAME


def click_module(name: str) -> Any:
    """The module ``name`` (``globals``, ``exceptions``) of the Click that Typer's own
    ``Context`` derives from.

    Found from the class rather than imported, since newer Typer releases vendor Click under
    ``typer._click`` and older ones use Click itself.
    """
    base = next(
        cls
        for cls in typer.Context.__mro__
        if cls.__name__ == "Context" and cls.__module__.endswith(".core")
    )
    return importlib.import_module(f"{base.__module__.rpartition('.')[0]}.{name}")


def current_context() -> Any | None:
    """The Click context of the command running now, or ``None`` outside one."""
    return click_module("globals").get_current_context(silent=True)


def help_command(command_path: str) -> str | None:
    """The ``help`` command for ``command_path`` (e.g. ``pup finetune help train`` for
    ``pup finetune train``), which pup does not intercept as it does ``--help``.

    ``None`` when ``command_path`` is not under this program.
    """
    for prog in (PUP_PROG_NAME, PROG_NAME):
        if command_path == prog or command_path.startswith(f"{prog} "):
            return f"{prog} help{command_path.removeprefix(prog)}"
    return None


def report_click_error(exc: Any) -> int:
    """Report a Click or Typer error (a usage error, mostly) as our own errors read; return
    its exit code.

    Two plain lines, on a terminal or off one: ``Error [usage]: <Click's message>``, which
    keeps its suggestions (``Did you mean ...``, ``Possible options: ...``), and a hint
    naming the ``help`` command of the command it is about. A group run without a command
    already printed its help on stdout as the error was made, so nothing is added.
    """
    if type(exc).__name__ == "NoArgsIsHelpError":
        return int(exc.exit_code)
    ctx = getattr(exc, "ctx", None)
    _drop_hidden_possibilities(exc, ctx)
    hint = None
    if ctx is not None:
        command = help_command(ctx.command_path) or f"{ctx.command_path} --help"
        hint = f"see `{command}`"
    usage = isinstance(exc, click_module("exceptions").UsageError)
    error_line(
        exc.format_message(), UsageError.error_code if usage else FinetuneError.error_code, hint
    )
    return int(exc.exit_code)


def _drop_hidden_possibilities(exc: Any, ctx: Any) -> None:
    """Leave the hidden options (old names kept for compatibility) out of a
    ``No such option`` error's suggestions, so it never suggests a name ``--help`` hides."""
    possibilities = getattr(exc, "possibilities", None)
    command = getattr(ctx, "command", None)
    if not possibilities or command is None:
        return
    hidden = {
        name
        for param in getattr(command, "params", ())
        if getattr(param, "hidden", False)
        for name in (*param.opts, *param.secondary_opts)
    }
    exc.possibilities = [name for name in possibilities if name not in hidden]


@contextmanager
def plain_click_errors() -> Iterator[None]:
    """Report the Click errors raised inside with :func:`report_click_error`, then exit with
    their exit code (``typer.Exit``), so Typer never prints its boxed error."""
    try:
        yield
    except click_module("exceptions").ClickException as exc:
        raise typer.Exit(report_click_error(exc)) from exc


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
    unless ``PUP_FINETUNE_QUIET`` is on.

    Never raises: the result and its exit code are decided already, and formatting what
    stderr says about them must not change either (with ``PUP_FINETUNE_DEBUG=1`` a failure's
    traceback goes to stderr)."""
    try:
        echo_warnings(result)
        if runtime.quiet:
            return
        found = summarize(path, result)
        if found is not None:
            output.summary(found.lines, found.line, full=found.full)
        sys.stderr.flush()
    except Exception as exc:
        _debug_traceback(exc, runtime)


def _debug_traceback(exc: BaseException, runtime: Runtime) -> None:
    """With ``PUP_FINETUNE_DEBUG=1``, ``exc``'s redacted traceback on stderr; never raises."""
    if not runtime.debug:
        return
    # A debug aid that failed has nothing left to report to.
    with contextlib.suppress(Exception):
        sys.stderr.write(redact("".join(traceback.format_exception(exc))))


INTERRUPTED_MESSAGE = "stopped by Ctrl-C"
"""What a Ctrl-C that reaches the top of a command says (``Error [interrupted]``)."""

INTERRUPTED_HINT = "what finished is kept; check where it stands before running it again"


def interrupted_exit(
    interrupt: BaseException, runtime: Runtime, *, command: str | None = None
) -> int:
    """Report a Ctrl-C no command turned into a result of its own: ``Error [interrupted]``,
    exit 1, as :class:`~pup_finetune.errors.Interrupted` reads, with the interrupt's notes."""
    err = Interrupted(INTERRUPTED_MESSAGE, hint=INTERRUPTED_HINT, interrupt=interrupt)
    emit_error(err, runtime=runtime, command=command)
    return err.exit_code


def handle_exception(
    exc: BaseException, runtime: Runtime, *, command: str | None = None, path: str | None = None
) -> int:
    """Report ``exc`` on stderr (and stdout for confirmation previews); return the exit code.

    ``command`` is the command as pup's envelope names it (see :func:`emit`) and ``path`` its
    path below the program (``dataset push``), which picks a preview's card
    (:func:`~pup_finetune.cli._summary.preview_card`), if known.
    """
    if isinstance(exc, ConfirmRequired):
        # The preview first: it is the result, and nothing built from it (its card) can keep
        # it from stdout or change the exit code.
        emit_preview(exc, runtime=runtime, command=command)
        try:
            card = preview_card(path, exc.preview, refused=exc.refused)
        except Exception as card_error:
            _debug_traceback(card_error, runtime)
            card = []
        try:
            echo_preview(exc, runtime=runtime, card=card)
        except Exception as echo_error:
            _debug_traceback(echo_error, runtime)
        return exc.exit_code
    if isinstance(exc, KeyboardInterrupt):
        return interrupted_exit(exc, runtime, command=command)
    if isinstance(exc, FinetuneError):
        emit_error(exc, runtime=runtime, command=command)
        return exc.exit_code
    if runtime.debug:
        sys.stderr.write(redact("".join(traceback.format_exception(exc))))
    detail = str(exc).strip().splitlines()
    message = f"{type(exc).__name__}: {detail[0]}" if detail else type(exc).__name__
    suffix = "" if runtime.debug else " (set PUP_FINETUNE_DEBUG=1 for a traceback)"
    error_line(f"{message}{suffix}", FinetuneError.error_code)
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
    warnings and the command's closing summary go to stderr (:func:`echo_outcome`, which
    never changes the exit code). A Ctrl-C the command did not turn into a result of its
    own (``train`` and ``evaluate`` stop gracefully first) is ``Error [interrupted]``, exit
    1. Under
    pup, a table or CSV of a list-shaped command shows its rows
    (:data:`~pup_finetune.cli._views.TABLE_VIEWS`). Apply below ``@app.command()``.
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
        except KeyboardInterrupt as exc:
            raise typer.Exit(interrupted_exit(exc, runtime, command=command)) from exc
        except Exception as exc:
            if _is_cli_framework_exception(exc):
                # Typer's own signals and usage errors (e.g. BadParameter, exit 2) keep their
                # handling.
                raise
            raise typer.Exit(handle_exception(exc, runtime, command=command, path=path)) from exc
        view = TABLE_VIEWS.get(path or "")
        if isinstance(result, CommandResult):
            emit(result.data, runtime=runtime, command=command, table_view=view)
            echo_outcome(result.data, runtime, path)
            if result.exit_code:
                raise typer.Exit(result.exit_code)
        elif result is not None:
            emit(result, runtime=runtime, command=command, table_view=view)
            echo_outcome(result, runtime, path)

    return wrapper


def run_app(app: typer.Typer, args: Sequence[str] | None = None) -> NoReturn:
    """Run ``app`` and exit, mapping exceptions that escape commands to exit codes.

    Typer/Click usage errors are reported by the app's own group (exit 2, see
    :func:`plain_click_errors`). The program is named :func:`prog_name`, so usage lines say
    ``pup finetune`` under pup. Every log record created from here on is redacted of known
    secrets, whichever handler emits it. A Ctrl-C outside any command (while Click parses,
    say) is ``Error [interrupted]``, exit 1, as inside one.
    """
    install_log_redaction()
    try:
        app(args=list(args) if args is not None else None, prog_name=prog_name())
    except KeyboardInterrupt as exc:
        sys.exit(interrupted_exit(exc, get_runtime()))
    except Exception as exc:
        sys.exit(handle_exception(exc, get_runtime()))
    sys.exit(0)
