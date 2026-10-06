# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Shared plumbing for commands: the runtime, result emission and error-to-exit-code mapping."""

from __future__ import annotations

import functools
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

import typer

from pup_finetune.errors import FinetuneError
from pup_finetune.output import emit, emit_error
from pup_finetune.runtime import Runtime

PROG_NAME = "pup-finetune"

_CLI_FRAMEWORK_MODULES = frozenset({"click", "typer"})


def get_runtime() -> Runtime:
    """The runtime for the current command, read from the environment."""
    return Runtime.from_env()


def handle_exception(exc: BaseException, runtime: Runtime) -> int:
    """Report ``exc`` on stderr (and stdout for confirmation previews); return the exit code."""
    if isinstance(exc, FinetuneError):
        emit_error(exc, runtime=runtime)
        return exc.exit_code
    if runtime.debug:
        traceback.print_exception(exc, file=sys.stderr)
    detail = str(exc).strip().splitlines()
    message = f"{type(exc).__name__}: {detail[0]}" if detail else type(exc).__name__
    suffix = "" if runtime.debug else " (set PUP_FINETUNE_DEBUG=1 for a traceback)"
    print(f"Error: unexpected {message}{suffix}", file=sys.stderr)
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


def run_command[**P](fn: Callable[P, Mapping[str, Any] | None]) -> Callable[P, None]:
    """Decorate a command so its returned result is emitted and its errors map to exit codes.

    The command returns the result mapping (or ``None`` when it already emitted output) and
    raises :class:`FinetuneError` for expected failures. Apply below ``@app.command()``.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        runtime = get_runtime()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            if _is_cli_framework_exception(exc):
                # Typer's own signals and usage errors (e.g. BadParameter, exit 2) keep their
                # handling.
                raise
            raise typer.Exit(handle_exception(exc, runtime)) from exc
        if result is not None:
            emit(result, runtime=runtime)

    return wrapper


def run_app(app: typer.Typer, args: Sequence[str] | None = None) -> NoReturn:
    """Run ``app`` and exit, mapping exceptions that escape commands to exit codes.

    Typer/Click usage errors keep their own handling (a message on stderr, exit 2).
    """
    try:
        app(args=list(args) if args is not None else None, prog_name=PROG_NAME)
    except Exception as exc:
        sys.exit(handle_exception(exc, get_runtime()))
    sys.exit(0)
