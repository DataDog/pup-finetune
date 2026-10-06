# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Command output: one JSON result on stdout, everything else on stderr.

Under pup (``PUP_OUTPUT`` set), the result is piped through ``pup format`` so pup applies the
user's output format, ``--jq`` filter and agent envelope. Standalone, it is printed as JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.text import Text

from pup_finetune.datadog.pup import FormatRunner, run_pup
from pup_finetune.errors import ConfirmRequired, FinetuneError
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime


def to_json(result: Mapping[str, Any]) -> str:
    """Serialize a result the way it is printed when not running under pup.

    Raises ``ValueError`` for NaN or Infinity, which strict JSON cannot represent, so a
    non-finite number never reaches stdout as invalid JSON.
    """
    return json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False)


def _first_line(text: str, limit: int = 200) -> str:
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line[:limit]


def _format_with_pup(document: str, runner: FormatRunner) -> str | None:
    """Return pup's rendering of ``document``, or ``None`` (with one stderr note) on failure."""
    try:
        completed = runner(["format"], document)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # ValueError covers decoding failures (UnicodeDecodeError) from custom runners.
        reason = str(exc) or type(exc).__name__
    else:
        if completed.returncode == 0:
            return completed.stdout
        detail = _first_line(completed.stderr)
        reason = f"exit {completed.returncode}" + (f": {detail}" if detail else "")
    print(f"note: pup format failed ({reason}); printing plain JSON", file=sys.stderr)
    return None


def emit(
    result: Mapping[str, Any],
    *,
    runtime: Runtime,
    runner: FormatRunner = run_pup,
) -> None:
    """Write exactly one result document to stdout."""
    document = to_json(result)
    text = _format_with_pup(document, runner) if runtime.under_pup else None
    if text is None:
        text = document
    if not text.endswith("\n"):
        text += "\n"
    sys.stdout.write(text)
    sys.stdout.flush()


def _redacted(value: Any) -> Any:
    """``value`` with every string inside it (keys included) passed through ``redact``."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {_redacted(key): _redacted(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_redacted(item) for item in value]
    return value


def emit_error(
    err: FinetuneError,
    *,
    runtime: Runtime,
    runner: FormatRunner = run_pup,
) -> None:
    """Report an expected failure: a concise message on stderr.

    A :class:`ConfirmRequired` also emits its preview as the stdout result, so agents can
    parse what would happen and the command that runs it. Every line, and every string in
    the preview, is passed through :func:`~pup_finetune.redaction.redact`, so a known secret
    never reaches the terminal.
    """
    if isinstance(err, ConfirmRequired):
        print(redact(f"Confirmation required: {err}"), file=sys.stderr)
        print(redact(f"Next: {err.rerun}"), file=sys.stderr)
    else:
        print(redact(f"Error: {err}"), file=sys.stderr)
    if err.hint:
        print(redact(f"Hint: {err.hint}"), file=sys.stderr)
    sys.stderr.flush()
    if isinstance(err, ConfirmRequired):
        emit(
            {
                "status": "confirmation_required",
                "preview": _redacted(err.preview),
                "next_command": redact(err.rerun),
            },
            runtime=runtime,
            runner=runner,
        )


def _stderr_is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def stderr_console() -> Console:
    """A rich console bound to stderr; output is plain when stderr is not a terminal.

    A non-terminal stderr never gets color or cursor control codes, even when the environment
    asks for them (``FORCE_COLOR``, ``TTY_COMPATIBLE``): agents and CI read these lines.
    """
    return Console(
        stderr=True,
        highlight=False,
        emoji=False,
        soft_wrap=True,
        force_terminal=None if _stderr_is_tty() else False,
    )


def info(message: str) -> None:
    """Write one progress line to stderr."""
    stderr_console().print(message, markup=False)


def warn(message: str) -> None:
    """Write one warning line to stderr."""
    console = stderr_console()
    if console.is_terminal:
        console.print(f"warning: {message}", style="yellow", markup=False)
    else:
        console.print(f"warning: {message}", markup=False)


@contextmanager
def status(message: str) -> Iterator[None]:
    """Show a spinner on a terminal, or print ``message`` once as a plain line otherwise."""
    console = stderr_console()
    if console.is_terminal:
        with console.status(Text(message)):
            yield
    else:
        console.print(message, markup=False)
        yield
