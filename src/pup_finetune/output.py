# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Command output: one JSON result on stdout, everything else on stderr.

Under pup (``PUP_OUTPUT`` set), the result is piped through ``pup format`` so pup applies the
user's output format, ``--jq`` filter and agent envelope (whose metadata names the command and
the result's ``next_command``). Standalone, it is printed as JSON.

After the result, stderr gets every warning it holds (:func:`echo_warnings`) and the closing
summary (:func:`summary`): what people at a terminal read last. Progress lines
(:func:`info`) start with the time of day (``HH:MM:SS``) when stderr is not a terminal, as in
a detached run's ``worker.log``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
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


def format_args(result: Mapping[str, Any], command: str | None = None) -> list[str]:
    """The ``pup format`` arguments for ``result``: its envelope's command and next action.

    ``--command`` is ``command`` (e.g. ``finetune dataset show``) and ``--next-action`` the
    result's ``next_command`` when that is a non-empty string; values are attached with ``=``
    so one that starts with ``-`` is never read as a flag.
    """
    args = ["format"]
    if command:
        args.append(f"--command={command}")
    next_command = result.get("next_command")
    if isinstance(next_command, str) and next_command.strip():
        args.append(f"--next-action={next_command}")
    return args


def _format_with_pup(document: str, args: list[str], runner: FormatRunner) -> str | None:
    """Return pup's rendering of ``document``, or ``None`` (with one stderr note) on failure."""
    try:
        completed = runner(args, document)
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
    command: str | None = None,
) -> None:
    """Write exactly one result document to stdout.

    Every string in ``result`` (keys included) is passed through
    :func:`~pup_finetune.redaction.redact` first, so a known secret that slipped into a result
    never reaches stdout. Under pup, ``command`` (the command as pup names it, e.g.
    ``finetune dataset show``) and the result's ``next_command`` go to ``pup format`` for its
    envelope's metadata (:func:`format_args`).
    """
    redacted = _redacted(result)
    document = to_json(redacted)
    text = (
        _format_with_pup(document, format_args(redacted, command), runner)
        if runtime.under_pup
        else None
    )
    if text is None:
        text = document
    if not text.endswith("\n"):
        text += "\n"
    sys.stdout.write(text)
    sys.stdout.flush()


def _redacted(value: Any) -> Any:
    """``value`` with every string inside it (keys included) passed through ``redact``.

    Values JSON cannot represent natively (paths, dates, arbitrary objects) are
    rendered with ``str`` first, as :func:`to_json` would, so their text is redacted too.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {_redacted(key): _redacted(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_redacted(item) for item in value]
    return redact(str(value))


def emit_error(
    err: FinetuneError,
    *,
    runtime: Runtime,
    runner: FormatRunner = run_pup,
    command: str | None = None,
) -> None:
    """Report an expected failure: a concise message on stderr.

    A :class:`ConfirmRequired` first emits its preview as the stdout result, so agents can
    parse what would happen and the command that runs it (``next_command`` ``None`` when its
    ``--confirm`` would be refused); then stderr gets the preview's warnings
    (:func:`echo_warnings`), its hint and, last, the ``Confirmation required`` and ``Next``
    lines.
    Every line, and every string in the preview (through :func:`emit`), is passed through
    :func:`~pup_finetune.redaction.redact`, so a known secret never reaches the terminal.
    """
    if isinstance(err, ConfirmRequired):
        emit(
            {
                "status": "confirmation_required",
                "preview": err.preview,
                "next_command": err.rerun,
            },
            runtime=runtime,
            runner=runner,
            command=command,
        )
        echo_warnings({"preview": err.preview})
        if err.hint:
            print(redact(f"Hint: {err.hint}"), file=sys.stderr)
        print(redact(f"Confirmation required: {err}"), file=sys.stderr)
        if err.rerun is not None:
            print(redact(f"Next: {err.rerun}"), file=sys.stderr)
    else:
        print(redact(f"Error: {err}"), file=sys.stderr)
        if err.hint:
            print(redact(f"Hint: {err.hint}"), file=sys.stderr)
    sys.stderr.flush()


WARNING_PATHS: tuple[tuple[str, ...], ...] = (
    (),
    ("summary",),
    ("preview",),
    ("preview", "evaluation"),
    ("preview", "publication"),
    ("evaluation",),
    ("evaluation", "publication"),
    ("publication",),
)
"""The mappings, by their keys from the result, whose ``warnings`` are pup-finetune's own:
the result's, a dataset summary's, a preview's, an evaluation's and its publication's.
Nothing else is searched, so a ``warnings`` key inside trace-derived data (a sample record,
a row's tool schema) is never printed as one of ours."""


def result_warnings(result: Any) -> list[str]:
    """Every warning ``result`` holds, each once, in order: the strings in the ``warnings``
    of each mapping :data:`WARNING_PATHS` names (missing ones skipped)."""
    found: list[str] = []
    for path in WARNING_PATHS:
        value: Any = result
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        items = value.get("warnings") if isinstance(value, Mapping) else None
        if isinstance(items, list | tuple):
            found.extend(text for text in items if isinstance(text, str) and text)
    return list(dict.fromkeys(found))


def echo_warnings(result: Any) -> None:
    """Print ``warning: <text>`` on stderr for every warning in ``result``
    (:func:`result_warnings`), redacted. Agents still read them in the JSON; this is so people
    at a terminal see them too."""
    for message in result_warnings(result):
        warn(redact(message))


def _stderr_is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


_now: Callable[[], datetime] = datetime.now
"""The local time of day progress lines are stamped with; tests replace it."""


def timestamp() -> str:
    """The local time of day, ``HH:MM:SS``, that starts a progress line off a terminal."""
    return _now().strftime("%H:%M:%S")


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


def stderr_is_terminal() -> bool:
    """Whether stderr is a terminal (as :func:`stderr_console` decides): people are reading,
    rather than an agent, a pipe or a log file."""
    return stderr_console().is_terminal


def info(message: str) -> None:
    """Write one progress line to stderr; off a terminal it starts with :func:`timestamp`."""
    console = stderr_console()
    line = message if console.is_terminal else f"{timestamp()} {message}"
    console.print(line, markup=False)


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
        console.print(f"{timestamp()} {message}", markup=False)
        yield


def summary(lines: Sequence[str], line: str) -> None:
    """Print a command's closing summary on stderr, after its result: ``lines`` on a
    terminal, else the one plain ``line`` (agents pay for every line). Redacted."""
    console = stderr_console()
    if console.is_terminal:
        for text in lines:
            console.print(redact(text), markup=False)
    else:
        console.print(redact(line), markup=False)
