# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Command output: one JSON result on stdout, everything else on stderr.

Under pup (``PUP_OUTPUT`` set), the result is piped through ``pup format`` so pup applies the
user's output format, ``--jq`` filter and envelope, when that pup version adds one (whose
metadata names the command and the result's ``next_command``); a table or CSV of a
list-shaped command shows its rows (:func:`table_filter`). Standalone, it is printed as JSON
(:func:`to_json`). Either way its measurements (losses, rates, times, spend estimates) keep 4
significant digits and values the user set are printed exactly
(:func:`pup_finetune.display.rounded_floats`); the files commands write keep theirs.

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
from typing import Any, Final

from rich.console import Console
from rich.text import Text

from pup_finetune.datadog.pup import FormatRunner, run_pup
from pup_finetune.display import rounded_floats
from pup_finetune.errors import ConfirmRequired, FinetuneError
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime

LINE_WIDTH: Final = 100
"""The widest line :func:`to_json` writes a list or object of plain values on, as one line."""


def to_json(result: Mapping[str, Any]) -> str:
    """Serialize a result the way it is printed when not running under pup.

    Keys keep the order the result was built in (most important first) and text keeps its
    characters (``21°C``, not ``21\\u00b0C``). Indented two spaces, except that a list or
    object holding only plain values (no list or object) is written on one line when that
    line fits in :data:`LINE_WIDTH`, so an epoch or a list of losses reads as one line.

    Raises ``ValueError`` for NaN or Infinity, which strict JSON cannot represent, so a
    non-finite number never reaches stdout as invalid JSON. Values JSON has no type for are
    written as their ``str``.
    """
    return _encode(result, 0)


def _scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


def _key(key: Any) -> str:
    """An object key as JSON writes it: a string, or a number, boolean or null as text."""
    if isinstance(key, str):
        return _scalar(key)
    if key is None or isinstance(key, bool | int | float):
        return _scalar(_scalar(key))
    raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")


def _encode(value: Any, depth: int, column: int = 0) -> str:
    """``value`` as JSON at nesting ``depth``, its first character at ``column``."""
    if isinstance(value, Mapping):
        prefixes = [f"{_key(key)}: " for key in value]
        children = list(value.values())
        opening, closing = "{", "}"
    elif isinstance(value, list | tuple):
        prefixes = ["" for _ in value]
        children = list(value)
        opening, closing = "[", "]"
    else:
        return _scalar(value)
    pairs = list(zip(prefixes, children, strict=True))
    if not pairs:
        return opening + closing
    if not any(isinstance(child, Mapping | list | tuple) for child in children):
        line = opening + ", ".join(prefix + _scalar(child) for prefix, child in pairs) + closing
        if column + len(line) <= LINE_WIDTH:
            return line
    inner = "  " * (depth + 1)
    body = ",\n".join(
        inner + prefix + _encode(child, depth + 1, len(inner) + len(prefix))
        for prefix, child in pairs
    )
    return f"{opening}\n{body}\n{'  ' * depth}{closing}"


def _first_line(text: str, limit: int = 200) -> str:
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line[:limit]


TABULAR_FORMATS: Final = frozenset({"table", "csv", "tsv"})
"""pup's output formats that render rows: a command's :func:`table view <format_args>`
applies to them."""


def format_args(
    result: Mapping[str, Any],
    command: str | None = None,
    *,
    jq: str | None = None,
) -> list[str]:
    """The ``pup format`` arguments for ``result``: its envelope's command and next action,
    and the ``--jq`` view to render.

    ``--command`` is ``command`` (e.g. ``finetune dataset show``) and ``--next-action`` the
    result's ``next_command`` when that is a non-empty string; ``--jq`` is ``jq`` when given.
    Values are attached with ``=`` so one that starts with ``-`` is never read as a flag.
    """
    args = ["format"]
    if command:
        args.append(f"--command={command}")
    next_command = result.get("next_command")
    if isinstance(next_command, str) and next_command.strip():
        args.append(f"--next-action={next_command}")
    if jq:
        args.append(f"--jq={jq}")
    return args


def table_filter(runtime: Runtime, view: str | None) -> str | None:
    """The ``--jq`` filter to pass ``pup format``: a command's table ``view`` (a jq
    expression picking its rows, e.g. ``.models``) when the user asked for a table, CSV or
    TSV and set no ``--jq`` filter of their own; else ``None``, so pup renders the whole
    result (and applies the user's filter)."""
    if view is None or runtime.jq_filter:
        return None
    output_format = (runtime.output_format or "").strip().lower()
    return view if output_format in TABULAR_FORMATS else None


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
    runner: FormatRunner | None = None,
    command: str | None = None,
    table_view: str | None = None,
) -> None:
    """Write exactly one result document to stdout.

    Every string in ``result`` (keys included) is passed through
    :func:`~pup_finetune.redaction.redact` first, so a known secret that slipped into a result
    never reaches stdout, and every measurement is rounded to
    :data:`~pup_finetune.display.SIGNIFICANT_DIGITS` significant digits
    (:func:`~pup_finetune.display.rounded_floats`; values the user set and the files on disk
    keep theirs). Under pup, ``command`` (the command as pup names it, e.g.
    ``finetune dataset show``) and the result's ``next_command`` go to ``pup format`` for its
    envelope's metadata (:func:`format_args`), and so does ``table_view``, the jq expression
    that picks the command's rows, when the user asked for a table (:func:`table_filter`).
    """
    redacted = rounded_floats(_redacted(result))
    document = to_json(redacted)
    args = format_args(redacted, command, jq=table_filter(runtime, table_view))
    # ``run_pup`` is looked up when called, so tests can serve pup format without a pup.
    run = runner if runner is not None else run_pup
    text = _format_with_pup(document, args, run) if runtime.under_pup else None
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


def emit_preview(
    err: ConfirmRequired,
    *,
    runtime: Runtime,
    runner: FormatRunner | None = None,
    command: str | None = None,
) -> None:
    """Write a :class:`ConfirmRequired`'s preview as the stdout result, so agents can parse
    what would happen and the command that runs it (``next_command`` ``None`` when its
    ``--confirm`` would be refused). Written before anything else is built from it, so the
    preview reaches stdout whatever happens to its card."""
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


def echo_preview(err: ConfirmRequired, *, runtime: Runtime, card: Sequence[str] = ()) -> None:
    """What stderr gets after a preview (:func:`emit_preview`): ``card`` (the preview's
    card, see :func:`pup_finetune.cli._summary.preview_card`; without one, the error's
    message), the preview's warnings (:func:`echo_warnings`), its hint and, last, the
    ``Next`` line. ``PUP_FINETUNE_QUIET`` keeps only the card's first line. The card is the
    same on a terminal and off one, and never colored. Every line is redacted."""
    message = str(err)
    lines = list(card) or [message[:1].upper() + message[1:]]
    for line in lines[:1] if runtime.quiet else lines:
        print(redact(line), file=sys.stderr)
    echo_warnings({"preview": err.preview})
    if err.hint:
        print(redact(f"Hint: {err.hint}"), file=sys.stderr)
    if err.rerun is not None:
        print(redact(f"Next: {err.rerun}"), file=sys.stderr)
    sys.stderr.flush()


def emit_error(
    err: FinetuneError,
    *,
    runtime: Runtime,
    runner: FormatRunner | None = None,
    command: str | None = None,
    card: Sequence[str] = (),
) -> None:
    """Report an expected failure: ``Error [<error_code>]: <message>`` and its hint on stderr,
    then ``warning: <note>`` for each of its notes (e.g. a resource that may remain).

    A :class:`ConfirmRequired` emits its preview as the stdout result (:func:`emit_preview`)
    and then ``card`` and the rest on stderr (:func:`echo_preview`).
    Every line, and every string in the preview (through :func:`emit`), is passed through
    :func:`~pup_finetune.redaction.redact`, so a known secret never reaches the terminal.
    """
    if isinstance(err, ConfirmRequired):
        emit_preview(err, runtime=runtime, runner=runner, command=command)
        echo_preview(err, runtime=runtime, card=card)
    else:
        error_line(str(err), err.error_code, err.hint, getattr(err, "__notes__", ()))


def error_line(message: str, code: str, hint: str | None = None, notes: Sequence[str] = ()) -> None:
    """Print ``Error [code]: message``, with a ``hint`` ``Hint: hint``, then ``warning: note``
    for each of ``notes`` on stderr, redacted.

    The one format of every error: ours, Click's usage errors and unexpected exceptions.
    """
    print(redact(f"Error [{code}]: {message}"), file=sys.stderr)
    if hint:
        print(redact(f"Hint: {hint}"), file=sys.stderr)
    sys.stderr.flush()
    for note in notes:
        warn(redact(note))


WARNING_PATHS: tuple[tuple[str, ...], ...] = (
    (),
    ("summary",),
    ("preview",),
    ("preview", "evaluation"),
    ("preview", "publication"),
    ("preview", "remote"),
    ("evaluation",),
    ("evaluation", "publication"),
    ("publication",),
)
"""The mappings, by their keys from the result, whose ``warnings`` are pup-finetune's own:
the result's, a dataset summary's, a preview's (and its remote job's), an evaluation's and its
publication's. Nothing else is searched, so a ``warnings`` key inside trace-derived data (a
sample record, a row's tool schema) is never printed as one of ours."""


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
    """Print ``note: <text>`` on stderr for every note in ``result`` (:func:`result_notes`),
    then ``warning: <text>`` for every warning in it (:func:`result_warnings`), redacted.
    Agents still read them in the JSON; this is so people at a terminal see them too."""
    for message in result_notes(result):
        note(redact(message))
    for message in result_warnings(result):
        warn(redact(message))


def result_notes(result: Any) -> list[str]:
    """The notes of ``result``, each once, in order: the strings in its top-level ``notes``.

    A note says what a command did besides its work (``created runs/.gitignore``); unlike a
    warning, it asks nothing of anyone."""
    items = result.get("notes") if isinstance(result, Mapping) else None
    if not isinstance(items, list | tuple):
        return []
    return list(dict.fromkeys(text for text in items if isinstance(text, str) and text))


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
    """Write one progress line to stderr, redacted; off a terminal it starts with
    :func:`timestamp`."""
    console = stderr_console()
    text = redact(message)
    console.print(text if console.is_terminal else f"{timestamp()} {text}", markup=False)


def note(message: str) -> None:
    """Write one ``note: <text>`` line to stderr: something done in passing, not a problem."""
    stderr_console().print(f"note: {message}", markup=False)


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


def summary(lines: Sequence[str], line: str, *, full: bool = False) -> None:
    """Print a command's closing summary on stderr, after its result: ``lines`` on a
    terminal, else the one plain ``line`` (agents pay for every line), or ``lines`` either
    way when ``full`` (``doctor``'s checklist). Redacted."""
    console = stderr_console()
    if full or console.is_terminal:
        for text in lines:
            console.print(redact(text), markup=False)
    else:
        console.print(redact(line), markup=False)
