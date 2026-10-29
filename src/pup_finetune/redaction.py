# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Scrub known secret values from text before it reaches stderr, stdout or a log.

Kept free of heavy imports so the output and error paths can always use it. Credential
resolution registers every secret it hands out; :func:`redact` then replaces those values,
and anything shaped like ``Bearer <token>``, with ``***``. A value shorter than
:data:`MIN_SECRET_LENGTH` is never registered.
"""

from __future__ import annotations

import functools
import logging
import re
import threading
from typing import Any, Final, Protocol

MASK = "***"
MIN_SECRET_LENGTH: Final = 16
"""The shortest value :func:`register_secret` masks. No API key or Datadog secret is shorter,
and masking a shorter value would rewrite ordinary words in every result and command line."""

_BEARER = re.compile(r"(\bBearer\s+)([^\s\"']+)", re.IGNORECASE)
"""A ``Bearer`` token: everything after it up to whitespace or a quote."""
_VARIABLE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
"""A shell variable reference, such as a usage example's ``$FIREWORKS_API_KEY``: no token."""

_lock = threading.Lock()
_secrets: set[str] = set()


class _Secret(Protocol):
    """Anything that wraps a secret, e.g. ``pydantic.SecretStr``."""

    def get_secret_value(self) -> str: ...


def register_secret(value: str | _Secret) -> None:
    """Remember ``value`` so :func:`redact` masks it from now on (for this process), unless it
    is shorter than :data:`MIN_SECRET_LENGTH`."""
    text = value if isinstance(value, str) else value.get_secret_value()
    if len(text) >= MIN_SECRET_LENGTH:
        with _lock:
            _secrets.add(text)


def clear_secrets() -> None:
    """Forget every registered secret. For tests."""
    with _lock:
        _secrets.clear()


def redact(text: str) -> str:
    """``text`` with every registered secret and ``Bearer`` token replaced by ``***``."""
    with _lock:
        # Longest first, so a secret that contains another is masked whole.
        secrets = sorted(_secrets, key=len, reverse=True)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, MASK)
    return _BEARER.sub(_mask_bearer, text)


def _mask_bearer(match: re.Match[str]) -> str:
    return match[0] if _VARIABLE.fullmatch(match[2]) else f"{match[1]}{MASK}"


_STANDARD_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))
"""Attributes every record has; anything else came from ``extra=`` or a filter."""


def _formatted_message(record: logging.LogRecord) -> str:
    """``record``'s message, even when a malformed log call cannot be formatted."""
    try:
        return record.getMessage()
    except Exception:
        try:
            return f"{record.msg!s} (unformattable args: {record.args!r})"
        except Exception:
            return "<unformattable log message>"


def _redacted_extra(value: Any) -> Any:
    """``value`` with every string inside it redacted, through dicts, lists and tuples.

    Keys are redacted too. Any other value is returned unchanged, so a formatter that renders
    it with ``%d`` or ``%f`` still can.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {_redacted_extra(key): _redacted_extra(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted_extra(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redacted_extra(item) for item in value)
    return value


def _scrub(record: logging.LogRecord) -> None:
    """Redact ``record``'s message, exception text, stack and extra fields in place. Never raises.

    The message is formatted once and its arguments dropped, so no handler or formatter can
    re-render a secret from them. Exception text is rendered here (formatters reuse a set
    ``exc_text``) so the traceback is redacted too. Attributes beyond the standard ones (the
    ``extra=`` fields a formatter may render or serialize) are redacted as well, including
    strings nested in dicts, lists and tuples.
    """
    record.msg = redact(_formatted_message(record))
    record.args = None
    for name, value in list(vars(record).items()):
        if name not in _STANDARD_RECORD_ATTRIBUTES:
            try:
                setattr(record, name, _redacted_extra(value))
            except Exception:
                setattr(record, name, "<unredactable extra field>")
    if record.exc_info and not record.exc_text:
        try:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        except Exception:
            record.exc_text = "<unformattable exception>"
    if record.exc_text:
        record.exc_text = redact(record.exc_text)
    if record.stack_info:
        record.stack_info = redact(record.stack_info)


class RedactingFilter(logging.Filter):
    """A logging filter that redacts each record's message, exception text and extra fields.

    For explicit use on a handler or logger; :func:`install_log_redaction` covers every
    record in the process without one.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        _scrub(record)
        return True


_REDACTING_MARKER = "_pup_finetune_redacting"


def install_log_redaction() -> None:
    """Redact every log record a logger of this process creates.

    Wraps ``logging.Logger.makeRecord``, so each record is scrubbed after its ``extra=``
    fields are applied and before any logger filter, handler or formatter sees it, whichever
    handlers exist now or are added later. Calling this again does not wrap it twice.
    """
    current = logging.Logger.makeRecord
    if getattr(current, _REDACTING_MARKER, False):
        return

    @functools.wraps(current)
    def make_record(self: logging.Logger, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = current(self, *args, **kwargs)
        _scrub(record)
        return record

    setattr(make_record, _REDACTING_MARKER, True)
    setattr(logging.Logger, "makeRecord", make_record)  # noqa: B010 - avoids a mypy method-assign ignore
