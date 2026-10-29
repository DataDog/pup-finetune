# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Helpers shared by the ``httpx`` clients that send a provider API key.

Each client checks its key with :func:`bearer_key` before it reaches a header (which also
registers it for redaction), names itself with :data:`CLIENT_SOURCE`, and turns failures into
short, redacted text with :func:`exception_reason` and :func:`error_detail`.
"""

from __future__ import annotations

from typing import Final

import httpx
from pydantic import SecretStr

from pup_finetune import __version__
from pup_finetune.credentials import KEY_SHAPE, well_formed
from pup_finetune.errors import UsageError
from pup_finetune.redaction import redact, register_secret

__all__ = ["CLIENT_SOURCE", "DETAIL_LIMIT", "bearer_key", "error_detail", "exception_reason"]

CLIENT_SOURCE: Final = f"pup-finetune/{__version__}"
"""How every client identifies itself (``User-Agent`` and any provider client-source header)."""
DETAIL_LIMIT: Final = 200
"""The most characters of a failure's text an error message quotes."""


def bearer_key(api_key: SecretStr, *, label: str, hint: str) -> str:
    """The value of ``api_key``, registered for redaction, once it is safe to send.

    ``label`` names the key in errors (``"the Fireworks API key"``) and ``hint`` says how to
    fix it. Raises :class:`UsageError` for an empty key, or one that is not
    :data:`~pup_finetune.credentials.KEY_SHAPE`. A malformed key is refused before it reaches
    a header: the HTTP layer would echo an illegal header value in its error, escaped, so
    redaction could not mask it whole, and redaction never masks a key shorter than
    :data:`~pup_finetune.redaction.MIN_SECRET_LENGTH`.
    """
    secret = api_key.get_secret_value()
    if not secret:
        raise UsageError(f"{label} is empty", hint=hint)
    if not well_formed(secret):
        raise UsageError(f"{label} is malformed: it must be {KEY_SHAPE}", hint=hint)
    register_secret(secret)
    return secret


def exception_reason(exc: BaseException) -> str:
    """``exc``'s type and the first line of its text, short and redacted."""
    lines = redact(str(exc)).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0][:DETAIL_LIMIT]}" if lines else type(exc).__name__


def error_detail(response: httpx.Response) -> str:
    """The error message in a failed response, short and redacted; empty when there is none.

    Reads ``error.message``, ``error`` or ``message`` from a JSON body, else the body text
    (never an HTML page).
    """
    text = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        text = str(message or body.get("message") or "")
    elif not response.headers.get("content-type", "").startswith("text/html"):
        text = response.text
    lines = redact(text).strip().splitlines()
    return lines[0][:DETAIL_LIMIT] if lines else ""
