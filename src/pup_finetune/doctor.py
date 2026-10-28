# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The checks behind ``pup finetune doctor``: their result type and detail helpers.

Every check reports a :class:`CheckResult`: a ``status`` (``pass``, ``fail``, ``optional``
or ``skipped``), a one-line ``detail``, what it is ``needed_for`` and how to ``fix`` it. A
``required`` check decides whether ``doctor`` is ready; one that is not required and does
not pass is ``optional``, never a failure. :mod:`pup_finetune.readiness` builds the
provider and judge checks from the registries. Check details must never contain secret
values.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Final, Literal

from pup_finetune.errors import FinetuneError
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime

__all__ = [
    "STATUSES",
    "CheckResult",
    "Status",
    "error_detail",
    "status_of",
    "unexpected_detail",
]

Status = Literal["pass", "fail", "optional", "skipped"]
"""How a check came out: ``pass``; ``fail`` (a required check that did not pass);
``optional`` (one that did not pass but is not needed now); ``skipped`` (not run, because
what it depends on did not pass)."""

STATUSES: Final[tuple[Status, ...]] = ("pass", "fail", "optional", "skipped")


def status_of(passed: bool, *, required: bool) -> Status:
    """``pass``, else ``fail`` for a required check and ``optional`` for another."""
    if passed:
        return "pass"
    return "fail" if required else "optional"


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckResult:
    """One entry in the doctor report.

    ``label`` is its line in the checklist ``doctor`` prints on stderr (default: the
    detail). ``needed_for`` says what needs it, ``fix`` how to make it pass. ``required``
    says whether it decides ``doctor``'s status; ``ok`` (whether it passed) stays in the JSON
    for one release, for scripts that read it.
    """

    name: str
    status: Status
    detail: str
    label: str = ""
    needed_for: str | None = None
    fix: str | None = None
    required: bool = True

    @property
    def ok(self) -> bool:
        """Whether it passed."""
        return self.status == "pass"

    @property
    def shown(self) -> str:
        """Its checklist text."""
        return self.label or self.detail

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "needed_for": self.needed_for,
            "fix": self.fix,
            "label": self.shown,
            "ok": self.ok,
            "required": self.required,
        }


def error_detail(exc: FinetuneError) -> str:
    """A check detail for an expected failure: its message, then its hint if it has one."""
    return f"{exc.message}; {exc.hint}" if exc.hint else exc.message


def unexpected_detail(exc: Exception) -> str:
    """A check detail for an unexpected exception, in one line; with
    ``PUP_FINETUNE_DEBUG=1`` its traceback is also printed to stderr."""
    debug = Runtime.from_env().debug
    if debug:
        sys.stderr.write(redact("".join(traceback.format_exception(exc))))
    lines = str(exc).strip().splitlines()
    reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    suffix = "" if debug else " (set PUP_FINETUNE_DEBUG=1 for a traceback)"
    return redact(f"check failed unexpectedly ({reason}){suffix}")
