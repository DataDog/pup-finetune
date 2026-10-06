# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The check registry behind ``pup finetune doctor``.

Modules add their own checks (credentials, providers, ...) with :func:`register_check` when
they are imported; ``doctor`` runs them after its built-in pup and Datadog login checks, so a
new check never needs an edit to the command. Check details must never contain secret values.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pup_finetune.errors import FinetuneError
from pup_finetune.runtime import Runtime


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """What a check function returns: whether it passed, and a one-line explanation."""

    ok: bool
    detail: str


CheckFn = Callable[[], CheckOutcome]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One entry in the doctor report. A failed ``required`` check makes ``doctor`` exit 1."""

    name: str
    ok: bool
    detail: str
    required: bool = True

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "required": self.required}


@dataclass(frozen=True, slots=True)
class _Registered:
    fn: CheckFn
    required: bool


_REGISTRY: dict[str, _Registered] = {}


def register_check(name: str, fn: CheckFn, *, required: bool = True) -> None:
    """Add a check that ``doctor`` runs, in registration order.

    Registering the same function under the same name again is a no-op; a different function
    under a taken name raises ``ValueError``.
    """
    if not name:
        raise ValueError("a doctor check needs a name")
    existing = _REGISTRY.get(name)
    if existing is not None and (existing.fn is not fn or existing.required != required):
        raise ValueError(f"doctor check {name!r} is already registered")
    _REGISTRY[name] = _Registered(fn, required)


def error_detail(exc: FinetuneError) -> str:
    """A check detail for an expected failure: its message, then its hint if it has one."""
    return f"{exc.message}; {exc.hint}" if exc.hint else exc.message


def run_check(name: str, fn: CheckFn, *, required: bool = True) -> CheckResult:
    """Run one check. A check that raises fails with the error as its detail.

    An unexpected exception is summarized in one line; with ``PUP_FINETUNE_DEBUG=1`` its
    traceback is also printed to stderr.
    """
    try:
        outcome = fn()
    except FinetuneError as exc:
        return CheckResult(name, False, error_detail(exc), required)
    except Exception as exc:
        debug = Runtime.from_env().debug
        if debug:
            traceback.print_exception(exc, file=sys.stderr)
        lines = str(exc).strip().splitlines()
        reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
        suffix = "" if debug else " (set PUP_FINETUNE_DEBUG=1 for a traceback)"
        return CheckResult(name, False, f"check failed unexpectedly ({reason}){suffix}", required)
    return CheckResult(name, outcome.ok, outcome.detail, required)


def run_registered_checks() -> list[CheckResult]:
    """Run every registered check, in registration order."""
    return [
        run_check(name, entry.fn, required=entry.required)
        for name, entry in list(_REGISTRY.items())
    ]
