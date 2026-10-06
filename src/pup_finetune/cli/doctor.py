# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune doctor`: check that pup, its Datadog login and this install are ready."""

from __future__ import annotations

import platform
from typing import Any

import typer

from pup_finetune import __version__
from pup_finetune.cli._common import CommandResult, run_command
from pup_finetune.datadog.pup import MIN_PUP_VERSION, Pup, format_version
from pup_finetune.doctor import CheckResult, error_detail, run_registered_checks
from pup_finetune.errors import ExternalError

_LOGIN_HINT = "run `pup auth login`"
_UPGRADE_HINT = "upgrade with `brew upgrade datadog/pack/pup`"


def _check_pup(pup: Pup) -> tuple[dict[str, Any], CheckResult]:
    minimum = format_version(MIN_PUP_VERSION)
    section: dict[str, Any] = {"path": None, "version": None, "min_version": minimum, "ok": False}
    try:
        section["path"] = pup.executable()
        found = pup.version()
    except ExternalError as exc:
        return section, CheckResult("pup", False, error_detail(exc))
    version = format_version(found)
    section["version"] = version
    if found < MIN_PUP_VERSION:
        detail = f"pup {version} is older than the minimum {minimum}; {_UPGRADE_HINT}"
        return section, CheckResult("pup", False, detail)
    section["ok"] = True
    return section, CheckResult("pup", True, f"pup {version} at {section['path']}")


def _check_auth(pup: Pup, *, pup_found: bool) -> tuple[dict[str, Any], CheckResult]:
    section: dict[str, Any] = {"authenticated": False, "site": None, "expires_at": None}
    if not pup_found:
        return section, CheckResult("datadog_auth", False, "skipped: pup is not available")
    try:
        status = pup.auth_status()
    except ExternalError as exc:
        # A specific hint from the pup client, when there is one, beats the generic login advice.
        detail = f"{exc.message}; {exc.hint or _LOGIN_HINT}"
        return section, CheckResult("datadog_auth", False, detail)
    section.update(
        authenticated=status["authenticated"],
        site=status["site"],
        expires_at=status["expires_at"],
    )
    if not status["authenticated"]:
        return section, CheckResult("datadog_auth", False, f"pup is not signed in; {_LOGIN_HINT}")
    site = status["site"] or "Datadog"
    return section, CheckResult("datadog_auth", True, f"pup is signed in to {site}")


def build_report(pup: Pup) -> dict[str, Any]:
    """The doctor report. ``ok`` is false when any required check failed."""
    pup_section, pup_check = _check_pup(pup)
    auth_section, auth_check = _check_auth(pup, pup_found=pup_section["path"] is not None)
    checks = [pup_check, auth_check, *run_registered_checks()]
    return {
        "pup": pup_section,
        "datadog_auth": auth_section,
        "python": platform.python_version(),
        "package": __version__,
        "checks": [check.to_json() for check in checks],
        "ok": all(check.ok for check in checks if check.required),
    }


@run_command
def doctor() -> CommandResult:
    """Check that pup, its Datadog login and this install are ready.

    Prints the report, and exits 1 when a required check fails.
    """
    report = build_report(Pup())
    return CommandResult(report, exit_code=0 if report["ok"] else 1)


def register(app: typer.Typer) -> None:
    """Add the `doctor` command to ``app``."""
    app.command(name="doctor")(doctor)
