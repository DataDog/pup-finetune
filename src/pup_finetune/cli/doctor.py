# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune doctor`: what this setup can do, and what to fix for the rest.

The report lists each check with its ``status`` (pass, fail, optional or skipped), what it
is needed for and how to fix it: pup and its Datadog login, then each provider's key and
packages (:mod:`pup_finetune.readiness`, from the registries), then the judge keys. Only
the required checks decide ``status``: pup and the login, plus the provider named with
``--provider``. ``ready`` says which steps can run now. Everything but the pup and login
checks is offline unless ``--provider NAME --check`` asks for the provider's live,
read-only preflight. A checklist goes to stderr after the JSON
(:func:`pup_finetune.cli._summary.doctor_summary`).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import typer

import pup_finetune
from pup_finetune import __version__, credentials, readiness
from pup_finetune.cli._common import CommandResult, get_runtime, run_command
from pup_finetune.cli._flags import require_enablers
from pup_finetune.cli._presentation import SET_UP, CommandDoc, Safety, describe
from pup_finetune.datadog.pup import MIN_PUP_VERSION, Pup, format_version
from pup_finetune.doctor import CheckResult, error_detail, run_registered_checks
from pup_finetune.errors import ExternalError
from pup_finetune.next_steps import command, step, template
from pup_finetune.providers import ProviderInfo, provider_info, provider_infos, registered_providers
from pup_finetune.runtime import Runtime

_LOGIN_HINT = "run `pup auth login`"
_UPGRADE_HINT = "upgrade with `brew upgrade datadog-labs/pack/pup`"
_DATADOG_NEEDED_FOR = "reading traces and pushing datasets"

DOCTOR_NEEDS: dict[str, str] = {"check": "provider_name"}
"""``doctor``'s options that need another one (see :mod:`pup_finetune.cli._flags`)."""


def _check_pup(pup: Pup) -> tuple[str | None, bool, CheckResult]:
    """The pup check, pup's version when it could be read, and whether pup was found."""
    minimum = format_version(MIN_PUP_VERSION)
    try:
        path = pup.executable()
    except ExternalError as exc:
        return (
            None,
            False,
            CheckResult(
                name="pup",
                status="fail",
                detail=error_detail(exc),
                label="pup not found",
                needed_for=_DATADOG_NEEDED_FOR,
                fix=exc.hint,
            ),
        )
    try:
        found = pup.version()
    except ExternalError as exc:
        return (
            None,
            True,
            CheckResult(
                name="pup",
                status="fail",
                detail=error_detail(exc),
                label=f"pup at {path}: version unknown",
                needed_for=_DATADOG_NEEDED_FOR,
                fix=exc.hint or _UPGRADE_HINT,
            ),
        )
    version = format_version(found)
    if found < MIN_PUP_VERSION:
        return (
            version,
            True,
            CheckResult(
                name="pup",
                status="fail",
                detail=f"pup {version} is older than the minimum {minimum}; {_UPGRADE_HINT}",
                label=f"pup {version} is older than {minimum}",
                needed_for=_DATADOG_NEEDED_FOR,
                fix=_UPGRADE_HINT,
            ),
        )
    return (
        version,
        True,
        CheckResult(
            name="pup",
            status="pass",
            detail=f"pup {version} at {path}",
            needed_for=_DATADOG_NEEDED_FOR,
        ),
    )


def _check_auth(pup: Pup, *, pup_found: bool) -> CheckResult:
    if not pup_found:
        return CheckResult(
            name="datadog_auth",
            status="skipped",
            detail="skipped: pup is not available",
            label="Datadog login not checked (no pup)",
            needed_for=_DATADOG_NEEDED_FOR,
        )
    try:
        status = pup.auth_status()
    except ExternalError as exc:
        # A specific hint from the pup client, when there is one, beats the generic login advice.
        hint = exc.hint or _LOGIN_HINT
        return CheckResult(
            name="datadog_auth",
            status="fail",
            detail=f"{exc.message}; {hint}",
            label="Datadog: login could not be read",
            needed_for=_DATADOG_NEEDED_FOR,
            fix=hint,
        )
    if not status["authenticated"]:
        return CheckResult(
            name="datadog_auth",
            status="fail",
            detail=f"pup is not signed in; {_LOGIN_HINT}",
            label="Datadog: not signed in",
            needed_for=_DATADOG_NEEDED_FOR,
            fix=_LOGIN_HINT,
        )
    site = status["site"] or "Datadog"
    expires = status.get("expires_at")
    return CheckResult(
        name="datadog_auth",
        status="pass",
        detail=f"pup is signed in to {site}"
        + (f" (its token expires at {expires}; pup refreshes it)" if expires else ""),
        label=f"Datadog: signed in to {site}",
        needed_for=_DATADOG_NEEDED_FOR,
    )


def _package_root() -> Path | None:
    """The checkout this package runs from (an editable install), or ``None``."""
    package = Path(pup_finetune.__file__).resolve().parent
    root = package.parent.parent if package.parent.name == "src" else None
    if root is None or not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        return None
    return root


CHECKOUT_UNKNOWN = "unknown"
"""``versions.checkout`` of a checkout install whose commit git cannot say."""

GIT_REPOSITORY_ENV = ("GIT_DIR", "GIT_WORK_TREE")
"""What git reads its repository from instead of ``-C``; removed, so git describes the
checkout this package runs from, never a repository the caller pointed git at."""


@lru_cache(maxsize=1)
def checkout_commit() -> str | None:
    """The git checkout this package runs from, as ``git describe --always --dirty
    --abbrev=7`` names it (``70ec3ee-dirty``), read at run time so it matches the working
    tree (the build version names the commit it was installed at); :data:`CHECKOUT_UNKNOWN`
    when it runs from a checkout git cannot describe, ``None`` when it does not run from a
    checkout."""
    root = _package_root()
    if root is None:
        return None
    git = shutil.which("git")
    if git is None:
        return CHECKOUT_UNKNOWN
    env = {key: value for key, value in os.environ.items() if key not in GIT_REPOSITORY_ENV}
    try:
        completed = subprocess.run(  # noqa: S603 - git, with fixed arguments
            [git, "-C", str(root), "describe", "--always", "--dirty", "--abbrev=7"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return CHECKOUT_UNKNOWN
    described = completed.stdout.strip()
    return described if completed.returncode == 0 and described else CHECKOUT_UNKNOWN


def _versions(pup_version: str | None) -> dict[str, Any]:
    return {
        "pup_finetune": __version__,
        "checkout": checkout_commit(),
        "pup": pup_version,
        "python": platform.python_version(),
    }


LOGIN_WHY = "the user runs this in their own terminal (it asks for the key)"
"""How a login step says who runs it: never an agent, which must not handle keys."""


def _login_step(info: ProviderInfo, key: CheckResult) -> dict[str, str]:
    uses = credentials.get_spec(key.name.partition(":")[0]).needed_for
    return step(command("auth", "login", info.credential), f"{LOGIN_WHY}, for {uses}")


def _next(
    checks: list[CheckResult],
    ready: dict[str, dict[str, Any]],
    named: ProviderInfo | None,
    *,
    live: bool,
    runtime: Runtime,
) -> tuple[str | None, list[dict[str, str]]]:
    """``next_command`` and ``next_steps``.

    ``next_command`` is the named provider's live check once every required check passed,
    else ``None``: a failed required check's fix comes first (the checklist's ``Next:``
    line shows the first one), and signing in is never a ``next_command``, since only the
    user may enter a key. ``next_steps`` signs in each provider whose key is not set (only
    the named one with ``--provider``; never one whose key could not be read, whose fix is
    the keychain), after pulling traces when that is ready.
    """
    next_command: str | None = None
    failed = [check for check in checks if check.required and not check.ok]
    if named is not None and not live and not failed:
        next_command = command("doctor", "--provider", named.name, "--check")
    steps: list[dict[str, str]] = []
    if ready["dataset pull"]["ready"]:
        steps.append(
            step(
                template("dataset", "pull", "<DIR>", "--ml-app", "<ML_APP>", "--from", "7d"),
                "pull your Agent Observability traces into a dataset directory, once "
                "`pup llm-obs spans search` showed the spans and filter to read",
            )
        )
    by_name = {check.name: check for check in checks}
    for info in provider_infos(runtime):
        if named is not None and info.name != named.name:
            continue
        key = by_name.get(f"{info.credential}:key")
        if key is not None and readiness.key_not_set(key):
            steps.append(_login_step(info, key))
    return next_command, steps


def build_report(
    pup: Pup, runtime: Runtime, *, provider: str | None = None, live: bool = False
) -> dict[str, Any]:
    """The doctor report: ``status`` is ``ready`` when every required check passed."""
    named = provider_info(provider, runtime) if provider is not None else None
    pup_version, pup_found, pup_check = _check_pup(pup)
    checks = [pup_check, _check_auth(pup, pup_found=pup_found)]
    for info in provider_infos(runtime):
        required = named is not None and info.name == named.name
        offline = readiness.provider_checks(info, required=required)
        checks += offline
        if required and live:
            checks += readiness.preflight_checks(info, runtime, offline)
    checks += readiness.judge_checks(runtime)
    checks += run_registered_checks()
    ready = readiness.ready_map(checks, runtime)
    is_ready = all(check.ok for check in checks if check.required)
    next_command, next_steps = _next(checks, ready, named, live=live, runtime=runtime)
    return {
        "status": "ready" if is_ready else "not_ready",
        "provider": named.name if named is not None else None,
        "mode": {"read_only": runtime.read_only, "dev": runtime.dev_mode},
        "versions": _versions(pup_version),
        "checks": [check.to_json() for check in checks],
        "ready": ready,
        "next_command": next_command,
        "next_steps": next_steps,
        "ok": is_ready,
    }


PROVIDERS = " or ".join(registered_providers())
"""The providers ``--provider`` takes, for its help (development-only ones left out)."""

DOC = CommandDoc(
    path="doctor",
    short="Show which steps this setup can run, and what to fix.",
    description=describe(
        "Check pup and its Datadog login, each provider's API key and packages, and the "
        "judge keys, and say which steps are ready. Only pup and the login are required, "
        "unless --provider makes a provider's checks required too; doctor exits 1 when a "
        "required check fails. A checklist follows the report on stderr.",
        "The report has `status` (`ready` or `not_ready`), `provider`, `mode`, `versions` "
        "(`checkout` is what `git describe` names when it runs from a git checkout, "
        "`unknown` when git cannot say), `checks` (each with `status`, `detail`, "
        "`needed_for`, `fix` and `required`), `ready` (each step, and what blocks it: train "
        "and evaluate on a provider are blocked by a failed `--check` preflight too) and "
        "`next_command` and `next_steps`. Signing in (`auth login`) is only ever a step, "
        "which the user runs in their own terminal, never `next_command`. Compatibility: "
        "earlier versions reported top-level `pup` (path, version, min_version) and "
        "`datadog_auth` (authenticated, site, expires_at) sections, and `python` and "
        "`package`; read the `pup` and `datadog_auth` checks (the login's detail names "
        "when its token expires) and `versions` instead. `ok` stays for one release.",
    ),
    examples=(
        "pup finetune doctor",
        "pup finetune doctor --provider baseten",
        "pup finetune doctor --provider fireworks --check",
    ),
    safety=Safety(
        data="asks pup for its login; --check makes read-only provider calls",
        recipients="nowhere (--check: read-only calls to the provider)",
        without="checks the setup",
    ),
    panel=SET_UP,
)


@run_command
def doctor(
    provider_name: Annotated[
        str | None,
        typer.Option(
            "--provider",
            metavar="NAME",
            help=f"Require this provider's checks too: {PROVIDERS}.",
            show_default=False,
        ),
    ] = None,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="With --provider: also preflight its models live (read-only; needs its key).",
        ),
    ] = False,
) -> CommandResult:
    """Print the doctor report, and exit 1 when a required check fails."""
    require_enablers(DOCTOR_NEEDS)
    report = build_report(Pup(), get_runtime(), provider=provider_name, live=check)
    return CommandResult(report, exit_code=0 if report["status"] == "ready" else 1)


def register(app: typer.Typer) -> None:
    """Add the `doctor` command to ``app``."""
    app.command(**DOC.typer())(doctor)
