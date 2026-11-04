# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What each step needs, checked from the registries: the provider and judge checks of
``doctor``, and which steps are ready.

Everything here is built from the provider registry (:class:`~pup_finetune.providers.
ProviderInfo`: each provider's key and packages) and the credential registry
(:class:`~pup_finetune.credentials.CredentialSpec`: each key's ``needed_for``), so a new
provider or judge key shows up in ``doctor`` without an edit here. The checks are offline:
a key is looked up (never read out), a package is found (never imported). Only
:func:`preflight_checks` calls a provider, read only, for ``doctor --provider NAME
--check``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pup_finetune import credentials, providers
from pup_finetune.doctor import CheckResult, error_detail, status_of, unexpected_detail
from pup_finetune.errors import FinetuneError
from pup_finetune.judge_models import EVALUATE_DEFAULTS, TRIAGE_DEFAULTS, JudgeDefault
from pup_finetune.providers import Package, ProviderInfo, get_provider, provider_infos
from pup_finetune.providers.base import (
    MAX_SEQ_LEN_FLAG,
    MAX_SEQ_LEN_OPTION,
    SERVERLESS,
    ModelProfile,
    Placement,
    Provider,
    TrainingContext,
    expected_placement,
    fitted_context,
    under_profile_context,
)
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime

__all__ = [
    "DATADOG_CHECKS",
    "KEY_NOT_SET",
    "READ_ONLY",
    "judge_checks",
    "judge_credential",
    "key_check",
    "key_not_set",
    "package_check",
    "preflight_checks",
    "provider_checks",
    "ready_map",
]

DATADOG_CHECKS: Final = ("pup", "datadog_auth")
"""The checks every Datadog read and write needs: ``doctor`` is ready when they pass."""

READ_ONLY: Final = "read_only"
"""What a step ``PUP_READ_ONLY`` refuses is blocked by, in :func:`ready_map`."""


def judge_credential(spec: str) -> str:
    """The credential a judge spec signs in with, by its prefix (see :mod:`pup_finetune.
    judges`): ``anthropic`` or ``openai``, else ``fireworks``. Imports no judge client."""
    prefix = spec.partition("/")[0].casefold()
    return prefix if prefix in ("anthropic", "openai") else "fireworks"


def _default_judges(
    table: Sequence[JudgeDefault], by_name: Mapping[str, CheckResult]
) -> tuple[str, ...]:
    """The judges a command defaults to by the checks that passed: a provider's key, and
    the package its judges need (as :func:`pup_finetune.judges.default_judges` picks)."""
    from pup_finetune.judges import JUDGE_PACKAGES, pick_default

    def source(name: str) -> str | None:
        needed = [f"{name}:key"]
        if (package := JUDGE_PACKAGES.get(name)) is not None:
            needed.append(f"{name}:{package.key}")
        passed = all((check := by_name.get(c)) is not None and check.ok for c in needed)
        return "check" if passed else None

    return pick_default(table, source)[0].judges


KEY_NOT_SET: Final = " key not set"
"""How the label of a key check whose key is not configured ends (:func:`key_not_set`)."""


def key_check(spec: credentials.CredentialSpec, *, required: bool) -> CheckResult:
    """Whether ``spec``'s key is configured, and where it comes from (never the key).

    A key that cannot be read (a locked keychain, a failing keychain backend) does not pass,
    with what to do about the keychain as its fix: signing in again would not help. Nor does
    a malformed key (not :data:`~pup_finetune.credentials.KEY_SHAPE`).
    """
    name = f"{spec.provider}:key"
    fix = credentials.login_hint(spec)
    needed_for = spec.needed_for or None
    source = credentials.key_source(spec.provider)
    if isinstance(source, credentials.KeyUnreadable | credentials.KeyMalformed):
        problem = "malformed" if isinstance(source, credentials.KeyMalformed) else "unreadable"
        return CheckResult(
            name=name,
            status=status_of(False, required=required),
            detail=source.detail,
            label=f"{spec.label} key {problem}",
            needed_for=needed_for,
            fix=source.fix,
            required=required,
        )
    if source is None:
        return CheckResult(
            name=name,
            status=status_of(False, required=required),
            detail=f"no {spec.provider} API key; {fix}",
            label=f"{spec.label}{KEY_NOT_SET}",
            needed_for=needed_for,
            fix=fix,
            required=required,
        )
    if source.endswith("-sdk"):
        detail = (
            f"no {spec.provider} API key, but the {spec.provider} SDK can sign in on its own "
            f"(source: {source})"
        )
    else:
        detail = f"{spec.provider} API key found (source: {source})"
    return CheckResult(
        name=name,
        status="pass",
        detail=detail,
        label=f"{spec.label} key ({credentials.shown_source(spec, source)})",
        needed_for=needed_for,
        required=required,
    )


def key_not_set(check: CheckResult) -> bool:
    """Whether ``check`` is a :func:`key_check` that found no key at all (rather than one it
    could not read): the one case where signing in is the fix."""
    return check.name.endswith(":key") and not check.ok and check.shown.endswith(KEY_NOT_SET)


def package_check(owner: str, label: str, package: Package, *, required: bool) -> CheckResult:
    """Whether ``package`` of the provider or judge ``owner`` (shown as ``label``) is
    installed: found, never imported (:func:`~pup_finetune.providers.module_state`). A
    package that is there but fails to load is told apart from a missing one; neither, nor
    anything unexpected while looking, raises."""
    name = f"{owner}:{package.key}"
    shown = f"{label} {package.short}" if package.short else package.label
    try:
        states = [providers.module_state(module) for module in package.modules]
    except Exception as exc:
        states = [("broken", unexpected_detail(exc))]
    broken = [reason for state, reason in states if state == "broken"]
    present = all(state == "installed" for state, _ in states)
    if present:
        detail, found = f"the {package.label} is installed", "installed"
    elif broken:
        detail = (
            f"the {package.label} is installed but cannot be loaded: {broken[0]}; {package.fix}"
        )
        found = "cannot be loaded"
    else:
        detail, found = f"the {package.label} is not installed; {package.fix}", "missing"
    return CheckResult(
        name=name,
        status=status_of(present, required=required),
        detail=redact(detail),
        label=f"{shown} {found}",
        needed_for=package.needed_for,
        fix=None if present else package.fix,
        required=required,
    )


def provider_checks(info: ProviderInfo, *, required: bool) -> list[CheckResult]:
    """``info``'s key, then each package it needs, offline. ``required`` when the user named
    the provider (``doctor --provider``)."""
    checks: list[CheckResult] = []
    if info.credential is not None:
        checks.append(key_check(credentials.get_spec(info.credential), required=required))
    checks += [
        package_check(info.name, info.label, package, required=required)
        for package in info.packages
    ]
    return checks


def _provider_credentials(runtime: Runtime) -> set[str]:
    return {info.credential for info in provider_infos(runtime) if info.credential is not None}


def judge_checks(runtime: Runtime) -> list[CheckResult]:
    """The keys of the judge-only providers (every registered credential no training
    provider signs in with), then the packages their judges need. Never required."""
    from pup_finetune.judges import JUDGE_PACKAGES

    specs = [
        spec
        for spec in credentials.all_specs()
        if spec.provider not in _provider_credentials(runtime)
    ]
    checks = [key_check(spec, required=False) for spec in specs]
    checks += [
        package_check(spec.provider, spec.label, JUDGE_PACKAGES[spec.provider], required=False)
        for spec in specs
        if spec.provider in JUDGE_PACKAGES
    ]
    return checks


ProviderFactory = Callable[[str, Runtime], Provider]


def preflight_context(
    provider: Provider, profile: ModelProfile, max_seq_len: int | None = None
) -> TrainingContext | None:
    """Preflight ``profile`` at the context a run would train it at, and return that context
    with its source when it is not the profile's own: ``max_seq_len``, else the provider's
    limit when that is shorter (:func:`fitted_context`). Raises like
    :meth:`~Provider.preflight`, so a ``max_seq_len`` over the provider's limit or the
    profile's context is refused. Leaves the provider's options as it found them: none.
    """
    try:
        if max_seq_len is not None:
            provider.validate_options({MAX_SEQ_LEN_OPTION: str(max_seq_len)})
        limit = provider.preflight(provider.training_profile(profile))
    finally:
        provider.validate_options({})
    if max_seq_len is not None:
        return TrainingContext(max_seq_len=max_seq_len, source=MAX_SEQ_LEN_FLAG)
    return fitted_context(profile, limit)


@dataclass(frozen=True, slots=True)
class ModelPreflight:
    """What a live preflight of one model found: the context a run would train it at when it
    is not the profile's own (:func:`preflight_context`), and where the provider expects to
    run it (``None`` for a provider that does not say)."""

    context: TrainingContext | None
    placement: Placement | None


def preflight_model(
    provider: Provider, profile: ModelProfile, max_seq_len: int | None = None
) -> ModelPreflight:
    """:func:`preflight_context` and :func:`~pup_finetune.providers.base.expected_placement`
    of ``profile``; raises like :meth:`~Provider.preflight`."""
    context = preflight_context(provider, profile, max_seq_len)
    return ModelPreflight(context=context, placement=expected_placement(provider, profile))


def preflight_checks(
    info: ProviderInfo,
    runtime: Runtime,
    offline: Sequence[CheckResult],
    *,
    factory: ProviderFactory | None = None,
    max_seq_len: int | None = None,
) -> list[CheckResult]:
    """``doctor --provider NAME --check``: the provider's live, read-only preflight of each
    model profile (what ``models list --check`` runs), at the context a run would train it at
    (:func:`preflight_model`), naming where the provider expects to run it. Skipped when one
    of its ``offline`` checks (the key and what training needs installed) did not pass.
    ``factory`` builds the provider (default: the registry's)."""
    blockers = [check.name for check in offline if not check.ok]
    try:
        provider = (factory or get_provider)(info.name, runtime)
        profiles = provider.models()
    except Exception as exc:
        detail = (
            redact(error_detail(exc)) if isinstance(exc, FinetuneError) else unexpected_detail(exc)
        )
        return [
            CheckResult(
                name=f"{info.name}:models",
                status="fail",
                detail=detail,
                label=f"{info.label} could not be loaded",
                needed_for=f"training on {info.name}",
                fix=redact(exc.hint) if isinstance(exc, FinetuneError) and exc.hint else None,
            )
        ]
    checks: list[CheckResult] = []
    for profile in profiles:
        name = f"{info.name}:model:{profile.alias}"
        needed_for = f"training {profile.alias} on {info.name}"
        if blockers:
            checks.append(
                CheckResult(
                    name=name,
                    status="skipped",
                    detail=f"skipped: {', '.join(blockers)} did not pass",
                    label=f"{profile.alias} preflight skipped",
                    needed_for=needed_for,
                )
            )
            continue
        try:
            found = preflight_model(provider, profile, max_seq_len)
        except FinetuneError as exc:
            checks.append(
                CheckResult(
                    name=name,
                    status="fail",
                    detail=redact(error_detail(exc)),
                    label=f"{profile.alias} cannot be fine-tuned on {info.label} now",
                    needed_for=needed_for,
                    fix=redact(exc.hint) if exc.hint else None,
                )
            )
            continue
        except Exception as exc:
            checks.append(
                CheckResult(
                    name=name,
                    status="fail",
                    detail=unexpected_detail(exc),
                    label=f"{profile.alias} preflight failed",
                    needed_for=needed_for,
                )
            )
            continue
        context, placement = found.context, found.placement
        at = "" if context is None else f" at {context.max_seq_len} tokens ({context.source})"
        detail = f"{info.label} can fine-tune {profile.base_model}{at} now"
        if context is not None and context.fitted:
            detail += f", {under_profile_context(profile)}"
        # Dedicated is what a provider without serverless always does, so only serverless
        # is news; models list --check reports the placement either way.
        serverless = placement == SERVERLESS
        if serverless:
            detail += ", expected on serverless capacity (else dedicated)"
        on = " (serverless)" if serverless else ""
        checks.append(
            CheckResult(
                name=name,
                status="pass",
                detail=detail,
                label=f"{profile.alias} can be fine-tuned on {info.label}{on}{at}",
                needed_for=needed_for,
            )
        )
    return checks


def _blocked(by_name: Mapping[str, CheckResult], names: Iterable[str]) -> list[str]:
    return [name for name in names if name in by_name and not by_name[name].ok]


def _failed_preflights(checks: Sequence[CheckResult], provider: str) -> list[str]:
    """The live preflight checks of ``provider`` (``doctor --provider NAME --check``) that
    failed: its models that cannot be fine-tuned now, or the provider that could not be
    loaded. A skipped preflight is left out: what it was skipped for blocks already."""
    return [
        check.name
        for check in checks
        if check.status == "fail"
        and (check.name == f"{provider}:models" or check.name.startswith(f"{provider}:model:"))
    ]


def _step(blocked_by: Sequence[str]) -> dict[str, Any]:
    return {"ready": not blocked_by, "blocked_by": list(blocked_by)}


def ready_map(checks: Sequence[CheckResult], runtime: Runtime) -> dict[str, dict[str, Any]]:
    """Each step, whether it is ready, and the checks (or :data:`READ_ONLY`) blocking it:
    the dataset steps on the Datadog checks, triage on its default judges' keys (the
    defaults the passing key checks choose), train and evaluate on each provider on its key
    and packages (evaluate on the default judge's key too) and on a failed live preflight,
    and deploy (and undeploy) on each provider that deploys on its key and the packages
    deploying needs. ``PUP_READ_ONLY`` blocks push and every paid run or remote write that
    leaves the machine."""
    by_name = {check.name: check for check in checks}
    read_only = [READ_ONLY] if runtime.read_only else []
    datadog = _blocked(by_name, DATADOG_CHECKS)
    triage_keys = dict.fromkeys(
        f"{judge_credential(j)}:key" for j in _default_judges(TRIAGE_DEFAULTS, by_name)
    )
    steps: dict[str, dict[str, Any]] = {
        "dataset pull": _step(datadog),
        "dataset show": _step([]),
        "dataset push": _step(datadog + read_only),
        "dataset triage": _step(_blocked(by_name, triage_keys)),
    }
    (judge,) = _default_judges(EVALUATE_DEFAULTS, by_name)
    judge_key = f"{judge_credential(judge)}:key"
    for info in provider_infos(runtime):
        key = [f"{info.credential}:key"] if info.credential is not None else []
        paid = [] if info.offline else read_only
        preflight = _failed_preflights(checks, info.name)
        train = [*key, *(f"{info.name}:{package.key}" for package in info.packages_for("train"))]
        steps[f"train on {info.name}"] = _step(_blocked(by_name, train) + preflight + paid)
        evaluate = [*key, *(f"{info.name}:{p.key}" for p in info.packages_for("evaluate"))]
        if not info.offline:
            evaluate.append(judge_key)
        steps[f"evaluate on {info.name}"] = _step(
            _blocked(by_name, dict.fromkeys(evaluate)) + preflight + paid
        )
        if info.deploys:
            deploy = [*key, *(f"{info.name}:{p.key}" for p in info.packages_for("deploy"))]
            steps[f"deploy on {info.name}"] = _step(_blocked(by_name, deploy) + paid)
    return steps
