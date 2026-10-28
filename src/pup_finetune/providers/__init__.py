# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training providers, registered by name and imported only when a command needs one.

The registry maps a provider name to a ``"module:attr"`` import path. Listing providers
imports no provider implementation, so ``--help`` and commands that never train never pay for a provider SDK.
:func:`get_provider` imports the module on first use and calls ``attr`` with the
:class:`~pup_finetune.runtime.Runtime`; it must return a
:class:`~pup_finetune.providers.base.Provider`. Development-only providers are visible only
when ``PUP_FINETUNE_DEV=1``. Each entry also names the provider's ``--provider-option``
keys (:func:`registered_option_keys`), so help can list them without importing a provider; a
test holds them to the providers' option models. It names every model profile each provider
serves too, by alias, with each name that selects it (:func:`known_models`,
:func:`providers_for_model`), so ``train`` can infer ``--provider`` from ``--model`` without
importing a provider; a test holds them to the providers' profiles.
Finally, it says what a provider needs before anything is paid for (:class:`ProviderInfo`):
its display label, the API key it signs in with, the optional packages each step needs
(:class:`Package`, found with :func:`importlib.util.find_spec`, never imported:
:func:`module_state`), the
training flags it refuses (:class:`RefusedFlag`), whether it deploys and whether it trains
remotely. ``doctor`` and the ``train``, ``resume``, ``evaluate`` and ``deploy`` previews read
it (:func:`require_packages`, :func:`require_supported_flags`, :func:`require_deploys`,
:func:`require_remote`), so nothing a preview can know fails only after ``--confirm``. The
protocols live in :mod:`pup_finetune.providers.base`.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

from pup_finetune.billing import BillingPart
from pup_finetune.errors import ExternalError, FinetuneError, UsageError, install_hint
from pup_finetune.providers.base import (
    Provider,
    RemoteJobs,
    SamplingUsage,
    SupportsRemote,
    TrainingUsage,
    selects,
)
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime

__all__ = [
    "STEPS",
    "ModuleState",
    "Package",
    "ProviderInfo",
    "RefusedFlag",
    "Step",
    "aliases_for_model",
    "get_provider",
    "installed",
    "known_models",
    "known_providers",
    "missing_packages",
    "module_state",
    "provider_info",
    "provider_infos",
    "providers_for_model",
    "register_provider",
    "registered_option_keys",
    "registered_providers",
    "remote_jobs",
    "require_deploys",
    "require_key",
    "require_packages",
    "require_remote",
    "require_supported_flags",
    "sampling_billing",
    "training_billing",
]

Step = Literal["train", "evaluate", "deploy"]
"""What a provider is asked to do; ``resume`` is ``train``."""

STEPS: Final[tuple[Step, ...]] = ("train", "evaluate", "deploy")

_DOING: Final[dict[Step, str]] = {
    "train": "training",
    "evaluate": "evaluating",
    "deploy": "deploying",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Package:
    """An optional package a provider needs for some steps, from one of our extras.

    ``key`` names its ``doctor`` check (``<provider>:<key>``); ``modules`` are the import
    names that must all be findable (:func:`installed`: found, never imported), ``extra`` the
    extra that installs them, ``label`` what it is called (``Fireworks Training SDK``),
    ``steps`` the steps that need it and ``needed_for`` that in words. ``short`` names it in
    ``doctor``'s checklist after the provider's label (``renderer (transformers)``) when the
    label does not name the provider.
    """

    key: str
    label: str
    modules: tuple[str, ...]
    extra: str
    steps: tuple[Step, ...]
    needed_for: str
    short: str = ""

    @property
    def fix(self) -> str:
        """How to install it without losing the other extras."""
        return install_hint(self.extra)


@dataclass(frozen=True, slots=True, kw_only=True)
class RefusedFlag:
    """A ``train`` flag a provider cannot honor: ``field`` is its :class:`TrainSpec` field,
    ``flag`` the option, ``reason`` why (``Baseten sets the LoRA alpha itself``)."""

    field: str
    flag: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderInfo:
    """What the registry knows about a provider without importing it.

    ``label`` is its display name (``Fireworks``); ``models`` its model profiles, each
    alias with every name that selects the profile, alias first (see
    :attr:`~pup_finetune.providers.base.ModelProfile.names`); ``credential`` the
    :class:`~pup_finetune.credentials.CredentialSpec` its key is registered under (``None``
    for one without a key); ``packages`` what it needs installed, by step; ``refuses`` the
    flags it cannot honor; ``deploys`` whether it deploys a promoted model; ``offline``
    whether it never leaves the machine (so ``PUP_READ_ONLY`` allows its runs); ``remote``
    whether ``train --remote`` can run its training loop in a job of the provider's.
    """

    name: str
    target: str
    dev_only: bool = False
    label: str = ""
    options: tuple[str, ...] = ()
    models: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    credential: str | None = None
    packages: tuple[Package, ...] = ()
    refuses: tuple[RefusedFlag, ...] = ()
    deploys: bool = True
    offline: bool = False
    remote: bool = False

    def packages_for(self, step: Step) -> tuple[Package, ...]:
        """The packages ``step`` needs."""
        return tuple(package for package in self.packages if step in package.steps)


_REGISTRY: dict[str, ProviderInfo] = {}


def register_provider(
    name: str,
    target: str,
    *,
    dev_only: bool = False,
    options: tuple[str, ...] = (),
    models: Mapping[str, Sequence[str]] | None = None,
    label: str = "",
    credential: str | None = None,
    packages: tuple[Package, ...] = (),
    refuses: tuple[RefusedFlag, ...] = (),
    deploys: bool = True,
    offline: bool = False,
    remote: bool = False,
) -> None:
    """Register ``name`` as the provider ``target`` (``"package.module:attr"``) builds, with
    the ``--provider-option`` keys it takes (``options``), the model profiles it serves
    (``models``: each alias with the names that select it, alias first) and what it needs
    (see :class:`ProviderInfo`; ``label`` defaults to ``name`` capitalized).

    Nothing is imported. Registering the same entry again is a no-op; a different entry under
    a taken name, a model whose names do not start with its alias, or a name two of its
    models share raises ``ValueError``.
    """
    if not name:
        raise ValueError("a provider needs a name")
    module, sep, attr = target.partition(":")
    if not (module and sep and attr):
        raise ValueError(f"provider target must look like 'module:attr', got {target!r}")
    table = {alias: tuple(names) for alias, names in (models or {}).items()}
    seen: set[str] = set()
    for alias, names in table.items():
        if not names or names[0] != alias:
            raise ValueError(f"provider {name!r}: the names of {alias!r} must start with it")
        for model_name in names:
            if model_name.casefold() in seen:
                raise ValueError(f"provider {name!r}: {model_name!r} names two of its models")
            seen.add(model_name.casefold())
    entry = ProviderInfo(
        name=name,
        target=target,
        dev_only=dev_only,
        label=label or name.capitalize(),
        options=tuple(options),
        models=MappingProxyType(table),
        credential=credential,
        packages=tuple(packages),
        refuses=tuple(refuses),
        deploys=deploys,
        offline=offline,
        remote=remote,
    )
    existing = _REGISTRY.get(name)
    if existing is not None and existing != entry:
        raise ValueError(f"provider {name!r} is already registered")
    _REGISTRY[name] = entry


def known_providers(runtime: Runtime) -> list[str]:
    """The provider names ``runtime`` may use, sorted. Imports no provider module."""
    return sorted(name for name, entry in _REGISTRY.items() if _visible(entry, runtime))


def registered_providers(*, include_dev: bool = False) -> list[str]:
    """Every registered provider name, in registration order; development-only ones only with
    ``include_dev``. What help lists, so it reads the same in every environment. Imports no
    provider module."""
    return [name for name, entry in _REGISTRY.items() if include_dev or not entry.dev_only]


def provider_infos(runtime: Runtime) -> list[ProviderInfo]:
    """What the registry knows about each provider ``runtime`` may use, in registration
    order. Imports no provider module."""
    return [entry for entry in _REGISTRY.values() if _visible(entry, runtime)]


def provider_info(name: str, runtime: Runtime) -> ProviderInfo:
    """What the registry knows about ``name``, without importing it; a :class:`UsageError`
    listing the known names when ``runtime`` may not use it."""
    entry = _REGISTRY.get(name)
    if entry is None or not _visible(entry, runtime):
        raise _unknown(name, runtime, dev_only=entry is not None)
    return entry


ModuleState = Literal["installed", "missing", "broken"]
"""Whether a module is found (``installed``), absent (``missing``), or there but failing
to load (``broken``: a parent package raises on import)."""


def module_state(module: str) -> tuple[ModuleState, str]:
    """Whether ``module`` is installed, found with :func:`importlib.util.find_spec` and not
    imported (a dotted name imports its parent packages only), and, when it is ``broken``,
    why (one redacted line).

    It is ``missing`` when it is not found or a :class:`ModuleNotFoundError` names it or one
    of its parent packages; any other failure (a parent that raises on import, or needs a
    module that is missing) makes it ``broken``. Never raises.
    """
    try:
        found = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module == missing or module.startswith(f"{missing}.")):
            return "missing", ""
        return "broken", _reason(exc)
    except ValueError as exc:
        # find_spec raises it for a module already imported without a spec.
        if sys.modules.get(module) is not None:
            return "installed", ""
        return "broken", _reason(exc)
    except Exception as exc:
        return "broken", _reason(exc)
    return ("installed", "") if found else ("missing", "")


def _reason(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return redact(f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__)


def installed(module: str) -> bool:
    """Whether ``module`` is installed and loads (:func:`module_state`), without importing
    it."""
    return module_state(module)[0] == "installed"


def missing_packages(name: str, runtime: Runtime, step: Step) -> list[Package]:
    """The packages provider ``name`` needs for ``step`` that are not installed."""
    info = provider_info(name, runtime)
    return [
        package
        for package in info.packages_for(step)
        if not all(installed(module) for module in package.modules)
    ]


def require_packages(name: str, runtime: Runtime, step: Step) -> None:
    """Raise :class:`UsageError` (exit 2) with the install hint when provider ``name`` lacks
    a package ``step`` needs, e.g. ``training on Fireworks needs the Fireworks Training SDK,
    which is not installed``. Imports nothing."""
    missing = missing_packages(name, runtime, step)
    if missing:
        info = provider_info(name, runtime)
        package = missing[0]
        broken = [reason for state, reason in map(module_state, package.modules) if reason]
        why = (
            f"which is installed but cannot be loaded ({broken[0]})"
            if broken
            else "which is not installed"
        )
        raise UsageError(
            f"{_DOING[step]} on {info.label} needs the {package.label}, {why}",
            hint=package.fix,
        )


def require_key(
    name: str, runtime: Runtime, *, needed_by: str, warnings: list[str] | None = None
) -> None:
    """Raise the :class:`UsageError` (exit 2) a ``train`` preview's preflight raises when no
    source has the API key provider ``name`` signs in with, naming ``needed_by``; with
    ``warnings`` (a read-only preview, which offers no ``--confirm`` to fail), append it there
    instead. A provider that signs in with no key (the fake) needs none. Never returns the
    key, and makes no call."""
    from pup_finetune import credentials

    credential = provider_info(name, runtime).credential
    if credential is None:
        return
    found = credentials.key_source(credential)
    if isinstance(found, credentials.KeyUnreadable):
        raise ExternalError(found.detail, hint=found.fix, source="the OS keychain")
    if found is not None:
        return
    error = credentials.missing_key_error(credential, needed_by=needed_by)
    if warnings is None:
        raise error
    warnings.append(f"{error.message}: {error.hint}")


def require_supported_flags(
    name: str,
    runtime: Runtime,
    values: Mapping[str, Any],
    *,
    hint: Callable[[str], str] = lambda flag: f"drop {flag}",
) -> None:
    """Raise :class:`UsageError` when ``values`` (``TrainSpec`` fields) set a flag provider
    ``name`` refuses, e.g. ``--lora-alpha is not supported on baseten: Baseten sets the LoRA
    alpha itself``, with ``hint`` of the refused flag as its hint."""
    for refused in provider_info(name, runtime).refuses:
        if values.get(refused.field) is not None:
            raise UsageError(
                f"{refused.flag} is not supported on {name}: {refused.reason}",
                hint=hint(refused.flag),
            )


def require_deploys(name: str, runtime: Runtime) -> None:
    """Raise :class:`UsageError` when provider ``name`` cannot deploy a model."""
    info = provider_info(name, runtime)
    if not info.deploys:
        raise UsageError(
            f"deploying on {info.label} is not available in this version of pup-finetune",
            hint="use the promoted model on the provider directly",
        )


def require_remote(name: str, runtime: Runtime) -> None:
    """Raise :class:`UsageError` when provider ``name`` cannot train in a job of its own."""
    info = provider_info(name, runtime)
    if not info.remote:
        raise _no_remote(info.label)


def remote_jobs(provider: Provider) -> RemoteJobs:
    """``provider``'s remote jobs; the :func:`require_remote` error when it has none."""
    if not isinstance(provider, SupportsRemote):
        entry = _REGISTRY.get(provider.name)
        raise _no_remote(entry.label if entry is not None else provider.name)
    return provider.remote_jobs()


def _no_remote(label: str) -> UsageError:
    return UsageError(
        f"training remotely on {label} is not available in this version of pup-finetune",
        hint="drop --remote; --detach trains in a background worker on this machine",
    )


def registered_option_keys(*, include_dev: bool = False) -> dict[str, tuple[str, ...]]:
    """Each registered provider's ``--provider-option`` keys, by name; development-only
    providers only with ``include_dev``. Imports no provider module."""
    return {
        name: entry.options
        for name, entry in _REGISTRY.items()
        if include_dev or not entry.dev_only
    }


def known_models(runtime: Runtime) -> dict[str, Mapping[str, tuple[str, ...]]]:
    """The model profiles of each provider ``runtime`` may use, by provider name, sorted by
    name: each alias with every name that selects it, alias first. Imports no provider
    module."""
    return {name: _REGISTRY[name].models for name in known_providers(runtime)}


def aliases_for_model(model: str, runtime: Runtime) -> dict[str, str]:
    """The alias of the profile that ``model`` names (its alias, or in any case the
    provider's model id or its tokenizer: :func:`~pup_finetune.providers.base.selects`), by
    each provider ``runtime`` may use that has one, sorted by provider. Imports no provider
    module."""
    return {
        name: alias
        for name, models in known_models(runtime).items()
        for alias, names in models.items()
        if selects(names, model)
    }


def providers_for_model(model: str, runtime: Runtime) -> list[str]:
    """The providers of :func:`aliases_for_model`: what ``train`` infers ``--provider``
    from."""
    return list(aliases_for_model(model, runtime))


def get_provider(name: str, runtime: Runtime) -> Provider:
    """Import and build the provider registered as ``name``.

    Raises :class:`UsageError` listing the known names for an unknown provider, and when the
    provider's module or a package it needs is not installed, whether that package is
    imported with the module or lazily, while the provider is built.
    """
    entry = _REGISTRY.get(name)
    if entry is None or not _visible(entry, runtime):
        raise _unknown(name, runtime, dev_only=entry is not None)
    factory = _load(name, entry.target)
    try:
        provider = factory(runtime)
    except ModuleNotFoundError as exc:
        raise _not_installed(name, entry.target.partition(":")[0], exc) from exc
    if not isinstance(provider, Provider):
        raise FinetuneError(f"provider {name!r} ({entry.target}) did not build a Provider")
    return provider


def training_billing(provider: Provider, runtime: Runtime, usage: TrainingUsage) -> BillingPart:
    """What training on ``usage`` bills on ``provider``, offline as its registry entry is."""
    return BillingPart(
        provider.training_billing(usage), provider_info(provider.name, runtime).offline
    )


def sampling_billing(name: str, runtime: Runtime, usage: SamplingUsage) -> BillingPart:
    """What generating ``usage`` bills on the provider ``name``, offline as its registry
    entry is."""
    return BillingPart(
        get_provider(name, runtime).sampling_billing(usage), provider_info(name, runtime).offline
    )


def _visible(entry: ProviderInfo, runtime: Runtime) -> bool:
    return runtime.dev_mode or not entry.dev_only


def _unknown(name: str, runtime: Runtime, *, dev_only: bool) -> UsageError:
    known = ", ".join(known_providers(runtime))
    hint = f"known providers: {known}" if known else "no providers are available"
    if dev_only:
        hint += f"; {name!r} is for development only and needs PUP_FINETUNE_DEV=1"
    return UsageError(f"unknown provider {name!r}", hint=hint)


def _load(name: str, target: str) -> Callable[[Runtime], object]:
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise _not_installed(name, module_name, exc) from exc
    try:
        factory: object = getattr(module, attr)
    except AttributeError as exc:
        raise FinetuneError(f"provider {name!r}: {module_name} has no attribute {attr!r}") from exc
    if not callable(factory):
        raise FinetuneError(f"provider {name!r}: {target} is not callable")
    return factory


def _not_installed(name: str, module_name: str, exc: ModuleNotFoundError) -> UsageError:
    """The error for a module missing while importing provider ``name`` or building it."""
    missing = exc.name or ""
    if _is_first_party(missing, module_name):
        # Name the target when it (or a parent) is what is missing, else our missing module.
        target_missing = (
            not missing or module_name == missing or module_name.startswith(f"{missing}.")
        )
        shown = module_name if target_missing else missing
        return UsageError(
            f"provider {name!r} is registered but its module {shown} is missing",
            hint="reinstall pup-finetune; the installation looks incomplete",
        )
    return UsageError(
        f"provider {name!r} needs the {missing!r} package, which is not installed",
        hint=f"install pup-finetune with the {name!r} extra: pup-finetune[{name}]",
    )


def _is_first_party(missing: str, module_name: str) -> bool:
    """Whether the missing module is the target, a parent of it, or any of our own modules.

    Any of these means the installation is incomplete; anything else is a third-party
    dependency the provider's extra installs.
    """
    return (
        not missing
        or missing == module_name
        or module_name.startswith(f"{missing}.")
        or missing == "pup_finetune"
        or missing.startswith("pup_finetune.")
    )


# Built-in providers register below by import path only, so listing them imports no
# provider implementation.
register_provider(
    "fake",
    "pup_finetune.providers.fake:FakeProvider",
    dev_only=True,
    options=(
        "loss_curve",
        "val_curve",
        "step_delay_s",
        "fail_at_step",
        "prefix_stable",
        "max_seq_len",
        "hold_file",
    ),
    models={"fake-model": ("fake-model", "fake/fake-model", "fake/tokenizer")},
    offline=True,
)
register_provider(
    "baseten",
    "pup_finetune.providers.baseten.provider:BasetenProvider",
    options=("max_seq_len",),
    models={
        "qwen3p5-9b": ("qwen3p5-9b", "Qwen/Qwen3.5-9B"),
        "glm-5p3-flash": ("glm-5p3-flash", "zai-org/GLM-5.3-Flash"),
        "qwen3p8-27b": ("qwen3p8-27b", "Qwen/Qwen3.8-27B"),
        "kimi-k3": ("kimi-k3", "moonshotai/Kimi-K3"),
    },
    label="Baseten",
    credential="baseten",
    packages=(
        Package(
            key="sdk",
            label="Baseten Loops SDK",
            modules=("baseten.loops",),
            extra="baseten",
            steps=("train", "evaluate"),
            needed_for="training and evaluation on baseten",
        ),
        Package(
            key="renderer",
            label="Hugging Face transformers package",
            modules=("transformers",),
            extra="hf",
            steps=("train", "evaluate"),
            needed_for="rendering rows for baseten",
            short="renderer (transformers)",
        ),
    ),
    refuses=(
        RefusedFlag(
            field="lora_alpha", flag="--lora-alpha", reason="Baseten sets the LoRA alpha itself"
        ),
    ),
    remote=True,
)
register_provider(
    "fireworks",
    "pup_finetune.providers.fireworks.provider:FireworksProvider",
    options=("max_seq_len",),
    models={
        "qwen3p8-27b": (
            "qwen3p8-27b",
            "accounts/fireworks/models/qwen3p8-27b",
            "Qwen/Qwen3.8-27B",
        ),
        "glm-5p3-flash": (
            "glm-5p3-flash",
            "accounts/fireworks/models/glm-5p3-flash",
            "zai-org/GLM-5.3-Flash",
        ),
        "kimi-k3": ("kimi-k3", "accounts/fireworks/models/kimi-k3", "moonshotai/Kimi-K3"),
        "deepseek-v4-flash-0731": (
            "deepseek-v4-flash-0731",
            "accounts/fireworks/models/deepseek-v4-flash-0731",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "muse-glimmer-30b": (
            "muse-glimmer-30b",
            "accounts/fireworks/models/muse-glimmer-30b",
            "meta-models/Muse-Glimmer-30B",
        ),
    },
    label="Fireworks",
    credential="fireworks",
    packages=(
        Package(
            key="sdk",
            label="Fireworks Training SDK",
            modules=("fireworks", "tinker"),
            extra="fireworks",
            steps=("train", "evaluate", "deploy"),
            needed_for="training, evaluation and deployment on fireworks",
        ),
        Package(
            key="renderer",
            label="Hugging Face transformers package",
            modules=("transformers",),
            extra="hf",
            steps=("train", "evaluate"),
            needed_for="rendering rows for fireworks",
            short="renderer (transformers)",
        ),
    ),
)

# Providers with an API key register its CredentialSpec in their (lightweight) package, so
# `auth`, `doctor` and every command that imports this registry know about the key. The
# judge-only providers (Anthropic, OpenAI) register theirs the same way, in judge_credentials.
from pup_finetune import judge_credentials as _judge_credentials  # noqa: E402, F401
from pup_finetune.providers import baseten as _baseten  # noqa: E402, F401
from pup_finetune.providers import fireworks as _fireworks  # noqa: E402, F401
