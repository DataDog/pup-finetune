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
test holds them to the providers' option models. The protocols live in
:mod:`pup_finetune.providers.base`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from pup_finetune.errors import FinetuneError, UsageError
from pup_finetune.providers.base import Provider
from pup_finetune.runtime import Runtime

__all__ = ["get_provider", "known_providers", "register_provider", "registered_option_keys"]


@dataclass(frozen=True, slots=True)
class _Entry:
    target: str
    dev_only: bool
    options: tuple[str, ...] = ()


_REGISTRY: dict[str, _Entry] = {}


def register_provider(
    name: str, target: str, *, dev_only: bool = False, options: tuple[str, ...] = ()
) -> None:
    """Register ``name`` as the provider ``target`` (``"package.module:attr"``) builds, with
    the ``--provider-option`` keys it takes (``options``).

    Nothing is imported. Registering the same entry again is a no-op; a different entry under
    a taken name raises ``ValueError``.
    """
    if not name:
        raise ValueError("a provider needs a name")
    module, sep, attr = target.partition(":")
    if not (module and sep and attr):
        raise ValueError(f"provider target must look like 'module:attr', got {target!r}")
    entry = _Entry(target, dev_only, tuple(options))
    existing = _REGISTRY.get(name)
    if existing is not None and existing != entry:
        raise ValueError(f"provider {name!r} is already registered")
    _REGISTRY[name] = entry


def known_providers(runtime: Runtime) -> list[str]:
    """The provider names ``runtime`` may use, sorted. Imports no provider module."""
    return sorted(name for name, entry in _REGISTRY.items() if _visible(entry, runtime))


def registered_option_keys(*, include_dev: bool = False) -> dict[str, tuple[str, ...]]:
    """Each registered provider's ``--provider-option`` keys, by name; development-only
    providers only with ``include_dev``. Imports no provider module."""
    return {
        name: entry.options
        for name, entry in _REGISTRY.items()
        if include_dev or not entry.dev_only
    }


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


def _visible(entry: _Entry, runtime: Runtime) -> bool:
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
    ),
)
register_provider(
    "fireworks",
    "pup_finetune.providers.fireworks.provider:FireworksProvider",
    options=("max_seq_len",),
)
register_provider(
    "baseten", "pup_finetune.providers.baseten.provider:BasetenProvider", options=("max_seq_len",)
)

# Providers with an API key register its CredentialSpec in their (lightweight) package, so
# `auth`, `doctor` and every command that imports this registry know about the key. The
# judge-only providers (Anthropic, OpenAI) register theirs the same way, in judge_credentials.
from pup_finetune import judge_credentials as _judge_credentials  # noqa: E402, F401
from pup_finetune.providers import baseten as _baseten  # noqa: E402, F401
from pup_finetune.providers import fireworks as _fireworks  # noqa: E402, F401
