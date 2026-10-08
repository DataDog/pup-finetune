# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A known setup for ``doctor`` tests: no provider keys, chosen packages, a healthy pup.

- :func:`no_provider_keys` removes every registered key variable from the environment and
  points ``HOME`` at an empty directory (so no ``~/.trussrc``); the shared conftest already
  keeps the real keychain out.
- :func:`installed_packages` makes :func:`pup_finetune.providers.installed` and
  :func:`~pup_finetune.providers.module_state` answer from a set of module names (and the
  ones that fail to load) instead of the installed packages.
- :func:`healthy_pup` is a :class:`~tests.fakes.pup.FakePup` that reports pup 1.10.7, signed
  in to datadoghq.com.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from pup_finetune import credentials, providers
from tests.fakes.pup import FakePup

ALL_PACKAGES = frozenset({"fireworks", "tinker", "transformers", "baseten.loops", "anthropic"})
"""Every module the registered providers and judges check for."""

AUTH_OK = {
    "authenticated": True,
    "expires_at": "2026-01-01T00:00:00Z",
    "org": None,
    "scopes": ["synthetic_read"],
    "site": "datadoghq.com",
    "token_type": "Bearer",
}


def no_provider_keys(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """No registered provider key in the environment, and an empty ``HOME``."""
    for spec in credentials.all_specs():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.setenv("HOME", str(home))


def installed_packages(
    monkeypatch: pytest.MonkeyPatch,
    modules: Iterable[str],
    *,
    broken: Mapping[str, str] | None = None,
) -> None:
    """Make the package checks find exactly ``modules``; ``broken`` maps a module that is
    there but fails to load to why."""
    present = frozenset(modules)
    failing = dict(broken or {})

    def state(module: str) -> tuple[providers.ModuleState, str]:
        if module in failing:
            return "broken", failing[module]
        return ("installed", "") if module in present else ("missing", "")

    monkeypatch.setattr(providers, "installed", lambda module: state(module)[0] == "installed")
    monkeypatch.setattr(providers, "module_state", state)


def healthy_pup(version: str = "pup 1.10.7\n") -> FakePup:
    """A pup that reports ``version`` and is signed in to datadoghq.com."""
    fake = FakePup().on("--version", raw=True, stdout=version)
    fake.on(
        "auth", "status", stdout="✅ Authenticated for site: datadoghq.com\n" + json.dumps(AUTH_OK)
    )
    return fake
