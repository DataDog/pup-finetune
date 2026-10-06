# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten's cheap registrations: the API key spec and its ``doctor`` checks.

Imported by :mod:`pup_finetune.providers`, so ``auth`` and ``doctor`` know Baseten without
loading the SDK or any heavy module. The checks read package metadata only.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

from pydantic import SecretStr

from pup_finetune import credentials
from pup_finetune.credentials import CredentialSpec
from pup_finetune.doctor import CheckOutcome, register_check
from pup_finetune.errors import ExternalError, UsageError

if TYPE_CHECKING:
    import httpx

__all__ = [
    "HF_INSTALL_HINT",
    "LOOPS_INSTALL_HINT",
    "SPEC",
    "TESTED_LOOPS_VERSION",
    "check_loops_sdk",
    "check_transformers",
    "register",
    "validate_key",
]

TESTED_LOOPS_VERSION = "0.26.0"
LOOPS_INSTALL_HINT = "install the baseten extra: pip install 'pup-finetune[baseten]'"
HF_INSTALL_HINT = "install the hf extra: pip install 'pup-finetune[hf]'"

_MODELS_URL = "https://api.baseten.co/v1/models"
_TIMEOUT_SECONDS = 15.0

# Tests set this to an httpx.MockTransport; None is httpx's real network transport.
_transport: httpx.BaseTransport | None = None


def validate_key(key: SecretStr) -> str | None:
    """Check ``key`` with one read-only Baseten call. Returns no account label."""
    import httpx  # deferred: only `auth login` validates keys

    headers = {"Authorization": f"Api-Key {key.get_secret_value()}"}
    try:
        with (
            httpx.Client(
                transport=_transport, timeout=_TIMEOUT_SECONDS, follow_redirects=False
            ) as client,
            client.stream("GET", _MODELS_URL, headers=headers) as response,
        ):
            status = response.status_code
    except httpx.HTTPError as exc:
        raise ExternalError(
            f"could not reach Baseten to check the API key ({type(exc).__name__})",
            source="baseten",
            hint="check your network connection and retry",
        ) from exc
    if status in (401, 403):
        raise UsageError(
            "Baseten rejected the API key",
            hint="create a new API key in your Baseten account settings and retry",
        )
    if status != 200:
        raise ExternalError(
            f"Baseten returned HTTP {status} while checking the API key", source="baseten"
        )
    return None


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def check_loops_sdk() -> CheckOutcome:
    """Whether the Baseten Loops SDK (needed for training and sampler serving) is installed."""
    version = _installed_version("baseten-loops")
    if version is None:
        return CheckOutcome(False, f"the Baseten Loops SDK is not installed; {LOOPS_INSTALL_HINT}")
    if version != TESTED_LOOPS_VERSION:
        return CheckOutcome(
            True,
            f"baseten-loops {version} is installed; warning: pup-finetune is tested with "
            f"{TESTED_LOOPS_VERSION} ({LOOPS_INSTALL_HINT})",
        )
    return CheckOutcome(True, f"baseten-loops {version} is installed")


def check_transformers() -> CheckOutcome:
    """Whether ``transformers`` (needed to render rows into model tokens) is installed."""
    version = _installed_version("transformers")
    if version is None:
        return CheckOutcome(False, f"transformers is not installed; {HF_INSTALL_HINT}")
    return CheckOutcome(True, f"transformers {version} is installed")


SPEC = CredentialSpec(
    provider="baseten",
    env_var="BASETEN_API_KEY",
    description="Baseten API key (Loops training, samplers, deployments)",
    validator=validate_key,
)


def register() -> None:
    """Register the key spec and checks; idempotent, so tests can re-register after isolating."""
    credentials.register(SPEC)
    register_check("baseten:loops-sdk", check_loops_sdk, required=False)
    register_check("rendering:transformers", check_transformers, required=False)


register()
