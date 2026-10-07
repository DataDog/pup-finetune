# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The API keys of the judge-only providers: Anthropic and OpenAI.

Importing this module is cheap and has one effect: it registers the ``anthropic``
(``ANTHROPIC_API_KEY``, keychain account ``anthropic``) and ``openai`` (``OPENAI_API_KEY``,
keychain account ``openai``) :class:`~pup_finetune.credentials.CredentialSpec`, so ``auth
login``, ``auth status`` and ``doctor`` know about the keys. The providers package imports
it, as it imports the Fireworks package. The validators load the Anthropic SDK or an HTTP
client only when ``auth login`` checks a key; neither makes a paid call (each lists models).

The Anthropic SDK can also sign in without a key from us: with ``ANTHROPIC_AUTH_TOKEN``, a
profile from ``ant auth login``, or workload identity federation. :func:`anthropic_sdk_login`
tells, from the environment and whether the SDK's profile files exist (never their content;
only the ``active_config`` pointer, which holds a profile name, is read), whether it would;
``auth status`` and ``doctor`` then report the source ``anthropic-sdk``, and an Anthropic
judge without a key builds its client without one, so the SDK signs in itself.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import SecretStr

from pup_finetune import credentials
from pup_finetune.errors import ExternalError, UsageError

if TYPE_CHECKING:
    import httpx

__all__ = [
    "ANTHROPIC",
    "ANTHROPIC_ENV_VAR",
    "OPENAI",
    "OPENAI_ENV_VAR",
    "OPENAI_MODELS_URL",
    "anthropic_sdk_login",
    "validate_anthropic_key",
    "validate_openai_key",
]

ANTHROPIC_ENV_VAR: Final = "ANTHROPIC_API_KEY"
OPENAI_ENV_VAR: Final = "OPENAI_API_KEY"
OPENAI_MODELS_URL: Final = "https://api.openai.com/v1/models"

_SDK_AUTH_TOKEN: Final = "ANTHROPIC_AUTH_TOKEN"  # noqa: S105 - a variable name, not a secret
_SDK_PROFILE: Final = "ANTHROPIC_PROFILE"
_SDK_CONFIG_DIR: Final = "ANTHROPIC_CONFIG_DIR"
_SDK_FEDERATION_VARS: Final = ("ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID")
_SDK_IDENTITY_TOKEN_VARS: Final = ("ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE")


def _sdk_config_dir(env: Mapping[str, str]) -> Path:
    """Where the Anthropic SDK and ``ant`` keep their profiles."""
    configured = env.get(_SDK_CONFIG_DIR, "").strip()
    if configured:
        return Path(configured)
    if sys.platform == "win32":
        appdata = env.get("APPDATA", "").strip()
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Anthropic"
    return Path.home() / ".config" / "anthropic"


def _active_config(directory: Path) -> str | None:
    """The profile name ``ant`` made active (``<config dir>/active_config``), if any."""
    try:
        name = (directory / "active_config").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return name or None


def _profile_exists(directory: Path, name: str) -> bool:
    """Whether profile ``name`` has a config file (``<config dir>/configs/<name>.json``)."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return False  # the SDK refuses such a name
    try:
        return (directory / "configs" / f"{name}.json").is_file()
    except OSError:
        return False


def anthropic_sdk_login(env: Mapping[str, str] = os.environ) -> bool:
    """Whether the Anthropic SDK would find a credential of its own in ``env``.

    In the SDK's order: ``ANTHROPIC_AUTH_TOKEN``; a selected profile (``ANTHROPIC_PROFILE``,
    ``ANTHROPIC_CONFIG_DIR`` or an ``active_config`` pointer), only when that profile's
    ``configs/<profile>.json`` exists, as the SDK fails otherwise; workload identity
    federation; or the ``default`` profile on disk. Only names and file existence are
    checked: no secret is read.
    """
    if env.get(_SDK_AUTH_TOKEN, "").strip():
        return True
    directory = _sdk_config_dir(env)
    selected = env.get(_SDK_PROFILE, "").strip()
    pointer = _active_config(directory)
    if selected or env.get(_SDK_CONFIG_DIR, "").strip() or pointer is not None:
        return _profile_exists(directory, selected or pointer or "default")
    if all(env.get(name, "").strip() for name in _SDK_FEDERATION_VARS) and any(
        env.get(name, "").strip() for name in _SDK_IDENTITY_TOKEN_VARS
    ):
        return True
    return _profile_exists(directory, "default")


def validate_anthropic_key(api_key: SecretStr) -> str | None:
    """Check ``api_key`` by listing one Anthropic model (no paid call); no account label.

    Raises :class:`UsageError` when Anthropic rejects the key or the ``anthropic`` extra is
    missing, and :class:`ExternalError` when Anthropic cannot be reached.
    """
    from pup_finetune.judges import anthropic_client_class

    with anthropic_client_class()(api_key, timeout_s=30.0) as client:
        client.check_key()
    return None


def validate_openai_key(
    api_key: SecretStr, *, transport: httpx.BaseTransport | None = None
) -> str | None:
    """Check ``api_key`` by listing OpenAI models (``GET /v1/models``, no paid call).

    Raises :class:`UsageError` when OpenAI rejects the key and :class:`ExternalError` when
    it cannot be reached or answers otherwise; every message is redacted. ``transport``
    replaces the network, for tests.
    """
    import httpx

    from pup_finetune.http_common import CLIENT_SOURCE, bearer_key, error_detail, exception_reason

    hint = credentials.login_hint(OPENAI)
    secret = bearer_key(api_key, label="the OpenAI API key", hint=hint)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Accept": "application/json",
        "User-Agent": CLIENT_SOURCE,
    }
    try:
        with httpx.Client(transport=transport, timeout=30.0, follow_redirects=False) as client:
            response = client.get(OPENAI_MODELS_URL, headers=headers)
    except httpx.HTTPError as exc:
        raise ExternalError(
            f"could not reach OpenAI to check the key ({exception_reason(exc)})",
            source="openai",
            hint="check the connection to api.openai.com and retry",
        ) from exc
    if response.status_code in (401, 403):
        detail = error_detail(response)
        raise UsageError(
            f"OpenAI rejected the API key (HTTP {response.status_code})"
            + (f": {detail}" if detail else ""),
            hint=hint,
        )
    if not response.is_success:
        detail = error_detail(response)
        raise ExternalError(
            f"checking the OpenAI key failed with HTTP {response.status_code}"
            + (f": {detail}" if detail else ""),
            source="openai",
            hint="retry later",
        )
    return None


ANTHROPIC: Final = credentials.CredentialSpec(
    provider="anthropic",
    env_var=ANTHROPIC_ENV_VAR,
    description="Anthropic API key, for LLM judges on Anthropic",
    keychain_account="anthropic",
    sdk_login=anthropic_sdk_login,
    other_secret_env=(_SDK_AUTH_TOKEN, "ANTHROPIC_IDENTITY_TOKEN"),
    validator=validate_anthropic_key,
)

OPENAI: Final = credentials.CredentialSpec(
    provider="openai",
    env_var=OPENAI_ENV_VAR,
    description="OpenAI API key, for LLM judges on OpenAI",
    keychain_account="openai",
    validator=validate_openai_key,
)

credentials.register(ANTHROPIC)
credentials.register(OPENAI)
