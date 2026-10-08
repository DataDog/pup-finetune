# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks provider: model profiles, preflight, training and the API key.

Importing this package is cheap and has one effect: it registers the Fireworks
:class:`~pup_finetune.credentials.CredentialSpec` (``FIREWORKS_API_KEY``, then the OS
keychain), so ``auth login fireworks``, ``auth status`` and ``doctor`` know about the key
without importing an HTTP client or a tokenizer. The provider registry imports it.

- :mod:`.models` is the table of supported models.
- :mod:`.api` is a small read-only client for the Fireworks REST API (no SDK).
- :mod:`.provider` has :class:`~.provider.FireworksProvider`, which the registry builds.
- :mod:`.client` wraps the Fireworks Training SDK (the only module that loads it), and
  :mod:`.backend` trains through it: sessions, checkpoints, promotion and sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import SecretStr

from pup_finetune import credentials
from pup_finetune.errors import UsageError

if TYPE_CHECKING:
    import httpx

__all__ = ["CREDENTIALS", "ENV_VAR", "PROVIDER_NAME", "validate_key"]

PROVIDER_NAME: Final = "fireworks"
ENV_VAR: Final = "FIREWORKS_API_KEY"


def validate_key(api_key: SecretStr, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Check ``api_key`` by listing its Fireworks accounts; the first account id, if any.

    Raises :class:`UsageError` when Fireworks rejects the key and
    :class:`~pup_finetune.errors.ExternalError` when it cannot be reached. ``transport``
    replaces the network, for tests.
    """
    from pup_finetune.providers.fireworks.api import FireworksApi, FireworksAuthError

    try:
        with FireworksApi(api_key, transport=transport) as api:
            accounts = api.list_accounts()
    except FireworksAuthError as exc:
        raise UsageError(f"Fireworks rejected the API key: {exc.message}", hint=exc.hint) from exc
    return accounts[0] if accounts else None


CREDENTIALS: Final = credentials.CredentialSpec(
    provider=PROVIDER_NAME,
    env_var=ENV_VAR,
    description="Fireworks API key, for training, evaluation and deployment on Fireworks",
    needed_for="training, evaluation and deployment on fireworks, and the default judges",
    keychain_account=PROVIDER_NAME,
    validator=validate_key,
)

credentials.register(CREDENTIALS)
