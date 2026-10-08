# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten: LoRA fine-tuning on Baseten Loops, and the API key.

Importing this package is cheap and has one effect: it registers the Baseten
:class:`~pup_finetune.credentials.CredentialSpec` (``BASETEN_API_KEY``, then the OS keychain,
then, read only, the key of the default remote in ``~/.trussrc``), so ``auth login baseten``,
``auth status`` and ``doctor`` know about the key without importing an HTTP client or a
tokenizer. The provider registry imports it.

- :mod:`.provider` has :class:`~.provider.BasetenProvider`, which the registry builds, and the
  model profiles.
- :mod:`.backend` trains: sessions, checkpoints and samplers.
- :mod:`.client` carries all Baseten traffic; it is the only module that loads the Loops SDK.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import SecretStr

from pup_finetune import credentials
from pup_finetune.errors import UsageError

if TYPE_CHECKING:
    import httpx

__all__ = ["CREDENTIALS", "ENV_VAR", "PROVIDER_NAME", "read_trussrc_key", "validate_key"]

PROVIDER_NAME: Final = "baseten"
ENV_VAR: Final = "BASETEN_API_KEY"

_APP_URL: Final = "https://app.baseten.co"
_TRUSSRC_DEFAULT_REMOTE: Final = "baseten"


def read_trussrc_key(path: Path | None = None) -> str | None:
    """The API key of the default Baseten remote in ``~/.trussrc`` (or ``path``).

    Read only. The default remote is the ``baseten`` section (what Baseten's CLI login
    writes), else the only section. ``None`` when the file is missing or unreadable, there is
    no default remote, it points at another URL or it keeps no inline ``api_key``.
    """
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(path or Path.home() / ".trussrc", encoding="utf-8")
    except (OSError, RuntimeError, UnicodeDecodeError, configparser.Error):
        return None
    sections = config.sections()
    if _TRUSSRC_DEFAULT_REMOTE in sections:
        remote = config[_TRUSSRC_DEFAULT_REMOTE]
    elif len(sections) == 1:
        remote = config[sections[0]]
    else:
        return None
    if remote.get("remote_url", _APP_URL).rstrip("/") != _APP_URL:
        return None
    return remote.get("api_key") or None


def validate_key(api_key: SecretStr, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Check ``api_key`` with one read-only Baseten call (listing models); no account label.

    Raises :class:`UsageError` when Baseten rejects the key or the key is malformed, and
    :class:`~pup_finetune.errors.ExternalError` when Baseten cannot be reached. ``transport``
    replaces the network, for tests.
    """
    from pup_finetune.providers.base import ProviderAuthError
    from pup_finetune.providers.baseten.client import BasetenHTTP

    try:
        with BasetenHTTP(api_key, transport=transport, timeout=15.0) as http:
            http.list_models()
    except ProviderAuthError as exc:
        raise UsageError(
            f"Baseten rejected the API key: {exc.message}",
            hint="create an API key in your Baseten account settings and retry",
        ) from exc
    return None


CREDENTIALS: Final = credentials.CredentialSpec(
    provider=PROVIDER_NAME,
    env_var=ENV_VAR,
    description="Baseten API key, for training and serving on Baseten Loops",
    needed_for="training, evaluation and deployment on baseten",
    keychain_account=PROVIDER_NAME,
    config_reader=read_trussrc_key,
    validator=validate_key,
)

credentials.register(CREDENTIALS)
