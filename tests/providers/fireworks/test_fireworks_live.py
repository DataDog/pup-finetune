# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Read-only checks against the real Fireworks API (``PUP_FINETUNE_LIVE=1``, a key in the env).

The key is read from ``FIREWORKS_API_KEY`` only, never from the keychain, and nothing is
paid for: preflight, the key check and the account lookup are all GET requests; no training
session is opened.
"""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from pup_finetune.providers.fireworks import validate_key
from pup_finetune.providers.fireworks.client import FireworksTrainingClient
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runtime import Runtime

pytestmark = pytest.mark.live


@pytest.fixture
def api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        pytest.skip("set FIREWORKS_API_KEY to run the live Fireworks tests")
    return key


def test_every_profile_passes_preflight(api_key: str) -> None:
    provider = FireworksProvider(Runtime(), env={"FIREWORKS_API_KEY": api_key})
    for profile in provider.models():
        provider.preflight(profile)


def test_the_key_validates(api_key: str) -> None:
    validate_key(SecretStr(api_key))


def test_the_training_client_resolves_the_account(api_key: str) -> None:
    client = FireworksTrainingClient(SecretStr(api_key), env={})
    try:
        assert client.account_id
        assert "/" not in client.account_id
    finally:
        client.close()
