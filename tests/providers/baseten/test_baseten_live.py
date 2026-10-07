# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The backend and provider contracts against the real Baseten Loops (``PUP_FINETUNE_LIVE=1``).

The backend suite is paid: it trains real, billed Loops runs and samplers with the real
``qwen3_5`` tokenizer. The provider suite is read only: preflight and checks ask Baseten, and
renderers load the real tokenizers. Both need the ``baseten`` and ``hf`` extras and a key in
``BASETEN_API_KEY`` (never the keychain). Every session deactivates its run and samplers when
it ends.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
from pydantic import SecretStr

from pup_finetune.providers.base import ModelProfile, Provider
from pup_finetune.providers.baseten.backend import BasetenBackend
from pup_finetune.providers.baseten.provider import BasetenProvider
from pup_finetune.rendering.families import QWEN3_5
from pup_finetune.rendering.hf import HFChatTemplateRenderer
from pup_finetune.runtime import Runtime
from tests.providers.baseten import test_baseten_backend as on_fake
from tests.providers.baseten import test_baseten_provider as provider_on_fake

pytestmark = pytest.mark.live


class TestBasetenBackendLive(on_fake.TestBasetenBackendContract):
    @pytest.fixture
    def renderer(self) -> HFChatTemplateRenderer:
        pytest.importorskip("transformers", reason="needs the hf extra")
        return QWEN3_5.renderer()

    @pytest.fixture
    def profile(self) -> ModelProfile:
        return dataclasses.replace(on_fake.PROFILE, max_seq_len=4096)  # the smallest trainers

    @pytest.fixture
    def backend(self, renderer: HFChatTemplateRenderer) -> BasetenBackend:
        pytest.importorskip("baseten.loops", reason="needs the baseten extra")
        key = os.environ.get("BASETEN_API_KEY", "").strip()
        if not key:
            pytest.skip("set BASETEN_API_KEY to run the live Baseten tests")
        return BasetenBackend(lambda: SecretStr(key), renderer=lambda _: renderer)


class TestBasetenProviderLive(provider_on_fake.TestBasetenProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        pytest.importorskip("baseten.loops", reason="needs the baseten extra")
        pytest.importorskip("transformers", reason="needs the hf extra")
        key = os.environ.get("BASETEN_API_KEY", "").strip()
        if not key:
            pytest.skip("set BASETEN_API_KEY to run the live Baseten tests")
        return BasetenProvider(Runtime(), env={"BASETEN_API_KEY": key})
