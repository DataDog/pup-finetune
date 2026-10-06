# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The fake provider passes the three contract suites; fake-only behavior is in test_fake.py."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pup_finetune.providers.base import ModelProfile
from pup_finetune.providers.fake import FAKE_PROFILE, FakeBackend, FakeProvider, FakeRenderer
from pup_finetune.runtime import Runtime
from tests.providers.contract import BackendContract, ProviderContract, RendererContract


class TestFakeRenderer(RendererContract):
    @pytest.fixture(params=[True, False], ids=["prefix_stable", "per_turn"])
    def renderer_factory(self, request: pytest.FixtureRequest) -> Callable[[], FakeRenderer]:
        return lambda: FakeRenderer(FAKE_PROFILE, prefix_stable=request.param)


class TestFakeBackend(BackendContract):
    @pytest.fixture
    def backend(self) -> FakeBackend:
        return FakeBackend()

    @pytest.fixture
    def profile(self) -> ModelProfile:
        return FAKE_PROFILE

    @pytest.fixture
    def renderer(self) -> FakeRenderer:
        return FakeRenderer(FAKE_PROFILE)


class _DurableFakeBackend(FakeBackend):
    """A fake whose sampler checkpoints outlive their session, as on some providers."""

    def is_live(self, session_id: str) -> bool:
        return True


class TestDurableFakeBackend(TestFakeBackend):
    @pytest.fixture
    def backend(self) -> FakeBackend:
        return _DurableFakeBackend()


class TestFakeProvider(ProviderContract):
    @pytest.fixture
    def provider(self) -> FakeProvider:
        return FakeProvider(Runtime(dev_mode=True))
