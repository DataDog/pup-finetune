# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Hugging Face chat-template renderer passes the renderer contract.

Offline against the fake tokenizer: token prefix-stable with the fast path on, merging
newlines across the prompt boundary (per turn), and both (the fast path refuses every row,
which then renders per turn); and the ``qwen3_5`` and ``glm5_3`` families on fake templates.
With ``PUP_FINETUNE_TOKENIZER_TESTS=1`` and the ``hf`` extra, also against the real, pinned
Qwen3.8, Qwen3.5, GLM-5.3, Kimi K3, DeepSeek-V4 and Muse Glimmer tokenizers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from pup_finetune.rendering import ChatMessage
from pup_finetune.rendering.families import (
    DEEPSEEK_V4,
    GLM5_3,
    KIMI_K3,
    MUSE_GLIMMER,
    QWEN3_5,
    QWEN3_8,
    ModelFamily,
)
from pup_finetune.rendering.hf import HFChatTemplateRenderer, TransformersTokenizer
from tests.fakes.tokenizer import FakeChatTokenizer, glm5_like
from tests.providers.contract import RendererContract

_FAKE_FAMILY = ModelFamily(
    key="fake_qwen3_8",
    tokenizer_model="example/tiny-chat",
    revision="0123456789abcdef0123456789abcdef01234567",
    parser="qwen3_xml",
    stop_tokens=("<|im_end|>",),
)


class TestHFChatTemplateRendererOnAFakeTokenizer(RendererContract):
    @pytest.fixture(
        params=[(True, False), (False, True), (True, True)],
        ids=["fast_path", "per_turn_merging", "fast_path_refused"],
    )
    def renderer_factory(
        self, request: pytest.FixtureRequest
    ) -> Callable[[], HFChatTemplateRenderer]:
        stable, merge = request.param
        family = dataclasses.replace(_FAKE_FAMILY, prefix_stable=stable)
        return lambda: family.renderer(tokenizer=FakeChatTokenizer(merge_newlines=merge))


@pytest.mark.tokenizer
class TestHFChatTemplateRendererOnQwen3_8(RendererContract):
    @pytest.fixture
    def renderer_factory(
        self, qwen3_8_tokenizer: TransformersTokenizer
    ) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: QWEN3_8.renderer(tokenizer=qwen3_8_tokenizer)


class TestQwen3_5OnAFakeTokenizer(RendererContract):
    @pytest.fixture
    def renderer_factory(self) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: QWEN3_5.renderer(tokenizer=FakeChatTokenizer(merge_newlines=True))


class TestGlm5_3OnAFakeTokenizer(RendererContract):
    @pytest.fixture
    def renderer_factory(self) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: GLM5_3.renderer(tokenizer=FakeChatTokenizer(template=glm5_like))


@pytest.mark.tokenizer
class TestHFChatTemplateRendererOnQwen3_5(RendererContract):
    @pytest.fixture
    def renderer_factory(
        self, qwen3_5_tokenizer: TransformersTokenizer
    ) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: QWEN3_5.renderer(tokenizer=qwen3_5_tokenizer)


@pytest.mark.tokenizer
class TestHFChatTemplateRendererOnGlm5_3(RendererContract):
    @pytest.fixture
    def renderer_factory(
        self, glm5_3_tokenizer: TransformersTokenizer
    ) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: GLM5_3.renderer(tokenizer=glm5_3_tokenizer)


@pytest.mark.tokenizer
class TestHFChatTemplateRendererOnKimiK3(RendererContract):
    @pytest.fixture
    def renderer_factory(
        self, kimi_k3_tokenizer: TransformersTokenizer
    ) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: KIMI_K3.renderer(tokenizer=kimi_k3_tokenizer)


@pytest.mark.tokenizer
class TestHFChatTemplateRendererOnDeepSeekV4(RendererContract):
    @pytest.fixture
    def renderer_factory(
        self, deepseek_v4_tokenizer: TransformersTokenizer
    ) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: DEEPSEEK_V4.renderer(tokenizer=deepseek_v4_tokenizer)


@pytest.mark.tokenizer
class TestHFChatTemplateRendererOnMuseGlimmer(RendererContract):
    @pytest.fixture
    def renderer_factory(
        self, muse_glimmer_tokenizer: TransformersTokenizer
    ) -> Callable[[], HFChatTemplateRenderer]:
        return lambda: MUSE_GLIMMER.renderer(tokenizer=muse_glimmer_tokenizer)

    @pytest.fixture
    def preamble(self) -> list[ChatMessage]:
        return [{"role": "system", "content": "You help."}]
