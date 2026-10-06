# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Hugging Face chat-template renderer passes the renderer contract.

Offline against the fake tokenizer: token prefix-stable with the fast path on, merging
newlines across the prompt boundary (per turn), and both (the fast path refuses every row,
which then renders per turn). With ``PUP_FINETUNE_TOKENIZER_TESTS=1`` and
the ``hf`` extra, also against the real, pinned Qwen3.8 tokenizer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from pup_finetune.rendering.families import QWEN3_8, ModelFamily
from pup_finetune.rendering.hf import HFChatTemplateRenderer, TransformersTokenizer
from tests.fakes.tokenizer import FakeChatTokenizer
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
