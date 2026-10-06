# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The model families we render for, each pinned to a tokenizer commit.

Both use their official chat templates with no extra template arguments. Their generation
prompts open the reasoning channel, so prefix-diff rendering keeps that opener out of the loss.
"""

from __future__ import annotations

from types import MappingProxyType

from pup_finetune.rendering.types import ModelFamily

FAMILIES: dict[str, ModelFamily] = {
    "qwen3_5": ModelFamily(
        key="qwen3_5",
        tokenizer_repo="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        max_seq_len=131_072,
        supports_reasoning=True,
        template_kwargs=MappingProxyType({}),
        boundary="none",
        stop_sequences=("<|im_end|>",),
    ),
    "glm5_3": ModelFamily(
        key="glm5_3",
        tokenizer_repo="zai-org/GLM-5.3-Flash",
        revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        max_seq_len=131_072,
        supports_reasoning=True,
        template_kwargs=MappingProxyType({}),
        # GLM stops on the next role tag: <|observation|> after tool calls, else <|user|>.
        boundary="glm_stop_tag",
        stop_sequences=("<|user|>", "<|observation|>"),
    ),
}
