# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Rendering: chat conversations to model tokens, and completions back to messages.

:mod:`pup_finetune.rendering.tokens` defines :class:`TokenizedExample`, the provider-neutral
token arrays a renderer emits; :mod:`pup_finetune.rendering.base` defines the
:class:`Renderer` protocol each model family implements on its Hugging Face chat template.
"""

from __future__ import annotations

from pup_finetune.rendering.base import (
    ChatMessage,
    ParsedResponse,
    Renderer,
    RendererError,
    RendererErrorCode,
    ToolSpec,
)
from pup_finetune.rendering.tokens import TokenizedExample

__all__ = [
    "ChatMessage",
    "ParsedResponse",
    "Renderer",
    "RendererError",
    "RendererErrorCode",
    "TokenizedExample",
    "ToolSpec",
]
