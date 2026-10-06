# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Rendering: turn canonical chat rows into model tokens and per-token loss weights.

:mod:`pup_finetune.rendering.types` holds the contract, :mod:`pup_finetune.rendering.families`
the pinned model families, :mod:`pup_finetune.rendering.render` the rendering itself and
:mod:`pup_finetune.rendering.hf` the Hugging Face tokenizers (``transformers`` loads lazily).
"""

from __future__ import annotations

from pup_finetune.rendering.families import FAMILIES
from pup_finetune.rendering.hf import RENDERING_VERSION, load_tokenizer, rendering_identity
from pup_finetune.rendering.render import RenderReport, check_rows, prompt_tokens, render_row
from pup_finetune.rendering.types import (
    ChatTokenizer,
    ModelFamily,
    RenderError,
    RenderErrorCode,
    RenderingIdentity,
    TokenDatum,
)

__all__ = [
    "FAMILIES",
    "RENDERING_VERSION",
    "ChatTokenizer",
    "ModelFamily",
    "RenderError",
    "RenderErrorCode",
    "RenderReport",
    "RenderingIdentity",
    "TokenDatum",
    "check_rows",
    "load_tokenizer",
    "prompt_tokens",
    "render_row",
    "rendering_identity",
]
