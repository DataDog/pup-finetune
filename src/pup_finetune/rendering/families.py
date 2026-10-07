# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The model families we render for, each pinned to a tokenizer commit.

A family is a table entry: the Hugging Face tokenizer and its 40-character revision, the
response parser (by name, from :data:`~pup_finetune.rendering.parsers.PARSERS`), the stop
tokens, any chat-template arguments, whether the one-example-per-row fast path is worth
trying, and, for models that stop by emitting the next role tag instead of an end-of-turn
token, which tag follows an assistant message. Adding a family is adding an entry here, plus
a parser when its output format is new. Adapted from the Baseten provider work by Fouad
Wahabi.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pup_finetune.rendering.base import ChatMessage
from pup_finetune.rendering.hf import COMMIT_SHA, ChatTokenizer, HFChatTemplateRenderer
from pup_finetune.rendering.parsers import PARSERS, get_parser


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """How one model family's chat template is pinned, rendered, stopped and parsed."""

    key: str
    tokenizer_model: str
    revision: str
    """The tokenizer repository's full 40-character lowercase commit SHA."""
    parser: str
    stop_tokens: tuple[str, ...]
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    prefix_stable: bool = False
    """Whether to try one example per row; each row is still checked when rendered."""
    next_role_tag: Callable[[ChatMessage], str] | None = None
    """For models that stop by emitting the next role tag: the tag after a message."""

    def __post_init__(self) -> None:
        if not COMMIT_SHA.fullmatch(self.revision):
            raise ValueError(
                f"model family {self.key!r} needs a full 40-character lowercase commit SHA, "
                f"got {self.revision!r}"
            )
        if self.parser not in PARSERS:
            raise ValueError(f"model family {self.key!r} names unknown parser {self.parser!r}")
        if not self.stop_tokens:
            raise ValueError(f"model family {self.key!r} needs at least one stop token")
        object.__setattr__(
            self, "chat_template_kwargs", MappingProxyType(dict(self.chat_template_kwargs))
        )

    def renderer(self, *, tokenizer: ChatTokenizer | None = None) -> HFChatTemplateRenderer:
        """A renderer for this family; ``tokenizer`` injects one instead of loading it."""
        return HFChatTemplateRenderer(
            self.tokenizer_model,
            self.revision,
            response_parser=get_parser(self.parser),
            stop_tokens=self.stop_tokens,
            prefix_stable=self.prefix_stable,
            chat_template_kwargs=self.chat_template_kwargs,
            next_role_tag=self.next_role_tag,
            name=self.key,
            tokenizer=tokenizer,
        )


QWEN3_8 = ModelFamily(
    key="qwen3_8",
    tokenizer_model="Qwen/Qwen3.8-27B",
    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    parser="qwen3_xml",
    stop_tokens=("<|im_end|>",),
    # Prefix-stable as text but not as tokens: the generation prompt ends "<think>\n" and the
    # whole render merges that newline with the next one, so every row renders per turn.
    prefix_stable=False,
)
"""Qwen3.8 with its official template: preserved reasoning history, XML tool calls."""

FAMILIES: Mapping[str, ModelFamily] = MappingProxyType({family.key: family for family in [QWEN3_8]})
"""Every model family by key, as model profiles name them in ``renderer``."""


def get_family(key: str) -> ModelFamily:
    """The family named ``key``; raises ``KeyError`` naming the known ones."""
    try:
        return FAMILIES[key]
    except KeyError:
        raise KeyError(f"unknown model family {key!r}; known: {', '.join(FAMILIES)}") from None
