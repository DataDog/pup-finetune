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
token, which tag follows an assistant message. A family may also run its tokenizer's own code
from the model's repository at the pinned commit (``trust_remote_code``), render through a
:mod:`~pup_finetune.rendering.encoders` encoder when the tokenizer ships no chat template
(``chat_encoder``), rewrite messages into the form its template renders faithfully
(``rewrite``), and refuse a prompt after which its template opens no assistant turn
(``check``). Adding a family is adding an entry here, plus a parser when its output format
is new. Adapted from the Baseten provider work by Fouad Wahabi.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pup_finetune.rendering.base import ChatMessage
from pup_finetune.rendering.encoders import ENCODERS, get_encoder
from pup_finetune.rendering.hf import (
    COMMIT_SHA,
    ChatTokenizer,
    HFChatTemplateRenderer,
    MessageRewrite,
    PromptCheck,
)
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
    trust_remote_code: bool = False
    """Whether the tokenizer runs code from the model's repository, at :attr:`revision`."""
    chat_encoder: str | None = None
    """The :data:`~pup_finetune.rendering.encoders.ENCODERS` key that renders the chat, for a
    tokenizer that ships no chat template."""
    rewrite: MessageRewrite | None = None
    """How messages are rewritten before the template renders them."""
    check: PromptCheck | None = None
    """Which prompts the template opens no assistant turn after: refused, with any trained turn
    after them."""

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
        if self.chat_encoder is not None and self.chat_encoder not in ENCODERS:
            raise ValueError(
                f"model family {self.key!r} names unknown chat encoder {self.chat_encoder!r}"
            )
        if self.chat_encoder is not None and self.chat_template_kwargs:
            raise ValueError(
                f"model family {self.key!r} renders with an encoder, which takes no "
                "chat-template arguments"
            )
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
            trust_remote_code=self.trust_remote_code,
            chat_encoder=get_encoder(self.chat_encoder) if self.chat_encoder else None,
            rewrite=self.rewrite,
            check=self.check,
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

QWEN3_5 = ModelFamily(
    key="qwen3_5",
    tokenizer_model="Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    parser="qwen3_5_xml",
    stop_tokens=("<|im_end|>",),
    # History drops reasoning before the last user turn, and "<think>\n" merges as in Qwen3.8.
    prefix_stable=False,
)
"""Qwen3.5 with its official template: reasoning kept after the last user turn only."""


def _glm_next_role_tag(message: ChatMessage) -> str:
    return "<|observation|>" if message.get("tool_calls") else "<|user|>"


GLM5_3 = ModelFamily(
    key="glm5_3",
    tokenizer_model="zai-org/GLM-5.3-Flash",
    revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    parser="glm_xml",
    stop_tokens=("<|user|>", "<|observation|>"),
    # Every turn boundary is a special token, so each turn's tokens prefix the whole render.
    prefix_stable=True,
    next_role_tag=_glm_next_role_tag,
)
"""GLM-5.3 with its official template: role tags as stops, ``<arg_key>`` tool calls."""

KIMI_K3 = ModelFamily(
    key="kimi_k3",
    tokenizer_model="moonshotai/Kimi-K3",
    revision="301be1b88c89c0d3a763da6301352cb8fe399e90",
    parser="kimi_xtml",
    stop_tokens=("<|end_of_msg|>",),
    # Every message closes on control tokens and reasoning stays in history, so each turn's
    # tokens prefix the whole render.
    prefix_stable=True,
    trust_remote_code=True,
)
"""Kimi K3 with its repository's tokenizer code, which renders the chat as XTML channels in
Python and encodes message text apart from its control tokens; preserved reasoning, maximum
thinking effort."""


def _deepseek_after_user_or_tool(messages: Sequence[Mapping[str, Any]]) -> str | None:
    # After a system or assistant message, or none, DeepSeek's own encoder writes the turn
    # straight after it with no assistant tag, so the prompt opens no turn to sample from.
    if messages and messages[-1].get("role") in ("user", "tool"):
        return None
    return "DeepSeek-V4's chat format opens an assistant turn only after a user or tool message"


DEEPSEEK_V4 = ModelFamily(
    key="deepseek_v4",
    tokenizer_model="deepseek-ai/DeepSeek-V4-Flash-0731",
    revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    parser="deepseek_dsml",
    stop_tokens=("<｜end▁of▁sentence｜>",),  # noqa: RUF001 - DeepSeek spells it with U+FF5C
    chat_encoder="deepseek_v4",
    # With tools every turn keeps its reasoning and prefixes the whole render; without
    # tools, turns before the last user turn lose theirs, and such rows render per turn.
    prefix_stable=True,
    check=PromptCheck("assistant_after_user_or_tool", "1", _deepseek_after_user_or_tool),
)
"""DeepSeek-V4 Flash 0731, whose repository ships its chat format as Python
(``encoding/encoding_dsv4.py``) and no template: DSML tool calls, thinking mode."""


def _muse_text_before_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    # The template writes only the calls of a message that has both. ATEM writes such text as
    # its own message to the user that does not end the turn (``end_turn``), before the calls.
    if not (message.get("tool_calls") and str(message.get("content") or "").strip()):
        return [dict(message)]
    text = {key: value for key, value in message.items() if key != "tool_calls"}
    calls = {key: value for key, value in message.items() if key not in _MUSE_TEXT}
    return [{**text, "end_turn": False}, calls]


_MUSE_TEXT = frozenset({"content", "reasoning_content"})

MUSE_GLIMMER = ModelFamily(
    key="muse_glimmer",
    tokenizer_model="meta-models/Muse-Glimmer-30B",
    revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
    parser="muse_atem",
    stop_tokens=("<|eot|>",),
    # Without a system message the template writes the day's date into the default one; the
    # date is pinned, to the template's knowledge cutoff, so a row renders the same every day.
    chat_template_kwargs={"current_date": "2026-01-04"},
    # Reasoning stays in history and each turn ends on <|eot|>, so its tokens prefix the
    # whole render.
    prefix_stable=True,
    rewrite=MessageRewrite("muse_text_before_calls", "1", _muse_text_before_calls),
)
"""Muse Glimmer with its official template: ATEM messages to ``self``, ``user`` or a tool,
preserved reasoning, text beside tool calls as its own message to the user, and the date of
the default system message pinned."""

FAMILIES: Mapping[str, ModelFamily] = MappingProxyType(
    {
        family.key: family
        for family in [QWEN3_8, QWEN3_5, GLM5_3, KIMI_K3, DEEPSEEK_V4, MUSE_GLIMMER]
    }
)
"""Every model family by key, as model profiles name them in ``renderer``."""


def get_family(key: str) -> ModelFamily:
    """The family named ``key``; raises ``KeyError`` naming the known ones."""
    try:
        return FAMILIES[key]
    except KeyError:
        raise KeyError(f"unknown model family {key!r}; known: {', '.join(FAMILIES)}") from None
