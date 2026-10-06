# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The :class:`Renderer` protocol: chat conversations in, model tokens out, and back again.

A renderer owns tokenization and response parsing for one model family. Renderers are
provider-neutral: they are built on the model's Hugging Face chat template and tokenizer, and
emit :class:`~pup_finetune.rendering.tokens.TokenizedExample` arrays that each training
backend converts to its SDK's datum type. Fireworks and Baseten serve the same open-weight
models, so they share the same renderer.

Messages and tools are in OpenAI chat format, as
:func:`~pup_finetune.data.jsonl.to_provider_dict` produces them: an assistant message may
carry ``weight`` ``0``, meaning "do not train on this turn".

**Prefix stability.** A renderer is prefix-stable when, for every assistant turn ``i``, the
tokens of :meth:`Renderer.generation_prompt_tokens` for ``messages[:i]`` are a prefix of the
tokens of the full conversation. Only then does training on one full render (every assistant
turn trainable at once) match what the model sees at inference. Chat templates are often
prefix-stable as text but not as tokens, because a tokenizer can merge characters across the
boundary where generation starts; renderers therefore tokenize segment-wise, splitting the
render at each generation-prompt end and assistant-turn end and tokenizing the segments
independently, which is how inference tokenizes them. A renderer that still cannot guarantee
it declares ``prefix_stable = False``, and the training loop renders each assistant turn
against its own prefix with :meth:`Renderer.per_turn_examples` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pup_finetune.errors import FinetuneError
from pup_finetune.rendering.tokens import TokenizedExample

ChatMessage = dict[str, Any]
"""One message in OpenAI chat format."""

ToolSpec = dict[str, Any]
"""One declared tool in OpenAI chat format: ``{"type": "function", "function": {...}}``."""


class RendererErrorCode(StrEnum):
    """Why rendering failed. The values are stable: they may be recorded with excluded rows."""

    NOT_PREFIX_STABLE = "not_prefix_stable"
    """A generation prompt is not a token prefix of the full render."""
    TEMPLATE_ERROR = "template_error"
    """The chat template raised while rendering the conversation."""
    UNSUPPORTED_MESSAGE = "unsupported_message"
    """The conversation holds something the template cannot express."""
    NO_TRAINABLE_TOKENS = "no_trainable_tokens"
    """Every assistant turn has ``weight`` ``0``, so nothing would be trained on."""
    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"
    """The pinned tokenizer or chat template could not be loaded."""


class RendererError(FinetuneError):
    """A conversation that could not be rendered. ``code`` says why; ``detail`` says where."""

    def __init__(self, code: RendererErrorCode | str, detail: str) -> None:
        self.code = RendererErrorCode(code)
        self.detail = detail
        super().__init__(f"{self.code.value}: {detail}")

    def __reduce__(self) -> tuple[Any, ...]:
        # ``args`` holds only the message, so the default would call ``__init__`` wrongly.
        return (type(self), (self.code, self.detail))


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """A completion parsed back into an OpenAI assistant message.

    ``format_valid`` is false when the completion broke the model's output format (an
    unclosed tool call, arguments that are not a JSON object, ...); ``message`` then holds the
    best-effort parse, usually the raw text as ``content``.
    """

    message: ChatMessage
    format_valid: bool


class Renderer(Protocol):
    """Tokenization and response parsing for one model family. See the module docstring."""

    @property
    def name(self) -> str:
        """The renderer's name, as model profiles refer to it."""
        ...

    @property
    def prefix_stable(self) -> bool:
        """Whether generation prompts are token prefixes of the full render."""
        ...

    @property
    def stop_token_ids(self) -> list[int]:
        """Token ids that end an assistant turn; samplers stop on any of them."""
        ...

    def supervised_example(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> TokenizedExample:
        """The whole conversation as one example, every assistant turn trainable.

        Assistant turns with ``weight`` ``0`` get zero weights. Raises :class:`RendererError`.
        """
        ...

    def per_turn_examples(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[TokenizedExample]:
        """One example per trainable assistant turn, each rendered against its own prefix.

        In each example only that turn is trainable. Used when :attr:`prefix_stable` is false.
        Raises :class:`RendererError`.
        """
        ...

    def generation_prompt_tokens(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[int]:
        """The prompt a sampler continues to generate the next assistant turn."""
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Token ids back to text, special tokens included."""
        ...

    def parse_response(self, text: str) -> ParsedResponse:
        """A decoded completion (the tokens after the generation prompt) as a message."""
        ...

    def identity(self) -> dict[str, str]:
        """What pins this renderer's output, recorded in a run so a resume renders the same.

        Includes at least the tokenizer, its revision, the chat template's hash and the
        renderer and parser versions. A mismatch on resume means "prepare a new run".
        """
        ...
