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

**Prefix stability.** A renderer is prefix-stable when, for every trained assistant turn
``i``, the tokens of :meth:`Renderer.generation_prompt_tokens` for ``messages[:i]`` are a
prefix of the tokens of the full conversation. Only then does training on one full render
(every trained turn at once) match what the model sees at inference; an untrained turn carries
no loss, so how the whole render shows it does not matter. A chat endpoint tokenizes the
whole prompt text as one string and the model generates the completion after it, so a
template that is prefix-stable as text can still fail as tokens: the tokenizer can merge
characters across the boundary where generation starts. A renderer that cannot guarantee it
declares ``prefix_stable = False``, and the training loop renders each assistant turn against
its own prefix with :meth:`Renderer.per_turn_examples` instead. A renderer that declares it may
still refuse one row by raising :class:`RendererError` with ``not_prefix_stable`` from
:meth:`Renderer.supervised_example`; the loop then renders that row per turn.
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
    """A generation prompt is not a token prefix of the full render.

    Raised only by :meth:`Renderer.supervised_example` to refuse one row, which callers then
    render per turn; a per-turn render never raises it.
    """
    TEMPLATE_ERROR = "template_error"
    """The chat template raised while rendering the conversation."""
    PROMPT_MISMATCH = "prompt_mismatch"
    """A turn's render does not continue its own generation prompt, as text or as tokens."""
    UNSUPPORTED_MESSAGE = "unsupported_message"
    """The conversation holds something the template cannot express."""
    NO_TRAINABLE_TOKENS = "no_trainable_tokens"
    """Every assistant turn has ``weight`` ``0``, so nothing would be trained on."""
    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"
    """Reserved. A tokenizer or chat template that cannot be loaded fails every row alike, so
    renderers report it as an :class:`~pup_finetune.errors.ExternalError` (or a
    :class:`~pup_finetune.errors.UsageError` for a missing extra), not as a per-row error."""


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

    def check_tokens(self) -> None:
        """Raise :class:`~pup_finetune.errors.UsageError` when a stop token, or a control
        token of the response parser, is not one token of the tokenizer, as the first sampler
        would."""
        ...

    def supervised_example(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> TokenizedExample:
        """The whole conversation as one example, every assistant turn trainable.

        Assistant turns with ``weight`` ``0`` get zero weights. Raises :class:`RendererError`;
        with code ``not_prefix_stable`` when this row cannot be one exact example, so callers
        fall back to :meth:`per_turn_examples`.
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

    def prompt_refusal(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> RendererError | None:
        """The :class:`RendererError` :meth:`generation_prompt_tokens` raises for ``messages``
        because of what they hold, or ``None``; it loads no tokenizer, so planning can leave
        out a case no sampler could continue. Training refuses a trained turn after such a
        prompt with the same error."""
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Token ids back to text, special tokens included."""
        ...

    def parse_response(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        """A decoded completion (the tokens after the generation prompt) as a message.

        ``tools`` are the tools the prompt declared. Pass them: a format that writes some
        argument values untyped (Qwen3.8's XML tool calls write strings unquoted) reads each
        one with the type its tool declares. Never raises.
        """
        ...

    def parse_completion(
        self, tokens: Sequence[int], tools: Sequence[ToolSpec] = ()
    ) -> ParsedResponse:
        """A sampled completion's token ids as a message, as :meth:`parse_response` reads
        their text, except that a format whose control tokens a reply can spell as text
        (Kimi K3) takes structure only from the control tokens sampled. Never raises."""
        ...

    def identity(self) -> dict[str, str]:
        """What pins this renderer's output, recorded in a run so a resume renders the same.

        Includes at least the tokenizer, its revision, the chat template's hash and the
        renderer and parser versions. A mismatch on resume means "prepare a new run".
        """
        ...
