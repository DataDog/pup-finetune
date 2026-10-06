# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The rendering contract: token datums, render errors, the tokenizer surface and model families."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class TokenDatum:
    """One training sequence, unshifted. ``weights[i] == 1`` trains on ``tokens[i]``."""

    tokens: tuple[int, ...]
    weights: tuple[int, ...]
    message_index: int
    """Index in ``row.messages`` of the assistant message this datum trains."""


class RenderErrorCode(StrEnum):
    """Why a row cannot be rendered. The values are stable: they are recorded with excluded rows."""

    NO_TRAINABLE_TURN = "no_trainable_turn"
    TEMPLATE_FAILED = "template_failed"
    PREFIX_MISMATCH = "prefix_mismatch"
    NO_TARGET_TOKENS = "no_target_tokens"
    EXCEEDS_CONTEXT = "exceeds_context"
    REASONING_UNSUPPORTED = "reasoning_unsupported"
    INVALID_TOKENS = "invalid_tokens"


class RenderError(ValueError):
    """A row the model's chat template cannot turn into training tokens.

    ``tokens`` and ``limit`` are set for :attr:`RenderErrorCode.EXCEEDS_CONTEXT`.
    """

    def __init__(
        self,
        code: RenderErrorCode,
        detail: str,
        *,
        tokens: int | None = None,
        limit: int | None = None,
    ) -> None:
        self.code = RenderErrorCode(code)
        self.detail = detail
        self.tokens = tokens
        self.limit = limit
        super().__init__(str(self))

    def __reduce__(self) -> tuple[Any, ...]:
        # ``args`` holds only the message, so the default would call ``__init__`` wrongly.
        return (type(self), (self.code, self.detail), {"tokens": self.tokens, "limit": self.limit})

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


class ChatTokenizer(Protocol):
    """The narrow tokenizer surface rendering needs; ``rendering.hf`` adapts the real one."""

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        """Token ids of ``messages`` rendered through the chat template."""
        ...

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text``, without adding special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """The text of ``token_ids``, special tokens included."""
        ...

    def token_id(self, token: str) -> int | None:
        """The id of ``token`` when it is a single special token, else ``None``."""
        ...

    @property
    def chat_template(self) -> str:
        """The chat template source."""
        ...


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """How one model family's chat template is pinned, rendered and stopped."""

    key: str
    tokenizer_repo: str
    revision: str
    """The tokenizer repo's 40-hex commit sha."""
    max_seq_len: int
    supports_reasoning: bool
    template_kwargs: Mapping[str, Any]
    boundary: Literal["none", "glm_stop_tag"]
    """``glm_stop_tag``: the model stops by emitting the next role tag, so it is a target."""
    stop_sequences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _SHA.fullmatch(self.revision):
            raise ValueError(f"model family {self.key!r} needs a 40-hex commit revision")
        if self.max_seq_len < 1:
            raise ValueError(f"model family {self.key!r} needs a positive max_seq_len")


@dataclass(frozen=True, slots=True)
class RenderingIdentity:
    """What fixes the rendered tokens; a run records it so a later change is detected."""

    version: str
    tokenizer_repo: str
    revision: str
    template_sha256: str
    template_kwargs: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tokenizer_repo": self.tokenizer_repo,
            "revision": self.revision,
            "template_sha256": self.template_sha256,
            "template_kwargs": dict(self.template_kwargs),
        }
