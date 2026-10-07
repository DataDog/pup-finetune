# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Rows in, training examples out: render each row and reject the ones that cannot train.

A prefix-stable renderer gives one example per row with every assistant turn trainable; any
other renderer, or a prefix-stable one refusing a row with ``not_prefix_stable``, gives one
example per trainable assistant turn, each against its own prefix (see
:mod:`pup_finetune.rendering.base`). A row's examples stay together: the batcher
shuffles and batches whole rows. A row is rejected whole, never truncated or repaired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pup_finetune.data.jsonl import to_provider_dict
from pup_finetune.data.schema import TrainingRow
from pup_finetune.rendering.base import Renderer, RendererError, RendererErrorCode
from pup_finetune.rendering.tokens import TokenizedExample


class RejectionCode(StrEnum):
    """Why a row was left out at train time, besides a renderer's own codes. Stable values."""

    EXCEEDS_CONTEXT = "exceeds_context"
    """An example of the row has more tokens than the model's training context."""
    NO_TRAINABLE_TOKENS = "no_trainable_tokens"
    """No target token of the row carries weight, so it would train on nothing."""


@dataclass(frozen=True, slots=True)
class RowExamples:
    """The examples rendered from one row, batched together."""

    row_id: str
    examples: tuple[TokenizedExample, ...]

    @property
    def num_tokens(self) -> int:
        """The rendered tokens of every example of the row."""
        return sum(example.num_tokens for example in self.examples)

    @property
    def num_trainable(self) -> int:
        """The target tokens with a positive weight, over every example of the row."""
        return sum(example.num_trainable for example in self.examples)


@dataclass(frozen=True, slots=True)
class RejectedRow:
    """A row left out of training: its id, a stable ``code`` and a human ``detail``."""

    row_id: str
    code: str
    detail: str

    def to_json(self) -> dict[str, str]:
        return {"row_id": self.row_id, "code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class BuildResult:
    """The rows that render within the context, the rejected ones, and the tokens per epoch."""

    examples: list[RowExamples]
    rejected: list[RejectedRow]
    train_tokens: int


def provider_row(row: TrainingRow) -> dict[str, Any]:
    """``row`` in the form :func:`build_examples` takes: provider form plus its ``id``."""
    return {"id": row.id, **to_provider_dict(row)}


def build_examples(
    rows: Sequence[Mapping[str, Any]], renderer: Renderer, *, max_seq_len: int
) -> BuildResult:
    """Render ``rows`` (provider form, see :func:`provider_row`) into training examples.

    A row is rejected with ``exceeds_context`` when any of its examples has more than
    ``max_seq_len`` tokens, with ``no_trainable_tokens`` when none of its target tokens
    carries weight, and with the renderer's code when the renderer raises
    :class:`RendererError`. ``train_tokens`` counts the tokens of the accepted rows: what one
    epoch trains on. A row without an ``id`` is named by its position, ``#<index>``.
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    accepted: list[RowExamples] = []
    rejected: list[RejectedRow] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or f"#{index}")
        try:
            examples = _render(row, renderer)
        except RendererError as exc:
            rejected.append(RejectedRow(row_id, exc.code.value, exc.detail))
            continue
        longest = max((example.num_tokens for example in examples), default=0)
        built = RowExamples(row_id, tuple(examples))
        if longest > max_seq_len:
            rejected.append(
                RejectedRow(
                    row_id,
                    RejectionCode.EXCEEDS_CONTEXT.value,
                    f"an example has {longest} tokens, over the {max_seq_len}-token context",
                )
            )
        elif built.num_trainable == 0:
            rejected.append(
                RejectedRow(
                    row_id,
                    RejectionCode.NO_TRAINABLE_TOKENS.value,
                    "no target token carries weight",
                )
            )
        else:
            accepted.append(built)
    return BuildResult(
        examples=accepted,
        rejected=rejected,
        train_tokens=sum(row.num_tokens for row in accepted),
    )


def _render(row: Mapping[str, Any], renderer: Renderer) -> list[TokenizedExample]:
    messages = list(row.get("messages") or [])
    tools = list(row.get("tools") or [])
    if renderer.prefix_stable:
        try:
            return [renderer.supervised_example(messages, tools)]
        except RendererError as exc:
            if exc.code is not RendererErrorCode.NOT_PREFIX_STABLE:
                raise
    return renderer.per_turn_examples(messages, tools)
