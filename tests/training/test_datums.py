# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

from typing import Any

import pytest

from pup_finetune.data.jsonl import to_provider_dict
from pup_finetune.data.schema import validate_row
from pup_finetune.providers.fake import FakeRenderer
from pup_finetune.rendering.base import ChatMessage, ToolSpec
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.training import RejectedRow, RejectionCode, build_examples, provider_row
from tests.fakes.training import chat_rows


class _ZeroWeightRenderer(FakeRenderer):
    """Renders like the fake but trains on nothing, without raising."""

    def supervised_example(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> TokenizedExample:
        example = super().supervised_example(messages, tools)
        return TokenizedExample(
            example.input_tokens, example.target_tokens, (0.0,) * len(example.weights)
        )


def test_a_prefix_stable_renderer_gives_one_example_per_row() -> None:
    rows = chat_rows(3, turns=2)
    renderer = FakeRenderer()
    result = build_examples(rows, renderer, max_seq_len=4096)
    assert [row.row_id for row in result.examples] == ["row-0", "row-1", "row-2"]
    assert all(len(row.examples) == 1 for row in result.examples)
    expected = renderer.supervised_example(rows[0]["messages"], [])
    assert result.examples[0].examples == (expected,)
    assert result.rejected == []
    assert result.train_tokens == sum(row.num_tokens for row in result.examples)


def test_a_renderer_that_is_not_prefix_stable_expands_each_turn() -> None:
    rows = chat_rows(2, turns=3)
    renderer = FakeRenderer(prefix_stable=False)
    result = build_examples(rows, renderer, max_seq_len=4096)
    assert [len(row.examples) for row in result.examples] == [3, 3]
    first = result.examples[0]
    assert first.examples == tuple(renderer.per_turn_examples(rows[0]["messages"], []))
    # Each example trains only its own turn, so every turn is trained exactly once.
    stable = build_examples(rows, FakeRenderer(), max_seq_len=4096).examples[0]
    assert first.num_trainable == stable.num_trainable
    assert first.num_tokens > stable.num_tokens


def test_a_row_over_the_context_is_rejected_whole() -> None:
    short, long = chat_rows(1, prefix="short")[0], chat_rows(1, turns=4, prefix="long")[0]
    renderer = FakeRenderer(prefix_stable=False)
    fits = max(ex.num_tokens for ex in renderer.per_turn_examples(short["messages"], []))
    result = build_examples([short, long], renderer, max_seq_len=fits)
    # The long row's first turns fit too, but one example does not: the whole row goes.
    assert [row.row_id for row in result.examples] == ["short-0"]
    [rejected] = result.rejected
    assert (rejected.row_id, rejected.code) == ("long-0", "exceeds_context")
    assert f"over the {fits}-token context" in rejected.detail
    assert result.train_tokens == result.examples[0].num_tokens


def test_a_row_at_exactly_the_context_is_kept() -> None:
    row = chat_rows(1)[0]
    size = FakeRenderer().supervised_example(row["messages"], []).num_tokens
    assert build_examples([row], FakeRenderer(), max_seq_len=size).rejected == []
    assert build_examples([row], FakeRenderer(), max_seq_len=size - 1).examples == []


def test_a_row_with_only_untrained_turns_is_rejected() -> None:
    row: dict[str, Any] = {
        "id": "context-only",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "weight": 0},
        ],
    }
    result = build_examples([row], FakeRenderer(), max_seq_len=4096)
    assert result.examples == []
    assert result.rejected == [
        RejectedRow(
            "context-only", "no_trainable_tokens", "no assistant turn has a non-zero weight"
        )
    ]


def test_a_row_whose_examples_carry_no_weight_is_rejected() -> None:
    result = build_examples(chat_rows(1), _ZeroWeightRenderer(), max_seq_len=4096)
    assert result.examples == []
    [rejected] = result.rejected
    assert rejected.code == RejectionCode.NO_TRAINABLE_TOKENS.value
    assert result.train_tokens == 0


def test_a_renderer_error_rejects_the_row_with_its_code() -> None:
    bad = {
        "id": "marker",
        "messages": [
            {"role": "user", "content": "<|end|>"},
            {"role": "assistant", "content": "ok"},
        ],
    }
    result = build_examples([bad, *chat_rows(1)], FakeRenderer(), max_seq_len=4096)
    assert [row.row_id for row in result.examples] == ["row-0"]
    [rejected] = result.rejected
    assert (rejected.row_id, rejected.code) == ("marker", "unsupported_message")
    assert rejected.to_json() == {
        "row_id": "marker",
        "code": "unsupported_message",
        "detail": rejected.detail,
    }


def test_a_row_without_an_id_is_named_by_position() -> None:
    row = {"messages": chat_rows(1)[0]["messages"]}
    result = build_examples([row], FakeRenderer(), max_seq_len=4096)
    assert result.examples[0].row_id == "#0"


def test_tools_are_rendered() -> None:
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    row = {**chat_rows(1)[0], "tools": [tool]}
    with_tools = build_examples([row], FakeRenderer(), max_seq_len=4096).examples[0]
    without = build_examples(chat_rows(1), FakeRenderer(), max_seq_len=4096).examples[0]
    assert with_tools.num_tokens > without.num_tokens


def test_provider_row_adds_the_id_to_the_provider_form() -> None:
    row = validate_row(
        {
            "id": "r1",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
    )
    assert provider_row(row) == {"id": "r1", **to_provider_dict(row)}


def test_max_seq_len_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        build_examples([], FakeRenderer(), max_seq_len=0)
