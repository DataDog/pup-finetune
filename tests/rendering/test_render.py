# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from pup_finetune.data.jsonl import read_rows, to_provider_dict
from pup_finetune.data.schema import TrainingRow, validate_row
from pup_finetune.rendering import (
    FAMILIES,
    ModelFamily,
    RenderError,
    RenderErrorCode,
    TokenDatum,
    check_rows,
    prompt_tokens,
    render_row,
)
from tests.fakes.tokenizer import FakeChatTokenizer

FIXTURE_ROWS = list(read_rows(Path(__file__).parents[1] / "fixtures" / "rows" / "tool_use.jsonl"))
QWEN = FAMILIES["qwen3_5"]
GLM = FAMILIES["glm5_3"]
LIMIT = 1_000_000
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a city up.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def make_row(
    *messages: dict[str, Any], tools: Sequence[Any] = (), row_id: str = "row"
) -> TrainingRow:
    return validate_row({"id": row_id, "messages": list(messages), "tools": list(tools)})


def user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def assistant(content: str = "", **fields: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": content, **fields}


def call(call_id: str, city: str) -> dict[str, Any]:
    arguments = json.dumps({"city": city})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": arguments},
    }


def tool(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def template_messages(row: TrainingRow, end: int) -> list[dict[str, Any]]:
    messages = []
    for message in row.messages[:end]:
        data = message.model_dump(mode="json", exclude_none=True, exclude={"weight"})
        for tool_call in data.get("tool_calls", []):
            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
        messages.append(data)
    return messages


def row_tools(row: TrainingRow) -> list[dict[str, Any]] | None:
    return [t.model_dump(mode="json", exclude_none=True) for t in row.tools] or None


def target_text(tokenizer: FakeChatTokenizer, datum: TokenDatum) -> str:
    return tokenizer.decode([t for t, w in zip(datum.tokens, datum.weights, strict=True) if w])


def weighted_spans(datum: TokenDatum) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, weight in enumerate((*datum.weights, 0)):
        if weight and start is None:
            start = index
        elif not weight and start is not None:
            spans.append((start, index))
            start = None
    return spans


def trained_indexes(row: TrainingRow) -> list[int]:
    return [
        index
        for index, m in enumerate(row.messages)
        if m.role == "assistant"
        and m.weight != 0
        and (m.content or m.tool_calls or m.reasoning_content)
    ]


SETUPS: dict[str, tuple[ModelFamily, Callable[[], FakeChatTokenizer]]] = {
    "chatml": (QWEN, lambda: FakeChatTokenizer()),
    "chatml-merged-newlines": (QWEN, lambda: FakeChatTokenizer(merge_newlines=True)),
    "glm": (GLM, lambda: FakeChatTokenizer("glm")),
}


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("row", FIXTURE_ROWS, ids=lambda row: row.id)
def test_each_target_is_its_reply_after_its_own_inference_prompt(
    setup: str, row: TrainingRow
) -> None:
    family, make_tokenizer = SETUPS[setup]
    tokenizer = make_tokenizer()
    datums = render_row(row, family, tokenizer, max_seq_len=LIMIT)
    assert [datum.message_index for datum in datums] == trained_indexes(row)
    messages = [m.model_dump(mode="json", exclude_none=True) for m in row.messages]
    for datum in datums:
        index = datum.message_index
        prompt = prompt_tokens(messages[:index], row_tools(row), family, tokenizer)
        assert len(datum.tokens) == len(datum.weights)
        assert set(datum.weights) == {0, 1}
        [(start, end)] = weighted_spans(datum)
        assert start == len(prompt)
        assert datum.tokens[:start] == tuple(prompt)
        rendered = tokenizer.render_text(template_messages(row, index + 1), tools=row_tools(row))
        if family.boundary == "glm_stop_tag":
            stop = "<|observation|>" if row.messages[index].tool_calls else "<|user|>"
            assert end == len(datum.tokens)
            assert tokenizer.decode(datum.tokens) == rendered + stop
            assert target_text(tokenizer, datum).endswith(stop)
        else:
            assert tokenizer.decode(datum.tokens) == rendered
            assert target_text(tokenizer, datum).endswith("<|im_end|>")
            assert tokenizer.decode(datum.tokens[end:]) == "\n"


def test_target_is_the_generated_reply_through_the_stop_token() -> None:
    tokenizer = FakeChatTokenizer()
    [datum] = render_row(
        make_row(user("Hi"), assistant("Hello there.")), QWEN, tokenizer, max_seq_len=LIMIT
    )
    assert datum.message_index == 1
    assert target_text(tokenizer, datum) == "\n</think>\n\nHello there.<|im_end|>"
    # The template's newline after the stop token is context the model never generates.
    assert (datum.tokens[-1], datum.weights[-1]) == (ord("\n"), 0)


def test_a_merged_newline_keeps_the_inference_prompt_and_re_encodes_the_reply() -> None:
    tokenizer = FakeChatTokenizer(merge_newlines=True)
    row = make_row(user("Hi"), assistant("Hello there."))
    prompt = prompt_tokens([user("Hi")], None, QWEN, tokenizer)
    full = tokenizer.render(
        template_messages(row, 2), tools=None, add_generation_prompt=False, template_kwargs={}
    )
    assert full[: len(prompt)] != prompt
    [datum] = render_row(row, QWEN, tokenizer, max_seq_len=LIMIT)
    assert datum.tokens[: len(prompt)] == tuple(prompt)
    assert tokenizer.decode(datum.tokens) == tokenizer.decode(full)
    assert target_text(tokenizer, datum) == "\n</think>\n\nHello there.<|im_end|>"


def test_history_tool_results_and_weight_zero_turns_carry_no_loss() -> None:
    row = make_row(
        {"role": "system", "content": "SYSTEM_TEXT"},
        user("USER_ONE"),
        assistant(tool_calls=[call("c1", "Northvale")]),
        tool("c1", "TOOL_RESULT"),
        assistant("ANSWER_ONE"),
        user("USER_TWO"),
        assistant("SKIPPED_TURN", weight=0),
        user("USER_THREE"),
        assistant("ANSWER_TWO"),
        tools=TOOLS,
    )
    for family, tokenizer in ((QWEN, FakeChatTokenizer()), (GLM, FakeChatTokenizer("glm"))):
        datums = render_row(row, family, tokenizer, max_seq_len=LIMIT)
        assert [datum.message_index for datum in datums] == [2, 4, 8]
        targets = [target_text(tokenizer, datum) for datum in datums]
        assert "Northvale" in targets[0]
        assert "ANSWER_ONE" in targets[1]
        assert "ANSWER_TWO" in targets[2]
        context = ("SYSTEM_TEXT", "USER_", "TOOL_RESULT", "SKIPPED_TURN", "Look a city up.")
        assert not any(text in target for text in context for target in targets)
        assert "SKIPPED_TURN" in tokenizer.decode(datums[2].tokens)
        assert sum("ANSWER_ONE" in target for target in targets) == 1


def test_each_target_keeps_its_own_reasoning_when_history_drops_it() -> None:
    tokenizer = FakeChatTokenizer()
    row = make_row(
        user("First?"),
        assistant("First.", reasoning_content="FIRST_REASON"),
        user("Second?"),
        assistant("Second.", reasoning_content="SECOND_REASON"),
    )
    first, second = render_row(row, QWEN, tokenizer, max_seq_len=LIMIT)
    assert "FIRST_REASON" in target_text(tokenizer, first)
    assert "FIRST_REASON" not in tokenizer.decode(second.tokens)
    assert "SECOND_REASON" in target_text(tokenizer, second)


def test_reasoning_only_and_tool_call_only_turns_are_targets_but_empty_turns_are_history() -> None:
    tokenizer = FakeChatTokenizer("glm")
    row = make_row(
        user("Go"),
        assistant(),
        user("Again"),
        assistant(reasoning_content="THINKING"),
        user("Check"),
        assistant(tool_calls=[call("c1", "Eastbrook")]),
        tool("c1", "ok"),
        assistant("Done."),
        tools=TOOLS,
    )
    datums = render_row(row, GLM, tokenizer, max_seq_len=LIMIT)
    assert [datum.message_index for datum in datums] == [3, 5, 7]
    assert target_text(tokenizer, datums[0]) == "THINKING</think><|user|>"


def test_quoted_role_markers_inside_a_reply_are_still_targets() -> None:
    tokenizer = FakeChatTokenizer()
    text = "Quoted <|im_start|>user\n and <|im_end|> café 🐢 東京"
    [datum] = render_row(
        make_row(user("Quote it"), assistant(text)), QWEN, tokenizer, max_seq_len=LIMIT
    )
    assert target_text(tokenizer, datum).endswith(f"{text}<|im_end|>")


def test_templates_get_parsed_tool_arguments_and_no_weights() -> None:
    tokenizer = FakeChatTokenizer()
    row = make_row(
        user("Weather?"),
        assistant(tool_calls=[call("c1", "Port Alder")]),
        tool("c1", "sunny"),
        assistant("Sunny.", weight=1),
        tools=TOOLS,
    )
    render_row(row, QWEN, tokenizer, max_seq_len=LIMIT)
    assert len(tokenizer.calls) == 4
    for recorded in tokenizer.calls:
        assert recorded.tools == row_tools(row)
        assert recorded.template_kwargs == {}
        for message in recorded.messages:
            assert "weight" not in message
            for tool_call in message.get("tool_calls", []):
                assert tool_call["function"]["arguments"] == {"city": "Port Alder"}


def test_family_template_kwargs_reach_every_render() -> None:
    tokenizer = FakeChatTokenizer(merge_newlines=True)
    family = dataclasses.replace(QWEN, template_kwargs=MappingProxyType({"enable_thinking": False}))
    [datum] = render_row(
        make_row(user("Hi"), assistant("Hello.")), family, tokenizer, max_seq_len=LIMIT
    )
    assert all(
        recorded.template_kwargs == {"enable_thinking": False} for recorded in tokenizer.calls
    )
    assert target_text(tokenizer, datum) == "Hello.<|im_end|>"


def test_a_row_over_the_context_limit_is_rejected_whole_and_never_truncated() -> None:
    tokenizer = FakeChatTokenizer()
    row = FIXTURE_ROWS[0]
    datums = render_row(row, QWEN, tokenizer, max_seq_len=LIMIT)
    longest = max(len(datum.tokens) for datum in datums)
    assert render_row(row, QWEN, tokenizer, max_seq_len=longest) == datums
    with pytest.raises(RenderError) as caught:
        render_row(row, QWEN, tokenizer, max_seq_len=longest - 1)
    error = caught.value
    assert error.code is RenderErrorCode.EXCEEDS_CONTEXT
    assert (error.tokens, error.limit) == (longest, longest - 1)


@pytest.mark.parametrize(
    "messages",
    [
        [user("Hi"), assistant("Not this one.", weight=0)],
        [user("Hi"), assistant()],
        [user("Hi"), assistant("Old.", weight=0), user("And?"), assistant()],
    ],
)
def test_a_row_without_a_trained_reply_has_no_trainable_turn(
    messages: list[dict[str, Any]],
) -> None:
    with pytest.raises(RenderError) as caught:
        render_row(make_row(*messages), QWEN, FakeChatTokenizer(), max_seq_len=LIMIT)
    assert caught.value.code is RenderErrorCode.NO_TRAINABLE_TURN


def test_reasoning_is_rejected_for_a_family_that_cannot_render_it() -> None:
    family = dataclasses.replace(QWEN, supports_reasoning=False)
    tokenizer = FakeChatTokenizer()
    history = assistant("Earlier.", reasoning_content="Hidden.", weight=0)
    row = make_row(user("Hi"), history, user("And?"), assistant("Now."))
    with pytest.raises(RenderError) as caught:
        render_row(row, family, tokenizer, max_seq_len=LIMIT)
    assert caught.value.code is RenderErrorCode.REASONING_UNSUPPORTED
    assert tokenizer.calls == []
    assert render_row(make_row(user("Hi"), assistant("Now.")), family, tokenizer, max_seq_len=LIMIT)


def test_glm_targets_end_with_the_role_tag_the_model_stops_on() -> None:
    tokenizer = FakeChatTokenizer("glm")
    row = make_row(
        user("Weather?"),
        assistant(tool_calls=[call("c1", "Northvale")]),
        tool("c1", "rain"),
        assistant("It rains."),
        tools=TOOLS,
    )
    tool_turn, answer = render_row(row, GLM, tokenizer, max_seq_len=LIMIT)
    assert target_text(tokenizer, tool_turn) == (
        "</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Northvale</arg_value>"
        "</tool_call><|observation|>"
    )
    assert target_text(tokenizer, answer) == "</think>It rains.<|user|>"
    for datum, stop in ((tool_turn, "<|observation|>"), (answer, "<|user|>")):
        assert (datum.tokens[-1], datum.weights[-1]) == (tokenizer.token_id(stop), 1)


@pytest.mark.parametrize(
    ("messages", "message"),
    [
        ([assistant("Unprompted.")], "messages[0]: the chat template failed: ValueError"),
        (
            [{"role": "system", "content": "S"}, assistant("No user.")],
            "No user query found",
        ),
    ],
)
def test_a_template_error_is_template_failed(messages: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(RenderError) as caught:
        render_row(make_row(*messages), QWEN, FakeChatTokenizer(), max_seq_len=LIMIT)
    assert caught.value.code is RenderErrorCode.TEMPLATE_FAILED
    assert message in caught.value.detail


class _RewritesHistory(FakeChatTokenizer):
    def render_text(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        marker = "PROMPT" if kwargs.get("add_generation_prompt") else "FULL"
        return marker + super().render_text(messages, **kwargs)


class _LossyEncode(FakeChatTokenizer):
    def render(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[int]:
        return FakeChatTokenizer.encode(self, self.render_text(messages, **kwargs))

    def encode(self, text: str) -> list[int]:
        return super().encode(text.replace("é", "e"))


@pytest.mark.parametrize(
    ("tokenizer", "detail"),
    [
        (_RewritesHistory(), "does not extend its inference prompt"),
        (_LossyEncode(merge_newlines=True), "re-encoding the reply changed its text"),
    ],
)
def test_a_reply_that_does_not_extend_its_prompt_is_a_prefix_mismatch(
    tokenizer: FakeChatTokenizer, detail: str
) -> None:
    with pytest.raises(RenderError) as caught:
        render_row(make_row(user("Hi"), assistant("Café.")), QWEN, tokenizer, max_seq_len=LIMIT)
    assert caught.value.code is RenderErrorCode.PREFIX_MISMATCH
    assert detail in caught.value.detail


class _Returns(FakeChatTokenizer):
    def __init__(self, tokens: Any) -> None:
        super().__init__()
        self.tokens = tokens

    def render(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[int]:
        return self.tokens  # type: ignore[no-any-return]


@pytest.mark.parametrize("tokens", [[], [-1], [True], ["1"], [1.0]])
def test_invalid_token_ids_are_rejected(tokens: Any) -> None:
    with pytest.raises(RenderError) as caught:
        render_row(
            make_row(user("Hi"), assistant("Hello.")), QWEN, _Returns(tokens), max_seq_len=LIMIT
        )
    assert caught.value.code is RenderErrorCode.INVALID_TOKENS


class _IgnoresAssistant(FakeChatTokenizer):
    def render_text(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        kept = [m for m in messages if m["role"] != "assistant"]
        return super().render_text(kept, tools=kwargs.get("tools"))


def test_a_reply_that_renders_nothing_has_no_target_tokens() -> None:
    with pytest.raises(RenderError) as caught:
        render_row(
            make_row(user("Hi"), assistant("Hello.")), QWEN, _IgnoresAssistant(), max_seq_len=LIMIT
        )
    assert caught.value.code is RenderErrorCode.NO_TARGET_TOKENS


class _NoEndOfTurn(FakeChatTokenizer):
    def render_text(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        return super().render_text(messages, **kwargs).replace("<|im_end|>", "")


def test_a_reply_without_its_stop_token_is_rejected() -> None:
    with pytest.raises(RenderError) as caught:
        render_row(
            make_row(user("Hi"), assistant("Hello.")), QWEN, _NoEndOfTurn(), max_seq_len=LIMIT
        )
    assert caught.value.code is RenderErrorCode.TEMPLATE_FAILED
    assert "does not end with a stop sequence" in caught.value.detail


class _NoRoleTags(FakeChatTokenizer):
    def token_id(self, token: str) -> int | None:
        return None


@pytest.mark.parametrize(
    ("family", "tokenizer"),
    [
        (dataclasses.replace(QWEN, stop_sequences=("</s>",)), FakeChatTokenizer()),
        (GLM, _NoRoleTags("glm")),
    ],
    ids=["none", "glm"],
)
def test_a_stop_sequence_must_be_a_single_token(
    family: ModelFamily, tokenizer: FakeChatTokenizer
) -> None:
    with pytest.raises(RenderError) as caught:
        render_row(make_row(user("Hi"), assistant("Hello.")), family, tokenizer, max_seq_len=LIMIT)
    assert caught.value.code is RenderErrorCode.TEMPLATE_FAILED
    assert "is not a single token" in caught.value.detail


def test_prompt_tokens_take_provider_messages() -> None:
    tokenizer = FakeChatTokenizer()
    row = make_row(
        user("Weather?"),
        assistant(tool_calls=[call("c1", "Northvale")], weight=0),
        tool("c1", "rain"),
        assistant("It rains."),
        tools=TOOLS,
    )
    provider = to_provider_dict(row)
    tokens = prompt_tokens(provider["messages"][:3], provider["tools"], QWEN, tokenizer)
    expected = tokenizer.render_text(
        template_messages(row, 3), tools=TOOLS, add_generation_prompt=True
    )
    assert tokenizer.decode(tokens) == expected
    assert all("weight" not in message for message in tokenizer.calls[-1].messages)
    assert expected.endswith("<|im_start|>assistant\n<think>\n")
    assert prompt_tokens([user("Hello")], None, QWEN, tokenizer) == tokenizer.encode(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


@pytest.mark.parametrize(
    ("messages", "detail"),
    [
        ([], "prompt: the chat template failed"),
        (
            [assistant(tool_calls=[{"id": "c", "function": {"name": "f", "arguments": "{"}}])],
            "not valid JSON",
        ),
        (
            [assistant(tool_calls=[{"id": "c", "function": {"name": "f", "arguments": "[1]"}}])],
            "must be an object",
        ),
    ],
)
def test_prompt_tokens_reject_what_the_template_cannot_render(
    messages: list[dict[str, Any]], detail: str
) -> None:
    with pytest.raises(RenderError) as caught:
        prompt_tokens(messages, None, QWEN, FakeChatTokenizer())
    assert caught.value.code is RenderErrorCode.TEMPLATE_FAILED
    assert detail in caught.value.detail


def test_check_rows_accepts_whole_rows_rejects_the_rest_and_counts_tokens() -> None:
    tokenizer = FakeChatTokenizer()
    limit = max(
        len(d.tokens)
        for r in FIXTURE_ROWS[:2]
        for d in render_row(r, QWEN, tokenizer, max_seq_len=LIMIT)
    )
    untrained = make_row(user("Hi"), assistant("Skip.", weight=0), row_id="untrained")
    long = make_row(user("x" * limit), assistant("Long."), row_id="long")
    rows = [FIXTURE_ROWS[0], untrained, long, FIXTURE_ROWS[1]]
    report = check_rows(iter(rows), QWEN, tokenizer, max_seq_len=limit)
    assert [row.id for row, _ in report.accepted] == [FIXTURE_ROWS[0].id, FIXTURE_ROWS[1].id]
    long_tokens = len(render_row(long, QWEN, tokenizer, max_seq_len=LIMIT)[0].tokens)
    assert report.rejected == [
        {
            "id": "untrained",
            "code": "no_trainable_turn",
            "detail": "no assistant message has content to train on and a weight other than 0",
        },
        {
            "id": "long",
            "code": "exceeds_context",
            "detail": f"a rendered datum has {long_tokens} tokens, over the limit of {limit}",
            "tokens": long_tokens,
            "limit": limit,
        },
    ]
    datums = [datum for _, row_datums in report.accepted for datum in row_datums]
    assert report.stats == {
        "rendered_datums": len(datums),
        "assistant_targets": len(datums),
        "context_tokens": sum(len(datum.tokens) for datum in datums),
        "target_tokens": sum(sum(datum.weights) for datum in datums),
        "max_context_tokens": max(len(datum.tokens) for datum in datums),
        "rejected_rows": 2,
    }
    for row, row_datums in report.accepted:
        assert row_datums == render_row(row, QWEN, tokenizer, max_seq_len=limit)


def test_check_rows_of_nothing_is_empty() -> None:
    report = check_rows([], QWEN, FakeChatTokenizer(), max_seq_len=LIMIT)
    assert (report.accepted, report.rejected) == ([], [])
    assert set(report.stats.values()) == {0}
