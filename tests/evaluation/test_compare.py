# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The compare target: resolving --compare-model, the chat sampler and the offline fake."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from pup_finetune.errors import UsageError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.compare import (
    COMPARE_MAX_TOKENS,
    FAKE_COMPARE_ANSWER,
    ChatSampler,
    FakeCompareSampler,
    resolve_compare,
)
from pup_finetune.evaluation.judge import check_judge_differs
from pup_finetune.inference import ChatClient, JudgeConfig
from pup_finetune.judges import parse_judge
from pup_finetune.runtime import Runtime
from tests.fakes.chat_api import API_KEY, BASE_URL, FakeChatApi, completion

TOOL = {"type": "function", "function": {"name": "explain_plan", "parameters": {"type": "object"}}}
PREFIX = [{"role": "system", "content": "Review."}, {"role": "user", "content": "SELECT 1"}]
CONFIG = JudgeConfig(
    provider="fireworks", model="accounts/fireworks/models/big", base_url=BASE_URL,
    temperature=0.0, max_tokens=COMPARE_MAX_TOKENS, reasoning_effort="none",
)  # fmt: skip


def sampler(api: FakeChatApi) -> ChatSampler:
    client = ChatClient(BASE_URL, SecretStr(API_KEY), transport=api.transport, max_attempts=1)
    return ChatSampler(client, CONFIG)


def tool_call(arguments: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": "explain_plan", "arguments": arguments},
    }


def test_the_chat_sampler_sends_the_prefix_and_tools_and_keeps_tool_calls() -> None:
    api = FakeChatApi()
    api.script.append(
        completion("", tool_calls=[tool_call('{"query": "SELECT 1"}')], finish_reason="tool_calls")
    )
    generation = sampler(api).generate(PREFIX, [TOOL], max_tokens=1000)
    (body,) = api.bodies
    assert body["messages"] == PREFIX
    assert body["tools"] == [TOOL]
    assert body["max_tokens"] == 1000
    assert body["reasoning_effort"] == "none"
    assert generation.message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call('{"query": "SELECT 1"}')],
    }
    assert generation.format_valid
    assert generation.stop_reason == "tool_calls"
    assert (generation.prompt_tokens, generation.completion_tokens) == (11, 7)


@pytest.mark.parametrize(
    ("response", "valid", "stop"),
    [
        (completion('{"type": "no_opt', finish_reason="length"), False, "length"),
        (
            completion("", tool_calls=[tool_call("{oops")], finish_reason="tool_calls"),
            False,
            "tool_calls",
        ),
        (completion("I will not help with that.", finish_reason="refusal"), True, "refusal"),
        (completion('{"type": "ddl"}'), True, "stop"),
    ],
)
def test_cut_off_answers_and_broken_calls_are_not_format_valid(
    response: Any, valid: bool, stop: str
) -> None:
    api = FakeChatApi()
    api.script.append(response)
    generation = sampler(api).generate(PREFIX, [], max_tokens=10)
    assert "tools" not in api.bodies[0]
    assert (generation.format_valid, generation.stop_reason) == (valid, stop)


def case(index: int) -> ReplayCase:
    return ReplayCase(
        id=f"r{index}:1", row_id=f"r{index}", trace_id=None, message_index=1,
        prefix=({"role": "user", "content": f"query {index}"},), tools=(),
        reference={"role": "assistant", "content": json.dumps({"type": "ddl"}), "reasoning_content": "x"},
    )  # fmt: skip


def test_the_fake_compare_model_answers_the_reference_for_most_cases() -> None:
    cases = [case(index) for index in range(200)]
    fake = FakeCompareSampler(cases, seed=0)
    answers = [fake.generate(list(c.prefix), list(c.tools), max_tokens=1).message for c in cases]
    agreed = sum(
        answer == {"role": "assistant", "content": '{"type": "ddl"}'} for answer in answers
    )
    assert 160 <= agreed <= 195
    assert all(a["content"] in ('{"type": "ddl"}', FAKE_COMPARE_ANSWER) for a in answers)
    again = FakeCompareSampler(cases, seed=0)
    assert [again.generate(list(c.prefix), [], max_tokens=1).message for c in cases] == answers
    unknown = fake.generate([{"role": "user", "content": "other"}], [], max_tokens=1)
    assert unknown.message["content"] == FAKE_COMPARE_ANSWER


DEV = Runtime(dev_mode=True)


def test_a_compare_spec_resolves_like_a_judge_with_the_compare_token_limit() -> None:
    target = resolve_compare("anthropic/claude-opus-5-5", None, runtime=DEV)
    assert target is not None
    assert (target.config.provider, target.config.model) == ("anthropic", "claude-opus-5-5")
    assert target.config.max_tokens == COMPARE_MAX_TOKENS
    assert target.config.reasoning_effort == "low"
    assert target.source == "flag"
    fireworks = resolve_compare("deepseek-v4p1-flash", None, runtime=DEV)
    assert fireworks is not None
    assert fireworks.config.model == "accounts/fireworks/models/deepseek-v4p1-flash"
    assert fireworks.config.reasoning_effort == "none"


def test_a_stored_compare_target_continues_without_the_flag() -> None:
    first = resolve_compare("openai/gpt-5.5", None, runtime=DEV)
    assert first is not None
    stored = first.identity({"run_id": "ft"})
    assert stored["spec"] == "openai/gpt-5.5"
    again = resolve_compare(None, stored, runtime=DEV)
    assert again is not None and again.source == "evaluation directory"
    assert again.config == first.config
    assert resolve_compare(None, None, runtime=DEV) is None


def test_the_fake_compare_model_needs_development_mode() -> None:
    target = resolve_compare("fake-compare", None, runtime=DEV)
    assert target is not None and target.offline
    with pytest.raises(UsageError, match="development only"):
        resolve_compare("fake-compare", None, runtime=Runtime())


def test_the_judge_must_not_be_the_compare_model() -> None:
    compare = resolve_compare("anthropic/claude-opus-5-5", None, runtime=DEV)
    assert compare is not None
    with pytest.raises(UsageError, match="is the compare model"):
        check_judge_differs(
            parse_judge("anthropic/claude-opus-5-5"), "base", compare=compare.config
        )
    check_judge_differs(parse_judge("anthropic/claude-sonnet-5-5"), "base", compare=compare.config)


def test_the_compare_billing_names_its_calls() -> None:
    target = resolve_compare("anthropic/claude-opus-5-5", None, runtime=DEV)
    assert target is not None
    assert target.billing(3) == (
        "Anthropic bills each compare-model call per token at its own rates, which "
        "pup-finetune does not know: 3 calls, at most 16,000 output tokens each"
    )
    fake = resolve_compare("fake-compare", None, runtime=DEV)
    assert fake is not None
    assert fake.billing(2).startswith("the fake compare model is offline and bills nothing")
