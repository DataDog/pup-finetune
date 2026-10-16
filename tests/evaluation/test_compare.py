# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The compare target: resolving --compare-model, the chat sampler and the offline fake."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from pup_finetune.errors import SettingsConflict, UsageError
from pup_finetune.evaluation import compare
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.compare import (
    COMPARE_MAX_TOKENS,
    FAKE_COMPARE_ANSWER,
    ChatSampler,
    FakeCompareSampler,
    attach_compare,
    compare_conflict,
    open_compare_sampler,
    resolve_compare,
)
from pup_finetune.evaluation.judge import check_judge_differs
from pup_finetune.evaluation.replay import EvalDir
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
    stored = first.identity()
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


def identity_of(spec: str) -> dict[str, Any]:
    target = resolve_compare(spec, None, runtime=DEV)
    assert target is not None
    return target.identity()


def test_a_compare_conflict_names_what_changed_and_a_free_out_dir(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    (tmp_path / "eval-2").mkdir()
    (tmp_path / "eval-2" / "evaluation.json").write_text("{}")
    stored = identity_of("openai/gpt-5.5")
    assert compare_conflict(None, stored, out_dir=out) is None
    assert compare_conflict(stored, identity_of("openai/gpt-5.5"), out_dir=out) is None
    conflict = compare_conflict(stored, identity_of("openai/gpt-5.4"), out_dir=out)
    assert conflict is not None
    assert "compares with openai/gpt-5.5, not openai/gpt-5.4 (changed: model)" in str(conflict)
    assert conflict.hint is not None
    assert conflict.hint.endswith(f"pass --out-dir {tmp_path / 'eval-3'}")


def test_attaching_another_compare_model_raises_the_same_conflict(tmp_path: Path) -> None:
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    attach_compare(eval_dir.compare_path, identity_of("openai/gpt-5.5"))
    wanted = identity_of("anthropic/claude-opus-5-5")
    expected = compare_conflict(identity_of("openai/gpt-5.5"), wanted, out_dir=eval_dir.path)
    assert expected is not None
    with pytest.raises(SettingsConflict) as raised:
        attach_compare(eval_dir.compare_path, wanted)
    assert (str(raised.value), raised.value.hint) == (str(expected), expected.hint)


def test_a_stored_compare_target_is_resolved_from_its_spec(tmp_path: Path) -> None:
    stored = identity_of("openai/gpt-5.5")
    edited = {**stored, "model": {**stored["model"], "base_url": "https://elsewhere.example/v1"}}
    target = resolve_compare(None, edited, runtime=DEV)
    assert target is not None and target.source == "evaluation directory"
    assert target.config.base_url == "https://api.openai.com/v1"
    conflict = compare_conflict(edited, target.identity(), out_dir=tmp_path / "eval")
    assert conflict is not None
    assert "(changed: base_url)" in str(conflict)
    fake = identity_of("fake-compare")
    with pytest.raises(UsageError, match="development only"):
        resolve_compare(None, fake, runtime=Runtime())


def test_an_openai_refusal_keeps_its_text_as_the_answer() -> None:
    api = FakeChatApi()
    api.script.append(completion(None, refusal="I can't help with that."))
    generation = sampler(api).generate(PREFIX, [], max_tokens=10)
    assert generation.message == {"role": "assistant", "content": "I can't help with that."}
    assert generation.stop_reason == "refusal"


def test_the_chat_sampler_sends_no_recorded_reasoning() -> None:
    api = FakeChatApi()
    api.script.append(completion("done"))
    thought = {"role": "assistant", "content": "Looking.", "reasoning_content": "hmm"}
    sampler(api).generate([*PREFIX, thought, {"role": "user", "content": "go"}], [], max_tokens=10)
    (body,) = api.bodies
    assert body["messages"][2] == {"role": "assistant", "content": "Looking."}


def test_a_compare_model_spec_error_names_the_compare_model() -> None:
    with pytest.raises(UsageError, match=r"^unknown compare model 'deepseek-flsh'") as info:
        resolve_compare("deepseek-flsh", None, runtime=DEV)
    assert info.value.hint is not None and "did you mean deepseek-v4p1-flash?" in info.value.hint
    with pytest.raises(UsageError, match=r"^the compare model 'openai/' has no model"):
        resolve_compare("openai/", None, runtime=DEV)


def test_the_compare_module_does_not_depend_on_replay() -> None:
    tree = ast.parse(Path(compare.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "pup_finetune.evaluation.replay" not in imported


def test_a_missing_anthropic_extra_names_the_compare_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(UsageError, match=r"^Anthropic compare models need the 'anthropic' package"):
        resolve_compare("anthropic/claude-opus-5-5", None, runtime=DEV)


def test_a_broken_anthropic_install_names_the_compare_model_when_it_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = resolve_compare("anthropic/claude-opus-5-5", None, runtime=DEV)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "pup_finetune.anthropic_client", raising=False)
    with (
        pytest.raises(UsageError, match=r"^Anthropic compare models need the 'anthropic' package"),
        open_compare_sampler(target, [], seed=0),
    ):
        pass


# --- prefixes a chat API refuses ----------------------------------------------------------

USER = {"role": "user", "content": "What changed?"}
SYSTEM = {"role": "system", "content": "Review."}
SAID = {"role": "assistant", "content": "Let me check."}
THOUGHT = {"role": "assistant", "content": "", "reasoning_content": "thinking only"}
CALLED = {"role": "assistant", "content": "", "tool_calls": [tool_call("{}")]}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "plan"}
LATER = {"role": "user", "content": "Never mind."}


@pytest.mark.parametrize(
    ("prefix", "openai", "anthropic"),
    [
        ([SYSTEM, USER], None, None),
        ([SYSTEM, USER, CALLED, RESULT], None, None),
        ([USER, CALLED, RESULT, SAID, LATER], None, None),
        # A turn of reasoning alone: Anthropic gets no turn for it, so it does not end there.
        ([USER, THOUGHT], None, None),
        ([USER, SAID], None, "ends_on_assistant"),
        ([USER, CALLED], "unanswered_tool_call", "unanswered_tool_call"),
        ([USER, CALLED, LATER], "unanswered_tool_call", "unanswered_tool_call"),
        ([USER, CALLED, LATER, RESULT], "unanswered_tool_call", "unanswered_tool_call"),
        # Anthropic gets one assistant turn, the call then the text, so the result follows it.
        ([USER, CALLED, SAID, RESULT], "unanswered_tool_call", None),
        # Anthropic gets every system message in its top-level system.
        ([SYSTEM, USER, SAID, SYSTEM, LATER], None, None),
        ([SYSTEM, USER, SAID, SYSTEM], None, "ends_on_assistant"),
        ([RESULT, USER], "orphan_tool_result", "orphan_tool_result"),
        ([USER, RESULT], "orphan_tool_result", "orphan_tool_result"),
        ([USER, SAID, RESULT, LATER], "orphan_tool_result", "orphan_tool_result"),
        ([], "no_messages", "no_messages"),
        ([SYSTEM], None, "no_messages"),
        ([SYSTEM, THOUGHT], None, "no_messages"),
        # Anthropic's first turn must be the user's; OpenAI takes an assistant greeting.
        ([SYSTEM, SAID, USER], None, "starts_on_assistant"),
        ([THOUGHT, SAID, LATER], None, "starts_on_assistant"),
        ([CALLED, RESULT, USER], None, "starts_on_assistant"),
        ([SYSTEM, THOUGHT, USER], None, None),
    ],
)
def test_the_prefixes_each_chat_api_refuses(
    prefix: list[dict[str, Any]], openai: str | None, anthropic: str | None
) -> None:
    assert compare.refused_prefix(prefix, "openai") == openai
    assert compare.refused_prefix(prefix, "anthropic") == anthropic
    # Fireworks and the offline fake take any prefix.
    assert compare.refused_prefix(prefix, "fireworks") is None
    assert compare.refused_prefix(prefix, "fake") is None


def test_the_skipped_cases_are_those_the_compare_provider_refuses() -> None:
    def shaped(index: int, prefix: list[dict[str, Any]]) -> ReplayCase:
        return ReplayCase(
            id=f"row:{index}", row_id="row", trace_id=None, message_index=index,
            prefix=tuple(prefix), tools=(), reference=SAID,
        )  # fmt: skip

    cases = [shaped(1, [USER]), shaped(2, [USER, SAID]), shaped(3, [USER, CALLED])]
    assert compare.compare_skips(cases, "anthropic") == {
        "row:2": "ends_on_assistant",
        "row:3": "unanswered_tool_call",
    }
    assert compare.compare_skips(cases, "openai") == {"row:3": "unanswered_tool_call"}
    assert compare.compare_skips(cases, "fireworks") == {}
