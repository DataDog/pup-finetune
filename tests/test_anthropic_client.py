# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Anthropic judge client against a fake SDK client: requests, answers and errors.

One test runs the real SDK client over a mock ``httpx2`` transport, so the request that
reaches the wire (beta header, fallbacks, output config) is checked too. No network.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

anthropic = pytest.importorskip("anthropic", reason="needs the anthropic extra")
httpx2 = pytest.importorskip("httpx2", reason="needs the anthropic extra")

from pup_finetune import redaction  # noqa: E402
from pup_finetune.anthropic_client import (  # noqa: E402
    EFFORT_MODELS,
    FALLBACK_BETA,
    FALLBACK_MODELS,
    AnthropicChatClient,
)
from pup_finetune.errors import ExternalError, FinetuneError, UsageError  # noqa: E402
from pup_finetune.evaluation.judge import VERDICT_SCHEMA  # noqa: E402
from pup_finetune.inference import ChatCompletion, ContextLimitError  # noqa: E402
from pup_finetune.judge_models import ANTHROPIC_FAST, ANTHROPIC_STRONG  # noqa: E402
from pup_finetune.judges import parse_judge  # noqa: E402
from pup_finetune.providers.base import (  # noqa: E402
    ProviderAuthError,
    ProviderCapacityError,
    TransientError,
)

API_KEY = "sk-ant-synthetic-0000-judge-key-4c2e"
MESSAGES = [
    {"role": "system", "content": "You judge."},
    {"role": "user", "content": "Is the candidate acceptable?"},
]
VERDICT = '{"pass": true, "reason": "same call"}'
URL = "https://api.anthropic.com/v1/messages"


def message(
    *blocks: Any,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model=model,
        usage=usage if usage is not None else SimpleNamespace(input_tokens=12, output_tokens=5),
    )


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


class FakeMessages:
    def __init__(self, answer: Callable[[dict[str, Any]], Any]) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        result = self.answer(params)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeSdk:
    """The part of ``anthropic.Anthropic`` the client uses, answering with ``answer``."""

    def __init__(self, answer: Callable[[dict[str, Any]], Any] | None = None) -> None:
        reply = answer or (lambda params: message(text(VERDICT)))
        self.messages = FakeMessages(reply)
        self.beta = SimpleNamespace(messages=FakeMessages(reply))
        self.models = SimpleNamespace(list=self._list)
        self.listed: list[dict[str, Any]] = []
        self.list_error: BaseException | None = None
        self.closed = False

    def _list(self, **params: Any) -> Any:
        self.listed.append(params)
        if self.list_error is not None:
            raise self.list_error
        return SimpleNamespace(data=[])

    def close(self) -> None:
        self.closed = True

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls + self.beta.messages.calls


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def chat(sdk: FakeSdk, key: str | None = API_KEY) -> AnthropicChatClient:
    return AnthropicChatClient(SecretStr(key) if key is not None else None, client=sdk)


def status_error(cls: type[Any], status: int, detail: str = "nope") -> Any:
    request = httpx2.Request("POST", URL)
    body = {"type": "error", "error": {"type": "x", "message": detail}}
    response = httpx2.Response(status, request=request, json=body)
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def assert_no_key(exc: BaseException) -> None:
    texts = [str(exc), repr(exc)]
    if isinstance(exc, FinetuneError):
        texts += [exc.message, exc.hint or ""]
    for value in texts:
        assert API_KEY not in value
        assert API_KEY[-8:] not in value


# --- requests --------------------------------------------------------------------------


def test_the_system_message_is_top_level_and_no_sampling_parameter_is_sent() -> None:
    sdk = FakeSdk()
    with chat(sdk) as client:
        client.complete(MESSAGES, model="claude-haiku-4-5", temperature=0.0, max_tokens=900)
    (params,) = sdk.messages.calls
    assert params["system"] == "You judge."
    assert params["messages"] == [{"role": "user", "content": "Is the candidate acceptable?"}]
    assert params["model"] == "claude-haiku-4-5"
    assert params["max_tokens"] == 900
    for name in ("temperature", "top_p", "top_k", "thinking"):
        assert name not in params
    assert sdk.beta.messages.calls == []
    assert sdk.closed


def test_every_system_message_joins_the_top_level_system() -> None:
    sdk = FakeSdk()
    later = [*MESSAGES, {"role": "assistant", "content": "Yes."}]
    later += [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why?"}]
    chat(sdk).complete(later, model="claude-haiku-4-5")
    (params,) = sdk.messages.calls
    assert params["system"] == "You judge.\n\nBe brief."
    assert [turn["role"] for turn in params["messages"]] == ["user", "assistant", "user"]


def test_max_tokens_defaults_to_the_anthropic_judge_limit() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES[1:], model="claude-haiku-4-5")
    assert sdk.messages.calls[0]["max_tokens"] == 16000
    assert "system" not in sdk.messages.calls[0]


def test_text_parts_are_sent_as_text() -> None:
    sdk = FakeSdk()
    parts = [{"type": "text", "text": "Is it "}, {"type": "image"}, {"type": "text", "text": "ok?"}]
    chat(sdk).complete([{"role": "user", "content": parts}], model="claude-haiku-4-5")
    assert sdk.messages.calls[0]["messages"] == [{"role": "user", "content": "Is it ok?"}]


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current conditions for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def call(call_id: str, city: str) -> dict[str, Any]:
    arguments = json.dumps({"city": city})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments},
    }


TOOL_TURNS = [
    {"role": "system", "content": "You answer about the weather."},
    {"role": "user", "content": "Weather in Northvale and Eastport?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [call("c1", "Northvale"), call("c2", "Eastport")],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "14C, rain"},
    {"role": "tool", "tool_call_id": "c2", "content": "20C, sun"},
    {
        "role": "assistant",
        "content": "Rain in Northvale, sun in Eastport.",
        "reasoning_content": "x",
    },
    {"role": "user", "content": "And tomorrow?"},
]


def test_tools_and_tool_turns_are_sent_as_anthropic_blocks() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(TOOL_TURNS, model="claude-opus-5-5", tools=[WEATHER_TOOL], max_tokens=100)
    (params,) = sdk.calls
    assert params["system"] == "You answer about the weather."
    assert params["tools"] == [
        {
            "name": "get_weather",
            "description": "Current conditions for a city.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert params["messages"] == [
        {"role": "user", "content": "Weather in Northvale and Eastport?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "get_weather",
                    "input": {"city": "Northvale"},
                },
                {
                    "type": "tool_use",
                    "id": "c2",
                    "name": "get_weather",
                    "input": {"city": "Eastport"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "14C, rain"},
                {"type": "tool_result", "tool_use_id": "c2", "content": "20C, sun"},
            ],
        },
        {"role": "assistant", "content": "Rain in Northvale, sun in Eastport."},
        {"role": "user", "content": "And tomorrow?"},
    ]


def tool_use(block_id: str, city: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name="get_weather", input={"city": city})


def test_a_tool_use_answer_is_an_openai_tool_call_that_round_trips() -> None:
    answer = message(text("Checking."), tool_use("toolu_1", "Northvale"), stop_reason="tool_use")
    sdk = FakeSdk(lambda params: answer)
    result = chat(sdk).complete(TOOL_TURNS[:2], model="claude-opus-5-5", tools=[WEATHER_TOOL])
    assert result.finish_reason == "tool_calls"
    assert result.message == {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
            }
        ],
    }
    following = [
        *TOOL_TURNS[:2],
        result.message,
        {"role": "tool", "tool_call_id": "toolu_1", "content": "14C"},
    ]
    chat(sdk).complete(following, model="claude-opus-5-5", tools=[WEATHER_TOOL])
    assert sdk.calls[-1]["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Northvale"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "14C"}],
        },
    ]


def test_tool_arguments_that_are_not_a_json_object_are_kept_raw() -> None:
    sdk = FakeSdk()
    broken = {"id": "c9", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
    turns = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [broken]},
    ]
    chat(sdk).complete(turns, model="claude-haiku-4-5")
    assert sdk.calls[0]["messages"][1]["content"][0]["input"] == {"_raw": "{oops"}


@pytest.mark.parametrize("model", sorted(FALLBACK_MODELS))
def test_models_with_fallbacks_use_the_beta_api_with_default_fallbacks(model: str) -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model=model, max_tokens=100, response_schema=VERDICT_SCHEMA)
    assert sdk.messages.calls == []
    (params,) = sdk.beta.messages.calls
    assert params["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert params["fallbacks"] == "default"
    assert params["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
    }


def test_a_compare_model_never_asks_for_fallbacks_and_a_judge_still_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pup_finetune.evaluation.compare import CompareTarget, open_compare_sampler
    from pup_finetune.evaluation.runner import open_judge_client

    sdk = FakeSdk()
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: sdk)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    config = parse_judge("anthropic/claude-opus-5-5")
    target = CompareTarget(config=config, source="flag")
    with open_compare_sampler(target, [], seed=0) as sampler:
        sampler.generate(MESSAGES, [], max_tokens=100)
    (compared,) = sdk.messages.calls
    assert "fallbacks" not in compared
    assert "betas" not in compared
    assert sdk.beta.messages.calls == []
    with open_judge_client(config) as client:
        client.complete(MESSAGES, model=config.model, max_tokens=100)
    (judged,) = sdk.beta.messages.calls
    assert judged["fallbacks"] == "default"


@pytest.mark.parametrize("model", sorted(EFFORT_MODELS - FALLBACK_MODELS))
def test_models_with_effort_only_get_low_effort_and_no_fallbacks(model: str) -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model=model, response_schema=VERDICT_SCHEMA)
    assert sdk.beta.messages.calls == []
    (params,) = sdk.messages.calls
    assert "fallbacks" not in params
    assert "betas" not in params
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["schema"] == VERDICT_SCHEMA


@pytest.mark.parametrize("spec", [ANTHROPIC_FAST, ANTHROPIC_STRONG])
def test_the_default_anthropic_judges_get_low_effort_and_the_verdict_schema(spec: str) -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model=parse_judge(spec).model, response_schema=VERDICT_SCHEMA)
    (params,) = sdk.beta.messages.calls or sdk.messages.calls
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["schema"] == VERDICT_SCHEMA


def test_haiku_gets_structured_output_but_neither_effort_nor_fallbacks() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model="claude-haiku-4-5-20251001", response_schema=VERDICT_SCHEMA)
    (params,) = sdk.messages.calls
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
    assert "fallbacks" not in params


def test_an_unknown_model_gets_no_output_config() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model="claude-legacy-2", response_schema=VERDICT_SCHEMA)
    assert "output_config" not in sdk.messages.calls[0]


def test_without_a_schema_there_is_no_format() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
    assert sdk.beta.messages.calls[0]["output_config"] == {"effort": "low"}


def test_a_dated_model_id_matches_its_model_but_a_longer_name_does_not() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model="claude-opus-5-20260101")
    chat(sdk).complete(MESSAGES, model="claude-opus-5-9")
    assert len(sdk.beta.messages.calls) == 1
    assert sdk.beta.messages.calls[0]["model"] == "claude-opus-5-20260101"
    assert "output_config" not in sdk.messages.calls[0]


# --- answers ---------------------------------------------------------------------------


def test_the_answer_is_the_text_blocks_without_thinking_or_fallback_blocks() -> None:
    sdk = FakeSdk(
        lambda params: message(
            SimpleNamespace(type="thinking", thinking='{"pass": false}', signature="s"),
            SimpleNamespace(type="fallback"),
            text('{"pass": true, '),
            text('"reason": "same call"}'),
            model="claude-opus-4-8",
        )
    )
    result = chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
    assert result == ChatCompletion(
        message={"role": "assistant", "content": VERDICT},
        usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        model="claude-opus-4-8",
        finish_reason="stop",
    )


@pytest.mark.parametrize(
    ("stop", "finish"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "refusal"),
        ("stop_sequence", "stop_sequence"),
        ("pause_turn", "pause_turn"),
    ],
)
def test_stop_reasons_map_to_finish_reasons(stop: str, finish: str) -> None:
    sdk = FakeSdk(lambda params: message(text(""), stop_reason=stop))
    assert chat(sdk).complete(MESSAGES, model="claude-opus-5-5").finish_reason == finish


def test_a_refusal_with_no_content_is_an_empty_refused_answer() -> None:
    sdk = FakeSdk(lambda params: message(stop_reason="refusal", usage=SimpleNamespace()))
    result = chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
    assert result.message == {"role": "assistant", "content": ""}
    assert result.finish_reason == "refusal"
    assert result.usage == {}


# --- errors ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "kind", "status"),
    [
        (lambda: status_error(anthropic.AuthenticationError, 401), ProviderAuthError, 401),
        (lambda: status_error(anthropic.PermissionDeniedError, 403), ProviderAuthError, 403),
        (lambda: status_error(anthropic.RateLimitError, 429), ProviderCapacityError, 429),
        (lambda: status_error(anthropic.OverloadedError, 529), ProviderCapacityError, 529),
        (lambda: status_error(anthropic.InternalServerError, 500), TransientError, 500),
        (lambda: status_error(anthropic.APIStatusError, 503), TransientError, 503),
        (lambda: status_error(anthropic.BadRequestError, 400), ExternalError, 400),
        (lambda: status_error(anthropic.NotFoundError, 404), ExternalError, 404),
    ],
)
def test_sdk_errors_map_most_specific_first(
    error: Callable[[], BaseException], kind: type[ExternalError], status: int
) -> None:
    sdk = FakeSdk(lambda params: error())
    with pytest.raises(ExternalError) as info:
        chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
    assert type(info.value) is kind
    assert info.value.source == "anthropic"
    assert f"HTTP {status}" in info.value.message
    assert "nope" in info.value.message
    if kind is ProviderAuthError:
        assert "pup finetune auth login anthropic" in (info.value.hint or "")
    assert_no_key(info.value)


@pytest.mark.parametrize(
    "error",
    [
        lambda: anthropic.CredentialsError("the profile's refresh token was revoked"),
        lambda: anthropic.WorkloadIdentityError("token exchange failed", status_code=400),
    ],
)
def test_a_failed_sdk_sign_in_is_an_auth_error_with_the_login_hint(
    error: Callable[[], BaseException],
) -> None:
    sdk = FakeSdk(lambda params: error())
    with pytest.raises(ProviderAuthError) as info:
        chat(sdk, key=None).complete(MESSAGES, model="claude-opus-5-5")
    assert info.value.source == "anthropic"
    assert "could not sign in" in info.value.message
    assert "pup finetune auth login anthropic" in (info.value.hint or "")
    assert "ant auth login" in (info.value.hint or "")


def test_timeouts_and_connection_failures_are_transient() -> None:
    request = httpx2.Request("POST", URL)
    for error in (
        anthropic.APITimeoutError(request=request),
        anthropic.APIConnectionError(message="Connection refused", request=request),
    ):
        sdk = FakeSdk(lambda params, error=error: error)
        with pytest.raises(TransientError) as info:
            chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
        assert "api.anthropic.com" in (info.value.hint or "")


def test_an_error_quoting_the_key_is_redacted() -> None:
    error = status_error(anthropic.BadRequestError, 400, detail=f"bad key {API_KEY} here")
    sdk = FakeSdk(lambda params: error)
    with pytest.raises(ExternalError) as info:
        chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
    assert "bad key *** here" in info.value.message
    assert_no_key(info.value)


def test_a_malformed_key_is_refused_before_any_call() -> None:
    sdk = FakeSdk()
    with pytest.raises(UsageError, match="malformed") as info:
        chat(sdk, key="sk-ant two words")
    assert "two words" not in str(info.value)


def test_check_key_lists_one_model_and_reports_a_rejected_key_as_usage() -> None:
    sdk = FakeSdk()
    chat(sdk).check_key()
    assert sdk.listed == [{"limit": 1}]
    sdk.list_error = status_error(anthropic.AuthenticationError, 401)
    with pytest.raises(UsageError, match="Anthropic rejected the API key") as info:
        chat(sdk).check_key()
    assert "pup finetune auth login anthropic" in (info.value.hint or "")
    sdk.list_error = status_error(anthropic.OverloadedError, 529)
    with pytest.raises(ProviderCapacityError):
        chat(sdk).check_key()


# --- credentials -----------------------------------------------------------------------


def test_for_judge_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[SecretStr | None] = []
    original = AnthropicChatClient.__init__

    def record(self: AnthropicChatClient, api_key: SecretStr | None, **options: Any) -> None:
        built.append(api_key)
        original(self, api_key, **options)

    monkeypatch.setattr(AnthropicChatClient, "__init__", record)
    config = parse_judge("anthropic/claude-opus-5-5")
    AnthropicChatClient.for_judge(config, env={"ANTHROPIC_API_KEY": API_KEY}, client=FakeSdk())
    AnthropicChatClient.for_judge(
        config, env={"ANTHROPIC_AUTH_TOKEN": "synthetic-token"}, client=FakeSdk()
    )
    assert [None if key is None else key.get_secret_value() for key in built] == [API_KEY, None]
    with pytest.raises(UsageError, match="no anthropic API key") as info:
        AnthropicChatClient.for_judge(config, env={}, client=FakeSdk())
    assert "pup finetune auth login anthropic" in (info.value.hint or "")


def test_the_key_is_registered_for_redaction() -> None:
    chat(FakeSdk())
    assert redaction.redact(f"x {API_KEY} y") == "x *** y"


def test_the_sdk_s_own_tokens_are_registered_for_redaction() -> None:
    token = "synthetic-sdk-auth-token-5d1e"  # noqa: S105 - synthetic
    error = status_error(anthropic.BadRequestError, 400, detail=f"token {token} is malformed")
    config = parse_judge("anthropic/claude-opus-5-5")
    sdk = FakeSdk(lambda params: error)
    with (
        AnthropicChatClient.for_judge(
            config, env={"ANTHROPIC_AUTH_TOKEN": token}, client=sdk
        ) as client,
        pytest.raises(ExternalError) as info,
    ):
        client.complete(MESSAGES, model="claude-opus-5-5")
    assert "token *** is malformed" in info.value.message
    assert token not in str(info.value)


def test_for_judge_does_not_read_the_keychain_twice_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pup_finetune import credentials

    reads: list[str] = []
    original = credentials._read_keychain

    def counting(spec: Any, backend: Any) -> str | None:
        reads.append(spec.provider)
        return original(spec, backend)

    monkeypatch.setattr(credentials, "_read_keychain", counting)
    with pytest.raises(UsageError, match="no anthropic API key") as info:
        AnthropicChatClient.for_judge(parse_judge("anthropic/claude-opus-5-5"), env={})
    assert reads == ["anthropic"]
    assert "ant auth login" in (info.value.hint or "")


# --- the SDK client it builds -----------------------------------------------------------


def test_the_sdk_client_is_built_with_its_retries_and_timeout() -> None:
    client = AnthropicChatClient(SecretStr(API_KEY), max_attempts=4, timeout_s=12.0)
    sdk = client._client
    assert isinstance(sdk, anthropic.Anthropic)
    assert sdk.max_retries == 3
    assert sdk.timeout == 12.0
    client.close()


def test_a_plain_http_base_url_is_refused() -> None:
    with pytest.raises(UsageError, match="not an https URL"):
        AnthropicChatClient(SecretStr(API_KEY), base_url="http://api.anthropic.example")


def test_an_sdk_sign_in_the_sdk_refuses_is_a_usage_error_with_the_login_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (("ANTHROPIC_CONFIG_DIR", str(tmp_path)), ("ANTHROPIC_PROFILE", "nosuch")):
        monkeypatch.setenv(name, value)
        with pytest.raises(UsageError, match="could not sign in") as info:
            AnthropicChatClient(None)
        assert "pup finetune auth login anthropic" in (info.value.hint or "")
        assert "ant auth login" in (info.value.hint or "")


# --- the real SDK client ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "output_config", "fallbacks"),
    [
        (
            "claude-opus-5-5",
            {"effort": "low", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            True,
        ),
        (
            "claude-opus-4-8",
            {"effort": "low", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            False,
        ),
        ("claude-haiku-4-5", {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}, False),
    ],
)
def test_the_real_sdk_client_sends_the_output_config_and_fallbacks_where_due(
    model: str, output_config: dict[str, Any], fallbacks: bool
) -> None:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_synthetic",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": VERDICT},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 30, "output_tokens": 9},
            },
        )

    sdk = anthropic.Anthropic(
        api_key=API_KEY,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
        max_retries=0,
    )
    with AnthropicChatClient(SecretStr(API_KEY), client=sdk) as client:
        result = client.complete(
            MESSAGES, model=model, max_tokens=16000, response_schema=VERDICT_SCHEMA
        )
    assert result.message["content"] == VERDICT
    assert result.finish_reason == "stop"
    (request,) = seen
    assert str(request.url).startswith(URL)
    assert request.headers["x-api-key"] == API_KEY
    body = json.loads(request.content)
    assert body["system"] == "You judge."
    assert body["output_config"] == output_config
    assert "temperature" not in body
    assert "betas" not in body
    if fallbacks:
        assert FALLBACK_BETA in request.headers["anthropic-beta"]
        assert body["fallbacks"] == "default"
    else:
        assert FALLBACK_BETA not in request.headers.get("anthropic-beta", "")
        assert "fallbacks" not in body


def test_a_prompt_longer_than_the_context_window_is_a_context_limit_error() -> None:
    detail = "prompt is too long: 2200057 tokens > 1000000 maximum"
    sdk = FakeSdk(lambda params: status_error(anthropic.BadRequestError, 400, detail=detail))
    with pytest.raises(ContextLimitError) as info:
        chat(sdk).complete(MESSAGES, model="claude-opus-5-5")
    assert info.value.source == "anthropic"
    assert "context window" in info.value.message
    assert detail in info.value.message
    assert_no_key(info.value)


def test_the_requested_reasoning_effort_is_the_output_effort() -> None:
    sdk = FakeSdk()
    chat(sdk).complete(MESSAGES, model="claude-opus-5-5", reasoning_effort="medium")
    assert sdk.beta.messages.calls[0]["output_config"] == {"effort": "medium"}
    unknown = FakeSdk()
    chat(unknown).complete(MESSAGES, model="claude-3-haiku", reasoning_effort="low")
    assert "output_config" not in unknown.messages.calls[0]


def test_the_real_sdk_client_sends_tools_and_reads_a_tool_use_answer() -> None:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "get_weather",
                     "input": {"city": "Northvale"}},
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 40, "output_tokens": 12},
            },
        )  # fmt: skip

    sdk = anthropic.Anthropic(
        api_key=API_KEY,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
        max_retries=0,
    )
    with AnthropicChatClient(SecretStr(API_KEY), client=sdk) as client:
        result = client.complete(TOOL_TURNS, model="claude-haiku-4-5", tools=[WEATHER_TOOL])
    assert result.message["tool_calls"] == [
        {
            "id": "toolu_01",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
        }
    ]
    assert result.finish_reason == "tool_calls"
    body = json.loads(seen[0].content)
    assert body["tools"][0]["name"] == "get_weather"
    assert body["messages"][2]["content"][1]["type"] == "tool_result"
