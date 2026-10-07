# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The OpenAI-compatible chat client, against a fake endpoint: requests, retries and errors."""

from __future__ import annotations

import email.utils
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.http_common import CLIENT_SOURCE
from pup_finetune.inference import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    MAX_WAIT_S,
    ChatClient,
    ChatCompletion,
    JudgeConfig,
)
from pup_finetune.providers.base import ProviderAuthError, ProviderCapacityError, TransientError
from tests.fakes.chat_api import API_KEY, BASE_URL, MODEL, FakeChatApi, always

NOW = 1_800_000_000.0
MESSAGES = [{"role": "user", "content": "Say hello to Northvale."}]


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


@pytest.fixture(autouse=True)
def _clean_secrets() -> Iterator[None]:
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def fake() -> FakeChatApi:
    return FakeChatApi(answer=always("Hello, Northvale!"))


@pytest.fixture
def sleeps() -> list[float]:
    return []


def client(
    fake: FakeChatApi,
    sleeps: list[float],
    *,
    key: str = API_KEY,
    rng: float = 0.5,
    **options: Any,
) -> ChatClient:
    options.setdefault("credential", "fireworks")
    return ChatClient(
        BASE_URL,
        SecretStr(key),
        transport=fake.transport,
        sleep=sleeps.append,
        rng=FixedRandom(rng),
        clock=lambda: NOW,
        **options,
    )


def assert_no_key(exc: BaseException, key: str = API_KEY) -> None:
    texts = [str(exc), repr(exc)]
    if isinstance(exc, FinetuneError):
        texts += [exc.message, exc.hint or ""]
    for text in texts:
        assert key not in text
        assert key[-8:] not in text


# --- requests ----------------------------------------------------------------------------


def test_complete_posts_a_chat_completion(fake: FakeChatApi, sleeps: list[float]) -> None:
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with client(fake, sleeps) as chat:
        result = chat.complete(MESSAGES, model=MODEL, tools=tools, temperature=0, max_tokens=64)
    assert result == ChatCompletion(
        message={"role": "assistant", "content": "Hello, Northvale!"},
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        model=MODEL,
        finish_reason="stop",
    )
    (request,) = fake.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["user-agent"] == CLIENT_SOURCE
    assert CLIENT_SOURCE.startswith("pup-finetune/")
    assert fake.bodies == [
        {"model": MODEL, "messages": MESSAGES, "tools": tools, "temperature": 0, "max_tokens": 64}
    ]
    assert sleeps == []


def test_unset_options_are_not_sent(fake: FakeChatApi, sleeps: list[float]) -> None:
    with client(fake, sleeps) as chat:
        chat.complete(MESSAGES, model=MODEL, tools=[])
    assert fake.bodies == [{"model": MODEL, "messages": MESSAGES}]


def test_a_trailing_slash_in_the_base_url_is_ignored(fake: FakeChatApi) -> None:
    chat = ChatClient(BASE_URL + "/", SecretStr(API_KEY), transport=fake.transport)
    with chat:
        chat.complete(MESSAGES, model=MODEL)
    assert fake.requests[0].url.path == "/v1/chat/completions"


@pytest.mark.parametrize(
    "base_url", ["http://inference.example.test/v1", "ftp://example.test", "https://", "not a url"]
)
def test_an_insecure_or_bad_base_url_is_refused(base_url: str) -> None:
    with pytest.raises(UsageError):
        ChatClient(base_url, SecretStr(API_KEY))


def test_a_url_httpx_cannot_parse_is_refused() -> None:
    with pytest.raises(UsageError, match="is not a valid URL"):
        ChatClient("https://[::1", SecretStr(API_KEY))


def test_plain_http_is_allowed_on_a_local_host(fake: FakeChatApi) -> None:
    with ChatClient("http://127.0.0.1:8000/v1", SecretStr(API_KEY), transport=fake.transport) as c:
        c.complete(MESSAGES, model=MODEL)
    assert str(fake.requests[0].url) == "http://127.0.0.1:8000/v1/chat/completions"


@pytest.mark.parametrize("key", ["", "two words", "line\nbreak", "naïve"])
def test_an_empty_or_malformed_key_is_refused_before_any_request(
    fake: FakeChatApi, sleeps: list[float], key: str
) -> None:
    with pytest.raises(UsageError, match=r"API key is (empty|malformed)") as info:
        client(fake, sleeps, key=key)
    assert "auth login fireworks" in (info.value.hint or "")
    assert fake.requests == []


def test_max_attempts_must_be_positive(fake: FakeChatApi, sleeps: list[float]) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        client(fake, sleeps, max_attempts=0)


# --- responses ---------------------------------------------------------------------------


def test_missing_fields_read_with_defaults(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script.append(httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    with client(fake, sleeps) as chat:
        result = chat.complete(MESSAGES, model=MODEL)
    assert result == ChatCompletion(
        message={"role": "assistant", "content": "hi"}, usage={}, model=MODEL, finish_reason=None
    )


def test_a_tool_call_message_is_returned_whole(fake: FakeChatApi, sleeps: list[float]) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    message = {"role": "assistant", "content": None, "tool_calls": [call]}
    body = {"choices": [{"message": message, "finish_reason": "tool_calls"}], "model": "served"}
    fake.script.append(httpx.Response(200, json=body))
    with client(fake, sleeps) as chat:
        result = chat.complete(MESSAGES, model=MODEL)
    assert result.message == message
    assert result.model == "served"
    assert result.finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("response", "problem"),
    [
        (httpx.Response(200, json={"id": "x"}), "no choices"),
        (httpx.Response(200, json={"choices": []}), "no choices"),
        (httpx.Response(200, json={"choices": "many"}), "no choices"),
        (httpx.Response(200, json=[1, 2]), "no choices"),
        (httpx.Response(200, json={"error": {"message": "model not found"}}), "no choices: model not found"),
        (httpx.Response(200, json={"choices": [{"text": "hi"}]}), "no message"),
        (httpx.Response(200, json={"choices": ["hi"]}), "no message"),
        (httpx.Response(200, text="<html>ok</html>"), "did not return JSON"),
        (httpx.Response(200, content=b"\xff\xfe\x00"), "did not return JSON"),
    ],
)  # fmt: skip
def test_an_unreadable_answer_is_an_external_error(
    fake: FakeChatApi, sleeps: list[float], response: httpx.Response, problem: str
) -> None:
    fake.script.append(response)
    with client(fake, sleeps) as chat, pytest.raises(ExternalError, match=problem) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert info.value.source == "fireworks"
    assert len(fake.requests) == 1


# --- retries -----------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_retried(
    fake: FakeChatApi, sleeps: list[float], status: int
) -> None:
    fake.script += [httpx.Response(status), httpx.Response(status)]
    with client(fake, sleeps) as chat:
        result = chat.complete(MESSAGES, model=MODEL)
    assert result.message["content"] == "Hello, Northvale!"
    assert len(fake.requests) == 3
    assert sleeps == [0.75, 1.5]


@pytest.mark.parametrize(("rng", "waits"), [(0.0, [0.5, 1.0, 2.0]), (1.0, [1.0, 2.0, 4.0])])
def test_backoff_doubles_with_jitter(
    fake: FakeChatApi, sleeps: list[float], rng: float, waits: list[float]
) -> None:
    fake.script += [httpx.Response(503)] * 3
    with client(fake, sleeps, rng=rng) as chat:
        chat.complete(MESSAGES, model=MODEL)
    assert sleeps == waits


def test_backoff_is_capped(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script += [httpx.Response(503)] * 8
    with client(fake, sleeps, rng=1.0, max_attempts=9) as chat:
        chat.complete(MESSAGES, model=MODEL)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, MAX_WAIT_S, MAX_WAIT_S, MAX_WAIT_S]


@pytest.mark.parametrize(
    ("retry_after", "wait"),
    [
        ("3", 3.0),
        ("0.25", 0.25),
        ("0", 0.0),
        ("-5", 0.0),
        ("120", MAX_WAIT_S),
        (email.utils.formatdate(NOW + 7, usegmt=True), 7.0),
        (email.utils.formatdate(NOW + 3600, usegmt=True), MAX_WAIT_S),
        (email.utils.formatdate(NOW - 60, usegmt=True), 0.0),
    ],
)
@pytest.mark.parametrize("status", [408, 429, 503])
def test_retry_after_is_honored_and_capped(
    fake: FakeChatApi, sleeps: list[float], status: int, retry_after: str, wait: float
) -> None:
    fake.script.append(httpx.Response(status, headers={"Retry-After": retry_after}))
    with client(fake, sleeps) as chat:
        chat.complete(MESSAGES, model=MODEL)
    assert sleeps == [pytest.approx(wait)]


@pytest.mark.parametrize(
    "retry_after",
    ["soon", "", "nan", "Mon, 99 Foo 2026 00:00:00", "Mon, 05 Oct 2026 00:00:00 -0000"],
)
def test_an_unreadable_retry_after_falls_back_to_backoff(
    fake: FakeChatApi, sleeps: list[float], retry_after: str
) -> None:
    fake.script.append(httpx.Response(503, headers={"Retry-After": retry_after}))
    with client(fake, sleeps) as chat:
        chat.complete(MESSAGES, model=MODEL)
    assert sleeps == [0.75]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_client_errors_are_never_retried(
    fake: FakeChatApi, sleeps: list[float], status: int
) -> None:
    fake.script.append(httpx.Response(status, headers={"Retry-After": "1"}))
    with client(fake, sleeps) as chat, pytest.raises(ExternalError):
        chat.complete(MESSAGES, model=MODEL)
    assert len(fake.requests) == 1
    assert sleeps == []


def test_a_redirect_is_not_followed(fake: FakeChatApi, sleeps: list[float]) -> None:
    location = "https://elsewhere.example.test/v1/chat/completions"
    fake.script.append(httpx.Response(307, headers={"Location": location}))
    with client(fake, sleeps) as chat, pytest.raises(ExternalError, match="HTTP 307"):
        chat.complete(MESSAGES, model=MODEL)
    assert len(fake.requests) == 1
    assert sleeps == []


def test_a_body_that_does_not_decode_is_an_external_error(
    fake: FakeChatApi, sleeps: list[float]
) -> None:
    fake.script.append(
        httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"not gzip at all")
        )
    )
    with client(fake, sleeps) as chat, pytest.raises(ExternalError, match="DecodingError") as info:
        chat.complete(MESSAGES, model=MODEL)
    assert info.value.source == "fireworks"
    assert len(fake.requests) == 1
    assert sleeps == []


def test_429_after_every_attempt_is_a_capacity_error(
    fake: FakeChatApi, sleeps: list[float]
) -> None:
    fake.script += [httpx.Response(429, json={"error": {"message": "rate limited"}})] * 4
    with client(fake, sleeps) as chat, pytest.raises(ProviderCapacityError) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert "HTTP 429 after 4 attempts: rate limited" in info.value.message
    assert "retry later" in (info.value.hint or "")
    assert info.value.source == "fireworks"
    assert len(fake.requests) == 4
    assert len(sleeps) == 3


def test_5xx_after_every_attempt_is_transient(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script += [httpx.Response(502)] * 2
    with client(fake, sleeps, max_attempts=2) as chat, pytest.raises(TransientError) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert "HTTP 502 after 2 attempts" in info.value.message
    assert len(fake.requests) == 2


def test_one_attempt_means_no_retry(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script.append(httpx.Response(503))
    with client(fake, sleeps, max_attempts=1) as chat, pytest.raises(TransientError):
        chat.complete(MESSAGES, model=MODEL)
    assert sleeps == []


def test_a_connection_error_is_retried(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script += [httpx.ConnectError("connection refused"), httpx.ReadTimeout("slow")]
    with client(fake, sleeps) as chat:
        result = chat.complete(MESSAGES, model=MODEL)
    assert result.message["content"] == "Hello, Northvale!"
    assert len(fake.requests) == 3
    assert sleeps == [0.75, 1.5]


def test_connection_errors_on_every_attempt_are_transient(
    fake: FakeChatApi, sleeps: list[float]
) -> None:
    fake.script += [httpx.ConnectError(f"refused, key {API_KEY}")] * 4
    with client(fake, sleeps) as chat, pytest.raises(TransientError) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert "could not reach inference.example.test" in info.value.message
    assert "after 4 attempts" in info.value.message
    assert "ConnectError" in info.value.message
    assert_no_key(info.value)


def test_a_timeout_on_every_attempt_says_so(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script += [httpx.ReadTimeout("slow")] * 2
    chat = client(fake, sleeps, max_attempts=2, timeout_s=5)
    with chat, pytest.raises(TransientError, match=r"timed out after 5s, after 2 attempts"):
        chat.complete(MESSAGES, model=MODEL)


# --- errors ------------------------------------------------------------------------------


@pytest.mark.parametrize(("status", "problem"), [(401, "not valid"), (403, "not allowed")])
def test_a_rejected_key_is_an_auth_error_with_a_login_hint(
    fake: FakeChatApi, sleeps: list[float], status: int, problem: str
) -> None:
    fake.script.append(httpx.Response(status, json={"error": {"message": f"bad key {API_KEY}"}}))
    with client(fake, sleeps) as chat, pytest.raises(ProviderAuthError, match=problem) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert "pup finetune auth login fireworks" in (info.value.hint or "")
    assert "FIREWORKS_API_KEY" in (info.value.hint or "")
    assert info.value.source == "fireworks"
    assert_no_key(info.value)


def test_a_wrong_key_is_rejected_by_the_endpoint(sleeps: list[float]) -> None:
    fake = FakeChatApi(api_key="sk-synthetic-other-key")
    with client(fake, sleeps) as chat, pytest.raises(ProviderAuthError):
        chat.complete(MESSAGES, model=MODEL)


def test_an_unknown_credential_gets_a_generic_hint(fake: FakeChatApi, sleeps: list[float]) -> None:
    fake.script.append(httpx.Response(401))
    with (
        client(fake, sleeps, credential="nowhere") as chat,
        pytest.raises(ProviderAuthError) as info,
    ):
        chat.complete(MESSAGES, model=MODEL)
    assert info.value.hint == "check the API key configured for nowhere"
    assert info.value.source == "nowhere"


def test_without_a_credential_errors_name_the_inference_source(fake: FakeChatApi) -> None:
    fake.script.append(httpx.Response(400))
    chat = ChatClient(BASE_URL, SecretStr(API_KEY), transport=fake.transport)
    with chat, pytest.raises(ExternalError) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert info.value.source == "inference"


def test_an_error_body_is_quoted_and_redacted(fake: FakeChatApi, sleeps: list[float]) -> None:
    detail = f"model not found (Authorization: Bearer {API_KEY})"
    fake.script.append(httpx.Response(404, json={"error": {"message": detail}}))
    with client(fake, sleeps) as chat, pytest.raises(ExternalError) as info:
        chat.complete(MESSAGES, model="missing-model")
    assert "a chat completion from missing-model failed with HTTP 404" in info.value.message
    assert "model not found" in info.value.message
    assert info.value.hint == "check the model name and the request"
    assert_no_key(info.value)


@pytest.mark.parametrize(
    "item",
    [
        httpx.Response(401, text=API_KEY),
        httpx.Response(400, text=f"echo {API_KEY}"),
        httpx.Response(500, json={"message": API_KEY}),
        httpx.Response(429, json={"error": API_KEY}),
        httpx.ConnectError(API_KEY),
        httpx.ReadTimeout(API_KEY),
        httpx.Response(200, text=API_KEY),
    ],
)
def test_the_key_never_appears_in_an_error(
    fake: FakeChatApi, sleeps: list[float], item: httpx.Response | Exception
) -> None:
    fake.script += [item] * 4
    with client(fake, sleeps) as chat, pytest.raises(ExternalError) as info:
        chat.complete(MESSAGES, model=MODEL)
    assert_no_key(info.value)
    cause = info.value.__cause__
    if cause is not None and not isinstance(cause, httpx.HTTPError):
        assert_no_key(cause)


# --- judge configuration -----------------------------------------------------------------


def test_the_judge_config_defaults() -> None:
    config = JudgeConfig()
    assert config.model == DEFAULT_JUDGE_MODEL == "accounts/fireworks/models/deepseek-v4p1-flash"
    assert config.base_url == DEFAULT_JUDGE_BASE_URL == "https://api.fireworks.ai/inference/v1"
    assert config.credential == "fireworks"
    assert config.temperature == 0
    assert config.max_tokens == 512


def test_the_judge_config_identity_is_json_and_tracks_every_setting() -> None:
    identity = JudgeConfig().identity()
    assert json.loads(json.dumps(identity)) == identity
    assert identity == {
        "model": DEFAULT_JUDGE_MODEL,
        "base_url": DEFAULT_JUDGE_BASE_URL,
        "credential": "fireworks",
        "temperature": 0.0,
        "max_tokens": 512,
    }
    assert JudgeConfig(model="other").identity() != identity
    assert JudgeConfig(max_tokens=64).identity() != identity


def test_the_judge_config_is_frozen_and_strict() -> None:
    config = JudgeConfig()
    with pytest.raises(ValidationError):
        config.model = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        JudgeConfig(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        JudgeConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        JudgeConfig(model="")


def test_a_judge_client_reads_its_key_from_the_credential(fake: FakeChatApi) -> None:
    config = JudgeConfig(base_url=BASE_URL)
    with ChatClient.for_judge(
        config, env={"FIREWORKS_API_KEY": API_KEY}, transport=fake.transport
    ) as chat:
        chat.complete(MESSAGES, model=config.model)
    assert fake.requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_a_judge_client_without_a_key_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="no fireworks API key"):
        ChatClient.for_judge(JudgeConfig(), env={})


def test_the_key_is_registered_for_redaction(fake: FakeChatApi, sleeps: list[float]) -> None:
    client(fake, sleeps).close()
    assert redaction.redact(f"x {API_KEY} y") == "x *** y"
