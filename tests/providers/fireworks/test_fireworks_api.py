# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The read-only Fireworks REST client, against a fake transport."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import __version__, redaction
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.providers.fireworks.api import (
    CLIENT_SOURCE,
    FireworksApi,
    FireworksAuthError,
    ModelInfo,
    model_path,
)
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi

MODEL_PATH = "/v1/accounts/fireworks/models/qwen3p8-27b"


@pytest.fixture(autouse=True)
def _clean_secrets() -> Iterator[None]:
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def fake() -> FakeFireworksApi:
    return FakeFireworksApi()


def client(fake: FakeFireworksApi, key: str = API_KEY) -> FireworksApi:
    return FireworksApi(SecretStr(key), transport=fake.transport, timeout_s=5)


def assert_no_key(exc: FinetuneError, key: str = API_KEY) -> None:
    for text in (str(exc), exc.message, exc.hint or "", repr(exc)):
        assert key not in text
        assert key[-8:] not in text


# --- requests ----------------------------------------------------------------------------


def test_model_info_reads_the_model(fake: FakeFireworksApi) -> None:
    with client(fake) as api:
        info = api.model_info("accounts/fireworks/models/qwen3p8-27b")
    assert info == ModelInfo(
        name="accounts/fireworks/models/qwen3p8-27b",
        state="READY",
        hugging_face_url="https://huggingface.co/Qwen/Qwen3.8-27B",
        training_context_length=131072,
        supervised_lora_tunable=True,
    )
    (request,) = fake.requests
    assert request.method == "GET"
    assert str(request.url) == f"https://api.fireworks.ai{MODEL_PATH}"


def test_every_request_identifies_the_client_and_sends_the_key(fake: FakeFireworksApi) -> None:
    with client(fake) as api:
        api.model_info("qwen3p8-27b")
        api.list_accounts()
    assert f"pup-finetune/{__version__}" == CLIENT_SOURCE
    for request in fake.requests:
        assert request.headers["x-fireworks-client-source"] == CLIENT_SOURCE
        assert request.headers["user-agent"] == CLIENT_SOURCE
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert [request.url.path for request in fake.requests] == [MODEL_PATH, "/v1/accounts"]


def test_a_bare_model_id_is_in_the_fireworks_account() -> None:
    assert model_path("qwen3p8-27b") == MODEL_PATH
    assert model_path("accounts/my-team/models/my-model") == "/v1/accounts/my-team/models/my-model"


@pytest.mark.parametrize(
    "name",
    ["", "../accounts", "accounts/fireworks/models/../x", "a/b", "Qwen3", "x?y=1", "x/", "-x"],
)
def test_a_name_outside_the_models_collection_is_refused(fake: FakeFireworksApi, name: str) -> None:
    with client(fake) as api, pytest.raises(UsageError, match="not a Fireworks model name"):
        api.model_info(name)
    assert fake.requests == []


def test_a_model_without_optional_fields_reads_with_defaults(fake: FakeFireworksApi) -> None:
    fake.models["bare"] = {"name": "accounts/fireworks/models/bare", "unknownField": [1]}
    with client(fake) as api:
        info = api.model_info("bare")
    assert info.state is None
    assert info.hugging_face_url is None
    assert info.training_context_length is None
    assert info.supervised_lora_tunable is False


def test_get_model_returns_the_record_or_none(fake: FakeFireworksApi) -> None:
    name = "accounts/synthetic-team/models/ft-synthetic-run"
    fake.account_models[name] = {"name": name, "createTime": "2026-10-06T12:00:00Z"}
    with client(fake) as api:
        assert api.get_model(name) == fake.account_models[name]
        assert api.get_model("accounts/synthetic-team/models/ft-missing") is None
    assert [request.url.path for request in fake.requests] == [
        f"/v1/{name}",
        "/v1/accounts/synthetic-team/models/ft-missing",
    ]


def test_get_model_still_fails_on_other_errors(fake: FakeFireworksApi) -> None:
    name = "accounts/synthetic-team/models/ft-synthetic-run"
    fake.status[f"/v1/{name}"] = 503
    with client(fake) as api, pytest.raises(ExternalError, match="HTTP 503"):
        api.get_model(name)


def test_list_accounts_returns_account_ids(fake: FakeFireworksApi) -> None:
    fake.accounts.append({"name": "accounts/second-team"})
    fake.accounts.append({"displayName": "no name"})
    with client(fake) as api:
        assert api.list_accounts() == ["synthetic-team", "second-team"]


def test_list_accounts_with_none(fake: FakeFireworksApi) -> None:
    fake.accounts.clear()
    with client(fake) as api:
        assert api.list_accounts() == []


def test_an_empty_key_is_refused_before_any_request() -> None:
    with pytest.raises(UsageError, match="key is empty"):
        FireworksApi(SecretStr(""))


@pytest.mark.parametrize(
    "key",
    [
        f"{API_KEY}\nfw-second-line",
        f"{API_KEY} fw-after-a-space",
        f"{API_KEY}\tfw-after-a-tab",
        f"{API_KEY}\x00fw-after-a-nul",
        f"{API_KEY}\u00e9fw-not-ascii",
    ],
)
def test_a_key_that_is_not_one_printable_word_is_refused_before_any_request(
    fake: FakeFireworksApi, key: str
) -> None:
    with pytest.raises(UsageError, match="key is malformed") as caught:
        client(fake, key)
    assert fake.requests == []
    assert_no_key(caught.value)
    for part in ("fw-second-line", "fw-after-a-space", "fw-after-a-tab", "fw-after-a-nul"):
        assert part not in str(caught.value)
    assert caught.value.hint is not None
    assert "pup finetune auth login fireworks" in caught.value.hint


def test_the_key_is_registered_for_redaction(fake: FakeFireworksApi) -> None:
    client(fake).close()
    assert redaction.redact(f"leaked {API_KEY}") == "leaked ***"


# --- failures ----------------------------------------------------------------------------


@pytest.mark.parametrize(("status", "problem"), [(401, "not valid"), (403, "not allowed")])
def test_a_rejected_key_is_an_auth_error(fake: FakeFireworksApi, status: int, problem: str) -> None:
    fake.status[MODEL_PATH] = status
    with client(fake) as api, pytest.raises(FireworksAuthError) as caught:
        api.model_info("qwen3p8-27b")
    exc = caught.value
    assert isinstance(exc, ExternalError)
    assert exc.source == "fireworks"
    assert exc.exit_code == 1
    assert problem in exc.message
    assert f"HTTP {status}" in exc.message
    assert exc.hint is not None
    assert "pup finetune auth login fireworks" in exc.hint
    assert_no_key(exc)


def test_a_wrong_key_is_an_auth_error(fake: FakeFireworksApi) -> None:
    other = "fw-synthetic-wrong-key-1234abcd"
    with client(fake, key=other) as api, pytest.raises(FireworksAuthError) as caught:
        api.list_accounts()
    assert_no_key(caught.value, key=other)


def test_a_missing_model_is_a_usage_error(fake: FakeFireworksApi) -> None:
    with client(fake) as api, pytest.raises(UsageError, match="Fireworks has no model 'gone'"):
        api.model_info("gone")


def test_a_missing_route_elsewhere_is_external(fake: FakeFireworksApi) -> None:
    fake.status["/v1/accounts"] = 404
    with client(fake) as api, pytest.raises(ExternalError, match="HTTP 404"):
        api.list_accounts()


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_a_retryable_status_is_external_with_a_retry_hint(
    fake: FakeFireworksApi, status: int
) -> None:
    fake.status[MODEL_PATH] = status
    with client(fake) as api, pytest.raises(ExternalError) as caught:
        api.model_info("qwen3p8-27b")
    exc = caught.value
    assert not isinstance(exc, FireworksAuthError)
    assert exc.source == "fireworks"
    assert f"HTTP {status}: synthetic failure" in exc.message
    assert exc.hint == "retry later"
    assert str(exc).startswith("fireworks: ")


def test_a_client_error_has_no_retry_hint(fake: FakeFireworksApi) -> None:
    fake.status[MODEL_PATH] = 400
    fake.bodies[MODEL_PATH] = json.dumps({"message": "bad request"})
    with client(fake) as api, pytest.raises(ExternalError) as caught:
        api.model_info("qwen3p8-27b")
    assert caught.value.message.endswith("HTTP 400: bad request")
    assert caught.value.hint is None


def test_an_error_body_echoing_the_key_is_redacted(fake: FakeFireworksApi) -> None:
    fake.status[MODEL_PATH] = 500
    fake.bodies[MODEL_PATH] = f"internal error for key {API_KEY} and Bearer {API_KEY}"
    with client(fake) as api, pytest.raises(ExternalError) as caught:
        api.model_info("qwen3p8-27b")
    assert "***" in caught.value.message
    assert_no_key(caught.value)


def test_a_long_error_body_is_cut_short(fake: FakeFireworksApi) -> None:
    fake.status[MODEL_PATH] = 500
    fake.bodies[MODEL_PATH] = "x" * 5000 + "\nsecond line"
    with client(fake) as api, pytest.raises(ExternalError) as caught:
        api.model_info("qwen3p8-27b")
    assert len(caught.value.message) < 300
    assert "second line" not in caught.value.message


TransportError = Callable[[httpx.Request], Exception]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (lambda r: httpx.ReadTimeout("read timed out", request=r), "timed out after 5s"),
        (lambda r: httpx.ConnectTimeout("connect timed out", request=r), "timed out after 5s"),
        (
            lambda r: httpx.ConnectError(f"refused, key {API_KEY}", request=r),
            "could not reach the Fireworks API (ConnectError: refused, key ***)",
        ),
    ],
    ids=["read-timeout", "connect-timeout", "connect-error"],
)
def test_transport_failures_are_external(
    fake: FakeFireworksApi, error: TransportError, expected: str
) -> None:
    fake.raise_on[MODEL_PATH] = error(httpx.Request("GET", f"https://api.fireworks.ai{MODEL_PATH}"))
    with client(fake) as api, pytest.raises(ExternalError) as caught:
        api.model_info("qwen3p8-27b")
    assert expected in caught.value.message
    assert caught.value.source == "fireworks"
    assert caught.value.hint is not None
    assert "api.fireworks.ai" in caught.value.hint
    assert_no_key(caught.value)


def test_a_body_that_is_not_json_is_external() -> None:
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    api = FireworksApi(SecretStr(API_KEY), transport=httpx.MockTransport(html))
    with api, pytest.raises(ExternalError, match="did not return JSON"):
        api.model_info("qwen3p8-27b")


def test_an_unreadable_model_is_external(fake: FakeFireworksApi) -> None:
    fake.models["odd"] = {"displayName": "no name", "trainingContextLength": "many"}
    with client(fake) as api, pytest.raises(ExternalError, match="unreadable model"):
        api.model_info("odd")


def test_an_unreadable_account_list_is_external() -> None:
    def odd(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"accounts": "nope"})

    with (
        FireworksApi(SecretStr(API_KEY), transport=httpx.MockTransport(odd)) as api,
        pytest.raises(ExternalError, match="unreadable account list"),
    ):
        api.list_accounts()


def test_redirects_are_not_followed() -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://elsewhere.example/"})

    with (
        FireworksApi(SecretStr(API_KEY), transport=httpx.MockTransport(redirect)) as api,
        pytest.raises(ExternalError, match="HTTP 302"),
    ):
        api.list_accounts()
