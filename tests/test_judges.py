# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Judge specs by provider prefix, provider defaults, and the client each judge gets."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from pup_finetune import judges, readiness
from pup_finetune.errors import UsageError
from pup_finetune.inference import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    ChatClient,
    ChatCompleter,
    JudgeConfig,
)
from pup_finetune.judge_models import EVALUATE_DEFAULTS, JUDGE_SHORT_NAMES, alias_of
from pup_finetune.judges import (
    ANTHROPIC_INSTALL_HINT,
    FIREWORKS_REASONING,
    JUDGE_PACKAGES,
    anthropic_client_class,
    check_available,
    destination,
    open_client,
    open_clients,
    parse_judge,
)

ANTHROPIC_KEY = "sk-ant-synthetic-0000-judge-key"
OPENAI_KEY = "sk-openai-synthetic-0000-judge-key"


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# --- parsing --------------------------------------------------------------------------


def test_an_anthropic_spec() -> None:
    config = parse_judge("anthropic/claude-opus-5-5")
    assert config == JudgeConfig(
        provider="anthropic",
        model="claude-opus-5-5",
        base_url="https://api.anthropic.com",
        credential="anthropic",
        temperature=None,
        max_tokens=16000,
        reasoning_effort="low",
    )
    assert config.spec == "anthropic/claude-opus-5-5"


def test_an_openai_spec() -> None:
    config = parse_judge(" openai/gpt-6-astra ")
    assert config.provider == "openai"
    assert config.model == "gpt-6-astra"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.credential == "openai"
    assert config.temperature is None
    assert config.max_tokens == 16000
    assert config.reasoning_effort == "low"  # the lowest the current model accepts
    assert config.spec == "openai/gpt-6-astra"


@pytest.mark.parametrize("spec", [DEFAULT_JUDGE_MODEL, f" {DEFAULT_JUDGE_MODEL} "])
def test_a_fireworks_spec_is_the_default_judge_s_settings(spec: str) -> None:
    config = parse_judge(spec)
    assert config == JudgeConfig(reasoning_effort="none")
    assert config.base_url == DEFAULT_JUDGE_BASE_URL
    assert config.spec == DEFAULT_JUDGE_MODEL


@pytest.mark.parametrize(
    ("model", "effort"),
    [("gpt-6-astra", "low"), ("o4-mini", "low"), ("gpt-4.1-mini", None), ("GPT-4o", None)],
)
def test_an_openai_judge_sends_an_effort_only_to_a_reasoning_model(
    model: str, effort: str | None
) -> None:
    # GPT-4-era models refuse the field ("Unrecognized request argument supplied").
    assert parse_judge(f"openai/{model}").reasoning_effort == effort


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("accounts/fireworks/models/deepseek-v4p1-flash", "none"),
        ("accounts/fireworks/models/glm-5p3-flash", "low"),  # thinking only: refuses none
        ("accounts/fireworks/models/another-judge", None),
        ("accounts/test/models/judge-x", None),
    ],
)
def test_a_fireworks_judge_asks_for_the_lowest_effort_its_model_takes(
    model: str, effort: str | None
) -> None:
    assert parse_judge(model).reasoning_effort == effort


@pytest.mark.parametrize(
    ("spec", "close"), [("glm-5p3-flsh", "glm-5p3-flash"), ("fake-judge", None), ("judge-x", None)]
)
def test_a_spec_without_a_slash_must_be_a_known_alias(spec: str, close: str | None) -> None:
    # A mistyped alias would otherwise be sent, as a model id, to a paid run.
    with pytest.raises(UsageError, match=rf"^unknown judge '{spec}'") as info:
        parse_judge(spec)
    hint = info.value.hint or ""
    if close is not None:
        assert hint.startswith(f"did you mean {close}? ")
    else:
        assert not hint.startswith("did you mean")
    for form in ("deepseek-v4p1-flash", "anthropic/MODEL", "openai/MODEL", "accounts/<acct>"):
        assert form in hint


def test_a_bare_fireworks_id_of_another_account_stays_fireworks() -> None:
    config = parse_judge("accounts/my-team/models/judge")
    assert (config.provider, config.model) == ("fireworks", "accounts/my-team/models/judge")


@pytest.mark.parametrize(
    "spec", ["accounts/test/models/judge-one", "gemini/pro", "fireworks/judge-x"]
)
def test_any_other_spec_with_a_slash_is_a_fireworks_model_id_as_written(spec: str) -> None:
    # Only anthropic/ and openai/ name a provider; there is no fireworks/ alias.
    config = parse_judge(spec)
    assert (config.provider, config.model) == ("fireworks", spec)
    assert config.spec == spec
    assert parse_judge(config.spec) == config


def test_a_provider_prefix_is_read_in_any_case() -> None:
    config = parse_judge("Anthropic/claude-opus-5-5")
    assert (config.provider, config.model) == ("anthropic", "claude-opus-5-5")
    assert config.spec == "anthropic/claude-opus-5-5"


@pytest.mark.parametrize(
    "spec",
    ["anthropic/claude-opus-5-5", "openai/gpt-x", DEFAULT_JUDGE_MODEL, "glm-5p3-flash"],
)
def test_a_spec_round_trips(spec: str) -> None:
    assert parse_judge(parse_judge(spec).spec) == parse_judge(spec)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "is empty"),
        ("   ", "is empty"),
        ("anthropic/", "has no model"),
        ("openai/  ", "has no model"),
    ],
)
def test_a_bad_spec_is_a_usage_error(spec: str, message: str) -> None:
    with pytest.raises(UsageError, match=message) as info:
        parse_judge(spec)
    if "empty" in message:
        assert "anthropic/<model>" in (info.value.hint or "")


def test_the_identity_includes_the_provider() -> None:
    fireworks = JudgeConfig(model="claude-opus-5-5")
    anthropic = JudgeConfig(provider="anthropic", model="claude-opus-5-5")
    assert fireworks.identity()["provider"] == "fireworks"
    assert anthropic.identity()["provider"] == "anthropic"
    assert fireworks.identity() != anthropic.identity()


def test_the_destination_names_the_provider_and_host() -> None:
    assert destination(JudgeConfig()) == "Fireworks (api.fireworks.ai)"
    assert destination(parse_judge("anthropic/claude-opus-5-5")) == "Anthropic (api.anthropic.com)"
    assert destination(parse_judge("openai/gpt-x")) == "OpenAI (api.openai.com)"


# --- availability ----------------------------------------------------------------------


def test_a_missing_anthropic_extra_is_a_usage_error_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "pup_finetune.anthropic_client", raising=False)
    config = parse_judge("anthropic/claude-opus-5-5")
    for attempt in (lambda: check_available(config), anthropic_client_class):
        with pytest.raises(UsageError, match="'anthropic' package") as info:
            attempt()
        assert info.value.hint == ANTHROPIC_INSTALL_HINT
        # uv sync is exact: the hint never offers a bare --extra, which drops the others.
        assert "`uv sync --all-extras`" in ANTHROPIC_INSTALL_HINT
        assert "a bare `uv sync --extra anthropic` removes the other extras" in (
            ANTHROPIC_INSTALL_HINT
        )
        assert "pip install 'pup-finetune[anthropic]'" in ANTHROPIC_INSTALL_HINT
    with pytest.raises(UsageError, match="'anthropic' package"), open_client(config):
        pass


def test_an_anthropic_sdk_without_httpx2_is_refused_in_the_preview_as_doctor_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "httpx2", None)
    with pytest.raises(
        UsageError, match="need the 'httpx2' package, which is not installed"
    ) as info:
        check_available(parse_judge("anthropic/claude-opus-5-5"))
    assert info.value.hint == ANTHROPIC_INSTALL_HINT
    package = JUDGE_PACKAGES["anthropic"]
    assert not readiness.package_check("anthropic", "Anthropic", package, required=False).ok
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    chosen, source = judges.default_judges(EVALUATE_DEFAULTS)
    assert chosen == (DEFAULT_JUDGE_MODEL,)
    assert source.endswith("Anthropic judges need the 'httpx2' package, which is not installed")


def test_other_judges_need_no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    check_available(parse_judge("openai/gpt-x"))
    check_available(JudgeConfig())


# --- clients ---------------------------------------------------------------------------


def test_an_openai_judge_sends_openai_s_request_to_api_openai_com(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = {"role": "assistant", "content": '{"pass": true, "reason": "ok"}'}
        return httpx.Response(200, json={"choices": [{"message": answer, "finish_reason": "stop"}]})

    original = ChatClient.__init__

    def on_mock_transport(self: ChatClient, *args: object, **options: object) -> None:
        options["transport"] = httpx.MockTransport(handle)
        original(self, *args, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(ChatClient, "__init__", on_mock_transport)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    config = parse_judge("openai/gpt-x")
    with open_client(config) as client:
        client.complete(
            [{"role": "user", "content": "judge"}],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_schema={"type": "object"},
        )
    (request,) = seen
    assert (request.url.host, request.url.path) == ("api.openai.com", "/v1/chat/completions")
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "gpt-x"
    assert body["max_completion_tokens"] == 16000
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body
    assert "max_tokens" not in body


def test_an_anthropic_judge_gets_the_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from pup_finetune.anthropic_client import AnthropicChatClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    with open_client(parse_judge("anthropic/claude-opus-5-5")) as client:
        assert isinstance(client, AnthropicChatClient)


def test_a_judge_without_its_key_is_a_usage_error_naming_its_login() -> None:
    for spec, provider in (("openai/gpt-x", "openai"), ("anthropic/claude-opus-5-5", "anthropic")):
        with (
            pytest.raises(UsageError, match=f"no {provider} API key") as info,
            open_client(parse_judge(spec)),
        ):
            pass
        assert f"pup finetune auth login {provider}" in (info.value.hint or "")


def test_a_council_gets_one_client_per_provider_opened_on_its_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    opened: list[str] = []
    closed: list[str] = []
    real_open_client = judges.open_client

    @contextmanager
    def recording(config: JudgeConfig, **options: object) -> Iterator[ChatCompleter]:
        with real_open_client(config) as client:
            opened.append(config.provider)
            try:
                yield client
            finally:
                closed.append(config.provider)

    monkeypatch.setattr(judges, "open_client", recording)
    monkeypatch.setattr(ChatClient, "complete", lambda self, messages, **options: None)
    # No Anthropic key: the council still opens, as no call has needed that provider yet.
    council = [
        parse_judge("openai/gpt-x"),
        parse_judge("anthropic/claude-opus-5-5"),
        parse_judge("openai/gpt-y"),
    ]
    with open_clients(council) as clients:
        assert sorted(clients) == ["anthropic", "openai"]
        assert opened == []
        for _ in range(2):
            clients["openai"].complete([{"role": "user", "content": "judge"}], model="gpt-x")
        assert opened == ["openai"]
        with pytest.raises(UsageError, match="no anthropic API key") as info:
            clients["anthropic"].complete([], model="claude-opus-5-5")
        assert "pup finetune auth login anthropic" in (info.value.hint or "")
    assert opened == closed == ["openai"]


def test_the_fake_judge_has_no_remote_client() -> None:
    from pup_finetune.evaluation.fake_judge import fake_judge_config

    with (
        pytest.raises(UsageError, match="no client serves fake judges"),
        open_client(fake_judge_config()),
    ):
        pass


def test_the_providers_table_covers_every_remote_provider() -> None:
    assert sorted(judges.PROVIDERS) == ["anthropic", "fireworks", "openai"]
    assert all(d.base_url.startswith("https://") for d in judges.PROVIDERS.values())


@pytest.mark.parametrize("alias", sorted(JUDGE_SHORT_NAMES))
def test_an_alias_resolves_to_its_fireworks_model_id(alias: str) -> None:
    config = parse_judge(alias)
    assert config.provider == "fireworks"
    assert config.model == JUDGE_SHORT_NAMES[alias]
    # The spec a preview prints and a directory freezes is the full id, never the alias.
    assert config.spec == JUDGE_SHORT_NAMES[alias]
    assert parse_judge(config.spec) == config
    assert config.reasoning_effort == FIREWORKS_REASONING[JUDGE_SHORT_NAMES[alias]]


def test_the_default_judges_have_aliases() -> None:
    assert alias_of(DEFAULT_JUDGE_MODEL) == "deepseek-v4p1-flash"
    assert alias_of("accounts/my-team/models/judge") == "accounts/my-team/models/judge"
    assert "deepseek-v4p1-flash" in judges.SPEC_FORMS


def test_prefixed_specs_are_never_aliases() -> None:
    assert parse_judge("openai/glm-5p3-flash").model == "glm-5p3-flash"
