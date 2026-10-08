# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Anthropic and OpenAI judge keys: registration, `auth status`, `doctor`, validators."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from pup_finetune import credentials, judge_credentials, redaction
from pup_finetune.cli import app
from pup_finetune.cli import doctor as doctor_cli
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.judge_credentials import (
    ANTHROPIC,
    OPENAI,
    OPENAI_MODELS_URL,
    anthropic_sdk_login,
    validate_anthropic_key,
    validate_openai_key,
)
from pup_finetune.judges import ANTHROPIC_INSTALL_HINT
from tests.fakes.pup import FakePup

# Before the autouse fixture hides the developer's own config directory.
_REAL_SDK_CONFIG_DIR = judge_credentials._sdk_config_dir

runner = CliRunner()
OPENAI_KEY = "sk-openai-synthetic-0000-judge-key-91fb"
ANTHROPIC_KEY = "sk-ant-synthetic-0000-judge-key-4c2e"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def status(provider: str) -> dict[str, object]:
    result = runner.invoke(app, ["auth", "status", provider])
    assert result.exit_code == 0, result.stderr
    (entry,) = json.loads(result.stdout)["providers"]
    assert isinstance(entry, dict)
    return entry


def doctor_checks(monkeypatch: pytest.MonkeyPatch, home: Path) -> dict[str, dict[str, object]]:
    for name in ("FIREWORKS_API_KEY", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    signed_in = {"authenticated": True, "site": "datadoghq.com", "expires_at": None}
    pup = FakePup().on("--version", raw=True, stdout="pup 1.10.7\n")
    pup.on("auth", "status", stdout=json.dumps(signed_in))
    monkeypatch.setattr(doctor_cli, "Pup", pup.client)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout + result.stderr
    return {check["name"]: check for check in json.loads(result.stdout)["checks"]}


# --- registration ----------------------------------------------------------------------


def test_the_specs() -> None:
    assert (ANTHROPIC.provider, ANTHROPIC.env_var) == ("anthropic", "ANTHROPIC_API_KEY")
    assert (OPENAI.provider, OPENAI.env_var) == ("openai", "OPENAI_API_KEY")
    assert ANTHROPIC.keychain_account == "anthropic"
    assert OPENAI.keychain_account == "openai"
    assert ANTHROPIC.keychain_service == OPENAI.keychain_service == "pup-finetune"
    assert ANTHROPIC.sdk_login is anthropic_sdk_login
    assert OPENAI.sdk_login is None
    for spec in (ANTHROPIC, OPENAI):
        assert credentials.get_spec(spec.provider) is spec
        assert spec.validator is not None
    assert ANTHROPIC.needed_for == "anthropic/... judges"
    assert (OPENAI.needed_for, OPENAI.label) == ("openai/... judges", "OpenAI")


def test_auth_status_lists_both_without_keys() -> None:
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0, result.stderr
    entries = {entry["provider"]: entry for entry in json.loads(result.stdout)["providers"]}
    assert entries["anthropic"] == {
        "provider": "anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "configured": False,
        "source": None,
        "needed_for": "anthropic/... judges",
    }
    assert entries["openai"]["configured"] is False


def test_auth_status_reports_the_source_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    for provider in ("openai", "anthropic"):
        entry = status(provider)
        assert (entry["configured"], entry["source"]) == (True, "env")
    assert OPENAI_KEY not in json.dumps([status("openai"), status("anthropic")])


def test_doctor_reports_missing_keys_without_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checks = doctor_checks(monkeypatch, tmp_path)
    for provider in ("anthropic", "openai"):
        check = checks[f"{provider}:key"]
        assert check["required"] is False
        assert check["status"] == "optional"
        assert check["needed_for"] == f"{provider}/... judges"
        assert f"pup finetune auth login {provider}" in str(check["fix"])


# --- the Anthropic SDK's own login ------------------------------------------------------


def test_an_auth_token_is_the_anthropic_sdk_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "synthetic-token")
    assert status("anthropic") == {
        "provider": "anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "configured": True,
        "source": "anthropic-sdk",
        "needed_for": "anthropic/... judges",
    }
    check = doctor_checks(monkeypatch, tmp_path)["anthropic:key"]
    assert check["status"] == "pass"
    assert check["label"] == "Anthropic key (Anthropic SDK login)"
    assert "source: anthropic-sdk" in str(check["detail"])
    assert "synthetic-token" not in str(check["detail"])


def test_a_key_wins_over_the_sdk_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "synthetic-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    assert status("anthropic")["source"] == "env"


@pytest.mark.parametrize(
    "env",
    [
        {"ANTHROPIC_AUTH_TOKEN": "t"},
        {
            "ANTHROPIC_FEDERATION_RULE_ID": "r",
            "ANTHROPIC_ORGANIZATION_ID": "o",
            "ANTHROPIC_IDENTITY_TOKEN_FILE": "/var/run/token",
        },
    ],
)
def test_the_sdk_login_is_detected_from_the_environment(env: dict[str, str]) -> None:
    assert anthropic_sdk_login(env)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"ANTHROPIC_AUTH_TOKEN": "  "},
        {"ANTHROPIC_FEDERATION_RULE_ID": "r", "ANTHROPIC_ORGANIZATION_ID": "o"},
        {"OPENAI_API_KEY": "x"},
    ],
)
def test_no_sdk_login_without_its_signals(env: dict[str, str]) -> None:
    assert not anthropic_sdk_login(env)


def test_a_selected_profile_counts_only_when_its_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_credentials, "_sdk_config_dir", _REAL_SDK_CONFIG_DIR)
    empty = {"ANTHROPIC_CONFIG_DIR": str(tmp_path)}
    assert not anthropic_sdk_login(empty)
    assert not anthropic_sdk_login({**empty, "ANTHROPIC_PROFILE": "nosuch"})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "work.json").write_text("{}")
    assert anthropic_sdk_login({**empty, "ANTHROPIC_PROFILE": "work"})
    assert not anthropic_sdk_login(empty)  # the default profile is still missing
    assert not anthropic_sdk_login({**empty, "ANTHROPIC_PROFILE": "../configs/work"})
    # An explicit profile wins over workload identity federation, as in the SDK.
    federation = {
        "ANTHROPIC_FEDERATION_RULE_ID": "r",
        "ANTHROPIC_ORGANIZATION_ID": "o",
        "ANTHROPIC_IDENTITY_TOKEN_FILE": "/var/run/token",
    }
    assert not anthropic_sdk_login({**empty, **federation})


def test_an_ant_login_profile_on_disk_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_credentials, "_sdk_config_dir", lambda env: tmp_path)
    assert not anthropic_sdk_login({})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.json").write_text("{}")
    assert anthropic_sdk_login({})
    (tmp_path / "configs" / "default.json").unlink()
    (tmp_path / "active_config").write_text("work\n")
    assert not anthropic_sdk_login({})  # the pointer names a profile that is missing
    (tmp_path / "configs" / "work.json").write_text("{}")
    assert anthropic_sdk_login({})


# --- validators ------------------------------------------------------------------------


def openai_transport(status_code: int, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status_code == 200:
            return httpx.Response(200, json={"object": "list", "data": []})
        message = f"Incorrect API key provided: {OPENAI_KEY}"
        return httpx.Response(status_code, json={"error": {"message": message}})

    return httpx.MockTransport(handle)


def test_the_openai_validator_lists_models_with_the_key() -> None:
    seen: list[httpx.Request] = []
    key = SecretStr(OPENAI_KEY)
    assert validate_openai_key(key, transport=openai_transport(200, seen)) is None
    (request,) = seen
    assert request.method == "GET"
    assert str(request.url) == OPENAI_MODELS_URL
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"


@pytest.mark.parametrize(
    ("code", "kind"), [(401, UsageError), (403, UsageError), (500, ExternalError)]
)
def test_the_openai_validator_reports_failures_without_the_key(
    code: int, kind: type[Exception]
) -> None:
    with pytest.raises(kind) as info:
        validate_openai_key(SecretStr(OPENAI_KEY), transport=openai_transport(code, []))
    assert type(info.value) is kind
    text = str(info.value) + repr(info.value) + str(getattr(info.value, "hint", ""))
    assert OPENAI_KEY not in text
    assert f"HTTP {code}" in str(info.value)
    if kind is UsageError:
        assert "pup finetune auth login openai" in str(getattr(info.value, "hint", ""))


def test_the_openai_validator_reports_an_unreachable_api() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ExternalError, match="could not reach OpenAI"):
        validate_openai_key(SecretStr(OPENAI_KEY), transport=httpx.MockTransport(refuse))


def test_auth_login_openai_validates_then_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring.core

    from tests.fakes.keyring import OS_KEYCHAIN_FAKES

    backend = OS_KEYCHAIN_FAKES[0]()
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend)
    checked: list[str] = []

    def validator(key: SecretStr) -> None:
        checked.append(key.get_secret_value())

    monkeypatch.setattr(
        credentials,
        "_REGISTRY",
        {
            **credentials._REGISTRY,
            "openai": credentials.CredentialSpec(
                provider="openai", env_var="OPENAI_API_KEY", validator=validator
            ),
        },
    )
    result = runner.invoke(app, ["auth", "login", "openai", "--stdin"], input=OPENAI_KEY)
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"provider": "openai", "stored": True, "source": "keychain"}
    assert checked == [OPENAI_KEY]
    assert OPENAI_KEY not in result.stdout + result.stderr
    assert status("openai")["source"] == "keychain"


def test_the_anthropic_validator_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "pup_finetune.anthropic_client", raising=False)
    with pytest.raises(UsageError, match="'anthropic' package") as info:
        validate_anthropic_key(SecretStr(ANTHROPIC_KEY))
    assert info.value.hint == ANTHROPIC_INSTALL_HINT


def test_auth_login_anthropic_validates_then_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring.core

    from tests.fakes.keyring import OS_KEYCHAIN_FAKES

    backend = OS_KEYCHAIN_FAKES[0]()
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend)
    checked: list[str] = []

    def validator(key: SecretStr) -> None:
        checked.append(key.get_secret_value())

    spec = credentials.CredentialSpec(
        provider="anthropic",
        env_var="ANTHROPIC_API_KEY",
        keychain_account="anthropic",
        validator=validator,
    )
    monkeypatch.setattr(credentials, "_REGISTRY", {**credentials._REGISTRY, "anthropic": spec})
    result = runner.invoke(app, ["auth", "login", "anthropic", "--stdin"], input=ANTHROPIC_KEY)
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "provider": "anthropic",
        "stored": True,
        "source": "keychain",
    }
    assert checked == [ANTHROPIC_KEY]
    assert ANTHROPIC_KEY not in result.stdout + result.stderr
    assert backend.get_password("pup-finetune", "anthropic") == ANTHROPIC_KEY
    assert status("anthropic")["source"] == "keychain"
