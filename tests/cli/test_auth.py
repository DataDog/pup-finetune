# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune auth`, end to end through the Typer app with a fake keychain."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import keyring.core
import pytest
import typer
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from pup_finetune import credentials, redaction
from pup_finetune import doctor as doctor_registry
from pup_finetune.cli import app
from pup_finetune.cli import auth as auth_cli
from pup_finetune.credentials import CredentialSpec
from pup_finetune.errors import UsageError
from tests.fakes.keyring import FailKeyring, LockedKeyring, MemoryKeyring, StickyKeyring

SECRET = "zq9Xsynthetic0000keyVALUEw7Pm"  # noqa: S105 - synthetic test value
ACME = CredentialSpec(provider="acme", env_var="ACME_API_KEY", description="Acme test provider")
BETA = CredentialSpec(provider="beta", env_var="BETA_API_KEY")

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setattr(credentials, "_REGISTRY", {})
    monkeypatch.setattr(doctor_registry, "_REGISTRY", {})
    for name in ("ACME_API_KEY", "BETA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def keychain(monkeypatch: pytest.MonkeyPatch) -> MemoryKeyring:
    backend = MemoryKeyring()
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend)
    return backend


def invoke(*args: str, stdin: str | None = None) -> Result:
    return runner.invoke(app, ["auth", *args], input=stdin)


def stored(backend: MemoryKeyring, provider: str = "acme") -> str | None:
    return backend.passwords.get(("pup-finetune", provider))


def assert_no_leak(text: str, secret: str = SECRET) -> None:
    assert secret not in text
    assert secret[:4] not in text
    assert secret[-4:] not in text


def file_contents(root: Path) -> list[bytes]:
    return [path.read_bytes() for path in root.rglob("*") if path.is_file()]


# --- status -----------------------------------------------------------------------------


def test_status_with_no_providers_is_an_empty_list() -> None:
    result = invoke("status")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"providers": []}


def test_status_reports_each_provider_and_source(
    keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials.register(BETA)
    credentials.register(ACME)
    result = invoke("status")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "providers": [
            {"provider": "acme", "env_var": "ACME_API_KEY", "configured": False, "source": None},
            {"provider": "beta", "env_var": "BETA_API_KEY", "configured": False, "source": None},
        ]
    }

    keychain.set_password("pup-finetune", "acme", SECRET)
    monkeypatch.setenv("BETA_API_KEY", SECRET[::-1])
    result = invoke("status")
    assert result.exit_code == 0, result.stderr
    entries = json.loads(result.stdout)["providers"]
    assert [(e["provider"], e["configured"], e["source"]) for e in entries] == [
        ("acme", True, "keychain"),
        ("beta", True, "env"),
    ]
    assert_no_leak(result.stdout + result.stderr)
    assert_no_leak(result.stdout + result.stderr, SECRET[::-1])


def test_status_for_one_provider(keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.register(ACME)
    credentials.register(BETA)
    keychain.set_password("pup-finetune", "acme", SECRET)
    monkeypatch.setenv("ACME_API_KEY", SECRET)
    result = invoke("status", "acme")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "providers": [
            {"provider": "acme", "env_var": "ACME_API_KEY", "configured": True, "source": "env"}
        ]
    }
    assert_no_leak(result.stdout + result.stderr)


def test_status_reports_a_locked_keychain_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyring.core, "_keyring_backend", LockedKeyring())
    credentials.register(ACME)
    result = invoke("status")
    assert result.exit_code == 0, result.stderr
    [entry] = json.loads(result.stdout)["providers"]
    assert entry["configured"] is False
    assert entry["source"] is None
    assert "could not read the acme key" in entry["error"]


def test_unknown_provider_is_a_usage_error_listing_the_registered_ones() -> None:
    credentials.register(ACME)
    credentials.register(BETA)
    for args in (("status", "nope"), ("login", "nope", "--stdin"), ("logout", "nope")):
        result = invoke(*args, stdin=SECRET)
        assert result.exit_code == 2, args
        assert "unknown provider 'nope'" in result.stderr
        assert "registered providers: acme, beta" in result.stderr
        assert result.stdout == ""


# --- login ------------------------------------------------------------------------------


def test_login_from_stdin_stores_the_stripped_key(keychain: MemoryKeyring) -> None:
    credentials.register(ACME)
    result = invoke("login", "acme", "--stdin", stdin=f"  {SECRET}\n")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"provider": "acme", "stored": True, "source": "keychain"}
    assert stored(keychain) == SECRET
    assert_no_leak(result.stdout + result.stderr)

    status = json.loads(invoke("status", "acme").stdout)["providers"][0]
    assert status["source"] == "keychain"


def test_login_prompts_with_hidden_input_on_a_terminal(
    keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def prompt_hidden(prompt: str) -> str:
        prompts.append(prompt)
        return SECRET

    monkeypatch.setattr(auth_cli, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(auth_cli, "_prompt_hidden", prompt_hidden)
    credentials.register(ACME)
    result = invoke("login", "acme")
    assert result.exit_code == 0, result.stderr
    assert prompts == ["acme API key (input is hidden): "]
    assert stored(keychain) == SECRET
    assert_no_leak(result.stdout + result.stderr)


def test_login_with_stdin_on_a_terminal_prompts_hidden_instead(
    keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def prompt_hidden(prompt: str) -> str:
        prompts.append(prompt)
        return SECRET

    monkeypatch.setattr(auth_cli, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(auth_cli, "_prompt_hidden", prompt_hidden)
    credentials.register(ACME)
    result = invoke("login", "acme", "--stdin", stdin="not-the-key")
    assert result.exit_code == 0, result.stderr
    assert prompts == ["acme API key (input is hidden): "]
    assert stored(keychain) == SECRET


def test_login_end_of_input_at_the_prompt_is_a_usage_error(
    keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    def prompt_hidden(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr(auth_cli, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(auth_cli, "_prompt_hidden", prompt_hidden)
    credentials.register(ACME)
    result = invoke("login", "acme")
    assert result.exit_code == 2
    assert "no acme API key was entered" in result.stderr
    assert "--stdin" in result.stderr
    assert "unexpected" not in result.stderr
    assert keychain.passwords == {}


def test_login_without_a_terminal_or_stdin_flag_refuses(keychain: MemoryKeyring) -> None:
    credentials.register(ACME)
    result = invoke("login", "acme", stdin=SECRET)
    assert result.exit_code == 2
    assert "no terminal" in result.stderr
    assert "--stdin" in result.stderr
    assert keychain.passwords == {}


def test_login_rejects_an_empty_key(keychain: MemoryKeyring) -> None:
    credentials.register(ACME)
    result = invoke("login", "acme", "--stdin", stdin=" \n")
    assert result.exit_code == 2
    assert "no acme API key was entered" in result.stderr
    assert keychain.passwords == {}


def test_login_has_no_option_that_takes_the_key() -> None:
    # Typer may vendor its CLI framework, so inspect the parameters by duck typing.
    auth_group: Any = getattr(typer.main.get_command(app), "commands", {})["auth"]
    for name in ("login", "status", "logout"):
        params = auth_group.commands[name].params
        options = [p for p in params if p.param_type_name == "option" and p.name != "help"]
        assert all(option.is_flag for option in options), name
        if name == "login":
            assert [option.name for option in options] == ["stdin"]
    credentials.register(ACME)
    result = invoke("login", "acme", "--key", SECRET)
    assert result.exit_code == 2


def test_login_validates_and_reports_the_account(keychain: MemoryKeyring) -> None:
    seen: list[SecretStr] = []

    def validator(value: SecretStr) -> str:
        seen.append(value)
        return "acct-synthetic"

    credentials.register(
        CredentialSpec(provider="acme", env_var="ACME_API_KEY", validator=validator)
    )
    result = invoke("login", "acme", "--stdin", stdin=SECRET)
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "provider": "acme",
        "stored": True,
        "source": "keychain",
        "account": "acct-synthetic",
    }
    assert [value.get_secret_value() for value in seen] == [SECRET]


def test_login_does_not_store_a_key_that_fails_validation(keychain: MemoryKeyring) -> None:
    def validator(value: SecretStr) -> str:
        raise UsageError("the acme API key was rejected", hint="create a new key")

    credentials.register(
        CredentialSpec(provider="acme", env_var="ACME_API_KEY", validator=validator)
    )
    result = invoke("login", "acme", "--stdin", stdin=SECRET)
    assert result.exit_code == 2
    assert "the acme API key was rejected" in result.stderr
    assert keychain.passwords == {}


def test_login_without_an_os_keychain_refuses_and_writes_no_file(tmp_path: Path) -> None:
    assert isinstance(keyring.core.get_keyring(), FailKeyring)
    calls: list[SecretStr] = []

    def validator(value: SecretStr) -> str:
        calls.append(value)
        return "acct-synthetic"

    credentials.register(
        CredentialSpec(provider="acme", env_var="ACME_API_KEY", validator=validator)
    )
    before = sorted(tmp_path.rglob("*"))
    result = invoke("login", "acme", "--stdin", stdin=SECRET)
    assert result.exit_code == 2
    assert "no OS keychain" in result.stderr
    assert "set ACME_API_KEY" in result.stderr
    assert sorted(tmp_path.rglob("*")) == before
    assert_no_leak(result.stdout + result.stderr)
    assert calls == []  # refused before the key was read or sent to the provider


def test_auth_works_in_read_only_mode(
    keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    credentials.register(ACME)
    assert invoke("login", "acme", "--stdin", stdin=SECRET).exit_code == 0
    assert invoke("status").exit_code == 0
    assert invoke("logout", "acme").exit_code == 0


# --- logout -----------------------------------------------------------------------------


def test_logout_deletes_the_stored_key(keychain: MemoryKeyring) -> None:
    credentials.register(ACME)
    keychain.set_password("pup-finetune", "acme", SECRET)
    result = invoke("logout", "acme")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"provider": "acme", "deleted": True}
    assert keychain.passwords == {}

    again = invoke("logout", "acme")
    assert again.exit_code == 0
    assert json.loads(again.stdout) == {"provider": "acme", "deleted": False}


def test_logout_that_cannot_delete_the_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = StickyKeyring()
    backend.set_password("pup-finetune", "acme", SECRET)
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend)
    credentials.register(ACME)
    result = invoke("logout", "acme")
    assert result.exit_code == 1
    assert "could not delete the acme key" in result.stderr
    assert result.stdout == ""
    assert stored(backend) == SECRET
    assert_no_leak(result.stdout + result.stderr)


def test_logout_without_an_os_keychain_deletes_nothing() -> None:
    credentials.register(ACME)
    result = invoke("logout", "acme")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"provider": "acme", "deleted": False}


# --- secret leaks -----------------------------------------------------------------------


@pytest.mark.parametrize("error", ["runtime", "usage"])
def test_a_first_login_that_fails_validation_never_leaks_the_key(
    error: str, keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A fresh process: nothing was registered for redaction by an earlier command.
    def validator(value: SecretStr) -> str:
        key = value.get_secret_value()
        if error == "runtime":
            raise RuntimeError(f"provider rejected {key}")
        raise UsageError(f"bad key {key}", hint=f"Authorization: Bearer {key}")

    credentials.register(
        CredentialSpec(provider="acme", env_var="ACME_API_KEY", validator=validator)
    )
    monkeypatch.setenv("PUP_FINETUNE_DEBUG", "1")
    result = invoke("login", "acme", "--stdin", stdin=SECRET)
    assert result.exit_code == (1 if error == "runtime" else 2)
    assert "***" in result.stderr
    assert_no_leak(result.stdout + result.stderr)
    assert keychain.passwords == {}
    assert all(SECRET.encode() not in content for content in file_contents(tmp_path))


def test_the_key_never_leaks_through_auth_commands_or_errors(
    keychain: MemoryKeyring, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def leaky_validator(value: SecretStr) -> str:
        raise RuntimeError(f"provider rejected {value.get_secret_value()}")

    def usage_validator(value: SecretStr) -> str:
        raise UsageError(
            f"bad key {value.get_secret_value()}",
            hint=f"Authorization: Bearer {value.get_secret_value()}",
        )

    credentials.register(ACME)
    credentials.register(
        CredentialSpec(provider="beta", env_var="BETA_API_KEY", validator=leaky_validator)
    )
    credentials.register(
        CredentialSpec(provider="gamma", env_var="GAMMA_API_KEY", validator=usage_validator)
    )
    results = [
        invoke("login", "acme", "--stdin", stdin=SECRET),
        invoke("status"),
        invoke("status", "acme"),
        invoke("login", "beta", "--stdin", stdin=SECRET),
        invoke("login", "gamma", "--stdin", stdin=SECRET),
    ]
    monkeypatch.setenv("ACME_API_KEY", SECRET)
    results.append(invoke("status"))
    monkeypatch.setenv("PUP_FINETUNE_DEBUG", "1")
    results.append(invoke("login", "beta", "--stdin", stdin=SECRET))
    results.append(invoke("logout", "acme"))

    assert [r.exit_code for r in results] == [0, 0, 0, 1, 2, 0, 1, 0]
    beta_error = results[3].stderr
    assert "RuntimeError: provider rejected ***" in beta_error
    assert "Error: bad key ***" in results[4].stderr
    assert "Bearer ***" in results[4].stderr
    assert "Traceback" in results[6].stderr
    for result in results:
        assert_no_leak(result.stdout + result.stderr)
    secret_bytes = SECRET.encode()
    assert all(secret_bytes not in content for content in file_contents(tmp_path))
