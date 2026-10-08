# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks API key: its registration, and the validator `auth login` runs."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from pup_finetune import credentials, redaction
from pup_finetune.cli import app
from pup_finetune.cli import doctor as doctor_cli
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.fireworks import CREDENTIALS, validate_key
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.setup import healthy_pup, no_provider_keys

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def test_the_spec() -> None:
    assert CREDENTIALS.provider == "fireworks"
    assert CREDENTIALS.env_var == "FIREWORKS_API_KEY"
    assert CREDENTIALS.keychain_service == "pup-finetune"
    assert CREDENTIALS.keychain_account == "fireworks"
    assert CREDENTIALS.description
    assert CREDENTIALS.validator is validate_key


def test_it_is_registered_with_what_it_is_needed_for() -> None:
    assert credentials.get_spec("fireworks") is CREDENTIALS
    assert CREDENTIALS.needed_for == (
        "training, evaluation and deployment on fireworks, and the default judges"
    )


def test_doctor_reports_a_missing_key_without_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    no_provider_keys(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor_cli, "Pup", healthy_pup().client)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout + result.stderr
    checks = {check["name"]: check for check in json.loads(result.stdout)["checks"]}
    fireworks = checks["fireworks:key"]
    assert fireworks["required"] is False
    assert fireworks["status"] == "optional"
    assert "pup finetune auth login fireworks" in fireworks["fix"]


def test_auth_status_reports_the_source_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    result = runner.invoke(app, ["auth", "status", "fireworks"])
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "providers": [
            {
                "provider": "fireworks",
                "env_var": "FIREWORKS_API_KEY",
                "configured": True,
                "source": "env",
                "needed_for": CREDENTIALS.needed_for,
            }
        ]
    }
    assert API_KEY not in result.stdout + result.stderr


def test_auth_status_lists_fireworks_with_no_key() -> None:
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0, result.stderr
    entries = {entry["provider"]: entry for entry in json.loads(result.stdout)["providers"]}
    assert entries["fireworks"]["configured"] is False


def test_the_cli_registers_the_key_without_heavy_imports() -> None:
    code = textwrap.dedent(
        """
        import json, sys
        from pup_finetune import credentials
        import pup_finetune.cli.auth

        heavy = {"transformers", "fireworks", "tinker", "httpx", "httpx2", "anthropic", "baseten"}
        print(json.dumps({
            "providers": [spec.provider for spec in credentials.all_specs()],
            "heavy": sorted({name.partition(".")[0] for name in sys.modules} & heavy),
        }))
        """
    )
    completed = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == {
        "providers": ["anthropic", "baseten", "fireworks", "openai"],
        "heavy": [],
    }


# --- the validator -----------------------------------------------------------------------


def test_a_valid_key_returns_the_first_account() -> None:
    fake = FakeFireworksApi()
    fake.accounts.append({"name": "accounts/second-team"})
    assert validate_key(SecretStr(API_KEY), transport=fake.transport) == "synthetic-team"
    assert [request.url.path for request in fake.requests] == ["/v1/accounts"]


def test_a_key_with_no_accounts_is_valid_without_a_label() -> None:
    fake = FakeFireworksApi()
    fake.accounts.clear()
    assert validate_key(SecretStr(API_KEY), transport=fake.transport) is None


def test_a_rejected_key_is_a_usage_error() -> None:
    fake = FakeFireworksApi()
    other = "fw-synthetic-rejected-key-5678"
    with pytest.raises(UsageError, match="Fireworks rejected the API key") as caught:
        validate_key(SecretStr(other), transport=fake.transport)
    assert caught.value.hint is not None
    assert "pup finetune auth login fireworks" in caught.value.hint
    assert other not in str(caught.value)


def test_an_unreachable_api_is_external() -> None:
    fake = FakeFireworksApi()
    fake.status["/v1/accounts"] = 502
    with pytest.raises(ExternalError, match="HTTP 502"):
        validate_key(SecretStr(API_KEY), transport=fake.transport)
