# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Baseten API key: its registration, the read-only ``~/.trussrc`` fallback, and the
validator `auth login` runs."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import credentials, redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import ProviderCapacityError, TransientError
from pup_finetune.providers.baseten import CREDENTIALS, read_trussrc_key, validate_key
from tests.fakes.baseten import Answer, baseten_transport

KEY = "synthetic-baseten-key-1234"
MODELS = "GET /v1/models"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def trussrc(home: Path, text: str) -> Path:
    path = home / ".trussrc"
    path.write_text(textwrap.dedent(text))
    return path


def test_the_spec_is_registered() -> None:
    assert credentials.get_spec("baseten") is CREDENTIALS
    assert CREDENTIALS.env_var == "BASETEN_API_KEY"
    assert CREDENTIALS.keychain_service == "pup-finetune"
    assert CREDENTIALS.keychain_account == "baseten"
    assert CREDENTIALS.description
    assert CREDENTIALS.config_reader is read_trussrc_key
    assert CREDENTIALS.validator is validate_key
    assert CREDENTIALS.needed_for == "training and evaluation on baseten"


def test_the_cli_registers_the_key_without_heavy_imports() -> None:
    code = textwrap.dedent(
        """
        import json, sys
        import pup_finetune.cli
        from pup_finetune import credentials

        print(json.dumps({
            "providers": [spec.provider for spec in credentials.all_specs()],
            "loaded": sorted(
                name for name in sys.modules
                if name.partition(".")[0] in ("baseten", "httpx", "transformers")
                or name.startswith("pup_finetune.providers.baseten.")
            ),
        }))
        """
    )
    completed = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == {
        "providers": ["anthropic", "baseten", "fireworks", "openai"],
        "loaded": [],
    }


# --- ~/.trussrc --------------------------------------------------------------------------


def test_the_key_of_the_baseten_remote_is_read(tmp_path: Path) -> None:
    trussrc(
        tmp_path,
        f"""
        [staging]
        remote_provider = baseten
        api_key = other-key
        remote_url = https://app.staging.example

        [baseten]
        remote_provider = baseten
        api_key = {KEY}
        remote_url = https://app.baseten.co/
        """,
    )
    assert read_trussrc_key() == KEY


def test_a_single_remote_is_the_default(tmp_path: Path) -> None:
    path = trussrc(tmp_path, f"[work]\nremote_provider = baseten\napi_key = {KEY}\n")
    assert read_trussrc_key(path) == KEY


def test_percent_signs_are_kept(tmp_path: Path) -> None:
    path = trussrc(tmp_path, "[baseten]\napi_key = abc%def\n")
    assert read_trussrc_key(path) == "abc%def"


@pytest.mark.parametrize(
    "text",
    [
        "[a]\napi_key = one\n[b]\napi_key = two\n",
        "[baseten]\napi_key = abc\nremote_url = https://app.example.com\n",
        "[baseten]\nauth_type = oauth\noauth_access_token = abc\n",
        "[baseten]\napi_key =\n",
        "api_key = no section header\n",
        "[baseten]\napi_key = a\n[baseten]\napi_key = b\n",
    ],
    ids=["no-default", "other-url", "oauth", "empty", "malformed", "duplicate"],
)
def test_no_usable_key_reads_as_none(tmp_path: Path, text: str) -> None:
    assert read_trussrc_key(trussrc(tmp_path, text)) is None


def test_a_missing_or_unreadable_file_reads_as_none(tmp_path: Path) -> None:
    assert read_trussrc_key() is None
    (tmp_path / ".trussrc").write_bytes(b"[baseten]\napi_key = \xff\n")
    assert read_trussrc_key() is None


def test_lookup_falls_back_to_the_trussrc_and_leaves_it_alone(tmp_path: Path) -> None:
    path = trussrc(tmp_path, f"[baseten]\napi_key = {KEY}\n")
    before = path.read_bytes(), path.stat().st_mtime_ns
    found = credentials.lookup("baseten", env={})
    assert found is not None
    assert (found.value.get_secret_value(), found.source) == (KEY, "provider-config")
    assert redaction.redact(f"key {KEY}") == "key ***"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    found = credentials.lookup("baseten", env={"BASETEN_API_KEY": "from-env"})
    assert found is not None
    assert found.source == "env"


# --- the validator -----------------------------------------------------------------------


def test_a_valid_key_lists_models_once_with_the_key() -> None:
    transport = baseten_transport({MODELS: (200, {"models": []})})
    assert validate_key(SecretStr(KEY), transport=transport) is None
    (request,) = transport.requests
    assert str(request.url) == "https://api.baseten.co/v1/models"
    assert request.headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_a_usage_error(status: int) -> None:
    transport = baseten_transport({MODELS: (status, {"detail": KEY})})
    with pytest.raises(UsageError, match="Baseten rejected the API key") as caught:
        validate_key(SecretStr(KEY), transport=transport)
    assert caught.value.hint == "create an API key in your Baseten account settings and retry"
    assert KEY not in str(caught.value)


@pytest.mark.parametrize(
    ("status", "error"),
    [(302, ExternalError), (404, ExternalError), (429, ProviderCapacityError)],
)
def test_any_other_refusal_is_external(status: int, error: type[ExternalError]) -> None:
    redirect = {"location": "https://example.com/"}
    answer = httpx.Response(status, headers=redirect if status == 302 else None)
    transport = baseten_transport({MODELS: answer})
    with pytest.raises(error, match=f"HTTP {status}"):
        validate_key(SecretStr(KEY), transport=transport)
    assert len(transport.requests) == 1  # no redirect followed


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


@pytest.mark.parametrize("answer", [(500, {}), _refuse], ids=["5xx", "unreachable"])
def test_a_failing_baseten_is_transient(answer: Answer) -> None:
    with pytest.raises(TransientError, match=r"Baseten|HTTP 500"):
        validate_key(SecretStr(KEY), transport=baseten_transport({MODELS: answer}))


@pytest.mark.parametrize("key", ["two words", "line\nbreak", "café", ""])
def test_a_malformed_key_is_refused_before_any_request(key: str) -> None:
    transport = baseten_transport({})
    with pytest.raises(UsageError, match=r"empty|malformed"):
        validate_key(SecretStr(key), transport=transport)
    assert transport.requests == []
