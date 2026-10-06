# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

from collections.abc import Callable, Iterator

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import credentials, redaction
from pup_finetune import doctor as doctor_registry
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.providers.baseten import registration

SECRET = "zq9Xsynthetic0000basetenKEYw7Pm"  # noqa: S105 - synthetic test value
KEY = SecretStr(SECRET)

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(credentials, "_REGISTRY", {})
    monkeypatch.setattr(doctor_registry, "_REGISTRY", {})
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def requests_seen(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[httpx.Request]]:
    """Answer Baseten calls with ``handler``; returns the list of requests it received."""

    def install(handler: Handler) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(registration, "_transport", httpx.MockTransport(record))
        return seen

    return install


def use_versions(monkeypatch: pytest.MonkeyPatch, **versions: str) -> None:
    installed = {name.replace("_", "-"): version for name, version in versions.items()}
    monkeypatch.setattr(registration, "_installed_version", installed.get)


def assert_no_secret(error: FinetuneError) -> None:
    for text in (str(error), error.message, error.hint or ""):
        assert SECRET not in text


# --- registration ---------------------------------------------------------------------


def test_register_adds_the_baseten_key_spec() -> None:
    registration.register()
    spec = credentials.get_spec("baseten")
    assert spec is registration.SPEC
    assert spec.env_var == "BASETEN_API_KEY"
    assert spec.description == "Baseten API key (Loops training, samplers, deployments)"
    assert spec.keychain_account == "baseten"
    assert spec.validator is registration.validate_key


def test_register_adds_optional_doctor_checks_and_is_idempotent() -> None:
    registration.register()
    registration.register()
    entries = doctor_registry._REGISTRY
    assert list(entries) == ["credentials:baseten", "baseten:loops-sdk", "rendering:transformers"]
    assert not any(entry.required for entry in entries.values())


def test_failed_checks_do_not_fail_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    use_versions(monkeypatch)
    registration.register()
    results = {result.name: result for result in doctor_registry.run_registered_checks()}
    assert not results["baseten:loops-sdk"].ok
    assert not results["rendering:transformers"].ok
    assert not any(result.required for result in results.values())


# --- validate_key ---------------------------------------------------------------------


def test_validate_key_sends_one_api_key_get(
    requests_seen: Callable[[Handler], list[httpx.Request]],
) -> None:
    seen = requests_seen(lambda _: httpx.Response(200, json={"models": []}))
    assert registration.validate_key(KEY) is None
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert str(request.url) == "https://api.baseten.co/v1/models"
    assert request.headers["Authorization"] == f"Api-Key {SECRET}"


@pytest.mark.parametrize("status", [401, 403])
def test_validate_key_rejected_key_is_a_usage_error(
    requests_seen: Callable[[Handler], list[httpx.Request]], status: int
) -> None:
    requests_seen(lambda _: httpx.Response(status, text=f"bad key {SECRET}"))
    with pytest.raises(UsageError) as caught:
        registration.validate_key(KEY)
    assert caught.value.message == "Baseten rejected the API key"
    assert_no_secret(caught.value)


@pytest.mark.parametrize("status", [404, 500, 503])
def test_validate_key_other_status_is_an_external_error_without_the_body(
    requests_seen: Callable[[Handler], list[httpx.Request]], status: int
) -> None:
    requests_seen(lambda _: httpx.Response(status, text="synthetic internal detail"))
    with pytest.raises(ExternalError) as caught:
        registration.validate_key(KEY)
    assert caught.value.source == "baseten"
    assert f"HTTP {status}" in caught.value.message
    assert "internal detail" not in str(caught.value)
    assert_no_secret(caught.value)


def test_validate_key_does_not_follow_redirects(
    requests_seen: Callable[[Handler], list[httpx.Request]],
) -> None:
    seen = requests_seen(
        lambda _: httpx.Response(302, headers={"Location": "https://elsewhere.example/models"})
    )
    with pytest.raises(ExternalError, match="HTTP 302"):
        registration.validate_key(KEY)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("synthetic refusal"), httpx.ReadTimeout("synthetic timeout")]
)
def test_validate_key_network_failure_is_an_external_error(
    requests_seen: Callable[[Handler], list[httpx.Request]], error: httpx.HTTPError
) -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise error

    requests_seen(fail)
    with pytest.raises(ExternalError, match="could not reach Baseten") as caught:
        registration.validate_key(KEY)
    assert caught.value.source == "baseten"
    assert type(error).__name__ in caught.value.message
    assert_no_secret(caught.value)


# --- doctor checks --------------------------------------------------------------------


def test_loops_sdk_check_missing_gives_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    use_versions(monkeypatch)
    outcome = registration.check_loops_sdk()
    assert not outcome.ok
    assert "not installed" in outcome.detail
    assert "pip install baseten-loops==0.26.0" in outcome.detail


def test_loops_sdk_check_tested_version(monkeypatch: pytest.MonkeyPatch) -> None:
    use_versions(monkeypatch, baseten_loops="0.26.0")
    outcome = registration.check_loops_sdk()
    assert outcome.ok
    assert outcome.detail == "baseten-loops 0.26.0 is installed"


def test_loops_sdk_check_warns_on_an_untested_version(monkeypatch: pytest.MonkeyPatch) -> None:
    use_versions(monkeypatch, baseten_loops="0.27.1")
    outcome = registration.check_loops_sdk()
    assert outcome.ok
    assert "baseten-loops 0.27.1 is installed; warning" in outcome.detail
    assert "tested with 0.26.0" in outcome.detail


def test_transformers_check(monkeypatch: pytest.MonkeyPatch) -> None:
    use_versions(monkeypatch)
    missing = registration.check_transformers()
    assert not missing.ok
    assert "pup-finetune[hf]" in missing.detail
    use_versions(monkeypatch, transformers="5.10.4")
    assert registration.check_transformers().detail == "transformers 5.10.4 is installed"


def test_installed_version_reads_package_metadata() -> None:
    assert registration._installed_version("httpx")
    assert registration._installed_version("synthetic-package-that-is-not-installed") is None
