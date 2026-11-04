# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune doctor`: checks with a purpose, readiness per step, and a checklist."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keyring.core
import pytest
from typer.testing import CliRunner, Result

from pup_finetune import __version__, credentials, providers, readiness, redaction
from pup_finetune.cli import app
from pup_finetune.cli import doctor as doctor_cli
from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import ExternalError, UsageError, install_hint
from pup_finetune.providers import provider_info
from pup_finetune.providers.base import (
    PROVIDER_LIMIT,
    Provider,
    under_profile_context,
)
from pup_finetune.providers.baseten.provider import PROFILES as BASETEN_PROFILES
from pup_finetune.providers.fireworks.models import PROFILES as FIREWORKS_PROFILES
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.readiness import preflight_checks
from pup_finetune.runtime import Runtime
from tests.fakes.baseten import (
    OVER_WORKSPACE_LIMIT,
    SHORT_PROFILE,
    WITHIN_WORKSPACE_LIMIT,
    WORKSPACE_LIMIT,
    FakeLoops,
    capabilities,
    serverless_capabilities,
    short_capabilities,
)
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.keyring import CRASHED_HOLDING, CrashingKeyring, LockedKeyring, MemoryKeyring
from tests.fakes.pup import FAKE_BINARY, FakePup
from tests.fakes.setup import ALL_PACKAGES, healthy_pup, installed_packages, no_provider_keys
from tests.providers.baseten.test_baseten_provider import CAPABILITIES as BASETEN_CAPABILITIES
from tests.providers.baseten.test_baseten_provider import KEY as BASETEN_KEY
from tests.providers.baseten.test_baseten_provider import provider_on
from tests.providers.baseten.test_baseten_provider import provider_on as baseten_provider_on

runner = CliRunner()

CHECKOUT_COMMIT = doctor_cli.checkout_commit
"""The real lookup; every other test replaces it with a fixed commit."""

CHECKS = [
    "pup",
    "datadog_auth",
    "baseten:key",
    "baseten:sdk",
    "baseten:renderer",
    "fireworks:key",
    "fireworks:sdk",
    "fireworks:renderer",
    "anthropic:key",
    "openai:key",
    "anthropic:sdk",
]
"""Every check, in order: pup and the login, each provider, then the judges."""

UsePup = Callable[[Pup], None]

SECRET = "zq9Xsynthetic0000keyVALUEw7Pm"  # noqa: S105 - synthetic test value


@pytest.fixture
def use_pup(monkeypatch: pytest.MonkeyPatch) -> UsePup:
    """Make the doctor command use a given Pup instead of the real one."""

    def install(pup: Pup) -> None:
        monkeypatch.setattr(doctor_cli, "Pup", lambda: pup)

    return install


@pytest.fixture(autouse=True)
def _known_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No keys, every package installed, a fixed checkout."""
    no_provider_keys(monkeypatch, tmp_path / "home")
    installed_packages(monkeypatch, ALL_PACKAGES)
    monkeypatch.setattr(doctor_cli, "checkout_commit", lambda: "cd9cc2e")
    redaction.clear_secrets()


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["doctor", *args])


def report_of(result: Result) -> dict[str, Any]:
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    return data


def checks_of(result: Result) -> dict[str, dict[str, Any]]:
    return {check["name"]: check for check in report_of(result)["checks"]}


# --- the report ----------------------------------------------------------------------------


def test_a_healthy_pup_is_ready_without_any_provider_key(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    report = report_of(result)
    assert set(report) == {
        "status",
        "provider",
        "mode",
        "versions",
        "checks",
        "ready",
        "next_command",
        "next_steps",
        "ok",
    }
    assert report["status"] == "ready"
    assert report["ok"] is True
    assert report["provider"] is None
    assert report["mode"] == {"read_only": False, "dev": False}
    assert report["versions"] == {
        "pup_finetune": __version__,
        "checkout": "cd9cc2e",
        "pup": "1.10.7",
        "python": platform.python_version(),
    }
    assert [check["name"] for check in report["checks"]] == CHECKS
    pup_check, auth_check = report["checks"][:2]
    assert pup_check == {
        "name": "pup",
        "status": "pass",
        "detail": f"pup 1.10.7 at {FAKE_BINARY}",
        "needed_for": "reading traces and pushing datasets",
        "fix": None,
        "label": f"pup 1.10.7 at {FAKE_BINARY}",
        "ok": True,
        "required": True,
    }
    assert auth_check["status"] == "pass"
    assert auth_check["label"] == "Datadog: signed in to datadoghq.com"
    # What the old report's datadog_auth section had: when the token expires.
    assert auth_check["detail"] == (
        "pup is signed in to datadoghq.com (its token expires at 2026-01-01T00:00:00Z; pup "
        "refreshes it)"
    )
    # A missing provider key is optional, never a failure, without --provider.
    key = checks_of(result)["fireworks:key"]
    assert key == {
        "name": "fireworks:key",
        "status": "optional",
        "detail": "no fireworks API key; run `pup finetune auth login fireworks` in your own "
        "terminal, or set FIREWORKS_API_KEY",
        "needed_for": "training, evaluation and deployment on fireworks, and the default judges",
        "fix": "run `pup finetune auth login fireworks` in your own terminal, or set "
        "FIREWORKS_API_KEY",
        "label": "Fireworks key not set",
        "ok": False,
        "required": False,
    }
    assert all(not check["required"] for check in report["checks"][2:])
    assert report["next_command"] is None
    assert report["next_steps"] == [
        {
            "command": "pup finetune dataset pull <DIR> --ml-app <ML_APP> --from 7d",
            "why": "pull your Agent Observability traces into a dataset directory, once "
            "`pup llm-obs spans search` showed the spans and filter to read",
        },
        {
            "command": "pup finetune auth login baseten",
            "why": f"{doctor_cli.LOGIN_WHY}, for training, evaluation and deployment on baseten",
        },
        {
            "command": "pup finetune auth login fireworks",
            "why": f"{doctor_cli.LOGIN_WHY}, for training, evaluation and deployment on "
            "fireworks, and the default judges",
        },
    ]
    assert "Bearer" not in result.stdout
    assert "scopes" not in result.stdout


def test_ready_says_which_steps_can_run(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    ready = report_of(invoke())["ready"]
    assert {name: step["ready"] for name, step in ready.items()} == {
        "dataset pull": True,
        "dataset show": True,
        "dataset push": True,
        "dataset triage": False,
        "train on fireworks": False,
        "evaluate on fireworks": False,
        "deploy on fireworks": False,
        "train on baseten": False,
        "evaluate on baseten": False,
        "deploy on baseten": False,
    }
    assert ready["dataset triage"]["blocked_by"] == ["fireworks:key"]
    assert ready["evaluate on baseten"]["blocked_by"] == ["baseten:key", "fireworks:key"]
    assert ready["deploy on baseten"]["blocked_by"] == ["baseten:key"]
    assert ready["deploy on fireworks"]["blocked_by"] == ["fireworks:key"]


def test_an_anthropic_key_makes_it_the_default_judges_and_readies_triage(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-synthetic-0000-test-key")
    use_pup(healthy_pup().client())
    ready = report_of(invoke())["ready"]
    assert ready["dataset triage"] == {"ready": True, "blocked_by": []}
    assert ready["evaluate on baseten"]["blocked_by"] == ["baseten:key"]


def test_an_anthropic_key_without_its_sdk_leaves_the_default_judges_on_fireworks(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-synthetic-0000-test-key")
    use_pup(healthy_pup().client())
    ready = report_of(invoke())["ready"]
    assert ready["dataset triage"]["blocked_by"] == ["fireworks:key"]
    assert ready["evaluate on baseten"]["blocked_by"] == ["baseten:key", "fireworks:key"]


def test_with_the_keys_set_every_step_is_ready(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    monkeypatch.setenv("BASETEN_API_KEY", "bt-synthetic-0000-test-key")
    use_pup(healthy_pup().client())
    result = invoke()
    report = report_of(result)
    assert all(step["ready"] for step in report["ready"].values())
    key = checks_of(result)["fireworks:key"]
    assert (key["status"], key["label"]) == ("pass", "Fireworks key (FIREWORKS_API_KEY)")
    assert key["detail"] == "fireworks API key found (source: env)"
    assert [step["command"] for step in report["next_steps"]] == [
        "pup finetune dataset pull <DIR> --ml-app <ML_APP> --from 7d"
    ]
    assert API_KEY not in result.stdout + result.stderr


def test_a_missing_package_is_reported_with_the_install_hint(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"tinker"})
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    sdk = checks_of(result)["fireworks:sdk"]
    assert sdk["status"] == "optional"
    assert sdk["label"] == "Fireworks Training SDK missing"
    assert sdk["fix"] == install_hint("fireworks")
    ready = report_of(result)["ready"]
    assert ready["train on fireworks"]["blocked_by"] == ["fireworks:sdk"]
    assert ready["deploy on fireworks"]["blocked_by"] == ["fireworks:sdk"]
    # Triage needs only the judges' key.
    assert ready["dataset triage"]["ready"] is True


def test_read_only_mode_blocks_push_and_paid_runs(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    report = report_of(invoke())
    assert report["mode"] == {"read_only": True, "dev": False}
    ready = report["ready"]
    assert ready["dataset push"]["blocked_by"] == ["read_only"]
    assert ready["train on fireworks"]["blocked_by"] == ["read_only"]
    assert ready["deploy on fireworks"]["blocked_by"] == ["read_only"]
    assert ready["dataset pull"]["ready"] is True
    assert ready["dataset triage"]["ready"] is True


def test_the_ready_line_says_which_steps_read_only_mode_blocks(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    lines = invoke().stderr.splitlines()
    ready = next(line for line in lines if line.startswith("Ready: "))
    assert ready == (
        "Ready: dataset pull, show, triage.  Not ready: dataset push (read-only), "
        "train on baseten (read-only), evaluate on baseten (read-only), deploy on "
        "baseten (read-only), train on fireworks (read-only), evaluate on fireworks (read-only), "
        "deploy on fireworks (read-only)."
    )


def test_dev_mode_adds_the_offline_fake_provider(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    use_pup(healthy_pup().client())
    report = report_of(invoke())
    assert report["mode"] == {"read_only": True, "dev": True}
    # The fake has no key or package to check, and never leaves the machine.
    assert [check["name"] for check in report["checks"]] == CHECKS
    assert report["ready"]["train on fake"] == {"ready": True, "blocked_by": []}
    assert report["ready"]["evaluate on fake"] == {"ready": True, "blocked_by": []}
    assert report["ready"]["deploy on fake"] == {"ready": True, "blocked_by": []}


# --- pup and the login ---------------------------------------------------------------------


def test_old_pup_fails_but_still_emits_the_report(use_pup: UsePup) -> None:
    use_pup(healthy_pup("pup 1.9.12\n").client())
    result = invoke()
    assert result.exit_code == 1
    report = report_of(result)
    assert (report["status"], report["ok"]) == ("not_ready", False)
    assert report["versions"]["pup"] == "1.9.12"
    pup_check = report["checks"][0]
    assert pup_check["status"] == "fail"
    assert "older than the minimum 1.10.7" in pup_check["detail"]
    assert pup_check["fix"] == "upgrade with `brew upgrade datadog-labs/pack/pup`"
    assert report["checks"][1]["status"] == "pass"
    assert report["ready"]["dataset pull"]["blocked_by"] == ["pup"]


def test_missing_pup_fails_and_skips_auth(use_pup: UsePup, tmp_path: Path) -> None:
    fake = FakePup()
    use_pup(Pup(runner=fake, environ={"PATH": str(tmp_path)}))
    result = invoke()
    assert result.exit_code == 1
    report = report_of(result)
    assert report["versions"]["pup"] is None
    pup_check, auth_check = report["checks"][:2]
    assert pup_check["status"] == "fail"
    assert "brew install datadog-labs/pack/pup" in pup_check["fix"]
    assert auth_check["status"] == "skipped"
    assert auth_check["detail"].startswith("skipped")
    assert report["ready"]["dataset show"]["ready"] is True
    assert report["ready"]["dataset pull"]["blocked_by"] == ["pup", "datadog_auth"]
    assert fake.calls == []


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        ({"stdout": "pup nightly\n"}, "could not parse a version"),
        ({"returncode": 1, "stderr": "Error: unknown flag\n"}, "failed (exit 1)"),
        ({"times_out": True}, "timed out"),
    ],
)
def test_failing_version_probe_fails_but_still_checks_auth(
    use_pup: UsePup, response: dict[str, Any], detail: str
) -> None:
    fake = healthy_pup().on("--version", raw=True, **response)
    use_pup(fake.client())
    result = invoke()
    assert result.exit_code == 1
    report = report_of(result)
    assert report["versions"]["pup"] is None
    pup_check, auth_check = report["checks"][:2]
    assert detail in pup_check["detail"]
    assert not pup_check["detail"].startswith("pup: ")
    assert auth_check["status"] == "pass"


def test_signed_out_fails(use_pup: UsePup) -> None:
    fake = healthy_pup().on(
        "auth", "status", json={"authenticated": False, "site": "datadoghq.com"}
    )
    use_pup(fake.client())
    result = invoke()
    assert result.exit_code == 1
    auth_check = checks_of(result)["datadog_auth"]
    assert auth_check["status"] == "fail"
    assert auth_check["fix"] == "run `pup auth login`"
    assert "pup auth login" in auth_check["detail"]


def test_auth_status_error_fails(use_pup: UsePup) -> None:
    fake = healthy_pup().on("auth", "status", returncode=1, stderr="Error: no stored token\n")
    use_pup(fake.client())
    result = invoke()
    assert result.exit_code == 1
    detail = checks_of(result)["datadog_auth"]["detail"]
    assert "no stored token" in detail
    assert "pup auth login" in detail


def test_auth_status_error_keeps_the_clients_hint(use_pup: UsePup) -> None:
    fake = healthy_pup().on("auth", "status", returncode=1, stderr="Error: 401 Unauthorized\n")
    use_pup(fake.client())
    detail = checks_of(invoke())["datadog_auth"]["detail"]
    assert detail.endswith("; check `pup auth status`, and run `pup auth login` to sign in again")


# --- --provider and --check ----------------------------------------------------------------


def test_a_named_provider_s_missing_key_fails_and_the_user_signs_in(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 1
    report = report_of(result)
    assert (report["status"], report["provider"]) == ("not_ready", "fireworks")
    checks = checks_of(result)
    assert checks["fireworks:key"]["status"] == "fail"
    assert checks["fireworks:key"]["required"] is True
    assert checks["fireworks:sdk"]["required"] is True
    # The other provider stays optional.
    assert checks["baseten:key"]["status"] == "optional"
    # Only the user enters a key: signing in is a step for them, never the next command.
    assert report["next_command"] is None
    logins = [s for s in report["next_steps"] if s["command"].startswith("pup finetune auth")]
    assert logins == [
        {
            "command": "pup finetune auth login fireworks",
            "why": f"{doctor_cli.LOGIN_WHY}, for training, evaluation and deployment on "
            "fireworks, and the default judges",
        }
    ]
    assert "own terminal" in doctor_cli.LOGIN_WHY


def test_a_named_provider_s_missing_sdk_fails(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"baseten.loops"})
    monkeypatch.setenv("BASETEN_API_KEY", "bt-synthetic-0000-test-key")
    use_pup(healthy_pup().client())
    result = invoke("--provider", "baseten")
    assert result.exit_code == 1
    sdk = checks_of(result)["baseten:sdk"]
    assert (sdk["status"], sdk["fix"]) == ("fail", install_hint("baseten"))
    assert report_of(result)["next_command"] is None


def test_a_ready_named_provider_suggests_its_live_check(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 0, result.stderr
    assert report_of(result)["next_command"] == "pup finetune doctor --provider fireworks --check"


def test_a_failed_pup_check_comes_before_the_live_check(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup("pup 1.9.12\n").client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 1
    assert report_of(result)["next_command"] is None
    assert (
        result.stderr.splitlines()[-1] == "Next: upgrade with `brew upgrade datadog-labs/pack/pup`"
    )


def test_a_named_provider_s_steps_are_the_only_ones_not_ready(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"fireworks"})
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    ready = next(line for line in result.stderr.splitlines() if line.startswith("Ready: "))
    assert ready == (
        "Ready: dataset pull, show, push, triage.  Not ready: train on fireworks, "
        "evaluate on fireworks, deploy on fireworks."
    )
    # The JSON keeps every step.
    assert report_of(result)["ready"]["train on baseten"]["ready"] is False


def test_doctor_without_check_builds_no_provider(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_provider(name: str, runtime: Runtime) -> Provider:
        raise AssertionError(f"doctor built the {name} provider")

    monkeypatch.setattr(readiness, "get_provider", no_provider)
    monkeypatch.setattr(providers, "get_provider", no_provider)
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    for args in ((), ("--provider", "fireworks"), ("--provider", "baseten")):
        result = invoke(*args)
        assert "AssertionError" not in result.stdout + result.stderr, args
        assert not any(":model" in check["name"] for check in report_of(result)["checks"])


@pytest.fixture
def fireworks_api(monkeypatch: pytest.MonkeyPatch) -> FakeFireworksApi:
    """Serve the Fireworks API from a fake for `doctor --check`, with the key in the env."""
    api = FakeFireworksApi()
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)

    def provider(name: str, runtime: Runtime) -> Provider:
        assert name == "fireworks"
        return FireworksProvider(
            runtime, env={"FIREWORKS_API_KEY": API_KEY}, transport=api.transport
        )

    monkeypatch.setattr(readiness, "get_provider", provider)
    return api


def test_check_preflights_the_named_provider_s_models(
    use_pup: UsePup, fireworks_api: FakeFireworksApi
) -> None:
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 0, result.stderr
    report = report_of(result)
    names = [check["name"] for check in report["checks"]]
    aliases = [profile.alias for profile in FIREWORKS_PROFILES]
    start = names.index("fireworks:key")
    assert names[start : start + 3 + len(aliases)] == [
        "fireworks:key",
        "fireworks:sdk",
        "fireworks:renderer",
        *(f"fireworks:model:{alias}" for alias in aliases),
    ]
    model = checks_of(result)["fireworks:model:qwen3p8-27b"]
    assert model["status"] == "pass"
    assert model["required"] is True
    assert model["label"] == "qwen3p8-27b can be fine-tuned on Fireworks"
    assert report["next_command"] is None
    assert [request.url.path for request in fireworks_api.requests] == [
        f"/v1/accounts/fireworks/models/{alias}" for alias in aliases
    ]
    assert API_KEY not in result.stdout + result.stderr


@pytest.fixture
def short_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve `doctor --check` a Baseten workspace that trains Qwen3.5 9B on fewer tokens
    than its profile's context, with the key in the env."""
    caps = short_capabilities(BASETEN_CAPABILITIES)
    monkeypatch.setenv("BASETEN_API_KEY", BASETEN_KEY.get_secret_value())

    def provider(name: str, runtime: Runtime) -> Provider:
        assert name == "baseten"
        return baseten_provider_on(FakeLoops(), routes={"GET /v1/loops/capabilities": (200, caps)})

    monkeypatch.setattr(readiness, "get_provider", provider)


@pytest.mark.usefixtures("short_workspace")
def test_check_passes_a_model_fitted_to_the_workspace_limit(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke("--provider", "baseten", "--check")
    assert result.exit_code == 0, result.stderr
    model = checks_of(result)["baseten:model:qwen3p5-9b"]
    assert model["status"] == "pass"
    assert model["label"] == (
        f"qwen3p5-9b can be fine-tuned on Baseten at {WORKSPACE_LIMIT} tokens ({PROVIDER_LIMIT})"
    )
    assert under_profile_context(SHORT_PROFILE) in model["detail"]
    assert checks_of(result)["baseten:model:qwen3p8-27b"]["label"] == (
        "qwen3p8-27b can be fine-tuned on Baseten"
    )


def test_check_says_which_models_the_workspace_runs_serverless(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    caps = serverless_capabilities(BASETEN_CAPABILITIES)
    monkeypatch.setenv("BASETEN_API_KEY", BASETEN_KEY.get_secret_value())

    def provider(name: str, runtime: Runtime) -> Provider:
        return baseten_provider_on(FakeLoops(), routes={"GET /v1/loops/capabilities": (200, caps)})

    monkeypatch.setattr(readiness, "get_provider", provider)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "baseten", "--check")
    assert result.exit_code == 0, result.stderr
    serverless = checks_of(result)["baseten:model:qwen3p5-9b"]
    assert serverless["label"] == "qwen3p5-9b can be fine-tuned on Baseten (serverless)"
    assert serverless["detail"].endswith(", expected on serverless capacity (else dedicated)")
    dedicated = checks_of(result)["baseten:model:glm-5p3-flash"]
    assert dedicated["label"] == "glm-5p3-flash can be fine-tuned on Baseten"
    assert "serverless" not in dedicated["detail"]


@pytest.mark.usefixtures("short_workspace")
def test_check_at_a_context_over_the_workspace_limit_fails_doctor(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke("--provider", "baseten", "--check", "--max-seq-len", str(OVER_WORKSPACE_LIMIT))
    assert result.exit_code == 1
    model = checks_of(result)["baseten:model:qwen3p5-9b"]
    assert model["status"] == "fail"
    assert f"shorter than the {OVER_WORKSPACE_LIMIT} to train at" in model["detail"]
    assert model["fix"] == f"pass --max-seq-len N with N at most {WORKSPACE_LIMIT}"


def test_max_seq_len_needs_check_in_doctor(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke("--provider", "baseten", "--max-seq-len", str(WITHIN_WORKSPACE_LIMIT))
    assert result.exit_code == 2
    assert "--max-seq-len needs --check" in result.stderr


def test_a_failed_preflight_fails_doctor(use_pup: UsePup, fireworks_api: FakeFireworksApi) -> None:
    fireworks_api.models["qwen3p8-27b"]["state"] = "DEPLOYING"
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    model = checks_of(result)["fireworks:model:qwen3p8-27b"]
    assert model["status"] == "fail"
    assert "its state is DEPLOYING, not READY" in model["detail"]
    assert "models list --provider fireworks --check" in model["fix"]
    # A model that cannot be fine-tuned now blocks training and evaluating on the provider.
    ready = report_of(result)["ready"]
    assert ready["train on fireworks"]["blocked_by"] == ["fireworks:model:qwen3p8-27b"]
    assert ready["evaluate on fireworks"]["blocked_by"] == ["fireworks:model:qwen3p8-27b"]
    assert ready["deploy on fireworks"]["ready"] is True


def test_a_provider_that_cannot_be_loaded_fails_the_check(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(name: str, runtime: Runtime) -> Provider:
        raise UsageError(
            "provider 'fireworks' is registered but its module is missing", hint="reinstall"
        )

    monkeypatch.setattr(readiness, "get_provider", broken)
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    loaded = checks_of(result)["fireworks:models"]
    assert (loaded["status"], loaded["label"]) == ("fail", "Fireworks could not be loaded")
    assert loaded["detail"] == (
        "provider 'fireworks' is registered but its module is missing; reinstall"
    )
    assert loaded["fix"] == "reinstall"
    ready = report_of(result)["ready"]
    assert ready["train on fireworks"]["blocked_by"] == ["fireworks:models"]
    assert ready["evaluate on fireworks"]["blocked_by"] == ["fireworks:models"]


@pytest.mark.parametrize(
    ("error", "env", "detail"),
    [
        (
            ExternalError(f"provider said {SECRET}", source="acme", hint=f"key {SECRET}"),
            {},
            "provider said ***; key ***",
        ),
        (
            RuntimeError(f"crashed with {SECRET}\nsecond line"),
            {},
            "check failed unexpectedly (RuntimeError: crashed with ***)"
            " (set PUP_FINETUNE_DEBUG=1 for a traceback)",
        ),
        (
            RuntimeError(f"crashed with {SECRET}\nsecond line"),
            {"PUP_FINETUNE_DEBUG": "1"},
            "check failed unexpectedly (RuntimeError: crashed with ***)",
        ),
    ],
)
def test_a_check_that_raises_is_reported_redacted(
    error: Exception,
    env: dict[str, str],
    detail: str,
    use_pup: UsePup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(name: str, runtime: Runtime) -> Provider:
        raise error

    redaction.register_secret(SECRET)
    monkeypatch.setattr(readiness, "get_provider", broken)
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = runner.invoke(app, ["doctor", "--provider", "fireworks", "--check"], env=env)
    assert result.exit_code == 1
    assert checks_of(result)["fireworks:models"]["detail"] == detail
    assert SECRET not in result.stdout + result.stderr
    assert ("Traceback" in result.stderr) == bool(env)


def test_check_skips_the_preflight_without_a_key(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def provider(name: str, runtime: Runtime) -> Provider:
        calls.append(name)
        return FireworksProvider(runtime, transport=FakeFireworksApi().transport)

    monkeypatch.setattr(readiness, "get_provider", provider)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    model = checks_of(result)["fireworks:model:qwen3p8-27b"]
    assert model["status"] == "skipped"
    assert model["detail"] == "skipped: fireworks:key did not pass"


def test_check_needs_a_provider() -> None:
    result = invoke("--check")
    assert result.exit_code == 2
    assert "--check needs --provider" in result.stderr


def test_an_unknown_provider_lists_the_known_ones() -> None:
    result = invoke("--provider", "nope")
    assert result.exit_code == 2
    assert "known providers: baseten, fireworks" in result.stderr


def test_the_fake_provider_needs_dev_mode() -> None:
    result = invoke("--provider", "fake")
    assert result.exit_code == 2
    assert "PUP_FINETUNE_DEV=1" in result.stderr


def test_the_provider_help_lists_the_registered_providers() -> None:
    assert doctor_cli.PROVIDERS == "baseten or fireworks"


# --- the checklist on stderr ---------------------------------------------------------------


def test_the_checklist_follows_the_report_on_stderr(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES - {"fireworks"})
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    version = __version__.split("+")[0]
    assert result.stderr.splitlines() == [
        f"pup-finetune {version} (checkout cd9cc2e)   read-only: off   dev: off",
        f"ok    pup 1.10.7 at {FAKE_BINARY}                  reading traces and pushing datasets",
        "ok    Datadog: signed in to datadoghq.com          reading traces and pushing datasets",
        "--    Baseten key not set                          only for training, evaluation and "
        "deployment on baseten",
        "ok    Baseten Loops SDK installed                  training and evaluation on baseten",
        "ok    Baseten renderer (transformers) installed    rendering rows for baseten",
        "ok    Fireworks key (FIREWORKS_API_KEY)            training, evaluation and deployment "
        "on fireworks, and the default judges",
        "--    Fireworks Training SDK missing               only for training, evaluation and "
        "deployment on fireworks",
        "ok    Fireworks renderer (transformers) installed  rendering rows for fireworks",
        "--    Anthropic/OpenAI keys not set                only for anthropic/... or openai/... "
        "judges",
        "ok    Anthropic SDK installed                      anthropic/... judges",
        "Ready: dataset pull, show, push, triage.  Not ready: train on baseten, "
        "evaluate on baseten, deploy on baseten, train on fireworks, evaluate on fireworks, "
        "deploy on fireworks.",
        "Next: pup finetune dataset pull <DIR> --ml-app <ML_APP> --from 7d",
    ]


def test_a_failing_doctor_still_prints_its_checklist(use_pup: UsePup, tmp_path: Path) -> None:
    use_pup(Pup(runner=FakePup(), environ={"PATH": str(tmp_path)}))
    result = invoke()
    assert result.exit_code == 1
    lines = result.stderr.splitlines()
    assert lines[1].startswith("FAIL  pup not found")
    assert lines[2].startswith("      fix: install pup with `brew install datadog-labs/pack/pup`")
    assert lines[3].startswith("skip  Datadog login not checked (no pup)")
    assert lines[-2].startswith("Ready: dataset show.  Not ready: dataset pull, push")
    assert lines[-1].startswith("Next: install pup with `brew install datadog-labs/pack/pup`")


def test_a_named_provider_s_failure_shows_its_fix_and_the_next_command(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 1
    lines = result.stderr.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("FAIL  Fireworks key"))
    assert lines[index + 1] == (
        "      fix: run `pup finetune auth login fireworks` in your own terminal, or set "
        "FIREWORKS_API_KEY"
    )
    assert lines[-1] == (
        "Next: run `pup finetune auth login fireworks` in your own terminal, or set "
        "FIREWORKS_API_KEY"
    )


def test_a_checkout_git_cannot_describe_says_so(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_cli, "checkout_commit", lambda: doctor_cli.CHECKOUT_UNKNOWN)
    use_pup(healthy_pup().client())
    result = invoke()
    assert report_of(result)["versions"]["checkout"] == "unknown"
    version = __version__.split("+")[0]
    assert result.stderr.splitlines()[0] == (
        f"pup-finetune {version} (checkout unknown)   read-only: off   dev: off"
    )


def test_an_installed_package_shows_its_plain_version(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_cli, "checkout_commit", lambda: None)
    use_pup(healthy_pup().client())
    result = invoke()
    assert report_of(result)["versions"]["checkout"] is None
    assert result.stderr.splitlines()[0] == (
        f"pup-finetune {__version__}   read-only: off   dev: off"
    )


# --- keys that cannot be read, and packages that cannot be loaded ----------------------------


def test_a_locked_keychain_says_to_unlock_it_not_to_sign_in(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(keyring.core, "_keyring_backend", LockedKeyring())
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 1
    key = checks_of(result)["fireworks:key"]
    assert (key["status"], key["label"]) == ("fail", "Fireworks key unreadable")
    assert "could not read the fireworks key from the OS keychain (KeyringLocked" in key["detail"]
    assert key["fix"] == "unlock the keychain and retry, or set FIREWORKS_API_KEY"
    report = report_of(result)
    assert report["next_command"] is None
    assert not any("auth login" in s["command"] for s in report["next_steps"])
    assert result.stderr.splitlines()[-1] == f"Next: {key['fix']}"


def test_a_crashing_keychain_backend_does_not_crash_doctor(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(keyring.core, "_keyring_backend", CrashingKeyring())
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    key = checks_of(result)["fireworks:key"]
    fix = "check that the OS keychain works and retry, or set FIREWORKS_API_KEY"
    assert (key["status"], key["label"], key["fix"]) == (
        "optional",
        "Fireworks key unreadable",
        fix,
    )
    assert key["detail"] == "could not read the fireworks key from the OS keychain (RuntimeError)"
    assert CRASHED_HOLDING not in result.stdout + result.stderr
    assert "pup finetune auth login fireworks" not in [
        s["command"] for s in report_of(result)["next_steps"]
    ]


def test_a_key_too_short_to_be_one_is_malformed_and_leaves_the_report_whole(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    assert "next_command" in report_of(result)
    key = checks_of(result)["anthropic:key"]
    assert (key["status"], key["label"]) == ("optional", "Anthropic key malformed")
    assert "anthropic API key (source: env) is malformed" in key["detail"]
    assert key["fix"] == "fix or unset ANTHROPIC_API_KEY, which is read before the OS keychain"


def test_a_malformed_key_in_the_keychain_is_fixed_by_signing_in_again(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    keychain = MemoryKeyring()
    spec = credentials.get_spec("anthropic")
    keychain.set_password(spec.keychain_service, spec.keychain_account, "x")
    monkeypatch.setattr(keyring.core, "_keyring_backend", keychain)
    use_pup(healthy_pup().client())
    key = checks_of(invoke())["anthropic:key"]
    assert (key["status"], key["label"]) == ("optional", "Anthropic key malformed")
    assert key["fix"] == (
        "run `pup finetune auth login anthropic` in your own terminal, or set ANTHROPIC_API_KEY"
    )


def test_a_short_key_that_is_a_word_never_rewrites_the_next_command(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "finetune")
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 0, result.stderr
    assert report_of(result)["next_command"] == "pup finetune doctor --provider fireworks --check"


def test_a_package_that_cannot_be_loaded_is_not_missing(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_packages(monkeypatch, ALL_PACKAGES, broken={"tinker": "ImportError: bad build"})
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    use_pup(healthy_pup().client())
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 1
    sdk = checks_of(result)["fireworks:sdk"]
    assert (sdk["status"], sdk["label"]) == ("fail", "Fireworks Training SDK cannot be loaded")
    assert sdk["detail"] == (
        "the Fireworks Training SDK is installed but cannot be loaded: ImportError: bad build; "
        f"{install_hint('fireworks')}"
    )
    assert sdk["fix"] == install_hint("fireworks")
    renderer = checks_of(result)["fireworks:renderer"]
    assert renderer["detail"] == "the Hugging Face transformers package is installed"


def test_a_package_lookup_that_raises_does_not_crash_doctor(
    use_pup: UsePup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(module: str) -> tuple[str, str]:
        raise RuntimeError("finder exploded")

    monkeypatch.setattr(providers, "module_state", explode)
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    sdk = checks_of(result)["fireworks:sdk"]
    assert sdk["status"] == "optional"
    assert sdk["label"] == "Fireworks Training SDK cannot be loaded"
    assert "check failed unexpectedly (RuntimeError: finder exploded)" in sdk["detail"]


def test_quiet_mode_prints_no_checklist(use_pup: UsePup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_QUIET", "1")
    use_pup(healthy_pup().client())
    assert invoke().stderr == ""


# --- the version ---------------------------------------------------------------------------


def test_the_checkout_commit_is_this_checkout_s_head() -> None:
    root = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    if git is None or not (root / ".git").exists():
        pytest.skip("not a git checkout")
    env = {
        key: value for key, value in os.environ.items() if key not in ("GIT_DIR", "GIT_WORK_TREE")
    }
    described = subprocess.run(  # noqa: S603 - git, with fixed arguments
        [git, "-C", str(root), "describe", "--always", "--dirty", "--abbrev=7"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()
    CHECKOUT_COMMIT.cache_clear()
    try:
        # The working tree may be edited between the two calls: compare the commits.
        assert CHECKOUT_COMMIT().removesuffix("-dirty") == described.removesuffix("-dirty")
    finally:
        CHECKOUT_COMMIT.cache_clear()


def test_the_checkout_ignores_a_repository_named_in_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("not a git checkout")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    CHECKOUT_COMMIT.cache_clear()
    try:
        found = CHECKOUT_COMMIT()
    finally:
        CHECKOUT_COMMIT.cache_clear()
    assert found not in (None, doctor_cli.CHECKOUT_UNKNOWN)


def test_doctor_is_listed_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert "doctor" in result.output


def test_a_baseten_preflight_names_a_workspace_limit_under_the_model_s_context() -> None:
    caps = capabilities(*(profile.base_model for profile in BASETEN_PROFILES))
    loops = FakeLoops()
    checks = preflight_checks(
        provider_info("baseten", Runtime()),
        Runtime(),
        [],
        factory=lambda name, runtime: provider_on(
            loops, routes={"GET /v1/loops/capabilities": (200, caps)}
        ),
    )
    by_name = {check.name: check for check in checks}
    big_profile = next(p for p in BASETEN_PROFILES if p.alias == "qwen3p8-27b")
    big = by_name[f"baseten:model:{big_profile.alias}"]
    assert big.status == "pass"
    assert big.detail.endswith(under_profile_context(big_profile))
    assert by_name["baseten:model:qwen3p5-9b"].detail.endswith("Qwen/Qwen3.5-9B now")
    redaction.clear_secrets()
