# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import __version__
from pup_finetune import doctor as registry
from pup_finetune.cli import app
from pup_finetune.cli import doctor as doctor_cli
from pup_finetune.datadog.pup import Pup
from pup_finetune.doctor import CheckOutcome
from pup_finetune.errors import UsageError
from tests.fakes.pup import FAKE_BINARY, FakePup

runner = CliRunner()

AUTH_OK = {
    "authenticated": True,
    "expires_at": "2026-01-01T00:00:00Z",
    "org": None,
    "scopes": ["synthetic_read"],
    "site": "datadoghq.com",
    "token_type": "Bearer",
}


def healthy_pup(version: str = "pup 1.10.7\n") -> FakePup:
    fake = FakePup().on("--version", raw=True, stdout=version)
    fake.on(
        "auth", "status", stdout="✅ Authenticated for site: datadoghq.com\n" + json.dumps(AUTH_OK)
    )
    return fake


UsePup = Callable[[Pup], None]


@pytest.fixture
def use_pup(monkeypatch: pytest.MonkeyPatch) -> UsePup:
    """Make the doctor command use a given Pup instead of the real one."""

    def install(pup: Pup) -> None:
        monkeypatch.setattr(doctor_cli, "Pup", lambda: pup)

    return install


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["doctor", *args])


def test_healthy_report_exits_zero(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    result = invoke()
    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {
        "pup": {"path": FAKE_BINARY, "version": "1.10.7", "min_version": "1.10.7", "ok": True},
        "datadog_auth": {
            "authenticated": True,
            "site": "datadoghq.com",
            "expires_at": "2026-01-01T00:00:00Z",
        },
        "python": platform.python_version(),
        "package": __version__,
        "checks": [
            {
                "name": "pup",
                "ok": True,
                "detail": f"pup 1.10.7 at {FAKE_BINARY}",
                "required": True,
            },
            {
                "name": "datadog_auth",
                "ok": True,
                "detail": "pup is signed in to datadoghq.com",
                "required": True,
            },
        ],
        "ok": True,
    }
    assert "Bearer" not in result.stdout
    assert "scopes" not in result.stdout


def test_old_pup_fails_but_still_emits_the_report(use_pup: UsePup) -> None:
    use_pup(healthy_pup("pup 1.9.12\n").client())
    result = invoke()
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["pup"]["ok"] is False
    assert report["pup"]["version"] == "1.9.12"
    assert "older than the minimum 1.10.7" in report["checks"][0]["detail"]
    assert report["datadog_auth"]["authenticated"] is True


def test_missing_pup_fails_and_skips_auth(use_pup: UsePup, tmp_path: Path) -> None:
    fake = FakePup()
    use_pup(Pup(runner=fake, environ={"PATH": str(tmp_path)}))
    result = invoke()
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["pup"] == {"path": None, "version": None, "min_version": "1.10.7", "ok": False}
    assert report["datadog_auth"] == {"authenticated": False, "site": None, "expires_at": None}
    pup_check, auth_check = report["checks"]
    assert "brew install datadog/pack/pup" in pup_check["detail"]
    assert auth_check["detail"].startswith("skipped")
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
    report = json.loads(result.stdout)
    assert report["pup"] == {
        "path": FAKE_BINARY,
        "version": None,
        "min_version": "1.10.7",
        "ok": False,
    }
    pup_check, auth_check = report["checks"]
    assert detail in pup_check["detail"]
    assert not pup_check["detail"].startswith("pup: ")
    assert auth_check["ok"] is True
    assert report["datadog_auth"]["authenticated"] is True


def test_signed_out_fails(use_pup: UsePup) -> None:
    fake = healthy_pup().on(
        "auth", "status", json={"authenticated": False, "site": "datadoghq.com"}
    )
    use_pup(fake.client())
    result = invoke()
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["datadog_auth"]["authenticated"] is False
    assert "pup auth login" in report["checks"][1]["detail"]


def test_auth_status_error_fails(use_pup: UsePup) -> None:
    fake = healthy_pup().on("auth", "status", returncode=1, stderr="Error: no stored token\n")
    use_pup(fake.client())
    result = invoke()
    assert result.exit_code == 1
    detail = json.loads(result.stdout)["checks"][1]["detail"]
    assert "no stored token" in detail
    assert "pup auth login" in detail


def test_auth_status_error_keeps_the_clients_hint(use_pup: UsePup) -> None:
    fake = healthy_pup().on("auth", "status", returncode=1, stderr="Error: 401 Unauthorized\n")
    use_pup(fake.client())
    detail = json.loads(invoke().stdout)["checks"][1]["detail"]
    assert detail.endswith("; check `pup auth status`, and run `pup auth login` to sign in again")


def test_registered_checks_are_reported_and_gate_the_exit_code(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())
    registry.register_check("credentials", lambda: CheckOutcome(True, "fireworks: env"))
    registry.register_check("tokenizer", lambda: CheckOutcome(False, "offline"), required=False)
    result = invoke()
    assert result.exit_code == 0
    checks = json.loads(result.stdout)["checks"]
    assert [c["name"] for c in checks] == ["pup", "datadog_auth", "credentials", "tokenizer"]
    assert checks[3] == {"name": "tokenizer", "ok": False, "detail": "offline", "required": False}

    registry.register_check("provider", lambda: CheckOutcome(False, "no key"))
    result = invoke()
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_a_crashing_check_fails_without_crashing_doctor(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())

    def boom() -> CheckOutcome:
        raise RuntimeError("kaboom\nsecond line")

    def usage() -> CheckOutcome:
        raise UsageError("bad config", hint="fix it")

    registry.register_check("boom", boom)
    registry.register_check("usage", usage)
    result = invoke()
    assert result.exit_code == 1
    checks = {c["name"]: c for c in json.loads(result.stdout)["checks"]}
    assert checks["boom"]["detail"] == (
        "check failed unexpectedly (RuntimeError: kaboom) (set PUP_FINETUNE_DEBUG=1 for a traceback)"
    )
    assert checks["usage"]["detail"] == "bad config; fix it"
    assert "Traceback" not in result.stderr


def test_a_crashing_check_prints_its_traceback_in_debug(use_pup: UsePup) -> None:
    use_pup(healthy_pup().client())

    def boom() -> CheckOutcome:
        raise RuntimeError("kaboom")

    registry.register_check("boom", boom)
    result = runner.invoke(app, ["doctor"], env={"PUP_FINETUNE_DEBUG": "1"})
    assert result.exit_code == 1
    checks = {c["name"]: c for c in json.loads(result.stdout)["checks"]}
    assert checks["boom"]["detail"] == "check failed unexpectedly (RuntimeError: kaboom)"
    assert "Traceback" in result.stderr


def test_register_check_rejects_conflicts() -> None:
    def fn() -> CheckOutcome:
        return CheckOutcome(True, "")

    registry.register_check("a", fn)
    registry.register_check("a", fn)  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        registry.register_check("a", lambda: CheckOutcome(True, ""))
    with pytest.raises(ValueError, match="needs a name"):
        registry.register_check("", fn)
    assert [check.name for check in registry.run_registered_checks()] == ["a"]


def test_doctor_is_listed_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert "doctor" in result.output
