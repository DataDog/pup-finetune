# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune models list`, end to end through the Typer app."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import redaction
from pup_finetune.cli import app
from pup_finetune.cli import models as models_cli
from pup_finetune.providers.base import ModelProfile, Provider
from pup_finetune.providers.fireworks.models import PROFILES
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runtime import Runtime
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi

runner = CliRunner()

QWEN = {
    "alias": "qwen3p8-27b",
    "names": ["qwen3p8-27b", "accounts/fireworks/models/qwen3p8-27b", "Qwen/Qwen3.8-27B"],
    "provider": "fireworks",
    "base_model": "accounts/fireworks/models/qwen3p8-27b",
    "max_seq_len": 131072,
    "supports_reasoning": True,
    "default_lora_rank": 8,
    "provider_options": ["max_seq_len"],
}
GLM = {
    "alias": "glm-5p3-flash",
    "names": [
        "glm-5p3-flash",
        "accounts/fireworks/models/glm-5p3-flash",
        "zai-org/GLM-5.3-Flash",
    ],
    "provider": "fireworks",
    "base_model": "accounts/fireworks/models/glm-5p3-flash",
    "max_seq_len": 65536,
    "supports_reasoning": True,
    "default_lora_rank": 8,
    "provider_options": ["max_seq_len"],
}
QWEN_INTERNALS = {
    "renderer": "qwen3_8",
    "tokenizer": "Qwen/Qwen3.8-27B",
    "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
"""What only ``--check`` adds to a profile."""
GLM_INTERNALS = {
    "renderer": "glm5_3",
    "tokenizer": "zai-org/GLM-5.3-Flash",
    "tokenizer_revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeFireworksApi:
    """Serve the Fireworks API from a fake for the `models` commands, with a key in the env."""
    api = FakeFireworksApi()

    def provider(name: str, runtime: Runtime) -> Provider:
        assert name == "fireworks"
        return FireworksProvider(
            runtime, env={"FIREWORKS_API_KEY": API_KEY}, transport=api.transport
        )

    monkeypatch.setattr(models_cli, "get_provider", provider)
    return api


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["models", "list", *args])


def output(result: Result) -> Any:
    return json.loads(result.stdout)


def test_lists_the_fireworks_profiles_without_a_key() -> None:
    result = invoke("--provider", "fireworks")
    assert result.exit_code == 0, result.stderr
    assert output(result) == {"provider": "fireworks", "models": [QWEN, GLM]}


def test_lists_the_fake_profiles_in_dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    result = invoke("--provider", "fake")
    assert result.exit_code == 0, result.stderr
    (model,) = output(result)["models"]
    assert model["alias"] == "fake-model"
    assert model["provider"] == "fake"
    assert "renderer" not in model


def test_the_fake_provider_needs_dev_mode() -> None:
    result = invoke("--provider", "fake")
    assert result.exit_code == 2
    assert "PUP_FINETUNE_DEV=1" in result.stderr


def test_an_unknown_provider_lists_the_known_ones() -> None:
    result = invoke("--provider", "nope")
    assert result.exit_code == 2
    assert "fireworks" in result.stderr


def test_without_a_provider_it_lists_every_provider_s_models() -> None:
    result = invoke()
    assert result.exit_code == 0, result.stderr
    body = output(result)
    assert body["provider"] is None
    assert [(m["provider"], m["alias"]) for m in body["models"]] == [
        ("fireworks", "qwen3p8-27b"),
        ("fireworks", "glm-5p3-flash"),
        ("baseten", "qwen3p5-9b"),
        ("baseten", "glm-5p3-flash"),
        ("baseten", "qwen3p8-27b"),
    ]
    assert body["models"][0] == QWEN
    assert all("tokenizer" not in model for model in body["models"])


def test_without_a_provider_dev_mode_lists_the_fake_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    providers = {model["provider"] for model in output(invoke())["models"]}
    assert providers == {"fake", "fireworks", "baseten"}


def test_check_preflights_each_model(fake: FakeFireworksApi) -> None:
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 0, result.stderr
    assert output(result) == {
        "provider": "fireworks",
        "models": [
            {**QWEN, **QWEN_INTERNALS, "preflight": {"ok": True}},
            {**GLM, **GLM_INTERNALS, "preflight": {"ok": True}},
        ],
        "ok": True,
    }
    assert [request.url.path for request in fake.requests] == [
        "/v1/accounts/fireworks/models/qwen3p8-27b",
        "/v1/accounts/fireworks/models/glm-5p3-flash",
    ]
    assert API_KEY not in result.stdout + result.stderr


def test_a_failed_check_is_reported_and_exits_1(fake: FakeFireworksApi) -> None:
    fake.models["qwen3p8-27b"]["state"] = "DEPLOYING"
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    body = output(result)
    assert body["ok"] is False
    preflight = body["models"][0]["preflight"]
    assert preflight["ok"] is False
    assert "its state is DEPLOYING, not READY" in preflight["error"]
    assert "models list --provider fireworks --check" in preflight["hint"]


class _TwoProfileProvider(FireworksProvider):
    """Serves a profile Fireworks does not have, ahead of the real one."""

    MISSING = dataclasses.replace(
        PROFILES[0], alias="qwen-missing", base_model="accounts/fireworks/models/qwen-missing"
    )

    def models(self) -> list[ModelProfile]:
        return [self.MISSING, *PROFILES]


def test_check_reports_every_profile_on_its_own(
    fake: FakeFireworksApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    def provider(name: str, runtime: Runtime) -> Provider:
        return _TwoProfileProvider(
            runtime, env={"FIREWORKS_API_KEY": API_KEY}, transport=fake.transport
        )

    monkeypatch.setattr(models_cli, "get_provider", provider)
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    body = output(result)
    assert body["ok"] is False
    missing, qwen, glm = body["models"]
    assert glm["preflight"] == {"ok": True}
    assert missing["alias"] == "qwen-missing"
    assert missing["preflight"]["ok"] is False
    assert "has no model" in missing["preflight"]["error"]
    assert qwen == {**QWEN, **QWEN_INTERNALS, "preflight": {"ok": True}}
    assert [request.url.path for request in fake.requests] == [
        "/v1/accounts/fireworks/models/qwen-missing",
        "/v1/accounts/fireworks/models/qwen3p8-27b",
        "/v1/accounts/fireworks/models/glm-5p3-flash",
    ]


def test_a_rejected_key_is_reported_without_the_key(fake: FakeFireworksApi) -> None:
    fake.api_key = "fw-synthetic-other-key"  # so the key the provider sends is rejected
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    preflight = output(result)["models"][0]["preflight"]
    assert preflight["error"].startswith("fireworks: looking up the Fireworks model")
    assert "pup finetune auth login fireworks" in preflight["hint"]
    assert API_KEY not in result.stdout + result.stderr


def test_check_without_a_key_explains_how_to_add_one() -> None:
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 1
    preflight = output(result)["models"][0]["preflight"]
    assert preflight == {
        "ok": False,
        "error": "no fireworks API key found",
        "hint": "run `pup finetune auth login fireworks` in your own terminal, or set "
        "FIREWORKS_API_KEY",
    }


def test_check_is_allowed_read_only(
    fake: FakeFireworksApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    result = invoke("--provider", "fireworks", "--check")
    assert result.exit_code == 0, result.stderr


def test_check_works_for_the_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    result = invoke("--provider", "fake", "--check")
    assert result.exit_code == 0, result.stderr
    assert output(result)["models"][0]["preflight"] == {"ok": True}


@pytest.mark.parametrize(
    ("args", "heavy"),
    [
        (["--provider", "fireworks"], ["transformers", "fireworks", "tinker", "httpx"]),
        ([], ["transformers", "fireworks", "tinker", "baseten"]),
    ],
)
def test_listing_imports_no_sdk_or_tokenizer(
    args: list[str], heavy: list[str], tmp_path: Path
) -> None:
    code = (
        textwrap.dedent(
            """
        import json, sys
        from pup_finetune.cli import app
        from pup_finetune.cli._common import run_app

        try:
            run_app(app, ["models", "list", *ARGS])
        except SystemExit as exc:
            code = exc.code
        heavy = set(HEAVY)
        loaded = sorted({name.partition(".")[0] for name in sys.modules} & heavy)
        print(json.dumps({"exit": code, "heavy": loaded}), file=sys.stderr)
        """
        )
        .replace("ARGS", repr(args))
        .replace("HEAVY", repr(heavy))
    )
    env = {k: v for k, v in os.environ.items() if k != "FIREWORKS_API_KEY"}
    env.update(HOME=str(tmp_path), PYTHON_KEYRING_BACKEND="keyring.backends.fail.Keyring")
    completed = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert json.loads(completed.stderr.strip().splitlines()[-1]) == {"exit": 0, "heavy": []}
    assert json.loads(completed.stdout)["models"][0]["alias"] == "qwen3p8-27b"
