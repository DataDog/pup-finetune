# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pup_finetune import credentials, redaction
from pup_finetune import doctor as doctor_registry
from pup_finetune.errors import UsageError
from pup_finetune.providers import DeployRequest, TrainRequest, get_provider
from pup_finetune.providers import baseten as baseten_package
from pup_finetune.providers.baseten import registration, training
from pup_finetune.providers.baseten.client import BasetenHTTP
from pup_finetune.providers.baseten.provider import BasetenProvider
from pup_finetune.training import SFTSettings

SECRET = "zq9Xsynthetic0000basetenKEYw7Pm"  # noqa: S105 - synthetic test value


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(credentials, "_REGISTRY", {})
    monkeypatch.setattr(doctor_registry, "_REGISTRY", {})
    registration.register()
    monkeypatch.setenv("BASETEN_API_KEY", SECRET)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def train_request(tmp_path: Path) -> TrainRequest:
    return TrainRequest(
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
        model="qwen3p5-9b",
        run_dir=tmp_path / "run",
        run_id="sft-test",
        settings=SFTSettings(),
    )


@pytest.fixture
def deploy_module(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, dict[str, Any]]]:
    """A stand-in ``deploy`` module that records each call."""
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    module = ModuleType("pup_finetune.providers.baseten.deploy")

    def recorder(name: str) -> Any:
        def call(target: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((name, target, kwargs))
            if "http" in kwargs:
                assert not kwargs["http"]._client.is_closed
            return {"called": name}

        return call

    for name in ("deploy_preview", "deploy", "undeploy"):
        setattr(module, name, recorder(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(baseten_package, "deploy", module, raising=False)
    return calls


def test_identity_and_models() -> None:
    provider = BasetenProvider()
    assert provider.name == "baseten"
    assert provider.model_names() == ["qwen3p5-9b", "glm-5p3-flash"]
    assert provider.default_model() == "qwen3p5-9b"


def test_the_registry_returns_the_baseten_provider() -> None:
    baseten_package._provider.cache_clear()
    try:
        assert isinstance(get_provider("baseten"), BasetenProvider)
    finally:
        baseten_package._provider.cache_clear()


def test_plan_training_returns_the_plan(
    monkeypatch: pytest.MonkeyPatch, train_request: TrainRequest
) -> None:
    seen: list[TrainRequest] = []

    def plan_training(request: TrainRequest) -> tuple[dict[str, Any], object]:
        seen.append(request)
        return {"run_id": request.run_id}, object()

    monkeypatch.setattr(training, "plan_training", plan_training)
    assert BasetenProvider().plan_training(train_request) == {"run_id": "sft-test"}
    assert seen == [train_request]


def test_train_resolves_the_key_and_closes_the_http_client(
    monkeypatch: pytest.MonkeyPatch, train_request: TrainRequest
) -> None:
    seen: dict[str, Any] = {}

    def train(request: TrainRequest, *, api_key: Any, http: BasetenHTTP) -> dict[str, Any]:
        seen.update(request=request, key=api_key.get_secret_value(), http=http)
        assert not http._client.is_closed
        return {"status": "completed"}

    monkeypatch.setattr(training, "train", train)
    assert BasetenProvider().train(train_request) == {"status": "completed"}
    assert (seen["request"], seen["key"]) == (train_request, SECRET)
    assert seen["http"]._client.is_closed


def test_train_without_a_key_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, train_request: TrainRequest
) -> None:
    monkeypatch.delenv("BASETEN_API_KEY")
    monkeypatch.setattr(training, "train", None)
    with pytest.raises(UsageError, match="no baseten API key found"):
        BasetenProvider().train(train_request)


def test_deploy_methods_delegate_to_the_deploy_module(
    deploy_module: list[tuple[str, Any, dict[str, Any]]], tmp_path: Path
) -> None:
    request = DeployRequest(
        run_dir=tmp_path,
        serving_mode="sampler",
        accelerator=None,
        max_seq_len=None,
        timeout_seconds=60.0,
    )
    provider = BasetenProvider()
    assert provider.deploy_preview(request) == {"called": "deploy_preview"}
    assert provider.deploy(request) == {"called": "deploy"}
    assert provider.undeploy(tmp_path) == {"called": "undeploy"}
    (preview, deploy, undeploy) = deploy_module
    assert preview == ("deploy_preview", request, {})
    assert deploy[:2] == ("deploy", request)
    assert set(deploy[2]) == {"api_key", "http"}
    assert deploy[2]["api_key"].get_secret_value() == SECRET
    assert undeploy[:2] == ("undeploy", tmp_path)
    assert set(undeploy[2]) == {"http"}
    assert all(call[2]["http"]._client.is_closed for call in (deploy, undeploy))
