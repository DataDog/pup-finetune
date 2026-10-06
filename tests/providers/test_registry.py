# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from pup_finetune import providers
from pup_finetune.errors import UsageError
from pup_finetune.providers import DeployRequest, TrainRequest, get_provider, provider_names
from pup_finetune.providers import baseten as baseten_package
from pup_finetune.training import SFTSettings


class SyntheticProvider:
    """Stands in for a provider package's PROVIDER."""


@pytest.fixture
def fresh_baseten_provider() -> Iterator[None]:
    baseten_package._provider.cache_clear()
    yield
    baseten_package._provider.cache_clear()


def test_provider_names() -> None:
    assert provider_names() == ["baseten"]


def test_unknown_provider_is_a_usage_error_listing_the_names() -> None:
    with pytest.raises(UsageError, match="unknown provider 'acme'") as caught:
        get_provider("acme")
    assert caught.value.exit_code == 2
    assert caught.value.hint == "available providers: baseten"


def test_only_registered_names_are_imported() -> None:
    with pytest.raises(UsageError):
        get_provider("baseten.registration")


def test_get_provider_imports_the_package_and_returns_its_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pup_finetune.providers.synthetic")
    module.PROVIDER = SyntheticProvider()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(providers, "_PROVIDERS", ("baseten", "synthetic"))
    assert provider_names() == ["baseten", "synthetic"]
    assert get_provider("synthetic") is module.PROVIDER


@pytest.mark.usefixtures("fresh_baseten_provider")
def test_baseten_provider_loads_its_module_on_first_access(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("pup_finetune.providers.baseten.provider")
    module.BasetenProvider = SyntheticProvider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    provider = get_provider("baseten")
    assert isinstance(provider, SyntheticProvider)
    assert get_provider("baseten") is provider
    assert baseten_package.PROVIDER is provider


def test_baseten_package_has_no_other_lazy_attributes() -> None:
    with pytest.raises(AttributeError, match="no attribute 'MISSING'"):
        _ = baseten_package.MISSING


def test_requests_are_frozen(tmp_path: Path) -> None:
    train = TrainRequest(
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
        model="qwen3p5-9b",
        run_dir=tmp_path / "run",
        run_id="sft-synthetic",
        settings=SFTSettings(),
    )
    deploy = DeployRequest(
        run_dir=tmp_path / "run",
        serving_mode="sampler",
        accelerator=None,
        max_seq_len=None,
        timeout_seconds=60.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        train.model = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        deploy.serving_mode = "dedicated"  # type: ignore[misc]


def test_import_registers_baseten_without_loading_heavy_modules() -> None:
    script = textwrap.dedent(
        """
        import json, sys
        import pup_finetune.providers
        from pup_finetune import credentials, doctor
        heavy = ["httpx", "transformers", "baseten", "pup_finetune.providers.baseten.provider"]
        print(json.dumps({
            "env_var": credentials.get_spec("baseten").env_var,
            "checks": list(doctor._REGISTRY),
            "loaded": [name for name in heavy if name in sys.modules],
        }))
        """
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    report = json.loads(completed.stdout)
    assert report["env_var"] == "BASETEN_API_KEY"
    assert report["checks"] == [
        "credentials:baseten",
        "baseten:loops-sdk",
        "rendering:transformers",
    ]
    assert report["loaded"] == []
