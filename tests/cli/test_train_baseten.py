# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train --provider baseten` end to end on the Loops fake: the preview, then a confirmed run,
whose deployment is refused until Baseten deployment lands.

The provider is the real :class:`BasetenProvider` with the Loops fake, its HTTP routes and the
fake tokenizer injected. Each command builds a new provider, as a new process would, while the
fake's runs and checkpoints persist between them, as Baseten's do.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import redaction
from pup_finetune.cli import app
from pup_finetune.cli import deploy as deploy_module
from pup_finetune.cli import train as train_module
from pup_finetune.providers.base import Provider
from pup_finetune.providers.baseten import client
from pup_finetune.runs import RunDir
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from tests.fakes.baseten import FakeLoops
from tests.fakes.tokenizer import FakeChatTokenizer
from tests.providers.baseten.test_baseten_provider import provider_on

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture
def loops() -> FakeLoops:
    return FakeLoops()


@pytest.fixture(autouse=True)
def _baseten(monkeypatch: pytest.MonkeyPatch, loops: FakeLoops) -> Iterator[None]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def get_provider(name: str, runtime: Runtime) -> Provider:
        assert name == "baseten"
        return provider_on(loops, FakeChatTokenizer(merge_newlines=True))

    monkeypatch.setattr(train_module, "get_provider", get_provider)
    monkeypatch.setattr(inputs_module, "get_provider", get_provider)
    monkeypatch.setattr(deploy_module, "get_provider", get_provider)
    yield
    redaction.clear_secrets()


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def train(dataset: Path, run_dir: Path, *extra: str) -> Result:
    return invoke(
        "train",
        "--provider",
        "baseten",
        "--model",
        "qwen3p5-9b",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        "--max-epochs",
        "2",
        *extra,
    )


def test_the_preview_asks_baseten_nothing_paid(
    dataset: Path, tmp_path: Path, loops: FakeLoops
) -> None:
    run_dir = tmp_path / "run"
    result = train(dataset, run_dir)
    assert result.exit_code == 2, result.stderr
    data = output(result)
    preview = data["preview"]
    assert (preview["provider"], preview["model"]) == ("baseten", "qwen3p5-9b")
    # The preview shows what identifies the model; run.json records the whole profile.
    assert set(preview["profile"]) == {"alias", "base_model", "max_seq_len"}
    assert preview["examples"]["train"] > 0
    assert preview["data_destination"] == "training data is sent to baseten"
    assert shlex.split(data["next_command"])[-1] == "--confirm"
    assert not run_dir.exists()
    assert loops.calls == []


def test_a_confirmed_run_trains_leaves_nothing_running_and_cannot_deploy_yet(
    dataset: Path, tmp_path: Path, loops: FakeLoops
) -> None:
    run_dir = tmp_path / "run"
    result = train(dataset, run_dir, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "completed"
    assert data["promoted_model"] is None
    checkpoint = (RunDir(run_dir).read_result() or {})["deployable_checkpoint"]
    assert data["deployable_checkpoint"] == checkpoint["name"]
    assert checkpoint["kind"] == "sampler"
    assert checkpoint["uri"].startswith("bt://loops:run1/sampler_weights/")
    assert loops.runs
    assert not any(run.active for run in loops.runs.values())
    assert "BASETEN_API_KEY" not in os.environ
    deployed = invoke("deploy", str(run_dir), "--confirm")
    assert deployed.exit_code == 2, deployed.stderr
    assert "deploying on Baseten is not available" in deployed.stderr
    assert not (run_dir / "deployment.json").exists()
