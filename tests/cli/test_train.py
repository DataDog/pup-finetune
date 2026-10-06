# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune train` through the Typer app, with a fake provider in the registry."""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import providers
from pup_finetune.cli import app
from pup_finetune.providers import TrainRequest
from pup_finetune.training import SFTSettings

runner = CliRunner()

ROWS = ["--train", "my rows/train.jsonl", "--validation", "rows/validation.jsonl"]


@dataclass
class FakeTrainingProvider:
    """Records the requests it gets; plans and trains nothing."""

    name: str = "fake"
    calls: list[tuple[str, TrainRequest]] = field(default_factory=list)

    def model_names(self) -> list[str]:
        return ["fake-small", "fake-large"]

    def default_model(self) -> str:
        return "fake-small"

    def plan_training(self, request: TrainRequest) -> dict[str, Any]:
        self.calls.append(("plan_training", request))
        return {"provider": self.name, "run_id": request.run_id, "model": request.model}

    def train(self, request: TrainRequest) -> dict[str, Any]:
        self.calls.append(("train", request))
        return {"status": "completed", "run_id": request.run_id, "run_dir": str(request.run_dir)}


def install(monkeypatch: pytest.MonkeyPatch, name: str = "fake") -> FakeTrainingProvider:
    """Register a fake provider package called ``name`` in the provider registry."""
    provider = FakeTrainingProvider(name=name)
    module = ModuleType(f"pup_finetune.providers.{name}")
    module.PROVIDER = provider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if name not in providers.provider_names():
        monkeypatch.setattr(providers, "_PROVIDERS", (*providers.provider_names(), name))
    return provider


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeTrainingProvider:
    monkeypatch.chdir(tmp_path)
    return install(monkeypatch)


def invoke(*args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, ["train", *ROWS, "--provider", "fake", *args], env=env)


def rerun(next_command: str) -> Result:
    argv = shlex.split(next_command)
    assert argv[:2] == ["pup", "finetune"]
    return runner.invoke(app, argv[2:])


def test_preview_without_confirm_prints_the_plan_and_the_exact_rerun(
    fake: FakeTrainingProvider,
) -> None:
    result = invoke("--run-id", "sft-synthetic")
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "status": "confirmation_required",
        "preview": {"provider": "fake", "run_id": "sft-synthetic", "model": "fake-small"},
        "next_command": "pup finetune train --train 'my rows/train.jsonl' "
        "--validation rows/validation.jsonl --provider fake --model fake-small "
        "--run-id sft-synthetic --run-dir finetune/runs/sft-synthetic --confirm",
    }
    assert "Confirmation required" in result.stderr
    assert fake.calls == [
        (
            "plan_training",
            TrainRequest(
                train_path=Path("my rows/train.jsonl"),
                validation_path=Path("rows/validation.jsonl"),
                model="fake-small",
                run_dir=Path("finetune/runs/sft-synthetic"),
                run_id="sft-synthetic",
                settings=SFTSettings(),
            ),
        )
    ]


def test_the_rerun_pins_the_generated_run_id_and_trains_what_was_previewed(
    fake: FakeTrainingProvider,
) -> None:
    preview = invoke()
    assert preview.exit_code == 2
    ((_, planned),) = fake.calls
    assert re.fullmatch(r"sft-\d{8}-\d{6}-[0-9a-f]{8}", planned.run_id)
    assert planned.run_dir == Path("finetune/runs") / planned.run_id
    fake.calls.clear()

    result = rerun(json.loads(preview.stdout)["next_command"])
    assert result.exit_code == 0, result.stderr
    assert fake.calls == [("plan_training", planned), ("train", planned)]
    assert json.loads(result.stdout) == {
        "status": "completed",
        "run_id": planned.run_id,
        "run_dir": str(planned.run_dir),
    }


def test_settings_flags_reach_the_request_and_the_rerun(fake: FakeTrainingProvider) -> None:
    flags = {
        "--epochs": "3",
        "--learning-rate": "0.0002",
        "--batch-size": "8",
        "--lora-rank": "16",
        "--max-seq-len": "4096",
        "--seed": "7",
        "--early-stopping-patience": "2",
        "--max-spend-usd": "12.5",
        "--hourly-rate-usd": "4.0",
    }
    args = [arg for item in flags.items() for arg in item]
    preview = invoke("--model", "fake-large", "--run-dir", "out/run 1", *args)
    assert preview.exit_code == 2
    ((_, planned),) = fake.calls
    assert planned.model == "fake-large"
    assert planned.run_dir == Path("out/run 1")
    assert planned.settings == SFTSettings(
        max_epochs=3,
        learning_rate=0.0002,
        batch_size=8,
        lora_rank=16,
        max_seq_len=4096,
        seed=7,
        early_stopping_patience=2,
        max_spend_usd=12.5,
        hourly_rate_usd=4.0,
    )
    next_command = json.loads(preview.stdout)["next_command"]
    assert "--run-dir 'out/run 1'" in next_command
    assert next_command.endswith(f"{' '.join(args)} --confirm")
    fake.calls.clear()

    assert rerun(next_command).exit_code == 0
    assert [request for _, request in fake.calls] == [planned, planned]


def test_confirm_plans_then_trains_and_prints_the_result(fake: FakeTrainingProvider) -> None:
    result = invoke("--run-id", "sft-synthetic", "--confirm")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "completed"
    assert [name for name, _ in fake.calls] == ["plan_training", "train"]


def test_auto_approve_and_agent_mode_never_imply_confirm(fake: FakeTrainingProvider) -> None:
    result = invoke(env={"PUP_AUTO_APPROVE": "true", "PUP_AGENT_MODE": "true"})
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "confirmation_required"
    assert [name for name, _ in fake.calls] == ["plan_training"]


def test_read_only_blocks_before_the_preview(fake: FakeTrainingProvider) -> None:
    result = invoke(env={"PUP_READ_ONLY": "true"})
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "PUP_READ_ONLY" in result.stderr
    assert fake.calls == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--epochs", "0"], "--epochs: Input should be greater than or equal to 1"),
        (["--learning-rate", "-1"], "--learning-rate: Input should be greater than 0"),
        (["--learning-rate", "nan"], "--learning-rate: Input should be a finite number"),
        (["--max-spend-usd", "5"], "--max-spend-usd and --hourly-rate-usd must be set together"),
    ],
)
def test_invalid_settings_exit_2_before_anything_runs(
    fake: FakeTrainingProvider, args: list[str], message: str
) -> None:
    result = invoke(*args, env={"PUP_READ_ONLY": "true"})
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"Error: invalid training settings: {message}" in result.stderr
    assert fake.calls == []


@pytest.mark.parametrize("run_id", ["../escape", ".hidden", "a/b", ""])
def test_invalid_run_id_is_a_usage_error(fake: FakeTrainingProvider, run_id: str) -> None:
    result = invoke("--run-id", run_id)
    assert result.exit_code == 2
    assert "invalid --run-id" in result.stderr
    assert fake.calls == []


def test_unknown_provider_is_a_usage_error(fake: FakeTrainingProvider) -> None:
    result = runner.invoke(app, ["train", *ROWS, "--provider", "acme"])
    assert result.exit_code == 2
    assert "unknown provider 'acme'" in result.stderr
    assert fake.calls == []


def test_provider_defaults_to_baseten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    provider = install(monkeypatch, "baseten")
    result = runner.invoke(app, ["train", *ROWS, "--run-id", "sft-synthetic"])
    assert result.exit_code == 2
    assert " --provider baseten " in json.loads(result.stdout)["next_command"]
    assert [name for name, _ in provider.calls] == ["plan_training"]
