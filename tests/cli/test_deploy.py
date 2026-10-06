# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune deploy` and `undeploy` through the Typer app, with a fake provider."""

from __future__ import annotations

import inspect
import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, get_args, get_origin

import pytest
import typer
from typer.testing import CliRunner, Result

from pup_finetune import providers
from pup_finetune.cli import app
from pup_finetune.providers import DeployRequest
from pup_finetune.training import RunDir

runner = CliRunner()


@dataclass
class FakeServingProvider:
    """Records the requests it gets; serves nothing."""

    name: str = "fake"
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def deploy_preview(self, request: DeployRequest) -> dict[str, Any]:
        self.calls.append(("deploy_preview", request))
        return {"serving_mode": request.serving_mode, "billing": "per hour"}

    def deploy(self, request: DeployRequest) -> dict[str, Any]:
        self.calls.append(("deploy", request))
        return {"state": "ready", "served_model": "synthetic-checkpoint"}

    def undeploy(self, run_dir: Path) -> dict[str, Any]:
        self.calls.append(("undeploy", run_dir))
        return {"state": "inactive"}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeServingProvider:
    provider = FakeServingProvider()
    module = ModuleType("pup_finetune.providers.fake")
    module.PROVIDER = provider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(providers, "_PROVIDERS", (*providers.provider_names(), "fake"))
    return provider


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = RunDir(tmp_path / "run 1")
    run.create()
    run.write_plan({"provider": "fake", "run_id": "sft-synthetic"})
    return run.path


def invoke(*args: str, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(app, list(args), env=env)


def rerun(next_command: str) -> Result:
    argv = shlex.split(next_command)
    assert argv[:2] == ["pup", "finetune"]
    return runner.invoke(app, argv[2:])


def request(run_dir: Path, **changes: Any) -> DeployRequest:
    fields: dict[str, Any] = {
        "run_dir": run_dir,
        "serving_mode": "dedicated",
        "accelerator": None,
        "max_seq_len": None,
        "timeout_seconds": 1800.0,
    }
    return DeployRequest(**{**fields, **changes})


def test_deploy_preview_uses_the_provider_from_the_run_plan(
    fake: FakeServingProvider, run_dir: Path
) -> None:
    result = invoke("deploy", "--run-dir", str(run_dir), "--serving-mode", "sampler")
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "status": "confirmation_required",
        "preview": {"serving_mode": "sampler", "billing": "per hour"},
        "next_command": shlex.join(
            ["pup", "finetune", "deploy", "--run-dir", str(run_dir), "--serving-mode", "sampler"]
        )
        + " --confirm",
    }
    assert fake.calls == [("deploy_preview", request(run_dir, serving_mode="sampler"))]


def test_deploy_rerun_keeps_the_dedicated_options(fake: FakeServingProvider, run_dir: Path) -> None:
    options = ["--accelerator", "H100:1", "--max-seq-len", "8192", "--timeout", "600.0"]
    preview = invoke("deploy", "--run-dir", str(run_dir), *options)
    assert preview.exit_code == 2
    next_command = json.loads(preview.stdout)["next_command"]
    assert next_command.endswith(f"--serving-mode dedicated {' '.join(options)} --confirm")
    expected = request(run_dir, accelerator="H100:1", max_seq_len=8192, timeout_seconds=600.0)
    assert fake.calls == [("deploy_preview", expected)]
    fake.calls.clear()

    result = rerun(next_command)
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"state": "ready", "served_model": "synthetic-checkpoint"}
    assert fake.calls == [("deploy_preview", expected), ("deploy", expected)]


@pytest.mark.parametrize(("command", "calls"), [("deploy", ["deploy_preview"]), ("undeploy", [])])
def test_auto_approve_and_agent_mode_never_imply_confirm(
    fake: FakeServingProvider, run_dir: Path, command: str, calls: list[str]
) -> None:
    env = {"PUP_AUTO_APPROVE": "true", "PUP_AGENT_MODE": "true"}
    result = invoke(command, "--run-dir", str(run_dir), env=env)
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "confirmation_required"
    assert [name for name, _ in fake.calls] == calls


def test_deploy_read_only_blocks_before_the_preview(
    fake: FakeServingProvider, run_dir: Path
) -> None:
    result = invoke("deploy", "--run-dir", str(run_dir), env={"PUP_READ_ONLY": "true"})
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "PUP_READ_ONLY" in result.stderr
    assert fake.calls == []


@pytest.mark.parametrize(
    "args", [["--serving-mode", "temporary"], ["--timeout", "0"], ["--timeout", "7201"]]
)
def test_deploy_rejects_invalid_options(
    fake: FakeServingProvider, run_dir: Path, args: list[str]
) -> None:
    result = invoke("deploy", "--run-dir", str(run_dir), *args)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert fake.calls == []


def test_deploy_needs_the_run_dir_of_a_training_run(
    fake: FakeServingProvider, tmp_path: Path
) -> None:
    result = invoke("deploy", "--run-dir", str(tmp_path / "missing"))
    assert result.exit_code == 1
    assert "plan.json does not exist" in result.stderr
    assert "Hint: pass the --run-dir of a training run" in result.stderr
    assert fake.calls == []


def test_a_plan_without_a_provider_is_a_state_error(
    fake: FakeServingProvider, run_dir: Path
) -> None:
    RunDir(run_dir).write_plan({"run_id": "sft-synthetic"})
    result = invoke("deploy", "--run-dir", str(run_dir))
    assert result.exit_code == 1
    assert "names no provider" in result.stderr
    assert fake.calls == []


def test_undeploy_preview_names_the_run_and_runs_nothing(
    fake: FakeServingProvider, run_dir: Path
) -> None:
    result = invoke("undeploy", "--run-dir", str(run_dir))
    assert result.exit_code == 2
    output = json.loads(result.stdout)
    assert output["status"] == "confirmation_required"
    assert output["preview"]["provider"] == "fake"
    assert output["preview"]["run_dir"] == str(run_dir)
    assert output["next_command"] == shlex.join(
        ["pup", "finetune", "undeploy", "--run-dir", str(run_dir), "--confirm"]
    )
    assert fake.calls == []


def test_undeploy_confirm_stops_the_recorded_deployment(
    fake: FakeServingProvider, run_dir: Path
) -> None:
    result = invoke("undeploy", "--run-dir", str(run_dir), "--confirm")
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"state": "inactive"}
    assert fake.calls == [("undeploy", run_dir)]


def test_undeploy_read_only_blocks_before_the_preview(
    fake: FakeServingProvider, run_dir: Path
) -> None:
    result = invoke("undeploy", "--run-dir", str(run_dir), env={"PUP_READ_ONLY": "true"})
    assert result.exit_code == 2
    assert result.stdout == ""
    assert fake.calls == []


def test_commands_are_listed_and_take_no_secret_option() -> None:
    result = runner.invoke(app, ["--help"])
    commands: Any = getattr(typer.main.get_command(app), "commands", {})
    for name in ("train", "deploy", "undeploy"):
        assert name in result.output
        options = [p.name for p in commands[name].params if p.param_type_name == "option"]
        assert not [o for o in options if any(w in o for w in ("key", "token", "secret"))]


def test_no_option_is_a_literal_which_typer_before_0_19_rejects() -> None:
    for command in app.registered_commands:
        assert command.callback is not None
        for param in inspect.signature(command.callback, eval_str=True).parameters.values():
            types = (param.annotation, *get_args(param.annotation))
            assert Literal not in {get_origin(t) for t in types}, param.name
