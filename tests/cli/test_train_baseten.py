# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train --provider baseten` end to end on the Loops fake: the preview, a confirmed run, then
`deploy` and `undeploy` of the checkpoint it records, and their receipt's failures.

The provider is the real :class:`BasetenProvider` with the Loops fake, its HTTP routes, the
fake tokenizer and a fake clock injected. Each command builds a new provider, as a new process
would, while the fake's runs, checkpoints and models persist between them, as Baseten's do.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner, Result

from pup_finetune import redaction
from pup_finetune.cli import app
from pup_finetune.cli import deploy as deploy_module
from pup_finetune.cli import train as train_module
from pup_finetune.providers.base import Provider
from pup_finetune.providers.baseten import client
from pup_finetune.runs import RunDir, model_id_for_run
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from tests.fakes.baseten import FakeDeployment, FakeLoops
from tests.fakes.fireworks_deploy import FakeClock
from tests.fakes.tokenizer import FakeChatTokenizer
from tests.providers.baseten.test_baseten_provider import KEY, provider_on

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture
def loops() -> FakeLoops:
    return FakeLoops()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(autouse=True)
def _baseten(monkeypatch: pytest.MonkeyPatch, loops: FakeLoops, clock: FakeClock) -> Iterator[None]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def get_provider(name: str, runtime: Runtime) -> Provider:
        assert name == "baseten"
        return provider_on(loops, FakeChatTokenizer(merge_newlines=True), clock=clock)

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
    assert preview["profile"]["renderer"] == "qwen3_5"
    assert preview["examples"]["train"] > 0
    assert preview["data_destination"] == "training data is sent to baseten"
    assert shlex.split(data["next_command"])[-1] == "--confirm"
    assert not run_dir.exists()
    assert loops.calls == []


def test_a_confirmed_run_trains_leaves_nothing_running_and_deploys_its_checkpoint(
    dataset: Path, tmp_path: Path, loops: FakeLoops
) -> None:
    run_dir = tmp_path / "run"
    result = train(dataset, run_dir, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["phase"] == "completed"
    assert data["promoted_model"] is None
    checkpoint = data["deployable_checkpoint"]
    assert checkpoint["kind"] == "sampler"
    assert checkpoint["uri"].startswith("bt://loops:run1/sampler_weights/")
    assert loops.runs
    assert not any(run.active for run in loops.runs.values())
    assert "BASETEN_API_KEY" not in os.environ

    preview = invoke("deploy", str(run_dir))
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["model_source"] == "deployable_checkpoint"
    assert output(preview)["preview"]["shape"] == "the provider's default shape for the model"
    assert loops.models == {}
    deployed = invoke("deploy", str(run_dir), "--confirm")
    assert deployed.exit_code == 0, deployed.stderr
    receipt = output(deployed)
    name = checkpoint["uri"].rsplit("/", 1)[1]
    assert receipt["model"] == checkpoint["uri"]
    assert receipt["deployment_id"] == model_id_for_run(RunDir(run_dir).run_id)
    assert receipt["deployment_name"] == "models/dmodel1/deployments/ddep1"
    assert receipt["inference_model"] == name
    assert receipt["url"] == "https://model-dmodel1.api.baseten.co/deployment/ddep1/sync/v1"
    assert (receipt["shape"], receipt["state"]) == ("H100:1", "READY")
    assert receipt["smoke_test"]["ok"] is True
    assert loops.model_named(receipt["deployment_id"]).deployments[0].replicas == {
        "min_replica": 0,
        "max_replica": 1,
    }
    assert KEY.get_secret_value() not in deployed.stdout + deployed.stderr

    removed = invoke("undeploy", str(run_dir), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is False
    assert loops.deleted == ["dmodel1"]
    assert checkpoint["uri"] in loops.checkpoints  # the checkpoint is kept


@pytest.fixture
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    assert train(dataset, path, "--confirm").exit_code == 0
    return path


def receipt_of(run_dir: Path) -> dict[str, Any]:
    receipt = RunDir(run_dir).read_deployment()
    assert receipt is not None
    return receipt


def undeploy_hint(run_dir: Path) -> str:
    return f"pup finetune undeploy {run_dir} --confirm"


def test_a_timed_out_deploy_leaves_a_receipt_to_undeploy(loops: FakeLoops, run_dir: Path) -> None:
    loops.statuses = ["DEPLOYING"]
    result = invoke("deploy", str(run_dir), "--timeout-minutes", "1", "--confirm")
    assert result.exit_code == 1
    assert "not ready after 1 minutes" in result.stderr
    assert undeploy_hint(run_dir) in result.stderr
    receipt = receipt_of(run_dir)
    assert (receipt["state"], receipt["error"]["type"]) == ("DEPLOYING", "DeploymentNotReady")
    assert invoke("undeploy", str(run_dir), "--confirm").exit_code == 0
    assert loops.deleted == ["dmodel1"]


def test_a_ctrl_c_while_creating_leaves_a_requested_receipt(
    loops: FakeLoops, run_dir: Path
) -> None:
    loops.lost["deploy"] = KeyboardInterrupt()
    assert invoke("deploy", str(run_dir), "--confirm").exit_code == 130
    receipt = receipt_of(run_dir)
    assert receipt["state"] == "REQUESTED"
    assert receipt["deployment_name"] is None
    request = receipt["request"]
    assert (request["base_model"], request["shape"]) == (receipt["model"], "H100:1")
    again = invoke("deploy", str(run_dir), "--confirm")
    assert again.exit_code == 1
    assert "already has the deployment" in again.stderr
    removed = invoke("undeploy", str(run_dir), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is False
    assert loops.deleted == ["dmodel1"]


def test_a_refused_create_leaves_no_receipt(loops: FakeLoops, run_dir: Path) -> None:
    loops.bdn = False
    result = invoke("deploy", str(run_dir), "--confirm")
    assert result.exit_code == 1
    assert "enable BDN volumes" in result.stderr
    assert "undeploy" not in result.stderr
    assert not RunDir(run_dir).deployment_path.exists()


def test_a_failed_lookup_leaves_no_receipt(loops: FakeLoops, run_dir: Path) -> None:
    result = invoke("deploy", str(run_dir), "--shape", "H100:8", "--confirm")
    assert result.exit_code == 2
    assert "no instance type with exactly H100:8" in result.stderr
    assert not RunDir(run_dir).deployment_path.exists()
    assert loops.models == {}


def test_a_lost_create_answer_is_adopted_when_it_matches(loops: FakeLoops, run_dir: Path) -> None:
    loops.lost["deploy"] = httpx.Response(504)
    result = invoke("deploy", str(run_dir), "--confirm")
    assert result.exit_code == 0, result.stderr
    assert output(result)["deployment_name"] == "models/dmodel1/deployments/ddep1"


def test_a_lost_create_answer_is_not_adopted_when_it_serves_another_checkpoint(
    loops: FakeLoops, run_dir: Path
) -> None:
    def serve_another(deployment: FakeDeployment) -> None:
        deployment.checkpoint_ids = ["ckpt-someone-else"]

    loops.after_deploy = serve_another
    loops.lost["deploy"] = httpx.Response(504)
    result = invoke("deploy", str(run_dir), "--confirm")
    assert result.exit_code == 2
    assert "that this deploy did not create" in result.stderr
    assert not RunDir(run_dir).deployment_path.exists()


def test_a_create_answered_without_ids_keeps_the_receipt(loops: FakeLoops, run_dir: Path) -> None:
    hidden: list[str] = []

    def describe_later(deployment: FakeDeployment) -> None:
        hidden.extend(deployment.checkpoint_ids or [])
        deployment.checkpoint_ids = None

    loops.after_deploy = describe_later
    loops.lost["deploy"] = httpx.Response(200, json={"model_id": "dmodel1"})
    result = invoke("deploy", str(run_dir), "--confirm")
    assert result.exit_code == 1
    assert undeploy_hint(run_dir) in result.stderr
    receipt = receipt_of(run_dir)
    assert receipt["state"] == "REQUESTED"
    assert "could not be read" in receipt["error"]["message"]
    # Baseten cannot yet tell the model is this run's: nothing is deleted, the receipt stays.
    refused = invoke("undeploy", str(run_dir), "--confirm")
    assert refused.exit_code == 1
    assert "cannot tell whether" in refused.stderr
    assert "remove" not in refused.stderr
    assert loops.deleted == []
    assert (receipt_of(run_dir)["state"], receipt_of(run_dir)["deleted_at"]) == ("REQUESTED", None)
    # Once it can, the undeploy deletes it.
    loops.model_named(receipt["deployment_id"]).deployments[0].checkpoint_ids = hidden
    assert invoke("undeploy", str(run_dir), "--confirm").exit_code == 0
    assert loops.deleted == ["dmodel1"]


def test_a_failed_smoke_test_exits_1_and_keeps_the_deployment(
    loops: FakeLoops, run_dir: Path
) -> None:
    loops.chat_script += [httpx.Response(503)] * 100
    result = invoke("deploy", str(run_dir), "--confirm")
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["smoke_test"]["ok"] is False
    assert "undeploy" in data["hint"]
    receipt = receipt_of(run_dir)
    assert (receipt["state"], receipt["deleted_at"]) == ("READY", None)
    assert loops.deleted == []
    assert loops.models


def test_undeploying_a_requested_deployment_never_deletes_another_model(
    loops: FakeLoops, run_dir: Path
) -> None:
    loops.fail["deploy"] = KeyboardInterrupt()  # before the create took effect
    assert invoke("deploy", str(run_dir), "--confirm").exit_code == 130
    receipt = receipt_of(run_dir)
    assert receipt["state"] == "REQUESTED"
    other = loops.sampler_checkpoint("someone-elses", base_model="Qwen/Qwen3.5-9B")
    loops.deploy_checkpoint(other, receipt["deployment_id"])
    refused = invoke("undeploy", str(run_dir), "--confirm")
    assert refused.exit_code == 1, refused.stderr
    assert "is not the one run" in refused.stderr
    assert loops.deleted == []
    assert receipt_of(run_dir)["deleted_at"] is None


def test_undeploying_a_requested_deployment_that_was_never_made_deletes_nothing(
    loops: FakeLoops, run_dir: Path
) -> None:
    loops.fail["deploy"] = KeyboardInterrupt()
    assert invoke("deploy", str(run_dir), "--confirm").exit_code == 130
    removed = invoke("undeploy", str(run_dir), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is True
    assert loops.deleted == []
    assert invoke("deploy", str(run_dir), "--confirm").exit_code == 0


def test_undeploying_a_deployment_already_gone_succeeds(loops: FakeLoops, run_dir: Path) -> None:
    assert invoke("deploy", str(run_dir), "--confirm").exit_code == 0
    loops.models.clear()  # deleted elsewhere
    removed = invoke("undeploy", str(run_dir), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is True
