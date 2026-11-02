# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train --provider baseten` end to end on the Loops fake: the preview, a confirmed run, then
`deploy` and `undeploy` of the checkpoint it records, and their receipt's failures; and what a
Ctrl-C or an unexpected error while a command creates a Loops trainer or sampler says may
remain.

The provider is the real :class:`BasetenProvider` with the Loops fake, its HTTP routes, the
fake tokenizer and a fake clock injected. Each command builds a new provider, as a new process
would, while the fake's runs, checkpoints and models persist between them, as Baseten's do.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner, Result

from pup_finetune import providers, redaction
from pup_finetune.cli import app
from pup_finetune.cli import deploy as deploy_module
from pup_finetune.cli import train as train_module
from pup_finetune.evaluation import benchmark as benchmark_module
from pup_finetune.evaluation import runner as eval_module
from pup_finetune.providers.base import (
    MAX_SEQ_LEN_FLAG,
    PROVIDER_LIMIT,
    Provider,
    under_profile_context,
)
from pup_finetune.providers.baseten import client
from pup_finetune.runs import RunDir, model_id_for_run
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from tests.fakes.baseten import (
    OVER_WORKSPACE_LIMIT,
    SHORT_PROFILE,
    WITHIN_WORKSPACE_LIMIT,
    WORKSPACE_LIMIT,
    Answer,
    FakeDeployment,
    FakeLoops,
    capabilities,
    short_capabilities,
)
from tests.fakes.fireworks_deploy import FakeClock
from tests.fakes.pushed import mark_pushed
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


@pytest.fixture
def routes() -> dict[str, Answer]:
    """Baseten HTTP routes the next command's provider answers before the fake's own."""
    return {}


@pytest.fixture(autouse=True)
def _baseten(
    monkeypatch: pytest.MonkeyPatch, loops: FakeLoops, clock: FakeClock, routes: dict[str, Answer]
) -> Iterator[None]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def get_provider(name: str, runtime: Runtime) -> Provider:
        assert name == "baseten"
        return provider_on(
            loops, FakeChatTokenizer(merge_newlines=True), clock=clock, routes=routes
        )

    monkeypatch.setattr(train_module, "get_provider", get_provider)
    monkeypatch.setattr(inputs_module, "get_provider", get_provider)
    monkeypatch.setattr(deploy_module, "get_provider", get_provider)
    monkeypatch.setattr(eval_module, "get_provider", get_provider)
    monkeypatch.setattr(benchmark_module, "get_provider", get_provider)
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
    return mark_pushed(path)


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


@pytest.fixture
def short_workspace(routes: dict[str, Answer]) -> None:
    """A workspace that trains Qwen3.5 9B on fewer tokens than its profile's context."""
    routes["GET /v1/loops/capabilities"] = (
        200,
        short_capabilities(capabilities(SHORT_PROFILE.base_model)),
    )


@pytest.mark.usefixtures("short_workspace")
def test_a_workspace_limit_under_the_model_s_context_is_trained_at_and_noted(
    dataset: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    result = train(dataset, run_dir)
    assert result.exit_code == 2, result.stderr
    assert (
        f"note: baseten trains {SHORT_PROFILE.base_model} on at most {WORKSPACE_LIMIT} tokens "
        f"for this account, {under_profile_context(SHORT_PROFILE)}"
    ) in result.stderr
    data = output(result)
    preview = data["preview"]
    assert preview["profile"]["max_seq_len"] == WORKSPACE_LIMIT
    settings = preview["settings"]
    assert (settings["max_seq_len"], settings["max_seq_len_source"]) == (
        WORKSPACE_LIMIT,
        PROVIDER_LIMIT,
    )
    assert "max_seq_len" not in preview["defaults_used"]
    assert f"context {WORKSPACE_LIMIT:,} ({PROVIDER_LIMIT})" in result.stderr
    # Pinned, so the run trains at the context the preview showed.
    words = shlex.split(data["next_command"])
    assert words[words.index("--max-seq-len") + 1] == str(WORKSPACE_LIMIT)

    confirmed = invoke(*words[2:])
    assert confirmed.exit_code == 0, confirmed.stderr
    assert RunDir(run_dir).profile().max_seq_len == WORKSPACE_LIMIT


@pytest.mark.usefixtures("short_workspace")
def test_a_context_flag_within_the_workspace_limit_is_kept_as_given(
    dataset: Path, tmp_path: Path
) -> None:
    result = train(dataset, tmp_path / "run", "--max-seq-len", str(WITHIN_WORKSPACE_LIMIT))
    assert result.exit_code == 2, result.stderr
    assert "note:" not in result.stderr
    settings = output(result)["preview"]["settings"]
    assert (settings["max_seq_len"], settings["max_seq_len_source"]) == (
        WITHIN_WORKSPACE_LIMIT,
        MAX_SEQ_LEN_FLAG,
    )


@pytest.mark.usefixtures("short_workspace")
@pytest.mark.parametrize(
    "flag",
    [
        ["--max-seq-len", str(OVER_WORKSPACE_LIMIT)],
        ["--provider-option", f"max_seq_len={OVER_WORKSPACE_LIMIT}"],
    ],
)
def test_a_context_flag_over_the_workspace_limit_is_refused(
    dataset: Path, tmp_path: Path, loops: FakeLoops, flag: list[str]
) -> None:
    run_dir = tmp_path / "run"
    result = train(dataset, run_dir, *flag)
    assert result.exit_code == 2
    assert (
        f"at most {WORKSPACE_LIMIT} tokens in this workspace, shorter than the {OVER_WORKSPACE_LIMIT} to train at"
        in result.stderr
    )
    assert f"pass --max-seq-len N with N at most {WORKSPACE_LIMIT}" in result.stderr
    assert not run_dir.exists()
    assert loops.calls == []


def test_a_confirmed_run_trains_leaves_nothing_running_and_deploys_its_checkpoint(
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

    preview = invoke("deploy", str(run_dir))
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["model_source"] == "deployable_checkpoint"
    assert output(preview)["preview"]["shape"] == "the provider's default shape for the model"
    assert loops.models == {}
    deployed = invoke("deploy", str(run_dir), "--confirm")
    assert deployed.exit_code == 0, deployed.stderr
    result = output(deployed)
    name = checkpoint["uri"].rsplit("/", 1)[1]
    assert result["model"] == checkpoint["uri"]
    assert result["deployment_id"] == model_id_for_run(RunDir(run_dir).run_id)
    assert result["deployment_name"] == "models/dmodel1/deployments/ddep1"
    assert (result["shape"], result["state"]) == ("H100:1", "READY")
    assert result["smoke_test"]["ok"] is True
    url = "https://model-dmodel1.api.baseten.co/deployment/ddep1/sync/v1"
    usage = result["usage"]
    assert (usage["base_url"], usage["model"], usage["api_key_env"]) == (
        url,
        name,
        "BASETEN_API_KEY",
    )
    assert '"Authorization: Bearer $BASETEN_API_KEY"' in usage["example"]
    receipt = receipt_of(run_dir)
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


def test_the_lora_alpha_is_refused_in_the_preview(
    dataset: Path, tmp_path: Path, loops: FakeLoops
) -> None:
    run_dir = tmp_path / "run"
    result = train(dataset, run_dir, "--lora-alpha", "16")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert (
        "Error [usage]: --lora-alpha is not supported on baseten: Baseten sets the LoRA alpha "
        "itself"
    ) in result.stderr
    assert "Hint: drop --lora-alpha" in result.stderr
    assert not run_dir.exists()
    assert loops.calls == []


def test_resuming_a_run_recorded_with_a_lora_alpha_says_to_start_a_new_run(
    dataset: Path, tmp_path: Path, loops: FakeLoops
) -> None:
    run_dir = tmp_path / "run"
    assert train(dataset, run_dir, "--confirm").exit_code == 0
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["spec"]["lora_alpha"] = 16
    (run_dir / "run.json").write_text(json.dumps(manifest))
    state = json.loads((run_dir / "state.json").read_text())
    (run_dir / "state.json").write_text(json.dumps({**state, "phase": "interrupted"}))
    result = invoke("resume", str(run_dir))
    assert result.exit_code == 2
    assert "--lora-alpha is not supported on baseten" in result.stderr
    assert "Hint: the run was recorded with --lora-alpha" in result.stderr
    assert "start a new run without --lora-alpha" in result.stderr
    assert "drop --lora-alpha" not in result.stderr


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
    return f"pup finetune undeploy {run_dir}"


def test_a_timed_out_deploy_leaves_a_receipt_to_undeploy(loops: FakeLoops, run_dir: Path) -> None:
    loops.statuses = ["DEPLOYING"]
    result = invoke("deploy", str(run_dir), "--timeout-minutes", "1", "--confirm")
    assert result.exit_code == 1
    assert "not ready after 1 minutes" in result.stderr
    assert undeploy_hint(run_dir) in result.stderr
    assert f"{undeploy_hint(run_dir)} --confirm" not in result.stderr
    receipt = receipt_of(run_dir)
    assert (receipt["state"], receipt["error"]["type"]) == ("DEPLOYING", "DeploymentNotReady")
    assert invoke("undeploy", str(run_dir), "--confirm").exit_code == 0
    assert loops.deleted == ["dmodel1"]


def test_a_ctrl_c_while_creating_leaves_a_requested_receipt(
    loops: FakeLoops, run_dir: Path
) -> None:
    loops.lost["deploy"] = KeyboardInterrupt()
    interrupted = invoke("deploy", str(run_dir), "--confirm")
    assert interrupted.exit_code == 1
    assert "Error [interrupted]: stopped by Ctrl-C\n" in interrupted.stderr
    receipt = receipt_of(run_dir)
    assert receipt["state"] == "REQUESTED"
    assert receipt["deployment_name"] is None
    request = receipt["request"]
    assert (request["base_model"], request["shape"]) == (receipt["model"], "H100:1")
    again = invoke("deploy", str(run_dir), "--confirm")
    assert again.exit_code == 2
    assert "Error [wrong_phase]: " in again.stderr
    assert "already has the deployment" in again.stderr
    removed = invoke("undeploy", str(run_dir), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is False
    assert loops.deleted == ["dmodel1"]


def left(what: str, session: str) -> str:
    return f"a {what} may remain in Loops session {session}; deactivate it in the Baseten UI"


def warned(result: Result, text: str) -> bool:
    return any(line.startswith("warning: ") and text in line for line in result.stderr.splitlines())


def evaluate(run_dir: Path) -> Result:
    return invoke(
        "evaluate",
        str(run_dir),
        "--judge-model",
        "fake-judge",
        "--no-publish",
        "--skip-calibration",
        "--confirm",
    )


def test_a_ctrl_c_while_training_creates_its_trainer_says_the_run_may_remain(
    loops: FakeLoops, dataset: Path, tmp_path: Path
) -> None:
    loops.lost["trainer"] = KeyboardInterrupt()
    result = train(dataset, tmp_path / "run", "--confirm")
    assert result.exit_code == 1, result.stderr
    assert "Error [interrupted]: training stopped at once" in result.stderr
    assert warned(result, left("run", "session1"))


def test_a_ctrl_c_while_train_evaluate_creates_a_sampler_says_it_may_remain(
    loops: FakeLoops, dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    loops.lost["sampler"] = KeyboardInterrupt()
    result = train(
        dataset,
        tmp_path / "run",
        "--evaluate",
        "--eval-judge-model",
        "fake-judge",
        "--eval-skip-calibration",
        "--confirm",
    )
    assert result.exit_code == 1, result.stderr
    assert output(result)["status"] == "completed"
    assert warned(result, left("sampler", "session1"))


def test_a_ctrl_c_while_evaluate_creates_a_sampler_says_it_may_remain(
    loops: FakeLoops, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    loops.lost["sampler"] = KeyboardInterrupt()
    result = evaluate(run_dir)
    assert result.exit_code == 1, result.stderr
    assert "Error [interrupted]: evaluation stopped at once" in result.stderr
    assert warned(result, left("sampler", "session2"))


def test_a_ctrl_c_while_evaluate_creates_a_sampler_wins_over_a_failed_deactivation(
    loops: FakeLoops, routes: dict[str, Answer], run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    loops.lost["sampler"] = KeyboardInterrupt()
    routes["POST /v1/loops/runs/run2/deactivate"] = (500, {})
    result = evaluate(run_dir)
    assert result.exit_code == 1, result.stderr
    assert "Error [interrupted]: evaluation stopped at once" in result.stderr
    assert warned(result, left("sampler", "session2"))
    assert warned(result, "could not deactivate Baseten run run2")


def test_an_unexpected_error_while_evaluate_creates_a_sampler_says_it_may_remain(
    loops: FakeLoops, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    loops.lost["sampler"] = RuntimeError("the connection broke")
    result = evaluate(run_dir)
    assert result.exit_code == 1, result.stderr
    assert "Error [unexpected]: RuntimeError: the connection broke" in result.stderr
    assert warned(result, left("sampler", "session2"))


def test_a_ctrl_c_while_benchmark_creates_a_sampler_says_it_may_remain(
    loops: FakeLoops, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    monkeypatch.setenv("BASETEN_API_KEY", KEY.get_secret_value())
    loops.lost["sampler"] = KeyboardInterrupt()
    result = invoke(
        "benchmark",
        str(dataset),
        "--models",
        "qwen3p5-9b",
        "--judge-model",
        "fake-judge",
        "--skip-calibration",
        "--confirm",
    )
    assert result.exit_code == 1, result.stderr
    assert "Error [interrupted]: stopped by Ctrl-C" in result.stderr
    assert warned(result, left("sampler", "session1"))


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
    assert f"{undeploy_hint(run_dir)} --confirm" not in result.stderr
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
    interrupted = invoke("deploy", str(run_dir), "--confirm")
    assert interrupted.exit_code == 1
    assert "Error [interrupted]: stopped by Ctrl-C\n" in interrupted.stderr
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
    interrupted = invoke("deploy", str(run_dir), "--confirm")
    assert interrupted.exit_code == 1
    assert "Error [interrupted]: stopped by Ctrl-C\n" in interrupted.stderr
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


def test_a_provider_that_does_not_deploy_is_refused_before_the_preview(
    monkeypatch: pytest.MonkeyPatch, loops: FakeLoops, run_dir: Path
) -> None:
    refusing = dataclasses.replace(providers._REGISTRY["baseten"], deploys=False)
    monkeypatch.setitem(providers._REGISTRY, "baseten", refusing)
    preview = invoke("deploy", str(run_dir))
    assert preview.exit_code == 2
    assert preview.stdout == ""
    assert "deploying on Baseten is not available in this version" in preview.stderr
    assert loops.models == {}
