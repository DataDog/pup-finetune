# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`deploy` and `undeploy` through the Typer app: previews, the receipt and its failures.

The development loop runs on the offline fake provider. The Fireworks paths train on the
in-memory Fireworks (as ``test_train_fireworks.py`` does) and deploy on the fake SDK client,
the fake chat endpoint and a fake clock, so nothing reaches the network or waits.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from pup_finetune import providers
from pup_finetune.cli import app
from pup_finetune.cli import deploy as deploy_module
from pup_finetune.cli import train as train_module
from pup_finetune.deployment import deploy_target
from pup_finetune.errors import StateError
from pup_finetune.providers import get_provider as real_get_provider
from pup_finetune.providers.base import Provider
from pup_finetune.providers.fireworks.deploy import FireworksDeployer
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runs import RunDir, model_id_for_run
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from tests.fakes.chat_api import FakeChatApi
from tests.fakes.fireworks import ACCOUNT, FakeFireworksTrainingClient, FakeSdkError
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.fireworks_deploy import SHAPE, FakeClock, FakeSdkFireworks, fake_deployer
from tests.fakes.pushed import mark_pushed
from tests.fakes.tokenizer import FakeChatTokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
RUN_ID = "ft-20261006123045-f1e2d3"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def rerun(command: str) -> Result:
    words = shlex.split(command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return mark_pushed(path)


def train(dataset: Path, run_dir: Path, provider: str = "fake", model: str = "fake-model") -> None:
    result = invoke(
        "train",
        "--provider",
        provider,
        "--model",
        model,
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        "--run-id",
        RUN_ID,
        "--max-epochs",
        "2",
        "--batch-size",
        "2",
        "--confirm",
    )
    assert result.exit_code == 0, result.stderr


@pytest.fixture
def fake_run(dataset: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    train(dataset, run_dir)
    return run_dir


# --- the development loop on the fake provider ---------------------------------------------


def test_deploy_previews_then_deploys_and_undeploys(fake_run: Path) -> None:
    preview = invoke("deploy", str(fake_run))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    undeploy = f"pup finetune undeploy {fake_run}"
    assert data["preview"]["undeploy_command"] == undeploy
    # The offline fake bills nothing, and says so as a training preview does.
    assert data["preview"]["billing"] == (
        f"the fake provider is offline and bills nothing: its deployment is only a record in "
        f"memory: run `{undeploy}`"
    )
    assert data["preview"]["data_recipients"] == [
        "the deployment request and one fixed smoke-test prompt (no rows) -> fake"
    ]
    assert data["preview"]["warnings"] == []
    card = preview.stderr.splitlines()
    assert (
        "  sends     the deployment request and one fixed smoke-test prompt (no rows) to fake"
        in card
    )
    assert "  cost      nothing: the fake provider runs offline" in card
    assert data["preview"]["deployment_id"] == model_id_for_run(RUN_ID)
    assert data["preview"]["replicas"] == {"min": 0, "max": 1}
    assert data["next_command"] == f"pup finetune deploy {fake_run} --confirm"
    assert not RunDir(fake_run).deployment_path.exists()

    deployed = rerun(data["next_command"])
    assert deployed.exit_code == 0, deployed.stderr
    result = output(deployed)
    assert result["state"] == "READY"
    assert result["billing"].startswith("the fake provider is offline and bills nothing")
    # Each fact once: how to call it is in usage, what the create sent in the receipt file.
    assert not {"inference_model", "url", "api_key_env", "request", "model_id"} & set(result)
    assert result["smoke_test"]["ok"] is True
    # Deleting what was just made is never the next command; how to call it is in usage.
    assert result["next_command"] is None
    assert result["undeploy_command"] == undeploy
    assert result["usage"] == {
        "base_url": "http://fake-inference.invalid/v1",
        "model": model_id_for_run(RUN_ID),
        "api_key_env": None,
        "example": "curl http://fake-inference.invalid/v1/chat/completions -H "
        "'Content-Type: application/json' -d "
        f'\'{{"model": "{model_id_for_run(RUN_ID)}", "messages": [{{"role": '
        '"user", "content": "Hello"}]}\'',
    }
    receipt = RunDir(fake_run).read_deployment()
    assert receipt is not None
    assert receipt["deployment_id"] == model_id_for_run(RUN_ID)
    assert receipt["deleted_at"] is None
    assert receipt["created_at"] and receipt["ready_at"]
    assert oct(RunDir(fake_run).deployment_path.stat().st_mode & 0o777) == "0o600"

    again = invoke("deploy", str(fake_run), "--confirm")
    assert again.exit_code == 2
    assert "Error [wrong_phase]: " in again.stderr
    assert "already has the deployment" in again.stderr

    undeploy_preview = invoke("undeploy", str(fake_run))
    assert undeploy_preview.exit_code == 2
    shown = output(undeploy_preview)["preview"]
    assert shown["deployment_id"] == model_id_for_run(RUN_ID)
    assert shown["data_recipients"] == ["the delete request -> fake"]
    assert shown["warnings"] == []

    # The undeploy command is the preview; its next_command adds --confirm.
    assert rerun(undeploy).exit_code == 2
    removed = rerun(output(undeploy_preview)["next_command"])
    assert removed.exit_code == 0, removed.stderr
    removed_data = output(removed)
    assert removed_data["state"] == "DELETED"
    assert removed_data["deleted_at"]
    assert removed_data["model_kept"] == receipt["model"]
    assert removed_data["next_command"] is None
    assert removed_data["next_steps"][0]["command"] == f"pup finetune deploy {fake_run}"
    assert removed_data["already_deleted"] is False
    assert set(removed_data) == {
        "deployment_id",
        "state",
        "deleted_at",
        "already_deleted",
        "model_kept",
        "receipt_file",
        "next_command",
        "next_steps",
    }
    # Undeploying again changes nothing and needs no confirmation.
    for args in ([], ["--confirm"]):
        twice = invoke("undeploy", str(fake_run), *args)
        assert twice.exit_code == 0, twice.stderr
        again = output(twice)
        assert again["deleted_at"] == removed_data["deleted_at"]
        assert again["already_deleted"] is True
        assert again["message"] == f"already deleted at {again['deleted_at']}; nothing to do"
    # Once undeployed, the run can be deployed again.
    assert invoke("deploy", str(fake_run), "--confirm").exit_code == 0


def test_undeploying_a_run_with_no_deployment_hints_only_what_can_run(fake_run: Path) -> None:
    done = invoke("undeploy", str(fake_run))
    assert done.exit_code == 2
    assert f"Hint: nothing to undeploy; deploy it with `pup finetune deploy {fake_run}`" in (
        done.stderr
    )
    # A run still training cannot be deployed: the hint never names deploy.
    state = fake_run / "state.json"
    state.write_text(json.dumps({**json.loads(state.read_text()), "phase": "training"}))
    training = invoke("undeploy", str(fake_run))
    assert training.exit_code == 2
    assert "Error [wrong_phase]: the run in " in training.stderr
    assert "deploy it" not in training.stderr
    assert f"`pup finetune status {fake_run}`" in training.stderr


def test_the_preview_repeats_the_flags(fake_run: Path) -> None:
    preview = invoke(
        "deploy",
        str(fake_run),
        "--deployment-id",
        "my-dep",
        "--shape",
        "s1",
        "--timeout-minutes",
        "5",
    )
    assert preview.exit_code == 2
    data = output(preview)
    assert data["preview"]["deployment_id"] == "my-dep"
    assert data["preview"]["shape"] == "s1"
    assert "5 minutes" in data["preview"]["action"]
    assert data["next_command"] == (
        f"pup finetune deploy {fake_run} --deployment-id my-dep --shape s1 "
        "--timeout-minutes 5 --confirm"
    )


def test_an_invalid_deployment_id_is_refused(fake_run: Path) -> None:
    result = invoke("deploy", str(fake_run), "--deployment-id", "Not_Valid")
    assert result.exit_code == 2
    assert "not a valid deployment id" in result.stderr


def _no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command reaches the provider (the only way to a remote call)."""

    def refuse(*_args: Any, **_kwargs: Any) -> Provider:
        raise AssertionError("a read-only preview must not reach the provider")

    monkeypatch.setattr(deploy_module, "get_provider", refuse)


def test_deploy_is_refused_read_only_but_previews(
    fake_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    _no_provider(monkeypatch)
    refused = invoke("deploy", str(fake_run), "--confirm")
    assert refused.exit_code == 2
    assert refused.stdout == ""
    assert "PUP_READ_ONLY" in refused.stderr
    assert "pup --read-only" in refused.stderr
    preview = invoke("deploy", str(fake_run))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    assert data["next_command"] is None
    assert data["preview"]["read_only"] == "--confirm is refused while PUP_READ_ONLY is set"
    assert data["preview"]["deployment_id"] == model_id_for_run(RUN_ID)
    assert "Next:" not in preview.stderr
    assert not RunDir(fake_run).deployment_path.exists()


def test_undeploy_is_refused_read_only_but_previews(
    fake_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke("deploy", str(fake_run), "--confirm").exit_code == 0
    receipt = RunDir(fake_run).deployment_path.read_text()
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    _no_provider(monkeypatch)
    refused = invoke("undeploy", str(fake_run), "--confirm")
    assert refused.exit_code == 2
    assert "PUP_READ_ONLY" in refused.stderr
    preview = invoke("undeploy", str(fake_run))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    assert data["next_command"] is None
    assert data["preview"]["read_only"] == "--confirm is refused while PUP_READ_ONLY is set"
    assert data["preview"]["action"].startswith("delete the deployment")
    assert RunDir(fake_run).deployment_path.read_text() == receipt


def test_auto_approve_does_not_confirm(fake_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_AUTO_APPROVE", "true")
    assert invoke("deploy", str(fake_run)).exit_code == 2
    assert not RunDir(fake_run).deployment_path.exists()


def test_a_run_that_has_not_completed_is_refused(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "crashed"
    result = invoke(
        "train",
        "--provider",
        "fake",
        "--model",
        "fake-model",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        "--provider-option",
        "fail_at_step=2",
        "--confirm",
    )
    assert result.exit_code == 1
    deployed = invoke("deploy", str(run_dir))
    assert deployed.exit_code == 2
    assert "Error [wrong_phase]: " in deployed.stderr
    assert "not completed" in deployed.stderr


def test_undeploy_without_a_deployment_is_an_error(fake_run: Path) -> None:
    result = invoke("undeploy", str(fake_run), "--confirm")
    assert result.exit_code == 2
    assert "Error [wrong_phase]: " in result.stderr
    assert "has no deployment" in result.stderr


def test_the_help_states_the_agent_rule() -> None:
    result = invoke("deploy", "--help")
    assert result.exit_code == 0
    text = " ".join(result.stdout.split())
    assert "explicit authorization" in text


# --- Fireworks -----------------------------------------------------------------------------


class FireworksWorld:
    """The in-memory Fireworks: training, deployments, the chat endpoint and the clock."""

    def __init__(self) -> None:
        self.training = FakeFireworksTrainingClient()
        self.sdk = FakeSdkFireworks()
        self.chat = FakeChatApi(api_key=API_KEY)
        self.clock = FakeClock()
        self.smoke_timeout_s = 600.0
        self.account = ACCOUNT
        """The account the key resolves to, for the commands run from now on."""

    def provider(self, runtime: Runtime) -> FireworksProvider:
        def deployer(key: SecretStr) -> FireworksDeployer:
            build = fake_deployer(
                self.sdk,
                self.chat,
                self.clock,
                smoke_timeout_s=self.smoke_timeout_s,
                account_id=self.account,
            )
            return build(key)

        return FireworksProvider(
            runtime,
            env={"FIREWORKS_API_KEY": API_KEY},
            transport=FakeFireworksApi().transport,
            tokenizer=FakeChatTokenizer(),
            training_client=lambda _key: self.training,
            deployer=deployer,
        )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FireworksWorld:
    world = FireworksWorld()

    def get_provider(name: str, runtime: Runtime) -> Provider:
        if name == "fireworks":
            return world.provider(runtime)
        return real_get_provider(name, runtime)

    monkeypatch.setattr(train_module, "get_provider", get_provider)
    monkeypatch.setattr(inputs_module, "get_provider", get_provider)
    monkeypatch.setattr(deploy_module, "get_provider", get_provider)
    return world


@pytest.fixture
def fireworks_run(world: FireworksWorld, dataset: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "fw-run"
    train(dataset, run_dir, provider="fireworks", model="qwen3p8-27b")
    return run_dir


def test_fireworks_deploy_reports_the_inference_model(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    model_id = model_id_for_run(RUN_ID)
    assert data["model"] == f"accounts/{ACCOUNT}/models/{model_id}"
    assert data["usage"]["model"] == f"accounts/{ACCOUNT}/deployments/{model_id}"
    assert data["usage"]["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert data["smoke_test"]["ok"] is True
    assert data["next_command"] is None
    assert data["undeploy_command"] == f"pup finetune undeploy {fireworks_run}"
    inference = data["usage"]["model"]
    # The key is named as a variable for the shell to expand, never as its value.
    assert data["usage"] == {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": inference,
        "api_key_env": "FIREWORKS_API_KEY",
        "example": "curl https://api.fireworks.ai/inference/v1/chat/completions "
        '-H "Authorization: Bearer $FIREWORKS_API_KEY" '
        "-H 'Content-Type: application/json' -d "
        f'\'{{"model": "{inference}", "messages": [{{"role": "user", '
        '"content": "Hello"}]}\'',
    }
    assert RunDir(fireworks_run).read_deployment()["api_key_env"] == "FIREWORKS_API_KEY"  # type: ignore[index]
    (create,) = [kwargs for name, kwargs in world.sdk.calls if name == "create"]
    assert create["base_model"] == data["model"]
    assert create["enable_hot_load"] is False
    assert (create["min_replica_count"], create["max_replica_count"]) == (0, 1)
    assert world.chat.bodies[-1]["model"] == inference
    assert API_KEY not in result.stdout + result.stderr

    removed = invoke("undeploy", str(fireworks_run), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert world.sdk.deleted == [model_id]
    assert output(removed)["already_deleted"] is False


def test_a_read_only_fireworks_deploy_preview_makes_no_call(
    world: FireworksWorld, fireworks_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = list(world.sdk.calls)
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    preview = invoke("deploy", str(fireworks_run))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    assert data["next_command"] is None
    assert data["preview"]["read_only"] == "--confirm is refused while PUP_READ_ONLY is set"
    # No shape matching, no create: the preview is built from the run alone.
    assert world.sdk.calls == calls
    assert not RunDir(fireworks_run).deployment_path.exists()


def test_a_missing_sdk_fails_the_deploy_preview(
    world: FireworksWorld, fireworks_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(providers, "installed", lambda module: module != "fireworks")
    preview = invoke("deploy", str(fireworks_run))
    assert preview.exit_code == 2
    assert preview.stdout == ""
    assert (
        "deploying on Fireworks needs the Fireworks Training SDK, which is not installed"
    ) in preview.stderr
    assert not RunDir(fireworks_run).deployment_path.exists()


def test_the_fireworks_deploy_preview_names_its_billing_and_recipients(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    preview = invoke("deploy", str(fireworks_run))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)["preview"]
    undeploy = f"pup finetune undeploy {fireworks_run}"
    assert "bills per GPU-second" in data["billing"]
    assert undeploy in data["billing"]
    assert data["data_recipients"] == [
        "the deployment request and one fixed smoke-test prompt (no rows) -> fireworks"
    ]
    lines = preview.stderr.splitlines()
    assert any(line.startswith("  sends     the deployment request") for line in lines)
    assert any(line.startswith("  cost      the deployment bills per GPU-second") for line in lines)


def test_a_missing_sdk_fails_the_undeploy_preview(
    world: FireworksWorld, fireworks_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 0
    monkeypatch.setattr(providers, "installed", lambda module: module != "fireworks")
    preview = invoke("undeploy", str(fireworks_run))
    assert preview.exit_code == 2
    assert preview.stdout == ""
    assert (
        "deploying on Fireworks needs the Fireworks Training SDK, which is not installed"
    ) in preview.stderr
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None and receipt["deleted_at"] is None


def test_a_timed_out_deploy_leaves_a_receipt(world: FireworksWorld, fireworks_run: Path) -> None:
    world.sdk.states = ["CREATING"]
    result = invoke("deploy", str(fireworks_run), "--timeout-minutes", "1", "--confirm")
    assert result.exit_code == 1
    assert "not ready after 1 minutes" in result.stderr
    assert f"pup finetune undeploy {fireworks_run}" in result.stderr
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "CREATING"
    assert receipt["error"]["type"] == "DeploymentNotReady"
    assert receipt["ready_at"] is None
    # The receipt lets it be torn down.
    assert invoke("undeploy", str(fireworks_run), "--confirm").exit_code == 0
    assert world.sdk.deleted == [model_id_for_run(RUN_ID)]


def test_a_failed_deployment_is_recorded(world: FireworksWorld, fireworks_run: Path) -> None:
    world.sdk.states = ["FAILED"]
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 1
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None and receipt["state"] == "FAILED"


def test_a_failed_smoke_test_keeps_the_receipt_and_exits_1(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.smoke_timeout_s = 30
    world.chat.script += [httpx.Response(503)] * 10
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["smoke_test"]["ok"] is False
    assert "undeploy" in data["hint"]
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "READY"
    assert receipt["smoke_test"]["ok"] is False
    assert receipt["deleted_at"] is None


def test_undeploying_a_deployment_already_gone_succeeds(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 0
    world.sdk.records.clear()  # deleted elsewhere
    result = invoke("undeploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 0, result.stderr
    assert output(result)["already_deleted"] is True


def test_no_secret_reaches_the_receipt(world: FireworksWorld, fireworks_run: Path) -> None:
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 0
    for path in fireworks_run.iterdir():
        if path.is_file():
            assert API_KEY not in path.read_text(encoding="utf-8", errors="replace"), path
    assert "FIREWORKS_API_KEY" not in os.environ or os.environ["FIREWORKS_API_KEY"] != API_KEY


def test_the_receipt_is_on_disk_before_the_wait(world: FireworksWorld, fireworks_run: Path) -> None:
    """A Ctrl-C while waiting is not a FinetuneError: only the early receipt records it."""
    world.sdk.states = ["CREATING"]
    seen: list[str | None] = []

    def interrupt(seconds: float) -> None:
        receipt = RunDir(fireworks_run).read_deployment()
        seen.append(receipt["state"] if receipt else None)
        raise KeyboardInterrupt

    world.clock.sleep = interrupt  # type: ignore[method-assign]
    interrupted = invoke("deploy", str(fireworks_run), "--confirm")
    assert interrupted.exit_code == 1  # Ctrl-C is Error [interrupted], exit 1
    assert "Error [interrupted]: stopped by Ctrl-C\n" in interrupted.stderr
    assert seen == ["CREATING"]
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "CREATING"
    assert (
        receipt["deployment_name"] == f"accounts/{ACCOUNT}/deployments/{model_id_for_run(RUN_ID)}"
    )
    removed = invoke("undeploy", str(fireworks_run), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert world.sdk.deleted == [model_id_for_run(RUN_ID)]


def test_a_ctrl_c_while_creating_leaves_a_receipt_to_undeploy(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.sdk.lost["create"] = KeyboardInterrupt()
    interrupted = invoke("deploy", str(fireworks_run), "--confirm")
    assert interrupted.exit_code == 1  # Ctrl-C is Error [interrupted], exit 1
    assert "Error [interrupted]: stopped by Ctrl-C\n" in interrupted.stderr
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "REQUESTED"
    assert receipt["deployment_id"] == model_id_for_run(RUN_ID)
    # What the create sent, to tell the deployment it made from another one under the id.
    (create,) = [kwargs for name, kwargs in world.sdk.calls if name == "create"]
    request = receipt["request"]
    assert request["base_model"] == create["base_model"] == receipt["model"]
    assert request["display_name"] == create["display_name"] == f"pup-finetune {RUN_ID}"
    assert request["shape"] == create["deployment_shape"]
    # Deploying again is refused until it is undeployed, which finds and deletes it.
    again = invoke("deploy", str(fireworks_run), "--confirm")
    assert again.exit_code == 2
    assert "Error [wrong_phase]: " in again.stderr
    assert "already has the deployment" in again.stderr
    removed = invoke("undeploy", str(fireworks_run), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is False
    assert world.sdk.deleted == [model_id_for_run(RUN_ID)]


def test_a_receipt_left_while_creating_names_its_account(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.sdk.lost["create"] = KeyboardInterrupt()
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 1  # Ctrl-C
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "REQUESTED"
    deployment_id = model_id_for_run(RUN_ID)
    assert receipt["deployment_name"] == f"accounts/{ACCOUNT}/deployments/{deployment_id}"
    # Under another account, a deployment of the same id there is never deleted, nor a 404
    # there taken for this one being gone.
    world.account = "other-team"
    refused = invoke("undeploy", str(fireworks_run), "--confirm")
    assert refused.exit_code == 2, refused.stderr
    assert f"FIREWORKS_ACCOUNT_ID={ACCOUNT}" in refused.stderr
    stored = RunDir(fireworks_run).read_deployment() or {}
    assert stored["deleted_at"] is None
    assert world.sdk.deleted == []


def test_a_failed_shape_lookup_leaves_no_receipt(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.sdk.fail["match_for_model"] = FakeSdkError("service unavailable", status_code=503)
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 1
    assert not RunDir(fireworks_run).deployment_path.exists()
    assert "create" not in world.sdk.call_names()
    assert "undeploy" not in result.stderr


def test_a_failed_shape_lookup_keeps_the_earlier_receipt(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 0
    assert invoke("undeploy", str(fireworks_run), "--confirm").exit_code == 0
    before = RunDir(fireworks_run).read_deployment()
    world.sdk.fail["match_for_model"] = FakeSdkError("service unavailable", status_code=503)
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 1
    assert RunDir(fireworks_run).read_deployment() == before


def foreign_deployment(world: FireworksWorld, deployment_id: str) -> None:
    """Another model's deployment under ``deployment_id``, made outside this run."""
    world.sdk.records[deployment_id] = {
        "name": f"accounts/{ACCOUNT}/deployments/{deployment_id}",
        "state": "READY",
        "base_model": f"accounts/{ACCOUNT}/models/someone-elses-model",
        "deployment_shape": f"{SHAPE}/versions/v2",
        "display_name": "someone else's deployment",
        "min_replica_count": 0,
        "max_replica_count": 1,
        "status": None,
    }


def test_undeploying_a_requested_deployment_never_deletes_another_one(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    deployment_id = "shared-name"
    world.sdk.fail["create"] = KeyboardInterrupt()  # before the create took effect
    interrupted = invoke(
        "deploy", str(fireworks_run), "--deployment-id", deployment_id, "--confirm"
    )
    assert interrupted.exit_code == 1  # Ctrl-C: Error [interrupted]
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None and receipt["state"] == "REQUESTED"
    # The id is someone else's deployment, which the create would have been refused for.
    foreign_deployment(world, deployment_id)
    preview = invoke("undeploy", str(fireworks_run))
    assert "read back the deployment" in output(preview)["preview"]["action"]
    refused = invoke("undeploy", str(fireworks_run), "--confirm")
    assert refused.exit_code == 1, refused.stderr
    assert "is not the one run" in refused.stderr
    assert str(RunDir(fireworks_run).deployment_path) in refused.stderr
    assert world.sdk.deleted == []
    assert "delete" not in world.sdk.call_names()
    assert deployment_id in world.sdk.records
    assert (RunDir(fireworks_run).read_deployment() or {})["deleted_at"] is None


def test_undeploying_a_requested_deployment_that_was_never_made_deletes_nothing(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.sdk.fail["create"] = KeyboardInterrupt()  # before the create took effect
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 1  # Ctrl-C
    removed = invoke("undeploy", str(fireworks_run), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert output(removed)["already_deleted"] is True
    assert "delete" not in world.sdk.call_names()
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None and receipt["deleted_at"] is not None
    # The run can be deployed again.
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 0


def test_a_lost_create_answer_is_adopted(world: FireworksWorld, fireworks_run: Path) -> None:
    world.sdk.lost["create"] = FakeSdkError("gateway timeout", status_code=504)
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 0, result.stderr
    assert output(result)["state"] == "READY"
    assert world.sdk.call_names().count("create") == 1
    assert world.sdk.client_options == [{"max_retries": 0}]


def test_a_create_answered_without_a_name_keeps_the_receipt_to_undeploy(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.sdk.create_without_name = True
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 1
    assert "undeploy" in result.stderr
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "REQUESTED"
    deployment_id = model_id_for_run(RUN_ID)
    assert receipt["deployment_name"] == f"accounts/{ACCOUNT}/deployments/{deployment_id}"
    assert "could not be read" in receipt["error"]["message"]
    removed = invoke("undeploy", str(fireworks_run), "--confirm")
    assert removed.exit_code == 0, removed.stderr
    assert world.sdk.deleted == [deployment_id]


def test_a_refused_create_leaves_no_receipt(world: FireworksWorld, fireworks_run: Path) -> None:
    world.sdk.fail["create"] = FakeSdkError("invalid deployment shape", status_code=400)
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 1
    assert not RunDir(fireworks_run).deployment_path.exists()
    assert "undeploy" not in result.stderr


def test_a_refused_redeploy_keeps_the_earlier_receipt(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 0
    assert invoke("undeploy", str(fireworks_run), "--confirm").exit_code == 0
    before = RunDir(fireworks_run).read_deployment()
    world.sdk.fail["create"] = FakeSdkError("invalid deployment shape", status_code=400)
    assert invoke("deploy", str(fireworks_run), "--confirm").exit_code == 1
    assert RunDir(fireworks_run).read_deployment() == before


def test_a_smoke_test_that_raises_is_recorded_with_an_undeploy_hint(
    world: FireworksWorld, fireworks_run: Path
) -> None:
    world.chat.api_key = "fw-synthetic-other-key"  # the inference call is refused: 401
    result = invoke("deploy", str(fireworks_run), "--confirm")
    assert result.exit_code == 1
    assert f"pup finetune undeploy {fireworks_run}" in result.stderr
    receipt = RunDir(fireworks_run).read_deployment()
    assert receipt is not None
    assert receipt["state"] == "READY"
    assert receipt["smoke_test"]["ok"] is False
    assert receipt["error"]["type"] == "ProviderAuthError"
    assert API_KEY not in json.dumps(receipt)


# --- what a run deploys --------------------------------------------------------------------


def _edit_result(run_dir: Path, **changes: Any) -> None:
    run = RunDir(run_dir)
    result = run.read_result()
    assert result is not None
    run.write_result({**result, **changes})


def test_a_run_without_a_promoted_model_deploys_its_deployable_checkpoint(
    fake_run: Path,
) -> None:
    checkpoint = {"name": "fake-ckpt-7", "uri": "fake://checkpoints/fake-ckpt-7"}
    _edit_result(fake_run, promoted_model=None, deployable_checkpoint=checkpoint)
    target = deploy_target(RunDir(fake_run))
    assert target.source == "deployable_checkpoint"
    assert target.model.uri == checkpoint["uri"]
    preview = output(invoke("deploy", str(fake_run)))["preview"]
    assert preview["model"] == checkpoint["uri"]
    assert preview["model_source"] == "deployable_checkpoint"


def test_a_run_with_neither_model_nor_checkpoint_is_refused(fake_run: Path) -> None:
    _edit_result(fake_run, promoted_model=None, deployable_checkpoint=None)
    with pytest.raises(StateError, match="neither a promoted model nor a deployable"):
        deploy_target(RunDir(fake_run))
    assert invoke("deploy", str(fake_run)).exit_code == 2


def test_a_cancelled_run_with_a_promoted_model_deploys(fake_run: Path) -> None:
    run = RunDir(fake_run)
    state = json.loads(run.state_path.read_text())
    state["phase"] = "cancelled"
    run.state_path.write_text(json.dumps(state))
    assert run.load_state().phase == "cancelled"
    target = deploy_target(run)
    assert target.source == "promoted_model"
    assert invoke("deploy", str(fake_run), "--confirm").exit_code == 0
