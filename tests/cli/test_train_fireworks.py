# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train --provider fireworks` and `resume` end to end, on the in-memory Fireworks.

The provider is the real :class:`FireworksProvider`, with the in-memory Training API, the
fake REST API (for preflight) and the fake tokenizer injected through its constructor. Each
command gets a new provider, as a new process would, while the in-memory Fireworks (its
sessions and checkpoints) persists between them, as the real service does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import providers
from pup_finetune.cli import app
from pup_finetune.cli import train as train_module
from pup_finetune.next_steps import command
from pup_finetune.providers.base import Provider
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runs import RunDir, model_id_for_run, resume_command
from pup_finetune.runtime import Runtime
from pup_finetune.training import inputs as inputs_module
from tests.fakes.fireworks import ACCOUNT, FakeFireworksTrainingClient
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.pushed import mark_pushed
from tests.fakes.tokenizer import FakeChatTokenizer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
RUN_ID = "ft-20261006123045-f1e2d3"

runner = CliRunner()


class CrashingFireworks(FakeFireworksTrainingClient):
    """The in-memory Fireworks, whose ``crash_at``-th forward-backward pass fails."""

    def __init__(self, crash_at: int | None = None) -> None:
        super().__init__()
        self.crash_at = crash_at
        self.passes = 0

    def _pending_failure(self, operation: str, what: str) -> Exception | None:
        if operation == "forward_backward":
            self.passes += 1
            if self.passes == self.crash_at:
                self.fail("forward_backward", "server")
        return super()._pending_failure(operation, what)


@pytest.fixture
def fw() -> CrashingFireworks:
    return CrashingFireworks()


@pytest.fixture(autouse=True)
def _fireworks(monkeypatch: pytest.MonkeyPatch, fw: CrashingFireworks) -> None:
    def get_provider(name: str, runtime: Runtime) -> Provider:
        assert name == "fireworks"
        return FireworksProvider(
            runtime,
            env={"FIREWORKS_API_KEY": API_KEY},
            transport=FakeFireworksApi().transport,
            tokenizer=FakeChatTokenizer(),
            training_client=lambda _key: fw,
        )

    monkeypatch.setattr(train_module, "get_provider", get_provider)
    monkeypatch.setattr(inputs_module, "get_provider", get_provider)


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
        "fireworks",
        "--model",
        "qwen3p8-27b",
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
        *extra,
    )


def _no_session_left(fw: FakeFireworksTrainingClient) -> bool:
    return not any(state.alive for state in fw.sessions.values())


def test_the_preview_without_a_key_is_a_usage_error(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fw: CrashingFireworks
) -> None:
    def keyless(name: str, runtime: Runtime) -> Provider:
        return FireworksProvider(
            runtime,
            env={},
            transport=FakeFireworksApi().transport,
            tokenizer=FakeChatTokenizer(),
            training_client=lambda _key: fw,
        )

    monkeypatch.setattr(train_module, "get_provider", keyless)

    result = train(dataset, tmp_path / "run")

    assert result.exit_code == 2
    assert result.stderr.startswith("Error [usage]: no fireworks API key found\n")


def test_train_on_fireworks_runs_end_to_end(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks
) -> None:
    run_dir = tmp_path / "run"
    preview = train(dataset, run_dir)
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["data_destination"] == "training data is sent to fireworks"
    assert fw.calls == []  # a preview opens no session

    result = train(dataset, run_dir, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "completed"
    model_id = model_id_for_run(RUN_ID)
    expected = {
        "provider": "fireworks",
        "model_id": model_id,
        "uri": f"accounts/{ACCOUNT}/models/{model_id}",
    }
    assert data["promoted_model"] == {key: expected[key] for key in ("model_id", "uri")}
    on_disk = RunDir(run_dir).read_result() or {}
    assert on_disk["promoted_model"] == expected
    assert on_disk["deployable_checkpoint"]["kind"] == "sampler"
    assert len(on_disk["epochs"]) == 2
    assert all(epoch["val_loss"] is not None for epoch in on_disk["epochs"])
    assert model_id in fw.promoted
    assert _no_session_left(fw)
    assert API_KEY not in result.stdout + result.stderr


def test_each_step_reaches_fireworks_at_its_scheduled_learning_rate(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks
) -> None:
    run_dir = tmp_path / "run"
    total = output(train(dataset, run_dir, "--lr-schedule", "linear"))["preview"]["max_steps"]
    result = train(dataset, run_dir, "--lr-schedule", "linear", "--confirm")
    assert result.exit_code == 0, result.stderr
    spec = RunDir(run_dir).spec()
    (run,) = fw.services[0].runs
    assert [adam.learning_rate for adam in run.adam] == [
        spec.learning_rate_at(step, total) for step in range(1, total + 1)
    ]
    assert run.adam[-1].learning_rate < run.adam[0].learning_rate


def test_read_only_allows_the_preview_but_not_training(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    run_dir = tmp_path / "run"
    preview = train(dataset, run_dir)
    assert preview.exit_code == 2
    assert output(preview)["next_command"] is None
    assert "refused   --confirm is refused while PUP_READ_ONLY is set" in preview.stderr
    # A paid provider's card says who bills it and what caps the spend.
    assert (
        "  bills     Fireworks bills the serverless training session at its own rates "
        "(pup-finetune does not know the rates)\n"
    ) in preview.stderr
    assert "  cost      no spend cap (add --max-spend-usd with --hourly-rate-usd)\n" in (
        preview.stderr
    )

    confirmed = train(dataset, run_dir, "--confirm")
    assert confirmed.exit_code == 2
    assert confirmed.stderr.startswith(
        "Error [read_only]: training on fireworks is a paid or remote run and PUP_READ_ONLY is set"
    )
    assert confirmed.stdout == ""
    assert not run_dir.exists()
    assert fw.calls == []


def test_read_only_refuses_resuming_a_fireworks_run(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    fw.crash_at = 6
    assert train(dataset, run_dir, "--confirm").exit_code == 1
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    preview = invoke("resume", str(run_dir))
    assert output(preview)["next_command"] is None
    # resume has no spend flag: the cap is the run's, set when it started.
    assert (
        "  cost      no spend cap (a run's cap is set when it starts; this one started without)\n"
    ) in preview.stderr
    assert "--hourly-rate-usd" not in output(preview)["preview"]["billing"]
    result = invoke("resume", str(run_dir), "--confirm")
    assert result.exit_code == 2
    assert "Error [read_only]: training on fireworks is a paid or remote run" in result.stderr


def test_resume_after_a_crash_finishes_and_promotes(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks
) -> None:
    run_dir = tmp_path / "run"
    fw.crash_at = 6  # in the second epoch (4 steps each), after the first epoch's checkpoints
    crashed = train(dataset, run_dir, "--confirm")
    assert crashed.exit_code == 1, crashed.stderr
    assert RunDir(run_dir).load_state().phase == "interrupted"
    assert _no_session_left(fw)
    training = [c for c in fw.checkpoints if c.kind == "training"]
    assert training, "the first epoch saved a training checkpoint to resume from"

    preview = invoke("resume", str(run_dir))
    assert preview.exit_code == 2, preview.stderr
    resumed = invoke("resume", str(run_dir), "--confirm")
    assert resumed.exit_code == 0, resumed.stderr
    data = output(resumed)
    assert data["status"] == "completed"
    assert data["promoted_model"]["model_id"] == model_id_for_run(RUN_ID)
    resumed_from = [
        call[1]
        for call in fw.calls
        if call[0] == "create_training_client_from_state_with_optimizer"
    ]
    assert resumed_from == [training[-1].path]
    assert _no_session_left(fw)


def test_a_run_whose_model_id_another_run_took_offers_no_resume_that_fails_again(
    dataset: Path, tmp_path: Path
) -> None:
    # A preview's pinned command run into two run directories: the second's model id is taken.
    assert train(dataset, tmp_path / "first", "--confirm").exit_code == 0
    second = tmp_path / "second"
    refused = train(dataset, second, "--confirm")
    assert refused.exit_code == 1, refused.stderr
    assert "another run id" in refused.stderr
    assert resume_command(second) not in refused.stderr

    status = invoke("status", str(second))
    assert status.exit_code == 0, status.stderr
    data = output(status)
    assert data["next_command"] is None
    commands = [entry["command"] for entry in data["next_steps"]]
    # A new run (train generates its id) of the same model on the same dataset, as written.
    retrain = command(
        "train", "--provider", "fireworks", "--dataset", dataset, "--model", "qwen3p8-27b"
    )
    assert retrain in commands
    assert resume_command(second) in commands


def test_a_resume_preview_of_a_run_whose_model_id_another_run_took_warns_it_may_be_refused(
    dataset: Path, tmp_path: Path
) -> None:
    assert train(dataset, tmp_path / "first", "--confirm").exit_code == 0
    second = tmp_path / "second"
    assert train(dataset, second, "--confirm").exit_code == 1
    preview = invoke("resume", str(second))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    # The model may have been deleted since, so the resume is still offered.
    assert data["next_command"] == f"{resume_command(second)} --confirm"
    model_id = model_id_for_run(RUN_ID)
    warnings = data["preview"]["warnings"]
    assert any(model_id in w and "refused again" in w for w in warnings), warnings


def test_a_max_seq_len_option_shorter_than_every_row_is_refused_in_the_preview(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks
) -> None:
    run_dir = tmp_path / "run"
    for confirm in ((), ("--confirm",)):
        result = train(dataset, run_dir, "--provider-option", "max_seq_len=16", *confirm)
        assert result.exit_code == 2, result.stderr
        assert "no rows to train on" in result.stderr
        assert "exceeds_context" in result.stderr
    assert fw.sessions == {}
    assert not (run_dir / "run.json").exists()


def test_the_max_seq_len_option_leaves_longer_rows_out_and_survives_a_resume(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks
) -> None:
    run_dir = tmp_path / "run"
    option = ("--provider-option", "max_seq_len=1300")
    preview = train(dataset, run_dir, *option)
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    assert "--provider-option max_seq_len=1300" in data["next_command"]
    assert data["preview"]["profile"]["max_seq_len"] == 1300
    # The fixture's rows render (with the fake tokenizer) to 214..1871 tokens: the 2 train
    # and 2 validation rows over 1300 are left out, the 5 + 3 others kept.
    assert data["preview"]["rejected_by_code"] == {"exceeds_context": 4}
    assert fw.calls == []

    fw.crash_at = 5  # in the second epoch (3 steps each), after the first epoch's checkpoints
    crashed = train(dataset, run_dir, *option, "--confirm")
    assert crashed.exit_code == 1, crashed.stderr
    assert RunDir(run_dir).profile().max_seq_len == 1300
    assert RunDir(run_dir).load_state().phase == "interrupted"

    resumed = invoke("resume", str(run_dir), "--confirm")
    assert resumed.exit_code == 0, resumed.stderr
    assert output(resumed)["status"] == "completed"
    calls = [call[0] for call in fw.calls]
    assert calls.count("create_training_client_from_state_with_optimizer") >= 1
    assert _no_session_left(fw)


# --- what the preview can know fails in the preview ----------------------------------------


def _without(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    """Make the package checks miss ``module``, as if its extra were not installed."""
    monkeypatch.setattr(providers, "installed", lambda name: name != module)


def test_a_missing_training_sdk_fails_the_preview_before_anything_is_created(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without(monkeypatch, "tinker")
    run_dir = tmp_path / "run"
    for args in ((), ("--detach", "--confirm")):
        result = train(dataset, run_dir, *args)
        assert result.exit_code == 2, result.stderr
        assert result.stdout == ""
        assert (
            "Error [usage]: training on Fireworks needs the Fireworks Training SDK, which is not "
            "installed"
        ) in result.stderr
        assert "uv sync --all-extras" in result.stderr
        assert not run_dir.exists()


def test_a_missing_renderer_fails_the_preview(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without(monkeypatch, "transformers")
    result = train(dataset, tmp_path / "run")
    assert result.exit_code == 2
    assert "needs the Hugging Face transformers package, which is not installed" in result.stderr


def test_a_missing_training_sdk_fails_the_resume_preview(
    dataset: Path, tmp_path: Path, fw: CrashingFireworks, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    fw.crash_at = 6
    assert train(dataset, run_dir, "--confirm").exit_code == 1
    _without(monkeypatch, "fireworks")
    result = invoke("resume", str(run_dir))
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "training on Fireworks needs the Fireworks Training SDK" in result.stderr


def test_a_missing_training_sdk_fails_the_evaluate_preview(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    assert train(dataset, run_dir, "--confirm").exit_code == 0
    _without(monkeypatch, "tinker")
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "evaluating on Fireworks needs the Fireworks Training SDK" in result.stderr
