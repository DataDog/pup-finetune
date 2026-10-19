# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What train, status and evaluate print, and the table views pup renders."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pup_finetune import doctor as registry
from pup_finetune import output
from pup_finetune.cli import _common, app
from pup_finetune.cli import doctor as doctor_cli
from pup_finetune.cli._presentation import COMMAND_ORDER
from pup_finetune.cli._summary import status_line
from pup_finetune.cli._views import (
    STATUS_KEYS,
    TABLE_VIEWS,
    evaluation_view,
    status_view,
    worker_state,
    worker_view,
)
from pup_finetune.training.worker import WorkerStatus
from tests.fakes.pup import FakePup

runner = CliRunner()


def worker(**fields: Any) -> WorkerStatus:
    values: dict[str, Any] = {
        "pid": 42,
        "alive": False,
        "heartbeat_age_s": 3.0,
        "stale": False,
        "starting": False,
        "running": False,
    }
    return WorkerStatus(**{**values, **fields})


@pytest.mark.parametrize(
    ("fields", "state"),
    [
        ({"alive": True, "running": True}, "running"),
        ({"alive": True, "starting": True, "running": True}, "starting"),
        ({"alive": True, "stale": True}, "stale"),
        ({"alive": False, "stale": True}, "exited"),
        ({}, "exited"),
    ],
)
def test_a_worker_reads_as_one_state(fields: dict[str, Any], state: str) -> None:
    assert worker_state(worker(**fields)) == state


def test_a_worker_is_shown_only_when_there_is_one() -> None:
    # A run that trained in the foreground and ended has none to speak of.
    assert worker_view(worker(), "completed") is None
    # A run that is training has its process, live or not; a detached run has its log.
    assert worker_view(worker(), "training") == {
        "pid": 42,
        "state": "exited",
        "heartbeat_age_s": 3.0,
        "log": None,
    }
    shown = worker_view(worker(log="run/worker.log"), "completed")
    assert shown is not None
    assert (shown["state"], shown["log"]) == ("exited", "run/worker.log")


def _status(**fields: Any) -> dict[str, Any]:
    data: dict[str, Any] = dict.fromkeys(STATUS_KEYS)
    data.pop("estimated_spend_usd")
    data.pop("watch_timed_out")
    data.update(
        steps_per_epoch=1,
        epochs_completed=2,
        phase="completed",
        worker=worker(),
        resume_mode="train",
        promoted_model={"provider": "fake", "model_id": "ft-1", "uri": "fake://models/ft-1"},
        deployable_checkpoint={"name": "s2s2-ab", "kind": "sampler", "uri": "fake://x"},
    )
    data.update(fields)
    return data


def test_status_prints_resume_mode_only_for_a_resumable_run() -> None:
    assert "resume_mode" not in status_view(_status())
    assert status_view(_status(phase="interrupted"))["resume_mode"] == "train"


def test_status_prints_the_decision_in_order_and_names_its_checkpoint() -> None:
    shown = status_view(_status())
    assert next(iter(shown)) == "phase"
    assert "worker" not in shown
    assert "steps_per_epoch" not in shown
    assert "epochs_completed" not in shown
    assert shown["promoted_model"] == {"model_id": "ft-1", "uri": "fake://models/ft-1"}
    assert shown["deployable_checkpoint"] == "s2s2-ab"
    # Keys that are sometimes null stay.
    assert shown["error"] is None
    assert shown["approx_eta_seconds"] is None


REMOTE = {"backend": "baseten job", "job_id": "j1", "state": "running"}


def test_status_hides_the_worker_while_a_remote_job_trains_the_run() -> None:
    stand_in = worker(pid=None, alive=True, running=True)
    shown = status_view(_status(phase="training", worker=stand_in, remote=REMOTE))
    assert "worker" not in shown
    assert shown["remote"] == REMOTE
    keys = list(shown)
    assert keys.index("finalize_error") < keys.index("remote") < keys.index("warnings")
    ended = {**REMOTE, "state": "ended"}
    shown = status_view(_status(phase="training", worker=worker(pid=None), remote=ended))
    assert shown["worker"]["state"] == "exited"
    assert "remote" not in status_view(_status())


def test_the_status_line_names_the_remote_job_s_state() -> None:
    fields = {"epoch": 1, "max_epochs": 2, "step": 3, "max_steps": 10, "val_losses": []}
    stand_in = worker(pid=None, alive=True, running=True, starting=True)
    starting = {**REMOTE, "state": "starting"}
    line = status_line(
        status_view(_status(phase="training", worker=stand_in, remote=starting, **fields))
    )
    assert line == "training: epoch 1/2 step 3/10 (Baseten job starting)"
    ended = status_view(
        _status(
            phase="training", worker=worker(pid=None), remote={**REMOTE, "state": "ended"}, **fields
        )
    )
    assert status_line(ended).endswith("(its process not running)")


def test_an_evaluation_without_a_summary_prints_as_it_is() -> None:
    skipped = {"status": "skipped", "reason": "it was cancelled", "next_command": None}
    assert evaluation_view(skipped) == skipped


def test_an_evaluation_names_only_the_targets_that_reused_results(tmp_path: Path) -> None:
    def target(generations: int, verdicts: int) -> dict[str, Any]:
        return {"reused": {"generations": generations, "verdicts": verdicts}}

    result = {
        "status": "complete",
        "out_dir": str(tmp_path),
        "summary": {"targets": {"base": target(4, 3), "tuned": target(0, 0)}},
    }
    assert evaluation_view(result)["reused"] == {"base": {"generations": 4, "verdicts": 3}}
    result["summary"] = {"targets": {"base": target(0, 0), "tuned": target(0, 0)}}
    assert evaluation_view(result)["reused"] == {}


def test_every_table_view_names_a_command() -> None:
    assert set(TABLE_VIEWS) <= set(COMMAND_ORDER)


ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"


@pytest.fixture
def formats(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """``pup format`` served without a pup: each call's arguments, the document echoed."""
    calls: list[list[str]] = []

    def run(args: Sequence[str], stdin: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(["pup", *args], 0, stdout=stdin, stderr="")

    monkeypatch.setattr(output, "run_pup", run)
    monkeypatch.setattr(registry, "_REGISTRY", {})
    healthy = FakePup().on("--version", raw=True, stdout="pup 1.10.7\n")
    healthy.on("auth", "status", stdout=json.dumps({"status": "valid"}))
    monkeypatch.setattr(doctor_cli, "Pup", healthy.client)
    return calls


def _setup(command: str, tmp_path: Path) -> list[str]:
    if command == "dataset sample":
        dataset = tmp_path / "ds"
        imported = runner.invoke(app, ["dataset", "import", str(dataset), "--file", str(ROWS)])
        assert imported.exit_code == 0, imported.stderr
        return ["dataset", "sample", str(dataset)]
    return {
        "models list": ["models", "list", "--provider", "fireworks"],
        "auth status": ["auth", "status"],
        "doctor": ["doctor"],
        "runs": ["runs", "--dir", str(tmp_path)],
    }[command]


@pytest.mark.parametrize("command", sorted(TABLE_VIEWS))
def test_under_pup_a_table_gets_the_command_s_view_and_a_filter_of_the_user_s_none(
    command: str,
    formats: list[list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(command, tmp_path)
    monkeypatch.setenv("PUP_OUTPUT", "table")
    formats.clear()
    runner.invoke(app, args)
    assert formats, "pup format was never run"
    assert f"--jq={TABLE_VIEWS[command]}" in formats[-1]
    assert f"--command=finetune {command}" in formats[-1]

    # The user's own --jq (PUP_FILTER) is pup's to apply: no view of ours is passed.
    monkeypatch.setenv("PUP_FILTER", ".")
    formats.clear()
    runner.invoke(app, args)
    assert formats
    assert not any(arg.startswith("--jq=") for arg in formats[-1])


def test_the_models_table_shows_the_columns_that_matter() -> None:
    view = TABLE_VIEWS["models list"]
    for column in ("alias", "provider", "base_model", "max_seq_len", "default_lora_rank"):
        assert column in view
    for hidden in ("tokenizer", "renderer"):
        assert hidden not in view


@pytest.mark.parametrize(
    ("args", "key"),
    [
        (["models", "list", "--provider", "fireworks"], "models"),
        (["auth", "status"], "providers"),
    ],
)
def test_a_command_passes_its_table_view_to_emit(
    args: list[str], key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str | None] = []
    emit = _common.emit

    def recording(result: Any, **kwargs: Any) -> None:
        seen.append(kwargs.get("table_view"))
        emit(result, **kwargs)

    monkeypatch.setattr(_common, "emit", recording)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert seen == [TABLE_VIEWS[" ".join(args[:2]) if args[0] == "models" else "auth status"]]
    # What the view picks is in the result.
    assert isinstance(json.loads(result.stdout)[key], list)
