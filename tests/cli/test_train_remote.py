# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`train --remote` end to end on the fakes: the preview and its refusals, the submit through a
fake `baseten` CLI, the job's worker run in-process on the uploaded copy, and `status`,
`resume`, `cancel` and `deploy` reading the job back from its logs.

The provider is the real :class:`BasetenProvider` on the Loops fake and the training job
routes of :class:`FakeTrainingJobs`; each command builds a new one, as a new process would.
"""

from __future__ import annotations

import json
import shlex
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from pup_finetune.training import remote as remote_module
from tests.fakes.baseten import FakeLoops, FakeTrainingJobs
from tests.fakes.baseten_cli import (
    LOST,
    CliAnswer,
    FakeBasetenCli,
    error_answer,
    exec_answer,
    interrupted_answer,
)
from tests.fakes.fireworks_deploy import FakeClock
from tests.fakes.pushed import mark_pushed
from tests.fakes.tokenizer import FakeChatTokenizer
from tests.providers.baseten.test_baseten_provider import KEY, provider_on

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
RUN_ID = "ft-remote"
PROJECT = f"pup-finetune-{RUN_ID}"
TEAM = "Baseten Research"

runner = CliRunner()


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@dataclass
class World:
    tmp: Path
    loops: FakeLoops
    jobs: FakeTrainingJobs
    cli: FakeBasetenCli
    dataset: Path
    run_dir: Path
    job_root: Path
    missing: tuple[str, ...] = ()

    def which(self, name: str) -> str | None:
        return None if name in self.missing else f"/usr/local/bin/{name}"

    def on_exec(self, staged: Path) -> None:
        """What Baseten does with an upload: the job's copy, and the job."""
        shutil.rmtree(self.job_root, ignore_errors=True)
        shutil.copytree(staged, self.job_root)
        self.jobs.create_job(PROJECT)

    def answer(self, *answers: CliAnswer) -> None:
        self.cli = FakeBasetenCli(answers, on_exec=self.on_exec)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    dataset = tmp_path / "ds"
    assert invoke("dataset", "import", str(dataset), "--file", str(ROWS)).exit_code == 0
    mark_pushed(dataset)
    loops, jobs, clock = FakeLoops(), FakeTrainingJobs(), FakeClock()
    found = World(
        tmp_path, loops, jobs, FakeBasetenCli(), dataset, tmp_path / "run", tmp_path / "job"
    )
    found.answer(exec_answer("job1", "proj1", PROJECT))

    def get_provider(name: str, runtime: Runtime) -> Provider:
        assert name == "baseten"
        return provider_on(
            loops,
            FakeChatTokenizer(merge_newlines=True),
            clock=clock,
            cli=found.cli,
            which=found.which,
            job_root=str(found.job_root),
            routes=jobs.routes(),
        )

    monkeypatch.setattr(train_module, "get_provider", get_provider)
    monkeypatch.setattr(inputs_module, "get_provider", get_provider)
    monkeypatch.setattr(deploy_module, "get_provider", get_provider)
    yield found
    redaction.clear_secrets()


def train(world: World, *extra: str, remote: bool = True) -> Result:
    return invoke(
        "train",
        *("--provider", "baseten", "--model", "qwen3p5-9b", "--dataset", str(world.dataset)),
        *("--run-dir", str(world.run_dir), "--run-id", RUN_ID, "--max-epochs", "1"),
        *(["--remote"] if remote else []),
        *extra,
    )


def card(stderr: str) -> list[str]:
    lines = stderr.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Preview only"))
    end = next(
        (i for i in range(start + 1, len(lines)) if not lines[i].startswith("  ")), len(lines)
    )
    return lines[start:end]


def record_of(world: World) -> dict[str, Any]:
    found = RunDir(world.run_dir).read_remote()
    assert found is not None
    return found


# --- the preview -------------------------------------------------------------------------


def test_the_preview_says_where_the_job_runs_and_what_it_uploads(world: World) -> None:
    result = train(world)
    assert result.exit_code == 2, result.stderr
    data = output(result)
    preview = data["preview"]
    remote = preview["remote"]
    assert list(remote) == [
        "backend",
        "team",
        "team_source",
        "project",
        "compute",
        "api_key_secret",
        "uploads",
        "baseten_cli",
        "truss_version",
        "warnings",
    ]
    assert (remote["backend"], remote["team"], remote["team_source"]) == (
        "baseten job",
        TEAM,
        "your only team",
    )
    assert remote["project"] == PROJECT
    assert remote["compute"] == {"cpu_count": 16, "memory": "64Gi", "accelerator": None}
    assert (remote["api_key_secret"], remote["truss_version"]) == (
        "pup_finetune_api_key",
        "0.18.33",
    )
    uploads = remote["uploads"]
    assert uploads["rows"] == {k: preview["rows"][k] for k in ("train", "validation")}
    assert uploads["code"].startswith("pup-finetune ")
    assert uploads["run_files"] == ["run.json", "state.json"]
    assert remote["warnings"][0].startswith("cancel cannot stop a remote run yet")
    assert preview["action"].endswith(
        f", in a Baseten CPU job in team {TEAM} that runs the training loop, so training "
        "continues while this machine sleeps"
    )
    assert preview["billing"].endswith(
        "; Baseten bills the CPU job that runs the training loop (16 vCPUs, 64 GiB) for the "
        "time it runs, at its own rates, which pup-finetune does not know: from the job's "
        "start until training ends"
    )
    assert preview["data_recipients"][-1] == (
        "train and validation rows, run files and pup-finetune source -> Baseten storage (the "
        "job's upload)"
    )
    assert shlex.split(data["next_command"])[-4:] == ["--remote", "--team", TEAM, "--confirm"]
    local = output(train(world, remote=False))["preview"]
    assert preview["settings"] == local["settings"]
    shown = card(result.stderr)
    assert len(shown) == 7
    assert f"  runs      a Baseten CPU job in team {TEAM}" in shown
    assert f"warning: {remote['warnings'][0]}" in result.stderr
    assert not world.run_dir.exists()
    assert world.cli.calls == []


def test_a_spend_cap_counts_the_trainer_only(world: World) -> None:
    result = train(world, "--max-spend-usd", "5", "--hourly-rate-usd", "10")
    assert result.exit_code == 2, result.stderr
    assert (
        "--max-spend-usd counts the trainer's time only; the CPU job bills on top"
        in output(result)["preview"]["remote"]["warnings"]
    )


def _not_a_checkout(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    installed = world.tmp / "site-packages" / "pup_finetune"
    installed.mkdir(parents=True)
    monkeypatch.setattr(
        train_module, "package_source", lambda: remote_module.package_source(installed)
    )


def _other_transformers(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    found = remote_module.package_source()
    other = remote_module.SourceTree(
        found.root, found.package, found.version, {"transformers": "0.1"}
    )
    monkeypatch.setattr(train_module, "package_source", lambda: other)


REFUSALS: list[tuple[str, Any, tuple[str, ...], str]] = [
    ("--team alone", None, ("--team", TEAM), "--team needs --remote"),
    ("--detach", None, ("--remote", "--detach"), "--remote already trains away"),
    ("--evaluate", None, ("--remote", "--evaluate"), "--evaluate needs the foreground"),
    ("no CLI", lambda w, m: setattr(w, "missing", ("baseten",)), ("--remote",), "baseten CLI"),
    ("no uv", lambda w, m: setattr(w, "missing", ("uv",)), ("--remote",), "needs uv"),
    ("not a checkout", _not_a_checkout, ("--remote",), "source checkout"),
    ("transformers", _other_transformers, ("--remote",), "transformers version uv.lock pins"),
    ("unknown team", None, ("--remote", "--team", "Nope"), "no team named 'Nope'"),
    (
        "several teams",
        lambda w, m: (
            w.jobs.teams.append({"id": "team2", "name": "Other", "default": False})
            or w.jobs.teams[0].update(default=False)
        ),
        ("--remote",),
        "several teams and none is the default",
    ),
    ("no secret", lambda w, m: w.jobs.secrets.update(team1=[]), ("--remote",), "has no secret"),
    (
        "a used project",
        lambda w, m: w.jobs.create_job(PROJECT),
        ("--remote",),
        f"already has project {PROJECT} with jobs",
    ),
]


@pytest.mark.parametrize(
    ("setup", "flags", "message"),
    [pytest.param(setup, flags, message, id=name) for name, setup, flags, message in REFUSALS],
)
def test_the_preview_refuses_what_would_fail(
    world: World, monkeypatch: pytest.MonkeyPatch, setup: Any, flags: tuple[str, ...], message: str
) -> None:
    if setup is not None:
        setup(world, monkeypatch)
    result = train(world, *flags, remote=False)
    assert result.exit_code == 2, result.stderr
    assert result.stdout == ""
    assert "Error [usage]: " in result.stderr
    assert message in result.stderr
    assert not world.run_dir.exists()
    assert world.cli.calls == []


def test_a_provider_without_remote_runs_is_refused(world: World) -> None:
    result = invoke(
        *("train", "--provider", "fireworks", "--model", "qwen3p8-27b"),
        *("--dataset", str(world.dataset), "--remote"),
    )
    assert result.exit_code == 2
    assert (
        "Error [usage]: training remotely on Fireworks is not available in this version of "
        "pup-finetune\nHint: drop --remote; --detach trains in a background worker on this "
        "machine\n"
    ) in result.stderr


def test_read_only_previews_but_refuses_the_submit(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    result = train(world)
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["next_command"] is None
    assert data["preview"]["read_only"]
    shown = card(result.stderr)
    assert len(shown) == 7
    assert not any(line.startswith("  settings") for line in shown)
    assert any(line.startswith("  runs") for line in shown)
    refused = train(world, "--confirm")
    assert refused.exit_code == 2
    assert "Error [read_only]: " in refused.stderr
    assert not world.run_dir.exists()
    assert world.cli.calls == []


# --- the submit --------------------------------------------------------------------------


def test_confirm_submits_the_run_and_records_its_job(world: World) -> None:
    result = train(world, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "started"
    assert data["remote"] == {
        "backend": "baseten job",
        "team": TEAM,
        "project": PROJECT,
        "job_id": "job1",
        "compute": {"cpu_count": 16, "memory": "64Gi", "accelerator": None, "gpu_count": None},
        "logs": "baseten train job logs --job-id job1",
    }
    assert data["upload"]["files"] > 100
    assert data["next_command"] == f"pup finetune status {world.run_dir} --watch"
    record = record_of(world)
    assert (record["state"], record["job_id"], record["project"]) == (
        "submitted",
        "job1",
        {"id": "proj1", "name": PROJECT},
    )
    assert (record["api_key_secret"], record["truss_version"], record["mode"]) == (
        "pup_finetune_api_key",
        "0.18.33",
        "train",
    )
    (call,) = world.cli.calls
    assert call.argv[call.argv.index("--team") + 1] == TEAM
    for name in (".truss_ignore", "app/uv.lock", "dataset/rows.jsonl", "run/run.json"):
        assert call.tree[name] == 0o600
    assert {mode for mode in call.tree.values()} == {0o600, 0o700}
    assert not any("__pycache__" in name for name in call.tree)
    assert (
        f"submitting run {RUN_ID} to a Baseten job in team {TEAM}; uploading "
        f"{data['upload']['files']} files, this can take a few minutes"
    ) in result.stderr
    assert f"run {RUN_ID} started in Baseten job job1 (team {TEAM})" in result.stderr
    assert KEY.get_secret_value() not in result.stdout + result.stderr
    assert KEY.get_secret_value() not in json.dumps(record)


def test_a_refused_submit_leaves_nothing_and_the_command_runs_again(world: World) -> None:
    world.answer(error_answer(2, message="no such team"), exec_answer("job1", "proj1", PROJECT))
    refused = train(world, "--confirm")
    assert refused.exit_code == 1
    assert "baseten loops exec failed (exit 2): no such team" in refused.stderr
    assert "Hint: nothing that bills was created" in refused.stderr
    assert not world.run_dir.exists()
    again = train(world, "--confirm")
    assert again.exit_code == 0, again.stderr
    assert record_of(world)["job_id"] == "job1"


def test_a_lost_answer_keeps_the_record_and_status_adopts_the_job(world: World) -> None:
    world.answer(LOST)
    lost = train(world, "--confirm")
    assert lost.exit_code == 1
    assert "it may have created something that bills" in lost.stderr
    assert f"`pup finetune status {world.run_dir}` looks for the job" in lost.stderr
    record = record_of(world)
    assert (record["state"], record["job_id"]) == ("submitting", None)
    assert record["error"]["type"] == "ExternalError"
    status = invoke("status", str(world.run_dir))
    assert status.exit_code == 0, status.stderr
    data = output(status)
    assert (
        "found Baseten job job1 for this run, which the submit did not report" in data["warnings"]
    )
    assert (data["remote"]["job_id"], data["remote"]["state"]) == ("job1", "queued")
    assert data["next_command"] == f"pup finetune status {world.run_dir} --watch"
    assert record_of(world)["job_id"] == "job1"


def test_a_ctrl_c_during_the_submit_keeps_the_record(world: World) -> None:
    world.answer(interrupted_answer())
    result = train(world, "--confirm")
    assert result.exit_code == 1
    assert f"Error [interrupted]: the submit of run {RUN_ID} stopped (Ctrl-C)" in result.stderr
    assert record_of(world)["state"] == "submitting"


# --- the job and status ------------------------------------------------------------------


def run_the_job(world: World) -> list[str]:
    """Run the job's worker on its copy of the run, as its start command does; its stdout."""
    worker = invoke("_worker", str(world.job_root / "run"), "--mode", "train", "--report-state")
    assert worker.exit_code == 0, worker.stderr
    lines = worker.stdout.splitlines()
    assert lines and all(line.startswith(f"PUPFT1 {RUN_ID} ") for line in lines)
    return lines


def seq_of(line: str) -> int:
    return int(line.split()[2])


def test_a_remote_run_trains_and_status_copies_it_back(world: World) -> None:
    assert train(world, "--confirm").exit_code == 0
    queued = output(invoke("status", str(world.run_dir)))
    assert (queued["phase"], queued["remote"]["state"]) == ("created", "queued")
    assert "worker" not in queued
    lines = run_the_job(world)
    world.jobs.set_status("job1", "TRAINING_JOB_RUNNING")
    world.jobs.emit("job1", ["Resolved 120 packages in 2s", *lines])
    world.jobs.set_status("job1", "TRAINING_JOB_COMPLETED")
    status = invoke("status", str(world.run_dir))
    assert status.exit_code == 0, status.stderr
    data = output(status)
    assert data["phase"] == "completed"
    assert data["deployable_checkpoint"]
    assert data["next_command"] == f"pup finetune evaluate {world.run_dir}"
    assert (data["remote"]["state"], data["remote"]["job_status"]) == (
        "ended",
        "TRAINING_JOB_COMPLETED",
    )
    assert data["warnings"] == []
    assert data["files"]["remote"] == str(world.run_dir / "remote.json")
    assert record_of(world)["ended"]["outcome"] == "reported"
    assert RunDir(world.run_dir).load_state().pid is None
    deploy = invoke("deploy", str(world.run_dir))
    assert deploy.exit_code == 2, deploy.stderr
    assert output(deploy)["preview"]["model_source"] == "deployable_checkpoint"


def test_the_watch_ends_after_the_final_report(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert train(world, "--confirm").exit_code == 0
    lines = run_the_job(world)
    last = max(seq_of(line) for line in lines)
    world.jobs.set_status("job1", "TRAINING_JOB_RUNNING")
    world.jobs.emit("job1", [line for line in lines if seq_of(line) < last])
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        world.jobs.emit("job1", [line for line in lines if seq_of(line) == last])
        world.jobs.set_status("job1", "TRAINING_JOB_COMPLETED")

    monkeypatch.setattr(train_module.time, "sleep", sleep)
    watch = invoke("status", str(world.run_dir), "--watch", "--interval", "1")
    assert watch.exit_code == 0, watch.stderr
    data = output(watch)
    assert (data["phase"], data["watch_timed_out"]) == ("completed", False)
    assert len(sleeps) == 1
    assert "(Baseten job running)" in watch.stderr


def test_a_job_that_died_shows_why_and_resume_is_next(world: World) -> None:
    assert train(world, "--confirm").exit_code == 0
    ended = datetime.now(UTC) - timedelta(minutes=5)
    world.jobs.jobs["job1"].created_at = ended - timedelta(minutes=10)
    world.jobs.set_status("job1", "TRAINING_JOB_FAILED", at=ended)
    world.jobs.emit("job1", ["OOMKilled"], replica=None, level="ERROR", at=ended)
    data = output(invoke("status", str(world.run_dir)))
    assert data["warnings"][0] == (
        "Baseten job job1 ended (TRAINING_JOB_FAILED) before its final state report: OOMKilled"
    )
    assert data["remote"]["state"] == "ended"
    assert data["next_command"] == f"pup finetune resume {world.run_dir}"


def test_a_resume_that_sees_the_end_first_leaves_its_warnings_to_status(world: World) -> None:
    assert train(world, "--confirm").exit_code == 0
    ended = datetime.now(UTC) - timedelta(minutes=5)
    world.jobs.jobs["job1"].created_at = ended - timedelta(minutes=10)
    world.jobs.set_status("job1", "TRAINING_JOB_STOPPED", at=ended)
    world.jobs.emit("job1", ["stopped by a user"], replica=None, level="ERROR", at=ended)
    end = [
        "Baseten job job1 ended (TRAINING_JOB_STOPPED) before its final state report: stopped "
        "by a user",
        "a Loops trainer may still bill: check the Loops runs in the Baseten UI",
    ]
    resume = invoke("resume", str(world.run_dir))
    assert resume.exit_code == 2, resume.stderr
    assert output(resume)["preview"]["warnings"][:2] == end
    status = output(invoke("status", str(world.run_dir)))
    assert status["remote"]["state"] == "ended"
    assert all(warning in status["warnings"] for warning in end)


def test_resume_and_cancel_wait_for_the_job(world: World) -> None:
    assert train(world, "--confirm").exit_code == 0
    resume = invoke("resume", str(world.run_dir))
    assert resume.exit_code == 2
    assert (
        f"Error [wrong_phase]: the run in {world.run_dir} trains in Baseten job job1 (queued)"
        in resume.stderr
    )
    assert "cancel cannot stop a Baseten job yet" in resume.stderr
    cancel = invoke("cancel", str(world.run_dir), "--confirm")
    assert cancel.exit_code == 2
    assert (
        f"Error [usage]: cancel cannot stop Baseten job job1 yet (team {TEAM}, project {PROJECT})"
    ) in cancel.stderr
    assert "stop the job in the Baseten UI" in cancel.stderr
    world.jobs.set_status(
        "job1", "TRAINING_JOB_STOPPED", at=datetime.now(UTC) - timedelta(minutes=5)
    )
    resume = invoke("resume", str(world.run_dir))
    assert resume.exit_code == 2, resume.stderr
    assert (
        "this run trained in Baseten job job1; this resume trains on this machine"
        in output(resume)["preview"]["warnings"]
    )
