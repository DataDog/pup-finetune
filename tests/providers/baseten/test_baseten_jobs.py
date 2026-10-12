# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten training jobs for ``train --remote`` on the fakes: the plan, the one submit and its
errors, and the reads of the job and its logs."""

from __future__ import annotations

import functools
import os
import shlex
import shutil
import signal
import stat
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    ProviderAuthError,
    ProviderCapacityError,
    RemoteJobGone,
    RemoteJobRejected,
    SubmittedJob,
)
from pup_finetune.providers.baseten.client import _CREATED_HINT, BasetenHTTP
from pup_finetune.providers.baseten.jobs import (
    BASETEN_CLI_ENV,
    EXEC_TIMEOUT_S,
    BasetenJobs,
    CliRunner,
    job_state,
    run_cli,
)
from pup_finetune.training.remote import ADOPT_AFTER_S
from tests.fakes.baseten import DEFAULT_TEAM, FakeLoops, FakeTrainingJobs
from tests.fakes.baseten_cli import (
    CliAnswer,
    FakeBasetenCli,
    error_answer,
    exec_answer,
    not_started_answer,
    timeout_answer,
)
from tests.providers.baseten.test_baseten_provider import KEY

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
PROJECT = "pup-finetune-ft-x"
COMMAND = ["/bin/sh", "-c", "true"]


def found(name: str) -> str | None:
    return f"/usr/local/bin/{name}"


def make(
    jobs: FakeTrainingJobs,
    cli: CliRunner | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = found,
    loops: FakeLoops | None = None,
) -> BasetenJobs:
    fake = loops or FakeLoops()
    return BasetenJobs(
        http=lambda: BasetenHTTP(KEY, transport=fake.transport(jobs.routes())),
        key=lambda: KEY,
        environ={"PATH": "/usr/bin"} if environ is None else environ,
        runner=cli or FakeBasetenCli(),
        which=which,
    )


@pytest.fixture(autouse=True)
def _forget_secrets() -> Any:
    yield
    redaction.clear_secrets()


def submit(jobs: BasetenJobs, tmp_path: Path) -> SubmittedJob:
    staged = tmp_path / "staged"
    staged.mkdir(exist_ok=True)
    return jobs.submit(jobs.plan(run_id="ft-x", team=None), staged, COMMAND)


def test_the_submit_runs_the_cli_once_with_the_key_in_its_environment_only(
    tmp_path: Path,
) -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "BASETEN_API_KEY": "another-key-entirely",
        "FIREWORKS_API_KEY": "fw-key",
        "DD_API_KEY": "dd-key",
        "BASETEN_PROFILE": "p",
        "BASETEN_REMOTE_URL": "https://elsewhere",
        "BASETEN_MANAGEMENT_API_URL_OVERRIDE": "https://elsewhere",
        "BASETEN_MODEL_API": "https://elsewhere",
        "BASETEN_TRUSS_VERSION": "0.1",
        "BASETEN_TRUSS_EXECUTABLE": "/opt/truss",
        "BASETEN_TRUSS_AUTH_REMOTE_URL": "https://elsewhere",
        "BASETEN_TRUSS_AUTH_API_KEY": "k",
        "PUP_OUTPUT": "json",
        "PUP_FILTER": ".x",
        "LOOPS_TEAM": "t",
        "INTERNAL_LOOPS_BACKEND_VERSION": "2",
        "NO_COLOR": "0",
    }
    cli = FakeBasetenCli([exec_answer("job1", "proj1", PROJECT)])
    submitted = submit(make(FakeTrainingJobs(), cli, environ=environ), tmp_path)
    assert submitted == SubmittedJob(
        job_id="job1",
        project_id="proj1",
        project=PROJECT,
        compute={"cpu_count": 16, "memory": "64Gi", "accelerator": None, "gpu_count": None},
    )
    (call,) = cli.calls
    assert call.argv == [
        *("/usr/local/bin/baseten", "loops", "exec", "--output", "json"),
        *("--dir", str(tmp_path / "staged"), "--team", DEFAULT_TEAM, "--project-name", PROJECT),
        *("--cpu-count", "16", "--memory", "64Gi", "--with-uv", "--no-ssh", "--no-api-key"),
        *("--secret", "BASETEN_API_KEY=pup_finetune_api_key", "--truss-version", "0.18.33"),
        *("--", *COMMAND),
    ]
    assert call.timeout == 900.0
    assert call.env == {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "BASETEN_API_KEY": KEY.get_secret_value(),
        "TRUSS_NO_UPDATE_CHECK": "1",
        "NO_COLOR": "1",
    }
    assert KEY.get_secret_value() not in " ".join(call.argv)


def test_the_upload_ignores_only_caches(tmp_path: Path) -> None:
    cli = FakeBasetenCli()
    submit(make(FakeTrainingJobs(), cli), tmp_path)
    assert cli.calls[0].tree == {".truss_ignore": 0o600}
    assert (tmp_path / "staged" / ".truss_ignore").read_text() == "__pycache__/\n*.pyc\n.venv/\n"


NOTHING = "nothing that bills was created"

ERRORS: list[tuple[str, CliAnswer, type[ExternalError], str]] = [
    ("exit 2", error_answer(2), RemoteJobRejected, NOTHING),
    ("exit 4", error_answer(4), RemoteJobRejected, NOTHING),
    ("exit 5", error_answer(5), RemoteJobRejected, NOTHING),
    ("a 4xx from the CLI", error_answer(1, status=400), RemoteJobRejected, NOTHING),
    ("a 4xx from truss", error_answer(6, status=422, truss=True), RemoteJobRejected, NOTHING),
    ("no process", not_started_answer(), RemoteJobRejected, NOTHING),
    ("exit 3", error_answer(3), ProviderAuthError, NOTHING),
    ("a 401", error_answer(1, status=401), ProviderAuthError, NOTHING),
    ("a 403", error_answer(6, status=403), ProviderAuthError, NOTHING),
    ("a 429", error_answer(1, status=429), ProviderCapacityError, NOTHING),
    ("a 408", error_answer(1, status=408), ExternalError, _CREATED_HINT),
    ("exit 1", error_answer(1), ExternalError, _CREATED_HINT),
    ("exit 6", error_answer(6), ExternalError, _CREATED_HINT),
    ("a timeout", timeout_answer(), ExternalError, _CREATED_HINT),
    ("unreadable output", CliAnswer(0, "Uploading... done"), ExternalError, _CREATED_HINT),
    ("no job id", CliAnswer(0, '{"project": {"id": "proj1"}}'), ExternalError, _CREATED_HINT),
]


@pytest.mark.parametrize(
    ("answer", "error", "hint"),
    [pytest.param(answer, error, hint, id=name) for name, answer, error, hint in ERRORS],
)
def test_each_failure_says_whether_anything_was_created(
    tmp_path: Path, answer: CliAnswer, error: type[ExternalError], hint: str
) -> None:
    cli = FakeBasetenCli([answer])
    with pytest.raises(ExternalError) as raised:
        submit(make(FakeTrainingJobs(), cli), tmp_path)
    assert type(raised.value) is error
    assert hint in str(raised.value.hint)
    assert len(cli.calls) == 1  # never retried


def test_error_text_is_short_and_redacted(tmp_path: Path) -> None:
    long = f"bad request for {KEY.get_secret_value()} " + "x" * 1000
    with pytest.raises(RemoteJobRejected) as raised:
        submit(make(FakeTrainingJobs(), FakeBasetenCli([error_answer(2, message=long)])), tmp_path)
    message = raised.value.message
    assert KEY.get_secret_value() not in message
    assert len(message.partition(": ")[2]) <= 300


TEAMS = [
    {"id": "team1", "name": "Research", "default": False},
    {"id": "team2", "name": "Platform", "default": True},
]


@pytest.mark.parametrize(
    ("teams", "given", "team", "source"),
    [
        (TEAMS, "Research", "Research", "--team"),
        (TEAMS[:1], None, "Research", "your only team"),
        (TEAMS, None, "Platform", "your default team"),
    ],
)
def test_the_team_is_given_the_only_one_or_the_default(
    teams: list[dict[str, Any]], given: str | None, team: str, source: str
) -> None:
    plan = make(FakeTrainingJobs(teams=teams)).plan(run_id="ft-x", team=given)
    assert (plan.team, plan.team_source, plan.project) == (team, source, PROJECT)
    assert plan.compute == {"cpu_count": 16, "memory": "64Gi", "accelerator": None}
    assert plan.cli == "/usr/local/bin/baseten"


@pytest.mark.parametrize(
    ("teams", "given", "message"),
    [
        (TEAMS, "Nope", "sees no team named 'Nope'"),
        ([{**team, "default": False} for team in TEAMS], None, "several teams and none"),
    ],
)
def test_an_unknown_or_ambiguous_team_is_refused(
    teams: list[dict[str, Any]], given: str | None, message: str
) -> None:
    with pytest.raises(UsageError, match=message) as raised:
        make(FakeTrainingJobs(teams=teams)).plan(run_id="ft-x", team=given)
    assert "pass --team with one of: 'Research', 'Platform'" in str(raised.value.hint)


def test_a_team_without_the_key_secret_is_refused() -> None:
    with pytest.raises(UsageError, match="has no secret pup_finetune_api_key") as raised:
        make(FakeTrainingJobs(secrets={"team1": ["hf_access_token"]})).plan(
            run_id="ft-x", team=None
        )
    assert "`baseten org secret set --name pup_finetune_api_key --team 'Baseten Research'`" in str(
        raised.value.hint
    )


def test_an_empty_project_is_reused_but_one_with_jobs_is_refused() -> None:
    jobs = FakeTrainingJobs()
    jobs.project(PROJECT)
    jobs.create_job(PROJECT, "Another team")
    assert make(jobs).plan(run_id="ft-x", team=None).project == PROJECT
    jobs.create_job(PROJECT)
    with pytest.raises(UsageError, match=f"already has project {PROJECT} with jobs") as raised:
        make(jobs).plan(run_id="ft-x", team=None)
    assert raised.value.hint == "pass another --run-id"


@pytest.mark.parametrize(
    ("environ", "missing", "message"),
    [
        ({}, "baseten", "needs the baseten CLI, which is not on PATH"),
        ({BASETEN_CLI_ENV: "/opt/nope"}, "/opt/nope", f"{BASETEN_CLI_ENV} names '/opt/nope'"),
        ({}, "uv", "needs uv, which is not on PATH"),
    ],
)
def test_a_missing_cli_or_uv_is_refused(
    environ: dict[str, str], missing: str, message: str
) -> None:
    def which(name: str) -> str | None:
        return None if name == missing else found(name)

    with pytest.raises(UsageError, match=message):
        make(FakeTrainingJobs(), environ=environ, which=which).plan(run_id="ft-x", team=None)


def test_the_configured_cli_wins() -> None:
    def which(name: str) -> str | None:
        return name if name.startswith("/") else found(name)

    plan = make(FakeTrainingJobs(), environ={BASETEN_CLI_ENV: "/opt/baseten"}, which=which).plan(
        run_id="ft-x", team=None
    )
    assert plan.cli == "/opt/baseten"


def test_log_pages_go_backward_without_repeating_a_line() -> None:
    jobs = FakeTrainingJobs(now=lambda: NOW - timedelta(minutes=30))
    job_id = jobs.create_job(PROJECT)
    start = NOW - timedelta(minutes=10)
    for index in range(2500):  # 2.5 lines a millisecond, so pages meet inside one
        jobs.emit(job_id, [f"line {index}"], at=start + timedelta(microseconds=400 * index))
    jobs.emit(job_id, ["platform line"], replica=None, at=start)
    remote = make(jobs)
    status = remote.job(project_id="proj1", job_id=job_id)
    pages = list(remote.output_pages(status, since=None, now=NOW))
    messages = [line.message for page in pages for line in page]
    assert messages == [f"line {index}" for index in reversed(range(2500))]
    assert len(pages) == 3
    ends = [int(query["end_epoch_millis"]) for query in jobs.log_queries]
    assert ends == sorted(ends, reverse=True)
    assert {query["start_epoch_millis"] for query in jobs.log_queries} == {
        str(int(status.created_at.timestamp() * 1000))  # type: ignore[union-attr]
    }


def test_no_log_window_is_longer_than_seven_days() -> None:
    jobs = FakeTrainingJobs(now=lambda: NOW - timedelta(days=10))
    job_id = jobs.create_job(PROJECT)
    remote = make(jobs)
    status = remote.job(project_id="proj1", job_id=job_id)
    assert list(remote.output_pages(status, since=NOW - timedelta(days=9), now=NOW)) == [[]]
    remote.failure(status, now=NOW)
    for query in jobs.log_queries:
        window = int(query["end_epoch_millis"]) - int(query["start_epoch_millis"])
        assert window <= timedelta(days=7).total_seconds() * 1000


def test_a_recent_report_reads_from_a_minute_before_it() -> None:
    jobs = FakeTrainingJobs(now=lambda: NOW - timedelta(hours=1))
    job_id = jobs.create_job(PROJECT)
    remote = make(jobs)
    status = remote.job(project_id="proj1", job_id=job_id)
    list(remote.output_pages(status, since=NOW - timedelta(minutes=5), now=NOW))
    (query,) = jobs.log_queries
    assert int(query["start_epoch_millis"]) == int((NOW - timedelta(minutes=6)).timestamp() * 1000)
    assert int(query["end_epoch_millis"]) == int((NOW + timedelta(minutes=1)).timestamp() * 1000)


def test_a_failure_is_the_newest_platform_error() -> None:
    jobs = FakeTrainingJobs(now=lambda: NOW - timedelta(hours=1))
    job_id = jobs.create_job(PROJECT)
    jobs.emit(job_id, ["older"], replica=None, level="ERROR", at=NOW - timedelta(minutes=9))
    jobs.emit(job_id, ["OOMKilled"], replica=None, level="ERROR", at=NOW - timedelta(minutes=8))
    jobs.emit(job_id, ["just info"], replica=None, at=NOW - timedelta(minutes=7))
    remote = make(jobs)
    status = remote.job(project_id="proj1", job_id=job_id)
    assert remote.failure(status, now=NOW) == "OOMKilled"
    assert jobs.log_queries[-1]["min_level"] == "ERROR"
    assert jobs.log_queries[-1]["limit"] == "1"


def test_find_keeps_the_project_s_jobs_created_after_the_submit() -> None:
    clock = [NOW - timedelta(hours=2)]
    jobs = FakeTrainingJobs(now=lambda: clock[0])
    jobs.create_job(PROJECT)
    clock[0] = NOW
    newer = jobs.create_job(PROJECT)
    undated = jobs.create_job(PROJECT)
    jobs.jobs[undated].created_at = None
    jobs.create_job(PROJECT, "Another team")
    jobs.create_job("pup-finetune-other")
    found = make(jobs).find(team=DEFAULT_TEAM, project=PROJECT, after=NOW - timedelta(minutes=1))
    assert [job.job_id for job in found] == [newer, undated]
    assert found[0].state == "queued"


def test_a_job_baseten_no_longer_has_is_gone() -> None:
    jobs = FakeTrainingJobs()
    job_id = jobs.create_job(PROJECT)
    jobs.fail["job"] = (404, {"detail": "not found"})
    with pytest.raises(RemoteJobGone, match="Baseten has no training job job1"):
        make(jobs).job(project_id="proj1", job_id=job_id)


def test_a_lost_submit_is_looked_for_longer_than_the_cli_may_run() -> None:
    assert EXEC_TIMEOUT_S < ADOPT_AFTER_S


@pytest.mark.parametrize(
    ("status", "state"),
    [
        ("TRAINING_JOB_PENDING", "queued"),
        ("TRAINING_JOB_DEPLOYING", "queued"),
        ("TRAINING_JOB_RUNNING", "running"),
        ("TRAINING_JOB_COMPLETED", "ended"),
        ("TRAINING_JOB_PREEMPTED", "ended"),
        ("TRAINING_JOB_SOMETHING_NEW", "unknown"),
    ],
)
def test_each_status_has_its_state(status: str, state: str) -> None:
    assert job_state(status) == state


def test_a_trainer_is_active_until_deactivated_and_unknown_when_unreadable() -> None:
    loops = FakeLoops()
    loops.sampler_checkpoint()
    run = next(iter(loops.runs.values()))
    remote = make(FakeTrainingJobs(), loops=loops)
    assert remote.trainer_active(run.run_id) is False
    run.active = True
    assert remote.trainer_active(run.run_id) is True
    assert remote.trainer_active("run404") is False
    failing = BasetenJobs(
        http=lambda: BasetenHTTP(KEY, transport=httpx.MockTransport(lambda r: httpx.Response(500))),
        key=lambda: KEY,
        environ={},
        runner=FakeBasetenCli(),
        which=found,
    )
    assert failing.trainer_active(run.run_id) is None


def test_the_real_cli_runs_without_stdin_or_the_key_on_its_command_line(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "fake-baseten"
    key = KEY.get_secret_value()
    script.write_text(
        "#!/bin/sh\n"
        "[ ! -t 0 ] || exit 9\n"
        "read -r line && exit 10\n"
        f'case "$*" in *{key}*) exit 8;; esac\n'
        f'[ "$BASETEN_API_KEY" = "{key}" ] || exit 7\n'
        '[ -z "$LOOPS_TEAM$PUP_OUTPUT" ] || exit 6\n'
        '[ "$TRUSS_NO_UPDATE_CHECK$NO_COLOR" = "11" ] || exit 5\n'
        'echo "uploading" >&2\n'
        'echo \'{"job_id": "job1", "project": {"id": "proj1", "name": "p"}}\'\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "uv").write_text("#!/bin/sh\n")
    (bin_dir / "uv").chmod(0o700)
    environ = {"PATH": "/usr/bin:/bin", BASETEN_CLI_ENV: str(script), "LOOPS_TEAM": "t"}
    jobs = make(
        FakeTrainingJobs(),
        run_cli,
        environ=environ,
        which=functools.partial(shutil.which, path=str(bin_dir)),
    )
    assert submit(jobs, tmp_path).job_id == "job1"


def test_a_second_ctrl_c_kills_a_cli_that_ignores_sigterm_at_once(tmp_path: Path) -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers are set on the main thread only")
    pid_file = tmp_path / "pid"
    script = f"trap '' TERM; sleep 30 & echo $! > {shlex.quote(str(pid_file))}; wait"

    def press_ctrl_c_twice() -> None:
        deadline = time.monotonic() + 10
        while not pid_file.exists() or not pid_file.read_text().strip():
            if time.monotonic() > deadline:
                return
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGINT)

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    presser = threading.Thread(target=press_ctrl_c_twice)
    try:
        presser.start()
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            run_cli(["/bin/sh", "-c", script], env={"PATH": "/usr/bin:/bin"}, timeout=60)
        presser.join()
    finally:
        signal.signal(signal.SIGINT, previous)
    assert time.monotonic() - started < 4
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while True:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, f"process {child} still runs"
        time.sleep(0.05)
