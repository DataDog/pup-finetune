# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""``train --remote``'s provider-neutral half: what the job is sent, the command it runs, and
how ``status`` copies its state back (on an in-memory ``RemoteJobs``)."""

from __future__ import annotations

import importlib.metadata
import json
import signal
import threading
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings
from pup_finetune.data.labels import Label
from pup_finetune.data.schema import TrainingRow, split_key, validate_row
from pup_finetune.data.split import SplitParams
from pup_finetune.errors import Interrupted, StateError, UsageError
from pup_finetune.providers.base import (
    JobState,
    JobStatus,
    LogLine,
    ProviderAuthError,
    RemoteJobGone,
    RemotePlan,
    SubmittedJob,
    TrainSpec,
    TransientError,
)
from pup_finetune.providers.fake import FAKE_PROFILE, FakeRenderer
from pup_finetune.runs import RunDir, RunState, dataset_content_hash, rows_content_hash
from pup_finetune.runtime import Runtime
from pup_finetune.storage import DirectoryLock
from pup_finetune.training.inputs import load_run_inputs
from pup_finetune.training.remote import (
    SourceTree,
    package_source,
    require_locked_renderer,
    stage_run,
    start_command,
    submit_remote,
    sync_remote,
)
from pup_finetune.training.report import Snapshot, encode
from tests.fakes.baseten_cli import tree

RUN_ID = "ft-remote"
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
TRANSFORMERS = importlib.metadata.version("transformers")


DIGITS_AS_LETTERS = str.maketrans("0123456789", "abcdefghij")
"""Requests that differ only in a number are copies of one, which share a split."""


def _row(row_id: str) -> TrainingRow:
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": f"question {row_id.translate(DIGITS_AS_LETTERS)}"},
                {"role": "assistant", "content": f"answer {row_id}"},
            ],
        }
    )


DROPPED = "r00"


@pytest.fixture
def dataset(tmp_path: Path) -> DatasetDir:
    dataset = DatasetDir(tmp_path / "ds")
    settings = DatasetSettings("import", SplitParams(42, 0.25, 0.25))
    dataset.add([_row(f"r{index:02d}") for index in range(24)], settings=settings)
    with DirectoryLock(dataset.path):
        dataset.append_labels([Label(id=DROPPED, keep=False, reason="bad")])
    assert all(dataset.rows_by_split().values()), "every split has rows"
    return dataset


@pytest.fixture
def run(tmp_path: Path, dataset: DatasetDir) -> RunDir:
    return RunDir.create(
        tmp_path / "run",
        provider="fake",
        profile=FAKE_PROFILE,
        spec=TrainSpec(),
        dataset_dir=dataset.path,
        dataset_sha256=dataset_content_hash(dataset),
        rendering=FakeRenderer().identity(),
        run_id=RUN_ID,
    )


@pytest.fixture
def source(tmp_path: Path) -> SourceTree:
    root = tmp_path / "checkout"
    package = root / "src" / "pup_finetune"
    (package / "cli" / "__pycache__").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "pup-finetune"\n')
    (root / "uv.lock").write_text(
        f'version = 1\n\n[[package]]\nname = "transformers"\nversion = "{TRANSFORMERS}"\n'
    )
    for name in ("__init__.py", "_version.py", "cli/__init__.py"):
        (package / name).write_text("# code\n")
    (package / "cli" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (package / "stale.pyc").write_bytes(b"\0")
    (package / ".env").write_text("BASETEN_API_KEY=secret\n")
    (package / ".cache").mkdir()
    (package / ".cache" / "notes.py").write_text("# notes\n")
    (tmp_path / "baseten.env").write_text("BASETEN_API_KEY=secret\n")
    (package / "linked.py").symlink_to(tmp_path / "baseten.env")
    (package / "linked").symlink_to(tmp_path, target_is_directory=True)
    return package_source(package)


# --- what the job gets -------------------------------------------------------------------


def test_the_staged_files_and_their_modes(run: RunDir, source: SourceTree, tmp_path: Path) -> None:
    dest = tmp_path / "staged"
    dest.mkdir(mode=0o700)
    counts = stage_run(run, dest, root="/b10/workspace", source=source)
    files = {
        "app/pyproject.toml",
        "app/uv.lock",
        "app/src/pup_finetune/__init__.py",
        "app/src/pup_finetune/_version.py",
        "app/src/pup_finetune/cli/__init__.py",
        "dataset/rows.jsonl",
        "dataset/settings.json",
        "dataset/splits.json",
        "run/run.json",
        "run/state.json",
    }
    dirs = {"app", "app/src", "app/src/pup_finetune", "app/src/pup_finetune/cli", "dataset", "run"}
    assert tree(dest) == {**dict.fromkeys(files, 0o600), **dict.fromkeys(dirs, 0o700)}
    assert counts["files"] == len(files)
    assert counts["bytes"] == sum((dest / name).stat().st_size for name in files)


def test_no_test_label_or_dropped_row_is_staged(
    run: RunDir, dataset: DatasetDir, source: SourceTree, tmp_path: Path
) -> None:
    dest = tmp_path / "staged"
    counts = stage_run(run, dest, root="/b10/workspace", source=source)
    kept = dataset.rows_by_split()
    staged = [json.loads(line)["id"] for line in (dest / "dataset/rows.jsonl").open()]
    assert staged == sorted(row.id for row in (*kept["train"], *kept["validation"]))
    assert (counts["train"], counts["validation"]) == (len(kept["train"]), len(kept["validation"]))
    assert DROPPED in {row.id for row in dataset.rows()}
    assert DROPPED not in staged
    assert not set(staged) & {row.id for row in kept["test"]}
    splits = json.loads((dest / "dataset/splits.json").read_text())
    assert set(splits["assignments"]) == {
        split_key(row) for row in (*kept["train"], *kept["validation"])
    }
    assert "test" not in splits["assignments"].values()
    assert json.loads((dest / "dataset/settings.json").read_text()) == {
        "split": dataset.settings()["split"]
    }
    assert not (dest / "dataset" / "labels.jsonl").exists()


def test_the_staged_run_names_the_staged_dataset_and_its_hash(
    run: RunDir, dataset: DatasetDir, source: SourceTree, tmp_path: Path
) -> None:
    dest = tmp_path / "staged"
    stage_run(run, dest, root="/b10/workspace", source=source)
    staged = json.loads((dest / "run/run.json").read_text())
    kept = dataset.rows_by_split()
    expected = rows_content_hash(
        {"train": kept["train"], "validation": kept["validation"], "test": []}
    )
    assert staged["dataset_dir"] == "/b10/workspace/dataset"
    assert staged["dataset_sha256"] == expected
    # The local run keeps the full hash, which evaluate checks.
    assert run.manifest()["dataset_sha256"] == dataset_content_hash(dataset) != expected
    assert json.loads((dest / "run/state.json").read_text())["phase"] == "created"


def test_the_job_s_worker_accepts_the_staged_copy(
    run: RunDir, dataset: DatasetDir, source: SourceTree, tmp_path: Path
) -> None:
    root = tmp_path / "job"
    root.mkdir()
    stage_run(run, root, root=str(root), source=source)
    _, inputs = load_run_inputs(RunDir(root / "run"), Runtime(dev_mode=True))
    _, local = load_run_inputs(run, Runtime(dev_mode=True))
    assert [row.row_id for row in inputs.train_rows] == [row.row_id for row in local.train_rows]
    assert inputs.rows["test"] == 0


def test_a_dataset_changed_since_the_run_was_created_is_not_staged(
    run: RunDir, dataset: DatasetDir, source: SourceTree, tmp_path: Path
) -> None:
    dataset.add([_row("late")], settings=DatasetSettings("import", SplitParams(42, 0.25, 0.25)))
    with pytest.raises(StateError, match="changed since the run"):
        stage_run(run, tmp_path / "x", root="/b10/workspace", source=source)


def test_an_install_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    installed = tmp_path / "site-packages" / "pup_finetune"
    installed.mkdir(parents=True)
    with pytest.raises(UsageError, match="source checkout"):
        package_source(installed)
    other = tmp_path / "other" / "src" / "pup_finetune"
    other.mkdir(parents=True)
    (tmp_path / "other" / "pyproject.toml").write_text('[project]\nname = "something"\n')
    (tmp_path / "other" / "uv.lock").write_text("version = 1\n")
    with pytest.raises(UsageError, match="source checkout"):
        package_source(other)


def test_this_checkout_is_one() -> None:
    found = package_source()
    assert (found.root / "pyproject.toml").is_file()
    assert found.locked["transformers"]


def test_a_transformers_version_other_than_the_locked_one_is_refused(
    source: SourceTree,
) -> None:
    require_locked_renderer(source)
    other = SourceTree(source.root, source.package, source.version, {"transformers": "0.0.1"})
    with pytest.raises(UsageError, match=r"transformers version uv\.lock pins \(0\.0\.1\)"):
        require_locked_renderer(other)


def plan(root: str = "/b10/workspace") -> RemotePlan:
    return RemotePlan(
        backend="baseten job",
        team="Team",
        team_source="--team",
        project=f"pup-finetune-{RUN_ID}",
        root=root,
        extra="baseten",
        cli="/usr/bin/baseten",
        compute={"cpu_count": 16},
        secret="pup_finetune_api_key",  # noqa: S106 - a secret's name
        tool_versions={"truss": "0.18.33"},
        billing="",
        warnings=(),
    )


def test_the_start_command_is_fixed() -> None:
    assert start_command(plan(), "train") == [
        "/bin/sh",
        "-c",
        "cd /b10/workspace && uv sync --quiet --frozen --no-dev --extra baseten "
        "--no-install-project --project app --python 3.12 && PYTHONPATH=/b10/workspace/app/src "
        "PYTHONUNBUFFERED=1 exec app/.venv/bin/python -P -m pup_finetune _worker "
        "/b10/workspace/run --mode train --report-state",
    ]


# --- sync --------------------------------------------------------------------------------


class FakeJobs:
    """``RemoteJobs`` in memory: one job, the lines it logged, and the jobs ``find`` sees."""

    def __init__(self) -> None:
        self.status = self.job_status("RUNNING", "running")
        self.lines: list[LogLine] = []
        self.found: list[JobStatus] = []
        self.error: Exception | None = None
        self.active: bool | None = False
        self.reason: str | None = "the container ran out of memory"
        self.asked: list[str] = []
        self.meanwhile: Callable[[], None] | None = None
        self.logs_error: Exception | None = None

    @staticmethod
    def job_status(
        status: str, state: JobState, *, job_id: str = "j1", updated: timedelta = timedelta(0)
    ) -> JobStatus:
        return JobStatus(
            job_id=job_id,
            project_id="p1",
            status=status,
            state=state,
            created_at=NOW - timedelta(minutes=5),
            updated_at=NOW - updated,
        )

    def plan(self, *, run_id: str, team: str | None) -> RemotePlan:
        raise NotImplementedError

    def submit(self, plan: RemotePlan, staged: Path, command: Sequence[str]) -> SubmittedJob:
        raise NotImplementedError

    def find(self, *, team: str, project: str, after: datetime) -> list[JobStatus]:
        return [job for job in self.found if job.created_at is not None and job.created_at >= after]

    def job(self, *, project_id: str, job_id: str) -> JobStatus:
        if self.error is not None:
            raise self.error
        return self.status

    def output_pages(
        self, job: JobStatus, *, since: datetime | None, now: datetime
    ) -> Iterator[list[LogLine]]:
        if self.meanwhile is not None:
            self.meanwhile()
        if self.logs_error is not None:
            raise self.logs_error
        yield sorted(self.lines, key=lambda line: line.timestamp_ns, reverse=True)

    def failure(self, job: JobStatus, *, now: datetime) -> str | None:
        return self.reason

    def trainer_active(self, session_id: str) -> bool | None:
        self.asked.append(session_id)
        return self.active

    def logs_command(self, job_id: str) -> str:
        return f"logs {job_id}"

    def report(self, run: RunDir, seq: int, *, final: bool = False, **state: Any) -> None:
        """Log a state report of the job's copy of ``run``, 30 s ago."""
        fields: dict[str, Any] = {"phase": "training", "step": 5, "pid": 7, **state}
        copy = RunState(run, heartbeat_at=(NOW - timedelta(seconds=40)).isoformat(), **fields)
        snapshot = Snapshot(
            run_id=RUN_ID,
            seq=seq,
            at=(NOW - timedelta(seconds=30)).isoformat(timespec="seconds"),
            final=final,
            files={"state.json": copy.to_json(), "epochs.json": [], "result.json": None},
            exit_error={"type": "SettingsConflict", "message": "changed"} if final else None,
        )
        start = int((NOW - timedelta(seconds=30)).timestamp() * 1e9)
        self.lines += [LogLine(start + i, line) for i, line in enumerate(encode(snapshot))]


def record(run: RunDir, **fields: Any) -> dict[str, Any]:
    data = {
        "backend": "baseten job",
        "state": "submitted",
        "team": "Team",
        "project": {"id": "p1", "name": f"pup-finetune-{RUN_ID}"},
        "job_id": "j1",
        "mode": "train",
        "submitted_at": (NOW - timedelta(minutes=6)).isoformat(timespec="seconds"),
        "job_status": None,
        "synced_at": None,
        "snapshot": None,
        "ended": None,
        "error": None,
        "logs": "logs j1",
        **fields,
    }
    run.write_remote(data)
    return data


def synced(run: RunDir, jobs: FakeJobs) -> tuple[dict[str, Any], Any, list[str]]:
    found = sync_remote(run, jobs, now=lambda: NOW)
    return found.view, found.worker, found.warnings


@pytest.mark.parametrize(
    ("status", "job_state", "reports", "updated", "state", "running", "starting"),
    [
        ("QUEUED", "queued", 0, 0, "queued", True, True),
        ("RUNNING", "running", 0, 0, "starting", True, True),
        ("RUNNING", "running", 1, 0, "running", True, False),
        ("COMPLETED", "ended", 2, 0, "ended", False, False),
        ("FAILED", "ended", 1, 0, "ending", True, False),
        ("FAILED", "ended", 1, 60, "ending", True, False),
        ("FAILED", "ended", 1, 200, "ended", False, False),
        ("STRANGE", "unknown", 0, 0, "unknown", True, False),
    ],
)
def test_each_job_and_report_has_its_state(
    run: RunDir,
    status: str,
    job_state: JobState,
    reports: int,
    updated: int,
    state: str,
    running: bool,
    starting: bool,
) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status(status, job_state, updated=timedelta(seconds=updated))
    for seq in range(reports):
        jobs.report(run, seq, final=seq == 1, step=seq + 5)
    view, worker, _ = synced(run, jobs)
    assert view["state"] == state
    assert (worker.running, worker.starting, worker.stale, worker.pid) == (
        running,
        starting,
        False,
        None,
    )
    saved = run.read_remote()
    assert saved is not None
    assert (saved["job_status"], saved["synced_at"]) == (status, NOW.isoformat(timespec="seconds"))
    assert (saved["state"] == "ended") == (state == "ended")
    if reports:
        local = run.load_state()
        assert (local.phase, local.step, local.pid) == ("training", reports + 4, None)
        assert view["report_age_s"] == 30


def test_a_final_report_ends_the_run_with_its_exit_error(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status("COMPLETED", "ended")
    jobs.report(run, 0)
    jobs.report(run, 1, final=True)
    view, worker, warnings = synced(run, jobs)
    assert view["state"] == "ended"
    assert worker.error == {"type": "SettingsConflict", "message": "changed"}
    assert warnings == []
    saved = run.read_remote()
    assert saved is not None
    assert saved["ended"]["outcome"] == "reported"
    assert saved["snapshot"]["seq"] == 1
    # Once ended, a sync reads nothing more.
    jobs.error = AssertionError("not read again")
    assert sync_remote(run, jobs, now=lambda: NOW).view["state"] == "ended"


def test_an_end_without_a_final_report_says_why_and_names_an_active_trainer(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status("FAILED", "ended", updated=timedelta(minutes=5))
    checkpoint = {
        "kind": "training",
        "name": "t1s5-ab12",
        "uri": "bt://loops:run7/state/t1s5-ab12",
        "epoch": 1,
        "step": 5,
        "created_in_session": "run7",
    }
    jobs.report(run, 0, checkpoints=[checkpoint])
    jobs.active = True
    _, worker, warnings = synced(run, jobs)
    assert warnings == [
        "Baseten job j1 ended (FAILED) before its final state report: the container ran out "
        "of memory",
        "Loops run run7 may still bill: deactivate it in the Baseten UI",
    ]
    assert jobs.asked == ["run7"]
    assert not worker.running
    saved = run.read_remote()
    assert saved is not None
    assert saved["ended"]["outcome"] == "no_report"
    jobs.error = AssertionError("not read again")
    assert sync_remote(run, jobs, now=lambda: NOW).warnings == warnings


def test_a_log_read_failure_after_the_end_does_not_end_the_run(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status("COMPLETED", "ended", updated=timedelta(minutes=5))
    jobs.report(run, 0, final=True)
    jobs.logs_error = TransientError("the logs returned HTTP 503", source="baseten")
    view, worker, warnings = synced(run, jobs)
    assert (view["state"], worker.running) == ("unknown", True)
    assert warnings == ["could not read the logs of Baseten job j1 (the logs returned HTTP 503)"]
    saved = run.read_remote()
    assert saved is not None
    assert (saved["state"], saved["ended"]) == ("submitted", None)
    jobs.logs_error = None
    view, _, warnings = synced(run, jobs)
    assert (view["state"], warnings) == ("ended", [])
    saved = run.read_remote()
    assert saved is not None
    assert saved["ended"]["outcome"] == "reported"


def test_a_job_the_provider_no_longer_has_ends_the_run(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.error = RemoteJobGone("Baseten has no training job j1", source="baseten")
    jobs.active = None
    view, worker, warnings = synced(run, jobs)
    assert (view["state"], worker.running) == ("ended", False)
    assert warnings == [
        "Baseten has no training job j1, so the run stays as its last state report left it",
        "a Loops trainer may still bill: check the Loops runs in the Baseten UI",
    ]
    saved = run.read_remote()
    assert saved is not None
    assert saved["ended"]["outcome"] == "gone"
    assert sync_remote(run, jobs, now=lambda: NOW).warnings == warnings


def test_an_end_before_any_checkpoint_warns_of_a_trainer_anyway(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status("FAILED", "ended", updated=timedelta(minutes=5))
    jobs.reason = None
    jobs.active = None
    _, _, warnings = synced(run, jobs)
    assert warnings == [
        "Baseten job j1 ended (FAILED) before its final state report",
        "a Loops trainer may still bill: check the Loops runs in the Baseten UI",
    ]


def test_a_busy_lock_skips_the_writes(run: RunDir) -> None:
    before = record(run)
    jobs = FakeJobs()
    jobs.report(run, 0, step=9)
    with run.lock():
        view, _, _ = synced(run, jobs)
    assert view["state"] == "starting"
    assert run.read_remote() == before
    assert run.load_state().phase == "created"


def test_a_final_report_a_busy_lock_kept_out_does_not_end_the_run(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status("COMPLETED", "ended", updated=timedelta(minutes=5))
    jobs.report(run, 0, final=True)
    with run.lock():
        view, worker, warnings = synced(run, jobs)
    assert (view["state"], worker.running, warnings) == ("ending", True, [])
    saved = run.read_remote()
    assert saved is not None
    assert (saved["state"], saved["snapshot"]) == ("submitted", None)
    view, worker, _ = synced(run, jobs)
    assert (view["state"], worker.running) == ("ended", False)
    assert run.load_state().step == 5


def test_a_slower_sync_writes_nothing_over_what_another_one_recorded(run: RunDir) -> None:
    before = record(run)
    jobs = FakeJobs()
    jobs.report(run, 2, step=9)
    newer = {
        **before,
        "state": "ended",
        "snapshot": {"seq": 3, "at": NOW.isoformat(timespec="seconds"), "final": True},
        "ended": {"at": NOW.isoformat(timespec="seconds"), "outcome": "reported"},
    }
    jobs.meanwhile = lambda: run.write_remote(newer)
    view, worker, _ = synced(run, jobs)
    assert (view["state"], worker.running) == ("starting", True)
    assert run.read_remote() == newer
    assert run.load_state().phase == "created"


def test_a_read_failure_keeps_the_last_copy(run: RunDir) -> None:
    before = record(run, synced_at="2026-10-09T11:50:00+00:00")
    jobs = FakeJobs()
    jobs.error = TransientError("the training job lookup returned HTTP 503", source="baseten")
    view, worker, warnings = synced(run, jobs)
    assert view["state"] == "unknown"
    assert worker.running
    assert warnings == [
        "could not read Baseten job j1 (the training job lookup returned HTTP 503); this shows "
        "the last copy from 2026-10-09T11:50:00+00:00"
    ]
    assert run.read_remote() == before


def test_an_auth_failure_propagates(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.error = ProviderAuthError("the key was refused", source="baseten")
    with pytest.raises(ProviderAuthError):
        synced(run, jobs)


def test_a_job_the_submit_did_not_report_is_adopted(run: RunDir) -> None:
    record(run, job_id=None, project={"id": None, "name": f"pup-finetune-{RUN_ID}"})
    jobs = FakeJobs()
    jobs.found = [jobs.job_status("RUNNING", "running", job_id="j7")]
    view, _, warnings = synced(run, jobs)
    assert warnings == ["found Baseten job j7 for this run, which the submit did not report"]
    assert (view["job_id"], view["state"]) == ("j7", "starting")
    saved = run.read_remote()
    assert saved is not None
    assert (saved["job_id"], saved["project"]["id"], saved["logs"]) == ("j7", "p1", "logs j7")


def test_no_job_twenty_minutes_after_the_submit_ends_the_run(run: RunDir) -> None:
    for minutes, state in ((19, "submitting"), (21, "ended")):
        submitted = (NOW - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        record(run, job_id=None, submitted_at=submitted)
        view, worker, warnings = synced(run, FakeJobs())
        assert (view["state"], worker.running) == (state, state != "ended")
    assert warnings == ["no Baseten job was created for this run, so nothing bills"]
    saved = run.read_remote()
    assert saved is not None
    assert saved["ended"]["outcome"] == "not_created"
    assert sync_remote(run, FakeJobs(), now=lambda: NOW).warnings == warnings


def test_no_job_yet_is_still_submitting(run: RunDir) -> None:
    record(run, job_id=None, submitted_at=(NOW - timedelta(minutes=2)).isoformat())
    view, worker, warnings = synced(run, FakeJobs())
    assert (view["state"], worker.running, warnings) == ("submitting", True, [])


def test_a_submit_still_running_holds_the_lock(run: RunDir) -> None:
    record(run, job_id=None)
    jobs = FakeJobs()
    jobs.found = [jobs.job_status("RUNNING", "running", job_id="j7")]
    with run.lock():
        view, _, _ = synced(run, jobs)
    assert view["state"] == "submitting"


def test_two_jobs_to_adopt_are_a_state_error(run: RunDir) -> None:
    record(run, job_id=None)
    jobs = FakeJobs()
    jobs.found = [
        jobs.job_status("RUNNING", "running", job_id="j7"),
        jobs.job_status("RUNNING", "running", job_id="j8"),
    ]
    with pytest.raises(StateError, match="j7, j8"):
        synced(run, jobs)


def test_a_running_job_that_sends_no_report_is_a_warning(run: RunDir) -> None:
    record(run)
    jobs = FakeJobs()
    jobs.status = jobs.job_status("RUNNING", "running", updated=timedelta(minutes=8))
    _, _, warnings = synced(run, jobs)
    assert warnings == ["Baseten job j1 sent no state report for 8m; read its logs: logs j1"]


# --- the submit --------------------------------------------------------------------------


class TerminatedSubmit(FakeJobs):
    """A submit during which the process gets SIGTERM; where it was staged."""

    def __init__(self) -> None:
        super().__init__()
        self.staged: list[Path] = []

    def submit(self, plan: RemotePlan, staged: Path, command: Sequence[str]) -> SubmittedJob:
        self.staged.append(staged)
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
            raise AssertionError("SIGTERM would stop the process at once")
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError("SIGTERM did not stop the submit")


def test_a_sigterm_during_the_submit_stops_it_as_ctrl_c_does(
    run: RunDir, source: SourceTree
) -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers are set on the main thread only")
    before = signal.getsignal(signal.SIGTERM)
    jobs = TerminatedSubmit()
    with pytest.raises(Interrupted, match="the job may exist"):
        submit_remote(run, jobs, plan(), source, mode="train", created_dir=False)
    assert not jobs.staged[0].exists()
    saved = run.read_remote()
    assert saved is not None
    assert saved["state"] == "submitting"
    assert signal.getsignal(signal.SIGTERM) is before
