# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""``train --remote``: a run's training loop in a provider job, and its state copied back.

:func:`submit_remote` stages what the job needs in a private temporary directory
(:func:`stage_run`: pup-finetune's source and lock file from the checkout it runs from, the
run's train and validation rows with their split assignments, and the run files) and
submits it once, through the provider's
:class:`~pup_finetune.providers.base.RemoteJobs`, with :func:`start_command`, which runs
``_worker --report-state`` in the job. ``remote.json`` records the job before the submit, so
a submit whose answer is lost is found again; a submit refused before anything was created
leaves no run files, so the same command can run again.

:func:`sync_remote` (``status``, ``resume`` and ``cancel`` call it first) reads the job and
the state reports in its logs (:mod:`pup_finetune.training.report`), copies the newest one
into the run directory and says where the job stands; its :class:`RemoteSync` carries a
:class:`~pup_finetune.training.worker.WorkerStatus` that stands in for the local worker.
"""

from __future__ import annotations

import importlib.metadata
import shlex
import shutil
import signal
import tempfile
import threading
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any, Final

import pup_finetune
from pup_finetune import next_steps
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.jsonl import write_rows
from pup_finetune.data.schema import split_key
from pup_finetune.data.split import SplitAssignments, SplitName
from pup_finetune.errors import (
    ExternalError,
    FinetuneError,
    Interrupted,
    LockBusy,
    SettingsConflict,
    StateError,
    UsageError,
)
from pup_finetune.output import info
from pup_finetune.providers.base import (
    JobStatus,
    LogLine,
    ProviderAuthError,
    ProviderCapacityError,
    RemoteJobGone,
    RemoteJobRejected,
    RemoteJobs,
    RemotePlan,
)
from pup_finetune.runs import Clock, RunDir, error_json, in_id_order, rows_content_hash, utcnow
from pup_finetune.storage import (
    LOCK_FILE_NAME,
    atomic_write_json,
    ensure_private_dir,
    make_private_dir,
    read_json,
)
from pup_finetune.training.inputs import NEW_RUN_HINT
from pup_finetune.training.report import Snapshot, apply_snapshot, newest_snapshot
from pup_finetune.training.worker import WorkerMode, WorkerStatus

__all__ = [
    "ADOPT_AFTER_S",
    "GRACE_S",
    "QUIET_WARN_S",
    "RemoteSync",
    "SourceTree",
    "ended_sync",
    "job_name",
    "package_source",
    "remote_ended",
    "remote_view",
    "require_locked_renderer",
    "stage_run",
    "start_command",
    "submit_remote",
    "sync_remote",
]

GRACE_S: Final = 120.0
"""How long after a job ended its final state report may still arrive in its logs."""
ADOPT_AFTER_S: Final = 1200.0
"""How long after a submit with no job id a job of the run may still appear: longer than a
provider's submit may run (Baseten's for 15 minutes), as it may outlive the command."""
QUIET_WARN_S: Final = 300.0
"""How long a running job may send no state report before ``status`` warns."""
CLOCK_SLACK: Final = timedelta(seconds=60)
PYTHON: Final = "3.12"
STAGED_SPLITS: Final[tuple[SplitName, ...]] = ("train", "validation")
NOT_A_CHECKOUT: Final = (
    "--remote needs pup-finetune installed from a source checkout, which this installation is not"
)


@dataclass(frozen=True, slots=True)
class SourceTree:
    """The checkout pup-finetune runs from: its ``root``, the ``package`` directory, the
    version and what ``uv.lock`` pins (``locked``: package name to version)."""

    root: Path
    package: Path
    version: str
    locked: Mapping[str, str]


def package_source(package: Path | None = None) -> SourceTree:
    """The checkout the running package (or ``package``) is in; :class:`UsageError` when it
    is not a source checkout of pup-finetune with a ``uv.lock``."""
    package = (package or Path(pup_finetune.__file__).parent).resolve()
    root = package.parents[1]
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        project, lock = {}, {}
    named = (project.get("project") or {}).get("name") == "pup-finetune"
    if not named or not lock or package != (root / "src" / "pup_finetune").resolve():
        raise UsageError(
            NOT_A_CHECKOUT, hint="install it from a checkout with `uv sync --all-extras`"
        )
    locked = {
        str(entry.get("name")): str(entry.get("version"))
        for entry in lock.get("package") or []
        if isinstance(entry, dict)
    }
    return SourceTree(root, package, pup_finetune.__version__, locked)


def require_locked_renderer(source: SourceTree) -> None:
    """Refuse a local ``transformers`` other than the one ``uv.lock`` pins: the job installs
    the locked one, and the renderer's identity names the version."""
    locked = source.locked.get("transformers")
    try:
        local = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        local = None
    if locked is None or local != locked:
        raise UsageError(
            f"--remote needs the transformers version uv.lock pins ({locked}), and this "
            f"installation has {local or 'none'}",
            hint="run `uv sync --all-extras` in the checkout",
        )


def _copy(source: Path, target: Path) -> int:
    ensure_private_dir(target.parent)
    shutil.copyfile(source, target)
    target.chmod(0o600)
    return target.stat().st_size


def stage_run(run: RunDir, dest: Path, *, root: str, source: SourceTree) -> dict[str, int]:
    """Write into ``dest`` what the job of ``run`` needs, its paths under ``root`` in the job:
    ``app/`` (the project files and package, without hidden files, symlinks or caches),
    ``dataset/`` (the kept train and validation rows, their split assignments and split
    settings) and ``run/`` (``run.json`` naming that dataset and its hash, and
    ``state.json``). Directories ``0700``, files ``0600``. Returns the files, bytes and rows
    per split staged."""
    manifest = run.manifest()
    dataset = DatasetDir(Path(str(manifest["dataset_dir"])))
    by_split = dataset.rows_by_split()
    if rows_content_hash(by_split) != manifest.get("dataset_sha256"):
        raise SettingsConflict(
            f"the dataset in {dataset.path} changed since the run in {run.path} was created",
            hint=NEW_RUN_HINT,
        )
    sizes = [
        _copy(source.root / name, dest / "app" / name) for name in ("pyproject.toml", "uv.lock")
    ]
    for path in sorted(source.package.rglob("*")):
        relative = path.relative_to(source.package)
        hidden = any(part.startswith(".") or part == "__pycache__" for part in relative.parts)
        if path.is_file() and not path.is_symlink() and not hidden and path.suffix != ".pyc":
            sizes.append(_copy(path, dest / "app" / "src" / "pup_finetune" / relative))
    staged: dict[SplitName, list[Any]] = {"train": [], "validation": [], "test": []}
    for split in STAGED_SPLITS:
        staged[split] = in_id_order(by_split[split])
    copy = DatasetDir(make_private_dir(dest / "dataset"))
    atomic_write_json(copy.settings_path, {"split": dataset.settings()["split"]})
    assignments = {split_key(row): split for split in STAGED_SPLITS for row in staged[split]}
    SplitAssignments(dataset.split_params(), assignments).save(copy.splits_path)
    write_rows(copy.rows_path, in_id_order([*staged["train"], *staged["validation"]]))
    digest = rows_content_hash(staged)
    if rows_content_hash(copy.rows_by_split()) != digest:
        raise StateError(f"the rows staged in {copy.path} do not read back as written")
    run_dir = make_private_dir(dest / "run")
    atomic_write_json(
        run_dir / run.run_path.name,
        {**manifest, "dataset_dir": f"{root}/dataset", "dataset_sha256": digest},
    )
    atomic_write_json(run_dir / run.state_path.name, read_json(run.state_path))
    for path in (copy.settings_path, copy.splits_path, copy.rows_path, *run_dir.iterdir()):
        sizes.append(path.stat().st_size)
    return {
        "files": len(sizes),
        "bytes": sum(sizes),
        "train": len(staged["train"]),
        "validation": len(staged["validation"]),
    }


def start_command(plan: RemotePlan, mode: WorkerMode) -> list[str]:
    """The job's command: install the locked dependencies, then run the worker on the staged
    run, reporting its state on stdout. Fixed: nothing in it comes from the user."""
    root = shlex.quote(plan.root)
    return [
        "/bin/sh",
        "-c",
        f"cd {root} && uv sync --quiet --frozen --no-dev --extra {plan.extra} "
        f"--no-install-project --project app --python {PYTHON}"
        f" && PYTHONPATH={root}/app/src PYTHONUNBUFFERED=1 exec app/.venv/bin/python -P -m "
        f"pup_finetune _worker {root}/run --mode {mode} --report-state",
    ]


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _parsed_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def job_name(record: Mapping[str, Any]) -> str:
    """``Baseten job J`` (``a Baseten job`` before its id is known), from the backend."""
    backend = str(record.get("backend") or "remote job")
    kind = backend[:1].upper() + backend[1:]
    job_id = record.get("job_id")
    return f"{kind} {job_id}" if job_id else f"a {kind}"


def _roll_back(run: RunDir, created_dir: bool) -> None:
    for name in (run.remote_path, run.state_path, run.run_path, run.path / LOCK_FILE_NAME):
        name.unlink(missing_ok=True)
    if created_dir:
        run.path.rmdir()


def _on_terminate(signum: int, frame: FrameType | None) -> None:
    signal.signal(signum, signal.SIG_IGN)  # a second one must not stop the cleanup
    raise KeyboardInterrupt


@contextmanager
def _terminate_interrupts() -> Iterator[None]:
    """Make SIGTERM and SIGHUP (unless ignored) stop the submit as Ctrl-C does, so the
    provider's process is stopped and the staged files removed; main thread only."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {
        signum: signal.signal(signum, _on_terminate)
        for signum in (signal.SIGTERM, signal.SIGHUP)
        if signal.getsignal(signum) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def submit_remote(
    run: RunDir,
    jobs: RemoteJobs,
    plan: RemotePlan,
    source: SourceTree,
    *,
    mode: WorkerMode,
    created_dir: bool,
    now: Clock = utcnow,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Stage ``run`` and submit it as a job ``plan`` describes; return ``remote.json`` and
    what was uploaded.

    The run's lock is held from the record to the answer and what follows it, so nothing
    trains or syncs the run meanwhile. A refusal before anything was created (and a failure
    to stage) removes the run files again (the directory too when ``created_dir``) and
    raises it. Any other failure may have created the job: the record is kept for ``status``
    to find it. SIGTERM and SIGHUP stop the submit as Ctrl-C does.
    """
    status = next_steps.command("status", run.path)
    record: dict[str, Any] = {
        "backend": plan.backend,
        "state": "submitting",
        "team": plan.team,
        "project": {"id": None, "name": plan.project},
        "job_id": None,
        "mode": mode,
        "compute": dict(plan.compute),
        "api_key_secret": plan.secret,
        **{f"{tool}_version": version for tool, version in plan.tool_versions.items()},
        "submitted_at": _iso(now()),
        "job_status": None,
        "synced_at": None,
        "snapshot": None,
        "ended": None,
        "error": None,
        "logs": None,
    }
    submitting = False
    with _terminate_interrupts(), run.lock():
        try:
            run.write_remote(record)
            with tempfile.TemporaryDirectory(prefix="pup-finetune-") as tmp:
                staged = Path(tmp)
                upload = stage_run(run, staged, root=plan.root, source=source)
                info(
                    f"submitting run {run.run_id} to {job_name(record)} in team {plan.team}; "
                    f"uploading {upload['files']} files, this can take a few minutes"
                )
                submitting = True
                job = jobs.submit(plan, staged, start_command(plan, mode))
            record.update(
                state="submitted",
                job_id=job.job_id,
                project={"id": job.project_id, "name": job.project},
                compute=dict(job.compute),
                logs=jobs.logs_command(job.job_id),
            )
            run.write_remote(record)
        except (RemoteJobRejected, ProviderAuthError, ProviderCapacityError):
            _roll_back(run, created_dir)
            raise
        except KeyboardInterrupt as exc:
            if not submitting:
                _roll_back(run, created_dir)
                raise
            raise Interrupted(
                f"the submit of run {run.run_id} stopped (Ctrl-C); the job may exist",
                hint=f"`{status}` looks for the job",
                interrupt=exc,
            ) from None
        except BaseException as exc:
            if not submitting:
                _roll_back(run, created_dir)
                raise
            record["error"] = error_json(exc)
            run.write_remote(record)
            if isinstance(exc, FinetuneError):
                found = f"`{status}` looks for the job"
                exc.hint = f"{exc.hint}; {found}" if exc.hint else found
            raise
    return record, {"files": upload["files"], "bytes": upload["bytes"]}


# --- sync ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RemoteSync:
    """Where a remote run's job stands: ``status``'s ``remote`` view, the worker that stands
    in for a local one (``None``: use the local one), and what to warn about."""

    view: dict[str, Any]
    worker: WorkerStatus | None
    warnings: list[str]


def remote_ended(record: Mapping[str, Any] | None) -> bool:
    """Whether no remote job is left to wait for: no record, or its job ended."""
    return record is None or record.get("state") == "ended"


def ended_sync(record: Mapping[str, Any]) -> RemoteSync:
    """The sync of a run whose job ended: its recorded view, the local worker, and the
    warnings its end gave, which every later sync repeats."""
    ended = record.get("ended")
    warnings = ended.get("warnings") if isinstance(ended, Mapping) else None
    return RemoteSync(remote_view(record), None, [str(text) for text in warnings or []])


def remote_view(
    record: Mapping[str, Any], *, state: str | None = None, report_age_s: float | None = None
) -> dict[str, Any]:
    """``remote.json`` as ``status`` shows it, in ``state`` (by default, the recorded one)."""
    project = record.get("project")
    recorded = record.get("state")
    return {
        "backend": record.get("backend"),
        "team": record.get("team"),
        "project": project.get("name") if isinstance(project, Mapping) else project,
        "job_id": record.get("job_id"),
        "state": state or (recorded if recorded in ("submitting", "ended") else "unknown"),
        "job_status": record.get("job_status"),
        "synced_at": record.get("synced_at"),
        "report_age_s": None if report_age_s is None else round(report_age_s),
        "logs": record.get("logs"),
    }


def _heartbeat_age(run: RunDir, current: datetime) -> float | None:
    try:
        beat = _parsed_time(run.load_state().heartbeat_at)
    except FinetuneError:
        return None
    return None if beat is None else max(0.0, round((current - beat).total_seconds(), 1))


def _stand_in(
    run: RunDir,
    current: datetime,
    *,
    running: bool,
    starting: bool = False,
    error: dict[str, Any] | None = None,
) -> WorkerStatus:
    """The worker a remote job stands for; never ``stale``, so nothing proposes ``cancel``."""
    return WorkerStatus(
        pid=None,
        alive=running,
        heartbeat_age_s=_heartbeat_age(run, current),
        stale=False,
        starting=starting,
        running=running,
        error=error,
    )


def _seq(record: Mapping[str, Any]) -> int:
    snapshot = record.get("snapshot")
    return int(snapshot.get("seq", -1)) if isinstance(snapshot, Mapping) else -1


def _behind(record: Mapping[str, Any], stored: Mapping[str, Any]) -> bool:
    """Whether ``stored`` (``remote.json`` now) knows more than ``record``, read before it:
    the job's end, its id or a newer state report."""
    return (
        (stored.get("state") == "ended" and record.get("state") != "ended")
        or (bool(stored.get("job_id")) and not record.get("job_id"))
        or _seq(stored) > _seq(record)
    )


def _save(run: RunDir, record: Mapping[str, Any]) -> None:
    """Write ``record`` under the run's lock; nothing while another process holds it, or
    when ``remote.json`` is gone or already knows more."""
    with suppress(LockBusy), run.lock():
        stored = run.read_remote()
        if stored is not None and not _behind(record, stored):
            run.write_remote(record)


def _since(moment: datetime | None, current: datetime) -> float | None:
    return None if moment is None else (current - moment).total_seconds()


def _adopt(
    run: RunDir, jobs: RemoteJobs, record: dict[str, Any], current: datetime
) -> tuple[str | None, list[str]]:
    """Look for the job of a submit that reported none; the state to show while there is
    none to adopt (``None`` once one was adopted), and what to warn about."""
    if run.lock().is_held():
        return "submitting", []
    submitted = _parsed_time(record.get("submitted_at")) or current
    project = dict(record.get("project") or {})
    found = jobs.find(
        team=str(record.get("team")),
        project=str(project.get("name")),
        after=submitted - CLOCK_SLACK,
    )
    if len(found) > 1:
        raise StateError(
            f"several jobs of project {project.get('name')} may be this run's: "
            f"{', '.join(job.job_id for job in found)}",
            hint=f"check them in the provider's UI, and record the run's job in {run.remote_path}",
        )
    if found:
        job = found[0]
        record.update(
            state="submitted",
            job_id=job.job_id,
            project={**project, "id": job.project_id},
            logs=jobs.logs_command(job.job_id),
        )
        return None, [f"found {job_name(record)} for this run, which the submit did not report"]
    if (current - submitted).total_seconds() < ADOPT_AFTER_S:
        return "submitting", []
    ending = [
        f"no {job_name(record).removeprefix('a ')} was created for this run, so nothing bills"
    ]
    record.update(
        state="ended", ended={"at": _iso(current), "outcome": "not_created", "warnings": ending}
    )
    return "ended", list(ending)


def _read_job(jobs: RemoteJobs, record: dict[str, Any], job_id: str) -> JobStatus:
    """The job now; found in its project by id when the submit did not report the project's."""
    project = dict(record.get("project") or {})
    if project.get("id"):
        return jobs.job(project_id=str(project["id"]), job_id=job_id)
    found = jobs.find(
        team=str(record.get("team")),
        project=str(project.get("name")),
        after=datetime.min.replace(tzinfo=UTC),
    )
    for job in found:
        if job.job_id == job_id:
            record["project"] = {**project, "id": job.project_id}
            return job
    raise ExternalError(f"no job {job_id} in project {project.get('name')}", source="remote")


def _newest_report(
    run: RunDir, jobs: RemoteJobs, job: JobStatus, since: datetime | None, current: datetime
) -> Snapshot | None:
    lines: list[LogLine] = []
    for page in jobs.output_pages(job, since=since, now=current):
        lines += page
        if (found := newest_snapshot(lines, run.run_id)) is not None:
            return found
    return None


def _apply(run: RunDir, snapshot: Snapshot, seen: Mapping[str, Any]) -> bool:
    """Copy ``snapshot`` into ``run`` and record it in ``remote.json`` as ``seen``, under the
    run's lock; whether the run now holds it (``False`` while another process holds the lock
    or once ``remote.json`` records a newer one). Raises :class:`StateError`, writing
    nothing, for a snapshot that is not a run's."""
    try:
        with run.lock():
            stored = run.read_remote()
            if stored is None or _seq(stored) > snapshot.seq:
                return False
            if _seq(stored) < snapshot.seq:
                apply_snapshot(run, snapshot)
                run.write_remote({**stored, "snapshot": dict(seen)})
    except LockBusy:
        return False
    return True


def _trainer_warning(run: RunDir, jobs: RemoteJobs) -> str | None:
    """What may still bill after a job that ended without its final report."""
    try:
        checkpoints = run.load_state().checkpoints
    except FinetuneError:
        checkpoints = []
    session = checkpoints[-1].get("created_in_session") if checkpoints else None
    active = jobs.trainer_active(str(session)) if session else None
    if active:
        return f"Loops run {session} may still bill: deactivate it in the Baseten UI"
    if active is None:
        return "a Loops trainer may still bill: check the Loops runs in the Baseten UI"
    return None


def sync_remote(run: RunDir, jobs: RemoteJobs, *, now: Clock = utcnow) -> RemoteSync:
    """Read ``run``'s job and its state reports, copy the newest report into the run
    directory, and say where the job stands (see the module docstring).

    A failed read is a warning and the state ``unknown``; :class:`ProviderAuthError`
    propagates, and a job the provider no longer has ends the run (``gone``). Nothing is
    written while another process holds the run's lock.
    """
    record = run.read_remote()
    if record is None:
        raise StateError(f"{run.path} has no {run.remote_path.name}")
    if remote_ended(record):
        return ended_sync(record)
    current = now()
    warnings: list[str] = []
    if not record.get("job_id"):
        try:
            waiting, warnings = _adopt(run, jobs, record, current)
        except ProviderAuthError:
            raise
        except ExternalError as exc:
            waiting = "unknown"
            warnings = [f"could not look for {job_name(record)} ({exc.message})"]
        if waiting is not None:
            if waiting != "unknown":
                record["synced_at"] = _iso(current)
                _save(run, record)
            running = waiting != "ended"
            worker = _stand_in(run, current, running=running, starting=waiting == "submitting")
            return RemoteSync(remote_view(record, state=waiting), worker, warnings)
    name = job_name(record)
    try:
        job = _read_job(jobs, record, str(record["job_id"]))
    except ProviderAuthError:
        raise
    except RemoteJobGone as exc:
        gone = [f"{exc.message}, so the run stays as its last state report left it"]
        if (trainer := _trainer_warning(run, jobs)) is not None:
            gone.append(trainer)
        record.update(
            state="ended",
            ended={"at": _iso(current), "outcome": "gone", "warnings": gone},
            synced_at=_iso(current),
        )
        _save(run, record)
        worker = _stand_in(run, current, running=False)
        return RemoteSync(remote_view(record), worker, [*warnings, *gone])
    except ExternalError as exc:
        warnings.append(
            f"could not read {name} ({exc.message}); this shows the last copy from "
            f"{record.get('synced_at') or record.get('submitted_at')}"
        )
        worker = _stand_in(run, current, running=True)
        return RemoteSync(remote_view(record, state="unknown"), worker, warnings)
    seen: dict[str, Any] = dict(record.get("snapshot") or {})
    busy, logs_read = False, True
    try:
        snapshot = _newest_report(run, jobs, job, _parsed_time(seen.get("at")), current)
    except ProviderAuthError:
        raise
    except ExternalError as exc:
        warnings.append(f"could not read the logs of {name} ({exc.message})")
        snapshot, logs_read = None, False
    if snapshot is not None and snapshot.seq > int(seen.get("seq", -1)):
        fresh = {
            "seq": snapshot.seq,
            "at": snapshot.at,
            "final": snapshot.final,
            "log_ns": snapshot.log_ns,
            "exit_error": snapshot.exit_error,
        }
        try:
            busy = not _apply(run, snapshot, fresh)
        except StateError as exc:
            warnings.append(f"the state report of {name} is not a run's ({exc.message})")
        else:
            if not busy:
                seen = fresh
                record["snapshot"] = seen
    final = seen.get("final") is True
    error = seen.get("exit_error") if final else None
    log_ns = seen.get("log_ns")
    age = None if not isinstance(log_ns, int) else max(0.0, current.timestamp() - log_ns / 1e9)
    starting = False
    ending: list[str] = []
    if job.state == "queued":
        state, running, starting = "queued", True, True
    elif job.state == "running":
        state, running, starting = ("running", True, False) if seen else ("starting", True, True)
        quiet = age if age is not None else _since(job.updated_at or job.created_at, current)
        if quiet is not None and quiet > QUIET_WARN_S:
            warnings.append(
                f"{name} sent no state report for {round(quiet / 60)}m; read its logs: "
                f"{record.get('logs')}"
            )
    elif job.state == "ended" and final:
        state, running = "ended", False
    elif job.state == "ended" and busy:
        state, running = "ending", True
    elif job.state == "ended" and not logs_read:
        state, running = "unknown", True
    elif job.state == "ended":
        ended_for = _since(job.updated_at or job.created_at, current)
        grace = ended_for is not None and ended_for < GRACE_S
        state, running = ("ending", True) if grace else ("ended", False)
        if state == "ended":
            try:
                reason = jobs.failure(job, now=current)
            except ExternalError:
                reason = None
            ending.append(
                f"{name} ended ({job.status}) before its final state report"
                + (f": {reason}" if reason else "")
            )
            if (trainer := _trainer_warning(run, jobs)) is not None:
                ending.append(trainer)
            warnings += ending
    else:
        state, running = "unknown", True
        warnings.append(
            f"{name} has status {job.status}, which pup-finetune does not know; status keeps "
            "watching it"
        )
    if state == "ended":
        outcome = "reported" if final else "no_report"
        ended = {"at": _iso(job.updated_at or current), "outcome": outcome, "warnings": ending}
        record.update(state="ended", ended=ended)
    record.update(job_status=job.status, synced_at=_iso(current))
    if not busy:
        _save(run, record)
    worker = _stand_in(run, current, running=running, starting=starting, error=error)
    return RemoteSync(remote_view(record, state=state, report_age_s=age), worker, warnings)
