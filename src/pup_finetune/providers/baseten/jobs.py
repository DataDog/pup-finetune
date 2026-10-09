# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten training jobs for ``train --remote``: the plan, the submit and the reads after it.

This is the only module that runs the ``baseten`` CLI, and it never imports a ``baseten``
package. :class:`BasetenJobs` is the provider's
:class:`~pup_finetune.providers.base.RemoteJobs`:

- :meth:`~BasetenJobs.plan` finds the CLI and ``uv``, picks the team (``--team``, else the
  key's only team, else its default one), checks that the team has the secret
  :data:`API_KEY_SECRET` the job reads its API key from, and refuses a project of the run's
  name that has jobs (an empty one, which a failed upload leaves, is reused). Read only.
- :meth:`~BasetenJobs.submit` runs ``baseten loops exec`` once (:func:`exec_argv`), with the
  key in the child's environment only (:func:`exec_env`), never on its command line or in a
  file. A refusal, an auth or a capacity error means nothing was created; any other failure
  may have created the job, and says so.
- :meth:`~BasetenJobs.find`, :meth:`~BasetenJobs.job`, :meth:`~BasetenJobs.output_pages`,
  :meth:`~BasetenJobs.failure` and :meth:`~BasetenJobs.trainer_active` read the job, its
  logs and its Loops trainer over :class:`~.client.BasetenHTTP`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import SecretStr

from pup_finetune.credentials import login_hint
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.http_common import bearer_key
from pup_finetune.providers.base import (
    JobState,
    JobStatus,
    LogLine,
    ProviderAuthError,
    ProviderCapacityError,
    RemoteJobGone,
    RemoteJobRejected,
    RemotePlan,
    SubmittedJob,
)
from pup_finetune.providers.baseten import CREDENTIALS, ENV_VAR
from pup_finetune.providers.baseten.client import (
    _AUTH_HINT,
    _CREATED_HINT,
    SOURCE,
    BasetenHTTP,
    _id,
)
from pup_finetune.redaction import redact
from pup_finetune.skill import npx_env
from pup_finetune.storage import atomic_write_text

__all__ = [
    "API_KEY_SECRET",
    "BASETEN_CLI_ENV",
    "JOB_ROOT",
    "TRUSS_VERSION",
    "BasetenJobs",
    "CliRunner",
    "CompletedCli",
    "exec_argv",
    "exec_env",
    "job_state",
    "run_cli",
]

BASETEN_CLI_ENV: Final = "PUP_FINETUNE_BASETEN_CLI"
TRUSS_VERSION: Final = "0.18.33"
"""The truss version the job is built with: the first whose ``--no-ssh`` opens no session."""
API_KEY_SECRET: Final = "pup_finetune_api_key"  # noqa: S105 - the name of a secret, not one
JOB_ROOT: Final = "/b10/workspace"
"""Where truss puts the upload in the job, and the command runs."""
CPU_COUNT: Final = 16
MEMORY: Final = "64Gi"
PROJECT_PREFIX: Final = "pup-finetune-"
EXEC_TIMEOUT_S: Final = 900.0
BACKEND: Final = "baseten job"
EXTRA: Final = "baseten"
TRUSS_IGNORE: Final = "__pycache__/\n*.pyc\n.venv/\n"
LOG_PAGE: Final = 1000
MAX_LOG_PAGES: Final = 5
LOG_RETENTION: Final = timedelta(days=7)
"""The longest log window Baseten answers."""
CLOCK_SLACK: Final = timedelta(seconds=60)
DETAIL_CHARS: Final = 300
NOTHING_CREATED: Final = "nothing that bills was created"
BILLING: Final = (
    f"Baseten bills the CPU job that runs the training loop ({CPU_COUNT} vCPUs, "
    f"{MEMORY.removesuffix('Gi')} GiB) for the time it runs, at its own rates, which "
    "pup-finetune does not know: from the job's start until training ends"
)

QUEUED: Final = frozenset(
    {
        "TRAINING_JOB_PENDING",
        "TRAINING_JOB_CREATED",
        "TRAINING_JOB_DEPLOYING",
        "TRAINING_JOB_QUEUED",
    }
)
RUNNING: Final = frozenset({"TRAINING_JOB_RUNNING"})
ENDED: Final = frozenset(
    {
        "TRAINING_JOB_COMPLETED",
        "TRAINING_JOB_FAILED",
        "TRAINING_JOB_DEPLOY_FAILED",
        "TRAINING_JOB_STOPPED",
        "TRAINING_JOB_PREEMPTED",
    }
)

_DROPPED_ENV: Final = (
    "BASETEN_PROFILE",
    "BASETEN_REMOTE_URL",
    "BASETEN_MANAGEMENT_API_URL_OVERRIDE",
    "BASETEN_MODEL_API",
    "BASETEN_TRUSS_VERSION",
    "BASETEN_TRUSS_EXECUTABLE",
    "BASETEN_TRUSS_AUTH_REMOTE_URL",
    "BASETEN_TRUSS_AUTH_API_KEY",
    "PUP_OUTPUT",
    "PUP_FILTER",
)
_DROPPED_PREFIXES: Final = ("LOOPS_", "INTERNAL_LOOPS_")
_REFUSED_EXIT: Final = frozenset({2, 4, 5})
_AUTH_EXIT: Final = 3
_KILL_WAIT_S: Final = 5.0


@dataclass(frozen=True, slots=True)
class CompletedCli:
    """How a CLI run ended: its exit code and output."""

    returncode: int
    stdout: str
    stderr: str


class CliRunner(Protocol):
    """Runs a CLI to completion, as :func:`run_cli` does; tests pass a fake."""

    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float
    ) -> CompletedCli: ...


def run_cli(argv: Sequence[str], *, env: Mapping[str, str], timeout: float) -> CompletedCli:
    """Run ``argv`` with ``env`` in a new session, no stdin, and its output captured.

    Raises ``OSError`` when it cannot start and :class:`subprocess.TimeoutExpired` after
    ``timeout`` seconds, once its process group is killed. A Ctrl-C sends the group SIGTERM,
    then SIGKILL after 5 s (at once on a second Ctrl-C), and is raised again.
    """
    process = subprocess.Popen(  # noqa: S603 - the configured CLI, no shell
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.communicate()
        raise
    except KeyboardInterrupt:
        _signal_group(process, signal.SIGTERM)
        try:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=_KILL_WAIT_S)
        finally:
            _signal_group(process, signal.SIGKILL)
        raise
    return CompletedCli(process.returncode, stdout, stderr)


def _signal_group(process: subprocess.Popen[str], signum: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signum)


def exec_env(environ: Mapping[str, str], key: SecretStr) -> dict[str, str]:
    """The ``baseten`` CLI's environment: ``environ`` without secrets or overrides that would
    send the job elsewhere, plus the API key."""
    secret = bearer_key(key, label="the Baseten API key", hint=login_hint(CREDENTIALS))
    env = {
        name: value
        for name, value in npx_env(environ).items()
        if name not in _DROPPED_ENV and not name.startswith(_DROPPED_PREFIXES)
    }
    return {**env, ENV_VAR: secret, "TRUSS_NO_UPDATE_CHECK": "1", "NO_COLOR": "1"}


def exec_argv(
    cli: str, *, staged: Path, team: str, project: str, command: Sequence[str]
) -> list[str]:
    """The ``baseten loops exec`` command that uploads ``staged`` and runs ``command``."""
    return [
        *(cli, "loops", "exec", "--output", "json", "--dir", str(staged)),
        *("--team", team, "--project-name", project),
        *("--cpu-count", str(CPU_COUNT), "--memory", MEMORY),
        *("--with-uv", "--no-ssh", "--no-api-key", "--secret", f"{ENV_VAR}={API_KEY_SECRET}"),
        *("--truss-version", TRUSS_VERSION, "--", *command),
    ]


def job_state(status: str) -> JobState:
    """What a training job's ``current_status`` means; ``unknown`` for any other."""
    if status in QUEUED:
        return "queued"
    if status in RUNNING:
        return "running"
    return "ended" if status in ENDED else "unknown"


class BasetenJobs:
    """Baseten training jobs that run a run's training loop; see the module docstring.

    ``http`` builds a client (the key resolved on first use), ``key`` resolves the key the
    CLI gets, ``environ`` is the environment the CLI's is made from, ``runner`` runs it
    (:func:`run_cli`), ``which`` finds executables and ``root`` is the job's working directory.
    """

    def __init__(
        self,
        *,
        http: Callable[[], BasetenHTTP],
        key: Callable[[], SecretStr],
        environ: Mapping[str, str],
        runner: CliRunner,
        which: Callable[[str], str | None],
        root: str = JOB_ROOT,
    ) -> None:
        self._http = http
        self._key = key
        self._environ = environ
        self._runner = runner
        self._which = which
        self._root = root

    def plan(self, *, run_id: str, team: str | None) -> RemotePlan:
        cli = self._cli()
        project = f"{PROJECT_PREFIX}{run_id}"
        with self._http() as http:
            name, team_id, source = _pick_team(http.teams(), team)
            if API_KEY_SECRET not in http.team_secret_names(team_id):
                raise UsageError(
                    f"Baseten team {name!r} has no secret {API_KEY_SECRET}, which the job reads "
                    "its API key from",
                    hint=f"create the secret in team {name!r}, holding a Baseten API key of this "
                    "workspace, in the Baseten UI or with `baseten org secret set --name "
                    f"{API_KEY_SECRET} --team {shlex.quote(name)}`, which reads the value from "
                    f"stdin; every job of team {name!r} can read it",
                )
            for found in http.training_projects():
                if (found.get("name"), found.get("team_name")) != (project, name):
                    continue
                if http.training_jobs(_id(found.get("id"), "project")):
                    raise UsageError(
                        f"Baseten team {name!r} already has project {project} with jobs",
                        hint="pass another --run-id",
                    )
        return RemotePlan(
            backend=BACKEND,
            team=name,
            team_source=source,
            project=project,
            root=self._root,
            extra=EXTRA,
            cli=cli,
            compute={"cpu_count": CPU_COUNT, "memory": MEMORY, "accelerator": None},
            secret=API_KEY_SECRET,
            tool_versions={"truss": TRUSS_VERSION},
            billing=BILLING,
            warnings=(
                "cancel cannot stop a remote run yet: stop its job in the Baseten UI, and "
                "deactivate its Loops trainer there too",
                f"the job reads a Baseten API key from the team secret {API_KEY_SECRET}; "
                "deploy, evaluate and a local resume use this machine's key, so both keys must "
                "belong to the same Baseten workspace",
                "the job downloads its Python packages from PyPI and the tokenizer from "
                "Hugging Face",
            ),
        )

    def submit(self, plan: RemotePlan, staged: Path, command: Sequence[str]) -> SubmittedJob:
        atomic_write_text(staged / ".truss_ignore", TRUSS_IGNORE)
        argv = exec_argv(
            plan.cli, staged=staged, team=plan.team, project=plan.project, command=command
        )
        env = exec_env(self._environ, self._key())
        try:
            done = self._runner(argv, env=env, timeout=EXEC_TIMEOUT_S)
        except OSError as exc:
            raise RemoteJobRejected(
                f"could not run {plan.cli} ({_detail(f'{type(exc).__name__}: {exc}')})",
                source=SOURCE,
                hint=NOTHING_CREATED,
            ) from exc
        except subprocess.TimeoutExpired:
            raise ExternalError(
                f"baseten loops exec did not finish within {EXEC_TIMEOUT_S:g} s",
                source=SOURCE,
                hint=_CREATED_HINT,
            ) from None
        if done.returncode != 0:
            raise _exec_error(done)
        answer = _last_json(done.stdout)
        try:
            job_id = _id(answer.get("job_id"), "job")
        except ExternalError:
            raise ExternalError(
                "baseten loops exec did not report the job it created",
                source=SOURCE,
                hint=_CREATED_HINT,
            ) from None
        found = answer.get("project")
        project = found if isinstance(found, dict) else {}
        compute = answer.get("compute")
        return SubmittedJob(
            job_id=job_id,
            project_id=_optional_id(project.get("id")),
            project=str(project.get("name") or plan.project),
            compute=compute if isinstance(compute, dict) else dict(plan.compute),
        )

    def find(self, *, team: str, project: str, after: datetime) -> list[JobStatus]:
        found: list[JobStatus] = []
        with self._http() as http:
            for record in http.training_projects():
                if (record.get("name"), record.get("team_name")) != (project, team):
                    continue
                project_id = _id(record.get("id"), "project")
                for job in http.training_jobs(project_id):
                    status = _job_status(job, project_id)
                    if status.created_at is None or status.created_at >= after:
                        found.append(status)
        return found

    def job(self, *, project_id: str, job_id: str) -> JobStatus:
        with self._http() as http:
            record = http.training_job(project_id, job_id)
        if record is None:
            raise RemoteJobGone(f"Baseten has no training job {job_id}", source=SOURCE)
        return _job_status(record, project_id)

    def output_pages(
        self, job: JobStatus, *, since: datetime | None, now: datetime
    ) -> Iterator[list[LogLine]]:
        """At most :data:`MAX_LOG_PAGES` pages of the job's container lines, newest first;
        each page ends at the oldest millisecond of the one before, without the lines both
        hold."""
        start_ms = _ms(_window_start(job, since, now))
        end_ms = _ms(now + CLOCK_SLACK)
        previous: set[tuple[object, object, object]] = set()
        for _ in range(MAX_LOG_PAGES):
            with self._http() as http:
                page = http.training_job_logs(
                    job.project_id, job.job_id, start_ms=start_ms, end_ms=end_ms, limit=LOG_PAGE
                )
            keys = [_line_key(line) for line in page]
            yield [
                LogLine(stamp, str(line.get("message") or ""))
                for line, key in zip(page, keys, strict=True)
                if key not in previous
                and line.get("replica") is not None
                and (stamp := _stamp(line)) is not None
            ]
            stamps = [stamp for line in page if (stamp := _stamp(line)) is not None]
            oldest_ms = min(stamps) // 1_000_000 if stamps else end_ms
            if len(page) < LOG_PAGE or oldest_ms >= end_ms:
                return
            end_ms, previous = oldest_ms, set(keys)

    def failure(self, job: JobStatus, *, now: datetime) -> str | None:
        """The newest error the platform logged for the job (failed jobs have no
        ``error_message``)."""
        with self._http() as http:
            page = http.training_job_logs(
                job.project_id,
                job.job_id,
                start_ms=_ms(_window_start(job, None, now)),
                end_ms=_ms(now + CLOCK_SLACK),
                limit=1,
                min_level="ERROR",
            )
        platform = [_detail(line.get("message")) for line in page if line.get("replica") is None]
        return platform[0] or None if platform else None

    def trainer_active(self, session_id: str) -> bool | None:
        try:
            with self._http() as http:
                record = http.get_run(session_id)
        except ExternalError:
            return None
        if record is None:
            return False
        status = record.get("status")
        name = status.get("name") if isinstance(status, dict) else status
        return name is not None and str(name).upper() != "INACTIVE"

    def logs_command(self, job_id: str) -> str:
        return f"baseten train job logs --job-id {shlex.quote(job_id)}"

    def _cli(self) -> str:
        """The ``baseten`` CLI (``PUP_FINETUNE_BASETEN_CLI``, else on ``PATH``), once ``uv``
        is found too."""
        configured = self._environ.get(BASETEN_CLI_ENV) or None
        cli = self._which(configured or "baseten")
        if cli is None and configured:
            raise UsageError(
                f"{BASETEN_CLI_ENV} names {configured!r}, which is not an executable",
                hint=f"set {BASETEN_CLI_ENV} to the path of the baseten CLI, or unset it",
            )
        if cli is None:
            raise UsageError(
                "--remote needs the baseten CLI, which is not on PATH",
                hint=f"install Baseten's CLI, or set {BASETEN_CLI_ENV} to its path",
            )
        if self._which("uv") is None:
            raise UsageError("--remote needs uv, which is not on PATH", hint="install uv")
        return cli


def _pick_team(teams: Sequence[Mapping[str, Any]], given: str | None) -> tuple[str, str, str]:
    """The team's name and id, and how it was chosen."""
    names = [str(team.get("name")) for team in teams]
    shown = ", ".join(repr(name) for name in names)
    hint = f"pass --team with one of: {shown}" if names else "check the key's Baseten workspace"
    if given is not None:
        found = [team for team in teams if team.get("name") == given]
        if not found:
            raise UsageError(f"this Baseten API key sees no team named {given!r}", hint=hint)
        return given, _id(found[0].get("id"), "team"), "--team"
    if len(teams) == 1:
        return names[0], _id(teams[0].get("id"), "team"), "your only team"
    defaults = [team for team in teams if team.get("default") is True]
    if len(defaults) == 1:
        return str(defaults[0].get("name")), _id(defaults[0].get("id"), "team"), "your default team"
    if not teams:
        raise UsageError("this Baseten API key sees no team", hint=hint)
    raise UsageError("this Baseten API key sees several teams and none is the default", hint=hint)


def _exec_error(done: CompletedCli) -> ExternalError:
    """The error for a failed ``baseten loops exec``, from its exit code and the HTTP status
    in its last JSON document (the Go CLI's envelope, or truss's own error)."""
    answer = _last_json(done.stdout)
    found = answer.get("error")
    error = found if isinstance(found, dict) else {}
    code = error.get("api_status_code", error.get("status_code"))
    status = code if isinstance(code, int) and not isinstance(code, bool) else None
    lines = [line for line in done.stderr.splitlines() if line.strip()]
    detail = _detail(error.get("message")) or (_detail(lines[-1]) if lines else "")
    message = f"baseten loops exec failed (exit {done.returncode}"
    message += f", HTTP {status})" if status is not None else ")"
    message += f": {detail}" if detail else ""
    if done.returncode == _AUTH_EXIT or status in (401, 403):
        return ProviderAuthError(message, source=SOURCE, hint=f"{NOTHING_CREATED}; {_AUTH_HINT}")
    if status == 429:
        return ProviderCapacityError(message, source=SOURCE, hint=f"{NOTHING_CREATED}; retry later")
    refused = status is not None and 400 <= status < 500 and status != 408
    if done.returncode in _REFUSED_EXIT or refused:
        return RemoteJobRejected(message, source=SOURCE, hint=NOTHING_CREATED)
    return ExternalError(message, source=SOURCE, hint=_CREATED_HINT)


def _last_json(text: str) -> dict[str, Any]:
    """The last JSON object in ``text``; empty when there is none."""
    decoder = json.JSONDecoder()
    found: dict[str, Any] = {}
    index = text.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            end = index + 1
        else:
            if isinstance(value, dict):
                found = value
        index = text.find("{", end)
    return found


def _detail(value: object) -> str:
    """``value`` as error text shows it: redacted, at most :data:`DETAIL_CHARS` characters."""
    return redact(str(value)).strip()[:DETAIL_CHARS] if value else ""


def _optional_id(value: object) -> str | None:
    try:
        return _id(value, "project")
    except ExternalError:
        return None


def _time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _job_status(record: Mapping[str, Any], project_id: str) -> JobStatus:
    status = str(record.get("current_status") or "")
    return JobStatus(
        job_id=_id(record.get("id"), "job"),
        project_id=project_id,
        status=status,
        state=job_state(status),
        created_at=_time(record.get("created_at")),
        updated_at=_time(record.get("updated_at")),
    )


def _window_start(job: JobStatus, since: datetime | None, now: datetime) -> datetime:
    """The latest of the job's start, a minute before ``since`` and the oldest time Baseten
    keeps logs for, with a minute to spare."""
    starts = [now - LOG_RETENTION + CLOCK_SLACK]
    if job.created_at is not None:
        starts.append(job.created_at)
    if since is not None:
        starts.append(since - CLOCK_SLACK)
    return max(starts)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _stamp(line: Mapping[str, Any]) -> int | None:
    """A log line's time in nanoseconds (Baseten sends it as a string)."""
    try:
        return int(str(line.get("timestamp")))
    except ValueError:
        return None


def _line_key(line: Mapping[str, Any]) -> tuple[object, object, object]:
    return (line.get("timestamp"), line.get("message"), line.get("replica"))
