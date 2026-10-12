# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake ``baseten`` CLI for ``train --remote``: a ``CliRunner`` that never runs one.

``FakeBasetenCli`` records each call (``CliCall``: the argv, the environment, the timeout and
the ``--dir`` tree with each entry's mode, as it was during the call) and answers with its
``answers`` in order, the last one repeating::

    cli = FakeBasetenCli([error_answer(2, message="no such team"), exec_answer("job1")])

An answer that ``creates`` the job (a success, a lost answer, a timeout, a Ctrl-C) first
calls ``on_exec`` with the ``--dir`` directory, which tests use to copy the upload to the job
root and add the job to a :class:`~tests.fakes.baseten.FakeTrainingJobs`.
"""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pup_finetune.providers.baseten.jobs import CompletedCli


@dataclass(frozen=True, slots=True)
class CliCall:
    argv: list[str]
    env: dict[str, str]
    timeout: float
    tree: dict[str, int]
    """Each path under ``--dir``, relative, with its permission bits."""


@dataclass(frozen=True, slots=True)
class CliAnswer:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    raises: BaseException | None = None
    creates: bool = False


def exec_answer(
    job_id: str = "job1", project_id: str | None = "proj1", project: str = "pup-finetune-x"
) -> CliAnswer:
    """What the CLI prints, as JSON, for a job it created."""
    body = {
        "job_id": job_id,
        "project": {"id": project_id, "name": project},
        "ssh_hostname": None,
        "compute": {"cpu_count": 16, "memory": "64Gi", "accelerator": None, "gpu_count": None},
    }
    return CliAnswer(0, f"Uploading...\n{json.dumps(body, indent=2)}\n", creates=True)


def error_answer(
    returncode: int,
    *,
    status: int | None = None,
    message: str = "refused",
    truss: bool = False,
    creates: bool = False,
) -> CliAnswer:
    """A failed submit: the Go CLI's error envelope (``api_status_code``), or truss's own
    (``status_code``) with ``truss``."""
    key = "status_code" if truss else "api_status_code"
    error: dict[str, Any] = {"message": message, **({key: status} if status else {})}
    return CliAnswer(returncode, json.dumps({"error": error}), "Error: failed\n", creates=creates)


LOST = CliAnswer(1, "", "connection reset\n", creates=True)
"""A submit whose answer was lost after the job was created."""


def timeout_answer() -> CliAnswer:
    return CliAnswer(raises=subprocess.TimeoutExpired(["baseten"], 900.0), creates=True)


def interrupted_answer() -> CliAnswer:
    return CliAnswer(raises=KeyboardInterrupt(), creates=True)


def not_started_answer() -> CliAnswer:
    return CliAnswer(raises=FileNotFoundError(2, "No such file or directory"))


def tree(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): stat.S_IMODE(path.lstat().st_mode)
        for path in sorted(root.rglob("*"))
    }


@dataclass
class FakeBasetenCli:
    answers: Sequence[CliAnswer] = (exec_answer(),)
    on_exec: Callable[[Path], None] | None = None
    calls: list[CliCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue = list(self.answers)

    def __call__(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float
    ) -> CompletedCli:
        staged = Path(argv[list(argv).index("--dir") + 1])
        self.calls.append(CliCall(list(argv), dict(env), timeout, tree(staged)))
        answer = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if answer.creates and self.on_exec is not None:
            self.on_exec(staged)
        if answer.raises is not None:
            raise answer.raises
        return CompletedCli(answer.returncode, answer.stdout, answer.stderr)
