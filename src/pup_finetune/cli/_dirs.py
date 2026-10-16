# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The wrong kind of directory: a dataset passed to a run command, a run to a dataset one, or
a dataset that is not in Datadog to a command that trains or benchmarks on it.

Without these checks the mistake surfaces as a missing file (``ds/run.json does not exist``)
or, for ``dataset import`` and ``dataset pull``, as a dataset written into a run directory.
Each error names what the directory is and the command that takes it.

:func:`resolve_run_dir` is what every command that takes a run calls first: it reads a bare
run id as ``runs/<id>`` (see :mod:`pup_finetune.workspace`), says so on stderr, and then
checks the directory is not a dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

from pup_finetune.curation.pipeline import pipeline_state
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import PUSH_FILE
from pup_finetune.display import display_path, quoted
from pup_finetune.errors import (
    DirectoryUsageError,
    FinetuneError,
    NotADatasetDir,
    NotARunDir,
    RunDirNotEmpty,
)
from pup_finetune.output import info
from pup_finetune.runs import RUN_FILE, RunDir, is_run_id
from pup_finetune.workspace import RUNS_DIR, find_run_dir

__all__ = [
    "RUN_DIR_BY_ID",
    "RUN_DIR_GIVEN",
    "RUN_DIR_NEWEST",
    "require_dataset_dir",
    "require_in_datadog",
    "require_new_run_dir",
    "require_run_dir",
    "resolve_run_dir",
    "resolve_run_ref",
]


def _q(path: Path) -> str:
    return quoted(path)


def _train_command(dataset: Path) -> str:
    return f"`pup finetune train --dataset {_q(dataset)} --model <MODEL>`"


def _is_run(path: Path) -> bool:
    return (path / RUN_FILE).is_file()


RUN_DIR_GIVEN = "given"
RUN_DIR_BY_ID = "run id"
RUN_DIR_NEWEST = "newest under ./runs"
"""How a command found its run (``run_dir_source``): the path as given, a bare run id read
as ``runs/<id>``, or, for ``status`` without one, the newest run under ``./runs``."""


def resolve_run_ref(path: Path, command: str) -> tuple[Path, str]:
    """The run directory ``path`` names for ``command`` and how it was found
    (:data:`RUN_DIR_GIVEN` or :data:`RUN_DIR_BY_ID`): ``path`` when it exists, else
    ``runs/<id>`` for a bare run id (named on stderr), checked with :func:`require_run_dir`.

    A path that exists but is no run, while ``runs/<path>`` is one, is refused naming
    ``runs/<id>``: the path shadows the run id.
    """
    found, by_id = find_run_dir(path)
    if by_id:
        info(f"run id {path}: using {display_path(found)}")
    elif found.exists() and not _is_run(found) and not DatasetDir(found).is_dataset():
        shadowed = RUNS_DIR / os.fspath(path)
        if is_run_id(os.fspath(path)) and _is_run(shadowed):
            raise NotARunDir(
                f"{path} is not a training run (it has no {RUN_FILE})",
                hint=f"the path {path} here shadows the run id {path}, which names "
                f"{display_path(shadowed)}: pass {display_path(shadowed)}",
            )
    require_run_dir(found, command)
    return found, RUN_DIR_BY_ID if by_id else RUN_DIR_GIVEN


def resolve_run_dir(path: Path, command: str) -> Path:
    """The run directory of :func:`resolve_run_ref`, without how it was found."""
    return resolve_run_ref(path, command)[0]


def require_run_dir(path: Path, command: str) -> None:
    """Raise :class:`NotARunDir` when ``path`` is a dataset directory rather than the run
    directory ``command`` (e.g. ``evaluate``) takes."""
    if _is_run(path) or not DatasetDir(path).is_dataset():
        return
    if command == "evaluate":
        hint = (
            "evaluate takes the run directory or run id as its argument, `pup finetune "
            "evaluate <RUN_DIR>`, and reads the dataset from the run; to train on this "
            "dataset first: "
            f"{_train_command(path)}"
        )
    else:
        hint = (
            f"{command} takes a training run's directory; to train on this dataset: "
            f"{_train_command(path)}"
        )
    raise NotARunDir(f"{path} is a dataset directory, not a training run", hint=hint)


def require_dataset_dir(path: Path) -> None:
    """Raise :class:`NotADatasetDir` when ``path`` is a training run's directory rather than a
    dataset directory, naming the dataset the run trained on."""
    if not _is_run(path) or DatasetDir(path).is_dataset():
        return
    try:
        dataset: str | None = str(RunDir(path).manifest().get("dataset_dir") or "") or None
    except FinetuneError:
        dataset = None
    where = (
        f"the run trained on the dataset in {dataset}; pass that directory instead, or "
        if dataset
        else "pass a dataset directory instead, or "
    )
    raise NotADatasetDir(
        f"{path} is a training run directory, not a dataset",
        hint=f"{where}see the run with `pup finetune status {_q(path)}`",
    )


def require_new_run_dir(path: Path) -> None:
    """Raise :class:`RunDirNotEmpty` when ``path``, given as a new run directory, is a
    dataset."""
    if DatasetDir(path).is_dataset():
        raise RunDirNotEmpty(
            f"{path} is a dataset directory, not a new run directory",
            hint=f"train on it with --dataset {_q(path)}, and omit --run-dir or pass a new one",
        )


def require_in_datadog(path: Path, command: str) -> None:
    """Raise :class:`~pup_finetune.errors.DirectoryUsageError` unless every kept row of the
    dataset in ``path`` is in Datadog: its push receipt lists them all and no push is in
    flight. ``command`` (``train``, ``benchmark``) names who refuses it.

    What trains is then what the Agent Observability dataset holds, so a run and its
    evaluation trace back to records in Datadog. A directory ``dataset fetch`` made has the
    receipt of the dataset it read, so it passes."""
    dataset = DatasetDir(path)
    dataset.settings()  # NotADatasetDir, with how to make one
    state = pipeline_state(dataset)
    if state.push == "pushed":
        return
    where = display_path(path)
    fetch = (
        "or train on a dataset already in Datadog: `pup finetune dataset fetch <DIR> "
        "--project <PROJECT> --name <NAME>`"
    )
    if state.push == "not_pushed":
        raise DirectoryUsageError(
            f"{command} reads a dataset that is in Datadog, and {where} was never pushed",
            hint=f"push its {state.kept} kept rows first: `pup finetune dataset push "
            f"{_q(path)} --name <NAME>` (a remote write; it previews first), {fetch}",
        )
    if state.push == "pushed_with_new_rows":
        missing = (
            f"{state.pending} of its kept rows are not"
            if state.pending
            else "its last push did not finish, so its rows may not be"
        )
        raise DirectoryUsageError(
            f"{command} reads a dataset that is in Datadog, and {missing} in dataset "
            f"{state.dataset_name!r} yet",
            hint=f"push the rest: `pup finetune dataset push {_q(path)}` sends them to the "
            f"dataset {PUSH_FILE} names (a remote write; it previews first), {fetch}",
        )
    raise DirectoryUsageError(
        f"{command} reads a dataset that is in Datadog, and whether {where} is cannot be "
        f"read: {PUSH_FILE} is unreadable",
        hint=f"`pup finetune dataset show {_q(path)}` says why; fix or move aside {PUSH_FILE}, "
        f"then push again, {fetch}",
    )
