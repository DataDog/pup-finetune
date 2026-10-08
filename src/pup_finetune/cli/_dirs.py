# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The wrong kind of directory: a dataset passed to a run command, or a run to a dataset one.

Without these checks the mistake surfaces as a missing file (``ds/run.json does not exist``)
or, for ``dataset import`` and ``dataset pull``, as a dataset written into a run directory.
Each error names what the directory is and the command that takes it.
"""

from __future__ import annotations

from pathlib import Path

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import quoted
from pup_finetune.errors import FinetuneError, NotADatasetDir, NotARunDir, RunDirNotEmpty
from pup_finetune.runs import RUN_FILE, RunDir

__all__ = ["require_dataset_dir", "require_new_run_dir", "require_run_dir"]


def _q(path: Path) -> str:
    return quoted(path)


def _train_command(dataset: Path) -> str:
    return (
        "`pup finetune train --provider <PROVIDER> --model <MODEL> "
        f"--dataset {_q(dataset)} --run-dir <RUN_DIR>`"
    )


def _is_run(path: Path) -> bool:
    return (path / RUN_FILE).is_file()


def require_run_dir(path: Path, command: str) -> None:
    """Raise :class:`NotARunDir` when ``path`` is a dataset directory rather than the run
    directory ``command`` (e.g. ``evaluate``) takes."""
    if _is_run(path) or not DatasetDir(path).is_dataset():
        return
    if command == "evaluate":
        hint = (
            "evaluate takes the run directory as its argument, `pup finetune evaluate "
            "<RUN_DIR>`, and reads the dataset from the run; to train on this dataset first: "
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
            hint=f"train on it with --dataset {_q(path)}, and pass a new or empty --run-dir",
        )
