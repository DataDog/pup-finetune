# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune runs`: the training runs under ``./runs`` (or ``--dir``), newest first."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import run_command
from pup_finetune.cli._presentation import TRAIN, CommandDoc, default_note, local
from pup_finetune.display import display_path
from pup_finetune.next_steps import command, step, template
from pup_finetune.workspace import RUNS_DIR, list_runs

DOC = CommandDoc(
    path="runs",
    short="List the training runs under ./runs, newest first.",
    description=(
        "List the training runs under ./runs (or --dir), newest first: run id, phase, "
        "model, provider, best epoch, promoted model, deployment state and directory. "
        "Every command that takes a RUN_DIR also takes one of these run ids."
    ),
    examples=("pup finetune runs", "pup finetune runs --dir old-runs"),
    safety=local("lists the runs"),
    panel=TRAIN,
)


@run_command
def runs(
    directory: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            metavar="DIR",
            help=f"The directory holding the runs. {default_note('./runs')}",
            show_default=False,
        ),
    ] = None,
) -> dict[str, Any]:
    root = RUNS_DIR if directory is None else directory
    listing = list_runs(root)
    warnings = list(listing.warnings)
    if directory is not None and not directory.is_dir():
        warnings.insert(0, f"{display_path(directory)} is not a directory")
    entries = [run.to_json() for run in listing.runs]
    if entries:
        next_command: str | None = command("status", Path(str(entries[0]["run_dir"])))
        next_steps: list[dict[str, str]] = []
    else:
        next_command = None
        next_steps = [
            step(
                template(
                    "train", "--provider", "<PROVIDER>", "--dataset", "<DIR>", "--model", "<MODEL>"
                ),
                "preview a training run (free); it goes in runs/RUN_ID",
            )
        ]
    return {
        "dir": display_path(root),
        "count": len(entries),
        "runs": entries,
        "warnings": warnings,
        "next_command": next_command,
        "next_steps": next_steps,
    }


def register(app: typer.Typer) -> None:
    """Add `runs` to ``app``."""
    app.command(**DOC.typer())(runs)
