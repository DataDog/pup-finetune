# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset`: build and inspect local dataset directories."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import run_command
from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings, read_import_file
from pup_finetune.data.split import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    SPLITS,
    SplitParams,
)

SAMPLE_IDS_PER_SPLIT = 5

_LOCAL_ONLY = "Reads and writes local files only, so it is allowed with PUP_READ_ONLY set."

dataset_app = typer.Typer(
    name="dataset",
    short_help="Build and inspect local training datasets.",
    help=(
        "Build and inspect dataset directories: accepted rows, excluded rows with the reason, "
        "and stable train / validation / test splits. Dataset files hold conversation "
        "content, so directories are created 0700 and files 0600."
    ),
    no_args_is_help=True,
)

DirArgument = Annotated[
    Path, typer.Argument(metavar="DIR", help="The dataset directory.", show_default=False)
]


def _quoted(path: Path) -> str:
    return shlex.quote(str(path))


@dataset_app.command(
    "import",
    short_help="Import canonical rows from a local JSONL file.",
    help=(
        "Validate canonical training rows from a local JSONL file and add them to DIR, "
        "creating it if needed. Invalid rows are not fatal: each is recorded in "
        "excluded.jsonl with its reason code. A row repeating a conversation already in the "
        "dataset is excluded as duplicate_conversation. Importing again appends rows with new "
        "ids; a known id with different content is an error. The split settings are frozen "
        f"when DIR is created, and later imports must pass the same ones. {_LOCAL_ONLY}"
    ),
)
@run_command
def import_rows(
    directory: DirArgument,
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            help="Canonical rows, one JSON object per line.",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
        ),
    ],
    seed: Annotated[int, typer.Option(help="Seed for the split assignment.")] = DEFAULT_SEED,
    validation_fraction: Annotated[
        float, typer.Option(help="Share of traces assigned to validation.")
    ] = DEFAULT_VALIDATION_FRACTION,
    test_fraction: Annotated[
        float, typer.Option(help="Share of traces assigned to test.")
    ] = DEFAULT_TEST_FRACTION,
) -> dict[str, Any]:
    params = SplitParams(seed, validation_fraction, test_fraction)
    rows, exclusions = read_import_file(file)
    dataset = DatasetDir(directory)
    result = dataset.add(rows, exclusions, settings=DatasetSettings("import", params))
    summary = result.summary
    return {
        "dataset_dir": str(dataset.path.resolve()),
        "accepted": summary["accepted"],
        "excluded": summary["excluded"],
        "excluded_by_code": summary["excluded_by_code"],
        "splits": summary["splits"],
        "warnings": summary["warnings"],
        "import": {
            "file": str(file),
            "added": result.added,
            "unchanged": result.unchanged,
            "excluded": result.excluded,
        },
        "next_command": f"pup finetune dataset show {_quoted(directory)}",
    }


@dataset_app.command(
    "show",
    short_help="Show a dataset's counts, exclusions, splits and settings.",
    help=(
        "Show a dataset directory: row counts, exclusion reasons, split sizes and warnings, "
        f"the frozen settings, and a few row ids per split. {_LOCAL_ONLY}"
    ),
)
@run_command
def show(directory: DirArgument) -> dict[str, Any]:
    dataset = DatasetDir(directory)
    settings = dataset.settings()
    by_split = dataset.rows_by_split()
    return {
        "dataset_dir": str(dataset.path.resolve()),
        "settings": settings,
        "summary": dataset.summary(),
        "sample_row_ids": {
            split: [row.id for row in by_split[split][:SAMPLE_IDS_PER_SPLIT]] for split in SPLITS
        },
        "next_command": f"pup finetune train --dataset {_quoted(directory)} ...",
    }


def register(app: typer.Typer) -> None:
    """Add the `dataset` command group to ``app``."""
    app.add_typer(dataset_app, name="dataset")
