# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset`: build and inspect local dataset directories."""

from __future__ import annotations

import shlex
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import current_context, get_runtime, run_command
from pup_finetune.cli._dirs import require_dataset_dir
from pup_finetune.cli._flags import flag_name, given, renamed
from pup_finetune.cli._presentation import (
    BUILD_A_DATASET,
    CommandDoc,
    Safety,
    describe,
    group,
    local,
)
from pup_finetune.curation.labels import Label, apply_labels, parse_labels
from pup_finetune.curation.pipeline import Pipeline, pipeline_state
from pup_finetune.curation.samples import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    SampleSplit,
    sample_rows,
)
from pup_finetune.data.convert import ReasoningMode
from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings, read_import_file
from pup_finetune.data.exclusions import explain
from pup_finetune.data.pull import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_TARGET_COUNT,
    Collection,
    PullOptions,
    frozen_window,
    pull,
    read_limits,
    rerun_command,
    resolve_window,
)
from pup_finetune.data.push import (
    IN_FLIGHT_SETTLE,
    PUSH_FILE,
    PushPlan,
    plan_push,
    push_dataset,
    push_path,
    read_push_state,
)
from pup_finetune.data.split import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    SPLITS,
    SplitParams,
)
from pup_finetune.datadog.datasets import DEFAULT_PROJECT, RECORDS_BATCH_SIZE
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import (
    Clock,
    SpanQuery,
    TimeBound,
    parse_time_bound,
)
from pup_finetune.display import display_path
from pup_finetune.errors import SettingsConflict, UsageError, require_confirm, require_writable
from pup_finetune.next_steps import command, step, template
from pup_finetune.output import info
from pup_finetune.workspace import ParentSetup, ignore_new_parent

SAMPLE_IDS_PER_SPLIT = 5
EXPLAINED_CODES = 5
"""How many of the most frequent exclusion codes get an explanation in the output."""

DEFAULT_FROM = "7d"
DEFAULT_TO = "now"

REVIEW_SAMPLE_SIZE = 20
"""How many rows the ``dataset sample`` a dataset's ``next_command`` names shows."""

clock: Clock = time.time
"""What ``now`` in ``--from`` and ``--to`` is resolved against; tests replace it."""

dataset_app = group(
    "dataset",
    short="Build, review and push a training dataset.",
    description=describe(
        "Build a dataset directory from your traces or a JSONL file, review and label its "
        "rows, and push the kept rows to Datadog. Rows are split into train, validation and "
        "test sets by trace, the same way every time.",
        "Dataset files hold conversation content, so directories are created 0700 and files 0600.",
    ),
)

FILTER_PANEL = "Filter"
LIMITS_PANEL = "Limits"
ROWS_PANEL = "Rows"
SPLITS_PANEL = "Splits"
"""The option panels of `dataset pull` (and `import`'s splits)."""

DirArgument = Annotated[
    Path, typer.Argument(metavar="DIR", help="The dataset directory.", show_default=False)
]
SeedOption = Annotated[
    int,
    typer.Option(metavar="N", help="Seed for the split assignment.", rich_help_panel=SPLITS_PANEL),
]
ValidationFractionOption = Annotated[
    float,
    typer.Option(metavar="X", help="Share of traces for validation.", rich_help_panel=SPLITS_PANEL),
]
TestFractionOption = Annotated[
    float, typer.Option(metavar="X", help="Share of traces for test.", rich_help_panel=SPLITS_PANEL)
]

PullMlAppOption = Annotated[
    str | None,
    typer.Option(
        "--ml-app",
        metavar="NAME",
        help="The ml_app whose LLM spans to read (a rerun takes the frozen one).",
        show_default=False,
        rich_help_panel=FILTER_PANEL,
    ),
]
SpanNameOption = Annotated[
    str | None,
    typer.Option(
        "--span-name",
        metavar="NAME",
        help="Only spans with this name.",
        show_default=False,
        rich_help_panel=FILTER_PANEL,
    ),
]
QueryOption = Annotated[
    str | None,
    typer.Option(
        "--query",
        metavar="TEXT",
        help="Only spans that match this search query.",
        show_default=False,
        rich_help_panel=FILTER_PANEL,
    ),
]
TagOption = Annotated[
    list[str] | None,
    typer.Option(
        "--tag",
        metavar="KEY:VALUE",
        help="Only spans with this tag; repeatable.",
        show_default=False,
        rich_help_panel=FILTER_PANEL,
    ),
]
FromOption = Annotated[
    str,
    typer.Option(
        "--from",
        metavar="TIME",
        help="Start: ISO 8601, epoch ms, or relative: 7d, 24h, now-7d, now.",
        rich_help_panel=FILTER_PANEL,
    ),
]
ToOption = Annotated[
    str,
    typer.Option(
        "--to",
        metavar="TIME",
        help="End, in the same forms as --from.",
        rich_help_panel=FILTER_PANEL,
    ),
]
TargetCountOption = Annotated[
    int,
    typer.Option(
        min=1,
        metavar="N",
        help="Stop once this many rows are accepted.",
        rich_help_panel=LIMITS_PANEL,
    ),
]
MaxCandidatesOption = Annotated[
    int,
    typer.Option(
        min=1,
        metavar="N",
        help="Read at most this many spans.",
        rich_help_panel=LIMITS_PANEL,
    ),
]
ReasoningOption = Annotated[
    ReasoningMode,
    typer.Option(
        metavar="MODE",
        help="omit or preserve reasoning messages.",
        rich_help_panel=ROWS_PANEL,
    ),
]
IncludeErrorsOption = Annotated[
    bool,
    typer.Option(
        "--include-errors",
        help="Keep errored LLM calls.",
        rich_help_panel=ROWS_PANEL,
    ),
]

_PULL_FLAGS = (
    "Spans are LLM spans of --ml-app in the time range, narrowed by --span-name, --query and "
    "--tag. Each span becomes a conversation (input then output messages; a plain input or "
    "output value is the user or assistant message); per trace, a conversation another one "
    "contains is dropped, so sub-agents stay their own rows. "
    "Errored LLM calls are excluded as span_error unless --include-errors is passed. Reading "
    "stops once --target-count rows are accepted or --max-candidates spans are read, "
    "unreadable ones included. A relative time is a number and a unit (s, m, h, d or w, or "
    "spelled out, such as 5min or 2hours), with or without now- in front (7d is now-7d), or "
    "now; it is resolved once, when the command starts. A bare number is epoch "
    "milliseconds, not seconds: one below 10^11 (before March 1973) is refused."
)


IMPORT_DOC = CommandDoc(
    path="dataset import",
    short="Import canonical rows from a local JSONL file.",
    description=describe(
        "Validate canonical training rows from --file and add them to DIR, creating it if "
        "needed. An invalid row is not fatal: it is recorded in excluded.jsonl with its "
        "reason code. The split settings are frozen when DIR is created.",
        "A row repeating a conversation already in the dataset is excluded as "
        "duplicate_conversation. Importing again appends rows with new ids; a known id with "
        "different content is an error. Later imports must pass the same split settings.",
    ),
    examples=(
        "pup finetune dataset import datasets/my-agent --file rows.jsonl",
        "pup finetune dataset import datasets/my-agent --file rows.jsonl --seed 7 --test-fraction 0.2",
    ),
    safety=local("imports rows into DIR"),
)

FETCH_DOC = CommandDoc(
    path="dataset fetch",
    short="Fetch an Agent Observability dataset to train on.",
    description=describe(
        "Read the current records of the dataset --name in --project into DIR, so what was "
        "curated in Datadog (records edited, added or removed in the product or through "
        "pup) is what trains. Each record becomes a canonical row: its input messages and "
        "expected output, its tools, and the split it names (records without one are "
        "assigned by trace). A record that is not a conversation is excluded.",
        "DIR freezes the dataset and its version: fetch a later version into a new DIR. "
        "The fetch writes the dataset's push receipt, so `evaluate` publishes its "
        "experiments onto the same dataset. Nothing is written to Datadog.",
    ),
    examples=(
        "pup finetune dataset fetch datasets/my-agent-v2 --project pup-finetune "
        "--name my-agent-sft",
        "pup finetune dataset fetch datasets/curated --project my-team --name curated-sft --seed 7",
    ),
    safety=Safety(
        data="reads Datadog",
        recipients="nowhere: reads the dataset through pup into local files",
        without="fetches the dataset into DIR",
    ),
)

SHOW_DOC = CommandDoc(
    path="dataset show",
    short="Show a dataset's counts, exclusions, splits and settings.",
    description=(
        "Show a dataset directory: row counts, exclusion reasons, split sizes and warnings, "
        "the frozen settings, where it was pushed, a few row ids per split, and its pipeline: "
        "how many rows are kept, dropped and unlabeled, whether a triage is complete, and "
        "how many kept rows are not pushed yet, which choose the next steps."
    ),
    examples=(
        "pup finetune dataset show datasets/my-agent",
        "pup finetune dataset show datasets/my-agent",
    ),
    safety=local("shows the dataset"),
)

PUSH_DOC = CommandDoc(
    path="dataset push",
    short="Push kept rows to Datadog (previews without --confirm).",
    description=describe(
        "Add one record per kept row of DIR to the Agent Observability dataset --name in "
        "--project, creating either when missing. Without --confirm, previews what would be "
        "sent. The receipt, push.json in DIR, makes a rerun send only rows not yet pushed, "
        "and names the dataset, so a rerun needs no --name or --project.",
        "Rows labeled drop are never sent. Each record's input is the conversation before "
        "the final assistant message, with its tools; its expected output is that message; "
        "its metadata holds the row id, split and source; it is tagged split:<split> and "
        "source:pup-finetune. When every kept row was pushed already, it reports the "
        "receipt and sends nothing. The receipt is saved after every batch, so an "
        "interrupted push resumes without adding a row twice. A directory pushes to one "
        "dataset: another --name or --project than the receipt's is an error. Rows labeled "
        "drop after they "
        "were pushed are reported as stale_records and are not deleted from the dataset; "
        "kept rows that changed after they were pushed are reported as changed_records and "
        "their records are not updated.",
    ),
    examples=(
        "pup finetune dataset push datasets/my-agent --name my-agent-sft",
        "pup finetune dataset push datasets/my-agent --name my-agent-sft --project pup-finetune-dev",
        "pup finetune dataset push datasets/my-agent",
    ),
    safety=Safety(
        data="--confirm writes the kept rows to Datadog",
        recipients="Datadog: an Agent Observability dataset",
        without="previews the records",
        confirm="pushes the kept rows (a remote write)",
        read_only="confirm_refused",
    ),
)

PULL_DOC = CommandDoc(
    path="dataset pull",
    short="Pull training rows from Datadog into a dataset directory.",
    description=describe(
        "Pull --ml-app's LLM spans from Datadog into DIR as training rows, with splits. "
        "The filter, the time range, --reasoning, --include-errors and the splits are frozen "
        "on the first pull, so a rerun needs no flag: `dataset pull DIR` resumes, and a "
        "higher --target-count reads further.",
        "Excluded rows are recorded in excluded.jsonl with their reason. On a rerun, a flag "
        "left out takes the frozen value and a flag with another value fails, so use a new "
        "directory for other settings. Raw pages are cached in DIR/raw/, so a rerun resumes "
        "without fetching them again, and a higher --target-count or --max-candidates (which "
        "are not frozen) reads further (a stored row that turns out to be a prefix of a "
        "longer conversation of its trace is then excluded as contained_conversation). A "
        "relative --from or --to, such as the default 7d, is resolved when the first "
        "pull starts and that window is frozen with the expressions: a rerun reads the same "
        f"window, however much later it runs, and its result's window says so. {_PULL_FLAGS}",
    ),
    examples=(
        "pup finetune dataset pull datasets/my-agent --ml-app my-agent --from 7d --span-name chat",
        "pup finetune dataset pull datasets/my-agent --target-count 500",
    ),
    safety=Safety(
        data="reads Datadog, writes DIR",
        recipients="DIR: reads Datadog through pup",
        without="pulls rows into DIR",
    ),
)

LABEL_DOC = CommandDoc(
    path="dataset label",
    short="Record keep/drop labels for a dataset's rows.",
    description=describe(
        'Record keep/drop decisions on accepted rows of DIR from --file, one {"id", "keep", '
        '"reason"} object per line; a reason is required to drop. Every id must be an '
        "accepted row, or nothing is written. A dropped row is left out of training and of "
        "the split files.",
        '"at" may be given per line and defaults to the current time. Rows labeled here are '
        "never relabeled by `dataset triage`. Labels are appended to labels.jsonl and the "
        "last label of an id wins; rows.jsonl is never changed.",
    ),
    examples=(
        "pup finetune dataset label datasets/my-agent --file labels.jsonl",
        "pup finetune dataset label datasets/my-agent --file review.jsonl",
    ),
    safety=local("records labels in DIR"),
)

SAMPLE_DOC = CommandDoc(
    path="dataset sample",
    short="Show a small, readable sample of a dataset's rows.",
    description=describe(
        "Show up to --count rows of DIR in brief, to review them before labeling: id, split, "
        "label, turns, tools, and the first user and final assistant messages cut to 300 "
        "characters. The same --seed picks the same rows. Rows labeled drop are left out "
        "unless --include-dropped is passed.",
        "A final message without text shows its tool calls as name(args). The full rows stay "
        "in rows.jsonl.",
    ),
    examples=(
        "pup finetune dataset sample datasets/my-agent -n 20",
        "pup finetune dataset sample datasets/my-agent --split test --include-dropped",
    ),
    safety=local("shows rows"),
)


def _quoted(path: Path) -> str:
    return shlex.quote(str(path))


def dataset_next(
    dataset: DatasetDir,
    directory: Path,
    *,
    summary: Mapping[str, Any] | None = None,
    sampled: bool = False,
) -> dict[str, Any]:
    """The ``pipeline`` block of a dataset directory, and the ``next_command`` and
    ``next_steps`` it leads to.

    With accepted rows and none labeled, the next command is a ``dataset sample`` to
    review (``None`` right after one, ``sampled``), and labeling them, by hand or with a
    judge council, are the steps; a fetched dataset (``dataset fetch``) was curated in
    Datadog, so its rows go straight to training. A triage that is not complete gets a step to finish it.
    Once rows are labeled, training is: its preview needs a model that cannot be chosen
    without guessing, so it is a step. So is the push: with ``--name`` while the dataset
    was never pushed, and without any flag (the receipt names the dataset) while kept rows
    are not pushed yet. ``directory`` is the directory as the user typed it; ``summary`` is
    the dataset's summary when the caller has it already. A pipeline file that cannot be
    read adds ``warnings`` (see :func:`merge_next`).
    """
    state = pipeline_state(dataset, summary)
    shown: dict[str, Any] = {"pipeline": state.to_json()}
    if state.warnings:
        shown["warnings"] = list(state.warnings)
    if not state.accepted:
        return {**shown, "next_command": None, "next_steps": []}
    label_step = step(
        template("dataset", "label", directory, "--file", "<LABELS_FILE>"),
        'record keep/drop decisions: write <LABELS_FILE> first, one {"id", "keep", '
        '"reason"} object per line (a reason is required to drop)',
    )
    triage_steps = (
        [
            step(
                template("dataset", "triage", directory, "--rubric", "<RUBRIC_FILE>"),
                "finish the triage: the same --rubric and --judges resume it, reusing the "
                "votes cast (paid; it previews first)",
            )
        ]
        if state.triage == "incomplete"
        else [
            step(
                template(
                    "dataset", "triage", directory, "--rubric", "<RUBRIC_FILE>", "--reset-triage"
                ),
                "the triage files cannot be read (see the warning): fix or remove them, or "
                "discard every vote and triage again (paid; it previews first)",
            )
        ]
        if state.triage == "unreadable"
        else []
    )
    source = dataset.settings().get("source")
    # A fetched dataset was curated in Datadog: its rows are the ones to train on.
    fetched = isinstance(source, Mapping) and source.get("kind") == "datadog_dataset"
    if not state.labeled and not fetched:
        triage_step = step(
            template(
                "dataset",
                "triage",
                directory,
                "--rubric",
                "<RUBRIC_FILE>",
                "--max-rows",
                REVIEW_SAMPLE_SIZE,
            ),
            "or have a council of judge models label the rows against a rubric (paid; it "
            "previews first; `pup finetune dataset triage --print-template` prints a rubric "
            "to start from)",
        )
        sample = command("dataset", "sample", directory, "--count", REVIEW_SAMPLE_SIZE)
        return {
            **shown,
            "next_command": None if sampled else sample,
            "next_steps": [
                label_step,
                *(triage_steps or [triage_step]),
                *_push_steps(state, directory),
            ],
        }
    steps = [label_step] if sampled else []
    steps += _pull_more_steps(dataset, directory, state)
    steps += triage_steps
    steps += [
        step(
            command("models", "list"),
            "see the models a provider can fine-tune",
        ),
        step(
            template("benchmark", directory, "--models", "<MODELS>"),
            "compare base models on the test split to choose the one to fine-tune (paid; it "
            "previews first)",
        ),
        step(
            template("train", "--dataset", directory, "--model", "<MODEL>"),
            "preview a training run on the kept rows (free); choose the model with the user "
            "(its provider is inferred, and the run goes in runs/RUN_ID)",
        ),
    ]
    if state.push == "not_pushed":
        steps.append(
            step(
                template("dataset", "push", directory, "--name", "<NAME>"),
                "push the kept rows to Datadog so an evaluation can be published as "
                "experiments (a remote write; it previews first)",
            )
        )
    steps += _push_steps(state, directory)
    return {**shown, "next_command": None, "next_steps": steps}


def _pull_more_steps(dataset: DatasetDir, directory: Path, state: Pipeline) -> list[dict[str, str]]:
    """A step to pull more rows when a review dropped some and fewer than the last pull's
    ``--target-count`` are kept: a rerun reads on to replace them (dropped rows do not count
    toward the target)."""
    limits = read_limits(dataset)
    if limits is None or limits[2] or not state.dropped or state.kept >= limits[0]:
        return []  # no pull, or the last one read every matching span
    return [
        step(
            command("dataset", "pull", directory),
            f"{state.kept} of --target-count {limits[0]} rows are kept (or not reviewed yet) "
            f"after the review dropped {state.dropped}: pull again to read on and "
            "replace them (it continues from the last pull), then review the new rows (a "
            "triage judges only rows without a label)",
        )
    ]


def merge_next(result: dict[str, Any], following: Mapping[str, Any]) -> dict[str, Any]:
    """``result`` with ``following`` (a :func:`dataset_next` block) added, its warnings
    after the result's own."""
    warnings = following.get("warnings") or []
    result.update({key: value for key, value in following.items() if key != "warnings"})
    if warnings:
        result["warnings"] = [*(result.get("warnings") or []), *warnings]
    return result


def _push_steps(state: Pipeline, directory: Path) -> list[dict[str, str]]:
    """The step for a dataset pushed before: the rows not pushed yet, or why it is unknown."""
    if state.push == "pushed_with_new_rows":
        return [
            step(
                command("dataset", "push", directory),
                f"push the {state.pending} kept rows not pushed yet to dataset "
                f"{state.dataset_name!r}, which {PUSH_FILE} names (a remote write; it "
                "previews first)",
            )
        ]
    if state.push == "unreadable":
        return [
            step(
                command("dataset", "show", directory),
                f"{PUSH_FILE} cannot be read, so whether the dataset was pushed is unknown; "
                "show reports why",
            )
        ]
    return []


def _ignore_note(directory: Path) -> ParentSetup:
    """Create ``./datasets`` with a ``.gitignore`` when ``directory`` is the first dataset
    in it (see :func:`~pup_finetune.workspace.ignore_new_parent`); the note saying so, or
    the warning that an existing ``./datasets`` is not ignored by git."""
    return ignore_new_parent(directory)


@dataset_app.command(**IMPORT_DOC.typer())
@run_command
def import_rows(
    directory: DirArgument,
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            metavar="FILE",
            help="Canonical rows, one JSON object per line.",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
        ),
    ],
    seed: SeedOption = DEFAULT_SEED,
    validation_fraction: ValidationFractionOption = DEFAULT_VALIDATION_FRACTION,
    test_fraction: TestFractionOption = DEFAULT_TEST_FRACTION,
) -> dict[str, Any]:
    params = SplitParams(seed, validation_fraction, test_fraction)
    require_dataset_dir(directory)
    rows, exclusions = read_import_file(file)
    dataset = DatasetDir(directory)
    notes = _ignore_note(directory)
    result = dataset.add(rows, exclusions, settings=DatasetSettings("import", params))
    summary = result.summary
    return merge_next(
        {
            "dataset_dir": display_path(dataset.path),
            "accepted": summary["accepted"],
            "excluded": summary["excluded"],
            "excluded_by_code": summary["excluded_by_code"],
            "splits": summary["splits"],
            "notes": notes.notes,
            "warnings": [*notes.warnings, *summary["warnings"]],
            "import": {
                "file": str(file),
                "added": result.added,
                "unchanged": result.unchanged,
                "excluded": result.excluded,
            },
        },
        dataset_next(dataset, directory, summary=summary),
    )


@dataset_app.command(**FETCH_DOC.typer())
@run_command
def fetch(
    directory: DirArgument,
    project: Annotated[
        str,
        typer.Option(
            "--project",
            metavar="NAME",
            help="The project the dataset belongs to.",
            show_default=False,
        ),
    ],
    name: Annotated[
        str,
        typer.Option("--name", metavar="NAME", help="The dataset to fetch.", show_default=False),
    ],
    seed: SeedOption = DEFAULT_SEED,
    validation_fraction: ValidationFractionOption = DEFAULT_VALIDATION_FRACTION,
    test_fraction: TestFractionOption = DEFAULT_TEST_FRACTION,
) -> dict[str, Any]:
    from pup_finetune.data.fetch import fetch_dataset
    from pup_finetune.datadog.datasets import LLMObsDatasets

    params = SplitParams(seed, validation_fraction, test_fraction)
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    notes = _ignore_note(directory)
    fetched = fetch_dataset(
        dataset, LLMObsDatasets(Pup()), project=project, name=name, split=params
    )
    summary = fetched.added.summary
    info(
        f"fetched {fetched.records} records of dataset {name!r} (version "
        f"{fetched.dataset_version}): {summary['accepted']} rows, {summary['excluded']} excluded"
    )
    return merge_next(
        {
            "dataset_dir": display_path(dataset.path),
            "accepted": summary["accepted"],
            "excluded": summary["excluded"],
            "excluded_by_code": summary["excluded_by_code"],
            "splits": summary["splits"],
            "notes": notes.notes,
            "warnings": [*notes.warnings, *summary["warnings"]],
            "fetch": {
                "project": project,
                "dataset_name": name,
                "dataset_id": fetched.dataset_id,
                "dataset_version": fetched.dataset_version,
                "records": fetched.records,
                "added": fetched.added.added,
                "unchanged": fetched.added.unchanged,
                "excluded": fetched.added.excluded,
            },
        },
        dataset_next(dataset, directory, summary=summary),
    )


@dataset_app.command(**SHOW_DOC.typer())
@run_command
def show(directory: DirArgument) -> dict[str, Any]:
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    settings = dataset.settings()
    by_split = dataset.rows_by_split()
    summary = dataset.summary()
    result = {
        "dataset_dir": display_path(dataset.path),
        "settings": settings,
        "summary": summary,
        "sample_row_ids": {
            split: [row.id for row in by_split[split][:SAMPLE_IDS_PER_SPLIT]] for split in SPLITS
        },
        "push": _push_brief(dataset),
    }
    return merge_next(result, dataset_next(dataset, directory, summary=summary))


@dataset_app.command(**PUSH_DOC.typer())
@run_command
def push(
    directory: DirArgument,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            metavar="NAME",
            help=f"The dataset to push to (a rerun takes the one {PUSH_FILE} names).",
            show_default=False,
        ),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            metavar="NAME",
            help=f"The project the dataset belongs to: {DEFAULT_PROJECT}, or the one "
            f"{PUSH_FILE} names.",
            show_default=False,
        ),
    ] = None,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Push (a remote write); without it, only preview."),
    ] = False,
) -> dict[str, Any]:
    runtime = get_runtime()
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    dataset.settings()  # A UsageError when this is not a dataset directory.
    state = read_push_state(dataset)
    defaults_used: list[str] = []
    if state is not None:
        # The receipt pins the dataset: an omitted flag is its value, another one an error.
        name = state.dataset_name if name is None else name
        project = state.project if project is None else project
        rerun: list[object] = ["dataset", "push", directory]
    elif name is None:
        raise UsageError(
            f"dataset push needs --name ({display_path(directory)} was never pushed)",
            hint="name the Agent Observability dataset to create or add to; later pushes "
            f"take it from {PUSH_FILE}",
        )
    else:
        if project is None:
            project, defaults_used = DEFAULT_PROJECT, ["project"]
        rerun = ["dataset", "push", directory, "--name", name]
        if project != DEFAULT_PROJECT:
            rerun += ["--project", project]
    plan = plan_push(dataset, project=project, name=name)
    plan.require_rows()
    if plan.needs_remote:
        preview = _push_preview(plan, defaults_used)
        if confirm:
            require_writable(runtime, "dataset push")
        # The preview reads local files only, so it is allowed under PUP_READ_ONLY.
        require_confirm(
            confirm,
            preview,
            command(*rerun, "--confirm"),
            read_only=runtime.read_only,
        )
        receipt = push_dataset(
            dataset, project=plan.project, name=plan.dataset_name, pup=Pup(), progress=info
        )
    else:
        info("nothing to push: every kept row was pushed already")
        receipt = plan.receipt()
    result: dict[str, Any] = {"dataset_dir": display_path(dataset.path), **receipt.to_json()}
    result["receipt_file"] = display_path(push_path(dataset))
    if warnings := _receipt_warnings(receipt.stale_records, receipt.changed_records):
        result["warnings"] = warnings
    return merge_next(result, dataset_next(dataset, directory))


def _push_preview(plan: PushPlan, defaults_used: Sequence[str]) -> dict[str, Any]:
    to_push = len(plan.pending)
    if plan.dataset_id is None:
        action = (
            f"create dataset {plan.dataset_name!r} in project {plan.project!r} (and the "
            f"project if missing), then add {to_push} records"
        )
    elif plan.in_flight:
        action = (
            f"check dataset {plan.dataset_id} for the records of an interrupted batch, then "
            f"add those of the {to_push} records it does not hold"
        )
    else:
        action = f"add {to_push} records to dataset {plan.dataset_id}"
    preview: dict[str, Any] = {
        "action": action,
        "billing": f"no model provider is called: the push writes {to_push} records to your "
        "Datadog organization, in batches of at most "
        f"{RECORDS_BATCH_SIZE} through pup",
        "dataset_dir": display_path(plan.directory),
        "project": plan.project,
        "dataset_name": plan.dataset_name,
        "dataset_id": plan.dataset_id,
        "rows": plan.pending_by_split(),
        "records_to_push": to_push,
        "already_pushed": plan.already_pushed,
        "stale_records": plan.stale_records,
        "changed_records": plan.changed_records,
        "sample_record": plan.pending[0].record if plan.pending else None,
        "settings": {
            "project": plan.project,
            "dataset_name": plan.dataset_name,
            "batch_size": RECORDS_BATCH_SIZE,
        },
        "defaults_used": list(defaults_used),
        "data_recipients": [
            f"kept rows -> Datadog (dataset {plan.dataset_name!r} in project {plan.project!r})"
        ],
    }
    warnings: list[str] = []
    if plan.in_flight:
        settle = int(IN_FLIGHT_SETTLE.total_seconds() // 60)
        warnings.append(
            f"an interrupted push sent {len(plan.in_flight)} records without learning whether "
            "they were added; the push lists the dataset's records first and sends only those "
            f"it does not find, and none until {settle} minutes after the batch was sent"
        )
    warnings += _receipt_warnings(plan.stale_records, plan.changed_records)
    if warnings:
        preview["warnings"] = warnings
    return preview


def _receipt_warnings(stale: Mapping[str, str], changed: Mapping[str, str]) -> list[str]:
    warnings: list[str] = []
    if stale:
        warnings.append(
            f"{len(stale)} records pushed earlier are of rows no longer kept (see "
            "stale_records); they are not deleted from the remote dataset, so remove them "
            "there if needed"
        )
    if changed:
        warnings.append(
            f"{len(changed)} kept rows changed after they were pushed (see changed_records); "
            "a push only adds records, so their remote records were not updated. To push "
            f"the rows as they are now, move {PUSH_FILE} aside and push to a new --name"
        )
    return warnings


@dataset_app.command(**PULL_DOC.typer())
@run_command
def pull_rows(
    directory: DirArgument,
    ml_app: PullMlAppOption = None,
    span_name: SpanNameOption = None,
    query: QueryOption = None,
    tag: TagOption = None,
    from_: FromOption = DEFAULT_FROM,
    to: ToOption = DEFAULT_TO,
    target_count: TargetCountOption = DEFAULT_TARGET_COUNT,
    max_candidates: MaxCandidatesOption = DEFAULT_MAX_CANDIDATES,
    reasoning: ReasoningOption = "omit",
    include_errors: IncludeErrorsOption = False,
    seed: SeedOption = DEFAULT_SEED,
    validation_fraction: ValidationFractionOption = DEFAULT_VALIDATION_FRACTION,
    test_fraction: TestFractionOption = DEFAULT_TEST_FRACTION,
) -> dict[str, Any]:
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    values = _with_frozen(
        dataset,
        directory,
        {
            "ml_app": ml_app,
            "span_name": span_name,
            "query": query,
            "tag": tag,
            "from_": from_,
            "to": to,
            "reasoning": reasoning,
            "include_errors": include_errors,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
    )
    params = SplitParams(values["seed"], values["validation_fraction"], values["test_fraction"])
    # The last pull's limits, for a rerun that leaves them out: it continues toward them.
    remembered = read_limits(dataset) if dataset.is_dataset() else None
    ctx = current_context()
    if remembered is not None and (ctx is None or not given(ctx, "target_count")):
        target_count = remembered[0]
    if remembered is not None and (ctx is None or not given(ctx, "max_candidates")):
        max_candidates = remembered[1]
    options = _pull_options(
        ml_app=values["ml_app"],
        span_name=values["span_name"],
        query=values["query"],
        tags=values["tag"],
        from_=values["from_"],
        to=values["to"],
        target_count=target_count,
        max_candidates=max_candidates,
        reasoning=values["reasoning"],
        include_errors=values["include_errors"],
    )
    frozen = frozen_window(options, dataset) is not None
    options = resolve_window(options, dataset, clock)
    window = _window(options, frozen=frozen)
    if frozen:
        info(
            f"reading the window the first pull froze: {window['from']} to {window['to']} "
            f"(--from {options.query.from_} --to {options.query.to} as of then)"
        )
    notes = _ignore_note(directory)
    collection, added = pull(Pup(), options, dataset, params, progress=info)
    summary = added.summary
    following = dataset_next(dataset, directory, summary=summary)
    pipeline_warnings: list[str] = following.pop("warnings", [])
    if collection.stopped == "max_candidates" and collection.counted < options.target_count:
        further = replace(options, max_candidates=options.max_candidates * 2)
        following["next_steps"].insert(
            0,
            step(
                rerun_command(directory, further),
                "read further: --max-candidates stopped this pull before --target-count rows "
                "were accepted (the frozen settings fill in the filter)",
            ),
        )
    return {
        "dataset_dir": display_path(dataset.path),
        "accepted": summary["accepted"],
        "excluded": summary["excluded"],
        "excluded_by_code": summary["excluded_by_code"],
        "exclusion_explanations": _explanations(summary["excluded_by_code"]),
        "splits": summary["splits"],
        "notes": notes.notes,
        "warnings": [
            *notes.warnings,
            *_collection_warnings(collection, options),
            *summary["warnings"],
            *pipeline_warnings,
        ],
        "window": window,
        "time_range": _time_range(options),
        "pull": {
            "matched_spans": collection.spans,
            "traces": collection.traces,
            "accepted": len(collection.rows),
            "added": added.added,
            "unchanged": added.unchanged,
            "excluded": added.excluded,
            "retired": added.retired,
            "updated": added.updated,
            **collection.stats(),
        },
        **following,
        "hint": (
            "review a sample of the accepted rows (`pup finetune dataset sample "
            f"{_quoted(directory)}`) and the exclusions ({dataset.excluded_path.name}) with "
            "the user before training, and record which rows to keep with `pup finetune "
            "dataset label`"
        ),
    }


def _with_frozen(dataset: DatasetDir, directory: Path, values: Mapping[str, Any]) -> dict[str, Any]:
    """``dataset pull``'s source flags ``values`` (by parameter name), each one left out
    taking the value the directory froze on its first pull.

    A flag given with another value than the frozen one is a :class:`SettingsConflict`
    naming the frozen value (any difference this misses, the settings check reports).
    Without frozen settings, ``--ml-app`` is required.
    """
    frozen = _frozen_pull_values(dataset)
    if frozen is None:
        if values["ml_app"] is None:
            if dataset.is_dataset():
                dataset.check_source("datadog")  # a dataset built by `dataset import`
            raise UsageError(
                f"dataset pull needs --ml-app ({display_path(directory)} has no settings yet)",
                hint="the first pull into a directory names the ml_app whose spans to read, "
                "and freezes it with the filter; a rerun takes them from the directory",
            )
        return dict(values)
    ctx = current_context()
    merged = dict(values)
    filled = [
        name for name in values if values[name] is None or ctx is None or not given(ctx, name)
    ]
    if ctx is not None:
        conflicts = [
            f"{flag_name(ctx.command, name)} {_shown_value(name, frozen[name])}"
            for name in values
            if name not in filled
            and _comparable(name, values[name]) != _comparable(name, frozen[name])
        ]
        if conflicts:
            raise SettingsConflict(
                f"{display_path(directory)} froze {', '.join(conflicts)} on its first pull",
                hint="leave those flags out to use the frozen values, or pull into a new "
                "directory for other settings",
            )
    for name in filled:
        merged[name] = frozen[name]
    if filled and ctx is not None:
        # Named only where the frozen value is not the flag's default.
        flags = ", ".join(
            flag_name(ctx.command, name)
            for name in filled
            if _comparable(name, frozen[name]) != _comparable(name, values[name])
        )
        if flags:
            info(f"{display_path(directory)} has frozen settings: using its {flags}")
    return merged


def _comparable(name: str, value: Any) -> Any:
    """A pull flag's value as the settings freeze it, so ``7d`` and ``now-7d`` compare
    equal, and so do tags in another order."""
    if name in ("from_", "to"):
        return parse_time_bound(str(value), "--from" if name == "from_" else "--to")
    if name == "tag":
        return _parse_tags(value or ())
    if name in ("span_name", "query"):
        return value or None
    return value


def _shown_value(name: str, value: Any) -> str:
    """A frozen flag value as an error shows it: tags as ``k:v`` words, a relative --from
    the short way (``7d``, as the help writes its default), text quoted."""
    if name == "from_" and isinstance(value, str):
        value = _short_time(value)
    if value is None or value == []:
        return "(none)"
    if isinstance(value, list):
        return " ".join(shlex.quote(str(item)) for item in value)
    if isinstance(value, bool):
        return "on" if value else "off"
    return shlex.quote(str(value))


def _frozen_pull_values(dataset: DatasetDir) -> dict[str, Any] | None:
    """The source flags the first pull into ``dataset`` froze, by parameter name, as they
    would be passed; ``None`` when ``dataset`` has no pull settings."""
    if not dataset.is_dataset():
        return None
    source = dataset.settings().get("source")
    if not isinstance(source, dict) or source.get("kind") != "datadog":
        return None
    raw_filter = source.get("filter")
    filter_: dict[str, Any] = raw_filter if isinstance(raw_filter, dict) else {}
    tags = filter_.get("tags")
    split = dataset.split_params()

    def expr(key: str, bound: str, default: str) -> str:
        value = source.get(key, filter_.get(bound))
        return default if value is None else str(value)

    return {
        "ml_app": filter_.get("ml_app"),
        "span_name": filter_.get("span_name"),
        "query": filter_.get("query"),
        "tag": [f"{key}:{value}" for key, value in sorted(tags.items())]
        if isinstance(tags, dict) and tags
        else None,
        "from_": expr("from_expr", "from", DEFAULT_FROM),
        "to": expr("to_expr", "to", DEFAULT_TO),
        "reasoning": source.get("reasoning", "omit"),
        "include_errors": bool(source.get("include_errors", False)),
        "seed": split.seed,
        "validation_fraction": split.validation_fraction,
        "test_fraction": split.test_fraction,
    }


@dataset_app.command(**LABEL_DOC.typer())
@run_command
def label(
    directory: DirArgument,
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            metavar="FILE",
            help="Labels, one JSON object per line.",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
        ),
    ],
) -> dict[str, Any]:
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    labels = _read_label_file(file, at=datetime.now(UTC))
    report = apply_labels(dataset, labels)
    summary = report.summary
    result = {
        "dataset_dir": display_path(dataset.path),
        "label": {
            "file": str(file),
            "labels": len(labels),
            "applied": report.applied,
            "unchanged": report.unchanged,
            "kept": report.kept,
            "dropped": report.dropped,
        },
        "accepted": summary["accepted"],
        "kept": summary["kept"],
        "dropped": summary["dropped"],
        "drop_reasons": summary["drop_reasons"],
        "splits": summary["splits"],
        "warnings": summary["warnings"],
    }
    return merge_next(result, dataset_next(dataset, directory, summary=summary))


@dataset_app.command(**SAMPLE_DOC.typer())
@run_command
def sample(
    directory: DirArgument,
    count: Annotated[
        int, typer.Option("-n", "--count", min=1, metavar="N", help="How many rows to show.")
    ] = DEFAULT_SAMPLE_SIZE,
    split: Annotated[
        SampleSplit,
        typer.Option("--split", metavar="SPLIT", help="train, validation, test or all."),
    ] = "all",
    seed: Annotated[
        int, typer.Option(metavar="N", help="Seed that picks the rows.")
    ] = DEFAULT_SAMPLE_SEED,
    include_dropped: Annotated[
        bool,
        typer.Option("--include-dropped", help="Also sample rows labeled drop."),
    ] = False,
    old_count: Annotated[
        int | None, typer.Option("--n", min=1, hidden=True, help="The old name of --count.")
    ] = None,
) -> dict[str, Any]:
    n = renamed("count", "old_count")
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    dataset.settings()  # A UsageError when this is not a dataset directory.
    rows, eligible = sample_rows(
        dataset, n=n, split=split, seed=seed, include_dropped=include_dropped
    )
    result = {
        "dataset_dir": display_path(dataset.path),
        "split": split,
        "seed": seed,
        "include_dropped": include_dropped,
        "eligible": eligible,
        "rows": rows,
        "rows_file": display_path(dataset.rows_path),
        "hint": (
            "review these rows with the user, then record decisions as JSON lines of "
            '{"id": ..., "keep": true|false, "reason": ...}; a reason is required to drop'
        ),
    }
    return merge_next(result, dataset_next(dataset, directory, sampled=True))


def _push_brief(dataset: DatasetDir) -> dict[str, Any] | None:
    """Where the dataset was pushed (from ``push.json``), or ``None`` when it never was."""
    state = read_push_state(dataset)
    if state is None:
        return None
    return {
        "project": state.project,
        "dataset_name": state.dataset_name,
        "dataset_id": state.dataset_id,
        "dataset_version": state.dataset_version,
        "records": len(state.record_ids),
        "pushed_at": state.pushed_at,
    }


def _read_label_file(path: Path, *, at: datetime) -> list[Label]:
    """The labels in ``path``, all ``manual``; ``at`` fills in lines that leave it out.

    A line claiming ``"by": "judge"`` is refused: judge labels come only from the council."""
    labels = parse_labels(
        path,
        defaults={"at": at},
        error=UsageError,
        hint='each line must be {"id": ..., "keep": true|false, "reason": ...}',
    )
    if not labels:
        raise UsageError(f"{path} has no labels")
    judged = next((label.id for label in labels if label.by == "judge"), None)
    if judged is not None:
        raise UsageError(
            f"{path} labels {judged!r} as decided by the judge council",
            hint='leave "by" out: `dataset label` records your decisions, and judge labels '
            "come only from `pup finetune dataset triage`",
        )
    return labels


# --- pull helpers ---------------------------------------------------------------------------


def _pull_options(
    *,
    ml_app: str,
    span_name: str | None,
    query: str | None,
    tags: Sequence[str] | None,
    from_: str,
    to: str,
    target_count: int,
    max_candidates: int,
    reasoning: ReasoningMode,
    include_errors: bool,
) -> PullOptions:
    if not ml_app.strip():
        raise UsageError("--ml-app must not be empty")
    span_query = SpanQuery(
        ml_app=ml_app,
        span_name=span_name or None,
        query=query or None,
        tags=_parse_tags(tags or ()),
        from_=_time_bound(from_, "--from"),
        to=_time_bound(to, "--to"),
    )
    return PullOptions(
        span_query,
        reasoning=reasoning,
        include_errors=include_errors,
        target_count=target_count,
        max_candidates=max_candidates,
    )


def _parse_tags(values: Iterable[str]) -> dict[str, str]:
    """``--tag KEY:VALUE`` values as a mapping; a key given two values is an error."""
    tags: dict[str, str] = {}
    for value in values:
        key, sep, item = value.partition(":")
        if not sep or not key.strip() or not item.strip():
            raise UsageError(f"--tag must be KEY:VALUE, not {value!r}")
        if tags.setdefault(key, item) != item:
            raise UsageError(f"--tag {key} is given twice, as {tags[key]!r} and {item!r}")
    return tags


def _time_bound(value: str, flag: str) -> str | int:
    """A ``--from`` or ``--to`` value, checked before anything is frozen or fetched
    (:func:`~pup_finetune.datadog.spans.parse_time_bound`)."""
    return parse_time_bound(value, flag)


def _window(options: PullOptions, *, frozen: bool) -> dict[str, Any]:
    """The window the export reads, in UTC ISO 8601 (an ISO 8601 bound as written), and
    whether it is the one the directory's first pull froze."""
    from_, to = options.export_query.from_, options.export_query.to
    return {
        "from": _utc(from_) or from_,
        "to": _utc(to) or to,
        "frozen": frozen,
    }


def _time_range(options: PullOptions) -> dict[str, Any]:
    """The time range as written and as exported; an epoch ms bound also in UTC ISO 8601."""
    from_, to = options.export_query.from_, options.export_query.to
    return {
        "from_expr": options.query.from_,
        "to_expr": options.query.to,
        "from": from_,
        "to": to,
        "from_utc": _utc(from_),
        "to_utc": _utc(to),
    }


def _utc(bound: TimeBound) -> str | None:
    if not isinstance(bound, int):
        return None
    try:
        moment = datetime.fromtimestamp(bound / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _short_time(bound: str) -> str:
    """A canonical relative --from the way the help writes it: ``now-7d`` as ``7d``
    (``--to 1d`` would read as a duration, so a --to keeps its ``now-``)."""
    return bound.removeprefix("now-")


def _explanations(excluded_by_code: dict[str, int]) -> dict[str, str]:
    """What the most frequent exclusion codes mean, most frequent first."""
    ranked = sorted(excluded_by_code.items(), key=lambda item: (-item[1], item[0]))
    explained: dict[str, str] = {}
    for code, _count in ranked[:EXPLAINED_CODES]:
        text = explain(code)
        if text is not None:
            explained[code] = text
    return explained


def _collection_warnings(collection: Collection, options: PullOptions) -> list[str]:
    warnings: list[str] = []
    unreadable = collection.export.invalid_spans
    if not collection.spans and unreadable:
        warnings.append(
            f"none of the {unreadable} spans returned could be read (they lack the fields of "
            "an LLM span, such as a trace id), so there was nothing to convert"
        )
        return warnings
    if not collection.spans:
        warnings.append(
            "no LLM spans matched; check --ml-app, --span-name, --query, --tag and the time "
            "range (--from, --to)"
        )
    elif collection.counted < options.target_count:
        if collection.stopped == "max_candidates":
            warnings.append(
                f"read --max-candidates {options.max_candidates} spans before "
                f"--target-count {options.target_count} rows were accepted; raise "
                "--max-candidates to read further"
            )
        else:
            warnings.append(
                f"only {collection.counted} rows were accepted from every matching span, "
                f"fewer than --target-count {options.target_count}; widen the filter or the "
                "time range for more"
            )
    if unreadable:
        warnings.append(
            f"{unreadable} spans could not be read and were skipped (excluded as unreadable_span)"
        )
    return warnings


def register(app: typer.Typer) -> None:
    """Add the `dataset` command group to ``app``."""
    app.add_typer(dataset_app, name="dataset", rich_help_panel=BUILD_A_DATASET)
