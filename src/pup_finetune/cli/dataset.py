# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset`: build and inspect local dataset directories."""

from __future__ import annotations

import re
import shlex
import time
from collections import Counter
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
    row_summary,
    sample_rows,
)
from pup_finetune.data.convert import ReasoningMode
from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings, read_import_file
from pup_finetune.data.exclusions import Exclusion, explain
from pup_finetune.data.pull import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_TARGET_COUNT,
    Collection,
    PullOptions,
    collect,
    frozen_window,
    pull,
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
from pup_finetune.data.schema import TrainingRow, split_key
from pup_finetune.data.split import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    SPLITS,
    SplitName,
    SplitParams,
    skew_warnings,
    split_counts,
    split_warnings,
)
from pup_finetune.datadog.datasets import DEFAULT_PROJECT, RECORDS_BATCH_SIZE
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import (
    RELATIVE_TIME_UNITS_MS,
    Clock,
    SpanQuery,
    TimeBound,
    epoch_ms,
    parse_time_bound,
)
from pup_finetune.display import display_path
from pup_finetune.errors import SettingsConflict, UsageError, require_confirm, require_writable
from pup_finetune.next_steps import command, step, template
from pup_finetune.output import info
from pup_finetune.workspace import ParentSetup, ignore_new_parent

SAMPLE_IDS_PER_SPLIT = 5
SAMPLE_SIZE = 5
"""How many accepted and excluded rows a preview shows."""
EXPLAINED_CODES = 5
"""How many of the most frequent exclusion codes get an explanation in the output."""

DEFAULT_FROM = "7d"
DEFAULT_TO = "now"
_DEFAULT_FROM_BOUND = parse_time_bound(DEFAULT_FROM, "--from")
"""The default ``--from`` as the query holds it (``now-7d``)."""
WIDER_FROM = "30d"
"""The ``--from`` a preview that accepted no row suggests when it read less than this."""
_WIDER_FROM_MS = 30 * RELATIVE_TIME_UNITS_MS["d"]

REVIEW_SAMPLE_SIZE = 20
"""How many rows the ``dataset sample`` a dataset's ``next_command`` names shows."""
DATASETS_DIR = Path("datasets")
"""Where ``dataset preview`` proposes to pull into: ``datasets/<ml_app>`` under the cwd."""

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
"""The option panels of `dataset pull` and `dataset preview` (and `import`'s splits)."""

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

MlAppOption = Annotated[
    str,
    typer.Option(
        "--ml-app",
        metavar="NAME",
        help="The ml_app whose LLM spans to read.",
        show_default=False,
        rich_help_panel=FILTER_PANEL,
    ),
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

_DATADOG_READ = Safety(
    data="reads Datadog, prints sample rows",
    recipients="your terminal: samples of trace content are printed (and reach the agent's "
    "model when an agent runs it); reads Datadog through pup",
    without="previews rows in memory",
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

PREVIEW_DOC = CommandDoc(
    path="dataset preview",
    short="Try a Datadog span filter; writes nothing.",
    description=describe(
        "Read --ml-app's LLM spans from Datadog and convert them into training rows in "
        "memory, writing nothing. Shows the time range read, how many rows would be accepted "
        "and why the rest would not, a few samples, and the `dataset pull` command with the "
        "same flags.",
        "The samples are trace content, printed to your terminal, so when an agent runs the "
        "preview they reach the agent's model. The pull goes into a new directory under "
        f"./datasets named after --ml-app. {_PULL_FLAGS}",
    ),
    examples=(
        "pup finetune dataset preview --ml-app my-agent",
        "pup finetune dataset preview --ml-app my-agent --span-name chat --from 3d",
    ),
    safety=_DATADOG_READ,
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
    judge council, are the steps. A triage that is not complete gets a step to finish it.
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
    if not state.labeled:
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


@dataset_app.command(**PREVIEW_DOC.typer())
@run_command
def preview(
    ml_app: MlAppOption,
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
    params = SplitParams(seed, validation_fraction, test_fraction)
    options = _pull_options(
        ml_app=ml_app,
        span_name=span_name,
        query=query,
        tags=tag,
        from_=from_,
        to=to,
        target_count=target_count,
        max_candidates=max_candidates,
        reasoning=reasoning,
        include_errors=include_errors,
    )
    options = options.resolved(epoch_ms(clock))
    collection = collect(Pup(), options, progress=info)
    excluded_by_code = _count_codes(exclusion.code.value for exclusion in collection.exclusions)
    counts = split_counts(collection.rows, params)
    warnings = _collection_warnings(collection, options)
    if collection.rows:
        # With no row accepted, the splits are empty for that reason alone.
        warnings += [
            *split_warnings(counts),
            *skew_warnings(_keys_by_split(collection.rows, params), params, more="pull more rows"),
        ]
    return {
        "query": options.settings(),
        "time_range": _time_range(options),
        "limits": {"target_count": target_count, "max_candidates": max_candidates},
        "matched_spans": collection.spans,
        "traces": collection.traces,
        "candidate_rows": (
            len(collection.rows) + collection.over_target + len(collection.exclusions)
        ),
        "accepted": len(collection.rows),
        "excluded": len(collection.exclusions),
        "excluded_by_code": excluded_by_code,
        "exclusion_explanations": _explanations(excluded_by_code),
        "splits": counts,
        "warnings": warnings,
        "sample_accepted": [row_summary(row) for row in collection.rows[:SAMPLE_SIZE]],
        "sample_excluded": _sample_exclusions(collection.exclusions),
        "stats": collection.stats(),
        "next_command": (
            _pull_command(_new_dataset_dir(ml_app), options, params) if collection.rows else None
        ),
        "next_steps": (
            [] if collection.rows else _no_rows_steps(collection, options, params, excluded_by_code)
        ),
    }


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
    if collection.stopped == "max_candidates" and len(collection.rows) < options.target_count:
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


def _new_dataset_dir(ml_app: str) -> str:
    """``./datasets/<ml_app>`` (made safe for a path), with ``-2``, ``-3``, ... when taken."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", ml_app.strip().lower()).strip("-.") or "dataset"
    candidate, n = DATASETS_DIR / slug, 2
    while candidate.exists():
        candidate, n = DATASETS_DIR / f"{slug}-{n}", n + 1
    return f"./{candidate}"


def _short_time(bound: str) -> str:
    """A canonical relative --from the way the help writes it: ``now-7d`` as ``7d``
    (``--to 1d`` would read as a duration, so a --to keeps its ``now-``)."""
    return bound.removeprefix("now-")


def _pull_command(directory: str, options: PullOptions, params: SplitParams) -> str:
    """The ``dataset pull`` command into ``directory`` with the flags that differ from the
    defaults (the filter, and the time range as written)."""
    return command("dataset", "pull", directory, *_source_args(options, params))


def _source_args(
    options: PullOptions,
    params: SplitParams,
    *,
    from_: str | None = None,
    include_errors: bool | None = None,
) -> list[str]:
    """The flags of a preview or pull that differ from the defaults: the filter, the time
    range as written (a relative --from the short way), the limits, rows and splits.
    ``from_`` and ``include_errors`` replace the options' own."""
    query = options.query
    args: list[str] = ["--ml-app", query.ml_app or ""]
    if query.span_name:
        args += ["--span-name", query.span_name]
    if query.query:
        args += ["--query", query.query]
    for key, value in sorted(query.tags.items()):
        args += ["--tag", f"{key}:{value}"]
    if from_ is not None:
        args += ["--from", from_]
    elif str(query.from_) != _DEFAULT_FROM_BOUND:
        args += ["--from", _short_time(str(query.from_))]
    if str(query.to) != DEFAULT_TO:
        args += ["--to", str(query.to)]
    if options.target_count != DEFAULT_TARGET_COUNT:
        args += ["--target-count", str(options.target_count)]
    if options.max_candidates != DEFAULT_MAX_CANDIDATES:
        args += ["--max-candidates", str(options.max_candidates)]
    if options.reasoning != "omit":
        args += ["--reasoning", options.reasoning]
    if options.include_errors if include_errors is None else include_errors:
        args.append("--include-errors")
    if params != SplitParams():
        args += [
            "--seed",
            str(params.seed),
            "--validation-fraction",
            str(params.validation_fraction),
            "--test-fraction",
            str(params.test_fraction),
        ]
    return args


_RELATIVE = re.compile(r"now-(?P<amount>[0-9]+)(?P<unit>[smhdw])")


def _no_rows_steps(
    collection: Collection,
    options: PullOptions,
    params: SplitParams,
    excluded_by_code: Mapping[str, int],
) -> list[dict[str, str]]:
    """What a preview that accepted no row suggests instead of a pull: keeping errored
    calls when they are what was excluded, a wider time range, and checking the filter."""
    steps: list[dict[str, str]] = []
    if collection.spans and excluded_by_code.get("span_error") and not options.include_errors:
        steps.append(
            step(
                template("dataset", "preview", *_source_args(options, params, include_errors=True)),
                f"{excluded_by_code['span_error']} spans were excluded as span_error: keep the "
                "errored LLM calls too",
            )
        )
    wider = _wider_from(options.query.from_)
    read = "every span read was excluded" if collection.spans else "no LLM span matched"
    steps.append(
        step(
            template("dataset", "preview", *_source_args(options, params, from_=wider)),
            f"{read} in the time range read; widen it with an earlier --from"
            + (" (an ISO 8601 date, or a relative time such as 90d)" if wider == "<FROM>" else ""),
        )
    )
    steps.append(
        step(
            template("dataset", "preview", "--ml-app", "<ML_APP>"),
            "check the filter: --ml-app must be the ml_app your agent reports its LLM spans "
            "under, and --span-name, --query and --tag each narrow it further, so leave them "
            "out to see every span",
        )
    )
    return steps


def _wider_from(bound: TimeBound) -> str:
    """The ``--from`` a preview that accepted no row suggests: :data:`WIDER_FROM` when the
    range read is a shorter relative one, else a placeholder."""
    match = _RELATIVE.fullmatch(bound) if isinstance(bound, str) else None
    if match is None:
        return "<FROM>"
    read_ms = int(match["amount"]) * RELATIVE_TIME_UNITS_MS[match["unit"]]
    return WIDER_FROM if read_ms < _WIDER_FROM_MS else "<FROM>"


def _keys_by_split(rows: Iterable[TrainingRow], params: SplitParams) -> dict[SplitName, list[str]]:
    """The split key of each of ``rows`` by the split ``params`` would assign it."""
    keys: dict[SplitName, list[str]] = {split: [] for split in SPLITS}
    for row in rows:
        key = split_key(row)
        keys[params.assign(key)].append(key)
    return keys


def _count_codes(codes: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(codes).items()))


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
    elif len(collection.rows) < options.target_count:
        if collection.stopped == "max_candidates":
            warnings.append(
                f"read --max-candidates {options.max_candidates} spans before "
                f"--target-count {options.target_count} rows were accepted; raise "
                "--max-candidates to read further"
            )
        else:
            warnings.append(
                f"only {len(collection.rows)} rows were accepted from every matching span, "
                f"fewer than --target-count {options.target_count}; widen the filter or the "
                "time range for more"
            )
    if unreadable:
        warnings.append(
            f"{unreadable} spans could not be read and were skipped (excluded as unreadable_span)"
        )
    return warnings


def _sample_exclusions(exclusions: Sequence[Exclusion]) -> list[dict[str, Any]]:
    """Up to :data:`SAMPLE_SIZE` exclusions: one per code first (most frequent codes first)."""
    counts = Counter(exclusion.code.value for exclusion in exclusions)
    firsts: dict[str, Exclusion] = {}
    for exclusion in exclusions:
        firsts.setdefault(exclusion.code.value, exclusion)
    ordered = sorted(firsts.values(), key=lambda e: (-counts[e.code.value], e.code.value))
    picked = ordered[:SAMPLE_SIZE]
    for exclusion in exclusions:
        if len(picked) >= SAMPLE_SIZE:
            break
        if not any(exclusion is chosen for chosen in picked):
            picked.append(exclusion)
    return [exclusion.to_json() for exclusion in picked]


def register(app: typer.Typer) -> None:
    """Add the `dataset` command group to ``app``."""
    app.add_typer(dataset_app, name="dataset", rich_help_panel=BUILD_A_DATASET)
