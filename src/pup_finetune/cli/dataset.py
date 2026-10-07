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
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import typer

from pup_finetune.cli._common import get_runtime, run_command
from pup_finetune.cli._dirs import require_dataset_dir
from pup_finetune.curation.labels import Label, apply_labels, parse_labels
from pup_finetune.curation.samples import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    SampleSplit,
    row_brief,
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
    pull,
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
    split_counts,
    split_warnings,
)
from pup_finetune.datadog.datasets import DEFAULT_PROJECT
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import Clock, SpanQuery, TimeBound, epoch_ms
from pup_finetune.errors import StateError, UsageError, require_confirm, require_writable
from pup_finetune.next_steps import command, step, template
from pup_finetune.output import info

SAMPLE_IDS_PER_SPLIT = 5
SAMPLE_SIZE = 5
"""How many accepted and excluded rows a preview shows."""
EXPLAINED_CODES = 5
"""How many of the most frequent exclusion codes get an explanation in the output."""

DEFAULT_FROM = "now-7d"
DEFAULT_TO = "now"

REVIEW_SAMPLE_SIZE = 20
"""How many rows the ``dataset sample`` a dataset's ``next_command`` names shows."""
DATASETS_DIR = Path("datasets")
"""Where ``dataset preview`` proposes to pull into: ``datasets/<ml_app>`` under the cwd."""

clock: Clock = time.time
"""What ``now`` in ``--from`` and ``--to`` is resolved against; tests replace it."""

_LOCAL_ONLY = "Reads and writes local files only, so it is allowed with PUP_READ_ONLY set."
_DATADOG_READ = "It only reads from Datadog (through pup), so it is allowed with PUP_READ_ONLY set."

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
SeedOption = Annotated[int, typer.Option(help="Seed for the split assignment.")]
ValidationFractionOption = Annotated[
    float, typer.Option(help="Share of traces assigned to validation.")
]
TestFractionOption = Annotated[float, typer.Option(help="Share of traces assigned to test.")]

MlAppOption = Annotated[
    str,
    typer.Option(
        "--ml-app", help="The ml_app whose LLM spans to read (required).", show_default=False
    ),
]
SpanNameOption = Annotated[
    str | None,
    typer.Option("--span-name", help="Only spans with this name.", show_default=False),
]
QueryOption = Annotated[
    str | None,
    typer.Option(
        "--query",
        help="Only spans that also match this search query (event search syntax).",
        show_default=False,
    ),
]
TagOption = Annotated[
    list[str] | None,
    typer.Option(
        "--tag",
        metavar="KEY:VALUE",
        help="Only spans with this tag; repeat for more (all must match).",
        show_default=False,
    ),
]
FromOption = Annotated[
    str,
    typer.Option(
        "--from",
        help=(
            "Start of the time range: ISO 8601, epoch ms, or a relative time: now, or "
            "now-<n><unit> with unit s, m, h, d or w, such as now-7d. A relative time is "
            "resolved once, when the command starts."
        ),
    ),
]
ToOption = Annotated[
    str, typer.Option("--to", help="End of the time range, in the same forms as --from.")
]
TargetCountOption = Annotated[
    int,
    typer.Option(min=1, help="Stop reading once this many rows are accepted."),
]
MaxCandidatesOption = Annotated[
    int,
    typer.Option(
        min=1, help="Read at most this many exported spans, counting ones that cannot be read."
    ),
]
ReasoningOption = Annotated[
    ReasoningMode,
    typer.Option(
        help="Drop reasoning messages, or keep them as the assistant's reasoning_content."
    ),
]
IncludeErrorsOption = Annotated[
    bool,
    typer.Option(
        "--include-errors",
        help="Keep errored LLM calls instead of excluding them as span_error.",
    ),
]

_PULL_FLAGS = (
    "Spans are LLM spans of --ml-app in the time range, narrowed by --span-name, --query and "
    "--tag. Each span becomes a conversation (input then output messages; a plain input or "
    "output value is the user or assistant message); per trace, a conversation another one "
    "contains is dropped, so sub-agents stay their own rows. "
    "Errored LLM calls are excluded as span_error unless --include-errors is passed. Reading "
    "stops once --target-count rows are accepted or --max-candidates spans are read, "
    "unreadable ones included."
)


def _quoted(path: Path) -> str:
    return shlex.quote(str(path))


def dataset_next(
    dataset: DatasetDir, directory: Path, *, accepted: int, sampled: bool = False
) -> dict[str, Any]:
    """``next_command`` and ``next_steps`` for a dataset directory, from its state.

    With accepted rows and none labeled, the next command is a ``dataset sample`` to
    review (``None`` right after one, ``sampled``), and labeling them, by hand or with a
    judge council, are the steps. Once rows are labeled, training is: its preview needs a
    provider, a model and a run directory that cannot be chosen without guessing, so it
    is a step, with a ``dataset push`` while the dataset was never pushed. ``directory``
    is the directory as the user typed it.
    """
    if not accepted:
        return {"next_command": None, "next_steps": []}
    label_step = step(
        template("dataset", "label", directory, "--file", "<LABELS_FILE>"),
        'record keep/drop decisions: write <LABELS_FILE> first, one {"id", "keep", '
        '"reason"} object per line (a reason is required to drop)',
    )
    if not dataset.labels():
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
        sample = command("dataset", "sample", directory, "--n", REVIEW_SAMPLE_SIZE)
        return {
            "next_command": None if sampled else sample,
            "next_steps": [label_step, triage_step],
        }
    steps = [label_step] if sampled else []
    steps += [
        step(
            template("models", "list", "--provider", "<PROVIDER>"),
            "see the models a provider can fine-tune",
        ),
        step(
            template(
                "train",
                "--provider",
                "<PROVIDER>",
                "--model",
                "<MODEL>",
                "--dataset",
                directory,
                "--run-dir",
                "<RUN_DIR>",
            ),
            "preview a training run on the kept rows (free); choose the provider and model "
            "with the user, and a new run directory",
        ),
    ]
    pushed = _push_receipt(dataset)
    if pushed is None:
        steps.append(
            step(
                template("dataset", "push", directory, "--name", "<NAME>"),
                "push the kept rows to Datadog so an evaluation can be published as "
                "experiments (a remote write; it previews first)",
            )
        )
    elif pushed is _UNREADABLE:
        steps.append(
            step(
                command("dataset", "show", directory),
                f"{PUSH_FILE} cannot be read, so whether the dataset was pushed is unknown; "
                "show reports why",
            )
        )
    return {"next_command": None, "next_steps": steps}


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
    seed: SeedOption = DEFAULT_SEED,
    validation_fraction: ValidationFractionOption = DEFAULT_VALIDATION_FRACTION,
    test_fraction: TestFractionOption = DEFAULT_TEST_FRACTION,
) -> dict[str, Any]:
    params = SplitParams(seed, validation_fraction, test_fraction)
    require_dataset_dir(directory)
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
        **dataset_next(dataset, directory, accepted=summary["accepted"]),
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
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    settings = dataset.settings()
    by_split = dataset.rows_by_split()
    summary = dataset.summary()
    return {
        "dataset_dir": str(dataset.path.resolve()),
        "settings": settings,
        "summary": summary,
        "sample_row_ids": {
            split: [row.id for row in by_split[split][:SAMPLE_IDS_PER_SPLIT]] for split in SPLITS
        },
        "push": _push_brief(dataset),
        **dataset_next(dataset, directory, accepted=summary["accepted"]),
    }


@dataset_app.command(
    "push",
    short_help="Push a dataset's kept rows to an Agent Observability dataset.",
    help=(
        "Add one record per kept row of DIR to the Agent Observability dataset --name in "
        "project --project, creating either when missing; rows labeled drop are never sent. "
        "Each record's input is the conversation before the final assistant message, with "
        "its tools; its expected output is that message; its metadata holds the row id, "
        "split and source; it is tagged split:<split> and source:pup-finetune. Without "
        "--confirm it only previews what would be sent; when every kept row was pushed "
        "already it reports the receipt and sends nothing. The receipt, push.json in DIR, "
        "records the dataset and the record of every row pushed and is saved after every "
        "batch, so pushing again sends only rows not yet pushed, and an interrupted push "
        "resumes without adding a row twice. A directory pushes to one dataset: another --name or --project is an "
        "error. Rows labeled drop after they were pushed are reported as stale_records and "
        "are not deleted from the dataset; kept rows that changed after they were pushed "
        "are reported as changed_records and their records are not updated. A remote "
        "write: with PUP_READ_ONLY set, it previews (from local files) but refuses --confirm."
    ),
)
@run_command
def push(
    directory: DirArgument,
    name: Annotated[
        str,
        typer.Option("--name", help="The dataset to push to (required).", show_default=False),
    ],
    project: Annotated[
        str, typer.Option("--project", help="The project the dataset belongs to.")
    ] = DEFAULT_PROJECT,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm", help="Push (a remote write); without it, only preview what is sent."
        ),
    ] = False,
) -> dict[str, Any]:
    runtime = get_runtime()
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    plan = plan_push(dataset, project=project, name=name)
    plan.require_rows()
    if plan.needs_remote:
        rerun = ["dataset", "push", directory, "--name", name]
        if project != DEFAULT_PROJECT:
            rerun += ["--project", project]
        if confirm:
            require_writable(runtime, "dataset push")
        # The preview reads local files only, so it is allowed under PUP_READ_ONLY.
        require_confirm(
            confirm,
            _push_preview(plan),
            command(*rerun, "--confirm"),
            read_only=runtime.read_only,
        )
        receipt = push_dataset(dataset, project=project, name=name, pup=Pup(), progress=info)
    else:
        info("nothing to push: every kept row was pushed already")
        receipt = plan.receipt()
    result: dict[str, Any] = {"dataset_dir": str(dataset.path.resolve()), **receipt.to_json()}
    result["receipt_file"] = str(push_path(dataset).resolve())
    if warnings := _receipt_warnings(receipt.stale_records, receipt.changed_records):
        result["warnings"] = warnings
    result.update(dataset_next(dataset, directory, accepted=dataset.summary()["accepted"]))
    return result


def _push_preview(plan: PushPlan) -> dict[str, Any]:
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
        "dataset_dir": str(plan.directory.resolve()),
        "project": plan.project,
        "dataset_name": plan.dataset_name,
        "dataset_id": plan.dataset_id,
        "action": action,
        "records_to_push": to_push,
        "records_to_push_by_split": plan.pending_by_split(),
        "already_pushed": plan.already_pushed,
        "kept_by_split": plan.kept_by_split,
        "stale_records": plan.stale_records,
        "changed_records": plan.changed_records,
        "sample_record": plan.pending[0].record if plan.pending else None,
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


@dataset_app.command(
    "preview",
    short_help="Try a Datadog span filter without writing anything.",
    help=(
        "Export LLM spans from Datadog, check and convert them into training rows, and "
        "assign splits, all in memory: nothing is written. Shows the time range it read, how "
        "many spans matched, how many rows would be accepted, why the rest would be excluded, a few samples, and the "
        "`dataset pull` command with the same flags, into a new directory under ./datasets "
        f"named after --ml-app. {_PULL_FLAGS} {_DATADOG_READ}"
    ),
)
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
        "warnings": [*_collection_warnings(collection, options), *split_warnings(counts)],
        "sample_accepted": [row_brief(row) for row in collection.rows[:SAMPLE_SIZE]],
        "sample_excluded": _sample_exclusions(collection.exclusions),
        "stats": collection.stats(),
        "next_command": _pull_command(_new_dataset_dir(ml_app), options, params),
    }


@dataset_app.command(
    "pull",
    short_help="Pull training rows from Datadog into a dataset directory.",
    help=(
        "Export LLM spans from Datadog, check and convert them into training rows, and add "
        "them to DIR with their splits, creating it if needed. Excluded rows are recorded in "
        "excluded.jsonl with their reason. The filter, --reasoning, --include-errors and the "
        "split settings are frozen in DIR on the first pull; a pull with other values fails, "
        "so use a new directory. Raw pages are cached in DIR/raw/, so re-running an "
        "interrupted pull with the same flags resumes without fetching them again, and a "
        "higher --target-count or --max-candidates reads further (a stored row that turns "
        "out to be a prefix of a longer conversation of its trace is then excluded as "
        "contained_conversation). A relative --from or --to, such as the default now-7d, is "
        "resolved when the first pull starts and that window is frozen with the expressions; "
        "a re-run with the same expressions reads the same window. "
        f"{_PULL_FLAGS} {_DATADOG_READ}"
    ),
)
@run_command
def pull_rows(
    directory: DirArgument,
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
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    options = resolve_window(options, dataset, clock)
    collection, added = pull(Pup(), options, dataset, params, progress=info)
    summary = added.summary
    return {
        "dataset_dir": str(dataset.path.resolve()),
        "accepted": summary["accepted"],
        "excluded": summary["excluded"],
        "excluded_by_code": summary["excluded_by_code"],
        "exclusion_explanations": _explanations(summary["excluded_by_code"]),
        "splits": summary["splits"],
        "warnings": [*_collection_warnings(collection, options), *summary["warnings"]],
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
        **dataset_next(dataset, directory, accepted=summary["accepted"]),
        "hint": (
            "review a sample of the accepted rows (`pup finetune dataset sample "
            f"{_quoted(directory)}`) and the exclusions ({dataset.excluded_path.name}) with "
            "the user before training, and record which rows to keep with `pup finetune "
            "dataset label`"
        ),
    }


@dataset_app.command(
    "label",
    short_help="Apply keep/drop labels to a dataset's rows.",
    help=(
        "Record keep/drop decisions on accepted rows of DIR from a JSONL file of "
        '{"id", "keep", "reason"} objects ("at" may be given per line and defaults to the '
        "current time). A reason is required to drop a row. Rows labeled here are never "
        "relabeled by `dataset triage`. Every id must be "
        "an accepted row, or nothing is written. Labels are appended to labels.jsonl and the "
        "last label of an id wins; rows.jsonl is never changed. A dropped row is left out of "
        f"training and of the split files. {_LOCAL_ONLY}"
    ),
)
@run_command
def label(
    directory: DirArgument,
    file: Annotated[
        Path,
        typer.Option(
            "--file",
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
    return {
        "dataset_dir": str(dataset.path.resolve()),
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
        **dataset_next(dataset, directory, accepted=summary["accepted"]),
    }


@dataset_app.command(
    "sample",
    short_help="Show a small, readable sample of a dataset's rows.",
    help=(
        "Show up to --n rows of DIR in brief, to review them before labeling: each row's id, "
        "split, label (keep, drop or none), assistant turns, tools used, and its first user "
        "and final assistant messages cut to 300 characters (a final message without text "
        "shows its tool calls as name(args)). The full rows stay in rows.jsonl. The same "
        "--seed picks the same rows. Rows labeled drop are left out unless "
        f"--include-dropped is passed. {_LOCAL_ONLY}"
    ),
)
@run_command
def sample(
    directory: DirArgument,
    n: Annotated[int, typer.Option("--n", min=1, help="How many rows to show.")] = (
        DEFAULT_SAMPLE_SIZE
    ),
    split: Annotated[SampleSplit, typer.Option(help="The split to sample from.")] = "all",
    seed: Annotated[int, typer.Option(help="Seed that picks the rows.")] = DEFAULT_SAMPLE_SEED,
    include_dropped: Annotated[
        bool,
        typer.Option("--include-dropped", help="Also sample rows labeled drop."),
    ] = False,
) -> dict[str, Any]:
    require_dataset_dir(directory)
    dataset = DatasetDir(directory)
    dataset.settings()  # A UsageError when this is not a dataset directory.
    rows, eligible = sample_rows(
        dataset, n=n, split=split, seed=seed, include_dropped=include_dropped
    )
    return {
        "dataset_dir": str(dataset.path.resolve()),
        "split": split,
        "seed": seed,
        "include_dropped": include_dropped,
        "eligible": eligible,
        "rows": rows,
        "rows_file": str(dataset.rows_path.resolve()),
        **dataset_next(dataset, directory, accepted=len(dataset.rows()), sampled=True),
        "hint": (
            "review these rows with the user, then record decisions as JSON lines of "
            '{"id": ..., "keep": true|false, "reason": ...}; a reason is required to drop'
        ),
    }


_UNREADABLE: Final = object()
"""What :func:`_push_receipt` returns for a ``push.json`` that cannot be read."""


def _push_receipt(dataset: DatasetDir) -> object | None:
    """``dataset``'s push receipt, ``None`` when it was never pushed, or
    :data:`_UNREADABLE` when ``push.json`` cannot be read."""
    try:
        return read_push_state(dataset)
    except StateError:
        return _UNREADABLE


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
        from_=_time_bound(from_),
        to=_time_bound(to),
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


def _time_bound(value: str) -> str | int:
    """A ``--from`` or ``--to`` value: epoch milliseconds as an integer, anything else as is."""
    text = value.strip()
    if not text:
        raise UsageError("--from and --to must not be empty")
    return int(text) if text.isascii() and text.isdigit() else text


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


def _pull_command(directory: str, options: PullOptions, params: SplitParams) -> str:
    """The ``dataset pull`` command into ``directory`` with the flags that differ from the
    defaults (the filter, and the time range as written)."""
    query = options.query
    args: list[str] = ["--ml-app", query.ml_app or ""]
    if query.span_name:
        args += ["--span-name", query.span_name]
    if query.query:
        args += ["--query", query.query]
    for key, value in sorted(query.tags.items()):
        args += ["--tag", f"{key}:{value}"]
    if str(query.from_) != DEFAULT_FROM:
        args += ["--from", str(query.from_)]
    if str(query.to) != DEFAULT_TO:
        args += ["--to", str(query.to)]
    if options.target_count != DEFAULT_TARGET_COUNT:
        args += ["--target-count", str(options.target_count)]
    if options.max_candidates != DEFAULT_MAX_CANDIDATES:
        args += ["--max-candidates", str(options.max_candidates)]
    if options.reasoning != "omit":
        args += ["--reasoning", options.reasoning]
    if options.include_errors:
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
    return command("dataset", "pull", directory, *args)


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
    app.add_typer(dataset_app, name="dataset")
