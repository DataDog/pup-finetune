# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset`: build and inspect local dataset directories."""

from __future__ import annotations

import shlex
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import run_command
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
)
from pup_finetune.data.schema import TrainingRow
from pup_finetune.data.split import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    SPLITS,
    SplitParams,
    split_counts,
    split_warnings,
)
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import SpanQuery
from pup_finetune.errors import UsageError
from pup_finetune.output import info

SAMPLE_IDS_PER_SPLIT = 5
SAMPLE_SIZE = 5
"""How many accepted and excluded rows a preview shows."""
EXPLAINED_CODES = 5
"""How many of the most frequent exclusion codes get an explanation in the output."""

DEFAULT_FROM = "now-7d"
DEFAULT_TO = "now"

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
        help="Start of the time range: ISO 8601, date math such as now-7d, or epoch ms.",
    ),
]
ToOption = Annotated[
    str, typer.Option("--to", help="End of the time range, in the same forms as --from.")
]
TargetCountOption = Annotated[
    int,
    typer.Option(min=1, help="Stop reading once this many rows are accepted."),
]
MaxCandidatesOption = Annotated[int, typer.Option(min=1, help="Read at most this many spans.")]
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
    "--tag. Each span becomes a conversation (input then output messages); per trace, a "
    "conversation another one contains is dropped, so sub-agents stay their own rows. "
    "Errored LLM calls are excluded as span_error unless --include-errors is passed. Reading "
    "stops once --target-count rows are accepted or --max-candidates spans are read."
)


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
    seed: SeedOption = DEFAULT_SEED,
    validation_fraction: ValidationFractionOption = DEFAULT_VALIDATION_FRACTION,
    test_fraction: TestFractionOption = DEFAULT_TEST_FRACTION,
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


@dataset_app.command(
    "preview",
    short_help="Try a Datadog span filter without writing anything.",
    help=(
        "Export LLM spans from Datadog, check and convert them into training rows, and "
        "assign splits, all in memory: nothing is written. Shows how many spans matched, how "
        "many rows would be accepted, why the rest would be excluded, a few samples, and the "
        f"`dataset pull` command with the same flags. {_PULL_FLAGS} {_DATADOG_READ}"
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
    collection = collect(Pup(), options, progress=info)
    excluded_by_code = _count_codes(exclusion.code.value for exclusion in collection.exclusions)
    counts = split_counts(collection.rows, params)
    return {
        "query": options.settings(),
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
        "sample_accepted": [_describe_row(row) for row in collection.rows[:SAMPLE_SIZE]],
        "sample_excluded": _sample_exclusions(collection.exclusions),
        "stats": collection.stats(),
        "next_command": _pull_command("<DIR>", options, params),
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
        "contained_conversation). --from and --to are frozen as written: a relative time "
        "such as the default now-7d is resolved by Datadog when each page is fetched, so pass "
        "ISO 8601 or epoch ms to pin the window of a pull you may resume later. "
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
    dataset = DatasetDir(directory)
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
        "pull": {
            "matched_spans": collection.spans,
            "traces": collection.traces,
            "accepted": len(collection.rows),
            "added": added.added,
            "unchanged": added.unchanged,
            "excluded": added.excluded,
            "retired": added.retired,
            **collection.stats(),
        },
        "next_command": f"pup finetune dataset show {_quoted(directory)}",
        "hint": (
            f"review a sample of the accepted rows ({dataset.rows_path.name}) and the "
            f"exclusions ({dataset.excluded_path.name}) with the user before training, and "
            "decide which rows to keep"
        ),
    }


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


def _pull_command(directory: str, options: PullOptions, params: SplitParams) -> str:
    """The ``dataset pull`` command with the same flags; ``directory`` is used as is."""
    query = options.query
    args: list[str] = ["--ml-app", query.ml_app or ""]
    if query.span_name:
        args += ["--span-name", query.span_name]
    if query.query:
        args += ["--query", query.query]
    for key, value in sorted(query.tags.items()):
        args += ["--tag", f"{key}:{value}"]
    args += ["--from", str(query.from_), "--to", str(query.to)]
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
    return f"pup finetune dataset pull {directory} {shlex.join(args)}"


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


def _describe_row(row: TrainingRow) -> dict[str, Any]:
    """A row in brief: its id, its assistant turns and the tools it calls."""
    tools = {call.function.name for message in row.messages for call in message.tool_calls or ()}
    return {
        "id": row.id,
        "turns": sum(1 for message in row.messages if message.role == "assistant"),
        "tools_used": sorted(tools),
    }


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
