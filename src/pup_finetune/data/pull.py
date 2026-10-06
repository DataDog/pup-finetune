# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training rows from Datadog: export LLM spans, check them, convert them, and stop in time.

:func:`collect` is what ``dataset preview`` runs in memory and ``dataset pull`` runs into a
dataset directory (:func:`pull`):

1. Export the spans matching the query a page at a time
   (:func:`~pup_finetune.datadog.spans.iter_span_pages`), reading at most
   ``max_candidates`` spans.
2. Exclude the spans that fail a span-level check (:mod:`pup_finetune.data.checks`).
3. Convert the rest into rows per trace (:func:`~pup_finetune.data.trajectories.rows_from_spans`),
   and exclude a row repeating the conversation of an earlier one as
   ``duplicate_conversation``.
4. After each page, stop once ``target_count`` rows are accepted, before fetching the next
   page; the accepted rows are cut to ``target_count``.

Conversion groups spans by trace, so it runs again over every span read after each page;
the last run is the result. A trace whose spans straddle the page the export stopped at is
converted from the spans read so far. Items the export could not read at all are excluded as
``unreadable_span``.

``target_count`` and ``max_candidates`` bound how far the export reads, not which spans
match, so they are not part of the frozen :meth:`PullOptions.settings`: a pull can be re-run
with higher limits to read further. When it does, a trace an earlier pull read only in part
may now show that a stored row is a prefix of a longer conversation: :func:`pull` retires
such a row as ``contained_conversation`` (see :func:`retirements`), so the latest conversion
of every span it read is what the dataset holds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pup_finetune.data.checks import CheckOptions, check_span
from pup_finetune.data.convert import ReasoningMode, check_reasoning
from pup_finetune.data.dataset_dir import AddResult, DatasetDir, DatasetSettings, SourceKind
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.schema import Message, TrainingRow, row_content_hash
from pup_finetune.data.split import SplitParams
from pup_finetune.data.trajectories import ConversionStats, rows_from_spans
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import (
    CACHED_PAGES_NOTE,
    ExportStats,
    LLMObsSpan,
    SpanQuery,
    UnreadableSpan,
    iter_span_pages,
)
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.storage import DirectoryLock, make_private_dir

DEFAULT_TARGET_COUNT = 200
DEFAULT_MAX_CANDIDATES = 5000

SOURCE_KIND: SourceKind = "datadog"

StopReason = Literal["target_count", "max_candidates", "exhausted"]
"""Why reading stopped: enough rows, the span cap, or no matching span left."""

Progress = Callable[[str], None]


@dataclass(frozen=True)
class PullOptions:
    """What to export and convert, and when to stop reading."""

    query: SpanQuery
    reasoning: ReasoningMode = "omit"
    include_errors: bool = False
    """Keep errored LLM calls instead of excluding them as ``span_error``."""
    target_count: int = DEFAULT_TARGET_COUNT
    """Stop reading once this many rows are accepted."""
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    """Read at most this many spans."""

    def __post_init__(self) -> None:
        check_reasoning(self.reasoning)
        if self.target_count < 1:
            raise UsageError(f"--target-count must be at least 1, not {self.target_count}")
        if self.max_candidates < 1:
            raise UsageError(f"--max-candidates must be at least 1, not {self.max_candidates}")

    def settings(self) -> dict[str, Any]:
        """What selects and shapes the rows, to freeze in the dataset's settings."""
        return {
            **self.query.settings(),
            "reasoning": self.reasoning,
            "include_errors": self.include_errors,
        }

    def dataset_settings(self, split: SplitParams) -> DatasetSettings:
        return DatasetSettings(SOURCE_KIND, split, self.settings())


@dataclass
class Collection:
    """What :func:`collect` read and produced."""

    rows: list[TrainingRow] = field(default_factory=list)
    """Accepted rows, at most ``target_count``, in the order conversion produced them."""
    exclusions: list[Exclusion] = field(default_factory=list)
    """Span-level check failures first, then conversion and duplicate exclusions, then
    ``unreadable_span`` ones."""
    export: ExportStats = field(default_factory=ExportStats)
    conversion: ConversionStats = field(default_factory=ConversionStats)
    spans: int = 0
    """Distinct spans read (by span id)."""
    duplicate_spans: int = 0
    """Spans skipped because a span with the same id was already read."""
    traces: int = 0
    """Distinct traces among the spans read."""
    failed_checks: int = 0
    """Spans excluded by a span-level check."""
    over_target: int = 0
    """Accepted rows left out because ``target_count`` was already reached."""
    stopped: StopReason = "exhausted"
    read_ids: set[str] = field(default_factory=set)
    """The row id (``{trace_id}:{span_id}``) of every span read."""
    produced_ids: set[str] = field(default_factory=set)
    """The row ids the checks and the last conversion yielded, rows over the target included.
    A read id missing here was collapsed into another conversation of its trace."""

    def stats(self) -> dict[str, Any]:
        """The export, check and conversion counts, as JSON."""
        return {
            "stopped": self.stopped,
            "pages_fetched": self.export.pages_fetched,
            "pages_cached": self.export.pages_cached,
            "timeout_retries": self.export.timeout_retries,
            "unreadable_spans": self.export.invalid_spans,
            "duplicate_spans": self.duplicate_spans,
            "failed_checks": self.failed_checks,
            "conversations_collapsed": self.conversion.collapsed,
            "rows_over_target": self.over_target,
        }


def collect(
    pup: Pup,
    options: PullOptions,
    *,
    cache_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
) -> Collection:
    """Export, check and convert the spans ``options`` select; see the module docstring.

    ``cache_dir`` is the raw page cache :func:`~pup_finetune.datadog.spans.iter_span_pages`
    resumes from; without one nothing is written.
    """
    query = replace(options.query, max_spans=options.max_candidates)
    pages = iter_span_pages(pup, query, cache_dir=cache_dir, sleep=sleep)
    check_options = CheckOptions(include_errors=options.include_errors)
    result = Collection(export=pages.stats)
    passing: list[LLMObsSpan] = []
    seen: set[str] = set()
    traces: set[str] = set()
    rows: list[TrainingRow] = []
    converted: list[Exclusion] = []
    stopped: StopReason | None = None
    for number, page in enumerate(pages, start=1):
        added = False
        for span in page:
            if span.span_id in seen:
                result.duplicate_spans += 1
                continue
            seen.add(span.span_id)
            traces.add(span.trace_id)
            result.read_ids.add(f"{span.trace_id}:{span.span_id}")
            exclusion = check_span(span, check_options)
            if exclusion is None:
                passing.append(span)
                added = True
            else:
                result.exclusions.append(exclusion)
        if added:
            rows, converted, result.conversion = _convert(passing, options.reasoning)
        if progress is not None:
            progress(f"page {number}: {len(seen)} spans read, {len(rows)} rows accepted")
        if len(rows) >= options.target_count:
            stopped = "target_count"
            break
    result.spans = len(seen)
    result.traces = len(traces)
    result.failed_checks = len(result.exclusions)
    result.exclusions.extend(converted)
    result.produced_ids = {row.id for row in rows} | {
        e.row_id for e in result.exclusions if e.row_id is not None
    }
    result.exclusions.extend(_unreadable(span) for span in pages.stats.unreadable)
    result.rows = rows[: options.target_count]
    result.over_target = len(rows) - len(result.rows)
    if stopped is None:
        stopped = "exhausted" if pages.stats.exhausted else "max_candidates"
    result.stopped = stopped
    return result


def _convert(
    spans: Iterable[LLMObsSpan], reasoning: ReasoningMode
) -> tuple[list[TrainingRow], list[Exclusion], ConversionStats]:
    """The rows and exclusions of ``spans``; a repeated conversation is a duplicate."""
    conversion = rows_from_spans(spans, reasoning=reasoning)
    rows: list[TrainingRow] = []
    exclusions: list[Exclusion] = []
    by_hash: dict[str, str] = {}
    for converted in conversion:
        row = converted.row
        if row is None:
            if converted.exclusion is not None:
                exclusions.append(converted.exclusion)
            continue
        digest = row_content_hash(row)
        other = by_hash.setdefault(digest, row.id)
        if other != row.id:
            detail = f"same conversation as row {other!r}"
            exclusions.append(Exclusion.for_row(row, ExclusionCode.DUPLICATE_CONVERSATION, detail))
            continue
        rows.append(row)
    return rows, exclusions, conversion.stats


def _unreadable(span: UnreadableSpan) -> Exclusion:
    """The exclusion of an item the export skipped, identified by where it was read."""
    row_id = f"{span.trace_id}:{span.span_id}" if span.trace_id and span.span_id else None
    return Exclusion.for_source(
        ExclusionCode.UNREADABLE_SPAN,
        f"page {span.page}, item {span.index + 1}: not a readable LLM span",
        row_id=row_id,
        trace_id=span.trace_id,
        span_id=span.span_id,
    )


def retirements(
    collection: Collection,
    stored_rows: Sequence[TrainingRow],
    stored_exclusions: Iterable[dict[str, object]],
) -> list[Exclusion]:
    """The stored rows and exclusions ``collection`` shows were collapsed, as exclusions.

    A stored id whose span ``collection`` read but no longer yields was collapsed into
    another conversation of its trace, typically a longer one on a page an earlier, shorter
    pull did not reach. A stored row is retired only once a row that contains it (the same
    messages and more, compared as conversion compares them) is accepted, in ``collection``
    or the dataset, so a row is never retired in favour of one cut by ``target_count``. A
    stored exclusion of such an id is re-recorded as ``contained_conversation``.
    """
    stale = collection.read_ids - collection.produced_ids
    if not stale:
        return []
    keepers = [*collection.rows, *(row for row in stored_rows if row.id not in stale)]
    retired: list[Exclusion] = []
    for row in stored_rows:
        if row.id not in stale:
            continue
        container = _container(row, keepers)
        if container is not None:
            detail = f"a prefix of row {container.id!r}, read by a later pull"
            retired.append(Exclusion.for_row(row, ExclusionCode.CONTAINED_CONVERSATION, detail))
    code = ExclusionCode.CONTAINED_CONVERSATION
    for record in stored_exclusions:
        row_id = record.get("row_id")
        if row_id in stale and record.get("code") != code.value:
            trace_id, span_id = record.get("trace_id"), record.get("span_id")
            retired.append(
                Exclusion.for_source(
                    code,
                    f"contained in a longer conversation of trace {trace_id!r}",
                    row_id=str(row_id),
                    trace_id=trace_id if isinstance(trace_id, str) else None,
                    span_id=span_id if isinstance(span_id, str) else None,
                )
            )
    return retired


def _container(row: TrainingRow, candidates: Iterable[TrainingRow]) -> TrainingRow | None:
    """The first of ``candidates`` in ``row``'s trace whose messages start with ``row``'s."""
    messages = [_comparable(message) for message in row.messages]
    for other in candidates:
        if other.id == row.id or other.source.trace_id != row.source.trace_id:
            continue
        if [_comparable(m) for m in other.messages[: len(messages)]] == messages:
            return other
    return None


def _comparable(message: Message) -> tuple[object, ...]:
    """What conversation compares a message by: role, content, tool call names and args."""
    calls = tuple(
        (call.function.name, call.function.arguments) for call in message.tool_calls or ()
    )
    return (message.role, message.content, calls)


def pull(
    pup: Pup,
    options: PullOptions,
    dataset: DatasetDir,
    split: SplitParams,
    *,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
) -> tuple[Collection, AddResult]:
    """Collect into ``dataset``: freeze its settings, export through its ``raw/`` cache, add.

    The settings are frozen (or checked) before anything is fetched, so a pull with other
    settings fails fast with :class:`~pup_finetune.errors.StateError`, and an interrupted
    first pull leaves a dataset its re-run resumes; an :class:`ExternalError` part-way says
    so in its hint. The cache is locked while the export reads it. Stored rows the new
    conversion collapsed are retired (:func:`retirements`).
    """
    settings = options.dataset_settings(split)
    dataset.create_or_check(settings)
    raw = make_private_dir(dataset.raw_path)
    with DirectoryLock(raw):
        try:
            collection = collect(pup, options, cache_dir=raw, sleep=sleep, progress=progress)
        except ExternalError as exc:
            raise _with_resume_note(exc, raw) from exc
    retired = retirements(collection, dataset.rows(), dataset.excluded())
    added = dataset.add(collection.rows, collection.exclusions, settings=settings, retired=retired)
    return collection, added


def _with_resume_note(error: ExternalError, raw: Path) -> ExternalError:
    """``error`` with a hint that the pages read so far are cached and a re-run resumes."""
    note = f"pages read so far are cached in {raw}; re-run the same command to resume"
    hint = (error.hint or "").removesuffix(f" {CACHED_PAGES_NOTE}")
    return ExternalError(
        error.message, source=error.source, hint=f"{hint}; {note}" if hint else note
    )
