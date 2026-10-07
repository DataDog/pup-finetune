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
   ``max_candidates`` export items, readable or not.
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
of every span it read is what the dataset holds. A re-run with lower limits, which stops before
that longer conversation, does not add the shorter row back, also when the longer
conversation repeats another trace's row and so was never stored (see :func:`contained`). With
``reasoning="preserve"`` a stored row a re-run reads more reasoning for takes that reasoning,
and one a re-run reads less for keeps its own (see :func:`reasoning_updates`); a stored row
that reasoning makes repeat another stored row is retired as a duplicate (see
:func:`repeat_retirements`).

A relative ``--from`` or ``--to`` (``now-7d``) is resolved to epoch milliseconds once, when a
collection or a dataset's first pull starts (:meth:`PullOptions.resolved`), and the export
sends the resolved window. A dataset freezes both: the resolved window in its ``filter`` and
the expressions as written in ``from_expr`` and ``to_expr``. A later pull with the same
expressions reuses the frozen window (:func:`resolve_window`), so it reads the same spans
however much later it runs.
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
from pup_finetune.data.schema import TrainingRow, row_content_hash, without_reasoning
from pup_finetune.data.split import SplitParams
from pup_finetune.data.trajectories import (
    ConversionStats,
    comparable_messages,
    rows_from_spans,
)
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import (
    CACHED_PAGES_NOTE,
    Clock,
    ExportStats,
    LLMObsSpan,
    SpanQuery,
    TimeBound,
    UnreadableSpan,
    epoch_ms,
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
    """The spans to export, with ``from_`` and ``to`` as written (``now-7d``)."""
    reasoning: ReasoningMode = "omit"
    include_errors: bool = False
    """Keep errored LLM calls instead of excluding them as ``span_error``."""
    target_count: int = DEFAULT_TARGET_COUNT
    """Stop reading once this many rows are accepted."""
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    """Read at most this many export items, readable or not (an unreadable one counts too)."""
    window: tuple[TimeBound, TimeBound] | None = None
    """The ``from`` and ``to`` the export sends, a relative time resolved to epoch ms;
    ``None`` until :meth:`resolved` or :meth:`with_window` sets it."""

    def __post_init__(self) -> None:
        check_reasoning(self.reasoning)
        if self.target_count < 1:
            raise UsageError(f"--target-count must be at least 1, not {self.target_count}")
        if self.max_candidates < 1:
            raise UsageError(f"--max-candidates must be at least 1, not {self.max_candidates}")

    def resolved(self, now_ms: int) -> PullOptions:
        """These options with the time range of ``query`` resolved at ``now_ms``."""
        query = self.query.resolved(now_ms)
        return self.with_window(query.from_, query.to)

    def with_window(self, from_: TimeBound, to: TimeBound) -> PullOptions:
        """These options exporting the window ``from_`` to ``to``, such as a frozen one."""
        return replace(self, window=(from_, to))

    @property
    def export_query(self) -> SpanQuery:
        """The query the export sends: ``query`` over :attr:`window`.

        Raises :class:`ValueError` when ``query`` has a relative time and no window is set,
        rather than letting Datadog resolve it again for every page.
        """
        if self.window is None:
            if self.query.is_relative:
                raise ValueError("resolve the relative time range of the query first")
            return self.query
        from_, to = self.window
        return replace(self.query, from_=from_, to=to)

    def settings(self) -> dict[str, Any]:
        """What selects and shapes the rows, to freeze in the dataset's settings.

        The ``filter`` holds the window the export sends; ``from_expr`` and ``to_expr`` the
        time range as written, which a later pull is matched by (:func:`resolve_window`).
        """
        return {
            **self.export_query.settings(),
            "from_expr": self.query.from_,
            "to_expr": self.query.to,
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
    duplicates: list[tuple[TrainingRow, str]] = field(default_factory=list)
    """Each row excluded as ``duplicate_conversation``, with the id of the row it repeats
    (which may be over the target)."""

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
    clock: Clock = time.time,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
) -> Collection:
    """Export, check and convert the spans ``options`` select; see the module docstring.

    ``cache_dir`` is the raw page cache :func:`~pup_finetune.datadog.spans.iter_span_pages`
    resumes from; without one nothing is written. Options without a window are resolved at
    ``clock`` first.
    """
    if options.window is None:
        options = options.resolved(epoch_ms(clock))
    query = replace(options.export_query, max_spans=options.max_candidates)
    pages = iter_span_pages(pup, query, cache_dir=cache_dir, sleep=sleep)
    check_options = CheckOptions(include_errors=options.include_errors)
    result = Collection(export=pages.stats)
    passing: list[LLMObsSpan] = []
    seen: set[str] = set()
    traces: set[str] = set()
    converted = _Converted()
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
            converted = _convert(passing, options.reasoning)
        accepted = len(converted.rows)
        if progress is not None:
            progress(f"page {number}: {len(seen)} spans read, {accepted} rows accepted")
        if accepted >= options.target_count:
            stopped = "target_count"
            break
    rows = converted.rows
    result.conversion = converted.stats
    result.duplicates = converted.duplicates
    result.spans = len(seen)
    result.traces = len(traces)
    result.failed_checks = len(result.exclusions)
    result.exclusions.extend(converted.exclusions)
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


@dataclass
class _Converted:
    rows: list[TrainingRow] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    duplicates: list[tuple[TrainingRow, str]] = field(default_factory=list)
    stats: ConversionStats = field(default_factory=ConversionStats)


def _convert(spans: Iterable[LLMObsSpan], reasoning: ReasoningMode) -> _Converted:
    """The rows and exclusions of ``spans``; a repeated conversation is a duplicate."""
    conversion = rows_from_spans(spans, reasoning=reasoning)
    result = _Converted()
    by_hash: dict[str, str] = {}
    for converted in conversion:
        row = converted.row
        if row is None:
            if converted.exclusion is not None:
                result.exclusions.append(converted.exclusion)
            continue
        digest = row_content_hash(row)
        other = by_hash.setdefault(digest, row.id)
        if other != row.id:
            detail = f"same conversation as row {other!r}"
            exclusion = Exclusion.for_row(row, ExclusionCode.DUPLICATE_CONVERSATION, detail)
            result.exclusions.append(exclusion)
            result.duplicates.append((row, other))
            continue
        result.rows.append(row)
    result.stats = conversion.stats
    return result


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
    or the dataset, so a row is never retired in favour of one cut by ``target_count``. The
    longer conversation can also be a duplicate of another trace's row: it then contains the
    stored row when the row it repeats is accepted, as that row trains on the same turns. A
    stored exclusion of such an id is re-recorded as ``contained_conversation``.
    """
    stale = collection.read_ids - collection.produced_ids
    if not stale:
        return []
    accepted = [*collection.rows, *(row for row in stored_rows if row.id not in stale)]
    accepted_ids = {row.id for row in accepted}
    repeats = {row.id: other for row, other in collection.duplicates if other in accepted_ids}
    duplicates = [row for row, _ in collection.duplicates if row.id in repeats]
    keepers = _Containers([*accepted, *duplicates])
    retired: list[Exclusion] = []
    for row in stored_rows:
        if row.id not in stale:
            continue
        container = keepers.find(row)
        if container is not None:
            detail = f"a prefix of row {container.id!r}, read by a later pull"
            if (other := repeats.get(container.id)) is not None:
                detail = (
                    f"a prefix of {container.id!r}, the same conversation as row {other!r}, "
                    "read by a later pull"
                )
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


def repeat_retirements(
    collection: Collection, stored_rows: Sequence[TrainingRow]
) -> list[Exclusion]:
    """The stored rows ``collection`` shows repeat another stored row, as exclusions.

    With ``reasoning="preserve"`` two traces can be stored with the same messages and
    different reasoning, so different content hashes, until a re-run reads more reasoning for
    one of them (:func:`reasoning_updates`) and the two become one conversation. Conversion
    then excludes the later one as ``duplicate_conversation``, but its id is still produced,
    so :func:`retirements` keeps it, and the dataset would hold the same conversation twice. A
    stored row conversion now finds repeating another stored row is retired as a duplicate
    instead. One repeating a row new to the dataset is left to the dataset, which keeps the
    stored row and excludes the new one when the two hash the same.
    """
    stored = {row.id: row for row in stored_rows}
    retired: list[Exclusion] = []
    for row, other in collection.duplicates:
        known = stored.get(row.id)
        if known is not None and other in stored:
            detail = f"same conversation as row {other!r}, read by a later pull"
            retired.append(Exclusion.for_row(known, ExclusionCode.DUPLICATE_CONVERSATION, detail))
    return retired


def contained(
    rows: Sequence[TrainingRow],
    stored_rows: Sequence[TrainingRow],
    retired: Iterable[Exclusion] = (),
    stored_exclusions: Iterable[dict[str, object]] = (),
) -> tuple[list[TrainingRow], list[Exclusion]]:
    """``rows`` minus those a stored row contains, and an exclusion for each one left out.

    The limits are not frozen, so a pull re-run with a lower ``target_count`` than the last
    one can stop before the page that showed a conversation continues, and produce again the
    shorter row an earlier pull already replaced by its longer version (:func:`retirements`).
    Such a row, new to the dataset, is excluded as ``contained_conversation`` instead of being
    added beside the longer one when a stored row of its trace that is not being ``retired``
    contains it, or when ``stored_exclusions`` already exclude its id as
    ``contained_conversation`` (re-recorded with the stored detail). The second covers a row
    retired because its trace's longer conversation repeats another trace's row: that longer
    conversation is a duplicate and never stored, so no stored row of the trace contains the
    row. Honouring the stored exclusion is safe because a container is only ever retired into
    a longer row that still contains it. Rows already stored are left to the dataset.
    """
    retired_ids = {exclusion.row_id for exclusion in retired}
    stored_ids = {row.id for row in stored_rows}
    containers = _Containers(row for row in stored_rows if row.id not in retired_ids)
    code = ExclusionCode.CONTAINED_CONVERSATION
    excluded_as_contained = {
        str(record["row_id"]): str(record.get("detail") or "")
        for record in stored_exclusions
        if record.get("code") == code.value and isinstance(record.get("row_id"), str)
    }
    kept: list[TrainingRow] = []
    excluded: list[Exclusion] = []
    for row in rows:
        if row.id in stored_ids:
            kept.append(row)
            continue
        container = containers.find(row)
        if container is not None:
            detail = f"a prefix of row {container.id!r}, read by an earlier pull"
            excluded.append(Exclusion.for_row(row, code, detail))
        elif row.id in excluded_as_contained:
            detail = excluded_as_contained[row.id] or "contained in a longer conversation"
            excluded.append(Exclusion.for_row(row, code, detail))
        else:
            kept.append(row)
    return kept, excluded


def reasoning_updates(
    rows: Sequence[TrainingRow], stored_rows: Sequence[TrainingRow]
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """``rows`` with stored ones as stored, and the stored rows they add reasoning to.

    With ``reasoning="preserve"`` a kept row takes the reasoning of the conversations
    collapsed into it (:mod:`pup_finetune.data.trajectories`) and keeps its id, so a re-run
    that reads more or less of its trace can yield a stored id with more or less reasoning.
    Such a row, equal to the stored one but for reasoning, is reconciled with it: each stored
    assistant message without reasoning takes the row's, and stored reasoning is never
    replaced. When nothing is filled in the stored row is returned among ``rows``, unchanged;
    otherwise the filled-in row is returned among the updates, for
    :meth:`~pup_finetune.data.dataset_dir.DatasetDir.add` to replace the stored one with. Any
    other difference is left to the dataset, which rejects it.
    """
    stored = {row.id: row for row in stored_rows}
    kept: list[TrainingRow] = []
    updated: list[TrainingRow] = []
    for row in rows:
        known = stored.get(row.id)
        merged = None if known is None else _fill_reasoning(known, row)
        if merged is None:
            kept.append(row)
        elif merged == known:
            kept.append(known)
        else:
            updated.append(merged)
    return kept, updated


def _fill_reasoning(stored: TrainingRow, row: TrainingRow) -> TrainingRow | None:
    """``stored`` with the reasoning of ``row`` where it has none, or ``None`` when the two
    differ in more than reasoning."""
    if without_reasoning(stored) != without_reasoning(row):
        return None
    messages = [
        mine.model_copy(update={"reasoning_content": theirs.reasoning_content})
        if mine.reasoning_content is None and theirs.reasoning_content is not None
        else mine
        for mine, theirs in zip(stored.messages, row.messages, strict=True)
    ]
    return stored.model_copy(update={"messages": messages})


class _Containers:
    """Rows that may contain another one, by trace, compared as conversion compares them.

    A row contains another when its messages start with the other's, by
    :func:`~pup_finetune.data.trajectories.comparable_messages`: canonical tool call
    arguments, and each tool message by the position of the call it answers, so retirement
    and conversion agree on which conversations are one.
    """

    def __init__(self, rows: Iterable[TrainingRow]) -> None:
        self._by_trace: dict[str | None, list[tuple[TrainingRow, list[str]]]] = {}
        for row in rows:
            self._by_trace.setdefault(row.source.trace_id, []).append((row, _comparable(row)))

    def find(self, row: TrainingRow) -> TrainingRow | None:
        """The first row in ``row``'s trace, other than ``row``, whose messages start with its."""
        parts = _comparable(row)
        for other, other_parts in self._by_trace.get(row.source.trace_id, ()):
            # A part depends only on its message and the ones before it, so the parts of
            # a prefix are a prefix of the parts.
            if other.id != row.id and other_parts[: len(parts)] == parts:
                return other
        return None


def _comparable(row: TrainingRow) -> list[str]:
    return comparable_messages(message.model_dump() for message in row.messages)


def resolve_window(options: PullOptions, dataset: DatasetDir, clock: Clock) -> PullOptions:
    """``options`` with the window a pull into ``dataset`` exports.

    When ``dataset`` already froze the same ``from_expr`` and ``to_expr``, that is its frozen
    window, however far ``clock`` has moved since; otherwise (a first pull, or other
    expressions, which the settings check then rejects) the time range resolved at ``clock``.
    Options that already have a window are returned unchanged.
    """
    if options.window is not None:
        return options
    source = dataset.settings().get("source") if dataset.is_dataset() else None
    if isinstance(source, dict):
        frozen = source.get("filter")
        same = (source.get("from_expr"), source.get("to_expr")) == (
            options.query.from_,
            options.query.to,
        )
        if same and isinstance(frozen, dict):
            return options.with_window(frozen.get("from"), frozen.get("to"))
    return options.resolved(epoch_ms(clock))


def pull(
    pup: Pup,
    options: PullOptions,
    dataset: DatasetDir,
    split: SplitParams,
    *,
    clock: Clock = time.time,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
) -> tuple[Collection, AddResult]:
    """Collect into ``dataset``: freeze its settings, export through its ``raw/`` cache, add.

    The window is resolved first (:func:`resolve_window`), so a re-run reads the window the
    first pull froze. The settings are frozen (or checked) before anything is fetched, so a
    pull with other settings fails fast with :class:`~pup_finetune.errors.StateError`, and
    an interrupted first pull leaves a dataset its re-run resumes; an :class:`ExternalError`
    part-way says so in its hint. Stored rows the new conversion collapsed are retired
    (:func:`retirements`), as are stored rows it found repeating another stored row
    (:func:`repeat_retirements`), stored rows it read more reasoning for are updated
    (:func:`reasoning_updates`), and new rows a stored row contains are excluded instead of
    added (:func:`contained`); the returned collection is what the export read and
    converted, before any of these.

    The cache is locked from the export through the add, so a concurrent pull into the same
    dataset fails with :class:`~pup_finetune.errors.LockBusy` instead of changing the stored
    rows this one reconciles against.
    """
    options = resolve_window(options, dataset, clock)
    settings = options.dataset_settings(split)
    dataset.create_or_check(settings)
    raw = make_private_dir(dataset.raw_path)
    # The raw/ lock is held through the add, so no other pull changes the stored rows between
    # the read below and the add. DatasetDir.add locks the dataset directory itself, through
    # another lock file, so holding this one does not block it.
    with DirectoryLock(raw):
        try:
            collection = collect(pup, options, cache_dir=raw, sleep=sleep, progress=progress)
        except ExternalError as exc:
            raise _with_resume_note(exc, raw) from exc
        stored = dataset.rows()
        stored_exclusions = dataset.excluded()
        retired = [
            *retirements(collection, stored, stored_exclusions),
            *repeat_retirements(collection, stored),
        ]
        rows, updated = reasoning_updates(collection.rows, stored)
        rows, exclusions = contained(rows, stored, retired, stored_exclusions)
        added = dataset.add(
            rows,
            [*collection.exclusions, *exclusions],
            settings=settings,
            retired=retired,
            updated=updated,
        )
    return collection, added


def _with_resume_note(error: ExternalError, raw: Path) -> ExternalError:
    """``error`` with a hint that the pages read so far are cached and a re-run resumes."""
    note = f"pages read so far are cached in {raw}; re-run the same command to resume"
    hint = (error.hint or "").removesuffix(f" {CACHED_PAGES_NOTE}")
    return ExternalError(
        error.message, source=error.source, hint=f"{hint}; {note}" if hint else note
    )
