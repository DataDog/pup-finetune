# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training rows from Datadog: export LLM spans, check them, convert them, and stop in time.

:func:`collect` is what ``dataset pull`` runs into a dataset directory (:func:`pull`):

1. Export the spans matching the query a page at a time
   (:func:`~pup_finetune.datadog.spans.iter_span_pages`), reading at most
   ``max_candidates`` export items, readable or not.
2. Exclude the spans that fail a span-level check (:mod:`pup_finetune.data.checks`).
3. Convert the rest into rows per trace (:func:`~pup_finetune.data.trajectories.rows_from_spans`),
   and exclude a row repeating the conversation of an earlier one as
   ``duplicate_conversation``.
4. After each page, stop once ``target_count`` rows are accepted, before fetching the next
   page; the accepted rows are cut to ``target_count``. Into a dataset directory, rows its
   labels drop do not count (:func:`pull` passes them as ``dropped``): the target is the rows
   to keep, so a pull re-run after a review reads on to replace the rows it dropped.

With ``max_rows_per_trace``, a trace's rows past the cap are excluded as ``trace_row_cap``
before they count toward the target: its first rows in time order, or the sample's. With
``sample="random"`` reading never stops at the target: every candidate up to
``max_candidates`` is read, then rows are dealt round-robin over traces in a seeded order and
the first ``target_count`` are kept (see :func:`_ordered`), so no trace dominates.

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
:func:`repeat_retirements` and :func:`reconciled_repeats`).

A relative ``--from`` or ``--to`` (``now-7d``) is resolved to epoch milliseconds once, when a
collection or a dataset's first pull starts (:meth:`PullOptions.resolved`), and the export
sends the resolved window. A dataset freezes both: the resolved window in its ``filter`` and
the expressions as written in ``from_expr`` and ``to_expr``. A later pull with the same
expressions reuses the frozen window (:func:`resolve_window`), so it reads the same spans
however much later it runs.
"""

from __future__ import annotations

import hashlib
import heapq
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, get_args

from pup_finetune.data.checks import CheckOptions, check_span
from pup_finetune.data.convert import ReasoningMode, check_reasoning
from pup_finetune.data.dataset_dir import AddResult, DatasetDir, DatasetSettings, SourceKind
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.schema import TrainingRow, row_content_hash, without_reasoning
from pup_finetune.data.split import DEFAULT_SEED, SplitParams
from pup_finetune.data.trajectories import (
    ConversionStats,
    comparable_messages,
    comparable_tools,
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
from pup_finetune.next_steps import command
from pup_finetune.storage import DirectoryLock, atomic_write_json, make_private_dir, read_json

DEFAULT_TARGET_COUNT = 200
DEFAULT_MAX_CANDIDATES = 5000

SOURCE_KIND: SourceKind = "datadog"

StopReason = Literal["target_count", "max_candidates", "exhausted"]
"""Why reading stopped: enough rows, the span cap, or no matching span left."""

SampleMode = Literal["first", "random"]
"""Which rows count toward the target: the first read, or a seeded sample spread over traces."""
SAMPLE_MODES: tuple[SampleMode, ...] = get_args(SampleMode)

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
    max_rows_per_trace: int | None = None
    """Keep at most this many rows of any one trace; the rest are ``trace_row_cap``."""
    sample: SampleMode = "first"
    """``first`` stops at ``target_count``; ``random`` samples all candidates (:func:`_ordered`)."""
    window: tuple[TimeBound, TimeBound] | None = None
    """The ``from`` and ``to`` the export sends, a relative time resolved to epoch ms;
    ``None`` until :meth:`resolved` or :meth:`with_window` sets it."""

    def __post_init__(self) -> None:
        check_reasoning(self.reasoning)
        if self.target_count < 1:
            raise UsageError(f"--target-count must be at least 1, not {self.target_count}")
        if self.max_candidates < 1:
            raise UsageError(f"--max-candidates must be at least 1, not {self.max_candidates}")
        if self.max_rows_per_trace is not None and self.max_rows_per_trace < 1:
            raise UsageError(
                f"--max-rows-per-trace must be at least 1, not {self.max_rows_per_trace}"
            )
        if self.sample not in SAMPLE_MODES:
            raise UsageError(
                f"--sample must be one of {', '.join(SAMPLE_MODES)}, not {self.sample!r}"
            )

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
            # Only when set, so the settings of a dataset pulled before these options still match.
            **({"max_rows_per_trace": self.max_rows_per_trace} if self.max_rows_per_trace else {}),
            **({"sample": self.sample} if self.sample != "first" else {}),
        }

    def dataset_settings(self, split: SplitParams) -> DatasetSettings:
        return DatasetSettings(SOURCE_KIND, split, self.settings())


@dataclass
class Collection:
    """What :func:`collect` read and produced."""

    rows: list[TrainingRow] = field(default_factory=list)
    """Accepted rows, at most ``target_count``, in the order they count toward it
    (:func:`_ordered`)."""
    exclusions: list[Exclusion] = field(default_factory=list)
    """Span-level check failures first, then conversion, duplicate and ``trace_row_cap``
    exclusions, then ``unreadable_span`` ones."""
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
    counted: int = 0
    """The rows that count toward ``target_count``: those not in ``dropped``."""
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
    dropped: AbstractSet[str] = frozenset(),
    seed: int = DEFAULT_SEED,
    held: AbstractSet[str] = frozenset(),
) -> Collection:
    """Export, check and convert the spans ``options`` select; see the module docstring.

    ``cache_dir`` is the raw page cache :func:`~pup_finetune.datadog.spans.iter_span_pages`
    resumes from; without one nothing is written. Options without a window are resolved at
    ``clock`` first. Rows whose id is in ``dropped`` do not count toward ``target_count``.
    ``seed`` and ``held`` (the ids of rows already stored) order a random sample (:func:`_ordered`).
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
        if options.sample == "random":
            if progress is not None:
                progress(f"page {number}: {len(seen)} spans read")
            continue
        if added:
            converted = _convert(passing, options)
        accepted = sum(1 for row in converted.rows if row.id not in dropped)
        if progress is not None:
            progress(f"page {number}: {len(seen)} spans read, {accepted} rows accepted")
        if accepted >= options.target_count:
            stopped = "target_count"
            break
    if options.sample == "random":
        converted = _convert(passing, options, seed, held)
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
    target = options.target_count
    if options.sample == "random":
        # Stored rows this read stops short of keep their places in the sample.
        target -= len(held - dropped - result.read_ids)
    result.rows = _up_to_target(rows, max(0, target), dropped)
    result.over_target = len(rows) - len(result.rows)
    result.counted = sum(1 for row in result.rows if row.id not in dropped)
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


def _convert(
    spans: Iterable[LLMObsSpan],
    options: PullOptions,
    seed: int = DEFAULT_SEED,
    held: AbstractSet[str] = frozenset(),
) -> _Converted:
    """The rows and exclusions of ``spans``, in :func:`_ordered`'s order: a row past the
    per-trace cap is excluded, then one repeating the conversation of a row kept before it."""
    conversion = rows_from_spans(spans, reasoning=options.reasoning)
    result = _Converted()
    rows: list[TrainingRow] = []
    for converted in conversion:
        if converted.row is not None:
            rows.append(converted.row)
        elif converted.exclusion is not None:
            result.exclusions.append(converted.exclusion)
    cap = options.max_rows_per_trace
    taken: Counter[str | None] = Counter()
    by_hash: dict[str, str] = {}

    def keep(row: TrainingRow) -> bool:
        trace = row.source.trace_id
        if cap is not None and taken[trace] >= cap:
            # Not hashed, so a copy of its conversation in a trace with room is kept.
            detail = f"past the first --max-rows-per-trace {cap} rows of trace {trace!r}"
            if options.sample == "random":
                detail = f"past the --max-rows-per-trace {cap} rows kept of trace {trace!r}"
            result.exclusions.append(Exclusion.for_row(row, ExclusionCode.TRACE_ROW_CAP, detail))
            return False
        other = by_hash.setdefault(row_content_hash(row), row.id)
        if other != row.id:
            detail = f"same conversation as row {other!r}"
            exclusion = Exclusion.for_row(row, ExclusionCode.DUPLICATE_CONVERSATION, detail)
            result.exclusions.append(exclusion)
            result.duplicates.append((row, other))
            return False
        taken[trace] += 1
        result.rows.append(row)
        return True

    _ordered(rows, options, seed, held, keep)
    result.stats = conversion.stats
    return result


def _ordered(
    rows: Sequence[TrainingRow],
    options: PullOptions,
    seed: int,
    held: AbstractSet[str],
    keep: Callable[[TrainingRow], bool],
) -> None:
    """Offer ``rows`` to ``keep`` in the order they count toward ``target_count``.

    With ``sample="first"`` the order is conversion's (time order). With ``sample="random"``
    rows are dealt in turns: each trace ranks its rows (``held`` ones first, then by
    ``sha256(f"sample:{seed}:{row id}")``), and the next turn goes to the trace with the
    fewest rows kept, ties broken by ``sha256(f"sample:{seed}:{trace id}")``, held rows
    leading. On its turn a trace offers rows until ``keep`` takes one, so a duplicate does
    not cost it its turn. No rank depends on how far a pull read, so a rerun keeps the rows
    it stored and deals on from them.
    """
    if options.sample == "first":
        for row in rows:
            keep(row)
        return
    by_trace: dict[str | None, list[TrainingRow]] = {}
    for row in rows:
        by_trace.setdefault(row.source.trace_id, []).append(row)
    # Each queue is reversed, so pop() takes a trace's next row.
    queues: list[list[TrainingRow]] = []
    turns: list[tuple[bool, int, str, int]] = []
    for trace, trace_rows in by_trace.items():
        queue = sorted(
            trace_rows, key=lambda row: (row.id not in held, _rank(seed, row.id)), reverse=True
        )
        turns.append((queue[-1].id not in held, 0, _rank(seed, str(trace)), len(queues)))
        queues.append(queue)
    heapq.heapify(turns)
    while turns:
        _, kept, rank, index = heapq.heappop(turns)
        queue = queues[index]
        kept += keep(queue.pop())
        if queue:
            heapq.heappush(turns, (queue[-1].id not in held, kept, rank, index))


def _rank(seed: int, key: str) -> str:
    # Salted, so the sample does not follow the split, which hashes f"{seed}:{trace id}".
    return hashlib.sha256(f"sample:{seed}:{key}".encode()).hexdigest()


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
            into = container.id
            detail = f"a prefix of row {container.id!r}, read by a later pull"
            if (other := repeats.get(container.id)) is not None:
                into = other
                detail = (
                    f"a prefix of {container.id!r}, the same conversation as row {other!r}, "
                    "read by a later pull"
                )
            retired.append(
                Exclusion.for_row(row, ExclusionCode.CONTAINED_CONVERSATION, detail, into=into)
            )
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

    The two are compared as the dataset will hold them, each stored row with the reasoning
    this conversion fills in (as :func:`reasoning_updates` does), since stored reasoning is
    never replaced: a re-run that reads less reasoning can yield two rows that repeat each
    other while the stored ones still differ, and neither is retired then.
    """
    stored = {row.id: row for row in stored_rows}
    current = {row.id: row for row in collection.rows}
    retired: list[Exclusion] = []
    for row, other in collection.duplicates:
        known, repeated = stored.get(row.id), stored.get(other)
        if known is None or repeated is None:
            continue
        mine = _fill_reasoning(known, row) or known
        theirs = repeated
        if (yielded := current.get(other)) is not None:
            theirs = _fill_reasoning(repeated, yielded) or repeated
        if row_content_hash(mine) == row_content_hash(theirs):
            detail = f"same conversation as row {other!r}, read by a later pull"
            code = ExclusionCode.DUPLICATE_CONVERSATION
            retired.append(Exclusion.for_row(known, code, detail, into=other))
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


def trace_capped(
    rows: Sequence[TrainingRow],
    stored_rows: Sequence[TrainingRow],
    cap: int | None,
    retired: Iterable[Exclusion] = (),
) -> tuple[list[TrainingRow], list[Exclusion]]:
    """``rows`` minus the new ones past ``cap`` rows of their trace in the dataset, and an
    exclusion for each one left out.

    The stored rows that are not being ``retired`` take their trace's places first, since a
    re-run that reads further can rank a trace's rows otherwise: a stored row collapsed into
    a longer one past the cap is not retired, for one. Rows already stored are left to the
    dataset.
    """
    if cap is None:
        return list(rows), []
    retired_ids = {exclusion.row_id for exclusion in retired}
    stored_ids = {row.id for row in stored_rows}
    taken = Counter(row.source.trace_id for row in stored_rows if row.id not in retired_ids)
    kept: list[TrainingRow] = []
    excluded: list[Exclusion] = []
    for row in rows:
        trace = row.source.trace_id
        if row.id in stored_ids:
            kept.append(row)
        elif taken[trace] < cap:
            taken[trace] += 1
            kept.append(row)
        else:
            detail = f"trace {trace!r} already has --max-rows-per-trace {cap} rows in the dataset"
            excluded.append(Exclusion.for_row(row, ExclusionCode.TRACE_ROW_CAP, detail))
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


def reconciled_repeats(
    rows: Sequence[TrainingRow],
    updated: Sequence[TrainingRow],
    stored_rows: Sequence[TrainingRow],
    retired: Iterable[Exclusion] = (),
) -> tuple[list[TrainingRow], list[TrainingRow], list[Exclusion]]:
    """``rows`` and ``updated`` without the stored rows they leave repeating another, and
    those rows' retirements.

    :func:`repeat_retirements` only sees the stored rows conversion flagged as duplicates,
    but stored reasoning is never replaced, so two stored rows can become one conversation
    without conversion seeing a repeat: stored A with reasoning and B without, read again
    with other reasoning for A (which A does not take) and A's old reasoning for B (which B
    does). Every stored row that is not being ``retired`` is compared as the dataset will
    hold it, with its update from :func:`reasoning_updates` applied; of each group holding
    one conversation, the first in dataset order is kept and the rest are retired as
    ``duplicate_conversation``, and dropped from ``rows`` and ``updated`` so the dataset
    neither keeps nor updates them.
    """
    retired_ids = {exclusion.row_id for exclusion in retired}
    held = {row.id: row for row in updated}
    first: dict[str, str] = {}
    repeats: list[Exclusion] = []
    for stored in stored_rows:
        if stored.id in retired_ids:
            continue
        other = first.setdefault(row_content_hash(held.get(stored.id, stored)), stored.id)
        if other != stored.id:
            detail = f"same conversation as row {other!r}, read by a later pull"
            code = ExclusionCode.DUPLICATE_CONVERSATION
            repeats.append(Exclusion.for_row(stored, code, detail, into=other))
    if not repeats:
        return list(rows), list(updated), []
    gone = {exclusion.row_id for exclusion in repeats}
    return (
        [row for row in rows if row.id not in gone],
        [row for row in updated if row.id not in gone],
        repeats,
    )


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

    A row contains another when it has the same tools and its messages start with the
    other's, by :func:`~pup_finetune.data.trajectories.comparable_tools` and
    :func:`~pup_finetune.data.trajectories.comparable_messages`: canonical tool call
    arguments, and each tool message by the position of the call it answers, so retirement
    and conversion agree on which conversations are one.
    """

    def __init__(self, rows: Iterable[TrainingRow]) -> None:
        self._by_trace: dict[str | None, list[tuple[TrainingRow, str, list[str]]]] = {}
        for row in rows:
            tools, parts = _comparable(row)
            self._by_trace.setdefault(row.source.trace_id, []).append((row, tools, parts))

    def find(self, row: TrainingRow) -> TrainingRow | None:
        """The first row in ``row``'s trace, other than ``row``, with its tools and whose
        messages start with its."""
        tools, parts = _comparable(row)
        for other, other_tools, other_parts in self._by_trace.get(row.source.trace_id, ()):
            # A part depends only on its message and the ones before it, so the parts of
            # a prefix are a prefix of the parts.
            if other.id != row.id and other_tools == tools and other_parts[: len(parts)] == parts:
                return other
        return None


def _comparable(row: TrainingRow) -> tuple[str, list[str]]:
    tools = comparable_tools(tool.model_dump() for tool in row.tools)
    return tools, comparable_messages(message.model_dump() for message in row.messages)


def resolve_window(options: PullOptions, dataset: DatasetDir, clock: Clock) -> PullOptions:
    """``options`` with the window a pull into ``dataset`` exports.

    When ``dataset`` already froze the same ``from_expr`` and ``to_expr``, that is its frozen
    window, however far ``clock`` has moved since; otherwise (a first pull, or other
    expressions, which the settings check then rejects) the time range resolved at ``clock``.
    Options that already have a window are returned unchanged.
    """
    if options.window is not None:
        return options
    window = frozen_window(options, dataset)
    if window is not None:
        return options.with_window(*window)
    return options.resolved(epoch_ms(clock))


def frozen_window(options: PullOptions, dataset: DatasetDir) -> tuple[TimeBound, TimeBound] | None:
    """The window ``dataset`` froze for the ``from_expr`` and ``to_expr`` of ``options``, or
    ``None`` when it froze none (a first pull) or froze other expressions."""
    source = dataset.settings().get("source") if dataset.is_dataset() else None
    if not isinstance(source, dict):
        return None
    frozen = source.get("filter")
    same = (source.get("from_expr"), source.get("to_expr")) == (
        options.query.from_,
        options.query.to,
    )
    if same and isinstance(frozen, dict):
        return frozen.get("from"), frozen.get("to")
    return None


def rerun_command(directory: Path, options: PullOptions) -> str:
    """The ``dataset pull`` that resumes or continues a pull into ``directory`` with
    ``options``: no source flag, as the directory's frozen settings fill them in, only the
    limits that differ from the defaults (which are not frozen)."""
    limits: list[object] = []
    if options.target_count != DEFAULT_TARGET_COUNT:
        limits += ["--target-count", options.target_count]
    if options.max_candidates != DEFAULT_MAX_CANDIDATES:
        limits += ["--max-candidates", options.max_candidates]
    return command("dataset", "pull", directory, *limits)


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
    (:func:`reasoning_updates`), stored rows those updates make repeat another are retired
    instead (:func:`reconciled_repeats`), new rows a stored row contains are excluded instead of
    added (:func:`contained`), and so are new rows past the per-trace cap (:func:`trace_capped`);
    the returned collection is what the export read and converted, before any of these. A
    retired row's drop label carries to the row that holds its conversation
    (:meth:`~pup_finetune.data.dataset_dir.DatasetDir.add`).

    The cache is locked from the export through the add, so a concurrent pull into the same
    dataset fails with :class:`~pup_finetune.errors.LockBusy` instead of changing the stored
    rows this one reconciles against, or the limits a rerun continues toward
    (:func:`write_limits`).
    """
    options = resolve_window(options, dataset, clock)
    settings = options.dataset_settings(split)
    dataset.create_or_check(settings)
    raw = make_private_dir(dataset.raw_path)
    # The raw/ lock is held through the add, so no other pull changes the stored rows between
    # the read below and the add. DatasetDir.add locks the dataset directory itself, through
    # another lock file, so holding this one does not block it.
    dropped = {row_id for row_id, label in dataset.labels().items() if not label.keep}
    with DirectoryLock(raw):
        write_limits(dataset, options)
        stored = dataset.rows()
        try:
            collection = collect(
                pup,
                options,
                cache_dir=raw,
                sleep=sleep,
                progress=progress,
                dropped=dropped,
                seed=split.seed,
                held={row.id for row in stored},
            )
        except ExternalError as exc:
            raise _with_resume_note(exc, raw, rerun_command(dataset.path, options)) from exc
        stored_exclusions = dataset.excluded()
        retired = [
            *retirements(collection, stored, stored_exclusions),
            *repeat_retirements(collection, stored),
        ]
        rows, updated = reasoning_updates(collection.rows, stored)
        rows, updated, repeats = reconciled_repeats(rows, updated, stored, retired)
        retired.extend(repeats)
        rows, exclusions = contained(rows, stored, retired, stored_exclusions)
        rows, capped = trace_capped(rows, stored, options.max_rows_per_trace, retired)
        added = dataset.add(
            rows,
            [*collection.exclusions, *exclusions, *capped],
            settings=settings,
            retired=retired,
            updated=updated,
        )
        exhausted = collection.stopped == "exhausted" and collection.over_target == 0
        write_limits(dataset, options, exhausted=exhausted)
    return collection, added


LIMITS_FILE = "pull_limits.json"
"""The limits of a dataset's last pull (``target_count``, ``max_candidates``): not frozen, they
are what a rerun without them continues toward."""


def read_limits(dataset: DatasetDir) -> tuple[int, int, bool] | None:
    """``(target_count, max_candidates, exhausted)`` of ``dataset``'s last pull (``exhausted``:
    it read every matching span and the target cut no row), or ``None``."""
    try:
        data = read_json(dataset.path / LIMITS_FILE)
    except (FileNotFoundError, ValueError):
        return None
    target, candidates = (
        (data.get("target_count"), data.get("max_candidates"))
        if isinstance(data, dict)
        else (None, None)
    )
    if type(target) is int and type(candidates) is int and target > 0 and candidates > 0:
        return target, candidates, bool(isinstance(data, dict) and data.get("exhausted"))
    return None


def write_limits(dataset: DatasetDir, options: PullOptions, *, exhausted: bool = False) -> None:
    """Record ``options``' limits as ``dataset``'s last pull's, and whether it read every
    matching span with no row cut by the target. :func:`pull` records them, under the
    ``raw/`` lock, when it starts, so the rerun that resumes a pull which failed part-way
    continues toward the limits it was given, and again once it ends."""
    atomic_write_json(
        dataset.path / LIMITS_FILE,
        {
            "target_count": options.target_count,
            "max_candidates": options.max_candidates,
            "exhausted": exhausted,
        },
    )


def _up_to_target(
    rows: Sequence[TrainingRow], target: int, dropped: AbstractSet[str]
) -> list[TrainingRow]:
    """The first ``rows`` that hold ``target`` rows not in ``dropped``, with the dropped rows
    among them (stored already, they are unchanged)."""
    kept: list[TrainingRow] = []
    counted = 0
    for row in rows:
        if counted >= target:
            break
        kept.append(row)
        counted += row.id not in dropped
    return kept


def _with_resume_note(error: ExternalError, raw: Path, rerun: str) -> ExternalError:
    """``error`` with a hint that the pages read so far are cached and ``rerun`` resumes."""
    note = f"pages read so far are cached in {raw}; `{rerun}` resumes (the frozen settings fill in the flags)"
    hint = (error.hint or "").removesuffix(f" {CACHED_PAGES_NOTE}")
    return ExternalError(
        error.message, source=error.source, hint=f"{hint}; {note}" if hint else note
    )
