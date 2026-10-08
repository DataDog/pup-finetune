# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""LLM span export through the Agent Observability Export API, over pup's OAuth.

:func:`iter_spans` pages through ``POST v2/llm-obs/v1/spans/events/search`` with
:meth:`Pup.api <pup_finetune.datadog.pup.Pup.api>`, following the cursor in
``meta.page.after``, and yields each span as an :class:`LLMObsSpan`. The models are tolerant:
they keep the fields conversion uses, ignore the rest, and a span that still does not fit is
skipped and counted in :class:`ExportStats` instead of failing the export.

With a ``cache_dir``, every raw page is written there before its spans are yielded, together
with ``cursor.json`` (the cursor to continue from), so an interrupted export resumes without
fetching the cached pages again. A cache belongs to the query and to the Datadog site and org
pup is logged in to (``pup auth status``, and the org id from ``v2/current_user`` when pup's
default session does not name its org), so it is never reused for another account.
:func:`iter_span_pages` yields the same spans a page at a time, so a caller can stop between
pages without fetching the next one.

A query's ``from_`` and ``to`` are sent as they are, epoch milliseconds as a decimal string:
the Export API accepts a time bound only as a string and rejects a JSON number. A relative
time such as ``now-7d`` is resolved by Datadog when each page is fetched;
:meth:`SpanQuery.resolved` turns it into epoch milliseconds first, so every page of an export
reads the same window. pup's own forms (``7d``, ``5min``, ``'2 hours'``) are read as relative
times too (:func:`canonical_relative_time`) and never sent as they are.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import ExternalError, SettingsConflict, StateError, UsageError
from pup_finetune.storage import (
    atomic_write_json,
    finite_json,
    make_private_dir,
    read_json,
    settings_hash,
)

EXPORT_PATH = "v2/llm-obs/v1/spans/events/search"
"""The Export API search route, relative to the Datadog site."""
CONTENT_TYPE = "application/vnd.api+json"
DEFAULT_PAGE_LIMIT = 1000

TIMEOUT_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 5.0, 10.0)
"""The waits before each retry of a page the API reported as timed out; one retry per entry."""

CURSOR_FILE_NAME = "cursor.json"

CACHED_PAGES_NOTE = "(cached pages are kept)"
"""Appended to the hint of a timed-out export that has a cache to resume from."""


TimeBound = str | int | None
"""A ``from`` or ``to`` bound: ISO 8601 or a relative time (text), epoch ms, or unbounded."""

Clock = Callable[[], float]
"""The current time in seconds since the epoch, like :func:`time.time`."""

RELATIVE_TIME_UNITS_MS: Mapping[str, int] = {
    "s": 1000,
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
    "w": 7 * 24 * 60 * 60 * 1000,
}
"""The units of a relative time, ``now-<n><unit>``, in milliseconds."""

RELATIVE_TIME_UNIT_WORDS: Mapping[str, str] = {
    **{word: "s" for word in ("sec", "secs", "second", "seconds")},
    **{word: "m" for word in ("min", "mins", "minute", "minutes")},
    **{word: "h" for word in ("hr", "hrs", "hour", "hours")},
    **{word: "d" for word in ("day", "days")},
    **{word: "w" for word in ("wk", "wks", "week", "weeks")},
}
"""The spelled-out units pup also takes (``5min``, ``2hours``, ``5 minutes``), any case, by
the unit letter they mean. A one-letter unit is lowercase only: ``1M`` is not a minute."""

_RELATIVE_TIME = re.compile(
    r"(?:(?P<now>now)(?:\s*-\s*(?P<ago>[0-9]+)\s*(?P<unit>[a-z]+))?"
    r"|(?P<bare>[0-9]+)\s*(?P<bare_unit>[a-z]+))",
    re.IGNORECASE,
)
_UNIT_SUFFIX = re.compile(r"[0-9]+\s*[a-z]+", re.IGNORECASE)
_RELATIVE_TIME_FORMS = (
    "7d, 24h, 30m, now-7d or now (a number and a unit: s, m, h, d or w, or spelled out, "
    "such as 5min, 2hours or '5 minutes')"
)


TIME_FORMS: Final = (
    "ISO 8601 (such as 2026-10-01 or 2026-10-01T12:00:00Z), epoch milliseconds (a bare "
    f"number), or a relative time: {_RELATIVE_TIME_FORMS}"
)
"""Every form a ``--from`` or ``--to`` takes, for a usage error."""

MIN_EPOCH_MS: Final = 10**11
"""The smallest bare number a ``--from`` or ``--to`` reads as epoch milliseconds (March
1973). A smaller one is far more likely epoch seconds, a date without dashes or a count of
days, so it is refused rather than read as a moment in 1970."""


def _bare_number_hint(value: str) -> str:
    """What a too-small bare number ``value`` probably meant, in the forms that say it."""
    meant = (
        f"{value} days? write {value}d"
        if len(value) <= 4
        else f"epoch seconds? append 000 ({value}000)"
    )
    date = (
        f"a date? write {value[:4]}-{value[4:6]}-{value[6:]}"
        if len(value) == 8
        else "a date? write it as 2026-10-01"
    )
    return f"{meant}; {date}"


def parse_time_bound(text: str, flag: str) -> str | int:
    """The ``flag`` (``--from`` or ``--to``) value ``text`` as a bound: epoch milliseconds
    as an integer, ISO 8601 as written, or a relative time in its canonical form.

    A relative time (``7d``, ``24 hours``, ``now-7d``, ``NOW``) becomes ``now`` or
    ``now-<n><unit>`` (:func:`canonical_relative_time`), so ``7d`` and ``now-7d`` freeze as
    the same expression. A bare number is epoch **milliseconds** (pup's own ``--from`` reads
    one as seconds); one below :data:`MIN_EPOCH_MS` is refused with a hint at what it
    probably meant (epoch seconds, a date, days). Checked here, before anything is frozen or fetched, so a value Datadog
    would refuse (``yesterday``) is a :class:`UsageError` naming the forms it takes
    (:data:`TIME_FORMS`).
    """
    value = text.strip()
    if not value:
        raise UsageError(f"{flag} must not be empty", hint=f"pass {TIME_FORMS}")
    if value.isascii() and value.isdigit():
        number = int(value)
        if number < MIN_EPOCH_MS:
            raise UsageError(
                f"{flag} {text!r} is too small for epoch milliseconds (a bare number)",
                hint=_bare_number_hint(value),
            )
        return number
    if is_relative_time(value):
        canonical = _canonical(value)
        if canonical is None:
            raise UsageError(f"cannot read {flag} {text!r}", hint=f"pass {TIME_FORMS}")
        return canonical
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise UsageError(f"cannot read {flag} {text!r}", hint=f"pass {TIME_FORMS}") from None
    return value


def is_relative_time(value: TimeBound) -> bool:
    """Whether ``value`` is a relative time: text starting with ``now``, in any case, or a
    number followed by letters (``7d``, ``5 minutes``)."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return text.lower().startswith("now") or _UNIT_SUFFIX.fullmatch(text) is not None


def canonical_relative_time(value: str) -> str:
    """The relative time ``value`` as ``now`` or ``now-<n><unit>`` (unit s, m, h, d or w):
    ``7d``, ``now - 7 days`` and ``now-7d`` all read ``now-7d``. Raises
    :class:`UsageError` when ``value`` is not one."""
    canonical = _canonical(value)
    if canonical is None:
        raise UsageError(
            f"cannot read the relative time {value!r}",
            hint=f"pass ISO 8601, epoch ms, or {_RELATIVE_TIME_FORMS}",
        )
    return canonical


def _canonical(value: str) -> str | None:
    match = _RELATIVE_TIME.fullmatch(value.strip())
    if match is None:
        return None
    if match["now"] is not None and match["ago"] is None:
        return "now"
    amount = match["ago"] if match["ago"] is not None else match["bare"]
    unit = _unit(match["unit"] if match["ago"] is not None else match["bare_unit"])
    if unit is None:
        return None
    return f"now-{int(amount)}{unit}"


def _unit(text: str) -> str | None:
    if len(text) == 1:
        return text if text in RELATIVE_TIME_UNITS_MS else None
    return RELATIVE_TIME_UNIT_WORDS.get(text.lower())


def resolve_time(value: TimeBound, now_ms: int) -> TimeBound:
    """``value`` as epoch milliseconds when it is a relative time, at ``now_ms``; else as is.

    A relative time is any form :func:`canonical_relative_time` reads; any other text that
    looks like one (:func:`is_relative_time`) raises :class:`UsageError` rather than being
    sent to Datadog unresolved. ISO 8601 text and epoch milliseconds are returned unchanged.
    """
    if not isinstance(value, str) or not is_relative_time(value):
        return value
    resolved = now_ms - _relative_offset_ms(value)
    if resolved < 0:
        raise UsageError(f"time {value!r} is before the epoch")
    return resolved


def epoch_ms(clock: Clock) -> int:
    """What ``clock`` reads, in whole epoch milliseconds."""
    return int(clock() * 1000)


def _relative_offset_ms(value: str) -> int:
    """How far before now the relative time ``value`` is, in milliseconds."""
    canonical = canonical_relative_time(value)
    if canonical == "now":
        return 0
    amount, unit = canonical[len("now-") : -1], canonical[-1]
    return int(amount) * RELATIVE_TIME_UNITS_MS[unit]


def page_file_name(number: int) -> str:
    """The cache file for the ``number``-th page (1-based): ``page-0001.json``."""
    return f"page-{number:04d}.json"


# --- span models ----------------------------------------------------------------------------

_TOLERANT = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


def _none_as_empty_list(value: object) -> object:
    return [] if value is None else value


def _none_as_empty_dict(value: object) -> object:
    return {} if value is None else value


class LLMObsToolCall(BaseModel):
    """A tool call the model made, on an assistant message."""

    model_config = _TOLERANT

    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    tool_id: str | None = None
    type: str | None = None


class LLMObsToolResult(BaseModel):
    """A tool's result, on a ``tool`` (or sometimes ``user``) message."""

    model_config = _TOLERANT

    name: str | None = None
    result: str | None = None
    tool_id: str | None = None
    type: str | None = None


class LLMObsMessage(BaseModel):
    """One message. ``role`` may be missing (output messages) or ``reasoning``.

    ``content`` is a string, or a list of content parts for multimodal input.
    """

    model_config = _TOLERANT

    role: str | None = None
    content: str | list[Any] | None = None
    tool_calls: list[LLMObsToolCall] = Field(default_factory=list)
    tool_results: list[LLMObsToolResult] = Field(default_factory=list)

    @field_validator("tool_calls", "tool_results", mode="before")
    @classmethod
    def _null_lists(cls, value: object) -> object:
        return _none_as_empty_list(value)


class LLMObsIO(BaseModel):
    """A span's input or output: a plain ``value``, chat ``messages``, or both."""

    model_config = _TOLERANT

    value: str | None = None
    messages: list[LLMObsMessage] | None = None


class LLMObsToolDefinition(BaseModel):
    """A tool the model was offered; ``schema_`` is its JSON-schema parameters (``schema``)."""

    model_config = _TOLERANT

    name: str | None = None
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class LLMObsSpan(BaseModel):
    """An LLM span from the Export API: the conversation the model saw and what it produced.

    ``span_id`` and ``trace_id`` are required; everything else is optional. ``start_ns`` is
    nanoseconds since the epoch and ``duration`` nanoseconds. ``tags`` are ``key:value``
    strings (a mapping is accepted and flattened).
    """

    model_config = _TOLERANT

    span_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    parent_id: str | None = None
    name: str | None = None
    span_kind: str | None = None
    ml_app: str | None = None
    model_name: str | None = None
    model_provider: str | None = None
    status: str | None = None
    start_ns: int | None = None
    duration: float | None = None
    input: LLMObsIO | None = None
    output: LLMObsIO | None = None
    tool_definitions: list[LLMObsToolDefinition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_definitions", mode="before")
    @classmethod
    def _null_lists(cls, value: object) -> object:
        return _none_as_empty_list(value)

    @field_validator("metadata", "evaluation", "metrics", mode="before")
    @classmethod
    def _null_dicts(cls, value: object) -> object:
        return _none_as_empty_dict(value)

    @field_validator("tags", mode="before")
    @classmethod
    def _flatten_tags(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, Mapping):
            return [f"{key}:{item}" for key, item in sorted(value.items())]
        return value


def parse_span(item: object) -> LLMObsSpan | None:
    """One ``data`` item (``{id, type, attributes}``) as an :class:`LLMObsSpan`, or ``None``.

    ``None`` means the item does not fit the model, even tolerantly. A missing
    ``attributes.span_id`` falls back to the item's ``id``.
    """
    if not isinstance(item, Mapping):
        return None
    attributes = item.get("attributes")
    if not isinstance(attributes, Mapping):
        return None
    data = dict(attributes)
    if not data.get("span_id") and isinstance(item.get("id"), str):
        data["span_id"] = item["id"]
    try:
        return LLMObsSpan.model_validate(data)
    except ValidationError:
        return None


# --- the query ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanQuery:
    """Which spans to export, and how many.

    ``from_`` and ``to`` take ISO 8601, a relative time such as ``now-7d`` or ``7d`` (see
    :func:`resolve_time`; text that looks like one but is not raises :class:`UsageError`),
    or epoch milliseconds.
    ``query`` is a search query string; ``tags`` must all match. ``max_spans`` caps how many
    export items :func:`iter_spans` reads, readable or not (``None``: all of them), so it
    yields at most that many spans and an export of unreadable items still stops.
    """

    ml_app: str | None = None
    span_name: str | None = None
    query: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    from_: TimeBound = None
    to: TimeBound = None
    span_kind: str = "llm"
    page_limit: int = DEFAULT_PAGE_LIMIT
    max_spans: int | None = None

    def __post_init__(self) -> None:
        if self.page_limit < 1:
            raise UsageError(f"page limit must be at least 1, not {self.page_limit}")
        if self.max_spans is not None and self.max_spans < 0:
            raise UsageError(f"max spans must not be negative, not {self.max_spans}")
        for bound in (self.from_, self.to):
            if isinstance(bound, str) and is_relative_time(bound):
                _relative_offset_ms(bound)
        object.__setattr__(self, "tags", dict(self.tags))

    @property
    def is_relative(self) -> bool:
        """Whether ``from_`` or ``to`` is a relative time, which Datadog resolves per page."""
        return is_relative_time(self.from_) or is_relative_time(self.to)

    def resolved(self, now_ms: int) -> SpanQuery:
        """This query with a relative ``from_`` or ``to`` resolved at ``now_ms`` to epoch ms."""
        return replace(
            self, from_=resolve_time(self.from_, now_ms), to=resolve_time(self.to, now_ms)
        )

    def to_filter(self) -> dict[str, Any]:
        """The ``filter`` as it identifies the query; unset fields are left out.

        Epoch milliseconds stay numbers here, as frozen settings and cache identities hold
        them; :meth:`request_body` sends them as strings.
        """
        candidates: dict[str, Any] = {
            "span_kind": self.span_kind,
            "ml_app": self.ml_app,
            "span_name": self.span_name,
            "query": self.query,
            "tags": dict(sorted(self.tags.items())),
            "from": self.from_,
            "to": self.to,
        }
        return {key: value for key, value in candidates.items() if value not in (None, "", {})}

    def settings(self) -> dict[str, Any]:
        """What selects the spans, as a stable mapping to freeze in a dataset's settings.

        ``max_spans`` is left out: it bounds how far an export reads, not which spans match,
        so an export can be resumed with a higher cap.
        """
        return {"filter": self.to_filter(), "page_limit": self.page_limit}

    def request_body(self, cursor: str | None = None) -> dict[str, Any]:
        """The JSON:API request body for one page, starting at ``cursor``."""
        page: dict[str, Any] = {"limit": self.page_limit}
        if cursor:
            page["cursor"] = cursor
        return {
            "data": {
                "type": "spans",
                "attributes": {
                    "filter": _wire_filter(self.to_filter()),
                    "page": page,
                    "sort": "timestamp",
                    "options": {"include_attachments": True},
                },
            }
        }


_QUERY_TERMS = (("ml_app", "@ml_app"), ("span_kind", "@meta.span.kind"), ("span_name", "@name"))


def _wire_filter(filter_: dict[str, Any]) -> dict[str, Any]:
    """``filter_`` as the Export API reads it: an epoch-ms ``from`` or ``to`` as a string.

    The API ignores the other fields when a ``query`` is set, so they are added to the query.
    """
    wire = {
        key: str(value) if key in ("from", "to") and isinstance(value, int) else value
        for key, value in filter_.items()
    }
    if "query" in wire:
        terms = [f"{name}:{_quoted(wire[key])}" for key, name in _QUERY_TERMS if key in wire]
        terms += [f"{key}:{_quoted(value)}" for key, value in filter_.get("tags", {}).items()]
        wire["query"] = " ".join([*terms, f"({wire['query']})"])
    return wire


def _quoted(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --- the export -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnreadableSpan:
    """A ``data`` item the export skipped: where it was, and whatever ids it carried."""

    page: int
    """The page it was on, 1-based, counting cached pages."""
    index: int
    """Its position in the page's ``data`` list, 0-based."""
    span_id: str | None = None
    trace_id: str | None = None


@dataclass
class ExportStats:
    """Counts for one :func:`iter_spans` export, updated as it is consumed."""

    pages_fetched: int = 0
    """Pages read from the API in this export."""
    pages_cached: int = 0
    """Pages re-read from the cache instead of the API."""
    spans: int = 0
    """Spans yielded."""
    invalid_spans: int = 0
    """Spans skipped because they did not fit :class:`LLMObsSpan`."""
    unreadable: list[UnreadableSpan] = field(default_factory=list)
    """Where each of the ``invalid_spans`` was, in the order they were read."""
    timeout_retries: int = 0
    """Retry attempts for pages the API reported as timed out."""
    exhausted: bool = False
    """Whether the export read to the end of the results.

    Stopping at ``max_spans`` counts only when that item was the last of a page with no next
    cursor; otherwise a matching span may be left, so this stays ``False``.
    """


class SpanExport(Iterator[LLMObsSpan]):
    """The iterator :func:`iter_spans` returns; :attr:`stats` fills in as it is consumed."""

    def __init__(self, spans: Iterator[LLMObsSpan], stats: ExportStats) -> None:
        self._spans = spans
        self.stats = stats

    def __iter__(self) -> SpanExport:
        return self

    def __next__(self) -> LLMObsSpan:
        return next(self._spans)


class SpanPageExport(Iterator[list[LLMObsSpan]]):
    """The iterator :func:`iter_span_pages` returns; :attr:`stats` fills in as it is consumed."""

    def __init__(self, pages: Iterator[list[LLMObsSpan]], stats: ExportStats) -> None:
        self._pages = pages
        self.stats = stats

    def __iter__(self) -> SpanPageExport:
        return self

    def __next__(self) -> list[LLMObsSpan]:
        return next(self._pages)


@dataclass(frozen=True)
class _Cursor:
    next_cursor: str | None
    pages: int
    done: bool


Page = Mapping[str, Any]
Sleep = Callable[[float], None]


def iter_spans(
    pup: Pup,
    query: SpanQuery,
    *,
    cache_dir: Path | None = None,
    sleep: Sleep = time.sleep,
    timeout_backoff: Sequence[float] = TIMEOUT_BACKOFF_SECONDS,
) -> SpanExport:
    """Export the spans matching ``query``, oldest first, as a lazy iterator.

    Pagination stops when the response has no next cursor or repeats one already followed.
    A page the API reports as ``timeout`` is retried after each wait in ``timeout_backoff``;
    when every retry times out, :class:`ExternalError` is raised.

    With ``cache_dir`` (made ``0700``, created if missing), each raw page is saved as
    ``page-0001.json``, ``page-0002.json``, ... and ``cursor.json`` records
    ``{next_cursor, pages, done}``, before the page's spans are yielded. A later call with
    the same query and directory re-reads the cached pages and continues from
    ``next_cursor``; a cache written for a different query, or for another Datadog site or
    org than pup's login (read with ``pup auth status`` first), raises :class:`StateError`.
    ``cursor.json`` records that ``account`` (``{site, org}``, nothing secret); when pup's
    default session reports no org name, the org is told apart by its id instead
    (``{site, org: null, org_id}``, from ``GET v2/current_user``).
    Re-read pages yield their spans again, so callers that persist results should dedupe
    by span id. Non-finite numbers in a page are stored as ``null``.
    """
    pages = iter_span_pages(
        pup, query, cache_dir=cache_dir, sleep=sleep, timeout_backoff=timeout_backoff
    )
    return SpanExport(_flatten(pages), pages.stats)


def iter_span_pages(
    pup: Pup,
    query: SpanQuery,
    *,
    cache_dir: Path | None = None,
    sleep: Sleep = time.sleep,
    timeout_backoff: Sequence[float] = TIMEOUT_BACKOFF_SECONDS,
) -> SpanPageExport:
    """The export :func:`iter_spans` makes, as one list of spans per page.

    The next page is fetched (or re-read from the cache) only when the next list is asked
    for, so a caller that stops between pages fetches nothing more. A page whose spans all
    failed to parse is an empty list; the page whose items reach ``max_spans`` (counting the
    ones that failed to parse) is cut there.
    """
    stats = ExportStats()
    pages = _pages(pup, query, cache_dir, stats, sleep, timeout_backoff)
    return SpanPageExport(_span_pages(pages, query.max_spans, stats), stats)


def _flatten(pages: Iterator[list[LLMObsSpan]]) -> Iterator[LLMObsSpan]:
    for spans in pages:
        yield from spans


def _span_pages(
    pages: Iterator[Page], max_spans: int | None, stats: ExportStats
) -> Iterator[list[LLMObsSpan]]:
    if max_spans == 0:
        return
    for number, page in enumerate(pages, start=1):
        spans: list[LLMObsSpan] = []
        items = _items(page)
        for index, item in enumerate(items):
            span = parse_span(item)
            if span is None:
                stats.invalid_spans += 1
                stats.unreadable.append(_unreadable(item, number, index))
            else:
                stats.spans += 1
                spans.append(span)
            # Every item counts, so an export of unreadable items stops too.
            if max_spans is not None and stats.spans + stats.invalid_spans >= max_spans:
                # The cap fell on the very last item: nothing is left to read.
                stats.exhausted = index == len(items) - 1 and _next_cursor(page) is None
                yield spans
                return
        yield spans
    stats.exhausted = True


def _unreadable(item: object, page: int, index: int) -> UnreadableSpan:
    """Where the unreadable ``item`` was, with the span and trace ids it has, if any."""
    attributes = item.get("attributes") if isinstance(item, Mapping) else None
    attributes = attributes if isinstance(attributes, Mapping) else {}
    span_id = attributes.get("span_id") or (item.get("id") if isinstance(item, Mapping) else None)
    trace_id = attributes.get("trace_id")
    return UnreadableSpan(
        page,
        index,
        span_id if isinstance(span_id, str) and span_id else None,
        trace_id if isinstance(trace_id, str) and trace_id else None,
    )


def _items(page: Page) -> list[Any]:
    data = page.get("data")
    return data if isinstance(data, list) else []


def _pages(
    pup: Pup,
    query: SpanQuery,
    cache_dir: Path | None,
    stats: ExportStats,
    sleep: Sleep,
    timeout_backoff: Sequence[float],
) -> Iterator[Page]:
    cursor = _Cursor(next_cursor=None, pages=0, done=False)
    query_hash = settings_hash(query.settings())
    # Every cursor followed so far, cached pages included, so a loop back is caught on resume.
    followed: set[str | None] = {None}
    account: Account = {}
    if cache_dir is not None:
        account = _account(pup)
        make_private_dir(cache_dir)
        cursor = _read_cursor(cache_dir, query_hash, account)
        for number in range(1, cursor.pages + 1):
            stats.pages_cached += 1
            cached = _read_page(cache_dir, number)
            followed.add(_next_cursor(cached))
            yield cached
        if cursor.done:
            return
    followed.add(cursor.next_cursor)
    while True:
        page = _fetch_page(
            pup,
            query,
            cursor.next_cursor,
            stats,
            sleep,
            timeout_backoff,
            cached=cache_dir is not None,
        )
        stats.pages_fetched += 1
        next_cursor = _next_cursor(page)
        done = next_cursor is None or next_cursor in followed
        followed.add(next_cursor)
        cursor = _Cursor(
            next_cursor=None if done else next_cursor, pages=cursor.pages + 1, done=done
        )
        if cache_dir is not None:
            atomic_write_json(cache_dir / page_file_name(cursor.pages), page, ensure_ascii=True)
            _write_cursor(cache_dir, cursor, query_hash, account)
        yield page
        if done:
            return


def _fetch_page(
    pup: Pup,
    query: SpanQuery,
    cursor: str | None,
    stats: ExportStats,
    sleep: Sleep,
    timeout_backoff: Sequence[float],
    *,
    cached: bool,
) -> Page:
    """One page; retries a page the API reports as timed out."""
    body = query.request_body(cursor)
    attempts = len(timeout_backoff) + 1
    for attempt in range(attempts):
        if attempt:
            stats.timeout_retries += 1
            sleep(timeout_backoff[attempt - 1])
        response = pup.api("POST", EXPORT_PATH, body, content_type=CONTENT_TYPE)
        page = _as_page(response)
        if _status(page) != "timeout":
            return page
    hint = "narrow the time range or filters, then retry"
    raise ExternalError(
        f"the span search timed out {attempts} times in a row",
        source="datadog",
        hint=f"{hint} {CACHED_PAGES_NOTE}" if cached else hint,
    )


def _as_page(response: object) -> Page:
    if not isinstance(response, Mapping) or not isinstance(response.get("data"), list):
        message = "the span search returned an unexpected response (no `data` list)"
        detail = _error_detail(response)
        raise ExternalError(f"{message}: {detail}" if detail else message, source="datadog")
    # Our JSON writers refuse NaN and Infinity, so a cached page must not hold them.
    return finite_json(response)


def _error_detail(response: object) -> str | None:
    """The first JSON:API ``errors[].detail`` (or ``title``) in ``response``, if any."""
    errors = response.get("errors") if isinstance(response, Mapping) else None
    if not isinstance(errors, list):
        return None
    for error in errors:
        if isinstance(error, Mapping):
            for key in ("detail", "title"):
                value = error.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _meta(page: Page) -> Mapping[str, Any]:
    meta = page.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _status(page: Page) -> str | None:
    status = _meta(page).get("status")
    return status if isinstance(status, str) else None


def _next_cursor(page: Page) -> str | None:
    paging = _meta(page).get("page")
    after = paging.get("after") if isinstance(paging, Mapping) else None
    return after if isinstance(after, str) and after else None


# --- the cache ------------------------------------------------------------------------------

Account = dict[str, str | None]
"""The Datadog ``site`` and ``org`` an export reads from, and the ``org_id`` when unnamed."""


def _account(pup: Pup) -> Account:
    """The site and org pup is logged in to; :class:`ExternalError` when it is not.

    pup's default session reports no org name, so ``org`` alone would not tell two orgs on
    one site apart: the org's id is read then (``org_id``).
    """
    status = pup.auth_status()
    if not status["authenticated"]:
        raise ExternalError(
            "pup is not logged in to Datadog", source="pup", hint="run `pup auth login`"
        )
    account: Account = {"site": status["site"], "org": status["org"]}
    if status["org"] is None:
        account["org_id"] = pup.current_org_id()
    return account


def _describe_account(account: object) -> str:
    if not isinstance(account, Mapping):
        return "an unknown account"
    site = account.get("site") or "?"
    if account.get("org") is None and account.get("org_id"):
        return f"site {site}, org id {account['org_id']}"
    return f"site {site}, org {account.get('org') or '?'}"


def _read_cursor(cache_dir: Path, query_hash: str, account: Account) -> _Cursor:
    path = cache_dir / CURSOR_FILE_NAME
    try:
        data = read_json(path)
    except FileNotFoundError:
        return _Cursor(next_cursor=None, pages=0, done=False)
    if not isinstance(data, dict):
        raise StateError(f"{path}: expected a JSON object")
    next_cursor, pages, done = data.get("next_cursor"), data.get("pages"), data.get("done")
    if (
        not isinstance(data.get("settings_hash"), str)
        or not (next_cursor is None or isinstance(next_cursor, str))
        or type(pages) is not int
        or pages < 0
        or not isinstance(done, bool)
        or (pages > 0 and not done and next_cursor is None)
    ):
        raise StateError(
            f"{path}: invalid span export cursor",
            hint=f"delete {cache_dir} to export from the start",
        )
    if data.get("settings_hash") != query_hash:
        raise SettingsConflict(
            f"{cache_dir} holds spans exported for a different query",
            hint="use a new directory, or delete this one to export from the start",
        )
    recorded = data.get("account")
    if recorded != account:
        unidentified = (
            isinstance(recorded, Mapping)
            and recorded.get("org") is None
            and not recorded.get("org_id")
        )
        raise SettingsConflict(
            f"{cache_dir} holds spans exported from {_describe_account(recorded)}, "
            f"but pup is logged in to {_describe_account(account)}",
            hint=(
                "the cache does not record which org it came from; use a new directory"
                if unidentified
                else "use a new directory, or log pup in to the account the cache came from"
            ),
        )
    return _Cursor(next_cursor=next_cursor, pages=pages, done=done)


def _write_cursor(cache_dir: Path, cursor: _Cursor, query_hash: str, account: Account) -> None:
    data: dict[str, Any] = {
        "next_cursor": cursor.next_cursor,
        "pages": cursor.pages,
        "done": cursor.done,
        "settings_hash": query_hash,
        "account": account,
    }
    atomic_write_json(cache_dir / CURSOR_FILE_NAME, data, ensure_ascii=True)


def _read_page(cache_dir: Path, number: int) -> Page:
    path = cache_dir / page_file_name(number)
    try:
        page = read_json(path)
    except FileNotFoundError as exc:
        raise StateError(
            f"{path} is missing from the span export cache",
            hint=f"delete {cache_dir} to export from the start",
        ) from exc
    if not isinstance(page, dict) or not isinstance(page.get("data"), list):
        raise StateError(
            f"{path}: expected a JSON object with a `data` list",
            hint=f"delete {cache_dir} to export from the start",
        )
    return page
