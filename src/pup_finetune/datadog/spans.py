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
fetching the cached pages again.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import ExternalError, StateError, UsageError
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

    ``from_`` and ``to`` take ISO 8601, date math such as ``now-7d``, or epoch milliseconds.
    ``query`` is a search query string; ``tags`` must all match. ``max_spans`` caps how many
    spans :func:`iter_spans` yields (``None``: all of them).
    """

    ml_app: str | None = None
    span_name: str | None = None
    query: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    from_: str | int | None = None
    to: str | int | None = None
    span_kind: str = "llm"
    page_limit: int = DEFAULT_PAGE_LIMIT
    max_spans: int | None = None

    def __post_init__(self) -> None:
        if self.page_limit < 1:
            raise UsageError(f"page limit must be at least 1, not {self.page_limit}")
        if self.max_spans is not None and self.max_spans < 0:
            raise UsageError(f"max spans must not be negative, not {self.max_spans}")
        object.__setattr__(self, "tags", dict(self.tags))

    def to_filter(self) -> dict[str, Any]:
        """The Export API ``filter``; unset fields are left out."""
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
                    "filter": self.to_filter(),
                    "page": page,
                    "sort": "timestamp",
                    "options": {"include_attachments": True},
                },
            }
        }


# --- the export -----------------------------------------------------------------------------


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
    timeout_retries: int = 0
    """Retry attempts for pages the API reported as timed out."""
    exhausted: bool = False
    """Whether the export read to the end of the results.

    ``False`` whenever it stopped at ``max_spans``, even if no matching span was left.
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
    ``next_cursor``; a cache written for a different query raises :class:`StateError`.
    Re-read pages yield their spans again, so callers that persist results should dedupe
    by span id. Non-finite numbers in a page are stored as ``null``.
    """
    stats = ExportStats()
    pages = _pages(pup, query, cache_dir, stats, sleep, timeout_backoff)
    return SpanExport(_spans(pages, query.max_spans, stats), stats)


def _spans(
    pages: Iterator[Page], max_spans: int | None, stats: ExportStats
) -> Iterator[LLMObsSpan]:
    if max_spans == 0:
        return
    for page in pages:
        for item in _items(page):
            span = parse_span(item)
            if span is None:
                stats.invalid_spans += 1
                continue
            stats.spans += 1
            yield span
            if max_spans is not None and stats.spans >= max_spans:
                return
    stats.exhausted = True


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
    if cache_dir is not None:
        make_private_dir(cache_dir)
        cursor = _read_cursor(cache_dir, query_hash)
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
            _write_cursor(cache_dir, cursor, query_hash)
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
        hint=f"{hint} (cached pages are kept)" if cached else hint,
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


def _read_cursor(cache_dir: Path, query_hash: str) -> _Cursor:
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
        raise StateError(
            f"{cache_dir} holds spans exported for a different query",
            hint="use a new directory, or delete this one to export from the start",
        )
    return _Cursor(next_cursor=next_cursor, pages=pages, done=done)


def _write_cursor(cache_dir: Path, cursor: _Cursor, query_hash: str) -> None:
    data: dict[str, Any] = {
        "next_cursor": cursor.next_cursor,
        "pages": cursor.pages,
        "done": cursor.done,
        "settings_hash": query_hash,
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
