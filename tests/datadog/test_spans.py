# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.datadog.spans import (
    CONTENT_TYPE,
    CURSOR_FILE_NAME,
    EXPORT_PATH,
    TIMEOUT_BACKOFF_SECONDS,
    LLMObsSpan,
    SpanQuery,
    epoch_ms,
    is_relative_time,
    iter_spans,
    page_file_name,
    parse_span,
    resolve_time,
)
from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.storage import atomic_write_json, read_json, settings_hash
from tests.fakes.llmobs import SYNTHETIC_ACCOUNT, FakeExportApi, fixture_pages, page, span_item
from tests.fakes.pup import FAKE_BINARY, FakePup


class Sleeps:
    """An injectable ``sleep`` that records the waits instead of sleeping."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def three_pages() -> list[dict[str, Any]]:
    return [
        page([span_item("s1"), span_item("s2")], "cursor-2"),
        page([span_item("s3"), span_item("s4")], "cursor-3"),
        page([span_item("s5")]),
    ]


def export(
    pages: list[dict[str, Any]], query: SpanQuery | None = None, **kwargs: Any
) -> tuple[FakeExportApi, list[LLMObsSpan], Any]:
    api = FakeExportApi(pages)
    pup = api.install(FakePup()).client()
    result = iter_spans(pup, query or SpanQuery(ml_app="synthetic-app"), sleep=Sleeps(), **kwargs)
    spans = list(result)
    return api, spans, result.stats


def ids(spans: list[LLMObsSpan]) -> list[str]:
    return [span.span_id for span in spans]


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- the request ----------------------------------------------------------------------------


def test_request_goes_through_pup_api_with_the_json_api_body() -> None:
    query = SpanQuery(
        ml_app="synthetic-app",
        span_name="chat",
        query="@status:ok",
        tags={"version": "1", "env": "synthetic"},
        from_="now-7d",
        to="now",
        page_limit=2,
    )
    api, _, _ = export(three_pages(), query)
    assert api.calls[0].argv == [
        FAKE_BINARY,
        "--no-agent",
        "--output",
        "json",
        "api",
        EXPORT_PATH,
        "-X",
        "POST",
        "--input",
        "-",
        "-H",
        f"Content-Type: {CONTENT_TYPE}",
    ]
    assert api.requests[0] == {
        "data": {
            "type": "spans",
            "attributes": {
                "filter": {
                    "span_kind": "llm",
                    "ml_app": "synthetic-app",
                    "span_name": "chat",
                    "query": "@status:ok",
                    "tags": {"env": "synthetic", "version": "1"},
                    "from": "now-7d",
                    "to": "now",
                },
                "page": {"limit": 2},
                "sort": "timestamp",
                "options": {"include_attachments": True},
            },
        }
    }
    assert api.requests[1]["data"]["attributes"]["page"] == {"limit": 2, "cursor": "cursor-2"}


def test_filter_leaves_out_unset_fields() -> None:
    assert SpanQuery().to_filter() == {"span_kind": "llm"}
    assert SpanQuery(from_=1767225600000, tags={}).to_filter() == {
        "span_kind": "llm",
        "from": 1767225600000,
    }


def test_settings_are_stable_and_ignore_max_spans() -> None:
    first = SpanQuery(ml_app="a", tags={"b": "2", "a": "1"}, max_spans=10)
    second = SpanQuery(ml_app="a", tags={"a": "1", "b": "2"}, max_spans=500)
    assert first.settings() == second.settings()
    assert settings_hash(first.settings()) == settings_hash(second.settings())
    assert first.settings() == {
        "filter": {"span_kind": "llm", "ml_app": "a", "tags": {"a": "1", "b": "2"}},
        "page_limit": 1000,
    }
    assert SpanQuery(ml_app="other").settings() != first.settings()


def test_query_rejects_bad_limits() -> None:
    with pytest.raises(UsageError):
        SpanQuery(page_limit=0)
    with pytest.raises(UsageError):
        SpanQuery(max_spans=-1)


# --- relative times -------------------------------------------------------------------------

NOW_MS = 1767225600000


@pytest.mark.parametrize(
    ("value", "offset_ms"),
    [
        ("now", 0),
        ("now-0d", 0),
        ("now-30s", 30 * 1000),
        ("now-15m", 15 * 60 * 1000),
        ("now-2h", 2 * 60 * 60 * 1000),
        ("now-7d", 7 * 24 * 60 * 60 * 1000),
        ("now-1w", 7 * 24 * 60 * 60 * 1000),
        (" now-1d ", 24 * 60 * 60 * 1000),
    ],
)
def test_a_relative_time_resolves_to_epoch_ms(value: str, offset_ms: int) -> None:
    assert is_relative_time(value)
    assert resolve_time(value, NOW_MS) == NOW_MS - offset_ms


@pytest.mark.parametrize(
    "value", ["2025-12-01T00:00:00Z", "2025-12-01", 1767225600000, "1767225600000", None, "²"]
)
def test_an_absolute_time_is_returned_unchanged(value: str | int | None) -> None:
    assert not is_relative_time(value)
    assert resolve_time(value, NOW_MS) == value


@pytest.mark.parametrize(
    "value", ["now-", "now-7", "now-d", "now-7y", "now-1M", "now+1h", "now - 7d", "Now", "nowish"]
)
def test_an_unreadable_relative_time_is_a_usage_error(value: str) -> None:
    assert is_relative_time(value)
    with pytest.raises(UsageError, match="cannot read the relative time"):
        resolve_time(value, NOW_MS)
    with pytest.raises(UsageError, match="cannot read the relative time"):
        SpanQuery(from_=value)
    with pytest.raises(UsageError, match="cannot read the relative time"):
        SpanQuery(to=value)


def test_a_relative_time_before_the_epoch_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="before the epoch"):
        resolve_time("now-1d", 1000)


def test_a_resolved_query_sends_epoch_ms() -> None:
    query = SpanQuery(ml_app="synthetic-app", from_="now-1h", to="now")
    assert query.is_relative
    resolved = query.resolved(NOW_MS)
    assert not resolved.is_relative
    assert resolved.to_filter()["from"] == NOW_MS - 60 * 60 * 1000
    assert resolved.to_filter()["to"] == NOW_MS
    assert resolved.ml_app == "synthetic-app"
    pinned = SpanQuery(from_="2025-12-01T00:00:00Z", to=NOW_MS)
    assert not pinned.is_relative
    assert pinned.resolved(0) == pinned


def test_epoch_ms_reads_the_clock_in_milliseconds() -> None:
    assert epoch_ms(lambda: 1767225600.1234) == 1767225600123


# --- pagination -----------------------------------------------------------------------------


def test_pages_through_every_cursor() -> None:
    api, spans, stats = export(three_pages())
    assert ids(spans) == ["s1", "s2", "s3", "s4", "s5"]
    assert api.cursors == [None, "cursor-2", "cursor-3"]
    assert (stats.pages_fetched, stats.pages_cached, stats.spans) == (3, 0, 5)
    assert stats.exhausted


@pytest.mark.parametrize("cached", [False, True])
def test_a_repeated_cursor_stops_pagination(tmp_path: Path, cached: bool) -> None:
    pages = [
        page([span_item("s1")], "cursor-2"),
        page([span_item("s2")], "cursor-2"),
    ]
    cache = tmp_path / "raw" if cached else None
    api, spans, stats = export(pages, cache_dir=cache)
    assert ids(spans) == ["s1", "s2"]
    assert api.cursors == [None, "cursor-2"]
    assert stats.exhausted
    if cache is not None:
        cursor = read_json(cache / CURSOR_FILE_NAME)
        assert (cursor["next_cursor"], cursor["pages"], cursor["done"]) == (None, 2, True)
        api, replayed, _ = export(pages, cache_dir=cache)
        assert ids(replayed) == ["s1", "s2"]
        assert api.requests == []


def test_a_resume_catches_a_cursor_followed_before_it(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    pages = [
        page([span_item("s1")], "cursor-2"),
        page([span_item("s2")], "cursor-3"),
        page([span_item("s3")], "cursor-2"),
    ]
    _, spans, _ = export(pages, SpanQuery(max_spans=2), cache_dir=cache)
    assert ids(spans) == ["s1", "s2"]
    api, spans, stats = export(pages, SpanQuery(), cache_dir=cache)
    assert ids(spans) == ["s1", "s2", "s3"]
    assert api.cursors == ["cursor-3"]
    assert stats.exhausted
    assert read_json(cache / CURSOR_FILE_NAME)["done"] is True


def test_an_empty_cursor_ends_pagination() -> None:
    first = page([span_item("s1")])
    first["meta"]["page"] = {"after": ""}
    api, spans, _ = export([first])
    assert ids(spans) == ["s1"]
    assert api.cursors == [None]


def test_spans_are_fetched_lazily() -> None:
    api = FakeExportApi(three_pages())
    result = iter_spans(api.install(FakePup()).client(), SpanQuery())
    assert api.requests == []
    assert next(result).span_id == "s1"
    assert api.cursors == [None]


def test_max_spans_stops_mid_page() -> None:
    api, spans, stats = export(three_pages(), SpanQuery(max_spans=3))
    assert ids(spans) == ["s1", "s2", "s3"]
    assert api.cursors == [None, "cursor-2"]
    assert not stats.exhausted


def test_max_spans_counts_invalid_spans() -> None:
    items: list[Any] = [
        "not an object",
        span_item("good-1"),
        span_item("bad-start", start_ns="yesterday"),
        span_item("good-2"),
        span_item("good-3"),
    ]
    _, spans, stats = export([page(items)], SpanQuery(max_spans=3))
    assert ids(spans) == ["good-1"]
    assert (stats.spans, stats.invalid_spans) == (1, 2)
    assert [u.index for u in stats.unreadable] == [0, 2]
    assert not stats.exhausted


def test_max_spans_stops_an_export_of_unreadable_items() -> None:
    pages = [page(["bad-1", "bad-2"], "cursor-2"), page(["bad-3"], "cursor-3"), page(["bad-4"])]
    api, spans, stats = export(pages, SpanQuery(max_spans=1))
    assert spans == []
    assert (stats.spans, stats.invalid_spans) == (0, 1)
    assert api.cursors == [None]
    assert not stats.exhausted


def test_max_spans_on_the_last_item_is_exhausted_even_when_unreadable() -> None:
    _, spans, stats = export([page([span_item("good-1"), "bad-1"])], SpanQuery(max_spans=2))
    assert ids(spans) == ["good-1"]
    assert stats.exhausted


def test_max_spans_zero_fetches_nothing() -> None:
    api, spans, _ = export(three_pages(), SpanQuery(max_spans=0))
    assert spans == []
    assert api.requests == []


# --- timeouts and failures ------------------------------------------------------------------


def test_a_timed_out_page_is_retried_with_backoff() -> None:
    api = FakeExportApi(three_pages())
    api.timeouts["cursor-2"] = 2
    sleeps = Sleeps()
    result = iter_spans(api.install(FakePup()).client(), SpanQuery(), sleep=sleeps)
    assert ids(list(result)) == ["s1", "s2", "s3", "s4", "s5"]
    assert api.cursors == [None, "cursor-2", "cursor-2", "cursor-2", "cursor-3"]
    assert sleeps.waits == list(TIMEOUT_BACKOFF_SECONDS[:2])
    assert result.stats.timeout_retries == 2


def test_a_page_that_keeps_timing_out_raises() -> None:
    api = FakeExportApi(three_pages())
    api.timeouts[None] = 10
    sleeps = Sleeps()
    result = iter_spans(api.install(FakePup()).client(), SpanQuery(), sleep=sleeps)
    with pytest.raises(ExternalError, match="timed out 4 times"):
        list(result)
    assert api.cursors == [None] * 4
    assert sleeps.waits == list(TIMEOUT_BACKOFF_SECONDS)


def test_timeouts_with_a_cache_never_cache_a_timed_out_page(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    pages = three_pages()
    api = FakeExportApi(pages)
    api.timeouts["cursor-2"] = 1
    api.timeouts["cursor-3"] = 10
    pup = api.install(FakePup()).client()
    with pytest.raises(ExternalError, match="timed out 4 times") as failure:
        list(iter_spans(pup, SpanQuery(), cache_dir=cache, sleep=Sleeps()))
    assert failure.value.hint is not None and "cached pages are kept" in failure.value.hint
    assert read_json(cache / page_file_name(2)) == pages[1]
    assert not (cache / page_file_name(3)).exists()
    cursor = read_json(cache / CURSOR_FILE_NAME)
    assert (cursor["next_cursor"], cursor["pages"], cursor["done"]) == ("cursor-3", 2, False)

    api.requests.clear()
    api.timeouts.clear()
    result = iter_spans(pup, SpanQuery(), cache_dir=cache, sleep=Sleeps())
    assert ids(list(result)) == ["s1", "s2", "s3", "s4", "s5"]
    assert api.cursors == ["cursor-3"]


def test_the_timeout_hint_mentions_the_cache_only_with_one() -> None:
    api = FakeExportApi(three_pages())
    api.timeouts[None] = 10
    result = iter_spans(api.install(FakePup()).client(), SpanQuery(), sleep=Sleeps())
    with pytest.raises(ExternalError) as failure:
        list(result)
    assert failure.value.hint is not None and "cache" not in failure.value.hint


def test_a_pup_failure_raises_external_error() -> None:
    api = FakeExportApi(three_pages())
    api.failures["cursor-2"] = 1
    result = iter_spans(api.install(FakePup()).client(), SpanQuery())
    with pytest.raises(ExternalError, match="502"):
        list(result)


@pytest.mark.parametrize(
    "response", [[], {}, {"data": {"not": "a list"}}, {"errors": [{"detail": "x"}]}, "text"]
)
def test_an_unexpected_response_raises(response: Any) -> None:
    fake = FakePup().on("api", EXPORT_PATH, json=response)
    with pytest.raises(ExternalError, match="unexpected response"):
        list(iter_spans(fake.client(), SpanQuery()))


def test_an_unexpected_response_reports_the_api_error(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    response = {"errors": [{"title": "Bad Request", "detail": "invalid filter"}]}
    fake = FakePup().on("api", EXPORT_PATH, json=response)
    fake.on("auth", "status", json={"authenticated": True, **SYNTHETIC_ACCOUNT})
    with pytest.raises(ExternalError, match="invalid filter"):
        list(iter_spans(fake.client(), SpanQuery(), cache_dir=cache))
    assert not (cache / CURSOR_FILE_NAME).exists()


# --- the cache ------------------------------------------------------------------------------


def test_cache_writes_every_raw_page_and_the_cursor(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    pages = three_pages()
    query = SpanQuery()
    export(pages, query, cache_dir=cache)
    assert sorted(path.name for path in cache.iterdir()) == [
        CURSOR_FILE_NAME,
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]
    assert mode(cache) == 0o700
    for number, expected in enumerate(pages, start=1):
        path = cache / page_file_name(number)
        assert mode(path) == 0o600
        assert read_json(path) == expected
    assert read_json(cache / CURSOR_FILE_NAME) == {
        "next_cursor": None,
        "pages": 3,
        "done": True,
        "settings_hash": settings_hash(query.settings()),
        "account": SYNTHETIC_ACCOUNT,
    }


def test_an_existing_cache_dir_is_made_private(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    cache.mkdir(mode=0o755)
    cache.chmod(0o755)
    export(three_pages(), cache_dir=cache)
    assert mode(cache) == 0o700


def test_a_finished_cache_is_replayed_without_fetching(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    export(three_pages(), cache_dir=cache)
    api, spans, stats = export(three_pages(), cache_dir=cache)
    assert ids(spans) == ["s1", "s2", "s3", "s4", "s5"]
    assert api.requests == []
    assert (stats.pages_fetched, stats.pages_cached) == (0, 3)
    assert stats.exhausted


def test_resumes_from_a_partial_cache_without_refetching(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    api = FakeExportApi(three_pages())
    api.failures["cursor-3"] = 1
    pup = api.install(FakePup()).client()
    seen: list[str] = []
    with pytest.raises(ExternalError):
        for span in iter_spans(pup, SpanQuery(), cache_dir=cache):
            seen.append(span.span_id)
    assert seen == ["s1", "s2", "s3", "s4"]
    assert read_json(cache / CURSOR_FILE_NAME)["next_cursor"] == "cursor-3"
    assert read_json(cache / CURSOR_FILE_NAME)["pages"] == 2

    api.requests.clear()
    result = iter_spans(pup, SpanQuery(), cache_dir=cache)
    assert ids(list(result)) == ["s1", "s2", "s3", "s4", "s5"]
    assert api.cursors == ["cursor-3"]
    assert (result.stats.pages_cached, result.stats.pages_fetched) == (2, 1)
    assert read_json(cache / CURSOR_FILE_NAME)["done"] is True


def test_resume_after_max_spans_continues_past_the_cached_pages(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    api, spans, _ = export(three_pages(), SpanQuery(max_spans=1), cache_dir=cache)
    assert ids(spans) == ["s1"]
    assert api.cursors == [None]
    api, spans, _ = export(three_pages(), SpanQuery(max_spans=4), cache_dir=cache)
    assert ids(spans) == ["s1", "s2", "s3", "s4"]
    assert api.cursors == ["cursor-2"]


def test_a_cache_for_another_query_is_refused(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    export(three_pages(), SpanQuery(ml_app="one"), cache_dir=cache)
    with pytest.raises(StateError, match="different query"):
        export(three_pages(), SpanQuery(ml_app="two"), cache_dir=cache)


@pytest.mark.parametrize(
    ("account", "message"),
    [
        ({"site": "datadoghq.com", "org": "another-org"}, "org another-org"),
        ({"site": "datadoghq.eu", "org": "synthetic-org"}, "site datadoghq.eu"),
    ],
    ids=["org", "site"],
)
def test_a_cache_for_another_datadog_account_is_refused(
    tmp_path: Path, account: dict[str, str], message: str
) -> None:
    cache = tmp_path / "raw"
    api = FakeExportApi(three_pages())
    pup = api.install(FakePup()).client()
    assert ids(list(iter_spans(pup, SpanQuery(), cache_dir=cache))) == ids_of_three_pages()
    api.requests.clear()
    api.account = account
    with pytest.raises(StateError, match=message) as failure:
        list(iter_spans(pup, SpanQuery(), cache_dir=cache))
    assert "site datadoghq.com, org synthetic-org" in failure.value.message
    assert api.requests == []


def test_a_cache_needs_pup_to_be_logged_in(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    fake = FakeExportApi(three_pages()).install(FakePup())
    fake.on("auth", "status", json={"authenticated": False})
    with pytest.raises(ExternalError, match="not logged in"):
        list(iter_spans(fake.client(), SpanQuery(), cache_dir=cache))
    assert not cache.exists()


def test_a_cache_without_its_account_is_refused(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    export(three_pages(), cache_dir=cache)
    cursor = read_json(cache / CURSOR_FILE_NAME)
    del cursor["account"]
    atomic_write_json(cache / CURSOR_FILE_NAME, cursor)
    with pytest.raises(StateError, match="an unknown account"):
        export(three_pages(), cache_dir=cache)


def ids_of_three_pages() -> list[str]:
    return ["s1", "s2", "s3", "s4", "s5"]


def test_a_missing_cached_page_is_reported(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    export(three_pages(), cache_dir=cache)
    (cache / page_file_name(2)).unlink()
    with pytest.raises(StateError, match="missing"):
        export(three_pages(), cache_dir=cache)


@pytest.mark.parametrize(
    "cursor",
    [
        {"pages": "two", "done": False},
        {"next_cursor": None, "pages": 3, "done": True},
    ],
)
def test_a_corrupt_cursor_is_reported(tmp_path: Path, cursor: dict[str, Any]) -> None:
    cache = tmp_path / "raw"
    export(three_pages(), cache_dir=cache)
    atomic_write_json(cache / CURSOR_FILE_NAME, cursor)
    with pytest.raises(StateError, match="invalid span export cursor"):
        export(three_pages(), cache_dir=cache)


@pytest.mark.parametrize("cached_page", [[], {"meta": {}}, {"data": "not a list"}])
def test_a_corrupt_cached_page_is_reported(tmp_path: Path, cached_page: Any) -> None:
    cache = tmp_path / "raw"
    export(three_pages(), cache_dir=cache)
    atomic_write_json(cache / page_file_name(2), cached_page)
    with pytest.raises(StateError, match="`data` list"):
        export(three_pages(), cache_dir=cache)


def test_non_finite_numbers_are_cached_as_null(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    item = span_item("s1", metrics={"input_tokens": 3, "score": float("nan")})
    _, spans, _ = export([page([item])], cache_dir=cache)
    assert spans[0].metrics == {"input_tokens": 3, "score": None}
    cached = read_json(cache / page_file_name(1))
    assert cached["data"][0]["attributes"]["metrics"]["score"] is None
    api, replayed, stats = export([page([item])], cache_dir=cache)
    assert replayed == spans
    assert api.requests == []
    assert stats.pages_cached == 1


def test_lone_surrogates_survive_caching_and_parsing(tmp_path: Path) -> None:
    cache = tmp_path / "raw"
    text = "a broken emoji \ud83d at the end"
    item = span_item("s1", input={"messages": [{"role": "user", "content": text}]})
    _, spans, _ = export([page([item])], cache_dir=cache)
    assert spans[0].input is not None and spans[0].input.messages is not None
    assert spans[0].input.messages[0].content == text
    cached = read_json(cache / page_file_name(1))
    assert cached["data"][0]["attributes"]["input"]["messages"][0]["content"] == text
    api, replayed, stats = export([page([item])], cache_dir=cache)
    assert replayed == spans
    assert api.requests == []
    assert stats.pages_cached == 1


# --- parsing --------------------------------------------------------------------------------


def fixture_spans() -> dict[str, LLMObsSpan]:
    _, spans, stats = export(fixture_pages())
    assert stats.invalid_spans == 0
    return {span.span_id: span for span in spans}


def test_fixture_pages_parse_tolerantly() -> None:
    spans = fixture_spans()
    assert len(spans) == 10
    trace = [span for span in spans.values() if span.trace_id == "synthetic-trace-0101"]
    assert len(trace) == 5
    inputs = [
        len(span.input.messages)
        for span in trace
        if span.input and span.input.messages and span.parent_id == "synthetic-span-0100"
    ]
    assert inputs == [2, 4, 6, 8]


def test_tool_calls_tool_results_and_definitions() -> None:
    span = fixture_spans()["synthetic-span-0103"]
    assert (span.name, span.ml_app) == ("chat", "synthetic-trip-planner")
    assert (span.model_name, span.model_provider) == ("example-model-large", "example-provider")
    assert span.input is not None and span.input.messages is not None
    roles = [message.role for message in span.input.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    with_content = span.input.messages[4]
    assert with_content.content and with_content.tool_calls[0].name == "search_hotels"
    assert with_content.tool_calls[0].arguments == {"city": "Northvale", "max_price": 150}
    assert with_content.tool_calls[0].type == "function"
    assert with_content.tool_calls[0].tool_id == "call_0101b"
    tool = span.input.messages[3]
    assert tool.content == ""
    assert tool.tool_results[0].tool_id == "call_0101a"
    assert tool.tool_results[0].result == (
        '{"city": "Northvale", "temp_c": 12, "conditions": "cloudy"}'
    )
    assert tool.tool_results[0].type == "tool_result"
    assert [definition.name for definition in span.tool_definitions] == [
        "get_weather",
        "search_hotels",
        "summarize_options",
    ]
    assert span.tool_definitions[0].description == "Current conditions for a city."
    assert span.tool_definitions[0].schema_ == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert span.start_ns == 1767225605000000000
    assert "env:synthetic" in span.tags


def test_odd_output_shapes() -> None:
    spans = fixture_spans()
    assert spans["synthetic-span-0201"].status == "error"
    reasoning = spans["synthetic-span-0301"].output
    assert reasoning is not None and reasoning.messages is not None
    assert [message.role for message in reasoning.messages] == ["reasoning", "assistant"]
    roleless = spans["synthetic-span-0401"].output
    assert roleless is not None and roleless.messages is not None
    assert roleless.messages[0].role is None
    assert spans["synthetic-span-0501"].output is None
    image = spans["synthetic-span-0601"].input
    assert image is not None and image.messages is not None
    parts = image.messages[1].content
    assert isinstance(parts, list) and [part["type"] for part in parts] == ["text", "image_url"]
    assert spans["synthetic-span-0104"].evaluation == {"helpfulness": "good"}


def test_nulls_and_tag_mappings_are_tolerated() -> None:
    span = parse_span(
        {
            "id": "s1",
            "type": "span",
            "attributes": {
                "trace_id": "t1",
                "tags": {"env": "synthetic", "app": "a"},
                "tool_definitions": None,
                "metadata": None,
                "metrics": None,
                "evaluation": None,
                "input": {"messages": [{"role": "user", "content": None, "tool_calls": None}]},
                "an_unknown_field": {"ignored": True},
            },
        }
    )
    assert span is not None
    assert span.span_id == "s1"
    assert span.tags == ["app:a", "env:synthetic"]
    assert (span.tool_definitions, span.metadata, span.metrics, span.evaluation) == ([], {}, {}, {})
    assert span.input is not None and span.input.messages is not None
    assert span.input.messages[0].tool_calls == []


def test_invalid_spans_are_skipped_and_counted() -> None:
    items: list[Any] = [
        span_item("good-1"),
        {"id": "no-trace", "type": "span", "attributes": {"span_id": "no-trace"}},
        span_item("bad-input", input="not an object"),
        span_item("bad-messages", input={"messages": [{"content": {"a": "dict"}}]}),
        span_item("bad-start", start_ns="yesterday"),
        {"id": "no-attributes", "type": "span"},
        "not an object",
        span_item("good-2"),
    ]
    _, spans, stats = export([page(items)])
    assert ids(spans) == ["good-1", "good-2"]
    assert stats.invalid_spans == 6
    assert stats.spans == 2
