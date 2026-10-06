# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training rows from the LLM spans of whole traces.

An agent calls its model many times per trace, and each call's input repeats the
conversation so far, so most spans hold a prefix of a later span's conversation.
:func:`rows_from_spans` converts every span (:func:`~pup_finetune.data.convert.span_conversation`)
and keeps, per trace, only the conversations no other one contains:

- A conversation whose messages are a strict prefix of a valid row's messages in the same
  trace is dropped: the longer row trains on the same turns. Sub-agents, with their own
  system prompts, are prefixes of nothing and stay separate rows.
- Of identical conversations, one is kept: a valid one over an invalid one, then the one with
  the latest ``start_ns``, then the one that arrived last.

Messages compare by role, content and tool calls (name, and arguments as canonical JSON), and
a tool message by the position of the call it answers (which message, which call in it), as
:func:`comparable_messages` gives them. Tool call ids and reasoning do not take part: a
generated id differs between spans, and a span's input rarely repeats the reasoning its
predecessor output. The position keeps apart conversations that pair the same results with
different calls of a parallel tool call. Canonical arguments keep together the same call
recorded as an object on one span and as formatted JSON text on another.

Each kept conversation is a row with the id ``{trace_id}:{span_id}``, or an
:class:`~pup_finetune.data.exclusions.Exclusion` with a stable code when it could not be
converted or is not a valid row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from pup_finetune.data.convert import (
    Conversation,
    ReasoningMode,
    check_reasoning,
    span_conversation,
)
from pup_finetune.data.exclusions import Exclusion
from pup_finetune.data.schema import RowError, TrainingRow, validate_row
from pup_finetune.datadog.spans import LLMObsSpan


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """One kept conversation: a valid ``row``, or the ``exclusion`` saying why it is not one."""

    row: TrainingRow | None = None
    exclusion: Exclusion | None = None


@dataclass
class ConversionStats:
    """Counts for one :func:`rows_from_spans` conversion, final once it is consumed."""

    spans: int = 0
    """Distinct spans read (by span id)."""
    duplicate_spans: int = 0
    """Spans skipped because a span with the same id was already read."""
    traces: int = 0
    """Distinct traces."""
    collapsed: int = 0
    """Conversations dropped as a prefix or a duplicate of another in the same trace."""
    rows: int = 0
    """Valid rows yielded."""
    exclusions: int = 0
    """Exclusions yielded."""


class RowConversion(Iterator[ConversionResult]):
    """The iterator :func:`rows_from_spans` returns; :attr:`stats` fills in as it is consumed."""

    def __init__(self, results: Iterator[ConversionResult], stats: ConversionStats) -> None:
        self._results = results
        self.stats = stats

    def __iter__(self) -> RowConversion:
        return self

    def __next__(self) -> ConversionResult:
        return next(self._results)


def rows_from_spans(
    spans: Iterable[LLMObsSpan], *, reasoning: ReasoningMode = "omit"
) -> RowConversion:
    """Convert ``spans`` into one result per distinct conversation of each trace.

    Spans are deduplicated by span id first (the first one read wins), as a resumed export
    replays its cached pages. A trace's spans may arrive anywhere in the input, so v1
    buffers it whole: the first ``next()`` reads every span (keeping each one's converted
    result, a row or an exclusion, and its prefix digests, not the span), then results are
    yielded trace by trace, in the order the traces first appeared and, within a trace, the
    order their spans did. The iterator interface leaves room to stream completed traces
    later.

    An unknown ``reasoning`` mode raises :class:`~pup_finetune.errors.UsageError` here,
    before any span is read.
    """
    check_reasoning(reasoning)
    stats = ConversionStats()
    return RowConversion(_results(spans, reasoning, stats), stats)


@dataclass(frozen=True, slots=True)
class _Candidate:
    result: ConversionResult
    prefixes: tuple[str, ...]
    """The digest of each prefix of the messages: ``prefixes[k]`` covers ``k + 1`` messages."""
    rank: tuple[bool, int, int]
    """Which of identical conversations to keep: valid, then latest ``start_ns``, then last."""

    @property
    def signature(self) -> str:
        return self.prefixes[-1] if self.prefixes else ""

    @property
    def valid(self) -> bool:
        return self.result.row is not None


def _results(
    spans: Iterable[LLMObsSpan], reasoning: ReasoningMode, stats: ConversionStats
) -> Iterator[ConversionResult]:
    traces: dict[str, list[_Candidate]] = {}
    seen: set[str] = set()
    for arrival, span in enumerate(spans):
        if span.span_id in seen:
            stats.duplicate_spans += 1
            continue
        seen.add(span.span_id)
        stats.spans += 1
        conversation = span_conversation(span, reasoning=reasoning)
        traces.setdefault(span.trace_id, []).append(_candidate(conversation, arrival))
    stats.traces = len(traces)
    for candidates in traces.values():
        kept = _collapse(candidates)
        stats.collapsed += len(candidates) - len(kept)
        for candidate in kept:
            if candidate.valid:
                stats.rows += 1
            else:
                stats.exclusions += 1
            yield candidate.result


def _candidate(conversation: Conversation, arrival: int) -> _Candidate:
    source = conversation.source
    row_id = f"{source.trace_id}:{source.span_id}"
    result = _convert(conversation, row_id)
    start_ns = source.start_ns if source.start_ns is not None else -1
    return _Candidate(
        result,
        _prefix_digests(conversation),
        (result.row is not None, start_ns, arrival),
    )


def _convert(conversation: Conversation, row_id: str) -> ConversionResult:
    source = conversation.source
    ids = {"row_id": row_id, "trace_id": source.trace_id, "span_id": source.span_id}
    if conversation.issues:
        code, detail = conversation.issues[0]
        if more := len(conversation.issues) - 1:
            detail = f"{detail} (and {more} more issue(s))"
        return ConversionResult(exclusion=Exclusion.for_source(code, detail, **ids))
    data = {
        "id": row_id,
        "messages": conversation.messages,
        "tools": conversation.tools,
        "source": source,
    }
    try:
        return ConversionResult(row=validate_row(data))
    except RowError as error:
        return ConversionResult(exclusion=Exclusion.for_source(error.code, error.detail, **ids))


def _collapse(candidates: list[_Candidate]) -> list[_Candidate]:
    """The candidates no other one in the trace contains, in their original order."""
    best: dict[str, _Candidate] = {}
    for candidate in candidates:
        kept = best.get(candidate.signature)
        if kept is None or candidate.rank > kept.rank:
            best[candidate.signature] = candidate
    # Every strict prefix of a valid row (the empty one too): a conversation that ends at
    # one is contained in that row.
    contained = {digest for c in candidates if c.valid for digest in ("", *c.prefixes[:-1])}
    return [
        candidate
        for candidate in candidates
        if best[candidate.signature] is candidate and candidate.signature not in contained
    ]


def _prefix_digests(conversation: Conversation) -> tuple[str, ...]:
    """A running digest over the messages' comparable parts, one per message."""
    digests: list[str] = []
    running = hashlib.sha256()
    for part in comparable_messages(conversation.messages):
        running.update(hashlib.sha256(part.encode("utf-8", "surrogatepass")).digest())
        digests.append(running.copy().hexdigest())
    return tuple(digests)


def comparable_messages(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    """What each of ``messages`` compares by when conversations are matched, one per message.

    ``messages`` are canonical message dicts (a row's messages as ``model_dump`` gives them
    work too). Each part covers the role, the content, the tool calls (name and arguments,
    the arguments as canonical JSON, so the same object recorded as a dict on one span and
    as formatted JSON on another compares equal) and, for a tool message, the position of
    the call it answers. Tool call ids and reasoning are left out (see the module doc). Two
    conversations share a prefix of ``k`` messages exactly when their first ``k`` parts are
    equal, as a part depends only on its message and the ones before it.
    """
    parts: list[str] = []
    # Where each tool call is, as [message index, call index], by its id.
    positions: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        calls = []
        for call_index, call in enumerate(message.get("tool_calls") or ()):
            function = call["function"]
            calls.append([function["name"], _canonical_arguments(function["arguments"])])
            positions.setdefault(call["id"], [index, call_index])
        answers = positions.get(message.get("tool_call_id") or "")
        # Not canonical_json: a non-text content part is kept as given and may hold anything.
        parts.append(
            json.dumps(
                [message["role"], message["content"], calls, answers],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=repr,
            )
        )
    return parts


def _canonical_arguments(arguments: str) -> str:
    """``arguments`` as compact JSON with sorted keys, or as given when it is not JSON.

    A string that is not JSON never equals the canonical form of one that is, so the two
    cannot be mistaken for each other.
    """
    try:
        return json.dumps(
            json.loads(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except (ValueError, RecursionError):
        return arguments
