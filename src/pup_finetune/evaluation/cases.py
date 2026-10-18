# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Replay cases: one recorded assistant action, the conversation before it, and its tools.

Replay evaluation asks a model to redo each recorded action. For every assistant message of
a test-split row that does something visible (non-empty content, or tool calls), the model
sees the recorded prefix (every message before that assistant message) plus the row's
tools, generates one assistant message, and that message is compared with the recorded one,
the *reference*. Tools are never executed: the conversation always continues from what was
recorded, not from what the model would have done.

Replay therefore measures agreement with the recorded behavior. It does not measure whether
the task succeeded: a model that solves the task another way scores as a disagreement, and
one that copies a recorded mistake scores as agreement.

Only trained turns become cases by default: an assistant message with ``weight`` ``0`` was
left out of training on purpose (a turn the agent should not imitate), so comparing a model
with it would reward the behavior training avoided. ``include_untrained=True`` keeps them.

Messages and tools are in the OpenAI form :func:`~pup_finetune.data.jsonl.to_provider_dict`
produces, minus ``weight``, which is training metadata a model never sees. The reference
keeps its ``reasoning_content`` for display; the metrics ignore it
(:func:`~pup_finetune.evaluation.metrics.score_case`).
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.schema import Message, TrainingRow
from pup_finetune.data.split import SPLITS, SplitName
from pup_finetune.display import joined
from pup_finetune.rendering.base import ChatMessage, Renderer, ToolSpec

_HASH_SPACE = 2**64


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """One assistant action to replay: ``prefix`` and ``tools`` in, compared with ``reference``.

    ``id`` is ``f"{row_id}:{message_index}"``, where ``message_index`` is the reference's
    position in the row's messages. ``future_tool_results`` are the tool messages that
    directly followed the reference in the recording: context for a judge reading the
    reference's calls, never shown to the model being evaluated.

    The cases of one row share their message and tool dicts (a reference is the same object
    as that message in a later case's ``prefix``), so treat them as read-only: copy a dict
    before changing it.
    """

    id: str
    row_id: str
    trace_id: str | None
    message_index: int
    prefix: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...]
    reference: ChatMessage
    future_tool_results: tuple[ChatMessage, ...] = ()
    split: SplitName = "test"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "row_id": self.row_id,
            "trace_id": self.trace_id,
            "message_index": self.message_index,
            "split": self.split,
            "prefix": list(self.prefix),
            "tools": list(self.tools),
            "reference": self.reference,
            "future_tool_results": list(self.future_tool_results),
        }


def has_visible_action(message: Message) -> bool:
    """Whether an assistant ``message`` does something: says something or calls a tool.

    Reasoning alone is not an action, and whitespace is not content.
    """
    return message.role == "assistant" and (
        bool(message.content.strip()) or bool(message.tool_calls)
    )


def case_indices(row: TrainingRow, *, include_untrained: bool = False) -> list[int]:
    """The message indices of ``row`` that are cases, before any cap, in order."""
    return [
        index
        for index, message in enumerate(row.messages)
        if has_visible_action(message) and (include_untrained or message.weight != 0)
    ]


def select_spread(count: int, cap: int, *, key: str, seed: int) -> list[int]:
    """``cap`` of the positions ``0..count-1``, evenly spaced, always with the last one.

    The positions are cut into ``cap`` equal buckets and one is taken from each, at the same
    offset in every bucket; the offset comes from ``sha256(f"replay:{seed}:{key}")``, so the
    choice is deterministic per key and seed, in every process, but does not always favor the
    first turn. (The ``replay:`` prefix keeps it independent of the split hash, which puts
    every test row's split key low.) The last bucket's pick is the last position. Returns
    every position when ``count <= cap``.
    """
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    if count <= cap:
        return list(range(count))
    digest = hashlib.sha256(f"replay:{seed}:{key}".encode()).hexdigest()
    phase = int(digest[:16], 16)
    # floor((i + phase / 2**64) * count / cap), in exact integer arithmetic.
    picks = [((i * _HASH_SPACE + phase) * count) // (cap * _HASH_SPACE) for i in range(cap)]
    picks[-1] = count - 1
    return picks


def _message_dict(message: Message) -> ChatMessage:
    return message.model_dump(mode="json", exclude_none=True, exclude={"weight"})


def row_cases(
    row: TrainingRow,
    *,
    max_cases_per_trajectory: int | None = None,
    seed: int = 0,
    include_untrained: bool = False,
    split: SplitName = "test",
) -> list[ReplayCase]:
    """The replay cases of one row, in message order (see :func:`build_cases`)."""
    indices = case_indices(row, include_untrained=include_untrained)
    if max_cases_per_trajectory is not None:
        picks = select_spread(len(indices), max_cases_per_trajectory, key=row.id, seed=seed)
        indices = [indices[pick] for pick in picks]
    if not indices:
        return []
    messages = [_message_dict(message) for message in row.messages]
    tools = tuple(tool.model_dump(mode="json", exclude_none=True) for tool in row.tools)
    cases: list[ReplayCase] = []
    for index in indices:
        following: list[ChatMessage] = []
        for message in messages[index + 1 :]:
            if message["role"] != "tool":
                break
            following.append(message)
        cases.append(
            ReplayCase(
                id=f"{row.id}:{index}",
                row_id=row.id,
                trace_id=row.source.trace_id,
                message_index=index,
                prefix=tuple(messages[:index]),
                tools=tools,
                reference=messages[index],
                future_tool_results=tuple(following),
                split=split,
            )
        )
    return cases


def build_cases(
    rows: Iterable[TrainingRow],
    *,
    max_cases_per_trajectory: int | None,
    seed: int,
    include_untrained: bool = False,
    split: SplitName = "test",
) -> list[ReplayCase]:
    """One case per trained assistant message with a visible action, ordered by row id, then index.

    ``max_cases_per_trajectory`` caps the cases per row (``None`` for no cap): the kept ones
    are evenly spaced over the row's eligible turns and always include its final one (see
    :func:`select_spread`). ``include_untrained`` also makes cases of ``weight`` ``0`` turns.
    ``split`` is recorded on each case. Two rows with the same id raise ``ValueError``,
    since their case ids would collide.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    if max_cases_per_trajectory is not None and max_cases_per_trajectory < 1:
        raise ValueError(
            f"max_cases_per_trajectory must be at least 1, got {max_cases_per_trajectory}"
        )
    ordered = sorted(rows, key=lambda row: row.id)
    for earlier, later in itertools.pairwise(ordered):
        if earlier.id == later.id:
            raise ValueError(f"row id {later.id!r} appears more than once")
    cases: list[ReplayCase] = []
    for row in ordered:
        cases.extend(
            row_cases(
                row,
                max_cases_per_trajectory=max_cases_per_trajectory,
                seed=seed,
                include_untrained=include_untrained,
                split=split,
            )
        )
    return cases


def cases_from_dataset(
    dataset: DatasetDir,
    *,
    split: SplitName = "test",
    max_cases_per_trajectory: int | None = None,
    seed: int = 0,
    include_untrained: bool = False,
) -> list[ReplayCase]:
    """The cases of a dataset's kept rows in ``split`` (rows labeled ``keep: false`` are out)."""
    return build_cases(
        dataset.rows_for_split(split),
        max_cases_per_trajectory=max_cases_per_trajectory,
        seed=seed,
        include_untrained=include_untrained,
        split=split,
    )


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why a case is left out at planning: ``model``'s renderer refuses its prompt
    (:meth:`~pup_finetune.rendering.base.Renderer.prompt_refusal`), so no sampler of that
    model could continue it. ``code`` is the renderer's stable error code."""

    model: str
    code: str
    detail: str


def refusals(cases: Sequence[ReplayCase], renderers: Mapping[str, Renderer]) -> dict[str, Refusal]:
    """The cases a renderer of ``renderers`` (by model) refuses, by case id, each with the
    first model that refuses it."""
    found: dict[str, Refusal] = {}
    for case in cases:
        for model, renderer in renderers.items():
            error = renderer.prompt_refusal(case.prefix, case.tools)
            if error is not None:
                found[case.id] = Refusal(model, error.code.value, error.detail)
                break
    return found


def refused_by_reason(refused: Mapping[str, Refusal]) -> dict[str, int]:
    return dict(sorted(Counter(refusal.code for refusal in refused.values()).items()))


def refusal_warning(refused: Mapping[str, Refusal], *, cases: int, models: int = 1) -> str | None:
    """What a preview and a result warn about the ``refused`` cases of ``cases``, ``None``
    when there are none; with several ``models``, each case is left out for all of them."""
    if not refused:
        return None
    first = next(iter(refused.values()))
    refusing = joined(sorted({refusal.model for refusal in refused.values()}))
    every = " for every model, so each is ranked on the same cases" if models > 1 else ""
    return (
        f"{len(refused)} of {cases} test cases are left out{every}: the chat format of "
        f"{refusing} cannot continue their prompt ({first.code}: {first.detail})"
    )
