# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A small, readable sample of a dataset's rows, for reviewing them before labeling.

Each row is shown in brief (its split, its label, its assistant turns and tools, and the
first user and final assistant messages, truncated); the full rows stay in ``rows.jsonl``.
The sample is a pure function of the rows and the seed: each row is ranked by a hash of the
seed and its id, so the same seed picks the same rows, and adding rows only lets a new row
take a place by outranking one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.labels import Label
from pup_finetune.data.schema import Message, TrainingRow, split_key
from pup_finetune.data.split import SPLITS, SplitName

SampleSplit = Literal["train", "validation", "test", "all"]

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SAMPLE_SEED = 0
MAX_TEXT_CHARS = 300
"""Longest text shown for a message; longer text is cut and ends with an ellipsis."""
ELLIPSIS = "…"


def sample_rows(
    dataset: DatasetDir,
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    split: SampleSplit = "all",
    seed: int = DEFAULT_SAMPLE_SEED,
    include_dropped: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Up to ``n`` rows of ``split`` described in brief, and how many rows were eligible.

    Rows labeled ``keep: false`` are left out unless ``include_dropped``. The sample is
    listed in row-id order.
    """
    if split != "all" and split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    # Rows first: an add saves the split assignments before the rows, so assignments read
    # after the rows cover every row read.
    rows = dataset.rows()
    splits = dataset.splits()
    labels = dataset.labels()
    eligible: list[tuple[TrainingRow, SplitName | None]] = []
    for row in rows:
        row_split = splits.get(split_key(row))
        label = labels.get(row.id)
        if split != "all" and row_split != split:
            continue
        if label is not None and not label.keep and not include_dropped:
            continue
        eligible.append((row, row_split))
    ranked = sorted(eligible, key=lambda item: _rank(seed, item[0].id))[: max(n, 0)]
    picked = sorted(ranked, key=lambda item: item[0].id)
    return [describe_row(row, row_split, labels) for row, row_split in picked], len(eligible)


def describe_row(
    row: TrainingRow, split: SplitName | None, labels: Mapping[str, Label]
) -> dict[str, Any]:
    """``row`` in brief, for a person or an agent to read: :func:`row_summary`, its split
    and its label."""
    label = labels.get(row.id)
    return {
        **row_summary(row),
        "split": split,
        "label": "none" if label is None else "keep" if label.keep else "drop",
    }


def row_summary(row: TrainingRow) -> dict[str, Any]:
    """:func:`row_brief` with the first user message and the final assistant message, cut
    to :data:`MAX_TEXT_CHARS`: enough to judge whether a span filter picks the right rows."""
    users = [message for message in row.messages if message.role == "user"]
    assistants = [message for message in row.messages if message.role == "assistant"]
    return {
        **row_brief(row),
        "first_user_message": truncate(users[0].content) if users else None,
        "final_assistant_message": render_assistant(assistants[-1]) if assistants else None,
    }


def row_brief(row: TrainingRow) -> dict[str, Any]:
    """A row's id, its assistant turns and the names of the tools it calls."""
    tools = {call.function.name for message in row.messages for call in message.tool_calls or ()}
    return {
        "id": row.id,
        "turns": sum(1 for message in row.messages if message.role == "assistant"),
        "tools_used": sorted(tools),
    }


def render_assistant(message: Message) -> str:
    """An assistant message's text, or when it has none, its tool calls as ``name(args)``."""
    if message.content.strip() or not message.tool_calls:
        return truncate(message.content)
    calls = ", ".join(
        f"{call.function.name}({call.function.arguments})" for call in message.tool_calls
    )
    return truncate(calls)


def truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """``text`` cut to at most ``limit`` characters, ending in an ellipsis when cut."""
    if len(text) <= limit:
        return text
    return text[: limit - len(ELLIPSIS)] + ELLIPSIS


def _rank(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{row_id}".encode()).hexdigest()
