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
take a place by outranking one. :func:`review_markdown` writes a sample in full, in reading
order, for a person to label.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
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
SYSTEM_PROMPT_LINES = 5
"""How many lines of a system prompt a review file shows."""
MAX_SYSTEM_PROMPT_CHARS = 800
MAX_TOOL_RESULT_LINES = 40
MAX_TOOL_RESULT_CHARS = 2000
"""Longest tool result a review file shows; a longer one is cut, its length noted."""


@dataclass(frozen=True, slots=True)
class Sample:
    """The rows a sample picked, in row-id order with their splits, the labels in force and
    how many rows were eligible."""

    picked: list[tuple[TrainingRow, SplitName | None]]
    labels: Mapping[str, Label]
    eligible: int

    def described(self) -> list[dict[str, Any]]:
        return [describe_row(row, split, self.labels) for row, split in self.picked]


def sample_rows(
    dataset: DatasetDir,
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    split: SampleSplit = "all",
    seed: int = DEFAULT_SAMPLE_SEED,
    include_dropped: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Up to ``n`` rows of ``split`` described in brief, and how many rows were eligible
    (see :func:`pick_sample`)."""
    sample = pick_sample(dataset, n=n, split=split, seed=seed, include_dropped=include_dropped)
    return sample.described(), sample.eligible


def pick_sample(
    dataset: DatasetDir,
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    split: SampleSplit = "all",
    seed: int = DEFAULT_SAMPLE_SEED,
    include_dropped: bool = False,
) -> Sample:
    """Up to ``n`` rows of ``split``, listed in row-id order.

    Rows labeled ``keep: false`` are left out unless ``include_dropped``.
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
    return Sample(picked, labels, len(eligible))


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


def review_markdown(
    sample: Sample, *, dataset_dir: str, split: SampleSplit, seed: int, label_command: str
) -> str:
    """``sample`` as a Markdown review file: how to record decisions, then each row in
    reading order with a labels line to copy. Message text is fenced, so it is never
    rendered as Markdown."""
    where = "every split" if split == "all" else f"the {split} split"
    steps = (
        "1. Read each row and decide whether to train on it.",
        "2. Copy each row's labels line into a new JSON Lines file, `<LABELS_FILE>`.",
        '3. To drop a row, set `"keep": false` and say why in `"reason"`: a drop needs one.',
        "4. Record the decisions:",
    )
    parts = [
        f"# Review: {len(sample.picked)} of {sample.eligible} rows of {_code(dataset_dir)}",
        f"Sampled from {where} with seed {seed}. This file holds trace content: it is private "
        "(mode 0600), so keep it out of git and do not share it.",
        "## How to record decisions",
        "\n".join(steps),
        _fenced(label_command, "sh"),
        "A row's last label wins; a row left out of the file stays unlabeled.",
    ]
    for index, (row, row_split) in enumerate(sample.picked, start=1):
        parts += _review_row(index, row, row_split, sample.labels.get(row.id))
    return "\n\n".join(parts) + "\n"


def _review_row(
    index: int, row: TrainingRow, split: SplitName | None, label: Label | None
) -> list[str]:
    brief = row_brief(row)
    tools = ", ".join(_code(name) for name in brief["tools_used"]) or "none"
    facts = [
        f"- Split: {split or 'none'}",
        f"- Label: {_label_text(label)}",
        f"- Assistant turns: {brief['turns']}; tools called: {tools}",
    ]
    if row.source.trace_id:
        facts.append(f"- Trace: {_code(row.source.trace_id)}")
    parts = ["---", f"## {index}. {_code(row.id)}", "\n".join(facts)]
    names = {call.id: call.function.name for m in row.messages for call in m.tool_calls or ()}
    for position, message in enumerate(row.messages):
        parts += _review_message(message, final=position == len(row.messages) - 1, names=names)
    line = {
        "id": row.id,
        "keep": True if label is None else label.keep,
        "reason": "" if label is None else label.reason,
    }
    return [
        *parts,
        "### Labels line",
        _fenced(json.dumps(line, ensure_ascii=False), "json"),
        'To drop this row, set `"keep": false` and give a `"reason"`.',
    ]


def _review_message(message: Message, *, final: bool, names: Mapping[str, str]) -> list[str]:
    if message.role == "system":
        text = message.content
        head = _head(text, SYSTEM_PROMPT_LINES, MAX_SYSTEM_PROMPT_CHARS)
        shown = f"; the first {len(head):,} characters are shown" if head is not None else ""
        note = f"_{_size(text)}{shown}._"
        return ["### System prompt", note, _fenced(text if head is None else head, "text")]
    if message.role == "tool":
        call_id = message.tool_call_id or ""
        heading = f"### Tool result: {_code(names.get(call_id, call_id))} ({_code(call_id)})"
        return [heading, *_text_block(message.content, cut=True)]
    if message.role == "user":
        return ["### User", *_text_block(message.content)]
    parts = ["### Final answer" if final else "### Assistant"]
    if message.content.strip() or not message.tool_calls:
        parts += _text_block(message.content)
    for call in message.tool_calls or ():
        arguments = json.dumps(json.loads(call.function.arguments), indent=2, ensure_ascii=False)
        parts += [
            f"Tool call {_code(call.function.name)} ({_code(call.id)}):",
            _fenced(arguments, "json"),
        ]
    return parts


def _text_block(text: str, *, cut: bool = False) -> list[str]:
    if not text.strip():
        return ["_(empty)_"]
    head = _head(text, MAX_TOOL_RESULT_LINES, MAX_TOOL_RESULT_CHARS) if cut else None
    if head is None:
        return [_fenced(text, "text")]
    return [
        _fenced(head, "text"),
        f"_Cut: {_size(text)} in all; the first {len(head):,} characters are shown (the full "
        "row is in rows.jsonl)._",
    ]


def _size(text: str) -> str:
    lines = len(text.splitlines())
    return f"{len(text):,} characters, {lines:,} line{'' if lines == 1 else 's'}"


def _head(text: str, lines: int, chars: int) -> str | None:
    """The first ``lines`` lines of ``text``, at most ``chars`` characters, or ``None`` when
    ``text`` fits."""
    split = text.splitlines()
    if len(split) <= lines and len(text) <= chars:
        return None
    return "\n".join(split[:lines])[:chars]


def _label_text(label: Label | None) -> str:
    if label is None:
        return "none"
    text = f"{'keep' if label.keep else 'drop'} ({label.by})"
    reason = " ".join(label.reason.split())
    return f"{text}: {reason}" if reason else text


_BACKTICKS = re.compile(r"`+")


def _fenced(text: str, info: str) -> str:
    """``text`` in a code fence longer than any run of backticks in it."""
    longest = max((len(run) for run in _BACKTICKS.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    body = text.rstrip("\n")
    return f"{fence}{info}\n{body}\n{fence}"


def _code(text: str) -> str:
    """``text`` as inline code on one line."""
    text = text.replace("\r", " ").replace("\n", " ")
    tick = "`" * (max((len(run) for run in _BACKTICKS.findall(text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{tick}{pad}{text}{pad}{tick}"
