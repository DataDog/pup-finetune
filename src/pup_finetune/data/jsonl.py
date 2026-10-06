# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Canonical rows on disk as JSON Lines, and the provider-facing form of a row."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pup_finetune.data.schema import Message, RowError, TrainingRow, validate_row
from pup_finetune.storage import atomic_write_text, canonical_json, read_jsonl_numbered


def read_rows(path: Path) -> Iterator[TrainingRow]:
    """Yield each row of a canonical JSONL file, validated.

    An invalid row raises :class:`RowError` carrying the file and line number; a line that
    is not a JSON object raises :class:`~pup_finetune.errors.StateError`.
    """
    path = Path(path)
    for line_number, data in read_jsonl_numbered(path):
        try:
            yield validate_row(data)
        except RowError as exc:
            raise RowError(exc.code, exc.detail, path=str(path), line=line_number) from exc


def row_to_dict(row: TrainingRow) -> dict[str, Any]:
    """The row as stored on disk: every field except unset (``None``) ones."""
    return row.model_dump(mode="json", exclude_none=True)


def write_rows(path: Path, rows: Iterable[TrainingRow]) -> int:
    """Replace ``path`` with ``rows`` as canonical JSONL, atomically and ``0600``.

    Returns the number of rows written. The parent directory must already exist.
    """
    lines = [canonical_json(row_to_dict(row)) + "\n" for row in rows]
    atomic_write_text(Path(path), "".join(lines))
    return len(lines)


def _provider_message(message: Message) -> dict[str, Any]:
    data = message.model_dump(mode="json", exclude_none=True, exclude={"weight"})
    if message.weight == 0:
        data["weight"] = 0
    return data


def to_provider_dict(row: TrainingRow) -> dict[str, Any]:
    """The row in OpenAI chat format: ``messages``, plus ``tools`` when any are declared.

    Our ``id``, ``source`` and ``schema_version`` are dropped, as are unset fields. ``weight``
    is kept only when it is ``0`` (do not train on that turn), since training is the default.
    """
    data: dict[str, Any] = {"messages": [_provider_message(m) for m in row.messages]}
    if row.tools:
        data["tools"] = [tool.model_dump(mode="json", exclude_none=True) for tool in row.tools]
    return data
