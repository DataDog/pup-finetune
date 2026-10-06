# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Why a row was left out of a dataset, as recorded in ``excluded.jsonl``.

A row is excluded either because it breaks the canonical schema (a
:class:`~pup_finetune.data.schema.RowErrorCode`), or for a reason the schema cannot see (an
:class:`ExclusionCode`): the source span could not be converted faithfully, or the row is
valid but does not belong in the dataset. Both sets of values are stable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pup_finetune.data.schema import RowError, RowErrorCode, TrainingRow


class ExclusionCode(StrEnum):
    """Reasons a row is excluded that are not schema violations. The values are stable."""

    DUPLICATE_CONVERSATION = "duplicate_conversation"
    """The same conversation (by content hash) is already in the dataset under another id."""
    NON_TEXT_CONTENT = "non_text_content"
    """A source message has a content part that is not text, such as an image."""
    UNSTRUCTURED_TOOL_MESSAGE = "unstructured_tool_message"
    """A tool message with no structured tool call or result to pair with, as sent by apps
    that write tool calls into the assistant's text instead of the tool-call fields."""


ExclusionReason = RowErrorCode | ExclusionCode
"""Any stable exclusion code."""


@dataclass(frozen=True, slots=True)
class Exclusion:
    """One excluded row: why, and whatever identifies it.

    ``line_sha256`` is the sha256 of the input line an invalid row was read from (without
    surrounding whitespace). It tells apart rows that have no id but share a location and a
    diagnostic, such as the same line of two files with the same name.
    """

    code: ExclusionReason
    detail: str
    row_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    line_sha256: str | None = None

    @classmethod
    def for_row(cls, row: TrainingRow, code: ExclusionCode, detail: str) -> Exclusion:
        return cls(code, detail, row.id, row.source.trace_id, row.source.span_id)

    @classmethod
    def for_source(
        cls,
        code: ExclusionReason,
        detail: str,
        *,
        row_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> Exclusion:
        """An exclusion for a candidate row that was never valid, such as a converted span.

        As in :meth:`for_invalid`, ids UTF-8 cannot encode are dropped and the detail is
        escaped, so the record can always be written.
        """
        return cls(
            code,
            detail.encode("utf-8", "backslashreplace").decode("utf-8"),
            _text(row_id),
            _text(trace_id),
            _text(span_id),
        )

    @classmethod
    def for_invalid(
        cls, data: Mapping[str, Any], error: RowError, *, where: str, line: bytes
    ) -> Exclusion:
        """An exclusion for ``data``, read from ``line``, that failed validation.

        Ids are taken when they are strings UTF-8 can encode: one holding a lone surrogate
        (from a ``\\ud800`` escape) is dropped, so the record can still be written.
        """
        source = data.get("source")
        source = source if isinstance(source, Mapping) else {}
        return cls(
            error.code,
            # A file name can hold undecodable bytes as surrogates; escape them.
            f"{where}: {error.detail}".encode("utf-8", "backslashreplace").decode("utf-8"),
            _text(data.get("id")),
            _text(source.get("trace_id")),
            _text(source.get("span_id")),
            hashlib.sha256(line.strip()).hexdigest(),
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code.value, "detail": self.detail}
        for name in ("row_id", "trace_id", "span_id", "line_sha256"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return value
