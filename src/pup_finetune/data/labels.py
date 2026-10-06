# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Keep/drop labels: the :class:`Label` record and its JSONL file format.

A :class:`Label` says whether one accepted row is kept for training, and why. A dataset's
``labels.jsonl`` is a log of them in which the last label of each row id wins; rows
themselves (``rows.jsonl``) are never changed. A row whose latest label is ``keep: false``
is left out of every training input and split export (see
:meth:`~pup_finetune.data.dataset_dir.DatasetDir.kept_rows`); a row with no label is kept.
Recording labels is :func:`pup_finetune.curation.labels.apply_labels`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from pup_finetune.data.schema import lone_surrogate
from pup_finetune.errors import FinetuneError, StateError
from pup_finetune.storage import parse_jsonl_line

LabelSource = Literal["agent", "human", "judge"]
"""Who decided: the coding agent, a person, or a judge model."""

TOP_DROP_REASONS = 10
"""How many of the most frequent drop reasons the dataset summary lists."""


class Label(BaseModel):
    """Keep or drop one accepted row. ``reason`` is required to drop it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(min_length=1)]
    """The row id, as in ``rows.jsonl``."""
    keep: StrictBool
    reason: str = ""
    by: LabelSource = "agent"
    at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    """When the decision was made; stored in UTC."""

    @field_validator("reason")
    @classmethod
    def _reason_is_encodable(cls, value: str) -> str:
        # JSON's \ud800 escape decodes to a lone surrogate, which UTF-8 cannot encode. Rejected
        # here, a label file reports its line instead of failing when the label is written.
        if (surrogate := lone_surrogate(value)) is not None:
            raise ValueError(f"has a lone surrogate ({surrogate}); UTF-8 cannot encode it")
        return value

    @field_validator("at")
    @classmethod
    def _at_is_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _a_drop_has_a_reason(self) -> Self:
        if not self.keep and not self.reason.strip():
            raise ValueError("a reason is required when keep is false")
        return self

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def parse_labels(
    path: Path,
    *,
    defaults: Mapping[str, Any] | None = None,
    error: type[FinetuneError] = StateError,
    hint: str | None = None,
) -> list[Label]:
    """Every label in the JSONL file at ``path``, in file order; blank lines are skipped.

    ``defaults`` fill in fields a line leaves out. A line that is not a JSON object or not a
    valid label raises ``error`` (with ``hint``) naming its line number.
    """
    path = Path(path)
    labels: list[Label] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record, problem = parse_jsonl_line(line)
            if problem is None:
                try:
                    labels.append(Label.model_validate({**(defaults or {}), **record}))
                    continue
                except ValidationError as exc:
                    problem = f"invalid label: {_label_problem(exc)}"
            raise error(f"{path}:{line_number}: {problem}", hint=hint)
    return labels


def read_labels(path: Path) -> list[Label]:
    """Every label in a ``labels.jsonl`` file, in the order recorded; none when it is missing.

    A line that is not a valid label raises :class:`StateError` naming its line number.
    """
    path = Path(path)
    return parse_labels(path) if path.exists() else []


def latest_labels(labels: Iterable[Label]) -> dict[str, Label]:
    """The last label of each row id: the decision in force."""
    return {label.id: label for label in labels}


def label_counts(latest: Mapping[str, Label], row_ids: Iterable[str]) -> dict[str, Any]:
    """How many of ``row_ids`` are labeled and dropped, and the top drop reasons.

    Labels of ids not in ``row_ids`` (rows a later pull retired) are not counted. Unlabeled
    rows count as kept.
    """
    labeled = [latest[row_id] for row_id in dict.fromkeys(row_ids) if row_id in latest]
    drops = [label for label in labeled if not label.keep]
    reasons = Counter(label.reason.strip() for label in drops)
    ranked = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    return {
        "labeled": len(labeled),
        "dropped": len(drops),
        "drop_reasons": [
            {"reason": reason, "count": count} for reason, count in ranked[:TOP_DROP_REASONS]
        ],
    }


def _label_problem(exc: ValidationError) -> str:
    """The first problem pydantic found in a label, as one line."""
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    message = str(first["msg"]).removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
