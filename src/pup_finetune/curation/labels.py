# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Recording keep/drop labels on a dataset's accepted rows.

The :class:`Label` record and the ``labels.jsonl`` format live in
:mod:`pup_finetune.data.labels` (the dataset directory reads them); they are re-exported
here. :func:`apply_labels` checks a batch of labels against a dataset and records it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.labels import (
    Label,
    LabelSource,
    latest_labels,
    parse_labels,
    read_labels,
)
from pup_finetune.errors import UsageError
from pup_finetune.storage import DirectoryLock

__all__ = [
    "Label",
    "LabelReport",
    "LabelSource",
    "apply_labels",
    "latest_labels",
    "parse_labels",
    "read_labels",
]

UNKNOWN_IDS_SHOWN = 10
"""How many unknown row ids an error lists."""


@dataclass(frozen=True, slots=True)
class LabelReport:
    """What one :func:`apply_labels` did, and the dataset summary afterwards."""

    applied: int
    """Labels appended to ``labels.jsonl``."""
    unchanged: int
    """Labels not appended: a later label of the same id in this batch replaced it, or the
    row already had the same decision, reason and source."""
    kept: int
    """Rows in this batch whose decision is now keep."""
    dropped: int
    """Rows in this batch whose decision is now drop."""
    summary: dict[str, Any]


def apply_labels(
    dataset: DatasetDir, labels: Iterable[Label], *, locked: bool = False
) -> LabelReport:
    """Record ``labels`` on ``dataset``'s accepted rows and refresh its summary.

    Holds the dataset's directory lock, or with ``locked`` relies on the caller holding it
    (the lock is not reentrant), so a caller can read the labels in force and record new
    ones under one lock. Every label must name an accepted row, or
    :class:`UsageError` lists the unknown ids and nothing is written. Within ``labels`` and
    across earlier ones, the last label of an id wins: only the last label of each id in
    ``labels`` is considered, and it is not appended when it repeats the decision in force
    (same ``keep``, ``reason`` and ``by``). So applying the same file twice changes nothing,
    even when the file changes its mind about a row. The new labels are written in one
    atomic replace of ``labels.jsonl``; ``rows.jsonl`` is never touched.
    """
    batch = list(labels)
    dataset.settings()  # A UsageError when this is not a dataset directory.
    lock: contextlib.AbstractContextManager[object] = (
        contextlib.nullcontext() if locked else DirectoryLock(dataset.path)
    )
    with lock:
        row_ids = {row.id for row in dataset.rows()}
        _check_known(batch, row_ids)
        current = dataset.labels()
        # Reduce the batch to its last label per id first: appending a file's intermediate
        # labels (drop, then keep) would grow labels.jsonl on every replay of that file.
        decided = latest_labels(batch)
        appended = [
            label for label in decided.values() if not _same_decision(current.get(label.id), label)
        ]
        summary = dataset.append_labels(appended)
    kept = sum(1 for label in decided.values() if label.keep)
    return LabelReport(
        applied=len(appended),
        unchanged=len(batch) - len(appended),
        kept=kept,
        dropped=len(decided) - kept,
        summary=summary,
    )


def _check_known(batch: list[Label], row_ids: set[str]) -> None:
    unknown = list(dict.fromkeys(label.id for label in batch if label.id not in row_ids))
    if not unknown:
        return
    shown = ", ".join(repr(row_id) for row_id in unknown[:UNKNOWN_IDS_SHOWN])
    more = len(unknown) - UNKNOWN_IDS_SHOWN
    suffix = f" and {more} more" if more > 0 else ""
    raise UsageError(
        f"{len(unknown)} labeled ids are not accepted rows of the dataset: {shown}{suffix}",
        hint=(
            "label only ids from rows.jsonl (see `pup finetune dataset sample DIR`); "
            "nothing was written"
        ),
    )


def _same_decision(current: Label | None, label: Label) -> bool:
    return current is not None and (current.keep, current.reason, current.by) == (
        label.keep,
        label.reason,
        label.by,
    )
