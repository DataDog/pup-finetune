# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Where a dataset directory stands in its pipeline: its rows, its triage and its push.

:func:`pipeline_state` reads only what the directory already records (``summary.json``'s
counts, ``labels.jsonl``, ``triage.json`` with ``judgments.jsonl``, and ``push.json``), so
every dataset command can report it as its ``pipeline`` block and choose what to suggest
next from it. A file it cannot read makes that part ``unreadable``, with a warning naming
the file and the problem (:attr:`Pipeline.warnings`), rather than a guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pup_finetune.curation.triage import TriageStatus, triage_status
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import read_push_state
from pup_finetune.errors import StateError

__all__ = ["Pipeline", "PushStatus", "pipeline_state"]

PushStatus = Literal["not_pushed", "pushed", "pushed_with_new_rows", "unreadable"]
"""``not_pushed`` without a ``push.json``; ``pushed`` when it lists every kept row;
``pushed_with_new_rows`` when kept rows are not pushed yet (``dataset push DIR`` sends
them); ``unreadable`` when ``push.json`` cannot be read."""


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A dataset's pipeline state; :meth:`to_json` is the ``pipeline`` block."""

    accepted: int
    kept: int
    dropped: int
    unlabeled: int
    triage: TriageStatus
    push: PushStatus
    pending: int | None
    """Kept rows not pushed yet; ``None`` when ``push.json`` cannot be read."""
    dataset_name: str | None = None
    """The dataset ``push.json`` names, when there is one."""
    warnings: tuple[str, ...] = ()
    """Why a part is ``unreadable``: the file and the problem, and what to do about it."""

    @property
    def labeled(self) -> int:
        return self.accepted - self.unlabeled

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": {
                "accepted": self.accepted,
                "kept": self.kept,
                "dropped": self.dropped,
                "unlabeled": self.unlabeled,
            },
            "triage": self.triage,
            "push": {"status": self.push, "pending": self.pending},
        }


def pipeline_state(dataset: DatasetDir, summary: Mapping[str, Any] | None = None) -> Pipeline:
    """``dataset``'s :class:`Pipeline`; ``summary`` is its :meth:`DatasetDir.summary` when
    the caller has it already."""
    counts = summary if summary is not None else dataset.summary()
    accepted = int(counts["accepted"])
    kept = int(counts["kept"])
    warnings: list[str] = []
    try:
        state = read_push_state(dataset)
    except StateError as exc:
        push: PushStatus = "unreadable"
        pending: int | None = None
        name = None
        warnings.append(
            f"the push receipt cannot be read ({exc.message}), so which kept rows were "
            "pushed is unknown; fix the file, or move it aside and push to a new --name"
        )
    else:
        if state is None:
            push, pending, name = "not_pushed", kept, None
        else:
            name = state.dataset_name
            pending = sum(1 for row in dataset.kept_rows() if row.id not in state.record_ids)
            push = "pushed_with_new_rows" if pending or state.in_flight else "pushed"
    triage, triage_warning = triage_status(dataset)
    if triage_warning is not None:
        warnings.insert(0, triage_warning)
    return Pipeline(
        accepted=accepted,
        kept=kept,
        dropped=int(counts["dropped"]),
        unlabeled=accepted - int(counts["labeled"]),
        triage=triage,
        push=push,
        pending=pending,
        dataset_name=name,
        warnings=tuple(warnings),
    )
