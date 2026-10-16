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

:func:`require_in_datadog` is the gate of ``train`` and ``benchmark``: every kept row they
read must be in Datadog as it is now. :meth:`Pipeline.push_gap` says what is missing and the
step that fixes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pup_finetune.curation.triage import TriageStatus, triage_status
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import PUSH_FILE, compare_receipt, read_push_state
from pup_finetune.data.schema import TrainingRow
from pup_finetune.data.split import SPLITS, SplitName
from pup_finetune.display import counted, display_path
from pup_finetune.errors import DirectoryUsageError, StateError
from pup_finetune.next_steps import command, step, template

__all__ = ["Pipeline", "PushGap", "PushStatus", "pipeline_state", "require_in_datadog"]

PushStatus = Literal[
    "not_pushed", "pushed", "pushed_with_new_rows", "pushed_with_changed_rows", "unreadable"
]
"""``not_pushed`` without a ``push.json``; ``pushed`` when it lists every kept row as it is
now; ``pushed_with_new_rows`` when kept rows are not pushed yet (``dataset push DIR`` sends
them); ``pushed_with_changed_rows`` when kept rows changed after they were pushed; ``unreadable``
when ``push.json`` cannot be read."""

SHOWN_CHANGED_IDS = 5
"""How many ids of rows changed after the push the gate names."""

FETCH_ANY = template("dataset", "fetch", "<DIR>", "--project", "<PROJECT>", "--name", "<NAME>")
"""The fetch of a dataset already in Datadog, which ``train`` and ``benchmark`` read too."""


@dataclass(frozen=True, slots=True)
class PushGap:
    """Why a dataset's kept rows are not all in Datadog as they are now, and the step that
    gets them there (:meth:`Pipeline.push_gap`)."""

    reason: str
    """What is missing, as it reads after ``train reads a dataset that is in Datadog, and``."""
    step: dict[str, str]
    """The ``next_steps`` entry that fixes it."""
    alternative: str | None = FETCH_ANY
    """Another way to a dataset in Datadog, ``None`` when :attr:`step` is it already."""


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
    changed: int | None = 0
    """Kept rows whose record changed since it was pushed; ``None`` when ``push.json``
    cannot be read."""
    changed_ids: tuple[str, ...] = ()
    """The ids of those rows, sorted."""
    stale: int = 0
    """Records ``push.json`` lists for rows no longer kept (dropped or retired since the
    push), which the dataset still holds."""
    dataset_name: str | None = None
    """The dataset ``push.json`` names, when there is one."""
    project: str | None = None
    """The project of that dataset."""
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
            "push": {"status": self.push, "pending": self.pending, "changed": self.changed},
        }

    def push_gap(self, directory: Path, command_name: str = "train") -> PushGap | None:
        """What keeps the kept rows from all being in Datadog as they are now, and the step
        that fixes it; ``None`` when they all are, or there are none. ``directory`` is the
        dataset directory as the user typed it, and ``command_name`` (``train``,
        ``benchmark``) what reads it."""
        name = self.dataset_name
        match self.push:
            case "pushed":
                return None
            case "not_pushed" if not self.kept:
                return None
            case "not_pushed":
                return PushGap(
                    f"{display_path(directory)} was never pushed",
                    step(
                        template("dataset", "push", directory, "--name", "<NAME>"),
                        f"push the {counted(self.kept, 'kept row')} to Datadog: train and "
                        "benchmark read a dataset that is in Datadog, and an evaluation "
                        "publishes its experiments on it (a remote write; it previews first)",
                    ),
                )
            case "pushed_with_new_rows" if self.pending:
                pending = self.pending
                return PushGap(
                    f"{pending} of its kept rows {'is' if pending == 1 else 'are'} not in "
                    f"dataset {name!r} yet",
                    step(
                        command("dataset", "push", directory),
                        f"push the {counted(pending, 'kept row')} not pushed yet to dataset "
                        f"{name!r}, which {PUSH_FILE} names (a remote write; it previews first)",
                    ),
                )
            case "pushed_with_new_rows":
                return PushGap(
                    f"its last push to dataset {name!r} did not finish",
                    step(
                        command("dataset", "push", directory),
                        f"finish the interrupted push to dataset {name!r}, which {PUSH_FILE} "
                        "names (a remote write; it previews first)",
                    ),
                )
            case "pushed_with_changed_rows":
                ids = list(self.changed_ids)
                shown = ", ".join(ids[:SHOWN_CHANGED_IDS])
                more = (
                    f" and {len(ids) - SHOWN_CHANGED_IDS} more"
                    if len(ids) > SHOWN_CHANGED_IDS
                    else ""
                )
                stale = (
                    f"; the fetch also brings back the {counted(self.stale, 'row')} dropped or "
                    "retired since the push, which you drop again with `dataset label` after "
                    "the fetch"
                    if self.stale
                    else ""
                )
                return PushGap(
                    f"{self.changed} of its kept rows changed after the push to dataset "
                    f"{name!r} ({'row' if len(ids) == 1 else 'rows'} {shown}{more})",
                    step(
                        template(
                            "dataset",
                            "fetch",
                            "<NEW_DIR>",
                            "--project",
                            self.project,
                            "--name",
                            name,
                        ),
                        f"a push only adds records, so dataset {name!r} holds those rows as "
                        f"they were pushed: fetch it into a new directory to {command_name} on "
                        f"what Datadog holds{stale} (to {command_name} on the rows as they are "
                        f"now, move {PUSH_FILE} aside and push them to a new --name)",
                    ),
                    alternative=None,
                )
            case "unreadable":
                return PushGap(
                    f"whether {display_path(directory)} was pushed is unknown: {PUSH_FILE} "
                    "cannot be read",
                    step(
                        command("dataset", "show", directory),
                        f"{PUSH_FILE} cannot be read, so whether the dataset was pushed is "
                        "unknown; show reports why",
                    ),
                )


def pipeline_state(
    dataset: DatasetDir,
    summary: Mapping[str, Any] | None = None,
    *,
    by_split: Mapping[SplitName, Sequence[TrainingRow]] | None = None,
) -> Pipeline:
    """``dataset``'s :class:`Pipeline`; ``summary`` is its :meth:`DatasetDir.summary` and
    ``by_split`` its :meth:`DatasetDir.rows_by_split` when the caller has them already. The
    kept, pending, changed and stale counts read ``by_split`` when it is given."""
    counts = summary if summary is not None else dataset.summary()
    accepted = int(counts["accepted"])
    kept = (
        int(counts["kept"])
        if by_split is None
        else sum(len(by_split.get(split, ())) for split in SPLITS)
    )
    warnings: list[str] = []
    changed_ids: tuple[str, ...] = ()
    stale = 0
    name = project = None
    try:
        state = read_push_state(dataset)
    except StateError as exc:
        push: PushStatus = "unreadable"
        pending: int | None = None
        changed: int | None = None
        warnings.append(
            f"the push receipt cannot be read ({exc.message}), so which kept rows were "
            "pushed is unknown; fix the file, or move it aside and push to a new --name"
        )
    else:
        if state is None:
            push, pending, changed = "not_pushed", kept, 0
        else:
            name, project = state.dataset_name, state.project
            rows = by_split if by_split is not None else dataset.rows_by_split()
            comparison = compare_receipt(state, rows)
            pending, changed = len(comparison.pending), len(comparison.changed)
            changed_ids, stale = tuple(sorted(comparison.changed)), len(comparison.stale)
            push = (
                "pushed_with_changed_rows"
                if changed
                else "pushed_with_new_rows"
                if pending or state.in_flight
                else "pushed"
            )
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
        changed=changed,
        changed_ids=changed_ids,
        stale=stale,
        dataset_name=name,
        project=project,
        warnings=tuple(warnings),
    )


def require_in_datadog(
    dataset: DatasetDir,
    by_split: Mapping[SplitName, Sequence[TrainingRow]],
    command_name: str,
    *,
    directory: Path,
    offline: bool,
) -> None:
    """Raise :class:`~pup_finetune.errors.DirectoryUsageError` unless every kept row of
    ``dataset`` in ``by_split`` (the rows the command goes on to read) is in Datadog as it
    is now: its push receipt lists each with the record it makes, and no push is in flight.

    ``command_name`` (``train``, ``benchmark``) names who refuses it, and ``directory`` is
    the directory as the user typed it. A directory ``dataset fetch`` made has the receipt
    of the dataset it read, so it passes. An ``offline`` command (the fake provider) reads
    any dataset.
    """
    if offline:
        return
    gap = pipeline_state(dataset, by_split=by_split).push_gap(directory, command_name)
    if gap is None:
        return
    hint = f"{gap.step['why']}: `{gap.step['command']}`"
    if gap.alternative is not None:
        hint += f"; or {command_name} on a dataset already in Datadog: `{gap.alternative}`"
    raise DirectoryUsageError(
        f"{command_name} reads a dataset that is in Datadog, and {gap.reason}", hint=hint
    )
