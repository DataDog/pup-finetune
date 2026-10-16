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

:func:`require_in_datadog` is the gate of ``train`` and ``benchmark``: they read a dataset
whose every kept row is in Datadog as it is now, and :meth:`Pipeline.push_step` is the one
advice on how to get there, for the gate and for ``next_steps`` alike.
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
from pup_finetune.data.split import SplitName
from pup_finetune.display import display_path
from pup_finetune.errors import DirectoryUsageError, StateError
from pup_finetune.next_steps import command, step, template

__all__ = ["Pipeline", "PushStatus", "pipeline_state", "require_in_datadog"]

PushStatus = Literal[
    "not_pushed", "pushed", "pushed_with_new_rows", "pushed_with_changed_rows", "unreadable"
]
"""``not_pushed`` without a ``push.json``; ``pushed`` when it lists every kept row as it is
now; ``pushed_with_new_rows`` when kept rows are not pushed yet (``dataset push DIR`` sends
them); ``pushed_with_changed_rows`` when kept rows changed after they were pushed (a push
only adds records, so only a push to a new dataset holds them as they are); ``unreadable``
when ``push.json`` cannot be read."""


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
            "push": {"status": self.push, "pending": self.pending, "changed": self.changed},
        }

    def push_step(self, directory: Path) -> dict[str, str] | None:
        """The step that puts every kept row in Datadog as it is now, or ``None`` when they
        all are. ``directory`` is the dataset directory as the user typed it."""
        if self.push == "not_pushed":
            return step(
                template("dataset", "push", directory, "--name", "<NAME>"),
                f"push the {self.kept} kept rows to Datadog: train and benchmark read a "
                "dataset that is in Datadog, and an evaluation publishes its experiments on "
                "it (a remote write; it previews first)",
            )
        if self.push == "pushed_with_new_rows":
            what = (
                f"push the {self.pending} kept rows not pushed yet to"
                if self.pending
                else "finish the interrupted push to"
            )
            return step(
                command("dataset", "push", directory),
                f"{what} dataset {self.dataset_name!r}, which {PUSH_FILE} names (a remote "
                "write; it previews first)",
            )
        if self.push == "pushed_with_changed_rows":
            return step(
                template("dataset", "push", directory, "--name", "<NEW_NAME>"),
                f"{self.changed} kept rows changed after they were pushed, and a push only "
                f"adds records, so dataset {self.dataset_name!r} holds them as they were: "
                f"move {PUSH_FILE} aside, then push every kept row to a new dataset (a remote "
                "write; it previews first)",
            )
        if self.push == "unreadable":
            return step(
                command("dataset", "show", directory),
                f"{PUSH_FILE} cannot be read, so whether the dataset was pushed is unknown; "
                "show reports why",
            )
        return None


def pipeline_state(
    dataset: DatasetDir,
    summary: Mapping[str, Any] | None = None,
    *,
    by_split: Mapping[SplitName, Sequence[TrainingRow]] | None = None,
) -> Pipeline:
    """``dataset``'s :class:`Pipeline`; ``summary`` is its :meth:`DatasetDir.summary` and
    ``by_split`` its :meth:`DatasetDir.rows_by_split` when the caller has them already."""
    counts = summary if summary is not None else dataset.summary()
    accepted = int(counts["accepted"])
    kept = int(counts["kept"])
    warnings: list[str] = []
    try:
        state = read_push_state(dataset)
    except StateError as exc:
        push: PushStatus = "unreadable"
        pending: int | None = None
        changed: int | None = None
        name = None
        warnings.append(
            f"the push receipt cannot be read ({exc.message}), so which kept rows were "
            "pushed is unknown; fix the file, or move it aside and push to a new --name"
        )
    else:
        if state is None:
            push, pending, changed, name = "not_pushed", kept, 0, None
        else:
            name = state.dataset_name
            unpushed, edited = compare_receipt(
                state, by_split if by_split is not None else dataset.rows_by_split()
            )
            pending, changed = len(unpushed), len(edited)
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
        dataset_name=name,
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
    the directory as the user typed it. What trains is then what the Agent Observability
    dataset holds, so a run and its evaluation trace back to records in Datadog; a directory
    ``dataset fetch`` made has the receipt of the dataset it read, so it passes. An
    ``offline`` command (the fake provider) never leaves the machine and reads any dataset.
    """
    if offline:
        return
    state = pipeline_state(dataset, by_split=by_split)
    advice = state.push_step(directory)
    if advice is None:
        return
    why = {
        "not_pushed": f"{display_path(dataset.path)} was never pushed",
        "pushed_with_new_rows": f"{state.pending} of its kept rows are not in dataset "
        f"{state.dataset_name!r} yet"
        if state.pending
        else f"its last push to dataset {state.dataset_name!r} did not finish",
        "pushed_with_changed_rows": f"{state.changed} of its kept rows changed after they "
        f"were pushed to dataset {state.dataset_name!r}",
        "unreadable": f"whether {display_path(dataset.path)} was pushed is unknown: "
        f"{PUSH_FILE} cannot be read",
    }[state.push]
    raise DirectoryUsageError(
        f"{command_name} reads a dataset that is in Datadog, and {why}",
        hint=f"{advice['why']}: `{advice['command']}`; or train on a dataset already in "
        "Datadog: `pup finetune dataset fetch <DIR> --project <PROJECT> --name <NAME>`",
    )
