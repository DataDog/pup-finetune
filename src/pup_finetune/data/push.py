# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Push a dataset directory's kept rows to an Agent Observability dataset.

:func:`plan_push` works out, from local files only, what a push would send: what ``dataset
push`` previews. :func:`push_dataset` runs it: it finds or creates the project, creates the
dataset, and adds one record per kept row (:func:`dataset_record`) in batches through
:class:`~pup_finetune.datadog.datasets.LLMObsDatasets`. Rows labeled ``keep: false`` are never
sent.

The receipt, ``push.json`` in the dataset directory, records the project and dataset and the
record id of every row pushed, and is saved after every batch, so:

- pushing again sends only the rows it does not list: nothing for an unchanged dataset, and
  only the new rows after more were added;
- an interrupted push resumes where it stopped. Before each batch is sent its row ids are
  saved as ``in_flight``; a push that finds them (the batch's response was lost: a timeout,
  an unreadable answer, the process stopped) first lists the dataset's records, adopts those
  whose ``metadata.row_id`` is in flight and sends only the rest again, so no row is added
  twice. A batch whose call timed out may still be being written, so until
  :data:`IN_FLIGHT_SETTLE` has passed since it was sent (``in_flight_at``) the rows not found
  are not sent again: the push adopts those found and stops, saying when to push again.

A receipt ties the directory to one dataset: pushing with another ``--name`` or ``--project``
is an error. The dataset's description carries the receipt's ``push_id``, so a push that
stopped before it learned the dataset id adopts only the dataset it created, never another
of the same name. Rows the receipt lists that are no longer kept (labeled drop after the
push) are reported as stale records; they are not deleted from the remote dataset.

The receipt also keeps a digest of the record pushed for each row (``record_digests``), so a
kept row that changed after it was pushed, such as one a resumed ``--reasoning preserve``
pull filled reasoning in for under the same id, is reported as a changed record: a push only
adds records, so its remote record keeps the content pushed before. The digests of a batch
are saved with it as ``in_flight_digests``, so a record adopted from an interrupted batch
gets the digest of the record sent, and a row changed since it was sent is reported too. A
row without a digest (pushed by a receipt that predates them) is never reported as changed.
"""

from __future__ import annotations

import hashlib
import secrets
import shlex
import signal
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.jsonl import to_provider_dict
from pup_finetune.data.schema import TrainingRow
from pup_finetune.data.split import SPLITS, SplitName
from pup_finetune.datadog.datasets import (
    RECORDS_BATCH_SIZE,
    LLMObsDatasets,
    RecordsListTruncated,
)
from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import (
    ExternalError,
    Interrupted,
    SettingsConflict,
    StateError,
    UsageError,
)
from pup_finetune.storage import (
    DirectoryLock,
    atomic_write_json,
    canonical_json,
    read_json,
    settings_hash,
)

PUSH_FILE = "push.json"
SOURCE_TAG = "source:pup-finetune"
PROJECT_DESCRIPTION = "Fine-tuning datasets pushed by pup-finetune."
DATASET_DESCRIPTION = "Curated fine-tuning conversations pushed by pup-finetune."
IN_FLIGHT_SETTLE = timedelta(minutes=10)
"""How long after a batch was sent its missing rows are not sent again: a records-add call
that timed out may still be adding them."""
SHOWN_ROW_IDS = 10
"""How many in-flight row ids an error names; ``push.json`` lists them all."""

_RESUME_HINT = (
    f"push again to resume: {PUSH_FILE} lists the batch as in_flight, and the next push "
    "adopts the records of it the dataset already holds before sending the rest, which it "
    f"sends only from {int(IN_FLIGHT_SETTLE.total_seconds() // 60)} minutes after the batch "
    "was sent, as the batch may still be being added"
)

_LOOK_UP_HINT = (
    f"pushing again lists the same records, so {PUSH_FILE} keeps the rows as in_flight; look "
    "them up in the dataset by metadata.row_id, add the ids of the records found to "
    f"record_ids in {PUSH_FILE}, remove every row from in_flight, and push again to send only "
    "the rows that are missing"
)

Progress = Callable[[str], None]
Clock = Callable[[], datetime]
"""The current time, timezone-aware."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def dataset_record(row: TrainingRow, split: SplitName) -> dict[str, Any]:
    """The Agent Observability record for ``row``: what the final assistant turn answers.

    ``input`` holds the messages before the final assistant message and the declared tools,
    in OpenAI chat form; ``expected_output`` is that final message.
    """
    provider = to_provider_dict(row)
    messages: list[dict[str, Any]] = provider["messages"]
    source = row.source
    return {
        "input": {"messages": messages[:-1], "tools": provider.get("tools", [])},
        "expected_output": messages[-1],
        "metadata": {
            "row_id": row.id,
            "split": split,
            "trace_id": source.trace_id,
            "span_id": source.span_id,
            "ml_app": source.ml_app,
            "source_model": source.model_name,
            "schema_version": row.schema_version,
        },
        "tags": [f"split:{split}", SOURCE_TAG],
    }


def record_digest(record: Mapping[str, Any]) -> str:
    """The sha256 hex digest of ``record`` (a :func:`dataset_record`) as canonical JSON."""
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def dataset_description(push_id: str) -> str:
    """The description of the dataset a receipt with ``push_id`` creates."""
    return f"{DATASET_DESCRIPTION} Push id: {push_id}."


@dataclass(frozen=True, slots=True)
class PushState:
    """``push.json``: the dataset a directory was pushed to and the record of each row."""

    project: str
    project_id: str
    dataset_name: str
    settings_hash: str
    push_id: str
    """Random; written into the description of the dataset this receipt creates."""
    dataset_id: str | None = None
    """``None`` until the dataset is created; a push that stopped before then adopts it."""
    dataset_version: int | None = None
    """The dataset version the last ``records-add`` reported; ``None`` when unknown."""
    record_ids: Mapping[str, str] = field(default_factory=dict)
    """The record id per row id, for every row pushed."""
    record_digests: Mapping[str, str] = field(default_factory=dict)
    """The :func:`record_digest` of the record pushed per row id; a row pushed by a receipt
    that predates digests has none."""
    in_flight: tuple[str, ...] = ()
    """Row ids of a batch sent whose record ids are not known yet; empty between batches."""
    in_flight_digests: Mapping[str, str] = field(default_factory=dict)
    """The :func:`record_digest` of each record in flight, as sent: a record adopted later
    gets this digest, so a row changed since is reported as a changed record. Empty between
    batches, and in a receipt written before it was recorded."""
    in_flight_at: datetime | None = None
    """When the batch in flight was sent (UTC); ``None`` between batches, and in a receipt
    written before it was recorded, whose batch is taken as settled."""
    pushed_at: str | None = None
    """When records were last added (ISO 8601, UTC)."""

    def to_json(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "push_id": self.push_id,
            "record_ids": dict(self.record_ids),
            "record_digests": dict(self.record_digests),
            "in_flight": list(self.in_flight),
            "in_flight_digests": dict(self.in_flight_digests),
            "in_flight_at": (
                None
                if self.in_flight_at is None
                else self.in_flight_at.isoformat(timespec="seconds")
            ),
            "pushed_at": self.pushed_at,
            "settings_hash": self.settings_hash,
        }

    @classmethod
    def from_json(cls, data: Any, *, where: str) -> PushState:
        if not isinstance(data, dict):
            raise StateError(f"{where}: expected a JSON object")

        def optional(key: str) -> str | None:
            value = data.get(key)
            if value is None or (isinstance(value, str) and value):
                return value
            raise StateError(f"{where}: {key} must be a non-empty string")

        def required(key: str) -> str:
            value = optional(key)
            if value is None:
                raise StateError(f"{where}: {key} must be a non-empty string")
            return value

        record_ids = data.get("record_ids")
        if not isinstance(record_ids, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in record_ids.items()
        ):
            raise StateError(f"{where}: record_ids must map row ids to record ids")
        # Missing in a receipt written before digests were recorded: every row's is unknown.
        record_digests = data.get("record_digests", {})
        if not isinstance(record_digests, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in record_digests.items()
        ):
            raise StateError(f"{where}: record_digests must map row ids to record digests")
        in_flight = data.get("in_flight", [])
        if not isinstance(in_flight, list) or not all(isinstance(i, str) for i in in_flight):
            raise StateError(f"{where}: in_flight must be a list of row ids")
        # Missing in a receipt written before they were recorded.
        in_flight_digests = data.get("in_flight_digests", {})
        if not isinstance(in_flight_digests, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in in_flight_digests.items()
        ):
            raise StateError(f"{where}: in_flight_digests must map row ids to record digests")
        in_flight_at = optional("in_flight_at")
        sent_at = _parse_time(in_flight_at) if in_flight_at is not None else None
        if in_flight_at is not None and sent_at is None:
            raise StateError(f"{where}: in_flight_at must be an ISO 8601 time with an offset")
        version = data.get("dataset_version")
        if version is not None and type(version) is not int:
            raise StateError(f"{where}: dataset_version must be an integer")
        return cls(
            project=required("project"),
            project_id=required("project_id"),
            dataset_name=required("dataset_name"),
            settings_hash=required("settings_hash"),
            push_id=required("push_id"),
            dataset_id=optional("dataset_id"),
            dataset_version=version,
            record_ids=dict(record_ids),
            record_digests=dict(record_digests),
            in_flight=tuple(in_flight),
            in_flight_digests=dict(in_flight_digests),
            in_flight_at=sent_at,
            pushed_at=optional("pushed_at"),
        )


def push_path(dataset: DatasetDir) -> Path:
    return dataset.path / PUSH_FILE


def read_push_state(dataset: DatasetDir) -> PushState | None:
    """The directory's ``push.json``, or ``None`` when it was never pushed."""
    path = push_path(dataset)
    try:
        data = read_json(path)
    except FileNotFoundError:
        return None
    return PushState.from_json(data, where=str(path))


@dataclass(frozen=True, slots=True)
class PendingRecord:
    """A kept row the remote dataset does not hold yet, and its record."""

    row_id: str
    split: SplitName
    record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReceiptComparison:
    """How a dataset's kept rows stand against its push receipt (:func:`compare_receipt`)."""

    pending: list[PendingRecord]
    """Kept rows the receipt lists no record for, train rows first, then validation and test."""
    changed: dict[str, str]
    """Kept rows whose record differs from the digest the receipt saved for it: row id to
    record id."""
    stale: dict[str, str]
    """Rows the receipt lists that are no longer kept: row id to record id."""


def compare_receipt(
    state: PushState | None, by_split: Mapping[SplitName, Sequence[TrainingRow]]
) -> ReceiptComparison:
    """The kept rows ``by_split`` against ``state`` (``None``: never pushed): those it lists
    no record for, those whose record changed since it was pushed (the record the row makes
    now differs from the digest the receipt saved for it), and the rows it lists that are no
    longer kept."""
    pushed = state.record_ids if state is not None else {}
    digests = state.record_digests if state is not None else {}
    pending: list[PendingRecord] = []
    changed: dict[str, str] = {}
    kept: set[str] = set()
    for split in SPLITS:
        for row in by_split.get(split, ()):
            kept.add(row.id)
            record = dataset_record(row, split)
            if row.id not in pushed:
                pending.append(PendingRecord(row.id, split, record))
            elif (sent := digests.get(row.id)) is not None and sent != record_digest(record):
                changed[row.id] = pushed[row.id]
    stale = {row_id: record_id for row_id, record_id in pushed.items() if row_id not in kept}
    return ReceiptComparison(pending, changed, stale)


@dataclass(frozen=True, slots=True)
class PushReceipt:
    """What :func:`push_dataset` did, and the receipt saved in ``push.json``."""

    state: PushState
    pushed: int
    """Records added by this push."""
    already_pushed: int
    """Kept rows an earlier push had sent, skipped this time."""
    records_by_split: dict[SplitName, int]
    """Kept rows the remote dataset holds, per split."""
    stale_records: dict[str, str]
    """Rows pushed earlier that are no longer kept: row id to record id."""
    changed_records: dict[str, str]
    """Kept rows that changed after they were pushed, whose remote records keep the content
    pushed before: row id to record id."""

    def to_json(self) -> dict[str, Any]:
        state = self.state
        return {
            "project": state.project,
            "project_id": state.project_id,
            "dataset_id": state.dataset_id,
            "dataset_name": state.dataset_name,
            "dataset_version": state.dataset_version,
            "pushed_at": state.pushed_at,
            "pushed": self.pushed,
            "already_pushed": self.already_pushed,
            "records": len(state.record_ids),
            "records_by_split": self.records_by_split,
            "stale_records": self.stale_records,
            "changed_records": self.changed_records,
        }


@dataclass(frozen=True, slots=True)
class PushPlan:
    """What a push of a dataset directory would send, worked out from local files only."""

    directory: Path
    project: str
    dataset_name: str
    state: PushState | None
    """The existing receipt, or ``None`` for a first push."""
    pending: list[PendingRecord]
    """The kept rows the receipt does not list, train rows first, then validation and test."""
    already_pushed: int
    """Kept rows the receipt already lists."""
    kept_by_split: dict[SplitName, int]
    stale_records: dict[str, str]
    """Rows the receipt lists that are no longer kept: row id to record id."""
    changed_records: dict[str, str]
    """Kept rows the receipt lists whose record differs from the one pushed (by its digest):
    row id to record id. They are not sent again: a push only adds records."""
    settings_hash: str

    @property
    def dataset_id(self) -> str | None:
        return self.state.dataset_id if self.state is not None else None

    @property
    def in_flight(self) -> tuple[str, ...]:
        """Row ids of a batch an interrupted push sent without learning its record ids."""
        return self.state.in_flight if self.state is not None else ()

    @property
    def empty(self) -> bool:
        """Nothing to push and nothing pushed: the directory has no kept rows to send."""
        return not self.pending and self.dataset_id is None

    @property
    def needs_remote(self) -> bool:
        """Whether a push would reach Datadog: rows to send, or an interrupted batch to check."""
        return bool(self.pending or self.in_flight)

    def pending_by_split(self) -> dict[SplitName, int]:
        counts: dict[SplitName, int] = dict.fromkeys(SPLITS, 0)
        for item in self.pending:
            counts[item.split] += 1
        return counts

    def require_rows(self) -> None:
        """Raise :class:`UsageError` when the plan is :attr:`empty`."""
        if self.empty:
            raise UsageError(
                f"{self.directory} has no kept rows to push",
                hint="add rows with `pup finetune dataset pull` or `dataset import`, or keep "
                "some with `pup finetune dataset label`",
            )

    def receipt(self, state: PushState | None = None) -> PushReceipt:
        """The receipt of this plan, pushed into ``state`` (by default the existing one)."""
        final = state if state is not None else self.state
        if final is None:
            raise ValueError("a plan without a receipt has pushed nothing")
        return PushReceipt(
            state=final,
            pushed=len(self.pending) if state is not None else 0,
            already_pushed=self.already_pushed,
            records_by_split=dict(self.kept_by_split),
            stale_records=self.stale_records,
            changed_records=self.changed_records,
        )


def plan_push(dataset: DatasetDir, *, project: str, name: str) -> PushPlan:
    """What pushing ``dataset`` to dataset ``name`` in ``project`` would send.

    Raises :class:`UsageError` when the receipt names another dataset or project, and
    :class:`StateError` when it was written for a dataset directory with other settings.
    """
    _require_name(project, "--project")
    _require_name(name, "--name")
    digest = settings_hash(dataset.settings())
    state = read_push_state(dataset)
    if state is not None:
        _check_state(dataset, state, project=project, name=name, digest=digest)
    by_split = dataset.rows_by_split()
    kept_by_split = {split: len(by_split[split]) for split in SPLITS}
    comparison = compare_receipt(state, by_split)
    return PushPlan(
        directory=dataset.path,
        project=project,
        dataset_name=name,
        state=state,
        pending=comparison.pending,
        already_pushed=sum(kept_by_split.values()) - len(comparison.pending),
        kept_by_split=kept_by_split,
        stale_records=comparison.stale,
        changed_records=comparison.changed,
        settings_hash=digest,
    )


class PushTerminated(Interrupted):
    """The push stopped because the process received SIGTERM; rerun it to resume."""


def push_dataset(
    dataset: DatasetDir,
    *,
    project: str,
    name: str,
    pup: Pup,
    batch_size: int = RECORDS_BATCH_SIZE,
    progress: Progress | None = None,
    clock: Clock = _utc_now,
) -> PushReceipt:
    """Push the kept rows of ``dataset`` to dataset ``name`` in ``project``.

    Holds the directory lock throughout, so labels cannot change mid-push and two pushes of
    one directory cannot race. SIGTERM stops it like Ctrl-C, removing the batch file. See the
    module docstring for resuming and the receipt. ``clock`` gives the time batches are sent
    and pushed at.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    dataset.settings()  # A clear UsageError when this is not a dataset directory.
    report = progress or (lambda _message: None)
    with _sigterm_raises(), DirectoryLock(dataset.path):
        plan = plan_push(dataset, project=project, name=name)
        plan.require_rows()
        if not plan.needs_remote:
            return plan.receipt()
        api = LLMObsDatasets(pup)
        if plan.state is not None and plan.state.in_flight:
            local = {item.row_id: item.record for item in plan.pending}
            _reconcile(dataset, plan.state, local, api, report, clock)
            plan = plan_push(dataset, project=project, name=name)
            if not plan.pending:
                return plan.receipt()
        state, dataset_id = _ensure_dataset(dataset, plan, api, report)
        state = _add_records(dataset, plan, state, dataset_id, api, batch_size, report, clock)
        return plan.receipt(state)


def _ensure_dataset(
    dataset: DatasetDir, plan: PushPlan, api: LLMObsDatasets, report: Progress
) -> tuple[PushState, str]:
    """The receipt and the dataset id, creating the project and dataset when needed."""
    state = plan.state
    if state is not None and state.dataset_id is not None:
        return state, state.dataset_id
    if state is None:
        project_id = api.ensure_project(plan.project, description=PROJECT_DESCRIPTION)
        if api.find_dataset(project_id, plan.dataset_name) is not None:
            raise UsageError(
                f"project {plan.project!r} already has a dataset named {plan.dataset_name!r}, "
                f"and {push_path(dataset)} does not record it",
                hint="pass a new --name: a dataset directory only adds records to a dataset "
                "it created",
            )
        state = PushState(
            project=plan.project,
            project_id=project_id,
            dataset_name=plan.dataset_name,
            settings_hash=plan.settings_hash,
            push_id=secrets.token_hex(8),
        )
        # Saved before the create, so a push that stops right after it adopts the dataset.
        _save(dataset, state)
        dataset_id = None
    else:
        found = api.find_dataset(state.project_id, plan.dataset_name)
        if found is not None and state.push_id not in found.description:
            raise UsageError(
                f"project {plan.project!r} has a dataset named {plan.dataset_name!r} that "
                f"{push_path(dataset)} did not create",
                hint=f"move {push_path(dataset)} aside and pass a new --name: a dataset "
                "directory only adds records to a dataset it created",
            )
        dataset_id = found.id if found is not None else None
    if dataset_id is None:
        report(f"creating dataset {plan.dataset_name!r} in project {plan.project!r}")
        dataset_id = api.create_dataset(
            state.project_id,
            plan.dataset_name,
            description=dataset_description(state.push_id),
            marker=state.push_id,
        )
    state = replace(state, dataset_id=dataset_id)
    _save(dataset, state)
    return state, dataset_id


def _add_records(
    dataset: DatasetDir,
    plan: PushPlan,
    state: PushState,
    dataset_id: str,
    api: LLMObsDatasets,
    batch_size: int,
    report: Progress,
    clock: Clock,
) -> PushState:
    total = len(plan.pending)
    for start in range(0, total, batch_size):
        batch = plan.pending[start : start + batch_size]
        row_ids = tuple(item.row_id for item in batch)
        # Saved before the call: if its answer is lost, the next push checks for these rows.
        digests = {item.row_id: record_digest(item.record) for item in batch}
        state = replace(
            state, in_flight=row_ids, in_flight_digests=digests, in_flight_at=_now(clock)
        )
        _save(dataset, state)
        try:
            added = api.add_records(state.project_id, dataset_id, [item.record for item in batch])
        except ExternalError as exc:
            raise ExternalError(
                f"adding a batch of {len(batch)} records to dataset {dataset_id} failed: "
                f"{exc.message}",
                source=exc.source,
                hint=f"{exc.hint}; {_RESUME_HINT}" if exc.hint else _RESUME_HINT,
            ) from exc
        record_ids = dict(state.record_ids)
        record_ids.update(zip(row_ids, added.record_ids, strict=True))
        state = replace(
            state,
            record_ids=record_ids,
            record_digests={**state.record_digests, **digests},
            in_flight=(),
            in_flight_digests={},
            in_flight_at=None,
            dataset_version=added.dataset_version,
            pushed_at=_now(clock).isoformat(),
        )
        _save(dataset, state)
        report(f"pushed {start + len(batch)}/{total} records")
    return state


def _reconcile(
    dataset: DatasetDir,
    state: PushState,
    local: Mapping[str, Mapping[str, Any]],
    api: LLMObsDatasets,
    report: Progress,
    clock: Clock,
) -> PushState:
    """Adopt the records of an interrupted batch (``in_flight``) that reached the dataset.

    A record adopted gets the digest of the record sent (``in_flight_digests``), so a row
    that changed since it was sent, such as one a resumed pull filled reasoning in for, is
    reported as a changed record. A receipt that predates those digests falls back to the
    digest of the row's current record in ``local`` (each kept row not pushed yet), and to
    none when the row is no longer kept.

    Raises :class:`StateError` when rows of it are missing within :data:`IN_FLIGHT_SETTLE` of
    when it was sent, after adopting those found: the rest may still be being added.
    """
    dataset_id = state.dataset_id
    if dataset_id is None:
        raise StateError(f"{push_path(dataset)}: in_flight is set but dataset_id is not")
    in_flight = set(state.in_flight)
    report(f"checking dataset {dataset_id} for {len(in_flight)} records of an interrupted batch")
    try:
        listing = api.list_records(dataset_id)
    except RecordsListTruncated as exc:
        # A record missing from a cut-off listing may still be in the dataset: resending its
        # row would add it twice. Listing again reads no further, so say how to settle it.
        raise _undecided(state, str(exc), hint=_LOOK_UP_HINT) from exc
    except ExternalError as exc:
        raise _undecided(
            state,
            str(exc),
            hint=f"push again once the dataset's records can be listed; {PUSH_FILE} lists the "
            "rows as in_flight, and pushing them now could add them twice",
        ) from exc
    if listing.unreadable:
        # Any item the listing could not read may be a record of the batch: resending its row
        # would add it twice.
        why = f"records-all listed {listing.unreadable} items that are not readable records"
        raise _undecided(state, why, hint=_LOOK_UP_HINT)
    found: dict[str, str] = {}
    for record in listing.records:
        row_id = record.metadata.get("row_id")
        if isinstance(row_id, str) and row_id in in_flight:
            found.setdefault(row_id, record.id)
    missing = tuple(row_id for row_id in state.in_flight if row_id not in found)
    sent_at = state.in_flight_at
    # A receipt without in_flight_at predates it: its batch is taken as settled.
    settled_at = sent_at + IN_FLIGHT_SETTLE if sent_at is not None else None
    unsettled = bool(missing) and settled_at is not None and clock() < settled_at
    state = replace(
        state,
        record_ids={**state.record_ids, **found},
        record_digests={**state.record_digests, **_sent_digests(state, found, local)},
        # Until the batch has settled its missing rows stay in flight, so none is sent twice.
        in_flight=missing if unsettled else (),
        in_flight_digests=(
            {row_id: d for row_id, d in state.in_flight_digests.items() if row_id in missing}
            if unsettled
            else {}
        ),
        in_flight_at=sent_at if unsettled else None,
        # Records were added since the version recorded, and the listing does not tell how many.
        dataset_version=None if found else state.dataset_version,
    )
    _save(dataset, state)
    if unsettled and sent_at is not None and settled_at is not None:
        raise StateError(
            f"{len(missing)} of the {len(in_flight)} records sent to dataset {dataset_id} at "
            f"{sent_at:%H:%M:%S} UTC by an interrupted push are not in it yet, and may still be "
            f"being added ({len(found)} found were recorded): {_some(missing)}",
            hint=(
                f"retry after {_minute_after(settled_at):%H:%M} UTC: {PUSH_FILE} keeps the "
                "missing rows as in_flight, and sending them sooner could add them twice"
            ),
        )
    report(
        f"{len(found)} of the {len(in_flight)} records were already in the dataset; "
        f"{len(missing)} are sent again"
    )
    return state


def _sent_digests(
    state: PushState, found: Iterable[str], local: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """The digest of the record sent for each adopted row id in ``found``, where known."""
    digests: dict[str, str] = {}
    for row_id in found:
        if (sent := state.in_flight_digests.get(row_id)) is not None:
            digests[row_id] = sent
        elif row_id in local:
            digests[row_id] = record_digest(local[row_id])
    return digests


def _now(clock: Clock) -> datetime:
    """The time in UTC to the second, as ``push.json`` records it."""
    return clock().astimezone(UTC).replace(microsecond=0)


def _parse_time(text: str) -> datetime | None:
    """``text`` as a UTC time, or ``None`` when it is not ISO 8601 with an offset."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.astimezone(UTC) if moment.tzinfo is not None else None


def _minute_after(moment: datetime) -> datetime:
    """``moment`` rounded up to the minute, so a time shown as HH:MM is not before it."""
    floor = moment.replace(second=0, microsecond=0)
    return floor if floor == moment else floor + timedelta(minutes=1)


def _undecided(state: PushState, why: str, *, hint: str) -> StateError:
    """The error of a push that cannot tell which in-flight records reached the dataset."""
    return StateError(
        f"cannot tell which of {len(state.in_flight)} records sent to dataset "
        f"{state.dataset_id} by an interrupted push were added ({why}); they may already be "
        f"there: {_some(state.in_flight)}",
        hint=hint,
    )


def _some(row_ids: Sequence[str]) -> str:
    shown = ", ".join(row_ids[:SHOWN_ROW_IDS])
    rest = len(row_ids) - SHOWN_ROW_IDS
    return f"{shown} and {rest} more" if rest > 0 else shown


def _save(dataset: DatasetDir, state: PushState) -> None:
    atomic_write_json(push_path(dataset), state.to_json())


def _check_state(
    dataset: DatasetDir, state: PushState, *, project: str, name: str, digest: str
) -> None:
    path = push_path(dataset)
    if state.settings_hash != digest:
        raise SettingsConflict(
            f"{path} was written for a dataset directory with other settings",
            hint=f"move {path} aside to push this directory as a new dataset",
        )
    if state.project != project or state.dataset_name != name:
        raise SettingsConflict(
            f"{dataset.path} was pushed to dataset {state.dataset_name!r} in project "
            f"{state.project!r}, not {name!r} in {project!r}",
            hint=(
                "leave --name and --project out (or pass --name "
                f"{shlex.quote(state.dataset_name)} --project {shlex.quote(state.project)}) "
                f"to add to that dataset; a directory pushes to "
                f"a new dataset only without a {PUSH_FILE}, so move {path} aside first to "
                "push to another one"
            ),
        )


def _require_name(value: str, flag: str) -> None:
    if not value.strip():
        raise UsageError(f"{flag} must not be empty")


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # A second SIGTERM must not stop the cleanup.
    raise PushTerminated(
        "the push stopped: the process received SIGTERM",
        hint="run the same command again to resume",
    )


@contextmanager
def _sigterm_raises() -> Iterator[None]:
    """Turn SIGTERM into :class:`PushTerminated`, so the batch file is removed on the way out.

    Signals can only be set on the main thread; elsewhere this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
