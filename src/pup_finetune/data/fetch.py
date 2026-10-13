# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fetch an Agent Observability dataset into a dataset directory, to train and evaluate on.

``dataset fetch DIR --project P --name N`` makes the dataset in Datadog the source of a run:
what was curated there (records edited, added or removed in the product, or by an agent
through pup) is what trains. The dataset's current records are read
(:meth:`~pup_finetune.datadog.datasets.LLMObsDatasets.list_records`) and each becomes a
canonical row, validated as ``dataset import`` validates rows:

- the messages are the record's ``input.messages`` followed by its ``expected_output`` (an
  assistant message); the tools are ``input.tools``;
- the row id is ``metadata.row_id`` (as ``dataset push`` writes it), else the record id, and
  the record id too when an earlier record already has that row id (a copied record); the
  source is ``metadata``'s ``trace_id``, ``span_id``, ``ml_app`` and ``source_model``;
- the split is ``metadata.split``; a record without one is assigned as any new row is, by its
  trace. Two records of one trace that name different splits are excluded as
  ``split_conflict``.

A record that is not a conversation is excluded as ``unreadable_record``, and an invalid one
with its row error code. A listing that holds items it cannot read as records is refused, as
they may be records the dataset needs. The directory's settings freeze the dataset (its project, name, id
and version): a later version is fetched into a new directory, so one directory is always
one version of the dataset. The fetch also writes the push receipt (``push.json``) of that
dataset, with every row's record, so ``evaluate`` publishes its experiments onto it and
``dataset push`` has nothing to send. Nothing is written to Datadog.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pup_finetune.data.dataset_dir import AddResult, DatasetDir, DatasetSettings, SourceKind
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.push import (
    PushState,
    dataset_record,
    push_path,
    read_push_state,
    record_digest,
)
from pup_finetune.data.schema import RowError, TrainingRow, split_key, validate_row
from pup_finetune.data.split import SPLITS, SplitName, SplitParams
from pup_finetune.datadog.datasets import DatasetInfo, DatasetRecord, LLMObsDatasets
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.storage import atomic_write_json, settings_hash

__all__ = ["SOURCE_KIND", "FetchResult", "fetch_dataset", "rows_from_records"]

SOURCE_KIND: SourceKind = "datadog_dataset"


@dataclass(slots=True)
class FetchedRows:
    """The rows read from a dataset's records, their exclusions, and the splits named."""

    rows: list[TrainingRow] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    splits: dict[str, SplitName] = field(default_factory=dict)
    """The split each split key (a trace) is named in by its records."""
    record_ids: dict[str, str] = field(default_factory=dict)
    """The record id of each row id."""


@dataclass(frozen=True, slots=True)
class FetchResult:
    """What a fetch read and added."""

    dataset_id: str
    dataset_version: int | None
    records: int
    added: AddResult


def _named_split(record: DatasetRecord) -> SplitName | None:
    named = record.metadata.get("split")
    return cast(SplitName, named) if named in SPLITS else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _row_data(record: DatasetRecord) -> dict[str, Any] | None:
    """The canonical row a record holds, before validation; ``None`` when it holds none."""
    given = record.input
    answer = record.expected_output
    messages = given.get("messages") if isinstance(given, Mapping) else None
    if not isinstance(messages, list) or not isinstance(answer, Mapping):
        return None
    if answer.get("role") != "assistant":
        return None
    tools = given.get("tools") if isinstance(given, Mapping) else None
    metadata = record.metadata
    return {
        "id": _text(metadata.get("row_id")) or record.id,
        "messages": [*messages, dict(answer)],
        "tools": tools if isinstance(tools, list) else [],
        "source": {
            "trace_id": _text(metadata.get("trace_id")),
            "span_id": _text(metadata.get("span_id")),
            "ml_app": _text(metadata.get("ml_app")),
            "model_name": _text(metadata.get("source_model")),
        },
    }


def rows_from_records(records: Sequence[DatasetRecord]) -> FetchedRows:
    """The rows of ``records`` (see the module docstring), in record order."""
    found = FetchedRows()
    for record in records:
        data = _row_data(record)
        metadata = record.metadata
        if data is None:
            found.exclusions.append(
                Exclusion.for_source(
                    ExclusionCode.UNREADABLE_RECORD,
                    f"record {record.id}: not chat messages with an assistant answer",
                    row_id=_text(metadata.get("row_id")) or record.id,
                    trace_id=_text(metadata.get("trace_id")),
                    span_id=_text(metadata.get("span_id")),
                )
            )
            continue
        if data["id"] in found.record_ids:
            data["id"] = record.id  # a copied record keeps the row id of its original
        try:
            row = validate_row(data)
        except RowError as error:
            found.exclusions.append(
                Exclusion.for_source(
                    error.code,
                    f"record {record.id}: {error.detail}",
                    row_id=data["id"],
                    trace_id=data["source"]["trace_id"],
                    span_id=data["source"]["span_id"],
                )
            )
            continue
        named = _named_split(record)
        key = split_key(row)
        if named is not None:
            known = found.splits.setdefault(key, named)
            if known != named:
                found.exclusions.append(
                    Exclusion.for_row(
                        row,
                        ExclusionCode.SPLIT_CONFLICT,
                        f"record {record.id} names {named}, another record of its trace {known}",
                    )
                )
                continue
        found.rows.append(row)
        found.record_ids[row.id] = record.id
    return found


def fetch_dataset(
    dataset: DatasetDir,
    datasets: LLMObsDatasets,
    *,
    project: str,
    name: str,
    split: SplitParams,
) -> FetchResult:
    """Fetch the current records of dataset ``name`` in ``project`` into ``dataset``.

    Raises :class:`UsageError` when the project or the dataset does not exist, and
    :class:`~pup_finetune.errors.SettingsConflict` when ``dataset`` holds another dataset or
    another version of it (see the module docstring).
    """
    project_id = datasets.find_project(project)
    if project_id is None:
        raise UsageError(
            f"no Agent Observability project named {project!r}",
            hint="pass the project name the dataset belongs to",
        )
    info = datasets.find_dataset(project_id, name)
    if info is None:
        raise UsageError(
            f"no dataset named {name!r} in project {project!r}",
            hint="pass the dataset's exact name",
        )
    listing = datasets.list_records(info.id)
    if listing.unreadable:
        raise ExternalError(
            f"the records of dataset {name!r} hold {listing.unreadable} items that cannot be "
            "read as records, which may be records the dataset needs",
            source="agent observability",
            hint="check the dataset in Agent Observability, then fetch again",
        )
    fetched = rows_from_records(listing.records)
    settings = DatasetSettings(
        SOURCE_KIND,
        split,
        {
            "project": project,
            "project_id": project_id,
            "dataset_name": name,
            "dataset_id": info.id,
            "dataset_version": info.current_version,
        },
    )
    added = dataset.add(
        fetched.rows, fetched.exclusions, settings=settings, split_overrides=fetched.splits
    )
    _write_receipt(dataset, fetched, project=project, project_id=project_id, name=name, info=info)
    return FetchResult(info.id, info.current_version, len(listing.records), added)


def _write_receipt(
    dataset: DatasetDir,
    fetched: FetchedRows,
    *,
    project: str,
    project_id: str,
    name: str,
    info: DatasetInfo,
) -> None:
    """``push.json`` naming the fetched dataset, with the record of every kept row."""
    previous = read_push_state(dataset)
    by_split = dataset.rows_by_split()
    record_ids: dict[str, str] = {}
    digests: dict[str, str] = {}
    for split_name, rows in by_split.items():
        for row in rows:
            record = fetched.record_ids.get(row.id)
            if record is not None:
                record_ids[row.id] = record
                digests[row.id] = record_digest(dataset_record(row, split_name))
    state = PushState(
        project=project,
        project_id=project_id,
        dataset_name=name,
        settings_hash=settings_hash(dataset.settings()),
        push_id=previous.push_id if previous is not None else secrets.token_hex(8),
        dataset_id=info.id,
        dataset_version=info.current_version,
        record_ids=record_ids,
        record_digests=digests,
    )
    atomic_write_json(push_path(dataset), state.to_json())
