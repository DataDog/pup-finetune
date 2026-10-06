# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Agent Observability projects, datasets and dataset records, through :class:`Pup`.

- Projects are listed with ``pup llm-obs projects list --filter-name`` and created with
  ``pup llm-obs projects create --file``.
- Datasets are listed and created through ``pup api`` on ``v2/llm-obs/v1/<project>/datasets``
  rather than ``pup llm-obs datasets create``, whose typed response pup 1.10.7 cannot read
  even though the dataset is created.
- Records are added with ``pup llm-obs datasets records-add --confirm``. Without ``--confirm``
  the endpoint only previews; callers run this after their own ``--confirm`` gate. They are
  listed with ``pup llm-obs datasets records-all``, which pages through every record.

Lists are read page by page (``meta.after``) until the name is found. Creating is idempotent
by name: when a create call fails or its response cannot be read, the project or dataset is
looked up again by name before the failure is reported, since the create may have gone
through. Files passed with ``--file`` are written ``0600`` in a private temporary directory
and removed afterwards, since records hold trace content.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pup_finetune.datadog.pup import Pup
from pup_finetune.errors import ExternalError

DEFAULT_PROJECT = "pup-finetune"
"""The project ``dataset push`` creates datasets in unless told otherwise."""
RECORDS_BATCH_SIZE = 100
"""Records sent per ``records-add`` call."""
LIST_MAX_PAGES = 100
"""Pages of a projects or datasets list read before giving up on a name."""
RECORDS_LIST_TIMEOUT_SECONDS = 600.0
"""``records-all`` pages through the whole dataset in one pup call."""

JSON_API = "application/vnd.api+json"
_SOURCE = "Agent Observability"


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """A dataset found by name: its id and description."""

    id: str
    description: str


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """A record of a dataset, as listed: its id and metadata."""

    id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RecordsAdded:
    """What one ``records-add`` call created: record ids in input order, and the version."""

    record_ids: list[str]
    dataset_version: int | None


class LLMObsDatasets:
    """Agent Observability projects, datasets and records, read and written through ``pup``."""

    def __init__(self, pup: Pup) -> None:
        self._pup = pup

    # --- projects -----------------------------------------------------------------------

    def find_project(self, name: str) -> str | None:
        """The id of the project named exactly ``name``, or ``None``."""

        def page(cursor: str | None) -> Any:
            args = ["llm-obs", "projects", "list", "--filter-name", name]
            return self._pup.run_json([*args, "--cursor", cursor] if cursor else args)

        item = _find_by_name(page, name, what="projects list")
        return None if item is None else str(item["id"])

    def ensure_project(self, name: str, *, description: str = "") -> str:
        """The id of the project named ``name``, created when there is none."""
        found = self.find_project(name)
        if found is not None:
            return found
        body = {
            "data": {"type": "projects", "attributes": {"name": name, "description": description}}
        }
        with _json_file(body) as path:
            return self._created(
                lambda: self._pup.run_json(["llm-obs", "projects", "create", "--file", str(path)]),
                lambda: self.find_project(name),
                what=f"project {name!r}",
            )

    # --- datasets -----------------------------------------------------------------------

    def find_dataset(self, project_id: str, name: str) -> DatasetInfo | None:
        """The dataset named exactly ``name`` in the project, or ``None``."""
        path = f"{_datasets_path(project_id)}?filter[name]={quote(name)}"

        def page(cursor: str | None) -> Any:
            return self._pup.api("GET", f"{path}&page[cursor]={quote(cursor)}" if cursor else path)

        item = _find_by_name(page, name, what="datasets list")
        if item is None:
            return None
        description = item["attributes"].get("description")
        return DatasetInfo(str(item["id"]), description if isinstance(description, str) else "")

    def create_dataset(
        self, project_id: str, name: str, *, description: str = "", marker: str | None = None
    ) -> str:
        """Create a dataset named ``name`` in the project; returns its id.

        When the response cannot be read, or the call fails, the dataset is looked up by name:
        the create may have succeeded all the same. With ``marker``, a dataset found that way
        is adopted only when its description contains ``marker``, so a same-name dataset
        someone else created is never taken for this one.
        """
        body = {
            "data": {"type": "datasets", "attributes": {"name": name, "description": description}}
        }

        def find() -> str | None:
            found = self.find_dataset(project_id, name)
            if found is None or (marker is not None and marker not in found.description):
                return None
            return found.id

        return self._created(
            lambda: self._pup.api("POST", _datasets_path(project_id), body, content_type=JSON_API),
            find,
            what=f"dataset {name!r}",
        )

    # --- records ------------------------------------------------------------------------

    def add_records(
        self, project_id: str, dataset_id: str, records: Sequence[Mapping[str, Any]]
    ) -> RecordsAdded:
        """Add ``records`` (``{input, expected_output, metadata, tags}`` each) to the dataset.

        Returns the created record ids, in the order of ``records``, and the dataset version
        after the insert.
        """
        if not records:
            raise ValueError("no records to add")
        args = ["llm-obs", "datasets", "records-add", "--project-id", project_id]
        args += ["--dataset-id", dataset_id]
        with _json_file(list(records)) as path:
            data = self._pup.run_json([*args, "--file", str(path), "--confirm"])
        return _records_added(data, expected=len(records), dataset_id=dataset_id)

    def list_records(self, dataset_id: str) -> list[DatasetRecord]:
        """Every record of the dataset, through ``records-all``; unreadable items are skipped."""
        data = self._pup.run_json(
            ["llm-obs", "datasets", "records-all", "--dataset-id", dataset_id],
            timeout=RECORDS_LIST_TIMEOUT_SECONDS,
        )
        items = data.get("records") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ExternalError("the records-all response has no records list", source=_SOURCE)
        found: list[DatasetRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            identifier = item.get("id")
            attributes = item.get("attributes")
            metadata = attributes.get("metadata") if isinstance(attributes, dict) else None
            if isinstance(identifier, str) and identifier and isinstance(metadata, dict):
                found.append(DatasetRecord(identifier, metadata))
        return found

    # --- helpers ------------------------------------------------------------------------

    @staticmethod
    def _created(create: Callable[[], Any], find: Callable[[], str | None], *, what: str) -> str:
        """Run ``create`` and return the new id, or the id ``find`` finds when that fails."""
        try:
            created = _object_id(create())
        except ExternalError as original:
            try:
                found = find()
            except ExternalError:
                raise original from None  # The create failure, not the lookup's, explains it.
            if found is None:
                raise
            return found
        if created is not None:
            return created
        found = find()
        if found is None:
            raise ExternalError(
                f"creating {what} returned no id, and no {what} was found afterwards",
                source=_SOURCE,
            )
        return found


def _datasets_path(project_id: str) -> str:
    return f"v2/llm-obs/v1/{quote(project_id, safe='')}/datasets"


@contextlib.contextmanager
def _json_file(data: Any) -> Iterator[Path]:
    """``data`` as a ``0600`` JSON file in a fresh private directory, removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="pup-finetune-") as directory:
        path = Path(directory) / "body.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)
        yield path


def _object_id(data: Any) -> str | None:
    """``data.id`` of a JSON:API response, or ``None`` when it has none."""
    if isinstance(data, dict) and isinstance(item := data.get("data"), dict):
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return None


def _find_by_name(
    page: Callable[[str | None], Any], name: str, *, what: str
) -> dict[str, Any] | None:
    """The item of a JSON:API list whose ``attributes.name`` is exactly ``name``, or ``None``.

    ``page(cursor)`` fetches one page (the first with ``None``); pages are read until the name
    is found or ``meta.after`` is empty. The item returned has a non-empty string ``id`` and an
    ``attributes`` object.
    """
    cursor: str | None = None
    for _ in range(LIST_MAX_PAGES):
        data = page(cursor)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ExternalError(f"the {what} response has no data list", source=_SOURCE)
        for item in items:
            if not isinstance(item, dict):
                continue
            attributes = item.get("attributes")
            identifier = item.get("id")
            if (
                isinstance(attributes, dict)
                and attributes.get("name") == name
                and isinstance(identifier, str)
                and identifier
            ):
                return item
        after = _next_cursor(data)
        if after is None or after == cursor:
            return None
        cursor = after
    raise ExternalError(
        f"the {what} has more than {LIST_MAX_PAGES} pages without {name!r}", source=_SOURCE
    )


def _next_cursor(data: dict[str, Any]) -> str | None:
    meta = data.get("meta")
    after = meta.get("after") if isinstance(meta, dict) else None
    return after if isinstance(after, str) and after else None


def _records_added(data: Any, *, expected: int, dataset_id: str) -> RecordsAdded:
    if not isinstance(data, dict):
        raise ExternalError("the records-add response is not a JSON object", source=_SOURCE)
    ids = data.get("created_record_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise ExternalError("the records-add response has no created_record_ids", source=_SOURCE)
    if len(ids) != expected:
        raise ExternalError(
            f"records-add created {len(ids)} records in dataset {dataset_id} for the "
            f"{expected} sent",
            source=_SOURCE,
        )
    version = data.get("dataset_version")
    return RecordsAdded(
        record_ids=list(ids),
        dataset_version=version if type(version) is int else None,
    )
