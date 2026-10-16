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
  listed with ``pup llm-obs datasets records-all``, which pages through every record in one
  call and takes no cursor, :data:`RECORDS_ALL_PAGE_SIZE` records a page; a listing it reports
  ``truncated``, that reached its page cap, or that lists one record twice (the dataset changed
  while it was read) is refused as :class:`RecordsListTruncated` rather than read as the whole
  dataset. Items it cannot read are counted in :attr:`RecordsListing.unreadable`, not dropped
  silently.

Lists are read page by page (``meta.after``) until the name is found. Creating is idempotent
by name: when a create call fails or its response cannot be read, the project or dataset is
looked up again by name before the failure is reported, since the create may have gone
through. Files passed with ``--file`` are written ``0600`` in a private temporary directory
and removed afterwards, since records hold trace content.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pup_finetune.datadog.jsonapi import JSON_API, object_id, private_json_file
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
RECORDS_ALL_MAX_PAGES = 200
"""The pages after which ``records-all`` stops, with only a warning on stderr: a listing of
this many pages may be missing records although it reports ``truncated: false``."""
RECORDS_ALL_PAGE_SIZE = 1000
"""Records a ``records-all`` page holds; at pup's default of 100 the page cap stops every
listing of a dataset over 20,000 records."""

_SOURCE = "Agent Observability"


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """A dataset found by name: its id, description and current version (``None`` when the
    listing does not say)."""

    id: str
    description: str
    current_version: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """A record of a dataset, as listed: its id, metadata, and its input, expected output and
    tags as stored (``None`` and empty when absent)."""

    id: str
    metadata: Mapping[str, Any]
    input: Any = None
    expected_output: Any = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecordsListing:
    """A dataset's records, as ``records-all`` lists them."""

    records: list[DatasetRecord]
    """The readable records, in listed order; a record without metadata has it empty."""
    unreadable: int = 0
    """Listed items that are not a record with an id, attributes and object (or no) metadata:
    any one of them may be a record the caller looks for."""


@dataclass(frozen=True, slots=True)
class RecordsAdded:
    """What one ``records-add`` call created: record ids in input order, and the version."""

    record_ids: list[str]
    dataset_version: int | None


class RecordsListTruncated(ExternalError):
    """``records-all`` listed only part of a dataset's records.

    pup cannot be asked to continue from where it stopped, so listing again reads no further.
    """


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
        with private_json_file(body) as path:
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
        version = item["attributes"].get("current_version")
        return DatasetInfo(
            str(item["id"]),
            description if isinstance(description, str) else "",
            version if type(version) is int else None,
        )

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
        with private_json_file(list(records)) as path:
            data = self._pup.run_json([*args, "--file", str(path), "--confirm"])
        return _records_added(data, expected=len(records), dataset_id=dataset_id)

    def list_records(self, dataset_id: str) -> RecordsListing:
        """Every record of the dataset, through ``records-all``, and how many items were unreadable.

        Raises :class:`RecordsListTruncated` when the response is marked ``truncated``,
        fetched :data:`RECORDS_ALL_MAX_PAGES` pages or lists a record twice, since records
        missing from it would be taken for records the dataset does not hold.
        """
        data = self._pup.run_json(
            [
                "llm-obs",
                "datasets",
                "records-all",
                "--dataset-id",
                dataset_id,
                "--limit",
                str(RECORDS_ALL_PAGE_SIZE),
            ],
            timeout=RECORDS_LIST_TIMEOUT_SECONDS,
        )
        items = data.get("records") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ExternalError("the records-all response has no records list", source=_SOURCE)
        pages = data.get("pages_fetched")
        capped = isinstance(pages, int) and pages >= RECORDS_ALL_MAX_PAGES
        if data.get("truncated") is True or capped:
            why = f"stopped after {pages} pages" if capped else "marked truncated"
            raise RecordsListTruncated(
                f"records-all listed only part of the records of dataset {dataset_id} "
                f"({len(items)} records, {why})",
                source=_SOURCE,
            )
        found: list[DatasetRecord] = []
        seen: set[str] = set()
        unreadable = 0
        for item in items:
            record = _dataset_record(item)
            if record is None:
                unreadable += 1
            elif record.id in seen:
                raise RecordsListTruncated(
                    f"records-all listed record {record.id} of dataset {dataset_id} twice: the "
                    "dataset changed while it was read, so other records may be missing",
                    source=_SOURCE,
                )
            else:
                seen.add(record.id)
                found.append(record)
        return RecordsListing(found, unreadable)

    # --- helpers ------------------------------------------------------------------------

    @staticmethod
    def _created(create: Callable[[], Any], find: Callable[[], str | None], *, what: str) -> str:
        """Run ``create`` and return the new id, or the id ``find`` finds when that fails."""
        try:
            created = object_id(create())
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


def _dataset_record(item: Any) -> DatasetRecord | None:
    """``item`` of a records-all listing as a record, or ``None`` when it cannot be read.

    A record whose metadata is absent or null is readable, with empty metadata.
    """
    if not isinstance(item, dict):
        return None
    identifier = item.get("id")
    attributes = item.get("attributes")
    if not isinstance(identifier, str) or not identifier or not isinstance(attributes, dict):
        return None
    metadata = attributes.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        return None
    tags = attributes.get("tags")
    return DatasetRecord(
        identifier,
        metadata,
        input=attributes.get("input"),
        expected_output=attributes.get("expected_output"),
        tags=tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else (),
    )


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
