# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fake Agent Observability projects, datasets and records, served through :class:`FakePup`.

It answers the calls :class:`~pup_finetune.datadog.datasets.LLMObsDatasets` makes::

    api = FakeDatasetsApi()
    pup = api.install(FakePup()).client()
    api.add_project("pup-finetune")  # an existing project; returns its id
    api.dataset_create_answer = "unreadable"  # create the dataset, but print text that is not JSON
    api.records_add_failures = {2}  # make the 2nd records-add call exit 1, adding nothing
    api.records_add_lost_responses = {3}  # the 3rd adds its records, then exits 1
    api.page_size = 2  # serve project and dataset lists 2 items a page, by meta.after
    api.records_all_limit = 3  # records-all lists only the first 3 records, marked truncated

Name filters match by substring, so callers must pick the exact name themselves. Every
``--file`` is read while the call runs; its permissions and those of its directory are kept
in :attr:`FakeDatasetsApi.file_modes`, and its path in :attr:`FakeDatasetsApi.file_paths`.
"""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from tests.fakes.pup import FakePup, PupCall, PupExit, PupStdout

CreateAnswer = Literal["ok", "unreadable", "fail_after_create", "fail", "no_id"]


@dataclass
class FakeDataset:
    id: str
    project_id: str
    name: str
    description: str
    version: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)


class FakeDatasetsApi:
    """In-memory projects, datasets and records behind the pup calls that reach them."""

    def __init__(self) -> None:
        self.projects: dict[str, str] = {}
        """Project id to name."""
        self.datasets: dict[str, FakeDataset] = {}
        self.project_create_answer: CreateAnswer = "ok"
        self.dataset_create_answer: CreateAnswer = "ok"
        self.records_add_failures: set[int] = set()
        """1-based numbers of the records-add calls that fail before adding anything."""
        self.records_add_lost_responses: set[int] = set()
        """1-based numbers of the records-add calls that add their records, then fail."""
        self.records_add_calls = 0
        self.records_all_fails = False
        """Make every records-all call exit 1."""
        self.records_all_limit: int | None = None
        """List only this many records in records-all, with ``truncated: true``."""
        self.page_size: int | None = None
        """Items per page of a projects or datasets list; ``None`` serves one page."""
        self.file_modes: list[tuple[int, int]] = []
        """``(file mode, directory mode)`` of every ``--file``."""
        self.file_paths: list[Path] = []
        self.content_types: list[str | None] = []
        """The Content-Type of every dataset create."""
        self._next_id = 0

    def install(self, fake: FakePup) -> FakePup:
        fake.on("llm-obs", "projects", "list", handler=self._list_projects)
        fake.on("llm-obs", "projects", "create", handler=self._create_project)
        fake.on("llm-obs", "datasets", "records-add", handler=self._records_add)
        fake.on("llm-obs", "datasets", "records-all", handler=self._records_all)
        fake.on("api", handler=self._api)
        return fake

    def add_project(self, name: str) -> str:
        project_id = self._new_id("project")
        self.projects[project_id] = name
        return project_id

    def add_dataset(self, project_id: str, name: str, description: str = "") -> FakeDataset:
        dataset = FakeDataset(self._new_id("dataset"), project_id, name, description)
        self.datasets[dataset.id] = dataset
        return dataset

    def dataset_named(self, name: str) -> FakeDataset:
        (match,) = [dataset for dataset in self.datasets.values() if dataset.name == name]
        return match

    # --- handlers -----------------------------------------------------------------------

    def _list_projects(self, call: PupCall) -> Any:
        wanted = _flag(call, "--filter-name") or ""
        items = [
            {"id": pid, "type": "projects", "attributes": {"name": name}}
            for pid, name in self.projects.items()
            if wanted in name
        ]
        return self._page(items, _flag(call, "--cursor"))

    def _create_project(self, call: PupCall) -> Any:
        body = self._read_file(call)
        attributes = body["data"]["attributes"]
        assert body["data"]["type"] == "projects"

        def create() -> str:
            return self.add_project(attributes["name"])

        return _answer(self.project_create_answer, create, kind="projects")

    def _api(self, call: PupCall) -> Any:
        path, method = call.args[1], call.args[call.args.index("-X") + 1]
        parts = urlsplit(path)
        segments = parts.path.split("/")
        if segments[:3] != ["v2", "llm-obs", "v1"] or segments[4:] != ["datasets"]:
            return PupExit(stderr=f"Error: fake API has no route {path}")
        project_id = segments[3]
        if project_id not in self.projects:
            return PupExit(stderr="Error: 404 Not Found: project")
        if method == "GET":
            query = parse_qs(parts.query)
            wanted = query.get("filter[name]", [""])[0]
            items = [
                {
                    "id": dataset.id,
                    "type": "datasets",
                    "attributes": {
                        "name": dataset.name,
                        "description": dataset.description,
                        "current_version": dataset.version,
                    },
                }
                for dataset in self.datasets.values()
                if dataset.project_id == project_id and wanted in dataset.name
            ]
            return self._page(items, query.get("page[cursor]", [None])[0])
        assert method == "POST"
        self.content_types.append(_header(call, "Content-Type"))
        body = call.stdin_json()
        assert body["data"]["type"] == "datasets"
        attributes = body["data"]["attributes"]

        def create() -> str:
            dataset = self.add_dataset(project_id, attributes["name"])
            dataset.description = attributes.get("description", "")
            return dataset.id

        return _answer(self.dataset_create_answer, create, kind="datasets")

    def _records_add(self, call: PupCall) -> Any:
        self.records_add_calls += 1
        records = self._read_file(call)
        if "--confirm" not in call.args:
            return {"preview": True, "planned_count": len(records)}
        if self.records_add_calls in self.records_add_failures:
            return PupExit(stderr="Error: 502 Bad Gateway")
        project_id = _flag(call, "--project-id")
        dataset = self.datasets.get(_flag(call, "--dataset-id") or "")
        if dataset is None or dataset.project_id != project_id:
            return PupExit(stderr="Error: 404 Not Found: dataset")
        ids = [self._new_id("record") for _ in records]
        dataset.records.extend(records)
        dataset.record_ids.extend(ids)
        dataset.version += 1
        if self.records_add_calls in self.records_add_lost_responses:
            return PupExit(stderr="Error: request timed out")
        return {
            "created_count": len(ids),
            "created_record_ids": ids,
            "dataset_id": dataset.id,
            "dataset_version": dataset.version,
            "project_id": project_id,
        }

    def _records_all(self, call: PupCall) -> Any:
        if self.records_all_fails:
            return PupExit(stderr="Error: 503 Service Unavailable")
        dataset = self.datasets.get(_flag(call, "--dataset-id") or "")
        if dataset is None:
            return PupExit(stderr="Error: 404 Not Found: dataset")
        records = [
            {
                "id": record_id,
                "type": "datasets",
                "attributes": {**record, "dataset_id": dataset.id},
            }
            for record_id, record in zip(dataset.record_ids, dataset.records, strict=True)
        ]
        truncated = self.records_all_limit is not None and len(records) > self.records_all_limit
        if truncated:
            records = records[: self.records_all_limit]
        return {
            "dataset_id": dataset.id,
            "kind": "full_list",
            "records": records,
            "returned": len(records),
            "truncated": truncated,
            "pages_fetched": 1,
        }

    # --- helpers ------------------------------------------------------------------------

    def _page(self, items: list[dict[str, Any]], cursor: str | None) -> Any:
        """One page of ``items`` from ``cursor`` (an offset), with the next in ``meta.after``."""
        if self.page_size is None:
            return {"data": items, "meta": {"after": ""}}
        start = int(cursor or 0)
        end = start + self.page_size
        return {"data": items[start:end], "meta": {"after": str(end) if end < len(items) else ""}}

    def _read_file(self, call: PupCall) -> Any:
        path = Path(_flag(call, "--file") or "")
        self.file_paths.append(path)
        self.file_modes.append(
            (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(path.parent.stat().st_mode))
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def _new_id(self, kind: str) -> str:
        self._next_id += 1
        return f"synthetic-{kind}-{self._next_id:04d}"


def _flag(call: PupCall, name: str) -> str | None:
    args = call.args
    return args[args.index(name) + 1] if name in args else None


def _header(call: PupCall, name: str) -> str | None:
    args = call.args
    for index, arg in enumerate(args[:-1]):
        if arg == "-H" and args[index + 1].lower().startswith(f"{name.lower()}:"):
            return args[index + 1].partition(":")[2].strip()
    return None


def _answer(answer: CreateAnswer, create: Any, *, kind: str) -> Any:
    if answer == "fail":
        return PupExit(stderr="Error: 500 Internal Server Error")
    created: str = create()
    if answer == "unreadable":
        return PupStdout("Error: failed to deserialize the response\n")
    if answer == "fail_after_create":
        return PupExit(stderr="Error: failed to deserialize the response")
    if answer == "no_id":
        return {"data": {"type": kind}}
    return {"data": {"id": created, "type": kind, "attributes": {}}}
