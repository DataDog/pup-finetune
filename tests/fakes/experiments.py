# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fake Agent Observability experiments and their events, served through :class:`FakePup`.

It answers the calls :class:`~pup_finetune.datadog.experiments.LLMObsExperiments` makes::

    api = FakeExperimentsApi()
    pup = api.install(FakePup()).client()
    api.create_answer = "fail_after_create"  # create the experiment, then exit 1
    api.create_answer = "no_id"  # create it, then answer without its id
    api.events_failures = {2}  # the 2nd events POST exits 1, storing nothing
    api.events_lost_responses = {3}  # the 3rd stores its events, then exits 1
    api.index_after_lists = 2  # events show in `events list` from its 3rd call on
    api.index_limit = 1  # `events list` never counts more than 1 event
    api.events_list_answer = "fail"  # `events list` exits 1 ("no_total": no total_matching)

Other ``pup api`` routes go to the handler registered for ``api`` before :meth:`install`
(e.g. :class:`~tests.fakes.datasets.FakeDatasetsApi`'s), so both fakes can serve one pup.
Every ``--file`` is read while the call runs; its permissions and those of its directory are
kept in :attr:`FakeExperimentsApi.file_modes`, and its path in
:attr:`FakeExperimentsApi.file_paths`.
"""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tests.fakes.pup import FakePup, Handler, PupCall, PupExit, PupStdout

CreateAnswer = Literal["ok", "unreadable", "no_id", "fail_after_create", "fail"]
EventsListAnswer = Literal["ok", "enveloped", "fail", "no_total"]


@dataclass
class FakeExperiment:
    id: str
    attributes: dict[str, Any]
    spans: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    """The ``attributes`` of every events POST that was stored."""

    @property
    def name(self) -> str:
        return str(self.attributes["name"])


class FakeExperimentsApi:
    """In-memory experiments and their events behind the pup calls that reach them."""

    def __init__(self) -> None:
        self.experiments: dict[str, FakeExperiment] = {}
        self.create_answer: CreateAnswer = "ok"
        self.create_bodies: list[dict[str, Any]] = []
        self.events_failures: set[int] = set()
        """1-based numbers of the events POSTs that fail before storing anything."""
        self.events_failure_stderr = "Error: 502 Bad Gateway"
        """What a failing events POST prints."""
        self.events_lost_responses: set[int] = set()
        """1-based numbers of the events POSTs that store their events, then fail."""
        self.events_calls = 0
        self.list_calls = 0
        self.index_after_lists = 0
        """``events list`` counts nothing for this many calls, then every stored span."""
        self.index_limit: int | None = None
        """The most events ``events list`` ever counts."""
        self.events_list_answer: EventsListAnswer = "ok"
        self.content_types: list[str | None] = []
        self.file_modes: list[tuple[int, int]] = []
        self.file_paths: list[Path] = []
        self._fallback: Handler | None = None
        self._next_id = 0

    def install(self, fake: FakePup) -> FakePup:
        self._fallback = fake.handler("api")
        fake.on("llm-obs", "experiments", "create", handler=self._create)
        fake.on("llm-obs", "experiments", "list", handler=self._list)
        fake.on("llm-obs", "experiments", "events", "list", handler=self._events_list)
        fake.on("api", handler=self._api)
        return fake

    def named(self, name: str) -> FakeExperiment:
        (match,) = [item for item in self.experiments.values() if item.name == name]
        return match

    # --- handlers -----------------------------------------------------------------------

    def _create(self, call: PupCall) -> Any:
        body = self._read_file(call)
        self.create_bodies.append(body)
        assert body["data"]["type"] == "experiments"
        if self.create_answer == "fail":
            return PupExit(stderr="Error: 500 Internal Server Error")
        attributes = dict(body["data"]["attributes"])
        name = attributes["name"]
        taken = {item.name for item in self.experiments.values()}
        if attributes.get("ensure_unique") and name in taken:
            suffix = 1
            while f"{name}-{suffix}" in taken:
                suffix += 1
            attributes["name"] = f"{name}-{suffix}"
        self._next_id += 1
        experiment = FakeExperiment(f"synthetic-experiment-{self._next_id:04d}", attributes)
        self.experiments[experiment.id] = experiment
        if self.create_answer == "fail_after_create":
            return PupExit(stderr="Error: request timed out")
        if self.create_answer == "unreadable":
            return PupStdout("Error: failed to deserialize the response\n")
        if self.create_answer == "no_id":
            return {"data": {"type": "experiments", "attributes": attributes}}
        return {"data": {"id": experiment.id, "type": "experiments", "attributes": attributes}}

    def _list(self, call: PupCall) -> Any:
        dataset_id = _flag(call, "--filter-dataset-id")
        return {
            "data": [
                {"id": item.id, "type": "experiments", "attributes": item.attributes}
                for item in self.experiments.values()
                if dataset_id is None or item.attributes.get("dataset_id") == dataset_id
            ]
        }

    def _events_list(self, call: PupCall) -> Any:
        self.list_calls += 1
        experiment = self.experiments.get(call.args[4])
        if experiment is None:
            return PupExit(stderr="Error: 404 Not Found: experiment")
        assert "--limit" in call.args
        if self.events_list_answer == "fail":
            return PupExit(stderr="Error: 503 Service Unavailable")
        seen = 0 if self.list_calls <= self.index_after_lists else len(experiment.spans)
        if self.index_limit is not None:
            seen = min(seen, self.index_limit)
        limit = int(_flag(call, "--limit") or 20)
        events = [{"id": span["span_id"]} for span in experiment.spans[:seen][:limit]]
        # The shape pup prints, as checked against the live service.
        if self.events_list_answer == "no_total":
            return {"events": events, "offset": 0, "returned": len(events)}
        if self.events_list_answer == "enveloped":
            return {"data": {"events": events, "total_matching": seen, "returned": len(events)}}
        return {"events": events, "offset": 0, "returned": len(events), "total_matching": seen}

    def _api(self, call: PupCall) -> Any:
        path = call.args[1]
        segments = path.split("/")
        if segments[:4] != ["v2", "llm-obs", "v1", "experiments"] or segments[5:] != ["events"]:
            if self._fallback is not None:
                return self._fallback(call)
            return PupExit(stderr=f"Error: fake API has no route {path}")
        assert call.args[call.args.index("-X") + 1] == "POST"
        experiment = self.experiments.get(segments[4])
        if experiment is None:
            return PupExit(stderr="Error: 404 Not Found: experiment")
        self.events_calls += 1
        self.content_types.append(_header(call, "Content-Type"))
        if self.events_calls in self.events_failures:
            return PupExit(stderr=self.events_failure_stderr)
        body = call.stdin_json()
        assert body["data"]["type"] == "experiments"
        attributes = body["data"]["attributes"]
        experiment.requests.append(attributes)
        experiment.spans.extend(attributes["spans"])
        experiment.metrics.extend(attributes["metrics"])
        if self.events_calls in self.events_lost_responses:
            return PupExit(stderr="Error: request timed out")
        return PupStdout("")  # an empty 2xx, as the API answers

    # --- helpers ------------------------------------------------------------------------

    def _read_file(self, call: PupCall) -> Any:
        path = Path(_flag(call, "--file") or "")
        self.file_paths.append(path)
        self.file_modes.append(
            (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(path.parent.stat().st_mode))
        )
        return json.loads(path.read_text(encoding="utf-8"))


def _flag(call: PupCall, name: str) -> str | None:
    args = call.args
    return args[args.index(name) + 1] if name in args else None


def _header(call: PupCall, name: str) -> str | None:
    args = call.args
    for index, arg in enumerate(args[:-1]):
        if arg == "-H" and args[index + 1].lower().startswith(f"{name.lower()}:"):
            return args[index + 1].partition(":")[2].strip()
    return None
