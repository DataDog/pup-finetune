# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake Agent Observability Export API, served through :class:`~tests.fakes.pup.FakePup`.

Pages are answered by the cursor in the request body: the first page for no cursor, each
later page for the ``meta.page.after`` of the page before it::

    api = FakeExportApi([page_one, page_two])
    pup = api.install(FakePup()).client()
    api.timeouts["cursor-2"] = 1  # answer `status: timeout` once before the real page
    api.failures[None] = 1  # make pup exit 1 once for the first page

Request bodies are recorded in :attr:`FakeExportApi.requests`. An unknown cursor makes pup
exit 1, the way the API rejects one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pup_finetune.datadog.spans import EXPORT_PATH
from tests.fakes.pup import FakePup, PupCall, PupExit

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "llmobs"


def fixture_pages() -> list[dict[str, Any]]:
    """The synthetic Export API pages in ``tests/fixtures/llmobs``, in cursor order."""
    return [
        json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        for name in ("search-page-1.json", "search-page-2.json")
    ]


def span_item(
    span_id: str, trace_id: str = "synthetic-trace-9000", **attributes: Any
) -> dict[str, Any]:
    """A minimal ``data`` item for an LLM span."""
    return {
        "id": span_id,
        "type": "span",
        "attributes": {"span_id": span_id, "trace_id": trace_id, "span_kind": "llm", **attributes},
    }


def page(items: Sequence[Any], after: str | None = None, *, status: str = "done") -> dict[str, Any]:
    """An Export API response holding ``items``, continuing at ``after``."""
    return {
        "data": list(items),
        "meta": {"page": {"after": after} if after else {}, "status": status, "elapsed": 10},
    }


SYNTHETIC_ACCOUNT: dict[str, str | None] = {"site": "datadoghq.com", "org": "synthetic-org"}
"""The Datadog account :class:`FakeExportApi` reports pup as logged in to."""


class FakeExportApi:
    """Serves ``pages`` by cursor; see the module docstring."""

    def __init__(self, pages: Sequence[Mapping[str, Any]]) -> None:
        self._pages: dict[str | None, Mapping[str, Any]] = {}
        cursor: str | None = None
        for item in pages:
            self._pages.setdefault(cursor, item)
            cursor = item.get("meta", {}).get("page", {}).get("after") or None
        self.requests: list[dict[str, Any]] = []
        self.calls: list[PupCall] = []
        self.timeouts: dict[str | None, int] = {}
        self.failures: dict[str | None, int] = {}
        self.account: dict[str, str | None] = dict(SYNTHETIC_ACCOUNT)

    @property
    def cursors(self) -> list[str | None]:
        """The cursor of every request, in order (``None`` for the first page)."""
        return [body["data"]["attributes"]["page"].get("cursor") for body in self.requests]

    def install(self, fake: FakePup) -> FakePup:
        """Answer the Export API route, and ``pup auth status`` with :attr:`account`, on ``fake``.

        Returns ``fake``. Change :attr:`account` to log the fake pup in elsewhere.
        """
        fake.on("auth", "status", handler=lambda call: {"authenticated": True, **self.account})
        return fake.on("api", EXPORT_PATH, handler=self)

    def __call__(self, call: PupCall) -> Any:
        self.calls.append(call)
        body = call.stdin_json()
        self.requests.append(body)
        cursor = body["data"]["attributes"]["page"].get("cursor")
        if self.failures.get(cursor, 0) > 0:
            self.failures[cursor] -= 1
            return PupExit(1, "Error: 502 Bad Gateway")
        if self.timeouts.get(cursor, 0) > 0:
            self.timeouts[cursor] -= 1
            return page([], status="timeout")
        if cursor not in self._pages:
            return PupExit(1, "Error: 400 Bad Request: invalid cursor")
        return self._pages[cursor]
