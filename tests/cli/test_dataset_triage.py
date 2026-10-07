# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`dataset triage` through the Typer app, with the judges' chat client replaced by a fake.

The fake is the real :class:`~pup_finetune.inference.ChatClient` over the fake chat
endpoint, so requests are built and parsed as they are for Fireworks.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import triage as triage_cli
from pup_finetune.curation.triage import DEFAULT_JUDGES, JUDGMENTS_FILE, rubric_template
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.inference import ChatClient, ChatCompleter
from tests.fakes.chat_api import API_KEY, BASE_URL, FakeChatApi, block

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def drop_weather(body: dict[str, Any]) -> str:
    """Judges that drop every row whose first user message mentions the weather."""
    conversation = block(body["messages"][-1]["content"], "conversation")
    first_user = next(m["content"] for m in conversation if m["role"] == "user")
    keep = "weather" not in first_user.lower()
    return json.dumps({"keep": keep, "reason": "fine" if keep else "off topic"})


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> FakeChatApi:
    api = FakeChatApi(answer=drop_weather)

    @contextmanager
    def open_client() -> Iterator[ChatCompleter]:
        with ChatClient(
            BASE_URL, SecretStr(API_KEY), max_attempts=1, transport=api.transport
        ) as client:
            yield client

    monkeypatch.setattr(triage_cli, "open_client", open_client)
    return api


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


@pytest.fixture
def rubric(tmp_path: Path) -> Path:
    path = tmp_path / "rubric.md"
    path.write_text(rubric_template(), encoding="utf-8")
    return path


def accepted(dataset: Path) -> int:
    return len(DatasetDir(dataset).rows())


def test_the_template_prints() -> None:
    result = invoke("dataset", "triage", "--print-template")
    assert result.exit_code == 0
    assert result.stdout == rubric_template()


def test_triage_needs_a_rubric_or_rules(dataset: Path) -> None:
    result = invoke("dataset", "triage", str(dataset))
    assert result.exit_code == 2
    assert "rubric or rules" in result.stderr


def test_the_preview_counts_the_calls_and_calls_nothing(
    dataset: Path, rubric: Path, chat: FakeChatApi
) -> None:
    result = invoke("dataset", "triage", str(dataset), "--rubric", str(rubric))
    assert result.exit_code == 2, result.stderr
    data = output(result)
    rows = accepted(dataset)
    preview = data["preview"]
    assert preview["rows_to_review"] == rows
    assert preview["judge_calls"] == rows * len(DEFAULT_JUDGES)
    assert preview["judges"] == list(DEFAULT_JUDGES)
    assert len(preview["rubric_sha256"]) == 64
    assert data["next_command"] == shlex.join(
        ["pup", "finetune", "dataset", "triage", str(dataset), "--rubric", str(rubric), "--confirm"]
    )
    assert chat.requests == []
    assert not (dataset / JUDGMENTS_FILE).exists()


def test_auto_approve_does_not_confirm(
    dataset: Path, rubric: Path, chat: FakeChatApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_AUTO_APPROVE", "true")
    result = invoke("dataset", "triage", str(dataset), "--rubric", str(rubric))
    assert result.exit_code == 2
    assert chat.requests == []


def test_triage_labels_rows_and_is_allowed_read_only(
    dataset: Path, chat: FakeChatApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--judges", "judge-a,judge-b,judge-c", "--confirm"]
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    rows = accepted(dataset)
    assert data["status"] == "complete"
    assert data["kept"] + data["dropped"] == rows
    assert data["undecided"] == 0
    assert data["dropped"] > 0
    assert data["top_drop_reasons"] == [{"reason": "off topic", "count": data["dropped"]}]
    assert data["calls_made"] == rows * 3
    assert data["next_command"] == f"pup finetune dataset sample {dataset}"
    assert data["next_commands"] == [
        f"pup finetune dataset sample {dataset}",
        f"pup finetune dataset push {dataset} --name ...",
        f"pup finetune train --dataset {dataset} ...",
    ]
    assert data["stale_judge_labels"] == 0
    assert data["dataset"]["dropped"] == data["dropped"]
    assert {body["model"] for body in chat.bodies} == {"judge-a", "judge-b", "judge-c"}
    labels = DatasetDir(dataset).labels()
    assert len(labels) == rows
    assert {label.by for label in labels.values()} == {"judge"}

    # Rerunning the same command makes no call.
    before = len(chat.requests)
    again = invoke(*args)
    assert again.exit_code == 0, again.stderr
    assert len(chat.requests) == before
    assert output(again)["labels"]["applied"] == 0


def test_failed_calls_leave_rows_undecided_and_exit_1(dataset: Path, chat: FakeChatApi) -> None:
    chat.script += [httpx.Response(500), httpx.Response(500)]
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--concurrency", "1", "--confirm"]
    result = invoke(*args)
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["status"] == "incomplete"
    assert data["calls_failed"] == 2
    # One call at a time, in review order: both judges of the first row failed.
    assert data["undecided"] == 1
    assert data["stale_judge_labels"] == 0
    assert data["next_command"] == shlex.join(["pup", "finetune", *args])
    retried = invoke(*args)
    assert retried.exit_code == 0, retried.stderr
    assert output(retried)["status"] == "complete"
    assert output(retried)["calls_made"] == 2


def test_a_changed_rubric_needs_a_reset(dataset: Path, chat: FakeChatApi) -> None:
    base = ["dataset", "triage", str(dataset), "--max-rows", "2", "--judges", "j1"]
    assert invoke(*base, "--rule", "first", "--confirm").exit_code == 0
    changed = invoke(*base, "--rule", "second", "--confirm")
    assert changed.exit_code == 1
    assert "--reset-triage" in changed.stderr
    preview = invoke(*base, "--rule", "second", "--reset-triage")
    assert preview.exit_code == 2
    assert output(preview)["preview"]["reset"] == {"votes_discarded": 2, "stale_judge_labels": 0}
    assert "--reset-triage" in output(preview)["next_command"]
    reset = invoke(*base, "--rule", "second", "--reset-triage", "--confirm")
    assert reset.exit_code == 0, reset.stderr
    assert output(reset)["rows_reviewed"] == 2


JUDGE_J1 = ["--judges", "j1", "--confirm"]


def test_a_reset_sample_leaves_stale_judge_labels_and_asks_for_every_row(
    dataset: Path, chat: FakeChatApi
) -> None:
    rows = accepted(dataset)
    base = ["dataset", "triage", str(dataset), "--judges", "j1"]
    assert invoke(*base, "--rule", "first", "--confirm").exit_code == 0
    preview = invoke(*base, "--rule", "second", "--max-rows", "1", "--reset-triage")
    assert output(preview)["preview"]["reset"]["stale_judge_labels"] == rows - 1
    reset = invoke(*base, "--rule", "second", "--max-rows", "1", "--reset-triage", "--confirm")
    assert reset.exit_code == 0, reset.stderr
    data = output(reset)
    assert data["status"] == "incomplete"
    assert data["stale_judge_labels"] == rows - 1
    assert "judge label" in data["hint"]
    assert data["next_command"] == shlex.join(
        ["pup", "finetune", "dataset", "triage", str(dataset), "--rule", "second", *JUDGE_J1]
    )
    final = invoke(*base, "--rule", "second", "--confirm")
    assert final.exit_code == 0, final.stderr
    assert output(final)["status"] == "complete"
    assert output(final)["stale_judge_labels"] == 0
