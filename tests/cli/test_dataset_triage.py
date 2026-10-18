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
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import triage as triage_cli
from pup_finetune.curation.triage import JUDGMENTS_FILE, rubric_template
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import display_path
from pup_finetune.inference import ChatClient, ChatCompleter
from pup_finetune.judge_models import FIREWORKS_JUDGES as DEFAULT_JUDGES
from tests.fakes.chat_api import API_KEY, BASE_URL, FakeChatApi, block

pytestmark = pytest.mark.usefixtures("judge_keys")

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
    def open_clients(judges: Sequence[str]) -> Iterator[Mapping[str, ChatCompleter]]:
        with ChatClient(
            BASE_URL, SecretStr(API_KEY), max_attempts=1, transport=api.transport
        ) as client:
            yield {"fireworks": client}

    monkeypatch.setattr(triage_cli, "open_clients", open_clients)
    return api


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def edited_rubric() -> str:
    """The rubric template with its Task section filled in, as a user would."""
    template = rubric_template()
    start = template.index("## Task") + len("## Task")
    end = template.index("## Keep")
    task = "\n\nA synthetic support agent that answers order questions with the order tool.\n\n"
    return template[:start] + task + template[end:]


@pytest.fixture
def rubric(tmp_path: Path) -> Path:
    path = tmp_path / "rubric.md"
    path.write_text(edited_rubric(), encoding="utf-8")
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
    assert preview["rows"]["to_review"] == rows
    assert preview["judge_calls"] == rows * len(DEFAULT_JUDGES)
    assert [judge["model"] for judge in preview["judges"]] == list(DEFAULT_JUDGES)
    assert len(preview["rubric_hash"]) == 12
    assert not {"rows_to_review", "rows_skipped_labeled"} & set(preview)  # under rows
    assert preview["approx_input_tokens"] > 0
    assert data["next_command"] == shlex.join(
        [
            *("pup", "finetune", "dataset", "triage", str(dataset), "--rubric", str(rubric)),
            *("--judges", ",".join(DEFAULT_JUDGES), "--confirm"),
        ]
    )
    assert preview["judges_source"] == "default: no Anthropic or OpenAI credential found"
    assert chat.requests == []
    assert not (dataset / JUDGMENTS_FILE).exists()


def test_an_unedited_template_rubric_warns_and_offers_no_confirm(
    dataset: Path, tmp_path: Path, chat: FakeChatApi
) -> None:
    template = tmp_path / "rubric.md"
    template.write_text(rubric_template(), encoding="utf-8")
    args = ["dataset", "triage", str(dataset), "--rubric", str(template), "--max-rows", "2"]
    result = invoke(*args)
    assert result.exit_code == 2, result.stderr
    data = output(result)
    warning = f"{template} still has the template's Task text; describe what the agent is for first"
    assert warning in data["preview"]["warnings"]
    assert data["next_command"] is None
    assert "--allow-template-rubric" in result.stderr
    # Why --confirm is refused, and what to change, are in the result too.
    assert data["preview"]["refused"] == (
        "--confirm is refused while the rubric's Task section is the template's"
    )
    assert "--allow-template-rubric" in data["preview"]["refused_hint"]

    refused = invoke(*args, "--confirm")
    assert refused.exit_code == 2
    assert refused.stdout == ""
    assert warning in refused.stderr
    assert chat.requests == []
    assert not (dataset / JUDGMENTS_FILE).exists()

    allowed = invoke(*args, "--allow-template-rubric")
    assert allowed.exit_code == 2, allowed.stderr
    assert warning in output(allowed)["preview"]["warnings"]
    assert "refused" not in output(allowed)["preview"]
    next_command = output(allowed)["next_command"]
    assert "--allow-template-rubric" in shlex.split(next_command)
    assert shlex.split(next_command)[-1] == "--confirm"
    judged = invoke(*shlex.split(next_command)[2:])
    assert judged.exit_code == 0, judged.stderr
    assert chat.requests


def test_an_edited_rubric_has_no_template_warning(
    dataset: Path, rubric: Path, chat: FakeChatApi
) -> None:
    result = invoke("dataset", "triage", str(dataset), "--rubric", str(rubric))
    assert result.exit_code == 2, result.stderr
    assert "warnings" not in output(result)["preview"]


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
    args += ["--judges", "fireworks/judge-a,fireworks/judge-b,fireworks/judge-c", "--confirm"]
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
    assert data["next_command"] is None
    assert [step["command"] for step in data["next_steps"]] == [
        "pup finetune models list",
        f"pup finetune dataset push {dataset} --name <NAME>",
    ]
    assert data["stale_judge_labels"] == 0
    assert data["dataset"]["dropped"] == data["dropped"]
    assert data["pipeline"] == {
        "rows": {
            "accepted": rows,
            "kept": data["kept"],
            "dropped": data["dropped"],
            "unlabeled": 0,
        },
        "triage": "complete",
        "push": {"status": "not_pushed", "pending": data["kept"], "changed": 0},
    }
    # The rubric is named as the preview names it.
    assert len(data["rubric_hash"]) == 12 and "rubric_sha256" not in data
    assert {body["model"] for body in chat.bodies} == {
        "fireworks/judge-a",
        "fireworks/judge-b",
        "fireworks/judge-c",
    }
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
    args += ["--judges", ",".join(DEFAULT_JUDGES), "--concurrency", "1", "--confirm"]
    result = invoke(*args)
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["status"] == "incomplete"
    assert data["calls_failed"] == 2
    # One call at a time, in review order: both judges of the first row failed.
    assert data["undecided"] == 1
    assert data["stale_judge_labels"] == 0
    assert data["pipeline"]["triage"] == "incomplete"
    assert data["pipeline"]["rows"]["unlabeled"] == 1
    assert data["next_command"] == shlex.join(["pup", "finetune", *args])
    retried = invoke(*args)
    assert retried.exit_code == 0, retried.stderr
    assert output(retried)["status"] == "complete"
    assert output(retried)["pipeline"]["triage"] == "complete"
    assert output(retried)["calls_made"] == 2


def test_a_changed_rubric_needs_a_reset(dataset: Path, chat: FakeChatApi) -> None:
    base = ["dataset", "triage", str(dataset), "--max-rows", "2", "--judges", "fireworks/j1"]
    first = invoke(*base, "--rule", "first", "--confirm")
    assert first.exit_code == 0
    # The run is complete, but the dataset's triage is not: --max-rows left rows unjudged.
    assert output(first)["status"] == "complete"
    assert output(first)["pipeline"]["triage"] == "incomplete"
    assert f"pup finetune dataset triage {dataset} --rubric <RUBRIC_FILE>" in [
        step["command"] for step in output(first)["next_steps"]
    ]
    changed = invoke(*base, "--rule", "second", "--confirm")
    assert changed.exit_code == 2
    assert "Error [settings_conflict]: " in changed.stderr
    assert "--reset-triage" in changed.stderr
    preview = invoke(*base, "--rule", "second", "--reset-triage")
    assert preview.exit_code == 2
    assert output(preview)["preview"]["reset"] == {"votes_discarded": 2, "stale_judge_labels": 0}
    assert "--reset-triage" in output(preview)["next_command"]
    reset = invoke(*base, "--rule", "second", "--reset-triage", "--confirm")
    assert reset.exit_code == 0, reset.stderr
    assert output(reset)["rows_reviewed"] == 2


JUDGE_J1 = ["--judges", "fireworks/j1", "--confirm"]


def test_a_reset_sample_leaves_stale_judge_labels_and_asks_for_every_row(
    dataset: Path, chat: FakeChatApi
) -> None:
    rows = accepted(dataset)
    base = ["dataset", "triage", str(dataset), "--judges", "fireworks/j1"]
    assert invoke(*base, "--rule", "first", "--confirm").exit_code == 0
    preview = invoke(*base, "--rule", "second", "--max-rows", "1", "--reset-triage")
    assert output(preview)["preview"]["reset"]["stale_judge_labels"] == rows - 1
    reset = invoke(*base, "--rule", "second", "--max-rows", "1", "--reset-triage", "--confirm")
    assert reset.exit_code == 0, reset.stderr
    data = output(reset)
    assert data["status"] == "incomplete"
    assert data["stale_judge_labels"] == rows - 1
    assert data["pipeline"]["triage"] == "incomplete"  # every row labeled, most labels stale
    assert data["pipeline"]["rows"]["unlabeled"] == 0
    assert "judge label" in data["hint"]
    assert data["next_command"] == shlex.join(
        ["pup", "finetune", "dataset", "triage", str(dataset), "--rule", "second", *JUDGE_J1]
    )
    final = invoke(*base, "--rule", "second", "--confirm")
    assert final.exit_code == 0, final.stderr
    assert output(final)["status"] == "complete"
    assert output(final)["stale_judge_labels"] == 0
    assert output(final)["pipeline"]["triage"] == "complete"


@pytest.mark.usefixtures("all_judge_keys")
def test_a_mixed_council_previews_every_provider_and_routes_each_judge(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    apis = {name: FakeChatApi(answer=drop_weather) for name in ("fireworks", "anthropic")}
    opened: list[list[str]] = []

    @contextmanager
    def open_clients(judges: Sequence[str]) -> Iterator[Mapping[str, ChatCompleter]]:
        opened.append(list(judges))
        with (
            ChatClient(BASE_URL, SecretStr(API_KEY), transport=apis["fireworks"].transport) as fw,
            ChatClient(BASE_URL, SecretStr(API_KEY), transport=apis["anthropic"].transport) as an,
        ):
            yield {"fireworks": fw, "anthropic": an}

    monkeypatch.setattr(triage_cli, "open_clients", open_clients)
    judges = "accounts/fireworks/models/judge-a,anthropic/claude-opus-5-5"
    base = ["dataset", "triage", str(dataset), "--rule", "stay on topic", "--judges", judges]
    preview = invoke(*base)
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    council = ["accounts/fireworks/models/judge-a", "anthropic/claude-opus-5-5"]
    # Each judge once, with its provider.
    assert [(e["model"], e["provider"]) for e in data["preview"]["judges"]] == [
        (council[0], "fireworks"),
        (council[1], "anthropic"),
    ]
    assert "Fireworks (api.fireworks.ai)" in data["preview"]["data_destination"]
    assert "Anthropic (api.anthropic.com)" in data["preview"]["data_destination"]
    words = shlex.split(data["next_command"])
    assert words[words.index("--judges") + 1] == ",".join(council)
    assert opened == []

    result = invoke(*base, "--confirm")
    assert result.exit_code == 0, result.stderr
    assert opened == [council]
    rows = accepted(dataset)
    assert {body["model"] for body in apis["fireworks"].bodies} == {council[0]}
    assert {body["model"] for body in apis["anthropic"].bodies} == {"claude-opus-5-5"}
    assert len(apis["fireworks"].bodies) == len(apis["anthropic"].bodies) == rows


def test_a_judge_with_no_model_is_a_usage_error(dataset: Path, chat: FakeChatApi) -> None:
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--judges", "openai/")
    assert result.exit_code == 2
    assert "has no model" in result.stderr
    assert chat.requests == []


def test_an_anthropic_judge_without_the_extra_fails_before_the_preview(
    dataset: Path, chat: FakeChatApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    args = [
        "dataset",
        "triage",
        str(dataset),
        "--rule",
        "x",
        "--judges",
        "anthropic/claude-opus-5-5",
    ]
    result = invoke(*args)
    assert result.exit_code == 2
    assert "uv sync --extra anthropic" in result.stderr
    assert "pup-finetune[anthropic]" in result.stderr


def test_a_row_too_long_for_its_judge_is_dropped_and_counted(
    dataset: Path, chat: FakeChatApi
) -> None:
    too_long = {
        "error": {
            "message": "The prompt is too long: 2200031, model maximum context length: 1048571"
        }
    }
    chat.script.append(httpx.Response(400, json=too_long))
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--judges", "fireworks/judge-a", "--concurrency", "1", "--confirm"]
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["too_long"] == 1
    assert data["calls_failed"] == 0
    assert any("context window" in item["reason"] for item in data["top_drop_reasons"])
    before = len(chat.requests)
    assert invoke(*args).exit_code == 0
    assert len(chat.requests) == before  # the row is not sent again


def test_agreement_compares_the_council_with_the_user_s_labels(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.jsonl"
    rows = {
        "row-weather-001": (True, ""),  # the council drops weather rows: a disagreement
        "row-weather-003": (False, "off topic"),
        "row-calendar-001": (True, ""),
        "row-calendar-002": (False, "too vague"),  # the council keeps it: a disagreement
    }
    labels.write_text(
        "".join(
            json.dumps({"id": row_id, "keep": keep, "reason": reason}) + "\n"
            for row_id, (keep, reason) in rows.items()
        )
    )
    labeled = invoke("dataset", "label", str(dataset), "--file", str(labels))
    assert labeled.exit_code == 0
    labels_before = (dataset / "labels.jsonl").read_text()
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--judges", "fireworks/judge-a,fireworks/judge-b", "--agreement"]

    preview = invoke(*args)
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["mode"] == "agreement"
    assert output(preview)["preview"]["rows"]["to_review"] == 4
    noisy = (
        "the agreement is measured on 4 rows you or the agent labeled; on fewer than 20 it is "
        "noisy (one row moves it by 25 points)"
    )
    assert any(w.startswith(noisy) for w in output(preview)["preview"]["warnings"])

    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["mode"] == "agreement"
    assert data["status"] == "complete"
    assert any(w.startswith(noisy) for w in data["warnings"])
    assert f"warning: {noisy}" in result.stderr
    agreement = data["agreement"]
    assert (agreement["rows_compared"], agreement["rows_agreed"], agreement["rate"]) == (4, 2, 0.5)
    assert agreement["confusion"] == {
        "keep_keep": 1,
        "keep_drop": 1,
        "drop_keep": 1,
        "drop_drop": 1,
    }
    assert "by" not in agreement
    assert set(agreement["judges"]) == {"fireworks/judge-a", "fireworks/judge-b"}
    assert sorted(item["row_id"] for item in agreement["disagreements_shown"]) == [
        "row-calendar-002",
        "row-weather-001",
    ]
    assert len(chat.requests) == 8
    assert (dataset / "labels.jsonl").read_text() == labels_before  # nothing labeled
    without = [word for word in args if word != "--agreement"]
    assert data["next_command"] == shlex.join(["pup", "finetune", *without, "--confirm"])
    assert data["next_steps"] == [
        {
            "command": shlex.join(["pup", "finetune", *args, "--reset-triage"]),
            "why": data["next_steps"][0]["why"],
        }
    ]

    # Labeling the rest afterwards reviews only the unlabeled rows.
    labeled = invoke(*without, "--confirm")
    assert labeled.exit_code == 0, labeled.stderr
    assert output(labeled)["rows_reviewed"] == accepted(dataset) - 4
    assert output(labeled)["rows_skipped_labeled"] == 4


def test_agreement_without_labels_is_a_usage_error(dataset: Path, chat: FakeChatApi) -> None:
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--agreement")
    assert result.exit_code == 2
    assert "label" in result.stderr
    assert chat.requests == []


def test_a_judge_alias_is_previewed_and_frozen_as_its_full_id(
    dataset: Path, chat: FakeChatApi
) -> None:
    full = "accounts/fireworks/models/glm-5p3-flash"
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    preview = invoke(*args, "--judges", "glm-5p3-flash")
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    assert [judge["model"] for judge in data["preview"]["judges"]] == [full]
    assert f"--judges {full}" in data["next_command"]
    result = invoke(*args, "--judges", "glm-5p3-flash", "--confirm")
    assert result.exit_code == 0, result.stderr
    frozen = json.loads((dataset / "triage.json").read_text(encoding="utf-8"))
    assert frozen["judges"] == [full]
    assert {body["model"] for body in chat.bodies} == {full}
    # The full id is the same judge: the rerun resumes and makes no call.
    before = len(chat.requests)
    again = invoke(*args, "--judges", full, "--confirm")
    assert again.exit_code == 0, again.stderr
    assert len(chat.requests) == before


def test_the_card_names_the_rubric_file_the_rules_and_the_hash(
    dataset: Path, rubric: Path, chat: FakeChatApi
) -> None:
    args = ["dataset", "triage", str(dataset), "--rubric", str(rubric)]
    result = invoke(*args, "--rule", "stay on topic", "--rule", "no secrets")
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert (preview["rubric_file"], preview["rule_count"]) == (str(rubric), 2)
    rubric_line = f"  rubric    rubric {rubric} + 2 rules (hash {preview['rubric_hash']})"
    assert rubric_line in result.stderr.splitlines()

    rules_only = invoke("dataset", "triage", str(dataset), "--rule", "stay on topic")
    hash_shown = output(rules_only)["preview"]["rubric_hash"]
    assert f"  rubric    1 rule (hash {hash_shown})" in rules_only.stderr.splitlines()


def row_file(tmp_path: Path, *row_ids: str) -> Path:
    path = tmp_path / "rows.txt"
    path.write_text("".join(f"{row_id}\n\n" for row_id in row_ids))  # blank lines are ignored
    return path


def test_rows_narrows_the_review_and_the_calls_previewed(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    subset = row_file(tmp_path, "row-weather-001", "row-calendar-001", "row-calendar-002")
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--judges", ",".join(DEFAULT_JUDGES), "--rows", str(subset)]
    preview = output(invoke(*args))
    assert preview["preview"]["rows"]["to_review"] == 3
    assert preview["preview"]["judge_calls"] == 3 * len(DEFAULT_JUDGES)
    assert preview["preview"]["settings"]["rows_file"] == str(subset)
    assert preview["next_command"] == shlex.join(["pup", "finetune", *args, "--confirm"])
    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["rows_reviewed"] == 3
    labeled = {row_id for row_id, label in DatasetDir(dataset).labels().items()}
    assert labeled == {"row-weather-001", "row-calendar-001", "row-calendar-002"}
    rest = output(invoke("dataset", "triage", str(dataset), "--rule", "stay on topic"))
    assert rest["preview"]["rows"]["to_review"] == accepted(dataset)
    assert rest["preview"]["votes_reused"] == 3 * len(DEFAULT_JUDGES)
    assert rest["preview"]["judge_calls_pending"] == (accepted(dataset) - 3) * len(DEFAULT_JUDGES)


def test_rows_skips_the_rows_you_labeled_and_agreement_compares_only_those(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"id": row_id, "keep": keep, "reason": "synthetic"}) + "\n"
            for row_id, keep in [
                ("row-weather-001", True),
                ("row-weather-003", False),
                ("row-calendar-001", True),
            ]
        )
    )
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    subset = row_file(tmp_path, "row-weather-001", "row-calendar-001", "row-calendar-002")
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic", "--rows", str(subset)]
    preview = output(invoke(*args))["preview"]
    assert (preview["rows"]["to_review"], preview["rows"]["skipped_labeled"]) == (1, 2)
    compared = invoke(*args, "--agreement", "--confirm")
    assert compared.exit_code == 0, compared.stderr
    data = output(compared)
    assert data["agreement"]["rows_compared"] == 2
    assert "--rows" not in shlex.split(data["next_command"])


def test_agreement_s_next_command_keeps_a_rule_that_reads_like_a_flag(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "row-weather-001", "keep": True}) + "\n")
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    args = ["dataset", "triage", str(dataset), "--rule=--agreement", "--max-rows", "3"]
    compared = invoke(*args, "--agreement", "--confirm")
    assert compared.exit_code == 0, compared.stderr
    words = shlex.split(output(compared)["next_command"])
    assert words[words.index("--rule") + 1] == "--agreement"
    assert words.count("--agreement") == 1
    assert words[words.index("--max-rows") + 1] == "3"


def test_rows_naming_only_rows_you_labeled_offers_no_confirm(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "row-weather-001", "keep": True}) + "\n")
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    subset = row_file(tmp_path, "row-weather-001")
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--rows", str(subset))
    assert result.exit_code == 2
    data = output(result)
    assert data["preview"]["rows"] == {"to_review": 0, "skipped_labeled": 1}
    assert data["next_command"] is None
    assert data["preview"]["refused"] == (
        "there is no row to review: every row --rows names is labeled by you or the agent"
    )
    assert chat.requests == []


def test_rows_naming_an_id_that_is_no_accepted_row_is_a_usage_error(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    subset = row_file(tmp_path, "row-weather-001", "synthetic-missing-1", "synthetic-missing-2")
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--rows", str(subset))
    assert result.exit_code == 2
    assert result.stderr.startswith(
        "Error [usage]: --rows names 2 ids that are not accepted rows of "
        f"{dataset}: 'synthetic-missing-1', 'synthetic-missing-2'\n"
    )
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--rows", str(empty))
    assert result.exit_code == 2
    assert "--rows names no row id" in result.stderr
    assert chat.requests == []


def test_agreement_on_rows_none_of_which_you_labeled_is_a_usage_error(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    subset = row_file(tmp_path, "row-weather-001")
    result = invoke(
        "dataset", "triage", str(dataset), "--rule", "x", "--rows", str(subset), "--agreement"
    )
    assert result.exit_code == 2
    assert "that --rows names carries a label" in result.stderr


ROW_CHECKS = ROWS.parent.parent / "evaluators" / "row_checks.py"
CHECKED = {"row-plain-001", "row-inventory-001", "row-weather-004"}


def test_checks_alone_label_at_once_with_no_judge_and_no_key(
    dataset: Path, chat: FakeChatApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    result = invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS))
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["mode"] == "checks"
    assert data["checks"] == {
        "file": display_path(ROW_CHECKS),
        "sha256": data["checks"]["sha256"],
        "functions": ["declares_tools", "no_tool_errors"],
        "rows_checked": accepted(dataset),
        "failed": 3,
        "failed_by_check": {"declares_tools": 1, "no_tool_errors": 2},
        "labels": {"applied": 3, "unchanged": 0},
    }
    assert data["stale_check_labels"] == 0
    assert "checks: 3 of 14 rows failed, 3 newly labeled drop" in result.stderr
    assert chat.requests == []
    labels = DatasetDir(dataset).labels()
    assert set(labels) == CHECKED
    assert labels["row-plain-001"].reason == "check declares_tools: the row declares no tools"
    assert {label.by for label in labels.values()} == {"check"}
    council = shlex.join(["pup", "finetune", "dataset", "triage", str(dataset)])
    council += f" --checks {ROW_CHECKS} --rubric <RUBRIC_FILE>"
    assert data["next_steps"][0]["command"] == council
    assert not (dataset / JUDGMENTS_FILE).exists()
    again = invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS), "--confirm")
    assert again.exit_code == 0, again.stderr
    assert output(again)["checks"]["labels"] == {"applied": 0, "unchanged": 3}


@pytest.mark.parametrize(
    "extra", [["--judges", "fireworks/j1"], ["--concurrency", "2"], ["--reset-triage"]]
)
def test_checks_alone_refuse_judge_flags(
    dataset: Path, chat: FakeChatApi, extra: list[str]
) -> None:
    flag = extra[0]
    result = invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS), *extra)
    assert result.exit_code == 2
    assert f"{flag} needs --rubric or --rule (a run with --checks alone calls no judge)" in (
        result.stderr
    )
    assert not (dataset / "labels.jsonl").exists()


def test_checks_run_first_and_the_council_judges_only_the_rest(
    dataset: Path, chat: FakeChatApi
) -> None:
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--checks", str(ROW_CHECKS), "--judges", "fireworks/j1"]
    preview = invoke(*args)
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    rows = accepted(dataset)
    assert data["preview"]["judge_calls"] == rows - 3
    assert data["preview"]["checks"]["failed"] == 3
    assert "labels" not in data["preview"]["checks"]
    assert data["preview"]["action"].startswith("label the 3 rows the checks fail (free), have")
    assert data["next_command"] == shlex.join(["pup", "finetune", *args, "--confirm"])
    assert "  checks    3 of 14 rows fail: labeled drop, never judged" in preview.stderr
    assert not (dataset / "labels.jsonl").exists()

    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["rows_reviewed"] == rows - 3
    assert data["checks"]["labels"] == {"applied": 3, "unchanged": 0}
    assert "too long, 3 dropped by checks;" in result.stderr
    judged = {
        json.loads(line)["row_id"] for line in (dataset / JUDGMENTS_FILE).read_text().splitlines()
    }
    assert judged.isdisjoint(CHECKED) and len(judged) == rows - 3
    by = {row_id: label.by for row_id, label in DatasetDir(dataset).labels().items()}
    assert {row_id for row_id, source in by.items() if source == "check"} == CHECKED
    assert data["pipeline"]["triage"] == "complete"

    without = invoke("dataset", "triage", str(dataset), "--rule", "stay on topic")
    assert output(without)["preview"]["rows"]["skipped_checked"] == 3


def test_a_checked_sample_finishes_with_the_checks(dataset: Path, chat: FakeChatApi) -> None:
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--checks", str(ROW_CHECKS), "--judges", "fireworks/j1"]
    result = invoke(*args, "--max-rows", "4", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert (data["status"], data["pipeline"]["triage"]) == ("complete", "incomplete")
    commands = [entry["command"] for entry in data["next_steps"]]
    assert shlex.join(["pup", "finetune", *args]) in commands
    assert not any(command.endswith(" --rubric <RUBRIC_FILE>") for command in commands)


def test_dataset_show_finishes_a_checked_sample_with_its_checks(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    checks = tmp_path / "checks.py"
    checks.write_text(ROW_CHECKS.read_text())
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--judges", "fireworks/j1", "--max-rows", "4", "--confirm"]
    assert invoke(*args, "--checks", str(ROW_CHECKS)).exit_code == 0
    # The checks file is recorded but not frozen: another one resumes the same triage.
    resumed = invoke(*args, "--checks", str(checks))
    assert resumed.exit_code == 0, resumed.stderr
    assert json.loads((dataset / "triage.json").read_text())["checks_file"] == str(checks)
    finish = shlex.join(["pup", "finetune", "dataset", "triage", str(dataset)])
    finish += " --rubric <RUBRIC_FILE>"
    steps = output(invoke("dataset", "show", str(dataset)))["next_steps"]
    assert f"{finish} --checks {shlex.quote(str(checks))}" in [s["command"] for s in steps]
    checks.unlink()
    steps = output(invoke("dataset", "show", str(dataset)))["next_steps"]
    entry = next(s for s in steps if s["command"].startswith(finish))
    assert entry["command"] == finish
    assert f"{checks}, which no longer exists" in entry["why"]


def test_an_edited_check_file_reports_stale_check_labels(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    assert invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS)).exit_code == 0
    edited = tmp_path / "row_checks.py"
    edited.write_text(ROW_CHECKS.read_text().split("\n\ndef no_tool_errors")[0] + "\n")
    subset = row_file(tmp_path, "row-plain-001", "row-inventory-001", "row-weather-001")
    args = ["dataset", "triage", str(dataset), "--checks", str(edited), "--rows", str(subset)]
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert (data["checks"]["rows_checked"], data["checks"]["failed"]) == (3, 1)
    assert (data["stale_check_labels"], data["stale_check_label_ids"]) == (
        1,
        ["row-inventory-001"],
    )
    assert "dataset label" in data["hint"]


def test_agreement_reports_the_checks_apart_from_the_council(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"id": row_id, "keep": keep, "reason": "synthetic"}) + "\n"
            for row_id, keep in [
                ("row-plain-001", False),
                ("row-inventory-001", True),
                ("row-calendar-001", True),
            ]
        )
    )
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic"]
    args += ["--checks", str(ROW_CHECKS), "--agreement"]
    preview = invoke(*args)
    assert preview.exit_code == 2, preview.stderr
    card = "  checks    2 of 3 labeled rows fail: compared with your labels apart, never judged"
    assert card in preview.stderr
    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    agreement = output(result)["agreement"]
    assert agreement["rows_compared"] == 1
    assert agreement["checks"] == {
        "declares_tools": {"failed": 1, "agree": 1},
        "no_tool_errors": {"failed": 1, "agree": 0},
    }
    alone = invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS), "--agreement")
    assert alone.exit_code == 0, alone.stderr
    assert output(alone)["agreement"]["checks"] == agreement["checks"]
    assert "checks: 2 of 3 labeled rows failed, nothing labeled; checks against" in alone.stderr
    council = output(alone)["next_steps"][0]
    assert council["command"].endswith(" --agreement --rubric <RUBRIC_FILE>")
    assert council["why"].startswith("measure the judge council against your labels")
    assert {label.by for label in DatasetDir(dataset).labels().values()} == {"manual"}


@pytest.mark.parametrize("confirm", [[], ["--confirm"]])
def test_an_agreement_whose_labeled_rows_all_fail_a_check_is_refused(
    dataset: Path, chat: FakeChatApi, tmp_path: Path, confirm: list[str]
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "row-plain-001", "keep": False, "reason": "x"}) + "\n")
    assert invoke("dataset", "label", str(dataset), "--file", str(labels)).exit_code == 0
    args = ["dataset", "triage", str(dataset), "--rule", "stay on topic", "--agreement"]
    result = invoke(*args, "--checks", str(ROW_CHECKS), "--judges", "fireworks/j1", *confirm)
    assert result.exit_code == 2
    assert result.stderr.startswith("Error [usage]: the checks fail every row labeled by you")
    assert chat.requests == []
    assert not (dataset / "triage.json").exists()
    alone = invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS), "--agreement")
    assert alone.exit_code == 0, alone.stderr
    assert not any("--rubric" in entry["command"] for entry in output(alone)["next_steps"])


def test_a_check_that_raises_fails_the_preview_before_any_call(
    dataset: Path, chat: FakeChatApi, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(row):\n    return row['missing']\n")
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--checks", str(broken))
    assert result.exit_code == 2
    assert result.stderr.startswith("Error [usage]: the check broken failed on the row ")
    assert "KeyError" in result.stderr
    assert chat.requests == []
    assert not (dataset / JUDGMENTS_FILE).exists()


def test_dataset_show_never_loads_the_checks(
    dataset: Path, chat: FakeChatApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke("dataset", "triage", str(dataset), "--checks", str(ROW_CHECKS)).exit_code == 0

    def refuse(path: Path) -> None:
        raise AssertionError("user code loaded")

    monkeypatch.setattr("pup_finetune.user_checks.load_checks", refuse)
    shown = invoke("dataset", "show", str(dataset))
    assert shown.exit_code == 0, shown.stderr
    assert output(shown)["pipeline"]["rows"]["dropped"] == 3
