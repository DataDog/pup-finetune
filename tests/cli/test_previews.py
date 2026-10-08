# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Previews as the approval: one shape (action, billing, rows, resolved settings, the
defaults used, an inputs digest), ``--expect`` refusing drift between the preview and the
confirm, and the card on stderr. Offline: the fake provider, judge, chat API and pup."""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from pup_finetune.cli import _common as common_module
from pup_finetune.cli import _summary as summary_module
from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.cli import triage as triage_cli
from pup_finetune.cli._summary import CARD_LINES
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.inference import ChatClient, ChatCompleter
from pup_finetune.runs import RunDir
from tests.fakes.chat_api import API_KEY, BASE_URL, FakeChatApi
from tests.fakes.datasets import FakeDatasetsApi
from tests.fakes.pup import FakePup

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
SHAPE = ("action", "billing", "rows", "settings", "defaults_used", "data_recipients")
"""What every preview of a paid or remote-changing command has."""

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def follow(next_command: str) -> Result:
    words = shlex.split(next_command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


def card(result: Result) -> list[str]:
    """The card: the stderr lines before the first warning, hint or ``Next`` line."""
    lines = result.stderr.splitlines()
    end = next(
        i for i, line in enumerate(lines) if line.startswith(("warning: ", "Hint: ", "Next: "))
    )
    return lines[:end]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def train_args(dataset: Path, run_dir: Path, *extra: str) -> list[str]:
    return [
        *("train", "--provider", "fake", "--model", "fake-model"),
        *("--dataset", str(dataset), "--run-dir", str(run_dir), *extra),
    ]


def drop_a_validation_row(dataset: Path) -> str:
    row = DatasetDir(dataset).rows_by_split()["validation"][0]
    labels = dataset.parent / "labels.jsonl"
    labels.write_text(json.dumps({"id": row.id, "keep": False, "reason": "drift test"}) + "\n")
    result = invoke("dataset", "label", str(dataset), "--file", str(labels))
    assert result.exit_code == 0, result.stderr
    return row.id


# --- train --------------------------------------------------------------------------------


def test_the_train_preview_shows_what_will_be_used(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--max-epochs", "3"))
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert all(key in preview for key in SHAPE)
    rows = preview["rows"]
    assert set(rows) == {"train", "validation", "test", "rejected"}
    assert preview["action"] == (
        f"fine-tune fake-model with LoRA rank 8 for up to 3 epochs on {rows['train']} rows, "
        f"validating on {rows['validation']}"
    )
    # The resolved values, with where they come from: never a null that hides one.
    settings = preview["settings"]
    assert settings["lora"] == {
        "rank": 8,
        "rank_source": "model default",
        "alpha": None,
        "alpha_source": "chosen by the provider",
    }
    assert (settings["max_seq_len"], settings["max_seq_len_source"]) == (4096, "model default")
    assert settings["max_epochs"] == 3
    assert "max_epochs" not in preview["defaults_used"]
    assert {"lora.rank", "learning_rate", "batch_size", "seed"} <= set(preview["defaults_used"])
    # The optimizer's fixed parameters and the full profile stay in run.json.
    assert "optimizer" not in settings and "spec" not in preview
    assert set(preview["profile"]) == {"alias", "base_model", "max_seq_len"}
    # The provider's billing sentence counts every token, validation included, and no price.
    billing = preview["billing"]
    assert f"{preview['tokens_per_epoch']:,} tokens per epoch" in billing
    assert f"{preview['validation_tokens']:,} validation tokens" in billing
    assert "no spend cap" in billing and "$" not in billing
    assert len(preview["inputs_digest"]) == 24


def test_values_the_user_typed_read_back_exactly(dataset: Path, tmp_path: Path) -> None:
    # Only measurements are rounded: a flag's value reads as typed, in settings and on the
    # card, and the next command carries it unchanged.
    flags = ("--max-spend-usd", "1234.5", "--hourly-rate-usd", "3.33333")
    result = invoke(*train_args(dataset, tmp_path / "run", *flags, "--learning-rate", "0.00012345"))
    data = output(result)
    settings = data["preview"]["settings"]
    assert settings["max_spend_usd"] == 1234.5
    assert settings["hourly_rate_usd"] == 3.33333
    assert settings["learning_rate"] == 0.00012345
    assert "--max-spend-usd 1234.5" in data["next_command"]
    assert "--learning-rate 0.00012345" in data["next_command"]
    assert "lr 1.2345e-4" in result.stderr
    status_dir = tmp_path / "run"
    assert follow(data["next_command"]).exit_code == 0
    status = output(invoke("status", str(status_dir)))
    assert status["settings"]["learning_rate"] == 0.00012345
    assert status["settings"]["max_spend_usd"] == 1234.5


def test_the_digest_is_stable_and_an_unchanged_confirm_trains(
    dataset: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    first = output(invoke(*train_args(dataset, run_dir, "--run-id", "ft-stable")))
    again = output(invoke(*train_args(dataset, run_dir, "--run-id", "ft-stable")))
    assert first["preview"]["inputs_digest"] == again["preview"]["inputs_digest"]
    words = shlex.split(first["next_command"])
    assert words[-3:] == ["--expect", first["preview"]["inputs_digest"], "--confirm"]
    confirmed = follow(first["next_command"])
    assert confirmed.exit_code == 0, confirmed.stderr
    manifest = RunDir(run_dir).manifest()
    assert manifest["settings"]["lora"]["rank"] == 8
    assert manifest["settings"]["optimizer"]["beta2"] == 0.95
    assert manifest["defaults_used"] == first["preview"]["defaults_used"]


def test_a_dataset_changed_after_the_preview_is_refused(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    data = output(invoke(*train_args(dataset, run_dir)))
    rows = data["preview"]["rows"]
    drop_a_validation_row(dataset)
    refused = follow(data["next_command"])
    assert refused.exit_code == 2
    assert refused.stdout == ""
    first = refused.stderr.splitlines()[0]
    assert first == (
        "Error [settings_conflict]: the dataset changed since the preview (now "
        f"{rows['train']} train / {rows['validation'] - 1} validation rows); preview again"
    )
    assert not run_dir.exists()  # nothing was trained
    # Without --expect, --confirm works as before.
    words = shlex.split(data["next_command"])
    index = words.index("--expect")
    assert follow(shlex.join(words[:index] + words[index + 2 :])).exit_code == 0


def test_expect_needs_confirm(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--expect", "0" * 24))
    assert result.exit_code == 2
    assert "--expect needs --confirm" in result.stderr


def test_the_train_card_is_short_and_generated_from_the_preview(
    dataset: Path, tmp_path: Path
) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run"))
    preview = output(result)["preview"]
    lines = card(result)
    assert 1 < len(lines) <= CARD_LINES
    assert (
        lines[0] == f"Preview only, nothing was run: train fake-model on fake in {tmp_path / 'run'}"
    )
    rows = preview["rows"]
    assert lines[1] == (
        f"  data      {rows['train']} train / {rows['validation']} validation rows; "
        f"{preview['tokens_per_epoch']:,} tokens per epoch, at most "
        f"{preview['max_tokens_total']:,} (+{preview['validation_tokens']:,} validation per epoch)"
    )
    assert lines[2] == (
        "  settings  up to 5 epochs (patience 1), batch 32, lr 1e-4, LoRA rank 8 (model default)"
    )
    # The offline fake provider bills nothing: no spend cap to suggest.
    assert lines[3:] == [
        "  sends     training rows to fake",
        "  cost      nothing: the fake provider runs offline",
    ]
    assert result.stderr.splitlines()[-1] == f"Next: {output(result)['next_command']}"
    assert "Confirmation required" not in result.stderr


def test_a_read_only_card_says_confirm_is_refused_and_never_cuts_the_cost(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    # A paid judge: the run leaves the machine, so --confirm is refused.
    judge = ("--eval-judge-model", "anthropic/claude-opus-5-5", "--eval-no-publish")
    result = invoke(*train_args(dataset, tmp_path / "run", "--evaluate", *judge))
    assert result.exit_code == 2
    assert output(result)["next_command"] is None
    lines = card(result)
    assert lines[1] == "  refused   --confirm is refused while PUP_READ_ONLY is set"
    # Too many rows for the card: the settings go, never who bills or what it costs.
    assert len(lines) == CARD_LINES
    assert [line.split()[0] for line in lines[1:]] == [
        "refused",
        "data",
        "evaluate",
        "sends",
        "bills",
        "cost",
    ]
    assert lines[-2] == (
        "  bills     Anthropic bills each judge call per token at its own rates "
        "(pup-finetune does not know the rates)"
    )
    assert lines[-1] == "  cost      nothing: the fake provider runs offline"
    assert "Next: " not in result.stderr


def test_a_read_only_run_that_stays_offline_is_allowed(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    result = invoke(*train_args(dataset, tmp_path / "run"))
    assert result.exit_code == 2
    assert output(result)["next_command"] is not None
    assert "refused" not in result.stderr


def _raises(*_args: object, **_kwargs: object) -> Any:
    raise ZeroDivisionError("division by zero")


def test_a_card_that_fails_never_costs_the_preview_or_its_exit_code(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(summary_module.PREVIEW_CARDS, "train", _raises)
    result = invoke(*train_args(dataset, tmp_path / "run"))
    assert result.exit_code == 2  # still a preview, not exit 1
    data = output(result)
    assert data["status"] == "confirmation_required"
    assert result.stderr.splitlines()[0].startswith("Preview only, nothing was run")
    assert result.stderr.splitlines()[-1] == f"Next: {data['next_command']}"
    assert "ZeroDivisionError" not in result.stderr
    monkeypatch.setenv("PUP_FINETUNE_DEBUG", "1")
    assert "ZeroDivisionError" in invoke(*train_args(dataset, tmp_path / "run")).stderr


def test_a_summary_or_stderr_that_fails_never_changes_the_exit_code(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(summary_module.SUMMARIES, "dataset show", _raises)
    shown = invoke("dataset", "show", str(dataset))
    assert shown.exit_code == 0, shown.stderr
    assert json.loads(shown.stdout)["dataset_dir"]
    # Even what prints the warnings and summary failing leaves the result and exit code.
    monkeypatch.setattr(common_module, "echo_warnings", _raises)
    again = invoke("dataset", "show", str(dataset))
    assert again.exit_code == 0, again.stderr
    assert again.stdout == shown.stdout


def test_quiet_keeps_only_the_card_s_first_line(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_QUIET", "1")
    result = invoke(*train_args(dataset, tmp_path / "run"))
    assert card(result) == [
        f"Preview only, nothing was run: train fake-model on fake in {tmp_path / 'run'}"
    ]


def test_train_evaluate_names_the_judge_in_its_action_and_billing(
    dataset: Path, tmp_path: Path
) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--evaluate"))
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert ", then evaluate " in preview["action"]
    # Each part of the billing names who bills it.
    assert preview["billing"].startswith("the fake provider is offline and bills nothing: ")
    assert "; the fake judge is offline and bills nothing: " in preview["billing"]
    assert all(key in preview["evaluation"] for key in ("action", "billing", "settings"))
    assert len(card(result)) <= CARD_LINES


# --- resume and status --------------------------------------------------------------------


@pytest.fixture
def crashed(dataset: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    result = invoke(
        *train_args(dataset, run_dir, "--batch-size", "2", "--max-epochs", "2"),
        *("--provider-option", "fail_at_step=5", "--confirm"),
    )
    assert result.exit_code == 1, result.stderr
    return run_dir


def test_the_resume_preview_has_the_shape_and_pins_the_run(crashed: Path) -> None:
    data = output(invoke("resume", str(crashed)))
    preview = data["preview"]
    assert all(key in preview for key in SHAPE)
    assert preview["settings"]["batch_size"] == 2
    assert "optimizer" not in preview["settings"]
    assert data["next_command"].endswith(f"--expect {preview['inputs_digest']} --confirm")
    resumed = follow(data["next_command"])
    assert resumed.exit_code == 0, resumed.stderr


def test_a_run_recorded_before_its_settings_still_previews_and_shows_its_status(
    crashed: Path,
) -> None:
    manifest = json.loads((crashed / "run.json").read_text(encoding="utf-8"))
    del manifest["settings"], manifest["defaults_used"]
    (crashed / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    status = output(invoke("status", str(crashed)))
    assert "settings" not in status
    preview = output(invoke("resume", str(crashed)))["preview"]
    assert preview["settings"]["lora"]["rank"] == 8  # resolved again from spec and profile
    assert preview["defaults_used"] == []


def test_status_shows_only_the_settings_that_differ_from_the_defaults(crashed: Path) -> None:
    status = output(invoke("status", str(crashed)))
    assert status["settings"] == {
        "max_epochs": 2,
        "batch_size": 2,
        "provider_options": {"fail_at_step": 5},
    }
    # Every setting, the optimizer's fixed parameters included, is in run.json.
    assert status["files"]["run"] == str(crashed / "run.json")
    recorded = RunDir(crashed).manifest()["settings"]
    assert recorded["optimizer"]["beta2"] == 0.95
    assert recorded["learning_rate"] == 1e-4


# --- dataset triage -----------------------------------------------------------------------


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> FakeChatApi:
    api = FakeChatApi(answer=lambda _body: json.dumps({"keep": True, "reason": "fine"}))

    @contextmanager
    def open_clients(judges: Sequence[str]) -> Iterator[Mapping[str, ChatCompleter]]:
        with ChatClient(
            BASE_URL, SecretStr(API_KEY), max_attempts=1, transport=api.transport
        ) as client:
            yield {"fireworks": client}

    monkeypatch.setattr(triage_cli, "open_clients", open_clients)
    return api


def triage(dataset: Path, *extra: str) -> Result:
    return invoke("dataset", "triage", str(dataset), "--rule", "stay on topic", *extra)


def test_the_triage_preview_lists_each_judge_once_and_estimates_tokens(dataset: Path) -> None:
    result = triage(dataset)
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert all(key in preview for key in SHAPE)
    assert "judge_endpoints" not in preview and "rubric_sha256" not in preview
    assert [judge["provider"] for judge in preview["judges"]] == ["fireworks", "fireworks"]
    assert len(preview["rubric_hash"]) == 12
    assert preview["approx_input_tokens"] > 0
    assert "approximate" in preview["approx_input_tokens_note"]
    assert preview["billing"].count("Fireworks bills") == 1
    lines = card(result)
    assert lines[0].startswith("Preview only, nothing was run: triage ")
    assert len(lines) <= CARD_LINES


def test_triage_refuses_rows_changed_after_the_preview(dataset: Path, chat: FakeChatApi) -> None:
    data = output(triage(dataset))
    drop_a_validation_row(dataset)  # a labeled row is not reviewed: one row fewer
    refused = follow(data["next_command"])
    assert refused.exit_code == 2
    assert "Error [settings_conflict]: the rows to review changed since the preview" in (
        refused.stderr
    )
    assert chat.requests == []
    again = output(triage(dataset))
    confirmed = follow(again["next_command"])
    assert confirmed.exit_code == 0, confirmed.stderr


# --- dataset push -------------------------------------------------------------------------


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePup:
    pup = FakePup()
    FakeDatasetsApi().install(pup)
    monkeypatch.setattr(dataset_cli, "Pup", pup.client)
    return pup


def push(dataset: Path, *extra: str) -> Result:
    return invoke("dataset", "push", str(dataset), "--name", "synthetic", *extra)


def test_the_push_preview_has_the_shape(dataset: Path, fake: FakePup) -> None:
    result = push(dataset)
    preview = output(result)["preview"]
    assert all(key in preview for key in SHAPE)
    assert preview["defaults_used"] == ["project"]
    assert "$" not in preview["billing"]
    assert card(result)[0] == (
        "Preview only, nothing was run: push "
        f"{preview['records_to_push']} records to Datadog dataset 'synthetic'"
    )


def test_push_refuses_records_changed_after_the_preview(dataset: Path, fake: FakePup) -> None:
    data = output(push(dataset))
    drop_a_validation_row(dataset)
    refused = follow(data["next_command"])
    assert refused.exit_code == 2
    assert (
        "Error [settings_conflict]: the records to push changed since the preview (now "
        f"{data['preview']['records_to_push'] - 1} records)"
    ) in refused.stderr
    assert fake.calls == []
    pushed = follow(output(push(dataset))["next_command"])
    assert pushed.exit_code == 0, pushed.stderr


# --- evaluate and deploy ------------------------------------------------------------------


@pytest.fixture
def completed(dataset: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "done"
    result = invoke(*train_args(dataset, run_dir, "--max-epochs", "1", "--confirm"))
    assert result.exit_code == 0, result.stderr
    return run_dir


def test_the_evaluate_preview_has_the_shape(completed: Path) -> None:
    result = invoke("evaluate", str(completed), "--seed", "3")
    preview = output(result)["preview"]
    assert all(key in preview for key in SHAPE)
    assert preview["settings"]["seed"] == 3
    assert "seed" not in preview["defaults_used"]
    assert "judge_model" in preview["defaults_used"]
    assert "; the fake judge is offline and bills nothing: " in preview["billing"]
    # Each setting once, under settings.
    assert not {"max_tokens", "concurrency", "seed", "max_cases_per_trajectory"} & set(preview)
    assert "max_cases" in preview["settings"]
    lines = card(result)
    assert lines[0].startswith("Preview only, nothing was run: evaluate run ")
    assert len(lines) <= CARD_LINES


def test_the_deploy_preview_has_settings_and_a_card(completed: Path) -> None:
    result = invoke("deploy", str(completed), "--timeout-minutes", "5")
    preview = output(result)["preview"]
    assert preview["settings"]["timeout_minutes"] == 5
    assert preview["defaults_used"] == ["deployment_id", "shape"]
    lines = card(result)
    assert lines[0].startswith("Preview only, nothing was run: deploy ")
    assert "wait up to 5 min" in lines[1]
    assert lines[2].startswith("  sends     the deployment request")
    assert lines[3] == "  cost      nothing: the fake provider runs offline"
