# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Options that do nothing without another one are refused, never silently ignored."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
import typer.main
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import train as train_module
from pup_finetune.cli import triage as triage_module
from pup_finetune.cli._flags import flag_name

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    assert invoke("dataset", "import", str(path), "--file", str(ROWS)).exit_code == 0
    return path


def train_args(dataset: Path, run_dir: Path, *extra: str) -> list[str]:
    return [
        "train",
        "--provider",
        "fake",
        "--model",
        "fake-model",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        *extra,
    ]


def _command(*path: str) -> Any:
    command: Any = typer.main.get_command(app)
    ctx = typer.Context(command, info_name="pup finetune")
    for name in path:
        command = command.get_command(ctx, name)
        assert command is not None, name
    return command


EVAL_FLAGS: list[tuple[str, ...]] = [
    ("--eval-out-dir", "x"),
    ("--eval-max-cases", "3"),
    ("--eval-max-cases-per-trajectory", "3"),  # the old name, hidden
    ("--eval-judge-model", "anthropic/claude-opus-5-5"),
    ("--eval-concurrency", "8"),
    ("--eval-max-tokens", "256"),
    ("--eval-seed", "1"),
    ("--eval-skip-calibration",),
    ("--eval-no-publish",),
]


@pytest.mark.parametrize("flag", EVAL_FLAGS, ids=lambda flag: flag[0])
def test_every_eval_option_needs_evaluate(
    dataset: Path, tmp_path: Path, flag: tuple[str, ...]
) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", *flag))
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"{flag[0]} needs --evaluate" in result.stderr
    assert not (tmp_path / "run").exists()


def test_an_eval_option_at_its_default_value_still_needs_evaluate(
    dataset: Path, tmp_path: Path
) -> None:
    # Typed, so the user meant it: the default value does not make it silently ignored.
    result = invoke(*train_args(dataset, tmp_path / "run", "--eval-concurrency", "4"))
    assert result.exit_code == 2
    assert "--eval-concurrency needs --evaluate" in result.stderr


def test_several_eval_options_are_named_together(dataset: Path, tmp_path: Path) -> None:
    extra = ("--eval-judge-model", "anthropic/claude-opus-5-5", "--eval-concurrency", "8")
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2
    assert "--eval-judge-model and --eval-concurrency need --evaluate" in result.stderr


def test_eval_options_with_evaluate_are_accepted(dataset: Path, tmp_path: Path) -> None:
    extra = ("--evaluate", "--eval-concurrency", "8", "--eval-seed", "1")
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2, result.stderr
    assert output(result)["status"] == "confirmation_required"


def test_the_spend_reserve_needs_a_spend_cap(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--spend-reserve-fraction", "0.2"))
    assert result.exit_code == 2
    assert "--spend-reserve-fraction needs --max-spend-usd" in result.stderr
    capped = invoke(
        *train_args(
            dataset,
            tmp_path / "run",
            "--spend-reserve-fraction",
            "0.2",
            "--max-spend-usd",
            "5",
            "--hourly-rate-usd",
            "10",
        )
    )
    assert capped.exit_code == 2, capped.stderr
    assert output(capped)["preview"]["settings"]["spend_reserve_fraction"] == 0.2


def test_a_falsy_enabler_value_still_counts_as_given(dataset: Path, tmp_path: Path) -> None:
    # --max-spend-usd 0 was passed: its own validation reports it, not the reserve's check.
    extra = ("--max-spend-usd", "0", "--spend-reserve-fraction", "0.1")
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2
    assert "--spend-reserve-fraction needs --max-spend-usd" not in result.stderr
    assert "--max-spend-usd" in result.stderr


def test_allow_template_rubric_needs_a_rubric(dataset: Path) -> None:
    result = invoke("dataset", "triage", str(dataset), "--rule", "x", "--allow-template-rubric")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--allow-template-rubric needs --rubric" in result.stderr


@pytest.mark.usefixtures("judge_keys")
def test_allow_template_rubric_is_not_repeated_for_an_edited_rubric(
    dataset: Path, tmp_path: Path
) -> None:
    rubric = tmp_path / "rubric.md"
    rubric.write_text("# Task\n\nA weather assistant.\n\n# Keep\n\nCorrect tool calls.\n")
    result = invoke(
        "dataset", "triage", str(dataset), "--rubric", str(rubric), "--allow-template-rubric"
    )
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["status"] == "confirmation_required"
    assert "--allow-template-rubric" not in data["next_command"]


@pytest.mark.parametrize(
    ("path", "needs"),
    [
        (("train",), train_module.TRAIN_NEEDS),
        (("status",), train_module.STATUS_NEEDS),
        (("dataset", "triage"), triage_module.TRIAGE_NEEDS),
    ],
)
def test_each_dependent_option_s_help_names_what_it_needs(
    path: tuple[str, ...], needs: dict[str, str]
) -> None:
    command = _command(*path)
    params = {param.name: param for param in command.params}
    for name, enabler in needs.items():
        assert name in params and enabler in params, name
        param = params[name]
        needed = flag_name(command, enabler)
        help_text = getattr(param, "help", None) or ""
        panel = getattr(param, "rich_help_panel", None) or ""
        if getattr(param, "hidden", False):
            assert help_text.startswith(f"With {needed}:"), (name, help_text)
        else:
            # Its help, or the title of the panel it is listed in, names what it needs.
            assert needed in help_text or f"(with {needed})" in panel, (name, help_text, panel)


@pytest.mark.parametrize(
    "extra",
    [("ds",), ("--rule", "stay on topic"), ("--max-rows", "5"), ("--confirm",)],
    ids=lambda extra: extra[0],
)
def test_print_template_takes_nothing_else(extra: tuple[str, ...]) -> None:
    result = invoke("dataset", "triage", "--print-template", *extra)
    assert result.exit_code == 2
    assert result.stdout == ""
    shown = "DIR" if extra[0] == "ds" else extra[0]
    assert f"--print-template takes nothing else, but {shown} was given too" in result.stderr


def test_cancel_s_confirm_says_what_it_does() -> None:
    confirm = next(param for param in _command("cancel").params if param.name == "confirm")
    help_text = getattr(confirm, "help", "") or ""
    assert "stop signal" in help_text
    assert "paid" not in help_text
