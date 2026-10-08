# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import math
from pathlib import Path

import pytest

from pup_finetune.display import display_path, quoted, rounded_floats, significant, word
from pup_finetune.next_steps import command, template


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return Path.cwd()


def test_a_path_inside_the_current_directory_reads_relative(cwd: Path) -> None:
    assert display_path(cwd / "runs" / "run1") == "runs/run1"
    assert display_path("runs/run1") == "runs/run1"
    assert display_path("./ds/") == "ds"
    assert display_path(cwd) == "."


def test_a_path_outside_the_current_directory_reads_absolute(cwd: Path) -> None:
    outside = cwd.parent / "elsewhere" / "ds"
    assert display_path(outside) == str(outside)
    assert display_path("../elsewhere/ds") == str(outside)


def test_a_path_through_a_link_to_the_current_directory_still_reads_relative(
    cwd: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    link = tmp_path_factory.mktemp("links") / "here"
    link.symlink_to(cwd, target_is_directory=True)
    assert display_path(link / "ds") == "ds"


def test_commands_and_hints_show_path_words_as_results_do(cwd: Path) -> None:
    assert command("status", cwd / "run", "--watch") == "pup finetune status run --watch"
    assert template("dataset", "push", cwd / "my ds", "--name", "<NAME>") == (
        "pup finetune dataset push 'my ds' --name <NAME>"
    )
    # A plain string is a value, not a path: it is left as written.
    assert command("evaluate", str(cwd / "run")) == f"pup finetune evaluate {cwd / 'run'}"
    assert word(cwd / "run") == "run"
    assert word(3) == "3"
    assert quoted(cwd / "my run") == "'my run'"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.014949082979001105, 0.01495),
        (1234.5678, 1235.0),
        (123456.0, 123500.0),
        (1e-4, 1e-4),
        (-2.718281828, -2.718),
        (0.0, 0.0),
        (1.2, 1.2),
    ],
)
def test_floats_keep_four_significant_digits(value: float, expected: float) -> None:
    assert significant(value) == expected


def test_non_finite_floats_are_left_for_the_serializer_to_refuse() -> None:
    assert math.isnan(significant(math.nan))
    assert significant(math.inf) == math.inf


def test_every_measurement_in_a_result_is_rounded_and_nothing_else_changes() -> None:
    result = {
        "loss": 1.23456,
        "steps": 7,
        "ok": True,
        "id": "1.23456",
        "losses": (2.34567,),
        "headline": {"judge_pass_rate": {"delta": 0.123456, "ci95": [0.012345, 0.234567]}},
    }
    assert rounded_floats(result) == {
        "loss": 1.235,
        "steps": 7,
        "ok": True,
        "id": "1.23456",
        "losses": [2.346],
        "headline": {"judge_pass_rate": {"delta": 0.1235, "ci95": [0.01235, 0.2346]}},
    }


def test_values_the_user_set_are_never_rounded() -> None:
    result = {
        "learning_rate": 0.00012345,
        "settings": {"max_spend_usd": 1234.5, "loss": 1.23456, "lora": {"alpha": 16.25}},
        "preview": {"settings": {"learning_rate": 0.00012345}},
    }
    assert rounded_floats(result) == result
