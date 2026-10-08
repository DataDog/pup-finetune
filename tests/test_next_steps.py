# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The next_command rule: what :mod:`pup_finetune.next_steps` builds and checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pup_finetune.next_steps import command, problems, step, template


def test_command_quotes_every_word() -> None:
    assert command("dataset", "show", Path("my ds")) == "pup finetune dataset show 'my ds'"
    assert (
        command("dataset", "sample", "ds", "--count", 20)
        == "pup finetune dataset sample ds --count 20"
    )


def test_template_leaves_only_placeholders_unquoted() -> None:
    assert template("dataset", "push", Path("my ds"), "--name", "<NAME>") == (
        "pup finetune dataset push 'my ds' --name <NAME>"
    )
    # Only the one style is a placeholder: anything else is quoted like any other word.
    assert template("x", "<dir>", "<A B>") == "pup finetune x '<dir>' '<A B>'"


@pytest.mark.parametrize(
    "result",
    [
        {"next_command": None},
        {"next_command": "pup finetune evaluate run --confirm"},
        {"next_command": None, "next_steps": []},
        {"next_steps": [step("pup finetune dataset push ds --name <NAME>", "to publish")]},
        {"nested": {"next_command": "pup finetune status 'my run' --watch"}},
    ],
)
def test_results_that_follow_the_rule(result: dict[str, Any]) -> None:
    assert problems(result) == []


@pytest.mark.parametrize(
    ("result", "found"),
    [
        ({"next_command": "pup finetune train --dataset ds ..."}, "holds '...'"),
        ({"next_command": "pup finetune dataset pull <DIR> --ml-app a"}, "not <UPPER_CASE>"),
        ({"next_command": ""}, "expected a command"),
        ({"next_command": "rm -rf ds"}, "is not a pup finetune command"),
        ({"next_commands": ["pup finetune x"]}, "use next_steps"),
        ({"next_steps": [{"command": "pup finetune x"}]}, "expected {command, why}"),
        ({"next_steps": [step("pup finetune x --name <name>", "y")]}, "not <UPPER_CASE>"),
        ({"next_steps": [step("pup finetune x --name ...", "y")]}, "holds '...'"),
        ({"next_steps": [step("pup finetune x", " ")]}, "expected a reason"),
        ({"evaluation": [{"next_command": 3}]}, "result.evaluation[0].next_command"),
    ],
)
def test_results_that_break_the_rule(result: dict[str, Any], found: str) -> None:
    (problem,) = problems(result)
    assert found in problem
