# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Renamed flags keep their old names for a release, `train --max-seq-len` is the
``max_seq_len`` provider option, and a judge alias is previewed and frozen as its full id.

Every test runs offline on the fake provider in development mode.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app

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
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
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


@pytest.fixture
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    result = invoke(*train_args(dataset, path, "--max-epochs", "1", "--confirm"))
    assert result.exit_code == 0, result.stderr
    return path


# --- renamed flags -----------------------------------------------------------------------


def test_sample_takes_count_and_n(dataset: Path) -> None:
    for flag in ("-n", "--count"):
        result = invoke("dataset", "sample", str(dataset), flag, "2")
        assert result.exit_code == 0, result.stderr
        assert len(output(result)["rows"]) == 2
        assert "renamed" not in result.stderr and "is now" not in result.stderr


def test_sample_s_old_n_still_works_with_a_warning(dataset: Path) -> None:
    result = invoke("dataset", "sample", str(dataset), "--n", "2")
    assert result.exit_code == 0, result.stderr
    assert len(output(result)["rows"]) == 2
    assert "--n is now --count" in result.stderr


def test_the_old_and_the_new_name_together_are_refused(dataset: Path) -> None:
    result = invoke("dataset", "sample", str(dataset), "--n", "2", "--count", "3")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--n is the old name of --count" in result.stderr


@pytest.mark.parametrize(
    ("path", "old", "new"),
    [
        (("dataset", "sample"), "--n", "--count"),
        (("evaluate",), "--max-cases-per-trajectory", "--max-cases"),
        (("train",), "--eval-max-cases-per-trajectory", "--eval-max-cases"),
    ],
)
def test_old_names_are_hidden_from_help(path: tuple[str, ...], old: str, new: str) -> None:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", invoke(*path, "--help").stdout)
    assert new in text
    assert f"{old} " not in text and f"{old}\n" not in text


def test_train_s_old_eval_max_cases_name_is_pinned_as_the_new_one(
    dataset: Path, tmp_path: Path
) -> None:
    extra = ("--evaluate", "--eval-max-cases-per-trajectory", "3")
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["preview"]["evaluation"]["settings"]["max_cases"] == 3
    assert "--eval-max-cases 3" in data["next_command"]
    assert "per-trajectory" not in data["next_command"]
    assert "--eval-max-cases-per-trajectory is now --eval-max-cases" in result.stderr


def test_the_old_eval_name_still_needs_evaluate(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--eval-max-cases-per-trajectory", "3"))
    assert result.exit_code == 2
    assert "--eval-max-cases-per-trajectory needs --evaluate" in result.stderr


def test_evaluate_takes_max_cases_and_its_old_name(run_dir: Path) -> None:
    new = invoke("evaluate", str(run_dir), "--max-cases", "1")
    assert new.exit_code == 2, new.stderr
    assert "--max-cases 1" in output(new)["next_command"]
    old = invoke("evaluate", str(run_dir), "--max-cases-per-trajectory", "1")
    assert old.exit_code == 2, old.stderr
    assert output(old)["next_command"] == output(new)["next_command"]
    assert "--max-cases-per-trajectory is now --max-cases" in old.stderr


# --- train --max-seq-len -----------------------------------------------------------------


def test_a_max_seq_len_over_the_context_names_the_flag_that_was_typed(
    dataset: Path, tmp_path: Path
) -> None:
    typed = invoke(*train_args(dataset, tmp_path / "run", "--max-seq-len", "99999"))
    assert typed.exit_code == 2
    assert typed.stderr.startswith(
        "Error [usage]: --max-seq-len 99999 is over the 4096-token context of fake-model\n"
    )
    option = invoke(
        *train_args(dataset, tmp_path / "run", "--provider-option", "max_seq_len=99999")
    )
    assert option.exit_code == 2
    assert "the max_seq_len provider option of 99999 is over" in option.stderr


def test_max_seq_len_caps_the_context_and_is_pinned_as_typed(dataset: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    preview = invoke(*train_args(dataset, run, "--max-seq-len", "1024"))
    assert preview.exit_code == 2, preview.stderr
    data = output(preview)
    assert data["preview"]["profile"]["max_seq_len"] == 1024
    assert data["preview"]["settings"]["provider_options"] == {"max_seq_len": 1024}
    assert data["preview"]["settings"]["max_seq_len"] == 1024
    assert data["preview"]["settings"]["max_seq_len_source"] == "--max-seq-len"
    assert "max_seq_len" not in data["preview"]["defaults_used"]
    assert "--max-seq-len 1024" in data["next_command"]
    assert "--provider-option" not in data["next_command"]

    as_option = invoke(*train_args(dataset, run, "--provider-option", "max_seq_len=1024"))
    # The same settings, each source named by the flag that set it.
    assert output(as_option)["preview"]["settings"] == {
        **data["preview"]["settings"],
        "max_seq_len_source": "--provider-option max_seq_len",
    }
    assert output(as_option)["preview"]["profile"] == data["preview"]["profile"]


def test_max_seq_len_is_recorded_in_run_json_as_the_provider_option(
    dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    result = invoke(*train_args(dataset, run, "--max-seq-len", "2048", "--max-epochs", "1"))
    confirmed = invoke(*shlex.split(output(result)["next_command"])[2:])
    assert confirmed.exit_code == 0, confirmed.stderr
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert manifest["spec"]["provider_options"] == {"max_seq_len": 2048}
    assert manifest["profile"]["max_seq_len"] == 2048


def test_max_seq_len_and_the_provider_option_together_are_refused(
    dataset: Path, tmp_path: Path
) -> None:
    extra = ("--max-seq-len", "1024", "--provider-option", "max_seq_len=1024")
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2
    assert "--max-seq-len and --provider-option max_seq_len" in result.stderr


def test_max_seq_len_over_the_model_s_context_is_refused(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--max-seq-len", "999999"))
    assert result.exit_code == 2
    assert "pass --max-seq-len N with N at most" in result.stderr


# --- judge aliases -----------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--judge-model"])
def test_evaluate_previews_a_judge_alias_as_its_full_id(run_dir: Path, flag: str) -> None:
    full = "accounts/fireworks/models/glm-5p3-flash"
    result = invoke("evaluate", str(run_dir), flag, "glm-5p3-flash")
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["preview"]["judge_model"] == full
    assert f"--judge-model {full}" in data["next_command"]


def test_train_evaluate_previews_a_judge_alias_as_its_full_id(
    dataset: Path, tmp_path: Path
) -> None:
    full = "accounts/fireworks/models/deepseek-v4p1-flash"
    extra = ("--evaluate", "--eval-judge-model", "deepseek-v4p1-flash")
    result = invoke(*train_args(dataset, tmp_path / "run", *extra))
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["preview"]["evaluation"]["judge_model"] == full
    assert f"--eval-judge-model {full}" in data["next_command"]
