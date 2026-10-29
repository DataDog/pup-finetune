# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What stderr gets after the result: every warning, then the closing summary.

On a terminal (faked here with a rich console that claims to be one) the summary is a few
lines; through a pipe (CliRunner's stderr) it is exactly one. ``PUP_FINETUNE_QUIET=1`` turns
the summary off, never the warnings.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from pup_finetune import output
from pup_finetune.cli import app
from pup_finetune.cli._summary import CAVEAT, MAX_LINES, SUMMARIES, format_duration, summarize
from pup_finetune.errors import ExternalError
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion, JudgeConfig

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output_of(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make stderr a terminal: a console that claims to be one, writing to the current
    ``sys.stderr`` (CliRunner's)."""

    def console() -> Console:
        return Console(
            file=sys.stderr, force_terminal=True, color_system=None, width=300, soft_wrap=True
        )

    monkeypatch.setattr(output, "stderr_console", console)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def train(dataset: Path, run_dir: Path, *extra: str) -> Result:
    return invoke(
        "train",
        "--provider",
        "fake",
        "--model",
        "fake-model",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run_dir),
        "--max-epochs",
        "2",
        *extra,
    )


@pytest.fixture
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    result = train(dataset, path, "--confirm")
    assert result.exit_code == 0, result.stderr
    return path


# --- through a pipe: one line --------------------------------------------------------------


def test_import_ends_with_one_summary_line(tmp_path: Path) -> None:
    result = invoke("dataset", "import", str(tmp_path / "ds"), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    data = output_of(result)
    assert result.stderr.splitlines()[-1] == (
        f"imported {ROWS}: 14 rows accepted, 0 excluded; 14 rows added, 0 unchanged. "
        f"Next: {data['next_command']}"
    )


def test_train_ends_with_its_outcome_in_one_line(dataset: Path, tmp_path: Path) -> None:
    result = train(dataset, tmp_path / "run", "--confirm")
    data = output_of(result)
    last = result.stderr.splitlines()[-1]
    assert last.startswith(f"run {data['run_id']} completed in ")
    assert ": 2 epochs, best epoch 2 (validation loss 2.100 -> 1.600); promoted model: " in last
    assert last.endswith(f". Next: {data['next_command']}")


def test_train_evaluate_without_publishing_names_no_flag_train_lacks(
    dataset: Path, tmp_path: Path
) -> None:
    result = train(dataset, tmp_path / "run", "--evaluate", "--eval-no-publish", "--confirm")
    assert result.exit_code == 0, result.stderr
    last = result.stderr.splitlines()[-1]
    assert "not published (publishing was turned off)" in last
    assert "--no-publish" not in result.stderr


def test_evaluate_prints_its_warning_then_one_summary_line(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir), "--judge-model", "fake-judge", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output_of(result)
    *_, warning, summary = result.stderr.splitlines()
    assert warning == f"warning: {data['warnings'][0]}"
    assert warning.startswith("warning: the evaluation was not published: ")
    assert summary == (
        "evaluation complete: 4 cases from 2 test conversations; judge fake-judge, calibration "
        "passed; judge agreement: base 0%, tuned 25%, tuned - base +25 pts (95% CI "
        f"0 to +50); 1 win, 0 regressions, 3 ties, 0 errors; {CAVEAT}; not published (see the "
        "warning above)"
    )
    assert "18:03:16 calibration passed: 12/12 checks" in result.stderr
    assert "18:03:16 evaluated 8/8 pairs (100%): 1 judged pass, 7 judged fail, 0 errors" in (
        result.stderr
    )
    assert "(ok)" not in result.stderr


def test_status_ends_with_the_status_line(run_dir: Path) -> None:
    result = invoke("status", str(run_dir))
    data = output_of(result)
    # The fake run takes a second on a slow runner, which the line then reports.
    elapsed = data["elapsed_seconds"]
    took = f" {format_duration(elapsed)} elapsed" if elapsed else ""
    assert result.stderr == (
        f"completed: epoch 2/2 step 2/2 loss 1.400 val 1.600 (best epoch 2){took}. Next: "
        f"{data['next_command']}\n"
    )


def test_a_preview_prints_its_card_then_its_warnings_then_the_next_line(
    tmp_path: Path,
) -> None:
    ds = tmp_path / "ds"
    imported = invoke(
        "dataset", "import", str(ds), "--file", str(ROWS), "--validation-fraction", "0"
    )
    assert imported.exit_code == 0, imported.stderr
    result = train(ds, tmp_path / "run")
    assert result.exit_code == 2
    data = output_of(result)
    lines = result.stderr.splitlines()
    warnings = [f"warning: {text}" for text in data["preview"]["warnings"]]
    assert warnings  # nothing to validate on
    card = lines[: -len(warnings) - 1]
    assert (
        card[0] == f"Preview only, nothing was run: train fake-model on fake in {tmp_path / 'run'}"
    )
    assert 1 < len(card) <= 6
    assert lines[len(card) : -1] == warnings
    assert lines[-1] == f"Next: {data['next_command']}"
    assert "\x1b" not in result.stderr  # off a terminal: plain text


def test_the_evaluation_cards_count_one_turn_in_the_singular(run_dir: Path, tmp_path: Path) -> None:
    dataset = Path(str(json.loads((run_dir / "run.json").read_text())["dataset_dir"]))
    previews = (
        invoke("evaluate", str(run_dir), "--max-cases", "1"),
        invoke("benchmark", str(dataset), "--models", "fake-model", "--max-cases", "1"),
    )
    for preview in previews:
        assert preview.exit_code == 2, preview.stderr
        assert " 1 turn per conversation" in preview.stderr


def test_quiet_turns_off_the_summary_but_not_the_warnings(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_QUIET", "1")
    result = invoke("evaluate", str(run_dir), "--judge-model", "fake-judge", "--confirm")
    assert result.exit_code == 0, result.stderr
    assert result.stderr.splitlines()[-1].startswith("warning: the evaluation was not published")
    assert "evaluation complete" not in result.stderr


def test_steps_progress_restores_a_line_per_step_and_per_pair(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_PROGRESS", "steps")
    trained = train(dataset, tmp_path / "run", "--batch-size", "3", "--confirm")
    assert "18:03:16 epoch 1/2 step 1/6: loss 2.000" in trained.stderr
    assert "18:03:16 epoch 1/2 step 3/6: loss 2.000" in trained.stderr
    evaluated = invoke(
        "evaluate", str(tmp_path / "run"), "--judge-model", "fake-judge", "--confirm"
    )
    # One line per pair, a fail with its cause; which pair finishes when depends on the pool.
    pairs = re.findall(
        r"^18:03:16 evaluated (\d)/8: (?:base|tuned) \S+ \(judged (?:pass\)|fail\): .+)$",
        evaluated.stderr,
        re.MULTILINE,
    )
    assert sorted(pairs) == [str(n) for n in range(1, 9)], evaluated.stderr


# --- on a terminal: a few lines ------------------------------------------------------------


@pytest.mark.usefixtures("terminal")
def test_on_a_terminal_train_shows_every_step_and_a_few_summary_lines(
    dataset: Path, tmp_path: Path
) -> None:
    result = train(dataset, tmp_path / "run", "--batch-size", "3", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output_of(result)
    lines = result.stderr.splitlines()
    assert "epoch 1/2 step 1/6: loss 2.000" in lines  # no timestamp on a terminal
    assert any(
        line.startswith("saved training checkpoint t1s3-")
        and line.endswith(" at epoch end (epoch 1, step 3)")
        for line in lines
    )
    assert lines[-3:] == [
        lines[-3],
        f"promoted model: {data['run_id']} (fake)",
        f"next: {data['next_command']}",
    ]
    assert lines[-3].startswith(f"run {data['run_id']} completed in ")


class DownForUntooledAnswers(FakeJudge):
    """The offline judge, unreachable for a text answer where the reference calls a tool:
    base's answers to the tool-call cases end in errors, so base is graded on fewer cases."""

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatCompletion:
        found = super().complete(messages, **kwargs)
        if "the reference calls a tool and the candidate does not" in found.message["content"]:
            raise ExternalError("the judge endpoint is unreachable", source="judge")
        return found


@pytest.mark.usefixtures("terminal")
def test_the_agreement_row_shows_the_rates_the_result_holds(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def opened(config: JudgeConfig, **options: Any) -> Iterator[ChatCompleter]:
        yield DownForUntooledAnswers()

    monkeypatch.setattr(runner_module, "open_judge_client", opened)
    result = invoke("evaluate", str(run_dir), "--skip-calibration", "--no-publish", "--confirm")
    data = output_of(result)
    assert data["status"] == "complete_with_errors", result.stderr
    rates = data["headline"]["judge_pass_rate"]
    assert rates["judged_cases"] == {"base": 2, "tuned": 4}
    assert rates["common_cases"] == 2
    base, tuned = (
        f"{round(rates[name] * 100)}% of {rates['judged_cases'][name]}"
        for name in ("base", "tuned")
    )
    common = {name: f"{round(rate * 100)}%" for name, rate in rates["common_pass_rate"].items()}
    lines = result.stderr.splitlines()
    row = next(line for line in lines if line.startswith("judge agreement"))
    assert row == f"{'judge agreement':<21}{base:<11}{tuned}"
    delta = next(line for line in lines if line.startswith("tuned - base: 0 pts"))
    assert delta.endswith(
        f" on 2 of 4 cases, tuned {common['tuned']} vs base {common['base']}; 0 wins, "
        "0 regressions, 2 ties, 2 errors"
    )
    status = invoke("status", str(run_dir)).stderr
    assert f"judge agreement base {base}, tuned {tuned}, tuned - base 0 pts" in status


@pytest.mark.usefixtures("terminal")
def test_on_a_terminal_evaluate_shows_the_table(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir), "--judge-model", "fake-judge", "--confirm")
    assert result.exit_code == 0, result.stderr
    lines = result.stderr.splitlines()
    # A line per pair, in the order the thread pool finishes them.
    assert "evaluated 8/8: " in result.stderr
    assert any(
        re.fullmatch(r"evaluated \d/8: tuned row-units-001:1 \(judged pass\)", line)
        for line in lines
    )
    start = lines.index(
        "evaluation complete: 4 cases from 2 test conversations; judge fake-judge, "
        "calibration passed"
    )
    assert lines[start:] == [
        lines[start],
        "                     base       tuned      tuned - base",
        "judge agreement      0%         25%        +25 pts (95% CI 0 to +50)",
        "paired: 1 win, 0 regressions, 3 ties, 0 errors",
        CAVEAT,
        "not published (see the warning above)",
    ]
    assert lines[start - 1].startswith("warning: the evaluation was not published")


# --- the summarizers -----------------------------------------------------------------------


def _calibration_failed() -> dict[str, Any]:
    checks = [{"ok": True}, {"ok": False}, {"ok": True}]
    return {
        "status": "calibration_failed",
        "judge_model": "j",
        "calibration": {"ok": False, "checks": checks},
    }


EXAMPLES: dict[str, dict[str, Any]] = {
    "dataset pull": {
        "accepted": 4,
        "excluded": 5,
        "excluded_by_code": {"a": 2, "b": 1, "c": 1, "d": 1},
        "splits": {"train": 2, "validation": 1, "test": 1},
        "pull": {"matched_spans": 10, "pages_fetched": 1, "pages_cached": 1, "added": 4},
        "next_command": "pup finetune dataset sample ./ds",
    },
    "dataset triage": {
        "status": "complete",
        "kept": 12,
        "dropped": 3,
        "undecided": 1,
        "too_long": 2,
        "calls_made": 36,
        "calls_failed": 0,
        "top_drop_reasons": [{"reason": "off topic", "count": 2}],
        "next_command": None,
    },
    "dataset push": {
        "dataset_name": "my-ds",
        "project": "pup-finetune-dev",
        "dataset_version": 3,
        "pushed": 12,
        "records": 20,
        "next_command": None,
    },
    "deploy": {
        "model": "accounts/a/models/ft-1",
        "deployment_id": "ft-1",
        "state": "READY",
        "smoke_test": {"ok": True},
        "usage": {"model": "accounts/a/deployments/ft-1", "base_url": "https://example.test/v1"},
        "undeploy_command": "pup finetune undeploy ./run",
        "next_command": None,
    },
    "undeploy": {"deployment_id": "ft-1", "model_kept": "fake://models/ft-1"},
    "cancel": {
        "run_id": "ft-1",
        "pid": 4242,
        "status": "cancelling",
        "next_command": "pup finetune status ./run --watch",
    },
}

EXPECTED: dict[str, str] = {
    "dataset pull": "read 10 spans on 2 pages: 4 rows accepted, 5 excluded (2 a, 1 b, 1 c, +1 more); "
    "4 rows added, 0 unchanged. Next: pup finetune dataset sample ./ds",
    "dataset triage": "triage complete: 12 kept, 3 dropped, 1 undecided, 2 too long; 36 judge "
    "calls, 0 failed",
    "dataset push": "pushed 12 records to dataset 'my-ds' in project 'pup-finetune-dev'; it holds "
    "20 records from this directory (version 3)",
    "deploy": "deployed ft-1 (ready); call it: model accounts/a/deployments/ft-1 at "
    "https://example.test/v1; it bills while a replica runs; delete it with: pup finetune "
    "undeploy ./run",
    "undeploy": "deleted the deployment ft-1; the model fake://models/ft-1 is kept",
    "cancel": "sent SIGTERM to process 4242: run ft-1 stops after its current step, saves a "
    "resumable checkpoint and promotes its best epoch. Next: pup finetune status ./run --watch",
}


@pytest.mark.parametrize("command", sorted(EXPECTED))
def test_each_summary_is_one_line_off_a_terminal_and_a_few_on_one(command: str) -> None:
    found = summarize(command, EXAMPLES[command])
    assert found is not None
    assert found.line == EXPECTED[command]
    assert "\n" not in found.line
    assert 1 <= len(found.lines) <= MAX_LINES


def test_doctor_prints_its_whole_checklist_on_or_off_a_terminal() -> None:
    result = {
        "versions": {"pup_finetune": "0.0.1.dev3+g1234567", "checkout": None},
        "mode": {"read_only": True, "dev": False},
        "checks": [
            {"name": "pup", "status": "pass", "label": "pup 1.0", "required": True},
            {
                "name": "datadog_auth",
                "status": "fail",
                "label": "Datadog: not signed in",
                "needed_for": "reading traces",
                "fix": "run `pup auth login`",
                "required": True,
            },
            {"name": "a:key", "status": "optional", "label": "A key not set", "needed_for": "a/x"},
        ],
        "ready": {
            "dataset show": {"ready": True},
            "dataset pull": {"ready": False},
            "dataset push": {"ready": False},
        },
        "next_command": None,
        "next_steps": [],
    }
    found = summarize("doctor", result)
    assert found is not None
    assert found.full
    assert found.lines == (
        "pup-finetune 0.0.1.dev3+g1234567   read-only: on   dev: off",
        "ok    pup 1.0",
        "FAIL  Datadog: not signed in  reading traces",
        "      fix: run `pup auth login`",
        "--    A key not set           only for a/x",
        "Ready: dataset show.  Not ready: dataset pull, push.",
        "Next: run `pup auth login`",
    )
    assert found.line == "\n".join(found.lines)


def test_triage_agreement_reports_the_rate_and_kappa() -> None:
    result = {
        "mode": "agreement",
        "status": "complete",
        "agreement": {
            "rows_compared": 20,
            "rows_agreed": 18,
            "rate": 0.9,
            "kappa": 0.784,
            "rows_undecided": 1,
            "rows_too_long": 0,
            "disagreements": 2,
        },
        "calls_made": 40,
        "calls_failed": 0,
        "next_command": "pup finetune dataset triage ds --rubric r.md --confirm",
    }
    found = summarize("dataset triage", result)
    assert found is not None
    assert found.lines[0] == (
        "agreement complete: the council agreed with your labels on 18/20 rows (90%), kappa 0.78"
    )
    assert found.line.endswith(
        "2 disagreements, 1 undecided, 0 too long. Next: pup finetune dataset triage ds "
        "--rubric r.md --confirm"
    )


CHECKS_AGREEMENT = {
    "declares_tools": {"failed": 1, "agree": 1},
    "no_tool_errors": {"failed": 2, "agree": 0},
}


def test_a_triage_with_checks_counts_the_rows_they_dropped() -> None:
    checks = {"failed": 4, "rows_checked": 22, "labels": {"applied": 3, "unchanged": 1}}
    found = summarize("dataset triage", {**EXAMPLES["dataset triage"], "checks": checks})
    assert found is not None
    assert found.lines[0] == (
        "triage complete: 12 kept, 3 dropped, 1 undecided, 2 too long, 3 newly dropped by checks"
    )


def test_an_agreement_with_checks_compares_each_check_with_the_labels() -> None:
    against = (
        "checks against your labels: declares_tools fails 1 row (1 labeled drop), "
        "no_tool_errors fails 2 rows (0 labeled drop)"
    )
    alone = {
        "mode": "checks",
        "checks": {"failed": 2, "rows_checked": 3, "labels": {"applied": 0, "unchanged": 0}},
        "agreement": {"checks": CHECKS_AGREEMENT},
        "next_command": None,
    }
    found = summarize("dataset triage", alone)
    assert found is not None
    assert found.line == f"checks: 2 of 3 labeled rows failed, nothing labeled; {against}"
    council = {
        "mode": "agreement",
        "status": "complete",
        "agreement": {"rows_compared": 1, "rows_agreed": 1, "checks": CHECKS_AGREEMENT},
        "next_command": None,
    }
    found = summarize("dataset triage", council)
    assert found is not None
    assert against in found.lines


def test_a_calibration_failure_says_nothing_was_evaluated() -> None:
    for command in ("evaluate", "train"):
        found = summarize(command, _calibration_failed())
        assert found is not None
        assert found.lines[0] == (
            "calibration failed: the judge j got 1 of 3 checks wrong, so nothing was evaluated"
        )


def test_each_comparison_counts_the_errors_of_its_own_two_targets() -> None:
    result = {
        "status": "complete_with_errors",
        "judge_model": "j",
        "calibration": {"ok": True},
        "errors": {"base": 1, "tuned": 2, "compare": 4},
        "headline": {
            "judge_pass_rate": {"base": 0.5, "tuned": 0.75, "compare": 0.5, "delta": 0.25},
            "paired": {"wins": 1, "regressions": 0, "ties": 3},
            "tuned_minus_compare": {"delta": 0.25, "wins": 1, "regressions": 0, "ties": 3},
            "cases": 4,
            "trajectories": 2,
        },
    }
    found = summarize("evaluate", result)
    assert found is not None
    (base_line,) = [line for line in found.lines if line.startswith("tuned - base")]
    (compare_line,) = [line for line in found.lines if line.startswith("tuned - compare")]
    assert base_line.endswith("3 ties, 3 errors")
    assert compare_line.endswith("3 ties, 6 errors")


PAIRED_ON_FOUR = {
    "base": 0.5,
    "tuned": 0.6667,
    "judged_cases": {"base": 4, "tuned": 6},
    "common_pass_rate": {"tuned": 1.0, "base": 0.5},
    "common_cases": 4,
    "delta": 0.5,
    "ci95": [0.0, 1.0],
}
"""Base ended in an error on 2 of 6 cases; tuned passes 4 of its 6, each of the 4 paired."""


def test_a_delta_is_shown_beside_the_rates_of_the_cases_it_pairs() -> None:
    polite = {
        "base": 0.25,
        "tuned": 0.5,
        "scored_cases": {"base": 4, "tuned": 6},
        "common_pass_rate": {"tuned": 0.75, "base": 0.25},
        "delta": 0.5,
        "ci95": None,
        "n": 4,
    }
    result = {
        "status": "complete_with_errors",
        "judge_model": "j",
        "calibration": {"ok": True},
        "errors": {"base": 2, "tuned": 0},
        "headline": {
            "judge_pass_rate": PAIRED_ON_FOUR,
            "paired": {"wins": 2, "regressions": 0, "ties": 2},
            "evaluator": "polite",
            "evaluators": {"polite": polite},
            "cases": 6,
            "trajectories": 3,
        },
    }
    found = summarize("evaluate", result)
    assert found is not None
    assert found.lines[1:5] == (
        "headline polite: base 25% of 4, tuned 50% of 6, tuned - base +50 pts on 4 of 6 cases, "
        "tuned 75% vs base 25%",
        f"{'':<21}{'base':<11}tuned",
        f"{'judge agreement':<21}{'50% of 4':<11}67% of 6",
        "tuned - base: +50 pts (95% CI 0 to +100) on 4 of 6 cases, tuned 100% vs base 50%; "
        "2 wins, 0 regressions, 2 ties, 2 errors",
    )
    assert (
        "judge agreement: base 50% of 4, tuned 67% of 6, tuned - base +50 pts (95% CI 0 to +100) "
        "on 4 of 6 cases, tuned 100% vs base 50%; 2 wins"
    ) in found.line


def test_with_a_compare_target_each_delta_names_the_rates_of_the_cases_it_pairs() -> None:
    result = {
        "status": "complete_with_errors",
        "judge_model": "j",
        "calibration": {"ok": True},
        "errors": {"base": 2, "tuned": 0, "compare": 0},
        "headline": {
            "judge_pass_rate": {
                **PAIRED_ON_FOUR,
                "compare": 0.75,
                "judged_cases": {"base": 4, "tuned": 6, "compare": 4},
            },
            "paired": {"wins": 2, "regressions": 0, "ties": 2},
            "tuned_minus_compare": {
                "common_pass_rate": {"tuned": 1.0, "compare": 0.75},
                "common_cases": 4,
                "delta": 0.25,
                "ci95": None,
                "wins": 1,
                "regressions": 0,
                "ties": 3,
                "skipped": 2,
            },
            "cases": 6,
            "trajectories": 3,
        },
    }
    found = summarize("evaluate", result)
    assert found is not None
    assert found.lines[1:4] == (
        "judge agreement: base 50% of 4, tuned 67% of 6, compare 75% of 4",
        "tuned - base: +50 pts (95% CI 0 to +100) on 4 of 6 cases, tuned 100% vs base 50%; "
        "2 wins, 0 regressions, 2 ties, 2 errors",
        "tuned - compare: +25 pts on 4 of 6 cases, tuned 100% vs compare 75%; 1 win, "
        "0 regressions, 3 ties, 0 errors",
    )


def test_a_published_comparison_lists_its_three_experiments_as_a_sentence() -> None:
    result = {
        "status": "complete",
        "judge_model": "j",
        "calibration": {"ok": True},
        "headline": {
            "judge_pass_rate": {"base": 0.5, "tuned": 0.75, "compare": 0.5, "delta": 0.25},
            "paired": {"wins": 1, "regressions": 0, "ties": 3},
            "tuned_minus_compare": {"delta": 0.25, "wins": 1, "regressions": 0, "ties": 3},
            "cases": 4,
            "trajectories": 2,
        },
        "publication": {
            "status": "published",
            "experiments": {
                target: {"name": f"pup-finetune-{target}-r"}
                for target in ("base", "tuned", "compare")
            },
        },
    }
    found = summarize("evaluate", result)
    assert found is not None
    assert found.lines[-1] == (
        "published as experiments pup-finetune-base-r, pup-finetune-tuned-r and "
        "pup-finetune-compare-r"
    )


def test_a_detached_train_names_its_worker_and_log() -> None:
    result = {
        "status": "started",
        "run_id": "ft-1",
        "pid": 42,
        "worker_log": "run/worker.log",
        "next_command": "pup finetune status run --watch",
    }
    found = summarize("train", result)
    assert found is not None
    assert found.lines == (
        "run ft-1 started in a background worker (pid 42)",
        "log: run/worker.log",
        "next: pup finetune status run --watch",
    )


def test_a_remote_train_names_its_job_team_and_project() -> None:
    result = {
        "status": "started",
        "run_id": "ft-1",
        "run_dir": "run",
        "remote": {
            "backend": "baseten job",
            "team": "Research",
            "project": "pup-finetune-ft-1",
            "job_id": "j1",
            "compute": {"cpu_count": 16},
            "logs": "baseten train job logs --job-id j1",
        },
        "upload": {"files": 120, "bytes": 2_600_000},
        "notes": [],
        "warnings": [],
        "next_command": "pup finetune status run --watch",
    }
    found = summarize("train", result)
    assert found is not None
    assert found.lines == (
        "run ft-1 started in Baseten job j1 (team Research)",
        "project: pup-finetune-ft-1",
        "next: pup finetune status run --watch",
    )
    assert found.line == (
        "run ft-1 started in Baseten job j1 (team Research); project: pup-finetune-ft-1. "
        "Next: pup finetune status run --watch"
    )


def test_an_unexpected_result_has_no_summary() -> None:
    assert summarize("evaluate", {"status": "complete"}) is None  # no summary in it
    assert summarize("dataset show", {"anything": 1}) is None  # no summarizer
    assert summarize(None, {}) is None


def test_every_summarized_command_exists() -> None:
    import typer.main

    group = typer.main.get_command(app)
    for path in SUMMARIES:
        command: Any = group
        for word in path.split():
            command = command.commands[word]
        assert command is not None


# --- real results ------------------------------------------------------------------------


def _ends_with_its_summary(command: str, result: Result) -> dict[str, Any]:
    """``result`` succeeded, and its stderr ends with the one-line summary of its own JSON
    (a summarizer that no longer fits the result it gets would print none)."""
    assert result.exit_code == 0, result.stderr
    data = output_of(result)
    found = summarize(command, data)
    assert found is not None, f"{command} has no summary for its result"
    assert result.stderr.splitlines()[-1] == found.line
    return data


@pytest.mark.usefixtures("judge_keys")
def test_the_dataset_commands_end_with_their_summaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from contextlib import contextmanager

    from pup_finetune.cli import dataset as dataset_cli
    from pup_finetune.cli import triage as triage_cli
    from pup_finetune.curation.triage import rubric_template
    from pup_finetune.inference import ChatCompletion
    from tests.fakes.datasets import FakeDatasetsApi
    from tests.fakes.llmobs import FakeExportApi, fixture_pages
    from tests.fakes.pup import FakePup

    fake = FakeDatasetsApi().install(FakeExportApi(fixture_pages()).install(FakePup()))
    monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    query = ("--ml-app", "synthetic-trip-planner", "--to", "now")
    target = tmp_path / "pulled"
    _ends_with_its_summary("dataset pull", invoke("dataset", "pull", str(target), *query))

    class Keep:
        def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
            content = json.dumps({"keep": True, "reason": "fine"})
            return ChatCompletion(message={"role": "assistant", "content": content})

    @contextmanager
    def open_clients(judges: Any) -> Any:
        yield {"fireworks": Keep()}

    monkeypatch.setattr(triage_cli, "open_clients", open_clients)
    template = rubric_template()
    rubric = tmp_path / "rubric.md"
    rubric.write_text(
        template[: template.index("## Task")]
        + "## Task\n\nA synthetic agent.\n\n"
        + template[template.index("## Keep") :],
        encoding="utf-8",
    )
    triaged = invoke("dataset", "triage", str(target), "--rubric", str(rubric), "--confirm")
    _ends_with_its_summary("dataset triage", triaged)
    pushed = invoke("dataset", "push", str(target), "--name", "synthetic-summary", "--confirm")
    _ends_with_its_summary("dataset push", pushed)


def test_deploy_and_undeploy_end_with_their_summaries(run_dir: Path) -> None:
    deployed = _ends_with_its_summary("deploy", invoke("deploy", str(run_dir), "--confirm"))
    assert deployed["state"] == "READY"
    _ends_with_its_summary("undeploy", invoke("undeploy", str(run_dir), "--confirm"))


def test_a_summary_that_fails_prints_its_traceback_in_debug_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert summarize("train", {"status": "started"}) is None
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("PUP_FINETUNE_DEBUG", "1")
    assert summarize("train", {"status": "started"}) is None
    assert "KeyError" in capsys.readouterr().err


def test_the_deploy_headline_names_a_different_deployment_id() -> None:
    found = summarize(
        "deploy", {"model": "accounts/a/models/m", "deployment_id": "d", "smoke_test": {"ok": True}}
    )
    assert found is not None
    assert found.lines[0] == "deployed m as d (ready)"


def test_the_headline_evaluator_s_line_comes_first() -> None:
    entry = {"base": 0.45, "tuned": 0.65, "delta": 0.2, "ci95": [0.05, 0.33], "n": 40}
    result = {
        "status": "complete",
        "judge_model": "j",
        "calibration": {"ok": True},
        "errors": {"base": 0, "tuned": 0},
        "headline": {
            "judge_pass_rate": {"base": 0.5, "tuned": 0.75, "delta": 0.25},
            "paired": {"wins": 1, "regressions": 0, "ties": 3},
            "evaluator": "all",
            "evaluators": {"polite": entry, "all": entry},
            "cases": 40,
            "trajectories": 20,
        },
    }
    found = summarize("evaluate", result)
    assert found is not None
    assert found.lines[1] == (
        "headline all (the judge and every evaluator): base 45%, tuned 65%, tuned - base "
        "+20 pts (95% CI +5 to +33)"
    )
    plain = summarize(
        "evaluate", {**result, "headline": {**result["headline"], "evaluator": "judge"}}
    )
    assert plain is not None and not plain.lines[1].startswith("headline")
