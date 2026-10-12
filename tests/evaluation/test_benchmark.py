# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The benchmark's ranking and recommendation, over results written into its directories."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pup_finetune.cli import app
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.evaluation import benchmark as benchmark_module
from pup_finetune.evaluation.benchmark import (
    BenchmarkPlan,
    BenchmarkSettings,
    Candidate,
    leaderboard,
    recommend,
)
from pup_finetune.evaluation.cases import ReplayCase, cases_from_dataset
from pup_finetune.evaluation.fake_judge import fake_judge_config
from pup_finetune.evaluation.replay import EvalDir
from pup_finetune.providers.fake import FAKE_PROFILE, FakeProvider
from pup_finetune.runtime import Runtime

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"


@pytest.fixture
def plan(tmp_path: Path) -> BenchmarkPlan:
    dataset = DatasetDir(tmp_path / "ds")
    imported = CliRunner().invoke(
        app, ["dataset", "import", str(dataset.path), "--file", str(ROWS)]
    )
    assert imported.exit_code == 0, imported.stderr
    cases = cases_from_dataset(dataset, split="test")
    assert len({case.row_id for case in cases}) >= 2
    return BenchmarkPlan(
        dataset_dir=dataset.path,
        candidates=(
            Candidate(provider="fake", profile=FAKE_PROFILE),
            Candidate(provider="fake", profile=dataclasses.replace(FAKE_PROFILE, alias="fake-b")),
        ),
        cases=cases,
        judge=fake_judge_config(),
        settings=BenchmarkSettings(out_dir=tmp_path / "bench"),
        runtime=Runtime.from_env(),
    )


def write_results(plan: BenchmarkPlan, candidate: Candidate, passed: dict[str, bool]) -> None:
    eval_dir = EvalDir(plan.model_dir(candidate))
    eval_dir.create()
    for case in plan.cases:
        if case.id in passed:
            eval_dir.append_result(result(candidate.target, case, passed[case.id]))


def result(target: str, case: ReplayCase, passed: bool) -> dict[str, Any]:
    return {
        "target": target,
        "case_id": case.id,
        "row_id": case.row_id,
        "status": "ok",
        "metrics": {},
        "judgment": {"passed": passed},
    }


def test_the_best_model_comes_first_with_the_gap_of_the_others(plan: BenchmarkPlan) -> None:
    worse, better = plan.candidates
    write_results(plan, worse, {case.id: False for case in plan.cases})
    write_results(plan, better, {case.id: True for case in plan.cases})

    board = leaderboard(plan)

    assert [entry["model"] for entry in board] == ["fake-b", "fake-model"]
    assert board[0]["judge_pass_rate"] == 1.0
    assert board[0]["status"] == "complete"
    assert board[1]["gap_to_best"]["judge_pass_rate"] == 1.0
    assert board[1]["gap_to_best"]["n_cases"] == len(plan.cases)
    assert board[1]["within_noise_of_best"] is False


def test_a_model_with_the_same_verdicts_is_within_noise(plan: BenchmarkPlan) -> None:
    verdicts = {case.id: index % 2 == 0 for index, case in enumerate(plan.cases)}
    for candidate in plan.candidates:
        write_results(plan, candidate, verdicts)

    board = leaderboard(plan)

    assert board[1]["gap_to_best"]["judge_pass_rate"] == 0.0
    assert board[1]["within_noise_of_best"] is True


def test_a_model_without_results_is_last_and_pending(plan: BenchmarkPlan) -> None:
    first, second = plan.candidates
    write_results(plan, second, {plan.cases[0].id: True})

    board = leaderboard(plan, {first.target: {"type": "ExternalError", "message": "down"}})

    assert [entry["model"] for entry in board] == ["fake-b", "fake-model"]
    assert board[0]["status"] == "partial"
    assert board[1]["status"] == "error"
    assert board[1]["judge_pass_rate"] is None
    assert board[1]["gap_to_best"] is None


def entry(model: str, rate: float | None, *, close: bool | None = None) -> dict[str, Any]:
    return {
        "model": model,
        "provider": "fake",
        "judge_pass_rate": rate,
        "within_noise_of_best": close,
    }


def test_recommend_fine_tunes_the_best_model_below_the_target() -> None:
    found = recommend([entry("a", 0.6), entry("b", 0.4, close=False)], target_pass_rate=0.9)

    assert found["verdict"] == "fine_tune"
    assert found["model"] == "a"
    assert found["within_noise"] == []


def test_recommend_names_the_models_within_noise() -> None:
    found = recommend([entry("a", 0.6), entry("b", 0.55, close=True)], target_pass_rate=0.9)

    assert found["within_noise"] == ["b"]
    assert "b is within noise" in found["reason"]


def test_recommend_says_a_base_model_at_the_target_may_need_no_training() -> None:
    found = recommend([entry("a", 0.95)], target_pass_rate=0.9)

    assert found["verdict"] == "base_meets_target"
    assert found["model"] == "a"


def test_recommend_without_any_judged_case() -> None:
    found = recommend([entry("a", None)], target_pass_rate=0.9)

    assert found == {
        "verdict": "no_result",
        "model": None,
        "reason": "no model has a judged case yet, so there is nothing to compare",
    }


def test_the_session_context_fits_the_longest_case(
    plan: BenchmarkPlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(plan.runtime)
    monkeypatch.setattr(benchmark_module, "get_provider", lambda name, runtime: provider)
    candidate = plan.candidates[0]
    renderer = provider.renderer(candidate.profile)
    longest = max(
        len(renderer.generation_prompt_tokens(list(case.prefix), list(case.tools)))
        for case in plan.cases
    )

    with benchmark_module._base_sampler(plan, candidate):
        pass

    [settings] = provider.backend().session_settings
    assert settings["max_seq_len"] == longest + plan.settings.max_tokens
    assert settings["max_seq_len"] < candidate.profile.max_seq_len
