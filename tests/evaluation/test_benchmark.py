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
    sample_conversations,
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
        judge_source="fake run",
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


def test_the_board_counts_answers_cut_off_or_unparseable(plan: BenchmarkPlan) -> None:
    candidate = plan.candidates[0]
    write_results(plan, candidate, {case.id: False for case in plan.cases})
    eval_dir = EvalDir(plan.model_dir(candidate))
    endings = [("length", False), ("stop", False), ("stop", True)]
    for case, (stop_reason, format_valid) in zip(plan.cases[:3], endings, strict=True):
        eval_dir.append_generation(
            {
                "target": candidate.target,
                "case_id": case.id,
                "message": {"role": "assistant", "content": "hm"},
                "format_valid": format_valid,
                "stop_reason": stop_reason,
            }
        )

    board = leaderboard(plan)
    [found] = [entry for entry in board if entry["model"] == candidate.target]
    result = benchmark_module._result(plan, board, calibration=None, cancelled=False)

    assert (found["truncated"], found["unparseable"]) == (1, 1)
    assert any(
        "1 of fake-model's generations reached --max-tokens" in w for w in result["warnings"]
    )
    assert any("1 of fake-model's answers could not be parsed" in w for w in result["warnings"])


def test_max_rows_keeps_whole_conversations_picked_by_the_seed(plan: BenchmarkPlan) -> None:
    rows = {case.row_id for case in plan.cases}
    picked = {seed: sample_conversations(plan.cases, 1, seed=seed) for seed in range(10)}

    for seed, cases in picked.items():
        assert cases == [case for case in plan.cases if case.row_id == cases[0].row_id]
        assert sample_conversations(plan.cases, 1, seed=seed) == cases
    assert {cases[0].row_id for cases in picked.values()} == rows
    assert sample_conversations(plan.cases, len(rows), seed=0) == plan.cases


def test_max_rows_is_in_a_models_identity_only_when_it_sampled(plan: BenchmarkPlan) -> None:
    candidate = plan.candidates[0]
    capped = dataclasses.replace(plan, settings=dataclasses.replace(plan.settings, max_rows=1))
    sampled = dataclasses.replace(capped, sampled_from=plan.trajectories)

    assert "max_rows" not in plan.identity(candidate)
    assert "max_rows" not in capped.identity(candidate)
    assert sampled.identity(candidate)["max_rows"] == 1


def test_a_small_sample_asks_for_a_larger_max_rows(plan: BenchmarkPlan) -> None:
    rows = len({case.row_id for case in plan.cases})
    sampled = dataclasses.replace(
        plan,
        cases=sample_conversations(plan.cases, 1, seed=0),
        settings=dataclasses.replace(plan.settings, max_rows=1),
        sampled_from=rows,
    )

    [warning] = sampled.split_warnings()

    assert warning.startswith(f"--max-rows 1 replays only 1 of the {rows} test conversations")
    assert "raise --max-rows" in warning


@pytest.mark.parametrize(("truncated", "trains"), [(1, True), (2, False)])
def test_cut_off_answers_ask_for_a_higher_max_tokens(
    plan: BenchmarkPlan, truncated: int, trains: bool
) -> None:
    done = {"status": "complete", "cases": 4, "unparseable": 0, "out_dir": "bench/fake-model"}
    board = [{**entry("fake-model", 0.5), **done, "truncated": truncated}]

    found = benchmark_module._result(plan, board, calibration=None, cancelled=False)

    train = f"pup finetune train --dataset {plan.dataset_dir} --model fake-model"
    assert found["next_command"] == (train if trains else None)
    assert found["next_steps"][0]["command"].endswith("--max-tokens 8192 --out-dir <NEW_DIR>")
    assert "not their length" in found["next_steps"][0]["why"]


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
    # A budget well under the fake model's context, so the session is sized, not capped.
    plan = dataclasses.replace(plan, settings=dataclasses.replace(plan.settings, max_tokens=256))
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
