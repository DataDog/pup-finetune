# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The benchmark's ranking and recommendation, over results written into its directories."""

from __future__ import annotations

import dataclasses
import shutil
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
    Leaderboard,
    leaderboard,
    plan_benchmark,
    recommend,
    sample_conversations,
)
from pup_finetune.evaluation.cases import Exclusion, ReplayCase, cases_from_dataset
from pup_finetune.evaluation.evaluators import (
    CheckEvaluator,
    EvaluatorSet,
    attach_evaluators,
    load_spec,
)
from pup_finetune.evaluation.fake_judge import fake_judge_config
from pup_finetune.evaluation.replay import EvalDir
from pup_finetune.providers.fake import FAKE_PROFILE, FakeProvider, FakeRenderer
from pup_finetune.runtime import Runtime

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
CHECKS = ROWS.parent.parent / "evaluators" / "checks.py"


@pytest.fixture
def plan(tmp_path: Path) -> BenchmarkPlan:
    dataset = DatasetDir(tmp_path / "ds")
    imported = CliRunner().invoke(
        app, ["dataset", "import", str(dataset.path), "--file", str(ROWS)]
    )
    assert imported.exit_code == 0, imported.stderr
    cases = cases_from_dataset(dataset, split="test")
    assert len({case.row_id for case in cases}) >= 2
    rendering = FakeRenderer().identity()
    return BenchmarkPlan(
        dataset_dir=dataset.path,
        candidates=(
            Candidate(provider="fake", profile=FAKE_PROFILE, rendering=rendering, excluded={}),
            Candidate(
                provider="fake",
                profile=dataclasses.replace(FAKE_PROFILE, alias="fake-b"),
                rendering=rendering,
                excluded={},
            ),
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


def write_error(plan: BenchmarkPlan, candidate: Candidate, case: ReplayCase, kind: str) -> None:
    EvalDir(plan.model_dir(candidate)).append_result(
        {
            "target": candidate.target,
            "case_id": case.id,
            "row_id": case.row_id,
            "status": "error",
            "error": {"stage": "generation", "type": kind, "message": "failed"},
        }
    )


def replaying(plan: BenchmarkPlan, model: str, case_ids: set[str]) -> BenchmarkPlan:
    refused = Exclusion("unsupported_message", "refused")
    candidates = tuple(
        candidate
        if candidate.target != model
        else dataclasses.replace(
            candidate,
            excluded={case.id: refused for case in plan.cases if case.id not in case_ids},
        )
        for candidate in plan.candidates
    )
    return dataclasses.replace(plan, candidates=candidates)


def by_model(board: Leaderboard) -> dict[str, dict[str, Any]]:
    return {entry["model"]: entry for entry in board.entries}


def test_the_best_model_comes_first_with_the_gap_of_the_others(plan: BenchmarkPlan) -> None:
    worse, better = plan.candidates
    write_results(plan, worse, {case.id: False for case in plan.cases})
    write_results(plan, better, {case.id: True for case in plan.cases})

    board = leaderboard(plan)

    assert [entry["model"] for entry in board.entries] == ["fake-b", "fake-model"]
    assert [entry["rank"] for entry in board.entries] == [1, 2]
    assert board.entries[0]["judge_pass_rate"] == 1.0
    assert board.entries[0]["status"] == "complete"
    assert board.entries[1]["gap_to_best"]["judge_pass_rate"] == 1.0
    assert board.entries[1]["within_noise_of_best"] is False
    assert board.common_cases == len(plan.cases)


def test_a_model_with_the_same_verdicts_is_within_noise(plan: BenchmarkPlan) -> None:
    verdicts = {case.id: index % 2 == 0 for index, case in enumerate(plan.cases)}
    for candidate in plan.candidates:
        write_results(plan, candidate, verdicts)

    board = leaderboard(plan)

    assert board.entries[1]["gap_to_best"]["judge_pass_rate"] == 0.0
    assert board.entries[1]["within_noise_of_best"] is True


def test_unfinished_models_are_unranked_in_the_order_given(plan: BenchmarkPlan) -> None:
    first, second = plan.candidates
    write_results(plan, second, {plan.cases[0].id: True})

    board = leaderboard(plan, {first.target: {"type": "ExternalError", "message": "down"}})

    assert [entry["model"] for entry in board.entries] == ["fake-model", "fake-b"]
    assert [entry["status"] for entry in board.entries] == ["error", "partial"]
    assert [entry["rank"] for entry in board.entries] == [None, None]
    assert board.entries[0]["judge_pass_rate"] is None
    assert board.entries[1]["judge_pass_rate"] == 1.0


def test_a_model_that_replays_few_cases_never_lifts_another_s_rate(plan: BenchmarkPlan) -> None:
    first = plan.cases[0].id
    plan = replaying(plan, "fake-b", {first})
    full, narrow = plan.candidates
    write_results(plan, full, {case.id: case.id == first for case in plan.cases})
    write_results(plan, narrow, {first: True})

    board = leaderboard(plan)

    entries = by_model(board)
    assert entries["fake-model"]["judge_pass_rate"] == 1 / len(plan.cases)
    assert entries["fake-model"]["judged_cases"] == len(plan.cases)
    assert entries["fake-b"]["judge_pass_rate"] == 1.0
    assert entries["fake-b"]["judged_cases"] == 1
    assert board.common_cases == 1
    assert recommend(board.entries, target_pass_rate=0.9)["verdict"] == "fine_tune"


def test_an_unfinished_model_never_narrows_the_comparison(plan: BenchmarkPlan) -> None:
    first = plan.cases[0].id
    finished, unfinished = plan.candidates
    write_results(plan, finished, {case.id: case.id == first for case in plan.cases})
    write_results(plan, unfinished, {first: True})

    board = leaderboard(plan)

    assert board.common_cases == len(plan.cases)
    best = board.entries[0]
    assert (best["model"], best["rank"]) == ("fake-model", 1)
    assert best["judge_pass_rate"] == best["common_pass_rate"] == 1 / len(plan.cases)
    found = recommend(board.entries, target_pass_rate=0.9)
    assert (found["verdict"], found["model"]) == ("fine_tune", "fake-model")


def test_a_partial_model_is_listed_unranked_beside_a_finished_one(plan: BenchmarkPlan) -> None:
    finished, partial = plan.candidates
    write_results(plan, finished, {case.id: True for case in plan.cases})
    write_results(plan, partial, {plan.cases[0].id: False, plan.cases[1].id: True})

    board = leaderboard(plan)

    ranked, unranked = board.entries
    assert (ranked["model"], ranked["finished"], ranked["rank"]) == ("fake-model", True, 1)
    assert unranked["model"] == "fake-b"
    assert unranked["finished"] is False
    assert unranked["rank"] is None
    assert unranked["judge_pass_rate"] == 0.5
    assert unranked["judged_cases"] == 2
    assert unranked["gap_to_best"] is None
    assert unranked["within_noise_of_best"] is None
    found = recommend(board.entries, target_pass_rate=0.9)
    assert (found["verdict"], found["model"]) == ("base_meets_target", "fake-model")
    assert "results are partial: fake-b can still advance" in found["reason"]


@pytest.mark.parametrize(("kind", "finished"), [("ExternalError", False), ("RendererError", True)])
def test_a_model_is_finished_once_a_rerun_cannot_change_it(
    plan: BenchmarkPlan, kind: str, finished: bool
) -> None:
    one, other = plan.candidates
    *answered, last = plan.cases
    write_results(plan, one, {case.id: True for case in plan.cases})
    write_results(plan, other, {case.id: True for case in answered})
    write_error(plan, other, last, kind)

    entry = by_model(leaderboard(plan))["fake-b"]

    assert entry["finished"] is finished
    assert (entry["rank"] is not None) is finished


def test_the_target_is_decided_on_the_model_s_own_rate(plan: BenchmarkPlan) -> None:
    failed = plan.cases[0].id
    plan = replaying(plan, "fake-b", {case.id for case in plan.cases if case.id != failed})
    best, worse = plan.candidates
    write_results(plan, best, {case.id: case.id != failed for case in plan.cases})
    write_results(plan, worse, {case.id: False for case in plan.cases if case.id != failed})

    board = leaderboard(plan)

    first = board.entries[0]
    assert (first["model"], first["common_pass_rate"], first["judge_pass_rate"]) == (
        "fake-model",
        1.0,
        0.8,
    )
    assert recommend(board.entries, target_pass_rate=0.9)["verdict"] == "fine_tune"
    assert recommend(board.entries, target_pass_rate=0.8)["verdict"] == "base_meets_target"


def test_a_model_judged_on_too_few_cases_is_never_held_to_the_target(
    plan: BenchmarkPlan,
) -> None:
    plan = replaying(plan, "fake-b", {plan.cases[0].id})
    narrow = plan.candidates[1]
    plan = dataclasses.replace(plan, candidates=(narrow,))
    write_results(plan, narrow, {plan.cases[0].id: True})

    found = recommend(leaderboard(plan).entries, target_pass_rate=0.9)

    assert found["verdict"] == "fine_tune"
    assert found["judge_pass_rate"] == 1.0
    assert found["coverage"] == 1 / len(plan.cases)


def test_finished_models_with_no_case_in_common_are_not_ranked(plan: BenchmarkPlan) -> None:
    first = plan.cases[0].id
    plan = replaying(plan, "fake-model", {first})
    plan = replaying(plan, "fake-b", {case.id for case in plan.cases if case.id != first})
    one, other = plan.candidates
    write_results(plan, one, {first: True})
    write_results(plan, other, {case.id: True for case in plan.cases if case.id != first})

    board = leaderboard(plan)

    assert board.common_cases == 0
    assert [entry["rank"] for entry in board.entries] == [None, None]
    found = recommend(board.entries, target_pass_rate=0.9)
    assert found["verdict"] == "no_result"
    assert "share no judged case" in found["reason"]


def test_no_finished_model_is_no_result_without_claiming_nothing_was_judged(
    plan: BenchmarkPlan,
) -> None:
    down, partial = plan.candidates
    write_results(plan, partial, {plan.cases[0].id: True})

    board = leaderboard(plan, {down.target: {"type": "ExternalError", "message": "down"}})
    found = recommend(board.entries, target_pass_rate=0.9)

    assert found["verdict"] == "no_result"
    assert "no model has a judged case" not in found["reason"]
    assert "no finished model has a judged case" in found["reason"]


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
    [found] = [entry for entry in board.entries if entry["model"] == candidate.target]
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
    done = {"status": "complete", "judged_cases": 4, "errors_unrenderable": 0}
    done |= {"unparseable": 0, "out_dir": "bench/fake-model"}
    board = Leaderboard([{**entry("fake-model", 0.5), **done, "truncated": truncated}], 1)

    found = benchmark_module._result(plan, board, calibration=None, cancelled=False)

    train = f"pup finetune train --provider fake --dataset {plan.dataset_dir} --model fake-model"
    assert found["next_command"] == (train if trains else None)
    assert found["next_steps"][0]["command"].endswith("--max-tokens 8192 --out-dir <NEW_DIR>")
    assert "not their length" in found["next_steps"][0]["why"]


def entry(
    model: str,
    rate: float | None,
    *,
    rank: int | None = 1,
    close: bool | None = None,
    judged: int = 10,
) -> dict[str, Any]:
    return {
        "model": model,
        "provider": "fake",
        "finished": True,
        "rank": rank if rate is not None else None,
        "judge_pass_rate": rate,
        "headline_pass_rate": rate,
        "judged_cases": judged if rate is not None else 0,
        "coverage": 1.0 if rate is not None else 0.0,
        "within_noise_of_best": close,
    }


def test_recommend_fine_tunes_the_best_model_below_the_target() -> None:
    found = recommend([entry("a", 0.6), entry("b", 0.4, rank=2, close=False)], target_pass_rate=0.9)

    assert found["verdict"] == "fine_tune"
    assert found["model"] == "a"
    assert found["within_noise"] == []


def test_recommend_names_the_models_within_noise() -> None:
    found = recommend([entry("a", 0.6), entry("b", 0.55, rank=2, close=True)], target_pass_rate=0.9)

    assert found["within_noise"] == ["b"]
    assert "b is within noise" in found["reason"]


def test_recommend_says_a_base_model_at_the_target_may_need_no_training() -> None:
    found = recommend([entry("a", 0.95)], target_pass_rate=0.9)

    assert found["verdict"] == "base_meets_target"
    assert found["model"] == "a"


@pytest.mark.parametrize(
    ("rate", "reason"),
    [
        (0.95, "a ranks first and passes the judge and every evaluator on 95% of its 10 judged"),
        (0.5, "every evaluator on 50% of its 10 judged cases, below the 90% target"),
        (None, "no model has a case the judge and every evaluator scored yet"),
    ],
)
def test_recommend_by_all_names_the_judge_and_every_evaluator(
    rate: float | None, reason: str
) -> None:
    found = recommend([entry("a", rate)], target_pass_rate=0.9, headline="all")

    assert reason in found["reason"]


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


def test_the_headline_evaluator_ranks_the_models_and_drives_the_recommendation(
    plan: BenchmarkPlan,
) -> None:
    short = CheckEvaluator(name="short", file=Path("checks.py"), function="short", file_sha256="0")
    evaluators = EvaluatorSet(evaluators=(short,), headline="short", spec=None, source="flag")
    plan = dataclasses.replace(plan, evaluators=evaluators)
    judged_worse, judged_better = plan.candidates
    for candidate, passed in ((judged_worse, False), (judged_better, True)):
        eval_dir = EvalDir(plan.model_dir(candidate))
        eval_dir.create()
        for case in plan.cases:
            line = result(candidate.target, case, passed)
            line["evaluators"] = {"short": {"status": "ok", "passed": not passed}}
            eval_dir.append_result(line)

    board = leaderboard(plan)

    first, second = board.entries
    assert [first["model"], second["model"]] == ["fake-model", "fake-b"]
    assert (first["headline_pass_rate"], first["judge_pass_rate"]) == (1.0, 0.0)
    assert first["evaluators"] == {"short": 1.0, "all": 0.0}
    assert first["common_pass_rate"] == 1.0
    gap = second["gap_to_best"]
    assert (gap["pass_rate"], "judge_pass_rate" in gap) == (1.0, False)
    found = benchmark_module._result(plan, board, calibration=None, cancelled=False)
    recommendation = found["recommendation"]
    assert (recommendation["model"], recommendation["headline"]) == ("fake-model", "short")
    assert (recommendation["pass_rate"], recommendation["verdict"]) == (1.0, "base_meets_target")
    assert "passes the evaluator short" in recommendation["reason"]


def checks(*names: str, headline: str) -> EvaluatorSet:
    found = tuple(
        CheckEvaluator(name=name, file=Path("checks.py"), function=name, file_sha256="0")
        for name in names
    )
    return EvaluatorSet(evaluators=found, headline=headline, spec=None, source="flag")


def write_scored(
    plan: BenchmarkPlan, candidate: Candidate, scores: dict[str, dict[str, bool | None]]
) -> None:
    """An ``ok`` result the judge passed per case of ``scores``, with each evaluator's entry:
    passed or failed, or an error for ``None``."""
    eval_dir = EvalDir(plan.model_dir(candidate))
    eval_dir.create()
    for case in plan.cases:
        if case.id in scores:
            line = result(candidate.target, case, True)
            line["evaluators"] = {
                name: {"status": "ok", "passed": passed}
                if passed is not None
                else {"status": "error", "passed": None}
                for name, passed in scores[case.id].items()
            }
            eval_dir.append_result(line)


def test_the_headline_ranks_the_finished_models_on_the_cases_it_scored_for_all(
    plan: BenchmarkPlan,
) -> None:
    first = plan.cases[0].id
    plan = dataclasses.replace(plan, evaluators=checks("short", headline="short"))
    plan = replaying(plan, "fake-b", {first})
    full, narrow = plan.candidates
    write_scored(plan, full, {case.id: {"short": case.id != first} for case in plan.cases})
    write_scored(plan, narrow, {first: {"short": True}})

    board = leaderboard(plan)

    assert board.common_cases == 1
    best, other = board.entries
    assert (best["model"], best["rank"], best["common_pass_rate"]) == ("fake-b", 1, 1.0)
    assert (other["model"], other["rank"], other["common_pass_rate"]) == ("fake-model", 2, 0.0)
    assert other["headline_pass_rate"] == (len(plan.cases) - 1) / len(plan.cases)
    assert other["judge_pass_rate"] == 1.0
    assert other["gap_to_best"] == {"pass_rate": 1.0, "ci95": None, "n_cases": 1}
    found = recommend(board.entries, target_pass_rate=0.9, headline="short")
    assert (found["model"], found["pass_rate"], found["verdict"]) == ("fake-b", 1.0, "fine_tune")
    assert found["coverage"] == 1 / len(plan.cases)


@pytest.mark.parametrize(("headline", "ranked"), [("short", True), ("broken", False)])
def test_only_a_score_the_headline_needs_keeps_a_model_unranked(
    plan: BenchmarkPlan, headline: str, ranked: bool
) -> None:
    plan = dataclasses.replace(plan, evaluators=checks("short", "broken", headline=headline))
    plan = dataclasses.replace(plan, candidates=plan.candidates[:1])
    [candidate] = plan.candidates
    failing = plan.cases[0].id
    write_scored(
        plan,
        candidate,
        {
            case.id: {"short": True, "broken": None if case.id == failing else True}
            for case in plan.cases
        },
    )

    board = leaderboard(plan)
    found = benchmark_module._result(plan, board, calibration=None, cancelled=False)

    [entry] = board.entries
    assert (entry["finished"], entry["rank"] is not None) == (ranked, ranked)
    assert board.rerun is True
    assert found["next_command"].startswith("pup finetune benchmark ")


@pytest.mark.parametrize("order", [1, -1])
def test_a_rerun_without_evaluators_continues_the_largest_attached_set(
    plan: BenchmarkPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: int
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    runtime = Runtime.from_env()
    profiles = [candidate.profile for candidate in plan.candidates]
    monkeypatch.setattr(FakeProvider, "models", lambda self: profiles)
    shutil.copy(CHECKS, tmp_path / "checks.py")
    check = '[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\nfile = "checks.py"\n'
    text = check.replace('"no_card_numbers"', '"text"') + 'function = "answers_in_text"\n'
    for candidate, body in zip(plan.candidates, (check, check + text), strict=True):
        spec = tmp_path / f"{candidate.target}.toml"
        spec.write_text(body)
        eval_dir = EvalDir(plan.model_dir(candidate))
        eval_dir.create()
        found = load_spec(spec, judge=plan.judge, runtime=runtime)
        attach_evaluators(eval_dir.evaluators_path, found)
    models = [("fake-model", "fake"), ("fake-b", "fake")][::order]
    again = plan_benchmark(runtime, plan.dataset_dir, models, plan.settings)
    assert again.evaluator_names == ["no_card_numbers", "text"]
    assert again.preview()["evaluators"]["source"] == "evaluation directory"


def test_a_rerun_without_evaluators_ranks_by_the_newest_of_equal_sets_and_rewrites_none(
    plan: BenchmarkPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    runtime = Runtime.from_env()
    profiles = [candidate.profile for candidate in plan.candidates]
    monkeypatch.setattr(FakeProvider, "models", lambda self: profiles)
    shutil.copy(CHECKS, tmp_path / "checks.py")
    spec = tmp_path / "checks.toml"
    check = '[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\nfile = "checks.py"\n'
    spec.write_text(check)
    both = [("fake-model", "fake"), ("fake-b", "fake")]
    flagged = dataclasses.replace(plan.settings, evaluators=spec)
    assert benchmark_module.execute(plan_benchmark(runtime, plan.dataset_dir, both, flagged))
    spec.write_text(f'headline = "no_card_numbers"\n{check}')
    benchmark_module.execute(plan_benchmark(runtime, plan.dataset_dir, both[1:], flagged))
    first = EvalDir(plan.model_dir(plan.candidates[0])).evaluators_path
    before = first.read_bytes()

    found = benchmark_module.execute(plan_benchmark(runtime, plan.dataset_dir, both, plan.settings))

    assert found["recommendation"]["headline"] == "no_card_numbers"
    assert first.read_bytes() == before
