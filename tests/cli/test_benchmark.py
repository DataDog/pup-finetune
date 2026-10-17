# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`benchmark` end to end on the offline fake provider and fake judge; nothing leaves the
machine."""

from __future__ import annotations

import json
import shlex
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.errors import ExternalError
from pup_finetune.evaluation import benchmark as benchmark_module
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompletion
from pup_finetune.providers.base import Sampler
from pup_finetune.providers.fake import FakeBackend, FakeRenderer

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()
OPEN_JUDGE_CLIENT = benchmark_module.open_judge_client


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def rerun(next_command: str) -> Result:
    words = shlex.split(next_command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def test_preview_lists_the_calls_and_runs_nothing(dataset: Path) -> None:
    result = invoke("benchmark", str(dataset), "--models", "fake-model")

    assert result.exit_code == 2, result.stderr
    data = output(result)
    preview = data["preview"]
    assert preview["models"][0]["model"] == "fake-model"
    assert preview["generation_calls"] == preview["cases"]
    assert preview["judge_calls"] == preview["cases"] + preview["calibration_calls"]
    assert "target_pass_rate" in preview["defaults_used"]
    assert preview["data_recipients"]
    assert data["next_command"].endswith("--judge-model fake-judge --confirm")
    assert not (dataset / "benchmark").exists()


def test_confirm_ranks_recommends_and_leads_to_training(dataset: Path) -> None:
    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))

    result = rerun(preview["next_command"])

    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    [entry] = data["leaderboard"]
    assert entry["model"] == "fake-model"
    assert entry["cases"] == preview["preview"]["cases"]
    assert data["recommendation"]["verdict"] == "fine_tune"
    assert data["recommendation"]["model"] == "fake-model"
    assert data["next_command"] == f"pup finetune train --dataset {dataset} --model fake-model"
    assert (dataset / "benchmark" / "benchmark.json").is_file()
    assert (dataset / "benchmark" / "fake-model" / "results.jsonl").is_file()
    assert "recommended: fine-tune fake-model" in result.stderr


def test_rerun_reuses_every_result(dataset: Path) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert invoke(*args).exit_code == 0
    generations = dataset / "benchmark" / "fake-model" / "generations.jsonl"
    before = generations.read_text()

    preview = output(invoke("benchmark", str(dataset), "--models", "fake-model"))["preview"]
    again = invoke(*args)

    assert preview["generation_calls"] == 0
    assert preview["calibration_calls"] == 0
    assert again.exit_code == 0, again.stderr
    assert generations.read_text() == before


def test_a_target_the_base_model_meets_needs_no_training(dataset: Path) -> None:
    result = invoke(
        "benchmark", str(dataset), "--models", "fake-model", "--target-pass-rate", "0", "--confirm"
    )

    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["recommendation"]["verdict"] == "base_meets_target"
    assert data["next_command"] is None
    assert data["next_steps"][0]["command"].startswith("pup finetune train --dataset")


def test_other_settings_in_the_same_out_dir_are_refused_at_preview(dataset: Path) -> None:
    assert invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm").exit_code == 0

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--max-tokens", "16")

    assert result.exit_code == 2
    assert "settings_conflict" in result.stderr
    assert "--out-dir" in result.stderr


def test_max_rows_replays_a_seeded_sample_and_needs_a_new_out_dir_to_change(
    dataset: Path,
) -> None:
    full = output(invoke("benchmark", str(dataset), "--models", "fake-model"))["preview"]
    previewed = invoke("benchmark", str(dataset), "--models", "fake-model", "--max-rows", "1")
    preview = output(previewed)

    shown = preview["preview"]
    assert shown["trajectories"] == shown["rows"]["test"] == 1 < full["trajectories"]
    assert shown["rows"]["test_total"] == full["rows"]["test_total"] == full["trajectories"]
    assert f"--max-rows 1 of {full['trajectories']} conversations" in previewed.stderr
    assert shown["cases"] == shown["generation_calls"] < full["cases"]
    assert f"(of {full['trajectories']}, --max-rows)" in shown["action"]
    assert shown["settings"]["max_rows"] == 1
    assert "max_rows" not in shown["defaults_used"]
    assert "--max-rows 1 --judge-model fake-judge --confirm" in preview["next_command"]
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["leaderboard"][0]["cases"] == shown["cases"]
    assert any("raise --max-rows" in warning for warning in data["warnings"])
    stored = (dataset / "benchmark" / "fake-model" / "generations.jsonl").read_text().splitlines()
    assert len({json.loads(line)["case_id"].rsplit(":", 1)[0] for line in stored}) == 1

    changed = invoke("benchmark", str(dataset), "--models", "fake-model", "--max-rows", "2")

    assert changed.exit_code == 2
    assert "settings_conflict" in changed.stderr
    assert "max_rows" in changed.stderr


def test_a_max_rows_that_samples_nothing_resumes_without_it(dataset: Path) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model")
    assert invoke(*args, "--max-rows", "50", "--confirm").exit_code == 0

    preview = invoke(*args)

    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["generation_calls"] == 0
    assert invoke(*args, "--confirm").exit_code == 0


def test_answers_cut_off_at_max_tokens_are_explained_while_and_after_it_runs(
    dataset: Path,
) -> None:
    result = invoke(
        "benchmark", str(dataset), "--models", "fake-model", "--max-tokens", "2", "--confirm"
    )

    assert result.exit_code == 0, result.stderr
    data = output(result)
    [entry] = data["leaderboard"]
    assert entry["truncated"] == entry["cases"] > 0
    assert any("reached --max-tokens (2)" in warning for warning in data["warnings"])
    assert "judged fail: cut off at --max-tokens 2; raise --max-tokens" in result.stderr
    assert f"({entry['cases']} cut off)" in result.stderr
    assert f"{entry['cases']} cut off at --max-tokens" in result.stderr.splitlines()[-1]
    assert data["next_command"] is None
    assert [step["command"] for step in data["next_steps"]] == [
        f"pup finetune benchmark {dataset} --models fake-model --judge-model fake-judge "
        "--max-tokens 4 --out-dir <NEW_DIR>",
        f"pup finetune train --dataset {dataset} --model fake-model",
    ]


def test_every_failed_pair_shows_the_judges_reason_with_progress_steps(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_PROGRESS", "steps")

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 0, result.stderr
    failed = [line for line in result.stderr.splitlines() if "(judged fail): " in line]
    assert len(failed) == output(result)["cases"]
    assert "the reference calls a tool and the candidate does not" in result.stderr


@pytest.mark.parametrize(
    ("models", "message"),
    [("fake-model,fake-model", "twice"), ("no-such-model", "unknown model"), (",", "no model")],
)
def test_bad_models_are_usage_errors(dataset: Path, models: str, message: str) -> None:
    result = invoke("benchmark", str(dataset), "--models", models)

    assert result.exit_code == 2
    assert message in result.stderr


def test_a_model_whose_session_fails_is_an_error_and_exits_1(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def no_capacity(plan: Any, candidate: Any) -> Iterator[Sampler]:
        raise ExternalError("no capacity for a sampler", source="fake")
        yield  # pragma: no cover

    monkeypatch.setattr(benchmark_module, "_base_sampler", no_capacity)

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 1
    data = output(result)
    assert data["status"] == "complete_with_errors"
    [entry] = data["leaderboard"]
    assert entry["status"] == "error"
    assert entry["error"]["message"].endswith("no capacity for a sampler")
    assert data["recommendation"]["verdict"] == "no_result"
    assert data["next_command"].startswith("pup finetune benchmark")


class AlwaysPass:
    """A judge that passes every action, a corrupted one too, so calibration fails."""

    def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            message={"role": "assistant", "content": '{"pass": true, "reason": "fine"}'}
        )


def test_a_judge_that_fails_calibration_stops_before_any_generation(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def always_pass(config: Any, **options: Any) -> Iterator[AlwaysPass]:
        yield AlwaysPass()

    monkeypatch.setattr(benchmark_module, "open_judge_client", always_pass)

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 1
    data = output(result)
    assert data["status"] == "calibration_failed"
    assert "--judge-model" in data["hint"]
    assert not (dataset / "benchmark" / "fake-model").exists()


def test_read_only_allows_an_offline_benchmark(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")

    result = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")

    assert result.exit_code == 0, result.stderr


def test_a_run_directory_is_not_a_dataset(dataset: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trained = invoke(
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run_dir),
        "--max-epochs", "1", "--confirm",
    )  # fmt: skip
    assert trained.exit_code == 0, trained.stderr

    result = invoke("benchmark", str(run_dir), "--models", "fake-model")

    assert result.exit_code == 2
    assert "not a dataset" in result.stderr


# --- reuse --------------------------------------------------------------------------------


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def no_session(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no session may open")


def test_a_model_with_every_answer_reused_opens_no_session(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert first.exit_code == 0, first.stderr
    args = ("benchmark", str(dataset), "--models", "fake-model", "--out-dir", str(dataset / "b2"))
    preview = output(invoke(*args))["preview"]
    cases = preview["cases"]
    [model] = preview["models"]
    assert model["generation_calls"] == preview["generation_calls"] == 0
    assert model["reused"] == {"generations": cases, "verdicts": cases}
    assert "(fake-model (no session for it))" in preview["action"]
    assert "every answer is already generated, so no session opens" in preview["action"]
    assert preview["data_recipients"] == [
        "test split -> the offline fake judge (judge; nothing is sent)"
    ]
    stored = lines(dataset / "benchmark" / "fake-model" / "generations.jsonl")
    assert preview["reuse"] == {
        "status": "on",
        "store": str(dataset / "replay_cache"),
        "oldest_sampled_at": min(line["sampled_at"] for line in stored),
        "saved": {
            "generation_calls": cases,
            "judge_calls": cases,
            "max_generated_tokens": cases * 4096,
            "sessions": 1,
        },
    }
    assert preview["settings"]["reuse"] is True
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    [entry] = output(result)["leaderboard"]
    assert entry["reused"] == {"generations": cases, "verdicts": cases}
    assert entry["judge_pass_rate"] == output(first)["leaderboard"][0]["judge_pass_rate"]
    assert f"reusing {cases} answers of fake-model ({cases} judged)" in result.stderr


@pytest.mark.usefixtures("judge_keys")
def test_a_benchmark_with_every_verdict_reused_needs_no_judge_key(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def fake_judge(config: Any, **options: Any) -> Iterator[FakeJudge]:
        yield FakeJudge()

    monkeypatch.setattr(benchmark_module, "open_judge_client", fake_judge)
    judged = ("benchmark", str(dataset), "--models", "fake-model")
    judged += ("--judge-model", "fireworks/some-judge", "--skip-calibration")
    assert invoke(*judged, "--confirm").exit_code == 0
    monkeypatch.delenv("FIREWORKS_API_KEY")
    monkeypatch.setattr(benchmark_module, "open_judge_client", OPEN_JUDGE_CLIENT)
    again = (*judged, "--out-dir", str(dataset / "b2"))
    preview = invoke(*again)
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["judge_calls"] == 0
    result = invoke(*again, "--confirm")
    assert result.exit_code == 0, result.stderr
    assert output(result)["status"] == "complete"


def test_no_reuse_is_frozen_in_the_out_dir(dataset: Path) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model")
    assert invoke(*args, "--confirm").exit_code == 0
    changed = invoke(*args, "--no-reuse")
    assert changed.exit_code == 2
    assert "settings_conflict" in changed.stderr
    assert "(changed: reuse)" in changed.stderr
    fresh = output(invoke(*args, "--no-reuse", "--out-dir", str(dataset / "fresh")))
    assert fresh["preview"]["reuse"] == {"status": "off"}
    assert fresh["preview"]["generation_calls"] == fresh["preview"]["cases"]
    assert "reuse" not in fresh["preview"]["defaults_used"]
    assert fresh["next_command"].endswith("--no-reuse --confirm")


def test_a_rerun_of_a_complete_benchmark_shares_its_answers(dataset: Path) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    first = invoke(*args)
    assert first.exit_code == 0, first.stderr
    store = dataset / "replay_cache"
    shutil.rmtree(store)
    assert invoke(*args).exit_code == 0
    shared = lines(store / "generations.jsonl")
    assert len(shared) == output(first)["cases"]
    assert {json.dumps(line["source"]) for line in shared} == {
        json.dumps({"command": "benchmark", "run_id": None})
    }
    assert len(lines(store / "verdicts.jsonl")) == len(shared)


def test_a_max_rows_sample_reuses_the_answers_it_matches(dataset: Path) -> None:
    assert invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm").exit_code == 0
    sample = invoke(
        "benchmark", str(dataset), "--models", "fake-model", "--max-rows", "1",
        "--out-dir", str(dataset / "sample"),
    )  # fmt: skip
    preview = output(sample)["preview"]
    assert preview["models"][0]["reused"] == {
        "generations": preview["cases"],
        "verdicts": preview["cases"],
    }
    assert preview["generation_calls"] == 0


def test_a_rerun_after_a_renderer_change_shares_nothing(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert invoke(*args).exit_code == 0
    store = dataset / "replay_cache" / "generations.jsonl"
    before = store.read_text()
    original = FakeRenderer.identity
    monkeypatch.setattr(
        FakeRenderer, "identity", lambda self: {**original(self), "renderer_version": "2"}
    )
    again = invoke(*args)
    assert again.exit_code == 0, again.stderr
    assert store.read_text() == before
