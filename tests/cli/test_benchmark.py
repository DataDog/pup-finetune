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
import re
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
from pup_finetune.providers.fake import FakeBackend, FakeProvider, FakeRenderer

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
    assert "to continue it" not in changed.stderr
    fresh = output(invoke(*args, "--no-reuse", "--out-dir", str(dataset / "fresh")))
    assert fresh["preview"]["reuse"] == {"status": "off"}
    assert fresh["preview"]["generation_calls"] == fresh["preview"]["cases"]
    assert "reuse" not in fresh["preview"]["defaults_used"]
    assert fresh["next_command"].endswith("--no-reuse --confirm")
    assert rerun(fresh["next_command"]).exit_code == 0
    again = invoke(*args, "--out-dir", str(dataset / "fresh"))
    assert again.exit_code == 2
    assert "Hint: it was made with --no-reuse: add --no-reuse to continue it, or" in again.stderr
    assert "a new --out-dir" in again.stderr


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


def unloadable(self: FakeProvider, profile: Any) -> Any:
    raise ExternalError("the tokenizer of fake-model cannot be loaded", source="huggingface")


def test_a_model_whose_renderer_cannot_load_is_an_error_entry(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke("benchmark", str(dataset), "--models", "fake-model", "--confirm").exit_code == 0
    monkeypatch.setattr(FakeProvider, "renderer", unloadable)
    args = ("benchmark", str(dataset), "--models", "fake-model", "--out-dir", str(dataset / "b2"))
    preview = invoke(*args)
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["models"][0]["reused"] == {"generations": 0, "verdicts": 0}
    result = invoke(*args, "--confirm")
    assert result.exit_code == 1, result.stderr
    [entry] = output(result)["leaderboard"]
    assert entry["error"]["message"].endswith("the tokenizer of fake-model cannot be loaded")
    assert (dataset / "b2" / "benchmark.json").exists()


def test_a_complete_rerun_whose_renderer_cannot_load_only_warns(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert invoke(*args).exit_code == 0
    shutil.rmtree(dataset / "replay_cache")
    loadable = FakeProvider.renderer
    monkeypatch.setattr(FakeProvider, "renderer", unloadable)
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    words = ["pup", "finetune", *args[:4], "--judge-model", "fake-judge", "--confirm"]
    assert output(result)["warnings"][-1] == (
        "base results were not shared: the tokenizer of fake-model cannot be loaded; run "
        f"`{shlex.join(words)}` to share them"
    )
    monkeypatch.setattr(FakeProvider, "renderer", loadable)
    assert rerun(shlex.join(words)).exit_code == 0
    assert (dataset / "replay_cache" / "generations.jsonl").exists()


def test_a_benchmark_made_before_reuse_neither_loads_its_renderer_nor_warns(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert invoke(*args).exit_code == 0
    (dataset / "benchmark" / "fake-model" / "reuse.json").unlink()
    shutil.rmtree(dataset / "replay_cache")
    monkeypatch.setattr(FakeProvider, "renderer", unloadable)
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    assert not [line for line in output(result)["warnings"] if "not shared" in line]
    assert not (dataset / "replay_cache").exists()


def test_evaluators_attach_per_model_and_one_added_later_opens_no_session(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    spec = spec_dir / "evals.toml"
    args = ("benchmark", str(dataset), "--models", "fake-model")
    assert invoke(*args, "--confirm").exit_code == 0

    def no_session(*_: Any, **__: Any) -> Any:
        raise AssertionError("no session may open")

    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    preview = output(invoke(*args, "--evaluators", str(spec)))
    shown = preview["preview"]
    assert (shown["generation_calls"], shown["judge_calls"]) == (0, 0)
    assert shown["evaluators"]["headline"] == "all"
    assert shown["settings"]["headline"] == "all"
    assert "and score each with polite, no_card_numbers and text_when_expected" in shown["action"]
    assert f"--evaluators {spec}" in preview["next_command"]
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    (entry,) = data["leaderboard"]
    assert set(entry["evaluators"]) == {"polite", "no_card_numbers", "text_when_expected", "all"}
    assert entry["headline_pass_rate"] == entry["evaluators"]["all"]
    assert data["recommendation"]["headline"] == "all"
    assert (dataset / "benchmark" / "fake-model" / "evaluators.json").is_file()
    stored = json.loads((dataset / "benchmark" / "calibration.json").read_text())
    assert stored["evaluators"]["polite"]["ok"] is True
    assert "ranked by all" in result.stderr


def test_an_evaluator_that_fails_calibration_can_be_fixed_under_its_name(
    dataset: Path, tmp_path: Path
) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    examples = spec_dir / "polite-examples.jsonl"
    swapped = {"true": "false", "false": "true"}
    examples.write_text(
        re.sub(r'(?<="pass": )(true|false)', lambda m: swapped[m[0]], examples.read_text())
    )
    result = invoke(
        "benchmark", str(dataset), "--models", "fake-model", "--confirm",
        "--evaluators", str(spec_dir / "evals.toml"),
    )  # fmt: skip
    assert result.exit_code == 1
    data = output(result)
    assert (data["status"], data["evaluator"]) == ("calibration_failed", "polite")
    assert "fix its rubric or examples, or pick another model for it" in data["hint"]


def test_evaluator_errors_are_reported_and_retried(dataset: Path, tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    outage = spec_dir / "outage"
    outage.touch()
    (spec_dir / "flaky.py").write_text(
        "import os\n\n\ndef same_as_reference(case, candidate):\n"
        f"    if candidate != case['reference'] and os.path.exists({str(outage)!r}):\n"
        "        raise ValueError('not the reference')\n"
        "    return True\n"
    )
    spec = spec_dir / "evals.toml"
    spec.write_text(
        'headline = "same_as_reference"\n[[evaluator]]\nname = "same_as_reference"\n'
        'type = "check"\nfile = "flaky.py"\n'
    )
    result = invoke(
        "benchmark", str(dataset), "--models", "fake-model", "--evaluators", str(spec), "--confirm"
    )
    assert result.exit_code == 0, result.stderr
    data = output(result)
    (entry,) = data["leaderboard"]
    assert (data["status"], entry["status"]) == ("complete_with_errors", "complete_with_errors")
    assert (entry["errors"], entry["evaluator_errors"]) == (0, data["cases"])
    assert f"{data['cases']} evaluator score(s) are missing after an error" in result.stderr
    assert data["recommendation"]["verdict"] == "no_result"
    assert "same_as_reference" in data["recommendation"]["reason"]
    assert data["next_command"] == (
        f"pup finetune benchmark {dataset} --models fake-model --judge-model fake-judge "
        f"--evaluators {spec}"
    )
    outage.unlink()
    preview = output(rerun(data["next_command"]))
    assert preview["preview"]["evaluators"]["items"][0]["reference_pass_rate"] == 1.0
    assert (preview["preview"]["generation_calls"], preview["preview"]["judge_calls"]) == (0, 0)
    retried = output(rerun(preview["next_command"]))
    assert retried["status"] == "complete"
    assert retried["leaderboard"][0]["headline_pass_rate"] == 1.0


def test_a_stored_set_s_command_runs_only_the_evaluators_it_previewed(
    dataset: Path, tmp_path: Path
) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    spec = spec_dir / "checks.toml"
    spec.write_text('[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\nfile = "checks.py"\n')
    args = ("benchmark", str(dataset), "--models", "fake-model")
    assert invoke(*args, "--evaluators", str(spec), "--confirm").exit_code == 0
    spec.write_text(
        spec.read_text() + '[[evaluator]]\nname = "text_when_expected"\ntype = "check"\n'
        'file = "checks.py"\nfunction = "answers_in_text"\n'
    )
    preview = output(invoke(*args))
    assert [item["name"] for item in preview["preview"]["evaluators"]["items"]] == [
        "no_card_numbers"
    ]
    assert "--evaluators" not in preview["next_command"]
    assert rerun(preview["next_command"]).exit_code == 0
    stored = json.loads((dataset / "benchmark" / "fake-model" / "evaluators.json").read_text())
    assert [item["name"] for item in stored["evaluators"]] == ["no_card_numbers"]


def test_a_rerun_with_nothing_left_records_the_spec_s_headline_and_files(
    dataset: Path, tmp_path: Path
) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    check = '[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\nfile = "checks.py"\n'
    (spec_dir / "checks.toml").write_text(check)
    args = ("benchmark", str(dataset), "--models", "fake-model", "--confirm")
    assert invoke(*args, "--evaluators", str(spec_dir / "checks.toml")).exit_code == 0
    moved = tmp_path / "moved"
    spec_dir.rename(moved)
    (moved / "checks.toml").write_text(f'headline = "no_card_numbers"\n{check}')
    assert invoke(*args, "--evaluators", str(moved / "checks.toml")).exit_code == 0
    stored = json.loads((dataset / "benchmark" / "fake-model" / "evaluators.json").read_text())
    assert stored["headline"] == "no_card_numbers"
    assert stored["evaluators"][0]["file"] == str(moved / "checks.py")
    assert output(invoke(*args))["recommendation"]["headline"] == "no_card_numbers"


def test_the_share_command_continues_the_evaluators_without_the_spec(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    spec = spec_dir / "checks.toml"
    spec.write_text('[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\nfile = "checks.py"\n')
    args = ("benchmark", str(dataset), "--models", "fake-model", "--evaluators", str(spec))
    assert invoke(*args, "--confirm").exit_code == 0
    shutil.rmtree(dataset / "replay_cache")
    loadable = FakeProvider.renderer
    monkeypatch.setattr(FakeProvider, "renderer", unloadable)
    warning = output(invoke(*args, "--confirm"))["warnings"][-1]
    shared = re.search(r"run `(.+)` to share them", warning)
    assert shared is not None and "--evaluators" not in shared[1]
    spec.write_text(
        spec.read_text() + '[[evaluator]]\nname = "text_when_expected"\ntype = "check"\n'
        'file = "checks.py"\nfunction = "answers_in_text"\n'
    )
    monkeypatch.setattr(FakeProvider, "renderer", loadable)
    assert rerun(shared[1]).exit_code == 0
    stored = json.loads((dataset / "benchmark" / "fake-model" / "evaluators.json").read_text())
    assert [item["name"] for item in stored["evaluators"]] == ["no_card_numbers"]


def test_a_conflict_s_hint_names_a_free_benchmark_directory(dataset: Path, tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    spec = spec_dir / "evals.toml"
    args = ("benchmark", str(dataset), "--models", "fake-model", "--evaluators", str(spec))
    assert invoke(*args, "--confirm").exit_code == 0
    spec.write_text(spec.read_text().replace('name = "polite"', 'name = "polite_v2"'))
    conflict = invoke(*args)
    assert conflict.exit_code == 2
    assert f"pass --out-dir {dataset / 'benchmark-2'}" in conflict.stderr


def test_recalibrating_the_judge_keeps_the_evaluators_calibrations(
    dataset: Path, tmp_path: Path
) -> None:
    spec_dir = tmp_path / "spec"
    shutil.copytree(ROWS.parent.parent / "evaluators", spec_dir)
    args = ("benchmark", str(dataset), "--models", "fake-model")
    args += ("--evaluators", str(spec_dir / "evals.toml"))
    assert invoke(*args, "--confirm").exit_code == 0
    path = dataset / "benchmark" / "calibration.json"
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({**stored, "judge": {**stored["judge"], "model": "another"}}))
    preview = output(invoke(*args))["preview"]
    assert preview["calibration_calls"] > 0
    assert preview["evaluators"]["items"][0]["calibration_calls"] == 0
    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    assert "calibrating the evaluator polite" not in result.stderr
    assert json.loads(path.read_text())["evaluators"] == stored["evaluators"]
