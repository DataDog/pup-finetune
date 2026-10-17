# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Base results shared by the evaluations of one dataset, on the offline fake provider.

Every test runs in development mode with the fake provider and judge (or a fake swapped in
for the judge's client); none makes a network or paid call.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion, JudgeConfig
from pup_finetune.providers.base import CheckpointRef
from pup_finetune.providers.fake import FakeBackend, FakeSession
from pup_finetune.runs import RunDir
from pup_finetune.storage import DirectoryLock

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


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        "--confirm",
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


@pytest.fixture
def first(dataset: Path, tmp_path: Path) -> Path:
    """A run trained on ``dataset`` and evaluated, which shares its base results."""
    path = tmp_path / "first"
    assert train(dataset, path).exit_code == 0
    result = invoke("evaluate", str(path), "--confirm")
    assert result.exit_code == 0, result.stderr
    return path


@pytest.fixture
def second(dataset: Path, tmp_path: Path) -> Path:
    """Another run on ``dataset``, trained with another learning rate and not evaluated."""
    path = tmp_path / "second"
    result = train(dataset, path, "--learning-rate", "0.0002")
    assert result.exit_code == 0, result.stderr
    return path


@pytest.fixture
def sampled(monkeypatch: pytest.MonkeyPatch) -> list[CheckpointRef | None]:
    """Every sampler a fake session makes: ``None`` for the base model."""
    made: list[CheckpointRef | None] = []
    original = FakeSession.sampler

    def sampler(self: FakeSession, ckpt: CheckpointRef | None) -> Any:
        made.append(ckpt)
        return original(self, ckpt)

    monkeypatch.setattr(FakeSession, "sampler", sampler)
    return made


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every fake session opened, by id."""
    opened: list[str] = []
    original = FakeBackend.open_session

    @contextmanager
    def open_session(self: FakeBackend, *args: Any, **kwargs: Any) -> Iterator[FakeSession]:
        with original(self, *args, **kwargs) as session:
            opened.append(session.session_id)
            yield session

    monkeypatch.setattr(FakeBackend, "open_session", open_session)
    return opened


class AlwaysPass:
    """A judge that passes everything, so it fails calibration."""

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            message={"role": "assistant", "content": '{"pass": true, "reason": "fine"}'}
        )


def use_judge(monkeypatch: pytest.MonkeyPatch, judge: ChatCompleter) -> None:
    @contextmanager
    def opened(config: JudgeConfig, **options: Any) -> Iterator[ChatCompleter]:
        yield judge

    monkeypatch.setattr(runner_module, "open_judge_client", opened)


# --- evaluate -----------------------------------------------------------------------------


def test_a_second_run_reuses_the_base_answers_and_verdicts_of_the_first(
    first: Path, second: Path, dataset: Path, sampled: list[CheckpointRef | None]
) -> None:
    data = output(invoke("evaluate", str(second)))
    preview = data["preview"]
    cases = preview["cases"]
    assert preview["generation_calls_by_target"] == {"base": 0, "tuned": cases}
    assert preview["judge_calls"] == cases + preview["calibration_calls"]
    assert preview["max_generated_tokens"] == cases * runner_module.DEFAULT_MAX_TOKENS
    stored = lines(first / "eval" / "generations.jsonl")
    assert preview["reuse"] == {
        "status": "on",
        "store": str(dataset / "replay_cache"),
        "by_target": {
            "base": {
                "generations": cases,
                "verdicts": cases,
                "oldest_sampled_at": min(line["sampled_at"] for line in stored),
                "from": [f"evaluate {RunDir(first).run_id}"],
            }
        },
        "saved": {
            "generation_calls": cases,
            "judge_calls": cases,
            "max_generated_tokens": cases * runner_module.DEFAULT_MAX_TOKENS,
            "samplers": 1,
        },
    }
    assert preview["action"] == (
        f"evaluate {cases} cases from 2 test conversations: reuse the base model's {cases} "
        f"answers ({cases} already judged) made earlier on this dataset, generate each with "
        "the tuned model on fake, and judge each new answer with the fake judge"
    )
    assert preview["settings"]["reuse"] is True
    assert "reuse" in preview["defaults_used"]
    assert "--no-reuse" not in data["next_command"]

    result = invoke("evaluate", str(second), "--confirm")
    assert result.exit_code == 0, result.stderr
    assert None not in sampled  # no base sampler
    assert f"reusing {cases} base answers ({cases} judged) from" in result.stderr
    done = output(result)
    assert done["reused"] == {"base": {"generations": cases, "verdicts": cases}}
    before = json.loads((first / "eval" / "summary.json").read_text())
    after = json.loads((second / "eval" / "summary.json").read_text())
    assert after["targets"]["base"]["judge_pass"] == before["targets"]["base"]["judge_pass"]
    assert after["targets"]["base"]["reused"] == {"generations": cases, "verdicts": cases}
    assert after["targets"]["tuned"]["reused"] == {"generations": 0, "verdicts": 0}
    base = [
        line for line in lines(second / "eval" / "generations.jsonl") if line["target"] == "base"
    ]
    source = {"command": "evaluate", "run_id": RunDir(first).run_id}
    assert {line["reused_from"]["run_id"] for line in base} == {source["run_id"]}
    assert sorted(line["sampled_at"] for line in base) == sorted(
        line["sampled_at"] for line in stored if line["target"] == "base"
    )
    results = [
        line for line in lines(second / "eval" / "results.jsonl") if line["target"] == "base"
    ]
    assert all(line["reused"] for line in results)
    # Only base is shared, and once: the store holds the first run's answers alone.
    store = lines(dataset / "replay_cache" / "generations.jsonl")
    assert len(store) == cases
    assert {line["source"]["run_id"] for line in store} == {source["run_id"]}


def test_another_max_tokens_reuses_nothing(first: Path, second: Path) -> None:
    preview = output(invoke("evaluate", str(second), "--max-tokens", "512"))["preview"]
    assert preview["reuse"]["by_target"] == {}
    assert preview["generation_calls_by_target"]["base"] == preview["cases"]
    assert preview["reuse"]["saved"]["samplers"] == 0


@pytest.mark.usefixtures("judge_keys")
def test_another_judge_reuses_the_answers_but_not_the_verdicts(
    first: Path, second: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_judge(monkeypatch, FakeJudge())
    judged = ["evaluate", str(second), "--judge-model", "fireworks/some-judge"]
    preview = output(invoke(*judged))["preview"]
    cases = preview["cases"]
    assert preview["reuse"]["by_target"]["base"]["generations"] == cases
    assert preview["reuse"]["by_target"]["base"]["verdicts"] == 0
    assert preview["generation_calls_by_target"]["base"] == 0
    assert preview["judge_calls"] == 2 * cases + preview["calibration_calls"]
    assert "judge each new answer with fireworks/some-judge" in preview["action"]
    result = invoke(*judged, "--confirm")
    assert result.exit_code == 0, result.stderr
    assert output(result)["reused"] == {"base": {"generations": cases, "verdicts": 0}}


def test_a_run_with_a_shorter_context_reuses_nothing(
    first: Path, dataset: Path, tmp_path: Path
) -> None:
    cut = tmp_path / "cut"
    assert train(dataset, cut, "--max-seq-len", "2048").exit_code == 0
    preview = output(invoke("evaluate", str(cut)))["preview"]
    assert preview["reuse"]["by_target"] == {}


def test_no_reuse_makes_the_answers_again_and_is_frozen(
    first: Path, second: Path, sampled: list[CheckpointRef | None]
) -> None:
    data = output(invoke("evaluate", str(second), "--no-reuse"))
    preview = data["preview"]
    assert preview["reuse"] == {"status": "off"}
    assert preview["generation_calls_by_target"]["base"] == preview["cases"]
    assert preview["settings"]["reuse"] is False
    assert "reuse" not in preview["defaults_used"]
    assert shlex.split(data["next_command"])[-2:] == ["--no-reuse", "--confirm"]
    result = invoke("evaluate", str(second), "--no-reuse", "--confirm")
    assert result.exit_code == 0, result.stderr
    assert None in sampled
    assert output(result)["reused"] == {}
    recorded = runner_module.recorded_evaluation(second)
    assert recorded is not None
    assert "--no-reuse" in shlex.split(recorded.command(second))
    # The directory froze it: without the flag, the preview fails as --confirm would.
    again = invoke("evaluate", str(second))
    assert again.exit_code == 2
    assert "Error [settings_conflict]" in again.stderr
    assert "(changed: reuse)" in again.stderr
    assert f"--out-dir {second / 'eval-2'}" in again.stderr


def test_an_evaluation_made_before_reuse_resumes_and_shares_its_results(
    first: Path, dataset: Path
) -> None:
    # As before the store existed: an identity without "reuse", and nothing shared.
    assert "reuse" not in json.loads((first / "eval" / "evaluation.json").read_text())
    for path in (dataset / "replay_cache").iterdir():
        path.unlink()
    preview = output(invoke("evaluate", str(first)))["preview"]
    assert preview["generation_calls"] == preview["judge_calls"] == 0
    result = invoke("evaluate", str(first), "--confirm")
    assert result.exit_code == 0, result.stderr
    assert len(lines(dataset / "replay_cache" / "generations.jsonl")) == preview["cases"]
    assert len(lines(dataset / "replay_cache" / "verdicts.jsonl")) == preview["cases"]


def test_a_busy_store_is_a_warning_and_the_evaluation_completes(
    dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    assert train(dataset, run).exit_code == 0
    store = dataset / "replay_cache"
    store.mkdir(mode=0o700)
    with DirectoryLock(store):
        result = invoke("evaluate", str(run), "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["warnings"][0] == (
        f"base results were not shared: another pup-finetune process is using {store}; the "
        "next run of this command shares them"
    )
    assert not (store / "generations.jsonl").exists()


@pytest.mark.usefixtures("judge_keys")
def test_a_failed_calibration_copies_nothing_and_another_judge_is_taken(
    first: Path, second: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner_module.open_judge_client
    use_judge(monkeypatch, AlwaysPass())
    failed = invoke("evaluate", str(second), "--judge-model", "fireworks/some-judge", "--confirm")
    assert failed.exit_code == 1, failed.stderr
    assert output(failed)["status"] == "calibration_failed"
    assert not (second / "eval" / "generations.jsonl").exists()
    assert not (second / "eval" / "results.jsonl").exists()
    monkeypatch.setattr(runner_module, "open_judge_client", original)
    fixed = invoke("evaluate", str(second), "--judge-model", "fake-judge", "--confirm")
    assert fixed.exit_code == 0, fixed.stderr
    assert output(fixed)["reused"]["base"]["verdicts"] > 0


def test_a_compare_model_with_every_base_answer_reused_samples_only_tuned(
    first: Path,
    second: Path,
    sampled: list[CheckpointRef | None],
    sessions: list[str],
) -> None:
    compared = ["evaluate", str(second), "--compare-model", "fake-compare"]
    preview = output(invoke(*compared))["preview"]
    cases = preview["cases"]
    assert preview["generation_calls_by_target"] == {
        "base": 0,
        "tuned": cases,
        "compare": cases + 1,  # and the probe
    }
    assert "generate each with the tuned model on fake and with fake-compare" in (preview["action"])
    result = invoke(*compared, "--confirm")
    assert result.exit_code == 0, result.stderr
    assert len(sessions) == 1
    assert len(sampled) == 1
    assert sampled[0] is not None


# --- train --evaluate ---------------------------------------------------------------------


def test_train_evaluate_reuses_the_base_answers_in_its_session(
    first: Path, dataset: Path, tmp_path: Path, sampled: list[CheckpointRef | None]
) -> None:
    run = tmp_path / "evaluated"
    args = [
        "train",
        "--provider",
        "fake",
        "--model",
        "fake-model",
        "--dataset",
        str(dataset),
        "--run-dir",
        str(run),
        "--max-epochs",
        "2",
        "--evaluate",
    ]
    preview = output(invoke(*args))["preview"]["evaluation"]
    cases = preview["cases"]
    assert preview["reuse"]["by_target"]["base"]["generations"] == cases
    assert preview["generation_calls_by_target"] == {"base": 0, "tuned": cases}
    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    assert None not in sampled
    evaluation = output(result)["evaluation"]
    assert evaluation["status"] == "complete"
    assert evaluation["reused"] == {"base": {"generations": cases, "verdicts": cases}}
