# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`evaluate --compare-model`: a third target, attached to a complete evaluation without
redoing base and tuned. Development mode, the fake provider, judge and compare model."""

from __future__ import annotations

import json
import shlex
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import judges
from pup_finetune.cli import app
from pup_finetune.inference import ChatCompletion, JudgeConfig
from pup_finetune.providers.base import ProviderAuthError
from pup_finetune.providers.fake import FakeBackend

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


def rerun(next_command: str) -> Result:
    words = shlex.split(next_command)
    assert words[:2] == ["pup", "finetune"]
    return invoke(*words[2:])


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    dataset, run = tmp_path / "ds", tmp_path / "run"
    assert invoke("dataset", "import", str(dataset), "--file", str(ROWS)).exit_code == 0
    trained = invoke(
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run),
        "--max-epochs", "1", "--confirm",
    )  # fmt: skip
    assert trained.exit_code == 0, trained.stderr
    return run


@pytest.fixture
def evaluated(run_dir: Path) -> Path:
    result = invoke("evaluate", str(run_dir), "--confirm", "--no-publish")
    assert result.exit_code == 0, result.stderr
    return run_dir


class CompareModel:
    """A compare model's chat client: it answers every call with text, and from call
    ``fail_from`` on (counting from 1) raises ``fail`` instead."""

    def __init__(self, fail: Exception | None = None, *, fail_from: int = 1) -> None:
        self.fail, self.fail_from = fail, fail_from
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls.append([dict(message) for message in messages])
        if self.fail is not None and len(self.calls) >= self.fail_from:
            raise self.fail
        content = "OK" if len(self.calls) == 1 else "I would explain the plan."
        return ChatCompletion(
            message={"role": "assistant", "content": content},
            model=str(kwargs["model"]),
            finish_reason="stop",
        )


def serve_compare(monkeypatch: pytest.MonkeyPatch, model: CompareModel) -> None:
    """Make every remote compare model (and judge) ``model``."""

    @contextmanager
    def open_client(config: JudgeConfig, **kwargs: Any) -> Iterator[CompareModel]:
        yield model

    monkeypatch.setattr(judges, "open_client", open_client)


def no_session(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no training session may open")


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_target(path: Path) -> Counter[str]:
    return Counter(str(line["target"]) for line in lines(path))


def test_a_complete_evaluation_offers_a_compare_step(evaluated: Path) -> None:
    data = output(invoke("evaluate", str(evaluated), "--confirm", "--no-publish"))
    (compare_step,) = [s for s in data["next_steps"] if "--compare-model" in s["command"]]
    assert compare_step["command"] == (
        f"pup finetune evaluate {evaluated} --judge-model fake-judge --compare-model <MODEL> "
        "--no-publish"
    )


def test_attaching_a_compare_model_redoes_neither_base_nor_tuned(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_dir = evaluated / "eval"
    before = {name: (eval_dir / name).read_bytes() for name in ("evaluation.json",)}
    stored_results = [line for line in lines(eval_dir / "results.jsonl")]
    cases = len(stored_results) // 2

    preview = output(
        invoke("evaluate", str(evaluated), "--compare-model", "fake-compare", "--no-publish")
    )
    shown = preview["preview"]
    assert shown["generation_calls_by_target"] == {"base": 0, "tuned": 0, "compare": cases}
    assert shown["already_done"] == 2 * cases
    assert shown["targets"]["compare"] == "fake-compare"
    assert shown["compare"] == {
        "model": "fake-compare", "provider": "fake", "endpoint": "offline",
        "max_tokens": 16000, "reasoning_effort": None, "source": "flag",
    }  # fmt: skip
    assert (
        "the offline fake compare model (compare; nothing is sent)" in shown["data_recipients"][0]
    )
    assert "--compare-model fake-compare" in preview["next_command"]

    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["errors"] == {"base": 0, "tuned": 0, "compare": 0}
    assert data["headline"]["judge_pass_rate"]["compare"] is not None
    assert set(data["headline"]["tuned_minus_compare"]) == {
        "delta",
        "ci95",
        "wins",
        "regressions",
        "ties",
    }
    assert by_target(eval_dir / "generations.jsonl") == {
        "base": cases,
        "tuned": cases,
        "compare": cases,
    }
    assert lines(eval_dir / "results.jsonl")[: 2 * cases] == stored_results
    assert {name: (eval_dir / name).read_bytes() for name in before} == before
    summary = json.loads((eval_dir / "summary.json").read_text())
    assert summary["tuned_minus_compare"]["n_cases"] == cases
    compare = json.loads((eval_dir / "compare.json").read_text())
    assert set(compare) == {"spec", "model"}
    assert compare["model"]["max_tokens"] == 16000
    assert compare["spec"] == "fake-compare"


def test_a_rerun_without_the_flag_continues_the_compare_target(evaluated: Path) -> None:
    attached = invoke(
        "evaluate", str(evaluated), "--compare-model", "fake-compare", "--confirm", "--no-publish"
    )
    assert attached.exit_code == 0, attached.stderr
    preview = output(invoke("evaluate", str(evaluated), "--no-publish"))
    assert preview["preview"]["compare"]["source"] == "evaluation directory"
    assert preview["preview"]["generation_calls"] == 0
    assert "--compare-model fake-compare" in preview["next_command"]
    again = rerun(preview["next_command"])
    assert again.exit_code == 0, again.stderr
    assert "compare" in output(again)["errors"]
    record = json.loads((evaluated / "evaluation.json").read_text())
    assert record["evaluations"]["eval"]["settings"]["compare_model"] == "fake-compare"


def test_another_compare_model_needs_a_new_out_dir(evaluated: Path) -> None:
    attached = invoke(
        "evaluate", str(evaluated), "--compare-model", "fake-compare", "--confirm", "--no-publish"
    )
    assert attached.exit_code == 0, attached.stderr
    other = invoke("evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish")
    assert other.exit_code == 2
    assert "Error [settings_conflict]" in other.stderr
    assert "--out-dir" in other.stderr


def test_another_compare_model_is_refused_even_before_a_compare_pair(evaluated: Path) -> None:
    eval_dir = evaluated / "eval"
    attached = invoke(
        "evaluate", str(evaluated), "--compare-model", "fake-compare", "--confirm", "--no-publish"
    )
    assert attached.exit_code == 0, attached.stderr
    for name in ("generations.jsonl", "results.jsonl"):
        path = eval_dir / name
        kept = [line for line in path.read_text().splitlines() if '"target":"compare"' not in line]
        path.write_text("".join(f"{line}\n" for line in kept))
    other = invoke(
        "evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish", "--confirm"
    )
    assert other.exit_code == 2
    assert "Error [settings_conflict]" in other.stderr
    assert by_target(eval_dir / "generations.jsonl")["compare"] == 0


def test_the_judge_may_not_be_the_compare_model(run_dir: Path) -> None:
    result = invoke(
        "evaluate",
        str(run_dir),
        "--judge-model",
        "openai/gpt-5.5",
        "--compare-model",
        "openai/gpt-5.5",
    )
    assert result.exit_code == 2
    assert "is the compare model" in result.stderr


def test_the_fake_compare_model_needs_development_mode(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUP_FINETUNE_DEV")
    result = invoke("evaluate", str(run_dir), "--compare-model", "fake-compare")
    assert result.exit_code == 2


def test_read_only_allows_an_offline_compare_and_refuses_a_paid_one(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    offline = output(
        invoke("evaluate", str(evaluated), "--compare-model", "fake-compare", "--no-publish")
    )
    assert offline["next_command"] is not None
    paid = invoke("evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish")
    assert paid.exit_code == 2
    data = output(paid)
    assert data["next_command"] is None
    assert "read_only" in data["preview"]
    assert "OpenAI api.openai.com (compare model)" in data["preview"]["data_recipients"][0]
    assert "OpenAI bills each compare-model call" in data["preview"]["billing"]


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [
        ("no key", 2),
        (ProviderAuthError("the API key is not valid (HTTP 401)", source="openai"), 1),
    ],
)
def test_a_compare_model_that_cannot_answer_fails_before_anything_is_written(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, failure: Any, exit_code: int
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if not isinstance(failure, str):
        serve_compare(monkeypatch, CompareModel(failure))
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    result = invoke(
        "evaluate", str(run_dir), "--compare-model", "openai/gpt-5.5", "--no-publish", "--confirm"
    )
    assert result.exit_code == exit_code, result.stderr
    assert not (run_dir / "eval").exists()
    assert not (run_dir / "evaluation.json").exists()


def test_the_probe_runs_once_when_the_compare_model_is_attached(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_dir = evaluated / "eval"
    cases = len(lines(eval_dir / "results.jsonl")) // 2
    model = CompareModel()
    serve_compare(monkeypatch, model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic")
    args = ("evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish")
    attached = invoke(*args, "--confirm")
    assert attached.exit_code == 0, attached.stderr
    assert len(model.calls) == cases + 1
    assert model.calls[0] == [{"role": "user", "content": "Reply with the word OK."}]
    for name in ("generations.jsonl", "results.jsonl"):
        path = eval_dir / name
        kept = [line for line in path.read_text().splitlines() if '"target":"compare"' not in line]
        path.write_text("".join(f"{line}\n" for line in kept))
    model.calls.clear()
    again = invoke(*args, "--confirm")
    assert again.exit_code == 0, again.stderr
    assert len(model.calls) == cases


def test_a_stored_compare_target_that_resolves_otherwise_now_is_refused(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = CompareModel()
    serve_compare(monkeypatch, model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic")
    attached = invoke(
        "evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish", "--confirm"
    )
    assert attached.exit_code == 0, attached.stderr
    path = evaluated / "eval" / "compare.json"
    stored = json.loads(path.read_text())
    stored["model"]["base_url"] = "https://elsewhere.example/v1"
    path.write_text(json.dumps(stored))
    model.calls.clear()
    for confirm in ((), ("--confirm",)):
        result = invoke("evaluate", str(evaluated), "--no-publish", *confirm)
        assert result.exit_code == 2
        assert "Error [settings_conflict]" in result.stderr
        assert "changed: base_url" in result.stderr
    assert model.calls == []
