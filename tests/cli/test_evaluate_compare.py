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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune import judges
from pup_finetune.cli import app
from pup_finetune.inference import ChatCompletion, ContextLimitError, JudgeConfig
from pup_finetune.providers.base import ProviderAuthError, ProviderCapacityError
from pup_finetune.providers.fake import FakeBackend, FakeSampler
from tests.fakes.text import letters

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
        self.options: list[dict[str, Any]] = []

    def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls.append([dict(message) for message in messages])
        self.options.append(kwargs)
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
    assert data["compare_model"] is None
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
    assert shown["generation_calls_by_target"] == {"base": 0, "tuned": 0, "compare": cases + 1}
    assert f"{cases + 1} calls" in shown["billing"]
    assert shown["already_done"] == 2 * cases
    assert shown["targets"]["compare"] == "fake-compare"
    assert shown["compare"] == {
        "model": "fake-compare", "provider": "fake", "endpoint": "offline",
        "max_tokens": 16000, "reasoning_effort": None, "source": "flag",
        "skipped": 0, "skipped_by_reason": {},
    }  # fmt: skip
    assert shown["data_recipients"] == [
        "test split -> the offline fake compare model (compare; nothing is sent), "
        "the offline fake judge (judge; nothing is sent)"
    ]
    assert (
        "reuse the base and tuned answers already generated, generate each with fake-compare"
        in (shown["action"])
    )
    assert "fake provider" not in shown["billing"]
    assert "sent to fake " not in shown["data_destination"]
    assert "--compare-model fake-compare" in preview["next_command"]

    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["compare_model"] == "fake-compare"
    assert data["errors"] == {"base": 0, "tuned": 0, "compare": 0}
    assert data["headline"]["judge_pass_rate"]["compare"] is not None
    assert set(data["headline"]["tuned_minus_compare"]) == {
        "delta",
        "ci95",
        "wins",
        "regressions",
        "ties",
        "skipped",
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


@pytest.mark.usefixtures("judge_keys")
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
@pytest.mark.usefixtures("all_judge_keys")
def test_a_compare_model_that_cannot_answer_is_never_attached(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, failure: Any, exit_code: int
) -> None:
    if isinstance(failure, str):
        monkeypatch.delenv("OPENAI_API_KEY")
    else:
        serve_compare(monkeypatch, CompareModel(failure))
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    result = invoke(
        "evaluate", str(run_dir), "--compare-model", "openai/gpt-5.5", "--no-publish", "--confirm"
    )
    assert result.exit_code == exit_code, result.stderr
    assert not (run_dir / "eval" / "compare.json").exists()
    assert not (run_dir / "eval" / "generations.jsonl").exists()
    assert not (run_dir / "evaluation.json").exists()


def test_the_probe_runs_once_when_the_compare_model_is_attached(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_dir = evaluated / "eval"
    cases = len(lines(eval_dir / "results.jsonl")) // 2
    model = CompareModel()
    serve_compare(monkeypatch, model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test-key")
    args = ("evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish")
    attached = invoke(*args, "--confirm")
    assert attached.exit_code == 0, attached.stderr
    assert len(model.calls) == cases + 1
    assert model.calls[0] == [{"role": "user", "content": "Reply with the word OK."}]
    probed, replayed = model.options[0], model.options[1]
    assert probed["max_tokens"] == replayed["max_tokens"] == 16000
    assert probed["reasoning_effort"] == replayed["reasoning_effort"]
    assert [tool["type"] for tool in probed["tools"]] == ["function"]
    for name in ("generations.jsonl", "results.jsonl"):
        path = eval_dir / name
        kept = [line for line in path.read_text().splitlines() if '"target":"compare"' not in line]
        path.write_text("".join(f"{line}\n" for line in kept))
    model.calls.clear()
    again = invoke(*args, "--confirm")
    assert again.exit_code == 0, again.stderr
    assert len(model.calls) == cases


def test_the_probe_sends_no_tool_when_no_test_case_has_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain.jsonl"
    plain.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"row-plain-{index:03d}",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Say {letters(index)}.",
                        },
                        {"role": "assistant", "content": f"The number is {index}."},
                    ],
                }
            )
            + "\n"
            for index in range(30)
        )
    )
    dataset, run = tmp_path / "ds", tmp_path / "run"
    imported = invoke(
        "dataset", "import", str(dataset), "--file", str(plain), "--test-fraction", "0.3"
    )
    assert imported.exit_code == 0, imported.stderr
    trained = invoke(
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run),
        "--max-epochs", "1", "--confirm",
    )  # fmt: skip
    assert trained.exit_code == 0, trained.stderr
    model = CompareModel()
    serve_compare(monkeypatch, model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test-key")
    result = invoke(
        "evaluate", str(run), "--compare-model", "openai/gpt-5.5", "--no-publish", "--confirm"
    )
    assert result.exit_code == 0, result.stderr
    assert len(model.calls) > 1
    assert [options["tools"] for options in model.options] == [None] * len(model.calls)


def test_a_compare_model_attached_again_under_a_new_identity_is_probed_first(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test-key")
    serve_compare(monkeypatch, CompareModel())

    def no_capacity(*args: Any, **kwargs: Any) -> Any:
        raise ProviderCapacityError("no capacity right now")

    monkeypatch.setattr(FakeBackend, "open_session", no_capacity)
    args = ("evaluate", str(run_dir), "--compare-model", "openai/gpt-5.5", "--no-publish")
    first = invoke(*args, "--confirm")
    assert first.exit_code == 1
    compare_json = run_dir / "eval" / "compare.json"
    assert compare_json.exists()

    revoked = CompareModel(
        ProviderAuthError("the API key is not valid (HTTP 401)", source="openai")
    )
    serve_compare(monkeypatch, revoked)
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    again = invoke(*args, "--seed", "7", "--confirm")
    assert again.exit_code == 1
    assert "Error [provider_auth]" in again.stderr
    assert len(revoked.calls) == 1
    assert not compare_json.exists()


def test_the_preview_counts_the_probe_exactly_when_the_run_attaches(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_capacity(*args: Any, **kwargs: Any) -> Any:
        raise ProviderCapacityError("no capacity right now")

    monkeypatch.setattr(FakeBackend, "open_session", no_capacity)
    args = ("evaluate", str(run_dir), "--compare-model", "fake-compare", "--no-publish")
    assert invoke(*args, "--confirm").exit_code == 1
    assert (run_dir / "eval" / "compare.json").exists()

    kept = output(invoke(*args))["preview"]
    assert kept["generation_calls_by_target"]["compare"] == kept["cases"]
    replaced = output(invoke(*args, "--seed", "7"))["preview"]
    assert replaced["generation_calls_by_target"]["compare"] == replaced["cases"] + 1


def test_a_stored_compare_target_that_resolves_otherwise_now_is_refused(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = CompareModel()
    serve_compare(monkeypatch, model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test-key")
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


def test_cases_too_long_for_the_compare_model_point_to_another_compare_model(
    evaluated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    too_long = ContextLimitError("the request is longer than the context window", source="openai")
    serve_compare(monkeypatch, CompareModel(too_long, fail_from=2))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test-key")
    result = invoke(
        "evaluate", str(evaluated), "--compare-model", "openai/gpt-5.5", "--no-publish", "--confirm"
    )
    data = output(result)
    assert data["status"] == "complete_with_errors"
    assert "longer than the compare model's context window" in data["hint"]
    assert "another --compare-model" in data["hint"]
    assert "--judge-model" not in data["hint"]


def test_cases_too_long_for_the_trained_model_do_not_blame_the_compare_model(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_long(self: FakeSampler, *args: Any, **kwargs: Any) -> Any:
        raise ContextLimitError("the request is longer than the context window", source="fake")

    monkeypatch.setattr(FakeSampler, "generate", too_long)
    result = invoke(
        "evaluate", str(run_dir), "--compare-model", "fake-compare", "--no-publish", "--confirm"
    )
    data = output(result)
    assert data["status"] == "complete_with_errors"
    assert data["errors"]["compare"] == 0
    assert "compare model's context window" not in data["hint"]


def test_the_action_names_what_is_reused_and_what_is_left(evaluated: Path) -> None:
    args = ("evaluate", str(evaluated), "--compare-model", "fake-compare", "--no-publish")
    assert invoke(*args, "--confirm").exit_code == 0
    done = output(invoke(*args))["preview"]
    assert done["generation_calls"] == done["judge_calls"] == 0
    assert "every answer is already generated and judged" in done["action"]
    assert "judge each answer" not in done["action"]

    results = evaluated / "eval" / "results.jsonl"
    kept = [line for line in results.read_text().splitlines() if '"target":"compare"' not in line]
    results.write_text("".join(f"{line}\n" for line in kept))
    judging = output(invoke(*args))["preview"]
    assert judging["generation_calls"] == 0
    assert judging["judge_calls"] == judging["cases"]
    assert (
        "reuse the base, tuned and compare answers already generated, and judge each answer"
        in judging["action"]
    )


def test_the_action_says_the_judge_is_calibrated_first_when_that_is_all_left(
    evaluated: Path,
) -> None:
    (evaluated / "eval" / "calibration.json").unlink()
    preview = output(invoke("evaluate", str(evaluated), "--no-publish"))["preview"]
    assert preview["judge_calls"] == preview["calibration_calls"] > 0
    assert preview["action"].startswith("calibrate the fake judge first, then summarize")
    assert "judge each answer" not in preview["action"]


def test_the_progress_line_lists_every_target_that_generates(run_dir: Path) -> None:
    result = invoke(
        "evaluate", str(run_dir), "--compare-model", "fake-compare", "--no-publish", "--confirm"
    )
    assert result.exit_code == 0, result.stderr
    assert "cases, base, tuned and compare (Ctrl-C stops)" in result.stderr


def trained_on(tmp_path: Path, rewrite: Callable[[dict[str, Any]], dict[str, Any] | None]) -> Path:
    """A run trained on the fixture rows as ``rewrite`` returns them (``None`` drops one),
    with its evaluation of base and tuned complete."""
    rows = tmp_path / "rows.jsonl"
    rewritten = (rewrite(json.loads(line)) for line in ROWS.read_text().splitlines())
    rows.write_text("".join(json.dumps(row) + "\n" for row in rewritten if row is not None))
    dataset, run = tmp_path / "ds", tmp_path / "run"
    assert invoke("dataset", "import", str(dataset), "--file", str(rows)).exit_code == 0
    assert invoke(
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run),
        "--max-epochs", "1", "--confirm",
    ).exit_code == 0  # fmt: skip
    assert invoke("evaluate", str(run), "--no-publish", "--confirm").exit_code == 0
    return run


def without_tool_results(row: dict[str, Any]) -> dict[str, Any]:
    """``row`` without its tool results: every turn after a tool call holds a call chat
    completions refuse every time."""
    return {**row, "messages": [m for m in row["messages"] if m["role"] != "tool"]}


@pytest.mark.usefixtures("all_judge_keys")
def test_cases_the_compare_model_s_api_refuses_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = trained_on(tmp_path, without_tool_results)
    first_call = {
        row["id"]: next(i for i, m in enumerate(row["messages"]) if m.get("tool_calls"))
        for row in map(json.loads, (tmp_path / "rows.jsonl").read_text().splitlines())
        if any(m.get("tool_calls") for m in row["messages"])
    }
    cases = [line for line in lines(run / "eval" / "results.jsonl") if line["target"] == "base"]
    skipped = sum(
        case["row_id"] in first_call and case["message_index"] > first_call[case["row_id"]]
        for case in cases
    )
    assert 0 < skipped < len(cases)
    model = CompareModel()
    serve_compare(monkeypatch, model)

    previewed = output(invoke("evaluate", str(run), "--compare-model", "openai/gpt-5.5"))
    shown = previewed["preview"]
    assert shown["compare"]["skipped"] == skipped
    assert shown["compare"]["skipped_by_reason"] == {"unanswered_tool_call": skipped}
    # One call per case it is sent, and the probe.
    assert shown["generation_calls_by_target"]["compare"] == len(cases) - skipped + 1
    warning = f"{skipped} of {len(cases)} cases are not sent to the compare model openai/gpt-5.5"
    assert any(text.startswith(warning) for text in shown["warnings"])

    result = rerun(previewed["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["errors"] == {"base": 0, "tuned": 0, "compare": 0}
    assert len(model.calls) == len(cases) - skipped + 1
    assert data["headline"]["tuned_minus_compare"]["skipped"] == skipped
    assert any(text.startswith(warning) for text in data["warnings"])
    summary = json.loads((run / "eval" / "summary.json").read_text())
    assert summary["targets"]["compare"]["skipped"] == skipped
    assert summary["targets"]["compare"]["judge_pass"]["n"] == len(cases) - skipped
    assert summary["tuned_minus_compare"]["n_cases"] == len(cases) - skipped


def every_case_after_an_unanswered_call(row: dict[str, Any]) -> dict[str, Any] | None:
    """``row`` with only its last assistant turn trained and no tool results, kept when a
    tool call comes before that turn: its one case holds a call no result answers."""
    row = without_tool_results(row)
    assistant = [i for i, m in enumerate(row["messages"]) if m["role"] == "assistant"]
    calls = [i for i, m in enumerate(row["messages"]) if m.get("tool_calls")]
    if not calls or calls[0] >= assistant[-1]:
        return None
    for i in assistant[:-1]:
        row["messages"][i]["weight"] = 0
    return row


@pytest.mark.usefixtures("judge_keys")
def test_a_compare_model_sent_no_case_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = trained_on(tmp_path, every_case_after_an_unanswered_call)
    model = CompareModel()
    serve_compare(monkeypatch, model)
    for confirm in ([], ["--confirm"]):
        result = invoke("evaluate", str(run), "--compare-model", "openai/gpt-5.5", *confirm)
        assert result.exit_code == 2, result.stderr
        assert result.stdout == ""
        assert "Error [usage]: the compare model openai/gpt-5.5 would be sent none of the " in (
            result.stderr
        )
        assert "hold a tool call with no tool result right after it" in result.stderr
    assert model.calls == []  # not even the probe
    assert not (run / "eval" / "compare.json").exists()
