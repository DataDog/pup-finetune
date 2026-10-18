# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`evaluate` and `train --evaluate` end to end on the offline fake provider and fake judge.

Every test runs in development mode; no test makes a network or paid call. The judge is the
offline fake judge (the default for a fake run), or a fake swapped in for
:func:`pup_finetune.evaluation.runner.open_judge_client`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.display import display_path as display
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.evaluation import cases as cases_module
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.evaluation import targets as targets_module
from pup_finetune.evaluation.fake_judge import FAKE_JUDGE_MODEL, FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion, ContextLimitError, JudgeConfig
from pup_finetune.providers import fake as fake_module
from pup_finetune.providers.fake import FakeRenderer
from pup_finetune.runs import RunDir, RunState, model_id_for_run
from pup_finetune.runtime import Runtime
from pup_finetune.storage import append_jsonl
from pup_finetune.training.budget import Budget
from tests.fakes.rows import import_with_refused_test_row

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()
OPEN_JUDGE_CLIENT = runner_module.open_judge_client


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
        "--max-epochs",
        "2",
        *extra,
    ]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


@pytest.fixture
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    result = invoke(*train_args(dataset, path, "--confirm"))
    assert result.exit_code == 0, result.stderr
    assert output(result)["status"] == "completed"
    return path


class AlwaysPass:
    """A judge that passes everything, so it passes corrupted actions and fails calibration."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatCompletion:
        self.calls += 1
        return ChatCompletion(
            message={"role": "assistant", "content": '{"pass": true, "reason": "fine"}'}
        )


class Outage:
    """A judge that cannot be reached: every call raises, as after the client's retries."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatCompletion:
        self.calls += 1
        raise ExternalError("the judge endpoint is unreachable", source="judge")


class TooLong(Outage):
    """A judge whose context window every case is longer than."""

    def complete(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        self.calls += 1
        raise ContextLimitError("the request is longer than the context window", source="judge")


class Interrupting(FakeJudge):
    """The offline judge, but its first call sends this process ``signum`` (or, for
    ``KeyboardInterrupt``, raises it, as a second Ctrl-C would).

    The call runs on a replay worker, and the main thread runs the signal handler only when
    it next gets the interpreter, so on a busy machine the worker could finish every other
    pair before the cancel is seen. With :attr:`cancel` set (see :func:`use_interrupting`),
    the call therefore waits, after the signal, until the evaluation's cancel event is set:
    exactly the pair in flight finishes, whatever the scheduling.
    """

    def __init__(self, signum: int | None = signal.SIGTERM) -> None:
        super().__init__()
        self.signum = signum
        self.fired = False
        self.cancel: threading.Event | None = None

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatCompletion:
        if not self.fired:
            self.fired = True
            if self.signum is None:
                raise KeyboardInterrupt
            os.kill(os.getpid(), self.signum)
            if self.cancel is not None and not self.cancel.wait(30):
                raise AssertionError("the signal never cancelled the evaluation")
        return super().complete(messages, **kwargs)


def use_judge(monkeypatch: pytest.MonkeyPatch, judge: ChatCompleter) -> None:
    @contextmanager
    def opened(config: JudgeConfig, **options: Any) -> Iterator[ChatCompleter]:
        yield judge

    monkeypatch.setattr(runner_module, "open_judge_client", opened)


def use_interrupting(monkeypatch: pytest.MonkeyPatch, judge: Interrupting) -> None:
    """:func:`use_judge` with ``judge``, which is handed the cancel event each replay runs
    with (the one a SIGTERM sets, directly or through the spend-cap watcher of `train
    --evaluate`), so it can wait for the cancel to be seen."""
    use_judge(monkeypatch, judge)
    replay = runner_module.replay

    def recording(*args: Any, cancel: threading.Event, **kwargs: Any) -> Any:
        judge.cancel = cancel
        return replay(*args, cancel=cancel, **kwargs)

    monkeypatch.setattr(runner_module, "replay", recording)


# --- evaluate: the preview ----------------------------------------------------------------


def test_the_preview_exits_2_with_the_calls_and_pins_the_rerun(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["status"] == "confirmation_required"
    preview = data["preview"]
    cases = preview["cases"]
    assert cases == 4  # two test rows, at most two turns each
    assert preview["generation_calls"] == 2 * cases
    assert preview["calibration_calls"] == 3 * 4
    assert preview["judge_calls"] == 2 * cases + preview["calibration_calls"]
    assert preview["max_generated_tokens"] == 2 * cases * runner_module.DEFAULT_MAX_TOKENS
    assert preview["targets"]["base"] == "fake/fake-model"
    assert "epoch" in preview["targets"]["tuned"]
    assert preview["judge_model"] == FAKE_JUDGE_MODEL
    assert preview["out_dir"] == str(run_dir / "eval")
    assert "task success" in preview["caveat"]
    assert not (run_dir / "eval").exists()  # a preview writes nothing
    # The settings typed (none) and the judge, which decides where the data goes.
    assert data["next_command"] == shlex.join(
        [
            "pup",
            "finetune",
            "evaluate",
            str(run_dir),
            "--judge-model",
            FAKE_JUDGE_MODEL,
            "--confirm",
        ]
    )


def test_the_preview_repeats_only_the_settings_given(run_dir: Path) -> None:
    out_dir = run_dir.parent / "elsewhere"
    result = invoke("evaluate", str(run_dir), "--out-dir", str(out_dir), "--max-tokens", "512")
    assert result.exit_code == 2, result.stderr
    assert output(result)["next_command"] == shlex.join(
        [
            "pup",
            "finetune",
            "evaluate",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--judge-model",
            FAKE_JUDGE_MODEL,
            "--max-tokens",
            "512",
            "--confirm",
        ]
    )


@pytest.mark.usefixtures("all_judge_keys")
def test_the_preview_names_the_judge_s_provider_and_pins_its_spec(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir), "--judge-model", "anthropic/claude-opus-5-5")
    assert result.exit_code == 2, result.stderr
    data = output(result)
    preview = data["preview"]
    assert preview["judge_model"] == "anthropic/claude-opus-5-5"
    assert preview["judge_provider"] == "anthropic"
    assert preview["judge_endpoint"] == "https://api.anthropic.com"
    assert "to Anthropic (api.anthropic.com) to judge" in preview["data_destination"]
    words = shlex.split(data["next_command"])
    assert words[words.index("--judge-model") + 1] == "anthropic/claude-opus-5-5"


def test_the_fake_judge_preview_sends_nothing_to_judge(run_dir: Path) -> None:
    preview = output(invoke("evaluate", str(run_dir)))["preview"]
    assert "offline fake judge" in preview["data_destination"]


def test_an_anthropic_judge_without_the_extra_fails_with_the_install_hint(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = invoke("evaluate", str(run_dir), "--judge-model", "anthropic/claude-opus-5-5")
    assert result.exit_code == 2
    assert "uv sync --extra anthropic" in result.stderr


def test_a_judge_with_no_model_is_a_usage_error(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir), "--judge-model", "openai/")
    assert result.exit_code == 2
    assert "has no model" in result.stderr


def test_skipping_calibration_saves_its_judge_calls(run_dir: Path) -> None:
    preview = output(invoke("evaluate", str(run_dir), "--skip-calibration"))["preview"]
    assert preview["calibration_calls"] == 0
    assert preview["judge_calls"] == 2 * preview["cases"]


def test_the_judge_must_not_be_the_model_being_trained(run_dir: Path) -> None:
    # Only the model id counts, whichever provider the judge names.
    for model in (
        "accounts/fake/models/fake-model",
        "fireworks/fake-model",
        "anthropic/fake-model",
    ):
        result = invoke("evaluate", str(run_dir), "--judge-model", model)
        assert result.exit_code == 2
        assert "being trained" in result.stderr


def test_the_judge_must_not_be_the_run_s_tuned_model(run_dir: Path) -> None:
    promoted = json.loads((run_dir / "result.json").read_text())["promoted_model"]
    for model in (promoted["model_id"], promoted["uri"], promoted["model_id"].upper()):
        result = invoke("evaluate", str(run_dir), "--judge-model", f"fireworks/{model}")
        assert result.exit_code == 2, result.stderr
        assert "this run's tuned model" in result.stderr
    assert not (run_dir / "eval").exists()


def test_read_only_allows_the_preview_but_not_the_evaluation(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    judge = ("--judge-model", "fireworks/some-judge")  # a paid judge: the run leaves the machine
    assert invoke("evaluate", str(run_dir), *judge).exit_code == 2
    result = invoke("evaluate", str(run_dir), *judge, "--confirm")
    assert result.exit_code == 2
    assert result.stderr.startswith(
        "Error [read_only]: evaluating on fake is a paid or remote run and PUP_READ_ONLY is set"
    )
    assert not (run_dir / "eval").exists()


def test_read_only_allows_an_evaluation_that_stays_offline(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fake provider and the fake judge, unpublished: nothing leaves the machine.
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    preview = invoke("evaluate", str(run_dir), "--no-publish")
    assert preview.exit_code == 2
    assert output(preview)["next_command"] is not None
    result = invoke("evaluate", str(run_dir), "--no-publish", "--confirm")
    assert result.exit_code == 0, result.stderr


def test_auto_approve_does_not_confirm(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_AUTO_APPROVE", "1")
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2, result.stderr
    assert output(result)["status"] == "confirmation_required"
    assert not (run_dir / "eval").exists()


def test_an_empty_test_split_has_nothing_to_evaluate(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cases_module, "cases_from_dataset", lambda *args, **kwargs: [])
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2
    assert "nothing to evaluate" in result.stderr


def test_a_changed_renderer_is_refused(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = FakeRenderer.identity
    monkeypatch.setattr(
        FakeRenderer, "identity", lambda self: {**identity(self), "version": "changed"}
    )
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2
    assert "Error [settings_conflict]: " in result.stderr
    assert "renderer changed" in result.stderr
    assert not (run_dir / "eval").exists()


def test_a_run_that_did_not_complete_is_refused(dataset: Path, tmp_path: Path) -> None:
    run = tmp_path / "crashed"
    crashed = invoke(*train_args(dataset, run, "--provider-option", "fail_at_step=1", "--confirm"))
    assert crashed.exit_code == 1
    result = invoke("evaluate", str(run))
    assert result.exit_code == 2
    assert "interrupted" in result.stderr


def test_a_failed_run_is_refused(run_dir: Path) -> None:
    path = run_dir / "state.json"
    state = json.loads(path.read_text())
    path.write_text(json.dumps({**state, "phase": "failed", "resume_mode": "finalize_only"}))
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2
    assert "failed, not completed or cancelled" in result.stderr


def test_a_run_cancelled_before_its_first_epoch_has_nothing_to_evaluate(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    # Stands in for a cancel request (``RunState.request_cancel``) before the first step.
    monkeypatch.setattr(RunState, "cancel_requested", lambda self: True)
    trained = invoke(*train_args(dataset, run, "--confirm"))
    assert output(trained)["status"] == "cancelled", trained.stderr
    assert output(trained)["best_epoch"] is None
    result = invoke("evaluate", str(run))
    assert result.exit_code == 2
    assert "Error [wrong_phase]: " in result.stderr
    assert "no best epoch" in result.stderr


def test_a_changed_dataset_is_refused(run_dir: Path, dataset: Path) -> None:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    extra = {**rows[0], "id": "row-extra-001"}
    extra["source"] = {**extra["source"], "trace_id": "synthetic-trace-extra"}
    extra["messages"] = [*extra["messages"], {"role": "user", "content": "And tomorrow?"}]
    extra["messages"].append({"role": "assistant", "content": "Tomorrow looks dry."})
    path = dataset / "extra.jsonl"
    append_jsonl(path, [extra])
    assert invoke("dataset", "import", str(dataset), "--file", str(path)).exit_code == 0
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2
    assert "Error [settings_conflict]: " in result.stderr
    assert "changed since" in result.stderr


# --- evaluate --confirm -------------------------------------------------------------------


def test_evaluate_writes_a_summary_where_the_tuned_model_wins(run_dir: Path) -> None:
    preview = output(invoke("evaluate", str(run_dir)))
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["calibration"]["ok"] is True
    # Not published for want of a push: no next command, and deploying is a step.
    assert data["next_command"] is None
    assert f"pup finetune deploy {run_dir}" in [step["command"] for step in data["next_steps"]]
    out = run_dir / "eval"
    assert data["files"] == {
        "summary": str(out / "summary.json"),
        "results": str(out / "results.jsonl"),
    }
    summary = json.loads((out / "summary.json").read_text())
    # What it prints is the decision; summary.json has the rest.
    headline = data["headline"]
    rates = headline["judge_pass_rate"]
    assert rates["delta"] == pytest.approx(summary["tuned_minus_base"]["judge_pass_rate"])
    assert rates["ci95"] == pytest.approx(summary["tuned_minus_base"]["ci95"])
    assert rates["tuned"] == summary["targets"]["tuned"]["judge_pass"]["rate"]
    assert headline["paired"] == {
        key: summary["paired"][key] for key in ("wins", "regressions", "ties")
    }
    assert (headline["cases"], headline["trajectories"]) == (
        summary["n_cases"],
        summary["n_trajectories"],
    )
    assert data["metrics"]["tool_decision_match"] == {
        target: summary["targets"][target]["metrics"]["tool_decision_match"]["rate"]
        for target in ("base", "tuned")
    }
    assert next(iter(data["metrics"])) == "tool_decision_match"
    assert data["errors"] == {"base": 0, "tuned": 0}
    assert data["caveat"] == summary["caveat"]
    assert "summary" not in data
    delta = summary["tuned_minus_base"]["judge_pass_rate"]
    assert delta > 0
    tuned = summary["targets"]["tuned"]["judge_pass"]["rate"]
    base = summary["targets"]["base"]["judge_pass"]["rate"]
    assert tuned > base
    assert summary["paired"]["wins"] > 0
    assert summary["paired"]["regressions"] == 0
    assert summary["errors"] == {"base": 0, "tuned": 0}
    assert summary["n_cases"] == preview["preview"]["cases"]
    assert summary["caveat"].startswith("Scores measure agreement")
    for name in ("evaluation.json", "generations.jsonl", "results.jsonl", "calibration.json"):
        assert (out / name).is_file()
    # The tuned checkpoint was re-created in a new session from the best training checkpoint.
    assert data["tuned_checkpoint"].startswith("ev")


def test_evaluating_again_generates_nothing(run_dir: Path) -> None:
    first = output(invoke("evaluate", str(run_dir), "--confirm"))
    generations = (run_dir / "eval" / "generations.jsonl").read_text()
    preview = output(invoke("evaluate", str(run_dir)))["preview"]
    assert preview["already_done"] == 2 * preview["cases"]
    assert preview["generation_calls"] == preview["judge_calls"] == 0
    again = invoke("evaluate", str(run_dir), "--confirm")
    assert again.exit_code == 0, again.stderr
    assert output(again)["tuned_checkpoint"] is None  # no session was opened
    assert output(again)["headline"] == first["headline"]
    assert output(again)["metrics"] == first["metrics"]
    assert (run_dir / "eval" / "generations.jsonl").read_text() == generations


def test_other_settings_in_the_same_out_dir_are_refused(run_dir: Path) -> None:
    assert invoke("evaluate", str(run_dir), "--confirm").exit_code == 0
    # The preview knows --confirm would refuse, so it fails instead of offering it.
    preview = invoke("evaluate", str(run_dir), "--max-tokens", "512")
    assert preview.exit_code == 2
    assert preview.stdout == ""
    assert "holds an evaluation with other settings (changed: max_tokens)" in preview.stderr
    assert f"Hint: evaluate into a new directory: pass --out-dir {run_dir / 'eval-2'}" in (
        preview.stderr
    )
    result = invoke("evaluate", str(run_dir), "--max-tokens", "512", "--confirm")
    assert result.exit_code == 2
    assert "--out-dir" in result.stderr
    elsewhere = invoke(
        "evaluate",
        str(run_dir),
        "--max-tokens",
        "512",
        "--out-dir",
        str(run_dir / "e2"),
        "--confirm",
    )
    assert elsewhere.exit_code == 0, elsewhere.stderr


@pytest.mark.usefixtures("judge_keys")
def test_a_failed_calibration_generates_nothing(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = AlwaysPass()
    use_judge(monkeypatch, judge)
    result = invoke("evaluate", str(run_dir), "--judge-model", "fireworks/some-judge", "--confirm")
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["status"] == "calibration_failed"
    assert data["calibration"]["ok"] is False
    assert judge.calls == 3 * 4  # calibration only
    # The progress line names the checks that failed: every corruption was judged a pass.
    lines = result.stderr.splitlines()
    [line] = [line for line in lines if line.startswith("18:03:16 calibration failed: ")]
    assert line.startswith("18:03:16 calibration failed: 4/12 checks passed; ")
    assert " corrupted (" in line and "): judged pass" in line and line.endswith(" more")
    assert result.stderr.splitlines()[-1] == (
        "calibration failed: the judge fireworks/some-judge got 8 of 12 checks wrong, so "
        "nothing was evaluated; the judge passed a corrupted action or failed a recorded "
        "one, so its verdicts cannot be trusted; pick another --judge-model (or rerun with "
        "--skip-calibration to evaluate anyway)"
    )
    assert not (run_dir / "eval" / "generations.jsonl").exists()
    assert not (run_dir / "eval" / "results.jsonl").exists()
    # With a sound judge, the same directory takes it: nothing was generated yet.
    monkeypatch.setattr(runner_module, "open_judge_client", OPEN_JUDGE_CLIENT)
    fixed = invoke("evaluate", str(run_dir), "--confirm")
    assert fixed.exit_code == 0, fixed.stderr


@pytest.mark.usefixtures("judge_keys")
def test_a_rerun_without_judge_model_keeps_the_judge_its_directory_holds(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_judge(monkeypatch, AlwaysPass())
    judged = ["evaluate", str(run_dir), "--judge-model", "fireworks/some-judge"]
    assert invoke(*judged, "--skip-calibration", "--confirm").exit_code == 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-synthetic-0000-test-key")
    preview = output(invoke("evaluate", str(run_dir), "--skip-calibration"))["preview"]
    assert (preview["judge_model"], preview["judge_source"]) == (
        "fireworks/some-judge",
        "evaluation directory",
    )


@pytest.mark.usefixtures("judge_keys")
def test_skip_calibration_judges_without_checking(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = AlwaysPass()
    use_judge(monkeypatch, judge)
    result = invoke(
        "evaluate",
        str(run_dir),
        "--judge-model",
        "fireworks/some-judge",
        "--skip-calibration",
        "--confirm",
    )
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["calibration"] is None
    assert judge.calls == 2 * data["headline"]["cases"]
    assert data["headline"]["judge_pass_rate"]["delta"] == 0.0


def test_a_judge_outage_is_not_a_complete_evaluation(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outage = Outage()
    use_judge(monkeypatch, outage)
    result = invoke("evaluate", str(run_dir), "--skip-calibration", "--confirm")
    assert result.exit_code == 1, result.stderr  # no target has a single verdict
    data = output(result)
    cases = data["headline"]["cases"]
    assert data["status"] == "complete_with_errors"
    assert data["errors"] == {"base": cases, "tuned": cases}
    assert data["headline"]["judge_pass_rate"]["tuned"] is None
    assert "retry" in data["hint"]
    assert data["next_command"] is not None
    # The generations are kept: the rerun only judges, and completes.
    monkeypatch.setattr(runner_module, "open_judge_client", OPEN_JUDGE_CLIENT)
    preview = output(invoke("evaluate", str(run_dir), "--skip-calibration"))["preview"]
    assert preview["generation_calls"] == 0
    assert preview["judge_calls"] == 2 * cases
    again = rerun(data["next_command"] + " --confirm")  # it previews without
    assert again.exit_code == 0, again.stderr
    assert output(again)["status"] == "complete"
    assert output(again)["errors"] == {"base": 0, "tuned": 0}


def test_a_cancel_keeps_what_finished_and_the_rerun_completes(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_interrupting(monkeypatch, Interrupting(signal.SIGTERM))
    args = ("evaluate", str(run_dir), "--skip-calibration", "--concurrency", "1", "--confirm")
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "partial"
    assert data["pending"] > 0
    assert json.loads((run_dir / "eval" / "summary.json").read_text())["status"] == "partial"
    assert data["next_command"] is not None
    again = rerun(data["next_command"] + " --confirm")  # it previews without
    assert again.exit_code == 0, again.stderr
    assert output(again)["status"] == "complete"
    assert output(again)["headline"]["judge_pass_rate"]["delta"] > 0


def test_a_second_ctrl_c_stops_at_once_with_how_to_continue(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_judge(monkeypatch, Interrupting(None))
    result = invoke(
        "evaluate", str(run_dir), "--skip-calibration", "--concurrency", "1", "--confirm"
    )
    assert result.exit_code == 1
    assert "stopped at once" in result.stderr
    assert "pup finetune evaluate" in result.stderr
    monkeypatch.setattr(runner_module, "open_judge_client", OPEN_JUDGE_CLIENT)
    again = invoke("evaluate", str(run_dir), "--skip-calibration", "--confirm")
    assert again.exit_code == 0, again.stderr
    assert output(again)["status"] == "complete"


def test_the_fake_judge_is_for_development_only() -> None:
    with pytest.raises(UsageError, match="development"):
        runner_module.resolve_judge(FAKE_JUDGE_MODEL, provider="other", runtime=Runtime())
    config, source = runner_module.resolve_judge(None, provider="other", runtime=Runtime())
    assert config == JudgeConfig(reasoning_effort="none")
    assert source == "default: no Anthropic or OpenAI credential found"
    dev = Runtime(dev_mode=True)
    fake, source = runner_module.resolve_judge(None, provider="fake", runtime=dev)
    assert (fake.model, source) == (FAKE_JUDGE_MODEL, "fake run")
    given, source = runner_module.resolve_judge("fw/x-y", provider="fake", runtime=dev)
    assert (given.model, source) == ("fw/x-y", "given")


def test_signals_cancel_first_and_stop_at_once_second() -> None:
    cancel = threading.Event()
    with runner_module.cancel_on_signals(cancel):
        os.kill(os.getpid(), signal.SIGINT)
        assert cancel.wait(5)
        os.kill(os.getpid(), signal.SIGTERM)  # a repeated SIGTERM changes nothing
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            threading.Event().wait(5)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# --- train --evaluate ---------------------------------------------------------------------


def test_train_previews_the_evaluation_calls(dataset: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    result = invoke(*train_args(dataset, run, "--evaluate", "--eval-max-tokens", "256"))
    assert result.exit_code == 2, result.stderr
    data = output(result)
    evaluation = data["preview"]["evaluation"]
    assert evaluation["cases"] == 4
    assert evaluation["generation_calls"] == 8
    assert evaluation["judge_calls"] == 8 + 12
    assert evaluation["max_generated_tokens"] == 8 * 256
    assert evaluation["out_dir"] == str(run / "eval")
    words = shlex.split(data["next_command"])
    assert "--evaluate" in words
    assert words[words.index("--eval-max-tokens") + 1] == "256"
    assert words[words.index("--eval-judge-model") + 1] == FAKE_JUDGE_MODEL
    assert not run.exists()


@pytest.mark.usefixtures("all_judge_keys")
def test_train_evaluate_names_every_recipient_the_judge_included(
    dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    extra = ("--evaluate", "--eval-judge-model", "anthropic/claude-opus-5-5")
    result = invoke(*train_args(dataset, run, *extra))
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert preview["data_recipients"] == [
        "training rows -> fake",
        "test split -> fake (generate), Anthropic api.anthropic.com (judge)",
    ]
    assert preview["data_destination"].startswith("training data is sent to fake; ")
    assert "Anthropic (api.anthropic.com) to judge" in preview["data_destination"]
    assert preview["evaluation"]["data_recipients"] == preview["data_recipients"][1:]


def test_train_evaluate_names_its_own_flag_to_skip_publishing(
    dataset: Path, tmp_path: Path
) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--evaluate"))
    assert result.exit_code == 2, result.stderr
    (warning,) = output(result)["preview"]["evaluation"]["warnings"]
    assert "will not be published" in warning
    assert warning.endswith("or pass --eval-no-publish")
    assert "--no-publish" not in warning.replace("--eval-no-publish", "")


def test_evaluate_names_its_recipients_and_its_flag_to_skip_publishing(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 2, result.stderr
    preview = output(result)["preview"]
    assert preview["data_recipients"] == [
        "test split -> fake (generate), the offline fake judge (judge; nothing is sent)"
    ]
    (warning,) = preview["warnings"]
    assert warning.endswith("or pass --no-publish")


@pytest.mark.usefixtures("judge_keys")
def test_read_only_previews_offer_no_confirm(
    dataset: Path, run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    judge = "fireworks/some-judge"  # a paid judge: the run would leave the machine
    for args in (
        train_args(dataset, tmp_path / "other", "--evaluate", "--eval-judge-model", judge),
        ["evaluate", str(run_dir), "--judge-model", judge],
    ):
        result = invoke(*args)
        assert result.exit_code == 2, result.stderr
        data = output(result)
        assert data["status"] == "confirmation_required"
        assert data["next_command"] is None
        assert data["preview"]["read_only"] == "--confirm is refused while PUP_READ_ONLY is set"
        assert "Next:" not in result.stderr
        assert "pup --read-only" in result.stderr
    assert not (tmp_path / "other").exists()
    assert not (run_dir / "eval").exists()


def test_train_evaluate_evaluates_in_the_training_session(dataset: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    preview = output(invoke(*train_args(dataset, run, "--evaluate")))
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "completed"
    assert data["warnings"] == []
    evaluation = data["evaluation"]
    assert evaluation["status"] == "complete"
    assert evaluation["headline"]["judge_pass_rate"]["delta"] > 0
    # The tuned target sampled the best epoch's own sampler checkpoint: no new session.
    best = json.loads((run / "result.json").read_text())["checkpoints"]["best_sampler"]
    assert evaluation["tuned_checkpoint"] == best["name"]
    assert "checkpoints" not in data  # they stay in result.json
    summary = json.loads((run / "eval" / "summary.json").read_text())
    assert evaluation["headline"]["cases"] == summary["n_cases"]
    assert evaluation["files"]["summary"] == str(run / "eval" / "summary.json")
    # A standalone evaluate finds it done.
    again = output(invoke("evaluate", str(run)))["preview"]
    assert again["generation_calls"] == 0


def test_train_evaluate_refuses_the_run_s_tuned_model_as_judge(
    dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    run_id = "ft-20261006123045-abcdef"
    judge = f"accounts/synthetic-team/models/{model_id_for_run(run_id)}"
    args = train_args(dataset, run, "--run-id", run_id, "--evaluate")
    result = invoke(*args, "--eval-judge-model", judge, "--confirm")
    assert result.exit_code == 2, result.stderr
    assert "this run's tuned model" in result.stderr
    assert not run.exists()


def test_train_evaluate_needs_the_foreground(dataset: Path, tmp_path: Path) -> None:
    result = invoke(*train_args(dataset, tmp_path / "run", "--evaluate", "--detach"))
    assert result.exit_code == 2
    assert "--evaluate" in result.stderr


@pytest.mark.usefixtures("judge_keys")
def test_train_evaluate_trains_nothing_when_calibration_fails(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_judge(monkeypatch, AlwaysPass())
    run = tmp_path / "run"
    result = invoke(
        *train_args(dataset, run, "--evaluate", "--eval-judge-model", "fireworks/x-y", "--confirm")
    )
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["status"] == "calibration_failed"
    assert not run.exists()
    # train spells the judge and calibration flags with --eval-, in the hint and the summary.
    for text in (data["hint"], result.stderr.splitlines()[-1]):
        assert "pick another --eval-judge-model (or rerun with --eval-skip-calibration" in text


def test_a_failed_evaluation_keeps_the_trained_run_and_exits_1(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no sampler today")

    monkeypatch.setattr(targets_module, "session_samplers", broken)
    run = tmp_path / "run"
    result = invoke(*train_args(dataset, run, "--evaluate", "--confirm"))
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["status"] == "completed"
    assert data["promoted_model"] is not None
    assert data["evaluation"]["status"] == "failed"
    assert "evaluate" in data["evaluation"]["next_command"]
    assert any("after training failed" in warning for warning in data["warnings"])
    assert RunDir(run).load_state().phase == "completed"


def test_the_retry_after_a_failed_train_evaluate_pins_its_settings(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no sampler today")

    with monkeypatch.context() as patch:
        patch.setattr(targets_module, "session_samplers", broken)
        run = tmp_path / "run"
        settings = (
            "--eval-max-tokens",
            "256",
            "--eval-seed",
            "3",
            "--eval-max-cases",
            "2",
        )
        result = invoke(*train_args(dataset, run, "--evaluate", *settings, "--confirm"))
    assert result.exit_code == 1, result.stderr
    retry = output(result)["evaluation"]["next_command"]
    words = shlex.split(retry)
    for flag, value in (("--max-tokens", "256"), ("--seed", "3")):
        assert words[words.index(flag) + 1] == value, retry
    assert words[words.index("--max-cases") + 1] == "2", retry
    assert "--judge-model" in words
    # Followed as written (with --confirm), evaluation.json's identity check accepts it.
    followed = invoke(*words[2:], "--confirm")
    assert followed.exit_code == 0, followed.stderr
    assert output(followed)["status"] == "complete"


@pytest.mark.parametrize("where", ["file", "earlier", "run_dir"])
def test_an_unusable_eval_out_dir_is_refused_before_training(
    dataset: Path, tmp_path: Path, where: str
) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    if where == "file":
        out.write_text("a file where the evaluation would go")
    elif where == "earlier":
        out.mkdir()
        (out / "evaluation.json").write_text("{}")
    else:
        out = run
    for extra in ((), ("--confirm",)):
        args = train_args(dataset, run, "--evaluate", "--eval-out-dir", str(out), *extra)
        result = invoke(*args)
        assert result.exit_code == 2, result.stderr
        assert "--eval-out-dir" in result.stderr
        assert not (run / "run.json").exists()  # nothing was trained


def test_train_evaluate_skips_a_budget_capped_run(dataset: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    args = train_args(dataset, run, "--max-train-tokens", "1", "--evaluate", "--confirm")
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["stopped_reason"] == "budget_exhausted"
    evaluation = data["evaluation"]
    assert evaluation["status"] == "skipped"
    # No epoch finished: nothing to evaluate, now or later, and the cause is the token cap.
    assert "no epoch finished" in evaluation["reason"]
    assert "token budget, --max-train-tokens" in evaluation["reason"]
    assert "spend cap" not in evaluation["reason"]
    assert evaluation["next_command"] is None
    assert data["next_command"] is None
    assert any("no epoch finished" in entry["why"] for entry in data["next_steps"])
    assert not (run / "eval" / "results.jsonl").exists()
    # The judge was never calibrated: the preview knew no epoch could finish.
    assert "calibrating the judge fake" not in result.stderr
    assert "not calibrating the judge" in result.stderr
    # status says the same: no evaluate for a run without a best epoch.
    status = output(invoke("status", str(run)))
    assert status["next_command"] is None
    assert "evaluation" not in status


def test_a_train_preview_warns_when_no_epoch_can_finish(dataset: Path, tmp_path: Path) -> None:
    preview = output(invoke(*train_args(dataset, tmp_path / "run")))["preview"]
    per_epoch = preview["tokens_per_epoch"]
    capped = output(
        invoke(*train_args(dataset, tmp_path / "run", "--max-train-tokens", str(per_epoch - 1)))
    )["preview"]
    assert any(
        "no epoch will finish" in warning and "nothing is promoted or evaluated" in warning
        for warning in capped["warnings"]
    )
    enough = output(
        invoke(*train_args(dataset, tmp_path / "run", "--max-train-tokens", str(per_epoch)))
    )["preview"]
    assert not any("no epoch will finish" in warning for warning in enough["warnings"])


def test_train_evaluate_names_the_token_budget_after_an_epoch(
    dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    per_epoch = output(invoke(*train_args(dataset, run)))["preview"]["tokens_per_epoch"]
    args = train_args(dataset, run, "--max-train-tokens", str(per_epoch), "--evaluate")
    result = invoke(*args, "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["stopped_reason"] == "budget_exhausted"
    assert data["best_epoch"] == 1
    evaluation = data["evaluation"]
    assert evaluation["status"] == "skipped"
    assert "token budget, --max-train-tokens" in evaluation["reason"]
    assert "spend cap" not in evaluation["reason"]
    assert shlex.split(evaluation["next_command"])[:3] == ["pup", "finetune", "evaluate"]
    # status names the same command, which previews without a settings conflict.
    status = output(invoke("status", str(run)))
    assert status["next_command"] == evaluation["next_command"]
    preview = rerun(status["next_command"])
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["status"] == "confirmation_required"


def test_status_after_train_evaluate_with_settings_continues_that_evaluation(
    dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    result = invoke(*train_args(dataset, run, "--evaluate", "--eval-seed", "7", "--confirm"))
    assert result.exit_code == 0, result.stderr
    assert output(result)["evaluation"]["status"] == "complete"
    status = output(invoke("status", str(run)))
    # Complete: status shows where it is and its headline. The dataset was never pushed, so
    # as after the evaluation, no deploy comes next: its steps push, then rerun to publish.
    assert status["evaluation"]["status"] == "complete"
    assert status["evaluation"]["out_dir"] == display(run / "eval")
    assert status["evaluation"]["headline"]["judge_pass_rate"]["tuned"] is not None
    evaluated = output(result)["evaluation"]
    assert evaluated["next_command"] is None
    assert status["next_command"] is None
    assert status["next_steps"] == evaluated["next_steps"][:3]
    push, again, deploy = (item["command"] for item in status["next_steps"])
    assert shlex.split(push)[2:4] == ["dataset", "push"]
    assert again.endswith(" --confirm") and "--seed 7" in again
    assert deploy == f"pup finetune deploy {shlex.quote(display(run))}"
    assert "evaluation complete in" in invoke("status", str(run)).stderr
    # Unpublished on request, it goes on to deploy.
    record = json.loads((run / "evaluation.json").read_text())
    record["evaluations"]["eval"]["publication"] = "disabled"
    (run / "evaluation.json").write_text(json.dumps(record))
    following = output(invoke("status", str(run)))["next_command"]
    assert following == f"pup finetune deploy {shlex.quote(display(run))}"
    # Not complete (as after a crash or a Ctrl-C): the recorded command, every setting
    # pinned, previews without the settings conflict a bare `evaluate` would hit.
    record = json.loads((run / "evaluation.json").read_text())
    record["evaluations"]["eval"]["status"] = "partial"
    record["evaluations"]["eval"]["complete"] = False
    (run / "evaluation.json").write_text(json.dumps(record))
    following = output(invoke("status", str(run)))["next_command"]
    assert "--seed 7" in following
    preview = rerun(following)
    assert preview.exit_code == 2, preview.stderr
    assert "settings_conflict" not in preview.stderr
    bare = invoke("evaluate", str(run))
    assert "settings_conflict" in bare.stderr


def test_status_after_evaluate_confirm_no_longer_says_evaluate(run_dir: Path) -> None:
    status = output(invoke("status", str(run_dir)))
    assert shlex.split(status["next_command"])[:3] == ["pup", "finetune", "evaluate"]
    # Unpublished on request (a missing push would hold the deploy back for a push first).
    evaluated = rerun(output(invoke("evaluate", str(run_dir), "--no-publish"))["next_command"])
    assert evaluated.exit_code == 0, evaluated.stderr
    status = output(invoke("status", str(run_dir)))
    assert status["evaluation"]["status"] == "complete"
    assert shlex.split(status["next_command"])[:3] == ["pup", "finetune", "deploy"]
    # Once deployed, there is no next command.
    deployed = rerun(output(invoke("deploy", str(run_dir)))["next_command"])
    assert deployed.exit_code == 0, deployed.stderr
    assert output(invoke("status", str(run_dir)))["next_command"] is None


SPEND = ("--max-spend-usd", "1000", "--hourly-rate-usd", "1")


def spend_cap_reached(monkeypatch: pytest.MonkeyPatch, reached: Callable[[], bool]) -> None:
    """Training never reaches the cap; the whole cap, reserve included (what the step after
    training is held to), is reached once ``reached()`` says so."""

    def spend_exhausted(self: Budget, *, use_reserve: bool = False) -> str | None:
        return "synthetic spend reached max_spend_usd" if use_reserve and reached() else None

    monkeypatch.setattr(Budget, "spend_exhausted", spend_exhausted)


def test_train_evaluate_skips_a_run_whose_spend_cap_is_reached(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spend_cap_reached(monkeypatch, lambda: True)
    run = tmp_path / "run"
    result = invoke(*train_args(dataset, run, *SPEND, "--evaluate", "--confirm"))
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["stopped_reason"] == "completed"
    evaluation = data["evaluation"]
    assert evaluation["status"] == "skipped"
    assert "spend cap" in evaluation["reason"]
    assert shlex.split(evaluation["next_command"])[:3] == ["pup", "finetune", "evaluate"]
    assert not (run / "eval" / "results.jsonl").exists()


class SlowAfterFirst(FakeJudge):
    """The offline judge; its first verdict marks the spend cap reached and takes a while,
    so the cap's watcher stops the evaluation before the next pair starts."""

    def __init__(self) -> None:
        super().__init__()
        self.judged = threading.Event()

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatCompletion:
        if not self.judged.is_set():
            self.judged.set()
            threading.Event().wait(0.3)
        return super().complete(messages, **kwargs)


def test_train_evaluate_stops_at_the_spend_cap(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = SlowAfterFirst()
    use_judge(monkeypatch, judge)
    spend_cap_reached(monkeypatch, judge.judged.is_set)
    monkeypatch.setattr(runner_module, "SPEND_POLL_S", 0.01)
    run = tmp_path / "run"
    args = train_args(dataset, run, *SPEND, "--evaluate", "--eval-skip-calibration")
    result = invoke(*args, "--eval-concurrency", "1", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "completed"
    evaluation = data["evaluation"]
    assert evaluation["status"] == "partial"
    assert evaluation["pending"] > 0
    assert any("spend cap" in warning for warning in evaluation["warnings"])
    assert shlex.split(evaluation["next_command"])[:3] == ["pup", "finetune", "evaluate"]


def test_train_evaluate_skips_a_cancelled_run_that_evaluate_then_evaluates(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    # Stands in for a cancel request (``RunState.request_cancel``) after the first epoch.
    monkeypatch.setattr(RunState, "cancel_requested", lambda self: self.best is not None)
    trained = invoke(*train_args(dataset, run, "--evaluate", "--confirm"))
    data = output(trained)
    assert (data["status"], data["stopped_reason"]) == ("cancelled", "cancelled"), trained.stderr
    assert data["best_epoch"] == 1
    evaluation = data["evaluation"]
    assert evaluation["status"] == "skipped"
    assert "cancelled" in evaluation["reason"]
    assert not (run / "eval" / "results.jsonl").exists()
    result = rerun(evaluation["next_command"] + " --confirm")  # it previews without
    assert result.exit_code == 0, result.stderr
    assert output(result)["status"] == "complete"
    assert (run / "eval" / "summary.json").is_file()


def test_a_cancel_during_the_evaluation_keeps_the_trained_run(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_interrupting(monkeypatch, Interrupting(signal.SIGTERM))
    run = tmp_path / "run"
    args = train_args(
        dataset,
        run,
        "--evaluate",
        "--eval-skip-calibration",
        "--eval-concurrency",
        "1",
        "--confirm",
    )
    result = invoke(*args)
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "completed"
    assert data["promoted_model"] is not None
    evaluation = data["evaluation"]
    assert evaluation["status"] == "partial"
    assert evaluation["pending"] > 0
    monkeypatch.setattr(runner_module, "open_judge_client", OPEN_JUDGE_CLIENT)
    again = rerun(evaluation["next_command"] + " --confirm")  # it previews without
    assert again.exit_code == 0, again.stderr
    assert output(again)["status"] == "complete"


# --- the evaluation layer -----------------------------------------------------------------


def test_loading_the_cli_does_not_load_the_evaluation_or_its_http_client() -> None:
    code = (
        "import sys, pup_finetune.cli; "
        "print(sorted(m for m in ('httpx', 'pup_finetune.evaluation.replay', "
        "'pup_finetune.datadog.experiments') if m in sys.modules)); "
        "print('pup_finetune.evaluation.runner' in sys.modules)"
    )
    done = subprocess.run(  # noqa: S603 - this interpreter, fixed arguments
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    # The CLI loads the runner, and nothing the runner drives.
    assert done.stdout.split() == ["[]", "True"]


def test_the_evaluation_package_exports_lazily() -> None:
    import pup_finetune.evaluation as evaluation

    assert evaluation.run_replay.__module__ == "pup_finetune.evaluation.replay"
    assert sorted(evaluation.__all__) == sorted(evaluation._EXPORTS)
    for name, module in evaluation._EXPORTS.items():
        assert getattr(evaluation, name) is getattr(importlib.import_module(module), name)
    with pytest.raises(AttributeError):
        _ = evaluation.no_such_name  # type: ignore[attr-defined]


def test_cases_too_long_for_the_judge_say_a_retry_cannot_fix_them(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_judge(monkeypatch, TooLong())
    result = invoke("evaluate", str(run_dir), "--skip-calibration", "--confirm")
    data = output(result)
    cases = data["headline"]["cases"]
    assert data["status"] == "complete_with_errors"
    summary = json.loads((run_dir / "eval" / "summary.json").read_text())
    assert summary["targets"]["tuned"]["errors_too_long"] == cases
    assert "retry cannot fix" in data["hint"]
    assert "larger context window" in data["hint"]


def test_generations_cut_off_at_max_tokens_are_counted_and_warned_about(run_dir: Path) -> None:
    result = invoke("evaluate", str(run_dir), "--max-tokens", "2", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    truncated = data["headline"]["truncated"]
    assert truncated["base"] > 0 and truncated["tuned"] > 0
    assert any("reached --max-tokens (2)" in warning for warning in data["warnings"])


def results_of(eval_dir: Path) -> list[dict[str, Any]]:
    path = eval_dir / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_case_whose_prompt_the_model_refuses_is_left_out_at_planning(tmp_path: Path) -> None:
    dataset, refused = import_with_refused_test_row(tmp_path)
    run = tmp_path / "run"
    assert invoke(*train_args(dataset, run, "--confirm")).exit_code == 0

    preview = output(invoke("evaluate", str(run), "--no-publish"))
    excluded = preview["preview"]["excluded_cases"]
    assert excluded > 0
    assert preview["preview"]["excluded_by_reason"] == {"unsupported_message": excluded}
    assert any(" test cases are left out: " in w for w in preview["preview"]["warnings"])

    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert any(" test cases are left out: " in w for w in data["warnings"])
    rows = {record["row_id"] for record in results_of(run / "eval")}
    assert rows and refused not in rows


def test_a_prompt_refused_while_sampling_is_an_error_no_rerun_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A renderer whose check misses what its sampler then refuses: the backstop.
    monkeypatch.setattr(FakeRenderer, "prompt_refusal", lambda self, messages, tools: None)
    dataset, refused = import_with_refused_test_row(tmp_path)
    run = tmp_path / "run"
    assert invoke(*train_args(dataset, run, "--confirm")).exit_code == 0

    result = invoke("evaluate", str(run), "--no-publish", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete_with_errors"
    unrenderable = [record for record in results_of(run / "eval") if record["status"] == "error"]
    assert {record["row_id"] for record in unrenderable} == {refused}
    assert {record["error"]["type"] for record in unrenderable} == {"RendererError"}
    summary = json.loads((run / "eval" / "summary.json").read_text())
    for target in summary["targets"].values():
        assert target["errors_unrenderable"] == target["errors"] > 0
    assert "which a retry cannot fix" in data["hint"]
    assert "evaluate" not in (data["next_command"] or "")
    assert output(invoke("status", str(run)))["next_command"].split()[2] == "deploy"

    again = invoke("evaluate", str(run), "--no-publish")
    assert output(again)["preview"]["generation_calls"] == 0


def test_previews_say_when_the_tokenizer_runs_repository_code(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = dataclasses.replace(fake_module.FAKE_PROFILE, trust_remote_code=True)
    monkeypatch.setattr(fake_module, "FAKE_PROFILE", remote)
    run = tmp_path / "run"
    notice = "tokenizer runs Python code from the Hugging Face repository fake/tokenizer"

    train = output(invoke(*train_args(dataset, run)))["preview"]
    assert any(notice in warning for warning in train["warnings"])
    assert invoke(*train_args(dataset, run, "--confirm")).exit_code == 0
    evaluate = output(invoke("evaluate", str(run), "--no-publish"))["preview"]
    assert any(notice in warning for warning in evaluate["warnings"])
