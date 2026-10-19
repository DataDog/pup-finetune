# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`evaluate --evaluators`: the user's judges and checks, added to an evaluation without
generating again. Development mode, the fake provider and the fake judge."""

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

from pup_finetune import judges
from pup_finetune.cli import app
from pup_finetune.errors import ExternalError
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import JudgeConfig
from pup_finetune.providers.fake import FakeBackend

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "fixtures" / "rows" / "tool_use.jsonl"
FIXTURES = ROOT / "fixtures" / "evaluators"

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


def no_session(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no training session may open")


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    target = tmp_path / "spec"
    shutil.copytree(FIXTURES, target)
    return target / "evals.toml"


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


def checks_only(spec: Path, headline: str = "no_card_numbers") -> Path:
    path = spec.with_name("checks.toml")
    path.write_text(
        f'headline = "{headline}"\n[[evaluator]]\nname = "no_card_numbers"\ntype = "check"\n'
        'file = "checks.py"\n'
    )
    return path


def test_the_preview_shows_the_evaluators_and_what_they_cost(run_dir: Path, spec: Path) -> None:
    result = invoke("evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish")
    assert result.exit_code == 2
    preview = output(result)["preview"]
    block = preview["evaluators"]
    assert (block["headline"], block["source"]) == ("all", "flag")
    polite, card, text = block["items"]
    cases = preview["cases"]
    assert polite | {"rubric_sha256": "-"} == {
        "name": "polite", "type": "judge", "model": "fake-judge", "provider": "fake",
        "rubric_sha256": "-", "attached": False, "calls": 2 * cases, "calibration": "examples",
        "calibration_calls": 2,
    }  # fmt: skip
    assert (card["type"], card["reference_pass_rate"]) == ("check", 1.0)
    assert text["function"] == "answers_in_text"
    assert preview["settings"]["headline"] == "all"
    assert "evaluators" in preview["settings"]
    assert "evaluators" not in preview["defaults_used"]
    assert preview["data_recipients"][-1] == (
        "test cases and answers -> the offline fake judge (evaluator polite; nothing is sent)"
    )
    assert "checks.py is Python you wrote" in preview["notes"][0]
    assert output(result)["next_command"].endswith(f"--evaluators {spec} --no-publish --confirm")


@pytest.mark.usefixtures("all_judge_keys")
def test_a_paid_evaluator_judge_bills_receives_and_warns_uncalibrated(
    run_dir: Path, spec: Path
) -> None:
    spec.write_text(
        '[[evaluator]]\nname = "tone"\ntype = "judge"\nmodel = "anthropic/claude-sonnet-5-5"\n'
        'rubric = "Pass when polite."\n'
    )
    preview = output(invoke("evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish"))[
        "preview"
    ]
    assert "Anthropic bills each evaluator call" in preview["billing"]
    assert "Anthropic api.anthropic.com (evaluator tone)" in preview["data_recipients"][-1]
    assert (
        "evaluator tone is not calibrated: add examples (answers you graded) to its spec"
        in preview["warnings"]
    )


def test_confirm_scores_and_heads_the_result_with_the_evaluator(run_dir: Path, spec: Path) -> None:
    preview = output(invoke("evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish"))
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    headline = data["headline"]
    assert headline["evaluator"] == "all"
    assert set(headline["evaluators"]) == {"polite", "no_card_numbers", "text_when_expected", "all"}
    assert headline["evaluators"]["no_card_numbers"]["base"] == 1.0
    assert "headline all (the judge and every evaluator)" in result.stderr
    stored = json.loads((run_dir / "eval" / "calibration.json").read_text())
    assert stored["evaluators"]["polite"]["ok"] is True
    assert (run_dir / "eval" / "evaluators.json").is_file()


def test_an_evaluator_added_to_a_complete_evaluation_generates_and_opens_nothing(
    evaluated: Path, spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_dir = evaluated / "eval"
    generations = (eval_dir / "generations.jsonl").read_bytes()
    monkeypatch.setattr(FakeBackend, "open_session", no_session)

    @contextmanager
    def no_client(*args: Any, **kwargs: Any) -> Iterator[Any]:
        raise AssertionError("adding a check opens no judge client")
        yield

    monkeypatch.setattr(judges, "open_client", no_client)
    checks = checks_only(spec)
    preview = output(
        invoke("evaluate", str(evaluated), "--evaluators", str(checks), "--no-publish")
    )
    shown = preview["preview"]
    assert (shown["generation_calls"], shown["judge_calls"]) == (0, 0)
    assert shown["action"].startswith("score ")
    assert "reusing the base and tuned answers" in shown["action"]
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    assert (eval_dir / "generations.jsonl").read_bytes() == generations
    assert output(result)["headline"]["evaluator"] == "no_card_numbers"


def test_a_complete_evaluation_offers_the_evaluators_step(evaluated: Path) -> None:
    data = output(invoke("evaluate", str(evaluated), "--confirm", "--no-publish"))
    (found,) = [s for s in data["next_steps"] if "--evaluators" in s["command"]]
    assert found["command"] == (
        f"pup finetune evaluate {evaluated} --judge-model fake-judge --no-publish "
        "--evaluators <EVALUATORS_FILE>"
    )
    assert "nothing is generated again" in found["why"]


def grow(spec: Path, marker: Path) -> None:
    spec.with_name("more.py").write_text(
        f"def short_answer(case, candidate):\n    open({str(marker)!r}, 'w').close()\n"
        "    return True\n"
    )
    spec.write_text(
        spec.read_text()
        + '[[evaluator]]\nname = "short_answer"\ntype = "check"\nfile = "more.py"\n'
    )


def test_a_stored_set_s_command_runs_only_the_evaluators_it_previewed(
    run_dir: Path, spec: Path
) -> None:
    args = ("evaluate", str(run_dir), "--no-publish")
    assert invoke(*args, "--evaluators", str(spec), "--confirm").exit_code == 0
    marker = spec.with_name("ran")
    grow(spec, marker)
    preview = output(invoke(*args, "--compare-model", "fake-compare"))
    assert preview["preview"]["evaluators"]["source"] == "evaluation directory"
    assert "--evaluators" not in preview["next_command"]
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    assert not marker.exists()
    stored = json.loads((run_dir / "eval" / "evaluators.json").read_text())
    assert [item["name"] for item in stored["evaluators"]] == [
        "polite", "no_card_numbers", "text_when_expected"
    ]  # fmt: skip


def test_steps_after_a_run_continue_its_evaluators_without_the_spec(
    run_dir: Path, spec: Path
) -> None:
    data = output(invoke("evaluate", str(run_dir), "--evaluators", str(spec), "--confirm"))
    assert data["publication"]["status"] == "skipped"
    marker = spec.with_name("ran")
    grow(spec, marker)
    steps = data["next_steps"] + output(invoke("status", str(run_dir)))["next_steps"]
    confirmed = [s["command"] for s in steps if s["command"].endswith("--confirm")]
    assert len(confirmed) == 2
    assert not [s["command"] for s in steps if "--evaluators" in s["command"]]
    for command in confirmed:
        result = rerun(command)
        assert result.exit_code == 0, result.stderr
    assert not marker.exists()


def test_a_changed_rubric_conflicts_and_a_renamed_one_scores(
    run_dir: Path, spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ("evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish")
    assert invoke(*args, "--confirm").exit_code == 0
    spec.write_text(spec.read_text().replace("courteous", "kind"))
    conflict = invoke(*args)
    assert conflict.exit_code == 2
    assert "settings_conflict" in conflict.stderr
    assert "keep polite as it was attached and add the new version under another name" in (
        conflict.stderr
    )
    spec.write_text(spec.read_text().replace('name = "polite"', 'name = "polite_v2"'))
    conflict = invoke(*args)
    assert "leaves out" in conflict.stderr
    spec.write_text(
        spec.read_text()
        + '[[evaluator]]\nname = "polite"\ntype = "judge"\nexamples = "polite-examples.jsonl"\n'
        'rubric = """\nPass when the answer is courteous and does not blame the user. Fail '
        'when it is curt or sarcastic.\n"""\n'
    )
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    renamed = invoke(*args, "--confirm")
    assert renamed.exit_code == 0, renamed.stderr
    assert "polite_v2" in output(renamed)["headline"]["evaluators"]


def test_changing_only_the_headline_costs_nothing(
    evaluated: Path, spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = checks_only(spec)
    assert invoke("evaluate", str(evaluated), "--evaluators", str(checks), "--no-publish",
                  "--confirm").exit_code == 0  # fmt: skip
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    checks_only(spec, headline="judge")
    preview = output(
        invoke("evaluate", str(evaluated), "--evaluators", str(checks), "--no-publish")
    )
    shown = preview["preview"]
    assert (shown["generation_calls"], shown["judge_calls"]) == (0, 0)
    assert shown["evaluators"]["items"][0]["attached"] is True
    result = rerun(preview["next_command"])
    assert output(result)["headline"]["evaluator"] == "judge"


def test_a_check_that_raises_on_a_reference_fails_the_preview(run_dir: Path, spec: Path) -> None:
    (spec.parent / "checks.py").write_text(
        "def no_card_numbers(case, candidate):\n    raise KeyError('content')\n\n\n"
        "def answers_in_text(case, candidate):\n    return 1\n"
    )
    result = invoke("evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish")
    assert result.exit_code == 2
    assert "Error [usage]: the check no_card_numbers failed on the reference of case" in (
        result.stderr
    )
    assert "KeyError" in result.stderr


def test_a_failed_custom_calibration_stops_before_anything_is_generated(
    run_dir: Path, spec: Path
) -> None:
    examples = spec.with_name("polite-examples.jsonl")
    examples.write_text(
        examples.read_text().replace('"pass": true', '"pass": X').replace('"pass": false', '"pass": true')
        .replace('"pass": X', '"pass": false')
    )  # fmt: skip
    result = invoke(
        "evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish", "--confirm"
    )
    assert result.exit_code == 1
    data = output(result)
    assert (data["status"], data["evaluator"]) == ("calibration_failed", "polite")
    assert "fix its rubric or examples, or pick another model for it" in data["hint"]
    assert "calibration failed: the evaluator polite got 2 of 2 checks wrong" in result.stderr
    assert not (run_dir / "eval" / "generations.jsonl").exists()
    assert not (run_dir / "eval" / "evaluators.json").exists()


def test_an_attached_evaluator_that_fails_calibration_keeps_its_name(
    evaluated: Path, spec: Path
) -> None:
    args = ("evaluate", str(evaluated), "--evaluators", str(spec), "--no-publish", "--confirm")
    assert invoke(*args, "--skip-calibration").exit_code == 0
    examples = spec.with_name("polite-examples.jsonl")
    swapped = {'"pass": true': '"pass": false', '"pass": false': '"pass": true'}
    examples.write_text(
        re.sub(r'"pass": (true|false)', lambda m: swapped[m[0]], examples.read_text())
    )
    result = invoke(*args)
    assert result.exit_code == 1
    data = output(result)
    assert data["evaluator"] == "polite"
    assert f"fix its examples, or pass --out-dir {evaluated / 'eval-2'} to change" in data["hint"]


def test_train_evaluate_then_evaluators_reuses_the_session_s_answers(
    tmp_path: Path, spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, run = tmp_path / "ds", tmp_path / "run"
    assert invoke("dataset", "import", str(dataset), "--file", str(ROWS)).exit_code == 0
    trained = invoke(
        "train", "--model", "fake-model", "--dataset", str(dataset), "--run-dir", str(run),
        "--max-epochs", "1", "--evaluate", "--eval-no-publish", "--confirm",
    )  # fmt: skip
    assert trained.exit_code == 0, trained.stderr
    monkeypatch.setattr(FakeBackend, "open_session", no_session)
    preview = output(invoke("evaluate", str(run), "--evaluators", str(spec), "--no-publish"))
    assert preview["preview"]["generation_calls"] == 0
    result = rerun(preview["next_command"])
    assert result.exit_code == 0, result.stderr
    assert output(result)["headline"]["evaluator"] == "all"


@pytest.mark.usefixtures("all_judge_keys")
def test_read_only_allows_fake_evaluators_and_refuses_a_paid_one(
    evaluated: Path, spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "1")
    offline = invoke("evaluate", str(evaluated), "--evaluators", str(spec), "--no-publish",
                     "--confirm")  # fmt: skip
    assert offline.exit_code == 0, offline.stderr
    paid = spec.with_name("paid.toml")
    paid.write_text(
        spec.read_text()
        + '[[evaluator]]\nname = "tone"\ntype = "judge"\nmodel = "openai/gpt-5.5"\n'
        'rubric = "Pass when polite."\n'
    )

    @contextmanager
    def client(config: JudgeConfig, **kwargs: Any) -> Iterator[FakeJudge]:
        yield FakeJudge()

    monkeypatch.setattr(judges, "open_client", client)
    refused = invoke("evaluate", str(evaluated), "--evaluators", str(paid), "--no-publish",
                     "--confirm")  # fmt: skip
    assert refused.exit_code == 2
    assert "read_only" in refused.stderr


def test_a_check_that_raises_on_answers_is_named_and_not_retried(run_dir: Path, spec: Path) -> None:
    (spec.parent / "strict.py").write_text(
        "def same(case, candidate):\n    if candidate != case['reference']:\n"
        "        raise ValueError('not the reference')\n    return True\n"
    )
    strict = spec.with_name("strict.toml")
    strict.write_text('[[evaluator]]\nname = "same"\ntype = "check"\nfile = "strict.py"\n')
    result = invoke(
        "evaluate", str(run_dir), "--evaluators", str(strict), "--no-publish", "--confirm"
    )
    assert result.exit_code == 0, result.stderr
    data = output(result)
    answers = 2 * data["headline"]["cases"]
    assert data["status"] == "complete_with_errors"
    found = data["evaluator_errors"]["same"]
    assert (found["errors"], found["first_error"]["type"]) == (answers, "CheckError")
    assert data["hint"] == (
        f"the check same raised on {answers} answer(s): ValueError: not the reference; a retry "
        "runs the same frozen file, so fix the check in a new file under a new evaluator name, "
        f"or pass --out-dir {run_dir / 'eval-2'}"
    )
    assert data["next_command"] == f"pup finetune deploy {run_dir}"
    assert not [warning for warning in data["warnings"] if "retry" in warning]
    assert f"0 errors, {answers} evaluator error(s)" in result.stderr
    assert f"complete with 0 errors and {answers} evaluator errors" in result.stderr


class Outage:
    """A judge that cannot be reached: every call raises."""

    def complete(self, messages: Any, **kwargs: Any) -> Any:
        raise ExternalError("the judge is down", source="judge")

    def close(self) -> None:
        pass


def test_an_evaluator_judge_outage_is_retried(
    run_dir: Path, spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def down(judges: Any) -> Iterator[dict[str, Any]]:
        yield {"fake": Outage()}

    monkeypatch.setattr(runner_module, "open_evaluator_clients", down)
    spec.write_text('[[evaluator]]\nname = "tone"\ntype = "judge"\nrubric = "Pass when polite."\n')
    result = invoke(
        "evaluate", str(run_dir), "--evaluators", str(spec), "--no-publish", "--confirm"
    )
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["evaluator_errors"]["tone"]["first_error"]["message"] == "judge: the judge is down"
    assert data["hint"] is None
    assert data["next_command"] == (
        f"pup finetune evaluate {run_dir} --judge-model fake-judge --no-publish"
    )
