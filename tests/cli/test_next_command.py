# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Every preview's ``next_command`` runs as written, and following ``next_command`` from a
fresh dataset reaches a deployment with nothing refused.

"Runs" means it is neither refused as a usage error (a placeholder, a missing file, a bad
flag) nor as a state error (a directory that does not fit): it does what it says, or
previews the paid step that follows. Everything runs on the offline fakes in development
mode (the fake provider and judge, a fake pup); the conftest checks every result against
the rule itself (:func:`pup_finetune.next_steps.problems`).
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

from pup_finetune.cli import _common, app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.cli import triage as triage_cli
from pup_finetune.curation.triage import rubric_template
from pup_finetune.display import display_path
from pup_finetune.errors import ConfirmRequired, StateError, UsageError
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion
from pup_finetune.next_steps import PROGRAM, problems
from pup_finetune.runs import RunDir
from tests.fakes.datasets import FakeDatasetsApi
from tests.fakes.pup import FakePup

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def raised(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Every expected failure a command reports, in order."""
    seen: list[BaseException] = []
    original = _common.handle_exception

    def recording(exc: BaseException, runtime: Any, **where: str | None) -> int:
        seen.append(exc)
        return original(exc, runtime, **where)

    monkeypatch.setattr(_common, "handle_exception", recording)
    return seen


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    assert problems(data) == []
    return data


def follow(next_command: str | None, raised: list[BaseException]) -> Result:
    """Run ``next_command`` as written; it must not be refused."""
    assert next_command is not None
    words = shlex.split(next_command)
    assert tuple(words[: len(PROGRAM)]) == PROGRAM
    before = len(raised)
    result = invoke(*words[len(PROGRAM) :])
    refused = [exc for exc in raised[before:] if isinstance(exc, UsageError | StateError)]
    assert not refused, f"{next_command!r} was refused: {refused}"
    if result.exit_code == 2:  # only a preview of the paid step that follows
        assert len(raised) > before, result.stderr  # not a Click usage error
        assert isinstance(raised[-1], ConfirmRequired), result.stderr
    else:
        assert result.exit_code == 0, result.stderr
    return result


def previewed(args: Sequence[str]) -> dict[str, Any]:
    """The preview of ``args``: it exits 2 and ends its next_command in --confirm."""
    result = invoke(*args)
    assert result.exit_code == 2, result.stderr
    data = output(result)
    assert data["status"] == "confirmation_required"
    assert shlex.split(data["next_command"])[-1] == "--confirm"
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    assert invoke("dataset", "import", str(path), "--file", str(ROWS)).exit_code == 0
    return path


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
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    assert invoke(*train_args(dataset, path, "--confirm")).exit_code == 0
    return path


class AlwaysKeep:
    """A triage judge that keeps every row."""

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatCompletion:
        content = json.dumps({"keep": True, "reason": "fine"})
        return ChatCompletion(message={"role": "assistant", "content": content})


# --- each preview -------------------------------------------------------------------------


def test_dataset_push_preview(
    monkeypatch: pytest.MonkeyPatch, dataset: Path, raised: list[BaseException]
) -> None:
    fake = FakeDatasetsApi().install(FakePup())
    monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    data = previewed(["dataset", "push", str(dataset), "--name", "synthetic-replay"])
    assert output(follow(data["next_command"], raised))["pushed"] > 0


def test_dataset_triage_preview(
    monkeypatch: pytest.MonkeyPatch, dataset: Path, tmp_path: Path, raised: list[BaseException]
) -> None:
    @contextmanager
    def open_clients(judges: Sequence[str]) -> Iterator[Mapping[str, ChatCompleter]]:
        yield {"fireworks": AlwaysKeep()}

    monkeypatch.setattr(triage_cli, "open_clients", open_clients)
    rubric = tmp_path / "rubric.md"
    template = rubric_template()
    task = template.index("## Keep")
    rubric.write_text(
        template[: template.index("## Task")]
        + "## Task\n\nA synthetic agent.\n\n"
        + template[task:],
        encoding="utf-8",
    )
    data = previewed(["dataset", "triage", str(dataset), "--rubric", str(rubric)])
    assert output(follow(data["next_command"], raised))["status"] == "complete"


@pytest.mark.parametrize("extra", [(), ("--evaluate",), ("--batch-size", "2", "--seed", "3")])
def test_train_preview(
    dataset: Path, tmp_path: Path, raised: list[BaseException], extra: tuple[str, ...]
) -> None:
    data = previewed(train_args(dataset, tmp_path / "run", *extra))
    assert output(follow(data["next_command"], raised))["status"] == "completed"


def test_resume_preview(dataset: Path, tmp_path: Path, raised: list[BaseException]) -> None:
    run = tmp_path / "run"
    crash = ("--batch-size", "1", "--provider-option", "fail_at_step=9", "--confirm")
    assert invoke(*train_args(dataset, run, *crash)).exit_code == 1
    status = output(invoke("status", str(run)))
    data = output(follow(status["next_command"], raised))  # the resume preview
    assert output(follow(data["next_command"], raised))["status"] == "completed"


@pytest.mark.parametrize("extra", [(), ("--max-tokens", "512", "--seed", "4"), ("--no-publish",)])
def test_evaluate_preview(
    run_dir: Path, raised: list[BaseException], extra: tuple[str, ...]
) -> None:
    data = previewed(["evaluate", str(run_dir), *extra])
    assert output(follow(data["next_command"], raised))["status"] == "complete"


def test_deploy_and_undeploy_previews(run_dir: Path, raised: list[BaseException]) -> None:
    deploy = previewed(["deploy", str(run_dir)])
    deployed = output(follow(deploy["next_command"], raised))
    assert deployed["state"] == "READY"
    assert deployed["next_command"] is None
    undeploy = previewed(["undeploy", str(run_dir)])
    # undeploy_command is the preview form: the user approves the --confirm it prints.
    assert undeploy["next_command"] == f"{deployed['undeploy_command']} --confirm"
    assert output(follow(undeploy["next_command"], raised))["state"] == "DELETED"


# --- following next_command ---------------------------------------------------------------


def test_following_next_command_from_a_dataset_to_a_deployment(
    monkeypatch: pytest.MonkeyPatch, dataset: Path, tmp_path: Path, raised: list[BaseException]
) -> None:
    @contextmanager
    def fake_judge(config: Any) -> Iterator[ChatCompleter]:
        yield FakeJudge()

    monkeypatch.setattr(runner_module, "open_judge_client", fake_judge)
    shown = output(follow(f"pup finetune dataset show {dataset}", raised))
    sampled = output(follow(shown["next_command"], raised))
    assert sampled["next_command"] is None
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps({"id": row["id"], "keep": True}) + "\n" for row in sampled["rows"])
    )
    labeled = output(invoke("dataset", "label", str(dataset), "--file", str(labels)))
    assert labeled["next_command"] is None
    assert any(
        step["command"].split()[2:4] == ["dataset", "push"] for step in labeled["next_steps"]
    )

    preview = previewed(train_args(dataset, tmp_path / "run"))
    trained = output(follow(preview["next_command"], raised))
    evaluate = output(follow(trained["next_command"], raised))  # a free preview
    evaluated = output(follow(evaluate["next_command"], raised))
    assert evaluated["status"] == "complete"
    # Not published (nothing was pushed): deploying goes on; the steps say how to publish.
    assert evaluated["next_command"].startswith("pup finetune deploy ")
    assert evaluated["next_steps"][0]["command"].endswith("--name <NAME>")
    deploy = output(follow(evaluated["next_command"], raised))
    deployed = output(follow(deploy["next_command"], raised))
    assert deployed["next_command"] is None
    receipt = RunDir(tmp_path / "run").read_deployment() or {}
    assert deployed["usage"]["model"] == receipt["inference_model"]


def test_an_unreadable_push_receipt_is_a_step_not_silence(dataset: Path, tmp_path: Path) -> None:
    from pup_finetune.data.push import PUSH_FILE

    (dataset / PUSH_FILE).write_text("{not json", encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    row_id = json.loads(ROWS.read_text().splitlines()[0])["id"]
    labels.write_text(json.dumps({"id": row_id, "keep": True}) + "\n")
    labeled = output(invoke("dataset", "label", str(dataset), "--file", str(labels)))
    commands = [entry["command"] for entry in labeled["next_steps"]]
    assert f"pup finetune dataset show {display_path(dataset)}" in commands
    assert not any(" dataset push " in entry for entry in commands)
