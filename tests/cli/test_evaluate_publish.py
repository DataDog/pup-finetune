# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`evaluate` publishing experiments, end to end: a fake run, a pushed fixture dataset and a
fake pup serving fake datasets and experiments. Development mode; no network or paid call."""

from __future__ import annotations

import json
import os
import shlex
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.datadog.experiments import (
    CHILD_SPAN_NAME,
    EXPERIMENTS_FILE,
    PUBLISHED_FILE,
    ROOT_SPAN_NAME,
)
from pup_finetune.errors import ExternalError
from pup_finetune.evaluation import runner as runner_module
from pup_finetune.evaluation.fake_judge import FakeJudge
from pup_finetune.inference import ChatCompleter, ChatCompletion, JudgeConfig
from tests.fakes.datasets import FakeDatasetsApi
from tests.fakes.experiments import FakeExperimentsApi
from tests.fakes.pup import FakePup, PupCall

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FINETUNE_DEV", "1")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePup:
    fake = FakePup()
    monkeypatch.setattr(dataset_cli, "Pup", fake.client)
    monkeypatch.setattr(runner_module, "Pup", fake.client)
    return fake


@pytest.fixture
def datasets(fake: FakePup) -> FakeDatasetsApi:
    api = FakeDatasetsApi()
    api.install(fake)
    return api


@pytest.fixture
def experiments(fake: FakePup, datasets: FakeDatasetsApi) -> FakeExperimentsApi:
    api = FakeExperimentsApi()
    api.install(fake)  # after the datasets: their routes still answer
    return api


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def output(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ds"
    result = invoke("dataset", "import", str(path), "--file", str(ROWS))
    assert result.exit_code == 0, result.stderr
    return path


def push(dataset: Path) -> None:
    result = invoke("dataset", "push", str(dataset), "--name", "synthetic-replay", "--confirm")
    assert result.exit_code == 0, result.stderr


def train(dataset: Path, run_dir: Path, *extra: str) -> Result:
    args = ["train", "--provider", "fake", "--model", "fake-model", "--dataset", str(dataset)]
    return invoke(*args, "--run-dir", str(run_dir), "--max-epochs", "2", *extra, "--confirm")


@pytest.fixture
def run_dir(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "run"
    result = train(dataset, path)
    assert result.exit_code == 0, result.stderr
    return path


class FailsOnce(FakeJudge):
    """The offline judge, but its first call fails, as an outage after the client's retries."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatCompletion:
        if not self.failed:
            self.failed = True
            raise ExternalError("the judge endpoint is unreachable", source="judge")
        return super().complete(messages, **kwargs)


def use_judge(monkeypatch: pytest.MonkeyPatch, judge: ChatCompleter) -> None:
    @contextmanager
    def opened(config: JudgeConfig, **options: Any) -> Iterator[ChatCompleter]:
        yield judge

    monkeypatch.setattr(runner_module, "open_judge_client", opened)


def edit_receipt(dataset: Path, **changes: Any) -> dict[str, Any]:
    path = dataset / "push.json"
    receipt: dict[str, Any] = {**json.loads(path.read_text()), **changes}
    path.write_text(json.dumps(receipt))
    return receipt


def experiments_calls(fake: FakePup) -> int:
    return sum(
        1
        for call in fake.calls
        if call.args[:2] == ("llm-obs", "experiments") or "/experiments" in " ".join(call.args)
    )


def events_posts(fake: FakePup) -> int:
    return sum(
        1 for call in fake.calls if call.args[:1] == ("api",) and call.args[1].endswith("/events")
    )


def test_evaluate_publishes_two_experiments_and_a_rerun_sends_nothing_new(
    fake: FakePup,
    datasets: FakeDatasetsApi,
    experiments: FakeExperimentsApi,
    dataset: Path,
    run_dir: Path,
) -> None:
    push(dataset)
    receipt = json.loads((dataset / "push.json").read_text())
    preview = output(invoke("evaluate", str(run_dir)))
    publish = preview["preview"]["publish"]
    assert publish["status"] == "planned"
    assert publish["dataset_id"] == receipt["dataset_id"]
    run_id = json.loads((run_dir / "run.json").read_text())["run_id"]
    assert publish["experiments"] == [f"pup-finetune-{t}-{run_id}" for t in ("base", "tuned")]
    spans = preview["preview"]["trajectories"] + preview["preview"]["cases"]
    assert publish["spans_per_experiment"] == {"base": spans, "tuned": spans}
    assert "Datadog" in preview["preview"]["data_destination"]
    assert preview["preview"]["data_recipients"][-1] == (
        "test cases, generated answers and scores -> Datadog (experiments)"
    )
    assert preview["preview"]["warnings"] == []
    assert experiments.experiments == {}  # a preview publishes nothing

    result = invoke(*shlex.split(preview["next_command"])[2:])
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    publication = data["publication"]
    assert publication["status"] == "published"
    assert data["warnings"] == []
    assert len(experiments.experiments) == 2
    cases = data["headline"]["cases"]
    rows = data["headline"]["trajectories"]
    for target in ("base", "tuned"):
        entry = publication["experiments"][target]
        experiment = experiments.experiments[entry["experiment_id"]]
        assert experiment.name == entry["name"]
        assert experiment.attributes["dataset_id"] == receipt["dataset_id"]
        assert experiment.attributes["dataset_version"] == receipt["dataset_version"]
        assert entry["spans"] == entry["submitted"] == rows + cases
        assert entry["indexed"] is True
        roots = [span for span in experiment.spans if span["name"] == ROOT_SPAN_NAME]
        children = [span for span in experiment.spans if span["name"] == CHILD_SPAN_NAME]
        assert len(roots) == rows
        assert len(children) == cases
        record_ids = set(receipt["record_ids"].values())
        assert {root["dataset_record_id"] for root in roots} <= record_ids
        assert {child["parent_id"] for child in children} == {root["span_id"] for root in roots}
        labels = {metric["label"] for metric in experiment.metrics}
        assert {"judge_pass", "trajectory_agreement", "completed_actions"} <= labels
    out = run_dir / "eval"
    for name in (EXPERIMENTS_FILE, PUBLISHED_FILE):
        assert (out / name).stat().st_mode & 0o777 == 0o600

    posts = events_posts(fake)
    again = invoke("evaluate", str(run_dir), "--confirm")
    assert again.exit_code == 0, again.stderr
    republished = output(again)["publication"]
    assert events_posts(fake) == posts  # nothing new was sent
    assert len(experiments.experiments) == 2  # nor created
    assert all(entry["submitted"] == 0 for entry in republished["experiments"].values())
    assert republished["experiments"] == {
        target: {**entry, "submitted": 0, "already_published": entry["spans"]}
        for target, entry in publication["experiments"].items()
    }


def test_without_a_push_receipt_the_evaluation_is_not_published(
    fake: FakePup, experiments: FakeExperimentsApi, run_dir: Path
) -> None:
    preview = output(invoke("evaluate", str(run_dir)))["preview"]
    assert preview["publish"]["status"] == "skipped"
    (warning,) = preview["warnings"]
    assert "dataset push" in warning
    result = invoke("evaluate", str(run_dir), "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"  # the local results are complete
    assert data["publication"]["status"] == "skipped"
    assert any("dataset push" in warning for warning in data["warnings"])
    assert fake.calls == []
    assert (run_dir / "eval" / "summary.json").is_file()
    # Not published: no next command; the steps push the dataset (with a name only the user
    # can pick) and rerun the evaluation, which reuses every result, or deploy unpublished.
    assert data["next_command"] is None
    push_step, rerun_step, deploy_step, compare_step = data["next_steps"]
    assert deploy_step["command"] == f"pup finetune deploy {run_dir}"
    assert "--compare-model <MODEL>" in compare_step["command"]
    dataset = Path(str(json.loads((run_dir / "run.json").read_text())["dataset_dir"]))
    assert push_step["command"] == f"pup finetune dataset push {dataset} --name <NAME>"
    assert rerun_step["command"] == (
        f"pup finetune evaluate {run_dir} --judge-model fake-judge --confirm"
    )
    assert "no new model calls" in rerun_step["why"]
    push(dataset)
    generations = (run_dir / "eval" / "generations.jsonl").read_text()
    again = invoke(*shlex.split(rerun_step["command"])[2:])
    assert again.exit_code == 0, again.stderr
    published = output(again)
    assert published["publication"]["status"] == "published"
    assert published["tuned_checkpoint"] is None  # no session was opened
    assert (run_dir / "eval" / "generations.jsonl").read_text() == generations
    # Complete and published: deploying is next (a free preview).
    assert published["next_command"] == f"pup finetune deploy {run_dir}"
    assert [step["command"] for step in published["next_steps"]] == [
        f"pup finetune evaluate {run_dir} --judge-model fake-judge --compare-model <MODEL>"
    ]


def test_status_gives_the_push_first_steps_as_the_run_stands_now(
    fake: FakePup,
    experiments: FakeExperimentsApi,
    dataset: Path,
    run_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    evaluated = output(invoke("evaluate", str(run_dir), "--confirm"))
    assert evaluated["publication"]["status"] == "skipped"
    assert shlex.split(evaluated["next_steps"][0]["command"])[4] == "ds"

    def steps() -> list[list[str]]:
        status = output(invoke("status", str(run_dir)))
        assert status["next_command"] is None
        return [shlex.split(step["command"]) for step in status["next_steps"]]

    # From another directory, every path is the run's and the dataset's from there.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    pushing, again, deploy = steps()
    assert pushing == ["pup", "finetune", "dataset", "push", str(dataset), "--name", "<NAME>"]
    assert again[:4] == ["pup", "finetune", "evaluate", str(run_dir)]
    assert again[-1] == "--confirm"
    assert deploy == ["pup", "finetune", "deploy", str(run_dir)]
    # Deployed: no deploy step.
    deployed = invoke("deploy", str(run_dir), "--confirm")
    assert deployed.exit_code == 0, deployed.stderr
    assert steps() == [pushing, again]
    # Pushed later: the evaluation's own command is next, and its preview publishes it.
    push(dataset)
    status = output(invoke("status", str(run_dir)))
    assert status["next_steps"] == []
    preview = invoke(*shlex.split(status["next_command"])[2:])
    assert preview.exit_code == 2, preview.stderr
    assert output(preview)["preview"]["publish"]["status"] == "planned"


def test_status_offers_no_next_command_while_the_push_receipt_is_unreadable(
    fake: FakePup, experiments: FakeExperimentsApi, dataset: Path, run_dir: Path
) -> None:
    assert (
        output(invoke("evaluate", str(run_dir), "--confirm"))["publication"]["status"] == "skipped"
    )
    (dataset / "push.json").write_text("{not json")
    status = output(invoke("status", str(run_dir)))
    assert status["next_command"] is None
    (rerun,) = status["next_steps"]
    assert "once its dataset can be read" in rerun["why"]


def test_no_publish_skips_publishing_silently(
    fake: FakePup, experiments: FakeExperimentsApi, dataset: Path, run_dir: Path
) -> None:
    push(dataset)
    calls = len(fake.calls)
    preview = output(invoke("evaluate", str(run_dir), "--no-publish"))
    assert preview["preview"]["publish"] == {"status": "disabled"}
    assert preview["preview"]["warnings"] == []
    assert "--no-publish" in shlex.split(preview["next_command"])
    result = invoke("evaluate", str(run_dir), "--no-publish", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["publication"] == {"status": "disabled"}
    assert data["warnings"] == []
    assert len(fake.calls) == calls
    assert experiments.experiments == {}


def test_a_publishing_failure_exits_1_and_the_rerun_publishes_the_rest(
    fake: FakePup, experiments: FakeExperimentsApi, dataset: Path, run_dir: Path
) -> None:
    push(dataset)
    experiments.events_failures = {2}
    result = invoke("evaluate", str(run_dir), "--confirm")
    assert result.exit_code == 1, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["publication"]["status"] == "failed"
    assert any("publishing the evaluation failed" in warning for warning in data["warnings"])
    assert data["next_command"] is not None
    again = invoke(*shlex.split(data["next_command"])[2:], "--confirm")
    assert again.exit_code == 0, again.stderr
    assert output(again)["publication"]["status"] == "published"
    assert len(experiments.experiments) == 2
    for experiment in experiments.experiments.values():
        span_ids = [span["span_id"] for span in experiment.spans]
        assert len(span_ids) == len(set(span_ids))


def test_train_evaluate_publishes_too(
    fake: FakePup, experiments: FakeExperimentsApi, dataset: Path, tmp_path: Path
) -> None:
    push(dataset)
    result = train(dataset, tmp_path / "run", "--evaluate")
    assert result.exit_code == 0, result.stderr
    evaluation = output(result)["evaluation"]
    assert evaluation["publication"]["status"] == "published"
    assert set(evaluation["publication"]["experiments"]) == {"base", "tuned"}
    assert len(experiments.experiments) == 2
    args = ["train", "--provider", "fake", "--model", "fake-model", "--dataset", str(dataset)]
    args += ["--run-dir", str(tmp_path / "other"), "--evaluate", "--eval-no-publish"]
    preview = output(invoke(*args))
    assert preview["preview"]["evaluation"]["publish"] == {"status": "disabled"}
    assert "--eval-no-publish" in shlex.split(preview["next_command"])


@pytest.mark.parametrize("problem", ["in_flight", "missing_record"])
def test_an_unfinished_push_is_not_published(
    fake: FakePup,
    experiments: FakeExperimentsApi,
    dataset: Path,
    run_dir: Path,
    problem: str,
) -> None:
    push(dataset)
    if problem == "in_flight":
        edit_receipt(dataset, in_flight=["row-in-flight"])
        reason = "did not finish"
    else:
        test_row = DatasetDir(dataset).rows_for_split("test")[0].id
        record_ids = json.loads((dataset / "push.json").read_text())["record_ids"]
        del record_ids[test_row]
        edit_receipt(dataset, record_ids=record_ids)
        reason = "1 test row(s)"
    preview = output(invoke("evaluate", str(run_dir)))["preview"]
    assert preview["publish"]["status"] == "skipped"
    assert reason in preview["publish"]["reason"]
    assert "dataset push" in preview["publish"]["reason"]
    result = invoke("evaluate", str(run_dir), "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["publication"]["status"] == "skipped"
    assert reason in data["publication"]["reason"]
    assert experiments_calls(fake) == 0
    assert experiments.experiments == {}
    # The push to finish is known: the same dataset and project.
    assert data["next_command"] is None
    assert data["next_steps"][0]["command"] == (
        f"pup finetune dataset push {dataset.resolve()} --name synthetic-replay "
        "--project pup-finetune"
    )


@pytest.mark.parametrize("digests", ["changed", "unchanged", "none"])
def test_a_test_row_changed_after_its_push_is_not_published(
    fake: FakePup,
    experiments: FakeExperimentsApi,
    dataset: Path,
    run_dir: Path,
    digests: str,
) -> None:
    push(dataset)
    receipt = json.loads((dataset / "push.json").read_text())
    test_row = DatasetDir(dataset).rows_for_split("test")[0].id
    if digests == "changed":
        # As if the row was updated in place (e.g. reasoning filled in) after its push.
        edit_receipt(dataset, record_digests={**receipt["record_digests"], test_row: "0" * 64})
    elif digests == "none":
        edit_receipt(dataset, record_digests={})  # a receipt that predates digests
    publish = output(invoke("evaluate", str(run_dir)))["preview"]["publish"]
    if digests != "changed":
        assert publish["status"] == "planned"
        return
    assert publish["status"] == "skipped"
    assert "1 test row(s)" in publish["reason"]
    assert test_row in publish["reason"]
    assert "--no-publish" in publish["reason"]
    result = invoke("evaluate", str(run_dir), "--confirm")
    assert result.exit_code == 0, result.stderr
    assert output(result)["publication"]["status"] == "skipped"
    assert experiments_calls(fake) == 0
    (step, _, _, _) = output(result)["next_steps"]
    assert step["command"] == f"pup finetune dataset push {dataset.resolve()} --name <NAME>"
    assert "push.json aside" in step["why"]


def test_a_corrupt_push_receipt_says_how_to_go_on(
    fake: FakePup, experiments: FakeExperimentsApi, dataset: Path, run_dir: Path
) -> None:
    push(dataset)
    (dataset / "push.json").write_text("{not json")
    result = invoke("evaluate", str(run_dir))
    assert result.exit_code == 1, result.stderr
    assert "push.json" in result.stderr
    assert "--no-publish" in result.stderr
    preview = invoke("evaluate", str(run_dir), "--no-publish")
    assert preview.exit_code == 2, preview.stderr  # the free preview, unblocked


def test_retried_errors_update_only_the_local_results_of_a_published_evaluation(
    fake: FakePup,
    experiments: FakeExperimentsApi,
    dataset: Path,
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    push(dataset)
    use_judge(monkeypatch, FailsOnce())
    result = invoke("evaluate", str(run_dir), "--skip-calibration", "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete_with_errors"
    assert data["publication"]["status"] == "published"  # its error span too
    assert "retry" in data["hint"]
    assert "these errors included" in data["hint"]
    assert "--out-dir" in data["hint"]
    error_spans = [
        span
        for experiment in experiments.experiments.values()
        for span in experiment.spans
        if span["name"] == CHILD_SPAN_NAME and span["status"] == "error"
    ]
    assert len(error_spans) == 1
    again = invoke(*shlex.split(data["next_command"])[2:], "--confirm")
    assert again.exit_code == 0, again.stderr
    retried = output(again)
    assert retried["status"] == "complete"  # the local results have every verdict
    assert retried["publication"]["status"] == "published"
    assert all(entry["submitted"] == 0 for entry in retried["publication"]["experiments"].values())
    (warning,) = retried["warnings"]
    assert "published with an earlier result" in warning


def test_a_cancel_while_publishing_is_continued_by_the_rerun(
    fake: FakePup, experiments: FakeExperimentsApi, dataset: Path, run_dir: Path
) -> None:
    push(dataset)
    serve = fake.handler("api")
    assert serve is not None

    def interrupting(call: PupCall) -> Any:
        answer = serve(call)
        if experiments.events_calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)  # the first Ctrl-C, as the first batch lands
        return answer

    fake.on("api", handler=interrupting)
    result = invoke("evaluate", str(run_dir), "--confirm")
    assert result.exit_code == 0, result.stderr
    data = output(result)
    assert data["status"] == "complete"
    assert data["publication"]["status"] == "cancelled"
    assert any("cancelled" in warning for warning in data["warnings"])
    assert data["next_command"] is not None
    assert experiments.events_calls == 1
    fake.on("api", handler=serve)
    again = invoke(*shlex.split(data["next_command"])[2:], "--confirm")
    assert again.exit_code == 0, again.stderr
    assert output(again)["publication"]["status"] == "published"
    for experiment in experiments.experiments.values():
        span_ids = [span["span_id"] for span in experiment.spans]
        assert len(span_ids) == len(set(span_ids))


def test_attaching_a_compare_model_publishes_one_more_experiment_in_the_same_comparison(
    fake: FakePup,
    datasets: FakeDatasetsApi,
    experiments: FakeExperimentsApi,
    dataset: Path,
    run_dir: Path,
) -> None:
    fake.on("auth", "status", json={"authenticated": True, "site": "datadoghq.com"})
    push(dataset)
    first = invoke("evaluate", str(run_dir), "--confirm")
    assert first.exit_code == 0, first.stderr
    published = output(first)["publication"]["experiments"]
    spans = {item.id: list(item.spans) for item in experiments.experiments.values()}
    posts = events_posts(fake)

    preview = output(invoke("evaluate", str(run_dir), "--compare-model", "fake-compare"))
    run_id = json.loads((run_dir / "run.json").read_text())["run_id"]
    assert preview["preview"]["publish"]["experiments"] == [
        f"pup-finetune-{target}-{run_id}" for target in ("base", "tuned", "compare")
    ]
    assert "publish three experiments" in preview["preview"]["action"]
    result = invoke(*shlex.split(preview["next_command"])[2:])
    assert result.exit_code == 0, result.stderr
    publication = output(result)["publication"]
    assert publication["status"] == "published"
    assert len(experiments.experiments) == 3
    for target in ("base", "tuned"):
        entry = publication["experiments"][target]
        assert entry["experiment_id"] == published[target]["experiment_id"]
        assert entry["submitted"] == 0
        assert (
            experiments.experiments[entry["experiment_id"]].spans == spans[entry["experiment_id"]]
        )
    compare = publication["experiments"]["compare"]
    created = experiments.experiments[compare["experiment_id"]]
    assert created.name == f"pup-finetune-compare-{run_id}"
    config = created.attributes["config"]
    assert config["compare_model"] == "fake-compare"
    assert config["provider"] == "fake"
    assert config["max_tokens"] == 16000
    assert "base_model" not in config and "checkpoint" not in config
    assert "the compare model (fake-compare)" in created.attributes["description"]
    assert compare["submitted"] == compare["spans"] > 0
    assert events_posts(fake) > posts
    ids = [publication["experiments"][t]["experiment_id"] for t in ("base", "tuned", "compare")]
    assert f"experimentIds={','.join(ids)}&" in publication["comparison_url"]
