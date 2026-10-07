# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Publishing a replay evaluation as experiments, over a fake pup and fake experiments API.

Every case, record and id here is synthetic; time comes from an injected clock, so nothing
sleeps.
"""

from __future__ import annotations

import dataclasses
import json
import stat
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.push import SOURCE_TAG, PushState
from pup_finetune.datadog.experiments import (
    CHILD_SPAN_NAME,
    EXPERIMENTS_FILE,
    PUBLISHED_FILE,
    ROOT_SPAN_NAME,
    ExperimentsPublisher,
    PublishSettings,
)
from pup_finetune.errors import ReadOnlyError
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.replay import METRICS_VERSION, TARGETS
from pup_finetune.runtime import Runtime
from tests.fakes.experiments import (
    CreateAnswer,
    EventsListAnswer,
    FakeExperimentsApi,
    ProcessDied,
)
from tests.fakes.pup import FakePup

START = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
RUN_ID = "run-synthetic-01"
SAMPLED_AT = "2029-12-30T11:00:00+00:00"
"""Over a day before :data:`START`: a metric stamped with it would be refused."""
DAY_MS = 24 * 3600 * 1000


def ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


class Clock:
    """A clock that only moves when :meth:`sleep` is called."""

    def __init__(self) -> None:
        self.now = START
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


def case(row_id: str, index: int) -> ReplayCase:
    return ReplayCase(
        id=f"{row_id}:{index}",
        row_id=row_id,
        trace_id=f"trace-{row_id}",
        message_index=index,
        prefix=({"role": "user", "content": f"Question {row_id}"},),
        tools=({"type": "function", "function": {"name": "lookup", "parameters": {}}},),
        reference={"role": "assistant", "content": f"Answer {index}"},
    )


def record(row_id: str) -> dict[str, Any]:
    return {
        "input": {"messages": [{"role": "user", "content": f"Question {row_id}"}], "tools": []},
        "expected_output": {"role": "assistant", "content": "Final answer"},
        "metadata": {"row_id": row_id, "span_id": f"span-{row_id}", "trace_id": f"trace-{row_id}"},
    }


class Evaluation:
    """Synthetic cases, generations and results of both targets: tuned passes, base fails."""

    def __init__(
        self, rows: int = 2, cases_per_row: int = 2, *, row_ids: Sequence[str] | None = None
    ) -> None:
        self.row_ids = list(row_ids) if row_ids else [f"row-{n:03d}" for n in range(rows)]
        self.cases = [case(row, 2 * i + 1) for row in self.row_ids for i in range(cases_per_row)]
        self.generations: dict[tuple[str, str], dict[str, Any]] = {}
        self.results: dict[tuple[str, str], dict[str, Any]] = {}
        for target in TARGETS:
            for item in self.cases:
                self.generations[(target, item.id)] = {
                    "target": target,
                    "case_id": item.id,
                    "message": {"role": "assistant", "content": f"{target} answer"},
                    "format_valid": True,
                    "latency_s": 1.5,
                    "sampled_at": SAMPLED_AT,
                }
                self.results[(target, item.id)] = {
                    "target": target,
                    "case_id": item.id,
                    "row_id": item.row_id,
                    "status": "ok",
                    "error": None,
                    "metrics": {"tool_name_match": target == "tuned", "format_valid": True},
                    "judgment": {"passed": target == "tuned", "reason": f"{target} reason"},
                    "judged_at": SAMPLED_AT,
                }
        self.records = {row: record(row) for row in self.row_ids}

    def fail(self, target: str, case_id: str) -> None:
        self.generations.pop((target, case_id), None)
        self.results[(target, case_id)] = {
            **self.results[(target, case_id)],
            "status": "error",
            "error": {"stage": "generation", "type": "TransientError", "message": "overloaded"},
            "metrics": None,
            "judgment": None,
        }

    def push(self) -> PushState:
        return PushState(
            project="pup-finetune-dev",
            project_id="synthetic-project-0001",
            dataset_name="synthetic",
            settings_hash="synthetic-hash",
            push_id="synthetic-push",
            dataset_id="synthetic-dataset-0001",
            dataset_version=3,
            record_ids={row: f"record-{row}" for row in self.row_ids},
        )

    def settings(self, *, eval_key: str = "synthetic-eval-key") -> PublishSettings:
        return PublishSettings(
            eval_key=eval_key,
            run_id=RUN_ID,
            provider="fake",
            base_model="fake/fake-model",
            tuned_checkpoint="fake://run/ckpt-2",
            judge_model="fake-judge",
            max_tokens=256,
            push=self.push(),
            records=self.records,
        )


@pytest.fixture
def api() -> FakeExperimentsApi:
    return FakeExperimentsApi()


@pytest.fixture
def fake(api: FakeExperimentsApi) -> FakePup:
    return api.install(FakePup())


@pytest.fixture
def clock() -> Clock:
    return Clock()


def publisher(
    fake: FakePup,
    evaluation: Evaluation,
    out_dir: Path,
    clock: Clock,
    *,
    runtime: Runtime | None = None,
    settings: PublishSettings | None = None,
    **options: Any,
) -> ExperimentsPublisher:
    out_dir.mkdir(mode=0o700, exist_ok=True)
    options.setdefault("sleep", clock.sleep)
    return ExperimentsPublisher(
        fake.client(),
        settings or evaluation.settings(),
        out_dir=out_dir,
        runtime=runtime or Runtime(),
        clock=clock,
        **options,
    )


def publish(
    fake: FakePup, evaluation: Evaluation, out_dir: Path, clock: Clock, **options: Any
) -> dict[str, Any]:
    return publisher(fake, evaluation, out_dir, clock, **options).publish(
        cases=evaluation.cases,
        generations=evaluation.generations,
        results=evaluation.results,
        summary={},
    )


def spans_named(spans: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [span for span in spans if span["name"] == name]


def metrics_of(metrics: Sequence[Mapping[str, Any]], span_id: str) -> dict[str, Mapping[str, Any]]:
    return {metric["label"]: metric for metric in metrics if metric["span_id"] == span_id}


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- experiments --------------------------------------------------------------------------


def test_one_experiment_per_target_is_created_on_the_pushed_dataset_version(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    assert report["status"] == "published"
    assert report["dataset_id"] == "synthetic-dataset-0001"
    assert report["dataset_version"] == 3
    assert len(api.create_bodies) == 2
    for target, body in zip(TARGETS, api.create_bodies, strict=True):
        attributes = body["data"]["attributes"]
        assert attributes["name"] == f"pup-finetune-{target}-{RUN_ID}"
        assert attributes["dataset_id"] == "synthetic-dataset-0001"
        assert attributes["project_id"] == "synthetic-project-0001"
        assert attributes["dataset_version"] == 3
        assert attributes["ensure_unique"] is True
        assert attributes["run_count"] == 1
        assert attributes["metadata"] == {
            "tags": [SOURCE_TAG, f"target:{target}", f"run_id:{RUN_ID}"]
        }
        assert attributes["config"] == {
            "provider": "fake",
            "base_model": "fake/fake-model",
            "target": target,
            "run_id": RUN_ID,
            "checkpoint": "fake://run/ckpt-2" if target == "tuned" else None,
            "judge_model": "fake-judge",
            "max_tokens": 256,
            "metrics_version": METRICS_VERSION,
        }
        assert f"synthetic-eval-key-{target}" in attributes["description"]
        experiment = report["experiments"][target]
        assert api.experiments[experiment["experiment_id"]].name == experiment["name"]
    # Bodies went through private files, removed afterwards.
    assert api.file_modes == [(0o600, 0o700)] * 2
    assert not any(path.exists() for path in api.file_paths)


def test_experiments_are_recorded_and_reused_on_resume(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    out = tmp_path / "eval"
    first = publish(fake, evaluation, out, clock)
    stored = json.loads((out / EXPERIMENTS_FILE).read_text())
    assert mode(out / EXPERIMENTS_FILE) == 0o600
    for target in TARGETS:
        assert stored[target]["experiment_id"] == first["experiments"][target]["experiment_id"]
        assert stored[target]["name"] == first["experiments"][target]["name"]
    again = publish(fake, evaluation, out, clock)
    assert len(api.create_bodies) == 2  # nothing created again
    assert len(api.experiments) == 2
    for target in TARGETS:
        assert again["experiments"][target]["experiment_id"] == stored[target]["experiment_id"]
        assert again["experiments"][target]["submitted"] == 0


def test_the_api_may_rename_an_experiment_to_keep_it_unique(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    publish(fake, evaluation, tmp_path / "first", clock)
    report = publish(fake, evaluation, tmp_path / "second", clock)
    assert report["experiments"]["base"]["name"] == f"pup-finetune-base-{RUN_ID}-1"


@pytest.mark.parametrize("answer", ["fail_after_create", "unreadable", "no_id"])
def test_a_lost_create_response_adopts_the_experiment_it_created(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path, answer: CreateAnswer
) -> None:
    evaluation = Evaluation()
    # Another evaluation's experiments on the same dataset must not be adopted.
    other = publisher(
        fake, evaluation, tmp_path / "other", clock, settings=evaluation.settings(eval_key="other")
    ).publish(
        cases=evaluation.cases,
        generations=evaluation.generations,
        results=evaluation.results,
        summary={},
    )
    before = set(api.experiments)
    api.create_answer = answer
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    assert report["status"] == "published"
    assert len(api.experiments) == 4
    ids = {entry["experiment_id"] for entry in report["experiments"].values()}
    assert ids == set(api.experiments) - before
    assert not ids & {entry["experiment_id"] for entry in other["experiments"].values()}
    stored = json.loads((tmp_path / "eval" / EXPERIMENTS_FILE).read_text())
    assert {entry["experiment_id"] for entry in stored.values()} == ids


def test_a_crash_between_create_and_save_adopts_the_experiment_on_rerun(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    out = tmp_path / "eval"
    api.create_answer = "crash_after_create"
    with pytest.raises(ProcessDied):
        publish(fake, evaluation, out, clock)
    (orphan,) = api.experiments.values()
    pending = json.loads((out / EXPERIMENTS_FILE).read_text())["base"]
    assert pending["dataset_id"] == "synthetic-dataset-0001"
    assert "experiment_id" not in pending
    nonce = pending["pending_nonce"]
    assert f"synthetic-eval-key-base-{nonce}" in orphan.attributes["description"]
    api.create_answer = "ok"
    report = publish(fake, evaluation, out, clock)
    assert report["status"] == "published"
    assert report["experiments"]["base"]["experiment_id"] == orphan.id
    assert len(api.experiments) == 2  # the base experiment was adopted, not created again
    assert [body["data"]["attributes"]["config"]["target"] for body in api.create_bodies] == [
        "base",
        "tuned",
    ]
    stored = json.loads((out / EXPERIMENTS_FILE).read_text())
    assert stored["base"]["experiment_id"] == orphan.id
    assert "pending_nonce" not in stored["base"]
    assert report["experiments"]["base"]["submitted"] == 6


def test_a_pending_create_not_found_is_made_again_with_its_nonce(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    out = tmp_path / "eval"
    api.create_answer = "fail"
    publish(fake, evaluation, out, clock)  # the create never went through
    nonce = json.loads((out / EXPERIMENTS_FILE).read_text())["base"]["pending_nonce"]
    api.create_answer = "ok"
    report = publish(fake, evaluation, out, clock)
    assert report["status"] == "published"
    assert len(api.experiments) == 2
    base = api.experiments[report["experiments"]["base"]["experiment_id"]]
    assert f"synthetic-eval-key-base-{nonce}" in base.attributes["description"]


def test_another_out_dir_never_adopts_a_pending_create_of_the_same_evaluation(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    api.create_answer = "crash_after_create"
    with pytest.raises(ProcessDied):
        publish(fake, evaluation, tmp_path / "crashed", clock)
    (orphan,) = api.experiments.values()
    api.create_answer = "ok"
    fresh = publish(fake, evaluation, tmp_path / "fresh", clock)
    assert fresh["status"] == "published"
    assert orphan.id not in {entry["experiment_id"] for entry in fresh["experiments"].values()}
    assert len(api.experiments) == 3
    # The crashed directory still adopts its own.
    resumed = publish(fake, evaluation, tmp_path / "crashed", clock)
    assert resumed["experiments"]["base"]["experiment_id"] == orphan.id
    assert len(api.experiments) == 4  # only its tuned experiment was new


def test_a_pending_create_on_another_dataset_is_replaced_with_a_warning(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    out = tmp_path / "eval"
    api.create_answer = "crash_after_create"
    with pytest.raises(ProcessDied):
        publish(fake, evaluation, out, clock)
    (orphan,) = api.experiments.values()
    api.create_answer = "ok"
    repushed = evaluation.settings()
    repushed = dataclasses.replace(
        repushed, push=dataclasses.replace(repushed.push, dataset_id="synthetic-dataset-0002")
    )
    report = publisher(fake, evaluation, out, clock, settings=repushed).publish(
        cases=evaluation.cases,
        generations=evaluation.generations,
        results=evaluation.results,
        summary={},
    )
    assert report["status"] == "published"
    assert report["experiments"]["base"]["experiment_id"] != orphan.id
    (warning,) = report["warnings"]
    assert "base experiment create on the dataset 'synthetic-dataset-0001'" in warning
    assert "pushed again" in warning


def test_a_create_not_found_afterwards_says_it_may_have_gone_through(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.create_answer = "no_id"
    fake.on("llm-obs", "experiments", "list", json={"data": []})  # e.g. on a later page
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert report["status"] == "failed"
    assert "returned no id" in report["error"]["message"]
    assert "may have gone through" in report["hint"]
    assert f"pup-finetune-base-{RUN_ID}" in report["hint"]


def test_a_failed_create_is_reported_not_raised(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.create_answer = "fail"
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert report["status"] == "failed"
    assert "500" in report["error"]["message"]
    assert "may have gone through" in report["hint"]
    assert report["error"]["hint"] == report["hint"]
    # Only the pending create is recorded, so a rerun looks for it before creating.
    stored = json.loads((tmp_path / "eval" / EXPERIMENTS_FILE).read_text())
    assert set(stored) == {"base"}
    assert set(stored["base"]) == {"pending_nonce", "dataset_id", "requested_at"}


def test_a_failure_keeps_the_hint_of_its_error(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.events_failures = {1}
    api.events_failure_stderr = "Error: 401 Unauthorized"
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert report["status"] == "failed"
    assert "pup auth login" in report["error"]["hint"]
    assert report["hint"] == report["error"]["hint"]


def test_a_failure_without_a_hint_says_to_run_again(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.events_failures = {1}
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert report["status"] == "failed"
    assert report["error"]["hint"] is None
    assert "run the same evaluate command again" in report["hint"]


def test_an_experiment_of_another_dataset_is_replaced_with_a_warning(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    out = tmp_path / "eval"
    first = publish(fake, evaluation, out, clock)
    repushed = evaluation.settings()
    repushed = dataclasses.replace(
        repushed, push=dataclasses.replace(repushed.push, dataset_id="synthetic-dataset-0002")
    )
    report = publisher(fake, evaluation, out, clock, settings=repushed).publish(
        cases=evaluation.cases,
        generations=evaluation.generations,
        results=evaluation.results,
        summary={},
    )
    assert report["status"] == "published"
    assert len(api.experiments) == 4
    stored = json.loads((out / EXPERIMENTS_FILE).read_text())
    assert len(report["warnings"]) == 2
    for target, warning in zip(TARGETS, report["warnings"], strict=True):
        old = first["experiments"][target]["experiment_id"]
        new = report["experiments"][target]["experiment_id"]
        assert new != old
        assert stored[target]["experiment_id"] == new
        assert stored[target]["dataset_id"] == "synthetic-dataset-0002"
        assert api.experiments[new].attributes["dataset_id"] == "synthetic-dataset-0002"
        assert report["experiments"][target]["submitted"] == 6  # a new experiment gets all
        assert f"{target} experiment {old!r}" in warning
        assert "pushed again" in warning


def test_an_unreadable_experiment_entry_is_replaced_with_a_warning(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    out = tmp_path / "eval"
    out.mkdir(mode=0o700)
    (out / EXPERIMENTS_FILE).write_text(json.dumps({"base": "garbage"}))
    report = publish(fake, Evaluation(), out, clock)
    assert report["status"] == "published"
    (warning,) = report["warnings"]
    assert "base experiment that is not reused" in warning
    assert "cannot be read" in warning
    stored = json.loads((out / EXPERIMENTS_FILE).read_text())
    assert stored["base"]["experiment_id"] == report["experiments"]["base"]["experiment_id"]


def test_publishing_is_a_remote_write(fake: FakePup, clock: Clock, tmp_path: Path) -> None:
    with pytest.raises(ReadOnlyError, match="PUP_READ_ONLY"):
        publish(fake, Evaluation(), tmp_path / "eval", clock, runtime=Runtime(read_only=True))
    assert fake.calls == []


# --- spans and metrics --------------------------------------------------------------------


def test_roots_and_children_are_linked_to_the_records(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    experiment = api.experiments[report["experiments"]["tuned"]["experiment_id"]]
    roots = spans_named(experiment.spans, ROOT_SPAN_NAME)
    children = spans_named(experiment.spans, CHILD_SPAN_NAME)
    assert len(roots) == 2
    assert len(children) == 4
    assert report["experiments"]["tuned"]["spans"] == 6
    for root in roots:
        row_id = root["meta"]["metadata"]["row_id"]
        assert root["parent_id"] == "undefined"
        assert root["dataset_record_id"] == f"record-{row_id}"
        assert root["status"] == "ok"
        assert root["meta"]["span"] == {"kind": "experiment"}
        assert root["meta"]["input"] == evaluation.records[row_id]["input"]
        assert root["meta"]["expected_output"] == evaluation.records[row_id]["expected_output"]
        assert root["meta"]["metadata"]["run_id"] == RUN_ID
        output = root["meta"]["output"]
        assert output["completed_actions"] == output["total_actions"] == 2
        assert output["status"] == "complete"
        assert [step["judge_pass"] for step in output["steps"]] == [True, True]
        assert output["steps"][0]["tool_name_match"] is True
        assert "dataset_record_id:record-" + row_id in root["tags"]
        assert f"experiment_id:{experiment.id}" in root["tags"]
        assert int(root["span_id"]) < 2**63
        assert len(root["trace_id"]) == 32
        int(root["trace_id"], 16)
        mine = [child for child in children if child["parent_id"] == root["span_id"]]
        assert len(mine) == 2
        for child in mine:
            assert child["trace_id"] == root["trace_id"]
            assert child["meta"]["parent_experiment_span_id"] == root["span_id"]
            assert child["meta"]["span"] == {"kind": "llm"}
            assert child["meta"]["output"] == {"role": "assistant", "content": "tuned answer"}
            assert child["meta"]["input"]["messages"] == [
                {"role": "user", "content": f"Question {row_id}"}
            ]
            assert child["meta"]["input"]["tools"][0]["function"]["name"] == "lookup"
            metadata = child["meta"]["metadata"]
            assert metadata["source_trace_id"] == f"trace-{row_id}"
            assert metadata["source_span_id"] == f"span-{row_id}"
            assert child["meta"]["expected_output"]["content"].startswith("Answer")
            assert child["status"] == "ok"
            assert child["duration"] == 1_500_000_000
            end = int(datetime.fromisoformat(SAMPLED_AT).timestamp()) * 10**9
            assert child["start_ns"] == end - child["duration"]
        assert root["start_ns"] == min(child["start_ns"] for child in mine)
    # The base experiment's spans have other ids: one evaluation, two experiments.
    base = api.experiments[report["experiments"]["base"]["experiment_id"]]
    assert not {span["span_id"] for span in base.spans} & {span["span_id"] for span in roots}


def test_metrics_are_stamped_at_submission_and_carry_the_judge_reason(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation()
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    now_ms = ms(clock())
    assert now_ms - ms(datetime.fromisoformat(SAMPLED_AT)) > DAY_MS
    for target in TARGETS:
        experiment = api.experiments[report["experiments"][target]["experiment_id"]]
        assert experiment.metrics
        assert {metric["timestamp_ms"] for metric in experiment.metrics} == {now_ms}
        assert all(metric["experiment_id"] == experiment.id for metric in experiment.metrics)
        for child in spans_named(experiment.spans, CHILD_SPAN_NAME):
            metrics = metrics_of(experiment.metrics, child["span_id"])
            assert set(metrics) == {"tool_name_match", "format_valid", "judge_pass"}
            judge = metrics["judge_pass"]
            assert judge["metric_type"] == "boolean"
            assert judge["boolean_value"] is (target == "tuned")
            assert judge["reasoning"] == f"{target} reason"
            assert judge["trace_id"] == child["trace_id"]
            assert metrics["tool_name_match"]["boolean_value"] is (target == "tuned")
        for root in spans_named(experiment.spans, ROOT_SPAN_NAME):
            metrics = metrics_of(experiment.metrics, root["span_id"])
            assert metrics["trajectory_agreement"]["metric_type"] == "score"
            expected = 1.0 if target == "tuned" else 0.0
            assert metrics["trajectory_agreement"]["score_value"] == expected
            assert metrics["completed_actions"]["score_value"] == 2
        assert all(request["tags"][0] == SOURCE_TAG for request in experiment.requests)
    assert set(api.content_types) == {"application/vnd.api+json"}


def test_a_metric_that_does_not_apply_is_not_sent(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=1, cases_per_row=1)
    for key in evaluation.results:
        evaluation.results[key]["metrics"] = {"tool_name_match": None, "format_valid": True}
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    experiment = api.experiments[report["experiments"]["tuned"]["experiment_id"]]
    (child,) = spans_named(experiment.spans, CHILD_SPAN_NAME)
    assert set(metrics_of(experiment.metrics, child["span_id"])) == {"format_valid", "judge_pass"}


def test_an_error_case_is_an_error_span_without_metrics(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=2, cases_per_row=2)
    failed = evaluation.cases[0]
    evaluation.fail("tuned", failed.id)
    for other in evaluation.cases[2:]:
        evaluation.fail("tuned", other.id)  # every case of the second row
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    experiment = api.experiments[report["experiments"]["tuned"]["experiment_id"]]
    children = {
        child["meta"]["metadata"]["case_id"]: child
        for child in spans_named(experiment.spans, CHILD_SPAN_NAME)
    }
    error = children[failed.id]
    assert error["status"] == "error"
    assert error["meta"]["error"] == {"message": "overloaded", "type": "TransientError"}
    assert error["meta"]["output"] is None
    assert metrics_of(experiment.metrics, error["span_id"]) == {}
    roots = {
        root["meta"]["metadata"]["row_id"]: root
        for root in spans_named(experiment.spans, ROOT_SPAN_NAME)
    }
    partial = roots[failed.row_id]
    assert partial["status"] == "ok"
    assert partial["meta"]["output"]["status"] == "partial"
    assert partial["meta"]["output"]["completed_actions"] == 1
    assert (
        metrics_of(experiment.metrics, partial["span_id"])["completed_actions"]["score_value"] == 1
    )
    nothing = roots[evaluation.cases[2].row_id]
    assert nothing["status"] == "error"
    assert metrics_of(experiment.metrics, nothing["span_id"]) == {}


def test_span_ids_are_the_same_in_every_run_of_one_evaluation(clock: Clock, tmp_path: Path) -> None:
    ids = []
    for name in ("one", "two"):
        api = FakeExperimentsApi()
        fake = api.install(FakePup())
        report = publish(fake, Evaluation(), tmp_path / name, clock)
        experiment = api.experiments[report["experiments"]["tuned"]["experiment_id"]]
        ids.append([(span["span_id"], span["trace_id"]) for span in experiment.spans])
    assert ids[0] == ids[1]


def test_a_row_id_that_looks_like_a_case_id_gets_ids_of_its_own(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    # The case ``row-a:1`` of the row ``row-a`` and the row ``row-a:1`` have the same name.
    evaluation = Evaluation(cases_per_row=1, row_ids=["row-a", "row-a:1"])
    out = tmp_path / "eval"
    api.events_failures = {2}
    first = publish(fake, evaluation, out, clock, batch_size=2)  # one row a request
    assert first["status"] == "failed"
    second = publish(fake, evaluation, out, clock, batch_size=2)
    assert second["status"] == "published"
    assert second["experiments"]["base"]["already_published"] == 2
    assert second["experiments"]["base"]["submitted"] == 2
    for experiment in api.experiments.values():
        assert len(experiment.spans) == 4
        assert len({span["span_id"] for span in experiment.spans}) == 4
        roots = spans_named(experiment.spans, ROOT_SPAN_NAME)
        assert {root["meta"]["metadata"]["row_id"] for root in roots} == {"row-a", "row-a:1"}
        assert len({root["trace_id"] for root in roots}) == 2


# --- batching and resuming ----------------------------------------------------------------


def test_at_most_100_spans_go_in_one_request_and_rows_stay_whole(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=40, cases_per_row=2)  # 120 spans per target
    report = publish(fake, evaluation, tmp_path / "eval", clock)
    experiment = api.experiments[report["experiments"]["tuned"]["experiment_id"]]
    sizes = [len(request["spans"]) for request in experiment.requests]
    assert sizes == [99, 21]
    for request in experiment.requests:
        span_ids = {span["span_id"] for span in request["spans"]}
        assert {metric["span_id"] for metric in request["metrics"]} <= span_ids
        roots = {span["span_id"] for span in request["spans"] if span["name"] == ROOT_SPAN_NAME}
        assert {s["parent_id"] for s in request["spans"] if s["name"] == CHILD_SPAN_NAME} <= roots
    assert len({span["span_id"] for span in experiment.spans}) == 120


def test_a_row_bigger_than_a_request_is_split(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=1, cases_per_row=5)
    report = publish(fake, evaluation, tmp_path / "eval", clock, batch_size=4)
    experiment = api.experiments[report["experiments"]["tuned"]["experiment_id"]]
    assert [len(request["spans"]) for request in experiment.requests] == [4, 2]
    assert experiment.requests[0]["spans"][0]["name"] == ROOT_SPAN_NAME


def test_a_failure_mid_way_resumes_without_sending_a_span_twice(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=4, cases_per_row=2)  # 12 spans per target
    out = tmp_path / "eval"
    api.events_failures = {2}
    first = publish(fake, evaluation, out, clock, batch_size=6)
    assert first["status"] == "failed"
    assert "502" in first["error"]["message"]
    published = json.loads((out / PUBLISHED_FILE).read_text())
    assert mode(out / PUBLISHED_FILE) == 0o600
    assert sum(len(spans) for spans in published.values()) == 6  # the batch that went
    first_ms = ms(clock())
    clock.now += timedelta(days=2)  # the rerun is days later: its metrics carry its time
    second = publish(fake, evaluation, out, clock, batch_size=6)
    assert second["status"] == "published"
    assert second["experiments"]["base"]["submitted"] == 6
    assert second["experiments"]["base"]["already_published"] == 6
    assert second["experiments"]["tuned"]["submitted"] == 12
    for experiment in api.experiments.values():
        span_ids = [span["span_id"] for span in experiment.spans]
        assert len(span_ids) == len(set(span_ids)) == 12
    base = api.experiments[second["experiments"]["base"]["experiment_id"]]
    (sent, resent) = base.requests
    assert {metric["timestamp_ms"] for metric in sent["metrics"]} == {first_ms}
    assert {metric["timestamp_ms"] for metric in resent["metrics"]} == {first_ms + 2 * DAY_MS}
    third = publish(fake, evaluation, out, clock, batch_size=6)
    assert all(entry["submitted"] == 0 for entry in third["experiments"].values())
    assert third["warnings"] == []


def test_a_lost_events_response_resends_its_batch_with_the_same_span_ids(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=4, cases_per_row=2)  # 12 spans per target
    out = tmp_path / "eval"
    api.events_lost_responses = {2}  # stored, but its response never arrived
    first = publish(fake, evaluation, out, clock, batch_size=6)
    assert first["status"] == "failed"
    published = json.loads((out / PUBLISHED_FILE).read_text())
    assert sum(len(spans) for spans in published.values()) == 6  # the lost batch is unknown
    base = api.experiments[
        json.loads((out / EXPERIMENTS_FILE).read_text())["base"]["experiment_id"]
    ]
    lost = [span["span_id"] for span in base.requests[1]["spans"]]
    second = publish(fake, evaluation, out, clock, batch_size=6)
    assert second["status"] == "published"
    assert second["experiments"]["base"]["submitted"] == 6
    resent = [span["span_id"] for span in base.requests[2]["spans"]]
    assert resent == lost  # the same batch, with the same deterministic ids
    # This fake keeps both copies; the API is relied on to identify spans by their ids.
    span_ids = [span["span_id"] for span in base.spans]
    assert len(span_ids) == 18
    assert len(set(span_ids)) == 12
    assert second["experiments"]["base"]["indexed"] is True


def test_a_cancel_stops_between_requests_and_a_rerun_publishes_the_rest(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=4, cases_per_row=2)  # 12 spans per target
    out = tmp_path / "eval"
    cancel = threading.Event()

    def progress(message: str) -> None:
        if message.startswith("published"):
            cancel.set()  # the first Ctrl-C, while the first batch was being sent

    first = publish(fake, evaluation, out, clock, batch_size=6, cancel=cancel, progress=progress)
    assert first["status"] == "cancelled"
    assert "again" in first["hint"]
    assert api.events_calls == 1
    assert api.list_calls == 0  # nothing read back
    second = publish(fake, evaluation, out, clock, batch_size=6)
    assert second["status"] == "published"
    assert second["experiments"]["base"]["submitted"] == 6
    assert second["experiments"]["tuned"]["submitted"] == 12
    assert len(api.experiments) == 2


def test_a_cancel_stops_waiting_for_the_read_back(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.index_limit = 1
    cancel = threading.Event()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        cancel.set()

    report = publish(fake, Evaluation(), tmp_path / "eval", clock, cancel=cancel, sleep=sleep)
    assert report["status"] == "published"
    assert clock.sleeps == [10.0]
    assert api.list_calls == 4
    assert len(report["warnings"]) == 2
    for target, warning in zip(TARGETS, report["warnings"], strict=True):
        assert f"the {target} experiment to be indexed (cancelled)" in warning
        assert "1 of its 6 spans" in warning


def test_a_span_published_with_an_earlier_result_is_reported_not_resent(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    evaluation = Evaluation(rows=1, cases_per_row=2)
    out = tmp_path / "eval"
    evaluation.fail("tuned", evaluation.cases[0].id)
    publish(fake, evaluation, out, clock)
    fixed = Evaluation(rows=1, cases_per_row=2)  # the error was retried and judged
    report = publish(fake, fixed, out, clock)
    assert report["experiments"]["tuned"]["submitted"] == 0
    (warning,) = report["warnings"]
    assert "2 span(s) of the tuned experiment" in warning  # the child and its root
    assert "--out-dir" in warning


# --- reading back -------------------------------------------------------------------------


def test_publishing_waits_until_the_events_are_indexed(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.index_after_lists = 2  # both experiments are counted at once, then again
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert clock.sleeps == [10.0]
    assert api.list_calls == 4
    assert all(entry["indexed"] for entry in report["experiments"].values())
    assert report["experiments"]["tuned"]["events_indexed"] == 6
    assert report["warnings"] == []


@pytest.mark.parametrize("answer", ["ok", "enveloped"])
def test_the_event_count_is_read_from_the_shape_pup_prints(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path, answer: str
) -> None:
    # pup prints total_matching at the top level (checked live); an envelope is read too.
    api.events_list_answer = answer  # type: ignore[assignment]
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert all(entry["indexed"] for entry in report["experiments"].values())
    assert report["warnings"] == []


def test_events_not_indexed_in_time_are_a_warning(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path
) -> None:
    api.index_limit = 1
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert report["status"] == "published"
    # One 120 s budget for both experiments, polled every 10 s: counts at 0, 10, ..., 120 s.
    assert clock.sleeps == [10.0] * 12
    assert api.list_calls == 2 * 13
    assert all(entry["indexed"] is False for entry in report["experiments"].values())
    assert all(entry["events_indexed"] == 1 for entry in report["experiments"].values())
    assert len(report["warnings"]) == 2
    for target, warning in zip(TARGETS, report["warnings"], strict=True):
        assert f"the {target} experiment shows 1 of its 6 spans after 120s" in warning


@pytest.mark.parametrize("answer", ["fail", "no_total"])
def test_events_that_cannot_be_counted_are_a_warning(
    fake: FakePup, api: FakeExperimentsApi, clock: Clock, tmp_path: Path, answer: EventsListAnswer
) -> None:
    api.events_list_answer = answer
    report = publish(fake, Evaluation(), tmp_path / "eval", clock)
    assert report["status"] == "published"
    assert clock.sleeps == []
    for target, warning in zip(TARGETS, report["warnings"], strict=True):
        entry = report["experiments"][target]
        assert entry["indexed"] is False
        assert entry["events_indexed"] is None
        assert f"could not count the events of the {target} experiment" in warning
    assert len(report["warnings"]) == 2
