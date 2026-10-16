# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Resumable replay: what is stored, what a rerun skips, errors, cancellation and the summary."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.errors import ExternalError, SettingsConflict, StateError
from pup_finetune.evaluation.bootstrap import paired_bootstrap_ci
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.compare import attach_compare, stored_compare
from pup_finetune.evaluation.judge import Judgment
from pup_finetune.evaluation.replay import (
    CAVEAT,
    EvalDir,
    PairKey,
    evaluation_identity,
    remaining_work,
    run_replay,
    summarize,
)
from pup_finetune.inference import JudgeConfig
from pup_finetune.providers.base import CheckpointRef, Generation, Sampler
from pup_finetune.rendering.base import ChatMessage, ToolSpec

CKPT = CheckpointRef(
    kind="training", name="t1s4-abcd", uri="fake://ckpt/t1", epoch=1, step=4, created_in_session="s"
)


def make_case(row: str, index: int) -> ReplayCase:
    return ReplayCase(
        id=f"{row}:{index}",
        row_id=row,
        trace_id=f"trace-{row}",
        message_index=index,
        prefix=({"role": "user", "content": f"question {row} {index}"},),
        tools=(),
        reference={"role": "assistant", "content": f"answer {row} {index}"},
    )


CASES = [make_case(row, index) for row in ("row-a", "row-b", "row-c") for index in (1, 3)]
WITH_COMPARE = dict.fromkeys(("base", "tuned", "compare"), CASES)


def identity(
    cases: Sequence[ReplayCase] = CASES, *, max_tokens: int = 64, seed: int = 0, **changes: Any
) -> dict[str, Any]:
    data = evaluation_identity(
        run_id="ft-test",
        tuned_checkpoint=CKPT,
        base_model="base/model",
        judge=JudgeConfig(model="judge/model"),
        cases=cases,
        max_tokens=max_tokens,
        rendering={"renderer": "fake"},
        seed=seed,
        max_cases_per_trajectory=2,
    )
    return {**data, **changes}


class ScriptedSampler:
    """Answers the reference for the questions in ``knows``, a generic text otherwise.

    ``fail`` lists questions it raises for; ``on_call`` runs before each answer.
    """

    def __init__(
        self,
        knows: Iterable[str] = (),
        *,
        fail: Iterable[str] = (),
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self.knows = set(knows)
        self.fail = set(fail)
        self.on_call = on_call
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        question = str(messages[-1]["content"])
        with self._lock:
            self.calls.append(question)
            count = len(self.calls)
        if self.on_call is not None:
            self.on_call(count)
        if question in self.fail:
            raise ExternalError("the sampler gave up after its retries", source="sampler")
        content = question.replace("question", "answer") if question in self.knows else "hm"
        return Generation(
            message={"role": "assistant", "content": content},
            format_valid=True,
            prompt_tokens=10,
            completion_tokens=3,
            latency_s=0.0,
            stop_reason="stop",
        )

    def close(self) -> None:
        pass


def q(case: ReplayCase) -> str:
    return str(case.prefix[-1]["content"])


class Judge:
    """Passes a generation equal to the reference; raises for the questions in ``outage``."""

    def __init__(self, outage: Iterable[str] = ()) -> None:
        self.outage = set(outage)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, case: ReplayCase, generation: Generation) -> Judgment:
        with self._lock:
            self.calls += 1
        if q(case) in self.outage:
            raise ExternalError("judge unreachable after its retries", source="judge")
        passed = generation.message.get("content") == case.reference.get("content")
        return Judgment(passed=passed, reason="same" if passed else "different")


def replay(
    out_dir: Path,
    samplers: Mapping[str, Sampler],
    judge: Judge,
    *,
    cases: Sequence[ReplayCase] = CASES,
    cancel: threading.Event | None = None,
    concurrency: int = 2,
    **options: Any,
) -> Any:
    return run_replay(
        cases,
        samplers,
        judge,
        out_dir=out_dir,
        identity=identity(cases),
        concurrency=concurrency,
        cancel=cancel,
        **options,
    )


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- storing and resuming ------------------------------------------------------------------


def test_every_pair_is_generated_scored_judged_and_stored(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    base, tuned = ScriptedSampler(), ScriptedSampler({q(case) for case in CASES})
    outcome = replay(out, {"base": base, "tuned": tuned}, Judge())
    assert outcome.complete
    assert len(base.calls) == len(tuned.calls) == len(CASES)
    generations, results = lines(out / "generations.jsonl"), lines(out / "results.jsonl")
    assert len(generations) == len(results) == 2 * len(CASES)
    assert {(r["target"], r["case_id"]) for r in results} == {
        (t, c.id) for t in ("base", "tuned") for c in CASES
    }
    first = next(r for r in results if r["target"] == "tuned")
    assert first["status"] == "ok"
    assert first["judgment"]["passed"] is True
    assert first["metrics"]["format_valid"] is True
    assert json.loads((out / "evaluation.json").read_text()) == identity()
    assert json.loads((out / "summary.json").read_text()) == outcome.summary
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    for name in ("generations.jsonl", "results.jsonl", "summary.json", "evaluation.json"):
        assert stat.S_IMODE((out / name).stat().st_mode) == 0o600


def test_a_rerun_skips_the_completed_pairs(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    replay(out, {"base": ScriptedSampler(), "tuned": ScriptedSampler()}, Judge())
    base, tuned, judge = ScriptedSampler(), ScriptedSampler(), Judge()
    again = replay(out, {"base": base, "tuned": tuned}, judge)
    assert again.complete
    assert base.calls == tuned.calls == []
    assert judge.calls == 0
    assert remaining_work(EvalDir(out), CASES) == {"done": 12, "generate": 0, "judge": 0}
    # Nothing left to generate: no sampler is needed at all.
    assert replay(out, {}, Judge()).complete


def test_a_rerun_does_only_what_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    first = ScriptedSampler(fail={q(CASES[0])})
    judge = Judge(outage={q(CASES[1])})
    outcome = replay(out, {"base": ScriptedSampler(), "tuned": first}, judge)
    assert outcome.complete  # every pair has a result, two of them errors
    assert outcome.summary["errors"] == {"base": 1, "tuned": 2}
    assert remaining_work(EvalDir(out), CASES) == {"done": 9, "generate": 1, "judge": 3}

    base, tuned, judge = ScriptedSampler(), ScriptedSampler(), Judge()
    again = replay(out, {"base": base, "tuned": tuned}, judge)
    # The failed generation is retried; the pairs the judge missed reuse their generation.
    assert tuned.calls == [q(CASES[0])]
    assert base.calls == []
    assert judge.calls == 3
    assert again.summary["errors"] == {"base": 0, "tuned": 0}


class SurrogateSampler(ScriptedSampler):
    """Calls a tool with an argument a parser decoded from ``"\\ud800"``: a lone surrogate."""

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        super().generate(messages, tools, max_tokens=max_tokens)
        arguments = json.dumps({"query": "half \ud800"}, ensure_ascii=False)
        call = {
            "id": "call-0",
            "type": "function",
            "function": {"name": "lookup", "arguments": arguments},
        }
        return Generation(
            message={"role": "assistant", "content": None, "tool_calls": [call]},
            format_valid=True,
            prompt_tokens=10,
            completion_tokens=3,
            latency_s=0.0,
            stop_reason="stop",
        )


def test_a_lone_surrogate_in_a_generated_tool_call_is_stored(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    tuned = SurrogateSampler()
    outcome = replay(out, {"base": ScriptedSampler(), "tuned": tuned}, Judge())
    assert outcome.complete
    assert len(tuned.calls) == len(CASES)
    generations = EvalDir(out).read_generations()
    stored = generations[("tuned", CASES[0].id)]["message"]["tool_calls"][0]
    assert json.loads(stored["function"]["arguments"]) == {"query": "half \ud800"}
    results = EvalDir(out).read_results()
    assert all(results[("tuned", case.id)]["status"] == "ok" for case in CASES)
    assert json.loads((out / "summary.json").read_text()) == outcome.summary

    # A rerun after the results are lost reuses the stored generations: nothing is paid twice.
    (out / "results.jsonl").unlink()
    again = SurrogateSampler()
    assert replay(out, {"base": ScriptedSampler(), "tuned": again}, Judge()).complete
    assert again.calls == []


def test_a_target_without_a_sampler_is_refused_while_it_has_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no sampler for tuned"):
        replay(tmp_path / "eval", {"base": ScriptedSampler()}, Judge())


def test_a_torn_last_line_is_done_again(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    replay(out, {"base": ScriptedSampler(), "tuned": ScriptedSampler()}, Judge())
    results = out / "results.jsonl"
    text = results.read_text().splitlines()
    torn = json.loads(text[-1])
    results.write_text("\n".join(text[:-1]) + "\n" + text[-1][:20])
    judge = Judge()
    again = replay(out, {}, judge)
    assert judge.calls == 1
    assert again.complete
    assert (torn["target"], torn["case_id"]) in EvalDir(out).read_results()


# --- identity -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"judge": JudgeConfig(model="other/judge").identity()}, "judge"),
        ({"base_model": "other/base"}, "base_model"),
        ({"rendering": {"renderer": "other"}}, "rendering"),
        ({"metrics_version": "0"}, "metrics_version"),
        ({"run_id": "ft-other"}, "run_id"),
        ({"tuned_checkpoint": {**dataclasses.asdict(CKPT), "uri": "fake://ckpt/t2"}}, "tuned"),
        ({"max_cases_per_trajectory": 1}, "max_cases_per_trajectory"),
        ({"seed": 1}, "seed"),
    ],
)
def test_other_settings_in_the_same_directory_are_refused(
    tmp_path: Path, change: dict[str, Any], field: str
) -> None:
    out = tmp_path / "eval"
    replay(out, {"base": ScriptedSampler(), "tuned": ScriptedSampler()}, Judge())
    with pytest.raises(StateError, match=field) as raised:
        run_replay(
            CASES,
            {"base": ScriptedSampler(), "tuned": ScriptedSampler()},
            Judge(),
            out_dir=out,
            identity=identity(**change),
        )
    assert "--out-dir" in (raised.value.hint or "")


def test_other_cases_or_max_tokens_are_refused(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    replay(out, {"base": ScriptedSampler(), "tuned": ScriptedSampler()}, Judge())
    fewer = CASES[:-1]
    with pytest.raises(StateError, match="cases_sha256"):
        run_replay(fewer, {}, Judge(), out_dir=out, identity=identity(fewer))
    with pytest.raises(StateError, match="max_tokens"):
        run_replay(CASES, {}, Judge(), out_dir=out, identity=identity(max_tokens=32))


def test_the_generations_are_capped_at_the_identitys_max_tokens(tmp_path: Path) -> None:
    seen: list[int] = []

    class Capped(ScriptedSampler):
        def generate(
            self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
        ) -> Generation:
            seen.append(max_tokens)
            return super().generate(messages, tools, max_tokens=max_tokens)

    run_replay(
        CASES,
        {"base": Capped(), "tuned": Capped()},
        Judge(),
        out_dir=tmp_path / "eval",
        identity=identity(max_tokens=32),
    )
    assert set(seen) == {32}


def test_an_identity_that_does_not_describe_the_call_is_a_bug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cases"):
        run_replay(CASES, {}, Judge(), out_dir=tmp_path, identity=identity(CASES[:1]))
    with pytest.raises(ValueError, match="max_tokens"):
        run_replay(CASES, {}, Judge(), out_dir=tmp_path, identity=identity(max_tokens=0))


def test_a_directory_without_work_takes_new_settings(tmp_path: Path) -> None:
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    eval_dir.check_identity(identity())
    eval_dir.write_calibration({"ok": False})
    eval_dir.check_identity(identity(judge=JudgeConfig(model="other/judge").identity()))
    assert eval_dir.read_calibration() is None
    assert eval_dir.stored_identity() == identity(judge=JudgeConfig(model="other/judge").identity())


# --- errors -------------------------------------------------------------------------------


def test_errors_are_recorded_and_left_out_of_every_rate(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    everything = {q(case) for case in CASES}
    tuned = ScriptedSampler(everything, fail={q(CASES[0])})
    judge = Judge(outage={q(CASES[1])})
    outcome = replay(out, {"base": ScriptedSampler(), "tuned": tuned}, judge)
    results = {(r["target"], r["case_id"]): r for r in lines(out / "results.jsonl")}
    failed = results[("tuned", CASES[0].id)]
    assert failed["status"] == "error"
    assert failed["error"]["stage"] == "generation"
    assert failed["judgment"] is None
    outage = results[("tuned", CASES[1].id)]
    assert outage["error"]["stage"] == "judge"
    assert outage["metrics"] is not None  # kept for inspection, not counted

    tuned_summary = outcome.summary["targets"]["tuned"]
    assert tuned_summary["errors"] == 2
    assert tuned_summary["errors_by_stage"] == {"generation": 1, "judge": 1}
    # Four tuned results are ok, all passing: errors do not count as failures.
    assert tuned_summary["judge_pass"] == {"rate": 1.0, "passed": 4, "n": 4}
    assert tuned_summary["metrics"]["format_valid"]["n"] == 4
    assert outcome.summary["targets"]["base"]["judge_pass"] == {"rate": 0.0, "passed": 0, "n": 5}
    # The judge outage hit both targets' case 1; only tuned failed to generate case 0.
    assert outcome.summary["errors"] == {"base": 1, "tuned": 2}
    assert outcome.summary["paired"]["both_error"] == 1
    assert outcome.summary["paired"]["one_error"] == 1
    assert outcome.summary["paired"]["n"] == 4


def test_stored_error_messages_are_redacted(tmp_path: Path) -> None:
    token = "abcdef0123456789opaque"  # noqa: S105 - synthetic test value

    class Leaky(ScriptedSampler):
        def generate(
            self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
        ) -> Generation:
            raise ExternalError(f"401 for Authorization: Bearer {token}", source="sampler")

    replay(tmp_path / "eval", {"base": Leaky(), "tuned": ScriptedSampler()}, Judge())
    text = (tmp_path / "eval" / "results.jsonl").read_text()
    assert token not in text
    failed = [r for r in lines(tmp_path / "eval" / "results.jsonl") if r["status"] == "error"]
    assert failed
    assert all("Bearer" in r["error"]["message"] for r in failed)


def test_answers_without_a_verdict_are_counted_per_target(tmp_path: Path) -> None:
    def judge(case: ReplayCase, generation: Generation) -> Judgment:
        if case.id == CASES[0].id:
            return Judgment(passed=False, reason="", raw="???", error="unparseable")
        return Judgment(passed=True, reason="fine")

    outcome = run_replay(
        CASES,
        {"base": ScriptedSampler(), "tuned": ScriptedSampler()},
        judge,
        out_dir=tmp_path / "eval",
        identity=identity(),
    )
    for target in ("base", "tuned"):
        summary = outcome.summary["targets"][target]
        assert summary["judge_no_verdict"] == 1
        assert summary["judge_no_verdict_by_error"] == {"unparseable": 1}
        assert summary["errors"] == 0  # an answer without a verdict is a failure, not an error


def test_a_failed_write_waits_for_the_pairs_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error that is not a sampler's or the judge's stops the run, but only once every
    pair already running has settled, so none writes after the lock is released."""
    out = tmp_path / "eval"
    running = threading.Event()
    release = threading.Event()
    settled: list[str] = []

    def slow(count: int) -> None:
        if count == 1:
            running.set()
            release.wait(5)
            threading.Event().wait(0.2)  # still running well after the failure
            settled.append("first")

    real_append = EvalDir.append_generation
    appended = 0

    def append(self: EvalDir, record: Mapping[str, Any]) -> None:
        nonlocal appended
        appended += 1
        if appended == 1:
            assert running.wait(5)
            release.set()
            raise OSError("disk full")
        real_append(self, record)

    monkeypatch.setattr(EvalDir, "append_generation", append)
    base = ScriptedSampler(on_call=slow)
    tuned = ScriptedSampler()
    with pytest.raises(OSError, match="disk full"):
        replay(out, {"base": base, "tuned": tuned}, Judge(), concurrency=2)
    assert settled == ["first"]  # the pair in flight finished before run_replay returned
    assert not any(thread.name.startswith("replay") for thread in threading.enumerate())


def test_an_existing_out_dir_is_made_private(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    out.mkdir()
    os.chmod(out, 0o755)  # noqa: S103 - the permissive mode create() must tighten
    EvalDir(out).create()
    assert stat.S_IMODE(out.stat().st_mode) == 0o700


# --- cancellation -------------------------------------------------------------------------


def test_a_cancel_keeps_what_finished_and_a_rerun_completes(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    cancel = threading.Event()

    def stop_after_three(count: int) -> None:
        if count >= 3:
            cancel.set()

    base = ScriptedSampler(on_call=stop_after_three)
    tuned = ScriptedSampler({q(case) for case in CASES})
    outcome = replay(out, {"base": base, "tuned": tuned}, Judge(), cancel=cancel, concurrency=1)
    assert outcome.cancelled
    assert not outcome.complete
    assert outcome.pending > 0
    assert outcome.summary["status"] == "partial"
    kept = EvalDir(out).read_results()
    assert 0 < len(kept) < 2 * len(CASES)
    assert json.loads((out / "summary.json").read_text())["status"] == "partial"

    base2, tuned2 = ScriptedSampler(), ScriptedSampler({q(case) for case in CASES})
    again = replay(out, {"base": base2, "tuned": tuned2}, Judge())
    assert again.complete
    assert again.summary["status"] == "complete"
    # Each pair was generated exactly once over both runs.
    assert len(base.calls) + len(base2.calls) == len(CASES)
    assert len(tuned.calls) + len(tuned2.calls) == len(CASES)


def test_a_second_interrupt_returns_at_once_and_writes_nothing_after(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    blocked = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def block_first(count: int) -> None:
        if count == 1:
            blocked.set()
            release.wait(5)
            finished.set()

    def interrupt(event: Mapping[str, Any]) -> None:
        assert blocked.wait(5)
        raise KeyboardInterrupt  # a second Ctrl-C, in the main thread

    base = ScriptedSampler(on_call=block_first)
    tuned = ScriptedSampler()
    with pytest.raises(KeyboardInterrupt):
        replay(out, {"base": base, "tuned": tuned}, Judge(), concurrency=2, on_progress=interrupt)
    assert not finished.is_set()  # returned while a generation was still blocked
    workers = [t for t in threading.enumerate() if t.name.startswith("replay")]
    assert workers
    assert all(worker.daemon for worker in workers)  # never joined at interpreter exit
    written = {path.name: path.read_bytes() for path in out.iterdir() if path.suffix == ".jsonl"}

    release.set()
    assert finished.wait(5)
    for worker in workers:
        worker.join(5)
    after = {path.name: path.read_bytes() for path in out.iterdir() if path.suffix == ".jsonl"}
    assert after == written  # the abandoned generation was not appended


def test_a_cancel_before_starting_runs_nothing(tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    base = ScriptedSampler()
    outcome = replay(
        tmp_path / "eval", {"base": base, "tuned": ScriptedSampler()}, Judge(), cancel=cancel
    )
    assert base.calls == []
    assert outcome.pending == 2 * len(CASES)


# --- publishing ---------------------------------------------------------------------------


class RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    def publish(
        self,
        *,
        cases: Sequence[ReplayCase],
        generations: Mapping[PairKey, Mapping[str, Any]],
        results: Mapping[PairKey, Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        self.published.append(
            {"cases": len(cases), "generations": len(generations), "results": len(results)}
        )
        return {"status": "published", "count": len(self.published)}


def test_the_publisher_gets_a_complete_evaluation_only(tmp_path: Path) -> None:
    publisher = RecordingPublisher()
    cancel = threading.Event()
    cancel.set()
    samplers = {"base": ScriptedSampler(), "tuned": ScriptedSampler()}
    cut = replay(tmp_path / "eval", samplers, Judge(), cancel=cancel, publisher=publisher)
    assert publisher.published == []
    assert cut.publication is None
    done = replay(tmp_path / "eval", samplers, Judge(), publisher=publisher)
    assert publisher.published == [{"cases": 6, "generations": 12, "results": 12}]
    assert done.publication == {"status": "published", "count": 1}  # what it reported


def test_the_default_publisher_reports_nothing(tmp_path: Path) -> None:
    samplers = {"base": ScriptedSampler(), "tuned": ScriptedSampler()}
    assert replay(tmp_path / "eval", samplers, Judge()).publication is None


# --- the summary --------------------------------------------------------------------------


def ok(target: str, case: ReplayCase, passed: bool) -> dict[str, Any]:
    return {
        "target": target,
        "case_id": case.id,
        "row_id": case.row_id,
        "status": "ok",
        "metrics": {"format_valid": True, "tool_name_match": None},
        "judgment": {"passed": passed, "reason": "", "raw": None, "error": None},
    }


def error(target: str, case: ReplayCase) -> dict[str, Any]:
    return {
        "target": target,
        "case_id": case.id,
        "row_id": case.row_id,
        "status": "error",
        "error": {"stage": "generation", "type": "ExternalError", "message": "x"},
        "metrics": None,
        "judgment": None,
    }


def results_for(
    verdicts: Mapping[str, tuple[bool | None, bool | None]], cases: Sequence[ReplayCase]
) -> dict[PairKey, dict[str, Any]]:
    """``verdicts[case id] = (base, tuned)``; ``None`` is an error result."""
    by_id = {case.id: case for case in cases}
    found: dict[PairKey, dict[str, Any]] = {}
    for case_id, pair in verdicts.items():
        for target, verdict in zip(("base", "tuned"), pair, strict=True):
            case = by_id[case_id]
            record = error(target, case) if verdict is None else ok(target, case, verdict)
            found[(target, case_id)] = record
    return found


def test_the_summary_compares_the_targets_pair_by_pair() -> None:
    verdicts = {
        "row-a:1": (False, True),  # win
        "row-a:3": (False, True),  # win
        "row-b:1": (True, False),  # regression
        "row-b:3": (True, True),  # tie
        "row-c:1": (None, None),  # both error
        "row-c:3": (None, True),  # one error
    }
    summary = summarize(CASES, results_for(verdicts, CASES), {}, seed=0)
    assert summary["paired"] == {
        "n": 4,
        "wins": 2,
        "regressions": 1,
        "ties": 1,
        "both_error": 1,
        "one_error": 1,
        "pending": 0,
    }
    delta = summary["tuned_minus_base"]
    assert delta["judge_pass_rate"] == pytest.approx(0.25)  # (1 + 1 - 1 + 0) / 4
    assert delta["n_cases"] == 4
    assert delta["n_trajectories"] == 2
    assert summary["targets"]["tuned"]["judge_pass"] == {"rate": 0.8, "passed": 4, "n": 5}
    assert summary["targets"]["base"]["judge_pass"] == {"rate": 0.5, "passed": 2, "n": 4}
    # Per trajectory: base a=0, b=1 (c has no ok base result); tuned a=1, b=0.5, c=1.
    assert summary["targets"]["base"]["trajectory_agreement"] == {"mean": 0.5, "n_trajectories": 2}
    tuned_agreement = summary["targets"]["tuned"]["trajectory_agreement"]
    assert tuned_agreement["mean"] == pytest.approx(2.5 / 3)
    assert summary["errors"] == {"base": 2, "tuned": 1}
    assert summary["caveat"] == CAVEAT
    assert summary["status"] == "complete"
    json.dumps(summary, allow_nan=False)


def test_a_regression_makes_the_delta_negative() -> None:
    verdicts = {case.id: (True, False) for case in CASES}
    summary = summarize(CASES, results_for(verdicts, CASES), {}, seed=0)
    assert summary["tuned_minus_base"]["judge_pass_rate"] == -1.0
    assert summary["tuned_minus_base"]["ci95"] == [-1.0, -1.0]
    assert summary["paired"]["regressions"] == len(CASES)


def test_without_results_every_number_is_null_not_nan() -> None:
    summary = summarize(CASES, {}, {}, seed=0)
    assert summary["status"] == "partial"
    assert summary["paired"]["pending"] == len(CASES)
    assert summary["tuned_minus_base"]["judge_pass_rate"] is None
    assert summary["tuned_minus_base"]["ci95"] is None
    assert summary["targets"]["base"]["judge_pass"]["rate"] is None
    assert summary["targets"]["base"]["trajectory_agreement"]["mean"] is None
    json.dumps(summary, allow_nan=False)


def test_one_trajectory_is_too_few_for_an_interval() -> None:
    cases = [make_case("row-a", index) for index in range(1, 6)]
    verdicts = {case.id: (False, True) for case in cases}
    summary = summarize(cases, results_for(verdicts, cases), {}, seed=0)
    assert summary["tuned_minus_base"]["judge_pass_rate"] == 1.0
    assert summary["tuned_minus_base"]["ci95"] is None


def test_the_summary_counts_generated_tokens() -> None:
    generations = {
        ("base", CASES[0].id): {"prompt_tokens": 10, "completion_tokens": 2},
        ("tuned", CASES[0].id): {"prompt_tokens": 11, "completion_tokens": 4},
        ("tuned", CASES[1].id): {"prompt_tokens": 9, "completion_tokens": 1},
    }
    summary = summarize(CASES, {}, generations, seed=0)
    assert summary["tokens"]["base"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "generations": 1,
    }
    assert summary["tokens"]["tuned"] == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "generations": 2,
    }


def test_the_bootstrap_is_deterministic_for_a_seed() -> None:
    clusters = [[1.0, 0.0], [0.0], [1.0, 1.0, -1.0], [0.0, 1.0], [-1.0]]
    first = paired_bootstrap_ci(clusters, seed=7)
    assert first == paired_bootstrap_ci(clusters, seed=7)
    assert first is not None
    low, high = first
    assert -1.0 <= low <= high <= 1.0
    others = {paired_bootstrap_ci(clusters, seed=seed) for seed in range(5)}
    assert len(others) > 1  # the seed matters


def test_the_bootstrap_resamples_whole_trajectories() -> None:
    # Resampling the two trajectories as wholes can only give 1 (both the first), 0 (both
    # the second) or 0.8 (one of each): the interval spans 0 to 1. Resampling the five
    # cases one by one would rarely reach 0.
    interval = paired_bootstrap_ci([[1.0, 1.0, 1.0, 1.0], [0.0]], seed=0)
    assert interval == (0.0, 1.0)


def test_the_bootstrap_needs_two_trajectories() -> None:
    assert paired_bootstrap_ci([[1.0, 0.0, 1.0]], seed=0) is None
    assert paired_bootstrap_ci([[1.0], []], seed=0) is None
    assert paired_bootstrap_ci([], seed=0) is None


def test_generations_cut_off_at_max_tokens_are_counted(tmp_path: Path) -> None:
    class Rambling(ScriptedSampler):
        def generate(
            self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
        ) -> Generation:
            found = super().generate(messages, tools, max_tokens=max_tokens)
            return dataclasses.replace(found, stop_reason="length", format_valid=False)

    outcome = replay(tmp_path / "eval", {"base": Rambling(), "tuned": ScriptedSampler()}, Judge())
    assert outcome.summary["targets"]["base"]["truncated"] == len(CASES)
    assert outcome.summary["targets"]["tuned"]["truncated"] == 0


class LiveRecorder:
    """A live publisher that records when it was started and offered results."""

    def __init__(self, sampler_calls: list[str]) -> None:
        self.sampler_calls = sampler_calls
        self.started_after_calls: int | None = None
        self.offered: list[int] = []
        self.published = False

    def start(self) -> str | None:
        self.started_after_calls = len(self.sampler_calls)
        return "https://example.invalid/compare"

    def publish_ready(
        self,
        *,
        cases: Sequence[ReplayCase],
        generations: Mapping[PairKey, Mapping[str, Any]],
        results: Mapping[PairKey, Mapping[str, Any]],
    ) -> None:
        self.offered.append(len(results))

    def publish(self, **kwargs: Any) -> dict[str, Any]:
        self.published = True
        return {"status": "published"}


def test_a_live_publisher_starts_before_the_first_pair_and_is_offered_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pup_finetune.evaluation.replay as replay_module

    monkeypatch.setattr(replay_module, "LIVE_PUBLISH_INTERVAL_S", 0.0)
    base = ScriptedSampler(on_call=lambda count: time.sleep(0.01))
    live = LiveRecorder(base.calls)
    outcome = replay(
        tmp_path / "eval", {"base": base, "tuned": ScriptedSampler()}, Judge(), publisher=live
    )
    assert live.started_after_calls == 0
    assert live.offered and max(live.offered) <= 2 * len(CASES)
    assert live.published
    assert outcome.link == "https://example.invalid/compare"


# --- a third target -----------------------------------------------------------------------


def _three_target_store(tmp_path: Path) -> tuple[dict[PairKey, Any], dict[PairKey, Any]]:
    samplers = {
        "base": ScriptedSampler(knows=[q(CASES[0])]),
        "tuned": ScriptedSampler(knows=[q(case) for case in CASES[:4]]),
        "compare": ScriptedSampler(knows=[q(case) for case in CASES[1:6]]),
    }
    run_replay(
        CASES,
        samplers,
        Judge(),
        out_dir=tmp_path / "eval",
        identity=identity(),
        replays=WITH_COMPARE,
        compare=compare_identity(),
    )
    eval_dir = EvalDir(tmp_path / "eval")
    return eval_dir.read_results(), eval_dir.read_generations()


def test_a_compare_target_adds_its_own_comparison_and_changes_no_other(tmp_path: Path) -> None:
    results, generations = _three_target_store(tmp_path)
    two = summarize(CASES, results, generations, seed=0)
    three = summarize(CASES, results, generations, seed=0, replays=WITH_COMPARE)
    assert three["paired"] == two["paired"]
    assert three["tuned_minus_base"] == two["tuned_minus_base"]
    assert "tuned_minus_compare" not in two
    # The compare target in place of base gives the same comparison.
    as_base = {
        ("base", case_id): result for (target, case_id), result in results.items()
        if target == "compare"
    } | {key: result for key, result in results.items() if key[0] == "tuned"}  # fmt: skip
    swapped = summarize(CASES, as_base, generations, seed=0)
    assert three["tuned_minus_compare"] == {
        **swapped["tuned_minus_base"],
        "paired": swapped["paired"],
    }
    assert three["tuned_minus_compare"]["paired"]["regressions"] == 2


def compare_identity(model: str = "compare/model", max_tokens: int = 100) -> dict[str, Any]:
    config = JudgeConfig(model=model, max_tokens=max_tokens)
    return {"spec": model, "model": config.identity()}


def test_the_compare_target_is_frozen_when_attached(tmp_path: Path) -> None:
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    attach_compare(eval_dir.compare_path, None)
    assert stored_compare(eval_dir.compare_path) is None
    attach_compare(eval_dir.compare_path, compare_identity())
    attach_compare(eval_dir.compare_path, compare_identity())
    assert stored_compare(eval_dir.compare_path) == compare_identity()
    # Another model is refused with no compare pair at all.
    with pytest.raises(SettingsConflict, match="compares with compare/model, not other/model"):
        attach_compare(eval_dir.compare_path, compare_identity("other/model"))


def test_a_plan_without_the_attached_compare_target_is_refused(tmp_path: Path) -> None:
    samplers = {"base": ScriptedSampler(), "tuned": ScriptedSampler(), "compare": ScriptedSampler()}
    out = tmp_path / "eval"
    run_replay(CASES, samplers, Judge(), out_dir=out, identity=identity(),
               replays=WITH_COMPARE, compare=compare_identity())  # fmt: skip
    summary_before = EvalDir(out).summary_path.read_bytes()
    with pytest.raises(SettingsConflict, match="planned without"):
        run_replay(CASES, {}, Judge(), out_dir=out, identity=identity())
    assert EvalDir(out).summary_path.read_bytes() == summary_before


def test_the_compare_target_samples_with_its_own_token_limit(tmp_path: Path) -> None:
    seen: dict[str, set[int]] = {}

    class Limits(ScriptedSampler):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def generate(
            self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
        ) -> Generation:
            seen.setdefault(self.name, set()).add(max_tokens)
            return super().generate(messages, tools, max_tokens=max_tokens)

    samplers = {name: Limits(name) for name in ("base", "tuned", "compare")}
    run_replay(CASES, samplers, Judge(), out_dir=tmp_path / "eval", identity=identity(max_tokens=64),
               replays=WITH_COMPARE, compare=compare_identity(max_tokens=16000))  # fmt: skip
    assert seen == {"base": {64}, "tuned": {64}, "compare": {16000}}


def test_a_new_identity_without_work_retires_the_compare_target(tmp_path: Path) -> None:
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    eval_dir.check_identity(identity())
    attach_compare(eval_dir.compare_path, compare_identity())
    eval_dir.check_identity(identity(seed=5))
    assert stored_compare(eval_dir.compare_path) is None


def test_a_target_replays_only_its_own_cases(tmp_path: Path) -> None:
    knows = [q(case) for case in CASES]
    samplers = {
        "base": ScriptedSampler(),
        "tuned": ScriptedSampler(knows=knows),
        "compare": ScriptedSampler(knows=knows),
    }
    replays = {"base": CASES, "tuned": CASES, "compare": CASES[2:]}
    outcome = run_replay(
        CASES, samplers, Judge(), out_dir=tmp_path / "eval", identity=identity(),
        replays=replays, compare=compare_identity(),
    )  # fmt: skip
    assert outcome.complete
    assert len(samplers["compare"].calls) == len(CASES) - 2
    assert len(samplers["tuned"].calls) == len(CASES)
    summary = outcome.summary
    compared = summary["targets"]["compare"]
    assert (compared["skipped"], compared["pending"]) == (2, 0)
    assert compared["judge_pass"] == {"rate": 1.0, "passed": 4, "n": 4}
    # tuned minus compare pairs only the cases both answered.
    versus = summary["tuned_minus_compare"]
    assert (versus["n_cases"], versus["paired"]["pending"]) == (4, 0)
    assert summary["targets"]["tuned"]["skipped"] == 0
    assert summary["tuned_minus_base"]["n_cases"] == len(CASES)
    assert ("compare", CASES[0].id) not in EvalDir(tmp_path / "eval").read_results()


def test_a_target_replays_only_cases_the_evaluation_holds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not hold"):
        run_replay(
            CASES[:2], {}, Judge(), out_dir=tmp_path, identity=identity(CASES[:2]),
            replays={"base": CASES[:2], "tuned": CASES},
        )  # fmt: skip
