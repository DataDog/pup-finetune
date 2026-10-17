# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The dataset's store of base results: keys, the index, and copying in and out."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.evaluation import reuse
from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.replay import EvalDir
from pup_finetune.evaluation.reuse import (
    ReuseIndex,
    Wanted,
    export_target,
    generation_key,
    import_target,
    key_inputs,
    plan_target,
    stamp,
    store_path,
    verdict_key,
)
from pup_finetune.inference import JudgeConfig
from pup_finetune.providers.fake import FAKE_PROFILE
from pup_finetune.storage import DirectoryLock, append_jsonl

JUDGE = JudgeConfig(model="judge/model")
RENDERING = {"renderer": "fake", "renderer_version": "1"}
SOURCE = {"command": "evaluate", "run_id": "ft-a"}
CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"id": "7"}'},
        }
    ],
}


def make_case(row: str, index: int = 1, **changes: Any) -> ReplayCase:
    case = ReplayCase(
        id=f"{row}:{index}",
        row_id=row,
        trace_id=f"trace-{row}",
        message_index=index,
        prefix=({"role": "user", "content": f"question {row}"},),
        tools=({"type": "function", "function": {"name": "lookup"}},),
        reference=CALL,
        future_tool_results=({"role": "tool", "tool_call_id": "c1", "content": "found"},),
    )
    return dataclasses.replace(case, **changes)


CASES = [make_case("row-a"), make_case("row-b")]


def key(case: ReplayCase, **changes: Any) -> str:
    settings: dict[str, Any] = {
        "provider": "fake",
        "profile": FAKE_PROFILE,
        "rendering": RENDERING,
        "max_tokens": 64,
        **changes,
    }
    inputs = key_inputs(
        settings["provider"], settings["profile"], settings["rendering"], settings["max_tokens"]
    )
    return generation_key(inputs, case)


INPUTS = key_inputs("fake", FAKE_PROFILE, RENDERING, 64)
KEYS = {case.id: key(case) for case in CASES}


def answer(text: str = "hi") -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def generation(case: ReplayCase, message: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    return {
        "key": KEYS[case.id],
        "message": dict(message),
        "format_valid": True,
        "prompt_tokens": 9,
        "completion_tokens": 2,
        "latency_s": 1.5,
        "stop_reason": "stop",
        "sampled_at": "2026-10-12T09:14:03+00:00",
        "source": SOURCE,
        **changes,
    }


def verdict(case: ReplayCase, message: Mapping[str, Any], *, passed: bool = True) -> dict[str, Any]:
    return {
        "key": verdict_key(JUDGE, case, message),
        "judgment": {"passed": passed, "reason": "same", "raw": "{}", "error": None},
        "judged_at": "2026-10-12T09:15:00+00:00",
    }


def store(dataset: Path, generations: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> None:
    path = store_path(dataset)
    path.mkdir(parents=True, exist_ok=True)
    append_jsonl(path / "generations.jsonl", generations)
    append_jsonl(path / "verdicts.jsonl", verdicts)


def load(dataset: Path, own: Mapping[Any, Any] | None = None) -> ReuseIndex:
    return ReuseIndex.load(dataset, JUDGE, [Wanted("base", CASES, KEYS, own or {})])


# --- keys ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "fireworks"},
        {"profile": dataclasses.replace(FAKE_PROFILE, base_model="other/model")},
        {"profile": dataclasses.replace(FAKE_PROFILE, max_seq_len=2048)},
        {"rendering": {**RENDERING, "renderer_version": "2"}},
        {"rendering": {**RENDERING, "transformers_version": "5.0.0"}},
        {"max_tokens": 65},
    ],
)
def test_the_generation_key_covers_what_the_model_is_sent(changes: dict[str, Any]) -> None:
    assert key(CASES[0], **changes) != key(CASES[0])


@pytest.mark.parametrize(
    "changes",
    [{"prefix": ({"role": "user", "content": "another question"},)}, {"tools": ()}],
)
def test_the_generation_key_covers_the_input(changes: dict[str, Any]) -> None:
    assert key(make_case("row-a", **changes)) != key(make_case("row-a"))


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "other:9"},
        {"row_id": "other"},
        {"split": "validation"},
        {"reference": answer("another reference")},
        {"future_tool_results": ()},
    ],
)
def test_the_generation_key_leaves_out_what_only_names_or_judges_the_case(
    changes: dict[str, Any],
) -> None:
    assert key(make_case("row-a", **changes)) == key(make_case("row-a"))


@pytest.mark.parametrize(
    ("case", "judge", "message"),
    [
        (CASES[0], JudgeConfig(model="judge/other"), answer()),
        (make_case("row-a", reference=answer("other")), JUDGE, answer()),
        (make_case("row-a", future_tool_results=()), JUDGE, answer()),
        (CASES[0], JUDGE, answer("bye")),
    ],
)
def test_the_verdict_key_covers_the_judge_and_its_request(
    case: ReplayCase, judge: JudgeConfig, message: dict[str, Any]
) -> None:
    assert verdict_key(judge, case, message) != verdict_key(JUDGE, CASES[0], answer())


def test_the_verdict_key_leaves_out_the_case_id_and_the_reasoning() -> None:
    thought = {**answer(), "reasoning_content": "let me think"}
    assert verdict_key(JUDGE, make_case("row-a", id="other:2"), thought) == verdict_key(
        JUDGE, CASES[0], answer()
    )


# --- the index ----------------------------------------------------------------------------


def test_a_missing_store_is_an_empty_index(tmp_path: Path) -> None:
    index = load(tmp_path)
    assert (index.generations, index.verdicts, index.warnings) == ({}, {}, ())
    assert index.store == tmp_path / "replay_cache"


def test_the_index_keeps_the_first_line_of_a_key_and_skips_a_torn_one(tmp_path: Path) -> None:
    a, b = CASES
    store(
        tmp_path,
        [generation(a, answer("first")), generation(a, answer("second"))],
        [verdict(a, answer("first"))],
    )
    with (store_path(tmp_path) / "generations.jsonl").open("a") as handle:
        handle.write(json.dumps(generation(b, answer()))[:40])
    index = load(tmp_path)
    assert index.generations[KEYS[a.id]]["message"] == answer("first")
    assert KEYS[b.id] not in index.generations
    assert list(index.verdicts) == [verdict_key(JUDGE, a, answer("first"))]


def test_the_index_loads_only_the_keys_needed(tmp_path: Path) -> None:
    a = CASES[0]
    other = make_case("row-z")
    unrelated = {**generation(a, answer()), "key": key(other)}
    store(tmp_path, [unrelated, generation(a, answer())], [verdict(other, answer())])
    index = load(tmp_path)
    assert list(index.generations) == [KEYS[a.id]]
    assert index.verdicts == {}


def test_the_index_loads_the_verdicts_of_the_directory_s_own_answers(tmp_path: Path) -> None:
    a = CASES[0]
    store(tmp_path, [], [verdict(a, answer("own"))])
    own = {("base", a.id): {"message": answer("own"), "format_valid": True}}
    assert list(load(tmp_path, own).verdicts) == [verdict_key(JUDGE, a, answer("own"))]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_store_is_an_empty_index_with_a_warning(tmp_path: Path) -> None:
    store(tmp_path, [generation(CASES[0], answer())], [])
    path = store_path(tmp_path) / "generations.jsonl"
    path.chmod(0)
    try:
        index = load(tmp_path)
    finally:
        path.chmod(0o600)
    assert index.generations == {}
    (warning,) = index.warnings
    assert warning.startswith("base results of earlier runs were not reused: ")
    assert "cannot be read" in warning


# --- planning and importing ---------------------------------------------------------------


def test_the_directory_s_answer_wins_over_the_store_s(tmp_path: Path) -> None:
    a, b = CASES
    store(tmp_path, [generation(a, answer("stored")), generation(b, answer())], [])
    own = {("base", a.id): {"message": answer("own"), "format_valid": True}}
    taken = plan_target(load(tmp_path, own), "base", CASES, KEYS, JUDGE, own, {})
    assert list(taken.generations) == [b.id]
    assert taken.oldest_sampled_at == "2026-10-12T09:14:03+00:00"
    assert taken.sources == ["evaluate ft-a"]


def test_a_pair_with_an_ok_result_takes_nothing(tmp_path: Path) -> None:
    a, b = CASES
    store(tmp_path, [generation(a, answer()), generation(b, answer())], [])
    results = {("base", a.id): {"status": "ok"}, ("base", b.id): {"status": "error"}}
    taken = plan_target(load(tmp_path), "base", CASES, KEYS, JUDGE, {}, results)
    assert list(taken.generations) == [b.id]


def test_a_malformed_answer_takes_no_verdict(tmp_path: Path) -> None:
    a = CASES[0]
    store(tmp_path, [generation(a, answer(), format_valid=False)], [verdict(a, answer())])
    taken = plan_target(load(tmp_path), "base", CASES, KEYS, JUDGE, {}, {})
    assert taken.counts() == {"generations": 1, "verdicts": 0}


def test_import_copies_the_answers_and_verdicts_and_then_nothing_is_left(tmp_path: Path) -> None:
    a, b = CASES
    store(
        tmp_path,
        [generation(a, CALL), generation(b, answer())],
        [verdict(a, CALL), verdict(b, answer(), passed=False)],
    )
    index = load(tmp_path)
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    taken = plan_target(index, "base", CASES, KEYS, JUDGE, {}, {})
    assert import_target(eval_dir, taken, CASES) == {"generations": 2, "verdicts": 2}
    generations, results = eval_dir.read_generations(), eval_dir.read_results()
    copied = generations[("base", a.id)]
    assert copied["reused_from"] == SOURCE
    assert copied["sampled_at"] == "2026-10-12T09:14:03+00:00"
    assert "key" not in copied and "source" not in copied
    result = results[("base", a.id)]
    assert result["reused"] is True
    assert result["judged_at"] == "2026-10-12T09:15:00+00:00"
    assert result["judgment"]["passed"] is True
    # The metrics are computed here: the stored call matches the reference.
    assert result["metrics"]["tool_name_match"] is True
    assert results[("base", b.id)]["metrics"]["tool_decision_match"] is False
    again = plan_target(index, "base", CASES, KEYS, JUDGE, generations, results)
    assert again.counts() == {"generations": 0, "verdicts": 0}


def test_import_skips_a_pair_the_directory_filled_meanwhile(tmp_path: Path) -> None:
    a = CASES[0]
    store(tmp_path, [generation(a, answer("stored"))], [verdict(a, answer("stored"))])
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    taken = plan_target(load(tmp_path), "base", CASES, KEYS, JUDGE, {}, {})
    own = {"target": "base", "case_id": a.id, "message": answer("own"), "format_valid": True}
    eval_dir.append_generation(own)
    assert import_target(eval_dir, taken, CASES) == {"generations": 0, "verdicts": 0}
    assert eval_dir.read_results() == {}


# --- exporting ----------------------------------------------------------------------------


def judged(
    eval_dir: EvalDir, target: str, case: ReplayCase, message: dict[str, Any], **judgment: Any
) -> None:
    eval_dir.append_generation(
        {
            "target": target,
            "case_id": case.id,
            "message": message,
            "format_valid": True,
            "sampled_at": "2026-10-13T10:00:00+00:00",
        }
    )
    eval_dir.append_result(
        {
            "target": target,
            "case_id": case.id,
            "status": "ok",
            "judgment": {"passed": True, "reason": "ok", "raw": "{}", "error": None, **judgment},
            "judged_at": "2026-10-13T10:00:01+00:00",
        }
    )


def made(dataset: Path) -> EvalDir:
    """An evaluation directory stamped with :data:`INPUTS`, as a run makes one."""
    eval_dir = EvalDir(dataset / "eval")
    eval_dir.create()
    stamp(eval_dir, INPUTS)
    return eval_dir


def stored(dataset: Path, name: str) -> list[dict[str, Any]]:
    path = store_path(dataset) / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_export_writes_the_target_s_own_answers_and_real_verdicts(tmp_path: Path) -> None:
    a, b = CASES
    eval_dir = made(tmp_path)
    judged(eval_dir, "base", a, answer("a"))
    judged(eval_dir, "base", b, answer("b"), raw=None, error="unparseable")
    judged(eval_dir, "tuned", a, answer("tuned"))
    assert export_target(tmp_path, eval_dir, "base", CASES, INPUTS, JUDGE, SOURCE) is None
    answers = stored(tmp_path, "generations.jsonl")
    assert [(line["key"], line["message"]) for line in answers] == [
        (KEYS[a.id], answer("a")),
        (KEYS[b.id], answer("b")),
    ]
    assert answers[0]["source"] == SOURCE
    assert answers[0]["sampled_at"] == "2026-10-13T10:00:00+00:00"
    (line,) = stored(tmp_path, "verdicts.jsonl")
    assert line["key"] == verdict_key(JUDGE, a, answer("a"))
    assert line["judged_at"] == "2026-10-13T10:00:01+00:00"
    assert oct(store_path(tmp_path).stat().st_mode & 0o777) == "0o700"
    assert oct((store_path(tmp_path) / "verdicts.jsonl").stat().st_mode & 0o777) == "0o600"


def test_export_skips_reused_answers_and_keys_the_store_has(tmp_path: Path) -> None:
    a, b = CASES
    store(tmp_path, [generation(a, answer("first"))], [verdict(a, answer("first"))])
    eval_dir = made(tmp_path)
    judged(eval_dir, "base", a, answer("first"))
    eval_dir.append_generation(
        {
            "target": "base",
            "case_id": b.id,
            "message": answer(),
            "format_valid": True,
            "reused_from": SOURCE,
        }
    )
    assert export_target(tmp_path, eval_dir, "base", CASES, INPUTS, JUDGE, SOURCE) is None
    assert len(stored(tmp_path, "generations.jsonl")) == 1
    assert len(stored(tmp_path, "verdicts.jsonl")) == 1


def test_export_to_a_busy_store_writes_nothing_and_warns(tmp_path: Path) -> None:
    eval_dir = made(tmp_path)
    judged(eval_dir, "base", CASES[0], answer())
    store_path(tmp_path).mkdir()
    with DirectoryLock(store_path(tmp_path)):
        warning = export_target(tmp_path, eval_dir, "base", CASES, INPUTS, JUDGE, SOURCE)
    assert warning == (
        "base results were not shared: another pup-finetune process is using "
        f"{store_path(tmp_path)}; the next run of this command shares them"
    )
    assert stored(tmp_path, "generations.jsonl") == []


def test_a_directory_stamped_with_other_inputs_shares_nothing(tmp_path: Path) -> None:
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    stamp(eval_dir, key_inputs("fake", FAKE_PROFILE, {**RENDERING, "renderer_version": "0"}, 64))
    judged(eval_dir, "base", CASES[0], answer())
    assert export_target(tmp_path, eval_dir, "base", CASES, INPUTS, JUDGE, SOURCE) is None
    assert not store_path(tmp_path).exists()


def test_the_stamp_is_written_only_while_the_directory_holds_no_answer(tmp_path: Path) -> None:
    eval_dir = made(tmp_path)
    judged(eval_dir, "base", CASES[0], answer())
    stamp(eval_dir, {**INPUTS, "v": "0"})
    assert json.loads((eval_dir.path / "reuse.json").read_text()) == INPUTS


@pytest.mark.parametrize(
    ("legacy", "version", "shared"), [(False, "1", 0), (True, "1", 1), (True, "2", 0)]
)
def test_an_unstamped_directory_shares_only_as_a_legacy_one_at_version_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, version: str, shared: int
) -> None:
    eval_dir = EvalDir(tmp_path / "eval")
    eval_dir.create()
    judged(eval_dir, "base", CASES[0], answer())
    monkeypatch.setattr(reuse, "REUSE_VERSION", version)
    inputs = key_inputs("fake", FAKE_PROFILE, RENDERING, 64)
    assert (
        export_target(tmp_path, eval_dir, "base", CASES, inputs, JUDGE, SOURCE, legacy=legacy)
        is None
    )
    assert len(stored(tmp_path, "generations.jsonl")) == shared
