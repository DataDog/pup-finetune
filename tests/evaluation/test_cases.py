# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings
from pup_finetune.data.labels import Label
from pup_finetune.data.schema import TrainingRow, validate_row
from pup_finetune.data.split import SplitParams
from pup_finetune.evaluation.cases import (
    ReplayCase,
    build_cases,
    case_indices,
    cases_from_dataset,
    row_cases,
    select_spread,
)
from pup_finetune.storage import DirectoryLock, canonical_json

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a term up.",
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        },
    }
]


def call(call_id: str, term: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"term": term})},
    }


def agent_row(row_id: str, turns: int, *, trace_id: str | None = None) -> TrainingRow:
    """A row with ``turns`` tool-calling assistant turns, then a final text answer."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You look things up."},
        {"role": "user", "content": "Find the terms."},
    ]
    for turn in range(turns):
        call_id = f"{row_id}-c{turn}"
        messages.append(
            {"role": "assistant", "content": "", "tool_calls": [call(call_id, f"t{turn}")]}
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": f"result {turn}"})
    messages.append({"role": "assistant", "content": "Done."})
    data: dict[str, Any] = {"id": row_id, "messages": messages, "tools": TOOLS}
    if trace_id is not None:
        data["source"] = {"trace_id": trace_id}
    return validate_row(data)


# --- one row ------------------------------------------------------------------------------


def test_each_assistant_action_is_a_case_with_the_prefix_before_it() -> None:
    row = agent_row("r1", 2, trace_id="trace-1")
    cases = row_cases(row)
    assert [case.message_index for case in cases] == [2, 4, 6]
    first = cases[0]
    assert first.id == "r1:2"
    assert first.row_id == "r1"
    assert first.trace_id == "trace-1"
    assert first.split == "test"
    assert [m["role"] for m in first.prefix] == ["system", "user"]
    assert first.reference["tool_calls"][0]["function"]["name"] == "lookup"
    assert list(first.tools) == TOOLS
    last = cases[-1]
    assert last.reference == {"role": "assistant", "content": "Done."}
    assert len(last.prefix) == 6


def test_a_row_without_a_source_trace_has_no_trace_id() -> None:
    assert row_cases(agent_row("x", 1))[0].trace_id is None


def test_future_tool_results_are_the_tool_messages_right_after_the_reference() -> None:
    row = validate_row(
        {
            "id": "par",
            "messages": [
                {"role": "user", "content": "Look up a and b."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call("a", "a"), call("b", "b")],
                },
                {"role": "tool", "tool_call_id": "a", "content": "A"},
                {"role": "tool", "tool_call_id": "b", "content": "B"},
                {"role": "user", "content": "And?"},
                {"role": "assistant", "content": "", "tool_calls": [call("c", "c")]},
                {"role": "tool", "tool_call_id": "c", "content": "C"},
                {"role": "assistant", "content": "A, B and C."},
            ],
            "tools": TOOLS,
        }
    )
    cases = row_cases(row)
    assert [case.message_index for case in cases] == [1, 5, 7]
    assert [m["content"] for m in cases[0].future_tool_results] == ["A", "B"]
    assert [m["content"] for m in cases[1].future_tool_results] == ["C"]
    assert cases[2].future_tool_results == ()
    # The model never sees what follows the reference.
    assert all(m["content"] != "A" for m in cases[0].prefix)


def test_turns_without_a_visible_action_are_not_cases() -> None:
    row = validate_row(
        {
            "id": "quiet",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "   ", "reasoning_content": "thinking only"},
                {"role": "user", "content": "Hello?"},
                {"role": "assistant", "content": "Hello."},
            ],
        }
    )
    assert case_indices(row) == [3]


def test_untrained_turns_are_left_out_unless_asked_for() -> None:
    row = validate_row(
        {
            "id": "w",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "skip me", "weight": 0},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "keep me", "weight": 1},
            ],
        }
    )
    assert [case.message_index for case in row_cases(row)] == [3]
    included = row_cases(row, include_untrained=True)
    assert [case.message_index for case in included] == [1, 3]
    # Weight is training metadata: it is never sent with the prefix or reference.
    assert all("weight" not in m for case in included for m in (*case.prefix, case.reference))


def test_reasoning_stays_on_the_reference_for_display() -> None:
    row = validate_row(
        {
            "id": "think",
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", "reasoning_content": "because"},
            ],
        }
    )
    (case,) = row_cases(row)
    assert case.reference["reasoning_content"] == "because"


def test_a_row_with_no_eligible_turn_has_no_cases() -> None:
    row = validate_row(
        {
            "id": "none",
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", "weight": 0},
            ],
        }
    )
    assert row_cases(row, max_cases_per_trajectory=1) == []


# --- caps ---------------------------------------------------------------------------------


@pytest.mark.parametrize("count", range(1, 30))
@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_select_spread_is_evenly_spaced_and_keeps_the_last(count: int, cap: int) -> None:
    picks = select_spread(count, cap, key="row", seed=7)
    assert picks == sorted(set(picks))
    assert len(picks) == min(count, cap)
    assert picks[-1] == count - 1
    if count > cap:
        # One pick per equal bucket of the positions.
        for bucket, pick in enumerate(picks):
            assert bucket * count // cap <= pick < (bucket + 1) * count // cap + 1


def test_select_spread_is_deterministic_and_varies_with_the_key_and_seed() -> None:
    assert select_spread(40, 4, key="a", seed=1) == select_spread(40, 4, key="a", seed=1)
    variants = {tuple(select_spread(40, 4, key=k, seed=s)) for k in "abcdef" for s in (1, 2)}
    assert len(variants) > 1
    assert all(variant[-1] == 39 for variant in variants)


def test_select_spread_is_the_same_in_every_process() -> None:
    # Base and tuned evaluations run in separate processes and must pick the same cases, so
    # the picks are pinned rather than only compared within one run.
    assert select_spread(40, 4, key="a", seed=1) == [2, 12, 22, 39]
    capped = build_cases([agent_row("r", 9)], max_cases_per_trajectory=3, seed=42)
    assert [case.id for case in capped] == ["r:6", "r:12", "r:20"]


def test_select_spread_rejects_a_cap_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_spread(3, 0, key="a", seed=0)


def test_a_cap_keeps_the_final_turn() -> None:
    row = agent_row("long", 9)
    every = case_indices(row)
    capped = row_cases(row, max_cases_per_trajectory=3, seed=42)
    assert len(capped) == 3
    assert capped[-1].message_index == every[-1]
    assert [c.message_index for c in capped] == sorted(c.message_index for c in capped)
    only = row_cases(row, max_cases_per_trajectory=1, seed=42)
    assert [c.message_index for c in only] == [every[-1]]


def test_a_cap_keeps_the_final_eligible_turn_when_the_last_one_is_untrained() -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Go."}]
    for turn in range(3):
        call_id = f"c{turn}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [call(call_id, "t")]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": "r"})
    messages.append({"role": "assistant", "content": "Done.", "weight": 0})
    row = validate_row({"id": "tail", "messages": messages, "tools": TOOLS})
    only = row_cases(row, max_cases_per_trajectory=1)
    assert [case.message_index for case in only] == [5]
    untrained = row_cases(row, max_cases_per_trajectory=1, include_untrained=True)
    assert [case.message_index for case in untrained] == [7]


def test_a_cap_above_the_turn_count_keeps_every_turn() -> None:
    row = agent_row("short", 1)
    assert len(row_cases(row, max_cases_per_trajectory=10)) == len(case_indices(row))


# --- many rows ----------------------------------------------------------------------------


def test_build_cases_orders_by_row_id_then_message_index() -> None:
    rows = [agent_row("b", 1), agent_row("a", 2), agent_row("c", 0)]
    cases = build_cases(rows, max_cases_per_trajectory=None, seed=0)
    assert [case.id for case in cases] == ["a:2", "a:4", "a:6", "b:2", "b:4", "c:2"]


def test_build_cases_is_deterministic() -> None:
    rows = [agent_row(f"r{i}", 6) for i in range(5)]
    first = build_cases(rows, max_cases_per_trajectory=2, seed=3)
    again = build_cases(list(reversed(rows)), max_cases_per_trajectory=2, seed=3)
    assert first == again
    assert len(first) == 10


def test_build_cases_rejects_repeated_row_ids_and_bad_caps() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_cases([agent_row("a", 1), agent_row("a", 1)], max_cases_per_trajectory=None, seed=0)
    with pytest.raises(ValueError, match="at least 1"):
        build_cases([agent_row("a", 1)], max_cases_per_trajectory=0, seed=0)
    with pytest.raises(ValueError, match="unknown split"):
        build_cases([], max_cases_per_trajectory=None, seed=0, split="holdout")  # type: ignore[arg-type]


def test_a_case_is_json_safe() -> None:
    (case, *_) = build_cases([agent_row("a", 1)], max_cases_per_trajectory=None, seed=0)
    data = case.to_json()
    assert json.loads(canonical_json(data)) == data
    assert data["id"] == "a:2"
    assert isinstance(data["prefix"], list)


def test_a_case_is_frozen() -> None:
    (case, *_) = row_cases(agent_row("a", 1))
    with pytest.raises(AttributeError):
        case.id = "other"  # type: ignore[misc]
    assert isinstance(case, ReplayCase)


# --- from a dataset -----------------------------------------------------------------------


def test_cases_from_dataset_use_kept_test_rows_only(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    rows = [agent_row(f"row-{i:02d}", i % 3, trace_id=f"trace-{i:02d}") for i in range(30)]
    dataset.add(rows, settings=DatasetSettings("import", SplitParams(5, 0.2, 0.4)))
    test_rows = dataset.rows_for_split("test")
    assert 1 < len(test_rows) < len(rows)

    cases = cases_from_dataset(dataset, max_cases_per_trajectory=2, seed=1)
    assert {case.row_id for case in cases} == {row.id for row in test_rows}
    assert all(case.split == "test" for case in cases)

    dropped = test_rows[0].id
    with DirectoryLock(dataset.path):
        dataset.append_labels([Label(id=dropped, keep=False, reason="off-topic")])
    after = cases_from_dataset(dataset, max_cases_per_trajectory=2, seed=1)
    assert dropped not in {case.row_id for case in after}
    assert after == [case for case in cases if case.row_id != dropped]

    validation = cases_from_dataset(dataset, split="validation")
    assert {case.row_id for case in validation} == {
        row.id for row in dataset.rows_for_split("validation")
    }
    assert all(case.split == "validation" for case in validation)
