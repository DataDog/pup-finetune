# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import hashlib
import json
import math
import stat
from collections import Counter
from pathlib import Path

import pytest

from pup_finetune.data.schema import TrainingRow, validate_row
from pup_finetune.data.split import (
    SPLITS,
    SplitAssignments,
    SplitName,
    SplitParams,
    assign_split,
    check_isolation,
    skew_warnings,
    split_counts,
    split_position,
    split_warnings,
)
from pup_finetune.errors import StateError, UsageError

KEYS = [f"synthetic-trace-{i:04d}" for i in range(2000)]


def make_row(row_id: str, text: str = "Hello", *, trace_id: str | None = None) -> TrainingRow:
    data: dict[str, object] = {
        "id": row_id,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"Reply to {text}"},
        ],
    }
    if trace_id is not None:
        data["source"] = {"trace_id": trace_id}
    return validate_row(data)


# --- assign_split -----------------------------------------------------------------------


def test_split_position_is_the_first_64_bits_of_the_sha256() -> None:
    digest = hashlib.sha256(b"42:synthetic-trace-0001").hexdigest()
    assert split_position("synthetic-trace-0001", seed=42) == int(digest[:16], 16) / 2**64


def test_assign_split_is_deterministic() -> None:
    first = [assign_split(k, seed=42, validation_fraction=0.1, test_fraction=0.1) for k in KEYS]
    again = [assign_split(k, seed=42, validation_fraction=0.1, test_fraction=0.1) for k in KEYS]
    assert first == again


def test_assign_split_depends_on_the_seed() -> None:
    a = [assign_split(k, seed=1, validation_fraction=0.3, test_fraction=0.3) for k in KEYS[:50]]
    b = [assign_split(k, seed=2, validation_fraction=0.3, test_fraction=0.3) for k in KEYS[:50]]
    assert a != b


def test_assign_split_roughly_follows_the_fractions() -> None:
    counts = Counter(
        assign_split(k, seed=42, validation_fraction=0.1, test_fraction=0.2) for k in KEYS
    )
    assert counts["test"] / len(KEYS) == pytest.approx(0.2, abs=0.03)
    assert counts["validation"] / len(KEYS) == pytest.approx(0.1, abs=0.03)
    assert counts["train"] / len(KEYS) == pytest.approx(0.7, abs=0.03)


def test_assign_split_thresholds_test_first_then_validation() -> None:
    for key in KEYS[:200]:
        position = split_position(key, seed=7)
        split = assign_split(key, seed=7, validation_fraction=0.25, test_fraction=0.25)
        expected = "test" if position < 0.25 else "validation" if position < 0.5 else "train"
        assert split == expected


def test_zero_fractions_put_everything_in_train() -> None:
    splits = {assign_split(k, seed=42, validation_fraction=0, test_fraction=0) for k in KEYS}
    assert splits == {"train"}


def test_zero_validation_fraction_assigns_no_validation() -> None:
    splits = {assign_split(k, seed=42, validation_fraction=0, test_fraction=0.5) for k in KEYS}
    assert splits == {"train", "test"}


def test_a_fraction_boundary_is_exclusive() -> None:
    key = KEYS[0]
    position = split_position(key, seed=42)
    # Exactly at the threshold the key is past it; just above, it is inside.
    assert assign_split(key, seed=42, validation_fraction=0, test_fraction=position) == "train"
    above = math.nextafter(position, 1)
    assert assign_split(key, seed=42, validation_fraction=0, test_fraction=above) == "test"
    assert assign_split(key, seed=42, validation_fraction=above, test_fraction=0) == "validation"


# --- SplitParams ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("validation", "test"),
    [(-0.1, 0.1), (0.1, -0.1), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.7, 0.4), (math.nan, 0.1)],
)
def test_split_params_reject_bad_fractions(validation: float, test: float) -> None:
    with pytest.raises(UsageError):
        SplitParams(42, validation, test)


@pytest.mark.parametrize(("validation", "test"), [(0.0, 0.0), (0.0, 0.99), (0.49, 0.5)])
def test_split_params_accept_edge_fractions(validation: float, test: float) -> None:
    assert SplitParams(42, validation, test).test_fraction == test


def test_split_params_from_json_rejects_malformed_data() -> None:
    with pytest.raises(StateError, match="invalid split settings"):
        SplitParams.from_json({"seed": "42"}, where="x")
    with pytest.raises(StateError, match="invalid split settings"):
        SplitParams.from_json(
            {"seed": 1, "validation_fraction": 0.6, "test_fraction": 0.6}, where="x"
        )


# --- SplitAssignments -------------------------------------------------------------------


def test_assignments_are_stable_as_keys_are_added(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    params = SplitParams(42, 0.2, 0.2)
    first = SplitAssignments.load(path, params)
    before = {key: first.assign(key) for key in KEYS[:100]}
    first.save(path)

    second = SplitAssignments.load(path, params)
    for key in KEYS[100:300]:
        second.assign(key)
    assert {key: second.get(key) for key in KEYS[:100]} == before
    assert all(second.get(key) == params.assign(key) for key in KEYS[100:300])


def test_a_recorded_assignment_never_changes(tmp_path: Path) -> None:
    params = SplitParams(42, 0.1, 0.1)
    key = KEYS[0]
    natural = params.assign(key)
    forced = next(split for split in SPLITS if split != natural)
    path = tmp_path / "splits.json"
    SplitAssignments(params, {key: forced}).save(path)
    assert SplitAssignments.load(path, params).assign(key) == forced


def test_splits_json_shape_and_permissions(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    assignments = SplitAssignments(SplitParams(7, 0.1, 0.2))
    assignments.assign("synthetic-trace-0001")
    assignments.save(path)
    data = json.loads(path.read_text())
    assert data == {
        "seed": 7,
        "validation_fraction": 0.1,
        "test_fraction": 0.2,
        "assignments": {"synthetic-trace-0001": assignments.get("synthetic-trace-0001")},
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_load_without_a_file_starts_empty(tmp_path: Path) -> None:
    loaded = SplitAssignments.load(tmp_path / "splits.json", SplitParams())
    assert loaded.assignments == {}


def test_load_rejects_different_params(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    SplitAssignments(SplitParams(42, 0.1, 0.1)).save(path)
    with pytest.raises(StateError, match="different split settings"):
        SplitAssignments.load(path, SplitParams(43, 0.1, 0.1))


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"seed": 1, "validation_fraction": 0.1, "test_fraction": 0.1, "assignments": []},
        {"seed": 1, "validation_fraction": 0.1, "test_fraction": 0.1, "assignments": {"k": "x"}},
    ],
)
def test_load_rejects_malformed_files(tmp_path: Path, data: object) -> None:
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(data))
    with pytest.raises(StateError):
        SplitAssignments.load(path, SplitParams(1, 0.1, 0.1))


# --- check_isolation and warnings -------------------------------------------------------


def test_check_isolation_passes_for_distinct_conversations() -> None:
    check_isolation(
        {"train": [make_row("a", "one")], "validation": [make_row("b", "two")], "test": []}
    )


def test_check_isolation_allows_repeats_within_one_split() -> None:
    check_isolation({"train": [make_row("a"), make_row("b")]})


def test_check_isolation_rejects_a_conversation_in_two_splits() -> None:
    train = make_row("row-a", "same", trace_id="synthetic-trace-1")
    test = make_row("row-b", "same", trace_id="synthetic-trace-2")
    with pytest.raises(StateError, match=r"'row-a' \(train\) and 'row-b' \(test\)"):
        check_isolation({"train": [train], "validation": [], "test": [test]})


def tool_row(row_id: str, call_id: str, trace_id: str) -> TrainingRow:
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": "Weather in Northvale?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "N"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "14°C", "tool_call_id": call_id},
                {"role": "assistant", "content": "It's 14°C."},
            ],
            "source": {"trace_id": trace_id},
        }
    )


def test_check_isolation_rejects_a_conversation_that_differs_only_in_tool_call_ids() -> None:
    train = tool_row("row-a", "call_1f2e", "synthetic-trace-1")
    validation = tool_row("row-b", "call_8c4d", "synthetic-trace-2")
    with pytest.raises(StateError, match=r"'row-a' \(train\) and 'row-b' \(validation\)"):
        check_isolation({"train": [train], "validation": [validation], "test": []})


def test_split_warnings_for_healthy_counts_are_empty() -> None:
    assert split_warnings({"train": 3, "validation": 1, "test": 1}) == []


def test_split_warnings_flag_empty_splits_and_few_train_rows() -> None:
    warnings = split_warnings({"train": 2, "validation": 0, "test": 0})
    assert len(warnings) == 3
    assert "only 2 training rows" in warnings[0]
    assert "validation split is empty" in warnings[1]
    assert "test split is empty" in warnings[2]


def test_split_counts_follow_the_params_by_trace() -> None:
    rows = [make_row(f"row-{i}", f"Q{i}", trace_id=KEYS[i // 2]) for i in range(40)]
    counts = split_counts(rows, SplitParams())
    assert sum(counts.values()) == 40
    assert set(counts) == set(SPLITS)
    assert all(count % 2 == 0 for count in counts.values())  # two rows per trace
    assert split_counts(rows, SplitParams(1, 0.0, 0.0)) == {"train": 40, "validation": 0, "test": 0}


# --- skew -----------------------------------------------------------------------------------


def keys(
    train: int, validation: int, test: int, *, rows_per_trace: int = 1
) -> dict[SplitName, list[str]]:
    """Split keys for that many traces per split, each with ``rows_per_trace`` rows."""
    counts: dict[SplitName, int] = {"train": train, "validation": validation, "test": test}
    return {
        split: [f"{split}-{n}" for n in range(count) for _ in range(rows_per_trace)]
        for split, count in counts.items()
    }


def test_a_small_skewed_split_is_warned_about_without_suggesting_the_seed() -> None:
    by_split = keys(7, 4, 2)
    by_split["validation"].append("validation-0")  # the fixture's 7 / 5 / 2 rows
    (warning,) = skew_warnings(by_split, SplitParams(), more="pull more rows")
    assert warning == (
        "validation has 5 of 14 rows (36%, 4 of 13 traces) though --validation-fraction is "
        "0.1: splits are assigned per trace, so small datasets vary; pull more rows"
    )
    assert "seed" not in warning


@pytest.mark.parametrize(
    ("train", "validation", "test", "skewed"),
    [
        (16, 2, 2, []),  # on target
        (17, 1, 2, []),  # 5 points under, and not over twice the fraction
        (15, 4, 1, ["validation"]),  # 20%: 10 points over
        (6, 1, 2, ["test"]),  # 22%: over twice the fraction, though 12 points off
        (14, 5, 1, ["validation"]),  # 25%
        (20, 0, 0, []),  # empty splits have their own warnings
    ],
)
def test_skew_is_more_than_twice_the_fraction_or_ten_points_off(
    train: int, validation: int, test: int, skewed: list[str]
) -> None:
    warnings = skew_warnings(keys(train, validation, test), SplitParams(), more="x")
    assert [w.split()[0] for w in warnings] == skewed


def test_more_than_twice_a_small_fraction_is_skewed_within_ten_points() -> None:
    params = SplitParams(validation_fraction=0.02, test_fraction=0.1)
    warnings = skew_warnings(keys(41, 3, 5), params, more="x")  # 6% against 2%
    assert [w.split()[0] for w in warnings] == ["validation"]


def test_ten_points_under_a_larger_fraction_is_skewed() -> None:
    params = SplitParams(validation_fraction=0.3, test_fraction=0.1)
    warnings = skew_warnings(keys(17, 2, 1), params, more="x")  # 10% against 30%
    assert [w.split()[0] for w in warnings] == ["validation"]


def test_skew_is_not_warned_about_once_there_are_enough_traces() -> None:
    assert skew_warnings(keys(29, 15, 5), SplitParams(), more="x") != []
    assert skew_warnings(keys(29, 15, 5, rows_per_trace=3), SplitParams(), more="x") != []
    assert skew_warnings(keys(30, 15, 5), SplitParams(), more="x") == []  # 50 traces
