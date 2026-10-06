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
    SplitParams,
    assign_split,
    check_isolation,
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


def test_split_warnings_for_healthy_counts_are_empty() -> None:
    assert split_warnings({"train": 3, "validation": 1, "test": 1}) == []


def test_split_warnings_flag_empty_splits_and_few_train_rows() -> None:
    warnings = split_warnings({"train": 2, "validation": 0, "test": 0})
    assert len(warnings) == 3
    assert "only 2 training rows" in warnings[0]
    assert "validation split is empty" in warnings[1]
    assert "test split is empty" in warnings[2]
