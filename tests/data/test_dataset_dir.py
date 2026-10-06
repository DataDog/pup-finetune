# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings, read_import_file
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.jsonl import read_rows, to_provider_dict
from pup_finetune.data.schema import RowErrorCode, TrainingRow, split_key, validate_row
from pup_finetune.data.split import SPLITS, SplitParams
from pup_finetune.errors import StateError, UsageError
from pup_finetune.storage import DirectoryLock

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"
INVALID = ROWS_DIR / "invalid.jsonl"
INVALID_EXPECTED: list[dict[str, Any]] = json.loads(
    (ROWS_DIR / "invalid.expected.json").read_text()
)

IMPORT = DatasetSettings("import", SplitParams(42, 0.1, 0.1))


@pytest.fixture
def loose_umask() -> Iterator[None]:
    previous = os.umask(0o002)
    yield
    os.umask(previous)


@pytest.fixture
def fixture_rows() -> list[TrainingRow]:
    return list(read_rows(TOOL_USE))


def make_row(row_id: str, text: str, *, trace_id: str | None = None) -> TrainingRow:
    data: dict[str, Any] = {
        "id": row_id,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"Reply to {text}"},
        ],
    }
    if trace_id is not None:
        data["source"] = {"trace_id": trace_id, "span_id": f"{trace_id}-span"}
    return validate_row(data)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


# --- add --------------------------------------------------------------------------------


def test_add_writes_every_file(tmp_path: Path, fixture_rows: list[TrainingRow]) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    result = dataset.add(fixture_rows, settings=IMPORT)
    assert (result.added, result.unchanged, result.excluded) == (len(fixture_rows), 0, 0)
    names = {p.name for p in dataset.path.iterdir()}
    assert {"settings.json", "rows.jsonl", "excluded.jsonl", "splits.json", "summary.json"} <= (
        names
    )
    assert "labels.jsonl" not in names
    assert dataset.rows() == fixture_rows
    assert json.loads(dataset.settings_path.read_text()) == {
        "source": {"kind": "import"},
        "split": {"seed": 42, "validation_fraction": 0.1, "test_fraction": 0.1},
    }
    assert json.loads(dataset.summary_path.read_text()) == result.summary


def test_directory_is_0700_and_files_are_0600(
    tmp_path: Path, loose_umask: object, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "parent" / "ds")
    dataset.add(fixture_rows, [Exclusion(RowErrorCode.INVALID_ROW, "bad")], settings=IMPORT)
    dataset.write_summary()
    dataset.export_split_files(tmp_path / "parent" / "ds" / "export")
    for directory in (dataset.path, dataset.path.parent, dataset.path / "export"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    files = [p for p in dataset.path.rglob("*") if p.is_file()]
    assert len(files) >= 8
    for path in files:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_an_existing_empty_directory_is_made_private(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    target = tmp_path / "ds"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    DatasetDir(target).add(fixture_rows, settings=IMPORT)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_a_non_empty_directory_that_is_not_a_dataset_is_refused(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    (tmp_path / "notes.txt").write_text("keep me")
    with pytest.raises(UsageError, match="not a dataset directory"):
        DatasetDir(tmp_path).add(fixture_rows, settings=IMPORT)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_a_file_path_is_refused(tmp_path: Path, fixture_rows: list[TrainingRow]) -> None:
    target = tmp_path / "file"
    target.write_text("")
    with pytest.raises(UsageError, match="not a directory"):
        DatasetDir(target).add(fixture_rows, settings=IMPORT)


def test_adding_the_same_rows_again_is_idempotent(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    exclusions = [Exclusion(RowErrorCode.INVALID_ROW, "rows.jsonl:3: bad", row_id="x")]
    dataset.add(fixture_rows, exclusions, settings=IMPORT)
    before = snapshot(dataset.path)
    result = dataset.add(fixture_rows, exclusions, settings=IMPORT)
    assert (result.added, result.unchanged, result.excluded) == (0, len(fixture_rows), 1)
    assert snapshot(dataset.path) == before


def test_new_rows_are_appended_and_old_assignments_kept(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows[:6], settings=IMPORT)
    before = dict(dataset.splits().assignments)
    result = dataset.add(fixture_rows, settings=IMPORT)
    assert (result.added, result.unchanged) == (len(fixture_rows) - 6, 6)
    assert [row.id for row in dataset.rows()] == [row.id for row in fixture_rows]
    after = dataset.splits().assignments
    assert {key: after[key] for key in before} == before
    assert set(after) == {split_key(row) for row in fixture_rows}


def test_a_known_id_with_different_content_is_an_error(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows, settings=IMPORT)
    before = snapshot(dataset.path)
    changed = make_row(fixture_rows[0].id, "something else")
    with pytest.raises(StateError, match="already in the dataset with different content"):
        dataset.add([make_row("row-new", "fresh"), changed], settings=IMPORT)
    assert snapshot(dataset.path) == before


def test_a_conflicting_id_within_one_add_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    with pytest.raises(StateError, match="'a' appears more than once in the input") as info:
        DatasetDir(target).add([make_row("a", "one"), make_row("a", "two")], settings=IMPORT)
    assert info.value.hint == "give each row in the input file a unique id"
    assert not DatasetDir(target).is_dataset()


def test_a_row_repeated_within_one_add_is_one_row(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    row = make_row("a", "one")
    result = dataset.add([row, row, make_row("b", "two")], settings=IMPORT)
    assert (result.added, result.unchanged, result.excluded) == (2, 0, 0)
    assert [r.id for r in dataset.rows()] == ["a", "b"]
    assert dataset.excluded() == []
    assert (result.summary["total"], result.summary["accepted"]) == (2, 2)


def test_changed_settings_are_rejected(tmp_path: Path, fixture_rows: list[TrainingRow]) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows[:3], settings=IMPORT)
    before = snapshot(dataset.path)
    for settings in (
        DatasetSettings("import", SplitParams(43, 0.1, 0.1)),
        DatasetSettings("import", SplitParams(42, 0.2, 0.1)),
        DatasetSettings("datadog", SplitParams(42, 0.1, 0.1)),
    ):
        with pytest.raises(StateError, match="settings changed"):
            dataset.add(fixture_rows, settings=settings)
    assert snapshot(dataset.path) == before


def test_a_split_change_names_the_stored_values(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows[:3], settings=DatasetSettings("import", SplitParams(7, 0.2, 0.05)))
    with pytest.raises(StateError, match="split settings changed") as info:
        dataset.add(fixture_rows, settings=IMPORT)
    assert info.value.hint == (
        "this dataset uses --seed 7 --validation-fraction 0.2 --test-fraction 0.05: "
        "pass the same values, or import into a new directory"
    )


def test_duplicate_conversations_are_excluded(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    original = make_row("row-a", "same", trace_id="synthetic-trace-1")
    copy = make_row("row-b", "same", trace_id="synthetic-trace-2")
    result = dataset.add([original, copy, make_row("row-c", "other")], settings=IMPORT)
    assert (result.added, result.excluded) == (2, 1)
    assert [row.id for row in dataset.rows()] == ["row-a", "row-c"]
    assert dataset.excluded() == [
        {
            "code": "duplicate_conversation",
            "detail": "same conversation as row 'row-a'",
            "row_id": "row-b",
            "trace_id": "synthetic-trace-2",
            "span_id": "synthetic-trace-2-span",
        }
    ]
    assert result.summary["excluded_by_code"] == {ExclusionCode.DUPLICATE_CONVERSATION: 1}
    # A later add repeating the conversation is excluded too, and recorded once.
    again = dataset.add([copy], settings=IMPORT)
    assert (again.added, again.unchanged, again.excluded) == (0, 0, 1)
    assert len(dataset.excluded()) == 1
    assert again.summary["excluded_by_code"] == {ExclusionCode.DUPLICATE_CONVERSATION: 1}
    assert (again.summary["total"], again.summary["excluded"]) == (3, 1)


def test_a_fixed_row_reimported_under_its_id_replaces_its_exclusion(tmp_path: Path) -> None:
    broken = json.loads(INVALID.read_text().splitlines()[0])
    assert broken["id"] == "bad-no-assistant"
    fixed = {
        **broken,
        "messages": [*broken["messages"], {"role": "assistant", "content": "Sunny."}],
    }
    broken_file, fixed_file = tmp_path / "broken.jsonl", tmp_path / "fixed.jsonl"
    broken_file.write_text(json.dumps(broken) + "\n")
    fixed_file.write_text(json.dumps(fixed) + "\n")
    dataset = DatasetDir(tmp_path / "ds")

    first = dataset.add(*read_import_file(broken_file), settings=IMPORT)
    assert first.summary["excluded_by_code"] == {"no_assistant_message": 1}

    summary = dataset.add(*read_import_file(fixed_file), settings=IMPORT).summary
    assert (summary["total"], summary["accepted"], summary["excluded"]) == (1, 1, 0)
    assert summary["excluded_by_code"] == {}
    assert dataset.excluded() == []
    assert [row.id for row in dataset.rows()] == ["bad-no-assistant"]

    # Importing the broken version again does not resurrect the exclusion.
    summary = dataset.add(*read_import_file(broken_file), settings=IMPORT).summary
    assert (summary["total"], summary["accepted"], summary["excluded"]) == (1, 1, 0)


def test_a_newer_exclusion_replaces_the_older_one_for_its_id(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(
        [make_row("a", "one")], [Exclusion(RowErrorCode.INVALID_ROW, "x", "bad")], settings=IMPORT
    )
    newer = Exclusion(RowErrorCode.SYSTEM_NOT_FIRST, "y", "bad")
    summary = dataset.add([], [newer], settings=IMPORT).summary
    assert dataset.excluded() == [newer.to_json()]
    assert summary["excluded_by_code"] == {"system_not_first": 1}
    assert summary["total"] == 2


def test_reimporting_invalid_rows_from_a_renamed_file_adds_no_exclusions(tmp_path: Path) -> None:
    renamed = tmp_path / "renamed.jsonl"
    renamed.write_text("\n" + INVALID.read_text())  # also shifts every line number
    dataset = DatasetDir(tmp_path / "ds")
    first = dataset.add(*read_import_file(INVALID), settings=IMPORT).summary
    second = dataset.add(*read_import_file(renamed), settings=IMPORT).summary
    assert second == first
    assert second["excluded"] == len(INVALID_EXPECTED)
    assert [record["detail"].split(":")[0] for record in dataset.excluded()] == [
        "renamed.jsonl"
    ] * len(INVALID_EXPECTED)


def test_a_stale_exclusion_of_an_accepted_row_is_not_counted(tmp_path: Path) -> None:
    # As if a write stopped after rows.jsonl and before excluded.jsonl.
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one")], settings=IMPORT)
    stale = Exclusion(RowErrorCode.INVALID_ROW, "old", "a").to_json()
    dataset.excluded_path.write_text(json.dumps(stale) + "\n")
    summary = dataset.summary()
    assert (summary["total"], summary["accepted"], summary["excluded"]) == (1, 1, 0)


def test_rows_of_one_trace_share_a_split(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    rows = [make_row(f"row-{i}", f"text {i}", trace_id=f"trace-{i // 3}") for i in range(60)]
    dataset.add(rows, settings=IMPORT)
    by_trace: dict[str, set[str]] = {}
    for split, split_rows in dataset.rows_by_split().items():
        for row in split_rows:
            by_trace.setdefault(split_key(row), set()).add(split)
    assert all(len(splits) == 1 for splits in by_trace.values())


def test_add_holds_the_directory_lock(tmp_path: Path, fixture_rows: list[TrainingRow]) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows[:1], settings=IMPORT)
    with DirectoryLock(dataset.path), pytest.raises(StateError, match="another pup-finetune"):
        dataset.add(fixture_rows, settings=IMPORT)


# --- reading ----------------------------------------------------------------------------


def test_summary_counts(tmp_path: Path, fixture_rows: list[TrainingRow]) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    exclusions = [
        Exclusion(RowErrorCode.INVALID_ROW, "a"),
        Exclusion(RowErrorCode.INVALID_ROW, "b"),
        Exclusion(RowErrorCode.SYSTEM_NOT_FIRST, "c"),
    ]
    summary = dataset.add(fixture_rows, exclusions, settings=IMPORT).summary
    split_counts = {split: len(dataset.rows_for_split(split)) for split in SPLITS}
    assert summary == {
        "total": len(fixture_rows) + 3,
        "accepted": len(fixture_rows),
        "excluded": 3,
        "excluded_by_code": {"invalid_row": 2, "system_not_first": 1},
        "splits": split_counts,
        "warnings": summary["warnings"],
    }
    assert sum(split_counts.values()) == len(fixture_rows)


def test_summary_warns_about_small_splits(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    params = DatasetSettings("import", SplitParams(42, 0.0, 0.0))
    summary = dataset.add([make_row("a", "one")], settings=params).summary
    assert summary["splits"] == {"train": 1, "validation": 0, "test": 0}
    assert len(summary["warnings"]) == 3


def test_kept_rows_are_all_rows_without_labels(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows, settings=IMPORT)
    assert not dataset.labels_path.exists()
    assert dataset.kept_rows() == dataset.rows()


def test_rows_for_split_partitions_the_rows(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows, settings=IMPORT)
    splits = dataset.splits()
    seen: list[str] = []
    for split in SPLITS:
        for row in dataset.rows_for_split(split):
            assert splits.get(split_key(row)) == split
            seen.append(row.id)
    assert sorted(seen) == sorted(row.id for row in fixture_rows)


def test_rows_for_split_rejects_unknown_splits(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one")], settings=IMPORT)
    with pytest.raises(ValueError, match="unknown split"):
        dataset.rows_for_split("holdout")  # type: ignore[arg-type]


def test_a_row_without_an_assignment_is_a_state_error(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows, settings=IMPORT)
    data = json.loads(dataset.splits_path.read_text())
    data["assignments"].pop(split_key(fixture_rows[0]))
    dataset.splits_path.write_text(json.dumps(data))
    with pytest.raises(StateError, match="has no split assignment"):
        dataset.rows_by_split()


def test_reading_a_missing_dataset_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="not a dataset directory"):
        DatasetDir(tmp_path / "missing").settings()


def test_isolation_is_checked_on_read(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one", trace_id="t1")], settings=IMPORT)
    # Simulate a hand-edited directory where a copy of the conversation sits in another split.
    copy = make_row("b", "one", trace_id="t2")
    with dataset.rows_path.open("a") as handle:
        handle.write(json.dumps(copy.model_dump(mode="json", exclude_none=True)) + "\n")
    data = json.loads(dataset.splits_path.read_text())
    data["assignments"]["t2"] = next(s for s in SPLITS if s != data["assignments"]["t1"])
    dataset.splits_path.write_text(json.dumps(data))
    with pytest.raises(StateError, match="same conversation in different splits"):
        dataset.export_split_files(tmp_path / "out")


# --- export -----------------------------------------------------------------------------


def test_export_split_files_writes_provider_form(
    tmp_path: Path, fixture_rows: list[TrainingRow]
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(fixture_rows, settings=IMPORT)
    paths = dataset.export_split_files(tmp_path / "out")
    assert set(paths) == set(SPLITS)
    for split, path in paths.items():
        assert path == tmp_path / "out" / f"{split}.jsonl"
        expected = [to_provider_dict(row) for row in dataset.rows_for_split(split)]
        assert jsonl(path) == expected


def test_export_writes_empty_files_for_empty_splits(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one")], settings=DatasetSettings("import", SplitParams(1, 0, 0)))
    paths = dataset.export_split_files(tmp_path / "out")
    assert paths["validation"].read_text() == ""
    assert paths["test"].read_text() == ""
    assert len(jsonl(paths["train"])) == 1


# --- read_import_file -------------------------------------------------------------------


def test_read_import_file_excludes_each_invalid_row_with_its_code() -> None:
    rows, exclusions = read_import_file(INVALID)
    assert rows == []
    assert [(e.row_id, e.code.value) for e in exclusions] == [
        (e["id"], e["code"]) for e in INVALID_EXPECTED
    ]
    for exclusion, expected in zip(exclusions, INVALID_EXPECTED, strict=True):
        assert exclusion.detail.startswith(f"invalid.jsonl:{expected['line']}: ")


def test_read_import_file_excludes_undecodable_json(tmp_path: Path) -> None:
    lines = TOOL_USE.read_text().splitlines()
    path = tmp_path / "hostile.jsonl"
    huge_int = '{"id": ' + "9" * 5000 + "}"
    deep = "[" * 100_000 + "]" * 100_000
    path.write_text("\n".join([lines[0], huge_int, deep, lines[1]]) + "\n")
    rows, exclusions = read_import_file(path)
    assert [row.id for row in rows] == [validate_row(json.loads(line)).id for line in lines[:2]]
    assert [e.code for e in exclusions] == [RowErrorCode.INVALID_ROW] * 2
    assert exclusions[0].detail.startswith("hostile.jsonl:2: invalid JSON: ")
    assert exclusions[1].detail.startswith("hostile.jsonl:3: invalid JSON: ")
    summary = DatasetDir(tmp_path / "ds").add(rows, exclusions, settings=IMPORT).summary
    assert (summary["accepted"], summary["excluded"]) == (2, 2)


def test_read_import_file_keeps_going_past_malformed_lines(tmp_path: Path) -> None:
    good = TOOL_USE.read_text().splitlines()[0]
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(
        b"not json\n[1, 2]\n\xff\xfe\n\n"
        + good.encode()
        + b'\n{"id": "x", "messages": [], "source": {"trace_id": "t", "span_id": 3}}\n'
    )
    rows, exclusions = read_import_file(path)
    assert [row.id for row in rows] == ["row-weather-001"]
    assert [e.to_json() for e in exclusions] == [
        {"code": "invalid_row", "detail": "mixed.jsonl:1: invalid JSON: Expecting value"},
        {"code": "invalid_row", "detail": "mixed.jsonl:2: expected a JSON object"},
        {"code": "invalid_row", "detail": "mixed.jsonl:3: not valid UTF-8"},
        {
            "code": "invalid_row",
            "detail": "mixed.jsonl:6: source.span_id: Input should be a valid string",
            "row_id": "x",
            "trace_id": "t",
        },
    ]
