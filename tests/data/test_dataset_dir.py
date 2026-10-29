# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.curation.labels import apply_labels
from pup_finetune.data import dataset_dir
from pup_finetune.data import jsonl as jsonl_module
from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings, SourceKind, read_import_file
from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.jsonl import read_rows, to_provider_dict
from pup_finetune.data.labels import Label
from pup_finetune.data.schema import (
    RowError,
    RowErrorCode,
    TrainingRow,
    split_key,
    validate_row,
)
from pup_finetune.data.split import SPLITS, SplitParams
from pup_finetune.errors import StateError, UsageError
from pup_finetune.storage import DirectoryLock
from tests.fakes.text import letters

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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


@pytest.mark.parametrize("stored", [False, True], ids=["in-the-input", "already-stored"])
@pytest.mark.parametrize("duplicate_first", [True, False], ids=["duplicate-first", "changed-first"])
def test_an_id_reused_after_a_duplicate_conversation_is_an_error(
    tmp_path: Path, stored: bool, duplicate_first: bool
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    original = make_row("a", "same")
    duplicate, changed = make_row("b", "same"), make_row("b", "different")
    rows = [duplicate, changed] if duplicate_first else [changed, duplicate]
    if stored:
        dataset.add([original], settings=IMPORT)
    else:
        rows.insert(0, original)
    with pytest.raises(StateError, match="'b' appears more than once in the input"):
        dataset.add(rows, settings=IMPORT)
    assert dataset.is_dataset() == stored
    if stored:
        assert [row.id for row in dataset.rows()] == ["a"]


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
    for settings, message in (
        (DatasetSettings("import", SplitParams(43, 0.1, 0.1)), "settings changed"),
        (DatasetSettings("import", SplitParams(42, 0.2, 0.1)), "settings changed"),
        (DatasetSettings("datadog", SplitParams(42, 0.1, 0.1)), "built by `dataset import`"),
    ):
        with pytest.raises(StateError, match=message):
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


def test_conversations_that_differ_only_in_tool_call_ids_are_duplicates(tmp_path: Path) -> None:
    def tool_row(row_id: str, call_id: str) -> TrainingRow:
        call = {"id": call_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
        return validate_row(
            {
                "id": row_id,
                "messages": [
                    {"role": "user", "content": "Look it up."},
                    {"role": "assistant", "tool_calls": [call]},
                    {"role": "tool", "content": "Found.", "tool_call_id": call_id},
                    {"role": "assistant", "content": "Here it is."},
                ],
            }
        )

    dataset = DatasetDir(tmp_path / "ds")
    result = dataset.add(
        [tool_row("row-a", "call_3e1a"), tool_row("row-b", "call_b70c")], settings=IMPORT
    )
    assert (result.added, result.excluded) == (1, 1)
    assert [row.id for row in dataset.rows()] == ["row-a"]
    assert dataset.excluded()[0]["code"] == "duplicate_conversation"


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


# --- retiring rows --------------------------------------------------------------------------


class Interrupted(Exception):
    """The failure a test injects to stop an add part-way."""


def interrupt_writes_of(
    monkeypatch: pytest.MonkeyPatch, path: Path, *, after: int = 0
) -> list[Path]:
    """Make the write of ``path`` that follows ``after`` earlier ones fail; returns every write."""
    real = dataset_dir.atomic_write_text
    writes: list[Path] = []

    def write(target: Path, text: str) -> None:
        earlier = sum(1 for written in writes if written == path)
        writes.append(Path(target))
        if Path(target) == path and earlier == after:
            raise Interrupted(str(target))
        real(target, text)

    # rows.jsonl is written through the jsonl module, the other files directly.
    monkeypatch.setattr(dataset_dir, "atomic_write_text", write)
    monkeypatch.setattr(jsonl_module, "atomic_write_text", write)
    return writes


def retire(row: TrainingRow) -> Exclusion:
    return Exclusion.for_row(row, ExclusionCode.CONTAINED_CONVERSATION, "a prefix of row 'long'")


def test_a_retired_row_is_removed_and_recorded(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    short, long = make_row("short", "one"), make_row("long", "two")
    dataset.add([short], settings=IMPORT)
    added = dataset.add([long], settings=IMPORT, retired=[retire(short)])
    assert [row.id for row in dataset.rows()] == ["long"]
    assert dataset.excluded() == [retire(short).to_json()]
    assert (added.added, added.retired) == (1, 1)


def test_retirements_are_recorded_before_the_rows_are_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    short, long = make_row("short", "one"), make_row("long", "two")
    other = Exclusion(RowErrorCode.INVALID_ROW, "bad", "broken")
    dataset.add([short], settings=IMPORT)
    # Stop after rows.jsonl is rewritten, before the final excluded.jsonl.
    writes = interrupt_writes_of(monkeypatch, dataset.excluded_path, after=1)
    with pytest.raises(Interrupted):
        dataset.add([long], [other], settings=IMPORT, retired=[retire(short)])
    assert writes.index(dataset.excluded_path) < writes.index(dataset.rows_path)
    # The retired row is gone, but its record is not: nothing is lost.
    assert [row.id for row in dataset.rows()] == ["long"]
    assert dataset.excluded() == [retire(short).to_json()]
    assert dataset.summary()["excluded_by_code"] == {"contained_conversation": 1}

    # A retry no longer sees the row to retire (it is gone and already recorded), and keeps
    # the record while it adds what the interrupted add did not.
    monkeypatch.undo()
    added = dataset.add([long], [other], settings=IMPORT)
    assert (added.added, added.unchanged) == (0, 1)
    assert dataset.excluded() == [retire(short).to_json(), other.to_json()]


def test_an_add_stopped_before_its_rows_leaves_the_retired_row_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    short, long = make_row("short", "one"), make_row("long", "two")
    dataset.add([short], settings=IMPORT)
    interrupt_writes_of(monkeypatch, dataset.rows_path)
    with pytest.raises(Interrupted):
        dataset.add([long], settings=IMPORT, retired=[retire(short)])
    # The row is still stored, so its staged record is not counted, and the retry retires it.
    assert [row.id for row in dataset.rows()] == ["short"]
    assert dataset.summary()["excluded"] == 0
    monkeypatch.undo()
    dataset.add([long], settings=IMPORT, retired=[retire(short)])
    assert [row.id for row in dataset.rows()] == ["long"]
    assert dataset.excluded() == [retire(short).to_json()]


def test_a_drop_whose_retirements_end_at_a_row_the_add_excludes_carries_nowhere(
    tmp_path: Path,
) -> None:
    # Nothing trains a row the dataset does not hold, so the dropped turns are out already.
    dataset = DatasetDir(tmp_path / "ds")
    short, middle, long = (make_row(row_id, row_id) for row_id in ("short", "middle", "long"))
    dataset.add([short, middle], settings=IMPORT)
    dataset.append_labels([Label(id="short", keep=False, reason="short is wrong")])
    code = ExclusionCode.CONTAINED_CONVERSATION
    retired = [
        Exclusion.for_row(short, code, "a prefix of row 'middle'", into="middle"),
        Exclusion.for_row(middle, code, "a prefix of row 'long'", into="long"),
    ]
    capped = Exclusion.for_row(long, ExclusionCode.TRACE_ROW_CAP, "past the cap")
    added = dataset.add([], [capped], settings=IMPORT, retired=retired)
    assert dataset.rows() == []
    assert added.carried == {}
    assert list(dataset.labels()) == ["short"]


def test_a_drop_a_keep_refuses_is_not_reported_when_another_drop_drops_the_row(
    tmp_path: Path,
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    prefix, middle, other, long = (
        make_row(row_id, row_id) for row_id in ("prefix", "middle", "other", "long")
    )
    dataset.add([prefix, middle, other, long], settings=IMPORT)
    dataset.append_labels(
        [
            Label(id="prefix", keep=False, reason="prefix is wrong"),
            Label(id="middle", keep=True, reason="middle is right"),
            Label(id="other", keep=False, reason="other is wrong"),
        ]
    )
    code = ExclusionCode.CONTAINED_CONVERSATION
    retired = [
        Exclusion.for_row(prefix, code, "a prefix of row 'middle'", into="middle"),
        Exclusion.for_row(middle, code, "a prefix of row 'long'", into="long"),
        Exclusion.for_row(other, code, "a prefix of row 'long'", into="long"),
    ]
    added = dataset.add([], settings=IMPORT, retired=retired)
    assert dataset.kept_rows() == []
    assert added.carried == {"long": "other"}
    assert added.not_carried == []


def test_retirements_that_form_a_cycle_are_refused(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    first, second = make_row("first", "one"), make_row("second", "two")
    dataset.add([first, second], settings=IMPORT)
    code = ExclusionCode.DUPLICATE_CONVERSATION
    retired = [
        Exclusion.for_row(first, code, "same conversation as row 'second'", into="second"),
        Exclusion.for_row(second, code, "same conversation as row 'first'", into="first"),
    ]
    with pytest.raises(ValueError, match="retirements form a cycle at 'first'"):
        dataset.add([], settings=IMPORT, retired=retired)
    assert [row.id for row in dataset.rows()] == ["first", "second"]


def with_reasoning(row: TrainingRow, text: str) -> TrainingRow:
    """``row`` with ``text`` as the reasoning of its last (assistant) message."""
    *before, last = row.messages
    thought = last.model_copy(update={"reasoning_content": text})
    return row.model_copy(update={"messages": [*before, thought]})


def test_an_updated_row_replaces_the_stored_one_in_place(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    first, second = make_row("first", "one", trace_id="t1"), make_row("second", "two")
    dataset.add([first, second], settings=IMPORT)
    splits = dataset.splits_path.read_bytes()
    thinking = with_reasoning(first, "Think.")
    added = dataset.add([], settings=IMPORT, updated=[thinking, second])
    assert (added.added, added.unchanged, added.updated) == (0, 0, 1)
    assert dataset.rows() == [thinking, second]
    assert dataset.splits_path.read_bytes() == splits
    assert dataset.summary()["accepted"] == 2


def test_an_update_must_name_a_stored_row_and_change_only_reasoning(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    stored = make_row("stored", "one")
    dataset.add([stored], settings=IMPORT)
    before = snapshot(dataset.path)
    other = make_row("stored", "changed text")
    with pytest.raises(ValueError, match="more than its reasoning"):
        dataset.add([], settings=IMPORT, updated=[other])
    with pytest.raises(ValueError, match="not stored"):
        dataset.add([], settings=IMPORT, updated=[make_row("unknown", "one")])
    thinking = with_reasoning(stored, "Think.")
    with pytest.raises(ValueError, match="both updated and added or retired"):
        dataset.add([], settings=IMPORT, retired=[retire(stored)], updated=[thinking])
    with pytest.raises(ValueError, match="both updated and added or retired"):
        dataset.add([thinking], settings=IMPORT, updated=[thinking])
    assert snapshot(dataset.path) == before


def test_an_update_must_not_repeat_another_stored_conversation(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    bare = make_row("bare", "one", trace_id="t1")
    thought = with_reasoning(make_row("thought", "one", trace_id="t2"), "Think.")
    dataset.add([bare, thought], settings=IMPORT)
    before = snapshot(dataset.path)
    with pytest.raises(ValueError, match="same conversation after the update"):
        dataset.add([], settings=IMPORT, updated=[with_reasoning(bare, "Think.")])
    assert snapshot(dataset.path) == before
    # Retiring the other row in the same add is how a pull resolves it.
    dataset.add(
        [], settings=IMPORT, retired=[retire(thought)], updated=[with_reasoning(bare, "Think.")]
    )
    assert [row.id for row in dataset.rows()] == ["bare"]


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
        "kept": len(fixture_rows),
        "labeled": 0,
        "dropped": 0,
        "drop_reasons": [],
        "splits": split_counts,
        "traces": len({split_key(row) for row in fixture_rows}),
        "split_groups": summary["split_groups"],
        "split_conflicts": {"groups": 0, "rows": 0, "row_ids": {}, "labels_file": None},
        "warnings": summary["warnings"],
    }
    assert sum(split_counts.values()) == len(fixture_rows)


def test_summary_warns_about_small_splits(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    params = DatasetSettings("import", SplitParams(42, 0.0, 0.0))
    summary = dataset.add([make_row("a", "one")], settings=params).summary
    assert summary["splits"] == {"train": 1, "validation": 0, "test": 0}
    assert len(summary["warnings"]) == 3


def single_traces(prefix: str, count: int) -> list[TrainingRow]:
    """``count`` unrelated rows, a trace each."""
    return [
        make_row(f"{prefix}-{n}", f"{prefix} {letters(n)}", trace_id=f"{prefix}-{n}")
        for n in range(count)
    ]


CHAINED = [
    make_row("a1", "Plan a trip for 2", trace_id="t1"),
    make_row("a2", "Book a hotel", trace_id="t1"),
    make_row("a3", "Rent a car", trace_id="t1"),
    make_row("b1", "Plan a trip for 3", trace_id="t2"),
    make_row("b2", "Pack 2 bags", trace_id="t2"),
    make_row("b3", "Buy a map", trace_id="t2"),
    make_row("c1", "Pack 3 bags", trace_id="t3"),
    make_row("c2", "Call a taxi", trace_id="t3"),
]
"""Eight rows in three traces chained by copies: t1 and t2 ask one trip, t2 and t3 one
packing list."""


def test_summary_warns_about_a_chained_group_over_a_tenth_of_the_rows(tmp_path: Path) -> None:
    rows = [*CHAINED, *single_traces("other", 47)]
    summary = DatasetDir(tmp_path / "ds").add(rows, settings=IMPORT).summary
    assert summary["traces"] == 50
    assert (
        "one group of rows that share a split holds 8 of 55 rows (15%, more than 10%) in 3 "
        "traces, so one split takes them all; its traces are likely chained through requests "
        "they share (e.g. rows a1, a2, a3)"
    ) in summary["warnings"]
    more = [*rows, *single_traces("more", 30)]
    summary = DatasetDir(tmp_path / "ds2").add(more, settings=IMPORT).summary
    assert not any(w.startswith("one group of rows") for w in summary["warnings"])


def test_summary_does_not_warn_about_a_group_in_a_dataset_under_50_traces(
    tmp_path: Path,
) -> None:
    rows = [*CHAINED, *single_traces("other", 46)]
    summary = DatasetDir(tmp_path / "ds").add(rows, settings=IMPORT).summary
    assert summary["traces"] == 49
    assert not any(w.startswith("one group of rows") for w in summary["warnings"])


def test_summary_does_not_warn_about_one_trace_over_a_tenth_of_the_rows(tmp_path: Path) -> None:
    agents = [make_row(f"agent-{n}", f"Agent {letters(n)}", trace_id="big") for n in range(12)]
    rows = [*agents, *single_traces("other", 49)]
    summary = DatasetDir(tmp_path / "ds").add(rows, settings=IMPORT).summary
    assert summary["traces"] == 50
    assert not any(w.startswith("one group of rows") for w in summary["warnings"])


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


def append_line(dataset: DatasetDir, row: TrainingRow) -> None:
    """Append ``row`` to ``rows.jsonl`` as a hand edit would."""
    with dataset.rows_path.open("a") as handle:
        handle.write(json.dumps(row.model_dump(mode="json", exclude_none=True)) + "\n")


@pytest.mark.parametrize("text", ["one", "changed"])
def test_a_repeated_row_id_is_a_state_error(tmp_path: Path, text: str) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one", trace_id="t1")], settings=IMPORT)
    append_line(dataset, make_row("a", text, trace_id="t1"))
    with pytest.raises(StateError, match="row id 'a' is on more than one line") as raised:
        dataset.rows()
    assert "dataset import" in str(raised.value.hint)
    with pytest.raises(StateError, match="on more than one line"):
        dataset.add([], settings=IMPORT)


def test_a_repeated_conversation_is_refused_until_one_copy_is_dropped(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one", trace_id="t1")], settings=IMPORT)
    append_line(dataset, make_row("a-copy", "one", trace_id="t1"))
    with pytest.raises(
        StateError, match=r"'a' \((\w+)\) and 'a-copy' \(\1\) hold the same"
    ) as raised:
        dataset.rows_by_split()
    assert "dataset label --file" in str(raised.value.hint)
    assert dataset.summary()["accepted"] == 2
    assert [row.id for row in dataset.rows()] == ["a", "a-copy"]
    apply_labels(dataset, [Label(id="a-copy", keep=False, reason="a copy")])
    assert [row.id for rows in dataset.rows_by_split().values() for row in rows] == ["a"]


def test_a_reimport_does_not_adopt_a_hand_added_copy(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one", trace_id="t1")], settings=IMPORT)
    append_line(dataset, make_row("a-copy", "one", trace_id="t2"))
    dataset.add([], settings=IMPORT)  # assigns the copy its group's split
    with pytest.raises(StateError, match="hold the same conversation in the"):
        dataset.rows_by_split()


def test_a_dropped_copy_in_another_split_still_blocks_an_import(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one", trace_id="t1")], settings=IMPORT)
    append_line(dataset, make_row("a-copy", "one", trace_id="t2"))
    data = json.loads(dataset.splits_path.read_text())
    data["assignments"]["t2"] = next(s for s in SPLITS if s != data["assignments"]["t1"])
    dataset.splits_path.write_text(json.dumps(data))
    with pytest.raises(StateError, match="in different splits") as raised:
        dataset.rows_by_split()
    assert "dataset label" not in str(raised.value.hint)
    apply_labels(dataset, [Label(id="a-copy", keep=False, reason="a copy")])
    assert [row.id for rows in dataset.rows_by_split().values() for row in rows] == ["a"]
    with pytest.raises(StateError, match="in different splits"):
        dataset.add([], settings=IMPORT)


@pytest.mark.parametrize(
    ("source", "command"),
    [
        ("import", "dataset import"),
        ("datadog", "dataset pull"),
        ("datadog_dataset", "dataset fetch"),
    ],
)
def test_the_no_split_hint_names_the_command_that_built_it(
    tmp_path: Path, source: SourceKind, command: str
) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([make_row("a", "one", trace_id="t1")], settings=DatasetSettings(source))
    data = json.loads(dataset.splits_path.read_text())
    data["assignments"].pop("t1")
    dataset.splits_path.write_text(json.dumps(data))
    with pytest.raises(StateError, match="row 'a' has no split assignment") as raised:
        dataset.rows_by_split()
    assert f"rerun `{command}` into this directory" in str(raised.value.hint)


def test_labels_of_unknown_ids_warn(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    rows = [make_row("a", "one", trace_id="t1"), make_row("b", "two", trace_id="t2")]
    dataset.add(rows, settings=IMPORT)
    apply_labels(dataset, [Label(id="b", keep=False, reason="off-topic")])
    dataset.add([], settings=IMPORT, retired=[retire(rows[1])])
    assert not [w for w in dataset.summary()["warnings"] if "labels.jsonl" in w]
    with dataset.labels_path.open("a") as handle:
        handle.write(json.dumps({"id": "a-typo", "keep": False, "reason": "off-topic"}) + "\n")
    assert [w for w in dataset.summary()["warnings"] if "labels.jsonl" in w] == [
        "1 label in labels.jsonl names no row of the dataset (e.g. 'a-typo'), so it changes "
        "nothing; record labels with `dataset label --file`, which refuses such ids"
    ]
    assert [row.id for row in dataset.kept_rows()] == ["a"]


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
    nan = '{"id": "row-nan", "messages": [], "source": {"evaluations": {"score": NaN}}}'
    path.write_text("\n".join([lines[0], huge_int, deep, nan, lines[1]]) + "\n")
    rows, exclusions = read_import_file(path)
    assert [row.id for row in rows] == [validate_row(json.loads(line)).id for line in lines[:2]]
    assert [e.code for e in exclusions] == [RowErrorCode.INVALID_ROW] * 3
    assert exclusions[0].detail.startswith("hostile.jsonl:2: invalid JSON: ")
    assert exclusions[1].detail.startswith("hostile.jsonl:3: invalid JSON: ")
    assert exclusions[2].detail == "hostile.jsonl:4: invalid JSON: NaN is not a JSON value"
    summary = DatasetDir(tmp_path / "ds").add(rows, exclusions, settings=IMPORT).summary
    assert (summary["accepted"], summary["excluded"]) == (2, 3)


def test_read_import_file_excludes_too_deeply_nested_arguments(tmp_path: Path) -> None:
    lines = TOOL_USE.read_text().splitlines()
    depth = 100_000
    nested = {
        "id": "row-deep-arguments",
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"x": ' + "[" * depth + "]" * depth + "}",
                        },
                    }
                ],
            },
        ],
    }
    path = tmp_path / "deep.jsonl"
    path.write_text("\n".join([lines[0], json.dumps(nested), lines[1]]) + "\n")
    rows, exclusions = read_import_file(path)
    assert [row.id for row in rows] == [validate_row(json.loads(line)).id for line in lines[:2]]
    assert [(e.row_id, e.code) for e in exclusions] == [
        ("row-deep-arguments", RowErrorCode.INVALID_ARGUMENTS_JSON)
    ]


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
    last = b'{"id": "x", "messages": [], "source": {"trace_id": "t", "span_id": 3}}'
    assert [e.to_json() for e in exclusions] == [
        {
            "code": "invalid_row",
            "detail": "mixed.jsonl:1: invalid JSON: Expecting value",
            "line_sha256": sha256(b"not json"),
        },
        {
            "code": "invalid_row",
            "detail": "mixed.jsonl:2: expected a JSON object",
            "line_sha256": sha256(b"[1, 2]"),
        },
        {
            "code": "invalid_row",
            "detail": "mixed.jsonl:3: not valid UTF-8",
            "line_sha256": sha256(b"\xff\xfe"),
        },
        {
            "code": "invalid_row",
            "detail": "mixed.jsonl:6: source.span_id: Input should be a valid string",
            "row_id": "x",
            "trace_id": "t",
            "line_sha256": sha256(last),
        },
    ]


def test_read_import_file_excludes_lone_surrogates_and_keeps_going(tmp_path: Path) -> None:
    # json.dumps writes a lone surrogate as a "\ud800" escape: valid JSON, but the decoded
    # text cannot be encoded as UTF-8.
    lines = TOOL_USE.read_text().splitlines()

    def chat(text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}]

    bad = [
        json.dumps({"id": "row-content", "messages": chat("\ud800")}),
        json.dumps(
            {
                "id": "row-trace",
                "messages": chat("hi"),
                "source": {"trace_id": "t\ud800", "span_id": "s"},
            }
        ),
        json.dumps({"id": "\udc00", "messages": chat("hi")}),
    ]
    path = tmp_path / "surrogates.jsonl"
    path.write_text("\n".join([lines[0], *bad, lines[1]]) + "\n")
    rows, exclusions = read_import_file(path)
    assert [row.id for row in rows] == [validate_row(json.loads(line)).id for line in lines[:2]]
    assert [(e.code, e.row_id, e.trace_id, e.span_id) for e in exclusions] == [
        (RowErrorCode.INVALID_ROW, "row-content", None, None),
        (RowErrorCode.INVALID_ROW, "row-trace", None, "s"),
        (RowErrorCode.INVALID_ROW, None, None, None),
    ]
    assert exclusions[0].detail == (
        "surrogates.jsonl:2: messages[0].content has a lone surrogate (U+D800); "
        "UTF-8 cannot encode it"
    )
    dataset = DatasetDir(tmp_path / "ds")
    summary = dataset.add(rows, exclusions, settings=IMPORT).summary
    assert (summary["accepted"], summary["excluded"]) == (2, 3)
    records = dataset.excluded()  # Written and read back as UTF-8.
    assert sorted(str(record.get("row_id")) for record in records) == [
        "None",
        "row-content",
        "row-trace",
    ]


def test_an_undecodable_file_name_is_escaped_in_the_exclusion() -> None:
    # Python decodes undecodable bytes in a file name to lone surrogates.
    error = RowError(RowErrorCode.INVALID_ROW, "id: Field required")
    exclusion = Exclusion.for_invalid({}, error, where="rows-\udcff.jsonl:1", line=b"{}")
    assert exclusion.detail == "rows-\\udcff.jsonl:1: id: Field required"


def test_id_less_invalid_rows_from_same_named_files_are_counted_apart(tmp_path: Path) -> None:
    # Same file name, line and diagnostic; only the line content tells them apart.
    first, second = tmp_path / "a" / "rows.jsonl", tmp_path / "b" / "rows.jsonl"
    for path, text in ((first, "one"), (second, "two")):
        path.parent.mkdir()
        path.write_text(json.dumps({"messages": [{"role": "user", "content": text}]}) + "\n")
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add(*read_import_file(first), settings=IMPORT)
    summary = dataset.add(*read_import_file(second), settings=IMPORT).summary
    details = [record["detail"] for record in dataset.excluded()]
    assert details == ["rows.jsonl:1: id: Field required"] * 2
    assert (summary["total"], summary["excluded"]) == (2, 2)

    # Re-importing either file is an exact repeat, so nothing is counted twice.
    before = snapshot(dataset.path)
    assert dataset.add(*read_import_file(first), settings=IMPORT).summary == summary
    assert snapshot(dataset.path) == before
