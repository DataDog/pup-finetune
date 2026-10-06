# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Keep/drop labels on a dataset directory, and the readable sample used to review rows."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pup_finetune.curation.labels import Label, apply_labels, read_labels
from pup_finetune.curation.samples import describe_row, render_assistant, sample_rows, truncate
from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings
from pup_finetune.data.schema import Message, TrainingRow, validate_row
from pup_finetune.data.split import SplitParams
from pup_finetune.errors import StateError, UsageError
from pup_finetune.providers.fake import FAKE_PROFILE, FakeRenderer
from pup_finetune.storage import DirectoryLock
from pup_finetune.training.inputs import build_inputs

ALL_TRAIN = DatasetSettings("import", SplitParams(42, 0.0, 0.0))
MIXED = DatasetSettings("import", SplitParams(7, 0.3, 0.3))
AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def make_row(row_id: str, text: str | None = None) -> TrainingRow:
    question = text if text is not None else f"question {row_id}"
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"answer {row_id}"},
            ],
        }
    )


def tool_row(row_id: str, arguments: dict[str, Any]) -> TrainingRow:
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps(arguments)},
    }
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": "find it"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call-1", "content": "found"},
                {"role": "assistant", "content": "", "tool_calls": [{**call, "id": "call-2"}]},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    )


def make_dataset(path: Path, ids: list[str], settings: DatasetSettings = ALL_TRAIN) -> DatasetDir:
    dataset = DatasetDir(path)
    dataset.add([make_row(row_id) for row_id in ids], settings=settings)
    return dataset


def keep(row_id: str) -> Label:
    return Label(id=row_id, keep=True, at=AT)


def drop(row_id: str, reason: str = "off topic") -> Label:
    return Label(id=row_id, keep=False, reason=reason, at=AT)


def kept_ids(dataset: DatasetDir) -> list[str]:
    return [row.id for row in dataset.kept_rows()]


# --- the label model --------------------------------------------------------------------


def test_a_drop_needs_a_reason() -> None:
    with pytest.raises(ValidationError, match="reason is required"):
        Label(id="a", keep=False)
    with pytest.raises(ValidationError, match="reason is required"):
        Label(id="a", keep=False, reason="   ")
    assert Label(id="a", keep=True).reason == ""


def test_labels_reject_unknown_fields_and_loose_values() -> None:
    with pytest.raises(ValidationError):
        Label.model_validate({"id": "a", "keep": True, "score": 3})
    with pytest.raises(ValidationError):
        Label.model_validate({"id": "a", "keep": "false", "reason": "x"})
    with pytest.raises(ValidationError):
        Label.model_validate({"id": "a", "keep": True, "by": "robot"})
    with pytest.raises(ValidationError):
        Label.model_validate({"id": "", "keep": True})


def test_label_times_are_stored_in_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    label = Label(id="a", keep=True, at=datetime(2026, 1, 2, 5, 4, 5, tzinfo=plus_two))
    assert label.to_json()["at"] == "2026-01-02T03:04:05Z"
    with pytest.raises(ValidationError):
        Label.model_validate({"id": "a", "keep": True, "at": "2026-01-02T03:04:05"})
    assert Label(id="a", keep=True).at.tzinfo == UTC


# --- apply_labels -----------------------------------------------------------------------


def test_apply_records_labels_and_drops_rows(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b", "c"])
    rows_before = dataset.rows_path.read_bytes()
    report = apply_labels(dataset, [drop("a"), keep("b")])
    assert (report.applied, report.unchanged, report.kept, report.dropped) == (2, 0, 1, 1)
    assert kept_ids(dataset) == ["b", "c"]
    assert read_labels(dataset.labels_path) == [drop("a"), keep("b")]
    assert stat.S_IMODE(dataset.labels_path.stat().st_mode) == 0o600
    assert dataset.rows_path.read_bytes() == rows_before  # rows are never changed
    assert [row.id for row in dataset.rows()] == ["a", "b", "c"]


def test_the_summary_counts_kept_and_dropped_rows(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b", "c", "d"])
    report = apply_labels(dataset, [drop("a", "too short"), drop("b", "too short"), drop("c")])
    summary = report.summary
    assert (summary["accepted"], summary["kept"], summary["dropped"]) == (4, 1, 3)
    assert summary["labeled"] == 3
    assert summary["drop_reasons"] == [
        {"reason": "too short", "count": 2},
        {"reason": "off topic", "count": 1},
    ]
    assert summary["splits"] == {"train": 1, "validation": 0, "test": 0}
    assert json.loads(dataset.summary_path.read_text()) == summary
    assert dataset.summary() == summary


def test_the_summary_lists_the_top_ten_drop_reasons(tmp_path: Path) -> None:
    ids = [f"r{i:02d}" for i in range(13)]
    dataset = make_dataset(tmp_path / "ds", ids)
    # Twelve reasons once each, and one that sorts last alphabetically twice.
    labels = [drop(row_id, f"reason {row_id}") for row_id in ids[:12]]
    labels += [drop("r12", "zzz repeated"), drop("r11", "zzz repeated")]
    summary = apply_labels(dataset, labels).summary
    assert summary["dropped"] == 13
    reasons = summary["drop_reasons"]
    assert len(reasons) == 10
    assert reasons[0] == {"reason": "zzz repeated", "count": 2}
    # Ties rank alphabetically: of the eleven reasons given once, the last two are cut.
    assert [entry["reason"] for entry in reasons[1:]] == [f"reason r{i:02d}" for i in range(9)]


def test_the_last_label_of_an_id_wins(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b"])
    report = apply_labels(dataset, [drop("a"), keep("a"), keep("b"), drop("b")])
    assert (report.kept, report.dropped) == (1, 1)
    assert kept_ids(dataset) == ["a"]
    # A later batch overrides an earlier one, and the log keeps both.
    apply_labels(dataset, [keep("b"), drop("a")])
    assert kept_ids(dataset) == ["b"]
    assert len(read_labels(dataset.labels_path)) == 6
    assert dataset.labels()["a"] == drop("a")


def test_applying_the_same_labels_again_changes_nothing(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b"])
    apply_labels(dataset, [drop("a"), keep("b")])
    before = dataset.labels_path.read_bytes()
    later = [Label(id="a", keep=False, reason="off topic", at=AT + timedelta(days=1))]
    report = apply_labels(dataset, later)
    assert (report.applied, report.unchanged, report.dropped) == (0, 1, 1)
    assert dataset.labels_path.read_bytes() == before


def test_a_new_source_or_reason_for_the_same_decision_is_recorded(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b"])
    apply_labels(dataset, [drop("a")])
    confirmed = Label(id="a", keep=False, reason="off topic", by="human", at=AT)
    report = apply_labels(dataset, [confirmed])
    assert (report.applied, report.unchanged) == (1, 0)
    assert dataset.labels()["a"].by == "human"
    report = apply_labels(dataset, [drop("a", "wrong tool")])
    assert report.applied == 1
    assert report.summary["drop_reasons"] == [{"reason": "wrong tool", "count": 1}]
    assert len(read_labels(dataset.labels_path)) == 3


def test_a_repeat_within_one_batch_is_recorded_once(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a"])
    report = apply_labels(dataset, [drop("a"), drop("a")])
    assert (report.applied, report.unchanged, report.dropped) == (1, 1, 1)
    assert read_labels(dataset.labels_path) == [drop("a")]


def test_unknown_ids_are_rejected_and_nothing_is_written(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a"])
    unknown = [keep(f"missing-{i:02d}") for i in range(12)]
    with pytest.raises(UsageError) as caught:
        apply_labels(dataset, [drop("a"), *unknown, keep("missing-00")])
    message = caught.value.message
    assert message.startswith("12 labeled ids are not accepted rows")
    assert "'missing-09'" in message
    assert "'missing-10'" not in message
    assert message.endswith("and 2 more")
    assert not dataset.labels_path.exists()
    assert kept_ids(dataset) == ["a"]


def test_excluded_rows_cannot_be_labeled(tmp_path: Path) -> None:
    dataset = DatasetDir(tmp_path / "ds")
    original = make_row("a")
    copy = original.model_copy(update={"id": "b"})
    dataset.add([original, copy], settings=ALL_TRAIN)
    assert dataset.summary()["excluded_by_code"] == {"duplicate_conversation": 1}
    with pytest.raises(UsageError, match="'b'"):
        apply_labels(dataset, [drop("b")])


def test_labeling_needs_a_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="not a dataset directory"):
        apply_labels(DatasetDir(tmp_path / "nothing"), [keep("a")])


def test_labeling_holds_the_directory_lock(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a"])
    with DirectoryLock(dataset.path), pytest.raises(StateError, match="another pup-finetune"):
        apply_labels(dataset, [drop("a")])
    assert not dataset.labels_path.exists()


def test_a_corrupt_labels_file_is_reported_with_its_line(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a"])
    dataset.labels_path.write_text(
        json.dumps(keep("a").to_json()) + "\n" + json.dumps({"id": "a", "keep": False}) + "\n"
    )
    with pytest.raises(StateError, match=r"labels\.jsonl:2: invalid label"):
        dataset.kept_rows()
    with pytest.raises(StateError, match=r"labels\.jsonl:2"):
        apply_labels(dataset, [keep("a")])


def test_labels_of_rows_no_longer_accepted_are_ignored(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b"])
    lines = [json.dumps(drop(row_id).to_json()) for row_id in ("a", "gone")]
    dataset.labels_path.write_text("\n".join(lines))  # no final newline
    summary = dataset.summary()
    assert (summary["kept"], summary["dropped"], summary["labeled"]) == (1, 1, 1)
    apply_labels(dataset, [drop("b")])
    assert [label.id for label in read_labels(dataset.labels_path)] == ["a", "gone", "b"]


# --- what labels leave out --------------------------------------------------------------


def test_dropped_rows_are_left_out_of_the_split_files(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b", "c"])
    apply_labels(dataset, [drop("b")])
    paths = dataset.export_split_files(tmp_path / "out")
    train = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    assert [row["messages"][0]["content"] for row in train] == ["question a", "question c"]


def test_dropped_rows_are_left_out_of_the_training_inputs(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b", "c"])
    apply_labels(dataset, [drop("a"), keep("b")])
    inputs = build_inputs(dataset, FakeRenderer(), FAKE_PROFILE)
    assert [row.row_id for row in inputs.train_rows] == ["b", "c"]


def test_adding_rows_keeps_the_label_counts(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b"])
    apply_labels(dataset, [drop("a")])
    summary = dataset.add([make_row("c")], settings=ALL_TRAIN).summary
    assert (summary["accepted"], summary["kept"], summary["dropped"]) == (3, 2, 1)


def test_dropping_a_row_changes_the_dataset_hash_of_the_inputs(tmp_path: Path) -> None:
    # A run checks this hash on resume, so relabeling its dataset makes a resume refuse.
    dataset = make_dataset(tmp_path / "ds", ["a", "b", "c"])
    before = build_inputs(dataset, FakeRenderer(), FAKE_PROFILE).dataset_sha256
    apply_labels(dataset, [drop("b")])
    assert build_inputs(dataset, FakeRenderer(), FAKE_PROFILE).dataset_sha256 != before


# --- samples ----------------------------------------------------------------------------


def test_a_sample_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    ids = [f"r{i:02d}" for i in range(30)]
    dataset = make_dataset(tmp_path / "ds", ids, MIXED)
    first, eligible = sample_rows(dataset, n=5, seed=3)
    again, _ = sample_rows(dataset, n=5, seed=3)
    other, _ = sample_rows(dataset, n=5, seed=4)
    assert eligible == 30
    assert first == again
    assert [row["id"] for row in first] != [row["id"] for row in other]
    assert [row["id"] for row in first] == sorted(row["id"] for row in first)


def test_adding_rows_lets_a_new_row_into_a_sample_only_by_outranking_one(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds", [f"r{i:02d}" for i in range(20)])
    first, _ = sample_rows(dataset, n=5, seed=9)
    added = [f"n{i:02d}" for i in range(20)]
    dataset.add([make_row(row_id) for row_id in added], settings=ALL_TRAIN)
    second, eligible = sample_rows(dataset, n=5, seed=9)
    assert eligible == 40
    before = {row["id"] for row in first}
    after = {row["id"] for row in second}
    assert after <= before | set(added)
    assert after & set(added)  # with these seeds some new rows do outrank old ones


def test_a_sample_draws_from_one_split(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", [f"r{i:02d}" for i in range(30)], MIXED)
    validation = [row.id for row in dataset.rows_for_split("validation")]
    rows, eligible = sample_rows(dataset, n=100, split="validation")
    assert eligible == len(validation)
    assert sorted(row["id"] for row in rows) == sorted(validation)
    assert {row["split"] for row in rows} == {"validation"}


def test_a_sample_leaves_out_dropped_rows_unless_asked(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "ds", ["a", "b", "c"])
    apply_labels(dataset, [drop("a"), keep("b")])
    rows, eligible = sample_rows(dataset)
    assert eligible == 2
    assert {row["id"]: row["label"] for row in rows} == {"b": "keep", "c": "none"}
    rows, eligible = sample_rows(dataset, include_dropped=True)
    assert eligible == 3
    assert {row["id"]: row["label"] for row in rows} == {"a": "drop", "b": "keep", "c": "none"}


def test_a_sample_describes_a_row_in_brief(tmp_path: Path) -> None:
    row = make_row("long", "x" * 400)
    described = describe_row(row, "train", {})
    assert described == {
        "id": "long",
        "split": "train",
        "label": "none",
        "turns": 1,
        "tools_used": [],
        "first_user_message": "x" * 299 + "…",
        "final_assistant_message": "answer long",
    }


def test_truncation_keeps_short_text_whole() -> None:
    assert truncate("x" * 300) == "x" * 300
    assert truncate("x" * 301) == "x" * 299 + "…"
    assert len(truncate("y" * 5000)) == 300


def test_a_final_message_without_text_shows_its_tool_calls() -> None:
    row = tool_row("calls", {"q": "status"})
    described = describe_row(row, "test", {})
    assert described["turns"] == 2
    assert described["tools_used"] == ["lookup"]
    assert described["final_assistant_message"] == 'lookup({"q": "status"})'
    long = tool_row("long", {"q": "z" * 500})
    rendered = describe_row(long, "test", {})["final_assistant_message"]
    assert rendered.startswith('lookup({"q": "zzz')
    assert len(rendered) == 300
    assert rendered.endswith("…")


def test_several_tool_calls_are_listed_in_order() -> None:
    message = Message.model_validate(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "function": {"name": "first", "arguments": "{}"}},
                {"id": "2", "function": {"name": "second", "arguments": '{"n": 2}'}},
            ],
        }
    )
    assert render_assistant(message) == 'first({}), second({"n": 2})'


def test_a_final_message_with_text_shows_its_text_not_its_calls() -> None:
    message = Message.model_validate(
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [{"id": "1", "function": {"name": "lookup", "arguments": "{}"}}],
        }
    )
    assert render_assistant(message) == "checking"


def test_long_assistant_text_is_truncated() -> None:
    message = Message.model_validate({"role": "assistant", "content": "w" * 400})
    rendered = render_assistant(message)
    assert len(rendered) == 300
    assert rendered == "w" * 299 + "…"
