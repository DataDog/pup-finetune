# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset label` and `dataset sample`, end to end through the Typer app."""

from __future__ import annotations

import json
import re
import shlex
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.curation.labels import Label, apply_labels
from pup_finetune.curation.samples import JUDGE_LINE_NOTE
from pup_finetune.data.dataset_dir import DatasetDir

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"
FIXTURE_IDS = sorted(json.loads(line)["id"] for line in TOOL_USE.read_text().splitlines())

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["dataset", *args])


def output_json(result: Result) -> dict[str, Any]:
    assert result.exit_code == 0, result.stderr
    data: dict[str, Any] = json.loads(result.stdout)
    return data


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    target = tmp_path / "ds"
    result = invoke("import", str(target), "--file", str(TOOL_USE))
    assert result.exit_code == 0, result.stderr
    return target


def label_file(tmp_path: Path, *records: dict[str, Any], name: str = "labels.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def label(directory: Path, file: Path, *extra: str) -> Result:
    return invoke("label", str(directory), "--file", str(file), *extra)


# --- label ------------------------------------------------------------------------------


def test_label_applies_labels_and_emits_the_report(dataset: Path, tmp_path: Path) -> None:
    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[0], "keep": False, "reason": "wrong tool"},
        {"id": FIXTURE_IDS[1], "keep": True, "reason": ""},
    )
    data = output_json(label(dataset, file))
    assert data["dataset_dir"] == str(dataset.resolve())
    assert data["label"] == {
        "file": str(file),
        "labels": 2,
        "applied": 2,
        "unchanged": 0,
        "kept": 1,
        "dropped": 1,
    }
    assert (data["accepted"], data["kept"], data["dropped"]) == (len(FIXTURE_IDS), 13, 1)
    assert data["drop_reasons"] == [{"reason": "wrong tool", "count": 1}]
    assert sum(data["splits"].values()) == 13
    quoted = shlex.quote(str(dataset))
    assert data["next_command"] is None
    assert [step["command"] for step in data["next_steps"]] == [
        "pup finetune models list",
        f"pup finetune dataset push {quoted} --name <NAME>",
    ]
    assert FIXTURE_IDS[0] not in {row.id for row in DatasetDir(dataset).kept_rows()}
    show = output_json(invoke("show", str(dataset)))
    assert show["summary"]["dropped"] == 1


def test_labels_are_manual_and_at_is_filled_in(dataset: Path, tmp_path: Path) -> None:
    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[0], "keep": True},
        {"id": FIXTURE_IDS[1], "keep": True, "at": "2026-01-02T03:04:05+01:00"},
    )
    output_json(label(dataset, file))
    records = [json.loads(line) for line in (dataset / "labels.jsonl").read_text().splitlines()]
    assert [record["by"] for record in records] == ["manual", "manual"]
    assert records[0]["at"].endswith("Z")
    assert records[1]["at"] == "2026-01-02T02:04:05Z"


@pytest.mark.parametrize("old", ["agent", "human"])
def test_a_label_file_from_before_says_agent_or_human_and_reads_as_manual(
    dataset: Path, tmp_path: Path, old: str
) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True, "by": old})
    output_json(label(dataset, file))
    [record] = [json.loads(line) for line in (dataset / "labels.jsonl").read_text().splitlines()]
    assert record["by"] == "manual"


def test_a_label_file_cannot_claim_the_judge_council(dataset: Path, tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True, "by": "judge"})
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "judge" in result.stderr
    assert not (dataset / "labels.jsonl").exists()


def test_label_has_no_by_option(dataset: Path, tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True})
    result = label(dataset, file, "--by", "human")
    assert result.exit_code == 2


def test_label_last_label_wins_across_runs(dataset: Path, tmp_path: Path) -> None:
    row_id = FIXTURE_IDS[0]
    drop = label_file(tmp_path, {"id": row_id, "keep": False, "reason": "bad"}, name="a.jsonl")
    keep = label_file(tmp_path, {"id": row_id, "keep": True}, name="b.jsonl")
    assert output_json(label(dataset, drop))["dropped"] == 1
    assert output_json(label(dataset, keep))["dropped"] == 0
    again = output_json(label(dataset, keep))
    assert (again["label"]["applied"], again["label"]["unchanged"]) == (0, 1)


def test_label_rejects_unknown_ids(dataset: Path, tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True}, {"id": "nope", "keep": True})
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "1 labeled id is not an accepted row of the dataset: 'nope'" in result.stderr
    assert result.stdout == ""
    assert not (dataset / "labels.jsonl").exists()


def test_label_requires_a_reason_to_drop(dataset: Path, tmp_path: Path) -> None:
    file = label_file(
        tmp_path, {"id": FIXTURE_IDS[0], "keep": True}, {"id": FIXTURE_IDS[1], "keep": False}
    )
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "labels.jsonl:2: invalid label: a reason is required when keep is false" in (
        result.stderr
    )
    assert not (dataset / "labels.jsonl").exists()


def test_label_rejects_a_lone_surrogate_in_a_reason(dataset: Path, tmp_path: Path) -> None:
    keep = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": True}, name="keep.jsonl")
    output_json(label(dataset, keep))
    before = (dataset / "labels.jsonl").read_bytes()
    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[1], "keep": True},
        {"id": FIXTURE_IDS[2], "keep": False, "reason": "bad \ud800"},
    )
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "labels.jsonl:2: invalid label: reason: has a lone surrogate (U+D800)" in result.stderr
    assert (dataset / "labels.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ("not json", "invalid JSON"),
        ('["a"]', "expected a JSON object"),
        ('{"id": "x", "keep": true, "extra": 1}', "extra"),
        ('{"keep": true}', "id"),
    ],
)
def test_label_reports_bad_lines(dataset: Path, tmp_path: Path, line: str, problem: str) -> None:
    file = tmp_path / "labels.jsonl"
    file.write_text("\n" + line + "\n")
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "labels.jsonl:2: " in result.stderr
    assert problem in result.stderr


def test_label_rejects_an_empty_file(dataset: Path, tmp_path: Path) -> None:
    file = tmp_path / "labels.jsonl"
    file.write_text("\n")
    result = label(dataset, file)
    assert result.exit_code == 2
    assert "has no labels" in result.stderr


def test_label_needs_a_dataset_directory(tmp_path: Path) -> None:
    file = label_file(tmp_path, {"id": "a", "keep": True})
    result = label(tmp_path / "missing", file)
    assert result.exit_code == 2
    assert "is not a dataset directory" in result.stderr


# --- sample -----------------------------------------------------------------------------


def sample(directory: Path, *extra: str) -> dict[str, Any]:
    return output_json(invoke("sample", str(directory), *extra))


def test_sample_shows_rows_in_brief(dataset: Path) -> None:
    data = sample(dataset, "--count", "3")
    assert data["dataset_dir"] == str(dataset.resolve())
    assert (data["split"], data["seed"], data["eligible"]) == ("all", 0, len(FIXTURE_IDS))
    assert len(data["rows"]) == 3
    for row in data["rows"]:
        assert set(row) == {
            "id",
            "split",
            "label",
            "turns",
            "tools_used",
            "first_user_message",
            "final_assistant_message",
        }
        assert row["label"] == "none"
    assert data["rows_file"] == str((dataset / "rows.jsonl").resolve())
    # Labels come from a file the reader writes, so labeling is a step, not next_command.
    assert data["next_command"] is None
    assert data["next_steps"][0]["command"] == (
        f"pup finetune dataset label {shlex.quote(str(dataset))} --file <LABELS_FILE>"
    )


def test_sample_is_deterministic_for_a_seed(dataset: Path) -> None:
    first = sample(dataset, "--count", "4", "--seed", "11")
    assert sample(dataset, "--count", "4", "--seed", "11") == first
    assert (
        sample(dataset, "--count", "100")["rows"]
        == sample(dataset, "--count", "100", "--seed", "5")["rows"]
    )


def test_sample_of_one_split(dataset: Path) -> None:
    train = {row.id for row in DatasetDir(dataset).rows_for_split("train")}
    data = sample(dataset, "--split", "train", "--count", "100")
    assert {row["id"] for row in data["rows"]} == train
    assert {row["split"] for row in data["rows"]} == {"train"}


def test_sample_hides_dropped_rows_unless_asked(dataset: Path, tmp_path: Path) -> None:
    dropped = FIXTURE_IDS[0]
    output_json(label(dataset, label_file(tmp_path, {"id": dropped, "keep": False, "reason": "x"})))
    shown = {row["id"] for row in sample(dataset, "--count", "100")["rows"]}
    assert dropped not in shown
    everything = sample(dataset, "--count", "100", "--include-dropped")["rows"]
    assert {row["id"]: row["label"] for row in everything}[dropped] == "drop"


def test_sample_renders_tool_calls_of_a_final_message(tmp_path: Path) -> None:
    row = {
        "id": "tool-final",
        "messages": [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "lookup", "arguments": '{"key": "k1"}'}}
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    target = tmp_path / "ds"
    assert invoke("import", str(target), "--file", str(rows)).exit_code == 0
    [described] = sample(target)["rows"]
    assert described["final_assistant_message"] == 'lookup({"key": "k1"})'
    assert described["tools_used"] == ["lookup"]


def test_sample_rejects_an_unknown_split(dataset: Path) -> None:
    result = invoke("sample", str(dataset), "--split", "holdout")
    assert result.exit_code == 2
    assert "--split" in _ANSI.sub("", result.stderr)


def test_sample_needs_a_dataset_directory(tmp_path: Path) -> None:
    result = invoke("sample", str(tmp_path / "missing"))
    assert result.exit_code == 2
    assert "is not a dataset directory" in result.stderr


# --- sample --markdown -------------------------------------------------------------------


def test_sample_writes_a_private_review_file_whose_labels_lines_label_the_rows(
    dataset: Path, tmp_path: Path
) -> None:
    from pup_finetune.cli._summary import summarize

    review = tmp_path / "review" / "sample.md"
    result = invoke("sample", str(dataset), "--count", "3", "--markdown", str(review))
    data = output_json(result)
    assert stat.S_IMODE(review.stat().st_mode) == 0o600
    assert stat.S_IMODE(review.parent.stat().st_mode) == 0o700
    assert data["markdown_file"] == str(review.resolve())
    assert data["rows_file"] == str((dataset / "rows.jsonl").resolve())
    assert "0600" in data["notes"][0] and "trace content" in data["notes"][0]
    assert f"note: {data['notes'][0]}" in result.stderr
    found = summarize("dataset sample", data)
    assert found is not None and any(data["markdown_file"] in line for line in found.lines)
    label_step = data["next_steps"][0]
    assert label_step["command"].endswith("--file <LABELS_FILE>")
    assert data["markdown_file"] in label_step["why"]
    text = review.read_text()
    assert text.index("## How to record decisions") < text.index(
        f"```sh\n{label_step['command']}\n```"
    )
    assert "trace content" in text.split("---")[0]
    for heading in ("### System prompt", "### User", "### Assistant", "### Final answer"):
        assert heading in text
    ids = [row["id"] for row in data["rows"]]
    assert re.findall(r"^## \d+\. `([^`]+)`$", text, re.MULTILINE) == ids
    assert text.count("- Label: none") == 3
    lines = re.findall(r'^\{"id": .*\}$', text, re.MULTILINE)
    assert lines == [json.dumps({"id": row_id, "keep": True, "reason": ""}) for row_id in ids]

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join([lines[0].replace('true, "reason": ""', 'false, "reason": "x"'), *lines[1:]])
    )
    assert output_json(label(dataset, labels))["label"] == {
        "file": str(labels),
        "labels": 3,
        "applied": 3,
        "unchanged": 0,
        "kept": 2,
        "dropped": 1,
    }


def test_a_review_file_renders_tool_calls_and_results_and_notes_what_it_cuts(
    tmp_path: Path,
) -> None:
    system = "\n".join(f"rule {index}" for index in range(30))
    arguments = json.dumps({"key": "k1", "note": "y" * 3000})
    row = {
        "id": "long-row",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "look up ```k1```"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "check k1 first",
                "weight": 0,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "lookup", "arguments": arguments}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 5000},
            {"role": "assistant", "content": "k1 is set."},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    target, review = tmp_path / "ds", tmp_path / "review.md"
    assert invoke("import", str(target), "--file", str(rows)).exit_code == 0
    sample(target, "--markdown", str(review))
    text = review.read_text()
    assert f"_{len(system):,} characters, 30 lines; the first 34 characters are shown._" in text
    assert "rule 4\n```" in text and "rule 5" not in text
    assert "````text\nlook up ```k1```\n````" in text
    pretty = json.dumps(json.loads(arguments), indent=2)
    assert (
        "### Assistant (not trained: weight 0)\n\nReasoning:\n\n```text\ncheck k1 first\n```\n\n"
        f"Tool call `lookup` (`c1`):\n\n```json\n{pretty[:2000]}\n```\n\n"
        f"_Cut: {len(pretty):,} characters, 4 lines in all; the first 2,000 characters are shown"
    ) in text
    assert "### Tool result: `lookup` (`c1`)\n\n```text\n" + "x" * 2000 + "\n```" in text
    assert "_Cut: 5,000 characters, 1 line in all; the first 2,000 characters are shown" in text
    assert "### Final answer\n\n```text\nk1 is set.\n```" in text


def test_a_review_file_holds_the_rows_split_and_count_pick(dataset: Path, tmp_path: Path) -> None:
    review = tmp_path / "train.md"
    data = sample(
        dataset, "--split", "train", "--count", "2", "--seed", "7", "--markdown", str(review)
    )
    text = review.read_text()
    ids = re.findall(r"^## \d+\. `([^`]+)`$", text, re.MULTILINE)
    assert ids == [row["id"] for row in data["rows"]] and len(ids) == 2
    assert re.findall(r"^- Split: (\S+)$", text, re.MULTILINE) == ["train", "train"]
    assert "Sampled from the train split with seed 7." in text


def test_a_review_file_copies_a_rows_current_label(dataset: Path, tmp_path: Path) -> None:
    dropped, reason = FIXTURE_IDS[0], "off topic ![x](http://attacker.example/p.png)"
    file = label_file(tmp_path, {"id": dropped, "keep": False, "reason": reason})
    output_json(label(dataset, file))
    review = tmp_path / "review.md"
    sample(dataset, "--count", "100", "--include-dropped", "--markdown", str(review))
    text = review.read_text()
    assert f"- Label: drop (manual): `{reason}`\n" in text
    assert json.dumps({"id": dropped, "keep": False, "reason": reason}) in text


def test_a_review_file_does_not_copy_a_judge_label_into_its_labels_line(
    dataset: Path, tmp_path: Path
) -> None:
    judged = FIXTURE_IDS[0]
    apply_labels(DatasetDir(dataset), [Label(id=judged, keep=True, reason="on task", by="judge")])
    review = tmp_path / "review.md"
    sample(dataset, "--count", "100", "--markdown", str(review))
    text = review.read_text()
    section = text[text.index(f". `{judged}`\n") :].split("\n---\n")[0]
    assert "- Label: keep (judge): `on task`" in section
    assert f"### Labels line\n\n{JUDGE_LINE_NOTE}\n\n```json\n" in section
    assert json.dumps({"id": judged, "keep": True, "reason": ""}) in section
    assert text.count(JUDGE_LINE_NOTE) == 1
    assert "keeps its current label (unlabeled rows stay unlabeled)" in text


def test_every_labels_line_of_a_review_file_with_a_judge_drop_is_a_valid_label(
    dataset: Path, tmp_path: Path
) -> None:
    judged = FIXTURE_IDS[0]
    apply_labels(DatasetDir(dataset), [Label(id=judged, keep=False, reason="off task", by="judge")])
    review = tmp_path / "review.md"
    sample(dataset, "--count", "100", "--include-dropped", "--markdown", str(review))
    lines = re.findall(r'^\{"id": .*\}$', review.read_text(), re.MULTILINE)
    assert len(lines) == len(FIXTURE_IDS)
    assert json.dumps({"id": judged, "keep": True, "reason": ""}) in lines
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(lines) + "\n")
    assert output_json(label(dataset, labels))["label"]["dropped"] == 0


def test_sample_replaces_only_a_review_file_it_wrote(dataset: Path, tmp_path: Path) -> None:
    rubric, notes = tmp_path / "rubric.md", tmp_path / "notes.md"
    rubric.write_text("# Task\n\nKeep rows that answer the question.\n")
    notes.write_text("# Review: Q3 planning notes\n")
    for path in (rubric, notes):
        before = path.read_bytes()
        result = invoke("sample", str(dataset), "--markdown", str(path))
        assert (result.exit_code, result.stdout) == (2, "")
        assert "exists and is not a review file" in _ANSI.sub("", result.stderr)
        assert path.read_bytes() == before
    review = tmp_path / "review.md"
    sample(dataset, "--count", "1", "--markdown", str(review))
    data = sample(dataset, "--count", "2", "--markdown", str(review))
    assert data["notes"][0].startswith(f"replaced the earlier review file {review.resolve()} ")
    assert review.read_text().startswith("# Review: 2 of ")


def test_sample_writes_no_review_file_when_no_row_matches(tmp_path: Path) -> None:
    target, review = tmp_path / "ds", tmp_path / "review.md"
    result = invoke("import", str(target), "--file", str(TOOL_USE), "--test-fraction", "0")
    assert result.exit_code == 0, result.stderr
    data = sample(target, "--split", "test", "--markdown", str(review))
    assert (data["rows"], data["markdown_file"]) == ([], None)
    assert not review.exists()
    assert data["notes"] == [f"no row matched the sample, so {review.resolve()} was not written"]
    assert data["hint"].startswith("review these rows with the user, then record")


def test_sample_refuses_a_markdown_path_that_is_a_directory_or_not_markdown(
    dataset: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "review.md"
    folder.mkdir()
    rows = (dataset / "rows.jsonl").read_bytes()
    for path, error in (
        (folder, "is a directory"),
        (dataset / "rows.jsonl", "--markdown must name a .md file"),
    ):
        result = invoke("sample", str(dataset), "--markdown", str(path))
        assert result.exit_code == 2
        assert result.stdout == ""
        assert error in _ANSI.sub("", result.stderr)
    assert (dataset / "rows.jsonl").read_bytes() == rows
    assert list(folder.iterdir()) == []


# --- read-only mode ---------------------------------------------------------------------


def test_label_and_sample_are_allowed_read_only(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    file = label_file(tmp_path, {"id": FIXTURE_IDS[0], "keep": False, "reason": "noise"})
    assert output_json(label(dataset, file))["dropped"] == 1
    assert sample(dataset, "--include-dropped")["eligible"] == len(FIXTURE_IDS)


# --- closing summaries ------------------------------------------------------------------


def test_label_and_sample_end_with_one_summary_line_of_their_result(
    dataset: Path, tmp_path: Path
) -> None:
    from pup_finetune.cli._summary import summarize

    file = label_file(
        tmp_path,
        {"id": FIXTURE_IDS[0], "keep": False, "reason": "synthetic drop"},
        {"id": FIXTURE_IDS[1], "keep": True},
    )
    labeled = label(dataset, file)
    data = output_json(labeled)
    found = summarize("dataset label", data)
    assert found is not None
    assert labeled.stderr.splitlines()[-1] == found.line
    assert found.line == (
        f"recorded 2 labels from {file} (1 keep, 1 drop, 0 unchanged); the dataset keeps "
        f"{len(FIXTURE_IDS) - 1} of {len(FIXTURE_IDS)} rows, 1 dropped"
    )

    sampled = invoke("sample", str(dataset), "--count", "3", "--split", "train")
    data = output_json(sampled)
    found = summarize("dataset sample", data)
    assert found is not None
    assert sampled.stderr.splitlines()[-1] == found.line
    assert found.line.startswith(
        f"sampled 3 of {data['eligible']} rows from the train split of {data['dataset_dir']} (seed "
    )
    assert "rows labeled drop left out" in found.line
    # Never the rows' content: only counts and where the full rows are.
    assert all(row["id"] not in found.line for row in data["rows"])
    assert len(found.lines) <= 8 and "\n" not in found.line
