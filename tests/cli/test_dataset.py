# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset import` and `dataset show`, end to end through the Typer app."""

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

ROWS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rows"
TOOL_USE = ROWS_DIR / "tool_use.jsonl"
INVALID = ROWS_DIR / "invalid.jsonl"
INVALID_EXPECTED: list[dict[str, Any]] = json.loads(
    (ROWS_DIR / "invalid.expected.json").read_text()
)
TOOL_USE_COUNT = len(TOOL_USE.read_text().splitlines())

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["dataset", *args])


def output_json(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def import_rows(directory: Path, *files: Path, extra: tuple[str, ...] = ()) -> Result:
    args: list[str] = ["import", str(directory)]
    for path in files:
        args += ["--file", str(path)]
    return invoke(*args, *extra)


@pytest.fixture
def mixed_file(tmp_path: Path) -> Path:
    path = tmp_path / "mixed.jsonl"
    path.write_text(TOOL_USE.read_text() + INVALID.read_text())
    return path


# --- import -----------------------------------------------------------------------------


def test_import_emits_counts_and_next_command(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    result = import_rows(target, TOOL_USE)
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["dataset_dir"] == str(target.resolve())
    assert data["accepted"] == TOOL_USE_COUNT
    assert data["excluded"] == 0
    assert data["excluded_by_code"] == {}
    assert set(data["splits"]) == {"train", "validation", "test"}
    assert sum(data["splits"].values()) == TOOL_USE_COUNT
    assert isinstance(data["warnings"], list)
    assert data["import"] == {
        "file": str(TOOL_USE),
        "added": TOOL_USE_COUNT,
        "unchanged": 0,
        "excluded": 0,
    }
    assert data["split_conflicts"] == {"groups": 0, "rows": 0, "row_ids": {}, "labels_file": None}
    # Nothing is labeled yet: review a sample; labeling is a step, with a file to write first.
    quoted = shlex.quote(str(target))
    assert data["next_command"] == f"pup finetune dataset sample {quoted} --count 20"
    assert [step["command"] for step in data["next_steps"]] == [
        f"pup finetune dataset label {quoted} --file <LABELS_FILE>",
        f"pup finetune dataset triage {quoted} --rubric <RUBRIC_FILE> --max-rows 20",
    ]


def test_import_records_invalid_rows_with_their_codes(tmp_path: Path, mixed_file: Path) -> None:
    target = tmp_path / "ds"
    result = import_rows(target, mixed_file)
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["accepted"] == TOOL_USE_COUNT
    assert data["excluded"] == len(INVALID_EXPECTED)
    expected_codes: dict[str, int] = {}
    for entry in INVALID_EXPECTED:
        expected_codes[entry["code"]] = expected_codes.get(entry["code"], 0) + 1
    assert data["excluded_by_code"] == expected_codes
    records = [json.loads(line) for line in (target / "excluded.jsonl").read_text().splitlines()]
    assert [(r["row_id"], r["code"]) for r in records] == [
        (e["id"], e["code"]) for e in INVALID_EXPECTED
    ]
    first_invalid_line = TOOL_USE_COUNT + 1
    assert records[0]["detail"].startswith(f"mixed.jsonl:{first_invalid_line}: ")


def test_import_is_idempotent(tmp_path: Path, mixed_file: Path) -> None:
    target = tmp_path / "ds"
    first = output_json(import_rows(target, mixed_file))
    files = {p.name: p.read_bytes() for p in target.iterdir()}
    again = import_rows(target, mixed_file)
    assert again.exit_code == 0, again.stderr
    second = output_json(again)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == files
    for key in ("accepted", "excluded", "excluded_by_code", "splits", "warnings"):
        assert second[key] == first[key]
    assert second["import"]["added"] == 0
    assert second["import"]["unchanged"] == TOOL_USE_COUNT


def test_import_appends_new_rows_and_keeps_splits(tmp_path: Path) -> None:
    lines = TOOL_USE.read_text().splitlines()
    head, full = tmp_path / "head.jsonl", tmp_path / "full.jsonl"
    head.write_text("\n".join(lines[:5]) + "\n")
    full.write_text("\n".join(lines) + "\n")
    target = tmp_path / "ds"
    import_rows(target, head)
    before = json.loads((target / "splits.json").read_text())["assignments"]
    result = output_json(import_rows(target, full))
    assert result["import"]["added"] == len(lines) - 5
    after = json.loads((target / "splits.json").read_text())["assignments"]
    assert {key: after[key] for key in before} == before


def test_import_excludes_duplicate_conversations(tmp_path: Path) -> None:
    first = json.loads(TOOL_USE.read_text().splitlines()[0])
    copy = {**first, "id": "row-weather-001-copy"}
    path = tmp_path / "dupes.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(copy) + "\n")
    data = output_json(import_rows(tmp_path / "ds", path))
    assert (data["accepted"], data["excluded"]) == (1, 1)
    assert data["excluded_by_code"] == {"duplicate_conversation": 1}


def test_import_rejects_a_known_id_with_different_content(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    import_rows(target, TOOL_USE)
    row = json.loads(TOOL_USE.read_text().splitlines()[0])
    row["messages"][-1]["content"] = "A different answer."
    changed = tmp_path / "changed.jsonl"
    changed.write_text(json.dumps(row) + "\n")
    result = import_rows(target, changed)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error [usage]: " in result.stderr
    assert "'row-weather-001' is already in the dataset with different content" in result.stderr


def test_import_rejects_changed_split_settings(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    import_rows(target, TOOL_USE, extra=("--seed", "7"))
    result = import_rows(target, TOOL_USE)
    assert result.exit_code == 2
    assert "Error [settings_conflict]: split settings changed" in result.stderr
    assert "--seed 7 --validation-fraction 0.1 --test-fraction 0.1" in result.stderr
    assert import_rows(target, TOOL_USE, extra=("--seed", "7")).exit_code == 0


def test_import_with_custom_split_settings(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    extra = ("--seed", "3", "--validation-fraction", "0", "--test-fraction", "0")
    result = import_rows(target, TOOL_USE, extra=extra)
    assert result.exit_code == 0, result.stderr  # warnings are not fatal
    data = output_json(result)
    assert data["splits"] == {"train": TOOL_USE_COUNT, "validation": 0, "test": 0}
    assert data["warnings"] == [
        "the validation split is empty, so training cannot select a checkpoint by validation loss",
        "the test split is empty, so there is nothing to evaluate on",
        "one group of rows that share a split holds 2 of 14 rows (14%, more than 10%) in 1 "
        "trace, so one split takes them all; a group that large across many traces is likely "
        "chained through requests they share (e.g. rows row-support-001, row-support-002)",
    ]
    settings = json.loads((target / "settings.json").read_text())
    assert settings["split"] == {"seed": 3, "validation_fraction": 0.0, "test_fraction": 0.0}


@pytest.mark.parametrize(
    "extra",
    [
        ("--validation-fraction", "1"),
        ("--test-fraction", "-0.1"),
        ("--validation-fraction", "0.5", "--test-fraction", "0.5"),
    ],
)
def test_import_rejects_bad_fractions(tmp_path: Path, extra: tuple[str, ...]) -> None:
    target = tmp_path / "ds"
    result = import_rows(target, TOOL_USE, extra=extra)
    assert result.exit_code == 2
    assert "fraction" in result.stderr
    assert not target.exists()


def test_import_requires_an_existing_file(tmp_path: Path) -> None:
    result = import_rows(tmp_path / "ds", tmp_path / "missing.jsonl")
    assert result.exit_code == 2


def test_import_refuses_a_non_dataset_directory(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("keep me")
    result = import_rows(tmp_path, TOOL_USE)
    assert result.exit_code == 2
    assert "not a dataset directory" in result.stderr


def test_import_creates_private_files(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    import_rows(target, TOOL_USE)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    for path in target.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_import_and_show_are_allowed_when_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    target = tmp_path / "ds"
    assert import_rows(target, TOOL_USE).exit_code == 0
    assert invoke("show", str(target)).exit_code == 0


# --- show -------------------------------------------------------------------------------


def test_show_emits_summary_settings_samples_and_next_command(
    tmp_path: Path, mixed_file: Path
) -> None:
    target = tmp_path / "ds"
    imported = output_json(import_rows(target, mixed_file))
    result = invoke("show", str(target))
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["dataset_dir"] == str(target.resolve())
    assert data["settings"] == {
        "source": {"kind": "import"},
        "split": {"seed": 42, "validation_fraction": 0.1, "test_fraction": 0.1},
    }
    assert data["summary"] == json.loads((target / "summary.json").read_text())
    assert data["summary"]["splits"] == imported["splits"]
    assert data["summary"]["excluded_by_code"] == imported["excluded_by_code"]
    assignments = json.loads((target / "splits.json").read_text())["assignments"]
    expected: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for line in (target / "rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row.get("source", {}).get("trace_id") or row["id"]  # as schema.split_key
        expected[assignments[key]].append(row["id"])
    assert data["sample_row_ids"] == {split: ids[:5] for split, ids in expected.items()}
    for split, ids in data["sample_row_ids"].items():
        assert len(ids) == min(5, imported["splits"][split])
    assert (
        data["next_command"] == f"pup finetune dataset sample {shlex.quote(str(target))} --count 20"
    )


def test_show_on_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    result = invoke("show", str(tmp_path / "missing"))
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "not a dataset directory" in result.stderr


def test_show_quotes_paths_in_next_command(tmp_path: Path) -> None:
    target = tmp_path / "my dataset"
    imported = output_json(import_rows(target, TOOL_USE))
    assert f" {shlex.quote(str(target))} " in imported["next_command"]
    shown = output_json(invoke("show", str(target)))
    assert shlex.split(shown["next_command"])[4] == str(target)
    for step in shown["next_steps"]:
        assert f" {shlex.quote(str(target))} " in step["command"]


def test_dataset_without_a_subcommand_shows_help() -> None:
    # Strip ANSI styling; Typer forces rich styling on CI (for example under GITHUB_ACTIONS).
    output = _ANSI.sub("", invoke().output)
    assert "import" in output
    assert "show" in output


# --- copies of one request ------------------------------------------------------------------


BOOKING = "Please book a room at the harbour hotel for {n} nights starting on 2026-03-{d:02d}."


def synthetic_row(row_id: str, text: str, trace_id: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"Done for {row_id}."},
        ],
        "source": {"trace_id": trace_id},
    }


def rows_file(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def copy_rows(first: int, count: int) -> list[dict[str, Any]]:
    """``count`` copies of one request (another number of nights each), a trace each."""
    return [
        synthetic_row(f"synthetic-copy-{n:03d}", BOOKING.format(n=n, d=n % 28 + 1), f"copy-{n:03d}")
        for n in range(first, first + count)
    ]


def other_rows(count: int) -> list[dict[str, Any]]:
    """``count`` unrelated requests, a trace each (told apart by letters: numbers fold)."""
    return [
        synthetic_row(
            f"synthetic-other-{n:03d}",
            "Tell me about item " + "".join(chr(ord("a") + int(d)) for d in str(n)),
            f"other-{n:03d}",
        )
        for n in range(count)
    ]


def test_copies_of_one_request_share_one_split(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    first = rows_file(tmp_path / "a.jsonl", [*copy_rows(0, 3), *other_rows(77)])
    data = output_json(import_rows(target, first))
    splits = json.loads((target / "splits.json").read_text())["assignments"]
    assert len({splits[f"copy-{n:03d}"] for n in range(3)}) == 1
    assert data["split_groups"] == 78
    assert data["split_conflicts"]["groups"] == 0
    # A later trace asking it again takes the same split.
    again = import_rows(target, rows_file(tmp_path / "b.jsonl", copy_rows(3, 1)))
    assert again.exit_code == 0, again.stderr
    splits = json.loads((target / "splits.json").read_text())["assignments"]
    assert splits["copy-003"] == splits["copy-000"]


def test_a_fixed_prompt_in_many_traces_keeps_a_split_per_trace_and_is_warned_about(
    tmp_path: Path,
) -> None:
    fixed = [synthetic_row(f"sum-{n}", "Summarize the above", f"t{n}") for n in range(6)]
    data = output_json(import_rows(tmp_path / "ds", rows_file(tmp_path / "a.jsonl", fixed)))
    assert data["split_groups"] == 6
    assert any(
        w.startswith("1 request repeats in more than 5 traces (5% of the dataset's traces, and")
        and "sum-0, sum-1, sum-2" in w
        for w in data["warnings"]
    )


def test_copies_recorded_apart_are_named_with_labels_to_drop_them(tmp_path: Path) -> None:
    target = tmp_path / "ds"
    # Two traces recorded in different splits, as an earlier grouping could leave them.
    first = [synthetic_row("a", BOOKING.format(n=1, d=2), "t1")]
    first.append(synthetic_row("b", "Something else entirely", "t2"))
    assert import_rows(target, rows_file(tmp_path / "a.jsonl", first)).exit_code == 0
    splits_path = target / "splits.json"
    stored = json.loads(splits_path.read_text())
    splits_path.write_text(json.dumps({**stored, "assignments": {"t1": "test", "t2": "train"}}))
    # A new row of t2 repeats t1's request: every trace is recorded, yet the copies are apart.
    linking = [synthetic_row("c", BOOKING.format(n=7, d=9), "t2")]
    result = import_rows(target, rows_file(tmp_path / "b.jsonl", linking))
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["split_conflicts"] == {
        "groups": 1,
        "rows": 1,
        "row_ids": {"test": ["a"]},
        "labels_file": "split_conflicts.jsonl",
    }
    assert any("1 kept row is a copy of a request" in w for w in data["warnings"])
    labels_file = target / "split_conflicts.jsonl"
    assert [json.loads(line) for line in labels_file.read_text().splitlines()] == [
        {
            "id": "a",
            "keep": False,
            "reason": "a copy of a request whose copies are mostly in the train split",
        }
    ]
    drop = data["next_steps"][0]["command"]
    assert data["next_steps"][0]["why"].startswith(
        "drop the 1 kept row outside its group's split (see split_conflicts)"
    )
    assert drop == f"pup finetune dataset label {shlex.quote(str(target))} --file " + shlex.quote(
        str(labels_file)
    )
    labeled = invoke(*shlex.split(drop)[3:])
    assert labeled.exit_code == 0, labeled.stderr
    assert output_json(labeled)["next_steps"][0]["command"] != drop
    assert not labels_file.exists()
    show = output_json(invoke("show", str(target)))
    assert show["summary"]["split_conflicts"]["groups"] == 0
