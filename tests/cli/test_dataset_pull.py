# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune dataset pull`, end to end over a fake Export API."""

from __future__ import annotations

import json
import re
import shlex
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from pup_finetune.cli import app
from pup_finetune.cli import dataset as dataset_cli
from pup_finetune.datadog.pup import Pup
from pup_finetune.datadog.spans import EXPORT_PATH
from pup_finetune.display import display_path
from tests.fakes.llmobs import FakeExportApi, fixture_pages, page, span_item
from tests.fakes.pup import FakePup

runner = CliRunner()

PAGE_2_CURSOR = "synthetic-cursor-page-2"
FIXTURE_ROWS = 4
FIXTURE_EXCLUSIONS = 3

UsePup = Callable[[Pup], None]

NOW_MS = 1767225600000
"""What the commands' clock reads: 2026-01-01T00:00:00Z."""
DAY_MS = 24 * 60 * 60 * 1000
WEEK_AGO_MS = NOW_MS - 7 * DAY_MS


class FakeClock:
    """A clock the test moves; reads seconds, like :func:`time.time`."""

    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms

    def __call__(self) -> float:
        return self.now_ms / 1000


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """The clock the dataset commands resolve ``now`` against."""
    fake = FakeClock(NOW_MS)
    monkeypatch.setattr(dataset_cli, "clock", fake)
    return fake


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run in an empty directory, so a relative dataset directory lands under it."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def use_pup(monkeypatch: pytest.MonkeyPatch) -> UsePup:
    """Make the dataset commands use a given Pup instead of the real one."""

    def install(pup: Pup) -> None:
        monkeypatch.setattr(dataset_cli, "Pup", lambda: pup)

    return install


@pytest.fixture
def api(use_pup: UsePup) -> FakeExportApi:
    """The fixture pages, served to the dataset commands."""
    export = FakeExportApi(fixture_pages())
    use_pup(export.install(FakePup()).client())
    return export


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["dataset", *args])


def output_json(result: Result) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def pull(directory: Path, *extra: str) -> Result:
    return invoke("pull", str(directory), "--ml-app", "synthetic-trip-planner", *extra)


def tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


SYSTEM = {"role": "system", "content": "You are a synthetic helper."}


def conversation(span_id: str, trace_id: str, *turns: str, **attributes: Any) -> dict[str, Any]:
    """A span item whose conversation is the system prompt, then user / assistant ``turns``."""
    messages = [SYSTEM] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(turns)
    ]
    return span_item(
        span_id,
        trace_id,
        input={"messages": messages[:-1]},
        output={"messages": messages[-1:]},
        **attributes,
    )


def serve(use_pup: UsePup, pages: list[dict[str, Any]]) -> FakeExportApi:
    export = FakeExportApi(pages)
    use_pup(export.install(FakePup()).client())
    return export


# --- what a pull reads and reports ---------------------------------------------------------


def fresh(*extra: str) -> Result:
    """A first pull into ``ds`` under the current directory."""
    return pull(Path("ds"), *extra)


def test_a_pull_explains_the_exclusion_codes(api: FakeExportApi) -> None:
    data = output_json(fresh())
    explanations = data["exclusion_explanations"]
    assert set(explanations) == set(data["excluded_by_code"])
    assert "--include-errors" in explanations["span_error"]
    assert all(isinstance(text, str) and text for text in explanations.values())


def test_a_pull_explains_unstructured_tool_messages(use_pup: UsePup) -> None:
    text_protocol = span_item(
        "synthetic-span-1",
        input={
            "messages": [
                {"role": "user", "content": "Weather in Northvale?"},
                {"role": "assistant", "content": "<call>get_weather Northvale</call>"},
                {"role": "tool", "content": "Sunny."},
            ]
        },
        output={"messages": [{"role": "assistant", "content": "It is sunny."}]},
    )
    use_pup(FakeExportApi([page([text_protocol])]).install(FakePup()).client())
    data = output_json(fresh())
    assert data["excluded_by_code"] == {"unstructured_tool_message": 1}
    assert "tool calls as text" in data["exclusion_explanations"]["unstructured_tool_message"]


def test_a_pull_with_include_errors_keeps_errored_calls(api: FakeExportApi) -> None:
    data = output_json(fresh("--include-errors"))
    assert "span_error" not in data["excluded_by_code"]
    assert data["excluded_by_code"]["no_assistant_message"] == 2


def test_a_pull_warns_when_max_candidates_stops_it_short(api: FakeExportApi) -> None:
    data = output_json(fresh("--max-candidates", "3"))
    assert data["pull"]["matched_spans"] == 3
    assert data["pull"]["stopped"] == "max_candidates"
    assert any("--max-candidates" in warning for warning in data["warnings"])


def test_a_pull_with_sample_random_reads_every_span_and_spreads_the_rows(
    api: FakeExportApi,
) -> None:
    data = output_json(fresh("--sample", "random", "--target-count", "3"))
    assert api.cursors == [None, PAGE_2_CURSOR]
    assert data["pull"]["stopped"] == "exhausted"
    rows = [json.loads(line) for line in Path("ds/rows.jsonl").read_text().splitlines()]
    traces = [row["source"]["trace_id"] for row in rows]
    assert len(traces) == len(set(traces)) == 3


def test_a_random_pull_cut_short_by_max_candidates_warns_it_samples_the_oldest_spans(
    api: FakeExportApi,
) -> None:
    data = output_json(fresh("--sample", "random", "--max-candidates", "3", "--target-count", "1"))
    assert data["pull"]["stopped"] == "max_candidates"
    (warning,) = [warning for warning in data["warnings"] if "oldest first" in warning]
    assert "--max-candidates 3" in warning
    assert "--from" not in warning  # the directory froze the window


def test_a_pull_reports_how_the_accepted_rows_spread_over_traces(use_pup: UsePup) -> None:
    busy = [conversation(f"span-busy-{n}", "busy", f"Q{n}?", f"A{n}.") for n in range(20)]
    quiet = [conversation(f"span-quiet-{n}", f"quiet-{n}", "Q?", f"A{n}.") for n in range(10)]
    pages = [page([*busy, *quiet])]

    def spread_warnings(data: dict[str, Any]) -> list[str]:
        return [warning for warning in data["warnings"] if "one trace gives" in warning]

    serve(use_pup, pages)
    data = output_json(pull(Path("a")))
    assert (data["accepted"], data["pull"]["accepted_traces"]) == (30, 11)
    assert data["pull"]["busiest_trace_rows"] == 20
    (warning,) = spread_warnings(data)
    assert "20 of the 30 accepted rows" in warning
    assert "--sample random --max-rows-per-trace N in a new directory" in warning
    # Not once --max-rows-per-trace is set, nor for a few rows.
    serve(use_pup, pages)
    capped = output_json(pull(Path("b"), "--max-rows-per-trace", "10"))
    assert (capped["accepted"], capped["pull"]["busiest_trace_rows"]) == (20, 10)
    assert spread_warnings(capped) == []
    serve(use_pup, pages)
    assert spread_warnings(output_json(pull(Path("c"), "--target-count", "19"))) == []


def test_max_candidates_equal_to_every_span_is_not_a_short_read(api: FakeExportApi) -> None:
    data = output_json(fresh("--max-candidates", "10"))
    assert data["pull"]["matched_spans"] == 10
    assert data["pull"]["stopped"] == "exhausted"
    assert not any("--max-candidates" in warning for warning in data["warnings"])


def test_a_pull_warns_when_no_span_matched(use_pup: UsePup) -> None:
    serve(use_pup, [page([])])
    data = output_json(fresh())
    assert data["pull"]["matched_spans"] == 0
    assert data["warnings"][0] == (
        "no LLM spans matched; check --ml-app, --span-name, --query, --tag and the time "
        "range (--from, --to) in a new directory (this one froze them)"
    )


def test_a_pull_warns_about_unreadable_spans_and_excludes_them(use_pup: UsePup) -> None:
    good = conversation("synthetic-span-1", "synthetic-trace-1", "Hello?", "Hi.")
    serve(use_pup, [page([good, {"id": "synthetic-span-2", "attributes": {}}])])
    data = output_json(fresh())
    assert data["pull"]["matched_spans"] == 1
    assert data["excluded_by_code"] == {"unreadable_span": 1}
    assert "unreadable_span" in data["exclusion_explanations"]
    assert data["pull"]["unreadable_spans"] == 1
    assert (
        "1 spans could not be read and were skipped (excluded as unreadable_span)"
        in data["warnings"]
    )
    assert not any("no LLM spans matched" in warning for warning in data["warnings"])


def test_a_pull_blames_unreadable_spans_when_none_could_be_read(use_pup: UsePup) -> None:
    serve(use_pup, [page(["not an object", {"id": "synthetic-span-2", "attributes": {}}])])
    data = output_json(fresh())
    assert data["pull"]["matched_spans"] == 0
    assert data["warnings"][0] == (
        "none of the 2 spans returned could be read (they lack the fields of an LLM span, "
        "such as a trace id), so there was nothing to convert"
    )
    assert not any("no LLM spans matched" in warning for warning in data["warnings"])


@pytest.mark.parametrize("bad", ["env", "env:", ":synthetic"])
def test_a_malformed_tag_is_a_usage_error(api: FakeExportApi, bad: str) -> None:
    result = fresh("--tag", bad)
    assert result.exit_code == 2
    assert "KEY:VALUE" in result.stderr
    assert api.requests == []


def test_a_tag_given_two_values_is_a_usage_error(api: FakeExportApi) -> None:
    result = fresh("--tag", "env:a", "--tag", "env:b")
    assert result.exit_code == 2
    assert "twice" in result.stderr


def test_there_is_no_dataset_preview(api: FakeExportApi) -> None:
    result = invoke("preview", "--ml-app", "synthetic-trip-planner")
    assert result.exit_code == 2
    assert "No such command 'preview'" in _ANSI.sub("", result.stderr)
    assert api.requests == []


def test_pup_auth_failures_surface_with_the_pup_hint(use_pup: UsePup) -> None:
    fake = FakePup().on("auth", "status", returncode=1, stderr="Error: 401 Unauthorized")
    use_pup(fake.on("api", EXPORT_PATH, returncode=1, stderr="Error: 401 Unauthorized").client())
    result = fresh()
    assert result.exit_code == 1
    assert "401 Unauthorized" in result.stderr
    assert "pup auth login" in result.stderr
    assert result.stdout == ""


def test_a_pull_is_allowed_read_only(
    api: FakeExportApi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    result = pull(tmp_path / "ds")
    assert result.exit_code == 0, result.stderr


# --- pull -----------------------------------------------------------------------------------


def test_pull_end_to_end(api: FakeExportApi, tmp_path: Path) -> None:
    target = tmp_path / "ds"
    result = pull(target, "--tag", "env:synthetic")
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["dataset_dir"] == display_path(target)
    assert (data["accepted"], data["excluded"]) == (FIXTURE_ROWS, FIXTURE_EXCLUSIONS)
    assert data["excluded_by_code"] == {
        "no_assistant_message": 1,
        "non_text_content": 1,
        "span_error": 1,
    }
    assert set(data["exclusion_explanations"]) == set(data["excluded_by_code"])
    assert sum(data["splits"].values()) == FIXTURE_ROWS
    assert data["pull"]["matched_spans"] == 10
    assert data["pull"]["added"] == FIXTURE_ROWS
    assert data["pull"]["pages_fetched"] == 2
    assert (
        data["next_command"]
        == f"pup finetune dataset sample {shlex.quote(display_path(target))} --count 20"
    )
    assert data["next_steps"][0]["command"].endswith(" --file <LABELS_FILE>")
    assert "review" in data["hint"]

    settings = json.loads((target / "settings.json").read_text())
    assert settings["source"]["kind"] == "datadog"
    assert settings["source"]["filter"]["ml_app"] == "synthetic-trip-planner"
    assert settings["source"]["filter"]["tags"] == {"env": "synthetic"}
    assert (settings["source"]["reasoning"], settings["source"]["include_errors"]) == (
        "omit",
        False,
    )
    rows = (target / "rows.jsonl").read_text().splitlines()
    assert len(rows) == FIXTURE_ROWS
    excluded = [json.loads(line) for line in (target / "excluded.jsonl").read_text().splitlines()]
    assert {record["code"] for record in excluded} == set(data["excluded_by_code"])
    assert sorted(p.name for p in (target / "raw").iterdir() if not p.name.startswith(".")) == [
        "cursor.json",
        "page-0001.json",
        "page-0002.json",
    ]

    show = invoke("show", str(target))
    assert show.exit_code == 0, show.stderr
    assert output_json(show)["summary"]["accepted"] == FIXTURE_ROWS


def test_pull_files_are_private(api: FakeExportApi, tmp_path: Path) -> None:
    target = tmp_path / "ds"
    assert pull(target).exit_code == 0
    assert mode(target) == 0o700
    for path in target.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert mode(path) == expected, path


def test_an_interrupted_pull_resumes_without_refetching(api: FakeExportApi, tmp_path: Path) -> None:
    target = tmp_path / "ds"
    api.failures[PAGE_2_CURSOR] = 1
    failed = pull(target)
    assert failed.exit_code == 1
    assert "502 Bad Gateway" in failed.stderr
    assert f"pages read so far are cached in {target / 'raw'}" in failed.stderr
    # The hint's command takes no flag: the directory's frozen settings fill them in.
    assert "`pup finetune dataset pull ds` resumes" in failed.stderr
    assert api.cursors == [None, PAGE_2_CURSOR]
    assert (target / "raw" / "page-0001.json").is_file()

    result = invoke("pull", "ds")
    assert result.exit_code == 0, result.stderr
    # Page 1 came from the cache; only page 2 was fetched again.
    assert api.cursors == [None, PAGE_2_CURSOR, PAGE_2_CURSOR]
    data = output_json(result)
    assert (data["pull"]["pages_cached"], data["pull"]["pages_fetched"]) == (1, 1)
    assert (data["accepted"], data["excluded"]) == (FIXTURE_ROWS, FIXTURE_EXCLUSIONS)


def test_rerunning_a_finished_pull_changes_nothing(api: FakeExportApi, tmp_path: Path) -> None:
    target = tmp_path / "ds"
    assert pull(target).exit_code == 0
    files = ("rows.jsonl", "excluded.jsonl", "summary.json", "splits.json")
    before = {name: (target / name).read_bytes() for name in files}
    result = pull(target)
    assert result.exit_code == 0, result.stderr
    assert api.cursors == [None, PAGE_2_CURSOR]
    data = output_json(result)
    assert (data["pull"]["added"], data["pull"]["unchanged"]) == (0, FIXTURE_ROWS)
    assert (data["pull"]["pages_cached"], data["pull"]["pages_fetched"]) == (2, 0)
    assert data["excluded"] == FIXTURE_EXCLUSIONS
    assert data["pull"]["retired"] == 0
    assert {name: (target / name).read_bytes() for name in files} == before


def test_pull_stops_at_the_target_count_and_reads_further_with_a_higher_one(
    api: FakeExportApi, tmp_path: Path
) -> None:
    target = tmp_path / "ds"
    data = output_json(pull(target, "--target-count", "2"))
    assert api.cursors == [None]
    assert data["accepted"] == 2
    assert data["pull"]["stopped"] == "target_count"
    assert data["pull"]["rows_over_target"] == 0

    # A rerun without --target-count keeps the last pull's; a higher one reads further.
    more = output_json(pull(target, "--target-count", "200"))
    assert api.cursors == [None, PAGE_2_CURSOR]
    assert more["accepted"] == FIXTURE_ROWS
    assert (more["pull"]["added"], more["pull"]["unchanged"]) == (2, 2)
    # The directory froze the filter and time range, so widening them needs a new one.
    assert any("time range in a new directory" in warning for warning in more["warnings"])


def test_a_rerun_retires_a_stored_prefix_of_a_conversation_on_a_later_page(
    use_pup: UsePup, tmp_path: Path
) -> None:
    pages = [
        page([conversation("s1", "T", "Q1", "A1")], after="cursor-2"),
        page([conversation("s2", "T", "Q1", "A1", "Q2", "A2")]),
    ]
    api = serve(use_pup, pages)
    target = tmp_path / "ds"
    first = output_json(pull(target, "--target-count", "1"))
    assert first["pull"]["stopped"] == "target_count"
    assert api.cursors == [None]

    result = pull(target, "--target-count", "200")
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    rows = [json.loads(line)["id"] for line in (target / "rows.jsonl").read_text().splitlines()]
    assert rows == ["T:s2"]
    excluded = [json.loads(line) for line in (target / "excluded.jsonl").read_text().splitlines()]
    assert [(record["row_id"], record["code"]) for record in excluded] == [
        ("T:s1", "contained_conversation")
    ]
    assert "T:s2" in excluded[0]["detail"]
    assert (data["accepted"], data["excluded"]) == (1, 1)
    assert data["pull"]["retired"] == 1
    assert "contained_conversation" in data["exclusion_explanations"]

    again = output_json(pull(target))
    assert (again["accepted"], again["excluded"], again["pull"]["retired"]) == (1, 1, 0)

    # The first, smaller pull again: it stops before page 2, but does not add T:s1 back.
    lower = output_json(pull(target, "--target-count", "1"))
    assert (lower["accepted"], lower["excluded"], lower["pull"]["added"]) == (1, 1, 0)
    rows = [json.loads(line)["id"] for line in (target / "rows.jsonl").read_text().splitlines()]
    assert rows == ["T:s2"]


def test_a_dropped_row_s_drop_carries_to_the_longer_row_a_rerun_retires_it_into(
    use_pup: UsePup, tmp_path: Path
) -> None:
    pages = [
        page([conversation("s1", "T", "Q1", "A1")], after="cursor-2"),
        page([conversation("s2", "T", "Q1", "A1", "Q2", "A2")]),
    ]
    serve(use_pup, pages)
    target = tmp_path / "ds"
    output_json(pull(target, "--target-count", "1"))
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "T:s1", "keep": False, "reason": "A1 is wrong"}) + "\n")
    assert invoke("label", str(target), "--file", str(labels)).exit_code == 0

    result = pull(target, "--target-count", "200")
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["pull"]["retired"] == 1
    assert data["pipeline"]["rows"] == {"accepted": 1, "kept": 0, "dropped": 1, "unlabeled": 0}
    carried = json.loads((target / "labels.jsonl").read_text().splitlines()[-1])
    assert (carried["id"], carried["keep"], carried["by"]) == ("T:s2", False, "manual")
    assert carried["carried_from"] == "T:s1"
    assert "T:s1" in carried["reason"] and "A1 is wrong" in carried["reason"]
    assert data["pull"]["carried_drops"] == [{"row_id": "T:s2", "from": "T:s1"}]
    assert any("T:s2" in note for note in data["notes"])

    again = output_json(pull(target))
    assert again["pull"]["carried_drops"] == []
    assert len((target / "labels.jsonl").read_text().splitlines()) == 2


def test_a_kept_row_s_label_does_not_carry_to_the_longer_row_that_retires_it(
    use_pup: UsePup, tmp_path: Path
) -> None:
    pages = [
        page([conversation("s1", "T", "Q1", "A1")], after="cursor-2"),
        page([conversation("s2", "T", "Q1", "A1", "Q2", "A2")]),
    ]
    serve(use_pup, pages)
    target = tmp_path / "ds"
    output_json(pull(target, "--target-count", "1"))
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": "T:s1", "keep": True}) + "\n")
    assert invoke("label", str(target), "--file", str(labels)).exit_code == 0

    data = output_json(pull(target, "--target-count", "200"))
    # Its new turns are unreviewed, so the longer row is left to review.
    assert data["pipeline"]["rows"] == {"accepted": 1, "kept": 1, "dropped": 0, "unlabeled": 1}
    assert data["pull"]["carried_drops"] == []


def test_a_person_s_keep_on_the_row_a_dropped_row_retires_into_stands_with_a_warning(
    use_pup: UsePup, tmp_path: Path
) -> None:
    # A1 continues on page 2 into the conversation B already holds.
    pages = [
        page(
            [conversation("b1", "B", "Q1", "A1", "Q2", "A2"), conversation("a1", "A", "Q1", "A1")],
            after="cursor-2",
        ),
        page([conversation("a2", "A", "Q1", "A1", "Q2", "A2")]),
    ]
    serve(use_pup, pages)
    target = tmp_path / "ds"
    output_json(pull(target, "--target-count", "2"))
    labels = tmp_path / "labels.jsonl"
    lines = [
        {"id": "B:b1", "keep": True, "reason": "complete answer"},
        {"id": "A:a1", "keep": False, "reason": "A1 is wrong"},
    ]
    labels.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert invoke("label", str(target), "--file", str(labels)).exit_code == 0

    result = pull(target, "--target-count", "200")
    assert result.exit_code == 0, result.stderr
    data = output_json(result)
    assert data["pipeline"]["rows"] == {"accepted": 1, "kept": 1, "dropped": 0, "unlabeled": 0}
    assert data["pull"]["carried_drops"] == []
    assert data["pull"]["drops_not_carried"] == [
        {
            "row_id": "B:b1",
            "kept_on": "B:b1",
            "kept": "complete answer",
            "from": "A:a1",
            "dropped": "A1 is wrong",
        }
    ]
    [warning] = [text for text in data["warnings"] if "A:a1" in text]
    assert all(part in warning for part in ("B:b1", "complete answer", "A1 is wrong"))
    assert f"warning: {warning}" in result.stderr


@pytest.mark.parametrize(
    "changed",
    [
        ("--reasoning", "preserve"),
        ("--include-errors",),
        ("--tag", "env:other"),
        ("--from", "now-30d"),
        ("--from", "1766620800000"),
        ("--to", "now-1h"),
        ("--seed", "7"),
        ("--sample", "random"),
        ("--max-rows-per-trace", "1"),
    ],
)
def test_pulling_with_other_settings_is_rejected(
    api: FakeExportApi, tmp_path: Path, changed: tuple[str, ...]
) -> None:
    target = tmp_path / "ds"
    assert pull(target).exit_code == 0
    requests = len(api.requests)
    result = pull(target, *changed)
    assert result.exit_code == 2
    # The error names the flag and the value the directory froze.
    assert result.stderr.startswith(f"Error [settings_conflict]: ds froze {changed[0]} ")
    # Rejected before anything was fetched.
    assert len(api.requests) == requests


def test_pull_refuses_a_non_empty_directory_that_is_not_a_dataset(
    api: FakeExportApi, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_text("not a dataset")
    result = pull(tmp_path)
    assert result.exit_code == 2
    assert "not a dataset directory" in result.stderr
    assert api.requests == []


def test_pull_into_an_imported_dataset_is_rejected(api: FakeExportApi, tmp_path: Path) -> None:
    rows = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
    target = tmp_path / "ds"
    assert invoke("import", str(target), "--file", str(rows)).exit_code == 0
    result = pull(target)
    assert result.exit_code == 2
    # The cause is in the message, not only in the hint.
    assert result.stderr.startswith(
        "Error [settings_conflict]: ds was built by `dataset import`; pull into a new DIR\n"
    )
    assert api.requests == []


@pytest.mark.parametrize("bad", ["yesterday", "²", "2026-13-01", "last week"])
def test_a_time_that_is_no_accepted_form_is_a_usage_error_before_anything_is_frozen(
    api: FakeExportApi, tmp_path: Path, bad: str
) -> None:
    result = pull(tmp_path / "ds", "--from", bad)
    assert result.exit_code == 2
    assert result.stderr.startswith(f"Error [usage]: cannot read --from {bad!r}\n")
    assert "ISO 8601" in result.stderr and "7d, 24h" in result.stderr
    assert "resume" not in result.stderr
    assert api.requests == []
    assert not (tmp_path / "ds").exists()  # nothing frozen to poison the directory


@pytest.mark.parametrize("bare", ["1728000000", "20261001", "7"])
def test_a_bare_number_too_small_for_epoch_ms_is_refused_before_anything_is_frozen(
    api: FakeExportApi, tmp_path: Path, bare: str
) -> None:
    for result in (pull(tmp_path / "ds", "--from", bare), pull(tmp_path / "ds", "--to", bare)):
        assert result.exit_code == 2
        assert "Error [usage]: --" in result.stderr
        assert "too small for epoch milliseconds" in result.stderr
        assert "Hint: " in result.stderr
    assert api.requests == []
    assert not (tmp_path / "ds").exists()


# --- the time range -------------------------------------------------------------------------


def sent_windows(api: FakeExportApi) -> list[tuple[object, object]]:
    """The ``from`` and ``to`` of every request sent to the Export API, epoch ms read back
    from the strings it is sent as."""
    filters = [body["data"]["attributes"]["filter"] for body in api.requests]
    return [(_sent_bound(f.get("from")), _sent_bound(f.get("to"))) for f in filters]


def _sent_bound(value: object) -> object:
    assert not isinstance(value, int), "the Export API takes a time bound only as a string"
    return int(value) if isinstance(value, str) and value.isdigit() else value


def test_a_pull_freezes_the_resolved_window_and_the_expressions(
    api: FakeExportApi, tmp_path: Path
) -> None:
    target = tmp_path / "ds"
    result = pull(target, "--from", "now-1d", "--to", "now-1h")
    assert result.exit_code == 0, result.stderr
    window = (NOW_MS - DAY_MS, NOW_MS - 60 * 60 * 1000)
    assert sent_windows(api) == [window, window]
    source = json.loads((target / "settings.json").read_text())["source"]
    assert (source["filter"]["from"], source["filter"]["to"]) == window
    assert (source["from_expr"], source["to_expr"]) == ("now-1d", "now-1h")
    time_range = output_json(result)["time_range"]
    assert (time_range["from"], time_range["to"]) == window
    assert (time_range["from_expr"], time_range["to_expr"]) == ("now-1d", "now-1h")


def test_a_pull_resumed_later_reads_the_frozen_window(
    api: FakeExportApi, tmp_path: Path, clock: FakeClock
) -> None:
    target = tmp_path / "ds"
    api.failures[PAGE_2_CURSOR] = 1
    assert pull(target).exit_code == 1
    clock.now_ms += 3 * DAY_MS
    result = pull(target)
    assert result.exit_code == 0, result.stderr
    assert api.cursors == [None, PAGE_2_CURSOR, PAGE_2_CURSOR]
    assert sent_windows(api) == [(WEEK_AGO_MS, NOW_MS)] * 3
    assert output_json(result)["time_range"]["to"] == NOW_MS
    assert output_json(result)["pull"]["pages_cached"] == 1


def test_a_finished_pull_rerun_later_with_a_higher_target_reads_the_same_window(
    api: FakeExportApi, tmp_path: Path, clock: FakeClock
) -> None:
    target = tmp_path / "ds"
    assert pull(target, "--target-count", "2").exit_code == 0
    clock.now_ms += 30 * DAY_MS
    result = pull(target, "--target-count", "200")
    assert result.exit_code == 0, result.stderr
    assert api.cursors == [None, PAGE_2_CURSOR]
    assert sent_windows(api) == [(WEEK_AGO_MS, NOW_MS)] * 2


def test_absolute_times_are_sent_unchanged(api: FakeExportApi, tmp_path: Path) -> None:
    target = tmp_path / "ds"
    result = pull(target, "--from", "2025-12-01T00:00:00Z", "--to", "1767225500000")
    assert result.exit_code == 0, result.stderr
    assert sent_windows(api)[0] == ("2025-12-01T00:00:00Z", 1767225500000)
    time_range = output_json(result)["time_range"]
    assert (time_range["from_utc"], time_range["to_utc"]) == (None, "2025-12-31T23:58:20.000Z")


@pytest.mark.parametrize(
    "bad", ["now-7", "now-7y", "now-d", "now+1h", "now-1M", "7y", "3 fortnights"]
)
def test_an_unreadable_relative_time_is_a_usage_error(
    api: FakeExportApi, tmp_path: Path, bad: str
) -> None:
    for result in (pull(tmp_path / "ds", "--from", bad), pull(tmp_path / "ds", "--to", bad)):
        assert result.exit_code == 2
        assert "Error [usage]: cannot read --" in result.stderr
    assert api.requests == []
    assert not (tmp_path / "ds").exists()


# --- flagless reruns ------------------------------------------------------------------------


def test_a_rerun_needs_no_flag_and_reads_the_window_the_first_pull_froze(
    api: FakeExportApi, tmp_path: Path, clock: FakeClock
) -> None:
    first = pull(tmp_path / "ds", "--from", "2d", "--span-name", "chat", "--include-errors")
    assert first.exit_code == 0, first.stderr
    window = (NOW_MS - 2 * DAY_MS, NOW_MS)
    assert output_json(first)["window"] == {
        "from": "2025-12-30T00:00:00.000Z",
        "to": "2026-01-01T00:00:00.000Z",
        "frozen": False,
    }
    # pup's `2d` is frozen as the expression `now-2d`.
    source = json.loads((tmp_path / "ds" / "settings.json").read_text())["source"]
    assert source["from_expr"] == "now-2d"

    clock.now_ms += 3 * DAY_MS
    rerun = invoke("pull", "ds")  # run from tmp_path: the directory as typed
    assert rerun.exit_code == 0, rerun.stderr
    data = output_json(rerun)
    assert data["window"] == {**output_json(first)["window"], "frozen": True}
    assert "reading the window the first pull froze: 2025-12-30T00:00:00.000Z to " in rerun.stderr
    assert "ds has frozen settings: using its --ml-app, --span-name, --from, --include-errors" in (
        rerun.stderr
    )
    filters = [body["data"]["attributes"]["filter"] for body in api.requests]
    assert {f["span_name"] for f in filters} == {"chat"}
    assert sent_windows(api)[-1] == window
    assert data["pipeline"]["rows"]["accepted"] == data["accepted"]
    # `now-2d` and `2d` are the same expression, so the window is the frozen one too.
    same = pull(tmp_path / "ds", "--from", "now-2d", "--span-name", "chat", "--include-errors")
    assert same.exit_code == 0, same.stderr
    assert output_json(same)["window"]["frozen"] is True


def test_a_rerun_with_another_value_than_the_frozen_one_still_fails(
    api: FakeExportApi, tmp_path: Path
) -> None:
    assert pull(tmp_path / "ds", "--span-name", "chat").exit_code == 0
    requests = len(api.requests)
    result = invoke("pull", "ds", "--span-name", "other", "--from", "now-1d")
    assert result.exit_code == 2
    assert result.stderr == (
        "Error [settings_conflict]: ds froze --span-name chat, --from 7d on its first "
        "pull\nHint: leave those flags out to use the frozen values, or pull into a new "
        "directory for other settings\n"
    )
    assert len(api.requests) == requests
    # The frozen value given again, in another form, is no conflict.
    assert invoke("pull", "ds", "--span-name", "chat", "--from", "7d").exit_code == 0


def test_a_flagless_pull_into_an_imported_dataset_is_a_settings_conflict(
    api: FakeExportApi, tmp_path: Path
) -> None:
    rows = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"
    assert invoke("import", "ds", "--file", str(rows)).exit_code == 0
    result = invoke("pull", "ds")
    assert result.exit_code == 2
    assert result.stderr.startswith(
        "Error [settings_conflict]: ds was built by `dataset import`; pull into a new DIR\n"
    )
    assert api.requests == []


def test_a_pull_freezes_the_sample_and_the_cap_and_a_rerun_takes_them(
    api: FakeExportApi, tmp_path: Path
) -> None:
    first = pull(tmp_path / "ds", "--sample", "random", "--max-rows-per-trace", "1")
    assert first.exit_code == 0, first.stderr
    # Trace synthetic-trace-0101 has two rows.
    assert output_json(first)["excluded_by_code"]["trace_row_cap"] == 1
    source = json.loads((tmp_path / "ds" / "settings.json").read_text())["source"]
    assert (source["sample"], source["max_rows_per_trace"]) == ("random", 1)
    rows = (tmp_path / "ds" / "rows.jsonl").read_bytes()

    rerun = invoke("pull", "ds")
    assert rerun.exit_code == 0, rerun.stderr
    assert "using its --ml-app, --sample, --max-rows-per-trace" in rerun.stderr
    assert output_json(rerun)["pull"]["added"] == 0
    assert (tmp_path / "ds" / "rows.jsonl").read_bytes() == rows
    other = invoke("pull", "ds", "--max-rows-per-trace", "2")
    assert other.exit_code == 2
    assert other.stderr.startswith(
        "Error [settings_conflict]: ds froze --max-rows-per-trace 1 on its first pull\n"
    )


def test_tags_given_in_another_order_are_the_frozen_ones(
    api: FakeExportApi, tmp_path: Path
) -> None:
    first = pull(tmp_path / "ds", "--tag", "env:synthetic", "--tag", "team:trips")
    assert first.exit_code == 0, first.stderr
    again = invoke("pull", "ds", "--tag", "team:trips", "--tag", "env:synthetic")
    assert again.exit_code == 0, again.stderr
    assert output_json(again)["window"]["frozen"] is True


def test_the_default_from_frozen_then_given_as_7d_later_reads_the_same_window(
    api: FakeExportApi, tmp_path: Path, clock: FakeClock
) -> None:
    first = pull(tmp_path / "ds")  # the default --from, 7d, frozen as now-7d
    assert first.exit_code == 0, first.stderr
    source = json.loads((tmp_path / "ds" / "settings.json").read_text())["source"]
    assert source["from_expr"] == "now-7d"
    clock.now_ms += 2 * DAY_MS
    rerun = invoke("pull", "ds", "--from", "7d")
    assert rerun.exit_code == 0, rerun.stderr
    data = output_json(rerun)
    assert data["window"] == {**output_json(first)["window"], "frozen": True}
    assert sent_windows(api)[-1] == (WEEK_AGO_MS, NOW_MS)
    assert "using its --ml-app\n" in rerun.stderr  # not --from: 7d is the frozen now-7d


def test_the_first_pull_into_a_directory_needs_ml_app(api: FakeExportApi, tmp_path: Path) -> None:
    result = invoke("pull", "ds")
    assert result.exit_code == 2
    assert result.stderr.startswith(
        "Error [usage]: dataset pull needs --ml-app (ds has no settings yet)\n"
    )
    assert api.requests == []
    assert not (tmp_path / "ds").exists()


def test_a_pull_stopped_by_max_candidates_steps_to_a_flagless_one_reading_further(
    api: FakeExportApi, tmp_path: Path
) -> None:
    result = pull(tmp_path / "ds", "--max-candidates", "3")
    assert result.exit_code == 0, result.stderr
    step = output_json(result)["next_steps"][0]
    assert step["command"] == "pup finetune dataset pull ds --max-candidates 6"
    further = invoke(*shlex.split(step["command"])[3:])
    assert further.exit_code == 0, further.stderr


def test_a_rerun_keeps_the_last_target_and_a_review_s_drops_are_replaced(
    api: FakeExportApi, tmp_path: Path
) -> None:
    target = tmp_path / "ds"
    assert output_json(pull(target, "--target-count", "2"))["accepted"] == 2
    again = output_json(pull(target))
    assert again["accepted"] == 2  # the remembered --target-count 2, not the default
    assert api.cursors == [None]  # page 1 is cached; nothing further is read
    first_row = json.loads((target / "rows.jsonl").read_text().splitlines()[0])["id"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": first_row, "keep": False, "reason": "synthetic"}) + "\n")
    labeled = output_json(invoke("label", str(target), "--file", str(labels)))
    pull_more = [step for step in labeled["next_steps"] if " dataset pull " in step["command"]]
    assert pull_more and "dropped 1" in pull_more[0]["why"]
    refilled = output_json(pull(target))
    assert refilled["pull"]["added"] == 1
    assert refilled["pipeline"]["rows"]["kept"] == 2


def test_a_random_pull_that_read_every_span_still_steps_to_replace_a_review_s_drops(
    api: FakeExportApi, tmp_path: Path
) -> None:
    target = tmp_path / "ds"
    first = output_json(pull(target, "--sample", "random", "--target-count", "2"))
    # Every span is read, but the target cut rows a rerun can replace drops with.
    assert (first["pull"]["stopped"], first["pull"]["rows_over_target"]) == ("exhausted", 2)
    first_row = json.loads((target / "rows.jsonl").read_text().splitlines()[0])["id"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": first_row, "keep": False, "reason": "synthetic"}) + "\n")
    labeled = output_json(invoke("label", str(target), "--file", str(labels)))
    pull_more = [step for step in labeled["next_steps"] if " dataset pull " in step["command"]]
    assert pull_more and pull_more[0]["command"].endswith("dataset pull ds")
    assert output_json(pull(target))["pull"]["added"] == 1


def test_the_pull_more_step_stops_once_every_matching_span_was_read(
    api: FakeExportApi, tmp_path: Path
) -> None:
    target = tmp_path / "ds"
    assert output_json(pull(target, "--target-count", "200"))["pull"]["stopped"] == "exhausted"
    first_row = json.loads((target / "rows.jsonl").read_text().splitlines()[0])["id"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"id": first_row, "keep": False, "reason": "synthetic"}) + "\n")
    labeled = output_json(invoke("label", str(target), "--file", str(labels)))
    assert not [step for step in labeled["next_steps"] if " dataset pull " in step["command"]]


def test_a_pull_that_failed_part_way_resumes_toward_the_limits_it_was_given(
    api: FakeExportApi, tmp_path: Path
) -> None:
    target = tmp_path / "ds"
    assert output_json(pull(target, "--target-count", "2"))["accepted"] == 2
    api.failures[PAGE_2_CURSOR] = 1
    assert pull(target, "--target-count", "200").exit_code == 1
    limits = json.loads((target / "pull_limits.json").read_text())
    assert limits["target_count"] == 200
    resumed = output_json(pull(target))  # what the failure's hint runs: no --target-count
    assert resumed["accepted"] == FIXTURE_ROWS
