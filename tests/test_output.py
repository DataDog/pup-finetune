# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from rich.console import Console

from pup_finetune import output, redaction
from pup_finetune.errors import ConfirmRequired, UsageError
from pup_finetune.output import emit, emit_error, info, status, to_json, warn
from pup_finetune.runtime import Runtime

SECRET = "zq9Xsynthetic0000keyVALUEw7Pm"  # noqa: S105 - synthetic test value
STANDALONE = Runtime()
UNDER_PUP = Runtime(output_format="yaml")


class FakePupRunner:
    """Records each call and answers with a fixed completed process (or raises)."""

    def __init__(
        self,
        *,
        stdout: str = "formatted-by-pup\n",
        stderr: str = "",
        returncode: int = 0,
        raises: BaseException | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.raises = raises
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, args: Sequence[str], stdin: str) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(args), stdin))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            ["pup", *args], self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def test_standalone_prints_sorted_indented_json(capsys: pytest.CaptureFixture[str]) -> None:
    runner = FakePupRunner()
    emit({"b": 1, "a": {"path": Path("runs/x")}}, runtime=STANDALONE, runner=runner)
    out, err = capsys.readouterr()
    assert out == json.dumps({"a": {"path": "runs/x"}, "b": 1}, indent=2, sort_keys=True) + "\n"
    assert err == ""
    assert runner.calls == []


def test_under_pup_pipes_json_through_pup_format(capsys: pytest.CaptureFixture[str]) -> None:
    runner = FakePupRunner(stdout="a: 1\n")
    emit({"a": 1}, runtime=UNDER_PUP, runner=runner)
    out, err = capsys.readouterr()
    assert runner.calls == [(["format"], to_json({"a": 1}))]
    assert out == "a: 1\n"
    assert err == ""


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("runtime", [STANDALONE, UNDER_PUP], ids=["standalone", "under-pup"])
def test_non_finite_numbers_are_rejected_not_printed(
    value: float, runtime: Runtime, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = FakePupRunner()
    with pytest.raises(ValueError, match="JSON compliant"):
        emit({"loss": value}, runtime=runtime, runner=runner)
    assert capsys.readouterr().out == ""
    assert runner.calls == []


def test_pup_output_gets_a_trailing_newline(capsys: pytest.CaptureFixture[str]) -> None:
    emit({"a": 1}, runtime=UNDER_PUP, runner=FakePupRunner(stdout="a: 1"))
    assert capsys.readouterr().out == "a: 1\n"


def test_pup_format_failure_falls_back_to_json(capsys: pytest.CaptureFixture[str]) -> None:
    runner = FakePupRunner(returncode=1, stdout="", stderr="Error: boom\nmore detail\n")
    emit({"a": 1}, runtime=UNDER_PUP, runner=runner)
    out, err = capsys.readouterr()
    assert json.loads(out) == {"a": 1}
    assert err.count("\n") == 1
    assert "pup format failed" in err
    assert "Error: boom" in err
    assert "more detail" not in err


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("pup not found on PATH"),
        subprocess.TimeoutExpired(["pup"], 30),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_runner_errors_fall_back_to_json(
    exc: BaseException, capsys: pytest.CaptureFixture[str]
) -> None:
    emit({"a": 1}, runtime=UNDER_PUP, runner=FakePupRunner(raises=exc))
    out, err = capsys.readouterr()
    assert json.loads(out) == {"a": 1}
    assert err.count("pup format failed") == 1
    assert err.count("\n") == 1


def test_missing_pup_on_path_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    emit({"a": 1}, runtime=UNDER_PUP)
    out, err = capsys.readouterr()
    assert json.loads(out) == {"a": 1}
    assert "pup not found on PATH" in err


def test_emit_error_writes_message_and_hint_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_error(UsageError("bad flag", hint="try --help"), runtime=STANDALONE)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "Error: bad flag\nHint: try --help\n"


def test_confirm_required_also_emits_preview_json(capsys: pytest.CaptureFixture[str]) -> None:
    err = ConfirmRequired(
        "preview only", preview={"rows": 3}, rerun="pup finetune dataset push d --confirm"
    )
    emit_error(err, runtime=STANDALONE)
    out, stderr = capsys.readouterr()
    assert json.loads(out) == {
        "status": "confirmation_required",
        "preview": {"rows": 3},
        "next_command": "pup finetune dataset push d --confirm",
    }
    assert "preview only" in stderr
    assert "pup finetune dataset push d --confirm" in stderr


@pytest.fixture
def registered_secret() -> Iterator[str]:
    redaction.clear_secrets()
    redaction.register_secret(SECRET)
    yield SECRET
    redaction.clear_secrets()


class _Holder:
    def __str__(self) -> str:
        return f"holder({SECRET})"


@pytest.mark.parametrize("runtime", [STANDALONE, UNDER_PUP], ids=["standalone", "under-pup"])
def test_emit_redacts_every_string_in_a_result(
    runtime: Runtime, registered_secret: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = FakePupRunner()
    result = {
        "detail": f"key={registered_secret}",
        f"k-{registered_secret}": [("Bearer abc.DEF", 1, 2.5, True, None)],
        "path": Path(f"/runs/{registered_secret}"),
    }
    emit(result, runtime=runtime, runner=runner)
    out = capsys.readouterr().out
    document = runner.calls[0][1] if runtime.under_pup else out
    assert registered_secret not in document
    assert json.loads(document) == {
        "detail": "key=***",
        "k-***": [["Bearer ***", 1, 2.5, True, None]],
        "path": "/runs/***",
    }


def test_confirm_required_redacts_sets_and_objects_in_the_preview(
    registered_secret: str, capsys: pytest.CaptureFixture[str]
) -> None:
    err = ConfirmRequired(
        "preview only",
        preview={"tags": {f"t:{registered_secret}"}, "obj": _Holder()},
        rerun=f"pup finetune x --name {registered_secret} --confirm",
    )
    emit_error(err, runtime=STANDALONE)
    out, stderr = capsys.readouterr()
    assert registered_secret not in out + stderr
    assert json.loads(out) == {
        "status": "confirmation_required",
        "preview": {"tags": ["t:***"], "obj": "holder(***)"},
        "next_command": "pup finetune x --name *** --confirm",
    }


def test_confirm_required_preview_goes_through_pup_under_pup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = FakePupRunner(stdout="status: confirmation_required\n")
    err = ConfirmRequired("preview only", preview={}, rerun="pup finetune x --confirm")
    emit_error(err, runtime=UNDER_PUP, runner=runner)
    assert len(runner.calls) == 1
    assert json.loads(runner.calls[0][1])["status"] == "confirmation_required"
    assert capsys.readouterr().out == "status: confirmation_required\n"


def test_progress_helpers_write_plain_lines_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info("pulling [spans] :fire:")
    warn("empty validation split")
    with status("training"):
        pass
    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines() == [
        "pulling [spans] :fire:",
        "warning: empty validation split",
        "training",
    ]
    assert "\x1b[" not in err


@pytest.mark.parametrize(
    "env", [{"FORCE_COLOR": "1"}, {"TTY_COMPATIBLE": "1"}, {"TTY_INTERACTIVE": "1"}]
)
def test_progress_stays_plain_on_non_tty_even_when_forced(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    warn("empty validation split")
    with status("training"):
        pass
    err = capsys.readouterr().err
    assert err.splitlines() == ["warning: empty validation split", "training"]
    assert "\x1b" not in err


def test_terminal_progress_does_not_parse_markup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Rich skips the transient status render on a dumb terminal, so ask for a capable one.
    monkeypatch.setenv("TERM", "xterm-256color")
    buffer = io.StringIO()
    terminal = Console(file=buffer, force_terminal=True, highlight=False, width=80)
    monkeypatch.setattr(output, "stderr_console", lambda: terminal)
    warn("bad [/oops] name")
    with status("loading [/y] dataset"):
        pass
    assert capsys.readouterr().out == ""
    rendered = buffer.getvalue()
    assert "warning: bad [/oops] name" in rendered
    assert "loading [/y] dataset" in rendered


def test_progress_lines_do_not_wrap(capsys: pytest.CaptureFixture[str]) -> None:
    long = "x" * 300
    info(long)
    assert capsys.readouterr().err == long + "\n"
