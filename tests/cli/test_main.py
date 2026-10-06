# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Exit codes and output discipline, exercised through test-only Typer apps."""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import pytest
import typer
from typer.testing import CliRunner

from pup_finetune import __version__, cli, redaction
from pup_finetune.cli._common import CommandResult, get_runtime, run_app, run_command
from pup_finetune.errors import (
    ConfirmRequired,
    ExternalError,
    FinetuneError,
    ReadOnlyError,
    StateError,
    UsageError,
    require_confirm,
)

runner = CliRunner()


def make_app(error: BaseException | None = None) -> typer.Typer:
    """A two-command app: `ok` returns a result, `fail` raises `error` from inside the command."""
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    @app.command()
    @run_command
    def ok(name: Annotated[str, typer.Option()] = "demo") -> Mapping[str, Any]:
        return {"name": name, "next_command": f"pup finetune show {name}"}

    @app.command()
    @run_command
    def fail() -> Mapping[str, Any]:
        assert error is not None
        raise error

    @app.command()
    @run_command
    def push(confirm: Annotated[bool, typer.Option("--confirm")] = False) -> Mapping[str, Any]:
        require_confirm(confirm, {"records": 3}, "pup finetune push --confirm")
        return {"status": "pushed"}

    @app.command()
    @run_command
    def needs(count: Annotated[int, typer.Option()]) -> Mapping[str, Any]:
        return {"count": count}

    @app.command()
    @run_command
    def env() -> Mapping[str, Any]:
        return {"read_only": get_runtime().read_only}

    @app.command()
    @run_command
    def report(code: Annotated[int, typer.Option()] = 1) -> CommandResult:
        return CommandResult({"ok": code == 0}, exit_code=code)

    return app


def test_success_emits_exactly_one_json_document() -> None:
    result = runner.invoke(make_app(), ["ok", "--name", "x"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"name": "x", "next_command": "pup finetune show x"}
    assert result.stderr == ""


@pytest.mark.parametrize("code", [0, 1])
def test_command_result_emits_one_json_document_then_exits_with_its_code(code: int) -> None:
    result = runner.invoke(make_app(), ["report", "--code", str(code)])
    assert result.exit_code == code
    assert json.loads(result.stdout) == {"ok": code == 0}
    assert result.stderr == ""


def test_command_sees_the_runtime_from_env() -> None:
    result = runner.invoke(make_app(), ["env"], env={"PUP_READ_ONLY": "true"})
    assert json.loads(result.stdout) == {"read_only": True}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (UsageError("bad input", hint="see --help"), 2),
        (ReadOnlyError("remote write blocked"), 2),
        (ExternalError("exit 1", source="pup"), 1),
        (StateError("corrupt run.json"), 1),
        (FinetuneError("generic"), 1),
    ],
)
def test_finetune_errors_map_to_exit_codes(error: FinetuneError, code: int) -> None:
    result = runner.invoke(make_app(error), ["fail"])
    assert result.exit_code == code
    assert result.stdout == ""
    assert result.stderr.startswith(f"Error: {error}\n")
    if error.hint:
        assert f"Hint: {error.hint}" in result.stderr


def test_confirm_required_emits_preview_json_and_exits_2() -> None:
    result = runner.invoke(make_app(), ["push"])
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "status": "confirmation_required",
        "preview": {"records": 3},
        "next_command": "pup finetune push --confirm",
    }
    assert "--confirm" in result.stderr


def test_auto_approve_does_not_imply_confirm() -> None:
    env = {"PUP_AUTO_APPROVE": "true", "PUP_AGENT_MODE": "true"}
    result = runner.invoke(make_app(), ["push"], env=env)
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "confirmation_required"


def test_confirm_runs_the_action() -> None:
    result = runner.invoke(make_app(), ["push", "--confirm"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"status": "pushed"}


def test_unexpected_error_is_one_line_without_traceback() -> None:
    result = runner.invoke(make_app(RuntimeError("kaboom\nsecond line")), ["fail"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Error: unexpected RuntimeError: kaboom (set PUP_FINETUNE_DEBUG=1 for a traceback)\n"
    )
    assert "second line" not in result.stderr


def test_unexpected_error_shows_traceback_in_debug() -> None:
    result = runner.invoke(
        make_app(RuntimeError("kaboom")), ["fail"], env={"PUP_FINETUNE_DEBUG": "1"}
    )
    assert result.exit_code == 1
    assert "Traceback" in result.stderr
    assert "Error: unexpected RuntimeError: kaboom" in result.stderr


def test_click_usage_errors_keep_exit_2() -> None:
    missing = runner.invoke(make_app(), ["needs"])
    assert missing.exit_code == 2
    assert missing.stdout == ""
    bad_type = runner.invoke(make_app(), ["needs", "--count", "many"])
    assert bad_type.exit_code == 2
    unknown = runner.invoke(make_app(), ["nope"])
    assert unknown.exit_code == 2


def test_bad_parameter_raised_in_a_command_keeps_exit_2() -> None:
    result = runner.invoke(make_app(typer.BadParameter("no good")), ["fail"])
    assert result.exit_code == 2
    assert "no good" in result.stderr


def test_abort_raised_in_a_command_keeps_click_handling() -> None:
    result = runner.invoke(make_app(typer.Abort()), ["fail"])
    assert result.exit_code == 1
    assert "Aborted" in result.stderr
    assert "unexpected" not in result.stderr


def test_exit_raised_in_a_command_keeps_its_code() -> None:
    result = runner.invoke(make_app(typer.Exit(3)), ["fail"])
    assert result.exit_code == 3
    assert result.stderr == ""


@pytest.fixture
def fake_pup_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a fake `pup` on PATH whose `format` prefixes its stdin with the requested format."""
    fake_pup = tmp_path / "pup"
    fake_pup.write_text('#!/bin/sh\nprintf "rendered-as-%s:" "$PUP_OUTPUT"\ncat\n')
    fake_pup.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    return fake_pup


def test_under_pup_success_goes_through_pup_format(fake_pup_on_path: Path) -> None:
    del fake_pup_on_path
    result = runner.invoke(make_app(), ["ok", "--name", "x"], env={"PUP_OUTPUT": "yaml"})
    assert result.exit_code == 0
    prefix = "rendered-as-yaml:"
    assert result.stdout.startswith(prefix)
    assert json.loads(result.stdout.removeprefix(prefix))["name"] == "x"
    assert result.stderr == ""


@pytest.mark.parametrize("code", [0, 1])
def test_under_pup_command_result_goes_through_pup_format_then_exits(
    fake_pup_on_path: Path, code: int
) -> None:
    del fake_pup_on_path
    result = runner.invoke(make_app(), ["report", "--code", str(code)], env={"PUP_OUTPUT": "yaml"})
    assert result.exit_code == code
    prefix = "rendered-as-yaml:"
    assert result.stdout.startswith(prefix)
    assert json.loads(result.stdout.removeprefix(prefix)) == {"ok": code == 0}
    assert result.stderr == ""


def test_under_pup_confirm_preview_goes_through_pup_format(fake_pup_on_path: Path) -> None:
    del fake_pup_on_path
    result = runner.invoke(make_app(), ["push"], env={"PUP_OUTPUT": "yaml"})
    assert result.exit_code == 2
    prefix = "rendered-as-yaml:"
    assert result.stdout.startswith(prefix)
    preview = json.loads(result.stdout.removeprefix(prefix))
    assert preview["status"] == "confirmation_required"
    assert "--confirm" in result.stderr


def test_under_pup_without_pup_prints_json_with_one_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    result = runner.invoke(make_app(), ["ok"], env={"PUP_OUTPUT": "yaml"})
    assert result.exit_code == 0
    assert json.loads(result.stdout)["name"] == "demo"
    assert result.stderr.count("pup format failed") == 1
    assert result.stderr.count("\n") == 1


def make_bare_app(error: BaseException) -> typer.Typer:
    """An app whose command is not wrapped by run_command, so errors reach run_app."""
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    @app.command()
    def boom(count: Annotated[int, typer.Option()] = 0) -> None:
        del count
        raise error

    return app


def run_and_exit_code(app: typer.Typer, args: list[str]) -> int:
    with pytest.raises(SystemExit) as info:
        run_app(app, args)
    code = info.value.code
    assert isinstance(code, int)
    return code


def test_run_app_maps_escaped_finetune_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_and_exit_code(make_bare_app(StateError("locked")), []) == 1
    assert capsys.readouterr().err == "Error: locked\n"


def test_run_app_emits_confirm_preview(capsys: pytest.CaptureFixture[str]) -> None:
    error = ConfirmRequired("preview only", preview={"a": 1}, rerun="pup finetune x --confirm")
    assert run_and_exit_code(make_bare_app(error), []) == 2
    out, err = capsys.readouterr()
    assert json.loads(out)["next_command"] == "pup finetune x --confirm"
    assert "preview only" in err


def test_run_app_maps_unexpected_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_and_exit_code(make_bare_app(KeyError("k")), []) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: unexpected KeyError: 'k' (set PUP_FINETUNE_DEBUG=1")
    assert err.count("\n") == 1
    assert "Traceback" not in err


def test_run_app_keeps_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_and_exit_code(make_bare_app(RuntimeError()), ["--count", "x"]) == 2
    assert capsys.readouterr().out == ""


def test_run_app_redacts_logs_from_handlers_added_later() -> None:
    secret = "zq9Xsynthetic0000logVALUEw7Pm"  # noqa: S105 - synthetic test value
    stream = io.StringIO()
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    @app.command()
    def noisy() -> None:
        redaction.register_secret(secret)
        logger = logging.getLogger("pup_finetune.tests.run_app")
        logger.addHandler(logging.StreamHandler(stream))
        logger.propagate = False
        logging.getLogger("pup_finetune.tests.run_app.child").warning("key=%s", secret)

    try:
        assert run_and_exit_code(app, []) == 0
    finally:
        logging.getLogger("pup_finetune.tests.run_app").handlers.clear()
        logging.getLogger("pup_finetune.tests.run_app").propagate = True
        redaction.clear_secrets()
    assert stream.getvalue() == "key=***\n"


def test_main_runs_the_real_app(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["pup-finetune", "--version"])
    with pytest.raises(SystemExit) as info:
        cli.main()
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__
