# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pup_finetune.runtime import Runtime


def test_empty_env_gives_defaults() -> None:
    runtime = Runtime.from_env({})
    assert runtime == Runtime()
    assert runtime.output_format is None
    assert runtime.jq_filter is None
    assert not runtime.read_only
    assert not runtime.agent_mode
    assert not runtime.agent_env
    assert not runtime.under_agent
    assert not runtime.dev_mode
    assert not runtime.debug
    assert not runtime.under_pup


def test_full_env_is_parsed() -> None:
    runtime = Runtime.from_env(
        {
            "PUP_OUTPUT": "table",
            "PUP_FILTER": ".id",
            "PUP_READ_ONLY": "true",
            "PUP_AGENT_MODE": "true",
            "PUP_FINETUNE_DEV": "1",
            "PUP_FINETUNE_DEBUG": "1",
        }
    )
    assert runtime == Runtime(
        output_format="table",
        jq_filter=".id",
        read_only=True,
        agent_mode=True,
        dev_mode=True,
        debug=True,
    )
    assert runtime.under_pup


@pytest.mark.parametrize("value", ["1", "TRUE", "True", "yes", ""])
def test_pup_agent_mode_requires_exact_true(value: str) -> None:
    assert not Runtime.from_env({"PUP_AGENT_MODE": value}).agent_mode


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "On", " true "])
def test_read_only_is_on_for_an_on_value(value: str) -> None:
    runtime = Runtime.from_env({"PUP_READ_ONLY": value})
    assert runtime.read_only
    assert runtime.warnings == ()


@pytest.mark.parametrize("value", ["", "0", "false", "False", "NO", "off"])
def test_read_only_is_off_only_for_an_off_value(value: str) -> None:
    runtime = Runtime.from_env({"PUP_READ_ONLY": value})
    assert not runtime.read_only
    assert runtime.warnings == ()


@pytest.mark.parametrize("value", ["maybe", "2", "enabled", "tru"])
def test_an_unrecognized_read_only_value_means_read_only_with_a_warning(value: str) -> None:
    runtime = Runtime.from_env({"PUP_READ_ONLY": value})
    assert runtime.read_only
    (warning,) = runtime.warnings
    assert f"PUP_READ_ONLY={value!r}" in warning
    assert "read-only" in warning


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_dev_mode_is_on_for_an_on_value(value: str) -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_DEV": value})
    assert runtime.dev_mode
    assert runtime.warnings == ()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "OFF"])
def test_dev_mode_is_off_for_an_off_value(value: str) -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_DEV": value})
    assert not runtime.dev_mode
    assert runtime.warnings == ()


@pytest.mark.parametrize("value", ["2", "dev", "enabled"])
def test_an_unrecognized_dev_value_is_off_with_a_warning(value: str) -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_DEV": value})
    assert not runtime.dev_mode
    (warning,) = runtime.warnings
    assert f"PUP_FINETUNE_DEV={value!r}" in warning
    assert "off" in warning


@pytest.mark.parametrize("value", ["true", "0", "yes", ""])
def test_debug_requires_exact_one(value: str) -> None:
    assert not Runtime.from_env({"PUP_FINETUNE_DEBUG": value}).debug


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_quiet_is_on_for_an_on_value(value: str) -> None:
    assert Runtime.from_env({"PUP_FINETUNE_QUIET": value}).quiet
    assert not Runtime.from_env({}).quiet


def test_an_unrecognized_quiet_value_is_off_with_a_warning() -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_QUIET": "please"})
    assert not runtime.quiet
    assert "PUP_FINETUNE_QUIET='please'" in runtime.warnings[0]


@pytest.mark.parametrize(
    ("value", "mode"), [("steps", "steps"), ("STEPS", "steps"), ("", "auto"), ("auto", "auto")]
)
def test_progress_is_steps_or_auto(value: str, mode: str) -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_PROGRESS": value})
    assert (runtime.progress, runtime.warnings) == (mode, ())
    assert Runtime.from_env({}).progress == "auto"


def test_an_unrecognized_progress_value_is_auto_with_a_warning() -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_PROGRESS": "all"})
    assert runtime.progress == "auto"
    assert runtime.warnings == (
        "PUP_FINETUNE_PROGRESS='all' is neither steps nor auto: treating it as auto",
    )


@pytest.mark.parametrize(
    ("env", "agent_mode", "agent_env"),
    [
        ({"PUP_AGENT_MODE": "true"}, True, False),
        ({"AI_AGENT": "some-agent"}, False, True),
        ({"CLAUDECODE": "1"}, False, True),
        ({"PUP_AGENT_MODE": "true", "CLAUDECODE": "1"}, True, True),
    ],
)
def test_a_coding_agent_is_detected_from_any_of_its_variables(
    env: dict[str, str], agent_mode: bool, agent_env: bool
) -> None:
    runtime = Runtime.from_env(env)
    assert (runtime.agent_mode, runtime.agent_env) == (agent_mode, agent_env)
    assert runtime.under_agent


def test_empty_agent_variables_are_not_an_agent() -> None:
    runtime = Runtime.from_env({"AI_AGENT": "", "CLAUDECODE": "", "PUP_AGENT_MODE": "false"})
    assert not runtime.under_agent


def test_empty_output_format_is_not_under_pup() -> None:
    assert not Runtime.from_env({"PUP_OUTPUT": ""}).under_pup


def test_default_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_OUTPUT", "json")
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    runtime = Runtime.from_env()
    assert runtime.output_format == "json"
    assert runtime.read_only


def test_runtime_is_frozen() -> None:
    runtime = Runtime()
    with pytest.raises(dataclasses.FrozenInstanceError):
        runtime.read_only = True  # type: ignore[misc]


def test_secret_env_vars_are_not_read() -> None:
    secret = "synthetic-secret-value"  # noqa: S105 - fake value for the test
    runtime = Runtime.from_env(
        {"DD_API_KEY": secret, "DD_APP_KEY": secret, "FIREWORKS_API_KEY": secret}
    )
    assert runtime == Runtime()
    assert secret not in repr(runtime)


def test_a_command_prints_each_environment_warning_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from pup_finetune.cli import app

    rows = Path(__file__).resolve().parent / "fixtures" / "rows" / "tool_use.jsonl"
    monkeypatch.setenv("PUP_READ_ONLY", "maybe")
    monkeypatch.setenv("PUP_FINETUNE_DEV", "dev")
    result = CliRunner().invoke(
        app, ["dataset", "import", str(tmp_path / "ds"), "--file", str(rows)]
    )
    assert result.exit_code == 0, result.stderr
    assert result.stderr.count("warning: PUP_READ_ONLY='maybe'") == 1
    assert result.stderr.count("warning: PUP_FINETUNE_DEV='dev'") == 1
