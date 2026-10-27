# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from pup_finetune.datadog.pup import (
    FIXED_FLAGS,
    MIN_PUP_VERSION,
    Pup,
    child_env,
    find_pup,
    parse_json_output,
    parse_version,
    run_pup,
    run_subprocess,
)
from pup_finetune.errors import ExternalError
from tests.fakes.pup import FAKE_BINARY, FakePup

# Synthetic stand-ins; never real credentials.
FAKE_TOKEN = "synthetic-access-token-0123456789"  # noqa: S105 - synthetic test value
USER_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/home/someone",
    "PUP_FILTER": ".data[0]",
    "PUP_OUTPUT": "table",
    "DD_ACCESS_TOKEN": FAKE_TOKEN,
    "DD_SITE": "datadoghq.com",
}


# --- argv and stdin -------------------------------------------------------------------------


def test_run_json_builds_argv_with_fixed_flags_first() -> None:
    fake = FakePup().on("llm-obs", "experiments", "list", json={"data": [{"id": "e1"}]})
    result = fake.client().run_json(["llm-obs", "experiments", "list", "--limit", "5"])
    assert result == {"data": [{"id": "e1"}]}
    assert fake.last.argv == [
        FAKE_BINARY,
        "--no-agent",
        "--output",
        "json",
        "llm-obs",
        "experiments",
        "list",
        "--limit",
        "5",
    ]
    assert fake.last.stdin is None
    assert fake.last.timeout == 120


def test_run_json_sends_input_as_json_on_stdin() -> None:
    fake = FakePup().on("llm-obs", "datasets", "records-add", json={"ok": True})
    body = {"records": [{"input": "synthetic question", "n": 1}]}
    fake.client().run_json(["llm-obs", "datasets", "records-add"], input=body, timeout=5)
    assert fake.last.stdin_json() == body
    assert fake.last.timeout == 5


def test_falsy_input_is_still_sent() -> None:
    fake = FakePup().on("x", json=None)
    fake.client().run_json(["x"], input={})
    assert fake.last.stdin == b"{}"


def test_empty_stdout_returns_none() -> None:
    fake = FakePup().on("api", stdout=b"\n  \n")
    assert fake.client().run_json(["api", "v2/thing", "-X", "DELETE"]) is None


def test_api_without_body() -> None:
    fake = FakePup().on("api", json={"data": []})
    assert fake.client().api("get", "v2/llm-obs/v1/projects") == {"data": []}
    assert fake.last.args == ("api", "v2/llm-obs/v1/projects", "-X", "GET")
    assert fake.last.stdin is None


def test_api_with_body_and_content_type() -> None:
    path = "v2/llm-obs/v1/spans/events/search"
    fake = FakePup().on("api", path, json={"data": [], "meta": {}})
    body = {"data": {"type": "spans", "attributes": {"filter": {"span_kind": "llm"}}}}
    fake.client().api("POST", path, body, content_type="application/vnd.api+json")
    assert fake.last.args == (
        "api",
        path,
        "-X",
        "POST",
        "--input",
        "-",
        "-H",
        "Content-Type: application/vnd.api+json",
    )
    assert fake.last.stdin_json() == body


def test_api_with_body_and_empty_2xx_response() -> None:
    fake = FakePup().on("api", stdout=b"")
    assert fake.client().api("POST", "v2/llm-obs/v1/experiments/e1/events", {"spans": []}) is None
    assert fake.last.args == (
        "api",
        "v2/llm-obs/v1/experiments/e1/events",
        "-X",
        "POST",
        "--input",
        "-",
    )


def test_api_rejects_a_path_that_looks_like_a_flag() -> None:
    with pytest.raises(ValueError, match="must not start"):
        FakePup().client().api("GET", "--help")


def test_longest_registered_prefix_wins() -> None:
    fake = FakePup().on("llm-obs", json="generic").on("llm-obs", "datasets", json="specific")
    pup = fake.client()
    assert pup.run_json(["llm-obs", "datasets", "list"]) == "specific"
    assert pup.run_json(["llm-obs", "spans", "search"]) == "generic"


# --- environment ----------------------------------------------------------------------------


def test_child_env_drops_output_settings_and_the_access_token() -> None:
    fake = FakePup().on("auth", "status", json={"authenticated": True})
    fake.client(USER_ENV).run_json(["auth", "status"])
    env = fake.last.env
    assert "PUP_FILTER" not in env
    assert "PUP_OUTPUT" not in env
    assert "DD_ACCESS_TOKEN" not in env
    assert env["DD_SITE"] == "datadoghq.com"
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/someone"


def test_keep_dd_token_override_passes_the_token_through() -> None:
    environ = {**USER_ENV, "PUP_FINETUNE_KEEP_DD_TOKEN": "1"}
    env = child_env(environ)
    assert env["DD_ACCESS_TOKEN"] == FAKE_TOKEN
    assert "PUP_FILTER" not in env
    assert "PUP_OUTPUT" not in env


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_keep_dd_token_1_keeps_the_token(value: str) -> None:
    assert "DD_ACCESS_TOKEN" not in child_env({**USER_ENV, "PUP_FINETUNE_KEEP_DD_TOKEN": value})


def test_child_env_does_not_modify_the_parent_env() -> None:
    environ = dict(USER_ENV)
    child_env(environ)
    assert environ == USER_ENV


def test_default_environ_is_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_FILTER", ".x")
    monkeypatch.setenv("SOME_SETTING", "kept")
    fake = FakePup().on("x", json=1)
    Pup(runner=fake, binary=FAKE_BINARY).run_json(["x"])
    assert fake.last.env["SOME_SETTING"] == "kept"
    assert "PUP_FILTER" not in fake.last.env


# --- errors ---------------------------------------------------------------------------------


def test_non_zero_exit_raises_external_error_with_trimmed_stderr() -> None:
    stderr = "\x1b[31mError:\x1b[0m dataset not found\n\nline two\nline three\nline four\n"
    fake = FakePup().on("llm-obs", "datasets", "get", returncode=1, stderr=stderr)
    with pytest.raises(ExternalError) as info:
        fake.client().run_json(["llm-obs", "datasets", "get", "--id", "d1"])
    err = info.value
    assert err.source == "pup"
    assert err.message == (
        "`pup llm-obs datasets get` failed (exit 1): Error: dataset not found; line two; line three"
    )
    assert "line four" not in err.message
    assert "\x1b" not in err.message
    assert err.exit_code == 1
    assert err.hint is None  # not an auth failure


def test_error_detail_falls_back_to_stdout_when_stderr_is_empty() -> None:
    stdout = f"Error: 401 Unauthorized: not logged in token={FAKE_TOKEN}\n"
    fake = FakePup().on("api", returncode=1, stdout=stdout)
    with pytest.raises(ExternalError) as info:
        fake.client(USER_ENV).api("GET", "v2/thing")
    err = info.value
    assert err.message == (
        "`pup api v2/thing` failed (exit 1): Error: 401 Unauthorized: not logged in token=[redacted]"
    )
    assert err.hint is not None
    assert "pup auth login" in err.hint


def test_error_message_is_capped() -> None:
    fake = FakePup().on("x", returncode=2, stderr="e" * 5000)
    with pytest.raises(ExternalError) as info:
        fake.client().run_json(["x"])
    assert len(info.value.message) < 400


def test_error_message_never_echoes_secrets() -> None:
    stderr = f"Error: 401 Unauthorized token={FAKE_TOKEN}\nAuthorization: Bearer abc.def.ghi\n"
    fake = FakePup().on("api", returncode=1, stderr=stderr)
    with pytest.raises(ExternalError) as info:
        fake.client(USER_ENV).api("GET", "v2/thing")
    err = info.value
    assert FAKE_TOKEN not in err.message
    assert "abc.def.ghi" not in err.message
    assert "[redacted]" in err.message
    assert err.hint is not None
    assert "pup auth login" in err.hint
    assert ".data[0]" not in err.message  # no env values


def test_empty_stderr_still_names_the_exit_code() -> None:
    fake = FakePup().on("x", returncode=3)
    with pytest.raises(ExternalError) as info:
        fake.client().run_json(["x"])
    assert info.value.message == "`pup x` failed (exit 3)"


def test_timeout_raises_external_error() -> None:
    fake = FakePup().on("llm-obs", "spans", times_out=True)
    with pytest.raises(ExternalError, match=r"`pup llm-obs spans search` timed out after 7s"):
        fake.client().run_json(["llm-obs", "spans", "search"], timeout=7)


def test_os_error_raises_external_error() -> None:
    fake = FakePup().on("x", raises=PermissionError(13, "Permission denied"))
    with pytest.raises(ExternalError, match="could not run `pup x`: Permission denied"):
        fake.client().run_json(["x"])


def test_invalid_json_raises_external_error() -> None:
    fake = FakePup().on("x", stdout="not json at all\n")
    with pytest.raises(ExternalError, match="did not print valid JSON"):
        fake.client().run_json(["x"])


def test_unregistered_command_fails_like_pup() -> None:
    with pytest.raises(ExternalError, match="no response"):
        FakePup().client().run_json(["nope"])


# --- tolerant parsing -----------------------------------------------------------------------


def test_parse_json_output_prefers_strict_json() -> None:
    assert parse_json_output('  {"a": [1, 2]}\n') == {"a": [1, 2]}
    assert parse_json_output("[1]") == [1]


def test_parse_json_output_skips_leading_human_lines() -> None:
    text = (
        "✅ Authenticated for site: datadoghq.com\n"
        "   Token expires in: 49m35s\n"
        "[info] refreshed token\n"
        '{\n  "authenticated": true,\n  "site": "datadoghq.com"\n}\n'
    )
    assert parse_json_output(text) == {"authenticated": True, "site": "datadoghq.com"}


def test_parse_json_output_finds_a_leading_array() -> None:
    assert parse_json_output('notice: token refreshed\n[{"id": 1}]\n') == [{"id": 1}]


def test_parse_json_output_skips_a_status_line_that_starts_with_a_bracket() -> None:
    assert parse_json_output('[2] retries left\n{"a": 1}\n') == {"a": 1}


def test_parse_json_output_rejects_truncated_pretty_printed_json() -> None:
    truncated = '{\n  "data": [\n    {"id": "a"},\n    {"id": "b"}\n  ],\n  "meta": {"page": \n'
    with pytest.raises(json.JSONDecodeError):
        parse_json_output(truncated)


def test_parse_json_output_ignores_indented_values() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_output('notice\n  {"a": 1}\n')


def test_parse_json_output_rejects_trailing_text_after_the_value() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_output('notice\n{"a": 1}\ntrailing notice\n')


def test_run_json_rejects_truncated_output() -> None:
    fake = FakePup().on("x", stdout='Token refreshed\n{\n  "data": [\n    {"id": "a"},\n')
    with pytest.raises(ExternalError, match="did not print valid JSON"):
        fake.client().run_json(["x"])


def test_parse_json_output_raises_without_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_output("no json {here\n")


def test_run_json_tolerates_leading_lines() -> None:
    fake = FakePup().on("auth", "status", stdout='Token refreshed\n{"authenticated": false}\n')
    assert fake.client().run_json(["auth", "status"]) == {"authenticated": False}


# --- version and auth -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pup 1.10.7\n", (1, 10, 7)),
        ("pup version 2.0.13-rc.1", (2, 0, 13)),
        ("1.11.0", (1, 11, 0)),
        ("pup dev", None),
        ("", None),
    ],
)
def test_parse_version(text: str, expected: tuple[int, int, int] | None) -> None:
    assert parse_version(text) == expected


def test_version_runs_plain_pup_version() -> None:
    fake = FakePup().on("--version", raw=True, stdout="pup 1.12.3\n")
    assert fake.client(USER_ENV).version() == (1, 12, 3)
    assert fake.last.argv == [FAKE_BINARY, "--version"]
    assert "PUP_OUTPUT" not in fake.last.env


def test_unparseable_version_raises() -> None:
    fake = FakePup().on("--version", raw=True, stdout="pup nightly\n")
    with pytest.raises(ExternalError, match="could not parse a version"):
        fake.client().version()


def test_min_version_is_the_verified_release() -> None:
    assert MIN_PUP_VERSION == (1, 10, 7)


def test_auth_status_keeps_only_safe_fields() -> None:
    payload = {
        "authenticated": True,
        "expires_at": "2026-01-01T00:00:00Z",
        "has_refresh": True,
        "org": None,
        "scopes": ["synthetic_read"],
        "site": "datadoghq.com",
        "status": "valid",
        "token_type": "Bearer",
        "access_token": FAKE_TOKEN,
    }
    stdout = "✅ Authenticated for site: datadoghq.com\n" + json.dumps(payload)
    fake = FakePup().on("auth", "status", stdout=stdout)
    status = fake.client().auth_status()
    assert status == {
        "authenticated": True,
        "site": "datadoghq.com",
        "org": None,
        "expires_at": "2026-01-01T00:00:00Z",
    }
    assert FAKE_TOKEN not in json.dumps(status)


def test_auth_status_treats_non_true_as_unauthenticated() -> None:
    fake = FakePup().on("auth", "status", json={"authenticated": "yes"})
    assert fake.client().auth_status()["authenticated"] is False


def test_auth_status_requires_an_object() -> None:
    fake = FakePup().on("auth", "status", json=[])
    with pytest.raises(ExternalError, match="JSON object"):
        fake.client().auth_status()


# --- finding the binary ---------------------------------------------------------------------


def _executable(directory: Path, name: str = "pup", script: str = "#!/bin/sh\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(script)
    path.chmod(0o755)
    return path


def test_find_pup_uses_path(tmp_path: Path) -> None:
    pup = _executable(tmp_path / "bin")
    assert find_pup({"PATH": str(tmp_path / "bin")}) == str(pup)


def test_find_pup_prefers_the_override(tmp_path: Path) -> None:
    _executable(tmp_path / "bin")
    custom = _executable(tmp_path / "custom", "my-pup")
    environ = {"PATH": str(tmp_path / "bin"), "PUP_FINETUNE_PUP": str(custom)}
    assert find_pup(environ) == str(custom)


def test_missing_pup_raises_with_install_hint(tmp_path: Path) -> None:
    pup = Pup(runner=FakePup(), environ={"PATH": str(tmp_path)})
    with pytest.raises(ExternalError) as info:
        pup.run_json(["x"])
    assert info.value.message == "pup not found on PATH"
    assert info.value.hint is not None
    assert "brew install datadog-labs/pack/pup" in info.value.hint
    assert "README" in info.value.hint


def test_bad_override_names_the_variable(tmp_path: Path) -> None:
    environ = {"PATH": str(tmp_path), "PUP_FINETUNE_PUP": str(tmp_path / "missing")}
    with pytest.raises(ExternalError, match="PUP_FINETUNE_PUP does not name an executable"):
        Pup(environ=environ).executable()


def test_constructing_pup_never_fails_without_pup(tmp_path: Path) -> None:
    Pup(environ={"PATH": str(tmp_path)})


# --- the real subprocess runner -------------------------------------------------------------


def test_real_runner_against_a_script_pup(tmp_path: Path) -> None:
    script = (
        "#!/bin/sh\n"
        'printf \'{"args": "%s", "filter": "%s", "token": "%s", "stdin": \' '
        '"$*" "${PUP_FILTER-unset}" "${DD_ACCESS_TOKEN-unset}"\n'
        "cat\n"
        "printf '}'\n"
    )
    pup_path = _executable(tmp_path / "bin", script=script)
    environ = {**USER_ENV, "PATH": "/usr/bin:/bin", "PUP_FINETUNE_PUP": str(pup_path)}
    result = Pup(environ=environ).run_json(["llm-obs", "spans", "search"], input={"q": 1})
    assert result == {
        "args": " ".join([*FIXED_FLAGS, "llm-obs", "spans", "search"]),
        "filter": "unset",
        "token": "unset",
        "stdin": {"q": 1},
    }


def test_run_subprocess_captures_bytes_and_exit_code(tmp_path: Path) -> None:
    script = _executable(tmp_path, "tool", "#!/bin/sh\necho out\necho err >&2\nexit 4\n")
    completed = run_subprocess([str(script)], input=None, env={"PATH": "/usr/bin:/bin"}, timeout=10)
    assert completed.returncode == 4
    assert completed.stdout == b"out\n"
    assert completed.stderr == b"err\n"


def test_run_subprocess_times_out(tmp_path: Path) -> None:
    script = _executable(tmp_path, "slow", "#!/bin/sh\nexec sleep 5\n")
    with pytest.raises(subprocess.TimeoutExpired):
        run_subprocess([str(script)], input=None, env={"PATH": "/usr/bin:/bin"}, timeout=0.2)


# --- `pup format` runner (used by output.emit) ----------------------------------------------


def test_run_pup_uses_list_args_stdin_and_inherited_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _executable(tmp_path, script='#!/bin/sh\nprintf "%s|%s|" "$1" "$PUP_OUTPUT"\ncat\n')
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("PUP_OUTPUT", "yaml")
    completed = run_pup(["format"], '{"a": 1}')
    assert completed.returncode == 0
    assert completed.stdout == 'format|yaml|{"a": 1}'


def test_run_pup_honors_the_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = _executable(tmp_path / "custom", "my-pup", "#!/bin/sh\nprintf custom\n")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("PUP_FINETUNE_PUP", str(custom))
    assert run_pup(["format"], "{}").stdout == "custom"


def test_run_pup_without_pup_raises_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="pup not found on PATH"):
        run_pup(["format"], "{}")
