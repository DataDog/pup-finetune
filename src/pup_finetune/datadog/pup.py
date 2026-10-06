# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The pup boundary: the only module that spawns ``pup``.

Internal calls run ``pup --no-agent --output json <args>`` with list arguments (never a shell),
a timeout and a scrubbed environment, and parse stdout as JSON. Failures become
:class:`~pup_finetune.errors.ExternalError` with a trimmed message that never echoes the
environment. The process runner is injectable so tests can use a fake ``pup``.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pup_finetune.errors import ExternalError

# The oldest pup release verified to have every command we use (`llm-obs` spans, datasets and
# experiments, `api`, `auth status` and `format`).
MIN_PUP_VERSION: tuple[int, int, int] = (1, 10, 7)

PUP_BINARY_ENV = "PUP_FINETUNE_PUP"
"""Path (or name on ``PATH``) of the pup executable to use instead of ``pup`` on ``PATH``."""
KEEP_DD_TOKEN_ENV = "PUP_FINETUNE_KEEP_DD_TOKEN"  # noqa: S105 - an env var name, not a secret
"""``1`` passes ``DD_ACCESS_TOKEN`` through to child pup processes (env-only authentication)."""

FIXED_FLAGS: tuple[str, ...] = ("--no-agent", "--output", "json")
"""Flags placed before the arguments of every internal pup call."""

INSTALL_HINT = (
    "install pup with `brew install datadog/pack/pup` (or see the pup README), "
    f"or set {PUP_BINARY_ENV} to its path"
)

DEFAULT_TIMEOUT_SECONDS = 120.0
_QUICK_TIMEOUT_SECONDS = 30.0
_FORMAT_TIMEOUT_SECONDS = 30.0

# pup applies these to its own output; internal reads must get plain JSON.
_OUTPUT_ENV: tuple[str, ...] = ("PUP_FILTER", "PUP_OUTPUT")
_TOKEN_ENV = "DD_ACCESS_TOKEN"  # noqa: S105 - an env var name, not a secret
# Values of these are redacted from any pup stderr we repeat.
_SECRET_ENV: tuple[str, ...] = ("DD_ACCESS_TOKEN", "DD_API_KEY", "DD_APP_KEY", "DD_APPLICATION_KEY")
_MIN_REDACT_LENGTH = 8
_MAX_DETAIL_LINES = 3
_MAX_DETAIL_CHARS = 300
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BEARER = re.compile(r"(?i)(bearer\s+)\S+")
_AUTH_FAILURE = re.compile(
    r"(?i)\b(401|403)\b|unauthori[sz]ed|forbidden|not authenticated|not logged in|auth login"
)
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class CompletedRun(Protocol):
    """What a :class:`PupRunner` returns; ``subprocess.CompletedProcess[bytes]`` fits."""

    @property
    def returncode(self) -> int: ...

    @property
    def stdout(self) -> bytes: ...

    @property
    def stderr(self) -> bytes: ...


class PupRunner(Protocol):
    """Runs ``argv`` (no shell) with ``input`` on stdin, exactly ``env`` and a timeout.

    Raises ``subprocess.TimeoutExpired`` on timeout and ``OSError`` when it cannot start.
    """

    def __call__(
        self, argv: list[str], *, input: bytes | None, env: dict[str, str], timeout: float
    ) -> CompletedRun: ...


def run_subprocess(
    argv: list[str], *, input: bytes | None, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[bytes]:
    """The default :class:`PupRunner`: ``subprocess.run`` with captured output, no shell."""
    return subprocess.run(  # noqa: S603 - resolved executable, list args, no shell
        argv,
        input=input,
        env=env,
        timeout=timeout,
        shell=False,
        check=False,
        capture_output=True,
    )


def find_pup(environ: Mapping[str, str] | None = None) -> str | None:
    """The pup executable: ``PUP_FINETUNE_PUP`` if set, else ``pup`` on ``PATH``; ``None`` if absent."""
    env = os.environ if environ is None else environ
    path = env.get("PATH")
    override = env.get(PUP_BINARY_ENV)
    return shutil.which(override or "pup", path=path)


def _missing_pup(environ: Mapping[str, str]) -> ExternalError:
    override = environ.get(PUP_BINARY_ENV)
    message = (
        f"{PUP_BINARY_ENV} does not name an executable: {override}"
        if override
        else "pup not found on PATH"
    )
    return ExternalError(message, source="pup", hint=INSTALL_HINT)


FormatRunner = Callable[[Sequence[str], str], "subprocess.CompletedProcess[str]"]
"""Runs ``pup <args>`` with the given text on stdin and returns the completed process."""


def run_pup(args: Sequence[str], stdin: str) -> subprocess.CompletedProcess[str]:
    """Run ``pup <args>`` as given, inheriting this process's environment unchanged.

    Only for ``pup format`` from :func:`pup_finetune.output.emit`, which needs ``PUP_OUTPUT``
    and ``PUP_FILTER`` to render the user's format. Use :class:`Pup` for everything else.
    Raises ``FileNotFoundError`` when pup cannot be found.
    """
    executable = find_pup()
    if executable is None:
        raise FileNotFoundError(_missing_pup(os.environ).message)
    return subprocess.run(  # noqa: S603 - resolved executable, list args, no shell
        [executable, *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=_FORMAT_TIMEOUT_SECONDS,
    )


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment for an internal pup call: a copy of ``environ``, scrubbed.

    ``PUP_FILTER`` and ``PUP_OUTPUT`` are removed so the user's ``--jq`` and output format never
    mangle our reads. ``DD_ACCESS_TOKEN`` is removed too, unless ``PUP_FINETUNE_KEEP_DD_TOKEN=1``:
    the token pup injects is short-lived, and without it the child pup loads and refreshes its
    own stored OAuth token, so long-running workers keep working.
    """
    env = dict(environ)
    for name in _OUTPUT_ENV:
        env.pop(name, None)
    if environ.get(KEEP_DD_TOKEN_ENV) != "1":
        env.pop(_TOKEN_ENV, None)
    return env


def parse_json_output(text: str) -> Any:
    """Parse pup's stdout as JSON, tolerating human-readable lines before the value.

    Strict parsing is tried first. If it fails, the first line that starts (at column 0) with
    ``{`` or ``[`` and begins a JSON value that runs to the end of the output, apart from
    whitespace, is parsed instead (some commands, such as ``pup auth status``, print status
    lines before the JSON). Indented lines are never candidates, so truncated pretty-printed
    output is an error rather than one of its inner fragments. Raises ``json.JSONDecodeError``
    when there is no such value.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        strict_error = exc
    decoder = json.JSONDecoder()
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.startswith(("{", "[")):
            try:
                value, end = decoder.raw_decode(text, offset)
            except json.JSONDecodeError:
                pass
            else:
                if not text[end:].strip():
                    return value
        offset += len(line)
    raise strict_error


def parse_version(text: str) -> tuple[int, int, int] | None:
    """The first ``X.Y.Z`` in ``text`` (e.g. ``pup 1.10.7``), or ``None``."""
    match = _VERSION.search(text)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return (major, minor, patch)


def format_version(version: tuple[int, int, int]) -> str:
    """``(1, 10, 7)`` as ``"1.10.7"``."""
    return ".".join(str(part) for part in version)


def _describe(args: Sequence[str]) -> str:
    """A short, secret-free label for a call: its leading subcommand words."""
    words = list(itertools.takewhile(lambda arg: not arg.startswith("-"), args))[:4]
    return " ".join(words or args[:1])


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class Pup:
    """Runs pup for internal reads and writes.

    ``binary`` and ``environ`` default to :func:`find_pup` and ``os.environ``, read when a call
    is made. Constructing a ``Pup`` never fails; calls raise :class:`ExternalError` when pup is
    missing.
    """

    def __init__(
        self,
        *,
        runner: PupRunner = run_subprocess,
        binary: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._runner = runner
        self._binary = binary
        self._environ = environ

    @property
    def environ(self) -> Mapping[str, str]:
        """The environment child processes are derived from."""
        return os.environ if self._environ is None else self._environ

    def executable(self) -> str:
        """The pup executable; raises :class:`ExternalError` with an install hint if absent."""
        if self._binary is None:
            found = find_pup(self.environ)
            if found is None:
                raise _missing_pup(self.environ)
            self._binary = found
        return self._binary

    def run_json(
        self,
        args: Sequence[str],
        *,
        input: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Any:
        """Run ``pup --no-agent --output json <args>`` and return its parsed JSON output.

        ``input``, when given, is sent JSON-encoded on stdin. Returns ``None`` when pup prints
        nothing. Raises :class:`ExternalError` on a non-zero exit, a timeout or invalid JSON.
        """
        stdin = None if input is None else json.dumps(input, allow_nan=False).encode("utf-8")
        completed = self._run(args, stdin=stdin, timeout=timeout)
        stdout = _decode(completed.stdout)
        if not stdout.strip():
            return None
        try:
            return parse_json_output(stdout)
        except json.JSONDecodeError as exc:
            raise ExternalError(
                f"`pup {_describe(args)}` did not print valid JSON", source="pup"
            ) from exc

    def api(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        *,
        content_type: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Any:
        """Call a Datadog API route through ``pup api`` (pup's OAuth); returns the JSON body.

        ``path`` is relative to the site, e.g. ``v2/llm-obs/v1/spans/events/search``. ``body``,
        when given, is sent as JSON on stdin. A 2xx response with an empty body returns ``None``.
        """
        if path.startswith("-"):
            raise ValueError(f"API path must not start with '-': {path!r}")
        args = ["api", path, "-X", method.upper()]
        if body is None:
            return self.run_json(args, timeout=timeout)
        args += ["--input", "-"]
        if content_type:
            args += ["-H", f"Content-Type: {content_type}"]
        return self.run_json(args, input=body, timeout=timeout)

    def version(self) -> tuple[int, int, int]:
        """The installed pup version, parsed from ``pup --version``."""
        completed = self._run(
            ["--version"], stdin=None, timeout=_QUICK_TIMEOUT_SECONDS, fixed_flags=False
        )
        text = _decode(completed.stdout).strip()
        version = parse_version(text)
        if version is None:
            first_line = text.splitlines()[0][:80] if text else ""
            raise ExternalError(
                f"could not parse a version from `pup --version` output {first_line!r}",
                source="pup",
            )
        return version

    def auth_status(self) -> dict[str, Any]:
        """pup's Datadog login: ``authenticated``, ``site``, ``org`` and ``expires_at`` only.

        Everything else ``pup auth status`` reports (token type, scopes, ...) is dropped, so
        nothing token-related reaches our output.
        """
        data = self.run_json(["auth", "status"], timeout=_QUICK_TIMEOUT_SECONDS)
        if not isinstance(data, dict):
            raise ExternalError("`pup auth status` did not print a JSON object", source="pup")
        return {
            "authenticated": data.get("authenticated") is True,
            "site": _optional_str(data.get("site")),
            "org": _optional_str(data.get("org")),
            "expires_at": _optional_str(data.get("expires_at")),
        }

    def _run(
        self,
        args: Sequence[str],
        *,
        stdin: bytes | None,
        timeout: float,
        fixed_flags: bool = True,
    ) -> CompletedRun:
        """Run pup; ``fixed_flags=False`` runs ``pup <args>`` exactly (for ``--version``)."""
        argv = [self.executable(), *(FIXED_FLAGS if fixed_flags else ()), *args]
        label = _describe(args)
        try:
            completed = self._runner(
                argv, input=stdin, env=child_env(self.environ), timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalError(
                f"`pup {label}` timed out after {timeout:g}s", source="pup"
            ) from exc
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise ExternalError(f"could not run `pup {label}`: {reason}", source="pup") from exc
        if completed.returncode != 0:
            # pup normally reports errors on stderr; fall back to stdout when stderr is empty.
            detail = self._error_detail(completed.stderr) or self._error_detail(completed.stdout)
            message = f"`pup {label}` failed (exit {completed.returncode})"
            hint = None
            if _AUTH_FAILURE.search(detail):
                hint = "check `pup auth status`, and run `pup auth login` to sign in again"
            raise ExternalError(
                f"{message}: {detail}" if detail else message, source="pup", hint=hint
            )
        return completed

    def _error_detail(self, output: bytes) -> str:
        """The first few non-empty output lines, without styling, secrets or bearer tokens."""
        text = _ANSI.sub("", _decode(output))
        for name in _SECRET_ENV:
            value = self.environ.get(name)
            if value and len(value) >= _MIN_REDACT_LENGTH:
                text = text.replace(value, "[redacted]")
        text = _BEARER.sub(r"\1[redacted]", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        detail = "; ".join(lines[:_MAX_DETAIL_LINES])
        if len(detail) > _MAX_DETAIL_CHARS:
            detail = detail[: _MAX_DETAIL_CHARS - 3] + "..."
        return detail
