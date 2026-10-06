# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake ``pup`` process runner for :class:`pup_finetune.datadog.pup.Pup`.

Register responses by the arguments that follow the fixed flags, matched by prefix (the
longest registered prefix wins)::

    fake = FakePup()
    fake.on("llm-obs", "experiments", "list", json={"data": []})
    fake.on("api", "v2/llm-obs/v1/spans/events/search", stdout=b"")
    fake.on("auth", "status", returncode=1, stderr="Error: not logged in")
    fake.on("llm-obs", "datasets", "list", times_out=True)
    fake.on("--version", raw=True, stdout="pup 1.10.7\n")
    fake.on("api", "v2/some/route", handler=lambda call: {"echo": call.stdin_json()})
    pup = fake.client()

``handler`` computes the answer from the :class:`PupCall`: its return value is printed as JSON,
or, when it is a :class:`PupExit`, pup exits with that code and stderr. ``raw=True`` answers
calls made without the fixed flags (only ``pup --version``); every other call must carry them.
Every call is recorded (argv, env, stdin, timeout) for assertions. A call with no registered
response exits 1, the way pup rejects an unknown command.
"""

from __future__ import annotations

import json as jsonlib
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pup_finetune.datadog.pup import FIXED_FLAGS, Pup

FAKE_BINARY = "/fake/bin/pup"

_UNSET: Any = object()


@dataclass(frozen=True)
class PupCall:
    """One recorded invocation."""

    argv: list[str]
    env: dict[str, str]
    stdin: bytes | None
    timeout: float

    @property
    def raw(self) -> bool:
        """Whether the call was made without the fixed flags."""
        return tuple(self.argv[1 : 1 + len(FIXED_FLAGS)]) != FIXED_FLAGS

    @property
    def args(self) -> tuple[str, ...]:
        """The arguments after the executable and the fixed flags (if present)."""
        return tuple(self.argv[1 if self.raw else 1 + len(FIXED_FLAGS) :])

    def stdin_json(self) -> Any:
        assert self.stdin is not None, "the call had no stdin"
        return jsonlib.loads(self.stdin)


@dataclass(frozen=True)
class PupExit:
    """What a ``handler`` returns to make pup fail instead of printing JSON."""

    returncode: int = 1
    stderr: str = "Error: fake pup failure"


Handler = Callable[[PupCall], Any]


@dataclass(frozen=True)
class _Response:
    returncode: int
    stdout: bytes
    stderr: bytes
    raises: BaseException | None
    times_out: bool
    handler: Handler | None = None


def _as_bytes(value: str | bytes) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else value


class FakePup:
    """A :class:`~pup_finetune.datadog.pup.PupRunner` answering from registered responses."""

    def __init__(self, binary: str = FAKE_BINARY) -> None:
        self.binary = binary
        self.calls: list[PupCall] = []
        self._responses: dict[tuple[bool, tuple[str, ...]], _Response] = {}

    def on(
        self,
        *prefix: str,
        json: Any = _UNSET,
        stdout: str | bytes = b"",
        stderr: str | bytes = b"",
        returncode: int = 0,
        raises: BaseException | None = None,
        times_out: bool = False,
        raw: bool = False,
        handler: Handler | None = None,
    ) -> FakePup:
        """Answer calls whose arguments start with ``prefix``.

        ``json`` is serialized to stdout (it overrides ``stdout``). ``raises`` is raised from
        the runner (e.g. ``OSError``); ``times_out`` raises ``subprocess.TimeoutExpired``.
        ``raw`` answers calls made without the fixed flags instead of calls with them.
        ``handler`` answers each call from the call itself (see the module docstring).
        """
        if json is not _UNSET:
            stdout = jsonlib.dumps(json)
        self._responses[(raw, tuple(prefix))] = _Response(
            returncode=returncode,
            stdout=_as_bytes(stdout),
            stderr=_as_bytes(stderr),
            raises=raises,
            times_out=times_out,
            handler=handler,
        )
        return self

    def client(self, environ: Mapping[str, str] | None = None) -> Pup:
        """A :class:`Pup` that runs through this fake (``environ`` defaults to empty)."""
        return Pup(runner=self, binary=self.binary, environ={} if environ is None else environ)

    def __call__(
        self, argv: list[str], *, input: bytes | None, env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        call = PupCall(argv=list(argv), env=dict(env), stdin=input, timeout=timeout)
        self.calls.append(call)
        assert argv[0] == self.binary, f"unexpected executable {argv[0]!r}"
        response = self._match(call.raw, call.args)
        if response is None:
            kind = "raw call" if call.raw else "call"
            message = f"Error: fake pup has no response for {kind}: {' '.join(call.args)}\n"
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=message.encode())
        if response.times_out:
            raise subprocess.TimeoutExpired(argv, timeout)
        if response.raises is not None:
            raise response.raises
        if response.handler is not None:
            answer = response.handler(call)
            if isinstance(answer, PupExit):
                stderr = answer.stderr.encode()
                return subprocess.CompletedProcess(argv, answer.returncode, b"", stderr)
            stdout = jsonlib.dumps(answer).encode()
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")
        return subprocess.CompletedProcess(
            argv, response.returncode, stdout=response.stdout, stderr=response.stderr
        )

    @property
    def last(self) -> PupCall:
        """The most recent call."""
        assert self.calls, "pup was never called"
        return self.calls[-1]

    def _match(self, raw: bool, args: tuple[str, ...]) -> _Response | None:
        matches = [
            prefix
            for is_raw, prefix in self._responses
            if is_raw == raw and args[: len(prefix)] == prefix
        ]
        if not matches:
            return None
        return self._responses[(raw, max(matches, key=len))]
