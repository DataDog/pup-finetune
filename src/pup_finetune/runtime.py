# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The runtime environment: what pup and the user told this process through env vars.

Only non-secret settings live here. Credentials are resolved elsewhere and never stored on
:class:`Runtime`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

AGENT_ENV_VARS: tuple[str, ...] = ("AI_AGENT", "CLAUDECODE")
"""Variables a coding agent sets for the commands it runs (besides pup's ``PUP_AGENT_MODE``)."""


OFF_VALUES: Final = frozenset({"", "0", "false", "no", "off"})
"""Values (case-insensitive) of a yes/no variable that mean off."""
ON_VALUES: Final = frozenset({"1", "true", "yes", "on"})
"""Values (case-insensitive) of a yes/no variable that mean on."""


def _text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value or None


def _read_only(env: Mapping[str, str], warnings: list[str]) -> bool:
    """``PUP_READ_ONLY``: read-only unless unset or off. A value that is neither on nor off
    counts as read-only (the safe reading), with a warning."""
    value = env.get("PUP_READ_ONLY")
    if value is None or value.strip().lower() in OFF_VALUES:
        return False
    if value.strip().lower() not in ON_VALUES:
        warnings.append(
            f"PUP_READ_ONLY={value!r} is neither on ({', '.join(sorted(ON_VALUES))}) nor off "
            f"({', '.join(sorted(OFF_VALUES - {''}))}, or empty): treating it as read-only"
        )
    return True


def _switch(env: Mapping[str, str], name: str, warnings: list[str]) -> bool:
    """A yes/no variable that is on only when set to an on value; anything else is off,
    with a warning for a non-empty value that is neither on nor off."""
    value = env.get(name)
    if value is None:
        return False
    text = value.strip().lower()
    if text in ON_VALUES:
        return True
    if text not in OFF_VALUES:
        warnings.append(
            f"{name}={value!r} is neither on ({', '.join(sorted(ON_VALUES))}) nor off: "
            "treating it as off"
        )
    return False


ProgressMode = Literal["auto", "steps"]
"""How many progress lines long commands print; see :attr:`Runtime.progress`."""


def _progress_mode(env: Mapping[str, str], warnings: list[str]) -> ProgressMode:
    """``PUP_FINETUNE_PROGRESS``: ``steps`` (any case), or ``auto`` when unset or empty; any
    other value is ``auto``, with a warning."""
    value = env.get("PUP_FINETUNE_PROGRESS")
    text = (value or "").strip().lower()
    if text == "steps":
        return "steps"
    if text and text != "auto":
        warnings.append(
            f"PUP_FINETUNE_PROGRESS={value!r} is neither steps nor auto: treating it as auto"
        )
    return "auto"


@dataclass(frozen=True, slots=True)
class Runtime:
    """Non-secret settings read from the environment once per command."""

    output_format: str | None = None
    """``PUP_OUTPUT``: set by pup when it dispatches us; ``None`` when run standalone."""
    jq_filter: str | None = None
    """``PUP_FILTER``: the user's ``--jq`` expression, applied by ``pup format``."""
    read_only: bool = False
    """``PUP_READ_ONLY`` set to anything but an off value (``0``, ``false``, ``no``, ``off``,
    any case, or empty): reject every remote write. pup sets it for ``pup --read-only``."""
    agent_mode: bool = False
    """``PUP_AGENT_MODE=true``: a coding agent is driving pup."""
    agent_env: bool = False
    """``AI_AGENT`` or ``CLAUDECODE`` set (non-empty): a coding agent started this process,
    through pup or not."""
    dev_mode: bool = False
    """``PUP_FINETUNE_DEV`` set to an on value (``1``, ``true``, ``yes``, ``on``, any case):
    running from a development checkout."""
    debug: bool = False
    """``PUP_FINETUNE_DEBUG=1``: print tracebacks for unexpected errors."""
    quiet: bool = False
    """``PUP_FINETUNE_QUIET`` set to an on value: no closing summary on stderr after the
    result (warnings are still printed)."""
    progress: ProgressMode = "auto"
    """``PUP_FINETUNE_PROGRESS``: ``steps`` prints a progress line per training step and per
    evaluated pair, as on a terminal; ``auto`` (unset) throttles them off a terminal."""
    warnings: tuple[str, ...] = field(default=(), compare=False)
    """What the environment holds that was read in the safe direction (e.g. an unrecognized
    ``PUP_READ_ONLY`` value); each command prints these once on stderr."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Runtime:
        """Read the runtime from ``env`` (default: ``os.environ``)."""
        source: Mapping[str, str] = os.environ if env is None else env
        warnings: list[str] = []
        read_only = _read_only(source, warnings)
        dev_mode = _switch(source, "PUP_FINETUNE_DEV", warnings)
        quiet = _switch(source, "PUP_FINETUNE_QUIET", warnings)
        progress = _progress_mode(source, warnings)
        return cls(
            output_format=_text(source, "PUP_OUTPUT"),
            jq_filter=_text(source, "PUP_FILTER"),
            read_only=read_only,
            agent_mode=source.get("PUP_AGENT_MODE") == "true",
            agent_env=any(source.get(name) for name in AGENT_ENV_VARS),
            dev_mode=dev_mode,
            debug=source.get("PUP_FINETUNE_DEBUG") == "1",
            quiet=quiet,
            progress=progress,
            warnings=tuple(warnings),
        )

    @property
    def under_pup(self) -> bool:
        """Whether pup dispatched this process (so results go through ``pup format``)."""
        return self.output_format is not None

    @property
    def under_agent(self) -> bool:
        """Whether a coding agent is driving this process: ``PUP_AGENT_MODE`` is true, or
        ``AI_AGENT`` or ``CLAUDECODE`` is set. Use it for defaults that suit an agent's
        bounded tool calls; it never satisfies ``--confirm``."""
        return self.agent_mode or self.agent_env
