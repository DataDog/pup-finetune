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
from dataclasses import dataclass


def _text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value or None


@dataclass(frozen=True, slots=True)
class Runtime:
    """Non-secret settings read from the environment once per command."""

    output_format: str | None = None
    """``PUP_OUTPUT``: set by pup when it dispatches us; ``None`` when run standalone."""
    jq_filter: str | None = None
    """``PUP_FILTER``: the user's ``--jq`` expression, applied by ``pup format``."""
    read_only: bool = False
    """``PUP_READ_ONLY=true``: reject every remote write."""
    auto_approve: bool = False
    """``PUP_AUTO_APPROVE=true``. Never satisfies ``--confirm`` for paid or remote actions."""
    agent_mode: bool = False
    """``PUP_AGENT_MODE=true``: a coding agent is driving pup."""
    dd_site: str | None = None
    """``DD_SITE``: the Datadog site, e.g. ``datadoghq.com``."""
    dev_mode: bool = False
    """``PUP_FINETUNE_DEV=1``: running from a development checkout."""
    debug: bool = False
    """``PUP_FINETUNE_DEBUG=1``: print tracebacks for unexpected errors."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Runtime:
        """Read the runtime from ``env`` (default: ``os.environ``)."""
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            output_format=_text(source, "PUP_OUTPUT"),
            jq_filter=_text(source, "PUP_FILTER"),
            read_only=source.get("PUP_READ_ONLY") == "true",
            auto_approve=source.get("PUP_AUTO_APPROVE") == "true",
            agent_mode=source.get("PUP_AGENT_MODE") == "true",
            dd_site=_text(source, "DD_SITE"),
            dev_mode=source.get("PUP_FINETUNE_DEV") == "1",
            debug=source.get("PUP_FINETUNE_DEBUG") == "1",
        )

    @property
    def under_pup(self) -> bool:
        """Whether pup dispatched this process (so results go through ``pup format``)."""
        return self.output_format is not None
