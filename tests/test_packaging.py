# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Packaging config: the sdist ships an explicit allowlist, never the whole working tree."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _sdist_include() -> list[Any]:
    config = tomllib.loads(_PYPROJECT.read_text())
    sdist = config.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
    include = sdist.get("sdist", {}).get("include")
    assert isinstance(include, list), "the sdist target needs an explicit include allowlist"
    return include


def test_sdist_include_is_an_explicit_allowlist() -> None:
    include = _sdist_include()
    assert {"/src", "/tests", "/pyproject.toml", "/README.md"} <= set(include)
    for entry in include:
        assert isinstance(entry, str)
        # Anchored, literal paths only: a bare or wildcard pattern could match untracked files.
        assert entry.startswith("/"), entry
        assert not any(char in entry for char in "*?["), entry
        assert entry != "/", entry
