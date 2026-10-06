#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fail when LICENSE-3rdparty.csv is out of step with the dependencies in pyproject.toml.

Usage: python scripts/check_3rdparty.py

Checked: the CSV header is exactly ``Component,Origin,License,Copyright``; every direct
dependency in ``[project.dependencies]`` and every ``[project.optional-dependencies]`` group has
a row (matched by normalized distribution name; self-references such as ``pup-finetune[hf]``
are ignored); and every License value is an allowed SPDX identifier, or an ``OR`` expression
composed only of allowed identifiers. Packages in ``LICENSE_EXCEPTIONS`` may also use
``NOASSERTION``. Stdlib only.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

CSV_HEADER: tuple[str, ...] = ("Component", "Origin", "License", "Copyright")

ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "PSF-2.0",
        "MPL-2.0",
    }
)

# Normalized names allowed the SPDX value NOASSERTION, each with the reason.
LICENSE_EXCEPTIONS: dict[str, str] = {
    "baseten-loops": "publishes no license; Baseten asked to publish one",
}

_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def normalize(name: str) -> str:
    """The PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """The normalized distribution name of a PEP 508 requirement string."""
    match = _NAME.match(requirement)
    if match is None:
        raise ValueError(f"cannot parse requirement: {requirement!r}")
    return normalize(match.group(1))


def declared_dependencies(pyproject: dict[str, Any]) -> set[str]:
    """Normalized names of every direct and optional dependency, minus self-references."""
    project: dict[str, Any] = pyproject.get("project", {})
    requirements: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    own_name = normalize(str(project.get("name", "")))
    return {requirement_name(r) for r in requirements} - {own_name}


def license_allowed(expression: str) -> bool:
    """Whether an SPDX expression is one allowed identifier or an ``OR`` of allowed ones."""
    # Grouping parentheses carry no meaning in a pure OR-expression; AND and WITH
    # terms survive the split and fail the allowlist.
    flat = expression.replace("(", " ").replace(")", " ")
    terms = [term.strip() for term in flat.split(" OR ")]
    return all(term in ALLOWED_LICENSES for term in terms)


def find_problems(pyproject_text: str, csv_text: str) -> list[str]:
    """Every mismatch between ``pyproject.toml`` and ``LICENSE-3rdparty.csv`` contents."""
    problems: list[str] = []
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, [])
    if tuple(header) != CSV_HEADER:
        problems.append(f"CSV header must be {','.join(CSV_HEADER)!r}, got {','.join(header)!r}")
        return problems

    components: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(CSV_HEADER):
            problems.append(f"line {line_no}: expected {len(CSV_HEADER)} columns, got {len(row)}")
            continue
        component, _origin, license_expr, _copyright = row
        name = normalize(component.strip())
        components.add(name)
        excepted = license_expr == "NOASSERTION" and name in LICENSE_EXCEPTIONS
        if not excepted and not license_allowed(license_expr):
            problems.append(f"line {line_no}: {component}: license {license_expr!r} not allowed")

    pyproject = tomllib.loads(pyproject_text)
    problems.extend(
        f"{name}: dependency has no row in LICENSE-3rdparty.csv"
        for name in sorted(declared_dependencies(pyproject) - components)
    )
    return problems


def _report(problems: Iterable[str]) -> None:
    print("LICENSE-3rdparty.csv problems:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)


def main(root: Path = REPO_ROOT) -> int:
    """Check the files under ``root``; print problems to stderr and return 1 if any."""
    problems = find_problems(
        (root / "pyproject.toml").read_text(encoding="utf-8"),
        (root / "LICENSE-3rdparty.csv").read_text(encoding="utf-8"),
    )
    if not problems:
        return 0
    _report(problems)
    return 1


if __name__ == "__main__":
    sys.exit(main())
