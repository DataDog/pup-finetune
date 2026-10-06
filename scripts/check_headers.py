#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fail when a git-tracked Python file or shell script lacks the license header.

Usage: python scripts/check_headers.py

Checked: ``*.py``, ``*.sh`` and every file under ``bin/``. A shebang line may precede the header.
Exempt: vendored code, test fixtures and the generated version file. Stdlib only.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

HEADER: tuple[str, ...] = (
    "# SPDX-License-Identifier: Apache-2.0",
    "#",
    "# Unless explicitly stated otherwise all files in this repository are licensed under the "
    "Apache License Version 2.0.",
    "#",
    "# This product includes software developed at Datadog (https://www.datadoghq.com/).",
    "# Copyright 2026-present Datadog, Inc.",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

EXEMPT_DIRS: tuple[str, ...] = ("src/pup_finetune/_vendor/", "tests/fixtures/")
EXEMPT_FILES: frozenset[str] = frozenset({"src/pup_finetune/_version.py"})


def is_exempt(path: str) -> bool:
    """Whether a repo-relative POSIX path is excluded from the check."""
    return path in EXEMPT_FILES or path.startswith(EXEMPT_DIRS)


def requires_header(path: str) -> bool:
    """Whether a repo-relative POSIX path must carry the license header."""
    if is_exempt(path):
        return False
    return path.endswith((".py", ".sh")) or path.startswith("bin/")


def has_header(text: str) -> bool:
    """Whether ``text`` starts with the exact header, optionally after a shebang line."""
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    return tuple(lines[: len(HEADER)]) == HEADER


def find_violations(root: Path, paths: Iterable[str]) -> list[str]:
    """The paths, relative to ``root``, that require the header but lack it (sorted)."""
    missing: list[str] = []
    for path in paths:
        if not requires_header(path):
            continue
        file = root / path
        if not file.is_file():  # tracked but deleted in the working tree
            continue
        text = file.read_bytes().decode("utf-8", errors="replace")
        if not has_header(text):
            missing.append(path)
    return sorted(missing)


def tracked_files(root: Path) -> list[str]:
    """Repo-relative paths of every file git tracks under ``root``."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found on PATH")
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [git, "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def main(root: Path = REPO_ROOT) -> int:
    """Check every tracked file under ``root``; print offenders to stderr and return 1 if any."""
    missing = find_violations(root, tracked_files(root))
    if not missing:
        return 0
    print("Missing or incorrect license header:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
