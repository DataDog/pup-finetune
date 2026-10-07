# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Only ``providers/baseten/client.py`` loads the Baseten Loops SDK, and only when it is used."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pup_finetune

SRC = Path(pup_finetune.__file__).parent
REPO = Path(__file__).resolve().parents[3]
CLIENT = SRC / "providers" / "baseten" / "client.py"
_DYNAMIC_IMPORTS = frozenset({"import_module", "__import__"})


def _is_sdk(name: str | None) -> bool:
    return name is not None and name.split(".", 1)[0] == "baseten"


def sdk_imports(source: str) -> list[str]:
    """The SDK modules ``source`` imports: statements, ``import_module`` and ``__import__``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if _is_sdk(alias.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _is_sdk(node.module):
            found.append(str(node.module))
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if (
                name in _DYNAMIC_IMPORTS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and _is_sdk(first.value)
            ):
                found.append(first.value)
    return found


def test_only_the_client_imports_the_sdk() -> None:
    assert sdk_imports(CLIENT.read_text(encoding="utf-8"))
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if path != CLIENT and sdk_imports(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_check_finds_sdk_imports() -> None:
    for line in (
        "import baseten",
        "import os, baseten.loops",
        "from baseten.loops import ServiceClient",
        "def f():\n    import baseten.loops as sdk",
        'importlib.import_module("baseten.loops")',
        '__import__("baseten")',
    ):
        assert sdk_imports(line), line
    for line in (
        "import basetenx",
        "from pup_finetune.providers.baseten import client",
        "from . import baseten",
        'importlib.import_module("basetenx")',
    ):
        assert not sdk_imports(line), line


def test_importing_the_backend_and_its_fake_loads_no_sdk() -> None:
    script = (
        "import json, sys\n"
        "import pup_finetune.providers.baseten.backend\n"
        "import tests.fakes.baseten\n"
        "print(json.dumps('baseten' in sys.modules))\n"
    )
    out = subprocess.run(  # noqa: S603 - a fixed script run by this interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert json.loads(out.stdout) is False
