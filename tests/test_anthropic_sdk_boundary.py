# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Only ``anthropic_client.py`` loads the Anthropic SDK (and its ``httpx2``), only when used."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pup_finetune

SRC = Path(pup_finetune.__file__).parent
REPO = Path(__file__).resolve().parents[1]
CLIENT = SRC / "anthropic_client.py"
SDK_MODULES = frozenset({"anthropic", "httpx2"})
_DYNAMIC_IMPORTS = frozenset({"import_module", "__import__"})


def _is_sdk(name: str | None) -> bool:
    return name is not None and name.split(".", 1)[0] in SDK_MODULES


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
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if path != CLIENT and sdk_imports(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_check_finds_sdk_imports() -> None:
    for line in (
        "import anthropic",
        "import os, httpx2",
        "from anthropic import Anthropic",
        "def f():\n    import anthropic.types",
        'importlib.import_module("anthropic")',
        '__import__("httpx2")',
    ):
        assert sdk_imports(line), line
    for line in (
        "import httpx",
        "from pup_finetune import anthropic_client",
        'importlib.import_module("pup_finetune.anthropic_client")',
    ):
        assert not sdk_imports(line), line


def _loaded_after(code: str) -> list[str]:
    script = (
        f"import json, sys\n{code}\n"
        "heavy = {'anthropic', 'httpx2'}\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & heavy)))\n"
    )
    out = subprocess.run(  # noqa: S603 - a fixed script run by this interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO
    )
    loaded: list[str] = json.loads(out.stdout.strip().splitlines()[-1])
    return loaded


def test_loading_the_cli_and_parsing_an_anthropic_judge_loads_no_sdk() -> None:
    loaded = _loaded_after(
        "import pup_finetune.cli\n"
        "from pup_finetune.judges import check_available, parse_judge\n"
        "from pup_finetune.curation import triage\n"
        "config = parse_judge('anthropic/claude-opus-5-5')\n"
        "check_available(config)\n"
        "triage.parse_judges('anthropic/claude-opus-5-5,openai/gpt-x')\n"
        "from pup_finetune.evaluation import runner\n"
    )
    assert loaded == []
