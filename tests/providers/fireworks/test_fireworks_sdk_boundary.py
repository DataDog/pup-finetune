# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Only ``providers/fireworks/client.py`` loads the Fireworks SDK, and only when it is used."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pup_finetune

SRC = Path(pup_finetune.__file__).parent
REPO = Path(__file__).resolve().parents[3]
CLIENT = SRC / "providers" / "fireworks" / "client.py"
SDK_MODULES = frozenset({"fireworks", "tinker"})
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
        "import tinker",
        "import os, tinker",
        "from fireworks.training.sdk import FiretitanServiceClient",
        "def f():\n    import fireworks.training.sdk.patches",
        'importlib.import_module("tinker")',
        'import_module("fireworks.training.sdk")',
        '__import__("tinker")',
    ):
        assert sdk_imports(line), line
    for line in (
        "import tinkerer",
        "from pup_finetune.providers.fireworks import api",
        "from . import fireworks",
        'importlib.import_module("tinkerer")',
    ):
        assert not sdk_imports(line), line


def _loaded_after(code: str) -> list[str]:
    """The heavy packages loaded once ``code`` ran in a fresh interpreter."""
    script = (
        f"import json, sys\n{code}\n"
        "heavy = {'fireworks', 'tinker', 'transformers', 'torch'}\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & heavy)))\n"
    )
    out = subprocess.run(  # noqa: S603 - a fixed script run by this interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO
    )
    loaded: list[str] = json.loads(out.stdout.strip().splitlines()[-1])
    return loaded


def test_loading_the_cli_and_the_provider_loads_no_sdk() -> None:
    loaded = _loaded_after(
        "import pup_finetune.cli\n"
        "from pup_finetune.providers import get_provider\n"
        "from pup_finetune.runtime import Runtime\n"
        "provider = get_provider('fireworks', Runtime())\n"
        "provider.resolve_model(provider.models()[0].alias)\n"
        "import pup_finetune.providers.fireworks.client\n"
        "import tests.fakes.fireworks\n"
    )
    assert loaded == []
