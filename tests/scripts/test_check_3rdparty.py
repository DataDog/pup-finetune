# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""scripts/check_3rdparty.py keeps LICENSE-3rdparty.csv in step with pyproject.toml."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_3rdparty.py"


def _load_script() -> ModuleType:
    # scripts/ is not a package (and is not shipped in the wheel), so load the file by path.
    spec = importlib.util.spec_from_file_location("check_3rdparty", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_3rdparty = _load_script()

_HEADER = "Component,Origin,License,Copyright\n"


def _pyproject(deps: list[str], extras: dict[str, list[str]] | None = None) -> str:
    # JSON arrays of strings are valid TOML arrays.
    lines = ["[project]", 'name = "pup-finetune"', f"dependencies = {json.dumps(deps)}"]
    if extras:
        lines.append("[project.optional-dependencies]")
        lines.extend(f"{group} = {json.dumps(reqs)}" for group, reqs in extras.items())
    return "\n".join(lines) + "\n"


def _row(component: str, license_expr: str = "MIT") -> str:
    return f"{component},https://pypi.org/project/{component}/,{license_expr},Example Authors\n"


def test_matching_files_pass() -> None:
    pyproject = _pyproject(["httpx>=0.28", "rich"], {"extra": ["tqdm>=4"]})
    csv_text = _HEADER + _row("httpx", "BSD-3-Clause") + _row("rich") + _row("tqdm")
    assert check_3rdparty.find_problems(pyproject, csv_text) == []


def test_missing_direct_dependency_is_reported() -> None:
    problems = check_3rdparty.find_problems(_pyproject(["httpx", "rich"]), _HEADER + _row("httpx"))
    assert problems == ["rich: dependency has no row in LICENSE-3rdparty.csv"]


def test_missing_optional_dependency_is_reported() -> None:
    pyproject = _pyproject(["httpx"], {"gpu": ["torch>=2; sys_platform == 'linux'"]})
    problems = check_3rdparty.find_problems(pyproject, _HEADER + _row("httpx"))
    assert problems == ["torch: dependency has no row in LICENSE-3rdparty.csv"]


def test_names_match_after_normalization() -> None:
    pyproject = _pyproject(["Typing_Extensions>=4", "ruamel.yaml[jinja2]==0.18"])
    csv_text = _HEADER + _row("typing-extensions") + _row("ruamel-yaml")
    assert check_3rdparty.find_problems(pyproject, csv_text) == []


def test_self_references_are_ignored() -> None:
    pyproject = _pyproject(["httpx"], {"hf": ["tqdm"], "all": ["pup-finetune[hf]", "PUP_finetune"]})
    csv_text = _HEADER + _row("httpx") + _row("tqdm")
    assert check_3rdparty.find_problems(pyproject, csv_text) == []


@pytest.mark.parametrize(
    "header",
    ["Component,Origin,License\n", "component,origin,license,copyright\n", "\n", ""],
)
def test_wrong_header_is_reported(header: str) -> None:
    problems = check_3rdparty.find_problems(_pyproject([]), header + _row("httpx"))
    assert len(problems) == 1
    assert problems[0].startswith("CSV header must be")


@pytest.mark.parametrize(
    "license_expr",
    [
        "MIT",
        "PSF-2.0",
        "MPL-2.0",
        "ISC",
        "Apache-2.0 OR MIT",
        "(BSD-2-Clause OR BSD-3-Clause)",
        "(MIT) OR (ISC)",
        "MIT OR (ISC OR BSD-3-Clause)",
    ],
)
def test_allowed_licenses_pass(license_expr: str) -> None:
    assert check_3rdparty.license_allowed(license_expr)


@pytest.mark.parametrize(
    "license_expr",
    [
        "GPL-3.0-only",
        "Apache-2.0 AND MIT",
        "MIT AND ISC",
        "MIT OR (ISC AND BSD-3-Clause)",
        "MIT OR GPL-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "",
        "()",
    ],
)
def test_disallowed_licenses_fail(license_expr: str) -> None:
    assert not check_3rdparty.license_allowed(license_expr)


def test_disallowed_license_row_is_reported() -> None:
    csv_text = _HEADER + _row("httpx", "GPL-3.0-only")
    problems = check_3rdparty.find_problems(_pyproject(["httpx"]), csv_text)
    assert problems == ["line 2: httpx: license 'GPL-3.0-only' not allowed"]


def test_noassertion_is_not_allowed() -> None:
    csv_text = _HEADER + _row("httpx", "NOASSERTION")
    problems = check_3rdparty.find_problems(_pyproject(["httpx"]), csv_text)
    assert problems == ["line 2: httpx: license 'NOASSERTION' not allowed"]


def test_row_with_wrong_column_count_is_reported() -> None:
    csv_text = _HEADER + "httpx,MIT\n"
    problems = check_3rdparty.find_problems(_pyproject([]), csv_text)
    assert problems == ["line 2: expected 4 columns, got 2"]


def test_quoted_fields_with_commas_parse() -> None:
    csv_text = _HEADER + 'rich,https://pypi.org/project/rich/,MIT,"Example, Inc."\n'
    assert check_3rdparty.find_problems(_pyproject(["rich"]), csv_text) == []


def test_main_passes_on_this_repository() -> None:
    assert check_3rdparty.main(_REPO_ROOT) == 0


def test_main_lists_problems_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(_pyproject(["httpx", "rich"]))
    (tmp_path / "LICENSE-3rdparty.csv").write_text(_HEADER + _row("httpx", "GPL-3.0-only"))
    assert check_3rdparty.main(tmp_path) == 1
    err = capsys.readouterr().err
    assert "rich: dependency has no row" in err
    assert "GPL-3.0-only" in err
