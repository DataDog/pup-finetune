# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fixture rows with one test-split row whose prompt the fake renderer refuses."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pup_finetune.cli import app
from pup_finetune.data.dataset_dir import DatasetDir

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"


def import_with_refused_test_row(tmp_path: Path, name: str = "ds") -> tuple[Path, str]:
    """``tool_use.jsonl`` imported into ``tmp_path / name``, its first test-split row's first
    message ending in ``<|end|>``, a template marker the fake renderer refuses in any prompt;
    and that row's id. Every other row, and every split, is unchanged."""
    probe = tmp_path / f"{name}-probe"
    imported = CliRunner().invoke(app, ["dataset", "import", str(probe), "--file", str(ROWS)])
    assert imported.exit_code == 0, imported.stderr
    refused = DatasetDir(probe).rows_for_split("test")[0].id
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["id"] == refused:
            row["messages"][0]["content"] += " <|end|>"
    edited = tmp_path / f"{name}.jsonl"
    edited.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = tmp_path / name
    imported = CliRunner().invoke(app, ["dataset", "import", str(path), "--file", str(edited)])
    assert imported.exit_code == 0, imported.stderr
    assert DatasetDir(path).rows_for_split("test")[0].id == refused
    return path, refused
