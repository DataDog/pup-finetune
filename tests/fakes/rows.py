# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fixture rows with test-split rows edited: a prompt refused by the fake renderer, a row
opened by an assistant turn, or every test row cut to one assistant turn."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pup_finetune.cli import app
from pup_finetune.data.dataset_dir import DatasetDir

ROWS = Path(__file__).resolve().parent.parent / "fixtures" / "rows" / "tool_use.jsonl"


def import_with_edited_test_row(
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    name: str = "ds",
    *,
    every: bool = False,
) -> tuple[Path, str]:
    """``tool_use.jsonl`` imported into ``tmp_path / name`` with ``edit`` applied to its first
    test-split row (to every one with ``every``); and the first one's id. Every other row, and
    every split, is unchanged."""
    probe = tmp_path / f"{name}-probe"
    imported = CliRunner().invoke(app, ["dataset", "import", str(probe), "--file", str(ROWS)])
    assert imported.exit_code == 0, imported.stderr
    test_ids = [row.id for row in DatasetDir(probe).rows_for_split("test")]
    edited_id = test_ids[0]
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["id"] in (test_ids if every else [edited_id]):
            edit(row)
    edited = tmp_path / f"{name}.jsonl"
    edited.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = tmp_path / name
    imported = CliRunner().invoke(app, ["dataset", "import", str(path), "--file", str(edited)])
    assert imported.exit_code == 0, imported.stderr
    assert DatasetDir(path).rows_for_split("test")[0].id == edited_id
    return path, edited_id


def _refused_by_the_fake_renderer(row: dict[str, Any]) -> None:
    row["messages"][0]["content"] += " <|end|>"


def import_with_refused_test_row(tmp_path: Path, name: str = "ds") -> tuple[Path, str]:
    """The fixture rows with the first test-split row's first message ending in ``<|end|>``,
    a template marker the fake renderer refuses in any prompt (see
    :func:`import_with_edited_test_row`)."""
    return import_with_edited_test_row(tmp_path, _refused_by_the_fake_renderer, name)


def _opened_by_the_assistant(row: dict[str, Any]) -> None:
    first = next(i for i, m in enumerate(row["messages"]) if m["role"] == "assistant")
    del row["messages"][:first]


def import_with_opening_assistant_test_row(tmp_path: Path, name: str = "ds") -> tuple[Path, str]:
    """The fixture rows with the first test-split row's messages before its first assistant
    turn dropped, so that turn has no prompt (see :func:`import_with_edited_test_row`)."""
    return import_with_edited_test_row(tmp_path, _opened_by_the_assistant, name)


def _the_first_assistant_turn_alone(row: dict[str, Any]) -> None:
    first = next(m for m in row["messages"] if m["role"] == "assistant")
    row["messages"] = [{"role": "assistant", "content": first["content"] or "Done."}]


def import_with_unprompted_test_split(tmp_path: Path, name: str = "ds") -> Path:
    """The fixture rows with every test-split row cut to its first assistant turn alone, so
    no test case has a message before it (see :func:`import_with_edited_test_row`)."""
    path, _ = import_with_edited_test_row(
        tmp_path, _the_first_assistant_turn_alone, name, every=True
    )
    return path
