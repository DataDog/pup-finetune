# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""docs/data-format.md: the hand-written parts match the code they describe."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pup_finetune.data.push import dataset_record
from pup_finetune.data.schema import validate_row
from pup_finetune.rendering.base import RendererErrorCode
from pup_finetune.training.datums import RejectionCode

DOC = Path(__file__).resolve().parents[1] / "docs" / "data-format.md"

# Never reported for a row: one makes the row render per turn, the other fails the whole run.
_NOT_ROW_CODES = {RendererErrorCode.NOT_PREFIX_STABLE, RendererErrorCode.TOKENIZER_UNAVAILABLE}


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _json_blocks() -> list[str]:
    return re.findall(r"^```json\n(.*?)^```$", _text(), flags=re.MULTILINE | re.DOTALL)


def _example_row() -> dict[str, Any]:
    row: dict[str, Any] = json.loads(_json_blocks()[0])
    return row


def _example_record() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(_json_blocks()[1])
    return record


def test_every_train_time_rejection_code_is_in_the_table() -> None:
    text = _text()
    codes = {code.value for code in RejectionCode}
    codes |= {code.value for code in RendererErrorCode if code not in _NOT_ROW_CODES}
    for code in sorted(codes):
        assert f"| `{code}` |" in text, f"{code} is missing from {DOC.name}"


def test_the_example_row_is_a_valid_row() -> None:
    row = validate_row(_example_row())
    assert row.id == "row-weather-001"


def test_the_example_record_is_what_push_builds_from_the_example_row() -> None:
    record = dataset_record(validate_row(_example_row()), "train")
    documented = _example_record()
    assert documented.keys() == record.keys()
    assert documented["input"].keys() == record["input"].keys()
    assert documented["expected_output"] == record["expected_output"]
    assert documented["metadata"] == record["metadata"]
    assert documented["tags"] == record["tags"]
