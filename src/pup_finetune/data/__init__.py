# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training data: the canonical row schema every provider consumes, and its JSONL files.

:mod:`pup_finetune.data.schema` defines :class:`TrainingRow` (OpenAI chat format plus its
source) and the stable :class:`RowErrorCode` reasons a row is rejected for;
:mod:`pup_finetune.data.jsonl` reads and writes rows and converts them for providers.
"""

from __future__ import annotations

from pup_finetune.data.jsonl import read_rows, row_to_dict, to_provider_dict, write_rows
from pup_finetune.data.schema import (
    SCHEMA_VERSION,
    FunctionCall,
    Message,
    RowError,
    RowErrorCode,
    SourceRef,
    Tool,
    ToolCall,
    ToolFunction,
    TrainingRow,
    row_content_hash,
    split_key,
    validate_row,
)

__all__ = [
    "SCHEMA_VERSION",
    "FunctionCall",
    "Message",
    "RowError",
    "RowErrorCode",
    "SourceRef",
    "Tool",
    "ToolCall",
    "ToolFunction",
    "TrainingRow",
    "read_rows",
    "row_content_hash",
    "row_to_dict",
    "split_key",
    "to_provider_dict",
    "validate_row",
    "write_rows",
]
