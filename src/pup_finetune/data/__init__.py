# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training data: the canonical row schema every provider consumes, and its JSONL files.

:mod:`pup_finetune.data.schema` defines :class:`TrainingRow` (OpenAI chat format plus its
source) and the stable :class:`RowErrorCode` reasons a row is rejected for;
:mod:`pup_finetune.data.jsonl` reads and writes rows and converts them for providers;
:mod:`pup_finetune.data.exclusions` records why a row was left out of a dataset;
:mod:`pup_finetune.data.convert` maps one LLM span to a canonical conversation and
:mod:`pup_finetune.data.trajectories` turns the spans of whole traces into rows;
:mod:`pup_finetune.data.split` assigns stable train / validation / test splits; and
:mod:`pup_finetune.data.dataset_dir` is the on-disk dataset directory.
"""

from __future__ import annotations

from pup_finetune.data.convert import Conversation, ReasoningMode, span_conversation
from pup_finetune.data.dataset_dir import AddResult, DatasetDir, DatasetSettings
from pup_finetune.data.exclusions import Exclusion, ExclusionCode, ExclusionReason
from pup_finetune.data.jsonl import read_rows, row_to_dict, scan_rows, to_provider_dict, write_rows
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
from pup_finetune.data.split import (
    SPLITS,
    SplitAssignments,
    SplitName,
    SplitParams,
    assign_split,
    check_isolation,
)
from pup_finetune.data.trajectories import (
    ConversionResult,
    ConversionStats,
    RowConversion,
    rows_from_spans,
)

__all__ = [
    "SCHEMA_VERSION",
    "SPLITS",
    "AddResult",
    "Conversation",
    "ConversionResult",
    "ConversionStats",
    "DatasetDir",
    "DatasetSettings",
    "Exclusion",
    "ExclusionCode",
    "ExclusionReason",
    "FunctionCall",
    "Message",
    "ReasoningMode",
    "RowConversion",
    "RowError",
    "RowErrorCode",
    "SourceRef",
    "SplitAssignments",
    "SplitName",
    "SplitParams",
    "Tool",
    "ToolCall",
    "ToolFunction",
    "TrainingRow",
    "assign_split",
    "check_isolation",
    "read_rows",
    "row_content_hash",
    "row_to_dict",
    "rows_from_spans",
    "scan_rows",
    "span_conversation",
    "split_key",
    "to_provider_dict",
    "validate_row",
    "write_rows",
]
