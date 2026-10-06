# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Replay evaluation: redo each recorded assistant action and compare with the recording.

:mod:`pup_finetune.evaluation.cases` turns test-split rows into replay cases (the
recorded prefix and tools in, the recorded assistant message as the reference), and
:mod:`pup_finetune.evaluation.metrics` scores a generated message against the reference with
deterministic tool-call metrics. Tools are never executed, so replay measures agreement with
the recorded behavior, not task success. Evaluation is provider-neutral: nothing here
imports a provider.
"""

from __future__ import annotations

from pup_finetune.evaluation.cases import (
    ReplayCase,
    build_cases,
    case_indices,
    cases_from_dataset,
    has_visible_action,
    row_cases,
    select_spread,
)
from pup_finetune.evaluation.metrics import (
    MetricName,
    MetricResult,
    aggregate,
    arguments_json_valid,
    arguments_schema_valid,
    parallel_call_set_match,
    reference_arguments_match,
    schema_errors,
    score_case,
    tool_decision_match,
    tool_name_match,
)

__all__ = [
    "MetricName",
    "MetricResult",
    "ReplayCase",
    "aggregate",
    "arguments_json_valid",
    "arguments_schema_valid",
    "build_cases",
    "case_indices",
    "cases_from_dataset",
    "has_visible_action",
    "parallel_call_set_match",
    "reference_arguments_match",
    "row_cases",
    "schema_errors",
    "score_case",
    "select_spread",
    "tool_decision_match",
    "tool_name_match",
]
