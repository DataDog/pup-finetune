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
deterministic tool-call metrics. :mod:`pup_finetune.evaluation.judge` asks an LLM judge,
through an OpenAI-compatible endpoint, whether a generated action is an acceptable
substitute for the reference, and calibrates that judge first
(:mod:`pup_finetune.evaluation.fake_judge` is an offline judge for development).
:mod:`pup_finetune.evaluation.replay` replays every case with the base and the tuned model,
resumably, and summarizes the comparison; :mod:`pup_finetune.evaluation.targets` makes those
two samplers from a training session. Tools are never executed, so replay measures
agreement with the recorded behavior, not task success. Evaluation is provider-neutral:
nothing here imports a provider SDK or client; it samples through the
:class:`~pup_finetune.providers.base.Sampler` protocol, and the judge talks to its endpoint
through :mod:`pup_finetune.inference`.
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
from pup_finetune.evaluation.judge import (
    CalibrationCheck,
    CalibrationReport,
    Judgment,
    calibrate,
    check_judge_differs,
    judge,
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
from pup_finetune.evaluation.replay import (
    CAVEAT,
    TARGETS,
    EvalDir,
    NoopPublisher,
    Publisher,
    ReplaySummary,
    evaluation_identity,
    paired_bootstrap_ci,
    run_replay,
    summarize,
)

__all__ = [
    "CAVEAT",
    "TARGETS",
    "CalibrationCheck",
    "CalibrationReport",
    "EvalDir",
    "Judgment",
    "MetricName",
    "MetricResult",
    "NoopPublisher",
    "Publisher",
    "ReplayCase",
    "ReplaySummary",
    "aggregate",
    "arguments_json_valid",
    "arguments_schema_valid",
    "build_cases",
    "calibrate",
    "case_indices",
    "cases_from_dataset",
    "check_judge_differs",
    "evaluation_identity",
    "has_visible_action",
    "judge",
    "paired_bootstrap_ci",
    "parallel_call_set_match",
    "reference_arguments_match",
    "row_cases",
    "run_replay",
    "schema_errors",
    "score_case",
    "select_spread",
    "summarize",
    "tool_decision_match",
    "tool_name_match",
]
