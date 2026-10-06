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
through :mod:`pup_finetune.inference`. :mod:`pup_finetune.evaluation.runner` plans and runs an
evaluation for the ``evaluate`` and ``train --evaluate`` commands.

The names below are imported on first use, so importing a submodule (the runner, which the
CLI loads for every command) does not load the judge's HTTP client. The ``judge`` function is
not among them: once :mod:`pup_finetune.evaluation.judge` is imported, that name is the
submodule, so import the function from it. ``__all__`` and ``_EXPORTS`` list the same names
(a test checks it).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORTS: dict[str, str] = {
    "CAVEAT": "pup_finetune.evaluation.replay",
    "CalibrationCheck": "pup_finetune.evaluation.judge",
    "CalibrationReport": "pup_finetune.evaluation.judge",
    "EvalDir": "pup_finetune.evaluation.replay",
    "Judgment": "pup_finetune.evaluation.judge",
    "MetricName": "pup_finetune.evaluation.metrics",
    "MetricResult": "pup_finetune.evaluation.metrics",
    "NoopPublisher": "pup_finetune.evaluation.replay",
    "Publisher": "pup_finetune.evaluation.replay",
    "ReplayCase": "pup_finetune.evaluation.cases",
    "ReplaySummary": "pup_finetune.evaluation.replay",
    "TARGETS": "pup_finetune.evaluation.replay",
    "aggregate": "pup_finetune.evaluation.metrics",
    "arguments_json_valid": "pup_finetune.evaluation.metrics",
    "arguments_schema_valid": "pup_finetune.evaluation.metrics",
    "build_cases": "pup_finetune.evaluation.cases",
    "calibrate": "pup_finetune.evaluation.judge",
    "case_indices": "pup_finetune.evaluation.cases",
    "cases_from_dataset": "pup_finetune.evaluation.cases",
    "check_judge_differs": "pup_finetune.evaluation.judge",
    "evaluation_identity": "pup_finetune.evaluation.replay",
    "has_visible_action": "pup_finetune.evaluation.cases",
    "paired_bootstrap_ci": "pup_finetune.evaluation.replay",
    "parallel_call_set_match": "pup_finetune.evaluation.metrics",
    "reference_arguments_match": "pup_finetune.evaluation.metrics",
    "row_cases": "pup_finetune.evaluation.cases",
    "run_replay": "pup_finetune.evaluation.replay",
    "schema_errors": "pup_finetune.evaluation.metrics",
    "score_case": "pup_finetune.evaluation.metrics",
    "select_spread": "pup_finetune.evaluation.cases",
    "summarize": "pup_finetune.evaluation.replay",
    "tool_decision_match": "pup_finetune.evaluation.metrics",
    "tool_name_match": "pup_finetune.evaluation.metrics",
}

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


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
