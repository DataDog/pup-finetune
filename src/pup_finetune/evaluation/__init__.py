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
two samplers from a training session, and :mod:`pup_finetune.evaluation.compare` the third,
the compare model of ``evaluate --compare-model``. Tools are never executed, so replay measures
agreement with the recorded behavior, not task success. Evaluation is provider-neutral:
nothing here imports a provider SDK or client; it samples through the
:class:`~pup_finetune.providers.base.Sampler` protocol, and the judge talks to its endpoint
through :mod:`pup_finetune.inference`. :mod:`pup_finetune.evaluation.runner` plans and runs an
evaluation for the ``evaluate`` and ``train --evaluate`` commands, and
:mod:`pup_finetune.evaluation.benchmark` replays the same cases with several base models for
``benchmark``.

Import names from their submodules. This package imports nothing, so importing the runner
(the CLI does, for every command) does not load the judge's HTTP client.
"""
