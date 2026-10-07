# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune evaluate`: replay a run's held-out test cases with the base and tuned model.

Evaluating is paid (it generates with both models and calls a judge), creates remote
sampling resources and publishes experiments to Datadog, so without ``--confirm`` it
previews the calls it would make (free, and allowed with ``PUP_READ_ONLY`` set) and prints
the exact command that runs it. With ``--confirm`` it runs
:meth:`~pup_finetune.evaluation.runner.StandaloneEvaluation.execute` and emits its result: the
judge is calibrated first (unless ``--skip-calibration``; a failed calibration stops before
any generation, with exit code 1 and the report), every case is replayed, and the complete
evaluation is published as Agent Observability experiments on the pushed dataset (unless
``--no-publish``; without a push receipt it is not published, with a warning). Ctrl-C or
SIGTERM stops it gracefully: the generations in flight finish, everything done is kept, and
running the command again continues where it stopped.

The planning, calibration, replay and publishing live in
:mod:`pup_finetune.evaluation.runner`, shared with ``train --evaluate``; this module only
parses the flags and emits the result.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, get_runtime, run_command
from pup_finetune.cli._dirs import require_run_dir
from pup_finetune.errors import require_confirm, require_writable
from pup_finetune.evaluation import runner

RunDirArgument = Annotated[
    Path,
    typer.Argument(
        metavar="RUN_DIR", help="A completed or cancelled run directory.", show_default=False
    ),
]


@run_command
def evaluate(
    run_dir: RunDirArgument,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Where the evaluation is written and resumed from (default: RUN_DIR/eval).",
            show_default=False,
        ),
    ] = None,
    max_cases_per_trajectory: Annotated[
        int,
        typer.Option(min=1, help="Most assistant turns replayed per test conversation."),
    ] = runner.DEFAULT_MAX_CASES_PER_TRAJECTORY,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge-model",
            help="The judge model: anthropic/<model> (ANTHROPIC_API_KEY, or the Anthropic "
            "SDK's own login), openai/<model> (OPENAI_API_KEY), or otherwise a Fireworks model "
            "id such as accounts/fireworks/models/<model> (FIREWORKS_API_KEY). "
            "Default: a Fireworks serverless model; the offline fake judge for a fake run.",
            show_default=False,
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option(min=1, max=64, help="Cases generated and judged at once.")
    ] = runner.DEFAULT_CONCURRENCY,
    max_tokens: Annotated[
        int, typer.Option(min=1, help="Most tokens each generation may produce.")
    ] = runner.DEFAULT_MAX_TOKENS,
    seed: Annotated[
        int, typer.Option(min=0, help="Seed for picking cases and the bootstrap.")
    ] = runner.DEFAULT_SEED,
    skip_calibration: Annotated[
        bool,
        typer.Option("--skip-calibration", help="Do not check the judge on known answers first."),
    ] = False,
    no_publish: Annotated[
        bool,
        typer.Option(
            "--no-publish",
            help="Do not publish the evaluation to Datadog as experiments (it is published "
            "when the dataset was pushed with `dataset push`).",
        ),
    ] = False,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Run it (paid); without it, only preview the calls.")
    ] = False,
) -> dict[str, Any] | CommandResult:
    runtime = get_runtime()
    require_run_dir(run_dir, "evaluate")
    settings = runner.EvalSettings(
        out_dir=out_dir if out_dir is not None else run_dir / runner.DEFAULT_OUT_DIR_NAME,
        max_cases_per_trajectory=max_cases_per_trajectory,
        judge_model=judge_model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
        skip_calibration=skip_calibration,
        publish=not no_publish,
    )
    evaluation = runner.plan_standalone(runtime, run_dir, settings)
    if confirm:
        require_writable(runtime, f"evaluating on {evaluation.provider.name}")
    rerun = [*evaluation.plan.evaluate_command(), "--confirm"]
    require_confirm(confirm, evaluation.preview(), shlex.join(rerun), read_only=runtime.read_only)
    data = evaluation.execute()
    code = runner.evaluation_exit_code(data)
    return CommandResult(data, exit_code=code) if code else data


def register(app: typer.Typer) -> None:
    """Add `evaluate` to ``app``."""
    app.command(
        "evaluate",
        short_help="Replay held-out conversations with the base and tuned model (paid).",
        help=(
            "Replay every held-out test conversation of a completed or cancelled run, one "
            "recorded assistant turn at a time, with the base model and the run's best "
            "checkpoint; score each answer with deterministic tool-call metrics and an LLM "
            "judge; and compare the two (pass rates, wins and regressions, and the "
            "tuned-minus-base delta with a bootstrap confidence interval). Tools are not "
            "executed, so the scores measure agreement with the recorded behavior, not task "
            "success. Without --confirm, previews the calls it would make. With --confirm, "
            "calibrates the judge first, then evaluates; rerun the same command to continue "
            "an evaluation that stopped. A complete evaluation is published to Datadog as "
            "two Agent Observability experiments on the dataset `dataset push` created "
            "(--no-publish skips it). The preview is allowed with PUP_READ_ONLY set; "
            "evaluating is not."
        ),
    )(evaluate)
