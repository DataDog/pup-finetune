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
running the command again continues where it stopped. ``--compare-model`` adds a third
target, another model replayed and judged alike; added to a complete evaluation, it reuses
the base and tuned results.

The planning, calibration, replay and publishing live in
:mod:`pup_finetune.evaluation.runner`, shared with ``train --evaluate``; this module only
parses the flags and emits the result.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, current_context, get_runtime, run_command
from pup_finetune.cli._dirs import RUN_DIR_BY_ID, resolve_run_ref
from pup_finetune.cli._flags import given, renamed
from pup_finetune.cli._presentation import (
    EVALUATE_AND_SHIP,
    CommandDoc,
    Safety,
    default_note,
    describe,
)
from pup_finetune.cli._views import evaluation_view
from pup_finetune.errors import PAID_RUN, require_confirm, require_writable
from pup_finetune.evaluation import runner
from pup_finetune.judge_models import BY_KEY

RunDirArgument = Annotated[
    Path,
    typer.Argument(
        metavar="RUN_DIR",
        help="A completed or cancelled run: its directory or run id.",
        show_default=False,
    ),
]

SETTING_PARAMS: dict[str, tuple[str, ...]] = {
    "judge_model": ("judge_model",),
    "max_cases": ("max_cases", "old_max_cases"),
    "max_tokens": ("max_tokens",),
    "concurrency": ("concurrency",),
    "seed": ("seed",),
    "calibration": ("skip_calibration",),
    "publish": ("no_publish",),
}
"""Each setting of an evaluation preview, and the parameters that set it."""


def _with_defaults_used(preview: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """``preview`` with ``defaults_used`` (the settings no flag set) after its ``settings``."""
    used = [
        name
        for name, params in SETTING_PARAMS.items()
        if not (ctx is not None and any(given(ctx, param) for param in params))
    ]
    shown: dict[str, Any] = {}
    for key, value in preview.items():
        shown[key] = value
        if key == "settings":
            shown["defaults_used"] = used
    return shown


def _with_source(result: dict[str, Any], source: str) -> dict[str, Any]:
    """``result`` with ``run_dir_source`` when the run was named by a bare run id: after its
    ``run_dir`` (a preview), else after its ``run_id``."""
    if source != RUN_DIR_BY_ID:
        return result
    after = "run_dir" if "run_dir" in result else "run_id"
    shown: dict[str, Any] = {}
    for key, value in result.items():
        shown[key] = value
        if key == after:
            shown["run_dir_source"] = source
    return shown


CASES = "Cases"
MODELS = "Models"
RUN = "Run"
"""``evaluate``'s option panels."""

DOC = CommandDoc(
    path="evaluate",
    short="Evaluate tuned vs base model (paid; previews without --confirm).",
    description=describe(
        "Replay each held-out test conversation of a run, one recorded assistant turn at a "
        "time, with the base model and the run's best checkpoint, and score both with "
        "tool-call metrics and an LLM judge. Without --confirm, previews the calls and where "
        "the data goes. A complete evaluation is published to Datadog as two experiments, "
        "three with --compare-model, when the dataset was pushed (--no-publish skips it).",
        "The result compares the two: pass rates, wins and regressions, and the "
        "tuned-minus-base delta with a bootstrap confidence interval. --compare-model, named "
        "as --judge-model is, replays a third model on the same cases, judged alike, with its "
        "own token limit (16000, thinking included) and the provider's lowest reasoning "
        "effort, and adds tuned-minus-compare; added to a complete evaluation, it reuses the "
        "base and tuned results and opens no training session. The directory keeps its "
        "compare model: a rerun without the flag continues it, and another one needs a new "
        "--out-dir. Tools are not executed, so the scores measure agreement with the "
        "recorded behavior, not task success. With --confirm, the judge is calibrated first, then every case is "
        "replayed; rerun the same command to continue an evaluation that stopped. A changed "
        "--max-tokens, judge, seed or --max-cases needs a new --out-dir. Without "
        "--judge-model, the judge is the one the --out-dir holds an evaluation by, else "
        "anthropic/claude-haiku-5-5 when an Anthropic credential resolves (ANTHROPIC_API_KEY, "
        "the OS keychain or the Anthropic SDK's login), else openai/gpt-5.4-mini when an "
        "OpenAI one does, else Fireworks' deepseek-v4p1-flash; the preview names it and "
        "where it came from (judge_source).",
    ),
    examples=(
        "pup finetune evaluate run",
        "pup finetune evaluate run --judge-model glm-5p3-flash",
        "pup finetune evaluate run --max-cases 4 --out-dir run/eval-4",
        "pup finetune evaluate run --compare-model anthropic/claude-opus-5-5",
    ),
    safety=Safety(
        data="sends test rows to the provider and the judge, results to Datadog",
        recipients="the training provider, the judge's provider, the compare model's provider "
        "(with --compare-model), Datadog (experiments)",
        without="previews the calls",
        confirm="evaluates (paid) and publishes the experiments",
        paid=True,
        read_only="confirm_refused",
        offline_allowed=True,
    ),
    panel=EVALUATE_AND_SHIP,
)


@run_command
def evaluate(
    run_dir: RunDirArgument,
    max_cases: Annotated[
        int,
        typer.Option(
            "--max-cases",
            min=1,
            metavar="N",
            help="Turns replayed per conversation.",
            rich_help_panel=CASES,
        ),
    ] = runner.DEFAULT_MAX_CASES_PER_TRAJECTORY,
    max_tokens: Annotated[
        int,
        typer.Option(
            min=1,
            metavar="N",
            help="Most tokens each generation may produce.",
            rich_help_panel=CASES,
        ),
    ] = runner.DEFAULT_MAX_TOKENS,
    seed: Annotated[
        int,
        typer.Option(
            min=0,
            metavar="N",
            help="Seed for picking cases and the bootstrap.",
            rich_help_panel=CASES,
        ),
    ] = runner.DEFAULT_SEED,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge-model",
            metavar="MODEL",
            help="An alias (glm-5p3-flash), anthropic/MODEL, openai/MODEL or a Fireworks id. "
            f"{default_note(BY_KEY)}",
            show_default=False,
            rich_help_panel=MODELS,
        ),
    ] = None,
    compare_model: Annotated[
        str | None,
        typer.Option(
            "--compare-model",
            metavar="MODEL",
            help="Also replay this model, named as --judge-model.",
            show_default=False,
            rich_help_panel=MODELS,
        ),
    ] = None,
    skip_calibration: Annotated[
        bool,
        typer.Option(
            "--skip-calibration",
            help="Do not check the judge on known answers first.",
            rich_help_panel=MODELS,
        ),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            metavar="DIR",
            help=f"Output directory. {default_note('RUN_DIR/eval')}",
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            min=1,
            max=64,
            metavar="N",
            help="Cases at once (1-64).",
            rich_help_panel=RUN,
        ),
    ] = runner.DEFAULT_CONCURRENCY,
    no_publish: Annotated[
        bool,
        typer.Option(
            "--no-publish",
            help="Do not publish the evaluation to Datadog.",
            rich_help_panel=RUN,
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm", help="Run it (paid); without it, only preview.", rich_help_panel=RUN
        ),
    ] = False,
    old_max_cases: Annotated[
        int | None,
        typer.Option(
            "--max-cases-per-trajectory", min=1, hidden=True, help="The old name of --max-cases."
        ),
    ] = None,
) -> dict[str, Any] | CommandResult:
    max_cases = renamed("max_cases", "old_max_cases")
    runtime = get_runtime()
    run_dir, source = resolve_run_ref(run_dir, "evaluate")
    settings = runner.EvalSettings(
        out_dir=out_dir if out_dir is not None else run_dir / runner.DEFAULT_OUT_DIR_NAME,
        max_cases_per_trajectory=max_cases,
        judge_model=judge_model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
        skip_calibration=skip_calibration,
        publish=not no_publish,
        compare_model=compare_model,
    )
    evaluation = runner.plan_standalone(runtime, run_dir, settings)
    # PUP_READ_ONLY refuses paid and remote runs, not one that never leaves the machine.
    offline = evaluation.plan.offline
    if confirm and not offline:
        require_writable(runtime, f"evaluating on {evaluation.provider.name}", what=PAID_RUN)
    rerun = [*evaluation.plan.evaluate_command(), "--confirm"]
    preview = _with_source(_with_defaults_used(evaluation.preview(), current_context()), source)
    require_confirm(
        confirm, preview, shlex.join(rerun), read_only=runtime.read_only and not offline
    )
    outcome = evaluation.execute()
    code = runner.evaluation_exit_code(outcome)
    # What it prints is the decision; summary.json in the evaluation directory has the rest.
    data = _with_source(evaluation_view(outcome), source)
    return CommandResult(data, exit_code=code) if code else data


def register(app: typer.Typer) -> None:
    """Add `evaluate` to ``app``."""
    app.command(**DOC.typer())(evaluate)
