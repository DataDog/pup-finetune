# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune benchmark`: compare base models on a dataset's test split before training.

Benchmarking is paid (it generates with every model and calls a judge), so without
``--confirm`` it previews the calls it would make (free, and allowed with ``PUP_READ_ONLY``
set) and prints the exact command that runs it. The planning, replay, ranking and
recommendation live in :mod:`pup_finetune.evaluation.benchmark`; this module only parses the
flags and emits the result.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, current_context, get_runtime, run_command
from pup_finetune.cli._dirs import require_dataset_dir
from pup_finetune.cli._flags import given
from pup_finetune.cli._presentation import TRAIN, CommandDoc, Safety, default_note, describe
from pup_finetune.cli.train import provider_for_model
from pup_finetune.errors import PAID_RUN, UsageError, require_confirm, require_writable
from pup_finetune.evaluation import benchmark, runner
from pup_finetune.judge_models import BY_KEY

SETTING_PARAMS: dict[str, tuple[str, ...]] = {
    "judge_model": ("judge_model",),
    "max_cases": ("max_cases",),
    "max_rows": ("max_rows",),
    "max_tokens": ("max_tokens",),
    "concurrency": ("concurrency",),
    "seed": ("seed",),
    "calibration": ("skip_calibration",),
    "target_pass_rate": ("target_pass_rate",),
    "reuse": ("no_reuse",),
    "evaluators": ("evaluators",),
}
"""Each setting of a benchmark preview, and the parameters that set it."""

MODELS = "Models"
CASES = "Cases"
JUDGE = "Judge"
RUN = "Run"
"""``benchmark``'s option panels."""

DOC = CommandDoc(
    path="benchmark",
    short="Compare base models before training (paid; previews without --confirm).",
    description=describe(
        "Replay each held-out test conversation of a dataset with several base models, one "
        "recorded assistant turn at a time, and score each with the tool-call metrics and the "
        "LLM judge evaluate uses. Without --confirm, previews the calls and where the data "
        "goes. The result ranks the models and recommends the one to fine-tune.",
        "The dataset must be in Datadog, as for `train`. "
        "Each model runs in a session of its own, one after another, into OUT/<model>; rerun "
        "the same command to continue a benchmark that stopped. A model below "
        "--target-pass-rate is worth fine-tuning; one at or above it may need no training. "
        "Models within noise of the best are named, so the cheapest to serve can be chosen. "
        "--max-rows replays a seeded sample of the test conversations instead, for a quicker, "
        "cheaper comparison. Tools are not executed, so the scores measure agreement with the "
        "recorded behavior, not task success. Base answers from another evaluation or "
        "benchmark of the same dataset, made under the same settings, are reused from "
        "DATASET/replay_cache, together with their judgments when the judge is the same, and "
        "a model with every answer reused opens no session; --no-reuse makes them again. "
        "--evaluators FILE adds your own LLM judges and Python checks to each model's "
        "directory, as for evaluate, and the spec's headline (judge, an evaluator or all) ranks "
        "the models and meets --target-pass-rate.",
    ),
    examples=(
        "pup finetune benchmark ds --models qwen3p5-9b,glm-5p3-flash",
        "pup finetune benchmark ds --models qwen3p5-9b,glm-5p3-flash --max-cases 1 --max-rows 20",
    ),
    safety=Safety(
        data="sends test rows to each model's provider and to the judge",
        recipients="each model's provider, the judge's provider",
        without="previews the calls",
        confirm="benchmarks every model (paid)",
        paid=True,
        read_only="confirm_refused",
        offline_allowed=True,
    ),
    panel=TRAIN,
)


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


@run_command
def benchmark_command(
    dataset_dir: Annotated[
        Path,
        typer.Argument(
            metavar="DATASET",
            help="The dataset directory whose test split is replayed.",
            show_default=False,
        ),
    ],
    models: Annotated[
        str,
        typer.Option(
            "--models",
            metavar="M1,M2",
            help="Base models to compare, comma-separated (see models list).",
            show_default=False,
            rich_help_panel=MODELS,
        ),
    ],
    target_pass_rate: Annotated[
        float,
        typer.Option(
            "--target-pass-rate",
            min=0.0,
            max=1.0,
            metavar="RATE",
            help="Headline pass rate at which a base model needs no fine-tuning.",
            rich_help_panel=MODELS,
        ),
    ] = benchmark.DEFAULT_TARGET_PASS_RATE,
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
    max_rows: Annotated[
        int | None,
        typer.Option(
            "--max-rows",
            min=1,
            metavar="N",
            help=f"Test conversations to sample and replay. {default_note('all')}",
            show_default=False,
            rich_help_panel=CASES,
        ),
    ] = None,
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
            rich_help_panel=JUDGE,
        ),
    ] = None,
    evaluators: Annotated[
        Path | None,
        typer.Option(
            "--evaluators",
            metavar="FILE",
            help="More judges and Python checks, from a TOML file.",
            show_default=False,
            rich_help_panel=JUDGE,
        ),
    ] = None,
    skip_calibration: Annotated[
        bool,
        typer.Option(
            "--skip-calibration",
            help="Do not check the judge on known answers first.",
            rich_help_panel=JUDGE,
        ),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            metavar="DIR",
            help=f"Output directory. {default_note('DATASET/benchmark')}",
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
    no_reuse: Annotated[
        bool,
        typer.Option(
            "--no-reuse",
            help="Do not reuse results of earlier runs.",
            rich_help_panel=RUN,
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm", help="Run it (paid); without it, only preview.", rich_help_panel=RUN
        ),
    ] = False,
) -> dict[str, Any] | CommandResult:
    runtime = get_runtime()
    require_dataset_dir(dataset_dir)
    aliases = [alias.strip() for alias in models.split(",") if alias.strip()]
    if not aliases:
        raise UsageError("--models names no model", hint="pass --models MODEL,MODEL")
    resolved = [(alias, provider_for_model(alias, None, runtime)[0]) for alias in aliases]
    settings = benchmark.BenchmarkSettings(
        out_dir=out_dir if out_dir is not None else dataset_dir / benchmark.DEFAULT_OUT_DIR_NAME,
        max_cases_per_trajectory=max_cases,
        max_rows=max_rows,
        judge_model=judge_model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed=seed,
        skip_calibration=skip_calibration,
        target_pass_rate=target_pass_rate,
        reuse=not no_reuse,
        evaluators=evaluators,
    )
    plan = benchmark.plan_benchmark(runtime, dataset_dir, resolved, settings)
    # PUP_READ_ONLY refuses paid and remote runs, not one that never leaves the machine.
    if confirm and not plan.offline:
        require_writable(runtime, "benchmarking models", what=PAID_RUN)
    preview = _with_defaults_used(plan.preview(), current_context())
    require_confirm(
        confirm,
        preview,
        shlex.join([*plan.command(), "--confirm"]),
        read_only=runtime.read_only and not plan.offline,
    )
    result = benchmark.execute(plan)
    code = benchmark.exit_code(result)
    return CommandResult(result, exit_code=code) if code else result


def register(app: typer.Typer) -> None:
    """Add `benchmark` to ``app``."""
    app.command(**DOC.typer())(benchmark_command)
