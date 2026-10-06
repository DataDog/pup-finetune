# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune train`: LoRA SFT on canonical training-row files (paid; previews by default)."""

from __future__ import annotations

import re
import secrets
import shlex
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from pydantic import ValidationError

from pup_finetune.cli._common import get_runtime, run_command
from pup_finetune.errors import UsageError, require_confirm, require_writable
from pup_finetune.providers import TrainRequest, get_provider, provider_names
from pup_finetune.training.settings import SFTSettings

if TYPE_CHECKING:
    from pydantic_core import ErrorDetails

RUNS_DIR = Path("finetune") / "runs"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_FIELDS = {
    "--epochs": "max_epochs",
    "--learning-rate": "learning_rate",
    "--batch-size": "batch_size",
    "--lora-rank": "lora_rank",
    "--max-seq-len": "max_seq_len",
    "--seed": "seed",
    "--early-stopping-patience": "early_stopping_patience",
    "--max-spend-usd": "max_spend_usd",
    "--hourly-rate-usd": "hourly_rate_usd",
}
_FLAGS = {field: flag for flag, field in _FIELDS.items()}


def _default(field: str) -> object:
    return SFTSettings.model_fields[field].default


def _new_run_id() -> str:
    """A fresh run id: ``sft-<UTC yyyymmdd-hhmmss>-<8 hex>``."""
    return f"sft-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"


def _describe(error: ErrorDetails) -> str:
    message = error["msg"].removeprefix("Value error, ")
    for field, name in _FLAGS.items():
        message = message.replace(field, name)
    loc = error["loc"]
    flag = _FLAGS.get(str(loc[0])) if loc else None
    return f"{flag}: {message}" if flag else message


def _settings(overrides: Mapping[str, float | None]) -> SFTSettings:
    values = {_FIELDS[flag]: value for flag, value in overrides.items() if value is not None}
    try:
        return SFTSettings.model_validate(values)
    except ValidationError as exc:
        details = "; ".join(_describe(error) for error in exc.errors())
        raise UsageError(
            f"invalid training settings: {details}", hint="see `pup finetune train --help`"
        ) from None


@run_command
def train(
    train_path: Annotated[
        Path, typer.Option("--train", help="Training rows: a JSONL file of canonical rows.")
    ],
    validation_path: Annotated[
        Path, typer.Option("--validation", help="Validation rows: a JSONL file of canonical rows.")
    ],
    provider_name: Annotated[
        str, typer.Option("--provider", help=f"Training provider: {', '.join(provider_names())}.")
    ] = "baseten",
    model: Annotated[
        str | None, typer.Option("--model", help="Model alias (default: the provider's default).")
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir", help=f"New or empty output directory (default: {RUNS_DIR}/<run-id>)."
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run name (default: sft-<UTC timestamp>-<random hex>)."),
    ] = None,
    epochs: Annotated[
        int | None,
        typer.Option("--epochs", help=f"Maximum epochs (default: {_default('max_epochs')})."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(
            "--learning-rate", help=f"Learning rate (default: {_default('learning_rate')})."
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help=f"Datums per optimizer step (default: {_default('batch_size')}).",
        ),
    ] = None,
    lora_rank: Annotated[
        int | None, typer.Option("--lora-rank", help="LoRA rank (default: the model's).")
    ] = None,
    max_seq_len: Annotated[
        int | None,
        typer.Option(
            "--max-seq-len", help="Context limit in tokens (default and maximum: the model's)."
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help=f"Seed (default: {_default('seed')}).")
    ] = None,
    early_stopping_patience: Annotated[
        int | None,
        typer.Option(
            "--early-stopping-patience",
            help="Epochs without validation improvement before stopping "
            f"(default: {_default('early_stopping_patience')}).",
        ),
    ] = None,
    max_spend_usd: Annotated[
        float | None,
        typer.Option(
            "--max-spend-usd",
            help="Stop before spending this much (USD); needs --hourly-rate-usd.",
        ),
    ] = None,
    hourly_rate_usd: Annotated[
        float | None,
        typer.Option(
            "--hourly-rate-usd", help="Your training price per hour (USD), for --max-spend-usd."
        ),
    ] = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Start the paid run; preview the plan otherwise.")
    ] = False,
) -> dict[str, Any]:
    overrides: dict[str, float | None] = {
        "--epochs": epochs,
        "--learning-rate": learning_rate,
        "--batch-size": batch_size,
        "--lora-rank": lora_rank,
        "--max-seq-len": max_seq_len,
        "--seed": seed,
        "--early-stopping-patience": early_stopping_patience,
        "--max-spend-usd": max_spend_usd,
        "--hourly-rate-usd": hourly_rate_usd,
    }
    settings = _settings(overrides)
    if run_id is None:
        run_id = _new_run_id()
    elif not _RUN_ID.fullmatch(run_id):
        raise UsageError(
            f"invalid --run-id {run_id!r}",
            hint="use letters, digits, '.', '_' and '-', starting with a letter or digit",
        )
    run_dir = run_dir or RUNS_DIR / run_id
    require_writable(get_runtime(), "training on Baseten")
    provider = get_provider(provider_name)
    model = model or provider.default_model()
    request = TrainRequest(
        train_path=train_path,
        validation_path=validation_path,
        model=model,
        run_dir=run_dir,
        run_id=run_id,
        settings=settings,
    )
    plan = provider.plan_training(request)
    # The rerun pins the generated run id and directory so it trains exactly what was previewed.
    rerun = ["pup", "finetune", "train", "--train", str(train_path)]
    rerun += ["--validation", str(validation_path), "--provider", provider.name, "--model", model]
    rerun += ["--run-id", run_id, "--run-dir", str(run_dir)]
    for flag, value in overrides.items():
        if value is not None:
            rerun += [flag, str(value)]
    require_confirm(confirm, plan, shlex.join([*rerun, "--confirm"]))
    return provider.train(request)


def register(app: typer.Typer) -> None:
    """Add the `train` command to ``app``."""
    app.command(
        name="train",
        short_help="Fine-tune a model with LoRA SFT (paid; previews without --confirm).",
        help=(
            "Fine-tune a model with LoRA SFT on canonical training and validation rows. Without "
            "--confirm, prints the plan (rows, rendering, settings, estimates) and the exact "
            "command that starts the paid run. Training runs in the foreground and writes its "
            "plan, epochs and result to the run directory."
        ),
    )(train)
