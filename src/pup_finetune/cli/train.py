# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune train`, `resume`, `status` and `cancel`, and the hidden `_worker`.

`train` and `resume` are paid: without ``--confirm`` they preview what would run (free, and
allowed with ``PUP_READ_ONLY`` set) and print the exact command that runs it. They train in
the foreground, with progress on stderr, or with ``--detach`` in a worker that keeps going
after the command returns (see :mod:`pup_finetune.training.worker`).
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from pup_finetune.cli._common import get_runtime, run_command
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.errors import (
    FinetuneError,
    StateError,
    UsageError,
    require_confirm,
    require_writable,
)
from pup_finetune.output import info
from pup_finetune.providers import get_provider
from pup_finetune.providers.base import ModelProfile, Provider, TrainSpec
from pup_finetune.runs import (
    RunDir,
    RunState,
    check_run_id,
    new_run_id,
    require_resumable,
)
from pup_finetune.storage import finite_json
from pup_finetune.training.inputs import RunInputs, build_inputs, load_run_inputs
from pup_finetune.training.worker import (
    WORKER_MODES,
    WorkerMode,
    cancel_training,
    cancellable_pid,
    lock_held,
    record_worker_exit,
    require_detach_supported,
    run_in_dir,
    spawn_worker,
    worker_status,
)

REJECTED_SAMPLE = 20
"""How many rejected rows a preview lists (all of them are counted by code)."""

# The TrainSpec fields that have a flag, and the flag.
_FLAGS: dict[str, str] = {
    "max_epochs": "--max-epochs",
    "patience": "--patience",
    "min_delta": "--min-delta",
    "learning_rate": "--learning-rate",
    "batch_size": "--batch-size",
    "lora_rank": "--lora-rank",
    "lora_alpha": "--lora-alpha",
    "seed": "--seed",
    "checkpoint_every_minutes": "--checkpoint-every-minutes",
    "max_train_tokens": "--max-train-tokens",
    "max_spend_usd": "--max-spend-usd",
    "hourly_rate_usd": "--hourly-rate-usd",
    "spend_reserve_fraction": "--spend-reserve-fraction",
    "microbatch_token_budget": "--microbatch-token-budget",
}

RunDirArgument = Annotated[
    Path, typer.Argument(metavar="RUN_DIR", help="The run directory.", show_default=False)
]
DetachOption = Annotated[
    bool,
    typer.Option(
        "--detach",
        help="Train in a background worker and return at once; follow it with `status --watch`.",
    ),
]
ConfirmOption = Annotated[
    bool, typer.Option("--confirm", help="Run it (paid); without it, only preview what runs.")
]


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _default(field: str) -> object:
    return TrainSpec.model_fields[field].get_default(call_default_factory=True)


# --- Progress ----------------------------------------------------------------------------


def _num(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else "n/a"


def describe_event(event: Mapping[str, Any]) -> str | None:
    """One progress line for a training loop event, or ``None`` for an unknown one."""
    kind = event.get("type")
    if kind == "step":
        return (
            f"epoch {event.get('epoch')} step {event.get('step')}: loss {_num(event.get('loss'))}"
        )
    if kind == "epoch":
        best = " (new best)" if event.get("new_best") else ""
        return (
            f"epoch {event.get('epoch')} done: train loss {_num(event.get('train_loss'))}, "
            f"validation loss {_num(event.get('val_loss'))}{best}"
        )
    if kind == "checkpoint":
        return (
            f"saved {event.get('kind')} checkpoint {event.get('name')} "
            f"(epoch {event.get('epoch')}, step {event.get('step')}, {event.get('reason')})"
        )
    if kind == "stopped":
        detail = f": {event['detail']}" if event.get("detail") else ""
        return f"stopping: {event.get('reason')}{detail}"
    if kind == "retry":
        return (
            f"retrying {event.get('call')} (attempt {event.get('attempt')}) after "
            f"{event.get('error')}, in {event.get('delay_s')}s"
        )
    return None


def _progress(event: dict[str, Any]) -> None:
    line = describe_event(event)
    if line is not None:
        info(line)


# --- train -------------------------------------------------------------------------------


def _provider_options(values: Sequence[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        key = key.strip()
        if not sep or not key:
            raise UsageError(
                f"--provider-option must look like key=value, got {value!r}",
                hint="for example --provider-option step_delay_s=0.5",
            )
        if key in options:
            raise UsageError(f"--provider-option {key} is given more than once")
        options[key] = raw
    return options


def _spec(values: Mapping[str, Any], provider_options: Mapping[str, Any]) -> TrainSpec:
    """The :class:`TrainSpec` from the flags that were given; errors name the flags."""
    data = {field: value for field, value in values.items() if value is not None}
    try:
        return TrainSpec.model_validate({**data, "provider_options": dict(provider_options)})
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            loc = error["loc"]
            name = _FLAGS.get(str(loc[0]), str(loc[0])) if loc else None
            message = error["msg"].removeprefix("Value error, ")
            for field, flag in _FLAGS.items():
                message = message.replace(field, flag)
            problems.append(f"{name}: {message}" if name else message)
        raise UsageError(
            f"invalid training settings: {'; '.join(problems)}",
            hint="see `pup finetune train --help`",
        ) from None


def _preview(
    provider: Provider,
    run_dir: Path,
    run_id: str,
    spec: TrainSpec,
    profile: ModelProfile,
    inputs: RunInputs,
) -> dict[str, Any]:
    warnings = []
    if not inputs.validation_rows:
        warnings.append(
            "the validation split has no rows to validate on, so the best epoch is chosen "
            "by training loss"
        )
    return {
        "provider": provider.name,
        "model": profile.alias,
        "profile": dataclasses.asdict(profile),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "rows": dict(inputs.rows),
        "examples": inputs.examples(),
        "rejected": len(inputs.rejected),
        "rejected_by_code": inputs.rejected_by_code(),
        "rejected_rows": [row.to_json() for row in inputs.rejected[:REJECTED_SAMPLE]],
        "tokens_per_epoch": inputs.tokens_per_epoch,
        "validation_tokens": inputs.validation.train_tokens,
        "max_tokens_total": inputs.tokens_per_epoch * spec.max_epochs,
        "spec": spec.model_dump(mode="json"),
        "data_destination": f"training data is sent to {provider.name}",
        "warnings": warnings,
    }


def _finished(run_dir: RunDir, result: Mapping[str, Any]) -> dict[str, Any]:
    """The result a foreground run emits: ``result.json`` plus where it is and what next."""
    return finite_json(
        {
            **result,
            "run_dir": str(run_dir.path),
            "next_command": _next_command(run_dir.path, str(result.get("phase")), running=False),
        }
    )


def _in_foreground(
    provider: Provider, run: RunDir, mode: WorkerMode, inputs: RunInputs
) -> dict[str, Any]:
    """Train ``run`` in this process (Ctrl-C cancels gracefully) and return what to emit."""
    try:
        result = run_in_dir(provider, run, mode, inputs, on_event=_progress)
    except KeyboardInterrupt:
        # A second Ctrl-C stops at once; the run recorded where it stopped.
        try:
            phase = run.load_state().phase
        except FinetuneError:
            phase = "interrupted"
        raise FinetuneError(
            f"training stopped at once (Ctrl-C): the run in {run.path} is {phase}",
            hint=f"resume it with `pup finetune resume {_q(run.path)}`",
        ) from None
    return _finished(run, result)


def _detached(run_dir: RunDir, run_id: str, pid: int) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.path),
        "run_id": run_id,
        "pid": pid,
        "status": "started",
        "worker_log": str(run_dir.worker_log_path),
        "next_command": f"pup finetune status {_q(run_dir.path)} --watch",
    }


@run_command
def train(
    provider_name: Annotated[
        str, typer.Option("--provider", help="The training provider.", show_default=False)
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset", help="The dataset directory to train on.", show_default=False),
    ],
    model: Annotated[
        str, typer.Option("--model", help="The model profile to fine-tune.", show_default=False)
    ],
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="A new or empty run directory.", show_default=False),
    ],
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="The run id (default: ft-<UTC timestamp>-<random hex>); also the promoted "
            "model's id.",
            show_default=False,
        ),
    ] = None,
    max_epochs: Annotated[
        int | None,
        typer.Option(help=f"Most epochs to train (default: {_default('max_epochs')})."),
    ] = None,
    patience: Annotated[
        int | None,
        typer.Option(
            help="Epochs without a validation-loss improvement before stopping early "
            f"(default: {_default('patience')})."
        ),
    ] = None,
    min_delta: Annotated[
        float | None,
        typer.Option(help=f"Smallest improvement that counts (default: {_default('min_delta')})."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(help=f"Learning rate (default: {_default('learning_rate')})."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help=f"Rows per optimizer step (default: {_default('batch_size')})."),
    ] = None,
    lora_rank: Annotated[int | None, typer.Option(help="LoRA rank (default: the model's).")] = None,
    lora_alpha: Annotated[
        int | None, typer.Option(help="LoRA alpha (default: the provider's).")
    ] = None,
    seed: Annotated[int | None, typer.Option(help=f"Seed (default: {_default('seed')}).")] = None,
    checkpoint_every_minutes: Annotated[
        float | None,
        typer.Option(
            help="Also save a resumable checkpoint this often, between epochs "
            f"(default: {_default('checkpoint_every_minutes')})."
        ),
    ] = None,
    max_train_tokens: Annotated[
        int | None,
        typer.Option(help="Stop before training on more rendered tokens than this."),
    ] = None,
    max_spend_usd: Annotated[
        float | None,
        typer.Option(help="Stop before the elapsed time costs more; needs --hourly-rate-usd."),
    ] = None,
    hourly_rate_usd: Annotated[
        float | None,
        typer.Option(help="Your training price per hour (USD), for --max-spend-usd."),
    ] = None,
    spend_reserve_fraction: Annotated[
        float | None,
        typer.Option(
            help="The share of --max-spend-usd kept for the checkpoint and promotion at the "
            "stop; training stops that much short of the cap "
            f"(default: {_default('spend_reserve_fraction')})."
        ),
    ] = None,
    microbatch_token_budget: Annotated[
        int | None,
        typer.Option(help="Pack each step into microbatches of at most this many tokens."),
    ] = None,
    provider_option: Annotated[
        list[str] | None,
        typer.Option(
            "--provider-option",
            metavar="KEY=VALUE",
            help="A provider-specific option; repeat for more.",
            show_default=False,
        ),
    ] = None,
    detach: DetachOption = False,
    confirm: ConfirmOption = False,
) -> dict[str, Any]:
    runtime = get_runtime()
    if detach:
        require_detach_supported()
    run_id = check_run_id(run_id) if run_id is not None else new_run_id()
    RunDir(run_dir).require_empty()
    provider = get_provider(provider_name, runtime)
    profile = provider.resolve_model(model)
    raw_options = _provider_options(provider_option or [])
    options = provider.validate_options(raw_options)
    # What the run trains with: an option may narrow the profile (e.g. cap its context),
    # so rows are rendered, previewed and recorded against the narrowed one.
    profile = provider.training_profile(profile)
    flags: dict[str, Any] = {
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "checkpoint_every_minutes": checkpoint_every_minutes,
        "max_train_tokens": max_train_tokens,
        "max_spend_usd": max_spend_usd,
        "hourly_rate_usd": hourly_rate_usd,
        "spend_reserve_fraction": spend_reserve_fraction,
        "microbatch_token_budget": microbatch_token_budget,
    }
    spec = _spec(flags, options)
    provider.preflight(profile)
    dataset = DatasetDir(dataset_dir)
    dataset.settings()  # a clear error when it is not a dataset directory
    renderer = provider.renderer(profile)
    inputs = build_inputs(dataset, renderer, profile)
    preview = _preview(provider, run_dir, run_id, spec, profile, inputs)
    # The rerun pins the run id and directory, so it trains exactly what was previewed.
    rerun = ["pup", "finetune", "train", "--provider", provider.name]
    rerun += ["--dataset", str(dataset_dir), "--model", model]
    rerun += ["--run-dir", str(run_dir), "--run-id", run_id]
    for field, value in flags.items():
        if value is not None:
            rerun += [_FLAGS[field], str(value)]
    for key, value in raw_options.items():
        rerun += ["--provider-option", f"{key}={value}"]
    if detach:
        rerun.append("--detach")
    if confirm:
        require_writable(runtime, f"training on {provider.name}")
    require_confirm(confirm, preview, shlex.join([*rerun, "--confirm"]))

    run = RunDir.create(
        run_dir,
        provider=provider.name,
        profile=profile,
        spec=spec,
        dataset_dir=dataset.path,
        dataset_sha256=inputs.dataset_sha256,
        rendering=renderer.identity(),
        run_id=run_id,
    )
    if detach:
        return _detached(run, run_id, spawn_worker(run, "train"))
    info(f"training run {run_id} in {run_dir} on {provider.name} (Ctrl-C cancels)")
    return _in_foreground(provider, run, "train", inputs)


# --- resume ------------------------------------------------------------------------------


def _resume_mode(run: RunDir, state: RunState) -> WorkerMode:
    """What resuming ``run`` does; raises when it cannot be resumed now."""
    worker = worker_status(run, phase=state.phase, pid=state.pid, heartbeat_at=state.heartbeat_at)
    if worker.starting:
        raise StateError(
            f"a worker for the run in {run.path} is starting (process {worker.pid})",
            hint=f"follow it with `pup finetune status {_q(run.path)} --watch`",
        )
    if state.phase == "created":
        return "train"  # it never started (e.g. its worker failed to start): start it
    if state.phase == "training":
        if lock_held(run):
            raise StateError(
                f"the run in {run.path} is still training (process {state.pid})",
                hint=f"wait for it, or cancel it with `pup finetune cancel {_q(run.path)}`",
            )
    else:
        require_resumable(state)
    return "finalize" if state.resume_mode == "finalize_only" else "resume"


def _resume_action(mode: WorkerMode, state: RunState) -> tuple[str, dict[str, Any] | None]:
    if mode == "train":
        return "train from the start (the run never started training)", None
    if mode == "finalize":
        return "finalize only: promote the best checkpoint (training already ended)", None
    point = state.last_resume_point()
    if point is None:
        return "train from the start (no training checkpoint was saved)", None
    where = {
        "epoch": point.epoch,
        "step": point.step,
        "steps_in_epoch": point.steps_in_epoch,
        "checkpoint": point.checkpoint.name,
    }
    if state.epochs >= point.epoch:  # saved at the end of that epoch
        return f"continue training with epoch {point.epoch + 1} (after step {point.step})", where
    return f"continue training in epoch {point.epoch} after step {point.step}", where


@run_command
def resume(
    run_dir: RunDirArgument, detach: DetachOption = False, confirm: ConfirmOption = False
) -> dict[str, Any]:
    runtime = get_runtime()
    if detach:
        require_detach_supported()
    run = RunDir(run_dir)
    manifest = run.manifest()
    state = run.load_state()
    mode = _resume_mode(run, state)
    provider, inputs = load_run_inputs(run, runtime, finalize_only=mode == "finalize")
    action, resume_from = _resume_action(mode, state)
    max_epochs = int(manifest["spec"]["max_epochs"])
    preview = {
        "run_id": manifest["run_id"],
        "run_dir": str(run_dir),
        "provider": provider.name,
        "model": manifest["profile"]["alias"],
        "phase": state.phase,
        "mode": mode,
        "action": action,
        "resume_from": resume_from,
        "epochs_completed": state.epochs,
        "remaining_epochs": 0 if mode == "finalize" else max(max_epochs - state.epochs, 0),
        "tokens_per_epoch": inputs.tokens_per_epoch,
        "data_destination": f"training data is sent to {provider.name}",
    }
    rerun = ["pup", "finetune", "resume", str(run_dir), *(["--detach"] if detach else [])]
    if confirm:
        require_writable(runtime, f"training on {provider.name}")
    require_confirm(confirm, preview, shlex.join([*rerun, "--confirm"]))
    if detach:
        return _detached(run, str(manifest["run_id"]), spawn_worker(run, mode))
    info(f"resuming run {manifest['run_id']} in {run_dir}: {action} (Ctrl-C cancels)")
    return _in_foreground(provider, run, mode, inputs)


# --- status ------------------------------------------------------------------------------


def _next_command(path: Path, phase: str, *, running: bool, stale: bool = False) -> str:
    if running:
        return f"pup finetune status {_q(path)} --watch"
    if phase == "completed":
        return f"pup finetune evaluate {_q(path)}"
    if phase == "training" and stale:
        return f"pup finetune cancel {_q(path)}"
    return f"pup finetune resume {_q(path)}"


def run_status(run: RunDir) -> dict[str, Any]:
    """What `status` reports for ``run``; every number finite or ``None``."""
    summary = run.status()
    worker = worker_status(
        run,
        phase=summary["phase"],
        pid=summary["pid"],
        heartbeat_at=summary["heartbeat_at"],
    )
    best = summary["best"] or {}
    stale_alive = worker.stale and worker.alive
    warnings = list(summary["warnings"])
    if worker.problem is not None:
        warnings.append(worker.problem)
    return finite_json(
        {
            "run_id": summary["run_id"],
            "run_dir": str(run.path),
            "provider": summary["provider"],
            "model": summary["model"],
            "phase": summary["phase"],
            "epoch": summary["epoch"],
            "step": summary["step"],
            "epochs_completed": summary["epochs_completed"],
            "max_epochs": summary["max_epochs"],
            "last_train_loss": summary["last_train_loss"],
            "val_losses": summary["val_losses"],
            "best_epoch": best.get("epoch"),
            "best_val_loss": best.get("val_loss"),
            "stopped_reason": summary["stopped_reason"],
            "promoted_model": summary["promoted_model"],
            "deployable_checkpoint": summary["deployable_checkpoint"],
            "resume_mode": summary["resume_mode"],
            "worker": worker.to_json(),
            "worker_log": worker.log,
            "error": summary["error"] or worker.error,
            "finalize_error": summary["finalize_error"],
            "warnings": warnings,
            "next_command": _next_command(
                run.path, summary["phase"], running=worker.running, stale=stale_alive
            ),
        }
    )


def _status_line(data: Mapping[str, Any]) -> str:
    worker = data["worker"]
    process = "running" if worker["running"] else "not running"
    if worker["pid"] is not None:
        process = f"pid {worker['pid']} {process}"
    return (
        f"{data['phase']}: epoch {data['epoch']}/{data['max_epochs']}, step {data['step']}, "
        f"loss {_num(data['last_train_loss'])}, best epoch {data['best_epoch'] or 'n/a'} "
        f"({process})"
    )


@run_command
def status(
    run_dir: RunDirArgument,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Print progress to stderr until training ends, then print the final status.",
        ),
    ] = False,
    interval: Annotated[
        float, typer.Option(min=0.1, help="Seconds between checks with --watch.")
    ] = 10.0,
) -> dict[str, Any]:
    run = RunDir(run_dir)
    data = run_status(run)
    if not watch:
        return data
    shown = None
    while True:
        line = _status_line(data)
        if line != shown:
            info(line)
            shown = line
        if not data["worker"]["running"]:
            return data
        time.sleep(interval)
        data = run_status(run)


# --- cancel ------------------------------------------------------------------------------


@run_command
def cancel(run_dir: RunDirArgument, confirm: ConfirmOption = False) -> dict[str, Any]:
    run = RunDir(run_dir)
    run_id = run.run_id
    pid = cancellable_pid(run)
    preview = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pid": pid,
        "action": f"send SIGTERM to process {pid}: training stops after the current step, "
        "saves a resumable checkpoint and promotes the best epoch so far",
    }
    require_confirm(
        confirm, preview, shlex.join(["pup", "finetune", "cancel", str(run_dir), "--confirm"])
    )
    pid = cancel_training(run)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pid": pid,
        "status": "cancelling",
        "next_command": f"pup finetune status {_q(run_dir)} --watch",
    }


# --- the worker --------------------------------------------------------------------------


@run_command
def worker(
    run_dir: RunDirArgument,
    mode: Annotated[str, typer.Option("--mode", help="train, resume or finalize.")],
) -> None:
    if mode not in WORKER_MODES:
        raise UsageError(
            f"unknown worker mode {mode!r}", hint=f"use one of {', '.join(WORKER_MODES)}"
        )
    run = RunDir(run_dir)
    try:
        runtime = get_runtime()
        # It runs what `train --confirm` or `resume --confirm` already allowed, but is a
        # command of its own: PUP_READ_ONLY must stop it too.
        require_writable(runtime, f"training on {run.manifest()['provider']}")
        info(f"worker {os.getpid()} started: {mode} {run_dir}")
        provider, inputs = load_run_inputs(run, runtime, finalize_only=mode == "finalize")
        result = run_in_dir(provider, run, mode, inputs, on_event=_progress)
    except BaseException as exc:
        record_worker_exit(run, mode, exc)
        raise
    record_worker_exit(run, mode, None)
    info(f"worker {os.getpid()} finished: the run is {result.get('phase')}")


def register(app: typer.Typer) -> None:
    """Add `train`, `resume`, `status`, `cancel` and the hidden `_worker` to ``app``."""
    app.command(
        "train",
        short_help="Fine-tune a model with LoRA SFT (paid; previews without --confirm).",
        help=(
            "Fine-tune a model with LoRA SFT on a dataset's train split, choosing the best "
            "epoch by validation loss and stopping early. Without --confirm, previews the "
            "run (rows and rendered tokens per split, rejected rows, tokens per epoch and "
            "in total, the settings, and where the data goes) and prints the exact command "
            "that starts it. With --confirm, trains in the foreground (Ctrl-C cancels "
            "gracefully: the best epoch is still promoted) or, with --detach, in a "
            "background worker. The preview is allowed with PUP_READ_ONLY set; training "
            "is not."
        ),
    )(train)
    app.command(
        "resume",
        short_help="Resume an interrupted, failed or cancelled run (paid).",
        help=(
            "Continue a run from its last training checkpoint, or only retry its promotion "
            "when training already ended. Without --confirm, previews where it resumes."
        ),
    )(resume)
    app.command(
        "status",
        short_help="Show a run's phase, progress, losses and worker.",
        help=(
            "Show a run's phase, epoch and step, losses, best epoch, promoted model and "
            "whether its training process is alive. With --watch, prints progress to "
            "stderr until training ends, then the final status."
        ),
    )(status)
    app.command(
        "cancel",
        short_help="Stop a training run gracefully.",
        help=(
            "Stop the process training a run: it finishes the current step, saves a "
            "resumable checkpoint and promotes the best epoch so far. Without --confirm, "
            "previews which process gets the signal."
        ),
    )(cancel)
    app.command("_worker", hidden=True)(worker)
