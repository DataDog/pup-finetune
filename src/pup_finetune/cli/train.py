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
after the command returns (see :mod:`pup_finetune.training.worker`), or with ``--remote`` in a
provider job (see :mod:`pup_finetune.training.remote`), which ``status``, ``resume`` and
``cancel`` read back first.

``train --evaluate`` also evaluates the run (see :mod:`pup_finetune.evaluation.runner`) in its
training session, after the best checkpoint is promoted and before the session closes, so
no second session is opened. The judge is calibrated, and ``--eval-out-dir`` checked, before
training starts: a failed calibration trains nothing. The run's end is recorded before the
evaluation starts, so a crash during it keeps the trained and promoted run. A failed
evaluation exits 1, with the trained run and the command that evaluates it again. It needs
the foreground: ``--detach`` with ``--evaluate`` is refused (evaluate a detached run with
``pup finetune evaluate`` once it completed). Like ``evaluate``, a complete evaluation is
published as experiments when the dataset was pushed, unless ``--eval-no-publish``. The
session's time still counts against ``--max-spend-usd``, so the evaluation is skipped when
the cap is reached already and stops, partial, once it is.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shlex
import signal
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from pydantic import ValidationError

from pup_finetune.cli._common import (
    CommandResult,
    current_context,
    get_runtime,
    prog_name,
    run_command,
)
from pup_finetune.cli._dirs import (
    RUN_DIR_BY_ID,
    RUN_DIR_NEWEST,
    require_dataset_dir,
    require_new_run_dir,
    resolve_run_ref,
)
from pup_finetune.cli._flags import given, renamed, require_enablers
from pup_finetune.cli._presentation import TRAIN as TRAIN_PANEL
from pup_finetune.cli._presentation import CommandDoc, Safety, default_note, describe
from pup_finetune.cli._summary import format_duration, status_line
from pup_finetune.cli._views import (
    changed_settings,
    files,
    recorded_evaluation_view,
    status_view,
    train_view,
)
from pup_finetune.curation.pipeline import require_in_datadog
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.display import counted, display_path, joined, quoted
from pup_finetune.errors import (
    PAID_RUN,
    FinetuneError,
    Interrupted,
    NotARunDir,
    UsageError,
    WrongPhase,
    require_confirm,
    require_writable,
)
from pup_finetune.evaluation import runner as evaluation
from pup_finetune.next_steps import PROGRAM, command, step, template
from pup_finetune.output import info
from pup_finetune.progress import TrainingProgress
from pup_finetune.providers import (
    get_provider,
    known_models,
    provider_info,
    provider_infos,
    providers_for_model,
    registered_option_keys,
    registered_providers,
    remote_jobs,
    require_packages,
    require_remote,
    require_supported_flags,
)
from pup_finetune.providers.base import (
    MAX_SEQ_LEN_FLAG,
    MAX_SEQ_LEN_OPTION_FLAG,
    OFFLINE_BILLING,
    ModelProfile,
    Provider,
    RemotePlan,
    TrainingUsage,
    TrainSpec,
)
from pup_finetune.rendering.base import Renderer
from pup_finetune.runs import (
    RUN_FILE,
    STATE_FILE,
    RunDir,
    RunState,
    check_run_id,
    new_run_id,
    require_resumable,
    resume_command,
    resume_hint,
)
from pup_finetune.runtime import Runtime
from pup_finetune.storage import finite_json
from pup_finetune.training.inputs import RunInputs, build_inputs, load_run_inputs
from pup_finetune.training.remote import (
    RemoteSync,
    SourceTree,
    ended_sync,
    job_name,
    package_source,
    remote_ended,
    require_locked_renderer,
    submit_remote,
    sync_remote,
)
from pup_finetune.training.report import StateReporter
from pup_finetune.training.runner import AfterFinalize
from pup_finetune.training.worker import (
    WORKER_MODES,
    WorkerMode,
    WorkerStatus,
    cancel_training,
    cancellable_pid,
    lock_held,
    record_worker_exit,
    require_detach_supported,
    run_in_dir,
    spawn_worker,
    worker_status,
)
from pup_finetune.workspace import (
    RUNS_DIR,
    ParentSetup,
    default_run_dir,
    ignore_new_parent,
    latest_run_dir,
)

REJECTED_SAMPLE = 20
"""How many rejected rows a preview lists (all of them are counted by code)."""

# The TrainSpec fields that have a flag, and the flag.
_FLAGS: dict[str, str] = {
    "max_epochs": "--max-epochs",
    "patience": "--patience",
    "min_delta": "--min-delta",
    "learning_rate": "--learning-rate",
    "lr_schedule": "--lr-schedule",
    "warmup_fraction": "--warmup-fraction",
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

TRAIN_NEEDS: dict[str, str] = {
    "spend_reserve_fraction": "max_spend_usd",
    "eval_out_dir": "evaluate",
    "eval_max_cases": "evaluate",
    "old_eval_max_cases": "evaluate",
    "eval_judge_model": "evaluate",
    "eval_concurrency": "evaluate",
    "eval_max_tokens": "evaluate",
    "eval_seed": "evaluate",
    "eval_skip_calibration": "evaluate",
    "eval_no_publish": "evaluate",
    "team": "remote",
}
"""``train``'s options that do nothing without another one: each needs the one it maps to
(see :mod:`pup_finetune.cli._flags`)."""

STATUS_NEEDS: dict[str, str] = {"interval": "watch", "timeout": "watch"}
"""``status``'s options that need ``--watch``."""

REQUIRED = "Required"
RUN = "Run"
TRAINING = "Training"
BUDGET = "Budget"
EVALUATE = "Evaluate after training (with --evaluate)"
ADVANCED = "Advanced"
"""``train``'s option panels. Every option of :data:`EVALUATE` needs ``--evaluate``."""

_THE_MODELS = default_note("the model's")

MAX_SEQ_LEN_OPTION = "max_seq_len"
"""The provider option ``--max-seq-len`` sets."""


def provider_choices() -> str:
    """The providers ``--provider`` takes, for its help: the registry's, development-only ones
    left out, so the help is the same in every environment."""
    return " or ".join(registered_providers())


def model_example() -> str:
    """An example of what ``--model`` takes, for its help: the first model alias of the first
    provider that every provider offers, e.g. ``glm-5p3-flash`` (``models list`` names them
    all); development-only providers left out. Imports no provider module."""
    infos = provider_infos(Runtime())
    aliases = infos[0].models if infos else ()
    return next((alias for alias in aliases if all(alias in i.models for i in infos)), "")


def provider_option_keys() -> str:
    """The ``--provider-option`` keys of each provider, for its help, e.g. ``Keys:
    max_seq_len (baseten, fireworks).``; development-only providers are left out, so the help
    is the same in every environment."""
    providers: dict[tuple[str, ...], list[str]] = {}
    for name, keys in sorted(registered_option_keys().items()):
        providers.setdefault(keys, []).append(name)
    groups = [
        f"{', '.join(keys) or 'none'} ({', '.join(names)})" for keys, names in providers.items()
    ]
    return f"Keys: {'; '.join(groups)}." if groups else ""


RunDirArgument = Annotated[
    Path,
    typer.Argument(metavar="RUN_DIR", help="The run: its directory or run id.", show_default=False),
]
DetachOption = Annotated[
    bool, typer.Option("--detach", help="Train in a background worker; see status --watch.")
]
ConfirmOption = Annotated[
    bool, typer.Option("--confirm", help="Run it (paid); without it, only preview.")
]
CancelConfirmOption = Annotated[
    bool,
    typer.Option(
        "--confirm", help="Send the stop signal; without it, preview which process gets it."
    ),
]


def _q(value: object) -> str:
    return quoted(value)


def _default(field: str) -> object:
    return TrainSpec.model_fields[field].get_default(call_default_factory=True)


def _int_default(field: str) -> int:
    """The :class:`TrainSpec` default of the int field ``field``."""
    value = _default(field)
    if not isinstance(value, int):
        raise TypeError(f"TrainSpec.{field} has no int default")
    return value


def _float_default(field: str) -> float:
    """The :class:`TrainSpec` default of the float field ``field``."""
    value = _default(field)
    if not isinstance(value, float):
        raise TypeError(f"TrainSpec.{field} has no float default")
    return value


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


def _with_max_seq_len(provider: str, options: Mapping[str, str], value: int) -> dict[str, str]:
    """``options`` with ``max_seq_len`` set to ``value`` (``--max-seq-len``)."""
    if MAX_SEQ_LEN_OPTION in options:
        raise UsageError(
            "--max-seq-len and --provider-option max_seq_len set the same thing",
            hint="pass --max-seq-len only",
        )
    if MAX_SEQ_LEN_OPTION not in registered_option_keys(include_dev=True).get(provider, ()):
        raise UsageError(f"--max-seq-len is not supported by the {provider} provider")
    return {**options, MAX_SEQ_LEN_OPTION: str(value)}


def _spec(values: Mapping[str, Any], provider_options: Mapping[str, Any]) -> TrainSpec:
    """The :class:`TrainSpec` from the flags that were given; errors name the flags."""
    data = {field: value for field, value in values.items() if value is not None}
    # Recorded as they validate again: every later validation (a worker's, a resume's)
    # decodes string values as JSON, so a string option is recorded JSON-encoded.
    recorded = {
        key: json.dumps(value) if isinstance(value, str) else value
        for key, value in provider_options.items()
    }
    try:
        return TrainSpec.model_validate({**data, "provider_options": recorded})
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
            hint=f"see `{prog_name()} help train`",
        ) from None


_SETTING_PARAMS: dict[str, str] = {
    "max_epochs": "max_epochs",
    "patience": "patience",
    "min_delta": "min_delta",
    "learning_rate": "learning_rate",
    "lr_schedule": "lr_schedule",
    "warmup_fraction": "warmup_fraction",
    "batch_size": "batch_size",
    "lora.rank": "lora_rank",
    "lora.alpha": "lora_alpha",
    "max_seq_len": "max_seq_len",
    "seed": "seed",
    "checkpoint_every_minutes": "checkpoint_every_minutes",
    "max_train_tokens": "max_train_tokens",
    "microbatch_token_budget": "microbatch_token_budget",
    "max_spend_usd": "max_spend_usd",
    "hourly_rate_usd": "hourly_rate_usd",
    "spend_reserve_fraction": "spend_reserve_fraction",
}
"""Each setting a preview's ``defaults_used`` can name, and the parameter that sets it."""


def _defaults_used(ctx: Any, options: Mapping[str, Any]) -> list[str]:
    """The settings of :data:`_SETTING_PARAMS` the command line left at their defaults (the
    context counts as given through ``--provider-option max_seq_len`` too)."""
    return [
        name
        for name, param in _SETTING_PARAMS.items()
        if not (ctx is not None and given(ctx, param))
        and not (name == "max_seq_len" and MAX_SEQ_LEN_OPTION in options)
    ]


def _rows(inputs: RunInputs) -> dict[str, int]:
    """The rows a run trains and validates on, the test rows, and the rows left out."""
    return {
        "train": len(inputs.train_rows),
        "validation": len(inputs.validation_rows),
        "test": inputs.rows["test"],
        "rejected": len(inputs.rejected),
    }


def _usage(
    spec: TrainSpec, inputs: RunInputs, *, epochs: int, resumed: bool = False
) -> TrainingUsage:
    return TrainingUsage(
        tokens_per_epoch=inputs.tokens_per_epoch,
        epochs=epochs,
        validation_tokens_per_epoch=inputs.validation.train_tokens,
        max_train_tokens=spec.max_train_tokens,
        max_spend_usd=spec.max_spend_usd,
        hourly_rate_usd=spec.hourly_rate_usd,
        resumed=resumed,
    )


def _action(profile: ModelProfile, spec: TrainSpec, rows: Mapping[str, int]) -> str:
    """What ``train --confirm`` does, in one sentence."""
    lora = spec.lora_config(profile)
    text = (
        f"fine-tune {profile.alias} with LoRA rank {lora.rank} for up to "
        f"{counted(spec.max_epochs, 'epoch')} on {counted(rows['train'], 'row')}"
    )
    if rows["validation"]:
        return f"{text}, validating on {rows['validation']}"
    return f"{text}, with no rows to validate on"


def _profile_brief(profile: ModelProfile) -> dict[str, Any]:
    """The profile as a preview shows it; ``run.json`` records all of it."""
    return {
        "alias": profile.alias,
        "base_model": profile.base_model,
        "max_seq_len": profile.max_seq_len,
    }


def no_epoch_can_finish(spec: TrainSpec, inputs: RunInputs) -> bool:
    """Whether ``--max-train-tokens`` is below one epoch's tokens: training then stops
    before the first epoch ends, so nothing is kept or evaluated."""
    cap = spec.max_train_tokens
    return cap is not None and cap < inputs.tokens_per_epoch


def no_epoch_warning(spec: TrainSpec, inputs: RunInputs) -> str:
    return (
        f"no epoch will finish: --max-train-tokens {spec.max_train_tokens} is below the "
        f"{inputs.tokens_per_epoch} tokens of one epoch (tokens_per_epoch), so nothing is "
        "kept or evaluated; set --max-train-tokens to at least tokens_per_epoch"
    )


def _preview(
    provider: Provider,
    run_dir: Path,
    run_id: str,
    spec: TrainSpec,
    profile: ModelProfile,
    inputs: RunInputs,
    *,
    provider_source: str,
    defaults_used: Sequence[str],
    max_seq_len_flag: str,
) -> dict[str, Any]:
    warnings = []
    if not inputs.validation_rows:
        warnings.append(
            "the validation split has no rows to validate on, so the best epoch is chosen "
            "by training loss"
        )
    if no_epoch_can_finish(spec, inputs):
        warnings.append(no_epoch_warning(spec, inputs))
    if (remote := profile.remote_code_warning()) is not None:
        warnings.append(remote)
    rows = _rows(inputs)
    usage = _usage(spec, inputs, epochs=spec.max_epochs)
    max_steps = spec.max_steps(len(inputs.train_rows))
    return {
        "action": _action(profile, spec, rows),
        "billing": provider.training_billing(usage),
        "provider": provider.name,
        "provider_source": provider_source,
        "model": profile.alias,
        "profile": _profile_brief(profile),
        "run_id": run_id,
        "run_dir": display_path(run_dir),
        "rows": rows,
        "examples": inputs.examples(),
        "rejected_by_code": inputs.rejected_by_code(),
        "rejected_rows": [row.to_json() for row in inputs.rejected[:REJECTED_SAMPLE]],
        "tokens_per_epoch": inputs.tokens_per_epoch,
        "validation_tokens": inputs.validation.train_tokens,
        "max_tokens_total": usage.max_tokens,
        "max_steps": max_steps,
        "warmup_steps": spec.warmup_steps(max_steps),
        "settings": spec.resolved_settings(profile, max_seq_len_flag=max_seq_len_flag),
        "defaults_used": list(defaults_used),
        "data_destination": f"training data is sent to {provider.name}",
        "data_recipients": [f"training rows -> {provider.name}"],
        "warnings": warnings,
    }


AFTER_WARNINGS: Final = ("next_command", "next_steps", "files")
"""What a foreground run's result shows after its ``notes`` and ``warnings``."""


def _finished(
    run_dir: RunDir,
    result: Mapping[str, Any],
    *,
    evaluated: bool = False,
    evaluation: Mapping[str, Any] | None = None,
    notes: ParentSetup | None = None,
) -> dict[str, Any]:
    """What a foreground run prints for ``result`` (``result.json``): the decision, what
    next, and where the files are (:func:`~pup_finetune.cli._views.train_view`).

    ``evaluated`` (``train --evaluate``) adds the ``evaluation``, whose next step, when it
    has one, comes first: continue it, publish it, deploy. ``notes`` adds what setting up
    ``./runs`` did (``notes``: that ``runs/.gitignore`` was created, say) and found
    (first in the ``warnings``).
    """
    next_steps = run_next(
        run_dir.path, str(result.get("phase")), promoted=_promoted(result), running=False
    )
    if evaluation is not None and "next_command" in evaluation:
        next_steps = {
            "next_command": evaluation["next_command"],
            "next_steps": [*evaluation.get("next_steps", []), *next_steps["next_steps"]],
        }
    view = train_view(run_dir, result, next_steps, evaluated=evaluated, evaluation=evaluation)
    if notes is not None:
        # The notes just before the warnings, as every result shows them.
        warnings = [*notes.warnings, *view.pop("warnings")]
        view = {
            **{key: value for key, value in view.items() if key not in AFTER_WARNINGS},
            "notes": list(notes.notes),
            "warnings": warnings,
            **{key: value for key, value in view.items() if key in AFTER_WARNINGS},
        }
    return finite_json(view)


def _in_foreground(
    provider: Provider,
    run: RunDir,
    mode: WorkerMode,
    inputs: RunInputs,
    *,
    after_finalize: AfterFinalize | None = None,
) -> dict[str, Any]:
    """Train ``run`` in this process (Ctrl-C cancels gracefully); return its ``result.json``."""
    try:
        result = run_in_dir(
            provider,
            run,
            mode,
            inputs,
            on_event=TrainingProgress(),
            after_finalize=after_finalize,
        )
    except KeyboardInterrupt:
        # A second Ctrl-C stops at once; the run recorded where it stopped.
        try:
            phase = run.load_state().phase
        except FinetuneError:
            phase = "interrupted"
        raise Interrupted(
            f"training stopped at once (Ctrl-C): the run in {run.path} is {phase}",
            hint=resume_hint(run.path),
        ) from None
    return dict(result)


def _detached(
    run_dir: RunDir, run_id: str, pid: int, notes: ParentSetup | None = None
) -> dict[str, Any]:
    return {
        "status": "started",
        "run_id": run_id,
        "run_dir": display_path(run_dir.path),
        "pid": pid,
        "worker_log": display_path(run_dir.worker_log_path),
        "notes": [] if notes is None else list(notes.notes),
        "warnings": [] if notes is None else list(notes.warnings),
        "next_command": watch_command(run_dir.path),
    }


REMOTE_UPLOAD: Final = (
    "train and validation rows, run files and pup-finetune source -> Baseten storage (the "
    "job's upload)"
)


def _add_remote_preview(
    preview: dict[str, Any],
    plan: RemotePlan,
    inputs: RunInputs,
    spec: TrainSpec,
    source: SourceTree,
) -> None:
    """Add to a ``train --remote`` preview where its training loop runs and what it uploads."""
    remote = plan.preview(
        {
            "rows": {"train": inputs.rows["train"], "validation": inputs.rows["validation"]},
            "code": f"pup-finetune {source.version} from {display_path(source.root)}",
            "run_files": [RUN_FILE, STATE_FILE],
        }
    )
    if spec.max_spend_usd is not None:
        remote["warnings"].append(
            "--max-spend-usd counts the trainer's time only; the CPU job bills on top"
        )
    preview["remote"] = remote
    preview["action"] += (
        f", in a Baseten CPU job in team {plan.team} that runs the training loop, so training "
        "continues while this machine sleeps"
    )
    preview["billing"] += f"; {plan.billing}"
    preview["data_destination"] += (
        "; the train and validation rows, their split assignments, the run files and "
        "pup-finetune's source are uploaded to Baseten storage for the job"
    )
    preview["data_recipients"] += [REMOTE_UPLOAD]


def _remote_started(
    run: RunDir,
    run_id: str,
    record: Mapping[str, Any],
    upload: Mapping[str, int],
    notes: ParentSetup,
) -> dict[str, Any]:
    return {
        "status": "started",
        "run_id": run_id,
        "run_dir": display_path(run.path),
        "remote": {
            "backend": record["backend"],
            "team": record["team"],
            "project": record["project"]["name"],
            "job_id": record["job_id"],
            "compute": record["compute"],
            "logs": record["logs"],
        },
        "upload": dict(upload),
        "notes": list(notes.notes),
        "warnings": list(notes.warnings),
        "next_command": watch_command(run.path),
    }


def _remote_sync(run: RunDir) -> RemoteSync | None:
    """Where the job of a run started with ``train --remote`` stands, its state copied into
    the run directory first; ``None`` for a run without one."""
    record = run.read_remote()
    if record is None:
        return None
    if remote_ended(record):
        return ended_sync(record)
    provider = get_provider(str(run.manifest()["provider"]), get_runtime())
    return sync_remote(run, remote_jobs(provider))


def provider_for_model(model: str, given: str | None, runtime: Runtime) -> tuple[str, str]:
    """The provider ``train`` uses for ``--model`` and where it came from: ``given``
    (``--provider``), else the one provider with a profile that ``model`` names
    (:func:`~pup_finetune.providers.providers_for_model`; development-only providers only in
    development mode). A name several providers offer, or none, is a usage error: the
    provider is never picked silently. Imports no provider module."""
    if given is not None:
        return given, "--provider"
    found = providers_for_model(model, runtime)
    if len(found) == 1:
        return found[0], f"inferred from --model {model}"
    if found:
        raise UsageError(
            f"{model} is offered by {joined(found)}; pass --provider",
            hint=f"for example --provider {found[0]}",
        )
    known = "; ".join(
        f"{', '.join(models)} ({name})" for name, models in known_models(runtime).items() if models
    )
    raise UsageError(
        f"unknown model {model!r}",
        hint=f"known models: {known}; `pup finetune models list` shows every name each accepts",
    )


@run_command
def train(
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset",
            metavar="DIR",
            help="The dataset directory to train on.",
            show_default=False,
            rich_help_panel=REQUIRED,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            metavar="MODEL",
            help=f"Model, e.g. {model_example()}; see models list.",
            show_default=False,
            rich_help_panel=REQUIRED,
        ),
    ],
    provider_name: Annotated[
        str | None,
        typer.Option(
            "--provider",
            metavar="NAME",
            help=f"Provider: {provider_choices()}. {default_note('from --model')}",
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir",
            metavar="DIR",
            help=f"A new or empty run directory. {default_note('runs/RUN_ID')}",
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            metavar="ID",
            help=f"Run and model id. {default_note('generated')}",
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach", help="Train in a background worker; see status.", rich_help_panel=RUN
        ),
    ] = False,
    remote: Annotated[
        bool,
        typer.Option("--remote", help="Train in a Baseten job; see status.", rich_help_panel=RUN),
    ] = False,
    team: Annotated[
        str | None,
        typer.Option(
            "--team",
            metavar="NAME",
            help=f"With --remote: Baseten team. {default_note('your default')}",
            show_default=False,
            rich_help_panel=RUN,
        ),
    ] = None,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Train (paid); without it, preview.", rich_help_panel=RUN),
    ] = False,
    max_epochs: Annotated[
        int, typer.Option(metavar="N", help="Most epochs to train.", rich_help_panel=TRAINING)
    ] = _int_default("max_epochs"),
    patience: Annotated[
        int,
        typer.Option(
            metavar="N", help="Epochs without improvement to stop.", rich_help_panel=TRAINING
        ),
    ] = _int_default("patience"),
    min_delta: Annotated[
        float,
        typer.Option(metavar="X", help="Least improvement that counts.", rich_help_panel=TRAINING),
    ] = _float_default("min_delta"),
    learning_rate: Annotated[
        float, typer.Option(metavar="X", help="Learning rate.", rich_help_panel=TRAINING)
    ] = _float_default("learning_rate"),
    lr_schedule: Annotated[
        str,
        typer.Option(metavar="NAME", help="Constant, linear or cosine.", rich_help_panel=TRAINING),
    ] = str(_default("lr_schedule")),
    warmup_fraction: Annotated[
        float,
        typer.Option(metavar="X", help="Share of steps warming up.", rich_help_panel=TRAINING),
    ] = _float_default("warmup_fraction"),
    batch_size: Annotated[
        int, typer.Option(metavar="N", help="Rows per optimizer step.", rich_help_panel=TRAINING)
    ] = _int_default("batch_size"),
    lora_rank: Annotated[
        int | None,
        typer.Option(
            metavar="N",
            help=f"LoRA rank. {_THE_MODELS}",
            rich_help_panel=TRAINING,
        ),
    ] = None,
    lora_alpha: Annotated[
        int | None,
        typer.Option(
            metavar="N",
            help=f"LoRA alpha. {default_note("the provider's")}",
            rich_help_panel=TRAINING,
        ),
    ] = None,
    max_seq_len: Annotated[
        int | None,
        typer.Option(
            "--max-seq-len",
            min=1,
            metavar="N",
            help=f"Context cap in tokens. {_THE_MODELS}",
            show_default=False,
            rich_help_panel=TRAINING,
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option(metavar="N", help="Seed.", rich_help_panel=TRAINING)
    ] = _int_default("seed"),
    max_train_tokens: Annotated[
        int | None,
        typer.Option(metavar="N", help="Stop before training more tokens.", rich_help_panel=BUDGET),
    ] = None,
    max_spend_usd: Annotated[
        float | None,
        typer.Option(metavar="X", help="Stop before spending more (USD).", rich_help_panel=BUDGET),
    ] = None,
    hourly_rate_usd: Annotated[
        float | None,
        typer.Option(
            metavar="X",
            help="Your price per hour, for --max-spend-usd.",
            rich_help_panel=BUDGET,
        ),
    ] = None,
    spend_reserve_fraction: Annotated[
        float,
        typer.Option(
            metavar="X",
            help="Share of --max-spend-usd kept.",
            rich_help_panel=BUDGET,
        ),
    ] = _float_default("spend_reserve_fraction"),
    evaluate: Annotated[
        bool,
        typer.Option(
            "--evaluate",
            help="Evaluate the run after training (paid).",
            rich_help_panel=EVALUATE,
        ),
    ] = False,
    eval_out_dir: Annotated[
        Path | None,
        typer.Option(
            "--eval-out-dir",
            metavar="DIR",
            help=f"Output directory. {default_note('RUN_DIR/eval')}",
            show_default=False,
            rich_help_panel=EVALUATE,
        ),
    ] = None,
    eval_max_cases: Annotated[
        int,
        typer.Option(
            "--eval-max-cases",
            min=1,
            metavar="N",
            help="Turns per conversation.",
            rich_help_panel=EVALUATE,
        ),
    ] = evaluation.DEFAULT_MAX_CASES_PER_TRAJECTORY,
    eval_judge_model: Annotated[
        str | None,
        typer.Option(
            "--eval-judge-model",
            metavar="MODEL",
            help="The judge, as evaluate --judge-model.",
            show_default=False,
            rich_help_panel=EVALUATE,
        ),
    ] = None,
    eval_concurrency: Annotated[
        int,
        typer.Option(
            min=1, max=64, metavar="N", help="Cases at once (1-64).", rich_help_panel=EVALUATE
        ),
    ] = evaluation.DEFAULT_CONCURRENCY,
    eval_max_tokens: Annotated[
        int,
        typer.Option(
            min=1, metavar="N", help="Most tokens per generation.", rich_help_panel=EVALUATE
        ),
    ] = evaluation.DEFAULT_MAX_TOKENS,
    eval_seed: Annotated[
        int,
        typer.Option(
            min=0, metavar="N", help="Seed for cases and bootstrap.", rich_help_panel=EVALUATE
        ),
    ] = evaluation.DEFAULT_SEED,
    eval_skip_calibration: Annotated[
        bool,
        typer.Option(
            "--eval-skip-calibration",
            help="Do not check the judge first.",
            rich_help_panel=EVALUATE,
        ),
    ] = False,
    eval_no_publish: Annotated[
        bool,
        typer.Option(
            "--eval-no-publish", help="Do not publish to Datadog.", rich_help_panel=EVALUATE
        ),
    ] = False,
    checkpoint_every_minutes: Annotated[
        float,
        typer.Option(metavar="X", help="Checkpoint interval.", rich_help_panel=ADVANCED),
    ] = _float_default("checkpoint_every_minutes"),
    microbatch_token_budget: Annotated[
        int | None,
        typer.Option(metavar="N", help="Most tokens per microbatch.", rich_help_panel=ADVANCED),
    ] = None,
    provider_option: Annotated[
        list[str] | None,
        typer.Option(
            "--provider-option",
            metavar="KEY=VALUE",
            help=f"Repeatable. {provider_option_keys()}",
            show_default=False,
            rich_help_panel=ADVANCED,
        ),
    ] = None,
    old_eval_max_cases: Annotated[
        int | None,
        typer.Option(
            "--eval-max-cases-per-trajectory",
            min=1,
            hidden=True,
            help="With --evaluate: the old name of --eval-max-cases.",
        ),
    ] = None,
) -> dict[str, Any] | CommandResult:
    require_enablers(TRAIN_NEEDS)
    eval_max_cases = renamed("eval_max_cases", "old_eval_max_cases")
    ctx = current_context()
    runtime = get_runtime()
    if detach:
        require_detach_supported()
    if evaluate and detach:
        raise UsageError(
            "--evaluate needs the foreground: the evaluation runs in the training session",
            hint="drop --detach, or evaluate the run with `pup finetune evaluate` once it "
            "completed",
        )
    if remote and detach:
        raise UsageError("--remote already trains away from this machine", hint="drop --detach")
    if remote and evaluate:
        raise UsageError(
            "--evaluate needs the foreground: the evaluation runs in the training session",
            hint="drop --evaluate, and evaluate the run with `pup finetune evaluate RUN` once "
            "status shows it completed",
        )
    run_id = check_run_id(run_id) if run_id is not None else new_run_id()
    default_dir = run_dir is None
    if run_dir is None:
        run_dir = default_run_dir(run_id)
    require_new_run_dir(run_dir)
    RunDir(run_dir).require_empty(default=default_dir)
    require_dataset_dir(dataset_dir)
    provider_name, provider_source = provider_for_model(model, provider_name, runtime)
    dataset = DatasetDir(dataset_dir)
    by_split = dataset.rows_by_split()
    require_in_datadog(
        dataset,
        by_split,
        "train",
        directory=dataset_dir,
        offline=provider_info(provider_name, runtime).offline,
    )
    # What the preview can know fails here, not after --confirm: a package the provider needs
    # (its SDK, the renderer), or a flag it cannot honor.
    if remote:
        require_remote(provider_name, runtime)
    for needed in ("train", "evaluate") if evaluate else ("train",):
        require_packages(provider_name, runtime, needed)
    provider = get_provider(provider_name, runtime)
    profile = provider.resolve_model(model)
    raw_options = _provider_options(provider_option or [])
    if max_seq_len is not None:
        raw_options = _with_max_seq_len(provider.name, raw_options, max_seq_len)
        if max_seq_len > profile.max_seq_len:
            # Named by the flag the user typed; the provider's own check names its option.
            raise UsageError(
                f"--max-seq-len {max_seq_len} is over the {profile.max_seq_len}-token context "
                f"of {profile.alias}",
                hint=f"pass --max-seq-len N with N at most {profile.max_seq_len}",
            )
    options = provider.validate_options(raw_options)
    # What the run trains with: an option may narrow the profile (e.g. cap its context),
    # so rows are rendered, previewed and recorded against the narrowed one.
    profile = provider.training_profile(profile)
    flags: dict[str, Any] = {
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "warmup_fraction": warmup_fraction,
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
    require_supported_flags(provider.name, runtime, flags)
    spec = _spec(flags, options)
    provider.preflight(profile)
    renderer = provider.renderer(profile)
    inputs = build_inputs(dataset, renderer, profile, by_split)
    defaults_used = _defaults_used(ctx, raw_options)
    seq_flag = MAX_SEQ_LEN_FLAG if max_seq_len is not None else MAX_SEQ_LEN_OPTION_FLAG
    preview = _preview(
        provider,
        run_dir,
        run_id,
        spec,
        profile,
        inputs,
        provider_source=provider_source,
        defaults_used=defaults_used,
        max_seq_len_flag=seq_flag,
    )
    rplan: RemotePlan | None = None
    if remote:
        jobs = remote_jobs(provider)
        source = package_source()
        require_locked_renderer(source)
        rplan = jobs.plan(run_id=run_id, team=team)
        _add_remote_preview(preview, rplan, inputs, spec, source)
    plan: evaluation.EvalPlan | None = None
    if evaluate:
        settings = evaluation.EvalSettings(
            out_dir=eval_out_dir
            if eval_out_dir is not None
            else run_dir / evaluation.DEFAULT_OUT_DIR_NAME,
            max_cases_per_trajectory=eval_max_cases,
            judge_model=eval_judge_model,
            concurrency=eval_concurrency,
            max_tokens=eval_max_tokens,
            seed=eval_seed,
            skip_calibration=eval_skip_calibration,
            publish=not eval_no_publish,
        )
        # Before anything is paid for: an out dir the evaluation cannot use fails here, not
        # after training.
        evaluation.require_new_eval_dir(settings.out_dir, run_dir)
        plan = evaluation.plan_evaluation(
            runtime,
            run_id=run_id,
            run_dir=run_dir,
            provider=provider.name,
            profile=profile,
            dataset=dataset,
            rendering=renderer.identity(),
            settings=settings,
            no_publish=evaluation.EVAL_NO_PUBLISH,
        )
        preview["evaluation"] = plan.preview(
            tuned="the best epoch of this run", tuned_checkpoint=None
        )
        preview["action"] += f", then {preview['evaluation']['action']}"
        # Each part names who bills it, so the card can say who bills what.
        preview["billing"] += f"; {preview['evaluation']['billing']}"
        # The consent fields name every recipient, the evaluation's judge included.
        preview["data_destination"] += f"; {preview['evaluation']['data_destination']}"
        preview["data_recipients"] += preview["evaluation"]["data_recipients"]
    # The rerun pins the provider, the model's alias, the run id and directory (even when
    # inferred or defaulted), so it trains exactly what was previewed.
    rerun = [*PROGRAM, "train", "--provider", provider.name]
    rerun += ["--dataset", str(dataset_dir), "--model", profile.alias]
    rerun += ["--run-dir", str(run_dir), "--run-id", run_id]
    for field, value in flags.items():
        if value is not None and given(ctx, field):
            rerun += [_FLAGS[field], str(value)]
    if max_seq_len is not None:
        rerun += ["--max-seq-len", str(max_seq_len)]
    for key, value in raw_options.items():
        if not (key == MAX_SEQ_LEN_OPTION and max_seq_len is not None):
            rerun += ["--provider-option", f"{key}={value}"]
    if detach:
        rerun.append("--detach")
    if rplan is not None:
        rerun += ["--remote", "--team", rplan.team]
    if plan is not None:
        rerun += plan.train_flags()
    # PUP_READ_ONLY refuses paid and remote runs; one that never leaves the machine (the
    # offline fake provider, and an evaluation that is offline too) is allowed.
    offline = provider.name == "fake" and (plan is None or plan.offline)
    if confirm and not offline:
        require_writable(runtime, f"training on {provider.name}", what=PAID_RUN)
    require_confirm(
        confirm,
        preview,
        shlex.join([*rerun, "--confirm"]),
        read_only=runtime.read_only and not offline,
    )
    # A new ./runs gets a .gitignore: runs hold trace-derived rows.
    notes = ignore_new_parent(run_dir)
    if rplan is not None:
        created_dir = not run_dir.exists()
        run = _create_run(
            provider,
            run_dir,
            run_id,
            profile,
            spec,
            dataset,
            inputs,
            renderer,
            defaults_used,
            max_seq_len_flag=seq_flag,
        )
        record, upload = submit_remote(
            run, jobs, rplan, source, mode="train", created_dir=created_dir
        )
        return _remote_started(run, run_id, record, upload, notes)
    if plan is not None:
        calls = plan.work(None).judge_calls
        with evaluation.open_judge_client(plan.judge, calls=calls) as client:
            # Before anything is trained: a judge that fails calibration stops the run here.
            # A run no epoch of which can finish is never evaluated: calibrating would only
            # spend judge calls (the preview warned).
            calibration = None
            if no_epoch_can_finish(spec, inputs):
                info(
                    "not calibrating the judge: no epoch can finish under --max-train-tokens, "
                    "so the run will not be evaluated"
                )
            else:
                calibration = evaluation.run_calibration(plan, client)
            if calibration is not None and not calibration.get("ok"):
                return CommandResult(evaluation.calibration_failed(plan, calibration), exit_code=1)
            hook = evaluation.EvaluateAfterTraining(
                plan,
                client,
                calibration=calibration,
                max_checkpoint_name_len=provider.backend().max_checkpoint_name_len,
            )
            run = _create_run(
                provider,
                run_dir,
                run_id,
                profile,
                spec,
                dataset,
                inputs,
                renderer,
                defaults_used,
                max_seq_len_flag=seq_flag,
            )
            info(f"training run {run_id} in {run_dir} on {provider.name}, then evaluating it")
            result = _in_foreground(provider, run, "train", inputs, after_finalize=hook)
            data = _finished(run, result, evaluated=True, evaluation=hook.outcome, notes=notes)
            code = 0 if hook.outcome is None else evaluation.evaluation_exit_code(hook.outcome)
            # The trained run is kept either way; a failed evaluation still fails the command.
            return CommandResult(data, exit_code=code) if code else data
    run = _create_run(
        provider,
        run_dir,
        run_id,
        profile,
        spec,
        dataset,
        inputs,
        renderer,
        defaults_used,
        max_seq_len_flag=seq_flag,
    )
    if detach:
        return _detached(run, run_id, spawn_worker(run, "train"), notes)
    info(f"training run {run_id} in {run_dir} on {provider.name} (Ctrl-C cancels)")
    return _finished(run, _in_foreground(provider, run, "train", inputs), notes=notes)


def _create_run(
    provider: Provider,
    run_dir: Path,
    run_id: str,
    profile: ModelProfile,
    spec: TrainSpec,
    dataset: DatasetDir,
    inputs: RunInputs,
    renderer: Renderer,
    defaults_used: Sequence[str],
    *,
    max_seq_len_flag: str,
) -> RunDir:
    return RunDir.create(
        run_dir,
        provider=provider.name,
        profile=profile,
        spec=spec,
        dataset_dir=dataset.path,
        dataset_sha256=inputs.dataset_sha256,
        rendering=renderer.identity(),
        run_id=run_id,
        defaults_used=defaults_used,
        max_seq_len_flag=max_seq_len_flag,
    )


def _by_id(source: str) -> dict[str, str]:
    """``run_dir_source`` for a result whose run was named by a bare run id, else nothing."""
    return {"run_dir_source": source} if source == RUN_DIR_BY_ID else {}


def _with_source(result: Mapping[str, Any], source: str) -> dict[str, Any]:
    """``result`` with :func:`_by_id` right after its ``run_dir``."""
    shown: dict[str, Any] = {}
    for key, value in result.items():
        shown[key] = value
        if key == "run_dir":
            shown.update(_by_id(source))
    return shown


# --- resume ------------------------------------------------------------------------------


def _resume_mode(run: RunDir, state: RunState) -> WorkerMode:
    """What resuming ``run`` does; raises when it cannot be resumed now."""
    worker = worker_status(run, phase=state.phase, pid=state.pid, heartbeat_at=state.heartbeat_at)
    if worker.starting:
        raise WrongPhase(
            f"a worker for the run in {run.path} is starting (process {worker.pid})",
            hint=f"follow it with `pup finetune status {_q(run.path)} --watch`",
        )
    if state.phase == "created":
        return "train"  # it never started (e.g. its worker failed to start): start it
    if state.phase == "training":
        if lock_held(run):
            raise WrongPhase(
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


def _run_settings(run: RunDir, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What the run trains with, as a preview shows it: ``run.json``'s ``settings`` without
    what no flag changes, or, for a run recorded before it had them, resolved again."""
    recorded = manifest.get("settings")
    if isinstance(recorded, Mapping):
        hidden = ("pipeline_depth", "optimizer")
        return {key: value for key, value in recorded.items() if key not in hidden}
    return run.spec().resolved_settings(run.profile())


def _resume_billing(provider: Provider, mode: WorkerMode, usage: TrainingUsage) -> str:
    if mode == "finalize":
        if provider.name == "fake":
            return f"{OFFLINE_BILLING}: it trains nothing, only promotes the best checkpoint"
        return (
            f"{provider.name} bills the session that promotes the best checkpoint at its own "
            "rates, which pup-finetune does not know: it trains nothing"
        )
    return provider.training_billing(usage)


@run_command
def resume(
    run_dir: RunDirArgument,
    detach: DetachOption = False,
    confirm: ConfirmOption = False,
) -> dict[str, Any]:
    runtime = get_runtime()
    if detach:
        require_detach_supported()
    run_dir, source = resolve_run_ref(run_dir, "resume")
    run = RunDir(run_dir)
    sync = _remote_sync(run)
    if sync is not None and sync.view["state"] != "ended":
        raise WrongPhase(
            f"the run in {run_dir} trains in {job_name(sync.view)} ({sync.view['state']})",
            hint=f"follow it with `pup finetune status {_q(run_dir)} --watch`; cancel cannot "
            "stop a Baseten job yet",
        )
    manifest = run.manifest()
    state = run.load_state()
    mode = _resume_mode(run, state)
    # Before the provider loads: a missing SDK fails the preview, not the worker.
    require_packages(str(manifest["provider"]), runtime, "train")
    try:
        require_supported_flags(str(manifest["provider"]), runtime, run.spec().model_dump())
    except UsageError as exc:
        # A resume takes no training flags: the run recorded the one refused.
        flag = exc.message.partition(" ")[0]
        exc.hint = (
            f"the run was recorded with {flag}, and a resume trains with its recorded "
            f"settings: start a new run without {flag} (`pup finetune train --provider "
            "<PROVIDER> --dataset <DIR> --model <MODEL>`)"
        )
        raise
    provider, inputs = load_run_inputs(run, runtime, finalize_only=mode == "finalize")
    action, resume_from = _resume_action(mode, state)
    spec = run.spec()
    max_epochs = int(manifest["spec"]["max_epochs"])
    remaining = 0 if mode == "finalize" else max(max_epochs - state.epochs, 0)
    usage = _usage(spec, inputs, epochs=remaining, resumed=True)
    rows = _rows(inputs)
    preview = {
        "action": action,
        "billing": _resume_billing(provider, mode, usage),
        "run_id": manifest["run_id"],
        "run_dir": display_path(run_dir),
        **_by_id(source),
        "provider": provider.name,
        "model": manifest["profile"]["alias"],
        "phase": state.phase,
        "mode": mode,
        "resume_from": resume_from,
        "epochs_completed": state.epochs,
        "remaining_epochs": remaining,
        "rows": rows,
        "tokens_per_epoch": inputs.tokens_per_epoch,
        "validation_tokens": inputs.validation.train_tokens,
        "max_tokens_total": usage.max_tokens,
        "settings": _run_settings(run, manifest),
        "defaults_used": list(manifest.get("defaults_used") or []),
        "data_destination": f"training data is sent to {provider.name}",
        "data_recipients": [f"training rows -> {provider.name}"],
        "warnings": [] if sync is None else list(sync.warnings),
    }
    if sync is not None and sync.view["job_id"]:
        preview["warnings"].append(
            f"this run trained in {job_name(sync.view)}; this resume trains on this machine"
        )
    rerun = [*PROGRAM, "resume", str(run_dir), *(["--detach"] if detach else [])]
    offline = provider.name == "fake"
    if confirm and not offline:
        require_writable(runtime, f"training on {provider.name}", what=PAID_RUN)
    require_confirm(
        confirm,
        preview,
        shlex.join([*rerun, "--confirm"]),
        read_only=runtime.read_only and not offline,
    )
    if detach:
        return _with_source(
            _detached(run, str(manifest["run_id"]), spawn_worker(run, mode)), source
        )
    info(f"resuming run {manifest['run_id']} in {run_dir}: {action} (Ctrl-C cancels)")
    return _with_source(_finished(run, _in_foreground(provider, run, mode, inputs)), source)


# --- status ------------------------------------------------------------------------------


def watch_command(path: Path) -> str:
    """`status --watch` for the run in ``path``; under a coding agent with ``--timeout``
    pinned to :data:`AGENT_WATCH_TIMEOUT_S`, so the command a result names says how long it
    blocks."""
    words = ["--watch"]
    if get_runtime().under_agent:
        words += ["--timeout", f"{AGENT_WATCH_TIMEOUT_S:g}s"]
    return command("status", path, *words)


NOTHING_TRAINED_STEP = (
    "no epoch finished, so there is nothing to evaluate or deploy: train a new run (with a "
    "--max-train-tokens of at least the preview's tokens_per_epoch, or a larger --max-spend-usd, "
    "if a cap stopped it; a free preview)"
)


def run_next(
    path: Path, phase: str, *, promoted: bool, running: bool, stale: bool = False
) -> dict[str, Any]:
    """``next_command`` and ``next_steps`` for the run in ``path``, from its state.

    A running run is watched, a stale one cancelled. A completed run with a model
    (``promoted``: a promoted model or a deployable checkpoint) goes on as
    :func:`~pup_finetune.evaluation.runner.next_after_training` says: evaluated (a free
    preview), its recorded evaluation continued, or, once that is complete, deployed (with
    no next command and the evaluation's steps while the missing push kept it unpublished). So
    does a cancelled one whose best epoch was promoted, with resuming it as a step. A
    completed run without one (no epoch finished, e.g. under a token budget below one
    epoch) has no next command: the step trains a new run. Anything else (interrupted,
    failed, or cancelled before any epoch) is resumed.
    """
    steps: list[dict[str, str]] = []
    next_command: str | None
    if running:
        next_command = watch_command(path)
    elif phase == "completed" and not promoted:
        next_command = None
        steps.append(
            step(
                template(
                    "train", "--provider", "<PROVIDER>", "--dataset", "<DIR>", "--model", "<MODEL>"
                ),
                NOTHING_TRAINED_STEP,
            )
        )
    elif phase == "completed":
        after = evaluation.next_after_training(path)
        next_command = after["next_command"]
        steps += after["next_steps"]
    elif phase == "cancelled" and promoted:
        after = evaluation.next_after_training(path)
        next_command = after["next_command"]
        steps += after["next_steps"]
        steps.append(
            step(resume_command(path), "to train the remaining epochs (paid; it previews first)")
        )
    elif phase == "training" and stale:
        next_command = command("cancel", path)
    else:
        next_command = resume_command(path)
    return {"next_command": next_command, "next_steps": steps}


def _promoted(result: Mapping[str, Any]) -> bool:
    """Whether ``result`` (``result.json`` or a status) has a model to evaluate and deploy."""
    return bool(result.get("promoted_model") or result.get("deployable_checkpoint"))


def _rounded(value: float | None, digits: int = 0) -> float | int | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value) if digits == 0 else round(value, digits)


def _timing(summary: Mapping[str, Any], *, live_s: float) -> dict[str, Any]:
    """``elapsed_seconds``, ``approx_eta_seconds`` and, with a rate, ``estimated_spend_usd``.

    ``live_s`` is the time since the state was saved while training is under way (else 0):
    it counts as elapsed. The ETA is the measured seconds per step as of that save
    (validation and checkpoints included) times the steps left to ``max_epochs``, so it is a
    rough estimate, neither bound: early stopping can end the run sooner, and the validation
    still to come (none measured yet in the first epoch) can make it later; ``None`` before
    the first step, once every step ran (only the last epoch's end and the promotion are
    left), or when the run is not training.
    """
    recorded = float(summary["elapsed_seconds"] or 0.0)
    step, max_steps = summary["step"], summary["max_steps"]
    eta = None
    if summary["phase"] == "training" and max_steps is not None and 0 < step < max_steps:
        eta = recorded / step * (max_steps - step)
    elapsed = recorded + live_s
    timing: dict[str, Any] = {
        "elapsed_seconds": _rounded(elapsed),
        "approx_eta_seconds": _rounded(eta),
    }
    rate = summary["hourly_rate_usd"]
    if isinstance(rate, int | float):
        timing["estimated_spend_usd"] = _rounded(elapsed / 3600 * rate, 2)
    return timing


def run_status(run: RunDir, *, source: str = "given") -> dict[str, Any]:
    """What `status` measures for ``run``; every number finite or ``None``.

    ``worker`` is the :class:`~pup_finetune.training.worker.WorkerStatus`; what `status`
    prints is :func:`~pup_finetune.cli._views.status_view` of this. ``source`` is how the
    command found the run (``run_dir_source``: ``given``, ``run id`` or ``newest under
    ./runs``). ``evaluation`` is the run's recorded evaluation
    (:func:`~pup_finetune.evaluation.runner.recorded_evaluation`), or ``None``.
    """
    sync = _remote_sync(run)
    summary = run.status()
    if sync is not None and sync.worker is not None:
        worker = sync.worker
    else:
        worker = worker_status(
            run,
            phase=summary["phase"],
            pid=summary["pid"],
            heartbeat_at=summary["heartbeat_at"],
        )
    best = summary["best"] or {}
    stale_alive = worker.stale and worker.alive
    warnings = list(summary["warnings"])
    if sync is not None:
        warnings += sync.warnings
    if worker.problem is not None:
        warnings.append(worker.problem)
    live = summary["phase"] == "training" and worker.running
    live_s = (worker.heartbeat_age_s or 0.0) if live else 0.0
    data: dict[str, Any] = finite_json(
        {
            "run_id": summary["run_id"],
            "run_dir": display_path(run.path),
            "run_dir_source": source,
            "provider": summary["provider"],
            "model": summary["model"],
            "phase": summary["phase"],
            "epoch": summary["epoch"],
            "step": summary["step"],
            "epochs_completed": summary["epochs_completed"],
            "max_epochs": summary["max_epochs"],
            "steps_per_epoch": summary["steps_per_epoch"],
            "max_steps": summary["max_steps"],
            "last_train_loss": summary["last_train_loss"],
            "val_losses": summary["val_losses"],
            "best_epoch": best.get("epoch"),
            "best_val_loss": best.get("val_loss"),
            **_timing(summary, live_s=live_s),
            "stopped_reason": summary["stopped_reason"],
            "promoted_model": summary["promoted_model"],
            "deployable_checkpoint": summary["deployable_checkpoint"],
            "evaluation": recorded_evaluation_view(evaluation.recorded_evaluation(run.path)),
            "settings": changed_settings(summary["settings"], summary["defaults_used"]),
            "resume_mode": summary["resume_mode"],
            "error": summary["error"] or worker.error,
            "finalize_error": summary["finalize_error"],
            "warnings": warnings,
            **run_next(
                run.path,
                summary["phase"],
                promoted=_promoted(summary),
                running=worker.running,
                stale=stale_alive,
            ),
            "files": files(
                run=run.run_path,
                state=run.state_path,
                epochs=run.epochs_path,
                result=run.result_path if run.result_path.exists() else None,
                **({"remote": run.remote_path} if sync is not None else {}),
            ),
        }
    )
    data["worker"] = worker
    if sync is not None:
        data["remote"] = sync.view
    return data


def _progress_line(data: Mapping[str, Any]) -> str:
    """The watch's progress line for what `status` measured: :func:`status_line` of what it
    prints, knowing also whether the epoch's last step is done and its validation runs."""
    return status_line(
        status_view(data),
        steps_per_epoch=data["steps_per_epoch"],
        epochs_completed=data["epochs_completed"],
    )


DURATION_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}
"""What a ``--timeout`` suffix multiplies the number by: none or ``s``, ``m`` and ``h``."""

WATCH_INTERVAL_S = 10.0
"""The default ``--interval`` of a watch."""

AGENT_WATCH_TIMEOUT_S = 90.0
"""The default ``--timeout`` of a watch under a coding agent: below the 120 s that agents'
shell tools commonly allow a command."""

_DURATION = re.compile(r"(\d+(?:\.\d+)?)([smh]?)")


def parse_duration(text: str) -> float:
    """Seconds in ``90s``, ``10m``, ``2h`` or a bare number of seconds; else :class:`UsageError`."""
    match = _DURATION.fullmatch(text.strip())
    seconds = float(match[1]) * DURATION_UNITS[match[2]] if match else 0.0
    if seconds <= 0 or not math.isfinite(seconds):
        raise UsageError(
            f"invalid --timeout {text!r}",
            hint="pass a positive duration such as 90s, 10m, 2h, or a number of seconds",
        )
    return seconds


def _watch_command(run_dir: Path, *, interval: float | None, timeout: str | None) -> str:
    """The `status --watch` command as typed: the run, then ``--interval`` and ``--timeout``."""
    words = [*PROGRAM, "status", str(run_dir), "--watch"]
    if interval is not None:
        words += ["--interval", f"{interval:g}"]
    if timeout is not None:
        words += ["--timeout", timeout]
    return shlex.join(words)


def _watched(data: Mapping[str, Any], *, timed_out: bool, next_command: str) -> dict[str, Any]:
    """What a watch prints: ``data`` with ``watch_timed_out`` and the given ``next_command``."""
    return status_view({**data, "watch_timed_out": timed_out, "next_command": next_command})


def _stopped_watching(data: Mapping[str, Any], why: str, again: str) -> tuple[str, str]:
    """The stderr line for a watch that stops before training ends, and the next command."""
    worker: WorkerStatus = data["worker"]
    remote = data.get("remote")
    if worker.running and isinstance(remote, Mapping) and remote.get("state") != "ended":
        return f"{why}; the run keeps training in {job_name(remote)}. Watch again: {again}", again
    if worker.running:
        pid = f" (pid {worker.pid})" if worker.pid is not None else ""
        what = (
            "the run keeps training" if data["phase"] == "training" else "its worker keeps starting"
        )
        return f"{why}; {what}{pid}. Watch again: {again}", again
    next_command = str(data["next_command"])
    return (
        f"{why}; the run is {data['phase']} and no process is training it. Next: {next_command}",
        next_command,
    )


@contextlib.contextmanager
def ctrl_c_interrupts() -> Iterator[None]:
    """Make Ctrl-C (SIGINT) raise :class:`KeyboardInterrupt` inside, whatever the process
    inherited (an ignored SIGINT included), and restore the previous handler after. Outside
    the main thread, where handlers cannot be set, it changes nothing."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _newest_run() -> Path:
    """What `status` shows when it is given no run: the newest run under ``./runs``, named
    on stderr; :class:`NotARunDir` when there is none."""
    found = latest_run_dir()
    if found is None:
        raise NotARunDir(
            f"no run under {RUNS_DIR}/ to show",
            hint="pass a run directory or run id, or start a run with `pup finetune train "
            "--provider <PROVIDER> --dataset <DIR> --model <MODEL>`",
        )
    info(f"the newest run under {RUNS_DIR}/: {display_path(found)}")
    return found


@run_command
def status(
    run_dir: Annotated[
        Path | None,
        typer.Argument(
            metavar="RUN_DIR",
            help=f"The run: its directory or run id. {default_note('the newest run in ./runs')}",
            show_default=False,
        ),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Print progress until training ends, then the status.",
        ),
    ] = False,
    interval: Annotated[
        float | None,
        typer.Option(
            min=0.1,
            metavar="X",
            help=f"With --watch: seconds between checks. {default_note(f'{WATCH_INTERVAL_S:g}')}",
            show_default=False,
        ),
    ] = None,
    timeout: Annotated[
        str | None,
        typer.Option(
            "--timeout",
            metavar="DURATION",
            help="With --watch: stop watching after DURATION (90s, 10m, 2h). "
            f"{default_note(f'{AGENT_WATCH_TIMEOUT_S:g}s under a coding agent, else none')}",
            show_default=False,
        ),
    ] = None,
) -> dict[str, Any]:
    require_enablers(STATUS_NEEDS)
    if run_dir is None:
        run_dir, source = _newest_run(), RUN_DIR_NEWEST
    else:
        run_dir, source = resolve_run_ref(run_dir, "status")
    limit = parse_duration(timeout) if timeout is not None else None
    runtime = get_runtime()
    quiet = runtime.quiet
    if limit is None and watch and runtime.under_agent:
        limit = AGENT_WATCH_TIMEOUT_S
    run = RunDir(run_dir)
    data = run_status(run, source=source)
    if not watch:
        return status_view(data)
    again = _watch_command(run_dir, interval=interval, timeout=timeout)
    pause = WATCH_INTERVAL_S if interval is None else interval
    deadline = None if limit is None else time.monotonic() + limit
    limit_s = limit or 0.0
    # Always with a unit (`--timeout 10` reads "10s"); a sub-second limit keeps its fraction.
    shown_limit = format_duration(limit_s) if limit_s >= 1 else f"{limit_s:g}s"
    timed_out = f"stopped watching after {shown_limit}"
    shown = None
    # Ctrl-C stops the watch even when this process started with SIGINT ignored (a
    # background job, or some agents' shells): the default handler is installed meanwhile.
    with ctrl_c_interrupts():
        try:
            while True:
                # A new line only when the run moved: the elapsed time alone does not count.
                progress = _progress_line({**data, "elapsed_seconds": 0})
                ended = not data["worker"].running
                # The closing summary repeats the last line, unless it is turned off.
                if progress != shown and not (ended and not quiet):
                    info(_progress_line(data))
                    shown = progress
                if ended:
                    return _watched(data, timed_out=False, next_command=data["next_command"])
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    message, next_command = _stopped_watching(data, timed_out, again)
                    info(message)
                    return _watched(data, timed_out=True, next_command=next_command)
                time.sleep(pause if left is None else min(pause, left))
                data = run_status(run, source=source)
        except KeyboardInterrupt:
            # Only this watch stops: the training process runs in its own session, and nothing
            # here signals it.
            data = run_status(run, source=source)
            message, next_command = _stopped_watching(data, "stopped watching", again)
            info(message)
            return _watched(data, timed_out=False, next_command=next_command)


# --- cancel ------------------------------------------------------------------------------


@run_command
def cancel(run_dir: RunDirArgument, confirm: CancelConfirmOption = False) -> dict[str, Any]:
    run_dir, source = resolve_run_ref(run_dir, "cancel")
    run = RunDir(run_dir)
    sync = _remote_sync(run)
    if sync is not None and sync.view["state"] != "ended":
        view = sync.view
        raise UsageError(
            f"cancel cannot stop {job_name(view)} yet (team {view['team']}, project "
            f"{view['project']})",
            hint="stop the job in the Baseten UI; a stop skips the run's final save and "
            "promotion, and its Loops trainer may keep billing until you deactivate it there "
            f"too; then run `pup finetune status {_q(run_dir)}`",
        )
    run_id = run.run_id
    pid = cancellable_pid(run)
    preview = {
        "run_id": run_id,
        "run_dir": display_path(run_dir),
        **_by_id(source),
        "pid": pid,
        "action": f"send SIGTERM to process {pid}: training stops after the current step, "
        "saves a resumable checkpoint and keeps the best epoch so far",
        **({"warnings": sync.warnings} if sync is not None and sync.warnings else {}),
    }
    require_confirm(confirm, preview, command("cancel", run_dir, "--confirm"))
    pid = cancel_training(run)
    return {
        "run_id": run_id,
        "run_dir": display_path(run_dir),
        **_by_id(source),
        "pid": pid,
        "status": "cancelling",
        "next_command": watch_command(run_dir),
    }


# --- the worker --------------------------------------------------------------------------


@run_command
def worker(
    run_dir: RunDirArgument,
    mode: Annotated[str, typer.Option("--mode", help="train, resume or finalize.")],
    report_state: Annotated[
        bool, typer.Option("--report-state", hidden=True, help="Report the state on stdout.")
    ] = False,
) -> None:
    if mode not in WORKER_MODES:
        raise UsageError(
            f"unknown worker mode {mode!r}", hint=f"use one of {', '.join(WORKER_MODES)}"
        )
    run = RunDir(run_dir)
    with StateReporter(run) if report_state else nullcontext():
        try:
            runtime = get_runtime()
            # It runs what `train --confirm` or `resume --confirm` already allowed, but is a
            # command of its own: PUP_READ_ONLY must stop it too.
            provider_name = str(run.manifest()["provider"])
            if provider_name != "fake":
                require_writable(runtime, f"training on {provider_name}", what=PAID_RUN)
            info(f"worker {os.getpid()} started: {mode} {run_dir}")
            provider, inputs = load_run_inputs(run, runtime, finalize_only=mode == "finalize")
            result = run_in_dir(provider, run, mode, inputs, on_event=TrainingProgress())
        except BaseException as exc:
            record_worker_exit(run, mode, exc)
            raise
        record_worker_exit(run, mode, None)
        info(f"worker {os.getpid()} finished: the run is {result.get('phase')}")


TRAIN_DOC = CommandDoc(
    path="train",
    short="Fine-tune a model (paid; previews without --confirm).",
    description=describe(
        "Fine-tune --model with LoRA SFT on --dataset's train split, keeping the best epoch "
        "by validation loss. Previews without --confirm; trains with it.",
        "--dataset must be in Datadog: every kept row pushed with `dataset push`, or the "
        "directory made by `dataset fetch` (the offline fake provider trains on any). "
        "The run goes in runs/RUN_ID unless --run-dir names another directory (a runs/ this "
        "creates gets a .gitignore, since runs hold trace-derived rows). --model takes a "
        "model's alias, or in any case the provider's model id or its Hugging Face tokenizer "
        "name, and "
        "--provider is inferred from it when one provider offers that model; the preview says "
        "so (provider_source), and its next_command pins both, with the alias. "
        "The preview lists rows and rendered tokens per split, the rejected rows, tokens per "
        "epoch and in total, the settings and where the data goes. --detach trains in a "
        "background worker; follow it with `status --watch`. Training stops early after --patience epochs without a "
        "validation-loss improvement; a Ctrl-C or `cancel` still keeps the best epoch. "
        "--learning-rate is constant unless --lr-schedule linear or cosine decays it over the "
        "steps of --max-epochs epochs (the preview's max_steps; early stopping or a cap ends "
        "the run first), after a linear warmup over --warmup-fraction of them (warmup_steps); "
        "it never reaches zero, and the other Adam parameters are fixed (run.json "
        "settings.optimizer). "
        "With --evaluate, the run is also evaluated on the test split in its training "
        "session once the best epoch is kept (see `evaluate`): the judge is calibrated "
        "and --eval-out-dir checked before training starts, so a failed calibration trains "
        "nothing; --detach is refused with it; and the evaluation stops, partial, at "
        "--max-spend-usd. --max-seq-len caps the context (and so the cost): longer rows "
        "are left out, as the preview shows (exceeds_context). "
        "--remote (Baseten only) runs the training loop in a Baseten CPU job, so training "
        "continues while this machine sleeps; --team names the team (default: your only or "
        "default team). It needs the baseten CLI and uv on PATH, a source checkout of "
        "pup-finetune and a team secret pup_finetune_api_key; it uploads the train and "
        "validation rows, the run files and pup-finetune's source to Baseten storage, and is "
        "refused with --detach and --evaluate. status copies the job's state into the run "
        "directory; cancel cannot stop the job yet.",
    ),
    examples=(
        "pup finetune train --dataset ds --model qwen3p5-9b --lr-schedule cosine",
        "pup finetune train --dataset ds --model muse-glimmer-30b",
    ),
    safety=Safety(
        data="sends rows to the provider (and the judge)",
        recipients="the training provider; with --remote, also Baseten storage (the job's "
        "upload); with --evaluate, also the judge's provider and Datadog (experiments, when "
        "the dataset was pushed)",
        without="previews the run",
        confirm="trains (paid)",
        paid=True,
        read_only="confirm_refused",
        offline_allowed=True,
    ),
    panel=TRAIN_PANEL,
)
RESUME_DOC = CommandDoc(
    path="resume",
    short="Resume a stopped run (paid; previews without --confirm).",
    description=(
        "Continue an interrupted, failed or cancelled run from its last training checkpoint, "
        "or only retry its promotion when training already ended. Without --confirm, "
        "previews where it resumes."
    ),
    examples=(
        "pup finetune resume ft-20261007183443-a8b181",
        "pup finetune resume run --detach",
    ),
    safety=Safety(
        data="sends rows to the provider",
        recipients="the training provider",
        without="previews where it resumes",
        confirm="resumes training (paid)",
        paid=True,
        read_only="confirm_refused",
        offline_allowed=True,
    ),
    panel=TRAIN_PANEL,
)
STATUS_DOC = CommandDoc(
    path="status",
    short="Show a run's phase, progress, losses and worker.",
    description=describe(
        "Show a run's phase, epoch and step, losses, best epoch, elapsed time and ETA, "
        "promoted model, and whether its process is alive; with no RUN_DIR, the newest run "
        "under ./runs. With --watch, prints progress to stderr until training ends, then the "
        "status; Ctrl-C or --timeout stops only the watch.",
        "RUN_DIR may also be a bare run id, read as runs/RUN_ID when no such path exists "
        "(here and in resume, cancel, evaluate, deploy and undeploy, which always need one). "
        "Training keeps going when the watch stops: the status then has watch_timed_out "
        "true, and the command exits 0. Under a coding agent (PUP_AGENT_MODE, AI_AGENT or "
        f"CLAUDECODE set), a watch stops after {AGENT_WATCH_TIMEOUT_S:g}s unless --timeout "
        "says otherwise, below the time agents' shell tools commonly allow a command. "
        "For a run started with train --remote, status first reads its Baseten job and logs "
        "and copies the run's state into the run directory.",
    ),
    examples=("pup finetune status", "pup finetune status run --watch --timeout 10m"),
    safety=Safety(
        data="local files; reads a remote run's Baseten job",
        recipients="nowhere: it reads a remote run's Baseten job and its logs",
        without="shows the run",
    ),
    panel=TRAIN_PANEL,
)
CANCEL_DOC = CommandDoc(
    path="cancel",
    short="Stop a training run gracefully (previews without --confirm).",
    description=(
        "Stop the process training a run: it finishes the current step, saves a resumable "
        "checkpoint and keeps the best epoch so far. Without --confirm, previews which "
        "process gets the signal."
    ),
    examples=("pup finetune cancel run", "pup finetune cancel ft-20261007183443-a8b181"),
    safety=Safety(
        data="signals the local training process",
        recipients="nowhere: a signal to a local process",
        without="previews which process gets the signal",
        confirm="sends it SIGTERM",
    ),
    panel=TRAIN_PANEL,
)


def register(app: typer.Typer) -> None:
    """Add `train`, `resume`, `status`, `cancel` and the hidden `_worker` to ``app``."""
    app.command(**TRAIN_DOC.typer())(train)
    app.command(**RESUME_DOC.typer())(resume)
    app.command(**STATUS_DOC.typer())(status)
    app.command(**CANCEL_DOC.typer())(cancel)
    app.command("_worker", hidden=True)(worker)
