# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten Loops LoRA SFT: the free plan, and the paid training run in the foreground."""

from __future__ import annotations

import math
import shlex
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn, TypeGuard

from pydantic import SecretStr

from pup_finetune import output
from pup_finetune.data.jsonl import read_rows
from pup_finetune.data.schema import RowError, TrainingRow
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.providers import TrainRequest
from pup_finetune.providers.baseten.client import (
    API_BASE,
    BasetenHTTP,
    LoopsModule,
    LoopsTrainer,
    loops_sdk,
    open_trainer,
)
from pup_finetune.providers.baseten.models import MIN_SEQ_LEN, BasetenModel, get_model
from pup_finetune.rendering import (
    ChatTokenizer,
    ModelFamily,
    TokenDatum,
    check_rows,
    load_tokenizer,
    rendering_identity,
)
from pup_finetune.storage import DirectoryLock
from pup_finetune.training import (
    BudgetGuard,
    EpochRecord,
    LoopResult,
    RunDir,
    SFTSettings,
    pack_microbatches,
    run_sft,
)

__all__ = ["METHOD", "PreparedData", "TerminationRequested", "plan_training", "train"]

METHOD = "Baseten Loops LoRA SFT"
_DEACTIVATE_ATTEMPTS = 3
_POLL_ATTEMPTS = 12
_POLL_INTERVAL_SECONDS = 1.0

# Tests replace this to skip the deactivation polling delay.
_sleep: Callable[[float], None] = time.sleep


class TerminationRequested(BaseException):
    """SIGTERM arrived; a BaseException so no ``except Exception`` swallows it before cleanup."""


@dataclass(frozen=True, slots=True)
class PreparedData:
    """What :func:`train` feeds the trainer, as :func:`plan_training` resolved it."""

    model: BasetenModel
    lora_rank: int
    max_seq_len: int
    token_budget: int
    train_rows: list[list[TokenDatum]]
    """Each accepted training row's datums, kept together when rows are shuffled."""
    validation_batches: list[list[TokenDatum]]


def plan_training(
    request: TrainRequest,
    *,
    tokenizer_loader: Callable[[ModelFamily], ChatTokenizer] = load_tokenizer,
) -> tuple[dict[str, Any], PreparedData]:
    """Read and render both files and size the run. Free: no paid or remote-write call.

    Training rows that cannot render are excluded and listed; any validation row that cannot
    render is a :class:`UsageError`, as is having no training row left.
    """
    model = get_model(request.model)
    family = model.family
    settings = request.settings
    limit = family.max_seq_len
    render_limit = min(settings.max_seq_len or limit, limit)
    token_budget = settings.microbatch_token_budget or limit
    lora_rank = settings.lora_rank or model.default_lora_rank
    train_rows = _read(request.train_path)
    validation_rows = _read(request.validation_path)
    tokenizer = tokenizer_loader(family)
    train = check_rows(train_rows, family, tokenizer, max_seq_len=render_limit)
    validation = check_rows(validation_rows, family, tokenizer, max_seq_len=render_limit)
    if validation.rejected:
        first = validation.rejected[0]
        raise UsageError(
            f"{len(validation.rejected)} validation rows cannot be rendered for {model.alias}",
            hint=f"fix or remove them; first: {first['id']}: {first['code']}: {first['detail']}",
        )
    if not validation.accepted:
        raise UsageError(f"{request.validation_path} has no rows")
    if not train.accepted:
        hint = None
        if train.rejected:
            first = train.rejected[0]
            hint = (
                f"{len(train.rejected)} rows rejected; "
                f"first: {first['id']}: {first['code']}: {first['detail']}"
            )
        raise UsageError(
            f"no row of {request.train_path} can be rendered for {model.alias}", hint=hint
        )
    rows = [datums for _, datums in train.accepted]
    train_datums = [datum for datums in rows for datum in datums]
    validation_datums = [datum for _, datums in validation.accepted for datum in datums]
    # Packing also refuses a datum over the microbatch token budget before anything is paid.
    train_batches = pack_microbatches(train_datums, token_budget)
    validation_batches = pack_microbatches(validation_datums, token_budget)
    longest = max(len(datum.tokens) for datum in (*train_datums, *validation_datums))
    if settings.max_seq_len is None:
        max_seq_len, source = _fitted_seq_len(longest, limit), "data"
    else:
        max_seq_len, source = render_limit, "flag"
    steps = math.ceil(len(train_datums) / settings.batch_size)
    guard = _budget_guard(settings)
    plan = {
        "run_id": request.run_id,
        "provider": "baseten",
        "method": METHOD,
        "model": model.alias,
        "base_model": model.base_model,
        "run_dir": str(request.run_dir),
        "dataset": {
            "train_path": str(request.train_path),
            "validation_path": str(request.validation_path),
            "train_rows": len(rows),
            "validation_rows": len(validation.accepted),
            "train_datums": len(train_datums),
            "validation_datums": len(validation_datums),
            "rejected_train_rows": train.rejected,
        },
        "config": {
            **settings.model_dump(
                exclude={"max_spend_usd", "hourly_rate_usd", "spend_reserve_fraction"}
            ),
            "lora_rank": lora_rank,
            "max_seq_len": max_seq_len,
            "max_seq_len_source": source,
            "microbatch_token_budget": token_budget,
            "capacity": "dedicated",
        },
        "budget": {
            "enabled": guard.enabled,
            "max_spend_usd": settings.max_spend_usd,
            "hourly_rate_usd": settings.hourly_rate_usd,
            "spend_reserve_fraction": settings.spend_reserve_fraction,
            "maximum_active_seconds": guard.limit_seconds,
        },
        "rendering": rendering_identity(family, tokenizer).as_dict(),
        "estimates": {
            "optimizer_steps_per_epoch": steps,
            "max_optimizer_steps": steps * settings.max_epochs,
            "train_microbatches": len(train_batches),
            "validation_microbatches": len(validation_batches),
            "train_target_tokens": sum(sum(datum.weights) for datum in train_datums),
            "max_context_tokens": longest,
        },
    }
    prepared = PreparedData(
        model=model,
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
        token_budget=token_budget,
        train_rows=rows,
        validation_batches=validation_batches,
    )
    return plan, prepared


def train(
    request: TrainRequest,
    *,
    api_key: SecretStr,
    http: BasetenHTTP,
    loops: LoopsModule | None = None,
    tokenizer_loader: Callable[[ModelFamily], ChatTokenizer] = load_tokenizer,
) -> dict[str, Any]:
    """Train on Baseten Loops in the foreground; returns ``result.json`` plus the next command.

    Paid once the trainer is requested. A missing SDK or a refused environment variable fails
    before anything is written. On every exit, SIGTERM included (on the main thread),
    the trainer is closed and its run deactivated, and ``result.json`` is written before any
    error is raised.
    """
    sdk = loops_sdk(loops)
    run_dir = RunDir(request.run_dir)
    run_dir.create()
    with DirectoryLock(run_dir.path):
        run_dir.create()  # again under the lock, in case another run started here meanwhile
        plan, data = plan_training(request, tokenizer_loader=tokenizer_loader)
        _preflight(http.capabilities(), data.model.base_model, data.max_seq_len)
        run_dir.write_plan(plan)
        run_dir.write_epochs([])
        run_dir.update_state(status="planned", provider="baseten", run_id=request.run_id)
        result = _run(request, plan, data, run_dir, api_key=api_key, http=http, loops=sdk)
    next_command = None
    if result["best_sampler_weights_uri"]:
        next_command = (
            f"pup finetune deploy --run-dir {shlex.quote(str(request.run_dir))} "
            "--serving-mode sampler"
        )
    return {**result, "run_dir": str(request.run_dir), "next_command": next_command}


def _read(path: Path) -> list[TrainingRow]:
    try:
        return list(read_rows(path))
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except RowError as exc:
        raise UsageError(str(exc), hint="fix the row and rerun") from exc


def _fitted_seq_len(longest: int, limit: int) -> int:
    """The smallest power of two ``>= longest``, at least :data:`MIN_SEQ_LEN`, at most ``limit``."""
    return min(max(MIN_SEQ_LEN, 1 << (longest - 1).bit_length()), limit)


def _budget_guard(settings: SFTSettings) -> BudgetGuard:
    return BudgetGuard(
        max_spend_usd=settings.max_spend_usd,
        hourly_rate_usd=settings.hourly_rate_usd,
        reserve_fraction=settings.spend_reserve_fraction,
    )


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _preflight(capabilities: Mapping[str, Any], base_model: str, max_seq_len: int) -> None:
    """Refuse before anything is paid unless Baseten offers ``base_model`` at ``max_seq_len``."""
    models = capabilities.get("supported_models")
    matches = [
        entry
        for entry in (models if isinstance(models, list) else [])
        if isinstance(entry, dict) and entry.get("model_name") == base_model
    ]
    if len(matches) != 1:
        raise UsageError(
            f"Baseten Loops does not offer {base_model} to this workspace",
            hint="choose another --model, or ask Baseten to enable this one",
        )
    entry = matches[0]
    context = entry.get("max_context_length")
    enabled_limit = entry.get("max_enabled_seq_len", context)
    if entry.get("enabled") is False or enabled_limit == 0:
        details = entry.get("enablement_details")
        details = details if isinstance(details, dict) else {}
        reason = details.get("reason_detail") or details.get("reason")
        raise UsageError(
            f"this Baseten workspace is not enabled to train {base_model}"
            + (f": {reason}" if reason else ""),
            hint=str(details.get("remediation") or "ask Baseten to enable the model"),
        )
    if not (_is_int(context) and _is_int(enabled_limit)):
        raise ExternalError(
            f"Baseten returned an invalid context limit for {base_model}", source="baseten"
        )
    limit = min(context, enabled_limit)
    if limit < max_seq_len:
        raise UsageError(
            f"Baseten trains {base_model} on at most {limit} tokens, below this run's "
            f"max_seq_len of {max_seq_len}",
            hint=f"rerun with --max-seq-len {limit}",
        )


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not interrupt cleanup
    raise TerminationRequested


@contextmanager
def _sigterm_raises() -> Iterator[None]:
    """Turn SIGTERM into :class:`TerminationRequested`; signals can only be set on the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


@contextmanager
def _interrupts_ignored() -> Iterator[None]:
    """Ignore SIGINT and SIGTERM (on the main thread) so cleanup and the result always finish."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)


def _best(epochs: Sequence[EpochRecord]) -> EpochRecord | None:
    return next((record for record in reversed(epochs) if record.new_best), None)


def _run(
    request: TrainRequest,
    plan: Mapping[str, Any],
    data: PreparedData,
    run_dir: RunDir,
    *,
    api_key: SecretStr,
    http: BasetenHTTP,
    loops: LoopsModule,
) -> dict[str, Any]:
    settings = request.settings
    budget = _budget_guard(settings)
    epochs: list[EpochRecord] = []
    trainer: LoopsTrainer | None = None
    session_id: str | None = None
    outcome: LoopResult | None = None
    error: BaseException | None = None

    def on_session(value: str) -> None:
        nonlocal session_id
        session_id = value
        run_dir.update_state(baseten_session_id=value)

    def on_epoch(record: EpochRecord) -> None:
        epochs.append(record)
        best = _best(epochs)
        run_dir.write_epochs([epoch.as_dict() for epoch in epochs])
        run_dir.update_state(
            completed_epoch=record.epoch,
            last_resumable_state_uri=record.resumable_state_uri,
            best_epoch=best.epoch if best else None,
            best_sampler_weights_uri=best.sampler_weights_uri if best else None,
        )

    with _sigterm_raises():
        try:
            budget.start()
            run_dir.update_state(status="provisioning")
            output.info(
                f"provisioning a dedicated Baseten Loops trainer for {data.model.base_model}; "
                "this can take several minutes"
            )
            trainer = open_trainer(
                api_key,
                base_model=data.model.base_model,
                rank=data.lora_rank,
                seed=settings.seed,
                max_seq_len=data.max_seq_len,
                run_id=request.run_id,
                loops=loops,
                on_session=on_session,
            )
            run_dir.update_state(status="training", baseten_run_id=trainer.run_id)
            outcome = run_sft(
                trainer,
                data.train_rows,
                data.validation_batches,
                settings,
                token_budget=data.token_budget,
                run_id=request.run_id,
                budget=budget,
                on_epoch=on_epoch,
            )
        except BaseException as exc:
            error = exc
        with _interrupts_ignored():
            cleanup, cleanup_error = _cleanup(trainer, http, error)
            status = outcome.status if outcome is not None and error is None else "failed"
            best = _best(epochs)
            result = {
                "status": status,
                "provider": "baseten",
                "run_id": request.run_id,
                "baseten_session_id": session_id,
                "baseten_run_id": trainer.run_id if trainer else None,
                "model": data.model.alias,
                "base_model": data.model.base_model,
                "epochs": [epoch.as_dict() for epoch in epochs],
                "best_epoch": best.epoch if best else None,
                "best_sampler_weights_uri": best.sampler_weights_uri if best else None,
                "last_resumable_state_uri": epochs[-1].resumable_state_uri if epochs else None,
                "stopped_early": bool(outcome and outcome.stopped_early and error is None),
                "budget": {
                    **plan["budget"],
                    "elapsed_seconds": budget.elapsed_seconds,
                    "stopped": status == "budget_stopped",
                },
                "cleanup": cleanup,
            }
            run_dir.write_result(result)
            run_dir.update_state(
                status=status,
                cleanup_status="failed" if cleanup["error"] else "completed",
                error_type=type(error).__name__ if error else None,
            )
    output.info(f"training {status}; results in {run_dir.path}")
    if error is not None:
        if trainer is None:  # nothing trained, but this run directory is no longer empty
            error.add_note("rerun with a new --run-dir")
        _raise_with_cleanup(error, cleanup_error)
    if cleanup_error is not None:
        raise cleanup_error
    return result


def _raise_with_cleanup(error: BaseException, cleanup_error: ExternalError | None) -> NoReturn:
    """Raise the training ``error`` with the manual cleanup steps it or a failed cleanup left."""
    steps = list(getattr(error, "__notes__", []))
    if cleanup_error:
        steps.append(f"{cleanup_error.message}; {cleanup_error.hint}")
    if isinstance(error, TerminationRequested):
        hint = "; ".join(steps) or None
        raise FinetuneError("training stopped: the process received SIGTERM", hint=hint) from error
    if isinstance(error, FinetuneError):
        error.hint = "; ".join(filter(None, [error.hint, *steps])) or None
    elif steps:
        output.warn("; ".join(steps))
    raise error


def _cleanup(
    trainer: LoopsTrainer | None, http: BasetenHTTP, error: BaseException | None
) -> tuple[dict[str, Any], ExternalError | None]:
    """Close the trainer and deactivate its run. Never raises, so the training error survives."""
    if trainer is None:
        # A trainer that never became ready leaves its manual cleanup step as a note.
        left = "; ".join(getattr(error, "__notes__", [])) or None
        return {"client_closed": True, "run_deactivated": False, "error": left}, None
    output.info(
        f"deactivating Baseten run {trainer.run_id}; interrupts are ignored until it finishes"
    )
    closed = True
    try:
        trainer.close()
    except Exception as exc:
        closed = False
        output.warn(f"could not close the Loops trainer client ({type(exc).__name__})")
    try:
        _deactivate(http, trainer.run_id)
    except Exception as exc:
        reason = exc.message if isinstance(exc, FinetuneError) else type(exc).__name__
        error = ExternalError(
            f"could not deactivate Baseten run {trainer.run_id} ({reason})",
            source="baseten",
            hint="deactivate it in the Baseten UI, or run: curl -X POST "
            f"{API_BASE}/v1/loops/runs/{trainer.run_id}/deactivate "
            '-H "Authorization: Api-Key $BASETEN_API_KEY"',
        )
        return {"client_closed": closed, "run_deactivated": False, "error": error.message}, error
    return {"client_closed": closed, "run_deactivated": True, "error": None}, None


def _inactive(run: Mapping[str, Any] | None) -> bool:
    if run is None:
        return True
    status = run.get("status")
    name = status.get("name") if isinstance(status, dict) else status
    return str(name).upper() == "INACTIVE"


def _deactivate(http: BasetenHTTP, run_id: str) -> None:
    """Deactivate ``run_id`` unless it already is, then wait for Baseten to report it inactive."""
    with suppress(ExternalError):  # an unknown state still gets the deactivation
        if _inactive(http.get_run(run_id)):
            return
    for attempt in range(1, _DEACTIVATE_ATTEMPTS + 1):
        try:
            http.deactivate_run(run_id)
            break
        except ExternalError:
            if attempt == _DEACTIVATE_ATTEMPTS:
                raise
            _sleep(float(attempt))
    last_error: ExternalError | None = None
    for _ in range(_POLL_ATTEMPTS):
        _sleep(_POLL_INTERVAL_SECONDS)
        try:
            if _inactive(http.get_run(run_id)):
                return
        except ExternalError as exc:
            last_error = exc
    raise last_error or ExternalError(f"Baseten run {run_id} is still active", source="baseten")
