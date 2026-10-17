# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What ``train``, ``resume``, ``status`` and ``evaluate`` print: the decision, and where the
rest is.

The files on disk keep everything (``result.json``, ``epochs.json``, ``state.json``,
``summary.json``); a command's stdout result keeps what decides the next step, built in the
order it is read, and names those files under ``files``. Keys that are sometimes ``None``
stay (stable keys with explicit nulls); only whole sections are left out: the checkpoints,
which stay in ``result.json``, the per-target detail of an evaluation, which stays in
``summary.json``, and a ``worker`` that does not exist.

:data:`TABLE_VIEWS` holds the jq expression each list-shaped command's rows are picked with
when the user asks pup for a table or CSV (see :func:`pup_finetune.output.table_filter`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pup_finetune.display import display_path
from pup_finetune.evaluation.runner import RecordedEvaluation, headline
from pup_finetune.runs import RESUMABLE_PHASES, RunDir
from pup_finetune.training.worker import WorkerStatus

__all__ = [
    "TABLE_VIEWS",
    "evaluation_view",
    "files",
    "status_view",
    "train_view",
    "worker_view",
]

TABLE_VIEWS: Final[dict[str, str]] = {
    "models list": (
        ".models | map({alias, provider, base_model, max_seq_len, default_lora_rank, "
        "supports_reasoning})"
    ),
    "auth status": ".providers",
    "runs": (
        ".runs | map({run_id, phase, model, provider, best_epoch, promoted_model, deployment, "
        "run_dir})"
    ),
    "doctor": ".checks | map({name, status, detail, needed_for})",
    "dataset sample": (
        '.rows | map({id, split, label, turns, tools: (.tools_used | join(", ")), '
        "first_user_message})"
    ),
}
"""Each list-shaped command's table view, by its path below the program: the jq expression
that picks its rows (``pup --output table finetune models list`` shows one row per model)."""


def files(**paths: str | Path | None) -> dict[str, str | None]:
    """The ``files`` section: each file's path as results show it (``None`` stays)."""
    return {name: None if path is None else display_path(path) for name, path in paths.items()}


def _seconds(value: object) -> int | None:
    """Whole seconds, or ``None`` for a value that is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return round(value)


def model_view(value: object) -> dict[str, Any] | None:
    """A promoted model as results show it: its id and the provider's name for it."""
    if not isinstance(value, Mapping):
        return None
    return {"model_id": value.get("model_id"), "uri": value.get("uri")}


def checkpoint_name(value: object) -> str | None:
    """A checkpoint's name (its record, URI included, stays in ``result.json``)."""
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    return str(name) if name is not None else None


def _epochs(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each finished epoch's losses; the best epoch says so with ``best: true``."""
    best = result.get("best_epoch")
    shown: list[dict[str, Any]] = []
    for epoch in result.get("epochs") or []:
        if not isinstance(epoch, Mapping):
            continue
        entry = {
            "epoch": epoch.get("epoch"),
            "train_loss": epoch.get("train_loss"),
            "val_loss": epoch.get("val_loss"),
        }
        if best is not None and epoch.get("epoch") == best:
            entry["best"] = True
        shown.append(entry)
    return shown


def train_view(
    run: RunDir,
    result: Mapping[str, Any],
    next_steps: Mapping[str, Any],
    *,
    evaluated: bool = False,
    evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What a foreground ``train`` or ``resume`` prints for ``result`` (``result.json``).

    ``status`` is how the run ended (its phase); ``next_steps`` holds ``next_command`` and
    ``next_steps``. When the run was ``evaluated`` (``train --evaluate``), ``evaluation`` is
    the :func:`evaluation_view` of its outcome (``None`` when it never ran).
    """
    manifest = run.manifest()
    view: dict[str, Any] = {
        "status": result.get("phase"),
        "run_id": result.get("run_id", manifest.get("run_id")),
        "run_dir": display_path(run.path),
        "model": (manifest.get("profile") or {}).get("alias"),
        "provider": manifest.get("provider"),
        "best_epoch": result.get("best_epoch"),
        "best_val_loss": result.get("best_val_loss"),
        "stopped_reason": result.get("stopped_reason"),
        "epochs": _epochs(result),
        "promoted_model": model_view(result.get("promoted_model")),
        "deployable_checkpoint": checkpoint_name(result.get("deployable_checkpoint")),
        "elapsed_seconds": _seconds(result.get("elapsed_seconds")),
        "error": result.get("error"),
        "finalize_error": result.get("finalize_error"),
    }
    if evaluated:
        view["evaluation"] = None if evaluation is None else evaluation_view(evaluation)
    view["warnings"] = list(result.get("warnings") or [])
    view["next_command"] = next_steps.get("next_command")
    view["next_steps"] = list(next_steps.get("next_steps") or [])
    view["files"] = files(result=run.result_path, epochs=run.epochs_path)
    return view


# --- status ------------------------------------------------------------------------------


def worker_state(worker: WorkerStatus) -> str:
    """``running``, ``starting``, ``stale`` (alive, but its heartbeat stopped or it hangs
    before training) or ``exited``."""
    if worker.starting:
        return "starting"
    if worker.running:
        return "running"
    if worker.stale and worker.alive:
        return "stale"
    return "exited"


def worker_view(worker: WorkerStatus, phase: str) -> dict[str, Any] | None:
    """The process training the run, or ``None`` when there is none to speak of: no worker
    was started for it, it is not training and no process is starting or running."""
    state = worker_state(worker)
    if worker.log is None and phase != "training" and state == "exited":
        return None
    return {
        "pid": worker.pid,
        "state": state,
        "heartbeat_age_s": worker.heartbeat_age_s,
        "log": None if worker.log is None else display_path(worker.log),
    }


STATUS_KEYS: Final = (
    "phase",
    "run_id",
    "run_dir",
    "run_dir_source",
    "provider",
    "model",
    "epoch",
    "max_epochs",
    "step",
    "max_steps",
    "last_train_loss",
    "val_losses",
    "best_epoch",
    "best_val_loss",
    "elapsed_seconds",
    "approx_eta_seconds",
    "estimated_spend_usd",
    "stopped_reason",
    "promoted_model",
    "deployable_checkpoint",
    "evaluation",
    "error",
    "finalize_error",
    "resume_mode",
    "worker",
    "remote",
    "warnings",
    "watch_timed_out",
    "settings",
    "next_command",
    "next_steps",
    "files",
)
"""What ``status`` prints, in order; ``estimated_spend_usd`` (with an hourly rate),
``evaluation`` (the run was evaluated), ``resume_mode`` (a resumable run), ``worker`` (one
exists, and no remote job trains the run), ``remote`` (the job of a run started with
``train --remote``), ``watch_timed_out``
(``--watch``) and ``settings`` (what the run trains with that differs from the defaults,
:func:`changed_settings`; every setting is in ``run.json``, ``files.run``) only when they
apply."""


FIXED_SETTINGS: Final = frozenset({"optimizer", "pipeline_depth"})
"""Recorded settings no flag changes, which ``status`` leaves to ``run.json``."""


def changed_settings(settings: object, defaults_used: object) -> dict[str, Any] | None:
    """The recorded ``settings`` of a run that differ from the defaults, as ``status`` shows
    them: those ``defaults_used`` does not name (``lora.rank`` naming the LoRA rank and its
    source), without what no flag changes; ``run.json`` (``files.run``) keeps all of them.
    ``None`` when every setting is a default, or the run recorded none."""
    if not isinstance(settings, Mapping):
        return None
    defaults = {str(name) for name in defaults_used} if isinstance(defaults_used, list) else set()
    shown: dict[str, Any] = {}
    for key, value in settings.items():
        if key in FIXED_SETTINGS or key in defaults:
            continue
        if key == "lora" and isinstance(value, Mapping):
            lora = {
                name: item
                for name, item in value.items()
                if f"lora.{name.removesuffix('_source')}" not in defaults
            }
            if lora:
                shown[key] = lora
        elif (key == "max_seq_len_source" and "max_seq_len" in defaults) or (
            key == "provider_options" and not value
        ):
            continue
        else:
            shown[key] = value
    return shown or None


def status_view(data: Mapping[str, Any]) -> dict[str, Any]:
    """What ``status`` prints from what it measured (see ``run_status``)."""
    phase = data["phase"]
    worker = data["worker"]
    shown: dict[str, Any] = {
        **data,
        "promoted_model": model_view(data["promoted_model"]),
        "deployable_checkpoint": checkpoint_name(data["deployable_checkpoint"]),
        "worker": worker_view(worker, phase) if isinstance(worker, WorkerStatus) else worker,
    }
    if phase not in RESUMABLE_PHASES:
        shown.pop("resume_mode", None)
    remote = data.get("remote")
    if isinstance(remote, Mapping) and remote.get("state") != "ended":
        shown["worker"] = None
    if shown["worker"] is None:
        shown.pop("worker")
    if shown.get("remote") is None:
        shown.pop("remote", None)
    if shown.get("evaluation") is None:
        shown.pop("evaluation", None)
    if shown.get("settings") is None:  # a run recorded before run.json had them
        shown.pop("settings", None)
    return {key: shown[key] for key in STATUS_KEYS if key in shown}


def recorded_evaluation_view(found: RecordedEvaluation | None) -> dict[str, Any] | None:
    """A run's recorded evaluation as ``status`` shows it: how it stands, its headline and
    where it is (``None`` when the run was never evaluated)."""
    if found is None:
        return None
    from pup_finetune.evaluation.replay import SUMMARY_FILE

    summary = found.out_dir / SUMMARY_FILE
    return {
        "status": found.status,
        "complete": found.complete,
        "out_dir": display_path(found.out_dir),
        "judge_model": found.judge_model,
        "headline": found.headline,
        "publication": found.publication,
        "files": files(summary=summary if summary.exists() else None),
    }


# --- evaluate ----------------------------------------------------------------------------


def evaluation_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """What an evaluation prints: its headline, each metric's rate per target, the errors,
    the publication and where ``summary.json`` and ``results.jsonl`` are.

    ``tuned_checkpoint`` names the sampler checkpoint the tuned target sampled (``None``
    when nothing was generated), which no file records. ``reused`` lists the targets that
    took answers or verdicts from the dataset's store, with how many.

    Only a replayed evaluation (one with a ``summary``) is slimmed; a calibration failure,
    or an evaluation ``train --evaluate`` skipped or that failed, prints as it is.
    """
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return dict(result)
    from pup_finetune.evaluation.metrics import MetricName
    from pup_finetune.evaluation.replay import RESULTS_FILE, SUMMARY_FILE

    targets: Mapping[str, Any] = summary.get("targets") or {}
    found = {name for target in targets.values() for name in (target.get("metrics") or {})}
    # In the metrics' own order (whether it calls a tool first), then any others.
    names = [str(name) for name in MetricName if name in found]
    names += sorted(found - set(names))
    out_dir = Path(str(result["out_dir"]))
    return {
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "judge_model": result.get("judge_model"),
        "compare_model": result.get("compare_model"),
        "headline": headline(summary),
        "metrics": {
            name: {
                target: (((targets.get(target) or {}).get("metrics") or {}).get(name) or {}).get(
                    "rate"
                )
                for target in targets
            }
            for name in names
        },
        "errors": {name: int((summary.get("errors") or {}).get(name, 0)) for name in targets},
        "pending": result.get("pending"),
        "reused": {
            name: dict(target["reused"])
            for name, target in targets.items()
            if any((target.get("reused") or {}).values())
        },
        "calibration": result.get("calibration"),
        "tuned_checkpoint": checkpoint_name(result.get("tuned_checkpoint")),
        "publication": result.get("publication"),
        "caveat": summary.get("caveat"),
        "warnings": list(result.get("warnings") or []),
        "hint": result.get("hint"),
        "next_command": result.get("next_command"),
        "next_steps": list(result.get("next_steps") or []),
        "files": files(summary=out_dir / SUMMARY_FILE, results=out_dir / RESULTS_FILE),
    }
