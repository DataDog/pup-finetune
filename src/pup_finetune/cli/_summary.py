# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The closing summary a command prints on stderr after its result.

Each summary is generated from the result mapping the command emitted, never from state of
its own, so it cannot say something the JSON does not. On a terminal it is a few short lines
(:attr:`Summary.lines`); off a terminal, where an agent pays for every line, it is exactly one
plain line (:attr:`Summary.line`). ``PUP_FINETUNE_QUIET=1`` turns it off.

:data:`SUMMARIES` maps a command's path (``train``, ``dataset pull``) to its summarizer;
:func:`summarize` is what :func:`~pup_finetune.cli._common.run_command` calls. A command
without a summarizer prints none.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime

__all__ = [
    "MAX_LINES",
    "SUMMARIES",
    "Summary",
    "format_duration",
    "status_line",
    "summarize",
]

MAX_LINES: Final = 8
"""The most lines a summary prints on a terminal."""

TOP_CODES: Final = 3
"""How many exclusion codes a summary names before ``+N more``."""

CAVEAT: Final = "agreement with recorded behavior, not task success"
"""What every evaluation summary says its numbers measure."""


@dataclass(frozen=True, slots=True)
class Summary:
    """A closing summary: ``lines`` on a terminal, ``line`` off one."""

    lines: tuple[str, ...]
    line: str


Summarizer = Callable[[Mapping[str, Any]], Summary | None]


def _summary(
    headline: str,
    details: Sequence[str] = (),
    *,
    brief: Sequence[str] = (),
    next_command: object = None,
) -> Summary:
    """A summary of ``headline`` and ``details`` (a terminal) or ``headline`` and ``brief``
    (one line), each ending with ``next_command`` when there is one."""
    lines = [headline, *details][: MAX_LINES - 1]
    line = "; ".join([headline, *brief])
    if isinstance(next_command, str) and next_command:
        lines.append(f"next: {next_command}")
        line += f". Next: {next_command}"
    return Summary(tuple(lines), line)


def summarize(command: str | None, result: Mapping[str, Any]) -> Summary | None:
    """The closing summary of ``command``'s ``result``, or ``None`` when it has none.

    A summarizer that fails on an unexpected result yields none: the result itself was
    printed already. With ``PUP_FINETUNE_DEBUG=1`` its redacted traceback goes to stderr, so
    a summary that went missing can be traced.
    """
    summarizer = SUMMARIES.get(command or "")
    if summarizer is None:
        return None
    try:
        return summarizer(result)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        if Runtime.from_env().debug:
            sys.stderr.write(redact("".join(traceback.format_exception(exc))))
        return None


# --- formatting --------------------------------------------------------------------------


def _n(count: object, noun: str, plural: str | None = None) -> str:
    """``1 row``, ``3 rows``."""
    value = int(count) if isinstance(count, int | float) else 0
    return f"{value} {noun if value == 1 else plural or noun + 's'}"


def _loss(value: object) -> str | None:
    return f"{value:#.3g}" if isinstance(value, int | float) else None


def _percent(rate: object) -> str:
    return f"{round(rate * 100)}%" if isinstance(rate, int | float) else "n/a"


def _points(value: float) -> str:
    """A difference of rates in percentage points: ``+25``, ``-5``, ``0``."""
    points = round(value * 100)
    return f"{points:+d}" if points else "0"


def _error_text(error: object) -> str | None:
    """The message of a run file's error (``{"type", "message"}``) or of a plain string."""
    if isinstance(error, Mapping):
        message = error.get("message")
        kind = error.get("type")
        return str(message or kind) if message or kind else None
    return str(error) if error else None


def format_duration(seconds: float, *, rough: bool = False) -> str:
    """``45s``, ``4m12s`` or ``2h05m``; ``rough`` drops the seconds past a minute (``4m``)."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m" if rough else f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _codes(by_code: object) -> str:
    """``1 span_error, 1 non_text_content`` for the most frequent codes, ``+N more``."""
    if not isinstance(by_code, Mapping) or not by_code:
        return ""
    ranked = sorted(by_code.items(), key=lambda item: (-int(item[1]), str(item[0])))
    shown = [f"{count} {code}" for code, count in ranked[:TOP_CODES]]
    if len(ranked) > TOP_CODES:
        shown.append(f"+{len(ranked) - TOP_CODES} more")
    return ", ".join(shown)


def _splits(splits: object) -> str | None:
    if not isinstance(splits, Mapping):
        return None
    return ", ".join(f"{name} {splits.get(name, 0)}" for name in ("train", "validation", "test"))


# --- status ------------------------------------------------------------------------------


def status_line(data: Mapping[str, Any]) -> str:
    """One status line, e.g. ``training: epoch 3/5 step 21/35 loss 1.10 val 1.35 (best e2)
    4m12s elapsed, eta ~6m``; parts that are not known yet are left out."""
    parts = [f"{data['phase']}: epoch {data['epoch']}/{data['max_epochs']}"]
    step, max_steps, per_epoch = data["step"], data["max_steps"], data["steps_per_epoch"]
    parts.append(f"step {step}/{max_steps}" if max_steps else f"step {step}")
    if (loss := _loss(data["last_train_loss"])) is not None:
        parts.append(f"loss {loss}")
    val_losses = data["val_losses"]
    if val_losses and (val := _loss(val_losses[-1])) is not None:
        parts.append(f"val {val}")
    if (
        data["phase"] == "training"
        and per_epoch
        and step == data["epoch"] * per_epoch
        and data["epochs_completed"] < data["epoch"]
    ):
        parts.append("validating")  # the epoch's last step is done; its end is not recorded
    if data["best_epoch"] is not None:
        parts.append(f"(best e{data['best_epoch']})")
    line = " ".join(parts)
    if data["elapsed_seconds"]:
        line += f" {format_duration(data['elapsed_seconds'])} elapsed"
        if data["approx_eta_seconds"] is not None:
            line += f", eta ~{format_duration(data['approx_eta_seconds'], rough=True)}"
    worker = data["worker"]
    if data["phase"] == "training" and not worker["running"]:
        pid = f"pid {worker['pid']}" if worker["pid"] is not None else "its process"
        line += f" ({pid} not running)"
    return line


def _model_line(result: Mapping[str, Any]) -> str | None:
    """What the run has to deploy, or why it has nothing."""
    promoted = result.get("promoted_model")
    if isinstance(promoted, Mapping) and promoted.get("model_id"):
        return f"promoted model: {promoted['model_id']} ({promoted.get('provider')})"
    checkpoint = result.get("deployable_checkpoint")
    if isinstance(checkpoint, Mapping) and checkpoint.get("name"):
        return f"deployable checkpoint: {checkpoint['name']}"
    if (finalize := _error_text(result.get("finalize_error"))) is not None:
        return f"not promoted: {finalize}"
    return None


def status_summary(result: Mapping[str, Any]) -> Summary:
    """``status``: the status line, and on a terminal the run's model."""
    details = [line] if (line := _model_line(result)) is not None else []
    if (error := _error_text(result.get("error"))) is not None:
        details.append(f"error: {error}")
    return _summary(status_line(result), details, next_command=result.get("next_command"))


# --- train and resume --------------------------------------------------------------------

_PHASES: Final = {
    "completed": "completed",
    "cancelled": "was cancelled",
    "interrupted": "was interrupted",
    "failed": "failed",
}
_STOPS: Final = {
    "early_stopped": " (stopped early)",
    "budget_exhausted": " (stopped at its budget)",
}


def _best(result: Mapping[str, Any]) -> str:
    epochs = [epoch for epoch in result.get("epochs") or [] if isinstance(epoch, Mapping)]
    best, best_val = result.get("best_epoch"), _loss(result.get("best_val_loss"))
    if best is None:
        return "no epoch finished"
    if best_val is None:
        return f"best epoch {best} (by training loss)"
    first = _loss(epochs[0].get("val_loss")) if epochs else None
    if first is not None and len(epochs) > 1 and first != best_val:
        return f"best epoch {best} (validation loss {first} -> {best_val})"
    return f"best epoch {best} (validation loss {best_val})"


def train_summary(result: Mapping[str, Any]) -> Summary | None:
    """``train`` and ``resume``: how the run ended, its best epoch, its model and next step;
    with ``--evaluate``, the evaluation too."""
    status = result.get("status")
    if status == "started":
        return _summary(
            f"run {result['run_id']} started in a background worker (pid {result['pid']})",
            [f"log: {result['worker_log']}"],
            next_command=result.get("next_command"),
        )
    if status == "calibration_failed":
        return evaluate_summary(result)
    phase = result.get("phase")
    if not isinstance(phase, str):
        return None
    epochs = len(result.get("epochs") or [])
    elapsed = result.get("elapsed_seconds")
    took = f" in {format_duration(elapsed)}" if isinstance(elapsed, int | float) else ""
    headline = (
        f"run {result.get('run_id')} {_PHASES.get(phase, phase)}{took}: "
        f"{_n(epochs, 'epoch')}{_STOPS.get(str(result.get('stopped_reason')), '')}, "
        f"{_best(result)}"
    )
    details: list[str] = []
    brief: list[str] = []
    if (model := _model_line(result)) is not None:
        details.append(model)
        brief.append(model)
    if (error := _error_text(result.get("error"))) is not None:
        details.append(f"error: {error}")
        brief.append(f"error: {error}")
    evaluation = result.get("evaluation")
    if isinstance(evaluation, Mapping):
        found = _evaluation_parts(evaluation, compact=True)
        details += found.lines
        brief.append(found.line)
    return _summary(headline, details, brief=brief, next_command=result.get("next_command"))


# --- evaluate ----------------------------------------------------------------------------


def _calibration(result: Mapping[str, Any]) -> str:
    calibration = result.get("calibration")
    if not isinstance(calibration, Mapping):
        return "calibration skipped"
    return "calibration passed" if calibration.get("ok") else "calibration failed"


def _agreement(target: Mapping[str, Any], *, cell: bool = False) -> str:
    """A target's judge pass rate: ``1/4 (25%)``, or ``1/4 25%`` as a table ``cell``."""
    rate = target.get("judge_pass") or {}
    percent = _percent(rate.get("rate"))
    return f"{rate.get('passed', 0)}/{rate.get('n', 0)} " + (percent if cell else f"({percent})")


def _difference(summary: Mapping[str, Any]) -> str:
    difference = summary.get("tuned_minus_base") or {}
    rate = difference.get("judge_pass_rate")
    if not isinstance(rate, int | float):
        return "n/a"
    text = f"{_points(rate)} pts"
    ci = difference.get("ci95")
    if isinstance(ci, Sequence) and len(ci) == 2:
        text += f" (95% CI {_points(ci[0])} to {_points(ci[1])})"
    return text


def _paired(summary: Mapping[str, Any]) -> str:
    paired = summary.get("paired") or {}
    errors = int(paired.get("both_error", 0)) + int(paired.get("one_error", 0))
    return (
        f"paired: {_n(paired.get('wins', 0), 'win')}, "
        f"{_n(paired.get('regressions', 0), 'regression')}, "
        f"{_n(paired.get('ties', 0), 'tie')}, {_n(errors, 'error')}"
    )


def _publication(result: Mapping[str, Any]) -> str | None:
    publication = result.get("publication")
    if not isinstance(publication, Mapping):
        return None
    status = publication.get("status")
    if status == "published":
        experiments = publication.get("experiments") or {}
        names = [
            str(entry.get("name"))
            for entry in experiments.values()
            if isinstance(entry, Mapping) and entry.get("name")
        ]
        return "published as experiments " + " and ".join(names) if names else "published"
    return {
        "skipped": "not published (see the warning above)",
        "disabled": "not published (publishing was turned off)",
        "failed": "publishing failed; next publishes the rest",
        "cancelled": "publishing was cancelled; next publishes the rest",
        "pending": "not published yet: it is published once every case has a result",
    }.get(str(status))


def _evaluation_parts(result: Mapping[str, Any], *, compact: bool) -> Summary:
    """An evaluation's lines and its one-line form, without the next command; ``compact``
    (inside ``train --evaluate``) leaves out the table."""
    status = result.get("status")
    if status == "calibration_failed":
        calibration = result.get("calibration") or {}
        checks = [check for check in calibration.get("checks") or [] if isinstance(check, dict)]
        wrong = sum(1 for check in checks if not check.get("ok"))
        headline = (
            f"calibration failed: the judge {result.get('judge_model')} got {wrong} of "
            f"{len(checks)} checks wrong, so nothing was evaluated"
        )
        # The hint spells the flags of the command that ran it (evaluate or train).
        hint = result.get("hint")
        if not isinstance(hint, str) or not hint:
            return Summary((headline,), headline)
        return Summary((headline, hint), f"{headline}; {hint}")
    if status in {"skipped", "failed"}:
        why = result.get("reason") or _error_text(result.get("error")) or ""
        headline = f"evaluation {status}" + (f": {why}" if why else "")
        return Summary((headline,), headline)
    summary = result["summary"]
    targets = summary["targets"]
    errors = int(result.get("errors") or 0)
    what = {
        "complete": "complete",
        "complete_with_errors": f"complete with {_n(errors, 'error')}",
        "partial": f"partial ({_n(result.get('pending', 0), 'pair')} left)",
    }.get(str(status), str(status))
    judge = result.get("judge_model")
    headline = (
        f"evaluation {what}: {_n(summary.get('n_cases'), 'case')} from "
        f"{_n(summary.get('n_trajectories'), 'test conversation')}"
        + (f"; judge {judge}" if judge else "")
        + f", {_calibration(result)}"
    )
    base, tuned, difference = (
        _agreement(targets["base"]),
        _agreement(targets["tuned"]),
        _difference(summary),
    )
    paired = _paired(summary)
    publication = _publication(result)
    agreement = f"judge agreement: base {base}, tuned {tuned}, tuned - base {difference}"
    if compact:
        lines = [headline, agreement, paired, CAVEAT]
    else:
        lines = [
            headline,
            f"{'':<21}{'base':<11}{'tuned':<11}tuned - base",
            f"{'judge agreement':<21}{_agreement(targets['base'], cell=True):<11}"
            f"{_agreement(targets['tuned'], cell=True):<11}{difference}",
            paired,
            CAVEAT,
        ]
    parts = [headline, agreement, paired.removeprefix("paired: "), CAVEAT]
    if publication is not None:
        lines.append(publication)
        parts.append(publication)
    return Summary(tuple(lines), "; ".join(parts))


def evaluate_summary(result: Mapping[str, Any]) -> Summary | None:
    """``evaluate``: the cases, judge and calibration, base against tuned with the CI, the
    paired verdicts, the caveat and the publication."""
    if "status" not in result:
        return None
    parts = _evaluation_parts(result, compact=False)
    headline, *details = parts.lines
    summary = _summary(headline, details, next_command=result.get("next_command"))
    line = parts.line + (f". Next: {result['next_command']}" if result.get("next_command") else "")
    return Summary(summary.lines, line)


# --- dataset -----------------------------------------------------------------------------


def _read(spans: object, pages: object) -> str:
    text = f"read {_n(spans, 'span')}"
    if isinstance(pages, int) and pages:
        text += f" on {_n(pages, 'page')}"
    return text


def _accepted(accepted: object, excluded: object, by_code: object) -> str:
    text = f"{_n(accepted, 'row')} accepted, {excluded or 0} excluded"
    codes = _codes(by_code)
    return f"{text} ({codes})" if codes else text


def preview_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset preview``: spans read, rows accepted and excluded by code; nothing written."""
    stats = result.get("stats") or {}
    pages = int(stats.get("pages_fetched", 0)) + int(stats.get("pages_cached", 0))
    headline = (
        f"{_read(result.get('matched_spans'), pages)}: "
        f"{_accepted(result.get('accepted'), result.get('excluded'), result.get('excluded_by_code'))}"
        " (nothing written)"
    )
    splits = _splits(result.get("splits"))
    details = [f"splits: {splits}"] if splits else []
    return _summary(headline, details, next_command=result.get("next_command"))


def pull_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset pull``: spans read, rows added, and the dataset's rows by code and split."""
    pull = result.get("pull") or {}
    pages = int(pull.get("pages_fetched", 0)) + int(pull.get("pages_cached", 0))
    headline = (
        f"{_read(pull.get('matched_spans'), pages)}: "
        f"{_accepted(result.get('accepted'), result.get('excluded'), result.get('excluded_by_code'))}"
    )
    added = f"{_n(pull.get('added', 0), 'row')} added, {pull.get('unchanged', 0)} unchanged"
    details = [added]
    if splits := _splits(result.get("splits")):
        details.append(f"splits: {splits}")
    return _summary(headline, details, brief=[added], next_command=result.get("next_command"))


def import_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset import``: rows accepted and excluded by code, added and split."""
    imported = result.get("import") or {}
    headline = (
        f"imported {imported.get('file')}: "
        f"{_accepted(result.get('accepted'), result.get('excluded'), result.get('excluded_by_code'))}"
    )
    added = f"{_n(imported.get('added', 0), 'row')} added, {imported.get('unchanged', 0)} unchanged"
    details = [added]
    if splits := _splits(result.get("splits")):
        details.append(f"splits: {splits}")
    return _summary(headline, details, brief=[added], next_command=result.get("next_command"))


def triage_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset triage``: kept, dropped, undecided and too long, or with ``--agreement`` how
    often the council agreed with the labels (and Cohen's kappa)."""
    calls = (
        f"{_n(result.get('calls_made', 0), 'judge call')}, {result.get('calls_failed', 0)} failed"
    )
    status = result.get("status")
    if result.get("mode") == "agreement":
        agreement = result.get("agreement") or {}
        kappa = agreement.get("kappa")
        headline = (
            f"agreement {status}: the council agreed with your labels on "
            f"{agreement.get('rows_agreed', 0)}/{agreement.get('rows_compared', 0)} rows "
            f"({_percent(agreement.get('rate'))}), kappa "
            + (f"{kappa:.2f}" if isinstance(kappa, int | float) else "n/a")
        )
        apart = (
            f"{_n(agreement.get('disagreements', 0), 'disagreement')}, "
            f"{agreement.get('rows_undecided', 0)} undecided, "
            f"{agreement.get('rows_too_long', 0)} too long"
        )
        return _summary(
            headline, [apart, calls], brief=[apart], next_command=result.get("next_command")
        )
    headline = (
        f"triage {status}: {result.get('kept', 0)} kept, {result.get('dropped', 0)} dropped, "
        f"{result.get('undecided', 0)} undecided, {result.get('too_long', 0)} too long"
    )
    details = [calls]
    reasons = [
        f"{entry.get('reason')} ({entry.get('count')})"
        for entry in result.get("top_drop_reasons") or []
        if isinstance(entry, Mapping)
    ]
    if reasons:
        details.append(f"top drop reasons: {'; '.join(reasons[:TOP_CODES])}")
    if stale := result.get("stale_judge_labels"):
        details.append(f"{_n(stale, 'judge label')} the votes no longer back")
    return _summary(headline, details, brief=[calls], next_command=result.get("next_command"))


def push_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset push``: what was sent, to which dataset, and what it holds now."""
    where = f"dataset {result.get('dataset_name')!r} in project {result.get('project')!r}"
    pushed = int(result.get("pushed") or 0)
    headline = (
        f"pushed {_n(pushed, 'record')} to {where}" if pushed else f"nothing new to push to {where}"
    )
    holds = f"it holds {_n(result.get('records', 0), 'record')} from this directory"
    if (version := result.get("dataset_version")) is not None:
        holds += f" (version {version})"
    return _summary(headline, [holds], brief=[holds], next_command=result.get("next_command"))


# --- deploy and undeploy -----------------------------------------------------------------


def deploy_summary(result: Mapping[str, Any]) -> Summary:
    """``deploy``: what serves, how to call it, and how to stop paying for it."""
    smoke = result.get("smoke_test") or {}
    state = "ready" if smoke.get("ok") else f"{result.get('state')}, smoke test failed"
    model, deployment = result.get("model_id"), result.get("deployment_id")
    named = f"{model}" if model == deployment else f"{model} as {deployment}"
    headline = f"deployed {named} ({state})"
    details: list[str] = []
    brief: list[str] = []
    usage = result.get("usage")
    if isinstance(usage, Mapping):
        call = f"call it: model {usage.get('model')} at {usage.get('base_url')}"
        details += [call, "example request: usage.example in the result"]
        brief.append(call)
    if undeploy := result.get("undeploy_command"):
        stop = f"it bills while a replica runs; delete it with: {undeploy}"
        details.append(stop)
        brief.append(stop)
    return _summary(headline, details, brief=brief, next_command=result.get("next_command"))


def undeploy_summary(result: Mapping[str, Any]) -> Summary:
    """``undeploy``: the deployment deleted and the model kept."""
    headline = f"deleted the deployment {result.get('deployment_id')}"
    kept = f"the model {result.get('model_kept') or result.get('model')} is kept"
    return _summary(headline, [kept], brief=[kept], next_command=result.get("next_command"))


def doctor_summary(result: Mapping[str, Any]) -> Summary:
    """``doctor``: how many checks passed, and which required ones failed."""
    checks = [check for check in result.get("checks") or [] if isinstance(check, Mapping)]
    passed = sum(1 for check in checks if check.get("ok"))
    failed = [
        f"{check.get('name')} ({check.get('detail')})"
        for check in checks
        if not check.get("ok") and check.get("required", True)
    ]
    line = f"doctor: {passed}/{len(checks)} checks passed"
    if failed:
        line += f"; failed: {'; '.join(failed)}"
    return Summary((line,), line)


SUMMARIES: Final[dict[str, Summarizer]] = {
    "train": train_summary,
    "resume": train_summary,
    "status": status_summary,
    "evaluate": evaluate_summary,
    "dataset preview": preview_summary,
    "dataset pull": pull_summary,
    "dataset import": import_summary,
    "dataset triage": triage_summary,
    "dataset push": push_summary,
    "deploy": deploy_summary,
    "undeploy": undeploy_summary,
    "doctor": doctor_summary,
}
"""Each command's summarizer, by its path below the program."""
