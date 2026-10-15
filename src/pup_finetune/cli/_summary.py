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

from pup_finetune.display import loss_text
from pup_finetune.redaction import redact
from pup_finetune.runtime import Runtime

__all__ = [
    "CARD_LINES",
    "MAX_LINES",
    "PREVIEW_CARDS",
    "SUMMARIES",
    "Summary",
    "format_duration",
    "preview_card",
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
    """A closing summary: ``lines`` on a terminal, ``line`` off one; ``lines`` either way
    when ``full`` (a checklist, which is the point of its command)."""

    lines: tuple[str, ...]
    line: str
    full: bool = False


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

    A summarizer that fails, however, yields none: the result itself was printed already.
    With ``PUP_FINETUNE_DEBUG=1`` its redacted traceback goes to stderr, so a summary that
    went missing can be traced.
    """
    summarizer = SUMMARIES.get(command or "")
    if summarizer is None:
        return None
    try:
        return summarizer(result)
    except Exception as exc:
        _debug_traceback(exc)
        return None


def _debug_traceback(exc: BaseException) -> None:
    if Runtime.from_env().debug:
        sys.stderr.write(redact("".join(traceback.format_exception(exc))))


# --- formatting --------------------------------------------------------------------------


def _n(count: object, noun: str, plural: str | None = None) -> str:
    """``1 row``, ``3 rows``."""
    value = int(count) if isinstance(count, int | float) else 0
    return f"{value} {noun if value == 1 else plural or noun + 's'}"


def _loss(value: object) -> str | None:
    return loss_text(value)


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

STOPPED_PHASES: Final = frozenset({"interrupted", "failed"})
"""The phases a run is in after an error stopped it."""


def status_line(
    data: Mapping[str, Any],
    *,
    steps_per_epoch: int | None = None,
    epochs_completed: int | None = None,
) -> str:
    """One status line for what ``status`` prints, e.g. ``training: epoch 3/5 step 21/35
    loss 1.100 val 1.350 (best epoch 2) 4m12s elapsed, eta ~6m``; parts that are not known yet are
    left out. With ``steps_per_epoch`` and ``epochs_completed`` (a watch knows them), an
    epoch whose last step is done but whose end is not recorded yet reads ``validating``."""
    phase = data["phase"]
    error = data.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    stopped = phase in STOPPED_PHASES
    # A run that stopped on an error says which, and that its step is the last one done.
    shown = f"{phase} ({code})" if stopped and code else str(phase)
    parts = [f"{shown}: epoch {data['epoch']}/{data['max_epochs']}"]
    step, max_steps = data["step"], data["max_steps"]
    counted = f"step {step}/{max_steps}" if max_steps else f"step {step}"
    parts.append(f"after {counted}" if stopped else counted)
    if (loss := _loss(data["last_train_loss"])) is not None:
        parts.append(f"loss {loss}")
    val_losses = data["val_losses"]
    if val_losses and (val := _loss(val_losses[-1])) is not None:
        parts.append(f"val {val}")
    if (
        data["phase"] == "training"
        and steps_per_epoch
        and epochs_completed is not None
        and step == data["epoch"] * steps_per_epoch
        and epochs_completed < data["epoch"]
    ):
        parts.append("validating")  # the epoch's last step is done; its end is not recorded
    if data["best_epoch"] is not None:
        parts.append(f"(best epoch {data['best_epoch']})")
    line = " ".join(parts)
    if data["elapsed_seconds"]:
        line += f" {format_duration(data['elapsed_seconds'])} elapsed"
        if data["approx_eta_seconds"] is not None:
            line += f", eta ~{format_duration(data['approx_eta_seconds'], rough=True)}"
    remote = data.get("remote")
    if isinstance(remote, Mapping) and remote.get("state") != "ended":
        return f"{line} ({_kind(remote.get('backend'))} {remote.get('state')})"
    worker = data.get("worker")
    state = worker.get("state") if isinstance(worker, Mapping) else None
    if data["phase"] == "training" and state not in {"running", "starting"}:
        pid = worker.get("pid") if isinstance(worker, Mapping) else None
        line += f" ({f'pid {pid}' if pid is not None else 'its process'} not running)"
    return line


def _kind(backend: object) -> str:
    """A remote job's kind as text says it: ``Baseten job`` for ``baseten job``."""
    text = str(backend or "remote job")
    return text[:1].upper() + text[1:]


def _model_line(result: Mapping[str, Any]) -> str | None:
    """What the run has to deploy, or why it has nothing."""
    promoted = result.get("promoted_model")
    if isinstance(promoted, Mapping) and promoted.get("model_id"):
        provider = result.get("provider")
        return f"promoted model: {promoted['model_id']}" + (f" ({provider})" if provider else "")
    checkpoint = result.get("deployable_checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        return f"deployable checkpoint: {checkpoint}"
    if (finalize := _error_text(result.get("finalize_error"))) is not None:
        return f"not promoted: {finalize}"
    return None


def _recorded_evaluation(evaluation: object) -> str | None:
    """The run's recorded evaluation in one line: how it stands, where, and its headline."""
    if not isinstance(evaluation, Mapping):
        return None
    status = str(evaluation.get("status")).replace("_", " ")
    text = f"evaluation {status} in {evaluation.get('out_dir')}"
    found = evaluation.get("headline")
    rates = found.get("judge_pass_rate") if isinstance(found, Mapping) else None
    if isinstance(rates, Mapping):
        text += (
            f": judge agreement base {_agreement(rates.get('base'))}, tuned "
            f"{_agreement(rates.get('tuned'))}, tuned - base {_difference(rates)}"
        )
    return text


def status_summary(result: Mapping[str, Any]) -> Summary:
    """``status``: the status line, and on a terminal the run's model, its recorded
    evaluation and its error (off a terminal, the evaluation and the error)."""
    details = [line] if (line := _model_line(result)) is not None else []
    brief: list[str] = []
    if (evaluation := _recorded_evaluation(result.get("evaluation"))) is not None:
        details.append(evaluation)
        brief.append(evaluation)
    if (error := _error_text(result.get("error"))) is not None:
        details.append(f"error: {error}")
        brief.append(f"error: {error}")
    return _summary(
        status_line(result), details, brief=brief, next_command=result.get("next_command")
    )


def cancel_summary(result: Mapping[str, Any]) -> Summary:
    """``cancel --confirm``: which process was signalled, and what the run does now."""
    headline = (
        f"sent SIGTERM to process {result.get('pid')}: run {result.get('run_id')} stops after "
        "its current step, saves a resumable checkpoint and promotes its best epoch"
    )
    return _summary(headline, next_command=result.get("next_command"))


def runs_summary(result: Mapping[str, Any]) -> Summary:
    """``runs``: how many runs there are, and the newest one."""
    found = [run for run in result.get("runs") or [] if isinstance(run, Mapping)]
    where = result.get("dir")
    if not found:
        return _summary(f"no runs under {where}")
    newest = found[0]
    headline = f"{_n(len(found), 'run')} under {where}"
    latest = f"newest {newest.get('run_id')} ({newest.get('phase')}, {newest.get('model')})"
    return _summary(headline, [latest], brief=[latest], next_command=result.get("next_command"))


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
    remote = result.get("remote")
    if status == "started" and isinstance(remote, Mapping):
        return _summary(
            f"run {result['run_id']} started in {_kind(remote.get('backend'))} "
            f"{remote.get('job_id')} (team {remote.get('team')})",
            [f"project: {remote.get('project')}"],
            brief=[f"project: {remote.get('project')}"],
            next_command=result.get("next_command"),
        )
    if status == "started":
        return _summary(
            f"run {result['run_id']} started in a background worker (pid {result['pid']})",
            [f"log: {result['worker_log']}"],
            next_command=result.get("next_command"),
        )
    if status == "calibration_failed":
        return evaluate_summary(result)
    phase = status
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


def _agreement(rate: object) -> str:
    """A target's judge pass rate: ``25%``, or ``n/a``."""
    return _percent(rate)


def _difference(rates: Mapping[str, Any]) -> str:
    delta = rates.get("delta")
    if not isinstance(delta, int | float):
        return "n/a"
    text = f"{_points(delta)} pts"
    ci = rates.get("ci95")
    if isinstance(ci, Sequence) and len(ci) == 2:
        text += f" (95% CI {_points(ci[0])} to {_points(ci[1])})"
    return text


def _verdicts(paired: Mapping[str, Any], errors: object, targets: Sequence[str]) -> str:
    """One comparison's paired verdicts, and the errors of the two ``targets`` it compares."""
    failed = sum(int(errors.get(name, 0)) for name in targets) if isinstance(errors, Mapping) else 0
    return (
        f"{_n(paired.get('wins', 0), 'win')}, "
        f"{_n(paired.get('regressions', 0), 'regression')}, "
        f"{_n(paired.get('ties', 0), 'tie')}, {_n(failed, 'error')}"
    )


def _publication(result: Mapping[str, Any]) -> str | None:
    publication = result.get("publication")
    if not isinstance(publication, Mapping):
        return None
    status = publication.get("status")
    link = publication.get("comparison_url")
    compare = f"; compare them at {link}" if isinstance(link, str) and link else ""
    if status == "published":
        experiments = publication.get("experiments") or {}
        names = [
            str(entry.get("name"))
            for entry in experiments.values()
            if isinstance(entry, Mapping) and entry.get("name")
        ]
        found = "published as experiments " + " and ".join(names) if names else "published"
        return found + compare
    line = {
        "skipped": "not published (see the warning above)",
        "disabled": "not published (publishing was turned off)",
        "failed": "publishing failed; next publishes the rest",
        "cancelled": "publishing was cancelled; next publishes the rest",
        "pending": "not fully published yet: the rest is published once every case has a result",
    }.get(str(status))
    return None if line is None else line + compare


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
    headline_data = result["headline"]
    rates = headline_data["judge_pass_rate"]
    errors = int(sum(int(count) for count in (result.get("errors") or {}).values()))
    what = {
        "complete": "complete",
        "complete_with_errors": f"complete with {_n(errors, 'error')}",
        "partial": f"partial ({_n(result.get('pending', 0), 'pair')} left)",
    }.get(str(status), str(status))
    judge = result.get("judge_model")
    headline = (
        f"evaluation {what}: {_n(headline_data.get('cases'), 'case')} from "
        f"{_n(headline_data.get('trajectories'), 'test conversation')}"
        + (f"; judge {judge}" if judge else "")
        + f", {_calibration(result)}"
    )
    base, tuned = _agreement(rates.get("base")), _agreement(rates.get("tuned"))
    difference = _difference(rates)
    paired = "paired: " + _verdicts(
        headline_data.get("paired") or {}, result.get("errors"), ("base", "tuned")
    )
    publication = _publication(result)
    if isinstance(headline_data.get("tuned_minus_compare"), Mapping):
        return _compared_parts(result, headline, publication)
    agreement = f"judge agreement: base {base}, tuned {tuned}, tuned - base {difference}"
    if compact:
        lines = [headline, agreement, paired, CAVEAT]
    else:
        lines = [
            headline,
            f"{'':<21}{'base':<11}{'tuned':<11}tuned - base",
            f"{'judge agreement':<21}{base:<11}{tuned:<11}{difference}",
            paired,
            CAVEAT,
        ]
    parts = [headline, agreement, paired.removeprefix("paired: "), CAVEAT]
    if publication is not None:
        lines.append(publication)
        parts.append(publication)
    return Summary(tuple(lines), "; ".join(parts))


def _compared_parts(result: Mapping[str, Any], headline: str, publication: str | None) -> Summary:
    """An evaluation with a compare target: each target's judge agreement, then each
    comparison with its paired verdicts."""
    headline_data = result["headline"]
    rates: Mapping[str, Any] = headline_data["judge_pass_rate"]
    versus: Mapping[str, Any] = headline_data["tuned_minus_compare"]
    names = [name for name in rates if name not in ("delta", "ci95")]
    found = [
        headline,
        "judge agreement: " + ", ".join(f"{name} {_agreement(rates[name])}" for name in names),
        f"tuned - base: {_difference(rates)}; "
        + _verdicts(headline_data.get("paired") or {}, result.get("errors"), ("base", "tuned")),
        f"tuned - compare: {_difference(versus)}; "
        + _verdicts(versus, result.get("errors"), ("tuned", "compare")),
        CAVEAT,
        *([publication] if publication is not None else []),
    ]
    return Summary(tuple(found), "; ".join(found))


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


def benchmark_summary(result: Mapping[str, Any]) -> Summary | None:
    """``benchmark``: the cases and judge, each model's judge agreement with its CI and gap
    to the best, the recommendation and the caveat."""
    status = result.get("status")
    if status == "calibration_failed":
        return _evaluation_parts(result, compact=False)
    board = [entry for entry in result.get("leaderboard") or [] if isinstance(entry, Mapping)]
    if status is None or not board:
        return None
    what = {"complete": "complete", "partial": "partial"}.get(str(status), "complete with errors")
    headline = (
        f"benchmark {what}: {_n(len(board), 'model')} on {_n(result.get('cases'), 'case')} from "
        f"{_n(result.get('trajectories'), 'test conversation')}; judge {result.get('judge_model')}"
    )
    width = max(len(str(entry.get("model"))) for entry in board) + 2
    rows = []
    for entry in board:
        ci = entry.get("ci95")
        interval = f" [{_percent(ci[0])}, {_percent(ci[1])}]" if isinstance(ci, list) else ""
        gap = entry.get("gap_to_best")
        if isinstance(gap, Mapping) and isinstance(gap.get("judge_pass_rate"), int | float):
            noise = ", within noise" if entry.get("within_noise_of_best") else ""
            interval += f"  {_points(-float(gap['judge_pass_rate']))} vs best{noise}"
        elif entry.get("status") == "error":
            interval = f"  {_error_text(entry.get('error')) or 'error'}"
        rows.append(
            f"{entry.get('model')!s:<{width}}{_percent(entry.get('judge_pass_rate'))}{interval}"
        )
    recommendation = result.get("recommendation") or {}
    verdict = {
        "fine_tune": f"recommended: fine-tune {recommendation.get('model')}",
        "base_meets_target": f"recommended: {recommendation.get('model')} already meets the "
        f"{_percent(recommendation.get('target_pass_rate'))} target; fine-tuning may add little",
    }.get(str(recommendation.get("verdict")), "no recommendation yet")
    return _summary(
        headline,
        [*rows, verdict, CAVEAT],
        brief=[*rows, verdict],
        next_command=result.get("next_command"),
    )


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


def label_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset label``: the labels recorded from the file, and the dataset's kept and
    dropped rows now."""
    labeled = result.get("label") or {}
    headline = (
        f"recorded {_n(labeled.get('applied', 0), 'label')} from {labeled.get('file')} "
        f"({labeled.get('kept', 0)} keep, {labeled.get('dropped', 0)} drop, "
        f"{labeled.get('unchanged', 0)} unchanged)"
    )
    now = (
        f"the dataset keeps {result.get('kept', 0)} of {_n(result.get('accepted', 0), 'row')}, "
        f"{result.get('dropped', 0)} dropped"
    )
    return _summary(headline, [now], brief=[now], next_command=result.get("next_command"))


def sample_summary(result: Mapping[str, Any]) -> Summary:
    """``dataset sample``: how many rows were shown out of how many, never their content."""
    rows = result.get("rows")
    shown = len(rows) if isinstance(rows, list) else 0
    split = result.get("split")
    where = f"the {split} split of " if split and split != "all" else ""
    headline = (
        f"sampled {shown} of {_n(result.get('eligible', 0), 'row')} from "
        f"{where}{result.get('dataset_dir')} (seed {result.get('seed')})"
    )
    if not result.get("include_dropped"):
        headline += ", rows labeled drop left out"
    full = f"full rows: {result.get('rows_file')}"
    return _summary(headline, [full], next_command=result.get("next_command"))


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
    model = str(result.get("model") or "").rsplit("/", 1)[-1] or None  # the model's id
    deployment = result.get("deployment_id")
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
        offline = "bills nothing" in str(result.get("billing") or "")
        bills = "it bills nothing (offline)" if offline else "it bills while a replica runs"
        stop = f"{bills}; delete it with: {undeploy}"
        details.append(stop)
        brief.append(stop)
    return _summary(headline, details, brief=brief, next_command=result.get("next_command"))


def undeploy_summary(result: Mapping[str, Any]) -> Summary:
    """``undeploy``: the deployment deleted and the model kept."""
    if result.get("message"):
        headline = f"the deployment {result.get('deployment_id')} was {result['message']}"
    else:
        headline = f"deleted the deployment {result.get('deployment_id')}"
    kept = f"the model {result.get('model_kept') or result.get('model')} is kept"
    return _summary(headline, [kept], brief=[kept], next_command=result.get("next_command"))


_MARKS: Final[dict[str, str]] = {"pass": "ok", "fail": "FAIL", "optional": "--", "skipped": "skip"}
"""Each check status's mark in ``doctor``'s checklist."""

_CHECK_WIDTH: Final = 44
_UNSET_KEY: Final = " key not set"


def _on(value: object) -> str:
    return "on" if value else "off"


def _doctor_header(result: Mapping[str, Any]) -> str:
    versions = result.get("versions") or {}
    mode = result.get("mode") or {}
    # A checkout names its commit, read at run time: the build's local version part names
    # the commit it was installed at, which may be stale.
    version = str(versions.get("pup_finetune") or "unknown")
    checkout = versions.get("checkout")
    name = f"pup-finetune {version.split('+')[0] if checkout else version}"
    if checkout:
        name += f" (checkout {checkout})"
    return f"{name}   read-only: {_on(mode.get('read_only'))}   dev: {_on(mode.get('dev'))}"


def _check_rows(checks: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str, str | None]]:
    """``(status, text, note, fix)`` per checklist line. Consecutive optional checks of keys
    that are not set share one line (``Anthropic/OpenAI keys not set``)."""
    rows: list[tuple[str, str, str, str | None]] = []
    unset: list[Mapping[str, Any]] = []

    def flush() -> None:
        if len(unset) == 1:
            check = unset[0]
            rows.append(("optional", str(check.get("label")), _note(check), None))
        elif unset:
            names = "/".join(str(c.get("label"))[: -len(_UNSET_KEY)] for c in unset)
            uses = _either([str(c.get("needed_for")) for c in unset if c.get("needed_for")])
            rows.append(("optional", f"{names} keys not set", f"only for {uses}", None))
        unset.clear()

    for check in checks:
        status = str(check.get("status"))
        label = str(check.get("label") or check.get("detail") or check.get("name"))
        if status == "optional" and label.endswith(_UNSET_KEY):
            unset.append(check)
            continue
        flush()
        fix = check.get("fix") if status == "fail" else None
        rows.append((status, label, _note(check), str(fix) if fix else None))
    flush()
    return rows


def _either(uses: Sequence[str]) -> str:
    """``a/... or b/... judges`` for ``a/... judges`` and ``b/... judges``."""
    last = {use.rpartition(" ")[2] for use in uses}
    if len(last) == 1 and all(" " in use for use in uses):
        return f"{' or '.join(use.rpartition(' ')[0] for use in uses)} {last.pop()}"
    return " or ".join(uses)


def _note(check: Mapping[str, Any]) -> str:
    needed_for = check.get("needed_for")
    if not needed_for:
        return ""
    return f"only for {needed_for}" if check.get("status") == "optional" else str(needed_for)


def _steps_text(names: Sequence[str], marks: Mapping[str, str] | None = None) -> str:
    """``dataset preview, pull, show`` for steps that share their first word and are one
    word after it; every other step in full. ``marks`` adds a note after a step's name
    (``dataset push (read-only)``)."""
    parts: list[str] = []
    previous = ""
    for name in names:
        first, _, rest = name.partition(" ")
        part = rest if first == previous and rest and " " not in rest else name
        mark = (marks or {}).get(name)
        parts.append(f"{part} ({mark})" if mark else part)
        previous = first if rest and " " not in rest else ""
    return ", ".join(parts)


def _other_provider(name: str, provider: object) -> bool:
    """Whether ``name`` is a step on a provider other than ``provider`` (``doctor
    --provider``), which the ``Not ready`` list leaves out."""
    _, on, where = name.rpartition(" on ")
    return bool(provider) and bool(on) and where != provider


def _ready_line(ready: Mapping[str, Any], provider: object = None) -> str | None:
    """``Ready: ...  Not ready: ...``. A step ``PUP_READ_ONLY`` blocks says ``(read-only)``;
    with ``--provider``, the steps on other providers are not listed as not ready."""
    from pup_finetune.readiness import READ_ONLY

    steps = {name: step for name, step in ready.items() if isinstance(step, Mapping)}
    if not steps:
        return None
    yes = [name for name, step in steps.items() if step.get("ready")]
    no = [
        name
        for name, step in steps.items()
        if not step.get("ready") and not _other_provider(name, provider)
    ]
    read_only = {
        name: "read-only" for name in no if READ_ONLY in (steps[name].get("blocked_by") or [])
    }
    parts = []
    if yes:
        parts.append(f"Ready: {_steps_text(yes)}.")
    if no:
        parts.append(f"Not ready: {_steps_text(no, read_only)}.")
    return "  ".join(parts)


def _doctor_next(result: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> str | None:
    """The command to run next, else the fix of the first required check that failed,
    else the first of ``next_steps``."""
    if isinstance(result.get("next_command"), str) and result["next_command"]:
        return str(result["next_command"])
    for check in checks:
        if check.get("required") and check.get("status") != "pass" and check.get("fix"):
            return str(check["fix"])
    steps = result.get("next_steps") or []
    for found in steps:
        if isinstance(found, Mapping) and found.get("command"):
            return str(found["command"])
    return None


def doctor_summary(result: Mapping[str, Any]) -> Summary:
    """``doctor``: a checklist, in full on a terminal or off one. A header with the version
    and mode, a line per check (mark, what it found, what needs it; a failure's fix below
    it), the steps that are ready and not, and what to do next."""
    checks = [check for check in result.get("checks") or [] if isinstance(check, Mapping)]
    lines = [_doctor_header(result)]
    rows = _check_rows(checks)
    width = min(max((len(text) for _, text, _, _ in rows), default=0), _CHECK_WIDTH) + 2
    for status, text, note, fix in rows:
        mark = _MARKS.get(status, status)
        line = f"{mark:<6}{text:<{width}}{note}" if note else f"{mark:<6}{text}"
        lines.append(line.rstrip())
        if fix:
            lines.append(f"{'':<6}fix: {fix}")
    ready = _ready_line(result.get("ready") or {}, result.get("provider"))
    if ready:
        lines.append(ready)
    following = _doctor_next(result, checks)
    if following:
        lines.append(f"Next: {following}")
    return Summary(tuple(lines), "\n".join(lines), full=True)


# --- previews ----------------------------------------------------------------------------

CARD_LINES: Final = 7
"""The most lines a preview's card prints, on a terminal or off one: the headline and six
rows."""

PREVIEW_HEADLINE: Final = "Preview only, nothing was run"
"""How every preview's card starts."""

_LABEL_WIDTH: Final = 10

KEPT_ROWS: Final = frozenset({"refused", "cost", "bills", "sends"})
"""The rows a card never drops: why ``--confirm`` is refused, what is billed, what it costs
and who is sent what."""

DROPPED_FIRST: Final = (
    "settings",
    "runs",
    "rubric",
    "publishes",
    "project",
    "judges",
    "evaluate",
    "data",
)
"""The rows a card too long for :data:`CARD_LINES` drops, first to last."""

Row = tuple[str, str]
"""A card row: its label and its text."""


def _row(label: str, text: object) -> Row | None:
    return (label, str(text)) if text else None


def _line(row: Row) -> str:
    label, text = row
    return f"  {label:<{_LABEL_WIDTH}}{text}"


def _count(value: object) -> str:
    return f"{value:,}" if isinstance(value, int) else "?"


def _rate(value: object) -> str:
    """A learning rate the short way: ``1e-4``, ``3e-5``, ``0.002``."""
    if not isinstance(value, int | float):
        return "?"
    if value and abs(value) < 1e-3:
        mantissa, _, exponent = f"{value:e}".partition("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return f"{value:g}"


def _sends(preview: Mapping[str, Any]) -> str | None:
    recipients = preview.get("data_recipients")
    if not isinstance(recipients, list) or not recipients:
        return None
    return "; ".join(str(entry).replace(" -> ", " to ") for entry in recipients)


UNKNOWN_RATES: Final = ", which pup-finetune does not know"
"""How a provider's billing sentence says pup-finetune does not know its prices."""

OFFLINE: Final = "nothing: the fake provider runs offline"
"""The cost row of a run on the offline fake provider, which bills nothing."""


def _bills(billing: object) -> str | None:
    """Who bills what, from a preview's billing sentence: the head (before ``: ``) of each
    ``; ``-separated part that has one, each once, without what bills nothing (an offline
    fake) and with "at rates pup-finetune does not know" said once."""
    if not isinstance(billing, str):
        return None
    heads: list[str] = []
    unknown = False
    for part in billing.split("; "):
        head, sep, _ = part.partition(": ")
        if not sep or "bills nothing" in head:
            continue
        unknown = unknown or head.endswith(UNKNOWN_RATES)
        head = head.removesuffix(UNKNOWN_RATES)
        if head not in heads:
            heads.append(head)
    if not heads:
        return None
    return "; ".join(heads) + (" (pup-finetune does not know the rates)" if unknown else "")


def _offline(preview: Mapping[str, Any]) -> bool:
    return preview.get("provider") == "fake"


def _spend(preview: Mapping[str, Any], *, resumed: bool = False) -> str:
    from pup_finetune.providers.base import spend_cap_text

    if _offline(preview):
        return OFFLINE
    settings = preview.get("settings") or {}
    cap, rate = settings.get("max_spend_usd"), settings.get("hourly_rate_usd")
    return spend_cap_text(
        cap if isinstance(cap, int | float) else None,
        rate if isinstance(rate, int | float) else None,
        resumed=resumed,
    )


def _billing_rows(preview: Mapping[str, Any], *, resumed: bool = False) -> list[Row | None]:
    """A training preview's ``bills`` row (who bills what) and ``cost`` row (the spend guard;
    on the offline fake provider, that it bills nothing)."""
    return [
        _row("bills", _bills(preview.get("billing"))),
        _row("cost", _spend(preview, resumed=resumed)),
    ]


def _token_data(preview: Mapping[str, Any]) -> str:
    rows = preview.get("rows") or {}
    text = f"{rows.get('train', 0)} train / {rows.get('validation', 0)} validation rows"
    if rows.get("rejected"):
        text += f" ({rows['rejected']} rejected)"
    text += (
        f"; {_count(preview.get('tokens_per_epoch'))} tokens per epoch, at most "
        f"{_count(preview.get('max_tokens_total'))}"
    )
    if preview.get("validation_tokens"):
        text += f" (+{_count(preview['validation_tokens'])} validation per epoch)"
    return text


MAX_SEQ_LEN_OPTION: Final = "max_seq_len"


def _training_settings(settings: Mapping[str, Any]) -> str:
    lora = settings.get("lora") or {}
    text = (
        f"up to {_n(settings.get('max_epochs'), 'epoch')} (patience {settings.get('patience')}), "
        f"batch {settings.get('batch_size')}, lr {_rate(settings.get('learning_rate'))}, "
        f"LoRA rank {lora.get('rank')} ({lora.get('rank_source')})"
    )
    if lora.get("alpha") is not None:
        text += f", alpha {lora['alpha']}"
    source = settings.get("max_seq_len_source")
    if source and source != "model default":
        text += f", context {_count(settings.get('max_seq_len'))} ({source})"
    options = settings.get("provider_options") or {}
    shown = {key: value for key, value in options.items() if key != MAX_SEQ_LEN_OPTION}
    if shown:
        text += ", options " + " ".join(f"{key}={value}" for key, value in shown.items())
    return text


def _train_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    settings = preview.get("settings") or {}
    rows = [
        _row("data", _token_data(preview)),
        _row("settings", _training_settings(settings)),
    ]
    remote = preview.get("remote")
    if isinstance(remote, Mapping):
        rows.append(_row("runs", f"a Baseten CPU job in team {remote.get('team')}"))
    evaluation = preview.get("evaluation")
    if isinstance(evaluation, Mapping):
        rows.append(_row("evaluate", _evaluation_calls(evaluation)))
    rows += [_row("sends", _sends(preview)), *_billing_rows(preview)]
    headline = f"train {preview.get('model')} on {preview.get('provider')}"
    if preview.get("run_dir"):
        headline += f" in {preview['run_dir']}"
    return headline, rows


def _resume_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    headline = f"resume run {preview.get('run_id')} on {preview.get('provider')}"
    rows = [_row("does", preview.get("action"))]
    if preview.get("mode") != "finalize":
        rows.append(_row("data", _token_data(preview)))
    rows += [_row("sends", _sends(preview)), *_billing_rows(preview, resumed=True)]
    return headline, rows


def _evaluation_calls(preview: Mapping[str, Any]) -> str:
    return (
        f"{_n(preview.get('cases'), 'case')} from {preview.get('trajectories')} test "
        f"conversations; {_count(preview.get('generation_calls'))} generations, "
        f"{_count(preview.get('judge_calls'))} judge calls"
    )


def _evaluate_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    settings = preview.get("settings") or {}
    publish = preview.get("publish") or {}
    headline = f"evaluate run {preview.get('run_id')} with judge {settings.get('judge_model')}"
    compare = preview.get("compare")
    if isinstance(compare, Mapping):
        headline += f", comparing with {compare.get('model')}"
    detail = (
        f"at most {_count(settings.get('max_tokens'))} tokens per generation, "
        f"{settings.get('max_cases')} turns per conversation, seed "
        f"{settings.get('seed')}"
    )
    if isinstance(compare, Mapping):
        detail += f" ({_count(compare.get('max_tokens'))} for the compare model)"
    experiments = len(publish.get("experiments") or [])
    publishes = f"{experiments} experiments to Datadog" if experiments else "nothing"
    rows = [
        _row("data", _evaluation_calls(preview)),
        _row("settings", detail),
        _row("sends", _sends(preview)),
        _row("publishes", publishes),
        _row(
            "bills",
            _bills(preview.get("billing")) or "nothing: the fake provider and judge run offline",
        ),
    ]
    return headline, rows


def _benchmark_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    settings = preview.get("settings") or {}
    models = ", ".join(str(model) for model in settings.get("models") or [])
    headline = f"benchmark {models} with judge {settings.get('judge_model')}"
    detail = (
        f"at most {_count(settings.get('max_tokens'))} tokens per generation, "
        f"{settings.get('max_cases')} turns per conversation, target pass rate "
        f"{_percent(settings.get('target_pass_rate'))}"
    )
    rows = [
        _row("data", _evaluation_calls(preview)),
        _row("settings", detail),
        _row("sends", _sends(preview)),
        _row(
            "bills",
            _bills(preview.get("billing")) or "nothing: the fake provider and judge run offline",
        ),
    ]
    return headline, rows


def _triage_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    judges = [judge for judge in preview.get("judges") or [] if isinstance(judge, Mapping)]
    rows = preview.get("rows") or {}
    headline = f"triage {_n(rows.get('to_review'), 'row')} with {_n(len(judges), 'judge')}"
    data = (
        f"{_count(preview.get('judge_calls_pending'))} judge calls "
        f"({_count(preview.get('votes_reused'))} votes reused), about "
        f"{_count(preview.get('approx_input_tokens'))} input tokens (approximate)"
    )
    from pup_finetune.judge_models import alias_of

    shown = ", ".join(
        f"{alias_of(str(judge.get('model')))} ({judge.get('provider')})" for judge in judges
    )
    return headline, [
        _row("data", data),
        _row("judges", shown),
        _row("rubric", _criteria(preview)),
        _row("sends", _sends(preview)),
        _row("bills", _bills(preview.get("billing")) or "nothing: the judges run offline"),
    ]


def _criteria(preview: Mapping[str, Any]) -> str:
    """What a triage judges by: ``rubric rubric.md + 2 rules (hash 0123456789ab)``."""
    source = str(preview.get("criteria") or "")
    parts: list[str] = []
    if "rubric" in source.split("+"):
        rubric_file = preview.get("rubric_file")
        parts.append(f"rubric {rubric_file}" if rubric_file else "a rubric")
    rules = preview.get("rule_count")
    if isinstance(rules, int) and rules:
        parts.append(_n(rules, "rule"))
    elif "rules" in source.split("+"):
        parts.append("rules")
    text = " + ".join(parts) or source
    if digest := preview.get("rubric_hash"):
        text += f" (hash {digest})"
    return text


def _push_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    rows = preview.get("rows") or {}
    headline = (
        f"push {_n(preview.get('records_to_push'), 'record')} to Datadog dataset "
        f"{preview.get('dataset_name')!r}"
    )
    data = (
        f"{rows.get('train', 0)} train / {rows.get('validation', 0)} validation / "
        f"{rows.get('test', 0)} test records; {preview.get('already_pushed', 0)} already pushed"
    )
    return headline, [
        _row("data", data),
        _row("project", preview.get("project")),
        _row("sends", _sends(preview)),
    ]


def _deploy_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    settings = preview.get("settings") or {}
    shape = settings.get("shape") or settings.get("shape_source")
    detail = (
        f"deployment {settings.get('deployment_id')}, shape {shape}, replicas "
        f"{settings.get('min_replicas')}-{settings.get('max_replicas')}, wait up to "
        f"{settings.get('timeout_minutes'):g} min"
        if isinstance(settings.get("timeout_minutes"), int | float)
        else None
    )
    headline = f"deploy {preview.get('model')} on {preview.get('provider')}"
    cost = OFFLINE if _offline(preview) else preview.get("billing")
    return headline, [
        _row("settings", detail),
        _row("sends", _sends(preview)),
        _row("cost", cost),
    ]


def _action_card(preview: Mapping[str, Any]) -> tuple[str, list[Row | None]]:
    return str(preview.get("action") or ""), []


PreviewCard = Callable[[Mapping[str, Any]], tuple[str, list[Row | None]]]

PREVIEW_CARDS: Final[dict[str, PreviewCard]] = {
    "train": _train_card,
    "resume": _resume_card,
    "evaluate": _evaluate_card,
    "benchmark": _benchmark_card,
    "dataset triage": _triage_card,
    "dataset push": _push_card,
    "deploy": _deploy_card,
}
"""Each previewing command's card, by its path; any other preview shows its ``action``."""


def _fitted(rows: list[Row]) -> list[Row]:
    """``rows`` cut to fit :data:`CARD_LINES` under the headline: the rows of
    :data:`DROPPED_FIRST` dropped in its order, never one of :data:`KEPT_ROWS`."""
    fitted = list(rows)
    for label in DROPPED_FIRST:
        if len(fitted) < CARD_LINES:
            break
        fitted = [row for row in fitted if row[0] != label]
    return fitted


def preview_card(
    command: str | None, preview: Mapping[str, Any], *, refused: str | None = None
) -> list[str]:
    """The card a preview prints on stderr, at most :data:`CARD_LINES` lines generated from
    its fields: what ``--confirm`` would do, then the data, the settings, who is sent what,
    who bills it and what it costs; ``refused`` (why ``--confirm`` would be refused) comes
    second. A card too long drops its lower-value rows (:func:`_fitted`), never the cost,
    the billing, the recipients or the refusal.

    A card that fails on an unexpected preview, however, falls back to the headline: the
    preview itself was printed already.
    """
    card = PREVIEW_CARDS.get(command or "", _action_card)
    try:
        what, found = card(preview)
    except Exception as exc:
        _debug_traceback(exc)
        what, found = "", []
    if not what and not found:
        return []  # nothing to say beyond the message: the caller prints that
    rows = [row for row in found if row]
    if refused:
        rows.insert(0, ("refused", refused))
    lines = [f"{PREVIEW_HEADLINE}: {what}" if what else PREVIEW_HEADLINE]
    return lines + [_line(row) for row in _fitted(rows)]


SUMMARIES: Final[dict[str, Summarizer]] = {
    "train": train_summary,
    "resume": train_summary,
    "status": status_summary,
    "cancel": cancel_summary,
    "runs": runs_summary,
    "evaluate": evaluate_summary,
    "benchmark": benchmark_summary,
    "dataset preview": preview_summary,
    "dataset pull": pull_summary,
    "dataset import": import_summary,
    "dataset label": label_summary,
    "dataset sample": sample_summary,
    "dataset triage": triage_summary,
    "dataset push": push_summary,
    "deploy": deploy_summary,
    "undeploy": undeploy_summary,
    "doctor": doctor_summary,
}
"""Each command's summarizer, by its path below the program."""
