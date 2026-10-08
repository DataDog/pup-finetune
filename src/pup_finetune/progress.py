# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Progress lines for training and evaluation, on stderr.

On a terminal every training step and every evaluated pair gets a line, as before. Off a
terminal (an agent, a pipe, a detached run's ``worker.log``), where every line costs, there
are fewer: a step line at most once per :data:`STEP_LINE_INTERVAL_S` or per
:data:`STEP_LINE_FRACTION` of an epoch, an epoch line with its checkpoint folded in, and an
evaluation line every :data:`EVAL_LINE_FRACTION` of the pairs plus one per error. Every line
off a terminal starts with the time of day (:func:`pup_finetune.output.info`).
``PUP_FINETUNE_PROGRESS=steps`` restores a line per step and per pair.

:class:`TrainingProgress` reads the training loop's events (see
:func:`pup_finetune.training.loop.run_training`), :class:`EvaluationProgress` the replay's
(see :func:`pup_finetune.evaluation.replay.run_replay`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any, Final

from pup_finetune.display import loss_text
from pup_finetune.output import info, stderr_is_terminal
from pup_finetune.runtime import Runtime

__all__ = [
    "EVAL_LINE_FRACTION",
    "STEP_LINE_FRACTION",
    "STEP_LINE_INTERVAL_S",
    "EvaluationProgress",
    "TrainingProgress",
    "describe_pair",
    "every_line",
]

STEP_LINE_INTERVAL_S: Final = 30.0
"""Off a terminal, a step line once this many seconds passed since the last line..."""
STEP_LINE_FRACTION: Final = 0.25
"""...or once the epoch moved on by this share of its steps since then."""
EVAL_LINE_FRACTION: Final = 0.10
"""Off a terminal, an evaluation line each time this share of the pairs is done."""

Write = Callable[[str], None]

_STOPS: Final = {
    "cancelled": "stopping: cancelled",
    "budget_exhausted": "stopping at the budget",
    "early_stopped": "stopping early: the loss stopped improving",
}


def every_line(runtime: Runtime | None = None) -> bool:
    """Whether to print a line per step and per pair: on a terminal, or with
    ``PUP_FINETUNE_PROGRESS=steps``."""
    mode = (runtime or Runtime.from_env()).progress
    return mode == "steps" or stderr_is_terminal()


def _loss(value: object) -> str:
    return loss_text(value) or "n/a"


class TrainingProgress:
    """The training loop's ``on_event``: one progress line per event worth one.

    ``every_step`` prints every step line (on a terminal or with
    ``PUP_FINETUNE_PROGRESS=steps``, by default); otherwise step lines are throttled.
    ``fold``, independently (default: off a terminal, ``steps`` mode included), folds each
    epoch-end training checkpoint into its epoch line instead of a line of its own;
    ``clock`` times the throttle.
    """

    def __init__(
        self,
        *,
        write: Write = info,
        every_step: bool | None = None,
        fold: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._write = write
        self._every_step = every_line() if every_step is None else every_step
        self._fold = (not stderr_is_terminal()) if fold is None else fold
        self._clock = clock
        self._steps_per_epoch: int | None = None
        self._max_steps: int | None = None
        self._max_epochs: int | None = None
        self._timer_minutes: float | None = None
        self._last_line_at: float | None = None
        self._last_step_shown: int | None = None
        self._folded: str | None = None

    def __call__(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        if kind == "plan":
            self._plan(event)
        elif kind == "step":
            self._step(event)
        elif kind == "checkpoint":
            self._checkpoint(event)
        elif kind == "epoch":
            self._epoch(event)
        elif kind == "stopped":
            reason = str(event.get("reason"))
            detail = f": {event['detail']}" if event.get("detail") else ""
            self._line(f"{_STOPS.get(reason, f'stopping: {reason}')}{detail}")
        elif kind == "retry":
            self._line(
                f"retrying {event.get('call')} (attempt {event.get('attempt')}) after "
                f"{event.get('error')}, in {event.get('delay_s')}s"
            )
        elif kind == "promoting":
            if event.get("val_loss") is not None:
                which = f"val {_loss(event.get('val_loss'))}"
            else:
                which = f"train {_loss(event.get('train_loss'))}"
            self._line(f"promoting epoch {event.get('epoch')} ({which}) ...")
        elif kind == "promoted":
            self._line(f"promoted {event.get('model_id')}")

    def _line(self, text: str) -> None:
        self._last_line_at = self._clock()
        self._write(text)

    def _plan(self, event: Mapping[str, Any]) -> None:
        self._steps_per_epoch = _positive(event.get("steps_per_epoch"))
        self._max_steps = _positive(event.get("max_steps"))
        self._max_epochs = _positive(event.get("max_epochs"))
        minutes = event.get("checkpoint_every_minutes")
        self._timer_minutes = float(minutes) if isinstance(minutes, int | float) else None
        step = event.get("step")
        self._last_step_shown = step if isinstance(step, int) else None

    def _epoch_of(self, epoch: object) -> str:
        return f"{epoch}/{self._max_epochs}" if self._max_epochs else str(epoch)

    def _step(self, event: Mapping[str, Any]) -> None:
        step = int(event.get("step") or 0)
        total = f"/{self._max_steps}" if self._max_steps else ""
        text = (
            f"epoch {self._epoch_of(event.get('epoch'))} step {step}{total}: "
            f"loss {_loss(event.get('loss'))}"
        )
        if self._every_step:
            self._last_step_shown = step
            self._line(text)
            return
        per_epoch = self._steps_per_epoch
        if per_epoch and step % per_epoch == 0 and self._last_step_shown is not None:
            return  # the epoch's last step: its epoch line follows
        now = self._clock()
        due = self._last_step_shown is None or self._last_line_at is None
        if not due and self._last_line_at is not None:
            due = now - self._last_line_at >= STEP_LINE_INTERVAL_S
        if not due and per_epoch and self._last_step_shown is not None:
            due = step - self._last_step_shown >= math.ceil(per_epoch * STEP_LINE_FRACTION)
        if due:
            self._last_step_shown = step
            self._line(text)

    def _reason(self, reason: object) -> str:
        if reason == "epoch":
            return "at epoch end"
        if reason == "time":
            if self._timer_minutes is None:
                return "on the timer"
            return f"on the {self._timer_minutes:g}-minute timer"
        if reason == "stop":
            return "at the stop"
        return str(reason)

    def _checkpoint(self, event: Mapping[str, Any]) -> None:
        if self._fold and event.get("reason") == "epoch":
            if event.get("kind") == "training":
                self._folded = str(event.get("name"))
            return  # the epoch line names it; the best epoch's sampler checkpoint is implied
        self._line(
            f"saved {event.get('kind')} checkpoint {event.get('name')} "
            f"{self._reason(event.get('reason'))} (epoch {event.get('epoch')}, step "
            f"{event.get('step')})"
        )

    def _epoch(self, event: Mapping[str, Any]) -> None:
        epoch = self._epoch_of(event.get("epoch"))
        text = f"epoch {epoch} done: train {_loss(event.get('train_loss'))}"
        if event.get("val_loss") is not None:
            text += f", val {_loss(event.get('val_loss'))}"
        if event.get("new_best"):
            text += " (new best)"
        if self._folded is not None:
            text += f", checkpoint {self._folded}"
            self._folded = None
        step = event.get("step")
        if isinstance(step, int):
            self._last_step_shown = step
        self._line(text)


def _positive(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def describe_pair(event: Mapping[str, Any]) -> str:
    """How one evaluated pair ended: ``judged pass``, ``judged fail`` (``no verdict: ...``
    when the judge's answer held none), or ``error at <stage>: <type>``."""
    if event.get("status") == "error":
        error = event.get("error") or {}
        return f"error at {error.get('stage', 'unknown')}: {error.get('type', 'error')}"
    if event.get("no_verdict"):
        return f"judged fail, no verdict: {event['no_verdict']}"
    return "judged pass" if event.get("passed") else "judged fail"


class EvaluationProgress:
    """The replay's ``on_progress``: a line per pair (``every_pair``, on a terminal by
    default), or a line each :data:`EVAL_LINE_FRACTION` of the pairs plus one per error.

    The replay reports finished pairs only: a retry inside a judge or sampler client (its
    backoff included) prints no line until the pair ends.
    """

    def __init__(self, *, write: Write = info, every_pair: bool | None = None) -> None:
        self._write = write
        self._every_pair = every_line() if every_pair is None else every_pair
        self._passed = 0
        self._failed = 0
        self._errors = 0
        self._shown = 0

    def __call__(self, event: Mapping[str, Any]) -> None:
        done, total = int(event.get("done") or 0), int(event.get("total") or 0)
        if event.get("status") == "error":
            self._errors += 1
        elif event.get("passed") and not event.get("no_verdict"):
            self._passed += 1
        else:
            self._failed += 1
        what = f"{event.get('target')} {event.get('case_id')}"
        if self._every_pair:
            self._write(f"evaluated {done}/{total}: {what} ({describe_pair(event)})")
            return
        if event.get("status") == "error":
            message = str((event.get("error") or {}).get("message") or "").strip()
            first = message.splitlines()[0][:200] if message else ""
            self._write(f"{what}: {describe_pair(event)}" + (f": {first}" if first else ""))
        step = max(1, math.ceil(total * EVAL_LINE_FRACTION)) if total else 1
        if done >= total or done - self._shown >= step:
            self._shown = done
            percent = round(done / total * 100) if total else 100
            self._write(
                f"evaluated {done}/{total} pairs ({percent}%): {self._passed} judged pass, "
                f"{self._failed} judged fail, {self._errors} "
                f"{'error' if self._errors == 1 else 'errors'}"
            )
