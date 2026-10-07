# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Progress lines for training and evaluation: per step on a terminal, throttled off one."""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from pup_finetune import output
from pup_finetune.progress import (
    EvaluationProgress,
    TrainingProgress,
    describe_pair,
    every_line,
)
from pup_finetune.runtime import Runtime


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def plan(steps_per_epoch: int, max_epochs: int = 2, **extra: Any) -> dict[str, Any]:
    return {
        "type": "plan",
        "steps_per_epoch": steps_per_epoch,
        "max_steps": steps_per_epoch * max_epochs,
        "max_epochs": max_epochs,
        "checkpoint_every_minutes": 15.0,
        "epoch": 1,
        "step": 0,
        **extra,
    }


def step(epoch: int, number: int, loss: float = 2.0) -> dict[str, Any]:
    return {"type": "step", "epoch": epoch, "step": number, "loss": loss}


def checkpoint(kind: str, name: str, epoch: int, number: int, reason: str) -> dict[str, Any]:
    return {
        "type": "checkpoint",
        "kind": kind,
        "name": name,
        "epoch": epoch,
        "step": number,
        "reason": reason,
    }


def epoch_end(epoch: int, number: int, *, val: float | None = 2.1) -> dict[str, Any]:
    return {
        "type": "epoch",
        "epoch": epoch,
        "step": number,
        "train_loss": 2.0,
        "val_loss": val,
        "new_best": True,
    }


def piped(clock: Clock) -> tuple[TrainingProgress, list[str]]:
    lines: list[str] = []
    return TrainingProgress(write=lines.append, every_step=False, fold=True, clock=clock), lines


# --- training, off a terminal ---------------------------------------------------------------


def test_off_a_terminal_a_step_line_comes_every_quarter_of_an_epoch() -> None:
    clock = Clock()
    progress, lines = piped(clock)
    progress(plan(steps_per_epoch=20))
    for number in range(1, 21):
        progress(step(1, number))
    # The first step, then every 5 steps (a quarter of 20); the last one waits for its epoch.
    assert [line.split(":")[0] for line in lines] == [
        "epoch 1/2 step 1/40",
        "epoch 1/2 step 6/40",
        "epoch 1/2 step 11/40",
        "epoch 1/2 step 16/40",
    ]


def test_off_a_terminal_a_slow_epoch_gets_a_step_line_every_30_seconds() -> None:
    clock = Clock()
    progress, lines = piped(clock)
    progress(plan(steps_per_epoch=1000))
    for number in range(1, 101):
        clock.now += 1.0  # 100 steps of a 1000-step epoch, a second each
        progress(step(1, number))
    assert [line.split(":")[0] for line in lines] == [
        "epoch 1/2 step 1/2000",
        "epoch 1/2 step 31/2000",
        "epoch 1/2 step 61/2000",
        "epoch 1/2 step 91/2000",
    ]


def test_off_a_terminal_the_epoch_line_names_its_checkpoint() -> None:
    progress, lines = piped(Clock())
    progress(plan(steps_per_epoch=1, max_epochs=5))
    progress(step(1, 1))
    progress(checkpoint("training", "t1s7-bdc2", 1, 7, "epoch"))
    progress(checkpoint("sampler", "s1s7-bdc2", 1, 7, "epoch"))
    progress(epoch_end(1, 7))
    assert lines == ["epoch 1/5 done: train 2.000, val 2.100 (new best), checkpoint t1s7-bdc2"]


def test_a_timed_or_stop_checkpoint_keeps_a_line_of_its_own() -> None:
    progress, lines = piped(Clock())
    progress(plan(steps_per_epoch=10))
    progress(checkpoint("training", "t1s4-ab", 1, 4, "time"))
    progress(checkpoint("training", "t1s6-ab", 1, 6, "stop"))
    assert lines == [
        "saved training checkpoint t1s4-ab on the 15-minute timer (epoch 1, step 4)",
        "saved training checkpoint t1s6-ab at the stop (epoch 1, step 6)",
    ]


def test_an_epoch_without_validation_shows_the_training_loss_only() -> None:
    progress, lines = piped(Clock())
    progress(plan(steps_per_epoch=1))
    progress(epoch_end(1, 1, val=None))
    assert lines == ["epoch 1/2 done: train 2.000 (new best)"]


def test_promotion_retries_and_stops_always_get_a_line() -> None:
    progress, lines = piped(Clock())
    progress(plan(steps_per_epoch=4))
    progress({"type": "retry", "call": "forward", "attempt": 1, "error": "X", "delay_s": 2.0})
    progress({"type": "stopped", "reason": "early_stopped", "epoch": 2})
    progress({"type": "stopped", "reason": "budget_exhausted", "detail": "the cap is spent"})
    progress({"type": "promoting", "epoch": 2, "val_loss": 1.6, "train_loss": 1.4})
    progress({"type": "promoting", "epoch": 2, "val_loss": None, "train_loss": 1.4})
    progress({"type": "promoted", "model_id": "ft-1", "epoch": 2})
    assert lines == [
        "retrying forward (attempt 1) after X, in 2.0s",
        "stopping early: the loss stopped improving",
        "stopping at the budget: the cap is spent",
        "promoting epoch 2 (val 1.600) ...",
        "promoting epoch 2 (train 1.400) ...",
        "promoted ft-1",
    ]


# --- training, on a terminal or with PUP_FINETUNE_PROGRESS=steps ---------------------------


def test_every_step_mode_prints_every_step_and_each_checkpoint_with_its_reason() -> None:
    lines: list[str] = []
    progress = TrainingProgress(write=lines.append, every_step=True, fold=False, clock=Clock())
    progress(plan(steps_per_epoch=2))
    progress(step(1, 1))
    progress(step(1, 2))
    progress(checkpoint("training", "t1s2-ab", 1, 2, "epoch"))
    progress(epoch_end(1, 2))
    assert lines == [
        "epoch 1/2 step 1/4: loss 2.000",
        "epoch 1/2 step 2/4: loss 2.000",
        "saved training checkpoint t1s2-ab at epoch end (epoch 1, step 2)",
        "epoch 1/2 done: train 2.000, val 2.100 (new best)",
    ]


def test_steps_mode_comes_from_the_environment_or_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert every_line(Runtime.from_env({})) is False
    assert every_line(Runtime.from_env({"PUP_FINETUNE_PROGRESS": "steps"})) is True
    terminal = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(output, "stderr_console", lambda: terminal)
    assert every_line(Runtime.from_env({})) is True


def test_the_default_progress_reads_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setenv("PUP_FINETUNE_PROGRESS", "steps")
    progress = TrainingProgress(write=lines.append, clock=Clock())  # off a terminal, steps
    progress(plan(steps_per_epoch=1))
    progress(step(1, 1))
    progress(checkpoint("training", "t1s1-ab", 1, 1, "epoch"))
    progress(epoch_end(1, 1))
    assert lines == [
        "epoch 1/2 step 1/2: loss 2.000",
        "epoch 1/2 done: train 2.000, val 2.100 (new best), checkpoint t1s1-ab",
    ]


# --- evaluation ----------------------------------------------------------------------------


def pair(done: int, total: int, status: str = "ok", **extra: Any) -> dict[str, Any]:
    return {
        "done": done,
        "total": total,
        "target": "tuned" if done % 2 else "base",
        "case_id": f"row-{done}:1",
        "status": status,
        "passed": extra.pop("passed", False),
        "no_verdict": extra.pop("no_verdict", None),
        "error": extra.pop("error", None),
    }


def test_a_pair_reads_judged_pass_judged_fail_or_the_error_and_its_stage() -> None:
    assert describe_pair(pair(1, 2, passed=True)) == "judged pass"
    assert describe_pair(pair(1, 2)) == "judged fail"
    assert describe_pair(pair(1, 2, no_verdict="unparseable")) == (
        "judged fail, no verdict: unparseable"
    )
    error = {"stage": "generation", "type": "TransientError", "message": "boom"}
    assert describe_pair(pair(1, 2, "error", error=error)) == (
        "error at generation: TransientError"
    )


def test_every_pair_mode_prints_a_line_per_pair() -> None:
    lines: list[str] = []
    progress = EvaluationProgress(write=lines.append, every_pair=True)
    progress(pair(1, 2, passed=True))
    progress(pair(2, 2))
    assert lines == [
        "evaluated 1/2: tuned row-1:1 (judged pass)",
        "evaluated 2/2: base row-2:1 (judged fail)",
    ]


def test_off_a_terminal_evaluation_prints_every_tenth_and_every_error() -> None:
    lines: list[str] = []
    progress = EvaluationProgress(write=lines.append, every_pair=False)
    error = {"stage": "judge", "type": "ProviderCapacityError", "message": "429\nmore"}
    for done in range(1, 41):
        if done == 7:
            progress(pair(done, 40, "error", error=error))
        else:
            progress(pair(done, 40, passed=done % 4 == 0))
    assert lines[0] == "evaluated 4/40 pairs (10%): 1 judged pass, 3 judged fail, 0 errors"
    assert lines[1] == "tuned row-7:1: error at judge: ProviderCapacityError: 429"
    assert lines[2] == "evaluated 8/40 pairs (20%): 2 judged pass, 5 judged fail, 1 error"
    assert lines[-1] == "evaluated 40/40 pairs (100%): 10 judged pass, 29 judged fail, 1 error"
    assert len(lines) == 11  # ten progress lines and the error
