# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training and resuming in a run directory, with the fake provider and a fake clock.

Each new :class:`FakeProvider` stands for a new process: its backend knows no live session,
so sampler checkpoints from an earlier one have expired.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.providers.base import MODEL_ID_PATTERN, TrainSpec
from pup_finetune.providers.fake import FAKE_PROFILE, FakeCrash, FakeProvider, FakeRenderer
from pup_finetune.runs import (
    RunDir,
    RunState,
    TerminationRequested,
    TrainingTerminated,
    model_id_for_run,
)
from pup_finetune.runtime import Runtime
from pup_finetune.storage import DirectoryLock
from pup_finetune.training import FinalizeFailed, RowExamples
from pup_finetune.training.runner import resume_training, start_training
from tests.fakes.training import row_examples

RUN_ID = "ft-20261006123045-abcdef"
TRAIN = row_examples(4)
VALIDATION = row_examples(2, prefix="val")
LONGEST = max(example.num_tokens for row in [*TRAIN, *VALIDATION] for example in row.examples)


class Clock:
    """Seconds; each finished step takes ``step_minutes``."""

    def __init__(self, step_minutes: float = 4) -> None:
        self.now = 0.0
        self.step_s = step_minutes * 60
        self.events: list[dict[str, Any]] = []

    def __call__(self) -> float:
        return self.now

    def on_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if event["type"] == "step":
            self.now += self.step_s


class CrashingClock(Clock):
    """A :class:`Clock` whose process crashes right after step ``crash_at`` finished."""

    def __init__(self, crash_at: int, step_minutes: float = 4) -> None:
        super().__init__(step_minutes)
        self.crash_at = crash_at

    def on_event(self, event: dict[str, Any]) -> None:
        super().on_event(event)
        if event["type"] == "step" and event["step"] == self.crash_at:
            raise RuntimeError(f"simulated crash after step {self.crash_at}")


def provider(**options: Any) -> FakeProvider:
    return FakeProvider(Runtime(dev_mode=True), **options)


def new_run(path: Path, **settings: Any) -> RunDir:
    fields: dict[str, Any] = {
        "max_epochs": 3,
        "batch_size": 1,
        "pipeline_depth": 1,
        "patience": 3,
        "checkpoint_every_minutes": 10,
    }
    return RunDir.create(
        path,
        provider="fake",
        profile=FAKE_PROFILE,
        spec=TrainSpec(**(fields | settings)),
        dataset_dir=path.parent / "dataset",
        dataset_sha256="d" * 64,
        rendering=FakeRenderer().identity(),
        run_id=RUN_ID,
    )


def start(
    run_dir: RunDir,
    fake: FakeProvider | None = None,
    clock: Clock | None = None,
    rows: list[RowExamples] = TRAIN,
) -> dict[str, Any]:
    clock = clock or Clock()
    return start_training(
        fake or provider(),
        run_dir,
        train_rows=rows,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=clock.on_event,
    )


def resume(
    run_dir: RunDir, fake: FakeProvider | None = None, clock: Clock | None = None
) -> dict[str, Any]:
    clock = clock or Clock()
    return resume_training(
        fake or provider(),
        run_dir,
        train_rows=TRAIN,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=clock.on_event,
    )


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def epoch_metrics(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    keys = ("epoch", "steps", "train_loss", "val_loss", "train_tokens", "new_best")
    return [tuple(epoch[key] for key in keys) for epoch in result["epochs"]]


def crash(run_dir: RunDir, step: int) -> None:
    with pytest.raises(FakeCrash, match="fail_at_step"):
        start(run_dir)
    assert run_dir.load_state().phase == "interrupted"
    assert run_dir.load_state().step == step - 1


# --- training ---------------------------------------------------------------------------


def test_a_run_trains_checkpoints_and_promotes(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    result = start(run_dir, fake)
    assert (result["phase"], result["stopped_reason"]) == ("completed", "completed")
    assert result["best_epoch"] == 3
    assert result["promoted_model"] == {
        "provider": "fake",
        "model_id": RUN_ID,
        "uri": f"fake://models/{RUN_ID}",
    }
    assert result["deployable_checkpoint"] == result["checkpoints"]["best_sampler"]
    assert (result["error"], result["finalize_error"]) == (None, None)
    assert set(result) >= {
        "best_epoch",
        "best_val_loss",
        "stopped_reason",
        "promoted_model",
        "deployable_checkpoint",
        "checkpoints",
        "warnings",
    }
    # Time-based checkpoints at steps 3, 7 and 11, beside the epoch ends.
    state = read(run_dir.state_path)
    training = [(c["epoch"], c["step"]) for c in state["checkpoints"] if c["kind"] == "training"]
    assert training == [(1, 3), (1, 4), (2, 7), (2, 8), (3, 11), (3, 12)]
    assert state["resume_point"]["checkpoint"] == result["checkpoints"]["last_training"]
    epochs = read(run_dir.epochs_path)
    assert [e["training_ckpt"]["step"] for e in epochs] == [4, 8, 12]
    # The session was sized from the spec and the longest example.
    [settings] = fake.backend().session_settings
    assert settings == {"seed": 42, "max_seq_len": LONGEST, "resume_from": None}
    assert FAKE_PROFILE.max_seq_len > LONGEST
    assert run_dir.status()["promoted_model"] == result["promoted_model"]


def test_a_crash_leaves_the_run_interrupted_with_its_last_checkpoint(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run", provider_options={"fail_at_step": 8})
    crash(run_dir, 8)
    result = read(run_dir.result_path)
    assert result["phase"] == "interrupted"
    assert result["error"]["type"] == "FakeCrash"
    assert result["checkpoints"]["last_training"]["step"] == 7
    state = run_dir.load_state()
    point = state.last_resume_point()
    assert point is not None
    assert (point.epoch, point.step, point.steps_in_epoch) == (2, 7, 3)


def test_an_unexpected_error_is_redacted_and_reported_as_external(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    redaction.register_secret("sk-synthetic-secret-value")

    def raises(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("upstream said sk-synthetic-secret-value")

    fake = provider()
    fake.backend().open_session = raises  # type: ignore[method-assign]
    try:
        with pytest.raises(ExternalError, match="RuntimeError") as info:
            start(run_dir, fake)
    finally:
        redaction.clear_secrets()
    assert "sk-synthetic" not in str(info.value)
    assert info.value.hint is not None and "resum" in info.value.hint
    state = run_dir.load_state()
    assert state.phase == "interrupted"
    assert state.error is not None and "sk-synthetic" not in state.error["message"]


# --- resume -----------------------------------------------------------------------------


@pytest.mark.parametrize("fail_at_step", [6, 8, 9])
def test_resume_after_a_crash_matches_an_uninterrupted_run(
    tmp_path: Path, fail_at_step: int
) -> None:
    unbroken = start(new_run(tmp_path / "unbroken"))
    run_dir = new_run(tmp_path / "run", provider_options={"fail_at_step": fail_at_step})
    crash(run_dir, fail_at_step)
    fake, clock = provider(), Clock()
    result = resume(run_dir, fake, clock)
    assert result["phase"] == "completed"
    assert epoch_metrics(result) == epoch_metrics(unbroken)
    assert epoch_metrics({"epochs": read(run_dir.epochs_path)}) == epoch_metrics(unbroken)
    assert result["best_epoch"] == unbroken["best_epoch"]
    assert result["best_val_loss"] == unbroken["best_val_loss"]
    assert result["promoted_model"] == unbroken["promoted_model"]
    # The resumed session started from the last training checkpoint before the crash, and
    # trained exactly the steps after it.
    [settings] = fake.backend().session_settings
    since = settings["resume_from"]
    assert since.kind == "training" and since.step < fail_at_step
    steps = [event["step"] for event in clock.events if event["type"] == "step"]
    assert steps == list(range(since.step + 1, 13))
    # The best epoch (3) was saved in the resumed session, so it is promoted as is.
    assert result["deployable_checkpoint"] == result["checkpoints"]["best_sampler"]
    assert result["deployable_checkpoint"]["created_in_session"] == fake.backend().opened[-1]
    assert not [ref for ref in fake.backend().saved if ref.name.startswith("r")]
    assert settings["max_seq_len"] == LONGEST


def test_a_best_epoch_from_the_crashed_session_is_saved_again_and_promoted(
    tmp_path: Path,
) -> None:
    options = {"fail_at_step": 6, "val_curve": [1.0, 2.0, 3.0]}
    run_dir = new_run(tmp_path / "run", provider_options=options)
    crash(run_dir, 6)
    fake = provider()
    result = resume(run_dir, fake)
    assert result["best_epoch"] == 1
    best = result["checkpoints"]["best_sampler"]
    deployable = result["deployable_checkpoint"]
    assert deployable["name"] == f"r1s4-{best['name'].split('-')[1]}"
    assert deployable["created_in_session"] == fake.backend().opened[-1]
    assert best["created_in_session"] != deployable["created_in_session"]
    assert result["promoted_model"]["model_id"] == RUN_ID
    assert deployable in read(run_dir.state_path)["checkpoints"]


def test_resume_needs_a_backend_that_can_resume(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run", provider_options={"fail_at_step": 6})
    crash(run_dir, 6)
    with pytest.raises(UsageError, match="cannot resume"):
        resume(run_dir, provider(supports_resume=False))
    assert run_dir.load_state().phase == "interrupted"


@pytest.mark.parametrize("phase", ["completed", "created"])
def test_resume_refuses_a_run_that_is_not_interrupted(tmp_path: Path, phase: str) -> None:
    run_dir = new_run(tmp_path / "run")
    if phase == "completed":
        start(run_dir)
    with pytest.raises(UsageError, match=f"it is {phase}") as info:
        resume(run_dir)
    assert info.value.hint is not None
    assert run_dir.load_state().phase == phase


def test_a_run_left_training_by_a_dead_process_resumes(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run", provider_options={"fail_at_step": 6})
    crash(run_dir, 6)
    state = run_dir.load_state()
    state.phase = "training"  # as if the process had been killed before recording anything
    state.save()
    with DirectoryLock(run_dir.path), pytest.raises(StateError, match="another"):
        resume(run_dir)  # a live trainer would hold the lock
    result = resume(run_dir)
    assert result["phase"] == "completed"
    assert [e["epoch"] for e in result["epochs"]] == [1, 2, 3]


def test_a_cancelled_run_resumes_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = new_run(tmp_path / "run")
    clock = Clock()
    cancel = [False]

    def on_event(event: dict[str, Any]) -> None:
        clock.on_event(event)
        cancel[0] = cancel[0] or event["type"] == "epoch"

    # Stands in for a cancel request (``RunState.request_cancel``) after the first epoch.
    monkeypatch.setattr(RunState, "cancel_requested", lambda self: cancel[0])
    result = start_training(
        provider(),
        run_dir,
        train_rows=TRAIN,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=on_event,
    )
    monkeypatch.undo()
    assert (result["phase"], result["stopped_reason"]) == ("cancelled", "cancelled")
    assert result["best_epoch"] == 1
    assert result["promoted_model"] is not None
    resumed = resume(run_dir)
    assert (resumed["phase"], resumed["stopped_reason"]) == ("completed", "completed")
    assert [e["epoch"] for e in resumed["epochs"]] == [1, 2, 3]


def test_without_validation_a_resumed_run_stays_on_the_loss_curve(tmp_path: Path) -> None:
    # Epoch 2 is no new best; the crash in epoch 3 resumes from step 7's timed checkpoint.
    options = {"loss_curve": [2.0, 2.5, 1.1], "fail_at_step": 10}
    run_dir = new_run(tmp_path / "run", provider_options=options)
    clock = Clock()
    with pytest.raises(FakeCrash, match="fail_at_step"):
        start_training(
            provider(),
            run_dir,
            train_rows=TRAIN,
            validation_rows=[],
            clock=clock,
            on_event=clock.on_event,
        )
    clock = Clock()
    result = resume_training(
        provider(),
        run_dir,
        train_rows=TRAIN,
        validation_rows=[],
        clock=clock,
        on_event=clock.on_event,
    )
    assert [epoch["train_loss"] for epoch in result["epochs"]] == [2.0, 2.5, 1.1]
    assert (result["stopped_reason"], result["best_epoch"]) == ("completed", 3)


# --- spend across attempts --------------------------------------------------------------


def test_a_resume_counts_the_time_spent_after_the_last_checkpoint(tmp_path: Path) -> None:
    # Only epoch-end checkpoints. 4-minute steps: the crash after step 8 comes 16 minutes
    # after epoch 1's checkpoint (at 16 minutes), and steps 5 to 8 are trained again.
    run_dir = new_run(tmp_path / "run", checkpoint_every_minutes=1000)
    with pytest.raises(ExternalError, match="simulated crash"):
        start(run_dir, clock=CrashingClock(crash_at=8))
    point = run_dir.load_state().last_resume_point()
    assert point is not None
    assert (point.step, point.elapsed_seconds) == (4, 16 * 60)
    assert read(run_dir.state_path)["elapsed_seconds"] == 32 * 60
    assert read(run_dir.result_path)["elapsed_seconds"] == 32 * 60
    result = resume(run_dir)
    assert result["phase"] == "completed"
    assert result["elapsed_seconds"] == (32 + 8 * 4) * 60  # not 16 + 8 * 4
    assert read(run_dir.state_path)["elapsed_seconds"] == result["elapsed_seconds"]


def test_repeated_crashes_after_the_last_checkpoint_stop_at_the_spend_cap(
    tmp_path: Path,
) -> None:
    # 1 USD a minute; training stops at 40 USD (50 less 20%). Every attempt crashes after
    # step 8, 16 minutes past epoch 1's checkpoint. Were only the checkpoint's 16 minutes
    # restored, every resume would crash there again, paying another 16 minutes each time.
    run_dir = new_run(
        tmp_path / "run",
        max_spend_usd=50.0,
        hourly_rate_usd=60.0,
        spend_reserve_fraction=0.2,
        checkpoint_every_minutes=1000,
    )
    with pytest.raises(ExternalError, match="simulated crash"):
        start(run_dir, clock=CrashingClock(crash_at=8))
    crashed = 0
    for _ in range(3):
        try:
            result = resume(run_dir, clock=CrashingClock(crash_at=8))
            break
        except ExternalError:
            crashed += 1
    # The resume starts at 32 minutes; steps 5 and 6 reach 40, before the crash at step 8.
    assert crashed == 0
    assert (result["phase"], result["stopped_reason"]) == ("completed", "budget_exhausted")
    assert result["elapsed_seconds"] == 40 * 60
    assert result["elapsed_seconds"] <= 50 * 60
    assert "less the 20% kept" in result["warnings"][0]
    assert result["checkpoints"]["last_training"]["step"] == 6  # paid from the reserve
    assert result["promoted_model"]["model_id"] == RUN_ID


# --- promotion --------------------------------------------------------------------------


def test_a_spend_stop_keeps_the_reserve_for_its_checkpoint_and_promotes_the_best(
    tmp_path: Path,
) -> None:
    # 1 USD a minute and 4-minute steps: training stops at 27 USD (30 less 10%), before step
    # 8, and the checkpoint of steps 5 to 7 is saved from the reserve.
    run_dir = new_run(
        tmp_path / "run", max_spend_usd=30.0, hourly_rate_usd=60.0, spend_reserve_fraction=0.1
    )
    result = start(run_dir)
    assert (result["phase"], result["stopped_reason"]) == ("completed", "budget_exhausted")
    assert [e["epoch"] for e in result["epochs"]] == [1]
    assert result["best_epoch"] == 1
    assert result["promoted_model"]["model_id"] == RUN_ID
    assert result["checkpoints"]["last_training"]["step"] == 7
    assert "less the 10% kept" in result["warnings"][-1]


def test_a_promotion_the_spent_reserve_cannot_pay_for_is_retried_on_resume(
    tmp_path: Path,
) -> None:
    # 5-minute steps at 1 USD a minute; training stops at 21.6 USD, the cap is 24 USD.
    run_dir = new_run(
        tmp_path / "run",
        max_spend_usd=24.0,
        hourly_rate_usd=60.0,
        spend_reserve_fraction=0.1,
        provider_options={"fail_at_step": 5},
    )
    with pytest.raises(FakeCrash):
        start(run_dir, clock=Clock(step_minutes=5))  # epoch 1 ends at 20 minutes
    # The resumed attempt starts at 20 minutes; step 5 takes it to 25, past the whole cap,
    # so neither a final checkpoint nor a new sampler checkpoint of epoch 1 can be paid for.
    with pytest.raises(FinalizeFailed, match="did not re-create epoch 1's sampler checkpoint"):
        resume(run_dir, clock=Clock(step_minutes=5))
    failed = read(run_dir.result_path)
    assert (failed["stopped_reason"], failed["promoted_model"]) == ("budget_exhausted", None)
    assert any("did not save a training checkpoint" in w for w in failed["warnings"])
    assert run_dir.load_state().resume_mode == "finalize_only"
    # Retrying only the promotion is the user's call, so it is not held to the budget.
    result = resume(run_dir)
    assert (result["phase"], result["best_epoch"]) == ("completed", 1)
    assert result["promoted_model"]["model_id"] == RUN_ID


def test_without_promotion_the_deployable_checkpoint_is_recorded(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider(supports_promotion=False)
    result = start(run_dir, fake)
    assert result["phase"] == "completed"
    assert result["promoted_model"] is None
    assert result["deployable_checkpoint"] == result["checkpoints"]["best_sampler"]
    assert result["deployable_checkpoint"]["epoch"] == 3
    assert fake.backend().promoted == []


def test_a_failed_promotion_keeps_the_result_and_resume_retries_only_it(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    with pytest.raises(FinalizeFailed, match="finalizing the run failed"):
        start(run_dir, fake)
    failed = read(run_dir.result_path)
    assert failed["phase"] == "failed"
    assert failed["stopped_reason"] == "completed"
    assert failed["best_epoch"] == 3
    assert failed["promoted_model"] is None
    assert failed["finalize_error"]["type"] == "FakeCrash"
    assert failed["error"] is None
    assert len(failed["epochs"]) == 3
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("failed", "finalize_only")
    assert run_dir.status()["resume_mode"] == "finalize_only"

    retry = provider()  # a new process: the best sampler checkpoint's session has ended
    result = resume(run_dir, retry)
    assert (result["phase"], result["stopped_reason"]) == ("completed", "completed")
    assert result["promoted_model"]["model_id"] == RUN_ID
    assert result["finalize_error"] is None
    assert epoch_metrics(result) == epoch_metrics(failed)
    # Only the promotion ran: a session from the best training checkpoint, no training.
    [settings] = retry.backend().session_settings
    assert settings["resume_from"].name == failed["checkpoints"]["best_training"]["name"]
    assert [ref.kind for ref in retry.backend().saved] == ["sampler"]
    assert result["deployable_checkpoint"]["name"].startswith("r3s12-")
    assert run_dir.load_state().resume_mode == "train"


def test_a_finalize_only_retry_that_fails_again_stays_resumable(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    with pytest.raises(FinalizeFailed):
        start(run_dir, fake)
    again = provider()
    again.backend().fail_promotions = 1
    with pytest.raises(FinalizeFailed):
        resume(run_dir, again)
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("failed", "finalize_only")
    assert read(run_dir.result_path)["best_epoch"] == 3
    assert resume(run_dir)["phase"] == "completed"


def test_a_finalize_only_retry_works_without_resume(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    with pytest.raises(FinalizeFailed):
        start(run_dir, fake)
    retry = provider(supports_resume=False)
    result = resume(run_dir, retry)
    assert result["promoted_model"]["model_id"] == RUN_ID
    [settings] = retry.backend().session_settings
    assert settings["resume_from"] is None  # a fresh session, with the best weights loaded


def test_a_failure_ending_the_session_keeps_the_promoted_model(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_session_ends = 1
    result = start(run_dir, fake)
    assert (result["phase"], result["stopped_reason"]) == ("completed", "completed")
    assert result["promoted_model"] == {
        "provider": "fake",
        "model_id": RUN_ID,
        "uri": f"fake://models/{RUN_ID}",
    }
    assert any("ending the training session failed" in w for w in result["warnings"])
    assert read(run_dir.result_path) == result
    assert run_dir.load_state().resume_mode == "train"
    with pytest.raises(UsageError, match="it is completed"):
        resume(run_dir)


def test_a_training_error_is_kept_when_ending_the_session_fails_too(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run", provider_options={"fail_at_step": 8})
    fake = provider()
    fake.backend().fail_session_ends = 1
    with pytest.raises(FakeCrash, match="fail_at_step") as info:
        start(run_dir, fake)
    assert any("ending the session also failed" in note for note in info.value.__notes__)
    state = run_dir.load_state()
    assert state.phase == "interrupted"
    assert state.error is not None and "fail_at_step" in state.error["message"]


def test_a_failed_promotion_is_kept_when_ending_the_session_fails_too(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    fake.backend().fail_session_ends = 1
    with pytest.raises(FinalizeFailed) as info:
        start(run_dir, fake)
    assert any("ending the session also failed" in note for note in info.value.__notes__)
    failed = read(run_dir.result_path)
    assert (failed["phase"], failed["finalize_error"]["type"]) == ("failed", "FakeCrash")
    assert "promoting" in failed["finalize_error"]["message"]
    assert run_dir.load_state().resume_mode == "finalize_only"
    # The retry fails to end its session as well, after promoting: the model is kept.
    retry = provider()
    retry.backend().fail_session_ends = 1
    result = resume(run_dir, retry)
    assert result["phase"] == "completed"
    assert result["promoted_model"]["model_id"] == RUN_ID
    assert any("ending the training session failed" in w for w in result["warnings"])


def test_a_failed_retry_that_cannot_end_its_session_keeps_the_promotion_error(
    tmp_path: Path,
) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    with pytest.raises(FinalizeFailed):
        start(run_dir, fake)
    retry = provider()
    retry.backend().fail_promotions = 1
    retry.backend().fail_session_ends = 1
    with pytest.raises(FinalizeFailed, match="promoting") as info:
        resume(run_dir, retry)
    assert any("ending the session also failed" in note for note in info.value.__notes__)
    assert info.value.result.best_epoch == 3
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("failed", "finalize_only")
    assert resume(run_dir)["promoted_model"]["model_id"] == RUN_ID


@pytest.mark.parametrize(
    ("interrupt", "raised"),
    [(KeyboardInterrupt, KeyboardInterrupt), (TerminationRequested, TrainingTerminated)],
)
def test_an_interrupt_during_promotion_is_raised_as_an_interrupt(
    tmp_path: Path, interrupt: type[BaseException], raised: type[BaseException]
) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    fake.backend().promotion_error = interrupt()
    with pytest.raises(raised):
        start(run_dir, fake)
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("interrupted", "finalize_only")
    failed = read(run_dir.result_path)
    assert (failed["stopped_reason"], failed["promoted_model"]) == ("completed", None)
    result = resume(run_dir)
    assert (result["phase"], result["promoted_model"]["model_id"]) == ("completed", RUN_ID)


def test_a_failed_promotion_whose_result_was_not_written_resumes_by_training(
    tmp_path: Path,
) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    fake.backend().fail_promotions = 1
    write_result = run_dir.write_result

    def write_fails_once(result: Any) -> None:
        run_dir.write_result = write_result  # type: ignore[method-assign]
        raise OSError("disk full")

    run_dir.write_result = write_fails_once  # type: ignore[method-assign]
    with pytest.raises(FinalizeFailed):
        start(run_dir, fake)
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("failed", "train")
    retry, clock = provider(), Clock()
    result = resume(run_dir, retry, clock)
    assert (result["phase"], result["stopped_reason"]) == ("completed", "completed")
    assert result["promoted_model"]["model_id"] == RUN_ID
    assert [e["epoch"] for e in result["epochs"]] == [1, 2, 3]
    assert not [event for event in clock.events if event["type"] == "step"]  # nothing trained


def test_the_model_id_is_the_run_id_made_safe() -> None:
    assert model_id_for_run(RUN_ID) == RUN_ID
    assert model_id_for_run("x--y") == "x--y"
    assert re.fullmatch(r"ft-2026-run-[0-9a-f]{8}", model_id_for_run("2026-run"))
    long_id = "a" * 64
    derived = model_id_for_run(long_id)
    assert len(derived) <= 63 and re.fullmatch("a" * 54 + r"-[0-9a-f]{8}", derived)
    assert model_id_for_run(long_id) == derived
    assert re.fullmatch(r"run-[0-9a-f]{8}", model_id_for_run("run-"))


@pytest.mark.parametrize(
    ("first", "second"),
    [("x--y", "x-y"), ("2026-run", "ft-2026-run"), ("run-", "run"), ("a" * 64, "a" * 63)],
)
def test_distinct_run_ids_get_distinct_model_ids(first: str, second: str) -> None:
    assert model_id_for_run(first) != model_id_for_run(second)


@pytest.mark.parametrize(
    "run_id", [RUN_ID, "2026-run", "a" * 64, "run-", "x--y", "9" * 64, "ft-" + "a-" * 31]
)
def test_the_model_id_fits_the_promotion_rules(run_id: str) -> None:
    assert MODEL_ID_PATTERN.fullmatch(model_id_for_run(run_id))


# --- after_finalize ------------------------------------------------------------------------


def test_after_finalize_runs_in_the_open_session_once_promoted(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path / "run")
    fake = provider()
    seen: list[dict[str, Any]] = []

    def after(session: Any, result: Any, cancel: Any, spend_guard: Any) -> None:
        backend = fake.backend()
        seen.append(
            {
                "live": backend.is_live(session.session_id),
                "promoted": result.promoted_model is not None,
                "cancel": cancel.is_set(),
            }
        )

    clock = Clock()
    result = start_training(
        fake,
        run_dir,
        train_rows=TRAIN,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=clock.on_event,
        after_finalize=after,
    )
    assert seen == [{"live": True, "promoted": True, "cancel": False}]
    assert result["phase"] == "completed"
    assert result["warnings"] == []


@pytest.mark.parametrize("capped", [False, True])
def test_after_finalize_gets_the_run_s_spend_guard(tmp_path: Path, capped: bool) -> None:
    spend = {"max_spend_usd": 10.0, "hourly_rate_usd": 1.0} if capped else {}
    run_dir = new_run(tmp_path / "run", **spend)
    clock = Clock()
    seen: list[Any] = []

    def after(session: Any, result: Any, cancel: Any, spend_guard: Any) -> None:
        seen.append(None if spend_guard is None else spend_guard())
        if spend_guard is not None:
            clock.now += 10 * 3600  # the session's time still counts, reserve included
            seen.append(spend_guard())

    result = start_training(
        provider(),
        run_dir,
        train_rows=TRAIN,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=clock.on_event,
        after_finalize=after,
    )
    assert result["phase"] == "completed"
    if not capped:
        assert seen == [None]
        return
    assert seen[0] is None
    assert "max_spend_usd 10.0" in seen[1]


@pytest.mark.parametrize("error", [RuntimeError("replay broke"), KeyboardInterrupt()])
def test_a_failing_after_finalize_only_warns(tmp_path: Path, error: BaseException) -> None:
    run_dir = new_run(tmp_path / "run")

    def after(session: Any, result: Any, cancel: Any, spend_guard: Any) -> None:
        raise error

    clock = Clock()
    result = start_training(
        provider(),
        run_dir,
        train_rows=TRAIN,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=clock.on_event,
        after_finalize=after,
    )
    assert result["phase"] == "completed"
    assert result["promoted_model"] is not None
    assert any("after training failed" in warning for warning in result["warnings"])
    assert run_dir.load_state().phase == "completed"


def test_the_run_is_recorded_finished_before_after_finalize_runs(tmp_path: Path) -> None:
    """A process killed during the hook leaves a completed run with its promoted model."""
    run_dir = new_run(tmp_path / "run")
    seen: list[dict[str, Any]] = []

    def after(session: Any, result: Any, cancel: Any, spend_guard: Any) -> None:
        stored = run_dir.read_result() or {}
        seen.append(
            {
                "phase": run_dir.load_state().phase,
                "result_phase": stored.get("phase"),
                "promoted": stored.get("promoted_model") is not None,
            }
        )
        raise RuntimeError("replay broke")

    clock = Clock()
    result = start_training(
        provider(),
        run_dir,
        train_rows=TRAIN,
        validation_rows=VALIDATION,
        clock=clock,
        on_event=clock.on_event,
        after_finalize=after,
    )
    assert seen == [{"phase": "completed", "result_phase": "completed", "promoted": True}]
    # Its warning is recorded once the hook returns.
    stored = run_dir.read_result() or {}
    assert any("after training failed" in warning for warning in stored["warnings"])
    assert stored["warnings"] == result["warnings"]
    # A finished run is not resumable, so nothing would train it again.
    with pytest.raises(UsageError, match="completed"):
        resume_training(provider(), run_dir, train_rows=TRAIN, validation_rows=VALIDATION)
