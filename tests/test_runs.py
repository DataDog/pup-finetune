# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import signal
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from pup_finetune.data.dataset_dir import DatasetDir, DatasetSettings
from pup_finetune.data.schema import TrainingRow, validate_row
from pup_finetune.data.split import SplitParams
from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.providers.base import LoraConfig, TrainSpec
from pup_finetune.providers.fake import (
    FAKE_PROFILE,
    FakeBackend,
    FakeCrash,
    FakeOptions,
    FakeRenderer,
)
from pup_finetune.runs import (
    RunDir,
    RunState,
    TerminationRequested,
    TrainingTerminated,
    checkpoint_namer,
    dataset_content_hash,
    execute_run,
    new_run_id,
)
from pup_finetune.storage import DirectoryLock
from pup_finetune.training import Budget, EpochMetrics, FinalizeFailed, TrainResult, run_training
from tests.fakes.training import SESSION, RecordingSession, row_examples

NOW = datetime(2026, 10, 6, 12, 30, 45, tzinfo=UTC)
SPEC = TrainSpec(max_epochs=4, batch_size=2, patience=1)


def now() -> datetime:
    return NOW


def create(path: Path, **overrides: Any) -> RunDir:
    fields: dict[str, Any] = {
        "provider": "fake",
        "profile": FAKE_PROFILE,
        "spec": SPEC,
        "dataset_dir": path.parent / "dataset",
        "dataset_sha256": "d" * 64,
        "rendering": FakeRenderer().identity(),
        "run_id": "ft-test-run",
        "now": now,
    }
    return RunDir.create(path, **(fields | overrides))


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def fake_training(
    options: FakeOptions | None = None, *, spec: TrainSpec = SPEC, run_id: str = "ft-test-run"
) -> Any:
    """A ``train`` callable for :func:`execute_run` that trains on the fake provider."""

    def train(state: RunState) -> TrainResult:
        backend = FakeBackend(options)
        with backend.open_session(FAKE_PROFILE, LoraConfig(rank=8), **SESSION) as session:
            return run_training(
                session,
                train_rows=row_examples(4),
                validation_rows=row_examples(2, prefix="val"),
                spec=spec,
                state=state,
                budget=Budget.from_spec(spec),
                checkpoint_namer=checkpoint_namer(run_id),
                max_checkpoint_name_len=backend.max_checkpoint_name_len,
            )

    return train


# --- creating a run ------------------------------------------------------------------------


def test_create_writes_the_manifest_and_initial_state(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    manifest = read(run_dir.run_path)
    assert manifest["run_id"] == "ft-test-run"
    assert manifest["provider"] == "fake"
    assert manifest["profile"] == dataclasses.asdict(FAKE_PROFILE)
    assert manifest["spec"] == SPEC.model_dump(mode="json")
    assert manifest["dataset_dir"] == str((tmp_path / "dataset").resolve())
    assert manifest["dataset_sha256"] == "d" * 64
    assert manifest["rendering"] == FakeRenderer().identity()
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["settings_hash"])
    assert manifest["created_at"] == "2026-10-06T12:30:45+00:00"
    assert run_dir.spec() == SPEC
    state = read(run_dir.state_path)
    assert state["phase"] == "created"
    assert (state["epoch"], state["step"], state["epochs"]) == (0, 0, 0)
    assert state["checkpoints"] == []
    assert state["best"] is None


def test_the_settings_hash_changes_with_the_settings(tmp_path: Path) -> None:
    first = create(tmp_path / "a").manifest()["settings_hash"]
    same = create(tmp_path / "b", run_id="ft-other").manifest()["settings_hash"]
    other = create(tmp_path / "c", spec=SPEC.model_copy(update={"seed": 1})).manifest()
    assert first == same
    assert other["settings_hash"] != first


def test_the_run_dir_is_private(tmp_path: Path) -> None:
    previous = os.umask(0o002)
    try:
        run_dir = create(tmp_path / "parent" / "run")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(run_dir.path.stat().st_mode) == 0o700
    for name in ("run.json", "state.json"):
        assert stat.S_IMODE((run_dir.path / name).stat().st_mode) == 0o600


def test_an_empty_existing_directory_is_taken_over(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir(mode=0o755)
    create(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_the_run_dir_must_be_new_or_empty(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    (target / "notes.txt").write_text("hello")
    with pytest.raises(StateError, match="is not empty"):
        create(target)
    with pytest.raises(StateError, match="is not empty"):
        create(create(tmp_path / "used").path)


def test_a_file_is_not_a_run_dir(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.write_text("")
    with pytest.raises(StateError, match="not a directory"):
        create(target)


def test_a_generated_run_id_has_the_documented_shape(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run", run_id=None)
    assert re.fullmatch(r"ft-20261006123045-[0-9a-f]{6}", run_dir.run_id)
    assert re.fullmatch(r"ft-\d{14}-[0-9a-f]{6}", new_run_id())
    assert new_run_id(NOW) != new_run_id(NOW)


@pytest.mark.parametrize("run_id", ["", "Upper", "-dash", "a/b", "x" * 65])
def test_invalid_run_ids_are_refused(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(UsageError, match="invalid run id"):
        create(tmp_path / "run", run_id=run_id)
    assert not (tmp_path / "run").exists()


def test_checkpoint_names_fit_seventeen_characters() -> None:
    name = checkpoint_namer("ft-20261006123045-abcdef")
    assert re.fullmatch(r"t3s120-[0-9a-f]{4}", name("training", 3, 120))
    assert re.fullmatch(r"s3s120-[0-9a-f]{4}", name("sampler", 3, 120))
    assert len(name("training", 99, 99999)) <= 17
    assert name("training", 1, 1) == checkpoint_namer("ft-20261006123045-abcdef")("training", 1, 1)
    assert name("training", 1, 1) != checkpoint_namer("ft-other")("training", 1, 1)


def _row(row_id: str, text: str) -> TrainingRow:
    return validate_row(
        {
            "id": row_id,
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"Reply to {text}"},
            ],
        }
    )


def test_dataset_content_hash_follows_the_rows(tmp_path: Path) -> None:
    settings = DatasetSettings("import", SplitParams(42, 0.1, 0.1))
    dataset = DatasetDir(tmp_path / "ds")
    dataset.add([_row("a", "one"), _row("b", "two")], settings=settings)
    first = dataset_content_hash(dataset)
    assert first == dataset_content_hash(DatasetDir(tmp_path / "ds"))
    dataset.add([_row("c", "three")], settings=settings)
    assert dataset_content_hash(dataset) != first


# --- run state -----------------------------------------------------------------------------


def test_state_transitions(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    state = run_dir.load_state()
    state.start()
    on_disk = read(run_dir.state_path)
    assert (on_disk["phase"], on_disk["pid"]) == ("training", os.getpid())
    assert on_disk["heartbeat_at"]
    state.transition("completed")
    assert run_dir.load_state().phase == "completed"
    with pytest.raises(StateError, match="from completed to training"):
        state.transition("training")


def test_the_heartbeat_advances_on_every_save(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    ticks = iter(datetime(2026, 10, 6, 13, 0, second, tzinfo=UTC) for second in range(10))
    state = run_dir.load_state(now=lambda: next(ticks))
    state.record_step(epoch=1, step=1)
    first = read(run_dir.state_path)["heartbeat_at"]
    state.record_step(epoch=1, step=2)
    second = read(run_dir.state_path)["heartbeat_at"]
    assert (first, second) == ("2026-10-06T13:00:00+00:00", "2026-10-06T13:00:01+00:00")


def test_warnings_are_saved_as_they_occur(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    state = run_dir.load_state()
    state.record_warning("the validation split is empty")
    assert run_dir.load_state().warnings == ["the validation split is empty"]
    assert run_dir.status()["warnings"] == ["the validation split is empty"]


@pytest.mark.parametrize(
    ("path", "bad"),
    [(("training",), "created"), (("training",), "training"), ((), "completed"), ((), "cancelled")],
)
def test_invalid_state_transitions_are_refused(
    tmp_path: Path, path: tuple[str, ...], bad: str
) -> None:
    state = create(tmp_path / "run").load_state()
    for phase in path:
        state.transition(phase)  # type: ignore[arg-type]
    with pytest.raises(StateError, match="a run cannot go from"):
        state.transition(bad)  # type: ignore[arg-type]


def test_an_unknown_phase_on_disk_is_an_error(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    run_dir.state_path.write_text(json.dumps({"phase": "exploded"}))
    with pytest.raises(StateError, match="unknown phase"):
        run_dir.load_state()


def test_reading_a_missing_run_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="does not exist"):
        RunDir(tmp_path / "nothing").status()


# --- executing a run -----------------------------------------------------------------------


def test_a_completed_run_writes_state_epochs_and_result(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    result = execute_run(run_dir, fake_training(FakeOptions(val_curve=[2.0, 1.5, 1.8])), now=now)
    assert read(run_dir.result_path) == result
    assert result["run_id"] == "ft-test-run"
    assert result["phase"] == "completed"
    assert result["stopped_reason"] == "early_stopped"
    assert (result["best_epoch"], result["best_val_loss"]) == (2, 1.5)
    assert result["error"] is None
    assert result["finished_at"] == "2026-10-06T12:30:45+00:00"
    epochs = read(run_dir.epochs_path)
    assert [e["epoch"] for e in epochs] == [1, 2, 3]
    assert epochs == result["epochs"]
    state = read(run_dir.state_path)
    assert state["phase"] == "completed"
    assert (state["epoch"], state["step"], state["epochs"]) == (3, 6, 3)
    assert [(c["kind"], c["epoch"]) for c in state["checkpoints"]] == [
        ("training", 1),
        ("sampler", 1),
        ("training", 2),
        ("sampler", 2),
        ("training", 3),
    ]
    assert state["best"]["epoch"] == 2
    assert state["best"]["sampler_ckpt"] == result["checkpoints"]["best_sampler"]
    for name in ("state.json", "epochs.json", "result.json"):
        assert stat.S_IMODE((run_dir.path / name).stat().st_mode) == 0o600


def test_the_directory_is_locked_while_training(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    inner = fake_training()

    def train(state: RunState) -> TrainResult:
        with (
            pytest.raises(StateError, match="another pup-finetune process"),
            DirectoryLock(run_dir.path),
        ):
            pass
        return inner(state)

    execute_run(run_dir, train)
    with DirectoryLock(run_dir.path):
        pass


def test_a_cancelled_run_ends_cancelled(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    inner = fake_training()

    def train(state: RunState) -> TrainResult:
        state.request_cancel()
        return inner(state)

    result = execute_run(run_dir, train)
    assert (result["phase"], result["stopped_reason"]) == ("cancelled", "cancelled")
    assert run_dir.load_state().phase == "cancelled"


def test_a_crash_interrupts_the_run_and_records_the_error(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    with pytest.raises(FakeCrash, match="fail_at_step"):
        execute_run(run_dir, fake_training(FakeOptions(fail_at_step=3)))
    result = read(run_dir.result_path)
    assert result["phase"] == "interrupted"
    assert result["stopped_reason"] is None
    assert result["error"]["type"] == "FakeCrash"
    assert "fail_at_step" in result["error"]["message"]
    assert (result["promoted_model"], result["finalize_error"]) == (None, None)
    # Epoch 1 finished before the crash, so its checkpoints are reported.
    assert result["best_epoch"] == 1
    assert result["checkpoints"]["last_training"]["epoch"] == 1
    assert [e["epoch"] for e in result["epochs"]] == [1]
    state = read(run_dir.state_path)
    assert state["phase"] == "interrupted"
    assert state["error"] == result["error"]
    assert state["resume_point"]["checkpoint"] == result["checkpoints"]["last_training"]


def test_result_json_is_written_on_keyboard_interrupt(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")

    def train(state: RunState) -> TrainResult:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_run(run_dir, train)
    result = read(run_dir.result_path)
    assert result["phase"] == "interrupted"
    assert result["error"]["type"] == "KeyboardInterrupt"
    assert (result["best_epoch"], result["checkpoints"]["last_training"], result["epochs"]) == (
        None,
        None,
        [],
    )
    assert run_dir.load_state().phase == "interrupted"


def test_sigterm_interrupts_the_run_and_still_writes_the_result(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    before = signal.getsignal(signal.SIGTERM)

    def train(state: RunState) -> TrainResult:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM should have interrupted training")

    with pytest.raises(TrainingTerminated, match="received SIGTERM") as info:
        execute_run(run_dir, train)
    assert isinstance(info.value.__cause__, TerminationRequested)
    assert read(run_dir.result_path)["phase"] == "interrupted"
    assert run_dir.load_state().phase == "interrupted"
    assert signal.getsignal(signal.SIGTERM) == before


def test_termination_notes_become_the_hint(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")

    def train(state: RunState) -> TrainResult:
        error = TerminationRequested()
        error.add_note("step 3 also failed: RuntimeError: boom")
        raise error

    with pytest.raises(TrainingTerminated) as info:
        execute_run(run_dir, train)
    assert info.value.hint == "step 3 also failed: RuntimeError: boom"
    assert read(run_dir.result_path)["error"]["notes"] == ["step 3 also failed: RuntimeError: boom"]
    assert run_dir.load_state().phase == "interrupted"


def _failing_write(run_dir: RunDir) -> None:
    def write_result(result: Any) -> None:
        raise OSError("disk full")

    run_dir.write_result = write_result  # type: ignore[method-assign]


def test_a_failed_result_write_is_noted_on_the_training_error(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    _failing_write(run_dir)

    def train(state: RunState) -> TrainResult:
        raise RuntimeError("training broke")

    with pytest.raises(ExternalError, match="RuntimeError: training broke") as info:
        execute_run(run_dir, train)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "could not record the run's result: disk full" in info.value.__cause__.__notes__
    assert not run_dir.result_path.exists()
    state = run_dir.load_state()
    assert state.phase == "interrupted"
    assert state.error is not None and state.error["message"] == "training broke"


def test_a_failed_result_write_after_training_fails_the_run(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    _failing_write(run_dir)
    with pytest.raises(StateError, match="disk full") as info:
        execute_run(run_dir, fake_training())
    assert isinstance(info.value.__cause__, OSError)
    state = run_dir.load_state()
    assert state.phase == "failed"
    assert state.error is not None and state.error["type"] == "OSError"


def test_an_unreadable_epochs_file_does_not_stop_the_result(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")

    def train(state: RunState) -> TrainResult:
        run_dir.epochs_path.write_text('"not a list"')
        raise RuntimeError("training broke")

    with pytest.raises(ExternalError, match="training broke"):
        execute_run(run_dir, train)
    result = read(run_dir.result_path)
    assert result["epochs"] == []
    assert any("could not read epochs.json" in warning for warning in result["warnings"])
    assert run_dir.load_state().phase == "interrupted"


def test_signals_are_ignored_while_the_result_is_written(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    seen: dict[str, Any] = {}
    write_result = run_dir.write_result

    def spy(result: Any) -> None:
        seen["sigint"] = signal.getsignal(signal.SIGINT)
        seen["sigterm"] = signal.getsignal(signal.SIGTERM)
        write_result(result)

    run_dir.write_result = spy  # type: ignore[method-assign]
    execute_run(run_dir, fake_training())
    assert seen == {"sigint": signal.SIG_IGN, "sigterm": signal.SIG_IGN}
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_run_that_already_ran_cannot_run_again(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    execute_run(run_dir, fake_training())
    with pytest.raises(StateError, match="from completed to training"):
        execute_run(run_dir, fake_training())


def test_status_summarizes_the_run(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    before = run_dir.status()
    assert (before["phase"], before["stopped_reason"], before["best"]) == ("created", None, None)
    execute_run(run_dir, fake_training(FakeOptions(val_curve=[2.0, 1.5, 1.8])))
    status = run_dir.status()
    assert status["run_id"] == "ft-test-run"
    assert (status["provider"], status["model"]) == ("fake", FAKE_PROFILE.alias)
    assert (status["phase"], status["stopped_reason"]) == ("completed", "early_stopped")
    assert (status["epoch"], status["epochs_completed"], status["max_epochs"]) == (3, 3, 4)
    assert status["best"]["epoch"] == 2
    assert status["pid"] == os.getpid()
    assert status["error"] is None


# --- resuming and finite numbers -----------------------------------------------------------


def test_only_interrupted_failed_or_cancelled_runs_resume(tmp_path: Path) -> None:
    for phase in ("interrupted", "failed", "cancelled"):
        state = create(tmp_path / phase).load_state()
        state.start()
        state.transition(phase)  # type: ignore[arg-type]
        state.resume()
        assert (state.phase, state.error) == ("training", None)
    fresh = create(tmp_path / "fresh").load_state()
    with pytest.raises(UsageError, match="it is created"):
        fresh.resume()
    fresh.start()
    with pytest.raises(UsageError, match="it is training"):
        fresh.resume()


def test_a_run_that_was_interrupted_must_resume_not_start(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    with pytest.raises(FakeCrash):
        execute_run(run_dir, fake_training(FakeOptions(fail_at_step=3)))
    with pytest.raises(StateError, match="from interrupted to training") as info:
        execute_run(run_dir, fake_training())
    assert info.value.hint is not None and "resume" in info.value.hint
    with pytest.raises(UsageError, match="it is created"):
        execute_run(create(tmp_path / "new"), fake_training(), resume=True)


def test_non_finite_losses_are_stored_as_null(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    nan_epoch = EpochMetrics(
        epoch=1,
        steps=2,
        train_loss=math.nan,
        val_loss=math.inf,
        train_tokens=10,
        new_best=True,
        training_ckpt=None,
        sampler_ckpt=None,
    )

    def train(state: RunState) -> TrainResult:
        state.record_epoch(nan_epoch, best=nan_epoch)
        return TrainResult(
            best_epoch=1,
            best_val_loss=math.nan,
            stopped_reason="completed",
            epochs=[nan_epoch],
            best_sampler_ckpt=None,
            best_training_ckpt=None,
            last_training_ckpt=None,
            warnings=[],
        )

    result = execute_run(run_dir, train)
    on_disk = read(run_dir.result_path)
    assert on_disk == result
    assert on_disk["best_val_loss"] is None
    assert (on_disk["epochs"][0]["train_loss"], on_disk["epochs"][0]["val_loss"]) == (None, None)
    assert read(run_dir.epochs_path)[0]["val_loss"] is None
    assert read(run_dir.state_path)["best"]["val_loss"] is None


def test_a_finalize_failure_keeps_the_result_and_marks_finalize_only(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    inner = fake_training()

    def train(state: RunState) -> TrainResult:
        raise FinalizeFailed(inner(state), RuntimeError("promotion broke"))

    with pytest.raises(FinalizeFailed):
        execute_run(run_dir, train)
    result = read(run_dir.result_path)
    assert (result["phase"], result["stopped_reason"]) == ("failed", "completed")
    assert result["finalize_error"]["message"] == "promotion broke"
    assert result["error"] is None
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("failed", "finalize_only")
    recovered = TrainResult.from_json(result)
    assert recovered.best_epoch == result["best_epoch"]
    assert [e.epoch for e in recovered.epochs] == [e["epoch"] for e in result["epochs"]]
    assert dataclasses.asdict(recovered.best_sampler_ckpt) == result["checkpoints"]["best_sampler"]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("interrupt", "raised"),
    [(KeyboardInterrupt, KeyboardInterrupt), (TerminationRequested, TrainingTerminated)],
)
def test_an_interrupted_finalize_is_reported_as_the_interrupt(
    tmp_path: Path, interrupt: type[BaseException], raised: type[BaseException]
) -> None:
    run_dir = create(tmp_path / "run")
    inner = fake_training()

    def train(state: RunState) -> TrainResult:
        result = inner(state)
        try:
            raise interrupt
        except BaseException as exc:
            raise FinalizeFailed(result, exc) from exc

    with pytest.raises(raised):
        execute_run(run_dir, train)
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("interrupted", "finalize_only")
    assert read(run_dir.result_path)["finalize_error"]["type"] == interrupt.__name__


def test_a_finalize_failure_that_cannot_be_recorded_resumes_by_training(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    _failing_write(run_dir)
    inner = fake_training()

    def train(state: RunState) -> TrainResult:
        raise FinalizeFailed(inner(state), RuntimeError("promotion broke"))

    with pytest.raises(FinalizeFailed) as info:
        execute_run(run_dir, train)
    assert info.value.hint is not None and "last training checkpoint" in info.value.hint
    assert not run_dir.result_path.exists()
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("failed", "train")


def _finalize_only(run_dir: RunDir) -> dict[str, Any]:
    """Leave ``run_dir`` failed after training, with only the promotion to retry."""
    inner = fake_training()

    def train(state: RunState) -> TrainResult:
        raise FinalizeFailed(inner(state), RuntimeError("promotion broke"))

    with pytest.raises(FinalizeFailed):
        execute_run(run_dir, train)
    assert run_dir.load_state().resume_mode == "finalize_only"
    return cast(dict[str, Any], read(run_dir.result_path))


def test_a_retry_that_stops_before_promoting_keeps_the_result_to_retry(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    before = _finalize_only(run_dir)

    def retry(state: RunState) -> TrainResult:
        raise KeyboardInterrupt  # e.g. while reading result.json

    with pytest.raises(KeyboardInterrupt):
        execute_run(run_dir, retry, resume=True)
    assert read(run_dir.result_path) == before
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("interrupted", "finalize_only")
    assert state.error is not None and state.error["type"] == "KeyboardInterrupt"


def test_a_retry_from_a_damaged_result_falls_back_to_training(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")
    _finalize_only(run_dir)
    run_dir.result_path.write_text('{"stopped_reason": "completed", "best_epoch": "two"}')

    def retry(state: RunState) -> TrainResult:
        return TrainResult.from_json(run_dir.read_result() or {})

    with pytest.raises(StateError, match=r"result\.json"):
        execute_run(run_dir, retry, resume=True)
    state = run_dir.load_state()
    assert (state.phase, state.resume_mode) == ("interrupted", "train")


def test_a_dropped_connection_is_an_external_error(tmp_path: Path) -> None:
    run_dir = create(tmp_path / "run")

    def train(state: RunState) -> TrainResult:
        backend = FakeBackend()
        with backend.open_session(FAKE_PROFILE, LoraConfig(rank=8), **SESSION) as inner:
            session = RecordingSession(
                inner, forward_backward_errors={3: ConnectionError("connection reset")}
            )
            return run_training(
                session,
                train_rows=row_examples(4),
                validation_rows=row_examples(2, prefix="val"),
                spec=SPEC,
                state=state,
                budget=Budget(),
                checkpoint_namer=checkpoint_namer("ft-test-run"),
                max_checkpoint_name_len=backend.max_checkpoint_name_len,
            )

    with pytest.raises(ExternalError, match="ConnectionError: connection reset") as info:
        execute_run(run_dir, train)
    assert not isinstance(info.value, StateError)
    assert run_dir.load_state().phase == "interrupted"


def test_train_result_from_json_needs_a_finished_training(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="did not end"):
        TrainResult.from_json({"stopped_reason": None})


@pytest.mark.parametrize(
    "damage",
    [{"best_epoch": "two"}, {"best_val_loss": [1]}, {"epochs": [{"epoch": 1}]}, {"epochs": [3]}],
)
def test_a_damaged_training_result_is_a_state_error(damage: dict[str, Any]) -> None:
    data = {"stopped_reason": "completed", "checkpoints": {}, "epochs": [], **damage}
    with pytest.raises(StateError, match=r"result\.json"):
        TrainResult.from_json(data)
