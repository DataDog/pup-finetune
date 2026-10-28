# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A remote job's state reports: snapshots as log lines, read back and applied to a run."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.errors import StateError
from pup_finetune.providers.base import CheckpointRef, LogLine
from pup_finetune.runs import RunDir, RunState
from pup_finetune.storage import atomic_write_json
from pup_finetune.training.records import EpochMetrics, ResumePoint
from pup_finetune.training.report import (
    MAX_CHUNK,
    Snapshot,
    StateReporter,
    apply_snapshot,
    encode,
    newest_snapshot,
    read_snapshot,
)
from pup_finetune.training.worker import record_worker_exit

RUN_ID = "ft-remote"


def snapshot(seq: int, *, final: bool = False, blob: int = 0, **state: Any) -> Snapshot:
    # Hashes compress badly, so a large blob takes several lines.
    noise = "".join(hashlib.sha256(f"{seq}:{i}".encode()).hexdigest() for i in range(blob // 32))
    return Snapshot(
        run_id=RUN_ID,
        seq=seq,
        at=f"2026-10-09T12:00:{seq:02d}+00:00",
        final=final,
        files={
            "state.json": {"phase": "training", "step": seq, "noise": noise, **state},
            "epochs.json": [],
            "result.json": None,
        },
        exit_error=None,
    )


def lines(*snapshots: Snapshot, start: int = 1_000) -> list[LogLine]:
    found = [line for item in snapshots for line in encode(item)]
    return [LogLine(start + index, line) for index, line in enumerate(found)]


def test_a_snapshot_round_trips_in_lines_of_at_most_max_chunk() -> None:
    sent = snapshot(3, blob=6000)
    encoded = encode(sent)
    assert len(encoded) > 3
    assert all(len(line) <= MAX_CHUNK for line in encoded)
    found = newest_snapshot(lines(sent), RUN_ID)
    assert found is not None
    assert (found.seq, found.files, found.final) == (3, sent.files, False)
    assert found.log_ns == 1_000 + len(encoded) - 1


def test_foreign_lines_and_other_runs_are_ignored() -> None:
    foreign = encode(
        Snapshot(run_id="ft-other", seq=9, at="2026", final=True, files={"x": 1}, exit_error=None)
    )
    noise = ["Resolved 120 packages", "PUPFT1 garbage", "epoch 1 step 3 loss 1.2", *foreign]
    found = newest_snapshot(
        [LogLine(i, text) for i, text in enumerate(noise)] + lines(snapshot(1), start=100),
        RUN_ID,
    )
    assert found is not None and found.seq == 1


def test_parts_out_of_order_still_decode() -> None:
    found_lines = lines(snapshot(2, blob=5000))
    shuffled = found_lines[1::2] + found_lines[::2][::-1]
    found = newest_snapshot(shuffled, RUN_ID)
    assert found is not None and found.seq == 2


def test_a_missing_part_falls_back_to_the_previous_complete_snapshot() -> None:
    newer = lines(snapshot(2, blob=5000), start=500)
    found = newest_snapshot(lines(snapshot(1)) + newer[:1] + newer[2:], RUN_ID)
    assert found is not None and found.seq == 1


def test_a_bad_digest_is_ignored_and_the_highest_seq_wins() -> None:
    good = lines(snapshot(0), snapshot(1), start=10)
    bad = lines(snapshot(2), start=100)
    head, _, chunk = bad[0].message.rpartition(" ")
    tampered = [LogLine(bad[0].timestamp_ns, f"{head} {'A' * len(chunk)}"), *bad[1:]]
    found = newest_snapshot(good + tampered, RUN_ID)
    assert found is not None and found.seq == 1


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def run_dir(tmp_path: Path) -> RunDir:
    run = RunDir(tmp_path / "run")
    run.path.mkdir()
    atomic_write_json(run.run_path, {"run_id": RUN_ID})
    return run


def saved(run: RunDir, moment: datetime, **fields: Any) -> RunState:
    state = RunState(run, now=lambda: moment, pid=4242, **fields)
    state.save()
    return state


def reports(written: list[str]) -> list[Snapshot]:
    found: list[Snapshot] = []
    for end in range(1, len(written) + 1):
        latest = newest_snapshot([LogLine(i, text) for i, text in enumerate(written[:end])], RUN_ID)
        if latest is not None and (not found or latest.seq != found[-1].seq):
            found.append(latest)
    return found


def test_the_reporter_writes_changes_and_keepalives_but_not_heartbeats(tmp_path: Path) -> None:
    run = run_dir(tmp_path)
    saved(run, datetime(2026, 10, 9, 12, 0, tzinfo=UTC))
    written: list[str] = []
    clock = Clock()
    with (
        pytest.raises(ValueError, match="boom"),
        StateReporter(run, write=written.append, clock=clock, thread=False) as reporter,
    ):
        assert len(reports(written)) == 1  # seq 0 when it starts
        assert not reporter.tick()
        saved(run, datetime(2026, 10, 9, 12, 0, 15, tzinfo=UTC))  # a heartbeat only
        assert not reporter.tick()
        saved(run, datetime(2026, 10, 9, 12, 0, 30, tzinfo=UTC), phase="training", step=3)
        assert reporter.tick()
        clock.now += 59
        assert not reporter.tick()
        clock.now += 2
        assert reporter.tick()  # a keepalive
        try:
            raise ValueError("boom")
        except ValueError as exc:
            record_worker_exit(run, "train", exc)
            raise
    sent = reports(written)
    assert [item.seq for item in sent] == [0, 1, 2, 3]
    assert [item.final for item in sent] == [False, False, False, True]
    assert sent[1].files["state.json"]["step"] == 3
    assert sent[-1].exit_error is not None
    assert sent[-1].exit_error["type"] == "ValueError"


def test_apply_clears_the_pid_and_keeps_the_copied_heartbeat(tmp_path: Path) -> None:
    run = run_dir(tmp_path)
    saved(run, datetime(2026, 10, 9, 11, 0, tzinfo=UTC))
    state = RunState(
        run, phase="training", step=7, pid=99, heartbeat_at="2026-10-09T12:00:00+00:00"
    )
    epoch = {"epoch": 1, "steps": 7, "train_loss": 1.5, "val_loss": None}
    apply_snapshot(
        run,
        Snapshot(
            run_id=RUN_ID,
            seq=4,
            at="2026-10-09T12:00:00+00:00",
            final=False,
            files={"state.json": state.to_json(), "epochs.json": [epoch], "result.json": None},
            exit_error=None,
        ),
    )
    applied = json.loads(run.state_path.read_text())
    assert (applied["phase"], applied["step"], applied["pid"]) == ("training", 7, None)
    assert applied["heartbeat_at"] == "2026-10-09T12:00:00+00:00"
    assert run.read_epochs() == [epoch]
    assert not run.result_path.exists()


@pytest.mark.parametrize(
    "files",
    [
        {"state.json": {"phase": "bogus"}, "epochs.json": [], "result.json": None},
        {"state.json": {"phase": "training"}, "epochs.json": ["x"], "result.json": None},
        {"epochs.json": [], "result.json": None},
    ],
)
def test_apply_refuses_what_is_not_a_run_s_files(tmp_path: Path, files: dict[str, Any]) -> None:
    run = run_dir(tmp_path)
    saved(run, datetime(2026, 10, 9, 11, 0, tzinfo=UTC))
    before = run.state_path.read_text()
    with pytest.raises(StateError):
        apply_snapshot(
            run,
            Snapshot(run_id=RUN_ID, seq=1, at="", final=False, files=files, exit_error=None),
        )
    assert run.state_path.read_text() == before
    assert not run.epochs_path.exists()


def checkpoint(kind: str, epoch: int) -> CheckpointRef:
    return CheckpointRef(
        kind=kind,
        name=f"{kind[0]}{epoch}s{2 * epoch}",
        uri=f"fake://{kind}/{epoch}",
        epoch=epoch,
        step=2 * epoch,
        created_in_session="job-session",
    )


def end_of(epoch: int) -> ResumePoint:
    return ResumePoint(
        checkpoint=checkpoint("training", epoch),
        epoch=epoch,
        step=2 * epoch,
        steps_in_epoch=2,
        epoch_loss_sum=2.0,
        epoch_weight=2.0,
        epoch_tokens=8,
        train_tokens=8 * epoch,
        elapsed_seconds=60.0 * epoch,
    )


def finished(epoch: int) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        steps=2,
        train_loss=1.0,
        val_loss=1.0 / epoch,
        train_tokens=8,
        new_best=True,
        training_ckpt=checkpoint("training", epoch),
        sampler_ckpt=checkpoint("sampler", epoch),
    )


def finish_epoch(run: RunDir, epoch: int) -> None:
    """Record epoch ``epoch``'s end as training does: its checkpoint, then the epoch."""
    state = run.load_state()
    state.record_checkpoint(end_of(epoch).checkpoint, resume_point=end_of(epoch))
    state.record_epoch(finished(epoch), best=finished(epoch))


def training_run(tmp_path: Path) -> RunDir:
    run = run_dir(tmp_path)
    saved(run, datetime(2026, 10, 9, 11, 0, tzinfo=UTC), phase="training")
    finish_epoch(run, 1)
    return run


class TrainsDuringRead(RunDir):
    """A run whose trainer finishes epoch 2 just before ``file`` is first opened."""

    def __init__(self, path: Path, file: str) -> None:
        super().__init__(path)
        self.file = file
        self.trained = False

    def _opened(self, name: str) -> Path:
        if name == self.file and not self.trained:
            self.trained = True
            finish_epoch(RunDir(self.path), 2)
        return self.path / name

    @property
    def state_path(self) -> Path:
        return self._opened("state.json")

    @property
    def epochs_path(self) -> Path:
        return self._opened("epochs.json")

    @property
    def result_path(self) -> Path:
        return self._opened("result.json")


@pytest.mark.parametrize("file", ["state.json", "epochs.json", "result.json"])
def test_a_snapshot_taken_while_an_epoch_is_recorded_can_be_resumed_from(
    tmp_path: Path, file: str
) -> None:
    run = training_run(tmp_path)
    taken = read_snapshot(TrainsDuringRead(run.path, file), seq=1, final=False)
    finished_epochs = len(taken.files["epochs.json"])
    point = taken.files["state.json"]["resume_point"]["epoch"]
    assert finished_epochs <= point <= finished_epochs + 1


def test_apply_refuses_a_checkpoint_older_than_the_finished_epochs(tmp_path: Path) -> None:
    run = training_run(tmp_path)
    before = {path: path.read_bytes() for path in (run.state_path, run.epochs_path)}
    state = RunState(run, phase="training", epochs=1, resume_point=end_of(1).to_json())
    stale = Snapshot(
        run_id=RUN_ID,
        seq=5,
        at="2026-10-09T12:00:00+00:00",
        final=False,
        files={
            "state.json": state.to_json(),
            "epochs.json": [finished(1).to_json(), finished(2).to_json()],
            "result.json": None,
        },
        exit_error=None,
    )
    with pytest.raises(StateError):
        apply_snapshot(run, stale)
    assert {path: path.read_bytes() for path in before} == before
