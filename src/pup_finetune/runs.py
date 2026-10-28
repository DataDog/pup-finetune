# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The run directory: what a training run was asked to do, where it is, and how it ended.

Files, all ``0600`` in a ``0700`` directory, written atomically:

- ``run.json`` - written once by :meth:`RunDir.create`: the run id, provider, model profile,
  :class:`~pup_finetune.providers.base.TrainSpec`, dataset directory and content hash,
  rendering identity, the hash of those settings, and when it was created.
- ``state.json`` - :class:`RunState`, rewritten at every step: the phase, epoch and step,
  the steps an epoch and the whole run take, the last step's training loss, every
  checkpoint, the resume point of the last training checkpoint, the best epoch, the loop's
  warnings, what a resume would do, the wall-clock seconds the spend guard counted over
  every attempt, a heartbeat and the training process id. With ``heartbeat_s``,
  :func:`execute_run` also rewrites it on a timer, so the heartbeat and the elapsed time
  stay fresh through slow steps, validation and promotion.
- ``epochs.json`` - one record per finished epoch, with the training checkpoint saved at its
  end.
- ``result.json`` - how the run ended; :func:`execute_run` writes it on every exit, also on
  failure, Ctrl-C or SIGTERM (which it reports as :class:`TrainingTerminated`): the best
  epoch and validation loss, why training stopped, the promoted model and the deployable
  checkpoint, the best and last checkpoints, the epochs, the warnings, the elapsed seconds
  over every attempt and any error.
- ``worker.json`` and ``worker.log`` - for a detached run, the worker process that was
  started and how it ended (written by :mod:`pup_finetune.training.worker`), and its output.
- ``deployment.json`` - the receipt of the run's deployment (written by
  :mod:`pup_finetune.deployment`): what was created where, its state, the smoke test and,
  once undeployed, when it was deleted.
- ``remote.json`` - for a run started with ``train --remote``, the job that runs its training
  loop (written by :mod:`pup_finetune.training.remote`): where it was submitted, its id, its
  last status and state report, and how it ended.

Every number written is finite: a non-finite loss is stored as ``null``.

:func:`execute_run` holds the directory's :class:`~pup_finetune.storage.DirectoryLock` while
training, so two processes never train in one run directory. Any error during training
leaves the run ``interrupted`` and resumable from its last training checkpoint. When only
the promotion at the end failed, the run is ``failed`` with the training result kept, and a
resume re-runs just the promotion (``resume_mode`` ``finalize_only``). With ``signals``
``cancel``, SIGTERM and Ctrl-C stop training gracefully instead (see :func:`execute_run`).
Work that follows training in the same process (``train --evaluate``) first records the
run's end with :func:`record_finished`, so a crash during that work cannot lose it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
import os
import re
import secrets
import signal
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, Literal, cast, get_args

from pup_finetune import next_steps
from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.schema import TrainingRow, row_content_hash
from pup_finetune.data.split import SplitName
from pup_finetune.errors import (
    ExternalError,
    FinetuneError,
    Interrupted,
    NotARunDir,
    RunDirNotEmpty,
    StateError,
    UsageError,
    WrongPhase,
)
from pup_finetune.providers.base import (
    MAX_MODEL_ID_LEN,
    MAX_SEQ_LEN_FLAG,
    MODEL_ID_PATTERN,
    CheckpointRef,
    ModelProfile,
    TrainSpec,
)
from pup_finetune.redaction import redact
from pup_finetune.storage import (
    LOCK_FILE_NAME,
    DirectoryLock,
    atomic_write_json,
    canonical_json,
    finite_json,
    make_private_dir,
    read_json,
    settings_hash,
)
from pup_finetune.training.loop import CheckpointKind, CheckpointNamer, FinalizeFailed
from pup_finetune.training.promotion import PromotionRequest
from pup_finetune.training.records import EpochMetrics, ResumePoint, TrainResult

RUN_FILE = "run.json"
STATE_FILE = "state.json"
EPOCHS_FILE = "epochs.json"
RESULT_FILE = "result.json"
WORKER_FILE = "worker.json"
WORKER_LOG = "worker.log"
DEPLOYMENT_FILE = "deployment.json"
REMOTE_FILE = "remote.json"
EVALUATIONS_FILE = "evaluation.json"
"""Each evaluation of the run (``evaluate`` and ``train --evaluate``), by its directory:
its settings, how it stands and its headline (see
:func:`pup_finetune.evaluation.runner.record_evaluation`)."""

Phase = Literal["created", "training", "completed", "failed", "cancelled", "interrupted"]
PHASES: tuple[Phase, ...] = get_args(Phase)

RESUMABLE_PHASES: frozenset[Phase] = frozenset({"interrupted", "failed", "cancelled"})
"""The phases :meth:`RunState.resume` continues from."""

ResumeMode = Literal["train", "finalize_only"]
"""What a resume does: continue training, or (training having ended) only re-run promotion."""
RESUME_MODES: tuple[ResumeMode, ...] = get_args(ResumeMode)

SignalMode = Literal["interrupt", "cancel"]
"""How :func:`execute_run` treats SIGTERM and Ctrl-C: stop at once, leaving the run
``interrupted``, or ask the loop to stop at the next step boundary (a graceful cancel)."""

_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    "created": frozenset({"training", "failed", "interrupted"}),
    "training": frozenset({"completed", "failed", "cancelled", "interrupted"}),
    **{phase: frozenset({"training"}) for phase in RESUMABLE_PHASES},
}

_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

_MODEL_ID_HASHED_PREFIX = MAX_MODEL_ID_LEN - 9  # room for a dash and 8 hex digits

STALE_TRAINING = "the training process stopped without recording why (it may have been killed)"


def resume_command(run_dir: Path) -> str:
    """The command that resumes the run in ``run_dir`` (a free preview first)."""
    return next_steps.command("resume", run_dir)


def resume_hint(run_dir: Path, hint: str | None = None) -> str:
    """``hint`` (by default, where the run resumes from, :func:`resumes_from`) with the
    resume command after it."""
    said = (hint or f"the run {resumes_from(run_dir)}").rstrip(". ")
    return f"{said}. Next: {resume_command(run_dir)}"


def resumes_from(run_dir: Path) -> str:
    """Where resuming the run in ``run_dir`` starts, in words: from its last training
    checkpoint, or from the start when it saved none (``state.json`` has no resume point)."""
    try:
        point = RunDir(run_dir).load_state().resume_point
    except Exception:  # an unreadable state says nothing more than "resumable"
        return "is resumable"
    if point is None:
        return "is resumable from the start: it saved no training checkpoint yet"
    return "is resumable from its last training checkpoint"


Clock = Callable[[], datetime]


def utcnow() -> datetime:
    """The current time in UTC; the default clock for run files."""
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _parsed_time(value: object) -> datetime | None:
    """An ISO 8601 time from a run file as an aware time, or ``None`` when it is not one."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _promotions_json(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The promotion requests of a saved state; see :class:`RunState`."""
    records = data.get("promotions")
    if records is not None:
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError("promotions must be a list of objects")
        return records
    legacy = _optional_str(data.get("promotion_requested_at"))
    if legacy is None:
        return []
    return [{"model_id": None, "epoch": None, "step": None, "requested_at": legacy}]


def _int(value: object) -> int:
    """``value`` as an ``int`` when it is one (not a bool), else ``0``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_str(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"expected a string, got {type(value).__name__}")


def is_run_id(text: str) -> bool:
    """Whether ``text`` is a valid run id (what :func:`check_run_id` accepts)."""
    return _RUN_ID.fullmatch(text) is not None


def check_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is a valid run id, else raise :class:`UsageError`."""
    if not _RUN_ID.fullmatch(run_id):
        raise UsageError(
            f"invalid run id {run_id!r}",
            hint="use 1 to 64 lowercase letters, digits and dashes, starting with a letter "
            "or digit",
        )
    return run_id


def new_run_id(now: datetime | None = None) -> str:
    """A fresh run id: ``ft-<yyyymmddHHMMSS>-<6 hex digits>``, in UTC."""
    moment = (now or utcnow()).astimezone(UTC)
    return f"ft-{moment:%Y%m%d%H%M%S}-{secrets.token_hex(3)}"


def checkpoint_namer(run_id: str) -> CheckpointNamer:
    """Checkpoint names ``<t|s|r><epoch>s<step>-<4 hex digits of the run id>``.

    ``t`` marks a training checkpoint, ``s`` a sampler one and ``r`` a sampler checkpoint
    re-created for promotion, e.g. ``t3s120-1a2b``, ``s3s120-1a2b`` and ``r3s120-1a2b``. Short
    enough for a 17-character limit up to epoch 99 and step 99999; the loop checks the
    backend's actual limit before training starts.
    """
    tag = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:4]
    prefixes: dict[CheckpointKind, str] = {"training": "t", "sampler": "s", "recreated": "r"}

    def name(kind: CheckpointKind, epoch: int, step: int) -> str:
        return f"{prefixes[kind]}{epoch}s{step}-{tag}"

    return name


def model_id_for_run(run_id: str) -> str:
    """The id to promote run ``run_id``'s best checkpoint to: the run id, made safe.

    It fully matches :data:`~pup_finetune.providers.base.MODEL_ID_PATTERN`, as
    :meth:`~pup_finetune.providers.base.TrainingSession.promote` requires: lowercase
    letters, digits and dashes, starting with a letter and not ending with a dash, at most
    :data:`~pup_finetune.providers.base.MAX_MODEL_ID_LEN` characters. A run id that already
    fits is used as is. Any other is made to fit (shortened, an ``ft-`` prefix when it does
    not start with a letter) and ends with a dash and 8 hex digits of its hash, so two run
    ids never map to the same model id (``x-y`` and ``x--y``, ``2026-run`` and
    ``ft-2026-run`` stay apart).
    """
    if MODEL_ID_PATTERN.fullmatch(run_id):
        return run_id
    safe = re.sub(r"[^a-z0-9]+", "-", run_id.lower()).strip("-")
    if not safe[:1].isalpha():
        safe = f"ft-{safe}".rstrip("-")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:_MODEL_ID_HASHED_PREFIX].rstrip('-')}-{digest}"


def dataset_content_hash(dataset: DatasetDir) -> str:
    """A sha256 over the dataset's kept rows, by split: see :func:`rows_content_hash`."""
    return rows_content_hash(dataset.rows_by_split())


def rows_content_hash(by_split: Mapping[SplitName, Sequence[TrainingRow]]) -> str:
    """A sha256 over rows by split: ids, content hashes, loss weights and call ids, in id order.

    The order of the rows does not change it, so whoever trains on rows it describes must
    put them in a canonical order, such as :func:`in_id_order`. Unlike
    :func:`~pup_finetune.data.schema.row_content_hash` (the conversation, for dedupe), it
    covers each message's weight, normalized (``0``, or ``1`` for ``1`` and unset), since a
    changed weight changes what a run trains on, and the tool call ids as stored, which the
    content hash leaves out but a chat template may render.
    """
    hashed = {split: sorted(_row_identity(row) for row in rows) for split, rows in by_split.items()}
    return hashlib.sha256(canonical_json(hashed).encode("utf-8")).hexdigest()


def _row_identity(row: TrainingRow) -> tuple[str, str, list[int], list[str]]:
    """What :func:`rows_content_hash` covers of ``row``, as it sorts rows."""
    weights = [0 if message.weight == 0 else 1 for message in row.messages]
    call_ids = [
        call_id
        for message in row.messages
        for call_id in (
            *(call.id for call in message.tool_calls or ()),
            *((message.tool_call_id,) if message.tool_call_id is not None else ()),
        )
    ]
    return (row.id, row_content_hash(row), weights, call_ids)


def in_id_order(rows: Iterable[TrainingRow]) -> list[TrainingRow]:
    """``rows`` in the order :func:`rows_content_hash` covers them: id, content, weights, ids."""
    return sorted(rows, key=_row_identity)


def _ckpt_json(ref: CheckpointRef | None) -> dict[str, Any] | None:
    return None if ref is None else dataclasses.asdict(ref)


def error_json(error: BaseException) -> dict[str, Any]:
    """``error`` as run files record it: its type, the code the error line shows for it
    (:data:`~pup_finetune.errors.ERROR_CODES`), and its message and notes redacted."""
    code = error.error_code if isinstance(error, FinetuneError) else FinetuneError.error_code
    return {
        "type": type(error).__name__,
        "code": Interrupted.error_code
        if isinstance(error, KeyboardInterrupt | TerminationRequested)
        else code,
        "message": redact(str(error)),
        "notes": [redact(str(note)) for note in getattr(error, "__notes__", [])],
    }


class RunDir:
    """The files of one training run under :attr:`path`. See the module docstring.

    Constructing one touches nothing on disk; :meth:`create` makes the directory.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        provider: str,
        profile: ModelProfile,
        spec: TrainSpec,
        dataset_dir: Path,
        dataset_sha256: str,
        rendering: Mapping[str, str],
        run_id: str | None = None,
        defaults_used: Sequence[str] = (),
        max_seq_len_flag: str = MAX_SEQ_LEN_FLAG,
        now: Clock = utcnow,
    ) -> RunDir:
        """Create a run in ``path``, which must be new or empty, and write ``run.json``.

        ``run_id`` defaults to :func:`new_run_id`. ``rendering`` is the renderer's
        :meth:`~pup_finetune.rendering.base.Renderer.identity`. ``run.json`` also records the
        values the run trains with (``settings``,
        :meth:`~pup_finetune.providers.base.TrainSpec.recorded_settings`) and the settings
        the user left at their defaults (``defaults_used``; ``max_seq_len_flag`` names the
        flag that cut the context, if one did); both follow from the rest, so
        neither is part of ``settings_hash``, and a run recorded before them has neither.
        """
        run_id = check_run_id(run_id if run_id is not None else new_run_id(now()))
        run_dir = cls(path)
        run_dir.require_empty()
        make_private_dir(run_dir.path)
        with run_dir.lock():
            run_dir.require_empty()
            settings = {
                "provider": provider,
                "profile": dataclasses.asdict(profile),
                "spec": spec.model_dump(mode="json"),
                "dataset_sha256": dataset_sha256,
                "rendering": dict(rendering),
            }
            manifest = {
                "run_id": run_id,
                **settings,
                "dataset_dir": str(Path(dataset_dir).resolve()),
                "settings_hash": settings_hash(settings),
                "settings": spec.recorded_settings(profile, max_seq_len_flag=max_seq_len_flag),
                "defaults_used": list(defaults_used),
                "created_at": _iso(now()),
            }
            atomic_write_json(run_dir.run_path, manifest)
            RunState(run_dir, now=now).save()
        return run_dir

    # --- paths ----------------------------------------------------------------------

    @property
    def run_path(self) -> Path:
        return self.path / RUN_FILE

    @property
    def state_path(self) -> Path:
        return self.path / STATE_FILE

    @property
    def epochs_path(self) -> Path:
        return self.path / EPOCHS_FILE

    @property
    def result_path(self) -> Path:
        return self.path / RESULT_FILE

    @property
    def worker_path(self) -> Path:
        return self.path / WORKER_FILE

    @property
    def worker_log_path(self) -> Path:
        return self.path / WORKER_LOG

    @property
    def deployment_path(self) -> Path:
        return self.path / DEPLOYMENT_FILE

    @property
    def evaluations_path(self) -> Path:
        return self.path / EVALUATIONS_FILE

    @property
    def remote_path(self) -> Path:
        return self.path / REMOTE_FILE

    def lock(self) -> DirectoryLock:
        """The directory's lock; hold it while training or changing the run."""
        return DirectoryLock(self.path)

    # --- reading --------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """``run.json``; :class:`NotARunDir` when there is none."""
        if not self.run_path.exists():
            listed = "`pup finetune runs` lists the runs under ./runs"
            if not self.path.exists():
                raise NotARunDir(f"no such run directory or run id: {self.path}", hint=listed)
            raise NotARunDir(
                f"{self.path} is not a training run (it has no {self.run_path.name})",
                hint=f"pass a training run's directory or run id; {listed}",
            )
        return self._read_object(self.run_path)

    @property
    def run_id(self) -> str:
        return str(self.manifest()["run_id"])

    def spec(self) -> TrainSpec:
        """The run's training settings, from ``run.json``."""
        return TrainSpec.model_validate(self.manifest()["spec"])

    def profile(self) -> ModelProfile:
        """The model profile the run trains, from ``run.json``."""
        try:
            return ModelProfile(**self.manifest()["profile"])
        except (TypeError, ValueError) as exc:
            raise StateError(f"{self.run_path}: invalid model profile ({exc})") from exc

    def load_state(self, *, now: Clock = utcnow) -> RunState:
        """``state.json`` as a :class:`RunState`."""
        return RunState.from_json(self, self._read_object(self.state_path), now=now)

    def read_epochs(self) -> list[dict[str, Any]]:
        """``epochs.json``; empty before the first epoch finished."""
        if not self.epochs_path.exists():
            return []
        data = read_json(self.epochs_path)
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise StateError(f"{self.epochs_path}: expected a JSON list of objects")
        return cast(list[dict[str, Any]], data)

    def read_result(self) -> dict[str, Any] | None:
        """``result.json``, or ``None`` while the run has not ended."""
        if not self.result_path.exists():
            return None
        return self._read_object(self.result_path)

    def read_deployment(self) -> dict[str, Any] | None:
        """``deployment.json``, or ``None`` when the run was never deployed."""
        if not self.deployment_path.exists():
            return None
        return self._read_object(self.deployment_path)

    def read_remote(self) -> dict[str, Any] | None:
        """``remote.json``, or ``None`` when the run was not started with ``train --remote``."""
        if not self.remote_path.exists():
            return None
        return self._read_object(self.remote_path)

    def read_evaluations(self) -> dict[str, Any] | None:
        """``evaluation.json``, or ``None`` when the run was never evaluated."""
        if not self.evaluations_path.exists():
            return None
        return self._read_object(self.evaluations_path)

    def status(self) -> dict[str, Any]:
        """A summary of the run: what it trains, where it is, and how it ended if it did.

        ``val_losses`` lists each finished epoch's validation loss, in order (``None`` for an
        epoch without validation). Everything about progress comes from one ``state.json``
        snapshot: an epoch ``epochs.json`` already holds but that snapshot has not recorded
        yet (it is written first) is left out, so the epochs, the best epoch and the step
        always agree. ``steps_per_epoch`` and ``max_steps`` are ``None`` until training
        started (a state saved before they were recorded takes them from the first finished
        epoch). ``elapsed_seconds`` is the spend guard's time over every attempt, as of the
        snapshot, and ``hourly_rate_usd`` the rate the run prices it at, if any. While the run
        is ``created`` or ``training`` (e.g. resumed), nothing is reported from
        ``result.json``, which then describes an earlier attempt.
        """
        manifest = self.manifest()
        state = self.load_state()
        result = None if state.phase in ("created", "training") else self.read_result()
        epochs = self.read_epochs()
        if state.phase == "training":
            epochs = [row for row in epochs if _int(row.get("epoch")) <= state.epochs]
        spec = manifest["spec"]
        per_epoch, max_steps = state.steps_per_epoch, state.max_steps
        if per_epoch is None and epochs and _int(epochs[0].get("steps")) > 0:
            per_epoch = _int(epochs[0].get("steps"))
        if max_steps is None and per_epoch is not None:
            max_steps = per_epoch * int(spec["max_epochs"])
        return {
            "run_id": manifest["run_id"],
            "path": str(self.path),
            "provider": manifest["provider"],
            "model": manifest["profile"]["alias"],
            "phase": state.phase,
            "epoch": state.epoch,
            "step": state.step,
            "epochs_completed": state.epochs,
            "max_epochs": spec["max_epochs"],
            "steps_per_epoch": per_epoch,
            "max_steps": max_steps,
            "last_train_loss": state.last_train_loss,
            "val_losses": [epoch.get("val_loss") for epoch in epochs],
            "best": state.best,
            "elapsed_seconds": state.elapsed_seconds,
            "hourly_rate_usd": spec.get("hourly_rate_usd"),
            "settings": manifest.get("settings"),
            "defaults_used": manifest.get("defaults_used"),
            "heartbeat_at": state.heartbeat_at,
            "pid": state.pid,
            "error": state.error,
            "resume_mode": state.resume_mode,
            "stopped_reason": result.get("stopped_reason") if result else None,
            "promoted_model": result.get("promoted_model") if result else None,
            "deployable_checkpoint": result.get("deployable_checkpoint") if result else None,
            "finalize_error": result.get("finalize_error") if result else None,
            "warnings": result.get("warnings", []) if result else list(state.warnings),
        }

    # --- writing --------------------------------------------------------------------

    def write_epochs(self, epochs: Sequence[Mapping[str, Any]]) -> None:
        atomic_write_json(self.epochs_path, finite_json([dict(epoch) for epoch in epochs]))

    def write_result(self, result: Mapping[str, Any]) -> None:
        atomic_write_json(self.result_path, finite_json(dict(result)))

    def write_deployment(self, receipt: Mapping[str, Any]) -> None:
        atomic_write_json(self.deployment_path, finite_json(dict(receipt)))

    def write_remote(self, record: Mapping[str, Any]) -> None:
        atomic_write_json(self.remote_path, finite_json(dict(record)))

    def write_evaluations(self, record: Mapping[str, Any]) -> None:
        atomic_write_json(self.evaluations_path, finite_json(dict(record)))

    # --- checks ---------------------------------------------------------------------

    def require_empty(self, *, default: bool = False) -> None:
        """Raise :class:`RunDirNotEmpty` unless :attr:`path` is missing or an empty directory,
        saying what it holds when that is a run. ``default``: :attr:`path` is the default
        ``runs/<run id>`` (no ``--run-dir`` was given), so the hint names ``--run-id``."""
        again = (
            "to train again, pass another --run-id (or omit it)"
            if default
            else "to train again, omit --run-dir or pass a new one"
        )
        if self.path.exists() and not self.path.is_dir():
            raise RunDirNotEmpty(f"{self.path} exists and is not a directory", hint=again)
        if self.path.is_dir() and any(e.name != LOCK_FILE_NAME for e in self.path.iterdir()):
            holds = self._holds()
            if holds is None:
                raise RunDirNotEmpty(f"{self.path} is not empty", hint=again)
            raise RunDirNotEmpty(
                f"{self.path} {holds}",
                hint=f"see it with `{next_steps.command('status', self.path)}`; {again}",
            )

    # --- internals ------------------------------------------------------------------

    def _holds(self) -> str | None:
        """What a non-empty :attr:`path` holds when that is a run (``already holds run ...
        (completed, best epoch 5)``), for the error that refuses a new run in it; else
        ``None``."""
        try:
            run_id = self.manifest().get("run_id")
            state = self.load_state()
        except FinetuneError:
            return None
        best = (state.best or {}).get("epoch")
        where = state.phase if best is None else f"{state.phase}, best epoch {best}"
        return f"already holds run {run_id} ({where})"

    def _read_object(self, path: Path, *, hint: str | None = None) -> dict[str, Any]:
        try:
            data = read_json(path)
        except FileNotFoundError as exc:
            raise StateError(f"{path} does not exist", hint=hint) from exc
        if not isinstance(data, dict):
            raise StateError(f"{path}: expected a JSON object")
        return data


class RunState:
    """``state.json``: where the run is. It is the :class:`~pup_finetune.training.loop.LoopState`
    the loop reports into, and every report is saved at once (with a fresh heartbeat).

    ``phase`` moves ``created`` -> ``training`` -> ``completed``, ``failed``, ``cancelled`` or
    ``interrupted``, and from the last three back to ``training`` on :meth:`resume`; other
    moves raise :class:`StateError`. ``step`` counts optimizer steps over the run and
    ``last_train_loss`` is the last step's training loss. ``steps_per_epoch`` and
    ``max_steps`` are the optimizer steps an epoch and the whole run take (at most), as the
    loop planned them when training last started; ``None`` before that, and in a state saved
    before they were recorded. ``resume_point`` is the last
    training checkpoint with where training stood (a
    :class:`~pup_finetune.training.records.ResumePoint` as JSON); ``resume_mode`` says what a
    resume does. :meth:`request_cancel` (e.g. from a signal handler) makes the loop stop at
    the next step boundary. :meth:`save` may be called from another thread
    (:meth:`keep_alive`); saves are serialized, and each writes the state as it is then.

    :attr:`elapsed_seconds` is the wall-clock time the spend guard counted over every attempt
    of the run. It never decreases: while :meth:`track_elapsed` follows a budget, it is the
    larger of the recorded value and the budget's, and every save records it. So a resume
    counts the time spent after the last training checkpoint too, up to the last save (at
    most one step or heartbeat before a crash).

    :attr:`promotions` are the run's promotion requests, one per checkpoint it asked its
    provider to promote: the model id, the checkpoint's epoch and step, and when it first
    asked (:meth:`record_promotion_request`), saved before that request and kept across
    resumes, so a model created since under that id can be told to be this run's, holding
    that checkpoint (see :class:`~pup_finetune.training.promotion.Promotion`). A request whose
    first attempt found the id taken is ``contested`` (:meth:`contest_promotion_request`). A
    state saved before these were recorded holds only ``promotion_requested_at``, the time of
    the run's first request; it is read as a request for an unknown checkpoint.
    """

    def __init__(
        self,
        run_dir: RunDir,
        *,
        phase: Phase = "created",
        epoch: int = 0,
        step: int = 0,
        epochs: int = 0,
        steps_per_epoch: int | None = None,
        max_steps: int | None = None,
        last_train_loss: float | None = None,
        checkpoints: Sequence[Mapping[str, Any]] = (),
        resume_point: Mapping[str, Any] | None = None,
        resume_mode: ResumeMode = "train",
        best: Mapping[str, Any] | None = None,
        warnings: Sequence[str] = (),
        heartbeat_at: str | None = None,
        pid: int | None = None,
        error: Mapping[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
        promotions: Sequence[Mapping[str, Any]] = (),
        now: Clock = utcnow,
    ) -> None:
        if not (math.isfinite(elapsed_seconds) and elapsed_seconds >= 0):
            raise ValueError(
                f"elapsed_seconds must be finite and not negative, got {elapsed_seconds}"
            )
        self.run_dir = run_dir
        self.phase: Phase = phase
        self.epoch = epoch
        self.step = step
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps
        self.last_train_loss = last_train_loss
        self.checkpoints = [dict(ckpt) for ckpt in checkpoints]
        self.resume_point = None if resume_point is None else dict(resume_point)
        self.resume_mode: ResumeMode = resume_mode
        self.best = None if best is None else dict(best)
        self.warnings = [str(warning) for warning in warnings]
        self.heartbeat_at = heartbeat_at
        self.pid = pid
        self.error = None if error is None else dict(error)
        self._elapsed_seconds = float(elapsed_seconds)
        self.promotions = [dict(record) for record in promotions]
        self._elapsed: Callable[[], float] | None = None
        self._now = now
        self._cancel = threading.Event()
        self._save_lock = threading.Lock()

    @classmethod
    def from_json(
        cls, run_dir: RunDir, data: Mapping[str, Any], *, now: Clock = utcnow
    ) -> RunState:
        phase = data.get("phase")
        if phase not in PHASES:
            raise StateError(f"{run_dir.state_path}: unknown phase {phase!r}")
        mode = data.get("resume_mode", "train")
        if mode not in RESUME_MODES:
            raise StateError(f"{run_dir.state_path}: unknown resume mode {mode!r}")
        loss = data.get("last_train_loss")
        per_epoch, max_steps = data.get("steps_per_epoch"), data.get("max_steps")
        try:
            return cls(
                run_dir,
                phase=cast(Phase, phase),
                epoch=int(data.get("epoch", 0)),
                step=int(data.get("step", 0)),
                epochs=int(data.get("epochs", 0)),
                steps_per_epoch=None if per_epoch is None else int(per_epoch),
                max_steps=None if max_steps is None else int(max_steps),
                last_train_loss=None if loss is None else float(loss),
                checkpoints=list(data.get("checkpoints") or []),
                resume_point=data.get("resume_point"),
                resume_mode=cast(ResumeMode, mode),
                best=data.get("best"),
                warnings=list(data.get("warnings") or []),
                heartbeat_at=data.get("heartbeat_at"),
                pid=data.get("pid"),
                error=data.get("error"),
                elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
                promotions=_promotions_json(data),
                now=now,
            )
        except (TypeError, ValueError) as exc:
            raise StateError(f"{run_dir.state_path}: invalid run state ({exc})") from exc

    def to_json(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "step": self.step,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "max_steps": self.max_steps,
            "last_train_loss": self.last_train_loss,
            "checkpoints": self.checkpoints,
            "resume_point": self.resume_point,
            "resume_mode": self.resume_mode,
            "best": self.best,
            "warnings": self.warnings,
            "heartbeat_at": self.heartbeat_at,
            "pid": self.pid,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "promotions": self.promotions,
        }

    @property
    def elapsed_seconds(self) -> float:
        """The spend guard's seconds over every attempt: see the class docstring."""
        if self._elapsed is None:
            return self._elapsed_seconds
        return max(self._elapsed_seconds, self._elapsed())

    def track_elapsed(self, elapsed: Callable[[], float]) -> None:
        """Follow ``elapsed`` (e.g. a budget's ``elapsed_seconds``) from now on.

        It must count every earlier attempt too, as a resumed budget does.
        """
        self._elapsed = elapsed

    def save(self) -> None:
        """Write ``state.json`` atomically, with a fresh heartbeat; non-finite numbers as null."""
        with self._save_lock:
            self.heartbeat_at = _iso(self._now())
            self._elapsed_seconds = self.elapsed_seconds
            atomic_write_json(self.run_dir.state_path, finite_json(self.to_json()))

    @contextmanager
    def keep_alive(self, interval_s: float) -> Iterator[None]:
        """Save the state every ``interval_s`` seconds from a background thread, so the
        heartbeat stays fresh while one step, validation or promotion takes long.

        A failed save is skipped (the next report or beat tries again); the first failure and
        the recovery are noted on stderr (``worker.log`` for a detached run). The thread stops
        before the context exits.
        """
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        stop = threading.Event()

        def note(message: str) -> None:
            with contextlib.suppress(OSError):
                os.write(2, f"{redact(message)}\n".encode())

        def beat() -> None:
            failing = False
            while not stop.wait(interval_s):
                try:
                    self.save()
                except Exception as exc:
                    if not failing:
                        note(f"could not refresh the run's heartbeat: {type(exc).__name__}: {exc}")
                    failing = True
                else:
                    if failing:
                        note("the run's heartbeat is refreshed again")
                    failing = False

        thread = threading.Thread(target=beat, name="run-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    # --- phases ---------------------------------------------------------------------

    def transition(self, phase: Phase, *, error: BaseException | None = None) -> None:
        """Move to ``phase`` and save; ``error`` is recorded (redacted) when given."""
        if phase not in _TRANSITIONS.get(self.phase, frozenset()):
            raise StateError(f"a run cannot go from {self.phase} to {phase}")
        self.phase = phase
        if error is not None:
            self.error = error_json(error)
        self.save()

    def start(self) -> None:
        """Enter ``training`` in this process, for a run that never trained."""
        if self.phase != "created":
            raise WrongPhase(
                f"a run cannot go from {self.phase} to training",
                hint="resume an interrupted, failed or cancelled run instead",
            )
        self.pid = os.getpid()
        self.transition("training")

    def resume(self) -> None:
        """Enter ``training`` again in this process, for an interrupted, failed or cancelled run.

        Raises :class:`WrongPhase` for a run in any other phase.
        """
        require_resumable(self)
        self.pid = os.getpid()
        self.error = None
        self.transition("training")

    def request_cancel(self) -> None:
        """Ask the loop to stop at the next step boundary. Safe from signal handlers."""
        self._cancel.set()

    @property
    def cancel_event(self) -> threading.Event:
        """The event :meth:`request_cancel` sets, for work that stops on the same signal."""
        return self._cancel

    # --- the loop's reports (LoopState) -----------------------------------------------

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def record_plan(self, *, steps_per_epoch: int, max_steps: int) -> None:
        self.steps_per_epoch, self.max_steps = steps_per_epoch, max_steps
        self.save()

    def record_step(self, *, epoch: int, step: int, loss: float | None = None) -> None:
        self.epoch, self.step = epoch, step
        if loss is not None:
            self.last_train_loss = loss
        self.save()

    def record_checkpoint(self, ref: CheckpointRef, *, resume_point: ResumePoint | None) -> None:
        self.checkpoints.append(dataclasses.asdict(ref))
        if resume_point is not None:
            self.resume_point = resume_point.to_json()
        self.save()

    def promotion_request(self, epoch: int, step: int) -> PromotionRequest | None:
        """The recorded request to promote the checkpoint at ``epoch`` and ``step``, if any."""
        for record in self.promotions:
            if (record.get("epoch"), record.get("step")) != (epoch, step):
                continue
            model_id, requested_at = (
                record.get("model_id"),
                _parsed_time(record.get("requested_at")),
            )
            if isinstance(model_id, str) and requested_at is not None:
                return PromotionRequest(
                    model_id=model_id,
                    epoch=epoch,
                    step=step,
                    requested_at=requested_at,
                    contested=record.get("contested") is True,
                )
        return None

    def any_promotion_requested(self) -> bool:
        """Whether the run recorded a request to promote any checkpoint."""
        return bool(self.promotions)

    def record_promotion_request(self, model_id: str, *, epoch: int, step: int) -> PromotionRequest:
        """Record a request to promote the checkpoint at ``epoch`` and ``step`` to
        ``model_id``, made now, in place of any earlier one for it, and save it before
        returning it, so a request made after the call is always covered. Its time is
        truncated to the second, as a resume reads it back.
        """
        moment = self._now().replace(microsecond=0)
        record = {"model_id": model_id, "epoch": epoch, "step": step, "requested_at": _iso(moment)}
        kept = [r for r in self.promotions if (r.get("epoch"), r.get("step")) != (epoch, step)]
        self.promotions = [*kept, record]
        self.save()
        return PromotionRequest(model_id=model_id, epoch=epoch, step=step, requested_at=moment)

    def contest_promotion_request(self, epoch: int, step: int) -> None:
        """Mark the request to promote the checkpoint at ``epoch`` and ``step`` contested, and
        save it: its model id was taken before the run could have promoted it."""
        for record in self.promotions:
            if (record.get("epoch"), record.get("step")) == (epoch, step):
                record["contested"] = True
        self.save()

    def record_warning(self, message: str) -> None:
        self.warnings.append(message)
        self.save()

    def record_epoch(self, metrics: EpochMetrics, *, best: EpochMetrics | None) -> None:
        epochs = [row for row in self.run_dir.read_epochs() if row.get("epoch") != metrics.epoch]
        self.run_dir.write_epochs([*epochs, metrics.to_json()])
        self.epoch, self.epochs = metrics.epoch, metrics.epoch
        if best is not None:
            self.best = {
                "epoch": best.epoch,
                "val_loss": best.val_loss,
                "train_loss": best.train_loss,
                "sampler_ckpt": _ckpt_json(best.sampler_ckpt),
                "training_ckpt": _ckpt_json(best.training_ckpt),
            }
        self.save()

    # --- reading back -----------------------------------------------------------------

    def last_training_checkpoint(self) -> dict[str, Any] | None:
        """The most recent training checkpoint recorded, if any."""
        return next((c for c in reversed(self.checkpoints) if c.get("kind") == "training"), None)

    def last_resume_point(self) -> ResumePoint | None:
        """:attr:`resume_point` parsed; ``None`` before the first training checkpoint."""
        if self.resume_point is None:
            return None
        return ResumePoint.from_json(self.resume_point)


def require_resumable(state: RunState) -> None:
    """Raise :class:`WrongPhase` unless ``state``'s run is interrupted, failed or cancelled."""
    if state.phase in RESUMABLE_PHASES:
        return
    why = {
        "created": "it has not started training yet; train it instead",
        "training": "it is still training; wait for it, or cancel it",
        "completed": "it already completed; start a new run to train again",
    }
    raise WrongPhase(
        f"cannot resume the run in {state.run_dir.path}: it is {state.phase}",
        hint=f"only an interrupted, failed or cancelled run can be resumed, and {why[state.phase]}",
    )


# --- Running in a run directory ----------------------------------------------------------


class TerminationRequested(BaseException):
    """SIGTERM arrived; a BaseException so no ``except Exception`` swallows it before cleanup."""


class TrainingTerminated(Interrupted):
    """Training stopped because the process received SIGTERM; the run is ``interrupted``.

    :func:`execute_run` raises it from the :class:`TerminationRequested` once the result is
    written, so the CLI reports it like any other failure. ``hint`` carries the notes the
    interruption collected, e.g. steps that also failed.
    """


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


_CANCEL_NOTICE = (
    b"cancel requested: stopping after the current step, then saving a checkpoint and "
    b"promoting the best epoch"
)


@contextmanager
def cancel_on_signals(cancel: threading.Event, notice: bytes) -> Iterator[None]:
    """Make SIGTERM and the first Ctrl-C set ``cancel``, writing ``notice`` to stderr.

    A repeated SIGTERM changes nothing; a second Ctrl-C raises ``KeyboardInterrupt``, which
    stops at once. After a Ctrl-C the notice says so. Main thread only, like every handler;
    elsewhere it changes nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_signal(signum: int, frame: FrameType | None) -> None:
        if cancel.is_set():
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            return
        cancel.set()
        with contextlib.suppress(OSError):
            tail = b" (press Ctrl-C again to stop at once)" if signum == signal.SIGINT else b""
            os.write(2, notice + tail + b"\n")

    sigs = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.signal(sig, on_signal) for sig in sigs}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)


def _signals_cancel(state: RunState) -> contextlib.AbstractContextManager[None]:
    """Make SIGTERM and Ctrl-C ask ``state`` to cancel at the next step boundary.

    A second Ctrl-C stops at once (the run is then ``interrupted``); see
    :func:`cancel_on_signals`.
    """
    return cancel_on_signals(state.cancel_event, _CANCEL_NOTICE)


@contextmanager
def _interrupts_ignored() -> Iterator[None]:
    """Ignore SIGINT and SIGTERM (on the main thread) so the result is always written."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)


def _final_phase(outcome: TrainResult | None, error: BaseException | None) -> Phase:
    if isinstance(error, FinalizeFailed):
        return "interrupted" if error.interrupted else "failed"
    if error is None and outcome is not None:
        return "cancelled" if outcome.stopped_reason == "cancelled" else "completed"
    return "interrupted"


def _result_json(
    run_dir: RunDir,
    state: RunState,
    run_id: str,
    phase: Phase,
    outcome: TrainResult | None,
    error: BaseException | None,
    finished_at: datetime,
) -> dict[str, Any]:
    """The run's ``result.json``. Built from memory, except for ``epochs.json`` (read with a
    fallback), so a damaged run directory cannot keep the result from being written.

    A :class:`FinalizeFailed` ``error`` keeps its training result, with the promotion error
    as ``finalize_error``; any other error is ``error``.
    """
    finalize_error: BaseException | None = None
    if isinstance(error, FinalizeFailed):
        outcome, finalize_error = error.result, error.cause
        error = None
    common = {
        "run_id": run_id,
        "phase": phase,
        "error": None if error is None else error_json(error),
        "finalize_error": None if finalize_error is None else error_json(finalize_error),
        "elapsed_seconds": state.elapsed_seconds,
        "finished_at": _iso(finished_at),
    }
    if outcome is not None:
        return finite_json({**common, **outcome.to_json()})
    # Training did not return: report what the run state recorded before it stopped.
    warnings = list(state.warnings)
    try:
        epochs = run_dir.read_epochs()
    except Exception as exc:
        epochs = []
        warnings.append(f"could not read {EPOCHS_FILE}: {redact(str(exc))}")
    best = state.best or {}
    return finite_json(
        {
            **common,
            "stopped_reason": None,
            "best_epoch": best.get("epoch"),
            "best_val_loss": best.get("val_loss"),
            "promoted_model": None,
            "deployable_checkpoint": None,
            "checkpoints": {
                "best_sampler": best.get("sampler_ckpt"),
                "best_training": best.get("training_ckpt"),
                "last_training": state.last_training_checkpoint(),
            },
            "epochs": epochs,
            "warnings": warnings,
        }
    )


def _record_end(
    run_dir: RunDir,
    state: RunState,
    run_id: str,
    outcome: TrainResult | None,
    error: BaseException | None,
    finished_at: datetime,
) -> tuple[dict[str, Any], BaseException | None]:
    """Write ``result.json``, then move to the final phase; each is attempted on its own.

    Never raises. Returns the result and the error to raise: the training ``error`` (with
    notes on what could not be recorded), else the first recording failure. After a
    :class:`FinalizeFailed` whose result was written, a resume only re-runs the promotion.
    A promotion retry that fails before it reaches the promotion keeps the ``result.json``
    it retries from, so a later resume can retry again.
    """
    if state.phase in _FINISHED_PHASES and outcome is None:
        # Recorded by record_finished, then what followed it raised (e.g. a stray Ctrl-C):
        # the run did finish, so keep that record and only raise the error.
        try:
            return run_dir.read_result() or {}, error
        except Exception as exc:
            if error is None:
                return {}, exc
            error.add_note(f"could not read the run's result: {redact(str(exc))}")
            return {}, error
    phase = _final_phase(outcome, error)
    kept = _kept_for_retry(run_dir, state, outcome, error)
    if kept is not None:
        return kept, _recorded_phase(state, phase, error)
    result = _result_json(run_dir, state, run_id, phase, outcome, error, finished_at)
    recorded = False
    try:
        run_dir.write_result(result)
        recorded = True
    except Exception as exc:
        if error is None:
            exc.add_note("training ended, but the run's result could not be recorded")
            error, phase = exc, "failed"
        else:
            error.add_note(f"could not record the run's result: {exc}")
    finished = error.result if isinstance(error, FinalizeFailed) else outcome
    if finished is not None:
        state.resume_mode = "finalize_only" if isinstance(error, FinalizeFailed) else "train"
        if not recorded:
            # Nothing on disk to retry the promotion from: a resume trains from the last
            # training checkpoint (at the end of training) and then promotes.
            state.resume_mode = "train"
            if isinstance(error, FinalizeFailed):
                error.hint = (
                    "the training result could not be recorded; resume the run to continue "
                    "from its last training checkpoint and promote"
                )
        _record_deployable(state, finished)
    return result, _recorded_phase(state, phase, error)


_FINISHED_PHASES: frozenset[Phase] = frozenset({"completed", "cancelled"})


def _record_deployable(state: RunState, finished: TrainResult) -> None:
    deployable = _ckpt_json(finished.deployable_ckpt)
    if deployable is not None and deployable not in state.checkpoints:
        state.checkpoints.append(deployable)  # e.g. re-created for promotion


def record_finished(
    run_dir: RunDir, state: RunState, outcome: TrainResult, *, now: Clock = utcnow
) -> None:
    """Record ``outcome`` as how the run ended now, from inside :func:`execute_run`'s ``train``.

    For work that follows training and promotion in the same process (``train --evaluate``
    evaluates in the training session): ``result.json`` is written and the run moves to
    ``completed`` (or ``cancelled``) first, so a process that dies during that work leaves
    a finished run with its promoted model recorded, not a ``training`` run a resume would
    train again. :func:`execute_run` records the end again when ``train`` returns, with
    what that work added (e.g. a warning). Raises when either cannot be written; the run is
    then recorded as usual when ``train`` returns.
    """
    phase = _final_phase(outcome, None)
    result = _result_json(run_dir, state, run_dir.run_id, phase, outcome, None, now())
    run_dir.write_result(result)
    state.resume_mode = "train"
    _record_deployable(state, outcome)
    state.transition(phase)


def _kept_for_retry(
    run_dir: RunDir, state: RunState, outcome: TrainResult | None, error: BaseException | None
) -> dict[str, Any] | None:
    """The ``result.json`` to keep when a promotion retry failed before promoting, if any.

    That retry never got to the promotion (e.g. it was interrupted while reading the result),
    so the training result on disk is still the one to retry from. When that file does not
    hold a finished training's result, ``None``: the resume falls back to training.
    """
    if state.resume_mode != "finalize_only" or outcome is not None or error is None:
        return None
    if isinstance(error, FinalizeFailed):
        return None
    try:
        kept = run_dir.read_result()
        TrainResult.from_json(kept or {})
    except Exception as exc:
        error.add_note(f"the training result cannot be retried from: {redact(str(exc))}")
        state.resume_mode = "train"
        return None
    return kept


def _recorded_phase(
    state: RunState, phase: Phase, error: BaseException | None
) -> BaseException | None:
    """Move ``state`` to ``phase``; returns ``error``, or the failure to record the phase.

    A run :func:`record_finished` already moved to ``phase`` is saved again as it is.
    """
    try:
        if state.phase == phase and error is None:
            state.save()
        else:
            state.transition(phase, error=error)
    except Exception as exc:
        if error is None:
            return exc
        error.add_note(f"could not record the run's phase ({phase}): {exc}")
    return error


def _for_cli(error: BaseException, resumable: Path | None) -> BaseException:
    """The error to raise for ``error``: a :class:`FinetuneError` the CLI reports, or an
    interrupt as it came. The original is chained as the cause, its message redacted.

    ``resumable`` is the run directory when the error left the run ``interrupted`` or
    ``failed``: every :class:`FinetuneError` returned then ends its hint with the resume
    command (:func:`resume_hint`).
    """
    converted = _converted(error)
    if resumable is not None and isinstance(converted, FinetuneError):
        converted.hint = resume_hint(resumable, converted.hint)
    return converted


def _converted(error: BaseException) -> BaseException:
    if isinstance(error, TerminationRequested):
        hint = "; ".join(getattr(error, "__notes__", [])) or None
        return TrainingTerminated("training stopped: the process received SIGTERM", hint=hint)
    if isinstance(error, FinalizeFailed) and error.interrupted:
        # Ctrl-C or SIGTERM stopped the promotion: report the interrupt, not a failure.
        for note in getattr(error, "__notes__", []):
            error.cause.add_note(note)
        return _converted(error.cause)
    if isinstance(error, FinetuneError) or not isinstance(error, Exception):
        return error
    detail = redact(str(error))
    message = f"training stopped: {type(error).__name__}{': ' + detail if detail else ''}"
    # A dropped connection or a timeout is the provider's failure, not the run directory's.
    if isinstance(error, OSError) and not isinstance(error, ConnectionError | TimeoutError):
        return StateError(f"{message} (writing the run directory)")
    return ExternalError(message, source="training")


def execute_run(
    run_dir: RunDir,
    train: Callable[[RunState], TrainResult],
    *,
    resume: bool = False,
    signals: SignalMode = "interrupt",
    heartbeat_s: float | None = None,
    now: Clock = utcnow,
) -> dict[str, Any]:
    """Run ``train`` in ``run_dir`` under its lock, and write ``result.json`` however it ends.

    ``train`` gets the :class:`RunState` to pass to the loop, typically opening a session and
    calling :func:`~pup_finetune.training.loop.run_training` in it. A new run must be
    ``created``. With ``resume`` it must be interrupted, failed or cancelled
    (:class:`WrongPhase` otherwise), or ``training`` with no process holding the lock: that
    process died without recording it, and the run is first marked ``interrupted``.

    The phase becomes ``completed`` or ``cancelled`` from the result. Any error during
    training makes it ``interrupted``, resumable from the last training checkpoint; a
    :class:`~pup_finetune.training.loop.FinalizeFailed` makes it ``failed`` (``interrupted``
    if Ctrl-C or SIGTERM stopped the promotion) with the training result kept and a resume
    that only retries the promotion. The error is re-raised once the result is written: a
    :class:`FinetuneError` as is, SIGTERM as :class:`TrainingTerminated` and Ctrl-C as
    ``KeyboardInterrupt`` (also when they stopped the promotion), a failed write of the run
    directory as :class:`StateError` and anything else (including a dropped connection) as
    :class:`ExternalError`. SIGINT and SIGTERM are ignored while the result
    is written. A failure to record the result is noted on the training error, or raised (as
    ``failed``) when training succeeded. Returns the result.

    With ``signals`` ``cancel`` (what the CLI uses), SIGTERM and the first Ctrl-C instead
    cancel gracefully: the loop stops at the next step boundary, saves a training checkpoint
    and promotes the best epoch, and the run ends ``cancelled`` with its result returned; a
    second Ctrl-C stops at once as above. With ``heartbeat_s``, the state is also saved every
    ``heartbeat_s`` seconds while ``train`` runs (see :meth:`RunState.keep_alive`).
    """
    with run_dir.lock(), contextlib.ExitStack() as handling:
        run_id = run_dir.run_id
        state = run_dir.load_state(now=now)
        if signals == "cancel":
            # Before entering training, so a cancel sent as soon as the state says training is
            # not lost: the handler only asks the loop to stop at its first step boundary.
            handling.enter_context(_signals_cancel(state))
        if resume:
            if state.phase == "training":
                # We hold the lock, so the process that was training has died (e.g. SIGKILL).
                state.transition("interrupted", error=StateError(STALE_TRAINING))
            state.resume()
        else:
            state.start()
        outcome: TrainResult | None = None
        error: BaseException | None = None
        alive = contextlib.nullcontext() if heartbeat_s is None else state.keep_alive(heartbeat_s)
        if signals != "cancel":
            handling.enter_context(_sigterm_raises())
        try:
            with alive:
                outcome = train(state)
        except BaseException as exc:
            error = exc
        with _interrupts_ignored():
            result, error = _record_end(run_dir, state, run_id, outcome, error, now())
    if error is not None:
        resumable = run_dir.path if state.phase in ("interrupted", "failed") else None
        raised = _for_cli(error, resumable)
        if raised is error or raised is error.__cause__:
            raise raised
        raise raised from error
    return result
