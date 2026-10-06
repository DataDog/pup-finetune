# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""An in-memory Fireworks Training API behind the wrapper interface of ``fireworks/client.py``.

:class:`FakeFireworksTrainingClient` implements
:class:`~pup_finetune.providers.fireworks.client.FireworksTraining` the way the real service
behaves, deterministically and with synthetic data only::

    fw = FakeFireworksTrainingClient()
    service = fw.service()  # session ts-<hex>
    run = service.create_lora_training_client(BASE_MODEL, 8)
    out = run.forward_backward(examples).result()  # metrics in Fireworks' own shape
    run.optim_step(AdamConfig(learning_rate=1e-4)).result()
    path = run.save_weights_for_sampler("e1-ab12").result()  # .../e1-ab12-<8 hex>
    rows = fw.list_session_checkpoints(service.session_name)

What it mimics:

- The server renames checkpoints: a training checkpoint saved as ``<name>`` is listed as
  ``run-<hex>-<name>``, a sampler checkpoint as ``run-<hex>-<name>-<8 hex>``; rows carry the
  checkpoint type, ``promotable`` (sampler checkpoints only) and an increasing ``createTime``.
- A name longer than :data:`MAX_CHECKPOINT_NAME_LEN` saves fine and fails at promote time.
- Closing a service ends its session; its sampler checkpoints can no longer be promoted,
  while its training checkpoints can still be resumed from in a new session.
- ``forward_backward`` reports ``loss:sum``, ``total_tokens:sum`` and ``response_tokens``;
  ``forward`` reports only ``total_tokens:sum``, with per-token logprobs per example. The
  per-token loss is ``loss_at(steps)``, which falls with each optimizer step that applied
  gradients, and is restored with the weights when a run resumes from a checkpoint.
- Sampling returns :attr:`FakeFireworksTrainingClient.sample_tokens`.

Failures are injected per operation with :meth:`FakeFireworksTrainingClient.fail`, as raw
SDK-shaped errors mapped by the real :func:`fireworks_error`, so they arrive as the wrapper
raises them (``ProviderCapacityError``, ``ProviderAuthError``, ``TransientError``, ...).
Every call is recorded in :attr:`FakeFireworksTrainingClient.calls`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    MODEL_ID_PATTERN,
    AdamConfig,
    PendingResult,
    ProviderCapacityError,
)
from pup_finetune.providers.fireworks.client import (
    CAPACITY_RELEASED_HINT,
    MAX_CHECKPOINT_NAME_LEN,
    SAMPLER_CHECKPOINT,
    SOURCE,
    TRAINING_CHECKPOINT,
    CheckpointRow,
    ForwardOutput,
    OptimStepOutput,
    SampledTokens,
    TokenSampler,
    TrainingRun,
    TrainingService,
    fireworks_error,
    sort_rows,
)
from pup_finetune.rendering.tokens import TokenizedExample

ACCOUNT = "synthetic-team"
BASE_MODEL = "accounts/fireworks/models/qwen3p8-27b"
DEFAULT_SAMPLE_TOKENS: tuple[int, ...] = (101, 102, 103, 248046)

FailureKind = Literal["capacity", "rate_limit", "auth", "server", "connection", "bad_request"]
OPERATIONS: frozenset[str] = frozenset(
    {
        "service",
        "create_lora_training_client",
        "create_training_client_from_state_with_optimizer",
        "create_sampling_client",
        "list_session_checkpoints",
        "promote_session_checkpoint",
        "forward_backward",
        "forward",
        "optim_step",
        "save_state",
        "save_weights_for_sampler",
        "load_state",
        "sample",
    }
)
"""The operations :meth:`FakeFireworksTrainingClient.fail` can fail."""

_START = datetime(2026, 1, 1, tzinfo=UTC)


class FakeSdkError(Exception):
    """An SDK-shaped failure: an HTTP status and the server's message."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIConnectionError(Exception):
    """Stands in for the SDK's connection error, which the wrapper recognizes by class name."""


_FAILURES: dict[str, Callable[[], Exception]] = {
    "capacity": lambda: FakeSdkError(
        "training capacity is temporarily exhausted (RESOURCE_EXHAUSTED)", status_code=409
    ),
    "auth": lambda: FakeSdkError("invalid API key", status_code=401),
    "server": lambda: FakeSdkError("internal error", status_code=503),
    "rate_limit": lambda: FakeSdkError("too many requests", status_code=429),
    "connection": lambda: APIConnectionError("Connection error."),
    "bad_request": lambda: FakeSdkError("invalid request", status_code=400),
}


def loss_at(steps: int) -> float:
    """The fake per-token loss after ``steps`` optimizer steps that applied gradients."""
    return 2.0 / (1.0 + 0.25 * steps)


def _hex(*parts: object, length: int) -> str:
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:length]


def _done[T](value: T) -> Future[T]:
    future: Future[T] = Future()
    future.set_result(value)
    return future


def _failed[T](error: BaseException) -> Future[T]:
    future: Future[T] = Future()
    future.set_exception(error)
    return future


@dataclass
class FakeCheckpoint:
    """A saved checkpoint: what the server stores and lists."""

    session_id: str
    run_id: str
    name: str
    kind: Literal["training", "sampler"]
    path: str
    steps: int
    row: CheckpointRow


@dataclass
class FakeSessionState:
    """One training session; ``alive`` turns false when its service closes."""

    session_id: str
    alive: bool = True
    runs: list[str] = field(default_factory=list)


class FakeRun:
    """A LoRA run in a fake session. See the module docstring."""

    def __init__(
        self, owner: FakeFireworksTrainingClient, session: FakeSessionState, *, steps: int = 0
    ) -> None:
        self._owner = owner
        self._session = session
        self.run_id = f"run-{_hex(session.session_id, len(session.runs), length=32)}"
        session.runs.append(self.run_id)
        self.steps = steps
        self.pending_gradients = False
        self.adam: list[AdamConfig] = []

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[ForwardOutput]:
        failure = self._owner._pending_failure("forward_backward", "a forward-backward pass")
        if failure is not None:
            return _failed(failure)
        self._check_alive()
        loss = loss_at(self.steps)
        weight = sum(sum(example.weights) for example in batch)
        self.pending_gradients = True
        metrics = {
            "loss:sum": loss * weight,
            "total_tokens:sum": float(sum(len(example.input_tokens) for example in batch)),
            "response_tokens": weight,
        }
        return _done(ForwardOutput(metrics, self._logprobs(batch, loss)))

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[ForwardOutput]:
        failure = self._owner._pending_failure("forward", "a forward pass")
        if failure is not None:
            return _failed(failure)
        self._check_alive()
        metrics = {"total_tokens:sum": float(sum(len(e.input_tokens) for e in batch))}
        return _done(ForwardOutput(metrics, self._logprobs(batch, loss_at(self.steps))))

    def optim_step(self, adam: AdamConfig) -> PendingResult[OptimStepOutput]:
        failure = self._owner._pending_failure("optim_step", "an optimizer step")
        if failure is not None:
            return _failed(failure)
        self._check_alive()
        self.adam.append(adam)
        if self.pending_gradients:
            self.steps += 1
            self.pending_gradients = False
        metrics = {"lr:last": adam.learning_rate, "step_id:last": float(self.steps)}
        return _done(OptimStepOutput(metrics))

    def save_state(self, name: str) -> PendingResult[str]:
        failure = self._owner._pending_failure("save_state", "saving a training checkpoint")
        if failure is not None:
            return _failed(failure)
        self._check_alive()
        checkpoint = self._owner._save(self._session, self, name, kind="training")
        return _done(checkpoint.path)

    def save_weights_for_sampler(self, name: str) -> PendingResult[str]:
        failure = self._owner._pending_failure("save_weights_for_sampler", "saving a sampler")
        if failure is not None:
            raise failure  # the SDK saves sampler weights synchronously
        self._check_alive()
        checkpoint = self._owner._save(self._session, self, name, kind="sampler")
        return _done(checkpoint.path)

    def load_state(self, path: str) -> PendingResult[None]:
        failure = self._owner._pending_failure("load_state", "loading a training checkpoint")
        if failure is not None:
            return _failed(failure)
        self._check_alive()
        self.steps = self._owner._training_checkpoint(path).steps
        self.pending_gradients = False
        return _done(None)

    def _check_alive(self) -> None:
        if not self._session.alive:
            raise ExternalError(
                f"the training session {self._session.session_id} has ended", source=SOURCE
            )

    @staticmethod
    def _logprobs(batch: Sequence[TokenizedExample], loss: float) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(-loss for _ in example.target_tokens) for example in batch)


class FakeSampler:
    """Returns the owner's canned tokens for every prompt."""

    def __init__(self, owner: FakeFireworksTrainingClient, model: str) -> None:
        self._owner = owner
        self.model = model
        self.closed = False
        self.prompts: list[tuple[int, ...]] = []

    def sample(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        stop: Sequence[int] = (),
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> PendingResult[SampledTokens]:
        failure = self._owner._pending_failure("sample", "sampling")
        if failure is not None:
            return _failed(failure)
        if self.closed:
            raise ExternalError("the sampler is closed", source=SOURCE)
        self.prompts.append(tuple(prompt))
        tokens = list(self._owner.sample_tokens)
        stop_reason = "length"
        for index, token in enumerate(tokens):
            if token in stop:
                tokens, stop_reason = tokens[: index + 1], "stop"
                break
        if len(tokens) > max_tokens:
            tokens, stop_reason = tokens[:max_tokens], "length"
        return _done(SampledTokens(tuple(tokens), stop_reason))

    def close(self) -> None:
        self.closed = True


class FakeService:
    """One fake training session. Closing it ends the session."""

    def __init__(self, owner: FakeFireworksTrainingClient, session: FakeSessionState) -> None:
        self._owner = owner
        self._session = session
        self.runs: list[FakeRun] = []
        self.samplers: list[FakeSampler] = []

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def session_name(self) -> str:
        return f"accounts/{self._owner.account_id}/trainingSessions/{self.session_id}"

    @property
    def closed(self) -> bool:
        return not self._session.alive

    def create_lora_training_client(
        self, base_model: str, rank: int, *, alpha: int | None = None, seed: int | None = None
    ) -> TrainingRun:
        self._owner.calls.append(("create_lora_training_client", base_model, rank, alpha, seed))
        self._start("create_lora_training_client", f"starting a LoRA run on {base_model}")
        run = FakeRun(self._owner, self._session)
        self.runs.append(run)
        return run

    def create_training_client_from_state_with_optimizer(self, path: str) -> TrainingRun:
        self._owner.calls.append(("create_training_client_from_state_with_optimizer", path))
        self._start(
            "create_training_client_from_state_with_optimizer",
            "resuming a Fireworks training checkpoint",
        )
        checkpoint = self._owner._training_checkpoint(path)
        run = FakeRun(self._owner, self._session, steps=checkpoint.steps)
        self.runs.append(run)
        return run

    def create_sampling_client(
        self, *, tokenizer: object, base_model: str | None = None, model_path: str | None = None
    ) -> TokenSampler:
        self._owner.calls.append(("create_sampling_client", base_model, model_path))
        if (base_model is None) == (model_path is None):
            raise UsageError("a Fireworks sampler needs exactly one of base_model and model_path")
        failure = self._owner._pending_failure("create_sampling_client", "opening a sampler")
        if failure is not None:
            raise failure
        if model_path is not None:
            self._owner._checkpoint(model_path, kind="sampler")
        sampler = FakeSampler(self._owner, model_path or self.session_name)
        self.samplers.append(sampler)
        return sampler

    def close(self) -> None:
        if self._session.alive:
            self._owner.calls.append(("close_service", self.session_id))
            self._session.alive = False

    def _start(self, operation: str, what: str) -> None:
        if not self._session.alive:
            raise ExternalError(f"the training session {self.session_id} has ended", source=SOURCE)
        failure = self._owner._pending_failure(operation, what)
        if failure is not None:
            if isinstance(failure, ProviderCapacityError):
                # As the real wrapper does: end the session, then say nothing is left running.
                self.close()
                failure.hint = CAPACITY_RELEASED_HINT
            raise failure


class FakeFireworksTrainingClient:
    """The Fireworks Training API in memory. See the module docstring."""

    def __init__(
        self, *, account_id: str = ACCOUNT, sample_tokens: Sequence[int] = DEFAULT_SAMPLE_TOKENS
    ) -> None:
        self._account_id = account_id
        self.sample_tokens: tuple[int, ...] = tuple(sample_tokens)
        self.sessions: dict[str, FakeSessionState] = {}
        self.services: list[FakeService] = []
        self.checkpoints: list[FakeCheckpoint] = []
        self.promoted: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False
        self._failures: dict[str, list[FailureKind]] = {}
        self._clock = 0

    @property
    def account_id(self) -> str:
        return self._account_id

    def fail(self, operation: str, kind: FailureKind, *, times: int = 1) -> None:
        """Fail the next ``times`` calls of ``operation`` (one of :data:`OPERATIONS`) with ``kind``."""
        if operation not in OPERATIONS:
            raise ValueError(f"unknown operation {operation!r}; known: {sorted(OPERATIONS)}")
        if kind not in _FAILURES:
            raise ValueError(f"unknown failure kind {kind!r}")
        self._failures.setdefault(operation, []).extend([kind] * times)

    def service(self) -> TrainingService:
        self.calls.append(("service",))
        failure = self._pending_failure("service", "starting a Fireworks training session")
        if failure is not None:
            raise failure
        session_id = f"ts-{_hex(self._account_id, len(self.sessions), length=24)}"
        session = FakeSessionState(session_id)
        self.sessions[session_id] = session
        service = FakeService(self, session)
        self.services.append(service)
        return service

    def list_session_checkpoints(self, session_name: str) -> list[CheckpointRow]:
        self.calls.append(("list_session_checkpoints", session_name))
        failure = self._pending_failure("list_session_checkpoints", "listing checkpoints")
        if failure is not None:
            raise failure
        session_id = session_name.rsplit("/", 1)[-1]
        if session_id not in self.sessions:
            raise ExternalError(f"no training session {session_name}", source=SOURCE)
        return sort_rows([c.row for c in self.checkpoints if c.session_id == session_id])

    def promote_session_checkpoint(
        self, resource_name: str, output_model_id: str, base_model: str
    ) -> dict[str, Any]:
        self.calls.append(("promote_session_checkpoint", resource_name, output_model_id))
        what = f"promoting a Fireworks checkpoint to the model {output_model_id}"
        if not MODEL_ID_PATTERN.fullmatch(output_model_id):
            raise UsageError(f"{output_model_id!r} is not a valid Fireworks model id")
        failure = self._pending_failure("promote_session_checkpoint", what)
        if failure is not None:
            raise failure
        checkpoint = next(
            (c for c in self.checkpoints if c.row.resource_name == resource_name), None
        )
        if checkpoint is None or not checkpoint.row.promotable:
            raise fireworks_error(FakeSdkError("checkpoint not found", status_code=404), what=what)
        if not self.sessions[checkpoint.session_id].alive:
            raise fireworks_error(
                FakeSdkError("the training session has expired", status_code=400), what=what
            )
        if len(checkpoint.name) > MAX_CHECKPOINT_NAME_LEN:
            raise fireworks_error(
                FakeSdkError(
                    f"checkpoint name {checkpoint.name!r} is longer than "
                    f"{MAX_CHECKPOINT_NAME_LEN} characters",
                    status_code=400,
                ),
                what=what,
            )
        if output_model_id in self.promoted:
            raise fireworks_error(FakeSdkError("model already exists", status_code=409), what=what)
        model = {
            "name": f"accounts/{self._account_id}/models/{output_model_id}",
            "kind": "HF_PEFT_ADDON",
            "state": "READY",
            "peftDetails": {"baseModel": base_model},
        }
        self.promoted[output_model_id] = model
        return dict(model)

    def close(self) -> None:
        self.closed = True

    # Internals shared with the fake runs and services.

    def _pending_failure(self, operation: str, what: str) -> Exception | None:
        queue = self._failures.get(operation)
        if not queue:
            return None
        return fireworks_error(_FAILURES[queue.pop(0)](), what=what)

    def _save(
        self,
        session: FakeSessionState,
        run: FakeRun,
        name: str,
        *,
        kind: Literal["training", "sampler"],
    ) -> FakeCheckpoint:
        stored = name if kind == "training" else f"{name}-{_hex(run.run_id, name, length=8)}"
        checkpoint_name = f"{run.run_id}-{stored}"
        self._clock += 1
        row = CheckpointRow(
            checkpoint_name=checkpoint_name,
            checkpoint_type=TRAINING_CHECKPOINT if kind == "training" else SAMPLER_CHECKPOINT,
            promotable=kind == "sampler",
            create_time=(_START + timedelta(seconds=self._clock))
            .isoformat()
            .replace("+00:00", "Z"),
            resource_name=(
                f"accounts/{self._account_id}/trainingSessions/{session.session_id}"
                f"/checkpoints/{checkpoint_name}"
            ),
        )
        checkpoint = FakeCheckpoint(
            session_id=session.session_id,
            run_id=run.run_id,
            name=name,
            kind=kind,
            path=f"{self._account_id}/{run.run_id}/{stored}",
            steps=run.steps,
            row=row,
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def _checkpoint(self, path: str, *, kind: Literal["training", "sampler"]) -> FakeCheckpoint:
        for checkpoint in self.checkpoints:
            if checkpoint.path == path and checkpoint.kind == kind:
                return checkpoint
        raise fireworks_error(
            FakeSdkError(f"no {kind} checkpoint at {path}", status_code=404),
            what=f"opening the {kind} checkpoint {path}",
        )

    def _training_checkpoint(self, path: str) -> FakeCheckpoint:
        return self._checkpoint(path, kind="training")
