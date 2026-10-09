# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten fakes: the Loops SDK (``FakeLoops``) and Baseten's HTTP routes (``BasetenTransport``).

``FakeLoops`` stands in for the ``baseten.loops`` module, in memory and deterministically;
code under test takes it as ``loops=``::

    loops = FakeLoops()
    service = loops_service(KEY, loops=loops)  # session session1
    trainer = service.create_lora_training_client(base_model="Qwen/Qwen3.5-9B", rank=8, ...)
    out = trainer.forward_backward(datums, "cross_entropy").result()
    trainer.optim_step(loops.AdamParams(learning_rate=1e-4)).result()
    path = trainer.save_weights_for_sampler("e1").result().path  # bt://loops:run1/sampler_weights/e1
    http = BasetenHTTP(KEY, transport=loops.transport())
    http.deactivate_run(trainer.run_id)

What it mimics:

- Every ``ServiceClient`` opens a new session (``session1``, ``session2``...) and every
  trainer a new run (``run1``...). Building a ``ServiceClient`` is what creates the session.
- Training calls return futures. Outputs have the SDK's shape: ``loss`` and
  ``metrics["loss"]`` are the per-token mean loss and ``loss_fn_outputs`` has, per datum, the
  ``logprobs`` of its targets (``None`` where the weight is 0). The per-token loss is
  :func:`loss_at` of the optimizer steps that applied gradients; a checkpoint saves that
  count and loading it restores it, so ``forward`` changes nothing and repeats exactly.
- Checkpoints are saved as ``bt://loops:<run>/state/<name>`` and
  ``bt://loops:<run>/sampler_weights/<name>``; an empty name or one over
  :data:`MAX_CHECKPOINT_NAME_LEN` is refused with HTTP 422. Loading a sampler, unknown or
  expired (:meth:`FakeLoops.expire`) checkpoint fails when its future is resolved.
- Closing a training client ends only its HTTP client, as in the SDK: every later call
  raises, but the run stays active until it is deactivated over HTTP
  (``POST /v1/loops/runs/<id>/deactivate``, served by :meth:`FakeLoops.transport`). A
  deactivated run reads ``INACTIVE`` and refuses training calls with HTTP 404.
- ``POST /v1/loops/samplers`` creates a sampler (``sampler1``...) on its own deployment
  (``model1``/``dep1``...) serving a saved sampler checkpoint (an unknown or expired one is
  refused with HTTP 404) or the base model; the deployment reads ``ACTIVE`` until it is
  deactivated (``POST /v1/models/<m>/deployments/<d>/deactivate``). A ``SamplingClient`` of
  it answers ``completion`` (token ids; by default the code points of ``Hello.``), cut after
  the first stop token id of ``SamplingParams.stop`` or at ``max_tokens``, and refuses to
  sample once closed, once its deployment is inactive or when its checkpoint expired.
- Serving: ``GET /v1/loops/checkpoints`` describes a saved checkpoint (``ckpt1``...; none for
  an unknown or expired one), and :meth:`FakeLoops.sampler_checkpoint` leaves one as a
  finished run does. ``GET /v1/instance_types`` lists :attr:`FakeLoops.instance_types`.
  ``POST /v1/loops/checkpoints/deploy`` creates a model under the requested name
  (``dmodel1``...) with one deployment (``ddep1``...) of the sampler checkpoints, or adds one to
  the model of that name, as Baseten does; it refuses (HTTP 400 or 404) an unknown instance
  type or checkpoint, and every create without BDN volumes (``bdn=False``). A deployment reads
  :attr:`FakeLoops.statuses` in turn (the last one stays), reports its autoscaling settings
  (:attr:`FakeLoops.replicas`, changed by ``PATCH .../autoscaling_settings``) once it is
  ``ACTIVE`` or scaled to zero, and its config's ``training_checkpoints``. ``GET /v1/models``
  (by ``name``), ``GET /v1/models/<m>``, its deployments and ``DELETE /v1/models/<m>`` serve
  :attr:`FakeLoops.models`. A deployment's chat route
  (``model-<m>.api.baseten.co/deployment/<d>/sync/v1/chat/completions``) answers the
  checkpoint's name with ``OK`` once it is ready, after :attr:`FakeLoops.chat_script`, and
  any other model with HTTP 404.

Failures are injected with :meth:`FakeLoops.fail_on`, at the call or in its future. Every SDK
call is recorded in ``loops.calls`` with its keyword arguments. :attr:`FakeLoops.fail` and
:attr:`FakeLoops.lost` inject an answer or an exception into a serving route (``deploy``,
``status``, ``delete``, ``autoscaling``), once, before it takes effect or after.

``BasetenTransport`` answers by ``"METHOD /path"`` and records every request::

    transport = baseten_transport(
        {
            "GET /v1/loops/capabilities": (200, {"supported_models": [...]}),
            "GET /v1/loops/runs/run1": [(200, active), (200, inactive)],  # in order; last repeats
            "POST /v1/loops/checkpoints/deploy": (200, {"model_id": "m1", "deployment_id": "d1"}),
        }
    )
    http = BasetenHTTP(KEY, transport=transport)
    transport.calls  # ["GET /v1/loops/capabilities", ...]

An answer is ``(status, json_body)``, an ``httpx.Response`` or a callable taking the request.
A request without a route goes to ``fallback`` (``FakeLoops.transport`` passes its own
routes), else fails the test.

``FakeTrainingJobs`` is what ``train --remote`` reads: teams and their secret names, training
projects, jobs with scripted statuses (:meth:`~FakeTrainingJobs.set_status`) and the log
lines a job wrote (:meth:`~FakeTrainingJobs.emit`); its :meth:`~FakeTrainingJobs.routes` merge
into :meth:`FakeLoops.transport`.
"""

from __future__ import annotations

import functools
import json as jsonlib
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from pup_finetune.providers.baseten.client import MAX_CHECKPOINT_NAME_LEN
from tests.fakes.chat_api import completion

Answer = tuple[int, Any] | httpx.Response | Callable[[httpx.Request], httpx.Response]

_CLOSED = "Cannot send a request, as the client has been closed."
_RUN_ROUTE = re.compile(r"/v1/loops/runs/([A-Za-z0-9_-]+)(/deactivate)?")
_DEPLOYMENT_ROUTE = re.compile(
    r"/v1/models/([A-Za-z0-9_-]+)/deployments/([A-Za-z0-9_-]+)(/deactivate)?"
)
_MODEL_ROUTE = re.compile(
    r"/v1/models/([A-Za-z0-9_-]+)"
    r"(?:/deployments(?:/([A-Za-z0-9_-]+)(/config|/autoscaling_settings)?)?)?"
)
_CHAT_ROUTE = re.compile(r"/deployment/([A-Za-z0-9_-]+)/sync/v1/chat/completions")
_CHAT_HOST = re.compile(r"model-([A-Za-z0-9_-]+)\.api\.baseten\.co")
_SERVING: frozenset[str] = frozenset({"ACTIVE", "SCALED_TO_ZERO"})

COMPLETION: tuple[int, ...] = tuple(ord(char) for char in "Hello.")
"""What a fake sampler answers by default: token ids a character-level tokenizer reads as text."""


def loss_at(steps: int) -> float:
    """The fake per-token loss after ``steps`` optimizer steps that applied gradients."""
    return 2.0 / (1.0 + 0.25 * steps)


def http_status_error(
    status: int,
    *,
    json: Any = None,
    headers: Mapping[str, str] | None = None,
    url: str = "https://api.baseten.co/v1/loops/trainers",
) -> httpx.HTTPStatusError:
    """The error the SDK raises for an HTTP error answer, with the response attached."""
    request = httpx.Request("POST", url)
    response = httpx.Response(status, json=json, headers=headers, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


# --- the Loops SDK types -----------------------------------------------------------------


@dataclass(frozen=True)
class LoopsCall:
    """One recorded SDK call."""

    method: str
    kwargs: dict[str, Any]


@dataclass
class FakeTensorData:
    data: list[Any]
    dtype: str
    shape: list[int]

    def tolist(self) -> list[Any]:
        return list(self.data)


@dataclass
class FakeModelInput:
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> FakeModelInput:
        return cls(list(tokens))

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, FakeTensorData] = field(default_factory=dict)


@dataclass
class FakeAdamParams:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


@dataclass
class FakeForwardBackwardOutput:
    loss: float = 0.0
    loss_fn_outputs: list[dict[str, FakeTensorData]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeOptimStepResponse:
    metrics: dict[str, float] | None = None


@dataclass
class FakeSaveWeightsResponse:
    path: str = ""


@dataclass
class FakeLoadWeightsResponse:
    step: int = 0


@dataclass
class FakeSamplingParams:
    max_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | list[int] | None = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    structured_outputs: dict[str, Any] | None = None


@dataclass
class FakeSampledSequence:
    tokens: list[int] = field(default_factory=list)
    stop_reason: Literal["stop", "length"] = "length"


@dataclass
class FakeSampleResponse:
    sequences: list[FakeSampledSequence] = field(default_factory=list)


@dataclass(frozen=True)
class FakeBasetenDeployment:
    """Stands in for ``baseten.loops.sampling_client.BasetenDeployment``."""

    base_url: str
    model_id: str
    deployment_id: str
    api_base_url: str


class FakeSamplerTerminatedError(RuntimeError):
    """Stands in for ``baseten.loops.SamplerTerminatedError``: the sampler stopped."""


class FakeServerShutdownError(Exception):
    """Stands in for ``baseten.loops.ServerShutdownError`` (the trainer pod is going away)."""


class FakeRemoteOpError(Exception):
    """Stands in for ``baseten.loops.RemoteOpError``: the trainer reports the call failed."""


# --- runs, checkpoints and clients -------------------------------------------------------


@dataclass
class FakeRun:
    """One Loops run: a trainer and its weights."""

    run_id: str
    session_id: str
    settings: dict[str, Any]
    active: bool = True
    steps: int = 0
    """Optimizer steps that applied gradients: the weights' state."""
    gradients: bool = False
    """Whether gradients accumulated since the last optimizer step."""


@dataclass
class FakeCheckpoint:
    path: str
    run_id: str
    kind: Literal["state", "sampler_weights"]
    steps: int
    expired: bool = False


@dataclass
class FakeSampler:
    """One Loops sampler: a Baseten deployment serving a checkpoint or the base model."""

    sampler_id: str
    session_id: str
    model_id: str
    deployment_id: str
    model_path: str | None
    base_model: str | None
    max_seq_len: int
    active: bool = True

    @property
    def base_url(self) -> str:
        return f"https://model-{self.model_id}.api.baseten.co/deployment/{self.deployment_id}/sync"

    def record(self) -> dict[str, Any]:
        """The sampler as ``POST /v1/loops/samplers`` describes it."""
        return {
            "id": self.sampler_id,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "base_url": self.base_url,
        }


def instance_type(gpu_type: str, gpu_count: int, *, cpus: int = 22) -> dict[str, Any]:
    """An instance type as ``GET /v1/instance_types`` lists it."""
    return {
        "id": f"{gpu_type}x{cpus}x{120 * gpu_count}",
        "name": f"{gpu_type} {gpu_count}x{cpus}",
        "memory_limit_mib": 122_880 * gpu_count,
        "millicpu_limit": cpus * 1000,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "gpu_memory_limit_mib": 81_920,
    }


@dataclass
class FakeDeployment:
    """A dedicated deployment of sampler checkpoints, made by the checkpoint deploy route."""

    model_id: str
    deployment_id: str
    checkpoint_ids: list[str] | None
    """What its config's ``training_checkpoints`` names; ``None`` leaves them out."""
    instance_type: dict[str, Any]
    statuses: list[str]
    replicas: dict[str, int]
    status: str = "BUILDING"

    def record(self, *, read: bool = False) -> dict[str, Any]:
        """The deployment as ``GET /v1/models/<m>/deployments/<d>`` describes it; ``read``
        moves it to its next status."""
        if read and self.statuses:
            self.status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        serving = self.status in _SERVING
        return {
            "id": self.deployment_id,
            "model_id": self.model_id,
            "name": self.deployment_id,
            "created_at": "2026-10-07T12:00:00Z",
            "is_production": True,
            "is_development": False,
            "status": self.status,
            "active_replica_count": 1 if self.status == "ACTIVE" else 0,
            "autoscaling_settings": dict(self.replicas) if serving else None,
            "instance_type_name": self.instance_type["name"],
            "environment": "production",
        }


@dataclass
class FakeModel:
    """A Baseten model and its deployments."""

    model_id: str
    name: str
    deployments: list[FakeDeployment] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        first = self.deployments[0] if self.deployments else None
        return {
            "id": self.model_id,
            "name": self.name,
            "created_at": "2026-10-07T12:00:00Z",
            "deployments_count": len(self.deployments),
            "production_deployment_id": first.deployment_id if first else None,
            "development_deployment_id": None,
            "instance_type_name": first.instance_type["name"] if first else "",
            "team_name": "Default Team",
        }


def _done(value: Any) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(value)
    return future


def _failed(error: BaseException) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_exception(error)
    return future


class FakeTrainingClient:
    """What ``create_lora_training_client`` returns: a client of one :class:`FakeRun`."""

    def __init__(self, loops: FakeLoops, run: FakeRun) -> None:
        self._loops = loops
        self._run = run
        self.run_id: str | None = run.run_id
        self.closed = False

    def forward_backward(
        self, data: list[FakeDatum], loss_fn: str = "cross_entropy"
    ) -> Future[FakeForwardBackwardOutput]:
        failed = self._call("forward_backward", {"data": data, "loss_fn": loss_fn})
        if failed is not None:
            return failed
        output = self._output(data)
        self._run.gradients = True
        return _done(output)

    def forward(
        self, data: list[FakeDatum], loss_fn: str = "cross_entropy"
    ) -> Future[FakeForwardBackwardOutput]:
        failed = self._call("forward", {"data": data, "loss_fn": loss_fn})
        return _done(self._output(data)) if failed is None else failed

    def optim_step(self, adam_params: FakeAdamParams) -> Future[FakeOptimStepResponse]:
        failed = self._call("optim_step", {"adam_params": adam_params})
        if failed is not None:
            return failed
        if self._run.gradients:
            self._run.steps += 1
            self._run.gradients = False
        return _done(FakeOptimStepResponse(metrics={"step": float(self._run.steps)}))

    def save_state(self, name: str, ttl_seconds: int | None = None) -> Future[Any]:
        return self._save("save_state", "state", name)

    def save_weights_for_sampler(self, name: str, ttl_seconds: int | None = None) -> Future[Any]:
        return self._save("save_weights_for_sampler", "sampler_weights", name)

    def load_state(self, path: str) -> Future[Any]:
        return self._load("load_state", path)

    def load_state_with_optimizer(self, path: str) -> Future[Any]:
        return self._load("load_state_with_optimizer", path)

    def close(self) -> None:
        self._loops._record("close", {})
        self.closed = True

    def _call(self, method: str, kwargs: dict[str, Any]) -> Future[Any] | None:
        """Record the call and refuse it as the SDK would; a failed future to return, if any."""
        failed = self._loops._record(method, kwargs)
        if self.closed:
            raise RuntimeError(_CLOSED)
        if not self._run.active:
            raise http_status_error(404, json={"detail": f"run {self._run.run_id} is not active"})
        return failed

    def _output(self, data: list[FakeDatum]) -> FakeForwardBackwardOutput:
        loss = loss_at(self._run.steps)
        outputs: list[dict[str, FakeTensorData]] = []
        for datum in data:
            inputs = datum.loss_fn_inputs
            weights = inputs["weights"].data
            if not len(weights) == len(inputs["target_tokens"].data) == datum.model_input.length:
                raise http_status_error(400, json={"detail": "datum lengths differ"})
            logprobs = [-loss if weight > 0 else None for weight in weights]
            outputs.append(
                {"logprobs": FakeTensorData(data=logprobs, dtype="float32", shape=[len(weights)])}
            )
        if not self._loops.logprobs:
            outputs = []
        return FakeForwardBackwardOutput(loss=loss, loss_fn_outputs=outputs, metrics={"loss": loss})

    def _save(
        self, method: str, kind: Literal["state", "sampler_weights"], name: str
    ) -> Future[Any]:
        failed = self._call(method, {"name": name})
        if not name or len(name) > MAX_CHECKPOINT_NAME_LEN:
            raise http_status_error(422, json={"detail": f"invalid checkpoint name {name!r}"})
        if failed is not None:
            return failed
        path = f"bt://loops:{self._run.run_id}/{kind}/{name}"
        self._loops.checkpoints[path] = FakeCheckpoint(
            path, self._run.run_id, kind, self._run.steps
        )
        return _done(FakeSaveWeightsResponse(path=path))

    def _load(self, method: str, path: str) -> Future[Any]:
        failed = self._call(method, {"path": path})
        if failed is not None:
            return failed
        checkpoint = self._loops.checkpoints.get(path)
        if checkpoint is None or checkpoint.expired or checkpoint.kind != "state":
            return _failed(FakeRemoteOpError(f"no training checkpoint at {path}"))
        self._run.steps = checkpoint.steps
        self._run.gradients = False
        return _done(FakeLoadWeightsResponse(step=checkpoint.steps))


class FakeSamplingClient:
    """What ``SamplingClient(...)`` returns: a client of one :class:`FakeSampler`."""

    def __init__(self, loops: FakeLoops, sampler: FakeSampler, checkpoint_path: str | None) -> None:
        self._loops = loops
        self.sampler = sampler
        self.checkpoint_path = checkpoint_path
        self.closed = False

    def sample(
        self,
        prompt: FakeModelInput,
        num_samples: int = 1,
        sampling_params: FakeSamplingParams | None = None,
    ) -> FakeSampleResponse:
        loops = self._loops
        kwargs = {"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params}
        loops._raise_on_failure("sample", kwargs)
        if self.closed:
            raise RuntimeError(_CLOSED)
        if not self.sampler.active:
            raise FakeSamplerTerminatedError(f"sampler {self.sampler.deployment_id} is INACTIVE")
        if self.checkpoint_path is not None and not loops._serves(self.checkpoint_path):
            raise http_status_error(404, json={"detail": "checkpoint not found"})
        params = sampling_params or FakeSamplingParams()
        if prompt.length + (params.max_tokens or 0) > self.sampler.max_seq_len:
            raise http_status_error(400, json={"detail": "longer than the sampler's context"})
        stops = params.stop if isinstance(params.stop, list) else []
        tokens = list(loops.completion)
        cut = next((index + 1 for index, token in enumerate(tokens) if token in stops), None)
        tokens = tokens[:cut]
        reason: Literal["stop", "length"] = "stop"
        if params.max_tokens is not None and len(tokens) > params.max_tokens:
            tokens, reason = tokens[: params.max_tokens], "length"
        sequence = FakeSampledSequence(tokens=tokens, stop_reason=reason)
        return FakeSampleResponse(sequences=[sequence] * num_samples)

    def close(self) -> None:
        self._loops._record("close", {})
        self.closed = True


class FakeServiceClient:
    """What ``ServiceClient(...)`` returns: one Loops session."""

    def __init__(self, loops: FakeLoops, session_id: str) -> None:
        self._loops = loops
        self.session_id = session_id

    def create_lora_training_client(self, base_model: str, **kwargs: Any) -> FakeTrainingClient:
        settings = {"base_model": base_model, **kwargs}
        self._loops._raise_on_failure("create_lora_training_client", settings)
        run = FakeRun(f"run{len(self._loops.runs) + 1}", self.session_id, settings)
        self._loops.runs[run.run_id] = run
        trainer = FakeTrainingClient(self._loops, run)
        self._loops.trainers.append(trainer)
        return trainer


class FakeLoops:
    """A fake ``baseten.loops`` module; see the module docstring.

    ``logprobs=False`` leaves ``loss_fn_outputs`` empty, so only the loss is reported;
    ``bdn=False`` refuses every checkpoint deployment, as a workspace without BDN volumes.
    """

    Datum = FakeDatum
    ModelInput = FakeModelInput
    TensorData = FakeTensorData
    AdamParams = FakeAdamParams
    SamplingParams = FakeSamplingParams
    BasetenDeployment = FakeBasetenDeployment
    ServerShutdownError = FakeServerShutdownError
    RemoteOpError = FakeRemoteOpError
    SamplerTerminatedError = FakeSamplerTerminatedError

    def __init__(
        self, *, logprobs: bool = True, completion: Sequence[int] = COMPLETION, bdn: bool = True
    ) -> None:
        self.logprobs = logprobs
        self.bdn = bdn
        self.completion = list(completion)
        self.calls: list[LoopsCall] = []
        self.sessions: list[str] = []
        self.runs: dict[str, FakeRun] = {}
        self.trainers: list[FakeTrainingClient] = []
        self.checkpoints: dict[str, FakeCheckpoint] = {}
        self.samplers: dict[str, FakeSampler] = {}
        self.sampling_clients: list[FakeSamplingClient] = []
        self._failures: dict[tuple[str, int], tuple[BaseException, bool]] = {}
        self.instance_types = [
            instance_type("H100", 1),
            instance_type("H100", 2),
            instance_type("A100", 1),
        ]
        self.models: dict[str, FakeModel] = {}
        self.deleted: list[str] = []
        """The ids of the models deleted, in order."""
        self._deployed = 0
        self.statuses = ["DEPLOYING", "ACTIVE"]
        """What successive reads of a new deployment report; the last one stays."""
        self.replicas = {"min_replica": 1, "max_replica": 1}
        """A new deployment's autoscaling replicas."""
        self.chat_script: list[httpx.Response | Exception] = []
        """What a deployment's chat route answers (or raises) first, in order."""
        self.fail: dict[str, httpx.Response | BaseException] = {}
        """What a serving route answers or raises once, before it takes effect."""
        self.lost: dict[str, httpx.Response | BaseException] = {}
        """What a serving route answers or raises once, after it took effect."""
        self.after_deploy: Callable[[FakeDeployment], None] | None = None
        """Called once with the next deployment a create makes, before it answers."""

    def ServiceClient(self, **kwargs: Any) -> FakeServiceClient:
        self._raise_on_failure("ServiceClient", kwargs)
        session_id = f"session{len(self.sessions) + 1}"
        self.sessions.append(session_id)
        return FakeServiceClient(self, session_id)

    def SamplingClient(self, **kwargs: Any) -> FakeSamplingClient:
        self._raise_on_failure("SamplingClient", kwargs)
        deployment = kwargs["deployment"]
        sampler = next(
            sampler
            for sampler in self.samplers.values()
            if (sampler.model_id, sampler.deployment_id, sampler.base_url)
            == (deployment.model_id, deployment.deployment_id, deployment.base_url)
        )
        client = FakeSamplingClient(self, sampler, kwargs.get("checkpoint_path"))
        self.sampling_clients.append(client)
        return client

    def fail_on(
        self, method: str, *, error: BaseException, call: int = 1, on_result: bool = False
    ) -> None:
        """Make the ``call``-th call (1-based) of ``method`` fail with ``error``.

        By default the call itself raises, as when the SDK cannot submit it; with
        ``on_result`` its future does, as when the trainer reports the call failed.
        """
        self._failures[(method, call)] = (error, on_result)

    def expire(self, path: str) -> None:
        """Expire the checkpoint at ``path``, as Baseten does after its time to live."""
        self.checkpoints[path].expired = True

    def sampler_checkpoint(self, name: str = "best", *, base_model: str = "Qwen/Qwen3.5-9B") -> str:
        """The ``bt://loops:`` path of a sampler checkpoint a finished, deactivated run left."""
        session_id = f"session{len(self.sessions) + 1}"
        self.sessions.append(session_id)
        run = FakeRun(f"run{len(self.runs) + 1}", session_id, {"base_model": base_model})
        run.active = False
        self.runs[run.run_id] = run
        path = f"bt://loops:{run.run_id}/sampler_weights/{name}"
        self.checkpoints[path] = FakeCheckpoint(path, run.run_id, "sampler_weights", 0)
        return path

    def checkpoint_id(self, path: str) -> str:
        """Baseten's id of the checkpoint at ``path``."""
        return f"ckpt{list(self.checkpoints).index(path) + 1}"

    def deploy_checkpoint(self, path: str, name: str) -> FakeModel:
        """Deploy the checkpoint at ``path`` as the model ``name``, as another client would."""
        body = {
            "checkpoint_ids": [self.checkpoint_id(path)],
            "model_name": name,
            "instance_type_id": self.instance_types[0]["id"],
            "hf_secret_name": "hf_access_token",
        }
        made = self._create_deployment(body)
        assert isinstance(made, FakeDeployment)
        return self.models[made.model_id]

    def model_named(self, name: str) -> FakeModel:
        (model,) = [model for model in self.models.values() if model.name == name]
        return model

    def deactivate(self, run_id: str) -> bool:
        """Deactivate ``run_id``; whether Baseten knows the run."""
        run = self.runs.get(run_id)
        if run is not None:
            run.active = False
        return run is not None

    def transport(
        self, routes: Mapping[str, Answer | list[Answer]] | None = None
    ) -> BasetenTransport:
        """A transport for ``BasetenHTTP`` serving ``routes`` first, then this fake's run routes."""
        return BasetenTransport(routes or {}, fallback=self._handle)

    @property
    def methods(self) -> list[str]:
        return [call.method for call in self.calls]

    def kwargs(self, method: str) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.calls if call.method == method]

    def _record(self, method: str, kwargs: dict[str, Any]) -> Future[Any] | None:
        """Record a call; raise its injected failure, or return it as a failed future."""
        self.calls.append(LoopsCall(method, kwargs))
        failure = self._failures.pop((method, len(self.kwargs(method))), None)
        if failure is None:
            return None
        error, on_result = failure
        if not on_result:
            raise error
        return _failed(error)

    def _raise_on_failure(self, method: str, kwargs: dict[str, Any]) -> None:
        failed = self._record(method, kwargs)
        if failed is not None:
            failed.result()

    def _serves(self, path: str) -> bool:
        """Whether ``path`` is a sampler checkpoint a sampler can serve."""
        checkpoint = self.checkpoints.get(path)
        return checkpoint is not None and not checkpoint.expired and checkpoint.kind != "state"

    def _create_sampler(self, request: httpx.Request) -> httpx.Response:
        body = jsonlib.loads(request.content)
        path = body.get("model_path")
        if body["session_id"] not in self.sessions or (path is not None and not self._serves(path)):
            return httpx.Response(404, json={"detail": "not found"})
        number = len(self.samplers) + 1
        sampler = FakeSampler(
            sampler_id=f"sampler{number}",
            session_id=body["session_id"],
            model_id=f"model{number}",
            deployment_id=f"dep{number}",
            model_path=path,
            base_model=body.get("base_model"),
            max_seq_len=body["max_seq_length"],
        )
        self.samplers[sampler.sampler_id] = sampler
        return httpx.Response(200, json={"sampler": sampler.record()})

    def _handle_deployment(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        model_id, deployment_id, deactivate = match.groups()
        found = (s for s in self.samplers.values() if s.deployment_id == deployment_id)
        sampler = next((s for s in found if s.model_id == model_id), None)
        if sampler is None:
            return httpx.Response(404, json={"detail": "not found"})
        if request.method == "POST" and deactivate:
            sampler.active = False
            return httpx.Response(200, json={"success": True})
        if request.method == "GET" and not deactivate:
            status = "ACTIVE" if sampler.active else "INACTIVE"
            return httpx.Response(
                200, json={"id": deployment_id, "model_id": model_id, "status": status}
            )
        raise AssertionError(f"no fake Baseten route for {request.method} {request.url.path}")

    def _injected(self, key: str, injected: dict[str, httpx.Response | BaseException]) -> Any:
        found = injected.pop(key, None)
        if isinstance(found, BaseException):
            raise found
        return found

    def _checkpoint_record(self, path: str) -> dict[str, Any]:
        checkpoint = self.checkpoints[path]
        return {
            "id": self.checkpoint_id(path),
            "run_id": checkpoint.run_id,
            "checkpoint_id": path.rsplit("/", 1)[1],
            "checkpoint_type": "lora",
            "base_model": self.runs[checkpoint.run_id].settings["base_model"],
            "target": "sampler" if checkpoint.kind == "sampler_weights" else "trainer",
            "sync_status": "COMPLETE",
        }

    def _find_checkpoints(self, request: httpx.Request) -> httpx.Response:
        path = request.url.params.get("checkpoint_path", "")
        found = self.checkpoints.get(path)
        records = [] if found is None or found.expired else [self._checkpoint_record(path)]
        return httpx.Response(200, json={"checkpoints": records})

    def _deploy(self, request: httpx.Request) -> httpx.Response:
        refused = self._injected("deploy", self.fail)
        if refused is not None:
            return refused
        made = self._create_deployment(jsonlib.loads(request.content))
        if isinstance(made, httpx.Response):
            return made
        hook, self.after_deploy = self.after_deploy, None
        if hook is not None:
            hook(made)
        lost = self._injected("deploy", self.lost)
        if lost is not None:
            return lost
        ids = {"model_id": made.model_id, "deployment_id": made.deployment_id}
        return httpx.Response(200, json=ids)

    def _create_deployment(self, body: dict[str, Any]) -> FakeDeployment | httpx.Response:
        """The deployment a checkpoint deploy of ``body`` makes, else Baseten's refusal."""
        types = [item for item in self.instance_types if item["id"] == body["instance_type_id"]]
        paths = {self.checkpoint_id(path): path for path in self.checkpoints}
        sent = [paths.get(checkpoint_id) for checkpoint_id in body["checkpoint_ids"]]
        if not self.bdn:
            return httpx.Response(
                400, json={"detail": "BDN volumes are not enabled for this organization"}
            )
        if not types or not body.get("hf_secret_name"):
            return httpx.Response(400, json={"detail": "invalid instance type or secret"})
        if not sent or not all(path is not None and self._serves(path) for path in sent):
            return httpx.Response(404, json={"detail": "checkpoint not found"})
        self._deployed += 1
        number = self._deployed
        named = [model for model in self.models.values() if model.name == body["model_name"]]
        model = named[0] if named else FakeModel(f"dmodel{number}", body["model_name"])
        self.models[model.model_id] = model
        deployment = FakeDeployment(
            model_id=model.model_id,
            deployment_id=f"ddep{number}",
            checkpoint_ids=list(body["checkpoint_ids"]),
            instance_type=types[0],
            statuses=list(self.statuses),
            replicas=dict(self.replicas),
        )
        model.deployments.append(deployment)
        return deployment

    def _dedicated(self, model_id: str, deployment_id: str) -> FakeDeployment | None:
        model = self.models.get(model_id)
        found = [d for d in model.deployments if d.deployment_id == deployment_id] if model else []
        return found[0] if found else None

    def _handle_model(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        model_id, deployment_id, sub = match.groups()
        route = (request.method, "deployments" in request.url.path, deployment_id, sub)
        model = self.models.get(model_id)
        if route == ("DELETE", False, None, None):
            failed = self._injected("delete", self.fail)
            if failed is not None:
                return failed
            if model is None:
                return httpx.Response(404, json={"detail": "not found"})
            del self.models[model_id]
            self.deleted.append(model_id)
            return self._injected("delete", self.lost) or httpx.Response(
                200, json={"id": model_id, "deleted": True}
            )
        if model is None:
            return httpx.Response(404, json={"detail": "not found"})
        if route == ("GET", False, None, None):
            return httpx.Response(200, json=model.record())
        if route == ("GET", True, None, None):
            records = [deployment.record() for deployment in model.deployments]
            return httpx.Response(200, json={"deployments": records})
        deployment = self._dedicated(model_id, str(deployment_id))
        if deployment is None:
            return httpx.Response(404, json={"detail": "not found"})
        if route[0] == "GET" and sub is None:
            failed = self._injected("status", self.fail)
            return failed or httpx.Response(200, json=deployment.record(read=True))
        if route[0] == "GET" and sub == "/config":
            checkpoints: dict[str, Any] = {"download_folder": "/tmp/training_checkpoints"}  # noqa: S108
            if deployment.checkpoint_ids is not None:
                checkpoints["loops_checkpoint_ids"] = deployment.checkpoint_ids
            config = {
                "model_name": model.name,
                "resources": {"instance_type": deployment.instance_type["id"]},
                "training_checkpoints": checkpoints,
            }
            return httpx.Response(200, json={"config": config, "raw_config": None})
        if route[0] == "PATCH" and sub == "/autoscaling_settings":
            failed = self._injected("autoscaling", self.fail)
            if failed is not None:
                return failed
            body = jsonlib.loads(request.content)
            deployment.replicas.update(
                {key: body[key] for key in body if key in deployment.replicas}
            )
            return httpx.Response(200, json={"status": "ACCEPTED", "message": "updated"})
        raise AssertionError(f"no fake Baseten route for {request.method} {request.url.path}")

    def _chat(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        host = _CHAT_HOST.fullmatch(request.url.host)
        deployment = self._dedicated(host[1], match[1]) if host else None
        if deployment is None or deployment.status not in _SERVING:
            return httpx.Response(404, json={"detail": "not found"})
        if self.chat_script:
            answer = self.chat_script.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        paths = {self.checkpoint_id(path): path for path in self.checkpoints}
        served = [paths[i].rsplit("/", 1)[1] for i in deployment.checkpoint_ids or [] if i in paths]
        model = jsonlib.loads(request.content).get("model")
        if model not in served:
            return httpx.Response(404, json={"message": f"The model `{model}` does not exist."})
        return completion("OK", model=str(model))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        if route == ("POST", "/v1/loops/samplers"):
            return self._create_sampler(request)
        if route == ("GET", "/v1/loops/checkpoints"):
            return self._find_checkpoints(request)
        if route == ("POST", "/v1/loops/checkpoints/deploy"):
            return self._deploy(request)
        if route == ("GET", "/v1/instance_types"):
            return httpx.Response(200, json={"instance_types": self.instance_types})
        if route == ("GET", "/v1/models"):
            name = request.url.params.get("name")
            models = [m.record() for m in self.models.values() if name in (None, m.name)]
            return httpx.Response(200, json={"models": models})
        chat = _CHAT_ROUTE.fullmatch(request.url.path)
        if chat is not None and request.method == "POST":
            return self._chat(request, chat)
        deployment = _DEPLOYMENT_ROUTE.fullmatch(request.url.path)
        if deployment is not None and deployment[1] not in self.models:
            return self._handle_deployment(request, deployment)
        model = _MODEL_ROUTE.fullmatch(request.url.path)
        if model is not None:
            return self._handle_model(request, model)
        match = _RUN_ROUTE.fullmatch(request.url.path)
        if match is None:
            raise AssertionError(f"no fake Baseten route for {request.method} {request.url.path}")
        run_id, deactivate = match.groups()
        run = self.runs.get(run_id)
        if run is None:
            return httpx.Response(404, json={"detail": "not found"})
        if request.method == "POST" and deactivate:
            self.deactivate(run_id)
            return httpx.Response(200, json={"id": run_id})
        if request.method == "GET" and not deactivate:
            status = "ACTIVE" if run.active else "INACTIVE"
            body = {"id": run_id, "session_id": run.session_id, "status": {"name": status}}
            return httpx.Response(200, json={"run": body})
        raise AssertionError(f"no fake Baseten route for {request.method} {request.url.path}")


# --- Baseten HTTP ----------------------------------------------------------------------


def _respond(answer: Answer, request: httpx.Request) -> httpx.Response:
    if isinstance(answer, httpx.Response):
        return answer
    if callable(answer):
        return answer(request)
    status, body = answer
    return httpx.Response(status, json=body)


class BasetenTransport(httpx.MockTransport):
    """An ``httpx.MockTransport`` answering by ``"METHOD /path"``; see the module docstring."""

    def __init__(
        self,
        routes: Mapping[str, Answer | list[Answer]],
        *,
        fallback: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        self.routes = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in routes.items()
        }
        self.fallback = fallback
        self.requests: list[httpx.Request] = []
        super().__init__(self._handle)

    @property
    def calls(self) -> list[str]:
        return [f"{request.method} {request.url.path}" for request in self.requests]

    def bodies(self, call: str) -> list[Any]:
        """The JSON bodies sent to ``call`` (``"METHOD /path"``), in order."""
        return [
            jsonlib.loads(request.content) if request.content else None
            for request in self.requests
            if f"{request.method} {request.url.path}" == call
        ]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        answers = self.routes.get(key)
        if answers:
            return _respond(answers.pop(0) if len(answers) > 1 else answers[0], request)
        if self.fallback is not None:
            return self.fallback(request)
        raise AssertionError(f"no fake Baseten route for {key}")


def baseten_transport(routes: Mapping[str, Answer | list[Answer]]) -> BasetenTransport:
    return BasetenTransport(routes)


def capabilities(*base_models: str, **fields: Any) -> dict[str, Any]:
    """A ``GET /v1/loops/capabilities`` body offering ``base_models``, each with ``fields``."""
    entry = {
        "max_seq_len": 131_072,
        "max_enabled_seq_len": 131_072,
        "max_context_length": 131_072,
        "supports_vision_language": False,
        "enabled": True,
        "enablement_details": None,
    }
    return {"supported_models": [{"model_name": name, **entry, **fields} for name in base_models]}


# --- Training jobs (train --remote) ------------------------------------------------------

_LEVELS: dict[str, int] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
DEFAULT_TEAM = "Baseten Research"


def _iso(moment: datetime) -> str:
    return moment.isoformat()


@dataclass
class FakeTrainingJob:
    """A training job: its status, when it was created (``None``: Baseten did not say) and
    last changed, and its log lines."""

    job_id: str
    project_id: str
    status: str
    created_at: datetime | None
    updated_at: datetime
    logs: list[dict[str, Any]] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "id": self.job_id,
            "training_project_id": self.project_id,
            "current_status": self.status,
            "error_message": None,
            "created_at": None if self.created_at is None else _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class FakeTrainingJobs:
    """Baseten's teams, team secrets, training projects, jobs and job logs, in memory.

    ``routes()`` answers ``GET /v1/teams``, ``GET /v1/teams/<id>/secrets``, ``GET
    /v1/training_projects``, ``GET .../<project>/jobs``, ``GET .../jobs/<job>`` and ``GET
    .../jobs/<job>/logs`` (``start_epoch_millis`` and ``end_epoch_millis`` inclusive,
    ``limit``, ``direction`` and ``min_level``) for what exists when it is called; pass them
    to :meth:`FakeLoops.transport`. ``fail`` answers a route kind (``teams``, ``secrets``,
    ``projects``, ``jobs``, ``job``, ``logs``) with its ``(status, body)`` instead.
    """

    def __init__(
        self,
        *,
        teams: list[dict[str, Any]] | None = None,
        secrets: Mapping[str, list[str]] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.now = now
        self.teams = (
            [{"id": "team1", "name": DEFAULT_TEAM, "default": True, "created_at": "2026-01-01"}]
            if teams is None
            else teams
        )
        self.secrets: dict[str, list[str]] = (
            {team["id"]: ["pup_finetune_api_key"] for team in self.teams}
            if secrets is None
            else dict(secrets)
        )
        self.projects: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, FakeTrainingJob] = {}
        self.fail: dict[str, tuple[int, Any]] = {}
        self.log_queries: list[dict[str, str]] = []

    def project(self, name: str, team: str = DEFAULT_TEAM) -> str:
        """The id of project ``name`` in ``team``, created when missing."""
        for project_id, project in self.projects.items():
            if (project["name"], project["team_name"]) == (name, team):
                return project_id
        project_id = f"proj{len(self.projects) + 1}"
        self.projects[project_id] = {"id": project_id, "name": name, "team_name": team}
        return project_id

    def create_job(
        self, project: str, team: str = DEFAULT_TEAM, *, status: str = "TRAINING_JOB_CREATED"
    ) -> str:
        project_id = self.project(project, team)
        job_id = f"job{len(self.jobs) + 1}"
        moment = self.now()
        self.jobs[job_id] = FakeTrainingJob(job_id, project_id, status, moment, moment)
        return job_id

    def set_status(self, job_id: str, status: str, *, at: datetime | None = None) -> None:
        job = self.jobs[job_id]
        job.status, job.updated_at = status, at or self.now()

    def emit(
        self,
        job_id: str,
        messages: Sequence[str],
        *,
        replica: str | None = "g00r0",
        level: str = "INFO",
        at: datetime | None = None,
    ) -> None:
        """Log ``messages`` for ``job_id``, a nanosecond apart from ``at`` (by default, now);
        a ``None`` replica is a platform line."""
        job = self.jobs[job_id]
        start = int((at or self.now()).timestamp() * 1_000_000_000)
        for index, message in enumerate(messages):
            line = {"timestamp": str(start + index), "message": message, "level": level}
            job.logs.append({**line, "replica": replica})

    def routes(self) -> dict[str, Answer]:
        routes: dict[str, Answer] = {
            "GET /v1/teams": lambda request: self._answer("teams", {"teams": self.teams}),
            "GET /v1/training_projects": lambda request: self._answer(
                "projects", {"training_projects": list(self.projects.values())}
            ),
        }
        for team in self.teams:
            routes[f"GET /v1/teams/{team['id']}/secrets"] = functools.partial(self._secrets, team)
        for project_id in self.projects:
            routes[f"GET /v1/training_projects/{project_id}/jobs"] = functools.partial(
                self._jobs, project_id
            )
        for job_id, job in self.jobs.items():
            url = f"GET /v1/training_projects/{job.project_id}/jobs/{job_id}"
            routes[url] = functools.partial(self._job, job_id)
            routes[f"{url}/logs"] = functools.partial(self._logs, job_id)
        return routes

    def _answer(self, kind: str, body: Any) -> httpx.Response:
        status, body = self.fail.get(kind, (200, body))
        return httpx.Response(status, json=body)

    def _secrets(self, team: Mapping[str, Any], request: httpx.Request) -> httpx.Response:
        names = self.secrets.get(team["id"], [])
        secrets = [
            {"id": f"s{i}", "name": name, "team_name": team["name"]} for i, name in enumerate(names)
        ]
        return self._answer("secrets", {"secrets": secrets})

    def _jobs(self, project_id: str, request: httpx.Request) -> httpx.Response:
        found = [job.record() for job in self.jobs.values() if job.project_id == project_id]
        body = {"training_project": self.projects[project_id], "training_jobs": found}
        return self._answer("jobs", body)

    def _job(self, job_id: str, request: httpx.Request) -> httpx.Response:
        job = self.jobs[job_id]
        body = {"training_project": self.projects[job.project_id], "training_job": job.record()}
        return self._answer("job", body)

    def _logs(self, job_id: str, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.log_queries.append(params)
        start, end = int(params["start_epoch_millis"]), int(params["end_epoch_millis"])
        least = _LEVELS.get(params.get("min_level", "DEBUG"), 0)
        found = [
            line
            for line in self.jobs[job_id].logs
            if start <= int(line["timestamp"]) // 1_000_000 <= end
            and _LEVELS.get(line["level"], 0) >= least
        ]
        found.sort(
            key=lambda line: int(line["timestamp"]), reverse=params.get("direction") == "desc"
        )
        return self._answer("logs", {"logs": found[: int(params.get("limit", "1000"))]})
