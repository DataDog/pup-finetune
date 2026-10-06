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

Failures are injected with :meth:`FakeLoops.fail_on`, at the call or in its future. Every SDK
call is recorded in ``loops.calls`` with its keyword arguments.

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
"""

from __future__ import annotations

import json as jsonlib
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from pup_finetune.providers.baseten.client import MAX_CHECKPOINT_NAME_LEN

Answer = tuple[int, Any] | httpx.Response | Callable[[httpx.Request], httpx.Response]

_CLOSED = "Cannot send a request, as the client has been closed."
_RUN_ROUTE = re.compile(r"/v1/loops/runs/([A-Za-z0-9_-]+)(/deactivate)?")
_DEPLOYMENT_ROUTE = re.compile(
    r"/v1/models/([A-Za-z0-9_-]+)/deployments/([A-Za-z0-9_-]+)(/deactivate)?"
)

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

    ``logprobs=False`` leaves ``loss_fn_outputs`` empty, so only the loss is reported.
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

    def __init__(self, *, logprobs: bool = True, completion: Sequence[int] = COMPLETION) -> None:
        self.logprobs = logprobs
        self.completion = list(completion)
        self.calls: list[LoopsCall] = []
        self.sessions: list[str] = []
        self.runs: dict[str, FakeRun] = {}
        self.trainers: list[FakeTrainingClient] = []
        self.checkpoints: dict[str, FakeCheckpoint] = {}
        self.samplers: dict[str, FakeSampler] = {}
        self.sampling_clients: list[FakeSamplingClient] = []
        self._failures: dict[tuple[str, int], tuple[BaseException, bool]] = {}

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

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("POST", "/v1/loops/samplers"):
            return self._create_sampler(request)
        deployment = _DEPLOYMENT_ROUTE.fullmatch(request.url.path)
        if deployment is not None:
            return self._handle_deployment(request, deployment)
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
