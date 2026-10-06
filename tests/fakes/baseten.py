# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten fakes: the Loops SDK module, and an HTTP router for ``BasetenHTTP``.

``FakeLoops`` stands in for ``baseten.loops``; code under test takes it as ``loops=``::

    loops = FakeLoops(train_losses=[2.0, 1.5], validation_losses=[1.0, 0.8])
    loops.fail_on("create_lora_training_client", error=http_status_error(429))
    loops.on_call("forward_backward", lambda: ..., call=2)

Losses answer call by call, cycling; ``metrics`` is added to every step output. Every SDK
call is recorded in ``loops.calls`` with its keyword arguments, and checkpoints come back as
``bt://loops:<run_id>/state/<name>`` and ``bt://loops:<run_id>/sampler_weights/<name>``.

``baseten_transport`` answers by ``"METHOD /path"`` and records every request::

    transport = baseten_transport(
        {
            "GET /v1/loops/capabilities": (200, {"supported_models": [...]}),
            "GET /v1/loops/runs/run1": [(200, active), (200, inactive)],  # in order; last repeats
            "POST /graphql/": graphql({"listed_instances": (200, {"data": ...})}),
        }
    )
    http = BasetenHTTP(KEY, transport=transport)
    transport.calls  # ["GET /v1/loops/capabilities", ...]

An answer is ``(status, json_body)``, an ``httpx.Response`` or a callable taking the request.
A request without a route fails the test.
"""

from __future__ import annotations

import json as jsonlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

Answer = tuple[int, Any] | httpx.Response | Callable[[httpx.Request], httpx.Response]


# --- the Loops SDK ---------------------------------------------------------------------


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


@dataclass
class FakeModelInput:
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: Sequence[int]) -> FakeModelInput:
        return cls(list(tokens))

    def to_ints(self) -> list[int]:
        return list(self.tokens)


@dataclass
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, FakeTensorData]


@dataclass
class FakeAdamParams:
    learning_rate: float = 1e-4


@dataclass
class FakeSamplingParams:
    max_tokens: int | None = None
    temperature: float = 1.0


@dataclass(frozen=True)
class FakeBasetenDeployment:
    base_url: str
    model_id: str
    deployment_id: str
    api_base_url: str


class FakeServerShutdownError(Exception):
    """Stands in for ``baseten.loops.ServerShutdownError`` (a transient failure)."""


@dataclass
class FakeStepOutput:
    loss: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeSaved:
    path: str


class FakeFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> Any:
        return self._value


class FakeTrainingClient:
    """What ``create_lora_training_client`` returns."""

    def __init__(self, loops: FakeLoops) -> None:
        self._loops = loops
        self.run_id = loops.run_id
        self.closed = False

    def forward_backward(self, data: list[Any], loss_fn: str = "cross_entropy") -> FakeFuture:
        return FakeFuture(self._loops._step("forward_backward", data, loss_fn))

    def forward(self, data: list[Any], loss_fn: str = "cross_entropy") -> FakeFuture:
        return FakeFuture(self._loops._step("forward", data, loss_fn))

    def optim_step(self, adam_params: FakeAdamParams) -> FakeFuture:
        self._loops._record("optim_step", {"adam_params": adam_params})
        return FakeFuture(FakeStepOutput())

    def save_state(self, name: str) -> FakeFuture:
        self._loops._record("save_state", {"name": name})
        return FakeFuture(FakeSaved(f"bt://loops:{self.run_id}/state/{name}"))

    def save_weights_for_sampler(self, name: str) -> FakeFuture:
        self._loops._record("save_weights_for_sampler", {"name": name})
        return FakeFuture(FakeSaved(f"bt://loops:{self.run_id}/sampler_weights/{name}"))

    def close(self) -> None:
        self._loops._record("close", {})
        self.closed = True


class FakeServiceClient:
    def __init__(self, loops: FakeLoops) -> None:
        self._loops = loops
        self.session_id = loops.session_id

    def create_lora_training_client(self, **kwargs: Any) -> FakeTrainingClient:
        self._loops._record("create_lora_training_client", kwargs)
        self._loops.trainer = FakeTrainingClient(self._loops)
        return self._loops.trainer


class FakeSamplingClient:
    def __init__(self, loops: FakeLoops) -> None:
        self._loops = loops

    def sample(self, **kwargs: Any) -> Any:
        self._loops._record("sample", kwargs)
        return {"sequences": [{"tokens": [1]}]}

    def close(self) -> None:
        self._loops._record("sampler_close", {})


class FakeLoops:
    """A fake ``baseten.loops`` module; see the module docstring."""

    Datum = FakeDatum
    ModelInput = FakeModelInput
    TensorData = FakeTensorData
    AdamParams = FakeAdamParams
    SamplingParams = FakeSamplingParams
    BasetenDeployment = FakeBasetenDeployment
    ServerShutdownError = FakeServerShutdownError

    def __init__(
        self,
        *,
        session_id: str = "session1",
        run_id: Any = "run1",
        train_losses: Sequence[float] = (1.0,),
        validation_losses: Sequence[float] = (1.0,),
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        self.session_id = session_id
        self.run_id = run_id
        self.train_losses = list(train_losses)
        self.validation_losses = list(validation_losses)
        self.metrics = dict(metrics or {})
        self.calls: list[LoopsCall] = []
        self.trainer: FakeTrainingClient | None = None
        self._hooks: dict[tuple[str, int], Callable[[], None]] = {}

    def ServiceClient(self, **kwargs: Any) -> FakeServiceClient:
        self._record("ServiceClient", kwargs)
        return FakeServiceClient(self)

    def SamplingClient(self, **kwargs: Any) -> FakeSamplingClient:
        self._record("SamplingClient", kwargs)
        return FakeSamplingClient(self)

    def fail_on(self, method: str, *, error: BaseException, call: int = 1) -> None:
        """Make the ``call``-th call (1-based) of ``method`` raise ``error``."""

        def fail() -> None:
            raise error

        self._hooks[(method, call)] = fail

    def on_call(self, method: str, action: Callable[[], None], *, call: int = 1) -> None:
        """Run ``action`` during the ``call``-th call of ``method``, after recording it."""
        self._hooks[(method, call)] = action

    @property
    def methods(self) -> list[str]:
        return [call.method for call in self.calls]

    def kwargs(self, method: str) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.calls if call.method == method]

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append(LoopsCall(method, kwargs))
        hook = self._hooks.get((method, len(self.kwargs(method))))
        if hook is not None:
            hook()

    def _step(self, method: str, data: list[Any], loss_fn: str) -> FakeStepOutput:
        self._record(method, {"data": data, "loss_fn": loss_fn})
        losses = self.train_losses if method == "forward_backward" else self.validation_losses
        loss = losses[(len(self.kwargs(method)) - 1) % len(losses)]
        return FakeStepOutput(loss=loss, metrics=dict(self.metrics))


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

    def __init__(self, routes: Mapping[str, Answer | list[Answer]]) -> None:
        self.routes = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in routes.items()
        }
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
        if not answers:
            raise AssertionError(f"no fake Baseten route for {key}")
        return _respond(answers.pop(0) if len(answers) > 1 else answers[0], request)


def baseten_transport(routes: Mapping[str, Answer | list[Answer]]) -> BasetenTransport:
    return BasetenTransport(routes)


def graphql(answers: Mapping[str, Answer]) -> Callable[[httpx.Request], httpx.Response]:
    """Answer a GraphQL POST by the first key found in its query, e.g. ``"listed_instances"``."""

    def handle(request: httpx.Request) -> httpx.Response:
        query = jsonlib.loads(request.content)["query"]
        for needle, answer in answers.items():
            if needle in query:
                return _respond(answer, request)
        raise AssertionError(f"no fake GraphQL answer for {query!r}")

    return handle
