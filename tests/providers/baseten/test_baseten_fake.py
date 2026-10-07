# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Loops fake behaves like the SDK: sessions, futures, checkpoints, close and deactivation."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune.errors import ExternalError
from pup_finetune.providers.baseten.client import (
    MAX_CHECKPOINT_NAME_LEN,
    BasetenHTTP,
    loops_service,
    to_loops_datum,
)
from pup_finetune.rendering.tokens import TokenizedExample
from tests.fakes.baseten import (
    FakeAdamParams,
    FakeDatum,
    FakeForwardBackwardOutput,
    FakeLoadWeightsResponse,
    FakeLoops,
    FakeModelInput,
    FakeOptimStepResponse,
    FakeRemoteOpError,
    FakeSaveWeightsResponse,
    FakeServerShutdownError,
    FakeTensorData,
    FakeTrainingClient,
    http_status_error,
    loss_at,
)

KEY = SecretStr("synthetic-baseten-key")
EXAMPLE = TokenizedExample((5, 6, 7), (6, 7, 8), (0.0, 1.0, 1.0))
TRAINER_SETTINGS: dict[str, Any] = {
    "base_model": "Qwen/Qwen3.5-9B",
    "rank": 8,
    "replicas": 1,
    "seed": 7,
    "max_seq_len": 1024,
    "with_sampler": False,
    "name": "pft-test",
}


def trainer(loops: FakeLoops) -> FakeTrainingClient:
    service: Any = loops_service(KEY, loops=loops)
    created: FakeTrainingClient = service.create_lora_training_client(**TRAINER_SETTINGS)
    return created


def datums(loops: FakeLoops) -> list[Any]:
    return [to_loops_datum(EXAMPLE, loops)]


def test_every_client_is_a_new_session_and_every_trainer_a_new_run() -> None:
    loops = FakeLoops()
    first, second = trainer(loops), trainer(loops)
    assert loops.sessions == ["session1", "session2"]
    assert (first.run_id, second.run_id) == ("run1", "run2")
    assert loops.runs["run2"].session_id == "session2"
    assert loops.runs["run1"].settings == TRAINER_SETTINGS


def test_steps_report_the_loss_and_the_logprobs_of_weighted_targets() -> None:
    loops = FakeLoops()
    out = trainer(loops).forward_backward(datums(loops), "cross_entropy").result()
    assert out.loss == out.metrics["loss"] == loss_at(0)
    (output,) = out.loss_fn_outputs
    assert output["logprobs"] == FakeTensorData([None, -loss_at(0), -loss_at(0)], "float32", [3])


def test_the_loss_falls_only_with_an_optimizer_step_that_applies_gradients() -> None:
    loops = FakeLoops()
    client = trainer(loops)
    adam = loops.AdamParams(learning_rate=1e-4)
    client.optim_step(adam).result()
    assert client.forward(datums(loops)).result().loss == loss_at(0)
    client.forward_backward(datums(loops)).result()
    assert client.optim_step(adam).result().metrics == {"step": 1.0}
    first = client.forward(datums(loops)).result()
    assert first.loss == loss_at(1)
    client.optim_step(adam).result()
    assert client.forward(datums(loops)).result() == first


def test_without_logprobs_only_the_loss_is_reported() -> None:
    loops = FakeLoops(logprobs=False)
    out = trainer(loops).forward(datums(loops)).result()
    assert (out.loss, out.loss_fn_outputs) == (loss_at(0), [])


def test_a_datum_whose_arrays_differ_in_length_is_refused() -> None:
    loops = FakeLoops()
    datum = to_loops_datum(EXAMPLE, loops)
    datum.model_input = FakeModelInput([5, 6])
    with pytest.raises(httpx.HTTPStatusError, match="HTTP 400"):
        trainer(loops).forward([datum])


def test_checkpoints_are_saved_under_the_run() -> None:
    loops = FakeLoops()
    client = trainer(loops)
    assert client.save_state("t1").result().path == "bt://loops:run1/state/t1"
    sampler = client.save_weights_for_sampler("s1").result().path
    assert sampler == "bt://loops:run1/sampler_weights/s1"
    assert loops.checkpoints[sampler].kind == "sampler_weights"


@pytest.mark.parametrize("name", ["", "x" * (MAX_CHECKPOINT_NAME_LEN + 1)])
def test_a_checkpoint_name_must_fit_the_limit(name: str) -> None:
    loops = FakeLoops()
    client = trainer(loops)
    for save in (client.save_state, client.save_weights_for_sampler):
        with pytest.raises(httpx.HTTPStatusError, match="HTTP 422"):
            save(name)
    assert client.save_state("x" * MAX_CHECKPOINT_NAME_LEN).result().path


def test_loading_a_training_checkpoint_restores_its_weights_in_any_run() -> None:
    loops = FakeLoops()
    first = trainer(loops)
    first.forward_backward(datums(loops)).result()
    first.optim_step(loops.AdamParams()).result()
    path = first.save_state("t1").result().path
    second = trainer(loops)
    assert second.load_state_with_optimizer(path).result().step == 1
    assert second.forward(datums(loops)).result().loss == loss_at(1)
    assert second.load_state(path).result().step == 1


@pytest.mark.parametrize("kind", ["sampler", "unknown", "expired"])
def test_only_a_live_training_checkpoint_loads(kind: str) -> None:
    loops = FakeLoops()
    client = trainer(loops)
    path = {
        "sampler": client.save_weights_for_sampler("s1").result().path,
        "unknown": "bt://loops:run1/state/missing",
        "expired": client.save_state("t1").result().path,
    }[kind]
    if kind == "expired":
        loops.expire(path)
    for load in (client.load_state, client.load_state_with_optimizer):
        with pytest.raises(FakeRemoteOpError, match="no training checkpoint"):
            load(path).result()


def test_a_closed_client_refuses_calls_but_its_run_stays_active() -> None:
    loops = FakeLoops()
    client = trainer(loops)
    client.close()
    with pytest.raises(RuntimeError, match="client has been closed"):
        client.forward(datums(loops))
    http = BasetenHTTP(KEY, transport=loops.transport())
    run = http.get_run("run1")
    assert run is not None
    assert run["status"] == {"name": "ACTIVE"}


def test_a_deactivated_run_is_inactive_and_refuses_training() -> None:
    loops = FakeLoops()
    client = trainer(loops)
    transport = loops.transport()
    http = BasetenHTTP(KEY, transport=transport)
    http.deactivate_run("run1")
    run = http.get_run("run1")
    assert run is not None
    assert run["status"] == {"name": "INACTIVE"}
    with pytest.raises(httpx.HTTPStatusError, match="HTTP 404"):
        client.forward_backward(datums(loops))
    assert http.get_run("run9") is None
    with pytest.raises(ExternalError, match="returned HTTP 404"):
        http.deactivate_run("run9")
    assert transport.calls[0] == "POST /v1/loops/runs/run1/deactivate"


def test_explicit_routes_come_before_the_fake_ones() -> None:
    loops = FakeLoops()
    trainer(loops)
    deactivate = "POST /v1/loops/runs/run1/deactivate"
    http = BasetenHTTP(KEY, transport=loops.transport({deactivate: [(500, {}), (200, {})]}))
    with pytest.raises(ExternalError, match="HTTP 500"):
        http.deactivate_run("run1")
    http.deactivate_run("run1")
    assert loops.runs["run1"].active


def test_failures_are_injected_at_the_call_or_in_its_future() -> None:
    loops = FakeLoops()
    client = trainer(loops)
    loops.fail_on("forward", error=http_status_error(503))
    loops.fail_on("forward", error=FakeServerShutdownError("going away"), call=2, on_result=True)
    with pytest.raises(httpx.HTTPStatusError):
        client.forward(datums(loops))
    pending = client.forward(datums(loops))
    with pytest.raises(FakeServerShutdownError):
        pending.result()
    assert client.forward(datums(loops)).result().loss == loss_at(0)
    assert loops.methods.count("forward") == 3


def test_a_failed_trainer_creation_creates_no_run() -> None:
    loops = FakeLoops()
    loops.fail_on("create_lora_training_client", error=http_status_error(429))
    with pytest.raises(httpx.HTTPStatusError):
        trainer(loops)
    assert loops.runs == {}


# --- the real SDK ------------------------------------------------------------------------


def _fields(cls: Any) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_the_fake_types_have_the_real_fields() -> None:
    sdk = pytest.importorskip("baseten.loops", reason="needs the baseten extra")
    pairs = [
        (FakeDatum, sdk.Datum),
        (FakeTensorData, sdk.TensorData),
        (FakeAdamParams, sdk.AdamParams),
        (FakeForwardBackwardOutput, sdk.ForwardBackwardOutput),
        (FakeOptimStepResponse, sdk.OptimStepResponse),
        (FakeSaveWeightsResponse, sdk.SaveWeightsResponse),
        (FakeLoadWeightsResponse, sdk.LoadWeightsResponse),
    ]
    for fake, real in pairs:
        assert _fields(fake) <= set(real.model_fields), fake.__name__
    defaults = sdk.AdamParams()
    assert dataclasses.asdict(FakeAdamParams()) == defaults.model_dump()
    assert {"from_ints", "to_ints", "length"} <= set(dir(sdk.ModelInput))


def test_every_call_the_fake_takes_binds_to_the_real_sdk() -> None:
    sdk = pytest.importorskip("baseten.loops", reason="needs the baseten extra")
    loops = FakeLoops()
    client = trainer(loops)
    client.forward_backward(datums(loops), "cross_entropy").result()
    client.forward(datums(loops), "cross_entropy").result()
    client.optim_step(loops.AdamParams()).result()
    path = client.save_state("t1").result().path
    client.save_weights_for_sampler("s1").result()
    client.load_state(path).result()
    client.load_state_with_optimizer(path).result()
    client.close()
    methods = {
        "ServiceClient": sdk.ServiceClient.__init__,
        "create_lora_training_client": sdk.ServiceClient.create_lora_training_client,
    }
    for call in loops.calls:
        real = methods.get(call.method) or getattr(sdk.TrainingClient, call.method)
        inspect.signature(real).bind(None, **call.kwargs)
