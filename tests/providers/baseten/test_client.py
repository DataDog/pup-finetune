# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.providers.baseten import client
from pup_finetune.providers.baseten.client import (
    API_BASE,
    BasetenHTTP,
    DeploymentRejected,
    LoopsTrainer,
    load_sampler_checkpoint,
    open_trainer,
)
from pup_finetune.providers.baseten.datums import IGNORE_INDEX
from pup_finetune.providers.baseten.registration import LOOPS_INSTALL_HINT
from pup_finetune.rendering import RenderError, TokenDatum
from tests.fakes.baseten import (
    BasetenTransport,
    FakeBasetenDeployment,
    FakeFuture,
    FakeLoops,
    FakeModelInput,
    FakeSaved,
    FakeServerShutdownError,
    FakeTensorData,
    baseten_transport,
    graphql,
    http_status_error,
)

SECRET = "zq9Xsynthetic0000basetenKEYw7Pm"  # noqa: S105 - synthetic test value
KEY = SecretStr(SECRET)
GUARDED_ENV = (
    "LOOPS_REQUEST_LOG",
    "LOOPS_REUSE_FROM_RUN_ID",
    "LOOPS_REUSE_FROM_SESSION_ID",
    "INTERNAL_LOOPS_BACKEND_VERSION",
    "LOOPS_TEAM",
    "LOOPS_BASE_URL",
)
SAMPLER_URL = "https://model-abc123.api.baseten.co/deployment/dep456/sync"
OPENAI_URL = f"{SAMPLER_URL}/v1"
DATUM = TokenDatum(tokens=(5, 6, 7, 8), weights=(0, 0, 1, 1), message_index=1)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    redaction.clear_secrets()
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", sleeps.append)
    yield sleeps
    redaction.clear_secrets()


@pytest.fixture
def sleeps(_isolated: list[float]) -> list[float]:
    return _isolated


@pytest.fixture
def no_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import baseten.loops`` fail, whatever is installed."""
    monkeypatch.setitem(sys.modules, "baseten", None)
    monkeypatch.setitem(sys.modules, "baseten.loops", None)


def make_http(routes: dict[str, Any]) -> tuple[BasetenHTTP, BasetenTransport]:
    transport = baseten_transport(routes)
    return BasetenHTTP(KEY, transport=transport), transport


def assert_clean(error: FinetuneError) -> None:
    for text in (str(error), error.message, error.hint or ""):
        assert SECRET not in text


def make_trainer(loops: FakeLoops, **overrides: Any) -> LoopsTrainer:
    settings: dict[str, Any] = {
        "base_model": "Qwen/Qwen3.5-9B",
        "rank": 8,
        "seed": 42,
        "max_seq_len": 64,
        "run_id": "sft-test",
        "loops": loops,
    }
    return open_trainer(KEY, **(settings | overrides))


# --- BasetenHTTP transport -----------------------------------------------------------


def test_requests_send_the_api_key_and_return_the_json() -> None:
    http, transport = make_http({"GET /v1/loops/capabilities": (200, {"supported_models": []})})
    assert http.capabilities() == {"supported_models": []}
    (request,) = transport.requests
    assert str(request.url) == f"{API_BASE}/v1/loops/capabilities"
    assert request.headers["Authorization"] == f"Api-Key {SECRET}"


def test_an_error_status_is_a_fixed_message_without_the_body() -> None:
    body = {"detail": f"internal trace for {SECRET}"}
    http, _ = make_http({"GET /v1/loops/capabilities": (500, body)})
    with pytest.raises(ExternalError) as caught:
        http.capabilities()
    assert str(caught.value) == "baseten: the Loops capabilities request returned HTTP 500"
    assert "internal trace" not in str(caught.value)
    assert_clean(caught.value)


def test_redirects_are_not_followed() -> None:
    redirect = httpx.Response(302, headers={"Location": "https://elsewhere.example/steal"})
    http, transport = make_http({"GET /v1/loops/capabilities": redirect})
    with pytest.raises(ExternalError, match="returned HTTP 302"):
        http.capabilities()
    assert transport.calls == ["GET /v1/loops/capabilities"]


def test_responses_over_4_mib_are_refused() -> None:
    huge = httpx.Response(200, content=b'"' + b"x" * (4 * 1024 * 1024) + b'"')
    http, _ = make_http({"GET /v1/loops/capabilities": huge})
    with pytest.raises(ExternalError, match="returned more than 4 MiB"):
        http.capabilities()


@pytest.mark.parametrize("content", [b"{not json", b'{"limit": NaN}', b"\xff\xfe"])
def test_malformed_json_is_an_external_error(content: bytes) -> None:
    http, _ = make_http({"GET /v1/loops/capabilities": httpx.Response(200, content=content)})
    with pytest.raises(ExternalError, match="returned malformed JSON"):
        http.capabilities()


def test_an_unexpected_shape_is_an_external_error() -> None:
    http, _ = make_http({"GET /v1/loops/capabilities": (200, ["not", "an", "object"])})
    with pytest.raises(ExternalError, match="returned an unexpected response"):
        http.capabilities()


def unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def test_network_errors_say_whether_a_write_may_have_landed() -> None:
    http, _ = make_http(
        {"GET /v1/loops/capabilities": unreachable, "POST /v1/loops/sessions": unreachable}
    )
    with pytest.raises(ExternalError, match=r"could not reach Baseten .* \(ConnectError\)") as read:
        http.capabilities()
    assert read.value.hint == "check your network connection and retry"
    with pytest.raises(ExternalError) as write:
        http.create_session()
    assert write.value.hint is not None
    assert "may have reached Baseten" in write.value.hint


def test_http_is_a_context_manager() -> None:
    http, _ = make_http({})
    with http as same:
        assert same is http


# --- REST routes ---------------------------------------------------------------------


def test_get_checkpoint_sends_the_path_and_returns_the_one_record() -> None:
    path = "bt://loops:run1/sampler_weights/ckpt"
    record = {"id": "ck1", "run_id": "run1", "checkpoint_id": "ckpt"}
    http, transport = make_http({"GET /v1/loops/checkpoints": (200, {"checkpoints": [record]})})
    assert http.get_checkpoint(path) == record
    assert transport.requests[0].url.params["checkpoint_path"] == path


@pytest.mark.parametrize("records", [[], [{"id": "a"}, {"id": "b"}], None])
def test_get_checkpoint_needs_exactly_one_record(records: Any) -> None:
    http, _ = make_http({"GET /v1/loops/checkpoints": (200, {"checkpoints": records})})
    with pytest.raises(ExternalError, match="exactly one checkpoint"):
        http.get_checkpoint("bt://loops:run1/sampler_weights/ckpt")


def test_get_run_returns_the_run_or_none_when_gone() -> None:
    run = {"id": "run1", "status": {"name": "ACTIVE"}}
    http, _ = make_http(
        {
            "GET /v1/loops/runs/run1": (200, {"run": run}),
            "GET /v1/loops/runs/gone": (404, {"detail": "not found"}),
        }
    )
    assert http.get_run("run1") == run
    assert http.get_run("gone") is None


def test_deactivate_run_posts_an_empty_object() -> None:
    http, transport = make_http({"POST /v1/loops/runs/run1/deactivate": (200, {"id": "run1"})})
    http.deactivate_run("run1")
    assert transport.bodies("POST /v1/loops/runs/run1/deactivate") == [{}]


def test_create_session_returns_its_id() -> None:
    http, transport = make_http({"POST /v1/loops/sessions": (200, {"session": {"id": "s1"}})})
    assert http.create_session() == "s1"
    assert transport.requests[0].content == b""


def test_an_invalid_id_from_baseten_is_refused() -> None:
    http, _ = make_http({"POST /v1/loops/sessions": (200, {"session": {"id": "../s1"}})})
    with pytest.raises(ExternalError, match="invalid Baseten session id"):
        http.create_session()


def test_create_sampler_asks_for_dedicated_capacity() -> None:
    sampler = {"id": "smp1", "model_id": "m1", "deployment_id": "d1", "base_url": SAMPLER_URL}
    http, transport = make_http({"POST /v1/loops/samplers": (200, {"sampler": sampler})})
    path = "bt://loops:run1/sampler_weights/ckpt"
    assert http.create_sampler(session_id="s1", model_path=path, max_seq_len=4096) == sampler
    assert transport.bodies("POST /v1/loops/samplers") == [
        {
            "session_id": "s1",
            "model_path": path,
            "max_seq_length": 4096,
            "availability_model": "dedicated",
        }
    ]


@pytest.mark.parametrize(("status", "rejected"), [(429, True), (422, True), (503, False)])
def test_a_4xx_to_a_sampler_create_means_nothing_was_created(status: int, rejected: bool) -> None:
    http, _ = make_http({"POST /v1/loops/samplers": (status, {})})
    with pytest.raises(ExternalError, match=f"returned HTTP {status}") as caught:
        http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        )
    assert isinstance(caught.value, DeploymentRejected) is rejected


def test_deployment_routes() -> None:
    base = "/v1/models/m1/deployments/d1"
    http, transport = make_http(
        {
            f"GET {base}": (200, {"id": "d1", "model_id": "m1", "status": "ACTIVE"}),
            f"POST {base}/activate": (200, {"success": True}),
            f"POST {base}/deactivate": (200, {"success": True}),
            "GET /v1/models/m1/deployments/gone": (404, {}),
        }
    )
    assert http.get_deployment("m1", "d1") == {"id": "d1", "model_id": "m1", "status": "ACTIVE"}
    assert http.get_deployment("m1", "gone") is None
    http.activate_deployment("m1", "d1")
    http.deactivate_deployment("m1", "d1")
    assert transport.calls[2:] == [f"POST {base}/activate", f"POST {base}/deactivate"]


@pytest.mark.parametrize("bad_id", ["../runs", "a/b", "", "a b"])
def test_ids_in_paths_are_validated_before_any_request(bad_id: str) -> None:
    http, transport = make_http({})
    with pytest.raises(ExternalError, match="invalid Baseten run id"):
        http.get_run(bad_id)
    assert transport.requests == []


# --- GraphQL -------------------------------------------------------------------------

INSTANCE = {
    "id": "H100x1",
    "name": "H100",
    "gpuType": "H100",
    "gpuCount": 1,
    "nodeCount": 1,
}


def test_instance_types_posts_the_truss_query_to_the_app_graphql_endpoint() -> None:
    answer = (200, {"data": {"listedInstances": [INSTANCE]}})
    http, transport = make_http({"POST /graphql/": graphql({"listed_instances": answer})})
    assert http.instance_types() == [INSTANCE]
    (request,) = transport.requests
    assert str(request.url) == "https://app.baseten.co/graphql/"
    assert request.headers["Authorization"] == f"Api-Key {SECRET}"
    body = json.loads(request.content)
    assert body == {"query": client._LISTED_INSTANCES}
    assert "listedInstances: listed_instances" in body["query"]
    assert "nodeCount: node_count" in body["query"]


def test_instance_types_refuses_graphql_errors() -> None:
    answer = (200, {"errors": [{"message": "nope"}]})
    http, _ = make_http({"POST /graphql/": graphql({"listed_instances": answer})})
    with pytest.raises(ExternalError, match="returned GraphQL errors"):
        http.instance_types()


CREATED = {
    "model_version": {"id": "dep1", "name": "pup-finetune-x", "model_id": "mod1"},
    "truss_config": "{}",
}


def test_create_checkpoint_deployment_sends_the_truss_request() -> None:
    answer = (200, {"data": {"create_model_version_from_inference_template": CREATED}})
    http, transport = make_http(
        {"POST /graphql/": graphql({"create_model_version_from_inference_template": answer})}
    )
    created = http.create_checkpoint_deployment(
        checkpoint_id="ck1", model_name="pup-finetune-x", instance_type_id="H100x1"
    )
    assert created == CREATED
    body = json.loads(transport.requests[0].content)
    assert body["query"] == client._CREATE_FROM_TEMPLATE
    assert body["variables"] == {
        "request": {
            "metadata": {"oracle_name": "pup-finetune-x"},
            "weights_sources": [
                {
                    "weight_source_type": "B10_LOOPS_CHECKPOINTING",
                    "b10_loops_checkpoint_weights_source": {
                        "checkpoint": {"loops_checkpoint_id": "ck1"}
                    },
                }
            ],
            "inference_stack": {"stack_type": "VLLM", "environment_variables": []},
            "instance_type_id": "H100x1",
            "dry_run": False,
        }
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            {"message": f"bdn is set, but BDN volumes are not enabled ({SECRET})"},
            "bdn is set, but BDN volumes are not enabled (***)",
        ),
        (
            {"message": "generic", "extensions": {"description": "a clearer description"}},
            "a clearer description",
        ),
        ({}, "no reason given"),
    ],
)
def test_graphql_errors_reject_the_deployment(error: dict[str, Any], expected: str) -> None:
    answer = (200, {"errors": [error], "data": None})
    http, _ = make_http(
        {"POST /graphql/": graphql({"create_model_version_from_inference_template": answer})}
    )
    with pytest.raises(DeploymentRejected) as caught:
        http.create_checkpoint_deployment(
            checkpoint_id="ck1", model_name="pup-finetune-x", instance_type_id="H100x1"
        )
    assert caught.value.message == f"Baseten refused to create the deployment: {expected}"
    assert_clean(caught.value)


def test_graphql_errors_next_to_a_created_model_return_the_model() -> None:
    answer = (
        200,
        {
            "errors": [{"message": "truss_config failed"}],
            "data": {"create_model_version_from_inference_template": CREATED},
        },
    )
    http, _ = make_http(
        {"POST /graphql/": graphql({"create_model_version_from_inference_template": answer})}
    )
    created = http.create_checkpoint_deployment(
        checkpoint_id="ck1", model_name="pup-finetune-x", instance_type_id="H100x1"
    )
    assert created == CREATED


@pytest.mark.parametrize(("status", "rejected"), [(400, True), (500, False)])
def test_an_http_error_to_the_graphql_create(status: int, rejected: bool) -> None:
    http, _ = make_http({"POST /graphql/": (status, {})})
    with pytest.raises(ExternalError) as caught:
        http.create_checkpoint_deployment(
            checkpoint_id="ck1", model_name="pup-finetune-x", instance_type_id="H100x1"
        )
    assert isinstance(caught.value, DeploymentRejected) is rejected


def test_a_create_without_a_deployment_has_an_unknown_outcome() -> None:
    answer = (200, {"data": {"create_model_version_from_inference_template": None}})
    http, _ = make_http(
        {"POST /graphql/": graphql({"create_model_version_from_inference_template": answer})}
    )
    with pytest.raises(ExternalError, match="outcome is unknown") as caught:
        http.create_checkpoint_deployment(
            checkpoint_id="ck1", model_name="pup-finetune-x", instance_type_id="H100x1"
        )
    assert not isinstance(caught.value, DeploymentRejected)
    assert caught.value.hint == "look for a model named pup-finetune-x in the Baseten UI"


# --- OpenAI-compatible routes --------------------------------------------------------


def test_openai_routes_of_a_deployment() -> None:
    path = "/deployment/dep456/sync/v1"
    reply = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    http, transport = make_http(
        {
            f"GET {path}/models": (200, {"data": [{"id": "ckpt", "max_model_len": 8192}]}),
            f"POST {path}/chat/completions": (200, reply),
        }
    )
    assert http.list_served_models(OPENAI_URL + "/") == [{"id": "ckpt", "max_model_len": 8192}]
    body = {"model": "ckpt", "messages": [{"role": "user", "content": "hi"}]}
    assert http.chat_completion(OPENAI_URL, body) == reply
    assert transport.requests[1].url.host == "model-abc123.api.baseten.co"
    assert transport.requests[0].extensions["timeout"]["read"] == 300.0
    assert transport.bodies(f"POST {path}/chat/completions") == [body]


@pytest.mark.parametrize(
    "url",
    [
        "http://model-abc.api.baseten.co/deployment/d/sync/v1",
        "https://model-abc.api.baseten.co.elsewhere.example/v1",
        "https://elsewhere.example/v1",
        "https://user:pass@model-abc.api.baseten.co/v1",
        "https://model-abc.api.baseten.co:8443/v1",
        "https://model-abc.api.baseten.co/v1?redirect=1",
        "https://model-abc.api.baseten.co:bad/v1",
    ],
)
def test_the_key_is_only_sent_to_baseten_inference_urls(url: str) -> None:
    http, transport = make_http({})
    with pytest.raises(ExternalError, match="non-Baseten URL"):
        http.list_served_models(url)
    assert transport.requests == []


# --- the Loops SDK: environment and import -------------------------------------------


@pytest.mark.usefixtures("no_sdk")
@pytest.mark.parametrize("name", GUARDED_ENV)
def test_risky_sdk_environment_is_refused_before_the_import(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "https://elsewhere.example" if name == "LOOPS_BASE_URL" else "x")
    with pytest.raises(UsageError, match=name) as caught:
        open_trainer(
            KEY, base_model="Qwen/Qwen3.5-9B", rank=8, seed=1, max_seq_len=64, run_id="sft-test"
        )
    assert caught.value.hint == f"unset {name} and retry"


def test_the_guard_also_runs_with_an_injected_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPS_REQUEST_LOG", "/tmp/requests.jsonl")  # noqa: S108 - never written
    loops = FakeLoops()
    with pytest.raises(UsageError, match="LOOPS_REQUEST_LOG"):
        make_trainer(loops)
    with pytest.raises(UsageError, match="LOOPS_REQUEST_LOG"):
        load_sampler_checkpoint(
            KEY,
            base_url=SAMPLER_URL,
            model_id="abc123",
            deployment_id="dep456",
            sampler_id="smp1",
            session_id="s1",
            checkpoint_path="bt://loops:run1/sampler_weights/ckpt",
            base_model="Qwen/Qwen3.5-9B",
            prompt=[1, 2],
            ready_timeout=60.0,
            loops=loops,
        )
    assert loops.calls == []


@pytest.mark.parametrize("value", ["https://api.baseten.co", "https://api.baseten.co/", ""])
def test_the_default_loops_base_url_is_allowed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LOOPS_BASE_URL", value)
    assert make_trainer(FakeLoops()).run_id == "run1"


@pytest.mark.usefixtures("no_sdk")
def test_a_missing_sdk_is_a_usage_error_with_the_install_hint() -> None:
    with pytest.raises(UsageError, match="the Baseten Loops SDK is not installed") as caught:
        open_trainer(
            KEY, base_model="Qwen/Qwen3.5-9B", rank=8, seed=1, max_seq_len=64, run_id="sft-test"
        )
    assert caught.value.hint == LOOPS_INSTALL_HINT
    assert caught.value.exit_code == 2


# --- open_trainer --------------------------------------------------------------------


def test_open_trainer_passes_the_key_and_settings_explicitly() -> None:
    loops = FakeLoops(session_id="sess9", run_id="run9")
    trainer = make_trainer(loops, rank=16, seed=7, max_seq_len=4096, run_id="sft-abc")
    assert (trainer.session_id, trainer.run_id) == ("sess9", "run9")
    assert loops.kwargs("ServiceClient") == [
        {"api_key": SECRET, "base_url": API_BASE, "availability_model": "dedicated"}
    ]
    assert loops.kwargs("create_lora_training_client") == [
        {
            "base_model": "Qwen/Qwen3.5-9B",
            "rank": 16,
            "replicas": 1,
            "seed": 7,
            "max_seq_len": 4096,
            "with_sampler": False,
            "name": "sft-abc",
        }
    ]
    assert "BASETEN_API_KEY" not in os.environ


ORPHAN = ["a run may remain in Loops session session1; deactivate it in the Baseten UI"]


@pytest.mark.parametrize(
    ("error", "message", "hint", "notes"),
    [
        (
            http_status_error(429, json={"detail": "no H100s"}, headers={"Retry-After": "30"}),
            "Baseten has no capacity for the trainer (HTTP 429): no H100s",
            "retry after 30 s",
            [],
        ),
        (
            http_status_error(429),
            "Baseten has no capacity for the trainer (HTTP 429)",
            "retry later",
            [],
        ),
        (
            http_status_error(400, json={"message": f"bad rank for key {SECRET}"}),
            "Baseten refused to create the trainer (HTTP 400): bad rank for key ***",
            None,
            [],
        ),
        (
            RuntimeError("Trainer deployment d1 entered terminal-failure state 'FAILED'"),
            "Baseten trainer provisioning failed: Trainer deployment d1 entered "
            "terminal-failure state 'FAILED'",
            None,
            ORPHAN,
        ),
        (
            TimeoutError("Trainer did not become ready in 3600s."),
            "creating the Baseten trainer failed (TimeoutError)",
            None,
            ORPHAN,
        ),
    ],
    ids=["capacity-retry-after", "capacity", "refused", "terminal-state", "timeout"],
)
def test_trainer_creation_errors_are_explained(
    error: Exception, message: str, hint: str | None, notes: list[str]
) -> None:
    loops = FakeLoops()
    sessions: list[str] = []
    loops.fail_on("create_lora_training_client", error=error)
    with pytest.raises(ExternalError) as caught:
        make_trainer(loops, on_session=sessions.append)
    assert (caught.value.message, caught.value.hint) == (message, hint)
    assert getattr(caught.value, "__notes__", []) == notes
    assert sessions == ["session1"]
    assert caught.value.__cause__ is error
    assert_clean(caught.value)


def test_an_interrupt_while_provisioning_names_the_session() -> None:
    loops = FakeLoops()
    loops.fail_on("create_lora_training_client", error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as caught:
        make_trainer(loops)
    assert caught.value.__notes__ == ORPHAN


def test_session_creation_errors_are_explained() -> None:
    loops = FakeLoops()
    loops.fail_on("ServiceClient", error=http_status_error(401, json={"detail": "bad key"}))
    with pytest.raises(ExternalError) as caught:
        make_trainer(loops)
    assert caught.value.message == "Baseten refused to create the Loops session (HTTP 401): bad key"
    assert "create_lora_training_client" not in loops.methods


def test_a_trainer_without_a_run_id_is_closed_and_refused() -> None:
    loops = FakeLoops(run_id=None)
    with pytest.raises(ExternalError, match="no valid run id") as caught:
        make_trainer(loops)
    assert loops.methods[-1] == "close"
    assert caught.value.hint == "deactivate the run of Loops session session1 in the Baseten UI"


# --- LoopsTrainer --------------------------------------------------------------------


def test_datums_are_shifted_into_loops_datums() -> None:
    loops = FakeLoops(train_losses=[0.25])
    result = make_trainer(loops).forward_backward([DATUM])
    (call,) = loops.kwargs("forward_backward")
    assert call["loss_fn"] == "cross_entropy"
    (sent,) = call["data"]
    assert sent.model_input == FakeModelInput([5, 6, 7])
    assert sent.loss_fn_inputs == {
        "target_tokens": FakeTensorData(data=[IGNORE_INDEX, 7, 8], dtype="int64", shape=[3]),
        "weights": FakeTensorData(data=[0.0, 1.0, 1.0], dtype="float32", shape=[3]),
    }
    assert (result.loss, result.active_tokens) == (0.25, 2.0)


def test_step_metrics_take_precedence_over_the_output() -> None:
    loops = FakeLoops(train_losses=[9.0], metrics={"loss": 0.5, "active_tokens": 7})
    result = make_trainer(loops).forward_backward([DATUM, DATUM])
    assert (result.loss, result.active_tokens) == (0.5, 7.0)


def test_a_malformed_step_result_is_an_external_error() -> None:
    loops = FakeLoops(metrics={"loss": "not-a-number"})  # type: ignore[dict-item]
    with pytest.raises(ExternalError, match="malformed step result"):
        make_trainer(loops).forward([DATUM])


def test_an_invalid_datum_is_refused_before_any_call() -> None:
    loops = FakeLoops()
    too_long = TokenDatum(tokens=tuple(range(100)), weights=(1,) * 100, message_index=1)
    trainer = make_trainer(loops, max_seq_len=64)
    with pytest.raises(RenderError, match="exceeds_context"):
        trainer.forward_backward([too_long])
    assert "forward_backward" not in loops.methods


def test_forward_retries_transient_failures(sleeps: list[float]) -> None:
    loops = FakeLoops(validation_losses=[0.75])
    loops.fail_on("forward", error=FakeServerShutdownError())
    loops.fail_on("forward", error=http_status_error(503), call=2)
    assert make_trainer(loops).forward([DATUM]).loss == 0.75
    assert len(loops.kwargs("forward")) == 3
    assert sleeps == [1.0, 2.0]


def test_forward_gives_up_after_three_attempts(sleeps: list[float]) -> None:
    loops = FakeLoops()
    for call in (1, 2, 3):
        loops.fail_on("forward", error=httpx.ReadTimeout("slow"), call=call)
    with pytest.raises(ExternalError, match=r"the Loops forward call failed \(ReadTimeout\)"):
        make_trainer(loops).forward([DATUM])
    assert sleeps == [1.0, 2.0]


def test_forward_does_not_retry_permanent_failures(sleeps: list[float]) -> None:
    loops = FakeLoops()
    loops.fail_on("forward", error=http_status_error(400))
    with pytest.raises(ExternalError, match=r"HTTP 400, HTTPStatusError"):
        make_trainer(loops).forward([DATUM])
    assert (len(loops.kwargs("forward")), sleeps) == (1, [])


def test_forward_backward_is_never_retried(sleeps: list[float]) -> None:
    loops = FakeLoops()
    loops.fail_on("forward_backward", error=FakeServerShutdownError())
    with pytest.raises(ExternalError, match="forward_backward call failed"):
        make_trainer(loops).forward_backward([DATUM])
    assert (len(loops.kwargs("forward_backward")), sleeps) == (1, [])


def test_optim_step_uses_adam_with_the_learning_rate() -> None:
    loops = FakeLoops()
    make_trainer(loops).optim_step(3e-4)
    (call,) = loops.kwargs("optim_step")
    assert call["adam_params"].learning_rate == 3e-4


def test_saves_return_the_checkpoint_paths() -> None:
    loops = FakeLoops()
    trainer = make_trainer(loops)
    assert trainer.save_state("sft-test-state-epoch-1") == (
        "bt://loops:run1/state/sft-test-state-epoch-1"
    )
    assert trainer.save_sampler_weights("sft-test-sampler-epoch-1") == (
        "bt://loops:run1/sampler_weights/sft-test-sampler-epoch-1"
    )


def test_a_sampler_path_of_another_run_is_refused() -> None:
    loops = FakeLoops()
    trainer = make_trainer(loops)
    assert loops.trainer is not None
    loops.trainer.run_id = "otherrun"
    with pytest.raises(ExternalError, match="unexpected sampler checkpoint path"):
        trainer.save_sampler_weights("ckpt")


def test_an_empty_checkpoint_path_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    loops = FakeLoops()
    trainer = make_trainer(loops)
    monkeypatch.setattr(loops.trainer, "save_state", lambda name: FakeFuture(FakeSaved("")))
    with pytest.raises(ExternalError, match="no path for checkpoint ckpt"):
        trainer.save_state("ckpt")


def test_close_closes_the_sdk_client() -> None:
    loops = FakeLoops()
    make_trainer(loops).close()
    assert loops.trainer is not None
    assert loops.trainer.closed


# --- load_sampler_checkpoint ---------------------------------------------------------


def load(loops: FakeLoops, **overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "base_url": SAMPLER_URL,
        "model_id": "abc123",
        "deployment_id": "dep456",
        "sampler_id": "smp1",
        "session_id": "s1",
        "checkpoint_path": "bt://loops:run1/sampler_weights/ckpt",
        "base_model": "Qwen/Qwen3.5-9B",
        "prompt": (1, 2, 3),
        "ready_timeout": 600.0,
        "loops": loops,
    }
    load_sampler_checkpoint(KEY, **(arguments | overrides))


def test_load_sampler_checkpoint_samples_one_token_through_the_checkpoint() -> None:
    loops = FakeLoops()
    load(loops)
    assert loops.methods == ["SamplingClient", "sample", "sampler_close"]
    (created,) = loops.kwargs("SamplingClient")
    assert created == {
        "api_key": SECRET,
        "base_model": "Qwen/Qwen3.5-9B",
        "deployment": FakeBasetenDeployment(
            base_url=SAMPLER_URL, model_id="abc123", deployment_id="dep456", api_base_url=API_BASE
        ),
        "sampler_id": "smp1",
        "session_id": "s1",
        "checkpoint_path": "bt://loops:run1/sampler_weights/ckpt",
        "ready_timeout": 600.0,
    }
    (sample,) = loops.kwargs("sample")
    assert sample["prompt"] == FakeModelInput([1, 2, 3])
    assert sample["num_samples"] == 1
    params = sample["sampling_params"]
    assert (params.max_tokens, params.temperature) == (1, 0)


def test_load_sampler_checkpoint_closes_the_client_on_failure() -> None:
    loops = FakeLoops()
    loops.fail_on("sample", error=http_status_error(500))
    with pytest.raises(ExternalError, match=r"failed \(HTTP 500, HTTPStatusError\)"):
        load(loops)
    assert loops.methods[-1] == "sampler_close"


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": "https://elsewhere.example/sync"},
        {"model_id": "../x"},
        {"deployment_id": ""},
    ],
)
def test_load_sampler_checkpoint_refuses_bad_targets(overrides: dict[str, Any]) -> None:
    loops = FakeLoops()
    with pytest.raises(ExternalError):
        load(loops, **overrides)
    assert loops.calls == []
