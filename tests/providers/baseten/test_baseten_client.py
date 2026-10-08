# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Baseten client: HTTP routes, the Loops SDK import and session, and datum conversion.

No test reaches the network: HTTP goes through the fake transport and the SDK is the fake
module. Tests that build real SDK types skip when the ``baseten`` extra is not installed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.http_common import CLIENT_SOURCE
from pup_finetune.providers.base import ProviderAuthError, ProviderCapacityError, TransientError
from pup_finetune.providers.baseten import client
from pup_finetune.providers.baseten.client import (
    API_BASE,
    IGNORE_INDEX,
    INSTALL_HINT,
    BasetenHTTP,
    CreatedDeployment,
    DeploymentRejected,
    inference_url,
    loops_failure,
    loops_sampling_client,
    loops_sdk,
    loops_service,
    loops_trainer,
    to_loops_datum,
)
from pup_finetune.rendering.tokens import TokenizedExample
from tests.fakes.baseten import (
    BasetenTransport,
    FakeLoops,
    FakeModelInput,
    FakeServerShutdownError,
    FakeTensorData,
    baseten_transport,
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
LOGIN_HINT = "run `pup finetune auth login baseten` in your own terminal, or set BASETEN_API_KEY"
AUTH_HINT = f"{LOGIN_HINT} to a valid key"
UNREADABLE = "succeeded, but its answer could not be read"
CREATED_HINT = "it may have created something that bills; check the Baseten UI"
DEPLOY = "POST /v1/loops/checkpoints/deploy"
EXAMPLE = TokenizedExample((5, 6, 7), (6, 7, 8), (0.0, 1.0, 0.5))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (*GUARDED_ENV, "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


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


# --- BasetenHTTP transport -----------------------------------------------------------


def test_requests_send_the_api_key_and_return_the_json() -> None:
    http, transport = make_http({"GET /v1/loops/capabilities": (200, {"supported_models": []})})
    assert http.capabilities() == {"supported_models": []}
    (request,) = transport.requests
    assert str(request.url) == f"{API_BASE}/v1/loops/capabilities"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    assert request.headers["User-Agent"] == CLIENT_SOURCE


def test_an_error_status_is_a_fixed_message_without_the_body() -> None:
    body = {"detail": f"internal trace for {SECRET}"}
    http, _ = make_http({"GET /v1/loops/capabilities": (400, body)})
    with pytest.raises(ExternalError) as caught:
        http.capabilities()
    assert str(caught.value) == "baseten: the Loops capabilities request returned HTTP 400"
    assert not isinstance(caught.value, TransientError)
    assert "internal trace" not in str(caught.value)
    assert_clean(caught.value)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_an_auth_error_for_any_call(status: int) -> None:
    body = {"detail": f"bad key {SECRET}"}
    http, _ = make_http(
        {
            "GET /v1/loops/runs/run1": (status, body),
            "POST /v1/loops/runs/run1/deactivate": (status, body),
            "POST /v1/loops/samplers": (status, body),
            DEPLOY: (status, body),
            "DELETE /v1/models/m1": (status, body),
        }
    )
    calls = (
        lambda: http.get_run("run1"),
        lambda: http.deactivate_run("run1"),
        lambda: http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        ),
        lambda: deploy(http),
        lambda: http.delete_model("m1"),
    )
    for call in calls:
        with pytest.raises(ProviderAuthError, match=f"returned HTTP {status}$") as caught:
            call()
        assert caught.value.hint == AUTH_HINT
        assert not isinstance(caught.value, TransientError | DeploymentRejected)
        assert_clean(caught.value)


@pytest.mark.parametrize(
    ("headers", "hint"), [({"Retry-After": "30"}, "retry after 30 s"), ({}, "retry later")]
)
def test_http_429_is_a_capacity_error_for_any_call(headers: dict[str, str], hint: str) -> None:
    busy = httpx.Response(429, json={}, headers=headers)
    http, _ = make_http(
        {
            "GET /v1/loops/capabilities": busy,
            "POST /v1/models/m1/deployments/d1/deactivate": busy,
            "POST /v1/loops/samplers": busy,
            DEPLOY: busy,
        }
    )
    calls = (
        http.capabilities,
        lambda: http.deactivate_deployment("m1", "d1"),
        lambda: http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        ),
        lambda: deploy(http),
    )
    for call in calls:
        with pytest.raises(ProviderCapacityError, match=r"returned HTTP 429$") as caught:
            call()
        assert caught.value.hint == hint
        assert not isinstance(caught.value, TransientError | DeploymentRejected)


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
    with pytest.raises(ExternalError, match="returned an unexpected response") as caught:
        http.capabilities()
    assert not isinstance(caught.value, TransientError)


def unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def test_network_errors_say_whether_a_write_may_have_landed() -> None:
    http, _ = make_http(
        {
            "GET /v1/loops/capabilities": unreachable,
            "POST /v1/loops/runs/run1/deactivate": unreachable,
            "POST /v1/loops/samplers": unreachable,
            DEPLOY: unreachable,
        }
    )
    reason = r"\(ConnectError: connection refused\)"
    with pytest.raises(ExternalError, match=rf"could not reach Baseten .* {reason}") as read:
        http.capabilities()
    assert isinstance(read.value, TransientError)
    assert read.value.hint == "check your network connection and retry"
    with pytest.raises(TransientError):
        http.deactivate_run("run1")
    with pytest.raises(ExternalError) as write:
        http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        )
    assert not isinstance(write.value, TransientError)
    assert write.value.hint is not None
    assert "may have reached Baseten" in write.value.hint
    # The deploy's caller reads back what it may have created, so its outcome is transient.
    with pytest.raises(TransientError, match=reason) as deployed:
        deploy(http)
    assert deployed.value.hint is not None
    assert "may have reached Baseten" in deployed.value.hint


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_retryable_failures_of_repeatable_calls_are_transient(status: int) -> None:
    http, _ = make_http(
        {
            "GET /v1/loops/runs/run1": (status, {}),
            "POST /v1/loops/runs/run1/deactivate": (status, {}),
            "POST /v1/models/m1/deployments/d1/deactivate": (status, {}),
            "GET /v1/instance_types": (status, {}),
            "GET /v1/models": (status, {}),
            "DELETE /v1/models/m1": (status, {}),
            "POST /v1/loops/samplers": (status, {}),
        }
    )
    repeatables = (
        lambda: http.get_run("run1"),
        lambda: http.deactivate_run("run1"),
        lambda: http.deactivate_deployment("m1", "d1"),
        http.instance_types,
        lambda: http.list_models(name="pft-run"),
        lambda: http.delete_model("m1"),
    )
    for repeatable in repeatables:
        with pytest.raises(TransientError, match=f"returned HTTP {status}"):
            repeatable()
    with pytest.raises(ExternalError) as created:
        http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        )
    assert not isinstance(created.value, TransientError | DeploymentRejected)


@pytest.mark.parametrize("bad", ["", "two words", "line\nbreak", "caf\u00e9"])
def test_an_empty_or_malformed_key_is_refused_before_any_request(bad: str) -> None:
    loops = FakeLoops()
    refused = "the Baseten API key is (empty|malformed)"
    with pytest.raises(UsageError, match=refused) as caught:
        BasetenHTTP(SecretStr(bad), transport=baseten_transport({}))
    assert caught.value.hint == LOGIN_HINT
    with pytest.raises(UsageError, match=refused):
        loops_service(SecretStr(bad), loops=loops)
    assert loops.calls == []


def test_the_key_is_registered_for_redaction() -> None:
    make_http({})
    assert redaction.redact(f"key {SECRET}") == f"key {redaction.MASK}"


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


def test_get_checkpoint_is_none_without_a_record() -> None:
    http, _ = make_http({"GET /v1/loops/checkpoints": (200, {"checkpoints": []})})
    assert http.get_checkpoint("bt://loops:run1/sampler_weights/ckpt") is None


@pytest.mark.parametrize("records", [[{"id": "a"}, {"id": "b"}], None])
def test_get_checkpoint_needs_at_most_one_record(records: Any) -> None:
    http, _ = make_http({"GET /v1/loops/checkpoints": (200, {"checkpoints": records})})
    with pytest.raises(ExternalError, match="at most one checkpoint"):
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


class BrokenBody(httpx.SyncByteStream):
    """A body the connection drops in the middle of."""

    def __iter__(self) -> Iterator[bytes]:
        yield b'{"model_id": '
        raise httpx.ReadError("connection reset")


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ((200, {"model_id": "../m1", "deployment_id": "d1"}), "invalid Baseten model id"),
        ((200, {"model_id": "m1"}), "invalid Baseten deployment id"),
        ((200, ["m1", "d1"]), "returned an unexpected response"),
        (httpx.Response(200, content=b"{not json"), "returned malformed JSON"),
        (httpx.Response(200, content=b"x" * (4 * 1024 * 1024 + 1)), "more than 4 MiB"),
        (httpx.Response(200, stream=BrokenBody()), "ReadError: connection reset"),
    ],
    ids=["bad-id", "no-id", "bad-shape", "bad-json", "too-big", "read-failure"],
)
def test_an_accepted_create_with_an_unreadable_answer_is_transient(
    answer: Any, reason: str
) -> None:
    http, _ = make_http({DEPLOY: answer})
    with pytest.raises(TransientError, match=UNREADABLE) as caught:
        deploy(http)
    assert reason in caught.value.message
    assert caught.value.hint == CREATED_HINT


def test_an_accepted_sampler_create_without_a_sampler_is_transient() -> None:
    http, _ = make_http({"POST /v1/loops/samplers": (200, {"sampler": "smp1"})})
    with pytest.raises(TransientError, match=UNREADABLE):
        http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        )


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


def test_create_sampler_can_serve_the_base_model() -> None:
    sampler = {"id": "smp1", "model_id": "m1", "deployment_id": "d1", "base_url": SAMPLER_URL}
    http, transport = make_http({"POST /v1/loops/samplers": (200, {"sampler": sampler})})
    http.create_sampler(session_id="s1", base_model="Qwen/Qwen3.5-9B", max_seq_len=1024)
    (body,) = transport.bodies("POST /v1/loops/samplers")
    assert (body["base_model"], "model_path" in body) == ("Qwen/Qwen3.5-9B", False)


@pytest.mark.parametrize("target", [{}, {"model_path": "bt://x", "base_model": "m"}])
def test_create_sampler_serves_exactly_one_model(target: dict[str, str]) -> None:
    http, transport = make_http({})
    with pytest.raises(ValueError, match="exactly one"):
        http.create_sampler(session_id="s1", max_seq_len=1024, **target)
    assert transport.requests == []


@pytest.mark.parametrize(
    ("status", "rejected"), [(400, True), (422, True), (408, False), (503, False)]
)
def test_a_4xx_to_a_sampler_create_means_nothing_was_created(status: int, rejected: bool) -> None:
    http, _ = make_http({"POST /v1/loops/samplers": (status, {"detail": f"no {SECRET}"})})
    with pytest.raises(ExternalError, match=f"returned HTTP {status}") as caught:
        http.create_sampler(
            session_id="s1", model_path="bt://loops:r/sampler_weights/c", max_seq_len=8
        )
    assert isinstance(caught.value, DeploymentRejected) is rejected
    assert not isinstance(caught.value, TransientError)
    assert caught.value.message.endswith(": no ***") is rejected
    assert_clean(caught.value)


def test_deployment_routes() -> None:
    base = "/v1/models/m1/deployments/d1"
    http, transport = make_http(
        {
            f"GET {base}": (200, {"id": "d1", "model_id": "m1", "status": "ACTIVE"}),
            f"POST {base}/deactivate": (200, {"success": True}),
            "GET /v1/models/m1/deployments/gone": (404, {}),
        }
    )
    assert http.get_deployment("m1", "d1") == {"id": "d1", "model_id": "m1", "status": "ACTIVE"}
    assert http.get_deployment("m1", "gone") is None
    http.deactivate_deployment("m1", "d1")
    assert transport.calls[2:] == [f"POST {base}/deactivate"]


def test_model_and_deployment_reads_and_the_autoscaling_update() -> None:
    base = "/v1/models/m1/deployments/d1"
    config = {"training_checkpoints": {"loops_checkpoint_ids": ["ck1"]}}
    http, transport = make_http(
        {
            "GET /v1/models/m1": (200, {"id": "m1", "name": "pft-run"}),
            "GET /v1/models/gone": (404, {}),
            "GET /v1/models/m1/deployments": (200, {"deployments": [{"id": "d1"}]}),
            "GET /v1/models/gone/deployments": (404, {}),
            f"GET {base}/config": (200, {"config": config, "raw_config": None}),
            "GET /v1/models/m1/deployments/d2/config": (200, {"config": None}),
            f"PATCH {base}/autoscaling_settings": (200, {"status": "ACCEPTED", "message": ""}),
        }
    )
    assert http.get_model("m1") == {"id": "m1", "name": "pft-run"}
    assert http.get_model("gone") is None
    assert http.list_deployments("m1") == [{"id": "d1"}]
    assert http.list_deployments("gone") == []
    assert http.deployment_config("m1", "d1") == config
    assert http.deployment_config("m1", "d2") == {}
    http.set_autoscaling("m1", "d1", min_replica=0, max_replica=1)
    config_request = transport.requests[4]
    assert config_request.url.params["output_format"] == "parsed"
    assert transport.bodies(f"PATCH {base}/autoscaling_settings") == [
        {"min_replica": 0, "max_replica": 1}
    ]


@pytest.mark.parametrize("status", [500, 503])
def test_the_autoscaling_update_is_safe_to_repeat(status: int) -> None:
    route = "PATCH /v1/models/m1/deployments/d1/autoscaling_settings"
    http, _ = make_http({route: (status, {})})
    with pytest.raises(TransientError):
        http.set_autoscaling("m1", "d1", min_replica=0, max_replica=1)


@pytest.mark.parametrize("bad_id", ["../runs", "a/b", "", "a b"])
def test_ids_in_paths_are_validated_before_any_request(bad_id: str) -> None:
    http, transport = make_http({})
    with pytest.raises(ExternalError, match="invalid Baseten run id"):
        http.get_run(bad_id)
    assert transport.requests == []


# --- dedicated checkpoint deployment ---------------------------------------------

INSTANCE = {"id": "H100x22x120", "name": "H100", "gpu_type": "H100", "gpu_count": 1}


def test_instance_types_lists_them_over_rest() -> None:
    http, transport = make_http({"GET /v1/instance_types": (200, {"instance_types": [INSTANCE]})})
    assert http.instance_types() == [INSTANCE]
    (request,) = transport.requests
    assert str(request.url) == f"{API_BASE}/v1/instance_types"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"


@pytest.mark.parametrize("body", [{}, {"instance_types": None}, {"instance_types": ["H100"]}])
def test_an_unexpected_instance_type_listing_is_refused(body: dict[str, Any]) -> None:
    http, _ = make_http({"GET /v1/instance_types": (200, body)})
    with pytest.raises(ExternalError, match="returned an unexpected response") as caught:
        http.instance_types()
    assert not isinstance(caught.value, TransientError)


def test_list_models_finds_the_models_named_exactly() -> None:
    models = [{"id": "m1", "name": "pft-run"}, {"id": "m2", "name": "pft-run-2"}]
    http, transport = make_http({"GET /v1/models": (200, {"models": models})})
    assert http.list_models() == models
    assert http.list_models(name="pft-run") == [models[0]]
    assert http.list_models(name="other") == []
    assert "name" not in transport.requests[0].url.params
    assert transport.requests[1].url.params["name"] == "pft-run"


def deploy(http: BasetenHTTP) -> CreatedDeployment:
    return http.deploy_checkpoints(
        checkpoint_id="ck1", model_name="pft-run", instance_type_id="H100x22x120"
    )


def test_deploy_checkpoints_sends_the_documented_request_once() -> None:
    http, transport = make_http({DEPLOY: (200, {"model_id": "m1", "deployment_id": "d1"})})
    assert deploy(http) == CreatedDeployment(model_id="m1", deployment_id="d1")
    assert transport.calls == [DEPLOY]
    assert transport.bodies(DEPLOY) == [
        {
            "checkpoint_ids": ["ck1"],
            "model_name": "pft-run",
            "instance_type_id": "H100x22x120",
            "hf_secret_name": "hf_access_token",
        }
    ]
    assert transport.requests[0].headers["Authorization"] == f"Bearer {SECRET}"


@pytest.mark.parametrize(
    ("body", "detail", "hint"),
    [
        (
            {"detail": f"BDN volumes are not enabled for {SECRET}"},
            "BDN volumes are not enabled for ***",
            "nothing was created; ask Baseten to enable BDN volumes for your workspace",
        ),
        ({"error": {"message": "unknown instance type"}}, "unknown instance type", None),
        ({}, "", None),
    ],
    ids=["bdn", "reason", "no-reason"],
)
def test_a_4xx_to_the_deploy_is_a_rejection_with_baseten_s_reason(
    body: dict[str, Any], detail: str, hint: str | None
) -> None:
    http, transport = make_http({DEPLOY: (400, body)})
    with pytest.raises(DeploymentRejected) as caught:
        deploy(http)
    expected = "the checkpoint deployment returned HTTP 400"
    assert caught.value.message == (f"{expected}: {detail}" if detail else expected)
    assert caught.value.hint == (hint or "nothing was created")
    assert not isinstance(caught.value, TransientError)
    assert transport.calls == [DEPLOY]
    assert_clean(caught.value)


def test_an_html_refusal_is_not_quoted() -> None:
    page = httpx.Response(404, text="<html>gone</html>", headers={"content-type": "text/html"})
    http, _ = make_http({DEPLOY: page})
    with pytest.raises(DeploymentRejected) as caught:
        deploy(http)
    assert caught.value.message == "the checkpoint deployment returned HTTP 404"


@pytest.mark.parametrize("status", [302, 408, 500, 501, 503])
def test_any_other_deploy_failure_is_transient_as_it_may_have_created_the_model(
    status: int,
) -> None:
    http, transport = make_http({DEPLOY: (status, {"detail": "boom"})})
    with pytest.raises(TransientError, match=f"returned HTTP {status}$") as caught:
        deploy(http)
    assert caught.value.hint == CREATED_HINT
    assert transport.calls == [DEPLOY]


def test_delete_model_deletes_it_or_finds_it_gone() -> None:
    http, transport = make_http(
        {
            "DELETE /v1/models/m1": (200, {"id": "m1", "deleted": True}),
            "DELETE /v1/models/gone": (404, {"detail": "not found"}),
        }
    )
    assert http.delete_model("m1") is True
    assert http.delete_model("gone") is False
    assert transport.calls == ["DELETE /v1/models/m1", "DELETE /v1/models/gone"]


@pytest.mark.parametrize("body", [{"id": "m1", "deleted": False}, {"id": "m1"}, []])
def test_a_deletion_baseten_does_not_confirm_is_an_error(body: Any) -> None:
    http, _ = make_http({"DELETE /v1/models/m1": (200, body)})
    with pytest.raises(ExternalError, match="returned an unexpected response") as caught:
        http.delete_model("m1")
    assert not isinstance(caught.value, TransientError)


def test_delete_model_validates_the_id_before_any_request() -> None:
    http, transport = make_http({})
    with pytest.raises(ExternalError, match="invalid Baseten model id"):
        http.delete_model("../m1")
    assert transport.requests == []


# --- inference URLs ------------------------------------------------------------------


def test_an_inference_url_loses_its_trailing_slash() -> None:
    assert inference_url(f"{SAMPLER_URL}/") == SAMPLER_URL


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
    with pytest.raises(ExternalError, match="non-Baseten URL"):
        inference_url(url)


# --- the Loops SDK: environment and import -------------------------------------------


@pytest.mark.usefixtures("no_sdk")
@pytest.mark.parametrize("name", GUARDED_ENV)
def test_risky_sdk_environment_is_refused_before_the_import(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "https://elsewhere.example" if name == "LOOPS_BASE_URL" else "x")
    with pytest.raises(UsageError, match=name) as caught:
        loops_sdk()
    assert caught.value.hint == f"unset {name} and retry"


@pytest.mark.parametrize("name", client._REJECTED_ENV)
def test_risky_sdk_environment_is_refused_even_when_empty(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "")
    with pytest.raises(UsageError, match=name):
        loops_service(KEY, loops=FakeLoops())


def test_the_guard_also_runs_with_an_injected_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPS_REQUEST_LOG", "/tmp/requests.jsonl")  # noqa: S108 - never written
    loops = FakeLoops()
    with pytest.raises(UsageError, match="LOOPS_REQUEST_LOG"):
        loops_service(KEY, loops=loops)
    assert loops.calls == []


@pytest.mark.parametrize("value", ["https://api.baseten.co", "https://api.baseten.co/", ""])
def test_the_default_loops_base_url_is_allowed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LOOPS_BASE_URL", value)
    assert loops_service(KEY, loops=FakeLoops()).session_id == "session1"


@pytest.mark.usefixtures("no_sdk")
def test_a_missing_sdk_is_a_usage_error_with_the_install_hint() -> None:
    with pytest.raises(UsageError, match="Baseten Loops SDK, which is not installed") as caught:
        loops_service(KEY)
    assert caught.value.hint == INSTALL_HINT
    assert caught.value.exit_code == 2


def test_a_broken_sdk_says_to_reinstall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # An installed SDK whose own dependency is missing.
    (tmp_path / "baseten" / "loops").mkdir(parents=True)
    (tmp_path / "baseten" / "__init__.py").write_text("")
    (tmp_path / "baseten" / "loops" / "__init__.py").write_text("import pft_missing_dependency\n")
    for name in [name for name in sys.modules if name.partition(".")[0] == "baseten"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(tmp_path)
    try:
        with pytest.raises(UsageError, match="installed but failed to load") as caught:
            loops_sdk()
    finally:
        sys.modules.pop("baseten", None)
    assert "--reinstall" in (caught.value.hint or "")


def test_the_real_sdk_provides_every_name_the_client_uses() -> None:
    sdk = pytest.importorskip("baseten.loops", reason="needs the baseten extra")
    loaded = loops_sdk()
    for name in client._SDK_NAMES:
        assert getattr(loaded, name) is getattr(sdk, name)
    assert loaded.BasetenDeployment is sdk.sampling_client.BasetenDeployment


# --- loops_service -------------------------------------------------------------------


def test_loops_service_passes_the_key_and_settings_explicitly() -> None:
    loops = FakeLoops()
    service = loops_service(KEY, loops=loops)
    assert service.session_id == "session1"
    assert loops.kwargs("ServiceClient") == [
        {"api_key": SECRET, "base_url": API_BASE, "availability_model": "dedicated"}
    ]
    assert "BASETEN_API_KEY" not in os.environ


def raised_from(error: Exception, cause: Exception) -> Exception:
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    ("error", "kind", "message", "hint"),
    [
        (
            http_status_error(429, json={"detail": "busy"}, headers={"Retry-After": "30"}),
            ProviderCapacityError,
            "Baseten has no capacity for the Loops session (HTTP 429): busy",
            "retry after 30 s",
        ),
        (
            http_status_error(429),
            ProviderCapacityError,
            "Baseten has no capacity for the Loops session (HTTP 429)",
            "retry later",
        ),
        (
            http_status_error(401, json={"message": f"bad key {SECRET}"}),
            ProviderAuthError,
            "Baseten refused to create the Loops session (HTTP 401): bad key ***",
            AUTH_HINT,
        ),
        (
            raised_from(RuntimeError("session failed"), http_status_error(403)),
            ProviderAuthError,
            "Baseten refused to create the Loops session (HTTP 403)",
            AUTH_HINT,
        ),
        (
            http_status_error(400, json={"error": {"message": "bad request"}}),
            ExternalError,
            "Baseten refused to create the Loops session (HTTP 400): bad request",
            None,
        ),
        (
            RuntimeError(f"session store unavailable for {SECRET}"),
            ExternalError,
            "creating the Baseten Loops session failed "
            "(RuntimeError: session store unavailable for ***)",
            None,
        ),
        (
            httpx.ConnectError("connection refused"),
            ExternalError,
            "creating the Baseten Loops session failed (ConnectError: connection refused)",
            None,
        ),
    ],
    ids=[
        "capacity-retry-after",
        "capacity",
        "auth",
        "auth-in-chain",
        "refused",
        "runtime",
        "network",
    ],
)
def test_session_creation_errors_are_explained(
    error: Exception, kind: type[ExternalError], message: str, hint: str | None
) -> None:
    loops = FakeLoops()
    loops.fail_on("ServiceClient", error=error)
    with pytest.raises(ExternalError) as caught:
        loops_service(KEY, loops=loops)
    assert type(caught.value) is kind
    assert (caught.value.message, caught.value.hint) == (message, hint)
    assert caught.value.source == "baseten"
    assert caught.value.__cause__ is error
    assert_clean(caught.value)


# --- loops_trainer and loops_sampling_client -----------------------------------------

TRAINER = {
    "base_model": "Qwen/Qwen3.5-9B",
    "rank": 8,
    "seed": 7,
    "max_seq_len": 2048,
    "name": "pup-finetune-test",
}


def test_loops_trainer_creates_one_dedicated_run_without_a_sampler() -> None:
    loops = FakeLoops()
    trainer = loops_trainer(loops_service(KEY, loops=loops), **TRAINER)
    assert trainer.run_id == "run1"
    assert loops.kwargs("create_lora_training_client") == [
        {**TRAINER, "replicas": 1, "with_sampler": False}
    ]


ORPHAN_HINT = "a run may remain in Loops session session1; deactivate it in the Baseten UI"
TRAINER_URL = "https://model-t1.api.baseten.co/deployment/d1/sync/init_trainer_server"


def _without_request(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.baseten.co/v1/loops/trainers")
    return httpx.HTTPStatusError("HTTP", request=request, response=httpx.Response(status))


@pytest.mark.parametrize(
    ("error", "kind", "message", "hint"),
    [
        (
            http_status_error(422, json={"detail": "unsupported model"}),
            ExternalError,
            "Baseten refused to create the trainer (HTTP 422): unsupported model",
            None,
        ),
        (
            http_status_error(429, json={"detail": "no H100s"}),
            ProviderCapacityError,
            "Baseten has no capacity for the trainer (HTTP 429): no H100s",
            "nothing was left running; retry later",
        ),
        (
            http_status_error(403),
            ProviderAuthError,
            "Baseten refused to create the trainer (HTTP 403)",
            AUTH_HINT,
        ),
        (
            RuntimeError(f"trainer never became ready for {SECRET}"),
            ExternalError,
            "creating the Baseten trainer failed (RuntimeError: trainer never became ready for ***)",
            ORPHAN_HINT,
        ),
        (
            httpx.ReadTimeout("slow"),
            ExternalError,
            "creating the Baseten trainer failed (ReadTimeout: slow)",
            ORPHAN_HINT,
        ),
        (
            http_status_error(503),
            ExternalError,
            "creating the Baseten trainer failed (HTTP 503)",
            ORPHAN_HINT,
        ),
        (
            http_status_error(400, json={"detail": "bad rank"}, url=TRAINER_URL),
            ExternalError,
            "creating the Baseten trainer failed (HTTP 400): bad rank",
            ORPHAN_HINT,
        ),
        (
            http_status_error(401, url=TRAINER_URL),
            ProviderAuthError,
            "creating the Baseten trainer failed (HTTP 401)",
            f"{ORPHAN_HINT}; {AUTH_HINT}",
        ),
        (
            http_status_error(429, url=TRAINER_URL),
            ProviderCapacityError,
            "Baseten has no capacity for the trainer (HTTP 429)",
            f"{ORPHAN_HINT}; retry later",
        ),
        (
            _without_request(400),
            ExternalError,
            "creating the Baseten trainer failed (HTTP 400)",
            ORPHAN_HINT,
        ),
    ],
    ids=[
        "refused",
        "capacity",
        "auth",
        "runtime",
        "timeout",
        "server-error",
        "refused-after-the-create",
        "auth-after-the-create",
        "capacity-after-the-create",
        "no-request",
    ],
)
def test_trainer_creation_errors_say_whether_a_run_may_remain(
    error: Exception, kind: type[ExternalError], message: str, hint: str | None
) -> None:
    loops = FakeLoops()
    service = loops_service(KEY, loops=loops)
    loops.fail_on("create_lora_training_client", error=error)
    with pytest.raises(ExternalError) as caught:
        loops_trainer(service, **TRAINER)
    assert type(caught.value) is kind
    assert (caught.value.message, caught.value.hint) == (message, hint)
    assert_clean(caught.value)


def test_an_interrupted_trainer_creation_notes_the_session() -> None:
    loops = FakeLoops()
    service = loops_service(KEY, loops=loops)
    loops.fail_on("create_lora_training_client", error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as caught:
        loops_trainer(service, **TRAINER)
    assert caught.value.__notes__ == [ORPHAN_HINT]


def test_a_trainer_without_a_valid_run_id_is_closed_and_refused() -> None:
    loops = FakeLoops()
    service: Any = loops_service(KEY, loops=loops)
    created = service.create_lora_training_client
    service.create_lora_training_client = lambda **kwargs: _with_run_id(created(**kwargs))
    with pytest.raises(ExternalError, match="no valid run id"):
        loops_trainer(service, **TRAINER)
    assert loops.trainers[0].closed


def _with_run_id(trainer: Any) -> Any:
    trainer.run_id = "../run1"
    return trainer


SAMPLER = {"id": "smp1", "model_id": "abc123", "deployment_id": "dep456", "base_url": SAMPLER_URL}


def test_loops_sampling_client_attaches_to_the_sampler_with_the_key() -> None:
    loops = FakeLoops()
    service = loops_service(KEY, loops=loops)
    created = BasetenHTTP(KEY, transport=loops.transport()).create_sampler(
        session_id=service.session_id, base_model="Qwen/Qwen3.5-9B", max_seq_len=1024
    )
    loops_sampling_client(
        KEY,
        created,
        base_model="Qwen/Qwen3.5-9B",
        session_id="session1",
        checkpoint_path=None,
        ready_timeout=60.0,
        timeout=30.0,
        loops=loops,
    )
    (kwargs,) = loops.kwargs("SamplingClient")
    assert kwargs.pop("deployment") == loops.BasetenDeployment(
        base_url=created["base_url"],
        model_id="model1",
        deployment_id="dep1",
        api_base_url=API_BASE,
    )
    assert kwargs == {
        "api_key": SECRET,
        "base_model": "Qwen/Qwen3.5-9B",
        "sampler_id": "sampler1",
        "session_id": "session1",
        "checkpoint_path": None,
        "ready_timeout": 60.0,
        "timeout": 30.0,
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ({"base_url": "https://evil.example.com/sync"}, "non-Baseten URL"),
        ({"base_url": None}, "without its URL"),
        ({"id": "a/b"}, "invalid Baseten sampler id"),
    ],
)
def test_loops_sampling_client_refuses_a_sampler_it_cannot_trust(
    field: dict[str, Any], message: str
) -> None:
    loops = FakeLoops()
    with pytest.raises(ExternalError, match=message):
        loops_sampling_client(
            KEY,
            {**SAMPLER, **field},
            base_model="Qwen/Qwen3.5-9B",
            session_id="session1",
            checkpoint_path=None,
            ready_timeout=60.0,
            timeout=30.0,
            loops=loops,
        )
    assert loops.calls == []


# --- loops_failure -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "repeatable", "kind", "message", "hint"),
    [
        (
            http_status_error(503),
            True,
            TransientError,
            "the call failed (HTTP 503, HTTPStatusError)",
            None,
        ),
        (
            http_status_error(503),
            False,
            ExternalError,
            "the call failed (HTTP 503, HTTPStatusError)",
            None,
        ),
        (httpx.ReadTimeout("slow"), True, TransientError, "the call failed (ReadTimeout)", None),
        (TimeoutError("slow"), True, TransientError, "the call failed (TimeoutError)", None),
        (
            FakeServerShutdownError("going away"),
            True,
            TransientError,
            "the call failed (FakeServerShutdownError)",
            None,
        ),
        (
            http_status_error(400),
            True,
            ExternalError,
            "the call failed (HTTP 400, HTTPStatusError)",
            None,
        ),
        (
            http_status_error(429, headers={"Retry-After": "5"}),
            True,
            ProviderCapacityError,
            "the call failed (HTTP 429, HTTPStatusError)",
            "retry after 5 s",
        ),
        (
            raised_from(RuntimeError(f"denied {SECRET}"), http_status_error(401)),
            True,
            ProviderAuthError,
            "the call failed (HTTP 401, RuntimeError)",
            AUTH_HINT,
        ),
    ],
    ids=[
        "5xx-repeatable",
        "5xx-not-repeatable",
        "read-timeout",
        "timeout",
        "shutdown",
        "4xx",
        "capacity",
        "auth-in-chain",
    ],
)
def test_loops_failure_maps_sdk_errors_to_the_shared_classes(
    error: Exception, repeatable: bool, kind: type[ExternalError], message: str, hint: str | None
) -> None:
    mapped = loops_failure(error, "the call", FakeLoops(), repeatable=repeatable)
    assert type(mapped) is kind
    assert (mapped.message, mapped.hint, mapped.source) == (message, hint, "baseten")
    assert_clean(mapped)


# --- datum conversion ----------------------------------------------------------------


def test_a_datum_carries_the_shifted_arrays_and_ignores_unweighted_targets() -> None:
    datum = to_loops_datum(EXAMPLE, FakeLoops())
    assert datum.model_input == FakeModelInput([5, 6, 7])
    assert datum.loss_fn_inputs == {
        "target_tokens": FakeTensorData(data=[IGNORE_INDEX, 7, 8], dtype="int64", shape=[3]),
        "weights": FakeTensorData(data=[0.0, 1.0, 0.5], dtype="float32", shape=[3]),
    }


def test_a_datum_builds_with_the_real_sdk() -> None:
    sdk = pytest.importorskip("baseten.loops", reason="needs the baseten extra")
    datum = to_loops_datum(EXAMPLE, loops_sdk())
    assert isinstance(datum, sdk.Datum)
    assert datum.model_input.to_ints() == [5, 6, 7]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [IGNORE_INDEX, 7, 8]
    assert datum.loss_fn_inputs["weights"].tolist() == [0.0, 1.0, 0.5]
    assert datum.loss_fn_inputs["weights"].shape == [3]
