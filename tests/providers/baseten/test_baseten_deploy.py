# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten deployments beyond the contract: the lookups before the create, the create and its
read-back, the wait, the replicas, the smoke test, ownership and delete.

The real :class:`BasetenProvider` runs against the Loops fake's serving routes and a fake
clock: nothing reaches the network or waits.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    DeploymentNotReady,
    DeploymentRequest,
    DeployOptions,
    Endpoint,
    ModelRef,
    ProviderAuthError,
    TransientError,
)
from pup_finetune.providers.baseten import client
from pup_finetune.providers.baseten.client import DeploymentRejected
from pup_finetune.providers.baseten.deploy import POLL_INTERVAL_S, SMOKE_PROMPT
from pup_finetune.providers.baseten.provider import PROFILES, BasetenProvider
from pup_finetune.runtime import Runtime
from tests.fakes.baseten import FakeLoops, FakeModel, instance_type
from tests.fakes.fireworks_deploy import FakeClock
from tests.fakes.tokenizer import FakeChatTokenizer

KEY = "synthetic-baseten-key"
QWEN = PROFILES[0]
NAME = "pft-run"
DEPLOY = "POST /v1/loops/checkpoints/deploy"
URL = "https://model-dmodel1.api.baseten.co/deployment/ddep1/sync/v1"
CHAT = "POST /deployment/ddep1/sync/v1/chat/completions"
AUTOSCALING = "PATCH /v1/models/dmodel1/deployments/ddep1/autoscaling_settings"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL", "BASETEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield
    redaction.clear_secrets()


class World:
    """The provider on the Loops fake and a fake clock, with a checkpoint a run left."""

    def __init__(self, loops: FakeLoops | None = None) -> None:
        self.loops = loops or FakeLoops()
        self.clock = FakeClock()
        self.transport = self.loops.transport()
        self.provider = BasetenProvider(
            Runtime(),
            env={"BASETEN_API_KEY": KEY},
            transport=self.transport,
            loops=self.loops,
            tokenizer=FakeChatTokenizer(merge_newlines=True),
            sleep=self.clock.sleep,
            clock=self.clock.now,
        )
        self.checkpoint = self.loops.sampler_checkpoint("best", base_model=QWEN.base_model)
        self.model = ModelRef(provider="baseten", model_id="best", uri=self.checkpoint)
        self.requests: list[DeploymentRequest] = []
        self.created: list[tuple[Endpoint, str]] = []

    def deploy(self, **options: Any) -> Endpoint:
        settings = DeployOptions(deployment_id=NAME, display_name="pup-finetune run", **options)
        return self.provider.deploy(
            self.model,
            settings,
            on_creating=self.requests.append,
            on_created=lambda endpoint, state: self.created.append((endpoint, state)),
        )

    def request(self, **changes: Any) -> DeploymentRequest:
        fields = {
            "deployment_id": NAME,
            "base_model": self.checkpoint,
            "display_name": "pup-finetune run",
            "shape": "H100:1",
        }
        return DeploymentRequest(**(fields | changes))

    def creates(self) -> int:
        return self.transport.calls.count(DEPLOY)


@pytest.fixture
def world() -> World:
    return World()


# --- the create ------------------------------------------------------------------------------


def test_deploy_creates_the_checkpoint_as_a_model_named_after_the_deployment_id(
    world: World,
) -> None:
    endpoint = world.deploy()
    assert endpoint == Endpoint(
        provider="baseten",
        deployment_id=NAME,
        model_id="best",
        inference_model="best",
        url=URL,
        name="models/dmodel1/deployments/ddep1",
        shape="H100:1",
    )
    assert world.transport.bodies(DEPLOY) == [
        {
            "checkpoint_ids": [world.loops.checkpoint_id(world.checkpoint)],
            "model_name": NAME,
            "instance_type_id": "H100x22x120",
            "hf_secret_name": "hf_access_token",
        }
    ]
    assert world.requests == [world.request()]
    assert world.created == [(endpoint, "CREATING")]
    assert world.loops.model_named(NAME).deployments[0].status == "ACTIVE"
    # Every lookup happens before the create.
    calls = world.transport.calls
    assert calls.index(DEPLOY) > max(
        calls.index(call)
        for call in ("GET /v1/loops/checkpoints", "GET /v1/instance_types", "GET /v1/models")
    )


def test_an_explicit_shape_needs_exactly_its_gpus(world: World) -> None:
    world.deploy(shape="H100:2")
    (body,) = world.transport.bodies(DEPLOY)
    assert body["instance_type_id"] == "H100x22x240"


def test_a_shape_baseten_does_not_offer_is_refused_before_anything_is_created(
    world: World,
) -> None:
    with pytest.raises(UsageError, match="no instance type with exactly H100:4") as caught:
        world.deploy(shape="H100:4")
    assert caught.value.hint == "available: H100:1, H100:2"
    assert world.requests == []
    assert world.creates() == 0


@pytest.mark.parametrize("shape", ["sampler", "h100:1", "H100", "H100:0"])
def test_a_malformed_shape_is_refused_before_any_request(world: World, shape: str) -> None:
    with pytest.raises(UsageError, match="is not a Baseten GPU shape") as caught:
        world.deploy(shape=shape)
    assert "TYPE:COUNT" in (caught.value.hint or "")
    assert world.transport.calls == []


@pytest.mark.parametrize(
    "model",
    [
        ModelRef(provider="fireworks", model_id="m", uri="bt://loops:run1/sampler_weights/best"),
        ModelRef(provider="baseten", model_id="m", uri="bt://loops:run1/state/best"),
    ],
)
def test_a_model_that_is_no_baseten_sampler_checkpoint_is_refused(
    world: World, model: ModelRef
) -> None:
    world.model = model
    with pytest.raises(UsageError):
        world.deploy()
    assert world.transport.calls == []


def test_a_checkpoint_baseten_no_longer_has_is_refused(world: World) -> None:
    world.loops.expire(world.checkpoint)
    with pytest.raises(UsageError, match="Baseten has no checkpoint") as caught:
        world.deploy()
    assert "expired" in (caught.value.hint or "")
    assert world.requests == []


def test_a_checkpoint_of_a_model_the_provider_does_not_serve_is_refused(world: World) -> None:
    world.model = ModelRef(
        provider="baseten",
        model_id="other",
        uri=world.loops.sampler_checkpoint("other", base_model="Some/Other-Model"),
    )
    with pytest.raises(UsageError, match="was trained on Some/Other-Model"):
        world.deploy()
    assert world.creates() == 0


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"run_id": "run9"}, "does not match its path"),
        ({"target": "trainer"}, "does not match its path"),
        ({"sync_status": "SYNCING"}, "still syncing"),
    ],
)
def test_a_checkpoint_record_that_does_not_fit_is_an_error(
    world: World, record: dict[str, Any], message: str
) -> None:
    found = world.loops._checkpoint_record(world.checkpoint) | record
    world.transport.routes["GET /v1/loops/checkpoints"] = [(200, {"checkpoints": [found]})]
    with pytest.raises(ExternalError, match=message):
        world.deploy()
    assert world.requests == []


def test_a_name_baseten_already_has_is_refused_before_anything_is_created(
    world: World,
) -> None:
    world.deploy()
    world.requests.clear()
    with pytest.raises(UsageError, match=f"already has a model named {NAME}") as caught:
        world.deploy()
    assert "--deployment-id" in (caught.value.hint or "")
    assert world.requests == []
    assert world.creates() == 1


def test_a_refused_create_keeps_baseten_s_bdn_hint() -> None:
    world = World(FakeLoops(bdn=False))
    with pytest.raises(DeploymentRejected, match="BDN volumes are not enabled") as caught:
        world.deploy()
    assert caught.value.hint == (
        "nothing was created; ask Baseten to enable BDN volumes for your workspace"
    )
    assert len(world.requests) == 1
    assert world.created == []
    assert world.loops.models == {}


def test_a_create_that_failed_before_taking_effect_is_transient(world: World) -> None:
    world.loops.fail["deploy"] = httpx.Response(503, json={"detail": "unavailable"})
    with pytest.raises(TransientError, match="returned HTTP 503"):
        world.deploy()
    assert world.creates() == 1
    assert world.created == []


@pytest.mark.parametrize(
    "lost",
    [
        httpx.Response(504),
        httpx.Response(200, json={"model_id": "dmodel1"}),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["504", "no-ids", "timeout"],
)
def test_a_create_whose_answer_was_lost_is_adopted(world: World, lost: Any) -> None:
    world.loops.lost["deploy"] = lost
    endpoint = world.deploy()
    assert endpoint.name == "models/dmodel1/deployments/ddep1"
    assert world.creates() == 1
    assert [state for _, state in world.created] == ["CREATING"]


def test_a_failed_create_finding_another_model_under_the_name_is_refused(world: World) -> None:
    other = world.loops.sampler_checkpoint("other", base_model=QWEN.base_model)

    def taken_meanwhile(_: httpx.Request) -> httpx.Response:
        world.loops.deploy_checkpoint(other, NAME)
        return httpx.Response(503)

    world.transport.routes[DEPLOY] = [taken_meanwhile]
    with pytest.raises(UsageError, match="that this deploy did not create"):
        world.deploy()
    assert world.created == []


def test_a_lost_create_landing_beside_another_deployment_stays_transient(world: World) -> None:
    other = world.loops.sampler_checkpoint("other", base_model=QWEN.base_model)
    # Another create of the same name lands too: Baseten adds both to one model.
    world.loops.after_deploy = lambda _: world.loops.deploy_checkpoint(other, NAME)
    world.loops.lost["deploy"] = httpx.Response(504)
    with pytest.raises(TransientError, match="returned HTTP 504"):
        world.deploy()
    assert world.created == []
    with pytest.raises(TransientError, match="cannot tell whether"):
        world.provider.owns_deployment(world.request())
    assert world.loops.deleted == []


def test_a_lost_create_baseten_does_not_describe_yet_stays_transient(world: World) -> None:
    world.loops.lost["deploy"] = httpx.Response(504)
    world.transport.routes["GET /v1/models/dmodel1/deployments"] = [(200, {"deployments": []})]
    with pytest.raises(TransientError, match="returned HTTP 504"):
        world.deploy()
    assert world.created == []


# --- the wait and the replicas ---------------------------------------------------------------


def test_deploy_polls_until_active(world: World) -> None:
    world.loops.statuses = ["BUILDING", "DEPLOYING", "LOADING_MODEL", "ACTIVE"]
    world.deploy()
    assert world.clock.sleeps == [POLL_INTERVAL_S] * 3


@pytest.mark.parametrize("status", ["BUILD_FAILED", "DEPLOY_FAILED", "UNHEALTHY", "INACTIVE"])
def test_a_bad_status_stops_the_wait(world: World, status: str) -> None:
    world.loops.statuses = ["DEPLOYING", status]
    with pytest.raises(DeploymentNotReady, match=f"is {status}") as caught:
        world.deploy()
    assert (caught.value.state, caught.value.timed_out) == (status, False)
    assert "undeploy" in (caught.value.hint or "")
    assert len(world.created) == 1
    assert world.loops.deleted == []


def test_the_wait_times_out(world: World) -> None:
    world.loops.statuses = ["DEPLOYING"]
    with pytest.raises(DeploymentNotReady, match="not ready after 1 minutes") as caught:
        world.deploy(timeout_s=60)
    assert (caught.value.state, caught.value.timed_out) == ("DEPLOYING", True)
    assert sum(world.clock.sleeps) == 60


@pytest.mark.parametrize("status", [503, 429])
def test_a_failed_read_while_waiting_is_retried(world: World, status: int) -> None:
    world.loops.fail["status"] = httpx.Response(status)
    world.deploy()
    assert world.clock.sleeps[0] == POLL_INTERVAL_S


def test_a_permanent_read_failure_while_waiting_stops(world: World) -> None:
    world.loops.fail["status"] = httpx.Response(400)
    with pytest.raises(ExternalError, match="deployment lookup returned HTTP 400"):
        world.deploy()
    assert len(world.created) == 1


def test_a_deployment_that_disappears_while_starting_is_an_error(world: World) -> None:
    world.loops.statuses = ["DEPLOYING"]
    world.transport.routes["GET /v1/models/dmodel1/deployments/ddep1"] = [(404, {})]
    with pytest.raises(ExternalError, match="disappeared while starting"):
        world.deploy()


def test_the_replicas_are_set_to_the_requested_ones(world: World) -> None:
    world.deploy()
    assert world.transport.bodies(AUTOSCALING) == [{"min_replica": 0, "max_replica": 1}]
    replicas = world.loops.model_named(NAME).deployments[0].replicas
    assert replicas == {"min_replica": 0, "max_replica": 1}


def test_replicas_baseten_already_has_are_left_alone(world: World) -> None:
    world.loops.replicas = {"min_replica": 0, "max_replica": 1}
    world.deploy()
    assert AUTOSCALING not in world.transport.calls


# --- the smoke test --------------------------------------------------------------------------


def test_the_smoke_test_names_the_checkpoint_on_the_deployment(world: World) -> None:
    endpoint = world.deploy()
    smoke = world.provider.smoke_test(endpoint)
    assert (smoke.ok, smoke.attempts) == (True, 1)
    (request,) = [r for r in world.transport.requests if f"{r.method} {r.url.path}" == CHAT]
    assert str(request.url) == f"{URL}/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    (body,) = world.transport.bodies(CHAT)
    assert body["model"] == "best"
    assert body["messages"] == [{"role": "user", "content": SMOKE_PROMPT}]


def test_the_smoke_test_retries_while_a_replica_starts(world: World) -> None:
    endpoint = world.deploy()
    world.loops.chat_script += [httpx.Response(503), httpx.Response(429)]
    smoke = world.provider.smoke_test(endpoint)
    assert (smoke.ok, smoke.attempts) == (True, 3)


def test_a_smoke_test_that_never_answers_fails_without_raising(world: World) -> None:
    endpoint = world.deploy()
    world.loops.chat_script += [httpx.Response(503)] * 100
    smoke = world.provider.smoke_test(endpoint)
    assert smoke.ok is False
    assert smoke.attempts > 1
    assert "503" in smoke.detail


def test_a_smoke_test_answer_retrying_cannot_fix_fails_at_once(world: World) -> None:
    endpoint = world.deploy()
    smoke = world.provider.smoke_test(Endpoint(**{**_fields(endpoint), "inference_model": "x"}))
    assert (smoke.ok, smoke.attempts) == (False, 1)
    assert "404" in smoke.detail


def test_a_rejected_key_fails_the_smoke_test_loudly(world: World) -> None:
    endpoint = world.deploy()
    world.loops.chat_script.append(httpx.Response(401))
    with pytest.raises(ProviderAuthError):
        world.provider.smoke_test(endpoint)


def test_the_smoke_test_sends_the_key_only_to_baseten(world: World) -> None:
    endpoint = Endpoint(
        provider="baseten",
        deployment_id=NAME,
        model_id="best",
        inference_model="best",
        url="https://example.test/v1",
    )
    smoke = world.provider.smoke_test(endpoint)
    assert smoke.ok is False
    assert "non-Baseten URL" in smoke.detail
    assert world.transport.calls == []


# --- ownership and delete --------------------------------------------------------------------


def _fields(endpoint: Endpoint) -> dict[str, Any]:
    return {
        "provider": endpoint.provider,
        "deployment_id": endpoint.deployment_id,
        "model_id": endpoint.model_id,
        "inference_model": endpoint.inference_model,
        "url": endpoint.url,
        "name": endpoint.name,
        "shape": endpoint.shape,
    }


def test_owns_deployment_reads_the_model_back_by_name(world: World) -> None:
    assert world.provider.owns_deployment(world.request()) is None
    world.deploy()
    assert world.provider.owns_deployment(world.request()) is True
    other = world.loops.sampler_checkpoint("other", base_model=QWEN.base_model)
    assert world.provider.owns_deployment(world.request(base_model=other)) is False
    assert world.provider.owns_deployment(world.request(shape="H100:2")) is False
    missing = f"{world.checkpoint}-other"
    assert world.provider.owns_deployment(world.request(base_model=missing)) is False
    assert world.provider.owns_deployment(world.request(shape=None)) is False


def test_owns_deployment_cannot_tell_when_baseten_does_not_say_what_it_serves(
    world: World,
) -> None:
    world.deploy()
    world.loops.model_named(NAME).deployments[0].checkpoint_ids = None
    with pytest.raises(TransientError, match="cannot tell whether") as caught:
        world.provider.owns_deployment(world.request())
    assert "undeploy again" in (caught.value.hint or "")


def test_owns_deployment_cannot_tell_for_a_model_with_other_deployments(world: World) -> None:
    world.deploy()
    model = world.loops.model_named(NAME)
    model.deployments.append(model.deployments[0])
    with pytest.raises(TransientError, match="cannot tell whether"):
        world.provider.owns_deployment(world.request())


def test_undeploy_deletes_the_model_and_is_idempotent(world: World) -> None:
    endpoint = world.deploy()
    assert world.provider.undeploy(endpoint) is True
    assert world.loops.deleted == ["dmodel1"]
    assert world.provider.undeploy(endpoint) is False
    assert world.loops.checkpoints[world.checkpoint].expired is False
    # The name is free again.
    world.deploy()


def test_undeploy_without_ids_finds_the_model_by_name(world: World) -> None:
    world.deploy()
    endpoint = Endpoint(provider="baseten", deployment_id=NAME, model_id="best", inference_model="")
    assert world.provider.undeploy(endpoint) is True
    assert world.loops.deleted == ["dmodel1"]
    assert world.provider.undeploy(endpoint) is False


def test_undeploy_refuses_a_model_no_longer_named_after_the_deployment(world: World) -> None:
    endpoint = world.deploy()
    world.loops.model_named(NAME).name = "renamed"
    with pytest.raises(UsageError, match="is not named pft-run: nothing was deleted"):
        world.provider.undeploy(endpoint)
    assert world.loops.deleted == []


def test_undeploy_refuses_several_models_of_the_name(world: World) -> None:
    world.deploy()
    world.loops.models["dmodel9"] = FakeModel("dmodel9", NAME)
    endpoint = Endpoint(provider="baseten", deployment_id=NAME, model_id="best", inference_model="")
    with pytest.raises(UsageError, match="2 models named pft-run"):
        world.provider.undeploy(endpoint)
    with pytest.raises(UsageError, match="2 models named pft-run"):
        world.provider.owns_deployment(world.request())
    assert world.loops.deleted == []


def test_another_provider_s_endpoint_is_refused(world: World) -> None:
    endpoint = Endpoint(provider="fireworks", deployment_id=NAME, model_id="m", inference_model="m")
    for call in (world.provider.undeploy, world.provider.smoke_test):
        with pytest.raises(UsageError, match="does not serve a fireworks endpoint"):
            call(endpoint)
    assert world.transport.calls == []


def test_a_failed_delete_is_an_error(world: World) -> None:
    endpoint = world.deploy()
    world.loops.fail["delete"] = httpx.Response(500)
    with pytest.raises(TransientError, match="model deletion returned HTTP 500"):
        world.provider.undeploy(endpoint)
    assert world.loops.models


def test_the_instance_types_match_on_their_gpus_only() -> None:
    loops = FakeLoops()
    loops.instance_types = [instance_type("A100", 1), instance_type("H100", 1, cpus=26)]
    world = World(loops)
    world.deploy()
    (body,) = world.transport.bodies(DEPLOY)
    assert body["instance_type_id"] == "H100x26x120"
