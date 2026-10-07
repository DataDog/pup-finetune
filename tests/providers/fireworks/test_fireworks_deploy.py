# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Fireworks deployments: shape choice, create parameters, polling, smoke test and delete.

The real :class:`FireworksDeploymentsClient` and :class:`FireworksDeployer` run against the
fake SDK client (``tests/fakes/fireworks_deploy.py``), the smoke test against the fake chat
endpoint, and time against a fake clock: nothing reaches the network or waits.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    DeploymentNotReady,
    DeployOptions,
    Endpoint,
    ModelRef,
    ProviderAuthError,
    TransientError,
)
from pup_finetune.providers.fireworks.api import CLIENT_SOURCE, CLIENT_SOURCE_HEADER
from pup_finetune.providers.fireworks.client import (
    FireworksDeploymentsClient,
    ShapeVersion,
)
from pup_finetune.providers.fireworks.deploy import (
    INFERENCE_URL,
    POLL_INTERVAL_S,
    SCALE_TO_ZERO_WINDOW,
    FireworksDeployer,
    choose_shape,
)
from tests.fakes.chat_api import FakeChatApi, completion
from tests.fakes.fireworks import ACCOUNT, APIConnectionError, FakeSdkError
from tests.fakes.fireworks_api import API_KEY
from tests.fakes.fireworks_deploy import (
    SHAPE,
    FakeClock,
    FakeSdkFireworks,
    fake_deployer,
    shape_version,
)

MODEL = ModelRef(
    provider="fireworks",
    model_id="ft-20261006-abc123",
    uri=f"accounts/{ACCOUNT}/models/ft-20261006-abc123",
)
OPTIONS = DeployOptions(deployment_id="ft-20261006-abc123", display_name="pup-finetune run")
NAME = f"accounts/{ACCOUNT}/deployments/ft-20261006-abc123"


@pytest.fixture(autouse=True)
def _clean_secrets() -> Iterator[None]:
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def sdk() -> FakeSdkFireworks:
    return FakeSdkFireworks()


@pytest.fixture
def chat() -> FakeChatApi:
    return FakeChatApi(api_key=API_KEY)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def deployer(sdk: FakeSdkFireworks, chat: FakeChatApi, clock: FakeClock) -> FireworksDeployer:
    return fake_deployer(sdk, chat, clock)(SecretStr(API_KEY))


def created_kwargs(sdk: FakeSdkFireworks) -> dict[str, object]:
    (kwargs,) = [kwargs for name, kwargs in sdk.calls if name == "create"]
    return kwargs


# --- shape choice --------------------------------------------------------------------------


def _version(shape: str, *, validated: bool, latest: bool) -> ShapeVersion:
    return ShapeVersion(
        name=f"{shape}/versions/1", shape=shape, validated=validated, latest_validated=latest
    )


def test_the_first_validated_latest_shape_is_chosen() -> None:
    versions = [
        _version("a", validated=False, latest=False),
        _version("b", validated=True, latest=False),
        _version("c", validated=True, latest=True),
        _version("d", validated=True, latest=True),
    ]
    assert choose_shape(versions, "m") == "c"


def test_no_validated_shape_asks_for_one() -> None:
    with pytest.raises(UsageError, match="no validated deployment shape") as caught:
        choose_shape([_version("a", validated=True, latest=False)], "m")
    assert "--shape" in (caught.value.hint or "")
    with pytest.raises(UsageError, match="no validated"):
        choose_shape([], "m")


def test_the_sdk_shape_listing_is_read_by_its_field_names(sdk: FakeSdkFireworks) -> None:
    sdk.shapes = [shape_version("accounts/a/deploymentShapes/old", latest=False), shape_version()]
    client = FireworksDeploymentsClient(SecretStr(API_KEY), account_id=ACCOUNT, sdk=sdk.connect)
    versions = client.match_shapes(MODEL.uri)
    assert [v.shape for v in versions] == ["accounts/a/deploymentShapes/old", SHAPE]
    assert versions[1] == ShapeVersion(
        name=f"{SHAPE}/versions/v2", shape=SHAPE, validated=True, latest_validated=True
    )
    assert sdk.calls == [("match_for_model", {"model": MODEL.uri})]


def test_shape_versions_fall_back_to_the_version_name(sdk: FakeSdkFireworks) -> None:
    no_snapshot = shape_version("accounts/a/deploymentShapes/plain")
    no_snapshot.snapshot = None
    nameless = shape_version()
    nameless.name = None
    nameless.snapshot = None
    sdk.shapes = [no_snapshot, nameless]
    client = FireworksDeploymentsClient(SecretStr(API_KEY), account_id=ACCOUNT, sdk=sdk.connect)
    (version,) = client.match_shapes(MODEL.uri)
    assert version.shape == "accounts/a/deploymentShapes/plain"
    assert version.name == "accounts/a/deploymentShapes/plain/versions/v2"


def test_a_deployment_without_its_name_is_an_error(sdk: FakeSdkFireworks) -> None:
    sdk.records["d1"] = {"name": "", "state": "READY", "base_model": MODEL.uri}
    client = FireworksDeploymentsClient(SecretStr(API_KEY), account_id=ACCOUNT, sdk=sdk.connect)
    with pytest.raises(ExternalError, match="without its name"):
        client.get_deployment("d1")


# --- creating ------------------------------------------------------------------------------


def test_deploy_creates_with_hot_load_off_and_scale_to_zero(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    endpoint = deployer.deploy(MODEL, OPTIONS)
    assert created_kwargs(sdk) == {
        "base_model": MODEL.uri,
        "deployment_id": "ft-20261006-abc123",
        "deployment_shape": SHAPE,
        "display_name": "pup-finetune run",
        "min_replica_count": 0,
        "max_replica_count": 1,
        "autoscaling_policy": {"scale_to_zero_window": SCALE_TO_ZERO_WINDOW},
        "enable_hot_load": False,
    }
    assert endpoint == Endpoint(
        provider="fireworks",
        deployment_id="ft-20261006-abc123",
        model_id=MODEL.model_id,
        inference_model=NAME,
        url=INFERENCE_URL,
        name=NAME,
        shape=f"{SHAPE}/versions/v2",
    )
    (connection,) = sdk.connections
    assert connection["account_id"] == ACCOUNT
    assert connection["api_key"] == API_KEY
    assert connection["base_url"] == "https://api.fireworks.ai"
    assert connection["default_headers"] == {CLIENT_SOURCE_HEADER: CLIENT_SOURCE}


def test_an_explicit_shape_skips_matching(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    options = DeployOptions(deployment_id="d1", shape="accounts/a/deploymentShapes/mine")
    deployer.deploy(MODEL, options)
    assert "match_for_model" not in sdk.call_names()
    assert created_kwargs(sdk)["deployment_shape"] == "accounts/a/deploymentShapes/mine"
    assert created_kwargs(sdk)["display_name"] == "d1"


def test_a_taken_deployment_id_is_a_usage_error(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.fail["create"] = FakeSdkError("deployment already exists", status_code=409)
    with pytest.raises(UsageError, match="as taken") as caught:
        deployer.deploy(MODEL, OPTIONS)
    assert "--deployment-id" in (caught.value.hint or "")


def test_an_id_serving_another_model_is_a_usage_error(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    other = f"accounts/{ACCOUNT}/models/someone-elses"
    sdk.records[OPTIONS.deployment_id] = {"name": NAME, "state": "READY", "base_model": other}
    with pytest.raises(UsageError, match=f"already has a Fireworks deployment.*{other}") as caught:
        deployer.deploy(MODEL, OPTIONS)
    assert "--deployment-id" in (caught.value.hint or "")
    assert OPTIONS.deployment_id not in sdk.deleted


def test_the_create_is_sent_without_sdk_retries(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    deployer.deploy(MODEL, OPTIONS)
    assert sdk.client_options == [{"max_retries": 0}]
    assert sdk.call_names().count("create") == 1


@pytest.mark.parametrize("status", [500, 503, 504, 408])
def test_a_create_whose_answer_was_lost_is_adopted(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks, status: int
) -> None:
    sdk.lost["create"] = FakeSdkError("upstream timed out", status_code=status)
    sdk.states = ["CREATING", "READY"]  # read back, then polled
    created: list[str] = []
    endpoint = deployer.deploy(MODEL, OPTIONS, on_created=lambda e, s: created.append(s))
    assert created == ["CREATING"]  # the receipt is still written
    assert endpoint.inference_model == NAME
    assert sdk.call_names() == ["match_for_model", "create", "get", "get"]


def test_a_dropped_connection_on_create_is_adopted(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.lost["create"] = APIConnectionError("connection reset")
    assert deployer.deploy(MODEL, OPTIONS).deployment_id == OPTIONS.deployment_id


def test_a_conflict_from_our_own_earlier_create_is_adopted(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    """An SDK retry (or a rerun) of a create that had succeeded answers HTTP 409."""
    sdk.records[OPTIONS.deployment_id] = {
        "name": NAME,
        "state": "CREATING",
        "base_model": MODEL.uri,
    }
    endpoint = deployer.deploy(MODEL, OPTIONS)
    assert endpoint.name == NAME
    assert sdk.call_names().count("create") == 1


def test_a_failed_create_that_made_nothing_raises_the_failure(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.fail["create"] = FakeSdkError("internal error", status_code=503)
    with pytest.raises(TransientError, match="creating the Fireworks deployment"):
        deployer.deploy(MODEL, OPTIONS)
    assert sdk.call_names() == ["match_for_model", "create", "get"]


def test_a_rejected_create_is_not_read_back(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.fail["create"] = FakeSdkError("invalid shape", status_code=400)
    with pytest.raises(ExternalError, match="HTTP 400"):
        deployer.deploy(MODEL, OPTIONS)
    assert "get" not in sdk.call_names()


def test_another_providers_model_is_refused(deployer: FireworksDeployer) -> None:
    with pytest.raises(UsageError, match="cannot deploy a fake model"):
        deployer.deploy(ModelRef(provider="fake", model_id="m", uri="fake://m"), OPTIONS)


# --- waiting -------------------------------------------------------------------------------


def test_deploy_polls_until_ready(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks, clock: FakeClock
) -> None:
    sdk.states = ["CREATING", "CREATING", "READY"]
    seen: list[tuple[str, str, int]] = []

    def on_created(endpoint: Endpoint, state: str) -> None:
        seen.append((endpoint.deployment_id, state, sdk.call_names().count("get")))

    endpoint = deployer.deploy(MODEL, OPTIONS, on_created=on_created)
    # Reported once created, before the first read.
    assert seen == [("ft-20261006-abc123", "CREATING", 0)]
    assert sdk.call_names() == ["match_for_model", "create", "get", "get", "get"]
    assert clock.sleeps == [POLL_INTERVAL_S] * 3
    assert endpoint.inference_model == NAME


@pytest.mark.parametrize("state", ["FAILED", "DELETING", "DELETED"])
def test_a_bad_state_stops_the_wait(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks, state: str
) -> None:
    sdk.states = ["CREATING", state]
    created: list[str] = []
    with pytest.raises(DeploymentNotReady, match=f"is {state}") as caught:
        deployer.deploy(MODEL, OPTIONS, on_created=lambda e, s: created.append(s))
    assert caught.value.state == state
    assert not caught.value.timed_out
    assert "undeploy" in (caught.value.hint or "")
    assert created == ["CREATING"]
    if state == "FAILED":
        assert "replica failed to start" in caught.value.message


def test_the_wait_times_out(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks, clock: FakeClock
) -> None:
    sdk.states = ["CREATING"]
    options = DeployOptions(deployment_id="slow", timeout_s=60)
    with pytest.raises(DeploymentNotReady, match="not ready after 1 minutes") as caught:
        deployer.deploy(MODEL, options)
    assert caught.value.timed_out
    assert caught.value.state == "CREATING"
    assert sum(clock.sleeps) == pytest.approx(60)
    assert "slow" in sdk.records  # it still exists


def test_a_failed_read_while_waiting_is_retried(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.states = ["READY"]
    sdk.fail["get"] = FakeSdkError("internal error", status_code=503)
    endpoint = deployer.deploy(MODEL, OPTIONS)
    assert endpoint.deployment_id == OPTIONS.deployment_id
    assert sdk.call_names().count("get") == 2


def test_reads_that_fail_until_the_deadline_time_out(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks, clock: FakeClock
) -> None:
    def failing(deployment_id: str) -> object:
        sdk.record("get", deployment_id=deployment_id)
        raise FakeSdkError("internal error", status_code=503)

    sdk.deployments.get = failing  # type: ignore[method-assign]
    options = DeployOptions(deployment_id="slow", timeout_s=60)
    with pytest.raises(DeploymentNotReady, match="not ready after 1 minutes") as caught:
        deployer.deploy(MODEL, options)
    assert caught.value.timed_out
    assert caught.value.state == "CREATING"
    assert sum(clock.sleeps) == pytest.approx(60)


def test_a_permanent_read_failure_while_waiting_stops(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks, clock: FakeClock
) -> None:
    sdk.fail["get"] = FakeSdkError("bad request", status_code=400)
    with pytest.raises(ExternalError, match="HTTP 400"):
        deployer.deploy(MODEL, OPTIONS)
    assert clock.sleeps == [POLL_INTERVAL_S]


def test_a_rejected_key_while_waiting_stops(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.fail["get"] = FakeSdkError("invalid API key", status_code=401)
    with pytest.raises(ProviderAuthError):
        deployer.deploy(MODEL, OPTIONS)


def test_a_deployment_that_disappears_is_an_error(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    sdk.states = []
    original = sdk.deployments.get

    def vanish(deployment_id: str) -> object:
        sdk.records.clear()
        return original(deployment_id)

    sdk.deployments.get = vanish  # type: ignore[method-assign]
    with pytest.raises(ExternalError, match="disappeared"):
        deployer.deploy(MODEL, OPTIONS)


# --- smoke test ----------------------------------------------------------------------------


def _endpoint() -> Endpoint:
    return Endpoint(
        provider="fireworks",
        deployment_id="d",
        model_id="m",
        inference_model=NAME,
        url=INFERENCE_URL,
    )


def test_the_smoke_test_names_the_deployment(
    deployer: FireworksDeployer, chat: FakeChatApi
) -> None:
    smoke = deployer.smoke_test(_endpoint())
    assert smoke.ok and smoke.attempts == 1 and smoke.latency_s == 0.0
    (request,) = chat.requests
    assert str(request.url) == f"{INFERENCE_URL}/chat/completions"
    assert chat.bodies[0]["model"] == NAME
    assert chat.bodies[0]["max_tokens"] > 0


def test_the_smoke_test_retries_while_scaling_from_zero(
    deployer: FireworksDeployer, chat: FakeChatApi, clock: FakeClock
) -> None:
    chat.script += [httpx.Response(503), httpx.Response(503), completion("OK")]
    smoke = deployer.smoke_test(_endpoint())
    assert smoke.ok
    assert smoke.attempts == 3
    assert clock.sleeps == [POLL_INTERVAL_S, POLL_INTERVAL_S]


def test_a_smoke_test_that_never_answers_fails_without_raising(
    sdk: FakeSdkFireworks, chat: FakeChatApi, clock: FakeClock
) -> None:
    deployer = fake_deployer(sdk, chat, clock, smoke_timeout_s=60)(SecretStr(API_KEY))
    chat.script += [httpx.Response(503)] * 20
    smoke = deployer.smoke_test(_endpoint())
    assert not smoke.ok
    assert smoke.latency_s is None
    assert "HTTP 503" in smoke.detail
    assert sum(clock.sleeps) == pytest.approx(60)
    assert API_KEY not in smoke.detail


def test_a_smoke_test_answer_retrying_cannot_fix_fails_at_once(
    deployer: FireworksDeployer, chat: FakeChatApi, clock: FakeClock
) -> None:
    chat.script += [httpx.Response(404, json={"error": {"message": "model not found"}})]
    smoke = deployer.smoke_test(_endpoint())
    assert not smoke.ok
    assert smoke.attempts == 1
    assert "HTTP 404" in smoke.detail
    assert clock.sleeps == []


def test_a_rejected_key_fails_the_smoke_test_loudly(
    deployer: FireworksDeployer, chat: FakeChatApi
) -> None:
    chat.api_key = "fw-synthetic-other-key"
    with pytest.raises(ProviderAuthError):
        deployer.smoke_test(_endpoint())


# --- deleting ------------------------------------------------------------------------------


def test_undeploy_deletes_and_is_idempotent(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    endpoint = deployer.deploy(MODEL, OPTIONS)
    assert deployer.undeploy(endpoint) is True
    assert sdk.deleted == [OPTIONS.deployment_id]
    assert deployer.undeploy(endpoint) is False


def test_undeploy_refuses_a_deployment_of_another_account(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    endpoint = Endpoint(
        provider="fireworks",
        deployment_id="d",
        model_id="m",
        inference_model="accounts/other-team/deployments/d",
        name="accounts/other-team/deployments/d",
    )
    with pytest.raises(UsageError, match="other-team") as caught:
        deployer.undeploy(endpoint)
    assert "FIREWORKS_ACCOUNT_ID=other-team" in (caught.value.hint or "")
    assert "delete" not in sdk.call_names()


def test_a_failed_delete_is_an_external_error(
    deployer: FireworksDeployer, sdk: FakeSdkFireworks
) -> None:
    endpoint = deployer.deploy(MODEL, OPTIONS)
    sdk.fail["delete"] = FakeSdkError("internal error", status_code=500)
    with pytest.raises(ExternalError, match="deleting the Fireworks deployment"):
        deployer.undeploy(endpoint)


# --- the installed SDK ---------------------------------------------------------------------


def test_the_wrapper_matches_the_installed_sdk() -> None:
    """The keyword arguments and fields the wrapper (and the fake) use exist in the SDK."""
    import inspect
    import typing

    fireworks = pytest.importorskip("fireworks", reason="needs the fireworks extra")
    from fireworks.resources.deployment_shape_versions import DeploymentShapeVersionsResource
    from fireworks.resources.deployments import DeploymentsResource
    from fireworks.types.autoscaling_policy_param import AutoscalingPolicyParam
    from fireworks.types.deployment import Deployment
    from fireworks.types.deployment_shape_version import DeploymentShapeVersion

    client = inspect.signature(fireworks.Fireworks).parameters
    assert {"api_key", "account_id", "base_url", "default_headers"} <= set(client)
    create = inspect.signature(DeploymentsResource.create).parameters
    assert {
        "base_model",
        "deployment_id",
        "deployment_shape",
        "display_name",
        "min_replica_count",
        "max_replica_count",
        "autoscaling_policy",
        "enable_hot_load",
    } <= set(create)
    assert "deployment_id" in inspect.signature(DeploymentsResource.get).parameters
    assert "deployment_id" in inspect.signature(DeploymentsResource.delete).parameters
    assert "model" in inspect.signature(DeploymentShapeVersionsResource.match_for_model).parameters
    assert "scale_to_zero_window" in AutoscalingPolicyParam.__annotations__
    assert {"name", "validated", "latest_validated", "snapshot"} <= set(
        DeploymentShapeVersion.model_fields
    )
    fields = Deployment.model_fields
    assert {"name", "state", "base_model", "deployment_shape", "create_time", "status"} <= set(
        fields
    )
    states = set(typing.get_args(typing.get_args(fields["state"].annotation)[0]))
    assert {"CREATING", "READY", "FAILED", "DELETING", "DELETED"} <= states
