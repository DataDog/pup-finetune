# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""In-memory Fireworks deployments, behind the SDK's ``Fireworks`` client, and a fake clock.

:class:`FakeSdkFireworks` stands in for the SDK client the real
:class:`~pup_finetune.providers.fireworks.client.FireworksDeploymentsClient` drives, with the
attribute names and keyword arguments of the installed SDK
(``deployment_shape_versions.match_for_model``, ``deployments.create/get/delete``)::

    sdk = FakeSdkFireworks()  # one validated shape; deployments go CREATING then READY
    sdk.states = ["CREATING", "CREATING", "READY"]  # what successive reads report
    sdk.fail["create"] = FakeSdkError("conflict", status_code=409)
    sdk.lost["create"] = FakeSdkError("timed out", status_code=504)  # created, answer lost
    client = FireworksDeploymentsClient(SecretStr(API_KEY), account_id=ACCOUNT, sdk=sdk.connect)

Every call is recorded in :attr:`FakeSdkFireworks.calls` with its keyword arguments.
:func:`fake_deployer` builds the provider's deployer on it, with the smoke test going to a
:class:`~tests.fakes.chat_api.FakeChatApi` and time kept by a :class:`FakeClock`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr

from pup_finetune.inference import ChatClient
from pup_finetune.providers.fireworks.client import FireworksDeploymentsClient
from pup_finetune.providers.fireworks.deploy import INFERENCE_URL, FireworksDeployer
from tests.fakes.chat_api import FakeChatApi
from tests.fakes.fireworks import ACCOUNT, FakeSdkError

SHAPE = f"accounts/{ACCOUNT}/deploymentShapes/synthetic-shape"


def shape_version(
    shape: str = SHAPE, *, version: str = "v2", validated: bool = True, latest: bool = True
) -> SimpleNamespace:
    """A deployment shape version as ``match_for_model`` lists it."""
    return SimpleNamespace(
        name=f"{shape}/versions/{version}",
        validated=validated,
        latest_validated=latest,
        snapshot=SimpleNamespace(name=shape),
        create_time=datetime(2026, 1, 1, tzinfo=UTC),
    )


@dataclass
class FakeClock:
    """A monotonic clock that only moves when :meth:`sleep` is called."""

    now_s: float = 1000.0
    sleeps: list[float] = field(default_factory=list)

    def now(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds


class _ShapeVersions:
    def __init__(self, sdk: FakeSdkFireworks) -> None:
        self._sdk = sdk

    def match_for_model(self, model: str) -> SimpleNamespace:
        self._sdk.record("match_for_model", model=model)
        return SimpleNamespace(deployment_shape_versions=list(self._sdk.shapes))


class _Deployments:
    def __init__(self, sdk: FakeSdkFireworks) -> None:
        self._sdk = sdk

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self._sdk.record("create", **kwargs)
        deployment_id = kwargs["deployment_id"]
        if deployment_id in self._sdk.records:
            raise FakeSdkError("deployment already exists", status_code=409)
        record: dict[str, Any] = {
            "name": f"accounts/{self._sdk.account_id}/deployments/{deployment_id}",
            "state": "CREATING",
            "base_model": kwargs["base_model"],
            "deployment_shape": f"{kwargs['deployment_shape']}/versions/v2",
            "create_time": datetime(2026, 10, 6, 12, 0, tzinfo=UTC),
            "status": SimpleNamespace(message=""),
        }
        self._sdk.records[deployment_id] = record
        lost = self._sdk.lost.pop("create", None)
        if lost is not None:
            raise lost
        return SimpleNamespace(**record)

    def get(self, deployment_id: str) -> SimpleNamespace:
        self._sdk.record("get", deployment_id=deployment_id)
        record = self._sdk.records.get(deployment_id)
        if record is None:
            raise FakeSdkError("deployment not found", status_code=404)
        if self._sdk.states:
            state = self._sdk.states.pop(0)
            record["state"] = state
            if state == "FAILED":
                record["status"] = SimpleNamespace(message="replica failed to start")
        return SimpleNamespace(**record)

    def delete(self, deployment_id: str) -> dict[str, Any]:
        self._sdk.record("delete", deployment_id=deployment_id)
        if self._sdk.records.pop(deployment_id, None) is None:
            raise FakeSdkError("deployment not found", status_code=404)
        self._sdk.deleted.append(deployment_id)
        return {}


class FakeSdkFireworks:
    """The SDK client, as far as deployments use it. See the module docstring."""

    def __init__(self) -> None:
        self.account_id = ACCOUNT
        self.shapes: list[SimpleNamespace] = [shape_version()]
        self.states: list[str] = ["READY"]
        """The state each read reports, in order; the last one read stays."""
        self.records: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self.fail: dict[str, Exception] = {}
        """An error to raise from the named call (``create``, ``get``, ...), once."""
        self.lost: dict[str, BaseException] = {}
        """An error to raise from the named call once it took effect (its answer was lost)."""
        self.client_options: list[dict[str, Any]] = []
        """The options of every ``with_options`` copy of the client, in order."""
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.connections: list[dict[str, Any]] = []
        self.closed = 0
        self.deployments = _Deployments(self)
        self.deployment_shape_versions = _ShapeVersions(self)

    def connect(self, **kwargs: Any) -> FakeSdkFireworks:
        """The SDK client's constructor: records its keyword arguments."""
        self.connections.append(kwargs)
        self.account_id = kwargs.get("account_id") or self.account_id
        return self

    def with_options(self, **kwargs: Any) -> FakeSdkFireworks:
        """The SDK client's copy with other options: records them, and is this fake."""
        self.client_options.append(kwargs)
        return self

    def record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        error = self.fail.pop(name, None)
        if error is not None:
            raise error

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def close(self) -> None:
        self.closed += 1


def fake_deployer(
    sdk: FakeSdkFireworks,
    chat: FakeChatApi,
    clock: FakeClock,
    *,
    smoke_timeout_s: float = 600.0,
    account_id: str = ACCOUNT,
) -> Callable[[SecretStr], FireworksDeployer]:
    """The provider's ``deployer`` factory, on ``sdk``, ``chat`` and ``clock``, for the key's
    account ``account_id``."""

    def build(key: SecretStr) -> FireworksDeployer:
        client = FireworksDeploymentsClient(key, account_id=account_id, sdk=sdk.connect)

        def open_chat() -> ChatClient:
            return ChatClient(
                INFERENCE_URL, key, credential="fireworks", max_attempts=1, transport=chat.transport
            )

        return FireworksDeployer(
            client,
            chat=open_chat,
            clock=clock.now,
            sleep=clock.sleep,
            smoke_timeout_s=smoke_timeout_s,
        )

    return build
