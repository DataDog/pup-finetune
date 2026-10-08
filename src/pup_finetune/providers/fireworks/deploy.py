# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""On-demand Fireworks deployments of a promoted model: create, wait, smoke-test, delete.

:class:`FireworksDeployer` drives :class:`~.client.FireworksDeployments` (the SDK, behind
``client.py``):

1. **Shape.** ``--shape`` when given, else the first shape version Fireworks matches to the
   model that is both validated and the latest validated one (:func:`choose_shape`).
2. **Create** the deployment with the promoted model as its base model, at most one replica
   and none when idle (it scales to zero after :data:`SCALE_TO_ZERO_WINDOW` without
   requests), and hot loading off: the SDK's own deployment helper turns it on by default,
   for training rollouts, which a served model does not need.
3. **Wait** until it is ``READY``, reading it every :data:`POLL_INTERVAL_S` seconds until the
   deadline (a read that fails transiently is retried, any other failure stops the wait);
   ``FAILED``, ``DELETING`` or ``DELETED`` stop the wait at once. A bad state or the deadline is
   :class:`~pup_finetune.providers.base.DeploymentNotReady`: the deployment may still exist.
4. **Smoke test** (:meth:`FireworksDeployer.smoke_test`): one short chat completion through
   the OpenAI-compatible inference API, naming the deployment
   (``accounts/<account>/deployments/<id>``, the form Fireworks documents for querying a
   deployment). A deployment scaled to zero answers HTTP 503 while a replica starts, so
   transient failures (5xx, 429, a dropped connection) are retried every
   :data:`POLL_INTERVAL_S` seconds for up to :data:`SMOKE_TIMEOUT_S`; any other failed
   answer fails the smoke test at once.

The clock and sleep are injectable, so tests run the whole flow instantly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Final, Protocol

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.inference import DEFAULT_JUDGE_BASE_URL, ChatCompleter
from pup_finetune.output import info
from pup_finetune.providers.base import (
    DeploymentNotReady,
    DeploymentRequest,
    DeployOptions,
    Endpoint,
    ModelRef,
    ProviderAuthError,
    ProviderCapacityError,
    SmokeTest,
    TransientError,
)
from pup_finetune.providers.fireworks import ENV_VAR, PROVIDER_NAME
from pup_finetune.providers.fireworks.client import (
    SOURCE,
    CreateRequest,
    DeploymentRecord,
    FireworksDeployments,
    ShapeVersion,
)
from pup_finetune.redaction import redact

__all__ = [
    "BAD_STATES",
    "INFERENCE_URL",
    "POLL_INTERVAL_S",
    "READY",
    "SCALE_TO_ZERO_WINDOW",
    "SMOKE_TIMEOUT_S",
    "ChatSession",
    "FireworksDeployer",
    "choose_shape",
    "deployment_name",
]

INFERENCE_URL: Final = DEFAULT_JUDGE_BASE_URL
"""The OpenAI-compatible inference API every deployment is queried through (Fireworks'
serverless inference, which also serves the default judges)."""
POLL_INTERVAL_S: Final = 15.0
SMOKE_TIMEOUT_S: Final = 10 * 60.0
"""How long the smoke test keeps retrying while a replica starts."""
SCALE_TO_ZERO_WINDOW: Final = "300s"
"""How long without requests before the deployment scales to zero: Fireworks' minimum."""
READY: Final = "READY"
BAD_STATES: Final = frozenset({"FAILED", "DELETING", "DELETED"})
SMOKE_PROMPT: Final = "Reply with the single word OK."
SMOKE_MAX_TOKENS: Final = 16


class ChatSession(ChatCompleter, AbstractContextManager["ChatSession"], Protocol):
    """A chat client that is also a context manager, closed on exit (as ``ChatClient`` is)."""


def deployment_name(account_id: str, deployment_id: str) -> str:
    """A deployment's resource name, which is also the model name inference requests use."""
    return f"accounts/{account_id}/deployments/{deployment_id}"


def _account_of(name: str | None) -> str | None:
    """The account in a deployment's resource name, ``accounts/<account>/deployments/<id>``."""
    parts = (name or "").split("/")
    if len(parts) == 4 and parts[0] == "accounts" and parts[2] == "deployments" and parts[1]:
        return parts[1]
    return None


def choose_shape(versions: Sequence[ShapeVersion], model: str) -> str:
    """The first validated, latest-validated shape in ``versions``.

    Raises :class:`UsageError` when there is none, with a hint to pass ``--shape``.
    """
    for version in versions:
        if version.validated and version.latest_validated:
            return version.shape
    found = f" ({len(versions)} matched, none validated)" if versions else ""
    raise UsageError(
        f"Fireworks has no validated deployment shape for {model} in this account{found}",
        hint="pass --shape with a deployment shape your account can use for this model",
    )


class FireworksDeployer:
    """Deploys, smoke-tests and deletes on Fireworks. See the module docstring.

    ``chat`` opens the client the smoke test sends its completion with (a context manager,
    closed after the test); ``clock`` is a monotonic clock in seconds and ``sleep`` waits.
    """

    def __init__(
        self,
        deployments: FireworksDeployments,
        *,
        chat: Callable[[], ChatSession],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_s: float = POLL_INTERVAL_S,
        smoke_timeout_s: float = SMOKE_TIMEOUT_S,
    ) -> None:
        self._deployments = deployments
        self._chat = chat
        self._clock = clock
        self._sleep = sleep
        self._poll_interval_s = poll_interval_s
        self._smoke_timeout_s = smoke_timeout_s

    @property
    def account_id(self) -> str:
        return self._deployments.account_id

    def close(self) -> None:
        self._deployments.close()

    def deployment_name(self, deployment_id: str) -> str:
        """The resource name the deployment ``deployment_id`` gets: in this key's account."""
        return deployment_name(self.account_id, deployment_id)

    def deploy(
        self,
        model: ModelRef,
        options: DeployOptions,
        *,
        on_creating: Callable[[DeploymentRequest], None] | None = None,
        on_created: Callable[[Endpoint, str], None] | None = None,
    ) -> Endpoint:
        """Create the deployment and wait until it is ``READY``; see the module docstring.

        ``on_creating`` is called once the shape is chosen, right before the create is sent.
        """
        if model.provider != PROVIDER_NAME:
            raise UsageError(f"the fireworks provider cannot deploy a {model.provider} model")
        shape = options.shape or choose_shape(self._deployments.match_shapes(model.uri), model.uri)
        display_name = (options.display_name or options.deployment_id)[:63]
        if on_creating is not None:
            on_creating(
                DeploymentRequest(
                    deployment_id=options.deployment_id,
                    base_model=model.uri,
                    name=self.deployment_name(options.deployment_id),
                    display_name=display_name,
                    shape=shape,
                    min_replicas=options.min_replicas,
                    max_replicas=options.max_replicas,
                )
            )
        record = self._deployments.create_deployment(
            base_model=model.uri,
            deployment_id=options.deployment_id,
            shape=shape,
            display_name=display_name,
            min_replica_count=options.min_replicas,
            max_replica_count=options.max_replicas,
            scale_to_zero_window=SCALE_TO_ZERO_WINDOW,
            enable_hot_load=False,
        )
        endpoint = self._endpoint(model, options.deployment_id, record, shape)
        if on_created is not None:
            on_created(endpoint, record.state)
        ready = self._wait_ready(options.deployment_id, record, timeout_s=options.timeout_s)
        return self._endpoint(model, options.deployment_id, ready, shape)

    def smoke_test(self, endpoint: Endpoint) -> SmokeTest:
        """One chat completion to ``endpoint``, retried while it scales up (HTTP 503, 429
        or a dropped connection). Never raises for a failed answer, only when the API key is
        rejected; an answer that retrying cannot fix (e.g. HTTP 404) fails it at once."""
        deadline = self._clock() + self._smoke_timeout_s
        attempts = 0
        with self._chat() as client:
            while True:
                attempts += 1
                started = self._clock()
                try:
                    client.complete(
                        [{"role": "user", "content": SMOKE_PROMPT}],
                        model=endpoint.inference_model,
                        temperature=0.0,
                        max_tokens=SMOKE_MAX_TOKENS,
                    )
                except ProviderAuthError:
                    raise
                except (TransientError, ProviderCapacityError) as exc:
                    detail = redact(str(exc))
                except ExternalError as exc:
                    return SmokeTest(ok=False, attempts=attempts, detail=redact(str(exc)))
                else:
                    latency = max(0.0, self._clock() - started)
                    return SmokeTest(ok=True, latency_s=round(latency, 3), attempts=attempts)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return SmokeTest(ok=False, attempts=attempts, detail=detail)
                self._sleep(min(self._poll_interval_s, remaining))

    def undeploy(self, endpoint: Endpoint) -> bool:
        """Delete the deployment; ``False`` when it was already gone. The model is kept.

        Raises :class:`UsageError` when the deployment's resource name places it in another
        account than the one the key resolves to now: a 404 there would not mean it is gone.
        """
        self._check_account(endpoint.name, endpoint.deployment_id)
        return self._deployments.delete_deployment(endpoint.deployment_id)

    def owns_deployment(self, request: DeploymentRequest) -> bool | None:
        """Whether the deployment under the requested id is the one ``request`` creates: of
        its base model, display name, shape and replica counts (as far as Fireworks reports
        them, as a create that failed checks it); ``None`` when there is none.

        Raises :class:`UsageError` when ``request.name`` places it in another account.
        """
        self._check_account(request.name, request.deployment_id)
        existing = self._deployments.get_deployment(request.deployment_id)
        if existing is None:
            return None
        create = CreateRequest(
            base_model=request.base_model,
            shape=request.shape or "",
            display_name=request.display_name or "",
            min_replica_count=request.min_replicas,
            max_replica_count=request.max_replicas,
        )
        return create.made(existing)

    # Internals.

    def _check_account(self, name: str | None, deployment_id: str) -> None:
        account = _account_of(name)
        if account is not None and account != self.account_id:
            raise UsageError(
                f"the deployment {deployment_id} was made in the Fireworks account "
                f"{account}, but this key and FIREWORKS_ACCOUNT_ID resolve to {self.account_id}",
                hint=f"set FIREWORKS_ACCOUNT_ID={account} (with a key of that account) and "
                "undeploy again",
            )

    def _endpoint(
        self, model: ModelRef, deployment_id: str, record: DeploymentRecord, shape: str
    ) -> Endpoint:
        name = record.name or self.deployment_name(deployment_id)
        return Endpoint(
            provider=PROVIDER_NAME,
            deployment_id=deployment_id,
            model_id=model.model_id,
            inference_model=name,
            url=INFERENCE_URL,
            name=name,
            shape=record.shape or shape,
            api_key_env=ENV_VAR,
        )

    def _wait_ready(
        self, deployment_id: str, record: DeploymentRecord, *, timeout_s: float
    ) -> DeploymentRecord:
        deadline = self._clock() + timeout_s
        hint = "undeploy it to stop any billing, then deploy again"
        current: DeploymentRecord | None = record
        state = record.state
        while True:
            if current is None:
                raise ExternalError(
                    f"the Fireworks deployment {deployment_id} disappeared while starting",
                    source=SOURCE,
                    hint="check the deployment in your Fireworks account, then deploy again",
                )
            state = current.state
            if state == READY:
                return current
            if current.state in BAD_STATES:
                detail = f": {current.detail}" if current.detail else ""
                raise DeploymentNotReady(
                    f"the Fireworks deployment {deployment_id} is {current.state}{detail}",
                    source=SOURCE,
                    state=current.state,
                    hint=hint,
                )
            remaining = deadline - self._clock()
            if remaining <= 0:
                minutes = timeout_s / 60
                raise DeploymentNotReady(
                    f"the Fireworks deployment {deployment_id} was not ready after "
                    f"{minutes:g} minutes (state {state})",
                    source=SOURCE,
                    state=state,
                    timed_out=True,
                    hint="it may still become ready: check it later, or " + hint,
                )
            self._sleep(min(self._poll_interval_s, remaining))
            try:
                current = self._deployments.get_deployment(deployment_id)
            except (TransientError, ProviderCapacityError) as exc:
                # A failed read is not a failed deployment: keep waiting, until the deadline.
                info(f"reading the deployment {deployment_id} failed, retrying: {exc.message}")
