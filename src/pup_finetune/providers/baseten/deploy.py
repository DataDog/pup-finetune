# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Dedicated Baseten deployments of a Loops checkpoint: create, wait, smoke-test, delete.

:func:`deploy` serves the ``bt://loops:<run>/sampler_weights/<name>`` checkpoint a
:class:`ModelRef` names (a run's deployable checkpoint: Baseten has no promotion) as a new
Baseten model named ``options.deployment_id``, through ``POST /v1/loops/checkpoints/deploy``:

1. **Lookups**, before ``on_creating``: Baseten's record of the checkpoint (a synced sampler
   checkpoint of a model the provider serves), the instance type with exactly the GPUs of the
   shape (``TYPE:COUNT``, :data:`DEFAULT_SHAPE` by default), and no model already named
   ``deployment_id``, as Baseten would add the deployment to it.
2. **Create**, sent once (it is not idempotent), reading ``HF_TOKEN`` from the workspace secret
   :data:`~.client.HF_SECRET_NAME`; the workspace needs BDN volumes. A refusal is a
   :class:`~.client.DeploymentRejected`: nothing was created. After any other failure the
   model is read back by name and adopted when it serves the checkpoint on the instance type;
   another model under the name is refused as a taken ``--deployment-id``, and with none (or
   one Baseten does not describe yet, or with several deployments) the failure is a
   :class:`TransientError`.
3. **Wait** until it is ``ACTIVE`` or scaled to zero, reading it every
   :data:`POLL_INTERVAL_S` seconds (a read that fails transiently is retried); a failed state
   or the deadline is :class:`DeploymentNotReady`. Its autoscaling is then set to the requested
   replicas, unless it already has them.

:func:`smoke_test` sends one short chat completion to the deployment's OpenAI-compatible
route, naming the checkpoint, retried while a replica starts. :func:`owns_deployment` reads a
model back by name and compares what it serves with a request, and :func:`undeploy` deletes
the model with its deployment (the checkpoint is kept), which frees the name. Nothing tears a
deployment down on its own: the caller's receipt says what to undeploy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeGuard

from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.inference import ChatCompleter
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
from pup_finetune.providers.baseten import PROVIDER_NAME
from pup_finetune.providers.baseten.client import (
    SOURCE,
    BasetenHTTP,
    CreatedDeployment,
    inference_url,
)
from pup_finetune.redaction import redact

__all__ = [
    "BAD_STATES",
    "CHECKPOINT_URI",
    "DEFAULT_SHAPE",
    "POLL_INTERVAL_S",
    "READY",
    "SMOKE_TIMEOUT_S",
    "ChatSession",
    "deploy",
    "deployment_url",
    "owns_deployment",
    "smoke_test",
    "undeploy",
]

CHECKPOINT_URI: Final = re.compile(
    r"bt://loops:([A-Za-z0-9_-]{1,128})/sampler_weights/([A-Za-z0-9_.-]+)"
)
"""A Loops sampler checkpoint: its run id and name."""
DEFAULT_SHAPE: Final = "H100:1"
POLL_INTERVAL_S: Final = 15.0
SMOKE_TIMEOUT_S: Final = 10 * 60.0
"""How long the smoke test keeps retrying while a replica starts."""
READY: Final = frozenset({"ACTIVE", "SCALED_TO_ZERO"})
BAD_STATES: Final = frozenset(
    {
        "BUILD_FAILED",
        "BUILD_STOPPED",
        "DEPLOY_FAILED",
        "FAILED",
        "UNHEALTHY",
        "DEACTIVATING",
        "INACTIVE",
    }
)
SMOKE_PROMPT: Final = "Reply with the single word OK."
SMOKE_MAX_TOKENS: Final = 16

_SHAPE: Final = re.compile(r"([A-Z][A-Z0-9_]*):([1-9][0-9]*)")
_NAME: Final = re.compile(r"models/([A-Za-z0-9_-]{1,128})/deployments/([A-Za-z0-9_-]{1,128})")
_TAKEN_HINT: Final = "pass another --deployment-id, or delete that Baseten model"


class ChatSession(ChatCompleter, AbstractContextManager["ChatSession"], Protocol):
    """A chat client that is also a context manager, closed on exit (as ``ChatClient`` is)."""


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    path: str
    run_id: str
    name: str


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def deployment_url(model_id: str, deployment_id: str) -> str:
    """The OpenAI base URL of a Baseten model deployment."""
    return f"https://model-{model_id}.api.baseten.co/deployment/{deployment_id}/sync/v1"


def _checkpoint(model: ModelRef) -> _Checkpoint:
    if model.provider != PROVIDER_NAME:
        raise UsageError(f"the baseten provider cannot deploy a {model.provider} model")
    match = CHECKPOINT_URI.fullmatch(model.uri)
    if match is None:
        raise UsageError(
            f"{model.uri!r} is not a Baseten Loops sampler checkpoint",
            hint="deploy a run's deployable_checkpoint, a "
            "bt://loops:<run>/sampler_weights/<name> path",
        )
    return _Checkpoint(match[0], match[1], match[2])


def _record_id(record: Mapping[str, Any] | None, checkpoint: _Checkpoint) -> str | None:
    """Baseten's id of ``checkpoint`` when ``record`` describes it; else ``None``."""
    if record is None:
        return None
    expected = {"run_id": checkpoint.run_id, "checkpoint_id": checkpoint.name, "target": "sampler"}
    record_id = record.get("id")
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    return record_id if isinstance(record_id, str) and record_id else None


def _checkpoint_record(http: BasetenHTTP, checkpoint: _Checkpoint, served: Collection[str]) -> str:
    """Baseten's id of the checkpoint, once Baseten confirms it, it synced and it serves."""
    record = http.get_checkpoint(checkpoint.path)
    if record is None:
        raise UsageError(
            f"Baseten has no checkpoint {checkpoint.path}",
            hint="it may have expired; train again, or deploy a checkpoint Baseten still has",
        )
    record_id = _record_id(record, checkpoint)
    if record_id is None:
        raise ExternalError(
            "Baseten's record of the checkpoint does not match its path", source=SOURCE
        )
    base_model = record.get("base_model")
    if base_model not in served:
        raise UsageError(
            f"checkpoint {checkpoint.name!r} was trained on {base_model}, which the baseten "
            "provider does not serve"
        )
    if record.get("sync_status") not in (None, "COMPLETE"):
        raise ExternalError(
            "the checkpoint is still syncing on Baseten",
            source=SOURCE,
            hint="deploy again in a few minutes",
        )
    return record_id


def _shape(shape: str | None) -> str:
    chosen = shape or DEFAULT_SHAPE
    if not _SHAPE.fullmatch(chosen):
        raise UsageError(
            f"--shape {chosen!r} is not a Baseten GPU shape",
            hint=f"pass --shape TYPE:COUNT, e.g. {DEFAULT_SHAPE}: Baseten serves a checkpoint "
            "on a dedicated deployment, which needs BDN volumes enabled for your workspace",
        )
    return chosen


def _instance_types(instances: Sequence[Mapping[str, Any]], shape: str) -> list[dict[str, str]]:
    """The instance types with exactly the GPUs of ``shape`` (never more), as id and name."""
    gpu, count = shape.split(":")
    found: list[dict[str, str]] = []
    for instance in instances:
        instance_id, name = instance.get("id"), instance.get("name")
        if (
            (instance.get("gpu_type"), instance.get("gpu_count")) == (gpu, int(count))
            and isinstance(instance_id, str)
            and instance_id
            and isinstance(name, str)
        ):
            found.append({"id": instance_id, "name": name})
    return found


def _instance_type(instances: Sequence[Mapping[str, Any]], shape: str) -> list[dict[str, str]]:
    """:func:`_instance_types`, refusing a shape Baseten does not offer."""
    found = _instance_types(instances, shape)
    if found:
        return found
    offered = sorted(
        {
            (str(item.get("gpu_type")), int(item["gpu_count"]))
            for item in instances
            if isinstance(item.get("gpu_type"), str)
            and _is_int(item.get("gpu_count"))
            and item["gpu_count"] > 0
        }
    )
    gpu = shape.split(":")[0]
    nearby = [item for item in offered if item[0] == gpu] or offered
    raise UsageError(
        f"Baseten offers no instance type with exactly {shape}",
        hint="available: " + ", ".join(f"{t}:{n}" for t, n in nearby)
        if nearby
        else "Baseten listed no GPU instance types",
    )


def _verdict(
    http: BasetenHTTP, model: Mapping[str, Any], checkpoint_id: str, instance_names: Collection[str]
) -> tuple[bool | None, CreatedDeployment | None]:
    """Whether ``model`` is the one a create of ``checkpoint_id`` on one of ``instance_names``
    made: ``(True, its ids)``, ``(False, None)`` when it serves something else, and
    ``(None, None)`` when Baseten does not describe its deployment yet or it has several (a
    create under a taken name adds one, so this create's may be among them)."""
    model_id = model.get("id")
    if not isinstance(model_id, str):
        raise ExternalError("Baseten listed a model without its id", source=SOURCE)
    deployments = http.list_deployments(model_id)
    if len(deployments) != 1:
        return None, None
    deployment_id = deployments[0].get("id")
    if not isinstance(deployment_id, str):
        raise ExternalError("Baseten listed a deployment without its id", source=SOURCE)
    instance = deployments[0].get("instance_type_name")
    if isinstance(instance, str) and instance not in instance_names:
        return False, None
    checkpoints = http.deployment_config(model_id, deployment_id).get("training_checkpoints")
    served = checkpoints.get("loops_checkpoint_ids") if isinstance(checkpoints, dict) else None
    if served is None:
        return None, None
    if served != [checkpoint_id]:
        return False, None
    return True, CreatedDeployment(model_id=model_id, deployment_id=deployment_id)


def _create(
    http: BasetenHTTP,
    name: str,
    checkpoint_id: str,
    instance_types: Sequence[Mapping[str, str]],
) -> CreatedDeployment:
    """Send the create once; after a failure that may have created it, read it back by name."""
    try:
        return http.deploy_checkpoints(
            checkpoint_id=checkpoint_id, model_name=name, instance_type_id=instance_types[0]["id"]
        )
    except TransientError as error:
        try:
            found = http.list_models(name=name)
            verdict, created = (
                _verdict(http, found[0], checkpoint_id, {t["name"] for t in instance_types})
                if len(found) == 1
                else (None, None)
            )
        except FinetuneError as read_error:
            raise error from read_error
        if verdict is False:
            raise UsageError(
                f"Baseten already has a model named {name} that this deploy did not create",
                hint=_TAKEN_HINT,
            ) from error
        if created is None:
            raise
        info(
            f"creating {name} reported a failure ({error.message}), but the Baseten model "
            "serving the checkpoint exists: using it"
        )
        return created


def _status(record: Mapping[str, Any], created: CreatedDeployment) -> str:
    status = record.get("status")
    if (
        record.get("id") != created.deployment_id
        or record.get("model_id") != created.model_id
        or not isinstance(status, str)
    ):
        raise ExternalError("Baseten described a different deployment", source=SOURCE)
    return status


def _wait_ready(
    http: BasetenHTTP,
    created: CreatedDeployment,
    *,
    timeout_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_interval_s: float,
) -> Mapping[str, Any]:
    deadline = clock() + timeout_s
    what = f"the Baseten deployment {created.deployment_id} of model {created.model_id}"
    hint = "undeploy it to stop any billing, then deploy again"
    status = "CREATING"
    while True:
        try:
            record = http.get_deployment(created.model_id, created.deployment_id)
        except (TransientError, ProviderCapacityError) as exc:
            # A failed read is not a failed deployment: keep waiting, until the deadline.
            info(f"reading {what} failed, retrying: {exc.message}")
        else:
            if record is None:
                raise ExternalError(
                    f"{what} disappeared while starting",
                    source=SOURCE,
                    hint="check the model in the Baseten UI, then deploy again",
                )
            status = _status(record, created)
            if status in READY:
                return record
            if status in BAD_STATES:
                raise DeploymentNotReady(
                    f"{what} is {status}", source=SOURCE, state=status, hint=hint
                )
        remaining = deadline - clock()
        if remaining <= 0:
            raise DeploymentNotReady(
                f"{what} was not ready after {timeout_s / 60:g} minutes (status {status})",
                source=SOURCE,
                state=status,
                timed_out=True,
                hint="it may still become ready: check it later, or " + hint,
            )
        sleep(min(poll_interval_s, remaining))


def _scale(
    http: BasetenHTTP, created: CreatedDeployment, record: Mapping[str, Any], options: DeployOptions
) -> None:
    """Give the deployment the requested replicas, unless Baseten reports it has them."""
    settings = record.get("autoscaling_settings")
    current = (
        (settings.get("min_replica"), settings.get("max_replica"))
        if isinstance(settings, dict)
        else None
    )
    if current == (options.min_replicas, options.max_replicas):
        return
    info(f"setting the deployment to {options.min_replicas} to {options.max_replicas} replicas")
    http.set_autoscaling(
        created.model_id,
        created.deployment_id,
        min_replica=options.min_replicas,
        max_replica=options.max_replicas,
    )


def deploy(
    http: BasetenHTTP,
    model: ModelRef,
    options: DeployOptions,
    *,
    served: Collection[str],
    on_creating: Callable[[DeploymentRequest], None] | None = None,
    on_created: Callable[[Endpoint, str], None] | None = None,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_interval_s: float = POLL_INTERVAL_S,
) -> Endpoint:
    """Create the deployment of ``model`` and wait until it serves; see the module docstring.

    ``served`` are the base models the provider serves. Raises :class:`UsageError` before
    anything is paid for a foreign or malformed ``model``, a checkpoint of another model, a
    shape Baseten does not offer or a taken name.
    """
    checkpoint = _checkpoint(model)
    shape = _shape(options.shape)
    checkpoint_id = _checkpoint_record(http, checkpoint, served)
    instance_types = _instance_type(http.instance_types(), shape)
    name = options.deployment_id
    if http.list_models(name=name):
        raise UsageError(f"Baseten already has a model named {name}", hint=_TAKEN_HINT)
    if on_creating is not None:
        on_creating(
            DeploymentRequest(
                deployment_id=name,
                base_model=model.uri,
                name=None,
                display_name=options.display_name or name,
                shape=shape,
                min_replicas=options.min_replicas,
                max_replicas=options.max_replicas,
            )
        )
    info(f"creating the Baseten model {name}, serving {checkpoint.name} on {shape}")
    created = _create(http, name, checkpoint_id, instance_types)
    endpoint = Endpoint(
        provider=PROVIDER_NAME,
        deployment_id=name,
        model_id=model.model_id,
        inference_model=checkpoint.name,
        url=deployment_url(created.model_id, created.deployment_id),
        name=f"models/{created.model_id}/deployments/{created.deployment_id}",
        shape=shape,
    )
    if on_created is not None:
        on_created(endpoint, "CREATING")
    record = _wait_ready(
        http,
        created,
        timeout_s=options.timeout_s,
        clock=clock,
        sleep=sleep,
        poll_interval_s=poll_interval_s,
    )
    _scale(http, created, record, options)
    return endpoint


def smoke_test(
    endpoint: Endpoint,
    *,
    chat: Callable[[str], ChatSession],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_interval_s: float = POLL_INTERVAL_S,
    timeout_s: float = SMOKE_TIMEOUT_S,
) -> SmokeTest:
    """One chat completion to ``endpoint``, retried while it scales up (HTTP 5xx, 429 or a
    dropped connection). Never raises for a failed answer, only when the API key is rejected;
    an answer that retrying cannot fix (e.g. HTTP 404) fails it at once. ``chat`` opens a
    client of a base URL."""
    try:
        url = inference_url(endpoint.url or "")
    except ExternalError as exc:
        return SmokeTest(ok=False, detail=redact(str(exc)))
    deadline = clock() + timeout_s
    attempts = 0
    with chat(url) as client:
        while True:
            attempts += 1
            started = clock()
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
                latency = max(0.0, clock() - started)
                return SmokeTest(ok=True, latency_s=round(latency, 3), attempts=attempts)
            remaining = deadline - clock()
            if remaining <= 0:
                return SmokeTest(ok=False, attempts=attempts, detail=detail)
            sleep(min(poll_interval_s, remaining))


def _named_model(http: BasetenHTTP, name: str) -> dict[str, Any] | None:
    """The one model named ``name``, ``None`` when there is none; refuses several."""
    models = http.list_models(name=name)
    if len(models) > 1:
        raise UsageError(
            f"Baseten has {len(models)} models named {name}: nothing was deleted",
            hint="delete the one this run created in the Baseten UI",
        )
    return models[0] if models else None


def owns_deployment(http: BasetenHTTP, request: DeploymentRequest) -> bool | None:
    """Whether the model named ``request.deployment_id`` is the one ``request`` creates: it
    serves the requested checkpoint, alone, on an instance type of the requested shape;
    ``None`` when Baseten has no model of that name.

    Raises :class:`UsageError` when several models have the name, and
    :class:`TransientError` when Baseten does not show what the model serves: either may be
    this run's.
    """
    model = _named_model(http, request.deployment_id)
    if model is None:
        return None
    match = CHECKPOINT_URI.fullmatch(request.base_model)
    if match is None or request.shape is None:
        return False
    checkpoint = _Checkpoint(match[0], match[1], match[2])
    checkpoint_id = _record_id(http.get_checkpoint(checkpoint.path), checkpoint)
    if checkpoint_id is None or not _SHAPE.fullmatch(request.shape):
        return False
    names = {t["name"] for t in _instance_types(http.instance_types(), request.shape)}
    verdict, _ = _verdict(http, model, checkpoint_id, names)
    if verdict is None:
        raise TransientError(
            f"cannot tell whether the Baseten model {request.deployment_id} is the one this "
            "run requested: nothing was deleted",
            source=SOURCE,
            hint="Baseten may not describe it yet: undeploy again in a few minutes, or check "
            "the model in the Baseten UI",
        )
    return verdict


def undeploy(http: BasetenHTTP, endpoint: Endpoint) -> bool:
    """Delete the endpoint's Baseten model and its deployment; ``False`` when it was already
    gone. The checkpoint is kept.

    Raises :class:`UsageError` when the model the endpoint names is no longer named after the
    deployment id, or several models are: nothing is deleted then.
    """
    named = _NAME.fullmatch(endpoint.name or "")
    if named is not None:
        model = http.get_model(named[1])
        if model is None:
            return False
        if model.get("name") != endpoint.deployment_id:
            raise UsageError(
                f"the Baseten model {named[1]} is not named {endpoint.deployment_id}: nothing "
                "was deleted",
                hint="check the model in the Baseten UI",
            )
        return http.delete_model(named[1])
    if endpoint.name is not None:
        raise UsageError(f"{endpoint.name!r} is not a Baseten deployment name")
    model = _named_model(http, endpoint.deployment_id)
    if model is None:
        return False
    model_id = model.get("id")
    if not isinstance(model_id, str):
        raise ExternalError("Baseten listed a model without its id", source=SOURCE)
    return http.delete_model(model_id)
