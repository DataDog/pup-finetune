# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Deploying a run's model and tearing the deployment down: what ``deploy`` and ``undeploy`` run.

:func:`deploy_target` reads what a finished run can deploy: its promoted model, else the
provider's deployable checkpoint. :func:`execute_deploy` deploys it through the run's
:class:`~pup_finetune.providers.base.Provider` and records the receipt, ``deployment.json``
in the run directory:

- before the provider is asked to create anything (``state`` ``REQUESTED``), so a Ctrl-C or
  a lost answer while it is created still leaves the id to undeploy, with the resource name
  the deployment gets (:meth:`~pup_finetune.providers.base.Provider.deployment_name`, e.g.
  ``accounts/<account>/deployments/<id>``), so undeploying it under another account is
  refused rather than deleting, or taking as gone, a deployment of the same id there; it is
  removed again when the provider refused the deployment, so nothing exists;
- as soon as the provider accepted the deployment, before waiting for it to serve, so a
  failure or timeout while waiting (or a Ctrl-C) still leaves a record of what exists and
  bills;
- once it serves (``state`` ``READY``), and again with the smoke test's result (a smoke test
  that raises is recorded as failed, with its error).

:func:`execute_undeploy` deletes the deployment and records ``deleted_at``; the promoted
model is kept. Deleting a deployment that is already gone succeeds.

A run has one deployment at a time: deploying a run whose receipt has no ``deleted_at`` is
refused until it is undeployed. Both hold the run directory's lock.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from pup_finetune.errors import FinetuneError, StateError, UsageError
from pup_finetune.output import info
from pup_finetune.providers.base import (
    MAX_MODEL_ID_LEN,
    MODEL_ID_PATTERN,
    DeploymentNotReady,
    DeployOptions,
    Endpoint,
    ModelRef,
    Provider,
    SmokeTest,
    TransientError,
)
from pup_finetune.redaction import redact
from pup_finetune.runs import Clock, RunDir, error_json, model_id_for_run, utcnow

__all__ = [
    "BILLING",
    "DEPLOYABLE_PHASES",
    "REQUESTED",
    "DeployTarget",
    "deploy_preview",
    "deploy_target",
    "deployment_id_for",
    "endpoint_from_receipt",
    "execute_deploy",
    "execute_undeploy",
    "require_not_deployed",
    "undeploy_command",
    "undeploy_preview",
]

DEPLOYABLE_PHASES: Final = frozenset({"completed", "cancelled"})
"""The run phases with a final model: completed, or cancelled (which promotes the best epoch)."""

REQUESTED: Final = "REQUESTED"
"""The receipt's ``state`` while the provider is asked to create the deployment."""

BILLING: Final = (
    "the deployment bills per GPU-second while a replica runs (it scales to zero after a "
    "few idle minutes and starts again on the next request) until it is undeployed"
)


def _q(value: object) -> str:
    return shlex.quote(str(value))


def undeploy_command(run_dir: RunDir) -> str:
    """The exact command that deletes the run's deployment."""
    return f"pup finetune undeploy {_q(run_dir.path)} --confirm"


@dataclass(frozen=True, slots=True, kw_only=True)
class DeployTarget:
    """What a run deploys: ``model``, on the run's ``provider``."""

    run_dir: RunDir
    run_id: str
    provider: str
    model: ModelRef
    source: str
    """``promoted_model`` or ``deployable_checkpoint``: where in ``result.json`` it came from."""


def deploy_target(run_dir: RunDir) -> DeployTarget:
    """The model a finished run deploys: its promoted model, else its deployable checkpoint.

    Raises :class:`StateError` when the run has not finished, or ended without either.
    """
    manifest = run_dir.manifest()
    run_id = str(manifest["run_id"])
    provider = str(manifest["provider"])
    state = run_dir.load_state()
    result = run_dir.read_result()
    status = f"pup finetune status {_q(run_dir.path)}"
    if state.phase not in DEPLOYABLE_PHASES or result is None:
        raise StateError(
            f"run {run_id} is {state.phase}, not completed: there is no final model to deploy",
            hint=f"check it with `{status}`; deploy once it has completed",
        )
    promoted = result.get("promoted_model")
    if isinstance(promoted, Mapping):
        return DeployTarget(
            run_dir=run_dir,
            run_id=run_id,
            provider=provider,
            model=_model_ref(promoted, provider, run_dir),
            source="promoted_model",
        )
    checkpoint = result.get("deployable_checkpoint")
    if isinstance(checkpoint, Mapping) and checkpoint.get("uri") and checkpoint.get("name"):
        return DeployTarget(
            run_dir=run_dir,
            run_id=run_id,
            provider=provider,
            model=ModelRef(
                provider=provider, model_id=str(checkpoint["name"]), uri=str(checkpoint["uri"])
            ),
            source="deployable_checkpoint",
        )
    raise StateError(
        f"run {run_id} has neither a promoted model nor a deployable checkpoint",
        hint=f"see `{status}`: the promotion may have failed; resume the run to retry it",
    )


def _model_ref(data: Mapping[str, Any], provider: str, run_dir: RunDir) -> ModelRef:
    try:
        model = ModelRef(
            provider=str(data["provider"]), model_id=str(data["model_id"]), uri=str(data["uri"])
        )
    except KeyError as exc:
        raise StateError(f"{run_dir.result_path}: promoted_model lacks {exc}") from exc
    if model.provider != provider:
        raise StateError(
            f"{run_dir.result_path}: the promoted model is a {model.provider} model, but the "
            f"run trained on {provider}"
        )
    return model


def deployment_id_for(run_id: str, override: str | None) -> str:
    """``override`` when given (it must be a valid id), else one derived from the run id."""
    if override is None:
        return model_id_for_run(run_id)
    if not MODEL_ID_PATTERN.fullmatch(override):
        raise UsageError(
            f"--deployment-id {override!r} is not a valid deployment id",
            hint="use lowercase letters, digits and dashes, starting with a letter and not "
            f"ending with a dash, at most {MAX_MODEL_ID_LEN} characters",
        )
    return override


def require_not_deployed(run_dir: RunDir) -> None:
    """Raise :class:`StateError` when the run has a deployment that was not undeployed."""
    receipt = run_dir.read_deployment()
    if receipt is None or receipt.get("deleted_at"):
        return
    raise StateError(
        f"run {run_dir.path} already has the deployment {receipt.get('deployment_id')} "
        f"(state {receipt.get('state')}, see {run_dir.deployment_path})",
        hint=f"undeploy it first: {undeploy_command(run_dir)}",
    )


def deploy_preview(target: DeployTarget, options: DeployOptions) -> dict[str, Any]:
    """What ``deploy --confirm`` would create, and how to tear it down."""
    minutes = options.timeout_s / 60
    return {
        "run_id": target.run_id,
        "run_dir": str(target.run_dir.path),
        "provider": target.provider,
        "model": target.model.uri,
        "model_source": target.source,
        "deployment_id": options.deployment_id,
        "shape": options.shape or "the first validated shape the provider matches to the model",
        "replicas": {"min": options.min_replicas, "max": options.max_replicas},
        "action": (
            f"create an on-demand deployment of {target.model.uri} on {target.provider}, wait "
            f"up to {minutes:g} minutes for it to serve, then send one chat completion as a "
            "smoke test"
        ),
        "billing": f"{BILLING}: run `{undeploy_command(target.run_dir)}`",
        "undeploy_command": undeploy_command(target.run_dir),
        "receipt_file": str(target.run_dir.deployment_path),
    }


def _receipt(target: DeployTarget, endpoint: Endpoint, state: str, now: Clock) -> dict[str, Any]:
    return {
        "provider": endpoint.provider,
        "run_id": target.run_id,
        "deployment_id": endpoint.deployment_id,
        "deployment_name": endpoint.name,
        "shape": endpoint.shape,
        "model": target.model.uri,
        "model_id": endpoint.model_id,
        "inference_model": endpoint.inference_model,
        "url": endpoint.url,
        "created_at": now().isoformat(timespec="seconds"),
        "state": state,
        "ready_at": None,
        "smoke_test": None,
        "error": None,
        "deleted_at": None,
    }


def _requested_receipt(
    target: DeployTarget, options: DeployOptions, name: str | None, now: Clock
) -> dict[str, Any]:
    """The receipt written before the provider is asked to create the deployment ``name``."""
    endpoint = Endpoint(
        provider=target.provider,
        deployment_id=options.deployment_id,
        model_id=target.model.model_id,
        inference_model="",
        name=name,
        shape=options.shape,
    )
    receipt = _receipt(target, endpoint, REQUESTED, now)
    receipt["inference_model"] = None
    return receipt


def endpoint_from_receipt(receipt: Mapping[str, Any]) -> Endpoint:
    """The endpoint a ``deployment.json`` receipt records."""
    try:
        return Endpoint(
            provider=str(receipt["provider"]),
            deployment_id=str(receipt["deployment_id"]),
            model_id=str(receipt["model_id"]),
            inference_model=str(receipt["inference_model"] or ""),
            url=receipt.get("url"),
            name=receipt.get("deployment_name"),
            shape=receipt.get("shape"),
        )
    except KeyError as exc:
        raise StateError(f"the deployment receipt lacks {exc}") from exc


def execute_deploy(
    provider: Provider,
    target: DeployTarget,
    options: DeployOptions,
    *,
    now: Clock = utcnow,
) -> tuple[dict[str, Any], bool]:
    """Deploy ``target``, smoke-test it and record the receipt; the receipt and whether the
    smoke test passed.

    A failure once the deployment may exist is recorded in the receipt (its ``state`` and
    ``error``) and raised with a hint to undeploy. A failure that shows nothing was created
    (the provider refused the request) restores the receipt that was there before, if any.
    """
    run_dir = target.run_dir
    with run_dir.lock():
        require_not_deployed(run_dir)
        previous = run_dir.read_deployment()
        name = provider.deployment_name(options.deployment_id)
        receipt = _requested_receipt(target, options, name, now)
        run_dir.write_deployment(receipt)
        created = False

        def on_created(endpoint: Endpoint, state: str) -> None:
            nonlocal created
            created = True
            receipt.update(_receipt(target, endpoint, state, now))
            run_dir.write_deployment(receipt)
            info(
                f"created deployment {endpoint.deployment_id} ({state}); waiting for it to "
                f"serve (receipt: {run_dir.deployment_path})"
            )

        try:
            endpoint = provider.deploy(target.model, options, on_created=on_created)
        except FinetuneError as exc:
            if not created and not isinstance(exc, TransientError):
                _restore(run_dir, previous)  # refused: nothing was created
                raise
            if isinstance(exc, DeploymentNotReady) and exc.state:
                receipt["state"] = exc.state
            _record_failure(run_dir, receipt, exc)
            raise
        if not created:  # a provider that did not report the creation
            receipt.update(_receipt(target, endpoint, "READY", now))
        receipt.update(
            {
                "deployment_name": endpoint.name,
                "shape": endpoint.shape,
                "inference_model": endpoint.inference_model,
                "url": endpoint.url,
                "state": "READY",
                "ready_at": now().isoformat(timespec="seconds"),
            }
        )
        run_dir.write_deployment(receipt)
        info(f"deployment {endpoint.deployment_id} is ready; sending a smoke test")
        try:
            smoke = provider.smoke_test(endpoint)
        except FinetuneError as exc:
            receipt["smoke_test"] = SmokeTest(ok=False, detail=redact(exc.message)).to_json()
            _record_failure(run_dir, receipt, exc)
            raise
        receipt["smoke_test"] = smoke.to_json()
        run_dir.write_deployment(receipt)
    return receipt, smoke.ok


def _record_failure(run_dir: RunDir, receipt: dict[str, Any], exc: FinetuneError) -> None:
    """Record ``exc`` in the receipt and add the undeploy hint to it."""
    receipt["error"] = error_json(exc)
    run_dir.write_deployment(receipt)
    exc.hint = (
        f"{exc.hint}; " if exc.hint else ""
    ) + f"the deployment may still exist and bill: {undeploy_command(run_dir)}"


def _restore(run_dir: RunDir, previous: Mapping[str, Any] | None) -> None:
    if previous is None:
        run_dir.deployment_path.unlink(missing_ok=True)
    else:
        run_dir.write_deployment(previous)


def undeploy_preview(run_dir: RunDir, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """What ``undeploy --confirm`` would delete."""
    return {
        "run_dir": str(run_dir.path),
        "provider": receipt.get("provider"),
        "deployment_id": receipt.get("deployment_id"),
        "deployment_name": receipt.get("deployment_name"),
        "state": receipt.get("state"),
        "action": f"delete the deployment {receipt.get('deployment_id')}; the model "
        f"{receipt.get('model')} is kept and can be deployed again",
    }


def execute_undeploy(provider: Provider, run_dir: RunDir, *, now: Clock = utcnow) -> dict[str, Any]:
    """Delete the run's deployment and record ``deleted_at``; the updated receipt.

    ``already_deleted`` in it is true when the provider no longer had the deployment.
    """
    with run_dir.lock():
        receipt = run_dir.read_deployment()
        if receipt is None:
            raise StateError(
                f"run {run_dir.path} has no deployment ({run_dir.deployment_path} does not exist)",
                hint=f"deploy it with `pup finetune deploy {_q(run_dir.path)}`",
            )
        if receipt.get("deleted_at"):
            return receipt
        deleted = provider.undeploy(endpoint_from_receipt(receipt))
        receipt.update(
            {
                "state": "DELETED",
                "deleted_at": now().isoformat(timespec="seconds"),
                "already_deleted": not deleted,
            }
        )
        run_dir.write_deployment(receipt)
    return receipt
