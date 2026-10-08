# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune deploy` and `undeploy`: serve a run's model on its provider, and stop.

Deploying creates paid, billing infrastructure, so without ``--confirm`` both commands only
preview. The previews are built from the run directory alone, so they run with
``PUP_READ_ONLY`` set; only ``--confirm`` is refused then. ``--confirm`` is never implied by
``PUP_AUTO_APPROVE`` or agent mode.

Fireworks' own command-line tool refuses to create deployments when an AI agent runs it.
pup-finetune creates them through the Fireworks API instead and relies on ``--confirm``, so
a coding agent must pass ``--confirm`` to ``deploy`` only with the user's explicit
authorization for that deployment.

The flow and the ``deployment.json`` receipt live in :mod:`pup_finetune.deployment`.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import (
    CommandResult,
    current_context,
    get_runtime,
    run_command,
    with_settings,
)
from pup_finetune.cli._dirs import RUN_DIR_BY_ID, resolve_run_ref
from pup_finetune.cli._flags import given
from pup_finetune.cli._presentation import (
    EVALUATE_AND_SHIP,
    CommandDoc,
    Safety,
    default_note,
    describe,
)
from pup_finetune.deployment import (
    BILLING,
    deploy_billing,
    deploy_preview,
    deploy_target,
    deployment_id_for,
    deployment_usage,
    execute_deploy,
    execute_undeploy,
    require_not_deployed,
    undeploy_command,
    undeploy_preview,
)
from pup_finetune.display import display_path
from pup_finetune.errors import (
    FinetuneError,
    UsageError,
    WrongPhase,
    require_confirm,
    require_writable,
)
from pup_finetune.next_steps import PROGRAM, command, step
from pup_finetune.providers import (
    get_provider,
    provider_info,
    require_deploys,
    require_packages,
)
from pup_finetune.providers.base import DeployOptions
from pup_finetune.runs import RunDir

DEFAULT_TIMEOUT_MINUTES = 30.0

_AGENT_RULE = (
    "Fireworks' own CLI refuses to create deployments when run by an AI agent; this command "
    "uses the API and relies on --confirm, so an agent must pass --confirm only with the "
    "user's explicit authorization."
)

RunDirArgument = Annotated[
    Path,
    typer.Argument(
        metavar="RUN_DIR", help="A completed run: its directory or run id.", show_default=False
    ),
]

DEPLOY_DOC = CommandDoc(
    path="deploy",
    short="Deploy a run's model (paid; previews without --confirm).",
    description=describe(
        "Create an on-demand deployment of a completed run's model on its provider, wait "
        "until it serves, and smoke-test it with one chat completion. Without --confirm, "
        "previews what would be created; once created, it bills until `undeploy RUN_DIR "
        "--confirm`. An agent must pass --confirm only with the user's explicit "
        "authorization.",
        "The deployment has at most one replica and scales to zero when idle. The receipt, "
        "deployment.json in RUN_DIR, is written as soon as the deployment exists, so a "
        "failed or timed-out wait (--timeout-minutes) still records it. The result says how "
        "to call the model (usage: the base URL, the model to request and an example request "
        "that reads the key from its environment variable) and how to delete it "
        f"(undeploy_command). {BILLING[0].upper()}{BILLING[1:]}. A failed smoke test exits 1 "
        f"and keeps the deployment. {_AGENT_RULE}",
    ),
    examples=("pup finetune deploy run", "pup finetune deploy run --timeout-minutes 45"),
    safety=Safety(
        data="creates a deployment that bills until undeploy",
        recipients="the provider: a deployment of the model",
        without="previews the deployment",
        confirm="creates it (billed while it runs)",
        paid=True,
        read_only="confirm_refused",
    ),
    panel=EVALUATE_AND_SHIP,
)
UNDEPLOY_DOC = CommandDoc(
    path="undeploy",
    short="Delete a run's deployment (previews without --confirm).",
    description=(
        "Delete the deployment recorded in RUN_DIR/deployment.json, so it stops billing, and "
        "record when; the model is kept and can be deployed again. Deleting a deployment that "
        "is already gone succeeds. Without --confirm, previews what would be deleted."
    ),
    examples=("pup finetune undeploy run", "pup finetune undeploy ft-20261007183443-a8b181"),
    safety=Safety(
        data="--confirm deletes the deployment",
        recipients="the provider: the delete request",
        without="previews the delete",
        confirm="deletes the deployment",
        read_only="confirm_refused",
    ),
    panel=EVALUATE_AND_SHIP,
)


@run_command
def deploy(
    run_dir: RunDirArgument,
    deployment_id: Annotated[
        str | None,
        typer.Option(
            "--deployment-id",
            metavar="ID",
            help=f"The deployment's id. {default_note('from the run id')}",
            show_default=False,
        ),
    ] = None,
    shape: Annotated[
        str | None,
        typer.Option(
            "--shape",
            metavar="SHAPE",
            help="The deployment shape. " + default_note("the provider's default for the model"),
            show_default=False,
        ),
    ] = None,
    timeout_minutes: Annotated[
        float,
        typer.Option(min=1, metavar="X", help="Minutes to wait for it to serve."),
    ] = DEFAULT_TIMEOUT_MINUTES,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Create it (paid); without it, only preview."),
    ] = False,
) -> dict[str, Any] | CommandResult:
    runtime = get_runtime()
    run_dir, source = resolve_run_ref(run_dir, "deploy")
    by_id = {"run_dir_source": source} if source == RUN_DIR_BY_ID else {}
    run = RunDir(run_dir)
    target = deploy_target(run)
    require_not_deployed(run)
    # What the preview can know fails here, not after --confirm.
    require_deploys(target.provider, runtime)
    require_packages(target.provider, runtime, "deploy")
    offline = provider_info(target.provider, runtime).offline
    try:
        options = DeployOptions(
            deployment_id=deployment_id_for(target.run_id, deployment_id),
            shape=shape or None,
            timeout_s=timeout_minutes * 60,
            display_name=f"pup-finetune {target.run_id}",
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    rerun = [*PROGRAM, "deploy", str(run_dir)]
    if deployment_id is not None:
        rerun += ["--deployment-id", deployment_id]
    if shape:
        rerun += ["--shape", shape]
    if timeout_minutes != DEFAULT_TIMEOUT_MINUTES:
        rerun += ["--timeout-minutes", f"{timeout_minutes:g}"]
    if confirm:
        require_writable(runtime, "deploy")
    settings = {
        "deployment_id": options.deployment_id,
        "shape": options.shape,
        "shape_source": "--shape" if options.shape else "chosen by the provider",
        "min_replicas": options.min_replicas,
        "max_replicas": options.max_replicas,
        "timeout_minutes": timeout_minutes,
    }
    ctx = current_context()
    defaults_used = [
        name
        for name in ("deployment_id", "shape", "timeout_minutes")
        if not (ctx is not None and given(ctx, name))
    ]
    # The preview is built from the run alone (no provider call), so it is allowed under
    # PUP_READ_ONLY.
    require_confirm(
        confirm,
        {
            **with_settings(
                deploy_preview(target, options, offline=offline), settings, defaults_used
            ),
            **by_id,
        },
        shlex.join([*rerun, "--confirm"]),
        read_only=runtime.read_only,
    )
    provider = get_provider(target.provider, runtime)
    receipt, smoke_ok = execute_deploy(provider, target, options)
    # Deleting what was just created is never the next command: it is a field of its own.
    usage = deployment_usage(receipt)
    data: dict[str, Any] = {
        **_deploy_shown(receipt, usage),
        "run_dir": display_path(run.path),
        **by_id,
        "receipt_file": display_path(run.deployment_path),
        "billing": deploy_billing(offline=offline),
        "usage": usage,
        "undeploy_command": undeploy_command(run),
        "next_command": None,
    }
    if not smoke_ok:
        data["hint"] = (
            "the deployment did not answer the smoke test; check it, or delete it so it stops "
            f"billing: {undeploy_command(run)}"
        )
        return CommandResult(data, exit_code=1)
    return data


RECEIPT_ONLY: tuple[str, ...] = ("request", "deleted_at", "model_id")
"""What ``deployment.json`` keeps that the ``deploy`` result leaves out: the create request
(its deployment id, model and shape are in the result already), a ``deleted_at`` that is
always null after a deploy, and the model id, which ``model`` names."""

USAGE_FIELDS: tuple[str, ...] = ("url", "inference_model", "api_key_env")
"""The receipt's fields ``usage`` repeats (as ``base_url``, ``model`` and ``api_key_env``),
left out of the result when it has a ``usage``."""


def _deploy_shown(receipt: Mapping[str, Any], usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """The receipt as the ``deploy`` result shows it, each fact once."""
    hidden = {*RECEIPT_ONLY, *(USAGE_FIELDS if usage is not None else ())}
    return {key: value for key, value in receipt.items() if key not in hidden}


def _no_deployment_hint(run: RunDir) -> str:
    """What to do about a run with nothing to undeploy: deploy it when it can be deployed,
    else see where it stands (``deploy`` would refuse a run that has not finished)."""
    try:
        deploy_target(run)
    except FinetuneError:
        return (
            "nothing to undeploy, and the run cannot be deployed yet; see "
            f"`{command('status', run.path)}`"
        )
    return f"nothing to undeploy; deploy it with `{command('deploy', run.path)}`"


@run_command
def undeploy(
    run_dir: RunDirArgument,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Delete it; without it, only preview."),
    ] = False,
) -> dict[str, Any]:
    runtime = get_runtime()
    run_dir, source = resolve_run_ref(run_dir, "undeploy")
    by_id = {"run_dir_source": source} if source == RUN_DIR_BY_ID else {}
    run = RunDir(run_dir)
    receipt = run.read_deployment()
    if receipt is None:
        raise WrongPhase(
            f"the run in {display_path(run.path)} has no deployment "
            f"({display_path(run.deployment_path)} does not exist)",
            hint=_no_deployment_hint(run),
        )
    deleted_before = receipt.get("deleted_at")
    if not deleted_before:
        # What the preview can know fails here, as for deploy: the delete goes through the
        # same packages.
        require_packages(str(receipt.get("provider")), runtime, "deploy")
        if confirm:
            require_writable(runtime, "undeploy")
        require_confirm(
            confirm,
            {**undeploy_preview(run, receipt), **by_id},
            command("undeploy", run_dir, "--confirm"),
            read_only=runtime.read_only,
        )
        provider = get_provider(str(receipt.get("provider")), runtime)
        receipt = execute_undeploy(provider, run)
    data: dict[str, Any] = {
        "deployment_id": receipt.get("deployment_id"),
        "state": receipt.get("state"),
        "deleted_at": receipt.get("deleted_at"),
        # True when nothing was left to delete: deleted by an earlier undeploy, or gone at
        # the provider already.
        "already_deleted": bool(deleted_before) or bool(receipt.get("already_deleted")),
        "model_kept": receipt.get("model"),
        "receipt_file": display_path(run.deployment_path),
        **by_id,
    }
    if deleted_before:
        data["message"] = f"already deleted at {deleted_before}; nothing to do"
    return {
        **data,
        "next_command": None,
        "next_steps": [
            step(command("deploy", run_dir), "deploy the kept model again (it previews first)")
        ],
    }


def register(app: typer.Typer) -> None:
    """Add `deploy` and `undeploy` to ``app``."""
    app.command(**DEPLOY_DOC.typer())(deploy)
    app.command(**UNDEPLOY_DOC.typer())(undeploy)
