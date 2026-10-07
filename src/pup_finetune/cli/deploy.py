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
from pathlib import Path
from typing import Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, get_runtime, run_command
from pup_finetune.cli._dirs import require_run_dir
from pup_finetune.deployment import (
    BILLING,
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
from pup_finetune.errors import StateError, UsageError, require_confirm, require_writable
from pup_finetune.next_steps import PROGRAM, command, step
from pup_finetune.providers import get_provider
from pup_finetune.providers.base import DeployOptions
from pup_finetune.runs import RunDir

DEFAULT_TIMEOUT_MINUTES = 30.0

_AGENT_RULE = (
    "Fireworks' own CLI refuses to create deployments when run by an AI agent; this command "
    "uses the API and relies on --confirm, so an agent must pass --confirm only with the "
    "user's explicit authorization."
)

RunDirArgument = Annotated[
    Path, typer.Argument(metavar="RUN_DIR", help="A completed run directory.", show_default=False)
]


@run_command
def deploy(
    run_dir: RunDirArgument,
    deployment_id: Annotated[
        str | None,
        typer.Option(
            "--deployment-id",
            help="The deployment's id (default: derived from the run id).",
            show_default=False,
        ),
    ] = None,
    shape: Annotated[
        str | None,
        typer.Option(
            "--shape",
            help="The deployment shape (default: the first validated shape the provider "
            "matches to the model).",
            show_default=False,
        ),
    ] = None,
    timeout_minutes: Annotated[
        float,
        typer.Option(min=1, help="How long to wait for the deployment to serve."),
    ] = DEFAULT_TIMEOUT_MINUTES,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Create it (paid); without it, only preview."),
    ] = False,
) -> dict[str, Any] | CommandResult:
    runtime = get_runtime()
    require_run_dir(run_dir, "deploy")
    run = RunDir(run_dir)
    target = deploy_target(run)
    require_not_deployed(run)
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
    # The preview is built from the run alone (no provider call), so it is allowed under
    # PUP_READ_ONLY.
    require_confirm(
        confirm,
        deploy_preview(target, options),
        shlex.join([*rerun, "--confirm"]),
        read_only=runtime.read_only,
    )
    provider = get_provider(target.provider, runtime)
    receipt, smoke_ok = execute_deploy(provider, target, options)
    # Deleting what was just created is never the next command: it is a field of its own.
    data: dict[str, Any] = {
        **receipt,
        "run_dir": str(run.path),
        "receipt_file": str(run.deployment_path),
        "billing": BILLING,
        "usage": deployment_usage(receipt),
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


@run_command
def undeploy(
    run_dir: RunDirArgument,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Delete it; without it, only preview."),
    ] = False,
) -> dict[str, Any]:
    runtime = get_runtime()
    require_run_dir(run_dir, "undeploy")
    run = RunDir(run_dir)
    receipt = run.read_deployment()
    if receipt is None:
        raise StateError(
            f"run {run.path} has no deployment ({run.deployment_path} does not exist)",
            hint=f"deploy it with `pup finetune deploy {shlex.quote(str(run_dir))}`",
        )
    if not receipt.get("deleted_at"):
        if confirm:
            require_writable(runtime, "undeploy")
        require_confirm(
            confirm,
            undeploy_preview(run, receipt),
            command("undeploy", run_dir, "--confirm"),
            read_only=runtime.read_only,
        )
        provider = get_provider(str(receipt.get("provider")), runtime)
        receipt = execute_undeploy(provider, run)
    return {
        **receipt,
        "run_dir": str(run.path),
        "receipt_file": str(run.deployment_path),
        "model_kept": receipt.get("model"),
        "next_command": None,
        "next_steps": [
            step(command("deploy", run_dir), "deploy the kept model again (it previews first)")
        ],
    }


def register(app: typer.Typer) -> None:
    """Add `deploy` and `undeploy` to ``app``."""
    app.command(
        "deploy",
        short_help="Deploy a run's model on its provider and smoke-test it (paid).",
        help=(
            "Create an on-demand deployment of a completed run's promoted model on the run's "
            "provider (at most one replica, scaling to zero when idle), wait until it serves "
            "(--timeout-minutes), and send one chat completion as a smoke test. The receipt, "
            "deployment.json in RUN_DIR, is written as soon as the deployment exists, so a "
            "failed or timed-out wait still records it. Without --confirm, previews what "
            "would be created. The result says how to call the model (usage: the base URL, "
            "the model to request and an example request that reads the key from its "
            "environment variable) and how to delete it (undeploy_command). "
            f"{BILLING[0].upper()}{BILLING[1:]}: run `pup finetune undeploy "
            "RUN_DIR --confirm`. "
            "A failed smoke test exits 1 and keeps the deployment. A remote write: with "
            f"PUP_READ_ONLY set, it previews but refuses --confirm. {_AGENT_RULE}"
        ),
    )(deploy)
    app.command(
        "undeploy",
        short_help="Delete a run's deployment so it stops billing.",
        help=(
            "Delete the deployment recorded in RUN_DIR/deployment.json and record when; the "
            "promoted model is kept and can be deployed again. Deleting a deployment that is "
            "already gone succeeds. Without --confirm, previews what would be deleted. A "
            "remote write: with PUP_READ_ONLY set, it previews but refuses --confirm."
        ),
    )(undeploy)
