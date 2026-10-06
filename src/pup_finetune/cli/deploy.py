# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune deploy` and `undeploy`: serve a run's best checkpoint, and stop serving it."""

from __future__ import annotations

import shlex
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from pup_finetune.cli._common import get_runtime, run_command
from pup_finetune.errors import StateError, require_confirm, require_writable
from pup_finetune.providers import DeployRequest, Provider, get_provider
from pup_finetune.training.run import PLAN_FILE, RunDir

DEFAULT_TIMEOUT_SECONDS = 1800.0


class ServingMode(StrEnum):
    """``--serving-mode`` values; an enum because typer before 0.19 rejects ``Literal``."""

    DEDICATED = "dedicated"
    SAMPLER = "sampler"


RunDirOption = Annotated[
    Path, typer.Option("--run-dir", help="The run directory of a finished training run.")
]


def _provider_of(run_dir: Path) -> Provider:
    name = RunDir(run_dir).read_plan().get("provider")
    if not isinstance(name, str):
        raise StateError(f"{run_dir / PLAN_FILE} names no provider")
    return get_provider(name)


@run_command
def deploy(
    run_dir: RunDirOption,
    serving_mode: Annotated[
        ServingMode,
        typer.Option(
            "--serving-mode",
            help="`dedicated`: a deployment on its own GPUs (needs --accelerator). "
            "`sampler`: a Loops sampler that loads the checkpoint.",
        ),
    ] = ServingMode.DEDICATED,
    accelerator: Annotated[
        str | None,
        typer.Option(
            "--accelerator", help="GPU type and count for dedicated serving, e.g. H100:1."
        ),
    ] = None,
    max_seq_len: Annotated[
        int | None,
        typer.Option("--max-seq-len", help="Serving context limit (default: the model's)."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=1, max=7200, help="Seconds to wait for the deployment."),
    ] = DEFAULT_TIMEOUT_SECONDS,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Create the deployment; preview it otherwise.")
    ] = False,
) -> dict[str, Any]:
    require_writable(get_runtime(), "deploying a checkpoint")
    provider = _provider_of(run_dir)
    mode: Literal["dedicated", "sampler"] = (
        "sampler" if serving_mode is ServingMode.SAMPLER else "dedicated"
    )
    request = DeployRequest(
        run_dir=run_dir,
        serving_mode=mode,
        accelerator=accelerator,
        max_seq_len=max_seq_len,
        timeout_seconds=timeout,
    )
    preview = provider.deploy_preview(request)
    rerun = ["pup", "finetune", "deploy", "--run-dir", str(run_dir), "--serving-mode", mode]
    if accelerator is not None:
        rerun += ["--accelerator", accelerator]
    if max_seq_len is not None:
        rerun += ["--max-seq-len", str(max_seq_len)]
    if timeout != DEFAULT_TIMEOUT_SECONDS:
        rerun += ["--timeout", str(timeout)]
    require_confirm(confirm, preview, shlex.join([*rerun, "--confirm"]))
    return provider.deploy(request)


@run_command
def undeploy(
    run_dir: RunDirOption,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Stop the deployment; preview it otherwise.")
    ] = False,
) -> dict[str, Any]:
    require_writable(get_runtime(), "undeploying a checkpoint")
    provider = _provider_of(run_dir)
    preview = {
        "provider": provider.name,
        "run_dir": str(run_dir),
        "action": "deactivate the deployment recorded in the run directory; keep the checkpoint",
    }
    rerun = shlex.join(["pup", "finetune", "undeploy", "--run-dir", str(run_dir), "--confirm"])
    require_confirm(confirm, preview, rerun)
    return provider.undeploy(run_dir)


def register(app: typer.Typer) -> None:
    """Add the `deploy` and `undeploy` commands to ``app``."""
    app.command(
        name="deploy",
        short_help="Serve a run's best checkpoint (paid; previews without --confirm).",
        help=(
            "Serve the run's best checkpoint behind an OpenAI-compatible endpoint and smoke-test "
            "it. It bills per hour until `pup finetune undeploy`. Without --confirm, prints what "
            "would be created and the exact command that creates it."
        ),
    )(deploy)
    app.command(
        name="undeploy",
        short_help="Stop serving a run's checkpoint (previews without --confirm).",
        help=(
            "Stop the deployment recorded in the run directory. The checkpoint and the receipt "
            "are kept. Without --confirm, prints what would be stopped."
        ),
    )(undeploy)
