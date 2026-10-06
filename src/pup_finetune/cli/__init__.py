# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Command-line interface: the root Typer app for `pup finetune`."""

from __future__ import annotations

from typing import Annotated

import typer

from pup_finetune import __version__
from pup_finetune.cli import auth, deploy, doctor, train
from pup_finetune.cli._common import PROG_NAME, run_app

app = typer.Typer(
    name=PROG_NAME,
    help=(
        "Curate a dataset from your Datadog Agent Observability traces, fine-tune an "
        "open-weight model, and evaluate it against the base model. "
        "Runs as the pup extension `pup finetune`."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    # Root help text lives in typer.Typer(help=...) above; this callback only hosts --version.
    del version  # handled eagerly by _version_callback


auth.register(app)
doctor.register(app)
train.register(app)
deploy.register(app)


def main() -> None:
    """Console-script entry point: run the app and exit with the mapped exit code."""
    run_app(app)
