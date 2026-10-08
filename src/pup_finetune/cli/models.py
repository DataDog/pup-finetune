# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune models`: the model profiles a provider supports, optionally preflighted."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from pup_finetune.cli._common import CommandResult, get_runtime, run_command
from pup_finetune.cli._presentation import SET_UP, CommandDoc, Safety, group
from pup_finetune.errors import FinetuneError
from pup_finetune.providers import get_provider
from pup_finetune.providers.base import ModelProfile, Provider
from pup_finetune.redaction import redact

models_app = group(
    "models",
    short="List the models a provider can fine-tune.",
    description="List the model profiles a provider can fine-tune.",
)

LIST_DOC = CommandDoc(
    path="models list",
    short="List a provider's model profiles.",
    description=(
        "List a provider's model profiles: alias, provider model, renderer, pinned "
        "tokenizer, context length and default LoRA rank; this needs no API key. With "
        "--check, also asks the provider whether each model can be fine-tuned now, and "
        "exits 1 when any cannot."
    ),
    examples=(
        "pup finetune models list --provider fireworks",
        "pup finetune models list --provider fireworks --check",
    ),
    safety=Safety(
        data="--check makes read-only provider calls",
        recipients="nowhere (--check: read-only calls to the provider)",
        without="lists the profiles",
    ),
)


def profile_entry(profile: ModelProfile) -> dict[str, Any]:
    """The JSON for one profile."""
    return {
        "alias": profile.alias,
        "base_model": profile.base_model,
        "renderer": profile.renderer,
        "tokenizer": profile.tokenizer_model,
        "tokenizer_revision": profile.tokenizer_revision,
        "max_seq_len": profile.max_seq_len,
        "default_lora_rank": profile.default_lora_rank,
        "supports_reasoning": profile.supports_reasoning,
    }


def _preflight(provider: Provider, profile: ModelProfile) -> dict[str, Any]:
    try:
        provider.preflight(profile)
    except FinetuneError as exc:
        result: dict[str, Any] = {"ok": False, "error": redact(str(exc))}
        if exc.hint:
            result["hint"] = redact(exc.hint)
        return result
    return {"ok": True}


@models_app.command(**LIST_DOC.typer())
@run_command
def list_models(
    provider_name: Annotated[
        str,
        typer.Option(
            "--provider", metavar="NAME", help="The training provider.", show_default=False
        ),
    ],
    check: Annotated[
        bool,
        typer.Option("--check", help="Preflight each model with the provider (needs its API key)."),
    ] = False,
) -> dict[str, Any] | CommandResult:
    provider = get_provider(provider_name, get_runtime())
    entries = []
    for profile in provider.models():
        entry = profile_entry(profile)
        if check:
            entry["preflight"] = _preflight(provider, profile)
        entries.append(entry)
    result: dict[str, Any] = {"provider": provider.name, "models": entries}
    if not check:
        return result
    ok = all(entry["preflight"]["ok"] for entry in entries)
    result["ok"] = ok
    return CommandResult(result, exit_code=0 if ok else 1)


def register(app: typer.Typer) -> None:
    """Add the `models` command group to ``app``."""
    app.add_typer(models_app, name="models", rich_help_panel=SET_UP)
