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
from pup_finetune.cli._presentation import SET_UP, CommandDoc, Safety, default_note, group
from pup_finetune.errors import FinetuneError
from pup_finetune.providers import get_provider, provider_info, provider_infos, registered_providers
from pup_finetune.providers.base import ModelProfile, Provider
from pup_finetune.redaction import redact

models_app = group(
    "models",
    short="List the models a provider can fine-tune.",
    description="List the model profiles a provider can fine-tune.",
)

LIST_DOC = CommandDoc(
    path="models list",
    short="List the model profiles each provider can fine-tune.",
    description=(
        "List the model profiles of every provider, or of --provider: alias, the names "
        "--model accepts for it, provider, provider model, context length, reasoning "
        "support, default LoRA rank, whether its tokenizer runs code from the model's "
        "repository (trust_remote_code) and the provider's --provider-option keys; this needs "
        "no API key. With --check, also shows each profile's renderer and pinned tokenizer, asks "
        "the provider whether each model can be fine-tuned now, and exits 1 when any cannot."
    ),
    examples=(
        "pup finetune models list",
        "pup finetune models list --provider fireworks --check",
    ),
    safety=Safety(
        data="--check makes read-only provider calls",
        recipients="nowhere (--check: read-only calls to the provider)",
        without="lists the profiles",
    ),
)


def profile_entry(
    profile: ModelProfile, options: tuple[str, ...], *, internals: bool = False
) -> dict[str, Any]:
    """The JSON for one profile; the renderer and tokenizer only with ``internals``."""
    entry: dict[str, Any] = {
        "alias": profile.alias,
        "names": list(profile.names),
        "provider": profile.provider,
        "base_model": profile.base_model,
        "max_seq_len": profile.max_seq_len,
        "supports_reasoning": profile.supports_reasoning,
        "default_lora_rank": profile.default_lora_rank,
        "trust_remote_code": profile.trust_remote_code,
        "provider_options": list(options),
    }
    if internals:
        entry.update(
            renderer=profile.renderer,
            tokenizer=profile.tokenizer_model,
            tokenizer_revision=profile.tokenizer_revision,
        )
    return entry


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
        str | None,
        typer.Option(
            "--provider",
            metavar="NAME",
            help=f"Only this provider: {' or '.join(registered_providers())}. "
            f"{default_note('every provider')}",
            show_default=False,
        ),
    ] = None,
    check: Annotated[
        bool,
        typer.Option("--check", help="Preflight each model with its provider (needs its API key)."),
    ] = False,
) -> dict[str, Any] | CommandResult:
    runtime = get_runtime()
    infos = (
        [provider_info(provider_name, runtime)]
        if provider_name is not None
        else provider_infos(runtime)
    )
    entries = []
    for info in infos:
        provider = get_provider(info.name, runtime)
        for profile in provider.models():
            entry = profile_entry(profile, info.options, internals=check)
            if check:
                entry["preflight"] = _preflight(provider, profile)
            entries.append(entry)
    result: dict[str, Any] = {"provider": provider_name, "models": entries}
    if not check:
        return result
    ok = all(entry["preflight"]["ok"] for entry in entries)
    result["ok"] = ok
    return CommandResult(result, exit_code=0 if ok else 1)


def register(app: typer.Typer) -> None:
    """Add the `models` command group to ``app``."""
    app.add_typer(models_app, name="models", rich_help_panel=SET_UP)
