# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""`pup finetune auth`: store, inspect and remove provider API keys in the OS keychain."""

from __future__ import annotations

import getpass
import sys
from typing import Annotated, Any

import typer
from pydantic import SecretStr

from pup_finetune import credentials
from pup_finetune import providers as _providers  # noqa: F401 - registers provider API keys
from pup_finetune.cli._common import run_command
from pup_finetune.cli._presentation import SET_UP, CommandDoc, Safety, default_note, describe, group
from pup_finetune.errors import ExternalError, UsageError

PROVIDERS = ", ".join(spec.provider for spec in credentials.all_specs())
"""The providers whose keys ``auth`` manages, for help texts: every registered credential
(:func:`pup_finetune.credentials.all_specs`)."""

auth_app = group(
    "auth",
    short="Store, check or remove provider API keys (OS keychain).",
    description=describe(
        "Manage provider API keys. A key is read from the provider's environment variable "
        "first, then the OS keychain (service `pup-finetune`, account PROVIDER). Datadog "
        "credentials are never involved: pup handles those.",
        "On macOS, reading a key that was stored with the `security` command may show a "
        "keychain access prompt the first time; that is expected.",
    ),
)

LOGIN_DOC = CommandDoc(
    path="auth login",
    short="Store a provider API key in the OS keychain.",
    description=(
        "Store a provider API key in the OS keychain, from a hidden prompt or, with --stdin, "
        "from piped input. The key is never an option value, so it stays out of shell "
        "history and process listings."
    ),
    examples=(
        "pup finetune auth login fireworks",
        "pup finetune auth login anthropic --stdin < anthropic.key",
    ),
    safety=Safety(
        data="writes the OS keychain; may check the key with the provider",
        recipients="the OS keychain; the provider, to check the key (read-only)",
        without="stores the key",
    ),
)
STATUS_DOC = CommandDoc(
    path="auth status",
    short="Show where each provider's API key comes from.",
    description=(
        "Report, per provider, whether an API key is configured and where it comes from: "
        "env, keychain, provider-config, or anthropic-sdk when the Anthropic SDK can sign in "
        "on its own, and what the key is needed for. Never prints the key."
    ),
    examples=("pup finetune auth status", "pup finetune auth status fireworks"),
    safety=Safety(
        data="reads the environment and the OS keychain",
        recipients="nowhere",
        without="reports where each key comes from",
    ),
)
LOGOUT_DOC = CommandDoc(
    path="auth logout",
    short="Delete a provider's API key from the OS keychain.",
    description="Delete a provider's API key from the OS keychain.",
    examples=("pup finetune auth logout fireworks", "pup finetune auth logout openai"),
    safety=Safety(
        data="deletes from the OS keychain", recipients="nowhere", without="deletes the key"
    ),
)


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _prompt_hidden(prompt: str) -> str:
    """Read one line from the terminal without echoing it."""
    return getpass.getpass(prompt)


def _read_key(spec: credentials.CredentialSpec, *, from_stdin: bool) -> SecretStr:
    # --stdin on a terminal would echo a pasted key, so a terminal always gets the hidden
    # prompt.
    if _stdin_is_tty():
        try:
            text = _prompt_hidden(f"{spec.provider} API key (input is hidden): ")
        except EOFError:
            raise UsageError(
                f"no {spec.provider} API key was entered",
                hint="type or paste the key at the prompt, or pipe the key in with --stdin",
            ) from None
    elif from_stdin:
        text = sys.stdin.read()
    else:
        raise UsageError(
            "there is no terminal to prompt for the key on",
            hint=f"run `pup finetune auth login {spec.provider}` in your own terminal, or pipe "
            "the key in with --stdin",
        )
    value = text.strip()
    if not value:
        raise UsageError(f"no {spec.provider} API key was entered")
    credentials.register_secret(value)
    return SecretStr(value)


@auth_app.command(**LOGIN_DOC.typer())
@run_command
def login(
    provider: Annotated[
        str, typer.Argument(metavar="PROVIDER", help=f"One of {PROVIDERS}.", show_default=False)
    ],
    stdin: Annotated[
        bool, typer.Option("--stdin", help="Read the key from stdin instead of prompting.")
    ] = False,
) -> dict[str, Any]:
    spec = credentials.get_spec(provider)
    # Refuse before asking for the key or calling the provider; store() checks again.
    credentials.ensure_storable(spec.provider)
    value = _read_key(spec, from_stdin=stdin)
    account = spec.validator(value) if spec.validator is not None else None
    credentials.store(spec.provider, value)
    result: dict[str, Any] = {"provider": spec.provider, "stored": True, "source": "keychain"}
    if account is not None:
        result["account"] = account
    return result


def _status_entry(spec: credentials.CredentialSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "provider": spec.provider,
        "env_var": spec.env_var,
        "configured": False,
        "source": None,
        "needed_for": spec.needed_for or None,
    }
    try:
        found = credentials.lookup(spec.provider)
    except ExternalError as exc:
        entry["error"] = credentials.redact(exc.message)
        return entry
    if found is not None:
        entry.update(configured=True, source=found.source)
    elif (sdk := credentials.sdk_login_source(spec.provider)) is not None:
        entry.update(configured=True, source=sdk)
    return entry


@auth_app.command(**STATUS_DOC.typer())
@run_command
def status(
    provider: Annotated[
        str | None,
        typer.Argument(
            metavar="PROVIDER",
            help=f"Only this provider ({PROVIDERS}). {default_note('all')}",
            show_default=False,
        ),
    ] = None,
) -> dict[str, Any]:
    specs = [credentials.get_spec(provider)] if provider is not None else credentials.all_specs()
    return {"providers": [_status_entry(spec) for spec in specs]}


@auth_app.command(**LOGOUT_DOC.typer())
@run_command
def logout(
    provider: Annotated[
        str, typer.Argument(metavar="PROVIDER", help=f"One of {PROVIDERS}.", show_default=False)
    ],
) -> dict[str, Any]:
    spec = credentials.get_spec(provider)
    return {"provider": spec.provider, "deleted": credentials.delete(spec.provider)}


def register(app: typer.Typer) -> None:
    """Add the `auth` command group to ``app``."""
    app.add_typer(auth_app, name="auth", rich_help_panel=SET_UP)
