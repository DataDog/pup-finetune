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
from pup_finetune.cli._common import run_command
from pup_finetune.errors import ExternalError, UsageError

_LOCAL_ONLY = (
    "Writes only to the local OS keychain (login may make one read-only call to the provider "
    "to check the key), so it is allowed with PUP_READ_ONLY set. "
    "Datadog credentials are never involved: pup handles those."
)

auth_app = typer.Typer(
    name="auth",
    short_help="Store, check or remove provider API keys (local keychain only).",
    help=(
        "Manage provider API keys. Keys are read from the provider's environment variable "
        "first, then the OS keychain (service `pup-finetune`, account `<provider>`). "
        f"{_LOCAL_ONLY} "
        "On macOS, reading a key that was stored with the `security` command may show a "
        "keychain access prompt the first time; that is expected."
    ),
    no_args_is_help=True,
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


@auth_app.command(
    "login",
    short_help="Store a provider API key in the OS keychain.",
    help=(
        "Store a provider API key in the OS keychain. Prompts with hidden input, or reads "
        "piped input with --stdin, e.g. `<secret-manager> read ... | pup finetune auth "
        "login <provider> --stdin`. The key is never accepted as an option value, so it "
        f"stays out of shell history and process listings. {_LOCAL_ONLY}"
    ),
)
@run_command
def login(
    provider: Annotated[str, typer.Argument(help="The provider, e.g. `fireworks`.")],
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
    }
    try:
        found = credentials.lookup(spec.provider)
    except ExternalError as exc:
        entry["error"] = exc.message
        return entry
    if found is not None:
        entry.update(configured=True, source=found.source)
    return entry


@auth_app.command(
    "status",
    short_help="Show where each provider's API key comes from.",
    help=(
        "Report, per provider, whether an API key is configured and where it comes from "
        f"(env, keychain or provider-config). Never prints the key. {_LOCAL_ONLY}"
    ),
)
@run_command
def status(
    provider: Annotated[
        str | None, typer.Argument(help="Only this provider (default: all registered).")
    ] = None,
) -> dict[str, Any]:
    specs = [credentials.get_spec(provider)] if provider is not None else credentials.all_specs()
    return {"providers": [_status_entry(spec) for spec in specs]}


@auth_app.command(
    "logout",
    short_help="Delete a provider's API key from the OS keychain.",
    help=f"Delete a provider's API key from the OS keychain. {_LOCAL_ONLY}",
)
@run_command
def logout(
    provider: Annotated[str, typer.Argument(help="The provider, e.g. `fireworks`.")],
) -> dict[str, Any]:
    spec = credentials.get_spec(provider)
    return {"provider": spec.provider, "deleted": credentials.delete(spec.provider)}


def register(app: typer.Typer) -> None:
    """Add the `auth` command group to ``app``."""
    app.add_typer(auth_app, name="auth")
