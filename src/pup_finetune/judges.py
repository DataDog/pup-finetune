# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Judges on any provider: which provider a judge spec names, and the client that serves it.

A judge spec (``--judge-model``, ``--eval-judge-model`` and each ``--judges`` entry) names
its provider by prefix:

- ``anthropic/<model>``, e.g. ``anthropic/claude-opus-5-5``: the Anthropic API, with the
  ``anthropic`` key (or the Anthropic SDK's own login), through the official SDK (the
  ``anthropic`` extra).
- ``openai/<model>``: the OpenAI API, with the ``openai`` key.
- Anything else is a Fireworks model id, used as written (e.g.
  ``accounts/fireworks/models/<model>``): Fireworks serverless inference, with the
  ``fireworks`` key. The default judge is one.

:func:`parse_judge` turns a spec into a :class:`~pup_finetune.inference.JudgeConfig` with
the provider's defaults (:data:`PROVIDERS`: endpoint, credential, whether a temperature is
sent, and the token limit; thinking shares the limit on Anthropic's and OpenAI's current
models, so theirs is larger). :func:`open_client` opens the client for a judge and
:func:`open_clients` one client per provider of a council. Nothing here imports a provider
SDK: :mod:`pup_finetune.anthropic_client` is loaded only when an Anthropic judge is used
(:func:`anthropic_client_class`), and :func:`check_available` says before then when its extra
is missing.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from pup_finetune.errors import UsageError
from pup_finetune.inference import (
    DEFAULT_JUDGE_BASE_URL,
    ChatClient,
    ChatCompleter,
    JudgeConfig,
    JudgeProvider,
)

if TYPE_CHECKING:
    from pup_finetune.anthropic_client import AnthropicChatClient

__all__ = [
    "PROVIDERS",
    "SPEC_FORMS",
    "JudgeProvider",
    "ProviderDefaults",
    "RemoteProvider",
    "anthropic_client_class",
    "check_available",
    "destination",
    "open_client",
    "open_clients",
    "parse_judge",
]

RemoteProvider = Literal["fireworks", "anthropic", "openai"]
"""The providers a judge spec can name."""

SPEC_FORMS: Final = (
    "anthropic/<model>, openai/<model>, or a Fireworks model id such as "
    "accounts/fireworks/models/<model>"
)
"""How a judge spec may be written, for help texts and errors."""

ANTHROPIC_INSTALL_HINT: Final = (
    "install the anthropic extra: `uv sync --extra anthropic` in a checkout, or "
    "`pip install 'pup-finetune[anthropic]'`"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderDefaults:
    """What a judge on one provider gets unless told otherwise."""

    name: str
    """How messages name the provider."""
    base_url: str
    credential: str
    temperature: float | None
    """The temperature sent, or ``None`` when the provider's models take none."""
    max_tokens: int


PROVIDERS: Final[dict[RemoteProvider, ProviderDefaults]] = {
    "fireworks": ProviderDefaults(
        name="Fireworks",
        base_url=DEFAULT_JUDGE_BASE_URL,
        credential="fireworks",
        temperature=0.0,
        max_tokens=512,
    ),
    # Current Claude models think on every request, with thinking counted against
    # max_tokens, and reject sampling parameters.
    "anthropic": ProviderDefaults(
        name="Anthropic",
        base_url="https://api.anthropic.com",
        credential="anthropic",
        temperature=None,
        max_tokens=16000,
    ),
    # Current OpenAI models reason, with reasoning tokens counted against the limit, and
    # take no temperature (see pup_finetune.inference).
    "openai": ProviderDefaults(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        credential="openai",
        temperature=None,
        max_tokens=16000,
    ),
}

_PREFIXED_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic", "openai"})
"""The providers a judge spec names by prefix; any other spec is a Fireworks model id."""


def parse_judge(spec: str) -> JudgeConfig:
    """The judge ``spec`` names (see the module docstring), with its provider's defaults.

    ``anthropic/<model>`` and ``openai/<model>`` name those providers (the prefix in any
    case); any other spec is a Fireworks model id. Raises :class:`UsageError` for an empty spec or an empty model
    after a provider prefix.
    """
    text = spec.strip()
    if not text:
        raise UsageError("the judge model is empty", hint=f"name it as {SPEC_FORMS}")
    prefix, slash, rest = text.partition("/")
    provider: RemoteProvider
    if slash and prefix.lower() in _PREFIXED_PROVIDERS:
        provider, model = cast(RemoteProvider, prefix.lower()), rest.strip()
        if not model:
            raise UsageError(f"the judge model {text!r} has no model after {prefix}/")
    else:
        provider, model = "fireworks", text
    defaults = PROVIDERS[provider]
    return JudgeConfig(
        provider=provider,
        model=model,
        base_url=defaults.base_url,
        credential=defaults.credential,
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
    )


def destination(config: JudgeConfig) -> str:
    """Where ``config``'s judge sends what it judges, e.g. ``Anthropic (api.anthropic.com)``."""
    host = config.base_url.split("//", 1)[-1].split("/", 1)[0]
    defaults = PROVIDERS.get(cast(RemoteProvider, config.provider))
    return f"{defaults.name if defaults is not None else config.provider} ({host})"


def check_available(config: JudgeConfig) -> None:
    """Raise :class:`UsageError` with an install hint when ``config``'s judge needs a
    package that is not installed (the ``anthropic`` extra). Imports nothing."""
    if config.provider == "anthropic" and importlib.util.find_spec("anthropic") is None:
        raise _anthropic_missing()


def _anthropic_missing() -> UsageError:
    return UsageError(
        "Anthropic judges need the 'anthropic' package, which is not installed",
        hint=ANTHROPIC_INSTALL_HINT,
    )


def anthropic_client_class() -> type[AnthropicChatClient]:
    """:class:`~pup_finetune.anthropic_client.AnthropicChatClient`, imported now; a
    :class:`UsageError` with an install hint when the ``anthropic`` extra is missing."""
    try:
        from pup_finetune.anthropic_client import AnthropicChatClient
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".", 1)[0] in ("anthropic", "httpx2"):
            raise _anthropic_missing() from exc
        raise
    return AnthropicChatClient


@contextmanager
def open_client(config: JudgeConfig) -> Iterator[ChatCompleter]:
    """The client for ``config``'s judge, with the key of its credential. Closed on exit.

    Fireworks and OpenAI judges get a :class:`~pup_finetune.inference.ChatClient` (with
    OpenAI's request policy for OpenAI), Anthropic judges an
    :class:`~pup_finetune.anthropic_client.AnthropicChatClient`. Raises :class:`UsageError`
    when no key is configured, or a package the judge needs is missing.
    """
    if config.provider == "anthropic":
        with anthropic_client_class().for_judge(config) as client:
            yield client
        return
    if config.provider not in ("fireworks", "openai"):
        raise UsageError(f"no client serves {config.provider} judges")
    with ChatClient.for_judge(config) as chat:
        yield chat


@contextmanager
def open_clients(configs: Iterable[JudgeConfig]) -> Iterator[dict[JudgeProvider, ChatCompleter]]:
    """One client per provider of ``configs``, by provider, all opened up front and closed
    together on exit (see :func:`open_client`)."""
    with ExitStack() as stack:
        clients: dict[JudgeProvider, ChatCompleter] = {}
        for config in configs:
            if config.provider not in clients:
                clients[config.provider] = stack.enter_context(open_client(config))
        yield clients
