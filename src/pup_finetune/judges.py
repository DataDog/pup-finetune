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
- Anything else is a Fireworks model id (e.g. ``accounts/fireworks/models/<model>``):
  Fireworks serverless inference, with the ``fireworks`` key. The default judge is one. An
  alias of :data:`JUDGE_SHORT_NAMES` (``deepseek-v4p1-flash``) stands for the full id, and the
  full id is what a spec resolves to (:attr:`~pup_finetune.inference.JudgeConfig.spec`), so a
  preview prints it and ``triage.json`` and ``evaluation.json`` freeze it: an alias that later
  names another model never changes the judge of an existing directory.

:func:`parse_judge` turns a spec into a :class:`~pup_finetune.inference.JudgeConfig` with
the provider's defaults (:data:`PROVIDERS`: endpoint, credential, whether a temperature is
sent, the token limit, and the reasoning effort; thinking shares the limit on Anthropic's and
OpenAI's current models, so theirs is larger). :func:`open_client` opens the client for a
judge and :func:`open_clients` one client per provider of a council. Nothing here imports a
provider SDK: :mod:`pup_finetune.anthropic_client` is loaded only when an Anthropic judge is
used (:func:`anthropic_client_class`), and :func:`check_available` says before then when its
extra is missing.

**Reasoning effort.** A verdict needs little thinking, so a judge asks for the lowest effort
its model accepts: ``low`` on Anthropic and OpenAI (the lowest their current models take;
OpenAI's GPT-4-era models, :data:`OPENAI_PLAIN_MODELS`, do not reason and refuse the field
with HTTP 400, so they are sent no reasoning effort), and on Fireworks the effort
:data:`FIREWORKS_REASONING` records for the model, as each one answered in October 2026
(DeepSeek V4.1 Flash takes ``none``; GLM-5.3 Flash thinks on every request and refuses
``none``, but ``low`` cuts its thinking to a few tokens). Any other Fireworks model is sent
no reasoning effort, so it thinks as its provider's default does.
"""

from __future__ import annotations

import difflib
import importlib.util
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from pup_finetune.errors import UsageError, install_hint
from pup_finetune.inference import (
    DEFAULT_JUDGE_BASE_URL,
    ChatClient,
    ChatCompleter,
    JudgeConfig,
    JudgeProvider,
)
from pup_finetune.judge_models import JUDGE_SHORT_NAMES, resolve_alias

if TYPE_CHECKING:
    from pup_finetune.anthropic_client import AnthropicChatClient

__all__ = [
    "FIREWORKS_REASONING",
    "JUDGE_SHORT_NAMES",
    "OPENAI_PLAIN_MODELS",
    "PROVIDERS",
    "SPEC_FORMS",
    "JudgeProvider",
    "ProviderDefaults",
    "RemoteProvider",
    "anthropic_client_class",
    "billing",
    "check_available",
    "destination",
    "open_client",
    "open_clients",
    "parse_judge",
    "recipient",
]

RemoteProvider = Literal["fireworks", "anthropic", "openai"]
"""The providers a judge spec can name."""

UNKNOWN_JUDGE_FORMS: Final = (
    f"an alias ({', '.join(JUDGE_SHORT_NAMES)}), anthropic/MODEL, openai/MODEL, or a Fireworks "
    "model id, accounts/<acct>/models/<id>"
)
"""The forms an unknown judge's hint names."""

SPEC_FORMS: Final = (
    f"an alias ({', '.join(JUDGE_SHORT_NAMES)}), anthropic/<model>, openai/<model>, or a "
    "Fireworks model id such as accounts/fireworks/models/<model>"
)
"""How a judge spec may be written, for help texts and errors."""

ANTHROPIC_INSTALL_HINT: Final = install_hint("anthropic")


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
    reasoning_effort: str | None
    """The reasoning effort sent, or ``None`` to send none (the model's default)."""


PROVIDERS: Final[dict[RemoteProvider, ProviderDefaults]] = {
    "fireworks": ProviderDefaults(
        name="Fireworks",
        base_url=DEFAULT_JUDGE_BASE_URL,
        credential="fireworks",
        temperature=0.0,
        max_tokens=512,
        reasoning_effort=None,
    ),
    # Current Claude models think on every request, with thinking counted against
    # max_tokens, and reject sampling parameters.
    "anthropic": ProviderDefaults(
        name="Anthropic",
        base_url="https://api.anthropic.com",
        credential="anthropic",
        temperature=None,
        max_tokens=16000,
        reasoning_effort="low",
    ),
    # Current OpenAI models reason, with reasoning tokens counted against the limit, and
    # take no temperature (see pup_finetune.inference).
    "openai": ProviderDefaults(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        credential="openai",
        temperature=None,
        max_tokens=16000,
        reasoning_effort="low",
    ),
}

FIREWORKS_REASONING: Final[dict[str, str]] = {
    "accounts/fireworks/models/deepseek-v4p1-flash": "none",
    "accounts/fireworks/models/glm-5p3-flash": "low",
}
"""The lowest reasoning effort each Fireworks model accepts, by model id (see the module
docstring); a model not listed is sent no reasoning effort."""

OPENAI_PLAIN_MODELS: Final = ("gpt-4", "gpt-3.5", "chatgpt-")
"""The prefixes of the OpenAI models that do not reason, which refuse ``reasoning_effort``
("Unrecognized request argument", checked on gpt-4.1-mini and gpt-4o-mini in October 2026)."""

_PREFIXED_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic", "openai"})
"""The providers a judge spec names by prefix; any other spec is a Fireworks model id."""


def parse_judge(spec: str, *, what: str = "judge") -> JudgeConfig:
    """The judge ``spec`` names (see the module docstring), with its provider's defaults.

    ``anthropic/<model>`` and ``openai/<model>`` name those providers (the prefix in any
    case); any other spec with a ``/`` is a Fireworks model id, and a spec without one must
    be an alias of one (:data:`JUDGE_SHORT_NAMES`), resolved to the id, so a mistyped alias is
    never sent to a paid run. The reasoning effort is the lowest the model takes (see the
    module docstring). Raises :class:`UsageError` for an empty spec, an empty model after a
    provider prefix, or an unknown alias (with the closest known one), naming the spec as
    ``what`` (``judge``, or ``compare model`` for ``--compare-model``).
    """
    text = spec.strip()
    if not text:
        raise UsageError(f"the {what} is empty", hint=f"name it as {SPEC_FORMS}")
    prefix, slash, rest = text.partition("/")
    provider: RemoteProvider
    if slash and prefix.lower() in _PREFIXED_PROVIDERS:
        provider, model = cast(RemoteProvider, prefix.lower()), rest.strip()
        if not model:
            raise UsageError(f"the {what} {text!r} has no model after {prefix}/")
    elif slash:
        provider, model = "fireworks", text
    else:
        provider, model = "fireworks", _known_alias(text, what)
    defaults = PROVIDERS[provider]
    effort = defaults.reasoning_effort
    if provider == "fireworks":
        effort = FIREWORKS_REASONING.get(model)
    elif provider == "openai" and model.lower().startswith(OPENAI_PLAIN_MODELS):
        effort = None
    return JudgeConfig(
        provider=provider,
        model=model,
        base_url=defaults.base_url,
        credential=defaults.credential,
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=effort,
    )


def _known_alias(text: str, what: str) -> str:
    """The Fireworks model id the alias ``text`` stands for; raises :class:`UsageError`
    naming the closest alias when it is none (and the spec as ``what``)."""
    if text in JUDGE_SHORT_NAMES:
        return resolve_alias(text)
    close = difflib.get_close_matches(text, list(JUDGE_SHORT_NAMES), n=1)
    mean = f"did you mean {close[0]}? " if close else ""
    raise UsageError(
        f"unknown {what} {text!r}: a {what} without a '/' must be one of the aliases",
        hint=f"{mean}name the {what} as {UNKNOWN_JUDGE_FORMS}",
    )


def _name_and_host(config: JudgeConfig) -> tuple[str, str]:
    host = config.base_url.split("//", 1)[-1].split("/", 1)[0]
    defaults = PROVIDERS.get(cast(RemoteProvider, config.provider))
    return (defaults.name if defaults is not None else config.provider), host


def destination(config: JudgeConfig) -> str:
    """Where ``config``'s judge sends what it judges, e.g. ``Anthropic (api.anthropic.com)``."""
    name, host = _name_and_host(config)
    return f"{name} ({host})"


def recipient(config: JudgeConfig) -> str:
    """``config``'s judge as a ``data_recipients`` entry names it, e.g.
    ``Anthropic api.anthropic.com``."""
    name, host = _name_and_host(config)
    return f"{name} {host}"


def billing(
    config: JudgeConfig, calls: int, *, input_tokens: int | None = None, what: str = "judge"
) -> str:
    """What ``calls`` calls to ``config``'s model bill, in one sentence for a preview: the
    provider bills per token at its own rates, which pup-finetune does not know.

    ``input_tokens`` is a rough estimate of the tokens the calls send, when known; ``what``
    names the calls (``judge``, or ``compare-model`` for the compare target).
    """
    sent = f", about {input_tokens:,} input tokens in all (approximate)" if input_tokens else ""
    counted = f"{calls:,} calls{sent}, at most {config.max_tokens:,} output tokens each"
    if config.provider == "fake":
        model = "judge" if what == "judge" else "compare model"
        return f"the fake {model} is offline and bills nothing: {counted}"
    name, _ = _name_and_host(config)
    return (
        f"{name} bills each {what} call per token at its own rates, which pup-finetune does "
        f"not know: {counted}"
    )


def check_available(config: JudgeConfig, *, what: str = "judge") -> None:
    """Raise :class:`UsageError` with an install hint when ``config``'s judge needs a
    package that is not installed (the ``anthropic`` extra), naming the model as ``what``
    (``judge``, or ``compare model``). Imports nothing."""
    if config.provider == "anthropic" and importlib.util.find_spec("anthropic") is None:
        raise _anthropic_missing(what)


def _anthropic_missing(what: str = "judge") -> UsageError:
    return UsageError(
        f"Anthropic {what}s need the 'anthropic' package, which is not installed",
        hint=ANTHROPIC_INSTALL_HINT,
    )


def anthropic_client_class(*, what: str = "judge") -> type[AnthropicChatClient]:
    """:class:`~pup_finetune.anthropic_client.AnthropicChatClient`, imported now; a
    :class:`UsageError` with an install hint naming the model as ``what`` when the
    ``anthropic`` extra is missing."""
    try:
        from pup_finetune.anthropic_client import AnthropicChatClient
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".", 1)[0] in ("anthropic", "httpx2"):
            raise _anthropic_missing(what) from exc
        raise
    return AnthropicChatClient


@contextmanager
def open_client(
    config: JudgeConfig,
    *,
    timeout_s: float | None = None,
    fallbacks: bool = True,
    what: str = "judge",
) -> Iterator[ChatCompleter]:
    """The client for ``config``'s judge, with the key of its credential. Closed on exit.

    Fireworks and OpenAI judges get a :class:`~pup_finetune.inference.ChatClient` (with
    OpenAI's request policy for OpenAI), Anthropic judges an
    :class:`~pup_finetune.anthropic_client.AnthropicChatClient`; ``timeout_s`` replaces the
    client's limit per attempt, and ``fallbacks`` ``False`` turns off Anthropic's
    server-side refusal fallbacks, which answer with another model. Raises
    :class:`UsageError` when no key is configured, or a package the judge needs is missing
    (naming the model as ``what``).
    """
    if config.provider == "anthropic":
        opened = anthropic_client_class(what=what).for_judge(
            config, timeout_s=timeout_s, fallbacks=fallbacks
        )
        with opened as client:
            yield client
        return
    if config.provider not in ("fireworks", "openai"):
        raise UsageError(f"no client serves {config.provider} judges")
    with ChatClient.for_judge(config, timeout_s=timeout_s) as chat:
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
