# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Chat completions from Anthropic's Messages API, through the official SDK.

This is the only module that imports the ``anthropic`` SDK (the ``anthropic`` extra); it is
loaded only when an Anthropic judge or compare model is used
(:func:`pup_finetune.judges.open_client`). The SDK 1.x is built on ``httpx2``: nothing from
the ``httpx`` package is ever passed to it.

:class:`AnthropicChatClient` answers :class:`~pup_finetune.inference.ChatCompleter` calls,
whose messages and tools are in the OpenAI form:

- The messages become the top-level ``system`` and alternating turns, with tool calls and
  their results as ``tool_use`` and ``tool_result`` blocks
  (:func:`~pup_finetune.anthropic_messages.anthropic_messages`, which the compare target's
  skip rule checks too). ``tools`` become Anthropic tools (``input_schema`` from
  ``parameters``). ``model`` is the bare model id and ``max_tokens`` the caller's limit,
  which thinking shares on current models.
- No sampling parameter (``temperature``, ``top_p``, ``top_k``) is ever sent: current models
  reject them.
- A ``response_schema`` becomes structured output, ``output_config.format`` with that JSON
  schema, on the models that support it (:data:`STRUCTURED_OUTPUT_MODELS`).
- The models that take an effort level (:data:`EFFORT_MODELS`) run at the requested
  ``reasoning_effort`` (a judge asks for ``low``, plenty for a verdict), :data:`EFFORT` when
  none is given.
- The models with server-side refusal fallbacks (:data:`FALLBACK_MODELS`) are called through
  the beta Messages API with ``fallbacks="default"``, so a declined request is retried on
  the fallback model the API picks for the refusal's category. A compare model is opened
  with ``fallbacks=False``: its answers must be its own.

The answer maps back to a :class:`~pup_finetune.inference.ChatCompletion`: the text blocks
concatenated (``thinking`` and ``fallback`` blocks are ignored), its ``tool_use`` blocks as
OpenAI ``tool_calls`` (``arguments`` the input as a JSON string), the usage, the model that
answered, and ``stop_reason`` as ``finish_reason`` (``end_turn`` is ``stop``, ``tool_use`` is
``tool_calls``, ``max_tokens`` and ``model_context_window_exceeded`` are ``length``,
``refusal`` stays ``refusal``, anything else passes through).

Errors are the SDK's typed exceptions, after its own retries (``max_retries`` is set
explicitly): a rejected key, or a failed SDK sign-in (``CredentialsError``,
``WorkloadIdentityError``), is a :class:`~pup_finetune.providers.base.ProviderAuthError`;
rate limiting (429) and overload (529) a
:class:`~pup_finetune.providers.base.ProviderCapacityError`; other server errors, timeouts
and connection failures a :class:`~pup_finetune.providers.base.TransientError`; a request
refused as longer than the model's context window ("prompt is too long") a
:class:`~pup_finetune.inference.ContextLimitError`; anything else an
:class:`~pup_finetune.errors.ExternalError`. Every message is redacted, and the key
is registered for redaction before it is used.

Without a key from the environment or the keychain, the SDK client is built without one, so
it signs in on its own (``ANTHROPIC_AUTH_TOKEN``, an ``ant auth login`` profile, workload
identity federation; see :mod:`pup_finetune.judge_credentials`). The tokens it would sign in
with from the environment are registered for redaction first, and a sign-in the SDK refuses
when the client is built is a :class:`~pup_finetune.errors.UsageError` with the login hint.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Final, Protocol, Self, cast

import anthropic
from pydantic import SecretStr

from pup_finetune import credentials
from pup_finetune.anthropic_messages import anthropic_messages
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.http_common import DETAIL_LIMIT, bearer_key
from pup_finetune.inference import (
    DEFAULT_MAX_ATTEMPTS,
    ChatCompletion,
    ContextLimitError,
    JudgeConfig,
    is_context_limit,
)
from pup_finetune.judges import PROVIDERS, login_hint
from pup_finetune.providers.base import ProviderAuthError, ProviderCapacityError, TransientError
from pup_finetune.redaction import redact

__all__ = [
    "BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "EFFORT",
    "EFFORT_MODELS",
    "FALLBACK_BETA",
    "FALLBACK_MODELS",
    "STRUCTURED_OUTPUT_MODELS",
    "AnthropicChatClient",
    "SdkClient",
]

BASE_URL: Final = PROVIDERS["anthropic"].base_url
DEFAULT_TIMEOUT_S: Final = 600.0
"""Each attempt's limit: a judge that thinks can take minutes."""
SOURCE: Final = "anthropic"
EFFORT: Final = "low"
_DEFAULT_MAX_TOKENS: Final = PROVIDERS["anthropic"].max_tokens

FALLBACK_MODELS: Final = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
"""The models called with server-side refusal fallbacks (``fallbacks="default"``)."""
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
EFFORT_MODELS: Final = FALLBACK_MODELS | frozenset(
    {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    }
)
"""The models sent ``output_config.effort``."""
STRUCTURED_OUTPUT_MODELS: Final = EFFORT_MODELS | frozenset(
    {"claude-opus-4-5", "claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"}
)
"""The models sent ``output_config.format`` for a ``response_schema``."""

_DATED: Final = re.compile(r"^(?P<base>.+?)-\d{8}$")
_FINISH_REASONS: Final = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "refusal",
}


class _Messages(Protocol):
    def create(self, **params: Any) -> Any: ...


class _Beta(Protocol):
    @property
    def messages(self) -> _Messages: ...


class _Models(Protocol):
    def list(self, **params: Any) -> Any: ...


class SdkClient(Protocol):
    """The part of ``anthropic.Anthropic`` this client uses; tests pass a fake."""

    @property
    def messages(self) -> _Messages: ...

    @property
    def beta(self) -> _Beta: ...

    @property
    def models(self) -> _Models: ...

    def close(self) -> None: ...


def _base_model(model: str) -> str:
    """``model`` without a date suffix (``claude-haiku-4-5-20251001`` is ``claude-haiku-4-5``)."""
    match = _DATED.match(model)
    return match.group("base") if match else model


class AnthropicChatClient:
    """A :class:`~pup_finetune.inference.ChatCompleter` on the Anthropic Messages API.

    ``api_key`` ``None`` lets the SDK sign in on its own (see the module docstring).
    ``client`` replaces the SDK client (tests pass a fake, or an SDK client on a mock
    transport); ``timeout_s`` bounds each attempt and ``max_attempts`` counts the SDK's
    retries in; ``fallbacks`` ``False`` never asks for server-side refusal fallbacks. Use it
    as a context manager, or call :meth:`close`.
    """

    def __init__(
        self,
        api_key: SecretStr | None,
        *,
        base_url: str = BASE_URL,
        client: SdkClient | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        fallbacks: bool = True,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._hint = _login_hint()
        secret = (
            bearer_key(api_key, label="the Anthropic API key", hint=self._hint)
            if api_key is not None
            else None
        )
        if client is None:
            if not base_url.startswith("https://"):
                raise UsageError(f"{base_url!r} is not an https URL")
            try:
                built = anthropic.Anthropic(
                    api_key=secret,
                    base_url=base_url,
                    timeout=timeout_s,
                    max_retries=max_attempts - 1,
                )
            except anthropic.AnthropicError as exc:  # the SDK's own sign-in, refused
                raise UsageError(
                    f"the Anthropic SDK could not sign in: {_detail(exc)}", hint=self._hint
                ) from exc
            client = cast(SdkClient, built)
        self._client = client
        self._fallbacks = fallbacks

    @classmethod
    def for_judge(
        cls,
        config: JudgeConfig,
        *,
        env: Mapping[str, str] | None = None,
        client: SdkClient | None = None,
        timeout_s: float | None = None,
        fallbacks: bool = True,
    ) -> AnthropicChatClient:
        """A client for ``config``'s judge (or compare model), with the key of its credential.

        The key comes from :func:`pup_finetune.credentials.lookup` (the environment, then
        the keychain; ``env`` replaces the environment). Without one, the client is built
        without a key when the Anthropic SDK can sign in on its own (the tokens it would use
        registered for redaction first), and otherwise raises the :class:`UsageError`
        :func:`~pup_finetune.credentials.resolve` would.
        """
        environment = os.environ if env is None else env
        timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        found = credentials.lookup(config.credential, env=environment)
        if found is not None:
            return cls(
                found.value,
                base_url=config.base_url,
                client=client,
                timeout_s=timeout_s,
                fallbacks=fallbacks,
            )
        if credentials.sdk_login_source(config.credential, env=environment) is None:
            raise credentials.missing_key_error(config.credential, hint=_login_hint())
        credentials.register_sdk_secrets(config.credential, env=environment)
        return cls(
            None, base_url=config.base_url, client=client, timeout_s=timeout_s, fallbacks=fallbacks
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    # --- calls ------------------------------------------------------------------------

    def check_key(self) -> None:
        """List one model, a free call: :class:`UsageError` when the key is rejected, and
        the errors of :meth:`complete` otherwise."""
        try:
            self._client.models.list(limit=1)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise UsageError(
                f"Anthropic rejected the API key (HTTP {exc.status_code}): {_detail(exc)}",
                hint=self._hint,
            ) from exc
        except anthropic.AnthropicError as exc:
            raise self._error("checking the Anthropic key", exc) from exc

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: Mapping[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatCompletion:
        """One message from ``model``, mapped as the module docstring says.

        ``temperature`` is ignored (current models reject it); ``max_tokens`` is required by
        the API, so it defaults to the Anthropic judge's limit.
        """
        del temperature  # never sent: current models reject sampling parameters
        what = f"a message from {model}"
        params = self._params(messages, model=model, max_tokens=max_tokens)
        if tools:
            params["tools"] = [_anthropic_tool(tool) for tool in tools]
        base = _base_model(model)
        output_config: dict[str, Any] = {}
        if base in EFFORT_MODELS:
            output_config["effort"] = reasoning_effort or EFFORT
        if response_schema is not None and base in STRUCTURED_OUTPUT_MODELS:
            output_config["format"] = {"type": "json_schema", "schema": dict(response_schema)}
        if output_config:
            params["output_config"] = output_config
        try:
            if self._fallbacks and base in FALLBACK_MODELS:
                message = self._client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                message = self._client.messages.create(**params)
        except anthropic.AnthropicError as exc:
            raise self._error(what, exc) from exc
        return _completion(message, model)

    def _params(
        self, messages: Sequence[Mapping[str, Any]], *, model: str, max_tokens: int | None
    ) -> dict[str, Any]:
        system, turns = anthropic_messages(messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS,
            "messages": turns,
        }
        if system is not None:
            params["system"] = system
        return params

    # --- errors -----------------------------------------------------------------------

    def _error(self, what: str, exc: anthropic.AnthropicError) -> ExternalError:
        """``exc`` as one of our errors, most specific first (see the module docstring)."""
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            problem = (
                "the API key is not valid"
                if isinstance(exc, anthropic.AuthenticationError)
                else "the API key is not allowed to do this"
            )
            return ProviderAuthError(
                f"{what} failed: {problem} (HTTP {exc.status_code}): {_detail(exc)}",
                source=SOURCE,
                hint=self._hint,
            )
        if isinstance(exc, anthropic.RateLimitError | anthropic.OverloadedError):
            return ProviderCapacityError(
                f"{what} failed with HTTP {exc.status_code} after retries: {_detail(exc)}",
                source=SOURCE,
                hint="Anthropic is rate limiting or overloaded; retry later",
            )
        if isinstance(exc, anthropic.APIStatusError):
            if is_context_limit(exc.status_code, None, _detail(exc)):
                return ContextLimitError(
                    f"{what} failed: the request is longer than the model's context window "
                    f"(HTTP {exc.status_code}): {_detail(exc)}",
                    source=SOURCE,
                    hint="pick a model with a larger context window",
                )
            if exc.status_code >= 500:
                return TransientError(
                    f"{what} failed with HTTP {exc.status_code} after retries: {_detail(exc)}",
                    source=SOURCE,
                    hint="retry later",
                )
            return ExternalError(
                f"{what} failed with HTTP {exc.status_code}: {_detail(exc)}",
                source=SOURCE,
                hint="check the model name and the request"
                if exc.status_code in (400, 404, 422)
                else None,
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return TransientError(
                f"{what} failed: timed out, after retries",
                source=SOURCE,
                hint="check the connection to api.anthropic.com and retry",
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return TransientError(
                f"{what} failed: could not reach Anthropic ({_detail(exc)}), after retries",
                source=SOURCE,
                hint="check the connection to api.anthropic.com and retry",
            )
        if isinstance(exc, anthropic.CredentialsError | anthropic.WorkloadIdentityError):
            return ProviderAuthError(
                f"{what} failed: the Anthropic SDK could not sign in: {_detail(exc)}",
                source=SOURCE,
                hint=self._hint,
            )
        return ExternalError(f"{what} failed: {_detail(exc)}", source=SOURCE)


def _login_hint() -> str:
    """How a user signs in to Anthropic: our key, or the SDK's own login."""
    return login_hint(SOURCE)


def _anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """An OpenAI function tool as an Anthropic tool."""
    function = tool.get("function")
    function = function if isinstance(function, Mapping) else tool
    parameters = function.get("parameters")
    return {
        "name": str(function.get("name", "")),
        "description": str(function.get("description") or ""),
        "input_schema": dict(parameters) if parameters else {"type": "object", "properties": {}},
    }


def _tool_calls(blocks: Sequence[Any]) -> list[dict[str, Any]]:
    """The ``tool_use`` blocks of an answer as OpenAI tool calls."""
    return [
        {
            "id": str(getattr(block, "id", "")),
            "type": "function",
            "function": {
                "name": str(getattr(block, "name", "")),
                "arguments": json.dumps(getattr(block, "input", None) or {}, ensure_ascii=False),
            },
        }
        for block in blocks
        if getattr(block, "type", None) == "tool_use"
    ]


def _detail(exc: anthropic.AnthropicError) -> str:
    """The API's error message for ``exc`` (else its text), short and redacted."""
    body = getattr(exc, "body", None)
    text = ""
    if isinstance(body, Mapping):
        error = body.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        text = str(message or "")
    if not text:
        text = str(getattr(exc, "message", "") or exc)
    lines = redact(text).strip().splitlines()
    return lines[0][:DETAIL_LIMIT] if lines else type(exc).__name__


def _completion(message: Any, model: str) -> ChatCompletion:
    """The SDK's message as a :class:`~pup_finetune.inference.ChatCompletion`."""
    blocks = list(getattr(message, "content", None) or [])
    texts = [
        str(getattr(block, "text", "") or "")
        for block in blocks
        if getattr(block, "type", None) == "text"
    ]
    answer: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
    if calls := _tool_calls(blocks):
        answer["tool_calls"] = calls
    usage: dict[str, Any] = {}
    raw_usage = getattr(message, "usage", None)
    prompt = getattr(raw_usage, "input_tokens", None)
    completion = getattr(raw_usage, "output_tokens", None)
    if isinstance(prompt, int) and isinstance(completion, int):
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    stop = getattr(message, "stop_reason", None)
    answered = getattr(message, "model", None)
    return ChatCompletion(
        message=answer,
        usage=usage,
        model=answered if isinstance(answered, str) and answered else model,
        finish_reason=_FINISH_REASONS.get(stop, stop) if isinstance(stop, str) else None,
    )
