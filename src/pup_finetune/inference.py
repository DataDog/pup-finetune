# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""An OpenAI-compatible chat completions client, over ``httpx`` (no SDK).

:class:`ChatClient` sends ``POST {base_url}/chat/completions`` with a ``Bearer`` key and
returns the first choice as a :class:`ChatCompletion`. LLM judges use it, and so will the
deployment smoke test. :class:`JudgeConfig` is the judge's model, endpoint and sampling
settings.

Retries are bounded: HTTP 408, 429 and 5xx and connection failures (timeouts included) are
retried up to ``max_attempts`` times in all, with exponential backoff and jitter, or after
the server's ``Retry-After`` (seconds or an HTTP date) when it sends one; every wait is
capped at :data:`MAX_WAIT_S`. Other 4xx answers, redirects (never followed) and responses
that cannot be read are never retried. When the attempts run out,
429 raises :class:`~pup_finetune.providers.base.ProviderCapacityError` and the rest
:class:`~pup_finetune.providers.base.TransientError`; 401 and 403 raise
:class:`~pup_finetune.providers.base.ProviderAuthError` with a login hint for the configured
credential; anything else is an :class:`~pup_finetune.errors.ExternalError`.

The key is sent only as a ``Bearer`` header. It is registered for redaction when the client is
built and every error message is redacted, so it never reaches output or logs.
"""

from __future__ import annotations

import email.utils
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, Protocol, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from pup_finetune import credentials
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.http_common import CLIENT_SOURCE, bearer_key, error_detail, exception_reason
from pup_finetune.providers.base import ProviderAuthError, ProviderCapacityError, TransientError

__all__ = [
    "DEFAULT_JUDGE_BASE_URL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_TIMEOUT_S",
    "MAX_WAIT_S",
    "ChatClient",
    "ChatCompleter",
    "ChatCompletion",
    "JudgeConfig",
]

DEFAULT_TIMEOUT_S: Final = 60.0
DEFAULT_MAX_ATTEMPTS: Final = 4
BASE_BACKOFF_S: Final = 1.0
"""The wait before the first retry, before jitter; it doubles for each later one."""
MAX_WAIT_S: Final = 30.0
"""The longest wait before any retry, ``Retry-After`` included."""

DEFAULT_JUDGE_MODEL: Final = "accounts/fireworks/models/deepseek-v4p1-flash"
DEFAULT_JUDGE_BASE_URL: Final = "https://api.fireworks.ai/inference/v1"

_DEFAULT_SOURCE: Final = "inference"
_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


class _Random(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """The first choice of a chat completion.

    ``message`` is the OpenAI assistant message as returned (``role`` defaults to
    ``assistant``), ``usage`` the token counts (empty when the server sent none), ``model``
    the model that answered (the requested one when the server does not say) and
    ``finish_reason`` why generation stopped (``stop``, ``length``, ``tool_calls``, ...).
    """

    message: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str | None = None


class ChatCompleter(Protocol):
    """Anything that answers a chat completion request, such as :class:`ChatClient`."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion: ...


class JudgeConfig(BaseModel):
    """Which model judges, where it is served and how it samples.

    ``credential`` names the registered :class:`~pup_finetune.credentials.CredentialSpec`
    whose key the endpoint takes. :meth:`identity` is what a resumed evaluation compares, so
    judgments made under other settings are never reused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(default=DEFAULT_JUDGE_MODEL, min_length=1)
    base_url: str = Field(default=DEFAULT_JUDGE_BASE_URL, min_length=1)
    credential: str = Field(default="fireworks", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)

    def identity(self) -> dict[str, Any]:
        """Every setting that changes a judgment, as a JSON object."""
        return self.model_dump(mode="json")


class ChatClient:
    """An OpenAI-compatible chat client. Use it as a context manager, or call :meth:`close`.

    ``credential`` names the registered credential the key belongs to, for the login hint on
    a rejected key; ``source`` names the service in errors (the credential, else
    ``inference``). ``transport`` replaces the network (tests pass an
    ``httpx.MockTransport``); ``timeout_s`` bounds each attempt. ``sleep``, ``rng`` and
    ``clock`` (for ``Retry-After`` dates) are injectable for tests.

    ``base_url`` must be ``https``, except on a local host, so the key never travels in the
    clear.
    """

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr,
        *,
        credential: str | None = None,
        source: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        rng: _Random | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        url = _checked_url(base_url)
        self._credential = credential
        self._source = source or credential or _DEFAULT_SOURCE
        secret = bearer_key(api_key, label=f"the {self._source} API key", hint=self._login_hint())
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._rng: _Random = rng if rng is not None else random.Random()  # noqa: S311 - jitter, not crypto
        self._clock = clock
        self._host = url.host
        self._client = httpx.Client(
            base_url=url,
            transport=transport,
            timeout=httpx.Timeout(timeout_s),
            headers={
                "Authorization": f"Bearer {secret}",
                "Accept": "application/json",
                "User-Agent": CLIENT_SOURCE,
            },
            follow_redirects=False,
        )

    @classmethod
    def for_judge(
        cls,
        config: JudgeConfig,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        rng: _Random | None = None,
        clock: Callable[[], float] = time.time,
    ) -> ChatClient:
        """A client for ``config``'s endpoint, with the key of its credential.

        The key comes from :func:`pup_finetune.credentials.resolve` (the environment, then
        the keychain); ``env`` replaces the environment. The other options are the
        constructor's. The credential spec (``fireworks`` by default) is registered when the
        providers package is imported, which importing this module does.
        """
        resolved = credentials.resolve(config.credential, env=os.environ if env is None else env)
        return cls(
            config.base_url,
            resolved.value,
            credential=config.credential,
            transport=transport,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            sleep=sleep,
            rng=rng,
            clock=clock,
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
        """Release the connection pool. Closing twice is a no-op."""
        self._client.close()

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        """One chat completion from ``model``; the first choice.

        ``tools`` are sent when there are any; ``temperature`` and ``max_tokens`` when set.
        Raises as the module docstring says.
        """
        body: dict[str, Any] = {"model": model, "messages": [dict(m) for m in messages]}
        if tools:
            body["tools"] = [dict(tool) for tool in tools]
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        response = self._post(body, what=f"a chat completion from {model}")
        return self._parse(response, model)

    # --- requests ---------------------------------------------------------------------

    def _post(self, body: dict[str, Any], *, what: str) -> httpx.Response:
        for attempt in range(1, self._max_attempts + 1):
            last = attempt == self._max_attempts
            try:
                response = self._client.post("chat/completions", json=body)
            except httpx.TransportError as exc:
                if last:
                    raise self._unreachable(what, exc, attempt) from exc
                self._sleep(self._backoff(attempt))
                continue
            except httpx.HTTPError as exc:  # e.g. a body that does not decode: not retried
                raise ExternalError(
                    f"{what} failed: {exception_reason(exc)}", source=self._source
                ) from exc
            status = response.status_code
            if response.is_success:
                return response
            if not _retryable(status):
                raise self._failure(what, response)
            if last:
                raise self._exhausted(what, response, attempt)
            self._sleep(self._wait(attempt, response))
        raise AssertionError("unreachable")  # pragma: no cover - the loop always returns or raises

    def _backoff(self, attempt: int) -> float:
        """The wait after failed ``attempt`` (1-based): doubling, capped, with jitter."""
        ceiling: float = min(MAX_WAIT_S, BASE_BACKOFF_S * 2.0 ** (attempt - 1))
        return ceiling * (0.5 + 0.5 * self._rng.random())

    def _wait(self, attempt: int, response: httpx.Response) -> float:
        retry_after = _retry_after_s(response.headers.get("retry-after"), self._clock)
        if retry_after is None:
            return self._backoff(attempt)
        return min(MAX_WAIT_S, retry_after)

    # --- errors -----------------------------------------------------------------------

    def _login_hint(self) -> str:
        if self._credential is not None:
            try:
                return credentials.login_hint(credentials.get_spec(self._credential))
            except UsageError:
                pass
        return f"check the API key configured for {self._source}"

    def _unreachable(self, what: str, exc: httpx.TransportError, attempts: int) -> TransientError:
        problem = (
            f"timed out after {self._timeout_s:g}s"
            if isinstance(exc, httpx.TimeoutException)
            else f"could not reach {self._host} ({exception_reason(exc)})"
        )
        return TransientError(
            f"{what} failed: {problem}, after {_attempts(attempts)}",
            source=self._source,
            hint=f"check the connection to {self._host} and retry",
        )

    def _failure(self, what: str, response: httpx.Response) -> ExternalError:
        status = response.status_code
        if status in (401, 403):
            problem = (
                "the API key is not valid"
                if status == 401
                else "the API key is not allowed to do this"
            )
            return ProviderAuthError(
                f"{what} failed: {problem} (HTTP {status})",
                source=self._source,
                hint=self._login_hint(),
            )
        detail = error_detail(response)
        return ExternalError(
            f"{what} failed with HTTP {status}" + (f": {detail}" if detail else ""),
            source=self._source,
            hint="check the model name and the request" if status in (400, 404, 422) else None,
        )

    def _exhausted(self, what: str, response: httpx.Response, attempts: int) -> ExternalError:
        status = response.status_code
        detail = error_detail(response)
        message = f"{what} failed with HTTP {status} after {_attempts(attempts)}" + (
            f": {detail}" if detail else ""
        )
        if status == 429:
            return ProviderCapacityError(
                message,
                source=self._source,
                hint="the endpoint is rate limiting or out of capacity; retry later",
            )
        return TransientError(message, source=self._source, hint="retry later")

    def _parse(self, response: httpx.Response, model: str) -> ChatCompletion:
        what = f"a chat completion from {model}"
        try:
            body = response.json()
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            raise ExternalError(
                f"{what} failed: the endpoint did not return JSON", source=self._source
            ) from exc
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            detail = error_detail(response)
            raise ExternalError(
                f"{what} failed: the answer has no choices" + (f": {detail}" if detail else ""),
                source=self._source,
                hint="check the model name and the endpoint",
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise ExternalError(
                f"{what} failed: the first choice has no message", source=self._source
            )
        message = {"role": "assistant", **message}
        usage = body.get("usage")
        answered = body.get("model")
        finish_reason = choice.get("finish_reason")
        return ChatCompletion(
            message=message,
            usage=dict(usage) if isinstance(usage, dict) else {},
            model=answered if isinstance(answered, str) and answered else model,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )


def _checked_url(base_url: str) -> httpx.URL:
    try:
        url = httpx.URL(base_url.rstrip("/"))
    except httpx.InvalidURL as exc:
        raise UsageError(f"{base_url!r} is not a valid URL") from exc
    secure = url.scheme == "https" or (url.scheme == "http" and url.host in _LOCAL_HOSTS)
    if secure and url.host:
        return url
    raise UsageError(
        f"{base_url!r} is not an https URL",
        hint="the API key is sent with every request, so the endpoint must use https",
    )


def _retryable(status: int) -> bool:
    return status in (408, 429) or status >= 500


def _attempts(count: int) -> str:
    return "1 attempt" if count == 1 else f"{count} attempts"


def _retry_after_s(value: str | None, clock: Callable[[], float]) -> float | None:
    """A ``Retry-After`` header in seconds from now: delta-seconds or an HTTP date.

    ``None`` when it is missing or unreadable; never negative.
    """
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            return None
        seconds = when.timestamp() - clock()
    if seconds != seconds:  # NaN
        return None
    return max(0.0, seconds)
