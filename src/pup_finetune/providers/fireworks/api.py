# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A small, read-only client for the Fireworks REST API, over ``httpx`` (no SDK).

It answers the two questions asked before anything is paid for: what Fireworks says about a
base model (:meth:`FireworksApi.model_info`, behind preflight) and whether an API key works
(:meth:`FireworksApi.list_accounts`, behind ``auth login fireworks``). It also reads a model's
record when one may already exist (:meth:`FireworksApi.get_model`, behind promotion).

The key is sent only as a ``Bearer`` header. It is registered for redaction when the client
is built, and every error message is redacted, so it never reaches output or logs. Failures
are :class:`ExternalError` with source ``fireworks``: :class:`FireworksAuthError` for a key
Fireworks rejects (HTTP 401 or 403), and a :class:`UsageError` for a model Fireworks does not
have (HTTP 404). Tests inject an ``httpx`` transport instead of reaching the network.
"""

from __future__ import annotations

import re
from types import TracebackType
from typing import Any, Final, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from pup_finetune.credentials import login_hint
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.http_common import CLIENT_SOURCE, bearer_key, error_detail, exception_reason
from pup_finetune.providers.base import ProviderAuthError
from pup_finetune.providers.fireworks import CREDENTIALS

__all__ = [
    "API_URL",
    "CLIENT_SOURCE",
    "CLIENT_SOURCE_HEADER",
    "DEFAULT_TIMEOUT_S",
    "FireworksApi",
    "FireworksAuthError",
    "ModelInfo",
    "checked_key",
]

API_URL: Final = "https://api.fireworks.ai"
DEFAULT_TIMEOUT_S: Final = 30.0
CLIENT_SOURCE_HEADER: Final = "X-Fireworks-Client-Source"
"""The header Fireworks reads the calling client from."""

SOURCE: Final = "fireworks"
_DEFAULT_ACCOUNT: Final = "fireworks"
_SEGMENT: Final = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_MODEL_NAME: Final = re.compile(rf"accounts/({_SEGMENT})/models/({_SEGMENT})")
_MODEL_ID: Final = re.compile(_SEGMENT)


class FireworksAuthError(ProviderAuthError):
    """Fireworks rejected the API key (HTTP 401) or refused it the request (HTTP 403)."""


class ModelInfo(BaseModel):
    """The fields of a Fireworks model that preflight reads; everything else is ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    name: str
    state: str | None = None
    hugging_face_url: str | None = Field(default=None, alias="huggingFaceUrl")
    training_context_length: int | None = Field(default=None, alias="trainingContextLength")
    supervised_lora_tunable: bool = Field(default=False, alias="supervisedLoraTunable")


def model_path(model: str) -> str:
    """The API path of ``model``: a resource name ``accounts/<a>/models/<m>``, or a bare id.

    A bare id is a model in the public ``fireworks`` account. Raises :class:`UsageError` for
    anything else, so a name can never reach outside the models collection.
    """
    if _MODEL_NAME.fullmatch(model):
        return f"/v1/{model}"
    if _MODEL_ID.fullmatch(model):
        return f"/v1/accounts/{_DEFAULT_ACCOUNT}/models/{model}"
    raise UsageError(
        f"{model!r} is not a Fireworks model name",
        hint="use a model id such as qwen3p8-27b, or accounts/<account>/models/<model>",
    )


class FireworksApi:
    """A read-only Fireworks REST client. Use it as a context manager, or call :meth:`close`.

    ``transport`` replaces the network (tests pass an ``httpx.MockTransport``);
    ``timeout_s`` bounds each request.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        base_url: str = API_URL,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        secret = checked_key(api_key)
        self._timeout_s = timeout_s
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout_s),
            headers={
                "Authorization": f"Bearer {secret}",
                "Accept": "application/json",
                "User-Agent": CLIENT_SOURCE,
                CLIENT_SOURCE_HEADER: CLIENT_SOURCE,
            },
            follow_redirects=False,
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

    def model_info(self, model: str) -> ModelInfo:
        """What Fireworks says about ``model`` (see :func:`model_path` for the accepted names).

        Raises :class:`UsageError` when Fireworks has no such model.
        """
        body = self._get(
            model_path(model),
            what=f"looking up the Fireworks model {model}",
            missing=f"Fireworks has no model {model!r}",
        )
        try:
            return ModelInfo.model_validate(body)
        except ValidationError as exc:
            raise ExternalError(
                f"the Fireworks API returned an unreadable model for {model} "
                f"({exc.error_count()} invalid field(s))",
                source=SOURCE,
            ) from exc

    def get_model(self, model: str) -> dict[str, Any] | None:
        """Fireworks' record of ``model`` (see :func:`model_path`), or ``None`` when it has none.

        Unlike :meth:`model_info` the record is returned as Fireworks sends it, and a missing
        model is not an error.
        """
        body = self._get(
            model_path(model), what=f"looking up the Fireworks model {model}", allow_missing=True
        )
        if body is None:
            return None
        if not isinstance(body, dict):
            raise ExternalError(
                f"the Fireworks API returned an unreadable model for {model}", source=SOURCE
            )
        return body

    def list_accounts(self) -> list[str]:
        """The ids of the accounts the key can use (the first page), e.g. ``["my-team"]``."""
        body = self._get("/v1/accounts", what="listing Fireworks accounts")
        accounts = (body.get("accounts") or []) if isinstance(body, dict) else None
        if not isinstance(accounts, list):
            raise ExternalError(
                "the Fireworks API returned an unreadable account list", source=SOURCE
            )
        ids: list[str] = []
        for account in accounts:
            name = account.get("name") if isinstance(account, dict) else None
            if isinstance(name, str) and name:
                ids.append(name.removeprefix("accounts/"))
        return ids

    def _get(
        self, path: str, *, what: str, missing: str | None = None, allow_missing: bool = False
    ) -> Any:
        try:
            response = self._client.get(path)
        except httpx.TimeoutException as exc:
            raise ExternalError(
                f"{what} timed out after {self._timeout_s:g}s",
                source=SOURCE,
                hint="check the connection to api.fireworks.ai and retry",
            ) from exc
        except httpx.HTTPError as exc:
            raise ExternalError(
                f"{what} failed: could not reach the Fireworks API ({exception_reason(exc)})",
                source=SOURCE,
                hint="check the connection to api.fireworks.ai and retry",
            ) from exc
        status = response.status_code
        if status in (401, 403):
            problem = (
                "the API key is not valid"
                if status == 401
                else "the API key is not allowed to do this"
            )
            raise FireworksAuthError(
                f"{what} failed: {problem} (HTTP {status})",
                source=SOURCE,
                hint=f"{login_hint(CREDENTIALS)} to a valid key",
            )
        if status == 404 and allow_missing:
            return None
        if status == 404 and missing is not None:
            raise UsageError(missing, hint="check the model name and that your account can use it")
        if not response.is_success:
            detail = error_detail(response)
            raise ExternalError(
                f"{what} failed with HTTP {status}" + (f": {detail}" if detail else ""),
                source=SOURCE,
                hint="retry later" if status == 429 or status >= 500 else None,
            )
        try:
            return response.json()
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            raise ExternalError(
                f"{what} failed: the Fireworks API did not return JSON", source=SOURCE
            ) from exc


def checked_key(api_key: SecretStr) -> str:
    """The value of ``api_key``, registered for redaction, once it is safe to send.

    Raises :class:`UsageError` for an empty or malformed key (see
    :func:`~pup_finetune.http_common.bearer_key`).
    """
    return bearer_key(api_key, label="the Fireworks API key", hint=login_hint(CREDENTIALS))
