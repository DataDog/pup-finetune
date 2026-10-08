# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""All Baseten traffic: REST over httpx, and the Loops SDK (imported here only).

- :class:`BasetenHTTP` is Baseten's REST routes. Every call sends ``Authorization: Bearer``,
  follows no redirects, has a timeout and reads at most 4 MiB. A failed call is an
  :class:`ExternalError` with source ``baseten``, the HTTP status and a fixed message, never
  the response body (a refused create quotes Baseten's reason, short and redacted). HTTP 401
  or 403 is a :class:`ProviderAuthError` and HTTP 429 a :class:`ProviderCapacityError`, for
  any call. It is a :class:`TransientError` for a call that is safe to repeat (a read, a
  deactivation, an update or a deletion) and failed in a way worth retrying (a timeout, a dropped
  connection, HTTP 408, 500, 502, 503 or 504), for a create Baseten accepted whose answer
  cannot be read, as what it created exists and may bill, and for a checkpoint deployment
  whose outcome is unknown, which the caller finds again by name.
- :func:`loops_sdk` imports the Loops SDK (``baseten-loops``, from the ``baseten`` extra) on
  first use, never when this module is imported, after refusing the environment variables
  that would make it send data elsewhere or reuse other resources.
  :func:`loops_service` opens a Loops session with the API key, base URL and availability
  passed explicitly, so none of them comes from the environment; :func:`loops_trainer`
  creates a LoRA trainer (a Loops run) in it and :func:`loops_sampling_client` connects to a
  sampler created over REST. :func:`loops_failure` maps a failed SDK call to an
  :class:`ExternalError` (auth and capacity as above), a :class:`TransientError` only for a
  call safe to repeat.
- :func:`to_loops_datum` converts a :class:`TokenizedExample` into the SDK's ``Datum``.

``tests/fakes/baseten.py`` has a fake of the SDK (``loops=``) and of the HTTP routes
(``transport=``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace, TracebackType
from typing import Any, Final, Protocol, Self, cast
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from pup_finetune.credentials import login_hint
from pup_finetune.errors import ExternalError, UsageError, install_hint, reinstall_hint
from pup_finetune.http_common import (
    CLIENT_SOURCE,
    DETAIL_LIMIT,
    bearer_key,
    error_detail,
    exception_reason,
)
from pup_finetune.providers.base import ProviderAuthError, ProviderCapacityError, TransientError
from pup_finetune.providers.baseten import CREDENTIALS
from pup_finetune.redaction import redact
from pup_finetune.rendering.tokens import TokenizedExample

__all__ = [
    "API_BASE",
    "HF_SECRET_NAME",
    "IGNORE_INDEX",
    "INSTALL_HINT",
    "MAX_CHECKPOINT_NAME_LEN",
    "SOURCE",
    "BasetenHTTP",
    "CreatedDeployment",
    "DeploymentRejected",
    "LoopsFuture",
    "LoopsModule",
    "LoopsSamplingClient",
    "LoopsService",
    "LoopsTrainingClient",
    "inference_url",
    "loops_failure",
    "loops_sampling_client",
    "loops_sdk",
    "loops_service",
    "loops_trainer",
    "to_loops_datum",
]

SOURCE: Final = "baseten"
API_BASE: Final = "https://api.baseten.co"
INSTALL_HINT: Final = install_hint("baseten")
MAX_CHECKPOINT_NAME_LEN: Final = 54
"""The longest checkpoint name pup-finetune saves on Loops."""
IGNORE_INDEX: Final = -100
"""The target a datum carries where its weight is 0: the cross-entropy ignore index."""
HF_SECRET_NAME: Final = "hf_access_token"  # noqa: S105 - the name of a secret, not one
"""The workspace secret a checkpoint deployment reads ``HF_TOKEN`` from, by default."""

_MAX_BODY: Final = 4 * 1024 * 1024
_CREATE_TIMEOUT: Final = 300.0
_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")
_TRANSIENT_STATUS: Final = frozenset({408, 500, 502, 503, 504})
_AUTH_STATUS: Final = frozenset({401, 403})
_LOGIN_HINT: Final = login_hint(CREDENTIALS)
_AUTH_HINT: Final = f"{_LOGIN_HINT} to a valid key"
_CREATED_HINT: Final = "it may have created something that bills; check the Baseten UI"
_TRAINERS: Final = "/trainers"
_REJECTED_ENV: Final = (
    "LOOPS_REQUEST_LOG",
    "LOOPS_REUSE_FROM_RUN_ID",
    "LOOPS_REUSE_FROM_SESSION_ID",
    "INTERNAL_LOOPS_BACKEND_VERSION",
    "LOOPS_TEAM",
)
_SDK_NAMES: Final = (
    "ServiceClient",
    "Datum",
    "ModelInput",
    "TensorData",
    "AdamParams",
    "SamplingClient",
    "SamplingParams",
    "ServerShutdownError",
)


class DeploymentRejected(ExternalError):
    """Baseten refused to create the deployment; nothing was created."""


@dataclass(frozen=True, slots=True)
class CreatedDeployment:
    """The model and deployment a checkpoint deployment created."""

    model_id: str
    deployment_id: str


class _HTTPStatusError(ExternalError):
    """Baseten answered with a non-2xx ``status``; ``detail`` is its reason for a refused create."""

    def __init__(self, label: str, status: int, detail: str = "") -> None:
        super().__init__(f"{label} returned HTTP {status}", source=SOURCE)
        self.status = status
        self.detail = detail


@contextmanager
def _rejected_on_4xx() -> Iterator[None]:
    """A 4xx answer to a create means Baseten created nothing: raise :class:`DeploymentRejected`.

    It quotes Baseten's reason. HTTP 401, 403 and 429 created nothing either; they stay auth
    and capacity errors.
    """
    try:
        yield
    except _HTTPStatusError as exc:
        if 400 <= exc.status < 500 and exc.status != 408:
            message = f"{exc.message}: {exc.detail}" if exc.detail else exc.message
            hint = "nothing was created"
            if "bdn volumes" in exc.detail.casefold():
                hint += "; ask Baseten to enable BDN volumes for your workspace"
            raise DeploymentRejected(message, source=SOURCE, hint=hint) from exc
        raise


@contextmanager
def _create_answer(label: str, *, create: bool = True) -> Iterator[None]:
    """Read the answer to ``label``, a create Baseten accepted when ``create`` is true.

    What the create made exists and may bill, so an answer that cannot be read is a
    :class:`TransientError`, never an error that reads as a refusal. Otherwise errors pass.
    """
    try:
        yield
    except TransientError:
        raise
    except (ExternalError, httpx.HTTPError) as exc:
        if not create:
            raise
        reason = exc.message if isinstance(exc, ExternalError) else exception_reason(exc)
        raise TransientError(
            f"{label} succeeded, but its answer could not be read ({reason})",
            source=SOURCE,
            hint=_CREATED_HINT,
        ) from exc


class BasetenHTTP:
    """Baseten's REST routes, authenticated with one API key."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            headers={
                "Authorization": f"Bearer {_checked_key(api_key)}",
                "Accept": "application/json",
                "User-Agent": CLIENT_SOURCE,
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- Loops and model management (REST) ---------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        label = "the Loops capabilities request"
        return _object(self._request("GET", f"{API_BASE}/v1/loops/capabilities", label), label)

    def get_checkpoint(self, checkpoint_path: str) -> dict[str, Any] | None:
        """The checkpoint record Baseten has for ``checkpoint_path``; ``None`` when it has none."""
        label = "the Loops checkpoint lookup"
        params = {"checkpoint_path": checkpoint_path}
        found = self._request("GET", f"{API_BASE}/v1/loops/checkpoints", label, params=params)
        records = _object(found, label).get("checkpoints")
        if not isinstance(records, list) or len(records) > 1:
            raise ExternalError(
                f"Baseten must return at most one checkpoint for {checkpoint_path}", source=SOURCE
            )
        return _object(records[0], label) if records else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """The Loops run, or ``None`` when Baseten no longer has it."""
        label = "the Loops run lookup"
        url = f"{API_BASE}/v1/loops/runs/{_id(run_id, 'run')}"
        found = self._request("GET", url, label, missing_ok=True)
        return None if found is None else _object(_object(found, label).get("run"), label)

    def deactivate_run(self, run_id: str) -> None:
        url = f"{API_BASE}/v1/loops/runs/{_id(run_id, 'run')}/deactivate"
        self._request("POST", url, "the Loops run deactivation", body={}, idempotent=True)

    def create_sampler(
        self,
        *,
        session_id: str,
        max_seq_len: int,
        model_path: str | None = None,
        base_model: str | None = None,
    ) -> dict[str, Any]:
        """A dedicated sampler serving ``model_path`` (a checkpoint) or ``base_model``."""
        if (model_path is None) == (base_model is None):
            raise ValueError("pass exactly one of model_path and base_model")
        label = "the Loops sampler creation"
        target = {"model_path": model_path} if base_model is None else {"base_model": base_model}
        body = {
            "session_id": _id(session_id, "session"),
            **target,
            "max_seq_length": max_seq_len,
            "availability_model": "dedicated",
        }
        with _rejected_on_4xx():
            found = self._request(
                "POST", f"{API_BASE}/v1/loops/samplers", label, body=body, timeout=_CREATE_TIMEOUT
            )
        with _create_answer(label):
            return _object(_object(found, label).get("sampler"), label)

    def get_deployment(self, model_id: str, deployment_id: str) -> dict[str, Any] | None:
        """The model deployment, or ``None`` when Baseten no longer has it."""
        label = "the deployment lookup"
        found = self._request(
            "GET", _deployment_url(model_id, deployment_id), label, missing_ok=True
        )
        return None if found is None else _object(found, label)

    def deactivate_deployment(self, model_id: str, deployment_id: str) -> None:
        url = f"{_deployment_url(model_id, deployment_id)}/deactivate"
        self._request("POST", url, "the deployment deactivation", idempotent=True)

    def list_deployments(self, model_id: str) -> list[dict[str, Any]]:
        """The model's deployments; none when Baseten no longer has the model."""
        label = "the deployment listing"
        url = f"{API_BASE}/v1/models/{_id(model_id, 'model')}/deployments"
        found = self._request("GET", url, label, missing_ok=True)
        return [] if found is None else _records(found, "deployments", label)

    def deployment_config(self, model_id: str, deployment_id: str) -> dict[str, Any]:
        """The deployment's parsed config (``training_checkpoints``, ``resources``...)."""
        label = "the deployment config lookup"
        url = f"{_deployment_url(model_id, deployment_id)}/config"
        found = self._request("GET", url, label, params={"output_format": "parsed"})
        config = _object(found, label).get("config")
        return {} if config is None else _object(config, label)

    def set_autoscaling(
        self, model_id: str, deployment_id: str, *, min_replica: int, max_replica: int
    ) -> None:
        url = f"{_deployment_url(model_id, deployment_id)}/autoscaling_settings"
        body = {"min_replica": min_replica, "max_replica": max_replica}
        self._request("PATCH", url, "the autoscaling update", body=body, idempotent=True)

    # --- Dedicated checkpoint deployment ----------------------------------------------

    def instance_types(self) -> list[dict[str, Any]]:
        """Baseten's instance types (``id``, ``name``, ``gpu_type``, ``gpu_count``...)."""
        label = "the instance type listing"
        found = self._request("GET", f"{API_BASE}/v1/instance_types", label)
        return _records(found, "instance_types", label)

    def list_models(self, *, name: str | None = None) -> list[dict[str, Any]]:
        """The workspace's models, or only those named exactly ``name``."""
        label = "the model listing"
        params = None if name is None else {"name": name}
        models = _records(
            self._request("GET", f"{API_BASE}/v1/models", label, params=params), "models", label
        )
        return models if name is None else [model for model in models if model.get("name") == name]

    def deploy_checkpoints(
        self,
        *,
        checkpoint_id: str,
        model_name: str,
        instance_type_id: str,
        hf_secret_name: str = HF_SECRET_NAME,
    ) -> CreatedDeployment:
        """Deploy the sampler checkpoint ``checkpoint_id`` as a new model named ``model_name``.

        Sent once: a repeat would create another deployment. A 4xx is a
        :class:`DeploymentRejected`. Any other failure, or an answer without valid ids, is a
        :class:`TransientError`, since the model may exist: find it with :meth:`list_models`.
        """
        label = "the checkpoint deployment"
        body = {
            "checkpoint_ids": [checkpoint_id],
            "model_name": model_name,
            "instance_type_id": instance_type_id,
            "hf_secret_name": hf_secret_name,
        }
        url = f"{API_BASE}/v1/loops/checkpoints/deploy"
        with _rejected_on_4xx():
            found = self._request(
                "POST", url, label, body=body, timeout=_CREATE_TIMEOUT, read_back=True
            )
        with _create_answer(label):
            created = _object(found, label)
            return CreatedDeployment(
                model_id=_id(created.get("model_id"), "model"),
                deployment_id=_id(created.get("deployment_id"), "deployment"),
            )

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """The model, or ``None`` when Baseten no longer has it."""
        label = "the model lookup"
        url = f"{API_BASE}/v1/models/{_id(model_id, 'model')}"
        found = self._request("GET", url, label, missing_ok=True)
        return None if found is None else _object(found, label)

    def delete_model(self, model_id: str) -> bool:
        """Delete the model and its deployments; ``False`` when Baseten no longer has it."""
        label = "the model deletion"
        url = f"{API_BASE}/v1/models/{_id(model_id, 'model')}"
        found = self._request("DELETE", url, label, missing_ok=True, idempotent=True)
        if found is None:
            return False
        if _object(found, label).get("deleted") is not True:
            raise ExternalError(f"{label} returned an unexpected response", source=SOURCE)
        return True

    # --- transport -----------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        label: str,
        *,
        body: Any = None,
        params: Mapping[str, str] | None = None,
        timeout: float | None = None,
        missing_ok: bool = False,
        idempotent: bool | None = None,
        read_back: bool = False,
    ) -> Any:
        """The parsed JSON of a 2xx answer; ``None`` for a 404 when ``missing_ok``.

        ``idempotent`` (by default, whether ``method`` is GET) says whether the call is safe
        to repeat, so that its retryable failures are :class:`TransientError`. A call that is
        not is a create: once Baseten accepted it, an answer that cannot be read is one too.
        With ``read_back`` (the caller reads back what a failed create made), so is any
        failure but a 4xx.
        """
        repeatable = method == "GET" if idempotent is None else idempotent
        try:
            with self._client.stream(
                method,
                url,
                json=body,
                params=params,
                timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout,
            ) as response:
                status = response.status_code
                if missing_ok and status == 404:
                    return None
                if status in _AUTH_STATUS:
                    raise ProviderAuthError(
                        f"{label} returned HTTP {status}", source=SOURCE, hint=_AUTH_HINT
                    )
                if status == 429:
                    raise ProviderCapacityError(
                        f"{label} returned HTTP 429",
                        source=SOURCE,
                        hint=_capacity_hint(response),
                    )
                if repeatable and status in _TRANSIENT_STATUS:
                    raise TransientError(
                        f"{label} returned HTTP {status}", source=SOURCE, hint="retry later"
                    )
                # A 408 is a timeout: what it did is unknown, so it is no refusal.
                refused = 400 <= status < 500 and status != 408
                if read_back and not response.is_success and not refused:
                    raise TransientError(
                        f"{label} returned HTTP {status}", source=SOURCE, hint=_CREATED_HINT
                    )
                if not response.is_success:
                    detail = _refusal_detail(response) if refused and not repeatable else ""
                    raise _HTTPStatusError(label, status, detail)
                with _create_answer(label, create=not repeatable):
                    return _parse(_read_capped(response, label), label)
        except httpx.HTTPError as exc:
            message = f"could not reach Baseten for {label} ({exception_reason(exc)})"
            if not repeatable:
                hint = "the request may have reached Baseten; check before retrying"
                unknown = TransientError if read_back else ExternalError
                raise unknown(message, source=SOURCE, hint=hint) from exc
            error = TransientError if isinstance(exc, httpx.TransportError) else ExternalError
            raise error(
                message, source=SOURCE, hint="check your network connection and retry"
            ) from exc


def _capacity_hint(response: httpx.Response) -> str:
    """When to retry after an HTTP 429, from its ``Retry-After`` seconds when it has them."""
    retry_after = response.headers.get("Retry-After", "")
    return f"retry after {retry_after} s" if retry_after.isdigit() else "retry later"


def _refusal_detail(response: httpx.Response) -> str:
    """Baseten's reason in the body of a refused create, short and redacted; may be empty."""
    try:
        content = _read_capped(response, "the refusal")
    except (ExternalError, httpx.HTTPError):
        return ""
    content_type = response.headers.get("content-type", "")
    body = httpx.Response(
        response.status_code, headers={"content-type": content_type}, content=content
    )
    return _response_detail(body)


def _read_capped(response: httpx.Response, label: str) -> bytes:
    payload = bytearray()
    for chunk in response.iter_bytes():
        payload += chunk
        if len(payload) > _MAX_BODY:
            raise ExternalError(f"{label} returned more than 4 MiB", source=SOURCE)
    return bytes(payload)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _parse(payload: bytes, label: str) -> Any:
    if not payload.strip():
        return {}
    try:
        return json.loads(payload, parse_constant=_reject_constant)
    except ValueError:
        raise ExternalError(f"{label} returned malformed JSON", source=SOURCE) from None


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExternalError(f"{label} returned an unexpected response", source=SOURCE)
    return value


def _records(found: Any, key: str, label: str) -> list[dict[str, Any]]:
    records = _object(found, label).get(key)
    if not isinstance(records, list):
        raise ExternalError(f"{label} returned an unexpected response", source=SOURCE)
    return [_object(record, label) for record in records]


def _id(value: object, kind: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ExternalError(f"invalid Baseten {kind} id", source=SOURCE)
    return value


def _deployment_url(model_id: str, deployment_id: str) -> str:
    return (
        f"{API_BASE}/v1/models/{_id(model_id, 'model')}"
        f"/deployments/{_id(deployment_id, 'deployment')}"
    )


def inference_url(url: str) -> str:
    """``url`` without a trailing slash; refuses anything but an HTTPS Baseten inference URL."""
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme == "https"
            and (parts.hostname or "").endswith(".api.baseten.co")
            and parts.port in (None, 443)
            and not (parts.username or parts.password or parts.query or parts.fragment)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ExternalError("refusing to send the API key to a non-Baseten URL", source=SOURCE)
    return url.rstrip("/")


def _checked_key(api_key: SecretStr) -> str:
    """``api_key``'s value, registered for redaction, once it is safe to send in a header.

    Raises :class:`UsageError` for an empty key, which would make the SDK read
    ``BASETEN_API_KEY``, or a malformed one (see :func:`~pup_finetune.http_common.bearer_key`).
    """
    return bearer_key(api_key, label="the Baseten API key", hint=_LOGIN_HINT)


def _clean(text: str) -> str:
    """The first line of ``text``, redacted and short, as :func:`error_detail` quotes a body."""
    lines = redact(text).strip().splitlines()
    return lines[0][:DETAIL_LIMIT] if lines else ""


# --- Loops SDK ---------------------------------------------------------------------------


class LoopsFuture(Protocol):
    """An SDK future (``APIFuture``): the result of a call that runs on the trainer."""

    def result(self, timeout: float | None = None) -> Any: ...


class LoopsTrainingClient(Protocol):
    """The parts of the SDK's ``TrainingClient`` (one Loops run) used here.

    Closing it releases the HTTP clients only: the run keeps its GPUs until it is deactivated
    through :meth:`BasetenHTTP.deactivate_run`.
    """

    @property
    def run_id(self) -> str | None: ...

    def forward_backward(self, data: list[Any], loss_fn: str) -> LoopsFuture: ...

    def forward(self, data: list[Any], loss_fn: str) -> LoopsFuture: ...

    def optim_step(self, adam_params: Any) -> LoopsFuture: ...

    def save_state(self, name: str) -> LoopsFuture: ...

    def save_weights_for_sampler(self, name: str) -> LoopsFuture: ...

    def load_state(self, path: str) -> LoopsFuture: ...

    def load_state_with_optimizer(self, path: str) -> LoopsFuture: ...

    def close(self) -> None: ...


class LoopsSamplingClient(Protocol):
    """The parts of the SDK's ``SamplingClient`` used here.

    Closing it releases its HTTP client only: the sampler keeps its GPU until its deployment
    is deactivated through :meth:`BasetenHTTP.deactivate_deployment`.
    """

    def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any: ...

    def close(self) -> None: ...


class LoopsService(Protocol):
    """The parts of the SDK's ``ServiceClient`` (one Loops session) used here."""

    @property
    def session_id(self) -> str: ...

    def create_lora_training_client(
        self,
        *,
        base_model: str,
        rank: int,
        replicas: int,
        seed: int | None,
        max_seq_len: int | None,
        with_sampler: bool,
        name: str | None,
    ) -> LoopsTrainingClient: ...


class LoopsModule(Protocol):
    """The parts of ``baseten.loops`` used here; tests pass a fake of the same shape."""

    @property
    def ServiceClient(self) -> Callable[..., LoopsService]: ...
    @property
    def Datum(self) -> Callable[..., Any]: ...
    @property
    def ModelInput(self) -> Any: ...
    @property
    def TensorData(self) -> Callable[..., Any]: ...
    @property
    def AdamParams(self) -> Callable[..., Any]: ...
    @property
    def SamplingClient(self) -> Callable[..., LoopsSamplingClient]: ...
    @property
    def SamplingParams(self) -> Callable[..., Any]: ...
    @property
    def BasetenDeployment(self) -> Callable[..., Any]: ...
    @property
    def ServerShutdownError(self) -> type[Exception]: ...


def _check_environment() -> None:
    """Refuse the SDK's env overrides that would send data elsewhere or reuse other resources.

    The SDK reads ``LOOPS_REQUEST_LOG`` when it is imported, so this runs first.
    """
    for name in _REJECTED_ENV:
        # The SDK honors some of these even when set to an empty string.
        if name in os.environ:
            raise UsageError(
                f"{name} is set; pup-finetune does not run Baseten Loops with it",
                hint=f"unset {name} and retry",
            )
    base_url = os.environ.get("LOOPS_BASE_URL")
    if base_url and base_url.rstrip("/") != API_BASE:
        raise UsageError(
            f"LOOPS_BASE_URL must be unset or {API_BASE}", hint="unset LOOPS_BASE_URL and retry"
        )


def loops_sdk(loops: LoopsModule | None = None) -> LoopsModule:
    """``loops``, else the imported SDK; first refuses environment overrides either way."""
    _check_environment()
    if loops is not None:
        return loops
    try:
        import baseten.loops as sdk
        from baseten.loops.sampling_client import BasetenDeployment
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] != "baseten":
            raise _broken_sdk(exc) from exc
        raise UsageError(
            "training on Baseten needs the Baseten Loops SDK, which is not installed",
            hint=INSTALL_HINT,
        ) from exc
    except ImportError as exc:
        raise _broken_sdk(exc) from exc
    names = {name: getattr(sdk, name) for name in _SDK_NAMES}
    return cast(LoopsModule, SimpleNamespace(**names, BasetenDeployment=BasetenDeployment))


def _broken_sdk(exc: ImportError) -> UsageError:
    return UsageError(
        f"the Baseten Loops SDK is installed but failed to load: {redact(str(exc))}",
        hint=reinstall_hint("baseten"),
    )


def loops_service(api_key: SecretStr, *, loops: LoopsModule | None = None) -> LoopsService:
    """A new Loops session for ``api_key``, on dedicated capacity.

    The key, the base URL and the availability are passed explicitly, so the SDK never reads
    them from the environment. Raises :class:`UsageError` when the SDK is missing, the key is
    malformed or the environment is refused, and :class:`ExternalError` when Baseten refuses
    the session (:class:`ProviderAuthError` for the key, :class:`ProviderCapacityError` for
    HTTP 429).
    """
    sdk = loops_sdk(loops)
    secret = _checked_key(api_key)
    try:
        return sdk.ServiceClient(api_key=secret, base_url=API_BASE, availability_model="dedicated")
    except Exception as exc:
        raise _creation_error(exc, "Loops session") from exc


def loops_trainer(
    service: LoopsService,
    *,
    base_model: str,
    rank: int,
    seed: int,
    max_seq_len: int,
    name: str,
) -> LoopsTrainingClient:
    """A dedicated LoRA trainer (a Loops run) in ``service``'s session, once it is ready. Paid.

    A failure that may leave a run behind (anything but a 4xx answer to the create itself, as
    the SDK calls the new run before it returns) says how to find it; a refusal for lack of
    capacity says nothing was left running.
    """
    session_id = _id(service.session_id, "session")
    try:
        trainer = service.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            replicas=1,
            seed=seed,
            max_seq_len=max_seq_len,
            with_sampler=False,
            name=name,
        )
    except Exception as exc:
        error = _creation_error(exc, "trainer", route=_TRAINERS)
        if not _refused(_http_response(exc), _TRAINERS):
            error.hint = "; ".join(filter(None, (_orphan_hint(session_id), error.hint)))
        elif isinstance(error, ProviderCapacityError):
            error.hint = f"nothing was left running; {error.hint}"
        raise error from exc
    except BaseException as exc:  # e.g. Ctrl-C while the trainer provisions
        exc.add_note(_orphan_hint(session_id))
        raise
    run_id = trainer.run_id
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        trainer.close()
        raise ExternalError(
            "Baseten returned no valid run id for the trainer",
            source=SOURCE,
            hint=_orphan_hint(session_id),
        )
    return trainer


def _orphan_hint(session_id: str) -> str:
    return f"a run may remain in Loops session {session_id}; deactivate it in the Baseten UI"


def loops_sampling_client(
    api_key: SecretStr,
    sampler: Mapping[str, Any],
    *,
    base_model: str,
    session_id: str,
    checkpoint_path: str | None,
    ready_timeout: float,
    timeout: float,
    loops: LoopsModule | None = None,
) -> LoopsSamplingClient:
    """An SDK client of ``sampler`` (as :meth:`BasetenHTTP.create_sampler` returned it).

    Its first sample waits up to ``ready_timeout`` seconds for the sampler to be ready, and
    a sample that takes over ``timeout`` seconds fails with a :class:`TimeoutError`;
    ``checkpoint_path`` routes every sample to that checkpoint, else the base model serves.
    """
    sdk = loops_sdk(loops)
    base_url = sampler.get("base_url")
    if not isinstance(base_url, str):
        raise ExternalError("Baseten described the sampler without its URL", source=SOURCE)
    deployment = sdk.BasetenDeployment(
        base_url=inference_url(base_url),
        model_id=_id(sampler.get("model_id"), "model"),
        deployment_id=_id(sampler.get("deployment_id"), "deployment"),
        api_base_url=API_BASE,
    )
    return sdk.SamplingClient(
        api_key=_checked_key(api_key),
        base_model=base_model,
        deployment=deployment,
        sampler_id=_id(sampler.get("id"), "sampler"),
        session_id=_id(session_id, "session"),
        checkpoint_path=checkpoint_path,
        ready_timeout=ready_timeout,
        timeout=timeout,
    )


def loops_failure(
    exc: Exception, label: str, loops: LoopsModule, *, repeatable: bool
) -> ExternalError:
    """``exc``, raised by an SDK call or its future, as an :class:`ExternalError`.

    It names only the HTTP status, read from ``exc`` or an exception it was raised from, and
    the error type. HTTP 429 is a :class:`ProviderCapacityError` and 401 or 403 a
    :class:`ProviderAuthError`. It is a :class:`TransientError` when the call is
    ``repeatable`` (changes nothing when repeated) and failed in a way worth retrying: a
    timeout, a dropped connection, a trainer shutting down or a retryable status.
    """
    response = _http_response(exc)
    status = None if response is None else response.status_code
    reason = f"HTTP {status}, {type(exc).__name__}" if status else type(exc).__name__
    message = f"{label} failed ({reason})"
    if response is not None and status == 429:
        return ProviderCapacityError(message, source=SOURCE, hint=_capacity_hint(response))
    if status in _AUTH_STATUS:
        return ProviderAuthError(message, source=SOURCE, hint=_AUTH_HINT)
    retryable = (httpx.TransportError, TimeoutError, ConnectionError, loops.ServerShutdownError)
    transient = status in _TRANSIENT_STATUS or isinstance(exc, retryable)
    error = TransientError if repeatable and transient else ExternalError
    return error(message, source=SOURCE)


def _creation_error(exc: Exception, what: str, *, route: str | None = None) -> ExternalError:
    """``exc``, raised by the SDK while creating the Baseten ``what``, as an error to raise.

    The HTTP status comes from ``exc`` or an exception it was raised from: 429 is a
    :class:`ProviderCapacityError`, 401 or 403 a :class:`ProviderAuthError`. It reads as a
    refusal only when :func:`_refused` says so.
    """
    response = _http_response(exc)
    if response is None:
        return ExternalError(
            f"creating the Baseten {what} failed ({exception_reason(exc)})", source=SOURCE
        )
    status = response.status_code
    detail = _response_detail(response)
    suffix = f": {detail}" if detail else ""
    if status == 429:
        return ProviderCapacityError(
            f"Baseten has no capacity for the {what} (HTTP 429){suffix}",
            source=SOURCE,
            hint=_capacity_hint(response),
        )
    if _refused(response, route):
        message = f"Baseten refused to create the {what} (HTTP {status}){suffix}"
    else:
        message = f"creating the Baseten {what} failed (HTTP {status}){suffix}"
    if status in _AUTH_STATUS:
        return ProviderAuthError(message, source=SOURCE, hint=_AUTH_HINT)
    return ExternalError(message, source=SOURCE)


def _refused(response: httpx.Response | None, route: str | None) -> bool:
    """Whether ``response`` refused a create, which then made nothing: a 4xx but 408, to
    ``POST <route>`` itself when ``route`` is given."""
    if response is None or not 400 <= response.status_code < 500 or response.status_code == 408:
        return False
    if route is None:
        return True
    try:
        request = response.request
    except RuntimeError:  # a response without its request
        return False
    return request.method == "POST" and request.url.path.endswith(route)


def _http_response(exc: BaseException) -> httpx.Response | None:
    """The HTTP response of ``exc``, else of the nearest exception it was raised from."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        response = getattr(current, "response", None)
        if isinstance(response, httpx.Response):
            return response
        current = current.__cause__ or current.__context__
    return None


def _response_detail(response: httpx.Response) -> str:
    """Baseten's ``detail`` in an error body, else what :func:`error_detail` reads."""
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail.strip():
        return _clean(detail)
    return error_detail(response)


def to_loops_datum(example: TokenizedExample, loops: LoopsModule) -> Any:
    """``example`` as a Loops ``Datum`` for the cross-entropy loss.

    ``example`` is already shifted, so its arrays go across position by position, with the
    target :data:`IGNORE_INDEX` wherever the weight is 0.
    """
    size = [len(example.input_tokens)]
    targets = [
        token if weight > 0 else IGNORE_INDEX
        for token, weight in zip(example.target_tokens, example.weights, strict=True)
    ]
    return loops.Datum(
        model_input=loops.ModelInput.from_ints(list(example.input_tokens)),
        loss_fn_inputs={
            "target_tokens": loops.TensorData(data=targets, dtype="int64", shape=size),
            "weights": loops.TensorData(data=list(example.weights), dtype="float32", shape=size),
        },
    )
