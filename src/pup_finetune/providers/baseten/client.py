# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""All Baseten traffic: REST and GraphQL over httpx, and the Loops SDK (imported here only).

Every call sends ``Authorization: Api-Key``, follows no redirects, has a timeout and reads at
most 4 MiB. A failed call becomes an :class:`ExternalError` with the HTTP status and a fixed
message, never the response body or headers. The SDK is optional and imported lazily, after
refusing the environment variables that would make it send data elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import SimpleNamespace, TracebackType
from typing import Any, Protocol, Self, cast
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.baseten.datums import shift
from pup_finetune.providers.baseten.registration import LOOPS_INSTALL_HINT, TESTED_LOOPS_VERSION
from pup_finetune.redaction import MASK, redact
from pup_finetune.rendering.types import TokenDatum
from pup_finetune.training.backend import StepResult

__all__ = [
    "API_BASE",
    "APP_BASE",
    "CHECKPOINT_URI",
    "TESTED_LOOPS_VERSION",
    "BasetenHTTP",
    "DeploymentRejected",
    "LoopsModule",
    "LoopsTrainer",
    "inference_url",
    "load_sampler_checkpoint",
    "loops_sdk",
    "open_trainer",
]

API_BASE = "https://api.baseten.co"
APP_BASE = "https://app.baseten.co"
CHECKPOINT_URI = re.compile(r"bt://loops:([A-Za-z0-9_-]+)/sampler_weights/([A-Za-z0-9_.-]+)")

_GRAPHQL_URL = f"{APP_BASE}/graphql/"
_MAX_BODY = 4 * 1024 * 1024
_CREATE_TIMEOUT = 300.0
_GRAPHQL_TIMEOUT = 120.0
_INFERENCE_TIMEOUT = 300.0
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DETAIL_LIMIT = 300
_REJECTED_ENV = (
    "LOOPS_REQUEST_LOG",
    "LOOPS_REUSE_FROM_RUN_ID",
    "LOOPS_REUSE_FROM_SESSION_ID",
    "INTERNAL_LOOPS_BACKEND_VERSION",
    "LOOPS_TEAM",
)
_SDK_NAMES = (
    "ServiceClient",
    "Datum",
    "ModelInput",
    "TensorData",
    "AdamParams",
    "SamplingParams",
    "SamplingClient",
    "ServerShutdownError",
)

# Tests replace this to skip retry backoff.
_sleep: Callable[[float], None] = time.sleep

# The GraphQL documents exactly as truss 0.18.32 sends them.
_CREATE_FROM_TEMPLATE = """
        mutation ($request: CreateModelVersionFromInferenceTemplateRequest!) {
            create_model_version_from_inference_template(request: $request) {
                model_version {
                    id
                    name
                    model_id
                }
                truss_config
            }
        }
        """
_LISTED_INSTANCES = """
        query Instances {
            listedInstances: listed_instances {
                id
                name
                millicpuLimit: millicpu_limit
                memoryLimit: memory_limit
                gpuCount: gpu_count
                gpuType: gpu_type
                gpuMemory: gpu_memory
                default
                displayName: display_name
                nodeCount: node_count
                price
                limitedCapacity: limited_capacity
            }
        }
        """


class DeploymentRejected(ExternalError):
    """Baseten refused to create the deployment; nothing was created."""


class _HTTPStatusError(ExternalError):
    """Baseten answered with a non-2xx ``status``."""

    def __init__(self, label: str, status: int) -> None:
        super().__init__(f"{label} returned HTTP {status}", source="baseten")
        self.status = status


@contextmanager
def _rejected_on_4xx() -> Iterator[None]:
    """A 4xx answer to a create means Baseten created nothing: raise :class:`DeploymentRejected`."""
    try:
        yield
    except _HTTPStatusError as exc:
        if 400 <= exc.status < 500:
            raise DeploymentRejected(exc.message, source="baseten") from exc
        raise


class BasetenHTTP:
    """Baseten's REST, GraphQL and OpenAI-compatible routes, authenticated with one API key."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            headers={
                "Authorization": f"Api-Key {api_key.get_secret_value()}",
                "Accept": "application/json",
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

    def get_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """The one checkpoint record Baseten has for ``checkpoint_path``."""
        label = "the Loops checkpoint lookup"
        params = {"checkpoint_path": checkpoint_path}
        found = self._request("GET", f"{API_BASE}/v1/loops/checkpoints", label, params=params)
        records = _object(found, label).get("checkpoints")
        if not isinstance(records, list) or len(records) != 1:
            raise ExternalError(
                f"Baseten must return exactly one checkpoint for {checkpoint_path}",
                source="baseten",
            )
        return _object(records[0], label)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """The Loops run, or ``None`` when Baseten no longer has it."""
        label = "the Loops run lookup"
        url = f"{API_BASE}/v1/loops/runs/{_id(run_id, 'run')}"
        found = self._request("GET", url, label, missing_ok=True)
        return None if found is None else _object(_object(found, label).get("run"), label)

    def deactivate_run(self, run_id: str) -> None:
        url = f"{API_BASE}/v1/loops/runs/{_id(run_id, 'run')}/deactivate"
        self._request("POST", url, "the Loops run deactivation", body={})

    def create_session(self) -> str:
        """A new Loops session's id."""
        label = "the Loops session creation"
        found = self._request(
            "POST", f"{API_BASE}/v1/loops/sessions", label, timeout=_CREATE_TIMEOUT
        )
        return _id(_object(_object(found, label).get("session"), label).get("id"), "session")

    def create_sampler(
        self, *, session_id: str, model_path: str, max_seq_len: int
    ) -> dict[str, Any]:
        """A dedicated sampler serving ``model_path``, as Baseten describes it."""
        label = "the Loops sampler creation"
        body = {
            "session_id": _id(session_id, "session"),
            "model_path": model_path,
            "max_seq_length": max_seq_len,
            "availability_model": "dedicated",
        }
        with _rejected_on_4xx():
            found = self._request(
                "POST", f"{API_BASE}/v1/loops/samplers", label, body=body, timeout=_CREATE_TIMEOUT
            )
        return _object(_object(found, label).get("sampler"), label)

    def get_deployment(self, model_id: str, deployment_id: str) -> dict[str, Any] | None:
        """The model deployment, or ``None`` when Baseten no longer has it."""
        label = "the deployment lookup"
        found = self._request(
            "GET", _deployment_url(model_id, deployment_id), label, missing_ok=True
        )
        return None if found is None else _object(found, label)

    def activate_deployment(self, model_id: str, deployment_id: str) -> None:
        url = f"{_deployment_url(model_id, deployment_id)}/activate"
        self._request("POST", url, "the deployment activation")

    def deactivate_deployment(self, model_id: str, deployment_id: str) -> None:
        url = f"{_deployment_url(model_id, deployment_id)}/deactivate"
        self._request("POST", url, "the deployment deactivation")

    # --- Dedicated checkpoint deployment (GraphQL) -------------------------------------

    def instance_types(self) -> list[dict[str, Any]]:
        """Baseten's instance types, with the GraphQL field names (``gpuType``, ``nodeCount``...)."""
        label = "the instance type listing"
        document = self._graphql(_LISTED_INSTANCES, label)
        if document.get("errors"):
            raise ExternalError(f"{label} returned GraphQL errors", source="baseten")
        listed = _object(document.get("data"), label).get("listedInstances")
        if not isinstance(listed, list):
            raise ExternalError(f"{label} returned an unexpected response", source="baseten")
        return [_object(item, label) for item in listed]

    def create_checkpoint_deployment(
        self, *, checkpoint_id: str, model_name: str, instance_type_id: str
    ) -> dict[str, Any]:
        """``{"model_version": {"id", "name", "model_id"}, "truss_config"}`` of the new model."""
        label = "the checkpoint deployment creation"
        request = {
            "metadata": {"oracle_name": model_name},
            "weights_sources": [
                {
                    "weight_source_type": "B10_LOOPS_CHECKPOINTING",
                    "b10_loops_checkpoint_weights_source": {
                        "checkpoint": {"loops_checkpoint_id": checkpoint_id}
                    },
                }
            ],
            "inference_stack": {"stack_type": "VLLM", "environment_variables": []},
            "instance_type_id": instance_type_id,
            "dry_run": False,
        }
        with _rejected_on_4xx():
            document = self._graphql(_CREATE_FROM_TEMPLATE, label, {"request": request})
        errors = document.get("errors")
        data = document.get("data")
        created = (
            data.get("create_model_version_from_inference_template")
            if isinstance(data, dict)
            else None
        )
        # GraphQL can report errors next to a created model; only a null mutation created nothing.
        if errors and created is None:
            raise DeploymentRejected(
                f"Baseten refused to create the deployment: {self._graphql_error(errors)}",
                source="baseten",
            )
        if not isinstance(created, dict):
            raise ExternalError(
                f"{label} returned no deployment; its outcome is unknown",
                source="baseten",
                hint=f"look for a model named {model_name} in the Baseten UI",
            )
        return created

    # --- OpenAI-compatible routes of a deployment --------------------------------------

    def list_served_models(self, openai_base_url: str) -> list[dict[str, Any]]:
        label = "the served model listing"
        url = f"{inference_url(openai_base_url)}/models"
        # The first request to a deployment scaled to zero waits for its cold start.
        found = self._request("GET", url, label, timeout=_INFERENCE_TIMEOUT)
        data = _object(found, label).get("data")
        if not isinstance(data, list):
            raise ExternalError(f"{label} returned an unexpected response", source="baseten")
        return [_object(item, label) for item in data]

    def chat_completion(self, openai_base_url: str, body: Mapping[str, Any]) -> dict[str, Any]:
        label = "the chat completion"
        url = f"{inference_url(openai_base_url)}/chat/completions"
        found = self._request("POST", url, label, body=dict(body), timeout=_INFERENCE_TIMEOUT)
        return _object(found, label)

    # --- transport -----------------------------------------------------------------------

    def _graphql(
        self, query: str, label: str, variables: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = dict(variables)
        found = self._request("POST", _GRAPHQL_URL, label, body=payload, timeout=_GRAPHQL_TIMEOUT)
        return _object(found, label)

    def _graphql_error(self, errors: Any) -> str:
        """The first error's description or message, as truss reports it."""
        first = errors[0] if isinstance(errors, list) and errors else None
        if not isinstance(first, dict):
            return "no reason given"
        extensions = first.get("extensions")
        description = extensions.get("description") if isinstance(extensions, dict) else None
        message = description if description is not None else first.get("message")
        return _clean(str(message), self._api_key) if message else "no reason given"

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
    ) -> Any:
        """The parsed JSON of a 2xx answer; ``None`` for a 404 when ``missing_ok``."""
        try:
            with self._client.stream(
                method,
                url,
                json=body,
                params=params,
                timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout,
            ) as response:
                if missing_ok and response.status_code == 404:
                    return None
                if not response.is_success:
                    raise _HTTPStatusError(label, response.status_code)
                payload = _read_capped(response, label)
        except httpx.HTTPError as exc:
            raise ExternalError(
                f"could not reach Baseten for {label} ({type(exc).__name__})",
                source="baseten",
                hint="the request may have reached Baseten; check before retrying"
                if method == "POST"
                else "check your network connection and retry",
            ) from exc
        return _parse(payload, label)


def _read_capped(response: httpx.Response, label: str) -> bytes:
    payload = bytearray()
    for chunk in response.iter_bytes():
        payload += chunk
        if len(payload) > _MAX_BODY:
            raise ExternalError(f"{label} returned more than 4 MiB", source="baseten")
    return bytes(payload)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _parse(payload: bytes, label: str) -> Any:
    if not payload.strip():
        return {}
    try:
        return json.loads(payload, parse_constant=_reject_constant)
    except ValueError:
        raise ExternalError(f"{label} returned malformed JSON", source="baseten") from None


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExternalError(f"{label} returned an unexpected response", source="baseten")
    return value


def _id(value: object, kind: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ExternalError(f"invalid Baseten {kind} id", source="baseten")
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
        raise ExternalError("refusing to send the API key to a non-Baseten URL", source="baseten")
    return url.rstrip("/")


def _clean(text: str, api_key: SecretStr) -> str:
    """``text`` on one line, without the key or other known secrets, at most 300 characters."""
    one_line = " ".join(text.split()).replace(api_key.get_secret_value(), MASK)
    return redact(one_line)[:_DETAIL_LIMIT]


def _status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _failure(exc: BaseException) -> str:
    status = _status(exc)
    return f"HTTP {status}, {type(exc).__name__}" if status else type(exc).__name__


# --- Loops SDK ---------------------------------------------------------------------------


class LoopsModule(Protocol):
    """The parts of ``baseten.loops`` used here; tests pass a fake of the same shape."""

    @property
    def ServiceClient(self) -> Callable[..., Any]: ...
    @property
    def Datum(self) -> Callable[..., Any]: ...
    @property
    def ModelInput(self) -> Any: ...
    @property
    def TensorData(self) -> Callable[..., Any]: ...
    @property
    def AdamParams(self) -> Callable[..., Any]: ...
    @property
    def SamplingParams(self) -> Callable[..., Any]: ...
    @property
    def SamplingClient(self) -> Callable[..., Any]: ...
    @property
    def BasetenDeployment(self) -> Callable[..., Any]: ...
    @property
    def ServerShutdownError(self) -> type[Exception]: ...


def _check_environment() -> None:
    """Refuse the SDK's env overrides that would send data elsewhere or reuse other resources."""
    for name in _REJECTED_ENV:
        if os.environ.get(name):
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
    except ImportError as exc:
        raise UsageError("the Baseten Loops SDK is not installed", hint=LOOPS_INSTALL_HINT) from exc
    names = {name: getattr(sdk, name) for name in _SDK_NAMES}
    return cast(LoopsModule, SimpleNamespace(**names, BasetenDeployment=BasetenDeployment))


class LoopsTrainer:
    """A Loops LoRA trainer as a :class:`~pup_finetune.training.backend.TrainingBackend`."""

    def __init__(
        self, client: Any, loops: LoopsModule, *, session_id: str, run_id: str, max_seq_len: int
    ) -> None:
        self._client = client
        self._loops = loops
        self._max_seq_len = max_seq_len
        self.session_id = session_id
        self.run_id = run_id

    def forward_backward(self, batch: Sequence[TokenDatum]) -> StepResult:
        datums, weights = self._datums(batch)
        output = self._call(
            "forward_backward",
            lambda: self._client.forward_backward(datums, "cross_entropy").result(),
        )
        return _step_result(output, weights)

    def forward(self, batch: Sequence[TokenDatum]) -> StepResult:
        datums, weights = self._datums(batch)
        output = self._call(
            "forward", lambda: self._client.forward(datums, "cross_entropy").result(), attempts=3
        )
        return _step_result(output, weights)

    def optim_step(self, learning_rate: float) -> None:
        params = self._loops.AdamParams(learning_rate=learning_rate)
        self._call("optim_step", lambda: self._client.optim_step(params).result())

    def save_state(self, name: str) -> str:
        return self._saved("save_state", name, lambda: self._client.save_state(name).result())

    def save_sampler_weights(self, name: str) -> str:
        path = self._saved(
            "save_weights_for_sampler",
            name,
            lambda: self._client.save_weights_for_sampler(name).result(),
        )
        match = CHECKPOINT_URI.fullmatch(path)
        if match is None or match[1] != self.run_id:
            raise ExternalError(
                "Loops returned an unexpected sampler checkpoint path", source="baseten"
            )
        return path

    def close(self) -> None:
        self._client.close()

    def _datums(self, batch: Sequence[TokenDatum]) -> tuple[list[Any], float]:
        loops = self._loops
        datums: list[Any] = []
        weights = 0.0
        for datum in batch:
            shifted = shift(datum, max_seq_len=self._max_seq_len)
            size = [len(shifted.inputs)]
            datums.append(
                loops.Datum(
                    model_input=loops.ModelInput.from_ints(shifted.inputs),
                    loss_fn_inputs={
                        "target_tokens": loops.TensorData(
                            data=shifted.targets, dtype="int64", shape=size
                        ),
                        "weights": loops.TensorData(
                            data=shifted.weights, dtype="float32", shape=size
                        ),
                    },
                )
            )
            weights += sum(shifted.weights)
        return datums, weights

    def _saved(self, operation: str, name: str, call: Callable[[], Any]) -> str:
        path = getattr(self._call(operation, call), "path", None)
        if not isinstance(path, str) or not path:
            raise ExternalError(f"Loops returned no path for checkpoint {name}", source="baseten")
        return path

    def _call(self, operation: str, call: Callable[[], Any], *, attempts: int = 1) -> Any:
        attempt = 1
        while True:
            try:
                return call()
            except Exception as exc:
                if attempt >= attempts or not self._transient(exc):
                    raise ExternalError(
                        f"the Loops {operation} call failed ({_failure(exc)})", source="baseten"
                    ) from exc
            _sleep(float(2 ** (attempt - 1)))
            attempt += 1

    def _transient(self, exc: Exception) -> bool:
        transient = (httpx.TransportError, TimeoutError, ConnectionError)
        if isinstance(exc, (*transient, self._loops.ServerShutdownError)):
            return True
        return _status(exc) in _TRANSIENT_STATUS


def _step_result(output: Any, weights: float) -> StepResult:
    try:
        metrics = dict(output.metrics or {})
        return StepResult(
            loss=float(metrics.get("loss", output.loss)),
            active_tokens=float(metrics.get("active_tokens", weights)),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExternalError("Loops returned a malformed step result", source="baseten") from exc


def open_trainer(
    api_key: SecretStr,
    *,
    base_model: str,
    rank: int,
    seed: int,
    max_seq_len: int,
    run_id: str,
    loops: LoopsModule | None = None,
    on_session: Callable[[str], None] | None = None,
) -> LoopsTrainer:
    """Create a Loops session and a dedicated LoRA trainer; blocks until it is ready. Paid.

    ``on_session`` gets the session id first. A failure that may leave a run in the session
    carries the manual cleanup step as an exception note.
    """
    sdk = loops_sdk(loops)
    try:
        service = sdk.ServiceClient(
            api_key=api_key.get_secret_value(), base_url=API_BASE, availability_model="dedicated"
        )
    except Exception as exc:
        raise _creation_error(exc, "Loops session", api_key) from exc
    session_id = _id(service.session_id, "session")
    if on_session is not None:
        on_session(session_id)
    try:
        client = service.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            replicas=1,
            seed=seed,
            max_seq_len=max_seq_len,
            with_sampler=False,
            name=run_id,
        )
    except Exception as exc:
        error = _creation_error(exc, "trainer", api_key)
        if _status(exc) is None:  # not an HTTP refusal, so the run may exist
            error.add_note(_orphan_hint(session_id))
        raise error from exc
    except BaseException as exc:  # e.g. Ctrl-C while the trainer provisions
        exc.add_note(_orphan_hint(session_id))
        raise
    trainer_run_id = client.run_id
    if not isinstance(trainer_run_id, str) or not _ID.fullmatch(trainer_run_id):
        client.close()
        raise ExternalError(
            "Baseten returned no valid run id for the trainer",
            source="baseten",
            hint=f"deactivate the run of Loops session {session_id} in the Baseten UI",
        )
    return LoopsTrainer(
        client, sdk, session_id=session_id, run_id=trainer_run_id, max_seq_len=max_seq_len
    )


def _creation_error(exc: Exception, what: str, api_key: SecretStr) -> ExternalError:
    status = _status(exc)
    if status is not None:
        response: Any = getattr(exc, "response", None)
        detail = _response_detail(response, api_key)
        suffix = f": {detail}" if detail else ""
        if status == 429:
            retry_after = str(response.headers.get("Retry-After", ""))
            return ExternalError(
                f"Baseten has no capacity for the {what} (HTTP 429){suffix}",
                source="baseten",
                hint=f"retry after {retry_after} s" if retry_after.isdigit() else "retry later",
            )
        return ExternalError(
            f"Baseten refused to create the {what} (HTTP {status}){suffix}", source="baseten"
        )
    if type(exc) is RuntimeError:
        message = f"Baseten {what} provisioning failed: {_clean(str(exc), api_key)}"
    else:
        message = f"creating the Baseten {what} failed ({type(exc).__name__})"
    return ExternalError(message, source="baseten")


def _orphan_hint(session_id: str) -> str:
    # The run may exist by now (e.g. it never became ready), and we have no id for it.
    return f"a run may remain in Loops session {session_id}; deactivate it in the Baseten UI"


def _response_detail(response: Any, api_key: SecretStr) -> str | None:
    """The ``detail``, ``message`` or ``error`` of an error body, else its text; cleaned."""
    try:
        payload = response.json()
    except Exception:
        text = str(getattr(response, "text", "") or "")
    else:
        fields = payload if isinstance(payload, dict) else {}
        values = (fields.get(key) for key in ("detail", "message", "error"))
        text = next((value for value in values if isinstance(value, str) and value.strip()), "")
    return _clean(text, api_key) or None


def load_sampler_checkpoint(
    api_key: SecretStr,
    *,
    base_url: str,
    model_id: str,
    deployment_id: str,
    sampler_id: str,
    session_id: str,
    checkpoint_path: str,
    base_model: str,
    prompt: Sequence[int],
    ready_timeout: float,
    loops: LoopsModule | None = None,
) -> None:
    """Route one sample to ``checkpoint_path``: vLLM serves the adapter on OpenAI routes after it."""
    sdk = loops_sdk(loops)
    deployment = sdk.BasetenDeployment(
        base_url=inference_url(base_url),
        model_id=_id(model_id, "model"),
        deployment_id=_id(deployment_id, "deployment"),
        api_base_url=API_BASE,
    )
    client = sdk.SamplingClient(
        api_key=api_key.get_secret_value(),
        base_model=base_model,
        deployment=deployment,
        sampler_id=sampler_id,
        session_id=session_id,
        checkpoint_path=checkpoint_path,
        ready_timeout=ready_timeout,
    )
    try:
        client.sample(
            prompt=sdk.ModelInput.from_ints(list(prompt)),
            num_samples=1,
            sampling_params=sdk.SamplingParams(max_tokens=1, temperature=0),
        )
    except Exception as exc:
        raise ExternalError(
            f"loading the checkpoint on the Baseten sampler failed ({_failure(exc)})",
            source="baseten",
        ) from exc
    finally:
        client.close()
