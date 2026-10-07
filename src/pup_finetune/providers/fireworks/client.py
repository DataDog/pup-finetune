# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks SDKs, behind a small typed interface. The only module that loads them.

Everything else in pup-finetune talks to the protocols defined here, never to the SDK. For
training, on the Fireworks Training SDK:

- :class:`FireworksTraining` is the account: it opens a :class:`TrainingService` (one
  serverless training session), lists a session's checkpoints, promotes one to a model and
  reads a model's record (over the REST API, :class:`~.api.FireworksApi`).
- :class:`TrainingService` creates the :class:`TrainingRun` (one LoRA adapter being trained,
  fresh or from a saved training checkpoint) and :class:`TokenSampler`\\s for the session.
- :class:`TrainingRun` takes :class:`~pup_finetune.rendering.tokens.TokenizedExample`\\s and
  :class:`~pup_finetune.providers.base.AdamConfig` and returns plain results
  (:class:`ForwardOutput`, :class:`OptimStepOutput`, checkpoint paths), so its callers never
  see an SDK type.

:class:`FireworksTrainingClient` implements them with the SDK (``fireworks-ai[training-sdk]``
and the ``tinker`` package it brings, from the ``fireworks`` extra); ``tests/fakes/fireworks.py``
implements them in memory. The SDK is imported on first use, never when this module is
imported, so the CLI and the provider stay cheap to load and work without the extra.

Every SDK failure surfaces as an :class:`~pup_finetune.errors.ExternalError` with source
``fireworks`` and a redacted message (see :func:`fireworks_error`): no capacity or quota is
:class:`~pup_finetune.providers.base.ProviderCapacityError`, a rejected key
:class:`~pup_finetune.providers.base.ProviderAuthError`, and a server error or dropped
connection :class:`~pup_finetune.providers.base.TransientError`. The API key is registered for
redaction when the client is built and is never part of a message.

For deployments, on the ``Fireworks`` client of the ``fireworks-ai`` package:
:class:`FireworksDeployments` matches deployment shapes to a model and creates, reads and
deletes on-demand deployments, returning plain :class:`ShapeVersion` and
:class:`DeploymentRecord` values; :class:`FireworksDeploymentsClient` implements it.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import SecretStr

from pup_finetune.credentials import login_hint
from pup_finetune.errors import ExternalError, FinetuneError, UsageError
from pup_finetune.output import info
from pup_finetune.providers.base import (
    MODEL_ID_PATTERN,
    AdamConfig,
    PendingResult,
    ProviderAuthError,
    ProviderCapacityError,
    TransientError,
)
from pup_finetune.providers.fireworks import CREDENTIALS
from pup_finetune.providers.fireworks.api import (
    API_URL,
    CLIENT_SOURCE,
    CLIENT_SOURCE_HEADER,
    FireworksApi,
    checked_key,
)
from pup_finetune.redaction import redact
from pup_finetune.rendering.tokens import TokenizedExample

if TYPE_CHECKING:
    import httpx

__all__ = [
    "ACCOUNT_ENV_VAR",
    "CAPACITY_HINT",
    "CAPACITY_RELEASED_HINT",
    "INSTALL_HINT",
    "MAX_CHECKPOINT_NAME_LEN",
    "SAMPLER_CHECKPOINT",
    "SOURCE",
    "TRAINING_CHECKPOINT",
    "TRAINING_URL",
    "CheckpointRow",
    "CreateRequest",
    "DeploymentRecord",
    "FireworksDeployments",
    "FireworksDeploymentsClient",
    "FireworksTraining",
    "FireworksTrainingClient",
    "ForwardOutput",
    "OptimStepOutput",
    "SampledTokens",
    "ShapeVersion",
    "StopTokenDecoder",
    "TokenSampler",
    "TrainingRun",
    "TrainingService",
    "fireworks_error",
    "resolve_account_id",
    "sort_rows",
    "to_adam_params",
    "to_tinker_datum",
]

SOURCE: Final = "fireworks"
TRAINING_URL: Final = "https://api.fireworks.ai/training/v1/serverless"
"""The Serverless Training endpoint every training session is opened on."""
ACCOUNT_ENV_VAR: Final = "FIREWORKS_ACCOUNT_ID"
"""Overrides the account lookup: the id recorded for the client and used to name a session
when the SDK does not report its resource name. It does not choose where training runs: the
SDK trains in the key's own account, and refuses a key that sees several accounts."""
INSTALL_HINT: Final = (
    "install the fireworks extra: uv sync --extra fireworks, "
    "or pip install 'pup-finetune[fireworks]'"
)

MAX_CHECKPOINT_NAME_LEN: Final = 17
"""The longest checkpoint name Fireworks can promote. It is checked only when promoting, long
after the checkpoint was saved, so callers enforce it before saving."""
TRAINING_CHECKPOINT: Final = "CHECKPOINT_TYPE_TRAINING_LORA"
"""The ``checkpointType`` of a resumable checkpoint (weights and optimizer state)."""
SAMPLER_CHECKPOINT: Final = "CHECKPOINT_TYPE_INFERENCE_LORA"
"""The ``checkpointType`` of a sampler checkpoint (weights only), the kind that is promoted."""

CAPACITY_HINT: Final = (
    "Fireworks has no training capacity or quota for this right now (an account runs a "
    "limited number of training sessions at once); retry later"
)
"""The hint of every :class:`ProviderCapacityError` the wrapper raises."""
CAPACITY_RELEASED_HINT: Final = (
    "Fireworks has no training capacity or quota for this right now (an account runs a "
    "limited number of training sessions at once); the session was ended and nothing was "
    "left running, so retry later"
)
"""The hint once a session that could not start a run has been ended."""

_LOSS_FN: Final = "cross_entropy"
_CLOUDFLARE_ENV_VARS: Final = ("CLOUDFLARE_ACCESS_CLIENT_ID", "CLOUDFLARE_ACCESS_CLIENT_SECRET")
_ACCOUNT_ID: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
_HTTP_STATUS: Final = re.compile(r"\bHTTP (\d{3})\b")
_CAPACITY_MARKERS: Final = (
    "out of capacity",
    "at capacity",
    "capacity_exhausted",
    "capacity is temporarily",
    "training capacity",
    "paused_capacity",
    "resource_exhausted",
    "quota",
)
_CONNECTION_ERRORS: Final = frozenset({"APIConnectionError", "APITimeoutError", "TransportError"})
_SKIPPED_DETAIL: Final = ("agent debug:", "docs:", "support:", "solution:")
_DETAIL_LIMIT: Final = 300
_CHAIN_LIMIT: Final = 8


# --- Results ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """What a forward (or forward-backward) pass over a batch returned.

    ``metrics`` are Fireworks' own, unchanged: a forward-backward pass reports ``loss:sum``,
    ``total_tokens:sum`` and ``response_tokens`` (the sum of the weights), a forward pass only
    ``total_tokens:sum``. ``logprobs`` has, for a forward pass, the log probability of each
    target token of each example in the batch, whatever its weight; its loss is computed from
    them. A forward-backward pass returns no logprobs (only a loss per example), so its
    ``logprobs`` is empty.
    """

    metrics: Mapping[str, float]
    logprobs: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """What an optimizer step returned: Fireworks' metrics, e.g. ``lr:last``, ``step_id:last``."""

    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class SampledTokens:
    """One sampled completion: its tokens and why sampling stopped (``stop`` or ``length``)."""

    tokens: tuple[int, ...]
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointRow:
    """One checkpoint of a training session, as Fireworks lists it.

    Fireworks renames what it saves: ``checkpoint_name`` is ``run-<hex>-<name>`` for a training
    checkpoint saved as ``<name>``, with a ``-<8 hex>`` suffix for a sampler checkpoint, so
    rows are told apart by ``checkpoint_type``, ``promotable`` and ``create_time`` rather than
    by name. ``resource_name`` is the full ``accounts/<a>/trainingSessions/<s>/checkpoints/<c>``
    name that :meth:`FireworksTraining.promote_session_checkpoint` takes.
    """

    checkpoint_name: str
    checkpoint_type: str
    promotable: bool
    create_time: str
    resource_name: str

    @property
    def created_at(self) -> datetime | None:
        """``create_time`` parsed, or ``None`` when it is missing or unreadable."""
        try:
            parsed = datetime.fromisoformat(self.create_time)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    @classmethod
    def from_api(cls, row: object) -> CheckpointRow:
        """The row in a Fireworks listing (camelCase keys).

        Raises :class:`ExternalError` for a row that is not an object or lacks its resource
        name or checkpoint type, so a changed listing fails loudly rather than reading as a
        session with nothing promotable.
        """
        if not isinstance(row, Mapping):
            raise ExternalError(
                "Fireworks listed a checkpoint that is not an object", source=SOURCE
            )
        resource_name = row.get("name")
        checkpoint_type = row.get("checkpointType") or row.get("checkpoint_type")
        if not isinstance(resource_name, str) or not resource_name:
            raise ExternalError("Fireworks listed a checkpoint without its name", source=SOURCE)
        if not isinstance(checkpoint_type, str) or not checkpoint_type:
            raise ExternalError(
                f"Fireworks listed the checkpoint {resource_name} without its type", source=SOURCE
            )
        checkpoint_name = row.get("checkpointName") or row.get("checkpoint_name")
        create_time = row.get("createTime") or row.get("create_time")
        return cls(
            checkpoint_name=str(checkpoint_name or resource_name.rsplit("/", 1)[-1]),
            checkpoint_type=checkpoint_type,
            promotable=row.get("promotable") is True,
            create_time=str(create_time or ""),
            resource_name=resource_name,
        )


_EPOCH: Final = datetime.fromtimestamp(0, UTC)


def sort_rows(rows: Sequence[CheckpointRow]) -> list[CheckpointRow]:
    """``rows`` oldest first by ``create_time``; rows without a readable time come first, in
    listing order."""
    return sorted(rows, key=lambda row: row.created_at or _EPOCH)


# --- The interface ------------------------------------------------------------------------


class TrainingRun(Protocol):
    """One LoRA adapter being trained in a session. Every call's result must be resolved."""

    @property
    def run_id(self) -> str | None:
        """Fireworks' id for the run (``run-<hex>``), when it reports one."""
        ...

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[ForwardOutput]:
        """The weighted cross-entropy loss of ``batch``; accumulates its gradients."""
        ...

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[ForwardOutput]:
        """The per-token logprobs of ``batch``, without touching gradients or weights."""
        ...

    def optim_step(self, adam: AdamConfig) -> PendingResult[OptimStepOutput]:
        """Apply the accumulated gradients with Adam."""
        ...

    def save_state(self, name: str) -> PendingResult[str]:
        """Save weights and optimizer state; the path to resume from.

        The path is ``<account>/run-<hex>/<name>``, for
        :meth:`TrainingService.create_training_client_from_state_with_optimizer` and
        :meth:`load_state`.
        """
        ...

    def save_weights_for_sampler(self, name: str) -> PendingResult[str]:
        """Save weights only; the path to sample from (Fireworks appends ``-<8 hex>``)."""
        ...

    def load_state(self, path: str) -> PendingResult[None]:
        """Replace the run's weights with those of the training checkpoint at ``path``."""
        ...


class StopTokenDecoder(Protocol):
    """What the SDK's sampler is given as its tokenizer.

    On the token-prompt path the SDK only decodes each stop token id to a string with it.
    """

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str: ...


class TokenSampler(Protocol):
    """Samples completions from a session's base model or one of its sampler checkpoints."""

    def sample(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        stop: Sequence[int] = (),
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> PendingResult[SampledTokens]:
        """One completion of the ``prompt`` tokens, stopping at any token in ``stop``."""
        ...

    def close(self) -> None:
        """Release the sampler. Closing twice is a no-op."""
        ...


class TrainingService(Protocol):
    """One serverless training session. Closing it lets the session expire.

    A session expires shortly after it is closed (or the process exits); its sampler
    checkpoints can then no longer be promoted, while its training checkpoints can still be
    resumed from in a new session.
    """

    @property
    def session_id(self) -> str:
        """Fireworks' id for the session (``ts-<hex>``)."""
        ...

    @property
    def session_name(self) -> str:
        """The session's full resource name, ``accounts/<a>/trainingSessions/<id>``."""
        ...

    def create_lora_training_client(
        self, base_model: str, rank: int, *, alpha: int | None = None, seed: int | None = None
    ) -> TrainingRun:
        """Start training a fresh LoRA adapter of ``rank`` on ``base_model``.

        ``alpha`` ``None`` leaves the LoRA alpha to Fireworks (twice the rank).
        """
        ...

    def create_training_client_from_state_with_optimizer(self, path: str) -> TrainingRun:
        """Continue training from the training checkpoint at ``path``, optimizer state included."""
        ...

    def create_sampling_client(
        self,
        *,
        tokenizer: StopTokenDecoder,
        base_model: str | None = None,
        model_path: str | None = None,
    ) -> TokenSampler:
        """A sampler for the sampler checkpoint at ``model_path``, or for the base model.

        Pass ``base_model`` (and no ``model_path``) for the base model: the session serves its
        own base weights, whatever ``base_model`` names. The SDK uses ``tokenizer`` only to
        decode the stop token ids it is given.
        """
        ...

    def close(self) -> None:
        """End the session. Closing twice is a no-op."""
        ...


class FireworksTraining(Protocol):
    """A Fireworks account's Training API: sessions, their checkpoints and promotion."""

    @property
    def account_id(self) -> str:
        """The key's account, e.g. ``my-team``; sessions train in it."""
        ...

    def service(self) -> TrainingService:
        """Open a new training session."""
        ...

    def list_session_checkpoints(self, session_name: str) -> list[CheckpointRow]:
        """Every checkpoint of the session ``session_name``, oldest first."""
        ...

    def promote_session_checkpoint(
        self, resource_name: str, output_model_id: str, base_model: str
    ) -> dict[str, Any]:
        """Turn a promotable (sampler) checkpoint into the model ``output_model_id``.

        ``resource_name`` comes from :attr:`CheckpointRow.resource_name`; ``output_model_id``
        fully matches :data:`~pup_finetune.providers.base.MODEL_ID_PATTERN`. Works only while
        the session that saved the checkpoint is alive. Returns Fireworks' model record.
        """
        ...

    def get_model(self, model_name: str) -> dict[str, Any] | None:
        """Fireworks' record of the model ``accounts/<a>/models/<id>``, or ``None`` when there
        is no such model. A free, read-only call."""
        ...

    def close(self) -> None:
        """Release the account-level HTTP client. Closing twice is a no-op."""
        ...


# --- Errors -------------------------------------------------------------------------------


def fireworks_error(exc: BaseException, *, what: str) -> FinetuneError:
    """The error to raise for ``exc``, a failure of ``what`` (e.g. ``"saving a checkpoint"``).

    A :class:`FinetuneError` is returned unchanged. Otherwise the HTTP status is read from
    the exception (its ``status_code``, its response's, or an ``HTTP <code>`` in its message)
    and from those it chains to: no capacity or quota (HTTP 429, or a message that says so)
    gives :class:`ProviderCapacityError`; HTTP 401 or 403 :class:`ProviderAuthError`; HTTP 408,
    5xx, a timeout or a connection failure :class:`TransientError`; anything else
    :class:`ExternalError`. The message is redacted and kept short.

    The capacity hint (:data:`CAPACITY_HINT`) does not say whether anything is still running:
    only the caller knows that. A session that could not start a run is ended by the wrapper,
    which then says so (:data:`CAPACITY_RELEASED_HINT`).
    """
    if isinstance(exc, FinetuneError):
        return exc
    chain = _chain(exc)
    status = next((code for code in map(_status_of, chain) if code is not None), None)
    text = " ".join(str(item) for item in chain).casefold()
    detail = _detail(exc)
    message = f"{what} failed" + (f" (HTTP {status})" if status is not None else "")
    message += f": {detail}" if detail else ""
    if status == 429 or any(marker in text for marker in _CAPACITY_MARKERS):
        return ProviderCapacityError(message, source=SOURCE, hint=CAPACITY_HINT)
    if status in (401, 403):
        return ProviderAuthError(
            message, source=SOURCE, hint=f"{login_hint(CREDENTIALS)} to a valid key"
        )
    if (status is not None and (status == 408 or status >= 500)) or any(
        _is_connection_error(item) for item in chain
    ):
        return TransientError(
            message, source=SOURCE, hint="check the connection to api.fireworks.ai and retry"
        )
    return ExternalError(message, source=SOURCE)


def _chain(exc: BaseException) -> list[BaseException]:
    """``exc`` and the exceptions it was raised from, newest first, without cycles."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < _CHAIN_LIMIT:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _status_of(exc: BaseException) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    match = _HTTP_STATUS.search(str(exc))
    return int(match.group(1)) if match else None


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return any(cls.__name__ in _CONNECTION_ERRORS for cls in type(exc).__mro__)


def _detail(exc: BaseException) -> str:
    """The first meaningful lines of ``exc``'s message, redacted and short."""
    lines = [line.strip() for line in redact(str(exc)).splitlines()]
    kept = [
        line.removeprefix("ERROR:").removeprefix("Cause:").strip()
        for line in lines
        if line and not line.casefold().startswith(_SKIPPED_DETAIL)
    ]
    text = "; ".join(kept[:2]) or type(exc).__name__
    return text if len(text) <= _DETAIL_LIMIT else text[: _DETAIL_LIMIT - 1] + "…"


def _call[T](what: str, fn: Callable[[], T]) -> T:
    """``fn()``, with an SDK failure mapped by :func:`fireworks_error`."""
    try:
        return fn()
    except FinetuneError:
        raise
    except Exception as exc:
        raise fireworks_error(exc, what=what) from exc


class _Pending[T]:
    """An SDK future whose failure is mapped to our errors and whose result is converted.

    Only the SDK call is mapped: a converter raises its own :class:`ExternalError` for a
    response it cannot read, and a bug in it surfaces as itself.
    """

    def __init__(self, what: str, future: _SdkFuture, convert: Callable[[Any], T]) -> None:
        self._what = what
        self._future = future
        self._convert = convert

    def result(self, timeout: float | None = None) -> T:
        value = _call(self._what, lambda: self._future.result(timeout=timeout))
        return self._convert(value)


# --- Conversions --------------------------------------------------------------------------


def _import(
    module: str, *, needs: str = "training on Fireworks needs the Fireworks Training SDK"
) -> Any:
    """Import ``module`` from the Fireworks extra, or raise :class:`UsageError` to install it.

    ``needs`` says what is missing, e.g. ``"training on Fireworks needs the ... SDK"``.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if not _is_sdk_module(exc.name or module):
            raise _broken_sdk(exc) from exc
        raise UsageError(f"{needs}, which is not installed", hint=INSTALL_HINT) from exc
    except ImportError as exc:
        raise _broken_sdk(exc) from exc


def _is_sdk_module(name: str) -> bool:
    return name.split(".", 1)[0] in ("fireworks", "tinker")


def _broken_sdk(exc: ImportError) -> UsageError:
    return UsageError(
        f"the Fireworks Training SDK is installed but failed to load: {redact(str(exc))}",
        hint="reinstall the fireworks extra: uv sync --extra fireworks --reinstall, "
        "or pip install --force-reinstall 'pup-finetune[fireworks]'",
    )


def _sdk() -> Any:
    return _import("fireworks.training.sdk")


def _tinker() -> Any:
    # The SDK patches tinker's types when its client loads; using tinker directly needs them
    # applied first.
    _import("fireworks.training.sdk.patches")
    return _import("tinker")


def to_tinker_datum(example: TokenizedExample) -> Any:
    """``example`` as a ``tinker.Datum`` for the cross-entropy loss."""
    tinker = _tinker()
    n = len(example.input_tokens)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(example.input_tokens)),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=list(example.target_tokens), dtype="int64", shape=[n]
            ),
            "weights": tinker.TensorData(data=list(example.weights), dtype="float32", shape=[n]),
        },
    )


def to_adam_params(adam: AdamConfig) -> Any:
    """``adam`` as ``tinker.AdamParams``."""
    return _tinker().AdamParams(
        learning_rate=adam.learning_rate,
        beta1=adam.beta1,
        beta2=adam.beta2,
        eps=adam.eps,
        weight_decay=adam.weight_decay,
        grad_clip_norm=adam.grad_clip_norm,
    )


def _metrics(raw: Any) -> dict[str, float]:
    metrics = getattr(raw, "metrics", None) or {}
    return {
        str(key): float(value)
        for key, value in dict(metrics).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _forward_backward_output(raw: Any) -> ForwardOutput:
    """``raw`` read as a :class:`ForwardOutput`; the loss is in the metrics (``loss:sum``)."""
    return ForwardOutput(metrics=_metrics(raw))


def _forward_output(raw: Any, *, examples: int) -> ForwardOutput:
    """``raw`` read as a :class:`ForwardOutput` for a batch of ``examples``.

    A forward pass reports no loss, so the logprobs are required: one tensor per example.
    """
    outputs = list(getattr(raw, "loss_fn_outputs", None) or [])
    if len(outputs) != examples:
        raise ExternalError(
            f"Fireworks returned {len(outputs)} outputs for a batch of {examples} examples",
            source=SOURCE,
        )
    logprobs: list[tuple[float, ...]] = []
    for output in outputs:
        tensor = output.get("logprobs") if isinstance(output, Mapping) else None
        values = getattr(tensor, "data", None)
        if values is None:
            raise ExternalError("Fireworks returned an output without logprobs", source=SOURCE)
        logprobs.append(tuple(float(value) for value in values))
    return ForwardOutput(metrics=_metrics(raw), logprobs=tuple(logprobs))


def _path(raw: Any) -> str:
    path = getattr(raw, "path", None)
    if not isinstance(path, str) or not path:
        raise ExternalError("Fireworks did not return the checkpoint's path", source=SOURCE)
    return path


def _sampled(raw: Any) -> SampledTokens:
    sequences = list(getattr(raw, "sequences", None) or [])
    if not sequences:
        raise ExternalError("Fireworks returned no sampled sequence", source=SOURCE)
    first = sequences[0]
    reason = getattr(first, "stop_reason", None)
    return SampledTokens(
        tokens=tuple(int(token) for token in first.tokens),
        stop_reason=str(reason) if reason is not None else None,
    )


# --- The SDK, as far as this module uses it -----------------------------------------------


class _SdkFuture(Protocol):
    def result(self, timeout: float | None = None) -> Any: ...


class _SdkTrainingClient(Protocol):
    @property
    def run_id(self) -> str | None: ...

    def forward_backward(self, data: list[Any], loss_fn: str) -> _SdkFuture: ...

    def forward(self, data: list[Any], loss_fn: str) -> _SdkFuture: ...

    def optim_step(self, adam_params: Any) -> _SdkFuture: ...

    def save_state(self, name: str) -> _SdkFuture: ...

    def save_weights_for_sampler(self, name: str) -> _SdkFuture: ...

    def load_state(self, path: str) -> _SdkFuture: ...


class _SdkSampler(Protocol):
    def sample(self, *, prompt: Any, num_samples: int, sampling_params: Any) -> _SdkFuture: ...

    def close(self) -> None: ...


class _SdkService(Protocol):
    @property
    def training_session_id(self) -> str | None: ...

    @property
    def training_session_name(self) -> str | None: ...

    def create_lora_training_client(
        self, *, base_model: str, rank: int, seed: int | None, alpha: int | None
    ) -> _SdkTrainingClient: ...

    def create_training_client_from_state_with_optimizer(self, path: str) -> _SdkTrainingClient: ...

    def create_sampling_client(
        self,
        *,
        tokenizer: object,
        base_model: str | None = ...,
        model_path: str | None = ...,
    ) -> _SdkSampler: ...

    def close(self) -> None: ...


class _SdkControlPlane(Protocol):
    def list_training_session_checkpoints(self, name: str) -> list[dict[str, Any]]: ...

    def promote_session_checkpoint(
        self, name: str, output_model_id: str, base_model: str
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


# --- The SDK-backed implementation --------------------------------------------------------


def resolve_account_id(
    api_key: SecretStr,
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """The key's account: ``FIREWORKS_ACCOUNT_ID`` when set, else the key's first account.

    The fallback lists the key's accounts with the REST API (a free call); ``transport``
    replaces the network, for tests. Raises :class:`UsageError` for a malformed account id or
    a key with no account, and :class:`ProviderAuthError` when Fireworks rejects the key.
    """
    environ = os.environ if env is None else env
    configured = environ.get(ACCOUNT_ENV_VAR, "")
    if configured.strip():
        return checked_account_id(configured, origin=ACCOUNT_ENV_VAR)
    with FireworksApi(api_key, transport=transport) as api:
        accounts = api.list_accounts()
    if not accounts:
        raise UsageError(
            "the Fireworks API key belongs to no account",
            hint=f"set {ACCOUNT_ENV_VAR}, or use a key created in your Fireworks account",
        )
    return accounts[0]


def checked_account_id(value: str, *, origin: str = "the account id") -> str:
    """``value`` as a bare Fireworks account id (``accounts/`` and spaces stripped).

    Raises :class:`UsageError` when it is not one; ``origin`` names where it came from.
    """
    account_id = value.strip().removeprefix("accounts/")
    if not _ACCOUNT_ID.fullmatch(account_id):
        raise UsageError(
            f"{origin} is not a Fireworks account id: {account_id!r}",
            hint=f"use the id in your account's URL, e.g. my-team, or unset {ACCOUNT_ENV_VAR}",
        )
    return account_id


class _Run:
    def __init__(self, client: _SdkTrainingClient) -> None:
        self._client = client

    @property
    def run_id(self) -> str | None:
        run_id = getattr(self._client, "run_id", None)
        return run_id if isinstance(run_id, str) else None

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[ForwardOutput]:
        what = "a Fireworks forward-backward pass"
        data = [to_tinker_datum(example) for example in batch]
        future = _call(what, lambda: self._client.forward_backward(data, _LOSS_FN))
        return _Pending(what, future, _forward_backward_output)

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[ForwardOutput]:
        what = "a Fireworks forward pass"
        data = [to_tinker_datum(example) for example in batch]
        future = _call(what, lambda: self._client.forward(data, _LOSS_FN))
        return _Pending(what, future, lambda raw: _forward_output(raw, examples=len(data)))

    def optim_step(self, adam: AdamConfig) -> PendingResult[OptimStepOutput]:
        what = "a Fireworks optimizer step"
        params = to_adam_params(adam)
        future = _call(what, lambda: self._client.optim_step(params))
        return _Pending(what, future, lambda raw: OptimStepOutput(_metrics(raw)))

    def save_state(self, name: str) -> PendingResult[str]:
        what = f"saving the Fireworks training checkpoint {name}"
        return _Pending(what, _call(what, lambda: self._client.save_state(name)), _path)

    def save_weights_for_sampler(self, name: str) -> PendingResult[str]:
        what = f"saving the Fireworks sampler checkpoint {name}"
        future = _call(what, lambda: self._client.save_weights_for_sampler(name))
        return _Pending(what, future, _path)

    def load_state(self, path: str) -> PendingResult[None]:
        what = "loading a Fireworks training checkpoint"
        future = _call(what, lambda: self._client.load_state(path))
        return _Pending(what, future, lambda _raw: None)


class _Sampler:
    def __init__(self, sampler: _SdkSampler) -> None:
        self._sampler = sampler
        self._closed = False

    def sample(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        stop: Sequence[int] = (),
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> PendingResult[SampledTokens]:
        what = "sampling from Fireworks"
        tinker = _tinker()
        params = tinker.SamplingParams(
            max_tokens=max_tokens, stop=list(stop) or None, temperature=temperature, seed=seed
        )
        model_input = tinker.ModelInput.from_ints(list(prompt))
        future = _call(
            what,
            lambda: self._sampler.sample(prompt=model_input, num_samples=1, sampling_params=params),
        )
        return _Pending(what, future, _sampled)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _call("closing a Fireworks sampler", self._sampler.close)


class _Service:
    def __init__(self, service: _SdkService, *, account_id: str) -> None:
        self._service = service
        self._account_id = account_id
        self._closed = False

    @property
    def session_id(self) -> str:
        session_id = self._service.training_session_id
        if not session_id:
            raise ExternalError("Fireworks did not report a training session id", source=SOURCE)
        return session_id

    @property
    def session_name(self) -> str:
        return (
            self._service.training_session_name
            or f"accounts/{self._account_id}/trainingSessions/{self.session_id}"
        )

    def create_lora_training_client(
        self, base_model: str, rank: int, *, alpha: int | None = None, seed: int | None = None
    ) -> TrainingRun:
        return self._start(
            f"starting a LoRA run on {base_model}",
            lambda: self._service.create_lora_training_client(
                base_model=base_model, rank=rank, seed=seed, alpha=alpha
            ),
        )

    def create_training_client_from_state_with_optimizer(self, path: str) -> TrainingRun:
        return self._start(
            "resuming a Fireworks training checkpoint",
            lambda: self._service.create_training_client_from_state_with_optimizer(path),
        )

    def create_sampling_client(
        self,
        *,
        tokenizer: StopTokenDecoder,
        base_model: str | None = None,
        model_path: str | None = None,
    ) -> TokenSampler:
        if (base_model is None) == (model_path is None):
            raise UsageError("a Fireworks sampler needs exactly one of base_model and model_path")
        sampler = _call(
            "opening a Fireworks sampler",
            lambda: self._service.create_sampling_client(
                tokenizer=tokenizer, base_model=base_model, model_path=model_path
            ),
        )
        return _Sampler(sampler)

    def close(self) -> None:
        """End the session, then stop its heartbeat; raises the first failure of the two.

        Closing is attempted once: the heartbeat stops with the holder even when the SDK's
        close fails, and the session then expires on its own.
        """
        if self._closed:
            return
        self._closed = True
        steps: list[tuple[str, Callable[[], None]]] = [
            ("closing the Fireworks training session", self._service.close)
        ]
        holder = getattr(self._service, "holder", None)
        close_holder = getattr(holder, "close", None)
        if callable(close_holder):
            steps.append(("stopping the Fireworks training session's heartbeat", close_holder))
        first: FinetuneError | None = None
        for what, step in steps:
            try:
                _call(what, step)
            except FinetuneError as exc:
                first = first or exc
        if first is not None:
            raise first

    def _start(self, what: str, create: Callable[[], _SdkTrainingClient]) -> TrainingRun:
        try:
            return _Run(_call(what, create))
        except ProviderCapacityError as exc:
            error = exc
        # Out of capacity: end the session too, so nothing is left provisioned, and say so
        # only once it is ended. Outside the handler, so a failed close is mapped on its own.
        try:
            self.close()
        except FinetuneError as cleanup:
            print(
                redact(f"warning: the Fireworks training session may still be open: {cleanup}"),
                file=sys.stderr,
            )
        else:
            error.hint = CAPACITY_RELEASED_HINT
        raise error


class FireworksTrainingClient:
    """:class:`FireworksTraining` on the Fireworks Training SDK. See the module docstring.

    ``account_id`` defaults to :func:`resolve_account_id` (``FIREWORKS_ACCOUNT_ID`` in
    ``env``, else the key's first account, looked up over ``transport``, which also carries
    :meth:`get_model`'s REST call). It is the id this
    client records and names sessions with; training itself runs in the key's own account,
    which the SDK resolves from the key (and refuses a key that sees several accounts).
    Building it checks the key's form and registers it for redaction; the SDK loads on the
    first session. Every request carries ``X-Fireworks-Client-Source: pup-finetune/<version>``.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        account_id: str | None = None,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._secret = checked_key(api_key)
        self._account_id = (
            checked_account_id(account_id)
            if account_id is not None
            else resolve_account_id(api_key, env=env, transport=transport)
        )
        self._headers = {CLIENT_SOURCE_HEADER: CLIENT_SOURCE}
        self._transport = transport
        self._control_plane: _SdkControlPlane | None = None

    @property
    def account_id(self) -> str:
        return self._account_id

    def service(self) -> TrainingService:
        _refuse_forwarded_credentials()
        sdk = _sdk()

        def create() -> _SdkService:
            service: _SdkService = sdk.FiretitanServiceClient(
                api_key=self._secret,
                base_url=TRAINING_URL,
                default_headers=dict(self._headers),
                managed_additional_headers=dict(self._headers),
            )
            # A short capacity or recovery conflict (HTTP 409) while a result is retrieved is
            # fatal unless a supervised serverless client opts in to the SDK's retry, which is
            # what this is. The opt-in needs the holder the SDK builds for a key it trains with.
            enable_retry = getattr(service, "_enable_serverless_supervised_409_retry", None)
            if hasattr(service, "holder") and callable(enable_retry):
                enable_retry()
            return service

        return _Service(
            _call("starting a Fireworks training session", create), account_id=self._account_id
        )

    def list_session_checkpoints(self, session_name: str) -> list[CheckpointRow]:
        control_plane = self._control()
        raw = _call(
            "listing the session's Fireworks checkpoints",
            lambda: control_plane.list_training_session_checkpoints(session_name),
        )
        return sort_rows([CheckpointRow.from_api(row) for row in raw])

    def promote_session_checkpoint(
        self, resource_name: str, output_model_id: str, base_model: str
    ) -> dict[str, Any]:
        if not MODEL_ID_PATTERN.fullmatch(output_model_id):
            raise UsageError(f"{output_model_id!r} is not a valid Fireworks model id")
        control_plane = self._control()
        model = _call(
            f"promoting a Fireworks checkpoint to the model {output_model_id}",
            lambda: control_plane.promote_session_checkpoint(
                name=resource_name, output_model_id=output_model_id, base_model=base_model
            ),
        )
        return dict(model)

    def get_model(self, model_name: str) -> dict[str, Any] | None:
        with FireworksApi(SecretStr(self._secret), transport=self._transport) as api:
            return api.get_model(model_name)

    def close(self) -> None:
        control_plane, self._control_plane = self._control_plane, None
        if control_plane is not None:
            _call("closing the Fireworks client", control_plane.close)

    def _control(self) -> _SdkControlPlane:
        if self._control_plane is None:
            sdk = _sdk()

            def create() -> _SdkControlPlane:
                client: _SdkControlPlane = sdk.FireworksClient(
                    api_key=self._secret, base_url=API_URL, additional_headers=dict(self._headers)
                )
                return client

            self._control_plane = _call("connecting to the Fireworks API", create)
        return self._control_plane


def _refuse_forwarded_credentials() -> None:
    """Refuse when the SDK would forward unrelated credentials from the environment.

    The SDK's HTTP layer copies ``CLOUDFLARE_ACCESS_CLIENT_ID`` and
    ``CLOUDFLARE_ACCESS_CLIENT_SECRET`` from the environment into every request's headers, so
    they would reach api.fireworks.ai.
    """
    present = [name for name in _CLOUDFLARE_ENV_VARS if os.environ.get(name)]
    if present:
        raise UsageError(
            f"{' and '.join(present)} {'is' if len(present) == 1 else 'are'} set, and the "
            "Fireworks Training SDK would send it to api.fireworks.ai",
            hint="unset it for pup finetune, e.g. env -u CLOUDFLARE_ACCESS_CLIENT_ID "
            "-u CLOUDFLARE_ACCESS_CLIENT_SECRET pup finetune ...",
        )


# --- Deployments --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeVersion:
    """One deployment shape version Fireworks matched to a model.

    ``shape`` is the shape's name, what :meth:`FireworksDeployments.create_deployment` takes
    (Fireworks records the version it resolves to); ``name`` is the version's own name.
    """

    name: str
    shape: str
    validated: bool
    latest_validated: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentRecord:
    """A Fireworks deployment, as the API reports it.

    ``name`` is its resource name, ``accounts/<account>/deployments/<id>``; ``state`` one of
    ``CREATING``, ``READY``, ``UPDATING``, ``FAILED``, ``DELETING``, ``DELETED`` (or
    ``STATE_UNSPECIFIED``); ``detail`` the server's status message, when it sent one.
    ``shape`` is the shape version Fireworks resolved the requested shape to;
    ``display_name`` and the replica counts are as the deployment was created, and ``None``
    when the API did not report them.
    """

    name: str
    state: str
    base_model: str
    shape: str | None = None
    create_time: str | None = None
    detail: str = ""
    display_name: str | None = None
    min_replica_count: int | None = None
    max_replica_count: int | None = None


class FireworksDeployments(Protocol):
    """A Fireworks account's on-demand deployments."""

    @property
    def account_id(self) -> str:
        """The account the deployments are made in, e.g. ``my-team``."""
        ...

    def match_shapes(self, model: str) -> list[ShapeVersion]:
        """The deployment shape versions the account can deploy ``model`` on, as listed."""
        ...

    def create_deployment(
        self,
        *,
        base_model: str,
        deployment_id: str,
        shape: str,
        display_name: str,
        min_replica_count: int,
        max_replica_count: int,
        scale_to_zero_window: str,
        enable_hot_load: bool,
    ) -> DeploymentRecord:
        """Create the deployment ``deployment_id`` of ``base_model`` on ``shape``.

        Sent once, never retried by the SDK (the API takes no idempotency key, so a retried
        create that had succeeded would answer HTTP 409). When the create fails in a way that
        may hide a success (a 409, a timeout, a server error or a dropped connection), the
        deployment is read back: an existing deployment under that id is returned as the one
        created only when it is this create's own, of ``base_model`` with ``display_name``,
        on ``shape`` and with the replica counts asked for (as far as the API reports them).
        Raises :class:`UsageError` when the id is taken by any other deployment, so one this
        create did not make is never recorded (and later undeployed) as its own, and
        :class:`TransientError` when the create succeeded but its answer cannot be read (no
        deployment name), as the deployment exists.
        """
        ...

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        """The deployment ``deployment_id``, or ``None`` when Fireworks has none (HTTP 404)."""
        ...

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete the deployment; ``False`` when it was already gone (HTTP 404)."""
        ...

    def close(self) -> None:
        """Release the HTTP client. Closing twice is a no-op."""
        ...


class _SdkDeploymentsResource(Protocol):
    def create(self, **kwargs: Any) -> Any: ...

    def get(self, deployment_id: str) -> Any: ...

    def delete(self, deployment_id: str) -> Any: ...


class _SdkShapeVersionsResource(Protocol):
    def match_for_model(self, model: str) -> Any: ...


class _SdkFireworks(Protocol):
    @property
    def deployments(self) -> _SdkDeploymentsResource: ...

    @property
    def deployment_shape_versions(self) -> _SdkShapeVersionsResource: ...

    def with_options(self, *, max_retries: int) -> _SdkFireworks: ...

    def close(self) -> None: ...


SdkFireworksFactory = Callable[..., _SdkFireworks]
"""Builds the SDK's ``Fireworks`` client from its keyword arguments (tests pass a fake)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateRequest:
    """What a create asked for, to tell a deployment it made from one already there."""

    base_model: str
    shape: str
    display_name: str
    min_replica_count: int
    max_replica_count: int

    def mismatch(self, existing: DeploymentRecord) -> str | None:
        """How ``existing`` differs from this request, or ``None`` when it matches.

        Fireworks records the shape version it resolves the shape to, so a shape matches
        itself or any of its versions. A field the API did not report is not compared.
        """
        if existing.display_name != self.display_name:
            return f"its display name is {existing.display_name or 'empty'!r}"
        shape = existing.shape
        if (
            shape is not None
            and shape != self.shape
            and not shape.startswith(f"{self.shape}/versions/")
        ):
            return f"it runs on the shape {shape}"
        for label, theirs, ours in (
            ("minimum", existing.min_replica_count, self.min_replica_count),
            ("maximum", existing.max_replica_count, self.max_replica_count),
        ):
            if theirs is not None and theirs != ours:
                return f"its {label} replica count is {theirs}, not {ours}"
        return None

    def made(self, existing: DeploymentRecord) -> bool:
        """Whether ``existing`` is the deployment this request creates: of its base model,
        and with nothing :meth:`mismatch` finds."""
        return existing.base_model == self.base_model and self.mismatch(existing) is None


def _deployment_record(raw: Any) -> DeploymentRecord:
    name = getattr(raw, "name", None)
    if not isinstance(name, str) or not name:
        raise ExternalError("Fireworks returned a deployment without its name", source=SOURCE)
    state = getattr(raw, "state", None)
    created = getattr(raw, "create_time", None)
    status = getattr(raw, "status", None)
    message = getattr(status, "message", None)
    shape = getattr(raw, "deployment_shape", None)
    base_model = getattr(raw, "base_model", None)
    display_name = getattr(raw, "display_name", None)
    return DeploymentRecord(
        name=name,
        state=str(state) if state else "STATE_UNSPECIFIED",
        base_model=str(base_model or ""),
        shape=str(shape) if shape else None,
        create_time=created.isoformat() if isinstance(created, datetime) else created or None,
        detail=redact(str(message))[:_DETAIL_LIMIT] if message else "",
        display_name=display_name if isinstance(display_name, str) else None,
        min_replica_count=_count(getattr(raw, "min_replica_count", None)),
        max_replica_count=_count(getattr(raw, "max_replica_count", None)),
    )


def _count(value: Any) -> int | None:
    return value if type(value) is int else None


def _shape_versions(raw: Any) -> list[ShapeVersion]:
    versions: list[ShapeVersion] = []
    for version in getattr(raw, "deployment_shape_versions", None) or []:
        name = getattr(version, "name", None)
        snapshot_name = getattr(getattr(version, "snapshot", None), "name", None)
        shape = snapshot_name or (name.split("/versions/", 1)[0] if name else None)
        if not isinstance(shape, str) or not shape:
            continue
        versions.append(
            ShapeVersion(
                name=str(name or shape),
                shape=shape,
                validated=getattr(version, "validated", None) is True,
                latest_validated=getattr(version, "latest_validated", None) is True,
            )
        )
    return versions


def _has_status(exc: BaseException, status: int) -> bool:
    return any(_status_of(item) == status for item in _chain(exc))


class FireworksDeploymentsClient:
    """:class:`FireworksDeployments` on the SDK's ``Fireworks`` client. See the module docstring.

    ``account_id`` defaults to :func:`resolve_account_id` (``FIREWORKS_ACCOUNT_ID`` in
    ``env``, else the key's first account). ``sdk`` builds the SDK client (default: the
    ``fireworks`` package's ``Fireworks``, loaded on first use). Every request carries
    ``X-Fireworks-Client-Source: pup-finetune/<version>``.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        account_id: str | None = None,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sdk: SdkFireworksFactory | None = None,
    ) -> None:
        self._secret = checked_key(api_key)
        self._account_id = (
            checked_account_id(account_id)
            if account_id is not None
            else resolve_account_id(api_key, env=env, transport=transport)
        )
        self._factory = sdk
        self._client: _SdkFireworks | None = None

    @property
    def account_id(self) -> str:
        return self._account_id

    def match_shapes(self, model: str) -> list[ShapeVersion]:
        client = self._sdk()
        raw = _call(
            f"matching Fireworks deployment shapes to {model}",
            lambda: client.deployment_shape_versions.match_for_model(model),
        )
        return _shape_versions(raw)

    def create_deployment(
        self,
        *,
        base_model: str,
        deployment_id: str,
        shape: str,
        display_name: str,
        min_replica_count: int,
        max_replica_count: int,
        scale_to_zero_window: str,
        enable_hot_load: bool,
    ) -> DeploymentRecord:
        if not MODEL_ID_PATTERN.fullmatch(deployment_id):
            raise UsageError(f"{deployment_id!r} is not a valid Fireworks deployment id")
        # The SDK retries a POST on a timeout, 408, 409, 429 or 5xx; a create must be sent once.
        client = self._sdk().with_options(max_retries=0)
        what = f"creating the Fireworks deployment {deployment_id}"
        try:
            raw = client.deployments.create(
                base_model=base_model,
                deployment_id=deployment_id,
                deployment_shape=shape,
                display_name=display_name,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                autoscaling_policy={"scale_to_zero_window": scale_to_zero_window},
                enable_hot_load=enable_hot_load,
            )
        except FinetuneError:
            raise
        except Exception as exc:
            error = fireworks_error(exc, what=what)
            conflict = _has_status(exc, 409)
            if not (conflict or isinstance(error, TransientError)):
                raise error from exc
            requested = CreateRequest(
                base_model=base_model,
                shape=shape,
                display_name=display_name,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )
            existing = self._created_anyway(deployment_id, requested, error)
            if existing is not None:
                return existing
            if conflict and not isinstance(error, ProviderCapacityError):
                raise UsageError(
                    f"Fireworks reports the deployment id {deployment_id} as taken (HTTP 409)",
                    hint="pass another --deployment-id, or delete the existing deployment",
                ) from exc
            raise error from exc
        try:
            return _deployment_record(raw)
        except ExternalError as exc:
            # The create was accepted, so the deployment exists and may bill: transient, so
            # the caller keeps its receipt and says how to undeploy it, rather than treating
            # it as refused.
            raise TransientError(
                f"{what} succeeded, but its answer could not be read: {exc.message}",
                source=SOURCE,
                hint=f"the deployment {deployment_id} was created",
            ) from exc

    def _created_anyway(
        self, deployment_id: str, requested: CreateRequest, error: FinetuneError
    ) -> DeploymentRecord | None:
        """The deployment a failed create may have made anyway, read back; ``None`` when
        there is none. Raises :class:`UsageError` when the id holds a deployment this create
        did not make: of another model, under another display name, or on another shape or
        replica counts."""
        try:
            existing = self.get_deployment(deployment_id)
        except FinetuneError as read_error:
            raise error from read_error
        if existing is None:
            return None
        hint = "pass another --deployment-id, or delete the existing deployment"
        if existing.base_model != requested.base_model:
            raise UsageError(
                f"the account already has a Fireworks deployment named {deployment_id}, "
                f"serving {existing.base_model or 'another model'}",
                hint=hint,
            )
        mismatch = requested.mismatch(existing)
        if mismatch is not None:
            raise UsageError(
                f"the account already has a Fireworks deployment named {deployment_id} that "
                f"this deploy did not create ({mismatch})",
                hint=hint,
            )
        info(
            f"creating {deployment_id} reported a failure ({error.message}), but the "
            f"deployment of {requested.base_model} exists: using it"
        )
        return existing

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        client = self._sdk()
        try:
            raw = client.deployments.get(deployment_id)
        except FinetuneError:
            raise
        except Exception as exc:
            if _has_status(exc, 404):
                return None
            raise fireworks_error(
                exc, what=f"reading the Fireworks deployment {deployment_id}"
            ) from exc
        return _deployment_record(raw)

    def delete_deployment(self, deployment_id: str) -> bool:
        client = self._sdk()
        try:
            client.deployments.delete(deployment_id)
        except FinetuneError:
            raise
        except Exception as exc:
            if _has_status(exc, 404):
                return False
            raise fireworks_error(
                exc, what=f"deleting the Fireworks deployment {deployment_id}"
            ) from exc
        return True

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            _call("closing the Fireworks client", client.close)

    def _sdk(self) -> _SdkFireworks:
        if self._client is None:
            factory = self._factory
            if factory is None:
                factory = _import(
                    "fireworks", needs="deploying on Fireworks needs the Fireworks SDK"
                ).Fireworks
            build = factory

            def create() -> _SdkFireworks:
                client: _SdkFireworks = build(
                    api_key=self._secret,
                    account_id=self._account_id,
                    base_url=API_URL,
                    default_headers={CLIENT_SOURCE_HEADER: CLIENT_SOURCE},
                )
                return client

            self._client = _call("connecting to the Fireworks API", create)
        return self._client


if TYPE_CHECKING:
    # mypy checks that the implementations satisfy the protocols.
    def _conforms(client: FireworksTrainingClient) -> FireworksTraining:
        return client

    def _run_conforms(run: _Run) -> TrainingRun:
        return run

    def _service_conforms(service: _Service) -> TrainingService:
        return service

    def _sampler_conforms(sampler: _Sampler) -> TokenSampler:
        return sampler

    def _deployments_conform(client: FireworksDeploymentsClient) -> FireworksDeployments:
        return client
