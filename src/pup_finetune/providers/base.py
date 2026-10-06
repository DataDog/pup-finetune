# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The provider seam: neutral types and the protocols each provider implements.

The shared training loop is written once against :class:`TrainingBackend`,
:class:`TrainingSession` and :class:`Sampler`; everything else a provider does (model
profiles, preflight, rendering, deployment) goes through :class:`Provider`. Nothing here
imports a provider SDK. See ``docs/architecture.md`` for how the pieces fit together.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.rendering.base import ChatMessage, Renderer, ToolSpec
from pup_finetune.rendering.tokens import TokenizedExample

if TYPE_CHECKING:
    from concurrent.futures import Future

# --- Models and training configuration ---------------------------------------------------

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProfile:
    """A base model a provider can fine-tune, and how to render conversations for it.

    ``alias`` is our short name (e.g. ``qwen3p8-27b``); ``base_model`` is the provider's id.
    ``tokenizer_model`` and ``tokenizer_revision`` pin the Hugging Face tokenizer and chat
    template; the revision must be a full 40-character lowercase commit SHA, never a branch
    or tag that can move. ``renderer`` names the renderer; ``max_seq_len`` is the training
    context limit in tokens.
    """

    provider: str
    alias: str
    base_model: str
    tokenizer_model: str
    tokenizer_revision: str
    renderer: str
    max_seq_len: int
    default_lora_rank: int = 8
    supports_reasoning: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not _COMMIT_SHA.fullmatch(self.tokenizer_revision):
            raise ValueError(
                "tokenizer_revision must be a full 40-character lowercase commit SHA, got "
                f"{self.tokenizer_revision!r}"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.default_lora_rank <= 0:
            raise ValueError(f"default_lora_rank must be positive, got {self.default_lora_rank}")


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """The LoRA adapter to train. ``alpha`` ``None`` leaves it to the provider's default."""

    rank: int
    alpha: int | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")


@dataclass(frozen=True, slots=True, kw_only=True)
class AdamConfig:
    """Adam optimizer parameters for one ``optim_step``.

    The fields mirror the Adam parameters the providers' training SDKs take, so a backend
    passes them through unchanged. ``grad_clip_norm`` ``0`` disables gradient clipping.
    """

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            self.grad_clip_norm,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Adam parameters must be finite")
        if self.learning_rate <= 0 or self.eps <= 0:
            raise ValueError("learning_rate and eps must be positive")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("beta1 and beta2 must be at least 0 and less than 1")
        if self.weight_decay < 0 or self.grad_clip_norm < 0:
            raise ValueError("weight_decay and grad_clip_norm must not be negative")


DEFAULT_SPEND_RESERVE_FRACTION = 0.1
"""The share of ``max_spend_usd`` kept back for the checkpoint saved when training stops."""


class TrainSpec(BaseModel):
    """How to train: the provider-neutral settings of one run, frozen into the run directory.

    ``patience`` is how many epochs without a validation-loss improvement of more than
    ``min_delta`` end training early. ``lora_rank`` ``None`` uses the profile's default.
    ``checkpoint_every_minutes`` adds a resumable checkpoint at a step boundary between the
    end-of-epoch ones. ``max_train_tokens`` stops cleanly before a step that would train on
    more rendered tokens than that. ``microbatch_token_budget`` packs each step's examples
    into microbatches of at most that many tokens (gradients accumulate over them); ``None``
    sends each step as one microbatch. ``max_spend_usd`` and ``hourly_rate_usd`` (both or
    neither) stop cleanly once the elapsed time at that rate would cost more than the cap;
    they measure wall-clock time, not the provider's bill. Training stops once the spend
    reaches ``max_spend_usd`` less ``spend_reserve_fraction`` of it, keeping that reserve for
    the calls in flight and the checkpoint saved at the stop. ``provider_options`` are
    checked by :meth:`Provider.validate_options`; the spec is frozen but that dict is not, so
    consumers use the normalized copy ``validate_options`` returns and never mutate
    ``provider_options`` in place.

    Build one from user input with :meth:`parse`, which raises :class:`UsageError` naming
    the offending fields; mapping fields to command-line flags is the command's job.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_epochs: PositiveInt = 5
    patience: PositiveInt = 1
    min_delta: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    learning_rate: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    batch_size: PositiveInt = 32
    lora_rank: PositiveInt | None = None
    lora_alpha: PositiveInt | None = None
    seed: NonNegativeInt = 42
    pipeline_depth: PositiveInt = 4
    checkpoint_every_minutes: float = Field(default=15.0, gt=0, allow_inf_nan=False)
    max_train_tokens: PositiveInt | None = None
    microbatch_token_budget: PositiveInt | None = None
    max_spend_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    hourly_rate_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    spend_reserve_fraction: float = Field(
        default=DEFAULT_SPEND_RESERVE_FRACTION, ge=0, lt=1, allow_inf_nan=False
    )
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spend_cap_needs_a_rate(self) -> Self:
        if (self.max_spend_usd is None) != (self.hourly_rate_usd is None):
            raise ValueError("max_spend_usd and hourly_rate_usd must be set together")
        return self

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> Self:
        """Validate ``data`` (field names as keys), or raise :class:`UsageError` naming fields."""
        try:
            return cls.model_validate(dict(data))
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                if error["loc"]
                else error["msg"]
                for error in exc.errors()
            ]
            raise UsageError(f"invalid training settings: {'; '.join(problems)}") from exc

    def lora_config(self, profile: ModelProfile) -> LoraConfig:
        """The adapter to train on ``profile``: ``lora_rank``, else the profile's default."""
        return LoraConfig(rank=self.lora_rank or profile.default_lora_rank, alpha=self.lora_alpha)

    def adam_config(self) -> AdamConfig:
        """The optimizer parameters for every step: a constant learning rate."""
        return AdamConfig(learning_rate=self.learning_rate)


class TransientError(ExternalError):
    """A provider call that failed in a way worth retrying: a timeout, a dropped connection.

    Backends raise it (or a :class:`TimeoutError` or :class:`ConnectionError`) only for calls
    that change nothing when repeated. The loop retries such failures for ``forward`` alone,
    never for ``forward_backward``, where a retry would accumulate the gradients twice.
    """


# --- Results the seam returns ------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointRef:
    """A saved checkpoint.

    ``training`` checkpoints hold weights and optimizer state and are resumable; ``sampler``
    checkpoints hold weights only and can be sampled from or promoted, on some providers only
    while the session that saved them is alive. ``uri`` is the provider's handle for it;
    ``created_in_session`` is the :attr:`TrainingSession.session_id` that saved it.
    """

    kind: Literal["training", "sampler"]
    name: str
    uri: str
    epoch: int
    step: int
    created_in_session: str


MAX_MODEL_ID_LEN = 63
"""The longest model id :meth:`TrainingSession.promote` is given; every provider accepts it."""

MODEL_ID_PATTERN = re.compile(rf"[a-z](?:[a-z0-9-]{{0,{MAX_MODEL_ID_LEN - 2}}}[a-z0-9])?")
"""What a model id given to :meth:`TrainingSession.promote` fully matches: lowercase letters,
digits and dashes, starting with a letter and not ending with a dash."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRef:
    """A promoted model: a checkpoint turned into a model the provider can deploy."""

    provider: str
    model_id: str
    uri: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Endpoint:
    """A deployment serving a model. ``inference_model`` is what chat requests name."""

    provider: str
    deployment_id: str
    model_id: str
    inference_model: str
    url: str | None = None


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """The loss of one forward pass over a batch.

    ``loss_sum`` is the weighted sum of per-token losses and ``weight`` the sum of the
    weights, so metrics from several batches combine by adding both before dividing.
    """

    loss_sum: float
    weight: float
    tokens: int

    @property
    def mean_loss(self) -> float:
        """The per-token mean loss; NaN when no token carried weight."""
        return self.loss_sum / self.weight if self.weight > 0 else math.nan


@dataclass(frozen=True, slots=True)
class Check:
    """One provider readiness check, reported by ``doctor``. Never holds secret values.

    It carries the same fields as :class:`pup_finetune.doctor.CheckResult`, but ``required``
    defaults to ``False`` here and to ``True`` there, so copy it across explicitly
    (``CheckResult(c.name, c.ok, c.detail, required=c.required)``) rather than relying on
    either default.
    """

    name: str
    ok: bool
    detail: str
    required: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Generation:
    """One sampled assistant turn, parsed into an OpenAI assistant message."""

    message: ChatMessage
    format_valid: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    stop_reason: str | None = None


# --- Protocols ---------------------------------------------------------------------------


class PendingResult[T](Protocol):
    """A result that may not be ready yet. Callers always call :meth:`result` eventually."""

    def result(self, timeout: float | None = None) -> T:
        """Wait for the result (at most ``timeout`` seconds), or raise the call's error."""
        ...


if TYPE_CHECKING:
    # mypy checks that a standard future satisfies the protocol, so backends may return one.
    def _a_future_is_a_pending_result(future: Future[int]) -> PendingResult[int]:
        return future


class Sampler(Protocol):
    """Generates assistant turns from a base model or a checkpoint. Close it when done."""

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        """The next assistant turn after ``messages``, greedily decoded."""
        ...

    def close(self) -> None:
        """Release the sampler's resources. Closing twice is a no-op."""
        ...


class TrainingSession(Protocol):
    """One live training session on a provider: a LoRA adapter being trained.

    Calls that return :class:`PendingResult` may run concurrently; the loop keeps several
    steps in flight and resolves every result, in order.
    """

    @property
    def session_id(self) -> str:
        """The provider's id for this session."""
        ...

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        """Compute the weighted cross-entropy loss and accumulate its gradients."""
        ...

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        """Compute the loss without touching gradients or weights (validation loss).

        Some backends' forward-only call returns per-token logprobs rather than a loss; they
        compute ``loss_sum`` as the weighted sum of the negative logprobs.
        """
        ...

    def optim_step(self, adam: AdamConfig) -> PendingResult[None]:
        """Apply the accumulated gradients with Adam."""
        ...

    def save_training_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        """Save weights and optimizer state, resumable with ``open_session(resume_from=)``.

        ``name`` is at most :attr:`TrainingBackend.max_checkpoint_name_len` characters.
        """
        ...

    def save_sampler_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        """Save weights only, to sample from or promote while this session is alive."""
        ...

    def load_weights(self, ckpt: CheckpointRef) -> None:
        """Replace the session's weights with ``ckpt``'s (e.g. the best epoch's)."""
        ...

    def promote(self, ckpt: CheckpointRef, model_id: str) -> ModelRef:
        """Turn a sampler checkpoint into a deployable model named ``model_id``.

        Only on a backend with :attr:`TrainingBackend.supports_promotion`. ``model_id``
        fully matches :data:`MODEL_ID_PATTERN` (lowercase letters, digits and dashes, starting
        with a letter, at most :data:`MAX_MODEL_ID_LEN` characters), which every provider
        accepts.
        """
        ...

    def sampler(self, ckpt: CheckpointRef | None) -> Sampler:
        """A sampler for a sampler checkpoint, or for the base model when ``ckpt`` is None."""
        ...


class TrainingBackend(Protocol):
    """Opens training sessions on one provider.

    Two capabilities are optional, and the loop asks before relying on them:
    :attr:`supports_resume` (``open_session(resume_from=)`` continues from a training
    checkpoint) and :attr:`supports_promotion` (:meth:`TrainingSession.promote` turns a sampler
    checkpoint into a deployable model; without it, the best sampler checkpoint is itself the
    deployable artifact).
    """

    @property
    def max_checkpoint_name_len(self) -> int:
        """The longest checkpoint name the provider accepts, enforced before saving."""
        ...

    @property
    def supports_resume(self) -> bool:
        """Whether ``open_session(resume_from=)`` continues from a training checkpoint."""
        ...

    @property
    def supports_promotion(self) -> bool:
        """Whether :meth:`TrainingSession.promote` works on this provider."""
        ...

    def open_session(
        self,
        profile: ModelProfile,
        lora: LoraConfig,
        *,
        resume_from: CheckpointRef | None = None,
        seed: int,
        max_seq_len: int,
    ) -> AbstractContextManager[TrainingSession]:
        """A session for ``profile``, fresh or resumed from a training checkpoint.

        ``seed`` is the run's seed and ``max_seq_len`` the longest example (in tokens) the
        session will train on, at most ``profile.max_seq_len``; a provider may size its
        trainer from them. Resuming needs :attr:`supports_resume`. Leaving the context ends
        the session and releases what it holds on the provider, whether training finished,
        stopped early or failed.
        """
        ...


@runtime_checkable
class Provider(Protocol):
    """Everything a training provider offers beyond the loop. Constructing one is cheap.

    :meth:`check` reports readiness without raising. :meth:`resolve_model` and
    :meth:`preflight` raise :class:`UsageError` for an unknown or unusable model;
    :meth:`validate_options` returns the normalized ``provider_options`` or raises
    :class:`UsageError`. :meth:`deploy` and :meth:`undeploy` are remote writes: callers gate
    them behind ``--confirm`` and ``PUP_READ_ONLY``.
    """

    @property
    def name(self) -> str:
        """The provider's registry name, e.g. ``fireworks``."""
        ...

    def check(self) -> list[Check]:
        """Readiness checks (credentials, SDK, reachability), for ``doctor``."""
        ...

    def models(self) -> list[ModelProfile]:
        """Every model profile this provider supports."""
        ...

    def resolve_model(self, alias_or_id: str) -> ModelProfile:
        """The profile for a profile alias or a provider model id."""
        ...

    def preflight(self, profile: ModelProfile) -> None:
        """Confirm with the provider that ``profile`` can be trained right now."""
        ...

    def renderer(self, profile: ModelProfile) -> Renderer:
        """The renderer for ``profile``, with its tokenizer pinned to the profile's revision."""
        ...

    def backend(self) -> TrainingBackend:
        """The training backend."""
        ...

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Check provider-specific ``TrainSpec.provider_options``; return them normalized."""
        ...

    def deploy(self, model: ModelRef, options: Mapping[str, Any]) -> Endpoint:
        """Deploy a promoted model and wait until it serves."""
        ...

    def undeploy(self, endpoint: Endpoint) -> None:
        """Tear a deployment down so it stops billing."""
        ...
