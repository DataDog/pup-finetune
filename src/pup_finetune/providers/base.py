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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

from pup_finetune.billing import BillingPart
from pup_finetune.errors import ExternalError, FinetuneError, StateError, UsageError
from pup_finetune.redaction import redact
from pup_finetune.rendering.base import ChatMessage, Renderer, ToolSpec
from pup_finetune.rendering.hf import COMMIT_SHA
from pup_finetune.rendering.tokens import TokenizedExample

if TYPE_CHECKING:
    from concurrent.futures import Future

    from pup_finetune.inference import ChatCompleter

# --- Models and training configuration ---------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProfile:
    """A base model a provider can fine-tune, and how to render conversations for it.

    ``alias`` is our short name (e.g. ``qwen3p8-27b``); ``base_model`` is the provider's id.
    ``tokenizer_model`` and ``tokenizer_revision`` pin the Hugging Face tokenizer and chat
    template; the revision must be a full 40-character lowercase commit SHA, never a branch
    or tag that can move. ``renderer`` names the renderer; ``max_seq_len`` is the training
    context limit in tokens. Any of :attr:`names` selects the profile (:func:`find_profile`).
    ``trust_remote_code`` says the tokenizer runs code from the model's repository, at the
    pinned revision.
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
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not COMMIT_SHA.fullmatch(self.tokenizer_revision):
            raise ValueError(
                "tokenizer_revision must be a full 40-character lowercase commit SHA, got "
                f"{self.tokenizer_revision!r}"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.default_lora_rank <= 0:
            raise ValueError(f"default_lora_rank must be positive, got {self.default_lora_rank}")

    def remote_code_warning(self) -> str | None:
        """What a preview warns when the tokenizer runs code from the model's repository."""
        if not self.trust_remote_code:
            return None
        return (
            f"the {self.alias} tokenizer runs Python code from the Hugging Face repository "
            f"{self.tokenizer_model} at commit {self.tokenizer_revision} on this machine "
            "(trust_remote_code)"
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Every name that selects this profile, alias first: the alias, the provider's
        model id and the Hugging Face tokenizer, each once."""
        return tuple(dict.fromkeys((self.alias, self.base_model, self.tokenizer_model)))


def selects(names: Sequence[str], name: str) -> bool:
    """Whether ``name`` selects the profile whose :attr:`~ModelProfile.names` are ``names``:
    its alias exactly, or its provider model id or Hugging Face tokenizer in any case."""
    return name == names[0] or name.casefold() in {other.casefold() for other in names[1:]}


def find_profile(profiles: Sequence[ModelProfile], name: str, *, provider: str) -> ModelProfile:
    """The profile of ``provider``'s ``profiles`` that ``name`` selects (:func:`selects`); a
    :class:`UsageError` listing the aliases for any other."""
    for profile in profiles:
        if selects(profile.names, name):
            return profile
    known = ", ".join(profile.alias for profile in profiles)
    raise UsageError(
        f"unknown {provider} model {name!r}",
        hint=f"known models: {known}; `pup finetune models list` shows every name each accepts",
    )


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


MODEL_DEFAULT: Final = "model default"
"""The source :meth:`TrainSpec.resolved_settings` names for a value the model profile gave."""

MAX_SEQ_LEN_FLAG: Final = "--max-seq-len"
"""The source of a context cut with ``--max-seq-len``; one cut with ``--provider-option
max_seq_len`` is :data:`MAX_SEQ_LEN_OPTION_FLAG`."""

MAX_SEQ_LEN_OPTION_FLAG: Final = "--provider-option max_seq_len"

DEFAULT_SPEND_RESERVE_FRACTION = 0.1
"""The share of ``max_spend_usd`` kept back for the checkpoint saved when training stops."""


class TrainSpec(BaseModel):
    """How to train: the provider-neutral settings of one run, frozen into the run directory.

    ``patience`` is how many epochs without a validation-loss improvement of more than
    ``min_delta`` end training early. ``learning_rate`` is the peak rate: after a linear
    warmup over ``warmup_fraction`` of the run's steps, ``lr_schedule`` keeps it (``constant``)
    or decays it toward zero (``linear``, ``cosine``) over :meth:`max_steps`, the steps of
    ``max_epochs`` epochs (see :meth:`learning_rate_at`). ``lora_rank`` ``None`` uses the
    profile's default.
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
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"
    warmup_fraction: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
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
        """The optimizer parameters at the peak learning rate; each step sends a copy at
        :meth:`learning_rate_at`."""
        return AdamConfig(learning_rate=self.learning_rate)

    def max_steps(self, train_rows: int) -> int:
        """The most optimizer steps a run on ``train_rows`` rows takes: the schedule's horizon."""
        return self.max_epochs * math.ceil(train_rows / self.batch_size)

    def warmup_steps(self, total_steps: int) -> int:
        """The steps of ``total_steps`` that warm up: ``warmup_fraction`` of them, rounded up."""
        # The fraction as written: 0.1 * 30 is 3.0000000000000004 in binary floating point.
        return math.ceil(Fraction(repr(self.warmup_fraction)) * total_steps)

    def learning_rate_at(self, step: int, total_steps: int) -> float:
        """The learning rate of optimizer ``step`` (1-based, over the run) of ``total_steps``.

        It ramps up linearly to ``learning_rate`` over :meth:`warmup_steps`, then stays there
        (``constant``) or decays over the rest (``linear``, ``cosine``); it never reaches zero.
        """
        warmup = self.warmup_steps(total_steps)
        if step <= warmup:
            return self.learning_rate * step / warmup
        if self.lr_schedule == "constant":
            return self.learning_rate
        progress = (step - 1 - warmup) / max(1, total_steps - warmup)
        if self.lr_schedule == "linear":
            return self.learning_rate * (1 - progress)
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def resolved_settings(
        self, profile: ModelProfile, *, max_seq_len_flag: str = MAX_SEQ_LEN_FLAG
    ) -> dict[str, Any]:
        """The values a run on ``profile`` (its training profile) trains with, as a preview
        shows them and ``run.json`` records them: the LoRA rank and the context resolved,
        each with where it comes from (a cut context with ``max_seq_len_flag``, the flag
        that set it), and an alpha left to the provider said so rather than guessed. The
        optimizer's fixed parameters are not here (see :meth:`recorded_settings`)."""
        lora = self.lora_config(profile)
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "learning_rate": self.learning_rate,
            "lr_schedule": self.lr_schedule,
            "warmup_fraction": self.warmup_fraction,
            "batch_size": self.batch_size,
            "lora": {
                "rank": lora.rank,
                "rank_source": "--lora-rank" if self.lora_rank is not None else MODEL_DEFAULT,
                "alpha": lora.alpha,
                "alpha_source": "--lora-alpha"
                if self.lora_alpha is not None
                else "chosen by the provider",
            },
            "max_seq_len": profile.max_seq_len,
            "max_seq_len_source": max_seq_len_flag
            if "max_seq_len" in self.provider_options
            else MODEL_DEFAULT,
            "seed": self.seed,
            "checkpoint_every_minutes": self.checkpoint_every_minutes,
            "max_train_tokens": self.max_train_tokens,
            "microbatch_token_budget": self.microbatch_token_budget,
            "max_spend_usd": self.max_spend_usd,
            "hourly_rate_usd": self.hourly_rate_usd,
            "spend_reserve_fraction": self.spend_reserve_fraction,
            "provider_options": dict(self.provider_options),
        }

    def recorded_settings(
        self, profile: ModelProfile, *, max_seq_len_flag: str = MAX_SEQ_LEN_FLAG
    ) -> dict[str, Any]:
        """What ``run.json`` records under ``settings``: :meth:`resolved_settings`, plus what
        no flag changes (the pipeline depth and the optimizer's fixed parameters), which
        ``status`` shows and a preview leaves out."""
        adam = self.adam_config()
        return {
            **self.resolved_settings(profile, max_seq_len_flag=max_seq_len_flag),
            "pipeline_depth": self.pipeline_depth,
            "optimizer": {
                "name": "adam",
                "beta1": adam.beta1,
                "beta2": adam.beta2,
                "eps": adam.eps,
                "weight_decay": adam.weight_decay,
                "grad_clip_norm": adam.grad_clip_norm,
            },
        }


# --- What a preview says is billed -------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingUsage:
    """What a training run would use, for :meth:`Provider.training_billing`.

    ``epochs`` is how many epochs are left to train (all of them for a new run).
    ``validation_tokens_per_epoch`` counts the tokens forwarded each epoch to compute the
    validation loss, which trains nothing. ``max_spend_usd`` and ``hourly_rate_usd`` are the
    run's wall-clock spend guard, if any (see :class:`TrainSpec`).
    """

    tokens_per_epoch: int
    epochs: int
    validation_tokens_per_epoch: int = 0
    max_train_tokens: int | None = None
    max_spend_usd: float | None = None
    hourly_rate_usd: float | None = None
    resumed: bool = False
    """Whether this is what resuming a run would use, whose spend cap was set when it
    started."""

    @property
    def max_tokens(self) -> int:
        """The most tokens it trains on: every epoch left, or ``max_train_tokens`` if less."""
        total = self.tokens_per_epoch * self.epochs
        return total if self.max_train_tokens is None else min(total, self.max_train_tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingUsage:
    """What an evaluation's generations would use, for :meth:`Provider.sampling_billing`."""

    calls: int
    max_tokens_per_call: int


OFFLINE_BILLING: Final = "the fake provider is offline and bills nothing"
"""How the offline fake provider's billing sentences start, before ``: `` and what it would
use."""


def spend_cap_text(
    max_spend_usd: float | None, hourly_rate_usd: float | None, *, resumed: bool = False
) -> str:
    """The run's spend guard in words: a cap at a rate the user gave, or that there is none.

    ``resumed`` says it of a run being resumed, whose cap was set when it started (``resume``
    has no flag to change it)."""
    if max_spend_usd is None or hourly_rate_usd is None:
        if resumed:
            return "no spend cap (a run's cap is set when it starts; this one started without)"
        return "no spend cap (add --max-spend-usd with --hourly-rate-usd)"
    rate = (
        f"the ${hourly_rate_usd:g} per hour set when the run started"
        if resumed
        else (f"your ${hourly_rate_usd:g} per hour")
    )
    return (
        f"stops at about ${max_spend_usd:g} of wall-clock time at {rate} "
        "(an estimate, not the provider's bill)"
    )


def training_usage_text(usage: TrainingUsage) -> str:
    """``usage`` in words, for a provider's billing sentence: the tokens trained per epoch
    and at most, the validation tokens forwarded, and the spend guard."""
    epochs = f"{usage.epochs} epoch{'' if usage.epochs == 1 else 's'}"
    text = f"{usage.tokens_per_epoch:,} tokens per epoch, at most {usage.max_tokens:,}"
    capped = usage.max_tokens < usage.tokens_per_epoch * usage.epochs
    text += " (--max-train-tokens)" if capped else f" over {epochs}"
    if usage.validation_tokens_per_epoch:
        text += (
            f", plus {usage.validation_tokens_per_epoch:,} validation tokens forwarded per epoch"
        )
    cap = spend_cap_text(usage.max_spend_usd, usage.hourly_rate_usd, resumed=usage.resumed)
    return f"{text}; {cap}"


def sampling_usage_text(usage: SamplingUsage) -> str:
    """``usage`` in words, for a provider's billing sentence."""
    total = usage.calls * usage.max_tokens_per_call
    return (
        f"{usage.calls:,} generations of at most {usage.max_tokens_per_call:,} tokens each "
        f"(at most {total:,} generated tokens), each sent its conversation so far"
    )


class TransientError(ExternalError):
    """A provider call that failed in a way worth retrying: a timeout, a dropped connection.

    Backends raise it (or a :class:`TimeoutError` or :class:`ConnectionError`) only for calls
    that change nothing when repeated. The loop retries such failures for ``forward`` alone,
    never for ``forward_backward``, where a retry would accumulate the gradients twice.
    """

    error_code: ClassVar[str] = "transient"


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying for an idempotent call such as ``forward``."""
    return isinstance(exc, TransientError | TimeoutError | ConnectionError)


class ProviderCapacityError(ExternalError):
    """The provider has no capacity or quota for the request right now (e.g. HTTP 429).

    It does not by itself mean nothing is running: a call in the middle of a session can
    raise it while the session is still open. A provider that ends what a failed start had
    provisioned says so in the hint. It is not a :class:`TransientError`: the loop does not
    retry it, the user does, later.
    """

    error_code: ClassVar[str] = "provider_capacity"


class ProviderAuthError(ExternalError):
    """The provider rejected the API key (HTTP 401) or refused it the request (HTTP 403)."""

    error_code: ClassVar[str] = "provider_auth"


class ModelIdTaken(StateError):
    """:meth:`TrainingSession.promote` found its model id taken by a model it does not adopt:
    another run's, or one it cannot tell is this run's. ``hint`` names the ways out, since a
    resume fails the same way while that model holds the id."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            hint="train a new run under another run id, or delete that model and then resume "
            "this run",
        )


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
    """A promoted model: a checkpoint turned into a model the provider can deploy.

    ``model_id`` is the provider-local id it was promoted as (the ``model_id`` passed to
    :meth:`TrainingSession.promote`); ``uri`` is the provider's fully qualified name for it,
    e.g. ``accounts/<account>/models/<model_id>`` on Fireworks.
    """

    provider: str
    model_id: str
    uri: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Endpoint:
    """A deployment serving a model. ``inference_model`` is what chat requests name.

    ``url`` is the base URL of the OpenAI-compatible API that serves it, when it has one,
    and ``api_key_env`` the environment variable a client conventionally reads the key it
    sends as its bearer token from (never the key itself), when it needs one; ``name`` is
    the provider's full resource name for the deployment and ``shape`` the hardware
    configuration it runs on, when the provider has those.
    """

    provider: str
    deployment_id: str
    model_id: str
    inference_model: str
    url: str | None = None
    name: str | None = None
    shape: str | None = None
    api_key_env: str | None = None


DEFAULT_DEPLOY_TIMEOUT_S: Final = 30 * 60.0
"""How long :meth:`Provider.deploy` waits for a new deployment to serve, by default."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeployOptions:
    """How to deploy a model: what :meth:`Provider.deploy` is given.

    ``deployment_id`` fully matches :data:`MODEL_ID_PATTERN`, which every provider accepts.
    ``shape`` names the provider's hardware configuration; ``None`` lets the provider pick a
    validated one for the model. ``min_replicas`` ``0`` lets the deployment scale to zero
    when idle. ``timeout_s`` bounds the wait for the deployment to serve.
    """

    deployment_id: str
    shape: str | None = None
    min_replicas: int = 0
    max_replicas: int = 1
    timeout_s: float = DEFAULT_DEPLOY_TIMEOUT_S
    display_name: str | None = None

    def __post_init__(self) -> None:
        if not MODEL_ID_PATTERN.fullmatch(self.deployment_id):
            raise ValueError(
                "deployment_id must be lowercase letters, digits and dashes, starting with a "
                f"letter, at most {MAX_MODEL_ID_LEN} characters, got {self.deployment_id!r}"
            )
        if self.min_replicas < 0 or self.max_replicas < max(self.min_replicas, 1):
            raise ValueError("replica counts must satisfy 0 <= min_replicas <= max_replicas >= 1")
        if not (math.isfinite(self.timeout_s) and self.timeout_s > 0):
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentRequest:
    """What :meth:`Provider.deploy` is about to ask the provider to create.

    ``name`` is the resource name the deployment gets (:meth:`Provider.deployment_name`),
    ``base_model`` the model URI it serves, and ``shape`` the hardware configuration the
    request names, once resolved. A receipt records it before the create is sent, so a
    deployment whose creation was never confirmed can later be told apart from another one
    under the same id (:meth:`Provider.owns_deployment`).
    """

    deployment_id: str
    base_model: str
    name: str | None = None
    display_name: str | None = None
    shape: str | None = None
    min_replicas: int = 0
    max_replicas: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "base_model": self.base_model,
            "name": self.name,
            "display_name": self.display_name,
            "shape": self.shape,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        """The request :meth:`to_json` wrote; raises ``ValueError`` when it is malformed."""
        try:
            deployment_id, base_model = data["deployment_id"], data["base_model"]
            minimum, maximum = data.get("min_replicas", 0), data.get("max_replicas", 1)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"a deployment request lacks {exc}") from exc
        optional = {key: data.get(key) for key in ("name", "display_name", "shape")}
        if not (isinstance(deployment_id, str) and isinstance(base_model, str)) or any(
            value is not None and not isinstance(value, str) for value in optional.values()
        ):
            raise ValueError("a deployment request holds a value of the wrong type")
        if type(minimum) is not int or type(maximum) is not int:
            raise ValueError("a deployment request's replica counts are not integers")
        return cls(
            deployment_id=deployment_id,
            base_model=base_model,
            min_replicas=minimum,
            max_replicas=maximum,
            **optional,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SmokeTest:
    """One chat completion sent to a new deployment: whether it answered, and how fast.

    ``latency_s`` is the time of the request that answered (``None`` when none did);
    ``attempts`` counts the requests sent, retries while it scaled up included; ``detail``
    says why it failed (redacted), empty when it passed.
    """

    ok: bool
    latency_s: float | None = None
    attempts: int = 0
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "latency_s": self.latency_s,
            "attempts": self.attempts,
            "detail": self.detail,
        }


SMOKE_PROMPT: Final = "Reply with the single word OK."
SMOKE_MAX_TOKENS: Final = 16


def run_smoke_test(
    client: ChatCompleter,
    model: str,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_interval_s: float,
    timeout_s: float,
) -> SmokeTest:
    """One chat completion of ``model`` through ``client``, retried every ``poll_interval_s``
    seconds for up to ``timeout_s`` while the deployment scales up (a :class:`TransientError`
    or :class:`ProviderCapacityError`). Never raises for a failed answer, only when the API
    key is rejected; any other failure fails it at once."""
    deadline = clock() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        started = clock()
        try:
            client.complete(
                [{"role": "user", "content": SMOKE_PROMPT}],
                model=model,
                temperature=0.0,
                max_tokens=SMOKE_MAX_TOKENS,
            )
        except ProviderAuthError:
            raise
        except (TransientError, ProviderCapacityError) as exc:
            detail = redact(str(exc))
        except ExternalError as exc:
            return SmokeTest(ok=False, attempts=attempts, detail=redact(str(exc)))
        else:
            latency = max(0.0, clock() - started)
            return SmokeTest(ok=True, latency_s=round(latency, 3), attempts=attempts)
        remaining = deadline - clock()
        if remaining <= 0:
            return SmokeTest(ok=False, attempts=attempts, detail=detail)
        sleep(min(poll_interval_s, remaining))


class DeploymentNotReady(ExternalError):
    """A deployment that was created but did not come to serve.

    ``state`` is the provider's last reported state (e.g. ``FAILED``); ``timed_out`` is true
    when the wait ran out first. The deployment may still exist and bill: undeploy it.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str,
        state: str | None,
        timed_out: bool = False,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, source=source, hint=hint)
        self.state = state
        self.timed_out = timed_out


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Generation:
    """One sampled assistant turn, parsed into an OpenAI assistant message."""

    message: ChatMessage
    format_valid: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    stop_reason: str | None = None


# --- Remote runs (train --remote) ----------------------------------------------------------


class RemoteJobRejected(ExternalError):
    """Refused before anything that bills was created."""


class RemoteJobGone(ExternalError):
    """The provider no longer has the job, e.g. it was deleted."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RemotePlan:
    """Where ``train --remote`` would run a run's training loop, as :meth:`RemoteJobs.plan`
    resolved it, read only.

    ``root`` is the job's working directory, ``extra`` the extra the job installs, ``cli`` the
    provider CLI that submits it, ``secret`` the name of the secret the job reads its API key
    from (never the key), ``billing`` what the job bills, for the preview's billing fields,
    and ``warnings`` what the preview must say.
    """

    backend: str
    team: str
    team_source: str
    project: str
    root: str
    extra: str
    cli: str
    compute: Mapping[str, Any]
    secret: str
    tool_versions: Mapping[str, str]
    billing: BillingPart
    warnings: tuple[str, ...]

    def preview(self, uploads: Mapping[str, Any]) -> dict[str, Any]:
        """The plan as a preview shows it (``preview.remote``), with what the job is sent
        (``uploads``)."""
        return {
            "backend": self.backend,
            "team": self.team,
            "team_source": self.team_source,
            "project": self.project,
            "compute": dict(self.compute),
            "api_key_secret": self.secret,
            "uploads": dict(uploads),
            "baseten_cli": self.cli,
            **{f"{tool}_version": version for tool, version in self.tool_versions.items()},
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmittedJob:
    """The job :meth:`RemoteJobs.submit` created, as the provider reported it."""

    job_id: str
    project_id: str | None
    project: str
    compute: Mapping[str, Any]


JobState = Literal["queued", "running", "ended", "unknown"]
"""Where a remote job stands, whatever the provider calls its status."""


@dataclass(frozen=True, slots=True, kw_only=True)
class JobStatus:
    """A remote job as the provider reports it: its ``status`` and what that means."""

    job_id: str
    project_id: str
    status: str
    state: JobState
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class LogLine:
    """One line a remote job's command wrote, with its time in nanoseconds."""

    timestamp_ns: int
    message: str


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

    def promote(
        self,
        ckpt: CheckpointRef,
        model_id: str,
        *,
        adopt_since: datetime | None = None,
        on_sending: Callable[[], None] | None = None,
    ) -> ModelRef:
        """Turn a sampler checkpoint into a deployable model named ``model_id``.

        Only on a backend with :attr:`TrainingBackend.supports_promotion`. ``model_id``
        fully matches :data:`MODEL_ID_PATTERN` (lowercase letters, digits and dashes, starting
        with a letter, at most :data:`MAX_MODEL_ID_LEN` characters), which every provider
        accepts. The returned :class:`ModelRef` carries ``model_id`` as given and the
        provider's fully qualified name for the model as ``uri``.

        ``on_sending`` is called right before the promotion is sent, after every lookup it
        needs, so the caller records that a model under ``model_id`` may be the run's only
        once one can be. A promotion can succeed on the provider while its response is lost,
        and a retry then finds ``model_id`` taken. ``adopt_since`` (shortly before the run sent
        that promotion) lets a provider that can look models up return such a model instead:
        one named ``model_id``, of the same base model, created no earlier than
        ``adopt_since``, so it is this run's and not another run's that used the same id. A
        provider that cannot tell ignores it. A model it finds under ``model_id`` and does not
        adopt raises :class:`ModelIdTaken`.
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
        stopped early or failed, and raises what :func:`raise_session_failure` picks.
        """
        ...


BODY_FAILED_FIRST = "the session failed first"


def raise_session_failure(body: BaseException | None, closing: Sequence[BaseException]) -> None:
    """Raise what a closed session ends with: ``body`` is what its body raised (``None``
    after a normal exit) and ``closing`` every failure to close it, most important first.

    The first interrupt (a :class:`BaseException` that is not an :class:`Exception`, such as
    ``KeyboardInterrupt``), from the body or from closing, wins, so the command reports it as
    interrupted; else the body's error; else the first failure to close. Every other one, and
    its own notes, become notes on it, so nothing is lost. Returns when nothing failed.
    """
    failures = [*([] if body is None else [body]), *closing]
    if not failures:
        return
    raised = next((exc for exc in failures if not isinstance(exc, Exception)), failures[0])
    for other in failures:
        if other is raised:
            continue
        label = BODY_FAILED_FIRST if other is body else "ending the session also failed"
        raised.add_note(f"{label}: {_described(other)}")
        for note in getattr(other, "__notes__", ()):
            raised.add_note(note)
    raise raised


def _described(error: BaseException) -> str:
    detail = str(error)
    hint = error.hint if isinstance(error, FinetuneError) else None
    text = f"{type(error).__name__}{': ' + detail if detail else ''}"
    return redact("; ".join(filter(None, (text, hint))))


@runtime_checkable
class Provider(Protocol):
    """Everything a training provider offers beyond the loop. Constructing one is cheap.

    Readiness is not the provider's to report: ``doctor`` checks what the registry says it
    needs (:class:`~pup_finetune.providers.ProviderInfo`, :mod:`pup_finetune.readiness`),
    offline, and calls only :meth:`models` and :meth:`preflight` for ``--check``.
    :meth:`resolve_model` and :meth:`preflight` raise :class:`UsageError` for an unknown or unusable model;
    :meth:`validate_options` returns the normalized ``provider_options`` or raises
    :class:`UsageError`. :meth:`deploy` and :meth:`undeploy` are remote writes: callers gate
    them behind ``--confirm`` and ``PUP_READ_ONLY``; :meth:`smoke_test` is a paid call.
    """

    @property
    def name(self) -> str:
        """The provider's registry name, e.g. ``fireworks``."""
        ...

    def models(self) -> list[ModelProfile]:
        """Every model profile this provider supports."""
        ...

    def resolve_model(self, alias_or_id: str) -> ModelProfile:
        """The profile any of whose :attr:`~ModelProfile.names` is ``alias_or_id``
        (:func:`find_profile`)."""
        ...

    def preflight(self, profile: ModelProfile) -> None:
        """Confirm with the provider that ``profile`` can be trained right now."""
        ...

    def renderer(self, profile: ModelProfile) -> Renderer:
        """The renderer for ``profile``, with its tokenizer pinned to the profile's revision."""
        ...

    def training_profile(self, profile: ModelProfile) -> ModelProfile:
        """``profile`` as a run trains it under the validated provider options.

        ``profile`` itself unless an option narrows it (e.g. caps its context). Call it after
        :meth:`validate_options`: rows are rendered against, and a run records, the profile
        it returns. Raises :class:`UsageError` when an option does not fit ``profile``.
        """
        ...

    def training_billing(self, usage: TrainingUsage) -> str:
        """One sentence for a training preview: what this provider bills a run for, and
        ``usage`` (:func:`training_usage_text`). Never a price it cannot know."""
        ...

    def sampling_billing(self, usage: SamplingUsage) -> str:
        """One sentence for an evaluation preview: what this provider bills the base and
        tuned models' generations for, and ``usage`` (:func:`sampling_usage_text`). Never a
        price it cannot know."""
        ...

    def backend(self) -> TrainingBackend:
        """The training backend."""
        ...

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Check provider-specific ``TrainSpec.provider_options``; return them normalized."""
        ...

    def deploy(
        self,
        model: ModelRef,
        options: DeployOptions,
        *,
        on_creating: Callable[[DeploymentRequest], None] | None = None,
        on_created: Callable[[Endpoint, str], None] | None = None,
    ) -> Endpoint:
        """Deploy a promoted model and wait (at most ``options.timeout_s``) until it serves.

        ``on_creating`` is called with what the create sends right before it is sent, after
        every lookup it needs (e.g. the shape), so the caller records a deployment that may
        come to exist only once one can, and a failure before it leaves nothing to undeploy.
        ``on_created`` is called with the endpoint and the provider's state for it as soon as
        the provider accepted the deployment, before the wait, so the caller can record what
        now exists (and bills) even if the wait fails. Raises :class:`DeploymentNotReady` when
        the deployment fails or the wait runs out; the deployment may then still exist.
        """
        ...

    def deployment_name(self, deployment_id: str) -> str | None:
        """The full resource name the deployment ``deployment_id`` gets, before it exists.

        ``None`` when the provider names deployments by id alone. The request passed to
        ``on_creating`` carries it, so a receipt left by an interrupted create still says
        where it was made (e.g. which account), and :meth:`undeploy` can refuse to delete in
        another one.
        """
        ...

    def owns_deployment(self, request: DeploymentRequest) -> bool | None:
        """Whether the deployment under ``request.deployment_id`` is the one ``request``
        asked for: ``True`` when it is, ``False`` when the id holds another deployment,
        ``None`` when there is none.

        For a receipt whose creation was never confirmed: the create may have failed because
        the id was taken, so only a deployment that matches the request (its model, display
        name and shape, as far as the provider reports them) may be deleted as this run's.
        Raises :class:`UsageError` when ``request.name`` places it where this provider
        cannot look (e.g. another account).
        """
        ...

    def smoke_test(self, endpoint: Endpoint) -> SmokeTest:
        """Send one short chat completion to ``endpoint``; never raises for a failed answer.

        Retries while the deployment scales up from zero.
        """
        ...

    def undeploy(self, endpoint: Endpoint) -> bool:
        """Tear a deployment down so it stops billing; the deployed model is kept.

        Idempotent: returns ``False`` when the deployment was already gone, ``True`` when
        this call deleted it.
        """
        ...


class RemoteJobs(Protocol):
    """Runs a run's training loop in a provider job (``train --remote``) and reads it back.

    :meth:`plan` and every read are free and change nothing; :meth:`submit` creates a job that
    bills, sent once and never retried. It raises :class:`RemoteJobRejected`,
    :class:`ProviderAuthError` or :class:`ProviderCapacityError` only when nothing was
    created, and any other :class:`ExternalError` when the job may exist.
    """

    def plan(self, *, run_id: str, team: str | None) -> RemotePlan:
        """Where run ``run_id`` would run, in ``team`` (``None``: the default team); raises
        :class:`UsageError` when it cannot."""
        ...

    def submit(self, plan: RemotePlan, staged: Path, command: Sequence[str]) -> SubmittedJob:
        """Upload ``staged`` and start ``command`` in a new job, as ``plan`` says."""
        ...

    def find(self, *, team: str, project: str, after: datetime) -> list[JobStatus]:
        """The jobs of ``project`` in ``team`` created at ``after`` or later (or at an
        unknown time)."""
        ...

    def job(self, *, project_id: str, job_id: str) -> JobStatus:
        """The job ``job_id`` as it stands now; :class:`RemoteJobGone` when there is none."""
        ...

    def output_pages(
        self, job: JobStatus, *, since: datetime | None, now: datetime
    ) -> Iterator[list[LogLine]]:
        """The lines ``job``'s command wrote, page by page, newest first, back to about
        ``since`` (by default, the job's start)."""
        ...

    def failure(self, job: JobStatus, *, now: datetime) -> str | None:
        """Why ``job`` failed, as the provider logged it, if it did; redacted and short."""
        ...

    def trainer_active(self, session_id: str) -> bool | None:
        """Whether the trainer of training session ``session_id`` is still active (and
        bills); ``None`` when that cannot be read."""
        ...

    def logs_command(self, job_id: str) -> str:
        """The command that shows job ``job_id``'s logs."""
        ...


@runtime_checkable
class SupportsRemote(Protocol):
    """A provider that can run a run's training loop in a job of its own."""

    def remote_jobs(self) -> RemoteJobs:
        """The remote jobs; builds nothing remote."""
        ...
