# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training on Fireworks: the :class:`~pup_finetune.providers.base.TrainingBackend` seam.

:class:`FireworksBackend` opens one serverless training session per
:meth:`~FireworksBackend.open_session` and always ends it on the way out, whether training
finished, stopped early, failed or was interrupted. Results still in flight are waited for
(briefly) when the session ends normally and abandoned when it ends with an error, so an
interrupt never blocks on the provider. A session that cannot start for lack of capacity
leaves nothing running.

:class:`FireworksSession` maps the shared loop's calls onto a
:class:`~pup_finetune.providers.fireworks.client.TrainingRun`:

- ``forward_backward`` reports Fireworks' own ``loss:sum`` over ``response_tokens``;
  ``forward`` reports no loss, so the validation loss is computed from the per-token logprobs
  it returns: ``loss_sum = -sum(logprob * weight)`` over every target token.
- Training checkpoints (``save_state``) are resumable in a new session; sampler checkpoints
  (``save_weights_for_sampler``) can be sampled from and promoted only while the session that
  saved them lives. Names are checked against the 17-character limit before saving, since
  Fireworks only enforces it when promoting.
- ``promote`` finds the checkpoint in the session's listing, where Fireworks has renamed it
  (``run-<hex>-<name>-<8 hex>``): the newest promotable sampler row carrying the name. It
  first looks the model up: one that already exists is adopted when it can only be this
  run's (the same base model, created since ``adopt_since``, shortly before the run first
  asked for a promotion), because an
  earlier attempt's promotion succeeded but its response was lost; any other is refused
  with a :class:`~pup_finetune.errors.StateError` rather than promoted into a conflict. A
  promotion that fails in a way that can hide a success (a timeout, a dropped connection, a
  server error, a conflict) looks the model up again and adopts it when it is there now.

:class:`FireworksSampler` renders the prompt with the profile's renderer, samples greedily up
to the renderer's stop tokens and parses the completion back into an OpenAI message; one cut
off at ``max_tokens`` is never ``format_valid``. Only sampling is retried (a bounded number of
times, with backoff), never a training call.

Nothing here imports the Fireworks SDK: the backend talks to
:class:`~pup_finetune.providers.fireworks.client.FireworksTraining`, which a factory builds
when a session opens.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from pup_finetune.errors import ExternalError, FinetuneError, StateError, UsageError
from pup_finetune.providers.base import (
    MAX_MODEL_ID_LEN,
    MODEL_ID_PATTERN,
    AdamConfig,
    CheckpointRef,
    Generation,
    LoraConfig,
    ModelProfile,
    ModelRef,
    PendingResult,
    ProviderAuthError,
    ProviderCapacityError,
    StepMetrics,
    TransientError,
    is_transient,
)
from pup_finetune.providers.fireworks import PROVIDER_NAME
from pup_finetune.providers.fireworks.client import (
    MAX_CHECKPOINT_NAME_LEN,
    SAMPLER_CHECKPOINT,
    SOURCE,
    CheckpointRow,
    FireworksTraining,
    ForwardOutput,
    SampledTokens,
    TokenSampler,
    TrainingRun,
    TrainingService,
    sort_rows,
)
from pup_finetune.redaction import redact
from pup_finetune.rendering.base import ChatMessage, Renderer, ToolSpec
from pup_finetune.rendering.tokens import TokenizedExample

if TYPE_CHECKING:
    from pup_finetune.providers.base import Sampler, TrainingBackend, TrainingSession

__all__ = [
    "DRAIN_TIMEOUT_S",
    "EXPIRED_HINT",
    "SAMPLE_ATTEMPTS",
    "FireworksBackend",
    "FireworksSampler",
    "FireworksSession",
    "eval_metrics",
    "promotable_row",
    "train_metrics",
]

DRAIN_TIMEOUT_S: Final = 120.0
"""How long, in all, a session that ends normally waits for results still in flight."""
SAMPLE_ATTEMPTS: Final = 3
"""How many times a sample is tried when it fails transiently (a timeout, a dropped
connection, a server error)."""
SAMPLE_BACKOFF_S: Final = 2.0
"""The wait before the second attempt; it doubles for each attempt after that."""
SAMPLE_TIMEOUT_S: Final = 600.0
"""How long one sample may take before it counts as timed out (and is retried)."""
EXPIRED_HINT: Final = (
    "Fireworks ends a session (and its sampler checkpoints) shortly after it closes; resume "
    "from a training checkpoint and save a new sampler checkpoint"
)

_LOSS_SUM: Final = "loss:sum"
_RESPONSE_TOKENS: Final = "response_tokens"
_TOTAL_TOKENS: Final = "total_tokens:sum"
_SUFFIX: Final = r"(?:-[0-9a-f]{8})?"
_LENGTH: Final = "length"


# --- Metrics -----------------------------------------------------------------------------


def train_metrics(output: ForwardOutput, batch: Sequence[TokenizedExample]) -> StepMetrics:
    """The :class:`StepMetrics` of a forward-backward pass over ``batch``.

    Fireworks reports ``loss:sum`` (the weighted sum of per-token losses), ``response_tokens``
    (the sum of the weights) and ``total_tokens:sum``; the weight falls back to the batch's own
    weights, and the token count to its input tokens, when either is missing. Raises
    :class:`ExternalError` when the loss is missing.
    """
    metrics = output.metrics
    if _LOSS_SUM not in metrics:
        raise ExternalError(
            f"Fireworks reported no {_LOSS_SUM} for a forward-backward pass", source=SOURCE
        )
    weight = metrics.get(_RESPONSE_TOKENS)
    return StepMetrics(
        loss_sum=float(metrics[_LOSS_SUM]),
        weight=float(weight) if weight is not None else _weight(batch),
        tokens=int(metrics.get(_TOTAL_TOKENS, _tokens(batch))),
    )


def eval_metrics(output: ForwardOutput, batch: Sequence[TokenizedExample]) -> StepMetrics:
    """The :class:`StepMetrics` of a forward pass, computed from its per-token logprobs.

    A forward pass reports no loss, so ``loss_sum`` is ``-sum(logprob * weight)`` over every
    target token of every example and ``weight`` the sum of the weights. Raises
    :class:`ExternalError` when Fireworks returned logprobs for a different number of
    examples, or of tokens, than were sent: a loss is never computed from a partial answer.
    """
    if len(output.logprobs) != len(batch):
        raise ExternalError(
            f"Fireworks returned logprobs for {len(output.logprobs)} examples in a batch of "
            f"{len(batch)}",
            source=SOURCE,
        )
    loss_sum = 0.0
    for index, (logprobs, example) in enumerate(zip(output.logprobs, batch, strict=True)):
        if len(logprobs) != len(example.target_tokens):
            raise ExternalError(
                f"Fireworks returned {len(logprobs)} logprobs for example {index}, which has "
                f"{len(example.target_tokens)} target tokens",
                source=SOURCE,
            )
        loss_sum -= sum(lp * w for lp, w in zip(logprobs, example.weights, strict=True))
    return StepMetrics(
        loss_sum=loss_sum,
        weight=_weight(batch),
        tokens=int(output.metrics.get(_TOTAL_TOKENS, _tokens(batch))),
    )


def _weight(batch: Sequence[TokenizedExample]) -> float:
    return float(sum(sum(example.weights) for example in batch))


def _tokens(batch: Sequence[TokenizedExample]) -> int:
    return sum(len(example.input_tokens) for example in batch)


# --- Promotion ---------------------------------------------------------------------------


def promotable_row(rows: Sequence[CheckpointRow], ckpt: CheckpointRef) -> CheckpointRow:
    """The listed row to promote for the sampler checkpoint ``ckpt``.

    Fireworks lists a sampler checkpoint saved as ``<name>`` as ``run-<hex>-<name>-<8 hex>``.
    The candidates are the sampler rows carrying ``ckpt.name`` that way; when some of them
    are exactly the saved path's last segment (``<name>-<8 hex>``), only those. Of the
    promotable candidates, the newest by ``createTime`` wins. Raises :class:`ExternalError`
    when no sampler row carries the name, or none that does is promotable.
    """
    pattern = re.compile(rf"(?:^|-){re.escape(ckpt.name)}{_SUFFIX}$")
    candidates = [
        row
        for row in rows
        if row.checkpoint_type == SAMPLER_CHECKPOINT and pattern.search(row.checkpoint_name)
    ]
    segment = ckpt.uri.rstrip("/").rsplit("/", 1)[-1]
    exact = [
        row
        for row in candidates
        if row.checkpoint_name == segment or row.checkpoint_name.endswith(f"-{segment}")
    ]
    candidates = exact or candidates
    if not candidates:
        raise ExternalError(
            f"Fireworks lists no sampler checkpoint named {ckpt.name!r} in the session "
            f"{ckpt.created_in_session}",
            source=SOURCE,
        )
    promotable = [row for row in candidates if row.promotable]
    if not promotable:
        raise ExternalError(
            f"Fireworks lists the sampler checkpoint {ckpt.name!r} as not promotable",
            source=SOURCE,
            hint="save a new sampler checkpoint and promote that one",
        )
    return sort_rows(promotable)[-1]


def _model_name(row: CheckpointRow, model_id: str) -> str:
    """The promoted model's name when Fireworks' answer omits it: in the checkpoint's account.

    Raises :class:`ExternalError` when the row's resource name names no account.
    """
    match = re.fullmatch(r"accounts/([^/]+)/trainingSessions/.+", row.resource_name)
    if match is None:
        raise ExternalError(
            f"Fireworks promoted the model {model_id!r} without naming it, and the checkpoint "
            f"{row.resource_name!r} names no account to find it in",
            source=SOURCE,
        )
    return f"accounts/{match.group(1)}/models/{model_id}"


def _session_model_name(session_name: str, model_id: str) -> str:
    """The name ``model_id`` is promoted as from the session ``session_name``: in its account.

    Raises :class:`ExternalError` when the session's resource name names no account.
    """
    match = re.fullmatch(r"accounts/([^/]+)/trainingSessions/[^/]+", session_name)
    if match is None:
        raise ExternalError(
            f"the training session {session_name!r} names no account to promote into",
            source=SOURCE,
        )
    return f"accounts/{match.group(1)}/models/{model_id}"


def _record_base_model(record: dict[str, Any]) -> str | None:
    """The base model a Fireworks model record names (a LoRA's ``peftDetails.baseModel``)."""
    peft = record.get("peftDetails")
    base = peft.get("baseModel") if isinstance(peft, dict) else None
    base = base or record.get("baseModel")
    return base if isinstance(base, str) and base else None


def _record_time(record: dict[str, Any]) -> datetime | None:
    """A Fireworks model record's ``createTime``, or ``None`` when missing or unreadable."""
    value = record.get("createTime")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _checked_model_id(model_id: str) -> str:
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise UsageError(
            f"{model_id!r} is not a valid Fireworks model id",
            hint=f"use 1 to {MAX_MODEL_ID_LEN} lowercase letters, digits and dashes, starting "
            "with a letter and not ending with a dash",
        )
    return model_id


# --- In-flight results -------------------------------------------------------------------


class _Tracked[S, T]:
    """A pending result converted when it resolves; ``done`` once it was waited for.

    The session waits, when it ends, only for the results nobody waited for.
    """

    def __init__(self, inner: PendingResult[S], convert: Callable[[S], T]) -> None:
        self._inner = inner
        self._convert = convert
        self.done = False

    def result(self, timeout: float | None = None) -> T:
        try:
            value = self._inner.result(timeout=timeout)
        except BaseException as exc:
            # A wait that ran out leaves the call in flight: the session still waits for it.
            if not _is_wait_timeout(exc):
                self.done = True
            raise
        self.done = True
        return self._convert(value)


def _is_wait_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` is a wait for a result running out, not the call failing.

    A standard future raises a bare :class:`TimeoutError`; the client maps it, like any SDK
    failure, to a :class:`TransientError` raised from it. Only the direct cause counts: a call
    that failed on an HTTP timeout chains a ``TimeoutError`` further down, and it is done.
    """
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, TransientError) and isinstance(exc.__cause__, TimeoutError)


def _nothing(_value: object) -> None:
    return None


# --- The sampler -------------------------------------------------------------------------


class _StopTokenDecoder:
    """The tokenizer the SDK's sampler is given: it only decodes the stop token ids."""

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self._renderer.decode(list(token_ids))


class FireworksSampler:
    """Greedy generation from a base model or sampler checkpoint; see the module docstring.

    ``sleep`` and ``clock`` (seconds) replace the real ones, for tests.
    """

    def __init__(
        self,
        sampler: TokenSampler,
        renderer: Renderer,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sampler = sampler
        self._renderer = renderer
        self._sleep = sleep
        self._clock = clock
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        if self._closed:
            raise StateError("this Fireworks sampler is closed")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        prompt = self._renderer.generation_prompt_tokens(messages, tools)
        stop = self._renderer.stop_token_ids
        sampled, latency = self._sample(prompt, max_tokens=max_tokens, stop=stop)
        parsed = self._renderer.parse_completion(sampled.tokens, tools)
        return Generation(
            message=parsed.message,
            # A completion cut off at max_tokens is not a whole turn, whatever it parses as.
            format_valid=parsed.format_valid and sampled.stop_reason != _LENGTH,
            prompt_tokens=len(prompt),
            completion_tokens=len(sampled.tokens),
            latency_s=latency,
            stop_reason=sampled.stop_reason,
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._sampler.close()

    def _sample(
        self, prompt: list[int], *, max_tokens: int, stop: list[int]
    ) -> tuple[SampledTokens, float]:
        """The sampled tokens and the seconds the successful attempt took."""
        delay = SAMPLE_BACKOFF_S
        for attempt in range(1, SAMPLE_ATTEMPTS + 1):
            started = self._clock()
            try:
                pending = self._sampler.sample(
                    prompt, max_tokens=max_tokens, stop=stop, temperature=0.0
                )
                sampled = pending.result(timeout=SAMPLE_TIMEOUT_S)
                return sampled, max(self._clock() - started, 0.0)
            except Exception as exc:
                if not is_transient(exc):
                    raise
                if attempt == SAMPLE_ATTEMPTS:
                    exc.add_note(f"sampling failed after {SAMPLE_ATTEMPTS} attempts")
                    raise
                _warn(
                    f"sampling failed ({_text(exc)}); retrying in {delay:g}s "
                    f"(attempt {attempt + 1} of {SAMPLE_ATTEMPTS})"
                )
            self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover


# --- The session -------------------------------------------------------------------------


class FireworksSession:
    """One live Fireworks training session; built by :meth:`FireworksBackend.open_session`."""

    def __init__(
        self,
        backend: FireworksBackend,
        *,
        client: FireworksTraining,
        service: TrainingService,
        run: TrainingRun,
        profile: ModelProfile,
        max_seq_len: int,
    ) -> None:
        self._backend = backend
        self._client = client
        self._service = service
        self._run = run
        self._profile = profile
        self._max_seq_len = max_seq_len
        self._session_id = service.session_id
        self._session_name = service.session_name
        self._in_flight: list[_Tracked[Any, Any]] = []
        self._samplers: list[FireworksSampler] = []
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_name(self) -> str:
        """The session's full resource name, ``accounts/<a>/trainingSessions/<id>``."""
        return self._session_name

    @property
    def run_id(self) -> str | None:
        """Fireworks' id for the LoRA run (``run-<hex>``), when it reports one."""
        return self._run.run_id

    @property
    def closed(self) -> bool:
        return self._closed

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        examples = self._checked_batch(batch)
        pending = _Tracked(
            self._run.forward_backward(examples), lambda o: train_metrics(o, examples)
        )
        return self._track(pending)

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        examples = self._checked_batch(batch)
        pending = _Tracked(self._run.forward(examples), lambda o: eval_metrics(o, examples))
        return self._track(pending)

    def optim_step(self, adam: AdamConfig) -> PendingResult[None]:
        self._require_open()
        return self._track(_Tracked(self._run.optim_step(adam), _nothing))

    def save_training_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        self._require_open()
        self._check_name(name)
        path = self._run.save_state(name).result()
        return self._ref("training", name, path, epoch=epoch, step=step)

    def save_sampler_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        self._require_open()
        self._check_name(name)
        path = self._run.save_weights_for_sampler(name).result()
        return self._ref("sampler", name, path, epoch=epoch, step=step)

    def load_weights(self, ckpt: CheckpointRef) -> None:
        self._require_open()
        if ckpt.kind != "training":
            raise UsageError(
                "Fireworks loads weights from a training checkpoint, not a sampler checkpoint"
            )
        self._run.load_state(ckpt.uri).result()

    def promote(
        self, ckpt: CheckpointRef, model_id: str, *, adopt_since: datetime | None = None
    ) -> ModelRef:
        self._require_open()
        _checked_model_id(model_id)
        session_name = self._live_session_name(ckpt, "promote")
        expected = _session_model_name(session_name, model_id)
        existing = self._client.get_model(expected)
        if existing is not None:
            # Taken before this call: an earlier attempt's promotion whose response was lost,
            # or another run's model under the same id; adopt_since tells them apart.
            return self._adopted(existing, expected, model_id, since=adopt_since, checked=False)
        row = promotable_row(self._client.list_session_checkpoints(session_name), ckpt)
        try:
            model = self._client.promote_session_checkpoint(
                row.resource_name, output_model_id=model_id, base_model=self._profile.base_model
            )
        except (ProviderAuthError, ProviderCapacityError):
            raise
        except ExternalError as exc:
            # A timeout, a dropped connection, a server error or a conflict can hide a
            # promotion that took place. The model was absent just before, so one that is
            # there now is this call's.
            try:
                existing = self._client.get_model(expected)
            except FinetuneError:
                raise exc from None
            if existing is None:
                raise
            return self._adopted(existing, expected, model_id, since=None, checked=True)
        name = model.get("name")
        if not isinstance(name, str) or not name:
            name = _model_name(row, model_id)
        return ModelRef(provider=PROVIDER_NAME, model_id=model_id, uri=name)

    def _adopted(
        self,
        record: dict[str, Any],
        expected: str,
        model_id: str,
        *,
        since: datetime | None,
        checked: bool,
    ) -> ModelRef:
        """The :class:`ModelRef` of ``record``, the existing model ``expected``, once it can
        only be this run's; :class:`StateError` otherwise.

        ``checked`` says the model was absent when this promotion started, so only its base
        model is compared; otherwise it must also have been created no earlier than
        ``since``, just before the run first asked for a promotion.
        """
        name = record.get("name")
        name = name if isinstance(name, str) and name else expected
        taken = (
            f"the Fireworks model {name} already exists, so the best checkpoint cannot be "
            f"promoted to the model id {model_id!r}"
        )
        hint = "train a new run under another run id, or delete that model in Fireworks"
        base = _record_base_model(record)
        if base is not None and base != self._profile.base_model:
            raise StateError(
                f"{taken}: it is a model of {base}, not of {self._profile.base_model}", hint=hint
            )
        if not checked:
            if since is None:
                raise StateError(taken, hint=hint)
            created = _record_time(record)
            if created is None:
                raise StateError(
                    f"{taken}, and Fireworks does not say when it was created, so it cannot "
                    "be told apart from another run's",
                    hint=hint,
                )
            if created < since:
                raise StateError(
                    f"{taken}: it was created at {created.isoformat()}, before this run "
                    f"first asked to promote it (allowing for clock skew, {since.isoformat()})",
                    hint=hint,
                )
        _warn(f"adopting the Fireworks model {name}, which this run already promoted")
        return ModelRef(provider=PROVIDER_NAME, model_id=model_id, uri=name)

    def sampler(self, ckpt: CheckpointRef | None) -> FireworksSampler:
        self._require_open()
        renderer = self._backend.renderer_for(self._profile)
        tokenizer = _StopTokenDecoder(renderer)
        if ckpt is None:
            token_sampler = self._service.create_sampling_client(
                tokenizer=tokenizer, base_model=self._profile.base_model
            )
        else:
            self._live_session_name(ckpt, "sample from")
            token_sampler = self._service.create_sampling_client(
                tokenizer=tokenizer, model_path=ckpt.uri
            )
        sampler = FireworksSampler(
            token_sampler, renderer, sleep=self._backend.sleep, clock=self._backend.clock
        )
        self._samplers.append(sampler)
        return sampler

    def list_checkpoints(self) -> list[CheckpointRow]:
        """Every checkpoint this session saved, as Fireworks lists them, oldest first."""
        self._require_open()
        return self._client.list_session_checkpoints(self._session_name)

    # Ending the session.

    def end(self, *, abandon: bool) -> list[BaseException]:
        """Stop accepting calls, settle what is in flight and close the samplers.

        Results in flight are waited for, up to :data:`DRAIN_TIMEOUT_S` in all, unless
        ``abandon``. Returns the failures of closing the samplers; the service is closed by
        the backend.
        """
        self._closed = True
        if not abandon:
            self._drain()
        self._in_flight.clear()
        errors: list[BaseException] = []
        for sampler in self._samplers:
            try:
                sampler.close()
            except Exception as exc:
                errors.append(exc)
        return errors

    def _drain(self) -> None:
        clock = self._backend.clock
        deadline = clock() + DRAIN_TIMEOUT_S
        for pending in self._in_flight:
            if pending.done:
                continue
            try:
                pending.result(timeout=max(deadline - clock(), 0.0))
            except Exception as exc:
                if _is_wait_timeout(exc):
                    _warn(
                        "a Fireworks call still in flight when the session ended was abandoned "
                        f"(the session waits {DRAIN_TIMEOUT_S:g}s in all for such calls)"
                    )
                else:
                    _warn(
                        "a Fireworks call still in flight when the session ended failed: "
                        f"{_text(exc)}"
                    )

    # Internals.

    def _track[S, T](self, pending: _Tracked[S, T]) -> _Tracked[S, T]:
        self._prune()
        self._in_flight.append(pending)
        return pending

    def _prune(self) -> None:
        self._in_flight = [pending for pending in self._in_flight if not pending.done]

    def _checked_batch(self, batch: Sequence[TokenizedExample]) -> list[TokenizedExample]:
        self._require_open()
        examples = list(batch)
        if not examples:
            raise ValueError("a batch needs at least one example")
        longest = max(example.num_tokens for example in examples)
        if longest > self._max_seq_len:
            raise UsageError(
                f"an example has {longest} tokens, over this session's "
                f"{self._max_seq_len}-token limit"
            )
        return examples

    def _check_name(self, name: str) -> None:
        limit = self._backend.max_checkpoint_name_len
        if not name or len(name) > limit:
            raise ValueError(
                f"Fireworks checkpoint names must be 1 to {limit} characters, got "
                f"{len(name)}: {name!r}"
            )

    def _ref(
        self, kind: Literal["training", "sampler"], name: str, path: str, *, epoch: int, step: int
    ) -> CheckpointRef:
        return CheckpointRef(
            kind=kind,
            name=name,
            uri=path,
            epoch=epoch,
            step=step,
            created_in_session=self._session_id,
        )

    def _live_session_name(self, ckpt: CheckpointRef, action: str) -> str:
        """The resource name of ``ckpt``'s session, which must still be open."""
        if ckpt.kind != "sampler":
            raise UsageError(f"can only {action} a sampler checkpoint, got a {ckpt.kind} one")
        session_name = self._backend.live_session_name(ckpt.created_in_session)
        if session_name is None:
            raise ExternalError(
                f"cannot {action} the sampler checkpoint {ckpt.name!r}: its session "
                f"{ckpt.created_in_session} has ended",
                source=SOURCE,
                hint=EXPIRED_HINT,
            )
        return session_name

    def _require_open(self) -> None:
        if self._closed:
            raise StateError(f"the Fireworks training session {self._session_id} has ended")


# --- The backend -------------------------------------------------------------------------


class FireworksBackend:
    """Opens Fireworks training sessions; see the module docstring.

    ``client`` builds the account's :class:`FireworksTraining` client when a session opens
    (resolving the API key then, not before); each session gets its own and closes it.
    ``renderer`` gives a profile's renderer, for sampling. ``max_seq_len`` (the provider
    option) caps the longest example a session trains on, below the profile's context.
    ``sleep`` and ``clock`` replace the real ones for the samplers' retries, for tests.
    """

    def __init__(
        self,
        client: Callable[[], FireworksTraining],
        *,
        renderer: Callable[[ModelProfile], Renderer],
        max_seq_len: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_factory = client
        self._renderer_factory = renderer
        self._max_seq_len = max_seq_len
        self.sleep = sleep
        self.clock = clock
        self._renderers: dict[str, Renderer] = {}
        self._live: dict[str, str] = {}

    @property
    def max_checkpoint_name_len(self) -> int:
        return MAX_CHECKPOINT_NAME_LEN

    @property
    def supports_resume(self) -> bool:
        return True

    @property
    def supports_promotion(self) -> bool:
        return True

    @contextmanager
    def open_session(
        self,
        profile: ModelProfile,
        lora: LoraConfig,
        *,
        resume_from: CheckpointRef | None = None,
        seed: int,
        max_seq_len: int,
    ) -> Iterator[FireworksSession]:
        self._check_request(profile, resume_from, seed=seed, max_seq_len=max_seq_len)
        client = self._client_factory()
        service: TrainingService | None = None
        session: FireworksSession | None = None
        failure: BaseException | None = None
        try:
            service = client.service()
            if resume_from is not None:
                run = service.create_training_client_from_state_with_optimizer(resume_from.uri)
            else:
                run = service.create_lora_training_client(
                    profile.base_model, lora.rank, alpha=lora.alpha, seed=seed
                )
            session = FireworksSession(
                self,
                client=client,
                service=service,
                run=run,
                profile=profile,
                max_seq_len=max_seq_len,
            )
            self._live[session.session_id] = session.session_name
            yield session
        except BaseException as exc:
            failure = exc
            raise
        finally:
            errors = self._end(client, service, session, abandon=failure is not None)
            if errors:
                # After a normal exit the first failure (closing the service, if that failed:
                # a session that may still be open) is raised; every other one is a note.
                raised = failure if failure is not None else errors[0]
                for error in errors if failure is not None else errors[1:]:
                    raised.add_note(f"ending the Fireworks session also failed: {_text(error)}")
                if failure is None:
                    raise raised

    def live_session_name(self, session_id: str) -> str | None:
        """The resource name of the session ``session_id`` while it is open here, else ``None``."""
        return self._live.get(session_id)

    def renderer_for(self, profile: ModelProfile) -> Renderer:
        """``profile``'s renderer, built once per backend."""
        renderer = self._renderers.get(profile.alias)
        if renderer is None:
            renderer = self._renderers[profile.alias] = self._renderer_factory(profile)
        return renderer

    def _check_request(
        self,
        profile: ModelProfile,
        resume_from: CheckpointRef | None,
        *,
        seed: int,
        max_seq_len: int,
    ) -> None:
        if profile.provider != PROVIDER_NAME:
            raise UsageError(f"Fireworks cannot train {profile.provider}/{profile.alias}")
        if not 0 < max_seq_len <= profile.max_seq_len:
            raise UsageError(
                f"max_seq_len must be 1 to {profile.max_seq_len} for {profile.alias}, "
                f"got {max_seq_len}"
            )
        if self._max_seq_len is not None and max_seq_len > self._max_seq_len:
            raise UsageError(
                f"a session with max_seq_len {max_seq_len} was requested, over the max_seq_len "
                f"provider option of {self._max_seq_len}",
                hint="train a new run with a larger --max-seq-len, or on shorter rows",
            )
        if seed < 0:
            raise UsageError(f"the seed must not be negative, got {seed}")
        if resume_from is not None and resume_from.kind != "training":
            raise UsageError("resume from a training checkpoint, not a sampler checkpoint")

    def _end(
        self,
        client: FireworksTraining,
        service: TrainingService | None,
        session: FireworksSession | None,
        *,
        abandon: bool,
    ) -> list[BaseException]:
        """End ``session``, then close ``service`` and ``client``; every failure.

        Failures to close the service or the client come first, since a service that failed
        to close may leave a billed session open; then those of closing the samplers.
        """
        closing: list[BaseException] = []
        ending: list[BaseException] = []
        try:
            if session is not None:
                self._live.pop(session.session_id, None)
                ending = session.end(abandon=abandon)
        finally:
            for close in (service.close if service is not None else None, client.close):
                if close is None:
                    continue
                try:
                    close()
                except Exception as exc:
                    closing.append(exc)
        return [*closing, *ending]


def _text(error: BaseException) -> str:
    detail = redact(str(error))
    return f"{type(error).__name__}{': ' + detail if detail else ''}"


def _warn(message: str) -> None:
    print(f"warning: {redact(message)}", file=sys.stderr)


if TYPE_CHECKING:
    # mypy checks that the implementations satisfy the protocols.
    def _conforms(
        backend: FireworksBackend, session: FireworksSession, sampler: FireworksSampler
    ) -> tuple[TrainingBackend, TrainingSession, Sampler]:
        return backend, session, sampler
