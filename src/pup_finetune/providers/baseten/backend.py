# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Baseten training backend: LoRA fine-tuning sessions on Baseten Loops.

- :class:`BasetenBackend` opens one :class:`BasetenSession` per ``open_session``: a Loops
  session and a dedicated LoRA trainer (a Loops run) whose context fits the session's longest
  example, rounded up to a power of two of at least :data:`MIN_SEQ_LEN` tokens. A trainer
  Baseten has no capacity for is a :class:`ProviderCapacityError`, with nothing left running.
  Leaving the context always closes the trainer client and deactivates the run and every
  sampler the session started, however the session ended; a deactivation that fails raises
  an error naming the run or deployment to deactivate by hand. Resuming loads a training
  checkpoint with its optimizer state into the new run.
- :class:`BasetenSession` sends each batch to the trainer as Loops datums and hands back the
  SDK futures, so steps pipeline. Only ``forward`` and sampling, which change nothing when
  repeated, fail as :class:`TransientError`. Checkpoints are ``bt://loops:`` paths that
  outlive the run. There is no promotion: the best sampler checkpoint is what gets deployed.
- :class:`BasetenSampler` serves the base model or a sampler checkpoint from a dedicated Loops
  sampler (its own GPU) with the profile's context, created over REST so that its deployment
  is known, and can be deactivated, before it is ready. It renders the prompt with the
  profile's renderer, samples greedily up to the renderer's stop tokens and parses the
  completion back, given the tools; one cut off at ``max_tokens`` is never ``format_valid``.
  Only sampling is retried (a bounded number of times, with backoff), and ``generate`` may be
  called from several threads at once, as the SDK's sampling client allows.

Everything that reaches Baseten goes through :mod:`.client`; tests pass the fakes in
``tests/fakes/baseten.py`` as ``loops=`` and ``transport=``.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from pydantic import SecretStr

from pup_finetune import output
from pup_finetune.errors import ExternalError, FinetuneError, StateError, UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    CheckpointRef,
    Generation,
    LoraConfig,
    ModelProfile,
    ModelRef,
    StepMetrics,
    TransientError,
    is_transient,
)
from pup_finetune.providers.baseten import PROVIDER_NAME
from pup_finetune.providers.baseten.client import (
    API_BASE,
    MAX_CHECKPOINT_NAME_LEN,
    SOURCE,
    BasetenHTTP,
    DeploymentRejected,
    LoopsFuture,
    LoopsModule,
    LoopsSamplingClient,
    LoopsTrainingClient,
    loops_failure,
    loops_sampling_client,
    loops_sdk,
    loops_service,
    loops_trainer,
    to_loops_datum,
)
from pup_finetune.rendering.base import ChatMessage, Renderer, ToolSpec
from pup_finetune.rendering.tokens import TokenizedExample

if TYPE_CHECKING:
    from pup_finetune.providers.base import Sampler, TrainingBackend, TrainingSession

__all__ = [
    "MIN_SEQ_LEN",
    "SAMPLE_ATTEMPTS",
    "BasetenBackend",
    "BasetenSampler",
    "BasetenSession",
    "fitted_seq_len",
]

MIN_SEQ_LEN: Final = 1024
"""The smallest context a trainer is created with."""
SAMPLER_READY_TIMEOUT: Final = 1800.0
"""How long a new sampler may take to become ready, in seconds, shared by a sample's attempts."""
SAMPLE_ATTEMPTS: Final = 3
"""How many times a sample is tried when it fails transiently (a timeout, a dropped
connection, a server error)."""
SAMPLE_BACKOFF_S: Final = 2.0
"""The wait before the second attempt; it doubles for each attempt after that."""
SAMPLE_TIMEOUT_S: Final = 600.0
"""How long one sample may take before it counts as timed out (and is retried)."""

_DEACTIVATE_ATTEMPTS: Final = 3
_POLL_ATTEMPTS: Final = 24
_POLL_SECONDS: Final = 5.0
_INACTIVE: Final = (None, "INACTIVE")
_CHECKPOINT_NAME: Final = re.compile(rf"[A-Za-z0-9_.-]{{1,{MAX_CHECKPOINT_NAME_LEN}}}")
_LOAD_PROMPT: Final[list[ChatMessage]] = [{"role": "user", "content": "Hello"}]
_LENGTH: Final = "length"

Sleep = Callable[[float], None]


def fitted_seq_len(max_seq_len: int, limit: int) -> int:
    """The trainer context for examples of up to ``max_seq_len`` tokens, at most ``limit``.

    It is the next power of two, at least :data:`MIN_SEQ_LEN`, so similar runs share sizes.
    """
    return min(max(MIN_SEQ_LEN, 1 << (max_seq_len - 1).bit_length()), limit)


class BasetenBackend:
    """Opens LoRA training sessions on Baseten Loops, each on its own dedicated trainer.

    ``api_key`` gives the API key when a session opens, not before. ``renderer`` gives a
    profile's renderer, which samplers render prompts and parse completions with.
    ``max_seq_len`` (the provider option) caps the longest example a session trains on, below
    the profile's context. ``loops`` and ``transport`` replace the SDK and the HTTP transport
    (tests pass the fakes), and ``sleep`` the wait between deactivation polls and sampling
    retries.
    """

    def __init__(
        self,
        api_key: Callable[[], SecretStr],
        *,
        renderer: Callable[[ModelProfile], Renderer],
        max_seq_len: int | None = None,
        loops: LoopsModule | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._renderer_factory = renderer
        self._max_seq_len = max_seq_len
        self._loops = loops
        self._transport = transport
        self._sleep = sleep
        self._renderers: dict[str, Renderer] = {}

    @property
    def max_checkpoint_name_len(self) -> int:
        return MAX_CHECKPOINT_NAME_LEN

    @property
    def supports_resume(self) -> bool:
        return True

    @property
    def supports_promotion(self) -> bool:
        return False

    @contextmanager
    def open_session(
        self,
        profile: ModelProfile,
        lora: LoraConfig,
        *,
        resume_from: CheckpointRef | None = None,
        seed: int,
        max_seq_len: int,
    ) -> Iterator[BasetenSession]:
        _check_session(profile, lora, resume_from, seed=seed, max_seq_len=max_seq_len)
        if self._max_seq_len is not None and max_seq_len > self._max_seq_len:
            raise UsageError(
                f"a session with max_seq_len {max_seq_len} was requested, over the max_seq_len "
                f"provider option of {self._max_seq_len}",
                hint="train a new run with a larger --max-seq-len, or on shorter rows",
            )
        api_key = self._api_key()
        sdk = loops_sdk(self._loops)
        context = fitted_seq_len(max_seq_len, profile.max_seq_len)
        service = loops_service(api_key, loops=sdk)
        output.info(
            f"provisioning a dedicated Baseten Loops trainer for {profile.base_model} "
            f"({context} tokens) in Loops session {service.session_id}; this can take minutes"
        )
        with BasetenHTTP(api_key, transport=self._transport) as http:
            trainer = loops_trainer(
                service,
                base_model=profile.base_model,
                rank=lora.rank,
                seed=seed,
                max_seq_len=context,
                name=f"pup-finetune-{uuid.uuid4().hex[:12]}",
            )
            session = BasetenSession(
                sdk=sdk,
                http=http,
                trainer=trainer,
                loops_session_id=service.session_id,
                profile=profile,
                max_seq_len=context,
                api_key=api_key,
                renderer=self.renderer_for,
                sleep=self._sleep,
            )
            try:
                if resume_from is not None:
                    session.load(resume_from, with_optimizer=True)
                yield session
            finally:
                session.close()

    def renderer_for(self, profile: ModelProfile) -> Renderer:
        """``profile``'s renderer, built once per backend."""
        renderer = self._renderers.get(profile.alias)
        if renderer is None:
            renderer = self._renderers[profile.alias] = self._renderer_factory(profile)
        return renderer


def _check_session(
    profile: ModelProfile,
    lora: LoraConfig,
    resume_from: CheckpointRef | None,
    *,
    seed: int,
    max_seq_len: int,
) -> None:
    if profile.provider != PROVIDER_NAME:
        raise UsageError(f"the Baseten backend cannot train {profile.provider}/{profile.alias}")
    if not 0 < max_seq_len <= profile.max_seq_len:
        raise UsageError(
            f"max_seq_len must be 1 to {profile.max_seq_len} for {profile.alias}, got {max_seq_len}"
        )
    if seed < 0:
        raise UsageError(f"the seed must not be negative, got {seed}")
    if lora.alpha is not None:
        raise UsageError("Baseten Loops sets the LoRA alpha itself", hint="drop --lora-alpha")
    if resume_from is not None:
        if resume_from.kind != "training":
            raise UsageError("resume from a training checkpoint, not a sampler checkpoint")
        _require_ours(resume_from)


class _Pending[T]:
    """An SDK future whose result is converted, and whose failure mapped, when resolved."""

    def __init__(
        self,
        future: LoopsFuture,
        convert: Callable[[Any], T],
        fail: Callable[[Exception], ExternalError],
    ) -> None:
        self._future = future
        self._convert = convert
        self._fail = fail

    def result(self, timeout: float | None = None) -> T:
        try:
            value = self._future.result(timeout)
        except FinetuneError:
            raise
        except Exception as exc:
            raise self._fail(exc) from exc
        return self._convert(value)


def _step_metrics(result: Any, batch: Sequence[TokenizedExample]) -> StepMetrics:
    """The batch's loss from the per-token logprobs, else from the mean loss Loops reports.

    The logprobs give the exact weighted sum; the mean is only approximate for a batch the
    SDK splits into several requests.
    """
    weight = sum(sum(example.weights) for example in batch)
    tokens = sum(example.num_tokens for example in batch)
    try:
        outputs = list(result.loss_fn_outputs or [])
        if outputs and all("logprobs" in out for out in outputs):
            loss_sum = 0.0
            for example, out in zip(batch, outputs, strict=True):
                pairs = zip(out["logprobs"].tolist(), example.weights, strict=True)
                # Loops leaves the unweighted positions null.
                loss_sum += sum(-float(logprob) * w for logprob, w in pairs if w > 0)
        else:
            loss_sum = float(result.metrics.get("loss", result.loss)) * weight
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExternalError("Loops returned a malformed step result", source=SOURCE) from exc
    return StepMetrics(loss_sum=loss_sum, weight=weight, tokens=tokens)


def _status_name(record: Mapping[str, Any] | None) -> str | None:
    """A run's or deployment's status (``"ACTIVE"``...), ``None`` when Baseten has none."""
    if record is None:
        return None
    status = record.get("status")
    return str(status.get("name") if isinstance(status, dict) else status).upper()


def _deactivate(
    what: str,
    url: str,
    status: Callable[[], str | None],
    deactivate: Callable[[], None],
    sleep: Sleep,
) -> None:
    """Deactivate ``what`` (at ``url``) unless it is inactive, then wait until it is.

    On failure, raises an :class:`ExternalError` with the manual step in its message, as
    callers may show only the message.
    """
    try:
        current: str | None = "UNKNOWN"
        with suppress(ExternalError):  # an unknown state still gets the deactivation
            current = status()
        if current in _INACTIVE:
            return
        attempts = 0 if current == "DEACTIVATING" else _DEACTIVATE_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                deactivate()
                break
            except ExternalError:
                if attempt == _DEACTIVATE_ATTEMPTS:
                    raise
                sleep(float(attempt))
        last_error: ExternalError | None = None
        for _ in range(_POLL_ATTEMPTS):
            sleep(_POLL_SECONDS)
            try:
                if status() in _INACTIVE:
                    return
            except ExternalError as exc:
                last_error = exc
        raise last_error or ExternalError(f"{what} is still active", source=SOURCE)
    except Exception as exc:
        reason = exc.message if isinstance(exc, FinetuneError) else type(exc).__name__
        raise ExternalError(
            f"could not deactivate {what} ({reason}); it may still bill: deactivate it in the "
            f"Baseten UI or POST {url}",
            source=SOURCE,
            hint=f'curl -X POST {url} -H "Authorization: Bearer $BASETEN_API_KEY"',
        ) from exc


class BasetenSession:
    """One Loops run training a LoRA adapter; built by :meth:`BasetenBackend.open_session`.

    ``session_id`` is the run's id: checkpoints are saved under it and it is what gets
    deactivated. ``max_seq_len`` is the trainer's context. Samplers are created in the Loops
    session ``loops_session_id``, with the profile's context, since evaluation prompts and
    their new tokens can be longer than any training example.
    """

    def __init__(
        self,
        *,
        sdk: LoopsModule,
        http: BasetenHTTP,
        trainer: LoopsTrainingClient,
        loops_session_id: str,
        profile: ModelProfile,
        max_seq_len: int,
        api_key: SecretStr,
        renderer: Callable[[ModelProfile], Renderer],
        sleep: Sleep,
    ) -> None:
        self._sdk = sdk
        self._http = http
        self._trainer = trainer
        self._run_id = str(trainer.run_id)
        self.loops_session_id = loops_session_id
        self._profile = profile
        self.max_seq_len = max_seq_len
        self._api_key = api_key
        self._renderer_for = renderer
        self._sleep = sleep
        self._samplers: list[BasetenSampler] = []
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._run_id

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> _Pending[StepMetrics]:
        examples = self._checked_batch(batch)
        datums = [to_loops_datum(example, self._sdk) for example in examples]
        return self._submit(
            "the Loops forward_backward",
            lambda: self._trainer.forward_backward(datums, "cross_entropy"),
            lambda out: _step_metrics(out, examples),
        )

    def forward(self, batch: Sequence[TokenizedExample]) -> _Pending[StepMetrics]:
        examples = self._checked_batch(batch)
        datums = [to_loops_datum(example, self._sdk) for example in examples]
        return self._submit(
            "the Loops forward",
            lambda: self._trainer.forward(datums, "cross_entropy"),
            lambda out: _step_metrics(out, examples),
            repeatable=True,
        )

    def optim_step(self, adam: AdamConfig) -> _Pending[None]:
        self._require_open()
        params = self._sdk.AdamParams(
            learning_rate=adam.learning_rate,
            beta1=adam.beta1,
            beta2=adam.beta2,
            eps=adam.eps,
            weight_decay=adam.weight_decay,
            grad_clip_norm=adam.grad_clip_norm,
        )
        return self._submit(
            "the Loops optim_step", lambda: self._trainer.optim_step(params), lambda _: None
        )

    def save_training_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        return self._save("training", name, epoch=epoch, step=step)

    def save_sampler_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        return self._save("sampler", name, epoch=epoch, step=step)

    def load_weights(self, ckpt: CheckpointRef) -> None:
        if ckpt.kind != "training":
            raise UsageError("Loops loads weights from a training checkpoint, not a sampler one")
        self.load(ckpt, with_optimizer=False)

    def load(self, ckpt: CheckpointRef, *, with_optimizer: bool) -> None:
        """Load training checkpoint ``ckpt``, with its optimizer state when resuming."""
        self._require_open()
        _require_ours(ckpt)
        trainer = self._trainer
        load = trainer.load_state_with_optimizer if with_optimizer else trainer.load_state
        try:
            self._wait(f"loading checkpoint {ckpt.name!r}", lambda: load(ckpt.uri))
        except ExternalError as exc:
            exc.hint = exc.hint or (
                "the checkpoint may have expired; resume from a later training checkpoint, "
                "or start a new run"
            )
            raise

    def promote(
        self, ckpt: CheckpointRef, model_id: str, *, adopt_since: datetime | None = None
    ) -> ModelRef:
        raise UsageError("Baseten has no promotion; the sampler checkpoint is deployed")

    def sampler(self, ckpt: CheckpointRef | None) -> BasetenSampler:
        """A dedicated Loops sampler for ``ckpt`` or the base model. Paid until closed."""
        self._require_open()
        if ckpt is not None:
            if ckpt.kind != "sampler":
                raise UsageError(
                    f"can only sample from a sampler checkpoint, not a {ckpt.kind} one"
                )
            _require_ours(ckpt)
        renderer = self._renderer_for(self._profile)
        try:
            return self._start_sampler(renderer, None if ckpt is None else ckpt.uri)
        except ExternalError as exc:
            if ckpt is not None and (exc.hint is None or isinstance(exc, DeploymentRejected)):
                expired = (
                    f"checkpoint {ckpt.name!r} may have expired; resume from a training "
                    "checkpoint and save a new sampler checkpoint"
                )
                exc.hint = f"{exc.hint}; {expired}" if exc.hint else expired
            raise

    def close(self) -> None:
        """End the session: stop its samplers, close the trainer client, deactivate the run.

        :meth:`BasetenBackend.open_session` calls it on exit. Every step is attempted, also
        after one is interrupted (Ctrl-C cuts only that step's wait); the failures are raised
        together, each with its manual step, else the interrupt is raised again.
        """
        if self._closed:
            return
        self._closed = True
        failures: list[ExternalError] = []
        interrupt: BaseException | None = None
        for step in (*(sampler.close for sampler in self._samplers), self._end_run):
            try:
                step()
            except ExternalError as exc:
                failures.append(exc)
            except BaseException as exc:
                interrupt = interrupt or exc
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExternalError(
                "; ".join(failure.message for failure in failures),
                source=SOURCE,
                hint="; ".join(failure.hint for failure in failures if failure.hint),
            )
        if interrupt is not None:
            raise interrupt

    def _end_run(self) -> None:
        try:
            self._trainer.close()
        except Exception as exc:
            output.warn(f"could not close the Loops trainer client ({type(exc).__name__})")
        finally:  # also when Ctrl-C cut the close
            output.info(f"deactivating Baseten run {self._run_id}")
            _deactivate(
                f"Baseten run {self._run_id}",
                f"{API_BASE}/v1/loops/runs/{self._run_id}/deactivate",
                lambda: _status_name(self._http.get_run(self._run_id)),
                lambda: self._http.deactivate_run(self._run_id),
                self._sleep,
            )

    def _start_sampler(self, renderer: Renderer, checkpoint_path: str | None) -> BasetenSampler:
        context = self._profile.max_seq_len
        try:
            record = self._http.create_sampler(
                session_id=self.loops_session_id,
                max_seq_len=context,
                model_path=checkpoint_path,
                base_model=self._profile.base_model if checkpoint_path is None else None,
            )
        except ExternalError as exc:  # without a hint, a server error: the sampler may exist
            exc.hint = exc.hint or (
                f"a sampler may remain in Loops session {self.loops_session_id}; deactivate it "
                "in the Baseten UI"
            )
            raise
        sampler = BasetenSampler(
            record,
            loops_session_id=self.loops_session_id,
            sdk=self._sdk,
            http=self._http,
            renderer=renderer,
            max_seq_len=context,
            sleep=self._sleep,
        )
        self._samplers.append(sampler)
        try:
            sampler.connect(
                self._api_key, base_model=self._profile.base_model, checkpoint_path=checkpoint_path
            )
        except BaseException:
            with suppress(ExternalError):  # close() reports it again when the session ends
                sampler.close()
            raise
        return sampler

    def _submit[T](
        self,
        label: str,
        call: Callable[[], LoopsFuture],
        convert: Callable[[Any], T],
        *,
        repeatable: bool = False,
    ) -> _Pending[T]:
        def fail(exc: Exception) -> ExternalError:
            return loops_failure(exc, label, self._sdk, repeatable=repeatable)

        try:
            future = call()
        except Exception as exc:
            raise fail(exc) from exc
        return _Pending(future, convert, fail)

    def _wait(self, label: str, call: Callable[[], LoopsFuture]) -> Any:
        return self._submit(label, call, lambda value: value).result()

    def _save(
        self, kind: Literal["training", "sampler"], name: str, *, epoch: int, step: int
    ) -> CheckpointRef:
        self._require_open()
        if not _CHECKPOINT_NAME.fullmatch(name):
            raise ValueError(
                f"checkpoint names must be 1 to {MAX_CHECKPOINT_NAME_LEN} letters, digits, "
                f"dots, dashes or underscores, got {name!r}"
            )
        trainer = self._trainer
        save = trainer.save_state if kind == "training" else trainer.save_weights_for_sampler
        path = getattr(self._wait(f"saving checkpoint {name!r}", lambda: save(name)), "path", None)
        run = f"bt://loops:{self._run_id}/"
        if not isinstance(path, str) or not (
            path.startswith(run) if kind == "training" else path == f"{run}sampler_weights/{name}"
        ):
            raise ExternalError(
                f"Loops returned an unexpected path for checkpoint {name!r}", source=SOURCE
            )
        return CheckpointRef(
            kind=kind, name=name, uri=path, epoch=epoch, step=step, created_in_session=self._run_id
        )

    def _checked_batch(self, batch: Sequence[TokenizedExample]) -> list[TokenizedExample]:
        self._require_open()
        examples = list(batch)
        if not examples:
            raise ValueError("a batch needs at least one example")
        longest = max(example.num_tokens for example in examples)
        if longest > self.max_seq_len:
            raise UsageError(
                f"an example has {longest} tokens, over this session's "
                f"{self.max_seq_len}-token trainer context"
            )
        return examples

    def _require_open(self) -> None:
        if self._closed:
            raise StateError(f"the Baseten training session {self._run_id} has ended")


def _require_ours(ckpt: CheckpointRef) -> None:
    if not ckpt.uri.startswith("bt://loops:"):
        raise UsageError(f"{ckpt.uri!r} is not a Baseten Loops checkpoint")


class BasetenSampler:
    """Greedy generation on a dedicated Loops sampler; built by :meth:`BasetenSession.sampler`.

    The sampler is a Baseten deployment with its own GPU: :meth:`close` deactivates it
    (closing again is a no-op), and its session deactivates it when it ends, if still up.
    :meth:`generate` is safe to call from several threads, as the SDK's sampling client is.
    """

    def __init__(
        self,
        record: Mapping[str, Any],
        *,
        loops_session_id: str,
        sdk: LoopsModule,
        http: BasetenHTTP,
        renderer: Renderer,
        max_seq_len: int,
        sleep: Sleep,
    ) -> None:
        model_id, deployment_id = record.get("model_id"), record.get("deployment_id")
        if not (isinstance(model_id, str) and isinstance(deployment_id, str)):
            raise TransientError(
                "Baseten described the new sampler without its deployment",
                source=SOURCE,
                hint=f"deactivate the sampler of Loops session {loops_session_id} in the "
                "Baseten UI",
            )
        self.model_id = model_id
        self.deployment_id = deployment_id
        self._record = dict(record)
        self._loops_session_id = loops_session_id
        self._sdk = sdk
        self._http = http
        self._renderer = renderer
        self._max_seq_len = max_seq_len
        self._sleep = sleep
        self._client: LoopsSamplingClient | None = None
        self._closed = False

    def connect(self, api_key: SecretStr, *, base_model: str, checkpoint_path: str | None) -> None:
        """Attach the SDK client; for a checkpoint, load it with one sample (which waits)."""
        self._client = loops_sampling_client(
            api_key,
            self._record,
            base_model=base_model,
            session_id=self._loops_session_id,
            checkpoint_path=checkpoint_path,
            ready_timeout=SAMPLER_READY_TIMEOUT / SAMPLE_ATTEMPTS,
            timeout=SAMPLE_TIMEOUT_S,
            loops=self._sdk,
        )
        if checkpoint_path is not None:
            prompt = self._renderer.generation_prompt_tokens(_LOAD_PROMPT, [])
            self._sample(prompt, 1, "loading the checkpoint on the Baseten sampler")

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        if self._closed:
            raise StateError("this Baseten sampler is closed")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        prompt = self._renderer.generation_prompt_tokens(messages, tools)
        if len(prompt) + max_tokens > self._max_seq_len:
            raise UsageError(
                f"a {len(prompt)}-token prompt and {max_tokens} new tokens exceed the "
                f"sampler's {self._max_seq_len}-token context"
            )
        sequence, latency = self._sample(prompt, max_tokens, "sampling on the Baseten sampler")
        tokens = list(sequence.tokens)
        parsed = self._renderer.parse_response(self._renderer.decode(tokens), tools)
        return Generation(
            message=parsed.message,
            # A completion cut off at max_tokens is not a whole turn, whatever it parses as.
            format_valid=parsed.format_valid and sequence.stop_reason != _LENGTH,
            prompt_tokens=len(prompt),
            completion_tokens=len(tokens),
            latency_s=latency,
            stop_reason=sequence.stop_reason,
        )

    def close(self) -> None:
        if self._closed:
            return
        client, self._client = self._client, None
        try:
            if client is not None:
                client.close()
        except Exception as exc:
            output.warn(f"could not close the Loops sampling client ({type(exc).__name__})")
        finally:  # also when Ctrl-C cut the close
            model_id, deployment_id = self.model_id, self.deployment_id
            _deactivate(
                f"the Baseten sampler deployment {model_id}/{deployment_id}",
                f"{API_BASE}/v1/models/{model_id}/deployments/{deployment_id}/deactivate",
                lambda: _status_name(self._http.get_deployment(model_id, deployment_id)),
                lambda: self._http.deactivate_deployment(model_id, deployment_id),
                self._sleep,
            )
        self._closed = True

    def _sample(self, prompt: list[int], max_tokens: int, label: str) -> tuple[Any, float]:
        """The one sequence sampled greedily after ``prompt``, and the seconds it took.

        A transient failure is retried, :data:`SAMPLE_ATTEMPTS` times in all.
        """
        client = self._client  # read once: close() may clear it from another thread
        if self._closed or client is None:
            raise StateError("this Baseten sampler is closed")
        params = self._sdk.SamplingParams(
            max_tokens=max_tokens, temperature=0.0, stop=self._renderer.stop_token_ids
        )
        delay = SAMPLE_BACKOFF_S
        for attempt in range(1, SAMPLE_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                response = client.sample(
                    prompt=self._sdk.ModelInput.from_ints(prompt),
                    num_samples=1,
                    sampling_params=params,
                )
                return response.sequences[0], max(time.monotonic() - started, 0.0)
            except Exception as exc:
                error = loops_failure(exc, label, self._sdk, repeatable=True)
                if not is_transient(error):
                    raise error from exc
                if attempt == SAMPLE_ATTEMPTS:
                    error.add_note(f"sampling failed after {SAMPLE_ATTEMPTS} attempts")
                    raise error from exc
                output.warn(
                    f"{error.message}; retrying in {delay:g}s "
                    f"(attempt {attempt + 1} of {SAMPLE_ATTEMPTS})"
                )
            self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover


if TYPE_CHECKING:
    # mypy checks that each class satisfies its protocol.
    def _conforms(
        backend: BasetenBackend, session: BasetenSession, sampler: BasetenSampler
    ) -> tuple[TrainingBackend, TrainingSession, Sampler]:
        return backend, session, sampler
