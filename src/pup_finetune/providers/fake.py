# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The offline ``fake`` provider: a deterministic reference implementation of the seam.

It implements every protocol in :mod:`pup_finetune.providers.base` and
:mod:`pup_finetune.rendering.base` with no network, SDK or credentials, so it serves three
purposes: the default test suite trains against it, the development CLI offers it as
``--provider fake`` (only with ``PUP_FINETUNE_DEV=1``), and it is the worked example a new
provider follows. ``tests/providers/contract`` holds the suites it passes.

- :class:`FakeRenderer` renders chat messages to a small text template, one segment at a
  time, and tokenizes each segment by splitting it into words and whitespace runs, each with
  an id derived from the word itself. Assistant completions carry weight 1.
- :class:`FakeBackend` and :class:`FakeSession` report losses from configurable per-epoch
  curves, enforce the 17-character checkpoint-name limit and the session-expiry rules of a
  real provider, record what was saved, promoted and sampled, and can simulate a crash or a
  failed promotion. Both optional capabilities (resume and promotion) are on by default and
  can be turned off, to exercise the paths of providers without them.
- :class:`FakeSampler` answers a fixed generic reply from the base model and the reference
  answer (looked up in an optional teacher table) from a tuned checkpoint.
- :class:`FakeProvider` ties them together under the name ``fake``.

Provider options (``TrainSpec.provider_options``, from ``--provider-option key=value``) are
listed on :class:`FakeOptions`; string values are decoded as JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NoReturn
from urllib.parse import parse_qs, quote, urlencode, urlsplit

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from pup_finetune.errors import FinetuneError, UsageError
from pup_finetune.providers.base import (
    MODEL_ID_PATTERN,
    AdamConfig,
    Check,
    CheckpointRef,
    Endpoint,
    Generation,
    LoraConfig,
    ModelProfile,
    ModelRef,
    StepMetrics,
)
from pup_finetune.rendering.base import (
    ChatMessage,
    ParsedResponse,
    RendererError,
    RendererErrorCode,
    ToolSpec,
)
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.runtime import Runtime
from pup_finetune.storage import canonical_json, loads_finite

if TYPE_CHECKING:
    from pup_finetune.providers.base import (
        PendingResult,
        Provider,
        Sampler,
        TrainingBackend,
        TrainingSession,
    )
    from pup_finetune.rendering.base import Renderer

__all__ = [
    "BASE_ANSWER",
    "FAKE_PROFILE",
    "MAX_CHECKPOINT_NAME_LEN",
    "PROVIDER_NAME",
    "TUNED_FALLBACK_ANSWER",
    "FakeBackend",
    "FakeCrash",
    "FakeOptions",
    "FakeProvider",
    "FakeRenderer",
    "FakeSampler",
    "FakeSession",
    "prompt_hash",
]

PROVIDER_NAME: Final = "fake"
MAX_CHECKPOINT_NAME_LEN: Final = 17
BASE_ANSWER: Final = "I can help with that."
"""What the base model (a sampler without a checkpoint) answers to every prompt."""
TUNED_FALLBACK_ANSWER: Final = "Here is the answer you need."
"""What a tuned checkpoint answers to a prompt missing from the teacher table."""

FAKE_PROFILE: Final = ModelProfile(
    provider=PROVIDER_NAME,
    alias="fake-model",
    base_model="fake/fake-model",
    tokenizer_model="fake/tokenizer",
    tokenizer_revision="0" * 40,
    renderer="fake",
    max_seq_len=4096,
    notes="Offline and deterministic; for development and tests only.",
)


def prompt_hash(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> str:
    """The key of a prompt in a teacher table: a hash of the messages (minus weights) and tools."""
    unweighted = [{k: v for k, v in message.items() if k != "weight"} for message in messages]
    payload = canonical_json({"messages": unweighted, "tools": list(tools)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- Rendering ---------------------------------------------------------------------------

_END: Final = "<|end|>"
_THINK: Final = "<|think|>"
_THINK_END: Final = "<|/think|>"
_CALL: Final = "<|call|>"
_ASSISTANT_HEADER: Final = "<|assistant|>\n"
_SPECIAL_TOKENS: Final = (
    "<|tools|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    _END,
    _THINK,
    _THINK_END,
    _CALL,
)
_SPECIAL_IDS: Final = {word: i for i, word in enumerate(_SPECIAL_TOKENS)}
_MARKER: Final = re.compile(r"<\|/?[a-z]+\|>")
_WORD: Final = re.compile(r"<\|/?[a-z]+\|>|\s+|[^\s<]+|<")
_ROLES: Final = frozenset({"system", "user", "assistant", "tool"})

# The template, in words; its hash goes into the renderer's identity.
_TEMPLATE: Final = (
    "fake-template v1: [<|tools|>\\n{tool json lines}<|end|>\\n] then per message "
    "<|role|>\\n{content}<|end|>\\n, a tool message "
    "<|tool|> {tool_call_id}\\n{content}<|end|>\\n; an assistant body is "
    "[<|think|>{reasoning}<|/think|>\\n]{content}[\\n<|call|>{call json}]*; "
    "a word's token id is 9 plus the big-endian 16-byte blake2b digest of its UTF-8 bytes"
)

# Every word ever tokenized in this process, by id, so any renderer can decode any other's ids.
_VOCAB_LOCK: Final = threading.Lock()
_VOCAB: Final[dict[int, str]] = {i: word for word, i in _SPECIAL_IDS.items()}


def _token_id(word: str) -> int:
    """The id of ``word``: fixed for the special tokens, otherwise derived from its bytes."""
    special = _SPECIAL_IDS.get(word)
    if special is not None:
        return special
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=16).digest()
    token = len(_SPECIAL_TOKENS) + int.from_bytes(digest, "big")
    with _VOCAB_LOCK:
        known = _VOCAB.setdefault(token, word)
    if known != word:
        raise RuntimeError(f"fake vocabulary collision: {known!r} and {word!r} share id {token}")
    return token


@dataclass(frozen=True, slots=True)
class _Segment:
    """Rendered text tokenized on its own, so no token spans two segments."""

    text: str
    trainable: bool = False


class FakeRenderer:
    """A deterministic renderer for the fake model. See the module docstring.

    The template, per message, is ``<|role|>\\n{content}<|end|>\\n``. Declared tools come first
    as one JSON line each under ``<|tools|>``; a tool message puts its ``tool_call_id`` on the
    header's line. An assistant body is optional ``<|think|>reasoning<|/think|>\\n``, then the
    content, then one ``<|call|>{"id", "name", "arguments"}`` JSON line per tool call. The
    trainable completion is the body plus ``<|end|>``, the only stop token.

    Rendering is segment-wise, so it is prefix-stable by construction. With
    ``prefix_stable=False`` the renderer declares otherwise and refuses
    :meth:`supervised_example`, which forces callers onto :meth:`per_turn_examples`.
    """

    def __init__(self, profile: ModelProfile = FAKE_PROFILE, *, prefix_stable: bool = True) -> None:
        self._profile = profile
        self._prefix_stable = prefix_stable

    @property
    def name(self) -> str:
        return "fake"

    @property
    def prefix_stable(self) -> bool:
        return self._prefix_stable

    @property
    def stop_token_ids(self) -> list[int]:
        return [_SPECIAL_IDS[_END]]

    # The tokenizer: words and whitespace runs, each id derived from the word, so every
    # renderer (in any process) gives the same text the same ids.

    def tokenize(self, text: str) -> list[int]:
        """``text`` as token ids; ``decode`` inverts it exactly."""
        return [_token_id(word) for word in _WORD.findall(text)]

    def decode(self, tokens: Sequence[int]) -> str:
        """The text of ``tokens``; every id must come from a ``tokenize`` in this process."""
        with _VOCAB_LOCK:
            words = [_VOCAB.get(token) for token in tokens]
        if any(word is None for word in words):
            raise ValueError("a token id is not in the fake vocabulary")
        return "".join(word for word in words if word is not None)

    # Training examples.

    def supervised_example(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> TokenizedExample:
        if not self._prefix_stable:
            raise RendererError(
                RendererErrorCode.NOT_PREFIX_STABLE,
                "this renderer is configured as not prefix-stable; use per_turn_examples",
            )
        trainable = self._trainable_turns(messages)
        return self._example(self._segments(messages, tools, trainable=set(trainable)))

    def per_turn_examples(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[TokenizedExample]:
        return [
            self._example(self._segments(messages[: turn + 1], tools, trainable={turn}))
            for turn in self._trainable_turns(messages)
        ]

    # Inference.

    def generation_prompt_tokens(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[int]:
        segments = [*self._segments(messages, tools, trainable=set()), _Segment(_ASSISTANT_HEADER)]
        return [token for segment in segments for token in self.tokenize(segment.text)]

    def render_completion(self, message: ChatMessage) -> str:
        """The completion text of an assistant ``message``, ``<|end|>`` included."""
        return self._assistant_body(message, where="the completion") + _END

    def parse_response(self, text: str) -> ParsedResponse:
        """``text`` as an assistant message, the inverse of :meth:`render_completion`.

        Text the renderer could not have produced is format-invalid, including a template
        marker anywhere but in its own place (the renderer refuses marker-bearing text).
        """
        body, end, rest = text.partition(_END)
        invalid = ParsedResponse({"role": "assistant", "content": text}, format_valid=False)
        if not end or rest.strip():
            return invalid
        message: ChatMessage = {"role": "assistant"}
        if body.startswith(_THINK):
            reasoning, closed, body = body[len(_THINK) :].partition(_THINK_END)
            if not closed or not body.startswith("\n"):
                return invalid
            message["reasoning_content"] = reasoning
            body = body[1:]
        content, *calls = body.split(_CALL)
        reasoning = message.get("reasoning_content", "")
        if any(_MARKER.search(part) for part in (reasoning, content, *calls)):
            return invalid
        tool_calls = []
        for i, call in enumerate(calls):
            parsed = _parse_call(call if i == len(calls) - 1 else call.removesuffix("\n"))
            if parsed is None:
                return invalid
            tool_calls.append(parsed)
        message["content"] = content.removesuffix("\n") if tool_calls else content
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ParsedResponse(message, format_valid=True)

    def identity(self) -> dict[str, str]:
        return {
            "renderer": self.name,
            "renderer_version": "1",
            "parser_version": "1",
            "tokenizer": self._profile.tokenizer_model,
            "tokenizer_revision": self._profile.tokenizer_revision,
            "chat_template_sha256": hashlib.sha256(_TEMPLATE.encode("utf-8")).hexdigest(),
            "prefix_stable": "true" if self._prefix_stable else "false",
        }

    # Internals.

    def _trainable_turns(self, messages: Sequence[ChatMessage]) -> list[int]:
        turns = [
            i
            for i, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("weight") != 0
        ]
        if not turns:
            raise RendererError(
                RendererErrorCode.NO_TRAINABLE_TOKENS, "no assistant turn has a non-zero weight"
            )
        return turns

    def _segments(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec], *, trainable: set[int]
    ) -> list[_Segment]:
        segments = []
        if tools:
            lines = "".join(f"{canonical_json(tool)}\n" for tool in tools)
            segments.append(_Segment(f"<|tools|>\n{_checked(lines, 'tools')}{_END}\n"))
        for i, message in enumerate(messages):
            role = message.get("role")
            where = f"message {i}"
            if role not in _ROLES:
                raise RendererError(
                    RendererErrorCode.UNSUPPORTED_MESSAGE, f"{where}: role {role!r}"
                )
            if role == "assistant":
                body = self._assistant_body(message, where=where)
                segments.append(_Segment(_ASSISTANT_HEADER))
                segments.append(_Segment(body + _END, trainable=i in trainable))
                segments.append(_Segment("\n"))
                continue
            header = f"<|{role}|>"
            if role == "tool":
                header += f" {_text(message, 'tool_call_id', where, required=True)}"
            segments.append(_Segment(f"{header}\n{_text(message, 'content', where)}{_END}\n"))
        return segments

    def _assistant_body(self, message: ChatMessage, *, where: str) -> str:
        parts = []
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            parts.append(f"{_THINK}{_text(message, 'reasoning_content', where)}{_THINK_END}\n")
        lines = [_text(message, "content", where)]
        for call in message.get("tool_calls") or []:
            try:
                function = call["function"]
                line = {
                    "id": call["id"],
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            except (KeyError, TypeError):
                raise RendererError(
                    RendererErrorCode.UNSUPPORTED_MESSAGE, f"{where}: malformed tool call"
                ) from None
            lines.append(_CALL + _checked(json.dumps(line, ensure_ascii=False), where))
        if len(lines) > 1 and not lines[0]:
            lines.pop(0)
        parts.append("\n".join(lines))
        return "".join(parts)

    def _example(self, segments: Sequence[_Segment]) -> TokenizedExample:
        tokens: list[int] = []
        weights: list[float] = []
        for segment in segments:
            ids = self.tokenize(segment.text)
            tokens.extend(ids)
            weights.extend([1.0 if segment.trainable else 0.0] * len(ids))
        return TokenizedExample.from_tokens(tokens, weights)


def _text(message: ChatMessage, key: str, where: str, *, required: bool = False) -> str:
    value = message.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise RendererError(
            RendererErrorCode.UNSUPPORTED_MESSAGE, f"{where}: {key} must be a string"
        )
    return _checked(value, where)


def _checked(text: str, where: str) -> str:
    if _MARKER.search(text):
        raise RendererError(
            RendererErrorCode.UNSUPPORTED_MESSAGE, f"{where}: text contains a template marker"
        )
    return text


def _parse_call(line: str) -> dict[str, Any] | None:
    """One ``<|call|>`` line as a chat tool call, or ``None`` when it is malformed.

    Malformed includes JSON the data schema rejects: ``NaN``, ``Infinity``, numbers beyond
    float range, integers too long to parse and nesting too deep to parse.
    """
    try:
        call = loads_finite(line)
        arguments = loads_finite(call["arguments"])
    except (ValueError, RecursionError, KeyError, TypeError):
        return None
    if (
        not isinstance(call, dict)
        or set(call) != {"id", "name", "arguments"}
        or not all(isinstance(call[key], str) and call[key] for key in ("id", "name"))
        or not isinstance(arguments, dict)
    ):
        return None
    return {
        "id": call["id"],
        "type": "function",
        "function": {"name": call["name"], "arguments": call["arguments"]},
    }


# --- Options -----------------------------------------------------------------------------

_Loss = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class FakeOptions(BaseModel):
    """The fake provider's options, all optional.

    ``loss_curve`` and ``val_curve`` are the mean training and validation loss of each epoch
    (the last value repeats past the end); ``step_delay_s`` delays every pending result;
    ``fail_at_step`` makes the ``forward_backward`` of that global step (1-based) fail with
    :class:`FakeCrash` once per lineage: only a fresh session crashes, so a session resumed
    from a checkpoint (with the same options) trains on past that step; ``prefix_stable``
    configures the renderer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    loss_curve: list[_Loss] = Field(
        default_factory=lambda: [2.0, 1.4, 1.1, 0.95, 0.9], min_length=1
    )
    val_curve: list[_Loss] = Field(
        default_factory=lambda: [2.1, 1.6, 1.35, 1.25, 1.2], min_length=1
    )
    step_delay_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    fail_at_step: PositiveInt | None = None
    prefix_stable: bool = True

    @classmethod
    def parse(cls, options: Mapping[str, Any]) -> FakeOptions:
        """Validate ``options``, decoding string values as JSON, or raise :class:`UsageError`."""
        unknown = sorted(set(options) - set(cls.model_fields))
        if unknown:
            raise UsageError(
                f"unknown fake provider option(s): {', '.join(unknown)}",
                hint=f"known options: {', '.join(cls.model_fields)}",
            )
        decoded = {key: _decode(key, value) for key, value in options.items()}
        try:
            return cls.model_validate(decoded)
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
            raise UsageError(f"invalid fake provider options: {'; '.join(problems)}") from exc


def _decode(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError is a ValueError
        raise UsageError(
            f"fake provider option {key!r} must be JSON, got {value!r}",
            hint="for example loss_curve=[2.0,1.5] or prefix_stable=false",
        ) from exc


# --- Training ----------------------------------------------------------------------------


class FakeCrash(FinetuneError):
    """The simulated crash ``fail_at_step`` asks for."""


class _Pending[T]:
    """A result computed at call time and released ``delay_s`` seconds later."""

    def __init__(self, outcome: Callable[[], T], delay_s: float) -> None:
        self._outcome = outcome
        self._ready_at = time.monotonic() + delay_s

    def result(self, timeout: float | None = None) -> T:
        remaining = self._ready_at - time.monotonic()
        if remaining > 0:
            if timeout is not None and timeout < remaining:
                time.sleep(max(timeout, 0.0))
                raise TimeoutError("the fake result is not ready yet")
            time.sleep(remaining)
        return self._outcome()


def _raises(error: BaseException) -> Callable[[], NoReturn]:
    def outcome() -> NoReturn:
        raise error

    return outcome


@dataclass(slots=True)
class _State:
    """What a fake checkpoint captures, so a resumed session stays on the loss curves.

    ``epoch`` indexes the curves for training (``0`` in the first epoch) and ``trained`` is
    the epoch the weights reflect. ``ended`` says an epoch-end call (a validation ``forward``
    or a sampler checkpoint) came since the last ``forward_backward``; ``steps_per_epoch`` is
    learned from the first sampler checkpoint, ``None`` until then.
    """

    step: int = 0
    epoch: int = 0
    trained: int | None = None
    ended: bool = False
    steps_per_epoch: int | None = None

    def to_query(self) -> str:
        return urlencode(
            {
                "step": self.step,
                "epoch": self.epoch,
                "trained": -1 if self.trained is None else self.trained,
                "v": int(self.ended),
                "spe": self.steps_per_epoch or 0,
            }
        )

    @classmethod
    def from_ref(cls, ref: CheckpointRef) -> _State:
        parts = urlsplit(ref.uri)
        query = parse_qs(parts.query)
        try:
            if parts.scheme != PROVIDER_NAME:
                raise ValueError(parts.scheme)
            trained = int(query["trained"][0])
            steps_per_epoch = int(query.get("spe", ["0"])[0])
            return cls(
                step=int(query["step"][0]),
                epoch=int(query["epoch"][0]),
                trained=None if trained < 0 else trained,
                ended=query["v"][0] == "1",
                steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
            )
        except (KeyError, IndexError, ValueError):
            raise UsageError(f"{ref.uri!r} is not a fake provider checkpoint") from None


def _at(curve: Sequence[float], epoch: int | None) -> float:
    return curve[min(epoch or 0, len(curve) - 1)]


class FakeBackend:
    """Opens :class:`FakeSession`\\ s and records what they do, for assertions.

    Checkpoint URIs carry the training state, so a backend in a new process can resume from
    one. A sampler checkpoint is promotable and samplable only while the session that saved it
    is open in this backend, as on a provider whose sessions expire when the process exits.
    ``options`` is read when a session opens.

    ``supports_resume`` and ``supports_promotion`` turn those capabilities off: resuming or
    promoting then fails. ``fail_promotions`` makes that many next ``promote`` calls fail with
    ``promotion_error`` (by default a :class:`FakeCrash`), and ``fail_session_ends`` that many
    next session exits (after the session closed) with a :class:`FakeCrash`, as when a
    provider's call to release the session fails. Both attributes count down, and tests may
    set them later.
    """

    def __init__(
        self,
        options: FakeOptions | None = None,
        *,
        teacher: Mapping[str, ChatMessage] | None = None,
        supports_resume: bool = True,
        supports_promotion: bool = True,
        fail_promotions: int = 0,
        fail_session_ends: int = 0,
    ) -> None:
        self.options = options or FakeOptions()
        self.teacher: dict[str, ChatMessage] = dict(teacher or {})
        self.fail_promotions = fail_promotions
        self.promotion_error: BaseException | None = None
        self.fail_session_ends = fail_session_ends
        self._supports_resume = supports_resume
        self._supports_promotion = supports_promotion
        self.opened: list[str] = []
        self.session_settings: list[dict[str, Any]] = []
        self.closed: list[str] = []
        self.saved: list[CheckpointRef] = []
        self.promoted: list[ModelRef] = []
        self.sampled: list[CheckpointRef | None] = []
        self._live: set[str] = set()
        self._prefix = uuid.uuid4().hex[:8]

    @property
    def max_checkpoint_name_len(self) -> int:
        return MAX_CHECKPOINT_NAME_LEN

    @property
    def supports_resume(self) -> bool:
        return self._supports_resume

    @property
    def supports_promotion(self) -> bool:
        return self._supports_promotion

    @contextmanager
    def open_session(
        self,
        profile: ModelProfile,
        lora: LoraConfig,
        *,
        resume_from: CheckpointRef | None = None,
        seed: int,
        max_seq_len: int,
    ) -> Iterator[FakeSession]:
        if profile.provider != PROVIDER_NAME:
            raise UsageError(f"the fake provider cannot train {profile.provider}/{profile.alias}")
        if not 0 < max_seq_len <= profile.max_seq_len:
            raise UsageError(
                f"max_seq_len must be 1 to {profile.max_seq_len} for {profile.alias}, "
                f"got {max_seq_len}"
            )
        if seed < 0:
            raise UsageError(f"the seed must not be negative, got {seed}")
        state = _State()
        if resume_from is not None:
            if not self._supports_resume:
                raise UsageError("this fake backend is configured without resume")
            if resume_from.kind != "training":
                raise UsageError("resume from a training checkpoint, not a sampler checkpoint")
            state = _State.from_ref(resume_from)
            state.epoch = max(state.epoch, resume_from.epoch - 1)
        session_id = f"fake-{self._prefix}-{len(self.opened) + 1}"
        session = FakeSession(
            self, profile, lora, session_id, state, resumed=resume_from is not None
        )
        self.opened.append(session.session_id)
        self.session_settings.append(
            {"seed": seed, "max_seq_len": max_seq_len, "resume_from": resume_from}
        )
        self._live.add(session.session_id)
        try:
            yield session
        finally:
            session.close()
            self._live.discard(session.session_id)
            self.closed.append(session.session_id)
            if self.fail_session_ends > 0:
                self.fail_session_ends -= 1
                raise FakeCrash(f"simulated failure ending session {session.session_id}")

    def is_live(self, session_id: str) -> bool:
        """Whether the session ``session_id`` is open in this backend."""
        return session_id in self._live


class FakeSession:
    """One fake training session; built by :meth:`FakeBackend.open_session`.

    The fake places itself on the loss curves from the call sequence, as the shared loop
    drives it: the first ``forward_backward`` after an epoch-end call (a validation pass,
    ``forward``, or a sampler checkpoint, which the loop saves only at an epoch's end) that
    followed an optimizer step starts the next epoch. The first sampler checkpoint also
    teaches it the steps per epoch (its ``step`` over its ``epoch``), so from then on the
    epoch follows the step count, also through epochs with neither call (no validation rows
    and no new best). A checkpoint's ``epoch`` (the loop's, from 1) is a lower bound on the
    epoch it is in. That bookkeeping is carried in checkpoints, so a resumed session continues
    on the curves; only a session resumed (without validation) from the first epoch's last
    training checkpoint, saved before the fake learned the steps per epoch, may run an epoch
    behind until the next epoch-end call. It is not model state: ``forward`` never changes
    the weights, and repeating it returns the same metrics.
    """

    def __init__(
        self,
        backend: FakeBackend,
        profile: ModelProfile,
        lora: LoraConfig,
        session_id: str,
        state: _State,
        *,
        resumed: bool = False,
    ) -> None:
        self._backend = backend
        self._options = backend.options
        self._profile = profile
        self.lora = lora
        self._session_id = session_id
        self._state = state
        self._resumed = resumed
        self._closed = False
        self._crashed = False
        self._renderer: FakeRenderer | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def step(self) -> int:
        """Optimizer steps applied so far, including those before a resume."""
        return self._state.step

    def forward_backward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        self._require_open()
        state = self._state
        if state.ended and state.trained == state.epoch:
            state.epoch += 1
        state.ended = False
        if state.steps_per_epoch is not None:
            state.epoch = max(state.epoch, state.step // state.steps_per_epoch)
        step = state.step + 1
        if step == self._options.fail_at_step and not self._resumed:
            self._crashed = True
            crash = FakeCrash(f"simulated crash at step {step} (fail_at_step)")
            return _Pending(_raises(crash), self._options.step_delay_s)
        return self._metrics(batch, _at(self._options.loss_curve, state.epoch))

    def forward(self, batch: Sequence[TokenizedExample]) -> PendingResult[StepMetrics]:
        self._require_open()
        self._state.ended = True
        return self._metrics(batch, _at(self._options.val_curve, self._state.trained))

    def optim_step(self, adam: AdamConfig) -> PendingResult[None]:
        self._require_open()
        self._state.step += 1
        self._state.trained = self._state.epoch
        return _Pending(lambda: None, self._options.step_delay_s)

    def save_training_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        self._require_open()
        self._state.epoch = max(self._state.epoch, epoch - 1)
        return self._save("training", name, epoch=epoch, step=step)

    def save_sampler_checkpoint(self, name: str, *, epoch: int, step: int) -> CheckpointRef:
        self._require_open()
        state = self._state
        state.ended = True
        if state.steps_per_epoch is None and epoch > 0 and step > 0 and step % epoch == 0:
            state.steps_per_epoch = step // epoch
        return self._save("sampler", name, epoch=epoch, step=step)

    def load_weights(self, ckpt: CheckpointRef) -> None:
        self._require_open()
        if ckpt.kind == "sampler":
            state = self._require_live_sampler(ckpt, "load weights from")
        else:
            state = _State.from_ref(ckpt)
        self._state.trained = state.trained

    def promote(self, ckpt: CheckpointRef, model_id: str) -> ModelRef:
        self._require_open()
        if not self._backend.supports_promotion:
            raise UsageError("this fake backend is configured without promotion")
        self._require_live_sampler(ckpt, "promote")
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise UsageError(
                f"invalid model id {model_id!r}",
                hint="use 1 to 63 lowercase letters, digits and dashes, starting with a letter "
                "and not ending with a dash",
            )
        if self._backend.fail_promotions > 0:
            self._backend.fail_promotions -= 1
            error = self._backend.promotion_error
            raise error or FakeCrash(f"simulated failure promoting {ckpt.name!r} (fail_promotions)")
        model = ModelRef(provider=PROVIDER_NAME, model_id=model_id, uri=f"fake://models/{model_id}")
        self._backend.promoted.append(model)
        return model

    def sampler(self, ckpt: CheckpointRef | None) -> FakeSampler:
        self._require_open()
        tuned = False
        if ckpt is not None:
            tuned = self._require_live_sampler(ckpt, "sample from").trained is not None
        self._backend.sampled.append(ckpt)
        if self._renderer is None:
            self._renderer = FakeRenderer(self._profile)
        return FakeSampler(self, self._renderer, tuned=tuned, teacher=self._backend.teacher)

    @property
    def closed(self) -> bool:
        """Whether the session has ended."""
        return self._closed

    def close(self) -> None:
        """End the session; :meth:`FakeBackend.open_session` calls it on exit."""
        self._closed = True

    def _metrics(self, batch: Sequence[TokenizedExample], mean: float) -> _Pending[StepMetrics]:
        weight = sum(sum(example.weights) for example in batch)
        tokens = sum(example.num_tokens for example in batch)
        metrics = StepMetrics(loss_sum=mean * weight, weight=weight, tokens=tokens)
        return _Pending(lambda: metrics, self._options.step_delay_s)

    def _save(
        self, kind: Literal["training", "sampler"], name: str, *, epoch: int, step: int
    ) -> CheckpointRef:
        self._require_open()
        if not name or len(name) > MAX_CHECKPOINT_NAME_LEN:
            raise ValueError(
                f"checkpoint names must be 1 to {MAX_CHECKPOINT_NAME_LEN} characters, "
                f"got {len(name)}: {name!r}"
            )
        path = f"{self._session_id}/{kind}/{quote(name, safe='')}"
        uri = f"fake://checkpoints/{path}?{self._state.to_query()}"
        ref = CheckpointRef(
            kind=kind,
            name=name,
            uri=uri,
            epoch=epoch,
            step=step,
            created_in_session=self.session_id,
        )
        self._backend.saved.append(ref)
        return ref

    def _require_open(self) -> None:
        if self._closed:
            raise FinetuneError(f"fake session {self._session_id} is closed")
        if self._crashed:
            raise FakeCrash(f"fake session {self._session_id} crashed (fail_at_step)")

    def _require_live_sampler(self, ckpt: CheckpointRef, action: str) -> _State:
        """The state of sampler checkpoint ``ckpt``, if its session is still open."""
        if ckpt.kind != "sampler":
            raise UsageError(f"can only {action} a sampler checkpoint, got a {ckpt.kind} one")
        state = _State.from_ref(ckpt)
        if not self._backend.is_live(ckpt.created_in_session):
            raise FinetuneError(
                f"cannot {action} sampler checkpoint {ckpt.name!r}: its session "
                f"{ckpt.created_in_session} has ended",
                hint="resume from a training checkpoint and save a new sampler checkpoint",
            )
        return state


class FakeSampler:
    """Greedy generation from the fake model; built by :meth:`FakeSession.sampler`.

    The base model answers :data:`BASE_ANSWER`. A tuned checkpoint answers the reference: the
    teacher table's message for :func:`prompt_hash` of the prompt, else
    :data:`TUNED_FALLBACK_ANSWER`. The answer goes through the renderer like a real
    completion, so ``max_tokens`` truncation yields ``format_valid=False``.
    """

    def __init__(
        self,
        session: FakeSession,
        renderer: FakeRenderer,
        *,
        tuned: bool,
        teacher: Mapping[str, ChatMessage],
    ) -> None:
        self.session = session
        self.renderer = renderer
        self.tuned = tuned
        self.teacher = teacher
        self._closed = False

    def generate(
        self, messages: list[ChatMessage], tools: list[ToolSpec], *, max_tokens: int
    ) -> Generation:
        if self._closed or self.session.closed:
            raise FinetuneError("this fake sampler is closed")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        prompt = self.renderer.generation_prompt_tokens(messages, tools)
        answer: ChatMessage = {"role": "assistant", "content": BASE_ANSWER}
        if self.tuned:
            fallback = {"role": "assistant", "content": TUNED_FALLBACK_ANSWER}
            answer = self.teacher.get(prompt_hash(messages, tools), fallback)
        completion = self.renderer.tokenize(self.renderer.render_completion(answer))
        stop_reason = "stop"
        if len(completion) > max_tokens:
            completion, stop_reason = completion[:max_tokens], "length"
        parsed = self.renderer.parse_response(self.renderer.decode(completion))
        return Generation(
            message=parsed.message,
            format_valid=parsed.format_valid,
            prompt_tokens=len(prompt),
            completion_tokens=len(completion),
            latency_s=0.0,
            stop_reason=stop_reason,
        )

    def close(self) -> None:
        self._closed = True


# --- The provider ------------------------------------------------------------------------


class FakeProvider:
    """The ``fake`` provider: one model profile, no credentials, an in-memory deployments table.

    :meth:`validate_options` also makes the validated options this provider's configuration,
    since nothing else carries ``provider_options`` to the renderer and backend: later
    :meth:`renderer` calls and newly opened sessions use them. ``supports_resume`` and
    ``supports_promotion`` configure the backend (see :class:`FakeBackend`).
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        teacher: Mapping[str, ChatMessage] | None = None,
        supports_resume: bool = True,
        supports_promotion: bool = True,
    ) -> None:
        self.runtime = runtime
        self.options = FakeOptions()
        self._backend = FakeBackend(
            self.options,
            teacher=teacher,
            supports_resume=supports_resume,
            supports_promotion=supports_promotion,
        )
        self.deployments: dict[str, Endpoint] = {}
        self.deploy_calls: list[tuple[ModelRef, dict[str, Any]]] = []
        self.undeployed: list[Endpoint] = []

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def check(self) -> list[Check]:
        return [Check("fake provider", True, "offline; needs no credentials or network")]

    def models(self) -> list[ModelProfile]:
        return [FAKE_PROFILE]

    def resolve_model(self, alias_or_id: str) -> ModelProfile:
        for profile in self.models():
            if alias_or_id in (profile.alias, profile.base_model):
                return profile
        known = ", ".join(profile.alias for profile in self.models())
        raise UsageError(f"unknown fake model {alias_or_id!r}", hint=f"known models: {known}")

    def preflight(self, profile: ModelProfile) -> None:
        """Nothing to check offline; only refuses another provider's profile."""
        self._own(profile)

    def renderer(self, profile: ModelProfile) -> FakeRenderer:
        self._own(profile)
        return FakeRenderer(profile, prefix_stable=self.options.prefix_stable)

    def backend(self) -> FakeBackend:
        return self._backend

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        parsed = FakeOptions.parse(options)
        self.options = self._backend.options = parsed
        return parsed.model_dump(exclude_unset=True)

    def deploy(self, model: ModelRef, options: Mapping[str, Any]) -> Endpoint:
        if model.provider != PROVIDER_NAME:
            raise UsageError(f"the fake provider cannot deploy a {model.provider} model")
        self.deploy_calls.append((model, dict(options)))
        endpoint = Endpoint(
            provider=PROVIDER_NAME,
            deployment_id=f"fake-deployment-{len(self.deploy_calls)}",
            model_id=model.model_id,
            inference_model=model.model_id,
        )
        self.deployments[endpoint.deployment_id] = endpoint
        return endpoint

    def undeploy(self, endpoint: Endpoint) -> None:
        if self.deployments.pop(endpoint.deployment_id, None) is None:
            raise FinetuneError(f"no fake deployment {endpoint.deployment_id!r}")
        self.undeployed.append(endpoint)

    def _own(self, profile: ModelProfile) -> None:
        if profile.provider != PROVIDER_NAME:
            raise UsageError(f"the fake provider does not serve {profile.provider}/{profile.alias}")


if TYPE_CHECKING:
    # mypy checks that every fake satisfies its protocol.
    def _conforms(
        provider: FakeProvider,
        renderer: FakeRenderer,
        backend: FakeBackend,
        session: FakeSession,
        sampler: FakeSampler,
    ) -> tuple[Provider, Renderer, TrainingBackend, TrainingSession, Sampler]:
        return provider, renderer, backend, session, sampler
