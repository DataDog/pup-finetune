# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Contract suites every provider passes: what "done" means for a new provider.

The classes here are not collected on their own. A provider's tests subclass each one under
a ``Test*`` name and override the fixtures it declares::

    class TestMyRenderer(RendererContract):
        @pytest.fixture
        def renderer_factory(self) -> Callable[[], Renderer]:
            return lambda: MyRenderer(...)

- :class:`RendererContract` needs ``renderer_factory``, which builds a fresh renderer each
  call; ``renderer`` (one built by it) and ``conversation`` have defaults.
- :class:`BackendContract` needs ``backend``, ``profile`` and ``renderer``; ``lora``,
  ``batch``, ``prompt`` and ``model_id`` have defaults.
- :class:`ProviderContract` needs ``provider``; ``model_ref`` and ``deploy_options`` have
  defaults.

The default suite runs them against fakes; a provider's live tests (marked ``live``) can run
them against the real service by overriding the fixtures.
"""

from __future__ import annotations

import dataclasses
import json
import math
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from pup_finetune.errors import FinetuneError, UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    CheckpointRef,
    DeploymentRequest,
    DeployOptions,
    Endpoint,
    Generation,
    LoraConfig,
    ModelProfile,
    ModelRef,
    Provider,
    SamplingUsage,
    StepMetrics,
    TrainingBackend,
    TrainingSession,
    TrainingUsage,
)
from pup_finetune.rendering.base import (
    ChatMessage,
    Renderer,
    RendererError,
    RendererErrorCode,
    ToolSpec,
)
from pup_finetune.rendering.tokens import TokenizedExample

__all__ = ["BackendContract", "Conversation", "ProviderContract", "RendererContract"]

Conversation = tuple[list[ChatMessage], list[ToolSpec]]
"""Messages and tools in OpenAI chat format, as ``to_provider_dict`` produces them."""

REQUIRED_IDENTITY_KEYS = frozenset(
    {
        "renderer",
        "renderer_version",
        "parser_version",
        "tokenizer",
        "tokenizer_revision",
        "chat_template_sha256",
    }
)

_WEATHER_TOOL: ToolSpec = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# Synthetic conversations covering plain answers, tool calls, multiple and untrained turns.
_CONVERSATIONS: list[Conversation] = [
    (
        [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello there!"},
        ],
        [],
    ),
    (
        [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "Weather in Northvale?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 14}'},
            {"role": "assistant", "content": "It is 14 degrees C in Northvale."},
        ],
        [_WEATHER_TOOL],
    ),
    (
        [
            {"role": "user", "content": "Count to two, one turn at a time."},
            {"role": "assistant", "content": "One.", "weight": 0},
            {"role": "user", "content": "Next."},
            {"role": "assistant", "content": "Two."},
        ],
        [],
    ),
]

_LOOKUP_TOOL: ToolSpec = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Find a record.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "flag": {"type": "string"},
                "label": {"type": "string"},
                "limit": {"type": "integer"},
                "exact": {"type": "boolean"},
            },
        },
    },
}

# String arguments that read as JSON: a template may write strings unquoted, so a parse must
# take their type from the declared tool.
_TYPED_ARGUMENTS = {"code": "123", "flag": "true", "label": "null", "limit": 5, "exact": False}


def _assistant_turns(messages: list[ChatMessage]) -> list[int]:
    return [i for i, message in enumerate(messages) if message["role"] == "assistant"]


def _trainable(message: ChatMessage) -> bool:
    return message.get("weight") != 0


def _unshifted(example: TokenizedExample) -> tuple[list[int], list[float]]:
    """The render's tokens and each token's own weight (the first token has none)."""
    tokens = [*example.input_tokens, example.target_tokens[-1]]
    return tokens, [0.0, *example.weights]


def _comparable(message: ChatMessage) -> dict[str, Any]:
    """An assistant message reduced to what a parse must reproduce.

    Tool-call ids are dropped (many templates do not render them) and arguments are compared
    as decoded objects, since a template may re-serialize them.
    """
    calls = [
        {"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
        for call in message.get("tool_calls") or []
    ]
    return {
        "role": message["role"],
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or None,
        "tool_calls": calls,
    }


def _completion_runs(
    renderer: Renderer, messages: list[ChatMessage], tools: list[ToolSpec], tokens: list[int]
) -> dict[int, tuple[int, int]]:
    """Each assistant turn's span ``[start, end)``: from its prompt's end to the next one's."""
    turns = _assistant_turns(messages)
    starts = [len(renderer.generation_prompt_tokens(messages[:turn], tools)) for turn in turns]
    ends = [*starts[1:], len(tokens)]
    return {turn: (start, end) for turn, start, end in zip(turns, starts, ends, strict=True)}


def _with(preamble: list[ChatMessage], messages: list[ChatMessage]) -> list[ChatMessage]:
    """``messages`` after ``preamble``, unless they already open with its role."""
    if preamble and messages[0].get("role") == preamble[0].get("role"):
        return [dict(message) for message in messages]
    return [*(dict(message) for message in preamble), *(dict(message) for message in messages)]


def _conversations(preamble: list[ChatMessage]) -> list[Conversation]:
    return [(_with(preamble, messages), list(tools)) for messages, tools in _CONVERSATIONS]


def _full_render(
    renderer: Renderer, messages: list[ChatMessage], tools: list[ToolSpec]
) -> TokenizedExample | None:
    """The row as one example, or ``None`` when the renderer renders it per turn."""
    if not renderer.prefix_stable:
        return None
    try:
        return renderer.supervised_example(messages, tools)
    except RendererError as exc:
        if exc.code is not RendererErrorCode.NOT_PREFIX_STABLE:
            raise
        return None


class RendererContract:
    """What every :class:`~pup_finetune.rendering.base.Renderer` guarantees."""

    @pytest.fixture
    def renderer_factory(self) -> Callable[[], Renderer]:
        raise NotImplementedError(
            "a RendererContract subclass must provide a `renderer_factory` fixture"
        )

    @pytest.fixture
    def renderer(self, renderer_factory: Callable[[], Renderer]) -> Renderer:
        return renderer_factory()

    @pytest.fixture
    def preamble(self) -> list[ChatMessage]:
        """Messages every conversation of the contract starts with; a renderer whose template
        needs a system message gives one."""
        return []

    @pytest.fixture(params=range(len(_CONVERSATIONS)), ids=["plain", "tool_calls", "untrained"])
    def conversation(
        self, request: pytest.FixtureRequest, preamble: list[ChatMessage]
    ) -> Conversation:
        messages, tools = _conversations(preamble)[request.param]
        return messages, tools

    def _examples(
        self, renderer: Renderer, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[tuple[list[ChatMessage], TokenizedExample, bool]]:
        """Each example with the messages it renders and whether it is a full render.

        As training builds them: one full render when the renderer is prefix-stable and does
        not refuse the row with ``not_prefix_stable``, else one per trainable turn.
        """
        full = _full_render(renderer, messages, tools)
        if full is not None:
            return [(messages, full, True)]
        turns = [turn for turn in _assistant_turns(messages) if _trainable(messages[turn])]
        examples = renderer.per_turn_examples(messages, tools)
        assert len(examples) == len(turns)
        return [
            (messages[: turn + 1], example, False)
            for turn, example in zip(turns, examples, strict=True)
        ]

    def test_names_itself(self, renderer: Renderer) -> None:
        assert renderer.name
        assert renderer.stop_token_ids

    def test_identity_pins_the_render(self, renderer: Renderer) -> None:
        identity = renderer.identity()
        assert set(identity) >= REQUIRED_IDENTITY_KEYS
        assert all(isinstance(value, str) and value for value in identity.values())
        assert renderer.identity() == identity

    def test_equal_identities_render_the_same_tokens(
        self, renderer_factory: Callable[[], Renderer], preamble: list[ChatMessage]
    ) -> None:
        first, second = renderer_factory(), renderer_factory()
        assert first.identity() == second.identity()
        # Rendering unrelated text first must not shift what either renders afterwards.
        warm_up: list[ChatMessage] = _with(
            preamble, [{"role": "user", "content": "Zebras and quokkas, 1729."}]
        )
        first.generation_prompt_tokens(warm_up, [])
        for messages, tools in _conversations(preamble):
            examples = first.per_turn_examples(messages, tools)
            assert examples == second.per_turn_examples(messages, tools)
            full = _full_render(first, messages, tools)
            assert full == _full_render(second, messages, tools)
            prompt = first.generation_prompt_tokens(messages, tools)
            assert prompt == second.generation_prompt_tokens(messages, tools)
            assert second.decode(prompt) == first.decode(prompt)
            for example in examples:
                tokens, _ = _unshifted(example)
                assert second.decode(tokens) == first.decode(tokens)

    def test_prefix_stable_is_truthful(
        self, renderer: Renderer, conversation: Conversation
    ) -> None:
        messages, tools = conversation
        for rendered, example, full in self._examples(renderer, messages, tools):
            tokens, _ = _unshifted(example)
            turns = (
                [turn for turn in _assistant_turns(rendered) if _trainable(rendered[turn])]
                if full
                else [len(rendered) - 1]
            )
            for turn in turns:
                prompt = renderer.generation_prompt_tokens(rendered[:turn], tools)
                assert tokens[: len(prompt)] == prompt, f"turn {turn} is not a token prefix"

    def test_only_assistant_completions_are_trained(
        self, renderer: Renderer, conversation: Conversation
    ) -> None:
        messages, tools = conversation
        for rendered, example, full in self._examples(renderer, messages, tools):
            tokens, weights = _unshifted(example)
            assert all(weight in (0.0, 1.0) for weight in weights)
            runs = _completion_runs(renderer, rendered, tools, tokens)
            first_start = min(start for start, _ in runs.values())
            assert not any(weights[:first_start]), "a token before the first completion is trained"
            trained_turns = (
                [turn for turn in runs if _trainable(rendered[turn])]
                if full
                else [len(rendered) - 1]
            )
            for turn, (start, end) in runs.items():
                span = weights[start:end]
                if turn not in trained_turns:
                    assert not any(span), f"turn {turn} must not be trained"
                    continue
                length = span.index(0.0) if 0.0 in span else len(span)
                assert length > 0, f"turn {turn} has no trainable tokens"
                assert not any(span[length:]), f"turn {turn} trains tokens after its completion"
                completion = tokens[start : start + length]
                assert completion[-1] in renderer.stop_token_ids
                parsed = renderer.parse_response(renderer.decode(completion), tools)
                assert parsed.format_valid
                assert _comparable(parsed.message) == _comparable(rendered[turn])

    def test_weight_zero_turns_are_never_trained(
        self, renderer: Renderer, preamble: list[ChatMessage]
    ) -> None:
        messages = _with(
            preamble,
            [
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello.", "weight": 0},
            ],
        )
        with pytest.raises(RendererError) as excinfo:
            renderer.per_turn_examples(messages, [])
        assert excinfo.value.code == "no_trainable_tokens"
        if renderer.prefix_stable:
            with pytest.raises(RendererError):
                renderer.supervised_example(messages, [])

    def test_parse_response_round_trips_rendered_messages(
        self, renderer: Renderer, preamble: list[ChatMessage]
    ) -> None:
        for messages, tools in _conversations(preamble):
            for turn in _assistant_turns(messages):
                trained = [*messages[:turn], {**messages[turn], "weight": 1}]
                example = renderer.per_turn_examples(trained, tools)[-1]
                tokens, weights = _unshifted(example)
                completion = [t for t, w in zip(tokens, weights, strict=True) if w > 0]
                parsed = renderer.parse_response(renderer.decode(completion), tools)
                assert parsed.format_valid
                assert _comparable(parsed.message) == _comparable(messages[turn])

    def test_parse_response_keeps_declared_argument_types(
        self, renderer: Renderer, preamble: list[ChatMessage]
    ) -> None:
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps(_TYPED_ARGUMENTS)},
        }
        messages = _with(
            preamble,
            [
                {"role": "user", "content": "Look up record 123."},
                {"role": "assistant", "content": "", "tool_calls": [call]},
            ],
        )
        example = renderer.per_turn_examples(messages, [_LOOKUP_TOOL])[-1]
        tokens, weights = _unshifted(example)
        completion = [t for t, w in zip(tokens, weights, strict=True) if w > 0]
        parsed = renderer.parse_response(renderer.decode(completion), [_LOOKUP_TOOL])
        assert parsed.format_valid
        assert _comparable(parsed.message) == _comparable(messages[-1])

    @pytest.mark.parametrize("text", ["", "no end marker", '{"unclosed": ', "<|", "\n\n"])
    @pytest.mark.parametrize("tools", [[], [_LOOKUP_TOOL], [{"type": "function"}, "not a tool"]])
    def test_parse_response_never_raises(
        self, renderer: Renderer, text: str, tools: list[ToolSpec]
    ) -> None:
        parsed = renderer.parse_response(text, tools)
        assert parsed.message["role"] == "assistant"
        assert isinstance(parsed.format_valid, bool)


def _session_settings(profile: ModelProfile) -> dict[str, Any]:
    """The ``seed`` and ``max_seq_len`` the suites open sessions with."""
    return {"seed": 42, "max_seq_len": profile.max_seq_len}


def _assert_expired(error: FinetuneError) -> None:
    """``error`` reports an expired checkpoint: not a usage mistake, and with a way out."""
    assert not isinstance(error, UsageError), error
    assert error.hint, error


class BackendContract:
    """What every :class:`~pup_finetune.providers.base.TrainingBackend` guarantees."""

    @pytest.fixture
    def backend(self) -> TrainingBackend:
        raise NotImplementedError("a BackendContract subclass must provide a `backend` fixture")

    @pytest.fixture
    def profile(self) -> ModelProfile:
        raise NotImplementedError("a BackendContract subclass must provide a `profile` fixture")

    @pytest.fixture
    def renderer(self) -> Renderer:
        raise NotImplementedError("a BackendContract subclass must provide a `renderer` fixture")

    @pytest.fixture
    def lora(self, profile: ModelProfile) -> LoraConfig:
        return LoraConfig(rank=profile.default_lora_rank)

    @pytest.fixture
    def batch(self, renderer: Renderer) -> list[TokenizedExample]:
        return [
            example
            for messages, tools in _CONVERSATIONS
            for example in renderer.per_turn_examples(messages, tools)
        ]

    @pytest.fixture
    def prompt(self) -> Conversation:
        return [{"role": "user", "content": "Say hello."}], []

    @pytest.fixture
    def model_id(self) -> str:
        return f"pft-contract-{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def session(
        self, backend: TrainingBackend, profile: ModelProfile, lora: LoraConfig
    ) -> Iterator[TrainingSession]:
        with backend.open_session(profile, lora, **_session_settings(profile)) as session:
            yield session

    def _train_one_step(
        self, session: TrainingSession, batch: list[TokenizedExample]
    ) -> StepMetrics:
        metrics = session.forward_backward(batch).result(timeout=60)
        assert session.optim_step(AdamConfig(learning_rate=1e-4)).result(timeout=60) is None
        return metrics

    def _assert_generation(self, generation: Generation) -> None:
        assert isinstance(generation, Generation)
        message = generation.message
        assert message["role"] == "assistant"
        assert isinstance(message.get("content", ""), str)
        assert isinstance(message.get("tool_calls", []), list)
        assert generation.prompt_tokens > 0
        assert generation.completion_tokens >= 0
        assert generation.latency_s >= 0

    def test_checkpoint_name_limit_is_positive(self, backend: TrainingBackend) -> None:
        assert backend.max_checkpoint_name_len > 0

    def test_capabilities_are_declared(self, backend: TrainingBackend) -> None:
        assert isinstance(backend.supports_resume, bool)
        assert isinstance(backend.supports_promotion, bool)

    def test_a_session_can_be_sized_below_the_profile_limit(
        self,
        backend: TrainingBackend,
        profile: ModelProfile,
        lora: LoraConfig,
        batch: list[TokenizedExample],
    ) -> None:
        longest = max(example.num_tokens for example in batch)
        with backend.open_session(profile, lora, seed=7, max_seq_len=longest) as session:
            assert math.isfinite(self._train_one_step(session, batch).mean_loss)

    def test_open_and_close_a_session(
        self, backend: TrainingBackend, profile: ModelProfile, lora: LoraConfig
    ) -> None:
        with backend.open_session(profile, lora, **_session_settings(profile)) as first:
            first_id = first.session_id
            assert isinstance(first_id, str) and first_id
        with backend.open_session(profile, lora, **_session_settings(profile)) as second:
            assert second.session_id != first_id

    def test_a_training_step_resolves_with_metrics(
        self, session: TrainingSession, batch: list[TokenizedExample]
    ) -> None:
        metrics = self._train_one_step(session, batch)
        assert isinstance(metrics, StepMetrics)
        assert metrics.weight == pytest.approx(sum(sum(example.weights) for example in batch))
        assert metrics.tokens > 0
        assert metrics.loss_sum >= 0
        assert math.isfinite(metrics.mean_loss)

    def test_pipelined_steps_resolve_in_order(
        self, session: TrainingSession, batch: list[TokenizedExample]
    ) -> None:
        adam = AdamConfig(learning_rate=1e-4)
        pending = [
            session.forward_backward(batch),
            session.optim_step(adam),
            session.forward_backward(batch),
            session.optim_step(adam),
        ]
        first, first_step, second, second_step = (call.result(timeout=60) for call in pending)
        weight = sum(sum(example.weights) for example in batch)
        for metrics in (first, second):
            assert isinstance(metrics, StepMetrics)
            assert metrics.weight == pytest.approx(weight)
            assert math.isfinite(metrics.mean_loss)
        assert first_step is None
        assert second_step is None

    def test_forward_has_no_side_effects(
        self, session: TrainingSession, batch: list[TokenizedExample]
    ) -> None:
        self._train_one_step(session, batch)
        first = session.forward(batch).result(timeout=60)
        second = session.forward(batch).result(timeout=60)
        assert isinstance(first, StepMetrics)
        assert second.loss_sum == pytest.approx(first.loss_sum, rel=1e-4)
        assert second.weight == first.weight
        assert math.isfinite(first.mean_loss)

    def test_checkpoint_names_over_the_limit_are_rejected(
        self, backend: TrainingBackend, session: TrainingSession
    ) -> None:
        too_long = "c" * (backend.max_checkpoint_name_len + 1)
        with pytest.raises(ValueError, match="checkpoint"):
            session.save_training_checkpoint(too_long, epoch=0, step=0)
        with pytest.raises(ValueError, match="checkpoint"):
            session.save_sampler_checkpoint(too_long, epoch=0, step=0)
        longest = "c" * backend.max_checkpoint_name_len
        assert session.save_training_checkpoint(longest, epoch=0, step=0).name == longest

    def test_resume_from_a_training_checkpoint_continues(
        self,
        backend: TrainingBackend,
        profile: ModelProfile,
        lora: LoraConfig,
        batch: list[TokenizedExample],
    ) -> None:
        with backend.open_session(profile, lora, **_session_settings(profile)) as session:
            self._train_one_step(session, batch)
            ckpt = session.save_training_checkpoint("e0s1", epoch=0, step=1)
            first_id = session.session_id
        assert isinstance(ckpt, CheckpointRef)
        assert (ckpt.kind, ckpt.epoch, ckpt.step) == ("training", 0, 1)
        assert ckpt.created_in_session == first_id
        settings = _session_settings(profile)
        if not backend.supports_resume:
            # Without resume, asking for it fails up front instead of training from scratch.
            with (
                pytest.raises(FinetuneError),
                backend.open_session(profile, lora, resume_from=ckpt, **settings),
            ):
                pass
            return
        with backend.open_session(profile, lora, resume_from=ckpt, **settings) as resumed:
            assert resumed.session_id != first_id
            metrics = self._train_one_step(resumed, batch)
            assert math.isfinite(metrics.mean_loss)

    def test_a_training_checkpoint_recreates_a_sampler_checkpoint(
        self,
        backend: TrainingBackend,
        profile: ModelProfile,
        lora: LoraConfig,
        batch: list[TokenizedExample],
        model_id: str,
    ) -> None:
        # How a run promotes a best epoch whose sampler checkpoint expired with its session:
        # load the epoch's training checkpoint in a new session and save its weights again.
        settings = _session_settings(profile)
        with backend.open_session(profile, lora, **settings) as session:
            self._train_one_step(session, batch)
            training = session.save_training_checkpoint("e1s1", epoch=1, step=1)
        resume_from = training if backend.supports_resume else None
        with backend.open_session(profile, lora, resume_from=resume_from, **settings) as later:
            later.load_weights(training)
            sampler = later.save_sampler_checkpoint("r1s1", epoch=1, step=1)
            assert (sampler.kind, sampler.epoch, sampler.step) == ("sampler", 1, 1)
            assert sampler.created_in_session == later.session_id
            if backend.supports_promotion:
                assert later.promote(sampler, model_id).model_id == model_id

    def test_a_sampler_checkpoint_is_promotable_while_open(
        self,
        backend: TrainingBackend,
        session: TrainingSession,
        batch: list[TokenizedExample],
        model_id: str,
    ) -> None:
        self._train_one_step(session, batch)
        ckpt = session.save_sampler_checkpoint("best", epoch=0, step=1)
        assert ckpt.kind == "sampler"
        assert ckpt.created_in_session == session.session_id
        if not backend.supports_promotion:
            # Without promotion, the sampler checkpoint itself is what gets deployed.
            with pytest.raises(FinetuneError):
                session.promote(ckpt, model_id)
            return
        model = session.promote(ckpt, model_id)
        assert isinstance(model, ModelRef)
        assert model.model_id == model_id
        assert model.uri

    def test_other_weights_are_not_promoted_under_a_promoted_model_id(
        self,
        backend: TrainingBackend,
        session: TrainingSession,
        batch: list[TokenizedExample],
        model_id: str,
    ) -> None:
        # A run promotes a new best under an id of its own; one that reused the id of an
        # earlier best must fail rather than report the earlier model as the new best's.
        if not backend.supports_promotion:
            pytest.skip("the backend does not promote")
        self._train_one_step(session, batch)
        first = session.save_sampler_checkpoint("first", epoch=1, step=1)
        self._train_one_step(session, batch)
        second = session.save_sampler_checkpoint("second", epoch=2, step=2)
        assert session.promote(first, model_id).model_id == model_id
        with pytest.raises(FinetuneError):
            session.promote(second, model_id)

    def test_a_sampler_checkpoint_from_a_closed_session_works_or_fails_cleanly(
        self,
        backend: TrainingBackend,
        profile: ModelProfile,
        lora: LoraConfig,
        batch: list[TokenizedExample],
        model_id: str,
    ) -> None:
        # Sampler checkpoints outlive their session on some providers only (see CheckpointRef).
        # Where they do, promoting and sampling work as usual; where they expire, each fails
        # as an expiry, not a usage mistake, and the error says how to recover.
        with backend.open_session(profile, lora, **_session_settings(profile)) as session:
            self._train_one_step(session, batch)
            ckpt = session.save_sampler_checkpoint("after-close", epoch=0, step=1)
        with backend.open_session(profile, lora, **_session_settings(profile)) as later:
            try:
                model = later.promote(ckpt, model_id) if backend.supports_promotion else None
            except FinetuneError as exc:
                _assert_expired(exc)
            else:
                if model is not None:
                    assert isinstance(model, ModelRef)
                    assert model.model_id == model_id
                    assert model.uri
            try:
                sampler = later.sampler(ckpt)
            except FinetuneError as exc:
                _assert_expired(exc)
            else:
                sampler.close()

    def test_the_base_sampler_works(self, session: TrainingSession, prompt: Conversation) -> None:
        messages, tools = prompt
        sampler = session.sampler(None)
        try:
            self._assert_generation(sampler.generate(messages, tools, max_tokens=64))
        finally:
            sampler.close()
        sampler.close()

    def test_a_checkpoint_sampler_returns_an_openai_message(
        self, session: TrainingSession, batch: list[TokenizedExample], prompt: Conversation
    ) -> None:
        self._train_one_step(session, batch)
        ckpt = session.save_sampler_checkpoint("sample", epoch=0, step=1)
        messages, tools = prompt
        sampler = session.sampler(ckpt)
        try:
            self._assert_generation(sampler.generate(messages, tools, max_tokens=64))
        finally:
            sampler.close()


class ProviderContract:
    """What every :class:`~pup_finetune.providers.base.Provider` guarantees."""

    @pytest.fixture
    def provider(self) -> Provider:
        raise NotImplementedError("a ProviderContract subclass must provide a `provider` fixture")

    @pytest.fixture
    def model_ref(self, provider: Provider) -> ModelRef:
        return ModelRef(
            provider=provider.name,
            model_id="pft-contract-model",
            uri=f"{provider.name}://models/pft-contract-model",
        )

    @pytest.fixture
    def deploy_options(self) -> DeployOptions:
        return DeployOptions(deployment_id="pft-contract-deployment")

    def test_is_a_provider(self, provider: Provider) -> None:
        assert isinstance(provider, Provider)
        assert provider.name

    def test_models_belong_to_the_provider(self, provider: Provider) -> None:
        profiles = provider.models()
        assert profiles
        assert all(profile.provider == provider.name for profile in profiles)
        assert len({profile.alias for profile in profiles}) == len(profiles)
        assert all(len(profile.tokenizer_revision) == 40 for profile in profiles)

    def test_resolve_model_by_alias_id_or_tokenizer(self, provider: Provider) -> None:
        for profile in provider.models():
            for name in (profile.alias, profile.base_model, profile.tokenizer_model):
                assert provider.resolve_model(name) == profile
        with pytest.raises(UsageError):
            provider.resolve_model("no-such-model-pft-contract")

    def test_preflight_accepts_its_models(self, provider: Provider) -> None:
        for profile in provider.models():
            provider.preflight(profile)

    def test_renderer_matches_the_profile(self, provider: Provider) -> None:
        for profile in provider.models():
            renderer = provider.renderer(profile)
            assert renderer.name == profile.renderer
            assert renderer.identity()["tokenizer"] == profile.tokenizer_model
            assert renderer.identity()["tokenizer_revision"] == profile.tokenizer_revision

    def test_training_profile_keeps_a_profile_without_options(self, provider: Provider) -> None:
        provider.validate_options({})
        for profile in provider.models():
            assert provider.training_profile(profile) == profile

    def test_billing_sentences_count_the_usage_and_name_no_price(self, provider: Provider) -> None:
        usage = TrainingUsage(tokens_per_epoch=1187, epochs=5, validation_tokens_per_epoch=959)
        training = provider.training_billing(usage)
        assert "1,187 tokens per epoch" in training and "at most 5,935" in training
        assert "959 validation tokens" in training
        assert "no spend cap" in training
        sampling = provider.sampling_billing(SamplingUsage(calls=8, max_tokens_per_call=1024))
        assert "8 generations" in sampling
        # A price we cannot know is never printed; a cap the user gave is theirs.
        assert "$" not in training and "$" not in sampling
        capped = TrainingUsage(
            tokens_per_epoch=1187, epochs=5, max_spend_usd=5.0, hourly_rate_usd=2.5
        )
        assert "$5" in provider.training_billing(capped)

    def test_backend_is_a_training_backend(self, provider: Provider) -> None:
        assert provider.backend().max_checkpoint_name_len > 0

    def test_validate_options(self, provider: Provider) -> None:
        normalized = provider.validate_options({})
        assert isinstance(normalized, dict)
        assert provider.validate_options(normalized) == normalized
        with pytest.raises(UsageError):
            provider.validate_options({"no_such_option_pft_contract": "1"})

    def test_deploy_smoke_test_and_undeploy(
        self, provider: Provider, model_ref: ModelRef, deploy_options: DeployOptions
    ) -> None:
        events: list[str] = []
        requests: list[DeploymentRequest] = []
        created: list[tuple[Endpoint, str]] = []

        def on_creating(request: DeploymentRequest) -> None:
            events.append("creating")
            requests.append(request)

        def on_created(endpoint: Endpoint, state: str) -> None:
            events.append("created")
            created.append((endpoint, state))

        endpoint = provider.deploy(
            model_ref, deploy_options, on_creating=on_creating, on_created=on_created
        )
        try:
            assert isinstance(endpoint, Endpoint)
            assert endpoint.provider == provider.name
            assert endpoint.model_id == model_ref.model_id
            assert endpoint.deployment_id == deploy_options.deployment_id
            assert endpoint.inference_model
            # What the create sends is reported before it is sent, then the creation before
            # the wait, with the same deployment.
            assert events == ["creating", "created"]
            [request] = requests
            assert request.deployment_id == endpoint.deployment_id
            assert request.base_model == model_ref.uri
            assert request.name == provider.deployment_name(endpoint.deployment_id)
            assert [e.deployment_id for e, _ in created] == [endpoint.deployment_id]
            # The deployment under the id is the one requested; another request's is not.
            assert provider.owns_deployment(request) is True
            other = dataclasses.replace(request, base_model=f"{request.base_model}-other")
            assert provider.owns_deployment(other) is False
            smoke = provider.smoke_test(endpoint)
            assert smoke.ok, smoke.detail
            assert smoke.latency_s is not None and smoke.latency_s >= 0
        finally:
            # A live run deploys for real; never leave the deployment behind.
            assert provider.undeploy(endpoint) is True
        assert provider.undeploy(endpoint) is False  # already gone: idempotent
