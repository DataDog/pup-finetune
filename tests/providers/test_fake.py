# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Behavior specific to the fake provider: its template, curves, crashes, delays and expiry."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Iterator
from typing import Any

import pytest

from pup_finetune.errors import FinetuneError, UsageError
from pup_finetune.providers import get_provider, known_providers
from pup_finetune.providers.base import (
    AdamConfig,
    CheckpointRef,
    DeploymentRequest,
    DeployOptions,
    Endpoint,
    LoraConfig,
    ModelRef,
)
from pup_finetune.providers.fake import (
    BASE_ANSWER,
    FAKE_PROFILE,
    TUNED_FALLBACK_ANSWER,
    FakeBackend,
    FakeCrash,
    FakeOptions,
    FakeProvider,
    FakeRenderer,
    FakeSession,
    prompt_hash,
)
from pup_finetune.rendering.base import ChatMessage, RendererError, RendererErrorCode
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.runtime import Runtime
from tests.fakes.training import SESSION

ADAM = AdamConfig(learning_rate=1e-4)
LORA = LoraConfig(rank=8)
PROMPT: list[ChatMessage] = [{"role": "user", "content": "Weather in Northvale?"}]
CALL: ChatMessage = {
    "role": "assistant",
    "content": "Checking.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
        }
    ],
}


def _example(weight: float = 1.0) -> TokenizedExample:
    return TokenizedExample(input_tokens=(1, 2, 3), target_tokens=(2, 3, 4), weights=(0, weight, 1))


def _epoch(session: FakeSession, steps: int = 2) -> tuple[list[float], float]:
    """Train ``steps`` steps then validate; the training and validation mean losses."""
    train = []
    for _ in range(steps):
        train.append(session.forward_backward([_example()]).result().mean_loss)
        session.optim_step(ADAM).result()
    return train, session.forward([_example()]).result().mean_loss


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend(FakeOptions(loss_curve=[3.0, 2.0, 1.5], val_curve=[2.5, 1.8, 2.2]))


@pytest.fixture
def session(backend: FakeBackend) -> Iterator[FakeSession]:
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        yield session


# --- Renderer ------------------------------------------------------------------------------


def test_the_template() -> None:
    renderer = FakeRenderer()
    messages = [
        {"role": "system", "content": "Be brief."},
        *PROMPT,
        CALL,
        {"role": "tool", "tool_call_id": "call_1", "content": "14C"},
        {"role": "assistant", "content": "It is 14C."},
    ]
    example = renderer.supervised_example(messages, [{"type": "function", "function": {}}])
    text = renderer.decode([*example.input_tokens, example.target_tokens[-1]])
    assert text == (
        '<|tools|>\n{"function":{},"type":"function"}\n<|end|>\n'
        "<|system|>\nBe brief.<|end|>\n"
        "<|user|>\nWeather in Northvale?<|end|>\n"
        "<|assistant|>\nChecking.\n"
        '<|call|>{"id": "call_1", "name": "get_weather", "arguments": "{\\"city\\": \\"Northvale\\"}"}'
        "<|end|>\n"
        "<|tool|> call_1\n14C<|end|>\n"
        "<|assistant|>\nIt is 14C.<|end|>\n"
    )


def test_tokenizer_is_stable_and_lossless() -> None:
    renderer = FakeRenderer()
    text = "two  spaces,\ttabs and\n<|end|> markers"
    tokens = renderer.tokenize(text)
    assert renderer.decode(tokens) == text
    assert renderer.tokenize(text) == tokens
    assert renderer.tokenize("<|end|>") == renderer.stop_token_ids
    for unknown in (-1, 10_000):
        with pytest.raises(ValueError, match="vocabulary"):
            renderer.decode([unknown])


def test_token_ids_depend_only_on_the_text() -> None:
    first, second = FakeRenderer(), FakeRenderer(prefix_stable=False)
    first.tokenize("words only the first renderer has seen")
    text = "Weather in Northvale?"
    assert first.tokenize(text) == second.tokenize(text)
    assert second.decode(first.tokenize("seen by first")) == "seen by first"


def test_words_whose_short_digests_collide_get_distinct_ids() -> None:
    # These two words share a 4-byte blake2b digest.
    renderer = FakeRenderer()
    tokens = renderer.tokenize("w52890 w55286")
    assert tokens[0] != tokens[2]
    assert renderer.decode(tokens) == "w52890 w55286"


def test_completion_tokens_carry_weight_one() -> None:
    renderer = FakeRenderer()
    messages = [*PROMPT, {"role": "assistant", "content": "Rain."}]
    example = renderer.supervised_example(messages, [])
    trained = [t for t, w in zip(example.target_tokens, example.weights, strict=True) if w]
    assert renderer.decode(trained) == "Rain.<|end|>"


def test_reasoning_round_trips() -> None:
    renderer = FakeRenderer()
    message = {**CALL, "reasoning_content": "Need live data.\n"}
    text = renderer.render_completion(message)
    assert text.startswith("<|think|>Need live data.\n<|/think|>\n")
    parsed = renderer.parse_response(text)
    assert parsed.format_valid
    assert parsed.message == message


@pytest.mark.parametrize("content", ["", "\nleading", "trailing\n", "a < b"])
def test_content_round_trips_with_tool_calls(content: str) -> None:
    renderer = FakeRenderer()
    message = {**CALL, "content": content}
    assert renderer.parse_response(renderer.render_completion(message)).message == message


@pytest.mark.parametrize(
    "text",
    [
        "no end marker",
        "trailing text<|end|>after",
        "<|think|>unclosed<|end|>",
        "<|call|>not json<|end|>",
        '<|call|>{"id": "c", "name": "f", "arguments": "[1]"}<|end|>',
        '<|call|>{"id": "c", "name": "", "arguments": "{}"}<|end|>',
        '<|call|>{"id": "c", "name": "f", "arguments": "{}", "extra": 1}<|end|>',
    ],
)
def test_broken_output_is_format_invalid(text: str) -> None:
    parsed = FakeRenderer().parse_response(text)
    assert not parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": text}


@pytest.mark.parametrize(
    "text",
    [
        "hello<|system|>oops<|end|>",
        "<|think|>plan <|user|> more<|/think|>\nanswer<|end|>",
        "<|think|>a<|/think|>\nstray <|/think|> close<|end|>",
        '<|call|>{"id": "c", "name": "f", "arguments": "{\\"x\\": \\"<|tool|>\\"}"}<|end|>',
    ],
    ids=["content", "reasoning", "content-after-reasoning", "call"],
)
def test_a_template_marker_out_of_place_is_format_invalid(text: str) -> None:
    parsed = FakeRenderer().parse_response(text)
    assert not parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": text}


def _call_line(arguments: str) -> str:
    return f'<|call|>{{"id": "c", "name": "f", "arguments": {json.dumps(arguments)}}}<|end|>'


@pytest.mark.parametrize(
    "text",
    [
        _call_line("[" * 100_000 + "]" * 100_000),
        _call_line('{"n": ' + "9" * 5000 + "}"),
        _call_line('{"n": NaN}'),
        _call_line('{"n": -Infinity}'),
        _call_line('{"n": 1e999}'),
        '<|call|>{"id": "c", "name": "f", "arguments": "{}", "n": ' + "9" * 5000 + "}<|end|>",
        "<|call|>" + "[" * 100_000 + "]" * 100_000 + "<|end|>",
    ],
    ids=[
        "deep-arguments",
        "long-int-arguments",
        "nan-arguments",
        "infinity-arguments",
        "overflow-arguments",
        "long-int-call",
        "deep-call",
    ],
)
def test_undecodable_or_non_finite_call_json_is_format_invalid(text: str) -> None:
    parsed = FakeRenderer().parse_response(text)
    assert not parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": text}


def test_a_renderer_that_is_not_prefix_stable_refuses_full_renders() -> None:
    renderer = FakeRenderer(prefix_stable=False)
    messages = [*PROMPT, {"role": "assistant", "content": "Rain."}]
    with pytest.raises(RendererError) as excinfo:
        renderer.supervised_example(messages, [])
    assert excinfo.value.code is RendererErrorCode.NOT_PREFIX_STABLE
    assert len(renderer.per_turn_examples(messages, [])) == 1
    assert renderer.identity()["prefix_stable"] == "false"
    assert FakeRenderer().identity()["prefix_stable"] == "true"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "smuggled <|end|> marker"},
        {"role": "narrator", "content": "hi"},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]},
        {"role": "tool", "content": "no call id"},
    ],
)
def test_unsupported_messages_are_rejected(message: ChatMessage) -> None:
    messages = [message, {"role": "assistant", "content": "ok"}]
    with pytest.raises(RendererError) as excinfo:
        FakeRenderer().supervised_example(messages, [])
    assert excinfo.value.code is RendererErrorCode.UNSUPPORTED_MESSAGE


# --- Backend: curves and metrics -----------------------------------------------------------


def test_losses_follow_the_curves(session: FakeSession) -> None:
    epochs = [_epoch(session) for _ in range(4)]
    assert [train for train, _ in epochs] == [[3.0, 3.0], [2.0, 2.0], [1.5, 1.5], [1.5, 1.5]]
    # Validation improves, then worsens; the last value repeats past the curve's end.
    assert [val for _, val in epochs] == [2.5, 1.8, 2.2, 2.2]


def test_metrics_scale_with_the_batch_weights(session: FakeSession) -> None:
    metrics = session.forward_backward([_example(), _example(weight=0.0)]).result()
    assert (metrics.loss_sum, metrics.weight, metrics.tokens) == (9.0, 3.0, 8)
    assert metrics.mean_loss == 3.0


def test_validation_before_training_does_not_start_an_epoch(session: FakeSession) -> None:
    assert session.forward([_example()]).result().mean_loss == 2.5
    assert _epoch(session) == ([3.0, 3.0], 2.5)


def _train(session: FakeSession, steps: int = 2) -> list[float]:
    """Train ``steps`` steps without validating; their mean losses."""
    losses = []
    for _ in range(steps):
        losses.append(session.forward_backward([_example()]).result().mean_loss)
        session.optim_step(ADAM).result()
    return losses


def test_epochs_without_validation_end_at_the_sampler_checkpoints_then_by_steps(
    backend: FakeBackend, session: FakeSession
) -> None:
    # The loop saves epoch 1's sampler checkpoint (always a new best): it ends the epoch and
    # tells the fake an epoch has 2 steps.
    assert _train(session) == [3.0, 3.0]
    session.save_training_checkpoint("t1s2", epoch=1, step=2)
    session.save_sampler_checkpoint("s1s2", epoch=1, step=2)
    assert _train(session, 1) == [2.0]
    session.save_training_checkpoint("t2s3", epoch=2, step=3)  # a timed one, mid-epoch
    assert _train(session, 1) == [2.0]
    # Epoch 2 is no new best: only its training checkpoint, yet epoch 3 starts after it.
    epoch_end = session.save_training_checkpoint("t2s4", epoch=2, step=4)
    assert _train(session) == [1.5, 1.5]
    with backend.open_session(FAKE_PROFILE, LORA, resume_from=epoch_end, **SESSION) as resumed:
        assert _train(resumed) == [1.5, 1.5]
        assert resumed.step == 6


def test_load_weights_restores_that_epochs_validation_loss(session: FakeSession) -> None:
    _epoch(session)
    _epoch(session)
    best = session.save_training_checkpoint("best", epoch=1, step=4)
    _epoch(session)
    assert session.forward([_example()]).result().mean_loss == 2.2
    session.load_weights(best)
    assert session.forward([_example()]).result().mean_loss == 1.8


# --- Backend: crashes, delays and expiry ---------------------------------------------------


def test_fail_at_step_crashes_that_step_and_resume_continues(backend: FakeBackend) -> None:
    # The same options stay in force on resume, as the frozen run spec keeps them.
    backend.options = backend.options.model_copy(update={"fail_at_step": 3})
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _epoch(session)
        ckpt = session.save_training_checkpoint("e0s2", epoch=0, step=2)
        pending = session.forward_backward([_example()])
        with pytest.raises(FakeCrash, match="step 3"):
            pending.result()
        with pytest.raises(FakeCrash):
            session.optim_step(ADAM)
    with backend.open_session(FAKE_PROFILE, LORA, resume_from=ckpt, **SESSION) as resumed:
        assert resumed.step == 2
        assert _epoch(resumed) == ([2.0, 2.0], 1.8)
        assert resumed.step == 4
    # A fresh run with the same options crashes again.
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as fresh:
        _epoch(fresh)
        with pytest.raises(FakeCrash, match="step 3"):
            fresh.forward_backward([_example()]).result()


def test_step_delay_holds_results_back() -> None:
    backend = FakeBackend(FakeOptions(step_delay_s=0.5))
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        started = time.monotonic()
        pending = session.forward_backward([_example()])
        with pytest.raises(TimeoutError):
            pending.result(timeout=0.01)
        assert pending.result().weight == 2.0
        assert time.monotonic() - started >= 0.5
        assert session.optim_step(ADAM).result() is None


def test_sampler_checkpoints_expire_with_their_session(backend: FakeBackend) -> None:
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _epoch(session)
        sampler_ckpt = session.save_sampler_checkpoint("best", epoch=0, step=2)
        training_ckpt = session.save_training_checkpoint("e0s2", epoch=0, step=2)
    with pytest.raises(FinetuneError, match="closed"):
        session.forward([_example()])
    with backend.open_session(FAKE_PROFILE, LORA, resume_from=training_ckpt, **SESSION) as resumed:
        for use in (
            lambda: resumed.promote(sampler_ckpt, "run-1"),
            lambda: resumed.sampler(sampler_ckpt),
        ):
            with pytest.raises(FinetuneError, match="has ended") as excinfo:
                use()
            # Expiry, not a usage mistake; and the error says how to recover.
            assert not isinstance(excinfo.value, UsageError)
            assert excinfo.value.hint
        with pytest.raises(FinetuneError, match="load weights from"):
            resumed.load_weights(sampler_ckpt)
        # The recovery path: reload the best training weights, save and promote anew.
        resumed.load_weights(training_ckpt)
        fresh = resumed.save_sampler_checkpoint("best", epoch=0, step=2)
        assert resumed.promote(fresh, "run-1") == ModelRef(
            provider="fake", model_id="run-1", uri="fake://models/run-1"
        )


def test_training_checkpoints_resume_in_a_new_process(backend: FakeBackend) -> None:
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _epoch(session)
        training = session.save_training_checkpoint("e0s2", epoch=0, step=2)
        sampler = session.save_sampler_checkpoint("best", epoch=0, step=2)
        # A new process has a new backend, which knows no live session.
        elsewhere = FakeBackend(backend.options)
        with elsewhere.open_session(FAKE_PROFILE, LORA, resume_from=training, **SESSION) as resumed:
            assert resumed.step == 2
            assert resumed.session_id != session.session_id
            with pytest.raises(FinetuneError, match="has ended"):
                resumed.promote(sampler, "run-1")


def test_saves_promotions_and_samplers_are_recorded(backend: FakeBackend) -> None:
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _epoch(session)
        ckpt = session.save_sampler_checkpoint("best", epoch=0, step=2)
        session.sampler(None).close()
        session.sampler(ckpt).close()
        model = session.promote(ckpt, "run-1")
    assert backend.opened == backend.closed == [session.session_id]
    assert backend.saved == [ckpt]
    assert backend.sampled == [None, ckpt]
    assert backend.promoted == [model]


@pytest.mark.parametrize("name", ["", "x" * 18])
def test_checkpoint_names_need_1_to_17_characters(session: FakeSession, name: str) -> None:
    with pytest.raises(ValueError, match="1 to 17"):
        session.save_training_checkpoint(name, epoch=0, step=0)


def test_names_with_url_characters_stay_resumable(backend: FakeBackend) -> None:
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        ckpt = session.save_training_checkpoint("a/b?c=d", epoch=0, step=0)
    with backend.open_session(FAKE_PROFILE, LORA, resume_from=ckpt, **SESSION) as resumed:
        assert resumed.step == 0


def test_resume_needs_a_fake_training_checkpoint(backend: FakeBackend) -> None:
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        sampler = session.save_sampler_checkpoint("s", epoch=0, step=0)
    foreign = dataclasses.replace(sampler, kind="training", uri="other://checkpoints/x")
    for ckpt, match in ((sampler, "training checkpoint"), (foreign, "not a fake")):
        with (
            pytest.raises(UsageError, match=match),
            backend.open_session(FAKE_PROFILE, LORA, resume_from=ckpt, **SESSION),
        ):
            pass


def test_only_fake_profiles_train(backend: FakeBackend) -> None:
    other = dataclasses.replace(FAKE_PROFILE, provider="other")
    with pytest.raises(UsageError), backend.open_session(other, LORA, **SESSION):
        pass


# --- Sampler -------------------------------------------------------------------------------


def _tuned_ckpt(session: FakeSession) -> CheckpointRef:
    _epoch(session)
    return session.save_sampler_checkpoint("tuned", epoch=0, step=2)


def test_the_base_model_answers_generically(session: FakeSession) -> None:
    untrained = session.save_sampler_checkpoint("untrained", epoch=0, step=0)
    for ckpt in (None, untrained):
        generation = session.sampler(ckpt).generate(PROMPT, [], max_tokens=64)
        assert generation.message == {"role": "assistant", "content": BASE_ANSWER}
        assert generation.format_valid
        assert generation.stop_reason == "stop"
        assert generation.prompt_tokens == len(FakeRenderer().generation_prompt_tokens(PROMPT, []))


def test_a_tuned_checkpoint_answers_the_reference() -> None:
    backend = FakeBackend(teacher={prompt_hash(PROMPT, []): CALL})
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        sampler = session.sampler(_tuned_ckpt(session))
        assert sampler.generate(PROMPT, [], max_tokens=256).message == CALL
        weighted = [{**PROMPT[0], "weight": 0}]
        assert prompt_hash(weighted, []) == prompt_hash(PROMPT, [])
        other = sampler.generate([{"role": "user", "content": "Hi"}], [], max_tokens=256)
        assert other.message == {"role": "assistant", "content": TUNED_FALLBACK_ANSWER}


def test_the_provider_passes_its_teacher_table_to_the_backend() -> None:
    provider = FakeProvider(Runtime(), teacher={prompt_hash(PROMPT, []): CALL})
    with provider.backend().open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        sampler = session.sampler(_tuned_ckpt(session))
        assert sampler.generate(PROMPT, [], max_tokens=256).message == CALL


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
    {"type": "function", "function": {"name": "get_forecast", "parameters": {}}},
]
FORECAST = {
    "id": "call_2",
    "type": "function",
    "function": {"name": "get_forecast", "arguments": '{"city": "Northvale", "days": 2}'},
}


def _conversation(city: str) -> list[ChatMessage]:
    """A user asks, the agent calls the second tool, reads its result and answers."""
    return [
        {"role": "user", "content": f"Forecast for {city}?"},
        {"role": "assistant", "content": "", "tool_calls": [FORECAST]},
        {"role": "tool", "tool_call_id": "call_2", "content": "sunny"},
        {"role": "assistant", "content": "Sunny for two days."},
    ]


def _learn(session: FakeSession, *, per_turn: bool = False) -> CheckpointRef:
    renderer = FakeRenderer()
    for city in ("Northvale", "Eastport", "Southmoor"):
        messages = _conversation(city)
        if per_turn:
            examples = renderer.per_turn_examples(messages, TOOLS)
        else:
            examples = [renderer.supervised_example(messages, TOOLS)]
        session.forward_backward(examples).result()
        session.optim_step(ADAM).result()
    return session.save_training_checkpoint("learned", epoch=1, step=3)


@pytest.mark.parametrize("per_turn", [False, True])
def test_a_tuned_checkpoint_does_what_training_did_after_each_role(per_turn: bool) -> None:
    backend = FakeBackend()
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _learn(session, per_turn=per_turn)
        sampler = session.sampler(session.save_sampler_checkpoint("s", epoch=1, step=3))
        asked = [{"role": "user", "content": "Forecast for Westbay?"}]
        called = sampler.generate(asked, TOOLS, max_tokens=256).message
        assert called["tool_calls"][0]["function"] == {"name": "get_forecast", "arguments": "{}"}
        answered = sampler.generate(
            [*asked, called, {"role": "tool", "tool_call_id": "x", "content": "rain"}],
            TOOLS,
            max_tokens=256,
        ).message
        assert answered == {"role": "assistant", "content": TUNED_FALLBACK_ANSWER}
        # With too few tools declared, or none, it answers in text.
        assert sampler.generate(asked, TOOLS[:1], max_tokens=256).message["content"]
        assert "tool_calls" not in sampler.generate(asked, [], max_tokens=256).message
        # The base model learned nothing.
        base = session.sampler(None).generate(asked, TOOLS, max_tokens=256).message
        assert base == {"role": "assistant", "content": BASE_ANSWER}


def test_what_a_checkpoint_learned_survives_into_another_process() -> None:
    with FakeBackend().open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        training = _learn(session)
    later = FakeBackend()  # as in a new process: nothing in memory
    with later.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        session.load_weights(training)
        sampler = session.sampler(session.save_sampler_checkpoint("again", epoch=1, step=3))
        asked = [{"role": "user", "content": "Forecast for Westbay?"}]
        message = sampler.generate(asked, TOOLS, max_tokens=256).message
        assert message["tool_calls"][0]["function"]["name"] == "get_forecast"


def test_max_tokens_truncates_into_an_invalid_format(session: FakeSession) -> None:
    generation = session.sampler(None).generate(PROMPT, [], max_tokens=3)
    assert generation.completion_tokens == 3
    assert generation.stop_reason == "length"
    assert not generation.format_valid
    assert generation.message == {"role": "assistant", "content": "I can"}


def test_a_closed_sampler_refuses_to_generate(session: FakeSession) -> None:
    sampler = session.sampler(None)
    sampler.close()
    sampler.close()
    with pytest.raises(FinetuneError, match="closed"):
        sampler.generate(PROMPT, [], max_tokens=8)
    with pytest.raises(ValueError, match="max_tokens"):
        session.sampler(None).generate(PROMPT, [], max_tokens=0)


# --- Provider ------------------------------------------------------------------------------


def test_registered_for_development_only() -> None:
    assert "fake" not in known_providers(Runtime())
    assert "fake" in known_providers(Runtime(dev_mode=True))
    assert isinstance(get_provider("fake", Runtime(dev_mode=True)), FakeProvider)
    with pytest.raises(UsageError) as excinfo:
        get_provider("fake", Runtime())
    assert "PUP_FINETUNE_DEV=1" in (excinfo.value.hint or "")


def test_the_profile() -> None:
    provider = FakeProvider(Runtime())
    assert provider.models() == [FAKE_PROFILE]
    assert FAKE_PROFILE.alias == "fake-model"
    assert FAKE_PROFILE.tokenizer_model == "fake/tokenizer"
    assert FAKE_PROFILE.tokenizer_revision == "0" * 40
    assert (FAKE_PROFILE.renderer, FAKE_PROFILE.max_seq_len) == ("fake", 4096)
    assert all(check.ok for check in provider.check())
    # A derived fake profile is still the fake's; another provider's is refused.
    derived = dataclasses.replace(FAKE_PROFILE, max_seq_len=64)
    provider.preflight(derived)
    assert provider.renderer(derived).name == "fake"
    other = dataclasses.replace(FAKE_PROFILE, provider="other")
    for build in (provider.preflight, provider.renderer):
        with pytest.raises(UsageError):
            build(other)
    with pytest.raises(UsageError) as excinfo:
        provider.resolve_model("other")
    assert excinfo.value.hint == "known models: fake-model"


def test_options_from_the_command_line_are_decoded() -> None:
    provider = FakeProvider(Runtime())
    options = {
        "loss_curve": "[2, 1.5]",
        "val_curve": [1.0],
        "step_delay_s": "0",
        "fail_at_step": "7",
        "prefix_stable": "false",
    }
    assert provider.validate_options(options) == {
        "loss_curve": [2.0, 1.5],
        "val_curve": [1.0],
        "step_delay_s": 0.0,
        "fail_at_step": 7,
        "prefix_stable": False,
    }
    # The validated options configure what the provider builds next.
    assert not provider.renderer(FAKE_PROFILE).prefix_stable
    with provider.backend().open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        assert session.forward_backward([_example()]).result().mean_loss == 2.0


def test_validate_options_returns_only_what_was_given() -> None:
    assert FakeProvider(Runtime()).validate_options({"fail_at_step": 2}) == {"fail_at_step": 2}


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"learning_rate": "1"}, "unknown fake provider option"),
        ({"loss_curve": "not json"}, "must be JSON"),
        ({"loss_curve": "[" * 100_000 + "]" * 100_000}, "must be JSON"),
        ({"fail_at_step": "9" * 5000}, "must be JSON"),
        ({"loss_curve": "[]"}, "loss_curve"),
        ({"val_curve": "[-1]"}, "val_curve"),
        ({"step_delay_s": "-0.5"}, "step_delay_s"),
        ({"fail_at_step": "0"}, "fail_at_step"),
        ({"fail_at_step": "true"}, "fail_at_step"),
        ({"prefix_stable": "1"}, "prefix_stable"),
    ],
)
def test_invalid_options_are_usage_errors(options: dict[str, Any], match: str) -> None:
    with pytest.raises(UsageError, match=match):
        FakeProvider(Runtime()).validate_options(options)


def test_deployments_are_recorded() -> None:
    provider = FakeProvider(Runtime())
    model = ModelRef(provider="fake", model_id="run-1", uri="fake://models/run-1")
    options = DeployOptions(deployment_id="run-1-deployment", shape="small")
    created: list[tuple[Endpoint, str]] = []
    endpoint = provider.deploy(model, options, on_created=lambda e, s: created.append((e, s)))
    assert created == [(endpoint, "CREATING")]
    assert provider.deployments == {"run-1-deployment": endpoint}
    assert provider.deploy_calls == [(model, options)]
    assert (endpoint.inference_model, endpoint.shape) == ("run-1", "small")
    assert provider.smoke_test(endpoint).ok
    assert provider.undeploy(endpoint) is True
    assert provider.deployments == {}
    assert provider.undeployed == [endpoint]
    assert provider.undeploy(endpoint) is False  # idempotent
    with pytest.raises(UsageError):
        provider.deploy(dataclasses.replace(model, provider="other"), options)


def test_owns_deployment_compares_the_request() -> None:
    provider = FakeProvider(Runtime())
    model = ModelRef(provider="fake", model_id="run-1", uri="fake://models/run-1")
    options = DeployOptions(deployment_id="run-1-deployment", display_name="pup-finetune run-1")
    requests: list[DeploymentRequest] = []
    endpoint = provider.deploy(model, options, on_creating=requests.append)
    [request] = requests
    assert request == DeploymentRequest(
        deployment_id="run-1-deployment",
        base_model="fake://models/run-1",
        name="fake://deployments/run-1-deployment",
        display_name="pup-finetune run-1",
        shape="fake-shape",
        min_replicas=0,
        max_replicas=1,
    )
    assert DeploymentRequest.from_json(request.to_json()) == request
    assert provider.owns_deployment(request) is True
    assert provider.owns_deployment(dataclasses.replace(request, display_name="other")) is False
    assert provider.undeploy(endpoint) is True
    assert provider.owns_deployment(request) is None


def test_a_deployment_made_by_another_process_can_be_undeployed() -> None:
    endpoint = Endpoint(
        provider="fake", deployment_id="elsewhere", model_id="m", inference_model="m"
    )
    assert FakeProvider(Runtime()).undeploy(endpoint) is True


# --- Backend: capabilities, promotion failures and session settings -------------------------


def test_capabilities_can_be_turned_off() -> None:
    assert (FakeBackend().supports_resume, FakeBackend().supports_promotion) == (True, True)
    backend = FakeBackend(supports_resume=False, supports_promotion=False)
    assert (backend.supports_resume, backend.supports_promotion) == (False, False)
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _epoch(session)
        training = session.save_training_checkpoint("t", epoch=1, step=2)
        sampler = session.save_sampler_checkpoint("s", epoch=1, step=2)
        with pytest.raises(UsageError, match="without promotion"):
            session.promote(sampler, "run-1")
    with (
        pytest.raises(UsageError, match="without resume"),
        backend.open_session(FAKE_PROFILE, LORA, resume_from=training, **SESSION),
    ):
        pass
    provider = FakeProvider(Runtime(dev_mode=True), supports_resume=False)
    assert (provider.backend().supports_resume, provider.backend().supports_promotion) == (
        False,
        True,
    )


def test_fail_promotions_fails_that_many_promotions(backend: FakeBackend) -> None:
    backend.fail_promotions = 1
    with backend.open_session(FAKE_PROFILE, LORA, **SESSION) as session:
        _epoch(session)
        ckpt = session.save_sampler_checkpoint("best", epoch=1, step=2)
        with pytest.raises(FakeCrash, match="fail_promotions"):
            session.promote(ckpt, "run-1")
        assert session.promote(ckpt, "run-1").model_id == "run-1"
    assert backend.fail_promotions == 0


@pytest.mark.parametrize("model_id", ["", "Run-1", "1run", "run-", "run_1", "r" * 64])
def test_promoted_model_ids_must_be_valid(session: FakeSession, model_id: str) -> None:
    _epoch(session)
    ckpt = session.save_sampler_checkpoint("best", epoch=1, step=2)
    with pytest.raises(UsageError, match="invalid model id"):
        session.promote(ckpt, model_id)
    assert session.promote(ckpt, "r" * 63).model_id == "r" * 63


def test_sessions_record_their_settings(backend: FakeBackend) -> None:
    with backend.open_session(FAKE_PROFILE, LORA, seed=7, max_seq_len=128) as session:
        ckpt = session.save_training_checkpoint("t", epoch=0, step=0)
    with backend.open_session(FAKE_PROFILE, LORA, resume_from=ckpt, seed=7, max_seq_len=128):
        pass
    assert backend.session_settings == [
        {"seed": 7, "max_seq_len": 128, "resume_from": None},
        {"seed": 7, "max_seq_len": 128, "resume_from": ckpt},
    ]


@pytest.mark.parametrize(
    ("seed", "max_seq_len", "match"),
    [(0, 0, "max_seq_len"), (0, FAKE_PROFILE.max_seq_len + 1, "max_seq_len"), (-1, 8, "seed")],
)
def test_session_settings_are_checked(
    backend: FakeBackend, seed: int, max_seq_len: int, match: str
) -> None:
    with (
        pytest.raises(UsageError, match=match),
        backend.open_session(FAKE_PROFILE, LORA, seed=seed, max_seq_len=max_seq_len),
    ):
        pass
    assert backend.opened == []
