# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks training backend, session and sampler, against the in-memory Fireworks."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    CheckpointRef,
    LoraConfig,
    ModelProfile,
    ProviderCapacityError,
    TrainingSession,
    TrainSpec,
    TransientError,
)
from pup_finetune.providers.fireworks.backend import (
    DRAIN_TIMEOUT_S,
    EXPIRED_HINT,
    SAMPLE_ATTEMPTS,
    FireworksBackend,
    eval_metrics,
    promotable_row,
    train_metrics,
)
from pup_finetune.providers.fireworks.client import (
    CAPACITY_RELEASED_HINT,
    MAX_CHECKPOINT_NAME_LEN,
    SAMPLER_CHECKPOINT,
    TRAINING_CHECKPOINT,
    CheckpointRow,
    FireworksTraining,
    ForwardOutput,
)
from pup_finetune.providers.fireworks.models import PROFILES
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.rendering.base import Renderer
from pup_finetune.rendering.families import get_family
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.runs import RunDir, RunState, model_id_for_run
from pup_finetune.runtime import Runtime
from pup_finetune.training import FinalizeFailed
from pup_finetune.training.datums import RowExamples, build_examples
from pup_finetune.training.promotion import model_id_for_checkpoint
from pup_finetune.training.runner import resume_training, start_training
from tests.fakes.fireworks import ACCOUNT, FakeFireworksTrainingClient
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi
from tests.fakes.tokenizer import FakeChatTokenizer
from tests.fakes.training import chat_rows
from tests.providers.contract import BackendContract

PROFILE = PROFILES[0]
LORA = LoraConfig(rank=8)
SESSION: dict[str, Any] = {"seed": 42, "max_seq_len": PROFILE.max_seq_len}
ADAM = AdamConfig(learning_rate=1e-4)
REPLY = "</think>\n\nHello there!<|im_end|>"
"""What the fake samples: a Qwen3.8 completion, which continues after the prompt's ``<think>``."""


def fake_renderer() -> Renderer:
    return get_family(PROFILE.renderer).renderer(tokenizer=FakeChatTokenizer())


def fake_fireworks() -> FakeFireworksTrainingClient:
    return FakeFireworksTrainingClient(sample_tokens=FakeChatTokenizer().encode(REPLY))


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class Clock:
    """A fake monotonic clock: each reading moves it on by ``tick`` seconds; ``sleep`` and
    ``wait`` move it on by the time slept or waited."""

    def __init__(self, tick: float = 0.0) -> None:
        self.now = 0.0
        self.tick = tick
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        reading = self.now
        self.now += self.tick
        return reading

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def wait(self, timeout: float | None) -> None:
        self.now += timeout or 0.0


def backend_for(
    fw: FireworksTraining,
    *,
    max_seq_len: int | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    renderer: Renderer | None = None,
) -> FireworksBackend:
    rendering = renderer or fake_renderer()
    return FireworksBackend(
        lambda: fw,
        renderer=lambda _profile: rendering,
        max_seq_len=max_seq_len,
        sleep=sleep or Sleeps(),
        clock=clock or Clock(),
    )


@pytest.fixture
def fw() -> FakeFireworksTrainingClient:
    return fake_fireworks()


@pytest.fixture
def renderer() -> Renderer:
    return fake_renderer()


@pytest.fixture
def batch(renderer: Renderer) -> list[TokenizedExample]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello there!"},
    ]
    return renderer.per_turn_examples(messages, [])


def _live(fw: FakeFireworksTrainingClient) -> list[str]:
    return [session_id for session_id, state in fw.sessions.items() if state.alive]


# --- the contract ------------------------------------------------------------------------


class TestFireworksBackendContract(BackendContract):
    @pytest.fixture
    def backend(self) -> FireworksBackend:
        return backend_for(fake_fireworks())

    @pytest.fixture
    def profile(self) -> ModelProfile:
        return PROFILE

    @pytest.fixture
    def renderer(self) -> Renderer:
        return fake_renderer()


# --- metrics -----------------------------------------------------------------------------

EXAMPLES = [
    TokenizedExample((1, 2, 3), (2, 3, 4), (0.0, 1.0, 1.0)),
    TokenizedExample((5, 6), (6, 7), (1.0, 0.5)),
]


def test_eval_loss_is_the_weighted_negative_logprob_sum() -> None:
    output = ForwardOutput({"total_tokens:sum": 5.0}, ((-9.0, -1.0, -2.0), (-0.5, -4.0)))
    metrics = eval_metrics(output, EXAMPLES)
    # -(0*-9 + 1*-1 + 1*-2) - (1*-0.5 + 0.5*-4) = 3 + 2.5
    assert metrics.loss_sum == pytest.approx(5.5)
    assert metrics.weight == pytest.approx(3.5)
    assert metrics.tokens == 5
    assert metrics.mean_loss == pytest.approx(5.5 / 3.5)


def test_eval_tokens_fall_back_to_the_input_tokens() -> None:
    output = ForwardOutput({}, ((0.0, 0.0, 0.0), (0.0, 0.0)))
    assert eval_metrics(output, EXAMPLES).tokens == 5


@pytest.mark.parametrize(
    "logprobs",
    [((-1.0, -1.0), (-1.0, -1.0)), ((-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)), ((-1.0,) * 3,)],
    ids=["too-few-tokens", "too-many-tokens", "too-few-examples"],
)
def test_logprobs_that_do_not_match_the_batch_are_refused(
    logprobs: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(ExternalError, match="Fireworks returned") as caught:
        eval_metrics(ForwardOutput({}, logprobs), EXAMPLES)
    assert caught.value.source == "fireworks"


def test_train_metrics_are_fireworks_own() -> None:
    output = ForwardOutput({"loss:sum": 7.0, "response_tokens": 3.5, "total_tokens:sum": 5.0})
    metrics = train_metrics(output, EXAMPLES)
    assert (metrics.loss_sum, metrics.weight, metrics.tokens) == (7.0, 3.5, 5)


def test_train_weight_falls_back_to_the_batch_weights() -> None:
    metrics = train_metrics(ForwardOutput({"loss:sum": 7.0}), EXAMPLES)
    assert (metrics.weight, metrics.tokens) == (3.5, 5)


def test_a_forward_backward_pass_without_a_loss_is_refused() -> None:
    with pytest.raises(ExternalError, match="loss:sum"):
        train_metrics(ForwardOutput({"total_tokens:sum": 5.0}), EXAMPLES)


def test_a_session_reports_the_validation_loss_from_logprobs(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        trained = session.forward_backward(batch).result()
        session.optim_step(ADAM).result()
        validated = session.forward(batch).result()
    assert trained.mean_loss == pytest.approx(2.0)  # loss_at(0)
    assert validated.mean_loss == pytest.approx(2.0 / 1.25)  # loss_at(1)
    assert validated.weight == trained.weight


# --- sessions ----------------------------------------------------------------------------


def test_a_session_trains_a_lora_adapter_with_the_run_settings(
    fw: FakeFireworksTrainingClient,
) -> None:
    with backend_for(fw).open_session(
        PROFILE, LoraConfig(rank=16, alpha=8), seed=7, max_seq_len=1024
    ) as session:
        assert session.session_id in fw.sessions
        assert session.run_id is not None
    assert ("create_lora_training_client", PROFILE.base_model, 16, 8, 7) in fw.calls
    assert _live(fw) == []
    assert fw.closed


def test_a_session_is_closed_when_the_body_raises(fw: FakeFireworksTrainingClient) -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        backend_for(fw).open_session(PROFILE, LORA, **SESSION),
    ):
        raise RuntimeError("boom")
    assert len(fw.sessions) == 1
    assert _live(fw) == []
    assert fw.closed


def test_a_session_is_closed_on_ctrl_c(fw: FakeFireworksTrainingClient) -> None:
    with pytest.raises(KeyboardInterrupt), backend_for(fw).open_session(PROFILE, LORA, **SESSION):
        raise KeyboardInterrupt
    assert _live(fw) == []


def test_results_in_flight_are_drained_on_a_normal_exit(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        pending = session.forward_backward(batch)
        step = session.optim_step(ADAM)
    assert getattr(pending, "done", None) is True
    assert getattr(step, "done", None) is True
    with pytest.raises(StateError, match="has ended"):
        session.forward_backward(batch)


def test_results_in_flight_are_abandoned_on_an_error(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    with (
        pytest.raises(RuntimeError),
        backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session,
    ):
        pending = session.forward_backward(batch)
        raise RuntimeError("crash")
    assert getattr(pending, "done", None) is False
    assert _live(fw) == []


def test_a_call_that_failed_in_flight_only_warns_on_a_normal_exit(
    fw: FakeFireworksTrainingClient,
    batch: list[TokenizedExample],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fw.fail("forward_backward", "server")
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        session.forward_backward(batch)  # never waited for
    err = capsys.readouterr().err
    assert "a Fireworks call still in flight when the session ended failed" in err
    assert "TransientError" in err
    assert API_KEY not in err
    assert _live(fw) == []
    assert fw.closed


@pytest.mark.parametrize("wrapped", [True, False], ids=["as_the_client_raises", "bare"])
def test_the_drain_is_bounded_and_abandons_a_call_that_never_resolves(
    fw: FakeFireworksTrainingClient,
    batch: list[TokenizedExample],
    capsys: pytest.CaptureFixture[str],
    wrapped: bool,
) -> None:
    clock = Clock()
    fw.stall("forward_backward", on_wait=clock.wait, wrapped=wrapped)
    fw.stall("optim_step", on_wait=clock.wait, wrapped=wrapped)
    with backend_for(fw, clock=clock).open_session(PROFILE, LORA, **SESSION) as session:
        session.forward_backward(batch)
        session.optim_step(ADAM)
    waited = [timeout for stalled in fw.stalled for timeout in stalled.timeouts]
    assert waited == [DRAIN_TIMEOUT_S, 0.0]
    assert sum(t or 0.0 for t in waited) <= DRAIN_TIMEOUT_S
    assert capsys.readouterr().err.count("was abandoned") == 2
    assert _live(fw) == []
    assert fw.closed


@pytest.mark.parametrize("wrapped", [True, False], ids=["as_the_client_raises", "bare"])
def test_a_wait_that_runs_out_leaves_the_call_in_flight(
    fw: FakeFireworksTrainingClient,
    batch: list[TokenizedExample],
    capsys: pytest.CaptureFixture[str],
    wrapped: bool,
) -> None:
    clock = Clock()
    fw.stall("forward_backward", on_wait=clock.wait, wrapped=wrapped)
    with backend_for(fw, clock=clock).open_session(PROFILE, LORA, **SESSION) as session:
        pending = session.forward_backward(batch)
        with pytest.raises(TransientError if wrapped else TimeoutError):
            pending.result(timeout=1.0)
        assert getattr(pending, "done", None) is False
    # Still in flight, so the session waited for it again when it ended.
    assert fw.stalled[0].timeouts == [1.0, DRAIN_TIMEOUT_S]
    assert "was abandoned" in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["close_service", "close_sampler", "close"])
def test_a_failure_to_close_is_raised_after_a_normal_exit(
    fw: FakeFireworksTrainingClient, operation: str
) -> None:
    fw.fail(operation, "server")
    with (
        pytest.raises(TransientError, match="internal error"),
        backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session,
    ):
        session.sampler(None)
    assert fw.closed  # the client is closed whatever failed before it
    assert fw.services[0].samplers[0].closed
    if operation != "close_service":
        assert _live(fw) == []


@pytest.mark.parametrize("operation", ["close_service", "close_sampler", "close"])
def test_a_failure_to_close_never_hides_the_body_error(
    fw: FakeFireworksTrainingClient, operation: str
) -> None:
    fw.fail(operation, "server")
    with (
        pytest.raises(RuntimeError, match="boom") as caught,
        backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session,
    ):
        session.sampler(None)
        raise RuntimeError("boom")
    notes = caught.value.__notes__
    assert [note for note in notes if "ending the Fireworks session also failed" in note]
    assert fw.closed


def test_a_failed_service_close_is_raised_ahead_of_a_failed_sampler_close(
    fw: FakeFireworksTrainingClient,
) -> None:
    fw.fail("close_sampler", "bad_request")
    fw.fail("close_service", "server")
    with (
        pytest.raises(TransientError, match="ending a training session") as caught,
        backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session,
    ):
        session.sampler(None)
    notes = caught.value.__notes__
    assert len(notes) == 1
    assert "ending the Fireworks session also failed" in notes[0]
    assert "closing a sampler" in notes[0]
    assert fw.closed


def test_no_capacity_for_a_session_starts_nothing(fw: FakeFireworksTrainingClient) -> None:
    fw.fail("service", "capacity")
    with (
        pytest.raises(ProviderCapacityError),
        backend_for(fw).open_session(PROFILE, LORA, **SESSION),
    ):
        pytest.fail("the body must not run")
    assert fw.sessions == {}
    assert fw.closed


def test_no_capacity_for_a_run_leaves_nothing_running(fw: FakeFireworksTrainingClient) -> None:
    fw.fail("create_lora_training_client", "capacity")
    with (
        pytest.raises(ProviderCapacityError) as caught,
        backend_for(fw).open_session(PROFILE, LORA, **SESSION),
    ):
        pytest.fail("the body must not run")
    assert caught.value.hint == CAPACITY_RELEASED_HINT
    assert len(fw.sessions) == 1
    assert _live(fw) == []


def test_a_failed_start_closes_the_session(fw: FakeFireworksTrainingClient) -> None:
    fw.fail("create_lora_training_client", "bad_request")
    with pytest.raises(ExternalError), backend_for(fw).open_session(PROFILE, LORA, **SESSION):
        pytest.fail("the body must not run")
    assert _live(fw) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"max_seq_len": 0}, "max_seq_len must be 1 to"),
        ({"max_seq_len": PROFILE.max_seq_len + 1}, "max_seq_len must be 1 to"),
        ({"seed": -1}, "seed must not be negative"),
    ],
)
def test_bad_session_settings_are_refused_before_anything_starts(
    fw: FakeFireworksTrainingClient, change: dict[str, Any], message: str
) -> None:
    with (
        pytest.raises(UsageError, match=message),
        backend_for(fw).open_session(PROFILE, LORA, **(SESSION | change)),
    ):
        pass
    assert fw.calls == []


def test_another_providers_profile_is_refused(fw: FakeFireworksTrainingClient) -> None:
    from pup_finetune.providers.fake import FAKE_PROFILE

    with (
        pytest.raises(UsageError, match="cannot train"),
        backend_for(fw).open_session(FAKE_PROFILE, LORA, seed=1, max_seq_len=16),
    ):
        pass
    assert fw.calls == []


def test_the_max_seq_len_option_caps_the_session(fw: FakeFireworksTrainingClient) -> None:
    backend = backend_for(fw, max_seq_len=64)
    with backend.open_session(PROFILE, LORA, seed=1, max_seq_len=64):
        pass
    with (
        pytest.raises(UsageError, match="over the max_seq_len provider option of 64") as caught,
        backend.open_session(PROFILE, LORA, seed=1, max_seq_len=65),
    ):
        pass
    assert caught.value.hint is not None and "--max-seq-len" in caught.value.hint
    assert len(fw.sessions) == 1


def test_an_empty_batch_is_refused(fw: FakeFireworksTrainingClient) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        with pytest.raises(ValueError, match="at least one example"):
            session.forward_backward([])
        with pytest.raises(ValueError, match="at least one example"):
            session.forward([])


def test_an_example_over_the_session_length_is_refused(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    longest = max(example.num_tokens for example in batch)
    with backend_for(fw).open_session(PROFILE, LORA, seed=1, max_seq_len=longest - 1) as session:
        with pytest.raises(UsageError, match="tokens, over this session's"):
            session.forward_backward(batch)
        with pytest.raises(UsageError, match="tokens, over this session's"):
            session.forward(batch)
    assert not any(call[0] == "forward_backward" for call in fw.calls)


# --- checkpoints -------------------------------------------------------------------------


def test_checkpoint_names_are_checked_before_saving(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    too_long = "n" * (MAX_CHECKPOINT_NAME_LEN + 1)
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        for save in (session.save_training_checkpoint, session.save_sampler_checkpoint):
            with pytest.raises(ValueError, match="1 to 17 characters"):
                save(too_long, epoch=1, step=1)
            with pytest.raises(ValueError, match="1 to 17 characters"):
                save("", epoch=1, step=1)
    assert fw.checkpoints == []


def test_checkpoints_carry_fireworks_paths(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        session.forward_backward(batch).result()
        session.optim_step(ADAM).result()
        training = session.save_training_checkpoint("t1s1-ab12", epoch=1, step=1)
        sampler = session.save_sampler_checkpoint("s1s1-ab12", epoch=1, step=1)
        rows = session.list_checkpoints()
    run_id = session.run_id
    assert training.uri == f"{ACCOUNT}/{run_id}/t1s1-ab12"
    assert training.kind == "training"
    assert sampler.kind == "sampler"
    assert sampler.uri.startswith(f"{ACCOUNT}/{run_id}/s1s1-ab12-")  # the server's suffix
    assert {training.created_in_session, sampler.created_in_session} == {session.session_id}
    assert [row.checkpoint_type for row in rows] == [TRAINING_CHECKPOINT, SAMPLER_CHECKPOINT]


def test_resume_continues_from_the_training_checkpoint_with_its_optimizer(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    backend = backend_for(fw)
    with backend.open_session(PROFILE, LORA, **SESSION) as session:
        session.forward_backward(batch).result()
        session.optim_step(ADAM).result()
        ckpt = session.save_training_checkpoint("t1s1-ab12", epoch=1, step=1)
    with backend.open_session(PROFILE, LORA, resume_from=ckpt, **SESSION) as resumed:
        assert resumed.session_id != session.session_id
        metrics = resumed.forward_backward(batch).result()
    assert ("create_training_client_from_state_with_optimizer", ckpt.uri) in fw.calls
    assert metrics.mean_loss == pytest.approx(2.0 / 1.25)  # the weights after one step
    assert [call[0] for call in fw.calls].count("create_lora_training_client") == 1


def test_resuming_from_a_sampler_checkpoint_is_refused(fw: FakeFireworksTrainingClient) -> None:
    ckpt = CheckpointRef(
        kind="sampler",
        name="s1",
        uri="a/run-1/s1-00000000",
        epoch=1,
        step=1,
        created_in_session="x",
    )
    with (
        pytest.raises(UsageError, match="training checkpoint"),
        backend_for(fw).open_session(PROFILE, LORA, resume_from=ckpt, **SESSION),
    ):
        pass
    assert fw.calls == []


def test_weights_load_from_a_training_checkpoint_only(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        start = session.save_training_checkpoint("t0s0-ab12", epoch=0, step=0)
        session.forward_backward(batch).result()
        session.optim_step(ADAM).result()
        sampler = session.save_sampler_checkpoint("s1s1-ab12", epoch=1, step=1)
        session.load_weights(start)
        assert session.forward(batch).result().mean_loss == pytest.approx(2.0)
        with pytest.raises(UsageError, match="training checkpoint"):
            session.load_weights(sampler)


# --- promotion ---------------------------------------------------------------------------


def _row(name: str, kind: str, *, promotable: bool, second: int) -> CheckpointRow:
    return CheckpointRow(
        checkpoint_name=name,
        checkpoint_type=kind,
        promotable=promotable,
        create_time=f"2026-01-01T00:00:{second:02d}Z",
        resource_name=f"accounts/{ACCOUNT}/trainingSessions/ts-1/checkpoints/{name}",
    )


def _sampler_ref(name: str, uri: str | None = None) -> CheckpointRef:
    return CheckpointRef(
        kind="sampler",
        name=name,
        uri=uri or f"{ACCOUNT}/run-abc/{name}",
        epoch=1,
        step=1,
        created_in_session="ts-1",
    )


def test_the_newest_promotable_sampler_row_with_the_name_wins() -> None:
    rows = [
        _row("run-abc-best-0000aaaa", SAMPLER_CHECKPOINT, promotable=True, second=3),
        _row("run-abc-best-0000cccc", SAMPLER_CHECKPOINT, promotable=True, second=9),
        _row("run-abc-best-0000bbbb", SAMPLER_CHECKPOINT, promotable=True, second=5),
        _row("run-abc-best", TRAINING_CHECKPOINT, promotable=True, second=20),
        _row("run-abc-best-0000dddd", SAMPLER_CHECKPOINT, promotable=False, second=30),
        _row("run-abc-best2-0000eeee", SAMPLER_CHECKPOINT, promotable=True, second=40),
        _row("run-abc-xbest-0000ffff", SAMPLER_CHECKPOINT, promotable=True, second=50),
    ]
    assert promotable_row(rows, _sampler_ref("best")).checkpoint_name == "run-abc-best-0000cccc"


def test_the_saved_path_pins_the_row_when_it_matches() -> None:
    rows = [
        _row("run-abc-best-0000aaaa", SAMPLER_CHECKPOINT, promotable=True, second=3),
        _row("run-abc-best-0000cccc", SAMPLER_CHECKPOINT, promotable=True, second=9),
    ]
    ckpt = _sampler_ref("best", uri=f"{ACCOUNT}/run-abc/best-0000aaaa")
    assert promotable_row(rows, ckpt).checkpoint_name == "run-abc-best-0000aaaa"


def test_no_sampler_row_with_the_name_is_an_error() -> None:
    rows = [
        _row("run-abc-best", TRAINING_CHECKPOINT, promotable=False, second=1),
        _row("run-abc-other-0000aaaa", SAMPLER_CHECKPOINT, promotable=True, second=2),
    ]
    with pytest.raises(ExternalError, match="lists no sampler checkpoint named 'best'"):
        promotable_row(rows, _sampler_ref("best"))


def test_a_row_that_is_not_promotable_is_an_error() -> None:
    rows = [_row("run-abc-best-0000aaaa", SAMPLER_CHECKPOINT, promotable=False, second=1)]
    with pytest.raises(ExternalError, match="not promotable") as caught:
        promotable_row(rows, _sampler_ref("best"))
    assert caught.value.hint


def test_promote_names_the_fireworks_model(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        session.forward_backward(batch).result()
        session.optim_step(ADAM).result()
        ckpt = session.save_sampler_checkpoint("s1s1-ab12", epoch=1, step=1)
        model = session.promote(ckpt, "ft-synthetic-run")
    assert model.provider == "fireworks"
    assert model.model_id == "ft-synthetic-run"
    assert model.uri == f"accounts/{ACCOUNT}/models/ft-synthetic-run"
    promoted = fw.promoted["ft-synthetic-run"]
    assert promoted["peftDetails"] == {"baseModel": PROFILE.base_model}


def _promotable(session: TrainingSession) -> CheckpointRef:
    return session.save_sampler_checkpoint("s1s1-ab12", epoch=1, step=1)


SINCE = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("kind", ["connection", "server", "conflict"])
def test_a_promotion_whose_response_was_lost_adopts_the_model(
    fw: FakeFireworksTrainingClient, kind: Any
) -> None:
    fw.lost_responses.append(kind)
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        model = session.promote(_promotable(session), "ft-synthetic-run")
    assert model.uri == f"accounts/{ACCOUNT}/models/ft-synthetic-run"
    promotes = [call for call in fw.calls if call[0] == "promote_session_checkpoint"]
    assert len(promotes) == 1


def test_a_failed_promotion_without_a_model_is_raised(fw: FakeFireworksTrainingClient) -> None:
    fw.fail("promote_session_checkpoint", "conflict")
    with (
        backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session,
        pytest.raises(ExternalError, match="HTTP 409"),
    ):
        session.promote(_promotable(session), "ft-synthetic-run")
    assert fw.promoted == {}


def test_a_failed_lookup_after_a_failed_promotion_raises_the_promotion_error(
    fw: FakeFireworksTrainingClient,
) -> None:
    fw.lost_responses.append("connection")
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        ckpt = _promotable(session)
        fw.fail("get_model", "server", after=1)
        with pytest.raises(TransientError, match="promoting"):
            session.promote(ckpt, "ft-synthetic-run")


def test_an_existing_model_of_this_run_is_adopted_without_promoting(
    fw: FakeFireworksTrainingClient,
) -> None:
    backend = backend_for(fw)
    fw.now = lambda: SINCE + timedelta(hours=1)
    with backend.open_session(PROFILE, LORA, **SESSION) as first:
        first.promote(_promotable(first), "ft-synthetic-run")
    with backend.open_session(PROFILE, LORA, **SESSION) as later:
        fw.calls.clear()
        model = later.promote(_promotable(later), "ft-synthetic-run", adopt_since=SINCE)
    assert model.uri == f"accounts/{ACCOUNT}/models/ft-synthetic-run"
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)


@pytest.mark.parametrize(
    ("created", "since", "match"),
    [
        (SINCE - timedelta(seconds=1), SINCE, "before this run first asked to promote it"),
        (SINCE, None, "already exists"),
        (None, SINCE, "does not say when it was created"),
    ],
)
def test_an_existing_model_that_may_be_another_run_s_is_refused(
    fw: FakeFireworksTrainingClient,
    created: datetime | None,
    since: datetime | None,
    match: str,
) -> None:
    name = f"accounts/{ACCOUNT}/models/ft-synthetic-run"
    record: dict[str, Any] = {"name": name, "peftDetails": {"baseModel": PROFILE.base_model}}
    if created is not None:
        record["createTime"] = created.isoformat()
    fw.promoted["ft-synthetic-run"] = record
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        ckpt = _promotable(session)
        with pytest.raises(StateError, match=match) as caught:
            session.promote(ckpt, "ft-synthetic-run", adopt_since=since)
    assert caught.value.hint
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)


def test_an_existing_model_of_another_base_model_is_refused(
    fw: FakeFireworksTrainingClient,
) -> None:
    name = f"accounts/{ACCOUNT}/models/ft-synthetic-run"
    fw.promoted["ft-synthetic-run"] = {
        "name": name,
        "peftDetails": {"baseModel": "accounts/fireworks/models/other-base"},
        "createTime": (SINCE + timedelta(hours=1)).isoformat(),
    }
    with (
        backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session,
        pytest.raises(StateError, match="other-base"),
    ):
        session.promote(_promotable(session), "ft-synthetic-run", adopt_since=SINCE)


class _NamelessPromotion(FakeFireworksTrainingClient):
    """Fireworks answering a promotion without the model's name."""

    def promote_session_checkpoint(
        self, resource_name: str, output_model_id: str, base_model: str
    ) -> dict[str, Any]:
        model = super().promote_session_checkpoint(resource_name, output_model_id, base_model)
        model.pop("name")
        return model


def test_a_promotion_answered_without_a_name_is_named_in_the_checkpoints_account() -> None:
    fw = _NamelessPromotion()
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        ckpt = session.save_sampler_checkpoint("s1s1-ab12", epoch=1, step=1)
        model = session.promote(ckpt, "ft-nameless")
    assert model.uri == f"accounts/{ACCOUNT}/models/ft-nameless"
    assert "ft-nameless" in fw.promoted


def test_a_nameless_promotion_of_an_unparseable_checkpoint_fails_clearly() -> None:
    from pup_finetune.providers.fireworks.backend import _model_name

    row = _row("run-abc-best-0000aaaa", SAMPLER_CHECKPOINT, promotable=True, second=1)
    assert _model_name(row, "ft-x") == f"accounts/{ACCOUNT}/models/ft-x"
    odd = CheckpointRow(
        checkpoint_name=row.checkpoint_name,
        checkpoint_type=row.checkpoint_type,
        promotable=True,
        create_time=row.create_time,
        resource_name="checkpoints/run-abc-best-0000aaaa",
    )
    with pytest.raises(ExternalError, match="names no account"):
        _model_name(odd, "ft-x")


@pytest.mark.parametrize("model_id", ["Upper", "ends-", "9lives", "a" * 64, "has_underscore"])
def test_an_invalid_model_id_is_refused_before_listing(
    fw: FakeFireworksTrainingClient, model_id: str
) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        ckpt = session.save_sampler_checkpoint("s0s0-ab12", epoch=0, step=0)
        with pytest.raises(UsageError, match="not a valid Fireworks model id"):
            session.promote(ckpt, model_id)
    assert not any(call[0] == "list_session_checkpoints" for call in fw.calls)


def test_promoting_after_the_session_ended_fails_clearly(fw: FakeFireworksTrainingClient) -> None:
    backend = backend_for(fw)
    with backend.open_session(PROFILE, LORA, **SESSION) as first:
        ckpt = first.save_sampler_checkpoint("s1s1-ab12", epoch=1, step=1)
    with backend.open_session(PROFILE, LORA, **SESSION) as later:
        with pytest.raises(ExternalError, match="has ended") as promoting:
            later.promote(ckpt, "ft-after-expiry")
        with pytest.raises(ExternalError, match="has ended") as sampling:
            later.sampler(ckpt)
    assert promoting.value.hint == sampling.value.hint == EXPIRED_HINT
    assert fw.promoted == {}


def test_a_training_checkpoint_cannot_be_promoted(fw: FakeFireworksTrainingClient) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        ckpt = session.save_training_checkpoint("t0s0-ab12", epoch=0, step=0)
        with pytest.raises(UsageError, match="sampler checkpoint"):
            session.promote(ckpt, "ft-training")


# --- sampling ----------------------------------------------------------------------------

PROMPT: list[dict[str, Any]] = [{"role": "user", "content": "Say hello."}]


def test_the_base_sampler_parses_the_completion(
    fw: FakeFireworksTrainingClient, renderer: Renderer
) -> None:
    with backend_for(fw, renderer=renderer).open_session(PROFILE, LORA, **SESSION) as session:
        sampler = session.sampler(None)
        generation = sampler.generate(PROMPT, [], max_tokens=64)
    assert generation.message["role"] == "assistant"
    assert generation.message.get("content") == "Hello there!"
    assert generation.format_valid
    assert generation.stop_reason == "stop"
    assert generation.prompt_tokens == len(renderer.generation_prompt_tokens(PROMPT, []))
    assert generation.completion_tokens == len(FakeChatTokenizer().encode(REPLY))
    assert ("create_sampling_client", PROFILE.base_model, None) in fw.calls
    service = fw.services[0]
    token_sampler = service.samplers[0]
    assert token_sampler.closed  # closed with the session
    assert token_sampler.prompts == [tuple(renderer.generation_prompt_tokens(PROMPT, []))]
    # Greedy, up to the renderer's stop tokens, which the SDK's decoder turns into strings.
    assert token_sampler.temperatures == [0.0]
    stop = list(token_sampler.stops[0])
    assert stop == renderer.stop_token_ids
    decoded = token_sampler.tokenizer.decode(stop[:1], skip_special_tokens=False)
    assert decoded == renderer.decode(stop[:1])


def test_the_sampler_parses_arguments_with_the_declared_tools(renderer: Renderer) -> None:
    # Qwen3.8 writes string values unquoted: only the tool's schema says "123" is a string.
    reply = (
        "</think>\n\n<tool_call>\n<function=lookup>\n<parameter=code>\n123\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    }
    fw = FakeFireworksTrainingClient(sample_tokens=FakeChatTokenizer().encode(reply))
    with backend_for(fw, renderer=renderer).open_session(PROFILE, LORA, **SESSION) as session:
        generation = session.sampler(None).generate(PROMPT, [tool], max_tokens=512)
    assert generation.format_valid
    [call] = generation.message["tool_calls"]
    assert json.loads(call["function"]["arguments"]) == {"code": "123", "limit": 5}


def test_latency_counts_only_the_successful_attempt(fw: FakeFireworksTrainingClient) -> None:
    clock = Clock(tick=1.0)  # each reading is a second later; sleeps add their own time
    fw.fail("sample", "server", times=SAMPLE_ATTEMPTS - 1)
    backend = backend_for(fw, sleep=clock.sleep, clock=clock)
    with backend.open_session(PROFILE, LORA, **SESSION) as session:
        generation = session.sampler(None).generate(PROMPT, [], max_tokens=64)
    assert clock.sleeps == [2.0, 4.0]
    assert generation.latency_s == pytest.approx(1.0)


def test_a_checkpoint_sampler_samples_the_saved_path(fw: FakeFireworksTrainingClient) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        ckpt = session.save_sampler_checkpoint("s0s0-ab12", epoch=0, step=0)
        sampler = session.sampler(ckpt)
        sampler.generate(PROMPT, [], max_tokens=64)
        sampler.close()
        sampler.close()
    assert ("create_sampling_client", None, ckpt.uri) in fw.calls


def test_a_truncated_completion_is_not_format_valid(fw: FakeFireworksTrainingClient) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        generation = session.sampler(None).generate(PROMPT, [], max_tokens=3)
    assert generation.stop_reason == "length"
    assert generation.completion_tokens == 3
    assert not generation.format_valid


def test_sampling_retries_a_transient_failure_with_backoff(
    fw: FakeFireworksTrainingClient, capsys: pytest.CaptureFixture[str]
) -> None:
    sleeps = Sleeps()
    fw.fail("sample", "server", times=SAMPLE_ATTEMPTS - 1)
    with backend_for(fw, sleep=sleeps).open_session(PROFILE, LORA, **SESSION) as session:
        generation = session.sampler(None).generate(PROMPT, [], max_tokens=64)
    assert generation.format_valid
    assert sleeps.calls == [2.0, 4.0]
    err = capsys.readouterr().err
    assert f"retrying in 2s (attempt 2 of {SAMPLE_ATTEMPTS})" in err
    assert f"retrying in 4s (attempt 3 of {SAMPLE_ATTEMPTS})" in err


def test_sampling_gives_up_after_a_bounded_number_of_attempts(
    fw: FakeFireworksTrainingClient,
) -> None:
    sleeps = Sleeps()
    fw.fail("sample", "connection", times=SAMPLE_ATTEMPTS)
    with backend_for(fw, sleep=sleeps).open_session(PROFILE, LORA, **SESSION) as session:
        sampler = session.sampler(None)
        with pytest.raises(TransientError) as caught:
            sampler.generate(PROMPT, [], max_tokens=64)
        assert f"sampling failed after {SAMPLE_ATTEMPTS} attempts" in caught.value.__notes__
        assert len(sleeps.calls) == SAMPLE_ATTEMPTS - 1
        assert sampler.generate(PROMPT, [], max_tokens=64).format_valid  # every failure used


@pytest.mark.parametrize("kind", ["bad_request", "capacity", "auth"])
def test_sampling_never_retries_a_lasting_failure(
    fw: FakeFireworksTrainingClient, kind: Any
) -> None:
    sleeps = Sleeps()
    fw.fail("sample", kind)
    with backend_for(fw, sleep=sleeps).open_session(PROFILE, LORA, **SESSION) as session:
        sampler = session.sampler(None)
        with pytest.raises(ExternalError):
            sampler.generate(PROMPT, [], max_tokens=64)
    assert sleeps.calls == []


def test_training_calls_are_never_retried(
    fw: FakeFireworksTrainingClient, batch: list[TokenizedExample]
) -> None:
    fw.fail("forward_backward", "server")
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        with pytest.raises(TransientError):
            session.forward_backward(batch).result()
        session.forward_backward(batch).result()  # the loop decides; the session does not retry


def test_a_closed_sampler_refuses_to_generate(fw: FakeFireworksTrainingClient) -> None:
    with backend_for(fw).open_session(PROFILE, LORA, **SESSION) as session:
        sampler = session.sampler(None)
        sampler.close()
        with pytest.raises(StateError, match="closed"):
            sampler.generate(PROMPT, [], max_tokens=8)
        with pytest.raises(ValueError, match="max_tokens"):
            session.sampler(None).generate(PROMPT, [], max_tokens=0)


# --- through the shared runner: promotion before the session ends ------------------------

RUN_ID = "ft-20261006123045-abcdef"


def _rows(renderer: Renderer, count: int, prefix: str) -> list[RowExamples]:
    return build_examples(
        chat_rows(count, prefix=prefix), renderer, max_seq_len=PROFILE.max_seq_len
    ).examples


def _provider(fw: FakeFireworksTrainingClient) -> FireworksProvider:
    return FireworksProvider(
        Runtime(),
        env={"FIREWORKS_API_KEY": API_KEY},
        transport=FakeFireworksApi().transport,
        tokenizer=FakeChatTokenizer(),
        training_client=lambda _key: fw,
    )


def _new_run(path: Path, renderer: Renderer, *, created_at: datetime | None = None) -> RunDir:
    spec = TrainSpec(max_epochs=2, batch_size=2, pipeline_depth=2, patience=2)
    clock = {} if created_at is None else {"now": lambda: created_at}
    return RunDir.create(
        path,
        provider="fireworks",
        profile=PROFILE,
        spec=spec,
        dataset_dir=path.parent / "dataset",
        dataset_sha256="d" * 64,
        rendering=renderer.identity(),
        run_id=RUN_ID,
        **clock,
    )


def test_a_run_promotes_its_best_epoch_before_the_session_ends(
    fw: FakeFireworksTrainingClient, renderer: Renderer, tmp_path: Path
) -> None:
    run_dir = _new_run(tmp_path / "run", renderer)
    result = start_training(
        _provider(fw),
        run_dir,
        train_rows=_rows(renderer, 4, "train"),
        validation_rows=_rows(renderer, 2, "val"),
    )
    assert result["phase"] == "completed"
    model_id = model_id_for_run(RUN_ID)
    assert result["promoted_model"] == {
        "provider": "fireworks",
        "model_id": model_id,
        "uri": f"accounts/{ACCOUNT}/models/{model_id}",
    }
    assert result["deployable_checkpoint"]["kind"] == "sampler"
    assert model_id in fw.promoted
    assert _live(fw) == []


class _RequestWatcher(FakeFireworksTrainingClient):
    """The in-memory Fireworks, noting the run's recorded promotion request at each promotion."""

    def __init__(self, run_dir: RunDir) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.recorded: list[str | None] = []

    def promote_session_checkpoint(
        self, resource_name: str, output_model_id: str, base_model: str
    ) -> dict[str, Any]:
        [request] = self.run_dir.load_state().promotions
        assert request["model_id"] == output_model_id
        self.recorded.append(request["requested_at"])
        return super().promote_session_checkpoint(resource_name, output_model_id, base_model)


def test_a_run_whose_promotion_response_was_lost_adopts_the_model_on_resume(
    renderer: Renderer, tmp_path: Path
) -> None:
    run_dir = _new_run(tmp_path / "run", renderer, created_at=SINCE - timedelta(hours=1))
    fw = _RequestWatcher(run_dir)
    train_rows, validation_rows = _rows(renderer, 4, "train"), _rows(renderer, 2, "val")
    # The promotion takes place but its response is lost, and so is the lookup after it.
    fw.lost_responses.append("connection")
    fw.fail("get_model", "connection", after=1)
    fw.now = lambda: SINCE + timedelta(seconds=2)
    with pytest.raises(FinalizeFailed):
        start_training(
            _provider(fw),
            run_dir,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=lambda: SINCE,
        )
    model_id = model_id_for_run(RUN_ID)
    assert model_id in fw.promoted
    assert (run_dir.read_result() or {})["promoted_model"] is None
    # The request's time was saved before the request was made.
    assert fw.recorded == [SINCE.isoformat()]

    # Every retry used to promote again and fail on the taken id; it now adopts the model,
    # created after the recorded request, however much later the retry runs.
    fw.calls.clear()
    result = resume_training(
        _provider(fw),
        run_dir,
        train_rows=train_rows,
        validation_rows=validation_rows,
        now=lambda: SINCE + timedelta(days=1),
    )
    assert result["phase"] == "completed"
    [request] = run_dir.load_state().promotions
    assert request["requested_at"] == SINCE.isoformat()
    assert result["promoted_model"]["uri"] == f"accounts/{ACCOUNT}/models/{model_id}"
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)
    assert _live(fw) == []


def test_a_new_best_after_a_cancelled_run_promoted_gets_its_own_model_and_adopts_it(
    renderer: Renderer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _new_run(tmp_path / "run", renderer, created_at=SINCE - timedelta(hours=1))
    fw = FakeFireworksTrainingClient()
    train_rows, validation_rows = _rows(renderer, 4, "train"), _rows(renderer, 2, "val")
    cancel = [False]

    def on_event(event: dict[str, Any]) -> None:
        cancel[0] = cancel[0] or event["type"] == "epoch"

    # Cancelled after epoch 1, which is promoted to the run's model id.
    monkeypatch.setattr(RunState, "cancel_requested", lambda self: cancel[0])
    fw.now = lambda: SINCE + timedelta(seconds=2)
    cancelled = start_training(
        _provider(fw),
        run_dir,
        train_rows=train_rows,
        validation_rows=validation_rows,
        on_event=on_event,
        now=lambda: SINCE,
    )
    monkeypatch.undo()
    first = model_id_for_run(RUN_ID)
    assert (cancelled["phase"], cancelled["best_epoch"]) == ("cancelled", 1)
    assert cancelled["promoted_model"]["model_id"] == first

    # The resume makes epoch 2 the best. The run's model id holds epoch 1 and was created
    # after the run first asked, so it would pass for this run's: epoch 2 gets its own id.
    # That promotion takes place but its response is lost, and so is the lookup after it.
    second = model_id_for_checkpoint(first, 2, 4)
    fw.lost_responses.append("connection")
    fw.fail("get_model", "connection", after=1)
    fw.now = lambda: SINCE + timedelta(hours=1, seconds=2)
    with pytest.raises(FinalizeFailed):
        resume_training(
            _provider(fw),
            run_dir,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=lambda: SINCE + timedelta(hours=1),
        )
    assert {first, second} <= set(fw.promoted)

    # The retry adopts epoch 2's model, not epoch 1's.
    fw.calls.clear()
    result = resume_training(
        _provider(fw),
        run_dir,
        train_rows=train_rows,
        validation_rows=validation_rows,
        now=lambda: SINCE + timedelta(days=1),
    )
    assert (result["phase"], result["best_epoch"]) == ("completed", 2)
    assert result["deployable_checkpoint"]["epoch"] == 2
    assert result["promoted_model"]["uri"] == f"accounts/{ACCOUNT}/models/{second}"
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)
    assert [(r["model_id"], r["epoch"]) for r in run_dir.load_state().promotions] == [
        (first, 1),
        (second, 2),
    ]
    assert _live(fw) == []


def test_a_model_another_run_directory_promoted_under_the_run_id_is_refused(
    fw: FakeFireworksTrainingClient, renderer: Renderer, tmp_path: Path
) -> None:
    train_rows, validation_rows = _rows(renderer, 4, "train"), _rows(renderer, 2, "val")
    # Run A starts first; run B, reusing its run id, promotes before A gets to promote.
    first = _new_run(tmp_path / "a" / "run", renderer, created_at=SINCE)
    second = _new_run(tmp_path / "b" / "run", renderer, created_at=SINCE + timedelta(minutes=5))
    fw.now = lambda: SINCE + timedelta(minutes=10)
    start_training(
        _provider(fw),
        second,
        train_rows=train_rows,
        validation_rows=validation_rows,
        now=fw.now,
    )
    fw.calls.clear()
    # B's model was created after A started, but well before A asked for a promotion.
    with pytest.raises(FinalizeFailed) as caught:
        start_training(
            _provider(fw),
            first,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=lambda: SINCE + timedelta(hours=1),
        )
    assert isinstance(caught.value.__cause__, StateError)
    assert "already exists" in str(caught.value.__cause__)
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)
    assert (first.read_result() or {})["promoted_model"] is None
    assert _live(fw) == []


def test_a_model_another_run_directory_promoted_just_before_under_the_run_id_is_refused(
    fw: FakeFireworksTrainingClient, renderer: Renderer, tmp_path: Path
) -> None:
    train_rows, validation_rows = _rows(renderer, 4, "train"), _rows(renderer, 2, "val")
    # A preview's pinned command run into two run directories: both finish 30 s apart.
    first = _new_run(tmp_path / "a" / "run", renderer, created_at=SINCE)
    second = _new_run(tmp_path / "b" / "run", renderer, created_at=SINCE)
    fw.now = lambda: SINCE + timedelta(minutes=10)
    start_training(
        _provider(fw), first, train_rows=train_rows, validation_rows=validation_rows, now=fw.now
    )
    fw.calls.clear()
    with pytest.raises(FinalizeFailed) as caught:
        start_training(
            _provider(fw),
            second,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=lambda: SINCE + timedelta(minutes=10, seconds=30),
        )
    assert isinstance(caught.value.__cause__, StateError)
    assert "already exists" in str(caught.value.__cause__)
    assert caught.value.hint is not None and "another run id" in caught.value.hint
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)
    assert (second.read_result() or {})["promoted_model"] is None
    assert _live(fw) == []

    # The resume the failed run offers still refuses that model, however recently it was made.
    with pytest.raises(FinalizeFailed) as again:
        resume_training(
            _provider(fw),
            second,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=lambda: SINCE + timedelta(minutes=11),
        )
    assert "already exists" in str(again.value.__cause__)
    assert (second.read_result() or {})["promoted_model"] is None

    # Once that model is deleted, the resume promotes this run's checkpoint under the id.
    taken = (first.read_result() or {})["promoted_model"]["model_id"]
    del fw.promoted[taken]
    fw.calls.clear()
    result = resume_training(
        _provider(fw),
        second,
        train_rows=train_rows,
        validation_rows=validation_rows,
        now=lambda: SINCE + timedelta(minutes=12),
    )
    assert result["promoted_model"]["model_id"] == taken
    assert any(call[0] == "promote_session_checkpoint" for call in fw.calls)
    assert _live(fw) == []


def test_a_promotion_lost_after_a_refused_one_is_adopted_on_resume(
    fw: FakeFireworksTrainingClient, renderer: Renderer, tmp_path: Path
) -> None:
    train_rows, validation_rows = _rows(renderer, 4, "train"), _rows(renderer, 2, "val")
    first = _new_run(tmp_path / "a" / "run", renderer, created_at=SINCE)
    second = _new_run(tmp_path / "b" / "run", renderer, created_at=SINCE)
    fw.now = lambda: SINCE + timedelta(minutes=10)
    start_training(
        _provider(fw), first, train_rows=train_rows, validation_rows=validation_rows, now=fw.now
    )
    with pytest.raises(FinalizeFailed):
        start_training(
            _provider(fw),
            second,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=lambda: SINCE + timedelta(minutes=10, seconds=30),
        )
    model_id = model_id_for_run(RUN_ID)
    del fw.promoted[model_id]
    # The resume's promotion takes place, but its response and the lookup after it are lost.
    fw.lost_responses.append("connection")
    fw.fail("get_model", "connection", after=1)
    fw.now = lambda: SINCE + timedelta(minutes=20)
    with pytest.raises(FinalizeFailed):
        resume_training(
            _provider(fw),
            second,
            train_rows=train_rows,
            validation_rows=validation_rows,
            now=fw.now,
        )
    assert model_id in fw.promoted

    fw.calls.clear()
    result = resume_training(
        _provider(fw),
        second,
        train_rows=train_rows,
        validation_rows=validation_rows,
        now=lambda: SINCE + timedelta(hours=1),
    )
    assert result["promoted_model"]["uri"] == f"accounts/{ACCOUNT}/models/{model_id}"
    assert not any(call[0] == "promote_session_checkpoint" for call in fw.calls)
    assert _live(fw) == []


def test_a_failed_promotion_is_retried_from_the_best_training_checkpoint(
    fw: FakeFireworksTrainingClient, renderer: Renderer, tmp_path: Path
) -> None:
    run_dir = _new_run(tmp_path / "run", renderer)
    train_rows, validation_rows = _rows(renderer, 4, "train"), _rows(renderer, 2, "val")
    fw.fail("promote_session_checkpoint", "server")
    with pytest.raises(FinalizeFailed):
        start_training(
            _provider(fw), run_dir, train_rows=train_rows, validation_rows=validation_rows
        )
    failed = run_dir.read_result() or {}
    assert failed["phase"] == "failed"
    assert failed["promoted_model"] is None
    assert _live(fw) == []
    first_session = failed["checkpoints"]["best_sampler"]["created_in_session"]

    # A new process: the first session (and its sampler checkpoints) has ended, so the
    # shared finalize re-creates the best epoch's sampler checkpoint from its training one.
    result = resume_training(
        _provider(fw), run_dir, train_rows=train_rows, validation_rows=validation_rows
    )
    assert result["phase"] == "completed"
    assert result["promoted_model"]["model_id"] == model_id_for_run(RUN_ID)
    deployable = result["deployable_checkpoint"]
    assert deployable["kind"] == "sampler"
    assert deployable["name"].startswith("r")
    assert deployable["created_in_session"] != first_session
    best_training = failed["checkpoints"]["best_training"]["uri"]
    assert ("create_training_client_from_state_with_optimizer", best_training) in fw.calls
    assert _live(fw) == []
