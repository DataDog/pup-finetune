# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Baseten backend, session and sampler, against the Loops fake: the backend contract,
then trainer sizing, datums and losses, Adam, errors, deactivation on every exit, and the
sampler lifecycle, retries and concurrency. Rendering is ``qwen3_5`` on the fake tokenizer."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.evaluation.targets import session_samplers
from pup_finetune.providers.base import (
    AdamConfig,
    CheckpointRef,
    LoraConfig,
    ModelProfile,
    ProviderAuthError,
    ProviderCapacityError,
    TransientError,
)
from pup_finetune.providers.baseten import client
from pup_finetune.providers.baseten.backend import (
    SAMPLE_ATTEMPTS,
    BasetenBackend,
    fitted_seq_len,
)
from pup_finetune.providers.baseten.client import DeploymentRejected, to_loops_datum
from pup_finetune.rendering.families import QWEN3_5
from pup_finetune.rendering.hf import HFChatTemplateRenderer
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.training.records import TrainResult
from tests.fakes.baseten import (
    Answer,
    FakeAdamParams,
    FakeForwardBackwardOutput,
    FakeLoops,
    FakeRemoteOpError,
    FakeSamplingParams,
    FakeServerShutdownError,
    FakeTensorData,
    http_status_error,
    loss_at,
)
from tests.fakes.tokenizer import FakeChatTokenizer
from tests.providers.baseten.test_baseten_client import AUTH_HINT
from tests.providers.contract import BackendContract

KEY = SecretStr("synthetic-baseten-key")
PROFILE = ModelProfile(
    provider="baseten",
    alias="qwen3p5-9b",
    base_model="Qwen/Qwen3.5-9B",
    tokenizer_model=QWEN3_5.tokenizer_model,
    tokenizer_revision=QWEN3_5.revision,
    renderer=QWEN3_5.key,
    max_seq_len=131072,
    supports_reasoning=True,
)
LORA = LoraConfig(rank=8)
BATCH = [
    TokenizedExample((5, 6, 7), (6, 7, 8), (0.0, 1.0, 0.5)),
    TokenizedExample((9, 10), (10, 11), (1.0, 1.0)),
]
WEIGHT = 3.5
ANSWER = "</think>\n\nHello there!<|im_end|>"
HI = [{"role": "user", "content": "Hi"}]
SAMPLER_CKPT = CheckpointRef(
    kind="sampler",
    name="s1",
    uri="bt://loops:run9/sampler_weights/s1",
    epoch=1,
    step=1,
    created_in_session="run9",
)
FOREIGN_CKPT = dataclasses.replace(SAMPLER_CKPT, kind="training", uri="fake://checkpoints/t1")
RUN_DEACTIVATE = "POST /v1/loops/runs/run1/deactivate"
RUN_LEFT = "a run may remain in Loops session session1; deactivate it in the Baseten UI"
SAMPLER_LEFT = "a sampler may remain in Loops session session1; deactivate it in the Baseten UI"
SAMPLER_DEACTIVATE = "POST /v1/models/model1/deployments/dep1/deactivate"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (*client._REJECTED_ENV, "LOOPS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    yield
    redaction.clear_secrets()


def fake_renderer() -> HFChatTemplateRenderer:
    return QWEN3_5.renderer(tokenizer=FakeChatTokenizer(merge_newlines=True))


def fake_loops(answer: str = ANSWER) -> FakeLoops:
    return FakeLoops(completion=FakeChatTokenizer(merge_newlines=True).encode(answer))


@pytest.fixture
def renderer() -> HFChatTemplateRenderer:
    return fake_renderer()


@pytest.fixture
def loops() -> FakeLoops:
    return fake_loops()


class Setup:
    """A backend on ``loops``, with ``routes`` answered before the fake's own."""

    def __init__(
        self,
        loops: FakeLoops,
        renderer: HFChatTemplateRenderer,
        routes: dict[str, Answer | list[Answer]] | None = None,
        sleep: Callable[[float], None] | None = None,
        max_seq_len: int | None = None,
        profile: ModelProfile = PROFILE,
    ) -> None:
        self.loops = loops
        self.profile = profile
        self.transport = loops.transport(routes)
        self.sleeps: list[float] = []
        self.keys = 0
        self.backend = BasetenBackend(
            self.key,
            renderer=lambda _: renderer,
            max_seq_len=max_seq_len,
            loops=loops,
            transport=self.transport,
            sleep=sleep or self.sleeps.append,
        )

    def key(self) -> SecretStr:
        self.keys += 1
        return KEY

    def session(self, **settings: Any) -> Any:
        return self.backend.open_session(
            self.profile, settings.pop("lora", LORA), **{"seed": 7, "max_seq_len": 900, **settings}
        )


@pytest.fixture
def setup(loops: FakeLoops, renderer: HFChatTemplateRenderer) -> Setup:
    return Setup(loops, renderer)


# --- the contract ------------------------------------------------------------------------


class TestBasetenBackendContract(BackendContract):
    @pytest.fixture
    def renderer(self) -> HFChatTemplateRenderer:
        return fake_renderer()

    @pytest.fixture
    def profile(self) -> ModelProfile:
        return PROFILE

    @pytest.fixture
    def backend(self, renderer: HFChatTemplateRenderer) -> BasetenBackend:
        loops = fake_loops()
        return BasetenBackend(
            lambda: KEY,
            renderer=lambda _: renderer,
            loops=loops,
            transport=loops.transport(),
            sleep=lambda _: None,
        )


# --- open_session --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("longest", "limit", "fitted"),
    [(1, 131072, 1024), (1024, 131072, 1024), (1025, 131072, 2048), (5000, 6000, 6000)],
)
def test_the_trainer_context_is_a_power_of_two_of_at_least_1024(
    longest: int, limit: int, fitted: int
) -> None:
    assert fitted_seq_len(longest, limit) == fitted


def test_a_session_is_one_dedicated_trainer_named_for_pup_finetune(setup: Setup) -> None:
    assert setup.keys == 0
    with setup.session(max_seq_len=3000) as session:
        assert (session.session_id, session.loops_session_id) == ("run1", "session1")
        assert session.max_seq_len == 4096
    assert setup.keys == 1
    (settings,) = setup.loops.kwargs("create_lora_training_client")
    name = settings.pop("name")
    assert name.startswith("pup-finetune-")
    assert settings == {
        "base_model": PROFILE.base_model,
        "rank": 8,
        "replicas": 1,
        "seed": 7,
        "max_seq_len": 4096,
        "with_sampler": False,
    }


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"max_seq_len": 0}, "max_seq_len must be 1 to 131072"),
        ({"max_seq_len": 131073}, "max_seq_len must be 1 to 131072"),
        ({"max_seq_len": 4097}, "over the max_seq_len provider option of 4096"),
        ({"seed": -1}, "seed must not be negative"),
        ({"lora": LoraConfig(rank=8, alpha=16)}, "LoRA alpha"),
        ({"resume_from": SAMPLER_CKPT}, "resume from a training checkpoint"),
        ({"resume_from": FOREIGN_CKPT}, "not a Baseten Loops checkpoint"),
    ],
)
def test_bad_session_settings_are_refused_before_anything_is_paid(
    loops: FakeLoops, renderer: HFChatTemplateRenderer, settings: dict[str, Any], message: str
) -> None:
    setup = Setup(loops, renderer, max_seq_len=4096)
    with pytest.raises(UsageError, match=message), setup.session(**settings):
        pass
    assert setup.loops.calls == []
    assert setup.keys == 0


def test_another_providers_profile_is_refused(setup: Setup) -> None:
    other = dataclasses.replace(PROFILE, provider="fake")
    with (
        pytest.raises(UsageError, match="cannot train fake/"),
        setup.backend.open_session(other, LORA, seed=7, max_seq_len=900),
    ):
        pass
    assert setup.loops.calls == []


def test_a_trainer_baseten_has_no_capacity_for_leaves_nothing_running(setup: Setup) -> None:
    setup.loops.fail_on(
        "create_lora_training_client",
        error=http_status_error(429, headers={"Retry-After": "60"}),
    )
    with pytest.raises(ProviderCapacityError) as caught, setup.session():
        pass
    assert caught.value.hint == "nothing was left running; retry after 60 s"
    assert setup.loops.runs == {}
    assert setup.transport.calls == []


def test_a_trainer_that_never_becomes_ready_says_its_run_may_remain(setup: Setup) -> None:
    setup.loops.lost["trainer"] = TimeoutError("the trainer did not become ready")
    with pytest.raises(ExternalError, match="TimeoutError") as caught, setup.session():
        pass
    assert caught.value.hint == RUN_LEFT


def test_the_renderer_is_built_once_per_profile(loops: FakeLoops) -> None:
    built: list[str] = []

    def renderer(profile: ModelProfile) -> HFChatTemplateRenderer:
        built.append(profile.alias)
        return fake_renderer()

    backend = BasetenBackend(
        lambda: KEY, renderer=renderer, loops=loops, transport=loops.transport()
    )
    with backend.open_session(PROFILE, LORA, seed=7, max_seq_len=900) as session:
        session.sampler(None).close()
        session.sampler(None).close()
    assert built == [PROFILE.alias]


# --- deactivation ----------------------------------------------------------------------


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_every_exit_closes_the_trainer_and_deactivates_the_run_and_samplers(
    setup: Setup, error: BaseException
) -> None:
    with pytest.raises(type(error)), setup.session() as session:
        session.sampler(None)
        raise error
    loops = setup.loops
    assert loops.trainers[0].closed
    assert not loops.runs["run1"].active
    assert not loops.samplers["sampler1"].active
    assert loops.sampling_clients[0].closed


def test_an_interrupted_cleanup_still_deactivates_the_run(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt  # Ctrl-C while a deactivation is awaited

    setup = Setup(loops, renderer, sleep=interrupted)
    with pytest.raises(KeyboardInterrupt), setup.session() as session:
        session.sampler(None)
    assert setup.transport.calls.count(SAMPLER_DEACTIVATE) == 1
    assert setup.transport.calls.count(RUN_DEACTIVATE) == 1
    assert setup.loops.trainers[0].closed
    assert not setup.loops.runs["run1"].active


@pytest.mark.parametrize("clients", ["trainers", "sampling_clients"])
def test_an_interrupted_client_close_still_deactivates(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, clients: str
) -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt  # Ctrl-C while the SDK client closes

    with pytest.raises(KeyboardInterrupt), setup.session() as session:
        session.sampler(None)
        monkeypatch.setattr(getattr(setup.loops, clients)[0], "close", interrupted)
    assert not setup.loops.samplers["sampler1"].active
    assert not setup.loops.runs["run1"].active


def test_an_interrupted_close_wins_over_a_failed_deactivation_and_names_it(
    loops: FakeLoops, renderer: HFChatTemplateRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt  # Ctrl-C while the SDK client closes

    setup = Setup(loops, renderer, {SAMPLER_DEACTIVATE: (500, {})})
    with pytest.raises(KeyboardInterrupt) as caught, setup.session() as session:
        session.sampler(None)
        monkeypatch.setattr(loops.trainers[0], "close", interrupted)
    [note] = caught.value.__notes__
    assert "could not deactivate the Baseten sampler deployment model1/dep1" in note
    assert not loops.runs["run1"].active


def test_a_failed_deactivation_names_the_manual_step(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    setup = Setup(loops, renderer, {RUN_DEACTIVATE: (500, {})})
    with pytest.raises(ExternalError) as caught, setup.session():
        pass
    url = "https://api.baseten.co/v1/loops/runs/run1/deactivate"
    assert "could not deactivate Baseten run run1" in caught.value.message
    assert f"POST {url}" in caught.value.message
    assert caught.value.hint == f'curl -X POST {url} -H "Authorization: Bearer $BASETEN_API_KEY"'
    assert setup.transport.calls.count(RUN_DEACTIVATE) == 3
    assert setup.sleeps == [1.0, 2.0]


def test_a_failed_deactivation_never_hides_the_body_error(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    setup = Setup(loops, renderer, {RUN_DEACTIVATE: (500, {})})
    with pytest.raises(RuntimeError, match="boom") as caught, setup.session():
        raise RuntimeError("boom")
    [note] = caught.value.__notes__
    assert "could not deactivate Baseten run run1" in note


def test_every_failed_deactivation_is_reported(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    routes: dict[str, Answer | list[Answer]] = {
        RUN_DEACTIVATE: (500, {}),
        SAMPLER_DEACTIVATE: (500, {}),
    }
    setup = Setup(loops, renderer, routes)
    with pytest.raises(ExternalError) as caught, setup.session() as session:
        session.sampler(None)
    assert "the Baseten sampler deployment model1/dep1" in caught.value.message
    assert "/v1/models/model1/deployments/dep1/deactivate" in (caught.value.hint or "")
    [note] = caught.value.__notes__
    assert "could not deactivate Baseten run run1" in note


def test_deactivation_waits_until_the_run_is_inactive(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    active: Answer = (200, {"run": {"id": "run1", "status": {"name": "ACTIVE"}}})
    inactive: Answer = (200, {"run": {"id": "run1", "status": {"name": "INACTIVE"}}})
    setup = Setup(loops, renderer, {"GET /v1/loops/runs/run1": [active, active, active, inactive]})
    with setup.session():
        pass
    assert setup.transport.calls.count(RUN_DEACTIVATE) == 1
    assert setup.sleeps == [5.0, 5.0, 5.0]


def test_a_deactivation_in_progress_is_awaited_not_repeated(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    deactivating: Answer = (200, {"id": "dep1", "model_id": "model1", "status": "DEACTIVATING"})
    inactive: Answer = (200, {"id": "dep1", "model_id": "model1", "status": "INACTIVE"})
    deployment = "GET /v1/models/model1/deployments/dep1"
    setup = Setup(loops, renderer, {deployment: [deactivating, inactive]})
    with setup.session() as session:
        session.sampler(None).close()
    assert SAMPLER_DEACTIVATE not in setup.transport.calls
    assert setup.transport.calls.count(deployment) == 2


def test_a_run_already_inactive_is_not_deactivated_again(setup: Setup) -> None:
    with setup.session():
        setup.loops.deactivate("run1")
    assert RUN_DEACTIVATE not in setup.transport.calls


# --- steps ---------------------------------------------------------------------------


def test_a_step_sends_each_example_as_a_datum(setup: Setup) -> None:
    with setup.session() as session:
        session.forward_backward(BATCH).result()
    (call,) = setup.loops.kwargs("forward_backward")
    assert call == {
        "data": [to_loops_datum(e, setup.loops) for e in BATCH],
        "loss_fn": "cross_entropy",
    }
    assert call["data"][0].loss_fn_inputs["target_tokens"].data == [-100, 7, 8]


@pytest.mark.parametrize("call", ["forward", "forward_backward"])
def test_an_empty_or_overlong_batch_never_reaches_loops(setup: Setup, call: str) -> None:
    overlong = TokenizedExample(tuple(range(1024)), tuple(range(1024)), (1.0,) * 1024)
    with setup.session() as session:
        with pytest.raises(ValueError, match="at least one example"):
            getattr(session, call)([])
        with pytest.raises(UsageError, match="1025 tokens, over this session's 1024-token"):
            getattr(session, call)([*BATCH, overlong])
    assert setup.loops.kwargs(call) == []


@pytest.mark.parametrize("logprobs", [True, False], ids=["logprobs", "mean"])
def test_step_metrics_sum_the_weighted_loss(
    renderer: HFChatTemplateRenderer, logprobs: bool
) -> None:
    setup = Setup(FakeLoops(logprobs=logprobs), renderer)
    with setup.session() as session:
        metrics = session.forward(BATCH).result()
    assert metrics.loss_sum == pytest.approx(loss_at(0) * WEIGHT)
    assert metrics.weight == WEIGHT
    assert metrics.tokens == 4 + 3


def _logprobs(*values: float | None) -> dict[str, FakeTensorData]:
    return {"logprobs": FakeTensorData(data=list(values), dtype="float32", shape=[len(values)])}


def test_the_loss_sums_the_weighted_logprobs_over_trained_tokens(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = FakeForwardBackwardOutput(
        loss=99.0, loss_fn_outputs=[_logprobs(None, -1.0, -2.0), _logprobs(-0.5, -0.25)]
    )
    with setup.session() as session:
        monkeypatch.setattr(setup.loops.trainers[0], "_output", lambda data: output)
        metrics = session.forward(BATCH).result()
    assert metrics.loss_sum == pytest.approx(1.0 + 2.0 * 0.5 + 0.5 + 0.25)


def test_outputs_without_logprobs_fall_back_to_the_mean_loss(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = FakeForwardBackwardOutput(loss_fn_outputs=[{}, {}], metrics={"loss": 2.0})
    with setup.session() as session:
        monkeypatch.setattr(setup.loops.trainers[0], "_output", lambda data: output)
        metrics = session.forward_backward(BATCH).result()
    assert metrics.loss_sum == pytest.approx(2.0 * WEIGHT)


@pytest.mark.parametrize(
    "output",
    [
        FakeForwardBackwardOutput(loss_fn_outputs=[_logprobs(-1.0, -1.0, -1.0)]),
        FakeForwardBackwardOutput(loss_fn_outputs=[_logprobs(-1.0), _logprobs(-1.0, -1.0)]),
        FakeForwardBackwardOutput(metrics={"loss": "high"}),  # type: ignore[dict-item]
    ],
    ids=["too-few-outputs", "too-few-logprobs", "loss-not-a-number"],
)
def test_a_malformed_step_result_is_an_external_error(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, output: FakeForwardBackwardOutput
) -> None:
    with setup.session() as session:
        monkeypatch.setattr(setup.loops.trainers[0], "_output", lambda data: output)
        with pytest.raises(ExternalError, match="malformed step result"):
            session.forward(BATCH).result()


def test_optim_step_passes_every_adam_parameter(setup: Setup) -> None:
    adam = AdamConfig(
        learning_rate=3e-4, beta1=0.8, beta2=0.99, eps=1e-8, weight_decay=0.01, grad_clip_norm=0
    )
    with setup.session() as session:
        assert session.optim_step(adam).result() is None
    assert setup.loops.kwargs("optim_step") == [
        {"adam_params": FakeAdamParams(3e-4, 0.8, 0.99, 1e-8, 0.01, 0.0)}
    ]


def test_each_optim_step_sends_its_own_learning_rate(setup: Setup) -> None:
    with setup.session() as session:
        for rate in (5e-5, 1e-4):
            session.optim_step(AdamConfig(learning_rate=rate)).result()
    sent = [kwargs["adam_params"].learning_rate for kwargs in setup.loops.kwargs("optim_step")]
    assert sent == [5e-5, 1e-4]


@pytest.mark.parametrize(
    ("call", "error", "on_result", "transient"),
    [
        ("forward", http_status_error(503), False, True),
        ("forward", FakeServerShutdownError("going away"), True, True),
        ("forward", httpx.ConnectError("reset"), True, True),
        ("forward", FakeRemoteOpError("bad batch"), True, False),
        ("forward", http_status_error(400), False, False),
        ("forward_backward", http_status_error(503), False, False),
        ("forward_backward", FakeServerShutdownError("going away"), True, False),
    ],
)
def test_only_forward_fails_as_transient(
    setup: Setup, call: str, error: Exception, on_result: bool, transient: bool
) -> None:
    setup.loops.fail_on(call, error=error, on_result=on_result)
    with setup.session() as session, pytest.raises(ExternalError) as caught:
        getattr(session, call)(BATCH).result()
    assert isinstance(caught.value, TransientError) is transient
    assert caught.value.message.startswith(f"the Loops {call} failed (")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, ProviderAuthError), (403, ProviderAuthError), (429, ProviderCapacityError)],
)
def test_a_refused_key_or_capacity_mid_session_keeps_its_class(
    setup: Setup, status: int, kind: type[ExternalError]
) -> None:
    setup.loops.fail_on("forward", error=http_status_error(status), on_result=True)
    with setup.session() as session, pytest.raises(kind):
        session.forward(BATCH).result()


def test_calls_after_the_session_ended_are_refused(setup: Setup) -> None:
    with setup.session() as session:
        pass
    with pytest.raises(StateError, match="session run1 has ended"):
        session.forward(BATCH)


# --- checkpoints ---------------------------------------------------------------------


def test_checkpoints_are_bt_paths_under_the_run(setup: Setup) -> None:
    with setup.session() as session:
        training = session.save_training_checkpoint("t1s2-ab", epoch=1, step=2)
        sampler = session.save_sampler_checkpoint("s1s2-ab", epoch=1, step=2)
    assert training.uri == "bt://loops:run1/state/t1s2-ab"
    assert sampler.uri == "bt://loops:run1/sampler_weights/s1s2-ab"
    assert (sampler.kind, sampler.created_in_session) == ("sampler", "run1")


@pytest.mark.parametrize("name", ["", "c" * 55, "a/b", "a b"])
def test_invalid_checkpoint_names_never_reach_loops(setup: Setup, name: str) -> None:
    with setup.session() as session, pytest.raises(ValueError, match="checkpoint"):
        session.save_sampler_checkpoint(name, epoch=0, step=0)
    assert setup.loops.kwargs("save_weights_for_sampler") == []


def test_resume_loads_weights_and_optimizer_state(setup: Setup) -> None:
    with setup.session() as session:
        session.forward_backward(BATCH).result()
        session.optim_step(AdamConfig(learning_rate=1e-4)).result()
        ckpt = session.save_training_checkpoint("t1", epoch=1, step=1)
    with setup.session(resume_from=ckpt) as resumed:
        assert resumed.session_id == "run2"
        assert resumed.forward(BATCH).result().mean_loss == pytest.approx(loss_at(1))
    assert setup.loops.kwargs("load_state_with_optimizer") == [{"path": ckpt.uri}]


def test_an_expired_checkpoint_fails_with_a_hint_and_deactivates_the_run(setup: Setup) -> None:
    with setup.session() as session:
        ckpt = session.save_training_checkpoint("t1", epoch=1, step=1)
    setup.loops.expire(ckpt.uri)
    with pytest.raises(ExternalError) as caught, setup.session(resume_from=ckpt):
        pass
    assert not isinstance(caught.value, UsageError)
    assert "expired" in (caught.value.hint or "")
    assert not setup.loops.runs["run2"].active


def test_weights_load_only_from_a_training_checkpoint(setup: Setup) -> None:
    with setup.session() as session:
        sampler = session.save_sampler_checkpoint("s1", epoch=1, step=1)
        training = session.save_training_checkpoint("t1", epoch=1, step=1)
        with pytest.raises(UsageError, match="training checkpoint"):
            session.load_weights(sampler)
        session.load_weights(training)
    assert setup.loops.kwargs("load_state") == [{"path": training.uri}]


def test_there_is_no_promotion(setup: Setup) -> None:
    with setup.session() as session:
        ckpt = session.save_sampler_checkpoint("s1", epoch=1, step=1)
        with pytest.raises(UsageError, match="no promotion; the sampler checkpoint is deployed"):
            session.promote(ckpt, "pft-model", adopt_since=None)


# --- samplers ------------------------------------------------------------------------


def test_the_base_sampler_serves_the_base_model_with_the_profile_context(
    setup: Setup, renderer: HFChatTemplateRenderer
) -> None:
    messages = [{"role": "user", "content": "Say hello."}]
    with setup.session() as session:
        assert session.max_seq_len == 1024
        sampler = session.sampler(None)
        generation = sampler.generate(messages, [], max_tokens=64)
    assert setup.transport.bodies("POST /v1/loops/samplers") == [
        {
            "session_id": "session1",
            "base_model": PROFILE.base_model,
            "max_seq_length": PROFILE.max_seq_len,
            "availability_model": "dedicated",
        }
    ]
    (connect,) = setup.loops.kwargs("SamplingClient")
    assert connect["checkpoint_path"] is None
    assert (connect["sampler_id"], connect["session_id"]) == ("sampler1", "session1")
    assert (connect["ready_timeout"], connect["timeout"]) == (600.0, 600.0)
    assert (
        connect["deployment"].base_url == "https://model-model1.api.baseten.co/deployment/dep1/sync"
    )
    (sample,) = setup.loops.kwargs("sample")
    prompt = renderer.generation_prompt_tokens(messages, [])
    assert sample["prompt"].tokens == prompt
    assert sample["sampling_params"] == FakeSamplingParams(
        max_tokens=64, temperature=0.0, stop=renderer.stop_token_ids
    )
    assert generation.message == {"role": "assistant", "content": "Hello there!"}
    assert generation.format_valid
    assert (generation.prompt_tokens, generation.stop_reason) == (len(prompt), "stop")
    assert generation.completion_tokens == len(setup.loops.completion)


def test_a_checkpoint_sampler_loads_the_checkpoint_with_one_sample(setup: Setup) -> None:
    with setup.session() as session:
        ckpt = session.save_sampler_checkpoint("s1", epoch=1, step=1)
        session.sampler(ckpt)
        (body,) = setup.transport.bodies("POST /v1/loops/samplers")
        assert body["model_path"] == ckpt.uri
        assert setup.loops.kwargs("SamplingClient")[0]["checkpoint_path"] == ckpt.uri
        (load,) = setup.loops.kwargs("sample")
        assert load["sampling_params"].max_tokens == 1


def test_a_tuned_only_evaluation_deploys_no_base_sampler(setup: Setup) -> None:
    with setup.session() as session:
        training = session.save_training_checkpoint("t1", epoch=1, step=1)
        result = TrainResult(
            best_epoch=1,
            best_val_loss=1.0,
            stopped_reason="completed",
            epochs=[],
            best_sampler_ckpt=None,
            best_training_ckpt=training,
            last_training_ckpt=training,
            warnings=[],
        )
        with session_samplers(
            session, result, run_id="ft-x", max_checkpoint_name_len=50, targets=["tuned"]
        ) as targets:
            assert list(targets.samplers) == ["tuned"]
    (body,) = setup.transport.bodies("POST /v1/loops/samplers")
    assert body["model_path"] == targets.tuned_checkpoint.uri  # type: ignore[union-attr]


def test_tool_call_arguments_keep_the_types_the_tools_declare(
    renderer: HFChatTemplateRenderer,
) -> None:
    answer = (
        "</think>\n\n<tool_call>\n<function=lookup>\n<parameter=id>\n123\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>"
    )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
    setup = Setup(fake_loops(answer), renderer)
    with setup.session() as session:
        generation = session.sampler(None).generate(HI, [tool], max_tokens=256)
    (call,) = generation.message["tool_calls"]
    assert call["function"] == {"name": "lookup", "arguments": '{"id": "123"}'}
    assert generation.format_valid


def test_closing_a_sampler_deactivates_it_once(setup: Setup) -> None:
    with setup.session() as session:
        sampler = session.sampler(None)
        sampler.close()
        sampler.close()
        assert not setup.loops.samplers["sampler1"].active
        assert setup.transport.calls.count(SAMPLER_DEACTIVATE) == 1
        with pytest.raises(StateError, match="sampler is closed"):
            sampler.generate(HI, [], max_tokens=8)
    assert setup.transport.calls.count(SAMPLER_DEACTIVATE) == 1


def test_a_sampler_does_not_outlive_its_session(setup: Setup) -> None:
    with setup.session() as session:
        sampler = session.sampler(None)
    assert not setup.loops.samplers["sampler1"].active
    with pytest.raises(StateError, match="sampler is closed"):
        sampler.generate(HI, [], max_tokens=8)


def test_a_generation_cut_at_max_tokens_is_not_format_valid(setup: Setup) -> None:
    with setup.session() as session:
        generation = session.sampler(None).generate(HI, [], max_tokens=3)
    assert (generation.completion_tokens, generation.stop_reason) == (3, "length")
    assert generation.message == {"role": "assistant", "content": "H"}
    assert not generation.format_valid


def test_a_prompt_that_overflows_the_sampler_is_refused(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    setup = Setup(loops, renderer, profile=dataclasses.replace(PROFILE, max_seq_len=2048))
    with setup.session() as session:
        sampler = session.sampler(None)
        with pytest.raises(UsageError, match="exceed the sampler's 2048-token context"):
            sampler.generate(HI, [], max_tokens=2048)
    assert setup.loops.kwargs("sample") == []


def test_a_transient_sampling_failure_is_retried_with_backoff(setup: Setup) -> None:
    setup.loops.fail_on("sample", error=httpx.ReadTimeout("slow"), call=1)
    setup.loops.fail_on("sample", error=FakeServerShutdownError("going away"), call=2)
    with setup.session() as session:
        generation = session.sampler(None).generate(HI, [], max_tokens=64)
        assert setup.sleeps == [2.0, 4.0]
    assert generation.message == {"role": "assistant", "content": "Hello there!"}
    assert len(setup.loops.kwargs("sample")) == 3


def test_sampling_gives_up_after_its_attempts(setup: Setup) -> None:
    for call in range(1, SAMPLE_ATTEMPTS + 1):
        setup.loops.fail_on("sample", error=TimeoutError("slow"), call=call)
    with setup.session() as session:
        sampler = session.sampler(None)
        with pytest.raises(
            TransientError, match="sampling on the Baseten sampler failed"
        ) as caught:
            sampler.generate(HI, [], max_tokens=8)
        assert setup.sleeps == [2.0, 4.0]
    assert caught.value.__notes__ == [f"sampling failed after {SAMPLE_ATTEMPTS} attempts"]


@pytest.mark.parametrize(
    "error",
    [FakeRemoteOpError("bad prompt"), http_status_error(429), http_status_error(401)],
    ids=["remote", "capacity", "auth"],
)
def test_a_sampling_failure_not_worth_retrying_is_raised_at_once(
    setup: Setup, error: Exception
) -> None:
    setup.loops.fail_on("sample", error=error)
    with setup.session() as session:
        sampler = session.sampler(None)
        with pytest.raises(ExternalError) as caught:
            sampler.generate(HI, [], max_tokens=8)
        assert setup.sleeps == []
    assert not isinstance(caught.value, TransientError)
    assert caught.value.__cause__ is error


def test_a_sampler_described_without_its_deployment_is_not_a_refusal(
    loops: FakeLoops, renderer: HFChatTemplateRenderer
) -> None:
    sampler = {"id": "sampler1", "base_url": "https://model-x.api.baseten.co/sync"}
    setup = Setup(loops, renderer, {"POST /v1/loops/samplers": (200, {"sampler": sampler})})
    with setup.session() as session, pytest.raises(TransientError) as caught:
        session.sampler(None)
    assert caught.value.hint == "deactivate the sampler of Loops session session1 in the Baseten UI"
    assert setup.loops.kwargs("SamplingClient") == []


@pytest.mark.parametrize("checkpoint", [False, True], ids=["base", "checkpoint"])
def test_a_server_error_creating_a_sampler_says_one_may_remain(
    loops: FakeLoops, renderer: HFChatTemplateRenderer, checkpoint: bool
) -> None:
    setup = Setup(loops, renderer, {"POST /v1/loops/samplers": (503, {})})
    with setup.session() as session:
        ckpt = session.save_sampler_checkpoint("s1", epoch=1, step=1) if checkpoint else None
        with pytest.raises(ExternalError, match="returned HTTP 503") as caught:
            session.sampler(ckpt)
    assert caught.value.hint == SAMPLER_LEFT


@pytest.mark.parametrize(
    ("lost", "kind", "hint"),
    [
        (
            httpx.ReadTimeout("lost"),
            ExternalError,
            "the request may have reached Baseten; check before retrying",
        ),
        (
            httpx.Response(200, content=b"<html>"),
            TransientError,
            "it may have created something that bills; check the Baseten UI",
        ),
    ],
    ids=["lost", "unreadable"],
)
def test_a_sampler_created_without_a_readable_answer_says_it_may_remain(
    setup: Setup, lost: httpx.Response | Exception, kind: type[ExternalError], hint: str
) -> None:
    setup.loops.lost["sampler"] = lost
    with setup.session() as session, pytest.raises(kind) as caught:
        session.sampler(None)
    assert caught.value.hint == f"{SAMPLER_LEFT}; {hint}"


def test_a_sampler_creation_cut_by_ctrl_c_says_it_may_remain(setup: Setup) -> None:
    setup.loops.lost["sampler"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught, setup.session() as session:
        session.sampler(None)
    assert caught.value.__notes__ == [SAMPLER_LEFT]
    assert not setup.loops.runs["run1"].active


@pytest.mark.parametrize(
    ("status", "kind", "hint"),
    [
        (422, DeploymentRejected, "nothing was created"),
        (403, ProviderAuthError, AUTH_HINT),
        (429, ProviderCapacityError, "retry later"),
    ],
    ids=["rejected", "auth", "capacity"],
)
def test_a_refused_sampler_creation_says_nothing_remains(
    loops: FakeLoops,
    renderer: HFChatTemplateRenderer,
    status: int,
    kind: type[ExternalError],
    hint: str,
) -> None:
    setup = Setup(loops, renderer, {"POST /v1/loops/samplers": (status, {})})
    with setup.session() as session, pytest.raises(kind) as caught:
        session.sampler(None)
    assert caught.value.hint == hint


def test_generate_can_be_called_from_several_threads(setup: Setup) -> None:
    with setup.session() as session:
        sampler = session.sampler(None)
        with ThreadPoolExecutor(max_workers=8) as pool:
            generations = list(
                pool.map(lambda _: sampler.generate(HI, [], max_tokens=64), range(32))
            )
    assert {generation.message["content"] for generation in generations} == {"Hello there!"}
    assert len(setup.loops.kwargs("sample")) == 32


def test_an_expired_sampler_checkpoint_fails_with_a_hint(setup: Setup) -> None:
    with setup.session() as session:
        ckpt = session.save_sampler_checkpoint("s1", epoch=1, step=1)
        setup.loops.expire(ckpt.uri)
        with pytest.raises(ExternalError) as caught:
            session.sampler(ckpt)
    assert not isinstance(caught.value, UsageError)
    assert caught.value.hint == (
        "nothing was created; checkpoint 's1' may have expired; resume from a training "
        "checkpoint and save a new sampler checkpoint"
    )
    assert setup.loops.samplers == {}


def test_a_sampler_that_fails_to_load_is_deactivated(setup: Setup) -> None:
    setup.loops.fail_on("sample", error=FakeRemoteOpError("adapter failed to load"))
    with setup.session() as session:
        ckpt = session.save_sampler_checkpoint("s1", epoch=1, step=1)
        with pytest.raises(ExternalError, match="loading the checkpoint on the Baseten sampler"):
            session.sampler(ckpt)
        assert not setup.loops.samplers["sampler1"].active
        assert setup.loops.sampling_clients[0].closed


def test_only_a_sampler_checkpoint_can_be_sampled(setup: Setup) -> None:
    with setup.session() as session:
        ckpt = session.save_training_checkpoint("t1", epoch=1, step=1)
        with pytest.raises(UsageError, match="only sample from a sampler checkpoint"):
            session.sampler(ckpt)
    assert setup.loops.samplers == {}
