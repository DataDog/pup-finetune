# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The in-memory Fireworks Training API behaves like the real one where the backend relies on it."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    ProviderAuthError,
    ProviderCapacityError,
    TransientError,
)
from pup_finetune.providers.fireworks.client import (
    CAPACITY_HINT,
    CAPACITY_RELEASED_HINT,
    SAMPLER_CHECKPOINT,
    TRAINING_CHECKPOINT,
)
from pup_finetune.rendering.tokens import TokenizedExample
from tests.fakes.fireworks import (
    ACCOUNT,
    BASE_MODEL,
    OPERATIONS,
    FakeFireworksTrainingClient,
    loss_at,
)

EXAMPLES = [
    TokenizedExample((1, 2, 3), (2, 3, 4), (0.0, 1.0, 1.0)),
    TokenizedExample((5, 6), (6, 7), (1.0, 1.0)),
]
ADAM = AdamConfig(learning_rate=1e-4)


@pytest.fixture
def fw() -> FakeFireworksTrainingClient:
    return FakeFireworksTrainingClient()


def test_sessions_have_ids_and_resource_names(fw: FakeFireworksTrainingClient) -> None:
    first, second = fw.service(), fw.service()
    assert re.fullmatch(r"ts-[0-9a-f]{24}", first.session_id)
    assert first.session_id != second.session_id
    assert first.session_name == f"accounts/{ACCOUNT}/trainingSessions/{first.session_id}"
    run = first.create_lora_training_client(BASE_MODEL, 8)
    assert run.run_id is not None and re.fullmatch(r"run-[0-9a-f]{32}", run.run_id)


def test_forward_backward_reports_fireworks_metrics(fw: FakeFireworksTrainingClient) -> None:
    run = fw.service().create_lora_training_client(BASE_MODEL, 8)
    out = run.forward_backward(EXAMPLES).result()
    assert set(out.metrics) == {"loss:sum", "total_tokens:sum", "response_tokens"}
    assert out.metrics["response_tokens"] == 4.0
    assert out.metrics["total_tokens:sum"] == 5.0
    assert out.metrics["loss:sum"] == pytest.approx(loss_at(0) * 4.0)
    assert out.logprobs == ()  # as live: a forward-backward pass returns no logprobs


def test_forward_reports_only_tokens_and_logprobs(fw: FakeFireworksTrainingClient) -> None:
    run = fw.service().create_lora_training_client(BASE_MODEL, 8)
    out = run.forward(EXAMPLES).result()
    assert dict(out.metrics) == {"total_tokens:sum": 5.0}
    weighted = sum(
        -lp * w
        for ex, lps in zip(EXAMPLES, out.logprobs, strict=True)
        for lp, w in zip(lps, ex.weights, strict=True)
    )
    assert weighted / 4.0 == pytest.approx(loss_at(0))


def test_the_loss_falls_only_with_applied_gradients(fw: FakeFireworksTrainingClient) -> None:
    run = fw.service().create_lora_training_client(BASE_MODEL, 8)
    step = run.optim_step(ADAM).result()  # nothing accumulated: no change
    assert dict(step.metrics) == {"lr:last": 1e-4, "step_id:last": 0.0}
    run.forward_backward(EXAMPLES).result()
    assert run.optim_step(ADAM).result().metrics["step_id:last"] == 1.0
    before = loss_at(0)
    after = run.forward(EXAMPLES).result().logprobs[0][0]
    assert -after == pytest.approx(loss_at(1)) and loss_at(1) < before


def test_the_server_renames_checkpoints(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    training = run.save_state("e1s4-ab12").result()
    sampler = run.save_weights_for_sampler("e1s4-ab12").result()
    assert training == f"{ACCOUNT}/{run.run_id}/e1s4-ab12"
    assert re.fullmatch(rf"{ACCOUNT}/{run.run_id}/e1s4-ab12-[0-9a-f]{{8}}", sampler)
    rows = fw.list_session_checkpoints(service.session_name)
    assert [row.checkpoint_type for row in rows] == [TRAINING_CHECKPOINT, SAMPLER_CHECKPOINT]
    assert [row.promotable for row in rows] == [False, True]
    assert rows[0].checkpoint_name == f"{run.run_id}-e1s4-ab12"
    assert rows[1].checkpoint_name == f"{run.run_id}-{sampler.rsplit('/', 1)[-1]}"
    assert rows[1].resource_name == f"{service.session_name}/checkpoints/{rows[1].checkpoint_name}"
    assert rows[0].created_at is not None and rows[1].created_at is not None
    assert rows[0].created_at < rows[1].created_at


def test_promotion_works_while_the_session_lives(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    run.save_weights_for_sampler("best-ab12").result()
    (row,) = fw.list_session_checkpoints(service.session_name)
    model = fw.promote_session_checkpoint(row.resource_name, "my-run", BASE_MODEL)
    assert model["name"] == f"accounts/{ACCOUNT}/models/my-run"
    assert model["kind"] == "HF_PEFT_ADDON"
    with pytest.raises(ExternalError, match="already exists"):
        fw.promote_session_checkpoint(row.resource_name, "my-run", BASE_MODEL)


def test_promotion_fails_once_the_session_expires(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    run.save_weights_for_sampler("best-ab12").result()
    service.close()
    (row,) = fw.list_session_checkpoints(service.session_name)
    with pytest.raises(ExternalError, match="expired"):
        fw.promote_session_checkpoint(row.resource_name, "my-run", BASE_MODEL)
    with pytest.raises(ExternalError, match="ended"):
        run.forward(EXAMPLES)


def test_training_checkpoints_are_not_promotable(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    service.create_lora_training_client(BASE_MODEL, 8).save_state("e1-ab12").result()
    (row,) = fw.list_session_checkpoints(service.session_name)
    with pytest.raises(ExternalError, match="not found"):
        fw.promote_session_checkpoint(row.resource_name, "my-run", BASE_MODEL)


def test_long_names_save_but_fail_at_promotion(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    run.save_weights_for_sampler("a-name-of-18-chars").result()
    (row,) = fw.list_session_checkpoints(service.session_name)
    with pytest.raises(ExternalError, match="17 characters"):
        fw.promote_session_checkpoint(row.resource_name, "my-run", BASE_MODEL)
    with pytest.raises(UsageError):
        fw.promote_session_checkpoint(row.resource_name, "Not_Valid", BASE_MODEL)


def test_a_run_resumes_from_a_closed_sessions_training_checkpoint(
    fw: FakeFireworksTrainingClient,
) -> None:
    first = fw.service()
    run = first.create_lora_training_client(BASE_MODEL, 8)
    run.forward_backward(EXAMPLES).result()
    run.optim_step(ADAM).result()
    path = run.save_state("e1-ab12").result()
    first.close()
    second = fw.service()
    resumed = second.create_training_client_from_state_with_optimizer(path)
    assert resumed.run_id != run.run_id
    assert -resumed.forward(EXAMPLES).result().logprobs[0][0] == pytest.approx(loss_at(1))
    with pytest.raises(ExternalError, match="no training checkpoint"):
        second.create_training_client_from_state_with_optimizer(f"{ACCOUNT}/run-x/missing")


def test_load_state_restores_weights(fw: FakeFireworksTrainingClient) -> None:
    run = fw.service().create_lora_training_client(BASE_MODEL, 8)
    path = run.save_state("e0-ab12").result()
    run.forward_backward(EXAMPLES).result()
    run.optim_step(ADAM).result()
    run.load_state(path).result()
    assert -run.forward(EXAMPLES).result().logprobs[0][0] == pytest.approx(loss_at(0))


def test_sampling_returns_canned_tokens(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    path = run.save_weights_for_sampler("best-ab12").result()
    sampler = service.create_sampling_client(tokenizer=object(), model_path=path)
    sample = sampler.sample([1, 2], max_tokens=10, stop=[103]).result()
    assert (sample.tokens, sample.stop_reason) == ((101, 102, 103), "stop")
    assert sampler.sample([1], max_tokens=2).result().stop_reason == "length"
    base = service.create_sampling_client(tokenizer=object(), base_model=BASE_MODEL)
    assert base.sample([1], max_tokens=10).result().tokens == (101, 102, 103, 248046)
    with pytest.raises(UsageError):
        service.create_sampling_client(tokenizer=object())
    with pytest.raises(ExternalError, match="no sampler checkpoint"):
        service.create_sampling_client(tokenizer=object(), model_path=f"{ACCOUNT}/run-x/nope")


def test_capacity_failures_leave_nothing_running(fw: FakeFireworksTrainingClient) -> None:
    fw.fail("service", "rate_limit")
    with pytest.raises(ProviderCapacityError) as refused:
        fw.service()
    assert refused.value.hint == CAPACITY_HINT
    service = fw.service()
    fw.fail("create_lora_training_client", "capacity")
    with pytest.raises(ProviderCapacityError) as caught:
        service.create_lora_training_client(BASE_MODEL, 8)
    assert caught.value.hint == CAPACITY_RELEASED_HINT
    assert service.closed  # type: ignore[attr-defined]


def test_a_rate_limit_mid_run_leaves_the_session_open(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    fw.fail("forward_backward", "rate_limit")
    with pytest.raises(ProviderCapacityError) as caught:
        run.forward_backward(EXAMPLES).result()
    assert caught.value.hint == CAPACITY_HINT
    assert not service.closed  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("kind", "error"),
    [("auth", ProviderAuthError), ("server", TransientError), ("connection", TransientError)],
)
def test_injected_failures_arrive_mapped(
    fw: FakeFireworksTrainingClient, kind: str, error: type[Exception]
) -> None:
    run = fw.service().create_lora_training_client(BASE_MODEL, 8)
    fw.fail("forward", kind, times=2)  # type: ignore[arg-type]
    for _ in range(2):
        pending = run.forward(EXAMPLES)
        with pytest.raises(error):
            pending.result()
    run.forward(EXAMPLES).result()
    fw.fail("save_weights_for_sampler", kind)  # type: ignore[arg-type]
    with pytest.raises(error):
        run.save_weights_for_sampler("e1-ab12")


def test_calls_are_recorded(fw: FakeFireworksTrainingClient) -> None:
    service = fw.service()
    service.create_lora_training_client(BASE_MODEL, 16, alpha=32, seed=7)
    service.close()
    assert fw.calls == [
        ("service",),
        ("create_lora_training_client", BASE_MODEL, 16, 32, 7),
        ("close_service", service.session_id),
    ]


def _operations(fw: FakeFireworksTrainingClient) -> dict[str, Callable[[], Any]]:
    """One call per operation the fake can fail, each resolved, on a fresh session."""
    service = fw.service()
    run = service.create_lora_training_client(BASE_MODEL, 8)
    training = run.save_state("e1-ab12").result()
    sampled = run.save_weights_for_sampler("e1-ab12").result()
    sampler = service.create_sampling_client(tokenizer=object(), model_path=sampled)
    row = next(r for r in fw.list_session_checkpoints(service.session_name) if r.promotable)
    return {
        "service": fw.service,
        "create_lora_training_client": lambda: service.create_lora_training_client(BASE_MODEL, 8),
        "create_training_client_from_state_with_optimizer": lambda: (
            service.create_training_client_from_state_with_optimizer(training)
        ),
        "create_sampling_client": lambda: service.create_sampling_client(
            tokenizer=object(), base_model=BASE_MODEL
        ),
        "list_session_checkpoints": lambda: fw.list_session_checkpoints(service.session_name),
        "promote_session_checkpoint": lambda: fw.promote_session_checkpoint(
            row.resource_name, "my-run", BASE_MODEL
        ),
        "get_model": lambda: fw.get_model(f"accounts/{fw.account_id}/models/my-run"),
        "forward_backward": lambda: run.forward_backward(EXAMPLES).result(),
        "forward": lambda: run.forward(EXAMPLES).result(),
        "optim_step": lambda: run.optim_step(ADAM).result(),
        "save_state": lambda: run.save_state("e2-ab12").result(),
        "save_weights_for_sampler": lambda: run.save_weights_for_sampler("e2-ab12").result(),
        "load_state": lambda: run.load_state(training).result(),
        "sample": lambda: sampler.sample([1], max_tokens=4).result(),
        "close_sampler": sampler.close,
        "close_service": service.close,
        "close": fw.close,
    }


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_every_operation_can_fail(operation: str) -> None:
    fw = FakeFireworksTrainingClient()
    call = _operations(fw)[operation]
    fw.fail(operation, "server")
    with pytest.raises(TransientError):
        call()
    call()  # the failure was used up


def test_every_operation_is_covered() -> None:
    assert set(_operations(FakeFireworksTrainingClient())) == OPERATIONS


def test_a_failed_service_close_leaves_the_session_running(
    fw: FakeFireworksTrainingClient,
) -> None:
    service = fw.service()
    fw.fail("close_service", "server")
    with pytest.raises(TransientError):
        service.close()
    assert fw.sessions[service.session_id].alive
    service.close()
    assert not fw.sessions[service.session_id].alive


def test_a_stalled_call_never_resolves(fw: FakeFireworksTrainingClient) -> None:
    waited: list[float | None] = []
    run = fw.service().create_lora_training_client(BASE_MODEL, 8)
    fw.stall("forward", on_wait=waited.append)
    pending = run.forward(EXAMPLES)
    # As the wrapper raises a wait that ran out: mapped, and raised from the TimeoutError.
    with pytest.raises(TransientError) as excinfo:
        pending.result(timeout=5.0)
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert waited == [5.0]
    assert fw.stalled[0].timeouts == [5.0]
    fw.stall("forward", wrapped=False)
    with pytest.raises(TimeoutError):
        run.forward(EXAMPLES).result(timeout=1.0)
    run.forward(EXAMPLES).result()  # the stall was used up
    with pytest.raises(ValueError, match="cannot stall"):
        fw.stall("sample")


def test_an_unknown_operation_cannot_be_failed(fw: FakeFireworksTrainingClient) -> None:
    with pytest.raises(ValueError, match="unknown operation"):
        fw.fail("forward_backwards", "server")
    with pytest.raises(ValueError, match="unknown failure kind"):
        fw.fail("forward", "flaky")  # type: ignore[arg-type]
