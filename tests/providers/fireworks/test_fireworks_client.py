# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks Training SDK wrapper: account, error mapping, conversions and adapters.

No test reaches the network: the SDK objects the wrapper drives are replaced with small
stand-ins, and the account lookup goes through the fake REST transport. Tests that build
real ``tinker`` types skip when the ``fireworks`` extra is not installed.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.providers.base import (
    AdamConfig,
    ProviderAuthError,
    ProviderCapacityError,
    TransientError,
)
from pup_finetune.providers.fireworks import client as fw
from pup_finetune.providers.fireworks.api import CLIENT_SOURCE, FireworksAuthError
from pup_finetune.providers.fireworks.client import (
    CAPACITY_HINT,
    CAPACITY_RELEASED_HINT,
    SAMPLER_CHECKPOINT,
    TRAINING_CHECKPOINT,
    TRAINING_URL,
    CheckpointRow,
    FireworksTrainingClient,
    ForwardOutput,
    fireworks_error,
    resolve_account_id,
    sort_rows,
)
from pup_finetune.rendering.tokens import TokenizedExample
from tests.fakes.fireworks import APIConnectionError, FakeSdkError
from tests.fakes.fireworks_api import API_KEY, FakeFireworksApi

EXAMPLE = TokenizedExample((1, 2, 3), (2, 3, 4), (0.0, 1.0, 1.0))
WHAT = "doing a thing on Fireworks"


@pytest.fixture(autouse=True)
def _clean_secrets() -> Iterator[None]:
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def key() -> SecretStr:
    return SecretStr(API_KEY)


def tinker() -> Any:
    pytest.importorskip("fireworks.training.sdk", reason="needs the fireworks extra")
    return pytest.importorskip("tinker", reason="needs the fireworks extra")


def done(value: Any) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(value)
    return future


def failed(error: BaseException) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_exception(error)
    return future


# --- account --------------------------------------------------------------------------------


def test_the_account_comes_from_the_environment_first() -> None:
    fake = FakeFireworksApi()
    env = {"FIREWORKS_ACCOUNT_ID": " accounts/my-team "}
    assert resolve_account_id(key(), env=env, transport=fake.transport) == "my-team"
    assert fake.requests == []


def test_the_account_falls_back_to_the_keys_first_account() -> None:
    fake = FakeFireworksApi()
    fake.accounts.append({"name": "accounts/second-team"})
    assert resolve_account_id(key(), env={}, transport=fake.transport) == "synthetic-team"
    (request,) = fake.requests
    assert request.url.path == "/v1/accounts"


def test_a_malformed_account_in_the_environment_is_refused() -> None:
    with pytest.raises(UsageError, match="FIREWORKS_ACCOUNT_ID"):
        resolve_account_id(key(), env={"FIREWORKS_ACCOUNT_ID": "Not An/Id"})


def test_a_key_with_no_account_is_refused() -> None:
    fake = FakeFireworksApi()
    fake.accounts.clear()
    with pytest.raises(UsageError, match="no account") as caught:
        resolve_account_id(key(), env={}, transport=fake.transport)
    assert "FIREWORKS_ACCOUNT_ID" in (caught.value.hint or "")


def test_a_rejected_key_is_an_auth_error() -> None:
    fake = FakeFireworksApi(api_key="fw-synthetic-other-key")
    with pytest.raises(ProviderAuthError) as caught:
        resolve_account_id(key(), env={}, transport=fake.transport)
    assert isinstance(caught.value, FireworksAuthError)
    assert API_KEY not in str(caught.value)


def test_the_client_resolves_its_account_and_registers_the_key() -> None:
    fake = FakeFireworksApi()
    client = FireworksTrainingClient(key(), env={}, transport=fake.transport)
    assert client.account_id == "synthetic-team"
    assert redaction.redact(f"key {API_KEY}") == "key ***"


def test_an_explicit_account_needs_no_lookup() -> None:
    fake = FakeFireworksApi()
    client = FireworksTrainingClient(key(), account_id="my-team", transport=fake.transport)
    assert client.account_id == "my-team"
    assert fake.requests == []


@pytest.mark.parametrize("value", ["", "two words", "line\nbreak"])
def test_an_empty_or_malformed_key_is_refused(value: str) -> None:
    with pytest.raises(UsageError, match="Fireworks API key"):
        FireworksTrainingClient(SecretStr(value), account_id="my-team")


# --- error mapping --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        FakeSdkError("too many requests", status_code=429),
        FakeSdkError("capacity_exhausted: retry later", status_code=409),
        FakeSdkError("Training capacity is temporarily unavailable"),
        FakeSdkError("RESOURCE_EXHAUSTED: quota exceeded for trainingSessions"),
        RuntimeError("ERROR: Failed to promote session checkpoint 'x' (HTTP 429)"),
    ],
)
def test_capacity_and_quota_failures_say_to_retry_later(error: Exception) -> None:
    mapped = fireworks_error(error, what=WHAT)
    assert isinstance(mapped, ProviderCapacityError)
    assert not isinstance(mapped, TransientError)
    assert mapped.source == "fireworks"
    assert mapped.message.startswith(f"{WHAT} failed")
    # The mapping cannot know whether anything is still running, so it does not say.
    assert mapped.hint == CAPACITY_HINT
    assert "retry later" in mapped.hint
    assert "nothing was left running" not in mapped.hint


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_points_at_auth_login(status: int) -> None:
    mapped = fireworks_error(FakeSdkError("denied", status_code=status), what=WHAT)
    assert isinstance(mapped, ProviderAuthError)
    assert f"(HTTP {status})" in mapped.message
    assert "pup finetune auth login fireworks" in (mapped.hint or "")


@pytest.mark.parametrize(
    "error",
    [
        FakeSdkError("internal error", status_code=500),
        FakeSdkError("unavailable", status_code=503),
        FakeSdkError("timeout", status_code=408),
        RuntimeError("ERROR: Failed to list checkpoints (HTTP 502)"),
        ConnectionError("connection reset"),
        TimeoutError("timed out"),
        APIConnectionError("Connection error."),
        httpx.ConnectError("connection refused"),
    ],
)
def test_server_errors_and_dropped_connections_are_transient(error: Exception) -> None:
    mapped = fireworks_error(error, what=WHAT)
    assert type(mapped) is TransientError


def test_the_status_is_read_from_a_response() -> None:
    error = Exception("failed")
    error.response = SimpleNamespace(status_code=401)  # type: ignore[attr-defined]
    assert isinstance(fireworks_error(error, what=WHAT), ProviderAuthError)


def test_the_status_is_read_through_the_cause_chain() -> None:
    try:
        try:
            raise FakeSdkError("denied", status_code=401)
        except FakeSdkError as inner:
            raise RuntimeError("the request failed") from inner
    except RuntimeError as outer:
        mapped = fireworks_error(outer, what=WHAT)
    assert isinstance(mapped, ProviderAuthError)
    assert "(HTTP 401)" in mapped.message


def test_a_capacity_message_wins_over_a_server_status() -> None:
    mapped = fireworks_error(FakeSdkError("quota exceeded", status_code=503), what=WHAT)
    assert isinstance(mapped, ProviderCapacityError)


def test_other_failures_are_plain_external_errors() -> None:
    mapped = fireworks_error(FakeSdkError("invalid request", status_code=400), what=WHAT)
    assert type(mapped) is ExternalError
    assert mapped.message == f"{WHAT} failed (HTTP 400): invalid request"


def test_our_own_errors_pass_through() -> None:
    error = UsageError("bad input")
    assert fireworks_error(error, what=WHAT) is error

    def fail() -> None:
        raise error

    with pytest.raises(UsageError) as caught:
        fw._call(WHAT, fail)
    assert caught.value is error
    assert caught.value.__cause__ is None


def test_messages_are_redacted_and_short() -> None:
    redaction.register_secret(API_KEY)
    text = (
        f"ERROR: Failed to promote (HTTP 400)\n  Cause: key {API_KEY} is wrong\n"
        "  Solution: fix it\n  Agent debug: search somewhere\n  Docs: https://example.invalid"
    )
    mapped = fireworks_error(RuntimeError(text), what=WHAT)
    assert API_KEY not in str(mapped)
    assert (
        mapped.message
        == f"{WHAT} failed (HTTP 400): Failed to promote (HTTP 400); key *** is wrong"
    )
    long = fireworks_error(RuntimeError("x" * 1000), what=WHAT)
    assert len(long.message) < 400


def test_real_sdk_errors_map_by_status() -> None:
    t = tinker()
    request = httpx.Request("POST", TRAINING_URL)

    def status_error(cls: Any, status: int) -> Exception:
        error: Exception = cls(
            "failed", response=httpx.Response(status, request=request), body=None
        )
        return error

    assert isinstance(
        fireworks_error(status_error(t.RateLimitError, 429), what=WHAT), ProviderCapacityError
    )
    assert isinstance(
        fireworks_error(status_error(t.AuthenticationError, 401), what=WHAT), ProviderAuthError
    )
    assert isinstance(
        fireworks_error(status_error(t.InternalServerError, 500), what=WHAT), TransientError
    )
    assert isinstance(
        fireworks_error(t.APIConnectionError(request=request), what=WHAT), TransientError
    )
    assert isinstance(
        fireworks_error(t.APITimeoutError(request=request), what=WHAT), TransientError
    )
    assert type(fireworks_error(status_error(t.BadRequestError, 400), what=WHAT)) is ExternalError


# --- checkpoint rows ------------------------------------------------------------------------


def test_rows_are_read_from_the_listing() -> None:
    row = CheckpointRow.from_api(
        {
            "name": "accounts/a/trainingSessions/ts-1/checkpoints/run-ab-e1s4-0001-1234abcd",
            "checkpointName": "run-ab-e1s4-0001-1234abcd",
            "checkpointType": SAMPLER_CHECKPOINT,
            "promotable": True,
            "createTime": "2026-01-01T00:00:05Z",
        }
    )
    assert row is not None
    assert row.promotable
    assert row.checkpoint_type == SAMPLER_CHECKPOINT
    assert row.created_at is not None and row.created_at.second == 5
    sparse = CheckpointRow.from_api(
        {"name": "accounts/a/trainingSessions/s/checkpoints/c", "checkpointType": "T"}
    )
    assert (sparse.checkpoint_name, sparse.promotable, sparse.created_at) == ("c", False, None)


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"checkpointName": "c", "checkpointType": SAMPLER_CHECKPOINT}, "without its name"),
        ({"name": "accounts/a/trainingSessions/s/checkpoints/c"}, "without its type"),
        ("accounts/a/trainingSessions/s/checkpoints/c", "not an object"),
    ],
)
def test_unreadable_rows_are_errors(row: object, match: str) -> None:
    with pytest.raises(ExternalError, match=match):
        CheckpointRow.from_api(row)


def test_rows_sort_oldest_first() -> None:
    def row(name: str, created: str) -> CheckpointRow:
        return CheckpointRow(
            checkpoint_name=name,
            checkpoint_type=TRAINING_CHECKPOINT,
            promotable=False,
            create_time=created,
            resource_name=f"accounts/a/trainingSessions/s/checkpoints/{name}",
        )

    rows = [
        row("c", "2026-01-01T00:00:03.5Z"),
        row("a", "2026-01-01T00:00:01Z"),
        row("x", ""),
        row("b", "2026-01-01T00:00:02.123456Z"),
    ]
    assert [r.checkpoint_name for r in sort_rows(rows)] == ["x", "a", "b", "c"]


# --- conversions ----------------------------------------------------------------------------


def test_an_example_becomes_a_tinker_datum() -> None:
    tinker()
    datum = fw.to_tinker_datum(EXAMPLE)
    assert datum.model_input.to_ints() == [1, 2, 3]
    target, weights = datum.loss_fn_inputs["target_tokens"], datum.loss_fn_inputs["weights"]
    assert (target.dtype, target.shape, target.data) == ("int64", [3], [2, 3, 4])
    assert (weights.dtype, weights.shape, weights.data) == ("float32", [3], [0.0, 1.0, 1.0])


def test_adam_config_becomes_tinker_adam_params() -> None:
    tinker()
    adam = AdamConfig(
        learning_rate=2e-4, beta1=0.8, beta2=0.9, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0
    )
    params = fw.to_adam_params(adam)
    assert (
        params.learning_rate,
        params.beta1,
        params.beta2,
        params.eps,
        params.weight_decay,
        params.grad_clip_norm,
    ) == (2e-4, 0.8, 0.9, 1e-8, 0.01, 1.0)


def test_a_missing_sdk_asks_for_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.setitem(sys.modules, "fireworks.training.sdk.patches", None)
    with pytest.raises(UsageError, match="not installed") as caught:
        fw.to_tinker_datum(EXAMPLE)
    assert "pup-finetune[fireworks]" in (caught.value.hint or "")


@pytest.mark.parametrize(
    "error",
    [
        ImportError("cannot import name 'Datum' from 'tinker'"),
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),
    ],
)
def test_a_broken_sdk_install_is_not_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch, error: ImportError
) -> None:
    def broken(_name: str) -> Any:
        raise error

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(UsageError, match="failed to load") as caught:
        fw._sdk()
    assert "not installed" not in caught.value.message
    assert str(error) in caught.value.message
    assert "--reinstall" in (caught.value.hint or "")


# --- the SDK adapters, over stand-ins -------------------------------------------------------


@dataclass
class StandInFuture:
    """A finished SDK future that records the timeout it is read with."""

    value: Any = None
    error: Exception | None = None
    timeouts: list[float | None] = field(default_factory=list)

    def result(self, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return self.value


@dataclass
class StandInTrainingClient:
    """The SDK training client's calls, answering with SDK-shaped results."""

    outputs: list[Any] = field(default_factory=list)
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    run_id: str | None = "run-0123"
    error: Exception | None = None
    futures: list[StandInFuture] = field(default_factory=list)

    def _answer(self, name: str, *args: Any) -> StandInFuture:
        self.calls.append((name, *args))
        error = self.error
        future = StandInFuture(error=error) if error else StandInFuture(self.outputs.pop(0))
        self.futures.append(future)
        return future

    def forward_backward(self, data: list[Any], loss_fn: str) -> StandInFuture:
        return self._answer("forward_backward", data, loss_fn)

    def forward(self, data: list[Any], loss_fn: str) -> StandInFuture:
        return self._answer("forward", data, loss_fn)

    def optim_step(self, adam_params: Any) -> StandInFuture:
        return self._answer("optim_step", adam_params)

    def save_state(self, name: str) -> StandInFuture:
        return self._answer("save_state", name)

    def save_weights_for_sampler(self, name: str) -> StandInFuture:
        if self.error is not None:
            raise self.error
        return self._answer("save_weights_for_sampler", name)

    def load_state(self, path: str) -> StandInFuture:
        return self._answer("load_state", path)


def forward_output(*logprobs: list[float], metrics: dict[str, Any] | None = None) -> Any:
    t = tinker()
    return t.ForwardBackwardOutput(
        loss_fn_output_type="ArrayRecord",
        loss_fn_outputs=[{"logprobs": t.TensorData(data=lp, dtype="float32")} for lp in logprobs],
        metrics=metrics or {"total_tokens:sum": 3.0},
    )


def test_a_run_converts_batches_and_results() -> None:
    t = tinker()
    # The live shape: a forward-backward pass returns a loss per example, not logprobs.
    output = t.ForwardBackwardOutput(
        loss_fn_output_type="cross_entropy",
        loss_fn_outputs=[
            {
                "loss": t.TensorData(data=[3.0], dtype="float32"),
                "num_tokens": t.TensorData(data=[3], dtype="int64"),
            }
        ],
        metrics={"loss:sum": 3.0, "total_tokens:sum": 3.0, "response_tokens": 2.0},
    )
    sdk = StandInTrainingClient(
        outputs=[
            output,
            SimpleNamespace(metrics={"lr:last": 1e-4, "step_id:last": 1}),
            SimpleNamespace(path="my-team/run-0123/e1s1-ab12"),
            SimpleNamespace(path="my-team/run-0123/e1s1-ab12-1234abcd"),
            None,
        ]
    )
    run = fw._Run(sdk)
    assert run.run_id == "run-0123"
    result = run.forward_backward([EXAMPLE]).result()
    assert result == ForwardOutput(
        {"loss:sum": 3.0, "total_tokens:sum": 3.0, "response_tokens": 2.0}
    )
    (_name, data, loss_fn) = sdk.calls[0]
    assert loss_fn == "cross_entropy"
    assert data[0].model_input.to_ints() == [1, 2, 3]
    step = run.optim_step(AdamConfig(learning_rate=1e-4)).result()
    assert dict(step.metrics) == {"lr:last": 1e-4, "step_id:last": 1.0}
    assert run.save_state("e1s1-ab12").result() == "my-team/run-0123/e1s1-ab12"
    assert run.save_weights_for_sampler("e1s1-ab12").result().endswith("-1234abcd")
    assert run.load_state("my-team/run-0123/e1s1-ab12").result() is None


def test_a_result_is_read_with_the_callers_timeout() -> None:
    sdk = StandInTrainingClient(outputs=[SimpleNamespace(path="p"), SimpleNamespace(path="q")])
    run = fw._Run(sdk)
    run.save_state("a").result(timeout=12.5)
    run.save_state("b").result()
    assert [future.timeouts for future in sdk.futures] == [[12.5], [None]]


def test_metrics_keep_only_numbers() -> None:
    raw = SimpleNamespace(metrics={"lr:last": 1e-4, "step_id:last": 3, "flag": True, "note": "x"})
    sdk = StandInTrainingClient(outputs=[raw])
    step = fw._Run(sdk).optim_step(AdamConfig(learning_rate=1e-4)).result()
    assert dict(step.metrics) == {"lr:last": 1e-4, "step_id:last": 3.0}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (lambda: forward_output([-1.0, -1.0, -1.0], [-1.0]), "2 outputs for a batch of 1"),
        (lambda: forward_output(), "0 outputs for a batch of 1"),
        (lambda: SimpleNamespace(loss_fn_outputs=[{"other": 1}], metrics={}), "without logprobs"),
    ],
)
def test_a_forward_pass_needs_logprobs_for_every_example(raw: Any, match: str) -> None:
    tinker()
    run = fw._Run(StandInTrainingClient(outputs=[raw()]))
    with pytest.raises(ExternalError, match=match):
        run.forward([EXAMPLE]).result()


def test_a_run_maps_failures_when_the_result_is_read() -> None:
    tinker()
    run = fw._Run(StandInTrainingClient(error=FakeSdkError("unavailable", status_code=503)))
    pending = run.forward([EXAMPLE])
    with pytest.raises(TransientError, match="a Fireworks forward pass failed"):
        pending.result()
    with pytest.raises(TransientError, match="sampler checkpoint"):
        run.save_weights_for_sampler("e1-ab12")


def test_a_rate_limit_mid_run_does_not_say_nothing_is_running() -> None:
    tinker()
    run = fw._Run(StandInTrainingClient(error=FakeSdkError("slow down", status_code=429)))
    with pytest.raises(ProviderCapacityError) as caught:
        run.forward_backward([EXAMPLE]).result()
    assert caught.value.hint == CAPACITY_HINT
    assert "nothing was left running" not in caught.value.hint


def test_a_missing_checkpoint_path_is_an_external_error() -> None:
    run = fw._Run(StandInTrainingClient(outputs=[SimpleNamespace(path="")]))
    with pytest.raises(ExternalError, match="path"):
        run.save_state("e1-ab12").result()


def test_a_converter_bug_is_not_reported_as_a_fireworks_failure() -> None:
    run = fw._Run(StandInTrainingClient(outputs=[SimpleNamespace(metrics=object())]))
    with pytest.raises(TypeError):
        run.optim_step(AdamConfig(learning_rate=1e-4)).result()


@dataclass
class StandInSampler:
    calls: list[dict[str, Any]] = field(default_factory=list)
    closed: int = 0
    sequences: list[Any] = field(
        default_factory=lambda: [SimpleNamespace(tokens=[7, 8, 9], stop_reason="stop")]
    )

    def sample(self, **kwargs: Any) -> StandInFuture:
        self.calls.append(kwargs)
        return StandInFuture(SimpleNamespace(sequences=self.sequences))

    def close(self) -> None:
        self.closed += 1


@dataclass
class StandInHolder:
    closed: int = 0
    error: Exception | None = None

    def close(self) -> None:
        self.closed += 1
        if self.error is not None:
            raise self.error


@dataclass
class StandInService:
    training_session_id: str | None = "ts-0123"
    training_session_name: str | None = None
    error: Exception | None = None
    close_error: Exception | None = None
    closed: int = 0
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    holder: StandInHolder | None = None
    retry_409: int = 0
    sampler: StandInSampler = field(default_factory=StandInSampler)
    run: StandInTrainingClient = field(default_factory=StandInTrainingClient)

    def create_lora_training_client(self, **kwargs: Any) -> StandInTrainingClient:
        self.calls.append(("create_lora_training_client", kwargs))
        if self.error is not None:
            raise self.error
        return self.run

    def create_training_client_from_state_with_optimizer(self, path: str) -> Any:
        self.calls.append(("from_state", path))
        if self.error is not None:
            raise self.error
        return self.run

    def create_sampling_client(self, **kwargs: Any) -> Any:
        self.calls.append(("create_sampling_client", kwargs))
        return self.sampler

    def _enable_serverless_supervised_409_retry(self) -> None:
        self.retry_409 += 1

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def test_a_service_names_its_session_and_starts_runs() -> None:
    sdk = StandInService()
    service = fw._Service(sdk, account_id="my-team")
    assert service.session_id == "ts-0123"
    assert service.session_name == "accounts/my-team/trainingSessions/ts-0123"
    sdk.training_session_name = "accounts/other/trainingSessions/ts-0123"
    assert service.session_name == "accounts/other/trainingSessions/ts-0123"
    service.create_lora_training_client("accounts/fireworks/models/m", 8, seed=3)
    assert sdk.calls[-1] == (
        "create_lora_training_client",
        {"base_model": "accounts/fireworks/models/m", "rank": 8, "seed": 3, "alpha": None},
    )
    service.create_training_client_from_state_with_optimizer("my-team/run-1/e1-ab12")
    assert sdk.calls[-1] == ("from_state", "my-team/run-1/e1-ab12")


def test_a_service_without_a_session_id_is_an_error() -> None:
    service = fw._Service(StandInService(training_session_id=None), account_id="my-team")
    with pytest.raises(ExternalError, match="session id"):
        service.session_id  # noqa: B018


def test_no_capacity_ends_the_session_and_says_so() -> None:
    holder = StandInHolder()
    sdk = StandInService(error=FakeSdkError("out of capacity", status_code=409), holder=holder)
    service = fw._Service(sdk, account_id="my-team")
    with pytest.raises(ProviderCapacityError) as caught:
        service.create_lora_training_client("accounts/fireworks/models/m", 8)
    assert caught.value.hint == CAPACITY_RELEASED_HINT
    assert (sdk.closed, holder.closed) == (1, 1)
    service.close()
    assert sdk.closed == 1


def test_a_failed_cleanup_is_reported_and_not_called_released(
    capsys: pytest.CaptureFixture[str],
) -> None:
    holder = StandInHolder(error=RuntimeError("heartbeat stuck"))
    sdk = StandInService(
        error=FakeSdkError("out of capacity", status_code=409),
        close_error=RuntimeError("close refused"),
        holder=holder,
    )
    service = fw._Service(sdk, account_id="my-team")
    with pytest.raises(ProviderCapacityError) as caught:
        service.create_lora_training_client("accounts/fireworks/models/m", 8)
    assert caught.value.hint == CAPACITY_HINT
    assert (sdk.closed, holder.closed) == (1, 1)  # the heartbeat stops even so
    err = capsys.readouterr().err
    assert "may still be open" in err
    assert "closing the Fireworks training session failed: close refused" in err


def test_closing_keeps_the_first_failure() -> None:
    holder = StandInHolder(error=RuntimeError("heartbeat stuck"))
    sdk = StandInService(close_error=RuntimeError("close refused"), holder=holder)
    service = fw._Service(sdk, account_id="my-team")
    with pytest.raises(ExternalError, match="closing the Fireworks training session failed"):
        service.close()
    assert holder.closed == 1
    holder_only = fw._Service(
        StandInService(holder=StandInHolder(error=OSError("x"))), account_id="a"
    )
    with pytest.raises(ExternalError, match="heartbeat"):
        holder_only.close()


def test_other_start_failures_leave_the_session_to_the_caller() -> None:
    sdk = StandInService(error=FakeSdkError("invalid", status_code=400))
    service = fw._Service(sdk, account_id="my-team")
    with pytest.raises(ExternalError):
        service.create_training_client_from_state_with_optimizer("p")
    assert sdk.closed == 0


def test_a_sampler_needs_exactly_one_target_and_converts_samples() -> None:
    tinker()
    sdk = StandInService()
    service = fw._Service(sdk, account_id="my-team")
    with pytest.raises(UsageError):
        service.create_sampling_client(tokenizer=object())
    with pytest.raises(UsageError):
        service.create_sampling_client(tokenizer=object(), base_model="m", model_path="p")
    sampler = service.create_sampling_client(tokenizer="tok", model_path="my-team/run-1/e1-x")
    assert sdk.calls[-1] == (
        "create_sampling_client",
        {"tokenizer": "tok", "base_model": None, "model_path": "my-team/run-1/e1-x"},
    )
    sample = sampler.sample([1, 2], max_tokens=16, stop=[9]).result()
    assert (sample.tokens, sample.stop_reason) == ((7, 8, 9), "stop")
    (call,) = sdk.sampler.calls
    assert call["num_samples"] == 1
    assert call["prompt"].to_ints() == [1, 2]
    params = call["sampling_params"]
    assert (params.max_tokens, params.stop, params.temperature) == (16, [9], 0.0)
    sampler.close()
    sampler.close()
    assert sdk.sampler.closed == 1


def test_an_empty_sample_response_is_an_error() -> None:
    tinker()
    sdk = StandInService(sampler=StandInSampler(sequences=[]))
    sampler = fw._Service(sdk, account_id="my-team").create_sampling_client(
        tokenizer="tok", base_model="m"
    )
    with pytest.raises(ExternalError, match="no sampled sequence"):
        sampler.sample([1], max_tokens=4).result()


def test_a_service_closes_once() -> None:
    sdk = StandInService()
    service = fw._Service(sdk, account_id="my-team")
    service.close()
    service.close()
    assert sdk.closed == 1


# --- the client, over a stand-in SDK --------------------------------------------------------


@dataclass
class StandInControlPlane:
    rows: list[Any] = field(default_factory=list)
    listed: list[tuple[Any, ...]] = field(default_factory=list)
    promoted: list[dict[str, Any]] = field(default_factory=list)
    closed: int = 0
    error: Exception | None = None

    def list_training_session_checkpoints(self, name: str) -> list[Any]:
        self.listed.append((name,))
        if self.error is not None:
            raise self.error
        return self.rows

    def promote_session_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        self.promoted.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"name": f"accounts/my-team/models/{kwargs['output_model_id']}"}

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A stand-in for the SDK module: records how its clients are constructed."""
    for name in ("CLOUDFLARE_ACCESS_CLIENT_ID", "CLOUDFLARE_ACCESS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    made: SimpleNamespace = SimpleNamespace(
        services=[], control_planes=[], plane=StandInControlPlane(), service=StandInService()
    )

    def service_client(**kwargs: Any) -> StandInService:
        made.services.append(kwargs)
        service: StandInService = made.service
        return service

    def fireworks_client(**kwargs: Any) -> StandInControlPlane:
        made.control_planes.append(kwargs)
        plane: StandInControlPlane = made.plane
        return plane

    module = SimpleNamespace(
        FiretitanServiceClient=service_client, FireworksClient=fireworks_client
    )
    monkeypatch.setattr(fw, "_sdk", lambda: module)
    return made


def test_a_service_is_opened_on_serverless_training_with_our_header(sdk: SimpleNamespace) -> None:
    client = FireworksTrainingClient(key(), account_id="my-team")
    service = client.service()
    assert service.session_name == "accounts/my-team/trainingSessions/ts-0123"
    (kwargs,) = sdk.services
    assert kwargs["api_key"] == API_KEY
    assert kwargs["base_url"] == TRAINING_URL == "https://api.fireworks.ai/training/v1/serverless"
    assert kwargs["default_headers"] == {"X-Fireworks-Client-Source": CLIENT_SOURCE}
    assert kwargs["managed_additional_headers"] == {"X-Fireworks-Client-Source": CLIENT_SOURCE}


def test_a_service_opts_in_to_the_sdks_capacity_conflict_retry(sdk: SimpleNamespace) -> None:
    FireworksTrainingClient(key(), account_id="my-team").service()
    assert sdk.service.retry_409 == 1


def test_a_service_without_a_holder_does_not_opt_in(sdk: SimpleNamespace) -> None:
    sdk.service = SimpleNamespace(
        training_session_id="ts-1",
        _enable_serverless_supervised_409_retry=lambda: pytest.fail("needs a holder"),
    )
    FireworksTrainingClient(key(), account_id="my-team").service()


@pytest.mark.parametrize("name", ["CLOUDFLARE_ACCESS_CLIENT_ID", "CLOUDFLARE_ACCESS_CLIENT_SECRET"])
def test_credentials_the_sdk_would_forward_are_refused(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "synthetic-value")
    client = FireworksTrainingClient(key(), account_id="my-team")
    with pytest.raises(UsageError, match=name) as caught:
        client.service()
    assert "synthetic-value" not in str(caught.value)
    assert sdk.services == []


def test_checkpoints_are_listed_typed_and_oldest_first(sdk: SimpleNamespace) -> None:
    sdk.plane.rows = [
        {
            "name": "accounts/a/trainingSessions/s/checkpoints/b",
            "checkpointType": SAMPLER_CHECKPOINT,
            "createTime": "2026-01-01T00:00:02Z",
        },
        {
            "name": "accounts/a/trainingSessions/s/checkpoints/a",
            "checkpointType": TRAINING_CHECKPOINT,
            "createTime": "2026-01-01T00:00:01Z",
        },
    ]
    client = FireworksTrainingClient(key(), account_id="my-team")
    rows = client.list_session_checkpoints("accounts/a/trainingSessions/s")
    assert [row.checkpoint_name for row in rows] == ["a", "b"]
    assert sdk.plane.listed == [("accounts/a/trainingSessions/s",)]
    (kwargs,) = sdk.control_planes
    assert kwargs["additional_headers"] == {"X-Fireworks-Client-Source": CLIENT_SOURCE}
    client.close()
    client.close()
    assert sdk.plane.closed == 1


def test_an_unreadable_listing_is_an_error(sdk: SimpleNamespace) -> None:
    sdk.plane.rows = [{"checkpointName": "no-name", "checkpointType": SAMPLER_CHECKPOINT}]
    client = FireworksTrainingClient(key(), account_id="my-team")
    with pytest.raises(ExternalError, match="without its name"):
        client.list_session_checkpoints("accounts/a/trainingSessions/s")


def test_listing_failures_are_mapped(sdk: SimpleNamespace) -> None:
    sdk.plane.error = RuntimeError("ERROR: Failed to list checkpoints (HTTP 403)")
    client = FireworksTrainingClient(key(), account_id="my-team")
    with pytest.raises(ProviderAuthError):
        client.list_session_checkpoints("accounts/a/trainingSessions/s")


def test_promotion_validates_the_model_id(sdk: SimpleNamespace) -> None:
    client = FireworksTrainingClient(key(), account_id="my-team")
    with pytest.raises(UsageError, match="model id"):
        client.promote_session_checkpoint("r", "Bad_Id", "base")
    model = client.promote_session_checkpoint("r", "run-1", "accounts/fireworks/models/m")
    assert model == {"name": "accounts/my-team/models/run-1"}
    assert sdk.plane.promoted == [
        {"name": "r", "output_model_id": "run-1", "base_model": "accounts/fireworks/models/m"}
    ]


def test_a_rate_limited_promotion_does_not_say_nothing_is_running(sdk: SimpleNamespace) -> None:
    sdk.plane.error = FakeSdkError("too many requests", status_code=429)
    client = FireworksTrainingClient(key(), account_id="my-team")
    with pytest.raises(ProviderCapacityError) as caught:
        client.promote_session_checkpoint("r", "run-1", "accounts/fireworks/models/m")
    assert caught.value.hint == CAPACITY_HINT
    assert "nothing was left running" not in caught.value.hint


def test_service_construction_failures_are_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**_kwargs: Any) -> Any:
        raise FakeSdkError("too many sessions", status_code=429)

    monkeypatch.setattr(fw, "_sdk", lambda: SimpleNamespace(FiretitanServiceClient=refuse))
    client = FireworksTrainingClient(key(), account_id="my-team")
    with pytest.raises(ProviderCapacityError, match="starting a Fireworks training session"):
        client.service()


@pytest.mark.parametrize("account_id", ["", "Not An/Id", "accounts/"])
def test_an_explicit_account_is_checked_like_the_environments(account_id: str) -> None:
    with pytest.raises(UsageError, match="not a Fireworks account id"):
        FireworksTrainingClient(key(), account_id=account_id)


# --- the calls fit the installed SDK --------------------------------------------------------


def test_the_wrapper_calls_fit_the_installed_sdk(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every keyword the wrapper passes binds to the real SDK signature, without a network."""
    t = tinker()
    real = importlib.import_module("fireworks.training.sdk")
    made: dict[str, list[dict[str, Any]]] = {"SamplingParams": [], "AdamParams": []}

    def recording(cls: Any) -> Any:
        def build(**kwargs: Any) -> Any:
            made[cls.__name__].append(kwargs)
            return cls(**kwargs)

        return build

    patched = SimpleNamespace(
        Datum=t.Datum,
        ModelInput=t.ModelInput,
        TensorData=t.TensorData,
        SamplingParams=recording(t.SamplingParams),
        AdamParams=recording(t.AdamParams),
    )
    monkeypatch.setattr(fw, "_tinker", lambda: patched)
    stand_in: StandInService = sdk.service
    stand_in.run.outputs = [
        forward_output([-1.0, -1.0, -1.0]),
        forward_output([-1.0, -1.0, -1.0]),
        SimpleNamespace(metrics={}),
        SimpleNamespace(path="p"),
        SimpleNamespace(path="q"),
        None,
    ]
    sdk.plane.rows = []

    client = FireworksTrainingClient(key(), account_id="my-team")
    service = client.service()
    service.create_lora_training_client("accounts/fireworks/models/m", 8, alpha=16, seed=1)
    run = service.create_training_client_from_state_with_optimizer("my-team/run-1/e1")
    run.forward_backward([EXAMPLE]).result()
    run.forward([EXAMPLE]).result()
    run.optim_step(AdamConfig(learning_rate=1e-4)).result()
    run.save_state("e1").result()
    run.save_weights_for_sampler("e1").result()
    run.load_state("p").result()
    sampler = service.create_sampling_client(tokenizer="tok", base_model="m")
    sampler.sample([1], max_tokens=4, stop=[2], seed=3).result()
    client.list_session_checkpoints("accounts/my-team/trainingSessions/ts-0123")
    client.promote_session_checkpoint("r", "run-1", "accounts/fireworks/models/m")

    def binds(fn: Any, *args: Any, **kwargs: Any) -> None:
        inspect.signature(fn).bind(*args, **kwargs)

    service_cls, training_cls = real.FiretitanServiceClient, real.FiretitanTrainingClient
    source = inspect.getsource(service_cls.__init__)
    assert all(name in source for name in sdk.services[0] if name != "api_key")
    assert "managed_additional_headers" in source and "default_headers" in source
    for name, call in stand_in.calls:
        if name == "create_lora_training_client":
            binds(service_cls.create_lora_training_client, None, **call)
        elif name == "from_state":
            binds(service_cls.create_training_client_from_state_with_optimizer, None, call)
        else:
            binds(service_cls._create_sampling_client, None, **call)
    for name, *args in stand_in.run.calls:
        binds(getattr(training_cls, name), None, *args)
    binds(real.FiretitanSamplingClient.sample, None, **stand_in.sampler.calls[0])
    for kwargs in sdk.control_planes:
        binds(real.FireworksClient.__init__, None, **kwargs)
    for (name,) in sdk.plane.listed:
        binds(real.FireworksClient.list_training_session_checkpoints, None, name)
    for kwargs in sdk.plane.promoted:
        binds(real.FireworksClient.promote_session_checkpoint, None, **kwargs)
    for cls in (t.SamplingParams, t.AdamParams):
        (kwargs,) = made[cls.__name__]
        assert set(kwargs) <= set(inspect.signature(cls).parameters), cls.__name__
