# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import json
import os
import signal
import stat
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.data import read_rows, write_rows
from pup_finetune.errors import ExternalError, FinetuneError, StateError, UsageError
from pup_finetune.providers import TrainRequest
from pup_finetune.providers.baseten import training
from pup_finetune.providers.baseten.client import CHECKPOINT_URI, BasetenHTTP
from pup_finetune.providers.baseten.registration import LOOPS_INSTALL_HINT
from pup_finetune.providers.baseten.training import TerminationRequested, plan_training, train
from pup_finetune.rendering import ModelFamily
from pup_finetune.training import SFTSettings, TrainingDataError
from tests.fakes.baseten import BasetenTransport, FakeLoops, baseten_transport, http_status_error
from tests.fakes.tokenizer import FakeChatTokenizer

SECRET = "zq9Xsynthetic0000basetenKEYw7Pm"  # noqa: S105 - synthetic test value
KEY = SecretStr(SECRET)
FIXTURES = Path(__file__).parents[2] / "fixtures" / "rows"
ROWS = list(read_rows(FIXTURES / "tool_use.jsonl"))
# Short rows (at most 545 fake tokens per datum) validate, so max_seq_len can cut training rows.
VALIDATION_IDS = {"row-units-001", "row-plain-001"}
BASE_MODEL = "Qwen/Qwen3.5-9B"
RUN = "/v1/loops/runs/run1"
RESULT_KEYS = {
    "status",
    "provider",
    "run_id",
    "baseten_session_id",
    "baseten_run_id",
    "model",
    "base_model",
    "epochs",
    "best_epoch",
    "best_sampler_weights_uri",
    "last_resumable_state_uri",
    "stopped_early",
    "budget",
    "cleanup",
}
CLEAN = {"client_closed": True, "run_deactivated": True, "error": None}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    redaction.clear_secrets()
    sleeps: list[float] = []
    monkeypatch.setattr(training, "_sleep", sleeps.append)
    yield sleeps
    redaction.clear_secrets()


@pytest.fixture
def sleeps(_isolated: list[float]) -> list[float]:
    return _isolated


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    write_rows(tmp_path / "train.jsonl", [r for r in ROWS if r.id not in VALIDATION_IDS])
    write_rows(tmp_path / "validation.jsonl", [r for r in ROWS if r.id in VALIDATION_IDS])
    return tmp_path / "run"


def fake_tokenizer(family: ModelFamily) -> FakeChatTokenizer:
    return FakeChatTokenizer()


def make_request(run_dir: Path, **settings: Any) -> TrainRequest:
    return TrainRequest(
        train_path=run_dir.parent / "train.jsonl",
        validation_path=run_dir.parent / "validation.jsonl",
        model="qwen3p5-9b",
        run_dir=run_dir,
        run_id="sft-test",
        settings=SFTSettings(**({"max_epochs": 2, "batch_size": 8} | settings)),
    )


def capabilities(**fields: Any) -> tuple[int, dict[str, Any]]:
    entry = {
        "model_name": BASE_MODEL,
        "max_context_length": 131_072,
        "max_enabled_seq_len": 131_072,
        "enabled": True,
    }
    return 200, {"supported_models": [{**entry, **fields}]}


def run_status(name: str) -> tuple[int, dict[str, Any]]:
    return 200, {"run": {"id": "run1", "status": {"name": name}}}


def routes() -> dict[str, Any]:
    """A run that trains, is deactivated and reads inactive on the first poll."""
    return {
        "GET /v1/loops/capabilities": capabilities(),
        f"GET {RUN}": [run_status("ACTIVE"), run_status("INACTIVE")],
        f"POST {RUN}/deactivate": (200, {"id": "run1"}),
    }


def run_train(
    run_dir: Path,
    *,
    loops: FakeLoops | None = None,
    transport: BasetenTransport | None = None,
    **settings: Any,
) -> dict[str, Any]:
    with BasetenHTTP(KEY, transport=transport or baseten_transport(routes())) as http:
        return train(
            make_request(run_dir, **settings),
            api_key=KEY,
            http=http,
            loops=loops or FakeLoops(),
            tokenizer_loader=fake_tokenizer,
        )


def read(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


# --- plan_training -------------------------------------------------------------------


def test_plan_reads_renders_and_sizes_the_run(run_dir: Path) -> None:
    plan, prepared = plan_training(make_request(run_dir), tokenizer_loader=fake_tokenizer)
    assert set(plan) == {
        "run_id",
        "provider",
        "method",
        "model",
        "base_model",
        "run_dir",
        "dataset",
        "config",
        "budget",
        "rendering",
        "estimates",
    }
    assert (plan["provider"], plan["method"]) == ("baseten", "Baseten Loops LoRA SFT")
    assert (plan["model"], plan["base_model"], plan["run_id"]) == (
        "qwen3p5-9b",
        BASE_MODEL,
        "sft-test",
    )
    assert plan["dataset"] == {
        "train_path": str(run_dir.parent / "train.jsonl"),
        "validation_path": str(run_dir.parent / "validation.jsonl"),
        "train_rows": 12,
        "validation_rows": 2,
        "train_datums": 30,
        "validation_datums": 3,
        "rejected_train_rows": [],
    }
    assert plan["config"] == {
        "max_epochs": 2,
        "early_stopping_patience": 1,
        "min_delta": 0.0,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "seed": 42,
        "lora_rank": 8,
        "max_seq_len": 131_072,
        "microbatch_token_budget": 131_072,
        "capacity": "dedicated",
    }
    assert plan["budget"] == {
        "enabled": False,
        "max_spend_usd": None,
        "hourly_rate_usd": None,
        "spend_reserve_fraction": 0.1,
        "maximum_active_seconds": None,
    }
    assert plan["rendering"]["tokenizer_repo"] == BASE_MODEL
    assert plan["estimates"]["optimizer_steps_per_epoch"] == 4
    assert plan["estimates"]["max_optimizer_steps"] == 8
    assert plan["estimates"]["max_context_tokens"] == 1692
    assert (prepared.lora_rank, prepared.max_seq_len, prepared.token_budget) == (
        8,
        131_072,
        131_072,
    )
    assert len(prepared.train_rows) == 12
    assert sum(len(batch) for batch in prepared.validation_batches) == 3
    json.dumps(plan)  # the plan is written as JSON


def test_plan_excludes_training_rows_that_do_not_fit(run_dir: Path) -> None:
    request = make_request(run_dir, max_seq_len=900, lora_rank=16, microbatch_token_budget=2000)
    plan, prepared = plan_training(request, tokenizer_loader=fake_tokenizer)
    assert plan["dataset"]["train_rows"] == len(prepared.train_rows) == 1
    rejected = plan["dataset"]["rejected_train_rows"]
    assert len(rejected) == 11
    assert rejected[0] == {
        "id": "row-weather-002",
        "code": "exceeds_context",
        "detail": "a rendered datum has 1087 tokens, over the limit of 900",
        "tokens": 1087,
        "limit": 900,
    }
    config = plan["config"]
    assert (config["max_seq_len"], config["lora_rank"], config["microbatch_token_budget"]) == (
        900,
        16,
        2000,
    )


def test_plan_caps_max_seq_len_at_the_model_limit(run_dir: Path) -> None:
    plan, _ = plan_training(
        make_request(run_dir, max_seq_len=10**7), tokenizer_loader=fake_tokenizer
    )
    assert plan["config"]["max_seq_len"] == 131_072


def test_plan_reports_the_spend_guard(run_dir: Path) -> None:
    request = make_request(run_dir, max_spend_usd=75.0, hourly_rate_usd=30.0)
    plan, _ = plan_training(request, tokenizer_loader=fake_tokenizer)
    assert plan["budget"]["enabled"] is True
    assert plan["budget"]["maximum_active_seconds"] == pytest.approx(8100.0)


def test_plan_refuses_validation_rows_that_cannot_render(run_dir: Path) -> None:
    with pytest.raises(UsageError, match="1 validation rows cannot be rendered") as caught:
        plan_training(make_request(run_dir, max_seq_len=500), tokenizer_loader=fake_tokenizer)
    assert caught.value.hint is not None
    assert "row-units-001: exceeds_context" in caught.value.hint


def test_plan_refuses_when_no_training_row_renders(run_dir: Path) -> None:
    with pytest.raises(UsageError, match=r"no row of .* can be rendered for qwen3p5-9b") as caught:
        plan_training(make_request(run_dir, max_seq_len=600), tokenizer_loader=fake_tokenizer)
    assert caught.value.hint == (
        "12 rows rejected; first: row-weather-001: exceeds_context: "
        "a rendered datum has 878 tokens, over the limit of 600"
    )


def test_plan_refuses_a_datum_over_the_microbatch_budget(run_dir: Path) -> None:
    with pytest.raises(TrainingDataError, match="over the microbatch token budget of 100"):
        plan_training(
            make_request(run_dir, microbatch_token_budget=100), tokenizer_loader=fake_tokenizer
        )


def test_plan_reports_unreadable_and_invalid_files(run_dir: Path) -> None:
    request = make_request(run_dir)
    (run_dir.parent / "empty.jsonl").write_text("")
    cases = [
        ({"train_path": run_dir.parent / "missing.jsonl"}, r"cannot read .*missing\.jsonl"),
        ({"train_path": FIXTURES / "invalid.jsonl"}, r"invalid\.jsonl:1: "),
        ({"validation_path": run_dir.parent / "empty.jsonl"}, r"empty\.jsonl has no rows"),
        ({"model": "nope"}, "unknown Baseten model"),
    ]
    for changes, message in cases:
        with pytest.raises(UsageError, match=message):
            plan_training(dataclasses.replace(request, **changes), tokenizer_loader=fake_tokenizer)


# --- train: success ------------------------------------------------------------------


def test_train_runs_the_loop_cleans_up_and_writes_the_contract(
    run_dir: Path, sleeps: list[float]
) -> None:
    handlers = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    loops = FakeLoops(validation_losses=[1.0, 0.8])
    transport = baseten_transport(routes())
    result = run_train(run_dir, loops=loops, transport=transport)

    assert set(result) == RESULT_KEYS | {"run_dir", "next_command"}
    assert result["status"] == "completed"
    assert (result["provider"], result["run_id"], result["model"]) == (
        "baseten",
        "sft-test",
        "qwen3p5-9b",
    )
    assert (result["baseten_session_id"], result["baseten_run_id"]) == ("session1", "run1")
    assert result["base_model"] == BASE_MODEL
    assert [epoch["validation_loss"] for epoch in result["epochs"]] == [1.0, 0.8]
    assert result["best_epoch"] == 2
    match = CHECKPOINT_URI.fullmatch(result["best_sampler_weights_uri"])
    assert match is not None
    assert match[1] == result["baseten_run_id"]
    assert result["last_resumable_state_uri"] == "bt://loops:run1/state/sft-test-state-epoch-2"
    assert result["stopped_early"] is False
    assert result["budget"]["stopped"] is False
    assert result["budget"]["elapsed_seconds"] > 0.0
    assert result["cleanup"] == CLEAN
    assert result["run_dir"] == str(run_dir)
    assert result["next_command"] == (
        f"pup finetune deploy --run-dir {run_dir} --serving-mode sampler"
    )

    assert read(run_dir, "result.json") == {
        key: value for key, value in result.items() if key in RESULT_KEYS
    }
    assert read(run_dir, "plan.json")["run_id"] == "sft-test"
    assert len(read(run_dir, "epochs.json")) == 2
    state = read(run_dir, "run-state.json")
    assert state["status"] == "completed"
    assert (state["baseten_session_id"], state["baseten_run_id"]) == ("session1", "run1")
    assert (state["best_epoch"], state["cleanup_status"], state["error_type"]) == (
        2,
        "completed",
        None,
    )
    for name in ("plan.json", "epochs.json", "run-state.json", "result.json"):
        assert stat.S_IMODE((run_dir / name).stat().st_mode) == 0o600

    assert loops.kwargs("create_lora_training_client") == [
        {
            "base_model": BASE_MODEL,
            "rank": 8,
            "replicas": 1,
            "seed": 42,
            "max_seq_len": 131_072,
            "with_sampler": False,
            "name": "sft-test",
        }
    ]
    assert loops.trainer is not None
    assert loops.trainer.closed
    assert transport.calls == [
        "GET /v1/loops/capabilities",
        f"GET {RUN}",
        f"POST {RUN}/deactivate",
        f"GET {RUN}",
    ]
    assert sleeps == [1.0]
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == handlers


def test_a_full_spend_guard_stops_the_run_normally(run_dir: Path) -> None:
    result = run_train(run_dir, max_spend_usd=1e-9, hourly_rate_usd=1e6)
    assert result["status"] == "budget_stopped"
    assert result["budget"]["enabled"] is True
    assert result["budget"]["stopped"] is True
    assert (result["epochs"], result["best_sampler_weights_uri"], result["next_command"]) == (
        [],
        None,
        None,
    )
    assert result["cleanup"] == CLEAN


def test_training_off_the_main_thread_leaves_signals_alone(run_dir: Path) -> None:
    handler = signal.getsignal(signal.SIGTERM)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            results.append(run_train(run_dir))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(timeout=60)
    assert errors == []
    assert results[0]["status"] == "completed"
    assert signal.getsignal(signal.SIGTERM) == handler


# --- train: refusals before anything is paid -----------------------------------------


@pytest.mark.parametrize(
    ("answer", "error", "message", "hint"),
    [
        (
            capabilities(model_name="other/model"),
            UsageError,
            f"Baseten Loops does not offer {BASE_MODEL} to this workspace",
            "choose another --model, or ask Baseten to enable this one",
        ),
        (
            capabilities(
                enabled=False,
                enablement_details={"reason_detail": "Not on this plan.", "remediation": "Ask."},
            ),
            UsageError,
            f"this Baseten workspace is not enabled to train {BASE_MODEL}: Not on this plan.",
            "Ask.",
        ),
        (
            capabilities(max_enabled_seq_len=0),
            UsageError,
            f"this Baseten workspace is not enabled to train {BASE_MODEL}",
            "ask Baseten to enable the model",
        ),
        (
            capabilities(max_enabled_seq_len=65_536),
            UsageError,
            f"Baseten trains {BASE_MODEL} on at most 65536 tokens, below this run's "
            "max_seq_len of 131072",
            "rerun with --max-seq-len 65536",
        ),
        (
            {
                "supported_models": [{"model_name": BASE_MODEL, "max_context_length": 32_768}],
            },
            UsageError,
            f"Baseten trains {BASE_MODEL} on at most 32768 tokens, below this run's "
            "max_seq_len of 131072",
            "rerun with --max-seq-len 32768",
        ),
        (
            capabilities(max_context_length=None),
            ExternalError,
            f"Baseten returned an invalid context limit for {BASE_MODEL}",
            None,
        ),
    ],
    ids=["not-offered", "not-enabled", "zero-limit", "enabled-limit", "context-limit", "invalid"],
)
def test_the_capability_preflight_refuses_before_anything_is_paid(
    run_dir: Path, answer: Any, error: type[FinetuneError], message: str, hint: str | None
) -> None:
    if isinstance(answer, dict):
        answer = (200, answer)
    loops = FakeLoops()
    transport = baseten_transport(routes())
    transport.routes["GET /v1/loops/capabilities"] = [answer]
    with pytest.raises(error) as caught:
        run_train(run_dir, loops=loops, transport=transport)
    assert (caught.value.message, caught.value.hint) == (message, hint)
    assert loops.calls == []
    assert transport.calls == ["GET /v1/loops/capabilities"]
    assert [p.name for p in run_dir.iterdir()] in ([], [".pup-finetune.lock"])


def test_a_used_run_dir_is_refused_before_any_call(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / "plan.json").write_text("{}")
    transport = baseten_transport({})
    with pytest.raises(StateError, match="is not empty"):
        run_train(run_dir, transport=transport)
    assert transport.requests == []


def test_a_missing_sdk_fails_before_the_run_dir_is_touched(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "baseten", None)
    monkeypatch.setitem(sys.modules, "baseten.loops", None)
    transport = baseten_transport({})
    with (
        BasetenHTTP(KEY, transport=transport) as http,
        pytest.raises(UsageError, match="SDK is not installed") as caught,
    ):
        train(make_request(run_dir), api_key=KEY, http=http, tokenizer_loader=fake_tokenizer)
    assert caught.value.hint == LOOPS_INSTALL_HINT
    assert not run_dir.exists()
    assert transport.requests == []


# --- train: failures and cleanup -----------------------------------------------------


def test_a_training_failure_cleans_up_and_writes_the_result(run_dir: Path) -> None:
    loops = FakeLoops()
    loops.fail_on("forward_backward", error=http_status_error(500), call=2)
    transport = baseten_transport(routes())
    with pytest.raises(ExternalError, match="forward_backward call failed") as caught:
        run_train(run_dir, loops=loops, transport=transport)
    assert caught.value.hint is None
    result = read(run_dir, "result.json")
    assert set(result) == RESULT_KEYS
    assert (result["status"], result["epochs"], result["best_sampler_weights_uri"]) == (
        "failed",
        [],
        None,
    )
    assert result["cleanup"] == CLEAN
    assert f"POST {RUN}/deactivate" in transport.calls
    state = read(run_dir, "run-state.json")
    assert (state["status"], state["error_type"]) == ("failed", "ExternalError")


def test_a_failure_after_a_best_epoch_keeps_its_checkpoint(run_dir: Path) -> None:
    loops = FakeLoops()
    loops.fail_on("forward", error=http_status_error(400), call=2)
    with pytest.raises(ExternalError, match="the Loops forward call failed"):
        run_train(run_dir, loops=loops)
    result = read(run_dir, "result.json")
    assert (result["status"], result["best_epoch"]) == ("failed", 1)
    assert (
        result["best_sampler_weights_uri"]
        == "bt://loops:run1/sampler_weights/sft-test-sampler-epoch-1"
    )


def test_a_trainer_that_never_starts_needs_no_deactivation(run_dir: Path) -> None:
    loops = FakeLoops()
    loops.fail_on("create_lora_training_client", error=http_status_error(429))
    transport = baseten_transport(routes())
    with pytest.raises(ExternalError, match="no capacity for the trainer") as caught:
        run_train(run_dir, loops=loops, transport=transport)
    assert caught.value.hint == "retry later; rerun with a new --run-dir"
    assert transport.calls == ["GET /v1/loops/capabilities"]
    result = read(run_dir, "result.json")
    assert (result["status"], result["baseten_run_id"]) == ("failed", None)
    assert result["baseten_session_id"] == "session1"
    assert result["cleanup"] == {"client_closed": True, "run_deactivated": False, "error": None}
    assert read(run_dir, "run-state.json")["cleanup_status"] == "completed"


def test_a_trainer_that_fails_to_become_ready_records_the_orphan_step(run_dir: Path) -> None:
    loops = FakeLoops()
    loops.fail_on("create_lora_training_client", error=RuntimeError("terminal state FAILED"))
    with pytest.raises(ExternalError, match="provisioning failed") as caught:
        run_train(run_dir, loops=loops)
    orphan = "a run may remain in Loops session session1; deactivate it in the Baseten UI"
    assert caught.value.hint == f"{orphan}; rerun with a new --run-dir"
    result = read(run_dir, "result.json")
    assert result["baseten_session_id"] == "session1"
    assert result["cleanup"] == {"client_closed": True, "run_deactivated": False, "error": orphan}
    state = read(run_dir, "run-state.json")
    assert (state["baseten_session_id"], state["cleanup_status"]) == ("session1", "failed")


def test_a_cleanup_failure_after_success_names_the_manual_step(
    run_dir: Path, sleeps: list[float]
) -> None:
    transport = baseten_transport(routes())
    transport.routes[f"POST {RUN}/deactivate"] = [(500, {})]
    with pytest.raises(ExternalError) as caught:
        run_train(run_dir, transport=transport)
    assert transport.calls.count(f"POST {RUN}/deactivate") == 3
    assert sleeps == [1.0, 2.0]
    assert caught.value.message == (
        "could not deactivate Baseten run run1 (the Loops run deactivation returned HTTP 500)"
    )
    assert caught.value.hint == (
        "deactivate it in the Baseten UI, or run: curl -X POST "
        "https://api.baseten.co/v1/loops/runs/run1/deactivate "
        '-H "Authorization: Api-Key $BASETEN_API_KEY"'
    )
    result = read(run_dir, "result.json")
    assert result["status"] == "completed"
    assert result["cleanup"] == {
        "client_closed": True,
        "run_deactivated": False,
        "error": caught.value.message,
    }
    assert read(run_dir, "run-state.json")["cleanup_status"] == "failed"


def test_a_run_that_stays_active_is_a_cleanup_failure(run_dir: Path, sleeps: list[float]) -> None:
    transport = baseten_transport(routes())
    transport.routes[f"GET {RUN}"] = [run_status("ACTIVE")]
    with pytest.raises(ExternalError, match=r"run1 \(Baseten run run1 is still active\)"):
        run_train(run_dir, transport=transport)
    assert sleeps == [1.0] * 12
    assert transport.calls.count(f"GET {RUN}") == 13


def test_transient_errors_while_polling_are_retried(run_dir: Path, sleeps: list[float]) -> None:
    transport = baseten_transport(routes())
    transport.routes[f"GET {RUN}"] = [
        run_status("ACTIVE"),
        (503, {}),
        run_status("INACTIVE"),
    ]
    assert run_train(run_dir, transport=transport)["cleanup"] == CLEAN
    assert sleeps == [1.0, 1.0]


def test_a_failed_state_check_and_post_still_deactivate(run_dir: Path, sleeps: list[float]) -> None:
    transport = baseten_transport(routes())
    transport.routes[f"GET {RUN}"] = [(503, {}), run_status("INACTIVE")]
    transport.routes[f"POST {RUN}/deactivate"] = [(503, {}), (200, {"id": "run1"})]
    assert run_train(run_dir, transport=transport)["cleanup"] == CLEAN
    assert transport.calls.count(f"POST {RUN}/deactivate") == 2
    assert sleeps == [1.0, 1.0]


@pytest.mark.parametrize("answer", [run_status("INACTIVE"), (404, {})], ids=["inactive", "gone"])
def test_an_inactive_run_is_not_deactivated_again(run_dir: Path, answer: Any) -> None:
    transport = baseten_transport(routes())
    transport.routes[f"GET {RUN}"] = [answer]
    assert run_train(run_dir, transport=transport)["cleanup"] == CLEAN
    assert f"POST {RUN}/deactivate" not in transport.calls


def test_training_and_cleanup_failures_raise_the_training_error_with_the_step(
    run_dir: Path,
) -> None:
    loops = FakeLoops()
    loops.fail_on("forward_backward", error=http_status_error(500))
    transport = baseten_transport(routes())
    transport.routes[f"POST {RUN}/deactivate"] = [(500, {})]
    with pytest.raises(ExternalError, match="forward_backward call failed") as caught:
        run_train(run_dir, loops=loops, transport=transport)
    assert caught.value.hint is not None
    assert "/v1/loops/runs/run1/deactivate" in caught.value.hint


def test_an_interrupt_still_prints_the_cleanup_step(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loops = FakeLoops()
    loops.fail_on("optim_step", error=KeyboardInterrupt())
    transport = baseten_transport(routes())
    transport.routes[f"POST {RUN}/deactivate"] = [(500, {})]
    with pytest.raises(KeyboardInterrupt):
        run_train(run_dir, loops=loops, transport=transport)
    assert "/v1/loops/runs/run1/deactivate" in capsys.readouterr().err
    assert read(run_dir, "result.json")["status"] == "failed"


def test_sigterm_stops_training_and_still_cleans_up(run_dir: Path) -> None:
    handler = signal.getsignal(signal.SIGTERM)
    loops = FakeLoops()
    loops.on_call("forward_backward", lambda: os.kill(os.getpid(), signal.SIGTERM))
    transport = baseten_transport(routes())
    with pytest.raises(FinetuneError, match="received SIGTERM") as caught:
        run_train(run_dir, loops=loops, transport=transport)
    assert isinstance(caught.value.__cause__, TerminationRequested)
    assert caught.value.exit_code == 1
    assert loops.trainer is not None
    assert loops.trainer.closed
    assert f"POST {RUN}/deactivate" in transport.calls
    result = read(run_dir, "result.json")
    assert (result["status"], result["cleanup"]) == ("failed", CLEAN)
    assert signal.getsignal(signal.SIGTERM) == handler


def test_sigterm_while_provisioning_names_the_session(run_dir: Path) -> None:
    loops = FakeLoops()
    loops.on_call("create_lora_training_client", lambda: os.kill(os.getpid(), signal.SIGTERM))
    transport = baseten_transport(routes())
    with pytest.raises(FinetuneError, match="received SIGTERM") as caught:
        run_train(run_dir, loops=loops, transport=transport)
    orphan = "a run may remain in Loops session session1; deactivate it in the Baseten UI"
    assert caught.value.hint == f"{orphan}; rerun with a new --run-dir"
    assert transport.calls == ["GET /v1/loops/capabilities"]
    result = read(run_dir, "result.json")
    assert (result["status"], result["cleanup"]["error"]) == ("failed", orphan)
    assert read(run_dir, "run-state.json")["cleanup_status"] == "failed"


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_signals_during_cleanup_are_ignored_until_the_result_is_written(
    run_dir: Path, signum: int
) -> None:
    handlers = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)

    def interrupted(request: httpx.Request) -> httpx.Response:
        os.kill(os.getpid(), signum)
        return httpx.Response(200, json=run_status("ACTIVE")[1])

    transport = baseten_transport(routes())
    transport.routes[f"GET {RUN}"] = [interrupted, run_status("INACTIVE")]
    result = run_train(run_dir, transport=transport)
    assert (result["status"], result["cleanup"]) == ("completed", CLEAN)
    assert f"POST {RUN}/deactivate" in transport.calls
    assert read(run_dir, "result.json")["status"] == "completed"
    assert read(run_dir, "run-state.json")["status"] == "completed"
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == handlers


# --- secrets -------------------------------------------------------------------------


@pytest.mark.parametrize("fail", [False, True], ids=["success", "refused"])
def test_the_key_never_reaches_output_or_files(
    run_dir: Path, capsys: pytest.CaptureFixture[str], fail: bool
) -> None:
    redaction.register_secret(SECRET)
    loops = FakeLoops()
    if fail:
        detail = {"detail": f"rejected key {SECRET}"}
        loops.fail_on("create_lora_training_client", error=http_status_error(400, json=detail))
    try:
        run_train(run_dir, loops=loops)
    except FinetuneError as exc:
        assert fail
        assert SECRET not in f"{exc} {exc.hint}"
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    files = [path for path in run_dir.parent.rglob("*") if path.is_file()]
    assert files
    for path in files:
        assert SECRET.encode() not in path.read_bytes()
    assert loops.kwargs("ServiceClient")[0]["api_key"] == SECRET
    assert "BASETEN_API_KEY" not in os.environ
