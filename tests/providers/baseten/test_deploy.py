# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.errors import ExternalError, StateError, UsageError
from pup_finetune.providers import DeployRequest
from pup_finetune.providers.baseten import deploy as serving
from pup_finetune.providers.baseten.client import BasetenHTTP, DeploymentRejected
from pup_finetune.providers.baseten.deploy import deploy, deploy_preview, undeploy
from pup_finetune.rendering import FAMILIES, ModelFamily, prompt_tokens
from pup_finetune.storage import DirectoryLock, atomic_write_json, read_json
from tests.fakes.baseten import BasetenTransport, FakeLoops, baseten_transport
from tests.fakes.baseten_serving import (
    BASE_MODEL,
    BASE_URL,
    CHECKPOINT_NAME,
    CHECKPOINT_PATH,
    DEPLOYMENT,
    DEPLOYMENT_ID,
    LOOKUP_CALL,
    MODEL_ID,
    OPENAI,
    FakeChat,
    FakeClock,
    checkpoint,
    created,
    dedicated_routes,
    deployment,
    instance,
    instances,
    sampler_routes,
    served_models,
    write_training_run,
)
from tests.fakes.tokenizer import FakeChatTokenizer

SECRET = "zq9Xsynthetic0000basetenKEYw7Pm"  # noqa: S105 - synthetic test value
KEY = SecretStr(SECRET)
CHECKPOINTS = "GET /v1/loops/checkpoints"
GRAPHQL = "POST /graphql/"
STATUS = f"GET {DEPLOYMENT}"
ACTIVATE = f"POST {DEPLOYMENT}/activate"
DEACTIVATE = f"POST {DEPLOYMENT}/deactivate"
MODELS = f"GET {OPENAI}/models"
CHAT = f"POST {OPENAI}/chat/completions"
SESSIONS = "POST /v1/loops/sessions"
SAMPLERS = "POST /v1/loops/samplers"
SMOKE = [MODELS, CHAT, CHAT, CHAT]


@pytest.fixture(autouse=True)
def _isolated() -> Iterator[None]:
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return write_training_run(tmp_path / "run")


def make_request(
    run_dir: Path,
    mode: str = "dedicated",
    *,
    accelerator: str | None = "H100:1",
    max_seq_len: int | None = None,
    timeout: float = 60.0,
) -> DeployRequest:
    return DeployRequest(
        run_dir=run_dir,
        serving_mode=mode,  # type: ignore[arg-type]
        accelerator=accelerator if mode == "dedicated" else None,
        max_seq_len=max_seq_len,
        timeout_seconds=timeout,
    )


def fake_tokenizer(family: ModelFamily) -> FakeChatTokenizer:
    return FakeChatTokenizer()


def run_deploy(
    request: DeployRequest,
    transport: BasetenTransport,
    *,
    loops: FakeLoops | None = None,
    clock: FakeClock | None = None,
    tokenizer_loader: Callable[[ModelFamily], Any] = fake_tokenizer,
) -> dict[str, Any]:
    clock = clock or FakeClock()
    with BasetenHTTP(KEY, transport=transport) as http:
        return deploy(
            request,
            api_key=KEY,
            http=http,
            loops=loops or FakeLoops(),
            tokenizer_loader=tokenizer_loader,
            sleep=clock.sleep,
            clock=clock,
        )


def run_undeploy(run_dir: Path, transport: BasetenTransport, clock: FakeClock | None = None) -> Any:
    clock = clock or FakeClock()
    with BasetenHTTP(KEY, transport=transport) as http:
        return undeploy(run_dir, http=http, sleep=clock.sleep, clock=clock)


def receipt(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = read_json(run_dir / "endpoint.json")
    return data


def recording(answer: tuple[int, Any], run_dir: Path, seen: list[Any]) -> Any:
    """``answer``, after recording the receipt as it is on disk when the request arrives."""

    def handle(request: httpx.Request) -> httpx.Response:
        path = run_dir / "endpoint.json"
        seen.append(read_json(path) if path.exists() else None)
        return httpx.Response(answer[0], json=answer[1])

    return handle


def assert_no_secret(run_dir: Path, *texts: str) -> None:
    for text in texts:
        assert SECRET not in text
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8", errors="replace")


def cleanup_command(run_dir: Path) -> str:
    return f"pup finetune undeploy --run-dir {run_dir.resolve()} --confirm"


def saved_receipt(run_dir: Path, mode: str, state: str, **fields: Any) -> dict[str, Any]:
    """Write the receipt an earlier deploy of ``mode`` left in ``state``."""
    identity = {
        "checkpoint_path": CHECKPOINT_PATH,
        "run_id": "run1",
        "checkpoint_name": CHECKPOINT_NAME,
        "base_model": BASE_MODEL,
    }
    data = {
        "schema_version": 1,
        "provider": "baseten",
        "owned": True,
        "serving_mode": mode,
        "checkpoint": identity,
        "settings": {"max_seq_len": 131_072}
        | ({"accelerator": "H100:1"} if mode == "dedicated" else {}),
        "state": state,
        "endpoint": {
            "model_id": MODEL_ID,
            "deployment_id": DEPLOYMENT_ID,
            "url": BASE_URL,
        },
        "served_model": CHECKPOINT_PATH if mode == "sampler" else CHECKPOINT_NAME,
        "openai_base_url": f"{BASE_URL}/v1",
        "cleanup_command": cleanup_command(run_dir),
    }
    if mode == "sampler":
        data |= {
            "session_id": "session1",
            "sampler": {
                "id": "smp1",
                "model_id": MODEL_ID,
                "deployment_id": DEPLOYMENT_ID,
                "base_url": BASE_URL,
            },
        }
    data |= fields
    atomic_write_json(run_dir / "endpoint.json", data)
    return data


# --- preview and settings ------------------------------------------------------------------


def test_preview_describes_a_dedicated_deployment_from_local_files(run_dir: Path) -> None:
    preview = deploy_preview(make_request(run_dir, max_seq_len=8192))
    assert preview == {
        "provider": "baseten",
        "action": "create",
        "serving_mode": "dedicated",
        "run_dir": str(run_dir.resolve()),
        "model": "qwen3p5-9b",
        "checkpoint_path": CHECKPOINT_PATH,
        "accelerator": "H100:1",
        "max_seq_len": 8192,
        "timeout_seconds": 60.0,
        "billing": "the dedicated deployment bills per hour until `pup finetune undeploy`",
        "cleanup_command": cleanup_command(run_dir),
    }


def test_preview_of_a_sampler_defaults_to_the_model_context(run_dir: Path) -> None:
    preview = deploy_preview(make_request(run_dir, "sampler"))
    assert (preview["accelerator"], preview["max_seq_len"]) == (None, 131_072)
    assert "Loops sampler bills per hour" in preview["billing"]


@pytest.mark.parametrize(
    ("mode", "fields", "message"),
    [
        ("dedicated", {"accelerator": None}, "needs --accelerator"),
        ("dedicated", {"accelerator": "h100:1"}, "needs --accelerator"),
        ("dedicated", {"accelerator": "H100"}, "needs --accelerator"),
        ("dedicated", {"accelerator": "H100:0"}, "needs --accelerator"),
        ("dedicated", {"max_seq_len": 1023}, "between 1024 and 131072"),
        ("sampler", {"max_seq_len": 131_073}, "between 1024 and 131072"),
        ("dedicated", {"timeout": 0.5}, "--timeout"),
        ("sampler", {"timeout": float("inf")}, "--timeout"),
        ("sampler", {"timeout": 7201.0}, "--timeout"),
        ("cloud", {}, "unknown serving mode"),
    ],
)
def test_invalid_settings_are_usage_errors(
    run_dir: Path, mode: str, fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(UsageError, match=message):
        deploy_preview(make_request(run_dir, mode, **fields))


def test_a_sampler_takes_no_accelerator(run_dir: Path) -> None:
    request = DeployRequest(run_dir, "sampler", "H100:1", None, 60.0)
    with pytest.raises(UsageError, match="only to --serving-mode dedicated"):
        deploy_preview(request)


@pytest.mark.parametrize(
    ("plan", "result", "message"),
    [
        ({"provider": "other"}, {}, "not a Baseten training run"),
        ({}, {"provider": "other"}, "not a Baseten training run"),
        ({}, {"best_sampler_weights_uri": None}, "no best sampler checkpoint"),
        ({}, {"best_sampler_weights_uri": "bt://loops:run1/state/x"}, "no best sampler"),
        ({}, {"baseten_run_id": "run2"}, "no best sampler checkpoint"),
        ({"model": "unknown"}, {}, "unknown Baseten model"),
    ],
)
def test_preview_needs_a_baseten_run_with_a_best_checkpoint(
    tmp_path: Path, plan: dict[str, Any], result: dict[str, Any], message: str
) -> None:
    run_dir = write_training_run(tmp_path / "run", plan=plan, result=result)
    with pytest.raises(UsageError, match=message):
        deploy_preview(make_request(run_dir))


def test_preview_refuses_a_plan_whose_base_model_differs(tmp_path: Path) -> None:
    run_dir = write_training_run(tmp_path / "run", plan={"base_model": "other/model"})
    with pytest.raises(StateError, match="base model"):
        deploy_preview(make_request(run_dir))


def test_preview_of_a_directory_without_a_run(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="does not exist"):
        deploy_preview(make_request(tmp_path))


def test_preview_resumes_the_same_deployment_and_refuses_a_different_live_one(
    run_dir: Path,
) -> None:
    saved_receipt(run_dir, "dedicated", "needs_attention")
    assert deploy_preview(make_request(run_dir))["action"] == "resume"
    with pytest.raises(StateError, match="records a different deployment") as caught:
        deploy_preview(make_request(run_dir, accelerator="H100:2"))
    assert caught.value.hint == (
        f"stop it first with `{cleanup_command(run_dir)}`, then rerun deploy"
    )
    with pytest.raises(StateError, match="different deployment"):
        deploy_preview(make_request(run_dir, "sampler"))


def test_preview_replaces_a_different_deployment_once_it_is_inactive(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "inactive")
    assert deploy_preview(make_request(run_dir, "sampler"))["action"] == "create"


def test_preview_refuses_a_receipt_with_an_unknown_outcome(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "creation_unknown", endpoint=None, model_name="pup-x")
    with pytest.raises(StateError, match="unknown outcome") as caught:
        deploy_preview(make_request(run_dir))
    assert "a model named pup-x" in str(caught.value.hint)


def test_preview_refuses_a_foreign_receipt(run_dir: Path) -> None:
    atomic_write_json(run_dir / "endpoint.json", {"schema_version": 1, "owned": False})
    with pytest.raises(StateError, match="not a deployment receipt"):
        deploy_preview(make_request(run_dir))


# --- dedicated -----------------------------------------------------------------------------


def test_dedicated_deploy_creates_waits_and_smoke_tests(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    at_create: list[Any] = []
    at_status: list[Any] = []
    listed = instances(instance("H100", 2), instance("H100", 1, node_count=2), instance("H100", 1))
    chat = FakeChat()
    routes = dedicated_routes(
        listed=listed, create=recording(created(), run_dir, at_create), chat=chat
    )
    routes[STATUS] = [
        recording(deployment("DEPLOYING"), run_dir, at_status),
        deployment("BUILDING"),
        deployment("ACTIVE"),
    ]
    transport = baseten_transport(routes)
    clock = FakeClock()

    result = run_deploy(make_request(run_dir), transport, clock=clock)

    assert transport.calls == [CHECKPOINTS, GRAPHQL, GRAPHQL, STATUS, STATUS, STATUS, *SMOKE]
    listing, create = transport.bodies(GRAPHQL)
    assert "listed_instances" in listing["query"]
    sent = create["variables"]["request"]
    assert sent["instance_type_id"] == "H100:1x26x234"
    assert sent["weights_sources"][0]["b10_loops_checkpoint_weights_source"] == {
        "checkpoint": {"loops_checkpoint_id": "ckpt1"}
    }
    model_name = sent["metadata"]["oracle_name"]
    assert model_name.startswith("pup-finetune-")
    # The intent is on disk before the paid create, and the IDs before the wait.
    assert at_create[0]["state"] == "creating"
    assert at_create[0]["model_name"] == model_name
    assert "endpoint" not in at_create[0]
    assert at_status[0]["state"] == "created"
    assert at_status[0]["endpoint"]["deployment_id"] == DEPLOYMENT_ID
    assert clock.sleeps == [5.0]

    openai_base_url = f"{BASE_URL}/v1"
    assert result == {
        "state": "ready",
        "serving_mode": "dedicated",
        "openai_base_url": openai_base_url,
        "served_model": CHECKPOINT_NAME,
        "endpoint": {"model_id": MODEL_ID, "deployment_id": DEPLOYMENT_ID, "url": BASE_URL},
        "advertised_max_seq_len": 131_072,
        "cleanup_command": cleanup_command(run_dir),
    }
    saved = receipt(run_dir)
    assert saved == {
        **result,
        "schema_version": 1,
        "provider": "baseten",
        "owned": True,
        "model_name": model_name,
        "checkpoint": {
            "checkpoint_path": CHECKPOINT_PATH,
            "run_id": "run1",
            "checkpoint_name": CHECKPOINT_NAME,
            "base_model": BASE_MODEL,
            "checkpoint_id": "ckpt1",
        },
        "settings": {"accelerator": "H100:1", "max_seq_len": 131_072},
    }
    assert [body["model"] for body in chat.bodies] == [CHECKPOINT_NAME] * 3
    text, tool, tool_result = chat.bodies
    assert "tools" not in text
    assert tool["tool_choice"] == "auto"
    assert tool["tools"][0]["function"]["name"] == "lookup_number"
    assert tool_result["messages"][-2:] == [
        {"role": "assistant", "content": None, "tool_calls": [LOOKUP_CALL]},
        {"role": "tool", "tool_call_id": "call_0", "content": "The value is seven."},
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    for line in (
        f"OpenAI base URL: {openai_base_url}",
        f"model: {CHECKPOINT_NAME}",
        f"Baseten model {MODEL_ID}, deployment {DEPLOYMENT_ID}",
        f"stop billing with: {cleanup_command(run_dir)}",
    ):
        assert line in captured.err
    assert all(r.headers["Authorization"] == f"Api-Key {SECRET}" for r in transport.requests)
    assert_no_secret(run_dir, captured.out, captured.err)


def test_dedicated_deploy_refuses_any_instance_but_an_exact_match(run_dir: Path) -> None:
    listed = instances(
        instance("H100", 2),
        instance("H100", 8),
        instance("H100", 1, node_count=2),
        instance("A100", 1),
        instance("CPU", 0),
    )
    transport = baseten_transport(dedicated_routes(listed=listed))
    with pytest.raises(UsageError, match="exactly H100:1") as caught:
        run_deploy(make_request(run_dir), transport)
    assert caught.value.hint == "available: H100:2, H100:8"
    assert transport.calls == [CHECKPOINTS, GRAPHQL]
    assert not (run_dir / "endpoint.json").exists()

    with pytest.raises(UsageError) as other:
        run_deploy(make_request(run_dir, accelerator="B200:1"), transport)
    assert other.value.hint == "available: A100:1, H100:2, H100:8"


def test_a_bdn_rejection_removes_the_receipt_and_suggests_sampler_mode(run_dir: Path) -> None:
    message = "bdn is set, but BDN volumes are not enabled for this organization"
    at_create: list[Any] = []
    rejected = recording((200, {"errors": [{"message": message}]}), run_dir, at_create)
    transport = baseten_transport(dedicated_routes(create=rejected))
    with pytest.raises(DeploymentRejected, match="BDN volumes are not enabled") as caught:
        run_deploy(make_request(run_dir), transport)
    assert caught.value.hint is not None
    assert "nothing was created; rerun with --serving-mode sampler" in caught.value.hint
    assert at_create[0]["state"] == "creating"
    assert not (run_dir / "endpoint.json").exists()
    assert transport.calls == [CHECKPOINTS, GRAPHQL, GRAPHQL]


def test_another_rejection_also_removes_the_receipt(run_dir: Path) -> None:
    rejected = (200, {"errors": [{"message": "quota exceeded"}]})
    transport = baseten_transport(dedicated_routes(create=rejected))
    with pytest.raises(DeploymentRejected) as caught:
        run_deploy(make_request(run_dir), transport)
    assert caught.value.hint == "nothing was created; fix the cause and rerun deploy"
    assert not (run_dir / "endpoint.json").exists()


@pytest.mark.parametrize("create", [(500, {}), created(model_id="Not-Valid")])
def test_a_create_with_an_unknown_outcome_keeps_the_receipt_and_blocks_reruns(
    run_dir: Path, create: tuple[int, Any]
) -> None:
    transport = baseten_transport(dedicated_routes(create=create))
    with pytest.raises(ExternalError) as caught:
        run_deploy(make_request(run_dir), transport)
    saved = receipt(run_dir)
    assert saved["state"] == "creation_unknown"
    assert "endpoint" not in saved
    assert f"a model named {saved['model_name']}" in str(caught.value.hint)

    again = baseten_transport({})
    with pytest.raises(StateError, match="unknown outcome"):
        run_deploy(make_request(run_dir), again)
    assert again.calls == []


def test_a_failed_deployment_needs_attention_and_is_kept(run_dir: Path) -> None:
    routes = dedicated_routes()
    routes[STATUS] = [deployment("DEPLOYING"), deployment("BUILD_FAILED")]
    transport = baseten_transport(routes)
    with pytest.raises(ExternalError, match="is BUILD_FAILED") as caught:
        run_deploy(make_request(run_dir), transport)
    assert caught.value.hint == (
        f"rerun deploy to resume, or stop it with `{cleanup_command(run_dir)}`"
    )
    saved = receipt(run_dir)
    assert saved["state"] == "needs_attention"
    assert saved["endpoint"]["model_id"] == MODEL_ID
    assert DEACTIVATE not in transport.calls


def test_waiting_times_out(run_dir: Path) -> None:
    routes = dedicated_routes()
    routes[STATUS] = deployment("DEPLOYING")
    clock = FakeClock()
    with pytest.raises(ExternalError, match="not ready after 12 s"):
        run_deploy(make_request(run_dir, timeout=12.0), baseten_transport(routes), clock=clock)
    assert clock.sleeps == [5.0, 5.0, 2.0]
    assert receipt(run_dir)["state"] == "needs_attention"


def test_a_deployment_that_disappears_while_waiting_fails(run_dir: Path) -> None:
    routes = dedicated_routes()
    routes[STATUS] = [deployment("DEPLOYING"), (404, {})]
    with pytest.raises(ExternalError, match="no longer has the deployment"):
        run_deploy(make_request(run_dir), baseten_transport(routes))


def test_a_mismatched_status_answer_fails(run_dir: Path) -> None:
    routes = dedicated_routes()
    routes[STATUS] = (200, {"id": "other", "model_id": MODEL_ID, "status": "ACTIVE"})
    with pytest.raises(ExternalError, match="different deployment"):
        run_deploy(make_request(run_dir), baseten_transport(routes))


def test_rerunning_resumes_and_activates_an_inactive_deployment(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "inactive", model_name="pup-finetune-old")
    routes = dedicated_routes()
    routes[STATUS] = [deployment("INACTIVE"), deployment("INACTIVE"), deployment("ACTIVE")]
    transport = baseten_transport(routes)
    result = run_deploy(make_request(run_dir), transport)
    assert transport.calls == [STATUS, STATUS, ACTIVATE, STATUS, *SMOKE]
    assert result["state"] == "ready"
    assert receipt(run_dir)["model_name"] == "pup-finetune-old"


def test_rerunning_creates_anew_when_the_deployment_is_gone(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "needs_attention", model_name="pup-finetune-old")
    routes = dedicated_routes()
    routes[STATUS] = [(404, {}), deployment("ACTIVE")]
    transport = baseten_transport(routes)
    run_deploy(make_request(run_dir), transport)
    assert transport.calls[:4] == [STATUS, CHECKPOINTS, GRAPHQL, GRAPHQL]
    assert receipt(run_dir)["model_name"] != "pup-finetune-old"


def test_a_different_live_deployment_is_refused_before_any_call(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "ready")
    transport = baseten_transport({})
    with pytest.raises(StateError, match="different deployment"):
        run_deploy(make_request(run_dir, max_seq_len=4096), transport)
    assert transport.calls == []


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"run_id": "run2"}, "does not match"),
        ({"checkpoint_id": "other"}, "does not match"),
        ({"base_model": "other/model"}, "does not match"),
        ({"target": "training"}, "does not match"),
        ({"id": "bad id!"}, "does not match"),
        ({"sync_status": "SYNCING"}, "still syncing"),
    ],
)
@pytest.mark.parametrize("mode", ["dedicated", "sampler"])
def test_the_checkpoint_must_match_and_be_synced(
    run_dir: Path, mode: str, fields: dict[str, Any], message: str
) -> None:
    routes = dedicated_routes() if mode == "dedicated" else sampler_routes()
    routes[CHECKPOINTS] = checkpoint(**fields)
    transport = baseten_transport(routes)
    with pytest.raises(ExternalError, match=message):
        run_deploy(make_request(run_dir, mode), transport)
    assert transport.calls == [CHECKPOINTS]
    assert not (run_dir / "endpoint.json").exists()


def test_a_synced_checkpoint_may_have_no_sync_status(run_dir: Path) -> None:
    routes = dedicated_routes()
    routes[CHECKPOINTS] = checkpoint(sync_status=None)
    assert run_deploy(make_request(run_dir), baseten_transport(routes))["state"] == "ready"


# --- sampler -------------------------------------------------------------------------------


def test_sampler_deploy_saves_ids_loads_the_checkpoint_and_smoke_tests(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    at_sampler: list[Any] = []
    at_status: list[Any] = []
    chat = FakeChat()
    routes = sampler_routes(chat=chat)
    routes[SAMPLERS] = recording(routes[SAMPLERS], run_dir, at_sampler)
    routes[STATUS] = [recording(deployment("DEPLOYING"), run_dir, at_status), deployment("ACTIVE")]
    transport = baseten_transport(routes)
    loops = FakeLoops()
    clock = FakeClock()

    result = run_deploy(make_request(run_dir, "sampler"), transport, loops=loops, clock=clock)

    assert transport.calls == [CHECKPOINTS, SESSIONS, SAMPLERS, STATUS, STATUS, *SMOKE]
    assert transport.bodies(SAMPLERS) == [
        {
            "session_id": "session1",
            "model_path": CHECKPOINT_PATH,
            "max_seq_length": 131_072,
            "availability_model": "dedicated",
        }
    ]
    assert at_sampler[0]["state"] == "creating"
    assert at_sampler[0]["session_id"] == "session1"
    assert at_status[0]["state"] == "waiting"
    assert at_status[0]["sampler"]["id"] == "smp1"
    assert at_status[0]["endpoint"]["deployment_id"] == DEPLOYMENT_ID
    assert clock.sleeps == [5.0]

    assert loops.methods == ["SamplingClient", "sample", "sampler_close"]
    (client,) = loops.kwargs("SamplingClient")
    assert client["api_key"] == SECRET
    assert (client["sampler_id"], client["session_id"]) == ("smp1", "session1")
    assert (client["checkpoint_path"], client["base_model"]) == (CHECKPOINT_PATH, BASE_MODEL)
    assert client["deployment"].base_url == BASE_URL
    assert client["ready_timeout"] == 60.0
    (sample,) = loops.kwargs("sample")
    hello = [{"role": "user", "content": "Hello"}]
    expected = prompt_tokens(hello, None, FAMILIES["qwen3_5"], FakeChatTokenizer())
    assert sample["prompt"].tokens == expected
    assert sample["sampling_params"].max_tokens == 1

    assert result == {
        "state": "ready",
        "serving_mode": "sampler",
        "openai_base_url": f"{BASE_URL}/v1",
        "served_model": CHECKPOINT_PATH,
        "endpoint": {"model_id": MODEL_ID, "deployment_id": DEPLOYMENT_ID, "url": BASE_URL},
        "advertised_max_seq_len": 131_072,
        "cleanup_command": cleanup_command(run_dir),
    }
    saved = receipt(run_dir)
    assert saved["sampler"] == {
        "id": "smp1",
        "model_id": MODEL_ID,
        "deployment_id": DEPLOYMENT_ID,
        "base_url": BASE_URL,
    }
    assert saved["settings"] == {"max_seq_len": 131_072}
    assert "checkpoint_id" not in saved["checkpoint"]
    assert [body["model"] for body in chat.bodies] == [CHECKPOINT_PATH] * 3
    captured = capsys.readouterr()
    assert f"model: {CHECKPOINT_PATH}" in captured.err
    assert_no_secret(run_dir, captured.out, captured.err)


def test_a_sampler_that_fails_its_smoke_test_is_stopped(run_dir: Path) -> None:
    routes = sampler_routes()
    routes[MODELS] = served_models({"id": "Qwen/Qwen3.5-9B", "max_model_len": 131_072})
    routes[STATUS] = [deployment("ACTIVE"), deployment("ACTIVE"), deployment("INACTIVE")]
    transport = baseten_transport(routes)
    with pytest.raises(ExternalError, match=f"does not list {CHECKPOINT_PATH}") as caught:
        run_deploy(make_request(run_dir, "sampler"), transport)
    assert caught.value.hint == "the sampler was stopped; rerun deploy to start a new one"
    assert transport.calls[-4:] == [MODELS, STATUS, DEACTIVATE, STATUS]
    assert receipt(run_dir)["state"] == "inactive"


def test_a_sampler_whose_checkpoint_does_not_load_is_stopped(run_dir: Path) -> None:
    loops = FakeLoops()
    loops.fail_on("sample", error=RuntimeError("boom"))
    routes = sampler_routes()
    routes[STATUS] = [deployment("ACTIVE"), deployment("ACTIVE"), deployment("INACTIVE")]
    transport = baseten_transport(routes)
    with pytest.raises(ExternalError, match="loading the checkpoint"):
        run_deploy(make_request(run_dir, "sampler"), transport, loops=loops)
    assert loops.methods == ["SamplingClient", "sample", "sampler_close"]
    assert transport.calls.count(DEACTIVATE) == 1
    assert MODELS not in transport.calls
    assert receipt(run_dir)["state"] == "inactive"


def test_a_sampler_with_an_unexpected_url_is_stopped_before_waiting(run_dir: Path) -> None:
    routes = sampler_routes(base_url="https://elsewhere.example.com/sync")
    routes[STATUS] = [deployment("ACTIVE"), deployment("INACTIVE")]
    transport = baseten_transport(routes)
    with pytest.raises(ExternalError, match="sampler unexpectedly"):
        run_deploy(make_request(run_dir, "sampler"), transport)
    assert transport.calls == [CHECKPOINTS, SESSIONS, SAMPLERS, STATUS, DEACTIVATE, STATUS]
    assert receipt(run_dir)["sampler"]["base_url"] == "https://elsewhere.example.com/sync"


def test_a_failed_sampler_cleanup_names_the_cleanup_command(run_dir: Path) -> None:
    routes = sampler_routes()
    routes[STATUS] = [deployment("DEPLOYING"), deployment("FAILED"), deployment("ACTIVE")]
    routes[DEACTIVATE] = (500, {})
    with pytest.raises(ExternalError, match="may still bill") as caught:
        run_deploy(make_request(run_dir, "sampler"), baseten_transport(routes))
    assert "the Baseten deployment is FAILED" in caught.value.message
    assert caught.value.hint == f"stop it with `{cleanup_command(run_dir)}`"
    assert isinstance(caught.value.__cause__, ExternalError)
    assert receipt(run_dir)["state"] == "cleanup_required"


def test_an_interrupted_sampler_wait_stops_the_sampler(run_dir: Path) -> None:
    routes = sampler_routes()
    routes[STATUS] = [deployment("DEPLOYING"), deployment("ACTIVE"), deployment("INACTIVE")]
    clock = FakeClock()

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    with (
        BasetenHTTP(KEY, transport=baseten_transport(routes)) as http,
        pytest.raises(KeyboardInterrupt),
    ):
        deploy(
            make_request(run_dir, "sampler"),
            api_key=KEY,
            http=http,
            loops=FakeLoops(),
            tokenizer_loader=fake_tokenizer,
            sleep=interrupt,
            clock=clock,
        )
    assert receipt(run_dir)["state"] == "inactive"


def test_invalid_sampler_ids_leave_an_unknown_outcome(run_dir: Path) -> None:
    transport = baseten_transport(sampler_routes(model_id="NOT VALID"))
    with pytest.raises(ExternalError, match="no valid deployment IDs") as caught:
        run_deploy(make_request(run_dir, "sampler"), transport)
    saved = receipt(run_dir)
    assert saved["state"] == "creation_unknown"
    assert saved["sampler"]["model_id"] == "NOT VALID"
    assert "Loops session session1" in str(caught.value.hint)
    assert DEACTIVATE not in transport.calls


def test_a_failed_sampler_create_leaves_an_unknown_outcome(run_dir: Path) -> None:
    routes = sampler_routes()
    routes[SAMPLERS] = (502, {})
    with pytest.raises(ExternalError, match="HTTP 502"):
        run_deploy(make_request(run_dir, "sampler"), baseten_transport(routes))
    saved = receipt(run_dir)
    assert (saved["state"], saved["session_id"]) == ("creation_unknown", "session1")


def test_a_refused_sampler_create_removes_the_receipt(run_dir: Path) -> None:
    routes = sampler_routes()
    with pytest.raises(DeploymentRejected, match="HTTP 429") as caught:
        run_deploy(
            make_request(run_dir, "sampler"), baseten_transport(routes | {SAMPLERS: (429, {})})
        )
    assert caught.value.hint == "nothing was created; fix the cause and rerun deploy"
    assert not (run_dir / "endpoint.json").exists()
    assert (
        run_deploy(make_request(run_dir, "sampler"), baseten_transport(routes))["state"] == "ready"
    )


def test_a_failed_session_create_writes_no_receipt(run_dir: Path) -> None:
    routes = sampler_routes()
    routes[SESSIONS] = (500, {})
    with pytest.raises(ExternalError, match="session creation returned HTTP 500"):
        run_deploy(make_request(run_dir, "sampler"), baseten_transport(routes))
    assert not (run_dir / "endpoint.json").exists()


def test_sampler_mode_needs_the_tokenizer_before_anything_is_paid(run_dir: Path) -> None:
    def missing(family: ModelFamily) -> Any:
        raise UsageError("transformers is not installed")

    transport = baseten_transport(sampler_routes())
    with pytest.raises(UsageError, match="transformers"):
        run_deploy(make_request(run_dir, "sampler"), transport, tokenizer_loader=missing)
    assert transport.calls == []


def test_sampler_mode_refuses_loops_overrides_before_anything_is_paid(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOPS_TEAM", "elsewhere")
    transport = baseten_transport(sampler_routes())
    with pytest.raises(UsageError, match="LOOPS_TEAM"):
        run_deploy(make_request(run_dir, "sampler"), transport)
    assert transport.calls == []


def test_rerunning_a_stopped_sampler_starts_a_new_session(run_dir: Path) -> None:
    saved_receipt(run_dir, "sampler", "inactive", session_id="session0")
    routes = sampler_routes()
    routes[STATUS] = [deployment("INACTIVE"), deployment("ACTIVE")]
    transport = baseten_transport(routes)
    run_deploy(make_request(run_dir, "sampler"), transport)
    assert transport.calls[:4] == [STATUS, CHECKPOINTS, SESSIONS, SAMPLERS]
    assert receipt(run_dir)["session_id"] == "session1"


def test_rerunning_resumes_a_live_sampler_and_reloads_its_checkpoint(run_dir: Path) -> None:
    saved_receipt(run_dir, "sampler", "needs_attention")
    routes = sampler_routes()
    routes[STATUS] = deployment("SCALED_TO_ZERO")
    transport = baseten_transport(routes)
    loops = FakeLoops()
    result = run_deploy(make_request(run_dir, "sampler"), transport, loops=loops)
    assert transport.calls == [STATUS, STATUS, *SMOKE]
    assert loops.methods == ["SamplingClient", "sample", "sampler_close"]
    assert result["state"] == "ready"


# --- smoke test ----------------------------------------------------------------------------


def smoke(routes: dict[str, Any], max_seq_len: int = 4096) -> int:
    with BasetenHTTP(KEY, transport=baseten_transport(routes)) as http:
        return serving._smoke(http, f"{BASE_URL}/v1", "ckpt", max_seq_len)


def smoke_routes(chat: FakeChat | None = None, *models: dict[str, Any]) -> dict[str, Any]:
    return {
        MODELS: served_models(*(models or ({"id": "ckpt", "max_model_len": 8192},))),
        CHAT: chat or FakeChat(),
    }


def test_smoke_reads_a_lora_context_from_its_parent() -> None:
    lora = {"id": "ckpt", "max_model_len": None, "parent": "base"}
    assert smoke(smoke_routes(None, lora, {"id": "base", "max_model_len": 16384})) == 16384


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ([{"id": "other", "max_model_len": 8192}], "does not list ckpt"),
        ([{"id": "ckpt", "max_model_len": 8192}] * 2, "does not list ckpt"),
        ([{"id": "ckpt"}], "advertises no context"),
        ([{"id": "ckpt", "max_model_len": "8192"}], "advertises no context"),
        ([{"id": "ckpt", "parent": "missing"}], "advertises no context"),
    ],
)
def test_smoke_needs_one_listed_model_with_a_context(
    models: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ExternalError, match=message):
        smoke(smoke_routes(None, *models))


def test_smoke_needs_the_requested_context() -> None:
    with pytest.raises(ExternalError, match="8192 tokens of context") as caught:
        smoke(smoke_routes(), max_seq_len=16384)
    assert caught.value.hint == "undeploy, then deploy again with --max-seq-len 8192"


@pytest.mark.parametrize(
    ("chat", "message"),
    [
        (FakeChat(text=""), "text smoke test got no visible answer"),
        (FakeChat(text=None), "text smoke test got no visible answer"),
        (FakeChat(final="  "), "tool-result smoke test got no visible answer"),
        (
            FakeChat(tool_reply={"content": "<tool_call>lookup_number 7</tool_call>"}),
            "only as text",
        ),
        (FakeChat(tool_reply={"content": "seven"}), "one lookup_number call"),
        (
            FakeChat(tool_reply={"tool_calls": [{**LOOKUP_CALL, "id": ""}]}),
            "one lookup_number call",
        ),
        (
            FakeChat(
                tool_reply={"tool_calls": [{**LOOKUP_CALL, "function": {"name": "lookup_number"}}]}
            ),
            "one lookup_number call",
        ),
        (
            FakeChat(
                tool_reply={
                    "tool_calls": [
                        {**LOOKUP_CALL, "function": {**LOOKUP_CALL["function"], "arguments": "{"}}
                    ]
                }
            ),
            "one lookup_number call",
        ),
        (FakeChat(tool_reply={"tool_calls": [LOOKUP_CALL, LOOKUP_CALL]}), "one lookup_number"),
    ],
)
def test_smoke_failures(chat: FakeChat, message: str) -> None:
    with pytest.raises(ExternalError, match=message):
        smoke(smoke_routes(chat))


def test_smoke_refuses_a_text_reply_with_tool_calls() -> None:
    class ToolsAlways(FakeChat):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            message = {"role": "assistant", "content": "hi", "tool_calls": [LOOKUP_CALL]}
            return httpx.Response(200, json={"choices": [{"message": message}]})

    with pytest.raises(ExternalError, match="text smoke test got no visible answer"):
        smoke(smoke_routes(ToolsAlways()))


def test_smoke_refuses_a_reply_without_a_message() -> None:
    routes = smoke_routes()
    routes[CHAT] = (200, {"choices": []})
    with pytest.raises(ExternalError, match="text smoke test got no message"):
        smoke(routes)


def test_smoke_warns_about_reasoning_markup(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke(smoke_routes(FakeChat(text="<think>\n</think>\nhello"))) == 8192
    assert "reasoning markup" in capsys.readouterr().err


# --- undeploy ------------------------------------------------------------------------------


def test_undeploy_deactivates_once_and_keeps_the_receipt(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved_receipt(run_dir, "sampler", "ready")
    at_deactivate: list[Any] = []
    transport = baseten_transport(
        {
            STATUS: [deployment("ACTIVE"), deployment("DEACTIVATING"), deployment("INACTIVE")],
            DEACTIVATE: recording((200, {}), run_dir, at_deactivate),
        }
    )
    clock = FakeClock()
    result = run_undeploy(run_dir, transport, clock)
    assert transport.calls == [STATUS, DEACTIVATE, STATUS, STATUS]
    assert at_deactivate[0]["state"] == "deactivating"
    assert clock.sleeps == [5.0]
    assert result == {
        "state": "inactive",
        "serving_mode": "sampler",
        "endpoint": {"model_id": MODEL_ID, "deployment_id": DEPLOYMENT_ID, "url": BASE_URL},
        "checkpoint_path": CHECKPOINT_PATH,
        "receipt": str((run_dir / "endpoint.json").resolve()),
    }
    saved = receipt(run_dir)
    assert saved["state"] == "inactive"
    assert saved["sampler"]["id"] == "smp1"
    assert_no_secret(run_dir, *capsys.readouterr())


@pytest.mark.parametrize("answer", [(404, {}), deployment("INACTIVE")])
def test_undeploy_of_a_stopped_or_missing_deployment_posts_nothing(
    run_dir: Path, answer: tuple[int, Any]
) -> None:
    saved_receipt(run_dir, "dedicated", "ready")
    for _ in range(2):
        transport = baseten_transport({STATUS: answer})
        assert run_undeploy(run_dir, transport)["state"] == "inactive"
        assert transport.calls == [STATUS]


def test_undeploy_does_not_deactivate_twice(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "deactivating")
    transport = baseten_transport({STATUS: [deployment("DEACTIVATING"), (404, {})]})
    assert run_undeploy(run_dir, transport)["state"] == "inactive"
    assert transport.calls == [STATUS, STATUS]


def test_a_failed_undeploy_requires_cleanup(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "ready")
    transport = baseten_transport({STATUS: deployment("ACTIVE"), DEACTIVATE: (500, {})})
    with pytest.raises(ExternalError, match="deactivation returned HTTP 500"):
        run_undeploy(run_dir, transport)
    assert receipt(run_dir)["state"] == "cleanup_required"


def test_undeploy_gives_up_after_two_minutes(run_dir: Path) -> None:
    saved_receipt(run_dir, "dedicated", "ready")
    transport = baseten_transport({STATUS: deployment("ACTIVE"), DEACTIVATE: (200, {})})
    clock = FakeClock()
    with pytest.raises(ExternalError, match="not inactive after 120 s"):
        run_undeploy(run_dir, transport, clock)
    assert sum(clock.sleeps) == 120.0
    assert transport.calls.count(DEACTIVATE) == 1
    assert receipt(run_dir)["state"] == "cleanup_required"


def test_undeploy_needs_a_receipt(run_dir: Path) -> None:
    with pytest.raises(StateError, match="records no deployment"):
        run_undeploy(run_dir, baseten_transport({}))


@pytest.mark.parametrize("endpoint", [None, {"model_id": "Not-Valid", "deployment_id": "dep1"}])
def test_undeploy_of_an_unknown_outcome_points_to_the_baseten_ui(
    run_dir: Path, endpoint: Any
) -> None:
    saved_receipt(run_dir, "dedicated", "creation_unknown", endpoint=endpoint, model_name="pup-x")
    transport = baseten_transport({})
    with pytest.raises(StateError, match="unknown outcome") as caught:
        run_undeploy(run_dir, transport)
    assert "a model named pup-x" in str(caught.value.hint)
    assert transport.calls == []
    assert receipt(run_dir)["state"] == "creation_unknown"


# --- locking ---------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["deploy", "undeploy"])
def test_commands_refuse_a_run_dir_in_use(run_dir: Path, command: str) -> None:
    saved_receipt(run_dir, "dedicated", "ready")
    transport = baseten_transport({})
    with DirectoryLock(run_dir), pytest.raises(StateError, match="another pup-finetune process"):
        if command == "deploy":
            run_deploy(make_request(run_dir), transport)
        else:
            run_undeploy(run_dir, transport)
    assert transport.calls == []


def test_the_receipt_is_written_privately(run_dir: Path) -> None:
    run_deploy(make_request(run_dir), baseten_transport(dedicated_routes()))
    assert (run_dir / "endpoint.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((run_dir / "endpoint.json").read_text())["state"] == "ready"
