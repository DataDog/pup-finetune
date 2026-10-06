# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Serving fakes: a finished training run on disk, a fake clock and healthy Baseten routes.

``dedicated_routes()`` and ``sampler_routes()`` answer a deployment that becomes ready and
passes every smoke test; tests override single routes for the branch they exercise::

    run_dir = write_training_run(tmp_path / "run")
    clock = FakeClock()  # pass clock= and sleep=clock.sleep; clock.sleeps records each sleep
    transport = baseten_transport(sampler_routes() | {f"GET {DEPLOYMENT}": deployment("FAILED")})

``FakeChat`` is the OpenAI chat route: a text reply, then one ``lookup_number`` call, then a
reply to the tool result. It records every request body in ``bodies``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from pup_finetune.storage import atomic_write_json, ensure_private_dir
from tests.fakes.baseten import Answer, graphql

MODEL_ID = "mdl1"
DEPLOYMENT_ID = "dep1"
DEPLOYMENT = f"/v1/models/{MODEL_ID}/deployments/{DEPLOYMENT_ID}"
BASE_URL = f"https://model-{MODEL_ID}.api.baseten.co/deployment/{DEPLOYMENT_ID}/sync"
OPENAI = f"/deployment/{DEPLOYMENT_ID}/sync/v1"
RUN_ID = "run1"
CHECKPOINT_NAME = "sft-test-sampler-epoch-2"
CHECKPOINT_PATH = f"bt://loops:{RUN_ID}/sampler_weights/{CHECKPOINT_NAME}"
BASE_MODEL = "Qwen/Qwen3.5-9B"
LOOKUP_CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "lookup_number", "arguments": '{"number": 7}'},
}


def write_training_run(
    run_dir: Path,
    *,
    plan: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
) -> Path:
    """A finished Baseten run with a best sampler checkpoint; ``plan``/``result`` override keys."""
    ensure_private_dir(run_dir)
    base_plan = {"provider": "baseten", "model": "qwen3p5-9b", "base_model": BASE_MODEL}
    atomic_write_json(run_dir / "plan.json", {"run_id": "sft-test", **base_plan, **(plan or {})})
    base_result = {
        "status": "completed",
        "provider": "baseten",
        "baseten_run_id": RUN_ID,
        "best_sampler_weights_uri": CHECKPOINT_PATH,
    }
    atomic_write_json(run_dir / "result.json", {**base_result, **(result or {})})
    return run_dir


class FakeClock:
    """A monotonic clock that only moves when ``sleep`` is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def deployment(status: str) -> tuple[int, dict[str, Any]]:
    return 200, {"id": DEPLOYMENT_ID, "model_id": MODEL_ID, "status": status}


def checkpoint(**fields: Any) -> tuple[int, dict[str, Any]]:
    record = {
        "id": "ckpt1",
        "run_id": RUN_ID,
        "checkpoint_id": CHECKPOINT_NAME,
        "base_model": BASE_MODEL,
        "target": "sampler",
        "sync_status": "COMPLETE",
    }
    return 200, {"checkpoints": [{**record, **fields}]}


def instance(gpu_type: str, gpu_count: int, *, node_count: int = 1) -> dict[str, Any]:
    """An instance type with an id in Baseten's format, e.g. ``H100:1x26x234``."""
    return {
        "id": f"{gpu_type}:{gpu_count}x26x234" + (f"x{node_count}n" if node_count > 1 else ""),
        "gpuType": gpu_type,
        "gpuCount": gpu_count,
        "nodeCount": node_count,
    }


def instances(*listed: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return 200, {"data": {"listedInstances": list(listed)}}


def created(model_id: str = MODEL_ID, deployment_id: str = DEPLOYMENT_ID) -> tuple[int, Any]:
    version = {"id": deployment_id, "name": "v1", "model_id": model_id}
    payload = {"model_version": version, "truss_config": "{}"}
    return 200, {"data": {"create_model_version_from_inference_template": payload}}


def served_models(*entries: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return 200, {"object": "list", "data": list(entries)}


class FakeChat:
    """The OpenAI chat route of a deployment that passes the smoke test unless told otherwise."""

    def __init__(
        self,
        *,
        text: Any = "hello",
        tool_reply: Mapping[str, Any] | None = None,
        final: Any = "The value is seven.",
    ) -> None:
        self.text = text
        self.tool_reply = dict(
            tool_reply or {"role": "assistant", "content": None, "tool_calls": [LOOKUP_CALL]}
        )
        self.final = final
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body["messages"][-1]["role"] == "tool":
            message: Mapping[str, Any] = {"role": "assistant", "content": self.final}
        elif "tools" in body:
            message = self.tool_reply
        else:
            message = {"role": "assistant", "content": self.text}
        return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})


def serving_routes(served_model: str, chat: FakeChat | None = None) -> dict[str, Any]:
    """Routes every mode uses: checkpoint lookup, deployment status and the OpenAI routes."""
    return {
        "GET /v1/loops/checkpoints": checkpoint(),
        f"GET {DEPLOYMENT}": [deployment("DEPLOYING"), deployment("ACTIVE")],
        f"POST {DEPLOYMENT}/activate": (200, {}),
        f"POST {DEPLOYMENT}/deactivate": (200, {}),
        f"GET {OPENAI}/models": served_models({"id": served_model, "max_model_len": 131_072}),
        f"POST {OPENAI}/chat/completions": chat or FakeChat(),
    }


def dedicated_routes(
    *, listed: Answer | None = None, create: Answer | None = None, chat: FakeChat | None = None
) -> dict[str, Any]:
    answers = {
        "listed_instances": listed or instances(instance("H100", 1)),
        "create_model_version_from_inference_template": create or created(),
    }
    return {**serving_routes(CHECKPOINT_NAME, chat), "POST /graphql/": graphql(answers)}


def sampler_routes(*, chat: FakeChat | None = None, **sampler: Any) -> dict[str, Any]:
    record = {"id": "smp1", "model_id": MODEL_ID, "deployment_id": DEPLOYMENT_ID}
    return {
        **serving_routes(CHECKPOINT_PATH, chat),
        "POST /v1/loops/sessions": (200, {"session": {"id": "session1"}}),
        "POST /v1/loops/samplers": (200, {"sampler": {**record, "base_url": BASE_URL, **sampler}}),
    }
