# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake OpenAI-compatible chat completions endpoint, served through an ``httpx`` transport.

::

    api = FakeChatApi(answer=lambda body: '{"pass": true, "reason": "same call"}')
    api.script.append(httpx.Response(429))  # answered (or raised) first, in order
    client = ChatClient(BASE_URL, SecretStr(API_KEY), transport=api.transport)

Requests are recorded in :attr:`FakeChatApi.requests` and their JSON bodies in
:attr:`FakeChatApi.bodies`. A request without the expected ``Bearer`` key gets HTTP 401.
:func:`matching_judge` answers like a judge that passes a candidate exactly when it equals
the reference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx

API_KEY = "sk-synthetic-0000-chat-key-7a1d"
BASE_URL = "https://inference.example.test/v1"
MODEL = "accounts/fireworks/models/synthetic-judge"

Answer = Callable[[dict[str, Any]], str | None]


def completion(
    content: str | None,
    *,
    model: str = MODEL,
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    refusal: str | None = None,
) -> httpx.Response:
    """A successful chat completion whose first choice says ``content`` (and makes
    ``tool_calls``, or declines with ``refusal``, when given)."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if refusal is not None:
        message["refusal"] = refusal
    return httpx.Response(
        200,
        json={
            "id": "cmpl-synthetic",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


def always(content: str) -> Answer:
    """An answer that is ``content`` whatever was asked."""
    return lambda body: content


class FakeChatApi:
    """``POST /v1/chat/completions``, in memory. See the module docstring."""

    def __init__(self, *, api_key: str = API_KEY, answer: Answer | None = None) -> None:
        self.api_key = api_key
        self.answer: Answer = answer or always('{"pass": true, "reason": "acceptable"}')
        self.script: list[httpx.Response | Exception] = []
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else {}
        self.bodies.append(body)
        if request.headers.get("authorization") != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"error": {"message": "invalid api key"}})
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return completion(self.answer(body), model=str(body.get("model", MODEL)))


def block(text: str, name: str) -> Any:
    """The decoded JSON of the ``<name>`` block in a judge prompt."""
    match = re.search(rf"<{name}>\n(.*?)\n</{name}>", text, re.DOTALL)
    assert match is not None, f"no <{name}> block"
    return json.loads(match.group(1))


def matching_judge(body: dict[str, Any]) -> str:
    """Pass exactly when the candidate action equals the reference action."""
    prompt = body["messages"][-1]["content"]
    same = block(prompt, "candidate_action") == block(prompt, "reference_action")
    reason = "the candidate matches the reference" if same else "the candidate differs"
    return json.dumps({"pass": same, "reason": reason})
