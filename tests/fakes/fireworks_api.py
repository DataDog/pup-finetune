# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A fake Fireworks REST API, served through an ``httpx`` transport, with synthetic data.

It answers the routes the read-only client uses::

    api = FakeFireworksApi()  # every profile's model READY and tunable; one account
    api.models["qwen3p8-27b"]["state"] = "DEPLOYING"
    api.account_models["accounts/synthetic-team/models/ft-run"] = {...}  # a tuned model
    api.status["/v1/accounts"] = 503  # answer that route with an HTTP error
    client = FireworksApi(SecretStr(KEY), transport=api.transport)

Requests are recorded in :attr:`FakeFireworksApi.requests`. A request without the expected
``Bearer`` key gets HTTP 401; an unknown model gets 404; ``raise_on`` makes a route raise an
``httpx`` transport error instead of answering.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from pup_finetune.providers.base import ModelProfile
from pup_finetune.providers.fireworks.models import PROFILES

API_KEY = "fw-synthetic-0000-test-key-9f3c"

_MODELS_PREFIX = "/v1/accounts/fireworks/models/"


def model_entry(profile: ModelProfile) -> dict[str, Any]:
    """What Fireworks answers for ``profile``'s base model when it can be LoRA-tuned."""
    return {
        "name": profile.base_model,
        "displayName": profile.alias,
        "state": "READY",
        "huggingFaceUrl": f"https://huggingface.co/{profile.tokenizer_model}",
        "trainingContextLength": profile.max_seq_len,
        "supervisedLoraTunable": True,
        "kind": "HF_BASE_MODEL",
    }


class FakeFireworksApi:
    """The Fireworks routes the client reads, in memory. See the module docstring."""

    def __init__(self, *, api_key: str = API_KEY) -> None:
        self.api_key = api_key
        self.models: dict[str, dict[str, Any]] = {
            profile.base_model.removeprefix("accounts/fireworks/models/"): model_entry(profile)
            for profile in PROFILES
        }
        self.account_models: dict[str, dict[str, Any]] = {}
        """Models of other accounts than ``fireworks``, by resource name."""
        self.accounts: list[dict[str, Any]] = [
            {"name": "accounts/synthetic-team", "displayName": "Synthetic team"}
        ]
        self.status: dict[str, int] = {}
        self.bodies: dict[str, str] = {}
        self.raise_on: dict[str, Exception] = {}
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.raise_on:
            raise self.raise_on[path]
        if request.headers.get("authorization") != f"Bearer {self.api_key}":
            return _json(401, {"error": {"message": "invalid API key"}})
        if path in self.status:
            body = self.bodies.get(path, json.dumps({"error": {"message": "synthetic failure"}}))
            return httpx.Response(self.status[path], text=body)
        if request.method == "GET" and path == "/v1/accounts":
            return _json(200, {"accounts": self.accounts, "totalSize": len(self.accounts)})
        if request.method == "GET" and path.startswith(_MODELS_PREFIX):
            model = self.models.get(path.removeprefix(_MODELS_PREFIX))
            if model is not None:
                return _json(200, model)
        if request.method == "GET" and path.startswith("/v1/accounts/"):
            model = self.account_models.get(path.removeprefix("/v1/"))
            if model is not None:
                return _json(200, model)
        return _json(404, {"error": {"message": "not found"}})


def _json(status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, json=body)
