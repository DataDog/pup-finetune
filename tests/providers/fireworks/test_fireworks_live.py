# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Checks against the real Fireworks API (``PUP_FINETUNE_LIVE=1``, a key in the env).

The key is read from ``FIREWORKS_API_KEY`` only, never from the keychain. Preflight, the key
check and the account lookup are free GET requests. The training test is paid (a short
serverless session) and needs the ``fireworks`` and ``hf`` extras: it trains two steps on
two tiny synthetic rows, saves a training and a sampler checkpoint, promotes the sampler
checkpoint to a uniquely named model (which stays in the account; its id is printed) and
samples 16 tokens from the base and the tuned model. The session is always ended.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Any

import pytest
from pydantic import SecretStr

from pup_finetune.providers.base import AdamConfig, LoraConfig
from pup_finetune.providers.fireworks import validate_key
from pup_finetune.providers.fireworks.client import (
    SAMPLER_CHECKPOINT,
    TRAINING_CHECKPOINT,
    FireworksTrainingClient,
)
from pup_finetune.providers.fireworks.provider import FireworksProvider
from pup_finetune.runtime import Runtime

pytestmark = pytest.mark.live


@pytest.fixture
def api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        pytest.skip("set FIREWORKS_API_KEY to run the live Fireworks tests")
    return key


def test_every_profile_passes_preflight(api_key: str) -> None:
    provider = FireworksProvider(Runtime(), env={"FIREWORKS_API_KEY": api_key})
    for profile in provider.models():
        provider.preflight(profile)


def test_the_key_validates(api_key: str) -> None:
    validate_key(SecretStr(api_key))


def test_the_training_client_resolves_the_account(api_key: str) -> None:
    client = FireworksTrainingClient(SecretStr(api_key), env={})
    try:
        assert client.account_id
        assert "/" not in client.account_id
    finally:
        client.close()


# Synthetic rows only: never real trace content.
_ROWS: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = [
    (
        [
            {"role": "user", "content": "Say hello to Northvale."},
            {"role": "assistant", "content": "Hello, Northvale!"},
        ],
        [],
    ),
    (
        [
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."},
        ],
        [],
    ),
]
_PROMPT: list[dict[str, Any]] = [{"role": "user", "content": "Say hello."}]
_TIMEOUT_S = 900


def test_train_checkpoint_promote_and_sample(api_key: str) -> None:
    pytest.importorskip("fireworks.training.sdk")
    pytest.importorskip("transformers")
    provider = FireworksProvider(Runtime(), env={"FIREWORKS_API_KEY": api_key})
    profile = provider.resolve_model("qwen3p8-27b")
    renderer = provider.renderer(profile)
    batch = [
        example
        for messages, tools in _ROWS
        for example in renderer.per_turn_examples(messages, tools)
    ]
    longest = max(example.num_tokens for example in batch)
    model_id = f"pft-live-{uuid.uuid4().hex[:12]}"
    lora = LoraConfig(rank=profile.default_lora_rank)
    with provider.backend().open_session(profile, lora, seed=42, max_seq_len=longest) as session:
        for _ in range(2):
            metrics = session.forward_backward(batch).result(timeout=_TIMEOUT_S)
            assert math.isfinite(metrics.mean_loss)
            assert metrics.weight > 0
            session.optim_step(AdamConfig(learning_rate=1e-4)).result(timeout=_TIMEOUT_S)
        assert math.isfinite(session.forward(batch).result(timeout=_TIMEOUT_S).mean_loss)

        training = session.save_training_checkpoint("t1s2-live", epoch=1, step=2)
        sampler_ckpt = session.save_sampler_checkpoint("s1s2-live", epoch=1, step=2)
        assert training.kind == "training" and sampler_ckpt.kind == "sampler"
        types = {row.checkpoint_type for row in session.list_checkpoints()}
        assert {TRAINING_CHECKPOINT, SAMPLER_CHECKPOINT} <= types

        model = session.promote(sampler_ckpt, model_id)
        print(f"promoted live test model: {model.uri}")
        assert model.model_id == model_id

        for ckpt in (None, sampler_ckpt):
            sampler = session.sampler(ckpt)
            try:
                generation = sampler.generate(_PROMPT, [], max_tokens=16)
            finally:
                sampler.close()
            assert generation.message["role"] == "assistant"
            assert 0 < generation.completion_tokens <= 16
