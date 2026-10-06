# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The default judge model, for real (``PUP_FINETUNE_LIVE=1``, a key in the env).

The key is read from ``FIREWORKS_API_KEY`` only, never from the keychain. One paid, tiny
serverless chat completion on a synthetic case.
"""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.judge import judge
from pup_finetune.inference import ChatClient, JudgeConfig

pytestmark = pytest.mark.live


@pytest.fixture
def api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        pytest.skip("set FIREWORKS_API_KEY to run the live judge test")
    return key


def test_the_default_judge_passes_an_identical_answer(api_key: str) -> None:
    config = JudgeConfig()
    reference = {"role": "assistant", "content": "Four."}
    case = ReplayCase(
        id="live-row:1",
        row_id="live-row",
        trace_id=None,
        message_index=1,
        prefix=({"role": "user", "content": "What is two plus two?"},),
        tools=(),
        reference=reference,
    )
    with ChatClient(config.base_url, SecretStr(api_key), credential=config.credential) as client:
        result = judge(case, reference, client, config, format_valid=True)
    assert result.error is None, result.raw
    assert result.passed, result.reason
