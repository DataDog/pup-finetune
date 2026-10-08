# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""An Anthropic and an OpenAI judge, for real (``PUP_FINETUNE_LIVE=1``, a key in the env).

Each key is read from its environment variable only (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``), never from the keychain, and each test is skipped without it. One paid,
small judge call per provider on a synthetic case.
"""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from pup_finetune.evaluation.cases import ReplayCase
from pup_finetune.evaluation.judge import judge
from pup_finetune.inference import OPENAI_POLICY, ChatClient
from pup_finetune.judges import parse_judge

pytestmark = pytest.mark.live

ANTHROPIC_JUDGE = "anthropic/claude-opus-5-5"
OPENAI_JUDGE = "openai/gpt-6-astra"


def key(env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if not value:
        pytest.skip(f"set {env_var} to run this live judge test")
    return value


def case() -> tuple[ReplayCase, dict[str, str]]:
    reference = {"role": "assistant", "content": "Four."}
    return (
        ReplayCase(
            id="live-row:1",
            row_id="live-row",
            trace_id=None,
            message_index=1,
            prefix=({"role": "user", "content": "What is two plus two?"},),
            tools=(),
            reference=reference,
        ),
        reference,
    )


def test_an_anthropic_judge_passes_an_identical_answer() -> None:
    pytest.importorskip("anthropic", reason="needs the anthropic extra")
    from pup_finetune.anthropic_client import AnthropicChatClient

    config = parse_judge(ANTHROPIC_JUDGE)
    replay, reference = case()
    with AnthropicChatClient(SecretStr(key("ANTHROPIC_API_KEY"))) as client:
        result = judge(replay, reference, client, config, format_valid=True)
    assert result.error is None, result.raw
    assert result.passed, result.reason


def test_an_openai_judge_passes_an_identical_answer() -> None:
    config = parse_judge(OPENAI_JUDGE)
    replay, reference = case()
    with ChatClient(
        config.base_url, SecretStr(key("OPENAI_API_KEY")), credential="openai", policy=OPENAI_POLICY
    ) as client:
        result = judge(replay, reference, client, config, format_valid=True)
    assert result.error is None, result.raw
    assert result.passed, result.reason
