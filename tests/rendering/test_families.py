# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

from pup_finetune.rendering.families import FAMILIES, QWEN3_8, ModelFamily, get_family
from pup_finetune.rendering.parsers import Qwen3XmlToolParser
from tests.fakes.tokenizer import SPECIAL_IDS, FakeChatTokenizer

SHA = "0123456789abcdef0123456789abcdef01234567"


def family(**overrides: Any) -> ModelFamily:
    fields: dict[str, Any] = {
        "key": "tiny",
        "tokenizer_model": "example/tiny-chat",
        "revision": SHA,
        "parser": "qwen3_xml",
        "stop_tokens": ("<|im_end|>",),
    }
    return ModelFamily(**{**fields, **overrides})


def test_every_family_is_pinned_to_a_commit() -> None:
    assert set(FAMILIES) == {"qwen3_8"}
    for entry in FAMILIES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", entry.revision)
    assert get_family("qwen3_8") is QWEN3_8


def test_qwen3_8_renders_per_turn_and_parses_xml_tool_calls() -> None:
    assert QWEN3_8.tokenizer_model == "Qwen/Qwen3.8-27B"
    assert not QWEN3_8.prefix_stable
    renderer = QWEN3_8.renderer(tokenizer=FakeChatTokenizer())
    assert renderer.name == "qwen3_8"
    assert not renderer.prefix_stable
    assert renderer.stop_token_ids == [SPECIAL_IDS["<|im_end|>"]]
    identity = renderer.identity()
    assert identity["renderer"] == "hf-chat-template"
    assert identity["parser"] == Qwen3XmlToolParser.name
    assert identity["tokenizer_revision"] == QWEN3_8.revision


def test_a_family_renderer_carries_its_template_kwargs() -> None:
    entry = family(chat_template_kwargs={"preserve_thinking": False})
    tokenizer = FakeChatTokenizer()
    entry.renderer(tokenizer=tokenizer).generation_prompt_tokens(
        [{"role": "user", "content": "Hi."}], []
    )
    assert tokenizer.calls[-1]["template_kwargs"] == {"preserve_thinking": False}
    with pytest.raises(TypeError):
        entry.chat_template_kwargs["x"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"revision": "main"}, "40-character"),
        ({"revision": SHA.upper()}, "40-character"),
        ({"parser": "nope"}, "unknown parser"),
        ({"stop_tokens": ()}, "stop token"),
    ],
)
def test_a_family_is_validated(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        family(**overrides)


def test_get_family_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known: qwen3_8"):
        get_family("nope")
    assert dataclasses.replace(QWEN3_8, prefix_stable=True).prefix_stable
