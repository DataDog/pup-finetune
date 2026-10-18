# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any

import pytest

from pup_finetune.rendering import TokenizedExample, hf
from pup_finetune.rendering.families import (
    DEEPSEEK_V4,
    FAMILIES,
    GLM5_3,
    KIMI_K3,
    MUSE_GLIMMER,
    QWEN3_5,
    QWEN3_8,
    ModelFamily,
    get_family,
)
from pup_finetune.rendering.hf import HFChatTemplateRenderer
from pup_finetune.rendering.parsers import GlmXmlToolParser, Qwen3XmlToolParser
from tests.fakes.tokenizer import (
    MUSE_SPECIALS,
    SPECIAL_IDS,
    FakeChatTokenizer,
    glm5_like,
    muse_like,
)

SHA = "0123456789abcdef0123456789abcdef01234567"

MAIN_IDENTITIES: dict[str, dict[str, str]] = {
    "qwen3_8": {
        "renderer": "hf-chat-template",
        "renderer_version": "1",
        "name": "qwen3_8",
        "parser": "qwen3_xml",
        "parser_version": "3",
        "tokenizer": "Qwen/Qwen3.8-27B",
        "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "chat_template_sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "chat_template_kwargs": "{}",
        "stop_tokens": '["<|im_end|>"]',
        "prefix_stable": "false",
    },
    "qwen3_5": {
        "renderer": "hf-chat-template",
        "renderer_version": "1",
        "name": "qwen3_5",
        "parser": "qwen3_5_xml",
        "parser_version": "1",
        "tokenizer": "Qwen/Qwen3.5-9B",
        "tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "chat_template_sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "chat_template_kwargs": "{}",
        "stop_tokens": '["<|im_end|>"]',
        "prefix_stable": "false",
    },
    "glm5_3": {
        "renderer": "hf-chat-template",
        "renderer_version": "1",
        "name": "glm5_3",
        "parser": "glm_xml",
        "parser_version": "1",
        "tokenizer": "zai-org/GLM-5.3-Flash",
        "tokenizer_revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        "chat_template_sha256": "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
        "chat_template_kwargs": "{}",
        "stop_tokens": '["<|user|>","<|observation|>"]',
        "prefix_stable": "true",
    },
}
"""What ``main``'s renderer identity is for each family it shipped, real template included
(``transformers_version`` aside): a run trained there resumes and evaluates only while the
identity stays the same."""


def family(**overrides: Any) -> ModelFamily:
    fields: dict[str, Any] = {
        "key": "tiny",
        "tokenizer_model": "example/tiny-chat",
        "revision": SHA,
        "parser": "qwen3_xml",
        "stop_tokens": ("<|im_end|>",),
    }
    return ModelFamily(**{**fields, **overrides})


@pytest.mark.parametrize("shipped", [QWEN3_8, QWEN3_5, GLM5_3], ids=lambda f: f.key)
def test_a_family_main_shipped_keeps_main_s_renderer_identity(shipped: ModelFamily) -> None:
    template = "a template"
    identity = shipped.renderer(tokenizer=FakeChatTokenizer(template_name=template)).identity()
    assert identity == {
        **MAIN_IDENTITIES[shipped.key],
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "transformers_version": hf.transformers_version(),
    }


@pytest.mark.tokenizer
@pytest.mark.parametrize("shipped", [QWEN3_8, QWEN3_5, GLM5_3], ids=lambda f: f.key)
def test_a_family_main_shipped_keeps_main_s_identity_with_its_real_template(
    shipped: ModelFamily, request: pytest.FixtureRequest
) -> None:
    tokenizer = request.getfixturevalue(f"{shipped.key}_tokenizer")
    identity = shipped.renderer(tokenizer=tokenizer).identity()
    assert identity == {
        **MAIN_IDENTITIES[shipped.key],
        "transformers_version": hf.transformers_version(),
    }


def test_every_family_is_pinned_to_a_commit() -> None:
    assert set(FAMILIES) == {
        "qwen3_8",
        "qwen3_5",
        "glm5_3",
        "kimi_k3",
        "deepseek_v4",
        "muse_glimmer",
    }
    for entry in FAMILIES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", entry.revision)
    assert get_family("qwen3_8") is QWEN3_8
    assert get_family("qwen3_5") is QWEN3_5
    assert get_family("glm5_3") is GLM5_3


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


def test_qwen3_5_renders_per_turn_and_reads_python_scalars_in_tool_calls() -> None:
    assert QWEN3_5.tokenizer_model == "Qwen/Qwen3.5-9B"
    assert QWEN3_5.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    renderer = QWEN3_5.renderer(tokenizer=FakeChatTokenizer(merge_newlines=True))
    assert renderer.name == "qwen3_5"
    assert not renderer.prefix_stable
    assert renderer.stop_token_ids == [SPECIAL_IDS["<|im_end|>"]]
    assert renderer.identity()["parser"] == "qwen3_5_xml"
    parsed = renderer.parse_response(
        "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=on>\nTrue\n</parameter>\n"
        "<parameter=unit>\nNone\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    assert parsed.format_valid
    [call] = parsed.message["tool_calls"]
    assert call["function"]["arguments"] == '{"on": true, "unit": null}'


def test_glm5_3_stops_on_the_next_role_tag_and_parses_arg_key_tool_calls() -> None:
    assert GLM5_3.tokenizer_model == "zai-org/GLM-5.3-Flash"
    assert GLM5_3.revision == "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
    assert GLM5_3.next_role_tag is not None
    assert GLM5_3.next_role_tag({"role": "assistant", "content": "Hi."}) == "<|user|>"
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    assert GLM5_3.next_role_tag({"role": "assistant", "content": "", "tool_calls": [call]}) == (
        "<|observation|>"
    )
    renderer = GLM5_3.renderer(tokenizer=FakeChatTokenizer(template=glm5_like))
    assert renderer.name == "glm5_3"
    assert renderer.prefix_stable
    assert renderer.stop_token_ids == [SPECIAL_IDS["<|user|>"], SPECIAL_IDS["<|observation|>"]]
    assert renderer.identity()["parser"] == GlmXmlToolParser.name
    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c", "content": "sunny"},
        {"role": "assistant", "content": "Sunny."},
    ]
    example = renderer.supervised_example(messages, [])
    trained = [t for t, w in zip(example.target_tokens, example.weights, strict=True) if w]
    assert renderer.decode(trained) == (
        "</think><tool_call>f</tool_call><|observation|></think>Sunny.<|user|>"
    )


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
        ({"chat_encoder": "nope"}, "unknown chat encoder"),
        (
            {"chat_encoder": "deepseek_v4", "chat_template_kwargs": {"x": 1}},
            "takes no chat-template arguments",
        ),
    ],
)
def test_a_family_is_validated(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        family(**overrides)


def test_get_family_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known: qwen3_8, qwen3_5, glm5_3"):
        get_family("nope")
    assert dataclasses.replace(QWEN3_8, prefix_stable=True).prefix_stable


def test_only_kimi_k3_runs_its_repository_s_tokenizer_code() -> None:
    assert [key for key, entry in FAMILIES.items() if entry.trust_remote_code] == ["kimi_k3"]
    assert [key for key, entry in FAMILIES.items() if entry.chat_encoder] == ["deepseek_v4"]


@pytest.mark.parametrize(
    ("entry", "options"),
    [
        (QWEN3_8, {"trust_remote_code": False, "template": True}),
        (KIMI_K3, {"trust_remote_code": True, "template": True}),
        (DEEPSEEK_V4, {"trust_remote_code": False, "template": False}),
    ],
)
def test_a_family_loads_its_tokenizer_at_its_revision_as_it_needs(
    entry: ModelFamily, options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[tuple[str, str, dict[str, Any]]] = []

    def load(model: str, revision: str, **given: Any) -> FakeChatTokenizer:
        loads.append((model, revision, given))
        return FakeChatTokenizer()

    monkeypatch.setattr(hf, "load_tokenizer", load)
    entry.renderer().generation_prompt_tokens([{"role": "user", "content": "Hi."}], [])
    assert loads == [(entry.tokenizer_model, entry.revision, options)]


def test_deepseek_v4_renders_through_its_encoder_not_the_tokenizer_s_template() -> None:
    tokenizer = FakeChatTokenizer()
    renderer = DEEPSEEK_V4.renderer(tokenizer=tokenizer)
    tokens = renderer.generation_prompt_tokens([{"role": "user", "content": "Hi."}], [])
    assert renderer.decode(tokens) == "<｜begin▁of▁sentence｜><｜User｜>Hi.<｜Assistant｜><think>"  # noqa: RUF001
    assert tokenizer.calls == []
    assert (
        renderer.identity()["chat_template_sha256"]
        != hf.hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    )


def muse_renderer() -> HFChatTemplateRenderer:
    return MUSE_GLIMMER.renderer(
        tokenizer=FakeChatTokenizer(template=muse_like, specials=MUSE_SPECIALS)
    )


def trained(renderer: HFChatTemplateRenderer, example: TokenizedExample) -> str:
    return renderer.decode(
        [t for t, w in zip(example.target_tokens, example.weights, strict=True) if w > 0]
    )


def test_muse_glimmer_pins_the_date_of_its_default_system_message() -> None:
    renderer = muse_renderer()
    messages = [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hey."}]
    (example,) = renderer.per_turn_examples(messages, [])
    assert "Current date: 2026-01-04." in renderer.decode(example.input_tokens)
    assert trained(renderer, example) == " to=user<|message|>Hey.<|eot|>"
    assert renderer.supervised_example(messages, []) == example


MUSE_TEXT_AND_CALL: list[dict[str, Any]] = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "Hi."},
    {
        "role": "assistant",
        "content": "Looking.",
        "reasoning_content": "Use f.",
        "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"x": "1"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c", "content": "done"},
    {"role": "assistant", "content": "Done."},
]


def test_muse_glimmer_writes_text_beside_tool_calls_as_its_own_message_to_the_user() -> None:
    renderer = muse_renderer()
    first, second = renderer.per_turn_examples(MUSE_TEXT_AND_CALL, [])
    text = trained(renderer, first)
    assert text == (
        " to=self<|message|>Use f.<|eom|>"
        "<|start|>assistant to=user<|message|>Looking.<|eom|>"
        '<|start|>assistant to=f<|message|><atem:function_calls>\n<atem:invoke name="f">\n'
        '<atem:parameter name="x">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>'
        "<|eot|>"
    )
    parsed = renderer.parse_response(text)
    assert parsed.format_valid
    assert parsed.message["content"] == "Looking."
    assert parsed.message["reasoning_content"] == "Use f."
    assert json.loads(parsed.message["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert "Looking.<|eom|>" in renderer.decode(second.input_tokens)
    without_reasoning = [
        {key: value for key, value in message.items() if key != "reasoning_content"}
        for message in MUSE_TEXT_AND_CALL
    ]
    whole = renderer.supervised_example(without_reasoning, [])
    turns = renderer.per_turn_examples(without_reasoning, [])
    assert trained(renderer, whole) == "".join(trained(renderer, turn) for turn in turns)


def test_a_family_s_message_rewrite_is_in_the_renderer_identity() -> None:
    assert muse_renderer().identity()["message_rewrite"] == "muse_text_before_calls@1"
    assert "message_rewrite" not in QWEN3_8.renderer(tokenizer=FakeChatTokenizer()).identity()
    assert MUSE_GLIMMER.rewrite is not None
    assert MUSE_GLIMMER.rewrite.messages([{"role": "user", "content": "Hi."}]) == [
        {"role": "user", "content": "Hi."}
    ]
