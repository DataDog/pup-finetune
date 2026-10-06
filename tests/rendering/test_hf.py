# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import functools
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from pup_finetune.data.schema import validate_row
from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.rendering import (
    FAMILIES,
    ChatTokenizer,
    ModelFamily,
    TokenDatum,
    hf,
    load_tokenizer,
    prompt_tokens,
    render_row,
    rendering_identity,
)
from tests.fakes.tokenizer import FakeChatTokenizer

SHA = "0123456789abcdef0123456789abcdef01234567"
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
# A Qwen-like template: the generation prompt opens <think>, so an empty block merges "\n\n".
TEMPLATE = (
    "{%- if greeting is defined %}{{- greeting + '\\n' }}{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n<think>\\n' + (m.reasoning_content or '') + '\\n</think>\\n\\n'"
    " + m.content + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}"
)


def save_tiny_tokenizer(path: Path, *, template: str | None = TEMPLATE) -> Any:
    """A byte-level BPE whose only merge is "\\n\\n", saved to ``path``; built offline."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {piece: index for index, piece in enumerate([*alphabet, "ĊĊ"])}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[("Ċ", "Ċ")]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, extra_special_tokens=SPECIALS, chat_template=template
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def tiny_family(path: Path) -> ModelFamily:
    return ModelFamily(
        key="tiny",
        tokenizer_repo=str(path),
        revision=SHA,
        max_seq_len=4096,
        supports_reasoning=True,
        template_kwargs=MappingProxyType({}),
        boundary="none",
        stop_sequences=("<|im_end|>",),
    )


@pytest.fixture
def tiny(tmp_path: Path) -> tuple[Any, ChatTokenizer]:
    raw = save_tiny_tokenizer(tmp_path)
    return raw, load_tokenizer(tiny_family(tmp_path))


def target_text(tokenizer: ChatTokenizer, datum: TokenDatum) -> str:
    return tokenizer.decode([t for t, w in zip(datum.tokens, datum.weights, strict=True) if w])


HELLO = [{"role": "user", "content": "Hi"}]


def test_load_tokenizer_adapts_the_transformers_tokenizer(tiny: tuple[Any, ChatTokenizer]) -> None:
    raw, tokenizer = tiny
    assert isinstance(tokenizer, hf.HFChatTokenizer)
    assert tokenizer.chat_template == TEMPLATE
    tokens = tokenizer.render(HELLO, tools=None, add_generation_prompt=True, template_kwargs={})
    assert type(tokens) is list
    assert all(type(token) is int for token in tokens)
    text = raw.apply_chat_template(HELLO, tokenize=False, add_generation_prompt=True)
    assert (
        tokenizer.decode(tokens)
        == text
        == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


def test_template_kwargs_reach_the_template(tiny: tuple[Any, ChatTokenizer]) -> None:
    _, tokenizer = tiny
    tokens = tokenizer.render(
        HELLO, tools=None, add_generation_prompt=False, template_kwargs={"greeting": "HELLO"}
    )
    assert tokenizer.decode(tokens).startswith("HELLO\n<|im_start|>user")


def test_encode_decode_and_special_token_ids(tiny: tuple[Any, ChatTokenizer]) -> None:
    raw, tokenizer = tiny
    end = raw.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.encode("a<|im_end|>") == [raw.convert_tokens_to_ids("a"), end]
    assert tokenizer.decode([end]) == "<|im_end|>"
    assert tokenizer.decode(tokenizer.encode(" spaced , text \n\n")) == " spaced , text \n\n"
    assert tokenizer.token_id("<|im_end|>") == end
    assert tokenizer.token_id("a") is None
    assert tokenizer.token_id("<|missing|>") is None


class _ReturnsBatchEncoding:
    """Wraps a tokenizer whose ``apply_chat_template`` ignores ``return_dict=False``."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.apply_chat_template(*args, **{**kwargs, "return_dict": True})


def test_render_normalizes_a_batch_encoding(tiny: tuple[Any, ChatTokenizer]) -> None:
    raw, tokenizer = tiny
    wrapped = hf.HFChatTokenizer(_ReturnsBatchEncoding(raw))
    kwargs: dict[str, Any] = {"tools": None, "add_generation_prompt": True, "template_kwargs": {}}
    assert wrapped.render(HELLO, **kwargs) == tokenizer.render(HELLO, **kwargs)


def test_render_row_through_a_real_tokenizer_re_encodes_a_merged_reply(
    tiny: tuple[Any, ChatTokenizer], tmp_path: Path
) -> None:
    _, tokenizer = tiny
    family = tiny_family(tmp_path)
    row = validate_row(
        {"id": "r", "messages": [*HELLO, {"role": "assistant", "content": "Hello café"}]}
    )
    prompt = prompt_tokens(HELLO, None, family, tokenizer)
    full = tokenizer.render(
        [*HELLO, {"role": "assistant", "content": "Hello café"}],
        tools=None,
        add_generation_prompt=False,
        template_kwargs={},
    )
    assert full[: len(prompt)] != prompt
    [datum] = render_row(row, family, tokenizer, max_seq_len=family.max_seq_len)
    assert datum.tokens[: len(prompt)] == tuple(prompt)
    assert tokenizer.decode(datum.tokens) == tokenizer.decode(full)
    assert target_text(tokenizer, datum) == "\n</think>\n\nHello café<|im_end|>"


def test_load_tokenizer_without_transformers_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(UsageError) as caught:
        load_tokenizer(FAMILIES["qwen3_5"])
    assert caught.value.hint == "install the hf extra: pip install 'pup-finetune[hf]'"


def test_a_tokenizer_that_cannot_load_is_an_external_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", unreachable)
    family = FAMILIES["qwen3_5"]
    with pytest.raises(ExternalError) as caught:
        load_tokenizer(family)
    assert caught.value.source == "huggingface"
    assert family.tokenizer_repo in caught.value.message
    assert family.revision in caught.value.message
    assert "OSError" in caught.value.message


def test_a_tokenizer_without_a_chat_template_is_an_external_error(tmp_path: Path) -> None:
    save_tiny_tokenizer(tmp_path, template=None)
    with pytest.raises(ExternalError, match="has no chat template") as caught:
        load_tokenizer(tiny_family(tmp_path))
    assert caught.value.source == "huggingface"


def test_rendering_identity_pins_the_implementation_tokenizer_and_template() -> None:
    pytest.importorskip("transformers")
    family = FAMILIES["glm5_3"]
    tokenizer = FakeChatTokenizer("glm")
    identity = rendering_identity(family, tokenizer)
    assert identity.as_dict() == {
        "version": f"pup-finetune-prefix-v1;transformers={metadata.version('transformers')}",
        "tokenizer_repo": "zai-org/GLM-5.3-Flash",
        "revision": family.revision,
        "template_sha256": hashlib.sha256(b"fake-glm-template-v1").hexdigest(),
        "template_kwargs": {},
    }
    assert rendering_identity(family, FakeChatTokenizer()) != identity


def test_rendering_identity_without_transformers_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    with pytest.raises(UsageError) as caught:
        rendering_identity(FAMILIES["qwen3_5"], FakeChatTokenizer())
    assert caught.value.hint == "install the hf extra: pip install 'pup-finetune[hf]'"


@functools.cache
def pinned(key: str) -> ChatTokenizer:
    pytest.importorskip("transformers")
    return load_tokenizer(FAMILIES[key])


CITY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SENTINEL_ROW = {
    "id": "sentinels",
    "messages": [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_ONE_SENTINEL"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASONING_SENTINEL",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "TOOL_RESULT_SENTINEL"},
        {"role": "assistant", "content": "ASSISTANT_ONE_SENTINEL"},
        {"role": "user", "content": "USER_TWO_SENTINEL"},
        {"role": "assistant", "content": "HISTORY_SENTINEL", "weight": 0},
        {"role": "user", "content": "USER_THREE_SENTINEL"},
        {"role": "assistant", "content": "ASSISTANT_TWO_SENTINEL café 🐢 東京"},
    ],
    "tools": [CITY_TOOL],
}
CONTEXT_SENTINELS = (
    "SYSTEM_SENTINEL",
    "USER_ONE_SENTINEL",
    "TOOL_RESULT_SENTINEL",
    "USER_TWO_SENTINEL",
    "HISTORY_SENTINEL",
    "USER_THREE_SENTINEL",
    "Current conditions",
)


@pytest.mark.tokenizer
@pytest.mark.parametrize("key", ["qwen3_5", "glm5_3"])
def test_pinned_tokenizer_trains_only_the_assistant_replies(key: str) -> None:
    family, tokenizer = FAMILIES[key], pinned(key)
    row = validate_row(SENTINEL_ROW)
    datums = render_row(row, family, tokenizer, max_seq_len=family.max_seq_len)
    assert [datum.message_index for datum in datums] == [2, 4, 8]
    targets = [target_text(tokenizer, datum) for datum in datums]
    for target, expected in zip(
        targets,
        (
            ("REASONING_SENTINEL", "get_weather", "Northvale"),
            ("ASSISTANT_ONE_SENTINEL",),
            ("ASSISTANT_TWO_SENTINEL café 🐢 東京",),
        ),
        strict=True,
    ):
        assert all(text in target for text in expected)
    joined = "".join(targets)
    assert joined.count("REASONING_SENTINEL") == 1
    assert not any(text in joined for text in CONTEXT_SENTINELS)
    assert "HISTORY_SENTINEL" in tokenizer.decode(datums[2].tokens)
    messages = [m.model_dump(mode="json", exclude_none=True) for m in row.messages]
    for datum in datums:
        prompt = prompt_tokens(messages[: datum.message_index], [CITY_TOOL], family, tokenizer)
        assert datum.tokens[: len(prompt)] == tuple(prompt)
        assert not any(datum.weights[: len(prompt)])
        assert datum.weights[len(prompt)] == 1
    if key == "qwen3_5":
        assert all(target.endswith("<|im_end|>") for target in targets)
        assert all(tokenizer.decode(datum.tokens[-1:]) == "\n" for datum in datums)
        assert all(datum.weights[-1] == 0 for datum in datums)
    else:
        assert targets[0].endswith("<|observation|>")
        assert all(target.endswith("<|user|>") for target in targets[1:])
        assert all(datum.weights[-1] == 1 for datum in datums)


@pytest.mark.tokenizer
def test_pinned_qwen_re_encodes_the_empty_thinking_block() -> None:
    family, tokenizer = FAMILIES["qwen3_5"], pinned("qwen3_5")
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
    prompt = prompt_tokens(messages[:1], None, family, tokenizer)
    full = tokenizer.render(messages, tools=None, add_generation_prompt=False, template_kwargs={})
    assert full[: len(prompt)] != prompt
    [datum] = render_row(
        validate_row({"id": "r", "messages": messages}), family, tokenizer, max_seq_len=4096
    )
    assert tokenizer.decode(datum.tokens) == tokenizer.decode(full)
    assert target_text(tokenizer, datum) == "\n</think>\n\nHello.<|im_end|>"


@pytest.mark.tokenizer
def test_pinned_rendering_identity_hashes_the_real_template() -> None:
    family, tokenizer = FAMILIES["qwen3_5"], pinned("qwen3_5")
    identity = rendering_identity(family, tokenizer)
    assert identity.tokenizer_repo == "Qwen/Qwen3.5-9B"
    assert identity.revision == family.revision
    expected = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    assert identity.template_sha256 == expected
    assert (
        identity.version
        == f"pup-finetune-prefix-v1;transformers={metadata.version('transformers')}"
    )
    assert json.dumps(identity.as_dict())
