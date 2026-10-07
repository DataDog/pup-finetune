# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.rendering import (
    ChatMessage,
    RendererError,
    RendererErrorCode,
    TokenizedExample,
    hf,
)
from pup_finetune.rendering.hf import HFChatTemplateRenderer, TransformersTokenizer
from pup_finetune.rendering.parsers import Qwen3XmlToolParser
from pup_finetune.training.datums import build_examples
from tests.fakes.tokenizer import (
    DOUBLE_NEWLINE_ID,
    SPECIAL_IDS,
    FakeChatTokenizer,
    qwen3_8_like,
    role_tag_after,
    role_tag_template,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
IM_END = SPECIAL_IDS["<|im_end|>"]

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

TOOL_ROW: list[ChatMessage] = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "Weather in Northvale?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Northvale"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny in Northvale."},
]

TWO_TURNS: list[ChatMessage] = [
    {"role": "user", "content": "Count to two."},
    {"role": "assistant", "content": "One."},
    {"role": "user", "content": "Next."},
    {"role": "assistant", "content": "Two."},
]


def renderer(
    tokenizer: FakeChatTokenizer | None = None, *, prefix_stable: bool = False, **kwargs: Any
) -> HFChatTemplateRenderer:
    return HFChatTemplateRenderer(
        "example/tiny-chat",
        SHA,
        response_parser=Qwen3XmlToolParser(),
        stop_tokens=["<|im_end|>"],
        prefix_stable=prefix_stable,
        tokenizer=tokenizer or FakeChatTokenizer(),
        **kwargs,
    )


def unshifted(example: TokenizedExample) -> tuple[list[int], list[float]]:
    return [*example.input_tokens, example.target_tokens[-1]], [0.0, *example.weights]


def trained_runs(r: HFChatTemplateRenderer, example: TokenizedExample) -> list[str]:
    """The decoded text of each maximal run of trained tokens."""
    tokens, weights = unshifted(example)
    runs: list[list[int]] = []
    for position, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
        if weight:
            if position == 0 or not weights[position - 1]:
                runs.append([])
            runs[-1].append(token)
    return [r.decode(run) for run in runs]


def test_each_trainable_turn_is_one_example_after_its_own_prompt() -> None:
    r = renderer()
    examples = r.per_turn_examples(TOOL_ROW, [WEATHER_TOOL])
    assert len(examples) == 2
    for turn, example in zip([2, 4], examples, strict=True):
        tokens, _ = unshifted(example)
        prompt = r.generation_prompt_tokens(TOOL_ROW[:turn], [WEATHER_TOOL])
        assert tokens[: len(prompt)] == prompt
        assert tokens[-1] == IM_END
    assert [trained_runs(r, example) for example in examples] == [
        [
            "\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNorthvale\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>"
        ],
        ["\n</think>\n\nIt is sunny in Northvale.<|im_end|>"],
    ]


def test_the_mask_covers_only_the_completion_through_the_stop_token() -> None:
    r = renderer()
    [example] = r.per_turn_examples(TWO_TURNS[:2], [])
    tokens, weights = unshifted(example)
    prompt = r.generation_prompt_tokens(TWO_TURNS[:1], [])
    assert not any(weights[: len(prompt)])
    assert all(weights[len(prompt) :])
    # The template's newline after <|im_end|> is not part of the example.
    assert r.decode(tokens).endswith("One.<|im_end|>")


def test_weight_zero_turns_stay_history_but_are_never_trained() -> None:
    r = renderer(prefix_stable=True)
    messages = [*TWO_TURNS[:1], {**TWO_TURNS[1], "weight": 0}, *TWO_TURNS[2:]]
    [example] = r.per_turn_examples(messages, [])
    assert "One." in r.decode(unshifted(example)[0])
    assert trained_runs(r, example) == ["\n</think>\n\nTwo.<|im_end|>"]
    full = r.supervised_example(messages, [])
    assert trained_runs(r, full) == ["\n</think>\n\nTwo.<|im_end|>"]
    # Nothing ours reaches the template.
    tokenizer = FakeChatTokenizer()
    renderer(tokenizer).per_turn_examples(messages, [])
    assert all("weight" not in m for call in tokenizer.calls for m in call["messages"])


def test_a_row_with_no_trainable_turn_is_refused() -> None:
    r = renderer(prefix_stable=True)
    messages = [TWO_TURNS[0], {**TWO_TURNS[1], "weight": 0}]
    for render in (r.per_turn_examples, r.supervised_example):
        with pytest.raises(RendererError) as excinfo:
            render(messages, [])
        assert excinfo.value.code is RendererErrorCode.NO_TRAINABLE_TOKENS


def test_a_token_merged_across_the_prompt_boundary_is_re_encoded_after_the_prompt() -> None:
    tokenizer = FakeChatTokenizer(merge_newlines=True)
    r = renderer(tokenizer)
    messages = TWO_TURNS[:2]
    prompt = r.generation_prompt_tokens(messages[:1], [])
    whole = tokenizer.encode(
        tokenizer.render_text(messages, tools=None, add_generation_prompt=False, template_kwargs={})
    )
    assert whole[: len(prompt)] != prompt  # "<think>\n" + "\n</think>" merged "\n\n"
    [example] = r.per_turn_examples(messages, [])
    tokens, _ = unshifted(example)
    assert tokens[: len(prompt)] == prompt
    assert trained_runs(r, example) == ["\n</think>\n\nOne.<|im_end|>"]
    # After the boundary the completion's own "\n\n" still merges, as the model emits it.
    assert DOUBLE_NEWLINE_ID in tokens[len(prompt) :]


def test_the_fast_path_gives_one_example_when_every_turn_is_a_token_prefix() -> None:
    r = renderer(prefix_stable=True)
    full = r.supervised_example(TOOL_ROW, [WEATHER_TOOL])
    tokens, _ = unshifted(full)
    per_turn = r.per_turn_examples(TOOL_ROW, [WEATHER_TOOL])
    assert trained_runs(r, full) == [run for e in per_turn for run in trained_runs(r, e)]
    for example in per_turn:
        turn_tokens, _ = unshifted(example)
        assert tokens[: len(turn_tokens)] == turn_tokens
    assert r.prefix_stable


def test_the_fast_path_is_refused_when_newlines_merge_across_the_boundary() -> None:
    r = renderer(FakeChatTokenizer(merge_newlines=True), prefix_stable=True)
    with pytest.raises(RendererError) as excinfo:
        r.supervised_example(TWO_TURNS, [])
    assert excinfo.value.code is RendererErrorCode.NOT_PREFIX_STABLE
    assert "messages[1]" in excinfo.value.detail
    assert len(r.per_turn_examples(TWO_TURNS, [])) == 2


def test_the_fast_path_is_refused_when_the_template_drops_reasoning_from_history() -> None:
    r = renderer(prefix_stable=True, chat_template_kwargs={"preserve_thinking": False})
    with pytest.raises(RendererError) as excinfo:
        r.supervised_example(TWO_TURNS, [])
    assert excinfo.value.code is RendererErrorCode.NOT_PREFIX_STABLE
    # The last turn's think block survives in the whole render, so only turn 1 differs.
    assert "messages[1]" in excinfo.value.detail


def test_rows_with_reasoning_always_render_per_turn() -> None:
    r = renderer(prefix_stable=True)
    messages = [TWO_TURNS[0], {**TWO_TURNS[1], "reasoning_content": "Start at one."}]
    with pytest.raises(RendererError) as excinfo:
        r.supervised_example(messages, [])
    assert excinfo.value.code is RendererErrorCode.NOT_PREFIX_STABLE
    [example] = r.per_turn_examples(messages, [])
    assert trained_runs(r, example) == ["Start at one.\n</think>\n\nOne.<|im_end|>"]


def test_tool_call_arguments_reach_the_template_as_objects() -> None:
    tokenizer = FakeChatTokenizer()
    renderer(tokenizer).per_turn_examples(TOOL_ROW, [WEATHER_TOOL])
    call = tokenizer.calls[-1]["messages"][2]["tool_calls"][0]
    assert call["function"]["arguments"] == {"city": "Northvale"}
    assert call["id"] == "call_1"


@pytest.mark.parametrize(
    "call",
    [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": '["Northvale"]'}},
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "not json"}},
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "3"}},
        {"id": "c", "type": "function", "function": {"name": "f"}},
        {"id": "c", "type": "function"},
        "oops",
    ],
    ids=["array", "not_json", "number", "no_arguments", "no_function", "not_a_mapping"],
)
def test_malformed_tool_calls_and_non_object_arguments_are_refused(call: Any) -> None:
    messages = [dict(m) for m in TOOL_ROW[:3]]
    messages[2] = {**messages[2], "tool_calls": [call]}
    with pytest.raises(RendererError) as excinfo:
        renderer().per_turn_examples(messages, [WEATHER_TOOL])
    assert excinfo.value.code is RendererErrorCode.UNSUPPORTED_MESSAGE
    assert "messages[2].tool_calls[0]" in excinfo.value.detail


def test_unset_reasoning_is_left_out_and_set_reasoning_passes_through() -> None:
    tokenizer = FakeChatTokenizer()
    messages = [TWO_TURNS[0], {**TWO_TURNS[1], "reasoning_content": None}]
    renderer(tokenizer).per_turn_examples(messages, [])
    assert "reasoning_content" not in tokenizer.calls[-1]["messages"][1]
    messages[1]["reasoning_content"] = "Think."
    renderer(tokenizer).per_turn_examples(messages, [])
    assert tokenizer.calls[-1]["messages"][1]["reasoning_content"] == "Think."


def test_a_failing_template_is_a_template_error() -> None:
    with pytest.raises(RendererError) as excinfo:
        renderer().per_turn_examples(
            [{"role": "system", "content": "Be brief."}, {"role": "assistant", "content": "Hi."}],
            [],
        )
    assert excinfo.value.code is RendererErrorCode.TEMPLATE_ERROR
    assert "No user query found" in excinfo.value.detail


def test_a_turn_without_its_stop_token_is_a_template_error() -> None:
    r = renderer(FakeChatTokenizer(template=role_tag_template))
    with pytest.raises(RendererError) as excinfo:
        r.per_turn_examples(TWO_TURNS[:2], [])
    assert excinfo.value.code is RendererErrorCode.TEMPLATE_ERROR
    assert "stop token" in excinfo.value.detail


def _prompt_that_the_turn_does_not_extend(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    text = qwen3_8_like(messages, tools, False, template_kwargs)
    return text + "<|im_start|>assistant\nX" if add_generation_prompt else text


def _assistant_turns_render_empty(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    parts = [
        f"<|im_start|>user\n{m['content']}<|im_end|>\n" if m["role"] == "user" else "<|im_start|>"
        for m in messages
    ]
    return "".join(parts) + ("<|im_start|>" if add_generation_prompt else "")


class _LossyFragmentTokenizer(FakeChatTokenizer):
    """Merges newlines, and loses the leading newline of a fragment encoded on its own."""

    def __init__(self) -> None:
        super().__init__(merge_newlines=True)

    def encode(self, text: str) -> list[int]:
        ids = super().encode(text)
        return ids[1:] if text.startswith("\n</think>") else ids


@pytest.mark.parametrize(
    ("tokenizer", "code", "reason"),
    [
        (
            FakeChatTokenizer(template=_prompt_that_the_turn_does_not_extend),
            RendererErrorCode.PROMPT_MISMATCH,
            "does not extend its generation prompt",
        ),
        (_LossyFragmentTokenizer(), RendererErrorCode.PROMPT_MISMATCH, "changed its text"),
        (
            FakeChatTokenizer(template=_assistant_turns_render_empty),
            RendererErrorCode.NO_TRAINABLE_TOKENS,
            "rendered no tokens",
        ),
    ],
    ids=["text_prefix", "round_trip", "empty_turn"],
)
def test_a_turn_that_does_not_continue_its_prompt_is_refused(
    tokenizer: FakeChatTokenizer, code: RendererErrorCode, reason: str
) -> None:
    for r in (renderer(tokenizer), renderer(tokenizer, prefix_stable=True)):
        for render in (r.per_turn_examples, r.supervised_example):
            with pytest.raises(RendererError) as excinfo:
                render(TWO_TURNS[:2], [])
            # Never not_prefix_stable: that would send the row to a second, doomed render.
            assert excinfo.value.code is code
            assert excinfo.value.detail.startswith("messages[1]: ")
            assert reason in excinfo.value.detail


def role_tag_renderer() -> HFChatTemplateRenderer:
    return HFChatTemplateRenderer(
        "example/role-tags",
        SHA,
        response_parser=Qwen3XmlToolParser(turn_end="<|user|>"),
        stop_tokens=["<|user|>", "<|observation|>"],
        next_role_tag=role_tag_after,
        prefix_stable=True,
        tokenizer=FakeChatTokenizer(template=role_tag_template),
    )


def test_role_tag_families_train_the_next_role_tag_as_the_stop() -> None:
    r = role_tag_renderer()
    examples = r.per_turn_examples(TOOL_ROW[1:], [])
    assert [trained_runs(r, e) for e in examples] == [
        [
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nNorthvale\n</parameter>\n"
            "</function>\n</tool_call><|observation|>"
        ],
        ["It is sunny in Northvale.<|user|>"],
    ]
    full = r.supervised_example(TOOL_ROW[1:], [])
    assert trained_runs(r, full) == [run for e in examples for run in trained_runs(r, e)]
    assert r.stop_token_ids == [SPECIAL_IDS["<|user|>"], SPECIAL_IDS["<|observation|>"]]


@pytest.mark.parametrize(
    ("make", "stop", "turn_end"),
    [
        (renderer, "<|im_end|>", "<|im_end|>"),
        (role_tag_renderer, "<|user|>", "<|user|>"),
        (role_tag_renderer, "<|observation|>", "<|user|>"),
    ],
    ids=["stop_token", "role_tag", "other_role_tag"],
)
def test_a_reply_that_spells_a_stop_token_is_refused(
    make: Callable[[], HFChatTemplateRenderer], stop: str, turn_end: str
) -> None:
    # Sampling ends at the first stop, so the text after it could never be generated.
    r = make()
    conversation: list[ChatMessage] = [
        {"role": "user", "content": "Quote it."},
        {"role": "assistant", "content": f"a {stop} b"},
    ]
    for render in (r.per_turn_examples, r.supervised_example):
        with pytest.raises(RendererError) as excinfo:
            render(conversation, [])
        assert excinfo.value.code is RendererErrorCode.UNSUPPORTED_MESSAGE
        assert excinfo.value.detail == (
            "messages[1]: the completion contains an embedded stop token"
        )
    [example] = r.per_turn_examples(TWO_TURNS[:2], [])
    assert trained_runs(r, example)[-1].endswith(f"One.{turn_end}")


def _role_tags_with_a_trailing_newline(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> str:
    text = "".join(
        role_tag_template([m], tools, False, template_kwargs)
        + ("\n" if m["role"] == "assistant" else "")
        for m in messages
    )
    return text + "<|assistant|>" if add_generation_prompt else text


def test_role_tag_families_do_not_train_the_templates_trailer_before_the_tag() -> None:
    r = HFChatTemplateRenderer(
        "example/role-tags",
        SHA,
        response_parser=Qwen3XmlToolParser(turn_end="<|user|>", prompt_opens_think=False),
        stop_tokens=["<|user|>", "<|observation|>"],
        next_role_tag=role_tag_after,
        prefix_stable=True,
        tokenizer=FakeChatTokenizer(template=_role_tags_with_a_trailing_newline),
    )
    examples = r.per_turn_examples(TWO_TURNS, [])
    assert [trained_runs(r, e) for e in examples] == [["One.<|user|>"], ["Two.<|user|>"]]
    # The whole render keeps the newline before the next tag, so the fast path refuses it.
    with pytest.raises(RendererError) as excinfo:
        r.supervised_example(TWO_TURNS, [])
    assert excinfo.value.code is RendererErrorCode.NOT_PREFIX_STABLE


def test_generation_prompt_tokens_tokenize_the_prompt_text_whole() -> None:
    tokenizer = FakeChatTokenizer(merge_newlines=True)
    r = renderer(tokenizer)
    text = tokenizer.render_text(
        TWO_TURNS[:3], tools=[WEATHER_TOOL], add_generation_prompt=True, template_kwargs={}
    )
    assert r.generation_prompt_tokens(TWO_TURNS[:3], [WEATHER_TOOL]) == tokenizer.encode(text)
    assert text.endswith("<|im_start|>assistant\n<think>\n")


def test_chat_template_kwargs_reach_every_render() -> None:
    tokenizer = FakeChatTokenizer()
    r = renderer(tokenizer, chat_template_kwargs={"preserve_thinking": True})
    r.per_turn_examples(TWO_TURNS, [])
    r.generation_prompt_tokens(TWO_TURNS[:1], [])
    assert tokenizer.calls
    assert all(call["template_kwargs"] == {"preserve_thinking": True} for call in tokenizer.calls)


def test_stop_token_ids_resolve_the_family_names() -> None:
    assert renderer().stop_token_ids == [IM_END]
    unknown = HFChatTemplateRenderer(
        "example/tiny-chat",
        SHA,
        response_parser=Qwen3XmlToolParser(),
        stop_tokens=["<|nope|>"],
        tokenizer=FakeChatTokenizer(),
    )
    with pytest.raises(UsageError, match="not a single token"):
        _ = unknown.stop_token_ids


def test_identity_pins_tokenizer_template_parser_and_transformers() -> None:
    r = renderer(chat_template_kwargs={"preserve_thinking": True})
    identity = r.identity()
    assert identity == {
        "renderer": "hf-chat-template",
        "renderer_version": hf.RENDERER_VERSION,
        "name": "hf-chat-template",
        "parser": "qwen3_xml",
        "parser_version": "3",
        "tokenizer": "example/tiny-chat",
        "tokenizer_revision": SHA,
        "chat_template_sha256": hashlib.sha256(b"fake-qwen3.8-like-v1").hexdigest(),
        "chat_template_kwargs": '{"preserve_thinking":true}',
        "stop_tokens": '["<|im_end|>"]',
        "prefix_stable": "false",
        "transformers_version": hf.transformers_version(),
    }
    other = renderer(FakeChatTokenizer(template_name="another-template"))
    assert other.identity()["chat_template_sha256"] != identity["chat_template_sha256"]


@pytest.mark.parametrize("revision", ["main", "v1.0", SHA.upper(), SHA[:39]])
def test_the_revision_must_be_a_full_commit_sha(revision: str) -> None:
    with pytest.raises(ValueError, match="40-character"):
        HFChatTemplateRenderer(
            "example/tiny-chat",
            revision,
            response_parser=Qwen3XmlToolParser(),
            stop_tokens=["<|im_end|>"],
        )


def test_parse_response_uses_the_response_parser() -> None:
    parsed = renderer().parse_response("\n</think>\n\nHello.<|im_end|>")
    assert parsed.format_valid
    assert parsed.message == {"role": "assistant", "content": "Hello."}


def test_the_tokenizer_loads_lazily_and_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[tuple[str, str]] = []

    def fake_load(model: str, revision: str) -> FakeChatTokenizer:
        loads.append((model, revision))
        return FakeChatTokenizer()

    monkeypatch.setattr(hf, "load_tokenizer", fake_load)
    r = HFChatTemplateRenderer(
        "example/tiny-chat", SHA, response_parser=Qwen3XmlToolParser(), stop_tokens=["<|im_end|>"]
    )
    assert loads == []
    r.per_turn_examples(TWO_TURNS, [])
    r.identity()
    assert loads == [("example/tiny-chat", SHA)]


def test_importing_the_cli_and_the_renderer_does_not_import_transformers() -> None:
    code = (
        "import sys, pup_finetune.cli, pup_finetune.rendering.hf, pup_finetune.rendering.families;"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('transformers', 'tokenizers', 'huggingface_hub', 'jinja2'))))"
    )
    result = subprocess.run(  # noqa: S603 - our own interpreter and code
        [sys.executable, "-c", f"import json; {code}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_load_tokenizer_without_transformers_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(UsageError) as excinfo:
        hf.load_tokenizer("example/tiny-chat", SHA)
    assert excinfo.value.hint == hf.INSTALL_HINT


def test_load_tokenizer_pins_the_revision_and_adapts_the_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    stub = _StubHFTokenizer()

    def recorder(*args: Any, **kwargs: Any) -> _StubHFTokenizer:
        calls.append((args, kwargs))
        return stub

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", recorder)
    loaded = hf.load_tokenizer("example/tiny-chat", SHA)
    assert calls == [(("example/tiny-chat",), {"revision": SHA, "trust_remote_code": False})]
    assert isinstance(loaded, TransformersTokenizer)
    assert loaded.chat_template == "{{ messages }}"
    assert loaded.token_id("<|im_end|>") == 1000

    stub.chat_template = None  # type: ignore[assignment]
    with pytest.raises(ExternalError, match="has no chat template") as excinfo:
        hf.load_tokenizer("example/tiny-chat", SHA)
    assert excinfo.value.source == "huggingface"


def _hub_error(kind: str) -> Exception:
    httpx = pytest.importorskip("httpx")
    errors = pytest.importorskip("huggingface_hub.errors")
    request = httpx.Request("GET", "https://huggingface.co/example/tiny-chat")
    hub_error: Exception = getattr(errors, kind)(
        f"{kind} for example/tiny-chat", response=httpx.Response(404, request=request)
    )
    # transformers re-raises Hub errors as the cause of an OSError.
    wrapped = OSError(f"{kind} while loading example/tiny-chat")
    wrapped.__cause__ = hub_error
    return wrapped


@pytest.mark.parametrize(
    ("make_error", "hint"),
    [
        (lambda: _hub_error("GatedRepoError"), "hf auth login"),
        (lambda: _hub_error("RepositoryNotFoundError"), "can read it"),
        (lambda: _hub_error("RevisionNotFoundError"), f"has no commit {SHA}"),
        (lambda: ImportError("sentencepiece is required"), "dependency failed to import"),
        (lambda: OSError("connection refused"), "connection to huggingface.co"),
    ],
    ids=["gated", "missing_repo", "missing_revision", "missing_backend", "network"],
)
def test_a_load_failure_says_why_and_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch, make_error: Callable[[], Exception], hint: str
) -> None:
    transformers = pytest.importorskip("transformers")
    error = make_error()

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", failing)
    with pytest.raises(ExternalError) as excinfo:
        hf.load_tokenizer("example/tiny-chat", SHA)
    assert hint in (excinfo.value.hint or "")
    assert str(error).splitlines()[0] in excinfo.value.message


def test_transformers_failing_to_import_is_not_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(name: str) -> Any:
        raise ImportError("cannot import name 'x' from 'tokenizers'", name="tokenizers")

    monkeypatch.setattr(hf.importlib, "import_module", broken)
    with pytest.raises(UsageError, match="failed to import") as excinfo:
        hf.load_tokenizer("example/tiny-chat", SHA)
    assert "cannot import name 'x'" in excinfo.value.message


def test_a_tokenizer_that_cannot_load_is_an_external_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", unreachable)
    with pytest.raises(ExternalError) as excinfo:
        hf.load_tokenizer("example/tiny-chat", SHA)
    assert excinfo.value.source == "huggingface"
    assert SHA in excinfo.value.message
    assert "OSError" in excinfo.value.message


class _StubHFTokenizer:
    """The slice of a ``transformers`` tokenizer :class:`TransformersTokenizer` calls."""

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "".join(m["content"] for m in messages)

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [1000] if text == "<|im_end|>" else [ord(c) for c in text]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join("<|im_end|>" if i == 1000 else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        # Like a tokenizer with an unknown token: anything else is the unk id.
        return {"<|im_end|>": 1000, "a": 97}.get(token, 0)


def test_the_transformers_adapter_renders_text_and_resolves_single_tokens() -> None:
    stub = _StubHFTokenizer()
    adapter = TransformersTokenizer(stub)
    text = adapter.render_text(
        [{"role": "user", "content": "Hi"}],
        tools=[],
        add_generation_prompt=True,
        template_kwargs={"enable_thinking": False},
    )
    assert text == "Hi"
    assert stub.kwargs == {
        "tools": None,
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    assert adapter.encode("Hi") == [72, 105]
    assert adapter.decode([72, 1000]) == "H<|im_end|>"
    assert adapter.token_id("<|im_end|>") == 1000
    assert adapter.token_id("a") == 97
    assert adapter.token_id("<|nope|>") is None  # the unk id, not the token
    assert adapter.chat_template == "{{ messages }}"
    stub.chat_template = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no chat template"):
        _ = adapter.chat_template


def test_training_renders_rows_the_fast_path_refuses_per_turn() -> None:
    r = renderer(FakeChatTokenizer(merge_newlines=True), prefix_stable=True)
    stable = renderer(prefix_stable=True)
    rows = [{"messages": TWO_TURNS, "tools": []}, {"messages": TOOL_ROW, "tools": [WEATHER_TOOL]}]
    result = build_examples(rows, r, max_seq_len=4096)
    assert result.rejected == []
    assert [len(row.examples) for row in result.examples] == [2, 2]
    assert result.examples[0].examples == tuple(r.per_turn_examples(TWO_TURNS, []))
    # Without the merge the same rows take the fast path, one example each.
    fast = build_examples(rows, stable, max_seq_len=4096)
    assert [len(row.examples) for row in fast.examples] == [1, 1]
