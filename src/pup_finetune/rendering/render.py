# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Render canonical rows into token datums, one per trained assistant message.

Each target message is rendered after its own inference prompt (the chat template with the
generation prompt on), so the loss covers exactly what the model generates after that prompt,
up to and including its stop token. Everything before it is zero-loss history. A row renders
all or nothing and is never truncated.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pup_finetune.data.schema import Message, TrainingRow
from pup_finetune.rendering.types import (
    ChatTokenizer,
    ModelFamily,
    RenderError,
    RenderErrorCode,
    TokenDatum,
)

_GLM_TOOL_STOP = "<|observation|>"
_GLM_REPLY_STOP = "<|user|>"


@dataclass(frozen=True, slots=True)
class RenderReport:
    """The rows that render, the rows that do not and why, and token counts of the accepted."""

    accepted: list[tuple[TrainingRow, list[TokenDatum]]]
    rejected: list[dict[str, Any]]
    """``{"id", "code", "detail"}``, plus ``"tokens"`` and ``"limit"`` for ``exceeds_context``."""
    stats: dict[str, int]


def render_row(
    row: TrainingRow, family: ModelFamily, tokenizer: ChatTokenizer, *, max_seq_len: int
) -> list[TokenDatum]:
    """One datum per trained assistant message of ``row``, in order; raise :class:`RenderError`.

    An assistant message is trained unless its ``weight`` is ``0`` or it has no text, tool calls
    or reasoning; untrained messages stay in later datums as zero-loss history.
    """
    if not family.supports_reasoning and any(m.reasoning_content for m in row.messages):
        raise RenderError(
            RenderErrorCode.REASONING_UNSUPPORTED,
            f"model family {family.key!r} cannot render reasoning_content",
        )
    targets = [index for index, message in enumerate(row.messages) if _is_target(message)]
    if not targets:
        raise RenderError(
            RenderErrorCode.NO_TRAINABLE_TURN,
            "no assistant message has content to train on and a weight other than 0",
        )
    messages = _template_messages(
        [m.model_dump(mode="json", exclude_none=True) for m in row.messages]
    )
    tools = [tool.model_dump(mode="json", exclude_none=True) for tool in row.tools] or None
    datums = [_render_target(messages, tools, index, family, tokenizer) for index in targets]
    longest = max(len(datum.tokens) for datum in datums)
    if longest > max_seq_len:
        raise RenderError(
            RenderErrorCode.EXCEEDS_CONTEXT,
            f"a rendered datum has {longest} tokens, over the limit of {max_seq_len}",
            tokens=longest,
            limit=max_seq_len,
        )
    return datums


def prompt_tokens(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    family: ModelFamily,
    tokenizer: ChatTokenizer,
) -> list[int]:
    """The inference prompt after ``messages`` (OpenAI chat format), as the model sees it."""
    return _render(
        tokenizer,
        _template_messages(messages),
        [dict(tool) for tool in tools] if tools else None,
        family,
        add_generation_prompt=True,
        where="prompt",
    )


def check_rows(
    rows: Iterable[TrainingRow],
    family: ModelFamily,
    tokenizer: ChatTokenizer,
    *,
    max_seq_len: int,
) -> RenderReport:
    """Render every row; a row that cannot render is rejected with its code, not raised."""
    accepted: list[tuple[TrainingRow, list[TokenDatum]]] = []
    rejected: list[dict[str, Any]] = []
    datums_count = context_tokens = target_tokens = max_context = 0
    for row in rows:
        try:
            datums = render_row(row, family, tokenizer, max_seq_len=max_seq_len)
        except RenderError as exc:
            rejected.append(_rejection(row.id, exc))
            continue
        accepted.append((row, datums))
        datums_count += len(datums)
        context_tokens += sum(len(datum.tokens) for datum in datums)
        target_tokens += sum(sum(datum.weights) for datum in datums)
        max_context = max(max_context, *(len(datum.tokens) for datum in datums))
    return RenderReport(
        accepted=accepted,
        rejected=rejected,
        stats={
            "rendered_datums": datums_count,
            "assistant_targets": datums_count,
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "max_context_tokens": max_context,
            "rejected_rows": len(rejected),
        },
    )


def _is_target(message: Message) -> bool:
    return (
        message.role == "assistant"
        and message.weight != 0
        and bool(message.content or message.tool_calls or message.reasoning_content)
    )


def _template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Messages as chat templates take them: no ``weight``, tool arguments as objects."""
    result: list[dict[str, Any]] = []
    for message in messages:
        data = {key: value for key, value in message.items() if key != "weight"}
        if data.get("tool_calls"):
            data["tool_calls"] = [_with_parsed_arguments(call) for call in data["tool_calls"]]
        result.append(data)
    return result


def _with_parsed_arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    # Chat templates iterate ``arguments|items``, so they need the object, not its JSON string.
    function = dict(call["function"])
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RenderError(
                RenderErrorCode.TEMPLATE_FAILED, "tool call arguments are not valid JSON"
            ) from exc
    if not isinstance(arguments, dict):
        raise RenderError(RenderErrorCode.TEMPLATE_FAILED, "tool call arguments must be an object")
    return {**call, "function": {**function, "arguments": arguments}}


def _render_target(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    index: int,
    family: ModelFamily,
    tokenizer: ChatTokenizer,
) -> TokenDatum:
    where = f"messages[{index}]"
    prompt = _render(
        tokenizer, messages[:index], tools, family, add_generation_prompt=True, where=where
    )
    tokens = _render(
        tokenizer, messages[: index + 1], tools, family, add_generation_prompt=False, where=where
    )
    if tokens[: len(prompt)] != prompt:
        tokens = _extend_prompt(tokenizer, prompt, tokens, where)
    if len(tokens) <= len(prompt):
        raise RenderError(
            RenderErrorCode.NO_TARGET_TOKENS, f"{where}: the reply rendered no tokens"
        )
    if family.boundary == "glm_stop_tag":
        stop = _GLM_TOOL_STOP if messages[index].get("tool_calls") else _GLM_REPLY_STOP
        tokens.append(_single_token(tokenizer, stop))
        end = len(tokens)
    else:
        end = _reply_end(tokens, len(prompt), family, tokenizer, where)
    weights = (0,) * len(prompt) + (1,) * (end - len(prompt)) + (0,) * (len(tokens) - end)
    return TokenDatum(tokens=tuple(tokens), weights=weights, message_index=index)


def _render(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    family: ModelFamily,
    *,
    add_generation_prompt: bool,
    where: str,
) -> list[int]:
    try:
        tokens = tokenizer.render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=family.template_kwargs,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
        raise RenderError(
            RenderErrorCode.TEMPLATE_FAILED, f"{where}: the chat template failed: {reason}"
        ) from exc
    if not tokens:
        raise RenderError(
            RenderErrorCode.INVALID_TOKENS, f"{where}: the chat template rendered no tokens"
        )
    return _checked(tokens, where)


def _checked(tokens: Sequence[int], where: str) -> list[int]:
    if any(type(token) is not int or token < 0 for token in tokens):
        raise RenderError(
            RenderErrorCode.INVALID_TOKENS, f"{where}: the tokenizer returned invalid token ids"
        )
    return list(tokens)


def _extend_prompt(
    tokenizer: ChatTokenizer, prompt: list[int], tokens: list[int], where: str
) -> list[int]:
    """The prompt's tokens plus the reply re-encoded, when BPE merged across their boundary.

    E.g. Qwen's prompt ends in ``<think>\\n`` and an empty thinking block continues ``\\n``:
    the conversation encodes ``\\n\\n`` as one token, but the model generates after the prompt.
    """
    prompt_text = tokenizer.decode(prompt)
    full_text = tokenizer.decode(tokens)
    if not full_text.startswith(prompt_text):
        raise RenderError(
            RenderErrorCode.PREFIX_MISMATCH,
            f"{where}: the rendered conversation does not extend its inference prompt",
        )
    extended = prompt + _checked(tokenizer.encode(full_text[len(prompt_text) :]), where)
    if tokenizer.decode(extended) != full_text:
        raise RenderError(
            RenderErrorCode.PREFIX_MISMATCH, f"{where}: re-encoding the reply changed its text"
        )
    return extended


def _reply_end(
    tokens: list[int], start: int, family: ModelFamily, tokenizer: ChatTokenizer, where: str
) -> int:
    """The index after the reply's last stop token; template text after it is not a target."""
    stops = {_single_token(tokenizer, stop) for stop in family.stop_sequences}
    for position in range(len(tokens) - 1, start - 1, -1):
        if tokens[position] in stops:
            return position + 1
    raise RenderError(
        RenderErrorCode.TEMPLATE_FAILED, f"{where}: the reply does not end with a stop sequence"
    )


def _single_token(tokenizer: ChatTokenizer, token: str) -> int:
    token_id = tokenizer.token_id(token)
    if token_id is None:
        raise RenderError(
            RenderErrorCode.TEMPLATE_FAILED, f"{token!r} is not a single token of this tokenizer"
        )
    return token_id


def _rejection(row_id: str, error: RenderError) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": row_id, "code": error.code.value, "detail": error.detail}
    if error.tokens is not None:
        entry["tokens"] = error.tokens
    if error.limit is not None:
        entry["limit"] = error.limit
    return entry
