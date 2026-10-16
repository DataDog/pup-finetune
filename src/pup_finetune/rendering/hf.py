# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The shared :class:`~pup_finetune.rendering.base.Renderer` on Hugging Face chat templates.

:class:`HFChatTemplateRenderer` renders conversations with a model's own chat template and
tokenizer, pinned to a commit, so every provider serving the model trains on the tokens its
deployed chat endpoint sees. It is provider-neutral; model families configure it (see
:mod:`pup_finetune.rendering.families`). Adapted from the Baseten provider work by Fouad
Wahabi.

**Per turn, by default.** Each trainable assistant turn ``i`` is one example: the generation
prompt for ``messages[:i]``, rendered and tokenized by the template **whole** (exactly how a
chat endpoint tokenizes a prompt), followed by the turn's completion tokens. The loss
covers the completion through its stop token and nothing after it. When the tokenizer merges
characters across the prompt boundary (Qwen3.8's prompt ends ``<think>\\n`` and an empty
reasoning block continues ``\\n``, which the whole render encodes as one ``\\n\\n`` token), the
completion is the rest of the rendered text encoded on its own, after the prompt's tokens.
Families whose models stop by emitting the next role tag (no end-of-turn token) append that
tag to the completion as its stop target, after dropping any whitespace the template renders
after the reply. A turn whose render does not continue its own prompt is refused with
``prompt_mismatch``; one whose completion holds a stop token before its own stop (content
that spells ``<|im_end|>`` encodes to that token) with ``unsupported_message``.

**One example per row, when it is exact.** A renderer built with ``prefix_stable=True`` (the
family pins it) also offers :meth:`HFChatTemplateRenderer.supervised_example`: the whole
conversation tokenized once with every trainable turn in the loss. It is checked per row: for
every trainable turn, that turn's per-turn example must be an exact token prefix of the whole
render, which also catches templates that drop reasoning from earlier turns. Rows carrying
``reasoning_content`` always render per turn. A row that fails raises :class:`RendererError`
with ``not_prefix_stable`` and :func:`~pup_finetune.training.datums.build_examples` renders it
per turn instead, so ``prefix_stable`` never claims more than each row delivers.

**The template's own tokens.** Prompts and turns are tokenized as the template tokenizes
them (:meth:`ChatTokenizer.render_tokens`): for a Jinja template that is the rendered text
encoded whole; a tokenizer whose own code renders the chat (``trust_remote_code``, Kimi K3)
may encode message text apart from its control tokens, so text that spells a control token
stays text. A family whose tokenizer ships no chat template renders through a
:class:`ChatEncoder` instead (:class:`EncodedChatTokenizer`, :mod:`.encoders`).

``transformers`` comes from the optional ``hf`` extra and is imported only when a tokenizer
is first needed, never when this module is imported. A family's ``trust_remote_code`` is
passed to it with the pinned revision, so that code comes from the model's repository at that
commit. Tests inject a :class:`ChatTokenizer`.
A tokenizer that cannot be loaded fails every row alike, so it is an :class:`ExternalError`
(a :class:`UsageError` when the extra is missing), never a per-row :class:`RendererError`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from pup_finetune.errors import ExternalError, UsageError, install_hint
from pup_finetune.redaction import redact
from pup_finetune.rendering.base import (
    ChatMessage,
    ParsedResponse,
    RendererError,
    RendererErrorCode,
    ToolSpec,
)
from pup_finetune.rendering.parsers import ResponseParser
from pup_finetune.rendering.tokens import TokenizedExample
from pup_finetune.storage import canonical_json

RENDERER = "hf-chat-template"
"""The implementation name recorded in :meth:`HFChatTemplateRenderer.identity`."""
RENDERER_VERSION = "1"
"""Bumped whenever the tokens rendered for some conversation change."""

INSTALL_HINT = install_hint("hf")
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
"""A full, lowercase git commit SHA: how tokenizer revisions are pinned (match with fullmatch)."""


class ChatTokenizer(Protocol):
    """The narrow tokenizer surface rendering needs; :class:`TransformersTokenizer` adapts HF."""

    def render_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> str:
        """``messages`` rendered through the chat template, as text."""
        ...

    def render_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        """``messages`` rendered through the chat template, as the template tokenizes them."""
        ...

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` as one string, adding no special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """The text of ``token_ids``, special tokens included."""
        ...

    def token_id(self, token: str) -> int | None:
        """The id of ``token`` when it is a single token of the vocabulary, else ``None``."""
        ...

    @property
    def chat_template(self) -> str:
        """The chat template's source."""
        ...


class TransformersTokenizer:
    """A ``transformers`` tokenizer adapted to :class:`ChatTokenizer`.

    ``remote_code`` says the tokenizer's code came from the model's repository; without a
    Jinja template, that code renders the chat, and its source stands in as
    :attr:`chat_template`.
    """

    def __init__(self, tokenizer: Any, *, remote_code: bool = False) -> None:
        self._tokenizer = tokenizer
        self._remote_code = remote_code

    def render_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> str:
        text = self._apply(messages, tools, add_generation_prompt, template_kwargs, tokenize=False)
        if not isinstance(text, str):
            raise TypeError(f"the chat template rendered {type(text).__name__}, not text")
        return text

    def render_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        tokens = self._apply(messages, tools, add_generation_prompt, template_kwargs, tokenize=True)
        if not isinstance(tokens, Sequence) or isinstance(tokens, str):
            raise TypeError(f"the chat template tokenized to {type(tokens).__name__}, not ids")
        return [int(token) for token in tokens]

    def _apply(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
        *,
        tokenize: bool,
    ) -> Any:
        return self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tools=[dict(tool) for tool in tools] if tools else None,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_dict=False,
            **template_kwargs,
        )

    def encode(self, text: str) -> list[int]:
        # Calling the tokenizer, not its encode: remote tokenizers (Kimi K3) log a warning on
        # every encode that is passed arguments.
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return [int(token) for token in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self._tokenizer.decode(
                list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )

    def token_id(self, token: str) -> int | None:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and self.encode(token) == [token_id]:
            return token_id
        return None

    @property
    def chat_template(self) -> str:
        template = self._tokenizer.chat_template
        if isinstance(template, str) and template:
            return template
        if self._remote_code:
            return _remote_source(self._tokenizer)
        raise ValueError("the tokenizer has no chat template")


def _remote_source(tokenizer: Any) -> str:
    """The source of the remote modules loaded for ``tokenizer``, read from the package that
    ``transformers`` stored the repository's code in at the pinned revision."""
    package = type(tokenizer).__module__.rpartition(".")[0]
    files = sorted(
        str(module.__file__)
        for name, module in list(sys.modules.items())
        if name.startswith(f"{package}.") and getattr(module, "__file__", None)
    )
    return "".join(
        f"# {Path(path).name}\n{Path(path).read_text(encoding='utf-8')}" for path in files
    )


def load_tokenizer(
    model: str, revision: str, *, trust_remote_code: bool = False, template: bool = True
) -> TransformersTokenizer:
    """``model``'s tokenizer at ``revision``, from the Hugging Face cache or Hub.

    ``trust_remote_code`` runs the tokenizer code of the model's repository at ``revision``;
    ``template`` false loads a tokenizer whose chat a :class:`ChatEncoder` renders. Raises
    :class:`UsageError` when ``transformers`` is not installed and :class:`ExternalError`
    when the tokenizer cannot be loaded or, with ``template``, renders no chat.
    """
    transformers = _transformers()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        raise _load_failure(model, revision, exc) from exc
    adapter = TransformersTokenizer(tokenizer, remote_code=trust_remote_code)
    if not template:
        return adapter
    try:
        adapter.chat_template  # noqa: B018 - fail at load, not at the first render
    except ValueError as exc:
        raise ExternalError(
            f"the {model} tokenizer at revision {revision} has no chat template",
            source="huggingface",
        ) from exc
    return adapter


class ChatEncoder(Protocol):
    """Messages to the text of a model's chat format, for a tokenizer with no chat template."""

    @property
    def name(self) -> str:
        """The encoder's key, as model families name it."""
        ...

    @property
    def version(self) -> str:
        """Bumped whenever the text rendered for some conversation changes."""
        ...

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        add_generation_prompt: bool,
    ) -> str:
        """``messages`` (tool-call arguments as objects) and ``tools`` as text."""
        ...


@dataclass(frozen=True, slots=True)
class EncodedChatTokenizer:
    """``tokenizer`` with its chat rendered by ``encoder``; the rest is the tokenizer's."""

    tokenizer: ChatTokenizer
    encoder: ChatEncoder

    def render_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> str:
        return self.encoder.render(messages, tools, add_generation_prompt=add_generation_prompt)

    def render_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        return self.encode(
            self.render_text(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                template_kwargs=template_kwargs,
            )
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def token_id(self, token: str) -> int | None:
        return self.tokenizer.token_id(token)

    @property
    def chat_template(self) -> str:
        return f"pup_finetune.rendering.encoders:{self.encoder.name}@{self.encoder.version}"


def _load_failure(model: str, revision: str, exc: Exception) -> ExternalError:
    """Why ``model``'s tokenizer did not load, with a hint that fits the cause."""
    # Match Hugging Face Hub errors by name: transformers re-raises them as the cause of an
    # OSError, and naming them here keeps huggingface_hub out of this module's imports.
    causes: list[BaseException] = []
    cause: BaseException | None = exc
    while cause is not None and cause not in causes:
        causes.append(cause)
        cause = cause.__cause__ or cause.__context__
    kinds = {kind.__name__ for cause in causes for kind in type(cause).__mro__}
    if "GatedRepoError" in kinds or "RepositoryNotFoundError" in kinds:
        hint = (
            f"check that {model} exists and that your Hugging Face account can read it "
            "(accept its terms on huggingface.co, then hf auth login)"
        )
    elif "RevisionNotFoundError" in kinds:
        hint = f"{model} has no commit {revision}; update the model family's pinned revision"
    elif "ImportError" in kinds:
        hint = f"a tokenizer dependency failed to import; {INSTALL_HINT}"
    else:
        hint = "check the connection to huggingface.co, or the Hugging Face cache when offline"
    lines = redact(str(exc)).strip().splitlines()
    reason = f"{type(exc).__name__}: {lines[0][:200]}" if lines else type(exc).__name__
    return ExternalError(
        f"could not load the {model} tokenizer at revision {revision} ({reason})",
        source="huggingface",
        hint=hint,
    )


def _transformers() -> Any:
    # Tokenizers need no PyTorch; hide transformers' import-time advice that models are absent.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        if exc.name == "transformers":
            raise UsageError(
                "Hugging Face transformers is not installed", hint=INSTALL_HINT
            ) from exc
        reason = redact(str(exc)).strip().splitlines()[0][:200] if str(exc).strip() else ""
        raise UsageError(
            f"Hugging Face transformers failed to import ({type(exc).__name__}: {reason})",
            hint=f"reinstall its dependencies; {INSTALL_HINT}",
        ) from exc


def transformers_version() -> str:
    """The installed ``transformers`` version, or ``"not installed"``."""
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return "not installed"


@dataclass(frozen=True, slots=True)
class _Turn:
    """One assistant turn rendered against its own prefix: ``tokens[start:]`` is trained."""

    tokens: list[int]
    start: int

    def example(self) -> TokenizedExample:
        weights = [0.0] * self.start + [1.0] * (len(self.tokens) - self.start)
        return TokenizedExample.from_tokens(self.tokens, weights)


class HFChatTemplateRenderer:
    """A model's Hugging Face chat template and tokenizer as a renderer. See the module docs.

    ``tokenizer_model`` and ``revision`` (a 40-character commit SHA) pin the tokenizer and
    template; ``tokenizer`` injects a :class:`ChatTokenizer` instead of loading one, for tests.
    ``stop_tokens`` name the tokens that end an assistant turn. ``next_role_tag``, for models
    that stop by emitting the next role tag, gives the tag that follows an assistant message;
    it is appended to that turn's completion as its stop target. ``chat_template_kwargs`` are
    passed to every template render. ``trust_remote_code`` loads the tokenizer with its
    repository's code; ``chat_encoder`` renders the chat for a tokenizer without a template;
    ``check`` names why the template cannot render a conversation (``None`` when it can), and
    such a row is refused with ``unsupported_message``.
    """

    def __init__(
        self,
        tokenizer_model: str,
        revision: str,
        *,
        response_parser: ResponseParser,
        stop_tokens: Sequence[str],
        prefix_stable: bool = False,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        next_role_tag: Callable[[ChatMessage], str] | None = None,
        trust_remote_code: bool = False,
        chat_encoder: ChatEncoder | None = None,
        check: Callable[[Sequence[ChatMessage]], str | None] | None = None,
        name: str = RENDERER,
        tokenizer: ChatTokenizer | None = None,
    ) -> None:
        if not COMMIT_SHA.fullmatch(revision):
            raise ValueError(
                f"revision must be a full 40-character lowercase commit SHA, got {revision!r}"
            )
        if not stop_tokens:
            raise ValueError("a renderer needs at least one stop token")
        self._tokenizer_model = tokenizer_model
        self._revision = revision
        self._parser = response_parser
        self._stop_tokens = tuple(stop_tokens)
        self._prefix_stable = prefix_stable
        self._template_kwargs = MappingProxyType(dict(chat_template_kwargs or {}))
        self._next_role_tag = next_role_tag
        self._trust_remote_code = trust_remote_code
        self._encoder = chat_encoder
        self._check = check
        self._name = name
        self._tokenizer = (
            EncodedChatTokenizer(tokenizer, chat_encoder)
            if tokenizer is not None and chat_encoder is not None
            else tokenizer
        )
        self._stop_token_ids: tuple[int, ...] | None = None
        self._lock = threading.Lock()

    # The Renderer protocol.

    @property
    def name(self) -> str:
        return self._name

    @property
    def prefix_stable(self) -> bool:
        """Whether :meth:`supervised_example` may be tried; it still checks every row."""
        return self._prefix_stable

    @property
    def stop_token_ids(self) -> list[int]:
        return list(self._stop_ids())

    def supervised_example(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> TokenizedExample:
        """The whole row as one example, when that trains exactly what the turns would.

        Raises :class:`RendererError` with ``not_prefix_stable`` when the row has
        ``reasoning_content``, a trainable turn's per-turn example is not a token prefix of
        the whole render, or an untrained turn's generation prompt is not: one example claims
        every turn saw the history the whole render shows, and a template that rewrites
        earlier turns later (DeepSeek-V4 without tools) breaks that claim even when only the
        last turn trains. Render it with :meth:`per_turn_examples` then.
        """
        turns = _trainable_turns(messages)
        self._refuse_unsupported(messages)
        if any(message.get("reasoning_content") for message in messages):
            raise RendererError(
                RendererErrorCode.NOT_PREFIX_STABLE, "rows with reasoning_content render per turn"
            )
        prepared = template_messages(messages)
        declared = list(tools) or None
        tokens = self._tokens(prepared, declared, generation=False, where="the conversation")
        if self._next_role_tag is not None and prepared[-1].get("role") == "assistant":
            tokens.append(self._single_token(self._next_role_tag(prepared[-1])))
        for index in _assistant_turns(messages):
            if index in turns:
                continue
            where = f"messages[{index}]"
            prompt = self._tokens(prepared[:index], declared, generation=True, where=where)
            if tokens[: len(prompt)] != prompt:
                raise RendererError(
                    RendererErrorCode.NOT_PREFIX_STABLE,
                    f"{where}: the turn's prompt is not a prefix of the whole render",
                )
        weights = [0.0] * len(tokens)
        end = 0
        for index in turns:
            turn = self._turn(prepared, declared, index)
            if tokens[: len(turn.tokens)] != turn.tokens:
                raise RendererError(
                    RendererErrorCode.NOT_PREFIX_STABLE,
                    f"messages[{index}]: the turn's tokens are not a prefix of the whole render",
                )
            weights[turn.start : len(turn.tokens)] = [1.0] * (len(turn.tokens) - turn.start)
            end = len(turn.tokens)
        return TokenizedExample.from_tokens(tokens[:end], weights[:end])

    def per_turn_examples(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[TokenizedExample]:
        turns = _trainable_turns(messages)
        self._refuse_unsupported(messages)
        prepared = template_messages(messages)
        declared = list(tools) or None
        return [self._turn(prepared, declared, index).example() for index in turns]

    def generation_prompt_tokens(
        self, messages: list[ChatMessage], tools: list[ToolSpec]
    ) -> list[int]:
        """The prompt the template renders for the next turn, tokenized as the template does."""
        self._refuse_unsupported(messages)
        prepared = template_messages(messages)
        return self._tokens(prepared, list(tools) or None, generation=True, where="the prompt")

    def decode(self, tokens: Sequence[int]) -> str:
        return self._chat_tokenizer().decode(tokens)

    def parse_response(self, text: str, tools: Sequence[ToolSpec] = ()) -> ParsedResponse:
        return self._parser.parse(text, tools)

    def identity(self) -> dict[str, str]:
        template = self._chat_tokenizer().chat_template
        return {
            "renderer": RENDERER,
            "renderer_version": RENDERER_VERSION,
            "name": self._name,
            "parser": self._parser.name,
            "parser_version": self._parser.version,
            "tokenizer": self._tokenizer_model,
            "tokenizer_revision": self._revision,
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "chat_template_kwargs": canonical_json(dict(self._template_kwargs)),
            "stop_tokens": canonical_json(list(self._stop_tokens)),
            "prefix_stable": "true" if self._prefix_stable else "false",
            "transformers_version": transformers_version(),
        }

    # Internals.

    def _chat_tokenizer(self) -> ChatTokenizer:
        with self._lock:
            if self._tokenizer is None:
                loaded = load_tokenizer(
                    self._tokenizer_model,
                    self._revision,
                    trust_remote_code=self._trust_remote_code,
                    template=self._encoder is None,
                )
                self._tokenizer = (
                    loaded if self._encoder is None else EncodedChatTokenizer(loaded, self._encoder)
                )
            return self._tokenizer

    def _refuse_unsupported(self, messages: Sequence[ChatMessage]) -> None:
        reason = self._check(messages) if self._check is not None else None
        if reason is not None:
            raise RendererError(RendererErrorCode.UNSUPPORTED_MESSAGE, reason)

    def _stop_ids(self) -> tuple[int, ...]:
        if self._stop_token_ids is None:
            self._stop_token_ids = tuple(self._single_token(token) for token in self._stop_tokens)
        return self._stop_token_ids

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        *,
        generation: bool,
        where: str,
    ) -> str:
        tokenizer = self._chat_tokenizer()
        try:
            return tokenizer.render_text(
                messages,
                tools=tools,
                add_generation_prompt=generation,
                template_kwargs=self._template_kwargs,
            )
        except Exception as exc:
            raise _template_failure(where, exc) from exc

    def _tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        *,
        generation: bool,
        where: str,
    ) -> list[int]:
        tokenizer = self._chat_tokenizer()
        try:
            tokens = tokenizer.render_tokens(
                messages,
                tools=tools,
                add_generation_prompt=generation,
                template_kwargs=self._template_kwargs,
            )
        except Exception as exc:
            raise _template_failure(where, exc) from exc
        return _valid_ids(tokens)

    def _encode(self, text: str) -> list[int]:
        return _valid_ids(self._chat_tokenizer().encode(text))

    def _single_token(self, token: str) -> int:
        token_id = self._chat_tokenizer().token_id(token)
        if token_id is None:
            raise UsageError(
                f"{token!r} is not a single token of the {self._tokenizer_model} tokenizer",
                hint="fix the model family's stop tokens",
            )
        return token_id

    def _turn(
        self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None, index: int
    ) -> _Turn:
        """Assistant turn ``index`` after its own generation prompt; see the module docs."""
        where = f"messages[{index}]"
        prompt_text = self._render(messages[:index], tools, generation=True, where=where)
        text = self._render(messages[: index + 1], tools, generation=False, where=where)
        if not text.startswith(prompt_text):
            raise RendererError(
                RendererErrorCode.PROMPT_MISMATCH,
                f"{where}: the rendered turn does not extend its generation prompt",
            )
        prompt = self._tokens(messages[:index], tools, generation=True, where=where)
        tokens = self._tokens(messages[: index + 1], tools, generation=False, where=where)
        if tokens[: len(prompt)] != prompt:
            # The tokenizer merged across the boundary; the model generates after the prompt.
            whole = tokens
            tokens = prompt + self._encode(text[len(prompt_text) :])
            if self.decode(tokens) != self.decode(whole):
                raise RendererError(
                    RendererErrorCode.PROMPT_MISMATCH,
                    f"{where}: encoding the turn after its prompt changed its text",
                )
        if len(tokens) <= len(prompt):
            raise RendererError(
                RendererErrorCode.NO_TRAINABLE_TOKENS, f"{where}: the turn rendered no tokens"
            )
        stops = set(self._stop_ids())
        if self._next_role_tag is not None:
            # The reply ends where the template's trailing whitespace starts; the tag follows.
            end = len(tokens)
            while end > len(prompt) and not self.decode(tokens[end - 1 : end]).strip():
                end -= 1
            if end == len(prompt):
                raise RendererError(
                    RendererErrorCode.NO_TRAINABLE_TOKENS, f"{where}: the turn rendered no text"
                )
            _refuse_embedded_stop(tokens[len(prompt) : end], stops, where)
            tag = self._single_token(self._next_role_tag(messages[index]))
            return _Turn([*tokens[:end], tag], len(prompt))
        for end in range(len(tokens), len(prompt), -1):
            if tokens[end - 1] in stops:
                _refuse_embedded_stop(tokens[len(prompt) : end - 1], stops, where)
                return _Turn(tokens[:end], len(prompt))
        raise RendererError(
            RendererErrorCode.TEMPLATE_ERROR, f"{where}: the turn does not end with a stop token"
        )


def _template_failure(where: str, exc: Exception) -> RendererError:
    reason = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
    return RendererError(
        RendererErrorCode.TEMPLATE_ERROR, f"{where}: the chat template failed: {reason}"
    )


def _valid_ids(tokens: Sequence[int]) -> list[int]:
    if any(type(token) is not int or token < 0 for token in tokens):
        raise RendererError(
            RendererErrorCode.TEMPLATE_ERROR, "the tokenizer returned invalid token ids"
        )
    return list(tokens)


def _refuse_embedded_stop(completion: Sequence[int], stops: set[int], where: str) -> None:
    """Refuse a completion holding a stop token before its own stop target.

    Content that spells a stop token (``<|im_end|>`` in a reply) encodes to its special id.
    Training on it would teach text after a stop, which sampling, ending at the first stop,
    can never produce.
    """
    if any(token in stops for token in completion):
        raise RendererError(
            RendererErrorCode.UNSUPPORTED_MESSAGE,
            f"{where}: the completion contains an embedded stop token",
        )


def _assistant_turns(messages: Sequence[ChatMessage]) -> list[int]:
    return [index for index, message in enumerate(messages) if message.get("role") == "assistant"]


def _trainable_turns(messages: Sequence[ChatMessage]) -> list[int]:
    turns = [index for index in _assistant_turns(messages) if messages[index].get("weight") != 0]
    if not turns:
        raise RendererError(
            RendererErrorCode.NO_TRAINABLE_TOKENS, "no assistant turn has a non-zero weight"
        )
    return turns


def template_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Messages as chat templates take them.

    ``weight`` is ours and dropped (the turn selection already honored it), unset
    ``reasoning_content`` is left out, and tool-call ``arguments`` become objects, since
    templates iterate them as a mapping.
    """
    prepared: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        data = {key: value for key, value in message.items() if key != "weight"}
        if data.get("reasoning_content") is None:
            data.pop("reasoning_content", None)
        if data.get("tool_calls"):
            data["tool_calls"] = [
                _with_object_arguments(call, f"messages[{index}].tool_calls[{position}]")
                for position, call in enumerate(data["tool_calls"])
            ]
        prepared.append(data)
    return prepared


def _with_object_arguments(call: Any, where: str) -> dict[str, Any]:
    if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
        raise RendererError(RendererErrorCode.UNSUPPORTED_MESSAGE, f"{where}: malformed tool call")
    function = dict(call["function"])
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if not isinstance(arguments, dict):
        raise RendererError(
            RendererErrorCode.UNSUPPORTED_MESSAGE, f"{where}: arguments must be a JSON object"
        )
    return {**call, "function": {**function, "arguments": arguments}}
