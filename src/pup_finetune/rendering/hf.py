# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Hugging Face tokenizers behind :class:`ChatTokenizer`; the only module importing transformers.

``transformers`` comes from the optional ``hf`` extra and is imported only when a tokenizer
is loaded, so the rest of the package works without it.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

from pup_finetune.errors import ExternalError, UsageError
from pup_finetune.rendering.types import ChatTokenizer, ModelFamily, RenderingIdentity

RENDERING_VERSION = "pup-finetune-prefix-v1"

_INSTALL_HINT = "install the hf extra: pip install 'pup-finetune[hf]'"


class HFChatTokenizer:
    """A ``transformers`` tokenizer adapted to :class:`ChatTokenizer`."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: Mapping[str, Any],
    ) -> list[int]:
        result = self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tools=[dict(tool) for tool in tools] if tools else None,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
        # transformers 5 defaults to a BatchEncoding; normalize in case return_dict is ignored.
        if isinstance(result, Mapping):
            result = result["input_ids"]
        return list(result)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self._tokenizer.decode(
                list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )

    def token_id(self, token: str) -> int | None:
        added = self._tokenizer.get_added_vocab().get(token)
        return added if isinstance(added, int) and self.encode(token) == [added] else None

    @property
    def chat_template(self) -> str:
        template = self._tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError("the tokenizer has no chat template")
        return template


def load_tokenizer(family: ModelFamily) -> ChatTokenizer:
    """Load ``family``'s tokenizer at its pinned revision (from the Hugging Face cache or Hub)."""
    transformers = _transformers()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            family.tokenizer_repo,
            revision=family.revision,
            trust_remote_code=False,
            # A generic config skips model-only checks (e.g. RoPE) a tokenizer does not need.
            config=transformers.PreTrainedConfig(),
        )
    except Exception as exc:
        raise ExternalError(
            f"could not load the {family.tokenizer_repo} tokenizer at revision "
            f"{family.revision} ({type(exc).__name__})",
            source="huggingface",
            hint="check the connection to huggingface.co, or the Hugging Face cache when offline",
        ) from exc
    adapter = HFChatTokenizer(tokenizer)
    try:
        adapter.chat_template  # noqa: B018
    except ValueError as exc:
        raise ExternalError(
            f"the {family.tokenizer_repo} tokenizer has no chat template", source="huggingface"
        ) from exc
    return adapter


def rendering_identity(family: ModelFamily, tokenizer: ChatTokenizer) -> RenderingIdentity:
    """What fixes ``family``'s rendered tokens: implementation, tokenizer revision and template."""
    try:
        transformers_version = metadata.version("transformers")
    except metadata.PackageNotFoundError as exc:
        raise UsageError("Hugging Face transformers is not installed", hint=_INSTALL_HINT) from exc
    return RenderingIdentity(
        version=f"{RENDERING_VERSION};transformers={transformers_version}",
        tokenizer_repo=family.tokenizer_repo,
        revision=family.revision,
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        template_kwargs=family.template_kwargs,
    )


def _transformers() -> Any:
    # Tokenizers need no PyTorch; hide transformers' import-time advice that models are unavailable.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise UsageError("Hugging Face transformers is not installed", hint=_INSTALL_HINT) from exc
