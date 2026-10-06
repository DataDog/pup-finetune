# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import copy
import dataclasses
from types import MappingProxyType
from typing import Any

import pytest

from pup_finetune import rendering
from pup_finetune.rendering import (
    FAMILIES,
    ModelFamily,
    RenderError,
    RenderErrorCode,
    RenderingIdentity,
    TokenDatum,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


def family(**overrides: Any) -> ModelFamily:
    fields: dict[str, Any] = {
        "key": "synthetic",
        "tokenizer_repo": "example/synthetic-model",
        "revision": SHA,
        "max_seq_len": 4096,
        "supports_reasoning": False,
        "template_kwargs": MappingProxyType({}),
        "boundary": "none",
        "stop_sequences": ("<|end|>",),
    }
    return ModelFamily(**(fields | overrides))


def test_render_error_codes_are_stable() -> None:
    assert [code.value for code in RenderErrorCode] == [
        "no_trainable_turn",
        "template_failed",
        "prefix_mismatch",
        "no_target_tokens",
        "exceeds_context",
        "reasoning_unsupported",
        "invalid_tokens",
    ]


def test_render_error_carries_its_code_detail_and_sizes() -> None:
    error = RenderError(RenderErrorCode.EXCEEDS_CONTEXT, "row is too long", tokens=9000, limit=4096)
    assert isinstance(error, ValueError)
    assert (error.code, error.detail, error.tokens, error.limit) == (
        RenderErrorCode.EXCEEDS_CONTEXT,
        "row is too long",
        9000,
        4096,
    )
    assert str(error) == "exceeds_context: row is too long"


def test_render_error_accepts_a_code_value_and_defaults_sizes_to_none() -> None:
    error = RenderError("no_trainable_turn", "nothing to train")  # type: ignore[arg-type]
    assert error.code is RenderErrorCode.NO_TRAINABLE_TURN
    assert error.tokens is None
    assert error.limit is None


def test_render_error_copies_with_all_fields() -> None:
    error = RenderError(RenderErrorCode.EXCEEDS_CONTEXT, "too long", tokens=5, limit=4)
    duplicate = copy.deepcopy(error)
    assert (duplicate.code, duplicate.detail, duplicate.tokens, duplicate.limit) == (
        RenderErrorCode.EXCEEDS_CONTEXT,
        "too long",
        5,
        4,
    )
    assert str(duplicate) == str(error)


def test_token_datum_is_frozen() -> None:
    datum = TokenDatum(tokens=(1, 2, 3), weights=(0, 1, 1), message_index=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        datum.tokens = (4,)  # type: ignore[misc]


@pytest.mark.parametrize("revision", ["main", SHA[:39], SHA.upper(), SHA + "0"])
def test_model_family_requires_a_full_commit_sha(revision: str) -> None:
    with pytest.raises(ValueError, match="40-hex"):
        family(revision=revision)


def test_model_family_requires_a_positive_context() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        family(max_seq_len=0)


def test_rendering_identity_as_dict_is_plain_and_copied() -> None:
    kwargs = MappingProxyType({"enable_thinking": True})
    identity = RenderingIdentity(
        version="pup-finetune-prefix-v1;transformers=0.0.0",
        tokenizer_repo="example/synthetic-model",
        revision=SHA,
        template_sha256="ab" * 32,
        template_kwargs=kwargs,
    )
    data = identity.as_dict()
    assert data == {
        "version": "pup-finetune-prefix-v1;transformers=0.0.0",
        "tokenizer_repo": "example/synthetic-model",
        "revision": SHA,
        "template_sha256": "ab" * 32,
        "template_kwargs": {"enable_thinking": True},
    }
    assert type(data["template_kwargs"]) is dict
    data["template_kwargs"]["enable_thinking"] = False
    assert kwargs["enable_thinking"] is True


def test_families_are_keyed_by_their_key() -> None:
    assert sorted(FAMILIES) == ["glm5_3", "qwen3_5"]
    for key, entry in FAMILIES.items():
        assert entry.key == key
        assert entry.max_seq_len == 131_072
        assert entry.supports_reasoning
        assert dict(entry.template_kwargs) == {}


def test_qwen3_5_family_is_pinned_and_stops_on_im_end() -> None:
    qwen = FAMILIES["qwen3_5"]
    assert qwen.tokenizer_repo == "Qwen/Qwen3.5-9B"
    assert qwen.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert qwen.boundary == "none"
    assert qwen.stop_sequences == ("<|im_end|>",)


def test_glm5_3_family_is_pinned_and_stops_on_the_next_role_tag() -> None:
    glm = FAMILIES["glm5_3"]
    assert glm.tokenizer_repo == "zai-org/GLM-5.3-Flash"
    assert glm.revision == "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
    assert glm.boundary == "glm_stop_tag"
    assert glm.stop_sequences == ("<|user|>", "<|observation|>")


def test_family_template_kwargs_are_read_only() -> None:
    with pytest.raises(TypeError):
        FAMILIES["qwen3_5"].template_kwargs["enable_thinking"] = False  # type: ignore[index]


def test_package_reexports_the_contract() -> None:
    contract = {
        "FAMILIES",
        "ChatTokenizer",
        "ModelFamily",
        "RenderError",
        "RenderErrorCode",
        "RenderingIdentity",
        "TokenDatum",
    }
    assert contract <= set(rendering.__all__)
    assert all(hasattr(rendering, name) for name in contract)
