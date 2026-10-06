# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import pickle

import pytest

from pup_finetune.errors import FinetuneError
from pup_finetune.rendering import ParsedResponse, RendererError, RendererErrorCode


def test_renderer_error_carries_a_stable_code_and_detail() -> None:
    error = RendererError("not_prefix_stable", "turn 2: prompt is not a prefix")
    assert error.code is RendererErrorCode.NOT_PREFIX_STABLE
    assert error.detail == "turn 2: prompt is not a prefix"
    assert error.message == "not_prefix_stable: turn 2: prompt is not a prefix"
    assert isinstance(error, FinetuneError)
    assert error.exit_code == 1


def test_renderer_error_rejects_unknown_codes() -> None:
    with pytest.raises(ValueError, match="not_a_code"):
        RendererError("not_a_code", "detail")


def test_renderer_error_survives_pickling() -> None:
    error = RendererError(RendererErrorCode.TEMPLATE_ERROR, "messages[3]: bad role")
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301 - our own object
    assert restored.code is RendererErrorCode.TEMPLATE_ERROR
    assert restored.detail == "messages[3]: bad role"
    assert str(restored) == str(error)


def test_parsed_response_holds_a_message_and_its_validity() -> None:
    parsed = ParsedResponse(message={"role": "assistant", "content": "hi"}, format_valid=True)
    assert parsed.message["content"] == "hi"
    assert parsed.format_valid
