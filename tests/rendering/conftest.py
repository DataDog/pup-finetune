# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The real Qwen3.8 tokenizer for ``tokenizer``-marked tests, downloaded once per session."""

from __future__ import annotations

from typing import Any

import pytest

from pup_finetune.rendering import hf
from pup_finetune.rendering.families import QWEN3_8
from pup_finetune.rendering.hf import TransformersTokenizer


@pytest.fixture(scope="session")
def qwen3_8_tokenizer() -> TransformersTokenizer:
    """The ``qwen3_8`` family's tokenizer at its pinned revision, loaded as renderers load it."""
    pytest.importorskip("transformers")
    return hf.load_tokenizer(QWEN3_8.tokenizer_model, QWEN3_8.revision)


@pytest.fixture(scope="session")
def qwen3_8_hf_tokenizer(qwen3_8_tokenizer: TransformersTokenizer) -> Any:
    """The ``transformers`` tokenizer behind :data:`qwen3_8_tokenizer`."""
    return qwen3_8_tokenizer._tokenizer
