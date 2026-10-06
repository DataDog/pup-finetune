# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten tests render with the fake tokenizer, so they run without the hf extra."""

from __future__ import annotations

from importlib import metadata

import pytest


@pytest.fixture(autouse=True)
def _transformers_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rendering identity records the transformers version, installed or not."""
    version = metadata.version
    monkeypatch.setattr(
        metadata, "version", lambda name: "0.0.0" if name == "transformers" else version(name)
    )
