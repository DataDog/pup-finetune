# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Shared pytest configuration: opt-in gates for live and tokenizer tests."""

from __future__ import annotations

import os

import pytest

pytest_plugins = ["pytester"]

_GATES: dict[str, str] = {
    "live": "PUP_FINETUNE_LIVE",
    "tokenizer": "PUP_FINETUNE_TOKENIZER_TESTS",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, env_var in _GATES.items():
        if os.environ.get(env_var) == "1":
            continue
        skip = pytest.mark.skip(reason=f"set {env_var}=1 to run {marker} tests")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
