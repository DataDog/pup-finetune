# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten: LoRA SFT on Baseten Loops, and serving of the best checkpoint.

Importing this package is cheap: ``PROVIDER`` loads the provider module on first access.
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pup_finetune.providers import Provider

__all__ = ["PROVIDER"]


@cache
def _provider() -> Provider:
    provider: Provider = importlib.import_module(f"{__name__}.provider").BasetenProvider()
    return provider


def __getattr__(name: str) -> Provider:
    if name == "PROVIDER":
        return _provider()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
