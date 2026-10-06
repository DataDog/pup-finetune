# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""pup-finetune: fine-tune open-weight models on Datadog Agent Observability traces."""

from __future__ import annotations


def _resolve_version() -> str:
    try:
        from pup_finetune._version import __version__ as generated

        return generated
    except ImportError:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pup-finetune")
        except PackageNotFoundError:
            return "0.0.0"


__version__: str = _resolve_version()

__all__ = ["__version__"]
