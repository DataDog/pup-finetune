# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Text for synthetic rows."""

from __future__ import annotations


def letters(n: int) -> str:
    """``n`` in letters, so texts told apart by it are not copies (numbers fold to ``0``)."""
    return "".join(chr(ord("a") + int(digit)) for digit in str(n))
