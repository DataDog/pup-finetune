# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Synthetic evaluator checks for the tests and the offline dev loop."""

from __future__ import annotations

import re
from typing import Any

_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def no_card_numbers(case: dict[str, Any], candidate: dict[str, Any]) -> bool | tuple[bool, str]:
    if _CARD.search(str(candidate.get("content") or "")):
        return False, "the answer shows a card number"
    return True


def answers_in_text(case: dict[str, Any], candidate: dict[str, Any]) -> bool | None:
    if case["reference"].get("tool_calls"):
        return None
    return not candidate.get("tool_calls")
