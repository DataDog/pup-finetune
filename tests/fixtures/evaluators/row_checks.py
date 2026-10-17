# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Synthetic `dataset triage --checks` row checks for the tests and the offline dev loop."""

from __future__ import annotations

import json
from typing import Any


def declares_tools(row: dict[str, Any]) -> bool | tuple[bool, str]:
    if not row.get("tools"):
        return False, "the row declares no tools"
    return True


def no_tool_errors(row: dict[str, Any]) -> bool | tuple[bool, str] | None:
    results = [m.get("content") or "" for m in row["messages"] if m["role"] == "tool"]
    if not results:
        return None
    for content in results:
        try:
            data = json.loads(content)
        except ValueError:
            continue
        if isinstance(data, dict) and "error" in data:
            return False, "a tool returned an error"
    return True
