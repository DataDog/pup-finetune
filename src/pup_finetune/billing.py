# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What a preview says it bills: ``billing``, every part's sentence, and ``paid_billing``,
the sentences of the parts that leave the machine.

A billing sentence is always a :class:`BillingPart`, which carries whether it runs offline,
so no caller pairs a sentence with a flag.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["BillingPart", "billing_fields", "merge_billing"]


@dataclass(frozen=True, slots=True)
class BillingPart:
    """One part of a preview's billing: who bills what, and whether it runs offline (the
    fake provider or judge), so it bills nothing."""

    sentence: str
    offline: bool


def billing_fields(parts: Iterable[BillingPart]) -> dict[str, Any]:
    """A preview's ``billing``, every part's sentence ``; ``-separated, and
    ``paid_billing``, the sentences of the parts that are not offline."""
    parts = list(parts)
    return {
        "billing": "; ".join(part.sentence for part in parts),
        "paid_billing": [part.sentence for part in parts if not part.offline],
    }


def merge_billing(preview: dict[str, Any], sub: Mapping[str, Any]) -> None:
    """Add ``sub``'s billing fields (a sub-preview's, or :func:`billing_fields`') after
    ``preview``'s."""
    preview["billing"] += f"; {sub['billing']}"
    preview["paid_billing"] += sub["paid_billing"]
