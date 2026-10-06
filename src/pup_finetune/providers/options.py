# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Provider options (``--provider-option key=value``), validated against a provider's model.

Every provider declares its options as a frozen pydantic model with ``extra="forbid"``, and
parses them with :func:`parse_options`: unknown keys are refused by name, string values are
decoded as JSON, and every failure is a :class:`UsageError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from pup_finetune.errors import UsageError

__all__ = ["parse_options"]


def parse_options[M: BaseModel](
    model: type[M], options: Mapping[str, Any], *, provider: str, example: str
) -> M:
    """Validate ``options`` as ``model``, decoding string values as JSON.

    ``provider`` names the provider in errors, and ``example`` (e.g. ``max_seq_len=32768``)
    is the hint for a value that is not JSON. Raises :class:`UsageError`.
    """
    unknown = sorted(set(options) - set(model.model_fields))
    if unknown:
        raise UsageError(
            f"unknown {provider} provider option(s): {', '.join(unknown)}",
            hint=f"known options: {', '.join(model.model_fields)}",
        )
    decoded = {
        key: _decode(key, value, provider=provider, example=example)
        for key, value in options.items()
    }
    try:
        return model.model_validate(decoded)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        raise UsageError(f"invalid {provider} provider options: {'; '.join(problems)}") from exc


def _decode(key: str, value: Any, *, provider: str, example: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError is a ValueError
        raise UsageError(
            f"{provider} provider option {key!r} must be JSON, got {value!r}",
            hint=f"for example {example}",
        ) from exc
