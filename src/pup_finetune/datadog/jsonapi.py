# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""JSON:API helpers shared by the Agent Observability clients that call :class:`Pup`.

:data:`JSON_API` is the media type their write routes take, :func:`private_json_file` writes a
body passed to pup with ``--file`` (``0600``, in a private temporary directory removed
afterwards, since bodies hold trace content), and :func:`object_id` reads ``data.id`` from a
response.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

__all__ = ["JSON_API", "object_id", "private_json_file"]

JSON_API: Final = "application/vnd.api+json"


@contextlib.contextmanager
def private_json_file(data: Any) -> Iterator[Path]:
    """``data`` as a ``0600`` JSON file in a fresh private directory, removed afterwards."""
    with tempfile.TemporaryDirectory(prefix="pup-finetune-") as directory:
        path = Path(directory) / "body.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)
        yield path


def object_id(data: Any) -> str | None:
    """``data.id`` of a JSON:API response, or ``None`` when it has none."""
    if isinstance(data, dict) and isinstance(item := data.get("data"), dict):
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return None
