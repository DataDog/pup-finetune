# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""How values read in what a command prints: paths, floats and lists.

Only stdout results and hints use these; the files a command writes keep their values as
they are (absolute paths in ``run.json``, unrounded losses in ``epochs.json``).

:func:`display_path` is the one path formatter of results, next commands and hints: relative
to the current directory when the path is inside it, else absolute. :func:`significant`
rounds a float to :data:`SIGNIFICANT_DIGITS` significant digits, and :func:`rounded_floats`
does so for every measurement in a result (:data:`MEASUREMENTS`: losses, rates, deltas,
elapsed times, spend estimates, ages), never for a value the user set (a flag, a
``settings`` block), which reads exactly as it was typed.
"""

from __future__ import annotations

import math
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ABSOLUTE_PATHS_ENV",
    "MEASUREMENTS",
    "SIGNIFICANT_DIGITS",
    "UNROUNDED",
    "counted",
    "display_path",
    "joined",
    "loss_text",
    "quoted",
    "rounded_floats",
    "significant",
    "word",
]

SIGNIFICANT_DIGITS: Final = 4
"""How many significant digits a measurement in a printed result keeps."""

MEASUREMENTS: Final = frozenset(
    {
        # losses
        "loss",
        "losses",
        "train_loss",
        "val_loss",
        "val_losses",
        "best_val_loss",
        "last_train_loss",
        # rates and their comparison (a metric's or the judge's ``base``, ``tuned`` and
        # ``compare``)
        "rate",
        "pass_rate",
        "headline_pass_rate",
        "reference_pass_rate",
        "kappa",
        "base",
        "tuned",
        "delta",
        "ci95",
        "tuned_minus_base",
        "compare",
        "tuned_minus_compare",
        # times, ages and spend
        "elapsed",
        "elapsed_s",
        "elapsed_seconds",
        "eta_seconds",
        "approx_eta_seconds",
        "heartbeat_age_s",
        "latency_s",
        "estimated_spend_usd",
        "spend_usd",
    }
)
"""The keys whose floats are measurements, which :func:`rounded_floats` rounds (a list's
items count as its key's). A float under any other key is printed exactly."""

UNROUNDED: Final = frozenset({"settings", "provider_options", "judge", "request"})
"""Keys whose whole value is what the user set or what is sent, never rounded, even where a
key inside it is in :data:`MEASUREMENTS`."""


ABSOLUTE_PATHS_ENV: Final = "PUP_FINETUNE_ABSOLUTE_PATHS"
"""Set to ``1`` (by :func:`~pup_finetune.training.worker.spawn_worker`) to show every path
absolute: a detached worker runs in the run directory, so a relative path in its log
(``resume .``) would not name the run for the person who reads it elsewhere."""


def display_path(path: str | os.PathLike[str]) -> str:
    """``path`` as results and hints show it: relative to the current directory when it is
    inside it (``.`` for the directory itself), else absolute; always absolute with
    :data:`ABSOLUTE_PATHS_ENV` set to ``1``.

    Paths are compared as written, without resolving symbolic links, then resolved, so a
    path through a link to the current directory (``/tmp`` for ``/private/tmp``) still reads
    as relative. An unreadable current directory leaves the path absolute.
    """
    text = os.fspath(path)
    if not text:
        return text
    absolute = Path(os.path.abspath(text))
    if os.environ.get(ABSOLUTE_PATHS_ENV) == "1":
        return str(absolute)
    try:
        cwd = Path(os.getcwd())
    except OSError:
        return str(absolute)
    for candidate, base in ((absolute, cwd), (_resolved(absolute), _resolved(cwd))):
        if candidate is None or base is None:
            continue
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            continue
        return str(relative) if relative.parts else "."
    return str(absolute)


def word(value: object) -> str:
    """``value`` as one word of a command or hint: a path (``os.PathLike``) through
    :func:`display_path`, anything else as its ``str``."""
    return display_path(value) if isinstance(value, os.PathLike) else str(value)


def quoted(value: object) -> str:
    """:func:`word`, shell-quoted: a path or value to paste into a hint's command."""
    return shlex.quote(word(value))


def joined(items: Sequence[str]) -> str:
    """``items`` as a sentence lists them: ``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def counted(count: int, noun: str) -> str:
    """``count`` and ``noun``, plural unless it is one: ``1 kept row``, ``2 kept rows``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _resolved(path: Path) -> Path | None:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


def significant(value: float, digits: int = SIGNIFICANT_DIGITS) -> float:
    """``value`` rounded to ``digits`` significant digits: ``0.01494908`` reads ``0.01495``,
    ``1234.5678`` reads ``1235.0``. NaN and infinities are returned as they are, so a
    serializer that refuses them still does."""
    if not math.isfinite(value) or value == 0:
        return value
    return float(f"{value:.{digits}g}")


def loss_text(value: object) -> str | None:
    """A loss as every stderr line shows it (progress, the watch line, the summaries):
    :data:`SIGNIFICANT_DIGITS` significant digits, trailing zeros kept (``2.000``,
    ``0.01235``); ``None`` for a value that is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return f"{value:#.{SIGNIFICANT_DIGITS}g}"


def rounded_floats(value: Any, key: object = None) -> Any:
    """``value`` with every measurement inside it (dicts, lists and tuples) passed through
    :func:`significant`: a float whose key, or whose list's key, is in :data:`MEASUREMENTS`,
    outside any :data:`UNROUNDED` key. Tuples become lists; everything else is left as it
    is, so a learning rate or spend cap reads exactly as the user typed it."""
    if isinstance(value, float):
        return significant(value) if key in MEASUREMENTS else value
    if isinstance(value, Mapping):
        return {
            name: item if name in UNROUNDED else rounded_floats(item, name)
            for name, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [rounded_floats(item, key) for item in value]
    return value
