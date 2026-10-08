# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What a preview pins, so that ``--confirm`` runs what the user approved.

A preview's ``inputs_digest`` is a short digest of what the action would use (the rendered
rows, the settings, the records to send), and its ``next_command`` passes it back as
``--expect``. With ``--confirm``, the command computes the digest again from what it is
about to run and refuses on a mismatch (:func:`require_expected`), so a dataset relabeled
between the preview and the confirm is never trained on, judged or pushed unseen.

The digest is :data:`DIGEST_LEN` hex characters, one equal segment per :class:`Part`, so a
mismatch can name the part that changed. Each part's value is hashed as canonical JSON, so
the digest is the same in every process for the same inputs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pup_finetune.errors import SettingsConflict, UsageError
from pup_finetune.storage import canonical_json

__all__ = [
    "DIGEST_LEN",
    "Part",
    "inputs_digest",
    "require_expected",
    "with_digest",
    "with_settings",
]

DIGEST_LEN: Final = 24
"""The hex characters of a digest."""

_HEX: Final = re.compile(rf"[0-9a-f]{{{DIGEST_LEN}}}")


@dataclass(frozen=True, slots=True)
class Part:
    """One part of what an action uses.

    ``name`` says what it is in a sentence (``the dataset``); ``value`` is what is hashed (any
    JSON value); ``now`` describes it as it is now, for the refusal (``7 train / 4 validation
    rows``), when that is cheap to say.
    """

    name: str
    value: Any
    now: str | None = None


def _segment_len(parts: Sequence[Part]) -> int:
    if not parts or DIGEST_LEN % len(parts):
        raise ValueError(f"a digest has 1 to {DIGEST_LEN} parts that divide {DIGEST_LEN}")
    return DIGEST_LEN // len(parts)


def _segments(parts: Sequence[Part]) -> list[str]:
    size = _segment_len(parts)
    return [
        hashlib.sha256(canonical_json(part.value).encode("utf-8")).hexdigest()[:size]
        for part in parts
    ]


def inputs_digest(parts: Sequence[Part]) -> str:
    """The digest of ``parts``, in order: what a preview shows as ``inputs_digest``."""
    return "".join(_segments(parts))


def with_digest(preview: Mapping[str, Any], digest: str) -> dict[str, Any]:
    """``preview`` with ``inputs_digest`` set to ``digest``, just before its ``warnings``."""
    shown = {key: value for key, value in preview.items() if key != "warnings"}
    shown["inputs_digest"] = digest
    if "warnings" in preview:
        shown["warnings"] = preview["warnings"]
    return shown


def with_settings(
    preview: Mapping[str, Any], settings: Mapping[str, Any], defaults_used: Sequence[str]
) -> dict[str, Any]:
    """``preview`` with ``settings`` (the values that will be used) and ``defaults_used``
    (the ones no flag set) just before its ``data_destination`` (else at the end)."""
    fields = {"settings": dict(settings), "defaults_used": list(defaults_used)}
    shown: dict[str, Any] = {}
    for key, value in preview.items():
        if key == "data_destination":
            shown.update(fields)
        shown[key] = value
    return {**shown, **fields} if "settings" not in shown else shown


def _and(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def require_expected(expected: str | None, parts: Sequence[Part], *, preview: str) -> None:
    """Raise :class:`SettingsConflict` unless ``expected`` (``--expect``) is the digest of
    ``parts``, or :class:`UsageError` when it is not a digest at all; nothing to check when
    it is ``None``.

    The error names the parts whose segment differs, with what they are now, and ``preview``
    (the command without ``--expect`` and ``--confirm``) is the hint.
    """
    if expected is None:
        return
    given = expected.strip().lower()
    segments = _segments(parts)
    if given == "".join(segments):
        return
    hint = f"preview again with `{preview}` and show the user what changed"
    if not _HEX.fullmatch(given):
        # A malformed value is a usage error; only a well-formed digest can show drift.
        raise UsageError(
            f"--expect {expected!r} is not an inputs_digest this command printed "
            f"({DIGEST_LEN} hex characters)",
            hint=f"preview with `{preview}` and run its next_command as written",
        )
    size = _segment_len(parts)
    changed = [
        part
        for index, (part, segment) in enumerate(zip(parts, segments, strict=True))
        if given[index * size : (index + 1) * size] != segment
    ]
    nows = [
        f"{part.name}: {part.now}" if len(changed) > 1 else part.now for part in changed if part.now
    ]
    now = "; ".join(nows)
    detail = f" (now {now})" if now else ""
    raise SettingsConflict(
        f"{_and([part.name for part in changed])} changed since the preview{detail}; preview again",
        hint=hint,
    )
