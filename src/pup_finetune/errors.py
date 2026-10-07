# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Expected failures and the exit codes they map to.

Exit codes: ``0`` success, ``1`` external or runtime failure, ``2`` usage, validation or a
missing ``--confirm``. Messages must never contain secret values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Final

from pup_finetune.runtime import Runtime


class FinetuneError(Exception):
    """Base class for failures reported to the user without a traceback."""

    exit_code: ClassVar[int] = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UsageError(FinetuneError):
    """Invalid arguments or input that the user can fix."""

    exit_code: ClassVar[int] = 2


class ConfirmRequired(FinetuneError):
    """A paid or remote-changing action previewed instead of running.

    ``preview`` describes what would happen; ``rerun`` is the exact command that runs it, or
    ``None`` when its ``--confirm`` would be refused (e.g. under ``PUP_READ_ONLY``).
    """

    exit_code: ClassVar[int] = 2

    def __init__(
        self,
        message: str,
        *,
        preview: Mapping[str, Any],
        rerun: str | None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.preview: dict[str, Any] = dict(preview)
        self.rerun = rerun


class ReadOnlyError(FinetuneError):
    """A remote write was attempted while ``PUP_READ_ONLY`` is set."""

    exit_code: ClassVar[int] = 2


class ExternalError(FinetuneError):
    """A dependency outside this process failed, e.g. ``pup`` or a provider API."""

    exit_code: ClassVar[int] = 1

    def __init__(self, message: str, *, source: str, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.source = source

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


class StateError(FinetuneError):
    """On-disk state is corrupted, locked, or does not match what this command expects."""

    exit_code: ClassVar[int] = 1


class LockBusy(StateError):
    """Another process holds the lock this one tried to take."""


READ_ONLY_HINT: Final = (
    "writes are refused while PUP_READ_ONLY is set (by `pup --read-only` or the "
    "environment); changing that is the user's call"
)
"""What to say when read-only mode refuses something: never to switch it off."""

READ_ONLY_PREVIEW: Final = "--confirm is refused while PUP_READ_ONLY is set"
"""The ``read_only`` entry of a preview whose ``--confirm`` read-only mode refuses."""


def install_hint(extra: str) -> str:
    """How to install the optional ``extra`` (e.g. ``anthropic``) without losing the others.

    ``uv sync`` is exact: a bare ``uv sync --extra X`` uninstalls every other extra.
    """
    return (
        f"install the {extra} extra: `uv sync --all-extras` in a checkout (a bare "
        f"`uv sync --extra {extra}` removes the other extras), or reinstall the tool with the "
        f"extra: `pip install 'pup-finetune[{extra}]'`"
    )


def reinstall_hint(extra: str) -> str:
    """How to reinstall the optional ``extra`` whose package is installed but broken."""
    return (
        f"reinstall the {extra} extra: `uv sync --all-extras --reinstall` in a checkout, or "
        f"`pip install --force-reinstall 'pup-finetune[{extra}]'`"
    )


def require_writable(runtime: Runtime, action: str) -> None:
    """Raise :class:`ReadOnlyError` when ``PUP_READ_ONLY`` forbids ``action`` (a remote write)."""
    if runtime.read_only:
        raise ReadOnlyError(
            f"{action} is a remote write and PUP_READ_ONLY is set", hint=READ_ONLY_HINT
        )


def require_confirm(
    confirm: bool,
    preview: Mapping[str, Any],
    rerun: str,
    *,
    read_only: bool = False,
    refused: str | None = None,
    hint: str | None = None,
) -> None:
    """Raise :class:`ConfirmRequired` with ``preview`` unless ``confirm`` is true.

    ``confirm`` must come from an explicit ``--confirm`` flag. ``PUP_AUTO_APPROVE`` and agent
    mode never satisfy it: paid and remote-changing actions always need an explicit
    ``--confirm``, so callers must not derive ``confirm`` from :class:`Runtime`.

    A preview never offers a ``--confirm`` that would be refused: with ``read_only`` (the
    action is a remote write and ``PUP_READ_ONLY`` is set) the preview says so under
    ``read_only`` and ``rerun`` is not offered; ``refused`` (why ``--confirm`` would be
    refused for another reason, with ``hint`` saying what to change) does the same.
    """
    if confirm:
        return
    if read_only:
        raise ConfirmRequired(
            f"preview only, nothing was run; {READ_ONLY_PREVIEW}",
            preview={**preview, "read_only": READ_ONLY_PREVIEW},
            rerun=None,
            hint=READ_ONLY_HINT,
        )
    if refused is not None:
        raise ConfirmRequired(
            f"preview only, nothing was run; {refused}", preview=preview, rerun=None, hint=hint
        )
    raise ConfirmRequired(
        "preview only, nothing was run; rerun with --confirm to proceed",
        preview=preview,
        rerun=rerun,
        hint=hint,
    )
