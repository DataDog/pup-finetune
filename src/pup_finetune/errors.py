# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Expected failures, the exit codes they map to and the stable codes the error line shows.

The rule: exit ``2`` when the user can fix it by changing the command or the directory
(usage, validation, a missing ``--confirm``, read-only, a wrong, missing or non-empty
directory, a frozen-settings conflict, the wrong phase); exit ``1`` when something outside
failed or local state is broken (a provider or pup failure, corruption, a lock held by a
live process, an interrupt). ``0`` is success.

Every class has an :attr:`~FinetuneError.error_code`, printed in the error line
(``Error [run_dir_not_empty]: ...``) and listed in ``docs/commands.md``
(:data:`ERROR_CODES`). The codes are a public contract: keep the set small and stable.
Messages must never contain secret values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Final

from pup_finetune.runtime import Runtime


class FinetuneError(Exception):
    """Base class for failures reported to the user without a traceback."""

    exit_code: ClassVar[int] = 1
    error_code: ClassVar[str] = "unexpected"
    """The stable code the error line shows (see :data:`ERROR_CODES`). Not ``code``, which
    some subclasses (e.g. ``RendererError``) use for their own reason codes."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UsageError(FinetuneError):
    """Invalid arguments or input that the user can fix."""

    exit_code: ClassVar[int] = 2
    error_code: ClassVar[str] = "usage"


class ConfirmRequired(FinetuneError):
    """A paid or remote-changing action previewed instead of running.

    ``preview`` describes what would happen; ``rerun`` is the exact command that runs it, or
    ``None`` when its ``--confirm`` would be refused (e.g. under ``PUP_READ_ONLY``), and then
    ``refused`` says why.
    """

    exit_code: ClassVar[int] = 2
    error_code: ClassVar[str] = "confirmation_required"

    def __init__(
        self,
        message: str,
        *,
        preview: Mapping[str, Any],
        rerun: str | None,
        hint: str | None = None,
        refused: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.preview: dict[str, Any] = dict(preview)
        self.rerun = rerun
        self.refused = refused


class ReadOnlyError(FinetuneError):
    """A remote write was attempted while ``PUP_READ_ONLY`` is set."""

    exit_code: ClassVar[int] = 2
    error_code: ClassVar[str] = "read_only"


class ExternalError(FinetuneError):
    """A dependency outside this process failed, e.g. ``pup`` or a provider API."""

    exit_code: ClassVar[int] = 1
    error_code: ClassVar[str] = "external"

    def __init__(self, message: str, *, source: str, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.source = source

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


class StateError(FinetuneError):
    """On-disk state is corrupted, locked, or does not match what this command expects.

    Exit ``1``: local state is broken. Where the user fixes it by changing the command or
    the directory, raise a :class:`DirectoryUsageError` subclass instead (exit ``2``).
    """

    exit_code: ClassVar[int] = 1
    error_code: ClassVar[str] = "state"


class LockBusy(StateError):
    """Another process holds the lock this one tried to take."""

    error_code: ClassVar[str] = "lock_busy"


class DirectoryUsageError(UsageError, StateError):
    """A directory the command cannot act on as given, which the user fixes by changing the
    command or the directory: exit ``2``.

    Both a :class:`UsageError` and a :class:`StateError`, so code that caught either where
    one of these is raised now still catches it.
    """

    exit_code: ClassVar[int] = 2
    error_code: ClassVar[str] = "usage"


class NotADatasetDir(DirectoryUsageError):
    """The directory is missing or is not a dataset directory (e.g. a training run's)."""

    error_code: ClassVar[str] = "not_a_dataset_dir"


class NotARunDir(DirectoryUsageError):
    """The directory is missing or is not a training run's (e.g. a dataset directory)."""

    error_code: ClassVar[str] = "not_a_run_dir"


class RunDirNotEmpty(DirectoryUsageError):
    """A new run was asked for in a directory that already holds something."""

    error_code: ClassVar[str] = "run_dir_not_empty"


class SettingsConflict(DirectoryUsageError):
    """The directory records other settings than the command asks for (frozen settings, an
    evaluation's identity, a dataset changed under a run): pass the same ones, or use a new
    directory."""

    error_code: ClassVar[str] = "settings_conflict"


class WrongPhase(DirectoryUsageError):
    """The run (or its deployment) is not in a phase the command acts on, e.g. evaluating
    or deploying a run that is still training, or resuming a completed one."""

    error_code: ClassVar[str] = "wrong_phase"


class Interrupted(FinetuneError):
    """Ctrl-C or SIGTERM stopped the command; what finished is kept, and rerunning (or
    resuming) continues it."""

    error_code: ClassVar[str] = "interrupted"


ERROR_CODES: Final[dict[str, tuple[int, str]]] = {
    "usage": (2, "a flag, argument or input is wrong; the message says which"),
    "confirmation_required": (
        2,
        "a preview: nothing was run; stdout holds it, with the `--confirm` command",
    ),
    "read_only": (
        2,
        "a paid run or a remote write while PUP_READ_ONLY is set; changing that is the user's call",
    ),
    "not_a_dataset_dir": (2, "the directory is missing or is not a dataset directory"),
    "not_a_run_dir": (2, "no such run directory or run id (`pup finetune runs` lists them)"),
    "run_dir_not_empty": (
        2,
        "a new run needs a new directory: omit --run-dir/--run-id, or pass new ones",
    ),
    "settings_conflict": (
        2,
        "the directory records other settings (leave the flags out: the frozen values are "
        "used; or use a new directory or `--out-dir`)",
    ),
    "wrong_phase": (2, "the run is not in a phase the command acts on (`status` says which)"),
    "lock_busy": (1, "another pup-finetune process is using the directory; wait for it"),
    "provider_auth": (1, "the provider rejected the API key; log in again"),
    "provider_capacity": (1, "the provider has no capacity or quota right now; retry later"),
    "transient": (1, "a timeout or a dropped connection; retrying is safe"),
    "external": (1, "pup, Datadog or a provider failed"),
    "state": (1, "a local file is unreadable or inconsistent"),
    "interrupted": (1, "Ctrl-C or SIGTERM stopped the command; rerun or resume it to continue"),
    "unexpected": (1, "a bug; PUP_FINETUNE_DEBUG=1 prints the traceback"),
}
"""Every code the error line can show: its exit code and what it means. ``docs/commands.md``
lists them, and a test checks every :class:`FinetuneError` subclass against them."""


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


def require_writable(
    runtime: Runtime, action: str, *, what: str = "a remote write", offline: bool = False
) -> None:
    """Raise :class:`ReadOnlyError` when ``PUP_READ_ONLY`` forbids ``action``, which is
    ``what`` (a remote write; :data:`PAID_RUN` for training and evaluating), unless it runs
    ``offline``: nothing it does leaves the machine."""
    if runtime.read_only and not offline:
        raise ReadOnlyError(f"{action} is {what} and PUP_READ_ONLY is set", hint=READ_ONLY_HINT)


PAID_RUN: Final = "a paid or remote run"
"""What ``train``, ``resume`` and ``evaluate`` with ``--confirm`` are to ``PUP_READ_ONLY``,
which refuses them, unless everything runs offline (the fake provider and judge)."""


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
            refused=READ_ONLY_PREVIEW,
        )
    if refused is not None:
        raise ConfirmRequired(
            f"preview only, nothing was run; {refused}",
            preview=preview,
            rerun=None,
            hint=hint,
            refused=refused,
        )
    raise ConfirmRequired(
        "preview only, nothing was run; rerun with --confirm to proceed",
        preview=preview,
        rerun=rerun,
        hint=hint,
    )


def require_approval(
    runtime: Runtime,
    action: str,
    *,
    confirm: bool,
    offline: bool,
    preview: Mapping[str, Any],
    rerun: str,
    what: str = "a remote write",
) -> None:
    """The gate of every paid or remote-changing ``action``, which is ``what`` to
    ``PUP_READ_ONLY``: return when ``confirm`` approves it and read-only mode allows it.

    ``PUP_READ_ONLY`` refuses ``--confirm`` (:func:`require_writable`) and a preview offers
    none, unless the action runs ``offline``: nothing it does leaves the machine (the offline
    fake provider and judge). Without ``confirm``, :func:`require_confirm` previews it, and
    the preview says under ``offline`` whether it runs offline.
    """
    if confirm:
        require_writable(runtime, action, what=what, offline=offline)
    require_confirm(
        confirm,
        {**preview, "offline": offline},
        rerun,
        read_only=runtime.read_only and not offline,
    )
