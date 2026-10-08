# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import contextlib
import importlib
import pkgutil
from collections.abc import Iterator

import pytest

import pup_finetune
from pup_finetune.errors import (
    ERROR_CODES,
    ConfirmRequired,
    ExternalError,
    FinetuneError,
    LockBusy,
    NotADatasetDir,
    NotARunDir,
    ReadOnlyError,
    RunDirNotEmpty,
    SettingsConflict,
    StateError,
    UsageError,
    WrongPhase,
    require_confirm,
    require_writable,
)
from pup_finetune.runtime import Runtime


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (FinetuneError("x"), 1),
        (UsageError("x"), 2),
        (ConfirmRequired("x", preview={}, rerun="pup finetune x --confirm"), 2),
        (ReadOnlyError("x"), 2),
        (ExternalError("x", source="pup"), 1),
        (StateError("x"), 1),
    ],
)
def test_exit_codes(error: FinetuneError, code: int) -> None:
    assert error.exit_code == code
    assert type(error).exit_code == code
    assert isinstance(error, FinetuneError)


def test_message_and_hint() -> None:
    err = UsageError("bad flag", hint="try --help")
    assert str(err) == "bad flag"
    assert err.message == "bad flag"
    assert err.hint == "try --help"
    assert StateError("broken").hint is None


def test_confirm_required_carries_preview_and_rerun() -> None:
    preview = {"cost": "estimate", "rows": 3}
    err = ConfirmRequired("needs confirm", preview=preview, rerun="pup finetune train --confirm")
    assert err.preview == preview
    assert err.preview is not preview
    assert err.rerun == "pup finetune train --confirm"


def test_external_error_names_its_source() -> None:
    err = ExternalError("exit 1", source="pup")
    assert err.source == "pup"
    assert str(err) == "pup: exit 1"


def test_require_writable_passes_when_not_read_only() -> None:
    require_writable(Runtime(), "dataset push")


def test_require_writable_rejects_when_read_only() -> None:
    with pytest.raises(ReadOnlyError, match="dataset push") as info:
        require_writable(Runtime(read_only=True), "dataset push")
    assert info.value.hint is not None


def test_require_confirm_passes_when_confirmed() -> None:
    require_confirm(True, {"a": 1}, "pup finetune x --confirm")


def test_require_confirm_raises_with_preview() -> None:
    with pytest.raises(ConfirmRequired) as info:
        require_confirm(False, {"a": 1}, "pup finetune x --confirm")
    assert info.value.preview == {"a": 1}
    assert info.value.rerun == "pup finetune x --confirm"


def test_require_confirm_documents_the_auto_approve_policy() -> None:
    # The behavior is covered end to end in tests/cli/test_main.py; here, the documented policy.
    assert "PUP_AUTO_APPROVE" in (require_confirm.__doc__ or "")


def _subclasses(cls: type[FinetuneError]) -> Iterator[type[FinetuneError]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _import_everything() -> None:
    """Import every module of the package, so every FinetuneError subclass is defined."""
    for module in pkgutil.walk_packages(pup_finetune.__path__, "pup_finetune."):
        with contextlib.suppress(ImportError):  # an extra that is not installed
            importlib.import_module(module.name)


def test_every_error_class_has_a_documented_code_with_its_exit_code() -> None:
    _import_everything()
    classes = [FinetuneError, *_subclasses(FinetuneError)]
    assert len(classes) > 20
    for cls in classes:
        assert cls.error_code in ERROR_CODES, f"{cls.__qualname__}: {cls.error_code!r}"
        exit_code, _ = ERROR_CODES[cls.error_code]
        assert cls.exit_code == exit_code, f"{cls.__qualname__} exits {cls.exit_code}"


def test_every_documented_code_is_used() -> None:
    _import_everything()
    used = {cls.error_code for cls in [FinetuneError, *_subclasses(FinetuneError)]}
    assert used == set(ERROR_CODES)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (NotADatasetDir("x"), "not_a_dataset_dir"),
        (NotARunDir("x"), "not_a_run_dir"),
        (RunDirNotEmpty("x"), "run_dir_not_empty"),
        (SettingsConflict("x"), "settings_conflict"),
        (WrongPhase("x"), "wrong_phase"),
    ],
)
def test_directory_mistakes_exit_2_and_are_still_caught_as_before(
    error: FinetuneError, code: str
) -> None:
    # Raised where a UsageError or a StateError was, so either kind of except still holds.
    assert isinstance(error, UsageError)
    assert isinstance(error, StateError)
    assert error.exit_code == 2
    assert error.error_code == code


def test_a_lock_held_by_another_process_exits_1() -> None:
    assert LockBusy("x").exit_code == 1
    assert LockBusy("x").error_code == "lock_busy"
