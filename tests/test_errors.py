# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import pytest

from pup_finetune.errors import (
    ConfirmRequired,
    ExternalError,
    FinetuneError,
    ReadOnlyError,
    StateError,
    UsageError,
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
