# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pup_finetune.errors import UsageError
from pup_finetune.redaction import register_secret
from pup_finetune.user_checks import CheckError, CheckVerdict, call, load_checks

CHECKS = """\
from json import loads


def first(case, candidate):
    return True


def _helper():
    return 1


def second(case, candidate):
    return None


def mutates(case, candidate):
    case["prefix"].append("x")
    return True
"""


def write(tmp_path: Path, text: str, name: str = "checks.py") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_lists_only_the_public_functions_the_file_defines_in_order(tmp_path: Path) -> None:
    module = load_checks(write(tmp_path, CHECKS))
    assert [name for name, _ in module.functions()] == ["first", "second", "mutates"]


def test_a_wrong_arity_or_a_missing_function_is_a_usage_error(tmp_path: Path) -> None:
    module = load_checks(write(tmp_path, CHECKS))
    assert module.function("first", 2) is dict(module.functions())["first"]
    with pytest.raises(UsageError, match="must take 1 argument"):
        module.function("first", 1)
    with pytest.raises(UsageError, match="no public function '_helper'"):
        module.function("_helper", 0)
    with pytest.raises(UsageError, match="no public function 'loads'"):
        module.function("loads", 1)


@pytest.mark.parametrize(
    ("text", "kind"),
    [("def broken(:\n", "SyntaxError"), ("import not_a_module_anywhere_0\n", "ModuleNotFound")],
)
def test_a_file_that_cannot_load_is_a_usage_error_naming_it(
    tmp_path: Path, text: str, kind: str
) -> None:
    with pytest.raises(UsageError, match=rf"bad\.py could not be loaded: {kind}"):
        load_checks(write(tmp_path, text, "bad.py"))
    with pytest.raises(UsageError, match="cannot be read"):
        load_checks(tmp_path / "missing.py")


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        (True, CheckVerdict(True)),
        (False, CheckVerdict(False)),
        (None, CheckVerdict(None)),
        ((False, " why "), CheckVerdict(False, "why")),
        ((True, "x" * 600), CheckVerdict(True, "x" * 500)),
    ],
)
def test_answers_normalise_to_a_verdict(answer: Any, verdict: CheckVerdict) -> None:
    assert call("c", lambda: answer) == verdict


@pytest.mark.parametrize("answer", [1, "yes", (1, 2), (True,), [True, "why"]])
def test_any_other_answer_is_a_check_error(answer: Any) -> None:
    with pytest.raises(CheckError, match="check c: returned"):
        call("c", lambda: answer)


def test_a_reason_utf8_cannot_encode_is_a_check_error() -> None:
    with pytest.raises(CheckError, match=r"check c: returned a reason with a lone surrogate"):
        call("c", lambda: (False, "\ud800 nope"))


def test_an_exception_becomes_a_redacted_check_error() -> None:
    register_secret("sk-test-secret-0001")

    def raises() -> None:
        raise ValueError("token sk-test-secret-0001 leaked")

    with pytest.raises(CheckError) as caught:
        call("c", raises)
    assert caught.value.check == "c"
    assert "sk-test-secret-0001" not in str(caught.value)
    assert "ValueError" in caught.value.message


def test_checks_get_deep_copies(tmp_path: Path) -> None:
    module = load_checks(write(tmp_path, CHECKS))
    case: dict[str, Any] = {"prefix": []}
    assert call("mutates", module.function("mutates", 2), case, {}).passed is True
    assert case == {"prefix": []}


def test_checks_run_one_at_a_time() -> None:
    running, overlaps = [0], []

    def slow() -> bool:
        running[0] += 1
        overlaps.append(running[0])
        time.sleep(0.01)
        running[0] -= 1
        return True

    threads = [threading.Thread(target=call, args=("c", slow)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(overlaps) == 1


def test_the_sha256_changes_with_any_byte(tmp_path: Path) -> None:
    first = load_checks(write(tmp_path, CHECKS))
    second = load_checks(write(tmp_path, CHECKS + "\n"))
    assert first.sha256 != second.sha256
    assert first.module is not second.module
