# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The opt-in gates in conftest.py skip live and tokenizer tests unless enabled."""

from __future__ import annotations

from pathlib import Path

import pytest

_CONFTEST = Path(__file__).with_name("conftest.py").read_text()

_TESTS = """
import pytest

@pytest.mark.live
def test_live():
    pass

@pytest.mark.tokenizer
def test_tokenizer():
    pass

def test_plain():
    pass
"""


@pytest.fixture
def gated(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(_CONFTEST)
    pytester.makeini(
        "[pytest]\nmarkers =\n    live: live\n    tokenizer: tokenizer\naddopts = --strict-markers\n"
    )
    pytester.makepyfile(_TESTS)
    return pytester


@pytest.fixture(autouse=True)
def _clear_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUP_FINETUNE_LIVE", raising=False)
    monkeypatch.delenv("PUP_FINETUNE_TOKENIZER_TESTS", raising=False)


def test_gated_tests_skip_by_default(gated: pytest.Pytester) -> None:
    gated.runpytest().assert_outcomes(passed=1, skipped=2)


def test_live_gate_enables_live_tests(
    gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_LIVE", "1")
    gated.runpytest().assert_outcomes(passed=2, skipped=1)


def test_tokenizer_gate_enables_tokenizer_tests(
    gated: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUP_FINETUNE_TOKENIZER_TESTS", "1")
    gated.runpytest().assert_outcomes(passed=2, skipped=1)


_KEY_TESTS = """
import os
import pytest

@pytest.mark.live
def test_live_keeps_the_key():
    assert os.environ["FIREWORKS_API_KEY"] == "fw-shell-key"

def test_plain_gets_no_key():
    assert "FIREWORKS_API_KEY" not in os.environ
"""


def test_only_live_tests_see_a_provider_key_from_the_shell(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeconftest(_CONFTEST)
    pytester.makeini("[pytest]\nmarkers =\n    live: live\n    tokenizer: tokenizer\n")
    pytester.makepyfile(_KEY_TESTS)
    monkeypatch.setenv("PUP_FINETUNE_LIVE", "1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-shell-key")
    pytester.runpytest().assert_outcomes(passed=2)
