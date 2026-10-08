# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The inputs digest a preview pins and ``--expect`` checks (:mod:`pup_finetune.approval`)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from pup_finetune.approval import (
    DIGEST_LEN,
    Part,
    inputs_digest,
    require_expected,
    with_digest,
    with_settings,
)
from pup_finetune.errors import SettingsConflict, UsageError

ROWS = Part("the dataset", {"train": ["a", "b"], "validation": ["c"]}, now="2 train / 1 validation")
SETTINGS = Part("the settings", {"epochs": 5, "rate": 1e-4})
PARTS = [ROWS, SETTINGS, Part("the evaluation", None)]


def test_a_digest_is_short_hex_and_the_same_for_the_same_inputs() -> None:
    digest = inputs_digest(PARTS)
    assert len(digest) == DIGEST_LEN
    assert int(digest, 16) >= 0
    # Key order does not matter: values are hashed as canonical JSON.
    reordered = [Part(ROWS.name, {"validation": ["c"], "train": ["a", "b"]}), SETTINGS, PARTS[2]]
    assert inputs_digest(reordered) == digest


def test_a_digest_is_the_same_in_another_process() -> None:
    code = (
        "from pup_finetune.approval import Part, inputs_digest\n"
        "print(inputs_digest([Part('the dataset', {'train': ['a', 'b'], 'validation': ['c']}),"
        " Part('the settings', {'epochs': 5, 'rate': 1e-4}), Part('the evaluation', None)]))"
    )
    other = subprocess.run(  # noqa: S603 - a fixed script run by this interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert other == inputs_digest(PARTS)


def test_the_expected_digest_passes_and_none_checks_nothing() -> None:
    require_expected(inputs_digest(PARTS), PARTS, preview="pup finetune train")
    require_expected(inputs_digest(PARTS).upper(), PARTS, preview="pup finetune train")
    require_expected(None, PARTS, preview="pup finetune train")


def test_a_mismatch_names_the_part_that_changed_and_how_it_is_now() -> None:
    expected = inputs_digest(PARTS)
    now = [Part(ROWS.name, {"train": ["a", "b"], "validation": []}, now="2 train / 0 validation")]
    with pytest.raises(SettingsConflict) as raised:
        require_expected(expected, [*now, *PARTS[1:]], preview="pup finetune train --x")
    assert raised.value.message == (
        "the dataset changed since the preview (now 2 train / 0 validation); preview again"
    )
    assert raised.value.hint is not None and "`pup finetune train --x`" in raised.value.hint
    assert raised.value.exit_code == 2


def test_several_changed_parts_are_all_named() -> None:
    expected = inputs_digest(PARTS)
    changed = [Part(ROWS.name, [], now="0 rows"), Part(SETTINGS.name, {"epochs": 1}), PARTS[2]]
    with pytest.raises(SettingsConflict, match="the dataset and the settings changed"):
        require_expected(expected, changed, preview="p")


def test_a_malformed_digest_is_refused() -> None:
    with pytest.raises(UsageError, match="is not an inputs_digest"):
        require_expected("nope", PARTS, preview="p")


def test_the_digest_and_settings_go_where_a_preview_shows_them() -> None:
    preview = {"action": "a", "data_destination": "d", "warnings": ["w"]}
    shown = with_digest(with_settings(preview, {"x": 1}, ["x"]), "abc")
    assert list(shown) == [
        "action",
        "settings",
        "defaults_used",
        "data_destination",
        "inputs_digest",
        "warnings",
    ]
