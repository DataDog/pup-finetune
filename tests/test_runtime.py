# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
from __future__ import annotations

import dataclasses

import pytest

from pup_finetune.runtime import Runtime


def test_empty_env_gives_defaults() -> None:
    runtime = Runtime.from_env({})
    assert runtime == Runtime()
    assert runtime.output_format is None
    assert runtime.jq_filter is None
    assert runtime.dd_site is None
    assert not runtime.read_only
    assert not runtime.auto_approve
    assert not runtime.agent_mode
    assert not runtime.dev_mode
    assert not runtime.debug
    assert not runtime.under_pup


def test_full_env_is_parsed() -> None:
    runtime = Runtime.from_env(
        {
            "PUP_OUTPUT": "table",
            "PUP_FILTER": ".id",
            "PUP_READ_ONLY": "true",
            "PUP_AUTO_APPROVE": "true",
            "PUP_AGENT_MODE": "true",
            "DD_SITE": "datadoghq.eu",
            "PUP_FINETUNE_DEV": "1",
            "PUP_FINETUNE_DEBUG": "1",
        }
    )
    assert runtime == Runtime(
        output_format="table",
        jq_filter=".id",
        read_only=True,
        auto_approve=True,
        agent_mode=True,
        dd_site="datadoghq.eu",
        dev_mode=True,
        debug=True,
    )
    assert runtime.under_pup


@pytest.mark.parametrize("value", ["1", "TRUE", "True", "yes", ""])
def test_pup_flags_require_exact_true(value: str) -> None:
    runtime = Runtime.from_env(
        {"PUP_READ_ONLY": value, "PUP_AUTO_APPROVE": value, "PUP_AGENT_MODE": value}
    )
    assert not runtime.read_only
    assert not runtime.auto_approve
    assert not runtime.agent_mode


@pytest.mark.parametrize("value", ["true", "0", "yes", ""])
def test_finetune_flags_require_exact_one(value: str) -> None:
    runtime = Runtime.from_env({"PUP_FINETUNE_DEV": value, "PUP_FINETUNE_DEBUG": value})
    assert not runtime.dev_mode
    assert not runtime.debug


def test_empty_output_format_is_not_under_pup() -> None:
    assert not Runtime.from_env({"PUP_OUTPUT": ""}).under_pup


def test_default_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUP_OUTPUT", "json")
    monkeypatch.setenv("PUP_READ_ONLY", "true")
    runtime = Runtime.from_env()
    assert runtime.output_format == "json"
    assert runtime.read_only


def test_runtime_is_frozen() -> None:
    runtime = Runtime()
    with pytest.raises(dataclasses.FrozenInstanceError):
        runtime.read_only = True  # type: ignore[misc]


def test_secret_env_vars_are_not_read() -> None:
    secret = "synthetic-secret-value"  # noqa: S105 - fake value for the test
    runtime = Runtime.from_env(
        {"DD_API_KEY": secret, "DD_APP_KEY": secret, "FIREWORKS_API_KEY": secret}
    )
    assert runtime == Runtime()
    assert secret not in repr(runtime)
