# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Shared pytest configuration: opt-in gates, a clean env, no real keychain, reset logging."""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import keyring.core
import pytest

from pup_finetune import credentials, judge_credentials, next_steps, output
from tests.fakes.keyring import OS_KEYCHAIN_FAKES, FailKeyring

pytest_plugins = ["pytester"]

# Runtime env vars a developer's shell (or pup itself) may set; tests start without them.
_RUNTIME_ENV: tuple[str, ...] = (
    "PUP_OUTPUT",
    "PUP_FILTER",
    "PUP_READ_ONLY",
    "PUP_AUTO_APPROVE",
    "PUP_AGENT_MODE",
    # What a coding agent sets for its commands (the tests may run under one).
    "AI_AGENT",
    "CLAUDECODE",
    "DD_SITE",
    "PUP_FINETUNE_DEV",
    "PUP_FINETUNE_DEBUG",
    "PUP_FINETUNE_QUIET",
    "PUP_FINETUNE_PROGRESS",
    "PUP_FINETUNE_PUP",
    "PUP_FINETUNE_KEEP_DD_TOKEN",
    "PUP_FINETUNE_BASETEN_CLI",
    # Terminal and color overrides that change how rich renders stderr.
    "FORCE_COLOR",
    "NO_COLOR",
    "TTY_COMPATIBLE",
    "TTY_INTERACTIVE",
    "COLUMNS",
    "TERM",
)

_GATES: dict[str, str] = {
    "live": "PUP_FINETUNE_LIVE",
    "tokenizer": "PUP_FINETUNE_TOKENIZER_TESTS",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, env_var in _GATES.items():
        if os.environ.get(env_var) == "1":
            continue
        skip = pytest.mark.skip(reason=f"set {env_var}=1 to run {marker} tests")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _ctrl_c_raises() -> Iterator[None]:
    """Make SIGINT raise :class:`KeyboardInterrupt` for the whole session, as in an
    interactive Python, even when pytest started with SIGINT ignored (``trap '' INT``, a
    background job, some agents' shells): the tests that signal a process expect the
    default. The handler pytest started with comes back after the session."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.fixture(autouse=True)
def _clean_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)


# What the Anthropic SDK signs in with on its own; tests start without any of it.
_ANTHROPIC_SDK_ENV: tuple[str, ...] = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    # Where the SDK sends requests when no base URL is passed (a gateway, say).
    "ANTHROPIC_BASE_URL",
)


@pytest.fixture(autouse=True)
def _no_anthropic_sdk_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Hide the developer's own Anthropic SDK login (env and ``ant auth login`` profiles)."""
    for name in _ANTHROPIC_SDK_ENV:
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path_factory.mktemp("anthropic-config")
    monkeypatch.setattr(judge_credentials, "_sdk_config_dir", lambda env: empty)


PROGRESS_TIME = datetime(2026, 10, 7, 18, 3, 16)
"""The time of day every test's progress lines are stamped with (``18:03:16``)."""


@pytest.fixture(autouse=True)
def _fixed_progress_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stamp progress lines with :data:`PROGRESS_TIME`, so tests can assert them exactly."""
    monkeypatch.setattr(output, "_now", lambda: PROGRESS_TIME)


@pytest.fixture(autouse=True)
def _no_real_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a keyring with no backend, so none can touch the real keychain."""
    monkeypatch.setattr(keyring.core, "_keyring_backend", FailKeyring())


@pytest.fixture(autouse=True)
def _trust_fake_keychains(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the OS keychain fakes pass the credentials allowlist, as real keychains do."""
    fakes = {credentials._backend_name(fake) for fake in OS_KEYCHAIN_FAKES}
    monkeypatch.setattr(
        credentials, "_OS_KEYCHAIN_BACKENDS", credentials._OS_KEYCHAIN_BACKENDS | fakes
    )


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Undo global logging changes (e.g. log redaction installed by ``run_app``)."""
    root = logging.getLogger()
    make_record = logging.Logger.makeRecord
    root_filters = list(root.filters)
    last_resort = logging.lastResort
    last_resort_filters = list(last_resort.filters) if last_resort is not None else []
    yield
    setattr(logging.Logger, "makeRecord", make_record)  # noqa: B010
    root.filters[:] = root_filters
    logging.lastResort = last_resort
    if last_resort is not None:
        last_resort.filters[:] = last_resort_filters


@pytest.fixture(autouse=True)
def _next_steps_rule(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Fail any test whose emitted result breaks the next_command rule
    (:func:`pup_finetune.next_steps.problems`): a next_command runs as written or is null,
    and next_steps use only <UPPER_CASE> placeholders. Tests marked ``any_next_command``
    emit such results on purpose."""
    if request.node.get_closest_marker("any_next_command") is not None:
        yield
        return
    found: list[str] = []
    to_json = output.to_json

    def checked(result: Any) -> str:
        found.extend(next_steps.problems(result))
        return to_json(result)

    monkeypatch.setattr(output, "to_json", checked)
    yield
    assert not found, "results break the next_command rule:\n" + "\n".join(found)
