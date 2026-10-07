# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Tests for :mod:`pup_finetune.concurrency`."""

from __future__ import annotations

import threading
import time

import pytest

from pup_finetune.concurrency import DaemonPool


def test_calls_run_and_their_results_and_errors_reach_the_futures() -> None:
    pool = DaemonPool(2, name="test")

    def fail() -> None:
        raise ValueError("boom")

    try:
        doubled = pool.submit(lambda x: x * 2, 21)
        failed = pool.submit(fail)
        assert doubled.result(timeout=5) == 42
        with pytest.raises(ValueError, match="boom"):
            failed.result(timeout=5)
    finally:
        pool.shutdown(wait=True)


def test_the_workers_are_daemon_threads() -> None:
    pool = DaemonPool(3, name="test-daemon")
    try:
        workers = [t for t in threading.enumerate() if t.name.startswith("test-daemon-")]
        assert len(workers) == 3
        assert all(worker.daemon for worker in workers)
    finally:
        pool.shutdown(wait=True)


def test_shutdown_without_waiting_leaves_a_blocked_call_and_cancels_the_rest() -> None:
    release = threading.Event()
    started = threading.Event()

    def blocked() -> str:
        started.set()
        release.wait(timeout=30)
        return "late"

    pool = DaemonPool(1, name="test-blocked")
    running = pool.submit(blocked)
    queued = pool.submit(lambda: "never")
    assert started.wait(timeout=5)
    began = time.monotonic()
    pool.shutdown(wait=False)
    assert time.monotonic() - began < 1
    assert queued.cancelled()
    assert not running.done()
    release.set()
    assert running.result(timeout=5) == "late"
