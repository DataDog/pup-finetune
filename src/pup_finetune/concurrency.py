# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Worker threads that never hold up the process's exit.

:class:`DaemonPool` runs calls on daemon threads, for the commands that make slow remote
calls in parallel (such as replay's generation and judge calls) and must stop at once on
Ctrl-C instead of waiting for a call in flight to return.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


class DaemonPool:
    """A fixed pool of daemon worker threads running submitted calls as :class:`Future`\\s.

    Unlike :class:`~concurrent.futures.ThreadPoolExecutor`, whose workers the interpreter
    joins at exit, an abandoned worker never delays the process's exit: after
    ``shutdown(wait=False)`` (on Ctrl-C), a call still blocked in a provider is left behind.
    """

    def __init__(self, workers: int, *, name: str) -> None:
        self._queue: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]] | None]
        self._queue = queue.SimpleQueue()
        self._threads = [
            threading.Thread(target=self._work, name=f"{name}-{n}", daemon=True)
            for n in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit[T](self, fn: Callable[..., T], *args: Any) -> Future[T]:
        future: Future[T] = Future()
        self._queue.put((future, lambda: fn(*args)))
        return future

    def shutdown(self, *, wait: bool) -> None:
        """Cancel the calls not started and stop the workers; with ``wait``, join them."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
        for _ in self._threads:
            self._queue.put(None)
        if wait:
            for thread in self._threads:
                thread.join()

    def _work(self) -> None:
        while (item := self._queue.get()) is not None:
            future, call = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = call()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
