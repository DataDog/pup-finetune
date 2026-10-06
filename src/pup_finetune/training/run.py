# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A training run directory: the plan, per-epoch records, run state and result as JSON files.

Every write is atomic and ``0600``. Callers hold :class:`~pup_finetune.storage.DirectoryLock` on
the directory while writing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pup_finetune.errors import StateError
from pup_finetune.storage import LOCK_FILE_NAME, atomic_write_json, ensure_private_dir, read_json

PLAN_FILE = "plan.json"
EPOCHS_FILE = "epochs.json"
STATE_FILE = "run-state.json"
RESULT_FILE = "result.json"


class RunDir:
    """The files of one training run under :attr:`path`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def create(self) -> None:
        """Create the directory (``0700``). An existing one must be empty, apart from its lock."""
        if self.path.is_dir() and any(e.name != LOCK_FILE_NAME for e in self.path.iterdir()):
            raise StateError(f"{self.path} is not empty", hint="use a new --run-dir")
        ensure_private_dir(self.path)

    def write_plan(self, plan: Mapping[str, Any]) -> None:
        atomic_write_json(self.path / PLAN_FILE, dict(plan))

    def write_epochs(self, epochs: Sequence[Mapping[str, Any]]) -> None:
        atomic_write_json(self.path / EPOCHS_FILE, [dict(epoch) for epoch in epochs])

    def update_state(self, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into the run state and return the new state."""
        state = self.read_state() if (self.path / STATE_FILE).exists() else {}
        state.update(fields)
        atomic_write_json(self.path / STATE_FILE, state)
        return state

    def write_result(self, result: Mapping[str, Any]) -> None:
        atomic_write_json(self.path / RESULT_FILE, dict(result))

    def read_plan(self) -> dict[str, Any]:
        return self._read(PLAN_FILE)

    def read_result(self) -> dict[str, Any]:
        return self._read(RESULT_FILE)

    def read_state(self) -> dict[str, Any]:
        return self._read(STATE_FILE)

    def _read(self, name: str) -> dict[str, Any]:
        path = self.path / name
        try:
            data = read_json(path)
        except FileNotFoundError as exc:
            raise StateError(
                f"{path} does not exist", hint="pass the --run-dir of a training run"
            ) from exc
        if not isinstance(data, dict):
            raise StateError(f"{path}: expected a JSON object")
        return data
