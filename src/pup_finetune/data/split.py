# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Deterministic train / validation / test splits that stay stable as rows are added.

A row's split comes from its split key (its trace id; see
:func:`~pup_finetune.data.schema.split_key`), so every row of a trace lands in the same split.
Assignments are persisted in ``splits.json``: once a key is assigned it never moves, and new
keys are assigned with the fractions frozen there.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, cast

from pup_finetune.data.schema import TrainingRow, row_content_hash, split_key
from pup_finetune.errors import StateError, UsageError
from pup_finetune.storage import atomic_write_json, read_json

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")

DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1

MIN_TRAIN_ROWS = 3
"""Fewer training rows than this produce a warning."""

_HASH_SPACE = 2**64


def split_position(key: str, *, seed: int) -> float:
    """Where ``key`` falls in ``[0, 1)``: the first 64 bits of ``sha256(f"{seed}:{key}")``."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:16], 16) / _HASH_SPACE


def assign_split(
    key: str, *, seed: int, validation_fraction: float, test_fraction: float
) -> SplitName:
    """The split for ``key``: ``test`` below ``test_fraction``, then ``validation``, else ``train``."""
    position = split_position(key, seed=seed)
    if position < test_fraction:
        return "test"
    if position < test_fraction + validation_fraction:
        return "validation"
    return "train"


@dataclass(frozen=True, slots=True)
class SplitParams:
    """The seed and fractions splits are assigned with. Invalid fractions raise UsageError."""

    seed: int = DEFAULT_SEED
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION

    def __post_init__(self) -> None:
        for name in ("validation_fraction", "test_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                flag = "--" + name.replace("_", "-")
                raise UsageError(f"{flag} must be at least 0 and less than 1, got {value}")
        if self.validation_fraction + self.test_fraction >= 1:
            raise UsageError(
                "--validation-fraction plus --test-fraction must be less than 1, "
                "so some rows are left to train on"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, where: str) -> Self:
        """Parse stored params; anything malformed raises :class:`StateError` naming ``where``."""
        seed = data.get("seed")
        fractions = [data.get("validation_fraction"), data.get("test_fraction")]
        if type(seed) is not int or not all(
            isinstance(f, int | float) and not isinstance(f, bool) for f in fractions
        ):
            raise StateError(f"{where}: invalid split settings")
        try:
            return cls(seed, float(cast(float, fractions[0])), float(cast(float, fractions[1])))
        except UsageError as exc:
            raise StateError(f"{where}: invalid split settings: {exc.message}") from exc

    def assign(self, key: str) -> SplitName:
        return assign_split(
            key,
            seed=self.seed,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
        )


@dataclass(slots=True)
class SplitAssignments:
    """The persisted split of every key seen so far (``splits.json``)."""

    params: SplitParams
    assignments: dict[str, SplitName] = field(default_factory=dict)

    def get(self, key: str) -> SplitName | None:
        return self.assignments.get(key)

    def assign(self, key: str) -> SplitName:
        """The split for ``key``: its recorded one, or a new assignment that is then recorded."""
        split = self.assignments.get(key)
        if split is None:
            split = self.assignments[key] = self.params.assign(key)
        return split

    def to_json(self) -> dict[str, Any]:
        return {**self.params.to_json(), "assignments": dict(self.assignments)}

    @classmethod
    def from_json(cls, data: Any, *, where: str) -> Self:
        if not isinstance(data, dict):
            raise StateError(f"{where}: expected a JSON object")
        params = SplitParams.from_json(data, where=where)
        raw = data.get("assignments", {})
        if not isinstance(raw, dict) or not all(
            isinstance(key, str) and value in SPLITS for key, value in raw.items()
        ):
            raise StateError(f"{where}: invalid split assignments")
        return cls(params, cast(dict[str, SplitName], dict(raw)))

    @classmethod
    def load(cls, path: Path, params: SplitParams) -> Self:
        """Read ``path``, or start empty when it is missing; its params must equal ``params``."""
        try:
            data = read_json(path)
        except FileNotFoundError:
            return cls(params)
        loaded = cls.from_json(data, where=str(path))
        if loaded.params != params:
            raise StateError(
                f"{path} was written with different split settings",
                hint="use the split settings the dataset was created with, or a new directory",
            )
        return loaded

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_json())


def check_isolation(rows_by_split: Mapping[SplitName, Iterable[TrainingRow]]) -> None:
    """Raise :class:`StateError` when the same conversation is in two splits.

    Conversations are compared by :func:`~pup_finetune.data.schema.row_content_hash`, so a
    duplicate under another id or trace still counts as leakage.
    """
    seen: dict[str, tuple[SplitName, str]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            digest = row_content_hash(row)
            other = seen.setdefault(digest, (split, row.id))
            if other[0] != split:
                raise StateError(
                    f"rows {other[1]!r} ({other[0]}) and {row.id!r} ({split}) hold the same "
                    "conversation in different splits",
                    hint="remove one of the two rows",
                )


def split_counts(rows: Iterable[TrainingRow], params: SplitParams) -> dict[SplitName, int]:
    """How many of ``rows`` each split would get under ``params``, by split key, unpersisted."""
    counts: dict[SplitName, int] = dict.fromkeys(SPLITS, 0)
    for row in rows:
        counts[params.assign(split_key(row))] += 1
    return counts


def split_warnings(counts: Mapping[SplitName, int]) -> list[str]:
    """Warnings, not errors, about split sizes: empty validation or test, too few train rows."""
    warnings: list[str] = []
    train = counts.get("train", 0)
    if train < MIN_TRAIN_ROWS:
        warnings.append(f"only {train} training rows; at least {MIN_TRAIN_ROWS} are recommended")
    if not counts.get("validation", 0):
        warnings.append(
            "the validation split is empty, so training cannot select a checkpoint "
            "by validation loss"
        )
    if not counts.get("test", 0):
        warnings.append("the test split is empty, so there is nothing to evaluate on")
    return warnings
