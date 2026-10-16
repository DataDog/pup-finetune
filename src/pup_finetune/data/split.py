# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Deterministic train / validation / test splits that stay stable as rows are added.

A row's split is recorded per split key (its trace id; see
:func:`~pup_finetune.data.schema.split_key`), so every row of a trace lands in the same split.
Copies of one request share a split too: rows whose user messages differ only in quoted
strings, numbers, case or whitespace (:func:`request_key`) and the rows of their traces form
one group (:func:`split_groups`), and :func:`assign_splits` gives a group one split, so a
test answer is not also trained on. Assignments are persisted in ``splits.json``: once a key
is assigned it never moves, and new keys are assigned with the fractions frozen there.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, cast

from pup_finetune.data.schema import TrainingRow, row_content_hash, split_key
from pup_finetune.errors import SettingsConflict, StateError, UsageError
from pup_finetune.storage import atomic_write_json, read_json

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")

DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1

MIN_TRAIN_ROWS = 3
"""Fewer training rows than this produce a warning."""
SKEW_MAX_TRACES = 50
"""With fewer traces than this, a split far from its fraction produces a warning
(:func:`skew_warnings`); with more, the per-trace assignment evens out."""
SKEW_POINTS = 0.10
"""How far, as a share of the rows, a split may be from its fraction before it is skewed
(it is also skewed at more than twice its fraction)."""

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
            raise SettingsConflict(
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


_QUOTED = re.compile(r""""(?:[^"\\]|\\.)*"|(?<!\w)'(?:[^'\\]|\\.)*'(?!\w)""")
"""A quoted string literal: double-quoted, or single-quoted where the quotes are not
apostrophes inside a word."""
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SPACE = re.compile(r"\s+")


def request_key(row: TrainingRow) -> str | None:
    """A hash of what ``row`` asks: every user message's text, lowercased, with quoted string
    literals as ``?``, numbers as ``0`` and whitespace collapsed. ``None`` when the row has
    no user text, so such rows are grouped by trace alone."""
    texts = []
    for message in row.messages:
        if message.role != "user":
            continue
        text = _NUMBER.sub("0", _QUOTED.sub("?", message.content.lower()))
        texts.append(_SPACE.sub(" ", text).strip())
    if not any(texts):
        return None
    return hashlib.sha256("\x1f".join(texts).encode()).hexdigest()


def split_groups(rows: Iterable[TrainingRow]) -> list[list[TrainingRow]]:
    """``rows`` in groups that share a split: the rows of one trace, joined with every row
    asking the same request (:func:`request_key`), transitively. Groups and the rows in each
    keep the order of ``rows``."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    listed = list(rows)
    for row in listed:
        trace = find(f"trace:{split_key(row)}")
        request = request_key(row)
        if request is not None:
            parent[find(f"request:{request}")] = trace
    groups: dict[str, list[TrainingRow]] = {}
    for row in listed:
        groups.setdefault(find(f"trace:{split_key(row)}"), []).append(row)
    return list(groups.values())


def assign_splits(rows: Iterable[TrainingRow], splits: SplitAssignments) -> int:
    """Assign each split key of ``rows`` that ``splits`` does not hold yet, by group
    (:func:`split_groups`); returns the number of new keys that joined groups already split
    two ways (``split_conflicts``).

    A recorded key never moves. A group with no key recorded takes the split of its first
    key in sorted order, so the result does not depend on the order of ``rows``. A new key
    joining a group whose recorded keys share one split takes it. One joining recorded keys
    of several splits (copies already apart, now linked by the new trace) takes the split
    holding most of the group's rows, the first of :data:`SPLITS` on a tie, and is counted.
    """
    conflicts = 0
    for group in split_groups(rows):
        keys = sorted({split_key(row) for row in group})
        new = [key for key in keys if splits.get(key) is None]
        if not new:
            continue
        recorded = Counter(
            held for row in group if (held := splits.get(split_key(row))) is not None
        )
        split: SplitName
        if not recorded:
            split = splits.params.assign(keys[0])
        else:
            split = max(SPLITS, key=lambda name: (recorded[name], -SPLITS.index(name)))
            if len(recorded) > 1:
                conflicts += len(new)
        for key in new:
            splits.assignments[key] = split
    return conflicts


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


def skew_warnings(
    keys_by_split: Mapping[SplitName, Sequence[str]], params: SplitParams, *, more: str
) -> list[str]:
    """Warnings about a validation or test split whose share of the rows is far from its
    fraction (more than twice it, or :data:`SKEW_POINTS` or more off) in a dataset of fewer
    than :data:`SKEW_MAX_TRACES` traces.

    ``keys_by_split`` holds the split key of every row, per split. ``more`` is what to do
    about it (``pull more rows``). Splits are assigned per group of traces, so a small dataset varies
    by chance; the seed of a frozen directory cannot change, so the warning never suggests
    it. An empty split is left to :func:`split_warnings`.
    """
    rows = sum(len(keys) for keys in keys_by_split.values())
    traces = {split: len(set(keys_by_split.get(split, ()))) for split in SPLITS}
    total_traces = sum(traces.values())
    if not rows or total_traces >= SKEW_MAX_TRACES:
        return []
    warnings: list[str] = []
    checked: tuple[tuple[SplitName, float], ...] = (
        ("validation", params.validation_fraction),
        ("test", params.test_fraction),
    )
    for split, fraction in checked:
        count = len(keys_by_split.get(split, ()))
        if not count or not fraction:
            continue
        share = count / rows
        if share > 2 * fraction or round(abs(share - fraction), 9) >= SKEW_POINTS:
            warnings.append(
                f"{split} has {count} of {rows} rows ({share:.0%}, {traces[split]} of "
                f"{total_traces} traces) though --{split}-fraction is {fraction:g}: splits "
                f"are assigned per trace and request, so small datasets vary; {more}"
            )
    return warnings
