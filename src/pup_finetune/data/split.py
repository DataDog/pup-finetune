# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Deterministic train / validation / test splits that stay stable as rows are added.

A row's split is recorded per split key (its trace id; see
:func:`~pup_finetune.data.schema.split_key`), so every row of a trace lands in the same split.
Copies of one request share a split too: rows whose user messages differ only in numbers,
case or whitespace (:func:`request_key`) and the rows of their traces form one group
(:func:`split_groups`), and :func:`assign_splits` gives a group one split, so a test answer
is not also trained on. A short request (a fixed prompt such as "Summarize the above") joins
nothing, and a request repeated in more rows than :func:`group_cap` allows is left
ungrouped and reported rather than pulling a large share of the dataset into one split.
Assignments are persisted in ``splits.json``: once a key is assigned it never moves, and new
keys are assigned with the fractions frozen there. Copies recorded in different splits
before they were grouped stay there; :func:`split_conflicts` names them.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
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


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SPACE = re.compile(r"\s+")

REQUEST_MIN_CHARS = 64
"""A request whose normalized user text is shorter than this joins nothing: short fixed
prompts are asked in unrelated conversations, so sharing one says nothing about copies."""
GROUP_SHARE = 0.05
"""The largest share of a dataset's rows a request may gather into one group."""
GROUP_MIN_ROWS = 4
"""The cap on a request's group in a dataset small enough that :data:`GROUP_SHARE` of it
is fewer rows, so a pair of copies always groups."""
SHOWN_CONFLICT_IDS = 10
"""How many row ids per split a summary names for split conflicts."""


def request_key(row: TrainingRow) -> str | None:
    """A hash of what ``row`` asks: every user message's text, lowercased, with numbers as
    ``0`` and whitespace collapsed. ``None`` when that text is shorter than
    :data:`REQUEST_MIN_CHARS`, so such rows are grouped by trace alone."""
    texts = []
    for message in row.messages:
        if message.role == "user":
            texts.append(_SPACE.sub(" ", _NUMBER.sub("0", message.content.lower())).strip())
    text = "\x1f".join(texts)
    if len(text) < REQUEST_MIN_CHARS:
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def group_cap(rows: int) -> int:
    """The most rows a request may gather into one group in a dataset of ``rows`` rows."""
    return max(GROUP_MIN_ROWS, math.floor(GROUP_SHARE * rows))


@dataclass(frozen=True, slots=True)
class SplitGroups:
    """Rows in the groups that share a split (:func:`split_groups`)."""

    groups: list[list[TrainingRow]]
    """The rows of each group, groups and rows in the order of the rows given."""
    ungrouped: list[list[str]]
    """The row ids of each request too common to group by: joining its copies would grow
    their group past :func:`group_cap`, so they are split by trace instead."""
    cap: int


def split_groups(rows: Iterable[TrainingRow]) -> SplitGroups:
    """``rows`` in groups that share a split: the rows of one trace, joined with every row
    asking the same request (:func:`request_key`), transitively, unless that would grow a
    group past :func:`group_cap` rows. Requests are joined in the order of their keys, so
    the groups depend on the set of rows only."""
    listed = list(rows)
    parent: dict[str, str] = {}
    size = Counter(split_key(row) for row in listed)

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    asked: dict[str, list[TrainingRow]] = {}
    for row in listed:
        if (request := request_key(row)) is not None:
            asked.setdefault(request, []).append(row)
    cap = group_cap(len(listed))
    ungrouped: list[list[str]] = []
    for request in sorted(asked):
        copies = asked[request]
        roots = sorted({find(split_key(row)) for row in copies})
        if len(roots) < 2:
            continue
        if sum(size[root] for root in roots) > cap:
            ungrouped.append([row.id for row in copies])
            continue
        for root in roots[1:]:
            parent[root] = roots[0]
            size[roots[0]] += size.pop(root)
    groups: dict[str, list[TrainingRow]] = {}
    for row in listed:
        groups.setdefault(find(split_key(row)), []).append(row)
    return SplitGroups(list(groups.values()), ungrouped, cap)


def _most(counts: Counter[SplitName]) -> SplitName:
    """The split ``counts`` holds most of, the first of :data:`SPLITS` on a tie."""
    return max(SPLITS, key=lambda name: (counts[name], -SPLITS.index(name)))


def assign_splits(
    rows: Iterable[TrainingRow], splits: SplitAssignments, *, dropped: Collection[str] = ()
) -> None:
    """Assign each split key of ``rows`` that ``splits`` does not hold yet, by group
    (:func:`split_groups`); ``dropped`` are the ids of rows labeled drop.

    A recorded key never moves. A new key joining a group with recorded keys takes the
    split holding most of the group's kept rows (all of its rows, when none is kept), the
    first of :data:`SPLITS` on a tie. A group with no key recorded takes the split of its
    first key in sorted order. Within one call the result depends on the set of rows only,
    not their order; rows added over several calls can land apart from where the same rows
    added at once would, since what is recorded stays.
    """
    for group in split_groups(rows).groups:
        keys = sorted({split_key(row) for row in group})
        new = [key for key in keys if splits.get(key) is None]
        if not new:
            continue
        held = [(row, splits.get(split_key(row))) for row in group]
        recorded = Counter(
            split for row, split in held if split is not None and row.id not in dropped
        ) or Counter(split for _row, split in held if split is not None)
        split = _most(recorded) if recorded else splits.params.assign(keys[0])
        for key in new:
            splits.assignments[key] = split


@dataclass(frozen=True, slots=True)
class SplitConflict:
    """A group whose kept rows are in more than one split: copies of a request recorded
    apart before they were grouped. Recorded splits never move, so dropping the copies
    outside :attr:`split` keeps the others from being both trained and tested on."""

    split: SplitName
    """The split holding most of the group's kept rows (the first of :data:`SPLITS` on a
    tie)."""
    outside: dict[SplitName, list[str]]
    """The ids of the group's kept rows in each other split."""


def split_conflicts(
    groups: SplitGroups, splits: SplitAssignments, *, dropped: Collection[str] = ()
) -> list[SplitConflict]:
    """Every group of ``groups`` whose rows not in ``dropped`` are in more than one split of
    ``splits``; a row without a recorded split is left out."""
    conflicts: list[SplitConflict] = []
    for group in groups.groups:
        by_split: dict[SplitName, list[str]] = {}
        for row in group:
            if row.id not in dropped and (split := splits.get(split_key(row))) is not None:
                by_split.setdefault(split, []).append(row.id)
        if len(by_split) < 2:
            continue
        most = _most(Counter({split: len(ids) for split, ids in by_split.items()}))
        conflicts.append(
            SplitConflict(most, {split: ids for split, ids in by_split.items() if split != most})
        )
    return conflicts


def conflict_summary(conflicts: Sequence[SplitConflict]) -> dict[str, Any]:
    """``conflicts`` as a dataset summary reports them: how many groups, how many kept rows
    are outside their group's split, and up to :data:`SHOWN_CONFLICT_IDS` of their ids per
    split."""
    outside: dict[SplitName, list[str]] = {}
    for conflict in conflicts:
        for split, ids in conflict.outside.items():
            outside.setdefault(split, []).extend(ids)
    return {
        "groups": len(conflicts),
        "rows": sum(len(ids) for ids in outside.values()),
        "row_ids": {
            split: sorted(outside[split])[:SHOWN_CONFLICT_IDS]
            for split in SPLITS
            if split in outside
        },
    }


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
