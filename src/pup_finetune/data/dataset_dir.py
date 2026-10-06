# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The on-disk dataset directory: accepted rows, exclusions, splits and a summary.

Files, all ``0600`` in a ``0700`` directory, since rows hold trace content:

- ``settings.json`` - frozen on first use: the source kind and its settings, and the split
  params. Later imports must pass the same settings.
- ``rows.jsonl`` - accepted canonical rows.
- ``excluded.jsonl`` - one ``{row_id?, trace_id?, span_id?, line_sha256?, code, detail}`` per
  excluded row: the latest reason per ``row_id``, and none for an id that has since been
  accepted.
- ``labels.jsonl`` - keep/drop labels (written by a later command; optional).
- ``splits.json`` - split assignments (see :mod:`pup_finetune.data.split`).
- ``summary.json`` - counts and warnings, rewritten after every change.
- ``raw/`` - for a Datadog pull, the raw Export API pages it read, so an interrupted pull
  resumes without fetching them again (see
  :func:`~pup_finetune.datadog.spans.iter_span_pages`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.jsonl import read_rows, scan_rows, to_provider_dict, write_rows
from pup_finetune.data.schema import RowError, TrainingRow, row_content_hash, split_key
from pup_finetune.data.split import (
    SPLITS,
    SplitAssignments,
    SplitName,
    SplitParams,
    check_isolation,
    split_warnings,
)
from pup_finetune.errors import StateError, UsageError
from pup_finetune.storage import (
    LOCK_FILE_NAME,
    DirectoryLock,
    FrozenSettings,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    ensure_private_dir,
    make_private_dir,
    read_json,
    read_jsonl,
)

SETTINGS_NAME = "settings"
ROWS_FILE = "rows.jsonl"
EXCLUDED_FILE = "excluded.jsonl"
LABELS_FILE = "labels.jsonl"
SPLITS_FILE = "splits.json"
SUMMARY_FILE = "summary.json"
RAW_DIR = "raw"

SourceKind = Literal["import", "datadog"]


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """What a dataset directory is built from; frozen in ``settings.json`` on first use."""

    source: SourceKind
    split: SplitParams = field(default_factory=SplitParams)
    source_options: Mapping[str, Any] = field(default_factory=dict)
    """Source-specific settings, e.g. the query of a Datadog pull."""

    def to_json(self) -> dict[str, Any]:
        return {
            "source": {**self.source_options, "kind": self.source},
            "split": self.split.to_json(),
        }


@dataclass(frozen=True, slots=True)
class AddResult:
    """What one :meth:`DatasetDir.add` did, and the dataset summary afterwards."""

    added: int
    unchanged: int
    excluded: int
    summary: dict[str, Any]
    retired: int = 0
    """Stored rows taken out and recorded as excluded."""


class DatasetDir:
    """A dataset directory at ``path``. Mutations hold a :class:`DirectoryLock` on it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def settings_path(self) -> Path:
        return self.path / f"{SETTINGS_NAME}.json"

    @property
    def rows_path(self) -> Path:
        return self.path / ROWS_FILE

    @property
    def excluded_path(self) -> Path:
        return self.path / EXCLUDED_FILE

    @property
    def labels_path(self) -> Path:
        return self.path / LABELS_FILE

    @property
    def splits_path(self) -> Path:
        return self.path / SPLITS_FILE

    @property
    def summary_path(self) -> Path:
        return self.path / SUMMARY_FILE

    @property
    def raw_path(self) -> Path:
        return self.path / RAW_DIR

    def is_dataset(self) -> bool:
        return self.settings_path.is_file()

    # --- reading ------------------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """The frozen settings; raises :class:`UsageError` when this is not a dataset."""
        try:
            data = read_json(self.settings_path)
        except FileNotFoundError:
            raise UsageError(
                f"{self.path} is not a dataset directory (it has no {self.settings_path.name})",
                hint=(
                    "create one with `pup finetune dataset import DIR --file rows.jsonl` or "
                    "`pup finetune dataset pull DIR --ml-app APP`"
                ),
            ) from None
        if not isinstance(data, dict):
            raise StateError(f"{self.settings_path}: expected a JSON object")
        return data

    def split_params(self) -> SplitParams:
        split = self.settings().get("split")
        if not isinstance(split, dict):
            raise StateError(f"{self.settings_path}: missing split settings")
        return SplitParams.from_json(split, where=str(self.settings_path))

    def splits(self) -> SplitAssignments:
        return SplitAssignments.load(self.splits_path, self.split_params())

    def rows(self) -> list[TrainingRow]:
        """Every accepted row, in the order it was added."""
        if not self.rows_path.exists():
            return []
        try:
            return list(read_rows(self.rows_path))
        except RowError as exc:
            raise StateError(f"stored row is invalid: {exc}") from exc

    def kept_rows(self) -> list[TrainingRow]:
        """The accepted rows minus those labeled ``keep: false``."""
        return self._kept(self.rows())

    def _kept(self, rows: list[TrainingRow]) -> list[TrainingRow]:
        dropped = self._dropped_row_ids()
        return [row for row in rows if row.id not in dropped]

    def _dropped_row_ids(self) -> frozenset[str]:
        # The seam for keep/drop labels (``labels.jsonl``), which arrive with `dataset label`.
        # Until then no row is dropped, and a missing labels file means the same.
        return frozenset()

    def excluded(self) -> list[dict[str, Any]]:
        """Every exclusion record, in the order it was recorded."""
        if not self.excluded_path.exists():
            return []
        return list(read_jsonl(self.excluded_path))

    def rows_by_split(self) -> dict[SplitName, list[TrainingRow]]:
        """The kept rows grouped by split; every split is present, possibly empty."""
        return _group_by_split(self.kept_rows(), self.splits(), where=self.splits_path)

    def rows_for_split(self, split: SplitName) -> list[TrainingRow]:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split!r}")
        return self.rows_by_split()[split]

    def summary(self) -> dict[str, Any]:
        """Counts of accepted and excluded rows, per exclusion code and per split, and warnings."""
        rows = self.rows()
        by_split = _group_by_split(self._kept(rows), self.splits(), where=self.splits_path)
        excluded = _reconcile(self.excluded(), (), accepted_ids={row.id for row in rows})
        by_code = Counter(str(record.get("code")) for record in excluded)
        counts = {split: len(by_split[split]) for split in SPLITS}
        return {
            "total": len(rows) + len(excluded),
            "accepted": len(rows),
            "excluded": len(excluded),
            "excluded_by_code": dict(sorted(by_code.items())),
            "splits": counts,
            "warnings": split_warnings(counts),
        }

    # --- writing ------------------------------------------------------------------------

    def write_summary(self) -> dict[str, Any]:
        """Recompute and write ``summary.json``; returns the summary."""
        with DirectoryLock(self.path):
            return self._write_summary()

    def _write_summary(self) -> dict[str, Any]:
        summary = self.summary()
        atomic_write_json(self.summary_path, summary)
        return summary

    def export_split_files(self, out_dir: Path) -> dict[str, Path]:
        """Write ``train``, ``validation`` and ``test`` JSONL files in provider form.

        Every split gets a file, empty when the split is. Returns the path per split.
        """
        by_split = self.rows_by_split()
        check_isolation(by_split)
        out = ensure_private_dir(Path(out_dir))
        paths: dict[str, Path] = {}
        for split in SPLITS:
            path = paths[split] = out / f"{split}.jsonl"
            lines = [canonical_json(to_provider_dict(row)) + "\n" for row in by_split[split]]
            atomic_write_text(path, "".join(lines))
        return paths

    def add(
        self,
        rows: Iterable[TrainingRow],
        exclusions: Iterable[Exclusion] = (),
        *,
        settings: DatasetSettings,
        retired: Iterable[Exclusion] = (),
    ) -> AddResult:
        """Add valid ``rows`` and record ``exclusions``, creating the directory if needed.

        A row whose id is already present is skipped when identical and raises
        :class:`StateError` when it differs. The same holds within ``rows``: an identical
        repeat of an earlier row is the same row and is dropped, and the same id with
        different content is an error. A new row repeating an existing conversation under
        another id is excluded as ``duplicate_conversation``. ``settings`` must match the
        frozen settings.

        An exclusion replaces any earlier one for the same ``row_id``, and exclusions of ids
        that are now accepted rows are dropped, so a fixed row re-imported under its id is
        counted once, as accepted.

        ``retired`` takes stored rows out: each is an exclusion whose ``row_id`` names an
        accepted row that no longer belongs, such as a conversation a later pull found inside
        a longer one. The row is removed from ``rows.jsonl`` and the exclusion recorded. A
        ``row_id`` that is not stored is recorded as an exclusion all the same, and one that
        ``rows`` also holds raises :class:`ValueError`.

        A validation error (a conflicting id, changed settings) leaves existing dataset files
        unchanged. If writing fails part-way, re-running the same add repairs the directory.
        """
        self._prepare()
        with DirectoryLock(self.path):
            fresh = not self.is_dataset()
            if not fresh:
                self._check_split_params(settings.split)
                FrozenSettings.check_or_write(self.path, SETTINGS_NAME, settings.to_json())
            retiring = list(retired)
            retired_ids = {e.row_id for e in retiring if e.row_id is not None}
            stored = self.rows()
            existing = [row for row in stored if row.id not in retired_ids]
            incoming = list(rows)
            if clash := retired_ids.intersection(row.id for row in incoming):
                raise ValueError(f"rows both added and retired: {sorted(clash)}")
            excluded = [*exclusions, *retiring]
            new_rows, unchanged = _merge(existing, incoming, excluded)
            all_rows = [*existing, *new_rows]
            splits = SplitAssignments.load(self.splits_path, settings.split)
            for row in all_rows:
                splits.assign(split_key(row))
            check_isolation(_group_by_split(all_rows, splits, where=self.splits_path))
            records = _reconcile(
                self.excluded(), excluded, accepted_ids={row.id for row in all_rows}
            )
            if fresh:
                make_private_dir(self.path)
                FrozenSettings.check_or_write(self.path, SETTINGS_NAME, settings.to_json())
            # Splits first: an assignment without its row is harmless if we stop part-way.
            # Rows before exclusions: a stale exclusion of an accepted row is skipped on read.
            splits.save(self.splits_path)
            write_rows(self.rows_path, all_rows)
            lines = "".join(canonical_json(record) + "\n" for record in records)
            atomic_write_text(self.excluded_path, lines)
            summary = self._write_summary()
        retired_rows = len(stored) - len(existing)
        return AddResult(len(new_rows), unchanged, len(excluded), summary, retired_rows)

    def create_or_check(self, settings: DatasetSettings) -> dict[str, Any]:
        """Create an empty dataset frozen with ``settings``, or check an existing one has them.

        Raises as :meth:`add` does: :class:`StateError` when the frozen settings differ, and
        :class:`UsageError` for a non-empty directory that is not a dataset. Returns the
        summary.
        """
        return self.add((), settings=settings).summary

    def _check_split_params(self, params: SplitParams) -> None:
        stored = self.split_params()
        if stored != params:
            raise StateError(
                f"split settings changed; {self.settings_path} records different ones",
                hint=(
                    f"this dataset uses {_split_flags(stored)}: pass the same values, "
                    "or import into a new directory"
                ),
            )

    def _prepare(self) -> None:
        """Create the directory, refusing a non-empty one that is not already a dataset."""
        if self.path.exists() and not self.is_dataset():
            if not self.path.is_dir():
                raise UsageError(f"{self.path} exists and is not a directory")
            if any(child.name != LOCK_FILE_NAME for child in self.path.iterdir()):
                raise UsageError(
                    f"{self.path} is not empty and is not a dataset directory",
                    hint="choose a new or empty directory",
                )
        ensure_private_dir(self.path)


def read_import_file(path: Path) -> tuple[list[TrainingRow], list[Exclusion]]:
    """The valid rows of a canonical JSONL file, and an exclusion for each invalid line."""
    path = Path(path)
    rows: list[TrainingRow] = []
    exclusions: list[Exclusion] = []
    for line_number, data, result, line in scan_rows(path):
        if isinstance(result, RowError):
            where = f"{path.name}:{line_number}"
            exclusions.append(Exclusion.for_invalid(data, result, where=where, line=line))
        else:
            rows.append(result)
    return rows, exclusions


def _merge(
    existing: list[TrainingRow], incoming: Iterable[TrainingRow], excluded: list[Exclusion]
) -> tuple[list[TrainingRow], int]:
    """New rows from ``incoming``, and how many were already stored; duplicates go to ``excluded``."""
    stored = {row.id: row for row in existing}
    by_hash = {row_content_hash(row): row.id for row in existing}
    batch: dict[str, TrainingRow] = {}
    new_rows: list[TrainingRow] = []
    unchanged = 0
    for row in incoming:
        known = stored.get(row.id)
        if known is not None:
            if known != row:
                raise StateError(
                    f"row id {row.id!r} is already in the dataset with different content",
                    hint="give the changed row a new id, or import into a new directory",
                )
            unchanged += 1
            continue
        earlier = batch.get(row.id)
        if earlier is not None:
            if earlier != row:
                raise StateError(
                    f"row id {row.id!r} appears more than once in the input with different content",
                    hint="give each row in the input file a unique id",
                )
            continue  # The same row listed twice: it is one row.
        digest = row_content_hash(row)
        other = by_hash.get(digest)
        if other is not None:
            # Claim the id even so: a later row reusing it with other content is the same id
            # conflict whichever of the two comes first.
            batch[row.id] = row
            detail = f"same conversation as row {other!r}"
            excluded.append(Exclusion.for_row(row, ExclusionCode.DUPLICATE_CONVERSATION, detail))
            continue
        batch[row.id] = row
        by_hash[digest] = row.id
        new_rows.append(row)
    return new_rows, unchanged


def _reconcile(
    records: Iterable[dict[str, Any]],
    exclusions: Iterable[Exclusion],
    *,
    accepted_ids: Collection[str],
) -> list[dict[str, Any]]:
    """Stored exclusion ``records`` updated with new ``exclusions``, in first-recorded order.

    A record with a ``row_id`` is identified by it: a newer exclusion replaces it in place,
    and it is dropped once that id is an accepted row. A record without one is identified by
    all of its fields, including its location and the hash of its input line, so only an
    exact repeat of it (the same line re-imported from the same place) is dropped.
    """
    merged: dict[str, dict[str, Any]] = {}
    for record in [*records, *(exclusion.to_json() for exclusion in exclusions)]:
        row_id = record.get("row_id")
        if isinstance(row_id, str):
            if row_id not in accepted_ids:
                merged[f"id:{row_id}"] = record
        else:
            merged.setdefault(f"record:{canonical_json(record)}", record)
    return list(merged.values())


def _split_flags(params: SplitParams) -> str:
    return (
        f"--seed {params.seed} --validation-fraction {params.validation_fraction} "
        f"--test-fraction {params.test_fraction}"
    )


def _group_by_split(
    rows: Iterable[TrainingRow], splits: SplitAssignments, *, where: Path
) -> dict[SplitName, list[TrainingRow]]:
    grouped: dict[SplitName, list[TrainingRow]] = {split: [] for split in SPLITS}
    for row in rows:
        split = splits.get(split_key(row))
        if split is None:
            raise StateError(
                f"{where}: row {row.id!r} has no split assignment",
                hint="re-run the import to repair the dataset directory",
            )
        grouped[split].append(row)
    return grouped
