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
- ``labels.jsonl`` - keep/drop labels, appended by ``dataset label`` (see
  :mod:`pup_finetune.data.labels`); the last label of a row id wins. Optional: a row
  with no label is kept, and a label of an id that is no longer accepted is ignored.
- ``splits.json`` - split assignments (see :mod:`pup_finetune.data.split`).
- ``summary.json`` - counts and warnings, rewritten after every change.
- ``push.json`` - written by ``dataset push``: the Agent Observability dataset the kept rows
  were pushed to and the record id of each (see :mod:`pup_finetune.data.push`).
- ``raw/`` - for a Datadog pull, the raw Export API pages it read, so an interrupted pull
  resumes without fetching them again (see
  :func:`~pup_finetune.datadog.spans.iter_span_pages`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.data.jsonl import read_rows, scan_rows, to_provider_dict, write_rows
from pup_finetune.data.labels import Label, label_counts, latest_labels, read_labels
from pup_finetune.data.schema import (
    RowError,
    TrainingRow,
    row_content_hash,
    split_key,
    without_reasoning,
)
from pup_finetune.data.split import (
    GENERIC_MIN_TRACES,
    GENERIC_PERCENT,
    LARGE_GROUP_SHARE,
    SPLITS,
    SplitAssignments,
    SplitConflict,
    SplitGroups,
    SplitName,
    SplitParams,
    assign_splits,
    check_distinct,
    check_isolation,
    conflict_summary,
    skew_warnings,
    split_conflicts,
    split_groups,
    split_warnings,
)
from pup_finetune.display import counted, display_path
from pup_finetune.errors import DirectoryUsageError, NotADatasetDir, SettingsConflict, StateError
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
SPLIT_CONFLICTS_FILE = "split_conflicts.jsonl"
"""Drop labels for the kept rows outside their group's split (see
:func:`~pup_finetune.data.split.split_conflicts`), for ``dataset label --file``; written
with the summary while there are any, removed once there are none."""
RAW_DIR = "raw"

SourceKind = Literal["import", "datadog", "datadog_dataset"]

SOURCE_COMMANDS: Final[dict[str, str]] = {
    "import": "dataset import",
    "datadog": "dataset pull",
    "datadog_dataset": "dataset fetch",
}
"""The command that builds a dataset from each source."""


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
class RefusedDrop:
    """A drop on a row :meth:`DatasetDir.add` retired that a person's keep refused to carry."""

    row_id: str
    """The stored row that holds the dropped row's conversation now, and trains its turns."""
    keep: Label
    """The person's keep: on :attr:`row_id`, or on a row this add retired into it."""
    drop: Label


@dataclass(frozen=True, slots=True)
class AddResult:
    """What one :meth:`DatasetDir.add` did, and the dataset summary afterwards."""

    added: int
    unchanged: int
    excluded: int
    summary: dict[str, Any]
    retired: int = 0
    """Stored rows taken out and recorded as excluded."""
    updated: int = 0
    """Stored rows replaced by a version with more reasoning."""
    carried: dict[str, str] = field(default_factory=dict)
    """The rows labeled drop because they hold the conversation of a dropped row this add
    retired, each with that row's id."""
    not_carried: list[RefusedDrop] = field(default_factory=list)
    """The drops of rows this add retired that a person's keep refused: the dropped turns
    train."""


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
        """The frozen settings; raises :class:`NotADatasetDir` when this is not a dataset."""
        try:
            data = read_json(self.settings_path)
        except FileNotFoundError:
            raise NotADatasetDir(
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
        """Every accepted row, in the order it was added; an id on two lines raises
        :class:`StateError`."""
        if not self.rows_path.exists():
            return []
        try:
            rows = list(read_rows(self.rows_path))
        except RowError as exc:
            raise StateError(f"stored row is invalid: {exc}") from exc
        counts = Counter(row.id for row in rows)
        if repeated := next((row_id for row_id, count in counts.items() if count > 1), None):
            raise StateError(
                f"{self.rows_path}: row id {repeated!r} is on more than one line",
                hint="a row id names one row: remove the extra line, or import the edited rows "
                "(with new ids) into a new directory with `dataset import`",
            )
        return rows

    def labels(self) -> dict[str, Label]:
        """The label in force per row id: the last one recorded. Empty without labels."""
        return latest_labels(read_labels(self.labels_path))

    def kept_rows(self) -> list[TrainingRow]:
        """The accepted rows minus those labeled ``keep: false``."""
        return self._kept(self.rows(), self.labels())

    @staticmethod
    def _kept(rows: list[TrainingRow], labels: Mapping[str, Label]) -> list[TrainingRow]:
        return [row for row in rows if (label := labels.get(row.id)) is None or label.keep]

    def excluded(self) -> list[dict[str, Any]]:
        """Every exclusion record, in the order it was recorded."""
        if not self.excluded_path.exists():
            return []
        return list(read_jsonl(self.excluded_path))

    def rows_by_split(self) -> dict[SplitName, list[TrainingRow]]:
        """The kept rows grouped by split; every split is present, possibly empty. Two kept
        rows holding one conversation raise :class:`StateError`."""
        by_split = _group_by_split(
            self.kept_rows(), self.splits(), where=self.splits_path, repair=self._source_command()
        )
        check_distinct(by_split)
        return by_split

    def rows_for_split(self, split: SplitName) -> list[TrainingRow]:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split!r}")
        return self.rows_by_split()[split]

    def summary(self) -> dict[str, Any]:
        """Counts of accepted, excluded, kept and dropped rows, and warnings.

        Exclusions are counted per code, kept rows per split, and drops per reason (the most
        frequent reasons only). ``split_conflicts`` counts the groups of copies whose kept
        rows are in several splits and names some of the rows outside their group's split.
        """
        return self._summary()[0]

    def _summary(self) -> tuple[dict[str, Any], list[SplitConflict]]:
        """:meth:`summary`, and every split conflict it counts."""
        rows = self.rows()
        labels = self.labels()
        kept = self._kept(rows, labels)
        splits = self.splits()
        by_split = _group_by_split(
            kept, splits, where=self.splits_path, repair=self._source_command()
        )
        excluded = _reconcile(self.excluded(), (), accepted_ids={row.id for row in rows})
        by_code = Counter(str(record.get("code")) for record in excluded)
        counts = {split: len(by_split[split]) for split in SPLITS}
        groups = split_groups(rows)
        dropped = {row_id for row_id, label in labels.items() if not label.keep}
        conflicts = split_conflicts(groups, splits, dropped=dropped)
        excluded_ids = {r["row_id"] for r in excluded if isinstance(r.get("row_id"), str)}
        unknown = sorted(set(labels) - {row.id for row in rows} - excluded_ids)
        summary = {
            "total": len(rows) + len(excluded),
            "accepted": len(rows),
            "excluded": len(excluded),
            "excluded_by_code": dict(sorted(by_code.items())),
            "kept": len(kept),
            **label_counts(labels, (row.id for row in rows)),
            "splits": counts,
            "traces": len({split_key(row) for row in rows}),
            "split_groups": len(groups.groups),
            "split_conflicts": {
                **conflict_summary(conflicts),
                "labels_file": SPLIT_CONFLICTS_FILE if conflicts else None,
            },
            "warnings": [
                *split_warnings(counts),
                *skew_warnings(
                    {split: [split_key(row) for row in by_split[split]] for split in SPLITS},
                    splits.params,
                    more=self._more_rows(),
                ),
                *_grouping_warnings(groups, conflicts),
                *_unknown_label_warnings(unknown),
            ],
        }
        return summary, conflicts

    def _source_command(self) -> str:
        """The command that built the dataset, which repairs it when rerun."""
        source = self.settings().get("source")
        kind = source.get("kind") if isinstance(source, Mapping) else None
        return SOURCE_COMMANDS.get(str(kind), "dataset import")

    def _more_rows(self) -> str:
        """What a skew warning advises: more rows, from the command that built the dataset."""
        source = self.settings().get("source")
        kind = source.get("kind") if isinstance(source, Mapping) else None
        return "import more rows" if kind == "import" else "pull more rows"

    # --- writing ------------------------------------------------------------------------

    def write_summary(self) -> dict[str, Any]:
        """Recompute and write ``summary.json``; returns the summary."""
        with DirectoryLock(self.path):
            return self._write_summary()

    def append_labels(self, labels: Sequence[Label]) -> dict[str, Any]:
        """Append ``labels`` to ``labels.jsonl`` and rewrite ``summary.json``; returns it.

        The caller holds the directory lock and has checked the labels (see
        :func:`~pup_finetune.curation.labels.apply_labels`). The file is rewritten as its
        current content plus ``labels`` in one atomic replace; ``rows.jsonl`` is untouched.
        """
        self._append_labels(labels)
        return self._write_summary()

    def _append_labels(self, labels: Iterable[Label]) -> None:
        lines = "".join(canonical_json(label.to_json()) + "\n" for label in labels)
        if lines:
            path = self.labels_path
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            if text and not text.endswith("\n"):
                text += "\n"
            atomic_write_text(path, text + lines)

    def _write_summary(self) -> dict[str, Any]:
        summary, conflicts = self._summary()
        path = self.path / SPLIT_CONFLICTS_FILE
        if conflicts:
            labels = [
                {
                    "id": row_id,
                    "keep": False,
                    "reason": f"a copy of a request whose copies are mostly in the "
                    f"{conflict.split} split",
                }
                for conflict in conflicts
                for split in SPLITS
                for row_id in sorted(conflict.outside.get(split, ()))
            ]
            atomic_write_text(path, "".join(canonical_json(label) + "\n" for label in labels))
        else:
            path.unlink(missing_ok=True)
        atomic_write_json(self.summary_path, summary)
        return summary

    def export_split_files(self, out_dir: Path) -> dict[str, Path]:
        """Write ``train``, ``validation`` and ``test`` JSONL files in provider form.

        Every split gets a file, empty when the split is. Returns the path per split.
        """
        by_split = self.rows_by_split()
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
        updated: Iterable[TrainingRow] = (),
        split_overrides: Mapping[str, SplitName] | None = None,
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

        ``updated`` replaces stored rows in place, by id: each must name a stored row and
        differ from it only in ``reasoning_content``, such as reasoning a later pull read for
        a row it already stored (see :func:`~pup_finetune.data.pull.reasoning_updates`). Its
        split assignment is unchanged. An id that is not stored, that ``rows`` or ``retired``
        also holds, a change to anything but reasoning, or an update that leaves the row
        holding the same conversation as another stored row raises :class:`ValueError`.

        A drop label in force on a ``retired`` row carries to the stored row that holds its
        conversation now (the exclusion's ``into``, followed through the rows this add
        retires), as a new drop label whose ``carried_from`` names it, so the turns the
        reviewer dropped stay out of training: it says ``by: manual``, which triage never
        relabels. A keep does not carry, as that row has turns nobody reviewed, and a drop
        does not replace a person's label (``by: manual``) on that row, nor carry over a
        person's keep on a row this add retired into it on the way, which was a decision on
        a whole row holding the conversation; it replaces a judge's or a check's.

        ``split_overrides`` assigns split keys to the splits a source already gave them (the
        splits recorded on a fetched dataset's records) before new keys are assigned; a key
        assigned to another split already raises :class:`StateError`.

        A validation error (a conflicting id, changed settings) leaves existing dataset files
        unchanged. If writing fails part-way, re-running the same add repairs the directory;
        the exclusions of ``retired`` and the drops they carry are written before the rows
        are, so a retired row is never gone from ``rows.jsonl`` without its record in
        ``excluded.jsonl`` or the drop it carries.
        """
        self._prepare()
        with DirectoryLock(self.path):
            fresh = not self.is_dataset()
            if not fresh:
                self.check_source(settings.source)
                self._check_split_params(settings.split)
                FrozenSettings.check_or_write(self.path, SETTINGS_NAME, settings.to_json())
            retiring = list(retired)
            retired_ids = {e.row_id for e in retiring if e.row_id is not None}
            stored = self.rows()
            existing = [row for row in stored if row.id not in retired_ids]
            incoming = list(rows)
            if clash := retired_ids.intersection(row.id for row in incoming):
                raise ValueError(f"rows both added and retired: {sorted(clash)}")
            existing, changed = _apply_updates(
                existing, list(updated), retired_ids | {row.id for row in incoming}
            )
            excluded = [*exclusions, *retiring]
            new_rows, unchanged = _merge(existing, incoming, excluded)
            all_rows = [*existing, *new_rows]
            splits = SplitAssignments.load(self.splits_path, settings.split)
            for key, split in (split_overrides or {}).items():
                if splits.get(key) not in (None, split):
                    raise StateError(
                        f"{self.splits_path} assigns {key!r} to {splits.get(key)}, not {split}"
                    )
                splits.assignments[key] = split
            labels = self.labels()
            carried, refused = _carried_drops(retiring, labels, {row.id for row in all_rows})
            dropped = {row_id for row_id, label in labels.items() if not label.keep}
            assign_splits(all_rows, splits, dropped=dropped | {c.id for c in carried.values()})
            repair = SOURCE_COMMANDS[settings.source]
            check_isolation(
                _group_by_split(all_rows, splits, where=self.splits_path, repair=repair)
            )
            records = _reconcile(
                self.excluded(), excluded, accepted_ids={row.id for row in all_rows}
            )
            if fresh:
                make_private_dir(self.path)
                FrozenSettings.check_or_write(self.path, SETTINGS_NAME, settings.to_json())
            # Splits first: an assignment without its row is harmless if we stop part-way.
            # The retirements next, before the rows they take out: a stale exclusion of an
            # accepted row is skipped on read, but a retired row gone from rows.jsonl without
            # its record could not be recovered, as nothing left on disk would show it.
            # Then the rows, and the exclusions last; a re-run of the same add repairs both.
            splits.save(self.splits_path)
            if retiring:
                staged = _reconcile(
                    self.excluded(), retiring, accepted_ids={row.id for row in existing}
                )
                self._write_excluded(staged)
            self._append_labels(carried.values())
            write_rows(self.rows_path, all_rows)
            self._write_excluded(records)
            summary = self._write_summary()
        retired_rows = len(stored) - len(existing)
        return AddResult(
            len(new_rows),
            unchanged,
            len(excluded),
            summary,
            retired_rows,
            changed,
            {label.id: row_id for row_id, label in carried.items()},
            refused,
        )

    def _write_excluded(self, records: Iterable[dict[str, Any]]) -> None:
        lines = "".join(canonical_json(record) + "\n" for record in records)
        atomic_write_text(self.excluded_path, lines)

    def create_or_check(self, settings: DatasetSettings) -> dict[str, Any]:
        """Create an empty dataset frozen with ``settings``, or check an existing one has them.

        Raises as :meth:`add` does: :class:`SettingsConflict` when the frozen settings differ,
        and :class:`NotADatasetDir` for a non-empty directory that is not a dataset. Returns the
        summary.
        """
        return self.add((), settings=settings).summary

    def check_source(self, kind: SourceKind) -> None:
        """Raise :class:`SettingsConflict` naming the cause when the dataset was built by the
        other command (``dataset import`` against ``dataset pull``)."""
        source = self.settings().get("source")
        stored = source.get("kind") if isinstance(source, Mapping) else None
        if stored == kind or stored not in SOURCE_COMMANDS:
            return
        built, adding = SOURCE_COMMANDS[stored], SOURCE_COMMANDS[kind]
        raise SettingsConflict(
            f"{display_path(self.path)} was built by `{built}`; {adding.split()[-1]} into a new "
            "DIR",
            hint=f"a dataset directory holds rows from one source: run `{adding}` into a new "
            f"or empty directory, or add to this one with `{built}`",
        )

    def _check_split_params(self, params: SplitParams) -> None:
        stored = self.split_params()
        if stored != params:
            raise SettingsConflict(
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
                raise NotADatasetDir(f"{self.path} exists and is not a directory")
            if any(child.name != LOCK_FILE_NAME for child in self.path.iterdir()):
                raise NotADatasetDir(
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
                raise DirectoryUsageError(
                    f"row id {row.id!r} is already in the dataset with different content",
                    hint="give the changed row a new id, or import into a new directory",
                )
            unchanged += 1
            continue
        earlier = batch.get(row.id)
        if earlier is not None:
            if earlier != row:
                raise DirectoryUsageError(
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


def _carried_drops(
    retired: Iterable[Exclusion], labels: Mapping[str, Label], stored: Collection[str]
) -> tuple[dict[str, Label], list[RefusedDrop]]:
    """The drop label each ``retired`` row with a drop in force passes to the ``stored`` row
    that holds its conversation (its exclusion's ``into``), by the retired row's id, and each
    drop a person's keep refuses.

    That row can be retired too, as when a prefix's longer row repeats another stored row:
    the drop passes to the row its retirements end at, or to none when that row is not
    stored, since nothing trains it. None passes when that row has a ``manual`` drop: a
    person's, or a drop carried already, so a rerun after a crash, which finds the drop it
    carried in force, carries nothing twice. A person's keep on that row or on a row along
    the way refuses the drop, unless another drop dropped the row. A judge's or a check's
    label is replaced, even a drop, as triage would review it again and could keep the row.
    """
    targets = {e.row_id: e.into for e in retired if e.row_id is not None}
    current = dict(labels)
    carried: dict[str, Label] = {}
    refused: list[RefusedDrop] = []
    for row_id, into in targets.items():
        chain: list[str] = []
        while into is not None:
            chain.append(into)
            into = targets.get(into)
        if not chain or chain[-1] not in stored:
            continue
        label = current.get(row_id)
        if label is None or label.keep:
            continue
        held = chain[-1]
        own = current.get(held)
        if own is not None and own.by == "manual" and not own.keep:
            continue
        decided = (current.get(hop) for hop in reversed(chain))
        keep = next((k for k in decided if k is not None and k.by == "manual" and k.keep), None)
        if keep is not None:
            refused.append(RefusedDrop(held, keep, label))
            continue
        current[held] = carried[row_id] = Label(
            id=held,
            keep=False,
            reason=f"carried from row {row_id!r}, which this row absorbed: {label.reason}",
            carried_from=row_id,
        )
    dropped = {label.id for label in carried.values()}
    return carried, [r for r in refused if r.row_id not in dropped]


def _apply_updates(
    existing: list[TrainingRow], updated: list[TrainingRow], taken: Collection[str]
) -> tuple[list[TrainingRow], int]:
    """``existing`` with each of ``updated`` in place of the stored row of its id.

    Returns the rows and how many of them changed. Raises :class:`ValueError` for an update
    of an id that is not stored, that is ``taken`` (added or retired in the same add), that
    changes anything but reasoning, or that leaves the row with the content hash of another.
    """
    if not updated:
        return existing, 0
    replacements: dict[str, TrainingRow] = {}
    for row in updated:
        if row.id in taken:
            raise ValueError(f"row {row.id!r} both updated and added or retired")
        replacements[row.id] = row
    stored = {row.id for row in existing}
    if missing := sorted(set(replacements) - stored):
        raise ValueError(f"updated rows are not stored: {missing}")
    rows: list[TrainingRow] = []
    changed = 0
    for row in existing:
        replacement = replacements.get(row.id, row)
        if replacement != row:
            if without_reasoning(replacement) != without_reasoning(row):
                raise ValueError(f"row {row.id!r} updated in more than its reasoning")
            changed += 1
        rows.append(replacement)
    # Filling in reasoning can make a row equal to another one that already had it: the two
    # would then be one conversation stored twice, which the caller must retire one of.
    by_hash: dict[str, str] = {}
    for row in rows:
        other = by_hash.setdefault(row_content_hash(row), row.id)
        if other != row.id and (row.id in replacements or other in replacements):
            raise ValueError(
                f"rows {other!r} and {row.id!r} hold the same conversation after the update"
            )
    return rows, changed


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


def _grouping_warnings(groups: SplitGroups, conflicts: Sequence[SplitConflict]) -> list[str]:
    """Warnings about copies of a request that cannot share a split (ones recorded in
    several splits before they were grouped, and generic requests), and about a group of
    traces so large that it is likely chained through prompts they share."""
    warnings: list[str] = []
    if conflicts:
        rows = sum(len(ids) for conflict in conflicts for ids in conflict.outside.values())
        warnings.append(
            f"{counted(rows, 'kept row')} {'is a copy' if rows == 1 else 'are copies'} of a "
            "request whose copies are mostly in another split "
            f"({counted(len(conflicts), 'group')}; see split_conflicts), recorded apart before "
            "they were grouped, so they may be both trained and tested on; recorded splits "
            f"never move, so drop them with `dataset label --file` and {SPLIT_CONFLICTS_FILE}"
        )
    if groups.generic:
        shown = ", ".join(groups.generic[0][:3])
        common = len(groups.generic)
        warnings.append(
            f"{counted(common, 'request')} {'repeats' if common == 1 else 'repeat'} in more "
            f"than {groups.limit} traces "
            f"({GENERIC_PERCENT}% of the dataset's traces, and at least {GENERIC_MIN_TRACES}), "
            "too common to group by, so their copies keep a split per trace (e.g. rows "
            f"{shown}); a fixed prompt many conversations share is such a request"
        )
    if chained := groups.chained():
        total = sum(len(group) for group in groups.groups)
        shown = ", ".join(row.id for row in chained[:3])
        traces = len({split_key(row) for row in chained})
        warnings.append(
            f"one group of rows that share a split holds {len(chained)} of {total} rows "
            f"({len(chained) / total:.0%}, more than {LARGE_GROUP_SHARE:.0%}) in {traces} "
            "traces, so one split takes them all; its traces are likely chained through "
            f"requests they share (e.g. rows {shown})"
        )
    return warnings


def _unknown_label_warnings(unknown: Sequence[str]) -> list[str]:
    """A warning about labels of ids that name no accepted or excluded row, such as a typo
    in a hand-appended label."""
    if not unknown:
        return []
    one = len(unknown) == 1
    shown = ", ".join(repr(row_id) for row_id in unknown[:3])
    return [
        f"{counted(len(unknown), 'label')} in {LABELS_FILE} {'names' if one else 'name'} no row "
        f"of the dataset (e.g. {shown}), so {'it changes' if one else 'they change'} nothing; "
        "record labels with `dataset label --file`, which refuses such ids"
    ]


def _group_by_split(
    rows: Iterable[TrainingRow], splits: SplitAssignments, *, where: Path, repair: str
) -> dict[SplitName, list[TrainingRow]]:
    """``rows`` by split; ``repair`` is the command that assigns a row without one."""
    grouped: dict[SplitName, list[TrainingRow]] = {split: [] for split in SPLITS}
    for row in rows:
        split = splits.get(split_key(row))
        if split is None:
            raise StateError(
                f"{where}: row {row.id!r} has no split assignment",
                hint=f"remove it from {ROWS_FILE}, or rerun `{repair}` into this directory, "
                "which assigns it a split",
            )
        grouped[split].append(row)
    return grouped
