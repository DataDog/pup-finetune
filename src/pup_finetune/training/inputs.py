# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""What a run trains on: a dataset's kept train and validation rows, rendered for its model.

:func:`build_inputs` renders a dataset for a preview or a new run; :func:`load_run_inputs`
renders it again for an existing run (a resume, or a detached worker), after checking that
the dataset and the renderer are still the ones the run was created with: the order of each
epoch depends on the rows, so other rows would resume at the wrong place.

Both read the dataset's rows once, and compute :attr:`RunInputs.dataset_sha256` (the run's
dataset identity) from that same snapshot, so a concurrent import cannot slip other rows in
between. Each split is rendered in row-id order, the order that hash covers, so the dataset
identity also fixes the training order: reordering ``rows.jsonl`` changes neither.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.schema import TrainingRow
from pup_finetune.data.split import SPLITS, SplitName
from pup_finetune.errors import StateError, UsageError
from pup_finetune.providers import get_provider
from pup_finetune.providers.base import ModelProfile, Provider
from pup_finetune.rendering.base import Renderer
from pup_finetune.runs import RunDir, in_id_order, rows_content_hash
from pup_finetune.runtime import Runtime
from pup_finetune.training.datums import (
    BuildResult,
    RejectedRow,
    RowExamples,
    build_examples,
    provider_row,
)

__all__ = ["RunInputs", "build_inputs", "load_run_inputs"]

NEW_RUN_HINT = "start a new run with `pup finetune train` to train on the current data"


@dataclass(frozen=True, slots=True)
class RunInputs:
    """The rendered train and validation rows of a dataset, and the kept rows per split.

    ``dataset_sha256`` is the content hash of the rows these were rendered from (see
    :func:`~pup_finetune.runs.rows_content_hash`).
    """

    rows: dict[SplitName, int]
    train: BuildResult
    validation: BuildResult
    dataset_sha256: str

    @property
    def train_rows(self) -> list[RowExamples]:
        return self.train.examples

    @property
    def validation_rows(self) -> list[RowExamples]:
        return self.validation.examples

    @property
    def tokens_per_epoch(self) -> int:
        """The rendered tokens of the accepted training rows: what one epoch trains on."""
        return self.train.train_tokens

    @property
    def rejected(self) -> list[RejectedRow]:
        """The train and validation rows left out, with why."""
        return [*self.train.rejected, *self.validation.rejected]

    def rejected_by_code(self) -> dict[str, int]:
        return dict(sorted(Counter(row.code for row in self.rejected).items()))

    def examples(self) -> dict[str, int]:
        """The examples rendered per split (more than the rows when rendered per turn)."""
        return {
            "train": sum(len(row.examples) for row in self.train_rows),
            "validation": sum(len(row.examples) for row in self.validation_rows),
        }


def build_inputs(dataset: DatasetDir, renderer: Renderer, profile: ModelProfile) -> RunInputs:
    """Render ``dataset``'s kept train and validation rows with ``renderer`` for ``profile``.

    Raises :class:`UsageError` when no training row is left to train on.
    """
    return _render(dataset.path, dataset.rows_by_split(), renderer, profile)


def _render(
    path: Path,
    by_split: dict[SplitName, list[TrainingRow]],
    renderer: Renderer,
    profile: ModelProfile,
) -> RunInputs:
    """``by_split`` (the dataset in ``path``) rendered, each split in row-id order."""

    def build(split: SplitName) -> BuildResult:
        rows = [provider_row(row) for row in in_id_order(by_split[split])]
        return build_examples(rows, renderer, max_seq_len=profile.max_seq_len)

    inputs = RunInputs(
        rows={split: len(by_split[split]) for split in SPLITS},
        train=build("train"),
        validation=build("validation"),
        dataset_sha256=rows_content_hash(by_split),
    )
    if not inputs.train_rows:
        rejected = Counter(row.code for row in inputs.train.rejected)
        why = "".join(f"; {count} rejected as {code}" for code, count in sorted(rejected.items()))
        raise UsageError(
            f"no rows to train on: the train split of {path} has "
            f"{inputs.rows['train']} kept rows{why}",
            hint="import more rows, or train a model with a longer context",
        )
    return inputs


def load_run_inputs(
    run_dir: RunDir, runtime: Runtime, *, finalize_only: bool = False
) -> tuple[Provider, RunInputs]:
    """The provider of the run in ``run_dir`` and its rows rendered again, as created.

    Resolves the provider (its credentials, too) from scratch and applies the run's provider
    options. Raises :class:`StateError` when the dataset's kept rows or the renderer's
    identity changed since the run was created. With ``finalize_only`` (only the promotion
    is left to retry), the error says so: the rows still set the promotion session's
    sequence length, so they must be the run's own.
    """
    manifest = run_dir.manifest()
    provider = get_provider(str(manifest["provider"]), runtime)
    provider.validate_options(run_dir.spec().provider_options)
    profile = run_dir.profile()
    dataset = DatasetDir(Path(str(manifest["dataset_dir"])))
    hint = NEW_RUN_HINT
    if finalize_only:
        hint = (
            f"only the promotion is left to retry: restore the dataset in {dataset.path} and "
            f"the {profile.alias} renderer as they were when the run was created, or {hint}"
        )
    by_split = dataset.rows_by_split()  # read once: what is checked is what is rendered
    if rows_content_hash(by_split) != manifest.get("dataset_sha256"):
        raise StateError(
            f"the dataset in {dataset.path} changed since the run in {run_dir.path} was "
            "created, so it cannot continue on the same rows",
            hint=hint,
        )
    renderer = provider.renderer(profile)
    if renderer.identity() != manifest.get("rendering"):
        raise StateError(
            f"the {profile.alias} renderer changed since the run in {run_dir.path} was created",
            hint=hint,
        )
    return provider, _render(dataset.path, by_split, renderer, profile)
