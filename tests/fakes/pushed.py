# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""A dataset directory as `dataset push` leaves it, without pushing anything.

:func:`mark_pushed` writes the push receipt a complete push of every kept row would have
written, so `train` and `benchmark` on a real provider (which read only a dataset that is in
Datadog) run on a dataset made with `dataset import`.
"""

from __future__ import annotations

from pathlib import Path

from pup_finetune.data.dataset_dir import DatasetDir
from pup_finetune.data.push import PushState, push_path
from pup_finetune.storage import atomic_write_json, settings_hash

DATASET_NAME = "synthetic-sft"
PROJECT = "pup-finetune-dev"


def mark_pushed(directory: Path, *, name: str = DATASET_NAME) -> Path:
    """Record every kept row of the dataset in ``directory`` as pushed to dataset ``name``;
    returns ``directory``."""
    dataset = DatasetDir(directory)
    state = PushState(
        project=PROJECT,
        project_id="synthetic-project-id",
        dataset_name=name,
        settings_hash=settings_hash(dataset.settings()),
        push_id="synthetic-push-id",
        dataset_id="synthetic-dataset-id",
        dataset_version=1,
        record_ids={row.id: f"synthetic-record-{n}" for n, row in enumerate(dataset.kept_rows())},
        pushed_at="2026-01-01T00:00:00+00:00",
    )
    atomic_write_json(push_path(dataset), state.to_json())
    return directory
