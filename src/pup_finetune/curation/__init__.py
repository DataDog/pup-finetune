# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Curation: deciding which accepted rows of a dataset to train on.

:mod:`pup_finetune.curation.labels` records keep/drop labels from the coding agent or a
person, and :mod:`pup_finetune.curation.samples` picks a small, readable sample of rows to
review before labeling. The data layer never imports this package: curation builds on
:mod:`pup_finetune.data`, not the other way round.
"""

from __future__ import annotations

from pup_finetune.curation.labels import (
    Label,
    LabelReport,
    LabelSource,
    apply_labels,
    latest_labels,
    parse_labels,
    read_labels,
)

__all__ = [
    "Label",
    "LabelReport",
    "LabelSource",
    "apply_labels",
    "latest_labels",
    "parse_labels",
    "read_labels",
]
