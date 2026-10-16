# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The paired bootstrap behind an evaluation's confidence intervals."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Final

__all__ = ["BOOTSTRAP_RESAMPLES", "MIN_CLUSTERS_FOR_CI", "paired_bootstrap_ci"]

BOOTSTRAP_RESAMPLES: Final = 2000
MIN_CLUSTERS_FOR_CI: Final = 2
"""The fewest trajectories with a paired case for which a confidence interval is reported."""


def _quantile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile of sorted ``values``, interpolating linearly between ranks."""
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """A percentile bootstrap CI of the mean paired difference, resampling whole clusters.

    Each cluster holds one trajectory's per-case differences (tuned minus base, each ``-1``,
    ``0`` or ``1``). A resample draws as many clusters as there are, with replacement, and
    takes the mean over all their cases, so the cases of one trajectory move together.
    Seeded with ``seed``: the same input gives the same interval. ``None`` with fewer than
    :data:`MIN_CLUSTERS_FOR_CI` non-empty clusters.
    """
    kept = [list(cluster) for cluster in clusters if cluster]
    if len(kept) < MIN_CLUSTERS_FOR_CI or resamples < 1:
        return None
    sums = [sum(cluster) for cluster in kept]
    counts = [len(cluster) for cluster in kept]
    rng = random.Random(seed)  # noqa: S311 - a reproducible resample, not crypto
    size = len(kept)
    stats: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        stats.append(sum(sums[i] for i in picks) / sum(counts[i] for i in picks))
    stats.sort()
    tail = (1.0 - level) / 2.0
    return _quantile(stats, tail), _quantile(stats, 1.0 - tail)
