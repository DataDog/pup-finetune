# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Training providers: the :class:`Provider` protocol, its requests and the lazy registry.

Each provider is the package ``pup_finetune.providers.<name>`` exposing ``PROVIDER``; it is
imported only when :func:`get_provider` asks for it. Importing this package registers every
provider's cheap parts (credential spec, ``doctor`` checks) and nothing heavier.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pup_finetune.errors import UsageError
from pup_finetune.providers.baseten import registration  # noqa: F401 - registers key spec, checks
from pup_finetune.training.settings import SFTSettings

__all__ = [
    "DeployRequest",
    "Provider",
    "TrainRequest",
    "get_provider",
    "provider_names",
]

_PROVIDERS: tuple[str, ...] = ("baseten",)


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """What ``pup finetune train`` asks a provider to plan or run."""

    train_path: Path
    validation_path: Path
    model: str
    run_dir: Path
    run_id: str
    settings: SFTSettings


@dataclass(frozen=True, slots=True)
class DeployRequest:
    """What ``pup finetune deploy`` asks a provider to preview or run."""

    run_dir: Path
    serving_mode: Literal["dedicated", "sampler"]
    accelerator: str | None
    max_seq_len: int | None
    timeout_seconds: float


class Provider(Protocol):
    """One training and serving backend. Every method returns the command's JSON result."""

    @property
    def name(self) -> str: ...

    def model_names(self) -> list[str]: ...

    def default_model(self) -> str: ...

    def plan_training(self, request: TrainRequest) -> dict[str, Any]:
        """Free: read and render the rows; no paid or remote-write calls."""
        ...

    def train(self, request: TrainRequest) -> dict[str, Any]:
        """Paid: run the training job."""
        ...

    def deploy_preview(self, request: DeployRequest) -> dict[str, Any]:
        """Free and local: what :meth:`deploy` would create."""
        ...

    def deploy(self, request: DeployRequest) -> dict[str, Any]:
        """Paid: serve the run's best checkpoint."""
        ...

    def undeploy(self, run_dir: Path) -> dict[str, Any]:
        """Stop the deployment recorded in ``run_dir``."""
        ...


def provider_names() -> list[str]:
    """The names :func:`get_provider` accepts."""
    return list(_PROVIDERS)


def get_provider(name: str) -> Provider:
    """The provider called ``name``, importing its package on first use."""
    if name not in _PROVIDERS:
        raise UsageError(
            f"unknown provider {name!r}",
            hint=f"available providers: {', '.join(provider_names())}",
        )
    provider: Provider = importlib.import_module(f"{__name__}.{name}").PROVIDER
    return provider
