# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Baseten :class:`~pup_finetune.providers.Provider`. Heavy modules load inside its methods."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pup_finetune import credentials
from pup_finetune.providers.baseten.models import DEFAULT_MODEL, MODELS

if TYPE_CHECKING:
    from pydantic import SecretStr

    from pup_finetune.providers import DeployRequest, Provider, TrainRequest
    from pup_finetune.providers.baseten.client import BasetenHTTP


def _http() -> tuple[SecretStr, BasetenHTTP]:
    from pup_finetune.providers.baseten.client import BasetenHTTP

    api_key = credentials.resolve("baseten").value
    return api_key, BasetenHTTP(api_key)


class BasetenProvider:
    """LoRA SFT on Baseten Loops, and serving of the run's best checkpoint."""

    @property
    def name(self) -> str:
        return "baseten"

    def model_names(self) -> list[str]:
        return list(MODELS)

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def plan_training(self, request: TrainRequest) -> dict[str, Any]:
        from pup_finetune.providers.baseten import training

        plan, _ = training.plan_training(request)
        return plan

    def train(self, request: TrainRequest) -> dict[str, Any]:
        from pup_finetune.providers.baseten import training

        api_key, http = _http()
        with http:
            return training.train(request, api_key=api_key, http=http)

    def deploy_preview(self, request: DeployRequest) -> dict[str, Any]:
        from pup_finetune.providers.baseten import deploy as serving

        return serving.deploy_preview(request)

    def deploy(self, request: DeployRequest) -> dict[str, Any]:
        from pup_finetune.providers.baseten import deploy as serving

        api_key, http = _http()
        with http:
            return serving.deploy(request, api_key=api_key, http=http)

    def undeploy(self, run_dir: Path) -> dict[str, Any]:
        from pup_finetune.providers.baseten import deploy as serving

        _, http = _http()
        with http:
            return serving.undeploy(run_dir, http=http)


if TYPE_CHECKING:
    _conforms: Provider = BasetenProvider()
