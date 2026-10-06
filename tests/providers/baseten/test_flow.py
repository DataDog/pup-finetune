# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Train, deploy and undeploy one run end to end, through the files each step leaves behind."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from pup_finetune import redaction
from pup_finetune.data import read_rows, write_rows
from pup_finetune.providers import DeployRequest, TrainRequest
from pup_finetune.providers.baseten import training
from pup_finetune.providers.baseten.client import BasetenHTTP
from pup_finetune.providers.baseten.deploy import deploy, deploy_preview, undeploy
from pup_finetune.rendering import ModelFamily
from pup_finetune.training import SFTSettings
from tests.fakes.baseten import FakeLoops, baseten_transport
from tests.fakes.baseten_serving import (
    BASE_MODEL,
    CHECKPOINT_PATH,
    DEPLOYMENT,
    FakeClock,
    deployment,
    sampler_routes,
)
from tests.fakes.tokenizer import FakeChatTokenizer

SECRET = "zq9Xsynthetic0000basetenKEYw7Pm"  # noqa: S105 - synthetic test value
KEY = SecretStr(SECRET)
ROWS = list(read_rows(Path(__file__).parents[2] / "fixtures" / "rows" / "tool_use.jsonl"))
VALIDATION_IDS = {"row-units-001", "row-plain-001"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(training, "_sleep", lambda seconds: None)
    redaction.clear_secrets()
    yield
    redaction.clear_secrets()


def fake_tokenizer(family: ModelFamily) -> FakeChatTokenizer:
    return FakeChatTokenizer()


def test_a_trained_run_deploys_and_undeploys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_rows(tmp_path / "train.jsonl", [r for r in ROWS if r.id not in VALIDATION_IDS])
    write_rows(tmp_path / "validation.jsonl", [r for r in ROWS if r.id in VALIDATION_IDS])
    run_dir = tmp_path / "run"
    request = TrainRequest(
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "validation.jsonl",
        model="qwen3p5-9b",
        run_dir=run_dir,
        run_id="sft-test",
        settings=SFTSettings(max_epochs=2, batch_size=8),
    )
    capabilities = {
        "supported_models": [
            {"model_name": BASE_MODEL, "max_context_length": 131_072, "enabled": True}
        ]
    }
    run = "/v1/loops/runs/run1"
    train_routes: dict[str, Any] = {
        "GET /v1/loops/capabilities": (200, capabilities),
        f"GET {run}": [
            (200, {"run": {"status": "ACTIVE"}}),
            (200, {"run": {"status": "INACTIVE"}}),
        ],
        f"POST {run}/deactivate": (200, {}),
    }
    with BasetenHTTP(KEY, transport=baseten_transport(train_routes)) as http:
        trained = training.train(
            request,
            api_key=KEY,
            http=http,
            loops=FakeLoops(validation_losses=[1.0, 0.5]),
            tokenizer_loader=fake_tokenizer,
        )
    assert trained["status"] == "completed"
    assert trained["best_sampler_weights_uri"] == CHECKPOINT_PATH
    assert trained["next_command"] == (
        f"pup finetune deploy --run-dir {run_dir} --serving-mode sampler"
    )

    deploy_request = DeployRequest(
        run_dir=run_dir,
        serving_mode="sampler",
        accelerator=None,
        max_seq_len=None,
        timeout_seconds=60.0,
    )
    assert deploy_preview(deploy_request)["checkpoint_path"] == CHECKPOINT_PATH
    clock = FakeClock()
    statuses = [deployment("ACTIVE"), deployment("ACTIVE"), deployment("INACTIVE")]
    transport = baseten_transport(sampler_routes() | {f"GET {DEPLOYMENT}": statuses})
    with BasetenHTTP(KEY, transport=transport) as http:
        served = deploy(
            deploy_request,
            api_key=KEY,
            http=http,
            loops=FakeLoops(),
            tokenizer_loader=fake_tokenizer,
            sleep=clock.sleep,
            clock=clock,
        )
        assert served["state"] == "ready"
        assert served["served_model"] == CHECKPOINT_PATH
        assert undeploy(run_dir, http=http, sleep=clock.sleep, clock=clock)["state"] == "inactive"
    assert transport.calls.count(f"POST {DEPLOYMENT}/deactivate") == 1

    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert all(SECRET not in path.read_text() for path in run_dir.iterdir() if path.is_file())
