# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""The Fireworks SDK's expected serverless notices are dropped; its other warnings are not."""

from __future__ import annotations

import logging

import pytest

from pup_finetune.providers.fireworks import client


@pytest.mark.parametrize(
    "name", ["fireworks.training.sdk.client", "fireworks.training.sdk.concurrency"]
)
def test_only_the_expected_notices_are_dropped(name: str, caplog: pytest.LogCaptureFixture) -> None:
    client._quiet_expected_sdk_notices()
    client._quiet_expected_sdk_notices()  # installing twice adds one filter
    logger = logging.getLogger(name)
    assert sum(isinstance(f, client._ExpectedSdkNotices) for f in logger.filters) == 1
    with caplog.at_level(logging.WARNING, logger=name):
        logger.warning(
            "base_model is ignored for serverless base-only sampling; the rollout host ..."
        )
        logger.warning("prefill_queue_duration is unavailable; AdaptiveConcurrencyController ...")
        logger.warning("the training session expired")
    assert [record.getMessage() for record in caplog.records] == ["the training session expired"]
