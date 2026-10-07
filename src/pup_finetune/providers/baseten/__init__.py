# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Baseten: LoRA fine-tuning on Baseten Loops. Only :mod:`.client` imports the Loops SDK."""

from typing import Final

__all__ = ["PROVIDER_NAME"]

PROVIDER_NAME: Final = "baseten"
