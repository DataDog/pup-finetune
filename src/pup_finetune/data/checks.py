# SPDX-License-Identifier: Apache-2.0
#
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2026-present Datadog, Inc.
"""Span-level checks: reasons to leave out an LLM span that its conversation cannot show.

The canonical schema (:mod:`pup_finetune.data.schema`) and conversion
(:mod:`pup_finetune.data.convert`) judge a span by the conversation it records. The checks
here judge the span itself, before it is converted, and exclude it whole with a stable
:class:`~pup_finetune.data.exclusions.ExclusionCode`. They run first, so a span that fails
one never becomes a row and never absorbs a shorter conversation of its trace.

The checks are pure functions of the span, applied in order; the first that fails wins:

- ``span_error`` - the span's ``status`` is ``"error"``: the call failed, so what it recorded
  is not a completed model turn. ``CheckOptions(include_errors=True)`` skips it.

No ``truncated_content`` check: the Export API (with ``include_attachments: true``) gives no
explicit signal that it shortened a span's content, and guessing from the text (a trailing
ellipsis, a length) would exclude real conversations. A check belongs here once the API
reports truncation explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pup_finetune.data.exclusions import Exclusion, ExclusionCode
from pup_finetune.datadog.spans import LLMObsSpan

ERROR_STATUS = "error"

SpanCheck = Callable[[LLMObsSpan], tuple[ExclusionCode, str] | None]
"""A check: the code and detail of why ``span`` is excluded, or ``None`` when it passes."""


def span_error(span: LLMObsSpan) -> tuple[ExclusionCode, str] | None:
    """Fails an errored LLM call (``status: "error"``)."""
    if span.status == ERROR_STATUS:
        return ExclusionCode.SPAN_ERROR, "the span's status is 'error'"
    return None


@dataclass(frozen=True, slots=True)
class CheckOptions:
    """Which checks run. ``include_errors`` keeps errored LLM calls (no ``span_error``)."""

    include_errors: bool = False

    def checks(self) -> tuple[SpanCheck, ...]:
        """The checks these options enable, in the order they are applied."""
        checks: list[SpanCheck] = []
        if not self.include_errors:
            checks.append(span_error)
        return tuple(checks)


def check_span(span: LLMObsSpan, options: CheckOptions | None = None) -> Exclusion | None:
    """The exclusion for the first check ``span`` fails, or ``None`` when it passes them all."""
    return run_checks(span, (options or CheckOptions()).checks())


def run_checks(span: LLMObsSpan, checks: Iterable[SpanCheck]) -> Exclusion | None:
    """The exclusion for the first of ``checks`` that ``span`` fails, in order, or ``None``.

    The exclusion carries the row id the span would have had, ``{trace_id}:{span_id}``.
    """
    for check in checks:
        failure = check(span)
        if failure is not None:
            code, detail = failure
            return Exclusion.for_source(
                code,
                detail,
                row_id=f"{span.trace_id}:{span.span_id}",
                trace_id=span.trace_id,
                span_id=span.span_id,
            )
    return None
